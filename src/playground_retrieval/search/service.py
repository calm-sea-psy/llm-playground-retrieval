"""검색 서비스: Hybrid(Dense + BM25) → Reranker 를 조립하고 한 번에 한 질의씩 돌린다."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from playground_corpus import EMBEDDING_MODELS, Chunk, SentenceTransformerEmbedder

from playground_retrieval.core import Filters, Hit, Retriever
from playground_retrieval.search import cross_encoder, hybrid, rerank
from playground_retrieval.search.analyzer import KiwiAnalyzer
from playground_retrieval.search.bm25 import Bm25Index, Bm25Retriever
from playground_retrieval.search.cross_encoder import CrossEncoderScorer
from playground_retrieval.search.dense import ChunkStore, DenseRetriever, VectorIndex
from playground_retrieval.search.hybrid import HybridRetriever
from playground_retrieval.search.rerank import Reranker

logger = logging.getLogger(__name__)

TOP_K = (
    5  # 요청에 k 가 없을 때 돌려주는 청크 수 (답변 생성에 넘길 만큼) — 제품이 설정으로 바꿔 넘긴다
)


class ChunkCatalog(ChunkStore, Protocol):
    """청크 원문 저장소 (PgStore). Dense 가 쓰는 조회에 BM25 색인용 전체 읽기를 더한 것."""

    def iter_chunks(self, collection: str, source: str) -> Iterator[Chunk]:
        """컬렉션의 청크 전부를 흘려 준다."""
        ...

    def doc_categories(self, collection: str, source: str) -> Mapping[str, frozenset[str]]:
        """컬렉션에 청크가 있는 문서마다 분류 집합."""
        ...


@dataclass(frozen=True)
class SearchSettings:
    """검색 서비스 구성. 기본값은 D12 에서 가장 높았던 Hybrid + Reranker 설정."""

    collection: str
    """검색할 Qdrant 컬렉션 · PG 청크 컬렉션 (scripts/ingest.py 로 적재)."""
    source: str
    """문서 제목을 붙일 PG documents 의 소스 이름."""
    model: str
    """질의 임베딩 모델. `EMBEDDING_MODELS` 의 키 — 컬렉션을 적재한 모델과 같아야 한다."""
    candidates: int = hybrid.CANDIDATES
    """Dense · BM25 가 각각 가져오는 후보 수."""
    rrf_k: int = hybrid.RRF_K
    rerank_candidates: int = rerank.CANDIDATES
    """리랭커에 넘기는 융합 후보 수."""
    reranker_model: str = cross_encoder.MODEL_ID
    top_k: int = TOP_K
    """요청에 k 가 없을 때 돌려주는 건수."""
    bm25_index: Path | None = None
    """scripts/build_bm25.py 로 저장한 BM25 색인 폴더. 없거나 비면 시작할 때 PG 청크로 만든다 (서브셋 규모)."""


@dataclass(frozen=True)
class SearchResult:
    """질의 하나의 검색 결과."""

    hits: tuple[Hit, ...]
    """리랭커 점수(0~1) 내림차순."""
    took_ms: float
    """검색 시간. 앞 질의를 기다린 시간은 빼고."""


class SearchService:
    """검색기를 들고 질의를 한 번에 하나씩 돌리는 서비스.

    GPU 모델(임베더 · 리랭커)과 PG 접속 하나를 같이 쓰므로 질의를 동시에 돌리지 않는다.

    예)
        service = SearchService.build(settings, pg, qdrant)
        result = await service.search("조선의 수도는?")
    """

    def __init__(
        self, retriever: Retriever, *, collection: str, chunk_count: int, top_k: int = TOP_K
    ) -> None:
        self._retriever = retriever
        self.collection = collection
        self.chunk_count = chunk_count
        self.top_k = top_k
        self._lock = asyncio.Lock()

    @classmethod
    def build(
        cls, settings: SearchSettings, pg: ChunkCatalog, qdrant: VectorIndex
    ) -> SearchService:
        """모델을 올리고 BM25 색인(저장한 것 또는 PG 청크로 새로)을 준비해 Hybrid + Reranker 를 조립한다.

        Raises:
          ValueError: 컬렉션에 청크가 없을 때 (적재 전이거나 이름이 틀렸을 때).
        """
        analyzer = KiwiAnalyzer()
        start = time.perf_counter()
        index = _bm25_index(settings, pg, analyzer)
        if not len(index):
            raise ValueError(f"collection has no chunks: {settings.collection!r}")
        logger.info("bm25 index: %d chunks in %.1fs", len(index), time.perf_counter() - start)
        bm25 = Bm25Retriever(
            index,
            analyzer,
            pg,
            collection=settings.collection,
            source=settings.source,
            doc_categories=pg.doc_categories(settings.collection, settings.source),
        )
        dense = DenseRetriever(
            SentenceTransformerEmbedder.load(EMBEDDING_MODELS[settings.model]),
            qdrant,
            pg,
            collection=settings.collection,
            source=settings.source,
        )
        fused = HybridRetriever([dense, bm25], rrf_k=settings.rrf_k, candidates=settings.candidates)
        reranker = Reranker(
            fused,
            CrossEncoderScorer.load(settings.reranker_model),
            candidates=settings.rerank_candidates,
        )
        return cls(
            reranker,
            collection=settings.collection,
            chunk_count=len(index),
            top_k=settings.top_k,
        )

    async def search(
        self, query: str, k: int | None = None, filters: Filters | None = None
    ) -> SearchResult:
        """질의와 관련 높은 청크 k건(없으면 top_k건)을 돌려준다. 검색은 작업 스레드에서 돈다."""
        async with self._lock:
            start = time.perf_counter()
            hits = await asyncio.to_thread(self._retriever.search, query, k or self.top_k, filters)
            took_ms = (time.perf_counter() - start) * 1000
        return SearchResult(hits=tuple(hits), took_ms=round(took_ms, 1))


def _bm25_index(settings: SearchSettings, pg: ChunkCatalog, analyzer: KiwiAnalyzer) -> Bm25Index:
    """저장한 색인이 있으면 읽고, 없으면 PG 청크로 만든다."""
    if settings.bm25_index is not None and (settings.bm25_index / "meta.json").exists():
        logger.info("bm25 index: loading %s", settings.bm25_index)
        return Bm25Index.load(settings.bm25_index)
    return Bm25Index.build(pg.iter_chunks(settings.collection, settings.source), analyzer)
