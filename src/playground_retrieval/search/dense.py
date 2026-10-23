"""벡터 검색기: 질의 임베딩 → Qdrant 에서 가까운 청크 id → PostgreSQL 에서 원문."""

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from playground_corpus import Chunk, Embedder

from playground_retrieval.core import Filters, Hit, VectorHit

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """DenseRetriever 가 부르는 벡터 검색 계약 (QdrantStore)."""

    def search(
        self,
        name: str,
        vector: NDArray[np.float32],
        k: int,
        *,
        doc_ids: Sequence[str] = (),
        categories: Sequence[str] = (),
    ) -> list[VectorHit]:
        """벡터와 가까운 청크 k건을 유사도 내림차순으로 돌려준다."""
        ...


class ChunkStore(Protocol):
    """DenseRetriever 가 부르는 원문 조회 계약 (PgStore)."""

    def get_chunks(self, collection: str, source: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        """청크 id 순서대로 청크를 돌려준다. 없는 id 는 빠진다."""
        ...


class DenseRetriever:
    """임베딩 모델 · 벡터 색인 · 원문 저장소를 묶어 Retriever 계약을 채운다.

    예)
        dense = DenseRetriever(embedder, qdrant, pg, collection="kowiki_bge-m3", source="kowiki")
        hits = dense.search("조선의 수도는?", k=50)
    """

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        store: ChunkStore,
        *,
        collection: str,
        source: str,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._store = store
        self.collection = collection
        self.source = source

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """질의와 코사인 유사도가 높은 청크 k건을 점수 내림차순으로 돌려준다."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k=}")
        vector = self._embedder.encode_queries([query])[0]
        vector_hits = self._index.search(
            self.collection,
            vector,
            k,
            doc_ids=sorted(filters.doc_ids) if filters else (),
            categories=sorted(filters.categories) if filters else (),
        )
        chunks = {
            chunk.id: chunk
            for chunk in self._store.get_chunks(
                self.collection, self.source, [hit.chunk_id for hit in vector_hits]
            )
        }
        missing = [hit.chunk_id for hit in vector_hits if hit.chunk_id not in chunks]
        if missing:  # Qdrant 와 PG 가 어긋난 경우 — 다시 인제스트해야 한다
            logger.warning(
                "%d chunks in %s are missing in PG: %s", len(missing), self.collection, missing[:5]
            )
        return [
            Hit(chunk=chunks[hit.chunk_id], score=hit.score)
            for hit in vector_hits
            if hit.chunk_id in chunks
        ]
