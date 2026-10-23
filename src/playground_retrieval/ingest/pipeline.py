"""인제스트 흐름: 문서 → PG documents → 청킹 → PG chunks → 임베딩(npy 캐시) → Qdrant → 실행 기록.

같은 키로 다시 넣으면 덮어쓰고, 임베딩은 캐시 샤드를 다시 계산하지 않으므로 중단된 뒤 다시 돌려도 된다.
문서를 어디서 읽을지 · 모델과 청크 설정을 무엇으로 할지는 쓰는 제품이 정해 넘긴다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from playground_corpus import Chunker, Document, Embedder, EmbeddingCache

from playground_retrieval.store import PgStore, QdrantStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """인제스트 한 번의 결과."""

    run_id: int
    """PG ingest_runs 행 id."""
    documents: int
    chunks: int
    points: int
    """적재 뒤 Qdrant 컬렉션의 점 개수 (이전에 넣은 것 포함)."""
    chunk_sec: float
    embed_sec: float
    """임베딩 시간. 캐시에 있던 샤드는 다시 계산하지 않아 짧아진다."""


def ingest_documents(
    documents: Sequence[Document],
    *,
    source: str,
    collection: str,
    chunker: Chunker,
    embedder: Embedder,
    cache: EmbeddingCache,
    pg: PgStore,
    qdrant: QdrantStore,
    dump_date: str | None = None,
    on_disk: bool = True,
) -> IngestResult:
    """문서를 청킹 · 임베딩해 PG(원문) · Qdrant(벡터)에 넣고 실행을 기록한다.

    Args:
      documents: 넣을 문서. 분류(categories)는 Qdrant payload 필터로 들어간다.
      source: PG documents 의 소스 이름 (예: `kowiki`). 청크 원문을 꺼낼 때 제목을 붙이는 기준.
      collection: PG chunks · Qdrant 컬렉션 이름 (보통 `{소스}_{모델}`).
      chunker: 청크 설정. 토크나이저는 embedder 모델의 것이어야 한다.
      embedder: 문서 임베딩 모델. spec 의 이름 · 차원을 실행 기록과 컬렉션에 쓴다.
      cache: 임베딩 npy 캐시 폴더.
      pg: 원문 저장소 (스키마가 없으면 만든다).
      qdrant: 벡터 저장소 (컬렉션이 없으면 만든다).
      dump_date: 원본 날짜. Qdrant payload 에 그대로.
      on_disk: Qdrant 원본 벡터를 디스크에 둘지 (int8 양자화 벡터는 늘 메모리).
    """
    spec = embedder.spec
    pg.create_schema()
    run_id = pg.start_run(source, collection, spec.name, chunker.config.max_tokens)
    pg.upsert_documents(source, documents)

    start = time.perf_counter()
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    chunk_sec = time.perf_counter() - start
    logger.info(
        "ingest %s: %d documents → %d chunks (%.0fs)",
        collection,
        len(documents),
        len(chunks),
        chunk_sec,
    )
    pg.upsert_chunks(collection, chunks)

    start = time.perf_counter()
    cache.write(embedder, [chunk.id for chunk in chunks], [chunk.text for chunk in chunks])
    _, vectors = cache.load()
    embed_sec = time.perf_counter() - start
    logger.info("ingest %s: embedded %d (%.0fs)", collection, len(vectors), embed_sec)

    qdrant.ensure_collection(collection, dim=spec.dim, on_disk=on_disk)
    qdrant.upsert(collection, chunks, vectors, _categories(documents), dump_date=dump_date)
    pg.finish_run(run_id, doc_count=len(documents), chunk_count=len(chunks))
    return IngestResult(
        run_id=run_id,
        documents=len(documents),
        chunks=len(chunks),
        points=qdrant.count(collection),
        chunk_sec=round(chunk_sec, 1),
        embed_sec=round(embed_sec, 1),
    )


def _categories(documents: Sequence[Document]) -> Mapping[str, Sequence[str]]:
    return {document.id: document.categories for document in documents}
