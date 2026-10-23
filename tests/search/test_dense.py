"""DenseRetriever 테스트: 가짜 의존성으로 조립 동작, 실제 PG · Qdrant 로 끝에서 끝 (서비스가 없으면 건너뜀)."""

import uuid
from collections.abc import Iterator, Sequence

import numpy as np
import psycopg
import pytest
from numpy.typing import NDArray
from playground_corpus import Chunk, Document, EmbeddingModelSpec, Section
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from playground_retrieval.core import Filters, VectorHit
from playground_retrieval.search import DenseRetriever
from playground_retrieval.store import PgConfig, PgStore, QdrantStore
from playground_retrieval.store.qdrant import URL as QDRANT_URL

_SPEC = EmbeddingModelSpec(name="fake", model_id="test/fake", dim=3, max_seq_length=16)
_VECTORS = {"서울": [1.0, 0.0, 0.0], "파리": [0.0, 1.0, 0.0], "수도": [0.7, 0.7, 0.0]}
CHUNKS = [
    Chunk(id="1-0", doc_id="1", title="서울", section="", text="서울\n한양", seq=0, token_count=3),
    Chunk(id="2-0", doc_id="2", title="파리", section="", text="파리\n센강", seq=0, token_count=3),
]


class FakeEmbedder:
    spec = _SPEC

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.array([_VECTORS[text] for text in texts], dtype=np.float32)

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self.encode_queries(texts)


class FakeIndex:
    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, object]] = []

    def search(self, name, vector, k, *, doc_ids=(), categories=()):
        self.calls.append(
            {"name": name, "k": k, "doc_ids": list(doc_ids), "categories": list(categories)}
        )
        return self.hits[:k]


class FakeStore:
    def get_chunks(self, collection: str, source: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        by_id = {chunk.id: chunk for chunk in CHUNKS}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def test_search_returns_chunks_in_vector_order_with_scores():
    index = FakeIndex([VectorHit("2-0", "2", 0.9), VectorHit("1-0", "1", 0.5)])
    dense = DenseRetriever(FakeEmbedder(), index, FakeStore(), collection="c", source="kowiki")

    hits = dense.search("파리", k=2)

    assert [(hit.chunk.id, hit.score) for hit in hits] == [("2-0", 0.9), ("1-0", 0.5)]


def test_search_skips_chunks_missing_in_store():
    index = FakeIndex([VectorHit("없음-0", "x", 0.9), VectorHit("1-0", "1", 0.5)])
    dense = DenseRetriever(FakeEmbedder(), index, FakeStore(), collection="c", source="kowiki")

    assert [hit.chunk.id for hit in dense.search("서울", k=2)] == ["1-0"]


def test_search_forwards_filters_to_index():
    index = FakeIndex([])
    dense = DenseRetriever(FakeEmbedder(), index, FakeStore(), collection="c", source="kowiki")

    dense.search(
        "서울", k=3, filters=Filters(doc_ids=frozenset({"2", "1"}), categories=frozenset({"도시"}))
    )

    assert index.calls == [{"name": "c", "k": 3, "doc_ids": ["1", "2"], "categories": ["도시"]}]


@pytest.fixture
def stores() -> Iterator[tuple[PgStore, QdrantStore, str]]:
    """실제 PG(새 스키마) · Qdrant(새 컬렉션)에 CHUNKS 를 넣어 두고 끝나면 지운다."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(PgConfig.from_env().conninfo, connect_timeout=2)
        QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False).get_collections()
    except (psycopg.OperationalError, ResponseHandlingException):
        pytest.skip("PG · Qdrant 가 떠 있지 않음 (docker compose up)")
    # 떠 있는지는 2초로 보고, 컬렉션 생성 같은 작업은 넉넉히 기다린다 (CI 러너에서 2초를 넘긴 적이 있다)
    client = QdrantClient(url=QDRANT_URL, timeout=30, check_compatibility=False)
    conn.execute(f"CREATE SCHEMA {name}")
    conn.execute(f"SET search_path TO {name}")
    conn.commit()
    pg, qdrant = PgStore(conn), QdrantStore(client)
    pg.create_schema()
    docs = [
        Document(id=c.doc_id, title=c.title, text=c.text, sections=(Section(0, "", c.text),))
        for c in CHUNKS
    ]
    pg.upsert_documents("kowiki", docs)
    pg.upsert_chunks(name, CHUNKS)
    qdrant.ensure_collection(name, dim=3)
    qdrant.upsert(name, CHUNKS, FakeEmbedder().encode_passages([c.title for c in CHUNKS]), {})
    try:
        yield pg, qdrant, name
    finally:
        qdrant.delete_collection(name)
        conn.rollback()
        conn.execute(f"DROP SCHEMA {name} CASCADE")
        conn.commit()
        pg.close()


def test_search_end_to_end_with_real_stores(stores: tuple[PgStore, QdrantStore, str]):
    pg, qdrant, name = stores
    dense = DenseRetriever(FakeEmbedder(), qdrant, pg, collection=name, source="kowiki")

    hits = dense.search("파리", k=2)
    filtered = dense.search("수도", k=2, filters=Filters(doc_ids=frozenset({"1"})))

    assert [hit.chunk for hit in hits] == [CHUNKS[1], CHUNKS[0]]
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert [hit.chunk.id for hit in filtered] == ["1-0"]
