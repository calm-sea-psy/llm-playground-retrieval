"""인제스트 흐름 테스트: 실제 PG(새 스키마) · Qdrant(새 컬렉션)에 넣고 Dense 검색으로 되찾는다. 서비스가 없으면 건너뛴다."""

import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import psycopg
import pytest
from numpy.typing import NDArray
from playground_corpus import (
    Chunker,
    ChunkingConfig,
    Document,
    EmbeddingCache,
    EmbeddingModelSpec,
    HFTokenizer,
    Section,
)
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from tokenizers import Tokenizer as Backend
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from playground_retrieval.ingest import ingest_documents
from playground_retrieval.search import DenseRetriever
from playground_retrieval.store import PgConfig, PgStore, QdrantStore
from playground_retrieval.store.qdrant import URL as QDRANT_URL

_SPEC = EmbeddingModelSpec(name="fake", model_id="test/fake", dim=3, max_seq_length=64)
_AXES = {"서울": 0, "파리": 1, "도쿄": 2}
DOCS = [
    Document(
        id="1",
        title="서울",
        text="서울 은 한국 의 수도 이다",
        sections=(Section(0, "", "서울 은 한국 의 수도 이다"),),
        categories=("도시",),
    ),
    Document(
        id="2",
        title="파리",
        text="파리 는 프랑스 의 수도 이다",
        sections=(Section(0, "", "파리 는 프랑스 의 수도 이다"),),
    ),
]


class FakeEmbedder:
    """제목 단어 축에 1 을 둔 3차원 벡터. 호출 수를 센다 (캐시 재사용 확인)."""

    spec = _SPEC

    def __init__(self) -> None:
        self.passage_calls = 0

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for row, text in enumerate(texts):
            for word, axis in _AXES.items():
                if word in text:
                    vectors[row, axis] = 1.0
        return vectors

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode(texts)

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.passage_calls += 1
        return self._encode(texts)


@pytest.fixture
def chunker(tmp_path: Path) -> Chunker:
    backend = Backend(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    return Chunker(
        HFTokenizer.from_file(tmp_path / "tokenizer.json"),
        ChunkingConfig(max_tokens=32, min_chars=1),
    )


@pytest.fixture
def stores() -> Iterator[tuple[PgStore, QdrantStore, str]]:
    name = f"test_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(PgConfig.from_env().conninfo, connect_timeout=2)
        QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False).get_collections()
    except (psycopg.OperationalError, ResponseHandlingException):
        pytest.skip("PG · Qdrant 가 떠 있지 않음")
    client = QdrantClient(url=QDRANT_URL, timeout=30, check_compatibility=False)
    conn.execute(f"CREATE SCHEMA {name}")
    conn.execute(f"SET search_path TO {name}")
    conn.commit()
    pg, qdrant = PgStore(conn), QdrantStore(client)
    try:
        yield pg, qdrant, name
    finally:
        qdrant.delete_collection(name)
        conn.rollback()
        conn.execute(f"DROP SCHEMA {name} CASCADE")
        conn.commit()
        pg.close()


def test_ingest_stores_chunks_vectors_and_run(
    stores: tuple[PgStore, QdrantStore, str], chunker: Chunker, tmp_path: Path
):
    pg, qdrant, collection = stores
    embedder = FakeEmbedder()
    cache = EmbeddingCache(tmp_path / "emb")

    result = ingest_documents(
        DOCS,
        source="test",
        collection=collection,
        chunker=chunker,
        embedder=embedder,
        cache=cache,
        pg=pg,
        qdrant=qdrant,
        dump_date="20261001",
    )

    assert (result.documents, result.chunks, result.points) == (2, 2, 2)
    assert pg.count("ingest_runs") == 1
    dense = DenseRetriever(embedder, qdrant, pg, collection=collection, source="test")
    hits = dense.search("파리", k=1)
    assert [hit.chunk.title for hit in hits] == ["파리"]
    assert dense.search("서울", k=2, filters=None)[0].chunk.doc_id == "1"


def test_ingest_again_overwrites_and_reuses_embedding_cache(
    stores: tuple[PgStore, QdrantStore, str], chunker: Chunker, tmp_path: Path
):
    pg, qdrant, collection = stores
    embedder = FakeEmbedder()
    cache = EmbeddingCache(tmp_path / "emb")
    kwargs = {
        "source": "test",
        "collection": collection,
        "chunker": chunker,
        "embedder": embedder,
        "cache": cache,
        "pg": pg,
        "qdrant": qdrant,
    }

    ingest_documents(DOCS, **kwargs)
    again = ingest_documents(DOCS, **kwargs)

    assert again.points == 2  # 같은 청크는 덮어쓴다
    assert embedder.passage_calls == 1  # 두 번째는 캐시 샤드를 다시 계산하지 않는다
    assert pg.count("ingest_runs") == 2
