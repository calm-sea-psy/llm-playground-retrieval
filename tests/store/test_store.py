"""저장소 테스트. PG · Qdrant 가 떠 있지 않으면(docker compose up) 연동 테스트는 건너뛴다."""

import uuid
from collections.abc import Iterator

import numpy as np
import psycopg
import pytest
from playground_corpus import Chunk, Document, Section
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from playground_retrieval.store import (
    PgConfig,
    PgStore,
    QdrantConfig,
    QdrantStore,
    chunk_payload,
    point_id,
)
from playground_retrieval.store.qdrant import URL as QDRANT_URL

DOCS = [
    Document(
        id="1",
        title="서울",
        text="수도",
        sections=(Section(0, "", "수도"),),
        categories=("도시",),
        meta={"url": "u1", "dump_date": "20261001"},
    ),
    Document(id="2", title="파리", text="수도", sections=(Section(0, "", "수도"),)),
]
CHUNKS = [
    Chunk(id="1-0", doc_id="1", title="서울", section="", text="서울\n수도", seq=0, token_count=3),
    Chunk(
        id="2-0",
        doc_id="2",
        title="파리",
        section="역사",
        text="파리 > 역사\n수도",
        seq=0,
        token_count=5,
    ),
]


QDRANT_ENV = ("QDRANT_MODE", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_GRPC", "QDRANT_TIMEOUT")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in QDRANT_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_qdrant_config_defaults_to_local_grpc(clean_env: pytest.MonkeyPatch):
    assert QdrantConfig.from_env() == QdrantConfig(
        url=QDRANT_URL, api_key=None, prefer_grpc=True, timeout=None
    )


def test_qdrant_config_ignores_cloud_address_unless_mode_is_cloud(clean_env: pytest.MonkeyPatch):
    clean_env.setenv("QDRANT_URL", "https://example.cloud.qdrant.io:6333")
    clean_env.setenv("QDRANT_API_KEY", "secret-key")

    local = QdrantConfig.from_env()
    clean_env.setenv("QDRANT_MODE", "local")

    assert (local.url, local.api_key) == (QDRANT_URL, None)  # 주소 · 키가 있어도 켜지 않으면 로컬
    assert QdrantConfig.from_env() == local


def test_qdrant_config_reads_cloud_settings_and_hides_key(clean_env: pytest.MonkeyPatch):
    clean_env.setenv("QDRANT_MODE", "cloud")
    clean_env.setenv("QDRANT_URL", "https://example.cloud.qdrant.io:6333")
    clean_env.setenv("QDRANT_API_KEY", "secret-key")
    clean_env.setenv("QDRANT_GRPC", "0")
    clean_env.setenv("QDRANT_TIMEOUT", "60")

    config = QdrantConfig.from_env()

    assert config.url == "https://example.cloud.qdrant.io:6333"
    assert config.api_key == "secret-key"
    assert (config.prefer_grpc, config.timeout) == (False, 60)
    assert "secret-key" not in repr(config)


def test_qdrant_config_rejects_cloud_without_url_and_unknown_mode(clean_env: pytest.MonkeyPatch):
    clean_env.setenv("QDRANT_MODE", "cloud")
    with pytest.raises(ValueError, match="needs QDRANT_URL"):
        QdrantConfig.from_env()

    clean_env.setenv("QDRANT_MODE", "Cloud")
    with pytest.raises(ValueError, match="must be one of"):
        QdrantConfig.from_env()


def test_qdrant_config_treats_empty_values_as_unset(clean_env: pytest.MonkeyPatch):
    clean_env.setenv("QDRANT_MODE", "")
    clean_env.setenv("QDRANT_URL", "")
    clean_env.setenv("QDRANT_API_KEY", "")  # .env.example 를 복사만 하고 비워 둔 경우

    config = QdrantConfig.from_env()

    assert (config.url, config.api_key) == (QDRANT_URL, None)


def test_point_id_is_stable_uuid():
    assert point_id("1-0") == point_id("1-0")
    assert point_id("1-0") != point_id("1-1")
    uuid.UUID(point_id("1-0"))


def test_chunk_payload_has_filters_but_no_text():
    payload = chunk_payload(CHUNKS[1], ["도시"], "20261001")

    assert payload == {
        "chunk_id": "2-0",
        "doc_id": "2",
        "title": "파리",
        "section": "역사",
        "seq": 0,
        "categories": ["도시"],
        "dump_date": "20261001",
    }


@pytest.fixture
def pg() -> Iterator[PgStore]:
    """테스트마다 새 스키마를 만들어 쓰고 지운다."""
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(PgConfig.from_env().conninfo, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL 이 떠 있지 않음 (docker compose up)")
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}")
    conn.commit()
    store = PgStore(conn)
    try:
        yield store
    finally:
        conn.rollback()
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        store.close()


def test_pg_upserts_and_reads_chunks_in_requested_order(pg: PgStore):
    pg.create_schema()
    pg.upsert_documents("kowiki", DOCS)
    pg.upsert_documents("kowiki", DOCS)  # 다시 넣어도 덮어쓴다
    pg.upsert_chunks("kowiki_test", CHUNKS)

    assert pg.count("documents") == 2
    assert pg.get_chunks("kowiki_test", "kowiki", ["2-0", "없음", "1-0"]) == [CHUNKS[1], CHUNKS[0]]


def test_pg_reads_all_chunks_and_categories_of_collection(pg: PgStore):
    pg.create_schema()
    pg.upsert_documents("kowiki", DOCS)
    pg.upsert_chunks("kowiki_test", CHUNKS)
    pg.upsert_chunks("other", CHUNKS[:1])

    assert pg.all_chunks("kowiki_test", "kowiki") == CHUNKS
    assert pg.all_chunks("other", "kowiki") == CHUNKS[:1]
    assert list(pg.iter_chunks("kowiki_test", "kowiki", batch_size=1)) == CHUNKS
    assert pg.collection_titles("other", "kowiki") == {"서울"}
    assert pg.count("chunks") == 3  # 흘려 읽은 뒤 rollback 해도 적재한 청크는 남는다
    assert pg.doc_categories("kowiki_test", "kowiki") == {
        "1": frozenset({"도시"}),
        "2": frozenset(),
    }
    assert pg.doc_categories("other", "kowiki") == {"1": frozenset({"도시"})}


def test_pg_records_ingest_runs(pg: PgStore):
    pg.create_schema()
    run_id = pg.start_run("kowiki", "kowiki_test", "bge-m3", 384)
    pg.finish_run(run_id, doc_count=2, chunk_count=2)

    assert pg.count("ingest_runs") == 1


@pytest.fixture
def qdrant() -> Iterator[tuple[QdrantStore, str]]:
    try:
        QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False).get_collections()
    except ResponseHandlingException:  # 접속 실패를 qdrant-client 가 감싸서 던진다
        pytest.skip("Qdrant 가 떠 있지 않음 (docker compose up)")
    # 떠 있는지는 2초로 보고, 컬렉션 생성 같은 작업은 넉넉히 기다린다 (CI 러너에서 2초를 넘긴 적이 있다)
    client = QdrantClient(url=QDRANT_URL, timeout=30, check_compatibility=False)
    name = f"test_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(client)
    try:
        yield store, name
    finally:
        store.delete_collection(name)


def test_qdrant_upserts_and_searches_with_filters(qdrant: tuple[QdrantStore, str]):
    store, name = qdrant
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    store.ensure_collection(name, dim=4)
    store.ensure_collection(name, dim=4)  # 이미 있으면 그대로

    store.upsert(name, CHUNKS, vectors, {"1": ["도시"]}, dump_date="20261001")
    store.upsert(name, CHUNKS, vectors, {"1": ["도시"]})  # 같은 청크는 덮어쓴다

    assert store.count(name) == 2
    query = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
    assert [hit.chunk_id for hit in store.search(name, query, k=2)] == ["1-0", "2-0"]
    assert [hit.chunk_id for hit in store.search(name, query, k=2, doc_ids=["2"])] == ["2-0"]
    assert [hit.chunk_id for hit in store.search(name, query, k=2, categories=["도시"])] == ["1-0"]
