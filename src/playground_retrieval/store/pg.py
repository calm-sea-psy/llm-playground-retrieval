"""PostgreSQL 저장소: 문서 · 청크 원문과 인제스트 실행 기록. 벡터는 Qdrant 에 둔다."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
from playground_corpus import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source      text NOT NULL,
    id          text NOT NULL,
    title       text NOT NULL,
    url         text,
    categories  text[] NOT NULL DEFAULT '{}',
    dump_date   text,
    PRIMARY KEY (source, id)
);
CREATE TABLE IF NOT EXISTS chunks (
    collection  text NOT NULL,
    id          text NOT NULL,
    doc_id      text NOT NULL,
    seq         integer NOT NULL,
    section     text NOT NULL,
    text        text NOT NULL,
    token_count integer NOT NULL,
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (collection, doc_id);
CREATE TABLE IF NOT EXISTS ingest_runs (
    id          bigserial PRIMARY KEY,
    source      text NOT NULL,
    collection  text NOT NULL,
    model       text NOT NULL,
    max_tokens  integer NOT NULL,
    started_at  timestamptz NOT NULL,
    finished_at timestamptz,
    doc_count   integer,
    chunk_count integer,
    status      text NOT NULL
);
"""
# 청크는 컬렉션(= 소스 × 임베딩 모델 × 청크 설정) 별로 다르므로 (collection, id) 로 구분한다
ITER_BATCH = 10_000  # 서버 쪽 커서가 한 번에 가져오는 행 수 (청크 1만 개 ≈ 15MB)


@dataclass(frozen=True)
class PgConfig:
    """접속 설정. docker-compose.yml 과 같은 기본값."""

    host: str = "localhost"
    port: int = 5432
    user: str = "rag"
    password: str = "rag"  # noqa: S105  # 로컬 개발용 기본값 (.env.example 과 같음)
    dbname: str = "rag"

    @classmethod
    def from_env(cls) -> PgConfig:
        """POSTGRES_HOST · POSTGRES_PORT · POSTGRES_USER · POSTGRES_PASSWORD · POSTGRES_DB 에서 읽는다."""
        default = cls()
        return cls(
            host=os.environ.get("POSTGRES_HOST", default.host),
            port=int(os.environ.get("POSTGRES_PORT", default.port)),
            user=os.environ.get("POSTGRES_USER", default.user),
            password=os.environ.get("POSTGRES_PASSWORD", default.password),
            dbname=os.environ.get("POSTGRES_DB", default.dbname),
        )

    @property
    def conninfo(self) -> str:
        """Psycopg 접속 문자열."""
        return f"host={self.host} port={self.port} user={self.user} password={self.password} dbname={self.dbname}"


class PgStore:
    """문서 · 청크를 넣고 읽는 저장소. 같은 키로 다시 넣으면 덮어쓴다.

    예)
        with PgStore.connect(PgConfig.from_env()) as store:
            store.create_schema()
            store.upsert_documents("kowiki", documents)
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._conn = connection

    @classmethod
    def connect(cls, config: PgConfig) -> PgStore:
        """접속을 연다. `with` 로 쓰면 끝날 때 닫는다."""
        return cls(psycopg.connect(config.conninfo, autocommit=False))

    def __enter__(self) -> PgStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """접속을 닫는다."""
        self._conn.close()

    def create_schema(self) -> None:
        """테이블이 없으면 만든다."""
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)
        self._conn.commit()

    def upsert_documents(self, source: str, documents: Sequence[Document]) -> int:
        """문서를 넣고(같은 키면 덮어씀) 넣은 수를 돌려준다."""
        rows = [
            (
                source,
                doc.id,
                doc.title,
                doc.meta.get("url"),
                list(doc.categories),
                doc.meta.get("dump_date"),
            )
            for doc in documents
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (source, id, title, url, categories, dump_date) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (source, id) DO UPDATE SET title = EXCLUDED.title, url = EXCLUDED.url, "
                "categories = EXCLUDED.categories, dump_date = EXCLUDED.dump_date",
                rows,
            )
        self._conn.commit()
        return len(rows)

    def upsert_chunks(self, collection: str, chunks: Sequence[Chunk]) -> int:
        """청크를 넣고(같은 키면 덮어씀) 넣은 수를 돌려준다."""
        rows = [
            (collection, c.id, c.doc_id, c.seq, c.section, c.text, c.token_count) for c in chunks
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (collection, id, doc_id, seq, section, text, token_count) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (collection, id) DO UPDATE SET doc_id = EXCLUDED.doc_id, seq = EXCLUDED.seq, "
                "section = EXCLUDED.section, text = EXCLUDED.text, token_count = EXCLUDED.token_count",
                rows,
            )
        self._conn.commit()
        return len(rows)

    def get_chunks(self, collection: str, source: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        """청크 id 순서대로 청크를 돌려준다. 문서 제목은 documents 에서 붙인다. 없는 id 는 빠진다."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT c.id, c.doc_id, d.title, c.section, c.text, c.seq, c.token_count "
                "FROM chunks c JOIN documents d ON d.source = %s AND d.id = c.doc_id "
                "WHERE c.collection = %s AND c.id = ANY(%s)",
                (source, collection, list(chunk_ids)),
            )
            by_id = {row[0]: Chunk(*row) for row in cur.fetchall()}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def all_chunks(self, collection: str, source: str) -> list[Chunk]:
        """컬렉션의 청크 전부를 (문서 id, 순서) 순으로 돌려준다. 메모리 색인(BM25)을 만들 때 쓴다."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT c.id, c.doc_id, d.title, c.section, c.text, c.seq, c.token_count "
                "FROM chunks c JOIN documents d ON d.source = %s AND d.id = c.doc_id "
                "WHERE c.collection = %s ORDER BY c.doc_id, c.seq",
                (source, collection),
            )
            return [Chunk(*row) for row in cur.fetchall()]

    def iter_chunks(
        self, collection: str, source: str, *, batch_size: int = ITER_BATCH
    ) -> Iterator[Chunk]:
        """컬렉션의 청크를 (문서 id, 순서) 순으로 흘려 준다. 서버 쪽 커서라 전체를 메모리에 올리지 않는다."""
        try:
            with self._conn.cursor(name="iter_chunks") as cur:
                cur.itersize = batch_size
                cur.execute(
                    "SELECT c.id, c.doc_id, d.title, c.section, c.text, c.seq, c.token_count "
                    "FROM chunks c JOIN documents d ON d.source = %s AND d.id = c.doc_id "
                    "WHERE c.collection = %s ORDER BY c.doc_id, c.seq",
                    (source, collection),
                )
                for row in cur:
                    yield Chunk(*row)
        finally:
            self._conn.rollback()  # 읽기만 한 트랜잭션을 닫는다

    def collection_titles(self, collection: str, source: str) -> set[str]:
        """컬렉션에 청크가 있는 문서의 제목. 평가 질문을 정답 문서가 있는 것으로 거를 때 쓴다."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT d.title FROM documents d WHERE d.source = %s AND EXISTS "
                "(SELECT 1 FROM chunks c WHERE c.collection = %s AND c.doc_id = d.id)",
                (source, collection),
            )
            return {title for (title,) in cur.fetchall()}

    def doc_categories(self, collection: str, source: str) -> dict[str, frozenset[str]]:
        """컬렉션에 청크가 있는 문서마다 분류 집합. BM25 의 분류 필터에 쓴다."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT d.id, d.categories FROM documents d WHERE d.source = %s AND EXISTS "
                "(SELECT 1 FROM chunks c WHERE c.collection = %s AND c.doc_id = d.id)",
                (source, collection),
            )
            return {doc_id: frozenset(categories) for doc_id, categories in cur.fetchall()}

    def start_run(self, source: str, collection: str, model: str, max_tokens: int) -> int:
        """인제스트 실행을 기록하고 실행 id 를 돌려준다."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_runs (source, collection, model, max_tokens, started_at, status) "
                "VALUES (%s, %s, %s, %s, %s, 'running') RETURNING id",
                (source, collection, model, max_tokens, datetime.now(tz=UTC)),
            )
            (run_id,) = cur.fetchone()
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: int, *, doc_count: int, chunk_count: int) -> None:
        """실행을 끝난 것으로 표시한다."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_runs SET finished_at = %s, doc_count = %s, chunk_count = %s, status = 'done' "
                "WHERE id = %s",
                (datetime.now(tz=UTC), doc_count, chunk_count, run_id),
            )
        self._conn.commit()

    def count(self, table: str) -> int:
        """테이블 행 수. table 은 documents · chunks · ingest_runs 중 하나."""
        if table not in {"documents", "chunks", "ingest_runs"}:
            raise ValueError(f"unknown table: {table=}")
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608  # 위에서 허용한 테이블 이름만
            (value,) = cur.fetchone()
        return value
