"""인제스트 흐름: 문서 → 청킹 → 임베딩 → PG · Qdrant 저장."""

from playground_retrieval.ingest.pipeline import IngestResult, ingest_documents

__all__ = ["IngestResult", "ingest_documents"]
