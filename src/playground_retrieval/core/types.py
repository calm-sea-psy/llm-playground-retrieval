"""검색 결과 · 검색 범위 자료형."""

from dataclasses import dataclass

from playground_corpus import Chunk


@dataclass(frozen=True)
class Hit:
    """검색 결과 한 건."""

    chunk: Chunk
    score: float
    """검색기마다 척도가 다르다 (dense 는 코사인, BM25 는 원점수, 리랭커는 0~1)."""


@dataclass(frozen=True)
class Filters:
    """검색 범위 제한. 빈 필드는 제한 없음."""

    doc_ids: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()
    """이 중 하나라도 속한 문서만."""


@dataclass(frozen=True)
class VectorHit:
    """벡터 색인(Qdrant · FAISS)의 검색 결과 한 건. 원문은 아직 붙지 않은 상태."""

    chunk_id: str
    doc_id: str
    score: float
    """코사인 유사도."""
