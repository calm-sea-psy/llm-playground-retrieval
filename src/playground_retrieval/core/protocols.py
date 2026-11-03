"""검색기를 바꿔 끼우는 계약."""

from typing import Protocol

from playground_retrieval.core.types import Filters, Hit


class Retriever(Protocol):
    """질의로 청크를 찾는 검색기 (Qdrant · BM25 · FAISS · 하이브리드)."""

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """질의와 가까운 청크 k건을 점수 내림차순으로 돌려준다."""
        ...
