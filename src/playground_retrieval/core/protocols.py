"""코퍼스 · 검색기를 바꿔 끼우는 계약."""

from collections.abc import Iterator
from typing import Protocol

from playground_corpus import Document

from playground_retrieval.core.types import Filters, Hit


class Source(Protocol):
    """문서를 내놓는 코퍼스. 청킹 이후 단계는 직접 부르지 않는다."""

    def documents(self) -> Iterator[Document]:
        """정제된 문서를 하나씩 돌려준다."""
        ...


class Retriever(Protocol):
    """질의로 청크를 찾는 검색기 (Qdrant · BM25 · FAISS · 하이브리드)."""

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """질의와 가까운 청크 k건을 점수 내림차순으로 돌려준다."""
        ...
