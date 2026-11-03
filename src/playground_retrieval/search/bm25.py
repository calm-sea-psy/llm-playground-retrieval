"""BM25 키워드 검색기. 역색인(`bm25_index`)으로 점수를 내고 원문은 저장소에서 꺼낸다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit
from playground_retrieval.search.analyzer import KiwiAnalyzer
from playground_retrieval.search.bm25_index import K1, B, Bm25Index
from playground_retrieval.search.dense import ChunkStore

__all__ = ["K1", "B", "Bm25Index", "Bm25Retriever"]  # K1 · B 는 위키가 이 모듈에서 가져간다


class Bm25Retriever:
    """BM25 색인으로 청크를 찾고 원문은 저장소에서 꺼낸다.

    예)
        bm25 = Bm25Retriever(Bm25Index.load(path), KiwiAnalyzer.load(), pg, collection="kowiki_kure-v1", source="kowiki")
        bm25 = Bm25Retriever.from_chunks(chunks, KiwiAnalyzer.load())  # 서브셋 · 테스트: 메모리에서 색인 + 원문
        hits = bm25.search("조선의 수도는?", k=5)
    """

    def __init__(
        self,
        index: Bm25Index,
        analyzer: KiwiAnalyzer,
        store: ChunkStore,
        *,
        collection: str,
        source: str,
        doc_categories: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._index = index
        self._analyzer = analyzer
        self._store = store
        self.collection = collection
        self.source = source
        self._doc_categories = doc_categories or {}

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        analyzer: KiwiAnalyzer,
        *,
        k1: float = K1,
        b: float = B,
        doc_categories: Mapping[str, frozenset[str]] | None = None,
    ) -> Bm25Retriever:
        """청크 목록으로 색인을 만들고 원문도 메모리에 둔다 (서브셋 평가 · 테스트용)."""
        index = Bm25Index.build(chunks, analyzer, k1=k1, b=b)
        return cls(
            index,
            analyzer,
            _MemoryChunks(chunks),
            collection="",
            source="",
            doc_categories=doc_categories,
        )

    @property
    def index(self) -> Bm25Index:
        """검색에 쓰는 역색인."""
        return self._index

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """질의와 BM25 점수가 높은 청크 k건을 점수 내림차순으로 돌려준다. 겹치는 단어가 없는 청크는 빼고."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k=}")
        scores = self._index.scores(self._analyzer.tokens(query))
        if filters is not None:
            scores[~self._allowed(filters)] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if len(candidates) > k:
            candidates = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
        chunk_ids = [self._index.chunk_ids[i] for i in ordered]
        chunks = {
            chunk.id: chunk
            for chunk in self._store.get_chunks(self.collection, self.source, chunk_ids)
        }
        return [
            Hit(chunk=chunks[self._index.chunk_ids[i]], score=float(scores[i]))
            for i in ordered
            if self._index.chunk_ids[i] in chunks
        ]

    def _allowed(self, filters: Filters) -> NDArray[np.bool_]:
        """청크 행마다 필터 통과 여부 (n,)."""
        doc_ids = self._index.doc_ids
        mask = np.ones(len(doc_ids), dtype=bool)
        if filters.doc_ids:
            mask &= np.isin(doc_ids, list(filters.doc_ids))
        if filters.categories:
            allowed_docs = [
                doc for doc, cats in self._doc_categories.items() if cats & filters.categories
            ]
            mask &= np.isin(doc_ids, allowed_docs)
        return mask


class _MemoryChunks:
    """청크 목록을 id 로 찾는 ChunkStore. collection · source 는 보지 않는다."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._by_id = {chunk.id: chunk for chunk in chunks}

    def get_chunks(self, collection: str, source: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        del collection, source  # 메모리에는 한 컬렉션만 있다
        return [self._by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in self._by_id]
