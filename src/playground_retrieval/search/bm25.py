"""BM25 키워드 검색기. 청크를 Kiwi 토큰으로 색인한 numpy 역색인으로 점수를 내고, 원문은 저장소에서 꺼낸다."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import batched
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit
from playground_retrieval.search.analyzer import KiwiAnalyzer
from playground_retrieval.search.dense import ChunkStore

logger = logging.getLogger(__name__)

K1 = 1.2  # Lucene 기본값. 단어 빈도가 점수에 미치는 포화 정도
B = 0.75  # Lucene 기본값. 청크 길이 보정 강도
BUILD_BATCH = 20_000  # 한 번에 형태소 분석하는 청크 수. 전체 규모에서도 배치 메모리는 수백 MB
_META = "meta.json"


class Bm25Index:
    """단어 → (청크 행 번호, 미리 계산한 BM25 가중치) 역색인. 단어별 구간이 offsets 로 나뉜 배열 셋.

    가중치 = idf × tf 포화 · 길이 보정, idf 는 Lucene 식 ln(1 + (N − df + 0.5) / (df + 0.5)).
    질의 단어가 나오는 청크만 더하므로 질의 하나가 청크 수에 비례하지 않는다.

    예)
        index = Bm25Index.build(chunks, KiwiAnalyzer())
        index.save(Path("data/index/bm25/kowiki_kure-v1"))
        index = Bm25Index.load(Path("data/index/bm25/kowiki_kure-v1"))
    """

    def __init__(
        self,
        terms: Sequence[str],
        offsets: NDArray[np.int64],
        rows: NDArray[np.int32],
        weights: NDArray[np.float32],
        chunk_ids: Sequence[str],
        doc_ids: Sequence[str],
    ) -> None:
        if len(offsets) != len(terms) + 1:
            raise ValueError(f"offsets must have len(terms) + 1 items, got {len(offsets)=}")
        if len(chunk_ids) != len(doc_ids):
            raise ValueError(f"chunk_ids and doc_ids differ: {len(chunk_ids)=} {len(doc_ids)=}")
        self._slot = {term: slot for slot, term in enumerate(terms)}
        self._terms = list(terms)
        self._offsets = offsets
        self._rows = rows
        self._weights = weights
        self.chunk_ids = tuple(chunk_ids)
        """행 번호 → 청크 id."""
        self.doc_ids = np.array(doc_ids)
        """행 번호 → 문서 id (필터용)."""

    @classmethod
    def build(
        cls,
        chunks: Iterable[Chunk],
        analyzer: KiwiAnalyzer,
        *,
        k1: float = K1,
        b: float = B,
        batch_size: int = BUILD_BATCH,
    ) -> Bm25Index:
        """청크를 배치로 형태소 분석해 역색인을 만든다. chunks 는 한 번만 훑는다 (PG 스트리밍 가능)."""
        slots: dict[str, int] = {}
        chunk_ids: list[str] = []
        doc_ids: list[str] = []
        lengths: list[int] = []
        term_parts: list[NDArray[np.int32]] = []
        row_parts: list[NDArray[np.int32]] = []
        tf_parts: list[NDArray[np.float32]] = []
        for batch in batched(chunks, batch_size, strict=False):  # 마지막 배치는 작아도 된다
            for tokens in analyzer.tokens_batch([chunk.text for chunk in batch]):
                counts = Counter(tokens)
                row = len(lengths)
                lengths.append(len(tokens))
                term_parts.append(
                    np.fromiter((slots.setdefault(t, len(slots)) for t in counts), np.int32)
                )
                row_parts.append(np.full(len(counts), row, dtype=np.int32))
                tf_parts.append(np.fromiter(counts.values(), np.float32))
            chunk_ids.extend(chunk.id for chunk in batch)
            doc_ids.extend(chunk.doc_id for chunk in batch)
            logger.info("bm25: %d chunks analyzed, %d terms", len(lengths), len(slots))
        offsets, rows, weights = _postings(
            np.concatenate(term_parts) if term_parts else np.empty(0, np.int32),
            np.concatenate(row_parts) if row_parts else np.empty(0, np.int32),
            np.concatenate(tf_parts) if tf_parts else np.empty(0, np.float32),
            np.array(lengths, dtype=np.float32),
            n_terms=len(slots),
            k1=k1,
            b=b,
        )
        terms = sorted(slots, key=slots.__getitem__)
        return cls(terms, offsets, rows, weights, chunk_ids, doc_ids)

    @classmethod
    def load(cls, directory: Path, *, mmap: bool = True) -> Bm25Index:
        """save() 한 폴더를 읽는다. mmap 이면 큰 배열은 디스크에 두고 필요한 구간만 읽는다."""
        mode = "r" if mmap else None
        return cls(
            terms=_read_json(directory / "terms.json"),
            offsets=np.load(directory / "offsets.npy", mmap_mode=mode),
            rows=np.load(directory / "rows.npy", mmap_mode=mode),
            weights=np.load(directory / "weights.npy", mmap_mode=mode),
            chunk_ids=_read_json(directory / "chunk_ids.json"),
            doc_ids=_read_json(directory / "doc_ids.json"),
        )

    def save(self, directory: Path, *, meta: Mapping[str, object] | None = None) -> None:
        """배열은 .npy, 단어 · id 목록은 JSON 으로 쓴다. meta 는 meta.json 에 그대로 (출처 기록용)."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "offsets.npy", self._offsets)
        np.save(directory / "rows.npy", self._rows)
        np.save(directory / "weights.npy", self._weights)
        _write_json(directory / "terms.json", self._terms)
        _write_json(directory / "chunk_ids.json", list(self.chunk_ids))
        _write_json(directory / "doc_ids.json", self.doc_ids.tolist())
        _write_json(directory / _META, {"chunks": len(self.chunk_ids), **(meta or {})})

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def scores(self, terms: Iterable[str]) -> NDArray[np.float32]:
        """청크마다 BM25 점수 (n,). 같은 단어가 여러 번이면 그만큼 더한다. 모르는 단어는 무시."""
        scores = np.zeros(len(self.chunk_ids), dtype=np.float32)
        for term, count in Counter(terms).items():
            slot = self._slot.get(term)
            if slot is None:
                continue
            start, end = self._offsets[slot], self._offsets[slot + 1]
            scores[self._rows[start:end]] += count * self._weights[start:end]
        return scores


class Bm25Retriever:
    """BM25 색인으로 청크를 찾고 원문은 저장소에서 꺼낸다.

    예)
        bm25 = Bm25Retriever(Bm25Index.load(path), KiwiAnalyzer(), pg, collection="kowiki_kure-v1", source="kowiki")
        bm25 = Bm25Retriever.from_chunks(chunks, KiwiAnalyzer())  # 서브셋 · 테스트: 메모리에서 색인 + 원문
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


def _postings(
    terms: NDArray[np.int32],
    rows: NDArray[np.int32],
    tf: NDArray[np.float32],
    lengths: NDArray[np.float32],
    *,
    n_terms: int,
    k1: float,
    b: float,
) -> tuple[NDArray[np.int64], NDArray[np.int32], NDArray[np.float32]]:
    """(단어, 청크, tf) 쌍을 단어 순으로 정렬하고 BM25 가중치를 계산한다. 단어 안에서는 청크 순."""
    order = np.lexsort((rows, terms))
    terms, rows, tf = terms[order], rows[order], tf[order]
    df = np.bincount(terms, minlength=n_terms)
    offsets = np.zeros(n_terms + 1, dtype=np.int64)
    np.cumsum(df, out=offsets[1:])
    total = len(lengths)
    idf = np.log1p((total - df + 0.5) / (df + 0.5)).astype(np.float32)
    average = float(lengths.mean()) if total else 0.0
    norm = k1 * (1 - b + b * lengths[rows] / average) if average else np.full_like(tf, k1)
    weights = (idf[terms] * tf * (k1 + 1) / (tf + norm)).astype(np.float32)
    return offsets, rows, weights


def _read_json(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
