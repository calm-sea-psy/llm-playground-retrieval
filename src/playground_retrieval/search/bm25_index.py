"""BM25 역색인: 청크를 Kiwi 토큰으로 색인해 numpy 배열 셋으로 들고, 저장 · 불러오기 · 점수 계산을 한다."""

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

from playground_retrieval.search.analyzer import KiwiAnalyzer

logger = logging.getLogger(__name__)

K1 = 1.2  # Lucene 기본값. 단어 빈도가 점수에 미치는 포화 정도
B = 0.75  # Lucene 기본값. 청크 길이 보정 강도
BUILD_BATCH = 20_000  # 한 번에 형태소 분석하는 청크 수. 전체 규모에서도 배치 메모리는 수백 MB
META_FILE = "meta.json"  # save() 가 마지막에 쓰는 파일. 있으면 저장된 색인으로 본다


class Bm25Index:
    """단어 → (청크 행 번호, 미리 계산한 BM25 가중치) 역색인. 단어별 구간이 offsets 로 나뉜 배열 셋.

    가중치 = idf × tf 포화 · 길이 보정, idf 는 Lucene 식 ln(1 + (N − df + 0.5) / (df + 0.5)).
    질의 단어가 나오는 청크만 더하므로 질의 하나가 청크 수에 비례하지 않는다.

    Args:
      terms: 단어 t개. 순서가 단어 번호.
      offsets: (t + 1,). 단어 i 의 구간은 rows[offsets[i] : offsets[i + 1]].
      rows: (p,). 구간마다 청크 행 번호. p 는 (단어, 청크) 쌍의 수.
      weights: (p,). rows 와 같은 순서의 BM25 가중치.
      chunk_ids: 청크 n개의 id.
      doc_ids: 청크 n개의 문서 id. chunk_ids 와 같은 순서.

    예)
        index = Bm25Index.build(chunks, KiwiAnalyzer.load())
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
        self.doc_ids = np.array(doc_ids, dtype=np.str_)
        """행 번호 → 문서 id (필터용). (n,)."""

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
        _write_json(directory / META_FILE, {"chunks": len(self.chunk_ids), **(meta or {})})

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
    """(단어, 청크, tf) 쌍을 단어 순으로 정렬하고 BM25 가중치를 계산한다. 단어 안에서는 청크 순.

    Args:
      terms: (p,). 쌍마다 단어 번호.
      rows: (p,). 쌍마다 청크 행 번호.
      tf: (p,). 쌍마다 단어 빈도.
      lengths: (n,). 청크마다 토큰 수.
      n_terms: 단어 수 t.
      k1: BM25 k1.
      b: BM25 b.

    Returns:
      (offsets, rows, weights). offsets 는 (t + 1,), 나머지 둘은 (p,).
    """
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
