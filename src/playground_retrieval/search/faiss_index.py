"""FAISS 벡터 색인. DenseRetriever 의 VectorIndex 계약을 채워 Qdrant 와 바꿔 끼운다 (ANN 벤치용, 서빙은 Qdrant)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from playground_retrieval.core import VectorHit

FLAT = "Flat"  # 전수 비교 — ANN 벤치의 정답
# 필터가 남긴 행이 이보다 적으면 그 행만 전수 비교한다 (Qdrant full_scan_threshold 와 같은 방식).
# HNSW 그래프 · IVF 목록 탐색은 필터가 좁을수록 허용된 후보를 k건 채우지 못한다
EXACT_FILTER_ROWS = 20_000


class FaissIndex:
    """메모리에 올린 FAISS 색인 하나 = 컬렉션 하나. 코사인 유사도(정규화 + 내적)로 찾는다.

    예)
        index = FaissIndex.build(vectors, chunk_ids, doc_ids, collection="kowiki_kure-v1", factory="HNSW32")
        dense = DenseRetriever(embedder, index, pg, collection="kowiki_kure-v1", source="kowiki")
    """

    def __init__(
        self,
        index: faiss.Index,
        chunk_ids: Sequence[str],
        doc_ids: Sequence[str],
        *,
        collection: str,
        doc_categories: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        if not index.ntotal == len(chunk_ids) == len(doc_ids):
            raise ValueError(
                f"index, chunk_ids and doc_ids must have the same length, "
                f"got {index.ntotal=} {len(chunk_ids)=} {len(doc_ids)=}"
            )
        ivf = faiss.try_extract_index_ivf(index)
        if ivf is not None:
            ivf.make_direct_map()  # 필터 전수 비교에서 행 번호로 벡터를 꺼낸다
        self.index = index
        """FAISS 색인. 벤치는 여기에 nprobe · efSearch 를 바꿔 넣는다."""
        self.chunk_ids = tuple(chunk_ids)
        """색인 행 번호 → 청크 id."""
        self.collection = collection
        self._doc_ids = np.array(doc_ids)
        self._doc_categories = doc_categories or {}

    @classmethod
    def build(
        cls,
        vectors: NDArray[np.float32],
        chunk_ids: Sequence[str],
        doc_ids: Sequence[str],
        *,
        collection: str,
        factory: str = FLAT,
        doc_categories: Mapping[str, frozenset[str]] | None = None,
    ) -> FaissIndex:
        """벡터를 정규화해 `faiss.index_factory` 문자열(Flat · HNSW32 · IVF1024,PQ64 …)대로 학습 · 추가한다.

        Args:
          vectors: (n, d). chunk_ids 와 같은 순서. 원본은 바꾸지 않는다.
          chunk_ids: 행마다 청크 id.
          doc_ids: 행마다 문서 id (필터용).
          collection: 이 색인이 담은 컬렉션 이름. search 의 name 과 같아야 한다.
          factory: FAISS index_factory 문자열.
          doc_categories: 문서 id → 분류 집합 (분류 필터용).
        """
        normalized = _normalize(vectors)
        index = faiss.index_factory(normalized.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(normalized)
        index.add(normalized)
        return cls(index, chunk_ids, doc_ids, collection=collection, doc_categories=doc_categories)

    def search(
        self,
        name: str,
        vector: NDArray[np.float32],
        k: int,
        *,
        doc_ids: Sequence[str] = (),
        categories: Sequence[str] = (),
    ) -> list[VectorHit]:
        """벡터와 가까운 청크 k건을 코사인 유사도 내림차순으로 돌려준다. doc_ids · categories 가 있으면 그 안에서만."""
        if name != self.collection:
            raise ValueError(f"this index holds {self.collection!r}, got {name=}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k=}")
        query = _normalize(vector.reshape(1, -1))
        params = None
        if doc_ids or categories:
            allowed = np.flatnonzero(self._allowed(doc_ids, categories)).astype(np.int64)
            if len(allowed) <= EXACT_FILTER_ROWS:
                return self._exact(query[0], allowed, k)
            params = _search_params(self.index, faiss.IDSelectorBatch(allowed))
        scores, rows = self.index.search(query, k, params=params)
        return [
            self._hit(row, score)
            for score, row in zip(scores[0], rows[0], strict=True)
            if row >= 0  # 후보가 k건보다 적으면 -1 로 채운다
        ]

    def _exact(
        self, query: NDArray[np.float32], rows: NDArray[np.int64], k: int
    ) -> list[VectorHit]:
        """허용된 행만 전수 비교. 압축 색인(PQ · SQ)은 복원한 근사 벡터로 비교한다."""
        if not len(rows):
            return []
        scores = self.index.reconstruct_batch(rows) @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._hit(int(rows[i]), float(scores[i])) for i in order]

    def _hit(self, row: int, score: float) -> VectorHit:
        return VectorHit(
            chunk_id=self.chunk_ids[row], doc_id=str(self._doc_ids[row]), score=float(score)
        )

    def _allowed(self, doc_ids: Sequence[str], categories: Sequence[str]) -> NDArray[np.bool_]:
        mask = np.ones(len(self._doc_ids), dtype=bool)
        if doc_ids:
            mask &= np.isin(self._doc_ids, list(doc_ids))
        if categories:
            wanted = frozenset(categories)
            allowed_docs = [doc for doc, cats in self._doc_categories.items() if cats & wanted]
            mask &= np.isin(self._doc_ids, allowed_docs)
        return mask


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2 정규화한 복사본. 정규화 벡터의 내적 = 코사인 유사도."""
    normalized = np.ascontiguousarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    return normalized


def _search_params(index: faiss.Index, selector: faiss.IDSelector) -> faiss.SearchParameters:
    """필터를 단 검색 인자. 색인 종류별 인자 클래스를 써야 nprobe · efSearch 가 유지된다."""
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        return faiss.SearchParametersIVF(sel=selector, nprobe=ivf.nprobe)
    if isinstance(index, faiss.IndexHNSW):
        return faiss.SearchParametersHNSW(sel=selector, efSearch=index.hnsw.efSearch)
    return faiss.SearchParameters(sel=selector)
