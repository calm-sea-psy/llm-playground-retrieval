"""후보를 cross-encoder 로 다시 매기는 리랭커. 다른 검색기 위에 얹는 Retriever 다."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from playground_retrieval.core import Filters, Hit, Retriever

CANDIDATES = (
    20  # 리랭커에 넘기는 후보 수. 위키 서브셋 평가(D12)에서 쓴 값 — 제품이 설정으로 바꿔 넘긴다
)


class PairScorer(Protocol):
    """(질의, 본문) 쌍마다 관련도 점수를 내는 모델."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        """쌍 n개의 점수 (n,). 0~1, 클수록 관련 있다."""
        ...


class Reranker:
    """기반 검색기에서 후보를 받아 PairScorer 점수로 다시 정렬한다.

    점수는 쌍마다 독립인 0~1 값이라 질의끼리 비교할 수 있다 — 답변 정책의 기준값(T_high · T_low)에 쓴다.

    예)
        reranker = Reranker(HybridRetriever([dense, bm25]), CrossEncoderScorer.load())
        hits = reranker.search("조선의 수도는?", k=5)
    """

    def __init__(
        self, base: Retriever, scorer: PairScorer, *, candidates: int = CANDIDATES
    ) -> None:
        if candidates < 1:
            raise ValueError(f"candidates must be >= 1, got {candidates=}")
        self._base = base
        self._scorer = scorer
        self._candidates = candidates

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """후보를 다시 매겨 점수 상위 k건을 점수 내림차순으로 돌려준다. 동점이면 기반 검색기 순서."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k=}")
        candidates = self._base.search(query, max(k, self._candidates), filters)
        if not candidates:
            return []
        scores = self._scorer.score([(query, hit.chunk.text) for hit in candidates])
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(chunk=candidates[i].chunk, score=float(scores[i])) for i in order]
