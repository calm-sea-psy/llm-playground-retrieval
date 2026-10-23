"""ANN 벤치마크 지표: 전수 검색(정답) 대비 recall@k 와 지연 분위수."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencySummary:
    """질의 하나씩 보낸 검색 시간 분포."""

    p50_ms: float
    p99_ms: float


def recall_at_k(found: Sequence[Sequence[str]], truth: Sequence[Sequence[str]], k: int) -> float:
    """질의마다 |찾은 상위 k ∩ 정답 상위 k| / k 의 평균.

    Args:
      found: 질의마다 ANN 이 돌려준 청크 id (순위순). k건보다 적으면 빈 자리는 틀린 것으로 센다.
      truth: 질의마다 전수 검색의 청크 id (순위순). k건 이상이어야 한다.
      k: 비교할 상위 건수.
    """
    if len(found) != len(truth):
        raise ValueError(
            f"found and truth must have the same length, got {len(found)=} {len(truth)=}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k=}")
    if not truth:
        return 0.0
    overlaps = [len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth, strict=True)]
    return float(np.mean(overlaps))


def latency_summary(latencies_ms: Sequence[float]) -> LatencySummary:
    """지연 목록의 p50 · p99 (ms, 소수 둘째 자리)."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    p50, p99 = np.percentile(np.asarray(latencies_ms), [50, 99])
    return LatencySummary(p50_ms=round(float(p50), 2), p99_ms=round(float(p99), 2))
