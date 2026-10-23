"""sentence-transformers CrossEncoder 로 (질의, 본문) 쌍 점수를 내는 PairScorer."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-reranker-v2-m3"
# 쌍 하나의 최대 토큰 수 (특수 토큰 포함). 모델은 8192 까지 받지만 청크 ≤ 512 + 질문 수십 토큰이라
# 1024 면 잘리지 않고, 길이를 묶어 둬야 긴 입력 하나가 배치 전체를 느리게 하지 않는다
MAX_LENGTH = 1024
BATCH_SIZE = 32


class PairTokenizer(Protocol):
    """쌍의 토큰 수를 세는 데 쓰는 토크나이저 (transformers 토크나이저)."""

    def __call__(
        self, text: str, text_pair: str, *, add_special_tokens: bool
    ) -> dict[str, list[int]]:
        """쌍을 인코딩한다."""
        ...


class CrossEncoderModel(Protocol):
    """CrossEncoderScorer 가 부르는 모델 계약 (sentence_transformers.CrossEncoder)."""

    tokenizer: PairTokenizer

    def predict(
        self, sentences: list[tuple[str, str]], *, batch_size: int, show_progress_bar: bool
    ) -> NDArray[np.float32]:
        """쌍마다 점수 (시그모이드 적용, 0~1)."""
        ...


class CrossEncoderScorer:
    """cross-encoder 리랭커 모델. 잘리는 쌍이 있으면 경고를 남긴다 (조용히 뒷부분이 사라지지 않게).

    예)
        scorer = CrossEncoderScorer.load()
        scores = scorer.score([(question, chunk.text) for chunk in chunks])
    """

    def __init__(
        self,
        model: CrossEncoderModel,
        *,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._model = model
        self.max_length = max_length
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        model_id: str = MODEL_ID,
        *,
        device: str | None = None,
        fp16: bool = True,
        max_length: int = MAX_LENGTH,
    ) -> CrossEncoderScorer:
        """모델을 내려받아(캐시) 올린다. GPU 가 있으면 GPU, fp16 이면 반정밀도로."""
        import torch  # noqa: PLC0415  # 무거운 런타임 — 리랭커를 쓸 때만 올린다
        from sentence_transformers import CrossEncoder  # noqa: PLC0415  # 같은 이유

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        model = CrossEncoder(
            model_id, device=device, max_length=max_length, model_kwargs={"torch_dtype": dtype}
        )
        return cls(model, max_length=max_length)

    def pair_tokens(self, query: str, passage: str) -> int:
        """쌍 하나를 모델에 넣을 때의 토큰 수 (특수 토큰 포함)."""
        return len(self._model.tokenizer(query, passage, add_special_tokens=True)["input_ids"])

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        """쌍 n개의 관련도 (n,). 0~1."""
        if not pairs:
            return np.empty(0, dtype=np.float32)
        too_long = [
            i
            for i, (query, passage) in enumerate(pairs)
            if self.pair_tokens(query, passage) > self.max_length
        ]
        if too_long:
            logger.warning(
                "%d of %d pairs exceed max_length=%d and will be truncated",
                len(too_long),
                len(pairs),
                self.max_length,
            )
        scores = self._model.predict(
            list(pairs), batch_size=self._batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float32)
