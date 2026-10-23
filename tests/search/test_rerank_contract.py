"""리랭커 입력 길이 계약 (실제 토크나이저).

리랭커는 (질문, 청크) 쌍을 한 입력으로 받아 청크보다 길다. MAX_LENGTH 를 넘으면 뒷부분이 조용히 잘린다.
청커 최대 크기(512토큰)의 청크 + 긴 질문이 MAX_LENGTH 안에 드는지, CrossEncoderScorer 가 세는 쌍 토큰 수가
모델 토크나이저와 같은지 확인한다. 토크나이저 파일은 HF 허브에서 받아 캐시 — 네트워크가 없으면 로컬은 skip,
CI(`CI` 환경변수)에서는 실패.
"""

import os

import pytest
from playground_corpus import Chunker, ChunkingConfig, Document, HFTokenizer, Section

from playground_retrieval.search import CrossEncoderScorer
from playground_retrieval.search.cross_encoder import MAX_LENGTH, MODEL_ID

transformers = pytest.importorskip("transformers")

LONGEST_CHUNK_TOKENS = 512  # 비교 후보 청크 크기 중 가장 큰 값
LONG_QUESTION = (
    "1592년 임진왜란 당시 조선 수군을 이끌고 한산도 대첩에서 학익진을 펼쳐 일본 수군을 크게 무찌른 "
    "인물이 그 뒤 삼도수군통제사로 임명되었는데, 그가 노량 해전에서 전사하기 전에 남긴 일기의 이름은 무엇인가?"
)
# 문단 · 문장 경계(줄바꿈 · 마침표)가 없어야 청커가 창을 512토큰까지 꽉 채운다 — 최악의 경우를 만든다
_PARAGRAPH = (
    "서울특별시는 대한민국의 수도이자 최대 도시이고 조선 시대에는 한양(漢陽)으로 불렸으며 "
    "1394년 태조 이성계가 도읍을 옮긴 뒤 600년 넘게 수도 역할을 해 왔고 "
    "2024년 기준 인구는 약 940만 명(9,386,034명)이며 1988년 하계 올림픽(Seoul Olympics)을 개최했고 "
)


class _TokenizerOnly:
    """모델 가중치 없이 CrossEncoderScorer.pair_tokens 를 쓰기 위한 껍데기."""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer

    def predict(self, sentences, *, batch_size, show_progress_bar):
        raise NotImplementedError


def _load() -> object:
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:  # 네트워크 · 허브 오류. CI 에서는 다시 던진다
        if os.environ.get("CI"):
            raise
        pytest.skip(f"{MODEL_ID} 토크나이저를 받을 수 없음: {type(e).__name__}")


def test_longest_chunk_with_long_question_fits_max_length():
    model_tokenizer = _load()
    chunker = Chunker(
        HFTokenizer.from_pretrained("BAAI/bge-m3"), ChunkingConfig(max_tokens=LONGEST_CHUNK_TOKENS)
    )
    document = Document(
        id="1",
        title="서울특별시",
        text=_PARAGRAPH * 30,
        sections=(Section(2, "역사", _PARAGRAPH * 30),),
    )
    longest = max(chunker.chunk(document), key=lambda chunk: chunk.token_count)
    scorer = CrossEncoderScorer(_TokenizerOnly(model_tokenizer))

    pair_tokens = scorer.pair_tokens(LONG_QUESTION, longest.text)

    assert longest.token_count == LONGEST_CHUNK_TOKENS
    assert pair_tokens == len(model_tokenizer(LONG_QUESTION, longest.text)["input_ids"])
    assert pair_tokens <= MAX_LENGTH
    assert model_tokenizer.model_max_length >= MAX_LENGTH
