import logging
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import CrossEncoderScorer, Reranker


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id, doc_id=chunk_id, title=chunk_id, section="", text=text, seq=0, token_count=1
    )


class FakeBase:
    """정해 둔 순서로 후보를 돌려주고, 받은 인자를 기록한다."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, Filters | None]] = []

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        self.calls.append((query, k, filters))
        return [Hit(chunk=c, score=1.0) for c in self.chunks[:k]]


class KeywordScorer:
    """본문에 질의 단어가 몇 개 들어 있는지로 0~1 점수. 받은 쌍을 기록한다."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        self.pairs.extend(pairs)
        values = [
            sum(word in passage for word in query.split()) / len(query.split())
            for query, passage in pairs
        ]
        return np.array(values, dtype=np.float32)


CHUNKS = [
    _chunk("a", "부산은 항구 도시"),
    _chunk("b", "서울은 조선의 수도"),
    _chunk("c", "서울 지하철"),
]


def test_search_reorders_candidates_by_scorer():
    hits = Reranker(FakeBase(CHUNKS), KeywordScorer()).search("서울 조선 수도", k=3)

    assert [hit.chunk.id for hit in hits] == ["b", "c", "a"]
    assert hits[0].score == pytest.approx(1.0)


def test_search_keeps_base_order_on_ties():
    hits = Reranker(FakeBase(CHUNKS), KeywordScorer()).search("없는단어", k=3)

    assert [hit.chunk.id for hit in hits] == ["a", "b", "c"]


def test_search_scores_query_with_each_candidate_text():
    scorer = KeywordScorer()

    Reranker(FakeBase(CHUNKS), scorer, candidates=2).search("서울", k=1)

    assert scorer.pairs == [("서울", "부산은 항구 도시"), ("서울", "서울은 조선의 수도")]


def test_search_asks_base_for_candidates_with_filters():
    base = FakeBase(CHUNKS)
    filters = Filters(doc_ids=frozenset({"b"}))

    Reranker(base, KeywordScorer(), candidates=20).search("서울", k=5, filters=filters)
    Reranker(base, KeywordScorer(), candidates=2).search("서울", k=5)  # k 가 더 크면 k 개를 받는다

    assert base.calls == [("서울", 20, filters), ("서울", 5, None)]


def test_search_returns_empty_without_calling_scorer():
    scorer = KeywordScorer()

    assert Reranker(FakeBase([]), scorer).search("서울", k=5) == []
    assert scorer.pairs == []


def test_reranker_rejects_bad_arguments():
    with pytest.raises(ValueError, match="candidates"):
        Reranker(FakeBase(CHUNKS), KeywordScorer(), candidates=0)
    with pytest.raises(ValueError, match="k must be"):
        Reranker(FakeBase(CHUNKS), KeywordScorer()).search("서울", k=0)


class FakeTokenizer:
    """글자 하나를 토큰 하나로 세고, 쌍에 특수 토큰 4개를 붙인다 (XLM-R 쌍과 같은 개수)."""

    def __call__(
        self, text: str, text_pair: str, *, add_special_tokens: bool
    ) -> dict[str, list[int]]:
        specials = 4 if add_special_tokens else 0
        return {"input_ids": [0] * (len(text) + len(text_pair) + specials)}


class FakeCrossEncoder:
    tokenizer = FakeTokenizer()

    def predict(self, sentences, *, batch_size, show_progress_bar):
        del batch_size, show_progress_bar  # CrossEncoder.predict 시그니처 맞춤
        return np.full(len(sentences), 0.5, dtype=np.float32)


def test_cross_encoder_scorer_warns_when_pairs_would_be_truncated(caplog: pytest.LogCaptureFixture):
    # 가짜 토크나이저는 글자 수 + 특수 토큰 4: 첫 쌍 2+5+4=11, 둘째 쌍 2+13+4=19
    scorer = CrossEncoderScorer(FakeCrossEncoder(), max_length=12)

    with caplog.at_level(logging.WARNING):
        scores = scorer.score([("질문", "짧은 본문"), ("질문", "아주 긴 본문 문장입니다")])

    assert scores.tolist() == [0.5, 0.5]
    assert "1 of 2 pairs exceed max_length=12" in caplog.text


def test_cross_encoder_scorer_handles_no_pairs():
    assert CrossEncoderScorer(FakeCrossEncoder()).score([]).shape == (0,)
