import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from playground_corpus import Chunk

from playground_retrieval.core import Filters
from playground_retrieval.search import Bm25Index, Bm25Retriever, KiwiAnalyzer
from playground_retrieval.search.bm25 import K1, B


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}-0", doc_id=doc_id, title=doc_id, section="", text=text, seq=0, token_count=0
    )


CHUNKS = [
    _chunk("서울", "서울특별시는 대한민국의 수도이다. 조선 시대에는 한양으로 불렸다."),
    _chunk("부산", "부산광역시는 대한민국 제2의 도시이며 항구 도시이다."),
    _chunk("파리", "파리는 프랑스의 수도이다. Paris 는 센강을 끼고 있다."),
]


@pytest.fixture(scope="module")
def retriever() -> Bm25Retriever:
    return Bm25Retriever.from_chunks(
        CHUNKS, KiwiAnalyzer(), doc_categories={"서울": frozenset({"한국 도시"})}
    )


def test_analyzer_drops_particles_and_endings():
    assert KiwiAnalyzer().tokens("서울특별시는 한양으로 불렸다. Paris") == [
        "서울",
        "특별시",
        "한양",
        "불리",
        "paris",
    ]


def test_search_ranks_matching_chunk_first(retriever: Bm25Retriever):
    hits = retriever.search("조선 시대 한양은 어디인가?", k=2)

    assert [hit.chunk.doc_id for hit in hits] == ["서울"]  # 겹치는 단어가 없는 청크는 빠진다
    assert hits[0].score > 0


def test_search_matches_despite_different_particles(retriever: Bm25Retriever):
    hits = retriever.search("프랑스에서 수도가 어디야", k=3)

    assert hits[0].chunk.doc_id == "파리"
    assert {hit.chunk.doc_id for hit in hits} == {"파리", "서울"}  # "수도" 가 겹치는 서울도 뒤에


def test_search_applies_filters(retriever: Bm25Retriever):
    by_doc = retriever.search("수도", k=3, filters=Filters(doc_ids=frozenset({"파리"})))
    by_category = retriever.search(
        "수도", k=3, filters=Filters(categories=frozenset({"한국 도시"}))
    )

    assert [hit.chunk.doc_id for hit in by_doc] == ["파리"]
    assert [hit.chunk.doc_id for hit in by_category] == ["서울"]


def test_search_returns_empty_when_no_terms_match(retriever: Bm25Retriever):
    assert retriever.search("에베레스트", k=3) == []


def test_search_rejects_non_positive_k(retriever: Bm25Retriever):
    with pytest.raises(ValueError, match="k must be"):
        retriever.search("수도", k=0)


def _reference_scores(tokenized: list[list[str]], query: list[str]) -> list[float]:
    """BM25 를 정의 그대로 계산한다 (Lucene idf, K1 · B)."""
    total = len(tokenized)
    average = sum(len(tokens) for tokens in tokenized) / total
    scores = []
    for tokens in tokenized:
        score = 0.0
        for term in query:
            df = sum(term in other for other in tokenized)
            tf = tokens.count(term)
            if not df or not tf:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(tokens) / average))
        scores.append(score)
    return scores


def test_index_scores_match_bm25_definition():
    analyzer = KiwiAnalyzer()
    index = Bm25Index.build(CHUNKS, analyzer, batch_size=2)  # 배치 경계를 넘겨도 같아야 한다
    query = analyzer.tokens("대한민국의 수도와 항구 도시 수도")

    expected = _reference_scores(analyzer.tokens_batch([c.text for c in CHUNKS]), query)

    np.testing.assert_allclose(index.scores(query), expected, rtol=1e-5)


def test_index_round_trips_through_disk(tmp_path: Path):
    analyzer = KiwiAnalyzer()
    index = Bm25Index.build(CHUNKS, analyzer)
    index.save(tmp_path / "bm25", meta={"collection": "test"})
    query = analyzer.tokens("프랑스 수도 한양")

    loaded = Bm25Index.load(tmp_path / "bm25")

    assert loaded.chunk_ids == index.chunk_ids
    assert loaded.doc_ids.tolist() == index.doc_ids.tolist()
    np.testing.assert_array_equal(loaded.scores(query), index.scores(query))


def test_retriever_reads_text_from_store():
    analyzer = KiwiAnalyzer()
    store = RecordingStore(CHUNKS)
    retriever = Bm25Retriever(
        Bm25Index.build(CHUNKS, analyzer), analyzer, store, collection="c", source="kowiki"
    )

    hits = retriever.search("한양", k=3)

    assert [hit.chunk for hit in hits] == [CHUNKS[0]]
    assert store.calls == [("c", "kowiki", ["서울-0"])]


class RecordingStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.calls: list[tuple[str, str, list[str]]] = []

    def get_chunks(self, collection: str, source: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        self.calls.append((collection, source, list(chunk_ids)))
        return [self.by_id[chunk_id] for chunk_id in chunk_ids]
