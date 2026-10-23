import pytest
from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import HybridRetriever


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=chunk_id[0],
        title=chunk_id,
        section="",
        text=chunk_id,
        seq=0,
        token_count=1,
    )


class FakeRetriever:
    """정해 둔 순서로 청크를 돌려주고, 받은 인자를 기록한다."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.calls: list[tuple[str, int, Filters | None]] = []

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        self.calls.append((query, k, filters))
        return [Hit(chunk=_chunk(c), score=100.0 - i) for i, c in enumerate(self.chunk_ids[:k])]


def test_search_fuses_by_reciprocal_rank():
    dense = FakeRetriever(["a", "b", "c"])
    bm25 = FakeRetriever(["b", "d"])

    hits = HybridRetriever([dense, bm25], rrf_k=60).search("질문", k=4)

    # b: 1/62 + 1/61 · a: 1/61 · d: 1/62 · c: 1/63
    assert [hit.chunk.id for hit in hits] == ["b", "a", "d", "c"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_search_breaks_ties_by_first_seen_order():
    hits = HybridRetriever([FakeRetriever(["a"]), FakeRetriever(["b"])]).search("질문", k=2)

    assert [hit.chunk.id for hit in hits] == ["a", "b"]


def test_search_asks_each_retriever_for_candidates_with_filters():
    dense, bm25 = FakeRetriever(["a"]), FakeRetriever(["b"])
    filters = Filters(doc_ids=frozenset({"a"}))

    HybridRetriever([dense, bm25], candidates=30).search("질문", k=5, filters=filters)

    assert dense.calls == bm25.calls == [("질문", 30, filters)]


def test_search_limits_to_k():
    hits = HybridRetriever([FakeRetriever(["a", "b", "c"])]).search("질문", k=2)

    assert len(hits) == 2


def test_hybrid_rejects_bad_arguments():
    with pytest.raises(ValueError, match="retrievers"):
        HybridRetriever([])
    with pytest.raises(ValueError, match="k must be"):
        HybridRetriever([FakeRetriever(["a"])]).search("질문", k=0)
