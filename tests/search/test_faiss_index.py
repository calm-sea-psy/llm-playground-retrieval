import numpy as np
import pytest

from playground_retrieval.search import FaissIndex, faiss_index

RNG = np.random.default_rng(0)
VECTORS = RNG.standard_normal((400, 16)).astype(np.float32)
CHUNK_IDS = [f"{i // 4}-{i % 4}" for i in range(400)]  # 문서마다 청크 4개
DOC_IDS = [str(i // 4) for i in range(400)]
CATEGORIES = {"0": frozenset({"도시"}), "1": frozenset({"도시", "수도"})}


def _exact_top(query: np.ndarray, k: int) -> list[str]:
    normalized = VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True)
    order = np.argsort(-(normalized @ (query / np.linalg.norm(query))))[:k]
    return [CHUNK_IDS[i] for i in order]


@pytest.mark.parametrize("factory", ["Flat", "HNSW16", "IVF4,Flat"])
def test_search_matches_exact_cosine_top(factory: str):
    index = FaissIndex.build(VECTORS, CHUNK_IDS, DOC_IDS, collection="c", factory=factory)
    if "IVF" in factory:
        index.index.nprobe = 4  # 목록을 전부 훑게 해서 정확히
    query = VECTORS[7] * 3.0  # 크기가 달라도 코사인으로 찾는다

    hits = index.search("c", query, k=5)

    assert [hit.chunk_id for hit in hits] == _exact_top(query, 5)
    assert hits[0].chunk_id == "1-3"
    assert hits[0].doc_id == "1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("factory", ["Flat", "HNSW16", "IVF4,Flat"])
def test_search_filters_by_doc_ids_and_categories(factory: str):
    index = FaissIndex.build(
        VECTORS, CHUNK_IDS, DOC_IDS, collection="c", factory=factory, doc_categories=CATEGORIES
    )

    by_doc = index.search("c", VECTORS[7], k=10, doc_ids=["5", "9"])
    by_category = index.search("c", VECTORS[7], k=10, categories=["수도"])
    both = index.search("c", VECTORS[7], k=10, doc_ids=["0", "1"], categories=["도시"])

    assert len(by_doc) == 8
    assert {hit.doc_id for hit in by_doc} == {"5", "9"}
    assert {hit.doc_id for hit in by_category} == {"1"}
    assert {hit.doc_id for hit in both} == {"0", "1"}


def test_wide_filter_uses_index_with_selector(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(faiss_index, "EXACT_FILTER_ROWS", 0)
    index = FaissIndex.build(VECTORS, CHUNK_IDS, DOC_IDS, collection="c")

    hits = index.search("c", VECTORS[7], k=10, doc_ids=["5", "9"])

    assert len(hits) == 8
    assert {hit.doc_id for hit in hits} == {"5", "9"}


def test_search_returns_empty_when_filter_matches_nothing():
    index = FaissIndex.build(VECTORS, CHUNK_IDS, DOC_IDS, collection="c")

    assert index.search("c", VECTORS[0], k=3, doc_ids=["없음"]) == []


def test_search_rejects_other_collection():
    index = FaissIndex.build(VECTORS, CHUNK_IDS, DOC_IDS, collection="c")

    with pytest.raises(ValueError, match="holds"):
        index.search("other", VECTORS[0], k=3)


def test_build_does_not_modify_input_vectors():
    vectors = VECTORS.copy()

    FaissIndex.build(vectors, CHUNK_IDS, DOC_IDS, collection="c")

    np.testing.assert_array_equal(vectors, VECTORS)
