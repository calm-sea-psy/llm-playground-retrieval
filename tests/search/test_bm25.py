from collections.abc import Sequence

import pytest
from playground_corpus import Chunk

from playground_retrieval.core import Filters
from playground_retrieval.search import Bm25Index, Bm25Retriever, KiwiAnalyzer, bm25, bm25_index


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
        CHUNKS, KiwiAnalyzer.load(), doc_categories={"서울": frozenset({"한국 도시"})}
    )


def test_analyzer_drops_particles_and_endings():
    assert KiwiAnalyzer.load().tokens("서울특별시는 한양으로 불렸다. Paris") == [
        "서울",
        "특별시",
        "한양",
        "불리",
        "paris",
    ]


def test_analyzer_constructor_does_not_create_kiwi():
    with pytest.raises(TypeError):
        KiwiAnalyzer()


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


def test_bm25_module_reexports_index_constants():
    assert (bm25.K1, bm25.B) == (bm25_index.K1, bm25_index.B)
    assert bm25.Bm25Index is bm25_index.Bm25Index


def test_retriever_reads_text_from_store():
    analyzer = KiwiAnalyzer.load()
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
