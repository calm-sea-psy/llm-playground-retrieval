import asyncio
import time

import pytest
from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import (
    KiwiAnalyzer,
    SearchService,
    SearchSettings,
    service as service_module,
)
from playground_retrieval.search.service import TOP_K

CHUNK = Chunk(
    id="1-0", doc_id="1", title="서울", section="", text="서울\n수도", seq=0, token_count=3
)


class RecordingRetriever:
    """받은 인자를 기록하고 k건을 돌려준다. 동시에 몇 개가 돌았는지도 센다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, Filters | None]] = []
        self.running = 0
        self.max_running = 0

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.calls.append((query, k, filters))
        time.sleep(0.01)  # 잠금이 없으면 이 사이에 다른 스레드가 들어온다
        self.running -= 1
        return [Hit(chunk=CHUNK, score=0.9)] * k


def _service(retriever: RecordingRetriever) -> SearchService:
    return SearchService(retriever, collection="test", chunk_count=1)


def test_search_uses_default_top_k_when_k_missing():
    retriever = RecordingRetriever()

    result = asyncio.run(_service(retriever).search("서울"))

    assert retriever.calls == [("서울", TOP_K, None)]
    assert len(result.hits) == TOP_K
    assert result.took_ms >= 0


def test_search_passes_k_and_filters():
    retriever = RecordingRetriever()
    filters = Filters(doc_ids=frozenset({"1"}))

    result = asyncio.run(_service(retriever).search("서울", 2, filters))

    assert retriever.calls == [("서울", 2, filters)]
    assert result.hits == (Hit(chunk=CHUNK, score=0.9),) * 2


def test_search_runs_one_query_at_a_time():
    retriever = RecordingRetriever()
    service = _service(retriever)

    async def run_many() -> None:
        await asyncio.gather(*(service.search(f"q{i}", 1) for i in range(5)))

    asyncio.run(run_many())

    assert len(retriever.calls) == 5
    assert retriever.max_running == 1


class _StopBuildError(Exception):
    """build 가 BM25 색인을 준비하는 데까지 왔는지 보려고 거기서 멈춘다 (모델 · 저장소 없이)."""


def test_build_uses_the_given_analyzer(monkeypatch: pytest.MonkeyPatch):
    given = KiwiAnalyzer(kiwi=object())  # Kiwi 를 올리지 않는다 — 넘긴 것이 그대로 쓰이는지만 본다
    seen = []

    def fake_index(settings, pg, analyzer):
        seen.append(analyzer)
        raise _StopBuildError

    monkeypatch.setattr(service_module, "_bm25_index", fake_index)
    settings = SearchSettings(collection="c", source="s", model="kure-v1")

    with pytest.raises(_StopBuildError):
        SearchService.build(settings, pg=None, qdrant=None, analyzer=given)

    assert seen == [given]
