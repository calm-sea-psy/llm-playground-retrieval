import math
from pathlib import Path

import numpy as np
from playground_corpus import Chunk

from playground_retrieval.search import Bm25Index, KiwiAnalyzer
from playground_retrieval.search.bm25_index import K1, META_FILE, B


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}-0", doc_id=doc_id, title=doc_id, section="", text=text, seq=0, token_count=0
    )


CHUNKS = [
    _chunk("서울", "서울특별시는 대한민국의 수도이다. 조선 시대에는 한양으로 불렸다."),
    _chunk("부산", "부산광역시는 대한민국 제2의 도시이며 항구 도시이다."),
    _chunk("파리", "파리는 프랑스의 수도이다. Paris 는 센강을 끼고 있다."),
]


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
    analyzer = KiwiAnalyzer.load()
    index = Bm25Index.build(CHUNKS, analyzer, batch_size=2)  # 배치 경계를 넘겨도 같아야 한다
    query = analyzer.tokens("대한민국의 수도와 항구 도시 수도")

    expected = _reference_scores(analyzer.tokens_batch([c.text for c in CHUNKS]), query)

    np.testing.assert_allclose(index.scores(query), expected, rtol=1e-5)


def test_index_round_trips_through_disk(tmp_path: Path):
    analyzer = KiwiAnalyzer.load()
    index = Bm25Index.build(CHUNKS, analyzer)
    index.save(tmp_path / "bm25", meta={"collection": "test"})
    query = analyzer.tokens("프랑스 수도 한양")

    loaded = Bm25Index.load(tmp_path / "bm25")

    assert (tmp_path / "bm25" / META_FILE).exists()
    assert loaded.chunk_ids == index.chunk_ids
    assert loaded.doc_ids.tolist() == index.doc_ids.tolist()
    np.testing.assert_array_equal(loaded.scores(query), index.scores(query))
