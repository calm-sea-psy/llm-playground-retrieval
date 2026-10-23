"""여러 검색기 결과를 순위로 합치는 하이브리드 검색기 (Reciprocal Rank Fusion)."""

from collections.abc import Sequence

from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit, Retriever

RRF_K = 60  # RRF 원 논문(Cormack et al., 2009) 기본값. 상위 순위 차이를 완만하게 만든다
CANDIDATES = (
    50  # 검색기마다 가져오는 후보 수. 위키 서브셋 평가(D12)에서 쓴 값 — 제품이 설정으로 바꿔 넘긴다
)


class HybridRetriever:
    """검색기마다 후보를 받아 RRF 점수 Σ 1 / (rrf_k + 순위) 로 합친다.

    점수 척도가 다른 검색기(BM25 원점수 · 코사인)를 순위만으로 합치므로 정규화가 필요 없다.
    같은 청크가 여러 검색기에 나오면 점수를 더한다.

    예)
        hybrid = HybridRetriever([dense, bm25])
        hits = hybrid.search("조선의 수도는?", k=20)
    """

    def __init__(
        self, retrievers: Sequence[Retriever], *, rrf_k: int = RRF_K, candidates: int = CANDIDATES
    ) -> None:
        if not retrievers:
            raise ValueError("retrievers must not be empty")
        self._retrievers = list(retrievers)
        self._rrf_k = rrf_k
        self._candidates = candidates

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """합친 점수 상위 k건을 점수 내림차순으로 돌려준다. 점수가 같으면 먼저 나온 검색기 순서."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k=}")
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            for rank, hit in enumerate(retriever.search(query, self._candidates, filters), start=1):
                chunks.setdefault(hit.chunk.id, hit.chunk)
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (self._rrf_k + rank)
        # sorted 는 안정 정렬이라 동점이면 처음 본 순서(검색기 순서 · 순위)가 유지된다
        ordered = sorted(scores, key=lambda chunk_id: -scores[chunk_id])
        return [Hit(chunk=chunks[chunk_id], score=scores[chunk_id]) for chunk_id in ordered[:k]]
