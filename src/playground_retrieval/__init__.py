"""검색 엔진 모듈. 하위 패키지에서 가져온다.

core    검색 결과 · 범위 자료형(Hit · Filters · VectorHit), 검색기 · 문서 소스 계약(Retriever · Source)
store   PostgreSQL(문서 · 청크 원문) · Qdrant(벡터) 저장소
search  BM25(Kiwi) · Dense · Hybrid(RRF) · Reranker · FAISS 색인 · SearchService
ingest  문서 → 청킹 → 임베딩 → 저장 흐름
eval    ANN 벤치 지표 · recall–QPS 그래프
"""
