# llm-playground-retrieval

llm-playground 제품들이 함께 쓰는 검색 엔진 모듈입니다. 패키지 이름은 `playground_retrieval` 입니다.
한국어 위키 QA([llm-playground-wiki](https://github.com/calm-sea-psy/llm-playground-wiki))에서 만들고 평가한 검색 부분을 떼어 냈습니다.

| 하위 패키지 | 내용 |
|---|---|
| `core` | 검색 결과 · 범위 자료형(`Hit` · `Filters` · `VectorHit`), 계약(`Retriever` · `Source`) |
| `store` | PostgreSQL(문서 · 청크 원문 · 실행 기록) · Qdrant(청크 벡터, int8 양자화) |
| `search` | BM25(Kiwi 형태소, 색인 저장 · 불러오기) · Dense · Hybrid(RRF) · Reranker(bge-reranker-v2-m3) · FAISS 색인 · `SearchService` |
| `ingest` | `ingest_documents` — 문서 → 청킹 → 임베딩(npy 캐시) → PG · Qdrant → 실행 기록 |
| `eval` | ANN 벤치 지표(recall@k · 지연 분위수) · recall–QPS 그래프 |

문서 · 청크 타입, 청커, 임베딩은 [llm-playground-corpus](https://github.com/calm-sea-psy/llm-playground-corpus) 를 씁니다.
검색기는 저장소를 Protocol 로 받으므로 Qdrant 대신 FAISS 색인, PG 대신 메모리 저장소를 끼울 수 있습니다.

## 설치

```bash
# 기본 — 검색기 · 저장소 · 인제스트 흐름
uv add "playground-retrieval @ git+https://github.com/calm-sea-psy/llm-playground-retrieval@v0.1.0"

# 리랭커 · 임베딩 모델 실행 (torch, sentence-transformers)
uv add "playground-retrieval[rerank] @ git+https://github.com/calm-sea-psy/llm-playground-retrieval@v0.1.0"
```

0.x 동안은 깨는 변경이 있을 수 있습니다. 이 레포와 함께 고칠 때는 같은 부모 폴더에 나란히 clone 하고 잠깐 path 로 바꿔 씁니다.

## 사용 예

```python
from playground_retrieval.search import (
    Bm25Index,
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    KiwiAnalyzer,
    Reranker,
)
from playground_retrieval.store import PgConfig, PgStore, QdrantStore

with PgStore.connect(PgConfig.from_env()) as pg:
    dense = DenseRetriever(
        embedder, QdrantStore.from_env(), pg, collection="kowiki_kure-v1", source="kowiki"
    )
    bm25 = Bm25Retriever(
        Bm25Index.load(index_dir), KiwiAnalyzer(), pg, collection="kowiki_kure-v1", source="kowiki"
    )
    hits = Reranker(HybridRetriever([dense, bm25]), scorer).search("조선의 수도는?", k=5)
```

`QdrantStore.from_env()` 는 기본으로 로컬 Qdrant(`http://localhost:6333`)에 붙는다. Qdrant Cloud 는 `QDRANT_MODE=cloud` 일 때만 `QDRANT_URL` · `QDRANT_API_KEY` 를 쓴다 — 주소 · 키가 환경에 있어도 모드를 켜지 않으면 로컬이다. `QDRANT_GRPC=0`(REST) · `QDRANT_TIMEOUT`(초)은 모드와 상관없이 읽는다.

## 개발

```bash
uv sync --all-extras
uv run pytest        # PG · Qdrant 연동 테스트는 서비스가 없으면 건너뜀
uv run ruff check
```

연동 테스트용 PostgreSQL · Qdrant 는 llm-playground-wiki 의 `docker-compose.yml` 과 같은 이미지 · 계정(localhost:5432 · 6333)을 씁니다.
