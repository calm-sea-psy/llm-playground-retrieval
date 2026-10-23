"""Qdrant 벡터 저장소: 청크 벡터 + 필터용 payload. 원문은 PostgreSQL 에 둔다."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from playground_corpus import Chunk
from qdrant_client import QdrantClient, models

from playground_retrieval.core import VectorHit

URL = "http://localhost:6333"
UPSERT_BATCH = 512  # 요청 하나에 1024d 벡터 512개 ≈ 2MB
_POINT_NAMESPACE = uuid.UUID(
    "6f3c7a52-0d1e-4c3b-9a55-2b8f1e7c4d90"
)  # 청크 id → 점 id (uuid5) 네임스페이스


@dataclass(frozen=True)
class QdrantConfig:
    """접속 설정. 기본값은 docker-compose.yml 의 로컬 Qdrant."""

    url: str = URL
    api_key: str | None = field(default=None, repr=False)
    """Qdrant Cloud 키. 로컬은 None. 로그에 찍히지 않게 repr 에서 뺀다."""
    prefer_grpc: bool = True
    """REST 는 질의 하나에 약 16ms 가 더 든다 (서버 검색 약 1.5ms, Windows + Docker Desktop). gRPC 는 약 1.8ms."""
    timeout: int | None = None
    """요청 시간 제한(초). None 이면 클라이언트 기본 5초."""

    @classmethod
    def from_env(cls) -> QdrantConfig:
        """QDRANT_URL · QDRANT_API_KEY · QDRANT_GRPC(0 이면 REST) · QDRANT_TIMEOUT 에서 읽는다.

        값은 `uv run --env-file .env.cloud ...` 처럼 파일로 넘길 수 있다 (.env* 는 git 제외).
        """
        timeout = os.environ.get("QDRANT_TIMEOUT")
        return cls(
            url=os.environ.get("QDRANT_URL") or URL,
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            prefer_grpc=os.environ.get("QDRANT_GRPC", "1") != "0",
            timeout=int(timeout) if timeout else None,
        )


@dataclass(frozen=True)
class IndexStatus:
    """컬렉션 색인 상태."""

    status: str
    """green(완료) · yellow(최적화 중) · grey · red."""
    points: int
    indexed: int
    """HNSW 색인에 들어간 벡터 수. points 보다 적으면 나머지는 전수 비교로 찾는다."""


def point_id(chunk_id: str) -> str:
    """청크 id 를 Qdrant 점 id(uuid)로. 같은 청크는 항상 같은 점이 되어 다시 넣으면 덮어쓴다."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def chunk_payload(
    chunk: Chunk, categories: Sequence[str], dump_date: str | None
) -> dict[str, object]:
    """필터 · 결과 표시에 쓰는 payload. 본문은 넣지 않는다."""
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "seq": chunk.seq,
        "categories": list(categories),
        "dump_date": dump_date,
    }


class QdrantStore:
    """컬렉션을 만들고 청크 벡터를 넣고 찾는 저장소.

    예)
        store = QdrantStore.from_env()
        store.ensure_collection("kowiki_bge-m3", dim=1024)
        store.upsert("kowiki_bge-m3", chunks, vectors, categories_by_doc, dump_date="20261001")
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    @classmethod
    def connect(cls, config: QdrantConfig) -> QdrantStore:
        """설정대로 클라이언트를 만든다. 실제 접속은 첫 요청 때 일어난다."""
        return cls(
            QdrantClient(
                url=config.url,
                api_key=config.api_key,
                prefer_grpc=config.prefer_grpc,
                timeout=config.timeout,
                check_compatibility=False,
            )
        )

    @classmethod
    def from_env(cls) -> QdrantStore:
        """환경변수 설정(QdrantConfig.from_env)으로 접속한다."""
        return cls.connect(QdrantConfig.from_env())

    def ensure_collection(self, name: str, dim: int, *, on_disk: bool = True) -> None:
        """컬렉션이 없으면 만든다: 코사인 거리, int8 스칼라 양자화(RAM), 원본 벡터는 on_disk."""
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE, on_disk=on_disk
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ),
        )
        for payload_field, schema in (
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("categories", models.PayloadSchemaType.KEYWORD),
        ):
            self._client.create_payload_index(name, field_name=payload_field, field_schema=schema)

    def delete_collection(self, name: str) -> None:
        """컬렉션을 지운다 (없으면 아무 일도 하지 않음)."""
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def upsert(
        self,
        name: str,
        chunks: Sequence[Chunk],
        vectors: NDArray[np.float32],
        categories_by_doc: Mapping[str, Sequence[str]],
        *,
        dump_date: str | None = None,
    ) -> int:
        """청크 벡터를 넣고(같은 청크면 덮어씀) 넣은 수를 돌려준다.

        Args:
          name: 컬렉션 이름.
          chunks: 청크 n개.
          vectors: (n, d). chunks 와 같은 순서.
          categories_by_doc: 문서 id → 분류. payload 필터용.
          dump_date: 덤프 날짜. payload 에 그대로.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length, got {len(chunks)=} {len(vectors)=}"
            )
        for start in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[start : start + UPSERT_BATCH]
            points = [
                models.PointStruct(
                    id=point_id(chunk.id),
                    vector=vectors[start + offset].tolist(),
                    payload=chunk_payload(
                        chunk, categories_by_doc.get(chunk.doc_id, ()), dump_date
                    ),
                )
                for offset, chunk in enumerate(batch)
            ]
            self._client.upsert(collection_name=name, points=points, wait=True)
        return len(chunks)

    def search(
        self,
        name: str,
        vector: NDArray[np.float32],
        k: int,
        *,
        doc_ids: Sequence[str] = (),
        categories: Sequence[str] = (),
        params: models.SearchParams | None = None,
    ) -> list[VectorHit]:
        """벡터와 가까운 청크 k건을 유사도 내림차순으로 돌려준다. doc_ids · categories 가 있으면 그 안에서만.

        Args:
          name: 컬렉션 이름.
          vector: 질의 벡터 (d,).
          k: 돌려받을 건수.
          doc_ids: 이 문서들 안에서만.
          categories: 이 분류 중 하나에 속한 문서에서만.
          params: hnsw_ef · exact · 양자화 사용 여부. 없으면 컬렉션 기본값 (ANN 벤치가 바꿔 본다).
        """
        conditions = []
        if doc_ids:
            conditions.append(
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
            )
        if categories:
            conditions.append(
                models.FieldCondition(key="categories", match=models.MatchAny(any=list(categories)))
            )
        response = self._client.query_points(
            collection_name=name,
            query=vector.tolist(),
            limit=k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            search_params=params,
            with_payload=["chunk_id", "doc_id"],
        )
        return [_vector_hit(point) for point in response.points]

    def search_batch(
        self,
        name: str,
        vectors: NDArray[np.float32],
        k: int,
        *,
        params: models.SearchParams | None = None,
    ) -> list[list[VectorHit]]:
        """질의 벡터 (n, d) 마다 가까운 청크 k건. 요청 하나로 보낸다 (ANN 벤치의 처리량 측정용)."""
        requests = [
            models.QueryRequest(
                query=vector.tolist(), limit=k, params=params, with_payload=["chunk_id", "doc_id"]
            )
            for vector in vectors
        ]
        responses = self._client.query_batch_points(collection_name=name, requests=requests)
        return [[_vector_hit(point) for point in response.points] for response in responses]

    def count(self, name: str) -> int:
        """컬렉션의 점 개수."""
        return self._client.count(collection_name=name, exact=True).count

    def index_status(self, name: str) -> IndexStatus:
        """컬렉션 상태와 HNSW 색인에 들어간 벡터 수. 적재 직후에는 색인을 뒤에서 만든다."""
        info = self._client.get_collection(name)
        return IndexStatus(
            status=str(info.status.value),
            points=info.points_count or 0,
            indexed=info.indexed_vectors_count or 0,
        )


def _vector_hit(point: models.ScoredPoint) -> VectorHit:
    return VectorHit(
        chunk_id=point.payload["chunk_id"], doc_id=point.payload["doc_id"], score=point.score
    )
