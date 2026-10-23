import xml.etree.ElementTree as ET

from playground_retrieval.eval import recall_qps_svg

SVG = "{http://www.w3.org/2000/svg}"
ROWS = [
    {"engine": "faiss", "index": "Flat", "param": "-", "recall_at_10": 1.0, "qps": 400},
    {
        "engine": "faiss",
        "index": "HNSW32",
        "param": "efSearch=16",
        "recall_at_10": 0.91,
        "qps": 26000,
    },
    {
        "engine": "faiss",
        "index": "HNSW32",
        "param": "efSearch=64",
        "recall_at_10": 0.97,
        "qps": 10000,
    },
    {
        "engine": "qdrant",
        "index": "HNSW int8",
        "param": "hnsw_ef=16",
        "recall_at_10": 0.88,
        "qps": 1400,
    },
    {
        "engine": "qdrant",
        "index": "HNSW int8",
        "param": "hnsw_ef=64",
        "recall_at_10": 0.95,
        "qps": 1200,
    },
]


def test_chart_is_valid_svg_with_one_curve_per_series():
    root = ET.fromstring(recall_qps_svg(ROWS, title="테스트"))  # noqa: S314  # 직접 만든 SVG

    curves = root.findall(f"{SVG}polyline")
    texts = [text.text for text in root.iter(f"{SVG}text")]

    assert len(curves) == 2  # Flat 은 곡선이 아니라 기준점
    assert len(curves[0].get("points", "").split()) == 2
    assert {"HNSW32", "HNSW int8", "Flat (전수 비교)", "테스트"} <= set(texts)
    assert "FAISS (CPU, 메모리)" in texts
    assert "Qdrant (Docker, gRPC)" in texts


def test_chart_puts_higher_qps_higher_on_screen():
    root = ET.fromstring(recall_qps_svg(ROWS, title="t"))  # noqa: S314  # 직접 만든 SVG
    points = root.findall(f"{SVG}polyline")[0].get("points", "").split()
    (x1, y1), (x2, y2) = (tuple(map(float, p.split(","))) for p in points)

    assert x1 < x2  # recall 이 큰 쪽이 오른쪽
    assert y1 < y2  # QPS 가 큰 쪽(efSearch=16)이 위쪽
