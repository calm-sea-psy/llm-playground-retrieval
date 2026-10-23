"""ANN 벤치 결과의 recall@10 – QPS 그래프 (리포트에 넣는 정적 SVG)."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape

import numpy as np

# 엔진마다 한 칸. 계열 색은 칸 안에서 정해진 순서로 (돌려 쓰지 않는다). 검증한 기본 팔레트 1~4번
# light · dark 쌍: 파랑 · 주황 · 청록 · 노랑
_SERIES_COLORS = (
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
)
_MARKERS = ("circle", "square", "triangle", "diamond")  # 색 말고도 계열을 가르는 2차 표지
_REFERENCE = {"Flat", "exact float"}  # 전수 검색 — 곡선 대신 회색 기준점
_ENGINE_TITLES = {"faiss": "FAISS (CPU, 메모리)", "qdrant": "Qdrant (Docker, gRPC)"}
PANEL_W, PANEL_H = 400, 280
_LEFT, _RIGHT, _TOP, _BOTTOM = 56, 16, 36, 44
_LEGEND_H = 66  # 범례 세 줄까지 (계열 4개 + 기준점, 한 줄에 둘)
_LEGEND_COL = 170
_FONT = "system-ui, -apple-system, 'Segoe UI', 'Malgun Gothic', sans-serif"


@dataclass(frozen=True)
class _Point:
    recall: float
    qps: float


def recall_qps_svg(rows: Sequence[Mapping[str, float | str | int]], *, title: str) -> str:
    """엔진별 칸에 색인 계열마다 (recall@10, QPS) 곡선을 그린 SVG 문자열.

    Args:
      rows: bench_ann 결과 행. engine · index · recall_at_10 · qps 를 쓴다.
      title: 그래프 제목.

    Returns:
      두 칸이 같은 축 범위를 쓰는 SVG (x: recall@10 선형, y: QPS 로그). 다크 모드 색을 함께 담는다.
    """
    engines = list(dict.fromkeys(str(row["engine"]) for row in rows))
    recalls = [float(row["recall_at_10"]) for row in rows]
    qps = [float(row["qps"]) for row in rows]
    x_min = max(0.0, np.floor(min(recalls) * 10) / 10)
    y_lo, y_hi = np.floor(np.log10(min(qps))), np.ceil(np.log10(max(qps)))
    width, height = PANEL_W * len(engines), PANEL_H + 34 + _LEGEND_H
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{_FONT}">',
        _style(),
        f'<rect class="bg" width="{width}" height="{height}"/>',
        f'<text class="t1" x="12" y="22" font-size="15" font-weight="600">{escape(title)}</text>',
    ]
    for panel, engine in enumerate(engines):
        engine_rows = [row for row in rows if row["engine"] == engine]
        parts.append(
            _panel(engine, engine_rows, x0=panel * PANEL_W, y0=34, x_min=x_min, y_log=(y_lo, y_hi))
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _style() -> str:
    light = " ".join(f".s{i}{{stroke:{c[0]};fill:{c[0]}}}" for i, c in enumerate(_SERIES_COLORS))
    dark = " ".join(f".s{i}{{stroke:{c[1]};fill:{c[1]}}}" for i, c in enumerate(_SERIES_COLORS))
    return (
        "<style>"
        ".bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .grid{stroke:#e4e3df} "
        ".axis{stroke:#a3a29c} .ref{fill:#8a8984;stroke:#8a8984} .line{fill:none;stroke-width:2} "
        f"{light} "
        "@media (prefers-color-scheme: dark){"
        ".bg{fill:#1a1a19} .t1{fill:#f0efec} .t2{fill:#c3c2b7} .grid{stroke:#33332f} "
        f".axis{{stroke:#5c5b55}} .ref{{fill:#9b9a93;stroke:#9b9a93}} {dark}}}"
        "</style>"
    )


def _panel(
    engine: str,
    rows: Sequence[Mapping[str, float | str | int]],
    *,
    x0: float,
    y0: float,
    x_min: float,
    y_log: tuple[float, float],
) -> str:
    plot_w = PANEL_W - _LEFT - _RIGHT
    plot_h = PANEL_H - _TOP - _BOTTOM

    def sx(recall: float) -> float:
        return x0 + _LEFT + (recall - x_min) / (1.0 - x_min) * plot_w

    def sy(qps: float) -> float:
        return y0 + _TOP + (1 - (np.log10(qps) - y_log[0]) / (y_log[1] - y_log[0])) * plot_h

    title = _ENGINE_TITLES.get(engine, engine)
    out = [
        f'<text class="t1" x="{x0 + _LEFT}" y="{y0 + 18}" font-size="13" font-weight="600">{escape(title)}</text>'
    ]
    out.extend(_axes(sx, sy, x_min, y_log, x0=x0, y0=y0, plot_w=plot_w, plot_h=plot_h))
    series: dict[str, list[_Point]] = {}
    for row in rows:
        point = _Point(float(row["recall_at_10"]), float(row["qps"]))
        series.setdefault(str(row["index"]), []).append(point)
    legend: list[tuple[str, str, str]] = []  # (모양, css, 이름)
    slot = 0
    for name, points in series.items():
        if name in _REFERENCE:
            point = points[0]
            out.append(_marker("circle", sx(point.recall), sy(point.qps), "ref"))
            legend.append(("circle", "ref", f"{name} (전수 비교)"))
            continue
        css = f"s{slot}"
        ordered = sorted(points, key=lambda p: p.recall)
        path = " ".join(f"{sx(p.recall):.1f},{sy(p.qps):.1f}" for p in ordered)
        out.append(f'<polyline class="line {css}" style="fill:none" points="{path}"/>')
        out.extend(_marker(_MARKERS[slot], sx(p.recall), sy(p.qps), css) for p in ordered)
        legend.append((_MARKERS[slot], css, name))
        slot += 1
    # 곡선이 recall 1.0 근처에 몰려 이름표가 겹치므로 범례로 (계열 식별은 색 + 모양)
    legend_y = y0 + PANEL_H + 4
    for i, (shape, css, name) in enumerate(legend):
        lx = x0 + _LEFT + (i % 2) * _LEGEND_COL
        ly = legend_y + (i // 2) * 18
        out.append(_marker(shape, lx + 5, ly, css))
        out.append(
            f'<text class="t1" x="{lx + 16}" y="{ly + 4}" font-size="11">{escape(name)}</text>'
        )
    return "\n".join(out)


def _axes(
    sx: Callable[[float], float],
    sy: Callable[[float], float],
    x_min: float,
    y_log: tuple[float, float],
    *,
    x0: float,
    y0: float,
    plot_w: float,
    plot_h: float,
) -> list[str]:
    out = []
    bottom = y0 + _TOP + plot_h
    for tick in np.arange(x_min, 1.0001, 0.1):
        x = sx(float(tick))
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{y0 + _TOP}" x2="{x:.1f}" y2="{bottom}"/>')
        out.append(
            f'<text class="t2" x="{x:.1f}" y="{bottom + 16}" font-size="11" text-anchor="middle">{tick:.1f}</text>'
        )
    for power in range(int(y_log[0]), int(y_log[1]) + 1):
        y = sy(10.0**power)
        out.append(
            f'<line class="grid" x1="{x0 + _LEFT}" y1="{y:.1f}" x2="{x0 + _LEFT + plot_w}" y2="{y:.1f}"/>'
        )
        label = f"{10**power:,}"
        out.append(
            f'<text class="t2" x="{x0 + _LEFT - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{label}</text>'
        )
    out.append(
        f'<line class="axis" x1="{x0 + _LEFT}" y1="{bottom}" x2="{x0 + _LEFT + plot_w}" y2="{bottom}"/>'
    )
    out.append(
        f'<text class="t2" x="{x0 + _LEFT + plot_w / 2:.1f}" y="{bottom + 34}" font-size="11" '
        'text-anchor="middle">recall@10 (FAISS Flat 대비)</text>'
    )
    out.append(
        f'<text class="t2" font-size="11" text-anchor="middle" '
        f'transform="translate({x0 + 14},{y0 + _TOP + plot_h / 2:.1f}) rotate(-90)">QPS (로그)</text>'
    )
    return out


def _marker(shape: str, x: float, y: float, css: str) -> str:
    r = 4.5  # 지름 9px — 8px 이상
    if shape == "square":
        return f'<rect class="{css}" x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1"/>'
    if shape == "triangle":
        return f'<polygon class="{css}" points="{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y + r:.1f} {x - r - 1:.1f},{y + r:.1f}"/>'
    if shape == "diamond":
        return f'<polygon class="{css}" points="{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y:.1f} {x:.1f},{y + r + 1:.1f} {x - r - 1:.1f},{y:.1f}"/>'
    return f'<circle class="{css}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
