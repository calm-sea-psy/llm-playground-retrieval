"""하위 패키지 사이 의존 방향 검사. 모듈은 제품 레포를 모르고, 검색기는 저장소 구현을 직접 import 하지 않는다.

core ← store · search · eval
core · store ← ingest
"""

import ast
from pathlib import Path

import pytest

PACKAGE = "playground_retrieval"
SRC = Path(__file__).parents[1] / "src" / PACKAGE

# 하위 패키지 → 가져와도 되는 다른 하위 패키지
ALLOWED_INTERNAL = {
    "core": set(),
    "store": {"core"},
    "search": {"core"},  # 저장소는 Protocol 로 받는다 — store 구현을 직접 가져오지 않는다
    "eval": {"core"},
    "ingest": {"core", "store"},
}
# 모듈 어디서도 쓰면 안 되는 것: 제품 레포 · 웹 · 설정 파일 · LLM 클라이언트 (ruff banned-api 와 같은 목록)
BANNED_EVERYWHERE = frozenset(
    {"playground_wiki", "playground_rag", "fastapi", "pydantic", "yaml", "ollama", "openai"}
)
# 하위 패키지별로 더 막는 것: 검색 · 지표는 DB 클라이언트를 직접 쓰지 않는다
BANNED_EXTERNAL = {
    "core": {"psycopg", "qdrant_client", "faiss", "torch"},
    "search": {"psycopg", "qdrant_client"},
    "eval": {"psycopg", "qdrant_client"},
}


def _imports(path: Path) -> list[tuple[int, str]]:
    """파일의 모든 import (함수 안 포함) 를 (줄, 모듈 이름) 으로."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def _violations(package: str, src: Path = SRC) -> list[tuple[str, int, str]]:
    """규칙을 어긴 import 를 (패키지 안 파일 경로, 줄, 모듈 이름) 으로."""
    allowed = ALLOWED_INTERNAL[package] | {package}
    banned = BANNED_EXTERNAL.get(package, set()) | BANNED_EVERYWHERE
    problems = []
    for path in sorted((src / package).rglob("*.py")):
        for line, module in _imports(path):
            parts = module.split(".")
            internal_breach = parts[0] == PACKAGE and len(parts) > 1 and parts[1] not in allowed
            if internal_breach or parts[0] in banned:
                problems.append((path.relative_to(src).as_posix(), line, module))
    return problems


def test_every_subpackage_has_a_rule():
    subpackages = {path.parent.name for path in SRC.glob("*/__init__.py")}

    assert subpackages == set(ALLOWED_INTERNAL)


@pytest.mark.parametrize("package", sorted(ALLOWED_INTERNAL))
def test_subpackage_imports_follow_layers(package: str):
    assert _violations(package) == []


def test_checker_catches_imports_inside_functions(tmp_path: Path):
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "bad.py").write_text(
        "from playground_retrieval.core import Hit\n"
        "def f():\n"
        "    from playground_retrieval.store import PgStore\n"
        "    import playground_wiki\n",
        encoding="utf-8",
    )

    assert _violations("search", tmp_path) == [
        ("search/bad.py", 3, "playground_retrieval.store"),
        ("search/bad.py", 4, "playground_wiki"),
    ]
