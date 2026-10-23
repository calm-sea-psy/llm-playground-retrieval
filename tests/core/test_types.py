from dataclasses import FrozenInstanceError

import pytest
from playground_corpus import Chunk

from playground_retrieval.core import Filters, Hit


def _chunk() -> Chunk:
    return Chunk(
        id="1-0",
        doc_id="1",
        title="서울특별시",
        section="역사",
        text="서울특별시 > 역사\n조선의 수도였다.",
        seq=0,
        token_count=12,
    )


def test_hit_is_frozen():
    hit = Hit(chunk=_chunk(), score=0.9)

    with pytest.raises(FrozenInstanceError):
        hit.score = 0.1  # type: ignore[misc]  # frozen 확인용


def test_filters_default_to_no_restriction():
    filters = Filters()

    assert not filters.doc_ids
    assert not filters.categories
