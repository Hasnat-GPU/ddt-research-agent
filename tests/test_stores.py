import numpy as np

from ddt import brain, metacog
from ddt.stores import episodes, gaps, tools


def v(i: int) -> np.ndarray:
    x = np.zeros(8, dtype=np.float32)
    x[i] = 1
    return x


def test_add_search_update():
    with brain.batch():
        a = gaps.add({"text": "part a", "status": "open"}, v(0))
        gaps.add({"text": "part b", "status": "open"}, v(1))
    assert gaps.count() == 2 and len(a) == 8
    found = gaps.search(v(0))
    assert [(e["text"], s) for e, s in found] == [("part a", 1.0)]
    with brain.batch():
        gaps.update(a, status="closed")
    assert [e["status"] for e in gaps.all()] == ["closed", "open"]


def test_rollback_undoes_every_store_in_one_batch():
    with brain.batch():
        episodes.add({"question": "q1"}, v(0))
    with brain.batch():
        episodes.add({"question": "q2"}, v(1))
        gaps.add({"text": "g", "status": "open"}, v(2))
        tools.add({"query": "t", "yield": 1}, v(3))
        metacog.learn([1, 0, 0, 0, 0, 0], True)
    assert (episodes.count(), gaps.count(), tools.count()) == (2, 1, 1) and metacog.load()["w"][0] > 0
    brain.rollback(1)
    assert (episodes.count(), gaps.count(), tools.count()) == (1, 0, 0) and metacog.load()["w"] == [0.0] * 6
