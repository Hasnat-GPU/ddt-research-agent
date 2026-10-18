import numpy as np

from ddt import brain, metacog
from ddt.stores import episodes

Q = np.zeros(8, dtype=np.float32)
Q[0] = 1


def test_predict_without_neighbours_uses_the_prior():
    p = metacog.predict(Q, 0, "t")
    assert (p["pass_rate"], p["searches"], p["basis"]) == (0.7, 3.0, 0)
    assert metacog.predict(Q, 4, "t")["searches"] == 0.0


def test_predict_with_neighbours():
    with brain.batch():
        episodes.add({"known": 0, "actual": {"searches": 3, "pass_rate": 0.5}}, Q)
        episodes.add({"known": 2, "actual": {"searches": 1, "pass_rate": 1.0}}, Q)
    p = metacog.predict(Q, 3, "t")
    assert p["basis"] == 2
    assert p["pass_rate"] == 0.75  # equal similarity: mean of 0.5 and 1.0
    assert p["searches"] == 0.5  # prior 0, plus the mean of the neighbours' errors against their prior: (0 + 1) / 2
    assert metacog.compare(p, {"pass_rate": 1.0, "searches": 0}) == {"pass_rate": 0.25, "searches": 0.5}


def test_learn_moves_w_toward_the_label():
    x = [1, 0.8, 0.5, 0.5, 0.5, 0.9]
    s = metacog.learn(x, True, {"w": [0.0] * 6, "lr": 0.5, "log": []})
    assert all(w > 0 for w in s["w"]) and s["log"][0] == {"x": x, "y": 1, "p": 0.5}
    s = metacog.learn(x, False, {"w": [0.0] * 6, "lr": 0.5, "log": []})
    assert all(w < 0 for w in s["w"])


def test_lr_zero_keeps_w(monkeypatch):
    monkeypatch.setenv("DDT_META_LR", "0")
    s = metacog.learn([1, 0.8, 0.5, 0.5, 0.5, 0.9], True, {"w": [0.0] * 6, "lr": 0.5, "log": []})
    assert s["w"] == [0.0] * 6 and len(s["log"]) == 1
