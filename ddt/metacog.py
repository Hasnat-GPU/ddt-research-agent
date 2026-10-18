"""The metacognitive layer: predicts each run from similar past episodes, and an online claim model
whose 6 weights update after every checked claim. Weights and log live in metacog.json in the brain."""
import json
import math
import os
from urllib.parse import urlparse

import numpy as np

from . import brain, selfmodel
from .stores import episodes

PRIOR_PASS = 0.7


def prior_searches(known: int) -> int:
    return 3 if known == 0 else 0


def predict(question_vec, known_count: int, topic: str, use_memory: bool = True) -> dict:
    """Pass rate and searches: the similarity-weighted means of the neighbours' actual values (HANDOFF-2 section 2).
    With no neighbours: the prior."""
    near = episodes.search(question_vec, k=5, min_sim=0.6) if use_memory else []
    pred = {"pass_rate": PRIOR_PASS, "searches": float(prior_searches(known_count)), "known": known_count,
            "basis": len(near), "neighbours": [e["id"] for e, _ in near]}
    if near:
        total = sum(s for _, s in near)
        pred["searches"] = sum(s * e["actual"]["searches"] for e, s in near) / total
        rated = [(e["actual"]["pass_rate"], s) for e, s in near if e["actual"]["pass_rate"] is not None]
        if rated:
            pred["pass_rate"] = sum(p * s for p, s in rated) / sum(s for _, s in rated)
    pred["pass_rate"], pred["searches"] = round(pred["pass_rate"], 3), round(pred["searches"], 3)
    return pred


def compare(prediction: dict, actual: dict) -> dict:
    pr = None if actual["pass_rate"] is None else round(abs(prediction["pass_rate"] - actual["pass_rate"]), 3)
    return {"pass_rate": pr, "searches": round(abs(prediction["searches"] - actual["searches"]), 3)}


# Online claim model ---------------------------------------------------------------------------

def _path():
    return brain.brain_dir() / "metacog.json"


def load() -> dict:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"w": [0.0] * 6, "lr": 0.5, "log": []}


def save(state: dict) -> None:
    if not brain.in_batch():
        with brain.batch():
            return save(state)
    brain.before_write()
    _path().write_text(json.dumps(state), encoding="utf-8")


def lr() -> float:
    return float(os.environ["DDT_META_LR"]) if os.environ.get("DDT_META_LR") else 0.5


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def p_pass(x: list[float], w: list[float]) -> float:
    return sigmoid(sum(a * b for a, b in zip(w, x)))


def domain_pass_rates() -> dict:
    return {d: p / n for d, (p, n) in domain_counts().items()}


def domain_counts() -> dict:
    """Exact counts for runs with check history; legacy successes are unknown.

    Do not invent repeated successful checks from deduplicated legacy claims.
    Only explicit check outcomes contribute to the current feature.
    """
    counts = {}
    for e in brain.entries():
        for outcome in e.get("checks", []) if e["kind"] == "run" else []:
            record_outcome(counts, outcome["url"], outcome["passed"])
    return counts


def record_outcome(counts, url, passed):
    c = counts.setdefault(urlparse(url).netloc, [0, 0])
    c[0] += int(passed)
    c[1] += 1


def update_weights(w, x, y, learning_rate):
    """Pure production/replay update, with the pre-update prediction."""
    if len(w) != 6 or len(x) != 6 or not all(math.isfinite(v) for v in [*w, *x, learning_rate]) or learning_rate < 0:
        raise ValueError("invalid online learning state")
    p = p_pass(x, w)
    return [wi + learning_rate * (int(y) - p) * xi for wi, xi in zip(w, x)], p


def features(claim: dict, claim_vec, question_vec, domains: dict, topic: str) -> list[float]:
    sm = selfmodel.get(topic)
    return [1.0, float(claim.get("confidence") or 0), min(len(claim["quote"].split()), 30) / 30,
            domains.get(urlparse(claim["source_url"]).netloc, 0.5),
            sm["passed"] / sm["checked"] if sm["checked"] else 0.5,
            float(np.dot(claim_vec, question_vec))]


def learn(x: list[float], y: bool, state: dict | None = None) -> dict:
    """Log x, y and the prediction, then one gradient step: w += lr * (y - p) * x."""
    if state is None and not brain.in_batch():
        with brain.batch():
            return learn(x, y)
    own = state is None
    state = load() if own else state
    weights, p = update_weights(state["w"], x, y, lr())
    state["log"] = (state["log"] + [{"x": x, "y": int(y), "p": round(p, 4)}])[-500:]
    state["lr"] = lr()
    state["w"] = weights
    if own:
        save(state)
    return state
