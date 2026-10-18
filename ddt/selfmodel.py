"""The self-model: per topic, how often claims pass the quote check, and the evidence bar it sets."""
import json

from . import brain

DEFAULT = {"runs": 0, "checked": 0, "passed": 0, "last5": [], "searches": 0, "evidence_bar": 1, "models": []}


def _path():
    return brain.brain_dir() / "selfmodel.json"


def load() -> dict:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def get(topic: str) -> dict:
    return {**DEFAULT, "last5": [], "models": [], **load().get(topic, {})}


def update(topic: str, results: list[bool], searches: int, model: str) -> str:
    brain.before_write()
    data = load()
    e = get(topic)
    old = e["evidence_bar"]
    e["runs"] += 1
    e["checked"] += len(results)
    e["passed"] += sum(results)
    e["searches"] += searches
    e["last5"] = (e["last5"] + [bool(r) for r in results])[-5:]
    if model and model not in e["models"]:
        e["models"].append(model)
    n, ok = len(e["last5"]), sum(e["last5"])
    if n >= 3 and ok < 0.8 * n:
        e["evidence_bar"] = 2
    elif n == 5 and ok == 5:
        e["evidence_bar"] = 1
    data[topic] = e
    _path().write_text(json.dumps(data, indent=1), encoding="utf-8")
    return f'evidence bar for "{topic}": {old} -> {e["evidence_bar"]} ({ok} of {n} claims passed)'
