"""The DDT brain outside the model: checked claims, runs and the self-model, versioned.

Files in the brain directory (brain/ unless DDT_BRAIN_DIR is set):
brain.jsonl (readable entries), brain.npy (row i = the i-th claim), selfmodel.json,
meta.json ({"version": N}) and versions/v<N>/ (the files as they were at version N).
"""
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from . import llm

FILES = ["brain.jsonl", "brain.npy", "selfmodel.json"]
MERGE_SIM = 0.95
_batch = {"open": False}


def brain_dir() -> Path:
    d = Path(os.environ.get("DDT_BRAIN_DIR", "brain"))
    d = d if d.is_absolute() else llm.ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def version() -> int:
    p = brain_dir() / "meta.json"
    return json.loads(p.read_text())["version"] if p.exists() else 0


def _set_version(n: int) -> None:
    (brain_dir() / "meta.json").write_text(json.dumps({"version": n}))


def snapshot() -> int:
    """Copy the brain files to versions/v<current version>/, then add 1 to the version."""
    d, v = brain_dir(), version()
    snap = d / "versions" / f"v{v}"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    for f in FILES:
        if (d / f).exists():
            shutil.copy2(d / f, snap / f)
    _set_version(v + 1)
    return v + 1


@contextmanager
def batch():
    """One snapshot for several writes: one research run is one brain version."""
    snapshot()
    _batch["open"] = True
    try:
        yield
    finally:
        _batch["open"] = False


def before_write() -> None:
    if not _batch["open"]:
        snapshot()


def entries() -> list[dict]:
    p = brain_dir() / "brain.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line] if p.exists() else []


def vectors() -> np.ndarray | None:
    p = brain_dir() / "brain.npy"
    return np.load(p) if p.exists() else None


def _save(items: list[dict], vecs: np.ndarray | None) -> None:
    d = brain_dir()
    (d / "brain.jsonl").write_text("".join(json.dumps(e) + "\n" for e in items), encoding="utf-8")
    if vecs is not None:
        np.save(d / "brain.npy", vecs)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_claims(claims: list[dict], topic: str, model: str) -> tuple[int, int]:
    """Merge a claim into a stored one at cosine similarity 0.95 or more; append every other claim."""
    if not claims:
        return 0, 0
    before_write()
    items, vecs = entries(), vectors()
    new_vecs = llm.embed([c["text"] for c in claims])
    claim_items = [e for e in items if e["kind"] == "claim"]
    if vecs is not None and vecs.shape[1] != new_vecs.shape[1]:
        raise RuntimeError("embedding size differs from the stored vectors; use the same embedding model")
    added = merged = 0
    for c, v in zip(claims, new_vecs):
        if vecs is not None and len(vecs):
            sims = vecs @ v
            i = int(np.argmax(sims))
            if sims[i] >= MERGE_SIM:
                e = claim_items[i]
                if c["source_url"] != e["source_url"] and c["source_url"] not in e["also_sources"]:
                    e["also_sources"].append(c["source_url"])
                merged += 1
                continue
        e = {"id": uuid.uuid4().hex[:8], "kind": "claim", "text": c["text"], "topic": topic,
             "source_url": c["source_url"], "also_sources": [], "quote": c["quote"],
             "confidence": c.get("confidence", 0), "origin": "checked claim", "version": version(),
             "created_at": _now(), "model": model, "stale": False}
        items.append(e)
        claim_items.append(e)
        vecs = v[None, :] if vecs is None else np.vstack([vecs, v[None, :]])
        added += 1
    _save(items, vecs)
    return added, merged


def add_run(record: dict) -> None:
    before_write()
    items = entries()
    items.append({"id": uuid.uuid4().hex[:8], "kind": "run", "origin": "run", "version": version(),
                  "created_at": _now(), **record})
    _save(items, None)


def retrieve(question: str, k: int = 5, min_sim: float = 0.75) -> list[dict]:
    """The k most similar non-stale claims at or above min_sim, each with its similarity."""
    vecs = vectors()
    if vecs is None or not len(vecs):
        return []
    claims = [e for e in entries() if e["kind"] == "claim"]
    q = llm.embed([question])[0]
    if q.shape[0] != vecs.shape[1]:
        print("[WARN] embedding size differs from the stored vectors; nothing retrieved", flush=True)
        return []
    sims = vecs @ q
    order = [int(i) for i in np.argsort(-sims) if sims[i] >= min_sim and not claims[i]["stale"]]
    return [{**claims[i], "similarity": round(float(sims[i]), 3)} for i in order[:k]]


def rollback(steps: int = 1) -> int:
    """Restore versions/v<version - steps>/ and make it the current version."""
    d, target = brain_dir(), version() - steps
    snap = d / "versions" / f"v{target}"
    if target < 0 or not snap.exists():
        raise ValueError(f"no snapshot for version {target}")
    for f in FILES:
        (d / f).unlink(missing_ok=True)
        if (snap / f).exists():
            shutil.copy2(snap / f, d / f)
    _set_version(target)
    return target


def counts() -> dict:
    items = entries()
    return {"claims": sum(e["kind"] == "claim" for e in items),
            "runs": sum(e["kind"] == "run" for e in items), "version": version()}
