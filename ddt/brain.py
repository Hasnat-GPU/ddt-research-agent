"""The DDT brain outside the model: checked claims, runs and the self-model, versioned.

Files in the brain directory (brain/ unless DDT_BRAIN_DIR is set):
brain.jsonl (readable entries), brain.npy (row i = the i-th claim), selfmodel.json, metacog.json,
the harness stores episodes, gaps and tools (.jsonl + .npy each), meta.json ({"version": N}) and versions/v<N>/ (the files as they were at version N).
"""
import json
import os
import uuid
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from . import llm, persistence

FILES = ["brain.jsonl", "brain.npy", "selfmodel.json", "metacog.json", "episodes.jsonl", "episodes.npy",
          "gaps.jsonl", "gaps.npy", "tools.jsonl", "tools.npy", "embedding.json"]
MERGE_SIM = 0.95
_local = threading.local()


def _root_dir() -> Path:
    d = Path(os.environ.get("DDT_BRAIN_DIR", "brain"))
    d = d if d.is_absolute() else llm.ROOT / d
    return d.resolve()


@contextmanager
def lock():
    root = _root_dir()
    with persistence.locked(root):
        if not in_batch():
            persistence.recover(root, FILES + ["meta.json"])
        yield


def in_batch() -> bool:
    return getattr(_local, "stage", None) is not None


def brain_dir() -> Path:
    if in_batch():
        return _local.stage
    with lock():
        return _root_dir()


def version() -> int:
    p = brain_dir() / "meta.json"
    return json.loads(p.read_text())["version"] if p.exists() else 0


def _set_version(n: int) -> None:
    (brain_dir() / "meta.json").write_text(json.dumps({"version": n}))


def snapshot() -> int:
    """Publish an empty version; ordinary mutations use batch()."""
    with batch():
        return version()


@contextmanager
def batch():
    """Stage one version and publish it only after all writes validate."""
    if in_batch():
        raise RuntimeError("nested brain batches are not supported")
    with lock():
        root, v = _root_dir(), version()
        persistence.copy_files(root, root / "versions" / f"v{v}", FILES)
        with persistence.transaction(root, FILES + ["meta.json"]) as stage:
            _local.stage = stage
            try:
                _set_version(v + 1)
                yield
                _validate()
            finally:
                _local.stage = None


def _validate():
    d = brain_dir()
    dimensions = set()
    for name in ("brain", "episodes", "gaps", "tools"):
        p, pv = d / f"{name}.jsonl", d / f"{name}.npy"
        items = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s] if p.exists() else []
        rows = sum(e["kind"] == "claim" for e in items) if name == "brain" else len(items)
        vecs = np.load(pv, allow_pickle=False) if pv.exists() else None
        if (vecs is None and rows) or (vecs is not None and (vecs.ndim != 2 or len(vecs) != rows or not np.isfinite(vecs).all())):
            raise RuntimeError(f"invalid row/vector alignment in {name}")
        if vecs is not None and len(vecs):
            dimensions.add(vecs.shape[1])
    if len(dimensions) > 1:
        raise RuntimeError("brain stores use different embedding dimensions")
    identity = d / "embedding.json"
    if identity.exists() and dimensions and dimensions != {json.loads(identity.read_text(encoding="utf-8"))["dimension"]}:
        raise RuntimeError("embedding identity does not match stored dimensions")
    for name in ("selfmodel.json", "metacog.json", "embedding.json", "meta.json"):
        if (d / name).exists():
            json.loads((d / name).read_text(encoding="utf-8"))


def before_write() -> None:
    if not in_batch():
        raise RuntimeError("brain writes require a batch")


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


def add_claims(claims: list[dict], topic: str, model: str, *, ids: list[str] | None = None) -> tuple[int, int]:
    """Merge a claim into a stored one at cosine similarity 0.95 or more; append every other claim."""
    if not claims:
        return 0, 0
    if not in_batch():
        with batch():
            return add_claims(claims, topic, model, ids=ids)
    before_write()
    items, vecs = entries(), vectors()
    new_vecs = llm.embed([c["text"] for c in claims])
    llm.record_embedding()
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
                if ids is not None:
                    ids.append(e["id"])
                continue
        e = {"id": uuid.uuid4().hex[:8], "kind": "claim", "text": c["text"], "topic": topic,
             "source_url": c["source_url"], "also_sources": [], "quote": c["quote"],
             "confidence": c.get("confidence", 0), "origin": "checked claim", "version": version(),
             "created_at": _now(), "model": model, "stale": False}
        items.append(e)
        claim_items.append(e)
        vecs = v[None, :] if vecs is None else np.vstack([vecs, v[None, :]])
        added += 1
        if ids is not None:
            ids.append(e["id"])
    _save(items, vecs)
    return added, merged


def add_run(record: dict) -> None:
    if not in_batch():
        with batch():
            return add_run(record)
    before_write()
    items = entries()
    items.append({"id": uuid.uuid4().hex[:8], "kind": "run", "origin": "run", "version": version(),
                  "created_at": _now(), **record})
    _save(items, None)


def retrieve(question: str, k: int = 5, min_sim: float = 0.75, vec=None) -> list[dict]:
    """The k most similar non-stale claims at or above min_sim, each with its similarity.
    vec: the question's vector, if the caller already has it."""
    vecs = vectors()
    if vecs is None or not len(vecs):
        return []
    claims = [e for e in entries() if e["kind"] == "claim"]
    q = llm.embed([question])[0] if vec is None else vec
    if q.shape[0] != vecs.shape[1]:
        print("[WARN] embedding size differs from the stored vectors; nothing retrieved", flush=True)
        return []
    sims = vecs @ q
    order = [int(i) for i in np.argsort(-sims) if sims[i] >= min_sim and not claims[i]["stale"]]
    return [{**claims[i], "similarity": round(float(sims[i]), 3)} for i in order[:k]]


def rollback(steps: int = 1) -> int:
    """Restore versions/v<version - steps>/ and make it the current version."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("rollback steps must be a positive integer")
    if in_batch():
        raise RuntimeError("cannot roll back during a batch")
    with lock():
        d, target = _root_dir(), version() - steps
        snap = d / "versions" / f"v{target}"
        if target < 0 or not snap.exists():
            raise ValueError(f"no snapshot for version {target}")
        with persistence.transaction(d, FILES + ["meta.json"]) as stage:
            _local.stage = stage
            try:
                persistence.copy_files(snap, stage, FILES)
                _set_version(target)
                _validate()
            finally:
                _local.stage = None
        return target


def counts() -> dict:
    items = entries()
    return {"claims": sum(e["kind"] == "claim" for e in items),
            "runs": sum(e["kind"] == "run" for e in items), "version": version()}


def reindex() -> int:
    """Explicitly rebuild every vector store with the selected embedding model.

    IDs and readable records stay unchanged. The entire migration is one
    recoverable version, including embedding identity, and can be rolled back.
    """
    with lock(), llm.embedding_session(use_brain=False), batch():
        d = brain_dir()
        for name, field in (("brain", "text"), ("episodes", "question"), ("gaps", "text"), ("tools", "query")):
            p = d / f"{name}.jsonl"
            items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line] if p.exists() else []
            if name == "brain":
                items = [e for e in items if e["kind"] == "claim"]
            if items:
                vecs = llm.embed([e[field] for e in items])
                np.save(d / f"{name}.npy", vecs)
            else:
                (d / f"{name}.npy").unlink(missing_ok=True)
        llm.record_embedding()
        new_version = version()
    return new_version
