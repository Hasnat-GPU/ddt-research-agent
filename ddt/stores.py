"""Vector stores for the harness: a <name>.jsonl (readable entries) plus <name>.npy (row i = entry i) pair."""
import json
import uuid

import numpy as np

from . import brain


class VectorStore:
    def __init__(self, name: str):
        self.name = name

    def _paths(self):
        d = brain.brain_dir()
        return d / f"{self.name}.jsonl", d / f"{self.name}.npy"

    def all(self) -> list[dict]:
        p, _ = self._paths()
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l] if p.exists() else []

    def _vectors(self) -> np.ndarray | None:
        _, p = self._paths()
        return np.load(p) if p.exists() else None

    def _save(self, items: list[dict], vecs: np.ndarray | None) -> None:
        brain.before_write()
        p, pv = self._paths()
        p.write_text("".join(json.dumps(e) + "\n" for e in items), encoding="utf-8")
        if vecs is not None:
            np.save(pv, vecs)

    def add(self, entry: dict, vec: np.ndarray) -> str:
        if not brain.in_batch():
            with brain.batch():
                return self.add(entry, vec)
        items, vecs = self.all(), self._vectors()
        e = {"id": uuid.uuid4().hex[:8], **entry, "version": brain.version(), "created_at": brain._now()}
        items.append(e)
        vecs = vec[None, :] if vecs is None else np.vstack([vecs, vec[None, :]])
        self._save(items, vecs.astype(np.float32))
        return e["id"]

    def search(self, vec: np.ndarray, k: int = 5, min_sim: float = 0.6,
               predicate=None, distinct: str | None = None) -> list[tuple[dict, float]]:
        vecs = self._vectors()
        if vecs is None or not len(vecs) or vecs.shape[1] != vec.shape[0]:
            return []
        items, sims = self.all(), vecs @ vec
        found, seen = [], set()
        for i in np.argsort(-sims):
            e = items[int(i)]
            if sims[i] < min_sim or (predicate is not None and not predicate(e)):
                continue
            if distinct is not None:
                value = e[distinct]
                if value in seen:
                    continue
                seen.add(value)
            found.append((e, round(float(sims[i]), 3)))
            if len(found) >= k:
                break
        return found if k > 0 else []

    def vectors_of(self, ids: list[str]) -> np.ndarray:
        idx = {e["id"]: i for i, e in enumerate(self.all())}
        return self._vectors()[[idx[i] for i in ids]]

    def update(self, id: str, **fields) -> None:
        if not brain.in_batch():
            with brain.batch():
                return self.update(id, **fields)
        items = self.all()
        for e in items:
            if e["id"] == id:
                e.update(fields)
        self._save(items, None)

    def count(self) -> int:
        return len(self.all())


episodes = VectorStore("episodes")  # context: one episode per run
gaps = VectorStore("gaps")  # state: open parts of questions
tools = VectorStore("tools")  # tool memory: search queries and their yield
