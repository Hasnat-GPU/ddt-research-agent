"""One client for Nebius Token Factory: the chat model and the embedding model."""
import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if os.environ.get("DDT_FAKE") != "1" and os.environ.get("DDT_SKIP_DOTENV") != "1":
    load_dotenv(ROOT / ".env")  # offline execution never needs secrets

MODEL_ID = os.environ.get("NEBIUS_MODEL", "")
_client = None
_embedding_session = ContextVar("embedding_session", default=None)
_last_embedding = ContextVar("last_embedding", default=None)


def output_dir() -> Path:
    return Path(os.environ.get("DDT_OUTPUT_DIR", str(ROOT)))


def fake() -> bool:
    return os.environ.get("DDT_FAKE") == "1"


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=os.environ["NEBIUS_BASE_URL"], api_key=os.environ["NEBIUS_API_KEY"],
                         timeout=180, max_retries=2)
    return _client


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.split("</think>")[-1].strip()  # a reply that starts inside the think block


def _chat(system: str, user: str) -> str:
    r = client().chat.completions.create(
        model=MODEL_ID, temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return _strip_think(r.choices[0].message.content)


def _parse(text: str) -> dict:
    out = json.loads(text[text.find("{"): text.rfind("}") + 1])
    if not isinstance(out, dict):
        raise ValueError("model response must be a JSON object")
    return out


def chat_json(system: str, user: str, validator=None) -> dict:
    for attempt in range(2):
        try:
            if fake():
                from tests import fakes
                out = fakes.chat_json(system, user)
            else:
                out = _parse(_chat(system, user))
            if not isinstance(out, dict):
                raise ValueError("model response must be an object")
            return validator(out) if validator else out
        except (ValueError, TypeError, KeyError) as exc:
            if attempt:
                raise ValueError("model returned invalid structured output after two attempts") from exc
            system += "\nCorrect the response: use exactly the required JSON types, fields and allowed values."


def chat_text(system: str, user: str) -> str:
    if fake():
        from tests import fakes
        return fakes.chat_text(system, user)
    return _chat(system, user)


def local_embed(texts: list[str]) -> np.ndarray:
    """Hashed bag of words: 512 dimensions, L2-normalized."""
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 512] += 1
    return _normalize(out)


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


def embed(texts: list[str]) -> np.ndarray:
    config = embedding_config()
    session = _embedding_session.get()
    if session is not None and session[0] != config:
        raise RuntimeError("embedding configuration changed during this run")
    use_brain = session is None or session[1]
    stored = None
    if use_brain:
        from . import brain
        d = brain.brain_dir()
        p = d / "embedding.json"
        stored = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        if stored is not None and any(stored.get(k) != v for k, v in config.items()):
            raise RuntimeError("embedding identity differs from this brain; use a separate brain or explicitly reindex")
        if stored is None and any((d / f"{name}.npy").exists() for name in ("brain", "episodes", "gaps", "tools")):
            raise RuntimeError("legacy brain has no embedding identity; preserve it and explicitly reindex into a new brain")
    if config["provider"] == "stand-in":
        from tests import fakes
        out = fakes.embed(texts)
    elif config["provider"] == "local":
        out = local_embed(texts)
    else:
        # Never silently switch vector spaces on a provider failure.
        r = client().embeddings.create(model=config["model"], input=texts)
        out = _normalize(np.array([d.embedding for d in r.data], dtype=np.float32))
    if out.ndim != 2 or len(out) != len(texts) or not np.isfinite(out).all():
        raise ValueError("embedding provider returned invalid vectors")
    identity = {**config, "dimension": out.shape[1]}
    if stored is not None and stored != identity:
        raise RuntimeError("embedding dimension changed; brain was not modified")
    previous = _last_embedding.get()
    if session is not None and previous is not None and previous != identity:
        raise RuntimeError("embedding identity changed during this run")
    _last_embedding.set(identity)
    return out


def embedding_config() -> dict:
    if fake():
        return {"provider": "stand-in", "model": "ddt-fake-v1", "normalization": "l2"}
    if os.environ.get("DDT_EMBED") == "local":
        return {"provider": "local", "model": "hashed-bow-512-v1", "normalization": "l2"}
    return {"provider": "nebius", "endpoint": os.environ.get("NEBIUS_BASE_URL", ""),
            "model": os.environ.get("NEBIUS_EMBED_MODEL", ""), "normalization": "l2"}


@contextmanager
def embedding_session(use_brain=True):
    token = _embedding_session.set((embedding_config(), use_brain))
    last = _last_embedding.set(None)
    try:
        yield
    finally:
        _last_embedding.reset(last)
        _embedding_session.reset(token)


def record_embedding():
    from . import brain
    brain.before_write()
    identity = _last_embedding.get()
    if identity is not None:
        (brain.brain_dir() / "embedding.json").write_text(json.dumps(identity), encoding="utf-8")
