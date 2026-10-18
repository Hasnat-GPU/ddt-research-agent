"""One client for Nebius Token Factory: the chat model and the embedding model."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # never overrides variables already set in the environment

MODEL_ID = os.environ.get("NEBIUS_MODEL", "")
_client = None


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
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def chat_json(system: str, user: str) -> dict:
    if fake():
        from tests import fakes
        return fakes.chat_json(system, user)
    try:
        return _parse(_chat(system, user))
    except json.JSONDecodeError:
        return _parse(_chat(system + "\nReturn only valid JSON.", user))


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
    if fake():
        from tests import fakes
        return fakes.embed(texts)
    if os.environ.get("DDT_EMBED") == "local":
        return local_embed(texts)
    try:
        r = client().embeddings.create(model=os.environ["NEBIUS_EMBED_MODEL"], input=texts)
        return _normalize(np.array([d.embedding for d in r.data], dtype=np.float32))
    except Exception as e:  # the brain still works with the weaker local embedding
        print(f"[WARN] embedding call failed ({type(e).__name__}); using local embedding", flush=True)
        return local_embed(texts)
