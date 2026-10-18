"""Deterministic stand-ins for the model and the web, used when DDT_FAKE=1. No network.

A question containing the word "unreliable" makes the stand-in model draft 2 made-up quotes out of 3,
so the self-model's evidence bar must rise (the stand-in version of check S5).
LATE pages are offline until LATE_ONLINE is set to True: the stand-in web gains a page between two runs
(the stand-in version of check S10). Words are compared by their first 5 letters ("stems").
"""
import json
import re

import numpy as np

from ddt.llm import local_embed

PAGES = {
    "https://example.org/nemotron-family": (
        "NVIDIA Nemotron 3 is a family of open models for agentic AI.\n"
        "The Nemotron 3 family comes in three sizes called Nano, Super and Ultra for different workloads.\n"
        "The models are released under the NVIDIA Open Model License, which allows commercial use."),
    "https://example.org/nemotron-deploy": (
        "Developers can run Nemotron models on NVIDIA GPUs in the cloud, in data centers or on workstations.\n"
        "Nemotron 3 Nano is designed for efficient inference on a single GPU with a long context window."),
    "https://example.org/eu-ai-act": (
        "The EU AI Act entered into force on 1 August 2024 after its publication in the Official Journal.\n"
        "Obligations for providers of general-purpose AI models apply from 2 August 2025 across the Union."),
    "https://example.org/eu-ai-act-fines": (
        "Fines under the EU AI Act can reach 35 million euros or 7 percent of worldwide annual turnover."),
    "https://example.org/startup-funding": (
        "European AI startups raised record venture funding in 2026 according to several industry trackers.\n"
        "Several large rounds in 2026 went to companies building foundation models in France and Germany."),
}
LATE = {
    "https://example.org/nemotron-benchmark": (
        "Nemotron 3 Super reached the top score on the agentic benchmark results table published in 2026."),
}
LATE_ONLINE = False
TOPICS = {"nemotron": 0, "act": 1, "funding": 2, "startup": 2, "unreliable": 3, "benchmark": 4}
STOP = {"what", "does", "which", "when", "where", "with", "under", "from", "nvidia", "nemotron"}


def _pages() -> dict:
    return {**PAGES, **LATE} if LATE_ONLINE else PAGES


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) > 3 and w not in STOP}


def _topic_of(text: str) -> str:
    t = text.lower()
    if "nemotron" in t:
        return "nvidia nemotron"
    if "ai act" in t:
        return "eu ai act"
    if "unreliable" in t:
        return "unreliable topic"
    return "fake topic"


def search(query: str) -> list[dict]:
    words = set(re.findall(r"\w+", query.lower()))
    pages = _pages()
    urls = sorted(pages, key=lambda u: -len(words & set(re.findall(r"\w+", u))))  # best match first
    return [{"url": u, "title": u.rsplit("/", 1)[-1], "content": pages[u][:200], "raw_content": pages[u]}
            for u in urls]


def extract(url: str) -> str:
    return _pages().get(url, "")


def chat_json(system: str, user: str) -> dict:
    data = json.loads(user)
    q = data["question"]
    if system.startswith("TASK: PLAN"):
        parts = [x.strip() for x in q.rstrip("?").split(" and ")]
        known = [c["id"] for c in data["known_claims"] if len(_stems(q) & _stems(c["text"])) >= 2]
        if known:
            return {"topic": _topic_of(q), "parts": parts, "covered": known, "queries": []}
        return {"topic": _topic_of(q), "parts": parts, "covered": [], "queries": [q, f"{q} overview", f"{q} details"]}
    claims = []
    for s in data["sources"][:3]:
        lines = s["excerpt"].split("\n")
        line = max(lines, key=lambda l: len(_stems(q) & _stems(l)))  # the first line on a tie
        quote = " ".join(line.split()[:12])
        if "unreliable" in q.lower() and len(claims) >= 1:
            quote = "this sentence was invented by the stand-in model and appears on no page at all"
        claims.append({"text": line, "source_url": s["url"], "quote": quote, "confidence": 0.8})
    return {"claims": claims, "enough": True, "reason": "stand-in: every part has a source", "next_queries": []}


def chat_text(system: str, user: str) -> str:
    claims = re.findall(r"^\[(\d+)\] (.+)$", user, flags=re.M)
    return " ".join(f"{text.rstrip('.')} [{n}]." for n, text in claims)


def embed(texts: list[str]) -> np.ndarray:
    """Texts on the same topic score about 0.8; identical texts score 1; other topics score low."""
    bow = local_embed(texts)
    bow[:, :16] = 0
    out = np.zeros_like(bow)
    for i, t in enumerate(texts):
        hits = {d for w, d in TOPICS.items() if w in t.lower()}
        for d in hits:
            out[i, d] = 1.0
        out[i] = out[i] / (np.linalg.norm(out[i]) or 1) * 0.894 + bow[i] / (np.linalg.norm(bow[i]) or 1) * 0.447
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n == 0, 1, n)).astype(np.float32)
