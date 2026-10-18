"""Tavily search and extract, plus the two planning calls to the chat model."""
import json
import os
import re

from . import llm

SEARCHES = {"run": 0, "total": 0}  # per-run counter (reset by the agent) and process total
SEARCH_CAP = int(os.environ["DDT_SEARCH_CAP"]) if os.environ.get("DDT_SEARCH_CAP") else None  # cap on the process total
EXCERPT_CHARS = 3000  # characters of each page shown to the model when drafting claims
_tavily = None


def _client():
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily


def search(query: str) -> list[dict]:
    if SEARCH_CAP is not None and SEARCHES["total"] >= SEARCH_CAP:
        raise RuntimeError(f"search cap of {SEARCH_CAP} reached")
    SEARCHES["run"] += 1
    SEARCHES["total"] += 1
    if llm.fake():
        from tests import fakes
        return fakes.search(query)
    r = _client().search(query=query, search_depth="advanced", max_results=5, include_raw_content=True)
    return [{"url": x["url"], "title": x.get("title") or "", "content": x.get("content") or "",
             "raw_content": x.get("raw_content") or ""} for x in r.get("results", [])]


def extract(url: str) -> str:
    if llm.fake():
        from tests import fakes
        return fakes.extract(url)
    r = _client().extract(urls=[url])
    results = r.get("results") or []
    return (results[0].get("raw_content") or "") if results else ""


PLAN_SYSTEM = """TASK: PLAN. You plan web research for a research agent.
You get a question, the claims the agent already knows (each with an id) and the evidence bar.
Return only JSON: {"topic": "2 to 4 lowercase words", "covered": [ids of known claims that answer part of the question], "queries": [web search queries for what is still missing]}.
Rules:
- Prefer the known claims. Add a query only for a part of the question that no known claim answers.
- If the known claims fully answer the question, "queries" is [].
- If no claims are known, return exactly 3 queries.
- Never return more than 3 queries.
- The evidence bar is how many claims from different sources each part of the question needs."""


def plan(question: str, known_claims: list[dict], evidence_bar: int) -> dict:
    user = json.dumps({"question": question, "evidence_bar": evidence_bar,
                       "known_claims": [{"id": c["id"], "text": c["text"]} for c in known_claims]})
    out = llm.chat_json(PLAN_SYSTEM, user)
    known_ids = {c["id"] for c in known_claims}
    return {"topic": str(out.get("topic") or "").strip().lower(),
            "covered": [i for i in out.get("covered") or [] if i in known_ids],
            "queries": [str(q) for q in (out.get("queries") or [])][:3]}


DRAFT_SYSTEM = """TASK: DRAFT. You extract claims that answer a research question from web sources.
You get the question, the evidence bar and numbered sources, each with its url and an excerpt of its text.
Return only JSON: {"claims": [{"text": "one factual sentence", "source_url": "...", "quote": "...", "confidence": 0.0}], "enough": true, "reason": "...", "next_queries": []}.
Rules:
- "quote" is 8 to 30 words copied exactly, character for character, from the excerpt of that same source. No ellipses, no changed words.
- "source_url" is only a url from the given sources.
- "confidence" is from 0 to 1.
- Each part of the question needs as many claims from different sources as the evidence bar says; otherwise "enough" is false.
- "reason" says in one sentence why the evidence is or is not enough.
- If "enough" is false, "next_queries" holds up to 2 web search queries for what is missing; otherwise [].
- Return 3 to 8 claims when the sources allow, covering every part of the question, from as many different sources as possible.
- Copy each quote from within one line of an excerpt."""


def excerpt(text: str, question: str, limit: int = EXCERPT_CHARS) -> str:
    """The lines and sentences of a page that share the most words with the question, in page order."""
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
    paras = [p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if len(p.strip()) > 40]
    ranked = sorted(range(len(paras)), key=lambda i: -len(words & set(re.findall(r"\w+", paras[i].lower()))))
    keep, size = [], 0
    for i in ranked:
        if size + len(paras[i]) > limit and keep:
            continue
        keep.append(i)
        size += len(paras[i])
    return "\n".join(paras[i][:limit] for i in sorted(keep))


def draft_claims(question: str, results: list[dict], evidence_bar: int) -> dict:
    sources = [{"n": i + 1, "url": r["url"], "title": r["title"],
                "excerpt": excerpt(r["raw_content"] or r["content"], question)} for i, r in enumerate(results)]
    user = json.dumps({"question": question, "evidence_bar": evidence_bar, "sources": sources})
    out = llm.chat_json(DRAFT_SYSTEM, user)
    urls = {r["url"] for r in results}
    claims = [{"text": str(c.get("text", "")), "source_url": str(c.get("source_url", "")),
               "quote": str(c.get("quote", "")), "confidence": float(c.get("confidence") or 0)}
              for c in out.get("claims") or [] if c.get("source_url") in urls]
    return {"claims": claims, "enough": bool(out.get("enough")), "reason": str(out.get("reason") or ""),
            "next_queries": [str(q) for q in (out.get("next_queries") or [])][:2]}
