"""The DDT loop: Read, Plan, Act, Check, Write, Reflect, Report."""
import re
from datetime import datetime

from . import brain, check, llm, selfmodel
from . import research as web

MAX_SEARCHES = 6

ANSWER_SYSTEM = """You write a short research answer (3 to 6 sentences) from numbered claims.
Use only the given claims; add no other facts. End every sentence with the number of each claim it uses, like [2].
Return plain text only."""


class Trace:
    def __init__(self, stamp: str):
        self.lines = []
        d = llm.ROOT / "traces"
        d.mkdir(exist_ok=True)
        self.path = _unique(d / f"{stamp}.log")

    def __call__(self, tag: str, text: str) -> None:
        line = f"[{tag}] {text}"
        self.lines.append(line)
        print(line, flush=True)

    def save(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _unique(p):
    n = 2
    while p.exists():
        p = p.with_name(f"{p.stem.split('~')[0]}~{n}{p.suffix}")
        n += 1
    return p


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def research(question: str, topic: str | None = None, use_brain: bool = True) -> dict:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace = Trace(stamp)
    web.SEARCHES["run"] = 0
    model = llm.MODEL_ID
    try:
        return _run(question, topic, use_brain, model, stamp, trace)
    finally:
        trace.save()


def _run(question, topic, use_brain, model, stamp, trace) -> dict:
    trace("READ", f'question: "{question}" | model: {model} | brain: {"on" if use_brain else "off"}')

    # 1. Read
    known = []
    if use_brain:
        known = brain.retrieve(question)
        trace("READ", f"{len(known)} known claims at similarity 0.75 or more (brain version {brain.version()})")
        for c in known:
            trace("READ", f"  {c['id']} sim {c['similarity']}: {c['text']}")
        if topic is None and known:
            topic = known[0]["topic"]
            trace("READ", f'topic from the closest known claim: "{topic}"')
        if topic is not None:
            trace("READ", f'evidence bar for "{topic}": {selfmodel.get(topic)["evidence_bar"]}')

    # 2. Plan
    bar = selfmodel.get(topic)["evidence_bar"] if use_brain and topic else 1
    p = web.plan(question, known, bar)
    if topic is None:
        topic = p["topic"] or "general"
        bar = selfmodel.get(topic)["evidence_bar"] if use_brain else 1
        trace("PLAN", f'topic from the plan: "{topic}" | evidence bar {bar}')
    covered = [c for c in known if c["id"] in p["covered"]]
    trace("PLAN", f'topic "{topic}" | covered by known claims: {[c["id"] for c in covered] or "none"}')
    trace("PLAN", f"queries: {p['queries'] or 'none'}")

    # 3. Act
    results, seen, drafted, queries = [], set(), [], p["queries"]
    if not queries:
        trace("DECIDE", "the known claims answer the question: no search")
    rounds = 0
    while queries:
        rounds += 1
        for q in queries:
            if web.SEARCHES["run"] >= MAX_SEARCHES:
                trace("DECIDE", f"search limit of {MAX_SEARCHES} reached")
                break
            hits = web.search(q)
            fresh = [h for h in hits if h["url"] not in seen]
            seen.update(h["url"] for h in fresh)
            results += fresh
            trace("SEARCH", f'#{web.SEARCHES["run"]} "{q}": {len(hits)} results, {len(fresh)} new pages')
        d = web.draft_claims(question, results, bar)
        drafted = d["claims"]
        trace("DECIDE", f"{len(drafted)} claims drafted | enough: {d['enough']} | reason: {d['reason']}")
        queries = []
        if not d["enough"] and rounds == 1 and web.SEARCHES["run"] < MAX_SEARCHES and d["next_queries"]:
            queries = d["next_queries"]
            trace("DECIDE", f"one more round: {queries}")
        elif not d["enough"]:
            trace("DECIDE", "stop: no more rounds; report what passed the check")

    # 4. Check
    passed, failed = [], []
    for c in drafted:
        ok = check.check_claim(c, check.page_text_for(c["source_url"], results))
        (passed if ok else failed).append(c)
        trace("CHECK", f'{"PASS" if ok else "FAIL"} {c["source_url"]} | "{c["quote"]}"')

    # 5. Write and 6. Reflect: one brain version per run
    if use_brain:
        with brain.batch():
            added, merged = brain.add_claims(passed, topic, model)
            brain.add_run({"question": question, "topic": topic, "searches": web.SEARCHES["run"],
                           "passed": len(passed), "added": added, "merged": merged,
                           "failed": [{"text": c["text"], "url": c["source_url"], "quote": c["quote"]} for c in failed],
                           "evidence_bar": bar, "model": model})
            trace("WRITE", f"{added} claims added, {merged} merged, {len(failed)} failed logged | brain version {brain.version()}")
            trace("REFLECT", selfmodel.update(topic, [True] * len(passed) + [False] * len(failed),
                                              web.SEARCHES["run"], model))
    else:
        trace("WRITE", "brain off: nothing written")

    # 7. Report
    claims = covered + passed
    report = _report(question, topic, model, bar, claims, covered, passed, failed, stamp)
    trace("REPORT", f"{report.relative_to(llm.ROOT)} | {len(claims)} cited claims, {web.SEARCHES['run']} searches")
    return {"report_path": str(report), "trace_path": str(trace.path), "searches": web.SEARCHES["run"],
            "known": len(covered), "new": len(passed), "dropped": len(failed), "evidence_bar": bar,
            "topic": topic, "model": model, "citations": sorted({c["source_url"] for c in claims})}


def _report(question, topic, model, bar, claims, covered, passed, failed, stamp):
    if claims:
        numbered = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(claims))
        answer = llm.chat_text(ANSWER_SYSTEM, f"Question: {question}\n\nClaims:\n{numbered}")
    else:
        answer = "No claim passed the quote check, so DDT gives no answer to this question."
    n = {id(c): i + 1 for i, c in enumerate(claims)}
    lines = [f"# {question}", "",
             f"Date: {datetime.now().isoformat(timespec='seconds')} | Model: {model} | Topic: {topic} | Evidence bar: {bar}",
             "", "## Answer", "", answer, "", "## Sources", ""]
    lines += [f'[{n[id(c)]}] {c["source_url"]} — "{c["quote"]}"' for c in claims] or ["None."]
    lines += ["", "## Known before this run", ""]
    lines += [f"- [{n[id(c)]}] {c['text']} (claim {c['id']}, similarity {c['similarity']})" for c in covered] or ["None."]
    lines += ["", "## New this run", ""]
    lines += [f"- [{n[id(c)]}] {c['text']}" for c in passed] or ["None."]
    lines += ["", "## Dropped: failed the quote check", ""]
    lines += [f'- {c["text"]} | {c["source_url"]} | "{c["quote"]}"' for c in failed] or ["None."]
    d = llm.ROOT / "reports"
    d.mkdir(exist_ok=True)
    path = _unique(d / f"{stamp}-{_slug(question)}.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
