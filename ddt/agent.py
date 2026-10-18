"""The DDT loop: Read, Predict, Plan, Act, Check, Write, Reflect, Compare, Report."""
import re
from datetime import datetime

from . import brain, check, llm, metacog, selfmodel
from .stores import episodes, gaps, tools
from . import research as web

MAX_SEARCHES = 6
ANSWERED = 0.75  # a claim at this cosine similarity or more answers a part or a gap

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
    qvec = llm.embed([question])[0]

    # 1. Read: memory (claims), state (open gaps), tool memory (useful past queries)
    known, open_gaps, past_queries = [], [], []
    if use_brain:
        known = brain.retrieve(question, vec=qvec)
        trace("READ", f"{len(known)} known claims at similarity 0.75 or more (brain version {brain.version()})")
        for c in known:
            trace("READ", f"  {c['id']} sim {c['similarity']}: {c['text']}")
        if topic is None and known:
            topic = known[0]["topic"]
            trace("READ", f'topic from the closest known claim: "{topic}"')
        if topic is not None:
            trace("READ", f'evidence bar for "{topic}": {selfmodel.get(topic)["evidence_bar"]}')
        open_gaps = [(g, s) for g, s in gaps.search(qvec, k=10, min_sim=0.6) if g["status"] == "open"][:5]
        trace("STATE", "open from before: " + ("; ".join(f'{g["id"]} sim {s}: {g["text"]}' for g, s in open_gaps) or "none"))
        for t, s in tools.search(qvec, k=20, min_sim=0.6):
            if t["yield"] >= 1 and t["query"] not in past_queries and len(past_queries) < 3:
                past_queries.append(t["query"])
        trace("TOOLS", f"past queries offered: {past_queries or 'none'}")

    # 2. Predict: the inner environment forecasts the run before acting
    pred = metacog.predict(qvec, len(known), topic or "", use_memory=use_brain)
    trace("PREDICT", f"expect pass rate {pred['pass_rate']}, {pred['searches']} searches, {pred['known']} known claims "
                     f"(from {pred['basis']} similar episodes{': ' + ', '.join(pred['neighbours']) if pred['neighbours'] else ''})")

    # 3. Plan, with the evidence bar already raised if the prediction is weak (regulation rule 1)
    low = pred["pass_rate"] < 0.6
    bar = selfmodel.get(topic)["evidence_bar"] if use_brain and topic else 1
    if low and bar < 2:
        bar = 2
        trace("META", f"predicted pass rate {pred['pass_rate']} is below 0.6: evidence bar raised to 2 for this run")
    p = web.plan(question, known, bar, [g["text"] for g, _ in open_gaps], past_queries)
    if topic is None:
        topic = p["topic"] or "general"
        bar = max(selfmodel.get(topic)["evidence_bar"] if use_brain else 1, 2 if low else 1)
        trace("PLAN", f'topic from the plan: "{topic}" | evidence bar {bar}')
    covered = [c for c in known if c["id"] in p["covered"]]
    trace("PLAN", f"parts: {p['parts']}")
    if known and not covered and not p["queries"]:  # a plan that neither uses memory nor searches contradicts itself
        covered, p["queries"] = repair_plan(p["parts"], known)
        trace("META", f"plan had no covered claims and no queries: repaired by cosine {ANSWERED} or more: "
                      f"covered {[c['id'] for c in covered] or 'none'}, queries {p['queries'] or 'none'}")
    if covered and p["queries"]:  # regulation rule 3: queries only for parts no known claim answers
        sims = llm.embed(p["parts"]) @ llm.embed([c["text"] for c in covered]).T
        if float(sims.max(axis=1).min()) >= ANSWERED:
            trace("META", f"every part is answered by a covered claim (cosine {ANSWERED} or more): queries dropped {p['queries']}")
            p["queries"] = []
    trace("PLAN", f'topic "{topic}" | covered by known claims: {[c["id"] for c in covered] or "none"}')
    trace("PLAN", f"queries: {p['queries'] or 'none'}")

    # 4. Act
    results, seen, drafted, queries, hits_by_query = [], set(), [], p["queries"], {}
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
            hits_by_query[q] = {h["url"] for h in hits}
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

    # 5. Check, with the online claim model's prediction before each check and its update after
    passed, failed = [], []
    state, domains = metacog.load(), metacog.domain_pass_rates()
    cvecs = llm.embed([c["text"] for c in drafted]) if drafted else []
    for c, cv in zip(drafted, cvecs):
        x = metacog.features(c, cv, qvec, domains, topic)
        pp = metacog.p_pass(x, state["w"])
        trace("META", f"p_pass {pp:.2f}" + (" | low confidence" if pp < 0.2 else ""))
        ok = check.check_claim(c, check.page_text_for(c["source_url"], results))
        (passed if ok else failed).append(c)
        trace("CHECK", f'{"PASS" if ok else "FAIL"} {c["source_url"]} | "{c["quote"]}"')
        if use_brain:
            metacog.learn(x, ok, state)

    used = covered + passed
    n = len(passed) + len(failed)
    actual = {"searches": web.SEARCHES["run"], "passed": len(passed), "failed": len(failed),
              "pass_rate": round(len(passed) / n, 3) if n else None}
    err = metacog.compare(pred, actual)

    # 6. Write: claims, run, episode, tool memory, gaps and weights, all in one brain version
    opened = closed = 0
    if use_brain:
        with brain.batch():
            added, merged = brain.add_claims(passed, topic, model)
            brain.add_run({"question": question, "topic": topic, "searches": web.SEARCHES["run"],
                           "passed": len(passed), "added": added, "merged": merged,
                           "failed": [{"text": c["text"], "url": c["source_url"], "quote": c["quote"]} for c in failed],
                           "evidence_bar": bar, "model": model})
            metacog.save(state)
            uvecs = llm.embed([c["text"] for c in used]) if used else None
            for g in gaps.all():  # close earlier gaps that a checked claim now answers
                if g["status"] == "open" and uvecs is not None and float((uvecs @ gaps.vectors_of([g["id"]])[0]).max()) >= ANSWERED:
                    gaps.update(g["id"], status="closed", closed_by_question=question, closed_version=brain.version())
                    closed += 1
            pvecs = llm.embed(p["parts"])
            for part, pv in zip(p["parts"], pvecs):  # open a gap for every part no checked claim answers
                if uvecs is not None and float((uvecs @ pv).max()) >= ANSWERED:
                    continue
                if any(g["status"] == "open" for g, _ in gaps.search(pv, k=1, min_sim=0.95)):
                    continue
                gaps.add({"text": part, "question": question, "topic": topic, "status": "open"}, pv)
                opened += 1
            if hits_by_query:
                qvecs = llm.embed(list(hits_by_query))
                for (q, urls), qv in zip(hits_by_query.items(), qvecs):
                    tools.add({"query": q, "question": question, "topic": topic, "results": len(urls),
                               "yield": sum(c["source_url"] in urls for c in passed)}, qv)
            episodes.add({"question": question, "topic": topic, "parts": p["parts"], "known": len(known),
                          "claims_used": [c["id"] for c in covered], "new_claims": len(passed),
                          "prediction": pred, "actual": actual, "error": err, "model": model}, qvec)
            trace("WRITE", f"{added} claims added, {merged} merged, {len(failed)} failed logged | brain version {brain.version()}")
            trace("STATE", f"gaps opened: {opened}, closed: {closed}")
            trace("REFLECT", selfmodel.update(topic, [True] * len(passed) + [False] * len(failed),
                                              web.SEARCHES["run"], model))
    else:
        trace("WRITE", "brain off: nothing written")

    # 7. Compare: the prediction against what happened
    pr = "n/a (no new claims)" if actual["pass_rate"] is None else f"{actual['pass_rate']} (error {err['pass_rate']})"
    trace("COMPARE", f"pass rate predicted {pred['pass_rate']} vs actual {pr}; "
                     f"searches predicted {pred['searches']} vs actual {actual['searches']} (error {err['searches']})")

    # 8. Report
    harness = {"prediction": pred, "actual": actual, "error": err, "opened": opened, "closed": closed,
               "open_from_before": [g["id"] for g, _ in open_gaps], "past_queries": past_queries}
    report = _report(question, topic, model, bar, used, covered, passed, failed, stamp, harness)
    trace("REPORT", f"{report.relative_to(llm.ROOT)} | {len(used)} cited claims, {web.SEARCHES['run']} searches")
    return {"report_path": str(report), "trace_path": str(trace.path), "searches": web.SEARCHES["run"],
            "known": len(covered), "new": len(passed), "dropped": len(failed), "evidence_bar": bar,
            "topic": topic, "model": model, "citations": sorted({c["source_url"] for c in used}), **harness}


def repair_plan(parts: list[str], known: list[dict]) -> tuple[list[dict], list[str]]:
    """Covered: the known claims that answer a part. Queries: the parts no known claim answers (at most 3)."""
    sims = llm.embed(parts) @ llm.embed([c["text"] for c in known]).T
    covered = [c for j, c in enumerate(known) if float(sims[:, j].max()) >= ANSWERED]
    return covered, [part for i, part in enumerate(parts) if float(sims[i].max()) < ANSWERED][:3]


def _report(question, topic, model, bar, claims, covered, passed, failed, stamp, harness):
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
    pr, ac, er = harness["prediction"], harness["actual"], harness["error"]
    lines += ["", "## Harness", "",
              f"- Predicted: pass rate {pr['pass_rate']}, {pr['searches']} searches (from {pr['basis']} similar episodes)",
              f"- Actual: pass rate {ac['pass_rate'] if ac['pass_rate'] is not None else 'n/a (no new claims)'}, {ac['searches']} searches",
              f"- Error: pass rate {er['pass_rate'] if er['pass_rate'] is not None else 'n/a'}, searches {er['searches']}",
              f"- Gaps: {len(harness['open_from_before'])} open from before, {harness['opened']} opened, {harness['closed']} closed",
              f"- Past queries offered: {', '.join(harness['past_queries']) or 'none'}"]
    d = llm.ROOT / "reports"
    d.mkdir(exist_ok=True)
    path = _unique(d / f"{stamp}-{_slug(question)}.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
