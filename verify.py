"""Runs checks S2 to S7 and both controls, prints PASS, FAIL or NOT TRIGGERED with the measured numbers,
and appends a dated section to the results file.

python verify.py           live: Nebius Token Factory and Tavily, at most 40 Tavily searches
python verify.py --fake    the same checks with the offline stand-ins (no network)

Brains used (all created fresh): brain_verify (S3, S4, S5), brain_control (failure control),
brain_s6 (a copy of brain_verify for S6), brain_s5fake (S5 stand-in), brain_s7 (S7).
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SEARCH_CAP = 40
MADE_UP_QUOTE = "the purple elephant negotiated quarterly tariffs with seventeen lunar bakeries on tuesday morning"
RESULTS_FILE = ROOT.parent / "Testing" / "results.md" if (ROOT.parent / "Testing").is_dir() else ROOT / "results.md"

lines: list[str] = []


def say(text: str) -> None:
    lines.append(text)
    print(text, flush=True)


def fresh_brain(name: str) -> None:
    shutil.rmtree(ROOT / name, ignore_errors=True)
    os.environ["DDT_BRAIN_DIR"] = name


def s3_rule(first: dict, second: dict) -> bool:
    """The second run answers from the brain: at most 1 search, at least 1 known claim, no new citation."""
    return second["searches"] <= 1 and second["known"] > 0 and set(second["citations"]) <= set(first["citations"])


def describe(q: str, first: dict, second: dict) -> str:
    return (f'  "{q}": searches {first["searches"]} -> {second["searches"]}, citations {len(first["citations"])} -> '
            f'{len(second["citations"])}, known claims used {second["known"]}, subset '
            f'{"yes" if set(second["citations"]) <= set(first["citations"]) else "no"}, rule {"holds" if s3_rule(first, second) else "fails"}')


def sources_of(report: str) -> list[tuple[str, str]]:
    text = Path(report).read_text(encoding="utf-8")
    return re.findall(r'^\[\d+\] (\S+) — "(.*)"$', text, flags=re.M)


def s5_check(agent, question: str, topic: str) -> tuple[str, str]:
    first = agent.research(question, topic=topic)
    raised = f'evidence bar for "{topic}": 1 -> 2' in Path(first["trace_path"]).read_text(encoding="utf-8")
    second = agent.research(question, topic=topic)
    shown = f'[READ] evidence bar for "{topic}": 2' in Path(second["trace_path"]).read_text(encoding="utf-8")
    numbers = (f'first run: {first["new"]} passed, {first["dropped"]} dropped, bar raised {"yes" if raised else "no"}; '
               f'second run trace shows evidence bar 2: {"yes" if shown else "no"}')
    if not raised:
        return "NOT TRIGGERED", numbers
    return ("PASS" if shown else "FAIL"), numbers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use the offline stand-ins")
    a = ap.parse_args()
    if a.fake:
        os.environ["DDT_FAKE"] = "1"
    from ddt import agent, brain, check, llm, research
    research.SEARCH_CAP = SEARCH_CAP
    qs = json.loads((ROOT / "tests" / "questions.json").read_text(encoding="utf-8"))
    A, B = qs["A"], qs["B"]
    stamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    say(f"verify.py run {stamp} | {'stand-ins (DDT_FAKE=1)' if a.fake else 'live'} | model {llm.MODEL_ID} | "
        f"embedding {os.environ.get('NEBIUS_EMBED_MODEL')} | Tavily cap {SEARCH_CAP} searches")
    results: dict[str, str] = {}
    first, second = {}, {}

    # S5 stand-in first, so its stand-in searches do not count against the live cap
    os.environ["DDT_FAKE"] = "1"
    fresh_brain("brain_s5fake")
    status, numbers = s5_check(agent, "What happened in the unreliable topic?", "unreliable topic")
    results["S5 stand-in"] = status
    say(f"S5 stand-in {status} | {numbers}")
    if not a.fake:
        os.environ.pop("DDT_FAKE")
    research.SEARCHES["total"] = 0

    try:
        # S3: each set A question with the brain, then each again
        say("--- S3 runs ---")
        fresh_brain("brain_verify")
        shutil.rmtree(ROOT / "brain_s6", ignore_errors=True)
        for q in A["questions"]:
            first[q] = agent.research(q, topic=A["topic"])
        for q in A["questions"]:
            second[q] = agent.research(q, topic=A["topic"])
        ok = all(s3_rule(first[q], second[q]) for q in A["questions"])
        results["S3"] = "PASS" if ok else "FAIL"
        say(f"S3 {results['S3']} | set A, {len(A['questions'])} questions")
        for q in A["questions"]:
            say(describe(q, first[q], second[q]))

        # Failure control: the same rule without the brain must fail
        say("--- failure control runs ---")
        fresh_brain("brain_control")
        q = A["questions"][0]
        c1 = agent.research(q, topic=A["topic"], use_brain=False)
        c2 = agent.research(q, topic=A["topic"], use_brain=False)
        rule = s3_rule(c1, c2)
        results["Failure control"] = "PASS" if not rule and c2["searches"] >= 3 else "CONTROL BROKEN"
        say(f"Failure control {results['Failure control']} | brain off, 1 question: S3 rule "
            f"{'holds' if rule else 'fails'} as it must; second run searches {c2['searches']} (needs 3 or more)")
        say(describe(q, c1, c2))

        # S4: research set B, then ask set A again
        say("--- S4 runs ---")
        os.environ["DDT_BRAIN_DIR"] = "brain_verify"
        for q in B["questions"]:
            agent.research(q, topic=B["topic"])
            if [e for e in brain.entries() if e["kind"] == "run"][-1]["added"] > 0:  # S6 needs a run that added claims
                shutil.rmtree(ROOT / "brain_s6", ignore_errors=True)
                shutil.copytree(ROOT / "brain_verify", ROOT / "brain_s6")
        third = {q: agent.research(q, topic=A["topic"]) for q in A["questions"]}
        ok = all(s3_rule(first[q], third[q]) for q in A["questions"])
        results["S4"] = "PASS" if ok else "FAIL"
        say(f"S4 {results['S4']} | after set B ({len(B['questions'])} questions), set A again:")
        for q in A["questions"]:
            say(describe(q, first[q], third[q]))
    except RuntimeError as e:
        say(f"STOPPED: {e}")

    # S2 and the citation control: re-fetch the sources of the latest report with 3 or more sources
    reports = [r["report_path"] for r in list(first.values()) + list(second.values())]
    reports = sorted({r for r in reports if len(sources_of(r)) >= 3}, key=os.path.getmtime)
    if reports:
        picked = random.sample(sources_of(reports[-1]), 3)
        passed, page = 0, ""
        for url, quote in picked:
            try:
                fetched = research.extract(url)
            except Exception as e:
                fetched = ""
                say(f"  S2 extract failed for {url}: {type(e).__name__}")
            page = page or fetched
            ok = check.check_claim({"quote": quote}, fetched)
            passed += ok
            say(f'  S2 {"pass" if ok else "fail"}: {url} | "{quote}"')
        results["S2"] = "PASS" if passed == 3 else "FAIL"
        say(f"S2 {results['S2']} | {passed} of 3 random claims from {Path(reports[-1]).name} found in their re-fetched source")
        made_up = check.check_claim({"quote": MADE_UP_QUOTE}, page)
        results["Citation control"] = "PASS" if not made_up and page else "CONTROL BROKEN"
        say(f"Citation control {results['Citation control']} | made-up quote against {picked[0][0]} "
            f"({len(page)} characters fetched): check_claim returned {made_up}")
    else:
        results["S2"] = results["Citation control"] = "NOT RUN"
        say("S2 NOT RUN | no report with 3 or more sources")
        say("Citation control NOT RUN")

    # S6: roll back one version on a copy of the brain taken right after a set B run that added claims
    if (ROOT / "brain_s6" / "brain.jsonl").exists():
        os.environ["DDT_BRAIN_DIR"] = "brain_s6"
        v = brain.version()
        before = {e["id"]: e for e in brain.entries() if e["kind"] == "claim"}
        last = [e for e in brain.entries() if e["kind"] == "run"][-1]
        brain.rollback(1)
        after = {e["id"] for e in brain.entries() if e["kind"] == "claim"}
        removed = set(before) - after
        expected = {i for i, e in before.items() if e["version"] == v}
        ok = last["added"] > 0 and len(removed) == last["added"] and removed == expected and after <= set(before)
        results["S6"] = "PASS" if ok else "FAIL"
        say(f'S6 {results["S6"]} | claims {len(before)} -> {len(after)} after rollback 1 (version {v} -> {brain.version()}); '
            f'the last run ("{last["question"]}") added {last["added"]}; removed exactly its claims: '
            f'{"yes" if removed == expected else "no"}; earlier claims kept: {"yes" if after <= set(before) else "no"}')
    else:
        results["S6"] = "NOT RUN"
        say("S6 NOT RUN | no set B run added claims")

    # S5 live: a topic where quotes often fail, run twice
    os.environ["DDT_BRAIN_DIR"] = "brain_verify"
    try:
        status, numbers = s5_check(agent, qs["S5"]["question"], qs["S5"]["topic"])
    except RuntimeError as e:
        status, numbers = "NOT RUN", str(e)
    results["S5"] = status
    say(f"S5 {status} | topic \"{qs['S5']['topic']}\": {numbers}")

    # S7: a second Token Factory model, S3 rule on one question, no code change (a separate process)
    left = SEARCH_CAP - research.SEARCHES["total"]
    if left >= 4:
        env = {**os.environ, "NEBIUS_MODEL": qs["S7"]["model"], "DDT_BRAIN_DIR": "brain_s7",
               "DDT_SEARCH_CAP": str(left), "PYTHONIOENCODING": "utf-8"}
        shutil.rmtree(ROOT / "brain_s7", ignore_errors=True)
        runs = []
        for _ in range(2):
            out = subprocess.run([sys.executable, "run.py", qs["S7"]["question"], "--topic", A["topic"]],
                                 cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            res = [l for l in out.stdout.splitlines() if l.startswith("[RESULT] ")]
            runs.append(json.loads(res[0][9:]) if res else None)
        if all(runs):
            research.SEARCHES["total"] += runs[0]["searches"] + runs[1]["searches"]
            results["S7"] = "PASS" if s3_rule(*runs) else "FAIL"
            say(f"S7 {results['S7']} | model {qs['S7']['model']}")
            say(describe(qs["S7"]["question"], *runs))
        else:
            results["S7"] = "FAIL"
            say(f"S7 FAIL | run.py with model {qs['S7']['model']} gave no result: {out.stderr[-300:]}")
    else:
        results["S7"] = "NOT RUN"
        say(f"S7 NOT RUN | only {left} searches left under the cap")

    say(f"Tavily searches used: {research.SEARCHES['total']} of {SEARCH_CAP}")
    if "CONTROL BROKEN" in results.values():
        say("CONTROL BROKEN: every result of this run is void.")
    say("Summary: " + ", ".join(f"{k} {v}" for k, v in results.items()))
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp} ({'stand-ins' if a.fake else 'live'})\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
