"""DDT command line.

python run.py "your question" [--topic T] [--no-brain]
python run.py --rollback [N]
python run.py --stats
"""
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from ddt import agent, brain, metacog, selfmodel  # noqa: E402
from ddt.stores import episodes, gaps, tools  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="DDT: a research agent with a brain outside the model.")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--topic")
    ap.add_argument("--no-brain", action="store_true", help="do not read or write the brain")
    ap.add_argument("--rollback", nargs="?", type=int, const=1, metavar="N", help="undo the last N brain versions")
    ap.add_argument("--stats", action="store_true", help="print the brain counts and the self-model")
    ap.add_argument("--reindex", action="store_true", help="explicitly rebuild all brain vectors with the selected embedding model (may call Nebius)")
    a = ap.parse_args()
    if a.reindex:
        print(f"brain reindexed at version {brain.reindex()}: {brain.counts()}")
    elif a.rollback is not None:
        print(f"brain rolled back to version {brain.rollback(a.rollback)}: {brain.counts()}")
    elif a.stats:
        print(f"brain: {brain.counts()}")
        print(f"{'topic':<30} {'runs':>4} {'checked':>7} {'passed':>6} {'searches':>8} {'bar':>3}  last5  models")
        for t, e in selfmodel.load().items():
            last5 = "".join("P" if r else "F" for r in e["last5"])
            print(f"{t:<30} {e['runs']:>4} {e['checked']:>7} {e['passed']:>6} {e['searches']:>8} "
                  f"{e['evidence_bar']:>3}  {last5:<5}  {', '.join(e['models'])}")
        g = gaps.all()
        print(f"harness: episodes {episodes.count()}, gaps open {sum(x['status'] == 'open' for x in g)}, "
              f"gaps closed {sum(x['status'] == 'closed' for x in g)}, tool memory {tools.count()} queries")
        m = metacog.load()
        print(f"metacognitive weights w: {[round(w, 3) for w in m['w']]} ({len(m['log'])} claims learned)")
        last = episodes.all()[-10:]
        if last:
            se = sum(e["error"]["searches"] for e in last) / len(last)
            pr = [e["error"]["pass_rate"] for e in last if e["error"]["pass_rate"] is not None]
            print(f"mean prediction error, last {len(last)} episodes: searches {se:.3f}, "
                  f"pass rate {sum(pr) / len(pr):.3f}" if pr else
                  f"mean prediction error, last {len(last)} episodes: searches {se:.3f}, pass rate n/a")
    elif a.question:
        r = agent.research(a.question, topic=a.topic, use_brain=not a.no_brain)
        print("[RESULT] " + json.dumps(r))
    else:
        ap.print_help()


if __name__ == "__main__":
    # Stats span several stores; keep that view consistent with research/undo.
    with brain.lock():
        main()
