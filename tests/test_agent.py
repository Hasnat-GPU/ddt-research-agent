from pathlib import Path

from ddt import agent, brain, metacog
from ddt.stores import gaps
from tests import fakes

Q1 = "What sizes does the Nemotron family come in and what are the Nemotron benchmark results?"
Q2 = "What are the Nemotron benchmark results?"


def test_second_run_answers_from_the_brain():
    first = agent.research("What sizes does the Nemotron family come in?", topic="nvidia nemotron")
    assert first["searches"] == 3 and first["new"] > 0
    second = agent.research("What sizes does the Nemotron family come in?", topic="nvidia nemotron")
    assert second["searches"] == 0 and set(second["citations"]) <= set(first["citations"])
    assert brain.counts()["version"] == 2  # one brain version per run
    trace = Path(second["trace_path"]).read_text(encoding="utf-8")
    assert "[PREDICT]" in trace and "[COMPARE]" in trace
    # HANDOFF-2 predictor: the neighbour (run 1) searched 3 times, so it predicts 3; run 2 searched 0
    assert second["prediction"]["searches"] == 3.0 and second["error"]["searches"] == 3.0
    assert len(metacog.load()["log"]) == first["new"] + first["dropped"]


def test_failed_quotes_raise_the_evidence_bar():
    first = agent.research("What happened in the unreliable topic?", topic="unreliable topic")
    assert first["dropped"] == 2
    second = agent.research("What happened in the unreliable topic?", topic="unreliable topic")
    assert 'evidence bar for "unreliable topic": 2' in open(second["trace_path"], encoding="utf-8").read()


def test_a_gap_opened_in_run_1_is_retrieved_and_closed_in_run_2(monkeypatch):
    first = agent.research(Q1, topic="nvidia nemotron")
    assert first["opened"] == 1
    gap = [g for g in gaps.all() if g["status"] == "open"][0]
    monkeypatch.setattr(fakes, "LATE_ONLINE", True)  # the stand-in web gains the benchmark page
    second = agent.research(Q2, topic="nvidia nemotron")
    assert gap["id"] in second["open_from_before"] and second["closed"] == 1
    assert [g["status"] for g in gaps.all() if g["id"] == gap["id"]] == ["closed"]


def test_brain_off_retrieves_no_gap_and_does_not_learn(monkeypatch):
    agent.research(Q1, topic="nvidia nemotron")
    log = len(metacog.load()["log"])
    monkeypatch.setattr(fakes, "LATE_ONLINE", True)
    off = agent.research(Q2, topic="nvidia nemotron", use_brain=False)
    assert off["open_from_before"] == [] and off["closed"] == 0
    assert len(metacog.load()["log"]) == log
