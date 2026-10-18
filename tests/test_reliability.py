"""Regressions for the reviewed persistence, evidence and validation failures."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ddt import agent, brain, llm, metacog, persistence, research, selfmodel
from ddt.stores import episodes, gaps, tools
from tests import fakes
from tests.test_brain import A, B
from tests.test_research import LICENCE_CLAIM, LICENCE_PART, SIZE_CLAIM, _embed


def state_bytes(root):
    return {f: (root / f).read_bytes() if (root / f).exists() else None for f in brain.FILES + ["meta.json"]}


def fill(label):
    with brain.batch():
        brain.add_claims([A if label == "first" else B], "topic", "fake")
        brain.add_run({"question": label, "checks": [{"url": A["source_url"], "passed": True}]})
        vec = llm.embed([label])[0]
        episodes.add({"question": label}, vec)
        gaps.add({"text": label, "status": "open"}, vec)
        tools.add({"query": label, "yield": 1}, vec)
        metacog.learn([1, 0, 0, 0, 0, 0], True)
        selfmodel.update("topic", [True], 1, "fake")


@pytest.mark.parametrize("filename", brain.FILES + ["meta.json"])
def test_publication_failure_restores_every_file(monkeypatch, filename):
    fill("first")
    root = brain.brain_dir()
    before = state_bytes(root)
    original = persistence.atomic_bytes
    failed = False

    def fail_after_write(path, data):
        nonlocal failed
        original(path, data)
        if path.parent == root and path.name == filename and not failed:
            failed = True
            raise OSError("injected disk failure")

    monkeypatch.setattr(persistence, "atomic_bytes", fail_after_write)
    with pytest.raises(OSError, match="injected"):
        fill("second")
    assert failed and state_bytes(root) == before
    assert not (root / ".transaction").exists()


def test_staging_failure_and_nested_batch_preserve_state():
    fill("first")
    root = brain.brain_dir()
    before = state_bytes(root)
    with pytest.raises(RuntimeError, match="nested"):
        with brain.batch():
            brain.add_claims([B], "topic", "fake")
            with brain.batch():
                pass
    assert state_bytes(root) == before
    with pytest.raises(RuntimeError, match="alignment"):
        with brain.batch():
            (brain.brain_dir() / "brain.jsonl").write_text("", encoding="utf-8")
    assert state_bytes(root) == before


@pytest.mark.parametrize("committed", [False, True])
def test_process_interruption_recovers_committed_boundary(committed):
    brain.add_claims([A], "topic", "fake")
    root = brain.brain_dir()
    before = state_bytes(root)
    code = '''
import os
from ddt import brain, persistence
from tests.test_brain import B
root = brain.brain_dir()
original = persistence.atomic_bytes
def interrupted(path, data):
    original(path, data)
    if COMMITTED:
        if path.name == "intent.json" and b"committed" in data:
            os._exit(91)
    elif path.parent == root and path.name == "brain.jsonl":
        os._exit(91)
persistence.atomic_bytes = interrupted
brain.add_claims([B], "topic", "fake")
'''.replace("COMMITTED", repr(committed))
    child = subprocess.run([sys.executable, "-B", "-c", code], cwd=llm.ROOT, capture_output=True, text=True)
    assert child.returncode == 91, child.stderr
    assert (root / ".transaction").exists()
    assert brain.counts()["claims"] == (2 if committed else 1)
    if not committed:
        assert state_bytes(root) == before
    assert not (root / ".transaction").exists()


def test_cross_process_lock_and_positive_rollback_steps():
    brain.add_claims([A], "topic", "fake")
    before = state_bytes(brain.brain_dir())
    code = 'from ddt import brain; brain.rollback(1)'
    with brain.lock():
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=llm.ROOT, capture_output=True, text=True)
    assert child.returncode != 0 and "brain is busy" in child.stderr
    assert state_bytes(brain.brain_dir()) == before
    for steps in [0, -1, True]:
        with pytest.raises(ValueError, match="positive"):
            brain.rollback(steps)


def test_rollback_restores_all_stores_and_embedding_identity():
    fill("first")
    root = brain.brain_dir()
    before = state_bytes(root)
    fill("second")
    brain.rollback(1)
    assert state_bytes(root) == before


def test_coverage_requires_distinct_sources_and_repairs_partial_question(monkeypatch):
    monkeypatch.setattr(llm, "embed", _embed)
    assert agent.repair_plan([LICENCE_PART], [LICENCE_CLAIM], 2)[1] == [LICENCE_PART]
    duplicate = {**LICENCE_CLAIM, "id": "duplicate"}
    assert agent.repair_plan([LICENCE_PART], [LICENCE_CLAIM, duplicate], 2)[1] == [LICENCE_PART]
    other = {**duplicate, "source_url": "https://other.example/license"}
    assert agent.repair_plan([LICENCE_PART], [LICENCE_CLAIM, other], 2)[1] == []
    assert agent.repair_plan(["What sizes?", LICENCE_PART], [SIZE_CLAIM], 1)[1] == [LICENCE_PART]


def test_planner_gets_source_identity(monkeypatch):
    captured = {}
    def response(system, user):
        captured.update(json.loads(user))
        return {"topic": "t", "parts": [LICENCE_PART], "covered": [LICENCE_CLAIM["id"]], "queries": []}
    monkeypatch.setattr(fakes, "chat_json", response)
    research.plan(LICENCE_PART, [LICENCE_CLAIM], 2)
    assert captured["known_claims"][0]["source_url"] == LICENCE_CLAIM["source_url"]
    assert captured["evidence_bar"] == 2


def test_explicit_planner_search_is_not_replaced_by_similar_claims(monkeypatch):
    brain.add_claims([A], "nvidia nemotron", "fake")
    monkeypatch.setattr(research, "plan", lambda *args: {"topic": "nvidia nemotron", "parts": [LICENCE_PART],
                                                       "covered": [], "queries": [LICENCE_PART]})
    result = agent.research(LICENCE_PART, topic="nvidia nemotron")
    assert result["searches"] == 1 and result["known"] == 0


def test_raised_bar_keeps_search_and_reports_incomplete_checked_evidence(monkeypatch):
    brain.add_claims([A], "nvidia nemotron", "fake")
    selfmodel.update("nvidia nemotron", [False, False, True], 0, "fake")
    result = agent.research("What sizes does the Nemotron family come in?", topic="nvidia nemotron")
    assert result["evidence_bar"] == 2 and result["searches"] > 0
    # The unreliable topic supplies only one passing source even though its draft says enough.
    selfmodel.update("unreliable topic", [False, False, True], 0, "fake")
    incomplete = agent.research("What happened in the unreliable topic?", topic="unreliable topic")
    assert not incomplete["complete"] and incomplete["unresolved_parts"]
    assert "Evidence still needed" in Path(incomplete["report_path"]).read_text(encoding="utf-8")


def test_legacy_embedding_is_preserved_until_explicit_migration():
    root = brain.brain_dir()
    np.save(root / "brain.npy", np.ones((1, 512), dtype=np.float32))
    before = state_bytes(root)
    with pytest.raises(RuntimeError, match="legacy brain"):
        llm.embed(["question"])
    assert state_bytes(root) == before


def test_explicit_reindex_preserves_ids_and_can_be_rolled_back(monkeypatch):
    fill("first")
    root = brain.brain_dir()
    before = state_bytes(root)
    records = {name: (root / name).read_bytes() for name in ("brain.jsonl", "episodes.jsonl", "gaps.jsonl", "tools.jsonl")}
    monkeypatch.setenv("DDT_FAKE", "0")
    monkeypatch.setenv("DDT_EMBED", "local")
    assert brain.reindex() == 2
    assert json.loads((root / "embedding.json").read_text())["provider"] == "local"
    assert all((root / name).read_bytes() == data for name, data in records.items())
    llm.embed(["question"])  # the new identity is accepted
    brain.rollback(1)
    assert state_bytes(root) == before


def test_reindex_failure_does_not_publish_partial_vectors(monkeypatch):
    fill("first")
    root = brain.brain_dir()
    before = state_bytes(root)
    original = llm.embed
    calls = 0
    def interrupted(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("embedding unavailable")
        return original(texts)
    monkeypatch.setattr(llm, "embed", interrupted)
    with pytest.raises(ConnectionError):
        brain.reindex()
    assert state_bytes(root) == before


@pytest.mark.parametrize("patch", [{"queries": "abc"}, {"queries": [""]}, {"parts": []},
                                   {"covered": ["unknown"]}, {"queries": ["a", "b", "c", "d"]}])
def test_invalid_plan_is_retried_then_rejected(monkeypatch, patch):
    calls = []
    def response(*args):
        calls.append(1)
        return {"topic": "t", "parts": ["part"], "covered": [], "queries": [], **patch}
    monkeypatch.setattr(fakes, "chat_json", response)
    with pytest.raises(ValueError, match="two attempts"):
        research.plan("q", [], 1)
    assert len(calls) == 2


@pytest.mark.parametrize("patch", [{"enough": "false"}, {"confidence": "0.9"}, {"confidence": float("nan")},
                                   {"confidence": 1.1}, {"source_url": "https://unknown.example"}])
def test_invalid_draft_is_rejected(monkeypatch, patch):
    claim = {**A, **{k: v for k, v in patch.items() if k != "enough"}}
    response = {"claims": [claim], "enough": patch.get("enough", True), "reason": "reason", "next_queries": []}
    monkeypatch.setattr(fakes, "chat_json", lambda *args: response)
    with pytest.raises(ValueError, match="two attempts"):
        research.draft_claims("q", [{"url": A["source_url"], "title": "a", "content": "source", "raw_content": ""}], 1)


def test_corrected_response_succeeds_and_invalid_run_does_not_commit(monkeypatch):
    responses = iter([{"queries": "bad"}, {"topic": "t", "parts": ["part"], "covered": [], "queries": ["query"]}])
    monkeypatch.setattr(fakes, "chat_json", lambda *args: next(responses))
    assert research.plan("q", [], 1)["queries"] == ["query"]
    monkeypatch.setattr(fakes, "chat_json", lambda *args: {"queries": "bad"})
    root = brain.brain_dir()
    before = state_bytes(root)
    with pytest.raises(ValueError):
        agent.research("a valid public question")
    assert state_bytes(root) == before
    with pytest.raises(ValueError, match="question"):
        agent.research("   ")


def test_eligible_entries_are_filtered_before_limits():
    q = np.array([1.0, 0.0], dtype=np.float32)
    eligible = np.array([0.9, np.sqrt(0.19)], dtype=np.float32)
    with brain.batch():
        for i in range(20):
            gaps.add({"text": str(i), "status": "closed"}, q)
            tools.add({"query": str(i), "yield": 0}, q)
        wanted = gaps.add({"text": "open", "status": "open"}, eligible)
        for _ in range(4):
            tools.add({"query": "useful", "yield": 1}, eligible)
    found = gaps.search(q, k=5, predicate=lambda g: g["status"] == "open")
    assert [g["id"] for g, _ in found] == [wanted]
    assert len(tools.search(q, k=3, predicate=lambda t: t["yield"] > 0, distinct="query")) == 1


def test_embedding_failure_never_falls_back_or_commits(monkeypatch):
    monkeypatch.setenv("DDT_FAKE", "0")
    monkeypatch.delenv("DDT_EMBED", raising=False)
    def unavailable(**kwargs):
        raise ConnectionError("provider unavailable")
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(embeddings=SimpleNamespace(create=unavailable)))
    root = brain.brain_dir()
    before = state_bytes(root)
    with pytest.raises(ConnectionError):
        brain.add_claims([A], "topic", "fake")
    assert state_bytes(root) == before


def test_embedding_identity_rejects_equal_dimension_model_change(monkeypatch):
    brain.add_claims([A], "topic", "fake")
    root = brain.brain_dir()
    before = state_bytes(root)
    monkeypatch.setenv("DDT_FAKE", "0")
    monkeypatch.setenv("DDT_EMBED", "local")  # also 512 dimensions, different vector space
    with pytest.raises(RuntimeError, match="identity"):
        llm.embed(["question"])
    assert state_bytes(root) == before


def test_embedding_configuration_is_pinned_for_the_run(monkeypatch):
    with llm.embedding_session():
        llm.embed(["question"])
        monkeypatch.setenv("DDT_FAKE", "0")
        with pytest.raises(RuntimeError, match="configuration changed"):
            llm.embed(["question"])


def test_check_history_survives_merge_and_rollback():
    with brain.batch():
        brain.add_claims([A] * 9, "topic", "fake")
        brain.add_run({"checks": [{"url": A["source_url"], "passed": True}] * 9 +
                                 [{"url": A["source_url"], "passed": False}]})
    assert brain.counts()["claims"] == 1
    assert metacog.domain_pass_rates() == {"example.org": 0.9}
    with brain.batch():
        brain.add_run({"checks": [{"url": A["source_url"], "passed": False}]})
    assert metacog.domain_pass_rates()["example.org"] == 9 / 11
    brain.rollback(1)
    assert metacog.domain_pass_rates()["example.org"] == 0.9


def test_episode_has_new_claim_ids_and_features_include_prior_checks(monkeypatch):
    features = []
    original = metacog.features
    def capture(*args):
        x = original(*args)
        features.append(x)
        return x
    monkeypatch.setattr(metacog, "features", capture)
    agent.research("What sizes does the Nemotron family come in?", topic="nvidia nemotron")
    ids = {c["id"] for c in brain.entries() if c["kind"] == "claim"}
    assert ids and set(episodes.all()[-1]["claims_used"]) == ids
    assert features[0][3] == 0.5 and features[1][3] == 1.0


def test_replay_uses_production_update_and_matches_persisted_weights():
    from verify import replay
    log = [{"x": [1, 0.8, 0.5, 0.5, 0.5, 0.9], "y": i % 3 != 0} for i in range(12)]
    with brain.batch():
        for row in log:
            metacog.learn(row["x"], row["y"])
    _, w = replay(log, 0.5)
    assert w == metacog.load()["w"]
    assert replay(log, 0.0)[1] == [0.0] * 6


def test_offline_import_never_reads_dotenv_or_opens_network():
    code = '''
import os, sys
from pathlib import Path
def guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)) and Path(os.fsdecode(args[0])).name == ".env":
        raise AssertionError("dotenv access")
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("network access")
sys.addaudithook(guard)
from ddt import agent
agent.research("What sizes does the Nemotron family come in?")
'''
    child = subprocess.run([sys.executable, "-B", "-c", code], cwd=llm.ROOT, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
