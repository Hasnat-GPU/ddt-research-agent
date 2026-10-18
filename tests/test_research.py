import numpy as np

from ddt import agent, llm, research

LICENCE_PART = "Under which licence are the Nemotron 3 models released?"
SIZE_CLAIM = {"id": "aaaa1111", "text": "Nemotron 3 comes in Nano, Super and Ultra sizes."}
LICENCE_CLAIM = {"id": "bbbb2222", "text": "Nemotron 3 is released under the NVIDIA Open Model License."}


def _embed(texts):
    """Licence texts point one way, size texts another: a same-topic claim that does not answer the part."""
    return np.array([[1.0, 0.0] if "icen" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


def test_plan_keeps_an_empty_answer_as_returned(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda system, user: {"topic": "t", "parts": ["a"], "covered": [], "queries": []})
    assert research.plan("q?", [SIZE_CLAIM], 1)["covered"] == []


def test_repair_does_not_cover_a_claim_that_answers_no_part(monkeypatch):
    """Seen live at 16:06: covered [] and queries []. A size claim must not stand in for the licence answer."""
    monkeypatch.setattr(llm, "embed", _embed)
    covered, queries = agent.repair_plan([LICENCE_PART], [SIZE_CLAIM])
    assert covered == [] and queries == [LICENCE_PART]


def test_repair_covers_a_claim_that_answers_the_part(monkeypatch):
    monkeypatch.setattr(llm, "embed", _embed)
    covered, queries = agent.repair_plan([LICENCE_PART], [SIZE_CLAIM, LICENCE_CLAIM])
    assert covered == [LICENCE_CLAIM] and queries == []
