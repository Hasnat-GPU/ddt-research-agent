from ddt import llm, research


def test_plan_without_queries_or_covered_ids_uses_the_known_claims(monkeypatch):
    """Seen live at 16:06: Nemotron returned "covered": [] and "queries": [] with 5 known claims."""
    monkeypatch.setattr(llm, "chat_json", lambda system, user: {"topic": "t", "parts": ["a"], "covered": [], "queries": []})
    known = [{"id": "aaaa1111", "text": "x"}, {"id": "bbbb2222", "text": "y"}]
    assert research.plan("q?", known, 1)["covered"] == ["aaaa1111", "bbbb2222"]


def test_plan_with_queries_keeps_covered_as_returned(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda system, user: {"topic": "t", "parts": ["a"], "covered": [], "queries": ["s"]})
    assert research.plan("q?", [{"id": "aaaa1111", "text": "x"}], 1)["covered"] == []
