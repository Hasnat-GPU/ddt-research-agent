from ddt import agent, brain


def test_second_run_answers_from_the_brain():
    first = agent.research("What sizes does the Nemotron family come in?", topic="nvidia nemotron")
    assert first["searches"] == 3 and first["new"] > 0
    second = agent.research("What sizes does the Nemotron family come in?", topic="nvidia nemotron")
    assert second["searches"] == 0 and set(second["citations"]) <= set(first["citations"])
    assert brain.counts()["version"] == 2  # one brain version per run


def test_failed_quotes_raise_the_evidence_bar():
    first = agent.research("What happened in the unreliable topic?", topic="unreliable topic")
    assert first["dropped"] == 2
    second = agent.research("What happened in the unreliable topic?", topic="unreliable topic")
    assert 'evidence bar for "unreliable topic": 2' in open(second["trace_path"], encoding="utf-8").read()
