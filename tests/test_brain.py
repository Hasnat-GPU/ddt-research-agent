from ddt import brain

A = {"text": "The Nemotron 3 family comes in three sizes: Nano, Super and Ultra.",
     "source_url": "https://example.org/a", "quote": "q", "confidence": 0.9}
B = {"text": "Nemotron models are released under the NVIDIA Open Model License.",
     "source_url": "https://example.org/b", "quote": "q", "confidence": 0.9}


def test_add_and_merge_at_095():
    assert brain.add_claims([A, B], "nvidia nemotron", "m") == (2, 0)
    same = {**A, "source_url": "https://example.org/c"}
    assert brain.add_claims([same], "nvidia nemotron", "m") == (0, 1)
    stored = [e for e in brain.entries() if e["kind"] == "claim"]
    assert len(stored) == 2 and stored[0]["also_sources"] == ["https://example.org/c"]
    assert brain.vectors().shape[0] == 2


def test_retrieve_returns_close_claims_only():
    brain.add_claims([A, B], "nvidia nemotron", "m")
    found = brain.retrieve("What sizes does the Nemotron family come in?")
    assert {c["text"] for c in found} == {A["text"], B["text"]}
    assert all(c["similarity"] >= 0.75 for c in found)
    assert brain.retrieve("When did the EU AI Act enter into force?") == []


def test_snapshot_and_rollback():
    brain.add_claims([A], "nvidia nemotron", "m")
    with brain.batch():
        brain.add_claims([B], "nvidia nemotron", "m")
        brain.add_run({"question": "q"})
    assert brain.counts() == {"claims": 2, "runs": 1, "version": 2}
    brain.rollback(1)
    assert brain.counts() == {"claims": 1, "runs": 0, "version": 1}
    brain.rollback(1)
    assert brain.counts() == {"claims": 0, "runs": 0, "version": 0}
