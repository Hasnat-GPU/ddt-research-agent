import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test runs offline with the stand-ins and its own empty brain."""
    monkeypatch.setenv("DDT_FAKE", "1")
    monkeypatch.setenv("DDT_BRAIN_DIR", str(tmp_path / "brain"))
