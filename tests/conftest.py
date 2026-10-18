import pytest

from tests import fakes


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test runs offline with the stand-ins, its own empty brain and the late pages offline."""
    monkeypatch.setenv("DDT_FAKE", "1")
    monkeypatch.setenv("DDT_BRAIN_DIR", str(tmp_path / "brain"))
    monkeypatch.delenv("DDT_META_LR", raising=False)
    monkeypatch.setattr(fakes, "LATE_ONLINE", False)
