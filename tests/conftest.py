import pytest
import os

# Set before importing project modules: collection must never open .env.
os.environ["DDT_FAKE"] = "1"
os.environ["DDT_SKIP_DOTENV"] = "1"

from tests import fakes


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test runs offline with the stand-ins, its own empty brain and the late pages offline."""
    monkeypatch.setenv("DDT_FAKE", "1")
    monkeypatch.setenv("DDT_BRAIN_DIR", str(tmp_path / "brain"))
    monkeypatch.setenv("DDT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("DDT_META_LR", raising=False)
    monkeypatch.setattr(fakes, "LATE_ONLINE", False)
