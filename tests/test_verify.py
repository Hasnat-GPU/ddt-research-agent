import json

import pytest

import verify


@pytest.mark.parametrize("fake", [True, False])
def test_verification_exit_allowances_are_explicit(fake):
    results = dict.fromkeys(verify.REQUIRED, "PASS")
    assert verify.result_exit_code(results, fake) == 0
    results.update(S5="NOT TRIGGERED", S10="NOT TRIGGERED")
    assert verify.result_exit_code(results, fake) == 0
    for status in ["FAIL", "VOID", "CONTROL BROKEN", "NOT RUN", "NOT TRIGGERED"]:
        assert verify.result_exit_code({**results, "S8": status}, fake) == 1
    assert verify.result_exit_code({k: v for k, v in results.items() if k != "S3"}, fake) == 1
    assert verify.result_exit_code({**results, "S9": "NOT TRIGGERED"}, fake) == 1


def test_interrupted_verification_saves_partial_results_and_fails(monkeypatch, tmp_path):
    def interrupted(args, results):
        results["S3"] = "PASS"
        raise KeyboardInterrupt()
    monkeypatch.setattr(verify, "_checks", interrupted)
    output = tmp_path / "verify"
    assert verify.main(["--fake", "--output-dir", str(output)]) == 1
    result = json.loads((output / "results.json").read_text())
    assert result["results"] == {"S3": "PASS", "Run error": "FAIL"}
    assert "Missing required results" in (output / "results.md").read_text()


def test_verification_refuses_to_reuse_an_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        verify.main(["--fake", "--output-dir", str(tmp_path)])
