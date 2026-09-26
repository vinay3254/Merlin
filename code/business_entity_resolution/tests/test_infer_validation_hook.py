from src.infer import maybe_run_validator


def test_maybe_run_validator_returns_none_when_script_absent(tmp_path):
    result = maybe_run_validator(
        output_dir=str(tmp_path / "output"),
        test_dir=str(tmp_path / "dataset" / "test"),
        validator_path=str(tmp_path / "utils" / "validate_submission.py"),
    )
    assert result is None


def test_maybe_run_validator_runs_script_when_present(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    validator_path = tmp_path / "utils" / "validate_submission.py"
    validator_path.parent.mkdir(parents=True)
    validator_path.write_text(
        "import sys\n"
        "print('PASS')\n"
        "sys.exit(0)\n"
    )
    result = maybe_run_validator(str(output_dir), str(test_dir), str(validator_path))
    assert result is not None
    assert "PASS" in result
