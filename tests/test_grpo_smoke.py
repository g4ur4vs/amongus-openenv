from amongus_env.grpo_smoke import run_grpo_smoke


def test_grpo_smoke_reports_missing_trl_without_failing() -> None:
    result = run_grpo_smoke(require_trl=False)

    assert result["ok"] is True
    assert result["trl_available"] in {True, False}
    assert result["reward"] == 0.0
    assert "Electrical" in result["step_summary"]


def test_grpo_smoke_requires_trl_when_requested() -> None:
    result = run_grpo_smoke(require_trl=True)

    if result["trl_available"]:
        assert result["ok"] is True
        assert "GRPOConfig" in result["trl_symbols"]
        assert "GRPOTrainer" in result["trl_symbols"]
    else:
        assert result["ok"] is False
        assert "Install" in result["message"]
