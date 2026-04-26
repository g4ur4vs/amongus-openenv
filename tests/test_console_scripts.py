from pathlib import Path


def test_pyproject_declares_eval_console_scripts() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'amongus-baseline-eval = "amongus_env.eval_suite:main"' in pyproject
    assert 'amongus-golden-trace = "amongus_env.golden_episode:main"' in pyproject
    assert 'amongus-grpo-smoke = "amongus_env.grpo_smoke:main"' in pyproject
    assert 'amongus-reasoning-trace = "amongus_env.reasoning_trace:main"' in pyproject
    assert 'amongus-deception-elo = "amongus_env.deception_elo:main"' in pyproject
    assert 'amongus-grpo-train = "amongus_env.grpo_train:main"' in pyproject
    assert 'amongus-deception-leaderboard = "amongus_env.deception_leaderboard:main"' in pyproject
    assert 'amongus-training-report = "amongus_env.training_report:main"' in pyproject
    assert 'amongus-policy-eval = "amongus_env.policy_eval:main"' in pyproject
    assert 'amongus-train-learned-bots = "amongus_env.learned_bots:main"' in pyproject
