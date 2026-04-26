from pathlib import Path


def test_pyproject_declares_eval_console_scripts() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'amongus-baseline-eval = "amongus_env.eval_suite:main"' in pyproject
    assert 'amongus-golden-trace = "amongus_env.golden_episode:main"' in pyproject
    assert 'amongus-grpo-smoke = "amongus_env.grpo_smoke:main"' in pyproject
    assert 'amongus-reasoning-trace = "amongus_env.reasoning_trace:main"' in pyproject
