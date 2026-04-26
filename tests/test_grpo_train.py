import json

from amongus_env.grpo_train import (
    DEFAULT_MODEL_ID,
    build_prompt_dataset,
    build_training_spec,
    main,
    make_environment_factory,
    rlvr_reward_func,
    run_grpo_dry_run,
)
from amongus_env.trl_adapter import AmongUsToolEnv


def test_training_spec_documents_model_config_and_reward_contract() -> None:
    spec = build_training_spec()

    assert spec["model_id"] == DEFAULT_MODEL_ID
    assert spec["grpo_config"] == {
        "num_generations": 8,
        "max_prompt_length": 512,
        "max_completion_length": 1024,
        "output_dir": "outputs/amongus-grpo",
    }
    assert spec["reward_contract"]["source"] == "AmongUsToolEnv.reward"
    assert spec["reward_contract"]["scope"] == "last_step_dense_rlvr"
    assert spec["reward_contract"]["aggregation"] == "one reward per environment"


def test_prompt_dataset_contains_tool_use_and_meeting_phase_prompts() -> None:
    prompts = build_prompt_dataset()

    assert len(prompts) >= 2
    assert any("move" in prompt["prompt"].lower() for prompt in prompts)
    assert any("meeting" in prompt["prompt"].lower() for prompt in prompts)


def test_environment_factory_returns_independent_tool_envs() -> None:
    factory = make_environment_factory(seed=11, impostor_ids=["blue"])

    first = factory()
    second = factory()
    first.reset()
    second.reset()
    first.move("Electrical")

    assert isinstance(first, AmongUsToolEnv)
    assert isinstance(second, AmongUsToolEnv)
    assert first is not second
    assert first.last_observation.location == "Electrical"
    assert second.last_observation.location == "Cafeteria"


def test_rlvr_reward_func_delegates_to_tool_env_rewards() -> None:
    first = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    second = AmongUsToolEnv(seed=1, impostor_ids=["red"])
    first.reset()
    second.reset()
    first.move("Electrical")
    second.kill("blue")

    assert rlvr_reward_func(environments=[first, second]) == [0.0, 0.5]


def test_grpo_dry_run_is_import_safe_without_training_extra() -> None:
    result = run_grpo_dry_run(require_trl=False)

    assert result["ok"] is True
    assert result["mode"] == "dry_run"
    assert result["trainer_constructed"] is False
    assert result["model_loaded"] is False
    assert result["reward_probe"]["reward"] == 0.0


def test_grpo_dry_run_requires_trl_when_requested() -> None:
    result = run_grpo_dry_run(require_trl=True)

    if result["trl_available"]:
        assert result["ok"] is True
        assert "GRPOConfig" in result["trl_symbols"]
        assert "GRPOTrainer" in result["trl_symbols"]
    else:
        assert result["ok"] is False
        assert "training extra" in result["message"]


def test_grpo_train_cli_prints_valid_json(capsys) -> None:
    main([])

    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "dry_run"
    assert result["trainer_constructed"] is False
