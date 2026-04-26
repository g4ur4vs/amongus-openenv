import json

from amongus_env.grpo_train import (
    DEFAULT_MODEL_ID,
    build_prompt_dataset,
    build_training_spec,
    main,
    make_environment_factory,
    rlvr_reward_func,
    run_grpo_dry_run,
    run_grpo_trainer_probe,
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
    assert spec["reward_contract"]["episode_scope"] == "episode_sum_dense_rlvr"
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


def test_rlvr_reward_func_supports_episode_return_aggregation() -> None:
    env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    env.reset()
    env.move("Electrical")
    env.complete_task()

    assert rlvr_reward_func(environments=[env], aggregation="episode_return") == [0.2]


def test_grpo_dry_run_is_import_safe_without_training_extra() -> None:
    result = run_grpo_dry_run(require_trl=False)

    assert result["ok"] is True
    assert result["mode"] == "dry_run"
    assert result["trainer_constructed"] is False
    assert result["model_loaded"] is False
    assert result["reward_probe"]["reward"] == 0.0
    assert result["reward_probe"]["episode_return"] == 0.0


def test_grpo_dry_run_requires_trl_when_requested() -> None:
    result = run_grpo_dry_run(require_trl=True)

    if result["trl_available"]:
        assert result["ok"] is True
        assert "GRPOConfig" in result["trl_symbols"]
        assert "GRPOTrainer" in result["trl_symbols"]
    else:
        assert result["ok"] is False
        assert "training extra" in result["message"]


def test_trainer_probe_requires_explicit_local_model_path() -> None:
    result = run_grpo_trainer_probe(local_model_path=None)

    assert result["ok"] is False
    assert result["mode"] == "trainer_probe"
    assert result["trainer_constructed"] is False
    assert "local model" in result["message"]


def test_trainer_probe_rejects_missing_local_model_path(tmp_path) -> None:
    missing_model = tmp_path / "missing-model"

    result = run_grpo_trainer_probe(local_model_path=str(missing_model))

    assert result["ok"] is False
    assert result["trainer_constructed"] is False
    assert "config.json" in result["message"]


def test_trainer_probe_rejects_hub_model_without_explicit_allow_flag() -> None:
    result = run_grpo_trainer_probe(
        local_model_path=None,
        model_id="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        allow_hub_model=False,
    )

    assert result["ok"] is False
    assert result["trainer_constructed"] is False
    assert "allow_hub_model" in result["message"]


def test_trainer_probe_can_use_injected_hub_model_when_explicitly_allowed() -> None:
    calls = []

    class FakeDataset:
        @staticmethod
        def from_list(rows):
            return rows

    class FakeConfig:
        def __init__(self, **kwargs):
            calls.append(("config", kwargs))

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls.append(("trainer", kwargs))

    result = run_grpo_trainer_probe(
        local_model_path=None,
        model_id="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        allow_hub_model=True,
        dataset_cls=FakeDataset,
        grpo_config_cls=FakeConfig,
        grpo_trainer_cls=FakeTrainer,
        train=False,
    )

    assert result["ok"] is True
    assert result["trainer_constructed"] is True
    assert result["model_loaded"] is True
    assert calls[0][1]["model_init_kwargs"] == {"local_files_only": False}
    assert calls[1][1]["model"] == "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def test_trainer_probe_can_use_injected_trainer_without_training(tmp_path) -> None:
    model_dir = tmp_path / "tiny-local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    calls = []

    class FakeDataset:
        @staticmethod
        def from_list(rows):
            calls.append(("dataset", rows))
            return rows

    class FakeConfig:
        def __init__(self, **kwargs):
            calls.append(("config", kwargs))
            self.kwargs = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls.append(("trainer", kwargs))

        def train(self):
            calls.append(("train", None))
            return None

    result = run_grpo_trainer_probe(
        local_model_path=str(model_dir),
        dataset_cls=FakeDataset,
        grpo_config_cls=FakeConfig,
        grpo_trainer_cls=FakeTrainer,
        train=False,
    )

    assert result["ok"] is True
    assert result["trainer_constructed"] is True
    assert result["trained"] is False
    assert [call[0] for call in calls] == ["dataset", "config", "trainer"]
    assert calls[1][1]["model_init_kwargs"] == {"local_files_only": True}


def test_trainer_probe_runs_train_only_when_explicitly_requested(tmp_path) -> None:
    model_dir = tmp_path / "tiny-local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    calls = []

    class FakeDataset:
        @staticmethod
        def from_list(rows):
            return rows

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self):
            calls.append("train")
            return {"train_loss": 0.0}

    result = run_grpo_trainer_probe(
        local_model_path=str(model_dir),
        dataset_cls=FakeDataset,
        grpo_config_cls=FakeConfig,
        grpo_trainer_cls=FakeTrainer,
        train=True,
    )

    assert result["ok"] is True
    assert result["trained"] is True
    assert calls == ["train"]


def test_trainer_probe_saves_final_model_when_requested(tmp_path) -> None:
    model_dir = tmp_path / "tiny-local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    output_dir = tmp_path / "output"
    calls = []

    class FakeDataset:
        @staticmethod
        def from_list(rows):
            return rows

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    class FakeModel:
        def save_pretrained(self, path):
            calls.append(("model", path))

    class FakeProcessingClass:
        def save_pretrained(self, path):
            calls.append(("processing", path))

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = FakeModel()
            self.processing_class = FakeProcessingClass()

        def train(self):
            return {"train_loss": 0.0}

    result = run_grpo_trainer_probe(
        local_model_path=str(model_dir),
        output_dir=str(output_dir),
        dataset_cls=FakeDataset,
        grpo_config_cls=FakeConfig,
        grpo_trainer_cls=FakeTrainer,
        train=True,
        save_trained_model=True,
    )

    assert result["ok"] is True
    assert result["saved_model_path"] == str(output_dir / "final_model")
    assert calls == [
        ("model", str(output_dir / "final_model")),
        ("processing", str(output_dir / "final_model")),
    ]


def test_grpo_train_cli_prints_valid_json(capsys) -> None:
    main([])

    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "dry_run"
    assert result["trainer_constructed"] is False
