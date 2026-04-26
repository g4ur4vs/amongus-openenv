from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Optional

from .completion_rollout import completion_episode_return_reward_func
from .trl_adapter import AmongUsToolEnv, reward_from_game_state

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/amongus-grpo"
DEFAULT_NUM_GENERATIONS = 8
DEFAULT_MAX_PROMPT_LENGTH = 512
DEFAULT_MAX_COMPLETION_LENGTH = 1024


def build_training_spec(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "grpo_config": build_grpo_config_kwargs(output_dir=output_dir),
        "reward_contract": {
            "source": "AmongUsToolEnv.reward",
            "scope": "last_step_dense_rlvr",
            "episode_scope": "episode_sum_dense_rlvr",
            "aggregation": "one reward per environment",
            "aggregation_parameter": "aggregation",
            "function": "amongus_env.trl_adapter.reward_from_game_state",
        },
        "rollout_contract": {
            "environment_factory": "make_environment_factory",
            "tool_surface": [
                "reset",
                "move",
                "complete_task",
                "fake_task",
                "vent",
                "kill",
                "report_body",
                "call_meeting",
                "speak",
                "pass_meeting",
                "vote",
            ],
        },
    }


def build_grpo_config_kwargs(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    return {
        "num_generations": DEFAULT_NUM_GENERATIONS,
        "max_prompt_length": DEFAULT_MAX_PROMPT_LENGTH,
        "max_completion_length": DEFAULT_MAX_COMPLETION_LENGTH,
        "output_dir": output_dir,
    }


def build_prompt_dataset() -> list[dict[str, str]]:
    return [
        {
            "prompt": (
                "You are controlling one Among Us player. Use tools to move through "
                "rooms, complete tasks if crewmate, or create pressure if impostor."
            )
        },
        {
            "prompt": (
                "A meeting is active. Use speak/pass_meeting before voting, and make "
                "claims that can be checked against the game state."
            )
        },
    ]


def make_environment_factory(
    seed: int = 0,
    controlled_player_id: str = "red",
    player_ids: Optional[list[str]] = None,
    impostor_ids: Optional[list[str]] = None,
) -> Callable[[], AmongUsToolEnv]:
    def factory() -> AmongUsToolEnv:
        return AmongUsToolEnv(
            seed=seed,
            controlled_player_id=controlled_player_id,
            player_ids=player_ids,
            impostor_ids=impostor_ids,
        )

    return factory


def rlvr_reward_func(environments: list[AmongUsToolEnv], **kwargs: Any) -> list[float]:
    return reward_from_game_state(environments, **kwargs)


def grpo_constant_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """TRL-shaped placeholder reward for trainer construction smoke tests."""
    return [0.0 for _ in completions]


def run_grpo_trainer_probe(
    local_model_path: Optional[str],
    model_id: str = DEFAULT_MODEL_ID,
    allow_hub_model: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    train: bool = False,
    use_cpu: bool = False,
    save_trained_model: bool = False,
    reward_mode: str = "constant",
    dataset_cls: Optional[Any] = None,
    grpo_config_cls: Optional[Any] = None,
    grpo_trainer_cls: Optional[Any] = None,
) -> dict[str, Any]:
    model_ref = local_model_path
    local_files_only = True
    if local_model_path is None:
        if not allow_hub_model:
            return _trainer_probe_result(
                ok=False,
                message=(
                    "A local model path is required unless allow_hub_model=True; "
                    "refusing to download a hub model implicitly."
                ),
                output_dir=output_dir,
            )
        model_ref = model_id
        local_files_only = False
    else:
        model_path = Path(local_model_path)
        config_path = model_path / "config.json"
        if not model_path.is_dir() or not config_path.is_file():
            return _trainer_probe_result(
                ok=False,
                local_model_path=str(model_path),
                message=f"Local model path must be a directory containing config.json: {model_path}",
                output_dir=output_dir,
            )
        model_ref = str(model_path)

    try:
        if dataset_cls is None:
            from datasets import Dataset as dataset_cls
        if grpo_config_cls is None or grpo_trainer_cls is None:
            from trl import GRPOConfig, GRPOTrainer

            grpo_config_cls = grpo_config_cls or GRPOConfig
            grpo_trainer_cls = grpo_trainer_cls or GRPOTrainer
    except Exception as exc:  # pragma: no cover - depends on optional training extra
        return _trainer_probe_result(
            ok=False,
            local_model_path=model_ref,
            message='Install training deps with: pip install -e ".[training]"',
            error=f"{exc.__class__.__name__}: {exc}",
            output_dir=output_dir,
        )

    try:
        train_dataset = dataset_cls.from_list(build_prompt_dataset())
        args = grpo_config_cls(
            output_dir=output_dir,
            num_generations=2,
            per_device_train_batch_size=2,
            max_completion_length=16,
            max_steps=1,
            report_to=[],
            logging_steps=1,
            save_strategy="no",
            use_cpu=use_cpu,
            optim="adamw_torch",
            model_init_kwargs={"local_files_only": local_files_only},
        )
        reward_func = (
            completion_episode_return_reward_func
            if reward_mode == "env_rollout"
            else grpo_constant_reward_func
        )
        trainer = grpo_trainer_cls(
            model=model_ref,
            reward_funcs=reward_func,
            args=args,
            train_dataset=train_dataset,
        )
        train_result = None
        saved_model_path = None
        if train:
            with contextlib.redirect_stdout(io.StringIO()):
                train_result = trainer.train()
            if save_trained_model:
                saved_model_path = str(Path(output_dir) / "final_model")
                Path(saved_model_path).mkdir(parents=True, exist_ok=True)
                trainer.model.save_pretrained(saved_model_path)
                processing_class = getattr(trainer, "processing_class", None)
                if processing_class is not None:
                    processing_class.save_pretrained(saved_model_path)
    except Exception as exc:  # pragma: no cover - depends on optional training stack
        return _trainer_probe_result(
            ok=False,
            local_model_path=model_ref,
            message="GRPO trainer probe failed during construction or train().",
            error=f"{exc.__class__.__name__}: {exc}",
            output_dir=output_dir,
        )

    return _trainer_probe_result(
        ok=True,
        local_model_path=model_ref,
        message=(
            "GRPOTrainer constructed and train() ran."
            if train
            else "GRPOTrainer constructed; train() was not run."
        ),
        trainer_constructed=True,
        model_loaded=True,
        trained=train,
        train_result=str(train_result) if train_result is not None else None,
        saved_model_path=saved_model_path,
        reward_mode=reward_mode,
        output_dir=output_dir,
    )


def _trainer_probe_result(
    ok: bool,
    message: str,
    local_model_path: Optional[str] = None,
    error: Optional[str] = None,
    trainer_constructed: bool = False,
    model_loaded: bool = False,
    trained: bool = False,
    train_result: Optional[str] = None,
    saved_model_path: Optional[str] = None,
    reward_mode: str = "constant",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": ok,
        "mode": "trainer_probe",
        "message": message,
        "error": error,
        "local_model_path": local_model_path,
        "trainer_constructed": trainer_constructed,
        "model_loaded": model_loaded,
        "trained": trained,
        "reward_mode": reward_mode,
        "train_result": train_result,
        "saved_model_path": saved_model_path,
        "training_spec": build_training_spec(
            model_id=local_model_path or DEFAULT_MODEL_ID,
            output_dir=output_dir,
        ),
    }


def run_grpo_dry_run(
    require_trl: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    trl_symbols, trl_error = _load_trl_symbols()
    factory = make_environment_factory(seed=1, impostor_ids=["blue"])
    env = factory()
    reset_summary = env.reset()
    step_summary = env.move("Electrical")
    reward = rlvr_reward_func(environments=[env])[0]
    trl_available = trl_error is None
    ok = trl_available or not require_trl
    message = (
        "TRL GRPO symbols imported; dry-run did not construct a trainer or load a model."
        if trl_available
        else 'Install the training extra with: pip install -e ".[training]"'
    )

    return {
        "schema_version": 1,
        "ok": ok,
        "mode": "dry_run",
        "message": message,
        "trl_available": trl_available,
        "trl_symbols": trl_symbols,
        "trl_error": trl_error,
        "trainer_constructed": False,
        "model_loaded": False,
        "training_spec": build_training_spec(model_id=model_id, output_dir=output_dir),
        "prompt_count": len(build_prompt_dataset()),
        "reward_probe": {
            "reset_summary": reset_summary,
            "step_summary": step_summary,
            "reward": reward,
            "episode_return": env.episode_return,
        },
    }


def _load_trl_symbols() -> tuple[list[str], Optional[str]]:
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # pragma: no cover - depends on optional training extra
        return [], f"{exc.__class__.__name__}: {exc}"
    return [GRPOConfig.__name__, GRPOTrainer.__name__], None


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Among Us GRPO training skeleton")
    parser.add_argument(
        "--require-trl",
        action="store_true",
        help="Fail the JSON status if TRL is not installed.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--construct-trainer",
        action="store_true",
        help="Construct GRPOTrainer using --local-model. Does not train unless --train is also set.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Call trainer.train(). Requires --construct-trainer and --local-model.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Local Hugging Face model directory containing config.json.",
    )
    parser.add_argument(
        "--allow-hub-model",
        action="store_true",
        help="Allow downloading --model-id from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU training even when GPU/MPS is available.",
    )
    parser.add_argument(
        "--save-trained-model",
        action="store_true",
        help="After --train, save model/tokenizer artifacts to output_dir/final_model.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["constant", "env_rollout"],
        default="constant",
        help="Use constant smoke reward or parsed completion env-rollout reward.",
    )
    args = parser.parse_args(argv)

    if args.construct_trainer or args.train:
        result = run_grpo_trainer_probe(
            local_model_path=args.local_model,
            model_id=args.model_id,
            allow_hub_model=args.allow_hub_model,
            output_dir=args.output_dir,
            train=args.train,
            use_cpu=args.use_cpu,
            save_trained_model=args.save_trained_model,
            reward_mode=args.reward_mode,
        )
    else:
        result = run_grpo_dry_run(
            require_trl=args.require_trl,
            model_id=args.model_id,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
