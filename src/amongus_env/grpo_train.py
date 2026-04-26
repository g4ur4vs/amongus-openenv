from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Optional

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
            "aggregation": "one reward per environment",
            "function": "amongus_env.trl_adapter.reward_from_game_state",
        },
        "rollout_contract": {
            "environment_factory": "make_environment_factory",
            "tool_surface": [
                "reset",
                "move",
                "complete_task",
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
    args = parser.parse_args(argv)

    result = run_grpo_dry_run(
        require_trl=args.require_trl,
        model_id=args.model_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
