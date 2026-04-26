from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .eval_suite import run_eval_suite


def summarize_training_result(train_json_path: Path) -> dict[str, Any]:
    result = json.loads(train_json_path.read_text())
    saved_model_path = result.get("saved_model_path")
    checkpoint_saved = bool(saved_model_path and Path(saved_model_path).exists())
    return {
        "source": str(train_json_path),
        "ok": result.get("ok") is True,
        "trainer_constructed": result.get("trainer_constructed") is True,
        "trained": result.get("trained") is True,
        "model_loaded": result.get("model_loaded") is True,
        "saved_model_path": saved_model_path,
        "checkpoint_saved": checkpoint_saved,
        "train_result": result.get("train_result"),
    }


def build_training_report(train_json_path: Optional[Path] = None) -> dict[str, Any]:
    baseline_eval = run_eval_suite()
    training_summary = (
        summarize_training_result(train_json_path)
        if train_json_path is not None
        else _missing_training_summary()
    )
    baseline_pass_rate = _pass_rate(baseline_eval)
    return {
        "schema_version": 1,
        "ok": baseline_eval["ok"] and training_summary["trained"],
        "baseline_env_eval": {
            "summary": baseline_eval["summary"],
            "pass_rate": baseline_pass_rate,
        },
        "rl_training": training_summary,
        "baseline_vs_rl": {
            "baseline_metric": "deterministic_environment_eval_pass_rate",
            "baseline_value": baseline_pass_rate,
            "rl_metric": None,
            "rl_value": None,
            "score_delta": None,
            "policy_improvement_claimed": False,
            "reason": (
                "GRPO trainer smoke is complete, but there is no model-policy evaluator "
                "that runs a checkpoint through Among Us episodes yet."
            ),
        },
        "pending_status": _pending_status(training_summary),
    }


def _missing_training_summary() -> dict[str, Any]:
    return {
        "source": None,
        "ok": False,
        "trainer_constructed": False,
        "trained": False,
        "model_loaded": False,
        "saved_model_path": None,
        "checkpoint_saved": False,
        "train_result": None,
    }


def _pass_rate(eval_result: dict[str, Any]) -> float:
    summary = eval_result["summary"]
    return round(summary["passed"] / summary["scenarios"], 10)


def _pending_status(training_summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "real_grpo_training_loop": {
            "status": "partial",
            "detail": (
                "GRPOTrainer.train() runs and can save a tiny checkpoint; "
                "the reward is still a trainer smoke reward, not an env rollout reward."
            ),
        },
        "episode_return_reward_contract": {
            "status": "done",
            "detail": "AmongUsToolEnv tracks episode_return and reward aggregation supports it.",
        },
        "better_meeting_simulation": {
            "status": "partial",
            "detail": "Accusations and richer verifiable claims exist; round-robin NLP is not implemented.",
        },
        "impostor_fake_tasks": {
            "status": "done",
            "detail": "Impostors can fake room tasks without task reward.",
        },
        "deception_elo_leaderboard": {
            "status": "done",
            "detail": "Repeated-trace Deception Elo leaderboard CLI exists.",
        },
        "react_three_visualization": {
            "status": "not_done",
            "detail": "Space is Gradio, not React/Three.js.",
        },
        "real_trained_model_system": {
            "status": "partial" if training_summary["checkpoint_saved"] else "not_done",
            "detail": "Tiny checkpoint saved." if training_summary["checkpoint_saved"] else "No saved checkpoint in the provided train JSON.",
        },
        "learned_bots": {
            "status": "not_done",
            "detail": "Bots remain deterministic scripted policies.",
        },
        "open_ended_nlp": {
            "status": "not_done",
            "detail": "Claims are deterministic parsers, not open-ended NLP.",
        },
        "vents": {
            "status": "done",
            "detail": "Vent action and 'I saw X vent' claim verification exist.",
        },
        "richer_space_ui": {
            "status": "done",
            "detail": "Space has eval, trace, Elo, leaderboard, and GRPO status tabs.",
        },
        "edge_case_evals": {
            "status": "done",
            "detail": "Eval suite covers fake tasks, vents, multi-impostor, and no-majority cases.",
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate baseline-vs-RL training report")
    parser.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="JSON output captured from amongus-grpo-train.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build_training_report(args.train_json), indent=2))


if __name__ == "__main__":
    main()
