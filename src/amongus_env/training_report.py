from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .eval_suite import run_eval_suite


def summarize_training_result(train_json_path: Path) -> dict[str, Any]:
    if not train_json_path.exists():
        return _missing_training_summary(
            source=str(train_json_path),
            error=f"Training JSON does not exist: {train_json_path}",
        )
    try:
        result = json.loads(train_json_path.read_text())
    except json.JSONDecodeError as exc:
        return _missing_training_summary(
            source=str(train_json_path),
            error=f"Training JSON is not valid JSON: {exc}",
        )
    saved_model_path = result.get("saved_model_path")
    checkpoint_saved = bool(saved_model_path and Path(saved_model_path).exists())
    return {
        "source": str(train_json_path),
        "error": None,
        "ok": result.get("ok") is True,
        "trainer_constructed": result.get("trainer_constructed") is True,
        "trained": result.get("trained") is True,
        "model_loaded": result.get("model_loaded") is True,
        "saved_model_path": saved_model_path,
        "checkpoint_saved": checkpoint_saved,
        "train_result": result.get("train_result"),
    }


def build_training_report(
    train_json_path: Optional[Path] = None,
    policy_eval_json_path: Optional[Path] = None,
) -> dict[str, Any]:
    baseline_eval = run_eval_suite()
    training_summary = (
        summarize_training_result(train_json_path)
        if train_json_path is not None
        else _missing_training_summary()
    )
    policy_eval = (
        json.loads(policy_eval_json_path.read_text())
        if policy_eval_json_path is not None
        else None
    )
    baseline_pass_rate = _pass_rate(baseline_eval)
    baseline_vs_rl = _baseline_vs_rl(baseline_pass_rate, policy_eval)
    return {
        "schema_version": 1,
        "ok": baseline_eval["ok"] and training_summary["trained"],
        "baseline_env_eval": {
            "summary": baseline_eval["summary"],
            "pass_rate": baseline_pass_rate,
        },
        "rl_training": training_summary,
        "policy_eval": policy_eval,
        "baseline_vs_rl": baseline_vs_rl,
        "pending_status": _pending_status(training_summary),
    }


def _missing_training_summary(
    source: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "error": error,
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


def _baseline_vs_rl(
    baseline_pass_rate: float,
    policy_eval: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if policy_eval is None:
        return {
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
        }
    comparison = policy_eval.get("comparison", {})
    delta = comparison.get("delta")
    return {
        "baseline_metric": "average_episode_return",
        "baseline_value": comparison.get("baseline_average_episode_return"),
        "rl_metric": "average_episode_return",
        "rl_value": comparison.get("rl_average_episode_return"),
        "score_delta": delta,
        "policy_improvement_claimed": bool(policy_eval.get("ok") and delta is not None and delta > 0),
        "reason": "Policy evaluator JSON supplied; comparison is episode-return based.",
    }


def _pending_status(training_summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "real_grpo_training_loop": {
            "status": "partial",
            "detail": (
                "GRPOTrainer.train() runs and can save a tiny checkpoint; "
                "env-rollout reward mode exists for parsed JSON action completions."
            ),
        },
        "episode_return_reward_contract": {
            "status": "done",
            "detail": "AmongUsToolEnv tracks episode_return and reward aggregation supports it.",
        },
        "better_meeting_simulation": {
            "status": "partial",
            "detail": "Synthetic round-robin preface, accusations, and richer verifiable claims exist; open debate is not implemented.",
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
            "detail": (
                "Tiny checkpoint saved and policy evaluator can score checkpoints."
                if training_summary["checkpoint_saved"]
                else "No saved checkpoint in the provided train JSON."
            ),
        },
        "learned_bots": {
            "status": "partial",
            "detail": "A supervised learned bot-vote policy can be trained from synthetic expert traces; it is not an RL-trained dialogue agent.",
        },
        "open_ended_nlp": {
            "status": "partial",
            "detail": "Meeting speech has a deterministic semantic fallback for natural claim phrasings; it is not full arbitrary-language understanding.",
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
    parser.add_argument(
        "--policy-eval-json",
        type=Path,
        default=None,
        help="JSON output captured from amongus-policy-eval.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build_training_report(args.train_json, args.policy_eval_json), indent=2))


if __name__ == "__main__":
    main()
