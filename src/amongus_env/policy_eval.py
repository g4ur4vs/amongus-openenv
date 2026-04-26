from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional

from .completion_rollout import run_completion_rollout

SCRIPTED_BASELINE_COMPLETION = """
{"type": "move", "room": "Electrical"}
{"type": "complete_task"}
"""


def build_policy_eval_report(
    rl_completions: Optional[list[str]] = None,
    checkpoint: Optional[str] = None,
    completion_generator: Optional[Callable[[str, str, int], str]] = None,
    num_episodes: int = 1,
    seed_start: int = 1,
    max_actions: int = 16,
) -> dict[str, Any]:
    baseline = _evaluate_completions(
        [SCRIPTED_BASELINE_COMPLETION] * num_episodes,
        policy_name="scripted_baseline",
        seed_start=seed_start,
        max_actions=max_actions,
    )
    rl = None
    if checkpoint is not None:
        rl = _evaluate_completions(
            [
                generate_checkpoint_completion(
                    checkpoint,
                    prompt=_policy_prompt(),
                    generator=completion_generator,
                )
                for _ in range(num_episodes)
            ],
            policy_name="checkpoint_completion",
            seed_start=seed_start,
            max_actions=max_actions,
            checkpoint=checkpoint,
        )
    elif rl_completions is not None:
        rl = _evaluate_completions(
            _repeat_to_length(rl_completions, num_episodes),
            policy_name="completion_policy",
            seed_start=seed_start,
            max_actions=max_actions,
        )
    comparison = _comparison(baseline, rl)
    return {
        "schema_version": 1,
        "ok": baseline["ok"] and (rl is None or rl["ok"]),
        "metric": "average_episode_return",
        "num_episodes": num_episodes,
        "seed_start": seed_start,
        "max_actions": max_actions,
        "baseline": baseline,
        "rl": rl,
        "comparison": comparison,
    }


def _evaluate_completions(
    completions: list[str],
    policy_name: str,
    seed_start: int,
    max_actions: int,
    checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    episodes = []
    for index, completion in enumerate(completions):
        rollout = run_completion_rollout(
            completion,
            seed=seed_start + index,
            impostor_ids=["blue"],
            max_actions=max_actions,
        )
        episodes.append(
            {
                "seed": seed_start + index,
                "episode_return": rollout["episode_return"],
                "valid_actions": rollout["valid_actions"],
                "steps": len(rollout["trace"]),
            }
        )
    average = round(
        sum(episode["episode_return"] for episode in episodes) / len(episodes),
        10,
    )
    result = {
        "ok": True,
        "policy": policy_name,
        "average_episode_return": average,
        "episodes": episodes,
    }
    if checkpoint is not None:
        result["checkpoint"] = checkpoint
    return result


def generate_checkpoint_completion(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 128,
    generator: Optional[Callable[[str, str, int], str]] = None,
) -> str:
    if generator is not None:
        return generator(checkpoint, prompt, max_new_tokens)
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise ImportError('Install policy eval deps with: pip install -e ".[training]"') from exc

    pipe = pipeline(
        "text-generation",
        model=checkpoint,
        tokenizer=checkpoint,
        device_map="auto",
    )
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return str(output[0]["generated_text"])


def _comparison(
    baseline: dict[str, Any],
    rl: Optional[dict[str, Any]],
) -> dict[str, Any]:
    baseline_value = baseline["average_episode_return"]
    rl_value = None if rl is None else rl["average_episode_return"]
    delta = None if rl_value is None else round(rl_value - baseline_value, 10)
    return {
        "baseline_average_episode_return": baseline_value,
        "rl_average_episode_return": rl_value,
        "delta": delta,
        "policy_improvement_claimed": delta is not None and delta > 0,
    }


def _repeat_to_length(values: list[str], length: int) -> list[str]:
    if not values:
        return [""] * length
    return [values[index % len(values)] for index in range(length)]


def _read_completions(path: Path) -> list[str]:
    text = path.read_text()
    if "\n---\n" in text:
        return [chunk.strip() for chunk in text.split("\n---\n")]
    return [text]


def _policy_prompt() -> str:
    return (
        "Emit only JSON Lines for Among Us actions. Example:\n"
        '{"type": "move", "room": "Electrical"}\n'
        '{"type": "complete_task"}\n'
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs RL policy completions")
    parser.add_argument("--rl-completions-file", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=16)
    args = parser.parse_args(argv)
    completions = (
        _read_completions(args.rl_completions_file)
        if args.rl_completions_file is not None
        else None
    )
    print(
        json.dumps(
            build_policy_eval_report(
                rl_completions=completions,
                checkpoint=args.checkpoint,
                num_episodes=args.num_episodes,
                seed_start=args.seed_start,
                max_actions=args.max_actions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
