from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from .models import (
    Action,
    ActionAdapter,
    CallMeeting,
    CompleteTask,
    FakeTask,
    Kill,
    Move,
    PassMeeting,
    ReportBody,
    Speak,
    Vent,
    Vote,
)
from .trl_adapter import AmongUsToolEnv


def parse_completion_actions(text: str) -> list[Action]:
    actions = []
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for item in parsed:
                action = _validate_action(item)
                if action is not None:
                    actions.append(action)
        else:
            action = _validate_action(parsed)
            if action is not None:
                actions.append(action)
    return actions


def run_completion_rollout(
    completion: Any,
    seed: int = 1,
    controlled_player_id: str = "red",
    player_ids: Optional[list[str]] = None,
    impostor_ids: Optional[list[str]] = None,
    max_actions: int = 16,
) -> dict[str, Any]:
    env = AmongUsToolEnv(
        seed=seed,
        controlled_player_id=controlled_player_id,
        player_ids=player_ids,
        impostor_ids=impostor_ids,
    )
    env.reset()
    actions = parse_completion_actions(_completion_text(completion))[:max_actions]
    for action in actions:
        _apply_action(env, action)
        if env.last_observation is not None and env.last_observation.done:
            break
    return {
        "episode_return": env.episode_return,
        "valid_actions": len(actions),
        "trace": env.get_rollout_trace(),
    }


def completion_episode_return_reward_func(
    completions: Iterable[Any],
    seed: int = 1,
    controlled_player_id: str = "red",
    player_ids: Optional[list[str]] = None,
    impostor_ids: Optional[list[str]] = None,
    max_actions: int = 16,
    **kwargs: Any,
) -> list[float]:
    return [
        run_completion_rollout(
            completion,
            seed=seed,
            controlled_player_id=controlled_player_id,
            player_ids=player_ids,
            impostor_ids=impostor_ids,
            max_actions=max_actions,
        )["episode_return"]
        for completion in completions
    ]


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = []
    if stripped.startswith("[") or stripped.startswith("{"):
        candidates.append(stripped)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            candidates.append(line)
    return candidates


def _validate_action(value: Any) -> Optional[Action]:
    if not isinstance(value, dict):
        return None
    try:
        return ActionAdapter.validate_python(value)
    except Exception:
        return None


def _apply_action(env: AmongUsToolEnv, action: Action) -> None:
    if isinstance(action, Move):
        env.move(action.room)
    elif isinstance(action, CompleteTask):
        env.complete_task()
    elif isinstance(action, FakeTask):
        env.fake_task()
    elif isinstance(action, Vent):
        env.vent(action.room)
    elif isinstance(action, Kill):
        env.kill(action.target_id)
    elif isinstance(action, ReportBody):
        env.report_body()
    elif isinstance(action, CallMeeting):
        env.call_meeting()
    elif isinstance(action, Speak):
        env.speak(action.message)
    elif isinstance(action, PassMeeting):
        env.pass_meeting()
    elif isinstance(action, Vote):
        env.vote(action.target_id)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)
