from __future__ import annotations

import json
from typing import Any

from .golden_episode import run_golden_episode

Check = dict[str, Any]
Trace = list[dict[str, Any]]

EXPECTED_LABELS = [
    "reset",
    "move_electrical",
    "complete_task",
    "call_meeting",
    "false_alibi",
    "vote_after_false_alibi",
]


def run_eval_suite() -> dict[str, Any]:
    return evaluate_trace(run_golden_episode())


def evaluate_trace(trace: Trace) -> dict[str, Any]:
    checks = [
        _check_labels_sequence(trace),
        _check_task_reward(trace),
        _check_meeting_protocol(trace),
        _check_false_alibi_penalty(trace),
        _check_bot_vote_ejects_false_claimant(trace),
    ]
    return {
        "schema_version": 1,
        "episode": "golden_false_alibi",
        "ok": all(check["ok"] for check in checks),
        "summary": {
            "steps": len(trace),
            "total_reward": round(
                sum(_observation(step).get("reward", 0.0) for step in trace),
                10,
            ),
        },
        "checks": checks,
    }


def _check_labels_sequence(trace: Trace) -> Check:
    labels = [step.get("label") for step in trace]
    return _check(
        "labels_sequence",
        labels == EXPECTED_LABELS,
        f"labels={labels}",
    )


def _check_task_reward(trace: Trace) -> Check:
    observation = _step_observation(trace, "complete_task")
    task_list = observation.get("task_list", [])
    first_task = task_list[0] if task_list else {}
    reward_ok = observation.get("reward") == 0.2
    task_ok = first_task.get("completed") is True
    return _check(
        "task_reward",
        reward_ok and task_ok,
        f"reward={observation.get('reward')}; first_task={first_task}",
    )


def _check_meeting_protocol(trace: Trace) -> Check:
    meeting = _step_observation(trace, "call_meeting")
    false_alibi = _step_observation(trace, "false_alibi")
    return _check(
        "meeting_protocol",
        meeting.get("voting_open") is False
        and meeting.get("meeting_turns_remaining") == 1
        and false_alibi.get("voting_open") is True
        and false_alibi.get("meeting_turns_remaining") == 0,
        (
            "meeting="
            f"{meeting.get('voting_open')}/{meeting.get('meeting_turns_remaining')}; "
            "after_speak="
            f"{false_alibi.get('voting_open')}/{false_alibi.get('meeting_turns_remaining')}"
        ),
    )


def _check_false_alibi_penalty(trace: Trace) -> Check:
    observation = _step_observation(trace, "false_alibi")
    claims = observation.get("claims", [])
    claim = claims[-1] if claims else {}
    return _check(
        "false_alibi_penalty",
        observation.get("reward") == -1.0
        and claim.get("kind") == "self_location"
        and claim.get("room") == "MedBay"
        and claim.get("truth_value") is False,
        f"reward={observation.get('reward')}; claim={claim}",
    )


def _check_bot_vote_ejects_false_claimant(trace: Trace) -> Check:
    observation = _step_observation(trace, "vote_after_false_alibi")
    return _check(
        "bot_vote_ejects_false_claimant",
        observation.get("reward") == -0.5
        and observation.get("message_log", [])[-1:] == ["Ejected red"],
        f"reward={observation.get('reward')}; last_message={observation.get('message_log', [])[-1:]}",
    )


def _check(check_id: str, ok: bool, detail: str) -> Check:
    return {"id": check_id, "ok": ok, "detail": detail}


def _step_observation(trace: Trace, label: str) -> dict[str, Any]:
    for step in trace:
        if step.get("label") == label:
            return _observation(step)
    return {}


def _observation(step: dict[str, Any]) -> dict[str, Any]:
    observation = step.get("observation")
    return observation if isinstance(observation, dict) else {}


def main() -> None:
    print(json.dumps(run_eval_suite(), indent=2))


if __name__ == "__main__":
    main()
