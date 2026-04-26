from __future__ import annotations

import json
from typing import Any

from .engine import AmongUsEngine
from .golden_episode import run_golden_episode
from .models import CallMeeting, CompleteTask, FakeTask, Kill, Move, PassMeeting, Speak, Vent, Vote
from .trace import record_step

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
    scenarios = [
        evaluate_trace(run_golden_episode()),
        _run_invalid_move_eval(),
        _run_all_tasks_eval(),
        _run_meeting_pass_eval(),
        _run_impostor_parity_eval(),
        _run_kill_cooldown_eval(),
        _run_impostor_fake_task_eval(),
        _run_vent_claim_eval(),
        _run_no_majority_eval(),
        _run_multi_impostor_eval(),
    ]
    passed = sum(1 for scenario in scenarios if scenario["ok"])
    return {
        "schema_version": 1,
        "ok": passed == len(scenarios),
        "summary": {
            "scenarios": len(scenarios),
            "passed": passed,
            "failed": len(scenarios) - passed,
        },
        "scenarios": scenarios,
    }


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


def _run_invalid_move_eval() -> dict[str, Any]:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    reset = engine.reset()
    invalid = engine.step(Move(room="Navigation"))
    trace = [
        _record("reset", "reset", reset),
        _record("invalid_move", "move to Navigation", invalid),
    ]
    observation = _step_observation(trace, "invalid_move")
    return _scenario_result(
        "invalid_move_no_state_change",
        trace,
        [
            _check(
                "invalid_move_penalty",
                observation.get("location") == "Cafeteria"
                and observation.get("reward") == -0.1
                and "Invalid move" in observation.get("message_log", [])[-1],
                f"location={observation.get('location')}; reward={observation.get('reward')}",
            )
        ],
    )


def _run_all_tasks_eval() -> dict[str, Any]:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    trace = [_record("reset", "reset", engine.reset())]
    for label, action_label, action in [
        ("move_electrical", "move to Electrical", Move(room="Electrical")),
        ("task_electrical", "complete Electrical task", CompleteTask()),
        ("move_cafeteria_1", "move to Cafeteria", Move(room="Cafeteria")),
        ("move_medbay", "move to MedBay", Move(room="MedBay")),
        ("task_medbay", "complete MedBay task", CompleteTask()),
        ("move_cafeteria_2", "move to Cafeteria", Move(room="Cafeteria")),
        ("move_admin", "move to Admin", Move(room="Admin")),
        ("task_admin", "complete Admin task", CompleteTask()),
    ]:
        trace.append(_record(label, action_label, engine.step(action)))
    final = _step_observation(trace, "task_admin")
    return _scenario_result(
        "crewmate_task_route",
        trace,
        [
            _check(
                "controlled_crewmate_tasks_complete",
                final.get("done") is False
                and final.get("winner") is None
                and final.get("phase") == "tasks"
                and final.get("reward") == 0.2
                and all(task.get("completed") for task in final.get("task_list", [])),
                f"reward={final.get('reward')}; winner={final.get('winner')}",
            )
        ],
    )


def _run_meeting_pass_eval() -> dict[str, Any]:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("call_meeting", "call meeting", engine.step(CallMeeting())),
        _record("pass_meeting", "pass meeting", engine.step(PassMeeting())),
        _record("vote_blue", "vote blue", engine.step(Vote(target_id="blue"))),
    ]
    passed = _step_observation(trace, "pass_meeting")
    vote = _step_observation(trace, "vote_blue")
    return _scenario_result(
        "meeting_pass_bot_majority",
        trace,
        [
            _check(
                "pass_opens_voting",
                passed.get("voting_open") is True
                and passed.get("meeting_turns_remaining") == 0
                and passed.get("reward") == 0.0,
                f"voting={passed.get('voting_open')}; reward={passed.get('reward')}",
            ),
            _check(
                "bot_votes_eject_controlled_target",
                vote.get("reward") == 1.5
                and vote.get("done") is True
                and vote.get("winner") == "crewmates"
                and vote.get("message_log", [])[-2:-1] == ["Ejected blue"],
                f"reward={vote.get('reward')}; ejection_message={vote.get('message_log', [])[-2:-1]}",
            ),
        ],
    )


def _run_impostor_parity_eval() -> dict[str, Any]:
    engine = AmongUsEngine(
        seed=1,
        impostor_ids=["red"],
        player_ids=["red", "blue", "green"],
    )
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("kill_blue", "kill blue", engine.step(Kill(target_id="blue"))),
    ]
    final = _step_observation(trace, "kill_blue")
    return _scenario_result(
        "impostor_parity_win",
        trace,
        [
            _check(
                "kill_plus_parity_win",
                final.get("done") is True
                and final.get("winner") == "impostors"
                and final.get("reward") == 1.5,
                f"reward={final.get('reward')}; winner={final.get('winner')}",
            )
        ],
    )


def _run_kill_cooldown_eval() -> dict[str, Any]:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"])
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("kill_blue", "kill blue", engine.step(Kill(target_id="blue"))),
        _record("kill_green", "kill green", engine.step(Kill(target_id="green"))),
    ]
    first = _step_observation(trace, "kill_blue")
    second = _step_observation(trace, "kill_green")
    return _scenario_result(
        "kill_cooldown_blocks_second_kill",
        trace,
        [
            _check(
                "second_kill_blocked_by_cooldown",
                first.get("reward") == 0.5
                and second.get("reward") == -0.1
                and second.get("message_log", [])[-1:] == ["Kill is on cooldown"],
                f"first={first.get('reward')}; second={second.get('reward')}",
            )
        ],
    )


def _run_impostor_fake_task_eval() -> dict[str, Any]:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"])
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("move_electrical", "move to Electrical", engine.step(Move(room="Electrical"))),
        _record("fake_task", "fake task in Electrical", engine.step(FakeTask())),
    ]
    final = _step_observation(trace, "fake_task")
    return _scenario_result(
        "impostor_fake_task",
        trace,
        [
            _check(
                "fake_task_no_reward",
                final.get("reward") == 0.0
                and final.get("task_list") == []
                and final.get("message_log", [])[-1:] == ["Faked task Fix Wiring"],
                f"reward={final.get('reward')}; last_message={final.get('message_log', [])[-1:]}",
            )
        ],
    )


def _run_vent_claim_eval() -> dict[str, Any]:
    engine = AmongUsEngine(
        seed=1,
        controlled_player_id="red",
        impostor_ids=["red"],
        player_ids=["red", "blue", "green", "yellow"],
    )
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("move_electrical", "move to Electrical", engine.step(Move(room="Electrical"))),
        _record("vent_medbay", "vent to MedBay", engine.step(Vent(room="MedBay"))),
    ]
    engine.config.controlled_player_id = "green"
    trace.extend(
        [
            _record("call_meeting", "call meeting", engine.step(CallMeeting())),
            _record("saw_vent", "claim: I saw red vent", engine.step(Speak(message="I saw red vent"))),
        ]
    )
    claim_observation = _step_observation(trace, "saw_vent")
    claims = claim_observation.get("claims", [])
    claim = claims[-1] if claims else {}
    return _scenario_result(
        "vent_claim_verification",
        trace,
        [
            _check(
                "true_vent_claim",
                claim.get("kind") == "saw_vent"
                and claim.get("target_id") == "red"
                and claim.get("truth_value") is True,
                f"claim={claim}",
            )
        ],
    )


def _run_no_majority_eval() -> dict[str, Any]:
    engine = AmongUsEngine(
        seed=1,
        impostor_ids=["blue"],
        player_ids=["red", "blue", "green"],
    )
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("call_meeting", "call meeting", engine.step(CallMeeting())),
        _record("pass_meeting", "pass meeting", engine.step(PassMeeting())),
    ]
    engine._bot_votes = lambda human_target_id: {}
    trace.append(_record("vote_blue", "vote blue", engine.step(Vote(target_id="blue"))))
    vote = _step_observation(trace, "vote_blue")
    return _scenario_result(
        "meeting_no_majority",
        trace,
        [
            _check(
                "no_majority_no_ejection",
                vote.get("reward") == 0.0
                and vote.get("phase") == "tasks"
                and vote.get("message_log", [])[-1:] == ["No majority; nobody ejected"],
                f"reward={vote.get('reward')}; last_message={vote.get('message_log', [])[-1:]}",
            )
        ],
    )


def _run_multi_impostor_eval() -> dict[str, Any]:
    engine = AmongUsEngine(
        seed=1,
        impostor_ids=["red", "blue"],
        player_ids=["red", "blue", "green", "yellow"],
    )
    trace = [
        _record("reset", "reset", engine.reset()),
        _record("kill_green", "kill green", engine.step(Kill(target_id="green"))),
    ]
    final = _step_observation(trace, "kill_green")
    return _scenario_result(
        "multi_impostor_parity",
        trace,
        [
            _check(
                "multi_impostor_parity_win",
                final.get("done") is True
                and final.get("winner") == "impostors"
                and final.get("reward") == 1.5,
                f"reward={final.get('reward')}; winner={final.get('winner')}",
            )
        ],
    )


def _scenario_result(episode: str, trace: Trace, checks: list[Check]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "episode": episode,
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


def _record(label: str, action: str, observation: Any) -> dict[str, Any]:
    return record_step(label, action, observation)


def main() -> None:
    print(json.dumps(run_eval_suite(), indent=2))


if __name__ == "__main__":
    main()
