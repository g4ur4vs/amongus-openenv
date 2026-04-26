from __future__ import annotations

import json
from typing import Any

from .engine import AmongUsEngine
from .models import CallMeeting, CompleteTask, Move, Observation, Speak, Vote


def run_golden_episode() -> list[dict[str, Any]]:
    """Run a deterministic episode that demonstrates verifiable social rewards."""
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    trace: list[dict[str, Any]] = []

    trace.append(_record("reset", "reset", engine.reset()))
    trace.append(
        _record(
            "move_electrical",
            "move to Electrical",
            engine.step(Move(room="Electrical")),
        )
    )
    trace.append(
        _record(
            "complete_task",
            "complete task in Electrical",
            engine.step(CompleteTask()),
        )
    )
    trace.append(
        _record(
            "call_meeting",
            "call emergency meeting",
            engine.step(CallMeeting()),
        )
    )
    trace.append(
        _record(
            "false_alibi",
            "claim: I was in MedBay",
            engine.step(Speak(message="I was in MedBay")),
        )
    )
    trace.append(
        _record(
            "vote_after_false_alibi",
            "vote blue after false alibi triggers bot votes",
            engine.step(Vote(target_id="blue")),
        )
    )

    return trace


def _record(label: str, action: str, observation: Observation) -> dict[str, Any]:
    return {
        "label": label,
        "action": action,
        "observation": observation.model_dump(mode="json"),
    }


def main() -> None:
    print(json.dumps(run_golden_episode(), indent=2))


if __name__ == "__main__":
    main()
