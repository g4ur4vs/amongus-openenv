from __future__ import annotations

import json
from typing import Any, Optional

from .engine import AmongUsEngine
from .models import CallMeeting, CompleteTask, Move, Speak, Vote
from .trace import VerificationResult, record_step


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


def run_golden_reasoning_trace() -> list[dict[str, Any]]:
    """Run the golden episode with judge-facing reasoning and verifications."""
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    trace: list[dict[str, Any]] = []

    trace.append(_record("reset", "reset", engine.reset()))
    trace.append(
        _record(
            "move_electrical",
            "move to Electrical",
            engine.step(Move(room="Electrical")),
            reasoning_text=(
                "<thought>Electrical is reachable from Cafeteria and has an open "
                "Fix Wiring task.</thought>"
            ),
            verifications=[
                VerificationResult(
                    check_id="location_history_updated",
                    ok=True,
                    detail="Controlled player location is Electrical after a legal move.",
                )
            ],
        )
    )
    trace.append(
        _record(
            "complete_task",
            "complete task in Electrical",
            engine.step(CompleteTask()),
            reasoning_text="<thought>Fix Wiring is open in my current room.</thought>",
            verifications=[
                VerificationResult(
                    check_id="task_reward_verified",
                    ok=True,
                    detail="Completing the Electrical task gives the +0.2 RLVR reward.",
                )
            ],
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
            reasoning_text=(
                "<thought>My private location history is Cafeteria -> Electrical, "
                "so claiming MedBay should be verifiably false.</thought>"
            ),
            verifications=[
                VerificationResult(
                    check_id="false_alibi_verified",
                    ok=True,
                    detail="Latest self-location claim is MedBay with truth_value=false.",
                )
            ],
        )
    )
    trace.append(
        _record(
            "vote_after_false_alibi",
            "vote blue after false alibi triggers bot votes",
            engine.step(Vote(target_id="blue")),
            reasoning_text=(
                "<thought>The false alibi has been logged, so bot votes should punish "
                "the false claimant instead of following my blue vote.</thought>"
            ),
            verifications=[
                VerificationResult(
                    check_id="false_claimant_ejected",
                    ok=True,
                    detail="Red is ejected after the false alibi is detected.",
                )
            ],
        )
    )

    return trace


def _record(
    label: str,
    action: str,
    observation: Any,
    reasoning_text: Optional[str] = None,
    verifications: Optional[list[VerificationResult]] = None,
) -> dict[str, Any]:
    return record_step(label, action, observation, reasoning_text, verifications)


def main() -> None:
    print(json.dumps(run_golden_episode(), indent=2))


if __name__ == "__main__":
    main()
