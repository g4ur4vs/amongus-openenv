import pytest
from pydantic import ValidationError

from amongus_env.engine import AmongUsEngine
from amongus_env.models import Move
from amongus_env.trace import (
    ReasoningLog,
    TraceStep,
    VerificationResult,
    parse_thought_tags,
    record_step,
)


def test_parse_thought_tags_extracts_first_thought_block() -> None:
    raw = "<thought>I was in Electrical, so MedBay is false.</thought> Vote blue."

    assert parse_thought_tags(raw) == "I was in Electrical, so MedBay is false."


def test_record_step_validates_and_serializes_optional_reasoning() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    observation = engine.reset()

    step = record_step(
        label="reset",
        action="reset",
        observation=observation,
        reasoning_text="<thought>Start in Cafeteria and inspect visible players.</thought>",
        verifications=[
            VerificationResult(
                check_id="spawn_location",
                ok=True,
                detail="Controlled player spawned in Cafeteria.",
            )
        ],
    )

    assert step["reasoning"]["thought"] == "Start in Cafeteria and inspect visible players."
    assert step["reasoning"]["raw"] == (
        "<thought>Start in Cafeteria and inspect visible players.</thought>"
    )
    assert step["verifications"][0]["check_id"] == "spawn_location"
    assert TraceStep.model_validate(step).observation.location == "Cafeteria"


def test_trace_step_rejects_unknown_fields() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    observation = engine.step(Move(room="Electrical"))

    with pytest.raises(ValidationError):
        TraceStep(
            label="bad",
            action="bad",
            observation=observation,
            reasoning=ReasoningLog(raw="<thought>x</thought>", thought="x"),
            unexpected=True,
        )
