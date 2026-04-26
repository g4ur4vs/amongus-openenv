from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import Field

from .models import Observation, StrictModel


THOUGHT_PATTERN = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)


class ReasoningLog(StrictModel):
    raw: str
    thought: Optional[str] = None


class VerificationResult(StrictModel):
    check_id: str
    ok: bool
    detail: str


class TraceStep(StrictModel):
    label: str
    action: str
    observation: Observation
    reasoning: Optional[ReasoningLog] = None
    verifications: list[VerificationResult] = Field(default_factory=list)


def parse_thought_tags(text: str) -> Optional[str]:
    match = THOUGHT_PATTERN.search(text)
    if match is None:
        return None
    thought = match.group(1).strip()
    return thought or None


def record_step(
    label: str,
    action: str,
    observation: Observation,
    reasoning_text: Optional[str] = None,
    verifications: Optional[list[VerificationResult]] = None,
) -> dict[str, Any]:
    reasoning = (
        ReasoningLog(raw=reasoning_text, thought=parse_thought_tags(reasoning_text))
        if reasoning_text is not None
        else None
    )
    step = TraceStep(
        label=label,
        action=action,
        observation=observation,
        reasoning=reasoning,
        verifications=verifications or [],
    )
    payload = {
        "label": step.label,
        "action": step.action,
        "observation": step.observation.model_dump(mode="json"),
    }
    if step.reasoning is not None:
        payload["reasoning"] = step.reasoning.model_dump(mode="json")
    if step.verifications:
        payload["verifications"] = [
            verification.model_dump(mode="json") for verification in step.verifications
        ]
    return payload
