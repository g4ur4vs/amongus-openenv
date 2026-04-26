from __future__ import annotations

from typing import Any, Optional, Union
from uuid import uuid4

from .engine import AmongUsEngine
from .models import Action, ActionAdapter, Observation

try:
    from openenv.core.env_server import create_app as _create_app
except Exception:  # pragma: no cover - depends on optional runtime install
    _create_app = None

try:
    from openenv.core.env_server.interfaces import Environment as _Environment
except Exception:  # pragma: no cover - depends on optional runtime install
    _Environment = object


class AmongUsEnvironment(_Environment):
    def __init__(
        self,
        seed: int = 0,
        controlled_player_id: str = "red",
        player_ids: Optional[list[str]] = None,
        impostor_ids: Optional[list[str]] = None,
    ) -> None:
        self.engine = AmongUsEngine(
            seed=seed,
            controlled_player_id=controlled_player_id,
            player_ids=player_ids,
            impostor_ids=impostor_ids,
        )
        self.episode_id = str(uuid4())
        self.step_count = 0

    def reset(self) -> Observation:
        self.episode_id = str(uuid4())
        self.step_count = 0
        return self.engine.reset()

    def step(self, action: Union[Action, dict[str, Any]]) -> Observation:
        parsed_action = (
            ActionAdapter.validate_python(action)
            if isinstance(action, dict)
            else action
        )
        self.step_count += 1
        return self.engine.step(parsed_action)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "done": self.engine.done,
            "phase": self.engine.phase.value,
            "winner": self.engine.winner.value if self.engine.winner else None,
            "voting_open": self.engine.voting_open,
            "meeting_turns_remaining": self.engine.meeting_turns_remaining,
        }


def create_http_app(environment: Optional[AmongUsEnvironment] = None) -> Any:
    if _create_app is None:
        raise RuntimeError("openenv is required to create the HTTP app")
    return _create_app(environment or AmongUsEnvironment(), Action, Observation)


app = create_http_app() if _create_app is not None else None
