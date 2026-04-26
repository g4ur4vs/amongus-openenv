from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class PlayerRole(str, Enum):
    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"


class Phase(str, Enum):
    TASKS = "tasks"
    MEETING = "meeting"
    COMPLETE = "complete"


class Winner(str, Enum):
    CREWMATES = "crewmates"
    IMPOSTORS = "impostors"


class ClaimKind(str, Enum):
    SELF_LOCATION = "self_location"
    SAW_PLAYER = "saw_player"


class Move(StrictModel):
    type: Literal["move"] = "move"
    room: str


class CompleteTask(StrictModel):
    type: Literal["complete_task"] = "complete_task"


class Kill(StrictModel):
    type: Literal["kill"] = "kill"
    target_id: str


class ReportBody(StrictModel):
    type: Literal["report_body"] = "report_body"


class CallMeeting(StrictModel):
    type: Literal["call_meeting"] = "call_meeting"


class Vote(StrictModel):
    type: Literal["vote"] = "vote"
    target_id: str


class Speak(StrictModel):
    type: Literal["speak"] = "speak"
    message: str


class PassMeeting(StrictModel):
    type: Literal["pass"] = "pass"


Action = Annotated[
    Union[
        Move,
        CompleteTask,
        Kill,
        ReportBody,
        CallMeeting,
        Vote,
        Speak,
        PassMeeting,
    ],
    Field(discriminator="type"),
]
ActionAdapter = TypeAdapter(Action)


class TaskState(StrictModel):
    room: str
    name: str
    completed: bool = False


class VisiblePlayer(StrictModel):
    player_id: str
    location: str
    alive: bool = True


class Claim(StrictModel):
    kind: ClaimKind
    speaker_id: str
    room: str
    truth_value: bool
    target_id: Optional[str] = None


class Observation(StrictModel):
    role: PlayerRole
    location: str
    visible_players: list[Union[str, VisiblePlayer]]
    task_list: list[TaskState]
    message_log: list[str]
    discussion_log: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    phase: Phase
    reward: float = 0.0
    done: bool = False
    winner: Optional[Winner] = None
    voting_open: bool = False
    meeting_turns_remaining: int = 0


class PlayerState(StrictModel):
    player_id: str
    role: PlayerRole
    location: str
    alive: bool = True
    ejected: bool = False


class GameConfig(StrictModel):
    seed: int = 0
    controlled_player_id: str = "red"
    player_ids: list[str] = Field(
        default_factory=lambda: ["red", "blue", "green", "yellow"]
    )
