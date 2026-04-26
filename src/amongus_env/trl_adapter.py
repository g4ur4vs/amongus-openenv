from __future__ import annotations

from typing import Any, Iterable, Optional

from .models import (
    CallMeeting,
    CompleteTask,
    Kill,
    Move,
    Observation,
    PassMeeting,
    ReportBody,
    Speak,
    Vote,
)
from .openenv_server import AmongUsEnvironment


class AmongUsToolEnv:
    """TRL `environment_factory` adapter.

    TRL exposes public methods on this class as tools. Each method converts
    a tool call into a typed OpenEnv action and stores the latest RLVR reward.
    """

    def __init__(
        self,
        seed: int = 0,
        controlled_player_id: str = "red",
        player_ids: Optional[list[str]] = None,
        impostor_ids: Optional[list[str]] = None,
    ) -> None:
        self.env = AmongUsEnvironment(
            seed=seed,
            controlled_player_id=controlled_player_id,
            player_ids=player_ids,
            impostor_ids=impostor_ids,
        )
        self.reward = 0.0
        self.last_observation: Optional[Observation] = None

    def reset(self, **kwargs: Any) -> str:
        """Reset the match and return the initial observation summary."""
        self.last_observation = self.env.reset()
        self.reward = self.last_observation.reward
        return self._summarize(self.last_observation)

    def move(self, room: str) -> str:
        """
        Move the controlled player to an adjacent room.

        Args:
            room: Destination room name.
        """
        return self._step(Move(room=room))

    def complete_task(self) -> str:
        """Complete an unfinished crewmate task in the current room."""
        return self._step(CompleteTask())

    def kill(self, target_id: str) -> str:
        """
        Kill a visible crewmate as the impostor.

        Args:
            target_id: Player id to kill.
        """
        return self._step(Kill(target_id=target_id))

    def report_body(self) -> str:
        """Report a body in the current room and enter the meeting phase."""
        return self._step(ReportBody())

    def call_meeting(self) -> str:
        """Call an emergency meeting."""
        return self._step(CallMeeting())

    def speak(self, message: str) -> str:
        """
        Make a meeting statement that may be parsed into verifiable claims.

        Args:
            message: Meeting message, e.g. "I was in Electrical".
        """
        return self._step(Speak(message=message))

    def pass_meeting(self) -> str:
        """Use the controlled player's required discussion turn without making a claim."""
        return self._step(PassMeeting())

    def vote(self, target_id: str) -> str:
        """
        Vote to eject a player during the meeting phase.

        Args:
            target_id: Player id to vote for.
        """
        return self._step(Vote(target_id=target_id))

    def _step(self, action: Any) -> str:
        self.last_observation = self.env.step(action)
        self.reward = self.last_observation.reward
        return self._summarize(self.last_observation)

    def _summarize(self, observation: Observation) -> str:
        """Return a compact tool result with the latest social evidence."""
        visible = ", ".join(observation.visible_players) or "none"
        tasks = ", ".join(
            f"{task.name}@{task.room}:{'done' if task.completed else 'open'}"
            for task in observation.task_list
        )
        if not tasks:
            tasks = "none"
        message_log = " | ".join(observation.message_log[-3:]) or "none"
        discussion_log = " | ".join(observation.discussion_log[-3:]) or "none"
        claims = " | ".join(
            self._summarize_claim(claim) for claim in observation.claims[-3:]
        ) or "none"
        return (
            f"role={observation.role.value}; location={observation.location}; "
            f"phase={observation.phase.value}; visible={visible}; tasks={tasks}; "
            f"voting_open={observation.voting_open}; "
            f"meeting_turns_remaining={observation.meeting_turns_remaining}; "
            f"message_log={message_log}; discussion_log={discussion_log}; "
            f"claims={claims}; "
            f"reward={observation.reward}; done={observation.done}; "
            f"winner={observation.winner.value if observation.winner else 'none'}"
        )

    def _summarize_claim(self, claim: Any) -> str:
        truth_value = "true" if claim.truth_value else "false"
        if claim.target_id:
            return (
                f"{claim.kind.value}({claim.speaker_id} saw "
                f"{claim.target_id} in {claim.room})={truth_value}"
            )
        return (
            f"{claim.kind.value}({claim.speaker_id} in "
            f"{claim.room})={truth_value}"
        )


def reward_from_game_state(environments: Iterable[AmongUsToolEnv], **kwargs: Any) -> list[float]:
    return [environment.reward for environment in environments]
