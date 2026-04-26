from __future__ import annotations

import re
import random
from collections import Counter
from typing import Iterable, Optional

from .models import (
    Action,
    CallMeeting,
    Claim,
    ClaimKind,
    CompleteTask,
    FakeTask,
    GameConfig,
    Kill,
    Move,
    Observation,
    PassMeeting,
    Phase,
    PlayerRole,
    PlayerState,
    ReportBody,
    Speak,
    TaskState,
    Vent,
    Vote,
    Winner,
)


ROOM_GRAPH = {
    "Cafeteria": {"Electrical", "MedBay", "Admin"},
    "Electrical": {"Cafeteria", "Storage", "Security"},
    "MedBay": {"Cafeteria", "Security"},
    "Admin": {"Cafeteria", "Storage"},
    "Storage": {"Admin", "Electrical", "Navigation"},
    "Navigation": {"Storage"},
    "Security": {"Electrical", "MedBay"},
}

VENT_GRAPH = {
    "Electrical": {"MedBay", "Security"},
    "MedBay": {"Electrical", "Security"},
    "Security": {"Electrical", "MedBay"},
}

DEFAULT_TASKS = [
    TaskState(room="Electrical", name="Fix Wiring"),
    TaskState(room="MedBay", name="Submit Scan"),
    TaskState(room="Admin", name="Swipe Card"),
]


class AmongUsEngine:
    def __init__(
        self,
        seed: int = 0,
        controlled_player_id: str = "red",
        player_ids: Optional[list[str]] = None,
        impostor_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = GameConfig(
            seed=seed,
            controlled_player_id=controlled_player_id,
            player_ids=player_ids or ["red", "blue", "green", "yellow"],
        )
        self._explicit_impostor_ids = impostor_ids is not None
        self._rng = random.Random(seed)
        self.impostor_count = 1
        self.impostor_ids = set(impostor_ids or [])
        self.players: dict[str, PlayerState] = {}
        self.tasks_by_player: dict[str, list[TaskState]] = {}
        self.location_history: dict[str, list[str]] = {}
        self.faked_tasks: set[tuple[str, str]] = set()
        self.vent_since_meeting: dict[str, bool] = {}
        self.discussion_log: list[str] = []
        self.claims: list[Claim] = []
        self.dead_bodies: set[str] = set()
        self.message_log: list[str] = []
        self.phase = Phase.TASKS
        self.winner: Optional[Winner] = None
        self.done = False
        self.kill_cooldowns: dict[str, int] = {}
        self.last_reward = 0.0
        self.voting_open = False
        self.meeting_turns_remaining = 0

    @property
    def controlled_player(self) -> PlayerState:
        return self.players[self.config.controlled_player_id]

    def reset(self) -> Observation:
        if not self._explicit_impostor_ids:
            self.impostor_ids = set(
                self._rng.sample(self.config.player_ids, k=self.impostor_count)
            )
        self.players = {}
        self.tasks_by_player = {}
        self.location_history = {}
        self.faked_tasks = set()
        self.vent_since_meeting = {player_id: False for player_id in self.config.player_ids}
        self.discussion_log = []
        self.claims = []
        self.dead_bodies = set()
        self.message_log = ["Match reset"]
        self.phase = Phase.TASKS
        self.winner = None
        self.done = False
        self.last_reward = 0.0
        self.voting_open = False
        self.meeting_turns_remaining = 0
        self.kill_cooldowns = {player_id: 0 for player_id in self.impostor_ids}

        for player_id in self.config.player_ids:
            role = (
                PlayerRole.IMPOSTOR
                if player_id in self.impostor_ids
                else PlayerRole.CREWMATE
            )
            self.players[player_id] = PlayerState(
                player_id=player_id,
                role=role,
                location="Cafeteria",
            )
            self.location_history[player_id] = ["Cafeteria"]
            if role is PlayerRole.CREWMATE:
                self.tasks_by_player[player_id] = [
                    task.model_copy() for task in DEFAULT_TASKS
                ]

        return self.observe()

    def observe(self, reward: Optional[float] = None) -> Observation:
        player = self.controlled_player
        visible_players = [
            other.player_id
            for other in self.players.values()
            if other.player_id != player.player_id
            and other.alive
            and not other.ejected
            and other.location == player.location
        ]
        return Observation(
            role=player.role,
            location=player.location,
            visible_players=sorted(visible_players),
            task_list=self.tasks_by_player.get(player.player_id, []),
            message_log=self.message_log[-20:],
            discussion_log=self.discussion_log[-20:],
            claims=self.claims[-20:],
            phase=self.phase,
            reward=self.last_reward if reward is None else reward,
            done=self.done,
            winner=self.winner,
            voting_open=self.voting_open,
            meeting_turns_remaining=self.meeting_turns_remaining,
        )

    def step(self, action: Action) -> Observation:
        if self.done:
            return self._illegal("Game is already complete")
        if isinstance(action, Move):
            reward = self._move(action.room)
        elif isinstance(action, CompleteTask):
            reward = self._complete_task()
        elif isinstance(action, FakeTask):
            reward = self._fake_task()
        elif isinstance(action, Vent):
            reward = self._vent(action.room)
        elif isinstance(action, Kill):
            reward = self._kill(action.target_id)
        elif isinstance(action, ReportBody):
            reward = self._report_body()
        elif isinstance(action, CallMeeting):
            reward = self._call_meeting()
        elif isinstance(action, Vote):
            reward = self._vote(action.target_id)
        elif isinstance(action, Speak):
            reward = self._speak(action.message)
        elif isinstance(action, PassMeeting):
            reward = self._pass_meeting()
        else:
            reward = self._illegal("Unsupported action").reward

        if not self.done:
            reward += self._check_win_conditions()
        self.last_reward = reward
        return self.observe(reward=reward)

    def _move(self, room: str) -> float:
        player = self.controlled_player
        if self.phase is not Phase.TASKS:
            return self._illegal("Cannot move during meeting").reward
        if not player.alive or player.ejected:
            return self._illegal("Eliminated players cannot move").reward
        if room not in ROOM_GRAPH[player.location]:
            return self._illegal(f"Invalid move from {player.location} to {room}").reward

        player.location = room
        self.location_history[player.player_id].append(room)
        self._tick_cooldowns()
        self.message_log.append(f"Moved to {room}")
        return 0.0

    def _complete_task(self) -> float:
        player = self.controlled_player
        if self.phase is not Phase.TASKS:
            return self._illegal("Cannot complete tasks during meeting").reward
        if player.role is not PlayerRole.CREWMATE:
            return self._illegal("Impostors cannot complete real tasks").reward

        for task in self.tasks_by_player.get(player.player_id, []):
            if task.room == player.location and not task.completed:
                task.completed = True
                self.message_log.append(f"Completed task {task.name}")
                return 0.2

        return self._illegal(f"No incomplete task in {player.location}").reward

    def _fake_task(self) -> float:
        player = self.controlled_player
        if self.phase is not Phase.TASKS:
            return self._illegal("Cannot fake tasks during meeting").reward
        if player.role is not PlayerRole.IMPOSTOR:
            return self._illegal("Crewmates cannot fake tasks").reward
        if not player.alive or player.ejected:
            return self._illegal("Eliminated players cannot fake tasks").reward

        task = self._task_for_room(player.location)
        if task is None:
            return self._illegal(f"No task to fake in {player.location}").reward
        key = (player.player_id, task.room)
        if key in self.faked_tasks:
            return self._illegal(f"Already faked task in {task.room}").reward

        self.faked_tasks.add(key)
        self._tick_cooldowns()
        self.message_log.append(f"Faked task {task.name}")
        return 0.0

    def _task_for_room(self, room: str) -> Optional[TaskState]:
        for task in DEFAULT_TASKS:
            if task.room == room:
                return task
        return None

    def _vent(self, room: str) -> float:
        player = self.controlled_player
        if self.phase is not Phase.TASKS:
            return self._illegal("Cannot vent during meeting").reward
        if player.role is not PlayerRole.IMPOSTOR:
            return self._illegal("Crewmates cannot vent").reward
        if not player.alive or player.ejected:
            return self._illegal("Eliminated players cannot vent").reward
        if room not in VENT_GRAPH.get(player.location, set()):
            return self._illegal(f"Invalid vent from {player.location} to {room}").reward

        player.location = room
        self.location_history[player.player_id].append(room)
        self.vent_since_meeting[player.player_id] = True
        self._tick_cooldowns()
        self.message_log.append(f"Vented to {room}")
        return 0.0

    def _kill(self, target_id: str) -> float:
        player = self.controlled_player
        if self.phase is not Phase.TASKS:
            return self._illegal("Cannot kill during meeting").reward
        if player.role is not PlayerRole.IMPOSTOR:
            return self._illegal("Crewmates cannot kill").reward
        if self.kill_cooldowns.get(player.player_id, 0) > 0:
            return self._illegal("Kill is on cooldown").reward

        target = self.players.get(target_id)
        if (
            target is None
            or target.role is PlayerRole.IMPOSTOR
            or not target.alive
            or target.ejected
            or target.location != player.location
        ):
            return self._illegal(f"Cannot kill {target_id}").reward

        target.alive = False
        self.dead_bodies.add(target_id)
        self.kill_cooldowns[player.player_id] = 2
        self.message_log.append(f"Killed {target_id}")
        return 0.5

    def _report_body(self) -> float:
        player = self.controlled_player
        body_here = any(
            self.players[player_id].location == player.location
            for player_id in self.dead_bodies
        )
        if self.phase is not Phase.TASKS or not body_here:
            return self._illegal("No reportable body here").reward

        self.phase = Phase.MEETING
        self._start_meeting_protocol()
        self.dead_bodies.clear()
        self.message_log.append("Reported body")
        return 0.0

    def _call_meeting(self) -> float:
        if self.phase is not Phase.TASKS:
            return self._illegal("Meeting already active").reward
        self.phase = Phase.MEETING
        self._start_meeting_protocol()
        self.message_log.append("Emergency meeting called")
        return 0.0

    def _start_meeting_protocol(self) -> None:
        self.voting_open = False
        self.meeting_turns_remaining = 1

    def _speak(self, message: str) -> float:
        if self.phase is not Phase.MEETING:
            return self._illegal("Cannot speak outside meeting").reward
        if self.voting_open:
            return self._illegal("Voting is already open").reward

        speaker_id = self.controlled_player.player_id
        entry = f"{speaker_id}: {message}"
        self.discussion_log.append(entry)
        self.message_log.append(entry)
        claim = self._parse_claim(speaker_id=speaker_id, message=message)
        if claim is not None:
            self.claims.append(claim)
            if (
                claim.kind in {ClaimKind.SELF_LOCATION, ClaimKind.SAW_VENT}
                and claim.truth_value is False
            ):
                self._open_voting()
                return -1.0
        self._open_voting()
        return 0.0

    def _pass_meeting(self) -> float:
        if self.phase is not Phase.MEETING:
            return self._illegal("Cannot pass outside meeting").reward
        if self.voting_open:
            return self._illegal("Voting is already open").reward

        speaker_id = self.controlled_player.player_id
        entry = f"{speaker_id}: pass"
        self.discussion_log.append(entry)
        self.message_log.append(entry)
        self._open_voting()
        return 0.0

    def _open_voting(self) -> None:
        self.meeting_turns_remaining = 0
        self.voting_open = True

    def _parse_claim(self, speaker_id: str, message: str) -> Optional[Claim]:
        room_pattern = "|".join(re.escape(room) for room in ROOM_GRAPH)

        self_location_match = re.fullmatch(
            rf"\s*i\s+was\s+in\s+({room_pattern})\s*",
            message,
            flags=re.IGNORECASE,
        )
        if self_location_match:
            room = self._canonical_room(self_location_match.group(1))
            return Claim(
                kind=ClaimKind.SELF_LOCATION,
                speaker_id=speaker_id,
                room=room,
                truth_value=room in self.location_history.get(speaker_id, []),
            )

        saw_player_match = re.fullmatch(
            rf"\s*i\s+saw\s+([A-Za-z0-9_-]+)\s+in\s+({room_pattern})\s*",
            message,
            flags=re.IGNORECASE,
        )
        if saw_player_match:
            target_id = saw_player_match.group(1)
            room = self._canonical_room(saw_player_match.group(2))
            return Claim(
                kind=ClaimKind.SAW_PLAYER,
                speaker_id=speaker_id,
                target_id=target_id,
                room=room,
                truth_value=room in self.location_history.get(target_id, []),
            )

        saw_vent_match = re.fullmatch(
            r"\s*i\s+saw\s+([A-Za-z0-9_-]+)\s+vent\s*",
            message,
            flags=re.IGNORECASE,
        )
        if saw_vent_match:
            target_id = saw_vent_match.group(1)
            return Claim(
                kind=ClaimKind.SAW_VENT,
                speaker_id=speaker_id,
                target_id=target_id,
                room=self.controlled_player.location,
                truth_value=self.vent_since_meeting.get(target_id, False),
            )

        accusation_match = re.fullmatch(
            r"\s*(?:i\s+accuse|i\s+think)\s+([A-Za-z0-9_-]+)(?:\s+is\s+the\s+impostor)?\s*",
            message,
            flags=re.IGNORECASE,
        )
        if accusation_match:
            target_id = accusation_match.group(1)
            target = self.players.get(target_id)
            return Claim(
                kind=ClaimKind.ACCUSE_IMPOSTOR,
                speaker_id=speaker_id,
                target_id=target_id,
                room=self.controlled_player.location,
                truth_value=target is not None and target.role is PlayerRole.IMPOSTOR,
            )

        return None

    def _canonical_room(self, room: str) -> str:
        lookup = {known_room.lower(): known_room for known_room in ROOM_GRAPH}
        return lookup[room.lower()]

    def _vote(self, target_id: str) -> float:
        if self.phase is not Phase.MEETING:
            return self._illegal("Cannot vote outside meeting").reward
        if not self.voting_open:
            return self._illegal("Cannot vote before discussion is complete").reward

        target = self.players.get(target_id)
        if target is None or target.ejected or not target.alive:
            return self._illegal(f"Cannot vote for {target_id}").reward

        ballots = {self.controlled_player.player_id: target_id}
        ballots.update(self._bot_votes(target_id))
        active_voters = [
            player
            for player in self.players.values()
            if player.alive and not player.ejected
        ]
        vote_counts = Counter(ballots.values())
        if not vote_counts:
            return self._no_majority()

        top_target_id, top_votes = vote_counts.most_common(1)[0]
        tied = list(vote_counts.values()).count(top_votes) > 1
        has_majority = top_votes > len(active_voters) / 2
        if tied or not has_majority:
            return self._no_majority()

        target = self.players[top_target_id]
        target.ejected = True
        target.alive = False
        self.message_log.append(f"Ejected {top_target_id}")
        reward = 0.0
        controlled = self.controlled_player
        if target.role is PlayerRole.IMPOSTOR:
            reward += 0.5 if controlled.role is PlayerRole.CREWMATE else -0.5
        elif target.player_id == controlled.player_id:
            reward -= 0.5
        self.phase = Phase.TASKS
        self._reset_meeting_protocol()
        return reward

    def _bot_votes(self, human_target_id: str) -> dict[str, str]:
        false_speaker_id = self._latest_false_penalized_claim_speaker()
        if false_speaker_id is None:
            accused_impostor_id = self._latest_true_accusation_target()
            if accused_impostor_id is not None:
                return self._bot_vote_map(accused_impostor_id)
            return {
                player.player_id: human_target_id
                for player in self.players.values()
                if player.player_id != self.controlled_player.player_id
                and player.alive
                and not player.ejected
            }
        target = self.players.get(false_speaker_id)
        if target is None or not target.alive or target.ejected:
            return {}
        return self._bot_vote_map(false_speaker_id)

    def _bot_vote_map(self, target_id: str) -> dict[str, str]:
        return {
            player.player_id: target_id
            for player in self.players.values()
            if player.player_id != self.controlled_player.player_id
            and player.alive
            and not player.ejected
        }

    def _latest_false_penalized_claim_speaker(self) -> Optional[str]:
        for claim in reversed(self.claims):
            if (
                claim.kind in {ClaimKind.SELF_LOCATION, ClaimKind.SAW_VENT}
                and claim.truth_value is False
            ):
                return claim.speaker_id
        return None

    def _latest_true_accusation_target(self) -> Optional[str]:
        for claim in reversed(self.claims):
            if (
                claim.kind is ClaimKind.ACCUSE_IMPOSTOR
                and claim.truth_value is True
                and claim.target_id is not None
            ):
                target = self.players.get(claim.target_id)
                if target is not None and target.alive and not target.ejected:
                    return claim.target_id
        return None

    def _no_majority(self) -> float:
        self.message_log.append("No majority; nobody ejected")
        self.phase = Phase.TASKS
        self._reset_meeting_protocol()
        return 0.0

    def _reset_meeting_protocol(self) -> None:
        self.voting_open = False
        self.meeting_turns_remaining = 0
        self.vent_since_meeting = {
            player_id: False for player_id in self.config.player_ids
        }

    def _check_win_conditions(self) -> float:
        alive_crewmates = [
            player
            for player in self.players.values()
            if player.role is PlayerRole.CREWMATE and player.alive and not player.ejected
        ]
        alive_impostors = [
            player
            for player in self.players.values()
            if player.role is PlayerRole.IMPOSTOR and player.alive and not player.ejected
        ]

        if not alive_impostors:
            return self._finish(Winner.CREWMATES)

        all_tasks_done = all(
            task.completed
            for tasks in self.tasks_by_player.values()
            for task in tasks
        )
        if all_tasks_done:
            return self._finish(Winner.CREWMATES)

        if len(alive_impostors) >= len(alive_crewmates):
            return self._finish(Winner.IMPOSTORS)

        return 0.0

    def _finish(self, winner: Winner) -> float:
        self.done = True
        self.phase = Phase.COMPLETE
        self._reset_meeting_protocol()
        self.winner = winner
        self.message_log.append(f"{winner.value} win")
        return 1.0 if self._controlled_side_won(winner) else 0.0

    def _controlled_side_won(self, winner: Winner) -> bool:
        role = self.controlled_player.role
        return (
            winner is Winner.CREWMATES
            and role is PlayerRole.CREWMATE
            or winner is Winner.IMPOSTORS
            and role is PlayerRole.IMPOSTOR
        )

    def _illegal(self, message: str) -> Observation:
        self.message_log.append(message)
        self.last_reward = -0.1
        return self.observe(reward=-0.1)

    def _tick_cooldowns(self) -> None:
        for player_id, cooldown in list(self.kill_cooldowns.items()):
            self.kill_cooldowns[player_id] = max(0, cooldown - 1)
