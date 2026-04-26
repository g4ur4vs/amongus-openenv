from amongus_env.engine import AmongUsEngine
from amongus_env.models import (
    CallMeeting,
    ClaimKind,
    CompleteTask,
    FakeTask,
    Kill,
    Move,
    PassMeeting,
    Phase,
    PlayerRole,
    ReportBody,
    Speak,
    Vent,
    Vote,
    Winner,
)


def test_reset_is_deterministic_and_spawns_in_cafeteria() -> None:
    first = AmongUsEngine(seed=7, impostor_ids=["blue"]).reset()
    second = AmongUsEngine(seed=7, impostor_ids=["blue"]).reset()

    assert first == second
    assert first.role is PlayerRole.CREWMATE
    assert first.location == "Cafeteria"
    assert first.phase is Phase.TASKS
    assert "Match reset" in first.message_log[-1]


def test_random_impostor_assignment_is_seed_reproducible() -> None:
    first = AmongUsEngine(seed=7)
    second = AmongUsEngine(seed=7)

    first.reset()
    second.reset()

    assert first.impostor_ids == second.impostor_ids


def test_random_impostor_assignment_varies_across_seeds() -> None:
    assignments = set()
    for seed in range(8):
        engine = AmongUsEngine(seed=seed)
        engine.reset()
        assignments.add(tuple(sorted(engine.impostor_ids)))

    assert len(assignments) > 1


def test_explicit_impostor_ids_override_seeded_random_assignment() -> None:
    first = AmongUsEngine(seed=1, impostor_ids=["blue"])
    second = AmongUsEngine(seed=99, impostor_ids=["blue"])

    first.reset()
    second.reset()

    assert first.impostor_ids == {"blue"}
    assert second.impostor_ids == {"blue"}


def test_invalid_movement_is_penalized_without_changing_location() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()

    observation = engine.step(Move(room="Navigation"))

    assert observation.location == "Cafeteria"
    assert observation.reward == -0.1
    assert "Invalid move" in observation.message_log[-1]


def test_location_history_tracks_reset_and_valid_moves_only() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()

    engine.step(Move(room="Electrical"))
    engine.step(Move(room="Navigation"))

    assert engine.location_history["red"] == ["Cafeteria", "Electrical"]
    assert engine.location_history["blue"] == ["Cafeteria"]


def test_visibility_only_includes_living_players_in_same_room() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(Move(room="Electrical"))

    engine.players["blue"].location = "Electrical"
    engine.players["green"].location = "MedBay"
    engine.players["yellow"].location = "Electrical"
    engine.players["yellow"].alive = False

    observation = engine.observe()

    assert observation.visible_players == ["blue"]


def test_crewmate_completes_current_room_task_for_dense_reward() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(Move(room="Electrical"))

    observation = engine.step(CompleteTask())

    assert observation.reward == 0.2
    assert observation.task_list[0].completed is True
    assert "Completed task" in observation.message_log[-1]


def test_impostor_fake_task_in_valid_room_has_no_task_reward() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"])
    engine.reset()
    engine.step(Move(room="Electrical"))

    observation = engine.step(FakeTask())

    assert observation.reward == 0.0
    assert observation.task_list == []
    assert "Faked task Fix Wiring" in observation.message_log[-1]


def test_crewmate_fake_task_is_illegal() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(Move(room="Electrical"))

    observation = engine.step(FakeTask())

    assert observation.reward == -0.1
    assert "Crewmates cannot fake tasks" in observation.message_log[-1]


def test_impostor_vent_legal_hop_updates_location() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"])
    engine.reset()
    engine.step(Move(room="Electrical"))

    observation = engine.step(Vent(room="MedBay"))

    assert observation.reward == 0.0
    assert observation.location == "MedBay"
    assert engine.location_history["red"][-1] == "MedBay"
    assert "Vented to MedBay" in observation.message_log[-1]


def test_crewmate_cannot_vent() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(Move(room="Electrical"))

    observation = engine.step(Vent(room="MedBay"))

    assert observation.reward == -0.1
    assert observation.location == "Electrical"
    assert "Crewmates cannot vent" in observation.message_log[-1]


def test_impostor_kill_marks_body_and_rewards_controlled_impostor() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"])
    engine.reset()
    engine.players["blue"].location = "Cafeteria"

    observation = engine.step(Kill(target_id="blue"))

    assert observation.reward == 0.5
    assert engine.players["blue"].alive is False
    assert "blue" in engine.dead_bodies
    assert "Killed blue" in observation.message_log[-1]


def test_report_body_enters_meeting_phase() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"])
    engine.reset()
    engine.step(Kill(target_id="blue"))

    observation = engine.step(ReportBody())

    assert observation.phase is Phase.MEETING
    assert observation.voting_open is False
    assert observation.meeting_turns_remaining == 1
    assert "Reported body" in observation.message_log[-1]


def test_meeting_prefills_sorted_non_controlled_speakers() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()

    observation = engine.step(CallMeeting())

    assert observation.voting_open is False
    assert observation.meeting_turns_remaining == 1
    assert observation.discussion_log == [
        "blue: pass",
        "green: pass",
        "yellow: pass",
    ]


def test_vote_before_discussion_turn_is_illegal() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())

    observation = engine.step(Vote(target_id="blue"))

    assert observation.reward == -0.1
    assert observation.phase is Phase.MEETING
    assert observation.voting_open is False
    assert observation.meeting_turns_remaining == 1
    assert "Cannot vote before discussion" in observation.message_log[-1]


def test_pass_opens_voting_without_claim_reward() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())

    observation = engine.step(PassMeeting())

    assert observation.reward == 0.0
    assert observation.phase is Phase.MEETING
    assert observation.voting_open is True
    assert observation.meeting_turns_remaining == 0
    assert observation.discussion_log[-1] == "red: pass"


def test_meeting_speech_parses_self_location_claim() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(Move(room="Electrical"))
    engine.step(CallMeeting())

    observation = engine.step(Speak(message="I was in Electrical"))

    assert observation.discussion_log[-1] == "red: I was in Electrical"
    assert observation.claims[-1].kind is ClaimKind.SELF_LOCATION
    assert observation.claims[-1].speaker_id == "red"
    assert observation.claims[-1].room == "Electrical"
    assert observation.claims[-1].truth_value is True
    assert observation.voting_open is True
    assert observation.meeting_turns_remaining == 0


def test_meeting_speech_parses_saw_player_claim() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.players["blue"].location = "MedBay"
    engine.location_history["blue"].append("MedBay")
    engine.step(CallMeeting())

    observation = engine.step(Speak(message="I saw blue in MedBay"))

    assert observation.claims[-1].kind is ClaimKind.SAW_PLAYER
    assert observation.claims[-1].target_id == "blue"
    assert observation.claims[-1].room == "MedBay"
    assert observation.claims[-1].truth_value is True


def test_meeting_speech_parses_true_vent_claim() -> None:
    engine = AmongUsEngine(
        seed=1,
        controlled_player_id="green",
        impostor_ids=["red"],
        player_ids=["red", "blue", "green", "yellow"],
    )
    engine.reset()
    engine.players["red"].location = "Electrical"
    engine.location_history["red"].append("Electrical")
    engine.step(CallMeeting())
    engine.vent_since_meeting["red"] = True

    observation = engine.step(Speak(message="I saw red vent"))

    assert observation.claims[-1].kind is ClaimKind.SAW_VENT
    assert observation.claims[-1].target_id == "red"
    assert observation.claims[-1].truth_value is True


def test_false_vent_claim_gets_hallucination_penalty() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())

    observation = engine.step(Speak(message="I saw blue vent"))

    assert observation.reward == -1.0
    assert observation.claims[-1].kind is ClaimKind.SAW_VENT
    assert observation.claims[-1].truth_value is False


def test_accusation_claim_guides_bot_votes_to_impostor() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())
    speak = engine.step(Speak(message="I accuse blue"))

    observation = engine.step(Vote(target_id="green"))

    assert speak.claims[-1].kind is ClaimKind.ACCUSE_IMPOSTOR
    assert speak.claims[-1].truth_value is True
    assert engine.players["blue"].ejected is True
    assert observation.winner is Winner.CREWMATES


def test_speaking_outside_meeting_is_illegal() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()

    observation = engine.step(Speak(message="I was in Cafeteria"))

    assert observation.reward == -0.1
    assert "Cannot speak outside meeting" in observation.message_log[-1]


def test_pass_outside_meeting_is_illegal() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()

    observation = engine.step(PassMeeting())

    assert observation.reward == -0.1
    assert observation.phase is Phase.TASKS
    assert observation.voting_open is False
    assert observation.meeting_turns_remaining == 0
    assert "Cannot pass outside meeting" in observation.message_log[-1]


def test_speak_after_voting_opens_is_illegal() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())
    engine.step(PassMeeting())

    observation = engine.step(Speak(message="I was in Cafeteria"))

    assert observation.reward == -0.1
    assert observation.voting_open is True
    assert "Voting is already open" in observation.message_log[-1]


def test_false_self_location_claim_gets_hallucination_penalty() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())

    observation = engine.step(Speak(message="I was in Electrical"))

    assert observation.reward == -1.0
    assert observation.claims[-1].truth_value is False


def test_bot_votes_mirror_controlled_vote_without_false_alibi() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())
    engine.step(PassMeeting())

    observation = engine.step(Vote(target_id="blue"))

    assert observation.reward == 1.5
    assert engine.players["blue"].ejected is True
    assert observation.done is True
    assert observation.winner is Winner.CREWMATES
    assert "Ejected blue" in observation.message_log[-2]


def test_bot_votes_eject_speaker_caught_in_false_alibi() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())
    engine.step(Speak(message="I was in Electrical"))

    observation = engine.step(Vote(target_id="blue"))

    assert engine.players["red"].ejected is True
    assert observation.reward == -0.5
    assert "Ejected red" in observation.message_log[-1]


def test_bot_votes_preserve_false_alibi_priority_over_controlled_vote() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    engine.reset()
    engine.step(CallMeeting())
    engine.step(Speak(message="I was in Electrical"))

    observation = engine.step(Vote(target_id="blue"))

    assert engine.players["red"].ejected is True
    assert engine.players["blue"].ejected is False
    assert observation.reward == -0.5
    assert "Ejected red" in observation.message_log[-1]


def test_majority_voting_tie_ejects_nobody_without_bot_ballots() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"], player_ids=["red", "blue", "green"])
    engine.reset()
    engine.step(CallMeeting())
    engine.step(PassMeeting())

    engine._bot_votes = lambda human_target_id: {}

    observation = engine.step(Vote(target_id="blue"))

    assert engine.players["blue"].ejected is False
    assert observation.phase is Phase.TASKS
    assert "No majority" in observation.message_log[-1]


def test_impostors_win_when_they_reach_parity() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["red"], player_ids=["red", "blue", "green"])
    engine.reset()

    observation = engine.step(Kill(target_id="blue"))

    assert observation.done is True
    assert observation.winner is Winner.IMPOSTORS
    assert observation.reward == 1.5
