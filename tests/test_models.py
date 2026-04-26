from amongus_env.models import (
    ActionAdapter,
    Claim,
    ClaimKind,
    FakeTask,
    Kill,
    Move,
    Observation,
    PassMeeting,
    Phase,
    PlayerRole,
    Speak,
    TaskState,
    Vent,
)


def test_action_adapter_validates_discriminated_actions() -> None:
    move = ActionAdapter.validate_python({"type": "move", "room": "Electrical"})
    kill = ActionAdapter.validate_python({"type": "kill", "target_id": "blue"})
    speak = ActionAdapter.validate_python(
        {"type": "speak", "message": "I was in Electrical"}
    )
    pass_meeting = ActionAdapter.validate_python({"type": "pass"})
    fake_task = ActionAdapter.validate_python({"type": "fake_task"})
    vent = ActionAdapter.validate_python({"type": "vent", "room": "MedBay"})

    assert move == Move(room="Electrical")
    assert kill == Kill(target_id="blue")
    assert speak == Speak(message="I was in Electrical")
    assert pass_meeting == PassMeeting()
    assert fake_task == FakeTask()
    assert vent == Vent(room="MedBay")


def test_observation_contains_training_surface() -> None:
    observation = Observation(
        role=PlayerRole.CREWMATE,
        location="Cafeteria",
        visible_players=["blue"],
        task_list=[TaskState(room="Electrical", name="Fix Wiring")],
        message_log=["Match reset"],
        discussion_log=["red: I was in Cafeteria"],
        claims=[
            Claim(
                kind=ClaimKind.SELF_LOCATION,
                speaker_id="red",
                room="Cafeteria",
                truth_value=True,
            )
        ],
        phase=Phase.TASKS,
        reward=0.0,
        done=False,
        winner=None,
        voting_open=False,
        meeting_turns_remaining=0,
    )

    assert observation.role is PlayerRole.CREWMATE
    assert observation.location == "Cafeteria"
    assert observation.visible_players == ["blue"]
    assert observation.task_list[0].completed is False
    assert observation.discussion_log == ["red: I was in Cafeteria"]
    assert observation.claims[0].truth_value is True
    assert observation.voting_open is False
    assert observation.meeting_turns_remaining == 0
