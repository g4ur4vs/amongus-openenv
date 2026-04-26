import json

from amongus_env.deception_elo import (
    compute_deception_elo,
    main,
    parse_ejected_player,
    update_elo,
)
from amongus_env.engine import AmongUsEngine
from amongus_env.golden_episode import run_golden_episode
from amongus_env.models import CompleteTask, Move
from amongus_env.trace import record_step


def test_elo_update_closed_form_1500_vs_1500_when_deception_is_caught() -> None:
    result = update_elo(deceiver_rating=1500.0, assembly_rating=1500.0, deceiver_score=0.0)

    assert result["expected"]["deceiver"] == 0.5
    assert result["after"] == {
        "deceiver": 1484.0,
        "assembly": 1516.0,
    }
    assert result["delta"] == {
        "deceiver": -16.0,
        "assembly": 16.0,
    }


def test_parse_ejected_player_from_message_log() -> None:
    assert parse_ejected_player(["Match reset", "Ejected red"]) == "red"
    assert parse_ejected_player(["No majority; nobody ejected"]) is None


def test_golden_episode_deception_elo_rewards_detection() -> None:
    result = compute_deception_elo(run_golden_episode())

    assert result["schema_version"] == 1
    assert result["applied"] is True
    assert result["outcome"] == {
        "event": "deceiver_caught",
        "false_speaker_id": "red",
        "ejected_id": "red",
        "deceiver_score": 0.0,
    }
    assert result["after"]["deceiver"] < result["before"]["deceiver"]
    assert result["after"]["assembly"] > result["before"]["assembly"]


def test_task_only_trace_does_not_update_deception_elo() -> None:
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"])
    trace = [
        record_step("reset", "reset", engine.reset()),
        record_step("move_electrical", "move to Electrical", engine.step(Move(room="Electrical"))),
        record_step("complete_task", "complete task", engine.step(CompleteTask())),
    ]

    result = compute_deception_elo(trace)

    assert result["applied"] is False
    assert result["before"] == result["after"]
    assert result["delta"] == {"deceiver": 0.0, "assembly": 0.0}


def test_deception_elo_cli_prints_valid_json(capsys) -> None:
    main()

    result = json.loads(capsys.readouterr().out)

    assert result["applied"] is True
    assert result["outcome"]["event"] == "deceiver_caught"
