import json

from amongus_env.deception_leaderboard import main, run_deception_leaderboard


def test_deception_leaderboard_aggregates_multiple_runs() -> None:
    result = run_deception_leaderboard(runs=3)

    assert result["schema_version"] == 1
    assert result["summary"]["runs"] == 3
    assert result["leaderboard"][0]["name"] == "assembly"
    assert result["leaderboard"][0]["rating"] > result["leaderboard"][1]["rating"]


def test_deception_leaderboard_uses_chained_elo_updates() -> None:
    result = run_deception_leaderboard(runs=2)

    assert result["leaderboard"] == [
        {"name": "assembly", "rating": 1530.530498471},
        {"name": "deceiver", "rating": 1469.469501529},
    ]


def test_deception_leaderboard_cli_prints_valid_json(capsys) -> None:
    main([])

    result = json.loads(capsys.readouterr().out)

    assert result["summary"]["runs"] == 1
    assert result["leaderboard"]
