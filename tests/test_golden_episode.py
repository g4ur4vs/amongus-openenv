import json

from amongus_env.golden_episode import main, run_golden_episode


def test_golden_episode_exposes_verifiable_social_trace() -> None:
    trace = run_golden_episode()

    labels = [step["label"] for step in trace]
    assert labels == [
        "reset",
        "move_electrical",
        "complete_task",
        "call_meeting",
        "false_alibi",
        "vote_after_false_alibi",
    ]

    false_alibi = trace[4]["observation"]
    vote = trace[5]["observation"]

    assert false_alibi["reward"] == -1.0
    assert false_alibi["claims"][-1]["truth_value"] is False
    assert false_alibi["claims"][-1]["room"] == "MedBay"
    assert vote["reward"] == -0.5
    assert vote["message_log"][-1] == "Ejected red"


def test_golden_episode_cli_prints_valid_json(capsys) -> None:
    main()

    output = capsys.readouterr().out
    trace = json.loads(output)

    assert trace[0]["label"] == "reset"
    assert trace[-1]["label"] == "vote_after_false_alibi"
