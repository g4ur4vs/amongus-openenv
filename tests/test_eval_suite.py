import json

from amongus_env.eval_suite import evaluate_trace, main, run_eval_suite
from amongus_env.golden_episode import run_golden_episode


def test_eval_suite_scores_current_golden_trace() -> None:
    result = evaluate_trace(run_golden_episode())

    assert result["schema_version"] == 1
    assert result["ok"] is True
    assert result["episode"] == "golden_false_alibi"
    assert result["summary"]["steps"] == 6
    assert result["summary"]["total_reward"] == -1.3
    assert [check["id"] for check in result["checks"]] == [
        "labels_sequence",
        "task_reward",
        "meeting_protocol",
        "false_alibi_penalty",
        "bot_vote_ejects_false_claimant",
    ]
    assert all(check["ok"] for check in result["checks"])


def test_eval_suite_runs_multiple_baseline_scenarios() -> None:
    result = run_eval_suite()

    assert result["schema_version"] == 1
    assert result["ok"] is True
    assert result["summary"] == {
        "scenarios": 10,
        "passed": 10,
        "failed": 0,
    }
    assert [scenario["episode"] for scenario in result["scenarios"]] == [
        "golden_false_alibi",
        "invalid_move_no_state_change",
        "crewmate_task_route",
        "meeting_pass_bot_majority",
        "impostor_parity_win",
        "kill_cooldown_blocks_second_kill",
        "impostor_fake_task",
        "vent_claim_verification",
        "meeting_no_majority",
        "multi_impostor_parity",
    ]


def test_eval_suite_fails_mutated_trace() -> None:
    trace = run_golden_episode()
    trace[4]["observation"]["reward"] = 0.0

    result = evaluate_trace(trace)
    failed_checks = {
        check["id"] for check in result["checks"] if check["ok"] is False
    }

    assert result["ok"] is False
    assert "false_alibi_penalty" in failed_checks


def test_eval_suite_fails_each_named_check_when_trace_is_mutated() -> None:
    mutations = {
        "labels_sequence": lambda trace: trace[0].update({"label": "wrong"}),
        "task_reward": lambda trace: trace[2]["observation"].update({"reward": 0.0}),
        "meeting_protocol": lambda trace: trace[3]["observation"].update(
            {"voting_open": True}
        ),
        "bot_vote_ejects_false_claimant": lambda trace: trace[5]["observation"].update(
            {"message_log": ["No majority; nobody ejected"]}
        ),
    }

    for check_id, mutate in mutations.items():
        trace = run_golden_episode()
        mutate(trace)

        result = evaluate_trace(trace)
        failed_checks = {
            check["id"] for check in result["checks"] if check["ok"] is False
        }

        assert result["ok"] is False
        assert check_id in failed_checks


def test_eval_suite_malformed_task_trace_fails_without_raising() -> None:
    trace = run_golden_episode()
    trace[2]["observation"]["task_list"] = []

    result = evaluate_trace(trace)
    task_reward = next(
        check for check in result["checks"] if check["id"] == "task_reward"
    )

    assert result["ok"] is False
    assert task_reward["ok"] is False


def test_eval_suite_cli_prints_valid_json(capsys) -> None:
    main()

    result = json.loads(capsys.readouterr().out)

    assert result["ok"] is True
    assert result["scenarios"]
