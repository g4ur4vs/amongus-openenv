import json

from amongus_env.policy_eval import (
    build_policy_eval_report,
    generate_checkpoint_completion,
    main,
)


def test_policy_eval_report_compares_scripted_baseline_and_completion_policy() -> None:
    report = build_policy_eval_report(
        rl_completions=[
            '{"type": "move", "room": "Electrical"}\n{"type": "complete_task"}',
        ],
        num_episodes=1,
    )

    assert report["schema_version"] == 1
    assert report["baseline"]["average_episode_return"] == 0.2
    assert report["rl"]["average_episode_return"] == 0.2
    assert report["comparison"]["delta"] == 0.0


def test_policy_eval_cli_prints_valid_json(tmp_path, capsys) -> None:
    completions = tmp_path / "completions.txt"
    completions.write_text('{"type": "move", "room": "Electrical"}\n{"type": "complete_task"}')

    main(["--rl-completions-file", str(completions), "--num-episodes", "1"])

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["rl"]["episodes"][0]["episode_return"] == 0.2


def test_checkpoint_completion_uses_injected_generator() -> None:
    completion = generate_checkpoint_completion(
        "unused-checkpoint",
        prompt="Emit action JSON",
        generator=lambda checkpoint, prompt, max_new_tokens: (
            '{"type": "move", "room": "Electrical"}'
        ),
    )

    assert completion == '{"type": "move", "room": "Electrical"}'


def test_policy_eval_report_can_use_checkpoint_completion_generator() -> None:
    report = build_policy_eval_report(
        checkpoint="unused-checkpoint",
        completion_generator=lambda checkpoint, prompt, max_new_tokens: (
            '{"type": "move", "room": "Electrical"}\n{"type": "complete_task"}'
        ),
        num_episodes=1,
    )

    assert report["rl"]["policy"] == "checkpoint_completion"
    assert report["rl"]["checkpoint"] == "unused-checkpoint"
    assert report["rl"]["average_episode_return"] == 0.2
