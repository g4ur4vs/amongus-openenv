from amongus_env.completion_rollout import (
    completion_episode_return_reward_func,
    parse_completion_actions,
    run_completion_rollout,
)
from amongus_env.models import CompleteTask, Move


def test_parse_completion_actions_reads_json_lines() -> None:
    actions = parse_completion_actions(
        """
        I will do the task route.
        {"type": "move", "room": "Electrical"}
        {"type": "complete_task"}
        """
    )

    assert actions == [Move(room="Electrical"), CompleteTask()]


def test_run_completion_rollout_returns_episode_return() -> None:
    result = run_completion_rollout(
        """
        {"type": "move", "room": "Electrical"}
        {"type": "complete_task"}
        """,
        seed=1,
        impostor_ids=["blue"],
    )

    assert result["episode_return"] == 0.2
    assert result["valid_actions"] == 2
    assert result["trace"][-1]["observation"]["reward"] == 0.2


def test_completion_episode_return_reward_func_scores_each_completion() -> None:
    rewards = completion_episode_return_reward_func(
        [
            '{"type": "move", "room": "Electrical"}\n{"type": "complete_task"}',
            "not an action",
        ],
        seed=1,
        impostor_ids=["blue"],
    )

    assert rewards == [0.2, 0.0]
