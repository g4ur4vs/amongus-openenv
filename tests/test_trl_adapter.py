from amongus_env.trl_adapter import AmongUsToolEnv, reward_from_game_state


def test_trl_tool_environment_exposes_public_tool_methods() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])

    initial_message = tool_env.reset()
    move_result = tool_env.move("Electrical")
    task_result = tool_env.complete_task()

    assert "Cafeteria" in initial_message
    assert "Electrical" in move_result
    assert "reward=0.2" in task_result
    assert tool_env.reward == 0.2


def test_reward_from_game_state_reads_environment_rewards() -> None:
    first = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    second = AmongUsToolEnv(seed=1, impostor_ids=["red"])
    first.reset()
    second.reset()
    first.move("Electrical")
    second.kill("blue")

    assert reward_from_game_state([first, second]) == [0.0, 0.5]


def test_reward_from_game_state_rejects_unknown_aggregation() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    tool_env.reset()

    try:
        reward_from_game_state([tool_env], aggregation="typo")
    except ValueError as exc:
        assert "Unknown reward aggregation" in str(exc)
    else:
        raise AssertionError("unknown aggregation should fail closed")


def test_episode_return_accumulates_and_can_be_used_as_reward_aggregation() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    tool_env.reset()

    tool_env.move("Electrical")
    tool_env.complete_task()

    assert tool_env.episode_return == 0.2
    assert reward_from_game_state([tool_env], aggregation="episode_return") == [0.2]
    assert reward_from_game_state([tool_env], aggregation="last_step") == [0.2]


def test_episode_return_resets_on_reset() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    tool_env.reset()
    tool_env.move("Electrical")
    tool_env.complete_task()

    tool_env.reset()

    assert tool_env.episode_return == 0.0


def test_rollout_trace_records_reset_and_tool_steps() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])

    tool_env.reset()
    tool_env.move("Electrical")

    trace = tool_env.get_rollout_trace()
    assert [step["label"] for step in trace] == ["reset", "move"]
    assert trace[-1]["observation"]["location"] == "Electrical"


def test_trl_tool_environment_exposes_fake_task_and_vent_methods() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["red"])
    tool_env.reset()
    tool_env.move("Electrical")

    fake_result = tool_env.fake_task()
    vent_result = tool_env.vent("MedBay")

    assert "reward=0.0" in fake_result
    assert "Faked task Fix Wiring" in fake_result
    assert "MedBay" in vent_result


def test_trl_claim_summary_distinguishes_vent_claims() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    tool_env.reset()
    tool_env.call_meeting()

    result = tool_env.speak("I saw blue vent")

    assert "claims=saw_vent(red saw blue vent)=false" in result


def test_trl_tool_environment_exposes_speak_method() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    tool_env.reset()
    tool_env.call_meeting()

    result = tool_env.speak("I was in Electrical")

    assert "reward=-1.0" in result
    assert "message_log=Match reset | Emergency meeting called | red: I was in Electrical" in result
    assert "discussion_log=red: I was in Electrical" in result
    assert "claims=self_location(red in Electrical)=false" in result
    assert tool_env.reward == -1.0


def test_trl_tool_environment_exposes_pass_meeting_method() -> None:
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    tool_env.reset()
    tool_env.call_meeting()

    result = tool_env.pass_meeting()

    assert "voting_open=True" in result
    assert "meeting_turns_remaining=0" in result
    assert tool_env.reward == 0.0
