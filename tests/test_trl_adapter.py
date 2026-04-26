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
