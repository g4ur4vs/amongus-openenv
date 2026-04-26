import json

from amongus_env.engine import AmongUsEngine
from amongus_env.learned_bots import (
    LearnedBotVotePolicy,
    main,
    train_learned_bot_vote_policy,
)
from amongus_env.models import CallMeeting, Speak, Vote, Winner


def test_train_learned_bot_policy_emits_json_artifact() -> None:
    artifact = train_learned_bot_vote_policy()

    assert artifact["schema_version"] == 1
    assert artifact["policy_type"] == "memorized_vote_target"
    assert artifact["n_examples"] >= 3
    assert artifact["lookup"]


def test_learned_bot_policy_matches_expert_false_claim_vote() -> None:
    policy = LearnedBotVotePolicy.from_artifact(train_learned_bot_vote_policy())
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"], bot_vote_policy=policy)
    engine.reset()
    engine.step(CallMeeting())
    engine.step(Speak(message="I was in Electrical"))

    observation = engine.step(Vote(target_id="blue"))

    assert engine.players["red"].ejected is True
    assert observation.reward == -0.5


def test_learned_bot_policy_matches_expert_true_accusation_vote() -> None:
    policy = LearnedBotVotePolicy.from_artifact(train_learned_bot_vote_policy())
    engine = AmongUsEngine(seed=1, impostor_ids=["blue"], bot_vote_policy=policy)
    engine.reset()
    engine.step(CallMeeting())
    engine.step(Speak(message="I accuse blue"))

    observation = engine.step(Vote(target_id="green"))

    assert engine.players["blue"].ejected is True
    assert observation.winner is Winner.CREWMATES


def test_train_learned_bots_cli_prints_valid_json(capsys) -> None:
    main([])

    artifact = json.loads(capsys.readouterr().out)
    assert artifact["policy_type"] == "memorized_vote_target"
