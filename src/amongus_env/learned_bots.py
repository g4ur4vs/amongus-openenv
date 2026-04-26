from __future__ import annotations

import argparse
import json
from typing import Any, Optional


class LearnedBotVotePolicy:
    """Tiny supervised bot policy trained to imitate deterministic expert votes."""

    def __init__(self, lookup: dict[str, str]) -> None:
        self.lookup = lookup

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "LearnedBotVotePolicy":
        return cls(lookup=dict(artifact["lookup"]))

    def bot_votes(self, engine: Any, human_target_id: str) -> dict[str, str]:
        key = feature_key_from_engine(engine, human_target_id)
        target_id = self.lookup.get(key, human_target_id)
        return {
            player.player_id: target_id
            for player in engine.players.values()
            if player.player_id != engine.controlled_player.player_id
            and player.alive
            and not player.ejected
        }


def train_learned_bot_vote_policy() -> dict[str, Any]:
    examples = _synthetic_examples()
    lookup = {example["feature_key"]: example["target_id"] for example in examples}
    return {
        "schema_version": 1,
        "policy_type": "memorized_vote_target",
        "trained_from": "scripted_engine_vote_rules_v1",
        "feature_spec": [
            "false_penalized_claim_speaker",
            "true_accusation_target",
            "human_target_id",
        ],
        "n_examples": len(examples),
        "lookup": lookup,
        "examples": examples,
    }


def feature_key_from_engine(engine: Any, human_target_id: str) -> str:
    false_speaker = engine._latest_false_penalized_claim_speaker() or "none"
    accusation_target = engine._latest_true_accusation_target() or "none"
    return feature_key(false_speaker, accusation_target, human_target_id)


def feature_key(
    false_speaker_id: Optional[str],
    true_accusation_target_id: Optional[str],
    human_target_id: str,
) -> str:
    return "|".join(
        [
            false_speaker_id or "none",
            true_accusation_target_id or "none",
            human_target_id,
        ]
    )


def _synthetic_examples() -> list[dict[str, str]]:
    player_ids = ["red", "blue", "green", "yellow"]
    examples: list[dict[str, str]] = []
    for human_target_id in player_ids:
        examples.append(
            {
                "feature_key": feature_key(None, None, human_target_id),
                "target_id": human_target_id,
                "label_source": "mirror_human_vote",
            }
        )
        for false_speaker_id in player_ids:
            examples.append(
                {
                    "feature_key": feature_key(
                        false_speaker_id, None, human_target_id
                    ),
                    "target_id": false_speaker_id,
                    "label_source": "false_penalized_claim",
                }
            )
        for accusation_target_id in player_ids:
            examples.append(
                {
                    "feature_key": feature_key(
                        None, accusation_target_id, human_target_id
                    ),
                    "target_id": accusation_target_id,
                    "label_source": "true_accusation",
                }
            )
    return examples


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train lightweight learned bot vote policy")
    parser.add_argument("--output", default=None, help="Optional path to write JSON artifact.")
    args = parser.parse_args(argv)
    artifact = train_learned_bot_vote_policy()
    output = json.dumps(artifact, indent=2)
    if args.output:
        with open(args.output, "w") as file:
            file.write(output)
    print(output)


if __name__ == "__main__":
    main()
