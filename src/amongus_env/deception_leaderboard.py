from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from .deception_elo import INITIAL_RATING, compute_deception_elo, update_elo
from .golden_episode import run_golden_episode


def run_deception_leaderboard(runs: int = 1) -> dict[str, Any]:
    deceiver_rating = INITIAL_RATING
    assembly_rating = INITIAL_RATING
    records = []
    for index in range(runs):
        trace = run_golden_episode()
        elo = compute_deception_elo(
            trace,
            initial_rating=deceiver_rating,
        )
        outcome = elo["outcome"]
        if outcome["deceiver_score"] is not None:
            update = update_elo(
                deceiver_rating=deceiver_rating,
                assembly_rating=assembly_rating,
                deceiver_score=outcome["deceiver_score"],
            )
            deceiver_rating = update["after"]["deceiver"]
            assembly_rating = update["after"]["assembly"]
            elo["before"] = {
                "deceiver": round(deceiver_rating - update["delta"]["deceiver"], 10),
                "assembly": round(assembly_rating - update["delta"]["assembly"], 10),
            }
            elo["after"] = update["after"]
            elo["delta"] = update["delta"]
        records.append(
            {
                "run_id": index + 1,
                "trace": "golden_false_alibi",
                "deception_elo": elo,
            }
        )

    leaderboard = sorted(
        [
            {"name": "deceiver", "rating": round(deceiver_rating, 10)},
            {"name": "assembly", "rating": round(assembly_rating, 10)},
        ],
        key=lambda item: item["rating"],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "summary": {
            "runs": runs,
            "top_side": leaderboard[0]["name"],
        },
        "leaderboard": leaderboard,
        "runs": records,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute Deception Elo leaderboard")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(run_deception_leaderboard(runs=args.runs), indent=2))


if __name__ == "__main__":
    main()
