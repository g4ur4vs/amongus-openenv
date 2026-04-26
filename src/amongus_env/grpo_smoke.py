from __future__ import annotations

import json
from typing import Any, Optional

from .trl_adapter import AmongUsToolEnv, reward_from_game_state


def run_grpo_smoke(require_trl: bool = False) -> dict[str, Any]:
    trl_symbols, trl_error = _load_trl_symbols()
    tool_env = AmongUsToolEnv(seed=1, impostor_ids=["blue"])
    reset_summary = tool_env.reset()
    step_summary = tool_env.move("Electrical")
    reward = reward_from_game_state([tool_env])[0]

    trl_available = trl_error is None
    ok = trl_available or not require_trl
    message = (
        "TRL GRPO symbols imported."
        if trl_available
        else 'Install the training extra with: pip install -e ".[training]"'
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "trl_available": trl_available,
        "trl_symbols": trl_symbols,
        "trl_error": trl_error,
        "message": message,
        "reset_summary": reset_summary,
        "step_summary": step_summary,
        "reward": reward,
    }


def _load_trl_symbols() -> tuple[list[str], Optional[str]]:
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # pragma: no cover - depends on optional training extra
        return [], f"{exc.__class__.__name__}: {exc}"
    return [GRPOConfig.__name__, GRPOTrainer.__name__], None


def main() -> None:
    result = run_grpo_smoke(require_trl=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
