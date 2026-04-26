from __future__ import annotations

import json

from .golden_episode import run_golden_reasoning_trace


def main() -> None:
    print(json.dumps(run_golden_reasoning_trace(), indent=2))


if __name__ == "__main__":
    main()
