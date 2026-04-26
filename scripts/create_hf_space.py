from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


SPACE_NAME = "amongus-openenv"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in the environment before running this script.")

    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(Path(__file__).resolve().parents[1]),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            ".pytest_cache/*",
            "__pycache__/*",
            "*.pyc",
        ],
    )
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
