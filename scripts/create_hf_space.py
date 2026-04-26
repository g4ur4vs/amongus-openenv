from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from huggingface_hub import HfApi


SPACE_NAME = "amongus-openenv"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in the environment before running this script.")

    api = HfApi(token=token)
    repo_id = resolve_repo_id(api=api)

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


def resolve_repo_id(
    environ: Mapping[str, str] | None = None,
    api: HfApi | None = None,
) -> str:
    environ = os.environ if environ is None else environ

    space_id = environ.get("HF_SPACE_ID")
    if space_id:
        if "/" not in space_id:
            raise SystemExit("HF_SPACE_ID must be in the form username_or_org/space_name.")
        return space_id

    username = environ.get("HF_USERNAME")
    space_name = environ.get("HF_SPACE_NAME", SPACE_NAME)
    if username:
        return f"{username}/{space_name}"

    if api is not None:
        try:
            return f"{api.whoami(cache=True)['name']}/{space_name}"
        except Exception as exc:
            raise SystemExit(
                "Could not infer your Hugging Face username without hitting a "
                "rate-limited endpoint. Set HF_SPACE_ID=username_or_org/space_name "
                "or HF_USERNAME=username_or_org, then rerun."
            ) from exc

    raise SystemExit(
        "Set HF_SPACE_ID=username_or_org/space_name or HF_USERNAME=username_or_org."
    )


if __name__ == "__main__":
    main()
