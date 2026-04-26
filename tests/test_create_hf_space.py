import pytest

from scripts.create_hf_space import resolve_repo_id


def test_resolve_repo_id_prefers_full_space_id() -> None:
    repo_id = resolve_repo_id(
        environ={
            "HF_SPACE_ID": "team/amongus-demo",
            "HF_USERNAME": "ignored",
        }
    )

    assert repo_id == "team/amongus-demo"


def test_resolve_repo_id_uses_username_and_default_space_name() -> None:
    repo_id = resolve_repo_id(environ={"HF_USERNAME": "g4ur4vs"})

    assert repo_id == "g4ur4vs/amongus-openenv"


def test_resolve_repo_id_rejects_missing_namespace_without_whoami() -> None:
    with pytest.raises(SystemExit, match="HF_SPACE_ID"):
        resolve_repo_id(environ={})
