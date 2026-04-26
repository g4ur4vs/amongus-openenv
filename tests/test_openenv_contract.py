import pytest

from amongus_env.models import Move, Observation, Phase
from amongus_env.openenv_server import AmongUsEnvironment, create_http_app


def test_openenv_environment_reset_and_step_return_observations() -> None:
    environment = AmongUsEnvironment(seed=1, impostor_ids=["blue"])

    reset_observation = environment.reset()
    step_observation = environment.step(Move(room="Electrical"))

    assert isinstance(reset_observation, Observation)
    assert isinstance(step_observation, Observation)
    assert step_observation.location == "Electrical"
    assert step_observation.phase is Phase.TASKS


def test_openenv_environment_accepts_action_payloads() -> None:
    environment = AmongUsEnvironment(seed=1, impostor_ids=["blue"])
    environment.reset()

    observation = environment.step({"type": "move", "room": "Electrical"})

    assert observation.location == "Electrical"


def test_http_app_factory_reports_missing_http_dependency_or_returns_app() -> None:
    try:
        app = create_http_app()
    except RuntimeError as exc:
        assert "fastapi" in str(exc).lower() or "openenv" in str(exc).lower()
    else:
        assert app is not None
