import pytest

from amongus_env.openenv_server import create_http_app


pytestmark = pytest.mark.integration


def test_openenv_http_reset_and_step_smoke() -> None:
    testclient = pytest.importorskip("starlette.testclient")

    client = testclient.TestClient(create_http_app())

    health_response = client.get("/health")
    assert health_response.status_code == 200

    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200
    reset_observation = _extract_observation(reset_response.json())
    assert reset_observation["location"] == "Cafeteria"
    assert reset_observation["phase"] == "tasks"

    step_response = client.post(
        "/step",
        json={"action": {"type": "move", "room": "Electrical"}},
    )
    assert step_response.status_code == 200
    step_observation = _extract_observation(step_response.json())
    assert step_observation["location"] == "Electrical"
    assert step_observation["phase"] == "tasks"


def _extract_observation(payload: dict) -> dict:
    if "observation" in payload:
        return payload["observation"]
    if "result" in payload and isinstance(payload["result"], dict):
        result = payload["result"]
        if "observation" in result:
            return result["observation"]
    return payload
