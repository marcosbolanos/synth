from fastapi.testclient import TestClient

from webui.app import app


def test_report_routes_have_empty_or_completed_pages() -> None:
    client = TestClient(app)

    assert client.get("/vital-investigation").status_code == 200
    assert client.get("/vital-transformer").status_code == 200
    assert client.get("/vital-augmentation").status_code == 200
    assert client.get("/vital-stream-scale").status_code == 200
    assert client.get("/vital-refiner").status_code == 200


def test_generated_file_rejects_traversal() -> None:
    client = TestClient(app)

    assert client.get("/files/../../pyproject.toml").status_code == 404
