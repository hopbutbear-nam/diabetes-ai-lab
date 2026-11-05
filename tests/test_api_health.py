from fastapi.testclient import TestClient
from diabetes_ai.serving.app import app

def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
