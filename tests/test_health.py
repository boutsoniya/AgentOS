from fastapi.testclient import TestClient
from apps.backend.app.main import app


def test_health():
    response = TestClient(app).get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
