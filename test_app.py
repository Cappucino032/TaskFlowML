from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict():
    task_data = {
        "category": "Work",
        "priority": 1,
        "estimated_duration": "1-2 hours",
        "user_productivity_score": 0.8
    }
    response = client.post("/predict", json=task_data)
    assert response.status_code == 200
    
    prediction = response.json()
    assert "optimal_time" in prediction
    assert "confidence" in prediction
    assert "reasoning" in prediction
    
    assert isinstance(prediction["confidence"], float)
    assert 0 <= prediction["confidence"] <= 1