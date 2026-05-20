from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Car price prediction API"}

def test_predict():
    sample_features = [2010, 150000, 2000, 1, 1, 1, 1, 0, 0]
    response = client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
