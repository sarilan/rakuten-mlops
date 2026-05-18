from fastapi.testclient import TestClient
from rakuten_api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API Rakuten"}

def test_predict():
    response = client.post(
        "/predict",
        json={"text": "Jeux Nintendo Switch Mario Kart console"}
    )
    assert response.status_code == 200
    assert "categorie" in response.json()