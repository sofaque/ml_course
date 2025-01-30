import requests

BASE_URL = "http://localhost:5000"

def test_health_check():
    """API response health check"""
    response = requests.get(BASE_URL)
    assert response.status_code == 200
    assert response.text == "It works! For prediction navigate to /predict_form"

def test_prediction():
    """ТTest message test"""
    data = {"comment": "This is a toxic comment."} 
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    
    prediction = response.json()
    
    # check output contains neccessary fielddds
    assert "label" in prediction, "Response JSON missing 'label'"
    assert "score" in prediction, "Response JSON missing 'score'"
    
    # check score is a number
    assert isinstance(prediction["score"], (int, float)), "Score is not a number"