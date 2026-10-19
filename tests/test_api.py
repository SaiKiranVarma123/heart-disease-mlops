import json
import numpy as np
from fastapi.testclient import TestClient
import heart_disease.api as api
class DummyModel:
    def predict_proba(self, frame): return np.array([[0.2,0.8]])
PAYLOAD={"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}
def test_predict(monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", type("P",(),{"exists":lambda self:True})())
    monkeypatch.setattr(api.joblib,"load",lambda _:DummyModel())
    monkeypatch.setattr(api, "METADATA_PATH", type("P",(),{"read_text":lambda self:json.dumps({"selected_model":"test","threshold":.5})})())
    with TestClient(api.app) as client:
        response=client.post("/predict",json=PAYLOAD)
        assert response.status_code==200 and response.json()["prediction"]==1
