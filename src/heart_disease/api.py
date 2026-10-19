import json, logging, time
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from .config import METADATA_PATH, MODEL_PATH
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
REQUESTS = Counter("api_requests_total", "API requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_duration_seconds", "API latency", ["path"])
STATE = {}
class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120); sex: int = Field(ge=0, le=1); cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300); chol: float = Field(gt=0, le=800); fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2); thalach: float = Field(gt=0, le=250); exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=10); slope: int = Field(ge=1, le=3); ca: float = Field(ge=0, le=4); thal: float = Field(ge=3, le=7)
@asynccontextmanager
async def lifespan(app):
    if not MODEL_PATH.exists(): raise RuntimeError("Model artifact missing. Run training first.")
    STATE["model"] = joblib.load(MODEL_PATH); STATE["metadata"] = json.loads(METADATA_PATH.read_text()); yield; STATE.clear()
app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
@app.middleware("http")
async def observe(request: Request, call_next):
    start=time.perf_counter(); status=500
    try:
        response=await call_next(request); status=response.status_code; return response
    finally:
        elapsed=time.perf_counter()-start; REQUESTS.labels(request.method,request.url.path,str(status)).inc(); LATENCY.labels(request.url.path).observe(elapsed); logger.info("method=%s path=%s status=%s duration_ms=%.2f",request.method,request.url.path,status,elapsed*1000)
@app.get("/health")
def health(): return {"status":"ok", "model":STATE["metadata"]["selected_model"]}
@app.post("/predict")
def predict(patient: Patient):
    try:
        probability=float(STATE["model"].predict_proba(pd.DataFrame([patient.model_dump()]))[0,1]); threshold=float(STATE["metadata"].get("threshold",.5))
        return {"prediction":int(probability>=threshold), "confidence":round(max(probability,1-probability),6), "heart_disease_probability":round(probability,6), "model":STATE["metadata"]["selected_model"], "disclaimer":"Educational model; not a medical diagnosis."}
    except Exception as exc:
        logger.exception("Prediction failed"); raise HTTPException(status_code=500, detail="Prediction failed") from exc
@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
