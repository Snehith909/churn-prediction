"""FastAPI serving layer for the churn model."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup, not per-request
    logger.info("Loading model at startup...")
    load_model()
    logger.info("Model loaded successfully")
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


class CustomerRecord(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: str
    Gender: str
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=15)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 619, "Geography": "France", "Gender": "Female",
                "Age": 42, "Tenure": 2, "Balance": 0, "NumOfProducts": 1,
                "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool


@app.get("/health")
def health_check():
    """Basic liveness/readiness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord):
    try:
        record_dict = record.model_dump()
        # add_engineered_features expects raw schema; identifier cols aren't required for inference
        result = predict_single(record_dict)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
