# Customer Churn Prediction — Production ML Pipeline

Predicts whether a bank customer will churn, served via a FastAPI endpoint,
containerized with Docker, with CI running tests on every push.

## Problem
Bank churn is ~20% of customers (imbalanced). Business goal: **flag likely churners
early** so retention teams can act — so recall on the churn class matters as much
as overall accuracy.

## Results (RandomForest baseline)
| Metric | Score |
|---|---|
| ROC-AUC | 0.865 |
| Recall (churn class) | 0.72 |
| Precision (churn class) | 0.53 |
| Accuracy | 0.815 |

`class_weight="balanced"` is used to counter the 80/20 imbalance rather than
optimizing for raw accuracy, which would just predict "no churn" for everyone.

## Project Structure
```
churn-project/
├── data/
│   ├── raw/              # original CSV, untouched
│   └── processed/        # (reserved for cached transforms)
├── src/
│   ├── config.py              # centralized paths, columns, hyperparameters
│   ├── data_ingestion.py      # load + schema validation
│   ├── feature_engineering.py # derived features (ratios, flags, age buckets)
│   ├── preprocessing.py       # sklearn ColumnTransformer (scale + one-hot)
│   ├── train.py                # full training pipeline + metrics + metadata
│   └── predict.py              # shared inference logic (used by API)
├── models/
│   ├── model.pkl               # trained sklearn Pipeline (preprocessing + model)
│   └── model_metadata.json     # training date, hyperparams, metrics, feature list
├── tests/
│   ├── test_preprocessing.py   # unit tests for feature/preprocessing logic
│   └── test_api.py              # integration tests for API endpoints
├── app.py                       # FastAPI serving layer
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml     # test + docker build on every push
```

## Key design decisions
- **Split before fit**: preprocessing (scaler/encoder) is fit only on train data,
  inside a single sklearn `Pipeline`, to prevent data leakage.
- **Feature engineering choices**: `BalanceSalaryRatio`, `ProductsPerTenure`,
  `IsZeroBalance`, and `AgeGroup` were added based on domain intuition
  (e.g. many churned accounts sit at zero balance — dormant before leaving).
- **Model persisted as one Pipeline object** (preprocessing + classifier together),
  so `predict.py` never risks preprocessing train/serve skew.
- **Metadata logging**: every training run saves hyperparameters + metrics +
  timestamp to `model_metadata.json` — poor man's experiment tracking without
  needing MLflow infra for a portfolio project.

## Running locally
```bash
pip install -r requirements.txt

# Train
python -m src.train

# Serve
uvicorn app:app --reload

# Test
pytest tests/ -v
```

## Running with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
curl http://localhost:8000/health
```

## Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619, "Geography": "France", "Gender": "Female",
    "Age": 42, "Tenure": 2, "Balance": 0, "NumOfProducts": 1,
    "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88
  }'
```

## Next steps (if extending)
- Swap flat-file metadata logging for MLflow/W&B experiment tracking
- Add data drift monitoring on incoming prediction requests
- Add a `/batch-predict` endpoint for scoring CSVs
- Try XGBoost/LightGBM and compare against the RandomForest baseline
