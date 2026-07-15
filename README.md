# Churn Prediction — Full Stack (Backend + Frontend)

Two independently deployable services in one repo:

```
churn-fullstack/
├── churn-project/      ← FastAPI backend (model training + serving)
│   └── README.md        (full details on the ML pipeline)
└── churn-dashboard/    ← Streamlit frontend (calls the backend API)
```

## Deploying both on Render (as 2 separate services, same repo)

### 1. Backend — `churn-project/`
- New Web Service → connect this repo
- **Root Directory:** `churn-project`
- **Runtime:** Docker (uses the included `Dockerfile`)
- Deploy → copy the resulting URL (e.g. `https://churn-api.onrender.com`)

### 2. Frontend — `churn-dashboard/`
- New Web Service → connect this repo again
- **Root Directory:** `churn-dashboard`
- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Environment variable:** `API_URL` = the backend URL from step 1
- Deploy

Both services build and redeploy independently — a UI tweak never triggers a model-service rebuild, and vice versa.

## Local development

```bash
# Terminal 1 — backend
cd churn-project
pip install -r requirements.txt
python -m src.train        # only needed once, to generate models/model.pkl
uvicorn app:app --reload

# Terminal 2 — frontend
cd churn-dashboard
pip install -r requirements.txt
export API_URL=http://localhost:8000   # Windows: set API_URL=http://localhost:8000
streamlit run app.py
```

See `churn-project/README.md` for model details, metrics, and design decisions.
