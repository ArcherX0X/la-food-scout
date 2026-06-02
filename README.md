# LA Food Scout

A machine learning application that predicts whether an LA restaurant is likely to be **highly rated (≥4.5★)** based on its neighborhood, place type, price level, and review count.

Data is collected from the **Google Places API** across 20 LA neighborhoods. The model is served via a **FastAPI** backend deployed on **Google Cloud Run**, and the interactive frontend is a **Streamlit** app also deployed on Cloud Run.

## Project Documents

| Document | Link |
|---|---|
| Proposal Slides | [presentations/proposal/wang-zhixian-proposal.pdf](presentations/proposal/wang-zhixian-proposal.pdf) |
| Final Slides | [presentations/final/](presentations/final/) |
| Course Requirements | [docs/course-requirements.md](docs/course-requirements.md) |
| EDA Charts | [notebooks/eda_charts.png](notebooks/eda_charts.png) |

## Live Demo

| Service | URL |
|---|---|
| Streamlit App | https://la-food-scout-app-886054929408.us-central1.run.app |
| FastAPI Docs | https://la-food-scout-api-886054929408.us-central1.run.app/docs |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      User (Browser)                      │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTPS
                          ▼
┌──────────────────────────────────────────────────────────┐
│          Streamlit App  (Google Cloud Run)                │
│          app/streamlit_app.py                            │
└─────────────────────────┬────────────────────────────────┘
                          │ POST /predict
                          │ GET  /meta
                          ▼
┌──────────────────────────────────────────────────────────┐
│          FastAPI Model API  (Google Cloud Run)            │
│          api/main.py                                     │
│          /health  /ready  /meta  /predict                │
└─────────────────────────┬────────────────────────────────┘
                          │ loads
                          ▼
┌──────────────────────────────────────────────────────────┐
│    Random Forest Classifier  (models/model.pkl)          │
│    Trained on 2,443 LA restaurants from Google Places    │
│    Features: neighborhood, place type, price, reviews    │
└──────────────────────────────────────────────────────────┘

Data pipeline (offline):
  Google Places API → scripts/collect.py → data/la_restaurants.csv
                    → scripts/train_model.py → models/model.pkl
```

## Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier (balanced class weights) |
| Target | `high_rated`: restaurant rating ≥ 4.5★ |
| Features | neighborhood, primary_type, price_level, log(review_count) |
| CV ROC-AUC | 0.66 |
| Test ROC-AUC | 0.65 |
| High-rated Recall | 0.64 |
| Training data | 2,443 LA restaurants |

## Repo Structure

```
la-food-scout/
├── api/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── app/
│   ├── streamlit_app.py     # Streamlit frontend
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── data/
│   ├── la_restaurants.csv   # Collected dataset (2,443 restaurants)
│   └── README.md            # Schema documentation
├── models/
│   ├── model.pkl            # Trained model artifact
│   └── README.md
├── notebooks/
│   ├── eda.ipynb            # Full EDA + methodology notebook
│   ├── eda_charts.png       # EDA summary charts
│   └── README.md
├── presentations/
│   ├── proposal/            # Proposal slides (May 2026)
│   └── final/               # Final slides (June 2026)
├── scripts/
│   ├── collect.py           # Google Places API data collection
│   ├── train_model.py       # Model training
│   ├── eda.py               # EDA chart generation
│   └── README.md
├── tests/
│   ├── test_api.py          # 7 automated API tests
│   └── README.md
├── docs/
│   └── course-requirements.md
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

## Local Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Podman

```bash
git clone https://github.com/ArcherX0X/la-food-scout.git
cd la-food-scout

uv venv
source .venv/bin/activate
uv pip install -r api/requirements.txt
```

### (Re)train the model

```bash
python scripts/train_model.py
```

### Run the API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Docs at [http://localhost:8080/docs](http://localhost:8080/docs).

### Run the Streamlit app locally

In a second terminal:

```bash
uv pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

### Run tests

```bash
uv pip install pytest httpx
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (503 if model missing) |
| GET | `/meta` | Lists valid neighborhoods and place types |
| POST | `/predict` | Returns high-rated probability |

### Example

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "neighborhood": "Koreatown",
    "primary_type": "korean_restaurant",
    "price_level": 2,
    "review_count": 500
  }'
```

Response:

```json
{
  "high_rated_probability": 0.47,
  "predicted_high_rated": false,
  "inputs": {
    "neighborhood": "Koreatown",
    "primary_type": "korean_restaurant",
    "price_level": 2,
    "review_count": 500
  }
}
```

## Container Build & Deploy

### API

```bash
# build
podman build --platform linux/amd64 -f api/Dockerfile \
  -t us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/api:latest .

# push
podman push us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/api:latest

# deploy
gcloud run deploy la-food-scout-api \
  --image us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/api:latest \
  --platform managed --region us-central1 --port 8080 --allow-unauthenticated
```

### Streamlit App

```bash
podman build --platform linux/amd64 -f app/Dockerfile \
  -t us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/app:latest .

podman push us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/app:latest

gcloud run deploy la-food-scout-app \
  --image us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/app:latest \
  --platform managed --region us-central1 --port 8080 --allow-unauthenticated \
  --set-env-vars API_URL=https://la-food-scout-api-886054929408.us-central1.run.app
```

## AI Assistant Usage

This project was developed with **Claude Code** (claude-sonnet-4-6) as the primary AI assistant.

### Tasks where AI assistance was most valuable

- **Scaffolding**: Generated the full FastAPI + Streamlit project structure, Dockerfiles, and test suite from a single description of requirements
- **Pipeline design**: Suggested the `ColumnTransformer` + `Pipeline` pattern for clean preprocessing that works inside the container without separate preprocessor artifacts
- **Debugging**: Identified a `TypeError` on `sorted()` with mixed NaN/string values in `primary_type` and fixed it immediately
- **Class imbalance**: Proposed switching from GradientBoosting to `RandomForest(class_weight="balanced")` to improve high-rated recall from 0.30 to 0.64
- **FastAPI lifespan**: Replaced the deprecated `@app.on_event("startup")` pattern with the modern `asynccontextmanager` lifespan handler

### Areas requiring human judgment

- Choosing the prediction target (binary high-rated classification vs. rating regression)
- Selecting neighborhoods and place types to include in data collection
- Deciding which features were meaningful vs. leaky

### Lessons learned

- AI assistants are excellent at generating boilerplate and catching deprecation warnings, but domain knowledge about the data (what Google Places `primary_type` values look like) still needs to come from the human
- Always verify that pinned package versions in `requirements.txt` are actually compatible — AI-generated versions can be slightly stale
