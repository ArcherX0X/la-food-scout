# api/

FastAPI application that serves predictions from the trained model.

## Files

| File | Description |
|---|---|
| `main.py` | FastAPI app with `/health`, `/ready`, `/meta`, `/predict` |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image (Python 3.12-slim, port 8080) |

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe — always 200 |
| GET | `/ready` | Readiness probe — 503 if model not loaded |
| GET | `/meta` | Returns valid neighborhood and place type values |
| POST | `/predict` | Returns `high_rated_probability` and `predicted_high_rated` |

Interactive docs available at `/docs` when running.

## Run locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Build & deploy

```bash
podman build --platform linux/amd64 -f api/Dockerfile \
  -t us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/api:latest .

podman push us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/api:latest

gcloud run deploy la-food-scout-api \
  --image us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/api:latest \
  --platform managed --region us-central1 --port 8080 --allow-unauthenticated
```
