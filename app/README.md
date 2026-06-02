# app/

Streamlit frontend that lets users interactively query the prediction API.

## Files

| File | Description |
|---|---|
| `streamlit_app.py` | Streamlit app — dropdowns for neighborhood, type, price; calls `/predict` |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image (Python 3.12-slim, port 8080) |

## Run locally

Start the API first (see `api/README.md`), then:

```bash
uv pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The app reads `API_URL` from the environment (defaults to `http://localhost:8080`).

## Build & deploy

```bash
podman build --platform linux/amd64 -f app/Dockerfile \
  -t us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/app:latest .

podman push us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/app:latest

gcloud run deploy la-food-scout-app \
  --image us-central1-docker.pkg.dev/mtcars-fastapi-zach/la-food-scout/app:latest \
  --platform managed --region us-central1 --port 8080 --allow-unauthenticated \
  --set-env-vars API_URL=https://la-food-scout-api-886054929408.us-central1.run.app
```
