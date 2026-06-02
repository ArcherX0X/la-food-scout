# LA Food Scout — Final Presentation Notes
# File to submit: wang-zhixian-final.pdf (or .pptx)
# Max: 7 slides (1 title + 6 content) | Duration: 10–12 min

---

## Slide 1 — Title

**LA Food Scout**
*Predicting Highly-Rated Restaurants in Los Angeles*

Zhixian Wang | STAT 418 | June 2, 2026

| | |
|---|---|
| Live App | https://la-food-scout-app-886054929408.us-central1.run.app |
| API Docs | https://la-food-scout-api-886054929408.us-central1.run.app/docs |
| GitHub | https://github.com/ArcherX0X/la-food-scout |

---

## Slide 2 — Project Overview & Motivation

**Problem**
LA has 30,000+ restaurants. Google ratings are tightly clustered — 89% of places are ≥4.0★ — so the real signal is ≥4.5★, which only 41% of restaurants reach.

**Question**
Can we predict whether a restaurant is highly rated (≥4.5★) *before* trying it, using only publicly available metadata?

**Why it matters**
- Tourists and locals want a quick filter before committing to a spot
- Restaurant owners can understand what factors correlate with high ratings
- Demonstrates an end-to-end ML pipeline on real geospatial data

**Approach**
1. Collect data via Google Places API across 20 LA neighborhoods
2. Train a binary classifier on neighborhood, cuisine type, price, and review volume
3. Serve predictions via a FastAPI backend + Streamlit frontend, both on Google Cloud Run

---

## Slide 3 — Data Collection & Preprocessing

### Collection

| Detail | Value |
|---|---|
| Source | Google Places API v1 — `searchNearby` endpoint |
| Neighborhoods | 20 across LA (Koreatown, Santa Monica, Downtown LA, Little Tokyo, …) |
| Search strategy | 5 points per neighborhood (center + N/S/E/W at ~1.3 km) to bypass the 20-result cap |
| Place types queried | `restaurant`, `cafe`, `bakery`, `bar` |
| De-duplication | By Google place ID |
| Final dataset | **2,443 unique restaurants** |

Fields collected: `name`, `rating`, `review_count`, `price_level`, `primary_type`, `lat/lon`, `business_status`, `neighborhood`

### Preprocessing

| Step | Detail |
|---|---|
| Drop missing ratings | 0 rows dropped (API only returns rated places) |
| Impute `price_level` | 637 missing (26%) → filled with median (= 2.0) |
| Drop missing `primary_type` | 3 rows dropped |
| Log-transform reviews | `review_count_log = log1p(review_count)` — raw range 1–28,054; log range 0–10.2 |
| Binary target | `high_rated = 1` if `rating ≥ 4.5` |

### Dataset at a Glance

| Stat | Value |
|---|---|
| Total restaurants | 2,443 |
| High-rated (≥4.5★) | 998 — **40.9%** |
| Neighborhoods | 20 |
| Unique place types | 144 |
| Median review count | 390 |
| Mean review count | 862 (right-skewed) |

### Key EDA Findings *(chart: `notebooks/eda_charts.png`)*

- **By neighborhood:** Little Tokyo highest (63% high-rated); Monterey Park lowest (24%)
- **By price level:** Budget (36.5%) vs. Expensive (49.5%) vs. Very Expensive (75%)
- **By place type:** Italian (54%), donut shops (52%), cocktail bars (51%) on top; convenience stores (0%), fast food (3%), grocery stores (7%) at bottom
- **Rating distribution:** 40% of restaurants fall in 4.25–4.5 range; only 2.6% are perfect 5.0★

---

## Slide 4 — Model Development & Evaluation

### Pipeline

```
ColumnTransformer
  ├── StandardScaler          → [price_level, review_count_log]
  └── OneHotEncoder           → [neighborhood, primary_type]
        ↓
RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced")
```

Packaged as a single `sklearn.Pipeline` — preprocessing is always bundled with the model, no separate artifact needed at inference.

### Model Comparison

| Model | CV ROC-AUC | Test ROC-AUC | HR Recall | HR Precision | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.678 ± 0.038 | 0.659 | 0.61 | 0.51 | 0.61 |
| Gradient Boosting | 0.664 ± 0.032 | 0.654 | 0.29 | 0.67 | 0.65 |
| **Random Forest (final)** | **0.659 ± 0.032** | **0.652** | **0.64** | 0.48 | 0.57 |
| MLP — 2 layers (64→32) | 0.664 ± 0.018 | 0.635 | 0.33 | 0.61 | 0.64 |
| MLP — 3 layers (256→128→64) | 0.654 ± 0.024 | 0.642 | 0.18 | 0.64 | 0.62 |

**Why neural networks underperform here:**
- **Dataset is too small** — 2,443 rows is far below the threshold where neural nets start to shine; they overfit or fail to generalize
- **Mostly categorical features** — tree-based methods handle one-hot encoded categoricals natively; MLPs treat all inputs as continuous and lose the inherent structure
- **No class imbalance handling** — `sklearn`'s `MLPClassifier` has no `class_weight` parameter, so both MLP variants collapse toward the majority class (recall 0.18–0.33)
- **Result:** MLP (large) had the worst recall of all five models (0.18), meaning it almost always predicted "Not High-Rated"

**Why Random Forest:** Only model that combined strong recall (0.64) with competitive AUC (0.652). `class_weight="balanced"` is the key — it internally upweights the minority class, preventing the model from ignoring high-rated restaurants entirely.

### Test Set Results (n = 489)

|  | Predicted Regular | Predicted High-Rated |
|---|---|---|
| **Actually Regular** | TN = 152 | FP = 137 |
| **Actually High-Rated** | FN = 73 | TP = 127 |

| Metric | Regular | High-Rated |
|---|---|---|
| Precision | 0.68 | 0.48 |
| Recall | 0.53 | **0.64** |
| F1 | 0.59 | 0.55 |
| Overall Accuracy | 0.57 | |
| Test ROC-AUC | **0.652** | |

### Feature Importances

| Feature Group | Importance |
|---|---|
| `primary_type` (144 OHE columns) | **63.5%** |
| `neighborhood` (20 OHE columns) | 17.9% |
| `review_count_log` | **15.0%** (strongest single feature) |
| `price_level` | 3.6% |

Top individual features: `review_count_log` (0.150), `convenience_store` type (0.109), `grocery_store` type (0.104), `fast_food_restaurant` type (0.096)

*(chart: `notebooks/model_charts.png` — ROC curve + feature group importance + model comparison)*

---

## Slide 5 — Solution Architecture

### Components

| Layer | Technology | Where |
|---|---|---|
| Data collection | Google Places API + `requests` | Local (one-time) |
| Dataset | `la_restaurants.csv` (2,443 rows) | GitHub repo |
| Model training | scikit-learn Pipeline | Local (one-time) |
| Model artifact | `model.pkl` (joblib) | Baked into API container |
| Prediction API | FastAPI 0.115 + Uvicorn | Google Cloud Run (`us-central1`) |
| Web app | Streamlit 1.45 | Google Cloud Run (`us-central1`) |
| Container registry | Google Artifact Registry | GCP project `mtcars-fastapi-zach` |
| Container build | Podman (`--platform linux/amd64`) | Local → pushed to registry |

### Data Flow

```
Google Places API
      │  collect.py  (one-time scrape)
      ▼
la_restaurants.csv
      │  train_model.py  (one-time train)
      ▼
model.pkl ──► baked into API container
                    │
                    ▼
             FastAPI  (Cloud Run)
             ├─ GET  /health
             ├─ GET  /ready
             ├─ GET  /meta  ──────────────────────────► Streamlit App  (Cloud Run)
             └─ POST /predict  ◄── prediction request ◄──────┘
                                                                    ▲
                                                                  User
```

### Key Design Decisions

- Model is bundled into the API container — no separate model store needed at this scale
- App and API are two independent Cloud Run services; `API_URL` is injected as an env var at deploy time
- Both containers target `linux/amd64` (built on Apple Silicon with `--platform` flag)
- `min-instances=0` — scales to zero when idle, stays within GCP free tier

### Tech Stack

Python 3.12 · FastAPI · Pydantic v2 · scikit-learn 1.6 · joblib · Streamlit · Podman · Google Cloud Run · Artifact Registry

---

## Slide 6 — Application Features & Demo

### Streamlit UI

4 inputs, populated live from the API's `/meta` endpoint — no hardcoded lists:

| Input | Options |
|---|---|
| Neighborhood | 20 LA neighborhoods |
| Place Type | 144 place types (italian_restaurant, sushi_restaurant, …) |
| Price Level | $ Budget / $$ Moderate / $$$ Expensive / $$$$ Very Expensive |
| Number of Reviews | Numeric input (default 500) |

**Output:**
- `st.progress` bar showing probability of ≥4.5★
- Green banner: *"Likely High-Rated ⭐ — XX% confidence"*
- Yellow banner: *"Probably Not High-Rated — XX% chance of ≥4.5★"*

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness — always `{"status":"ok"}` |
| GET | `/ready` | Readiness — 503 if model not loaded |
| GET | `/meta` | Returns all valid neighborhoods and place types |
| POST | `/predict` | 4 fields in → probability + bool out |

Interactive docs at `/docs` (Swagger UI, auto-generated by FastAPI).

### Testing

7 pytest tests covering: health, ready, meta structure, valid prediction, missing field (422), invalid price range (422), negative review count (422).

### Live Demo Script

1. Open app → dropdowns populate from `/meta` (proves API connectivity)
2. Try: *Koreatown / korean_restaurant / $$ / 500 reviews* → ~40% probability
3. Try: *Little Tokyo / italian_restaurant / $$$ / 1000 reviews* → higher probability
4. Open `/docs` → expand `POST /predict` → Try it out → run manually

---

## Slide 7 — Challenges, Learnings & Future Work

### Technical Challenges

| Challenge | Solution |
|---|---|
| Class imbalance (40% high-rated) | `class_weight="balanced"` — boosted HR recall from 0.29 → 0.64 |
| `TypeError` in `sorted()` on mixed NaN/string `primary_type` | Added `.dropna()` before sort when building `/meta` response |
| Apple Silicon → linux/amd64 container builds | `podman build --platform linux/amd64` flag |
| FastAPI `@app.on_event("startup")` deprecation | Replaced with `asynccontextmanager` lifespan handler |
| Google Places API 20-result cap per call | 5 search points per neighborhood (center + 4 directional offsets) |

### Role of AI Assistants (Claude Code)

- Scaffolded entire project structure (FastAPI + Streamlit + Dockerfiles + pytest suite) from a single requirements description
- Suggested the `ColumnTransformer` + `Pipeline` pattern — ensures preprocessing is always bundled with the model
- Identified `class_weight="balanced"` as the fix for low recall
- Handled all Cloud Run / Artifact Registry deployment commands
- **Saved ~10 hours on boilerplate.** Domain knowledge — which features to include, what the 4.5★ threshold means in context — still required human judgment.

### Key Lessons Learned

- AI coding assistants are excellent at idiomatic boilerplate; weak at knowing *what the data means*
- Always verify AI-generated `requirements.txt` version pins — they can be slightly stale
- `sklearn.Pipeline` is worth the setup cost: one artifact, zero preprocessing bugs at inference time
- Scale-to-zero Cloud Run works well for course projects — free tier is sufficient for demo duration

### Future Improvements

- Add SHAP explanations in the UI — show *why* a prediction was made (top contributing features)
- More features: operating hours, photo count, days open per week
- Map view with prediction heatmap overlaid on an LA map (Folium or PyDeck)
- CI/CD with GitHub Actions for automated test + deploy on push to `main`
- Collect user feedback in the app to retrain the model over time
