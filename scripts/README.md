# scripts/

Offline data pipeline scripts. Run these in order to reproduce the dataset and model from scratch.

## Scripts

### 1. `collect.py` — Data Collection

Queries the Google Places API (Nearby Search v1) for restaurants, cafes, bakeries, and bars across 20 LA neighborhoods.

**Requires**: `.env` file with `GOOGLE_PLACES_API_KEY` (see `.env.example`)

```bash
python scripts/collect.py
```

Outputs: `data/la_restaurants.csv`

Estimated API cost: ~$13 for a full run (400 requests × $0.032). The Places API gives $300/month free credit.

---

### 2. `eda.py` — Exploratory Data Analysis

Generates summary charts and saves them to `notebooks/eda_charts.png`.

```bash
python scripts/eda.py
```

For the full interactive notebook with narrative and model evaluation, see `notebooks/eda.ipynb`.

---

### 3. `train_model.py` — Model Training

Trains a balanced Random Forest classifier and saves the full sklearn pipeline to `models/model.pkl`.

```bash
python scripts/train_model.py
```

Outputs: `models/model.pkl`

Prints CV ROC-AUC, test ROC-AUC, and classification report.
