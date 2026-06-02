"""
Train a binary classifier to predict high-rated LA restaurants (rating >= 4.5).

Features: price_level, review_count, neighborhood, primary_type
Target:   high_rated (1 = rating >= 4.5, 0 = below)

Saves models/model.pkl and models/preprocessor.pkl
"""

import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "la_restaurants.csv"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"

NUMERIC_FEATURES = ["price_level", "review_count_log"]
CATEGORICAL_FEATURES = ["neighborhood", "primary_type"]
TARGET = "high_rated"
THRESHOLD = 4.5


def load_and_prepare(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["rating", "review_count"])
    df[TARGET] = (df["rating"] >= THRESHOLD).astype(int)
    df["review_count_log"] = np.log1p(df["review_count"])
    df["price_level"] = df["price_level"].fillna(df["price_level"].median())
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    from sklearn.ensemble import RandomForestClassifier
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])


def main() -> None:
    df = load_and_prepare(DATA_PATH)
    print(f"Dataset: {len(df)} restaurants, {df[TARGET].mean()*100:.1f}% high-rated")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Regular", "High-Rated"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "threshold": THRESHOLD,
        "neighborhoods": sorted(df["neighborhood"].unique().tolist()),
        "place_types": sorted(df["primary_type"].dropna().unique().tolist()),
    }, MODEL_DIR / "model.pkl")

    print(f"\nModel saved to {MODEL_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()
