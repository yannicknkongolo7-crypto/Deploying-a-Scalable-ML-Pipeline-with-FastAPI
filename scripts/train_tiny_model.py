#!/usr/bin/env python3
import pathlib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import save_model

# Repo root
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Data
df = pd.read_csv(ROOT / "data" / "census.csv").sample(n=4000, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process + train a tiny model (small + fast)
X, y, encoder, _ = process_data(
    df, categorical_features=cat_features, label="salary", training=True
)
clf = LogisticRegression(max_iter=400, n_jobs=None)  # tiny pickle, trains in seconds
clf.fit(X, y)

# Save to paths expected by main.py
(ROOT := ROOT / "model").mkdir(exist_ok=True)
save_model(clf, ROOT / "model.pkl")
save_model(encoder, ROOT / "encoder.pkl")

print("[CI] Tiny model + encoder written to model/")
