from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

OUT = Path("model")
OUT.mkdir(parents=True, exist_ok=True)

# tiny synthetic dataset
X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=0)

num_cols = list(range(X.shape[1]))

pipe = Pipeline(
    steps=[
        ("pre", ColumnTransformer([("num", "passthrough", num_cols)])),
        ("clf", LogisticRegression(max_iter=200)),
    ]
)

pipe.fit(X, y)

# save artifacts with the exact names your app expects
joblib.dump(pipe, OUT / "model.pkl")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
joblib.dump(enc, OUT / "encoder.pkl")

print("Artifacts saved to:", OUT.resolve())
