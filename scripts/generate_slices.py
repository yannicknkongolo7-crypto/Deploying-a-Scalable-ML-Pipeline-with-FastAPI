from pathlib import Path
import pandas as pd

from ml.model import load_model
from ml.data import process_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"

# Load artifacts
model = load_model(MODEL_PATH)
encoder = load_model(ENCODER_PATH)
LB_PATH = PROJECT_ROOT / "model" / "lb.pkl"
lb = None
if LB_PATH.exists():
    lb = load_model(LB_PATH)

# Read the dataset (if present)
data_path = PROJECT_ROOT / "data" / "census.csv"
if not data_path.exists():
    print("No dataset found at data/census.csv; skipping slice generation")
    exit(0)

df = pd.read_csv(data_path)

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

label = "salary"

slice_values = {
    "education": ["Bachelors", "HS-grad", "Masters"],
    "workclass": ["Private", "Self-emp-not-inc"],
    "sex": ["Male", "Female"],
}

out_lines = []
for col, vals in slice_values.items():
    for v in vals:
        df_slice = df[df[col] == v]
        if df_slice.empty:
            continue
        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = model.predict(X_slice)
        # compute simple metrics
        from sklearn.metrics import precision_score, recall_score, f1_score

        try:
            precision = precision_score(y_slice, preds, zero_division=1)
            recall = recall_score(y_slice, preds, zero_division=1)
            f1 = f1_score(y_slice, preds, zero_division=1)
        except Exception:
            precision = recall = f1 = 0.0

        out_lines.append(f"{col}={v}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

(PROJECT_ROOT / "slice_output.txt").write_text("\n".join(out_lines))
print("Wrote slice_output.txt")
