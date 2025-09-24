"""Model training, persistence, inference, and slice metrics."""

import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Train a machine learning model and return it.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validate the trained model using precision, recall, and F1 (beta=1).

    Returns (precision, recall, fbeta).
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inference and return predictions."""
    return model.predict(X)


def save_model(model, path):
    """
    Serialize a model (or encoder) to a file.

    We prefer joblib (faster/robust for numpy arrays). If joblib fails for any
    reason, fall back to standard pickle for maximum compatibility.
    """
    # Ensure the directory exists (handle '' when saving to CWD).
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        joblib.dump(model, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(path):
    """
    Load a serialized artifact from disk.

    Try joblib first (how our CI training script saves), then fall back to pickle.
    """
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Compute precision/recall/F1 on a slice of the data
    (rows where `column_name == slice_value`).
    """
    # Filter to the slice
    slice_df = data[data[column_name] == slice_value]
    if slice_df.shape[0] == 0:
        return 0.0, 0.0, 0.0

    # Process using the *trained* encoder/lb
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        cat
