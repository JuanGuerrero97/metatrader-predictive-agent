"""Utilities for the MetaTrader predictive agent.

This module provides helper functions for saving and loading models,
creating directories safely, and computing evaluation metrics.
"""

from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report



def ensure_dir(path: str | Path) -> Path:
    """Ensure that a directory exists. Create it if missing.

    Args:
        path: Path to directory.

    Returns:
        Path object corresponding to the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_model(model, path: str | Path) -> None:
    """Save a trained model to disk using joblib.

    Args:
        model: Trained scikit‑learn model or pipeline.
        path: Destination path (including filename ending in .joblib).
    """
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """Load a saved model from disk.

    Args:
        path: Path to the saved .joblib file.

    Returns:
        The loaded model object.
    """
    return joblib.load(path)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a classification model and return key metrics.

    Args:
        model: Fitted model.
        X_test: Feature matrix for testing.
        y_test: Ground truth labels.

    Returns:
        Dictionary with accuracy and classification report as string.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {"accuracy": acc, "report": report}
