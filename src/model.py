"""Training and inference script for the MetaTrader predictive agent.

This script provides functions to train a classification model on
MetaTrader data and to generate predictions using a trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .data_utils import load_data, compute_technical_indicators, prepare_features, split_data
from .utils import ensure_dir, save_model, load_model, evaluate_model


def train_model(data_path: Path, model_path: Path, test_size: float = 0.2, random_state: int = 42) -> None:
    """Train a random forest classifier using historical data and save the model.

    Args:
        data_path: Path to the CSV file containing raw price data.
        model_path: Path where the trained model will be saved (.joblib).
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
    """
    df_raw = load_data(data_path)
    df_feat = compute_technical_indicators(df_raw)
    X, y = prepare_features(df_feat)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=random_state,
            )),
        ]
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics['report'])
    ensure_dir(model_path.parent)
    save_model(model, model_path)
    print(f"Modelo guardado en {model_path}")


def predict(model_path: Path, data_path: Path) -> pd.DataFrame:
    """Generate predictions and probability of 'buy' from a trained model for new data.

    Args:
        model_path: Path to the saved model (.joblib).
        data_path: Path to the CSV file with new data for prediction.

    Returns:
        DataFrame with features, binary predictions and probability of buying.
    """
    df_raw = load_data(data_path)
    df_feat = compute_technical_indicators(df_raw)
    X, _ = prepare_features(df_feat)
    model = load_model(model_path)
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    result = X.copy()
    result["prediction"] = y_pred
    result["probability_buy"] = proba
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaTrader Predictive Agent Model")
    parser.add_argument("--train", action="store_true", help="Entrenar un nuevo modelo")
    parser.add_argument("--predict", action="store_true", help="Generar predicciones con un modelo existente")
    parser.add_argument("--data_path", type=str, required=True, help="Ruta al archivo CSV de datos")
    parser.add_argument("--model_path", type=str, default="models/model.joblib", help="Ruta para guardar o cargar el modelo")
    args = parser.parse_args()
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    if args.train:
        train_model(data_path, model_path)
    elif args.predict:
        result = predict(model_path, data_path)
        print(result.tail())
    else:
        parser.error("Debe especificar --train o --predict")


if __name__ == "__main__":
    main()
