"""Data utilities for the MetaTrader predictive agent.

This module includes functions to load trading data, generate technical indicators,
prepare feature matrices, split the data into train and test sets, and perform
basic feature engineering. Ensuring reproducibility and proper data handling helps
produce more reliable and ethical financial predictions.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import ta

def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators (SMA, RSI, etc.) from price data.

    Args:
        df: DataFrame with at least a 'Close' column representing closing prices.

    Returns:
        DataFrame with additional columns for technical indicators.
    """
    df_ind = df.copy()
    # Simple moving averages
    df_ind['SMA_5'] = ta.trend.sma_indicator(df_ind['Close'], window=5)
    df_ind['SMA_20'] = ta.trend.sma_indicator(df_ind['Close'], window=20)
    # Relative strength index
    df_ind['RSI'] = ta.momentum.rsi(df_ind['Close'], window=14)
    # Fill missing values created by indicators
    df_ind.fillna(method='bfill', inplace=True)
    df_ind.fillna(method='ffill', inplace=True)
    return df_ind


def prepare_features(df: pd.DataFrame, target_column: str = 'target'):
    """Prepare feature matrix X and target vector y from raw DataFrame.

    Args:
        df: DataFrame containing features and target.
        target_column: Name of the target column.

    Returns:
        Tuple (X, y) where X is a DataFrame of features and y is a Series of the target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split features and target into training and testing sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test tuple.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
