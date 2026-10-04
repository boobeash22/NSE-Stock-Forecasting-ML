from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.api import VAR


@dataclass
class ForecastResult:
    metrics: Dict[str, float]
    predictions: np.ndarray


class LNNBenchmarkModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_cols = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = list(X.columns)
        self.model.fit(X[self.feature_cols].astype(float), y.astype(float))
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X[self.feature_cols].astype(float))


class NNARModel:
    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=800,
            random_state=random_state,
        )
        self.scaler = None
        self.feature_cols = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.preprocessing import StandardScaler

        self.feature_cols = list(X.columns)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X[self.feature_cols].astype(float))
        self.model.fit(X_scaled, y.astype(float))
        return self

    def predict(self, X: pd.DataFrame):
        if self.scaler is None:
            raise ValueError("NNAR model has not been fitted.")
        X_scaled = self.scaler.transform(X[self.feature_cols].astype(float))
        return self.model.predict(X_scaled)


class VARModelWrapper:
    def __init__(self, p: int = 3):
        self.p = p
        self.model = None
        self.result = None

    def fit(self, df: pd.DataFrame):
        self.model = VAR(df.astype(float))
        self.result = self.model.fit(self.p)
        return self.result

    def predict(self, y_history: pd.DataFrame, steps: int):
        if self.result is None:
            raise ValueError("VAR model has not been fitted.")
        arr = y_history.astype(float).to_numpy()
        forecast = self.result.forecast(y=arr, steps=steps)
        return forecast[:, 0]


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def combine_forecasts(forecasts: Dict[str, np.ndarray], metrics: Dict[str, Dict[str, float]]) -> np.ndarray:
    weights = {}
    for name, stats in metrics.items():
        weights[name] = 1.0 / max(stats["RMSE"], 1e-8)

    total_weight = sum(weights.values())
    predictions = []
    for name, pred in forecasts.items():
        predictions.append((weights[name] / total_weight) * np.asarray(pred).ravel())

    combined = np.sum(np.vstack(predictions), axis=0)
    return combined
