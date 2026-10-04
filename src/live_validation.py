import numpy as np
import pandas as pd

from src.dataset import LNNBenchmarkModel, NNARModel, evaluate_predictions


def walk_forward_validate(df: pd.DataFrame, target_col: str = "Close", horizon: int = 5, lookback: int = 20):
    from src.dataset import create_multihorizon_dataset

    recent = df.sort_values("date").reset_index(drop=True).tail(120).copy()
    if len(recent) <= lookback + horizon:
        return None

    dataset, feature_cols = create_multihorizon_dataset(recent, [horizon], lookback=lookback, target_col=target_col)
    if dataset.empty:
        return None

    train = dataset.iloc[:-10].copy()
    test = dataset.iloc[-10:].copy()
    X_train = train.drop(columns=["date", "symbol", "target", "horizon"], errors="ignore")
    y_train = train["target"]
    X_test = test.drop(columns=["date", "symbol", "target", "horizon"], errors="ignore")
    y_test = test["target"]

    lnn = LNNBenchmarkModel()
    lnn.fit(X_train, y_train)
    pred_lnn = lnn.predict(X_test)

    nnar = NNARModel(hidden_layer_sizes=(32, 16), random_state=42)
    nnar.fit(X_train, y_train)
    pred_nnar = nnar.predict(X_test)

    metrics_lnn = evaluate_predictions(y_test, pred_lnn)
    metrics_nnar = evaluate_predictions(y_test, pred_nnar)

    return {
        "horizon": horizon,
        "LNN": metrics_lnn,
        "NNAR": metrics_nnar,
    }
