import pandas as pd


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out = out.dropna(subset=["Close", "Volume"]).copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def create_multihorizon_dataset(df: pd.DataFrame, horizons, lookback: int = 20, target_col: str = "Close"):
    frame = df.copy().sort_values("date").reset_index(drop=True)

    # Add lagged target values if missing
    for lag in range(1, lookback + 1):
        frame[f"{target_col}_lag_{lag}"] = frame[target_col].shift(lag)

    # Rolling predictors
    frame["rolling_mean_5"] = frame[target_col].shift(1).rolling(5).mean()
    frame["rolling_std_5"] = frame[target_col].shift(1).rolling(5).std().fillna(0)
    frame["rolling_mean_10"] = frame[target_col].shift(1).rolling(10).mean()
    frame["rolling_std_10"] = frame[target_col].shift(1).rolling(10).std().fillna(0)

    exclude_cols = {"date", "symbol", target_col, "Open", "High", "Low", "Adj Close"}
    feature_cols = [
        c for c in frame.columns
        if c not in exclude_cols and not c.startswith("Target_")
    ]

    rows = []
    for horizon in horizons:
        for idx in range(lookback, len(frame) - horizon):
            row = {
                "date": frame.loc[idx, "date"],
                "symbol": frame.loc[idx, "symbol"],
                "horizon": horizon,
                "target": frame.loc[idx + horizon, target_col],
            }
            for feat in feature_cols:
                row[feat] = frame.loc[idx, feat]
            rows.append(row)

    dataset = pd.DataFrame(rows)
    return dataset, feature_cols


def split_time_series(df: pd.DataFrame, train_ratio: float, validation_ratio: float, test_ratio: float):
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + validation_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test
