import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rolling_up = up.ewm(alpha=1 / period, min_periods=period).mean()
    rolling_down = down.ewm(alpha=1 / period, min_periods=period).mean()
    rs = rolling_up / rolling_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).fillna(0)


def engineer_features(df: pd.DataFrame, target_col: str = "Close") -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    out["Return"] = out[target_col].pct_change().fillna(0)
    out["VolumeChange"] = out["Volume"].pct_change().fillna(0)

    for window in [5, 10, 20]:
        out[f"SMA_{window}"] = out[target_col].rolling(window, min_periods=1).mean()
        out[f"STD_{window}"] = out[target_col].rolling(window, min_periods=1).std().fillna(0)

    out["RSI_14"] = calculate_rsi(out[target_col], 14)
    out["MACD"] = calculate_macd(out[target_col])
    out["Volatility_10"] = out["Return"].rolling(10, min_periods=1).std().fillna(0)

    for lag in [1, 2, 3, 5, 10]:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    return out
