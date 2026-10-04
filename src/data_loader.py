from typing import Dict, List

import pandas as pd
import yfinance as yf


def fetch_market_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    market_data: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if df.empty:
            continue

        df = df.reset_index()
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        df["symbol"] = symbol
        columns = ["date", "symbol"] + [col for col in df.columns if col not in ["date", "symbol"]]
        df = df[columns]
        market_data[symbol] = df

    return market_data
