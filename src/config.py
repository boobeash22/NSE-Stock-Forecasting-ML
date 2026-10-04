from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"
    target_col: str = "Close"
    lookback: int = 20
    horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
