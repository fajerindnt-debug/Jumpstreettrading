"""
JST V2 — ADVANCED QUANTITATIVE ALPHA RESEARCH TERMINAL
Core Architecture & Data Pipeline
Version: 2.0.0

Signal-only research system.
No broker execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import hashlib
import numpy as np
import pandas as pd


JST_VERSION = "2.0.0"

REQUIRED_COLUMNS = [
    "time", "open", "high", "low", "close", "volume"
]


class JSTDataError(Exception):
    """Raised when market data fails validation."""


@dataclass
class DataQualityReport:
    source: str
    rows_received: int
    rows_valid: int
    duplicate_timestamps: int
    missing_values: int
    invalid_ohlc_rows: int
    stale_data: bool
    latest_timestamp: Optional[str]
    dataset_hash: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JSTDataPipeline:

    def __init__(
        self,
        stale_after_minutes: int = 30,
        min_rows: int = 80,
    ):
        self.stale_after_minutes = stale_after_minutes
        self.min_rows = min_rows

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        payload = pd.util.hash_pandas_object(
            df[REQUIRED_COLUMNS],
            index=True
        ).values.tobytes()

        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _invalid_ohlc_mask(df: pd.DataFrame):
        return (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:

        if raw is None or not isinstance(raw, pd.DataFrame):
            raise JSTDataError("Input must be a DataFrame.")

        if raw.empty:
            raise JSTDataError("Dataset is empty.")

        df = raw.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        aliases = {
            "datetime": "time",
            "timestamp": "time",
            "date": "time",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }

        df = df.rename(
            columns={
                k: v for k, v in aliases.items()
                if k in df.columns
            }
        )

        required = ["time", "open", "high", "low", "close"]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise JSTDataError(
                f"Missing columns: {missing}"
            )

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df["time"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce"
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["time", "open", "high", "low", "close"]
        ).copy()

        if df.empty:
            raise JSTDataError("No valid OHLC rows.")

        duplicate_count = int(
            df.duplicated("time", keep="last").sum()
        )

        df = (
            df.sort_values("time")
            .drop_duplicates("time", keep="last")
            .reset_index(drop=True)
        )

        prices = df[
            ["open", "high", "low", "close"]
        ].to_numpy(dtype=float)

        if not np.isfinite(prices).all():
            raise JSTDataError("Non-finite prices detected.")

        if (prices <= 0).any():
            raise JSTDataError("Non-positive prices detected.")

        invalid = self._invalid_ohlc_mask(df)

        if invalid.any():
            raise JSTDataError(
                f"{int(invalid.sum())} invalid candles detected."
            )

        if (df["volume"].dropna() < 0).any():
            raise JSTDataError("Negative volume detected.")

        if len(df) < self.min_rows:
            raise JSTDataError(
                f"Insufficient data: {len(df)} rows. "
                f"Minimum: {self.min_rows}."
            )

        df.attrs["duplicate_timestamps_removed"] = duplicate_count
        df.attrs["rows_received"] = len(raw)

        return df
