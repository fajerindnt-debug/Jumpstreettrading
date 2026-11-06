
"""
JST V2 — ADVANCED QUANTITATIVE ALPHA RESEARCH TERMINAL
PHASE 1 — CORE DATA PIPELINE
Version: 2.0.1

Signal-only research system.
No broker execution.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import hashlib
import numpy as np
import pandas as pd


JST_VERSION = "2.0.1"

REQUIRED_COLUMNS = [
    "time", "open", "high", "low", "close", "volume"
]

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
}


class JSTDataError(Exception):
    """Raised when market data fails validation."""


@dataclass
class DataQualityReport:
    source: str
    timeframe: Optional[str]
    rows_received: int
    rows_valid: int
    invalid_timestamp_rows: int
    duplicate_timestamps: int
    missing_candles: int
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
        timeframe: Optional[str] = None,
        input_timezone: str = "UTC",
    ):

        if stale_after_minutes < 0:
            raise ValueError("stale_after_minutes must be >= 0")

        if min_rows < 2:
            raise ValueError("min_rows must be >= 2")

        if timeframe is not None:
            timeframe = timeframe.upper()

            if timeframe not in TIMEFRAME_MINUTES:
                raise ValueError(
                    f"Unsupported timeframe: {timeframe}"
                )

        self.stale_after_minutes = stale_after_minutes
        self.min_rows = min_rows
        self.timeframe = timeframe
        self.input_timezone = input_timezone

        self.last_diagnostics = {}

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

    def _normalize_timestamp(self, series):

        parsed = pd.to_datetime(
            series,
            errors="coerce"
        )

        # Naive timestamps are localized using the configured timezone.
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(
                self.input_timezone,
                ambiguous="NaT",
                nonexistent="NaT"
            )

        return parsed.dt.tz_convert("UTC")

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:

        if raw is None or not isinstance(raw, pd.DataFrame):
            raise JSTDataError(
                "Input must be a pandas DataFrame."
            )

        if raw.empty:
            raise JSTDataError("Dataset is empty.")

        rows_received = len(raw)

        df = raw.copy()

        df.columns = [
            str(col).strip().lower()
            for col in df.columns
        ]

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
                key: value
                for key, value in aliases.items()
                if key in df.columns
            }
        )

        required = [
            "time", "open", "high", "low", "close"
        ]

        missing_columns = [
            col for col in required
            if col not in df.columns
        ]

        if missing_columns:
            raise JSTDataError(
                f"Missing columns: {missing_columns}"
            )

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df["time"] = self._normalize_timestamp(
            df["time"]
        )

        invalid_timestamp_rows = int(
            df["time"].isna().sum()
        )

        df = df.dropna(
            subset=["time"]
        ).copy()

        for col in [
            "open", "high", "low", "close", "volume"
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        price_columns = [
            "open", "high", "low", "close"
        ]

        missing_price_rows = int(
            df[price_columns].isna().any(axis=1).sum()
        )

        df = df.dropna(
            subset=price_columns
        ).copy()

        if df.empty:
            raise JSTDataError(
                "No valid timestamp/OHLC rows remain."
            )

        duplicate_count = int(
            df.duplicated(
                subset=["time"],
                keep="last"
            ).sum()
        )

        df = (
            df.sort_values("time")
            .drop_duplicates(
                subset=["time"],
                keep="last"
            )
            .reset_index(drop=True)
        )

        prices = df[price_columns].to_numpy(
            dtype=float
        )

        if not np.isfinite(prices).all():
            raise JSTDataError(
                "Non-finite OHLC values detected."
            )

        if (prices <= 0).any():
            raise JSTDataError(
                "Non-positive price values detected."
            )

        invalid_mask = self._invalid_ohlc_mask(df)

        invalid_ohlc_rows = int(
            invalid_mask.sum()
        )

        if invalid_ohlc_rows:
            raise JSTDataError(
                f"{invalid_ohlc_rows} invalid OHLC candles detected."
            )

        volume_values = df["volume"].dropna().to_numpy(
            dtype=float
        )

        if not np.isfinite(volume_values).all():
            raise JSTDataError(
                "Non-finite volume values detected."
            )

        if (volume_values < 0).any():
            raise JSTDataError(
                "Negative volume detected."
            )

        if len(df) < self.min_rows:
            raise JSTDataError(
                f"Insufficient data: {len(df)} rows. "
                f"Minimum required: {self.min_rows}."
            )

        missing_candles = 0

        if self.timeframe is not None:

            interval = pd.Timedelta(
                minutes=TIMEFRAME_MINUTES[self.timeframe]
            )

            differences = df["time"].diff()

            missing_candles = int(
                np.maximum(
                    (differences / interval).fillna(1).round().astype(int) - 1,
                    0
                ).sum()
            )

        missing_values = int(
            df[REQUIRED_COLUMNS].isna().sum().sum()
        )

        self.last_diagnostics = {
            "rows_received": rows_received,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "missing_price_rows": missing_price_rows,
            "duplicate_timestamps": duplicate_count,
            "missing_candles": missing_candles,
            "missing_values": missing_values,
            "invalid_ohlc_rows": invalid_ohlc_rows,
        }

        df.attrs["jst_diagnostics"] = (
            self.last_diagnostics.copy()
        )

        return df

    def quality_report(
        self,
        market: pd.DataFrame,
        source: str = "unknown"
    ) -> DataQualityReport:

        if market is None or market.empty:
            raise JSTDataError(
                "Cannot report on empty market data."
            )

        latest = market["time"].iloc[-1]

        now = pd.Timestamp.now(tz="UTC")

        age_minutes = (
            now - latest
        ).total_seconds() / 60

        stale = (
            age_minutes > self.stale_after_minutes
        )

        diagnostics = market.attrs.get(
            "jst_diagnostics",
            self.last_diagnostics
        )

        status = "STALE" if stale else "VALID"

        if diagnostics.get("missing_candles", 0) > 0:
            status = "GAPS_DETECTED"

        return DataQualityReport(
            source=source,
            timeframe=self.timeframe,
            rows_received=int(
                diagnostics.get(
                    "rows_received",
                    len(market)
                )
            ),
            rows_valid=len(market),
            invalid_timestamp_rows=int(
                diagnostics.get(
                    "invalid_timestamp_rows",
                    0
                )
            ),
            duplicate_timestamps=int(
                diagnostics.get(
                    "duplicate_timestamps",
                    0
                )
            ),
            missing_candles=int(
                diagnostics.get(
                    "missing_candles",
                    0
                )
            ),
            missing_values=int(
                diagnostics.get(
                    "missing_values",
                    0
                )
            ),
            invalid_ohlc_rows=int(
                diagnostics.get(
                    "invalid_ohlc_rows",
                    0
                )
            ),
            stale_data=bool(stale),
            latest_timestamp=latest.isoformat(),
            dataset_hash=self._hash_dataframe(market),
            status=status,
        )
