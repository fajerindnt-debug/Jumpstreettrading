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
    def engineer_features(
        self,
        market: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate causal quantitative features.
        All features use current and historical observations only.
        """

        if market is None or market.empty:
            raise JSTDataError("Cannot process empty market data.")

        df = market.copy()

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)

        # =====================================
        # 1. RETURNS & PRICE TRANSFORMATION
        # =====================================

        df["return_1"] = close.pct_change(1)
        df["return_3"] = close.pct_change(3)
        df["return_5"] = close.pct_change(5)
        df["return_10"] = close.pct_change(10)

        df["log_return"] = np.log(
            close / close.shift(1)
        )

        # =====================================
        # 2. EMA TREND STRUCTURE
        # =====================================

        for span in [9, 20, 50, 100, 200]:

            ema = close.ewm(
                span=span,
                adjust=False
            ).mean()

            df[f"ema_{span}"] = ema

            df[f"ema_gap_{span}"] = (
                close / ema - 1.0
            )

        df["trend_bullish"] = (
            (df["ema_20"] > df["ema_50"])
            & (df["ema_50"] > df["ema_200"])
        ).astype(int)

        df["trend_bearish"] = (
            (df["ema_20"] < df["ema_50"])
            & (df["ema_50"] < df["ema_200"])
        ).astype(int)

        df["ema_20_slope"] = (
            df["ema_20"] - df["ema_20"].shift(5)
        ) / df["ema_20"].shift(5)

        df["ema_50_slope"] = (
            df["ema_50"] - df["ema_50"].shift(5)
        ) / df["ema_50"].shift(5)

        # =====================================
        # 3. VOLATILITY ENGINE
        # =====================================

        returns = df["return_1"]

        for window in [5, 10, 20, 50]:

            df[f"volatility_{window}"] = (
                returns.rolling(
                    window,
                    min_periods=window
                ).std()
            )

        # =====================================
        # 4. TRUE RANGE & ATR
        # =====================================

        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        df["true_range"] = true_range

        for period in [7, 14, 21]:

            df[f"atr_{period}"] = (
                true_range.rolling(
                    period,
                    min_periods=period
                ).mean()
            )

        df["atr_pct"] = (
            df["atr_14"] / close
        )

        # =====================================
        # 5. MOMENTUM ENGINE
        # =====================================

        for period in [3, 5, 10, 20]:

            df[f"momentum_{period}"] = (
                close - close.shift(period)
            )

            df[f"roc_{period}"] = (
                close.pct_change(period) * 100
            )

        # =====================================
        # 6. RSI ENGINE
        # =====================================

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (
            100 / (1 + rs)
        )

        rsi = rsi.mask(
            (avg_loss == 0) & (avg_gain > 0),
            100
        )

        rsi = rsi.mask(
            (avg_gain == 0) & (avg_loss > 0),
            0
        )

        rsi = rsi.mask(
            (avg_gain == 0) & (avg_loss == 0),
            50
        )

        df["rsi_14"] = rsi

        # =====================================
        # 7. CANDLE ANATOMY
        # =====================================

        candle_range = (high - low).replace(0, np.nan)

        upper_base = pd.concat(
            [open_, close],
            axis=1
        ).max(axis=1)

        lower_base = pd.concat(
            [open_, close],
            axis=1
        ).min(axis=1)

        df["candle_body"] = close - open_

        df["candle_body_pct"] = (
            (close - open_) / open_
        )

        df["upper_wick"] = high - upper_base
        df["lower_wick"] = lower_base - low

        df["body_to_range"] = (
            (close - open_).abs() / candle_range
        )

        df["upper_wick_ratio"] = (
            df["upper_wick"] / candle_range
        )

        df["lower_wick_ratio"] = (
            df["lower_wick"] / candle_range
        )

        df["close_position"] = (
            (close - low) / candle_range
        )

        # =====================================
        # 8. PRICE POSITION & Z-SCORE
        # =====================================

        for window in [20, 50]:

            rolling_mean = close.rolling(
                window,
                min_periods=window
            ).mean()

            rolling_std = close.rolling(
                window,
                min_periods=window
            ).std()

            df[f"zscore_{window}"] = (
                (close - rolling_mean)
                / rolling_std.replace(0, np.nan)
            )

        # =====================================
        # 9. VOLUME FEATURES
        # =====================================

        volume = df["volume"]

        df["volume_change"] = volume.pct_change()

        volume_mean = volume.rolling(
            20,
            min_periods=20
        ).mean()

        volume_std = volume.rolling(
            20,
            min_periods=20
        ).std()

        df["volume_zscore"] = (
            (volume - volume_mean)
            / volume_std.replace(0, np.nan)
        )

        # =====================================
        # 10. FINAL FEATURE CLEANING
        # =====================================

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return df
