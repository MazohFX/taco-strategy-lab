import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="TACO Strategy Lab", layout="wide")


@dataclass
class Settings:
    cycle_length: int
    smoothing: int
    softness: float
    mode: str
    trade_direction: str
    start_year: int
    end_year: int
    upper: float
    lower: float
    risk_pct: float
    stop_pct: float
    tp_mode: str
    rr: float
    fixed_tp_pct: float
    exit_on_zero: bool
    time_exit: bool
    exit_after_bars: int
    initial_capital: float
    commission_pct: float


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "date": ["date", "time", "datetime", "timestamp"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close", "adj_close"],
    }
    mapped = {}
    for target, names in aliases.items():
        for name in names:
            if name in df.columns:
                mapped[target] = name
                break
    missing = [key for key in aliases if key not in mapped]
    if missing:
        raise ValueError(f"CSV braucht Spalten fuer: {', '.join(missing)}")
    out = df[[mapped["date"], mapped["open"], mapped["high"], mapped["low"], mapped["close"]]].copy()
    out.columns = ["date", "open", "high", "low", "close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date")
    return out.set_index("date")


def make_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2015-01-01", "2026-06-04", freq="B")
    rng = np.random.default_rng(7)
    asset_returns = rng.normal(0.00025, 0.012, len(dates))
    comp_returns = rng.normal(0.00005, 0.007, len(dates))
    asset_close = 7500 * np.exp(np.cumsum(asset_returns))
    comp_close = 100 * np.exp(np.cumsum(comp_returns))

    def ohlc_from_close(close: np.ndarray) -> pd.DataFrame:
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, len(close)))
        spread = np.abs(rng.normal(0.006, 0.004, len(close)))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)

    return ohlc_from_close(asset_close), ohlc_from_close(comp_close)


def load_yahoo(symbol: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        data = yf.download(symbol, start="2010-01-01", progress=False, auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data = data.reset_index()
        return normalize_ohlc(data)
    except Exception:
        return None


ASSET_PRESETS = {
    "UK100 proxy: FTSE 100 Index (^FTSE)": "^FTSE",
    "GER40 proxy: DAX Index (^GDAXI)": "^GDAXI",
    "US100 proxy: Nasdaq 100 (^NDX)": "^NDX",
