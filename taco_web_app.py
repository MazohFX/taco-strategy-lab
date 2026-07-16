import json
import math
import calendar
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from edge_validation import evaluate_edge, kelly_position_size


# ── Muster-Notizen (JSON-Persistenz) ─────────────────────────────────────────
_NOTES_FILE = Path(__file__).parent / "pattern_notes.json"


def _notes_key(symbol: str, entry: str, exit_: str, richtung: str) -> str:
    return f"{symbol}|{entry}|{exit_}|{richtung}"


def _load_notes() -> dict:
    if _NOTES_FILE.exists():
        try:
            return json.loads(_NOTES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_notes(notes: dict) -> None:
    _NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_data(ttl=6 * 60 * 60)
def load_ohlc_data(symbol: str, source: str = "mt5", fallback_to_yahoo: bool = False):
    from pathlib import Path as _Path
    _mt5_dir = _Path(__file__).parent / "data" / "mt5"
    _path = _mt5_dir / f"{symbol.upper()}.csv"
    if not _path.exists():
        _path = _mt5_dir / f"{symbol.lower()}.csv"
    if not _path.exists():
        return None
    try:
        _df = pd.read_csv(_path)
        _df.columns = [str(c).strip().lower() for c in _df.columns]
        _date_col = next((c for c in _df.columns if c in ("date","time","datetime","timestamp")), None)
        if _date_col is None:
            return None
        _df[_date_col] = pd.to_datetime(_df[_date_col], errors="coerce")
        _df = _df.dropna(subset=[_date_col]).rename(columns={
            _date_col: "Date",
            next((c for c in _df.columns if c in ("open","o")), "open"): "Open",
            next((c for c in _df.columns if c in ("high","h")), "high"): "High",
            next((c for c in _df.columns if c in ("low","l")),  "low"):  "Low",
            next((c for c in _df.columns if c in ("close","c","adj close")), "close"): "Close",
        })
        for _col in ["Open","High","Low","Close"]:
            _df[_col] = pd.to_numeric(_df[_col], errors="coerce")
        _df = _df.dropna(subset=["Open","High","Low","Close"]).sort_values("Date").drop_duplicates("Date")
        return _df[["Date","Open","High","Low","Close"]]
    except Exception:
        return None


APP_NAME = "Quant Taco Swing Strategie"

st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: #060b12;
    }
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2rem !important;
        max-width: min(96vw, 120rem) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1520 !important;
        border-right: 1px solid rgba(148,163,184,.10) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.6rem;
    }

    /* Sidebar Logo / Title area */
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* ── Navigation Radio als vertikale Pill-Nav ── */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 2px !important;
        flex-direction: column !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        background: transparent;
        border-radius: 6px;
        padding: 8px 12px !important;
        margin: 0 !important;
        cursor: pointer;
        transition: background .15s;
        align-items: center;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(98,200,232,.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] p {
        font-size: .82rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        letter-spacing: .01em;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
        background: rgba(98,200,232,.12) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] p {
        color: #62c8e8 !important;
        font-weight: 700 !important;
    }
    /* Radio-Kreis ausblenden */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {
        display: none !important;
    }

    /* ── Sidebar Caption / Label ── */
    [data-testid="stSidebar"] .stCaption {
        color: #334155 !important;
        font-size: .70rem !important;
        padding: 0 12px;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: .72rem !important;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #475569 !important;
        margin: 16px 0 4px 4px !important;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(148,163,184,.08);
    }

    /* ── Sidebar Header (st.header in sidebar) ── */
    [data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] h3 {
        font-size: .72rem !important;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #475569 !important;
    }

    /* ── Main Title ── */
    h1[data-testid="stHeading"], .stTitle {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #e2e8f0 !important;
        letter-spacing: -.01em;
    }
    h2 { color: #cbd5e1 !important; font-size: 1.1rem !important; font-weight: 700 !important; }
    h3 { color: #94a3b8 !important; font-size: .95rem !important; font-weight: 600 !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #0d1520 !important;
        border: 1px solid rgba(148,163,184,.10) !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: .72rem !important; text-transform: uppercase; letter-spacing: .05em; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.1rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: .78rem !important; }

    /* ── Plotly Charts ── */
    div[data-testid="stPlotlyChart"] {
        background: #0d1520 !important;
        border: 1px solid rgba(148,163,184,.10) !important;
        border-radius: 8px !important;
        padding: 6px !important;
        overflow: hidden;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.10) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* ── Buttons ── */
    [data-testid="stButton"] button {
        background: #62c8e8 !important;
        color: #0f172a !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: .80rem !important;
        padding: 8px 16px !important;
        transition: opacity .15s;
    }
    [data-testid="stButton"] button:hover { opacity: .85; }
    [data-testid="stButton"] button[kind="secondary"] {
        background: #1e293b !important;
        color: #94a3b8 !important;
    }

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] select,
    div[data-baseweb="select"] {
        background: #0d1520 !important;
        border-color: rgba(148,163,184,.15) !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
    }

    /* ── Checkboxes ── */
    [data-testid="stCheckbox"] label p {
        font-size: .82rem !important;
        color: #94a3b8 !important;
    }

    /* ── Divider / hr ── */
    hr { border-color: rgba(148,163,184,.10) !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] { color: #62c8e8 !important; }

    /* ── Warning / Info boxes ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border: 1px solid rgba(148,163,184,.12) !important;
        background: #0d1520 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #060b12; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    slippage_pct: float


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


@st.cache_data(ttl=6 * 60 * 60)
def load_yahoo(symbol: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        data = yf.download(symbol, start="1990-01-01", progress=False, auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data = data.reset_index()
        return normalize_ohlc(data)
    except Exception:
        return None


@st.cache_data(ttl=6 * 60 * 60)
def load_seasonality_data(symbol: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        data = yf.download(symbol, start="1900-01-01", progress=False, auto_adjust=False, threads=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data = data.reset_index()
        return normalize_ohlc(data)
    except Exception:
        return None


def normalize_loader_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    if "date" not in data.columns and "datetime" not in data.columns and "time" not in data.columns:
        data = data.reset_index()
    return normalize_ohlc(data)


def get_mt5_base_symbol(asset_label: str, yahoo_symbol: str) -> str:
    explicit_map = {
        "CAD=X": "USDCAD",
        "CHF=X": "USDCHF",
        "JPY=X": "USDJPY",
        "^GSPC": "US500",
        "^NDX": "US100",
        "^DJI": "US30",
        "^GDAXI": "GER40",
        "^FTSE": "UK100",
        "^AXJO": "AUS200",
        "^N225": "JPN225",
        "DX-Y.NYB": "DXY",
        "GC=F": "XAUUSD",
        "SI=F": "XAGUSD",
    }
    if yahoo_symbol in explicit_map:
        return explicit_map[yahoo_symbol]

    label_head = asset_label.split(" proxy:")[0].split(" / ")[-1].split(" ")[0]
    if label_head and label_head.isalnum():
        return label_head.upper()

    return (
        str(yahoo_symbol)
        .replace("=X", "")
        .replace("=F", "")
        .replace("^", "")
        .replace("-", "")
        .replace(".", "")
        .upper()
    )


def get_presidential_cycle(year: int) -> str:
    cycle_map = {
        0: "Election Year",
        1: "Post-Election Year",
        2: "Midterm Year",
        3: "Pre-Election Year",
    }
    return cycle_map[int(year) % 4]


def filter_years_by_lookback_and_cycle(
    df: pd.DataFrame,
    lookback_years: int | None,
    cycle_filter: str,
) -> list[int]:
    if df is None or df.empty:
        return []
    years = sorted(pd.Index(df.index.year).unique().astype(int).tolist())
    completed_years = [year for year in years if year < date.today().year]
    if completed_years:
        years = completed_years
    if lookback_years is not None and years:
        last_year = max(years)
        years = [year for year in years if year >= last_year - int(lookback_years) + 1]
    if cycle_filter != "Alle Jahre":
        target = cycle_filter.replace("US Presidential Cycle: ", "")
        years = [year for year in years if get_presidential_cycle(year) == target]
    return years


def _valid_month_day(year: int, month: int, day: int) -> pd.Timestamp:
    last_day = calendar.monthrange(int(year), int(month))[1]
    return pd.Timestamp(year=int(year), month=int(month), day=min(int(day), last_day))


def _seasonality_base_layout(title: str, height: int = 420) -> dict:
    return {
        "title": {"text": title, "font": {"size": 17, "color": "#e2e8f0"}, "y": 0.98},
        "height": height,
        "paper_bgcolor": "#111923",
        "plot_bgcolor": "#111923",
        "font": {"color": "#cbd5e1", "size": 11},
        "margin": {"l": 42, "r": 18, "t": 62, "b": 34},
        "xaxis": {
            "gridcolor": "rgba(148,163,184,.12)",
            "zerolinecolor": "rgba(148,163,184,.12)",
            "showline": False,
            "linecolor": "rgba(148,163,184,.18)",
            "tickfont": {"color": "#94a3b8", "size": 10},
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,.12)",
            "zerolinecolor": "rgba(148,163,184,.12)",
            "showline": False,
            "linecolor": "rgba(148,163,184,.18)",
            "tickfont": {"color": "#94a3b8", "size": 10},
        },
        "showlegend": False,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        "hovermode": "x unified",
    }


def build_seasonal_curve(df: pd.DataFrame, selected_years: list[int]) -> pd.DataFrame:
    if df is None or df.empty or not selected_years:
        return pd.DataFrame()

    frames = []
    for year in selected_years:
        year_df = df[df.index.year == int(year)].copy()
        year_df = year_df[~((year_df.index.month == 2) & (year_df.index.day == 29))]
        if len(year_df) < 20:
            continue
        # Jede Kalendertag-Luecke (Wochenenden/Feiertage) mit dem letzten bekannten Schlusskurs
        # auffuellen, damit jedes Jahr fuer JEDEN Kalendertag einen Wert liefert. Ohne das wechselt
        # die Menge der beitragenden Jahre von Tag zu Tag (je nachdem wo Wochenenden/Feiertage genau
        # liegen), was bei stark unterschiedlichen Jahresverlaeufen einen kuenstlichen Zickzack erzeugt.
        full_range = pd.date_range(start=pd.Timestamp(year=int(year), month=1, day=1), end=pd.Timestamp(year=int(year), month=12, day=31), freq="D")
        full_range = full_range[~((full_range.month == 2) & (full_range.day == 29))]
        daily_close = year_df["close"].reindex(year_df.index.union(full_range)).sort_index().ffill().bfill()
        daily_close = daily_close.reindex(full_range)
        base = float(year_df["close"].iloc[0])
        if not base:
            continue
        normalized = daily_close / base * 100
        frames.append(
            pd.DataFrame(
                {
                    "month": full_range.month,
                    "day": full_range.day,
                    "year": int(year),
                    "indexed": normalized.to_numpy(),
                }
            )
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    curve = combined.groupby(["month", "day"], as_index=False)["indexed"].mean()
    curve["plot_date"] = [pd.Timestamp(year=2001, month=int(m), day=int(d)) for m, d in zip(curve["month"], curve["day"])]
    curve = curve.sort_values("plot_date").reset_index(drop=True)
    curve["day_label"] = curve["plot_date"].dt.strftime("%b %d")
    return curve[["plot_date", "day_label", "month", "day", "indexed"]]


def analyze_seasonal_window(
    df: pd.DataFrame,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
    selected_years: list[int],
) -> pd.DataFrame:
    rows = []
    if df is None or df.empty or not selected_years:
        return pd.DataFrame()

    crosses_year = (int(end_month), int(end_day)) < (int(start_month), int(start_day))
    for year in selected_years:
        start_date = _valid_month_day(int(year), int(start_month), int(start_day))
        end_year = int(year) + 1 if crosses_year else int(year)
        end_date = _valid_month_day(end_year, int(end_month), int(end_day))

        entry_candidates = df[df.index >= start_date]
        entry_candidates = entry_candidates[entry_candidates.index <= end_date]
        exit_candidates = df[df.index <= end_date]
        exit_candidates = exit_candidates[exit_candidates.index >= start_date]
        if entry_candidates.empty or exit_candidates.empty:
            continue

        entry = entry_candidates.iloc[0]
        entry_date = entry_candidates.index[0]
        exit_ = exit_candidates.iloc[-1]
        exit_date = exit_candidates.index[-1]
        if exit_date < entry_date:
            continue

        period = df[(df.index >= entry_date) & (df.index <= exit_date)]
        if period.empty:
            continue

        entry_price = float(entry["close"])
        exit_price = float(exit_["close"])
        profit = exit_price - entry_price
        profit_pct = profit / entry_price * 100 if entry_price else np.nan
        max_rise = (float(period["high"].max()) - entry_price) / entry_price * 100 if entry_price else np.nan
        max_drop = (float(period["low"].min()) - entry_price) / entry_price * 100 if entry_price else np.nan
        rows.append(
            {
                "Start Date": entry_date.date(),
                "Start Price": entry_price,
                "End Date": exit_date.date(),
                "End Price": exit_price,
                "Profit": profit,
                "Profit %": profit_pct,
                "Max Rise": max_rise,
                "Max Drop": max_drop,
                "Year": int(year),
                "Presidential Cycle": get_presidential_cycle(int(year)),
            }
        )

    return pd.DataFrame(rows)


def compute_season_stats(df: pd.DataFrame, trades: pd.DataFrame, active_years: list[int], all_years) -> dict:
    profit_pct = trades["Profit %"] if not trades.empty else pd.Series(dtype=float)
    profit_points = trades["Profit"] if not trades.empty else pd.Series(dtype=float)
    avg_return = profit_pct.mean() if not profit_pct.empty else np.nan
    std_return = profit_pct.std(ddof=1) if len(profit_pct) > 1 else np.nan
    sharpe = avg_return / std_return if std_return and not pd.isna(std_return) else np.nan
    if not trades.empty:
        holding_days = (
            pd.to_datetime(trades["End Date"]) - pd.to_datetime(trades["Start Date"])
        ).dt.days.clip(lower=1)
        avg_holding_days = float(holding_days.mean())
    else:
        avg_holding_days = np.nan
    if pd.isna(avg_return) or pd.isna(avg_holding_days) or avg_holding_days <= 0:
        annualized_window_return = np.nan
    else:
        annualized_window_return = ((1 + avg_return / 100) ** (365.25 / avg_holding_days) - 1) * 100
    if not trades.empty:
        trading_day_counts = []
        for _, trade in trades.iterrows():
            start_dt = pd.Timestamp(trade["Start Date"])
            end_dt = pd.Timestamp(trade["End Date"])
            trading_day_counts.append(len(df[(df.index >= start_dt) & (df.index <= end_dt)]))
        avg_trading_days = float(np.mean(trading_day_counts)) if trading_day_counts else np.nan
        gains_count = int((profit_pct > 0).sum())
        losses_count = int((profit_pct < 0).sum())
        sorted_returns = trades.sort_values("Year")["Profit %"].to_list()
        current_streak = 0
        current_side = "none"
        for value in reversed(sorted_returns):
            side = "gains" if value > 0 else "losses" if value < 0 else "flat"
            if current_streak == 0:
                current_side = side
                current_streak = 1 if side != "flat" else 0
            elif side == current_side:
                current_streak += 1
            else:
                break
        downside_std = profit_pct[profit_pct < 0].std(ddof=1)
        sortino = avg_return / abs(downside_std) if downside_std and not pd.isna(downside_std) else np.nan
    else:
        avg_trading_days = np.nan
        gains_count = losses_count = current_streak = 0
        current_side = "none"
        sortino = np.nan
    stats = {
        "Pattern-Jahre": len(active_years),
        "Rest-Jahre": max(len(all_years) - len(active_years), 0),
        "Trades": len(trades),
        "Gains": gains_count,
        "Losses": losses_count,
        "Current Streak": current_streak,
        "Trading Days": avg_trading_days,
        "Calendar Days": avg_holding_days,
        "Annualized Return": annualized_window_return,
        "Average Return": avg_return,
        "Median Return": profit_pct.median() if not profit_pct.empty else np.nan,
        "Winning Trades %": (profit_pct.gt(0).mean() * 100) if not profit_pct.empty else np.nan,
        "Avg Profit Punkte": profit_points.mean() if not profit_points.empty else np.nan,
        "Average Profit %": avg_return,
        "Average Gain": profit_pct[profit_pct > 0].mean() if not profit_pct.empty else np.nan,
        "Average Loss": profit_pct[profit_pct < 0].mean() if not profit_pct.empty else np.nan,
        "Max Profit": profit_points.max() if not profit_points.empty else np.nan,
        "Max Loss": profit_points.min() if not profit_points.empty else np.nan,
        "Standard Deviation": std_return,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Volatility": std_return,
    }
    if len(trades) > 0:
        rise_probability = gains_count / len(trades) * 100
        fall_probability = losses_count / len(trades) * 100
        flat_count = max(len(trades) - gains_count - losses_count, 0)
    else:
        rise_probability = fall_probability = np.nan
        flat_count = 0
    if pd.isna(rise_probability) or pd.isna(fall_probability):
        dominant_probability = np.nan
        dominant_label = "n/a"
    else:
        dominant_probability = rise_probability if rise_probability >= fall_probability else fall_probability
        dominant_label = "Rise" if rise_probability >= fall_probability else "Fall"
    dominant_color = "#62c8e8" if dominant_label == "Rise" else "#c25f50"
    streak_label = f"{current_streak} {current_side}" if current_streak else "0"
    return {
        "trades": trades,
        "stats": stats,
        "profit_points": profit_points,
        "rise_probability": rise_probability,
        "fall_probability": fall_probability,
        "flat_count": flat_count,
        "dominant_probability": dominant_probability,
        "dominant_label": dominant_label,
        "dominant_color": dominant_color,
        "streak_label": streak_label,
        "gains_count": gains_count,
        "losses_count": losses_count,
    }


def render_season_stats_panel(result: dict, key_suffix: str = "") -> None:
    trades = result["trades"]
    stats = result["stats"]
    profit_points = result["profit_points"]
    rise_probability = result["rise_probability"]
    fall_probability = result["fall_probability"]
    flat_count = result["flat_count"]
    dominant_probability = result["dominant_probability"]
    dominant_label = result["dominant_label"]
    dominant_color = result["dominant_color"]
    streak_label = result["streak_label"]
    gains_count = result["gains_count"]
    losses_count = result["losses_count"]

    def fmt_stat(value: float, suffix: str = "", digits: int = 2) -> str:
        if pd.isna(value):
            return "n/a"
        if digits == 0:
            return f"{value:,.0f}{suffix}"
        return f"{value:,.{digits}f}{suffix}"

    donut_values = [gains_count, losses_count, flat_count]
    if sum(donut_values) == 0:
        donut_values = [1, 0, 0]
    donut = go.Figure(
        go.Pie(
            labels=["Rise", "Fall", "Flat"],
            values=donut_values,
            hole=0.62,
            marker={"colors": ["#62c8e8", "#c25f50", "#334155"]},
            textinfo="none",
            sort=False,
        )
    )
    donut.update_layout(**_seasonality_base_layout("", 172))
    donut.update_layout(showlegend=False, margin={"l": 12, "r": 12, "t": 8, "b": 8})
    donut.add_annotation(
        text="n/a" if pd.isna(dominant_probability) else f"{dominant_probability:.0f}%",
        x=0.5,
        y=0.55,
        showarrow=False,
        font={"color": dominant_color, "size": 20},
    )
    donut.add_annotation(
        text=dominant_label,
        x=0.5,
        y=0.40,
        showarrow=False,
        font={"color": "#cbd5e1", "size": 10},
    )
    st.plotly_chart(donut, width="stretch", config={"displayModeBar": False}, key=f"season_stats_donut_{key_suffix}")
    st.markdown(
        f"""
        <div class="season-panel">
            <div class="season-panel-title">Probability</div>
            <div class="season-stat-grid">
                <div class="season-stat"><strong>{fmt_stat(rise_probability, "%")}</strong>Rise odds</div>
                <div class="season-stat negative"><strong>{fmt_stat(fall_probability, "%")}</strong>Fall odds</div>
                <div class="season-stat neutral"><strong>{stats["Gains"]}</strong>Rise years</div>
                <div class="season-stat neutral"><strong>{stats["Losses"]}</strong>Fall years</div>
            </div>
        </div>
        <div class="season-panel">
            <div class="season-panel-title">Return</div>
            <div class="season-stat-grid">
                <div class="season-stat"><strong>{fmt_stat(stats["Annualized Return"], "%")}</strong>Annualized</div>
                <div class="season-stat"><strong>{fmt_stat(stats["Winning Trades %"], "%")}</strong>Winning trades</div>
                <div class="season-stat"><strong>{fmt_stat(stats["Average Return"], "%")}</strong>Average return</div>
                <div class="season-stat"><strong>{fmt_stat(stats["Median Return"], "%")}</strong>Median return</div>
            </div>
        </div>
        <div class="season-panel">
            <div class="season-panel-title">Profit</div>
            <div class="season-stat-grid">
                <div class="season-stat"><strong>{fmt_stat(profit_points.sum(), " pts")}</strong>Total profit</div>
                <div class="season-stat"><strong>{fmt_stat(stats["Avg Profit Punkte"], " pts")}</strong>Average profit</div>
                <div class="season-stat"><strong>{fmt_stat(stats["Max Profit"], " pts")}</strong>Max profit</div>
                <div class="season-stat negative"><strong>{fmt_stat(stats["Max Loss"], " pts")}</strong>Max loss</div>
            </div>
        </div>
        <div class="season-panel">
            <div class="season-panel-title">Gains / Losses</div>
            <div class="season-stat-grid">
                <div class="season-stat"><strong>{stats["Gains"]}</strong>Gains</div>
                <div class="season-stat negative"><strong>{stats["Losses"]}</strong>Losses</div>
                <div class="season-stat"><strong>{fmt_stat(stats["Average Gain"], "%")}</strong>Avg gain</div>
                <div class="season-stat negative"><strong>{fmt_stat(stats["Average Loss"], "%")}</strong>Avg loss</div>
                <div class="season-stat"><strong>{fmt_stat(trades["Max Rise"].max() if not trades.empty else np.nan, "%")}</strong>Max rise</div>
                <div class="season-stat negative"><strong>{fmt_stat(trades["Max Drop"].min() if not trades.empty else np.nan, "%")}</strong>Max drop</div>
            </div>
        </div>
        <div class="season-panel">
            <div class="season-panel-title">Miscellaneous</div>
            <div class="season-stat-grid">
                <div class="season-stat neutral"><strong>{stats["Trades"]}</strong>Trades</div>
                <div class="season-stat neutral"><strong>{streak_label}</strong>Current streak</div>
                <div class="season-stat neutral"><strong>{fmt_stat(stats["Trading Days"], "", 0)}</strong>Trading days</div>
                <div class="season-stat neutral"><strong>{fmt_stat(stats["Calendar Days"], "", 0)}</strong>Calendar days</div>
                <div class="season-stat neutral"><strong>{fmt_stat(stats["Standard Deviation"], "%")}</strong>Std. deviation</div>
                <div class="season-stat neutral"><strong>{fmt_stat(stats["Sharpe Ratio"], "")}</strong>Sharpe ratio</div>
                <div class="season-stat neutral"><strong>{fmt_stat(stats["Sortino Ratio"], "")}</strong>Sortino ratio</div>
                <div class="season-stat neutral"><strong>{fmt_stat(stats["Volatility"], "%")}</strong>Volatility</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


MAG7_TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
}


def is_nasdaq_asset(asset_label: str, symbol: str) -> bool:
    text = f"{asset_label} {symbol}".upper()
    return "US100" in text or "NASDAQ" in text or "^NDX" in text


@st.cache_data(ttl=60 * 60)
def load_mag7_snapshot() -> pd.DataFrame:
    try:
        import yfinance as yf

        symbols = list(MAG7_TICKERS.keys())
        raw = yf.download(symbols, period="1y", progress=False, auto_adjust=True, threads=False)
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])

        rows = []
        for ticker in symbols:
            if ticker not in close.columns:
                continue
            series = close[ticker].dropna()
            if len(series) < 50:
                continue
            last = float(series.iloc[-1])
            ret_5d = (last / float(series.iloc[-6]) - 1) * 100 if len(series) >= 6 else np.nan
            ret_1m = (last / float(series.iloc[-22]) - 1) * 100 if len(series) >= 22 else np.nan
            ret_3m = (last / float(series.iloc[-64]) - 1) * 100 if len(series) >= 64 else np.nan
            ret_ytd = (last / float(series[series.index.year == series.index[-1].year].iloc[0]) - 1) * 100
            sma_50 = float(series.rolling(50).mean().iloc[-1])
            sma_200 = float(series.rolling(200).mean().iloc[-1]) if len(series) >= 200 else np.nan
            rows.append(
                {
                    "Ticker": ticker,
                    "Name": MAG7_TICKERS[ticker],
                    "Close": last,
                    "5D %": ret_5d,
                    "1M %": ret_1m,
                    "3M %": ret_3m,
                    "YTD %": ret_ytd,
                    "Above 50D": last > sma_50 if not pd.isna(sma_50) else False,
                    "Above 200D": last > sma_200 if not pd.isna(sma_200) else False,
                }
            )
        return pd.DataFrame(rows).sort_values("1M %", ascending=False)
    except Exception:
        return pd.DataFrame()


def render_mag7_panel() -> None:
    mag7 = load_mag7_snapshot()
    if mag7.empty:
        st.info("MAG7 Snapshot konnte gerade nicht geladen werden. Die Kursdaten kommen separat ueber Yahoo Finance.")
        return

    breadth = float(mag7["Above 50D"].mean() * 100)
    leader = mag7.iloc[0]
    laggard = mag7.iloc[-1]

    st.markdown("### Nasdaq MAG7 Breadth")
    top_cols = st.columns([1.2, 1.2, 1.2])
    top_cols[0].markdown(
        f"<div class='season-panel'><div class='season-panel-title'>MAG7 Breadth</div>"
        f"<div class='season-stat'><strong>{breadth:.0f}%</strong>ueber 50-Tage-Linie</div></div>",
        unsafe_allow_html=True,
    )
    top_cols[1].markdown(
        f"<div class='season-panel'><div class='season-panel-title'>Staerkster 1M</div>"
        f"<div class='season-stat'><strong>{leader['Ticker']} {leader['1M %']:+.2f}%</strong>{leader['Name']}</div></div>",
        unsafe_allow_html=True,
    )
    top_cols[2].markdown(
        f"<div class='season-panel'><div class='season-panel-title'>Schwaechster 1M</div>"
        f"<div class='season-stat negative'><strong>{laggard['Ticker']} {laggard['1M %']:+.2f}%</strong>{laggard['Name']}</div></div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    colors = np.where(mag7["1M %"] >= 0, "#62c8e8", "#c25f50")
    fig.add_trace(go.Bar(x=mag7["Ticker"], y=mag7["1M %"], marker_color=colors))
    fig.update_layout(**_seasonality_base_layout("MAG7 1-Month Performance", 300))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    display = mag7[["Ticker", "Name", "5D %", "1M %", "3M %", "YTD %", "Above 50D", "Above 200D"]].copy()
    st.dataframe(display, width="stretch", hide_index=True)


@st.cache_data(ttl=60 * 60)
def scan_top_seasonal_setups(
    df: pd.DataFrame,
    selected_years: list[int],
    top_n: int = 4,
    min_days: int = 10,
    max_days: int = 75,
    step_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or not selected_years:
        return pd.DataFrame(), pd.DataFrame()

    base_days = pd.date_range("2001-01-01", "2001-12-31", freq=f"{step_days}D")
    rows = []
    min_trades = max(5, min(8, len(selected_years)))
    for start_marker in base_days:
        for holding_days in range(min_days, max_days + 1, step_days):
            end_marker = start_marker + pd.Timedelta(days=holding_days)
            if end_marker.year > 2001:
                end_marker = pd.Timestamp(year=2001, month=end_marker.month, day=end_marker.day)
            trades = analyze_seasonal_window(
                df,
                int(start_marker.month),
                int(start_marker.day),
                int(end_marker.month),
                int(end_marker.day),
                selected_years,
            )
            if len(trades) < min_trades:
                continue

            returns = trades["Profit %"].dropna()
            if returns.empty:
                continue
            avg_return = float(returns.mean())
            median_return = float(returns.median())
            win_rate = float((returns > 0).mean() * 100)
            fall_rate = float((returns < 0).mean() * 100)
            std_return = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
            long_score = avg_return * (win_rate / 100) * math.sqrt(len(returns))
            short_score = (-avg_return) * (fall_rate / 100) * math.sqrt(len(returns))
            period_label = f"{start_marker.strftime('%d %b')} - {end_marker.strftime('%d %b')}"
            rows.append(
                {
                    "Period": period_label,
                    "Start": start_marker.strftime("%d.%m"),
                    "End": end_marker.strftime("%d.%m"),
                    "Trades": len(trades),
                    "Avg Return %": avg_return,
                    "Median Return %": median_return,
                    "Win Rate %": win_rate,
                    "Fall Rate %": fall_rate,
                    "Std %": std_return,
                    "Long Score": long_score,
                    "Short Score": short_score,
                    "Avg Profit": float(trades["Profit"].mean()),
                    "Max Rise %": float(trades["Max Rise"].max()),
                    "Max Drop %": float(trades["Max Drop"].min()),
                }
            )

    setups = pd.DataFrame(rows)
    if setups.empty:
        return pd.DataFrame(), pd.DataFrame()
    long_setups = setups[setups["Avg Return %"] > 0].sort_values(
        ["Long Score", "Win Rate %", "Avg Return %"], ascending=False
    ).head(top_n)
    short_setups = setups[setups["Avg Return %"] < 0].sort_values(
        ["Short Score", "Fall Rate %", "Avg Return %"], ascending=[False, False, True]
    ).head(top_n)
    return long_setups.reset_index(drop=True), short_setups.reset_index(drop=True)


def _chance_box_html(direction: str, rank: int, row: "pd.Series") -> str:
    avg = float(row["Avg Return %"])
    win = float(row["Win Rate %"]) if direction == "long" else float(row["Fall Rate %"])
    trades = int(row["Trades"])
    period = str(row["Period"])
    win_label = "Win Rate" if direction == "long" else "Fall Rate"
    avg_sign = "+" if avg >= 0 else ""
    border_color = "#62c8e8" if direction == "long" else "#c25f50"
    val_color = "#62c8e8" if avg >= 0 else "#c25f50"
    dir_color = "#62c8e8" if direction == "long" else "#c25f50"
    dir_label = "LONG" if direction == "long" else "SHORT"
    return f"""
<div style="background:#141c28;border-radius:8px;padding:16px 18px 14px 18px;
            border-left:4px solid {border_color};position:relative;margin-bottom:4px;">
  <span style="position:absolute;top:12px;right:14px;color:#334155;font-size:.72rem;font-weight:700;">#{rank + 1}</span>
  <div style="font-size:.68rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;
              color:{dir_color};margin-bottom:4px;">{dir_label}</div>
  <div style="color:#e2e8f0;font-size:1.05rem;font-weight:700;margin-bottom:12px;">{period}</div>
  <div style="display:flex;gap:24px;">
    <div>
      <div style="font-size:1.12rem;font-weight:800;color:{val_color};">{avg_sign}{avg:.2f}%</div>
      <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Ø Bewegung</div>
    </div>
    <div>
      <div style="font-size:1.12rem;font-weight:800;color:#e2e8f0;">{win:.0f}%</div>
      <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">{win_label}</div>
    </div>
    <div>
      <div style="font-size:1.12rem;font-weight:800;color:#e2e8f0;">{trades}</div>
      <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Trades</div>
    </div>
  </div>
</div>"""


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _fetch_ki_analyse(symbol: str, asset_name: str, api_key: str) -> str:
    today_str = date.today().strftime("%d. %B %Y")
    prompt = f"""Du bist ein erfahrener quantitativer Marktanalyst. Heute ist der {today_str}.

Erstelle eine aktuelle Marktanalyse für: {asset_name} ({symbol})

Analysiere folgende Bereiche mit aktuellen Zahlen und Fakten:
1. Zinspolitik: Welche Zentralbank ist zuständig, aktueller Leitzins, letzte Entscheidung, Ausblick
2. Inflation: Aktuelle Inflationsrate, Trend (steigend/fallend)
3. Arbeitsmarkt: Aktuelle Arbeitslosenquote, letzte Jobdaten (z.B. NFP), Stellenabbau vs. Neueinstellungen
4. Wirtschaftswachstum: BIP-Wachstum, Rezessionsrisiko
5. Geopolitik: Relevante politische Ereignisse die den Markt beeinflussen
6. Markt-Regime: Risikobereit (Risk-On) oder Risikoscheu (Risk-Off)

Schreibe einen kompakten deutschen Fließtext in 6-8 Sätzen mit konkreten Zahlen. Keine Aufzählungen. Beende mit einer klaren Einschätzung: bullish, neutral oder bearish."""

    # Kein try/except um den eigentlichen API-Call: st.cache_data cacht nur den
    # Rueckgabewert, keine Exceptions. Wuerde hier ein Fehler-String zurueckgegeben,
    # waere der (z.B. eine Quota-Meldung) fuer die vollen 7 Tage eingefroren.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def _grounding_tool_config():
        try:
            return types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        except Exception:
            return None  # aeltere SDK-Version ohne Grounding-Tool-Support

    # Reihenfolge: 1) primaeres Modell mit Live-Suche, 2) dasselbe Modell ohne Grounding
    # (Grounding laeuft oft ueber ein eigenes, strengeres Kontingent als normale
    # Textgenerierung -- hilft bei Quota-Fehlern), 3) anderes Modell ganz ohne Grounding
    # (hilft, wenn genau das primaere Modell gerade ueberlastet ist -- "503 UNAVAILABLE /
    # high demand" ist ein Kapazitaetsproblem bei Google, kein Konto-/Quota-Problem und
    # betrifft typischerweise nur ein einzelnes Modell zu einem bestimmten Zeitpunkt).
    attempts = [
        {"model": "gemini-3.5-flash", "config": _grounding_tool_config()},
        {"model": "gemini-3.5-flash", "config": None},
        {"model": "gemini-2.5-flash", "config": None},
    ]
    last_exc: Exception | None = None
    for i, attempt in enumerate(attempts):
        kwargs = {"model": attempt["model"], "contents": prompt}
        if attempt["config"] is not None:
            kwargs["config"] = attempt["config"]
        try:
            response = client.models.generate_content(**kwargs)
            return response.text.strip()
        except Exception as exc:
            last_exc = exc
            if i == len(attempts) - 1:
                raise
    raise last_exc


def render_ki_analyse(asset_label: str, symbol: str) -> None:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin:24px 0 10px 0;">
            <span style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">KI Marktanalyse</span>
            <span style="font-size:.68rem;font-weight:600;color:#62c8e8;background:rgba(98,200,232,.12);
                         border:1px solid rgba(98,200,232,.25);border-radius:4px;padding:2px 7px;">
                Gemini AI · Google Search · Wöchentlich aktuell
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.markdown(
            """
            <div style="background:#0d1520;border:1px solid rgba(239,68,68,.25);border-radius:8px;
                        padding:14px 16px;font-size:.82rem;color:#94a3b8;">
                🔑 Kein <code>GEMINI_API_KEY</code> gefunden.<br>
                <span style="color:#64748b;">Streamlit Cloud → App → Settings → Secrets → eintragen:</span><br>
                <code style="color:#62c8e8;">GEMINI_API_KEY = "AIza..."</code><br><br>
                Kostenlosen API-Key erhältst du unter
                <a href="https://aistudio.google.com" target="_blank" style="color:#62c8e8;">aistudio.google.com</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    asset_name = asset_label.split(" proxy:")[0].strip() if " proxy:" in asset_label else asset_label
    cache_key = f"ki_analyse_manual_{symbol}"

    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        force_refresh = st.button("🔄 Jetzt aktualisieren", key=f"ki_btn_{symbol}")
    with col_info:
        st.markdown(
            "<span style='font-size:.72rem;color:#475569;line-height:2.6;display:inline-block;'>"
            "Sucht live im Web · Zinsen · Makro · Geopolitik · Sentiment · alle 4h automatisch neu"
            "</span>",
            unsafe_allow_html=True,
        )

    if force_refresh:
        _fetch_ki_analyse.clear()
        st.session_state.pop(cache_key, None)

    try:
        with st.spinner(f"Claude durchsucht das Web nach aktuellen Daten zu {asset_name}..."):
            result = _fetch_ki_analyse(symbol, asset_name, api_key)
    except Exception as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower() or "rate limit" in msg.lower():
            st.warning(f"⏳ Gemini Rate-Limit/Quota erreicht — bitte kurz warten und erneut versuchen. Rohfehler: {msg[:300]}")
        else:
            st.error(f"API-Fehler: {msg[:500]}")
        return

    result_lower = result.lower()
    last_200 = result_lower[-200:]
    if "bullish" in last_200:
        bias, bias_color = "BULLISH", "#62c8e8"
    elif "bearish" in last_200:
        bias, bias_color = "BEARISH", "#c25f50"
    else:
        bias, bias_color = "NEUTRAL", "#94a3b8"

    # Reihenfolge wichtig: erst Zahlen hervorheben, danach **bold** in <strong> umwandeln --
    # sonst faengt die Zahlen-Regex Ziffern aus dem gerade eingefuegten style="...1.1em"-CSS.
    highlighted = re.sub(
        r"(?<![\w.,])(\d{1,3}(?:\.\d{3})*(?:,\d+)?\s?%?)(?!\w)",
        r'<span style="color:#facc15;font-weight:700;">\1</span>',
        result,
    )
    highlighted = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#ffffff;font-size:1.1em;">\1</strong>', highlighted)

    st.markdown(
        f"""
        <div style="background:#0d1520;border:1px solid rgba(148,163,184,.10);border-radius:8px;
                    padding:20px 22px 16px 22px;margin:4px 0 16px 0;position:relative;">
            <span style="position:absolute;top:16px;right:16px;font-size:1.05rem;font-weight:800;
                         color:{bias_color};background:rgba(0,0,0,.4);
                         border:1px solid {bias_color}55;border-radius:6px;padding:6px 16px;">
                {bias}
            </span>
            <div style="font-size:1.15rem;color:#cbd5e1;line-height:1.85;padding-right:120px;">
                {highlighted}
            </div>
            <div style="display:flex;gap:16px;margin-top:14px;padding-top:12px;
                        border-top:1px solid rgba(148,163,184,.07);">
                <span style="font-size:.65rem;color:#334155;">
                    🤖 Gemini AI + Google Search
                </span>
                <span style="font-size:.65rem;color:#334155;">
                    📅 {date.today().strftime("%d.%m.%Y")} · Auto-Refresh alle 4h
                </span>
                <span style="font-size:.65rem;color:#1e293b;">
                    ⚠️ Kein Finanzrat
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_seasonal_setups(df: pd.DataFrame, active_years: list[int]) -> None:
    long_setups, short_setups = scan_top_seasonal_setups(df, active_years, top_n=2)
    if long_setups.empty and short_setups.empty:
        return

    st.subheader("Top 4 Trading Chancen")

    # Baue Liste: Top 2 Long + Top 2 Short, interleaved: L1 S1 / L2 S2
    long_list = [(int(i), row) for i, row in long_setups.iterrows()]
    short_list = [(int(i), row) for i, row in short_setups.iterrows()]

    pairs = []
    for idx in range(max(len(long_list), len(short_list))):
        left = ("long", long_list[idx][0], long_list[idx][1]) if idx < len(long_list) else None
        right = ("short", short_list[idx][0], short_list[idx][1]) if idx < len(short_list) else None
        pairs.append((left, right))

    for left, right in pairs:
        col_a, col_b = st.columns(2)
        if left:
            with col_a:
                st.markdown(_chance_box_html(left[0], left[1], left[2]), unsafe_allow_html=True)
        if right:
            with col_b:
                st.markdown(_chance_box_html(right[0], right[1], right[2]), unsafe_allow_html=True)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-Konfidenzintervall für Winrate. Gibt (low, high) als Dezimalwerte zurück."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _compute_stars(wr: float, avg_ret: float, avg_dd: float, max_dd: float,
                   sharpe: float, robustheit: str, atr_pct: float,
                   n_trades: int = 10) -> int:
    """1–5 Sterne — Robustheit setzt harte Obergrenze, Rest sind Qualitätspunkte."""
    # Harte Obergrenzen durch Robustheit
    rob_cap = {"🟢 Stark": 5, "✅ Robust": 4, "⚠️ Sensitiv": 3, "❌ Fragil": 2, "—": 3}
    max_stars = rob_cap.get(robustheit, 3)

    score = 0.0
    # Winrate (0–25 Punkte) — weniger Gewicht als früher
    score += min(25, max(0, (wr - 0.60) / 0.40 * 25))

    # Avg Profit absolut (0–25 Punkte): <0.5% = kaum Punkte, 1.5%+ = voll
    score += min(25, max(0, (avg_ret * 100 - 0.3) / 1.2 * 25))

    # Profit/Risiko-Verhältnis (0–25 Punkte): Ø Gewinn vs Ø DD
    if avg_dd and abs(avg_dd) > 0:
        pnl_risk = (avg_ret * 100) / abs(avg_dd * 100)
        score += min(25, max(0, (pnl_risk - 0.5) / 2.0 * 25))
    else:
        score += min(25, max(0, avg_ret * 100 / 2.0 * 25))

    # Sharpe (0–25 Punkte)
    if not np.isnan(sharpe):
        score += min(25, max(0, sharpe / 3.0 * 25))

    # Harte Strafen für schlechtes Profit/Risiko
    if avg_ret * 100 < 0.4:          max_stars = min(max_stars, 2)
    elif avg_ret * 100 < 0.8:        max_stars = min(max_stars, 3)
    elif avg_ret * 100 < 1.2:        max_stars = min(max_stars, 4)

    # Wilson-CI Strafe: breites CI = unsichere WR = Sterne begrenzen
    wins_est = round(wr * n_trades)
    ci_low, ci_high = wilson_ci(wins_est, n_trades)
    ci_width = ci_high - ci_low
    if ci_width > 0.55:    max_stars = min(max_stars, 2)  # sehr unsicher (n<6)
    elif ci_width > 0.40:  max_stars = min(max_stars, 3)  # unsicher (n≈8)
    elif ci_width > 0.28:  max_stars = min(max_stars, 4)  # mäßig sicher (n≈12)

    # Sterne aus Score: 0–40=1, 40–55=2, 55–70=3, 70–85=4, 85+=5
    if score >= 85:   raw = 5
    elif score >= 70: raw = 4
    elif score >= 55: raw = 3
    elif score >= 40: raw = 2
    else:             raw = 1

    return min(raw, max_stars)


def _wr_stats_for_mask(
    long_ret: np.ndarray,
    short_ret: np.ndarray,
    long_dd: np.ndarray,
    short_dd: np.ndarray,
    entry_years: np.ndarray,
    mask: np.ndarray,
    dir_: str,
    min_trades: int,
) -> dict | None:
    if mask.sum() < min_trades:
        return None
    ret = long_ret[mask] if dir_ == "long" else short_ret[mask]
    dd = long_dd[mask] if dir_ == "long" else short_dd[mask]
    yr = entry_years[mask]
    _, first_idx = np.unique(yr, return_index=True)
    ret, dd = ret[first_idx], dd[first_idx]
    nt = len(ret)
    if nt < min_trades:
        return None
    wr = float((ret > 0).sum() / nt)
    avg_ret = ret.mean()
    std_ret = ret.std(ddof=1) if nt > 1 else 0.0
    return {"wr": wr, "nt": nt, "avg_ret": avg_ret, "std_ret": std_ret,
            "avg_dd": dd.mean(), "max_dd": dd.min()}


def _scan_patterns_cached(
    year_data: dict,          # {year: {doys, closes, highs, lows, atrs}}
    sorted_years: np.ndarray,
    end_year: int,
    lookback_years: int,
    entry_doy: int,
    exit_doy: int,
    directions: tuple[str, ...],
    min_winrate: float,
    min_trades: int,
) -> list[dict]:
    """Kalenderbasierter Scanner: Entry und Exit sind feste DOY-Punkte."""
    _raw_end = end_year
    y5  = end_year - 5  + 1
    y10 = end_year - 10 + 1
    y15 = end_year - 15 + 1
    y20 = end_year - 20 + 1
    year_start_primary = end_year - lookback_years + 1
    all_years_start = end_year - 20 + 1
    data_start_year = int(sorted_years.min())
    has_5j  = data_start_year <= y5
    has_10j = data_start_year <= y10
    has_15j = data_start_year <= y15
    has_20j = data_start_year <= y20

    # Für jedes Jahr: Entry- und Exit-Close sowie DD-Daten sammeln
    trades: list[dict] = []
    for yr in sorted_years:
        if yr < all_years_start or yr > end_year:
            continue
        yd = year_data.get(int(yr))
        if yd is None:
            continue
        yr_doys = yd["doys"]
        # Erster Handelstag >= entry_doy
        ei = int(np.searchsorted(yr_doys, entry_doy))
        if ei >= len(yr_doys):
            continue
        # Erster Handelstag >= exit_doy
        xi = int(np.searchsorted(yr_doys, exit_doy))
        if xi >= len(yr_doys) or xi <= ei:
            continue
        ep = yd["closes"][ei]
        xp = yd["closes"][xi]
        # DD: Low-Wicks NACH Entry-Close bis Exit (inkl.)
        dd_slice = slice(ei + 1, xi + 1)
        min_low  = yd["lows"][dd_slice].min()  if xi > ei else ep
        max_high = yd["highs"][dd_slice].max() if xi > ei else ep
        long_ret  = (xp - ep) / ep
        short_ret = (ep - xp) / ep
        long_dd   = (min_low  - ep) / ep
        short_dd  = (ep - max_high) / ep
        trades.append({
            "yr": int(yr), "ep": ep,
            "long_ret": long_ret, "short_ret": short_ret,
            "long_dd": long_dd, "short_dd": short_dd,
            "atr": yd["atrs"][ei],
            "td": xi - ei,  # tatsächliche Handelstage
            "entry_actual_doy": int(yr_doys[ei]),
            "exit_actual_doy":  int(yr_doys[xi]),
        })

    if len(trades) < min_trades:
        return []

    def _stats_for_years(yr_start: int, dir_: str, min_t: int | None = None) -> dict | None:
        sub = [t for t in trades if t["yr"] >= yr_start]
        if len(sub) < (min_t if min_t is not None else min_trades):
            return None
        rets = np.array([t["long_ret"] if dir_ == "long" else t["short_ret"] for t in sub])
        dds  = np.array([t["long_dd"]  if dir_ == "long" else t["short_dd"]  for t in sub])
        nt = len(rets)
        wr = float((rets > 0).sum() / nt)
        avg_ret = rets.mean()
        std_ret = rets.std(ddof=1) if nt > 1 else 0.0
        return {"wr": wr, "nt": nt, "avg_ret": avg_ret, "std_ret": std_ret,
                "avg_dd": dds.mean(), "max_dd": dds.min()}

    rows = []
    primary_trades = [t for t in trades if t["yr"] >= year_start_primary]
    if len(primary_trades) < min_trades:
        return []

    for dir_ in directions:
        stats = _stats_for_years(year_start_primary, dir_)
        if stats is None or stats["wr"] < min_winrate:
            continue

        wr        = stats["wr"]
        avg_ret   = stats["avg_ret"]
        std_ret   = stats["std_ret"]
        nt        = stats["nt"]
        avg_dd    = stats["avg_dd"]
        max_dd_val = stats["max_dd"]
        avg_td    = np.mean([t["td"] for t in primary_trades])

        sharpe = avg_ret / std_ret * np.sqrt(252 / max(avg_td, 1)) if std_ret > 1e-10 else np.nan
        sqn    = avg_ret / std_ret * np.sqrt(nt) * 100 if std_ret > 1e-10 else np.nan

        wr_5j  = (_stats_for_years(y5,  dir_, min_t=3) or {}).get("wr", np.nan)
        wr_10j = (_stats_for_years(y10, dir_, min_t=max(3, int(10 * 0.6))) or {}).get("wr", np.nan)
        wr_15j = (_stats_for_years(y15, dir_, min_t=max(3, int(15 * 0.6))) or {}).get("wr", np.nan)
        # Für 20J: wenn nicht genug Daten, alle verfügbaren Jahre nehmen
        wr_20j_raw = (_stats_for_years(y20, dir_) or {}).get("wr", np.nan)
        if not has_20j or np.isnan(wr_20j_raw):
            wr_20j_raw = (_stats_for_years(data_start_year, dir_) or {}).get("wr", np.nan)
        wr_20j = round(wr_20j_raw * 100, 1) if not np.isnan(wr_20j_raw) else np.nan
        wr_5j  = round(wr_5j  * 100, 1) if has_5j  and not np.isnan(wr_5j)  else np.nan
        wr_10j = round(wr_10j * 100, 1) if has_10j and not np.isnan(wr_10j) else np.nan
        wr_15j = round(wr_15j * 100, 1) if has_15j and not np.isnan(wr_15j) else np.nan

        # Robustheit: bidirektional ±3..±7 Tage + ATR-Effizienz
        avg_atr_pct_pre = float(np.mean([t["atr"] / t["ep"] for t in primary_trades if t["ep"] > 0]) * 100)
        avg_ret_abs = abs(avg_ret)
        atr_efficient = avg_atr_pct_pre > 0 and (avg_ret_abs / (avg_atr_pct_pre / 100)) >= 0.4

        robust_wins = robust_total = 0
        for offset in list(range(-7, -2)) + list(range(3, 8)):  # ±3..±7 bidirektional
            alt_entry = entry_doy + offset
            if alt_entry < 1 or alt_entry > 365: continue
            alt_trades_primary = []
            for yr in sorted_years:
                if yr < year_start_primary or yr > end_year: continue
                yd = year_data.get(int(yr))
                if yd is None: continue
                yr_doys = yd["doys"]
                ei2 = int(np.searchsorted(yr_doys, alt_entry))
                xi2 = int(np.searchsorted(yr_doys, exit_doy))
                if ei2 >= len(yr_doys) or xi2 >= len(yr_doys) or xi2 <= ei2: continue
                ep2, xp2 = yd["closes"][ei2], yd["closes"][xi2]
                ret2 = (xp2 - ep2)/ep2 if dir_ == "long" else (ep2 - xp2)/ep2
                alt_trades_primary.append(ret2)
            if len(alt_trades_primary) >= min_trades:
                alt_wr = (np.array(alt_trades_primary) > 0).mean()
                robust_total += 1
                if alt_wr >= min_winrate:
                    robust_wins += 1

        if robust_total == 0:
            robustheit = "—"
        else:
            _rob_ratio = robust_wins / robust_total
            # ATR-Effizienz senkt Robustheit um eine Stufe wenn Profit < 0.4× ATR
            if _rob_ratio >= 0.80 and atr_efficient:   robustheit = "🟢 Stark"
            elif _rob_ratio >= 0.80:                   robustheit = "✅ Robust"   # gute WR aber ATR-schwach
            elif _rob_ratio >= 0.60:                   robustheit = "✅ Robust"
            elif _rob_ratio >= 0.40:                   robustheit = "⚠️ Sensitiv"
            else:                                      robustheit = "❌ Fragil"

        avg_atr_pct = float(np.mean([t["atr"] / t["ep"] for t in primary_trades if t["ep"] > 0]) * 100)

        # Entry/Exit Label aus DOY (Wochenende → nächsten Montag)
        try:
            _ref = pd.Timestamp(year=2000, month=1, day=1)  # Schaltjahr → DOY stimmt mit Kalender überein
            _entry_ts = _ref + pd.Timedelta(days=entry_doy - 1)
            _exit_ts  = _ref + pd.Timedelta(days=exit_doy  - 1)
            if _entry_ts.weekday() == 5: _entry_ts += pd.Timedelta(days=2)  # Sa → Mo
            if _entry_ts.weekday() == 6: _entry_ts += pd.Timedelta(days=1)  # So → Mo
            if _exit_ts.weekday()  == 5: _exit_ts  += pd.Timedelta(days=2)
            if _exit_ts.weekday()  == 6: _exit_ts  += pd.Timedelta(days=1)
            entry_label = _entry_ts.strftime("%d. %b")
            exit_label  = _exit_ts.strftime("%d. %b")
        except Exception:
            entry_label = f"DOY {entry_doy}"
            exit_label  = f"DOY {exit_doy}"

        hold = int(round(avg_td))  # Ø Handelstage für Anzeige

        # Bester Späteinstieg (Entry DOY +3..+7)
        best_late_wr   = None
        best_late_label = None
        for offset in range(3, 8):
            alt_entry = entry_doy + offset
            alt_t = []
            for yr in sorted_years:
                if yr < year_start_primary or yr > end_year: continue
                yd = year_data.get(int(yr))
                if yd is None: continue
                yr_doys = yd["doys"]
                ei2 = int(np.searchsorted(yr_doys, alt_entry))
                xi2 = int(np.searchsorted(yr_doys, exit_doy))
                if ei2 >= len(yr_doys) or xi2 >= len(yr_doys) or xi2 <= ei2: continue
                ep2, xp2 = yd["closes"][ei2], yd["closes"][xi2]
                ret2 = (xp2 - ep2)/ep2 if dir_ == "long" else (ep2 - xp2)/ep2
                alt_t.append(ret2)
            if len(alt_t) >= min_trades:
                alt_wr2 = float((np.array(alt_t) > 0).mean() * 100)
                if best_late_wr is None or alt_wr2 > best_late_wr:
                    best_late_wr = alt_wr2
                    try:
                        best_late_label = (_ref + pd.Timedelta(days=alt_entry - 1)).strftime("%d. %b")
                    except Exception:
                        best_late_label = f"DOY {alt_entry}"

        row = {
            "Richtung": "Long" if dir_ == "long" else "Short",
            "Entry": entry_label,
            "Exit": exit_label,
            "Haltedauer (TD)": hold,
            "n (Jahre)": nt,
            "WR 5J %":  wr_5j,
            "WR 10J %": wr_10j,
            "WR 15J %": wr_15j,
            "WR 20J %": wr_20j,
            "Ø Profit %": round(avg_ret * 100, 2),
            "Ø DD %": round(avg_dd * 100, 2),
            "Max DD %": round(max_dd_val * 100, 2),
            "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else np.nan,
            "SQN": round(sqn, 2) if not np.isnan(sqn) else np.nan,
            "Ø ATR %": round(avg_atr_pct, 3) if not np.isnan(avg_atr_pct) else np.nan,
            "Robustheit": robustheit,
            "⭐ Rating": _compute_stars(wr, avg_ret, avg_dd, max_dd_val, sharpe, robustheit, avg_atr_pct, n_trades=len(primary_trades)),
            "Bester Späteinstieg": best_late_label or "—",
            "WR Späteinstieg %": round(best_late_wr, 1) if best_late_wr is not None else np.nan,
            "_entry_doy": entry_doy,
            "_exit_doy":  exit_doy,
        }
        rows.append(row)
    return rows


def scan_seasonality_patterns(
    df: pd.DataFrame,
    lookback_years: int,
    min_winrate: float,
    holding_periods: list[int],   # jetzt: Kalendertage (cal days)
    directions: list[str],
) -> pd.DataFrame:
    _raw_end = int(df.index.year.max())
    end_year = _raw_end - 1 if _raw_end >= pd.Timestamp.now().year else _raw_end
    start_year = end_year - 20 + 1
    sub = df[df.index.year >= start_year].copy()

    # ATR(14) berechnen
    _tr = pd.DataFrame({
        "hl": sub["high"] - sub["low"],
        "hc": (sub["high"] - sub["close"].shift(1)).abs(),
        "lc": (sub["low"]  - sub["close"].shift(1)).abs(),
    }).max(axis=1)
    sub = sub.copy()
    sub["atr"] = _tr.ewm(span=14, adjust=False).mean()

    # year_data aufbauen: {year -> {doys, closes, highs, lows, atrs}} — sortiert nach DOY
    year_data: dict = {}
    for yr, grp in sub.groupby(sub.index.year):
        grp_sorted = grp.sort_values(by=grp.index.name if grp.index.name else "index")
        yr_doys = grp_sorted.index.dayofyear.values.astype(int)
        sort_idx = np.argsort(yr_doys)
        year_data[int(yr)] = {
            "doys":   yr_doys[sort_idx],
            "closes": grp_sorted["close"].values.astype(float)[sort_idx],
            "highs":  grp_sorted["high"].values.astype(float)[sort_idx],
            "lows":   grp_sorted["low"].values.astype(float)[sort_idx],
            "atrs":   grp_sorted["atr"].values.astype(float)[sort_idx],
        }

    sorted_years = np.array(sorted(year_data.keys()))
    min_trades = max(int(lookback_years * 0.8), 3)

    all_rows: list[dict] = []
    # Scan: alle Entry-DOYs × alle Kalender-Haltedauern
    for entry_doy in range(1, 363):
        for cal_hold in holding_periods:
            exit_doy = entry_doy + cal_hold
            if exit_doy > 366:
                continue
            all_rows.extend(
                _scan_patterns_cached(
                    year_data, sorted_years, end_year, lookback_years,
                    entry_doy, exit_doy,
                    tuple(directions), min_winrate, min_trades,
                )
            )

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    wr_cols = [c for c in result.columns if c.startswith("WR ") and "J %" in c and "20" not in c and "Spät" not in c]
    sort_col = wr_cols[0] if wr_cols else result.columns[0]
    result = result.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return result


def _render_muster_detail() -> None:
    detail = st.session_state.get("muster_detail", {})
    saved_dfs = st.session_state.get("muster_dataframes", {})
    row = detail.get("row")
    symbol_str = detail.get("symbol", "—")
    lookback = detail.get("lookback", 10)

    _from_analyse = detail.get("from_analyse", False)
    _back_label = "← Neue Analyse" if _from_analyse else "← Zurück zum Scanner"
    if st.button(_back_label):
        st.session_state.pop("muster_detail", None)
        if _from_analyse:
            st.session_state.pop("muster_analyse_detail", None)
        st.rerun()

    richtung = row["Richtung"]
    farbe = "#4ade80" if richtung == "Long" else "#f87171"
    pfeil = "▲" if richtung == "Long" else "▼"
    wr_5  = row.get("WR 5J %",  float("nan"))
    wr_10 = row.get("WR 10J %", float("nan"))
    wr_15 = row.get("WR 15J %", float("nan"))
    wr_20 = row.get("WR 20J %", float("nan"))
    atr_val  = row.get("Ø ATR %", float("nan"))
    stars    = int(row.get("⭐ Rating", 1))
    star_str = "⭐" * stars + "☆" * (5 - stars)
    star_clr = ["#f87171","#fbbf24","#fbbf24","#a3e635","#4ade80"][stars - 1]
    profit_val = float(row.get("Ø Profit %", 0))
    profit_farbe = "#4ade80" if profit_val >= 0 else "#f87171"

    def _badge(label: str, value: str, color: str = "#9fb0c7", bold: bool = False) -> str:
        fw = "font-weight:700;" if bold else ""
        return (
            f'<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);'
            f'border-radius:7px;padding:10px 16px;min-width:110px;">'
            f'<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;'
            f'letter-spacing:.06em;margin-bottom:3px;">{label}</div>'
            f'<div style="color:{color};font-size:1rem;{fw}">{value}</div>'
            f'</div>'
        )

    # Wilson-CI für Header (beste verfügbare WR: 20J > 10J > 5J)
    _hdr_wr = wr_20 if pd.notna(wr_20) else (wr_10 if pd.notna(wr_10) else (wr_5 if pd.notna(wr_5) else None))
    _hdr_n  = 20 if pd.notna(wr_20) else (10 if pd.notna(wr_10) else 5)
    if _hdr_wr is not None:
        _ci_hdr_lo, _ci_hdr_hi = wilson_ci(round(_hdr_wr / 100 * _hdr_n), _hdr_n)
        _ci_hdr_lo *= 100; _ci_hdr_hi *= 100
        _ci_hdr_w = _ci_hdr_hi - _ci_hdr_lo
        _ci_hdr_clr   = "#4ade80" if _ci_hdr_w < 28 else ("#fbbf24" if _ci_hdr_w < 42 else "#f87171")
        _ci_hdr_label = "eng ✓" if _ci_hdr_w < 28 else ("mittel" if _ci_hdr_w < 42 else "breit ⚠")
    else:
        _ci_hdr_lo, _ci_hdr_hi, _ci_hdr_clr, _ci_hdr_label = 0, 100, "#6b7fa3", "—"

    st.markdown(
        f"""<div style="background:#0d1520;border:1px solid rgba(148,163,184,.15);
        border-radius:10px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
          <span style="color:#fff;font-size:1.6rem;font-weight:900;letter-spacing:.02em;">{symbol_str}</span>
          <span style="background:{farbe}22;border:1px solid {farbe}55;border-radius:5px;
            padding:4px 12px;color:{farbe};font-weight:800;font-size:1rem;">{pfeil} {richtung}</span>
          <span style="color:#6b7fa3;font-size:1rem;">📅 {row['Entry']} → {row['Exit']} &nbsp;·&nbsp; ⏱ {row['Haltedauer (TD)']} Handelstage</span>
          <div style="margin-left:auto;display:flex;flex-direction:column;align-items:center;gap:4px;">
            <span style="background:{star_clr}18;border:1px solid {star_clr}44;
              border-radius:8px;padding:5px 14px;font-size:1.15rem;letter-spacing:2px;"
              title="{stars}/5 Sterne">{star_str}</span>
            <span style="color:{_ci_hdr_clr};font-size:.72rem;font-weight:700;
              background:{_ci_hdr_clr}15;border:1px solid {_ci_hdr_clr}44;
              border-radius:5px;padding:2px 8px;white-space:nowrap;">
              CI {_ci_hdr_lo:.0f}–{_ci_hdr_hi:.0f}% · {_ci_hdr_label}
            </span>
          </div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          {_badge("WR 5J", f"{wr_5:.1f}%" if pd.notna(wr_5) else "—", "#f0c040", True)}
          {_badge("WR 10J", f"{wr_10:.1f}%" if pd.notna(wr_10) else "—", "#f0c040", True)}
          {_badge("WR 15J", f"{wr_15:.1f}%" if pd.notna(wr_15) else "—", "#fb923c", True)}
          {_badge("WR 20J", f"{wr_20:.1f}%" if pd.notna(wr_20) else "—", "#a78bfa", True)}
          {_badge("Ø Profit", f"{profit_val:+.2f}%", profit_farbe, True)}
          {_badge("Sharpe", str(row.get("Sharpe", "—")))}
          {_badge("SQN", str(row.get("SQN", "—")))}
          {_badge("Ø ATR(14)", f"{atr_val:.3f}%" if pd.notna(atr_val) else "—", "#94a3b8")}
          {_badge("Robustheit", str(row.get("Robustheit", "—")))}
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    _rob_legend = (
        "<div style='background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:10px;padding:18px 22px;margin-bottom:22px;'>"
        "<div style='color:#94a3b8;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;'>Robustheit · Legende</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 12px 8px 0;text-align:left;'>Stufe</th>"
        "<th style='color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 12px 8px 0;text-align:left;'>Schwelle</th>"
        "<th style='color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 0 8px 0;text-align:left;'>Bedeutung</th>"
        "</tr></thead>"
        "<tbody>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#4ade80;font-weight:700;font-size:.9rem;'>🟢 Stark</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#4ade8015;border:1px solid #4ade8030;border-radius:4px;padding:2px 10px;color:#4ade80;font-size:.8rem;font-family:monospace;white-space:nowrap;'>≥ 80 %</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Breites, zuverlässiges Fenster — funktioniert auch bei verschobenem Einstieg</td>"
        "</tr>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#a3e635;font-weight:700;font-size:.9rem;'>✅ Robust</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#a3e63515;border:1px solid #a3e63530;border-radius:4px;padding:2px 10px;color:#a3e635;font-size:.8rem;font-family:monospace;white-space:nowrap;'>60 – 79 %</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Solides Muster mit Spielraum — Nachbar-Tage funktionieren mehrheitlich</td>"
        "</tr>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#fbbf24;font-weight:700;font-size:.9rem;'>⚠️ Sensitiv</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#fbbf2415;border:1px solid #fbbf2430;border-radius:4px;padding:2px 10px;color:#fbbf24;font-size:.8rem;font-family:monospace;white-space:nowrap;'>40 – 59 %</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Muster funktioniert, hängt aber eng am exakten Datum — wenig Puffer</td>"
        "</tr>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#f87171;font-weight:700;font-size:.9rem;'>❌ Fragil</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#f8717115;border:1px solid #f8717130;border-radius:4px;padding:2px 10px;color:#f87171;font-size:.8rem;font-family:monospace;white-space:nowrap;'>&lt; 40 %</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Stark datumsabhängig — kleine Verschiebung bricht das Muster</td>"
        "</tr>"
        "</tbody></table>"
        "<div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(148,163,184,.07);color:#374151;font-size:.74rem;'>"
        "Basis: 5 benachbarte Einstiegstage (DOY +3 bis +7) — Anteil bestandener Tests ergibt die Stufe."
        "</div></div>"
    )
    st.markdown(_rob_legend, unsafe_allow_html=True)

    st.markdown("""
<div style='background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:10px;padding:18px 22px;margin-bottom:22px;'>
  <div style='color:#94a3b8;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;'>Was macht der CI-Test?</div>
  <div style='color:#6b7fa3;font-size:.85rem;line-height:1.65;'>
    Der <strong style='color:#cbd5e1;'>Konfidenzintervall-Test (CI)</strong> beantwortet eine einfache Frage:
    <em style='color:#94a3b8;'>„Ist die gemessene Winrate wirklich ein echtes Muster — oder könnte sie auch zufällig entstanden sein?"</em>
    <br><br>
    Stell dir vor, du wirfst eine Münze 10 Mal und bekommst 7× Kopf. Das sieht nach 70% Winrate aus — aber mit nur 10 Würfen könnte das purer Zufall sein.
    Genau das prüft der CI-Test: Er berechnet einen <strong style='color:#cbd5e1;'>Bereich</strong> (z.B. 35–93%), in dem die echte Wahrscheinlichkeit mit 95% Sicherheit liegt.
    Enthält dieser Bereich die 50%-Marke, ist das Muster statistisch <strong style='color:#f87171;'>nicht von Zufall unterscheidbar</strong>.
    <br><br>
    Je mehr Trades (Jahre) ein Muster hat und je höher die Winrate, desto <strong style='color:#4ade80;'>enger und positiver</strong> wird das Intervall —
    und desto sicherer kannst du sein, dass das Muster real ist und kein Datenzufall.
  </div>
</div>""", unsafe_allow_html=True)

    _ci_legend = (
        "<div style='background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:10px;padding:18px 22px;margin-bottom:22px;'>"
        "<div style='color:#94a3b8;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;'>Konfidenzintervall (CI) · Legende</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 12px 8px 0;text-align:left;'>Stufe</th>"
        "<th style='color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 12px 8px 0;text-align:left;'>Bereich</th>"
        "<th style='color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 0 8px 0;text-align:left;'>Bedeutung</th>"
        "</tr></thead>"
        "<tbody>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#4ade80;font-weight:700;font-size:.9rem;'>🟢 Eng & positiv</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#4ade8015;border:1px solid #4ade8030;border-radius:4px;padding:2px 10px;color:#4ade80;font-size:.8rem;font-family:monospace;white-space:nowrap;'>z.B. 72–85%</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Winrate statistisch klar positiv — das Muster ist robust und kein Zufall</td>"
        "</tr>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#fbbf24;font-weight:700;font-size:.9rem;'>🟡 Breit oder nah an 50%</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#fbbf2415;border:1px solid #fbbf2430;border-radius:4px;padding:2px 10px;color:#fbbf24;font-size:.8rem;font-family:monospace;white-space:nowrap;'>z.B. 52–81%</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Tendenz vorhanden, aber Stichprobe zu klein — Ergebnis könnte zufällig sein</td>"
        "</tr>"
        "<tr style='border-top:1px solid rgba(148,163,184,.07);'>"
        "<td style='padding:9px 12px 9px 0;'><span style='color:#f87171;font-weight:700;font-size:.9rem;'>🔴 Enthält 50%</span></td>"
        "<td style='padding:9px 12px 9px 0;'><span style='background:#f8717115;border:1px solid #f8717130;border-radius:4px;padding:2px 10px;color:#f87171;font-size:.8rem;font-family:monospace;white-space:nowrap;'>z.B. 44–78%</span></td>"
        "<td style='padding:9px 0;color:#6b7fa3;font-size:.84rem;'>Kein statistischer Nachweis — Winrate nicht besser als Münzwurf</td>"
        "</tr>"
        "</tbody></table>"
        "<div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(148,163,184,.07);color:#374151;font-size:.74rem;'>"
        "Das CI zeigt den Bereich, in dem die echte Winrate mit 95% Wahrscheinlichkeit liegt (Wilson-Intervall). "
        "Je enger und weiter von 50% entfernt, desto glaubwürdiger das Muster. Wenige Trades → breites CI → mehr Vorsicht."
        "</div></div>"
    )
    st.markdown(_ci_legend, unsafe_allow_html=True)

    # ── Gegenmuster-Check ─────────────────────────────────────────────────────
    _cur_entry_doy = int(row.get("_entry_doy", 0))
    _cur_exit_doy  = int(row.get("_exit_doy",  0))
    _opp_dir       = "Short" if richtung == "Long" else "Long"
    _all_results   = st.session_state.get("muster_scan_result", pd.DataFrame())
    _conflicts     = []
    if isinstance(_all_results, pd.DataFrame) and not _all_results.empty and "_entry_doy" in _all_results.columns:
        for _, _r in _all_results.iterrows():
            if _r.get("Symbol") != symbol_str:
                continue
            if _r.get("Richtung") != _opp_dir:
                continue
            _wr10 = _r.get("WR 10J %", 0) or 0
            if pd.isna(_wr10) or float(_wr10) < 70:
                continue
            _re = int(_r.get("_entry_doy", 0))
            _rx = int(_r.get("_exit_doy",  0))
            if _re == 0 or _rx == 0:
                continue
            if _re < _cur_exit_doy and _rx > _cur_entry_doy:
                _conflicts.append(_r)
    if _conflicts:
        _warn_rows = ""
        for _c in _conflicts:
            _c_wr    = _c["WR 10J %"] if "WR 10J %" in _c.index else "—"
            _c_rob   = _c["Robustheit"] if "Robustheit" in _c.index else "—"
            _c_stars = "⭐" * int(_c["⭐ Rating"]) if "⭐ Rating" in _c.index else "⭐"
            _c_entry = _c["Entry"] if "Entry" in _c.index else "—"
            _c_exit  = _c["Exit"]  if "Exit"  in _c.index else "—"
            _warn_rows += (
                f"<tr style='border-top:1px solid rgba(248,113,113,.12);'>"
                f"<td style='padding:8px 14px 8px 0;color:#f87171;font-weight:700;font-size:.88rem;'>▼ {_opp_dir}</td>"
                f"<td style='padding:8px 14px 8px 0;color:#cbd5e1;font-size:.88rem;'>{_c_entry} → {_c_exit}</td>"
                f"<td style='padding:8px 14px 8px 0;color:#fbbf24;font-size:.88rem;font-weight:700;'>{_c_wr}%</td>"
                f"<td style='padding:8px 14px 8px 0;color:#94a3b8;font-size:.85rem;'>{_c_rob}</td>"
                f"<td style='padding:8px 0;font-size:.85rem;'>{_c_stars}</td>"
                f"</tr>"
            )
        st.markdown(f"""
<div style="background:#1a0a0a;border:1px solid #f8717155;border-radius:10px;padding:18px 22px;margin-bottom:22px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
    <span style="font-size:1.2rem;">⚠️</span>
    <span style="color:#f87171;font-weight:700;font-size:.95rem;letter-spacing:.02em;">Gegenläufiges Muster erkannt — möglicher Gegenwind</span>
  </div>
  <div style="color:#94a3b8;font-size:.82rem;margin-bottom:14px;">
    Für <strong style="color:#cbd5e1;">{symbol_str}</strong> existiert im gleichen Zeitfenster ein <strong style="color:#f87171;">{_opp_dir}-Muster</strong> mit ≥ 70% Winrate (10J).
    Das deutet auf erhöhte Unsicherheit hin — beide Saisonalitäten konkurrieren in diesem Zeitraum.
  </div>
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>
      <th style="color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 14px 8px 0;text-align:left;">Richtung</th>
      <th style="color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 14px 8px 0;text-align:left;">Zeitfenster</th>
      <th style="color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 14px 8px 0;text-align:left;">WR 10J</th>
      <th style="color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0 14px 8px 0;text-align:left;">Robustheit</th>
      <th style="color:#475569;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:0;text-align:left;">Rating</th>
    </tr></thead>
    <tbody>{_warn_rows}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

    df_sym = saved_dfs.get(symbol_str)
    if df_sym is None or df_sym.empty:
        st.warning("Kein DataFrame verfügbar — Scanner nochmal starten.")
        return

    # ── Bessere Zeiten im Jahr für dieses Muster ─────────────────────────────
    _alt_cal_hold = _cur_exit_doy - _cur_entry_doy
    if _alt_cal_hold > 0:
        _alt_dir_str = richtung.lower()
        _alt_wr_key  = f"WR {lookback}J %"
        _alt_key = f"muster_alt_scan_{symbol_str}_{_cur_entry_doy}_{_cur_exit_doy}_{_alt_dir_str}_{lookback}"
        if _alt_key not in st.session_state:
            with st.spinner("Suche bessere Zeitfenster im Jahresverlauf …"):
                st.session_state[_alt_key] = scan_seasonality_patterns(
                    df_sym, lookback_years=lookback, min_winrate=0.0,
                    holding_periods=[_alt_cal_hold], directions=[_alt_dir_str],
                )
        _alt_scan = st.session_state[_alt_key]

        with st.expander(
            "🔍 Andere Zeiten im Jahr für dieses Muster (gleiche Haltedauer, gleiche Richtung)",
            expanded=False,
        ):
            _cur_wr = row.get(_alt_wr_key, float("nan"))
            if _alt_scan is None or _alt_scan.empty or _alt_wr_key not in _alt_scan.columns:
                st.info("Keine ausreichenden Daten für einen Jahresvergleich gefunden.")
            else:
                _cand = _alt_scan[_alt_scan["Richtung"] == richtung].copy()
                _cand = _cand[(_cand["_entry_doy"] - _cur_entry_doy).abs() > 10]
                _cand = _cand[_cand[_alt_wr_key].notna()]
                if pd.notna(_cur_wr):
                    _cand = _cand[_cand[_alt_wr_key] > _cur_wr]
                _cand = _cand.sort_values(_alt_wr_key, ascending=False).head(5)
                if _cand.empty:
                    st.success(
                        f"Keine besseren Zeitfenster gefunden — {row['Entry']} → {row['Exit']} "
                        f"scheint für {symbol_str} {richtung} im Jahresvergleich bereits stark zu sein."
                    )
                else:
                    _cur_wr_str = f"{_cur_wr:.1f}%" if pd.notna(_cur_wr) else "—"
                    st.caption(
                        f"Aktuelles Fenster: {_alt_wr_key.strip()} {_cur_wr_str} · gesucht: gleiche Haltedauer "
                        f"({_alt_cal_hold} Kalendertage), andere Jahreszeit, höhere Winrate."
                    )
                    for _ai, _arow in _cand.iterrows():
                        _a_c1, _a_c2 = st.columns([6, 1])
                        with _a_c1:
                            st.markdown(
                                f"📅 **{_arow['Entry']} → {_arow['Exit']}** · {_alt_wr_key.strip()}: "
                                f"**{_arow[_alt_wr_key]:.1f}%** (statt {_cur_wr_str}) · "
                                f"Ø Profit {_arow['Ø Profit %']:+.2f}% · {_arow['Robustheit']} · "
                                f"{'⭐' * int(_arow['⭐ Rating'])}"
                            )
                        with _a_c2:
                            if st.button("→ Laden", key=f"alt_load_{_ai}_{_alt_key}"):
                                _new_detail = {
                                    "row": _arow.to_dict(),
                                    "symbol": symbol_str,
                                    "lookback": lookback,
                                    "from_analyse": _from_analyse,
                                }
                                st.session_state["muster_detail"] = _new_detail
                                if _from_analyse:
                                    # render_muster_analyse() seedet "muster_detail" bei jedem Rerun
                                    # aus diesem Key neu — muss mit aktualisiert werden.
                                    st.session_state["muster_analyse_detail"] = {
                                        "detail": _new_detail,
                                        "dfs": saved_dfs,
                                    }
                                st.rerun()

    _month_map_d = {
        "Jan": 1, "Feb": 2, "Mär": 3, "Mar": 3, "Apr": 4, "Mai": 5, "May": 5,
        "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Okt": 10, "Oct": 10,
        "Nov": 11, "Dez": 12, "Dec": 12,
    }
    dir_str = richtung.lower()
    # DOY-basiert — exakt gleiche Logik wie der Scanner
    _entry_doy = int(row.get("_entry_doy", 0))
    _exit_doy  = int(row.get("_exit_doy",  0))
    _max_yr = int(df_sym.index.year.max())
    end_y = _max_yr - 1 if _max_yr >= pd.Timestamp.now().year else _max_yr
    start_y = end_y - 20 + 1

    # year_data für DOY-Lookup aufbauen
    _det_year_data: dict = {}
    for _yr, _grp in df_sym.groupby(df_sym.index.year):
        _yr_doys = _grp.index.dayofyear.values.astype(int)
        _sidx = np.argsort(_yr_doys)
        _det_year_data[int(_yr)] = {
            "doys":   _yr_doys[_sidx],
            "closes": _grp["close"].values.astype(float)[_sidx],
            "highs":  _grp["high"].values.astype(float)[_sidx],
            "lows":   _grp["low"].values.astype(float)[_sidx],
            "dates":  _grp.index[_sidx],
        }

    trade_rows = []
    for yr in range(start_y, end_y + 1):
        _yd = _det_year_data.get(yr)
        if _yd is None:
            continue
        try:
            _yr_doys = _yd["doys"]
            ei = int(np.searchsorted(_yr_doys, _entry_doy))
            xi = int(np.searchsorted(_yr_doys, _exit_doy))
            if ei >= len(_yr_doys) or xi >= len(_yr_doys) or xi <= ei:
                continue
            entry_date = _yd["dates"][ei]
            exit_date  = _yd["dates"][xi]
            ep = _yd["closes"][ei]
            xp = _yd["closes"][xi]
            # Für DD/MFE: ab dem Tag NACH Entry-Close (Entry-Wick vor Close zählt nicht)
            period_dd = df_sym[(df_sym.index > entry_date) & (df_sym.index <= exit_date)]
            if dir_str == "long":
                ret_pct = (xp - ep) / ep * 100
                dd_pct  = (float(period_dd["low"].min())  - ep) / ep * 100 if not period_dd.empty else 0.0
                mfe_pct = (float(period_dd["high"].max()) - ep) / ep * 100 if not period_dd.empty else 0.0
            else:
                ret_pct = (ep - xp) / ep * 100
                dd_pct  = (ep - float(period_dd["high"].max())) / ep * 100 if not period_dd.empty else 0.0
                mfe_pct = (ep - float(period_dd["low"].min()))  / ep * 100 if not period_dd.empty else 0.0
            trade_rows.append({
                "Jahr": yr,
                "Entry Datum": entry_date.strftime("%d.%m.%Y"),
                "Exit Datum":  exit_date.strftime("%d.%m.%Y"),
                "Return %": round(ret_pct, 2),
                "Max DD %": round(dd_pct, 2),
                "Max MFE %": round(mfe_pct, 2),
                "W/L": "✅ Win" if ret_pct > 0 else "❌ Loss",
                "_entry_ts": entry_date,
                "_entry_price": ep,
            })
        except Exception:
            continue

    if not trade_rows:
        st.warning("Keine Trades rekonstruierbar.")
        return

    tdf = pd.DataFrame(trade_rows).sort_values("Jahr").reset_index(drop=True)
    tdf_5  = tdf[tdf["Jahr"] >= end_y -  5 + 1]
    tdf_10 = tdf[tdf["Jahr"] >= end_y - 10 + 1]
    tdf_15 = tdf[tdf["Jahr"] >= end_y - 15 + 1]
    tdf_20 = tdf

    wins_10 = (tdf_10["Return %"] > 0).sum()
    wins_20 = (tdf_20["Return %"] > 0).sum()
    avg_dd_10 = tdf_10["Max DD %"].mean()
    max_dd_10 = tdf_10["Max DD %"].min()
    avg_dd_20 = tdf_20["Max DD %"].mean()
    max_dd_20 = tdf_20["Max DD %"].min()

    # DD-Statistik Banner
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
          <div style="background:#0a1220;border:1px solid rgba(248,113,113,.25);border-radius:8px;padding:14px 18px;">
            <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Ø DD — 10J</div>
            <div style="color:#f87171;font-size:1.1rem;font-weight:700;">{avg_dd_10:.2f}%</div>
            <div style="color:#6b7fa3;font-size:.8rem;">aus {len(tdf_10)} Jahren · {wins_10}W / {len(tdf_10)-wins_10}L</div>
          </div>
          <div style="background:#0a1220;border:1px solid rgba(248,113,113,.25);border-radius:8px;padding:14px 18px;">
            <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Max DD — 10J</div>
            <div style="color:#f87171;font-size:1.1rem;font-weight:700;">{max_dd_10:.2f}%</div>
            <div style="color:#6b7fa3;font-size:.8rem;">schlechtester intraday Rückgang</div>
          </div>
          <div style="background:#0a1220;border:1px solid rgba(167,139,250,.25);border-radius:8px;padding:14px 18px;">
            <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Ø DD — 20J</div>
            <div style="color:#a78bfa;font-size:1.1rem;font-weight:700;">{avg_dd_20:.2f}%</div>
            <div style="color:#6b7fa3;font-size:.8rem;">aus {len(tdf_20)} Jahren · {wins_20}W / {len(tdf_20)-wins_20}L</div>
          </div>
          <div style="background:#0a1220;border:1px solid rgba(167,139,250,.25);border-radius:8px;padding:14px 18px;">
            <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Max DD — 20J</div>
            <div style="color:#a78bfa;font-size:1.1rem;font-weight:700;">{max_dd_20:.2f}%</div>
            <div style="color:#6b7fa3;font-size:.8rem;">schlechtester intraday Rückgang</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Echtheits-Check ──────────────────────────────────────────────────────
    st.markdown("### 🔍 Echtheits-Check: Netto-Edge nach Kosten")
    _ec_c1, _ec_c2, _ec_c3, _ec_c4 = st.columns(4)
    with _ec_c1:
        _ec_comm = st.number_input("Commission %", 0.0, 2.0, 0.05, step=0.01, key=f"ec_comm_{symbol_str}_{row['Entry']}")
    with _ec_c2:
        _ec_slip = st.number_input("Slippage %", 0.0, 2.0, 0.02, step=0.01, key=f"ec_slip_{symbol_str}_{row['Entry']}")
    with _ec_c3:
        _perc_pct = st.slider("Perzentil-Stop %", 70, 95, 85, step=5, key=f"ec_perc_{symbol_str}_{row['Entry']}")
    with _ec_c4:
        _stddev_k = st.slider("StdDev-Faktor k", 1.0, 3.0, 1.75, step=0.25, key=f"ec_k_{symbol_str}_{row['Entry']}")
    _atr_mult = st.slider("ATR-Multiplikator", 1.0, 3.0, 1.75, step=0.25, key=f"ec_atr_{symbol_str}_{row['Entry']}")
    _kostenpuffer = 2.0 * (_ec_slip + _ec_comm)
    tdf["Netto Return %"] = (tdf["Return %"] - _kostenpuffer).round(2)

    def _fenster_data(sub):
        if len(sub) < 2: return None
        avg_net = (sub["Return %"] - _kostenpuffer).mean()
        avg_dd  = abs(sub["Max DD %"].mean())
        ratio   = avg_net / avg_dd if avg_dd > 0 else float("inf")
        return {"avg_net": avg_net, "ratio": ratio, "wins": (sub["Return %"] > 0).sum(), "n": len(sub)}

    def _ampel_html(label, d):
        if d is None:
            return (f'<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:8px;padding:14px 18px;">'
                    f'<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Netto-Edge {label}</div>'
                    f'<div style="color:#475569;">—</div></div>')
        if d["ratio"] >= 1.5 and d["avg_net"] > 0:   sym,farbe,txt="🟢","#4ade80","Robust positiv"
        elif d["ratio"] >= 0.8 and d["avg_net"] > 0: sym,farbe,txt="🟡","#f0c040","Knapp / Break-even"
        else:                                          sym,farbe,txt="🔴","#f87171","Verlierer nach Kosten"
        return (f'<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:8px;padding:14px 18px;">'
                f'<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Netto-Edge {label}</div>'
                f'<div style="color:{farbe};font-size:1.05rem;font-weight:700;margin-bottom:3px;">{sym} {txt}</div>'
                f'<div style="color:#9fb0c7;font-size:.8rem;">Ø netto {d["avg_net"]:+.2f}% · Ratio {d["ratio"]:.2f} · {d["wins"]}W/{d["n"]-d["wins"]}L</div></div>')

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">'
        + "".join([_ampel_html("5J",_fenster_data(tdf_5)),_ampel_html("10J",_fenster_data(tdf_10)),
                   _ampel_html("15J",_fenster_data(tdf_15)),_ampel_html("20J",_fenster_data(tdf_20))])
        + '</div>', unsafe_allow_html=True)

    # ── Edge-Validierung (statistische Handelbarkeits-Prüfung) ───────────────
    st.markdown("### 📐 Edge-Validierung")
    _edge_trades = pd.DataFrame({"pnl": (tdf["Netto Return %"] / 100.0).tolist()})
    _edge_result = evaluate_edge(_edge_trades, min_trades=200, alpha=0.05, min_sharpe_oos=1.0)

    _status_style = {
        "handelbar":       ("#4ade80", "🟢 Handelbar"),
        "grenzwertig":     ("#f0c040", "🟡 Grenzwertig"),
        "nicht handelbar": ("#f87171", "🔴 Nicht handelbar"),
    }
    _ev_farbe, _ev_label = _status_style[_edge_result["status"]]
    _ev_p = _edge_result["p_value"]
    _ev_sharpe = _edge_result["sharpe_oos"]
    _ev_wlo, _ev_whi = _edge_result["wilson_ci"]
    _ev_p_str = "n/a" if math.isnan(_ev_p) else f"{_ev_p:.4f}"
    _ev_sharpe_str = "n/a" if math.isnan(_ev_sharpe) else f"{_ev_sharpe:.2f}"

    st.markdown(
        f"""<div style="background:#0a1220;border:1px solid {_ev_farbe}55;border-radius:8px;
        padding:14px 18px;margin-bottom:14px;">
          <span style="background:{_ev_farbe}22;border:1px solid {_ev_farbe}55;border-radius:6px;
            padding:4px 12px;color:{_ev_farbe};font-weight:800;font-size:1rem;">{_ev_label}</span>
          <span style="color:#9fb0c7;font-size:.85rem;margin-left:12px;">
            n={_edge_result['n_trades']} · p={_ev_p_str}
            · Sharpe OOS={_ev_sharpe_str}
            · Wilson-CI Winrate=[{_ev_wlo*100:.1f}%, {_ev_whi*100:.1f}%]
          </span>
        </div>""",
        unsafe_allow_html=True,
    )
    if _edge_result["reasons"]:
        st.caption("⚠️ " + " · ".join(_edge_result["reasons"]))

    if _edge_result["status"] != "nicht handelbar":
        _kelly_balance = st.number_input(
            "Kontostand für Sizing-Empfehlung (€)", 100.0, 10_000_000.0, 10_000.0, step=100.0,
            key=f"ev_balance_{symbol_str}_{row['Entry']}",
        )
        _net_returns = (tdf["Netto Return %"] / 100.0)
        _wins = _net_returns[_net_returns > 0]
        _losses = _net_returns[_net_returns <= 0]
        _win_rate = len(_wins) / len(_net_returns) if len(_net_returns) > 0 else 0.0
        _avg_win = float(_wins.mean()) if len(_wins) > 0 else 0.0
        _avg_loss = float(abs(_losses.mean())) if len(_losses) > 0 else 0.0
        _rr_ratio = _avg_win / _avg_loss if _avg_loss > 0 else 0.0

        if _rr_ratio > 0:
            _kelly = kelly_position_size(
                win_rate=_win_rate,
                reward_risk_ratio=_rr_ratio,
                account_balance=_kelly_balance,
                kelly_fraction=0.25,
                max_risk_pct=0.01,
                n_trades_used=_edge_result["n_trades"],
                min_trades=200,
            )
            _risk_amount = _kelly["risk_amount"]
            _lots = _risk_amount / (_kostenpuffer * _kelly_balance) if _kostenpuffer > 0 else float("nan")
            st.info(
                f"**Empfohlene Positionsgröße:** {_kelly['risk_pct_used']*100:.2f}% Risiko "
                f"bzw. {_risk_amount:.2f}€ bei Kontostand {_kelly_balance:,.0f}€ "
                f"(Quarter-Kelly{', gecappt auf PropFirm-Limit 1%' if _kelly['capped'] else ''})."
            )
            if _kelly["warning"]:
                st.warning(_kelly["warning"])
        else:
            st.caption("Kelly-Sizing nicht berechenbar (kein Verlust-Trade in der Stichprobe).")

    # ── Stop-Loss-Empfehlung ─────────────────────────────────────────────────
    st.markdown("### 🛡️ Stop-Loss-Empfehlung")
    _dd_abs  = tdf["Max DD %"].abs()
    _sl_perc = float(np.percentile(_dd_abs, _perc_pct)) + _kostenpuffer
    _sl_std  = float(_dd_abs.mean() + _stddev_k * _dd_abs.std(ddof=1)) + _kostenpuffer
    _atr_pct_vals = []
    try:
        _h=df_sym["high"].to_numpy();_l=df_sym["low"].to_numpy();_c=df_sym["close"].to_numpy()
        _n=len(_c);_tr=np.zeros(_n);_tr[0]=_h[0]-_l[0]
        for _i in range(1,_n): _tr[_i]=max(_h[_i]-_l[_i],abs(_h[_i]-_c[_i-1]),abs(_l[_i]-_c[_i-1]))
        _aw=np.full(_n,np.nan)
        if _n>=14:
            _aw[13]=_tr[:14].mean()
            for _i in range(14,_n): _aw[_i]=(_aw[_i-1]*13+_tr[_i])/14
        _as=pd.Series(_aw,index=df_sym.index)
        for _,_r in tdf.iterrows():
            _ets=_r["_entry_ts"];_ep=_r["_entry_price"]
            if _ets in _as.index and not np.isnan(_as[_ets]) and _ep>0:
                _atr_pct_vals.append(_as[_ets]/_ep*100)
    except Exception: pass
    _atr_ok=len(_atr_pct_vals)>0
    _sl_atr=float(np.mean(_atr_pct_vals))*_atr_mult+_kostenpuffer if _atr_ok else 0.0
    _methoden={"Perzentil":_sl_perc,"Avg+StdDev":_sl_std}
    if _atr_ok: _methoden["ATR"]=_sl_atr
    _empf=max(_methoden,key=_methoden.get)
    if _empf=="ATR": _begr=f"ATR-Stop am größten (Ø {float(np.mean(_atr_pct_vals)):.2f}% × {_atr_mult}×) — Volatilität übersteigt die hist. DD-Quantile."
    elif _empf=="Avg+StdDev": _begr=f"Avg+StdDev-Stop am größten — Ausreißer-DD {_dd_abs.max():.2f}% hebt σ={_dd_abs.std(ddof=1):.2f}% stark an."
    else: _begr=f"Perzentil-Stop am größten — {_perc_pct}. Perzentil ({float(np.percentile(_dd_abs,_perc_pct)):.2f}%) übertrifft andere Methoden."

    def _sl_card(name,val,empf):
        b="border:2px solid #4ade80;" if empf else "border:1px solid rgba(148,163,184,.15);"
        tag=('<div style="display:inline-block;background:#14532d;color:#4ade80;font-size:.7rem;font-weight:700;border-radius:4px;padding:2px 8px;margin-bottom:6px;">✅ Empfohlen</div>' if empf else "")
        fc="#4ade80" if empf else "#9fb0c7"
        return (f'<div style="background:#0a1220;{b}border-radius:8px;padding:16px 20px;">{tag}'
                f'<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">{name}</div>'
                f'<div style="color:{fc};font-size:1.2rem;font-weight:700;">{val:.2f}%</div>'
                f'<div style="color:#475569;font-size:.75rem;margin-top:3px;">inkl. Kostenpuffer {_kostenpuffer:.2f}%</div></div>')

    _cards=[_sl_card("Perzentil",_sl_perc,_empf=="Perzentil"),_sl_card("Avg + StdDev",_sl_std,_empf=="Avg+StdDev")]
    if _atr_ok: _cards.append(_sl_card("ATR",_sl_atr,_empf=="ATR"))
    else: _cards.append('<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:8px;padding:16px 20px;"><div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">ATR</div><div style="color:#475569;">keine ATR-Daten verfügbar</div></div>')
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat({len(_cards)},1fr);gap:14px;margin-bottom:10px;">{"".join(_cards)}</div>'
                f'<div style="color:#6b7fa3;font-size:.82rem;margin-bottom:22px;">💡 {_begr}</div>',unsafe_allow_html=True)

    # Tabelle — full width
    st.markdown("### 📋 Per-Jahr Ergebnisse")
    def _color_ret(v):
        return "color:#4ade80;font-weight:600" if v > 0 else "color:#f87171;font-weight:600"
    def _color_dd(v):
        return "color:#f87171" if v < 0 else "color:#9fb0c7"
    _tdf_show = tdf.drop(columns=["_entry_ts","_entry_price"],errors="ignore")
    st.dataframe(
        _tdf_show.style
        .map(_color_ret, subset=["Return %","Netto Return %","Max MFE %"])
        .map(_color_dd,  subset=["Max DD %"])
        .format({"Return %":"{:+.2f}%","Netto Return %":"{:+.2f}%","Max DD %":"{:+.2f}%","Max MFE %":"{:+.2f}%"}),
        use_container_width=True,
        height=min(40*len(tdf)+42,700),
    )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Charts — unter der Tabelle
    cumret = (1 + tdf["Return %"] / 100).cumprod() * 100 - 100
    end_color = "#4ade80" if cumret.iloc[-1] >= 0 else "#f87171"

    n_years = len(tdf)
    # Bei vielen Jahren jeden 2. Tick zeigen damit Labels lesbar bleiben
    tick_vals = [str(y) for y in tdf["Jahr"]]
    tick_text = [str(y) if i % (2 if n_years > 12 else 1) == 0 else "" for i, y in enumerate(tdf["Jahr"])]

    _chart_cfg = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,18,32,1)",
        font=dict(color="#9fb0c7", size=13),
        margin=dict(l=10, r=10, t=10, b=60),
        showlegend=False,
        xaxis=dict(
            gridcolor="rgba(148,163,184,.07)",
            tickvals=tick_vals, ticktext=tick_text,
            tickangle=-45, tickfont=dict(size=12),
        ),
        yaxis=dict(gridcolor="rgba(148,163,184,.07)", ticksuffix="%", tickfont=dict(size=12)),
    )

    st.markdown("### 📈 Equity Kurve (kumuliert)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tdf["Jahr"].astype(str), y=cumret.round(2),
        mode="lines+markers",
        line=dict(color=end_color, width=2.5),
        marker=dict(size=8, color=end_color),
        fill="tozeroy",
        fillcolor="rgba(74,222,128,0.08)" if cumret.iloc[-1] >= 0 else "rgba(248,113,113,0.08)",
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,.3)", line_width=1)
    fig.update_layout(height=380, **_chart_cfg)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Jahresrenditen")
    st.markdown(
        "<div style='color:#6b7fa3;font-size:.82rem;margin-bottom:12px;'>"
        "Ein Balken pro Handelsjahr in diesem Saisonfenster. &nbsp;"
        "<span style='color:#00e676;font-weight:700;'>■</span> Gewinner &nbsp;"
        "<span style='color:#f87171;font-weight:700;'>■</span> Verlierer</div>",
        unsafe_allow_html=True,
    )
    _bar_ret = tdf["Return %"].round(2).tolist()
    _bar_yrs = tdf["Jahr"].astype(str).tolist()
    _max_abs = max(abs(v) for v in _bar_ret) if _bar_ret else 5
    # Nulllinie bei 40% von links → positive Seite = 60%, negative = 40%
    _zero_pct = 40
    _pos_range = 60   # % of track width for max positive value
    _neg_range = 40   # % of track width for max negative value

    _rows_html = ""
    for _yr, _val in zip(_bar_yrs, _bar_ret):
        _clr = "#00e676" if _val >= 0 else "#f87171"
        _label = f"{_val:+.2f}%"
        if _val >= 0:
            _bar_left = _zero_pct
            _bar_w    = (_val / _max_abs) * _pos_range
        else:
            _bar_w    = (abs(_val) / _max_abs) * _neg_range
            _bar_left = _zero_pct - _bar_w
        _rows_html += (
            f"<div style='display:flex;align-items:center;margin-bottom:7px;gap:14px;'>"
            f"  <div style='width:38px;color:#94a3b8;font-size:13px;flex-shrink:0;text-align:right;'>{_yr}</div>"
            f"  <div style='flex:1;background:#1e2a3a;border-radius:3px;height:24px;position:relative;'>"
            f"    <div style='position:absolute;left:{_bar_left:.2f}%;width:{_bar_w:.2f}%;height:100%;"
            f"background:{_clr};border-radius:2px;'></div>"
            f"    <div style='position:absolute;left:{_zero_pct}%;width:1px;height:100%;"
            f"background:rgba(148,163,184,.35);'></div>"
            f"  </div>"
            f"  <div style='width:62px;color:{_clr};font-size:13px;font-weight:700;"
            f"text-align:right;flex-shrink:0;font-family:monospace;'>{_label}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='background:#111827;border-radius:10px;padding:20px 16px 14px 16px;'>"
        f"{_rows_html}</div>",
        unsafe_allow_html=True,
    )

    # ── Profit-Qualität: stark vs. schwach (gesamt / 10J / 5J) ────────────────
    _threshold   = 0.50
    _atr_pct_val = float(row.get("Ø ATR %", 0) or 0)
    _all_years   = sorted(tdf["Jahr"].unique())
    _end_yr_det  = max(_all_years) if _all_years else 2025

    def _quality_block(rets: list, label: str) -> str:
        n_total  = len(rets)
        if n_total == 0:
            return ""
        n_wins    = sum(1 for v in rets if v > 0)
        n_stark   = sum(1 for v in rets if v >= _threshold)
        n_schwach = sum(1 for v in rets if 0 <= v < _threshold)
        n_neg     = sum(1 for v in rets if v < 0)
        avg_p     = sum(rets) / n_total
        ok        = avg_p >= _atr_pct_val * 0.25 if _atr_pct_val > 0 else True
        atr_clr   = "#4ade80" if ok else "#f87171"
        # Wilson-CI für Winrate
        ci_lo, ci_hi = wilson_ci(n_wins, n_total)
        ci_w = ci_hi - ci_lo
        ci_clr = "#4ade80" if ci_w < 0.28 else ("#fbbf24" if ci_w < 0.42 else "#f87171")
        ci_label = "eng ✓" if ci_w < 0.28 else ("mittel" if ci_w < 0.42 else "breit ⚠")
        return f"""
        <div style="background:#070f1a;border-radius:6px;padding:8px 10px;margin-bottom:4px;">
          <div style="color:#94a3b8;font-size:.68rem;font-weight:700;text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:6px;">{label}</div>
          <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
            <div style="background:#0d1828;border:1px solid #4ade8033;border-radius:6px;padding:7px 10px;">
              <div style="color:#6b7fa3;font-size:.65rem;text-transform:uppercase;">Stark ≥0.50%</div>
              <div style="color:#4ade80;font-size:1.1rem;font-weight:800;">{n_stark}<span style="color:#6b7fa3;font-size:.8rem;"> / {n_total}J</span></div>
            </div>
            <div style="background:#0d1828;border:1px solid #fbbf2433;border-radius:6px;padding:7px 10px;">
              <div style="color:#6b7fa3;font-size:.65rem;text-transform:uppercase;">Schwach &lt;0.50%</div>
              <div style="color:#fbbf24;font-size:1.1rem;font-weight:800;">{n_schwach}<span style="color:#6b7fa3;font-size:.8rem;"> / {n_total}J</span></div>
            </div>
            <div style="background:#0d1828;border:1px solid #f8717133;border-radius:6px;padding:7px 10px;">
              <div style="color:#6b7fa3;font-size:.65rem;text-transform:uppercase;">Verlierer</div>
              <div style="color:#f87171;font-size:1.1rem;font-weight:800;">{n_neg}<span style="color:#6b7fa3;font-size:.8rem;"> / {n_total}J</span></div>
            </div>
            <div style="background:#0d1828;border:1px solid {atr_clr}33;border-radius:6px;padding:7px 10px;">
              <div style="color:#6b7fa3;font-size:.65rem;text-transform:uppercase;">Ø Profit vs ATR</div>
              <div style="color:{atr_clr};font-size:1.1rem;font-weight:800;">{avg_p:+.2f}%<span style="color:#6b7fa3;font-size:.72rem;"> / {_atr_pct_val:.2f}%</span></div>
            </div>
            <div style="background:#0d1828;border:1px solid {ci_clr}33;border-radius:6px;padding:7px 10px;">
              <div style="color:#6b7fa3;font-size:.65rem;text-transform:uppercase;">Wilson-CI 95%</div>
              <div style="color:{ci_clr};font-size:.95rem;font-weight:800;">{ci_lo*100:.0f}–{ci_hi*100:.0f}%</div>
              <div style="color:{ci_clr};font-size:.65rem;">{ci_label}</div>
            </div>
          </div>
        </div>"""

    _rets_all = _bar_ret
    _rets_10  = [r for yr, r in zip(_bar_yrs, _bar_ret) if int(yr) >= _end_yr_det - 9]
    _rets_5   = [r for yr, r in zip(_bar_yrs, _bar_ret) if int(yr) >= _end_yr_det - 4]

    _quality_html = (
        _quality_block(_rets_all, f"Gesamt ({len(_rets_all)}J)") +
        _quality_block(_rets_10,  f"Letzte 10 Jahre") +
        _quality_block(_rets_5,   f"Letzte 5 Jahre")
    )
    st.markdown(
        f"<div style='margin:14px 0 8px 0;'>{_quality_html}</div>",
        unsafe_allow_html=True,
    )

    # ── ATR-Kurve ──────────────────────────────────────────────────────────────
    st.markdown("### 📉 ATR(14) Verlauf — Tagesvolatilität im Musterfenster")
    st.markdown(
        "<div style='color:#6b7fa3;font-size:.82rem;margin-bottom:12px;'>"
        "Durchschnittliche Tagesvolatilität (High–Low) zum Einstiegszeitpunkt, pro Jahr. "
        "Hohe ATR-Werte = volatileres Umfeld = größere Schwankungen während des Musters.</div>",
        unsafe_allow_html=True,
    )

    # ATR(14) für jedes Entry-Jahr berechnen
    _atr_rows = []
    for _yr_atr in sorted(tdf["Jahr"].unique()):
        _entry_str = tdf.loc[tdf["Jahr"] == _yr_atr, "Entry Datum"].values[0]
        try:
            _ed = pd.to_datetime(_entry_str, format="%d.%m.%Y")
            # ATR-Fenster: 30 Tage vor Entry
            _window = df_sym[(_df_end := _ed) - pd.Timedelta(days=45):_ed]
            if len(_window) >= 10:
                _tr = pd.concat([
                    _window["high"] - _window["low"],
                    (_window["high"] - _window["close"].shift(1)).abs(),
                    (_window["low"]  - _window["close"].shift(1)).abs(),
                ], axis=1).max(axis=1)
                _atr14 = _tr.ewm(span=14, adjust=False).mean().iloc[-1]
                _ep = float(df_sym.loc[_ed, "close"]) if _ed in df_sym.index else float(_window["close"].iloc[-1])
                _atr_rows.append({"Jahr": str(_yr_atr), "ATR %": round(_atr14 / _ep * 100, 4)})
        except Exception:
            pass

    if _atr_rows:
        _atr_df = pd.DataFrame(_atr_rows)
        _atr_avg = _atr_df["ATR %"].mean()
        _atr_colors = ["#fb923c" if v > _atr_avg * 1.3 else "#60a5fa" for v in _atr_df["ATR %"]]

        fig_atr = go.Figure()
        # Fläche unter der Kurve
        fig_atr.add_trace(go.Scatter(
            x=_atr_df["Jahr"], y=_atr_df["ATR %"],
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2.5),
            marker=dict(size=9, color=_atr_colors, line=dict(width=1.5, color="#0a1220")),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)",
            hovertemplate="<b>%{x}</b><br>ATR(14): %{y:.4f}%<extra></extra>",
            name="ATR(14) %",
        ))
        # Durchschnittslinie
        fig_atr.add_hline(
            y=_atr_avg,
            line_dash="dot", line_color="rgba(148,163,184,.45)", line_width=1.5,
            annotation_text=f"Ø {_atr_avg:.4f}%",
            annotation_position="right",
            annotation_font=dict(color="#6b7fa3", size=11),
        )
        # Hochvolatile Jahre markieren (>130% des Durchschnitts)
        for _, _r in _atr_df[_atr_df["ATR %"] > _atr_avg * 1.3].iterrows():
            fig_atr.add_annotation(
                x=_r["Jahr"], y=_r["ATR %"],
                text="⚠️", showarrow=False, yshift=14,
                font=dict(size=13),
            )
        fig_atr.update_layout(
            height=300,
            xaxis=dict(
                tickfont=dict(size=12, color="#94a3b8"),
                gridcolor="rgba(148,163,184,.06)",
                showgrid=True,
            ),
            yaxis=dict(
                ticksuffix="%", tickfont=dict(size=11, color="#6b7fa3"),
                gridcolor="rgba(148,163,184,.07)",
            ),
            margin=dict(l=10, r=80, t=10, b=30),
            plot_bgcolor="#0a1220",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            showlegend=False,
        )
        st.plotly_chart(fig_atr, use_container_width=True)

    # ATR-Erklärung
    _atr_disp = row.get("Ø ATR %", np.nan)
    _atr_num  = float(_atr_disp) if pd.notna(_atr_disp) else None
    if _atr_num is not None:
        if _atr_num < 0.4:
            _atr_lvl = "niedrig"; _atr_clr = "#60a5fa"; _atr_emoji = "🔵"
            _atr_meaning = "Das Instrument bewegt sich täglich wenig. Saisonale Muster liefern hier oft saubere, ruhige Verläufe — aber die absoluten Gewinne je Trade sind begrenzt."
        elif _atr_num < 0.8:
            _atr_lvl = "moderat"; _atr_clr = "#4ade80"; _atr_emoji = "🟢"
            _atr_meaning = "Gute Balance zwischen Beweglichkeit und Kontrolle. Saisonale Muster entfalten hier ihr volles Potenzial — ausreichend Profit-Spielraum bei überschaubarem Rauschen."
        elif _atr_num < 1.4:
            _atr_lvl = "hoch"; _atr_clr = "#fbbf24"; _atr_emoji = "🟡"
            _atr_meaning = "Das Instrument schwankt stark. Gewinne können größer sein, aber auch Fehlsignale und Intraday-Rauschen nehmen zu. Stop-Loss muss entsprechend weiter gesetzt werden."
        else:
            _atr_lvl = "sehr hoch"; _atr_clr = "#f87171"; _atr_emoji = "🔴"
            _atr_meaning = "Extrem volatile Bewegungen — jede Kerze kann mehrere Prozent betragen. Saisonale Muster sind hier schwerer handelbar, da breite Stops und hohes Kapitalrisiko nötig sind."

        _atr_profit_ratio = abs(float(row.get("Ø Profit %", 0) or 0)) / _atr_num if _atr_num > 0 else 0
        if _atr_profit_ratio >= 0.6:
            _ratio_text = f"✅ Der Ø-Profit ({row.get('Ø Profit %', '—')}%) beträgt <strong>{_atr_profit_ratio:.1f}×</strong> des ATR — das Muster verdient seinen Volatilitätseinsatz."
            _ratio_clr  = "#4ade80"
        elif _atr_profit_ratio >= 0.4:
            _ratio_text = f"⚠️ Der Ø-Profit ({row.get('Ø Profit %', '—')}%) beträgt <strong>{_atr_profit_ratio:.1f}×</strong> des ATR — akzeptabel, aber Vorsicht bei Slippage."
            _ratio_clr  = "#fbbf24"
        else:
            _ratio_text = f"❌ Der Ø-Profit ({row.get('Ø Profit %', '—')}%) beträgt nur <strong>{_atr_profit_ratio:.1f}×</strong> des ATR — das Muster verdient kaum mehr als eine typische Tageskerze."
            _ratio_clr  = "#f87171"

        st.markdown(f"""
<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:10px;padding:20px 24px;margin-top:8px;">
  <div style="color:#94a3b8;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;">ATR(14) — Einordnung & Bewertung</div>
  <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:16px;">
    <div style="font-size:1.6rem;line-height:1;">{_atr_emoji}</div>
    <div>
      <div style="color:{_atr_clr};font-size:.95rem;font-weight:700;margin-bottom:4px;">
        Ø ATR(14) = {_atr_num:.3f}% — Volatilität <span style="text-transform:uppercase;">{_atr_lvl}</span>
      </div>
      <div style="color:#6b7fa3;font-size:.85rem;line-height:1.55;">{_atr_meaning}</div>
    </div>
  </div>
  <div style="background:#0d1828;border-radius:7px;padding:12px 16px;margin-bottom:14px;">
    <div style="color:{_ratio_clr};font-size:.88rem;line-height:1.5;">{_ratio_text}</div>
  </div>
  <div style="border-top:1px solid rgba(148,163,184,.08);padding-top:14px;">
    <div style="color:#475569;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">Allgemeine Richtwerte für saisonale Muster</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
      <div style="background:#0d1828;border:1px solid #60a5fa22;border-radius:6px;padding:10px 12px;">
        <div style="color:#60a5fa;font-size:.75rem;font-weight:700;">🔵 &lt; 0.4%</div>
        <div style="color:#475569;font-size:.72rem;margin-top:3px;">Niedrig — ruhige, planbare Bewegungen</div>
      </div>
      <div style="background:#0d1828;border:1px solid #4ade8022;border-radius:6px;padding:10px 12px;">
        <div style="color:#4ade80;font-size:.75rem;font-weight:700;">🟢 0.4 – 0.8%</div>
        <div style="color:#475569;font-size:.72rem;margin-top:3px;">Ideal — bestes Verhältnis Profit/Risiko</div>
      </div>
      <div style="background:#0d1828;border:1px solid #fbbf2422;border-radius:6px;padding:10px 12px;">
        <div style="color:#fbbf24;font-size:.75rem;font-weight:700;">🟡 0.8 – 1.4%</div>
        <div style="color:#475569;font-size:.72rem;margin-top:3px;">Hoch — weite Stops nötig, mehr Rauschen</div>
      </div>
      <div style="background:#0d1828;border:1px solid #f8717122;border-radius:6px;padding:10px 12px;">
        <div style="color:#f87171;font-size:.75rem;font-weight:700;">🔴 &gt; 1.4%</div>
        <div style="color:#475569;font-size:.72rem;margin-top:3px;">Sehr hoch — schwer handelbar, hohes Risiko</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_seasonality_muster() -> None:
    # Detail-Page wenn aktiv
    if "muster_detail" in st.session_state:
        _render_muster_detail()
        return

    st.markdown(
        """<div style="margin-bottom:20px;">
          <div style="color:#fff;font-size:1.6rem;font-weight:900;letter-spacing:.01em;margin-bottom:4px;">
            📅 Seasonality Muster Scanner
          </div>
          <div style="color:#6b7fa3;font-size:.95rem;">
            Scannt historische OHLC-Daten auf wiederkehrende Muster mit hoher Winrate · Pepperstone MT5 Daily
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col_ctrl, col_main = st.columns([1, 3], gap="medium")

    # Permanent CSV folder — auto-download from data branch if missing
    from pathlib import Path as _Path
    import requests as _requests
    _MT5_DIR = _Path(__file__).parent / "data" / "mt5"
    _GITHUB_RAW = "https://raw.githubusercontent.com/MazohFX/taco-strategy-lab/data"
    _ALL_SYMBOLS = [
        "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD","AUS200","CADJPY","CHFJPY",
        "EURAUD","EURCAD","EURGBP","EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD",
        "GBPJPY","GBPNZD","GBPUSD","GER40","JPN225","NZDCAD","NZDCHF","NZDJPY",
        "NZDUSD","UK100","US30","US500","USDCAD","USDCHF","USDJPY","XAGUSD","XAUUSD",
    ]
    if not _MT5_DIR.exists() or len(list(_MT5_DIR.glob("*.csv"))) < len(_ALL_SYMBOLS):
        _MT5_DIR.mkdir(parents=True, exist_ok=True)
        _dl_bar = st.progress(0, text="Lade CSV-Daten vom Server…")
        for _i, _sym in enumerate(_ALL_SYMBOLS):
            _fpath = _MT5_DIR / f"{_sym}.csv"
            if not _fpath.exists():
                try:
                    _r = _requests.get(f"{_GITHUB_RAW}/{_sym}.csv", timeout=15)
                    if _r.status_code == 200:
                        _fpath.write_bytes(_r.content)
                except Exception:
                    pass
            _dl_bar.progress((_i + 1) / len(_ALL_SYMBOLS), text=f"Lade {_sym}…")
        _dl_bar.empty()
    _available_symbols = sorted([f.stem.upper() for f in _MT5_DIR.glob("*.csv")]) if _MT5_DIR.exists() else []

    with col_ctrl:
        st.markdown("<div style='color:#94a3b8;font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;'>Einstellungen</div>", unsafe_allow_html=True)

        daten_modus = st.radio("Datenquelle", ["Repo (permanent)", "CSV Upload"], horizontal=True)

        if daten_modus == "Repo (permanent)":
            if _available_symbols:
                selected_symbols = st.multiselect(
                    "Symbole auswählen", _available_symbols, default=_available_symbols,
                    help="Alle Daten sind permanent im Repo gespeichert"
                )
            else:
                st.warning("Keine CSVs im data/mt5/ Ordner gefunden.")
                selected_symbols = []
            csv_files = []
        else:
            csv_files = st.file_uploader(
                "Pepperstone CSV (daily OHLC)", type=["csv"],
                help="Mehrere CSVs gleichzeitig möglich",
                accept_multiple_files=True,
            )
            selected_symbols = []

        import datetime as _dt2
        _cur_yr = _dt2.date.today().year
        _end_yr = _cur_yr - 1  # letztes vollständiges Jahr
        lookback = st.radio("Analysezeitraum (Filter gilt für)", [5, 10, 15, 20], format_func=lambda x: f"{x}J ({_end_yr-x+1}–{_end_yr})", horizontal=True, index=1)
        dir_choice = st.multiselect("Richtung", ["Long", "Short"], default=["Long", "Short"])
        min_wr = st.slider("Min. Winrate %", 60, 100, 70, step=5)
        hold_min = st.number_input("Musterlänge min (Kalendertage)", 1, 60, 5)
        hold_max = st.number_input("Musterlänge max (Kalendertage)", 1, 120, 20)
        hold_step = st.number_input("Schritt", 1, 10, 1)
        holding_periods = list(range(int(hold_min), int(hold_max) + 1, int(hold_step)))
        run_scan = st.button("🔍 Scanner starten", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("<div style='color:#94a3b8;font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;'>Walk-Forward Validierung</div>", unsafe_allow_html=True)
        _wfa_defaults = {5: (4, 2), 10: (10, 5), 15: (10, 8), 20: (12, 7)}
        _wfa_is_def, _wfa_folds_def = _wfa_defaults.get(lookback, (10, 5))
        wfa_min_is = st.number_input("Start IS-Fenster (Jahre)", min_value=5, max_value=20, value=_wfa_is_def,
                                      help="Erste N Jahre als In-Sample-Startfenster")
        wfa_min_folds = st.number_input("Min. Folds für ✅ Badge", min_value=2, max_value=20, value=_wfa_folds_def,
                                         help="Muster muss in mind. N Folds als IS-Kandidat erschienen sein")
        run_wfa = st.button("🔄 Walk-Forward validieren", use_container_width=True,
                            help="Rechenintensiv — kann mehrere Minuten dauern")

    with col_main:
        if daten_modus == "Repo (permanent)" and not selected_symbols:
            st.info("Bitte wähle mindestens ein Symbol aus und starte den Scanner.")
            return
        if daten_modus == "CSV Upload" and not csv_files:
            st.info("Bitte lade eine oder mehrere Pepperstone CSV-Dateien hoch und starte den Scanner.")
            return

        with st.expander("❓ Walk-Forward Validierung — was bedeuten die Einstellungen?", expanded=False):
            st.markdown(
                """
<div style="font-size:.88rem;color:#cbd5e1;line-height:1.8;">
<div style="color:#f0c040;font-weight:700;font-size:.95rem;margin-bottom:6px;">📖 Wie funktioniert Walk-Forward?</div>
Das Fenster rollt Jahr für Jahr vorwärts. Die ersten <b>N Jahre</b> sind In-Sample (IS) — dort wird das Muster gesucht.
Das jeweils nächste Jahr ist Out-of-Sample (OOS) — dort wird geprüft, ob das Muster auch dort funktioniert hat.
</div>
""", unsafe_allow_html=True)

            st.markdown("**🔁 Beispiel mit IS-Fenster = 10 (Datenbasis 2006–2025)**")
            st.markdown("""
| Fold | In-Sample — Training | OOS-Test Jahr |
|:----:|----------------------|:-------------:|
| 1 | 2006 – 2015 | 2016 |
| 2 | 2006 – 2016 | 2017 |
| 3 | 2006 – 2017 | 2018 |
| 4 | 2006 – 2018 | 2019 |
| 5 | 2006 – 2019 | 2020 |
| … | … | … |
| 9 | 2006 – 2023 | 2024 |
""")
            st.markdown(
                """
<div style="font-size:.88rem;color:#cbd5e1;line-height:1.8;margin-top:8px;">
<div style="color:#f0c040;font-weight:700;font-size:.95rem;margin-bottom:4px;">🏅 Min. Folds für ✅ Badge</div>
Ein <b>Fold</b> = ein einzelner Test-Durchlauf mit einem bestimmten IS-Zeitraum und einem OOS-Jahr.
Im Beispiel oben mit IS-Fenster 10 entstehen <b>9 Folds</b> (Fold 1 bis Fold 9).
<br>
Ein Muster bekommt nur dann <b>✅ OOS-validiert</b>, wenn es in mindestens N dieser Folds als IS-Kandidat aufgetaucht ist.
<br><i>Beispiel: Min. Folds = 5 → das Muster muss in mindestens 5 der 9 Test-Durchläufe gefunden worden sein.</i>
<br>Das filtert Zufallstreffer heraus — je höher der Wert, desto strenger der Filter.
<br><br>
<div style="color:#f0c040;font-weight:700;font-size:.95rem;margin-bottom:4px;">📊 Analysezeitraum vs. IS-Fenster</div>
Der <b>Analysezeitraum</b> (Radio-Button links) filtert nur, welche Muster <i>angezeigt</i> werden —
die WFA nutzt immer alle verfügbaren Daten der CSVs unabhängig davon.
<br>
<b style="color:#4ade80;">→ Tipp: Stelle links auf 20J</b>, damit nur Muster durchkommen, die über 20 Jahre konsistent waren.
Die WFA hat dann auch mehr Folds (~8–10 statt 4–5) → robustere Badges.
<br><br>
<div style="color:#f0c040;font-weight:700;font-size:.95rem;margin-bottom:4px;">🧹 Was filtert die WFA konkret heraus?</div>
Beim Scan werden zigtausende Tag/Richtung/Haltedauer-Kombinationen getestet (Multiple-Comparisons-Problem) —
bei so vielen Versuchen sieht im Schnitt immer irgendetwas wie ein "Muster" aus, obwohl es reiner Zufall ist.
Die WFA filtert genau diese Zufallstreffer heraus: Sie verlangt, dass ein Muster nicht nur einmal, sondern über
mehrere <b>unabhängige</b> rollierende Fenster (Folds) hinweg immer wieder als Kandidat erscheint.
<br>
<b style="color:#f87171;">Was die WFA NICHT erkennt:</b> echte strukturelle Brüche (z. B. eine veränderte Marktstruktur) —
sie prüft nur Konsistenz in der Vergangenheit, keine Garantie für die Zukunft.
<br><br>
<div style="color:#f0c040;font-weight:700;font-size:.95rem;margin-bottom:4px;">⚖️ 10 Jahre vs. 20 Jahre — ist die WFA bei einem 10J-Muster überhaupt aussagekräftig?</div>
Mit Vorsicht zu genießen. Ein saisonales Muster hat typischerweise nur <b>1 Trade pro Jahr</b>. Bei 10 Jahren
Lookback sind das nur 10 Datenpunkte — statistisch sehr klein. Die WFA nutzt zwar immer die komplette
CSV-Historie für ihre Folds, aber wenn das Muster nur über 10 Jahre gesucht wurde, hatte es auch nur 10 Chancen,
sich zu bestätigen, bevor es überhaupt als Kandidat zählt → hohes Risiko für einen reinen Zufallstreffer.
Bei 20J-Analyse verdoppelt sich sowohl die Stichprobe als auch die Anzahl unabhängiger Folds (~8–10 statt ~4–5) —
eine deutlich härtere Hürde.
<br>
<b style="color:#4ade80;">→ Praxis-Empfehlung:</b> 10J-Muster nur als Beobachtungsliste/Kandidaten behandeln, nicht direkt handeln.
Für reales PropFirm-Trading ausschließlich <b>20J-validierte Muster mit ≥ 7 Folds</b> verwenden.
<br><br>
<div style="color:#f0c040;font-weight:700;font-size:.95rem;margin-bottom:4px;">📐 Zusammenspiel mit der Edge-Validierung (Badge weiter unten)</div>
Die Edge-Validierung verlangt standardmäßig <code>min_trades=200</code>. Für saisonale Jahres-Muster
(max. ~20–25 Trades über 20–25 Jahre Historie) ist das strukturell unerreichbar — sie zeigt bei
Saisonalitäts-Mustern deshalb fast immer <b style="color:#f87171;">🔴 "Nicht handelbar"</b>, selbst wenn das
Muster WFA-validiert ist. Das ist kein Fehler, sondern ein ehrliches Signal: Mit ~20 Jahres-Trades lässt sich
keine Aussage auf 200-Trades-Niveau treffen. <b>Für saisonale Muster bleibt deshalb die WFA-Foldzahl die
primäre Validierungsmethode</b>, nicht der <code>min_trades</code>-Schwellenwert der allgemeinen Edge-Validierung.
</div>
""", unsafe_allow_html=True)

            st.markdown("**✅ Empfohlene Einstellungen**")
            st.markdown("""
| Ziel | Analysezeitraum (links) | IS-Fenster | Min. Folds |
|------|:-----------------------:|:----------:|:----------:|
| Maximale Robustheit | **20J** | **10–12** | **7** |
| Ausgewogen (Standard) | **10J** | **10** | **5** |
| Mehr Muster sehen | **10J** | **8** | **3** |
""")
            st.info("💡 Bei 20J Datenbasis entstehen ~8–10 Folds — mehr Folds = robusteres ✅ Badge.")

        if not run_scan and "muster_scan_result" not in st.session_state:
            st.info("Einstellungen wählen und 'Scanner starten' klicken.")
            return

        if run_scan:
            directions = [d.lower() for d in dir_choice]
            if not directions or not holding_periods:
                st.warning("Bitte Richtung und Halteperioden auswählen.")
                return

            # Build unified file list: repo files or uploads
            import io as _io
            file_entries: list[tuple[str, object]] = []
            if daten_modus == "Repo (permanent)":
                for sym in selected_symbols:
                    fpath = _MT5_DIR / f"{sym}.csv"
                    if fpath.exists():
                        file_entries.append((sym, fpath))
            else:
                for csv_file in csv_files:
                    sym = csv_file.name.replace(".csv", "").replace(".CSV", "").upper()
                    file_entries.append((sym, csv_file))

            all_results = []
            loaded_dfs: dict = {}
            errors = []
            progress = st.progress(0, text="Scanne…")
            for idx, (symbol, src) in enumerate(file_entries):
                progress.progress((idx + 1) / max(len(file_entries), 1), text=f"Scanne {symbol}…")
                try:
                    raw = pd.read_csv(src)
                    df_loaded = normalize_ohlc(raw)
                except Exception as e:
                    errors.append(f"{symbol}: {e}")
                    continue
                if df_loaded.empty:
                    errors.append(f"{symbol}: Keine gültigen OHLC-Daten")
                    continue
                available_years = df_loaded.index.year.max() - df_loaded.index.year.min() + 1
                if available_years < lookback:
                    errors.append(f"{symbol}: Nur {available_years} Jahre Daten")
                    continue
                res = scan_seasonality_patterns(
                    df_loaded,
                    lookback_years=lookback,
                    min_winrate=min_wr / 100,
                    holding_periods=holding_periods,
                    directions=directions,
                )
                if not res.empty:
                    res.insert(0, "Symbol", symbol)
                    all_results.append(res)
                loaded_dfs[symbol] = df_loaded

            progress.empty()
            if errors:
                st.warning("Fehler bei: " + " | ".join(errors))

            result = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
            if not result.empty:
                sort_col2 = f"WR {lookback}J %"
                if sort_col2 not in result.columns:
                    sort_col2 = result.columns[0]
                result = result.sort_values(sort_col2, ascending=False).reset_index(drop=True)

            n_src = len(selected_symbols) if daten_modus == "Repo (permanent)" else len(csv_files)
            st.session_state["muster_scan_result"] = result
            st.session_state["muster_dataframes"] = loaded_dfs
            st.session_state["muster_csv_name"] = f"{n_src} Symbole"

        # ── Walk-Forward Validierung ──────────────────────────────────────────
        if run_wfa:
            loaded_dfs_wfa = st.session_state.get("muster_dataframes", {})
            if not loaded_dfs_wfa:
                st.warning("Bitte zuerst den Scanner starten, um Daten zu laden.")
            else:
                from seasonality_wfa import run_seasonality_wfa, wfa_results_to_dataframe
                directions_wfa = [d.lower() for d in dir_choice]
                _wfa_bar = st.progress(0, text="Starte Walk-Forward Validierung…")
                def _wfa_progress(frac: float, txt: str) -> None:
                    _wfa_bar.progress(min(frac, 1.0), text=txt)
                wfa_res = run_seasonality_wfa(
                    symbol_dfs=loaded_dfs_wfa,
                    holding_periods=holding_periods,
                    directions=directions_wfa,
                    min_winrate=min_wr / 100,
                    min_trades=max(int(wfa_min_is * 0.8), 3),
                    min_is_years=int(wfa_min_is),
                    min_folds_for_badge=int(wfa_min_folds),
                    progress_callback=_wfa_progress,
                )
                _wfa_bar.empty()
                st.session_state["muster_wfa_result"] = wfa_res

        # WFA-Ergebnis anzeigen (falls vorhanden)
        _wfa_result = st.session_state.get("muster_wfa_result")
        if _wfa_result is not None:
            from seasonality_wfa import wfa_results_to_dataframe, wfa_badge_for_row
            wfa_df = wfa_results_to_dataframe(_wfa_result)
            n_validated = (wfa_df["Status"] == "✅ OOS-validiert").sum() if not wfa_df.empty else 0
            n_is_only   = (wfa_df["Status"] == "⚠️ Nur IS").sum() if not wfa_df.empty else 0
            st.markdown(
                f"""<div style="background:#0a1220;border:1px solid rgba(74,222,128,.2);border-radius:10px;
                    padding:14px 20px;margin-bottom:18px;">
                  <div style="color:#94a3b8;font-size:.75rem;font-weight:700;letter-spacing:.1em;
                      text-transform:uppercase;margin-bottom:8px;">Walk-Forward Ergebnis</div>
                  <span style="color:#4ade80;font-weight:800;font-size:1.1rem;">✅ {n_validated} OOS-validiert</span>
                  &nbsp;&nbsp;
                  <span style="color:#fbbf24;font-size:1rem;">⚠️ {n_is_only} Nur IS</span>
                  &nbsp;&nbsp;
                  <span style="color:#6b7fa3;font-size:.9rem;">IS-Start: {_wfa_result.min_is_years} Jahre · Badge ab {int(wfa_min_folds)} Folds</span>
                </div>""",
                unsafe_allow_html=True,
            )
            if not wfa_df.empty:
                st.dataframe(wfa_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ WFA-Ergebnis CSV",
                    wfa_df.to_csv(index=False).encode("utf-8"),
                    file_name="seasonality_wfa_result.csv",
                    mime="text/csv",
                )
            st.markdown("---")

        result = st.session_state.get("muster_scan_result", pd.DataFrame())
        csv_name = st.session_state.get("muster_csv_name", "")

        if result.empty:
            st.warning(f"Keine Muster mit ≥{min_wr}% Winrate gefunden.")
            return

        # Parse entry month for sorting Jan–Dez
        _month_map = {
            "Jan": 1, "Feb": 2, "Mär": 3, "Mar": 3, "Apr": 4, "Mai": 5, "May": 5,
            "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Okt": 10, "Oct": 10,
            "Nov": 11, "Dez": 12, "Dec": 12,
        }
        def _entry_sort_key(entry_str: str) -> int:
            try:
                parts = str(entry_str).strip().split(".")
                day = int(parts[0].strip())
                mon = parts[1].strip()[:3]
                return _month_map.get(mon, 0) * 100 + day
            except Exception:
                return 999
        result["_sort_key"] = result["Entry"].apply(_entry_sort_key)

        _wr_primary_col = f"WR {lookback}J %"

        n_long  = (result["Richtung"] == "Long").sum()
        n_short = (result["Richtung"] == "Short").sum()
        _wr_vals = pd.to_numeric(result.get(_wr_primary_col, pd.Series(dtype=float)), errors="coerce").dropna()
        top_wr   = f"{_wr_vals.max():.0f}%" if not _wr_vals.empty else "—"
        avg_wr   = f"{_wr_vals.mean():.0f}%" if not _wr_vals.empty else "—"

        st.markdown(
            f"""<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px;">
              <div style="background:#0a1220;border:1px solid rgba(74,222,128,.2);border-radius:8px;padding:12px 16px;">
                <div style="color:#6b7fa3;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;">Muster gesamt</div>
                <div style="color:#fff;font-size:1.4rem;font-weight:800;">{len(result)}</div>
              </div>
              <div style="background:#0a1220;border:1px solid rgba(74,222,128,.15);border-radius:8px;padding:12px 16px;">
                <div style="color:#6b7fa3;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;">Long / Short</div>
                <div style="font-size:1.1rem;font-weight:700;">
                  <span style="color:#4ade80;">▲ {n_long}</span>
                  <span style="color:#475569;"> / </span>
                  <span style="color:#f87171;">▼ {n_short}</span>
                </div>
              </div>
              <div style="background:#0a1220;border:1px solid rgba(240,196,64,.2);border-radius:8px;padding:12px 16px;">
                <div style="color:#6b7fa3;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;">Höchste WR {lookback}J</div>
                <div style="color:#f0c040;font-size:1.4rem;font-weight:800;">{top_wr}</div>
              </div>
              <div style="background:#0a1220;border:1px solid rgba(148,163,184,.15);border-radius:8px;padding:12px 16px;">
                <div style="color:#6b7fa3;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;">Ø WR {lookback}J</div>
                <div style="color:#e5edf8;font-size:1.4rem;font-weight:800;">{avg_wr}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        cf1, cf2, cf3 = st.columns([1, 1, 2])
        with cf1:
            dir_filter = st.radio("Richtung", ["Alle", "Long", "Short"], horizontal=True)
        with cf2:
            sort_by = st.radio("Sortierung", ["Winrate", "Datum"], horizontal=True)
        with cf3:
            _monate = ["Alle Monate", "Januar", "Februar", "März", "April", "Mai", "Juni",
                       "Juli", "August", "September", "Oktober", "November", "Dezember"]
            monat_filter = st.selectbox("Monat", _monate, index=0, label_visibility="visible", key="muster_monat_filter")
        monat_nr = _monate.index(monat_filter)

        display = result if dir_filter == "Alle" else result[result["Richtung"] == dir_filter].copy()
        if monat_nr > 0:
            display = display[display["Entry"].apply(
                lambda e: _month_map.get(str(e).strip().split(".")[-1].strip()[:3], 0) == monat_nr
            )].copy()
        if sort_by == "Datum":
            display = display.sort_values("_sort_key").reset_index(drop=True)
        display = display.drop(columns=["_sort_key"], errors="ignore")
        # Sterne als lesbare Darstellung
        if "⭐ Rating" in display.columns:
            display["⭐ Rating"] = display["⭐ Rating"].apply(
                lambda s: "⭐" * int(s) + "☆" * (5 - int(s)) if pd.notna(s) else "—"
            )

        def color_richtung(val: str) -> str:
            return "color: #4ade80; font-weight:bold" if val == "Long" else "color: #f87171; font-weight:bold"

        def color_num(val: object) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return "color: #4ade80" if float(val) > 0 else "color: #f87171"

        # Numeric coerce
        for _nc in ["Sharpe", "SQN", "WR 5J %", "WR 10J %", "WR 15J %", "WR 20J %", "WR Späteinstieg %", "Ø ATR %"]:
            if _nc in display.columns:
                display[_nc] = pd.to_numeric(display[_nc], errors="coerce")

        def color_wr(val: object) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "color:#9fb0c7"
            v = float(val)
            if v >= 80: return "color:#4ade80;font-weight:700"
            if v >= 70: return "color:#a3e635;font-weight:600"
            if v >= 60: return "color:#facc15"
            return "color:#f87171"

        wr_display_cols = [c for c in ["WR 5J %", "WR 10J %", "WR 15J %", "WR 20J %"] if c in display.columns]
        num_cols = ["Ø Profit %", "Ø DD %", "Max DD %", "Sharpe", "SQN"]
        existing_num_cols = [c for c in num_cols if c in display.columns]

        # WFA-Badge in Tabelle einfügen (wenn WFA-Ergebnis vorhanden)
        _wfa_res_for_display = st.session_state.get("muster_wfa_result")
        if _wfa_res_for_display is not None:
            from seasonality_wfa import wfa_badge_for_row
            def _get_wfa_badge(row: pd.Series) -> str:
                return wfa_badge_for_row(
                    _wfa_res_for_display,
                    symbol=str(row.get("Symbol", "")),
                    entry_doy=int(row.get("_entry_doy", 0)),
                    exit_doy=int(row.get("_exit_doy", 0)),
                    direction="long" if row.get("Richtung") == "Long" else "short",
                )
            display["WFA"] = display.apply(_get_wfa_badge, axis=1)

        style_obj = (
            display.style
            .map(color_richtung, subset=["Richtung"])
            .map(color_num, subset=existing_num_cols)
            .map(color_wr, subset=wr_display_cols)
        )
        fmt: dict = {}
        for _wc in [c for c in display.columns if "WR " in c and "%" in c]:
            fmt[_wc] = "{:.1f}%"
        fmt.update({
            "Ø Profit %": "{:+.2f}%",
            "Ø DD %": "{:+.2f}%",
            "Max DD %": "{:+.2f}%",
            "Sharpe": "{:.2f}",
            "SQN": "{:.2f}",
        })
        styled = style_obj.format({k: v for k, v in fmt.items() if k in display.columns}, na_rep="-")
        with st.expander("📋 Alle Muster anzeigen", expanded=False):
            try:
                st.dataframe(styled, use_container_width=True, height=420)
            except Exception:
                st.dataframe(display, use_container_width=True, height=420)

        csv_export = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ CSV Export",
            data=csv_export,
            file_name=f"seasonality_muster_{lookback}y_wr{min_wr}.csv",
            mime="text/csv",
        )

        # ── Top Setups ausgewählter Monat ──────────────────────────────────────
        import datetime as _dt
        _month_names_list = ["", "Januar", "Februar", "März", "April", "Mai", "Juni",
                             "Juli", "August", "September", "Oktober", "November", "Dezember"]
        # Ausgewählten Monat aus Dropdown nutzen, sonst aktuellen Monat
        if monat_nr > 0:
            current_month = monat_nr
        else:
            current_month = _dt.date.today().month
        current_month_name = _month_names_list[current_month]

        result_clean = result.drop(columns=["_sort_key"], errors="ignore")

        # WFA-Status pro Muster ermitteln (für Top-Setups-Badge/Filter — Sterne allein sagen
        # nichts über OOS-Validierung aus, siehe Walk-Forward-Erklärung oben)
        _wfa_res_top = st.session_state.get("muster_wfa_result")
        def _wfa_status_for_row(r: pd.Series) -> str:
            if _wfa_res_top is None:
                return ""
            from seasonality_wfa import wfa_badge_for_row
            return wfa_badge_for_row(
                _wfa_res_top, symbol=str(r.get("Symbol", "")),
                entry_doy=int(r.get("_entry_doy", 0)), exit_doy=int(r.get("_exit_doy", 0)),
                direction="long" if r.get("Richtung") == "Long" else "short",
            )
        result_clean["WFA Status"] = result_clean.apply(_wfa_status_for_row, axis=1)

        _today = _dt.date.today()
        # Wochenende: nächsten Montag als effektiven "heute" nutzen
        if _today.weekday() == 5: _today = _today + _dt.timedelta(days=2)  # Sa → Mo
        if _today.weekday() == 6: _today = _today + _dt.timedelta(days=1)  # So → Mo
        # Für zukünftige Monate: ab dem 1. des Monats filtern
        if current_month != _today.month:
            _today_day = 1
        else:
            _today_day = _today.day
        _today_month = current_month

        def _entry_day_key(entry_str: str) -> int:
            try:
                return int(str(entry_str).strip().split(".")[0])
            except Exception:
                return 99

        # Nur Muster im ausgewählten Monat UND Entry >= nächster Handelstag
        def _is_relevant(entry_str: str) -> bool:
            parts = str(entry_str).strip().split(".")
            try:
                mon = _month_map.get(parts[1].strip()[:3], 0)
                day = int(parts[0].strip())
                return mon == _today_month and day >= _today_day
            except Exception:
                return False

        month_mask = result_clean["Entry"].apply(_is_relevant)
        _sort_top = _wr_primary_col if _wr_primary_col and _wr_primary_col in result_clean.columns else result_clean.columns[0]

        _top_raw = result_clean[month_mask].copy()
        _top_raw["_day_key"] = _top_raw["Entry"].apply(_entry_day_key)
        _top_raw["_stars"]   = pd.to_numeric(_top_raw.get("⭐ Rating", 3), errors="coerce").fillna(3)
        _top_raw["_sharpe"]  = pd.to_numeric(_top_raw.get("Sharpe", 0),    errors="coerce").fillna(0)

        # Nur 5 Sterne
        _top_raw = _top_raw[_top_raw["_stars"] >= 5]

        # Pro Symbol: Cluster-Dedup — Entries innerhalb von 3 Tagen = gleiche Opportunity
        # Sortiere nach Symbol + Entry-Tag, dann beste Sterne/Sharpe nach vorne
        _top_raw = _top_raw.sort_values(
            ["Symbol", "_day_key", "_stars", "_sharpe"],
            ascending=[True, True, False, False]
        )
        deduped_rows = []
        for _sym, _grp in _top_raw.groupby("Symbol", sort=False):
            _last_day = -99
            for _, _r in _grp.iterrows():
                if _r["_day_key"] - _last_day > 3:  # neuer Cluster
                    deduped_rows.append(_r)
                    _last_day = _r["_day_key"]
        _top_raw = pd.DataFrame(deduped_rows) if deduped_rows else _top_raw.head(0)

        top_month = (
            _top_raw
            .sort_values(["Symbol", "_sharpe", "_day_key"], ascending=[True, False, True])
            .drop(columns=["_day_key", "_stars", "_sharpe"], errors="ignore")
            .reset_index(drop=True)
        )

        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;margin:28px 0 16px 0;
            border-top:1px solid rgba(148,163,184,.1);padding-top:24px;">
              <div style="color:#f0c040;font-size:1.2rem;">⭐</div>
              <div>
                <div style="color:#fff;font-size:1.1rem;font-weight:800;">
                  Top Trading Setups — {current_month_name}
                </div>
                <div style="color:#6b7fa3;font-size:.8rem;">
                  Beste Muster im aktuellen Monat · nach Symbol gruppiert · nach WR {lookback}J sortiert
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        if _wfa_res_top is None:
            st.caption(
                "⚠️ Walk-Forward noch nicht gelaufen — Sterne basieren nur auf In-Sample-Daten, "
                "keines dieser Setups ist OOS-validiert. Erst '🔄 Walk-Forward validieren' klicken."
            )
        else:
            _only_validated = st.checkbox(
                "Nur ✅ WFA-validierte Setups zeigen", value=False, key="top_only_wfa_validated",
            )
            if _only_validated:
                top_month = top_month[top_month["WFA Status"] == "✅ OOS-validiert"].reset_index(drop=True)

        saved_dfs = st.session_state.get("muster_dataframes", {})

        _notes_all = _load_notes()

        if top_month.empty:
            st.info(f"Keine Muster mit ≥{min_wr}% Winrate im {current_month_name} gefunden.")
        else:
            _prev_symbol = None
            for i, row in top_month.iterrows():
                symbol_str = str(row.get("Symbol", "—"))
                if symbol_str != _prev_symbol:
                    if _prev_symbol is not None:
                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='color:#e5edf8;font-size:1.05rem;font-weight:800;"
                        f"letter-spacing:.04em;margin:14px 0 5px 4px;'>"
                        f"{symbol_str}</div>",
                        unsafe_allow_html=True,
                    )
                    _prev_symbol = symbol_str
                richtung   = row["Richtung"]
                farbe      = "#4ade80" if richtung == "Long" else "#f87171"
                pfeil      = "▲" if richtung == "Long" else "▼"
                profit_val = float(row.get("Ø Profit %", 0))
                profit_clr = "#4ade80" if profit_val >= 0 else "#f87171"
                wr_primary = row.get(_wr_primary_col, float("nan"))
                wr_5_val   = row.get("WR 5J %",  float("nan"))
                wr_5_str   = f"{wr_5_val:.0f}%"  if pd.notna(wr_5_val)  else "—"
                wr_5_clr   = "#4ade80" if pd.notna(wr_5_val)  and float(wr_5_val)  >= 70 else "#f87171"
                wr_10_val  = row.get("WR 10J %", float("nan"))
                wr_10_str  = f"{wr_10_val:.0f}%" if pd.notna(wr_10_val) else "—"
                wr_10_clr  = "#4ade80" if pd.notna(wr_10_val) and float(wr_10_val) >= 70 else "#f87171"
                wr_15_val  = row.get("WR 15J %", float("nan"))
                wr_15_str  = f"{wr_15_val:.0f}%" if pd.notna(wr_15_val) else "—"
                wr_15_clr  = "#4ade80" if pd.notna(wr_15_val) and float(wr_15_val) >= 70 else "#f87171"
                wr_20_val  = row.get("WR 20J %", float("nan"))
                wr_20_str  = f"{wr_20_val:.0f}%" if pd.notna(wr_20_val) else "—"
                wr_20_clr  = "#4ade80" if pd.notna(wr_20_val) and float(wr_20_val) >= 70 else "#f87171"
                wr_prim_str = f"{wr_primary:.0f}%" if pd.notna(wr_primary) else "—"
                sharpe_str = f"{row.get('Sharpe','—')}"
                hold_td    = row.get("Haltedauer (TD)", "—")
                # Sterne + Robustheit für Karte
                _stars_raw  = row.get("⭐ Rating", 3)
                _stars_int  = int(_stars_raw) if pd.notna(_stars_raw) else 3
                _star_str   = "⭐" * _stars_int + "☆" * (5 - _stars_int)
                _rob_val    = str(row.get("Robustheit", "—"))
                _rob_clr    = {"🟢 Stark": "#4ade80", "✅ Robust": "#a3e635", "⚠️ Sensitiv": "#facc15", "❌ Fragil": "#f87171"}.get(_rob_val, "#6b7fa3")
                # WFA-Badge live aus session_state holen
                _wfa_res_top = st.session_state.get("muster_wfa_result")
                if _wfa_res_top is not None:
                    try:
                        from seasonality_wfa import wfa_badge_for_row
                        _wfa_val = wfa_badge_for_row(
                            _wfa_res_top,
                            symbol=symbol_str,
                            entry_doy=int(row.get("_entry_doy", 0)),
                            exit_doy=int(row.get("_exit_doy", 0)),
                            direction="long" if richtung == "Long" else "short",
                        ) or "⚠️ Nur IS"
                    except Exception:
                        _wfa_val = "— nicht WFA-getestet"
                else:
                    _wfa_val = "— nicht WFA-getestet"
                _wfa_clr    = {"✅ OOS-validiert": "#4ade80", "⚠️ Nur IS": "#fbbf24"}.get(_wfa_val, "#475569")
                # Wilson-CI für Übersichtskarte (10J Fenster)
                _n_10 = round(float(wr_10_val) / 100 * 10) if pd.notna(wr_10_val) else 0
                _ci_lo_10, _ci_hi_10 = wilson_ci(_n_10, 10) if pd.notna(wr_10_val) else (0.0, 1.0)
                _ci_w_10 = _ci_hi_10 - _ci_lo_10
                _ci_clr_10 = "#4ade80" if _ci_w_10 < 0.28 else ("#fbbf24" if _ci_w_10 < 0.42 else "#f87171")
                _ci_str_10 = f"{_ci_lo_10*100:.0f}–{_ci_hi_10*100:.0f}%" if pd.notna(wr_10_val) else "—"

                # Notiz für dieses Muster laden
                _nkey = _notes_key(symbol_str, str(row.get("Entry", "")), str(row.get("Exit", "")), richtung)
                _cur_note = _notes_all.get(_nkey, "")

                col_info, col_btn = st.columns([6, 1])
                with col_info:
                    st.markdown(
                        f"""<div style="background:#0a1220;border-left:3px solid {farbe};
                        border-radius:0 6px 6px 0;padding:10px 16px;margin-bottom:5px;
                        display:flex;gap:0;align-items:stretch;">
                          <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center;flex:1;">
                            <span style="background:{farbe}22;border:1px solid {farbe}44;border-radius:4px;
                              padding:2px 10px;color:{farbe};font-weight:700;font-size:.85rem;white-space:nowrap;">
                              {pfeil} {richtung}
                            </span>
                            <span style="color:#e5edf8;font-size:.9rem;">
                              📅 <b>{row['Entry']}</b> → <b>{row['Exit']}</b>
                            </span>
                            <span style="color:#6b7fa3;font-size:.85rem;">⏱ {hold_td} TD</span>
                            <span style="color:{wr_5_clr};font-weight:700;font-size:.9rem;">
                              WR 5J: {wr_5_str}
                            </span>
                            <span style="color:{wr_10_clr};font-weight:700;font-size:.9rem;">
                              WR 10J: {wr_10_str}
                            </span>
                            <span style="color:{wr_15_clr};font-weight:700;font-size:.9rem;">
                              WR 15J: {wr_15_str}
                            </span>
                            <span style="color:{wr_20_clr};font-weight:700;font-size:.9rem;">
                              WR 20J: {wr_20_str}
                            </span>
                            <span style="color:{profit_clr};font-size:.85rem;">Ø {profit_val:+.2f}%</span>
                            <span style="color:#475569;font-size:.8rem;">Sharpe {sharpe_str}</span>
                            <span style="font-size:.95rem;letter-spacing:.02em;">{_star_str}</span>
                            <span style="color:{_rob_clr};font-size:.8rem;font-weight:600;">{_rob_val}</span>
                            <span style="color:{_wfa_clr};font-size:.8rem;font-weight:700;">{_wfa_val}</span>
                            <span style="color:{_ci_clr_10};font-size:.78rem;font-weight:600;border:1px solid {_ci_clr_10}44;border-radius:3px;padding:1px 6px;">CI {_ci_str_10}</span>
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("Detail →", key=f"detail_{i}"):
                        st.session_state["muster_detail"] = {
                            "row": row.to_dict(),
                            "symbol": symbol_str,
                            "lookback": lookback,
                        }
                        st.rerun()

                # ── Notiz-Bereich ──────────────────────────────────────────────
                with st.expander(
                    f"📝 Fundamentaler Grund {'· ' + _cur_note[:40] + ('…' if len(_cur_note) > 40 else '') if _cur_note else '· kein Grund hinterlegt'}",
                    expanded=False,
                ):
                    if not _cur_note:
                        st.markdown(
                            "<div style='color:#475569;font-size:.75rem;font-style:italic;margin-bottom:6px;'>"
                            "ℹ️ Kein fundamentaler Grund hinterlegt</div>",
                            unsafe_allow_html=True,
                        )
                    _new_note = st.text_area(
                        "Notiz",
                        value=_cur_note,
                        key=f"note_ta_{i}",
                        height=80,
                        label_visibility="collapsed",
                        placeholder="z.B. JP-Fiskaljahresende, US-Optionsverfall, Quarter-End Rebalancing …",
                    )
                    if st.button("💾 Speichern", key=f"note_save_{i}"):
                        _notes_all[_nkey] = _new_note.strip()
                        _save_notes(_notes_all)
                        st.success("Notiz gespeichert.")


def render_seasonality_lab() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #070b13; }
        .block-container {
            padding-top: 1.2rem;
            max-width: min(96vw, 118rem);
        }
        .season-toolbar {
            background: #141c28;
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 6px;
            padding: 12px 14px;
            margin: 8px 0 10px 0;
        }
        .season-title-row {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #e5edf8;
            font-size: 1.08rem;
            font-weight: 700;
        }
        .season-pill {
            color: #9fb0c7;
            border: 1px solid rgba(148,163,184,.26);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: .72rem;
            font-weight: 600;
        }
        .season-panel {
            background: #141c28;
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 6px;
            padding: 9px 12px;
            margin-bottom: 8px;
        }
        .season-panel-title {
            color: #9fb0c7;
            text-transform: uppercase;
            letter-spacing: .03em;
            font-size: .68rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 7px;
        }
        .season-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
        }
        .season-stat {
            text-align: center;
            color: #9fb0c7;
            font-size: .68rem;
            line-height: 1.15;
        }
        .season-stat strong {
            display: block;
            color: #63c7e8;
            font-size: .90rem;
            line-height: 1.1;
            margin-bottom: 2px;
        }
        .season-stat.negative strong { color: #e36d5c; }
        .season-stat.neutral strong { color: #d6e3f3; }
        .season-year-caption {
            color: #9fb0c7;
            font-size: .76rem;
            margin: 8px 0 2px 0;
        }
        .season-year-strip {
            background: #141c28;
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 6px;
            padding: 8px 10px 2px 10px;
            margin: 8px 0 10px 0;
        }
        .season-period-badge {
            display: inline-block;
            background: #62c8e8;
            color: #0f172a;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: .78rem;
            font-weight: 800;
            margin-top: 2px;
        }
        div[data-testid="stPlotlyChart"] {
            background: #141c28;
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 6px;
            padding: 6px;
        }
        div[data-testid="stPlotlyChart"]:has(.js-plotly-plot) {
            overflow: hidden;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15,23,42,.96), rgba(15,23,42,.72));
            border: 1px solid rgba(59,130,246,.22);
            border-radius: 8px;
            padding: 10px 12px;
        }
        [data-testid="stMetricLabel"] { color: #93c5fd; }
        [data-testid="stMetricValue"] { color: #e0f2fe; font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="season-toolbar">
            <div class="season-title-row">
                <span>Seasonality Lab</span>
                <span class="season-pill">Yahoo Daily</span>
                <span class="season-pill">Independent from TACO</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def parse_period_text(raw_period: str) -> tuple[int, int, int, int] | None:
        match = re.fullmatch(r"\s*(\d{1,2})[./](\d{1,2})\s*[-–]\s*(\d{1,2})[./](\d{1,2})\s*", raw_period or "")
        if not match:
            return None
        start_day_raw, start_month_raw, end_day_raw, end_month_raw = [int(part) for part in match.groups()]
        try:
            _valid_month_day(2001, start_month_raw, start_day_raw)
            _valid_month_day(2001, end_month_raw, end_day_raw)
        except Exception:
            return None
        return start_month_raw, start_day_raw, end_month_raw, end_day_raw

    def format_period_from_markers(start_marker_value: pd.Timestamp, end_marker_value: pd.Timestamp) -> str:
        return f"{start_marker_value.day:02d}.{start_marker_value.month:02d} - {end_marker_value.day:02d}.{end_marker_value.month:02d}"

    pending_period_text = st.session_state.pop("seasonality_pending_period_text", None)
    if pending_period_text:
        st.session_state["seasonality_period_text"] = pending_period_text
        st.session_state["seasonality_period_from_selection"] = True

    control_cols = st.columns([1.35, 1.0, 1.15, 1.0])
    with control_cols[0]:
        asset_label = st.selectbox("Asset", list(ASSET_PRESETS.keys()), key="seasonality_asset")
        default_symbol = ASSET_PRESETS[asset_label]
        mt5_base_symbol = get_mt5_base_symbol(asset_label, default_symbol)
        if "seasonality_symbol" not in st.session_state:
            st.session_state["seasonality_symbol"] = default_symbol
        if st.session_state.get("seasonality_last_asset_label") != asset_label:
            st.session_state["seasonality_last_asset_label"] = asset_label
            st.session_state["seasonality_symbol"] = default_symbol
            st.session_state.pop("seasonality_manual_period", None)
            st.session_state.pop("seasonality_pending_period_text", None)
            for state_key in list(st.session_state.keys()):
                if state_key.startswith("seasonality_year_"):
                    del st.session_state[state_key]
        data_source_label = st.selectbox(
            "Data Source",
            ["Pepperstone MT5 CSV", "Yahoo Finance"],
            key="seasonality_data_source",
        )
        if st.session_state.get("seasonality_last_data_source") != data_source_label:
            st.session_state["seasonality_last_data_source"] = data_source_label
            st.session_state.pop("seasonality_manual_period", None)
            st.session_state.pop("seasonality_pending_period_text", None)
            for state_key in list(st.session_state.keys()):
                if state_key.startswith("seasonality_year_"):
                    del st.session_state[state_key]
        # Crypto-Assets haben keine MT5 CSV → automatisch Yahoo
        _crypto_symbols = {"BTC-USD", "ETH-USD"}
        _force_yahoo = default_symbol in _crypto_symbols
        if _force_yahoo and data_source_label == "Pepperstone MT5 CSV":
            data_source_label = "Yahoo Finance"

        if data_source_label == "Pepperstone MT5 CSV":
            symbol = mt5_base_symbol
            st.text_input("MT5 CSV Symbol", value=mt5_base_symbol, disabled=True)
        else:
            symbol = st.text_input("Yahoo Symbol", value=default_symbol if _force_yahoo else "", key="seasonality_symbol")

    if not symbol.strip():
        st.warning("Bitte ein Symbol eingeben.")
        return

    if data_source_label == "Pepperstone MT5 CSV":
        with st.spinner(f"Lade Pepperstone-MT5-CSV fuer {symbol}..."):
            loaded_df = load_ohlc_data(symbol.strip(), source="mt5")
        df = normalize_loader_ohlc(loaded_df) if loaded_df is not None else None
    else:
        with st.spinner(f"Lade maximale Yahoo-Historie fuer {symbol}..."):
            df = load_seasonality_data(symbol.strip())

    if df is None or df.empty:
        st.warning("Fuer diese Datenquelle wurden keine verwertbaren Tagesdaten gefunden.")
        return

    all_years = sorted(pd.Index(df.index.year).unique().astype(int).tolist())
    available_year_count = len(all_years)
    completed_pattern_years = [year for year in all_years if year < date.today().year]
    pattern_year_count = len(completed_pattern_years) if completed_pattern_years else available_year_count

    def build_lookback_options(available_years: int) -> list[str]:
        standard_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
        options = [f"{years} Jahre" for years in standard_steps if years <= available_years]
        if available_years > 0 and available_years not in standard_steps:
            options.append(f"{available_years} Jahre")
        options.append(f"Max verfuegbare Jahre ({available_years} Jahre)")
        return options

    with control_cols[1]:
        lookback_options = build_lookback_options(pattern_year_count)
        previous_lookback = st.session_state.get("seasonality_lookback")
        if previous_lookback not in lookback_options:
            st.session_state["seasonality_lookback"] = lookback_options[-1]
        lookback_label = st.selectbox(
            "Lookback",
            lookback_options,
            key="seasonality_lookback",
        )
        lookback_years = None if lookback_label.startswith("Max") else int(lookback_label.split()[0])
    with control_cols[2]:
        cycle_filter = st.selectbox(
            "Praesidenten-Zyklus",
            [
                "Alle Jahre",
                "US Presidential Cycle: Election Year",
                "US Presidential Cycle: Post-Election Year",
                "US Presidential Cycle: Midterm Year",
                "US Presidential Cycle: Pre-Election Year",
            ],
        )
    with control_cols[3]:
        period_text = st.text_input(
            "Zeitraum",
            value="26.06 - 29.07",
            key="seasonality_period_text",
            help="Format: TT.MM - TT.MM, z.B. 26.06 - 29.07",
        )
        parsed_period = parse_period_text(period_text)
        if parsed_period is None:
            st.warning("Bitte Zeitraum im Format TT.MM - TT.MM eingeben, z.B. 26.06 - 29.07.")
            return
        start_month, start_day, end_month, end_day = parsed_period
        period_token = f"{start_day:02d}.{start_month:02d}_{end_day:02d}.{end_month:02d}"
        if st.session_state.get("seasonality_period_token") != period_token:
            st.session_state["seasonality_period_token"] = period_token
            if st.session_state.pop("seasonality_period_from_selection", False):
                pass
            else:
                st.session_state.pop("seasonality_manual_period", None)
        st.markdown(
            f"<span class='season-period-badge'>{_valid_month_day(2001, start_month, start_day).strftime('%d %b')} - {_valid_month_day(2001, end_month, end_day).strftime('%d %b')}</span>",
            unsafe_allow_html=True,
        )

    st.markdown(f"**Data Source:** {data_source_label}")
    if all_years:
        st.caption(
            f"Datenabdeckung fuer {symbol.strip()}: "
            f"{df.index.min().date()} bis {df.index.max().date()} "
            f"({len(all_years)} Kalenderjahre mit Daten, {pattern_year_count} abgeschlossene Pattern-Jahre)."
        )
    selected_years = filter_years_by_lookback_and_cycle(df, lookback_years, cycle_filter)
    if len(selected_years) < 3:
        st.warning("Fuer diese Auswahl sind weniger als drei Pattern-Jahre verfuegbar. Bitte Lookback oder Filter erweitern.")
    if not selected_years:
        return

    st.markdown("<div class='season-year-strip'>", unsafe_allow_html=True)
    cycle_label = cycle_filter.replace("US Presidential Cycle: ", "")
    st.markdown(f"<div class='season-year-caption'>Select years to display: {cycle_label}</div>", unsafe_allow_html=True)
    year_cols = st.columns(12)
    active_years = []
    year_key_seed = f"{symbol.strip()}_{lookback_label}_{cycle_filter}"
    for idx, year in enumerate(selected_years):
        checked = year_cols[idx % 12].checkbox(
            str(year),
            value=True,
            key=f"seasonality_year_{year_key_seed}_{year}",
        )
        if checked:
            active_years.append(year)
    st.markdown("</div>", unsafe_allow_html=True)

    if len(active_years) < 1:
        st.warning("Bitte mindestens ein Pattern-Jahr auswaehlen.")
        return

    active_years_token = tuple(active_years)
    active_years_changed = (
        tuple(st.session_state.get("seasonality_active_years_token", ())) != active_years_token
    )

    curve = build_seasonal_curve(df, active_years)
    if curve.empty:
        st.warning("Aus den ausgewaehlten Jahren konnte keine saisonale Kurve gebaut werden.")
        return

    today = pd.Timestamp(year=2001, month=date.today().month, day=date.today().day if not (date.today().month == 2 and date.today().day == 29) else 28)
    input_start_marker = pd.Timestamp(year=2001, month=int(start_month), day=min(int(start_day), calendar.monthrange(2001, int(start_month))[1]))
    input_end_marker = pd.Timestamp(year=2001, month=int(end_month), day=min(int(end_day), calendar.monthrange(2001, int(end_month))[1]))

    def parse_chart_period(selection_state) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if not selection_state:
            return None
        selection = getattr(selection_state, "selection", None)
        if selection is None and isinstance(selection_state, dict):
            selection = selection_state.get("selection")
        if not selection:
            return None
        boxes = getattr(selection, "box", None)
        if boxes is None and isinstance(selection, dict):
            boxes = selection.get("box", [])
        for box in boxes or []:
            raw_range = getattr(box, "range", None)
            if raw_range is None and isinstance(box, dict):
                raw_range = box.get("range")
            raw_x = None
            if raw_range is not None:
                raw_x = getattr(raw_range, "x", None)
                if raw_x is None and isinstance(raw_range, dict):
                    raw_x = raw_range.get("x")
            if raw_x is None:
                raw_x = getattr(box, "x", None)
                if raw_x is None and isinstance(box, dict):
                    raw_x = box.get("x")
            if raw_x is None or len(raw_x) < 2:
                continue
            start = pd.Timestamp(raw_x[0])
            end = pd.Timestamp(raw_x[1])
            if end < start:
                start, end = end, start
            return (
                pd.Timestamp(year=2001, month=int(start.month), day=int(start.day)),
                pd.Timestamp(year=2001, month=int(end.month), day=int(end.day)),
            )
        points = getattr(selection, "points", None)
        if points is None and isinstance(selection, dict):
            points = selection.get("points", [])
        x_values = []
        for point in points or []:
            raw_x = getattr(point, "x", None)
            if raw_x is None and isinstance(point, dict):
                raw_x = point.get("x")
            if raw_x is not None:
                x_values.append(pd.Timestamp(raw_x))
        if len(x_values) < 2:
            return None
        start = min(x_values)
        end = max(x_values)
        return (
            pd.Timestamp(year=2001, month=int(start.month), day=int(start.day)),
            pd.Timestamp(year=2001, month=int(end.month), day=int(end.day)),
        )

    def parse_absolute_chart_period(selection_state) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if not selection_state:
            return None
        selection = getattr(selection_state, "selection", None)
        if selection is None and isinstance(selection_state, dict):
            selection = selection_state.get("selection")
        if not selection:
            return None
        boxes = getattr(selection, "box", None)
        if boxes is None and isinstance(selection, dict):
            boxes = selection.get("box", [])
        for box in boxes or []:
            raw_range = getattr(box, "range", None)
            if raw_range is None and isinstance(box, dict):
                raw_range = box.get("range")
            raw_x = None
            if raw_range is not None:
                raw_x = getattr(raw_range, "x", None)
                if raw_x is None and isinstance(raw_range, dict):
                    raw_x = raw_range.get("x")
            if raw_x is None:
                raw_x = getattr(box, "x", None)
                if raw_x is None and isinstance(box, dict):
                    raw_x = box.get("x")
            if raw_x is None or len(raw_x) < 2:
                continue
            start = pd.Timestamp(raw_x[0])
            end = pd.Timestamp(raw_x[1])
            if end < start:
                start, end = end, start
            return (start, end)
        points = getattr(selection, "points", None)
        if points is None and isinstance(selection, dict):
            points = selection.get("points", [])
        x_values = []
        for point in points or []:
            raw_x = getattr(point, "x", None)
            if raw_x is None and isinstance(point, dict):
                raw_x = point.get("x")
            if raw_x is not None:
                x_values.append(pd.Timestamp(raw_x))
        if len(x_values) < 2:
            return None
        return (min(x_values), max(x_values))

    def chart_selection_is_empty(selection_state) -> bool:
        if not selection_state:
            return False
        selection = getattr(selection_state, "selection", None)
        if selection is None and isinstance(selection_state, dict):
            selection = selection_state.get("selection")
        if selection is None:
            return False
        boxes = getattr(selection, "box", None)
        points = getattr(selection, "points", None)
        lasso = getattr(selection, "lasso", None)
        if isinstance(selection, dict):
            boxes = selection.get("box", boxes)
            points = selection.get("points", points)
            lasso = selection.get("lasso", lasso)
        return not (boxes or points or lasso)

    manual_period = st.session_state.get("seasonality_manual_period")
    if manual_period:
        display_start_marker = pd.Timestamp(manual_period[0])
        display_end_marker = pd.Timestamp(manual_period[1])
    else:
        display_start_marker = None
        display_end_marker = None

    asset_short = asset_label.split(" proxy:")[0].replace(" proxy", "")
    years_text = f"{len(active_years)} Years"
    main_col, stat_col = st.columns([4.35, 1.2])
    with main_col:
        chart_curve = curve.copy()
        chart_curve["indexed_display"] = chart_curve["indexed"].rolling(7, center=True, min_periods=1).mean()
        chart_floor = min(float(chart_curve["indexed_display"].min()), 100.0)
        chart_ceiling = max(float(chart_curve["indexed_display"].max()), 100.0)
        chart_padding = max((chart_ceiling - chart_floor) * 0.16, 1.5)
        chart_label_y = chart_ceiling + chart_padding * 0.58

        def marker_curve_value(marker: pd.Timestamp) -> float:
            nearest_idx = (chart_curve["plot_date"] - marker).abs().idxmin()
            return float(chart_curve.loc[nearest_idx, "indexed_display"])

        if display_start_marker is not None and display_end_marker is not None:
            start_marker_value = marker_curve_value(display_start_marker)
            end_marker_value = marker_curve_value(display_end_marker)
        else:
            start_marker_value = end_marker_value = None
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_curve["plot_date"],
                y=chart_curve["indexed_display"],
                mode="lines",
                line={"color": "rgba(98,200,232,.18)", "width": 0},
                fill="tozeroy",
                fillcolor="rgba(98,200,232,.13)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_curve["plot_date"],
                y=chart_curve["indexed_display"],
                mode="lines",
                name="Average Seasonal Trend",
                line={"color": "#62c8e8", "width": 2.1, "shape": "spline", "smoothing": 0.55},
                hovertemplate="%{x|%b %d}<br>Index %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_vline(x=today, line_color="#c0267a", line_width=2)
        if display_start_marker is not None and display_end_marker is not None:
            fig.add_vline(x=display_start_marker, line_color="rgba(226,232,240,.82)", line_width=1.3)
            fig.add_vline(x=display_end_marker, line_color="rgba(226,232,240,.82)", line_width=1.3)
            if display_end_marker >= display_start_marker:
                fig.add_vrect(x0=display_start_marker, x1=display_end_marker, fillcolor="#62c8e8", opacity=0.11, line_width=0)
            else:
                fig.add_vrect(x0=display_start_marker, x1=pd.Timestamp("2001-12-31"), fillcolor="#62c8e8", opacity=0.11, line_width=0)
                fig.add_vrect(x0=pd.Timestamp("2001-01-01"), x1=display_end_marker, fillcolor="#62c8e8", opacity=0.11, line_width=0)
        fig.update_layout(**_seasonality_base_layout(f"Seasonal Trend of {asset_short} over {years_text}", 700))
        fig.update_layout(
            dragmode="select",
            uirevision=f"seasonality_full_year_{asset_short}",
        )
        fig.add_annotation(
            text="seasonality",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.52,
            showarrow=False,
            font={"size": 54, "color": "rgba(148,163,184,.105)"},
        )
        if display_start_marker is not None and display_end_marker is not None:
            for marker, marker_value in [(display_start_marker, start_marker_value), (display_end_marker, end_marker_value)]:
                fig.add_annotation(
                    x=marker,
                    y=chart_label_y,
                    text=f"{marker.strftime('%d %b')}: {marker_value:.2f}",
                    showarrow=False,
                    bgcolor="rgba(31,41,55,.92)",
                    bordercolor="rgba(226,232,240,.22)",
                    borderpad=4,
                    font={"size": 10, "color": "#dbeafe"},
                )
        fig.update_xaxes(
            tickformat="%b",
            dtick="M1",
            showspikes=False,
            fixedrange=True,
            range=[pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31")],
        )
        fig.update_yaxes(title="", range=[chart_floor - chart_padding, chart_ceiling + chart_padding])
        chart_selection = st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": "reset",
                "staticPlot": False,
            },
            key="seasonality_main_curve",
            on_select="rerun",
            selection_mode=("box",),
        )

    selected_chart_period = parse_chart_period(chart_selection)
    if selected_chart_period:
        selected_token = tuple(marker.isoformat() for marker in selected_chart_period)
        if selected_token != tuple(st.session_state.get("seasonality_manual_period", ())):
            st.session_state["seasonality_manual_period"] = selected_token
            st.session_state["seasonality_chart_selection_active"] = True
            st.session_state["seasonality_just_set_manual_period"] = selected_token
            st.session_state["seasonality_pending_period_text"] = format_period_from_markers(
                selected_chart_period[0],
                selected_chart_period[1],
            )
            st.rerun()
        else:
            st.session_state.pop("seasonality_just_set_manual_period", None)
    elif (
        st.session_state.get("seasonality_chart_selection_active")
        and st.session_state.get("seasonality_manual_period")
        and not active_years_changed
        and chart_selection_is_empty(chart_selection)
    ):
        just_set_period = st.session_state.pop("seasonality_just_set_manual_period", None)
        if just_set_period != tuple(st.session_state.get("seasonality_manual_period", ())):
            st.session_state.pop("seasonality_manual_period", None)
            st.session_state["seasonality_chart_selection_active"] = False
            st.rerun()

    st.session_state["seasonality_active_years_token"] = active_years_token

    manual_period = st.session_state.get("seasonality_manual_period")
    if manual_period:
        analysis_start_marker = pd.Timestamp(manual_period[0])
        analysis_end_marker = pd.Timestamp(manual_period[1])
    else:
        analysis_start_marker = input_start_marker
        analysis_end_marker = input_end_marker

    trades = analyze_seasonal_window(
        df,
        int(analysis_start_marker.month),
        int(analysis_start_marker.day),
        int(analysis_end_marker.month),
        int(analysis_end_marker.day),
        active_years,
    )

    season_result = compute_season_stats(df, trades, active_years, all_years)
    rise_probability = season_result["rise_probability"]
    with stat_col:
        render_season_stats_panel(season_result, key_suffix="main")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    next30_col, next30_stat_col = st.columns([4.35, 1.2])
    real_today = date.today()
    forecast_end = real_today + pd.Timedelta(days=30)

    next30_context_token = f"{symbol.strip()}_{active_years_token}"
    if st.session_state.get("seasonality_next30_context_token") != next30_context_token:
        st.session_state["seasonality_next30_context_token"] = next30_context_token
        st.session_state.pop("seasonality_next30_manual_period", None)
        st.session_state.pop("seasonality_next30_selection_active", None)
        st.session_state.pop("seasonality_next30_just_set_manual_period", None)

    next30_manual_period = st.session_state.get("seasonality_next30_manual_period")
    if next30_manual_period:
        next30_display_start = pd.Timestamp(next30_manual_period[0])
        next30_display_end = pd.Timestamp(next30_manual_period[1])
    else:
        next30_display_start = next30_display_end = None

    next30_chart_selection = None
    with next30_col:
        forecast_rows = []
        for offset in range(31):
            real_date = real_today + pd.Timedelta(days=offset)
            lookup_month, lookup_day = real_date.month, real_date.day
            if lookup_month == 2 and lookup_day == 29:
                lookup_day = 28
            match = chart_curve[(chart_curve["month"] == lookup_month) & (chart_curve["day"] == lookup_day)]
            if match.empty:
                continue
            forecast_rows.append({"date": pd.Timestamp(real_date), "indexed": float(match["indexed_display"].iloc[0])})
        forecast_df = pd.DataFrame(forecast_rows)
        if forecast_df.empty:
            st.info("Keine Daten fuer die naechsten 30 Tage verfuegbar.")
        else:
            forecast_floor = min(float(forecast_df["indexed"].min()), 100.0)
            forecast_ceiling = max(float(forecast_df["indexed"].max()), 100.0)
            forecast_padding = max((forecast_ceiling - forecast_floor) * 0.16, 1.0)
            forecast_label_y = forecast_ceiling + forecast_padding * 0.58

            def forecast_marker_value(marker: pd.Timestamp) -> float:
                nearest_idx = (forecast_df["date"] - marker).abs().idxmin()
                return float(forecast_df.loc[nearest_idx, "indexed"])

            if next30_display_start is not None and next30_display_end is not None:
                next30_start_value = forecast_marker_value(next30_display_start)
                next30_end_value = forecast_marker_value(next30_display_end)
            else:
                next30_start_value = next30_end_value = None

            forecast_fig = go.Figure()
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["indexed"],
                    mode="lines",
                    line={"color": "rgba(98,200,232,.18)", "width": 0},
                    fill="tozeroy",
                    fillcolor="rgba(98,200,232,.13)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["indexed"],
                    mode="lines",
                    name="Next 30 Days",
                    line={"color": "#62c8e8", "width": 2.1, "shape": "spline", "smoothing": 0.55},
                    hovertemplate="%{x|%d %b}<br>Index %{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )
            forecast_fig.add_vline(x=forecast_df["date"].iloc[0], line_color="#c0267a", line_width=2)
            if next30_display_start is not None and next30_display_end is not None:
                forecast_fig.add_vline(x=next30_display_start, line_color="rgba(226,232,240,.82)", line_width=1.3)
                forecast_fig.add_vline(x=next30_display_end, line_color="rgba(226,232,240,.82)", line_width=1.3)
                forecast_fig.add_vrect(x0=next30_display_start, x1=next30_display_end, fillcolor="#62c8e8", opacity=0.11, line_width=0)
                for marker, marker_value in [(next30_display_start, next30_start_value), (next30_display_end, next30_end_value)]:
                    forecast_fig.add_annotation(
                        x=marker,
                        y=forecast_label_y,
                        text=f"{marker.strftime('%d %b')}: {marker_value:.2f}",
                        showarrow=False,
                        bgcolor="rgba(31,41,55,.92)",
                        bordercolor="rgba(226,232,240,.22)",
                        borderpad=4,
                        font={"size": 10, "color": "#dbeafe"},
                    )
            forecast_fig.update_layout(
                **_seasonality_base_layout(f"Seasonal Forecast of {asset_short} — Next 30 Days ({years_text})", 700)
            )
            forecast_fig.update_layout(
                dragmode="select",
                uirevision=f"seasonality_next30_{asset_short}_{active_years_token}",
            )
            forecast_fig.update_xaxes(tickformat="%d %b", showspikes=False, fixedrange=True)
            forecast_fig.update_yaxes(title="", range=[forecast_floor - forecast_padding, forecast_ceiling + forecast_padding])
            next30_chart_selection = st.plotly_chart(
                forecast_fig,
                width="stretch",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": "reset",
                    "staticPlot": False,
                },
                key="seasonality_next30_curve",
                on_select="rerun",
                selection_mode=("box",),
            )
            st.caption("Abschnitt auswaehlen (Box-Select), um Winrate & Bewegung fuer genau dieses Zeitfenster zu sehen.")

    selected_next30_period = parse_absolute_chart_period(next30_chart_selection)
    if selected_next30_period:
        selected_next30_token = tuple(marker.isoformat() for marker in selected_next30_period)
        if selected_next30_token != tuple(st.session_state.get("seasonality_next30_manual_period", ())):
            st.session_state["seasonality_next30_manual_period"] = selected_next30_token
            st.session_state["seasonality_next30_selection_active"] = True
            st.session_state["seasonality_next30_just_set_manual_period"] = selected_next30_token
            st.rerun()
        else:
            st.session_state.pop("seasonality_next30_just_set_manual_period", None)
    elif (
        st.session_state.get("seasonality_next30_selection_active")
        and st.session_state.get("seasonality_next30_manual_period")
        and not active_years_changed
        and chart_selection_is_empty(next30_chart_selection)
    ):
        just_set_next30 = st.session_state.pop("seasonality_next30_just_set_manual_period", None)
        if just_set_next30 != tuple(st.session_state.get("seasonality_next30_manual_period", ())):
            st.session_state.pop("seasonality_next30_manual_period", None)
            st.session_state["seasonality_next30_selection_active"] = False
            st.rerun()

    next30_manual_period = st.session_state.get("seasonality_next30_manual_period")
    if next30_manual_period:
        forecast_start_marker = pd.Timestamp(next30_manual_period[0])
        forecast_end_marker = pd.Timestamp(next30_manual_period[1])
    else:
        forecast_start_marker = pd.Timestamp(real_today)
        forecast_end_marker = pd.Timestamp(forecast_end)

    forecast_trades = analyze_seasonal_window(
        df,
        int(forecast_start_marker.month),
        int(forecast_start_marker.day),
        int(forecast_end_marker.month),
        int(forecast_end_marker.day),
        active_years,
    )

    with next30_stat_col:
        forecast_result = compute_season_stats(df, forecast_trades, active_years, all_years)
        render_season_stats_panel(forecast_result, key_suffix="next30")

    if trades.empty:
        st.warning("Der gewaehlte saisonale Zeitraum enthaelt keine vollstaendigen historischen Pattern-Trades.")
        return

    # ── Bessere Zeiten im Jahr für dieses Muster ─────────────────────────────
    _sl_crosses_year = (analysis_end_marker.month, analysis_end_marker.day) < (analysis_start_marker.month, analysis_start_marker.day)
    _sl_hold_days = 0 if _sl_crosses_year else (analysis_end_marker - analysis_start_marker).days
    if _sl_hold_days > 0:
        _sl_cur_entry_doy = int(analysis_start_marker.dayofyear)
        _sl_key = (
            f"season_alt_scan_{symbol.strip()}_{data_source_label}_"
            f"{'-'.join(str(y) for y in sorted(active_years))}_{_sl_cur_entry_doy}_{_sl_hold_days}"
        )
        if _sl_key not in st.session_state:
            with st.spinner("Suche bessere Zeitfenster im Jahresverlauf …"):
                _sl_year_data: dict = {}
                _sl_sub = df[df.index.year.isin(active_years)]
                for _yr, _grp in _sl_sub.groupby(_sl_sub.index.year):
                    _doys = _grp.index.dayofyear.values.astype(int)
                    _sidx = np.argsort(_doys)
                    _sl_year_data[int(_yr)] = {
                        "doys":   _doys[_sidx],
                        "closes": _grp["close"].values.astype(float)[_sidx],
                    }
                _sl_min_trades = max(3, len(active_years) // 2)
                _sl_rows = []
                for _entry_doy in range(1, 363):
                    _exit_doy = _entry_doy + _sl_hold_days
                    if _exit_doy > 365:
                        continue
                    _rets = []
                    for _yr in active_years:
                        _yd = _sl_year_data.get(int(_yr))
                        if _yd is None:
                            continue
                        _doys = _yd["doys"]
                        _ei = int(np.searchsorted(_doys, _entry_doy))
                        _xi = int(np.searchsorted(_doys, _exit_doy))
                        if _ei >= len(_doys) or _xi >= len(_doys) or _xi <= _ei:
                            continue
                        _ep, _xp = _yd["closes"][_ei], _yd["closes"][_xi]
                        _rets.append((_xp - _ep) / _ep * 100)
                    if len(_rets) >= _sl_min_trades:
                        _rets_arr = np.array(_rets)
                        _sl_rows.append({
                            "entry_doy": _entry_doy, "exit_doy": _exit_doy,
                            "wr": float((_rets_arr > 0).mean() * 100),
                            "avg_ret": float(_rets_arr.mean()),
                            "n": len(_rets),
                        })
                st.session_state[_sl_key] = pd.DataFrame(_sl_rows)
        _sl_scan = st.session_state[_sl_key]

        with st.expander("🔍 Andere Zeiten im Jahr für dieses Muster (gleiche Haltedauer)", expanded=False):
            if _sl_scan is None or _sl_scan.empty:
                st.info("Keine ausreichenden Daten für einen Jahresvergleich gefunden.")
            else:
                _sl_cand = _sl_scan[(_sl_scan["entry_doy"] - _sl_cur_entry_doy).abs() > 10].copy()
                _sl_cand = _sl_cand[_sl_cand["wr"] > rise_probability]
                _sl_cand = _sl_cand.sort_values("wr", ascending=False).head(5)
                if _sl_cand.empty:
                    st.success(
                        f"Keine besseren Zeitfenster gefunden — {period_text} scheint für "
                        f"{symbol.strip()} im Jahresvergleich bereits stark zu sein."
                    )
                else:
                    st.caption(
                        f"Aktuelles Fenster: Rise odds {rise_probability:.1f}% · gesucht: gleiche Haltedauer "
                        f"({_sl_hold_days} Kalendertage), andere Jahreszeit, höhere Trefferquote."
                    )
                    for _si, _srow in _sl_cand.iterrows():
                        _s_start = pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=int(_srow["entry_doy"]) - 1)
                        _s_end   = pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=int(_srow["exit_doy"]) - 1)
                        _s_c1, _s_c2 = st.columns([6, 1])
                        with _s_c1:
                            st.markdown(
                                f"📅 **{_s_start.strftime('%d.%m')} → {_s_end.strftime('%d.%m')}** · "
                                f"Rise odds: **{_srow['wr']:.1f}%** (statt {rise_probability:.1f}%) · "
                                f"Ø Return {_srow['avg_ret']:+.2f}% · n={int(_srow['n'])}"
                            )
                        with _s_c2:
                            if st.button("→ Laden", key=f"sl_alt_load_{_si}_{_sl_key}"):
                                st.session_state.pop("seasonality_manual_period", None)
                                st.session_state.pop("seasonality_chart_selection_active", None)
                                st.session_state.pop("seasonality_just_set_manual_period", None)
                                st.session_state["seasonality_pending_period_text"] = (
                                    f"{_s_start.day:02d}.{_s_start.month:02d} - {_s_end.day:02d}.{_s_end.month:02d}"
                                )
                                st.rerun()

    if is_nasdaq_asset(asset_label, symbol):
        render_mag7_panel()

    selected_df = df[df.index.year.isin(active_years)].copy()
    selected_df = selected_df[~((selected_df.index.month == 2) & (selected_df.index.day == 29))]
    selected_df["daily_return_pct"] = selected_df["close"].pct_change() * 100
    selected_df["weekday"] = selected_df.index.day_name()
    selected_df["month_name"] = selected_df.index.month_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    month_order = list(calendar.month_name)[1:]
    weekday_stats = selected_df.groupby("weekday")["daily_return_pct"].mean().reindex(weekday_order).reset_index()
    weekday_stats.columns = ["Weekday", "Average Return %"]
    monthly_stats = selected_df.groupby("month_name")["daily_return_pct"].mean().reindex(month_order).reset_index()
    monthly_stats.columns = ["Month", "Average Return %"]

    chart_cols = st.columns(2)
    with chart_cols[0]:
        weekday_colors = np.where(weekday_stats["Average Return %"] >= 0, "#62c8e8", "#c25f50")
        weekday_fig = go.Figure(go.Bar(x=weekday_stats["Weekday"], y=weekday_stats["Average Return %"], marker_color=weekday_colors))
        weekday_fig.update_layout(**_seasonality_base_layout("Average Return by Weekday", 320))
        weekday_fig.add_annotation(text="TACO", xref="paper", yref="paper", x=0.5, y=0.52, showarrow=False, font={"size": 42, "color": "rgba(148,163,184,.12)"})
        st.plotly_chart(weekday_fig, width="stretch")
    with chart_cols[1]:
        month_colors = np.where(monthly_stats["Average Return %"] >= 0, "#62c8e8", "#c25f50")
        month_fig = go.Figure(go.Bar(x=monthly_stats["Month"], y=monthly_stats["Average Return %"], marker_color=month_colors))
        month_fig.update_layout(**_seasonality_base_layout("Average Return by Month", 320))
        month_fig.add_annotation(text="TACO", xref="paper", yref="paper", x=0.5, y=0.52, showarrow=False, font={"size": 42, "color": "rgba(148,163,184,.12)"})
        st.plotly_chart(month_fig, width="stretch")

    lower_cols = st.columns(2)
    trades_chronological = trades.sort_values("Year").copy()
    trades_chronological["Cumulative Profit"] = trades_chronological["Profit"].cumsum()
    trades_chronological["Cumulative Profit %"] = trades_chronological["Profit %"].cumsum()
    with lower_cols[0]:
        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(x=trades_chronological["Year"], y=trades_chronological["Cumulative Profit"], mode="lines+markers", name="Points", line={"color": "#22d3ee"}))
        cum_fig.add_trace(go.Scatter(x=trades_chronological["Year"], y=trades_chronological["Cumulative Profit %"], mode="lines+markers", name="Percent", line={"color": "#a78bfa"}))
        cum_fig.update_layout(**_seasonality_base_layout("Cumulative Profit fuer den Zeitraum", 340))
        cum_fig.add_annotation(text="TACO", xref="paper", yref="paper", x=0.5, y=0.52, showarrow=False, font={"size": 42, "color": "rgba(148,163,184,.12)"})
        st.plotly_chart(cum_fig, width="stretch")
    with lower_cols[1]:
        colors = np.where(trades_chronological["Profit %"] >= 0, "#62c8e8", "#c25f50")
        pattern_fig = go.Figure(go.Bar(x=trades_chronological["Year"], y=trades_chronological["Profit %"], marker_color=colors))
        pattern_fig.update_layout(**_seasonality_base_layout("Pattern Returns", 340))
        pattern_fig.add_annotation(text="TACO", xref="paper", yref="paper", x=0.5, y=0.52, showarrow=False, font={"size": 42, "color": "rgba(148,163,184,.12)"})
        st.plotly_chart(pattern_fig, width="stretch")

    with st.spinner("Scanne Top-4 saisonale Long- und Short-Setups..."):
        render_top_seasonal_setups(df, active_years)

    render_ki_analyse(asset_label, symbol)

    trades_display = trades.sort_values("Year", ascending=False).reset_index(drop=True)

    def color_profit_cells(value: float | int | str) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if numeric > 0:
            return "color: #22c55e; font-weight: 700;"
        if numeric < 0:
            return "color: #ef4444; font-weight: 700;"
        return ""

    st.subheader("Pattern Trades")
    styled_trades = trades_display.style.map(color_profit_cells, subset=["Profit", "Profit %"])
    st.dataframe(styled_trades, width="stretch")

    downloads = st.columns(3)
    downloads[0].download_button(
        "Seasonality Trades CSV",
        trades_display.to_csv(index=False).encode("utf-8"),
        "seasonality_trades.csv",
        "text/csv",
    )
    downloads[1].download_button(
        "Seasonal Curve CSV",
        curve.to_csv(index=False).encode("utf-8"),
        "seasonal_curve.csv",
        "text/csv",
    )
    stats_csv = pd.concat(
        [
            monthly_stats.assign(Type="Month").rename(columns={"Month": "Bucket"}),
            weekday_stats.assign(Type="Weekday").rename(columns={"Weekday": "Bucket"}),
        ],
        ignore_index=True,
    )
    downloads[2].download_button(
        "Monthly / Weekday Stats CSV",
        stats_csv.to_csv(index=False).encode("utf-8"),
        "seasonality_month_weekday_stats.csv",
        "text/csv",
    )


ASSET_PRESETS = {
    "UK100 proxy: FTSE 100 Index (^FTSE)": "^FTSE",
    "GER40 proxy: DAX Index (^GDAXI)": "^GDAXI",
    "US100 proxy: Nasdaq 100 (^NDX)": "^NDX",
    "S&P500 / US500 proxy: S&P 500 (^GSPC)": "^GSPC",
    "US30 proxy: Dow Jones Industrial Average (^DJI)": "^DJI",
    "EURUSD (EURUSD=X)": "EURUSD=X",
    "GBPUSD (GBPUSD=X)": "GBPUSD=X",
    "NZDUSD (NZDUSD=X)": "NZDUSD=X",
    "USDCAD (CAD=X)": "CAD=X",
    "USDCHF (CHF=X)": "CHF=X",
    "AUDUSD (AUDUSD=X)": "AUDUSD=X",
    "USDJPY (JPY=X)": "JPY=X",
    "EURGBP (EURGBP=X)": "EURGBP=X",
    "EURNZD (EURNZD=X)": "EURNZD=X",
    "EURCAD (EURCAD=X)": "EURCAD=X",
    "GBPNZD (GBPNZD=X)": "GBPNZD=X",
    "GBPCAD (GBPCAD=X)": "GBPCAD=X",
    "EURAUD (EURAUD=X)": "EURAUD=X",
    "GBPAUD (GBPAUD=X)": "GBPAUD=X",
    "AUDNZD (AUDNZD=X)": "AUDNZD=X",
    "AUDCAD (AUDCAD=X)": "AUDCAD=X",
    "AUDJPY (AUDJPY=X)": "AUDJPY=X",
    "NZDCAD (NZDCAD=X)": "NZDCAD=X",
    "NZDJPY (NZDJPY=X)": "NZDJPY=X",
    "NZDCHF (NZDCHF=X)": "NZDCHF=X",
    "AUDCHF (AUDCHF=X)": "AUDCHF=X",
    "GBPJPY (GBPJPY=X)": "GBPJPY=X",
    "CHFJPY (CHFJPY=X)": "CHFJPY=X",
    "EURJPY (EURJPY=X)": "EURJPY=X",
    "DXY proxy: US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
    "Gold futures (GC=F)": "GC=F",
    "Silver futures (SI=F)": "SI=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
}


COMPARISON_PRESETS = {
    "DXY proxy: US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
    "Gold futures (GC=F)": "GC=F",
    "10Y Treasury Note futures (ZN=F)": "ZN=F",
    "Euro FX futures (6E=F)": "6E=F",
    "Silver futures (SI=F)": "SI=F",
    "Oil futures (CL=F)": "CL=F",
}


def classify_fear_greed(score: float) -> str:
    if score <= 24:
        return "Extreme Fear"
    if score <= 44:
        return "Fear"
    if score <= 55:
        return "Neutral"
    if score <= 75:
        return "Greed"
    return "Extreme Greed"


@st.cache_data(ttl=30 * 60)
def load_fear_greed() -> dict | None:
    try:
        import requests

        today = date.today().isoformat()
        urls = [
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}",
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
            "Origin": "https://edition.cnn.com",
            "Referer": "https://edition.cnn.com/markets/fear-and-greed",
            "Sec-Fetch-Site": "same-site",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
        }
        payload = None
        for url in urls:
            response = requests.get(url, headers=headers, timeout=8)
            if response.ok:
                payload = response.json()
                break
        if not payload:
            return None

        current = payload.get("fear_and_greed", {})
        if not current and "data" in payload:
            current = payload["data"].get("fear_and_greed", {})
        score = current.get("score")
        if isinstance(score, dict):
            score = score.get("value")
        if score is None:
            hist = payload.get("fear_and_greed_historical", {}).get("data", [])
            if hist:
                score = hist[-1].get("y")
        if score is None:
            return None

        score = float(score)
        rating = current.get("rating") or current.get("status") or classify_fear_greed(score)
        rating = str(rating).replace("_", " ").title()
        return {
            "score": score,
            "rating": rating,
            "previous_close": current.get("previous_close"),
            "previous_1_week": current.get("previous_1_week"),
            "previous_1_month": current.get("previous_1_month"),
            "previous_1_year": current.get("previous_1_year"),
            "updated": current.get("timestamp") or current.get("last_updated"),
        }
    except Exception:
        return None


def render_fear_greed_panel() -> None:
    data = load_fear_greed()
    st.subheader("CNN Fear & Greed Index")
    if not data:
        st.warning("Fear & Greed konnte gerade nicht geladen werden. CNN kann externe Requests zeitweise blockieren.")
        st.markdown("[CNN Fear & Greed Index oeffnen](https://edition.cnn.com/markets/fear-and-greed)")
        return

    score = data["score"]
    rating = data["rating"]
    gauge_color = "#22c55e" if score > 55 else "#f59e0b" if score >= 45 else "#ef4444"

    left, right = st.columns([1.15, 2])
    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"size": 44}},
                title={"text": rating},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(239,68,68,.30)"},
                        {"range": [25, 45], "color": "rgba(249,115,22,.24)"},
                        {"range": [45, 55], "color": "rgba(148,163,184,.24)"},
                        {"range": [55, 75], "color": "rgba(132,204,22,.24)"},
                        {"range": [75, 100], "color": "rgba(34,197,94,.30)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": score},
                },
            )
        )
        fig.update_layout(height=220, margin=dict(l=16, r=16, t=36, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cols = st.columns(5)
        cols[0].metric("Now", f"{score:.0f}", rating)
        cols[1].metric("Prev Close", "n/a" if data["previous_close"] is None else f"{float(data['previous_close']):.0f}")
        cols[2].metric("1 Week", "n/a" if data["previous_1_week"] is None else f"{float(data['previous_1_week']):.0f}")
        cols[3].metric("1 Month", "n/a" if data["previous_1_month"] is None else f"{float(data['previous_1_month']):.0f}")
        cols[4].metric("1 Year", "n/a" if data["previous_1_year"] is None else f"{float(data['previous_1_year']):.0f}")
        st.caption("Separates Marktstimmungs-Panel. Es beeinflusst den TACO Backtest nicht.")
        st.markdown("[Quelle: CNN Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed)")


COT_WATCHLIST = [
    ("S&P500 Futures", ["E-MINI S&P 500", "S&P 500 STOCK INDEX", "S&P 500"]),
    ("US30 Futures", ["DOW JONES", "DJIA"]),
    ("NQ Futures", ["NASDAQ-100 Consolidated", "NASDAQ MINI", "MICRO E-MINI NASDAQ-100"]),
    ("EURO Futures", ["EURO FX"]),
    ("CANADA Futures", ["CANADIAN DOLLAR"]),
    ("YEN Futures", ["JAPANESE YEN"]),
    ("CHF Futures", ["SWISS FRANC"]),
    ("Pfund Futures", ["BRITISH POUND"]),
    ("AUD Futures", ["AUSTRALIAN DOLLAR"]),
    ("NZD Futures", ["NEW ZEALAND DOLLAR"]),
    ("DXY", ["U.S. DOLLAR INDEX", "US DOLLAR INDEX", "DOLLAR INDEX"]),
    ("Gold", ["GOLD"]),
    ("Silver", ["SILVER"]),
    ("Copper", ["COPPER"]),
    ("Platinum", ["PLATINUM"]),
]


def infer_cot_query_from_asset(asset_label: str | None, asset_symbol: str | None) -> tuple[str, str]:
    text = f"{asset_label or ''} {asset_symbol or ''}".upper()
    if "US100" in text or "NASDAQ" in text or "^NDX" in text:
        return "NASDAQ-100 Consolidated", "US100/Nasdaq proxy"
    if "S&P500" in text or "US500" in text or "S&P 500" in text or "^GSPC" in text:
        return "E-MINI S&P 500", "S&P500/US500 proxy"
    if "US30" in text or "DOW" in text or "^DJI" in text:
        return "DOW JONES", "US30/Dow proxy"
    if "EURUSD" in text or "EURO" in text or "6E" in text:
        return "EURO FX", "EURUSD proxy"
    if "GBPUSD" in text or "BRITISH POUND" in text or "6B" in text:
        return "BRITISH POUND", "GBPUSD proxy"
    if "AUDUSD" in text or "AUSTRALIAN" in text or "6A" in text:
        return "AUSTRALIAN DOLLAR", "AUDUSD proxy"
    if "NZD" in text or "NEW ZEALAND" in text:
        return "NEW ZEALAND DOLLAR", "NZD proxy"
    if "USDCAD" in text or "CAD=X" in text or "CANADIAN" in text or "6C" in text:
        return "CANADIAN DOLLAR", "USDCAD proxy"
    if "USDCHF" in text or "CHF=X" in text or "SWISS" in text or "6S" in text:
        return "SWISS FRANC", "USDCHF proxy"
    if "USDJPY" in text or "JPY=X" in text or "JAPANESE" in text or "6J" in text:
        return "JAPANESE YEN", "USDJPY proxy"
    if "UK100" in text or "FTSE" in text or "GER40" in text or "DAX" in text:
        return "E-MINI S&P 500", "Risk proxy for UK100/GER40"
    return "E-MINI S&P 500", "default risk proxy"


def cot_bias_label(score: float) -> str:
    if score >= 65:
        return "Strong Long"
    if score >= 55:
        return "Long"
    if score > 45:
        return "Neutral"
    if score > 35:
        return "Short"
    return "Strong Short"


@st.cache_data(ttl=12 * 60 * 60)
def load_cot_cme_legacy() -> tuple[pd.DataFrame, str | None]:
    try:
        from io import StringIO

        import requests

        url = "https://www.cftc.gov/dea/newcot/deafut.txt"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
        rows = pd.DataFrame({
            "market": raw["Market_and_Exchange_Names"].astype(str).str.strip(),
            "code": raw.get("CFTC_Contract_Market_Code", ""),
            "open_interest": pd.to_numeric(raw["Open_Interest_All"], errors="coerce"),
            "noncomm_long": pd.to_numeric(raw["Noncommercial_Positions_Long_All"], errors="coerce"),
            "noncomm_short": pd.to_numeric(raw["Noncommercial_Positions_Short_All"], errors="coerce"),
            "noncomm_spread": pd.to_numeric(raw["Noncommercial_Positions_Spread_All"], errors="coerce"),
            "comm_long": pd.to_numeric(raw["Commercial_Positions_Long_All"], errors="coerce"),
            "comm_short": pd.to_numeric(raw["Commercial_Positions_Short_All"], errors="coerce"),
            "total_long": pd.to_numeric(raw["Total_Reportable_Positions_Long_All"], errors="coerce"),
            "total_short": pd.to_numeric(raw["Total_Reportable_Positions_Short_All"], errors="coerce"),
            "retail_long": pd.to_numeric(raw["Nonreportable_Positions_Long_All"], errors="coerce"),
            "retail_short": pd.to_numeric(raw["Nonreportable_Positions_Short_All"], errors="coerce"),
        })
        report_date = None
        if "Report_Date_as_YYYY-MM-DD" in raw.columns and not raw.empty:
            report_date = str(raw["Report_Date_as_YYYY-MM-DD"].iloc[0])
        return rows.dropna(subset=["open_interest"]), report_date
    except Exception:
        pass

    try:
        import html
        import re
        import requests

        url = "https://www.cftc.gov/dea/futures/deacmesf.htm"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        response.raise_for_status()
        text = html.unescape(response.text)
        pre_match = re.search(r"<pre>(.*?)</pre>", text, flags=re.I | re.S)
        report = pre_match.group(1) if pre_match else text
        blocks = re.split(r"\n(?=[A-Z0-9][A-Za-z0-9/&., \-\(\)]+ - CHICAGO MERCANTILE EXCHANGE\s+Code-)", report)
        rows = []
        report_date = None
        for block in blocks:
            header = re.search(r"^\s*(.*?)\s+- CHICAGO MERCANTILE EXCHANGE\s+Code-([A-Z0-9+]+)", block, flags=re.M)
            if not header:
                continue
            date_match = re.search(r"POSITIONS AS OF\s+([0-9/]+)", block)
            if date_match:
                report_date = date_match.group(1)
            oi_match = re.search(r"OPEN INTEREST:\s*([0-9,]+)", block)
            nums = re.search(
                r"COMMITMENTS\s*\n\s*([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)\s+([0-9,\-]+)",
                block,
            )
            if not nums:
                continue

            def to_int(value: str) -> int:
                return int(value.replace(",", ""))

            values = [to_int(x) for x in nums.groups()]
            rows.append({
                "market": header.group(1).strip(),
                "code": header.group(2).strip(),
                "open_interest": to_int(oi_match.group(1)) if oi_match else np.nan,
                "noncomm_long": values[0],
                "noncomm_short": values[1],
                "noncomm_spread": values[2],
                "comm_long": values[3],
                "comm_short": values[4],
                "total_long": values[5],
                "total_short": values[6],
                "retail_long": values[7],
                "retail_short": values[8],
            })
        return pd.DataFrame(rows), report_date
    except Exception:
        return pd.DataFrame(), None


def cot_group_stats(row: pd.Series, long_col: str, short_col: str) -> dict:
    long_value = float(row[long_col])
    short_value = float(row[short_col])
    total = long_value + short_value
    score = long_value / total * 100 if total > 0 else 50.0
    net = long_value - short_value
    return {
        "long": long_value,
        "short": short_value,
        "net": net,
        "score": score,
        "label": cot_bias_label(score),
    }


def render_cot_gauge(title: str, stats: dict) -> None:
    score = stats["score"]
    label = stats["label"]
    color = "#22c55e" if score > 55 else "#ef4444" if score < 45 else "#94a3b8"
    st.markdown(f"### {title}")
    st.markdown(f"**{label}**")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": ""},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "ticktext": ["Short", "25", "Neutral", "75", "Long"]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "rgba(239,68,68,.30)"},
                    {"range": [35, 45], "color": "rgba(249,115,22,.22)"},
                    {"range": [45, 55], "color": "rgba(148,163,184,.25)"},
                    {"range": [55, 65], "color": "rgba(132,204,22,.22)"},
                    {"range": [65, 100], "color": "rgba(34,197,94,.30)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": score},
            },
        )
    )
    fig.update_layout(height=225, margin=dict(l=8, r=8, t=18, b=8))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
        **Long:** {stats['long']:,.0f}  
        **Short:** {stats['short']:,.0f}  
        **Net:** {stats['net']:,.0f}  
        **Bereinigt:** Long / (Long + Short) = {score:.1f}%
        """
    )


def find_cot_market(markets: list[str], queries: list[str]) -> str | None:
    for query in queries:
        match = next((m for m in markets if query.upper() in m.upper()), None)
        if match:
            return match
    return None


def build_cot_options(markets: list[str]) -> dict[str, str]:
    options = {}
    for label, queries in COT_WATCHLIST:
        market = find_cot_market(markets, queries)
        if market:
            options[label] = market
    return options


def render_cot_panel(auto_match: bool, asset_label: str | None, asset_symbol: str | None) -> list[str]:
    data, report_date = load_cot_cme_legacy()
    st.subheader("CFTC COT Positioning")
    if data.empty:
        st.warning("COT-Daten konnten gerade nicht geladen werden. Die CFTC-Seite kann externe Requests zeitweise blockieren.")
        st.markdown("[CFTC Commitments of Traders oeffnen](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)")
        return []

    markets = data["market"].tolist()
    options = build_cot_options(markets)
    if not options:
        st.warning("Die gewuenschte COT-Watchlist wurde in der aktuellen CFTC-Datei nicht gefunden.")
        return markets
    default_label = "S&P500 Futures" if "S&P500 Futures" in options else next(iter(options))
    inferred_query, inferred_reason = infer_cot_query_from_asset(asset_label, asset_symbol)
    inferred_market = find_cot_market(markets, [inferred_query]) or options[default_label]
    inferred_label = next((label for label, market in options.items() if market == inferred_market), default_label)
    default_selection = inferred_label if auto_match else default_label
    selected_label = st.selectbox(
        "COT Market",
        list(options.keys()),
        index=list(options.keys()).index(default_selection) if default_selection in options else 0,
        disabled=auto_match,
        help="COT-Daten sind Wochen-Daten. Die Auswahl betrifft nur das Positionierungs-Panel, nicht TACO.",
        key="cot_market_select",
    )
    selected_market = options[selected_label]
    if auto_match:
        selected_market = inferred_market
        selected_label = inferred_label
        st.caption(f"Auto-match aktiv: {inferred_reason} -> {selected_label} ({selected_market}). Fuer UK100/GER40 ist das ein Risk-Proxy, kein direkter CFD-COT-Markt.")
    row = data.loc[data["market"] == selected_market].iloc[0]
    st.caption(
        f"Wochenbasierte COT-Daten, Report-Datum: {report_date or 'n/a'} | Auswahl: {selected_label} | CFTC-Markt: {selected_market} | "
        f"Open Interest: {row['open_interest']:,.0f}. Separates Positionierungs-Panel, nicht Teil der TACO-Logik."
    )

    noncomm = cot_group_stats(row, "noncomm_long", "noncomm_short")
    comm = cot_group_stats(row, "comm_long", "comm_short")
    retail = cot_group_stats(row, "retail_long", "retail_short")

    cols = st.columns(3)
    with cols[0]:
        render_cot_gauge("Non Commercials", noncomm)
        st.write("Spekulative grosse Marktteilnehmer. Long-Bias bedeutet, dass diese Gruppe netto eher auf steigende Kurse positioniert ist.")
    with cols[1]:
        render_cot_gauge("Commercials", comm)
        st.write("Hedger/Commercials. Sie sind oft gegenlaeufig zu Spekulanten positioniert; die Anzeige zeigt trotzdem rein die bereinigte Long/Short-Balance.")
    with cols[2]:
        render_cot_gauge("Retail Trader", retail)
        st.write("Non-Reportable Positions. Das sind kleinere, nicht meldepflichtige Positionen, hier als Retail Trader zusammengefasst.")

    st.markdown("[Quelle: CFTC Commitments of Traders](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)")
    return markets


def tanh_bounded(x: pd.Series) -> pd.Series:
    clipped = x.clip(-10, 10).fillna(0)
    expv = np.exp(2 * clipped)
    return (expv - 1) / (expv + 1)


def calculate_oscillator(asset: pd.DataFrame, comp: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    df = asset.join(comp[["close"]].rename(columns={"close": "comp_close"}), how="inner")
    ratio = df["close"] / df["comp_close"]
    ratio_mean = ratio.rolling(settings.cycle_length).mean()
    ratio_std = ratio.rolling(settings.cycle_length).std()
    ratio_z = (ratio - ratio_mean) / ratio_std

    ret_asset = np.log(df["close"] / df["close"].shift(1))
    ret_comp = np.log(df["comp_close"] / df["comp_close"].shift(1))
    spread = ret_asset - ret_comp
    spread_z = spread.ewm(span=settings.cycle_length, adjust=False).mean() / spread.rolling(settings.cycle_length).std()

    score = ratio_z if settings.mode == "Ratio Z-Score" else spread_z
    osc_raw = 100 * tanh_bounded(score / settings.softness)
    df["osc"] = osc_raw.ewm(span=settings.smoothing, adjust=False).mean()
    return df.dropna()


def backtest(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    equity = settings.initial_capital
    position = None
    trades = []
    equity_rows = []

    years = (df.index.year >= settings.start_year) & (df.index.year <= settings.end_year)
    data = df.loc[years].copy()

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i - 1]
        date = data.index[i]

        if position:
            position["high"] = max(position["high"], row["high"])
            position["low"] = min(position["low"], row["low"])

            side = position["side"]
            exit_price = None
            exit_reason = None

            if side == "Long":
                hit_stop = row["low"] <= position["stop"]
                hit_tp = not math.isnan(position["tp"]) and row["high"] >= position["tp"]
                if hit_stop:
                    exit_price, exit_reason = position["stop"], "Stop Loss"
                elif hit_tp:
                    exit_price, exit_reason = position["tp"], "Take Profit"
                elif settings.exit_on_zero and prev["osc"] < 0 <= row["osc"]:
                    exit_price, exit_reason = row["close"], "Zero Exit"
            else:
                hit_stop = row["high"] >= position["stop"]
                hit_tp = not math.isnan(position["tp"]) and row["low"] <= position["tp"]
                if hit_stop:
                    exit_price, exit_reason = position["stop"], "Stop Loss"
                elif hit_tp:
                    exit_price, exit_reason = position["tp"], "Take Profit"
                elif settings.exit_on_zero and prev["osc"] > 0 >= row["osc"]:
                    exit_price, exit_reason = row["close"], "Zero Exit"

            if settings.time_exit and i - position["bar"] >= settings.exit_after_bars and exit_price is None:
                exit_price, exit_reason = row["close"], "Time Exit"

            if exit_price is not None:
                if side == "Long":
                    exit_price = exit_price * (1 - settings.slippage_pct / 100)
                else:
                    exit_price = exit_price * (1 + settings.slippage_pct / 100)
                pnl_points = exit_price - position["entry"] if side == "Long" else position["entry"] - exit_price
                gross = pnl_points * position["qty"]
                commission = settings.commission_pct / 100 * (position["entry"] * position["qty"] + exit_price * position["qty"])
                pnl = gross - commission
                equity += pnl
                mae = ((position["low"] - position["entry"]) / position["entry"] * 100) if side == "Long" else ((position["entry"] - position["high"]) / position["entry"] * 100)
                mfe = ((position["high"] - position["entry"]) / position["entry"] * 100) if side == "Long" else ((position["entry"] - position["low"]) / position["entry"] * 100)
                realized_pct = pnl_points / position["entry"] * 100
                risk_points = abs(position["entry"] - position["stop"])
                r_multiple = pnl_points / risk_points if risk_points > 0 else np.nan
                stop_breach_pct = 0.0
                if side == "Long":
                    stop_breach_pct = max(0.0, (position["stop"] - position["low"]) / position["entry"] * 100)
                else:
                    stop_breach_pct = max(0.0, (position["high"] - position["stop"]) / position["entry"] * 100)
                trades.append({
                    "entry_date": position["date"],
                    "exit_date": date,
                    "side": side,
                    "entry": position["entry"],
                    "exit": exit_price,
                    "reason": exit_reason,
                    "qty": position["qty"],
                    "pnl": pnl,
                    "pnl_pct_equity": pnl / max(settings.initial_capital, 1) * 100,
                    "realized_pct": realized_pct,
                    "r_multiple": r_multiple,
                    "mae_pct": mae,
                    "mfe_pct": mfe,
                    "stop_breach_pct": stop_breach_pct,
                    "bars": i - position["bar"],
                })
                position = None

        if position is None:
            long_signal = row["osc"] < settings.lower and row["osc"] > prev["osc"] or prev["osc"] < settings.lower <= row["osc"]
            short_signal = row["osc"] > settings.upper and row["osc"] < prev["osc"] or prev["osc"] > settings.upper >= row["osc"]
            allow_long = settings.trade_direction in ["Long & Short", "Long Only"]
            allow_short = settings.trade_direction in ["Long & Short", "Short Only"]

            if long_signal and allow_long:
                entry = row["close"] * (1 + settings.slippage_pct / 100)
                stop = entry * (1 - settings.stop_pct / 100)
                risk_cash = equity * settings.risk_pct / 100
                qty = risk_cash / max(entry - stop, 1e-9)
                risk_points = entry - stop
                tp = entry + risk_points * settings.rr if settings.tp_mode == "Risk Reward" else entry * (1 + settings.fixed_tp_pct / 100) if settings.tp_mode == "Fixed %" else math.nan
                position = {"side": "Long", "entry": entry, "stop": stop, "tp": tp, "qty": qty, "date": date, "bar": i, "high": row["high"], "low": row["low"]}
            elif short_signal and allow_short:
                entry = row["close"] * (1 - settings.slippage_pct / 100)
                stop = entry * (1 + settings.stop_pct / 100)
                risk_cash = equity * settings.risk_pct / 100
                qty = risk_cash / max(stop - entry, 1e-9)
                risk_points = stop - entry
                tp = entry - risk_points * settings.rr if settings.tp_mode == "Risk Reward" else entry * (1 - settings.fixed_tp_pct / 100) if settings.tp_mode == "Fixed %" else math.nan
                position = {"side": "Short", "entry": entry, "stop": stop, "tp": tp, "qty": qty, "date": date, "bar": i, "high": row["high"], "low": row["low"]}

        equity_rows.append({"date": date, "equity": equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame(columns=["equity"])

    if trades_df.empty:
        metrics = {
            "Trades": 0,
            "Winrate": np.nan,
            "Profit Factor": np.nan,
            "Net Profit": 0,
            "Max DD": np.nan,
            "Intratrade MAE": np.nan,
            "Avg MFE": np.nan,
            "Avg Realized Win": np.nan,
            "Avg Realized Loss": np.nan,
            "Avg R": np.nan,
            "Expectancy R": np.nan,
            "Max Loss Streak": 0,
            "Stop Breach Count": 0,
            "Stop Breach Avg": np.nan,
        }
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        dd = equity_df["equity"] / equity_df["equity"].cummax() - 1
        loss_flags = (trades_df["pnl"] <= 0).astype(int).tolist()
        max_loss_streak = 0
        current_loss_streak = 0
        for flag in loss_flags:
            if flag:
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_loss_streak = 0
        stop_breaches = trades_df[trades_df["stop_breach_pct"] > 0]
        metrics = {
            "Trades": len(trades_df),
            "Winrate": len(wins) / len(trades_df) * 100,
            "Profit Factor": gross_profit / gross_loss if gross_loss else np.nan,
            "Net Profit": equity - settings.initial_capital,
            "Max DD": dd.min() * 100,
            "Intratrade MAE": trades_df["mae_pct"].mean(),
            "Avg MFE": trades_df["mfe_pct"].mean(),
            "Avg Realized Win": wins["realized_pct"].mean() if not wins.empty else np.nan,
            "Avg Realized Loss": losses["realized_pct"].mean() if not losses.empty else np.nan,
            "Avg R": trades_df["r_multiple"].mean(),
            "Expectancy R": trades_df["r_multiple"].mean(),
            "Max Loss Streak": max_loss_streak,
            "Stop Breach Count": len(stop_breaches),
            "Stop Breach Avg": stop_breaches["stop_breach_pct"].mean() if not stop_breaches.empty else 0.0,
        }
    return trades_df, equity_df, metrics


def optimize_in_sample(
    asset: pd.DataFrame,
    comp: pd.DataFrame,
    base_settings: Settings,
    cycles: list[int],
    stop_values: list[float],
    in_start_year: int,
    in_end_year: int,
    min_trades: int,
    max_loss_streak: int,
) -> tuple[dict | None, pd.DataFrame]:
    """Optimize only inside the in-sample window. The OOS year is not touched here."""
    rows = []
    for cycle in cycles:
        for stop_pct in stop_values:
            test_settings = Settings(
                cycle_length=int(cycle),
                smoothing=base_settings.smoothing,
                softness=base_settings.softness,
                mode=base_settings.mode,
                trade_direction=base_settings.trade_direction,
                start_year=int(in_start_year),
                end_year=int(in_end_year),
                upper=base_settings.upper,
                lower=base_settings.lower,
                risk_pct=base_settings.risk_pct,
                stop_pct=float(stop_pct),
                tp_mode=base_settings.tp_mode,
                rr=base_settings.rr,
                fixed_tp_pct=base_settings.fixed_tp_pct,
                exit_on_zero=base_settings.exit_on_zero,
                time_exit=base_settings.time_exit,
                exit_after_bars=base_settings.exit_after_bars,
                initial_capital=base_settings.initial_capital,
                commission_pct=base_settings.commission_pct,
                slippage_pct=base_settings.slippage_pct,
            )
            df = calculate_oscillator(asset, comp, test_settings)
            _, _, metrics = backtest(df, test_settings)
            if (
                metrics["Trades"] < min_trades
                or metrics["Max Loss Streak"] > max_loss_streak
                or pd.isna(metrics["Profit Factor"])
            ):
                continue
            rows.append({
                "Cycle Length": int(cycle),
                "Stop Loss %": float(stop_pct),
                "Trades": metrics["Trades"],
                "Winrate": metrics["Winrate"],
                "Profit Factor": metrics["Profit Factor"],
                "Net Profit": metrics["Net Profit"],
                "Max DD": metrics["Max DD"],
                "Expectancy R": metrics["Expectancy R"],
                "Max Loss Streak": metrics["Max Loss Streak"],
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return None, results

    results = results.sort_values(
        ["Profit Factor", "Expectancy R", "Net Profit", "Max DD"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return results.iloc[0].to_dict(), results


def evaluate_oos_year(
    asset: pd.DataFrame,
    comp: pd.DataFrame,
    base_settings: Settings,
    best_params: dict,
    oos_year: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Apply the in-sample-selected parameters to exactly one out-of-sample year."""
    oos_settings = Settings(
        cycle_length=int(best_params["Cycle Length"]),
        smoothing=base_settings.smoothing,
        softness=base_settings.softness,
        mode=base_settings.mode,
        trade_direction=base_settings.trade_direction,
        start_year=int(oos_year),
        end_year=int(oos_year),
        upper=base_settings.upper,
        lower=base_settings.lower,
        risk_pct=base_settings.risk_pct,
        stop_pct=float(best_params["Stop Loss %"]),
        tp_mode=base_settings.tp_mode,
        rr=base_settings.rr,
        fixed_tp_pct=base_settings.fixed_tp_pct,
        exit_on_zero=base_settings.exit_on_zero,
        time_exit=base_settings.time_exit,
        exit_after_bars=base_settings.exit_after_bars,
        initial_capital=base_settings.initial_capital,
        commission_pct=base_settings.commission_pct,
        slippage_pct=base_settings.slippage_pct,
    )
    df = calculate_oscillator(asset, comp, oos_settings)
    trades, equity, metrics = backtest(df, oos_settings)
    return metrics, trades, equity


def run_walk_forward(
    asset: pd.DataFrame,
    comp: pd.DataFrame,
    base_settings: Settings,
    wf_start_year: int,
    wf_end_year: int,
    in_sample_years: int,
    cycles: list[int],
    stop_values: list[float],
    min_trades: int,
    max_loss_streak: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run rolling walk-forward optimization without using the OOS year in-sample."""
    yearly_rows = []
    trade_rows = []
    optimization_rows = []

    for oos_year in range(int(wf_start_year), int(wf_end_year) + 1):
        in_end = oos_year - 1
        in_start = oos_year - int(in_sample_years)

        best_params, opt_results = optimize_in_sample(
            asset,
            comp,
            base_settings,
            cycles,
            stop_values,
            in_start,
            in_end,
            min_trades,
            max_loss_streak,
        )

        if not opt_results.empty:
            opt_results = opt_results.copy()
            opt_results["OOS Year"] = oos_year
            opt_results["In Sample"] = f"{in_start}-{in_end}"
            optimization_rows.append(opt_results)

        if best_params is None:
            yearly_rows.append({
                "OOS Jahr": oos_year,
                "In Sample Zeitraum": f"{in_start}-{in_end}",
                "Cycle Length": np.nan,
                "Stop Loss %": np.nan,
                "Trades": 0,
                "Winrate": np.nan,
                "Profit Factor": np.nan,
                "Net Profit": 0.0,
                "Max DD": np.nan,
                "Expectancy R": np.nan,
                "Max Loss Streak": np.nan,
                "Status": "No valid in-sample params",
            })
            continue

        metrics, trades, _ = evaluate_oos_year(asset, comp, base_settings, best_params, oos_year)
        yearly_rows.append({
            "OOS Jahr": oos_year,
            "In Sample Zeitraum": f"{in_start}-{in_end}",
            "Cycle Length": int(best_params["Cycle Length"]),
            "Stop Loss %": float(best_params["Stop Loss %"]),
            "Trades": metrics["Trades"],
            "Winrate": metrics["Winrate"],
            "Profit Factor": metrics["Profit Factor"],
            "Net Profit": metrics["Net Profit"],
            "Max DD": metrics["Max DD"],
            "Expectancy R": metrics["Expectancy R"],
            "Max Loss Streak": metrics["Max Loss Streak"],
            "Status": "OK",
        })

        if not trades.empty:
            oos_trades = trades.copy()
            oos_trades.insert(0, "OOS Jahr", oos_year)
            trade_rows.append(oos_trades[["OOS Jahr", "entry_date", "exit_date", "side", "entry", "exit", "pnl", "r_multiple"]])

    yearly = pd.DataFrame(yearly_rows)
    wf_trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(
        columns=["OOS Jahr", "entry_date", "exit_date", "side", "entry", "exit", "pnl", "r_multiple"]
    )
    optimization = pd.concat(optimization_rows, ignore_index=True) if optimization_rows else pd.DataFrame()

    if wf_trades.empty:
        equity_curve = pd.DataFrame(columns=["date", "equity"])
        summary = {
            "OOS Jahre": len(yearly),
            "OOS Trades": 0,
            "Avg Profit Factor": np.nan,
            "Avg Expectancy R": np.nan,
            "Walk Forward Winrate": np.nan,
            "Walk Forward Max DD": np.nan,
        }
    else:
        wf_trades = wf_trades.sort_values("exit_date").reset_index(drop=True)
        equity_curve = pd.DataFrame({
            "date": wf_trades["exit_date"],
            "equity": base_settings.initial_capital + wf_trades["pnl"].cumsum(),
        })
        dd = equity_curve["equity"] / equity_curve["equity"].cummax() - 1
        wins = wf_trades[wf_trades["pnl"] > 0]
        summary = {
            "OOS Jahre": len(yearly),
            "OOS Trades": len(wf_trades),
            "Avg Profit Factor": yearly["Profit Factor"].replace([np.inf, -np.inf], np.nan).mean(),
            "Avg Expectancy R": yearly["Expectancy R"].mean(),
            "Walk Forward Winrate": len(wins) / len(wf_trades) * 100,
            "Walk Forward Max DD": dd.min() * 100,
        }

    return yearly, wf_trades, equity_curve, summary


def plot_backtest_charts(df: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, settings: Settings) -> None:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    if not trades.empty:
        longs = trades[trades["side"] == "Long"]
        shorts = trades[trades["side"] == "Short"]
        if not longs.empty:
            fig.add_trace(go.Scatter(x=longs["entry_date"], y=longs["entry"], mode="markers", name="Long", marker=dict(color="#2f6bff", symbol="triangle-up", size=10)))
        if not shorts.empty:
            fig.add_trace(go.Scatter(x=shorts["entry_date"], y=shorts["entry"], mode="markers", name="Short", marker=dict(color="#ff3b3b", symbol="triangle-down", size=10)))
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    osc_fig = go.Figure()
    osc_fig.add_trace(go.Scatter(x=df.index, y=df["osc"], mode="lines", name="TACO Oscillator", line=dict(color="#bd37dc", width=2)))
    osc_fig.add_hline(y=settings.upper, line_dash="dash", line_color="rgba(255,0,0,.45)")
    osc_fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,100,100,.45)")
    osc_fig.add_hline(y=settings.lower, line_dash="dash", line_color="rgba(0,200,90,.45)")
    osc_fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[-110, 110]))
    st.plotly_chart(osc_fig, use_container_width=True)

    if not equity.empty:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=equity.index, y=equity["equity"], mode="lines", name="Equity", line=dict(color="#2aa889", width=2)))
        equity_fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(equity_fig, use_container_width=True)


EDGE_HOLDING_PERIODS = [5, 20, 60]
EDGE_BOOTSTRAP_RUNS = 10_000
EDGE_RANDOM_RUNS = 1_000


def find_taco_reversal_events(df: pd.DataFrame, settings: Settings) -> dict[str, np.ndarray]:
    data = df[(df.index.year >= settings.start_year) & (df.index.year <= settings.end_year)].copy()
    if len(data) < 2:
        return {"Long": np.array([], dtype=int), "Short": np.array([], dtype=int)}

    osc = data["osc"].to_numpy(dtype=float)
    long_positions = np.flatnonzero((osc[:-1] < settings.lower) & (osc[1:] > settings.lower)) + 1
    short_positions = np.flatnonzero((osc[:-1] > settings.upper) & (osc[1:] < settings.upper)) + 1
    return {"Long": long_positions.astype(int), "Short": short_positions.astype(int)}


def calculate_forward_returns(close: np.ndarray, positions: np.ndarray, holding_period: int, side: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    positions = positions[positions + int(holding_period) < len(close)]
    if positions.size == 0:
        return np.array([], dtype=float)

    current = close[positions]
    future = close[positions + int(holding_period)]
    if side == "Long":
        returns = future / current - 1
    else:
        returns = current / future - 1
    return returns[np.isfinite(returns)]


def bootstrap_mean_ci(returns: np.ndarray, runs: int = EDGE_BOOTSTRAP_RUNS, seed: int = 42) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return {"ci90_low": np.nan, "ci90_high": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, returns.size, size=(int(runs), returns.size))
    sample_means = returns[sample_idx].mean(axis=1)
    return {
        "ci90_low": float(np.quantile(sample_means, 0.05)),
        "ci90_high": float(np.quantile(sample_means, 0.95)),
        "ci95_low": float(np.quantile(sample_means, 0.025)),
        "ci95_high": float(np.quantile(sample_means, 0.975)),
    }


def random_benchmark(
    close: np.ndarray,
    event_count: int,
    holding_period: int,
    side: str,
    runs: int = EDGE_RANDOM_RUNS,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    if event_count <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    eligible_positions = np.arange(0, len(close) - int(holding_period), dtype=int)
    if eligible_positions.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    random_means = []
    random_returns_all = []
    replace = event_count > eligible_positions.size
    for _ in range(int(runs)):
        positions = rng.choice(eligible_positions, size=int(event_count), replace=replace)
        returns = calculate_forward_returns(close, positions, int(holding_period), side)
        if returns.size:
            random_means.append(float(returns.mean()))
            random_returns_all.append(returns)

    if not random_means:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.asarray(random_means, dtype=float), np.concatenate(random_returns_all)


def edge_rating(ci95_low: float, ci95_high: float, event_mean: float, random_mean: float, random_q95: float) -> tuple[str, str]:
    if pd.isna(ci95_low) or pd.isna(ci95_high) or pd.isna(event_mean) or pd.isna(random_mean):
        return "ROT", "Nicht genug Daten"
    if ci95_low <= 0 <= ci95_high or ci95_low <= 0:
        return "ROT", "95% CI enthaelt 0"
    if event_mean <= random_mean:
        return "ROT", "Event Mean nicht besser als Random Mean"
    if pd.notna(random_q95) and event_mean > random_q95:
        return "GRUEN", "CI > 0 und klar besser als Random"
    return "GELB", "CI > 0, aber nur leicht besser als Random"


def build_taco_edge_results(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[tuple[str, int], dict], dict[str, np.ndarray], pd.DataFrame]:
    data = df[(df.index.year >= settings.start_year) & (df.index.year <= settings.end_year)].copy()
    close = data["close"].to_numpy(dtype=float)
    events = find_taco_reversal_events(df, settings)
    rows = []
    details = {}

    for side, positions in events.items():
        for holding_period in EDGE_HOLDING_PERIODS:
            valid_positions = positions[positions + holding_period < len(data)]
            event_returns = calculate_forward_returns(close, valid_positions, holding_period, side)
            ci = bootstrap_mean_ci(
                event_returns,
                seed=10_000 + holding_period + (0 if side == "Long" else 1_000),
            )
            random_means, random_returns = random_benchmark(
                close,
                len(event_returns),
                holding_period,
                side,
                seed=20_000 + holding_period + (0 if side == "Long" else 1_000),
            )
            event_mean = float(np.mean(event_returns)) if event_returns.size else np.nan
            random_mean = float(np.mean(random_means)) if random_means.size else np.nan
            random_q05 = float(np.quantile(random_means, 0.05)) if random_means.size else np.nan
            random_q95 = float(np.quantile(random_means, 0.95)) if random_means.size else np.nan
            rating, rating_reason = edge_rating(ci["ci95_low"], ci["ci95_high"], event_mean, random_mean, random_q95)

            rows.append({
                "Side": side,
                "Holding Period": holding_period,
                "Anzahl Events": int(event_returns.size),
                "Mean Return": event_mean,
                "Median Return": float(np.median(event_returns)) if event_returns.size else np.nan,
                "Winrate": float(np.mean(event_returns > 0)) if event_returns.size else np.nan,
                "Standard Deviation": float(np.std(event_returns, ddof=1)) if event_returns.size > 1 else np.nan,
                "Best Trade": float(np.max(event_returns)) if event_returns.size else np.nan,
                "Worst Trade": float(np.min(event_returns)) if event_returns.size else np.nan,
                "90% CI Low": ci["ci90_low"],
                "90% CI High": ci["ci90_high"],
                "95% CI Low": ci["ci95_low"],
                "95% CI High": ci["ci95_high"],
                "Mean Event Return": event_mean,
                "Mean Random Return": random_mean,
                "5%-Quantil Random": random_q05,
                "95%-Quantil Random": random_q95,
                "Edge Bewertung": rating,
                "Begruendung": rating_reason,
            })
            details[(side, holding_period)] = {
                "event_returns": event_returns,
                "random_means": random_means,
                "random_returns": random_returns,
                "positions": valid_positions,
            }

    event_rows = []
    for side, positions in events.items():
        for pos in positions:
            if 0 <= pos < len(data):
                event_rows.append({
                    "Date": data.index[pos],
                    "Side": side,
                    "Close": float(data["close"].iloc[pos]),
                    "TACO": float(data["osc"].iloc[pos]),
                })
    events_df = pd.DataFrame(event_rows).sort_values("Date") if event_rows else pd.DataFrame(columns=["Date", "Side", "Close", "TACO"])
    return pd.DataFrame(rows), details, events, events_df


def format_edge_percent_table(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    pct_cols = [
        "Mean Return",
        "Median Return",
        "Winrate",
        "Standard Deviation",
        "Best Trade",
        "Worst Trade",
        "90% CI Low",
        "90% CI High",
        "95% CI Low",
        "95% CI High",
        "Mean Event Return",
        "Mean Random Return",
        "5%-Quantil Random",
        "95%-Quantil Random",
    ]
    for col in pct_cols:
        out[col] = out[col] * 100
    return out


def render_taco_edge_discovery(df: pd.DataFrame, settings: Settings) -> None:
    st.header("TACO Edge Discovery")
    st.caption(
        "Reine Event-Studie auf Daily-Daten: TACO-Reversal-Events, fixe Forward-Returns, Bootstrap-CIs "
        "und Random-Benchmark. Keine Strategie-Exits, keine Filter, keine Optimierung."
    )

    if df.empty:
        st.warning("Keine TACO-Daten fuer die Edge Discovery.")
        return

    data = df[(df.index.year >= settings.start_year) & (df.index.year <= settings.end_year)].copy()
    if len(data) < max(EDGE_HOLDING_PERIODS) + 2:
        st.warning("Zu wenige Daily-Bars fuer 60 Handelstage Forward Return.")
        return

    with st.spinner("Berechne Events, 10.000 Bootstrap-Resamples und 1.000 Random-Benchmarks je Setup."):
        results, details, events, events_df = build_taco_edge_results(df, settings)

    long_count = int(len(events["Long"]))
    short_count = int(len(events["Short"]))
    cols = st.columns(4)
    cols[0].metric("Long Events", f"{long_count:,}")
    cols[1].metric("Short Events", f"{short_count:,}")
    cols[2].metric("Start", data.index.min().date().isoformat())
    cols[3].metric("Ende", data.index.max().date().isoformat())

    if results.empty:
        st.warning("Keine auswertbaren Events gefunden.")
        return

    formatted = format_edge_percent_table(results)
    st.subheader("Edge Summary")
    st.dataframe(
        formatted.style.format(
            {
                "Mean Return": "{:.2f}%",
                "Median Return": "{:.2f}%",
                "Winrate": "{:.1f}%",
                "Standard Deviation": "{:.2f}%",
                "Best Trade": "{:.2f}%",
                "Worst Trade": "{:.2f}%",
                "90% CI Low": "{:.2f}%",
                "90% CI High": "{:.2f}%",
                "95% CI Low": "{:.2f}%",
                "95% CI High": "{:.2f}%",
                "Mean Event Return": "{:.2f}%",
                "Mean Random Return": "{:.2f}%",
                "5%-Quantil Random": "{:.2f}%",
                "95%-Quantil Random": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

    green_count = int((results["Edge Bewertung"] == "GRUEN").sum())
    yellow_count = int((results["Edge Bewertung"] == "GELB").sum())
    red_count = int((results["Edge Bewertung"] == "ROT").sum())
    if green_count == 0:
        verdict = "ROT: In dieser Stichprobe liefert TACO alleine keine robuste statistisch nachweisbare Predictive Power."
    elif red_count == 0 and yellow_count == 0:
        verdict = "GRUEN: In dieser Stichprobe zeigt TACO alleine ueber alle getesteten Event-Gruppen robuste Predictive Power."
    else:
        verdict = "GEMISCHT: TACO zeigt nur in Teilen der Stichprobe statistische Edge; das ist kein pauschaler Beweis fuer Predictive Power."
    st.info(verdict)

    st.subheader("TACO Chart mit Events")
    taco_fig = go.Figure()
    taco_fig.add_trace(go.Scatter(x=data.index, y=data["osc"], mode="lines", name="TACO", line=dict(color="#bd37dc", width=2)))
    taco_fig.add_hline(y=settings.upper, line_dash="dash", line_color="rgba(255,0,0,.5)")
    taco_fig.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,.35)")
    taco_fig.add_hline(y=settings.lower, line_dash="dash", line_color="rgba(0,200,90,.5)")
    if not events_df.empty:
        longs = events_df[events_df["Side"] == "Long"]
        shorts = events_df[events_df["Side"] == "Short"]
        if not longs.empty:
            taco_fig.add_trace(go.Scatter(x=longs["Date"], y=longs["TACO"], mode="markers", name="Long Event", marker=dict(color="#22c55e", symbol="triangle-up", size=9)))
        if not shorts.empty:
            taco_fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["TACO"], mode="markers", name="Short Event", marker=dict(color="#ef4444", symbol="triangle-down", size=9)))
    taco_fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[-110, 110]))
    st.plotly_chart(taco_fig, use_container_width=True)

    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        selected_side = st.selectbox("Event-Seite", ["Long", "Short"], key="edge_side")
    with chart_cols[1]:
        selected_holding_period = st.selectbox("Holding Period", EDGE_HOLDING_PERIODS, key="edge_holding_period")

    selected = details[(selected_side, int(selected_holding_period))]
    event_returns = selected["event_returns"]
    random_means = selected["random_means"]

    visual_cols = st.columns(2)
    with visual_cols[0]:
        event_hist = go.Figure()
        event_hist.add_trace(go.Histogram(x=event_returns * 100, nbinsx=30, name="Event Returns", marker_color="#38bdf8"))
        event_hist.update_layout(height=320, title=f"{selected_side} Event Returns {selected_holding_period}D", xaxis_title="Return %", yaxis_title="Anzahl", margin=dict(l=20, r=20, t=50, b=35))
        st.plotly_chart(event_hist, use_container_width=True)
    with visual_cols[1]:
        random_hist = go.Figure()
        random_hist.add_trace(go.Histogram(x=random_means * 100, nbinsx=30, name="Random Mean Returns", marker_color="#94a3b8"))
        random_hist.update_layout(height=320, title=f"Random Mean Returns {selected_holding_period}D", xaxis_title="Mean Return %", yaxis_title="Anzahl", margin=dict(l=20, r=20, t=50, b=35))
        st.plotly_chart(random_hist, use_container_width=True)

    box_fig = go.Figure()
    box_fig.add_trace(go.Box(y=event_returns * 100, name=f"{selected_side} Events", marker_color="#22c55e" if selected_side == "Long" else "#ef4444", boxmean=True))
    box_fig.update_layout(height=300, title="Boxplot Event Returns", yaxis_title="Return %", margin=dict(l=20, r=20, t=50, b=35))
    st.plotly_chart(box_fig, use_container_width=True)

    st.subheader("Event-Liste")
    st.dataframe(events_df, use_container_width=True)


st.markdown(
    f"""
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px;">
        <span style="font-size:1.45rem;font-weight:800;color:#e2e8f0;letter-spacing:-.02em;">{APP_NAME}</span>
        <span style="font-size:.72rem;font-weight:600;color:#334155;text-transform:uppercase;letter-spacing:.08em;">Beta</span>
    </div>
    <div style="font-size:.78rem;color:#475569;margin-bottom:6px;">
        Backtest · Edge Discovery · Seasonality · Walk Forward · COT · Fear &amp; Greed
    </div>
    <hr style="border:none;border-top:1px solid rgba(148,163,184,.08);margin:0 0 12px 0;">
    """,
    unsafe_allow_html=True,
)

CORE_METRICS = ["Trades", "Winrate", "Profit Factor", "Net Profit", "Max DD", "Expectancy R", "Max Loss Streak"]
PRACTICE_METRICS = ["Avg Realized Win", "Avg Realized Loss", "Avg R", "Intratrade MAE", "Avg MFE", "Stop Breach Count", "Stop Breach Avg"]

st.sidebar.markdown(
    """
    <div style="padding:0 4px 16px 4px;">
        <div style="font-size:1.0rem;font-weight:800;color:#e2e8f0;letter-spacing:-.01em;">TACO Lab</div>
        <div style="font-size:.68rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;margin-top:1px;">Quant Swing Strategy</div>
    </div>
    <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.09em;color:#334155;padding:0 4px 6px 4px;border-bottom:1px solid rgba(148,163,184,.07);margin-bottom:6px;">Navigation</div>
    """,
    unsafe_allow_html=True,
)
def _momi_backtest_engine(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Kern-Backtest für die Mo-Mi Strategie. Wiederverwendbar für WFA IS+OOS Folds.
    params-Keys: entry_dow, exit_dow, entry_hour, entry_min, exit_hour, exit_min,
                 ma_type, ma_period, filter_mode, use_adx, adx_thresh,
                 sl_pct, use_trail, trail_trig, trail_off
    Optionale Kosten-Keys (Default 0 → kein Effekt, rückwärtskompatibel):
                 spread_pts (Round-Turn-Spread in Punkten), commission_pct (Round-Turn-Kommission
                 in % der Notional)
    Optionaler fill_mode-Key (Default "close" → bisheriges Verhalten, rückwärtskompatibel):
                 "close" = Fill im Signal-Bar (z.B. Montag-Close), "next_open" = Fill zum Open
                 des nächsten Bars (z.B. Dienstag-Open, entspricht Pine-Default ohne
                 process_orders_on_close)
    Gibt (df_trades, equity_series) zurück.
    """
    ma_period   = params["ma_period"]
    ma_type     = params["ma_type"]
    filter_mode = params["filter_mode"]

    ma_vals = df["Close"].ewm(span=ma_period, adjust=False).mean() if ma_type == "EMA" else df["Close"].rolling(ma_period).mean()
    df = df.copy()
    df["MA"] = ma_vals

    # ADX
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr    = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up    = hi - hi.shift(); dn = lo.shift() - lo
    pdm   = pd.Series(np.where((up>dn)&(up>0), up, 0), index=df.index)
    ndm   = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=df.index)
    pdi   = 100 * pdm.rolling(14).mean() / atr14
    ndi   = 100 * ndm.rolling(14).mean() / atr14
    dx    = (100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)).fillna(0)
    df["ADX"] = dx.rolling(14).mean()

    # Auf Daily-Daten sind alle Stunden = 0 → Stunden-Check weglassen
    is_daily = df.index.hour.nunique() == 1 and df.index.hour[0] == 0
    if is_daily:
        df["entry_time"] = df.index.dayofweek == params["entry_dow"]
        df["exit_time"]  = df.index.dayofweek == params["exit_dow"]
    else:
        df["entry_time"] = (df.index.dayofweek == params["entry_dow"]) & \
                           (df.index.hour == params["entry_hour"]) & \
                           (df.index.minute == params["entry_min"])
        df["exit_time"]  = (df.index.dayofweek == params["exit_dow"]) & \
                           (df.index.hour == params["exit_hour"]) & \
                           (df.index.minute == params["exit_min"])

    if filter_mode == "Close > MA":
        df["ma_ok"] = df["Close"] > df["MA"]
    elif filter_mode == "MA steigt":
        df["ma_ok"] = df["MA"] > df["MA"].shift(1)
    else:
        df["ma_ok"] = True

    df["adx_ok"] = (df["ADX"] > params["adx_thresh"]) if params["use_adx"] else True
    df["entry_signal"] = df["entry_time"] & df["ma_ok"] & df["adx_ok"] & df["MA"].notna()

    # Volatility Targeting (Moreira/Muir 2017, Harvey et al. 2018): Positionsgröße zusätzlich zur
    # SL-basierten Risikogröße mit target_vol / realisierte_vol skalieren — kleinere Position bei
    # zuletzt hoher Vola, größere bei niedriger. EWMA-Vola nutzt nur Renditen VOR dem aktuellen Bar
    # (shift(1)), damit kein Lookahead entsteht — am Entry-Tag ist die Vola von heute ja noch unbekannt.
    if params.get("use_vol_target", False):
        _halflife = max(1, params.get("vol_halflife", 20))
        _target_vol_ann = params.get("vol_target_pct", 15.0) / 100.0
        _rets = df["Close"].pct_change()
        _ewma_var = (_rets ** 2).ewm(halflife=_halflife, min_periods=int(_halflife * 3)).mean()
        _realized_vol_ann = np.sqrt(_ewma_var * 252)
        df["vol_scalar"] = (_target_vol_ann / _realized_vol_ann).shift(1).clip(0.2, 3.0).fillna(1.0)
    else:
        df["vol_scalar"] = 1.0

    capital = 10_000.0
    position, entry_price, sl_price = 0, 0.0, 0.0
    trail_high, qty, trail_active = 0.0, 0.0, False
    pending_entry, entry_ts = False, None
    trades, equity_curve = [], []

    sl_pct    = params["sl_pct"]
    use_trail = params["use_trail"]
    trail_trig = params["trail_trig"]
    trail_off  = params["trail_off"]
    half_spread    = params.get("spread_pts", 0.0) / 2.0
    commission_pct = params.get("commission_pct", 0.0)
    # Swap/Übernacht-Finanzierung: einmal pro KALENDER-Nacht zwischen Entry- und Exit-Datum,
    # nicht pro Handelstag — eine übers Wochenende gehaltene Position zahlt so automatisch für
    # alle Wochenend-Nächte, unabhängig davon, an welchem Tag der Broker das intern verbucht.
    swap_pts_per_night = params.get("swap_pts_per_night", 0.0)
    # "close"     = Fill zum Signal-Bar-Close (z.B. Montag-Close) — Standardverhalten
    # "next_open" = Fill zum Open des NÄCHSTEN Bars (z.B. Dienstag-Open) — Pine-Default ohne process_orders_on_close
    fill_mode = params.get("fill_mode", "close")

    # Rohe numpy-Arrays statt df.iterrows() — iterrows() baut pro Zeile ein komplettes
    # Series-Objekt (teuer) und ist bei Grid-Searches mit zehntausenden Aufrufen der
    # dominante Flaschenhals (Ensemble: Modi × Läufe × Folds × Grid-Kombinationen).
    close_arr        = df["Close"].to_numpy(dtype=float)
    high_arr         = df["High"].to_numpy(dtype=float)
    low_arr          = df["Low"].to_numpy(dtype=float)
    open_arr         = df["Open"].to_numpy(dtype=float)
    exit_time_arr    = df["exit_time"].to_numpy()
    entry_signal_arr = df["entry_signal"].to_numpy()
    vol_scalar_arr   = df["vol_scalar"].to_numpy(dtype=float)
    idx              = df.index

    for i in range(len(df)):
        c, h, l, o = close_arr[i], high_arr[i], low_arr[i], open_arr[i]

        # Entry vom Vortag fällig (next_open-Modus): jetzt zum Open dieses Bars füllen
        if pending_entry and position == 0:
            entry_price    = o + half_spread
            sl_price       = entry_price * (1 - sl_pct / 100)
            risk_per_trade = capital * (params.get("risk_pct", 1.0) / 100)
            risk_per_unit  = entry_price - sl_price
            qty            = (risk_per_trade / risk_per_unit) * vol_scalar_arr[i] if risk_per_unit > 0 else 0
            trail_high     = o
            trail_active   = False
            position       = 1
            pending_entry  = False
            entry_ts       = idx[i]

        if position == 1:
            if use_trail and trail_active:
                trail_high = max(trail_high, h)
                sl_price   = max(sl_price, trail_high * (1 - trail_off / 100))
            elif use_trail and h >= entry_price * (1 + trail_trig / 100):
                trail_active = True
                trail_high   = h

            exit_price, reason = None, None
            if l <= sl_price:
                exit_price, reason = sl_price, "SL"
            elif exit_time_arr[i]:
                exit_price, reason = c, "Time Exit"

            if exit_price is not None:
                # Verkauf zum Bid (Close/Stop minus halber Spread) + optionale Kommission
                exit_price_net = exit_price - half_spread
                pnl = (exit_price_net - entry_price) * qty
                if commission_pct > 0:
                    pnl -= qty * entry_price * (commission_pct / 100)
                nights_held = max((idx[i] - entry_ts).days, 0) if entry_ts is not None else 0
                swap_cost = nights_held * swap_pts_per_night * qty
                if swap_cost:
                    pnl -= swap_cost
                capital += pnl
                trades.append({"Zeit": idx[i], "Entry": entry_price, "Exit": exit_price_net,
                                "PnL $": round(pnl, 2), "Grund": reason,
                                "Naechte": nights_held, "Swap $": round(swap_cost, 2)})
                position, trail_active = 0, False

        if position == 0 and not pending_entry and entry_signal_arr[i]:
            if fill_mode == "next_open":
                # Erst am nächsten Bar-Open füllen (siehe oben) — hier nur vormerken
                pending_entry = True
            else:
                # Kauf zum Ask (Close plus halber Spread) — Fill im selben Bar
                entry_price    = c + half_spread
                sl_price       = entry_price * (1 - sl_pct / 100)
                risk_per_trade = capital * (params.get("risk_pct", 1.0) / 100)
                risk_per_unit  = entry_price - sl_price
                qty            = (risk_per_trade / risk_per_unit) * vol_scalar_arr[i] if risk_per_unit > 0 else 0
                trail_high     = c
                trail_active   = False
                position       = 1
                entry_ts       = idx[i]

        equity_curve.append(capital)

    eq = pd.Series(equity_curve, index=df.index)
    return pd.DataFrame(trades), eq


def _momi_metrics(df_trades: pd.DataFrame, equity: pd.Series) -> dict:
    """Berechnet alle Kennzahlen aus einem Trades-DataFrame + Equity-Series."""
    if df_trades.empty:
        return {"n": 0, "wr": 0, "pf": 0, "sharpe": 0, "max_dd": 0, "total_ret": 0}
    n   = len(df_trades)
    wr  = (df_trades["PnL $"] > 0).sum() / n * 100
    gp  = df_trades.loc[df_trades["PnL $"] > 0, "PnL $"].sum()
    gl  = df_trades.loc[df_trades["PnL $"] <= 0, "PnL $"].abs().sum()
    pf  = gp / gl if gl > 0 else 9.99
    dd  = (equity - equity.cummax()) / equity.cummax() * 100
    ps  = df_trades["PnL $"]
    sharpe = (ps.mean() / ps.std() * np.sqrt(252)) if ps.std() > 0 else 0
    total_ret = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100
    avg_ret   = df_trades["PnL $"].mean() / equity.iloc[0] * 100 if n > 0 else 0
    return {"n": n, "wr": round(wr,1), "pf": round(pf,3),
            "sharpe": round(sharpe,3), "max_dd": round(dd.min(),2),
            "total_ret": round(total_ret,2), "avg_ret": round(avg_ret,3)}


def render_yen_momi_strategie() -> None:
    """Weekday MA Long Strategy — Mo-Einstieg, Mi-Ausstieg, EMA/SMA Filter + ADX + WFA."""
    import datetime as _dt

    st.header("Yen Mo-Mi Strategie")
    st.caption("Long-only · Zeitbasierter Entry/Exit · EMA/SMA + ADX Filter · Trailing Stop · Walk-Forward Analyse")

    # ── Daten laden ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("Mo-Mi: Einstellungen")
        momi_symbol = st.text_input("Symbol (Yahoo)", "GBPJPY=X", key="momi_sym")
        momi_start  = st.date_input("Von", _dt.date(2024, 1, 1), key="momi_start")
        momi_end    = st.date_input("Bis", _dt.date.today(),     key="momi_end")
        momi_tf     = st.selectbox("Timeframe", ["1h", "30m", "15m"], key="momi_tf")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Entry**")
        entry_day  = st.selectbox("Entry-Tag",    ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag"], index=0, key="momi_ed")
        entry_hour = st.number_input("Entry-Stunde", 0, 23, 15, key="momi_eh")
        entry_min  = st.number_input("Entry-Minute", 0, 59, 0,  key="momi_em")
    with col2:
        st.markdown("**Exit**")
        exit_day   = st.selectbox("Exit-Tag",     ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag"], index=2, key="momi_xd")
        exit_hour  = st.number_input("Exit-Stunde",  0, 23, 17, key="momi_xh")
        exit_min   = st.number_input("Exit-Minute",  0, 59, 0,  key="momi_xm")
    with col3:
        st.markdown("**Indikator & Risk**")
        ma_type    = st.selectbox("MA-Typ",    ["EMA", "SMA"], key="momi_mat")
        ma_period  = st.selectbox("MA-Periode", [20, 50, 100, 200], key="momi_map")
        filter_mode = st.selectbox("Filter-Modus", ["Close > MA", "MA steigt", "Kein Filter"], key="momi_fm")
        use_adx    = st.checkbox("ADX-Filter", True, key="momi_adx")
        adx_thresh = st.slider("ADX-Schwelle", 10, 40, 20, key="momi_adxt")
        sl_pct     = st.number_input("Stop-Loss %", 0.1, 10.0, 1.5, step=0.1, key="momi_sl")
        use_trail  = st.checkbox("Trailing Stop", True, key="momi_trail")
        trail_trig = st.number_input("Trail-Trigger %", 0.1, 5.0, 0.5, step=0.1, key="momi_tt")
        trail_off  = st.number_input("Trail-Abstand %", 0.1, 5.0, 0.4, step=0.1, key="momi_to")

    if not st.button("▶ Backtest starten", key="momi_run"):
        st.info("Parameter einstellen und Backtest starten.")
        return

    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance fehlt: `pip install yfinance`")
        return

    with st.spinner("Daten laden …"):
        df_raw = yf.download(momi_symbol, start=str(momi_start), end=str(momi_end),
                             interval=momi_tf, auto_adjust=True, progress=False)
    if df_raw.empty:
        st.error("Keine Daten geladen — Symbol oder Zeitraum prüfen.")
        return
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)

    # ── Indikatoren ──────────────────────────────────────────────────────────
    df = df_raw.copy()
    ma_vals = df["Close"].ewm(span=ma_period, adjust=False).mean() if ma_type == "EMA" else df["Close"].rolling(ma_period).mean()
    df["MA"] = ma_vals

    # ADX
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr   = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    up_move  = hi - hi.shift()
    dn_move  = lo.shift() - lo
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0), index=df.index)
    ndm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0), index=df.index)
    pdi = 100 * pdm.rolling(14).mean() / atr14
    ndi = 100 * ndm.rolling(14).mean() / atr14
    dx  = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    df["ADX"] = dx.rolling(14).mean()

    # ── Wochentag-Mapping ────────────────────────────────────────────────────
    day_map = {"Montag": 0, "Dienstag": 1, "Mittwoch": 2, "Donnerstag": 3, "Freitag": 4}
    entry_dow = day_map[entry_day]
    exit_dow  = day_map[exit_day]

    df["entry_time"] = (df.index.dayofweek == entry_dow) & (df.index.hour == entry_hour) & (df.index.minute == entry_min)
    df["exit_time"]  = (df.index.dayofweek == exit_dow)  & (df.index.hour == exit_hour)  & (df.index.minute == exit_min)

    if filter_mode == "Close > MA":
        df["ma_ok"] = df["Close"] > df["MA"]
    elif filter_mode == "MA steigt":
        df["ma_ok"] = df["MA"] > df["MA"].shift(1)
    else:
        df["ma_ok"] = True

    df["adx_ok"] = (df["ADX"] > adx_thresh) if use_adx else True
    df["entry_signal"] = df["entry_time"] & df["ma_ok"] & df["adx_ok"] & df["MA"].notna()

    # ── Backtest ─────────────────────────────────────────────────────────────
    capital, position, entry_price, sl_price = 10_000.0, 0, 0.0, 0.0
    trail_high, qty = 0.0, 0.0
    trail_active = False
    trades, equity_curve = [], []

    for ts, row in df.iterrows():
        c = float(row["Close"])
        h = float(row["High"])
        l = float(row["Low"])

        if position == 1:
            # Trailing Stop aktualisieren
            if use_trail and trail_active:
                trail_high = max(trail_high, h)
                trail_sl   = trail_high * (1 - trail_off / 100)
                sl_price   = max(sl_price, trail_sl)
            elif use_trail and h >= entry_price * (1 + trail_trig / 100):
                trail_active = True
                trail_high   = h

            hit_sl = l <= sl_price
            hit_te = bool(row["exit_time"])

            exit_price, reason = None, None
            if hit_sl:
                exit_price, reason = sl_price, "SL"
            elif hit_te:
                exit_price, reason = c, "Time Exit"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * qty
                capital += pnl
                trades.append({"Zeit": ts, "Richtung": "Long", "Entry": entry_price,
                                "Exit": exit_price, "PnL $": round(pnl, 2), "Grund": reason})
                position, trail_active = 0, False

        if position == 0 and row["entry_signal"]:
            entry_price    = c
            sl_price       = c * (1 - sl_pct / 100)
            risk_per_trade = capital * (params.get("risk_pct", 1.0) / 100)
            risk_per_unit  = entry_price - sl_price
            qty            = (risk_per_trade / risk_per_unit) if risk_per_unit > 0 else 0
            trail_high     = c
            trail_active   = False
            position       = 1

        equity_curve.append(capital)

    df["Equity"] = equity_curve

    # ── Kennzahlen ───────────────────────────────────────────────────────────
    df_trades = pd.DataFrame(trades)
    if df_trades.empty:
        st.warning("Keine Trades — Filter oder Zeitfenster anpassen.")
        return

    total_ret  = (df["Equity"].iloc[-1] - 10_000) / 10_000 * 100
    n          = len(df_trades)
    wins       = (df_trades["PnL $"] > 0).sum()
    wr         = wins / n * 100
    gp         = df_trades.loc[df_trades["PnL $"] > 0, "PnL $"].sum()
    gl         = df_trades.loc[df_trades["PnL $"] <= 0, "PnL $"].abs().sum()
    pf         = gp / gl if gl > 0 else float("inf")
    eq         = df["Equity"]
    dd         = (eq - eq.cummax()) / eq.cummax() * 100
    max_dd     = dd.min()
    pnl_s      = df_trades["PnL $"]
    sharpe     = (pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0

    # ── KPI-Kacheln ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Gesamtrendite",  f"{total_ret:.2f}%")
    k2.metric("Trades",         n)
    k3.metric("Win-Rate",       f"{wr:.1f}%")
    k4.metric("Profit Factor",  f"{pf:.2f}")
    k5.metric("Sharpe Ratio",   f"{sharpe:.2f}")
    st.metric("Max. Drawdown",  f"{max_dd:.2f}%")

    # ── Equity Curve ─────────────────────────────────────────────────────────
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="Equity", line=dict(color="#00d4aa", width=2)))
    fig_eq.update_layout(title="Equity Curve", height=300, template="plotly_dark", margin=dict(t=40, b=20))
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Preis + MA + Entry/Exit ───────────────────────────────────────────────
    fig_c = go.Figure()
    fig_c.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                   low=df["Low"], close=df["Close"], name="Preis",
                                   increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    fig_c.add_trace(go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type} {ma_period}",
                               line=dict(color="orange", width=1.5)))
    for _, tr_ in df_trades.iterrows():
        fig_c.add_vline(x=tr_["Zeit"], line_color="lime" if tr_["Grund"] != "SL" else "red",
                        line_width=1, line_dash="dot")
    fig_c.update_layout(title="Preis + MA", height=400, template="plotly_dark",
                        xaxis_rangeslider_visible=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Trade-Tabelle ────────────────────────────────────────────────────────
    st.subheader("Trade-Liste")
    st.dataframe(df_trades.style.map(lambda v: "color: #26a69a" if isinstance(v, (int, float)) and v > 0
                                     else ("color: #ef5350" if isinstance(v, (int, float)) and v < 0 else ""),
                                     subset=["PnL $"]), use_container_width=True)

    # ── Exit-Grund Verteilung ────────────────────────────────────────────────
    exit_counts = df_trades["Grund"].value_counts()
    fig_pie = go.Figure(go.Pie(labels=exit_counts.index, values=exit_counts.values,
                               hole=0.4, marker_colors=["#ef5350", "#26a69a", "#ffa726"]))
    fig_pie.update_layout(title="Exit-Gründe", height=280, template="plotly_dark", margin=dict(t=40, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # WALK-FORWARD ANALYSE
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Walk-Forward Analyse")
    st.caption("IS-Periode optimiert beste Parameter → OOS-Periode testet diese ungesehen. Mehrere Folds = Robustheitsnachweis.")

    wfa_col1, wfa_col2, wfa_col3 = st.columns(3)
    with wfa_col1:
        wfa_is_months  = st.number_input("IS-Fenster (Monate)", 3, 24, 6, key="wfa_is")
        wfa_oos_months = st.number_input("OOS-Fenster (Monate)", 1, 12, 2, key="wfa_oos")
    with wfa_col2:
        wfa_min_trades = st.number_input("Min. IS-Trades für gültigen Fold", 3, 30, 5, key="wfa_mint")
        wfa_opt_metric = st.selectbox("Optimierungsziel", ["profit_factor", "sharpe", "win_rate"], key="wfa_met")
    with wfa_col3:
        wfa_min_folds  = st.number_input("Min. OOS-Folds für ✅ Badge", 2, 10, 3, key="wfa_mf")
        st.markdown(" ")
        run_wfa = st.button("🔄 Walk-Forward starten", use_container_width=True, key="wfa_btn")

    # Grid-Suchraum für IS-Optimierung
    with st.expander("Grid-Suchraum anpassen"):
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            grid_sl   = st.multiselect("Stop-Loss %",    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], default=[1.0, 1.5, 2.0], key="wfa_gsl")
            grid_ma   = st.multiselect("MA-Periode",     [20, 50, 100, 200],               default=[20, 50],        key="wfa_gma")
        with gc2:
            grid_adx  = st.multiselect("ADX-Schwelle",   [15, 20, 25, 30],                 default=[20, 25],        key="wfa_gadx")
            grid_tt   = st.multiselect("Trail-Trigger %",[0.3, 0.5, 0.8, 1.0],             default=[0.5, 0.8],      key="wfa_gtt")
        with gc3:
            grid_to   = st.multiselect("Trail-Abstand %",[0.2, 0.3, 0.4, 0.5],             default=[0.3, 0.4],      key="wfa_gto")

    if not run_wfa:
        st.info("Parameter festlegen und Walk-Forward starten.")
    else:
        if df_raw.empty:
            st.error("Keine Rohdaten — zuerst oben einen Backtest starten.")
        elif not all([grid_sl, grid_ma, grid_adx, grid_tt, grid_to]):
            st.warning("Bitte mindestens einen Wert pro Grid-Parameter auswählen.")
        else:
            # Basis-Params aus den Eingaben oben (Zeit/Richtung bleiben fix)
            base_params = {
                "entry_dow":  day_map[entry_day],
                "exit_dow":   day_map[exit_day],
                "entry_hour": int(entry_hour),
                "entry_min":  int(entry_min),
                "exit_hour":  int(exit_hour),
                "exit_min":   int(exit_min),
                "ma_type":    ma_type,
                "filter_mode": filter_mode,
                "use_adx":    use_adx,
                "use_trail":  use_trail,
            }

            # Folds generieren: rollierendes IS+OOS Fenster
            all_dates = df_raw.index
            total_start = all_dates[0]
            total_end   = all_dates[-1]
            is_delta    = pd.DateOffset(months=int(wfa_is_months))
            oos_delta   = pd.DateOffset(months=int(wfa_oos_months))

            folds = []
            fold_start = total_start
            while True:
                is_end  = fold_start + is_delta
                oos_end = is_end    + oos_delta
                if oos_end > total_end:
                    break
                folds.append({"is_start": fold_start, "is_end": is_end,
                               "oos_start": is_end,    "oos_end": oos_end})
                fold_start = fold_start + oos_delta

            if len(folds) < 2:
                st.warning(f"Zu wenig Daten für WFA. Bitte längeren Zeitraum laden oder IS/OOS-Fenster verkleinern.")
            else:
                st.info(f"**{len(folds)} Folds** generiert — IS: {wfa_is_months} Monate / OOS: {wfa_oos_months} Monate")

                progress_bar = st.progress(0, text="Walk-Forward läuft …")
                wfa_rows = []

                for fi, fold in enumerate(folds):
                    progress_bar.progress((fi) / len(folds),
                                          text=f"Fold {fi+1}/{len(folds)} — IS optimieren …")

                    df_is  = df_raw[(df_raw.index >= fold["is_start"])  & (df_raw.index < fold["is_end"])]
                    df_oos = df_raw[(df_raw.index >= fold["oos_start"]) & (df_raw.index < fold["oos_end"])]

                    if len(df_is) < 50 or len(df_oos) < 10:
                        continue

                    # IS-Optimierung: Grid Search
                    best_score, best_params = -np.inf, None
                    from itertools import product as _prod
                    for sl_, ma_, adx_, tt_, to_ in _prod(grid_sl, grid_ma, grid_adx, grid_tt, grid_to):
                        p = {**base_params, "sl_pct": sl_, "ma_period": ma_,
                             "adx_thresh": adx_, "trail_trig": tt_, "trail_off": to_}
                        try:
                            tr_, eq_ = _momi_backtest_engine(df_is.copy(), p)
                        except Exception:
                            continue
                        if len(tr_) < int(wfa_min_trades):
                            continue
                        m_ = _momi_metrics(tr_, eq_)
                        score = m_.get(wfa_opt_metric.replace("win_rate","wr"), m_["pf"])
                        if score > best_score:
                            best_score, best_params = score, p

                    if best_params is None:
                        wfa_rows.append({"Fold": fi+1,
                                         "IS": f"{fold['is_start'].date()} – {fold['is_end'].date()}",
                                         "OOS": f"{fold['oos_start'].date()} – {fold['oos_end'].date()}",
                                         "IS Trades": 0, "IS PF": "–", "IS Sharpe": "–",
                                         "OOS Trades": 0, "OOS PF": "–", "OOS Sharpe": "–",
                                         "OOS Ret %": "–", "Status": "⚠️ Keine IS-Params"})
                        continue

                    # OOS mit besten IS-Params
                    tr_is,  eq_is  = _momi_backtest_engine(df_is.copy(),  best_params)
                    tr_oos, eq_oos = _momi_backtest_engine(df_oos.copy(), best_params)
                    m_is  = _momi_metrics(tr_is,  eq_is)
                    m_oos = _momi_metrics(tr_oos, eq_oos)

                    status = "✅ OOS-validiert" if (m_oos["n"] >= 1 and m_oos["pf"] > 1.0 and m_oos["total_ret"] > 0) else "❌ OOS-Fail"

                    wfa_rows.append({
                        "Fold":        fi + 1,
                        "IS":          f"{fold['is_start'].date()} – {fold['is_end'].date()}",
                        "OOS":         f"{fold['oos_start'].date()} – {fold['oos_end'].date()}",
                        "Beste MA":    f"{best_params['ma_type']} {best_params['ma_period']}",
                        "Bester SL":   f"{best_params['sl_pct']}%",
                        "IS Trades":   m_is["n"],
                        "IS PF":       m_is["pf"],
                        "IS Sharpe":   m_is["sharpe"],
                        "OOS Trades":  m_oos["n"],
                        "OOS PF":      m_oos["pf"],
                        "OOS Sharpe":  m_oos["sharpe"],
                        "OOS Ret %":   m_oos["total_ret"],
                        "Status":      status,
                    })

                progress_bar.progress(1.0, text="Walk-Forward abgeschlossen ✓")

                wfa_df = pd.DataFrame(wfa_rows)
                n_ok   = (wfa_df["Status"] == "✅ OOS-validiert").sum()
                n_fail = (wfa_df["Status"] == "❌ OOS-Fail").sum()

                # ── Gesamt-Badge ─────────────────────────────────────────────
                if n_ok >= int(wfa_min_folds):
                    badge_color, badge_text = "#22c55e", f"✅ ROBUST — {n_ok}/{len(wfa_df)} Folds OOS-validiert"
                elif n_ok > 0:
                    badge_color, badge_text = "#f0c040", f"⚠️ TEILWEISE — {n_ok}/{len(wfa_df)} Folds validiert"
                else:
                    badge_color, badge_text = "#ef5350", f"❌ NICHT ROBUST — 0/{len(wfa_df)} Folds bestanden"

                st.markdown(
                    f'<div style="background:{badge_color}22;border:1px solid {badge_color};'
                    f'border-radius:8px;padding:12px 18px;font-weight:700;font-size:1.1rem;'
                    f'color:{badge_color};margin:12px 0;">{badge_text}</div>',
                    unsafe_allow_html=True)

                # ── Fold-Tabelle ─────────────────────────────────────────────
                st.dataframe(wfa_df, use_container_width=True, hide_index=True)

                # ── OOS Equity Kurven ─────────────────────────────────────────
                st.subheader("OOS Equity Kurven je Fold")
                fig_wfa = go.Figure()
                colors  = ["#00d4aa","#ffa726","#42a5f5","#ef5350","#ab47bc","#66bb6a","#26c6da"]
                for fi, fold in enumerate(folds):
                    if fi >= len(wfa_rows):
                        break
                    row_ = wfa_rows[fi]
                    if row_["OOS Trades"] == 0:
                        continue
                    bp = next((r for r in [wfa_rows[fi]] if r.get("Beste MA")), None)
                    if bp is None:
                        continue
                    # Rebuild OOS equity für Plot
                    df_oos_plot = df_raw[(df_raw.index >= fold["oos_start"]) & (df_raw.index < fold["oos_end"])]
                    # best_params aus wfa_rows nicht direkt verfügbar → skip detailed plot
                    fig_wfa.add_annotation(x=fold["oos_start"], y=fi * 500 + 10000,
                                           text=f"F{fi+1}", showarrow=False,
                                           font=dict(color=colors[fi % len(colors)]))

                # Stattdessen: OOS-Rendite als Bar-Chart je Fold
                fold_labels = [f"F{r['Fold']}" for r in wfa_rows]
                oos_rets    = [r["OOS Ret %"] if isinstance(r["OOS Ret %"], (int,float)) else 0 for r in wfa_rows]
                bar_colors  = ["#22c55e" if v > 0 else "#ef5350" for v in oos_rets]
                fig_bar = go.Figure(go.Bar(x=fold_labels, y=oos_rets,
                                           marker_color=bar_colors, text=[f"{v:.1f}%" for v in oos_rets],
                                           textposition="outside"))
                fig_bar.update_layout(title="OOS-Rendite je Fold", height=300,
                                      template="plotly_dark", yaxis_title="Rendite %",
                                      margin=dict(t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── CSV Download ──────────────────────────────────────────────
                st.download_button("⬇️ WFA-Ergebnis als CSV",
                                   data=wfa_df.to_csv(index=False).encode(),
                                   file_name="momi_wfa.csv", mime="text/csv")


_WFA_COIN_CACHE_FILE = "wfa_coin_cache.json"
_GIST_DESCRIPTION   = "TACO Lab WFA Coin Cache"
_GIST_FILENAME      = "taco_wfa_cache.json"

def _gist_token():
    try:
        return st.secrets.get("GITHUB_GIST_TOKEN")
    except Exception:
        return None

def _find_gist_id(token: str) -> str | None:
    import requests
    try:
        r = requests.get(
            "https://api.github.com/gists",
            headers={"Authorization": f"token {token}"},
            timeout=10
        )
        if r.status_code == 200:
            for g in r.json():
                if g.get("description") == _GIST_DESCRIPTION:
                    return g["id"]
    except Exception:
        pass
    return None

def _save_coin_cache(coin_dict: dict) -> None:
    """Speichert wfa_coin_trades in GitHub Gist (überlebt Reboots) + lokaler Fallback."""
    import json, requests
    serializable = {}
    for tkr, cdata in coin_dict.items():
        serializable[tkr] = {
            "trades_json":  cdata["trades"].to_json(orient="records", date_format="iso"),
            "symbol_name":  cdata["symbol_name"],
            "base_params":  cdata["base_params"],
        }
    content = json.dumps(serializable)

    # GitHub Gist speichern
    token = _gist_token()
    if token:
        try:
            headers = {"Authorization": f"token {token}",
                       "Accept": "application/vnd.github.v3+json"}
            gist_id = _find_gist_id(token)
            payload = {"description": _GIST_DESCRIPTION,
                       "public": False,
                       "files": {_GIST_FILENAME: {"content": content}}}
            if gist_id:
                requests.patch(f"https://api.github.com/gists/{gist_id}",
                               headers=headers, json=payload, timeout=15)
            else:
                requests.post("https://api.github.com/gists",
                              headers=headers, json=payload, timeout=15)
        except Exception:
            pass

    # Lokaler Fallback (funktioniert zumindest innerhalb einer Session)
    try:
        with open(_WFA_COIN_CACHE_FILE, "w") as f:
            f.write(content)
    except Exception:
        pass

def _load_coin_cache() -> dict:
    """Lädt wfa_coin_trades — zuerst GitHub Gist, dann lokale Datei."""
    import json, requests

    raw = None

    # 1. Versuch: GitHub Gist
    token = _gist_token()
    if token:
        try:
            gist_id = _find_gist_id(token)
            if gist_id:
                r = requests.get(f"https://api.github.com/gists/{gist_id}",
                                 headers={"Authorization": f"token {token}"},
                                 timeout=10)
                if r.status_code == 200:
                    raw = json.loads(r.json()["files"][_GIST_FILENAME]["content"])
        except Exception:
            pass

    # 2. Fallback: lokale JSON-Datei
    if raw is None:
        try:
            with open(_WFA_COIN_CACHE_FILE, "r") as f:
                raw = json.load(f)
        except Exception:
            return {}

    result = {}
    for tkr, cdata in raw.items():
        try:
            df = pd.read_json(cdata["trades_json"], orient="records")
            if "Entry-Datum" in df.columns:
                df["Entry-Datum"] = pd.to_datetime(df["Entry-Datum"])
            if "Exit-Datum" in df.columns:
                df["Exit-Datum"] = pd.to_datetime(df["Exit-Datum"])
            result[tkr] = {
                "trades":      df,
                "symbol_name": cdata["symbol_name"],
                "base_params": cdata["base_params"],
            }
        except Exception:
            continue
    return result

# ── Generischer WFA/Ensemble-Ergebnis-Cache (GitHub Gist, überlebt Reboots/Redeploys) ──
# Nutzt denselben Gist wie der Coin-Cache oben (gleicher Token, gleiche Gist-ID), aber
# eine eigene Datei darin — begrenzte Anzahl fester "Slots" (z.B. "dax_wfa_close"),
# nicht pro Parameter-Kombination, damit die Gist-Datei nicht unbegrenzt wächst.
_WFA_RESULT_GIST_FILENAME = "taco_wfa_result_cache.json"
# Feste Feldreihenfolge für die komprimierte param_stability-Persistenz (Liste statt Dict
# pro Fold-Eintrag — spart die wiederholten Schlüsselnamen bei vielen Grid-Kombinationen).
_PARAM_STAB_FIELDS = ("n", "pf", "sharpe", "max_dd", "total_ret", "avg_ret")

def _wfa_json_encode(obj):
    """Wandelt DataFrames/Series/Tuple-Keys in JSON-kompatible Strukturen um (rekursiv)."""
    if isinstance(obj, pd.DataFrame):
        dt_cols = [c for c in obj.columns if pd.api.types.is_datetime64_any_dtype(obj[c])]
        return {"__df__": True, "json": obj.to_json(orient="records", date_format="iso"), "dt_cols": dt_cols}
    if isinstance(obj, pd.Series):
        return {"__series__": True,
                "index": [str(i) for i in obj.index],
                "index_is_dt": isinstance(obj.index, pd.DatetimeIndex),
                "values": [None if pd.isna(v) else float(v) for v in obj.values],
                "name": obj.name}
    if isinstance(obj, pd.Timestamp):
        return {"__ts__": True, "iso": obj.isoformat()}
    if isinstance(obj, dict):
        if any(isinstance(k, tuple) for k in obj.keys()):
            return {"__tdict__": True,
                    "items": [[list(k) if isinstance(k, tuple) else k, _wfa_json_encode(v)] for k, v in obj.items()]}
        return {k: _wfa_json_encode(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [_wfa_json_encode(x) for x in obj]}
    if isinstance(obj, list):
        return [_wfa_json_encode(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def _wfa_json_decode(obj):
    import io
    if isinstance(obj, dict):
        if obj.get("__df__"):
            df = pd.read_json(io.StringIO(obj["json"]), orient="records")
            for c in obj.get("dt_cols", []):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c])
            return df
        if obj.get("__series__"):
            idx = pd.to_datetime(obj["index"]) if obj.get("index_is_dt") else obj["index"]
            return pd.Series(obj["values"], index=idx, name=obj.get("name"))
        if obj.get("__ts__"):
            return pd.Timestamp(obj["iso"])
        if obj.get("__tdict__"):
            return {tuple(k) if isinstance(k, list) else k: _wfa_json_decode(v) for k, v in obj["items"]}
        if obj.get("__tuple__"):
            return tuple(_wfa_json_decode(x) for x in obj["items"])
        return {k: _wfa_json_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_wfa_json_decode(x) for x in obj]
    return obj

def _fetch_gist_file_full_content(file_entry: dict, headers: dict) -> str:
    """GitHub trunkiert 'content' in Gist-API-Antworten bei Dateien > 1MB und setzt
    'truncated': true — in dem Fall muss der volle Inhalt separat über 'raw_url'
    geholt werden, sonst ist die zurückgegebene 'content' kein valides JSON mehr."""
    import requests
    if file_entry.get("truncated") and file_entry.get("raw_url"):
        r = requests.get(file_entry["raw_url"], headers=headers, timeout=30)
        r.raise_for_status()
        return r.text
    return file_entry.get("content", "")


def _save_wfa_result(slot: str, payload: dict) -> tuple[bool, str]:
    """Speichert ein WFA/Ensemble-Ergebnis dauerhaft im GitHub Gist unter einem festen Slot-Namen.
    Gibt (erfolgreich, Grund) zurück — der Grund wird im UI angezeigt, damit ein Fehlschlag
    (fehlender Token, Timeout, GitHub-API-Fehler …) nicht stillschweigend verschwindet."""
    import requests
    token = _gist_token()
    if not token:
        return False, "kein GITHUB_GIST_TOKEN in den Streamlit Secrets hinterlegt"
    _payload_kb = 0
    try:
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        gist_id = _find_gist_id(token)
        existing = {}
        if gist_id:
            r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=15)
            if r.status_code == 200:
                files = r.json().get("files", {}) or {}
                if _WFA_RESULT_GIST_FILENAME in files:
                    _raw = _fetch_gist_file_full_content(files[_WFA_RESULT_GIST_FILENAME], headers)
                    try:
                        existing = json.loads(_raw)
                    except json.JSONDecodeError:
                        # Vorhandene Datei ist beschädigt/nicht mehr lesbar — nicht abbrechen,
                        # sondern beim nächsten Speichern mit frischem, validem Inhalt überschreiben.
                        existing = {}
            elif r.status_code not in (404,):
                return False, f"GET Gist fehlgeschlagen (HTTP {r.status_code})"
        existing[slot] = _wfa_json_encode(payload)
        content = json.dumps(existing)
        _payload_kb = len(content) / 1024
        gist_payload = {"description": _GIST_DESCRIPTION, "public": False,
                         "files": {_WFA_RESULT_GIST_FILENAME: {"content": content}}}
        if gist_id:
            resp = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=headers, json=gist_payload, timeout=45)
        else:
            resp = requests.post("https://api.github.com/gists", headers=headers, json=gist_payload, timeout=45)
        if resp.status_code in (200, 201):
            return True, ""
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return False, f"Zeitüberschreitung beim Speichern ({_payload_kb:.0f} KB Payload)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _load_wfa_result(slot: str) -> dict | None:
    """Lädt ein zuvor gespeichertes WFA/Ensemble-Ergebnis aus dem GitHub Gist, falls vorhanden."""
    import requests
    token = _gist_token()
    if not token:
        return None
    try:
        gist_id = _find_gist_id(token)
        if not gist_id:
            return None
        _headers = {"Authorization": f"token {token}"}
        r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=_headers, timeout=15)
        if r.status_code != 200:
            return None
        files = r.json().get("files", {}) or {}
        if _WFA_RESULT_GIST_FILENAME not in files:
            return None
        _raw = _fetch_gist_file_full_content(files[_WFA_RESULT_GIST_FILENAME], _headers)
        try:
            existing = json.loads(_raw)
        except json.JSONDecodeError:
            return None
        if slot not in existing:
            return None
        return _wfa_json_decode(existing[slot])
    except Exception:
        return None

def render_btc_wfa() -> None:
    """Crypto WeekdayMA WFA — Sonntag Entry / Montag Exit auf BTC-USD Daily."""
    import datetime as _dt
    from itertools import product as _prod

    # Coin-Cache aus Datei laden falls session_state noch leer
    if "wfa_coin_trades" not in st.session_state:
        st.session_state["wfa_coin_trades"] = _load_coin_cache()

    # btc_wfa_ran + mc_trades aus persistentem Coin-Cache wiederherstellen (überlebt App-Neustart)
    _symbol_map_early = {
        "BTC — Bitcoin":       ("BTC-USD",  _dt.date(2018, 1, 1)),
        "ETH — Ethereum":      ("ETH-USD",  _dt.date(2018, 1, 1)),
        "SOL — Solana":        ("SOL-USD",  _dt.date(2020, 4, 1)),
        "XRP — Ripple":        ("XRP-USD",  _dt.date(2018, 1, 1)),
        "ADA — Cardano":       ("ADA-USD",  _dt.date(2018, 1, 1)),
        "DOGE — Dogecoin":     ("DOGE-USD", _dt.date(2018, 1, 1)),
        "AVAX — Avalanche":    ("AVAX-USD", _dt.date(2020, 9, 1)),
        "LINK — Chainlink":    ("LINK-USD", _dt.date(2019, 1, 1)),
    }
    _early_key = st.session_state.get("btc_symbol", "BTC — Bitcoin")
    if _early_key not in _symbol_map_early:
        _early_key = "BTC — Bitcoin"
    selected_name = _early_key
    _yf_ticker, _default_start = _symbol_map_early[selected_name]

    # Wenn WFA-Ergebnis für diesen Ticker im JSON-Cache vorhanden → Session-State wiederherstellen
    _coin_cache_for_ticker = st.session_state.get("wfa_coin_trades", {}).get(_yf_ticker)
    if _coin_cache_for_ticker and not _coin_cache_for_ticker["trades"].empty:
        if not st.session_state.get("btc_wfa_ran"):
            st.session_state["btc_wfa_ran"] = True
        if f"mc_trades_{_yf_ticker}" not in st.session_state:
            st.session_state[f"mc_trades_{_yf_ticker}"] = _coin_cache_for_ticker["trades"]

    st.header(f"{selected_name.split('—')[0].strip()} WeekdayMA — Walk-Forward Analyse")

    with st.expander("ℹ️ Was wird hier getestet und wie funktioniert es?", expanded=False):
        st.markdown("""
**Strategie:** Long-only Wochentag-Momentum auf BTC/USD (Daily-Daten via Yahoo Finance)
- **Entry:** Jeden Sonntag, wenn Kurs über dem gleitenden Durchschnitt liegt (MA-Filter) und der Trend stark genug ist (ADX-Filter)
- **Exit:** Montag-Schlusskurs (zeitbasiert) — oder früher durch Stop-Loss / Trailing-Stop
- **Idee:** BTC tendiert historisch dazu, Wochenenden für Aufwärtsbewegungen zu nutzen (weniger institutioneller Verkaufsdruck)

---

**Walk-Forward Analyse (WFA) — wie es funktioniert:**

Der gesamte Zeitraum (z.B. 2018–2026) wird in mehrere **Folds** aufgeteilt. Jeder Fold besteht aus zwei Phasen:

```
Fold 1: [2018–2019 optimieren (IS)] → [2019–2020 blind testen (OOS)]
Fold 2: [2019–2020 optimieren (IS)] → [2020–2021 blind testen (OOS)]
Fold 3: [2020–2021 optimieren (IS)] → [2021–2022 blind testen (OOS)]
... usw.
```

- **In-Sample (IS):** Das System testet automatisch hunderte Parameter-Kombinationen und findet die beste für diesen Zeitraum
- **Out-of-Sample (OOS):** Diese beste Kombination wird auf dem **nächsten, unbekannten** Zeitraum getestet — ohne Anpassung
- **Die entscheidende Frage:** Sind die besten Parameter über alle Folds ähnlich? Funktionieren sie auch blind?
  - ✅ **Ja → ROBUST** — die Strategie hat eine echte Edge, kein Zufall
  - ❌ **Nein → OVERFITTED** — die Parameter funktionieren nur auf den Trainingsdaten, nicht in der Realität

---

**Was bedeuten die WFA-Ergebnisse für TradingView?**

Wenn der WFA z.B. `EMA 50 + SL 1.8% + Trail 0.3%` als stabile Kombination identifiziert, bedeutet das: **Diese Parameter haben in mehreren unabhängigen Zeiträumen funktioniert** — nicht nur einmal zufällig.

Du kannst diese Werte in deinem Pine Script in TradingView einstellen. **Wichtig:** TradingView macht dann keinen WFA — es ist ein normaler Backtest mit fixen Parametern. TradingView dient nur zur visuellen Kontrolle (Chart, Trades, Equity-Kurve). Den Robustheitstest hat Python bereits erledigt.

---

**Warum unterscheidet sich die Tradeanzahl von TradingView?**
- Python nutzt **Daily-Daten**: 1 Bar = 1 Tag, Entry am Tagesschlusskurs des Sonntags
- TradingView nutzt **4H-Daten**: Entry nur wenn die exakte Uhrzeit getroffen wird + EMA/ADX reagiert feiner
- **Grundregel:** Mehr Trades ≠ besser. Qualität entscheidet die Win-Rate — TradingView filtert strenger
        """)

    st.caption(f"Strategie: Sonntag-Entry / Montag-Exit auf {_yf_ticker} · Daily-Daten via yfinance · Rollierender IS/OOS-Test")

    # ════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Strategie-Parameter (vorausgefüllt mit TradingView-Bestresultat)
    # ════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("---")
        st.subheader("Crypto WFA: Parameter")
        _symbol_map = {
            "BTC — Bitcoin":       ("BTC-USD",  _dt.date(2018, 1, 1)),
            "ETH — Ethereum":      ("ETH-USD",  _dt.date(2018, 1, 1)),
            "SOL — Solana":        ("SOL-USD",  _dt.date(2020, 4, 1)),
            "XRP — Ripple":        ("XRP-USD",  _dt.date(2018, 1, 1)),
            "ADA — Cardano":       ("ADA-USD",  _dt.date(2018, 1, 1)),
            "DOGE — Dogecoin":     ("DOGE-USD", _dt.date(2018, 1, 1)),
            "AVAX — Avalanche":    ("AVAX-USD", _dt.date(2020, 9, 1)),
            "LINK — Chainlink":    ("LINK-USD", _dt.date(2019, 1, 1)),
        }
        selected_name = st.selectbox("Symbol", list(_symbol_map.keys()), key="btc_symbol")
        _yf_ticker, _default_start = _symbol_map[selected_name]
        btc_start = st.date_input("Daten ab", _default_start, key="btc_start")
        btc_end   = st.date_input("Daten bis", _dt.date.today(), key="btc_end")

    # ── Strategie-Parameter ──────────────────────────────────────────────
    st.subheader("Strategie-Parameter")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**Entry / Exit**")
        day_map_full = {"Montag":0,"Dienstag":1,"Mittwoch":2,"Donnerstag":3,"Freitag":4,"Samstag":5,"Sonntag":6}
        entry_day  = st.selectbox("Entry-Tag", list(day_map_full.keys()), index=6, key="btc_ed")  # Sonntag
        entry_hour = st.number_input("Entry-Stunde", 0, 23, 22, key="btc_eh")
        exit_day   = st.selectbox("Exit-Tag",   list(day_map_full.keys()), index=0, key="btc_xd")  # Montag
        exit_hour  = st.number_input("Exit-Stunde",  0, 23, 22, key="btc_xh")
    with pc2:
        st.markdown("**MA & Filter**")
        ma_type     = st.selectbox("MA-Typ",      ["EMA","SMA"],                          key="btc_mt")
        ma_period   = st.selectbox("MA-Periode",  [20, 50, 100, 200], index=0,            key="btc_mp")
        filter_mode = st.selectbox("Filter-Modus",["Kein Filter (nur Zeit)","Close > MA","MA steigt"], key="btc_fm")
        use_adx     = st.checkbox("ADX-Filter", False, key="btc_adx")
        adx_thresh  = st.number_input("ADX-Schwelle", 5, 50, 20, key="btc_adxt")
    with pc3:
        st.markdown("**Risk Management**")
        use_sl     = st.checkbox("Stop-Loss", True, key="btc_sl")
        sl_pct     = st.number_input("SL %", 0.1, 20.0, 1.8, step=0.1, key="btc_slp")
        use_trail  = st.checkbox("Trailing Stop", True, key="btc_tr")
        trail_trig = st.number_input("Trail-Trigger %", 0.1, 10.0, 0.2, step=0.1, key="btc_tt")
        trail_off  = st.number_input("Trail-Abstand %", 0.1, 10.0, 0.2, step=0.1, key="btc_to")
        st.markdown("---")
        risk_pct   = st.number_input("Risiko pro Trade %", 0.1, 5.0, 1.0, step=0.1, key="btc_risk",
                                     help="1% = bei 100.000€ Konto riskierst du 1.000€ pro Trade (basierend auf SL-Abstand)")

    # ── WFA-Konfiguration ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Walk-Forward Konfiguration")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        is_months  = st.number_input("IS-Fenster (Monate)", 6, 36, 18, key="btc_is")
        oos_months = st.number_input("OOS-Fenster (Monate)", 3, 18, 12, key="btc_oos")
    with wc2:
        min_trades  = st.number_input("Min. IS-Trades", 5, 50, 10, key="btc_mint")
        opt_metric  = st.selectbox("Optimierungsziel", ["pf","sharpe","wr"], format_func=lambda x: {"pf":"Profit Factor","sharpe":"Sharpe","wr":"Win-Rate"}[x], key="btc_om")
    with wc3:
        min_folds   = st.number_input("Min. Folds für ✅ ROBUST", 2, 8, 4, key="btc_mf")
        st.markdown(" ")

    # ── Grid-Suchraum ─────────────────────────────────────────────────────
    with st.expander("Grid-Suchraum (IS-Optimierung)"):
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            g_sl   = st.multiselect("SL %",           [0.5,1.0,1.5,1.8,2.0,2.5,3.0], default=[1.0,1.5,1.8,2.0], key="btc_gsl")
            g_ma   = st.multiselect("MA-Periode",      [20,50,100,200],                default=[20,50],            key="btc_gma")
        with gc2:
            g_fm   = st.multiselect("Filter-Modus",   ["Kein Filter (nur Zeit)","Close > MA","MA steigt"],
                                     default=["Kein Filter (nur Zeit)","Close > MA"],   key="btc_gfm")
            g_adx  = st.multiselect("ADX-Schwelle",   [15,20,25,30],                  default=[20],               key="btc_gadx")
        with gc3:
            g_tt   = st.multiselect("Trail-Trigger %",[0.1,0.2,0.3,0.5],              default=[0.2,0.3],          key="btc_gtt")
            g_to   = st.multiselect("Trail-Abstand %",[0.1,0.2,0.3,0.4],              default=[0.2,0.3],          key="btc_gto")

        st.markdown("---")
        _opt_days = st.checkbox("Entry/Exit-Tag mit optimieren", value=False, key="btc_opt_days",
                                help="WFA testet automatisch verschiedene Wochentag-Kombinationen — erhöht die Laufzeit deutlich")
        if _opt_days:
            _dow_opts = {"Montag":0,"Dienstag":1,"Mittwoch":2,"Donnerstag":3,"Freitag":4,"Samstag":5,"Sonntag":6}
            gd1, gd2 = st.columns(2)
            g_entry_days = gd1.multiselect("Entry-Tag testen",
                                            list(_dow_opts.keys()), default=["Freitag","Samstag","Sonntag"],
                                            key="btc_g_eday")
            g_exit_days  = gd2.multiselect("Exit-Tag testen",
                                            list(_dow_opts.keys()), default=["Montag","Dienstag"],
                                            key="btc_g_xday")
            g_entry_dows = [_dow_opts[d] for d in g_entry_days]
            g_exit_dows  = [_dow_opts[d] for d in g_exit_days]
            n_day_combos = len(g_entry_dows) * len(g_exit_dows)
            n_total = len(g_sl)*len(g_ma)*len(g_fm)*len(g_tt)*len(g_to)*n_day_combos
            st.caption(f"⚠️ {n_day_combos} Tag-Kombinationen × Grid = **{n_total} Kombinationen je Fold** — kann mehrere Minuten dauern")
        else:
            g_entry_dows = [day_map_full[entry_day]]
            g_exit_dows  = [day_map_full[exit_day]]

    _btn_col1, _btn_col2 = st.columns([1, 1])
    with _btn_col1:
        run_btn = st.button("🔄 WFA starten", type="primary", use_container_width=True, key="btc_run")
    with _btn_col2:
        _ens_quick = st.button("⚡ Ensemble WFA starten (5×)", use_container_width=True, key="btc_ens_quick",
                               help="Startet direkt den 5-fachen Durchlauf — WFA muss einmal zuvor gelaufen sein",
                               disabled=not st.session_state.get("btc_wfa_ran", False))
    if run_btn:
        st.session_state["btc_wfa_ran"] = True
        st.session_state["btc_wfa_params_key"] = str(btc_start) + str(btc_end) + str(is_months) + str(oos_months)
    if _ens_quick:
        st.session_state["ens_running"] = True

    # ════════════════════════════════════════════════════════════════════════
    # DATEN LADEN — läuft immer, liefert Trades für Monte Carlo
    # ════════════════════════════════════════════════════════════════════════
    try:
        import yfinance as yf
    except ImportError:
        st.error("`pip install yfinance` fehlt.")
        return

    cache_key = f"btc_df_{_yf_ticker}_{btc_start}_{btc_end}"
    if cache_key not in st.session_state or run_btn:
        with st.spinner(f"{_yf_ticker} Daily-Daten laden …"):
            st.session_state[cache_key] = yf.download(
                _yf_ticker, start=str(btc_start), end=str(btc_end),
                interval="1d", auto_adjust=True, progress=False)
    df_raw = st.session_state[cache_key]

    if df_raw.empty:
        st.error("Keine BTC-Daten geladen.")
        return
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)

    st.success(f"✓ {len(df_raw)} Daily-Bars geladen ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    # ════════════════════════════════════════════════════════════════════════
    # FULL-SAMPLE BACKTEST (zur Orientierung)
    # ════════════════════════════════════════════════════════════════════════
    base_params = {
        "entry_dow":   day_map_full[entry_day],
        "exit_dow":    day_map_full[exit_day],
        "entry_hour":  int(entry_hour),
        "entry_min":   0,
        "exit_hour":   int(exit_hour),
        "exit_min":    0,
        "ma_type":     ma_type,
        "ma_period":   int(ma_period),
        "filter_mode": filter_mode,
        "use_adx":     use_adx,
        "adx_thresh":  float(adx_thresh),
        "sl_pct":      float(sl_pct)  if use_sl    else 999.0,
        "use_trail":   use_trail,
        "trail_trig":  float(trail_trig),
        "trail_off":   float(trail_off),
        "risk_pct":    float(risk_pct),
    }

    with st.spinner("Full-Sample Backtest …"):
        tr_full, eq_full = _momi_backtest_engine(df_raw.copy(), base_params)
        m_full = _momi_metrics(tr_full, eq_full)

    # Sofort speichern — MC findet Trades auch ohne WFA-Lauf
    if not tr_full.empty:
        st.session_state[f"mc_trades_{_yf_ticker}"] = tr_full

    st.markdown("---")
    st.subheader("Full-Sample Ergebnis (zur Orientierung, KEIN WFA)")
    fc1,fc2,fc3,fc4,fc5 = st.columns(5)
    fc1.metric("Rendite",       f"{m_full['total_ret']:.1f}%")
    fc2.metric("Trades",         m_full['n'])
    fc3.metric("Win-Rate",      f"{m_full['wr']:.1f}%")
    fc4.metric("Profit Factor", f"{m_full['pf']:.2f}")
    fc5.metric("Sharpe",        f"{m_full['sharpe']:.2f}")

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(x=eq_full.index, y=eq_full.values,
                                  line=dict(color="#f7931a", width=2), name="Equity (Full)"))
    fig_full.update_layout(title="Full-Sample Equity Curve", height=250,
                           template="plotly_dark", margin=dict(t=35,b=15))
    st.plotly_chart(fig_full, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # WALK-FORWARD ANALYSE
    # ════════════════════════════════════════════════════════════════════════
    _wfa_enabled = st.session_state.get("btc_wfa_ran", False)

    st.markdown("---")
    st.subheader("Walk-Forward Analyse")

    if not _wfa_enabled:
        n_months_total = (btc_end.year - btc_start.year) * 12 + (btc_end.month - btc_start.month)
        est_folds = max(0, (n_months_total - is_months) // oos_months)
        st.info(f"WFA noch nicht gestartet — klicke '🔄 WFA starten'. "
                f"Geschätzte Folds: **{est_folds}**")

    folds = []
    if _wfa_enabled:
        is_d  = pd.DateOffset(months=int(is_months))
        oos_d = pd.DateOffset(months=int(oos_months))
        fs    = df_raw.index[0]
        while True:
            ie = fs + is_d
            oe = ie + oos_d
            if oe > df_raw.index[-1]:
                break
            folds.append({"is_start": fs, "is_end": ie, "oos_start": ie, "oos_end": oe})
            fs = fs + oos_d

        if len(folds) < 2:
            st.warning("Zu wenig Daten für WFA. Zeitraum verlängern oder IS/OOS-Fenster verkleinern.")
            _wfa_enabled = False
        else:
            st.info(f"**{len(folds)} Folds** · IS {is_months}M / OOS {oos_months}M  "
                    f"· Grid-Größe: {len(g_sl)*len(g_ma)*len(g_fm)*len(g_tt)*len(g_to)} Kombinationen je Fold")

    wfa_cache_key = (f"btc_wfa_results_{_yf_ticker}_{btc_start}_{btc_end}"
                     f"_{is_months}_{oos_months}_{entry_day}_{exit_day}"
                     f"_{ma_type}_{sl_pct}_{use_trail}_{trail_trig}_{trail_off}")

    if _wfa_enabled and (run_btn or wfa_cache_key not in st.session_state):
        progress = st.progress(0, text="Walk-Forward läuft …")
        wfa_rows, oos_equities = [], []
        param_stability: dict = {}

        for fi, fold in enumerate(folds):
            progress.progress(fi / len(folds), text=f"Fold {fi+1}/{len(folds)} — optimiere IS …")
    
            df_is  = df_raw[(df_raw.index >= fold["is_start"]) & (df_raw.index < fold["is_end"])].copy()
            df_oos = df_raw[(df_raw.index >= fold["oos_start"]) & (df_raw.index < fold["oos_end"])].copy()
    
            if len(df_is) < 30 or len(df_oos) < 5:
                continue
    
            # IS Grid Search — teste ALLE Kombinationen und sammle OOS-Ergebnis je Combo
            best_score, best_p = -np.inf, None
            adx_grid = g_adx if use_adx else [float(adx_thresh)]
    
            for sl_, ma_, fm_, adx_, tt_, to_, ed_, xd_ in _prod(
                    g_sl, g_ma, g_fm, adx_grid, g_tt, g_to, g_entry_dows, g_exit_dows):
                if ed_ == xd_:
                    continue  # Entry- und Exit-Tag müssen verschieden sein
                p = {**base_params,
                     "sl_pct":      sl_,
                     "ma_period":   ma_,
                     "filter_mode": fm_,
                     "adx_thresh":  adx_,
                     "trail_trig":  tt_,
                     "trail_off":   to_,
                     "entry_dow":   ed_,
                     "exit_dow":    xd_}
                try:
                    tr_, eq_ = _momi_backtest_engine(df_is.copy(), p)
                except Exception:
                    continue
                if len(tr_) < int(min_trades):
                    continue
                m_ = _momi_metrics(tr_, eq_)
                score = m_[opt_metric]
                if score > best_score:
                    best_score, best_p = score, p.copy()
    
                # OOS sofort für diese Kombination berechnen → Stabilitätsanalyse
                try:
                    tr_oos_, eq_oos_ = _momi_backtest_engine(df_oos.copy(), p)
                    m_oos_ = _momi_metrics(tr_oos_, eq_oos_)
                    key = (sl_, tt_, to_, ma_, fm_)
                    if key not in param_stability:
                        param_stability[key] = []
                    param_stability[key].append(m_oos_)
                except Exception:
                    pass
    
            if best_p is None:
                wfa_rows.append({"Fold": fi+1,
                                 "IS": f"{fold['is_start'].date()} – {fold['is_end'].date()}",
                                 "OOS": f"{fold['oos_start'].date()} – {fold['oos_end'].date()}",
                                 "Bester SL":"–","Bestes MA":"–","Bester FM":"–",
                                 "IS Trades":0,"IS PF":"–","IS Sharpe":"–",
                                 "OOS Trades":0,"OOS PF":"–","OOS Ret %":"–","Ø Ret/Trade %":"–","OOS Sharpe":"–",
                                 "Status":"⚠️ Kein IS-Ergebnis"})
                continue
    
            tr_is,  eq_is  = _momi_backtest_engine(df_is.copy(),  best_p)
            tr_oos, eq_oos = _momi_backtest_engine(df_oos.copy(), best_p)
            m_is  = _momi_metrics(tr_is,  eq_is)
            m_oos = _momi_metrics(tr_oos, eq_oos)
    
            oos_equities.append((fi+1, eq_oos))
    
            ok = m_oos["n"] >= 1 and m_oos["pf"] > 1.0 and m_oos["total_ret"] > 0
            _dow_names = {0:"Mo",1:"Di",2:"Mi",3:"Do",4:"Fr",5:"Sa",6:"So"}
            wfa_rows.append({
                "Fold":       fi+1,
                "IS":         f"{fold['is_start'].date()} – {fold['is_end'].date()}",
                "OOS":        f"{fold['oos_start'].date()} – {fold['oos_end'].date()}",
                "Bester SL":  f"{best_p['sl_pct']}%",
                "Bestes MA":  f"{best_p['ma_type']} {best_p['ma_period']}",
                "Bester FM":  best_p['filter_mode'],
                "Entry→Exit": f"{_dow_names.get(best_p['entry_dow'],'?')}→{_dow_names.get(best_p['exit_dow'],'?')}",
                "IS Trades":  m_is["n"],
                "IS PF":      m_is["pf"],
                "IS Sharpe":  m_is["sharpe"],
                "OOS Trades":    m_oos["n"],
                "OOS PF":        m_oos["pf"],
                "OOS Ret %":     m_oos["total_ret"],
                "Ø Ret/Trade %": m_oos["avg_ret"],
                "OOS Sharpe":    m_oos["sharpe"],
                "Status":        "✅ Bestanden" if ok else "❌ Fail",
            })
    
            progress.progress(1.0, text="Walk-Forward abgeschlossen ✓")
        st.session_state[wfa_cache_key] = {
            "wfa_rows":        wfa_rows,
            "oos_equities":    oos_equities,
            "param_stability": param_stability,
            "base_params":     base_params,
            "grids":           (g_sl, g_ma, g_fm, g_tt, g_to),
            "full_trades":     tr_full,
        }
        # Trades auch direkt unter Ticker-Key speichern — MC findet sie ohne exakten Cache-Key
        st.session_state[f"mc_trades_{_yf_ticker}"] = tr_full
        # Multi-Coin Cache: Trades für diesen Ticker speichern + persistent sichern
        try:
            _mc_tr, _ = _momi_backtest_engine(df_raw.copy(), base_params)
            if "wfa_coin_trades" not in st.session_state:
                st.session_state["wfa_coin_trades"] = {}
            st.session_state["wfa_coin_trades"][_yf_ticker] = {
                "trades":       _mc_tr,
                "symbol_name":  selected_name,
                "base_params":  base_params,
            }
            _save_coin_cache(st.session_state["wfa_coin_trades"])
        except Exception:
            pass

    # Ergebnisse aus Cache laden
    if not _wfa_enabled or wfa_cache_key not in st.session_state:
        _cached = None
    else:
        _cached = st.session_state[wfa_cache_key]
    if _cached is None:
        wfa_rows, oos_equities, param_stability = [], [], {}
        base_params_display = base_params
        g_sl, g_ma, g_fm, g_tt, g_to = [], [], [], [], []
        _wfa_enabled = False
    else:
        wfa_rows        = _cached["wfa_rows"]
        oos_equities    = _cached["oos_equities"]
        param_stability = _cached["param_stability"]
        base_params     = _cached["base_params"]
        g_sl, g_ma, g_fm, g_tt, g_to = _cached["grids"]

    if _wfa_enabled and not wfa_rows:
        st.error("Keine WFA-Ergebnisse — Parameter oder Zeitraum anpassen.")
        _wfa_enabled = False

    if _wfa_enabled:
        wfa_df = pd.DataFrame(wfa_rows)
        n_ok   = (wfa_df["Status"] == "✅ Bestanden").sum()
        n_tot  = len(wfa_df)

        # ── Gesamt-Badge ─────────────────────────────────────────────────────
        if n_ok >= int(min_folds):
            bc, bt = "#22c55e", f"✅ ROBUST — {n_ok}/{n_tot} Folds bestanden · Strategie empfohlen"
        elif n_ok >= 2:
            bc, bt = "#f0c040", f"⚠️ INSTABIL — nur {n_ok}/{n_tot} Folds bestanden · mit Vorsicht handeln"
        elif n_ok == 1:
            bc, bt = "#ef5350", f"❌ NICHT EMPFOHLEN — nur 1/{n_tot} Fold bestanden · Strategie funktioniert auf diesem Asset NICHT zuverlässig"
        else:
            bc, bt = "#ef5350", f"❌ GESCHEITERT — 0/{n_tot} Folds bestanden · Strategie NICHT für dieses Asset geeignet"

        st.markdown(
            f'<div style="background:{bc}22;border:2px solid {bc};border-radius:10px;'
            f'padding:16px 24px;font-weight:800;font-size:1.2rem;color:{bc};margin:16px 0;">'
            f'{bt}</div>', unsafe_allow_html=True)

        # ── KPI-Zusammenfassung über alle OOS-Folds ───────────────────────────
        oos_pfs   = [r["OOS PF"]    for r in wfa_rows if isinstance(r["OOS PF"],    (int,float))]
        oos_rets  = [r["OOS Ret %"] for r in wfa_rows if isinstance(r["OOS Ret %"], (int,float))]
        oos_shs   = [r["OOS Sharpe"] for r in wfa_rows if isinstance(r.get("OOS Sharpe"), (int,float))]

        sc1,sc2,sc3,sc4 = st.columns(4)
        sc1.metric("Ø OOS Profit Factor", f"{np.mean(oos_pfs):.2f}"  if oos_pfs  else "–")
        sc2.metric("Ø OOS Rendite",       f"{np.mean(oos_rets):.1f}%" if oos_rets else "–")
        sc3.metric("Ø OOS Sharpe",        f"{np.mean(oos_shs):.2f}"   if oos_shs  else "–")
        sc4.metric("Bestandene Folds",    f"{n_ok}/{n_tot}")

        # ── Fold-Tabelle ──────────────────────────────────────────────────────
        st.subheader("Fold-Ergebnisse")
        def _color_status(v):
            if v == "✅ Bestanden": return "color:#22c55e;font-weight:700"
            if v == "❌ Fail":      return "color:#ef5350;font-weight:700"
            return "color:#f0c040"
        def _color_num(v):
            if isinstance(v, (int,float)):
                return "color:#22c55e" if v > 0 else "color:#ef5350"
            return ""
        num_cols = [c for c in ["OOS Ret %","OOS PF","OOS Sharpe","Ø Ret/Trade %"] if c in wfa_df.columns]
        styled = wfa_df.style\
            .map(_color_status, subset=["Status"])\
            .map(_color_num,    subset=num_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── OOS Rendite Bar-Chart ─────────────────────────────────────────────
        st.subheader("OOS-Rendite je Fold")
        fold_labels = [f"Fold {r['Fold']}" for r in wfa_rows]
        bar_colors  = ["#22c55e" if isinstance(r["OOS Ret %"],(int,float)) and r["OOS Ret %"]>0
                       else "#ef5350" for r in wfa_rows]
        bar_vals    = [r["OOS Ret %"] if isinstance(r["OOS Ret %"],(int,float)) else 0 for r in wfa_rows]

        fig_bar = go.Figure(go.Bar(
            x=fold_labels, y=bar_vals, marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in bar_vals], textposition="outside",
            width=0.5))
        fig_bar.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
        fig_bar.update_layout(title="OOS-Rendite je Fold (grün = profitabel)",
                              height=380, template="plotly_dark",
                              yaxis_title="Rendite %",
                              xaxis=dict(tickfont=dict(size=13)),
                              margin=dict(t=50,b=60,l=60,r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Gestapelte OOS Equity Curves ──────────────────────────────────────
        if oos_equities:
            # ── Kombinierte OOS Equity (Folds hintereinander = simulierter Live-Handel) ──
            st.subheader("Kombinierte OOS Equity Kurve")
            st.caption("Alle OOS-Folds hintereinander — so hätte sich das Kapital im echten Handel entwickelt (nur blind getestete Perioden, kein IS)")

            # Folds chronologisch sortieren und aneinanderhängen
            oos_equities_sorted = sorted(oos_equities, key=lambda x: x[1].index[0])
            combined_vals, combined_idx = [], []
            running_capital = 10_000.0
            for fi, eq in oos_equities_sorted:
                scale = running_capital / eq.iloc[0]
                scaled = eq * scale
                combined_vals.extend(scaled.values.tolist())
                combined_idx.extend(eq.index.tolist())
                running_capital = scaled.iloc[-1]

            combined_eq = pd.Series(combined_vals, index=combined_idx)

            # Drawdown berechnen
            roll_max = combined_eq.cummax()
            drawdown = (combined_eq - roll_max) / roll_max * 100

            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(
                x=combined_eq.index, y=combined_eq.values,
                fill="tozeroy", fillcolor="rgba(247,147,26,0.15)",
                line=dict(color="#f7931a", width=2.5),
                name="Equity (OOS kombiniert)"))
            # Fold-Grenzen als vertikale Linien
            colors_fold = ["#42a5f5","#00d4aa","#ab47bc","#ffa726","#66bb6a","#ef5350","#26c6da","#f7931a"]
            for i, (fi, eq) in enumerate(oos_equities_sorted):
                _fold_color = colors_fold[i % len(colors_fold)]
                fig_combined.add_vline(x=eq.index[0], line_dash="dot", line_color=_fold_color)
                fig_combined.add_annotation(x=eq.index[0], y=1, yref="paper", yanchor="bottom",
                                            text=f"F{fi}", showarrow=False, font=dict(color=_fold_color))
            fig_combined.add_hline(y=10_000, line_color="white", line_dash="dash", line_width=1)
            total_ret = (running_capital - 10_000) / 10_000 * 100
            fig_combined.update_layout(
                title=f"OOS Equity — 10.000€ Start → {running_capital:,.0f}€ ({total_ret:+.1f}%) | Nur blind getestete Perioden",
                height=420, template="plotly_dark",
                yaxis_title="Kapital (€)", xaxis_title="",
                margin=dict(t=55, b=20, l=70, r=20))
            st.plotly_chart(fig_combined, use_container_width=True)

            # Drawdown Chart
            fig_dd = go.Figure(go.Scatter(
                x=combined_eq.index, y=drawdown.values,
                fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                line=dict(color="#ef5350", width=1.5), name="Drawdown %"))
            fig_dd.update_layout(
                title=f"Drawdown — Max: {drawdown.min():.1f}%",
                height=200, template="plotly_dark",
                yaxis_title="DD %", margin=dict(t=40, b=20, l=70, r=20))
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── Einzelne Folds übereinander (zum Vergleich) ───────────────────
            with st.expander("Einzelne OOS-Folds im Vergleich"):
                fig_eq = go.Figure()
                colors = ["#f7931a","#00d4aa","#42a5f5","#ab47bc","#ffa726","#66bb6a","#ef5350","#26c6da"]
                for fi, eq in oos_equities:
                    norm = eq / eq.iloc[0] * 100
                    fig_eq.add_trace(go.Scatter(x=eq.index, y=norm.values,
                                                name=f"Fold {fi}",
                                                line=dict(color=colors[(fi-1) % len(colors)], width=1.5)))
                fig_eq.add_hline(y=100, line_color="white", line_dash="dash", line_width=1)
                fig_eq.update_layout(title="OOS Equity je Fold (normiert auf 100 = Startkapital)",
                                     height=350, template="plotly_dark", margin=dict(t=40,b=20))
                st.plotly_chart(fig_eq, use_container_width=True)

        # ── CSV Download ──────────────────────────────────────────────────────
        st.download_button("⬇️ WFA-Ergebnis als CSV",
                           data=wfa_df.to_csv(index=False).encode(),
                           file_name="btc_weekday_wfa.csv", mime="text/csv")

        # ════════════════════════════════════════════════════════════════════════
        # PARAMETER-STABILITÄTSANALYSE
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("Parameter-Stabilitätsanalyse")
        st.caption("Welche SL / Trailing-Kombinationen liefern konsistent über ALLE Folds gute OOS-Ergebnisse?")

        if param_stability:
            stab_rows = []
            for (sl_, tt_, to_, ma_, fm_), results in param_stability.items():
                if len(results) < 2:
                    continue
                pfs      = [r["pf"]         for r in results if r["n"] > 0]
                rets     = [r["total_ret"]   for r in results if r["n"] > 0]
                sharpes  = [r["sharpe"]      for r in results if r["n"] > 0]
                trades   = [r["n"]           for r in results]
                n_pos    = sum(1 for r in rets if r > 0) if rets else 0
                n_folds  = len(results)

                if not pfs:
                    continue

                stab_rows.append({
                    "SL %":             f"{sl_:.2f}%",
                    "Trail-Trig %":     f"{tt_:.2f}%",
                    "Trail-Off %":      f"{to_:.2f}%",
                    "MA":               f"{ma_}",
                    "Filter":           fm_,
                    "Folds getestet":   n_folds,
                    "Profitable Folds": n_pos,
                    "Konsistenz %":     round(n_pos / n_folds * 100, 0),
                    "Ø OOS PF":         round(np.mean(pfs), 2),
                    "Ø OOS Ret %":      f"{round(np.mean(rets), 2):.2f}%",
                    "Ø OOS Sharpe":     round(np.mean(sharpes), 2),
                    "Min OOS Ret %":    f"{round(np.min(rets), 2):.2f}%",
                    "Max DD Ø":         f"{round(np.mean([r['max_dd'] for r in results]), 2):.2f}%",
                })

            if stab_rows:
                df_stab = pd.DataFrame(stab_rows)
                # Sortierung: erst Konsistenz, dann Ø PF
                df_stab = df_stab.sort_values(
                    ["Konsistenz %", "Ø OOS PF", "Ø OOS Ret %"],
                    ascending=[False, False, False]
                ).reset_index(drop=True)

                # Top 3 hervorheben
                st.markdown("#### 🏆 Top-10 stabilste Setups")
                top10 = df_stab.head(10).copy()

                def _hl_konsistenz(v):
                    if isinstance(v, (int, float)):
                        if v >= 80: return "color:#22c55e;font-weight:700"
                        if v >= 60: return "color:#f0c040"
                        return "color:#ef5350"
                    return ""

                def _hl_num(v):
                    if isinstance(v, (int, float)):
                        return "color:#22c55e" if v > 0 else "color:#ef5350"
                    return ""

                styled_stab = top10.style\
                    .map(_hl_konsistenz, subset=["Konsistenz %"])\
                    .map(_hl_num, subset=["Ø OOS PF"])
                st.dataframe(styled_stab, use_container_width=True, hide_index=True)

                # Heatmap: SL% vs Trail-Trig% → Ø Konsistenz
                st.markdown("#### Heatmap: SL% × Trail-Trigger% → Konsistenz %")
                pivot = df_stab.pivot_table(
                    values="Konsistenz %",
                    index="SL %",
                    columns="Trail-Trig %",
                    aggfunc="mean"
                ).round(0)

                fig_heat = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[f"Trail {c}%" for c in pivot.columns],
                    y=[f"SL {r}%" for r in pivot.index],
                    colorscale="RdYlGn",
                    zmin=0, zmax=100,
                    text=pivot.values.round(0).astype(str),
                    texttemplate="%{text}%",
                    showscale=True,
                    colorbar=dict(title="Konsistenz %")
                ))
                fig_heat.update_layout(
                    title="Ø Konsistenz je SL% / Trail-Trigger% Kombination",
                    height=350, template="plotly_dark",
                    margin=dict(t=50, b=30, l=80, r=20)
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # Bestes Setup hervorheben
                best = df_stab.iloc[0]
                st.success(
                    f"**Stabilstes Setup:** SL {best['SL %']} · "
                    f"Trail-Trigger {best['Trail-Trig %']} · Trail-Abstand {best['Trail-Off %']} · "
                    f"MA {best['MA']} · Filter: {best['Filter']} → "
                    f"**{int(best['Konsistenz %'])}% Konsistenz** · "
                    f"Ø OOS PF {best['Ø OOS PF']} · Ø OOS Ret {best['Ø OOS Ret %']}"
                )

                # Download
                st.download_button("⬇️ Stabilitätsanalyse als CSV",
                                   data=df_stab.to_csv(index=False).encode(),
                                   file_name="btc_param_stability.csv", mime="text/csv")
            else:
                st.info("Zu wenig Daten für Stabilitätsanalyse — mehr Folds oder breiteren Grid verwenden.")

        # ════════════════════════════════════════════════════════════════════════
        # ENSEMBLE WFA — N Läufe mit versetzten Startdaten
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("Ensemble WFA — Mehrfach-Lauf")
        st.markdown("""
    Führt den WFA **N Mal** durch, jedes Mal mit einem um 1 Monat versetzten Startdatum.
    So entstehen echte, unterschiedliche Folds — und du siehst welche Parameter **immer wieder** gewinnen, unabhängig vom Startpunkt.
        """)

        if "ens_running" not in st.session_state:
            st.session_state["ens_running"] = False

        # Status-Badge: Ensemble bereits gelaufen?
        _ens_status_key = f"ens_results_{_yf_ticker}"
        if _ens_status_key in st.session_state:
            _es = st.session_state[_ens_status_key]
            st.success(f"✅ Ensemble bereits gelaufen — {_es['n_runs']} Läufe · Zeitraum: {_es['tested_period']} · "
                       f"Ergebnisse werden unten angezeigt. Neu starten um zu aktualisieren.")
        else:
            st.info("⏳ Ensemble noch nicht gestartet — klicke '▶ Ensemble WFA starten'.")

        ec1, ec2 = st.columns(2)
        n_runs = ec1.number_input("Anzahl Läufe", min_value=3, max_value=10, value=5, step=1, key="ens_runs")
        if ec2.button("▶ Ensemble WFA starten", type="primary", key="ens_run_btn"):
            st.session_state["ens_running"] = True

        if st.session_state["ens_running"]:
            import datetime as _dt2
            from dateutil.relativedelta import relativedelta

            st.info(f"Starte {n_runs} WFA-Läufe mit versetzten Startdaten …")
            ens_progress = st.progress(0, text="Ensemble läuft …")

            ensemble_stability: dict = {}  # key=(sl,tt,to,ma,fm) → list of OOS metrics über ALLE Läufe

            for run_i in range(int(n_runs)):
                run_start = pd.Timestamp(btc_start) + pd.DateOffset(months=run_i)
                df_run    = df_raw[df_raw.index >= run_start].copy()

                if len(df_run) < 60:
                    continue

                # Folds für diesen Lauf
                is_d_r  = pd.DateOffset(months=int(is_months))
                oos_d_r = pd.DateOffset(months=int(oos_months))
                folds_r, fs_r = [], df_run.index[0]
                while True:
                    ie_r = fs_r + is_d_r
                    oe_r = ie_r + oos_d_r
                    if oe_r > df_run.index[-1]:
                        break
                    folds_r.append({"is_start": fs_r, "is_end": ie_r,
                                     "oos_start": ie_r, "oos_end": oe_r})
                    fs_r = fs_r + oos_d_r

                if len(folds_r) < 2:
                    continue

                adx_grid_r = g_adx if use_adx else [float(adx_thresh)]

                for fold_r in folds_r:
                    df_is_r  = df_run[(df_run.index >= fold_r["is_start"]) & (df_run.index < fold_r["is_end"])].copy()
                    df_oos_r = df_run[(df_run.index >= fold_r["oos_start"]) & (df_run.index < fold_r["oos_end"])].copy()

                    if len(df_is_r) < 30 or len(df_oos_r) < 5:
                        continue

                    for sl_, ma_, fm_, adx_, tt_, to_ in _prod(g_sl, g_ma, g_fm, adx_grid_r, g_tt, g_to):
                        p = {**base_params,
                             "sl_pct":      sl_,
                             "ma_period":   ma_,
                             "filter_mode": fm_,
                             "adx_thresh":  adx_,
                             "trail_trig":  tt_,
                             "trail_off":   to_}
                        try:
                            tr_is_r, _ = _momi_backtest_engine(df_is_r.copy(), p)
                            if len(tr_is_r) < int(min_trades):
                                continue
                            tr_oos_r, eq_oos_r = _momi_backtest_engine(df_oos_r.copy(), p)
                            m_oos_r = _momi_metrics(tr_oos_r, eq_oos_r)
                            key = (sl_, tt_, to_, ma_, fm_)
                            if key not in ensemble_stability:
                                ensemble_stability[key] = []
                            ensemble_stability[key].append(m_oos_r)
                        except Exception:
                            continue

                ens_progress.progress((run_i + 1) / int(n_runs),
                                       text=f"Lauf {run_i+1}/{int(n_runs)} abgeschlossen")

            ens_progress.progress(1.0, text=f"Ensemble abgeschlossen ✓  ({int(n_runs)} Läufe)")
            st.session_state["ens_running"] = False

            # Auswertung
            if not ensemble_stability:
                st.error("Keine Ensemble-Ergebnisse — Grid oder Zeitraum anpassen.")
            else:
                # Buy & Hold Referenz für den gesamten Zeitraum
                bh_start_price = float(df_raw["Close"].iloc[0])
                bh_end_price   = float(df_raw["Close"].iloc[-1])
                bh_total_ret   = (bh_end_price / bh_start_price - 1) * 100
                tested_period  = f"{df_raw.index[0].date()} – {df_raw.index[-1].date()}"

                ens_rows = []
                ens_raw_params = {}  # key=rank → raw param dict für Backtest
                for (sl_, tt_, to_, ma_, fm_), results in ensemble_stability.items():
                    if len(results) < 3:
                        continue
                    pfs    = [r["pf"]        for r in results if r["n"] > 0]
                    rets   = [r["total_ret"] for r in results if r["n"] > 0]
                    sharps = [r["sharpe"]    for r in results if r["n"] > 0]
                    trades_list = [r["n"]    for r in results if r["n"] > 0]
                    avg_ret_list = [r.get("avg_ret", 0) for r in results if r["n"] > 0]
                    if not pfs:
                        continue
                    n_pos = sum(1 for r in rets if r > 0)
                    n_tot = len(results)
                    ens_rows.append({
                        "_sl": sl_, "_tt": tt_, "_to": to_, "_ma": ma_, "_fm": fm_,
                        "SL %":             f"{sl_:.2f}%",
                        "Trail-Trig %":     f"{tt_:.2f}%",
                        "Trail-Off %":      f"{to_:.2f}%",
                        "MA":               f"{ma_}",
                        "Filter":           fm_,
                        "Konsistenz %":     round(n_pos / n_tot * 100, 0),
                        "Ø OOS PF":         round(np.mean(pfs), 2),
                        "Ø OOS Ret %":      round(np.mean(rets), 2),
                        "Ø Profit/Trade %": round(np.mean(avg_ret_list), 3) if avg_ret_list else 0,
                        "Ø OOS Sharpe":     round(np.mean(sharps), 2),
                        "Getestete Folds":  n_tot,
                        "Zeitraum":         tested_period,
                    })

                if ens_rows:
                    df_ens = pd.DataFrame(ens_rows).sort_values(
                        ["Konsistenz %", "Ø OOS PF"], ascending=[False, False]
                    ).reset_index(drop=True)

                    # Ergebnisse persistent im Session-State speichern
                    _ens_cache_key = f"ens_results_{_yf_ticker}"
                    display_cols_ens = ["SL %","Trail-Trig %","Trail-Off %","MA","Filter",
                                        "Konsistenz %","Ø OOS PF","Ø OOS Ret %","Ø Profit/Trade %","Ø OOS Sharpe","Zeitraum"]
                    st.session_state[_ens_cache_key] = {
                        "df_ens":       df_ens,
                        "bh_total_ret": bh_total_ret,
                        "tested_period": tested_period,
                        "n_runs":       int(n_runs),
                        "display_cols": display_cols_ens,
                    }

                    st.success(f"**Ensemble abgeschlossen** — {len(df_ens)} Kombinationen über {int(n_runs)} Läufe · Zeitraum: {tested_period}")

                    # ── Top-10 Tabelle ────────────────────────────────────────────
                    st.markdown(f"### 🏆 Top-10 stabilste Setups über {int(n_runs)} Läufe")
                    st.caption(f"Zeitraum: **{tested_period}** · Buy & Hold BTC in diesem Zeitraum: **{bh_total_ret:+.1f}%**")

                    top10_ens = df_ens.head(10)[display_cols_ens].copy()
                    top10_ens["Ø OOS Ret %"]      = top10_ens["Ø OOS Ret %"].apply(lambda v: f"{v:.2f}%")
                    top10_ens["Ø Profit/Trade %"] = top10_ens["Ø Profit/Trade %"].apply(lambda v: f"{v:.3f}%")

                    def _hl_k(v):
                        if not isinstance(v, (int, float)): return ""
                        if v >= 80: return "color:#22c55e;font-weight:700"
                        if v >= 60: return "color:#f0c040"
                        return "color:#ef5350"
                    st.dataframe(top10_ens.style.map(_hl_k, subset=["Konsistenz %"]),
                                 use_container_width=True, hide_index=True)

                    best_ens = df_ens.iloc[0]
                    st.success(
                        f"**Robustestes Setup:** SL {best_ens['SL %']} · "
                        f"Trail-Trigger {best_ens['Trail-Trig %']} · Trail-Abstand {best_ens['Trail-Off %']} · "
                        f"MA {best_ens['MA']} · Filter: {best_ens['Filter']} → "
                        f"**{int(best_ens['Konsistenz %'])}% Konsistenz** · "
                        f"Ø OOS PF {best_ens['Ø OOS PF']} · Ø OOS Ret {best_ens['Ø OOS Ret %']:.2f}%"
                    )

                    # ── Equity Kurve vs Buy & Hold für Top-10 ────────────────────
                    st.markdown("### Equity Kurve vs Buy & Hold — Top-10 Setups")
                    st.caption("Strategie (orange) vs reines BTC halten (blau) über den gesamten Testzeitraum")

                    bh_equity = df_raw["Close"] / df_raw["Close"].iloc[0] * 10_000

                    for rank, row in df_ens.head(10).iterrows():
                        p_full = {**base_params,
                                  "sl_pct":      row["_sl"],
                                  "ma_period":   int(row["_ma"]),
                                  "filter_mode": row["_fm"],
                                  "trail_trig":  row["_tt"],
                                  "trail_off":   row["_to"]}
                        try:
                            tr_f, eq_f = _momi_backtest_engine(df_raw.copy(), p_full)
                            m_f = _momi_metrics(tr_f, eq_f)
                        except Exception:
                            continue

                        total_ret_f = m_f["total_ret"]
                        label = (f"#{rank+1} · SL {row['SL %']} · Trail {row['Trail-Trig %']} · "
                                 f"MA {row['MA']} · {row['Filter']}")

                        with st.expander(f"#{rank+1} — Rendite: {total_ret_f:+.1f}% vs Buy&Hold: {bh_total_ret:+.1f}% · {label}"):
                            # KPI-Zeile
                            k1,k2,k3,k4,k5 = st.columns(5)
                            k1.metric("Gesamt-Rendite",  f"{total_ret_f:+.1f}%")
                            k2.metric("Buy & Hold",      f"{bh_total_ret:+.1f}%")
                            k3.metric("Outperformance",  f"{total_ret_f - bh_total_ret:+.1f}%")
                            k4.metric("Trades",          m_f["n"])
                            k5.metric("Profit Factor",   f"{m_f['pf']:.2f}")

                            k6,k7,k8 = st.columns(3)
                            k6.metric("Win-Rate",        f"{m_f['wr']:.1f}%")
                            k7.metric("Max Drawdown",    f"{m_f['max_dd']:.1f}%")
                            k8.metric("Sharpe",          f"{m_f['sharpe']:.2f}")

                            # Chart
                            fig_vs = go.Figure()
                            fig_vs.add_trace(go.Scatter(
                                x=bh_equity.index, y=bh_equity.values,
                                name="Buy & Hold BTC",
                                line=dict(color="#4a9eff", width=2),
                                fill="tozeroy", fillcolor="rgba(74,158,255,0.05)"))
                            fig_vs.add_trace(go.Scatter(
                                x=eq_f.index, y=eq_f.values,
                                name="Strategie",
                                line=dict(color="#f7931a", width=2.5),
                                fill="tozeroy", fillcolor="rgba(247,147,26,0.1)"))
                            fig_vs.add_hline(y=10_000, line_color="white",
                                             line_dash="dash", line_width=1)
                            fig_vs.update_layout(
                                title=f"Setup #{rank+1}: {label}",
                                height=350, template="plotly_dark",
                                yaxis_title="Kapital (€)",
                                legend=dict(orientation="h", y=1.05),
                                margin=dict(t=50, b=20, l=60, r=20))
                            st.plotly_chart(fig_vs, use_container_width=True)

                    # ── Heatmap ───────────────────────────────────────────────────
                    st.markdown("### Heatmap: SL% × Trail-Trigger% → Konsistenz %")
                    pivot_ens = df_ens.copy()
                    pivot_ens["SL_num"] = pivot_ens["SL %"].str.replace("%","").astype(float)
                    pivot_ens["TT_num"] = pivot_ens["Trail-Trig %"].str.replace("%","").astype(float)
                    heat_ens = pivot_ens.pivot_table(
                        values="Konsistenz %", index="SL_num", columns="TT_num", aggfunc="mean").round(0)
                    fig_heat_ens = go.Figure(go.Heatmap(
                        z=heat_ens.values,
                        x=[f"Trail {c}%" for c in heat_ens.columns],
                        y=[f"SL {r}%"    for r in heat_ens.index],
                        colorscale="RdYlGn", zmin=0, zmax=100,
                        text=heat_ens.values.round(0).astype(str),
                        texttemplate="%{text}%", showscale=True,
                        colorbar=dict(title="Konsistenz %")))
                    fig_heat_ens.update_layout(
                        title=f"Ensemble-Konsistenz über {int(n_runs)} Läufe",
                        height=350, template="plotly_dark",
                        margin=dict(t=50, b=30, l=80, r=20))
                    st.plotly_chart(fig_heat_ens, use_container_width=True)

                    st.download_button("⬇️ Ensemble-Ergebnis als CSV",
                                       data=df_ens[display_cols_ens].to_csv(index=False).encode(),
                                       file_name="btc_ensemble_wfa.csv", mime="text/csv")

        # ── Gespeicherte Ensemble-Ergebnisse anzeigen (auch nach Reload) ──────
        _ens_cache_key = f"ens_results_{_yf_ticker}"
        if not st.session_state.get("ens_running") and _ens_cache_key in st.session_state:
            _ec = st.session_state[_ens_cache_key]
            _df_ens_c   = _ec["df_ens"]
            _bh_ret_c   = _ec["bh_total_ret"]
            _period_c   = _ec["tested_period"]
            _n_runs_c   = _ec["n_runs"]
            _dcols_c    = _ec["display_cols"]
            st.markdown("---")
            st.markdown(f"### 🏆 Top-10 Ensemble-Setups ({_n_runs_c} Läufe) — gespeichertes Ergebnis")
            st.caption(f"Zeitraum: **{_period_c}** · Buy & Hold: **{_bh_ret_c:+.1f}%**")

            _t10 = _df_ens_c.head(10)[_dcols_c].copy()
            _t10["Ø OOS Ret %"]      = _t10["Ø OOS Ret %"].apply(lambda v: f"{v:.2f}%" if isinstance(v, (int,float)) else v)
            _t10["Ø Profit/Trade %"] = _t10["Ø Profit/Trade %"].apply(lambda v: f"{v:.3f}%" if isinstance(v, (int,float)) else v)

            def _hl_k2(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 80: return "color:#22c55e;font-weight:700"
                if v >= 60: return "color:#f0c040"
                return "color:#ef5350"
            st.dataframe(_t10.style.map(_hl_k2, subset=["Konsistenz %"]),
                         use_container_width=True, hide_index=True)

            _best_c = _df_ens_c.iloc[0]
            st.success(
                f"**Robustestes Setup:** SL {_best_c['SL %']} · "
                f"Trail-Trigger {_best_c['Trail-Trig %']} · Trail-Abstand {_best_c['Trail-Off %']} · "
                f"MA {_best_c['MA']} · Filter: {_best_c['Filter']} → "
                f"**{int(_best_c['Konsistenz %'])}% Konsistenz** · "
                f"Ø OOS PF {_best_c['Ø OOS PF']} · Ø OOS Ret {_best_c['Ø OOS Ret %']:.2f}%"
            )
            st.download_button("⬇️ Ensemble als CSV (gespeichert)",
                               data=_df_ens_c[_dcols_c].to_csv(index=False).encode(),
                               file_name="btc_ensemble_wfa.csv", mime="text/csv",
                               key="ens_dl_cached")

    # ════════════════════════════════════════════════════════════════════════
    # MONTE CARLO — Prop Trading Challenge Simulator
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Monte Carlo — Prop Trading Challenge Simulator")
    st.markdown("""
Simuliert **1.000 mögliche Zukunften** deiner Strategie durch zufälliges Mischen (Bootstrap) der echten Trade-Ergebnisse.
Zeigt dir wie wahrscheinlich es ist, eine Prop-Firm Challenge zu bestehen — bevor du echtes Geld riskierst.
    """)

    # Trades aus Full-Sample Backtest holen — wird immer berechnet (oben)
    mc_trades_raw = st.session_state.get(f"mc_trades_{_yf_ticker}")
    if mc_trades_raw is None or mc_trades_raw.empty:
        mc_trades_raw = st.session_state.get(wfa_cache_key, {}).get("full_trades")
    # Letzter Fallback: tr_full liegt im aktuellen Scope
    if (mc_trades_raw is None or mc_trades_raw.empty) and not tr_full.empty:
        mc_trades_raw = tr_full

    if mc_trades_raw is None or mc_trades_raw.empty:
        st.info("Kein Backtest-Ergebnis — Seite neu laden und Parameter prüfen.")
    else:
        n_real = len(mc_trades_raw)
        real_rets = mc_trades_raw["PnL $"].values
        st.success(f"**{n_real} echte Trades** aus dem Full-Sample Backtest geladen · "
                   f"Win-Rate: {(real_rets > 0).mean()*100:.1f}% · "
                   f"Ø Trade: {real_rets.mean():.2f}$ · "
                   f"Trades/Jahr: ~{n_real / max(1, (df_raw.index[-1]-df_raw.index[0]).days / 365):.0f}")

        if n_real < 20:
            st.warning(f"⚠️ Nur {n_real} Trades — Monte Carlo Ergebnisse haben hohe Unsicherheit. "
                       f"Mind. 50 Trades empfohlen für verlässliche Aussagen.")

        st.info("ℹ️ Kein Zeitlimit — die Simulation läuft bis das Profit-Ziel erreicht **oder** "
                "die Drawdown-Grenze verletzt wird. Genau wie moderne Prop Firms.")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Challenge-Regeln (Prop Firm)**")
            challenge_capital    = st.number_input("Startkapital ($)", 10_000, 200_000, 100_000, step=10_000, key="mc_cap")
            profit_target_p1_pct = st.number_input("Phase 1 Profit-Ziel (%)", 1.0, 20.0, 8.0, step=0.5, key="mc_pt_p1",
                                                    help="FTMO Phase 1: 8% — Evaluierung")
            profit_target_p2_pct = st.number_input("Phase 2 Profit-Ziel (%)", 1.0, 20.0, 5.0, step=0.5, key="mc_pt_p2",
                                                    help="FTMO Phase 2: 5% — Verification (gleiche DD-Regeln)")
            max_total_dd_pct     = st.number_input("Max. Total Drawdown (%)", 1.0, 20.0, 10.0, step=0.5, key="mc_tdd",
                                                    help="FTMO: 10% — gilt für BEIDE Phasen")
            max_daily_dd_pct     = st.number_input("Max. Daily Drawdown (%)", 0.5, 10.0, 5.0, step=0.5, key="mc_ddd",
                                                    help="FTMO: 5% — gilt für BEIDE Phasen")
        with mc2:
            st.markdown("**Simulation**")
            n_sims         = st.number_input("Anzahl Simulationen", 500, 5000, 1000, step=500, key="mc_sims")
            max_trades_sim = st.number_input("Max. Trades pro Simulation", 10, 500, 100, step=10, key="mc_maxt",
                                              help="Sicherheitsnetz: nach X Trades ohne Ergebnis gilt die Sim als 'nicht bestanden'")
            risk_per_trade = st.number_input("Risiko pro Trade (%)", 0.1, 5.0, 1.0, step=0.1, key="mc_risk")

        run_mc = st.button("▶ Monte Carlo starten", type="primary", key="mc_run_btn")
        if run_mc:
            st.session_state["mc_running"] = True
        if st.session_state.get("mc_running"):
            st.session_state["mc_running"] = False

            mc_progress = st.progress(0, text="Monte Carlo läuft …")

            # Trade-Returns normieren
            win_mask  = real_rets > 0
            avg_win   = real_rets[win_mask].mean()  if win_mask.any()   else 1
            avg_loss  = real_rets[~win_mask].mean() if (~win_mask).any() else -1
            norm_rets = np.where(real_rets > 0,
                                  real_rets / avg_win,
                                 -real_rets / avg_loss)

            n_sims_int = int(n_sims)
            results, all_paths_p1 = [], []

            def _run_phase(cap, target_pct, norm_r, risk_pct, dd_total, dd_daily, max_t):
                """Simuliert eine Phase; gibt (passed, fail_reason, n_trades, final_pct, path) zurück."""
                capital   = float(cap)
                peak      = capital
                day_start = capital
                path      = [capital]
                n_trades  = 0
                failed    = False
                reason    = ""
                while True:
                    nr      = norm_r[np.random.randint(len(norm_r))]
                    pnl     = capital * (risk_pct / 100) * nr
                    capital += pnl
                    peak    = max(peak, capital)
                    path.append(capital)
                    n_trades += 1
                    daily_dd = (capital - day_start) / day_start * 100
                    if daily_dd < -dd_daily:
                        failed, reason = True, "Daily DD"; break
                    day_start = capital
                    total_dd = (capital - peak) / peak * 100
                    if total_dd < -dd_total:
                        failed, reason = True, "Total DD"; break
                    profit = (capital - cap) / cap * 100
                    if profit >= target_pct:
                        break
                    if n_trades >= max_t:
                        break
                final_pct = (capital - cap) / cap * 100
                passed    = not failed and final_pct >= target_pct
                return passed, reason, n_trades, final_pct, path

            for sim_i in range(n_sims_int):
                # ── Phase 1 ──
                p1_passed, p1_reason, p1_trades, p1_pct, p1_path = _run_phase(
                    challenge_capital, profit_target_p1_pct, norm_rets,
                    risk_per_trade, max_total_dd_pct, max_daily_dd_pct, int(max_trades_sim))

                # ── Phase 2 (nur wenn Phase 1 bestanden) ──
                p2_passed, p2_reason, p2_trades, p2_pct = False, "", 0, 0.0
                if p1_passed:
                    p2_passed, p2_reason, p2_trades, p2_pct, _ = _run_phase(
                        challenge_capital, profit_target_p2_pct, norm_rets,
                        risk_per_trade, max_total_dd_pct, max_daily_dd_pct, int(max_trades_sim))

                results.append({
                    "p1_passed":  p1_passed,
                    "p1_reason":  p1_reason,
                    "p1_trades":  p1_trades,
                    "p1_pct":     p1_pct,
                    "p2_passed":  p2_passed,
                    "p2_reason":  p2_reason,
                    "p2_trades":  p2_trades,
                    "p2_pct":     p2_pct,
                    "payout":     p1_passed and p2_passed,
                })
                if sim_i < 200:
                    all_paths_p1.append(p1_path)

                if sim_i % 100 == 0:
                    mc_progress.progress(sim_i / n_sims_int, text=f"Simulation {sim_i}/{n_sims_int} …")

            mc_progress.progress(1.0, text="Monte Carlo abgeschlossen ✓")

            df_res = pd.DataFrame(results)
            n_p1_pass  = df_res["p1_passed"].sum()
            n_p1_fail  = (~df_res["p1_passed"]).sum()
            n_p2_pass  = df_res["p2_passed"].sum()   # = payout
            n_p2_fail  = (df_res["p1_passed"] & ~df_res["p2_passed"]).sum()
            n_payout   = df_res["payout"].sum()

            p1_pct_val   = n_p1_pass / n_sims_int * 100
            p2_cond_pct  = n_p2_pass / n_p1_pass * 100 if n_p1_pass > 0 else 0   # P(P2|P1)
            payout_pct   = n_payout  / n_sims_int * 100                            # P(P1∩P2)

            avg_t_p1 = df_res.loc[df_res["p1_passed"], "p1_trades"].mean() if n_p1_pass > 0 else 0
            avg_t_p2 = df_res.loc[df_res["p2_passed"], "p2_trades"].mean() if n_p2_pass > 0 else 0

            # ── Payout Badge ──────────────────────────────────────────────
            if payout_pct >= 40:
                bc, bt = "#22c55e", f"✅ GUTE PAYOUT-CHANCE — {payout_pct:.1f}% der Simulationen bestehen BEIDE Phasen"
            elif payout_pct >= 15:
                bc, bt = "#f0c040", f"⚠️ MÖGLICH — {payout_pct:.1f}% der Simulationen bestehen BEIDE Phasen"
            else:
                bc, bt = "#ef5350", f"❌ SCHWIERIG — nur {payout_pct:.1f}% der Simulationen erhalten einen Payout"

            st.markdown(
                f'<div style="background:{bc}22;border:2px solid {bc};border-radius:10px;'
                f'padding:16px 24px;font-weight:800;font-size:1.3rem;color:{bc};margin:16px 0;">'
                f'{bt}</div>', unsafe_allow_html=True)

            # ── KPIs ──────────────────────────────────────────────────────
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Phase 1 besteht",    f"{p1_pct_val:.1f}%",  f"{n_p1_pass}/{n_sims_int}")
            k2.metric("Phase 2 | P1 ok",    f"{p2_cond_pct:.1f}%", f"{n_p2_pass}/{n_p1_pass if n_p1_pass else '–'}",
                      help="Wahrscheinlichkeit Phase 2 zu bestehen, wenn Phase 1 schon bestanden ist")
            k3.metric("💰 Payout",           f"{payout_pct:.1f}%",  f"{n_payout}/{n_sims_int}",
                      help="P(Phase1) × P(Phase2|Phase1) — echte Auszahlungswahrscheinlichkeit")
            k4.metric("Ø Wochen Phase 1",   f"{avg_t_p1:.0f} Wo."  if avg_t_p1 > 0 else "–",
                      help="Durchschnittliche Dauer bis Phase 1 bestanden (bei bestandenen Sims)")
            k5.metric("Ø Wochen Phase 2",   f"{avg_t_p2:.0f} Wo."  if avg_t_p2 > 0 else "–",
                      help="Durchschnittliche Dauer bis Phase 2 bestanden (bei bestandenen Sims)")
            k6.metric("Ø Gesamt",           f"{avg_t_p1+avg_t_p2:.0f} Wo." if (avg_t_p1+avg_t_p2) > 0 else "–",
                      help="Gesamtdauer bis erster Payout")

            p1_dd_fail = (df_res["p1_reason"] == "Daily DD").sum()
            p1_td_fail = (df_res["p1_reason"] == "Total DD").sum()
            p2_dd_fail = (df_res["p2_reason"] == "Daily DD").sum()
            p2_td_fail = (df_res["p2_reason"] == "Total DD").sum()
            st.caption(
                f"Phase 1 — Daily DD: {p1_dd_fail}x · Total DD: {p1_td_fail}x  |  "
                f"Phase 2 — Daily DD: {p2_dd_fail}x · Total DD: {p2_td_fail}x")

            # ── Kreisdiagramm ─────────────────────────────────────────────
            pie_labels = ["❌ Phase 1 gescheitert", "⚠️ Phase 2 gescheitert", "💰 Payout erhalten"]
            pie_values = [n_p1_fail, n_p2_fail, n_payout]
            pie_colors = ["#ef5350", "#f0c040", "#22c55e"]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker=dict(colors=pie_colors, line=dict(color="#1a1a2e", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13),
                hole=0.4,
                pull=[0, 0, 0.07],
            ))
            fig_pie.update_layout(
                title=f"Challenge-Ergebnis aus {n_sims_int} Simulationen",
                height=380, template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=60, b=60, l=20, r=20),
                annotations=[dict(
                    text=f"<b>{payout_pct:.1f}%</b><br>Payout",
                    x=0.5, y=0.5, font_size=16,
                    font_color="#22c55e", showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.info(
                f"**Warum ist Phase 2 leichter als Phase 1?**  "
                f"Niedrigeres Ziel ({profit_target_p2_pct}% statt {profit_target_p1_pct}%) bei gleichen DD-Grenzen. "
                f"Daher: P(Phase 2 | Phase 1 bestanden) = **{p2_cond_pct:.1f}%** > P(Phase 1) = **{p1_pct_val:.1f}%**. "
                f"Gesamtchance: {p1_pct_val:.1f}% × {p2_cond_pct:.1f}% = **{payout_pct:.1f}% Payout**."
            )

            # ── Phase-1-Pfad-Chart ────────────────────────────────────────
            if all_paths_p1:
                st.subheader("Phase 1 — Simulierte Verläufe")
                fig_mc = go.Figure()
                for path in all_paths_p1:
                    ok = (path[-1] - challenge_capital) / challenge_capital * 100 >= profit_target_p1_pct
                    fig_mc.add_trace(go.Scatter(
                        y=path, mode="lines",
                        line=dict(color="#22c55e" if ok else "#ef5350", width=0.5),
                        opacity=0.12, showlegend=False))
                max_len   = max(len(p) for p in all_paths_p1)
                padded    = [p + [p[-1]] * (max_len - len(p)) for p in all_paths_p1]
                mean_path = np.mean(padded, axis=0)
                fig_mc.add_trace(go.Scatter(y=mean_path, mode="lines",
                                             line=dict(color="white", width=2.5, dash="dash"), name="Ø Pfad"))
                fig_mc.add_hline(y=challenge_capital * (1 + profit_target_p1_pct/100),
                                  line_color="#22c55e", line_dash="dot", line_width=2,
                                  annotation_text=f"Phase 1 Ziel +{profit_target_p1_pct}%", annotation_position="right")
                fig_mc.add_hline(y=challenge_capital * (1 - max_total_dd_pct/100),
                                  line_color="#ef5350", line_dash="dot", line_width=2,
                                  annotation_text=f"Ruin -{max_total_dd_pct}%", annotation_position="right")
                fig_mc.add_hline(y=challenge_capital, line_color="white", line_dash="dash", line_width=1)
                fig_mc.update_layout(
                    title=f"Phase 1 — {n_sims_int} Simulationen · Grün = bestanden · Rot = gescheitert",
                    height=420, template="plotly_dark",
                    yaxis_title="Kapital ($)", xaxis_title="Trade #",
                    margin=dict(t=50, b=30, l=70, r=130))
                st.plotly_chart(fig_mc, use_container_width=True)

            # ── Trades-Histogramme ────────────────────────────────────────
            if n_p1_pass > 0:
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    fig_tw1 = go.Figure(go.Histogram(
                        x=df_res.loc[df_res["p1_passed"], "p1_trades"], nbinsx=20,
                        marker_color="#42a5f5"))
                    fig_tw1.update_layout(
                        title=f"Phase 1: Wochen bis Ziel (Ø {avg_t_p1:.0f} Wo.)",
                        height=220, template="plotly_dark",
                        xaxis_title="Trades/Wochen", yaxis_title="Anzahl",
                        margin=dict(t=40, b=30, l=50, r=10))
                    st.plotly_chart(fig_tw1, use_container_width=True)
                with col_h2:
                    if n_p2_pass > 0:
                        fig_tw2 = go.Figure(go.Histogram(
                            x=df_res.loc[df_res["p2_passed"], "p2_trades"], nbinsx=20,
                            marker_color="#22c55e"))
                        fig_tw2.update_layout(
                            title=f"Phase 2: Wochen bis Ziel (Ø {avg_t_p2:.0f} Wo.)",
                            height=220, template="plotly_dark",
                            xaxis_title="Trades/Wochen", yaxis_title="Anzahl",
                            margin=dict(t=40, b=30, l=50, r=10))
                        st.plotly_chart(fig_tw2, use_container_width=True)

            if n_real < 30:
                st.warning(f"⚠️ Nur {n_real} echte Trades — Monte Carlo Ergebnisse haben hohe Unsicherheit. "
                           f"Mind. 50 Trades für verlässliche Aussagen.")

    # ════════════════════════════════════════════════════════════════════════
    # MULTI-COIN MONTE CARLO
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Multi-Coin Monte Carlo — Portfolio Challenge Simulator")
    st.markdown("""
Kombiniert die Trade-Historien **mehrerer Coins** in einer gemeinsamen Challenge-Simulation.
Alle Coins handeln am **gleichen Wochentag** → Verluste kommen oft gleichzeitig (Korrelation bleibt erhalten).
    """)

    coin_trades_all = st.session_state.get("wfa_coin_trades", {})

    # Cache-Verwaltung
    _cc1, _cc2 = st.columns([4, 1])
    _cc1.caption(f"💾 {len(coin_trades_all)} Coin(s) gespeichert — Daten überleben Seiten-Reload automatisch")
    if _cc2.button("🗑️ Cache leeren", key="mmc_clear_cache", help="Alle gespeicherten Coin-Daten löschen"):
        st.session_state["wfa_coin_trades"] = {}
        import os, requests as _req
        try: os.remove(_WFA_COIN_CACHE_FILE)
        except Exception: pass
        # Auch Gist leeren
        _tkn = _gist_token()
        if _tkn:
            try:
                _gid = _find_gist_id(_tkn)
                if _gid:
                    _req.patch(f"https://api.github.com/gists/{_gid}",
                               headers={"Authorization": f"token {_tkn}"},
                               json={"files": {_GIST_FILENAME: {"content": "{}"}}},
                               timeout=10)
            except Exception: pass
        st.rerun()

    if len(coin_trades_all) < 2:
        st.info(f"{'1 Coin gespeichert' if len(coin_trades_all) == 1 else 'Noch keine Coins gespeichert'}. "
                f"Führe WFA für mindestens 2 Coins durch, dann erscheint hier die Multi-Coin Simulation.")
        if coin_trades_all:
            done_ticker = list(coin_trades_all.keys())[0]
            done_name   = coin_trades_all[done_ticker]["symbol_name"]
            st.success(f"✓ Bereits gespeichert: **{done_name}** ({len(coin_trades_all[done_ticker]['trades'])} Trades)")
    else:
        # ── Coin-Ranking Tabelle ─────────────────────────────────────────────
        st.markdown("### Coin-Ranking nach WFA-Qualität")
        st.caption("Nur Coins mit ✅ Empfohlen werden automatisch für die Simulation vorausgewählt.")

        ranking_rows = []
        auto_select  = []
        for tkr, cdata in coin_trades_all.items():
            tr = cdata["trades"]
            name = cdata["symbol_name"].split("—")[0].strip()
            if tr.empty:
                ranking_rows.append({"Coin": name, "Ticker": tkr, "Trades": 0,
                                     "Win-Rate": "—", "Profit Factor": "—",
                                     "Ø Ret/Trade": "—", "Empfehlung": "❌ Keine Daten"})
                continue
            rets  = tr["PnL $"].values
            n     = len(rets)
            wr    = (rets > 0).mean() * 100
            gp    = rets[rets > 0].sum()
            gl    = abs(rets[rets <= 0].sum())
            pf    = gp / gl if gl > 0 else float("inf")
            avg_r = rets.mean()
            # Qualitätsschwellen — PF ≥ 1.2 reicht, WR-Minimum 35%
            # (asymmetrische Strategien mit großen Wins können <45% WR haben)
            qualified = pf >= 1.2 and wr >= 35 and n >= 20
            if qualified:
                auto_select.append(tkr)
                rec = "✅ Empfohlen"
            else:
                reasons = []
                if wr < 35:  reasons.append(f"Win-Rate {wr:.0f}%<35%")
                if pf < 1.2: reasons.append(f"PF {pf:.2f}<1.2")
                if n < 20:   reasons.append(f"Nur {n} Trades")
                rec = "❌ " + " · ".join(reasons)
            ranking_rows.append({
                "Coin": name, "Ticker": tkr, "Trades": n,
                "Win-Rate": f"{wr:.1f}%",
                "Profit Factor": f"{pf:.2f}",
                "Ø Ret/Trade ($)": f"{avg_r:.2f}",
                "Empfehlung": rec,
            })

        df_rank = pd.DataFrame(ranking_rows).sort_values(
            "Empfehlung", key=lambda s: s.str.startswith("✅"), ascending=False
        ).reset_index(drop=True)

        def _color_rec(val):
            if str(val).startswith("✅"): return "background-color:#22c55e22;color:#22c55e"
            return "background-color:#ef535022;color:#ef5350"

        st.dataframe(
            df_rank.drop(columns=["Ticker"]).style.map(_color_rec, subset=["Empfehlung"]),
            use_container_width=True, hide_index=True
        )

        n_qual = len(auto_select)
        if n_qual == 0:
            st.error("Kein Coin hat die Qualitätsschwellen erreicht (Win-Rate ≥35% + PF ≥1.2). "
                     "Alle Assets per Hand auswählen oder Grids anpassen.")
        elif n_qual == 1:
            st.warning(f"Nur 1 Coin qualifiziert — mindestens 2 für Multi-Coin MC nötig. "
                       f"Du kannst trotzdem weitere manuell hinzufügen.")
        else:
            st.success(f"**{n_qual} Coins empfohlen** für den Multi-Coin MC · "
                       f"Risiko-Tipp: max. {1.0/n_qual:.1f}% pro Coin damit Daily DD sicher bleibt")

        # Coin-Auswahl für Multi-Simulation
        all_names = {tkr: cdata["symbol_name"].split("—")[0].strip()
                     for tkr, cdata in coin_trades_all.items()}
        selected_tickers = st.multiselect(
            "Coins für Simulation auswählen (Empfohlene vorausgewählt)",
            options=list(all_names.keys()),
            default=[t for t in auto_select if t in all_names] or list(all_names.keys())[:2],
            format_func=lambda t: f"{'✅' if t in auto_select else '⚠️'} {all_names[t]}",
            key="mc_multi_tickers"
        )

        if len(selected_tickers) < 2:
            st.warning("Bitte mindestens 2 Coins auswählen.")
        else:
            # Parameter
            mmc1, mmc2 = st.columns(2)
            with mmc1:
                st.markdown("**Challenge-Regeln**")
                mmc_cap    = st.number_input("Startkapital ($)", 10_000, 200_000, 100_000, step=10_000, key="mmc_cap")
                mmc_pt_p1  = st.number_input("Phase 1 Profit-Ziel (%)", 1.0, 20.0, 8.0, step=0.5, key="mmc_pt_p1")
                mmc_pt_p2  = st.number_input("Phase 2 Profit-Ziel (%)", 1.0, 20.0, 5.0, step=0.5, key="mmc_pt_p2")
                mmc_tdd    = st.number_input("Max. Total Drawdown (%)", 1.0, 20.0, 10.0, step=0.5, key="mmc_tdd")
                mmc_ddd    = st.number_input("Max. Daily Drawdown (%)", 0.5, 10.0, 5.0, step=0.5, key="mmc_ddd")
            with mmc2:
                st.markdown("**Simulation**")
                mmc_sims   = st.number_input("Anzahl Simulationen", 500, 5000, 1000, step=500, key="mmc_sims")
                mmc_maxt   = st.number_input("Max. Trades (Sicherheitsnetz)", 10, 1000, 300, step=50, key="mmc_maxt")
                mmc_risk   = st.number_input("Risiko pro Trade pro Coin (%)", 0.1, 3.0, 0.5, step=0.1, key="mmc_risk",
                                              help="z.B. 0.5% pro Coin × 4 Coins = 2% gesamt pro Woche")

            run_mmc = st.button("▶ Multi-Coin Monte Carlo starten", type="primary", key="mmc_run_btn")
            if run_mmc:
                st.session_state["mmc_running"] = True
            if st.session_state.get("mmc_running"):
                st.session_state["mmc_running"] = False

                mmc_prog = st.progress(0, text="Multi-Coin Monte Carlo läuft …")

                # Trades pro Coin normieren
                coin_norm = {}
                for tkr in selected_tickers:
                    tr = coin_trades_all[tkr]["trades"]
                    if tr.empty:
                        continue
                    rets = tr["PnL $"].values
                    wm = rets[rets > 0].mean() if (rets > 0).any() else 1
                    lm = rets[rets <= 0].mean() if (rets <= 0).any() else -1
                    coin_norm[tkr] = np.where(rets > 0, rets / wm, -rets / lm)

                active_tickers = list(coin_norm.keys())
                n_coins = len(active_tickers)
                n_sims_mmc = int(mmc_sims)
                results_mmc, paths_p1 = [], []

                def _mmc_phase(cap, target_pct, c_norm, tickers, risk_pct, tdd, ddd, max_w):
                    capital = float(cap); peak = capital; day_start = capital
                    path = [capital]; n_weeks = 0; failed = False
                    while True:
                        week_pnl = sum(capital * (risk_pct/100) * c_norm[t][np.random.randint(len(c_norm[t]))]
                                       for t in tickers)
                        capital += week_pnl; n_weeks += 1; path.append(capital)
                        peak = max(peak, capital)
                        profit = (capital - cap) / cap * 100
                        total_dd = (capital - cap) / cap * 100
                        daily_dd = (capital - day_start) / day_start * 100 if day_start > 0 else 0
                        day_start = capital
                        if profit >= target_pct: break
                        if total_dd <= -tdd or daily_dd <= -ddd or n_weeks >= max_w:
                            failed = True; break
                    final = (capital - cap) / cap * 100
                    return not failed and final >= target_pct, n_weeks, final, path

                for sim_i in range(n_sims_mmc):
                    p1_ok, p1_w, p1_pct, p1_path = _mmc_phase(
                        mmc_cap, mmc_pt_p1, coin_norm, active_tickers,
                        mmc_risk, mmc_tdd, mmc_ddd, int(mmc_maxt))
                    p2_ok, p2_w, p2_pct = False, 0, 0.0
                    if p1_ok:
                        p2_ok, p2_w, p2_pct, _ = _mmc_phase(
                            mmc_cap, mmc_pt_p2, coin_norm, active_tickers,
                            mmc_risk, mmc_tdd, mmc_ddd, int(mmc_maxt))
                    results_mmc.append({
                        "p1_ok": p1_ok, "p1_weeks": p1_w, "p1_pct": p1_pct,
                        "p2_ok": p2_ok, "p2_weeks": p2_w,
                        "payout": p1_ok and p2_ok,
                    })
                    if sim_i < 200:
                        paths_p1.append(p1_path)
                    mmc_prog.progress((sim_i + 1) / n_sims_mmc)

                mmc_prog.progress(1.0, text="Multi-Coin Monte Carlo abgeschlossen ✓")
                df_mmc = pd.DataFrame(results_mmc)

                n_p1      = df_mmc["p1_ok"].sum()
                n_p2      = df_mmc["p2_ok"].sum()
                n_payout  = df_mmc["payout"].sum()
                n_p1_fail = n_sims_mmc - n_p1
                n_p2_fail = n_p1 - n_p2

                p1_pct_v   = n_p1     / n_sims_mmc * 100
                p2_cond_v  = n_p2     / n_p1       * 100 if n_p1 > 0 else 0
                payout_v   = n_payout / n_sims_mmc * 100
                avg_w_p1   = df_mmc.loc[df_mmc["p1_ok"],  "p1_weeks"].mean() if n_p1 > 0 else 0
                avg_w_p2   = df_mmc.loc[df_mmc["p2_ok"],  "p2_weeks"].mean() if n_p2 > 0 else 0

                # ── Payout-Badge ──────────────────────────────────────────
                color_mmc = "#22c55e" if payout_v >= 30 else "#f59e0b" if payout_v >= 10 else "#ef5350"
                label_mmc = "SEHR GUT" if payout_v >= 30 else "MACHBAR" if payout_v >= 10 else "SCHWIERIG"
                coin_labels = " + ".join([all_names[t] for t in active_tickers])
                st.markdown(f"""
<div style="background:{color_mmc}22;border:2px solid {color_mmc};border-radius:10px;padding:14px 22px;margin:12px 0">
<span style="color:{color_mmc};font-size:1.2em;font-weight:800">{label_mmc} — {payout_v:.1f}% Payout-Wahrscheinlichkeit</span>
<br><small style="color:#ccc">{n_coins} Coins gleichzeitig · {mmc_risk}% Risiko/Coin/Woche · Korrelation berücksichtigt</small>
</div>""", unsafe_allow_html=True)

                # ── KPIs ──────────────────────────────────────────────────
                mm1,mm2,mm3,mm4,mm5,mm6 = st.columns(6)
                mm1.metric("Phase 1 besteht",  f"{p1_pct_v:.1f}%",  f"{n_p1}/{n_sims_mmc}")
                mm2.metric("Phase 2 | P1 ok",  f"{p2_cond_v:.1f}%", f"{n_p2}/{n_p1 if n_p1 else '–'}")
                mm3.metric("💰 Payout",          f"{payout_v:.1f}%",  f"{n_payout}/{n_sims_mmc}")
                mm4.metric("Ø Wochen Phase 1",  f"{avg_w_p1:.0f} Wo." if avg_w_p1 > 0 else "–")
                mm5.metric("Ø Wochen Phase 2",  f"{avg_w_p2:.0f} Wo." if avg_w_p2 > 0 else "–")
                mm6.metric("Ø Gesamt",           f"{avg_w_p1+avg_w_p2:.0f} Wo." if (avg_w_p1+avg_w_p2) > 0 else "–")

                # ── Kreisdiagramm ─────────────────────────────────────────
                fig_pie_m = go.Figure(go.Pie(
                    labels=["❌ Phase 1 gescheitert", "⚠️ Phase 2 gescheitert", "💰 Payout erhalten"],
                    values=[n_p1_fail, n_p2_fail, n_payout],
                    marker=dict(colors=["#ef5350","#f0c040","#22c55e"],
                                line=dict(color="#1a1a2e", width=2)),
                    textinfo="label+percent", textfont=dict(size=12),
                    hole=0.4, pull=[0, 0, 0.07],
                ))
                fig_pie_m.update_layout(
                    title=f"Multi-Coin Challenge ({coin_labels}) — {n_sims_mmc} Simulationen",
                    height=360, template="plotly_dark",
                    legend=dict(orientation="h", y=-0.15),
                    margin=dict(t=60, b=60, l=20, r=20),
                    annotations=[dict(text=f"<b>{payout_v:.1f}%</b><br>Payout",
                                      x=0.5, y=0.5, font_size=16,
                                      font_color="#22c55e", showarrow=False)])
                st.plotly_chart(fig_pie_m, use_container_width=True)

                st.info(
                    f"Phase 2 leichter als Phase 1 (Ziel {mmc_pt_p2}% statt {mmc_pt_p1}%) → "
                    f"P(Phase 2 | Phase 1 ok) = **{p2_cond_v:.1f}%**. "
                    f"Gesamtchance: {p1_pct_v:.1f}% × {p2_cond_v:.1f}% = **{payout_v:.1f}% Payout**.")

                # ── Phase-1-Pfade ─────────────────────────────────────────
                if paths_p1:
                    fig_mmc = go.Figure()
                    for pi, path in enumerate(paths_p1):
                        ok_i = results_mmc[pi]["p1_ok"]
                        fig_mmc.add_trace(go.Scatter(
                            y=path, mode="lines",
                            line=dict(color="#22c55e" if ok_i else "#ef5350", width=0.5),
                            opacity=0.25, showlegend=False))
                    fig_mmc.add_hline(y=mmc_cap * (1 + mmc_pt_p1/100),
                                       line_color="#22c55e", line_dash="dot",
                                       annotation_text=f"Phase 1 Ziel +{mmc_pt_p1}%")
                    fig_mmc.add_hline(y=mmc_cap * (1 - mmc_tdd/100),
                                       line_color="#ef5350", line_dash="dot",
                                       annotation_text=f"DD-Limit -{mmc_tdd}%")
                    fig_mmc.update_layout(
                        title=f"Phase 1 — Multi-Coin MC · {coin_labels}",
                        height=400, template="plotly_dark",
                        yaxis_title="Kapital ($)", xaxis_title="Wochen",
                        margin=dict(t=50, b=30, l=70, r=130))
                    st.plotly_chart(fig_mmc, use_container_width=True)


def render_dax_ema_wfa() -> None:
    """DAX EMA Strategie — Montag Entry / Mittwoch Exit auf GER40 Daily (Pepperstone MT5 CSV, ab 2008-03-26)."""
    import datetime as _dt
    from itertools import product as _prod

    _dax_ticker = "GER40"
    selected_name = "DAX — Germany 40 (GER40)"

    if not st.session_state.get("dax_wfa_ran") and f"dax_mc_trades_{_dax_ticker}" in st.session_state:
        st.session_state["dax_wfa_ran"] = True

    st.header("DAX EMA Strategie — Walk-Forward Analyse")

    with st.expander("ℹ️ Was wird hier getestet und wie funktioniert es?", expanded=False):
        st.markdown("""
**Strategie:** Long-only Wochentag-Momentum auf DAX/GER40 (Pepperstone-MT5-Daily-Daten, dieselbe Quelle wie im TradingView-Chart)
- **Entry:** Jeden Montag-Schlusskurs, wenn Kurs über dem EMA liegt (MA-Filter, optional) und der Trend stark genug ist (ADX-Filter, optional)
- **Exit:** Mittwoch-Schlusskurs (zeitbasiert) — oder früher durch Stop-Loss / Trailing-Stop
- **Vorlage:** basiert auf deiner Pine-Script-Strategie "WD-MA" (WeekdayMA Long Strategy), auf TradingView **auf dem Tageschart** getestet

---

**Datenbasis:**

Läuft auf `data/mt5/GER40.csv` — echte Pepperstone-MT5-Tagesdaten (identische Quelle wie im Seasonality Lab), verfügbar ab **2008-03-26**
(davor liefert die Historie nur Wochenbars, die deshalb ausgeschlossen werden). Da dein TradingView-Test ebenfalls auf dem Tageschart lief,
ist das hier **keine Approximation**, sondern dieselbe Bar-Auflösung und Datenquelle — Entry = Montag-Schlusskurs, Exit = Mittwoch-Schlusskurs.

---

**Handelskosten:**

Spread und Kommission werden in JEDEM Backtest berücksichtigt (Full-Sample, WFA, Ensemble, Monte-Carlo-Trades) — nicht nur zur Anzeige.
Der Spread wird hälftig auf Entry (Kauf zum Ask) und Exit (Verkauf zum Bid) angerechnet, die Kommission als % der Positionsgröße pro Round-Turn.
Defaults (Pepperstone GER40): **1,5 Punkte Spread**, **0% Kommission** (Pepperstone erhebt auf Index-CFDs standardmäßig keine Kommission — die Kosten stecken im Spread). Beides ist links einstellbar; 0/0 rechnet wie zuvor ohne Kosten.

---

**Walk-Forward Analyse (WFA) — wie es funktioniert:**

Der gesamte Zeitraum wird in mehrere **Folds** aufgeteilt. Jeder Fold besteht aus zwei Phasen:

```
Fold 1: [IS-Fenster optimieren] → [OOS-Fenster blind testen]
Fold 2: [IS-Fenster optimieren] → [OOS-Fenster blind testen]
... usw.
```

- **In-Sample (IS):** Das System testet hunderte Parameter-Kombinationen und findet die beste für diesen Zeitraum
- **Out-of-Sample (OOS):** Diese beste Kombination wird auf dem **nächsten, unbekannten** Zeitraum getestet — ohne Anpassung
- ✅ **Stabil über viele Folds → ROBUST** — echte Edge, kein Zufall
- ❌ **Instabil → OVERFITTED** — funktioniert nur auf den Trainingsdaten
        """)

    st.caption(f"Strategie: Montag-Entry / Mittwoch-Exit auf {_dax_ticker} (DAX/GER40) · Pepperstone-MT5-Daily-Daten · Rollierender IS/OOS-Test")

    st.markdown(
        '<div style="background:#1e293b55;border:1px solid rgba(148,163,184,.25);border-radius:8px;'
        'padding:10px 16px;font-size:.85rem;color:#94a3b8;margin:4px 0 16px 0;">'
        '📌 <b>Referenz-Ergebnis (WFA vom 2026-07-02, enger Standard-Suchraum SL 1.0–2.0% / MA 100–200 / Trail 0.1–0.2%):</b><br>'
        'Robustestes Setup: SL 2.00% · Trail-Trigger 0.10% · Trail-Abstand 0.10% · MA 100 · Filter: Kein Filter (nur Zeit) '
        '→ 100% Konsistenz · Ø OOS PF 4.63 · Ø OOS Ret 21.62% — mit Spread 1,5 Pkt / 0% Kommission gerechnet.'
        '</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Strategie-Parameter (vorausgefüllt mit TradingView-Bestresultat)
    # ════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("---")
        st.subheader("DAX WFA: Parameter")
        dax_start = st.date_input("Daten ab", _dt.date(2008, 3, 26), min_value=_dt.date(2008, 3, 26), key="dax_start",
                                  help="Pepperstone-MT5-Historie liefert vor diesem Datum nur Wochenbars, keine Tagesbars.")
        dax_end   = st.date_input("Daten bis", _dt.date.today(), key="dax_end")

        st.markdown("---")
        if st.button("🧪 Cloud-Speicher testen (paar Sekunden, ohne WFA)", key="dax_gist_test_btn",
                     use_container_width=True,
                     help="Prüft nur, ob Speichern/Laden im GitHub-Gist funktioniert — ohne die komplette "
                          "Walk-Forward-Analyse neu zu rechnen. Für schnelles Debugging."):
            import time as _time
            _t0 = _time.time()
            _test_ok, _test_reason = _save_wfa_result("connectivity_test", {"ping": str(_dt.datetime.now())})
            _elapsed = _time.time() - _t0
            if _test_ok:
                _load_back = _load_wfa_result("connectivity_test")
                if _load_back is not None:
                    st.success(f"✅ Speichern + Laden funktioniert ({_elapsed:.1f}s)")
                else:
                    st.warning(f"⚠️ Speichern OK, aber Laden direkt danach fehlgeschlagen ({_elapsed:.1f}s)")
            else:
                st.error(f"❌ {_test_reason} ({_elapsed:.1f}s)")

    # ── Strategie-Parameter ──────────────────────────────────────────────
    st.subheader("Strategie-Parameter")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown("**Entry / Exit**")
        day_map_full = {"Montag":0,"Dienstag":1,"Mittwoch":2,"Donnerstag":3,"Freitag":4,"Samstag":5,"Sonntag":6}
        entry_day  = st.selectbox("Entry-Tag", list(day_map_full.keys()), index=0, key="dax_ed")  # Montag
        entry_hour = st.number_input("Entry-Stunde (nur bei Intraday relevant)", 0, 23, 10, key="dax_eh")
        exit_day   = st.selectbox("Exit-Tag",   list(day_map_full.keys()), index=2, key="dax_xd")  # Mittwoch
        exit_hour  = st.number_input("Exit-Stunde (nur bei Intraday relevant)",  0, 23, 22, key="dax_xh")
        st.caption("Auf Daily-Daten zählt nur der Wochentag — Stunde wird ignoriert.")
        fill_mode_label = st.selectbox(
            "Entry-Fill",
            ["Montag Close (Signal-Bar)", "Dienstag Open (nächster Bar)"],
            index=0, key="dax_fillmode",
            help="TradingView (Pine-Default ohne process_orders_on_close) füllt Orders erst zum Open "
                 "des NÄCHSTEN Bars, nicht zum Signal-Bar-Close. 'Dienstag Open' bildet das nach.")
        fill_mode = "close" if fill_mode_label.startswith("Montag") else "next_open"
    with pc2:
        st.markdown("**MA & Filter**")
        ma_type     = st.selectbox("MA-Typ",      ["EMA","SMA"], index=0,                key="dax_mt")
        ma_period   = st.selectbox("MA-Periode",  [20, 50, 100, 200], index=3,           key="dax_mp")
        filter_mode = st.selectbox("Filter-Modus",["Kein Filter (nur Zeit)","Close > MA","MA steigt"], index=0, key="dax_fm")
        use_adx     = st.checkbox("ADX-Filter", False, key="dax_adx")
        adx_thresh  = st.number_input("ADX-Schwelle", 5, 50, 20, key="dax_adxt")
    with pc3:
        st.markdown("**Risk Management**")
        use_sl     = st.checkbox("Stop-Loss", True, key="dax_sl")
        sl_pct     = st.number_input("SL %", 0.1, 20.0, 1.5, step=0.1, key="dax_slp")
        use_trail  = st.checkbox("Trailing Stop", True, key="dax_tr")
        trail_trig = st.number_input("Trail-Trigger %", 0.1, 10.0, 0.1, step=0.1, key="dax_tt")
        trail_off  = st.number_input("Trail-Abstand %", 0.1, 10.0, 0.1, step=0.1, key="dax_to")
        st.markdown("---")
        risk_pct   = st.number_input("Risiko pro Trade %", 0.1, 5.0, 1.0, step=0.1, key="dax_risk",
                                     help="1% = bei 100.000€ Konto riskierst du 1.000€ pro Trade (basierend auf SL-Abstand)")
        st.markdown("---")
        use_vol_target = st.checkbox("Volatility Targeting", False, key="dax_voltarget",
                                     help="Skaliert die Positionsgröße zusätzlich zur SL-basierten Risikogröße nach "
                                          "Moreira/Muir (2017) und Harvey et al. (2018): kleinere Position, wenn "
                                          "GER40 zuletzt volatiler war als das Ziel, größere bei ruhigerem Markt. "
                                          "Für Aktienindizes wie GER40 laut beiden Papern der Fall mit dem robustesten Effekt.")
        if use_vol_target:
            vol_target_pct = st.number_input("Ziel-Volatilität (% p.a.)", 5.0, 40.0, 15.0, step=1.0, key="dax_voltgt",
                                             help="Annualisierte Volatilität, auf die skaliert wird. GER40 lag historisch "
                                                  "meist zwischen 15–25% p.a. — niedriger Zielwert = defensiver.")
            vol_halflife = st.number_input("Half-Life (Tage)", 5, 90, 20, step=5, key="dax_volhl",
                                           help="Wie schnell die Vola-Schätzung auf neue Kursbewegungen reagiert "
                                                "(20 Tage = Standardwert aus Harvey et al. 2018).")
        else:
            vol_target_pct, vol_halflife = 15.0, 20
    with pc4:
        st.markdown("**Handelskosten**")
        spread_pts = st.number_input("Spread (Punkte, Round-Turn)", 0.0, 20.0, 1.5, step=0.1, key="dax_spread",
                                     help="Pepperstone GER40: Ø ca. 1.0–1.5 Punkte je nach Marktlage/Rollover. "
                                          "Wird hälftig auf Entry (Ask) und Exit (Bid) angerechnet.")
        commission_pct = st.number_input("Kommission (% Notional, Round-Turn)", 0.0, 1.0, 0.0, step=0.01, key="dax_comm",
                                         help="Pepperstone erhebt auf Index-CFDs standardmäßig KEINE Kommission — "
                                              "Kosten stecken komplett im Spread. Nur für Razor-artige Konten mit "
                                              "separater Kommission relevant.")
        swap_pts_per_night = st.number_input("Swap Long (Punkte pro gehaltene Nacht)", 0.0, 20.0, 0.0, step=0.1, key="dax_swap",
                                             help="Übernacht-Finanzierungskosten für gehaltene Long-Positionen — im Backtest bisher "
                                                  "komplett ignoriert, kann bei mehrtägigem Halten spürbar sein. Trag hier den echten "
                                                  "Swap-Long-Wert für GER40 aus deinem Pepperstone-Kontoauszug ein (in Punkten pro Nacht) "
                                                  "— den kenne ich nicht, der ändert sich mit den Zinsen und ist je Kontotyp verschieden. "
                                                  "Zählt EINE Nacht pro Kalendertag zwischen Entry und Exit — auch Wochenend-Nächte, falls "
                                                  "eine Position übers Wochenende läuft (0 = wie bisher, keine Swap-Kosten).")
        st.caption("0 Punkte / 0% = wie bisher ohne Handelskosten rechnen.")

    # ── WFA-Konfiguration ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Walk-Forward Konfiguration")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        is_months  = st.number_input("IS-Fenster (Monate)", 6, 36, 18, key="dax_is")
        oos_months = st.number_input("OOS-Fenster (Monate)", 3, 18, 12, key="dax_oos")
    with wc2:
        min_trades  = st.number_input("Min. IS-Trades", 5, 50, 10, key="dax_mint")
        opt_metric  = st.selectbox("Optimierungsziel", ["pf","sharpe","wr"], format_func=lambda x: {"pf":"Profit Factor","sharpe":"Sharpe","wr":"Win-Rate"}[x], key="dax_om")
    with wc3:
        min_folds   = st.number_input("Min. Folds für ✅ ROBUST", 2, 8, 4, key="dax_mf")
        st.markdown(" ")

    # ── Grid-Suchraum ─────────────────────────────────────────────────────
    with st.expander("Grid-Suchraum (IS-Optimierung)"):
        st.caption("Standard-Suchraum bewusst breiter als bei Crypto: testet auch WEITE Trailing-Abstände (bis 1.0%), "
                   "nicht nur enge — sonst gewinnt der engste Trail per Konstruktion, weil er nie zur Auswahl stand.")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            g_sl   = st.multiselect("SL %",           [0.5,1.0,1.5,1.8,2.0,2.5,3.0], default=[1.0,1.5,2.0,2.5],  key="dax_gsl")
            g_ma   = st.multiselect("MA-Periode",      [20,50,100,200],                default=[50,100,200],       key="dax_gma")
        with gc2:
            g_fm   = st.multiselect("Filter-Modus",   ["Kein Filter (nur Zeit)","Close > MA","MA steigt"],
                                     default=["Kein Filter (nur Zeit)","Close > MA","MA steigt"],   key="dax_gfm")
            g_adx  = st.multiselect("ADX-Schwelle",   [15,20,25,30],                  default=[20],               key="dax_gadx")
        with gc3:
            g_tt   = st.multiselect("Trail-Trigger %",[0.1,0.2,0.3,0.5,0.75,1.0],     default=[0.1,0.3,0.5,1.0],  key="dax_gtt")
            g_to   = st.multiselect("Trail-Abstand %",[0.1,0.2,0.3,0.4,0.75,1.0],     default=[0.1,0.3,0.4,1.0],  key="dax_gto")
        n_grid_combos = len(g_sl)*len(g_ma)*len(g_fm)*len(g_tt)*len(g_to)
        st.caption(f"Aktueller Suchraum: {n_grid_combos} Kombinationen je Fold — bei vielen Folds kann ein Lauf mehrere Minuten dauern.")

        st.markdown("---")
        _opt_days = st.checkbox("Entry/Exit-Tag mit optimieren", value=False, key="dax_opt_days",
                                help="WFA testet automatisch verschiedene Wochentag-Kombinationen — erhöht die Laufzeit deutlich")
        if _opt_days:
            _dow_opts = {"Montag":0,"Dienstag":1,"Mittwoch":2,"Donnerstag":3,"Freitag":4,"Samstag":5,"Sonntag":6}
            gd1, gd2 = st.columns(2)
            g_entry_days = gd1.multiselect("Entry-Tag testen",
                                            list(_dow_opts.keys()), default=["Montag","Dienstag"],
                                            key="dax_g_eday")
            g_exit_days  = gd2.multiselect("Exit-Tag testen",
                                            list(_dow_opts.keys()), default=["Mittwoch","Donnerstag"],
                                            key="dax_g_xday")
            g_entry_dows = [_dow_opts[d] for d in g_entry_days]
            g_exit_dows  = [_dow_opts[d] for d in g_exit_days]
            n_day_combos = len(g_entry_dows) * len(g_exit_dows)
            n_total = len(g_sl)*len(g_ma)*len(g_fm)*len(g_tt)*len(g_to)*n_day_combos
            st.caption(f"⚠️ {n_day_combos} Tag-Kombinationen × Grid = **{n_total} Kombinationen je Fold** — kann mehrere Minuten dauern")
        else:
            g_entry_dows = [day_map_full[entry_day]]
            g_exit_dows  = [day_map_full[exit_day]]

    test_both_fills = st.checkbox(
        "Beide Fill-Modi automatisch testen & speichern (Montag Close + Dienstag Open)",
        value=True, key="dax_test_both_fills",
        help="WFA und Ensemble berechnen dann bei jedem Start BEIDE Varianten und speichern sie getrennt — "
             "kein manuelles Umschalten + erneutes Starten nötig, um sie zu vergleichen. Dauert etwa doppelt so lange.")

    _btn_col1, _btn_col2 = st.columns([1, 1])
    with _btn_col1:
        run_btn = st.button("🔄 WFA starten", type="primary", use_container_width=True, key="dax_run")
    with _btn_col2:
        _ens_quick = st.button("⚡ Ensemble WFA starten (5×)", use_container_width=True, key="dax_ens_quick",
                               help="Startet direkt den 5-fachen Durchlauf — WFA muss einmal zuvor gelaufen sein",
                               disabled=not st.session_state.get("dax_wfa_ran", False))
    if run_btn:
        st.session_state["dax_wfa_ran"] = True
        st.session_state["dax_wfa_params_key"] = str(dax_start) + str(dax_end) + str(is_months) + str(oos_months)
    if _ens_quick:
        st.session_state["dax_ens_running"] = True

    # ════════════════════════════════════════════════════════════════════════
    # DATEN LADEN — Pepperstone MT5 CSV (echte GER40-Broker-Daten, dieselbe Quelle
    # wie im TradingView-Test) · läuft immer, liefert Trades für Monte Carlo
    # ════════════════════════════════════════════════════════════════════════
    cache_key = f"dax_df_{_dax_ticker}_{dax_start}_{dax_end}"
    if cache_key not in st.session_state or run_btn:
        with st.spinner("GER40 Pepperstone-MT5 Daily-Daten laden …"):
            _loaded = load_ohlc_data("GER40", source="mt5")
            if _loaded is None or _loaded.empty:
                st.session_state[cache_key] = pd.DataFrame()
            else:
                _df_mt5 = _loaded.set_index("Date")
                _df_mt5.index = pd.to_datetime(_df_mt5.index).tz_localize(None)
                # Vor 2008-03-26 liefert die CSV Wochen- statt Tagesbars (MT5-Historientiefe) —
                # ausschließen, sonst verfälscht das die Wochentag-Logik.
                _df_mt5 = _df_mt5[_df_mt5.index >= pd.Timestamp("2008-03-26")]
                _df_mt5 = _df_mt5[(_df_mt5.index >= pd.Timestamp(dax_start)) & (_df_mt5.index <= pd.Timestamp(dax_end))]
                st.session_state[cache_key] = _df_mt5
    df_raw = st.session_state[cache_key]

    if df_raw.empty:
        st.error("Keine DAX-Daten geladen — `data/mt5/GER40.csv` fehlt oder Zeitraum liegt außerhalb der Historie (ab 2008-03-26 verfügbar).")
        return

    st.success(f"✓ {len(df_raw)} Daily-Bars geladen ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    # ════════════════════════════════════════════════════════════════════════
    # FULL-SAMPLE BACKTEST (zur Orientierung)
    # ════════════════════════════════════════════════════════════════════════
    base_params = {
        "entry_dow":   day_map_full[entry_day],
        "exit_dow":    day_map_full[exit_day],
        "entry_hour":  int(entry_hour),
        "entry_min":   0,
        "exit_hour":   int(exit_hour),
        "exit_min":    0,
        "ma_type":     ma_type,
        "ma_period":   int(ma_period),
        "filter_mode": filter_mode,
        "use_adx":     use_adx,
        "adx_thresh":  float(adx_thresh),
        "sl_pct":      float(sl_pct)  if use_sl    else 999.0,
        "use_trail":   use_trail,
        "trail_trig":  float(trail_trig),
        "trail_off":   float(trail_off),
        "risk_pct":    float(risk_pct),
        "spread_pts":     float(spread_pts),
        "commission_pct": float(commission_pct),
        "swap_pts_per_night": float(swap_pts_per_night),
        "use_vol_target": use_vol_target,
        "vol_target_pct": float(vol_target_pct),
        "vol_halflife":   int(vol_halflife),
        "fill_mode":      fill_mode,
    }

    with st.spinner("Full-Sample Backtest …"):
        tr_full, eq_full = _momi_backtest_engine(df_raw.copy(), base_params)
        m_full = _momi_metrics(tr_full, eq_full)

    # Sofort speichern — MC findet Trades auch ohne WFA-Lauf
    if not tr_full.empty:
        st.session_state[f"dax_mc_trades_{_dax_ticker}"] = tr_full

    st.markdown("---")
    st.subheader("Full-Sample Ergebnis (zur Orientierung, KEIN WFA)")
    fc1,fc2,fc3,fc4,fc5 = st.columns(5)
    fc1.metric("Rendite",       f"{m_full['total_ret']:.1f}%")
    fc2.metric("Trades",         m_full['n'])
    fc3.metric("Win-Rate",      f"{m_full['wr']:.1f}%")
    fc4.metric("Profit Factor", f"{m_full['pf']:.2f}")
    fc5.metric("Sharpe",        f"{m_full['sharpe']:.2f}")

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(x=eq_full.index, y=eq_full.values,
                                  line=dict(color="#f7931a", width=2), name="Equity (Full)"))
    fig_full.update_layout(title="Full-Sample Equity Curve", height=250,
                           template="plotly_dark", margin=dict(t=35,b=15))
    st.plotly_chart(fig_full, use_container_width=True)

    # ── Entry-Fill-Vergleich: Montag-Close vs. Dienstag-Open ──────────────
    st.markdown("---")
    st.subheader("Entry-Fill-Vergleich: Montag-Close vs. Dienstag-Open")
    st.caption("Gleiche Parameter, gleicher Zeitraum — nur der Fill-Zeitpunkt unterscheidet sich. "
               "Zeigt direkt, ob der 1-Bar-frühere Einstieg (Montag-Close) mehr Rendite bringt als der "
               "TradingView-Standard-Fill (Dienstag-Open).")
    with st.spinner("Fill-Vergleich läuft …"):
        _params_close     = {**base_params, "fill_mode": "close"}
        _params_next_open = {**base_params, "fill_mode": "next_open"}
        _tr_close,     _eq_close     = _momi_backtest_engine(df_raw.copy(), _params_close)
        _tr_next_open, _eq_next_open = _momi_backtest_engine(df_raw.copy(), _params_next_open)
        _m_close     = _momi_metrics(_tr_close,     _eq_close)
        _m_next_open = _momi_metrics(_tr_next_open, _eq_next_open)

    _cmp_df = pd.DataFrame([
        {"Fill-Modus": "Montag Close (Signal-Bar)",   "Rendite %": _m_close["total_ret"],     "Trades": _m_close["n"],
         "Win-Rate %": _m_close["wr"],     "Profit Factor": _m_close["pf"],     "Sharpe": _m_close["sharpe"],     "Max DD %": _m_close["max_dd"]},
        {"Fill-Modus": "Dienstag Open (nächster Bar)", "Rendite %": _m_next_open["total_ret"], "Trades": _m_next_open["n"],
         "Win-Rate %": _m_next_open["wr"], "Profit Factor": _m_next_open["pf"], "Sharpe": _m_next_open["sharpe"], "Max DD %": _m_next_open["max_dd"]},
    ])
    st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

    _better = "Montag Close" if _m_close["total_ret"] > _m_next_open["total_ret"] else "Dienstag Open"
    _diff   = abs(_m_close["total_ret"] - _m_next_open["total_ret"])
    st.info(f"**{_better}** liefert im Full-Sample-Test die höhere Rendite "
            f"({_m_close['total_ret']:.1f}% vs. {_m_next_open['total_ret']:.1f}%, Differenz {_diff:.1f} Prozentpunkte). "
            f"Der aktuell in Strategie-Parameter gewählte Fill-Modus (**{fill_mode_label}**) wird unten in WFA/Ensemble/Monte-Carlo verwendet — "
            f"stell ihn ggf. um, falls der andere Modus hier besser abschneidet.")

    # ════════════════════════════════════════════════════════════════════════
    # WALK-FORWARD ANALYSE
    # ════════════════════════════════════════════════════════════════════════
    _wfa_enabled = st.session_state.get("dax_wfa_ran", False)

    st.markdown("---")
    st.subheader("Walk-Forward Analyse")

    if not _wfa_enabled:
        n_months_total = (dax_end.year - dax_start.year) * 12 + (dax_end.month - dax_start.month)
        est_folds = max(0, (n_months_total - is_months) // oos_months)
        st.info(f"WFA noch nicht gestartet — klicke '🔄 WFA starten'. "
                f"Geschätzte Folds: **{est_folds}**")

    folds = []
    if _wfa_enabled:
        is_d  = pd.DateOffset(months=int(is_months))
        oos_d = pd.DateOffset(months=int(oos_months))
        fs    = df_raw.index[0]
        while True:
            ie = fs + is_d
            oe = ie + oos_d
            if oe > df_raw.index[-1]:
                break
            folds.append({"is_start": fs, "is_end": ie, "oos_start": ie, "oos_end": oe})
            fs = fs + oos_d

        if len(folds) < 2:
            st.warning("Zu wenig Daten für WFA. Zeitraum verlängern oder IS/OOS-Fenster verkleinern.")
            _wfa_enabled = False
        else:
            st.info(f"**{len(folds)} Folds** · IS {is_months}M / OOS {oos_months}M  "
                    f"· Grid-Größe: {len(g_sl)*len(g_ma)*len(g_fm)*len(g_tt)*len(g_to)} Kombinationen je Fold")

    def _dax_wfa_cache_key(_fm: str) -> str:
        return (f"dax_wfa_results_{_dax_ticker}_{dax_start}_{dax_end}"
                f"_{is_months}_{oos_months}_{entry_day}_{exit_day}"
                f"_{ma_type}_{sl_pct}_{use_trail}_{trail_trig}_{trail_off}"
                f"_{_fm}_{spread_pts}_{commission_pct}")

    def _dax_ens_cache_key(_fm: str) -> str:
        return f"dax_ens_results_{_dax_ticker}_{_fm}"

    def _dax_stability_best(_param_stability: dict) -> dict | None:
        """Leitet aus param_stability (WFA) das konsistenteste Setup ab — für Monte-Carlo-Quelle."""
        _rows = []
        for (sl_, tt_, to_, ma_, fm_), _results in _param_stability.items():
            if len(_results) < 2:
                continue
            _pfs  = [r["pf"] for r in _results if r["n"] > 0]
            _rets = [r["total_ret"] for r in _results if r["n"] > 0]
            if not _pfs:
                continue
            _n_pos = sum(1 for r in _rets if r > 0) if _rets else 0
            _n_folds = len(_results)
            _rows.append({
                "_sl": sl_, "_tt": tt_, "_to": to_, "_ma": ma_, "_fm": fm_,
                "_konsistenz": _n_pos / _n_folds * 100,
                "_pf": np.mean(_pfs),
                "_ret": np.mean(_rets) if _rets else 0,
            })
        if not _rows:
            return None
        _df = pd.DataFrame(_rows).sort_values(
            ["_konsistenz", "_pf", "_ret"], ascending=[False, False, False]).reset_index(drop=True)
        _b = _df.iloc[0]
        return {
            "sl_pct": float(_b["_sl"]), "ma_period": int(_b["_ma"]), "filter_mode": _b["_fm"],
            "trail_trig": float(_b["_tt"]), "trail_off": float(_b["_to"]),
            "key": (_b["_sl"], _b["_tt"], _b["_to"], _b["_ma"], _b["_fm"]),
            "label": (f"SL {_b['_sl']:.2f}% · Trail {_b['_tt']:.2f}%/{_b['_to']:.2f}% · "
                      f"MA {int(_b['_ma'])} · {_b['_fm']} ({int(_b['_konsistenz'])}% Konsistenz)"),
        }

    wfa_cache_key = _dax_wfa_cache_key(fill_mode)

    # ── Einmalig pro Session: gespeicherte WFA-Ergebnisse aus dem GitHub Gist laden ──
    # (überstehen App-Reboots/Redeploys — sonst wäre nach jedem Neustart alles weg)
    if not st.session_state.get("dax_wfa_gist_restored"):
        st.session_state["dax_wfa_gist_restored"] = True
        for _fm in ("close", "next_open"):
            _k = _dax_wfa_cache_key(_fm)
            if _k not in st.session_state:
                _restored = _load_wfa_result(f"dax_wfa_{_fm}")
                if _restored is not None:
                    # param_stability wurde fürs Gist auf reine Zahlenlisten komprimiert
                    # (siehe _dax_wfa_persist_blob) — hier wieder in die Dict-Form zurück,
                    # die _dax_stability_best() und die Stabilitätstabelle erwarten.
                    _restored["param_stability"] = {
                        key: [dict(zip(_PARAM_STAB_FIELDS, row)) for row in rows]
                        for key, rows in _restored.get("param_stability", {}).items()
                    }
                    if "best_setup_oos_trades" in _restored:
                        _restored["param_stability_trades"] = {
                            tuple(_restored["best_setup_key"]): [_restored["best_setup_oos_trades"]]
                        }
                    else:
                        _restored.setdefault("param_stability_trades", {})
                    st.session_state[_k] = _restored
                    st.session_state["dax_wfa_ran"] = True
                    st.session_state.setdefault("dax_wfa_restored_modes", []).append(_fm)

    if st.session_state.get("dax_wfa_restored_modes"):
        _restored_labels = ", ".join("Montag Close" if m == "close" else "Dienstag Open"
                                      for m in st.session_state.pop("dax_wfa_restored_modes"))
        st.caption(f"☁️ WFA-Ergebnis aus dem Cloud-Speicher wiederhergestellt ({_restored_labels}) — keine Neuberechnung nötig.")

    def _dax_wfa_persist_blob(_full_result: dict) -> dict:
        """Kompakte Version des WFA-Ergebnisses fürs Gist: lässt die Trades JEDER Grid-Kombi weg
        (würde die Gist-Datei sprengen) und behält nur die OOS-Trades des robustesten Setups —
        genug für Charts, Tabellen und die 'empfohlene' Monte-Carlo-Quelle nach einem Neustart.
        param_stability wird zusätzlich von Dicts auf reine Zahlenlisten komprimiert — bei
        vielen Grid-Kombinationen × Folds sparen die wegfallenden Schlüsselnamen genug Platz,
        um sicher unter GitHubs 1MB-Trunkierungsgrenze für Gist-Dateien zu bleiben."""
        _blob = {k: v for k, v in _full_result.items() if k != "param_stability_trades"}
        _blob["param_stability"] = {
            key: [[r.get(f, 0) for f in _PARAM_STAB_FIELDS] for r in results]
            for key, results in _full_result["param_stability"].items()
        }
        _best = _dax_stability_best(_full_result["param_stability"])
        if _best is not None:
            _best_trades = _full_result["param_stability_trades"].get(_best["key"])
            if _best_trades:
                _blob["best_setup_key"] = list(_best["key"])
                _blob["best_setup_oos_trades"] = pd.concat(_best_trades, ignore_index=True)
        return _blob

    def _run_dax_wfa(_base_params_variant: dict, _mode_label: str) -> dict:
        progress = st.progress(0, text=f"Walk-Forward läuft ({_mode_label}) …")
        wfa_rows, oos_equities, oos_trades_all = [], [], []
        param_stability: dict = {}
        param_stability_trades: dict = {}  # key -> Liste der OOS-Trades-DataFrames dieser exakten Kombi (über Folds)

        for fi, fold in enumerate(folds):
            progress.progress(fi / len(folds), text=f"{_mode_label}: Fold {fi+1}/{len(folds)} — optimiere IS …")

            df_is  = df_raw[(df_raw.index >= fold["is_start"]) & (df_raw.index < fold["is_end"])].copy()
            df_oos = df_raw[(df_raw.index >= fold["oos_start"]) & (df_raw.index < fold["oos_end"])].copy()

            if len(df_is) < 30 or len(df_oos) < 5:
                continue

            # IS Grid Search — teste ALLE Kombinationen und sammle OOS-Ergebnis je Combo
            best_score, best_p = -np.inf, None
            adx_grid = g_adx if use_adx else [float(adx_thresh)]

            for sl_, ma_, fm_, adx_, tt_, to_, ed_, xd_ in _prod(
                    g_sl, g_ma, g_fm, adx_grid, g_tt, g_to, g_entry_dows, g_exit_dows):
                if ed_ == xd_:
                    continue  # Entry- und Exit-Tag müssen verschieden sein
                p = {**_base_params_variant,
                     "sl_pct":      sl_,
                     "ma_period":   ma_,
                     "filter_mode": fm_,
                     "adx_thresh":  adx_,
                     "trail_trig":  tt_,
                     "trail_off":   to_,
                     "entry_dow":   ed_,
                     "exit_dow":    xd_}
                try:
                    tr_, eq_ = _momi_backtest_engine(df_is.copy(), p)
                except Exception:
                    continue
                if len(tr_) < int(min_trades):
                    continue
                m_ = _momi_metrics(tr_, eq_)
                score = m_[opt_metric]
                if score > best_score:
                    best_score, best_p = score, p.copy()

                # OOS sofort für diese Kombination berechnen → Stabilitätsanalyse
                try:
                    tr_oos_, eq_oos_ = _momi_backtest_engine(df_oos.copy(), p)
                    m_oos_ = _momi_metrics(tr_oos_, eq_oos_)
                    key = (sl_, tt_, to_, ma_, fm_)
                    if key not in param_stability:
                        param_stability[key] = []
                        param_stability_trades[key] = []
                    param_stability[key].append(m_oos_)
                    if not tr_oos_.empty:
                        param_stability_trades[key].append(tr_oos_)
                except Exception:
                    pass

            if best_p is None:
                wfa_rows.append({"Fold": fi+1,
                                 "IS": f"{fold['is_start'].date()} – {fold['is_end'].date()}",
                                 "OOS": f"{fold['oos_start'].date()} – {fold['oos_end'].date()}",
                                 "Bester SL":"–","Bestes MA":"–","Bester FM":"–",
                                 "IS Trades":0,"IS PF":"–","IS Sharpe":"–",
                                 "OOS Trades":0,"OOS PF":"–","OOS Ret %":"–","Ø Ret/Trade %":"–","OOS Sharpe":"–",
                                 "Status":"⚠️ Kein IS-Ergebnis"})
                continue

            tr_is,  eq_is  = _momi_backtest_engine(df_is.copy(),  best_p)
            tr_oos, eq_oos = _momi_backtest_engine(df_oos.copy(), best_p)
            m_is  = _momi_metrics(tr_is,  eq_is)
            m_oos = _momi_metrics(tr_oos, eq_oos)

            oos_equities.append((fi+1, eq_oos))
            if not tr_oos.empty:
                oos_trades_all.append(tr_oos)

            ok = m_oos["n"] >= 1 and m_oos["pf"] > 1.0 and m_oos["total_ret"] > 0
            _dow_names = {0:"Mo",1:"Di",2:"Mi",3:"Do",4:"Fr",5:"Sa",6:"So"}
            wfa_rows.append({
                "Fold":       fi+1,
                "IS":         f"{fold['is_start'].date()} – {fold['is_end'].date()}",
                "OOS":        f"{fold['oos_start'].date()} – {fold['oos_end'].date()}",
                "Bester SL":  f"{best_p['sl_pct']}%",
                "Bestes MA":  f"{best_p['ma_type']} {best_p['ma_period']}",
                "Bester FM":  best_p['filter_mode'],
                "Entry→Exit": f"{_dow_names.get(best_p['entry_dow'],'?')}→{_dow_names.get(best_p['exit_dow'],'?')}",
                "IS Trades":  m_is["n"],
                "IS PF":      m_is["pf"],
                "IS Sharpe":  m_is["sharpe"],
                "OOS Trades":    m_oos["n"],
                "OOS PF":        m_oos["pf"],
                "OOS Ret %":     m_oos["total_ret"],
                "Ø Ret/Trade %": m_oos["avg_ret"],
                "OOS Sharpe":    m_oos["sharpe"],
                "Status":        "✅ Bestanden" if ok else "❌ Fail",
            })

        progress.progress(1.0, text=f"Walk-Forward ({_mode_label}) abgeschlossen ✓")
        _tr_full_variant, _ = _momi_backtest_engine(df_raw.copy(), _base_params_variant)
        _oos_trades_concat = pd.concat(oos_trades_all, ignore_index=True) if oos_trades_all else pd.DataFrame()
        return {
            "wfa_rows":        wfa_rows,
            "oos_equities":    oos_equities,
            "param_stability": param_stability,
            "param_stability_trades": param_stability_trades,
            "base_params":     _base_params_variant,
            "grids":           (g_sl, g_ma, g_fm, g_tt, g_to),
            "full_trades":     _tr_full_variant,
            "oos_trades":      _oos_trades_concat,
        }

    def _run_dax_global_benchmark(_base_params_variant: dict) -> dict | None:
        """Optimiert EINMAL auf dem GESAMTEN Zeitraum, ohne IS/OOS-Split — der klassische
        Überanpassungs-Vergleichswert: wie gut sieht das im Nachhinein beste Setup aus, wenn
        man (unehrlich) auf allen Daten gleichzeitig optimiert? Die Lücke zum echten WFA-OOS-
        Ergebnis zeigt, wie stark die Strategie von der Walk-Forward-Disziplin abhängt."""
        best_score, best_p = -np.inf, None
        adx_grid = g_adx if use_adx else [float(adx_thresh)]
        for sl_, ma_, fm_, adx_, tt_, to_, ed_, xd_ in _prod(
                g_sl, g_ma, g_fm, adx_grid, g_tt, g_to, g_entry_dows, g_exit_dows):
            if ed_ == xd_:
                continue
            p = {**_base_params_variant, "sl_pct": sl_, "ma_period": ma_, "filter_mode": fm_,
                 "adx_thresh": adx_, "trail_trig": tt_, "trail_off": to_, "entry_dow": ed_, "exit_dow": xd_}
            try:
                tr_, eq_ = _momi_backtest_engine(df_raw.copy(), p)
            except Exception:
                continue
            if len(tr_) < int(min_trades):
                continue
            m_ = _momi_metrics(tr_, eq_)
            score = m_[opt_metric]
            if score > best_score:
                best_score, best_p = score, p.copy()
        if best_p is None:
            return None
        tr_g, eq_g = _momi_backtest_engine(df_raw.copy(), best_p)
        m_g = _momi_metrics(tr_g, eq_g)
        return {"params": best_p, "metrics": m_g}

    if _wfa_enabled and (run_btn or wfa_cache_key not in st.session_state):
        _other_fm = "next_open" if fill_mode == "close" else "close"
        _modes_to_run = [fill_mode, _other_fm] if (run_btn and test_both_fills) else [fill_mode]
        for _fm in _modes_to_run:
            _mode_label = "Montag Close" if _fm == "close" else "Dienstag Open"
            _variant_params = {**base_params, "fill_mode": _fm}
            _wfa_result = _run_dax_wfa(_variant_params, _mode_label)
            with st.spinner(f"{_mode_label}: Globaler Benchmark (Überanpassungs-Check) …"):
                _wfa_result["global_benchmark"] = _run_dax_global_benchmark(_variant_params)
            st.session_state[_dax_wfa_cache_key(_fm)] = _wfa_result
            _wfa_saved, _wfa_save_reason = _save_wfa_result(f"dax_wfa_{_fm}", _dax_wfa_persist_blob(_wfa_result))
            if _wfa_saved:
                st.caption(f"☁️ {_mode_label}: WFA-Ergebnis dauerhaft gespeichert — übersteht Reboots/Neustarts.")
            else:
                st.caption(f"⚠️ {_mode_label}: Konnte nicht dauerhaft gespeichert werden — {_wfa_save_reason}. "
                           f"Bleibt nur für diese Browser-Session erhalten.")
        # Trades auch direkt unter Ticker-Key speichern — MC findet sie ohne exakten Cache-Key
        st.session_state[f"dax_mc_trades_{_dax_ticker}"] = tr_full

    # Ergebnisse aus Cache laden
    if not _wfa_enabled or wfa_cache_key not in st.session_state:
        _cached = None
    else:
        _cached = st.session_state[wfa_cache_key]
    if _cached is None:
        wfa_rows, oos_equities, param_stability = [], [], {}
        base_params_display = base_params
        g_sl, g_ma, g_fm, g_tt, g_to = [], [], [], [], []
        wfa_oos_trades = pd.DataFrame()
        _wfa_global_benchmark = None
        _wfa_enabled = False
    else:
        wfa_rows        = _cached["wfa_rows"]
        oos_equities    = _cached["oos_equities"]
        param_stability = _cached["param_stability"]
        base_params     = _cached["base_params"]
        g_sl, g_ma, g_fm, g_tt, g_to = _cached["grids"]
        wfa_oos_trades  = _cached.get("oos_trades", pd.DataFrame())
        # .get() statt [] — ältere gespeicherte Ergebnisse (vor diesem Feature) haben den Key noch nicht
        _wfa_global_benchmark = _cached.get("global_benchmark")

    if _wfa_enabled and not wfa_rows:
        st.error("Keine WFA-Ergebnisse — Parameter oder Zeitraum anpassen.")
        _wfa_enabled = False

    _wfa_best_setup = None  # wird unten gefüllt, falls Parameter-Stabilitätsanalyse ein Ergebnis liefert

    if _wfa_enabled:
        wfa_df = pd.DataFrame(wfa_rows)
        n_ok   = (wfa_df["Status"] == "✅ Bestanden").sum()
        n_tot  = len(wfa_df)

        # ── Gesamt-Badge ─────────────────────────────────────────────────────
        if n_ok >= int(min_folds):
            bc, bt = "#22c55e", f"✅ ROBUST — {n_ok}/{n_tot} Folds bestanden · Strategie empfohlen"
        elif n_ok >= 2:
            bc, bt = "#f0c040", f"⚠️ INSTABIL — nur {n_ok}/{n_tot} Folds bestanden · mit Vorsicht handeln"
        elif n_ok == 1:
            bc, bt = "#ef5350", f"❌ NICHT EMPFOHLEN — nur 1/{n_tot} Fold bestanden · Strategie funktioniert auf diesem Asset NICHT zuverlässig"
        else:
            bc, bt = "#ef5350", f"❌ GESCHEITERT — 0/{n_tot} Folds bestanden · Strategie NICHT für dieses Asset geeignet"

        st.markdown(
            f'<div style="background:{bc}22;border:2px solid {bc};border-radius:10px;'
            f'padding:16px 24px;font-weight:800;font-size:1.2rem;color:{bc};margin:16px 0;">'
            f'{bt}</div>', unsafe_allow_html=True)

        _wfa_fill_label = "Montag Close (Signal-Bar)" if base_params.get("fill_mode", "close") == "close" else "Dienstag Open (nächster Bar)"
        st.caption(f"📌 Getesteter Fill-Modus in diesem WFA-Lauf: **{_wfa_fill_label}** · "
                   f"Spread {base_params.get('spread_pts', 0):.1f} Pkt · Kommission {base_params.get('commission_pct', 0):.2f}% — "
                   f"ändere den Fill-Modus oben und klicke erneut 'WFA starten', um den anderen Modus zu testen.")

        # ── Statistische Signifikanz der gepoolten OOS-Trades (Wilcoxon + Wilson-CI) ──
        # Beantwortet: ist die Kante über alle Folds hinweg von Zufall unterscheidbar,
        # unabhängig davon wie viele Folds das grobe ✅/❌-Badge oben zählt.
        if not wfa_oos_trades.empty:
            _sig = evaluate_edge(wfa_oos_trades, min_trades=30, alpha=0.05, min_sharpe_oos=1.0, pnl_col="PnL $")
            _sig_style = {"handelbar": ("#22c55e", "🟢 Statistisch signifikant"),
                          "grenzwertig": ("#f0c040", "🟡 Grenzwertig"),
                          "nicht handelbar": ("#ef5350", "🔴 Nicht signifikant")}
            _sig_color, _sig_label = _sig_style[_sig["status"]]
            _sig_p = "n/a" if math.isnan(_sig["p_value"]) else f"{_sig['p_value']:.4f}"
            _sig_sharpe = "n/a" if math.isnan(_sig["sharpe_oos"]) else f"{_sig['sharpe_oos']:.2f}"
            _sig_wlo, _sig_whi = _sig["wilson_ci"]
            st.markdown(
                f"""<div style="background:#0a1220;border:1px solid {_sig_color}55;border-radius:8px;
                padding:12px 18px;margin:8px 0 14px 0;">
                  <span style="background:{_sig_color}22;border:1px solid {_sig_color}55;border-radius:6px;
                    padding:4px 12px;color:{_sig_color};font-weight:800;font-size:.95rem;">{_sig_label}</span>
                  <span style="color:#9fb0c7;font-size:.85rem;margin-left:12px;">
                    OOS-Trades gepoolt: n={_sig['n_trades']} · Wilcoxon p={_sig_p}
                    · Sharpe OOS={_sig_sharpe} · Wilson-CI Winrate=[{_sig_wlo*100:.1f}%, {_sig_whi*100:.1f}%]
                  </span>
                </div>""",
                unsafe_allow_html=True,
            )
            if _sig["reasons"]:
                st.caption("⚠️ " + " · ".join(_sig["reasons"]))

        # ── Empfohlenes Setup, fett & prominent (nicht der Überanpassungs-Benchmark!) ──
        # Dieselbe Auswahl-Logik, die auch die Monte-Carlo-Trade-Quelle "WFA robustestes
        # Setup" speist — EIN Wert, keine zwei widersprüchlichen "besten" Setups mehr.
        _wfa_recommended = _dax_stability_best(param_stability)
        if _wfa_recommended is not None:
            st.markdown(
                f"""<div style="background:#052e1655;border:2px solid #22c55e;border-radius:10px;
                padding:16px 22px;margin:10px 0 18px 0;">
                  <div style="color:#4ade80;font-weight:800;font-size:.8rem;letter-spacing:.05em;
                    text-transform:uppercase;margin-bottom:6px;">🏆 Empfohlenes Setup ({_wfa_fill_label})</div>
                  <div style="color:#e2e8f0;font-weight:800;font-size:1.15rem;">{_wfa_recommended['label']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption("Das ist die walk-forward-validierte Empfehlung (robustestes Setup über alle Folds) — "
                       "nicht zu verwechseln mit dem 'Globalen Benchmark' oben, der bewusst überoptimiert ist "
                       "und nur als Vergleichswert für den Überanpassungs-Check dient, nicht als Empfehlung.")
        else:
            st.info("Noch kein robustes Setup ermittelbar — Parameter-Stabilitätsanalyse weiter unten prüfen.")

        # ── Vergleich: beide Fill-Modi (falls beide schon berechnet wurden) ────
        _cmp_rows = []
        for _fm, _label in [("close", "Montag Close"), ("next_open", "Dienstag Open")]:
            _k = _dax_wfa_cache_key(_fm)
            if _k in st.session_state:
                _c = st.session_state[_k]
                _c_df = pd.DataFrame(_c["wfa_rows"])
                if not _c_df.empty:
                    _c_ok  = (_c_df["Status"] == "✅ Bestanden").sum()
                    _c_tot = len(_c_df)
                    _c_pfs = [r["OOS PF"] for r in _c["wfa_rows"] if isinstance(r["OOS PF"], (int,float))]
                    _c_rets = [r["OOS Ret %"] for r in _c["wfa_rows"] if isinstance(r["OOS Ret %"], (int,float))]
                    _c_shs = [r["OOS Sharpe"] for r in _c["wfa_rows"] if isinstance(r.get("OOS Sharpe"), (int,float))]
                    _cmp_rows.append({
                        "Fill-Modus": _label, "Folds bestanden": f"{_c_ok}/{_c_tot}",
                        "Ø OOS PF": round(np.mean(_c_pfs), 2) if _c_pfs else "–",
                        "Ø OOS Rendite %": round(np.mean(_c_rets), 1) if _c_rets else "–",
                        "Ø OOS Sharpe": round(np.mean(_c_shs), 2) if _c_shs else "–",
                    })
        if len(_cmp_rows) == 2:
            st.markdown("#### Vergleich: Montag Close vs. Dienstag Open (WFA)")
            st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)
        elif len(_cmp_rows) == 1:
            st.caption("ℹ️ Nur ein Fill-Modus bisher getestet — Checkbox oben aktivieren und 'WFA starten' "
                       "klicken, um beide zu speichern und hier zu vergleichen.")

        # ── KPI-Zusammenfassung über alle OOS-Folds ───────────────────────────
        oos_pfs   = [r["OOS PF"]    for r in wfa_rows if isinstance(r["OOS PF"],    (int,float))]
        oos_rets  = [r["OOS Ret %"] for r in wfa_rows if isinstance(r["OOS Ret %"], (int,float))]
        oos_shs   = [r["OOS Sharpe"] for r in wfa_rows if isinstance(r.get("OOS Sharpe"), (int,float))]

        sc1,sc2,sc3,sc4 = st.columns(4)
        sc1.metric("Ø OOS Profit Factor", f"{np.mean(oos_pfs):.2f}"  if oos_pfs  else "–")
        sc2.metric("Ø OOS Rendite",       f"{np.mean(oos_rets):.1f}%" if oos_rets else "–")
        sc3.metric("Ø OOS Sharpe",        f"{np.mean(oos_shs):.2f}"   if oos_shs  else "–")
        sc4.metric("Bestandene Folds",    f"{n_ok}/{n_tot}")

        # ── Fold-Tabelle ──────────────────────────────────────────────────────
        st.subheader("Fold-Ergebnisse")
        def _color_status(v):
            if v == "✅ Bestanden": return "color:#22c55e;font-weight:700"
            if v == "❌ Fail":      return "color:#ef5350;font-weight:700"
            return "color:#f0c040"
        def _color_num(v):
            if isinstance(v, (int,float)):
                return "color:#22c55e" if v > 0 else "color:#ef5350"
            return ""
        num_cols = [c for c in ["OOS Ret %","OOS PF","OOS Sharpe","Ø Ret/Trade %"] if c in wfa_df.columns]
        styled = wfa_df.style\
            .map(_color_status, subset=["Status"])\
            .map(_color_num,    subset=num_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── OOS Rendite Bar-Chart ─────────────────────────────────────────────
        st.subheader("OOS-Rendite je Fold")
        fold_labels = [f"Fold {r['Fold']}" for r in wfa_rows]
        bar_colors  = ["#22c55e" if isinstance(r["OOS Ret %"],(int,float)) and r["OOS Ret %"]>0
                       else "#ef5350" for r in wfa_rows]
        bar_vals    = [r["OOS Ret %"] if isinstance(r["OOS Ret %"],(int,float)) else 0 for r in wfa_rows]

        fig_bar = go.Figure(go.Bar(
            x=fold_labels, y=bar_vals, marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in bar_vals], textposition="outside",
            width=0.5))
        fig_bar.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
        fig_bar.update_layout(title="OOS-Rendite je Fold (grün = profitabel)",
                              height=380, template="plotly_dark",
                              yaxis_title="Rendite %",
                              xaxis=dict(tickfont=dict(size=13)),
                              margin=dict(t=50,b=60,l=60,r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Gestapelte OOS Equity Curves ──────────────────────────────────────
        if oos_equities:
            # ── Kombinierte OOS Equity (Folds hintereinander = simulierter Live-Handel) ──
            st.subheader("Kombinierte OOS Equity Kurve")
            st.caption("Alle OOS-Folds hintereinander — so hätte sich das Kapital im echten Handel entwickelt (nur blind getestete Perioden, kein IS)")

            # Folds chronologisch sortieren und aneinanderhängen
            oos_equities_sorted = sorted(oos_equities, key=lambda x: x[1].index[0])
            combined_vals, combined_idx = [], []
            running_capital = 10_000.0
            for fi, eq in oos_equities_sorted:
                scale = running_capital / eq.iloc[0]
                scaled = eq * scale
                combined_vals.extend(scaled.values.tolist())
                combined_idx.extend(eq.index.tolist())
                running_capital = scaled.iloc[-1]

            combined_eq = pd.Series(combined_vals, index=combined_idx)

            # Drawdown berechnen
            roll_max = combined_eq.cummax()
            drawdown = (combined_eq - roll_max) / roll_max * 100

            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(
                x=combined_eq.index, y=combined_eq.values,
                fill="tozeroy", fillcolor="rgba(247,147,26,0.15)",
                line=dict(color="#f7931a", width=2.5),
                name="Equity (OOS kombiniert)"))
            # Fold-Grenzen als vertikale Linien
            colors_fold = ["#42a5f5","#00d4aa","#ab47bc","#ffa726","#66bb6a","#ef5350","#26c6da","#f7931a"]
            for i, (fi, eq) in enumerate(oos_equities_sorted):
                _fold_color = colors_fold[i % len(colors_fold)]
                fig_combined.add_vline(x=eq.index[0], line_dash="dot", line_color=_fold_color)
                fig_combined.add_annotation(x=eq.index[0], y=1, yref="paper", yanchor="bottom",
                                            text=f"F{fi}", showarrow=False, font=dict(color=_fold_color))
            fig_combined.add_hline(y=10_000, line_color="white", line_dash="dash", line_width=1)
            total_ret = (running_capital - 10_000) / 10_000 * 100
            fig_combined.update_layout(
                title=f"OOS Equity — 10.000€ Start → {running_capital:,.0f}€ ({total_ret:+.1f}%) | Nur blind getestete Perioden",
                height=420, template="plotly_dark",
                yaxis_title="Kapital (€)", xaxis_title="",
                margin=dict(t=55, b=20, l=70, r=20))
            st.plotly_chart(fig_combined, use_container_width=True)

            # Drawdown Chart
            fig_dd = go.Figure(go.Scatter(
                x=combined_eq.index, y=drawdown.values,
                fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                line=dict(color="#ef5350", width=1.5), name="Drawdown %"))
            fig_dd.update_layout(
                title=f"Drawdown — Max: {drawdown.min():.1f}%",
                height=200, template="plotly_dark",
                yaxis_title="DD %", margin=dict(t=40, b=20, l=70, r=20))
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── Überanpassungs-Gap ────────────────────────────────────────────
            # WFA-OOS (blind, ehrlich) vs. Globaler Benchmark (im Nachhinein auf ALLEN
            # Daten optimiert, kein OOS-Split — strukturell zu optimistisch). Große Lücke
            # = das "beste" Setup ist stark an genau diese eine Historie angepasst.
            if _wfa_global_benchmark is not None:
                st.markdown("### 🎯 Überanpassungs-Gap")
                _glob_ret = _wfa_global_benchmark["metrics"]["total_ret"]
                _gap = _glob_ret - total_ret
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric("WFA OOS Rendite (blind, ehrlich)", f"{total_ret:+.1f}%")
                gc2.metric("Globaler Benchmark (im Nachhinein optimiert)", f"{_glob_ret:+.1f}%")
                gc3.metric("Überanpassungs-Gap", f"{_gap:+.1f}%", delta_color="inverse")
                _gp = _wfa_global_benchmark["params"]
                st.caption(f"⚠️ Setup im globalen Benchmark (NICHT die Empfehlung — nur Vergleichswert): "
                           f"SL {_gp['sl_pct']}% · Trail {_gp['trail_trig']}%/{_gp['trail_off']}% · "
                           f"MA {_gp['ma_type']} {_gp['ma_period']} · Filter: {_gp['filter_mode']} — "
                           f"auf dem GESAMTEN Zeitraum ohne OOS-Trennung gefunden, deshalb strukturell zu gut. "
                           f"Die tatsächliche Empfehlung steht im grünen Kasten '🏆 Empfohlenes Setup' weiter oben. "
                           f"Ensemble ist hier nicht mit eingerechnet — das ist eine separate Analyse weiter unten.")
                if _gap > 15:
                    st.warning(f"⚠️ Große Lücke ({_gap:+.1f}%) — das im Nachhinein beste Setup performt deutlich besser als "
                               f"die ehrliche Walk-Forward-Auswahl. Hinweis auf Überanpassung an diese eine Historie.")
                elif _gap < 5:
                    st.success(f"✅ Kleine Lücke ({_gap:+.1f}%) — WFA-OOS liegt nah am optimistischen Bestfall, "
                               f"spricht für eine robuste, nicht überangepasste Strategie.")
            else:
                st.caption("ℹ️ Kein globaler Benchmark verfügbar (z.B. altes gespeichertes Ergebnis vor diesem Feature) — "
                           "einmal 'WFA starten' erneut klicken, um den Überanpassungs-Gap zu berechnen.")

            # ── Einzelne Folds übereinander (zum Vergleich) ───────────────────
            with st.expander("Einzelne OOS-Folds im Vergleich"):
                fig_eq = go.Figure()
                colors = ["#f7931a","#00d4aa","#42a5f5","#ab47bc","#ffa726","#66bb6a","#ef5350","#26c6da"]
                for fi, eq in oos_equities:
                    norm = eq / eq.iloc[0] * 100
                    fig_eq.add_trace(go.Scatter(x=eq.index, y=norm.values,
                                                name=f"Fold {fi}",
                                                line=dict(color=colors[(fi-1) % len(colors)], width=1.5)))
                fig_eq.add_hline(y=100, line_color="white", line_dash="dash", line_width=1)
                fig_eq.update_layout(title="OOS Equity je Fold (normiert auf 100 = Startkapital)",
                                     height=350, template="plotly_dark", margin=dict(t=40,b=20))
                st.plotly_chart(fig_eq, use_container_width=True)

        # ── CSV Download ──────────────────────────────────────────────────────
        st.download_button("⬇️ WFA-Ergebnis als CSV",
                           data=wfa_df.to_csv(index=False).encode(),
                           file_name="dax_ema_wfa.csv", mime="text/csv")

        # ════════════════════════════════════════════════════════════════════════
        # PARAMETER-STABILITÄTSANALYSE
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("Parameter-Stabilitätsanalyse")
        st.caption("Welche SL / Trailing-Kombinationen liefern konsistent über ALLE Folds gute OOS-Ergebnisse?")

        if param_stability:
            stab_rows = []
            for (sl_, tt_, to_, ma_, fm_), results in param_stability.items():
                if len(results) < 2:
                    continue
                pfs      = [r["pf"]         for r in results if r["n"] > 0]
                rets     = [r["total_ret"]   for r in results if r["n"] > 0]
                sharpes  = [r["sharpe"]      for r in results if r["n"] > 0]
                trades   = [r["n"]           for r in results]
                n_pos    = sum(1 for r in rets if r > 0) if rets else 0
                n_folds  = len(results)

                if not pfs:
                    continue

                stab_rows.append({
                    "_sl": sl_, "_tt": tt_, "_to": to_, "_ma": ma_, "_fm": fm_,
                    "SL %":             f"{sl_:.2f}%",
                    "Trail-Trig %":     f"{tt_:.2f}%",
                    "Trail-Off %":      f"{to_:.2f}%",
                    "MA":               f"{ma_}",
                    "Filter":           fm_,
                    "Folds getestet":   n_folds,
                    "Profitable Folds": n_pos,
                    "Konsistenz %":     round(n_pos / n_folds * 100, 0),
                    "Ø OOS PF":         round(np.mean(pfs), 2),
                    "Ø OOS Ret %":      f"{round(np.mean(rets), 2):.2f}%",
                    "Ø OOS Sharpe":     round(np.mean(sharpes), 2),
                    "Min OOS Ret %":    f"{round(np.min(rets), 2):.2f}%",
                    "Max DD Ø":         f"{round(np.mean([r['max_dd'] for r in results]), 2):.2f}%",
                })

            if stab_rows:
                df_stab = pd.DataFrame(stab_rows)
                # Sortierung: erst Konsistenz, dann Ø PF
                df_stab = df_stab.sort_values(
                    ["Konsistenz %", "Ø OOS PF", "Ø OOS Ret %"],
                    ascending=[False, False, False]
                ).reset_index(drop=True)

                # Top 3 hervorheben
                st.markdown("#### 🏆 Top-10 stabilste Setups")
                top10 = df_stab.head(10).copy()

                def _hl_konsistenz(v):
                    if isinstance(v, (int, float)):
                        if v >= 80: return "color:#22c55e;font-weight:700"
                        if v >= 60: return "color:#f0c040"
                        return "color:#ef5350"
                    return ""

                def _hl_num(v):
                    if isinstance(v, (int, float)):
                        return "color:#22c55e" if v > 0 else "color:#ef5350"
                    return ""

                styled_stab = top10.style\
                    .map(_hl_konsistenz, subset=["Konsistenz %"])\
                    .map(_hl_num, subset=["Ø OOS PF"])
                st.dataframe(styled_stab, use_container_width=True, hide_index=True)

                # Heatmap: SL% vs Trail-Trig% → Ø Konsistenz
                st.markdown("#### Heatmap: SL% × Trail-Trigger% → Konsistenz %")
                pivot = df_stab.pivot_table(
                    values="Konsistenz %",
                    index="SL %",
                    columns="Trail-Trig %",
                    aggfunc="mean"
                ).round(0)

                fig_heat = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[f"Trail {c}%" for c in pivot.columns],
                    y=[f"SL {r}%" for r in pivot.index],
                    colorscale="RdYlGn",
                    zmin=0, zmax=100,
                    text=pivot.values.round(0).astype(str),
                    texttemplate="%{text}%",
                    showscale=True,
                    colorbar=dict(title="Konsistenz %")
                ))
                fig_heat.update_layout(
                    title="Ø Konsistenz je SL% / Trail-Trigger% Kombination",
                    height=350, template="plotly_dark",
                    margin=dict(t=50, b=30, l=80, r=20)
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # Bestes Setup hervorheben
                best = df_stab.iloc[0]
                st.success(
                    f"**Stabilstes Setup:** SL {best['SL %']} · "
                    f"Trail-Trigger {best['Trail-Trig %']} · Trail-Abstand {best['Trail-Off %']} · "
                    f"MA {best['MA']} · Filter: {best['Filter']} → "
                    f"**{int(best['Konsistenz %'])}% Konsistenz** · "
                    f"Ø OOS PF {best['Ø OOS PF']} · Ø OOS Ret {best['Ø OOS Ret %']}"
                )
                _wfa_best_setup = {
                    "sl_pct":      float(best["_sl"]),
                    "ma_period":   int(best["_ma"]),
                    "filter_mode": best["_fm"],
                    "trail_trig":  float(best["_tt"]),
                    "trail_off":   float(best["_to"]),
                    "label":       (f"SL {best['SL %']} · Trail {best['Trail-Trig %']}/{best['Trail-Off %']} · "
                                    f"MA {best['MA']} · {best['Filter']} ({int(best['Konsistenz %'])}% Konsistenz)"),
                }

                # Download
                st.download_button("⬇️ Stabilitätsanalyse als CSV",
                                   data=df_stab.to_csv(index=False).encode(),
                                   file_name="dax_param_stability.csv", mime="text/csv")
            else:
                st.info("Zu wenig Daten für Stabilitätsanalyse — mehr Folds oder breiteren Grid verwenden.")

        # ════════════════════════════════════════════════════════════════════════
        # ENSEMBLE WFA — N Läufe mit versetzten Startdaten
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("Ensemble WFA — Mehrfach-Lauf")
        st.markdown("""
    Führt den WFA **N Mal** durch, jedes Mal mit einem um 1 Monat versetzten Startdatum.
    So entstehen echte, unterschiedliche Folds — und du siehst welche Parameter **immer wieder** gewinnen, unabhängig vom Startpunkt.
        """)

        if "dax_ens_running" not in st.session_state:
            st.session_state["dax_ens_running"] = False

        # Einmalig pro Session: gespeicherte Ensemble-Ergebnisse aus dem GitHub Gist laden
        if not st.session_state.get("dax_ens_gist_restored"):
            st.session_state["dax_ens_gist_restored"] = True
            for _fm in ("close", "next_open"):
                _k = _dax_ens_cache_key(_fm)
                if _k not in st.session_state:
                    _restored_ens = _load_wfa_result(f"dax_ens_{_fm}")
                    if _restored_ens is not None:
                        st.session_state[_k] = _restored_ens
                        st.session_state.setdefault("dax_ens_restored_modes", []).append(_fm)

        if st.session_state.get("dax_ens_restored_modes"):
            _restored_ens_labels = ", ".join("Montag Close" if m == "close" else "Dienstag Open"
                                              for m in st.session_state.pop("dax_ens_restored_modes"))
            st.caption(f"☁️ Ensemble-Ergebnis aus dem Cloud-Speicher wiederhergestellt ({_restored_ens_labels}) — keine Neuberechnung nötig.")

        # Status-Badge: Ensemble bereits gelaufen?
        _ens_status_key = _dax_ens_cache_key(fill_mode)
        if _ens_status_key in st.session_state:
            _es = st.session_state[_ens_status_key]
            st.success(f"✅ Ensemble bereits gelaufen — {_es['n_runs']} Läufe · Zeitraum: {_es['tested_period']} · "
                       f"Ergebnisse werden unten angezeigt. Neu starten um zu aktualisieren.")
        else:
            st.info("⏳ Ensemble noch nicht gestartet — klicke '▶ Ensemble WFA starten'.")

        ec1, ec2 = st.columns(2)
        n_runs = ec1.number_input("Anzahl Läufe", min_value=3, max_value=10, value=5, step=1, key="dax_ens_runs")
        if ec2.button("▶ Ensemble WFA starten", type="primary", key="dax_ens_run_btn"):
            st.session_state["dax_ens_running"] = True
        _ens_n_combos = len(g_sl)*len(g_ma)*len(g_fm)*len(g_tt)*len(g_to)
        _ens_modes_factor = 2 if test_both_fills else 1
        st.caption(f"⏱️ Grober Richtwert: {_ens_n_combos} Kombinationen × ~{n_runs} Läufe × {_ens_modes_factor} Fill-Modus/-Modi — "
                   f"bei breitem Suchraum kann das **mehrere Dutzend Minuten** dauern. Die Fortschrittsanzeige aktualisiert sich "
                   f"jetzt pro Fold statt nur pro Lauf, sollte also nicht mehr eingefroren wirken.")

        if st.session_state["dax_ens_running"]:
            _other_fm_ens = "next_open" if fill_mode == "close" else "close"
            _modes_to_run = [fill_mode, _other_fm_ens] if test_both_fills else [fill_mode]
            st.info(f"Starte {n_runs} WFA-Läufe mit versetzten Startdaten … ({len(_modes_to_run)} Fill-Modus/-Modi)")

            for _fm in _modes_to_run:
                _mode_label = "Montag Close" if _fm == "close" else "Dienstag Open"
                _variant_params = {**base_params, "fill_mode": _fm}
                ens_progress = st.progress(0, text=f"Ensemble läuft ({_mode_label}) …")

                ensemble_stability: dict = {}  # key=(sl,tt,to,ma,fm) → list of OOS metrics über ALLE Läufe

                for run_i in range(int(n_runs)):
                    run_start = pd.Timestamp(dax_start) + pd.DateOffset(months=run_i)
                    df_run    = df_raw[df_raw.index >= run_start].copy()

                    if len(df_run) < 60:
                        continue

                    # Folds für diesen Lauf
                    is_d_r  = pd.DateOffset(months=int(is_months))
                    oos_d_r = pd.DateOffset(months=int(oos_months))
                    folds_r, fs_r = [], df_run.index[0]
                    while True:
                        ie_r = fs_r + is_d_r
                        oe_r = ie_r + oos_d_r
                        if oe_r > df_run.index[-1]:
                            break
                        folds_r.append({"is_start": fs_r, "is_end": ie_r,
                                         "oos_start": ie_r, "oos_end": oe_r})
                        fs_r = fs_r + oos_d_r

                    if len(folds_r) < 2:
                        continue

                    adx_grid_r = g_adx if use_adx else [float(adx_thresh)]

                    for fold_idx, fold_r in enumerate(folds_r):
                        ens_progress.progress(
                            min(0.999, (run_i + fold_idx / max(1, len(folds_r))) / int(n_runs)),
                            text=f"{_mode_label}: Lauf {run_i+1}/{int(n_runs)} — Fold {fold_idx+1}/{len(folds_r)} …")

                        df_is_r  = df_run[(df_run.index >= fold_r["is_start"]) & (df_run.index < fold_r["is_end"])].copy()
                        df_oos_r = df_run[(df_run.index >= fold_r["oos_start"]) & (df_run.index < fold_r["oos_end"])].copy()

                        if len(df_is_r) < 30 or len(df_oos_r) < 5:
                            continue

                        for sl_, ma_, fm_, adx_, tt_, to_ in _prod(g_sl, g_ma, g_fm, adx_grid_r, g_tt, g_to):
                            p = {**_variant_params,
                                 "sl_pct":      sl_,
                                 "ma_period":   ma_,
                                 "filter_mode": fm_,
                                 "adx_thresh":  adx_,
                                 "trail_trig":  tt_,
                                 "trail_off":   to_}
                            try:
                                tr_is_r, _ = _momi_backtest_engine(df_is_r.copy(), p)
                                if len(tr_is_r) < int(min_trades):
                                    continue
                                tr_oos_r, eq_oos_r = _momi_backtest_engine(df_oos_r.copy(), p)
                                m_oos_r = _momi_metrics(tr_oos_r, eq_oos_r)
                                key = (sl_, tt_, to_, ma_, fm_)
                                if key not in ensemble_stability:
                                    ensemble_stability[key] = []
                                ensemble_stability[key].append(m_oos_r)
                            except Exception:
                                continue

                    ens_progress.progress((run_i + 1) / int(n_runs),
                                           text=f"{_mode_label}: Lauf {run_i+1}/{int(n_runs)} abgeschlossen")

                ens_progress.progress(1.0, text=f"Ensemble ({_mode_label}) abgeschlossen ✓  ({int(n_runs)} Läufe)")

                # Auswertung
                if not ensemble_stability:
                    st.error(f"Keine Ensemble-Ergebnisse für {_mode_label} — Grid oder Zeitraum anpassen.")
                    continue

                # Buy & Hold Referenz für den gesamten Zeitraum
                bh_start_price = float(df_raw["Close"].iloc[0])
                bh_end_price   = float(df_raw["Close"].iloc[-1])
                bh_total_ret   = (bh_end_price / bh_start_price - 1) * 100
                tested_period  = f"{df_raw.index[0].date()} – {df_raw.index[-1].date()}"

                ens_rows = []
                for (sl_, tt_, to_, ma_, fm_), results in ensemble_stability.items():
                    if len(results) < 3:
                        continue
                    pfs    = [r["pf"]        for r in results if r["n"] > 0]
                    rets   = [r["total_ret"] for r in results if r["n"] > 0]
                    sharps = [r["sharpe"]    for r in results if r["n"] > 0]
                    avg_ret_list = [r.get("avg_ret", 0) for r in results if r["n"] > 0]
                    if not pfs:
                        continue
                    n_pos = sum(1 for r in rets if r > 0)
                    n_tot = len(results)
                    ens_rows.append({
                        "_sl": sl_, "_tt": tt_, "_to": to_, "_ma": ma_, "_fm": fm_,
                        "SL %":             f"{sl_:.2f}%",
                        "Trail-Trig %":     f"{tt_:.2f}%",
                        "Trail-Off %":      f"{to_:.2f}%",
                        "MA":               f"{ma_}",
                        "Filter":           fm_,
                        "Konsistenz %":     round(n_pos / n_tot * 100, 0),
                        "Ø OOS PF":         round(np.mean(pfs), 2),
                        "Ø OOS Ret %":      round(np.mean(rets), 2),
                        "Ø Profit/Trade %": round(np.mean(avg_ret_list), 3) if avg_ret_list else 0,
                        "Ø OOS Sharpe":     round(np.mean(sharps), 2),
                        "Getestete Folds":  n_tot,
                        "Zeitraum":         tested_period,
                    })

                if not ens_rows:
                    st.info(f"Zu wenig Daten für {_mode_label} Ensemble-Auswertung.")
                    continue

                df_ens = pd.DataFrame(ens_rows).sort_values(
                    ["Konsistenz %", "Ø OOS PF"], ascending=[False, False]
                ).reset_index(drop=True)

                # Ergebnisse persistent im Session-State speichern (getrennt je Fill-Modus)
                display_cols_ens = ["SL %","Trail-Trig %","Trail-Off %","MA","Filter",
                                    "Konsistenz %","Ø OOS PF","Ø OOS Ret %","Ø Profit/Trade %","Ø OOS Sharpe","Zeitraum"]
                _ens_result = {
                    "df_ens":       df_ens,
                    "bh_total_ret": bh_total_ret,
                    "tested_period": tested_period,
                    "n_runs":       int(n_runs),
                    "display_cols": display_cols_ens,
                    "fill_mode_label": _mode_label,
                }
                st.session_state[_dax_ens_cache_key(_fm)] = _ens_result
                _ens_saved, _ens_save_reason = _save_wfa_result(f"dax_ens_{_fm}", _ens_result)

                _ens_persist_note = ("☁️ dauerhaft gespeichert" if _ens_saved
                                     else f"⚠️ nur für diese Session ({_ens_save_reason})")

                if _fm != fill_mode:
                    # Nicht aktuell ausgewählter Modus: nur speichern, keine volle Anzeige (siehe Vergleichstabelle unten)
                    st.success(f"✅ {_mode_label}: {len(df_ens)} Kombinationen gespeichert · "
                               f"Top-Konsistenz {int(df_ens.iloc[0]['Konsistenz %'])}% · Ø OOS PF {df_ens.iloc[0]['Ø OOS PF']} · "
                               f"{_ens_persist_note}")
                    continue

                st.success(f"**Ensemble abgeschlossen** — {len(df_ens)} Kombinationen über {int(n_runs)} Läufe · "
                           f"Zeitraum: {tested_period} · {_ens_persist_note}")

                # ── Top-10 Tabelle ────────────────────────────────────────────
                st.markdown(f"### 🏆 Top-10 stabilste Setups über {int(n_runs)} Läufe")
                st.caption(f"Zeitraum: **{tested_period}** · Buy & Hold DAX in diesem Zeitraum: **{bh_total_ret:+.1f}%**")
                st.caption(f"📌 Getesteter Fill-Modus: **{_mode_label}** — Modus oben ändern und "
                           f"'Ensemble WFA starten' erneut klicken, um den anderen Modus zu testen.")

                top10_ens = df_ens.head(10)[display_cols_ens].copy()
                top10_ens["Ø OOS Ret %"]      = top10_ens["Ø OOS Ret %"].apply(lambda v: f"{v:.2f}%")
                top10_ens["Ø Profit/Trade %"] = top10_ens["Ø Profit/Trade %"].apply(lambda v: f"{v:.3f}%")

                def _hl_k(v):
                    if not isinstance(v, (int, float)): return ""
                    if v >= 80: return "color:#22c55e;font-weight:700"
                    if v >= 60: return "color:#f0c040"
                    return "color:#ef5350"
                st.dataframe(top10_ens.style.map(_hl_k, subset=["Konsistenz %"]),
                             use_container_width=True, hide_index=True)

                best_ens = df_ens.iloc[0]
                st.success(
                    f"**Robustestes Setup:** SL {best_ens['SL %']} · "
                    f"Trail-Trigger {best_ens['Trail-Trig %']} · Trail-Abstand {best_ens['Trail-Off %']} · "
                    f"MA {best_ens['MA']} · Filter: {best_ens['Filter']} → "
                    f"**{int(best_ens['Konsistenz %'])}% Konsistenz** · "
                    f"Ø OOS PF {best_ens['Ø OOS PF']} · Ø OOS Ret {best_ens['Ø OOS Ret %']:.2f}%"
                )

                # ── Equity Kurve vs Buy & Hold für Top-10 ────────────────────
                st.markdown("### Equity Kurve vs Buy & Hold — Top-10 Setups")
                st.caption("Strategie (orange) vs reines DAX halten (blau) über den gesamten Testzeitraum")

                bh_equity = df_raw["Close"] / df_raw["Close"].iloc[0] * 10_000

                for rank, row in df_ens.head(10).iterrows():
                    p_full = {**_variant_params,
                              "sl_pct":      row["_sl"],
                              "ma_period":   int(row["_ma"]),
                              "filter_mode": row["_fm"],
                              "trail_trig":  row["_tt"],
                              "trail_off":   row["_to"]}
                    try:
                        tr_f, eq_f = _momi_backtest_engine(df_raw.copy(), p_full)
                        m_f = _momi_metrics(tr_f, eq_f)
                    except Exception:
                        continue

                    total_ret_f = m_f["total_ret"]
                    label = (f"#{rank+1} · SL {row['SL %']} · Trail {row['Trail-Trig %']} · "
                             f"MA {row['MA']} · {row['Filter']}")

                    with st.expander(f"#{rank+1} — Rendite: {total_ret_f:+.1f}% vs Buy&Hold: {bh_total_ret:+.1f}% · {label}"):
                        # KPI-Zeile
                        k1,k2,k3,k4,k5 = st.columns(5)
                        k1.metric("Gesamt-Rendite",  f"{total_ret_f:+.1f}%")
                        k2.metric("Buy & Hold",      f"{bh_total_ret:+.1f}%")
                        k3.metric("Outperformance",  f"{total_ret_f - bh_total_ret:+.1f}%")
                        k4.metric("Trades",          m_f["n"])
                        k5.metric("Profit Factor",   f"{m_f['pf']:.2f}")

                        k6,k7,k8 = st.columns(3)
                        k6.metric("Win-Rate",        f"{m_f['wr']:.1f}%")
                        k7.metric("Max Drawdown",    f"{m_f['max_dd']:.1f}%")
                        k8.metric("Sharpe",          f"{m_f['sharpe']:.2f}")

                        # Chart
                        fig_vs = go.Figure()
                        fig_vs.add_trace(go.Scatter(
                            x=bh_equity.index, y=bh_equity.values,
                            name="Buy & Hold DAX",
                            line=dict(color="#4a9eff", width=2),
                            fill="tozeroy", fillcolor="rgba(74,158,255,0.05)"))
                        fig_vs.add_trace(go.Scatter(
                            x=eq_f.index, y=eq_f.values,
                            name="Strategie",
                            line=dict(color="#f7931a", width=2.5),
                            fill="tozeroy", fillcolor="rgba(247,147,26,0.1)"))
                        fig_vs.add_hline(y=10_000, line_color="white",
                                         line_dash="dash", line_width=1)
                        fig_vs.update_layout(
                            title=f"Setup #{rank+1}: {label}",
                            height=350, template="plotly_dark",
                            yaxis_title="Kapital (€)",
                            legend=dict(orientation="h", y=1.05),
                            margin=dict(t=50, b=20, l=60, r=20))
                        st.plotly_chart(fig_vs, use_container_width=True)

                # ── Heatmap ───────────────────────────────────────────────────
                st.markdown("### Heatmap: SL% × Trail-Trigger% → Konsistenz %")
                pivot_ens = df_ens.copy()
                pivot_ens["SL_num"] = pivot_ens["SL %"].str.replace("%","").astype(float)
                pivot_ens["TT_num"] = pivot_ens["Trail-Trig %"].str.replace("%","").astype(float)
                heat_ens = pivot_ens.pivot_table(
                    values="Konsistenz %", index="SL_num", columns="TT_num", aggfunc="mean").round(0)
                fig_heat_ens = go.Figure(go.Heatmap(
                    z=heat_ens.values,
                    x=[f"Trail {c}%" for c in heat_ens.columns],
                    y=[f"SL {r}%"    for r in heat_ens.index],
                    colorscale="RdYlGn", zmin=0, zmax=100,
                    text=heat_ens.values.round(0).astype(str),
                    texttemplate="%{text}%", showscale=True,
                    colorbar=dict(title="Konsistenz %")))
                fig_heat_ens.update_layout(
                    title=f"Ensemble-Konsistenz über {int(n_runs)} Läufe",
                    height=350, template="plotly_dark",
                    margin=dict(t=50, b=30, l=80, r=20))
                st.plotly_chart(fig_heat_ens, use_container_width=True)

                st.download_button("⬇️ Ensemble-Ergebnis als CSV",
                                   data=df_ens[display_cols_ens].to_csv(index=False).encode(),
                                   file_name="dax_ensemble_wfa.csv", mime="text/csv",
                                   key=f"dax_ens_dl_{_fm}")

            st.session_state["dax_ens_running"] = False

            # ── Vergleich beider Fill-Modi (falls beide vorhanden) ──────────
            _ens_cmp_rows = []
            for _fm2, _label2 in [("close", "Montag Close"), ("next_open", "Dienstag Open")]:
                _k2 = _dax_ens_cache_key(_fm2)
                if _k2 in st.session_state:
                    _e2 = st.session_state[_k2]
                    _best2 = _e2["df_ens"].iloc[0]
                    _ens_cmp_rows.append({
                        "Fill-Modus": _label2,
                        "Top-Konsistenz %": int(_best2["Konsistenz %"]),
                        "Ø OOS PF": _best2["Ø OOS PF"],
                        "Ø OOS Ret %": _best2["Ø OOS Ret %"],
                        "Ø OOS Sharpe": _best2["Ø OOS Sharpe"],
                    })
            if len(_ens_cmp_rows) == 2:
                st.markdown("#### Vergleich: Montag Close vs. Dienstag Open (Ensemble, jeweils bestes Setup)")
                st.dataframe(pd.DataFrame(_ens_cmp_rows), use_container_width=True, hide_index=True)

        # ── Gespeicherte Ensemble-Ergebnisse anzeigen (auch nach Reload) ──────
        _ens_cache_key = _dax_ens_cache_key(fill_mode)
        if not st.session_state.get("dax_ens_running") and _ens_cache_key in st.session_state:
            _ec = st.session_state[_ens_cache_key]
            _df_ens_c   = _ec["df_ens"]
            _bh_ret_c   = _ec["bh_total_ret"]
            _period_c   = _ec["tested_period"]
            _n_runs_c   = _ec["n_runs"]
            _dcols_c    = _ec["display_cols"]
            _fill_c     = _ec.get("fill_mode_label", "unbekannt (vor diesem Update gelaufen)")
            st.markdown("---")
            st.markdown(f"### 🏆 Top-10 Ensemble-Setups ({_n_runs_c} Läufe) — gespeichertes Ergebnis")
            st.caption(f"Zeitraum: **{_period_c}** · Buy & Hold: **{_bh_ret_c:+.1f}%**")
            st.caption(f"📌 Getesteter Fill-Modus: **{_fill_c}** — Modus oben ändern und "
                       f"'Ensemble WFA starten' erneut klicken, um den anderen Modus zu testen.")

            _t10 = _df_ens_c.head(10)[_dcols_c].copy()
            _t10["Ø OOS Ret %"]      = _t10["Ø OOS Ret %"].apply(lambda v: f"{v:.2f}%" if isinstance(v, (int,float)) else v)
            _t10["Ø Profit/Trade %"] = _t10["Ø Profit/Trade %"].apply(lambda v: f"{v:.3f}%" if isinstance(v, (int,float)) else v)

            def _hl_k2(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 80: return "color:#22c55e;font-weight:700"
                if v >= 60: return "color:#f0c040"
                return "color:#ef5350"
            st.dataframe(_t10.style.map(_hl_k2, subset=["Konsistenz %"]),
                         use_container_width=True, hide_index=True)

            _best_c = _df_ens_c.iloc[0]
            st.success(
                f"**Robustestes Setup:** SL {_best_c['SL %']} · "
                f"Trail-Trigger {_best_c['Trail-Trig %']} · Trail-Abstand {_best_c['Trail-Off %']} · "
                f"MA {_best_c['MA']} · Filter: {_best_c['Filter']} → "
                f"**{int(_best_c['Konsistenz %'])}% Konsistenz** · "
                f"Ø OOS PF {_best_c['Ø OOS PF']} · Ø OOS Ret {_best_c['Ø OOS Ret %']:.2f}%"
            )
            st.download_button("⬇️ Ensemble als CSV (gespeichert)",
                               data=_df_ens_c[_dcols_c].to_csv(index=False).encode(),
                               file_name="dax_ensemble_wfa.csv", mime="text/csv",
                               key="dax_ens_dl_cached")

    # ════════════════════════════════════════════════════════════════════════
    # MONTE CARLO — Prop Trading Challenge Simulator
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Monte Carlo — Prop Trading Challenge Simulator")
    st.markdown("""
Simuliert **1.000 mögliche Zukunften** deiner Strategie durch zufälliges Mischen (Bootstrap) der echten Trade-Ergebnisse.
Zeigt dir wie wahrscheinlich es ist, eine Prop-Firm Challenge zu bestehen — bevor du echtes Geld riskierst.
    """)

    # ── Trade-Quelle für Monte Carlo auswählen (NUR WFA/Ensemble-Ergebnisse) ──
    _dow_label = {"Montag":"Mo","Dienstag":"Di","Mittwoch":"Mi","Donnerstag":"Do","Freitag":"Fr","Samstag":"Sa","Sonntag":"So"}

    mc_fill_choice = st.radio("Fill-Modus für Monte Carlo", ["Montag Close", "Dienstag Open"],
                               key="dax_mc_fill", horizontal=True,
                               help="Unabhängig vom Fill-Modus oben in den Strategie-Parametern — lädt direkt die "
                                    "WFA/Ensemble-Ergebnisse, die für DIESEN Fill-Modus gespeichert wurden.")
    _mc_fm = "close" if mc_fill_choice == "Montag Close" else "next_open"

    _mc_wfa_cache = st.session_state.get(_dax_wfa_cache_key(_mc_fm))
    _mc_ens_cache = st.session_state.get(_dax_ens_cache_key(_mc_fm))
    _mc_wfa_best  = _dax_stability_best(_mc_wfa_cache["param_stability"]) if _mc_wfa_cache else None
    _mc_oos_trades = _mc_wfa_cache.get("oos_trades") if _mc_wfa_cache else None

    # Reine OOS-Trades NUR des robustesten Setups (exakt bekannte Parameter + keine Full-Sample-Verzerrung)
    _mc_best_oos_trades = None
    if _mc_wfa_best is not None and _mc_wfa_cache is not None:
        _psl_trades = _mc_wfa_cache.get("param_stability_trades", {}).get(_mc_wfa_best["key"])
        if _psl_trades:
            _mc_best_oos_trades = pd.concat(_psl_trades, ignore_index=True)

    _mc_source_options = []
    if _mc_best_oos_trades is not None and not _mc_best_oos_trades.empty:
        _mc_source_options.append("WFA robustestes Setup — nur seine OOS-Trades (empfohlen)")
    if _mc_wfa_best is not None:
        _mc_source_options.append("WFA robustestes Setup — Full-Sample")
    if _mc_oos_trades is not None and not _mc_oos_trades.empty:
        _mc_source_options.append("WFA kombinierte OOS-Trades (alle Folds, wechselnde Setups)")
    if _mc_ens_cache is not None and not _mc_ens_cache["df_ens"].empty:
        _mc_source_options.append("Ensemble robustestes Setup — Full-Sample")

    if not _mc_source_options:
        st.warning(f"Für **{mc_fill_choice}** liegen noch keine WFA/Ensemble-Ergebnisse vor. Bitte oben zuerst "
                   f"'🔄 WFA starten' bzw. '⚡ Ensemble WFA starten' ausführen — mit aktivierter Checkbox "
                   f"'Beide Fill-Modi automatisch testen' bekommst du beide Modi in einem Durchgang.")
        mc_trades_raw = None
    else:
        mc_source = st.radio(
            "Trade-Quelle für Monte Carlo", _mc_source_options, index=0, key="dax_mc_source",
            help="WFA robustestes Setup — nur seine OOS-Trades: EIN festes, bekanntes Setup (SL/Trail/MA/Filter), "
                 "aber nur dessen echte blinde Out-of-Sample-Trades über die Folds, in denen es getestet wurde. "
                 "Beste Kombination aus 'weiß ich, was ich trade' + 'ehrlich getestet'.\n"
                 "WFA robustestes Setup — Full-Sample: dasselbe Setup, aber über den KOMPLETTEN Zeitraum "
                 "nachbacktestet (einfacher, aber leicht optimistisch).\n"
                 "WFA kombinierte OOS-Trades: alle Folds mit ihren jeweils EIGENEN besten (ggf. unterschiedlichen) "
                 "Setups — du weißt am Ende nicht mehr genau, welche Parameter das waren.\n"
                 "Ensemble robustestes Setup: über 5 versetzte Läufe konsistentestes Setup, Full-Sample nachbacktestet.")

        _base_for_mc = {**base_params, "fill_mode": _mc_fm}

        if mc_source == "WFA robustestes Setup — nur seine OOS-Trades (empfohlen)":
            mc_trades_raw = _mc_best_oos_trades
            _mc_source_label = f"WFA robustestes Setup, nur OOS-Trades ({mc_fill_choice}) — {_mc_wfa_best['label']}"
        elif mc_source == "WFA robustestes Setup — Full-Sample":
            _p_best = {**_base_for_mc, **{k: v for k, v in _mc_wfa_best.items() if k not in ("label", "key")}}
            mc_trades_raw, _ = _momi_backtest_engine(df_raw.copy(), _p_best)
            _mc_source_label = f"WFA robustestes Setup ({mc_fill_choice}) — {_mc_wfa_best['label']}"
        elif mc_source == "WFA kombinierte OOS-Trades (alle Folds, wechselnde Setups)":
            mc_trades_raw = _mc_oos_trades
            _mc_source_label = f"WFA kombinierte OOS-Trades ({mc_fill_choice}, alle Folds, blind getestet)"
        else:
            _best_ens_row = _mc_ens_cache["df_ens"].iloc[0]
            _p_ens = {**_base_for_mc,
                      "sl_pct":      float(_best_ens_row["_sl"]),
                      "ma_period":   int(_best_ens_row["_ma"]),
                      "filter_mode": _best_ens_row["_fm"],
                      "trail_trig":  float(_best_ens_row["_tt"]),
                      "trail_off":   float(_best_ens_row["_to"])}
            mc_trades_raw, _ = _momi_backtest_engine(df_raw.copy(), _p_ens)
            _mc_source_label = (f"Ensemble robustestes Setup ({mc_fill_choice}) — "
                                f"SL {_best_ens_row['SL %']} · Trail {_best_ens_row['Trail-Trig %']}/{_best_ens_row['Trail-Off %']} · "
                                f"MA {_best_ens_row['MA']} · {_best_ens_row['Filter']} "
                                f"({int(_best_ens_row['Konsistenz %'])}% Konsistenz)")

    if mc_trades_raw is None or mc_trades_raw.empty:
        if _mc_source_options:
            st.info("Kein Backtest-Ergebnis für diese Quelle — Parameter/Zeitraum prüfen.")
    else:
        n_real = len(mc_trades_raw)
        real_rets = mc_trades_raw["PnL $"].values
        st.success(f"**{n_real} echte Trades** geladen ({_mc_source_label}) · "
                   f"Win-Rate: {(real_rets > 0).mean()*100:.1f}% · "
                   f"Ø Trade: {real_rets.mean():.2f}$ · "
                   f"Trades/Jahr: ~{n_real / max(1, (df_raw.index[-1]-df_raw.index[0]).days / 365):.0f}")
        if mc_source == "WFA robustestes Setup — nur seine OOS-Trades (empfohlen)":
            st.markdown(
                '<div style="background:#1e293b55;border:1px solid rgba(148,163,184,.25);border-radius:8px;'
                'padding:10px 16px;font-size:.85rem;color:#94a3b8;margin:8px 0 12px 0;">'
                f'📌 <b>Getestetes Setup (fest, bekannt):</b> {_mc_wfa_best["label"]} · '
                f'Fill-Modus: <b>{mc_fill_choice}</b><br>'
                '✅ Beste Kombination: EIN festes Setup (du weißt genau, welches SL/Trailing/MA), aber nur seine '
                'echten, blinden Out-of-Sample-Trades — keine Full-Sample-Rückschau-Verzerrung.'
                '</div>', unsafe_allow_html=True)
        elif mc_source == "WFA robustestes Setup — Full-Sample":
            st.markdown(
                '<div style="background:#1e293b55;border:1px solid rgba(148,163,184,.25);border-radius:8px;'
                'padding:10px 16px;font-size:.85rem;color:#94a3b8;margin:8px 0 12px 0;">'
                f'📌 <b>Getestetes Setup (WFA robustestes, Full-Sample):</b> {_mc_wfa_best["label"]} · '
                f'Fill-Modus: <b>{mc_fill_choice}</b><br>'
                '⚠️ Leicht optimistisch: dieses Setup wurde anhand seiner Performance über genau diesen Zeitraum '
                '(inkl. der OOS-Fenster) als "robustestes" ausgewählt.'
                '</div>', unsafe_allow_html=True)
        elif mc_source == "WFA kombinierte OOS-Trades (alle Folds, wechselnde Setups)":
            st.markdown(
                '<div style="background:#1e293b55;border:1px solid rgba(148,163,184,.25);border-radius:8px;'
                'padding:10px 16px;font-size:.85rem;color:#94a3b8;margin:8px 0 12px 0;">'
                '📌 <b>Getestet: echte Out-of-Sample-Trades aus allen WFA-Folds</b> — jeder Fold nutzte sein '
                '<u>eigenes</u>, nur auf dem IS-Fenster optimiertes bestes Setup (kann von Fold zu Fold variieren). '
                '⚠️ Du weißt am Ende nicht mehr genau, welche exakten Parameter das waren — nutze dafür lieber '
                '"WFA robustestes Setup — nur seine OOS-Trades".'
                '</div>', unsafe_allow_html=True)
        elif mc_source == "Ensemble robustestes Setup — Full-Sample":
            st.markdown(
                '<div style="background:#1e293b55;border:1px solid rgba(148,163,184,.25);border-radius:8px;'
                'padding:10px 16px;font-size:.85rem;color:#94a3b8;margin:8px 0 12px 0;">'
                f'📌 <b>Getestetes Setup (Ensemble robustestes, Full-Sample):</b> {_mc_source_label.split("— ")[-1]} · '
                f'Fill-Modus: <b>{mc_fill_choice}</b><br>'
                '⚠️ Leicht optimistisch, aus demselben Grund wie beim WFA-Setup — über 5 versetzte Läufe ausgewählt, '
                'dann auf demselben Gesamtzeitraum nachbacktestet.'
                '</div>', unsafe_allow_html=True)

        if n_real < 20:
            st.warning(f"⚠️ Nur {n_real} Trades — Monte Carlo Ergebnisse haben hohe Unsicherheit. "
                       f"Mind. 50 Trades empfohlen für verlässliche Aussagen.")

        st.info("ℹ️ Kein Zeitlimit — die Simulation läuft bis das Profit-Ziel erreicht **oder** "
                "die Drawdown-Grenze verletzt wird. Genau wie moderne Prop Firms.")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("**Challenge-Regeln (Prop Firm)**")
            challenge_capital    = st.number_input("Startkapital ($)", 10_000, 200_000, 100_000, step=10_000, key="dax_mc_cap")
            profit_target_p1_pct = st.number_input("Phase 1 Profit-Ziel (%)", 1.0, 20.0, 8.0, step=0.5, key="dax_mc_pt_p1",
                                                    help="FTMO Phase 1: 8% — Evaluierung")
            profit_target_p2_pct = st.number_input("Phase 2 Profit-Ziel (%)", 1.0, 20.0, 5.0, step=0.5, key="dax_mc_pt_p2",
                                                    help="FTMO Phase 2: 5% — Verification (gleiche DD-Regeln)")
            max_total_dd_pct     = st.number_input("Max. Total Drawdown (%)", 1.0, 20.0, 10.0, step=0.5, key="dax_mc_tdd",
                                                    help="FTMO: 10% — gilt für BEIDE Phasen")
            max_daily_dd_pct     = st.number_input("Max. Daily Drawdown (%)", 0.5, 10.0, 5.0, step=0.5, key="dax_mc_ddd",
                                                    help="FTMO: 5% — gilt für BEIDE Phasen")
        with mc2:
            st.markdown("**Simulation**")
            n_sims         = st.number_input("Anzahl Simulationen", 500, 5000, 1000, step=500, key="dax_mc_sims")
            max_trades_sim = st.number_input("Max. Trades pro Simulation", 10, 500, 100, step=10, key="dax_mc_maxt",
                                              help="Sicherheitsnetz: nach X Trades ohne Ergebnis gilt die Sim als 'nicht bestanden'")
            risk_per_trade = st.number_input("Risiko pro Trade (%)", 0.1, 5.0, 1.0, step=0.1, key="dax_mc_risk")
        with mc3:
            st.markdown("**Funded Account — Payout-Ziele**")
            payout1_pct = st.number_input("1. Payout-Ziel (%)", 0.5, 20.0, 1.0, step=0.5, key="dax_mc_payout1",
                                          help="Profitziel im Funded Account für den ersten Payout")
            payout2_pct = st.number_input("2. Payout-Ziel (%)", 0.5, 20.0, 2.0, step=0.5, key="dax_mc_payout2",
                                          help="Profitziel im Funded Account für den zweiten, höheren Payout")
            st.caption("Simuliert wird ab dem Kapital am Ende von Phase 2 weiter, mit denselben DD-Regeln — nur für Sims, die Phase 1+2 bestanden haben.")

        run_mc = st.button("▶ Monte Carlo starten", type="primary", key="dax_mc_run_btn")
        if run_mc:
            st.session_state["dax_mc_running"] = True
        if st.session_state.get("dax_mc_running"):
            st.session_state["dax_mc_running"] = False

            mc_progress = st.progress(0, text="Monte Carlo läuft …")

            # Trade-Returns normieren
            win_mask  = real_rets > 0
            avg_win   = real_rets[win_mask].mean()  if win_mask.any()   else 1
            avg_loss  = real_rets[~win_mask].mean() if (~win_mask).any() else -1
            norm_rets = np.where(real_rets > 0,
                                  real_rets / avg_win,
                                 -real_rets / avg_loss)

            n_sims_int = int(n_sims)
            results, all_paths_p1 = [], []

            def _run_phase(cap, target_pct, norm_r, risk_pct, dd_total, dd_daily, max_t):
                """Simuliert eine Phase; gibt (passed, fail_reason, n_trades, final_pct, path) zurück."""
                capital   = float(cap)
                peak      = capital
                day_start = capital
                path      = [capital]
                n_trades  = 0
                failed    = False
                reason    = ""
                while True:
                    nr      = norm_r[np.random.randint(len(norm_r))]
                    pnl     = capital * (risk_pct / 100) * nr
                    capital += pnl
                    peak    = max(peak, capital)
                    path.append(capital)
                    n_trades += 1
                    daily_dd = (capital - day_start) / day_start * 100
                    if daily_dd < -dd_daily:
                        failed, reason = True, "Daily DD"; break
                    day_start = capital
                    total_dd = (capital - peak) / peak * 100
                    if total_dd < -dd_total:
                        failed, reason = True, "Total DD"; break
                    profit = (capital - cap) / cap * 100
                    if profit >= target_pct:
                        break
                    if n_trades >= max_t:
                        break
                final_pct = (capital - cap) / cap * 100
                passed    = not failed and final_pct >= target_pct
                return passed, reason, n_trades, final_pct, path

            def _run_funded_phase(cap, target1_pct, target2_pct, norm_r, risk_pct, dd_total, dd_daily, max_t):
                """Simuliert den Funded Account nach bestandener Phase 2; trackt zwei Payout-Meilensteine
                entlang EINES durchgehenden Pfads (kein Reset zwischen den Zielen)."""
                capital   = float(cap)
                peak      = capital
                day_start = capital
                n_trades  = 0
                p1_ok, p1_trades = False, 0
                p2_ok, p2_trades = False, 0
                stop_target = max(target1_pct, target2_pct)
                while True:
                    nr      = norm_r[np.random.randint(len(norm_r))]
                    pnl     = capital * (risk_pct / 100) * nr
                    capital += pnl
                    peak    = max(peak, capital)
                    n_trades += 1
                    daily_dd = (capital - day_start) / day_start * 100
                    if daily_dd < -dd_daily:
                        break
                    day_start = capital
                    total_dd = (capital - peak) / peak * 100
                    if total_dd < -dd_total:
                        break
                    profit = (capital - cap) / cap * 100
                    if not p1_ok and profit >= target1_pct:
                        p1_ok, p1_trades = True, n_trades
                    if not p2_ok and profit >= target2_pct:
                        p2_ok, p2_trades = True, n_trades
                    if profit >= stop_target:
                        break
                    if n_trades >= max_t:
                        break
                return p1_ok, p1_trades, p2_ok, p2_trades

            for sim_i in range(n_sims_int):
                # ── Phase 1 ──
                p1_passed, p1_reason, p1_trades, p1_pct, p1_path = _run_phase(
                    challenge_capital, profit_target_p1_pct, norm_rets,
                    risk_per_trade, max_total_dd_pct, max_daily_dd_pct, int(max_trades_sim))

                # ── Phase 2 (nur wenn Phase 1 bestanden) ──
                p2_passed, p2_reason, p2_trades, p2_pct = False, "", 0, 0.0
                if p1_passed:
                    p2_passed, p2_reason, p2_trades, p2_pct, _ = _run_phase(
                        challenge_capital, profit_target_p2_pct, norm_rets,
                        risk_per_trade, max_total_dd_pct, max_daily_dd_pct, int(max_trades_sim))

                # ── Funded Account: Payout-Meilensteine (nur wenn Phase 2 bestanden) ──
                fp1_ok, fp1_trades, fp2_ok, fp2_trades = False, 0, False, 0
                if p2_passed:
                    funded_start_cap = challenge_capital * (1 + p2_pct / 100)
                    fp1_ok, fp1_trades, fp2_ok, fp2_trades = _run_funded_phase(
                        funded_start_cap, payout1_pct, payout2_pct, norm_rets,
                        risk_per_trade, max_total_dd_pct, max_daily_dd_pct, int(max_trades_sim))

                results.append({
                    "p1_passed":  p1_passed,
                    "p1_reason":  p1_reason,
                    "p1_trades":  p1_trades,
                    "p1_pct":     p1_pct,
                    "p2_passed":  p2_passed,
                    "p2_reason":  p2_reason,
                    "p2_trades":  p2_trades,
                    "p2_pct":     p2_pct,
                    "payout":     p1_passed and p2_passed,
                    "fp1_ok":     fp1_ok,
                    "fp1_trades": fp1_trades,
                    "fp2_ok":     fp2_ok,
                    "fp2_trades": fp2_trades,
                })
                if sim_i < 200:
                    all_paths_p1.append(p1_path)

                if sim_i % 100 == 0:
                    mc_progress.progress(sim_i / n_sims_int, text=f"Simulation {sim_i}/{n_sims_int} …")

            mc_progress.progress(1.0, text="Monte Carlo abgeschlossen ✓")

            df_res = pd.DataFrame(results)
            n_p1_pass  = df_res["p1_passed"].sum()
            n_p1_fail  = (~df_res["p1_passed"]).sum()
            n_p2_pass  = df_res["p2_passed"].sum()   # = payout = Funded-Basis
            n_p2_fail  = (df_res["p1_passed"] & ~df_res["p2_passed"]).sum()
            n_payout   = df_res["payout"].sum()
            n_funded   = n_p2_pass

            p1_pct_val   = n_p1_pass / n_sims_int * 100
            p2_cond_pct  = n_p2_pass / n_p1_pass * 100 if n_p1_pass > 0 else 0   # P(P2|P1)
            payout_pct   = n_payout  / n_sims_int * 100                            # P(P1∩P2)

            avg_t_p1 = df_res.loc[df_res["p1_passed"], "p1_trades"].mean() if n_p1_pass > 0 else 0
            avg_t_p2 = df_res.loc[df_res["p2_passed"], "p2_trades"].mean() if n_p2_pass > 0 else 0

            # ── Funded-Payout-Kennzahlen ────────────────────────────────────
            n_payout1 = df_res["fp1_ok"].sum()
            n_payout2 = df_res["fp2_ok"].sum()
            pct_payout1_funded = n_payout1 / n_funded * 100 if n_funded > 0 else 0
            pct_payout2_funded = n_payout2 / n_funded * 100 if n_funded > 0 else 0
            pct_payout1_all    = n_payout1 / n_sims_int * 100
            pct_payout2_all    = n_payout2 / n_sims_int * 100
            avg_wk_fp1 = df_res.loc[df_res["fp1_ok"], "fp1_trades"].mean() if n_payout1 > 0 else 0
            avg_wk_fp2 = df_res.loc[df_res["fp2_ok"], "fp2_trades"].mean() if n_payout2 > 0 else 0
            _full1 = df_res[df_res["fp1_ok"]]
            _full2 = df_res[df_res["fp2_ok"]]
            avg_total_wk_payout1 = (_full1["p1_trades"] + _full1["p2_trades"] + _full1["fp1_trades"]).mean() if len(_full1) > 0 else 0
            avg_total_wk_payout2 = (_full2["p1_trades"] + _full2["p2_trades"] + _full2["fp2_trades"]).mean() if len(_full2) > 0 else 0

            # ── Payout Badge ──────────────────────────────────────────────
            if payout_pct >= 40:
                bc, bt = "#22c55e", f"✅ GUTE PAYOUT-CHANCE — {payout_pct:.1f}% der Simulationen bestehen BEIDE Phasen"
            elif payout_pct >= 15:
                bc, bt = "#f0c040", f"⚠️ MÖGLICH — {payout_pct:.1f}% der Simulationen bestehen BEIDE Phasen"
            else:
                bc, bt = "#ef5350", f"❌ SCHWIERIG — nur {payout_pct:.1f}% der Simulationen erhalten einen Payout"

            st.markdown(
                f'<div style="background:{bc}22;border:2px solid {bc};border-radius:10px;'
                f'padding:16px 24px;font-weight:800;font-size:1.3rem;color:{bc};margin:16px 0;">'
                f'{bt}</div>', unsafe_allow_html=True)

            # ── KPIs ──────────────────────────────────────────────────────
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Phase 1 besteht",    f"{p1_pct_val:.1f}%",  f"{n_p1_pass}/{n_sims_int}")
            k2.metric("Phase 2 | P1 ok",    f"{p2_cond_pct:.1f}%", f"{n_p2_pass}/{n_p1_pass if n_p1_pass else '–'}",
                      help="Wahrscheinlichkeit Phase 2 zu bestehen, wenn Phase 1 schon bestanden ist")
            k3.metric("💰 Payout",           f"{payout_pct:.1f}%",  f"{n_payout}/{n_sims_int}",
                      help="P(Phase1) × P(Phase2|Phase1) — echte Auszahlungswahrscheinlichkeit")
            k4.metric("Ø Wochen Phase 1",   f"{avg_t_p1:.0f} Wo."  if avg_t_p1 > 0 else "–",
                      help="Durchschnittliche Dauer bis Phase 1 bestanden (bei bestandenen Sims)")
            k5.metric("Ø Wochen Phase 2",   f"{avg_t_p2:.0f} Wo."  if avg_t_p2 > 0 else "–",
                      help="Durchschnittliche Dauer bis Phase 2 bestanden (bei bestandenen Sims)")
            k6.metric("Ø Gesamt",           f"{avg_t_p1+avg_t_p2:.0f} Wo." if (avg_t_p1+avg_t_p2) > 0 else "–",
                      help="Gesamtdauer bis erster Payout")

            p1_dd_fail = (df_res["p1_reason"] == "Daily DD").sum()
            p1_td_fail = (df_res["p1_reason"] == "Total DD").sum()
            p2_dd_fail = (df_res["p2_reason"] == "Daily DD").sum()
            p2_td_fail = (df_res["p2_reason"] == "Total DD").sum()
            st.caption(
                f"Phase 1 — Daily DD: {p1_dd_fail}x · Total DD: {p1_td_fail}x  |  "
                f"Phase 2 — Daily DD: {p2_dd_fail}x · Total DD: {p2_td_fail}x")

            # ── Kreisdiagramm ─────────────────────────────────────────────
            pie_labels = ["❌ Phase 1 gescheitert", "⚠️ Phase 2 gescheitert", "💰 Payout erhalten"]
            pie_values = [n_p1_fail, n_p2_fail, n_payout]
            pie_colors = ["#ef5350", "#f0c040", "#22c55e"]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker=dict(colors=pie_colors, line=dict(color="#1a1a2e", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13),
                hole=0.4,
                pull=[0, 0, 0.07],
            ))
            fig_pie.update_layout(
                title=f"Challenge-Ergebnis aus {n_sims_int} Simulationen",
                height=380, template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=60, b=60, l=20, r=20),
                annotations=[dict(
                    text=f"<b>{payout_pct:.1f}%</b><br>Payout",
                    x=0.5, y=0.5, font_size=16,
                    font_color="#22c55e", showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.info(
                f"**Warum ist Phase 2 leichter als Phase 1?**  "
                f"Niedrigeres Ziel ({profit_target_p2_pct}% statt {profit_target_p1_pct}%) bei gleichen DD-Grenzen. "
                f"Daher: P(Phase 2 | Phase 1 bestanden) = **{p2_cond_pct:.1f}%** > P(Phase 1) = **{p1_pct_val:.1f}%**. "
                f"Gesamtchance: {p1_pct_val:.1f}% × {p2_cond_pct:.1f}% = **{payout_pct:.1f}% Payout**."
            )

            # ── Phase-1-Pfad-Chart ────────────────────────────────────────
            if all_paths_p1:
                st.subheader("Phase 1 — Simulierte Verläufe")
                fig_mc = go.Figure()
                for path in all_paths_p1:
                    ok = (path[-1] - challenge_capital) / challenge_capital * 100 >= profit_target_p1_pct
                    fig_mc.add_trace(go.Scatter(
                        y=path, mode="lines",
                        line=dict(color="#22c55e" if ok else "#ef5350", width=0.5),
                        opacity=0.12, showlegend=False))
                max_len   = max(len(p) for p in all_paths_p1)
                padded    = [p + [p[-1]] * (max_len - len(p)) for p in all_paths_p1]
                mean_path = np.mean(padded, axis=0)
                fig_mc.add_trace(go.Scatter(y=mean_path, mode="lines",
                                             line=dict(color="white", width=2.5, dash="dash"), name="Ø Pfad"))
                fig_mc.add_hline(y=challenge_capital * (1 + profit_target_p1_pct/100),
                                  line_color="#22c55e", line_dash="dot", line_width=2,
                                  annotation_text=f"Phase 1 Ziel +{profit_target_p1_pct}%", annotation_position="right")
                fig_mc.add_hline(y=challenge_capital * (1 - max_total_dd_pct/100),
                                  line_color="#ef5350", line_dash="dot", line_width=2,
                                  annotation_text=f"Ruin -{max_total_dd_pct}%", annotation_position="right")
                fig_mc.add_hline(y=challenge_capital, line_color="white", line_dash="dash", line_width=1)
                fig_mc.update_layout(
                    title=f"Phase 1 — {n_sims_int} Simulationen · Grün = bestanden · Rot = gescheitert",
                    height=420, template="plotly_dark",
                    yaxis_title="Kapital ($)", xaxis_title="Trade #",
                    margin=dict(t=50, b=30, l=70, r=130))
                st.plotly_chart(fig_mc, use_container_width=True)

            # ── Trades-Histogramme ────────────────────────────────────────
            if n_p1_pass > 0:
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    fig_tw1 = go.Figure(go.Histogram(
                        x=df_res.loc[df_res["p1_passed"], "p1_trades"], nbinsx=20,
                        marker_color="#42a5f5"))
                    fig_tw1.update_layout(
                        title=f"Phase 1: Wochen bis Ziel (Ø {avg_t_p1:.0f} Wo.)",
                        height=220, template="plotly_dark",
                        xaxis_title="Trades/Wochen", yaxis_title="Anzahl",
                        margin=dict(t=40, b=30, l=50, r=10))
                    st.plotly_chart(fig_tw1, use_container_width=True)
                with col_h2:
                    if n_p2_pass > 0:
                        fig_tw2 = go.Figure(go.Histogram(
                            x=df_res.loc[df_res["p2_passed"], "p2_trades"], nbinsx=20,
                            marker_color="#22c55e"))
                        fig_tw2.update_layout(
                            title=f"Phase 2: Wochen bis Ziel (Ø {avg_t_p2:.0f} Wo.)",
                            height=220, template="plotly_dark",
                            xaxis_title="Trades/Wochen", yaxis_title="Anzahl",
                            margin=dict(t=40, b=30, l=50, r=10))
                        st.plotly_chart(fig_tw2, use_container_width=True)

            # ════════════════════════════════════════════════════════════════
            # FUNDED ACCOUNT — PAYOUT-WAHRSCHEINLICHKEIT (1% / 2%)
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("Funded Account — Payout-Wahrscheinlichkeit")
            st.caption(f"Simuliert wird ab dem Kapital am Ende von Phase 2 weiter (gleiche DD-Regeln) — "
                       f"Basis: {n_funded}/{n_sims_int} Sims, die Phase 1 + Phase 2 bestanden haben (Funded Account).")

            fp_c1, fp_c2, fp_c3, fp_c4 = st.columns(4)
            fp_c1.metric(f"{payout1_pct:.1f}% Payout | Funded", f"{pct_payout1_funded:.1f}%",
                         f"{n_payout1}/{n_funded if n_funded else '\u2013'}",
                         help="Anteil der Funded-Sims, die im Funded Account das erste Payout-Ziel erreichen")
            fp_c2.metric(f"{payout1_pct:.1f}% Payout | alle Sims", f"{pct_payout1_all:.1f}%", f"{n_payout1}/{n_sims_int}",
                         help="Gleiche Kennzahl, aber bezogen auf ALLE Simulationen (inkl. an Phase 1/2 gescheiterte)")
            fp_c3.metric(f"{payout2_pct:.1f}% Payout | Funded", f"{pct_payout2_funded:.1f}%",
                         f"{n_payout2}/{n_funded if n_funded else '\u2013'}",
                         help="Anteil der Funded-Sims, die im Funded Account das zweite, höhere Payout-Ziel erreichen")
            fp_c4.metric(f"{payout2_pct:.1f}% Payout | alle Sims", f"{pct_payout2_all:.1f}%", f"{n_payout2}/{n_sims_int}",
                         help="Gleiche Kennzahl, aber bezogen auf ALLE Simulationen (inkl. an Phase 1/2 gescheiterte)")

            fp_w1, fp_w2 = st.columns(2)
            fp_w1.metric(f"Ø Wochen im Funded Account bis {payout1_pct:.1f}% Payout",
                         f"{avg_wk_fp1:.0f} Wo." if avg_wk_fp1 > 0 else "\u2013")
            fp_w2.metric(f"Ø Wochen im Funded Account bis {payout2_pct:.1f}% Payout",
                         f"{avg_wk_fp2:.0f} Wo." if avg_wk_fp2 > 0 else "\u2013")

            st.markdown("#### Zusammenfassung — Gesamtdauer bis zum Payout")
            st.caption("Phase 1 + Phase 2 + Funded Account zusammengerechnet, nur für Sims die den jeweiligen Payout tatsächlich erreichen.")
            sum_c1, sum_c2 = st.columns(2)
            sum_c1.metric(f"Ø Wochen gesamt bis {payout1_pct:.1f}% Payout",
                          f"{avg_total_wk_payout1:.0f} Wo." if avg_total_wk_payout1 > 0 else "\u2013")
            sum_c2.metric(f"Ø Wochen gesamt bis {payout2_pct:.1f}% Payout",
                          f"{avg_total_wk_payout2:.0f} Wo." if avg_total_wk_payout2 > 0 else "\u2013")

            if n_payout1 > 0 or n_payout2 > 0:
                fp_h1, fp_h2 = st.columns(2)
                with fp_h1:
                    if n_payout1 > 0:
                        fig_fp1 = go.Figure(go.Histogram(
                            x=df_res.loc[df_res["fp1_ok"], "fp1_trades"], nbinsx=20,
                            marker_color="#ffa726"))
                        fig_fp1.update_layout(
                            title=f"{payout1_pct:.1f}% Payout: Wochen im Funded Account (Ø {avg_wk_fp1:.0f} Wo.)",
                            height=220, template="plotly_dark",
                            xaxis_title="Trades/Wochen", yaxis_title="Anzahl",
                            margin=dict(t=40, b=30, l=50, r=10))
                        st.plotly_chart(fig_fp1, use_container_width=True)
                with fp_h2:
                    if n_payout2 > 0:
                        fig_fp2 = go.Figure(go.Histogram(
                            x=df_res.loc[df_res["fp2_ok"], "fp2_trades"], nbinsx=20,
                            marker_color="#ab47bc"))
                        fig_fp2.update_layout(
                            title=f"{payout2_pct:.1f}% Payout: Wochen im Funded Account (Ø {avg_wk_fp2:.0f} Wo.)",
                            height=220, template="plotly_dark",
                            xaxis_title="Trades/Wochen", yaxis_title="Anzahl",
                            margin=dict(t=40, b=30, l=50, r=10))
                        st.plotly_chart(fig_fp2, use_container_width=True)

            if n_real < 30:
                st.warning(f"⚠️ Nur {n_real} echte Trades — Monte Carlo Ergebnisse haben hohe Unsicherheit. "
                           f"Mind. 50 Trades für verlässliche Aussagen.")


def render_pdh_pdl_strategie() -> None:
    """PDH/PDL Proximity Reversal Strategy — Python-Backtest."""
    import datetime as _dt

    st.header("PDH/PDL Proximity Reversal Strategie")
    st.caption("Long & Short · Previous Day High/Low Zonen · ATR-basierter Stop · RR-Ratio")

    with st.sidebar:
        st.markdown("---")
        st.subheader("PDH/PDL: Einstellungen")
        pdp_symbol = st.text_input("Symbol (Yahoo)", "NQ=F", key="pdp_sym")
        pdp_start  = st.date_input("Von", _dt.date(2024, 1, 1), key="pdp_start")
        pdp_end    = st.date_input("Bis", _dt.date.today(),     key="pdp_end")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Proximity Zone**")
        zone_mode  = st.selectbox("Zonen-Modus", ["ATR", "Prozent"], key="pdp_zm")
        atr_mult   = st.number_input("ATR-Multiplikator", 0.0, 2.0, 0.1, step=0.05, key="pdp_am")
        pct_zone   = st.number_input("Zonen-Größe %", 0.0, 2.0, 0.1, step=0.05, key="pdp_pz")
    with col2:
        st.markdown("**Entry-Logik**")
        use_close  = st.checkbox("Trigger auf Close (statt H/L)", False, key="pdp_uc")
        long_on    = st.checkbox("Long-Setups", True, key="pdp_lo")
        short_on   = st.checkbox("Short-Setups", True, key="pdp_so")
    with col3:
        st.markdown("**Risk**")
        sl_buf     = st.number_input("SL-Puffer (Ticks)", 0.0, 20.0, 0.0, key="pdp_slb")
        rr_ratio   = st.number_input("Risk-Reward-Ratio", 0.5, 10.0, 2.0, step=0.1, key="pdp_rr")

    if not st.button("▶ Backtest starten", key="pdp_run"):
        st.info("Parameter einstellen und Backtest starten.")
        return

    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance fehlt: `pip install yfinance`")
        return

    with st.spinner("Daten laden …"):
        # M5-Daten (yfinance: max 60 Tage für 5m)
        df_m5 = yf.download(pdp_symbol, start=str(pdp_start), end=str(pdp_end),
                            interval="5m", auto_adjust=True, progress=False)
        df_d  = yf.download(pdp_symbol, start=str(pdp_start), end=str(pdp_end),
                            interval="1d", auto_adjust=True, progress=False)

    if df_m5.empty or df_d.empty:
        st.error("Keine Daten — Symbol und Zeitraum prüfen. Hinweis: yfinance liefert 5m-Daten nur für die letzten ~60 Tage.")
        return

    for _d in [df_m5, df_d]:
        if isinstance(_d.columns, pd.MultiIndex):
            _d.columns = _d.columns.get_level_values(0)
    df_m5.index = pd.to_datetime(df_m5.index).tz_localize(None)
    df_d.index  = pd.to_datetime(df_d.index).tz_localize(None)

    # ── Daily ATR + PDH/PDL ───────────────────────────────────────────────────
    dd = df_d.copy()
    tr_d = pd.concat([dd["High"] - dd["Low"],
                      (dd["High"] - dd["Close"].shift()).abs(),
                      (dd["Low"]  - dd["Close"].shift()).abs()], axis=1).max(axis=1)
    dd["ATR14"] = tr_d.rolling(14).mean()
    dd["PDH"]   = dd["High"].shift(1)
    dd["PDL"]   = dd["Low"].shift(1)
    dd = dd[["PDH", "PDL", "ATR14"]].dropna()

    m5 = df_m5.copy()
    m5["date"] = m5.index.normalize()
    dd.index   = pd.to_datetime(dd.index).normalize()
    m5 = m5.merge(dd, left_on="date", right_index=True, how="left")
    m5[["PDH","PDL","ATR14"]] = m5[["PDH","PDL","ATR14"]].ffill()

    # ── Zonen berechnen ───────────────────────────────────────────────────────
    if zone_mode == "ATR":
        zone_h = m5["ATR14"] * atr_mult
        zone_l = m5["ATR14"] * atr_mult
    else:
        zone_h = m5["PDH"] * (pct_zone / 100)
        zone_l = m5["PDL"] * (pct_zone / 100)

    m5["hzt"] = m5["PDH"] + zone_h
    m5["hzb"] = m5["PDH"] - zone_h
    m5["lzt"] = m5["PDL"] + zone_l
    m5["lzb"] = m5["PDL"] - zone_l

    m5["in_high"] = (m5["Close"] >= m5["hzb"]) & (m5["Close"] <= m5["hzt"])
    m5["in_low"]  = (m5["Close"] >= m5["lzb"]) & (m5["Close"] <= m5["lzt"])

    if use_close:
        m5["short_trig"] = m5["Close"] < m5["Close"].shift(1)
        m5["long_trig"]  = m5["Close"] > m5["Close"].shift(1)
    else:
        m5["short_trig"] = m5["Close"] < m5["Low"].shift(1)
        m5["long_trig"]  = m5["Close"] > m5["High"].shift(1)

    m5["long_sig"]  = long_on  & m5["in_low"]  & m5["long_trig"]  & m5["PDH"].notna()
    m5["short_sig"] = short_on & m5["in_high"] & m5["short_trig"] & m5["PDH"].notna()

    # ── Backtest ─────────────────────────────────────────────────────────────
    capital, position, entry_price = 10_000.0, 0, 0.0
    sl_price, tp_price, qty = 0.0, 0.0, 0.0
    trades, equity_curve = [], []
    tick = 0.25 if float(m5["Close"].iloc[-1]) > 1000 else 0.01

    for ts, row in m5.iterrows():
        c, h, l = float(row["Close"]), float(row["High"]), float(row["Low"])

        if position == 1:
            if l <= sl_price:
                pnl = (sl_price - entry_price) * qty
                capital += pnl
                trades.append({"Zeit": ts, "Dir": "Long", "Entry": entry_price, "Exit": sl_price, "PnL $": round(pnl,2), "Grund": "SL"})
                position = 0
            elif h >= tp_price:
                pnl = (tp_price - entry_price) * qty
                capital += pnl
                trades.append({"Zeit": ts, "Dir": "Long", "Entry": entry_price, "Exit": tp_price, "PnL $": round(pnl,2), "Grund": "TP"})
                position = 0

        elif position == -1:
            if h >= sl_price:
                pnl = (entry_price - sl_price) * qty
                capital += pnl
                trades.append({"Zeit": ts, "Dir": "Short", "Entry": entry_price, "Exit": sl_price, "PnL $": round(pnl,2), "Grund": "SL"})
                position = 0
            elif l <= tp_price:
                pnl = (entry_price - tp_price) * qty
                capital += pnl
                trades.append({"Zeit": ts, "Dir": "Short", "Entry": entry_price, "Exit": tp_price, "PnL $": round(pnl,2), "Grund": "TP"})
                position = 0

        if position == 0:
            buf = sl_buf * tick
            if row["long_sig"]:
                entry_price = c
                sl_price    = float(row["Low"]) - buf
                risk        = entry_price - sl_price
                if risk > 0:
                    tp_price = entry_price + risk * rr_ratio
                    qty      = (capital * 0.10) / entry_price
                    position = 1
            elif row["short_sig"]:
                entry_price = c
                sl_price    = float(row["High"]) + buf
                risk        = sl_price - entry_price
                if risk > 0:
                    tp_price = entry_price - risk * rr_ratio
                    qty      = (capital * 0.10) / entry_price
                    position = -1

        equity_curve.append(capital)

    m5["Equity"] = equity_curve
    df_trades = pd.DataFrame(trades)

    if df_trades.empty:
        st.warning("Keine Trades — Zone oder Zeitraum anpassen.")
        return

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_ret = (m5["Equity"].iloc[-1] - 10_000) / 10_000 * 100
    n   = len(df_trades)
    wr  = (df_trades["PnL $"] > 0).sum() / n * 100
    gp  = df_trades.loc[df_trades["PnL $"] > 0, "PnL $"].sum()
    gl  = df_trades.loc[df_trades["PnL $"] <= 0, "PnL $"].abs().sum()
    pf  = gp / gl if gl > 0 else float("inf")
    eq  = m5["Equity"]
    dd_s = (eq - eq.cummax()) / eq.cummax() * 100
    max_dd = dd_s.min()
    ps  = df_trades["PnL $"]
    sharpe = (ps.mean() / ps.std() * np.sqrt(252)) if ps.std() > 0 else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Gesamtrendite", f"{total_ret:.2f}%")
    k2.metric("Trades",        n)
    k3.metric("Win-Rate",      f"{wr:.1f}%")
    k4.metric("Profit Factor", f"{pf:.2f}")
    k5.metric("Sharpe",        f"{sharpe:.2f}")
    st.metric("Max. Drawdown", f"{max_dd:.2f}%")

    # ── Equity Curve ─────────────────────────────────────────────────────────
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=m5.index, y=m5["Equity"], line=dict(color="#00d4aa", width=2), name="Equity"))
    fig_eq.update_layout(title="Equity Curve", height=280, template="plotly_dark", margin=dict(t=40,b=20))
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── PDH/PDL Zonen-Chart ───────────────────────────────────────────────────
    fig_c = go.Figure()
    fig_c.add_trace(go.Candlestick(x=m5.index, open=m5["Open"], high=m5["High"],
                                   low=m5["Low"], close=m5["Close"], name="M5",
                                   increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    fig_c.add_trace(go.Scatter(x=m5.index, y=m5["PDH"], name="PDH", line=dict(color="red",   width=1, dash="dash")))
    fig_c.add_trace(go.Scatter(x=m5.index, y=m5["PDL"], name="PDL", line=dict(color="green", width=1, dash="dash")))
    fig_c.update_layout(title="M5 Preis + PDH/PDL", height=400, template="plotly_dark",
                        xaxis_rangeslider_visible=False, margin=dict(t=40,b=20))
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Trades ───────────────────────────────────────────────────────────────
    st.subheader("Trade-Liste")
    st.dataframe(df_trades.style.map(lambda v: "color: #26a69a" if isinstance(v,(int,float)) and v > 0
                                     else ("color: #ef5350" if isinstance(v,(int,float)) and v < 0 else ""),
                                     subset=["PnL $"]), use_container_width=True)

    ec = df_trades["Grund"].value_counts()
    fig_p = go.Figure(go.Pie(labels=ec.index, values=ec.values, hole=0.4,
                             marker_colors=["#ef5350","#26a69a","#ffa726"]))
    fig_p.update_layout(title="Exit-Gründe", height=260, template="plotly_dark", margin=dict(t=40,b=10))
    st.plotly_chart(fig_p, use_container_width=True)


def render_muster_analyse() -> None:
    import datetime as _dt2
    from pathlib import Path as _Path

    _MT5_DIR2 = _Path(__file__).parent / "data" / "mt5"
    _syms2 = sorted([f.stem.upper() for f in _MT5_DIR2.glob("*.csv")]) if _MT5_DIR2.exists() else []

    # Wenn Analyse bereits berechnet → Detailansicht zeigen
    if "muster_analyse_detail" in st.session_state:
        _det = st.session_state["muster_analyse_detail"]
        st.session_state["muster_detail"]     = _det["detail"]
        st.session_state["muster_dataframes"] = _det["dfs"]
        _render_muster_detail()
        return

    st.markdown("## 🔍 Muster Analyse — Manuelle Eingabe")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        symbol    = st.selectbox("Symbol", _syms2) if _syms2 else st.text_input("Symbol")
        direction = st.radio("Richtung", ["Long", "Short"], horizontal=True)
    with col2:
        entry_day   = st.number_input("Entry Tag",   1, 31, 23)
        entry_month = st.number_input("Entry Monat", 1, 12, 6)
    with col3:
        exit_day   = st.number_input("Exit Tag",   1, 31, 4)
        exit_month = st.number_input("Exit Monat", 1, 12, 7)

    if not st.button("Analysieren", type="primary"):
        st.info("Symbol und Datum eingeben, dann 'Analysieren' klicken.")
        return

    csv_path = _MT5_DIR2 / f"{symbol.upper()}.csv"
    if not csv_path.exists():
        st.error(f"Keine Daten für {symbol}."); return

    df_m = normalize_ohlc(pd.read_csv(csv_path))
    if df_m.empty:
        st.error("Keine gültigen OHLC-Daten."); return

    try:
        entry_doy_m = _dt2.date(2000, int(entry_month), int(entry_day)).timetuple().tm_yday
        exit_doy_m  = _dt2.date(2000, int(exit_month),  int(exit_day)).timetuple().tm_yday
    except ValueError as e:
        st.error(f"Ungültiges Datum: {e}"); return

    # ATR berechnen
    df_m["_atr"] = (df_m["high"] - df_m["low"]).ewm(span=14).mean()

    # year_data aufbauen
    _yd_map: dict = {}
    for _yr, _g in df_m.groupby(df_m.index.year):
        _doys = _g.index.dayofyear.values.astype(int)
        _si   = np.argsort(_doys)
        _yd_map[int(_yr)] = {
            "doys":   _doys[_si],
            "closes": _g["close"].values.astype(float)[_si],
            "highs":  _g["high"].values.astype(float)[_si],
            "lows":   _g["low"].values.astype(float)[_si],
            "atrs":   _g["_atr"].values.astype(float)[_si],
            "dates":  _g.index[_si],
        }

    _max_yr   = df_m.index.year.max()
    _end_yr   = _max_yr - 1 if _max_yr >= pd.Timestamp.now().year else _max_yr
    _data_start = df_m.index.year.min()
    _y20 = _end_yr - 20 + 1; _y15 = _end_yr - 15 + 1
    _y10 = _end_yr - 10 + 1; _y5  = _end_yr - 5  + 1
    _has_20j = _data_start <= _y20; _has_15j = _data_start <= _y15
    _has_10j = _data_start <= _y10; _has_5j  = _data_start <= _y5
    _dir_str = direction.lower()

    # Alle Trades sammeln (alle verfügbaren Jahre)
    _trades: list = []
    for _yr in sorted(_yd_map.keys()):
        if _yr > _end_yr: continue
        _yd = _yd_map[_yr]
        _ei = int(np.searchsorted(_yd["doys"], entry_doy_m))
        _xi = int(np.searchsorted(_yd["doys"], exit_doy_m))
        if _ei >= len(_yd["doys"]) or _xi >= len(_yd["doys"]) or _xi <= _ei: continue
        _ep = _yd["closes"][_ei]; _xp = _yd["closes"][_xi]
        _sl = _yd["lows"][_ei+1:_xi+1].min()  if _xi > _ei else _ep
        _sh = _yd["highs"][_ei+1:_xi+1].max() if _xi > _ei else _ep
        _trades.append({
            "yr": _yr, "ep": _ep,
            "long_ret":  (_xp - _ep) / _ep,
            "short_ret": (_ep - _xp) / _ep,
            "long_dd":   (_sl - _ep) / _ep,
            "short_dd":  (_ep - _sh) / _ep,
            "atr": _yd["atrs"][_ei],
            "td": _xi - _ei,
        })

    def _ma_stats(yr_start: int) -> dict | None:
        _sub = [t for t in _trades if t["yr"] >= yr_start]
        if len(_sub) < 3: return None
        _rets = np.array([t[f"{_dir_str}_ret"] for t in _sub])
        _dds  = np.array([t[f"{_dir_str}_dd"]  for t in _sub])
        _nt = len(_rets); _wr = float((_rets > 0).sum() / _nt)
        _ar = _rets.mean(); _sr = _rets.std(ddof=1) if _nt > 1 else 0.0
        return {"wr": _wr, "nt": _nt, "avg_ret": _ar, "std_ret": _sr,
                "avg_dd": _dds.mean(), "max_dd": _dds.min()}

    _s10 = _ma_stats(_y10) or {}
    _wr5  = round((_ma_stats(_y5)  or {}).get("wr", np.nan) * 100, 1) if _has_5j  else np.nan
    _wr10 = round((_ma_stats(_y10) or {}).get("wr", np.nan) * 100, 1) if _has_10j else np.nan
    _wr15 = round((_ma_stats(_y15) or {}).get("wr", np.nan) * 100, 1) if _has_15j else np.nan
    _wr20_raw = (_ma_stats(_y20) or {}).get("wr", np.nan) if _has_20j else np.nan
    if np.isnan(_wr20_raw): _wr20_raw = (_ma_stats(_data_start) or {}).get("wr", np.nan)
    _wr20 = round(_wr20_raw * 100, 1) if not np.isnan(_wr20_raw) else np.nan

    _avg_ret  = _s10.get("avg_ret", np.nan)
    _std_ret  = _s10.get("std_ret", 0.0)
    _nt10     = _s10.get("nt", 0)
    _avg_td   = float(np.mean([t["td"] for t in _trades if t["yr"] >= _y10])) if _nt10 else 10
    _sharpe   = round(_avg_ret / _std_ret * np.sqrt(252 / max(_avg_td, 1)), 2) if _std_ret > 1e-10 else np.nan
    _sqn      = round(_avg_ret / _std_ret * np.sqrt(_nt10) * 100, 2) if _std_ret > 1e-10 else np.nan
    _avg_atr  = float(np.mean([t["atr"] / t["ep"] for t in _trades if t["yr"] >= _y10 and t["ep"] > 0]) * 100) if _nt10 else 0.0

    # Robustheit
    _prim_trades = [t for t in _trades if t["yr"] >= _y10]
    _atr_eff = _avg_atr > 0 and (abs(_avg_ret) / (_avg_atr / 100)) >= 0.4
    _rob_wins = _rob_total = 0
    for _off in list(range(-7, -2)) + list(range(3, 8)):
        _alt_e = entry_doy_m + _off
        if _alt_e < 1 or _alt_e > 365: continue
        _alt_rets = []
        for _yr, _yd in _yd_map.items():
            if _yr < _y10 or _yr > _end_yr: continue
            _ei2 = int(np.searchsorted(_yd["doys"], _alt_e))
            _xi2 = int(np.searchsorted(_yd["doys"], exit_doy_m))
            if _ei2 >= len(_yd["doys"]) or _xi2 >= len(_yd["doys"]) or _xi2 <= _ei2: continue
            _ep2, _xp2 = _yd["closes"][_ei2], _yd["closes"][_xi2]
            _alt_rets.append((_xp2 - _ep2)/_ep2 if _dir_str == "long" else (_ep2 - _xp2)/_ep2)
        if len(_alt_rets) >= 3:
            _rob_total += 1
            if (np.array(_alt_rets) > 0).mean() >= 0.6: _rob_wins += 1
    if _rob_total == 0:
        _robustheit = "—"
    else:
        _rr = _rob_wins / _rob_total
        if _rr >= 0.80 and _atr_eff:  _robustheit = "🟢 Stark"
        elif _rr >= 0.80:              _robustheit = "✅ Robust"
        elif _rr >= 0.60:              _robustheit = "✅ Robust"
        elif _rr >= 0.40:              _robustheit = "⚠️ Sensitiv"
        else:                          _robustheit = "❌ Fragil"

    # Stern-Rating (1–5)
    _wr10_f = (_s10.get("wr", 0) or 0) * 100
    _score = 0
    if _wr10_f >= 80: _score += 2
    elif _wr10_f >= 70: _score += 1
    if not np.isnan(_sharpe) and _sharpe >= 1.5: _score += 1
    if _avg_atr > 0 and abs(_avg_ret) / (_avg_atr / 100) >= 0.6: _score += 1
    if _robustheit in ("🟢 Stark",): _score += 1
    _stars = max(1, min(5, _score + 1))

    _mon_names = ['','Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez']
    _row = {
        "Symbol":          symbol.upper(),
        "Richtung":        direction,
        "Entry":           f"{int(entry_day):02d}. {_mon_names[int(entry_month)]}",
        "Exit":            f"{int(exit_day):02d}. {_mon_names[int(exit_month)]}",
        "Haltedauer (TD)": int(_avg_td),
        "_entry_doy":      entry_doy_m,
        "_exit_doy":       exit_doy_m,
        "WR 5J %":  _wr5,
        "WR 10J %": _wr10,
        "WR 15J %": _wr15,
        "WR 20J %": _wr20,
        "Ø Profit %": round(_avg_ret * 100, 2) if not np.isnan(_avg_ret) else np.nan,
        "Sharpe":    round(_sharpe, 2) if not np.isnan(_sharpe) else np.nan,
        "SQN":       round(_sqn, 2)    if not np.isnan(_sqn)    else np.nan,
        "Ø ATR %":  round(_avg_atr, 3),
        "Robustheit": _robustheit,
        "⭐ Rating":  _stars,
    }
    st.session_state["muster_analyse_detail"] = {
        "detail": {"row": _row, "symbol": symbol.upper(), "lookback": 10, "from_analyse": True},
        "dfs":    {symbol.upper(): df_m},
    }
    st.rerun()


# ── Extra: Makro-Kalender + Sentiment ────────────────────────────────────────
# Hinweis: alles hier ist eine vereinfachte redaktionelle Faustregel, kein
# validiertes Handelsmodell. Ausschliesslich offizielle/kostenlose Quellen:
# BLS (Release-Kalender), Fed/EZB/BoE (Zinstermine, von Hand gepflegt),
# Alpha Vantage (Actual-Werte), yfinance (Kurse), CFTC Socrata (COT-Historie).

EXTRA_ASSETS = {
    "Gold":       "GC=F",
    "DXY":        "DX-Y.NYB",
    "Nasdaq 100": "^NDX",
    "S&P 500":    "^GSPC",
    "EUR":        "EURUSD=X",
    "GBP":        "GBPUSD=X",
    "AUD":        "AUDUSD=X",
    "NZD":        "NZDUSD=X",
    "CAD":        "CAD=X",
    "JPY":        "JPY=X",
}

# USD-quotierte Ticker (USDCAD/USDJPY): ein Kursanstieg bedeutet eine SCHWAECHERE
# Fremdwaehrung, daher Momentum/Return fuer diese Assets invertieren (wie in der
# Waehrungsmatrix, CURRENCY_SYMBOLS weiter unten).
INVERT_MOMENTUM_ASSETS = {"CAD", "JPY"}

# Reuse der bestehenden COT-Watchlist (COT_WATCHLIST, Zeile ~4088) statt Duplikat.
ASSET_TO_COT_LABEL = {
    "Gold":       "Gold",
    "DXY":        "DXY",
    "Nasdaq 100": "NQ Futures",
    "S&P 500":    "S&P500 Futures",
    "EUR":        "EURO Futures",
    "GBP":        "Pfund Futures",
    "AUD":        "AUD Futures",
    "NZD":        "NZD Futures",
    "CAD":        "CANADA Futures",
    "JPY":        "YEN Futures",
}

# Zinstermine von Hand gepflegt, Quelle: federalreserve.gov/monetarypolicy/fomccalendars.htm,
# ecb.europa.eu/press/calendars/mgcgc, bankofengland.co.uk/monetary-policy/upcoming-mpc-dates
# Muss jaehrlich manuell aktualisiert werden.
CENTRAL_BANK_EVENTS = [
    {"date": date(2026, 1, 27), "time": "—",      "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 1, 28), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 3, 17), "time": "—",      "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 3, 18), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 4, 28), "time": "—",      "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 4, 29), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 6, 16), "time": "—",      "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 6, 17), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 7, 28), "time": "—",      "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 7, 29), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 9, 15), "time": "—",      "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 9, 16), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 10, 27), "time": "—",       "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 10, 28), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 12, 8), "time": "—",       "currency": "USD", "event": "FOMC Zinsentscheid (Tag 1)", "impact": "High"},
    {"date": date(2026, 12, 9), "time": "14:00 ET", "currency": "USD", "event": "FOMC Zinsentscheid", "impact": "High"},
    {"date": date(2026, 1, 25), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 3, 19), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 4, 30), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 6, 11), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 7, 23), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 9, 10), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 10, 29), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 12, 17), "time": "14:15 CET", "currency": "EUR", "event": "EZB Zinsentscheid", "impact": "High"},
    {"date": date(2026, 2, 5), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 3, 19), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 4, 30), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 6, 18), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 7, 30), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 9, 17), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 11, 5), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
    {"date": date(2026, 12, 17), "time": "12:00 UK", "currency": "GBP", "event": "BoE Zinsentscheid", "impact": "High"},
]

BLS_IMPACT = {
    "employment situation":                  "High",
    "consumer price index":                  "High",
    "producer price index":                  "Medium",
    "job openings and labor turnover":       "Medium",
    "employment cost index":                 "Medium",
    "import/export price":                   "Low",
    "real earnings":                         "Low",
    "productivity":                          "Low",
}

# Actual-vs-Previous Richtungslogik pro Makro-Indikator und Asset.
# +1 = Ueberraschung nach oben wirkt bullish fuer das Asset, -1 = bearish.
# Vereinfachte redaktionelle Faustregel, kein validiertes Modell.
DIRECTION_MATRIX = {
    "CPI":                {"Gold": +1, "DXY": +1, "Nasdaq 100": -1, "S&P 500": -1, "EUR": -1, "GBP": -1, "AUD": -1, "NZD": -1, "CAD": -1, "JPY": -1},
    "NONFARM_PAYROLL":    {"Gold": -1, "DXY": +1, "Nasdaq 100": -1, "S&P 500": -1, "EUR": -1, "GBP": -1, "AUD": -1, "NZD": -1, "CAD": -1, "JPY": -1},
    "UNEMPLOYMENT":       {"Gold": +1, "DXY": -1, "Nasdaq 100": +1, "S&P 500": +1, "EUR": +1, "GBP": +1, "AUD": +1, "NZD": +1, "CAD": +1, "JPY": +1},
    "FEDERAL_FUNDS_RATE": {"Gold": -1, "DXY": +1, "Nasdaq 100": -1, "S&P 500": -1, "EUR": -1, "GBP": -1, "AUD": -1, "NZD": -1, "CAD": -1, "JPY": -1},
    "RETAIL_SALES":       {"Gold": -1, "DXY": +1, "Nasdaq 100": +1, "S&P 500": +1, "EUR": -1, "GBP": -1, "AUD": -1, "NZD": -1, "CAD": -1, "JPY": -1},
    "REAL_GDP":           {"Gold": -1, "DXY": +1, "Nasdaq 100": +1, "S&P 500": +1, "EUR": -1, "GBP": -1, "AUD": -1, "NZD": -1, "CAD": -1, "JPY": -1},
}
# Fremdwaehrungs-Richtung = -DXY-Richtung (vereinfachte Annahme: Dollarstaerke wirkt
# symmetrisch gegenlaeufig auf USD-Kreuze). Redaktionelle Faustregel, kein Modell.


def classify_bls_impact(release_name: str) -> str:
    lowered = release_name.lower()
    for key, impact in BLS_IMPACT.items():
        if key in lowered:
            return impact
    return "Low"


@st.cache_data(ttl=6 * 60 * 60)
def fetch_bls_calendar(year: int, month: int) -> pd.DataFrame:
    try:
        from io import StringIO
        import requests

        url = f"https://www.bls.gov/schedule/{year}/{month:02d}_sched.htm"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        raw = None
        for table in tables:
            cols = [str(c).strip().lower() for c in table.columns]
            if any("date" in c for c in cols) and any("release" in c for c in cols):
                raw = table
                break
        if raw is None:
            return pd.DataFrame()

        raw.columns = [str(c).strip() for c in raw.columns]
        date_col = next((c for c in raw.columns if "date" in c.lower()), None)
        time_col = next((c for c in raw.columns if "time" in c.lower()), None)
        release_col = next((c for c in raw.columns if "release" in c.lower()), None)
        if date_col is None or release_col is None:
            return pd.DataFrame()

        out = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col].astype(str), errors="coerce").dt.date,
            "time": raw[time_col].astype(str) if time_col else "8:30 AM",
            "event": raw[release_col].astype(str).str.strip(),
        })
        out = out.dropna(subset=["date", "event"])
        out["currency"] = "USD"
        out["impact"] = out["event"].apply(classify_bls_impact)
        return out[["date", "time", "currency", "event", "impact"]]
    except Exception:
        return pd.DataFrame()


# BLS blockt automatisierte Requests von vielen Cloud-/Rechenzentrums-IPs (auch Streamlit
# Cloud) fast immer -- fetch_bls_calendar() liefert dort praktisch nie Daten. FRED (St. Louis
# Fed) stellt dieselben offiziellen BLS-Release-Termine ueber eine echte, fuer Bots gedachte
# API bereit und wird nicht geblockt. Braucht einen kostenlosen FRED_API_KEY.
FRED_RELEASE_IDS = {
    10: ("Consumer Price Index (CPI)", "High"),
    50: ("Employment Situation (NFP)", "High"),
}


def get_fred_api_key() -> str:
    import os

    try:
        key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("FRED_API_KEY", "")


@st.cache_data(ttl=12 * 60 * 60)
def fetch_fred_release_dates(release_id: int, event_name: str, impact: str, api_key: str, start: date, end: date) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    try:
        import requests

        params = {
            "release_id": release_id,
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": start.isoformat(),
            "realtime_end": end.isoformat(),
            "include_release_dates_with_no_data": "true",
        }
        response = requests.get("https://api.stlouisfed.org/fred/release/dates", params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        rows = [
            {"date": pd.to_datetime(d["date"]).date(), "time": "8:30 AM ET", "currency": "USD", "event": event_name, "impact": impact}
            for d in data.get("release_dates", [])
        ]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def get_economic_calendar(start: date, end: date) -> pd.DataFrame:
    months = sorted({(start.year, start.month), (end.year, end.month)})
    bls_frames = [fetch_bls_calendar(y, m) for y, m in months]
    bls = pd.concat(bls_frames, ignore_index=True) if any(not f.empty for f in bls_frames) else pd.DataFrame()

    fred_key = get_fred_api_key()
    fred_frames = [
        fetch_fred_release_dates(rid, name, impact, fred_key, start, end)
        for rid, (name, impact) in FRED_RELEASE_IDS.items()
    ]
    fred = pd.concat(fred_frames, ignore_index=True) if any(not f.empty for f in fred_frames) else pd.DataFrame()

    cb = pd.DataFrame(CENTRAL_BANK_EVENTS)
    frames = [f for f in (bls, fred, cb) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "time", "currency", "event", "impact"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[(combined["date"] >= start) & (combined["date"] <= end)]
    combined = combined.drop_duplicates(subset=["date", "event"])
    impact_rank = {"High": 0, "Medium": 1, "Low": 2}
    combined["_rank"] = combined["impact"].map(impact_rank).fillna(3)
    return combined.sort_values(["_rank", "date"]).drop(columns="_rank").reset_index(drop=True)


def get_alpha_vantage_key() -> str:
    import os

    try:
        key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")


ALPHA_VANTAGE_FUNCTIONS = {
    "CPI": "CPI",
    "NONFARM_PAYROLL": "NONFARM_PAYROLL",
    "UNEMPLOYMENT": "UNEMPLOYMENT",
    "FEDERAL_FUNDS_RATE": "FEDERAL_FUNDS_RATE",
    "RETAIL_SALES": "RETAIL_SALES",
    "REAL_GDP": "REAL_GDP",
}


@st.cache_data(ttl=6 * 60 * 60)
def fetch_alpha_vantage_indicator(function: str, api_key: str) -> pd.DataFrame:
    # Kein try/except um den Request: Alpha Vantage antwortet bei Rate-Limit/Fehlern mit
    # HTTP 200 und einem "Note"/"Information"/"Error Message"-Feld statt "data" -- das wuerde
    # sonst als "keine Daten" fuer die vollen 6h TTL eingefroren (gleiche Falle wie zuvor bei
    # Gemini). Stattdessen hier explizit erkennen und raisen, damit st.cache_data den
    # Fehlerzustand NICHT cacht und der naechste Aufruf es erneut versucht.
    if not api_key:
        return pd.DataFrame()
    import requests

    url = f"https://www.alphavantage.co/query?function={function}&apikey={api_key}"
    response = requests.get(url, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if "Note" in payload or "Information" in payload or "Error Message" in payload:
        raise RuntimeError(payload.get("Note") or payload.get("Information") or payload.get("Error Message"))
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date")


def macro_surprise_pct(df: pd.DataFrame) -> float | None:
    """Actual (letzter Wert) vs. Previous (vorletzter Wert) in %. Kein Konsensus/Forecast verfuegbar."""
    if df is None or len(df) < 2:
        return None
    latest = float(df["value"].iloc[-1])
    previous = float(df["value"].iloc[-2])
    if previous == 0:
        return None
    return (latest - previous) / abs(previous) * 100


def compute_news_score(asset: str, api_key: str) -> tuple[float, pd.DataFrame]:
    rows = []
    contributions = []
    for indicator, function in ALPHA_VANTAGE_FUNCTIONS.items():
        try:
            df = fetch_alpha_vantage_indicator(function, api_key)
        except Exception:
            df = pd.DataFrame()
        surprise = macro_surprise_pct(df)
        direction = DIRECTION_MATRIX.get(indicator, {}).get(asset, 0)
        actual = float(df["value"].iloc[-1]) if surprise is not None else None
        previous = float(df["value"].iloc[-2]) if surprise is not None else None
        contribution = None
        if surprise is not None:
            contribution = float(np.clip(surprise / 10.0, -1, 1)) * direction
            contributions.append(contribution)
        rows.append({
            "Indikator": indicator,
            "Actual": actual,
            "Previous": previous,
            "Ueberraschung %": round(surprise, 2) if surprise is not None else None,
            "Richtung": direction,
            "Beitrag": round(contribution, 3) if contribution is not None else None,
        })
    news_score = float(np.clip(np.mean(contributions), -1, 1)) if contributions else 0.0
    return news_score, pd.DataFrame(rows)


COT_SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


@st.cache_data(ttl=24 * 60 * 60)
def fetch_cot_history(market_name: str, weeks: int = 5) -> pd.DataFrame:
    try:
        import requests

        params = {
            "$where": f"market_and_exchange_names='{market_name}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": str(weeks),
        }
        response = requests.get(COT_SOCRATA_URL, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
        df["long"] = pd.to_numeric(df.get("noncomm_positions_long_all"), errors="coerce")
        df["short"] = pd.to_numeric(df.get("noncomm_positions_short_all"), errors="coerce")
        df["net_noncomm"] = df["long"] - df["short"]
        return df.dropna(subset=["report_date", "net_noncomm"]).sort_values("report_date")
    except Exception:
        return pd.DataFrame()


def cot_bias_score(market_name: str | None) -> dict:
    """Score in [-1, 1]: Vorzeichen Netto-Position (60%) + Vorzeichen 4-Wochen-Trend (40%)."""
    if not market_name:
        return {"score": 0.0, "net": None, "trend": None, "available": False}
    hist = fetch_cot_history(market_name, weeks=5)
    if hist.empty:
        return {"score": 0.0, "net": None, "trend": None, "available": False}
    net_latest = float(hist["net_noncomm"].iloc[-1])
    net_sign = float(np.sign(net_latest))
    trend = float(hist["net_noncomm"].iloc[-1] - hist["net_noncomm"].iloc[0]) if len(hist) >= 2 else 0.0
    trend_sign = float(np.sign(trend))
    score = 0.6 * net_sign + 0.4 * trend_sign
    return {"score": score, "net": net_latest, "trend": trend, "available": True}


@st.cache_data(ttl=24 * 60 * 60)
def find_cot_socrata_market(query: str) -> str | None:
    """Sucht den exakten CFTC-Marktnamen direkt in der Socrata-API (nicht ueber deafut.txt,
    da diese von manchen Cloud-IPs geblockt wird und dann die gesamte COT-Aufloesung blockiert)."""
    try:
        import requests

        params = {
            "$select": "market_and_exchange_names",
            "$where": f"upper(market_and_exchange_names) like '%{query.upper()}%'",
            "$group": "market_and_exchange_names",
            "$limit": "10",
        }
        response = requests.get(COT_SOCRATA_URL, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        names = [row["market_and_exchange_names"] for row in data]
        exact = next((n for n in names if n.split(" - ")[0].strip().upper() == query.upper()), None)
        return exact or names[0]
    except Exception:
        return None


def resolve_cot_market_name(cot_label: str) -> str | None:
    queries = next((q for label, q in COT_WATCHLIST if label == cot_label), None)
    if not queries:
        return None
    for query in queries:
        market = find_cot_socrata_market(query)
        if market:
            return market
    return None


@st.cache_data(ttl=15 * 60)
def fetch_extra_price_data(symbol: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        data = yf.download(symbol, period="3mo", interval="1d", progress=False, auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        return data.reset_index()
    except Exception:
        return pd.DataFrame()


def price_returns(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {"day_pct": None, "week_pct": None}
    closes = df["Close"].dropna()
    day_pct = float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100) if len(closes) >= 2 else None
    week_pct = float((closes.iloc[-1] / closes.iloc[-6] - 1) * 100) if len(closes) >= 6 else None
    return {"day_pct": day_pct, "week_pct": week_pct}


def momentum_score(pct_return: float | None, scale: float) -> float:
    if pct_return is None:
        return 0.0
    return float(np.clip(pct_return / scale, -1, 1))


def sentiment_label(score: float) -> str:
    if score < -0.5:
        return "Bearish"
    if score < -0.15:
        return "Leicht Bearish"
    if score < 0.15:
        return "Neutral"
    if score < 0.5:
        return "Leicht Bullish"
    return "Bullish"


def compute_asset_sentiment(asset: str, symbol: str, api_key: str, timeframe: str) -> dict:
    price_df = fetch_extra_price_data(symbol)
    returns = price_returns(price_df)
    invert = asset in INVERT_MOMENTUM_ASSETS
    day_pct = returns["day_pct"]
    week_pct = returns["week_pct"]
    if invert:
        day_pct = -day_pct if day_pct is not None else None
        week_pct = -week_pct if week_pct is not None else None

    is_today = timeframe == "Heute"
    pct = day_pct if is_today else week_pct
    mom_score = momentum_score(pct, 1.5 if is_today else 3.0)

    news_score, news_detail = compute_news_score(asset, api_key)

    cot_label = ASSET_TO_COT_LABEL.get(asset, "")
    cot_market = resolve_cot_market_name(cot_label)
    cot_stats = cot_bias_score(cot_market)

    weights = (0.50, 0.35, 0.15) if is_today else (0.35, 0.25, 0.40)
    total = weights[0] * mom_score + weights[1] * news_score + weights[2] * cot_stats["score"]
    total = float(np.clip(total, -1, 1))

    return {
        "score": total,
        "label": sentiment_label(total),
        "momentum_score": mom_score,
        "news_score": news_score,
        "cot_score": cot_stats["score"],
        "cot_available": cot_stats["available"],
        "day_pct": day_pct,
        "week_pct": week_pct,
        "news_detail": news_detail,
        "price_df": price_df,
    }


def _news_detail_row_style(row: pd.Series) -> list[str]:
    styles = [""] * len(row)
    actual = row.get("Actual")
    previous = row.get("Previous")
    if pd.notna(actual):
        styles[row.index.get_loc("Actual")] = "color: #f8fafc; font-weight: 700;"
    if pd.notna(actual) and pd.notna(previous):
        idx_prev = row.index.get_loc("Previous")
        if previous < actual:
            styles[idx_prev] = "color: #f87171; font-weight: 600;"
        elif previous > actual:
            styles[idx_prev] = "color: #4ade80; font-weight: 600;"
    return styles


def _impact_color(impact: str) -> str:
    return {"High": "background-color: rgba(239,68,68,.25)", "Medium": "background-color: rgba(249,115,22,.20)", "Low": "background-color: rgba(148,163,184,.18)"}.get(impact, "")


# ── Multi-Timeframe Waehrungsmatrix ──────────────────────────────────────────
# Short (15m) / Mid (4h): reines Preis-Momentum auf Intraday-Kursen (yfinance).
# Long (Struktur): bestehender Momentum+COT-Ansatz (kein News-Anteil, da die
# DIRECTION_MATRIX nur fuer Gold/DXY/Nasdaq/S&P definiert ist).
# Vereinfachte redaktionelle Faustregel, kein validiertes Modell.

CURRENCY_MATRIX_ASSETS = ["CAD", "EUR", "GBP", "AUD", "DXY", "NZD", "CHF", "JPY"]

# yfinance-Ticker sind USD-Kreuze; invert=True heisst: der Ticker notiert USD je Einheit
# Fremdwaehrung (z.B. USDCAD), ein Anstieg bedeutet also eine SCHWAECHERE Fremdwaehrung.
CURRENCY_SYMBOLS = {
    "CAD": {"symbol": "CAD=X",     "invert": True},
    "EUR": {"symbol": "EURUSD=X",  "invert": False},
    "GBP": {"symbol": "GBPUSD=X",  "invert": False},
    "AUD": {"symbol": "AUDUSD=X",  "invert": False},
    "DXY": {"symbol": "DX-Y.NYB",  "invert": False},
    "NZD": {"symbol": "NZDUSD=X",  "invert": False},
    "CHF": {"symbol": "CHF=X",     "invert": True},
    "JPY": {"symbol": "JPY=X",     "invert": True},
}

# Reuse der bestehenden COT_WATCHLIST-Labels (Zeile ~4088) statt Duplikat.
CURRENCY_TO_COT_LABEL = {
    "CAD": "CANADA Futures",
    "EUR": "EURO Futures",
    "GBP": "Pfund Futures",
    "AUD": "AUD Futures",
    "DXY": "DXY",
    "NZD": "NZD Futures",
    "CHF": "CHF Futures",
    "JPY": "YEN Futures",
}


@st.cache_data(ttl=15 * 60)
def fetch_intraday_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data = data.reset_index()
        data = data.rename(columns={data.columns[0]: "Date"})
        return data
    except Exception:
        return pd.DataFrame()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    try:
        indexed = df.set_index("Date")
        out = indexed.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        return out.reset_index()
    except Exception:
        return pd.DataFrame()


def intraday_momentum_score(df: pd.DataFrame, lookback_bars: int, scale_pct: float, invert: bool) -> float | None:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    closes = df["Close"].dropna()
    if len(closes) <= lookback_bars:
        return None
    pct = float((closes.iloc[-1] / closes.iloc[-1 - lookback_bars] - 1) * 100)
    if invert:
        pct = -pct
    return float(np.clip(pct / scale_pct, -1, 1))


def matrix_percent_label(score: float) -> tuple[float, str]:
    percent = float(np.clip((score + 1) / 2 * 100, 0, 100))
    if percent >= 60:
        label = "BULLISH"
    elif percent <= 40:
        label = "BEARISH"
    else:
        label = "NEUTRAL"
    return percent, label


def compute_currency_matrix_row(currency: str) -> dict:
    meta = CURRENCY_SYMBOLS[currency]
    symbol, invert = meta["symbol"], meta["invert"]

    m15 = fetch_intraday_data(symbol, "15m", "5d")
    short_score = intraday_momentum_score(m15, lookback_bars=8, scale_pct=0.3, invert=invert)

    h1 = fetch_intraday_data(symbol, "60m", "60d")
    h4 = resample_ohlc(h1, "4h")
    mid_score = intraday_momentum_score(h4, lookback_bars=5, scale_pct=1.0, invert=invert)

    daily = fetch_extra_price_data(symbol)
    week_pct = price_returns(daily)["week_pct"]
    mom_daily = momentum_score(week_pct, 3.0)
    if invert:
        mom_daily = -mom_daily

    cot_market = resolve_cot_market_name(CURRENCY_TO_COT_LABEL.get(currency, ""))
    cot_stats = cot_bias_score(cot_market)
    long_score = None if week_pct is None and not cot_stats["available"] else float(np.clip(0.6 * mom_daily + 0.4 * cot_stats["score"], -1, 1))

    return {"Short (15m)": short_score, "Mid (4h)": mid_score, "Long (Struktur)": long_score}


def _matrix_cell_color(value: str) -> str:
    if "BULLISH" in value:
        return "background-color: rgba(34,197,94,.35); color: white"
    if "BEARISH" in value:
        return "background-color: rgba(239,68,68,.35); color: white"
    if "NEUTRAL" in value:
        return "background-color: rgba(148,163,184,.20)"
    return ""


def render_currency_matrix_section() -> None:
    st.subheader("💱 Multi-Timeframe Waehrungsmatrix")
    st.caption(
        "Short (15m) und Mid (4h): reines Preis-Momentum auf Intraday-Kursen (yfinance, kann bei "
        "Feiertagen/Datenluecken 'n/a' zeigen). Long (Struktur): Momentum (60%, wochenbasiert) + "
        "CFTC-COT-Score (40%). Kein Makro-News-Anteil in dieser Matrix. Vereinfachte Heuristik, "
        "kein validiertes Modell."
    )
    rows = []
    for currency in CURRENCY_MATRIX_ASSETS:
        scores = compute_currency_matrix_row(currency)
        row = {"Asset": currency}
        for col_label, score in scores.items():
            if score is None:
                row[col_label] = "n/a"
            else:
                percent, label = matrix_percent_label(score)
                row[col_label] = f"{percent:.0f}% {label}"
        rows.append(row)

    matrix_df = pd.DataFrame(rows).set_index("Asset")
    cols_to_style = list(matrix_df.columns)
    st.dataframe(matrix_df.style.map(_matrix_cell_color, subset=cols_to_style), use_container_width=True)


def render_extra_makro_sentiment() -> None:
    st.markdown("## 🌐 Extra: Makro-Kalender & Sentiment")
    st.caption(
        "Quellen: CPI-/NFP-Termine ueber FRED-API (zuverlaessig, braucht FRED_API_KEY) mit "
        "BLS-Live-Kalender als Zusatzversuch (wird von Cloud-IPs oft geblockt), FOMC/EZB/BoE-Termine "
        "(fest hinterlegt, jaehrlich von Hand aktualisiert), Alpha Vantage (nur Actual, kein "
        "Forecast/Konsensus — verwendet wird Actual vs. Previous als Naeherung), yfinance (Kurse), "
        "CFTC COT (Netto-Position + 4-Wochen-Trend)."
    )

    timeframe = st.radio("Zeitraum", ["Heute", "Diese Woche"], horizontal=True)
    today = date.today()
    if timeframe == "Heute":
        start, end = today, today
    else:
        start = today - pd.Timedelta(days=today.weekday())
        start = start if isinstance(start, date) else start.date()
        end = start + pd.Timedelta(days=6)
        end = end if isinstance(end, date) else end.date()

    api_key = get_alpha_vantage_key()
    if not api_key:
        st.info("Kein ALPHA_VANTAGE_API_KEY in st.secrets/.env gefunden — News-Score wird neutral (0) gesetzt, bis ein Key hinterlegt ist.")

    # ── Abschnitt 1: Wirtschaftskalender ─────────────────────────────────────
    st.subheader("📅 Wirtschaftskalender")
    if not get_fred_api_key():
        st.info(
            "Kein FRED_API_KEY hinterlegt — CPI-/NFP-Termine kommen dann nur ueber den "
            "BLS-Live-Kalender, der von Cloud-IPs (auch Streamlit Cloud) meist geblockt wird. "
            "Kostenloser Key: fred.stlouisfed.org/docs/api/api_key.html"
        )
    calendar_df = get_economic_calendar(start, end)
    if calendar_df.empty:
        st.warning("Keine Kalenderdaten verfuegbar (BLS ggf. gerade nicht erreichbar) oder keine Termine im gewaehlten Zeitraum.")
    else:
        display_df = calendar_df.rename(columns={
            "date": "Datum", "time": "Uhrzeit", "currency": "Waehrung", "event": "Event", "impact": "Impact",
        })
        st.dataframe(
            display_df.style.map(_impact_color, subset=["Impact"]),
            use_container_width=True, hide_index=True,
        )

    # ── Abschnitt 2: Sentiment-Kacheln ───────────────────────────────────────
    st.subheader("🧭 Sentiment pro Asset")
    st.caption(
        "Momentum: yfinance-Kurse, alle 15 Min. neu geladen — 'Heute' = Return letzter Handelstag "
        "vs. Vortag, 'Diese Woche' = Return letzte 5 Handelstage. COT: CFTC-Socrata-API, "
        "wochenaktuell (Cache 24h). News: Alpha Vantage, Cache 6h (siehe Hinweis oben)."
    )

    results = {}
    assets_list = list(EXTRA_ASSETS.items())
    for row_start in range(0, len(assets_list), 5):
        row_assets = assets_list[row_start:row_start + 5]
        cols = st.columns(len(row_assets))
        for col, (asset, symbol) in zip(cols, row_assets):
            result = compute_asset_sentiment(asset, symbol, api_key, timeframe)
            results[asset] = result
            with col:
                ret_label = "Tagesreturn" if timeframe == "Heute" else "Wochenreturn"
                ret_value = result["day_pct"] if timeframe == "Heute" else result["week_pct"]
                st.metric(
                    asset,
                    result["label"],
                    f"{ret_value:+.2f}% ({ret_label})" if ret_value is not None else "n/a",
                )
                st.caption(f"Score: {result['score']:+.2f}")

    # ── Abschnitt 3: Detailaufschluesselung ──────────────────────────────────
    st.subheader("🔍 Detailaufschluesselung")
    for asset, result in results.items():
        with st.expander(f"{asset} — {result['label']} ({result['score']:+.2f})"):
            sub_cols = st.columns(3)
            sub_cols[0].metric("Momentum-Score", f"{result['momentum_score']:+.2f}")
            sub_cols[1].metric("COT-Score", f"{result['cot_score']:+.2f}" if result["cot_available"] else "n/a")
            sub_cols[2].metric("News-Score", f"{result['news_score']:+.2f}")
            if not result["cot_available"]:
                st.caption("COT-Historie fuer dieses Asset gerade nicht verfuegbar.")
            st.markdown("**Makro-Ueberraschungen (Actual vs. Previous)**")
            st.caption("Actual = weiss. Previous rot = frueherer Wert niedriger als Actual, gruen = frueherer Wert hoeher als Actual.")
            if api_key:
                st.dataframe(
                    result["news_detail"].style.apply(_news_detail_row_style, axis=1),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("Kein Alpha-Vantage-Key hinterlegt — Makro-Ueberraschungen nicht verfuegbar.")
            st.caption("Retail-Sentiment: nicht verfuegbar (kein kostenloser, ToS-konformer Anbieter bekannt). Gewicht = 0.")

    # ── Abschnitt 4: Candlestick-Chart ───────────────────────────────────────
    st.subheader("📈 Kursverlauf (3 Monate)")
    chart_asset = st.selectbox("Asset", list(EXTRA_ASSETS.keys()), key="extra_chart_asset")
    chart_df = results[chart_asset]["price_df"]
    if chart_df.empty:
        st.warning(f"Keine Kursdaten fuer {chart_asset} verfuegbar.")
    else:
        fig = go.Figure(go.Candlestick(
            x=chart_df["Date"], open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"], name=chart_asset,
        ))
        fig.update_layout(title=f"{chart_asset} — 3 Monate", height=420, template="plotly_dark",
                           xaxis_rangeslider_visible=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Abschnitt 5: CFTC COT Positionierung (bestehendes COT-Modul) ─────────
    st.subheader("📊 CFTC COT Positionierung (Detail)")
    auto_match_extra_cot = st.checkbox("Auto-match an oben gewaehltes Asset", value=False, key="extra_auto_match_cot")
    st.caption(
        "Standardmaessig frei waehlbar (Dropdown direkt unten: NQ, EURO, CANADA, YEN, CHF, Pfund, "
        "AUD, NZD Futures, Silver, Copper, Platinum, ...). Haekchen aktivieren, um den COT-Markt "
        "stattdessen automatisch an das oben gewaehlte Chart-Asset zu koppeln."
    )
    render_cot_panel(auto_match_extra_cot, chart_asset, EXTRA_ASSETS[chart_asset])

    # ── Abschnitt 6: Waehrungsmatrix ──────────────────────────────────────────
    render_currency_matrix_section()

    # ── Abschnitt 7: KI Marktanalyse (bestehendes Gemini-Modul) ──────────────
    render_ki_analyse(chart_asset, EXTRA_ASSETS[chart_asset])


# ── Extra: COT Commercials vs. Non-Commercials Edge-Analyse ──────────────────
# Prueft je FX-Future-Paar, ob die Positionierung der Commercials (Hedger) oder
# der Non-Commercials (grosse Spekulanten) historisch die hoehere Aussagekraft
# fuer die nachfolgende Kursbewegung hatte. Nutzt die komplette Legacy-COT-
# Historie (nicht nur die letzten 5 Wochen wie fetch_cot_history oben), daher
# eigene Fetch-Funktionen statt Wiederverwendung von fetch_cot_history/
# cot_bias_score -- COT_SOCRATA_URL (Konstante oben) wird geteilt.

COT_EDGE_CONTRACTS = {
    "EUR/USD (CME Euro-FX-Future, Proxy)": {"cftc_name": "EURO FX - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6E=F"},
    "GBP/USD (CME British-Pound-Future, Proxy)": {"cftc_name": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6B=F"},
    "USD/JPY (CME Yen-Future, Proxy)": {"cftc_name": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6J=F"},
    "USD/CHF (CME Franken-Future, Proxy)": {"cftc_name": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6S=F"},
    "USD/CAD (CME Kanada-Dollar-Future, Proxy)": {"cftc_name": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6C=F"},
    "AUD/USD (CME Australien-Dollar-Future, Proxy)": {"cftc_name": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6A=F"},
    "NZD/USD (CME Neuseeland-Dollar-Future, Proxy)": {"cftc_name": "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6N=F"},
    "USD/MXN (CME Peso-Future, Proxy)": {"cftc_name": "MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE", "yahoo_ticker": "6M=F"},
    "USD Index (ICE US Dollar Index, Proxy)": {"cftc_name": "USD INDEX - ICE FUTURES U.S.", "yahoo_ticker": "DX-Y.NYB"},
}


@st.cache_data(ttl=24 * 60 * 60)
def _fetch_cot_edge_history(cftc_name: str, start_year: int) -> pd.DataFrame:
    """Laedt die komplette Legacy-COT-Historie (Comm + Non-Comm, normiert auf % Open Interest)."""
    try:
        import requests

        params = {
            "$where": f"market_and_exchange_names = '{cftc_name}' "
                      f"AND report_date_as_yyyy_mm_dd >= '{start_year}-01-01T00:00:00.000'",
            "$select": "report_date_as_yyyy_mm_dd,comm_positions_long_all,"
                       "comm_positions_short_all,noncomm_positions_long_all,"
                       "noncomm_positions_short_all,open_interest_all",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 5000,
        }
        response = requests.get(COT_SOCRATA_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None).astype("datetime64[ns]")
        numeric_cols = [
            "comm_positions_long_all", "comm_positions_short_all",
            "noncomm_positions_long_all", "noncomm_positions_short_all",
            "open_interest_all",
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["net_comm"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
        df["net_noncomm"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["net_comm_pct"] = df["net_comm"] / df["open_interest_all"]
        df["net_noncomm_pct"] = df["net_noncomm"] / df["open_interest_all"]

        return df[["report_date", "net_comm_pct", "net_noncomm_pct", "open_interest_all"]].sort_values("report_date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=24 * 60 * 60)
def _fetch_cot_edge_price(yahoo_ticker: str, start_year: int) -> pd.DataFrame:
    try:
        import yfinance as yf

        px = yf.download(yahoo_ticker, start=f"{start_year}-01-01", auto_adjust=True, progress=False)
        if px.empty:
            return pd.DataFrame()
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px = px[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "close"})
        px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).astype("datetime64[ns]")
        return px
    except Exception:
        return pd.DataFrame()


@dataclass
class CotEdgeResult:
    pair: str
    n_obs: int
    corr_comm: float
    corr_noncomm: float
    hitrate_comm_level: float
    hitrate_noncomm_level: float
    hitrate_comm_momentum: float
    hitrate_noncomm_momentum: float
    winner_level: str
    winner_momentum: str
    n_extreme: int
    hitrate_comm_extreme: float
    hitrate_noncomm_extreme: float
    winner_extreme: str


def _analyze_cot_edge_pair(cot_df: pd.DataFrame, price_df: pd.DataFrame, pair_name: str,
                            horizon_weeks: int, zscore_window: int = 52,
                            extreme_zscore: float = 1.0):
    """
    Fuer jeden COT-Report-Termin:
      - Signal-Kurs = naechster Handelstag ab report_date + 3 Kalendertage (~Freitag Report-Tag)
      - Forward Return = Kurs `horizon_weeks` COT-Termine spaeter / Signal-Kurs - 1
      - Level-Signal = Vorzeichen des rollierenden z-Score der Netto-Position (% OI)
      - Momentum-Signal = Vorzeichen der Wochenveraenderung der Netto-Position
      - Trefferquote = Anteil der Faelle, in denen Signal-Vorzeichen == Vorzeichen des Forward Return
      - Divergenz-Test: Trefferquote nur fuer Wochen mit |z-Score| >= extreme_zscore
    """
    if cot_df.empty or price_df.empty:
        return None

    df = cot_df.copy().sort_values("report_date").reset_index(drop=True)
    price_df = price_df.sort_values("date").reset_index(drop=True)
    df["signal_date"] = df["report_date"] + pd.Timedelta(days=3)
    # signal_date muss exakt datetime64[ns] sein (siehe astype in den fetch-Funktionen oben) --
    # pandas 3.x lehnt merge_asof zwischen z.B. datetime64[us] und datetime64[s] als
    # "incompatible merge keys" ab, obwohl beide Seiten sortierte, NaT-freie Datumsspalten sind.
    df = pd.merge_asof(
        df.sort_values("signal_date"), price_df.rename(columns={"date": "signal_date"}),
        on="signal_date", direction="forward",
    ).rename(columns={"close": "signal_price"})
    df = df.sort_values("report_date").reset_index(drop=True)

    df["future_price"] = df["signal_price"].shift(-horizon_weeks)
    df["fwd_return"] = df["future_price"] / df["signal_price"] - 1

    for col in ["net_comm_pct", "net_noncomm_pct"]:
        roll_mean = df[col].rolling(zscore_window, min_periods=13).mean()
        roll_std = df[col].rolling(zscore_window, min_periods=13).std()
        df[f"{col}_z"] = (df[col] - roll_mean) / roll_std

    df["net_comm_pct_chg"] = df["net_comm_pct"].diff()
    df["net_noncomm_pct_chg"] = df["net_noncomm_pct"].diff()

    df = df.dropna(subset=["fwd_return", "net_comm_pct_z", "net_noncomm_pct_z",
                            "net_comm_pct_chg", "net_noncomm_pct_chg"])
    if len(df) < 30:
        return None

    ret_sign = np.sign(df["fwd_return"])

    def hit_rate(signal: pd.Series) -> float:
        sig_sign = np.sign(signal)
        valid = sig_sign != 0
        return float((sig_sign[valid] == ret_sign[valid]).mean() * 100)

    hitrate_comm_level = hit_rate(df["net_comm_pct_z"])
    hitrate_noncomm_level = hit_rate(df["net_noncomm_pct_z"])
    hitrate_comm_mom = hit_rate(df["net_comm_pct_chg"])
    hitrate_noncomm_mom = hit_rate(df["net_noncomm_pct_chg"])

    corr_comm = float(df["net_comm_pct_z"].corr(df["fwd_return"]))
    corr_noncomm = float(df["net_noncomm_pct_z"].corr(df["fwd_return"]))

    extreme_comm = df[df["net_comm_pct_z"].abs() >= extreme_zscore]
    extreme_noncomm = df[df["net_noncomm_pct_z"].abs() >= extreme_zscore]

    def hit_rate_subset(sub: pd.DataFrame, col: str) -> float:
        if len(sub) < 10:
            return float("nan")
        sig_sign = np.sign(sub[col])
        sub_ret_sign = np.sign(sub["fwd_return"])
        valid = sig_sign != 0
        return float((sig_sign[valid] == sub_ret_sign[valid]).mean() * 100)

    hitrate_comm_extreme = hit_rate_subset(extreme_comm, "net_comm_pct_z")
    hitrate_noncomm_extreme = hit_rate_subset(extreme_noncomm, "net_noncomm_pct_z")
    n_extreme = min(len(extreme_comm), len(extreme_noncomm))

    if np.isnan(hitrate_comm_extreme) and np.isnan(hitrate_noncomm_extreme):
        winner_extreme = "zu wenig Extremwerte"
    elif np.isnan(hitrate_comm_extreme):
        winner_extreme = "Non-Commercials"
    elif np.isnan(hitrate_noncomm_extreme):
        winner_extreme = "Commercials"
    else:
        winner_extreme = "Commercials" if hitrate_comm_extreme >= hitrate_noncomm_extreme else "Non-Commercials"

    return CotEdgeResult(
        pair=pair_name,
        n_obs=len(df),
        corr_comm=corr_comm,
        corr_noncomm=corr_noncomm,
        hitrate_comm_level=hitrate_comm_level,
        hitrate_noncomm_level=hitrate_noncomm_level,
        hitrate_comm_momentum=hitrate_comm_mom,
        hitrate_noncomm_momentum=hitrate_noncomm_mom,
        winner_level="Commercials" if hitrate_comm_level >= hitrate_noncomm_level else "Non-Commercials",
        winner_momentum="Commercials" if hitrate_comm_mom >= hitrate_noncomm_mom else "Non-Commercials",
        n_extreme=n_extreme,
        hitrate_comm_extreme=hitrate_comm_extreme,
        hitrate_noncomm_extreme=hitrate_noncomm_extreme,
        winner_extreme=winner_extreme,
    )


def render_extra_cot_edge_analysis() -> None:
    st.markdown("## 🔬 Extra: COT Commercials vs. Non-Commercials")
    st.caption(
        "Statistisches Werkzeug, keine Anlageberatung. Prueft je Waehrungspaar, ob die "
        "Positionierung der Commercials (Hedger) oder der Non-Commercials (grosse Spekulanten) "
        "im CFTC-COT-Report historisch die hoehere Trefferquote fuer die nachfolgende "
        "Kursbewegung hatte. COT-Daten gibt es nur fuer Futures (CFTC reguliert keine "
        "Spot-Maerkte) — sowohl Positionierung als auch Kursreturns kommen deshalb konsequent "
        "vom selben CME/ICE-Waehrungsfuture, nicht vom Spot-Kurs. Ergebnisse haengen stark von "
        "Zeitraum/Horizont ab und sind nicht garantiert stabil in der Zukunft. Quellen: CFTC "
        "Socrata-API (Legacy Futures Only) + yfinance."
    )

    col1, col2, col3 = st.columns(3)
    start_year = col1.number_input("Startjahr", min_value=1986, max_value=date.today().year, value=2006, step=1, key="cot_edge_start_year")
    horizon_weeks = col2.number_input("Horizont (COT-Wochen)", min_value=1, max_value=12, value=1, step=1, key="cot_edge_horizon")
    extreme_zscore = col3.number_input("Extrem-Schwelle (z-Score)", min_value=0.5, max_value=3.0, value=1.0, step=0.1, key="cot_edge_extreme_z")

    pair_names = list(COT_EDGE_CONTRACTS.keys())
    selected_pairs = st.multiselect("Paare", pair_names, default=pair_names, key="cot_edge_pairs")

    if st.button("Analyse starten", key="cot_edge_run") and selected_pairs:
        results: list[CotEdgeResult] = []
        progress = st.progress(0.0)
        status = st.empty()
        for i, pair_name in enumerate(selected_pairs):
            meta = COT_EDGE_CONTRACTS[pair_name]
            status.text(f"Lade Daten fuer {pair_name} ...")
            cot_df = _fetch_cot_edge_history(meta["cftc_name"], int(start_year))
            price_df = _fetch_cot_edge_price(meta["yahoo_ticker"], int(start_year))
            res = _analyze_cot_edge_pair(cot_df, price_df, pair_name, int(horizon_weeks),
                                          extreme_zscore=float(extreme_zscore))
            if res is not None:
                results.append(res)
            progress.progress((i + 1) / len(selected_pairs))
        status.empty()
        progress.empty()
        st.session_state["cot_edge_results"] = results
        if not results:
            st.warning("Keine Ergebnisse — zu wenig Daten fuer die gewaehlten Paare/Zeitraum.")

    results = st.session_state.get("cot_edge_results", [])
    if not results:
        st.info("Noch keine Analyse gelaufen — Paare waehlen und 'Analyse starten' klicken.")
        return

    out = pd.DataFrame([{
        "Paar": r.pair,
        "N": r.n_obs,
        "Trefferquote Comm. (Level) %": round(r.hitrate_comm_level, 1),
        "Trefferquote Non-Comm. (Level) %": round(r.hitrate_noncomm_level, 1),
        "Sieger (Level)": r.winner_level,
        "Trefferquote Comm. (Momentum) %": round(r.hitrate_comm_momentum, 1),
        "Trefferquote Non-Comm. (Momentum) %": round(r.hitrate_noncomm_momentum, 1),
        "Sieger (Momentum)": r.winner_momentum,
        "N (extreme Wochen)": r.n_extreme,
        "Trefferquote Comm. (Extrem) %": "n/a" if np.isnan(r.hitrate_comm_extreme) else round(r.hitrate_comm_extreme, 1),
        "Trefferquote Non-Comm. (Extrem) %": "n/a" if np.isnan(r.hitrate_noncomm_extreme) else round(r.hitrate_noncomm_extreme, 1),
        "Sieger (Extrem)": r.winner_extreme,
        "Korr. Comm.": round(r.corr_comm, 3),
        "Korr. Non-Comm.": round(r.corr_noncomm, 3),
    } for r in results])

    COT_EDGE_COMM_COLOR = "#e8a23d"
    COT_EDGE_NONCOMM_COLOR = "#62c8e8"

    # ── Grafik: Trefferquote je Paar (Level-Signal) ──────────────────────────
    st.subheader("📈 Trefferquote je Paar — Level-Signal")
    chart_pairs = [r.pair for r in results]
    comm_vals = [round(r.hitrate_comm_level, 1) for r in results]
    noncomm_vals = [round(r.hitrate_noncomm_level, 1) for r in results]
    axis_max = max([65.0, *comm_vals, *noncomm_vals]) + 8

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=chart_pairs, x=comm_vals, name="Commercials", orientation="h",
        marker_color=COT_EDGE_COMM_COLOR, text=[f"{v:.1f}%" for v in comm_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=chart_pairs, x=noncomm_vals, name="Non-Commercials", orientation="h",
        marker_color=COT_EDGE_NONCOMM_COLOR, text=[f"{v:.1f}%" for v in noncomm_vals], textposition="outside",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#475569")
    fig.add_annotation(x=50, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                        text="50% = Zufall", font={"color": "#94a3b8", "size": 10})
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#111923", plot_bgcolor="#111923",
        font={"color": "#cbd5e1", "size": 12},
        height=max(340, 95 * len(chart_pairs)),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Trefferquote %", range=[0, axis_max], gridcolor="rgba(148,163,184,.12)",
                    tickfont={"color": "#94a3b8", "size": 11}),
        yaxis=dict(tickfont={"color": "#e2e8f0", "size": 12}, automargin=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabellen: kompakt je Signal-Typ, kein horizontales Scrollen noetig ──
    st.subheader("📊 Ergebnis je Paar")

    pct_cols = ["Commercials %", "Non-Commercials %"]

    def _highlight_winner(df: pd.DataFrame, col: str):
        def _color(v):
            if v == "Commercials":
                return f"color:{COT_EDGE_COMM_COLOR};font-weight:700"
            if v == "Non-Commercials":
                return f"color:{COT_EDGE_NONCOMM_COLOR};font-weight:700"
            return ""
        styler = df.style.map(_color, subset=[col])
        # explizite Formatierung noetig, da die Extrem-Tabelle "n/a"-Strings mit floats mischt
        # (object-dtype) -- ohne .format() zeigt Styler sonst 6 statt 1 Nachkommastelle.
        fmt = {c: (lambda v: v if isinstance(v, str) else f"{v:.1f}") for c in pct_cols if c in df.columns}
        return styler.format(fmt)

    tab_level, tab_momentum, tab_extreme, tab_raw = st.tabs(
        ["Level-Signal", "Momentum-Signal", "Extrem-Divergenz", "Alle Rohdaten"]
    )

    with tab_level:
        level_df = pd.DataFrame([{
            "Paar": r.pair, "N": r.n_obs,
            "Commercials %": round(r.hitrate_comm_level, 1),
            "Non-Commercials %": round(r.hitrate_noncomm_level, 1),
            "Sieger": r.winner_level,
        } for r in results])
        st.dataframe(_highlight_winner(level_df, "Sieger"), use_container_width=True, hide_index=True)

    with tab_momentum:
        momentum_df = pd.DataFrame([{
            "Paar": r.pair, "N": r.n_obs,
            "Commercials %": round(r.hitrate_comm_momentum, 1),
            "Non-Commercials %": round(r.hitrate_noncomm_momentum, 1),
            "Sieger": r.winner_momentum,
        } for r in results])
        st.dataframe(_highlight_winner(momentum_df, "Sieger"), use_container_width=True, hide_index=True)

    with tab_extreme:
        extreme_df = pd.DataFrame([{
            "Paar": r.pair, "N (extrem)": r.n_extreme,
            "Commercials %": "n/a" if np.isnan(r.hitrate_comm_extreme) else round(r.hitrate_comm_extreme, 1),
            "Non-Commercials %": "n/a" if np.isnan(r.hitrate_noncomm_extreme) else round(r.hitrate_noncomm_extreme, 1),
            "Sieger": r.winner_extreme,
        } for r in results])
        st.dataframe(_highlight_winner(extreme_df, "Sieger"), use_container_width=True, hide_index=True)

    with tab_raw:
        st.dataframe(out, use_container_width=True, hide_index=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("CSV herunterladen", data=csv_bytes, file_name="cot_edge_analysis.csv", mime="text/csv")

    st.subheader("🏆 Zusammenfassung")
    summary_cols = st.columns(3)
    for col, (label, col_name) in zip(summary_cols, [
        ("Level-Signal", "Sieger (Level)"), ("Momentum-Signal", "Sieger (Momentum)"), ("Extrem-Divergenz", "Sieger (Extrem)"),
    ]):
        counts = out[col_name].value_counts()
        winner = counts.idxmax()
        col.metric(label, winner, f"{counts.max()} von {len(out)} Paaren")


test_mode = st.sidebar.radio("", ["Manual Backtest", "TACO Edge Discovery", "Cycle Scanner", "SL Scanner", "TACO Radar", "Walk Forward Analysis", "Seasonality Lab", "Seasonality Muster", "Muster Analyse", "Yen Mo-Mi Strategie", "Crypto WeekdayMA WFA", "DAX EMA Strategie", "Extra: Makro & Sentiment", "Extra: COT Commercials vs. Spekulanten"], horizontal=False, label_visibility="collapsed")

if test_mode == "Seasonality Lab":
    render_seasonality_lab()
    st.stop()

if test_mode == "Seasonality Muster":
    render_seasonality_muster()
    st.stop()

if test_mode == "Muster Analyse":
    render_muster_analyse()
    st.stop()

if test_mode == "Yen Mo-Mi Strategie":
    render_yen_momi_strategie()
    st.stop()

if test_mode == "Crypto WeekdayMA WFA":
    render_btc_wfa()
    st.stop()

if test_mode == "DAX EMA Strategie":
    render_dax_ema_wfa()
    st.stop()

if test_mode == "Extra: Makro & Sentiment":
    render_extra_makro_sentiment()
    st.stop()

if test_mode == "Extra: COT Commercials vs. Spekulanten":
    render_extra_cot_edge_analysis()
    st.stop()

with st.sidebar:
    auto_match_cot = st.checkbox("Auto-match COT market to selected asset", True)

    st.header("Daten")
    data_mode = st.radio("Datenquelle", ["Demo", "CSV Upload", "Yahoo Symbol"], horizontal=True)
    asset_df = comp_df = None
    if data_mode == "CSV Upload":
        asset_file = st.file_uploader("Chart Asset CSV", type=["csv"])
        comp_file = st.file_uploader("Comparison Asset CSV", type=["csv"])
        if asset_file and comp_file:
            asset_df = normalize_ohlc(pd.read_csv(asset_file))
            comp_df = normalize_ohlc(pd.read_csv(comp_file))
    elif data_mode == "Yahoo Symbol":
        asset_preset = st.selectbox("Chart Asset Preset", list(ASSET_PRESETS.keys()))
        comp_preset = st.selectbox("Comparison Asset Preset", list(COMPARISON_PRESETS.keys()))
        asset_symbol = st.text_input("Chart Asset Symbol", ASSET_PRESETS[asset_preset])
        comp_symbol = st.text_input("Comparison Asset Symbol", COMPARISON_PRESETS[comp_preset])
        st.caption("Hinweis: Yahoo liefert freie Index-/Futures-Proxies, nicht zwingend deinen exakten CFD-Brokerkurs.")
        if st.button("Daten laden"):
            asset_df = load_yahoo(asset_symbol)
            comp_df = load_yahoo(comp_symbol)
            if asset_df is None or comp_df is None:
                st.warning("Yahoo-Daten konnten nicht geladen werden. Nutze CSV oder Demo.")
    else:
        asset_preset = "Demo"
        asset_symbol = "Demo"
        asset_df, comp_df = make_demo_data()

    if test_mode not in ("Cycle Scanner", "SL Scanner", "Walk Forward Analysis"):
        st.header("Einstellungen")
        enable_take_profit = st.checkbox("Enable Take Profit", True)
        selected_tp_mode = st.selectbox("Take Profit Mode", ["Risk Reward", "Fixed %"]) if enable_take_profit else "None"
        settings = Settings(
            cycle_length=st.number_input("Cycle Length", 2, 100, 10),
            smoothing=st.number_input("Glaettung", 1, 50, 5),
            softness=st.number_input("Normalization Softness", 0.25, 5.0, 1.35, step=0.05),
            mode=st.selectbox("Mode", ["Ratio Z-Score", "Return Spread"]),
            trade_direction=st.selectbox("Trade Direction", ["Long & Short", "Long Only", "Short Only"]),
            start_year=st.number_input("Start Year", 1900, 2100, 2015),
            end_year=st.number_input("End Year", 1900, 2100, 2026),
            upper=st.number_input("Upper Bound", value=75.0),
            lower=st.number_input("Lower Bound", value=-75.0),
            risk_pct=st.number_input("Risk Per Trade %", 0.1, 10.0, 1.0, step=0.5),
            stop_pct=st.number_input("Fixed Stop Loss %", 0.05, 20.0, 0.65, step=0.05),
            tp_mode=selected_tp_mode,
            rr=st.number_input("Take Profit R Multiple", 0.1, 20.0, 2.0, step=0.1),
            fixed_tp_pct=st.number_input("Fixed Take Profit %", 0.05, 50.0, 1.3, step=0.05),
            exit_on_zero=st.checkbox("Exit When Oscillator Returns To Zero", False),
            time_exit=st.checkbox("Enable Exit After X Bars", False),
            exit_after_bars=st.number_input("Exit After X Bars", 1, 500, 20),
            initial_capital=st.number_input("Initial Capital", 100.0, 1_000_000.0, 10_000.0, step=100.0),
            commission_pct=st.number_input("Commission %", 0.0, 2.0, 0.05, step=0.01),
            slippage_pct=st.number_input("Slippage %", 0.0, 2.0, 0.02, step=0.01),
        )
        if not enable_take_profit and not settings.exit_on_zero and not settings.time_exit:
            st.warning(
                "Take Profit ist deaktiviert und es ist kein Zero-Line- oder Time-Exit aktiv. "
                "Dann hat die Strategie nur den Stop Loss als echten Exit."
            )

    if test_mode == "Walk Forward Analysis":
        # Alle nicht genutzten Scanner-Vars auf Defaults setzen
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_radar = False
        # WF-Variablen werden in der Inline-Toolbar im Hauptbereich gesetzt
        wf_assets = []
        wf_comps = []
        wf_directions = []
        wf_start_year = 2015
        wf_end_year = 2026
        wf_in_sample_years = 20
        wf_cycle_from = 5
        wf_cycle_to = 30
        wf_cycle_step = 1
        wf_sl_from = 0.25
        wf_sl_to = 2.0
        wf_sl_step = 0.05
        wf_min_trades = 30
        wf_max_loss_streak = 5
        run_wf = False
    elif test_mode == "TACO Radar":
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_wf = False

        st.header("TACO Radar")
        radar_assets = st.multiselect(
            "Radar Assets",
            list(ASSET_PRESETS.keys()),
            default=[
                "EURUSD (EURUSD=X)",
                "GBPUSD (GBPUSD=X)",
                "AUDUSD (AUDUSD=X)",
                "NZDUSD (NZDUSD=X)",
                "USDCAD (CAD=X)",
                "USDCHF (CHF=X)",
                "USDJPY (JPY=X)",
                "UK100 proxy: FTSE 100 Index (^FTSE)",
                "GER40 proxy: DAX Index (^GDAXI)",
                "US100 proxy: Nasdaq 100 (^NDX)",
                "S&P500 / US500 proxy: S&P 500 (^GSPC)",
                "US30 proxy: Dow Jones Industrial Average (^DJI)",
            ],
        )
        radar_comps = st.multiselect(
            "Radar Comparison Assets",
            list(COMPARISON_PRESETS.keys()),
            default=["DXY proxy: US Dollar Index (DX-Y.NYB)", "Gold futures (GC=F)", "10Y Treasury Note futures (ZN=F)"],
        )
        radar_cycle_from = st.number_input("Radar Cycle From", 2, 100, 5)
        radar_cycle_to = st.number_input("Radar Cycle To", 2, 100, 30)
        radar_cycle_step = st.number_input("Radar Cycle Step", 1, 20, 1)
        radar_min_trades = st.number_input("Radar Min Trades", 1, 1000, 30)
        radar_max_loss_streak = st.number_input("Radar Max Loss Streak", 0, 100, 5)
        radar_top_cycles = st.number_input("Cycle Vorschlaege pro Signal", 1, 10, 5)
        radar_near_zone = st.number_input("Near Zone Buffer", 0.0, 50.0, 5.0, step=1.0)
        run_radar = st.button("Run TACO Radar", type="primary")
    elif test_mode == "Cycle Scanner":
        # Controls rendered inline in main content below
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_wf = False
        run_radar = False
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
    elif test_mode == "SL Scanner":
        # Controls rendered inline in main content below
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        run_wf = False
        run_radar = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
    else:
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_radar = False
        run_wf = False

if test_mode == "TACO Edge Discovery":
    if asset_df is None or comp_df is None:
        st.info("Bitte Daten laden oder Demo nutzen.")
        st.stop()
    edge_df = calculate_oscillator(asset_df, comp_df, settings)
    render_taco_edge_discovery(edge_df, settings)
    st.stop()

# ── Cycle Scanner inline toolbar ──────────────────────────────────────────────
if test_mode == "Cycle Scanner":
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;"
        "color:#9fb0c7;font-weight:700;margin-bottom:4px;'>Cycle Scanner — Assets & Scan-Bereich</div>",
        unsafe_allow_html=True,
    )
    _cs_row1 = st.columns([2, 2, 1])
    with _cs_row1[0]:
        scan_assets = st.multiselect(
            "Assets",
            list(ASSET_PRESETS.keys()),
            default=[
                "UK100 proxy: FTSE 100 Index (^FTSE)",
                "GER40 proxy: DAX Index (^GDAXI)",
                "US100 proxy: Nasdaq 100 (^NDX)",
                "S&P500 / US500 proxy: S&P 500 (^GSPC)",
                "US30 proxy: Dow Jones Industrial Average (^DJI)",
            ],
        )
    with _cs_row1[1]:
        scan_comps = st.multiselect(
            "Comparison Assets",
            list(COMPARISON_PRESETS.keys()),
            default=["DXY proxy: US Dollar Index (DX-Y.NYB)", "Gold futures (GC=F)", "10Y Treasury Note futures (ZN=F)"],
        )
    with _cs_row1[2]:
        scan_directions = st.multiselect(
            "Directions",
            ["Long Only", "Short Only", "Long & Short"],
            default=["Long Only", "Short Only"],
        )
    _cs_row2 = st.columns([1, 1, 1, 1, 1, 1.4])
    with _cs_row2[0]:
        scan_cycle_from = st.number_input("Cycle Von", 2, 100, 5)
    with _cs_row2[1]:
        scan_cycle_to = st.number_input("Cycle Bis", 2, 100, 30)
    with _cs_row2[2]:
        scan_cycle_step = st.number_input("Step", 1, 20, 1)
    with _cs_row2[3]:
        top_curve_min_trades = st.number_input("Min Trades", 1, 1000, 50)
    with _cs_row2[4]:
        top_curve_max_loss_streak = st.number_input("Max Loss Streak", 0, 100, 5)
    with _cs_row2[5]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_scan = st.button("Run Cycle Scan", type="primary", use_container_width=True)

    with st.expander("Basis-Parameter (Oscillator & Trade-Setup)"):
        _bp1 = st.columns([1, 1, 1, 1, 1])
        with _bp1[0]:
            _cs_smoothing = st.number_input("Glaettung", 1, 50, 5, key="cs_smoothing")
        with _bp1[1]:
            _cs_softness = st.number_input("Normalization Softness", 0.25, 5.0, 1.35, step=0.05, key="cs_softness")
        with _bp1[2]:
            _cs_mode = st.selectbox("Mode", ["Ratio Z-Score", "Return Spread"], key="cs_mode")
        with _bp1[3]:
            _cs_start_year = st.number_input("Start Year", 1900, 2100, 2015, key="cs_start_year")
        with _bp1[4]:
            _cs_end_year = st.number_input("End Year", 1900, 2100, 2026, key="cs_end_year")
        _bp2 = st.columns([1, 1, 1, 1, 1])
        with _bp2[0]:
            _cs_upper = st.number_input("Upper Bound", value=75.0, key="cs_upper")
        with _bp2[1]:
            _cs_lower = st.number_input("Lower Bound", value=-75.0, key="cs_lower")
        with _bp2[2]:
            _cs_stop_pct = st.number_input("Fixed Stop Loss %", 0.05, 20.0, 0.65, step=0.05, key="cs_stop_pct")
        with _bp2[3]:
            _cs_risk_pct = st.number_input("Risk Per Trade %", 0.1, 10.0, 1.0, step=0.5, key="cs_risk_pct")
        with _bp2[4]:
            _cs_initial_capital = st.number_input("Initial Capital", 100.0, 1_000_000.0, 10_000.0, step=100.0, key="cs_initial_capital")
        _bp3 = st.columns([1, 1, 1, 1, 1])
        with _bp3[0]:
            _cs_enable_tp = st.checkbox("Enable Take Profit", True, key="cs_enable_tp")
        with _bp3[1]:
            _cs_tp_mode = st.selectbox("TP Mode", ["Risk Reward", "Fixed %"], key="cs_tp_mode") if _cs_enable_tp else "None"
        with _bp3[2]:
            _cs_rr = st.number_input("TP R Multiple", 0.1, 20.0, 2.0, step=0.1, key="cs_rr")
        with _bp3[3]:
            _cs_commission = st.number_input("Commission %", 0.0, 2.0, 0.05, step=0.01, key="cs_commission")
        with _bp3[4]:
            _cs_slippage = st.number_input("Slippage %", 0.0, 2.0, 0.02, step=0.01, key="cs_slippage")

    settings = Settings(
        cycle_length=10,
        smoothing=_cs_smoothing,
        softness=_cs_softness,
        mode=_cs_mode,
        trade_direction="Long & Short",
        start_year=_cs_start_year,
        end_year=_cs_end_year,
        upper=_cs_upper,
        lower=_cs_lower,
        risk_pct=_cs_risk_pct,
        stop_pct=_cs_stop_pct,
        tp_mode=_cs_tp_mode if _cs_enable_tp else "None",
        rr=_cs_rr,
        fixed_tp_pct=1.3,
        exit_on_zero=False,
        time_exit=False,
        exit_after_bars=20,
        initial_capital=_cs_initial_capital,
        commission_pct=_cs_commission,
        slippage_pct=_cs_slippage,
    )

# ── Walk Forward Analysis inline toolbar ──────────────────────────────────────
elif test_mode == "Walk Forward Analysis":
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;"
        "color:#9fb0c7;font-weight:700;margin-bottom:4px;'>Walk Forward Analysis — Assets & Scan-Bereich</div>",
        unsafe_allow_html=True,
    )
    _wf_row1 = st.columns([2, 2, 1])
    with _wf_row1[0]:
        wf_assets = st.multiselect(
            "Assets",
            list(ASSET_PRESETS.keys()),
            default=[
                "EURUSD (EURUSD=X)",
                "GBPUSD (GBPUSD=X)",
                "AUDUSD (AUDUSD=X)",
                "USDJPY (JPY=X)",
                "US500 proxy: S&P 500 (^GSPC)" if "US500 proxy: S&P 500 (^GSPC)" in ASSET_PRESETS else list(ASSET_PRESETS.keys())[0],
            ],
        )
    with _wf_row1[1]:
        wf_comps = st.multiselect(
            "Comparison Assets",
            list(COMPARISON_PRESETS.keys()),
            default=["DXY proxy: US Dollar Index (DX-Y.NYB)", "Gold futures (GC=F)", "10Y Treasury Note futures (ZN=F)"],
        )
    with _wf_row1[2]:
        wf_directions = st.multiselect(
            "Directions",
            ["Long Only", "Short Only", "Long & Short"],
            default=["Long Only", "Short Only"],
        )
    _wf_row2 = st.columns([1, 1, 1, 1, 1, 1, 1.4])
    with _wf_row2[0]:
        wf_start_year = st.number_input("Start Jahr", 1900, 2100, 2015, key="wf_start_year")
    with _wf_row2[1]:
        wf_end_year = st.number_input("End Jahr", 1900, 2100, 2026, key="wf_end_year")
    with _wf_row2[2]:
        wf_in_sample_years = st.number_input("IS Fenster (J)", 1, 50, 20, key="wf_is_years")
    with _wf_row2[3]:
        wf_cycle_from = st.number_input("Cycle Von", 2, 100, 5, key="wf_cycle_from")
    with _wf_row2[4]:
        wf_cycle_to = st.number_input("Cycle Bis", 2, 100, 30, key="wf_cycle_to")
    with _wf_row2[5]:
        wf_cycle_step = st.number_input("Step", 1, 20, 1, key="wf_cycle_step")
    with _wf_row2[6]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_wf = st.button("Run Walk Forward", type="primary", use_container_width=True)

    with st.expander("Stop Loss & Filter-Einstellungen"):
        _wf_bp = st.columns([1, 1, 1, 1, 1])
        with _wf_bp[0]:
            wf_sl_from = st.number_input("SL Von %", 0.05, 20.0, 0.25, step=0.05, key="wf_sl_from")
        with _wf_bp[1]:
            wf_sl_to = st.number_input("SL Bis %", 0.05, 20.0, 2.00, step=0.05, key="wf_sl_to")
        with _wf_bp[2]:
            wf_sl_step = st.number_input("SL Step %", 0.05, 5.0, 0.05, step=0.05, key="wf_sl_step")
        with _wf_bp[3]:
            wf_min_trades = st.number_input("Min IS Trades", 1, 1000, 30, key="wf_min_trades")
        with _wf_bp[4]:
            wf_max_loss_streak = st.number_input("Max Loss Streak", 0, 100, 5, key="wf_max_loss_streak")

    with st.expander("Basis-Parameter (Oscillator & Trade-Setup)"):
        _wf_base1 = st.columns([1, 1, 1, 1, 1])
        with _wf_base1[0]:
            _wf_smoothing = st.number_input("Glaettung", 1, 50, 5, key="wf_smoothing")
        with _wf_base1[1]:
            _wf_softness = st.number_input("Normalization Softness", 0.25, 5.0, 1.35, step=0.05, key="wf_softness")
        with _wf_base1[2]:
            _wf_mode = st.selectbox("Mode", ["Ratio Z-Score", "Return Spread"], key="wf_mode")
        with _wf_base1[3]:
            _wf_upper = st.number_input("Upper Bound", value=75.0, key="wf_upper")
        with _wf_base1[4]:
            _wf_lower = st.number_input("Lower Bound", value=-75.0, key="wf_lower")
        _wf_base2 = st.columns([1, 1, 1, 1, 1])
        with _wf_base2[0]:
            _wf_risk_pct = st.number_input("Risk Per Trade %", 0.1, 10.0, 1.0, step=0.5, key="wf_risk_pct")
        with _wf_base2[1]:
            _wf_enable_tp = st.checkbox("Enable Take Profit", True, key="wf_enable_tp")
        with _wf_base2[2]:
            _wf_tp_mode = st.selectbox("TP Mode", ["Risk Reward", "Fixed %"], key="wf_tp_mode") if _wf_enable_tp else "None"
        with _wf_base2[3]:
            _wf_rr = st.number_input("TP R Multiple", 0.1, 20.0, 2.0, step=0.1, key="wf_rr")
        with _wf_base2[4]:
            _wf_initial_capital = st.number_input("Initial Capital", 100.0, 1_000_000.0, 10_000.0, step=100.0, key="wf_initial_capital")

    settings = Settings(
        cycle_length=10,
        smoothing=_wf_smoothing,
        softness=_wf_softness,
        mode=_wf_mode,
        trade_direction="Long & Short",
        start_year=int(wf_start_year),
        end_year=int(wf_end_year),
        upper=_wf_upper,
        lower=_wf_lower,
        risk_pct=_wf_risk_pct,
        stop_pct=float(wf_sl_from),
        tp_mode=_wf_tp_mode if _wf_enable_tp else "None",
        rr=_wf_rr,
        fixed_tp_pct=1.3,
        exit_on_zero=False,
        time_exit=False,
        exit_after_bars=20,
        initial_capital=_wf_initial_capital,
        commission_pct=0.05,
        slippage_pct=0.02,
    )

# ── SL Scanner inline toolbar ──────────────────────────────────────────────────
elif test_mode == "SL Scanner":
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;"
        "color:#9fb0c7;font-weight:700;margin-bottom:4px;'>SL Scanner — Asset & SL-Bereich</div>",
        unsafe_allow_html=True,
    )
    _sl_row1 = st.columns([2, 2, 1])
    with _sl_row1[0]:
        sl_asset_preset = st.selectbox("Asset", list(ASSET_PRESETS.keys()))
    with _sl_row1[1]:
        sl_comp_preset = st.selectbox("Comparison Asset", list(COMPARISON_PRESETS.keys()))
    with _sl_row1[2]:
        sl_directions = st.multiselect("Directions", ["Long Only", "Short Only", "Long & Short"], default=["Long Only"])
    _sl_row2 = st.columns([1, 1, 1, 1.4])
    with _sl_row2[0]:
        sl_from = st.number_input("SL Von %", 0.05, 20.0, 0.25, step=0.05)
    with _sl_row2[1]:
        sl_to = st.number_input("SL Bis %", 0.05, 20.0, 2.00, step=0.05)
    with _sl_row2[2]:
        sl_step = st.number_input("Step %", 0.05, 5.0, 0.05, step=0.05)
    with _sl_row2[3]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_sl_scan = st.button("Run SL Scan", type="primary", use_container_width=True)

    with st.expander("Basis-Parameter (Oscillator & Trade-Setup)"):
        _slbp1 = st.columns([1, 1, 1, 1, 1])
        with _slbp1[0]:
            _sl_cycle = st.number_input("Cycle Length", 2, 100, 10, key="sl_cycle")
        with _slbp1[1]:
            _sl_smoothing = st.number_input("Glaettung", 1, 50, 5, key="sl_smoothing")
        with _slbp1[2]:
            _sl_mode = st.selectbox("Mode", ["Ratio Z-Score", "Return Spread"], key="sl_mode")
        with _slbp1[3]:
            _sl_start_year = st.number_input("Start Year", 1900, 2100, 2015, key="sl_start_year")
        with _slbp1[4]:
            _sl_end_year = st.number_input("End Year", 1900, 2100, 2026, key="sl_end_year")
        _slbp2 = st.columns([1, 1, 1, 1, 1])
        with _slbp2[0]:
            _sl_upper = st.number_input("Upper Bound", value=75.0, key="sl_upper")
        with _slbp2[1]:
            _sl_lower = st.number_input("Lower Bound", value=-75.0, key="sl_lower")
        with _slbp2[2]:
            _sl_risk_pct = st.number_input("Risk Per Trade %", 0.1, 10.0, 1.0, step=0.5, key="sl_risk_pct")
        with _slbp2[3]:
            _sl_initial_capital = st.number_input("Initial Capital", 100.0, 1_000_000.0, 10_000.0, step=100.0, key="sl_initial_capital")
        with _slbp2[4]:
            _sl_softness = st.number_input("Normalization Softness", 0.25, 5.0, 1.35, step=0.05, key="sl_softness")
        _slbp3 = st.columns([1, 1, 1, 1, 1])
        with _slbp3[0]:
            _sl_enable_tp = st.checkbox("Enable Take Profit", True, key="sl_enable_tp")
        with _slbp3[1]:
            _sl_tp_mode = st.selectbox("TP Mode", ["Risk Reward", "Fixed %"], key="sl_tp_mode") if _sl_enable_tp else "None"
        with _slbp3[2]:
            _sl_rr = st.number_input("TP R Multiple", 0.1, 20.0, 2.0, step=0.1, key="sl_rr")
        with _slbp3[3]:
            _sl_commission = st.number_input("Commission %", 0.0, 2.0, 0.05, step=0.01, key="sl_commission")
        with _slbp3[4]:
            _sl_slippage = st.number_input("Slippage %", 0.0, 2.0, 0.02, step=0.01, key="sl_slippage")

    settings = Settings(
        cycle_length=_sl_cycle,
        smoothing=_sl_smoothing,
        softness=_sl_softness,
        mode=_sl_mode,
        trade_direction="Long & Short",
        start_year=_sl_start_year,
        end_year=_sl_end_year,
        upper=_sl_upper,
        lower=_sl_lower,
        risk_pct=_sl_risk_pct,
        stop_pct=0.65,
        tp_mode=_sl_tp_mode if _sl_enable_tp else "None",
        rr=_sl_rr,
        fixed_tp_pct=1.3,
        exit_on_zero=False,
        time_exit=False,
        exit_after_bars=20,
        initial_capital=_sl_initial_capital,
        commission_pct=_sl_commission,
        slippage_pct=_sl_slippage,
    )

render_fear_greed_panel()

cot_asset_label = sl_asset_preset if test_mode == "SL Scanner" else asset_preset if data_mode == "Yahoo Symbol" else "Demo"
cot_asset_symbol = ASSET_PRESETS.get(cot_asset_label, asset_symbol if "asset_symbol" in locals() else None)
render_cot_panel(auto_match_cot, cot_asset_label, cot_asset_symbol)

with st.expander("Info: Wie funktioniert der TACO Backtest?", expanded=True):
    st.markdown(
        """
        **Strategie-Ablauf:** Die Strategie wartet auf ein bestaetigtes TACO-Signal zum Daily-Close.
        Danach wird der Entry zum Schlusskurs der Tageskerze simuliert. Stop Loss und Take Profit werden
        direkt beim Entry fix berechnet und nicht nachtraeglich verschoben.

        **Berechnung:** Das Chart-Asset wird mit einem Vergleichsasset verglichen. Im Modus `Ratio Z-Score`
        wird `Chart Close / Comparison Close` berechnet, per Mittelwert und Standardabweichung normalisiert
        und mit `tanh()` auf etwa `-100` bis `+100` begrenzt.

        **Long Entry:** Der Oszillator ist unter der Unterbewertungszone und dreht nach oben, oder er kreuzt
        von unten zurueck ueber die Unterbewertungszone.

        **Short Entry:** Der Oszillator ist ueber der Ueberbewertungszone und dreht nach unten, oder er kreuzt
        von oben zurueck unter die Ueberbewertungszone.

        **Risk:** `Risk Per Trade %` bestimmt dein Kontorisiko. `Fixed Stop Loss %` bestimmt den Abstand vom
        Entry zum Stop. Die Positionsgroesse wird daraus automatisch berechnet. Beispiel: Bei 10.000 USD Konto,
        1% Risiko und 0,70% Stop wird die Positionsgroesse so gewaehlt, dass ein sauberer Stop etwa 100 USD
        Verlust entspricht.

        **Exit:** Je nach Einstellung per Risk-Reward-Target, Fixed-%-Target, Zero-Line-Exit oder Time-Exit.
        Wenn `Enable Take Profit` deaktiviert ist, arbeitet die Strategie ohne festes Profit Target und beendet
        Trades nur ueber Stop, Zero-Line-Exit oder Time-Exit.

        **Wichtig bei deaktiviertem Take Profit:** Wenn `Enable Take Profit` ausgeschaltet ist, sollte mindestens
        `Exit When Oscillator Returns To Zero` oder `Enable Exit After X Bars` aktiviert sein. Sonst hat die
        Strategie nur den Stop Loss als echten Exit und Gewinner koennen theoretisch sehr lange offen bleiben.

        **Manual Backtest:** Du testest ein einzelnes Setup visuell.

        **Cycle Scanner:** Die App testet viele Cycle Lengths ueber ausgewaehlte Assets, Vergleichsassets
        und Richtungen.

        **SL Scanner:** Die App testet einen Stop-Loss-Bereich, z.B. 0,25% bis 2,00% in 0,05er-Schritten.
        Wichtig fuer die Auswertung sind hier `Avg Realized Win`, `Avg Realized Loss`, `Expectancy R`,
        `Max Loss Streak`, `Intratrade MAE` und `Stop Breach`.

        **Walk Forward Analysis:** Fuer jedes Out-of-Sample-Jahr werden Cycle Length und Stop Loss % nur auf
        den vorherigen In-Sample-Jahren optimiert. Das getestete OOS-Jahr wird bei der Optimierung strikt nicht
        verwendet. Danach wird der beste Parametersatz nur auf genau dieses eine OOS-Jahr angewendet.

        **Hinweis:** `Intratrade MAE` misst den groessten Kerzen-Gegenlauf waehrend des Trades. Das kann tiefer
        sein als dein Stop, weil Daily-Kerzen nur Open/High/Low/Close liefern. `Avg Realized Loss` und `Avg R`
        zeigen dagegen, was im Backtest tatsaechlich realisiert wurde.

        **Slippage:** `Slippage %` verschlechtert Entry und Exit leicht. Beispiel: 0,02% bedeutet, dass Long-Entries
        etwas hoeher und Long-Exits etwas tiefer simuliert werden. Das macht den Backtest konservativer.
        """
    )

if test_mode == "Walk Forward Analysis":
    st.caption(
        "Realistische Out-of-Sample-Validierung: Jedes Testjahr wird nur mit Parametern gehandelt, "
        "die aus den vorherigen In-Sample-Jahren bestimmt wurden. Das OOS-Jahr ist nie Teil der Optimierung."
    )

    if not run_wf:
        st.info("Assets, Comparison Assets und Directions auswählen. Danach auf 'Run Walk Forward' klicken.")
        st.stop()

    if not wf_assets or not wf_comps or not wf_directions:
        st.warning("Bitte mindestens ein Asset, ein Comparison Asset und eine Direction auswählen.")
        st.stop()
    if wf_end_year < wf_start_year:
        st.error("End Jahr muss ≥ Start Jahr sein.")
        st.stop()
    if wf_cycle_to < wf_cycle_from:
        st.error("Cycle Bis muss ≥ Cycle Von sein.")
        st.stop()
    if wf_sl_to < wf_sl_from:
        st.error("SL Bis muss ≥ SL Von sein.")
        st.stop()

    wf_cycles = list(range(int(wf_cycle_from), int(wf_cycle_to) + 1, int(wf_cycle_step)))
    wf_stop_values = np.round(np.arange(float(wf_sl_from), float(wf_sl_to) + float(wf_sl_step) / 2, float(wf_sl_step)), 4).tolist()

    # Alle Asset × Comparison × Direction Kombinationen durchlaufen
    from itertools import product as _iterproduct
    wf_combos = [(a, c, d) for a, c, d in _iterproduct(wf_assets, wf_comps, wf_directions)]
    all_wf_yearly = []
    all_wf_trades = []
    all_wf_equity = []
    all_wf_summary = []

    _wf_progress = st.progress(0, text="Starte Walk Forward…")
    for _wf_i, (wf_asset_key, wf_comp_key, wf_dir) in enumerate(wf_combos):
        _wf_progress.progress((_wf_i) / max(len(wf_combos), 1),
                               text=f"WFA: {wf_asset_key[:30]} vs {wf_comp_key[:20]} ({wf_dir}) …")
        wf_asset_sym = ASSET_PRESETS.get(wf_asset_key, wf_asset_key)
        wf_comp_sym  = COMPARISON_PRESETS.get(wf_comp_key, wf_comp_key)
        wf_asset_data = load_yahoo(wf_asset_sym)
        wf_comp_data  = load_yahoo(wf_comp_sym)
        if wf_asset_data is None or wf_comp_data is None:
            st.warning(f"Daten konnten nicht geladen werden: {wf_asset_key} / {wf_comp_key} — übersprungen.")
            continue
        _wf_settings = Settings(
            cycle_length=10,
            smoothing=settings.smoothing,
            softness=settings.softness,
            mode=settings.mode,
            trade_direction=wf_dir,
            start_year=int(wf_start_year),
            end_year=int(wf_end_year),
            upper=settings.upper,
            lower=settings.lower,
            risk_pct=settings.risk_pct,
            stop_pct=settings.stop_pct,
            tp_mode=settings.tp_mode,
            rr=settings.rr,
            fixed_tp_pct=settings.fixed_tp_pct,
            exit_on_zero=settings.exit_on_zero,
            time_exit=settings.time_exit,
            exit_after_bars=settings.exit_after_bars,
            initial_capital=settings.initial_capital,
            commission_pct=settings.commission_pct,
            slippage_pct=settings.slippage_pct,
        )
        wf_yearly, wf_trades, wf_equity, wf_summary = run_walk_forward(
            wf_asset_data,
            wf_comp_data,
            _wf_settings,
            int(wf_start_year),
            int(wf_end_year),
            int(wf_in_sample_years),
            wf_cycles,
            wf_stop_values,
            int(wf_min_trades),
            int(wf_max_loss_streak),
        )
        # Kombo-Label für Zusammenführung
        _lbl = f"{wf_asset_key[:20]} / {wf_comp_key[:15]} / {wf_dir}"
        wf_yearly["Kombination"] = _lbl
        wf_trades["Kombination"] = _lbl
        all_wf_yearly.append(wf_yearly)
        all_wf_trades.append(wf_trades)
        all_wf_equity.append((wf_equity, _lbl))
        all_wf_summary.append({"Kombination": _lbl, **wf_summary})

    _wf_progress.progress(1.0, text="Walk Forward abgeschlossen")
    _wf_progress.empty()

    if not all_wf_summary:
        st.warning("Keine Ergebnisse — prüfe Datenquellen und Einstellungen.")
        st.stop()

    # ── Zusammenfassung über alle Kombinationen ──────────────────────────────
    st.subheader("Walk Forward — Gesamtübersicht")
    wf_summary_df = pd.DataFrame(all_wf_summary)
    st.dataframe(wf_summary_df, use_container_width=True, hide_index=True)

    # ── Equity-Chart: alle Kombinationen als Linien ──────────────────────────
    if all_wf_equity:
        wf_equity_fig = go.Figure()
        for eq_ser, lbl in all_wf_equity:
            if not eq_ser.empty:
                wf_equity_fig.add_trace(go.Scatter(
                    x=eq_ser["date"] if "date" in eq_ser.columns else eq_ser.index,
                    y=eq_ser["equity"] if "equity" in eq_ser.columns else eq_ser.values,
                    mode="lines", name=lbl,
                ))
        wf_equity_fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                                     yaxis_title="Equity", legend=dict(font_size=10))
        st.plotly_chart(wf_equity_fig, use_container_width=True)

    # ── Detailtabellen ────────────────────────────────────────────────────────
    _sel_combo = st.selectbox("Detail-Ansicht für Kombination:", [s["Kombination"] for s in all_wf_summary])
    _sel_idx = [s["Kombination"] for s in all_wf_summary].index(_sel_combo)
    wf_yearly_sel = all_wf_yearly[_sel_idx]
    wf_trades_sel = all_wf_trades[_sel_idx]

    st.subheader("Walk Forward Jahresergebnisse")
    st.dataframe(wf_yearly_sel.drop(columns=["Kombination"], errors="ignore"), use_container_width=True)

    st.subheader("Walk Forward Trades")
    st.dataframe(wf_trades_sel.drop(columns=["Kombination"], errors="ignore"), use_container_width=True)

    yearly_csv = pd.concat(all_wf_yearly, ignore_index=True).to_csv(index=False).encode("utf-8")
    trades_csv = pd.concat(all_wf_trades, ignore_index=True).to_csv(index=False).encode("utf-8")
    col_a, col_b = st.columns(2)
    col_a.download_button("WF Jahresergebnisse als CSV laden", yearly_csv, "taco_walk_forward_years.csv", "text/csv")
    col_b.download_button("WF Trades als CSV laden", trades_csv, "taco_walk_forward_trades.csv", "text/csv")
    st.stop()

if test_mode == "TACO Radar":
    st.subheader("TACO Radar")
    st.caption(
        "Der Radar scannt mehrere Assets automatisch. Er zeigt nur Maerkte, bei denen der aktuelle TACO-Oszillator "
        "in oder nahe einer Ueber-/Unterbewertungszone liegt und die Cycle-Kennzahlen robust genug sind."
    )

    if not run_radar:
        st.info("Waehle links Radar Assets, Comparison Assets und Cycle Range aus. Danach auf Run TACO Radar klicken.")
        st.stop()

    if radar_cycle_to < radar_cycle_from:
        st.error("Radar Cycle To muss groesser oder gleich Radar Cycle From sein.")
        st.stop()

    radar_combos = []
    for asset_name in radar_assets:
        for comp_name in radar_comps:
            for cycle in range(int(radar_cycle_from), int(radar_cycle_to) + 1, int(radar_cycle_step)):
                radar_combos.append((asset_name, comp_name, cycle))

    if not radar_combos:
        st.warning("Bitte mindestens ein Radar Asset und ein Comparison Asset auswaehlen.")
        st.stop()

    data_cache = {}
    radar_rows = []
    progress = st.progress(0)

    for idx, (asset_name, comp_name, cycle) in enumerate(radar_combos, start=1):
        asset_symbol = ASSET_PRESETS[asset_name]
        comp_symbol = COMPARISON_PRESETS[comp_name]

        if asset_symbol not in data_cache:
            data_cache[asset_symbol] = load_yahoo(asset_symbol)
        if comp_symbol not in data_cache:
            data_cache[comp_symbol] = load_yahoo(comp_symbol)

        asset_data = data_cache[asset_symbol]
        comp_data = data_cache[comp_symbol]
        if asset_data is None or comp_data is None:
            progress.progress(idx / len(radar_combos))
            continue

        for direction in ["Long Only", "Short Only"]:
            radar_settings = Settings(
                cycle_length=int(cycle),
                smoothing=settings.smoothing,
                softness=settings.softness,
                mode=settings.mode,
                trade_direction=direction,
                start_year=settings.start_year,
                end_year=settings.end_year,
                upper=settings.upper,
                lower=settings.lower,
                risk_pct=settings.risk_pct,
                stop_pct=settings.stop_pct,
                tp_mode=settings.tp_mode,
                rr=settings.rr,
                fixed_tp_pct=settings.fixed_tp_pct,
                exit_on_zero=settings.exit_on_zero,
                time_exit=settings.time_exit,
                exit_after_bars=settings.exit_after_bars,
                initial_capital=settings.initial_capital,
                commission_pct=settings.commission_pct,
                slippage_pct=settings.slippage_pct,
            )
            radar_df = calculate_oscillator(asset_data, comp_data, radar_settings)
            if radar_df.empty:
                continue

            latest = radar_df.iloc[-1]
            latest_date = radar_df.index[-1]
            osc = float(latest["osc"])
            side = "Long" if direction == "Long Only" else "Short"
            in_zone = osc <= settings.lower if side == "Long" else osc >= settings.upper
            near_zone = osc <= settings.lower + radar_near_zone if side == "Long" else osc >= settings.upper - radar_near_zone
            if not near_zone:
                continue

            _, _, radar_metrics = backtest(radar_df, radar_settings)
            trades = radar_metrics["Trades"]
            max_loss_streak = radar_metrics["Max Loss Streak"]
            pf = radar_metrics["Profit Factor"]
            if trades < int(radar_min_trades) or max_loss_streak > int(radar_max_loss_streak) or pd.isna(pf):
                continue

            signal_status = "In Extremzone" if in_zone else "Nahe Extremzone"
            radar_score = (
                float(pf) * 2.0
                + float(radar_metrics["Expectancy R"] or 0) * 1.5
                + float(radar_metrics["Winrate"] or 0) / 100
                - float(max_loss_streak) * 0.08
                - abs(float(radar_metrics["Max DD"] or 0)) * 0.03
            )
            radar_rows.append({
                "Asset": asset_name.split(" proxy:")[0].replace(" proxy", ""),
                "Comparison": comp_name.split(" proxy:")[0].replace(" futures", ""),
                "Asset Symbol": asset_symbol,
                "Comparison Symbol": comp_symbol,
                "Signal": side,
                "Status": signal_status,
                "Latest Date": latest_date.date(),
                "Current Osc": osc,
                "Cycle": int(cycle),
                "Trades": trades,
                "Winrate": radar_metrics["Winrate"],
                "Profit Factor": pf,
                "Net Profit": radar_metrics["Net Profit"],
                "Max DD": radar_metrics["Max DD"],
                "Expectancy R": radar_metrics["Expectancy R"],
                "Max Loss Streak": max_loss_streak,
                "Avg Realized Win": radar_metrics["Avg Realized Win"],
                "Avg Realized Loss": radar_metrics["Avg Realized Loss"],
                "Radar Score": radar_score,
            })
        progress.progress(idx / len(radar_combos))

    radar_results = pd.DataFrame(radar_rows)
    if radar_results.empty:
        st.warning(
            "Aktuell keine Radar-Signale mit deinen Filtern. Du kannst links Near Zone Buffer erhoehen, "
            "Radar Min Trades senken oder Max Loss Streak etwas erhoehen."
        )
        st.stop()

    top_signal_rows = []
    grouped = radar_results.groupby(["Asset", "Comparison", "Signal", "Status"], dropna=False)
    for keys, group in grouped:
        top_group = group.sort_values(["Radar Score", "Profit Factor", "Net Profit"], ascending=[False, False, False]).head(int(radar_top_cycles))
        best = top_group.iloc[0]
        top_signal_rows.append({
            "Asset": keys[0],
            "Comparison": keys[1],
            "Signal": keys[2],
            "Status": keys[3],
            "Current Osc": best["Current Osc"],
            "Best Cycle": int(best["Cycle"]),
            "Best Winrate": best["Winrate"],
            "Best PF": best["Profit Factor"],
            "Best Max DD": best["Max DD"],
            "Best Loss Streak": int(best["Max Loss Streak"]),
            "Top Cycles": ", ".join(str(int(c)) for c in top_group["Cycle"].tolist()),
            "Latest Date": best["Latest Date"],
            "Radar Score": best["Radar Score"],
        })

    signal_overview = pd.DataFrame(top_signal_rows).sort_values(["Status", "Radar Score"], ascending=[True, False])
    st.subheader("Aktuelle TACO Radar Signale")
    st.dataframe(signal_overview.drop(columns=["Radar Score"]), use_container_width=True)

    st.subheader("Cycle-Vorschlaege pro Signal")
    for _, signal in signal_overview.iterrows():
        signal_group = radar_results[
            (radar_results["Asset"] == signal["Asset"])
            & (radar_results["Comparison"] == signal["Comparison"])
            & (radar_results["Signal"] == signal["Signal"])
            & (radar_results["Status"] == signal["Status"])
        ].sort_values(["Radar Score", "Profit Factor", "Net Profit"], ascending=[False, False, False]).head(int(radar_top_cycles))

        title = (
            f"{signal['Asset']} vs {signal['Comparison']} | {signal['Signal']} | "
            f"{signal['Status']} | Osc {signal['Current Osc']:.1f}"
        )
        with st.expander(title, expanded=False):
            st.dataframe(
                signal_group[[
                    "Cycle",
                    "Trades",
                    "Winrate",
                    "Profit Factor",
                    "Net Profit",
                    "Max DD",
                    "Expectancy R",
                    "Max Loss Streak",
                    "Avg Realized Win",
                    "Avg Realized Loss",
                ]],
                use_container_width=True,
            )

    csv = radar_results.to_csv(index=False).encode("utf-8")
    st.download_button("Radar Ergebnisse als CSV laden", data=csv, file_name="taco_radar.csv", mime="text/csv")
    st.stop()

if asset_df is None or comp_df is None:
    if test_mode == "Cycle Scanner" and not run_scan:
        st.info("Waehle oben Assets, Comparison Assets und Cycle Range aus. Danach auf Run Cycle Scan klicken.")
        st.stop()
    if test_mode == "Cycle Scanner" and run_scan:
        pass
    elif test_mode == "SL Scanner" and not run_sl_scan:
        st.info("Waehle oben Asset, Comparison Asset, Direction und SL Range aus. Danach auf Run SL Scan klicken.")
        st.stop()
    elif test_mode == "SL Scanner" and run_sl_scan:
        pass
    else:
        st.info("Bitte Daten laden oder Demo nutzen.")
        st.stop()

if test_mode == "SL Scanner":
    if not run_sl_scan:
        st.info("Waehle oben Asset, Comparison Asset, Direction und SL Range aus. Danach auf Run SL Scan klicken.")
        st.stop()

    if sl_to < sl_from:
        st.error("SL To muss groesser oder gleich SL From sein.")
        st.stop()

    asset_symbol = ASSET_PRESETS[sl_asset_preset]
    comp_symbol = COMPARISON_PRESETS[sl_comp_preset]
    asset_data = load_yahoo(asset_symbol)
    comp_data = load_yahoo(comp_symbol)
    if asset_data is None or comp_data is None:
        st.error("Yahoo-Daten konnten fuer den SL Scanner nicht geladen werden.")
        st.stop()

    sl_values = np.round(np.arange(float(sl_from), float(sl_to) + float(sl_step) / 2, float(sl_step)), 4)
    rows = []
    progress = st.progress(0)
    combos = [(direction, sl) for direction in sl_directions for sl in sl_values]
    if not combos:
        st.warning("Bitte mindestens eine Direction auswaehlen.")
        st.stop()

    for idx, (direction, sl) in enumerate(combos, start=1):
        sl_settings = Settings(
            cycle_length=settings.cycle_length,
            smoothing=settings.smoothing,
            softness=settings.softness,
            mode=settings.mode,
            trade_direction=direction,
            start_year=settings.start_year,
            end_year=settings.end_year,
            upper=settings.upper,
            lower=settings.lower,
            risk_pct=settings.risk_pct,
            stop_pct=float(sl),
            tp_mode=settings.tp_mode,
            rr=settings.rr,
            fixed_tp_pct=settings.fixed_tp_pct,
            exit_on_zero=settings.exit_on_zero,
            time_exit=settings.time_exit,
            exit_after_bars=settings.exit_after_bars,
            initial_capital=settings.initial_capital,
            commission_pct=settings.commission_pct,
            slippage_pct=settings.slippage_pct,
        )
        sl_df = calculate_oscillator(asset_data, comp_data, sl_settings)
        sl_trades, _, sl_metrics = backtest(sl_df, sl_settings)
        rows.append({
            "Asset": sl_asset_preset.split(" proxy:")[0].replace(" proxy", ""),
            "Comparison": sl_comp_preset.split(" proxy:")[0].replace(" futures", ""),
            "Asset Symbol": asset_symbol,
            "Comparison Symbol": comp_symbol,
            "Direction": direction,
            "Cycle": settings.cycle_length,
            "SL %": float(sl),
            "Trades": sl_metrics["Trades"],
            "Winrate": sl_metrics["Winrate"],
            "Profit Factor": sl_metrics["Profit Factor"],
            "Net Profit": sl_metrics["Net Profit"],
            "Max DD": sl_metrics["Max DD"],
            "Avg Realized Win": sl_metrics["Avg Realized Win"],
            "Avg Realized Loss": sl_metrics["Avg Realized Loss"],
            "Avg R": sl_metrics["Avg R"],
            "Expectancy R": sl_metrics["Expectancy R"],
            "Max Loss Streak": sl_metrics["Max Loss Streak"],
            "Intratrade MAE": sl_metrics["Intratrade MAE"],
            "Avg MFE": sl_metrics["Avg MFE"],
            "Stop Breach Count": sl_metrics["Stop Breach Count"],
            "Stop Breach Avg": sl_metrics["Stop Breach Avg"],
        })
        progress.progress(idx / len(combos))

    sl_results = pd.DataFrame(rows).sort_values(["Profit Factor", "Net Profit"], ascending=[False, False]).reset_index(drop=True)
    st.subheader("SL Scanner Ergebnisse")
    st.caption("Suche robuste Stop-Zonen, nicht nur den besten Einzelwert. Gute Stops bleiben oft ueber mehrere benachbarte SL-Stufen stabil.")
    st.dataframe(sl_results, use_container_width=True)

    st.subheader("Top robuste SL-Bereiche")
    robust_rows = []
    for direction, group in sl_results.dropna(subset=["Profit Factor"]).groupby("Direction"):
        group = group.sort_values("SL %")
        for _, row in group.iterrows():
            sl = row["SL %"]
            neighbors = group[group["SL %"].between(sl - sl_step, sl + sl_step)]
            if len(neighbors) >= 2:
                robust_rows.append({
                    "Direction": direction,
                    "Center SL %": sl,
                    "Neighbor Count": len(neighbors),
                    "Avg Profit Factor": neighbors["Profit Factor"].mean(),
                    "Avg Net Profit": neighbors["Net Profit"].mean(),
                    "Avg Expectancy R": neighbors["Expectancy R"].mean(),
                    "Worst Max DD": neighbors["Max DD"].min(),
                    "Avg Trades": neighbors["Trades"].mean(),
                    "Avg Realized Loss": neighbors["Avg Realized Loss"].mean(),
                })
    robust_sl = pd.DataFrame(robust_rows)
    if not robust_sl.empty:
        robust_sl = robust_sl.sort_values(["Avg Profit Factor", "Avg Expectancy R"], ascending=[False, False])
        st.dataframe(robust_sl.head(30), use_container_width=True)

    st.subheader("Ausgewaehltes SL-Setup visualisieren")
    labels = [
        f"#{idx} | {row.Direction} | SL {row['SL %']:.2f}% | PF {row['Profit Factor']:.2f} | ExpR {row['Expectancy R']:.2f} | Net {row['Net Profit']:.0f}"
        for idx, row in sl_results.head(100).iterrows()
    ]
    selected_label = st.selectbox("SL-Ergebnis anzeigen", labels)
    selected_idx = int(selected_label.split(" | ")[0].replace("#", ""))
    selected = sl_results.loc[selected_idx]
    selected_settings = Settings(
        cycle_length=int(selected["Cycle"]),
        smoothing=settings.smoothing,
        softness=settings.softness,
        mode=settings.mode,
        trade_direction=str(selected["Direction"]),
        start_year=settings.start_year,
        end_year=settings.end_year,
        upper=settings.upper,
        lower=settings.lower,
        risk_pct=settings.risk_pct,
        stop_pct=float(selected["SL %"]),
        tp_mode=settings.tp_mode,
        rr=settings.rr,
        fixed_tp_pct=settings.fixed_tp_pct,
        exit_on_zero=settings.exit_on_zero,
        time_exit=settings.time_exit,
        exit_after_bars=settings.exit_after_bars,
        initial_capital=settings.initial_capital,
        commission_pct=settings.commission_pct,
        slippage_pct=settings.slippage_pct,
    )
    selected_df = calculate_oscillator(asset_data, comp_data, selected_settings)
    selected_trades, selected_equity, selected_metrics = backtest(selected_df, selected_settings)
    metric_cols = st.columns(7)
    for col, key in zip(metric_cols, CORE_METRICS):
        val = selected_metrics[key]
        col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")
    practice_cols = st.columns(7)
    for col, key in zip(practice_cols, PRACTICE_METRICS):
        val = selected_metrics[key]
        col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")
    plot_backtest_charts(selected_df, selected_trades, selected_equity, selected_settings)
    st.subheader("Trades des ausgewaehlten SL-Setups")
    st.dataframe(selected_trades, use_container_width=True)

    csv = sl_results.to_csv(index=False).encode("utf-8")
    st.download_button("SL Scanner Ergebnisse als CSV laden", data=csv, file_name="taco_sl_scan.csv", mime="text/csv")
    st.stop()

if test_mode == "Cycle Scanner":
    if not run_scan:
        st.info("Waehle oben Assets, Comparison Assets und Cycle Range aus. Danach auf Run Cycle Scan klicken.")
        st.stop()

    if scan_cycle_to < scan_cycle_from:
        st.error("Cycle To muss groesser oder gleich Cycle From sein.")
        st.stop()

    rows = []
    progress = st.progress(0)
    combos = []
    for asset_name in scan_assets:
        for comp_name in scan_comps:
            for direction in scan_directions:
                for cycle in range(int(scan_cycle_from), int(scan_cycle_to) + 1, int(scan_cycle_step)):
                    combos.append((asset_name, comp_name, direction, cycle))

    if not combos:
        st.warning("Bitte mindestens ein Asset, ein Comparison Asset und eine Direction auswaehlen.")
        st.stop()

    data_cache = {}
    for idx, (asset_name, comp_name, direction, cycle) in enumerate(combos, start=1):
        asset_symbol = ASSET_PRESETS[asset_name]
        comp_symbol = COMPARISON_PRESETS[comp_name]

        if asset_symbol not in data_cache:
            data_cache[asset_symbol] = load_yahoo(asset_symbol)
        if comp_symbol not in data_cache:
            data_cache[comp_symbol] = load_yahoo(comp_symbol)

        asset_data = data_cache[asset_symbol]
        comp_data = data_cache[comp_symbol]
        if asset_data is None or comp_data is None:
            progress.progress(idx / len(combos))
            continue

        scan_settings = Settings(
            cycle_length=int(cycle),
            smoothing=settings.smoothing,
            softness=settings.softness,
            mode=settings.mode,
            trade_direction=direction,
            start_year=settings.start_year,
            end_year=settings.end_year,
            upper=settings.upper,
            lower=settings.lower,
            risk_pct=settings.risk_pct,
            stop_pct=settings.stop_pct,
            tp_mode=settings.tp_mode,
            rr=settings.rr,
            fixed_tp_pct=settings.fixed_tp_pct,
            exit_on_zero=settings.exit_on_zero,
            time_exit=settings.time_exit,
            exit_after_bars=settings.exit_after_bars,
            initial_capital=settings.initial_capital,
            commission_pct=settings.commission_pct,
            slippage_pct=settings.slippage_pct,
        )

        scan_df = calculate_oscillator(asset_data, comp_data, scan_settings)
        scan_trades, _, scan_metrics = backtest(scan_df, scan_settings)
        rows.append({
            "Asset": asset_name.split(" proxy:")[0].replace(" proxy", ""),
            "Comparison": comp_name.split(" proxy:")[0].replace(" futures", ""),
            "Asset Symbol": asset_symbol,
            "Comparison Symbol": comp_symbol,
            "Direction": direction,
            "Cycle": cycle,
            "Trades": scan_metrics["Trades"],
            "Winrate": scan_metrics["Winrate"],
            "Profit Factor": scan_metrics["Profit Factor"],
            "Net Profit": scan_metrics["Net Profit"],
            "Max DD": scan_metrics["Max DD"],
            "Expectancy R": scan_metrics["Expectancy R"],
            "Max Loss Streak": scan_metrics["Max Loss Streak"],
            "Avg Realized Win": scan_metrics["Avg Realized Win"],
            "Avg Realized Loss": scan_metrics["Avg Realized Loss"],
            "Avg R": scan_metrics["Avg R"],
            "Intratrade MAE": scan_metrics["Intratrade MAE"],
            "Avg MFE": scan_metrics["Avg MFE"],
            "Stop Breach Count": scan_metrics["Stop Breach Count"],
            "Stop Breach Avg": scan_metrics["Stop Breach Avg"],
        })
        progress.progress(idx / len(combos))

    results = pd.DataFrame(rows)
    if results.empty:
        st.error("Keine Scanner-Ergebnisse. Pruefe Datenquelle oder Symbole.")
        st.stop()

    results = results.sort_values(["Profit Factor", "Net Profit"], ascending=[False, False])
    results = results.reset_index(drop=True)
    st.subheader("Cycle Scanner Ergebnisse")
    st.caption("Sortiere nicht nur nach dem besten Einzelwert. Suche stabile Cluster ueber mehrere benachbarte Cycles.")
    st.dataframe(results, use_container_width=True)

    st.subheader("Top robuste Bereiche")
    robust_rows = []
    grouped = results.dropna(subset=["Profit Factor"]).groupby(["Asset", "Comparison", "Direction"])
    for keys, group in grouped:
        group = group.sort_values("Cycle")
        for _, row in group.iterrows():
            cycle = row["Cycle"]
            neighbors = group[group["Cycle"].between(cycle - scan_cycle_step, cycle + scan_cycle_step)]
            if len(neighbors) >= 2:
                robust_rows.append({
                    "Asset": keys[0],
                    "Comparison": keys[1],
                    "Direction": keys[2],
                    "Center Cycle": cycle,
                    "Neighbor Count": len(neighbors),
                    "Avg Profit Factor": neighbors["Profit Factor"].mean(),
                    "Avg Net Profit": neighbors["Net Profit"].mean(),
                    "Worst Max DD": neighbors["Max DD"].min(),
                    "Avg Trades": neighbors["Trades"].mean(),
                })
    robust = pd.DataFrame(robust_rows)
    if not robust.empty:
        robust = robust.sort_values(["Avg Profit Factor", "Avg Net Profit"], ascending=[False, False])
        st.dataframe(robust.head(30), use_container_width=True)

    st.subheader("Top 5 Scanner Equity-Kurven")
    st.caption(
        "Diese Ansicht filtert zuerst nach Mindestanzahl Trades und maximaler Losing Streak. "
        "Danach werden die besten Setups nach Profit Factor, Expectancy R und Net Profit als Equity-Kurven angezeigt."
    )
    top_candidates = results[
        (results["Trades"] >= int(top_curve_min_trades))
        & (results["Max Loss Streak"] <= int(top_curve_max_loss_streak))
        & results["Profit Factor"].notna()
    ].copy()
    if top_candidates.empty:
        st.info("Keine Top-5-Kurven fuer diese Filter. Senke links Min Trades oder erhoehe Max Loss Streak.")
    else:
        top_candidates["Top Score"] = (
            top_candidates["Profit Factor"].fillna(0) * 2.0
            + top_candidates["Expectancy R"].fillna(0) * 1.5
            + top_candidates["Winrate"].fillna(0) / 100
            - top_candidates["Max Loss Streak"].fillna(0) * 0.08
            - top_candidates["Max DD"].abs().fillna(0) * 0.03
        )
        top5 = top_candidates.sort_values(["Top Score", "Net Profit"], ascending=[False, False]).head(5).reset_index(drop=True)
        st.dataframe(
            top5[[
                "Asset",
                "Comparison",
                "Direction",
                "Cycle",
                "Trades",
                "Winrate",
                "Profit Factor",
                "Net Profit",
                "Max DD",
                "Expectancy R",
                "Max Loss Streak",
            ]],
            use_container_width=True,
        )

        equity_fig = go.Figure()
        for _, row in top5.iterrows():
            curve_settings = Settings(
                cycle_length=int(row["Cycle"]),
                smoothing=settings.smoothing,
                softness=settings.softness,
                mode=settings.mode,
                trade_direction=str(row["Direction"]),
                start_year=settings.start_year,
                end_year=settings.end_year,
                upper=settings.upper,
                lower=settings.lower,
                risk_pct=settings.risk_pct,
                stop_pct=settings.stop_pct,
                tp_mode=settings.tp_mode,
                rr=settings.rr,
                fixed_tp_pct=settings.fixed_tp_pct,
                exit_on_zero=settings.exit_on_zero,
                time_exit=settings.time_exit,
                exit_after_bars=settings.exit_after_bars,
                initial_capital=settings.initial_capital,
                commission_pct=settings.commission_pct,
                slippage_pct=settings.slippage_pct,
            )
            curve_asset = data_cache[row["Asset Symbol"]]
            curve_comp = data_cache[row["Comparison Symbol"]]
            curve_df = calculate_oscillator(curve_asset, curve_comp, curve_settings)
            _, curve_equity, _ = backtest(curve_df, curve_settings)
            if curve_equity.empty:
                continue
            equity_fig.add_trace(
                go.Scatter(
                    x=curve_equity.index,
                    y=curve_equity["equity"],
                    mode="lines",
                    name=f"{row['Asset']} / {row['Comparison']} / {row['Direction']} / Cycle {int(row['Cycle'])}",
                )
            )
        equity_fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Equity",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(equity_fig, use_container_width=True)

    st.subheader("Ausgewaehltes Scanner-Setup visualisieren")
    st.caption("Waehle ein Ergebnis aus der Scanner-Tabelle aus. Danach wird es wie ein manueller Backtest mit Chart, Oszillator, Equity und Trades angezeigt.")
    labels = [
        f"#{idx} | {row.Asset} | {row.Comparison} | {row.Direction} | Cycle {int(row.Cycle)} | PF {row['Profit Factor']:.2f} | Net {row['Net Profit']:.0f}"
        for idx, row in results.head(100).iterrows()
    ]
    selected_label = st.selectbox("Scanner-Ergebnis anzeigen", labels)
    selected_idx = int(selected_label.split(" | ")[0].replace("#", ""))
    selected = results.loc[selected_idx]

    selected_asset_data = data_cache[selected["Asset Symbol"]]
    selected_comp_data = data_cache[selected["Comparison Symbol"]]
    selected_settings = Settings(
        cycle_length=int(selected["Cycle"]),
        smoothing=settings.smoothing,
        softness=settings.softness,
        mode=settings.mode,
        trade_direction=str(selected["Direction"]),
        start_year=settings.start_year,
        end_year=settings.end_year,
        upper=settings.upper,
        lower=settings.lower,
        risk_pct=settings.risk_pct,
        stop_pct=settings.stop_pct,
        tp_mode=settings.tp_mode,
        rr=settings.rr,
        fixed_tp_pct=settings.fixed_tp_pct,
        exit_on_zero=settings.exit_on_zero,
        time_exit=settings.time_exit,
        exit_after_bars=settings.exit_after_bars,
        initial_capital=settings.initial_capital,
        commission_pct=settings.commission_pct,
        slippage_pct=settings.slippage_pct,
    )
    selected_df = calculate_oscillator(selected_asset_data, selected_comp_data, selected_settings)
    selected_trades, selected_equity, selected_metrics = backtest(selected_df, selected_settings)
    metric_cols = st.columns(7)
    for col, key in zip(metric_cols, CORE_METRICS):
        val = selected_metrics[key]
        col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")
    practice_cols = st.columns(7)
    for col, key in zip(practice_cols, PRACTICE_METRICS):
        val = selected_metrics[key]
        col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")
    plot_backtest_charts(selected_df, selected_trades, selected_equity, selected_settings)
    st.subheader("Trades des ausgewaehlten Scanner-Setups")
    st.dataframe(selected_trades, use_container_width=True)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Scanner Ergebnisse als CSV laden", data=csv, file_name="taco_cycle_scan.csv", mime="text/csv")
    st.stop()

if asset_df is None or comp_df is None:
    st.info("Bitte Daten laden oder Demo nutzen.")
    st.stop()

df = calculate_oscillator(asset_df, comp_df, settings)
trades, equity, metrics = backtest(df, settings)

cols = st.columns(7)
for col, key in zip(cols, CORE_METRICS):
    val = metrics[key]
    col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")

practice_cols = st.columns(7)
for col, key in zip(practice_cols, PRACTICE_METRICS):
    val = metrics[key]
    col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")

plot_backtest_charts(df, trades, equity, settings)

st.subheader("Trades")
st.dataframe(trades, use_container_width=True)
