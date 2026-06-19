import math
import calendar
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
        normalized = year_df["close"] / year_df["close"].iloc[0] * 100
        frames.append(
            pd.DataFrame(
                {
                    "month": year_df.index.month,
                    "day": year_df.index.day,
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

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            return "__quota__"
        return f"__error__{msg}"


def render_ki_analyse(asset_label: str, symbol: str) -> None:
    api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

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

    with st.spinner(f"Claude durchsucht das Web nach aktuellen Daten zu {asset_name}..."):
        result = _fetch_ki_analyse(symbol, asset_name, api_key)

    if result == "__quota__":
        st.warning("⏳ Gemini Free-Tier Limit erreicht — bitte 1 Minute warten und dann erneut auf '🔄 Jetzt aktualisieren' klicken.")
        return
    if str(result).startswith("__error__"):
        st.error(f"API-Fehler: {result.replace('__error__', '')}")
        return

    result_lower = result.lower()
    last_200 = result_lower[-200:]
    if "bullish" in last_200:
        bias, bias_color = "BULLISH", "#62c8e8"
    elif "bearish" in last_200:
        bias, bias_color = "BEARISH", "#c25f50"
    else:
        bias, bias_color = "NEUTRAL", "#94a3b8"

    st.markdown(
        f"""
        <div style="background:#0d1520;border:1px solid rgba(148,163,184,.10);border-radius:8px;
                    padding:20px 22px 16px 22px;margin:4px 0 16px 0;position:relative;">
            <span style="position:absolute;top:16px;right:16px;font-size:.70rem;font-weight:700;
                         color:{bias_color};background:rgba(0,0,0,.4);
                         border:1px solid {bias_color}55;border-radius:4px;padding:3px 10px;">
                {bias}
            </span>
            <div style="font-size:.84rem;color:#cbd5e1;line-height:1.8;padding-right:80px;">
                {result}
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


def _compute_stars(wr: float, avg_ret: float, avg_dd: float, max_dd: float,
                   sharpe: float, robustheit: str, atr_pct: float) -> int:
    """1–5 Sterne — Robustheit setzt harte Obergrenze, Rest sind Qualitätspunkte."""
    # Harte Obergrenzen durch Robustheit
    rob_cap = {"🟢 Stark": 5, "✅ Robust": 4, "⚠️ Sensitiv": 3, "❌ Fragil": 2, "—": 3}
    max_stars = rob_cap.get(robustheit, 3)

    score = 0.0
    # Winrate (0–35 Punkte)
    score += min(35, max(0, (wr - 0.60) / 0.40 * 35))
    # Avg Profit vs ATR (0–25 Punkte)
    if atr_pct and atr_pct > 0:
        score += min(25, max(0, (avg_ret * 100 / atr_pct) / 2.0 * 25))
    else:
        score += min(25, max(0, avg_ret * 100 / 3.0 * 25))
    # Avg DD (0–20 Punkte): wenig DD = gut
    score += min(20, max(0, (1 - abs(avg_dd) / 0.05) * 20))
    # Sharpe (0–20 Punkte)
    if not np.isnan(sharpe):
        score += min(20, max(0, sharpe / 3.0 * 20))

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

    def _stats_for_years(yr_start: int, dir_: str) -> dict | None:
        sub = [t for t in trades if t["yr"] >= yr_start]
        if len(sub) < min_trades:
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

        wr_5j  = (_stats_for_years(y5,  dir_) or {}).get("wr", np.nan)
        wr_10j = (_stats_for_years(y10, dir_) or {}).get("wr", np.nan)
        wr_15j = (_stats_for_years(y15, dir_) or {}).get("wr", np.nan)
        # Für 20J: wenn nicht genug Daten, alle verfügbaren Jahre nehmen
        wr_20j_raw = (_stats_for_years(y20, dir_) or {}).get("wr", np.nan)
        if not has_20j or np.isnan(wr_20j_raw):
            wr_20j_raw = (_stats_for_years(data_start_year, dir_) or {}).get("wr", np.nan)
        wr_20j = round(wr_20j_raw * 100, 1) if not np.isnan(wr_20j_raw) else np.nan
        wr_5j  = round(wr_5j  * 100, 1) if has_5j  and not np.isnan(wr_5j)  else np.nan
        wr_10j = round(wr_10j * 100, 1) if has_10j and not np.isnan(wr_10j) else np.nan
        wr_15j = round(wr_15j * 100, 1) if has_15j and not np.isnan(wr_15j) else np.nan

        # Robustheit: benachbarte Entry-DOYs +3..+7 testen
        robust_wins = robust_total = 0
        for offset in range(3, 8):
            alt_entry = entry_doy + offset
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
            if _rob_ratio >= 0.80:   robustheit = "🟢 Stark"
            elif _rob_ratio >= 0.60: robustheit = "✅ Robust"
            elif _rob_ratio >= 0.40: robustheit = "⚠️ Sensitiv"
            else:                    robustheit = "❌ Fragil"

        avg_atr_pct = float(np.mean([t["atr"] / t["ep"] for t in primary_trades if t["ep"] > 0]) * 100)

        # Entry/Exit Label aus DOY
        try:
            _ref = pd.Timestamp(year=2000, month=1, day=1)  # Schaltjahr → DOY stimmt mit Kalender überein
            entry_label = (_ref + pd.Timedelta(days=entry_doy - 1)).strftime("%d. %b")
            exit_label  = (_ref + pd.Timedelta(days=exit_doy  - 1)).strftime("%d. %b")
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
            "⭐ Rating": _compute_stars(wr, avg_ret, avg_dd, max_dd_val, sharpe, robustheit, avg_atr_pct),
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

    if st.button("← Zurück zum Scanner"):
        st.session_state.pop("muster_detail", None)
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

    st.markdown(
        f"""<div style="background:#0d1520;border:1px solid rgba(148,163,184,.15);
        border-radius:10px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
          <span style="color:#fff;font-size:1.6rem;font-weight:900;letter-spacing:.02em;">{symbol_str}</span>
          <span style="background:{farbe}22;border:1px solid {farbe}55;border-radius:5px;
            padding:4px 12px;color:{farbe};font-weight:800;font-size:1rem;">{pfeil} {richtung}</span>
          <span style="color:#6b7fa3;font-size:1rem;">📅 {row['Entry']} → {row['Exit']} &nbsp;·&nbsp; ⏱ {row['Haltedauer (TD)']} Handelstage</span>
          <span style="margin-left:auto;background:{star_clr}18;border:1px solid {star_clr}44;
            border-radius:8px;padding:5px 14px;font-size:1.15rem;letter-spacing:2px;"
            title="{stars}/5 Sterne">{star_str}</span>
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

    df_sym = saved_dfs.get(symbol_str)
    if df_sym is None or df_sym.empty:
        st.warning("Kein DataFrame verfügbar — Scanner nochmal starten.")
        return

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
        if len(sub) < 2:
            return None
        avg_net = sub["Netto Return %"].mean()
        avg_dd  = abs(sub["Max DD %"].mean())
        ratio   = avg_net / avg_dd if avg_dd > 0 else float("inf")
        return {"avg_net": avg_net, "ratio": ratio, "wins": (sub["Return %"] > 0).sum(), "n": len(sub)}

    def _ampel_html(label, d):
        if d is None:
            return (f'<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);'                    f'border-radius:8px;padding:14px 18px;">'                    f'<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;'                    f'letter-spacing:.06em;margin-bottom:4px;">Netto-Edge {label}</div>'                    f'<div style="color:#475569;">—</div></div>')
        if d["ratio"] >= 1.5 and d["avg_net"] > 0:
            sym, farbe, txt = "🟢", "#4ade80", "Robust positiv"
        elif d["ratio"] >= 0.8 and d["avg_net"] > 0:
            sym, farbe, txt = "🟡", "#f0c040", "Knapp / Break-even"
        else:
            sym, farbe, txt = "🔴", "#f87171", "Verlierer nach Kosten"
        return (f'<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);'                f'border-radius:8px;padding:14px 18px;">'                f'<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;'                f'letter-spacing:.06em;margin-bottom:4px;">Netto-Edge {label}</div>'                f'<div style="color:{farbe};font-size:1.05rem;font-weight:700;margin-bottom:3px;">'                f'{sym} {txt}</div>'                f'<div style="color:#9fb0c7;font-size:.8rem;">Ø netto {d["avg_net"]:+.2f}% &nbsp;·&nbsp; '                f'Ratio {d["ratio"]:.2f} &nbsp;·&nbsp; {d["wins"]}W/{d["n"]-d["wins"]}L</div></div>')

    _badges = "".join([
        _ampel_html("5J",  _fenster_data(tdf_5)),
        _ampel_html("10J", _fenster_data(tdf_10)),
        _ampel_html("15J", _fenster_data(tdf_15)),
        _ampel_html("20J", _fenster_data(tdf_20)),
    ])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">'        f'{_badges}</div>',
        unsafe_allow_html=True,
    )

    # ── Stop-Loss-Empfehlung ─────────────────────────────────────────────────
    st.markdown("### 🛡️ Stop-Loss-Empfehlung")

    _dd_abs = tdf["Max DD %"].abs()
    _sl_perc = float(np.percentile(_dd_abs, _perc_pct)) + _kostenpuffer
    _sl_std  = float(_dd_abs.mean() + _stddev_k * _dd_abs.std(ddof=1)) + _kostenpuffer

    _atr_pct_vals = []
    try:
        _h = df_sym["high"].to_numpy(); _l = df_sym["low"].to_numpy(); _c = df_sym["close"].to_numpy()
        _n = len(_c); _tr = np.zeros(_n); _tr[0] = _h[0] - _l[0]
        for _i in range(1, _n):
            _tr[_i] = max(_h[_i]-_l[_i], abs(_h[_i]-_c[_i-1]), abs(_l[_i]-_c[_i-1]))
        _atr_w = np.full(_n, np.nan)
        if _n >= 14:
            _atr_w[13] = _tr[:14].mean()
            for _i in range(14, _n): _atr_w[_i] = (_atr_w[_i-1] * 13 + _tr[_i]) / 14
        _atr_s = pd.Series(_atr_w, index=df_sym.index)
        for _, _r in tdf.iterrows():
            _ets = _r["_entry_ts"]; _ep = _r["_entry_price"]
            if _ets in _atr_s.index and not np.isnan(_atr_s[_ets]) and _ep > 0:
                _atr_pct_vals.append(_atr_s[_ets] / _ep * 100)
    except Exception:
        pass

    _atr_ok = len(_atr_pct_vals) > 0
    _sl_atr  = float(np.mean(_atr_pct_vals)) * _atr_mult + _kostenpuffer if _atr_ok else 0.0
    _methoden = {"Perzentil": _sl_perc, "Avg+StdDev": _sl_std}
    if _atr_ok: _methoden["ATR"] = _sl_atr
    _empf = max(_methoden, key=_methoden.get)

    if _empf == "ATR":
        _begr = f"ATR-Stop am größten (Ø {float(np.mean(_atr_pct_vals)):.2f}% × {_atr_mult}×) — aktuelle Volatilität übersteigt die historischen DD-Quantile."
    elif _empf == "Avg+StdDev":
        _begr = f"Avg+StdDev-Stop am größten — Ausreißer-DD {_dd_abs.max():.2f}% hebt σ={_dd_abs.std(ddof=1):.2f}% stark an."
    else:
        _begr = f"Perzentil-Stop am größten — {_perc_pct}. Perzentil des hist. DD ({float(np.percentile(_dd_abs, _perc_pct)):.2f}%) übertrifft andere Methoden."

    def _sl_card(name, val, empf):
        border = "border:2px solid #4ade80;" if empf else "border:1px solid rgba(148,163,184,.15);"
        tag = ('<div style="display:inline-block;background:#14532d;color:#4ade80;font-size:.7rem;'               'font-weight:700;border-radius:4px;padding:2px 8px;margin-bottom:6px;">✅ Empfohlen</div>' if empf else "")
        farbe = "#4ade80" if empf else "#9fb0c7"
        return (f'<div style="background:#0a1220;{border}border-radius:8px;padding:16px 20px;">'                f'{tag}<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;'                f'letter-spacing:.06em;margin-bottom:4px;">{name}</div>'                f'<div style="color:{farbe};font-size:1.2rem;font-weight:700;">{val:.2f}%</div>'                f'<div style="color:#475569;font-size:.75rem;margin-top:3px;">'                f'inkl. Kostenpuffer {_kostenpuffer:.2f}%</div></div>')

    _cards = [_sl_card("Perzentil", _sl_perc, _empf == "Perzentil"),
              _sl_card("Avg + StdDev", _sl_std, _empf == "Avg+StdDev")]
    if _atr_ok:
        _cards.append(_sl_card("ATR", _sl_atr, _empf == "ATR"))
    else:
        _cards.append('<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);'                      'border-radius:8px;padding:16px 20px;">'                      '<div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;'                      'letter-spacing:.06em;margin-bottom:4px;">ATR</div>'                      '<div style="color:#475569;">keine ATR-Daten verfügbar</div></div>')

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({len(_cards)},1fr);'        f'gap:14px;margin-bottom:10px;">{"".join(_cards)}</div>'        f'<div style="color:#6b7fa3;font-size:.82rem;margin-bottom:22px;">💡 {_begr}</div>',
        unsafe_allow_html=True,
    )

    # Tabelle — full width
    st.markdown("### 📋 Per-Jahr Ergebnisse")

    def _color_ret(v):
        return "color:#4ade80;font-weight:600" if v > 0 else "color:#f87171;font-weight:600"
    def _color_dd(v):
        return "color:#f87171" if v < 0 else "color:#9fb0c7"

    _tdf_show = tdf.drop(columns=["_entry_ts", "_entry_price"], errors="ignore")
    st.dataframe(
        _tdf_show.style
        .map(_color_ret, subset=["Return %", "Netto Return %", "Max MFE %"])
        .map(_color_dd, subset=["Max DD %"])
        .format({"Return %": "{:+.2f}%", "Netto Return %": "{:+.2f}%",
                 "Max DD %": "{:+.2f}%", "Max MFE %": "{:+.2f}%"}),
        use_container_width=True,
        height=min(40 * len(tdf) + 42, 700),
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

        # ── ATR-Bewertung ─────────────────────────────────────────────────────
        _avg_ret_abs = abs(tdf["Return %"].mean())
        _ratio_atr   = _atr_avg / _avg_ret_abs if _avg_ret_abs > 0 else float("inf")

        # Trend: letzte 3 Jahre vs. Gesamtdurchschnitt
        _recent_3 = _atr_df.tail(3)["ATR %"].mean() if len(_atr_df) >= 3 else _atr_avg
        _trend_up  = _recent_3 > _atr_avg * 1.25  # letzten 3J deutlich über Ø

        # Basisrating nach ATR/Profit-Ratio
        if _ratio_atr < 0.5:
            _sterne = 5
            _rating_txt = "Exzellent — ATR sehr gering im Verhältnis zum Profit"
            _rating_farbe = "#4ade80"
        elif _ratio_atr < 1.0:
            _sterne = 4
            _rating_txt = "Gut — ATR kontrollierbar, Pattern klar dominant"
            _rating_farbe = "#86efac"
        elif _ratio_atr < 1.5:
            _sterne = 3
            _rating_txt = "Mittel — ATR in Höhe des Profits, Stop-Loss kritisch"
            _rating_farbe = "#f0c040"
        elif _ratio_atr < 2.5:
            _sterne = 2
            _rating_txt = "Schwach — ATR dominiert Profit, Stop-Loss Risiko hoch"
            _rating_farbe = "#fb923c"
        else:
            _sterne = 1
            _rating_txt = "Kritisch — ATR zu groß, Pattern kaum handelbar ohne weiten Stop"
            _rating_farbe = "#f87171"

        # Trend-Malus: steigende Volatilität in den letzten 3 Jahren
        if _trend_up and _sterne > 1:
            _sterne -= 1
            _trend_hinweis = (f" ⚠️ Volatilität zuletzt gestiegen (Ø letzte 3J: {_recent_3:.3f}% "
                              f"vs. Ø gesamt: {_atr_avg:.3f}%) — ein Stern Abzug.")
        else:
            _trend_hinweis = ""

        _sterne_str = "★" * _sterne + "☆" * (5 - _sterne)

        _atr_erklaerung = f"""
<div style="background:#0a1220;border:1px solid rgba(148,163,184,.12);border-radius:10px;
padding:20px 24px;margin-top:16px;">

  <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
    <div style="color:#60a5fa;font-size:1.05rem;font-weight:700;">📖 ATR — Was bedeutet das für dieses Muster?</div>
    <div style="color:{_rating_farbe};font-size:1.4rem;letter-spacing:2px;">{_sterne_str}</div>
    <div style="color:{_rating_farbe};font-size:.9rem;font-weight:700;">{_sterne}/5 — {_rating_txt}{_trend_hinweis}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
    <div>
      <div style="color:#94a3b8;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Was ist ATR?</div>
      <div style="color:#cbd5e1;font-size:.88rem;line-height:1.6;">
        Der <b>Average True Range (ATR-14)</b> misst die durchschnittliche Tages­volatilität
        über 14 Handelstage vor dem Entry. Er zeigt, wie viel sich der Kurs typischerweise
        <i>innerhalb eines Tages</i> bewegt (High minus Low, bereinigt um Over­night-Gaps).
        Je höher der ATR, desto mehr Spielraum braucht ein Stop-Loss — und desto
        schwieriger ist es, das Pattern ohne vorzeitigen Stop-Out zu handeln.
      </div>
    </div>
    <div>
      <div style="color:#94a3b8;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Bewertungslogik</div>
      <div style="color:#cbd5e1;font-size:.88rem;line-height:1.6;">
        Entscheidend ist das <b>ATR-zu-Profit-Verhältnis</b>: Wie groß ist die tägliche
        Schwankung im Vergleich zum erwarteten Muster­gewinn?<br><br>
        <span style="color:#4ade80;">✅ ATR/Profit &lt; 0.5×</span> → Pattern dominiert klar über Rauschen (5★)<br>
        <span style="color:#86efac;">✅ ATR/Profit 0.5–1.0×</span> → kontrollierbar, guter Edge (4★)<br>
        <span style="color:#f0c040;">⚠️ ATR/Profit 1.0–1.5×</span> → Stop-Loss Sizing kritisch (3★)<br>
        <span style="color:#fb923c;">⛔ ATR/Profit 1.5–2.5×</span> → Rauschen dominiert (2★)<br>
        <span style="color:#f87171;">🚫 ATR/Profit &gt; 2.5×</span> → kaum handelbar (1★)
      </div>
    </div>
  </div>

  <div style="background:#060c16;border-radius:7px;padding:12px 16px;display:flex;gap:24px;flex-wrap:wrap;">
    <div>
      <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">Ø ATR (20J)</div>
      <div style="color:#60a5fa;font-size:1rem;font-weight:700;">{_atr_avg:.3f}%</div>
    </div>
    <div>
      <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">Ø Return (brutto)</div>
      <div style="color:#4ade80;font-size:1rem;font-weight:700;">{_avg_ret_abs:.2f}%</div>
    </div>
    <div>
      <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">ATR / Profit-Ratio</div>
      <div style="color:{_rating_farbe};font-size:1rem;font-weight:700;">{_ratio_atr:.2f}×</div>
    </div>
    <div>
      <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">Volatilitäts-Trend</div>
      <div style="color:{"#f87171" if _trend_up else "#4ade80"};font-size:1rem;font-weight:700;">{"↑ steigend" if _trend_up else "→ stabil"}</div>
    </div>
    <div>
      <div style="color:#6b7fa3;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;">Ø ATR letzte 3J</div>
      <div style="color:#60a5fa;font-size:1rem;font-weight:700;">{_recent_3:.3f}%</div>
    </div>
  </div>

</div>
"""
        st.markdown(_atr_erklaerung, unsafe_allow_html=True)


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
        hold_max = st.number_input("Musterlänge max (Kalendertage)", 1, 120, 28)
        hold_step = st.number_input("Schritt", 1, 10, 1)
        holding_periods = list(range(int(hold_min), int(hold_max) + 1, int(hold_step)))
        run_scan = st.button("🔍 Scanner starten", type="primary", use_container_width=True)

    with col_main:
        if daten_modus == "Repo (permanent)" and not selected_symbols:
            st.info("Bitte wähle mindestens ein Symbol aus und starte den Scanner.")
            return
        if daten_modus == "CSV Upload" and not csv_files:
            st.info("Bitte lade eine oder mehrere Pepperstone CSV-Dateien hoch und starte den Scanner.")
            return

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
            monat_filter = st.selectbox("Monat", _monate, index=0, label_visibility="visible")
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

        # ── Top Setups aktueller Monat ──────────────────────────────────────
        import datetime as _dt
        current_month = _dt.date.today().month
        current_month_name = [
            "", "Januar", "Februar", "März", "April", "Mai", "Juni",
            "Juli", "August", "September", "Oktober", "November", "Dezember"
        ][current_month]

        result_clean = result.drop(columns=["_sort_key"], errors="ignore")
        month_mask = result_clean["Entry"].apply(
            lambda e: _month_map.get(str(e).strip().split(".")[-1].strip()[:3], 0) == current_month
        )
        _sort_top = _wr_primary_col if _wr_primary_col and _wr_primary_col in result_clean.columns else result_clean.columns[0]
        _top_raw = (
            result_clean[month_mask]
            .sort_values(_sort_top, ascending=False)
            .head(15)
        )
        # Nach Symbol gruppieren — alle Einträge desselben Symbols zusammen
        top_month = (
            _top_raw
            .sort_values(["Symbol", _sort_top], ascending=[True, False])
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

        saved_dfs = st.session_state.get("muster_dataframes", {})

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
        if data_source_label == "Pepperstone MT5 CSV":
            symbol = mt5_base_symbol
            st.text_input("MT5 CSV Symbol", value=mt5_base_symbol, disabled=True)
        else:
            symbol = st.text_input("Yahoo Symbol", key="seasonality_symbol")

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
    def fmt_stat(value: float, suffix: str = "", digits: int = 2) -> str:
        if pd.isna(value):
            return "n/a"
        if digits == 0:
            return f"{value:,.0f}{suffix}"
        return f"{value:,.{digits}f}{suffix}"

    if len(trades) > 0:
        rise_probability = gains_count / len(trades) * 100
        fall_probability = losses_count / len(trades) * 100
        flat_count = max(len(trades) - gains_count - losses_count, 0)
        flat_probability = flat_count / len(trades) * 100
    else:
        rise_probability = fall_probability = flat_probability = np.nan
        flat_count = 0
    if pd.isna(rise_probability) or pd.isna(fall_probability):
        dominant_probability = np.nan
        dominant_label = "n/a"
    else:
        dominant_probability = rise_probability if rise_probability >= fall_probability else fall_probability
        dominant_label = "Rise" if rise_probability >= fall_probability else "Fall"
    dominant_color = "#62c8e8" if dominant_label == "Rise" else "#c25f50"
    streak_label = f"{current_streak} {current_side}" if current_streak else "0"
    with stat_col:
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
        st.plotly_chart(donut, width="stretch", config={"displayModeBar": False})
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

    if trades.empty:
        st.warning("Der gewaehlte saisonale Zeitraum enthaelt keine vollstaendigen historischen Pattern-Trades.")
        return

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
test_mode = st.sidebar.radio("", ["Manual Backtest", "TACO Edge Discovery", "Cycle Scanner", "SL Scanner", "TACO Radar", "Walk Forward Analysis", "Seasonality Lab", "Seasonality Muster"], horizontal=False, label_visibility="collapsed")

if test_mode == "Seasonality Lab":
    render_seasonality_lab()
    st.stop()

if test_mode == "Seasonality Muster":
    render_seasonality_muster()
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

    if test_mode not in ("Cycle Scanner", "SL Scanner"):
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

        st.header("Walk Forward Analysis")
        wf_asset_preset = st.selectbox("WF Asset", list(ASSET_PRESETS.keys()))
        wf_comp_preset = st.selectbox("WF Comparison Asset", list(COMPARISON_PRESETS.keys()))
        wf_start_year = st.number_input("Walk Forward Start Year", 1900, 2100, 2015)
        wf_end_year = st.number_input("Walk Forward End Year", 1900, 2100, 2026)
        wf_in_sample_years = st.number_input("In Sample Window Years", 1, 50, 20)
        wf_cycle_from = st.number_input("WF Cycle From", 2, 100, 5)
        wf_cycle_to = st.number_input("WF Cycle To", 2, 100, 30)
        wf_cycle_step = st.number_input("WF Cycle Step", 1, 20, 1)
        wf_sl_from = st.number_input("WF Stop Loss From %", 0.05, 20.0, 0.25, step=0.05)
        wf_sl_to = st.number_input("WF Stop Loss To %", 0.05, 20.0, 2.00, step=0.05)
        wf_sl_step = st.number_input("WF Stop Loss Step %", 0.05, 5.0, 0.05, step=0.05)
        wf_min_trades = st.number_input("WF Min In-Sample Trades", 1, 1000, 30)
        wf_max_loss_streak = st.number_input("WF Max In-Sample Loss Streak", 0, 100, 5)
        run_wf = st.button("Run Walk Forward", type="primary")
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
    st.subheader("Walk Forward Analysis")
    st.caption(
        "Realistische Out-of-Sample-Validierung: Jedes Testjahr wird nur mit Parametern gehandelt, "
        "die aus den vorherigen In-Sample-Jahren bestimmt wurden. Das OOS-Jahr ist nie Teil der Optimierung."
    )

    if not run_wf:
        st.info("Waehle links Asset, Comparison, Fenster und Optimierungsbereiche aus. Danach auf Run Walk Forward klicken.")
        st.stop()

    if wf_end_year < wf_start_year:
        st.error("Walk Forward End Year muss groesser oder gleich Start Year sein.")
        st.stop()
    if wf_cycle_to < wf_cycle_from:
        st.error("WF Cycle To muss groesser oder gleich WF Cycle From sein.")
        st.stop()
    if wf_sl_to < wf_sl_from:
        st.error("WF Stop Loss To muss groesser oder gleich WF Stop Loss From sein.")
        st.stop()

    wf_asset_symbol = ASSET_PRESETS[wf_asset_preset]
    wf_comp_symbol = COMPARISON_PRESETS[wf_comp_preset]
    wf_asset_data = load_yahoo(wf_asset_symbol)
    wf_comp_data = load_yahoo(wf_comp_symbol)
    if wf_asset_data is None or wf_comp_data is None:
        st.error("Yahoo-Daten konnten fuer die Walk Forward Analysis nicht geladen werden.")
        st.stop()

    wf_cycles = list(range(int(wf_cycle_from), int(wf_cycle_to) + 1, int(wf_cycle_step)))
    wf_stop_values = np.round(np.arange(float(wf_sl_from), float(wf_sl_to) + float(wf_sl_step) / 2, float(wf_sl_step)), 4).tolist()

    with st.spinner("Walk Forward Analysis laeuft. Je breiter Cycle/SL-Range, desto laenger dauert es."):
        wf_yearly, wf_trades, wf_equity, wf_summary = run_walk_forward(
            wf_asset_data,
            wf_comp_data,
            settings,
            int(wf_start_year),
            int(wf_end_year),
            int(wf_in_sample_years),
            wf_cycles,
            wf_stop_values,
            int(wf_min_trades),
            int(wf_max_loss_streak),
        )

    st.subheader("Walk Forward Gesamtauswertung")
    summary_cols = st.columns(6)
    for col, (key, value) in zip(summary_cols, wf_summary.items()):
        col.metric(key, "n/a" if pd.isna(value) else f"{value:,.2f}")

    if not wf_equity.empty:
        wf_equity_fig = go.Figure()
        wf_equity_fig.add_trace(go.Scatter(x=wf_equity["date"], y=wf_equity["equity"], mode="lines", name="Walk Forward Equity"))
        wf_equity_fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Equity")
        st.plotly_chart(wf_equity_fig, use_container_width=True)

    st.subheader("Walk Forward Jahresergebnisse")
    st.dataframe(wf_yearly, use_container_width=True)

    st.subheader("Walk Forward Trades")
    st.dataframe(wf_trades, use_container_width=True)

    yearly_csv = wf_yearly.to_csv(index=False).encode("utf-8")
    trades_csv = wf_trades.to_csv(index=False).encode("utf-8")
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
