import math
import calendar
from dataclasses import dataclass
from datetime import date

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
        "title": title,
        "height": height,
        "paper_bgcolor": "#0b1220",
        "plot_bgcolor": "#0b1220",
        "font": {"color": "#dbeafe"},
        "margin": {"l": 35, "r": 20, "t": 55, "b": 35},
        "xaxis": {
            "gridcolor": "rgba(148,163,184,.16)",
            "zerolinecolor": "rgba(148,163,184,.16)",
            "showline": True,
            "linecolor": "rgba(148,163,184,.30)",
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,.16)",
            "zerolinecolor": "rgba(148,163,184,.16)",
            "showline": True,
            "linecolor": "rgba(148,163,184,.30)",
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
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


def render_seasonality_lab() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #070b13; }
        .block-container { padding-top: 1.2rem; }
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
            padding: 10px 12px;
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
            font-size: .70rem;
            line-height: 1.15;
        }
        .season-stat strong {
            display: block;
            color: #63c7e8;
            font-size: .95rem;
            line-height: 1.1;
            margin-bottom: 2px;
        }
        .season-stat.negative strong { color: #e36d5c; }
        .season-stat.neutral strong { color: #d6e3f3; }
        div[data-testid="stPlotlyChart"] {
            background: #141c28;
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 6px;
            padding: 6px;
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

    control_cols = st.columns([1.3, 1.0, 1.1, 1.2])
    with control_cols[0]:
        asset_label = st.selectbox("Asset", list(ASSET_PRESETS.keys()), key="seasonality_asset")
        default_symbol = ASSET_PRESETS[asset_label]
        symbol = st.text_input("Yahoo Symbol", value=default_symbol, key="seasonality_symbol")
    with control_cols[1]:
        lookback_label = st.selectbox(
            "Lookback",
            ["5 Jahre", "10 Jahre", "15 Jahre", "20 Jahre", "25 Jahre", "30 Jahre", "35 Jahre", "40 Jahre", "45 Jahre", "Max verfuegbare Jahre"],
            index=9,
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
        month_names = list(calendar.month_name)[1:]
        c1, c2 = st.columns(2)
        start_month_name = c1.selectbox("Start Monat", month_names, index=5)
        start_month = month_names.index(start_month_name) + 1
        start_day = c2.number_input("Start Tag", 1, 31, 26)
        c3, c4 = st.columns(2)
        end_month_name = c3.selectbox("End Monat", month_names, index=6)
        end_month = month_names.index(end_month_name) + 1
        end_day = c4.number_input("End Tag", 1, 31, 29)

    if not symbol.strip():
        st.warning("Bitte ein Yahoo-Symbol eingeben.")
        return

    with st.spinner(f"Lade maximale Yahoo-Historie fuer {symbol}..."):
        df = load_seasonality_data(symbol.strip())

    if df is None or df.empty:
        st.warning("Yahoo hat fuer dieses Symbol keine verwertbaren Tagesdaten geliefert.")
        return

    all_years = sorted(pd.Index(df.index.year).unique().astype(int).tolist())
    selected_years = filter_years_by_lookback_and_cycle(df, lookback_years, cycle_filter)
    if len(selected_years) < 3:
        st.warning("Fuer diese Auswahl sind weniger als drei Pattern-Jahre verfuegbar. Bitte Lookback oder Filter erweitern.")
    if not selected_years:
        return

    curve = build_seasonal_curve(df, selected_years)
    trades = analyze_seasonal_window(df, int(start_month), int(start_day), int(end_month), int(end_day), selected_years)
    if curve.empty:
        st.warning("Aus den ausgewaehlten Jahren konnte keine saisonale Kurve gebaut werden.")
        return

    today = pd.Timestamp(year=2001, month=date.today().month, day=date.today().day if not (date.today().month == 2 and date.today().day == 29) else 28)
    start_marker = pd.Timestamp(year=2001, month=int(start_month), day=min(int(start_day), calendar.monthrange(2001, int(start_month))[1]))
    end_marker = pd.Timestamp(year=2001, month=int(end_month), day=min(int(end_day), calendar.monthrange(2001, int(end_month))[1]))

    asset_short = asset_label.split(" proxy:")[0].replace(" proxy", "")
    years_text = f"{len(selected_years)} Years"
    main_col, stat_col = st.columns([3.1, 1.1])
    with main_col:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["plot_date"],
                y=curve["indexed"],
                mode="lines",
                name="Average Seasonal Trend",
                line={"color": "#22d3ee", "width": 3},
            )
        )
        fig.add_vline(x=today, line_color="#ef4444", line_width=2)
        fig.add_vline(x=start_marker, line_color="#38bdf8", line_width=1, line_dash="dash")
        fig.add_vline(x=end_marker, line_color="#38bdf8", line_width=1, line_dash="dash")
        if end_marker >= start_marker:
            fig.add_vrect(x0=start_marker, x1=end_marker, fillcolor="#38bdf8", opacity=0.14, line_width=0)
        else:
            fig.add_vrect(x0=start_marker, x1=pd.Timestamp("2001-12-31"), fillcolor="#38bdf8", opacity=0.14, line_width=0)
            fig.add_vrect(x0=pd.Timestamp("2001-01-01"), x1=end_marker, fillcolor="#38bdf8", opacity=0.14, line_width=0)
        fig.update_layout(**_seasonality_base_layout(f"Seasonal Trend of {asset_short} over {years_text}", 500))
        fig.add_annotation(
            text="TACO",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.52,
            showarrow=False,
            font={"size": 58, "color": "rgba(148,163,184,.12)"},
        )
        fig.update_xaxes(tickformat="%b", dtick="M1")
        fig.update_yaxes(title="Indexed Performance")
        st.plotly_chart(fig, width="stretch")

    profit_pct = trades["Profit %"] if not trades.empty else pd.Series(dtype=float)
    profit_points = trades["Profit"] if not trades.empty else pd.Series(dtype=float)
    avg_return = profit_pct.mean() if not profit_pct.empty else np.nan
    std_return = profit_pct.std(ddof=1) if len(profit_pct) > 1 else np.nan
    sharpe = avg_return / std_return if std_return and not pd.isna(std_return) else np.nan
    stats = {
        "Pattern-Jahre": len(selected_years),
        "Rest-Jahre": max(len(all_years) - len(selected_years), 0),
        "Annualized Return": curve["indexed"].iloc[-1] - 100,
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
        "Volatility": std_return,
    }
    def fmt_stat(value: float, suffix: str = "", digits: int = 2) -> str:
        if pd.isna(value):
            return "n/a"
        if digits == 0:
            return f"{value:,.0f}{suffix}"
        return f"{value:,.{digits}f}{suffix}"

    pattern_share = len(selected_years) / len(all_years) * 100 if all_years else 0
    rest_share = max(100 - pattern_share, 0)
    with stat_col:
        donut = go.Figure(
            go.Pie(
                labels=["Pattern", "Rest"],
                values=[pattern_share, rest_share],
                hole=0.62,
                marker={"colors": ["#62c8e8", "#334155"]},
                textinfo="none",
                sort=False,
            )
        )
        donut.update_layout(**_seasonality_base_layout("", 150))
        donut.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0})
        donut.add_annotation(text=f"{pattern_share:.0f}%", x=0.5, y=0.5, showarrow=False, font={"color": "#dbeafe", "size": 18})
        st.plotly_chart(donut, width="stretch")
        st.markdown(
            f"""
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
                    <div class="season-stat"><strong>{fmt_stat(stats["Average Gain"], "%")}</strong>Avg gain</div>
                    <div class="season-stat negative"><strong>{fmt_stat(stats["Average Loss"], "%")}</strong>Avg loss</div>
                    <div class="season-stat"><strong>{fmt_stat(trades["Max Rise"].max() if not trades.empty else np.nan, "%")}</strong>Max rise</div>
                    <div class="season-stat negative"><strong>{fmt_stat(trades["Max Drop"].min() if not trades.empty else np.nan, "%")}</strong>Max drop</div>
                </div>
            </div>
            <div class="season-panel">
                <div class="season-panel-title">Miscellaneous</div>
                <div class="season-stat-grid">
                    <div class="season-stat neutral"><strong>{len(selected_years)}</strong>Pattern years</div>
                    <div class="season-stat neutral"><strong>{max(len(all_years) - len(selected_years), 0)}</strong>Rest years</div>
                    <div class="season-stat neutral"><strong>{fmt_stat(stats["Standard Deviation"], "%")}</strong>Std. deviation</div>
                    <div class="season-stat neutral"><strong>{fmt_stat(stats["Sharpe Ratio"], "")}</strong>Sharpe ratio</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if trades.empty:
        st.warning("Der gewaehlte saisonale Zeitraum enthaelt keine vollstaendigen historischen Pattern-Trades.")
        return

    selected_df = df[df.index.year.isin(selected_years)].copy()
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
    trades_sorted = trades.sort_values("Year").copy()
    trades_sorted["Cumulative Profit"] = trades_sorted["Profit"].cumsum()
    trades_sorted["Cumulative Profit %"] = trades_sorted["Profit %"].cumsum()
    with lower_cols[0]:
        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(x=trades_sorted["Year"], y=trades_sorted["Cumulative Profit"], mode="lines+markers", name="Points", line={"color": "#22d3ee"}))
        cum_fig.add_trace(go.Scatter(x=trades_sorted["Year"], y=trades_sorted["Cumulative Profit %"], mode="lines+markers", name="Percent", line={"color": "#a78bfa"}))
        cum_fig.update_layout(**_seasonality_base_layout("Cumulative Profit fuer den Zeitraum", 340))
        cum_fig.add_annotation(text="TACO", xref="paper", yref="paper", x=0.5, y=0.52, showarrow=False, font={"size": 42, "color": "rgba(148,163,184,.12)"})
        st.plotly_chart(cum_fig, width="stretch")
    with lower_cols[1]:
        colors = np.where(trades_sorted["Profit %"] >= 0, "#62c8e8", "#c25f50")
        pattern_fig = go.Figure(go.Bar(x=trades_sorted["Year"], y=trades_sorted["Profit %"], marker_color=colors))
        pattern_fig.update_layout(**_seasonality_base_layout("Pattern Returns", 340))
        pattern_fig.add_annotation(text="TACO", xref="paper", yref="paper", x=0.5, y=0.52, showarrow=False, font={"size": 42, "color": "rgba(148,163,184,.12)"})
        st.plotly_chart(pattern_fig, width="stretch")

    st.subheader("Pattern Trades")
    st.dataframe(trades_sorted, width="stretch")

    downloads = st.columns(3)
    downloads[0].download_button(
        "Seasonality Trades CSV",
        trades_sorted.to_csv(index=False).encode("utf-8"),
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
    "NZDJPY (NZDJPY=X)": "NZDJPY=X",
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


st.title("TACO Strategy Lab")
st.caption("Python-Backtester und visuelle Website fuer den TACO Asset Comparison Oscillator.")

CORE_METRICS = ["Trades", "Winrate", "Profit Factor", "Net Profit", "Max DD", "Expectancy R", "Max Loss Streak"]
PRACTICE_METRICS = ["Avg Realized Win", "Avg Realized Loss", "Avg R", "Intratrade MAE", "Avg MFE", "Stop Breach Count", "Stop Breach Avg"]

test_mode = st.sidebar.radio("Modus", ["Manual Backtest", "Cycle Scanner", "SL Scanner", "TACO Radar", "Walk Forward Analysis", "Seasonality Lab"], horizontal=False)

if test_mode == "Seasonality Lab":
    render_seasonality_lab()
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
        st.header("Scanner")
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
        scan_comps = st.multiselect(
            "Comparison Assets",
            list(COMPARISON_PRESETS.keys()),
            default=["DXY proxy: US Dollar Index (DX-Y.NYB)", "Gold futures (GC=F)", "10Y Treasury Note futures (ZN=F)"],
        )
        scan_directions = st.multiselect(
            "Directions",
            ["Long Only", "Short Only", "Long & Short"],
            default=["Long Only", "Short Only"],
        )
        scan_cycle_from = st.number_input("Cycle From", 2, 100, 5)
        scan_cycle_to = st.number_input("Cycle To", 2, 100, 30)
        scan_cycle_step = st.number_input("Cycle Step", 1, 20, 1)
        top_curve_min_trades = st.number_input("Top Curves Min Trades", 1, 1000, 50)
        top_curve_max_loss_streak = st.number_input("Top Curves Max Loss Streak", 0, 100, 5)
        run_scan = st.button("Run Cycle Scan", type="primary")
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_wf = False
    elif test_mode == "SL Scanner":
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        st.header("SL Scanner")
        sl_asset_preset = st.selectbox("SL Scanner Asset", list(ASSET_PRESETS.keys()))
        sl_comp_preset = st.selectbox("SL Scanner Comparison", list(COMPARISON_PRESETS.keys()))
        sl_directions = st.multiselect("SL Scanner Directions", ["Long Only", "Short Only", "Long & Short"], default=["Long Only"])
        sl_from = st.number_input("SL From %", 0.05, 20.0, 0.25, step=0.05)
        sl_to = st.number_input("SL To %", 0.05, 20.0, 2.00, step=0.05)
        sl_step = st.number_input("SL Step %", 0.05, 5.0, 0.05, step=0.05)
        run_sl_scan = st.button("Run SL Scan", type="primary")
        run_wf = False
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
        st.info("Waehle links die Assets, Comparison Assets und Cycle Range aus. Danach auf Run Cycle Scan klicken.")
        st.stop()
    if test_mode == "Cycle Scanner" and run_scan:
        pass
    elif test_mode == "SL Scanner" and not run_sl_scan:
        st.info("Waehle links Asset, Comparison Asset, Direction und SL Range aus. Danach auf Run SL Scan klicken.")
        st.stop()
    elif test_mode == "SL Scanner" and run_sl_scan:
        pass
    else:
        st.info("Bitte Daten laden oder Demo nutzen.")
        st.stop()

if test_mode == "SL Scanner":
    if not run_sl_scan:
        st.info("Waehle links Asset, Comparison Asset, Direction und SL Range aus. Danach auf Run SL Scan klicken.")
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
        st.info("Waehle links die Assets, Comparison Assets und Cycle Range aus. Danach auf Run Cycle Scan klicken.")
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
