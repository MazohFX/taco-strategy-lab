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
    "S&P500 / US500 proxy: S&P 500 (^GSPC)": "^GSPC",
    "US30 proxy: Dow Jones Industrial Average (^DJI)": "^DJI",
    "EURUSD (EURUSD=X)": "EURUSD=X",
    "GBPUSD (GBPUSD=X)": "GBPUSD=X",
    "AUDUSD (AUDUSD=X)": "AUDUSD=X",
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
                pnl_points = exit_price - position["entry"] if side == "Long" else position["entry"] - exit_price
                gross = pnl_points * position["qty"]
                commission = settings.commission_pct / 100 * (position["entry"] * position["qty"] + exit_price * position["qty"])
                pnl = gross - commission
                equity += pnl
                mae = ((position["low"] - position["entry"]) / position["entry"] * 100) if side == "Long" else ((position["entry"] - position["high"]) / position["entry"] * 100)
                mfe = ((position["high"] - position["entry"]) / position["entry"] * 100) if side == "Long" else ((position["entry"] - position["low"]) / position["entry"] * 100)
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
                    "mae_pct": mae,
                    "mfe_pct": mfe,
                    "bars": i - position["bar"],
                })
                position = None

        if position is None:
            long_signal = row["osc"] < settings.lower and row["osc"] > prev["osc"] or prev["osc"] < settings.lower <= row["osc"]
            short_signal = row["osc"] > settings.upper and row["osc"] < prev["osc"] or prev["osc"] > settings.upper >= row["osc"]
            allow_long = settings.trade_direction in ["Long & Short", "Long Only"]
            allow_short = settings.trade_direction in ["Long & Short", "Short Only"]

            if long_signal and allow_long:
                entry = row["close"]
                stop = entry * (1 - settings.stop_pct / 100)
                risk_cash = equity * settings.risk_pct / 100
                qty = risk_cash / max(entry - stop, 1e-9)
                risk_points = entry - stop
                tp = entry + risk_points * settings.rr if settings.tp_mode == "Risk Reward" else entry * (1 + settings.fixed_tp_pct / 100) if settings.tp_mode == "Fixed %" else math.nan
                position = {"side": "Long", "entry": entry, "stop": stop, "tp": tp, "qty": qty, "date": date, "bar": i, "high": row["high"], "low": row["low"]}
            elif short_signal and allow_short:
                entry = row["close"]
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
        metrics = {"Trades": 0, "Winrate": np.nan, "Profit Factor": np.nan, "Net Profit": 0, "Max DD": np.nan, "Avg MAE": np.nan, "Avg MFE": np.nan}
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        dd = equity_df["equity"] / equity_df["equity"].cummax() - 1
        metrics = {
            "Trades": len(trades_df),
            "Winrate": len(wins) / len(trades_df) * 100,
            "Profit Factor": gross_profit / gross_loss if gross_loss else np.nan,
            "Net Profit": equity - settings.initial_capital,
            "Max DD": dd.min() * 100,
            "Avg MAE": trades_df["mae_pct"].mean(),
            "Avg MFE": trades_df["mfe_pct"].mean(),
        }
    return trades_df, equity_df, metrics


st.title("TACO Strategy Lab")
st.caption("Python-Backtester und visuelle Website fuer den TACO Asset Comparison Oscillator.")

with st.sidebar:
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
        asset_df, comp_df = make_demo_data()

    st.header("Einstellungen")
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
        tp_mode=st.selectbox("Take Profit Mode", ["Risk Reward", "Fixed %", "None"]),
        rr=st.number_input("Take Profit R Multiple", 0.1, 20.0, 2.0, step=0.1),
        fixed_tp_pct=st.number_input("Fixed Take Profit %", 0.05, 50.0, 1.3, step=0.05),
        exit_on_zero=st.checkbox("Exit When Oscillator Returns To Zero", False),
        time_exit=st.checkbox("Enable Exit After X Bars", False),
        exit_after_bars=st.number_input("Exit After X Bars", 1, 500, 20),
        initial_capital=st.number_input("Initial Capital", 100.0, 1_000_000.0, 10_000.0, step=100.0),
        commission_pct=st.number_input("Commission %", 0.0, 2.0, 0.05, step=0.01),
    )

with st.expander("Info: Wie funktioniert der TACO Backtest?", expanded=True):
    st.markdown(
        """
        **Berechnung:** Das Chart-Asset wird mit einem Vergleichsasset verglichen. Im Modus `Ratio Z-Score`
        wird `Chart Close / Comparison Close` berechnet, per Mittelwert und Standardabweichung normalisiert
        und mit `tanh()` auf etwa `-100` bis `+100` begrenzt.

        **Long Entry:** Der Oszillator ist unter der Unterbewertungszone und dreht nach oben, oder er kreuzt
        von unten zurueck ueber die Unterbewertungszone.

        **Short Entry:** Der Oszillator ist ueber der Ueberbewertungszone und dreht nach unten, oder er kreuzt
        von oben zurueck unter die Ueberbewertungszone.

        **Risk:** `Risk Per Trade %` bestimmt dein Kontorisiko. `Fixed Stop Loss %` bestimmt den Abstand vom
        Entry zum Stop. Die Positionsgroesse wird daraus automatisch berechnet.

        **Exit:** Je nach Einstellung per Risk-Reward-Target, Fixed-%-Target, Zero-Line-Exit oder Time-Exit.
        """
    )

if asset_df is None or comp_df is None:
    st.info("Bitte Daten laden oder Demo nutzen.")
    st.stop()

df = calculate_oscillator(asset_df, comp_df, settings)
trades, equity, metrics = backtest(df, settings)

cols = st.columns(7)
for col, key in zip(cols, ["Trades", "Winrate", "Profit Factor", "Net Profit", "Max DD", "Avg MAE", "Avg MFE"]):
    val = metrics[key]
    col.metric(key, "n/a" if pd.isna(val) else f"{val:,.2f}")

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
if not trades.empty:
    longs = trades[trades["side"] == "Long"]
    shorts = trades[trades["side"] == "Short"]
    fig.add_trace(go.Scatter(x=longs["entry_date"], y=longs["entry"], mode="markers", name="Long", marker=dict(color="#2f6bff", symbol="triangle-up", size=10)))
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

st.subheader("Trades")
st.dataframe(trades, use_container_width=True)
