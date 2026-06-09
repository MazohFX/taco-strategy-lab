import math
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

render_fear_greed_panel()

with st.sidebar:
    test_mode = st.radio("Modus", ["Manual Backtest", "Cycle Scanner", "SL Scanner"], horizontal=False)
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

    if test_mode == "Cycle Scanner":
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
        run_scan = st.button("Run Cycle Scan", type="primary")
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
    elif test_mode == "SL Scanner":
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        run_scan = False
        st.header("SL Scanner")
        sl_asset_preset = st.selectbox("SL Scanner Asset", list(ASSET_PRESETS.keys()))
        sl_comp_preset = st.selectbox("SL Scanner Comparison", list(COMPARISON_PRESETS.keys()))
        sl_directions = st.multiselect("SL Scanner Directions", ["Long Only", "Short Only", "Long & Short"], default=["Long Only"])
        sl_from = st.number_input("SL From %", 0.05, 20.0, 0.25, step=0.05)
        sl_to = st.number_input("SL To %", 0.05, 20.0, 2.00, step=0.05)
        sl_step = st.number_input("SL Step %", 0.05, 5.0, 0.05, step=0.05)
        run_sl_scan = st.button("Run SL Scan", type="primary")
    else:
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False

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

        **Manual Backtest:** Du testest ein einzelnes Setup visuell.

        **Cycle Scanner:** Die App testet viele Cycle Lengths ueber ausgewaehlte Assets, Vergleichsassets
        und Richtungen.

        **SL Scanner:** Die App testet einen Stop-Loss-Bereich, z.B. 0,25% bis 2,00% in 0,05er-Schritten.
        Wichtig fuer die Auswertung sind hier `Avg Realized Win`, `Avg Realized Loss`, `Expectancy R`,
        `Max Loss Streak`, `Intratrade MAE` und `Stop Breach`.

        **Hinweis:** `Intratrade MAE` misst den groessten Kerzen-Gegenlauf waehrend des Trades. Das kann tiefer
        sein als dein Stop, weil Daily-Kerzen nur Open/High/Low/Close liefern. `Avg Realized Loss` und `Avg R`
        zeigen dagegen, was im Backtest tatsaechlich realisiert wurde.
        """
    )

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
