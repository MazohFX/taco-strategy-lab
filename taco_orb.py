"""TACO Lab – Opening Range Strategie (NYSE 9:30 ET) mit M15-Candle-Close-Bestätigung.

Regeln
- Opening Range (OR) = Hoch/Tief der ersten 15/30/60 Minuten ab 9:30 New York.
- Signal = erste M15-Kerze, die ÜBER dem OR-Hoch (Long) bzw. UNTER dem OR-Tief (Short) schließt.
  Einstieg zum Schlusskurs dieser Kerze, max. ein Trade pro Tag.
- Stop: andere OR-Seite, OR-Mitte oder x * Daily-ATR(14). Ziel: Tagesschluss 16:00 oder 1R/2R/3R.
- Stop wird zuerst geprüft, wenn Stop und Ziel in derselben Kerze liegen (konservativ).
- Kosten (Spread + Slippage) in Punkten pro Trade werden abgezogen.
Daten: Dukascopy M15 (UTC, Bid) aus data/dukascopy_intraday/<SYMBOL>_M15.csv(.gz).
Perioden: In-Sample 2015–2020, Out-of-Sample 2021–2023, Holdout ab 2024.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "dukascopy_intraday"

MARKETS = {
    "NQ (Nasdaq 100)": {"symbol": "USATECHIDXUSD", "cost": 2.0, "unit": "Punkte"},
    "US30 (Dow Jones)": {"symbol": "USA30IDXUSD", "cost": 3.0, "unit": "Punkte"},
    "Gold (XAUUSD)": {"symbol": "XAUUSD", "cost": 0.35, "unit": "$"},
}
STOPS = {
    "Andere Seite der Range": "OR_andere",
    "Mitte der Range": "OR_mitte",
    "0,10 × Daily-ATR": "0.10",
    "0,25 × Daily-ATR": "0.25",
}
BE_OPTIONS = {"Aus": 0.0, "bei +0,5 R": 0.5, "bei +0,7 R": 0.7, "bei +1,0 R": 1.0}
TARGETS = {"Tagesschluss (16:00 NY)": 0, "1R": 1, "2R": 2, "3R": 3}
LAST_ENTRY = {"bis 17:00 (11:00 NY)": 660, "bis 18:00 (12:00 NY)": 720, "bis 22:00 (ganzer Handelstag)": 960}
FILTERS = [
    "Alle Signale", "Nur Long", "Nur Short",
    "Gap > 0,3 ATR in Ausbruchsrichtung", "Gap in Ausbruchsrichtung", "Gap gegen Ausbruchsrichtung",
    "Range eng", "Range mittel", "Range weit",
    "Mit Trend (Vortag über/unter EMA20)", "Gegen Trend",
]
FILTER_TEXT = {
    "Alle Signale": ("Jeder Ausbruch wird gehandelt – Long und Short, egal wie der Tag eröffnet hat.",
                     "Ob der reine Ausbruch aus der Range die Tagesrichtung vorhersagt."),
    "Nur Long": ("Nur Ausbrüche nach OBEN (M15-Schluss über dem Range-Hoch). Ausbrüche nach unten werden ignoriert.",
                 "Ob Long-Ausbrüche allein besser sind. Achtung: Indizes steigen langfristig – Long profitiert davon automatisch."),
    "Nur Short": ("Nur Ausbrüche nach UNTEN (M15-Schluss unter dem Range-Tief). Ausbrüche nach oben werden ignoriert.",
                  "Ob Short-Ausbrüche allein funktionieren – gegen den langfristigen Aufwärtstrend der Indizes."),
    "Gap > 0,3 ATR in Ausbruchsrichtung": (
        "Nur Tage mit großem Eröffnungs-Gap (15:30-Kurs mehr als 0,3 × ATR vom Vortagesschluss 22:00 entfernt) – "
        "und nur der Ausbruch in Gap-Richtung: Gap nach oben → nur Long, Gap nach unten → nur Short. "
        "Kleines Gap → kein Trade.",
        "„Gap and Go“: Wenn über Nacht etwas passiert ist (News, Zahlen) und der Kurs nach der Eröffnung in dieselbe "
        "Richtung ausbricht, läuft er dann eher weiter?"),
    "Gap in Ausbruchsrichtung": (
        "Nur Ausbrüche in Richtung des Gaps, egal wie groß das Gap ist (Gap hoch → nur Long, Gap runter → nur Short).",
        "Wie oben, aber ohne Mindestgröße – zeigt, ob die Größe des Gaps wichtig ist."),
    "Gap gegen Ausbruchsrichtung": (
        "Nur Ausbrüche GEGEN das Gap (Gap hoch, aber Ausbruch nach unten → Short; Gap runter, Ausbruch nach oben → Long).",
        "Ob es sich lohnt, wenn der Markt das Gap nach der Eröffnung wieder schließen will („Gap Fill“)."),
    "Range eng": ("Nur Tage, an denen die Range im Vergleich zum ATR klein ist (unteres Drittel).",
                  "Ob ein Ausbruch aus einer engen, ruhigen Eröffnung sauberer läuft. Grenzen nur aus 2015–2020 bestimmt."),
    "Range mittel": ("Nur Tage mit mittelgroßer Range (mittleres Drittel im Verhältnis zum ATR).",
                     "Vergleich zu enger und weiter Range."),
    "Range weit": ("Nur Tage, an denen die Range schon sehr groß ist (oberes Drittel im Verhältnis zum ATR).",
                   "Ob nach einer wilden Eröffnung noch Bewegung übrig ist – oder ob der Tag schon „verbraucht“ ist."),
    "Mit Trend (Vortag über/unter EMA20)": (
        "Long nur, wenn der Vortagesschluss über dem 20-Tage-Durchschnitt (EMA20) lag; Short nur, wenn darunter.",
        "Ob Ausbrüche in Richtung des übergeordneten Trends besser laufen."),
    "Gegen Trend": ("Long nur, wenn der Vortag unter der EMA20 schloss; Short nur, wenn darüber.",
                    "Ob Ausbrüche gegen den Trend (Gegenbewegungen) funktionieren."),
}

PRESETS = {
    "NQ Gap & Go – vorab festgelegte Regel": dict(market="NQ (Nasdaq 100)", or_min=15, stop="0,25 × Daily-ATR",
                                                 target="Tagesschluss (16:00 NY)", last="bis 17:00 (11:00 NY)",
                                                 filt="Gap > 0,3 ATR in Ausbruchsrichtung"),
    "NQ Gap & Go – stabilste Variante": dict(market="NQ (Nasdaq 100)", or_min=60, stop="Mitte der Range",
                                            target="Tagesschluss (16:00 NY)", last="bis 22:00 (ganzer Handelstag)",
                                            filt="Gap > 0,3 ATR in Ausbruchsrichtung"),
    "Klassisch: 15 Min, Stop andere Seite, bis Schluss": dict(market="NQ (Nasdaq 100)", or_min=15,
                                                             stop="Andere Seite der Range",
                                                             target="Tagesschluss (16:00 NY)",
                                                             last="bis 22:00 (ganzer Handelstag)", filt="Alle Signale"),
}
PERIODS = [("In-Sample", 2015, 2020), ("Out-of-Sample", 2021, 2023), ("Holdout", 2024, 2100)]


# --------------------------------------------------------------------------- Daten
def data_path(symbol: str) -> Path | None:
    for ext in (".csv.gz", ".csv"):
        p = DATA_DIR / f"{symbol}_M15{ext}"
        if p.exists():
            return p
    return None


def load_days(symbol: str):
    """Liest M15, schneidet auf 9:30–16:00 ET und liefert Tageslisten + Tageskontext."""
    path = data_path(symbol)
    if path is None:
        return None
    df = pd.read_csv(path, usecols=["datetime", "open", "high", "low", "close", "ticks"])
    ts = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    mins = (ts.dt.hour * 60 + ts.dt.minute).values
    rth = (mins >= 570) & (mins < 960)
    d = df[rth].copy()
    d["mins"] = mins[rth]
    d["date"] = ts[rth].dt.date.values
    days = {}
    for date, g in d.groupby("date", sort=True):
        if len(g) < 20:                      # halbe Handelstage / Datenlücken auslassen
            continue
        days[date] = (g["mins"].values, g["high"].values, g["low"].values, g["close"].values, g["open"].values[0],
                      float(g["ticks"].values[0]))
    dates = list(days)
    o = np.array([days[x][4] for x in dates]); h = np.array([days[x][1].max() for x in dates])
    lo = np.array([days[x][2].min() for x in dates]); c = np.array([days[x][3][-1] for x in dates])
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift().values
    ema = pd.Series(c).ewm(span=20, adjust=False).mean().shift().values
    vol0 = pd.Series([days[x][5] for x in dates], dtype=float)
    rel_vol = (vol0 / vol0.rolling(20).mean().shift()).values          # Volumen 1. Viertelstunde vs. Ø 20 Tage
    ctx = pd.DataFrame({"atr": atr, "gap": (o - pc) / atr, "trend": np.sign(pc - ema), "rel_vol": rel_vol}, index=dates)
    idx = pd.to_datetime(pd.Series(dates))
    vix = _load_vix()
    ctx["vix"] = vix.reindex(idx.values).ffill().shift(1).values if vix is not None else np.nan
    news = _news_dates()
    ctx["news"] = [d in news for d in dates]
    return days, ctx


NEWS_CODES = {"consumer-price-index-mm": "CPI", "nonfarm-payrolls": "NFP", "fed-interest-rate-decision": "FOMC"}


def _load_vix():
    p = Path(__file__).resolve().parent / "data" / "vix" / "VIX.csv"
    if not p.exists():
        return None
    v = pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"]
    full = pd.date_range(v.index.min(), pd.Timestamp.today().normalize())
    return v.reindex(full).ffill()


def _news_dates():
    """US-Handelstage mit CPI, NFP oder FOMC-Zinsentscheid (aus dem MT5-Makrokalender)."""
    base = Path(__file__).resolve().parent / "data" / "macro_calendar"
    frames = [pd.read_csv(f) for f in (base / "mt5_history.csv.gz", base / "mt5_recent.csv") if f.exists()]
    if not frames:
        return set()
    cal = pd.concat(frames)
    cal = cal[(cal.currency == "USD") & cal.event_code.isin(NEWS_CODES)]
    t = pd.to_datetime(cal.time_utc).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    return set(t.dt.date)


def next_session_info(days) -> dict:
    """ATR(14) aus der US-Handelszeit inkl. letztem Datentag = Wert für die NÄCHSTE Sitzung."""
    dates = list(days)
    h = np.array([days[x][1].max() for x in dates]); lo = np.array([days[x][2].min() for x in dates])
    c = np.array([days[x][3][-1] for x in dates]); pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values
    ema = pd.Series(c).ewm(span=20, adjust=False).mean().values
    return dict(last_date=dates[-1], close=float(c[-1]), atr=float(atr[-1]), ema20=float(ema[-1]))


# --------------------------------------------------------------------------- Simulation
def simulate(days, ctx, or_min: int, stop: str, target_R: int, last_entry: int, cost: float,
             be_R: float = 0.0) -> pd.DataFrame:
    """be_R > 0: Stop auf Einstieg (Break-even), sobald der Kurs +be_R × Risiko erreicht hat (gilt ab der nächsten Kerze)."""
    rows = []
    n_or = or_min // 15
    for date, (m, H, L, C, _o, _v) in days.items():
        atr = ctx.at[date, "atr"]
        if not np.isfinite(atr):
            continue
        k = int(np.searchsorted(m, 570 + or_min))
        if k < n_or or k >= len(m):
            continue
        hi, lo = H[:k].max(), L[:k].min()
        cc = C[k:]; mm = m[k:]
        ok = ((cc > hi) | (cc < lo)) & (mm + 15 <= last_entry)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        i = k + idx[0]
        d = 1 if C[i] > hi else -1
        e = C[i]
        if stop == "OR_andere":
            s = lo if d > 0 else hi
        elif stop == "OR_mitte":
            s = (hi + lo) / 2
        else:
            s = e - d * float(stop) * atr
        risk = d * (e - s)
        if not risk > 0:
            continue
        tgt = e + d * target_R * risk if target_R else None
        r = None; exit_min = m[-1]; cur = s; be_done = False
        be_lvl = e + d * be_R * risk if be_R else None
        for j in range(i + 1, len(C)):
            if (d > 0 and L[j] <= cur) or (d < 0 and H[j] >= cur):
                r = d * (cur - e) / risk; exit_min = m[j]; break
            if tgt is not None and ((d > 0 and H[j] >= tgt) or (d < 0 and L[j] <= tgt)):
                r = float(target_R); exit_min = m[j]; break
            if be_lvl is not None and not be_done and ((d > 0 and H[j] >= be_lvl) or (d < 0 and L[j] <= be_lvl)):
                cur = e; be_done = True
        if r is None:
            r = d * (C[-1] - e) / risk
        rows.append((date, "Long" if d > 0 else "Short", f"{m[i]//60:02d}:{m[i]%60:02d}",
                     f"{(exit_min+15)//60:02d}:{(exit_min+15)%60:02d}", e, s, risk, r - cost / risk,
                     (hi - lo) / atr, ctx.at[date, "gap"], ctx.at[date, "trend"], ctx.at[date, "rel_vol"],
                     ctx.at[date, "vix"], ctx.at[date, "news"]))
    t = pd.DataFrame(rows, columns=["Datum", "Richtung", "Einstieg (NY)", "Ausstieg (NY)", "Einstiegskurs",
                                    "Stop", "Risiko", "R", "Range_ATR", "Gap_ATR", "Trend", "RelVol", "VIX", "News"])
    t["Jahr"] = pd.to_datetime(t["Datum"]).dt.year
    t["Periode"] = np.select([t.Jahr <= 2020, t.Jahr <= 2023], ["In-Sample", "Out-of-Sample"], "Holdout")
    return t


def apply_filter(t: pd.DataFrame, filt: str) -> pd.DataFrame:
    if t.empty:
        return t
    d = np.where(t.Richtung == "Long", 1, -1)
    q1, q2 = t.loc[t.Periode == "In-Sample", "Range_ATR"].quantile([1 / 3, 2 / 3])   # Schwellen nur aus IS
    gap_dir = np.sign(t.Gap_ATR.values)
    m = {
        "Alle Signale": np.ones(len(t), bool),
        "Nur Long": d > 0, "Nur Short": d < 0,
        "Gap > 0,3 ATR in Ausbruchsrichtung": (np.abs(t.Gap_ATR.values) > 0.3) & (gap_dir == d),
        "Gap in Ausbruchsrichtung": gap_dir == d, "Gap gegen Ausbruchsrichtung": gap_dir == -d,
        "Range eng": t.Range_ATR.values <= q1,
        "Range mittel": (t.Range_ATR.values > q1) & (t.Range_ATR.values <= q2),
        "Range weit": t.Range_ATR.values > q2,
        "Mit Trend (Vortag über/unter EMA20)": t.Trend.values == d, "Gegen Trend": t.Trend.values == -d,
    }[filt]
    return t[m].reset_index(drop=True)


# --------------------------------------------------------------------------- Kennzahlen
def metrics(r: np.ndarray, years: float) -> dict:
    from scipy import stats
    n = len(r)
    if n == 0:
        return dict(n=0)
    mean = r.mean(); sd = r.std(ddof=1) if n > 1 else np.nan
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    p = float(stats.t.sf(t, n - 1)) if np.isfinite(t) else np.nan       # einseitig: Erwartungswert > 0
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    per_year = n / years if years > 0 else np.nan
    return dict(n=n, wr=(r > 0).mean(), avg=mean, total=r.sum(), pf=wins / losses if losses > 0 else np.inf,
                sharpe=mean / sd * np.sqrt(per_year) if sd and sd > 0 else np.nan, t=t, p=p, maxdd=dd,
                per_year=per_year)


def period_metrics(t: pd.DataFrame) -> dict:
    out = {}
    for name, y0, y1 in PERIODS:
        sub = t[(t.Jahr >= y0) & (t.Jahr <= y1)]
        yrs = (sub.Jahr.max() - sub.Jahr.min() + 1) if len(sub) else 0
        if name == "Holdout" and len(sub):
            last = pd.to_datetime(sub.Datum.max())
            yrs = (last - pd.Timestamp("2024-01-01")).days / 365.25
        out[name] = metrics(sub.R.values, yrs)
    return out


def verdict(pm: dict):
    oos, ho = pm["Out-of-Sample"], pm["Holdout"]
    if oos.get("n", 0) < 30 or ho.get("n", 0) < 30:
        return "⚪", "Zu wenige Trades für eine Aussage"
    if oos["avg"] > 0 and ho["avg"] > 0 and oos["p"] < 0.05 and ho["p"] < 0.05:
        return "🟢", "Bestätigt: Out-of-Sample UND Holdout signifikant positiv"
    if oos["avg"] > 0 and ho["avg"] > 0 and pm["In-Sample"].get("avg", 0) <= 0:
        return "🟠", "Uneinheitlich: In-Sample negativ, danach positiv – eher Marktphase als stabile Edge"
    if oos["avg"] > 0 and ho["avg"] > 0:
        return "🟡", "Positiv auf neuen Daten, aber nicht signifikant – nur mit kleinem Risiko forward testen"
    return "🔴", "Keine Edge: auf neuen Daten nicht positiv"


# --------------------------------------------------------------------------- Streamlit-Seite
def render_orb_strategy() -> None:
    import streamlit as st
    import plotly.graph_objects as go

    st.title("⏰ Opening Range Strategie")
    st.caption("Ausbruch aus der Range der ersten Minuten nach der NYSE-Eröffnung (9:30 New York = 15:30 deutsche Zeit). "
               "Einstieg nur mit **M15-Schlusskurs** außerhalb der Range.")

    @st.cache_data(show_spinner="Lade M15-Daten …")
    def _days(symbol):
        return load_days(symbol)

    @st.cache_data(show_spinner="Simuliere Trades …")
    def _trades(symbol, or_min, stop, target_R, last_entry, cost, be_R):
        days, ctx = _days(symbol)
        return simulate(days, ctx, or_min, stop, target_R, last_entry, cost, be_R)

    # ---- Einstellungen
    preset = st.selectbox("Vorlage", ["Eigene Einstellungen"] + list(PRESETS), index=1, key="orb_preset")
    if preset != st.session_state.get("orb_last_preset"):
        st.session_state["orb_last_preset"] = preset
        if preset in PRESETS:
            p = PRESETS[preset]
            st.session_state.update({"orb_market": p["market"], "orb_or": p["or_min"], "orb_stop": p["stop"],
                                     "orb_target": p["target"], "orb_last": p["last"], "orb_filter": p["filt"],
                                     "orb_be": "Aus", "orb_f_news": False, "orb_f_vol": False, "orb_f_vix": False})
            for v in MARKETS.values():
                st.session_state.pop(f"orb_cost_{v['symbol']}", None)
    c1, c2, c3 = st.columns(3)
    market = c1.selectbox("Markt", list(MARKETS), key="orb_market")
    or_min = c2.radio("Länge der Range (Minuten)", [15, 30, 60], horizontal=True, key="orb_or")
    filt = c3.selectbox("Filter", FILTERS, key="orb_filter")
    c4, c5, c6, c7 = st.columns(4)
    stop_lbl = c4.selectbox("Stop", list(STOPS), key="orb_stop")
    tgt_lbl = c5.selectbox("Ziel", list(TARGETS), key="orb_target")
    last_lbl = c6.selectbox("Signal zählt (dt. Zeit)", list(LAST_ENTRY), key="orb_last",
                            help="Eingestiegen wird frühestens nach dem Ende der Range (bei 15 Min. um 15:45). "
                                 "Hier stellst du ein, bis wann ein Ausbruch noch als Signal zählt. "
                                 "Kommt bis dahin kein M15-Schluss außerhalb der Range, gibt es an dem Tag keinen Trade.")
    mk = MARKETS[market]
    cost = c7.number_input(f"Kosten pro Trade ({mk['unit']})", 0.0, 50.0, float(mk["cost"]), 0.05, key=f"orb_cost_{mk['symbol']}",
                           help="Spread + Slippage für Ein- und Ausstieg zusammen. Kommission hier mit einrechnen.")
    c8, c9 = st.columns([1, 3])
    be_lbl = c8.selectbox("Break-even", list(BE_OPTIONS), key="orb_be",
                          help="Stop auf den Einstiegskurs ziehen, sobald der Trade so weit im Plus war "
                               "(+0,7 R = 70 % des Stop-Abstands). Gilt ab der nächsten M15-Kerze.")
    with c9:
        st.markdown("**Zusatz-Filter** (kombinierbar mit dem Filter oben)")
        x1, x2, x3 = st.columns(3)
        f_news = x1.checkbox("Keine CPI/NFP/FOMC-Tage", key="orb_f_news",
                             help="An Tagen mit US-Inflationsdaten, Arbeitsmarktbericht oder Fed-Zinsentscheid nicht handeln.")
        f_vol = x2.checkbox("Nur hohes Eröffnungs-Volumen", key="orb_f_vol",
                            help="Volumen der ersten Viertelstunde über dem Durchschnitt der letzten 20 Tage.")
        f_vix = x3.checkbox("Nur VIX > 20", key="orb_f_vix", help="Nur handeln, wenn der VIX am Vortag über 20 schloss.")

    # ---- Erklärung: was wird getradet / getestet
    with st.expander("🔎 Was wird hier getradet – und was wird getestet?"):
        trade_txt, test_txt = FILTER_TEXT[filt]
        end = {15: "15:45", 30: "16:00", 60: "16:30"}[or_min]
        deadline = {"bis 17:00 (11:00 NY)": "17:00", "bis 18:00 (12:00 NY)": "18:00", "bis 22:00 (ganzer Handelstag)": "22:00"}[last_lbl]
        stop_e = {"Andere Seite der Range": "auf der anderen Seite der Range",
                  "Mitte der Range": "in der Mitte der Range",
                  "0,10 × Daily-ATR": "0,10 × ATR vom Einstieg entfernt",
                  "0,25 × Daily-ATR": "0,25 × ATR vom Einstieg entfernt"}[stop_lbl]
        tgt_e = "Ausstieg zum Schlusskurs um 22:00, wenn der Stop vorher nicht getroffen wurde" if TARGETS[tgt_lbl] == 0 \
            else f"Ausstieg bei {tgt_lbl} Gewinn (Gewinn = {TARGETS[tgt_lbl]} × Stop-Abstand), sonst Stop oder Schlusskurs 22:00"
        st.markdown(
            f"**Getradet wird (Filter „{filt}“):** {trade_txt}\n\n"
            f"**Ablauf jedes Trades:**\n"
            f"1. 15:30–{end}: Hoch und Tief der ersten {or_min} Minuten = Range.\n"
            f"2. Ab {end}: Die erste M15-Kerze, die über dem Hoch (Long) oder unter dem Tief (Short) **schließt**, "
            f"ist das Signal. Einstieg zum Schlusskurs dieser Kerze. Nur ein Trade pro Tag.\n"
            f"3. Kommt bis {deadline} Uhr kein Signal → kein Trade an diesem Tag.\n"
            f"4. Stop {stop_e}. {tgt_e}.\n"
            + (f"   Break-even: Stop auf Einstieg, sobald der Trade {be_lbl.replace('bei ', '')} im Plus war.\n" if BE_OPTIONS[be_lbl] else "")
            + f"5. Von jedem Ergebnis werden {cost} {mk['unit']} Kosten abgezogen.\n"
            + ("6. Zusatz-Filter: " + ", ".join(x for x, on in [("keine CPI/NFP/FOMC-Tage", f_news),
                ("nur Eröffnungs-Volumen über dem 20-Tage-Schnitt", f_vol), ("nur VIX > 20", f_vix)] if on) + ".\n"
               if (f_news or f_vol or f_vix) else "")
            + "\n"
            f"**Getestet wird:** {test_txt}\n\n"
            "**Wie wird getestet?** Alle Trades seit 2015 werden Kerze für Kerze simuliert und in drei Zeiträume geteilt: "
            "2015–2020 (In-Sample, hier wurden die Regeln gesucht), 2021–2023 (Out-of-Sample) und 2024–heute (Holdout). "
            "Eine echte Edge muss auch in den beiden neuen Zeiträumen positiv sein – idealerweise mit p-Wert unter 0,05.")

    loaded = _days(mk["symbol"])
    if loaded is not None:
        _render_today_box(st, loaded[0], market, mk, or_min, stop_lbl, filt, last_lbl)
    if loaded is None:
        st.error(f"Keine M15-Daten für {market} gefunden ({mk['symbol']}_M15.csv in data/dukascopy_intraday). "
                 "Am Mac mit **Import_Indizes_starten.command** laden.")
        return
    raw = _trades(mk["symbol"], or_min, STOPS[stop_lbl], TARGETS[tgt_lbl], LAST_ENTRY[last_lbl], cost,
                  BE_OPTIONS[be_lbl])
    t = apply_filter(raw, filt)
    if f_news:
        t = t[~t.News.astype(bool)]
    if f_vol:
        t = t[t.RelVol > 1.0]
    if f_vix:
        t = t[t.VIX > 20]
    t = t.reset_index(drop=True)
    if t.empty:
        st.warning("Keine Trades mit diesen Einstellungen.")
        return
    pm = period_metrics(t)

    # ---- Ampel
    icon, text = verdict(pm)
    st.markdown(f"### {icon} {text}")

    # ---- Kennzahlen je Periode
    cols = st.columns(3)
    labels = {"In-Sample": "In-Sample 2015–2020<br><small>(hier wurde gesucht)</small>",
              "Out-of-Sample": "Out-of-Sample 2021–2023<br><small>(neue Daten)</small>",
              "Holdout": "Holdout 2024–heute<br><small>(ganz neue Daten)</small>"}
    for col, (name, _, _) in zip(cols, PERIODS):
        mtr = pm[name]
        with col:
            st.markdown(f"**{labels[name]}**", unsafe_allow_html=True)
            if mtr.get("n", 0) == 0:
                st.info("keine Trades"); continue
            color = "#22c55e" if mtr["avg"] > 0 else "#ef4444"
            st.markdown(f"<div style='font-size:2rem;font-weight:700;color:{color}'>{mtr['avg']:+.3f} R</div>"
                        f"<div style='opacity:.7;margin-top:-6px'>Ø pro Trade (nach Kosten)</div>",
                        unsafe_allow_html=True)
            sig = "✅ signifikant" if mtr["p"] < 0.05 else "❌ nicht signifikant"
            st.markdown(
                f"| | |\n|---|---|\n"
                f"| Trades | {mtr['n']} (≈{mtr['per_year']:.0f}/Jahr) |\n"
                f"| Trefferquote | {mtr['wr']:.0%} |\n"
                f"| Summe | {mtr['total']:+.1f} R |\n"
                f"| p-Wert | {mtr['p']:.3f} {sig} |\n"
                f"| Sharpe (pro Jahr) | {mtr['sharpe']:.2f} |\n"
                f"| Profit-Faktor | {mtr['pf']:.2f} |\n"
                f"| Max. Drawdown | {mtr['maxdd']:.1f} R |")

    if filt != "Alle Signale":
        base = period_metrics(raw)
        rows = {"Mit Filter": [pm[n]["avg"] if pm[n].get("n") else np.nan for n, _, _ in PERIODS],
                "Ohne Filter (alle Signale)": [base[n]["avg"] if base[n].get("n") else np.nan for n, _, _ in PERIODS]}
        cmp_df = pd.DataFrame(rows, index=[n for n, _, _ in PERIODS]).T
        cmp_df["Wirkung des Filters"] = ["", ""]
        diffs = cmp_df.iloc[0, :3] - cmp_df.iloc[1, :3]
        st.markdown(f"**Was bringt der Filter?** Ø R pro Trade mit und ohne Filter „{filt}“ "
                    f"(Verbesserung: {', '.join(f'{n} {d:+.3f}' for n, d in diffs.items())})")
        st.dataframe(cmp_df.iloc[:, :3].style.format("{:+.3f}"), width="stretch")

    with st.expander("🧪 Getestete Verbesserungen (Break-even, News, Volumen, VIX) – Ergebnis"):
        st.markdown(
            "Getestet an der Regel *NQ Gap & Go – vorab festgelegt*. **Entschieden wurde nur auf 2015–2020** "
            "(6 Ideen → Schwelle p < 0,0083). Ø R pro Trade nach Kosten:\n\n"
            "| Variante | 2015–2020 | 2021–2023 | 2024–heute | Max. DD Holdout |\n|---|---|---|---|---|\n"
            "| Basis (ohne) | +0,235 | +0,085 | +0,090 | −25 R |\n"
            "| Break-even bei +0,5 R | +0,187 | +0,151 | ±0,000 | −20 R |\n"
            "| Break-even bei +0,7 R | +0,227 | +0,150 | +0,088 | −20 R |\n"
            "| Break-even bei +1,0 R | +0,202 | +0,149 | +0,117 | −19 R |\n"
            "| Keine CPI/NFP/FOMC-Tage | +0,253 | +0,049 | +0,101 | −20 R |\n"
            "| Nur hohes Eröffnungs-Volumen | +0,281 | +0,155 | +0,084 | −19 R |\n"
            "| Nur VIX > 20 | −0,046 | +0,196 | −0,029 | −11 R |\n\n"
            "**Ergebnis:** Keine Idee verbessert die Regel auf 2015–2020 signifikant (alle p > 0,07) – "
            "keine wurde deshalb in die Regel übernommen. Break-even ändert den Erwartungswert kaum, "
            "**verkleinert aber die Drawdowns** (Holdout −25 R → ca. −20 R) und senkt die Trefferquote "
            "(viele Trades enden bei ±0). Das ist Geschmackssache bzw. hilfreich für Prop-Firm-Drawdown-Grenzen.")

    with st.expander("ℹ️ Was bedeuten die Kennzahlen?"):
        st.markdown(
            "- **R** = Vielfaches des Risikos. +0,10 R heißt: bei 0,5 % Risiko pro Trade im Schnitt +0,05 % pro Trade.\n"
            "- **p-Wert** = Wahrscheinlichkeit, dieses Ergebnis auch ohne echte Edge zu sehen (einseitig, Ø > 0). "
            "Unter 0,05 = signifikant.\n"
            "- **Sharpe (pro Jahr)** = Ø R ÷ Schwankung × √(Trades pro Jahr). Über 1 gilt als gut.\n"
            "- **Profit-Faktor** = Summe Gewinne ÷ Summe Verluste. Über 1,0 = profitabel.\n"
            "- **Max. Drawdown** = größter Rückgang der Equity in R.\n"
            "- **Wichtig:** Im In-Sample wurden über 4.600 Varianten durchsucht – dort sieht die beste Variante immer gut aus. "
            "Aussagekräftig sind nur Out-of-Sample und Holdout.")

    # ---- Equity-Kurve
    t = t.copy(); t["Equity"] = t.R.cumsum()
    fig = go.Figure()
    band = {"In-Sample": "rgba(148,163,184,0.10)", "Out-of-Sample": "rgba(59,130,246,0.12)", "Holdout": "rgba(34,197,94,0.12)"}
    for name, y0, y1 in PERIODS:
        sub = t[(t.Jahr >= y0) & (t.Jahr <= y1)]
        if len(sub):
            fig.add_vrect(x0=sub.Datum.min(), x1=sub.Datum.max(), fillcolor=band[name], line_width=0,
                          annotation_text=name, annotation_position="top left")
    fig.add_trace(go.Scatter(x=t.Datum, y=t.Equity, mode="lines", line=dict(color="#22d3ee", width=2), name="Equity (R)"))
    fig.update_layout(title="Equity-Kurve in R (nach Kosten)", height=380, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis_title="R kumuliert", template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

    # ---- Jahre
    yr = t.groupby("Jahr").R.agg(["sum", "count", lambda r: (r > 0).mean()]).reset_index()
    yr.columns = ["Jahr", "R", "Trades", "Trefferquote"]
    fig2 = go.Figure(go.Bar(x=yr.Jahr.astype(str), y=yr.R, marker_color=np.where(yr.R > 0, "#22c55e", "#ef4444"),
                            text=[f"{v:+.1f}" for v in yr.R], textposition="outside",
                            hovertext=[f"{n} Trades, Treffer {w:.0%}" for n, w in zip(yr.Trades, yr.Trefferquote)]))
    fig2.update_layout(title="Ergebnis pro Jahr (R)", height=320, margin=dict(l=10, r=10, t=50, b=10), template="plotly_dark")
    st.plotly_chart(fig2, width="stretch")

    # ---- Long vs Short
    ls = t.groupby(["Periode", "Richtung"]).R.agg(["count", "mean"]).round(3).unstack("Richtung")
    ls = ls.reindex(["In-Sample", "Out-of-Sample", "Holdout"])
    ls.columns = [f"{b} {'Trades' if a == 'count' else 'Ø R'}" for a, b in ls.columns]
    st.markdown("**Long vs. Short**")
    st.dataframe(ls, width="stretch")

    with st.expander(f"📋 Alle {len(t)} Trades"):
        show = t[["Datum", "Periode", "Richtung", "Einstieg (NY)", "Ausstieg (NY)", "Einstiegskurs", "Stop", "Risiko", "R", "Gap_ATR"]]
        st.dataframe(show.round(3), width="stretch", height=400)
        st.download_button("📥 Trades als CSV", show.to_csv(index=False).encode("utf-8"),
                           file_name=f"orb_{mk['symbol']}_{or_min}min.csv", mime="text/csv", key="orb_dl")

    with st.expander("📏 Regeln & Daten"):
        st.markdown(
            f"- Range: Hoch/Tief von 9:30 bis {9 + (30 + or_min) // 60}:{(30 + or_min) % 60:02d} New York.\n"
            "- Einstieg: Schlusskurs der ersten M15-Kerze außerhalb der Range, höchstens ein Trade pro Tag.\n"
            f"- Stop: {stop_lbl}. Ziel: {tgt_lbl}. Signal zählt {last_lbl}.\n"
            "- Liegen Stop und Ziel in derselben Kerze, zählt der Stop (konservativ).\n"
            f"- Kosten: {cost} {mk['unit']} pro Trade abgezogen.\n"
            "- Gap = Eröffnung 9:30 gegenüber Schluss 16:00 des Vortags, gemessen in Daily-ATR(14).\n"
            "- ATR(14) wird nur aus der US-Handelszeit 9:30–16:00 NY berechnet – NICHT der ATR der 24h-Tageskerze im MT5 "
            "(der ist größer).\n"
            "- Daten: Dukascopy-CFD, M15, Bid-Kurse, 2015 bis heute.")


def _render_today_box(st, days, market, mk, or_min, stop_lbl, filt, last_lbl):
    info = next_session_info(days)
    u = mk["unit"]
    de = lambda x, n=1: f"{x:,.{n}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    with st.container(border=True):
        st.markdown(f"#### 📅 Heute handeln – {market}")
        last = pd.Timestamp(info["last_date"])
        st.caption(f"Berechnet aus den Daten bis **{last:%d.%m.%Y}** (letzter Handelstag in der Datenbank). "
                   "Den Vortagesschluss unten kannst du mit deinem Chart überschreiben.")
        c1, c2 = st.columns([1, 2])
        prev = c1.number_input(f"Schlusskurs Vortag um 16:00 NY / 22:00 dt. Zeit", value=round(info["close"], 2),
                               step=1.0 if u == "Punkte" else 0.1, key=f"orb_prev_{mk['symbol']}",
                               help="Kurs der letzten M15-Kerze vor 16:00 New York (15:45–16:00), nicht der 23-Uhr-Kurs des CFD.")
        atr = info["atr"]
        thr = 0.3 * atr
        c2.markdown(
            f"| | |\n|---|---|\n"
            f"| ATR(14) nur US-Handelszeit | **{de(atr)} {u}** |\n"
            f"| Gap-Schwelle (0,3 × ATR) | **{de(thr)} {u}** |\n"
            f"| 🟢 Long nur, wenn 15:30-Eröffnung **über** | **{de(prev + thr, 2)}** |\n"
            f"| 🔴 Short nur, wenn 15:30-Eröffnung **unter** | **{de(prev - thr, 2)}** |")
        stop_txt = {"0,25 × Daily-ATR": f"{de(0.25 * atr)} {u} vom Einstieg (0,25 × ATR)",
                    "0,10 × Daily-ATR": f"{de(0.10 * atr)} {u} vom Einstieg (0,10 × ATR)",
                    "Mitte der Range": "genau in der Mitte zwischen Range-Hoch und Range-Tief",
                    "Andere Seite der Range": "auf der anderen Seite der Range (Long: unter dem Tief, Short: über dem Hoch)"}[stop_lbl]
        end = {15: "15:45", 30: "16:00", 60: "16:30"}[or_min]
        deadline = {"bis 17:00 (11:00 NY)": "17:00", "bis 18:00 (12:00 NY)": "18:00", "bis 22:00 (ganzer Handelstag)": "21:45"}[last_lbl]
        gap_rule = filt == "Gap > 0,3 ATR in Ausbruchsrichtung"
        steps = []
        if gap_rule:
            steps.append(f"**15:30** – Eröffnungskurs ansehen. Liegt er zwischen {de(prev - thr, 2)} und {de(prev + thr, 2)} → **heute kein Trade**. "
                         f"Darüber → nur Long möglich. Darunter → nur Short möglich.")
        steps.append(f"**15:30–{end}** – Hoch und Tief der ersten {or_min} Minuten markieren (= Range).")
        steps.append(f"**Ab {end}** – auf jeden M15-Schluss achten: Schluss über Range-Hoch → Long, unter Range-Tief → Short"
                     + (" (nur in Gap-Richtung)" if gap_rule else "") + ". Einstieg zum Schlusskurs.")
        steps.append(f"**Stop** – {stop_txt}.")
        steps.append(f"**Kein Signal bis {deadline} Uhr** → kein Trade heute. **Ausstieg** – zum Schlusskurs um 22:00 (letzte M15-Kerze 21:45–22:00) bzw. vorher am Ziel oder Stop.")
        st.markdown("\n".join(f"{i}. {x}" for i, x in enumerate(steps, 1)))
        st.caption("Uhrzeiten in deutscher Zeit, gültig solange die USA und Deutschland gleichzeitig Sommer- bzw. Winterzeit haben. "
                   "In den 1–3 Wochen der Zeitumstellung (März/Oktober-November) liegt alles eine Stunde früher.")
