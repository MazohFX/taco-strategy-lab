"""TACO Lab Modul "Makro + Technik": BOS + Fib + Opens-Setup, gefiltert nach VIX-Stimmung.

Frage: Laufen die BOS-Trades besser, wenn ihre Richtung zur Risikostimmung (VIX) passt?

Variante (a) "VIX-Regime + Risiko-Empfindlichkeit":
    Risk-on/off-Score aus dem VIX (Niveau gegen 5 Jahre + feste Zonen, 10-Tage-Veraenderung --
    dieselbe Logik wie im Makro-Dashboard, nur ohne Aktienindizes). Jede Waehrung hat eine feste
    Risiko-Empfindlichkeit (AUD/NZD +1 ... JPY -1, siehe macro_dashboard.RISK_BETA). Risk-on ->
    bevorzugte Richtung = Basis minus Quote; Risk-off -> umgekehrt.
Variante (b) "gemessene Korrelation":
    Statt der Faustregel die rollierende 12-Monats-Korrelation des Paars mit dem VIX, die zum
    Zeitpunkt des Trades bekannt war.

Kein Blick in die Zukunft: Fuer jeden Trade zaehlt der VIX-Schluss des letzten US-Handelstags
VOR dem Fuelltag. Rechenkern der Trades: bos_research (unveraendert). Holdout ab 2024 versiegelt.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import macro_dashboard as md

ROOT = Path(__file__).resolve().parent
VIX_CSV = ROOT / "data" / "vix" / "VIX.csv"

# Futures = Waehrung gegen USD. Vorzeichen +1: Paar = Future-Richtung, -1: invertiert.
FUTURE_PAIRS = {"EUR": ("EURUSD", 1), "GBP": ("GBPUSD", 1), "AUD": ("AUDUSD", 1), "NZD": ("NZDUSD", 1),
                "CAD": ("USDCAD", -1), "CHF": ("USDCHF", -1), "JPY": ("USDJPY", -1)}
APP_NEUTRAL = 0.15      # |Risk-Score| darunter = keine klare Stimmung
CORR_MIN = 0.10         # (b): |Korrelation| darunter = Paar haengt nicht am VIX
CORR_WINDOW = 252
ZONES = ["Ruhig", "Normal", "Stress", "Panik"]


# ── Daten ────────────────────────────────────────────────────────────────────
def load_vix(extra: pd.Series | None = None) -> pd.Series:
    """VIX-Tagesschluss: Repo-CSV (ab 2005), optional mit frischeren Werten verlaengert."""
    s = pd.Series(dtype=float)
    if VIX_CSV.exists():
        df = pd.read_csv(VIX_CSV, parse_dates=["date"])
        s = df.set_index("date")["close"].astype(float)
    if extra is not None and not extra.empty:
        extra = extra.copy()
        extra.index = pd.to_datetime(extra.index).normalize()
        s = pd.concat([s, extra[extra.index > (s.index.max() if len(s) else pd.Timestamp.min)]])
    return s[~s.index.duplicated(keep="last")].sort_index()


def vix_features(vix: pd.Series) -> pd.DataFrame:
    """Pro Handelstag (Schlusskurs bekannt): Zone, 5J-Perzentil, 10T-Veraenderung, Sprung,
    Risk-on-Score (-1..+1, + = Risk-on) -- wie md.risk_appetite, nur vektorisiert und ohne Aktien."""
    c = md.CONFIG
    v = vix.dropna().astype(float)
    win = 252 * c["vix_history_years"]
    # Perzentil mit Gleichstands-Mitte (sonst gilt ein konstanter VIX als "100. Perzentil")
    pct = v.rolling(win, min_periods=250).apply(
        lambda a: (a < a[-1]).mean() + 0.5 * (a == a[-1]).mean(), raw=True)
    level = -np.tanh((v - c["vix_neutral_level"]) / c["vix_level_scale"])
    chg = (v / v.shift(c["vix_change_lookback_days"]) - 1) * 100
    # Solange noch kein Jahr Historie fuer das Perzentil da ist: nur die feste Einordnung
    part_level = (0.5 * (-(pct - 0.5) * 2) + 0.5 * level).fillna(level)
    part_chg = -np.tanh(chg / c["vix_change_scale_pct"])
    app = pd.concat([part_level, part_chg], axis=1).mean(axis=1, skipna=True).clip(-1, 1)
    spike = (v / v.shift(md.VIX_SPIKE_DAYS) - 1) * 100 >= md.VIX_SPIKE_PCT
    zone = pd.cut(v, [-np.inf, 15, 20, 30, np.inf], labels=ZONES, right=False).astype(str)
    eff = zone.copy()
    eff[spike & zone.isin(["Ruhig", "Normal"])] = "Stress"
    eff[spike & (zone == "Stress")] = "Panik"
    return pd.DataFrame({"vix": v, "pct5y": pct, "chg10": chg, "app": app, "spike": spike,
                         "zone": zone, "zone_eff": eff})


def pair_daily_close(sym: str) -> pd.Series:
    p = ROOT / "data" / "mt5" / f"{sym}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p, usecols=["date", "close"])
    s = df.set_index(pd.to_datetime(df["date"], format="%Y.%m.%d"))["close"].astype(float)
    return s[s.index.weekday < 5]


def rolling_vix_corr(price: pd.Series, vix: pd.Series, window: int = CORR_WINDOW) -> pd.Series:
    """Rollierende Korrelation der Tagesrenditen (Ausreisser gekappt), Stand jeweils Tagesschluss."""
    df = pd.concat([price.rename("p"), vix.rename("v")], axis=1).dropna()
    r = df.pct_change().dropna()
    r = r.clip(lower=r.quantile(0.005), upper=r.quantile(0.995), axis=1)
    return r["p"].rolling(window, min_periods=window // 2).corr(r["v"])


# ── Markierung der Trades ────────────────────────────────────────────────────
def annotate(trades: pd.DataFrame, feats: pd.DataFrame, corr: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Haengt an jeden gefuellten Trade den VIX-Zustand vom letzten Handelstag VOR dem Fuelltag
    und die Einstufung mit/gegen/neutral fuer (a) und (b)."""
    t = trades.copy()
    if t.empty:
        return t
    t["decision_day"] = pd.to_datetime(t["fill_dt"]).dt.normalize()
    f = feats.reset_index().rename(columns={feats.index.name or "index": "vix_day"})
    f = f.rename(columns={f.columns[0]: "vix_day"}).sort_values("vix_day")
    t = t.sort_values("decision_day")
    t = pd.merge_asof(t, f, left_on="decision_day", right_on="vix_day", direction="backward",
                      allow_exact_matches=False)
    beta = t["sym"].map(md.risk_diff)
    risk_dir = np.where(beta.abs() >= md.RISK_MIN_DIFF, np.sign(beta), 0)
    app_dir = np.where(t["app"].abs() >= APP_NEUTRAL, np.sign(t["app"]), 0)
    t["bias_a"] = risk_dir * app_dir
    t["side_a"] = _side(t["dir"], t["bias_a"])
    t["corr_vix"] = np.nan
    if corr:
        parts = []
        for sym, g in t.groupby("sym"):
            cs = corr.get(sym)
            if cs is None or cs.dropna().empty:
                parts.append(g)
                continue
            cdf = cs.dropna().rename("corr_vix_").reset_index()
            cdf.columns = ["corr_day", "corr_vix_"]
            g = pd.merge_asof(g.sort_values("decision_day"), cdf.sort_values("corr_day"),
                              left_on="decision_day", right_on="corr_day", direction="backward",
                              allow_exact_matches=False)
            g["corr_vix"] = g["corr_vix_"]
            parts.append(g.drop(columns=["corr_vix_", "corr_day"]))
        t = pd.concat(parts, ignore_index=True)
        app_dir = np.where(t["app"].abs() >= APP_NEUTRAL, np.sign(t["app"]), 0)
    corr_dir = np.where(t["corr_vix"].abs() >= CORR_MIN, -np.sign(t["corr_vix"].fillna(0)), 0)
    t["bias_b"] = corr_dir * app_dir
    t["side_b"] = _side(t["dir"], t["bias_b"])
    return t.sort_values("fill_dt").reset_index(drop=True)


def _side(direction: pd.Series, bias) -> pd.Series:
    bias = pd.Series(bias, index=direction.index)
    out = np.where(bias == 0, "neutral", np.where(np.sign(direction) == bias, "mit", "gegen"))
    return pd.Series(out, index=direction.index)


def to_future_view(t: pd.DataFrame) -> pd.DataFrame:
    """Nur die 7 USD-Paare, Richtung in Future-Sicht (USDCAD long = CAD-Future short)."""
    inv = {pair: (cur, sign) for cur, (pair, sign) in FUTURE_PAIRS.items()}
    t = t[t["sym"].isin(inv)].copy()
    t["market"] = t["sym"].map(lambda s: f"{inv[s][0]} ({s})")
    t["fdir"] = t["dir"] * t["sym"].map(lambda s: inv[s][1])
    return t


# ── Statistik ────────────────────────────────────────────────────────────────
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = mean / se if se and se > 0 else float("nan")
    p = math.erfc(abs(t) / math.sqrt(2)) if not math.isnan(t) else float("nan")
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    eq = r.cumsum()
    return {"n": n, "er": mean, "se": se, "p": p, "win": float((r > 0.05).mean() * 100),
            "pf": float(pos / neg) if neg > 0 else float("inf"), "total": float(r.sum()),
            "maxdd": float((eq - eq.cummax()).min())}


def diff_test(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Welch-Test: Ist E[R] von a hoeher als von b? (Differenz, zweiseitiger p-Wert)"""
    a, b = a.dropna(), b.dropna()
    if len(a) < 10 or len(b) < 10:
        return float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return float(d), float("nan")
    return float(d), math.erfc(abs(d / se) / math.sqrt(2))


def bh_adjust(p: list[float]) -> list[float]:
    """Benjamini-Hochberg (FDR) fuer mehrere Maerkte gleichzeitig."""
    idx = [i for i, x in enumerate(p) if not (x is None or math.isnan(x))]
    out = [float("nan")] * len(p)
    if not idx:
        return out
    vals = sorted(((p[i], i) for i in idx))
    m = len(vals)
    adj = [0.0] * m
    prev = 1.0
    for k in range(m - 1, -1, -1):
        prev = min(prev, vals[k][0] * m / (k + 1))
        adj[k] = prev
    for k, (_, i) in enumerate(vals):
        out[i] = adj[k]
    return out


def filter_verdict(n_mit: int, d: float, q: float, er_mit: float) -> tuple[str, str]:
    if n_mit < 30 or math.isnan(d):
        return "⚪", "zu wenige Trades"
    if d > 0 and er_mit > 0 and not math.isnan(q) and q <= 0.05:
        return "🟢", "Filter hilft (signifikant)"
    if d > 0 and er_mit > 0 and not math.isnan(q) and q <= 0.15:
        return "🟡", "Filter hilft eher (grenzwertig)"
    if d > 0:
        return "🟠", "besser, aber nicht von Zufall unterscheidbar"
    return "🔴", "Filter hilft nicht"


def comparison_table(t: pd.DataFrame, side_col: str, group_col: str) -> pd.DataFrame:
    rows, pvals = [], []
    for name, g in t.groupby(group_col):
        s_all, s_mit = stats(g.R), stats(g[g[side_col] == "mit"].R)
        s_geg, s_neu = stats(g[g[side_col] == "gegen"].R), stats(g[g[side_col] == "neutral"].R)
        d, p = diff_test(g[g[side_col] == "mit"].R, g[g[side_col] == "gegen"].R)
        pvals.append(p)
        rows.append({"Markt": name,
                     "Alle: Trades": s_all["n"], "Alle: E[R]": s_all.get("er"),
                     "Mit VIX: Trades": s_mit["n"], "Mit VIX: E[R]": s_mit.get("er"),
                     "Mit VIX: Gewinn %": s_mit.get("win"), "Mit VIX: PF": s_mit.get("pf"),
                     "Gegen VIX: Trades": s_geg["n"], "Gegen VIX: E[R]": s_geg.get("er"),
                     "Neutral: Trades": s_neu["n"],
                     "Δ E[R] (mit − gegen)": d, "p": p, "_er_mit": s_mit.get("er", float("nan"))})
    q = bh_adjust(pvals)
    for row, qq in zip(rows, q):
        row["q (BH)"] = qq
        icon, txt = filter_verdict(row["Mit VIX: Trades"], row["Δ E[R] (mit − gegen)"], qq, row["_er_mit"])
        row["Urteil"] = f"{icon} {txt}"
        row.pop("_er_mit")
    return pd.DataFrame(rows)


def zone_direction_table(t: pd.DataFrame, dir_col: str = "dir") -> pd.DataFrame:
    """E[R] je VIX-Zone, getrennt nach Long/Short -- 'was bedeutet die Zone fuer die Richtung?'"""
    rows = []
    for z in ZONES:
        g = t[t["zone_eff"] == z]
        for lab, d in (("Long", 1), ("Short", -1)):
            s = stats(g[g[dir_col] == d].R)
            rows.append({"VIX-Zone": z, "Richtung": lab, "Trades": s["n"], "E[R]": s.get("er"),
                         "Gewinn %": s.get("win"), "p": s.get("p")})
    return pd.DataFrame(rows)


# ── UI ───────────────────────────────────────────────────────────────────────
def _fmt(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    fmt = {c: "{:+.3f}" for c in df.columns if "E[R]" in c or c.startswith("Δ")}
    fmt.update({c: "{:.3f}" for c in ("p", "q (BH)") if c in df.columns})
    fmt.update({c: "{:.1f}" for c in df.columns if "Gewinn" in c})
    fmt.update({c: "{:.2f}" for c in df.columns if c.endswith("PF")})

    def col_er(v):
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return "color:#4ade80" if v > 0 else "color:#f87171" if v < 0 else ""
        return ""

    er_cols = [c for c in df.columns if "E[R]" in c or c.startswith("Δ")]
    return df.style.format(fmt, na_rep="–").map(col_er, subset=er_cols)


def render_makro_technik() -> None:
    import plotly.graph_objects as go
    import streamlit as st
    import bos_pipeline as bp
    import taco_bos_setup as tbs

    st.header("🧭 Makro + Technik")
    st.caption("BOS + Fib + Opens-Setup, gefiltert nach der VIX-Risikostimmung. Frage: Laufen die Trades besser, "
               "wenn ihre Richtung zur Stimmung passt? Deine Makro-Analyse bleibt auf der Makro-Seite.")
    with st.expander("ℹ️ Wie wird getestet?", expanded=False):
        st.markdown(f"""
**1 · Trades:** dein BOS + Fib + Opens-Setup (gleicher Rechenkern wie „BOS + Fib + Opens Setup“), alle gefüllten Trades.

**2 · VIX-Zustand je Trade:** VIX-Schluss des letzten US-Handelstags **vor** dem Fülltag (kein Blick in die Zukunft):
Zone (< 15 ruhig · 15–20 normal · 20–30 Stress · > 30 Panik, mit Hochstufung bei einem VIX-Sprung ≥ {md.VIX_SPIKE_PCT:.0f} %
in {md.VIX_SPIKE_DAYS} Tagen) und Risk-on-Score (wie im Makro-Dashboard, ohne Aktienindizes). |Score| < {APP_NEUTRAL} = neutral.

**3 · Mit oder gegen die Stimmung?**
- **(a) Faustregel:** Risiko-Empfindlichkeit der Währungen (AUD/NZD +1 … JPY −1). Risk-on → Richtung Basis-minus-Quote
  bevorzugt, Risk-off → umgekehrt. Paare mit ähnlicher Empfindlichkeit (z. B. AUDNZD) sind immer „neutral“.
- **(b) Gemessen:** rollierende 12-Monats-Korrelation des Paars mit dem VIX (bekannt vor dem Trade). Paar fällt bei steigendem VIX
  → bei Risk-on Long bevorzugt. |Korrelation| < {CORR_MIN} = neutral.

**4 · Auswertung:** E[R] (Ergebnis in R pro Trade) „mit“ gegen „gegen“ die Stimmung, Welch-Test für den Unterschied,
Benjamini-Hochberg-Korrektur über alle Märkte (q). **Holdout ab {bp.HOLDOUT_START}** bleibt versiegelt, bis du ihn öffnest –
dann ist er dein echter Out-of-Sample-Test. Erst entscheiden, dann öffnen.
""")

    c1, c2, c3, c4, c5 = st.columns(5)
    universe = c1.radio("Märkte", ["Futures (7 USD-Paare)", "Alle 28 Forex-Paare"], key="mt_universe")
    tf = c2.selectbox("Timeframe", ["D1", "H4", "H1"], key="mt_tf")
    k = c3.select_slider("Swing-Schwelle (× ATR)", [1.0, 1.5, 2.0, 2.5, 3.0], value=1.5, key="mt_k")
    entry = c4.radio("Einstieg", [0.618, 0.786], horizontal=True, key="mt_entry")
    method = c5.radio("Methode", ["(a) Faustregel", "(b) gemessene Korrelation"], key="mt_method")
    side_col = "side_a" if method.startswith("(a)") else "side_b"
    open_holdout = st.checkbox(f"🔓 Holdout öffnen (ab {bp.HOLDOUT_START})", value=False, key="mt_holdout")

    stamp = tbs._data_stamp(tf)
    avail = tbs._availability_cached(tf, stamp)
    pairs = sorted(p for p in avail if universe.startswith("Alle") or p in {v[0] for v in FUTURE_PAIRS.values()})
    if not pairs:
        st.error(f"Keine {tf}-Daten für die gewählten Märkte gefunden.")
        return
    with st.spinner("Berechne BOS-Trades und VIX-Zustand je Trade …"):
        trades = tbs.compute_trades(tf, float(k), float(entry), tuple(pairs), stamp)
        if trades.empty:
            st.info("Keine BOS-Ereignisse gefunden.")
            return
        f = trades[trades.filled == 1].copy()
        vix = load_vix(md.fetch_daily_close("^VIX", "1y"))
        feats = vix_features(vix)
        corr = {s: rolling_vix_corr(pair_daily_close(s), vix) for s in pairs} if side_col == "side_b" else None
        t = annotate(f, feats, corr)
    t = t.dropna(subset=["app"])
    is_oos = t["fill_dt"] >= pd.Timestamp(bp.HOLDOUT_START)
    ins, oos = t[~is_oos], t[is_oos]

    futures = universe.startswith("Futures")
    view = to_future_view(ins) if futures else ins.assign(market=ins["sym"], fdir=ins["dir"])
    view_oos = to_future_view(oos) if futures else oos.assign(market=oos["sym"], fdir=oos["dir"])

    # Gesamtbild
    s_all, s_mit, s_geg = stats(view.R), stats(view[view[side_col] == "mit"].R), stats(view[view[side_col] == "gegen"].R)
    d, p = diff_test(view[view[side_col] == "mit"].R, view[view[side_col] == "gegen"].R)
    icon, txt = filter_verdict(s_mit["n"], d, p, s_mit.get("er", float("nan")))
    st.markdown(f"### {icon} Gesamt (In-Sample bis {bp.HOLDOUT_START}): {txt}")
    if not math.isnan(d) and d < 0 and not math.isnan(p) and p <= 0.05:
        st.info(f"Auffällig: Trades **gegen** die VIX-Stimmung liefen hier signifikant besser "
                f"({s_geg.get('er', 0):+.3f} R gegen {s_mit.get('er', 0):+.3f} R, p = {p:.3f}). Dein Setup steigt "
                "an Rücksetzern ein – das passt eher zu Gegenbewegungen nach Stimmungsübertreibungen. "
                "Nur eine Beobachtung im In-Sample: vor jeder Regel erst im Holdout prüfen.")
    m = st.columns(5)
    m[0].metric("Alle Trades", f"{s_all['n']}", f"E[R] {s_all.get('er', 0):+.3f}", delta_color="off")
    m[1].metric("Mit VIX-Stimmung", f"{s_mit['n']}", f"E[R] {s_mit.get('er', 0):+.3f}" if s_mit["n"] else "–", delta_color="off")
    m[2].metric("Gegen VIX-Stimmung", f"{s_geg['n']}", f"E[R] {s_geg.get('er', 0):+.3f}" if s_geg["n"] else "–", delta_color="off")
    m[3].metric("Unterschied mit − gegen", f"{d:+.3f} R" if not math.isnan(d) else "–", f"p = {p:.3f}" if not math.isnan(p) else "", delta_color="off")
    m[4].metric("Max DD mit / alle", f"{s_mit.get('maxdd', 0):.1f} / {s_all.get('maxdd', 0):.1f} R")

    fig = go.Figure()
    for lab, g, dash in (("Alle Trades", view, "dot"), ("Nur mit VIX-Stimmung", view[view[side_col] == "mit"], None),
                         ("Nur gegen VIX-Stimmung", view[view[side_col] == "gegen"], "dash")):
        g = g.sort_values("fill_dt")
        fig.add_scatter(x=g.fill_dt, y=g.R.cumsum(), mode="lines", name=lab, line=dict(dash=dash))
    md._plotly_layout(fig, height=340, title="Equity in R (In-Sample)")
    st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["💱 Je Markt", "🌡️ Je VIX-Zone", "🔓 Holdout", "📄 Trades"])
    with tabs[0]:
        st.markdown("**Hilft der VIX-Filter auf diesem Markt?** „Mit“ = Trade-Richtung passt zur Stimmung.")
        tab = comparison_table(view, side_col, "market")
        st.dataframe(_fmt(tab), use_container_width=True, hide_index=True)
        st.caption("q (BH) = p-Wert, korrigiert dafür, dass viele Märkte gleichzeitig getestet werden. "
                   "Ein echter Filter sollte auf den meisten Märkten in dieselbe Richtung zeigen, nicht nur auf einem.")
    with tabs[1]:
        st.markdown("**Was bringt jede VIX-Zone für Long und Short?**"
                    + (" Richtung in Future-Sicht (CAD-Future long = USDCAD short)." if futures else ""))
        sel = st.selectbox("Markt", ["Alle zusammen"] + sorted(view["market"].unique()), key="mt_zone_market")
        g = view if sel == "Alle zusammen" else view[view["market"] == sel]
        st.dataframe(_fmt(zone_direction_table(g, "fdir")), use_container_width=True, hide_index=True)
        st.caption("So siehst du z. B., ob CAD-Longs in Stress-Phasen schlechter laufen als in ruhigen Phasen.")
    with tabs[2]:
        if not open_holdout:
            st.info(f"Holdout ab {bp.HOLDOUT_START} ist versiegelt ({len(view_oos)} Trades). Erst Regel festlegen "
                    "(Methode, Timeframe, Märkte), dann oben öffnen – sonst ist der Test wertlos.")
        elif view_oos.empty:
            st.info("Keine Trades im Holdout.")
        else:
            st.warning("Holdout geöffnet: Ab jetzt nichts mehr an den Einstellungen optimieren.")
            st.dataframe(_fmt(comparison_table(view_oos, side_col, "market")), use_container_width=True, hide_index=True)
            so_m, so_g = stats(view_oos[view_oos[side_col] == "mit"].R), stats(view_oos[view_oos[side_col] == "gegen"].R)
            st.caption(f"Out-of-Sample gesamt: mit {so_m['n']} Trades E[R] {so_m.get('er', float('nan')):+.3f} · "
                       f"gegen {so_g['n']} Trades E[R] {so_g.get('er', float('nan')):+.3f}")
    with tabs[3]:
        cols = ["market", "sym", "dir", "fdir", "fill_dt", "vix", "zone_eff", "app", "corr_vix", "side_a", "side_b", "R"]
        st.dataframe(view[[c for c in cols if c in view]].sort_values("fill_dt", ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button("📥 Trades als CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name=f"makro_technik_{tf}_k{k}_e{entry}.csv", mime="text/csv", key="mt_dl")
