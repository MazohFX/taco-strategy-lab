"""
TACO Lab Modul: Key Level -> BOS -> Fib-Zone -> Konfluenzen (Tobis Checkliste)

Rechenkern: bos_research/bos_pipeline.py (Swings/BOS) und bos_research/full_setup.py (Trades).
Hier: interaktive Filter, Schwellen-Scan, Key-Level- und Konfluenz-Vergleich, Stabilitaet je Paar/Jahr.
Holdout (ab 2024) ist standardmaessig versiegelt.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bos_research"))
import bos_pipeline as bp  # noqa: E402
import full_setup as fs  # noqa: E402
import hypothesis_scan as hs  # noqa: E402

TF_OPTIONS = ["D1", "H4", "H1", "M15"]
KL_SOURCES = {"Jahr-1 Profil (POC/VAH/VAL)": "kl_prof_Y", "Quartal-1 Profil": "kl_prof_Q",
              "Monat-1 Profil": "kl_prof_M", "Jahres-Open": "kl_open_Y", "Quartals-Open": "kl_open_Q",
              "Monats-Open": "kl_open_M"}
CONF_LABELS = {"Monats-Open": "M", "3-Monats-Open": "Q", "Jahres-Open": "Y", "Wochen-Open": "W",
               "Tages-Open": "D", "POC lfd. Monat": "POC", "VAH lfd. Monat": "VAH", "VAL lfd. Monat": "VAL"}
DEFAULTS = dict(bos_kl=[], bos_tol=0.5, bos_zone="0.618–0.786", bos_conf=[], bos_conf_mode="alle gewählten",
                bos_conf_min=0, bos_imp=(0.0, 10.0), bos_trend="Alle", bos_dir="Beide")
CHECKLIST = dict(bos_kl=["Jahr-1 Profil (POC/VAH/VAL)", "Quartal-1 Profil", "Jahres-Open", "Quartals-Open"],
                 bos_tol=0.5, bos_zone="0.618–1", bos_conf=[], bos_conf_mode="alle gewählten", bos_conf_min=1,
                 bos_imp=(0.0, 10.0), bos_trend="Nur Gegentrend / CHoCH", bos_dir="Beide")


def _available(tf: str) -> dict:
    out = {}
    for s in bp.FX:
        df = bp.load(s, tf, str(ROOT))
        if df is not None and len(df) >= 500:
            out[s] = (f"{df.dt.iloc[0]:%Y-%m-%d}", f"{df.dt.iloc[-1]:%Y-%m-%d}", len(df))
    return out


@st.cache_data(show_spinner=False)
def _availability_cached(tf: str, stamp: float):
    return _available(tf)


@st.cache_data(show_spinner=False)
def compute_trades(tf: str, k: float, entry: float, pairs: tuple, stamp: float) -> pd.DataFrame:
    rows = []
    for s in pairs:
        df = bp.load(s, tf, str(ROOT))
        if df is None or len(df) < 500:
            continue
        rows += fs.run(s, df, bp.events(s, df, k), entry)
    return pd.DataFrame(rows)


def _data_stamp(tf: str) -> float:
    folder = ROOT / "data" / ("mt5" if tf == "D1" else "dukascopy_intraday")
    return max((f.stat().st_mtime for f in list(folder.glob("*.csv")) + list(folder.glob("*.csv.gz"))), default=0.0)


def _stats(r: pd.Series, dates: pd.Series) -> dict:
    n = len(r)
    if n == 0:
        return dict(n=0)
    yrs = max((dates.max() - dates.min()).days / 365.25, 1.0)
    mean = r.mean(); sd = r.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    tstat = mean / se if se and se > 0 else float("nan")
    p = math.erfc(abs(tstat) / math.sqrt(2)) if not math.isnan(tstat) else float("nan")
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    streak = cur = 0
    for x in r.values:
        cur = cur + 1 if x < -0.05 else 0; streak = max(streak, cur)
    return dict(n=n, per_year=n / yrs, win=(r > 0.05).mean() * 100, be=(r.abs() <= 0.05).mean() * 100,
                er=mean, se=se, t=tstat, p=p, total=r.sum(), maxdd=dd, streak=streak)


def _verdict(s: dict) -> tuple[str, str]:
    if s.get("n", 0) < 100:
        return "⚪", "Zu wenige Trades (< 100) für eine Aussage"
    if s["er"] > 0 and s["p"] <= 0.05:
        return "🟢", "Positiver Erwartungswert, statistisch signifikant (p ≤ 0,05)"
    if s["er"] > 0 and s["p"] <= 0.10:
        return "🟡", "Positiv, aber nur grenzwertig signifikant (0,05 < p ≤ 0,10)"
    if s["er"] > 0:
        return "🟠", "Leicht positiv, aber nicht von Zufall unterscheidbar"
    return "🔴", "Kein Vorteil (Erwartungswert ≤ 0)"


def _stats_table(groups: dict) -> pd.DataFrame:
    rows = []
    for name, g in groups.items():
        if len(g) == 0:
            rows.append({"Gruppe": name, "Trades": 0}); continue
        s = _stats(g.R, g.fill_dt)
        rows.append({"Gruppe": name, "Trades": s["n"], "Trades/Jahr": round(s["per_year"], 1),
                     "Gewinn %": round(s["win"], 1), "E[R]": round(s["er"], 3), "± SE": round(s["se"], 3),
                     "p-Wert": round(s["p"], 3), "Summe R": round(s["total"], 1),
                     "Max DD (R)": round(s["maxdd"], 1), "Längste Verlustserie": s["streak"],
                     "Urteil": _verdict(s)[0]})
    return pd.DataFrame(rows)


def _kl_mask(f: pd.DataFrame, sources: list, tol: float) -> pd.Series:
    if not sources:
        return pd.Series(True, index=f.index)
    m = pd.Series(False, index=f.index)
    for s in sources:
        col = KL_SOURCES[s]
        if col in f:
            m |= f[col] <= tol
    return m


def _conf_mask(f, zone_sfx, conf, mode, cmin):
    m = pd.Series(True, index=f.index)
    cols = [f"{CONF_LABELS[c]}_{zone_sfx}" for c in conf if f"{CONF_LABELS[c]}_{zone_sfx}" in f]
    if cols:
        hit = f[cols]
        m &= hit.all(axis=1) if mode == "alle gewählten" else hit.any(axis=1)
    if cmin > 0:
        allc = [f"{v}_{zone_sfx}" for v in CONF_LABELS.values() if f"{v}_{zone_sfx}" in f]
        m &= f[allc].sum(axis=1) >= cmin
    return m


def _struct_mask(f, trend, dirsel):
    m = pd.Series(True, index=f.index)
    if trend == "Nur mit Trend (HH+HL)":
        m &= (f.hh == 1) & (f.hl == 1)
    elif trend == "Nur Gegentrend / CHoCH":
        m &= (f.hh == 0) | (f.hl == 0)
    if dirsel != "Beide":
        m &= f.dir == (1 if dirsel == "Long" else -1)
    return m


def _equity(sel, base):
    fig = go.Figure()
    s = sel.sort_values("fill_dt"); b = base.sort_values("fill_dt")
    fig.add_scatter(x=s.fill_dt, y=s.R.cumsum(), mode="lines", name="Auswahl")
    fig.add_scatter(x=b.fill_dt, y=b.R.cumsum(), mode="lines", name="Alle gefüllten Trades", line=dict(dash="dot"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="kumuliert (R)", title="Equity in R")
    return fig


def render_bos_setup() -> None:
    mode = st.radio("Ansicht", ["🔬 Hypothesen-Scanner (automatisch)", "🔎 Einzelanalyse (manuell)"],
                    horizontal=True, key="bos_mode")
    if mode.startswith("🔬"):
        render_scanner()
        return
    st.header("🧭 Key Level → BOS → Fib-Zone → Konfluenzen")
    for k_, v_ in DEFAULTS.items():
        st.session_state.setdefault(k_, v_)

    with st.expander("ℹ️ Was wird hier getestet?", expanded=False):
        st.markdown(f"""
**Checkliste, Schritt für Schritt (mechanisch, ohne Repaint):**
1. **Key Level (optional als Filter):** Der Ursprung des Impulses (Fib 1) liegt höchstens *Toleranz × ATR* von einem
   Level der höheren Zeitebene entfernt: POC/VAH/VAL des Volumenprofils der **vorherigen** Periode (Jahr, Quartal,
   Monat; Value Area 68 %) oder der Jahres-/Quartals-/Monats-Open.
2. **BOS:** ATR-ZigZag (Swing bestätigt nach *k × ATR(14)* Rücklauf), erster **Schlusskurs** jenseits des letzten
   bestätigten Swings. *Gegentrend / CHoCH* = Umkehr am Level, wie in der Checkliste gedacht.
3. **Fib-Einstieg:** Fib 1 = Ursprung, Fib 0 = Extrem (läuft bis zur Füllung weiter). Limit am 0.618 oder 0.786,
   verfällt nach {fs.FILL_BARS} Kerzen oder wenn Fib 1 vorher erreicht wird.
4. **Konfluenzen** in der Zone (0.618–0.786 oder 0.618–1): Jahres-/Quartals-/Monats-/Wochen-/Tages-Open, POC/VAH/VAL
   des laufenden Monats (Monatsbeginn → Kerze vor dem Einstieg).
5. **Management:** Stop Fib 1 · drei gleiche Teile mit TP −0.27 / −0.62 / −1 · Break-even, sobald Fib 0 überschritten
   ist · Stop und Ziel in derselben Kerze = Stop · Timeout {fs.HOLD_BARS} Kerzen · Spread als Kosten.

**Ergebnis in R** (1 R = Abstand Einstieg → Stop). **Holdout ab {bp.HOLDOUT_START}** ist standardmäßig ausgeblendet.
Alle Konfluenz- und Key-Level-Werte nutzen nur Daten, die zum jeweiligen Zeitpunkt bekannt waren.
""")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("📋 Video-Checkliste laden", key="bos_preset"):
        for k_, v_ in CHECKLIST.items():
            st.session_state[k_] = v_
        st.rerun()
    if b2.button("↺ Filter zurücksetzen", key="bos_reset"):
        for k_, v_ in DEFAULTS.items():
            st.session_state[k_] = v_
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    tf = c1.selectbox("Timeframe", TF_OPTIONS, index=0, key="bos_tf")
    k = c2.select_slider("Swing-Schwelle (× ATR)", [1.0, 1.5, 2.0, 2.5, 3.0], value=1.5, key="bos_k")
    entry = c3.radio("Einstieg", [0.618, 0.786], horizontal=True, key="bos_entry")
    open_holdout = c4.checkbox("Holdout öffnen (ab 2024)", value=False, key="bos_holdout")
    if open_holdout:
        st.warning("Holdout geöffnet: Jede Anpassung, die du jetzt noch vornimmst, macht den Out-of-Sample-Test wertlos.")

    stamp = _data_stamp(tf)
    avail = _availability_cached(tf, stamp)
    if not avail:
        st.error(f"Keine {tf}-Daten gefunden. Für H4/H1/M15 den Import starten (Import_starten.command).")
        return
    with st.expander(f"📦 Datenabdeckung {tf} ({len(avail)} Paare)", expanded=False):
        st.dataframe(pd.DataFrame(avail, index=["von", "bis", "Kerzen"]).T, width="stretch")
        if tf != "D1":
            st.caption("Der Import läuft ggf. noch – Ergebnisse werden mit jedem Neuladen vollständiger.")
    pairs = st.multiselect("Paare", list(avail), default=list(avail), key=f"bos_pairs_{tf}")
    if not pairs:
        return

    with st.spinner("Berechne BOS-Ereignisse und Trades … (M15 kann beim ersten Mal 1–2 Minuten dauern)"):
        t = compute_trades(tf, float(k), float(entry), tuple(sorted(pairs)), stamp)
    if t.empty:
        st.info("Keine BOS-Ereignisse gefunden."); return
    if not open_holdout:
        t = t[(t.dt < bp.HOLDOUT_START) & ~(t.get("fill_dt", t.dt) >= bp.HOLDOUT_START)]
    f = t[t.filled == 1].copy()
    if f.empty:
        st.info("Keine gefüllten Trades."); return

    st.subheader("Checkliste / Filter")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**1 · Key Level** (leer = keine Bedingung)")
        kl = st.multiselect("Ursprung (Fib 1) an …", list(KL_SOURCES), key="bos_kl")
        tol = st.slider("Toleranz (× ATR)", 0.1, 2.0, step=0.1, key="bos_tol")
        st.markdown("**2 · Struktur / BOS**")
        trend = st.radio("Struktur", ["Alle", "Nur mit Trend (HH+HL)", "Nur Gegentrend / CHoCH"], key="bos_trend")
    with g2:
        st.markdown("**3 · Fib-Zone & Konfluenzen**")
        zone = st.radio("Zone für Konfluenzen", ["0.618–0.786", "0.618–1"], horizontal=True, key="bos_zone")
        conf = st.multiselect("Konfluenzen in der Zone", list(CONF_LABELS), key="bos_conf")
        mode = st.radio("Bedingung", ["alle gewählten", "mindestens eine"], horizontal=True, key="bos_conf_mode")
        cmin = st.number_input("Mindestanzahl beliebiger Konfluenzen", 0, 8, step=1, key="bos_conf_min")
    with g3:
        st.markdown("**Weitere Filter**")
        imp = st.slider("Impulsbein beim Bruch (× ATR)", 0.0, 10.0, step=0.25, key="bos_imp")
        dirsel = st.radio("Richtung", ["Beide", "Long", "Short"], horizontal=True, key="bos_dir")
    zsfx = "n" if zone == "0.618–0.786" else "w"

    hi = np.inf if imp[1] >= 10 else imp[1]
    m_kl = _kl_mask(f, kl, tol)
    m_conf = _conf_mask(f, zsfx, conf, mode, cmin)
    m_struct = _struct_mask(f, trend, dirsel)
    m_imp = (f.impulse_atr >= imp[0]) & (f.impulse_atr < hi)
    sel = f[m_kl & m_conf & m_struct & m_imp]

    s = _stats(sel.R, sel.fill_dt) if len(sel) else dict(n=0)
    icon, text = _verdict(s) if s["n"] else ("⚪", "Keine Trades")
    st.markdown(f"### {icon} {text}")
    if s["n"]:
        m = st.columns(7)
        m[0].metric("Trades", f"{s['n']}", f"{s['per_year']:.1f}/Jahr", delta_color="off")
        m[1].metric("E[R] pro Trade", f"{s['er']:+.3f}", f"± {s['se']:.3f}", delta_color="off")
        m[2].metric("p-Wert", f"{s['p']:.3f}")
        m[3].metric("Gewinn-Trades", f"{s['win']:.1f} %", f"{s['be']:.0f} % ~Break-even", delta_color="off")
        m[4].metric("Summe", f"{s['total']:+.1f} R")
        m[5].metric("Max Drawdown", f"{s['maxdd']:.1f} R")
        m[6].metric("Längste Verlustserie", f"{s['streak']}")
        st.caption(f"BOS gesamt: {len(t)} · gefüllt: {len(f)} ({len(f)/max(len(t),1):.0%}) · nach Filter: {len(sel)}")
        st.plotly_chart(_equity(sel, f), width="stretch")

    tabs = st.tabs(["🎯 Key Level", "📏 BOS-Größe", "🧩 Konfluenzen", "💱 Je Paar", "📅 Je Jahr", "📄 Trades"])

    with tabs[0]:
        st.markdown("**Hilft das Key Level?** Vergleich bei sonst gleichen Filtern (Struktur, Konfluenzen, Impuls).")
        base = f[m_conf & m_struct & m_imp]
        groups = {"Ohne Key-Level-Bedingung": base}
        for name, col in KL_SOURCES.items():
            if col in base:
                groups[f"{name} ≤ {tol:g} ATR"] = base[base[col] <= tol]
        if kl:
            groups[f"Auswahl ({len(kl)} Quellen) ≤ {tol:g} ATR"] = base[_kl_mask(base, kl, tol)]
            groups["Auswahl NICHT erfüllt"] = base[~_kl_mask(base, kl, tol)]
        st.dataframe(_stats_table(groups), width="stretch", hide_index=True)
        if kl:
            rows = []
            for x in np.arange(0.1, 2.01, 0.1):
                r = base[_kl_mask(base, kl, x)].R
                if len(r) >= 30:
                    rows.append((x, r.mean(), 1.96 * r.std() / math.sqrt(len(r)), len(r)))
            if rows:
                d = pd.DataFrame(rows, columns=["tol", "er", "ci", "n"])
                fig = go.Figure()
                fig.add_scatter(x=d.tol, y=d.er + d.ci, mode="lines", line=dict(width=0), showlegend=False)
                fig.add_scatter(x=d.tol, y=d.er - d.ci, mode="lines", line=dict(width=0), fill="tonexty", name="95 %-Band")
                fig.add_scatter(x=d.tol, y=d.er, mode="lines+markers", name="E[R]", text=d.n,
                                hovertemplate="Toleranz %{x:.1f} ATR<br>E[R] %{y:.3f}<br>n=%{text}")
                fig.add_hline(y=0, line_dash="dash")
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Key-Level-Toleranz (× ATR)", yaxis_title="E[R] pro Trade")
                st.plotly_chart(fig, width="stretch")
                st.caption("Ein echtes Key-Level-Signal zeigt sich als bessere Werte bei enger Toleranz, die mit wachsender Toleranz abflachen.")

    with tabs[1]:
        st.markdown("**Ab welcher Bruchgröße wird der Trade besser?** Alle übrigen Filter aktiv, nur der Impuls-Regler wird ignoriert.")
        base = f[m_kl & m_conf & m_struct]
        rows = []
        for x in np.arange(1.0, 6.01, 0.25):
            r = base[base.impulse_atr >= x].R
            if len(r) >= 30:
                rows.append((x, r.mean(), 1.96 * r.std() / math.sqrt(len(r)), len(r)))
        if rows:
            d = pd.DataFrame(rows, columns=["x", "er", "ci", "n"])
            fig = go.Figure()
            fig.add_scatter(x=d.x, y=d.er + d.ci, mode="lines", line=dict(width=0), showlegend=False)
            fig.add_scatter(x=d.x, y=d.er - d.ci, mode="lines", line=dict(width=0), fill="tonexty", name="95 %-Band")
            fig.add_scatter(x=d.x, y=d.er, mode="lines+markers", name="E[R]", text=d.n,
                            hovertemplate="Impuls ≥ %{x} ATR<br>E[R] %{y:.3f}<br>n=%{text}")
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Mindest-Impuls beim Bruch (× ATR)", yaxis_title="E[R] pro Trade")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Zu wenige Trades für den Schwellen-Scan (mind. 30 je Punkt).")
        buckets = [(0, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 6), (6, np.inf)]
        st.dataframe(_stats_table({f"{a:g}–{b:g} ATR": base[(base.impulse_atr >= a) & (base.impulse_atr < b)]
                                   for a, b in buckets}), width="stretch", hide_index=True)

    with tabs[2]:
        st.markdown(f"**Welche Konfluenz bringt etwas?** Zone {zone}, Key Level und Struktur wie oben gefiltert.")
        base = f[m_kl & m_struct & m_imp]
        groups = {"Alle (mit Key Level/Struktur-Filter)": base}
        for lab, c in CONF_LABELS.items():
            col = f"{c}_{zsfx}"
            if col in base:
                groups[f"{lab} in Zone"] = base[base[col] == 1]
        allc = [f"{v}_{zsfx}" for v in CONF_LABELS.values() if f"{v}_{zsfx}" in base]
        cnt = base[allc].sum(axis=1)
        for nmin in (0, 1, 2, 3):
            groups[f"genau {nmin} Konfluenzen" if nmin < 3 else "3+ Konfluenzen"] = base[cnt == nmin] if nmin < 3 else base[cnt >= 3]
        st.dataframe(_stats_table(groups), width="stretch", hide_index=True)

    with tabs[3]:
        st.dataframe(_stats_table({p: sel[sel.sym == p] for p in sorted(sel.sym.unique())}), width="stretch", hide_index=True)
        st.caption("Ein echter Vorteil sollte auf den meisten Paaren positiv sein, nicht nur auf einzelnen.")

    with tabs[4]:
        yrs = sel.fill_dt.dt.year if len(sel) else pd.Series(dtype=int)
        st.dataframe(_stats_table({str(y): sel[yrs == y] for y in sorted(yrs.unique())}), width="stretch", hide_index=True)

    with tabs[5]:
        cols = ["sym", "dir", "dt", "fill_dt", "entry_px", "stop_px", "impulse_atr", "kl_prof_type", "kl_prof_dist",
                "kl_open_type", "kl_open_dist"] + [f"{v}_{zsfx}" for v in CONF_LABELS.values()] + ["hh", "hl", "be", "R"]
        st.dataframe(sel[[c for c in cols if c in sel]], width="stretch", hide_index=True)
        st.download_button("📥 Trades als CSV", sel.to_csv(index=False).encode("utf-8"),
                           file_name=f"bos_setup_{tf}_k{k}_e{entry}.csv", mime="text/csv", key="bos_dl")



@st.cache_data(show_spinner=False)
def _scan_trades(tf: str, stamp: float):
    bar = st.progress(0.0, text="Berechne Trades für alle Swing-Schwellen und Einstiege …")
    tr = hs.load_trades(tf, str(ROOT), progress=lambda x, t: bar.progress(x, text=t))
    bar.empty()
    return tr


@st.cache_data(show_spinner=False)
def _scan_run(tf: str, stamp: float, split: str, min_n: int, q: float):
    tr = _scan_trades(tf, stamp)
    return hs.scan(tr, split, min_n_disc=min_n, q=q)


def render_scanner() -> None:
    st.header("🔬 Hypothesen-Scanner")
    with st.expander("ℹ️ Wie funktioniert das – und was bedeutet der p-Wert?", expanded=False):
        st.markdown("""
**p-Wert:** Wie wahrscheinlich wäre ein mindestens so gutes Ergebnis, wenn die Kombination in Wahrheit *keinen*
Vorteil hätte (E[R] = 0)? p = 0,01 heißt: nur 1 % Zufallswahrscheinlichkeit – **aber nur für EINEN vorher
festgelegten Test.**

**Das Problem beim Durchprobieren:** Testest du 10.000 Kombinationen, liefern ~500 davon *rein zufällig* p < 0,05.
Deshalb gilt hier:
1. **Entdeckung** (ältere Daten): Alle Kombinationen werden getestet, dann **Benjamini-Hochberg-Korrektur** über
   alle Tests (FDR q: höchstens q-Anteil der „Treffer" darf Zufall sein).
2. **Bestätigung** (neuere, ungesehene Daten): Nur die Überlebenden werden erneut getestet und wieder korrigiert.
3. **Robustheit:** ≥ 60 % der Paare und Jahre positiv, Nachbar-Swing-Schwellen nicht negativ.
4. **Holdout ab 2024** bleibt versiegelt – erst öffnen, wenn du *eine* Kandidaten-Regel endgültig festlegst.

**Faustregeln:** Einzelner, vorher festgelegter Test: p < 0,05. Ergebnis aus vielen Tests: korrigiert (BH) bestehen
oder t-Wert ≥ 3 (Harvey, Liu & Zhu 2016). Wichtiger als jeder p-Wert: **hält es auf Daten, die beim Suchen nicht
benutzt wurden?** Eine „100 %-Edge" gibt es nicht – nur Kandidaten, die alle Hürden überstehen.
""")
    c1, c2, c3, c4 = st.columns(4)
    tf = c1.selectbox("Timeframe", TF_OPTIONS, key="scan_tf")
    default_split = {"D1": 2017}.get(tf, 2020)
    split_year = c2.number_input("Bestätigung ab Jahr", 2010, 2023, default_split, key=f"scan_split_{tf}")
    min_n = c3.number_input("Min. Trades in Entdeckung", 30, 1000, 100, step=10, key="scan_minn")
    q = c4.select_slider("FDR q", [0.05, 0.10, 0.20], value=0.10, key="scan_q")
    st.caption(f"Entdeckung: alles vor {split_year} · Bestätigung: {split_year}–2023 · Holdout ab {bp.HOLDOUT_START[:4]} versiegelt. "
               f"Getestet werden {len(hs.K_GRID)} Swing-Schwellen × {len(hs.ENTRY_GRID)} Einstiege × Key Level × Toleranz × "
               f"Struktur × Mindest-Impuls × Konfluenz (Zone 0.618–0.786 und 0.618–1).")
    if not st.button("▶️ Scan starten", key="scan_go", type="primary") and not st.session_state.get(f"scan_done_{tf}"):
        st.info("Klick auf „Scan starten“. D1 dauert ca. 30 Sekunden, M15 mehrere Minuten (danach zwischengespeichert).")
        return
    st.session_state[f"scan_done_{tf}"] = True
    stamp = _data_stamp(tf)
    with st.spinner("Teste alle Kombinationen …"):
        res, info = _scan_run(tf, stamp, f"{int(split_year)}-01-01", int(min_n), float(q))
    if res is None or len(res) == 0:
        st.warning("Keine Kombination mit genügend Trades."); return

    m = st.columns(5)
    m[0].metric("Kombinationen", f"{info['n_tested']:,}".replace(",", "."), help="Anzahl getesteter Filterkombinationen")
    m[1].metric("Roh p < 0,05", info["n_raw_p05"], help="Treffer OHNE Korrektur für viele Tests")
    m[2].metric("Zufall erwartet", f"≈ {info['expected_false_at_05']}",
                help="So viele Roh-Treffer liefert reiner Zufall bei dieser Anzahl Tests (5 %)")
    m[3].metric("Nach Korrektur", info["n_bh_disc"],
                help="Überleben die Benjamini-Hochberg-Korrektur im Entdeckungszeitraum")
    m[4].metric("Bestätigt", info["n_bh_val"], help="Halten zusätzlich auf den neueren, ungesehenen Jahren")
    if info["n_bh_val"] == 0:
        st.error("**Keine Kombination besteht Entdeckung + Bestätigung.** Die Treffer mit kleinen p-Werten sind mit "
                 "hoher Wahrscheinlichkeit Zufall (Data Mining). Die Liste unten zeigt die besten Rohwerte zur Einordnung.")
    else:
        st.success(f"{info['n_bh_val']} Kombination(en) bestehen beide Stufen – jetzt Robustheit prüfen (Spalte Urteil).")

    trades = _scan_trades(tf, stamp)
    top = res.sort_values(["bh_val", "bh_disc", "p_disc"], ascending=[False, False, True]).head(30).copy()
    rob = [hs.robustness(trades, r) for _, r in top.iterrows()]
    for key in ("pairs_pos", "years_pos", "neighbours_pos"):
        top[key] = [x[key] for x in rob]
    top["Urteil"] = [hs.verdict(r) for _, r in top.iterrows()]
    show = top.rename(columns={"k": "Swing×ATR", "entry": "Einstieg", "key_level": "Key Level", "tol": "Tol ATR",
                               "struktur": "Struktur", "impuls_min": "Impuls ≥", "konfluenz": "Konfluenz",
                               "zone": "Zone", "n_disc": "n Entd.", "er_disc": "E[R] Entd.", "p_disc": "p Entd.",
                               "n_val": "n Best.", "er_val": "E[R] Best.", "p_val": "p Best.",
                               "pairs_pos": "Paare +", "years_pos": "Jahre +", "neighbours_pos": "Nachbarn +"})
    cols = ["Urteil", "Swing×ATR", "Einstieg", "Key Level", "Tol ATR", "Struktur", "Impuls ≥", "Konfluenz", "Zone",
            "n Entd.", "E[R] Entd.", "p Entd.", "n Best.", "E[R] Best.", "p Best.", "Paare +", "Jahre +", "Nachbarn +"]
    st.subheader("Top 30 Hypothesen")
    disp = show[cols].reset_index(drop=True).copy()
    disp["Tol ATR"] = disp["Tol ATR"].map(lambda x: "–" if x is None or pd.isna(x) else f"{x:.2f}")
    disp["Swing×ATR"] = disp["Swing×ATR"].map(lambda x: f"{x:.1f}")
    disp["Einstieg"] = disp["Einstieg"].map(lambda x: f"{x:.3f}")
    disp["Impuls ≥"] = disp["Impuls ≥"].map(lambda x: "–" if x == 0 else f"{x:.0f} ATR")
    disp["Urteil"] = disp["Urteil"].replace({"⚪ Nicht signifikant nach Korrektur": "⚪ Zufall möglich",
                                             "🔴 In Bestätigung durchgefallen": "🔴 Bestätigung ✗",
                                             "🟡 Bestätigung schwach": "🟡 Bestätigung schwach"})
    for c_ in ("E[R] Entd.", "E[R] Best."):
        disp[c_] = disp[c_].round(3)
    disp["p Entd."] = disp["p Entd."].round(5); disp["p Best."] = disp["p Best."].round(3)
    for c_ in ("Paare +", "Jahre +", "Nachbarn +"):
        disp[c_] = (disp[c_] * 100).round(0)
    disp.index = disp.index + 1
    st.dataframe(disp, width="stretch", column_config={
        c_: st.column_config.NumberColumn(c_, format="%d %%") for c_ in ("Paare +", "Jahre +", "Nachbarn +")})

    st.subheader("Detail einer Hypothese")
    idx = st.selectbox("Rang", list(range(1, len(top) + 1)), key="scan_pick")
    r = top.iloc[idx - 1]; sel = rob[idx - 1]["sel"].sort_values("fill_dt")
    split_ts = pd.Timestamp(f"{int(split_year)}-01-01")
    fig = go.Figure()
    fig.add_scatter(x=sel.fill_dt, y=sel.R.cumsum(), mode="lines", name="Equity (R)")
    fig.add_vline(x=split_ts, line_dash="dash")
    fig.add_annotation(x=split_ts, y=1, yref="paper", text="Beginn Bestätigung", showarrow=False, xanchor="left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="kumuliert (R)")
    st.plotly_chart(fig, width="stretch")
    d1, d2 = st.columns(2)
    d1.dataframe(_stats_table({p_: sel[sel.sym == p_] for p_ in sorted(sel.sym.unique())}), width="stretch", hide_index=True)
    yrs = sel.fill_dt.dt.year
    d2.dataframe(_stats_table({str(y): sel[yrs == y] for y in sorted(yrs.unique())}), width="stretch", hide_index=True)
