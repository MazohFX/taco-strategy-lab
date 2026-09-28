"""
DAX Freitag-Short — Szenario-Scanner + Walk-Forward-Analyse.

Python-Nachbau der Pine-Strategie DAX_Freitag_Short_Strategy.pine:
  - Jeden Freitag Short zum Open der Kerze, die zur Einstiegszeit (Europe/Berlin) beginnt
  - Zeit-Exit zum Open der Kerze, die zur Ausstiegszeit beginnt
  - Stop Loss = Einstieg + X % (intrabar über das High geprüft)
  - Sicherheits-Exit zum Close der letzten Freitagskerze, falls keine Kerze zur Ausstiegszeit existiert
  - Kein Trade, wenn die Einstiegskerze fehlt (Feiertag / Datenlücke)

Daten: data/mt5_intraday/GER40_FRI_M15.csv (Dukascopy, BID-Kurse, UTC), erzeugt mit
dukascopy_ger40_friday_import.py. H1 wird aus M15 auf Berliner Stunden resampled.

Kostenmodell (BID-Daten):
  Short-Entry verkauft zum Bid (= Chartkurs), Exit kauft zum Ask (= Bid + Spread).
  SL ist ein Ask-Level (Einstieg × (1 + SL%)) → löst auf dem Bid-Chart bei SL − Spread aus.
  Kommission als % Notional pro Round-Turn.

Die komplette Engine arbeitet auf Tages-Arrays (Freitage × Kerzen-Slots), dadurch lassen sich
alle Einstiegs-/Ausstiegszeiten × SL-Stufen vektorisiert in Sekunden durchrechnen.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TZ = "Europe/Berlin"
DATA_PATH = Path(__file__).parent / "data" / "mt5_intraday" / "GER40_FRI_M15.csv"
SL_STEPS = [round(0.20 + 0.05 * i, 2) for i in range(37)]  # 0.20 … 2.00
NO_SL = None
METRICS = {
    "pf":     "Profit Factor",
    "total":  "Summe Rendite %",
    "sharpe": "Sharpe",
    "ret_dd": "Rendite / Max DD",
}
KIND_LABEL = {0: "Zeit", 1: "SL", 2: "Sicherheit"}


# =============================================================================
# DATEN
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_arrays(tf_min: int, mtime: float) -> dict | None:
    """Liest die M15-CSV und baut Freitag × Slot-Arrays (Berliner Zeit)."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    ts = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(TZ)
    df = df.assign(ts=ts)
    df = df[df["ts"].dt.dayofweek == 4].drop_duplicates("ts").sort_values("ts")
    if df.empty:
        return None

    df["day"] = df["ts"].dt.tz_localize(None).dt.normalize()
    df["slot"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) // tf_min
    if tf_min > 15:
        df = df.groupby(["day", "slot"], sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), spread=("spread", "mean")).reset_index()

    days = np.sort(df["day"].unique())
    K = 1440 // tf_min
    D = len(days)
    d_idx = np.searchsorted(days, df["day"].values)
    s_idx = df["slot"].values.astype(int)

    arr = {}
    for col, key in (("open", "O"), ("high", "H"), ("low", "L"), ("close", "C"), ("spread", "S")):
        a = np.full((D, K), np.nan)
        a[d_idx, s_idx] = df[col].values
        arr[key] = a

    valid = ~np.isnan(arr["O"])
    # nächste vorhandene Kerze ab Slot s (−1 = keine mehr an diesem Freitag)
    nv = np.full((D, K + 1), -1, dtype=int)
    for s in range(K - 1, -1, -1):
        nv[:, s] = np.where(valid[:, s], s, nv[:, s + 1])
    has_any = valid.any(axis=1)
    last_idx = np.where(has_any, K - 1 - np.argmax(valid[:, ::-1], axis=1), 0)
    rows = np.arange(D)

    arr.update(days=days, K=K, tf_min=tf_min, valid=valid, nv=nv[:, :K],
               last_idx=last_idx, last_close=arr["C"][rows, last_idx], has_any=has_any)
    return arr


def _slot_label(slot: int, tf_min: int) -> str:
    m = slot * tf_min
    return f"{m // 60:02d}:{m % 60:02d}"


def _sl_label(p) -> str:
    return "ohne SL" if p is None else f"{p:.2f} %"


# =============================================================================
# ENGINE
# =============================================================================

def _spread_matrix(A: dict, spread_mode: str, spread_pts: float) -> np.ndarray:
    if spread_mode == "data":
        S = A["S"].copy()
        # Kerzen ohne Spread-Info (sollte nicht vorkommen) → Median
        S[np.isnan(S) & A["valid"]] = np.nanmedian(A["S"])
        return S
    return np.where(A["valid"], spread_pts, np.nan)


def _exit_prices(A: dict, S: np.ndarray, xs: np.ndarray):
    """Zeit-Exit je (Freitag, Exit-Slot): Ask-Preis + ob Kerze existiert + Limit-Slot für SL-Scan."""
    rows = np.arange(len(A["days"]))[:, None]
    exitk = A["nv"][:, xs]                               # (D, X)
    exists = exitk >= 0
    ek = np.where(exists, exitk, 0)
    li = A["last_idx"]
    ask = np.where(exists, A["O"][rows, ek] + S[rows, ek],
                   (A["last_close"] + S[rows[:, 0], li])[:, None])
    # SL wird bis zur Kerze VOR der Exit-Kerze geprüft, ohne Exit-Kerze bis inkl. letzter Kerze
    limit = np.where(exists, exitk, (A["last_idx"] + 1)[:, None])
    return ask, exists, limit


def _returns_for(A: dict, S: np.ndarray, e: int, p, xs: np.ndarray, exit_cache, comm_pct: float):
    """Rendite % (X, D) für Einstieg-Slot e, SL p und alle Exit-Slots xs (> e). NaN = kein Trade.
    kind (X, D): 0 Zeit-Exit, 1 SL, 2 Sicherheits-Exit."""
    O, H = A["O"], A["H"]
    D, K = O.shape
    rows = np.arange(D)
    entry = O[:, e]
    # Nur die Einstiegskerze muss existieren. (Pine braucht zusätzlich die Kerze davor als Signal-Kerze —
    # die fehlt bei Dukascopy 2015–2018 vor 08:00, beim Broker aber nicht; daher hier nicht verlangt.)
    ok = ~np.isnan(entry)
    s_e = S[:, e]

    ask_x, exists_x, limit_x = exit_cache
    if p is None:
        sl = np.zeros((D, len(xs)), dtype=bool)
        sl_ask = np.zeros(D)
    else:
        sl_level_ask = entry * (1.0 + p / 100.0)
        trig_bid = sl_level_ask - s_e
        with np.errstate(invalid="ignore"):
            hit = H[:, e:] >= trig_bid[:, None]
        any_hit = hit.any(axis=1)
        hitk = np.where(any_hit, e + hit.argmax(axis=1), K)
        sl = hitk[:, None] < limit_x
        hk = np.minimum(hitk, K - 1)
        # Gap über den SL → Fill zum Open der Trigger-Kerze
        sl_ask = np.maximum(O[rows, hk], trig_bid) + s_e

    exit_ask = np.where(sl, sl_ask[:, None], ask_x)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = (entry[:, None] - exit_ask) / entry[:, None] * 100.0 - comm_pct
    ret[~ok] = np.nan
    kind = np.where(sl, 1, np.where(exists_x, 0, 2))
    return ret.T, kind.T


def _metrics(R: np.ndarray) -> dict:
    """Kennzahlen je Zeile aus Rendite-Matrix (C, D) mit NaN = kein Trade."""
    tr = ~np.isnan(R)
    n = tr.sum(axis=1)
    Rz = np.where(tr, R, 0.0)
    gp = np.where(Rz > 0, Rz, 0.0).sum(axis=1)
    gl = -np.where(Rz < 0, Rz, 0.0).sum(axis=1)
    wins = (Rz > 0).sum(axis=1)
    total = Rz.sum(axis=1)
    nn = np.maximum(n, 1)
    mean = total / nn
    var = (np.where(tr, (Rz - mean[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(n - 1, 1)
    std = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = np.where(gl > 0, gp / gl, np.where(gp > 0, 9.99, 0.0))
        sharpe = np.where(std > 0, mean / std * np.sqrt(52), 0.0)
    cum = np.cumsum(Rz, axis=1)
    peak = np.maximum(np.maximum.accumulate(cum, axis=1), 0.0)
    dd = (peak - cum).max(axis=1) if R.shape[1] else np.zeros(len(R))
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_dd = np.where(dd > 0, total / dd, np.where(total > 0, 99.0, 0.0))
    return {"n": n, "wr": wins / nn * 100, "pf": np.minimum(pf, 9.99), "total": total,
            "avg": mean, "sharpe": sharpe, "dd": dd, "ret_dd": ret_dd}


@st.cache_data(show_spinner=False)
def _full_scan(tf_min: int, mtime: float, d_from, d_to, entry_slots: tuple, exit_slots: tuple,
               sl_list: tuple, spread_mode: str, spread_pts: float, comm_pct: float) -> dict:
    """Alle Einstiege × Ausstiege × SL-Stufen. Ergebnis-Arrays (E, X, P)."""
    A = _load_arrays(tf_min, mtime)
    mask = (A["days"] >= np.datetime64(d_from)) & (A["days"] <= np.datetime64(d_to))
    A = _subset(A, mask)
    S = _spread_matrix(A, spread_mode, spread_pts)
    xs_all = np.array(exit_slots)
    exit_cache = _exit_prices(A, S, xs_all)

    E, X, P = len(entry_slots), len(xs_all), len(sl_list)
    out = {k: np.full((E, X, P), np.nan) for k in ("n", "wr", "pf", "total", "avg", "sharpe", "dd", "ret_dd")}
    for ie, e in enumerate(entry_slots):
        sel = np.where(xs_all > e)[0]
        if not len(sel):
            continue
        cache_e = tuple(c[:, sel] for c in exit_cache)
        for ip, p in enumerate(sl_list):
            R, _ = _returns_for(A, S, e, p, xs_all[sel], cache_e, comm_pct)
            m = _metrics(R)
            for k in out:
                out[k][ie, sel, ip] = m[k]
    out["n_days"] = int(len(A["days"]))
    return out


def _subset(A: dict, mask: np.ndarray) -> dict:
    B = dict(A)
    for k in ("days", "O", "H", "L", "C", "S", "valid", "nv", "last_idx", "last_close", "has_any"):
        B[k] = A[k][mask]
    return B


def _scenario_trades(A: dict, S: np.ndarray, e: int, x: int, p, comm_pct: float) -> pd.DataFrame:
    xs = np.array([x])
    R, kind = _returns_for(A, S, e, p, xs, _exit_prices(A, S, xs), comm_pct)
    r, k = R[0], kind[0]
    m = ~np.isnan(r)
    return pd.DataFrame({"Datum": pd.to_datetime(A["days"][m]), "Rendite %": r[m],
                         "Exit": [KIND_LABEL[int(v)] for v in k[m]]})


def _smooth(M: np.ndarray, sl_list: tuple) -> np.ndarray:
    """Nachbarschafts-Mittel (±1 Einstieg, ±1 Ausstieg, ±1 SL-Stufe), NaN-tolerant.
    Die 'ohne SL'-Ebene wird separat (nur Zeit-Nachbarn) geglättet."""
    out = np.full_like(M, np.nan)
    sl_idx = [i for i, p in enumerate(sl_list) if p is not None]
    no_idx = [i for i, p in enumerate(sl_list) if p is None]
    for idx, span_p in ((sl_idx, 1), (no_idx, 0)):
        if not idx:
            continue
        sub = M[:, :, idx]
        pad = np.pad(sub, ((1, 1), (1, 1), (span_p, span_p)), constant_values=np.nan)
        stack = []
        E, X, P = sub.shape
        for de in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for dp in range(-span_p, span_p + 1):
                    stack.append(pad[1 + de:1 + de + E, 1 + dx:1 + dx + X, span_p + dp:span_p + dp + P])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sm = np.nanmean(np.stack(stack), axis=0)
        sm[np.isnan(sub)] = np.nan
        out[:, :, idx] = sm
    return out


# =============================================================================
# UI-HELFER
# =============================================================================

def _equity_fig(curves: list[tuple[str, pd.Series, str]], title: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for name, s, color in curves:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, line=dict(color=color, width=2)))
    fig.update_layout(title=title, height=height, template="plotly_dark", margin=dict(t=40, b=15),
                      yaxis_title="kumulierte Rendite %", legend=dict(orientation="h", y=-0.15))
    return fig


def _banner(color: str, text: str, sub: str = "") -> None:
    st.markdown(
        f'<div style="background:{color}18;border:1px solid {color}66;border-radius:10px;'
        f'padding:14px 18px;margin:8px 0 14px 0;">'
        f'<div style="color:{color};font-weight:800;font-size:1.1rem;">{text}</div>'
        f'<div style="color:#9fb0c7;font-size:.85rem;margin-top:4px;">{sub}</div></div>',
        unsafe_allow_html=True)


def _fmt_scn(s: dict, tf_min: int) -> str:
    return (f"Short {_slot_label(s['e'], tf_min)} → Exit {_slot_label(s['x'], tf_min)} · "
            f"SL {_sl_label(s['p'])}")


# =============================================================================
# SEITE
# =============================================================================

def render_dax_friday_short(save_fn=None, load_fn=None, evaluate_edge_fn=None) -> None:
    st.header("DAX Freitag Short — Szenario-Scanner & WFA")

    with st.expander("ℹ️ Was wird hier getestet?", expanded=False):
        st.markdown("""
**Strategie** (1:1 wie `DAX_Freitag_Short_Strategy.pine`): jeden Freitag Short am **Open der Kerze, die zur
Einstiegszeit beginnt** (Berliner Zeit inkl. Sommer-/Winterzeit), Ausstieg am **Open der Kerze zur Ausstiegszeit**.
Stop Loss = Einstieg + X %, intrabar über das Kerzen-High geprüft (bei Gap über den SL: Fill zum Open).
Fehlt die Kerze zur Ausstiegszeit → Sicherheits-Exit zum Close der letzten Freitagskerze. Nie übers Wochenende.

**Daten:** Dukascopy GER40 (DEU.IDX) Tick-Daten → M15-Kerzen, nur Freitage. H1 wird daraus auf volle Berliner
Stunden zusammengefasst. Die Kurse sind **Bid**: Short verkauft zum Bid, der Exit kauft zum Ask (Bid + Spread).
Der SL ist ein Ask-Level und löst auf dem Bid-Chart daher schon bei *SL − Spread* aus.

**Datenabdeckung:** Dukascopy liefert den DAX-CFD 2015–2018 erst ab 08:00 — Einstiege vor 08:00 haben daher
weniger Trades (nur ~75 % der Freitage). Der Filter „Min. Trades“ berücksichtigt das.

**Scanner:** testet **jede** Einstiegszeit × **jede** Ausstiegszeit × **jede** SL-Stufe (0,20–2,00 % in 0,05er
Schritten, optional ohne SL). Bei M15 sind das bis zu ~170.000 Szenarien.

**Vorsicht Overfitting:** Wer 170.000 Varianten testet, findet immer etwas, das rückblickend gut aussieht.
Deshalb gibt es (1) die **Robustheits-Wertung**: Ein Szenario wird nach dem Durchschnitt seiner Nachbarn bewertet
(±1 Kerze Einstieg, ±1 Kerze Ausstieg, ±1 SL-Stufe). Ein einzelner Glückstreffer fällt damit raus. Und (2) den
**WFA-Reiter**, der ein gewähltes Szenario auf unbekannten Daten prüft.
""")

    mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
    if not DATA_PATH.exists():
        st.error("`data/mt5_intraday/GER40_FRI_M15.csv` fehlt — bitte zuerst "
                 "`python3 dukascopy_ger40_friday_import.py --start 2014-01-01` ausführen.")
        return

    # ── Gemeinsame Einstellungen ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tf_label = st.radio("Timeframe", ["M15", "H1"], horizontal=True, key="dfs_tf")
        tf_min = 15 if tf_label == "M15" else 60
    A = _load_arrays(tf_min, mtime)
    if A is None or not len(A["days"]):
        st.error("Keine Freitags-Daten in der CSV gefunden.")
        return
    first_day = pd.Timestamp(A["days"][0]).date()
    last_day = pd.Timestamp(A["days"][-1]).date()
    with c2:
        d_from = st.date_input("Daten ab", first_day, min_value=first_day, max_value=last_day, key="dfs_from")
        d_to = st.date_input("Daten bis", last_day, min_value=first_day, max_value=last_day, key="dfs_to")
    with c3:
        spread_mode_lbl = st.radio("Spread", ["Fest", "Echter Dukascopy-Spread je Kerze"], key="dfs_spmode",
                                   help="Fest: konstanter Spread in Punkten. Echt: gemittelter Tick-Spread der "
                                        "Einstiegs- bzw. Ausstiegskerze (früh morgens deutlich breiter).")
        spread_mode = "data" if spread_mode_lbl.startswith("Echt") else "fixed"
        spread_pts = st.number_input("Spread (Punkte, Round-Turn)", 0.0, 20.0, 1.0, 0.1, key="dfs_spread",
                                     disabled=spread_mode == "data")
    with c4:
        comm_pct = st.number_input("Kommission (% Notional, Round-Turn)", 0.0, 1.0, 0.0, 0.005,
                                   format="%.3f", key="dfs_comm")
        st.caption(f"{len(A['days'])} Freitage verfügbar ({first_day} → {last_day})")

    n_days_cov = int(((A["days"] >= np.datetime64(d_from)) & (A["days"] <= np.datetime64(d_to))).sum())
    if n_days_cov < 20:
        st.warning("Zu wenige Freitage im gewählten Zeitraum.")
        return

    tab_scan, tab_wfa = st.tabs(["🔎 Szenario-Scanner (Top 3)", "🔄 Walk-Forward-Analyse"])

    with tab_scan:
        _render_scanner(A, tf_min, mtime, d_from, d_to, spread_mode, spread_pts, comm_pct)
    with tab_wfa:
        _render_wfa(A, tf_min, d_from, d_to, spread_mode, spread_pts, comm_pct,
                    save_fn, load_fn, evaluate_edge_fn)


# -----------------------------------------------------------------------------
# REITER 1: SCANNER
# -----------------------------------------------------------------------------

def _render_scanner(A, tf_min, mtime, d_from, d_to, spread_mode, spread_pts, comm_pct):
    K = A["K"]
    n_days_cov = int(((A["days"] >= np.datetime64(d_from)) & (A["days"] <= np.datetime64(d_to))).sum())
    # Nur Slots anbieten, in denen überhaupt gehandelt wird (≥ 50 % der Freitage mit Kerze)
    coverage = A["valid"].mean(axis=0)
    tradable = np.where(coverage >= 0.5)[0]
    lo_h = int(tradable.min() * tf_min // 60) if len(tradable) else 0
    hi_h = int(min(23, (tradable.max() * tf_min) // 60)) if len(tradable) else 23

    st.subheader("Suchraum")
    s1, s2, s3 = st.columns(3)
    with s1:
        e_rng = st.slider("Einstieg zwischen (Stunde)", 0, 23, (lo_h, hi_h), key="dfs_erng")
        x_rng = st.slider("Ausstieg zwischen (Stunde)", 0, 23, (lo_h, hi_h), key="dfs_xrng",
                          help="Obere Grenze inklusive der ganzen Stunde (z. B. 21 → bis 21:45 auf M15).")
    with s2:
        sl_sel = st.select_slider("SL-Bereich %", options=SL_STEPS, value=(0.20, 2.00), key="dfs_slrng")
        with_no_sl = st.checkbox("Zusätzlich ohne Stop Loss testen", True, key="dfs_nosl")
    with s3:
        metric = st.selectbox("Ranking nach", list(METRICS), format_func=METRICS.get, key="dfs_metric")
        min_trades = st.number_input("Min. Trades", 10, 5000, max(30, int(0.5 * n_days_cov)),
                                     key="dfs_mint", help="Szenarien mit weniger Trades werden ignoriert.")
        robust = st.checkbox("Robustheits-Wertung (Nachbarn mitteln)", True, key="dfs_robust")

    slots_per_h = 60 // tf_min
    entry_slots = tuple(range(e_rng[0] * slots_per_h, (e_rng[1] + 1) * slots_per_h))
    exit_slots = tuple(range(x_rng[0] * slots_per_h, min(K, (x_rng[1] + 1) * slots_per_h)))
    sl_list = tuple(p for p in SL_STEPS if sl_sel[0] <= p <= sl_sel[1]) + ((None,) if with_no_sl else ())
    n_pairs = sum(1 for e in entry_slots for x in exit_slots if x > e)
    st.caption(f"**{n_pairs * len(sl_list):,}** Szenarien ({n_pairs:,} Zeit-Kombinationen × {len(sl_list)} SL-Stufen)"
               .replace(",", "."))

    if st.button("🔎 Alle Szenarien testen", type="primary", key="dfs_scan_btn"):
        with st.spinner("Rechne alle Szenarien durch …"):
            res = _full_scan(tf_min, mtime, d_from, d_to, entry_slots, exit_slots, sl_list,
                             spread_mode, float(spread_pts), float(comm_pct))
        st.session_state["dfs_scan"] = {
            "res": res, "entry_slots": entry_slots, "exit_slots": exit_slots, "sl_list": sl_list,
            "tf_min": tf_min, "d_from": d_from, "d_to": d_to, "spread_mode": spread_mode,
            "spread_pts": float(spread_pts), "comm_pct": float(comm_pct)}

    scan = st.session_state.get("dfs_scan")
    if not scan:
        st.info("Suchraum einstellen und **Alle Szenarien testen** klicken.")
        return
    if scan["tf_min"] != tf_min:
        st.warning(f"Angezeigtes Ergebnis stammt aus einem {('M15' if scan['tf_min'] == 15 else 'H1')}-Scan — "
                   "bitte neu scannen.")
        return

    res, es, xs, ps = scan["res"], scan["entry_slots"], scan["exit_slots"], scan["sl_list"]
    raw = res[metric].copy()
    raw[(res["n"] < min_trades) | np.isnan(res["n"])] = np.nan
    score = _smooth(raw, ps) if robust else raw

    flat = np.argsort(np.nan_to_num(score, nan=-np.inf), axis=None)[::-1]
    picks, rows = [], []
    for fi in flat:
        ie, ix, ip = np.unravel_index(fi, score.shape)
        if np.isnan(score[ie, ix, ip]):
            break
        cand = {"e": es[ie], "x": xs[ix], "p": ps[ip], "idx": (ie, ix, ip)}
        if len(rows) < 25:
            rows.append(cand)
        if len(picks) < 3 and all(abs(cand["e"] - q["e"]) > 1 or abs(cand["x"] - q["x"]) > 1 for q in picks):
            picks.append(cand)
        if len(picks) >= 3 and len(rows) >= 25:
            break

    if not picks:
        st.warning("Kein Szenario erfüllt die Mindest-Trades — Filter lockern.")
        return

    st.session_state["dfs_top3"] = [{"e": q["e"], "x": q["x"], "p": q["p"], "tf_min": tf_min} for q in picks]

    # Einzelauswertung der Top 3
    Asub = _subset(A, (A["days"] >= np.datetime64(scan["d_from"])) & (A["days"] <= np.datetime64(scan["d_to"])))
    S = _spread_matrix(Asub, scan["spread_mode"], scan["spread_pts"])

    st.markdown("---")
    st.subheader("🏆 Top 3 Szenarien")
    st.caption(f"Ranking: {METRICS[metric]}{' (Nachbarschafts-Mittel)' if robust else ''} · "
               f"Top 3 unterscheiden sich in Einstieg oder Ausstieg um mehr als eine Kerze.")
    colors = ["#f7931a", "#38bdf8", "#a78bfa"]
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(len(picks))
    curves = []
    for i, (q, col) in enumerate(zip(picks, cols)):
        ie, ix, ip = q["idx"]
        tr = _scenario_trades(Asub, S, q["e"], q["x"], q["p"], scan["comm_pct"])
        n_sl = int((tr["Exit"] == "SL").sum())
        n_time = int((tr["Exit"] == "Zeit").sum())
        n_safe = int((tr["Exit"] == "Sicherheit").sum())
        with col:
            st.markdown(f"### {medals[i]} {_slot_label(q['e'], tf_min)} → {_slot_label(q['x'], tf_min)}")
            st.markdown(f"**SL {_sl_label(q['p'])}**")
            m1, m2 = st.columns(2)
            m1.metric("Trades", int(res["n"][ie, ix, ip]))
            m2.metric("Win-Rate", f"{res['wr'][ie, ix, ip]:.1f} %")
            m1.metric("Profit Factor", f"{res['pf'][ie, ix, ip]:.2f}")
            m2.metric("Summe Rendite", f"{res['total'][ie, ix, ip]:.1f} %")
            m1.metric("Ø Trade", f"{res['avg'][ie, ix, ip]:.3f} %")
            m2.metric("Max DD", f"{res['dd'][ie, ix, ip]:.1f} %")
            st.caption(f"Sharpe {res['sharpe'][ie, ix, ip]:.2f} · Rendite/DD {res['ret_dd'][ie, ix, ip]:.2f}"
                       + (f" · Robust-Score {score[ie, ix, ip]:.2f}" if robust else "")
                       + f"<br>Exits: {n_sl}× SL · {n_time}× Zeit · {n_safe}× Sicherheit", unsafe_allow_html=True)
        curves.append((f"{medals[i]} {_fmt_scn(q, tf_min)}", tr.set_index("Datum")["Rendite %"].cumsum(), colors[i]))

    st.plotly_chart(_equity_fig(curves, "Equity der Top 3 (Summe Rendite % je Trade, 1× Notional)"),
                    use_container_width=True)
    st.info("➡️ Im Reiter **🔄 Walk-Forward-Analyse** kannst du eins der drei Szenarien auswählen und testen.")

    # Jahresrenditen
    yr_rows = {}
    for i, q in enumerate(picks):
        tr = _scenario_trades(Asub, S, q["e"], q["x"], q["p"], scan["comm_pct"])
        yr_rows[f"{medals[i]} {_slot_label(q['e'], tf_min)}→{_slot_label(q['x'], tf_min)}"] = \
            tr.groupby(tr["Datum"].dt.year)["Rendite %"].sum().round(2)
    if yr_rows:
        st.markdown("#### Rendite je Jahr (%)")
        yr_df = pd.DataFrame(yr_rows)
        st.dataframe(yr_df.style.map(lambda v: "color:#22c55e" if v > 0 else "color:#ef5350")
                     .format("{:.2f}"), use_container_width=True)

    # Heatmap Einstieg × Ausstieg (bester SL je Zelle)
    st.markdown("#### Heatmap: Einstieg × Ausstieg (beste SL-Stufe je Zelle)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hm = np.nanmax(score, axis=2)
    fig = go.Figure(go.Heatmap(
        z=hm.T, x=[_slot_label(e, tf_min) for e in es], y=[_slot_label(x, tf_min) for x in xs],
        colorscale="RdYlGn", colorbar=dict(title=METRICS[metric][:10]),
        hovertemplate="Einstieg %{x} · Ausstieg %{y}<br>Score %{z:.2f}<extra></extra>"))
    fig.update_layout(height=520, template="plotly_dark", margin=dict(t=20, b=20),
                      xaxis_title="Einstieg (Berlin)", yaxis_title="Ausstieg (Berlin)")
    st.plotly_chart(fig, use_container_width=True)

    # Top-25-Tabelle
    st.markdown("#### Top 25 Szenarien")
    tbl = []
    for q in rows:
        ie, ix, ip = q["idx"]
        tbl.append({"Einstieg": _slot_label(q["e"], tf_min), "Ausstieg": _slot_label(q["x"], tf_min),
                    "SL": _sl_label(q["p"]), "Trades": int(res["n"][ie, ix, ip]),
                    "Win-Rate %": round(res["wr"][ie, ix, ip], 1), "PF": round(res["pf"][ie, ix, ip], 2),
                    "Summe %": round(res["total"][ie, ix, ip], 2), "Ø Trade %": round(res["avg"][ie, ix, ip], 3),
                    "Max DD %": round(res["dd"][ie, ix, ip], 2), "Sharpe": round(res["sharpe"][ie, ix, ip], 2),
                    "Score": round(float(score[ie, ix, ip]), 3)})
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# REITER 2: WFA
# -----------------------------------------------------------------------------

def _build_folds(days: np.ndarray, is_m: int, oos_m: int, offset_days: int = 0) -> list[dict]:
    t = pd.to_datetime(days)
    fs = t[0] + pd.Timedelta(days=offset_days)
    folds = []
    while True:
        ie = fs + pd.DateOffset(months=is_m)
        oe = ie + pd.DateOffset(months=oos_m)
        if oe > t[-1] + pd.Timedelta(days=1):
            break
        folds.append({"is_start": fs, "is_end": ie, "oos_start": ie, "oos_end": oe})
        fs = fs + pd.DateOffset(months=oos_m)
    return folds


def _neighborhood(base: dict, K: int, n_e: int, n_x: int, n_p: int) -> list[tuple]:
    es = [e for e in range(base["e"] - n_e, base["e"] + n_e + 1) if 0 <= e < K]
    xs = [x for x in range(base["x"] - n_x, base["x"] + n_x + 1) if 0 <= x < K]
    if base["p"] is None:
        ps = [None]
    else:
        i0 = SL_STEPS.index(base["p"])
        ps = SL_STEPS[max(0, i0 - n_p): i0 + n_p + 1]
    combos = [(e, x, p) for e in es for x in xs for p in ps if x > e]
    base_key = (base["e"], base["x"], base["p"])
    return combos if base_key in combos else [base_key] + combos


def _run_wfa(A, S, combos, base_key, folds, metric, min_is, comm_pct):
    """Rendite-Matrix aller Kombis einmal berechnen, dann je Fold IS optimieren / OOS prüfen."""
    days = pd.to_datetime(A["days"])
    # Kombis nach (e, p) gruppieren → ein Engine-Aufruf je Gruppe
    R = np.full((len(combos), len(days)), np.nan)
    groups: dict = {}
    for ci, (e, x, p) in enumerate(combos):
        groups.setdefault((e, p), []).append((ci, x))
    for (e, p), items in groups.items():
        xs = np.array([x for _, x in items])
        Rg, _ = _returns_for(A, S, e, p, xs, _exit_prices(A, S, xs), comm_pct)
        for (ci, _), row in zip(items, Rg):
            R[ci] = row
    base_ci = combos.index(base_key)

    rows, oos_chain, oos_fixed, stab = [], [], [], {c: [] for c in combos}
    for fi, f in enumerate(folds):
        is_m = (days >= f["is_start"]) & (days < f["is_end"])
        oos_m = (days >= f["oos_start"]) & (days < f["oos_end"])
        if is_m.sum() < 10 or oos_m.sum() < 3:
            continue
        m_is = _metrics(R[:, is_m])
        m_oos = _metrics(R[:, oos_m])
        sc = np.where(m_is["n"] >= min_is, m_is[metric], -np.inf)
        for ci, c in enumerate(combos):
            stab[c].append({k: float(m_oos[k][ci]) for k in ("n", "pf", "total", "sharpe", "dd", "wr")})
        label = f"{f['is_start'].date()} – {f['is_end'].date()}"
        oos_label = f"{f['oos_start'].date()} – {f['oos_end'].date()}"
        fx = {k: float(m_oos[k][base_ci]) for k in m_oos}
        fix_ok = fx["n"] >= 1 and fx["pf"] > 1.0 and fx["total"] > 0
        oos_fixed.append(pd.Series(R[base_ci, oos_m], index=days[oos_m]))
        if not np.isfinite(sc.max()):
            rows.append({"Fold": fi + 1, "IS": label, "OOS": oos_label, "Status": "⚠️ Kein IS-Ergebnis",
                         "Fix OOS PF": round(fx["pf"], 2), "Fix OOS Ret %": round(fx["total"], 2),
                         "Fix Status": "✅" if fix_ok else "❌"})
            continue
        b = int(np.argmax(sc))
        e, x, p = combos[b]
        ok = m_oos["n"][b] >= 1 and m_oos["pf"][b] > 1.0 and m_oos["total"][b] > 0
        oos_chain.append(pd.Series(R[b, oos_m], index=days[oos_m]))
        rows.append({
            "Fold": fi + 1, "IS": label, "OOS": oos_label,
            "Bester Einstieg": e, "Bester Ausstieg": x, "Bester SL": _sl_label(p),
            "IS Trades": int(m_is["n"][b]), "IS PF": round(float(m_is["pf"][b]), 2),
            "OOS Trades": int(m_oos["n"][b]), "OOS WR %": round(float(m_oos["wr"][b]), 1),
            "OOS PF": round(float(m_oos["pf"][b]), 2), "OOS Ret %": round(float(m_oos["total"][b]), 2),
            "OOS Max DD %": round(float(m_oos["dd"][b]), 2),
            "Status": "✅ Bestanden" if ok else "❌ Fail",
            "Fix OOS PF": round(fx["pf"], 2), "Fix OOS Ret %": round(fx["total"], 2),
            "Fix Status": "✅" if fix_ok else "❌",
        })
    chain = pd.concat(oos_chain).dropna() if oos_chain else pd.Series(dtype=float)
    fixed = pd.concat(oos_fixed).dropna() if oos_fixed else pd.Series(dtype=float)
    return {"rows": rows, "chain": chain, "fixed": fixed, "stab": stab, "base_full": R[base_ci]}


def _render_wfa(A, tf_min, d_from, d_to, spread_mode, spread_pts, comm_pct, save_fn, load_fn, evaluate_edge_fn):
    K = A["K"]
    top3 = [s for s in st.session_state.get("dfs_top3", []) if s["tf_min"] == tf_min]

    st.subheader("1 · Strategie wählen")
    options = [f"{['🥇', '🥈', '🥉'][i]} {_fmt_scn(s, tf_min)}" for i, s in enumerate(top3)] + ["✏️ Manuell eingeben"]
    choice = st.radio("Szenario", options, key=f"dfs_wfa_pick_{tf_min}",
                      help="Die Top 3 kommen aus dem Scanner-Reiter (gleicher Timeframe).")
    if not top3:
        st.caption("Noch kein Scanner-Ergebnis für diesen Timeframe — Top 3 erscheinen nach dem Scan hier.")
    if choice.startswith("✏️"):
        mc1, mc2, mc3 = st.columns(3)
        slot_opts = list(range(K))
        e = mc1.selectbox("Einstieg", slot_opts, index=min(8 * 60 // tf_min, K - 1),
                          format_func=lambda s: _slot_label(s, tf_min), key=f"dfs_man_e_{tf_min}")
        x = mc2.selectbox("Ausstieg", slot_opts, index=min(21 * 60 // tf_min, K - 1),
                          format_func=lambda s: _slot_label(s, tf_min), key=f"dfs_man_x_{tf_min}")
        p = mc3.selectbox("Stop Loss", [None] + SL_STEPS, index=SL_STEPS.index(0.5) + 1,
                          format_func=_sl_label, key=f"dfs_man_p_{tf_min}")
        if x <= e:
            st.error("Ausstieg muss nach dem Einstieg liegen.")
            return
        base = {"e": e, "x": x, "p": p}
    else:
        base = top3[options.index(choice)]

    st.subheader("2 · Walk-Forward Konfiguration")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        is_months = st.number_input("IS-Fenster (Monate)", 6, 60, 24, key="dfs_is")
        oos_months = st.number_input("OOS-Fenster (Monate)", 3, 24, 12, key="dfs_oos")
    with w2:
        opt_metric = st.selectbox("Optimierungsziel (IS)", ["pf", "sharpe", "total", "ret_dd"],
                                  format_func=METRICS.get, key="dfs_wfa_metric")
        min_is = st.number_input("Min. IS-Trades", 5, 300, 30, key="dfs_wfa_minis")
    with w3:
        n_e = st.number_input("Einstieg ± Kerzen", 0, 8, 2, key="dfs_ne",
                              help="IS-Optimierung sucht in dieser Nachbarschaft um das gewählte Szenario.")
        n_x = st.number_input("Ausstieg ± Kerzen", 0, 8, 2, key="dfs_nx")
        n_p = st.number_input("SL ± Stufen (0,05 %)", 0, 10, 3, key="dfs_np", disabled=base["p"] is None)
    with w4:
        min_pass = st.number_input("Min. bestandene Folds % für ✅ ROBUST", 30, 100, 60, 5, key="dfs_minpass",
                                   help="Anteil statt fester Anzahl — sonst wäre z. B. 4/10 schon 'robust'.")
        do_ens = st.checkbox("Ensemble (5× versetzte Starts)", True, key="dfs_ens")

    combos = _neighborhood(base, K, int(n_e), int(n_x), int(n_p))
    base_key = (base["e"], base["x"], base["p"])
    Asub = _subset(A, (A["days"] >= np.datetime64(d_from)) & (A["days"] <= np.datetime64(d_to)))
    folds = _build_folds(Asub["days"], int(is_months), int(oos_months))
    st.caption(f"**{len(folds)} Folds** · IS {is_months} M / OOS {oos_months} M · "
               f"{len(combos)} Parameter-Kombinationen je Fold · Pass-Kriterium je Fold: OOS PF > 1 und OOS-Rendite > 0")
    if len(folds) < 2:
        st.warning("Zu wenig Daten für die WFA — Zeitraum verlängern oder Fenster verkleinern.")
        return

    slot = f"dax_fri_wfa_{tf_min}"
    run_key = (tf_min, base_key, str(d_from), str(d_to), int(is_months), int(oos_months), opt_metric, int(min_is),
               int(n_e), int(n_x), int(n_p), spread_mode, float(spread_pts), float(comm_pct), bool(do_ens))

    if st.button("🔄 WFA starten", type="primary", key="dfs_wfa_btn"):
        S = _spread_matrix(Asub, spread_mode, spread_pts)
        with st.spinner("Walk-Forward läuft …"):
            wfa = _run_wfa(Asub, S, combos, base_key, folds, opt_metric, int(min_is), float(comm_pct))
            ens = []
            if do_ens:
                step = max(1, int(oos_months * 30 / 5))
                for k in range(5):
                    f_k = _build_folds(Asub["days"], int(is_months), int(oos_months), offset_days=k * step)
                    if len(f_k) < 2:
                        continue
                    r_k = _run_wfa(Asub, S, combos, base_key, f_k, opt_metric, int(min_is), float(comm_pct))
                    ok_k = sum(1 for r in r_k["rows"] if r["Status"] == "✅ Bestanden")
                    fix_k = sum(1 for r in r_k["rows"] if r.get("Fix Status") == "✅")
                    ens.append({"Lauf": k + 1, "Start-Versatz (Tage)": k * step, "Folds": len(r_k["rows"]),
                                "WFA bestanden": ok_k, "Fix bestanden": fix_k,
                                "WFA OOS Summe %": round(float(r_k["chain"].sum()), 2),
                                "Fix OOS Summe %": round(float(r_k["fixed"].sum()), 2)})
        result = {"key": run_key, "base": base, "wfa": wfa, "ens": ens}
        st.session_state[slot] = result
        if save_fn is not None:
            blob = {"key": list(map(str, run_key)), "base": {"e": base["e"], "x": base["x"], "p": base["p"]},
                    "rows": wfa["rows"], "ens": ens,
                    "chain": wfa["chain"], "fixed": wfa["fixed"]}
            ok, why = save_fn(slot, blob)
            if ok:
                st.caption("☁️ WFA-Ergebnis im Cloud-Speicher gesichert.")
            else:
                st.caption(f"⚠️ Cloud-Speicher: {why}")

    result = st.session_state.get(slot)
    if result is None and load_fn is not None and not st.session_state.get(f"{slot}_restored"):
        st.session_state[f"{slot}_restored"] = True
        blob = load_fn(slot)
        if blob:
            result = {"key": None, "base": blob["base"], "ens": blob.get("ens", []),
                      "wfa": {"rows": blob["rows"], "chain": blob["chain"], "fixed": blob["fixed"], "stab": None}}
            st.session_state[slot] = result
            st.caption("☁️ Letztes WFA-Ergebnis aus dem Cloud-Speicher wiederhergestellt.")

    if result is None:
        st.info("Szenario wählen und **🔄 WFA starten** klicken.")
        return
    if result["key"] is not None and result["key"] != run_key:
        st.caption("ℹ️ Einstellungen wurden geändert — angezeigt wird das letzte Ergebnis "
                   f"(**{_fmt_scn(result['base'], tf_min)}**). Neu starten, um zu aktualisieren.")

    _show_wfa(result, tf_min, int(min_pass), evaluate_edge_fn)


def _show_wfa(result, tf_min, min_pass, evaluate_edge_fn):
    wfa, base = result["wfa"], result["base"]
    rows = wfa["rows"]
    if not rows:
        st.error("Keine WFA-Ergebnisse — Parameter oder Zeitraum anpassen.")
        return
    df = pd.DataFrame(rows)
    n_tot = len(df)
    n_ok = int((df["Status"] == "✅ Bestanden").sum())
    n_fix = int((df["Fix Status"] == "✅").sum())

    st.markdown("---")
    st.subheader(f"Ergebnis · {_fmt_scn(base, tf_min)}")
    if n_ok / n_tot * 100 >= min_pass and n_ok >= 2:
        bc, bt = "#22c55e", f"✅ ROBUST — {n_ok}/{n_tot} Folds bestanden · Strategie empfohlen"
    elif n_ok / n_tot >= 0.4:
        bc, bt = "#f0c040", f"⚠️ INSTABIL — nur {n_ok}/{n_tot} Folds bestanden · mit Vorsicht handeln"
    elif n_ok == 1:
        bc, bt = "#ef5350", f"❌ NICHT EMPFOHLEN — nur 1/{n_tot} Fold bestanden"
    else:
        bc, bt = "#ef5350", f"❌ GESCHEITERT — 0/{n_tot} Folds bestanden"
    _banner(bc, bt, f"WFA (IS-Optimierung in der Nachbarschaft): OOS-Summe {wfa['chain'].sum():.2f} % · "
                    f"Fixes Szenario ohne Nachoptimierung: {n_fix}/{n_tot} Folds positiv, "
                    f"OOS-Summe {wfa['fixed'].sum():.2f} %")

    if evaluate_edge_fn is not None and len(wfa["chain"]) >= 10:
        try:
            sig = evaluate_edge_fn(pd.DataFrame({"PnL $": wfa["chain"].values}), min_trades=30, alpha=0.05,
                                   min_sharpe_oos=1.0, pnl_col="PnL $")
            style = {"handelbar": ("#22c55e", "🟢 Statistisch signifikant"),
                     "grenzwertig": ("#f0c040", "🟡 Grenzwertig"),
                     "nicht handelbar": ("#ef5350", "🔴 Nicht signifikant")}
            col, lab = style.get(sig["status"], ("#9fb0c7", sig["status"]))
            p_txt = "n/a" if np.isnan(sig["p_value"]) else f"{sig['p_value']:.4f}"
            lo, hi = sig["wilson_ci"]
            _banner(col, lab, f"OOS-Trades gepoolt: n={sig['n_trades']} · Wilcoxon p={p_txt} · "
                              f"Wilson-CI Winrate=[{lo * 100:.1f}%, {hi * 100:.1f}%]")
        except Exception:
            pass

    # Fold-Tabelle
    st.subheader("Fold-Ergebnisse")
    show = df.copy()
    for c in ("Bester Einstieg", "Bester Ausstieg"):
        if c in show:
            show[c] = show[c].map(lambda s: _slot_label(int(s), tf_min) if pd.notna(s) else "–")

    def _c_status(v):
        return "color:#22c55e;font-weight:700" if "✅" in str(v) else "color:#ef5350" if "❌" in str(v) else ""

    def _c_num(v):
        return ("color:#22c55e" if v > 0 else "color:#ef5350") if isinstance(v, (int, float)) and not pd.isna(v) else ""

    num_cols = [c for c in ("OOS Ret %", "Fix OOS Ret %") if c in show]
    st.dataframe(show.style.map(_c_status, subset=[c for c in ("Status", "Fix Status") if c in show])
                 .map(_c_num, subset=num_cols), use_container_width=True, hide_index=True)
    st.caption("**Status** = in-sample optimierte Parameter auf dem folgenden OOS-Fenster. "
               "**Fix** = dein gewähltes Szenario unverändert auf demselben OOS-Fenster.")

    # OOS-Rendite je Fold
    st.subheader("OOS-Rendite je Fold")
    fig = go.Figure()
    if "OOS Ret %" in df:
        fig.add_trace(go.Bar(x=df["Fold"], y=df["OOS Ret %"], name="WFA (optimiert)", marker_color="#f7931a"))
    fig.add_trace(go.Bar(x=df["Fold"], y=df["Fix OOS Ret %"], name="Fixes Szenario", marker_color="#38bdf8"))
    fig.update_layout(barmode="group", height=300, template="plotly_dark", margin=dict(t=20, b=15),
                      xaxis_title="Fold", yaxis_title="OOS-Rendite %")
    st.plotly_chart(fig, use_container_width=True)

    # Kombinierte OOS-Equity
    st.subheader("Kombinierte OOS Equity Kurve")
    curves = []
    if len(wfa["chain"]):
        curves.append(("WFA-Kette (optimiert je Fold)", wfa["chain"].cumsum(), "#f7931a"))
    if len(wfa["fixed"]):
        curves.append(("Fixes Szenario (nur OOS-Zeiträume)", wfa["fixed"].cumsum(), "#38bdf8"))
    st.plotly_chart(_equity_fig(curves, "Nur Out-of-Sample-Trades, aneinandergereiht"), use_container_width=True)

    # Parameter-Stabilität
    stab = wfa.get("stab")
    if stab:
        st.subheader("Parameter-Stabilitätsanalyse")
        st.caption("Welche Kombinationen aus der Nachbarschaft liefern konsistent über ALLE Folds positive OOS-Ergebnisse?")
        srows = []
        for (e, x, p), res in stab.items():
            res = [r for r in res if r["n"] > 0]
            if len(res) < 2:
                continue
            rets = [r["total"] for r in res]
            n_pos = sum(1 for r in rets if r > 0)
            srows.append({"Einstieg": _slot_label(e, tf_min), "Ausstieg": _slot_label(x, tf_min), "SL": _sl_label(p),
                          "Gewählt": "⭐" if (e, x, p) == (base["e"], base["x"], base["p"]) else "",
                          "Folds": len(res), "Profitable Folds": n_pos,
                          "Konsistenz %": round(n_pos / len(res) * 100, 0),
                          "Ø OOS PF": round(float(np.mean([r["pf"] for r in res])), 2),
                          "Ø OOS Ret %": round(float(np.mean(rets)), 2),
                          "Min OOS Ret %": round(float(np.min(rets)), 2),
                          "Ø Max DD %": round(float(np.mean([r["dd"] for r in res])), 2)})
        if srows:
            sdf = pd.DataFrame(srows).sort_values(["Konsistenz %", "Ø OOS PF", "Ø OOS Ret %"],
                                                  ascending=False).reset_index(drop=True)
            base_row = sdf[sdf["Gewählt"] == "⭐"]
            if len(base_row):
                rank = int(base_row.index[0]) + 1
                st.caption(f"Dein Szenario steht auf Platz **{rank} von {len(sdf)}** der Nachbarschaft.")

            def _c_k(v):
                if isinstance(v, (int, float)):
                    return "color:#22c55e;font-weight:700" if v >= 80 else "color:#f0c040" if v >= 60 else "color:#ef5350"
                return ""
            st.markdown("#### 🏆 Top-10 stabilste Setups")
            st.dataframe(sdf.head(10).style.map(_c_k, subset=["Konsistenz %"]),
                         use_container_width=True, hide_index=True)

    # Ensemble
    if result.get("ens"):
        st.subheader("Ensemble WFA — 5× versetzte Starts")
        edf = pd.DataFrame(result["ens"])
        st.dataframe(edf, use_container_width=True, hide_index=True)
        pass_rate = edf["WFA bestanden"].sum() / max(1, edf["Folds"].sum()) * 100
        fix_rate = edf["Fix bestanden"].sum() / max(1, edf["Folds"].sum()) * 100
        st.caption(f"Über alle Läufe: WFA {pass_rate:.0f} % der Folds bestanden · fixes Szenario {fix_rate:.0f} % positiv. "
                   "Stabil ist ein Ergebnis, wenn es nicht vom zufälligen Startdatum der Folds abhängt.")
