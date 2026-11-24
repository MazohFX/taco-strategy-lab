"""TACO Lab Seite „Zig Zag Strategie Test“.

Python-Port des Pine-Scripts „Zig Zag“ (ZigZag Sweep Strategy, 3 Positionen) — Engine: zigzag_engine.py.
Bewertet wird mit DENSELBEN Verfahren wie auf der Seite „DAX EMA Strategie“:
  * IS-Top-N Split-Test (In-Sample → Out-of-Sample → Holdout) und rollierender IS-Top-N Walk-Forward
    (dax_topn_wfa.py: build_folds / run_all_combos / split_topn / topn_walk_forward / aggregate_setups)
  * Overfitting-Check: PBO per CSCV + Deflated Sharpe, gleiche Ampel-Schwellen
  * Monte Carlo Prop-Challenge-Simulator (Phase 1 / Phase 2 / Funded-Payouts) mit identischer Logik
"""
from __future__ import annotations

import datetime as _dt
import math
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dax_topn_wfa as _topn
import zigzag_engine as ze

ENGINE_VERSION = 1
_GIST_SLOT = "zigzag_topn"
_DEV_DEFAULTS = {"D1": [1.5, 2.5, 3.5], "H4": [1.0, 1.5, 2.5], "H1": [0.5, 1.0, 1.5], "M15": [0.3, 0.5, 1.0]}
_DEV_OPTIONS = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0]
ZZ_SYMBOLS = ["USDJPY", "XAUUSD", "XAGUSD"]   # Paare, für die die Pine-Strategie gedacht ist
_OPT_LBL = {"pf": "Profit Factor", "sharpe": "Sharpe", "wr": "Win-Rate"}


# ════════════════════════════════════════════════════════════════════════════
# Daten & Engine über mehrere Paare
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load(symbol: str, tf: str) -> pd.DataFrame:
    return ze.load_bars(symbol, tf)


def _multi_engine(frames: dict, base: dict):
    """Engine im Format von dax_topn_wfa.run_all_combos: (df, params) → (trades, equity).
    `df` wird ignoriert — gerechnet wird auf allen gewählten Paaren (je Paar eigene 3 Slots), Trades gepoolt."""
    def _eng(_df_unused, params):
        parts = []
        for sym, df in frames.items():
            p = {**params, "symbol": sym,
                 "commission_pct": base["comm_by_sym"].get(sym, params.get("commission_pct", 0.004))}
            tr, _ = ze.backtest(df, p)
            if not tr.empty:
                tr = tr.assign(Paar=sym)
                parts.append(tr)
        if not parts:
            return pd.DataFrame(columns=["Zeit", "Ret %"]), None
        tr = pd.concat(parts, ignore_index=True).sort_values(["Zeit", "Entry-Zeit"], kind="stable").reset_index(drop=True)
        return tr, None
    return _eng


def _build_combos(g_dir, g_dev, g_depth, g_ret, g_sl, g_tp, g_tb, g_mc) -> list[dict]:
    out = []
    for dm, dev, dp, rv, sl, tp, tb, mc in product(g_dir, g_dev, g_depth, g_ret, g_sl, g_tp, g_tb, g_mc):
        tp_mode, tp_atr = ("time", 0.0) if tp == 0 else ("atr", float(tp))
        key = (dm, float(dev), int(dp), float(rv), float(sl), tp_mode, float(tp_atr), int(tb), int(mc))
        out.append({"key": key, "params": {"dir_mode": dm, "deviation": float(dev), "depth": int(dp),
                                           "retrace": float(rv), "sl_val": float(sl), "tp_mode": tp_mode,
                                           "tp_atr": float(tp_atr) if tp_atr else 1.0, "time_bars": int(tb),
                                           "max_conc": int(mc)}})
    return out


def _fmt_sl(sl, sl_mode) -> str:
    return f"{sl:g}× ATR" if sl_mode == "atr" else f"{sl:g}%"


def _param_cols(k, sl_mode) -> dict:
    dm, dev, dp, rv, sl, tpm, tpa, tb, mc = k
    return {"Richtung": "Continuation" if dm == "continuation" else "Reversal", "Deviation %": f"{dev:g}",
            "Pivot-Legs": int(dp), "Validität": f"{rv:g}", "Stop": _fmt_sl(sl, sl_mode),
            "TP": "nur Zeit" if tpm == "time" else f"{tpa:g}× ATR", "Time-Exit": f"{tb} Bars", "Max Pos.": mc}


def _label(k, sl_mode) -> str:
    dm, dev, dp, rv, sl, tpm, tpa, tb, mc = k
    return " · ".join([("Cont" if dm == "continuation" else "Rev"), f"Dev {dev:g}%", f"Legs {dp}",
                       f"SL {_fmt_sl(sl, sl_mode)}", ("TP Zeit" if tpm == "time" else f"TP {tpa:g}×ATR"),
                       f"{tb} Bars", f"{mc} Pos."] + ([f"Val {rv:g}"] if rv != 0.5 else []))


def _is_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ════════════════════════════════════════════════════════════════════════════
# Monte Carlo — identische Logik wie DAX EMA Strategie
# ════════════════════════════════════════════════════════════════════════════
def _run_phase(cap, target_pct, norm_r, risk_pct, dd_total, dd_daily, max_t):
    capital = float(cap); peak = capital; day_start = capital
    path = [capital]; n_trades = 0; failed = False; reason = ""
    while True:
        nr = norm_r[np.random.randint(len(norm_r))]
        capital += capital * (risk_pct / 100) * nr
        peak = max(peak, capital)
        path.append(capital)
        n_trades += 1
        if (capital - day_start) / day_start * 100 < -dd_daily:
            failed, reason = True, "Daily DD"; break
        day_start = capital
        if (capital - peak) / peak * 100 < -dd_total:
            failed, reason = True, "Total DD"; break
        if (capital - cap) / cap * 100 >= target_pct:
            break
        if n_trades >= max_t:
            break
    final_pct = (capital - cap) / cap * 100
    return (not failed and final_pct >= target_pct), reason, n_trades, final_pct, path


def _run_funded_phase(cap, t1, t2, norm_r, risk_pct, dd_total, dd_daily, max_t):
    capital = float(cap); peak = capital; day_start = capital; n_trades = 0
    p1_ok, p1_t, p2_ok, p2_t = False, 0, False, 0
    stop_target = max(t1, t2)
    while True:
        nr = norm_r[np.random.randint(len(norm_r))]
        capital += capital * (risk_pct / 100) * nr
        peak = max(peak, capital)
        n_trades += 1
        if (capital - day_start) / day_start * 100 < -dd_daily:
            break
        day_start = capital
        if (capital - peak) / peak * 100 < -dd_total:
            break
        profit = (capital - cap) / cap * 100
        if not p1_ok and profit >= t1:
            p1_ok, p1_t = True, n_trades
        if not p2_ok and profit >= t2:
            p2_ok, p2_t = True, n_trades
        if profit >= stop_target or n_trades >= max_t:
            break
    return p1_ok, p1_t, p2_ok, p2_t


def _render_mc(res: dict | None, sl_mode: str, risk_default: float) -> None:
    st.subheader("Monte Carlo — Prop Trading Challenge Simulator")
    st.markdown("Simuliert **1.000 mögliche Zukunften** der Strategie durch zufälliges Mischen (Bootstrap) der echten "
                "Trade-Ergebnisse — gleiche Rechenlogik wie auf der Seite *DAX EMA Strategie*.")
    sp = (res or {}).get("split") or {}
    if not sp.get("top"):
        st.info("Noch kein Ergebnis — im Tab 📊 Überblick **▶ Test starten** klicken. Danach kannst du hier jedes "
                "Setup aus der Top-N-Tabelle simulieren.")
        return
    top = sp["top"]
    s0, s1, s2, s3 = (pd.Timestamp(sp[k]) for k in ("start", "oos_start", "hold", "end"))
    src_opts = ["Setup aus der Top-N-Tabelle (📊 Überblick)", "IS-Top-N: immer den IS-Sieger handeln — OOS-Trades",
                "IS-Top-N: häufigstes Top-Setup — seine OOS-Trades"]
    src = st.radio("Was simulieren?", src_opts, horizontal=True, key="zz_mc_mode")
    risk0 = float(res.get("risk_pct", 1.0))
    years = None
    if src == src_opts[0]:
        sel = st.selectbox("Setup", range(len(top)), key="zz_mc_sel",
                           format_func=lambda i: f"Rang {top[i]['rank']} · " + " · ".join(
                               f"{k}: {v}" for k, v in _param_cols(top[i]["key"], sl_mode).items()))
        per_opts = ["Nur Out-of-Sample (empfohlen)", "In-Sample + Out-of-Sample"]
        if st.session_state.get("zz_show_hold"):
            per_opts.append("Out-of-Sample + Holdout")
        per = st.radio("Welche Trades?", per_opts, horizontal=True, key="zz_mc_per",
                       help="Nur Out-of-Sample = ehrlichste Basis (blinde Trades). In-Sample-Trades sind die, nach denen "
                            "das Setup ausgewählt wurde → leicht zu optimistisch. Holdout erscheint nur, wenn er im "
                            "Überblick aufgedeckt ist.")
        a, b = {"Nur Out-of-Sample (empfohlen)": (s1, s2), "In-Sample + Out-of-Sample": (s0, s2),
                "Out-of-Sample + Holdout": (s1, s3 + pd.Timedelta(days=1))}[per]
        t = top[sel]
        r = np.array([x[1] for x in t["trades"] if a <= pd.Timestamp(x[0]) < b], dtype=float)
        years = max((min(b, s3) - a).days / 365.25, 0.01)
        m = _topn.metrics_from_rets(r, years)
        st.dataframe(pd.DataFrame([{"Rang": t["rank"], **_param_cols(t["key"], sl_mode),
                                    "Zeitraum": f"{a.date()} – {min(b, s3).date()}", "Trades": m["n"], "PF": m["pf"],
                                    "Win-Rate %": m["wr"], "Rendite %": m["total_ret"], "Max DD %": m["max_dd"]}]),
                     use_container_width=True, hide_index=True)
        label = f"Rang {t['rank']} · {_label(t['key'], sl_mode)} · {per}"
    elif src == src_opts[1]:
        ok = [f for f in res["folds"] if not f.get("empty")]
        r = np.concatenate([np.asarray(f["winner_oos_rets"], float) for f in ok]) if ok else np.array([])
        years = max(len(ok) * res["oos_months"] / 12.0, 0.01)
        label = "IS-Top-N-Verfahren: OOS-Trades des jeweiligen IS-Siegers (blind)"
    else:
        agg = res.get("agg") or []
        r = np.asarray(agg[0]["pooled_rets"], float) if agg else np.array([])
        years = max(agg[0]["oos_folds"] * res["oos_months"] / 12.0, 0.01) if agg else None
        label = f"Häufigstes Top-Setup — {_label(agg[0]['key'], sl_mode)}" if agg else ""
    if len(r) == 0:
        st.info("Keine Trades für diese Auswahl.")
        return

    n_real = len(r)
    tpy = n_real / years if years else n_real
    st.success(f"**{n_real} echte Trades** geladen ({label}) · Win-Rate: {(r > 0).mean()*100:.1f}% · "
               f"Ø Trade: {r.mean():.3f}% vom Konto · Trades/Jahr: ~{tpy:.0f}")
    if n_real < 20:
        st.warning(f"⚠️ Nur {n_real} Trades — Monte Carlo Ergebnisse haben hohe Unsicherheit. "
                   f"Mind. 50 Trades empfohlen für verlässliche Aussagen.")
    st.info("ℹ️ Kein Zeitlimit — die Simulation läuft bis das Profit-Ziel erreicht **oder** die Drawdown-Grenze verletzt "
            "wird. Wie auf der DAX-Seite zählt jeder Trade als eigener Tag für den Daily-DD (bei bis zu 3 parallelen "
            "Positionen ist das leicht optimistisch).")
    if risk0 != risk_default:
        st.caption(f"ℹ️ Trades wurden mit {risk0:g} % Risiko gerechnet — Monte Carlo skaliert auf das unten eingestellte Risiko.")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**Challenge-Regeln (Prop Firm)**")
        cap = st.number_input("Startkapital ($)", 10_000, 200_000, 100_000, step=10_000, key="zz_mc_cap")
        pt1 = st.number_input("Phase 1 Profit-Ziel (%)", 1.0, 20.0, 8.0, step=0.5, key="zz_mc_pt1", help="FTMO Phase 1: 8%")
        pt2 = st.number_input("Phase 2 Profit-Ziel (%)", 1.0, 20.0, 5.0, step=0.5, key="zz_mc_pt2", help="FTMO Phase 2: 5%")
        tdd = st.number_input("Max. Total Drawdown (%)", 1.0, 20.0, 10.0, step=0.5, key="zz_mc_tdd", help="FTMO: 10%")
        ddd = st.number_input("Max. Daily Drawdown (%)", 0.5, 10.0, 5.0, step=0.5, key="zz_mc_ddd", help="FTMO: 5%")
    with mc2:
        st.markdown("**Simulation**")
        n_sims = int(st.number_input("Anzahl Simulationen", 500, 5000, 1000, step=500, key="zz_mc_sims"))
        max_t = int(st.number_input("Max. Trades pro Simulation", 10, 500, 100, step=10, key="zz_mc_maxt",
                                    help="Sicherheitsnetz: nach X Trades ohne Ergebnis gilt die Sim als 'nicht bestanden'"))
        risk = st.number_input("Risiko pro Trade (%)", 0.1, 5.0, 1.0, step=0.1, key="zz_mc_risk")
    with mc3:
        st.markdown("**Funded Account — Payout-Ziele**")
        po1 = st.number_input("1. Payout-Ziel (%)", 0.5, 20.0, 1.0, step=0.5, key="zz_mc_po1")
        po2 = st.number_input("2. Payout-Ziel (%)", 0.5, 20.0, 2.0, step=0.5, key="zz_mc_po2")
        st.caption("Simuliert wird ab dem Kapital am Ende von Phase 2 weiter, mit denselben DD-Regeln — nur für Sims, "
                   "die Phase 1+2 bestanden haben.")

    if not st.button("▶ Monte Carlo starten", type="primary", key="zz_mc_run"):
        return
    prog = st.progress(0, text="Monte Carlo läuft …")
    norm = r / risk0                      # R-Multiples (Rendite in % des Kontos / verwendetes Risiko-%)
    results, paths = [], []
    for i in range(n_sims):
        p1, p1r, p1t, p1p, path = _run_phase(cap, pt1, norm, risk, tdd, ddd, max_t)
        p2, p2r, p2t, p2p = False, "", 0, 0.0
        if p1:
            p2, p2r, p2t, p2p, _ = _run_phase(cap, pt2, norm, risk, tdd, ddd, max_t)
        f1, f1t, f2, f2t = False, 0, False, 0
        if p2:
            f1, f1t, f2, f2t = _run_funded_phase(cap * (1 + p2p / 100), po1, po2, norm, risk, tdd, ddd, max_t)
        results.append({"p1_passed": p1, "p1_reason": p1r, "p1_trades": p1t, "p2_passed": p2, "p2_reason": p2r,
                        "p2_trades": p2t, "payout": p1 and p2, "fp1_ok": f1, "fp1_trades": f1t, "fp2_ok": f2,
                        "fp2_trades": f2t})
        if i < 200:
            paths.append(path)
        if i % 100 == 0:
            prog.progress(i / n_sims, text=f"Simulation {i}/{n_sims} …")
    prog.progress(1.0, text="Monte Carlo abgeschlossen ✓")

    d = pd.DataFrame(results)
    n_p1 = int(d["p1_passed"].sum()); n_p2 = int(d["p2_passed"].sum()); n_pay = int(d["payout"].sum())
    n_p1f = n_sims - n_p1; n_p2f = int((d["p1_passed"] & ~d["p2_passed"]).sum())
    p1_pct = n_p1 / n_sims * 100
    p2_cond = n_p2 / n_p1 * 100 if n_p1 else 0
    pay_pct = n_pay / n_sims * 100
    t_p1 = d.loc[d["p1_passed"], "p1_trades"].mean() if n_p1 else 0
    t_p2 = d.loc[d["p2_passed"], "p2_trades"].mean() if n_p2 else 0
    wk = 52.0 / tpy if tpy > 0 else 1.0           # Trades → Wochen
    n_f1, n_f2 = int(d["fp1_ok"].sum()), int(d["fp2_ok"].sum())
    t_f1 = d.loc[d["fp1_ok"], "fp1_trades"].mean() if n_f1 else 0
    t_f2 = d.loc[d["fp2_ok"], "fp2_trades"].mean() if n_f2 else 0
    g1, g2 = d[d["fp1_ok"]], d[d["fp2_ok"]]
    tot1 = (g1["p1_trades"] + g1["p2_trades"] + g1["fp1_trades"]).mean() if len(g1) else 0
    tot2 = (g2["p1_trades"] + g2["p2_trades"] + g2["fp2_trades"]).mean() if len(g2) else 0

    if pay_pct >= 40:
        bc, bt = "#22c55e", f"✅ GUTE PAYOUT-CHANCE — {pay_pct:.1f}% der Simulationen bestehen BEIDE Phasen"
    elif pay_pct >= 15:
        bc, bt = "#f0c040", f"⚠️ MÖGLICH — {pay_pct:.1f}% der Simulationen bestehen BEIDE Phasen"
    else:
        bc, bt = "#ef5350", f"❌ SCHWIERIG — nur {pay_pct:.1f}% der Simulationen erhalten einen Payout"
    st.markdown(f'<div style="background:{bc}22;border:2px solid {bc};border-radius:10px;padding:16px 24px;'
                f'font-weight:800;font-size:1.3rem;color:{bc};margin:16px 0;">{bt}</div>', unsafe_allow_html=True)

    def _tw(t):
        return f"{t:.0f} Tr. (~{t*wk:.0f} Wo.)" if t > 0 else "–"
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Phase 1 besteht", f"{p1_pct:.1f}%", f"{n_p1}/{n_sims}")
    k2.metric("Phase 2 | P1 ok", f"{p2_cond:.1f}%", f"{n_p2}/{n_p1 if n_p1 else '–'}")
    k3.metric("💰 Payout", f"{pay_pct:.1f}%", f"{n_pay}/{n_sims}", help="P(Phase1) × P(Phase2|Phase1)")
    k4.metric("Ø Dauer Phase 1", _tw(t_p1))
    k5.metric("Ø Dauer Phase 2", _tw(t_p2))
    k6.metric("Ø Gesamt", _tw(t_p1 + t_p2))
    st.caption(f"Phase 1 — Daily DD: {(d['p1_reason']=='Daily DD').sum()}x · Total DD: {(d['p1_reason']=='Total DD').sum()}x  |  "
               f"Phase 2 — Daily DD: {(d['p2_reason']=='Daily DD').sum()}x · Total DD: {(d['p2_reason']=='Total DD').sum()}x  ·  "
               f"Wochen = Trades × 52 / {tpy:.0f} Trades pro Jahr")

    fig = go.Figure(go.Pie(labels=["❌ Phase 1 gescheitert", "⚠️ Phase 2 gescheitert", "💰 Payout erhalten"],
                           values=[n_p1f, n_p2f, n_pay], hole=0.4, pull=[0, 0, 0.07], textinfo="label+percent",
                           marker=dict(colors=["#ef5350", "#f0c040", "#22c55e"], line=dict(color="#1a1a2e", width=2))))
    fig.update_layout(title=f"Challenge-Ergebnis aus {n_sims} Simulationen", height=380, template="plotly_dark",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=60, b=60, l=20, r=20),
                      annotations=[dict(text=f"<b>{pay_pct:.1f}%</b><br>Payout", x=0.5, y=0.5, font_size=16,
                                        font_color="#22c55e", showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

    if paths:
        st.subheader("Phase 1 — Simulierte Verläufe")
        fm = go.Figure()
        for pth in paths:
            okp = (pth[-1] - cap) / cap * 100 >= pt1
            fm.add_trace(go.Scatter(y=pth, mode="lines", line=dict(color="#22c55e" if okp else "#ef5350", width=0.5),
                                    opacity=0.12, showlegend=False))
        ml = max(len(pth) for pth in paths)
        fm.add_trace(go.Scatter(y=np.mean([pth + [pth[-1]] * (ml - len(pth)) for pth in paths], axis=0), mode="lines",
                                line=dict(color="white", width=2.5, dash="dash"), name="Ø Pfad"))
        fm.add_hline(y=cap * (1 + pt1 / 100), line_color="#22c55e", line_dash="dot", line_width=2,
                     annotation_text=f"Phase 1 Ziel +{pt1}%", annotation_position="right")
        fm.add_hline(y=cap * (1 - tdd / 100), line_color="#ef5350", line_dash="dot", line_width=2,
                     annotation_text=f"Ruin -{tdd}%", annotation_position="right")
        fm.add_hline(y=cap, line_color="white", line_dash="dash", line_width=1)
        fm.update_layout(title=f"Phase 1 — Grün = bestanden · Rot = gescheitert", height=420, template="plotly_dark",
                         yaxis_title="Kapital ($)", xaxis_title="Trade #", margin=dict(t=50, b=30, l=70, r=130))
        st.plotly_chart(fm, use_container_width=True)

    st.markdown("---")
    st.subheader("Funded Account — Payout-Wahrscheinlichkeit")
    st.caption(f"Basis: {n_p2}/{n_sims} Sims, die Phase 1 + Phase 2 bestanden haben.")
    f1c, f2c, f3c, f4c = st.columns(4)
    f1c.metric(f"{po1:.1f}% Payout | Funded", f"{(n_f1 / n_p2 * 100) if n_p2 else 0:.1f}%", f"{n_f1}/{n_p2 or '–'}")
    f2c.metric(f"{po1:.1f}% Payout | alle Sims", f"{n_f1 / n_sims * 100:.1f}%", f"{n_f1}/{n_sims}")
    f3c.metric(f"{po2:.1f}% Payout | Funded", f"{(n_f2 / n_p2 * 100) if n_p2 else 0:.1f}%", f"{n_f2}/{n_p2 or '–'}")
    f4c.metric(f"{po2:.1f}% Payout | alle Sims", f"{n_f2 / n_sims * 100:.1f}%", f"{n_f2}/{n_sims}")
    w1, w2 = st.columns(2)
    w1.metric(f"Ø Dauer im Funded Account bis {po1:.1f}%", _tw(t_f1))
    w2.metric(f"Ø Dauer im Funded Account bis {po2:.1f}%", _tw(t_f2))
    st.markdown("#### Zusammenfassung — Gesamtdauer bis zum Payout")
    s1c, s2c = st.columns(2)
    s1c.metric(f"Ø gesamt bis {po1:.1f}% Payout", _tw(tot1))
    s2c.metric(f"Ø gesamt bis {po2:.1f}% Payout", _tw(tot2))
    if n_real < 30:
        st.warning(f"⚠️ Nur {n_real} echte Trades — Monte Carlo Ergebnisse haben hohe Unsicherheit.")


# ════════════════════════════════════════════════════════════════════════════
# Seite
# ════════════════════════════════════════════════════════════════════════════
def render_zigzag_strategy(save_fn=None, load_fn=None) -> None:
    st.header("Zig Zag Strategie Test")
    st.caption("Python-Port deines Pine-Scripts „ZigZag Sweep Strategy 3-Positionen“ · bewertet mit denselben "
               "Verfahren wie die DAX EMA Strategie (IS-Top-N, PBO, Deflated Sharpe, Monte Carlo)")

    with st.expander("ℹ️ Was wird hier getestet?", expanded=False):
        st.markdown("""
**Strategie (1:1 aus dem Pine-Script):**
- **ZigZag-Level:** Pivots mit `Pivot-Legs/2` Bars links + rechts. Kommt ein Gegen-Pivot mit mindestens *Deviation %*
  Abstand, wird der letzte Pivot zum **Level**. Das Level wird **gültig**, sobald der Kurs die 50 %-Mitte (Validität)
  zurückgelaufen ist.
- **Sweep:** Ein gültiges Level wird überschritten (High > Level bzw. Low < Level) → **Entry am Close** dieses Bars.
  *Continuation:* High-Sweep → Long, Low-Sweep → Short. *Reversal:* umgekehrt.
- **Bis zu 3 Positionen** gleichzeitig, jede mit eigenem SL / TP / Time-Exit. Nur gleichgerichtet: ein Signal gegen die
  offene Richtung wird übersprungen (wie TradingView-Netting).
- **Stop:** x × Tages-ATR (des Vortages) oder % vom Kurs · **TP:** nur Time-Exit oder x × Tages-ATR · **Time-Exit** nach N Bars am Close.

**Unterschiede zu TradingView:** SL und TP im selben Bar → SL zählt zuerst (pessimistisch). Gap über SL/TP → Fill zum Open.
Die Python-Engine wurde Trade für Trade gegen eine wörtliche Bar-für-Bar-Übersetzung des Pine-Codes geprüft.

**Paare:** USDJPY, XAUUSD, XAGUSD (vorausgewählt). **Daten:** D1 = Pepperstone-MT5 (`data/mt5`), H4/H1/M15 = Dukascopy (`data/dukascopy_intraday`, ab 2015, UTC). Fehlt ein Timeframe (z.B. XAUUSD H4), wird er aus M15 zusammengefasst. XAGUSD gibt es bisher nur auf D1.
**Kosten:** Kommission % je Order (Entry + Exit) aus dem Pine-Kommentar: FX-Majors 0,004 %, FX-Minors 0,020 %, Metalle 0,0075 %, plus optionaler Spread.
        """)

    t_over, t_set, t_det, t_mc, t_single = st.tabs(
        ["📊 Überblick", "⚙️ Einstellungen", "🔍 Robustheit (rollierend)", "🎲 Monte Carlo", "🧰 Einzel-Backtest"])

    # ════════════════════════════ ⚙️ EINSTELLUNGEN ════════════════════════════
    with t_set:
        st.subheader("Markt & Daten")
        m1, m2, m3 = st.columns([1, 3, 1])
        with m1:
            tf = st.selectbox("Timeframe", ze.TIMEFRAMES, index=1, key="zz_tf")
        syms_all = ze.available_symbols(tf)
        with m2:
            _def = [s for s in ZZ_SYMBOLS if s in syms_all] or syms_all[:1]
            _miss = [s for s in ZZ_SYMBOLS if s not in syms_all]
            syms = st.multiselect("Paare (je Paar eigene 3 Slots, Trades werden gepoolt)", syms_all, default=_def,
                                  key=f"zz_syms_{tf}")
            if _miss:
                st.caption(f"⚠️ Keine {tf}-Daten für {', '.join(_miss)} — nur auf D1 (Pepperstone) verfügbar.")
        with m3:
            d_start = st.date_input("Daten ab", _dt.date(2006, 1, 1) if tf == "D1" else _dt.date(2015, 1, 1),
                                    key=f"zz_start_{tf}")

        st.subheader("Strategie (fest, nicht im Grid)")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            sl_mode = st.selectbox("Stop-Modus", ["atr", "pct"], format_func=lambda x: {"atr": "x Tages-ATR", "pct": "% vom Kurs"}[x],
                                   key="zz_slmode")
            atr_len = st.number_input("ATR-Länge", 1, 50, 14, key="zz_atrlen")
        with s2:
            max_levels = st.number_input("Max aktive Level (Speicher)", 20, 2000, 300, key="zz_maxlv",
                                         help="Wie im Pine: Obergrenze für noch nicht gesweepte Level.")
            risk_pct = st.number_input("Risiko pro Trade %", 0.1, 5.0, 1.0, step=0.1, key="zz_risk",
                                       help="Sizing wie im Pine (qty = Risiko / Stop-Abstand). Pine-Default 3.000 $ auf 100.000 $ = 3 %.")
        with s3:
            comm_override = st.checkbox("Kommission für alle Paare überschreiben", False, key="zz_comm_ovr")
            comm_val = st.number_input("Kommission % je Order", 0.0, 0.2, 0.004, step=0.001, format="%.4f",
                                       key="zz_comm", disabled=not comm_override,
                                       help="Ohne Haken: Preset je Paar (Major 0,004 · Minor 0,020 · Metall 0,0075).")
        with s4:
            spread_pips = st.number_input("Spread (Pips, Round-Turn)", 0.0, 50.0, 0.0, step=0.1, key="zz_spread",
                                          help="Pine-Default: 0 (nur Kommission). Dukascopy-Kurse sind Bid.")

        st.markdown("---")
        st.subheader("Grid-Suchraum (In-Sample-Auswahl)")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            g_dir = st.multiselect("Richtung", ["continuation", "reversal"], default=["continuation", "reversal"], key="zz_gdir")
            g_dev = st.multiselect("Deviation %", _DEV_OPTIONS, default=_DEV_DEFAULTS[tf], key=f"zz_gdev_{tf}")
        with g2:
            g_depth = st.multiselect("Pivot-Legs (depth)", [4, 6, 8, 10, 14, 20], default=[6, 10, 14], key="zz_gdepth")
            g_ret = st.multiselect("50%-Validität", [0.0, 0.382, 0.5, 0.618], default=[0.5], key="zz_gret")
        with g3:
            _slo = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if sl_mode == "atr" else [0.25, 0.5, 1.0, 1.5, 2.5]
            _sld = [1.0, 1.5, 2.0] if sl_mode == "atr" else [0.5, 1.0, 2.5]
            g_sl = st.multiselect("Stop (× ATR)" if sl_mode == "atr" else "Stop %", _slo, default=_sld, key=f"zz_gsl_{sl_mode}")
            g_tp = st.multiselect("TP (0 = nur Time-Exit, sonst × ATR)", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], default=[0.0], key="zz_gtp")
        with g4:
            g_tb = st.multiselect("Time-Exit (Bars)", [5, 10, 15, 20, 30, 40, 60], default=[10, 20, 40], key="zz_gtb")
            g_mc = st.multiselect("Max gleichzeitige Positionen", [1, 2, 3], default=[3], key="zz_gmc")
        combos = _build_combos(g_dir, g_dev, g_depth, g_ret, g_sl, g_tp, g_tb, g_mc)
        st.caption(f"Aktueller Suchraum: **{len(combos)} Varianten** × {len(syms)} Paar(e).")

        st.markdown("---")
        st.markdown("**Top-N-Test, Holdout & rollierender Robustheits-Check** (wie DAX EMA)")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            mode_lbl = st.selectbox("IS-Fenster", ["Wachsend (ab Datenstart)", "Rollierend (feste Länge)"], key="zz_mode")
            wf_mode = "anchored" if mode_lbl.startswith("Wachsend") else "rolling"
            is_m = st.number_input("Erstes / festes IS-Fenster (Monate)", 12, 120, 36, step=6, key="zz_is")
        with w2:
            oos_m = st.number_input("OOS-Fenster (Monate)", 3, 24, 12, key="zz_oos")
            top_n = st.number_input("Top-N (beste In-Sample-Setups)", 3, 15, 10, key="zz_topn")
            split_d = st.date_input("Out-of-Sample ab (Split-Test)", _dt.date(2020, 1, 1), key="zz_split")
        with w3:
            hold_d = st.date_input("Holdout ab", _dt.date(2024, 1, 1), key="zz_hold")
            min_tr = st.number_input("Min. IS-Trades", 5, 200, 20, key="zz_mintr")
        with w4:
            opt_metric = st.selectbox("Optimierungsziel", ["pf", "sharpe", "wr"], format_func=_OPT_LBL.get, key="zz_opt")
            n_blocks = st.selectbox("PBO-Blöcke (CSCV)", [8, 10, 12], index=1, key="zz_blocks")

    # ════════════════════════════ Daten ════════════════════════════
    if not syms:
        with t_over:
            st.warning("Unter ⚙️ Einstellungen mindestens ein Paar wählen.")
        return
    frames = {}
    for s in syms:
        df = _load(s, tf)
        if not df.empty:
            frames[s] = df[df.index >= pd.Timestamp(d_start)]
    frames = {s: df for s, df in frames.items() if len(df) > 300}
    if not frames:
        with t_over:
            st.error(f"Keine Daten für {', '.join(syms)} auf {tf}.")
        return
    comm_by_sym = {s: (float(comm_val) if comm_override else ze.COST_PRESETS[ze.asset_class(s)]) for s in frames}
    base = {"tf": tf, "sl_mode": sl_mode, "atr_len": int(atr_len), "max_levels": int(max_levels),
            "risk_pct": float(risk_pct), "spread_pips": float(spread_pips), "comm_by_sym": comm_by_sym}
    engine = _multi_engine(frames, base)
    d0 = min(df.index[0] for df in frames.values()).normalize()
    d1 = max(df.index[-1] for df in frames.values())
    hold_ts = pd.Timestamp(hold_d)
    folds = _topn.build_folds(d0, min(d1, hold_ts - pd.Timedelta(days=1)), int(oos_m), wf_mode, int(is_m)) if hold_ts > d0 else []
    dummy = pd.DataFrame({"close": [1.0, 1.0]}, index=[d0, d1])
    cfg_key = (f"v{ENGINE_VERSION}|{tf}|{sorted(frames)}|{d_start}|{sl_mode}|{atr_len}|{max_levels}|{risk_pct}|"
               f"{comm_by_sym}|{spread_pips}|{g_dir}|{g_dev}|{g_depth}|{g_ret}|{g_sl}|{g_tp}|{g_tb}|{g_mc}|"
               f"{wf_mode}|{is_m}|{oos_m}|{top_n}|{split_d}|{hold_d}|{min_tr}|{opt_metric}|{n_blocks}")

    if "zz_result" not in st.session_state and not st.session_state.get("zz_gist_tried") and load_fn:
        st.session_state["zz_gist_tried"] = True
        rest = load_fn(_GIST_SLOT)
        if rest is not None and rest.get("engine_version") == ENGINE_VERSION:
            st.session_state["zz_result"] = rest

    def _compute():
        prog = st.progress(0.0, text="Backtest aller Varianten …")
        base_p = {k: v for k, v in base.items() if k != "comm_by_sym"}
        tbk = _topn.run_all_combos(dummy, base_p, combos, engine,
                                   progress_cb=lambda f, t: prog.progress(min(0.85, f * 0.85), text=t))
        prog.progress(0.86, text="Split-Test: Top-N nach In-Sample → OOS → Holdout …")
        split = _topn.split_topn(tbk, d0, pd.Timestamp(split_d), hold_ts, d1, opt_metric, int(min_tr), int(top_n))
        prog.progress(0.88, text="Rollierender Check: IS-Ranking → Top-N → OOS …")
        wf = _topn.topn_walk_forward(tbk, folds, opt_metric, int(min_tr), int(top_n))
        agg = _topn.aggregate_setups(wf["details"], float(oos_m) / 12.0)
        prog.progress(0.94, text="Overfitting-Check (PBO + Deflated Sharpe) …")
        M = _topn.monthly_matrix(tbk, d0, hold_ts)
        pbo = _topn.cscv_pbo(M, int(n_blocks))
        srs = (M.mean() / M.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        best = srs.idxmax() if srs.notna().any() else None
        dsr = (_topn.deflated_sharpe(M[best].to_numpy(), srs.to_numpy(), M.shape[1]) if best is not None
               else {"dsr": np.nan, "sr": np.nan, "sr0": np.nan})
        prog.empty()
        folds_out = []
        for f in wf["folds"]:
            fo = {k: v for k, v in f.items() if k != "winner_oos_rets"}
            fo["winner_oos_rets"] = [float(x) for x in f.get("winner_oos_rets", [])]
            folds_out.append(fo)
        agg_out = []
        if not agg.empty:
            for _, rr in agg.head(30).iterrows():
                row = {c: rr[c] for c in agg.columns if c != "pooled_rets"}
                row["pooled_rets"] = [float(x) for x in rr["pooled_rets"]]
                agg_out.append(row)
        res = {"engine_version": ENGINE_VERSION, "cfg_key": cfg_key, "sl_mode": sl_mode,
               "created": str(pd.Timestamp.now(tz="Europe/Berlin").strftime("%d.%m.%Y %H:%M")),
               "market": f"{', '.join(sorted(frames))} · {tf}", "n_variants": len(combos), "folds": folds_out,
               "details": [[{"rank": d["rank"], "key": d["key"], "is": d["is"], "oos": d["oos"]} for d in det]
                           for det in wf["details"]],
               "agg": agg_out, "pbo": {"pbo": pbo["pbo"], "n_splits": pbo["n_splits"]},
               "dsr": {**dsr, "best_key": best}, "risk_pct": float(risk_pct), "opt_metric": opt_metric,
               "top_n": int(top_n), "oos_months": int(oos_m), "split": split}
        st.session_state["zz_result"] = res
        if save_fn:
            ok, why = save_fn(_GIST_SLOT, res)
            if not ok:
                st.caption(f"⚠️ Ergebnis nur für diese Session gespeichert — {why}")

    # ════════════════════════════ 📊 ÜBERBLICK ════════════════════════════
    with t_over:
        st.caption(f"**Aktuelle Einstellungen:** {', '.join(sorted(frames))} · {tf} · {len(combos)} Varianten · "
                   f"Stop {'× ATR' if sl_mode == 'atr' else '%'} · Risiko {risk_pct:g} % · Top-{int(top_n)} · "
                   f"OOS ab {split_d} · Holdout ab {hold_ts.date()} — ändern unter ⚙️ Einstellungen")
        cb, ci = st.columns([1, 2])
        bad_split = pd.Timestamp(split_d) >= hold_ts or pd.Timestamp(split_d) <= d0
        with cb:
            run = st.button("▶ Test starten", type="primary", key="zz_run", use_container_width=True,
                            disabled=len(folds) < 2 or not combos or bad_split)
        with ci:
            st.caption("Alle Varianten werden nur In-Sample bewertet; die besten Top-N laufen unverändert Out-of-Sample "
                       f"und im Holdout. Daten: {d0.date()} – {d1.date()} · {len(folds)} rollierende Folds.")
        if bad_split:
            st.warning("„Out-of-Sample ab“ muss nach dem Datenstart und vor „Holdout ab“ liegen (⚙️ Einstellungen).")
        if run:
            _compute()

        res = st.session_state.get("zz_result")
        if res is None or "split" not in res:
            st.info("Noch kein Ergebnis — **▶ Test starten** klicken.")
        else:
            rs_sl = res.get("sl_mode", sl_mode)
            if res.get("cfg_key") != cfg_key:
                st.warning(f"Angezeigt: Ergebnis vom {res.get('created', '?')} ({res.get('market', '')}) mit **anderen "
                           f"Einstellungen** — ▶ Test starten, um mit den aktuellen neu zu rechnen.")
            sp = res["split"]
            a0, a1, a2, a3 = (pd.Timestamp(sp[k]) for k in ("start", "oos_start", "hold", "end"))
            st.markdown(f"### Top-{len(sp['top'])} Setups: In-Sample → Out-of-Sample → Holdout")
            st.caption(f"🟦 **In-Sample** {a0.date()} – {a1.date()} (nur hier wird sortiert, nach {_OPT_LBL[res['opt_metric']]}) · "
                       f"🟩 **Out-of-Sample** {a1.date()} – {a2.date()} · 🟧 **Holdout** {a2.date()} – {a3.date()} · "
                       f"{res['n_variants']} Varianten · {res.get('market', '')} · Ergebnis vom {res.get('created', '?')}")
            show_h = st.checkbox("🟧 Holdout-Ergebnisse anzeigen", key="zz_show_hold",
                                 help="Der Holdout ist dein letzter unberührter Test. Wählst du anhand dieser Spalten ein "
                                      "Setup aus oder änderst danach Regeln, ist er kein echter Test mehr.")
            if not sp["top"]:
                st.warning(f"Keine Variante hat im In-Sample mindestens {min_tr} Trades — Min. IS-Trades senken oder mehr Paare wählen.")
            rows = []
            for t in sp["top"]:
                r = {"Rang": t["rank"], **_param_cols(t["key"], rs_sl),
                     "IS PF": round(t["is"]["pf"], 2), "IS Trades": t["is"]["n"], "IS Rend. %": round(t["is"]["total_ret"], 1),
                     "OOS PF": round(t["oos"]["pf"], 2), "OOS Trades": t["oos"]["n"],
                     "OOS Rend. %": round(t["oos"]["total_ret"], 1), "OOS MaxDD %": round(t["oos"]["max_dd"], 1)}
                if show_h:
                    r.update({"Holdout PF": round(t["hold"]["pf"], 2), "Holdout Trades": t["hold"]["n"],
                              "Holdout Rend. %": round(t["hold"]["total_ret"], 1)})
                rows.append(r)
            if rows:
                tdf = pd.DataFrame(rows)
                pfc = [c for c in ("IS PF", "OOS PF", "Holdout PF") if c in tdf.columns]
                st.dataframe(tdf.style.map(lambda v: ("color:#22c55e" if v >= 1.1 else ("color:#f0c040" if v >= 1.0 else "color:#ef5350"))
                                           if isinstance(v, (int, float)) else "", subset=pfc),
                             use_container_width=True, hide_index=True)

                e1, e2 = st.columns([3, 1])
                with e1:
                    esel = st.selectbox("Equity-Kurve für", range(len(sp["top"])), key="zz_eq_sel",
                                        format_func=lambda i: f"Rang {sp['top'][i]['rank']} · {_label(sp['top'][i]['key'], rs_sl)}")
                with e2:
                    eall = st.checkbox(f"Alle Top-{len(sp['top'])} einblenden", key="zz_eq_all")
                fig = go.Figure()
                ymin, ymax = 10_000, 10_000
                for i, t in enumerate(sp["top"]):
                    if not (eall or i == esel) or not t["trades"]:
                        continue
                    dd_ = pd.to_datetime([x[0] for x in t["trades"]])
                    rr = np.array([x[1] for x in t["trades"]], float)
                    if not show_h:
                        mk = dd_ < a2
                        dd_, rr = dd_[mk], rr[mk]
                    eq = np.insert(np.cumprod(1 + rr / 100) * 10_000, 0, 10_000)
                    dd_ = dd_.insert(0, a0)
                    ymin, ymax = min(ymin, eq.min()), max(ymax, eq.max())
                    sel = i == esel
                    fig.add_trace(go.Scatter(x=dd_, y=eq, mode="lines", name=f"Rang {t['rank']}", line_shape="hv",
                                             line=dict(width=2.6 if sel else 1, color="#f8fafc" if sel else None),
                                             opacity=1.0 if sel else 0.45))
                for xa, xb, col, lbl in [(a0, a1, "rgba(59,130,246,0.10)", "In-Sample"),
                                         (a1, a2, "rgba(34,197,94,0.10)", "Out-of-Sample"),
                                         (a2, a3, "rgba(249,115,22,0.12)" if show_h else "rgba(100,116,139,0.10)",
                                          "Holdout" if show_h else "Holdout (verdeckt)")]:
                    fig.add_vrect(x0=xa, x1=xb, fillcolor=col, line_width=0, layer="below", annotation_text=lbl,
                                  annotation_position="top left", annotation_font=dict(size=11, color="#94a3b8"))
                fig.add_hline(y=10_000, line_dash="dash", line_color="#64748b", line_width=1)
                fig.update_layout(height=340, template="plotly_dark", showlegend=eall,
                                  yaxis=dict(title="Kapital (Start 10.000)", range=[ymin * 0.97, ymax * 1.03]),
                                  margin=dict(t=30, b=30, l=60, r=10), legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"📋 Alle {res['n_variants']} getesteten Varianten (In-Sample-Rangliste)"):
                adf = pd.DataFrame([{"IS-Rang": a["rank"] if a["rank"] is not None else "– (zu wenig Trades)",
                                     **_param_cols(a["key"], rs_sl), "IS PF": round(a["is"]["pf"], 2),
                                     "IS Trades": a["is"]["n"], "IS WR %": a["is"]["wr"],
                                     "IS Rend. %": round(a["is"]["total_ret"], 1), "IS MaxDD %": round(a["is"]["max_dd"], 1)}
                                    for a in sp["all"]])
                st.dataframe(adf, use_container_width=True, hide_index=True, height=360)
                st.download_button("⬇️ Alle Varianten als CSV", adf.to_csv(index=False).encode(),
                                   file_name="zigzag_alle_varianten_is.csv", mime="text/csv", key="zz_dl_all")

            tf_ok = [f for f in res["folds"] if not f.get("empty")]
            if tf_ok:
                rets = np.concatenate([np.asarray(f["winner_oos_rets"], float) for f in tf_ok])
                yrs = len(tf_ok) * res["oos_months"] / 12.0
                pm = _topn.metrics_from_rets(rets, yrs)
                cagr = ((1 + pm["total_ret"] / 100) ** (1 / yrs) - 1) * 100 if yrs > 0 and pm["total_ret"] > -100 else 0
                n_pos = sum(f["winner_oos"]["total_ret"] > 0 for f in tf_ok)
                pbo_v, dsr_v = res["pbo"]["pbo"], res["dsr"].get("dsr")
                ok_pf, ok_f = pm["pf"] >= 1.15, n_pos / len(tf_ok) >= 2 / 3
                ok_pbo = (not _is_num(pbo_v)) or pbo_v < 0.3
                ok_dsr = _is_num(dsr_v) and dsr_v >= 0.95
                if ok_pf and ok_f and ok_pbo and ok_dsr:
                    vc, vt, vs = "#22c55e", "🟢 Robuster Edge", "Hält Out-of-Sample, kein Overfitting, statistisch signifikant."
                elif pm["pf"] >= 1.05 and n_pos / len(tf_ok) >= 0.5 and ((not _is_num(pbo_v)) or pbo_v < 0.5):
                    vc, vt = "#f0c040", "🟡 Schwacher Edge"
                    vs = ("OOS positiv, aber nach Korrektur für die vielen getesteten Varianten nicht signifikant"
                          if not ok_dsr else "OOS positiv, aber knapp") + " — nur mit kleinem Risiko / Forward-Test."
                else:
                    vc, vt, vs = "#ef5350", "🔴 Kein belastbarer Edge", "Out-of-Sample nicht stabil profitabel."
                st.markdown("### Robustheits-Check (rollierend, ohne Holdout)")
                st.markdown(f'<div style="border:2px solid {vc};background:{vc}18;border-radius:12px;padding:12px 18px;'
                            f'margin:4px 0 10px 0;"><span style="font-size:1.2rem;font-weight:800;color:{vc};">{vt}</span>'
                            f'<span style="color:#cbd5e1;margin-left:12px;">{vs}</span></div>', unsafe_allow_html=True)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("PF (OOS, jährlich neu gewählt)", f"{pm['pf']:.2f}")
                k2.metric("Rendite p.a. (OOS)", f"{cagr:+.1f}%")
                k3.metric("Max Drawdown (OOS)", f"{pm['max_dd']:.1f}%")
                k4.metric("Profitable Folds", f"{n_pos}/{len(tf_ok)}")
                st.caption("Details (je Fold, PBO, Deflated Sharpe) → Tab 🔍 Robustheit · Prop-Challenge → 🎲 Monte Carlo")

    # ════════════════════════════ 🔍 ROBUSTHEIT ════════════════════════════
    with t_det:
        res = st.session_state.get("zz_result")
        tf_ok = [f for f in res["folds"] if not f.get("empty")] if res else []
        if not tf_ok:
            st.info("Noch kein Ergebnis — im Tab 📊 Überblick **▶ Test starten** klicken.")
        else:
            rs_sl = res.get("sl_mode", sl_mode)
            pct = [f["winner_oos_pct"] for f in tf_ok if _is_num(f.get("winner_oos_pct"))]
            pbo_v, dsr_v = res["pbo"]["pbo"], res["dsr"].get("dsr")
            st.markdown("#### Overfitting-Check")
            c1, c2, c3 = st.columns(3)
            c1.metric("PBO", f"{pbo_v*100:.0f}%" if _is_num(pbo_v) else "–",
                      help="Probability of Backtest Overfitting (CSCV). < 20 % gut · > 50 % = Overfitting.")
            c2.metric("Deflated Sharpe", f"{dsr_v*100:.0f}%" if _is_num(dsr_v) else "–",
                      help=f"Wahrscheinlichkeit, dass die Sharpe echt > 0 ist — korrigiert für {res['n_variants']} Varianten. > 95 % = signifikant.")
            c3.metric("Ø OOS-Platz des IS-Siegers", f"{np.mean(pct)*100:.0f}%" if pct else "–",
                      help="Perzentil unter allen Varianten im OOS-Fenster. 50 % = IS-Auswahl bringt nichts.")
            st.markdown("#### Je Fold: IS-Sieger → Out-of-Sample")
            ft = pd.DataFrame([{"OOS-Fenster": f"{pd.Timestamp(f['oos_start']).date()} – {pd.Timestamp(f['oos_end']).date()}",
                                "IS-Sieger": _label(f["winner"], rs_sl), "IS PF": round(f["winner_is"]["pf"], 2),
                                "OOS PF": round(f["winner_oos"]["pf"], 2), "OOS Trades": f["winner_oos"]["n"],
                                "OOS Rendite %": round(f["winner_oos"]["total_ret"], 1),
                                "OOS-Platz": f"{f['winner_oos_pct']*100:.0f}%" if _is_num(f.get("winner_oos_pct")) else "–"}
                               for f in tf_ok])
            st.dataframe(ft.style.map(lambda v: ("color:#22c55e" if v > 0 else "color:#ef5350") if isinstance(v, (int, float)) else "",
                                      subset=["OOS Rendite %"]), use_container_width=True, hide_index=True)
            if res["agg"]:
                st.markdown(f"#### Rangliste: am häufigsten in den IS-Top-{res['top_n']}")
                at = pd.DataFrame([{"#": i + 1, "Setup": _label(r["key"], rs_sl), f"× Top-{res['top_n']}": f"{r['n_top']}/{len(tf_ok)}",
                                    "Ø IS-Rang": round(r["avg_rank"], 1), "IS PF": round(r["is_pf"], 2),
                                    "OOS PF": round(r["oos_pf"], 2), "OOS Trades": r["oos_n"], "OOS WR %": r["oos_wr"],
                                    "OOS Rendite %": round(r["oos_total_ret"], 1), "OOS Max DD %": round(r["oos_max_dd"], 1),
                                    "OOS-Folds +": f"{r['oos_folds_pos']}/{r['oos_folds']}"} for i, r in enumerate(res["agg"][:15])])
                st.dataframe(at, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Rangliste als CSV", at.to_csv(index=False).encode(), file_name="zigzag_is_topn.csv",
                                   mime="text/csv", key="zz_dl_agg")
            with st.expander(f"Top-{res['top_n']} je Fold im Detail"):
                for fi, det in enumerate(res["details"]):
                    if not det:
                        continue
                    fd = res["folds"][fi]
                    st.markdown(f"**OOS {pd.Timestamp(fd['oos_start']).date()} – {pd.Timestamp(fd['oos_end']).date()}**")
                    st.dataframe(pd.DataFrame([{"IS-Rang": d["rank"], "Setup": _label(d["key"], rs_sl), "IS PF": d["is"]["pf"],
                                                "IS Trades": d["is"]["n"], "OOS PF": d["oos"]["pf"],
                                                "OOS Rendite %": d["oos"]["total_ret"]} for d in det]),
                                 use_container_width=True, hide_index=True)

    # ════════════════════════════ 🎲 MONTE CARLO ════════════════════════════
    with t_mc:
        res = st.session_state.get("zz_result")
        _render_mc(res, (res or {}).get("sl_mode", sl_mode), float(risk_pct))

    # ════════════════════════════ 🧰 EINZEL-BACKTEST ════════════════════════════
    with t_single:
        st.caption("Ein einzelnes Setup über den kompletten Zeitraum — zum Vergleich mit dem TradingView-Strategy-Tester. "
                   "Kein Walk-Forward, nur zur Orientierung.")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            p_dir = st.selectbox("Richtung", ["continuation", "reversal"], key="zz_p_dir")
            p_dev = st.number_input("Deviation %", 0.01, 100.0, 2.5, step=0.1, key="zz_p_dev")
        with e2:
            p_depth = st.number_input("Pivot-Legs (depth)", 2, 50, 10, key="zz_p_depth")
            p_ret = st.number_input("50%-Validität", 0.0, 1.0, 0.5, step=0.05, key="zz_p_ret")
        with e3:
            p_sl = st.number_input("Stop (× ATR)" if sl_mode == "atr" else "Stop %", 0.1, 20.0,
                                   1.5 if sl_mode == "atr" else 2.5, step=0.1, key=f"zz_p_sl_{sl_mode}")
            p_tpm = st.selectbox("TP-Modus", ["time", "atr"], format_func=lambda x: {"time": "nur Time-Exit", "atr": "x Tages-ATR"}[x], key="zz_p_tpm")
        with e4:
            p_tpa = st.number_input("TP × Tages-ATR", 0.1, 20.0, 1.0, step=0.1, key="zz_p_tpa", disabled=p_tpm == "time")
            p_tb = st.number_input("Time-Exit nach X Bars", 1, 500, 20, key="zz_p_tb")
            p_mc = st.number_input("Max gleichzeitige Positionen", 1, 3, 3, key="zz_p_mc")
        params = {**{k: v for k, v in base.items() if k != "comm_by_sym"}, "dir_mode": p_dir, "deviation": float(p_dev),
                  "depth": int(p_depth), "retrace": float(p_ret), "sl_val": float(p_sl), "tp_mode": p_tpm,
                  "tp_atr": float(p_tpa), "time_bars": int(p_tb), "max_conc": int(p_mc)}
        parts, stats_rows = [], []
        for s, df in frames.items():
            tr, _, stt = ze.backtest(df, {**params, "symbol": s, "commission_pct": comm_by_sym[s]}, return_stats=True)
            if not tr.empty:
                parts.append(tr.assign(Paar=s))
            stats_rows.append({"Paar": s, "Trades": len(tr), "Sweeps": stt["levels_swept"],
                               "übersprungen (Gegenrichtung)": stt["skip_opposite"],
                               "übersprungen (alle Slots voll)": stt["skip_full"],
                               "Kommission %": comm_by_sym[s]})
        trs = pd.concat(parts, ignore_index=True).sort_values("Zeit").reset_index(drop=True) if parts else pd.DataFrame()
        yrs = max((d1 - d0).days / 365.25, 0.01)
        m = _topn.metrics_from_rets(trs["Ret %"].to_numpy(float) if not trs.empty else np.array([]), yrs)
        k = st.columns(6)
        k[0].metric("Trades", m["n"]); k[1].metric("Trades/Jahr", f"{m['n']/yrs:.1f}")
        k[2].metric("Win-Rate", f"{m['wr']:.1f}%"); k[3].metric("Profit Factor", f"{m['pf']:.2f}")
        k[4].metric("Rendite", f"{m['total_ret']:.1f}%"); k[5].metric("Max DD", f"{m['max_dd']:.1f}%")
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
        if not trs.empty:
            eq = np.cumprod(1 + trs["Ret %"].to_numpy(float) / 100) * 10_000
            fe = go.Figure(go.Scatter(x=trs["Zeit"], y=eq, mode="lines", line=dict(color="#f7931a", width=2), line_shape="hv"))
            fe.update_layout(title=f"Equity (Start 10.000, {risk_pct:g} % Risiko je Trade)", height=280,
                             template="plotly_dark", margin=dict(t=35, b=15))
            st.plotly_chart(fe, use_container_width=True)
            yr = trs.assign(Jahr=trs["Zeit"].dt.year).groupby("Jahr").agg(
                Trades=("R", "size"), WinRate=("R", lambda x: round((x > 0).mean() * 100, 1)),
                Summe_R=("R", lambda x: round(x.sum(), 2)), Rendite=("Ret %", lambda x: round((np.prod(1 + x / 100) - 1) * 100, 2)))
            yr.columns = ["Trades", "Win-Rate %", "Summe R", "Rendite %"]
            c1, c2 = st.columns([1, 2])
            c1.dataframe(yr, use_container_width=True)
            show = trs[["Paar", "Entry-Zeit", "Zeit", "Richtung", "Slot", "Entry", "Exit", "Grund", "Bars", "R", "Ret %"]].copy()
            show["Slot"] = "z" + show["Slot"].astype(str)
            show["R"] = show["R"].round(3); show["Ret %"] = show["Ret %"].round(3)
            c2.dataframe(show.rename(columns={"Zeit": "Exit-Zeit"}), use_container_width=True, hide_index=True, height=360)
            st.download_button("⬇️ Trades als CSV", show.to_csv(index=False).encode(), file_name="zigzag_trades.csv",
                               mime="text/csv", key="zz_dl_tr")
