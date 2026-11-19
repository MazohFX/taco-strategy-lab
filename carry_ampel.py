"""Carry-Ampel: Phase und Gefahr von Carry-Trades je Forex-Paar.

Baustein fuer die Seite "Extra: Makro & Sentiment" (macro_dashboard.py). Nutzt die
dort bereits geladenen Daten (MT5-Zinsentscheide, VIX, CFTC-COT-Historien) und laedt
nur zusaetzlich die Tageskurse der Paare mit nennenswerter Zinsdifferenz (Yahoo).

Grundlage (Backtests Nov. 2026, siehe Methodik-Text unten):
  * Trend- und VIX-Filter wirken bei Carry wie eine Crash-Versicherung (2008: -25 % -> -1 bis -8 %),
    verbessern die Sharpe Ratio aber nicht.
  * Ueberfuellte Spekulanten-Positionierung (COT-Spread im 90.-100. Perzentil der letzten 3 Jahre)
    verdoppelt die Quote starker 4-Wochen-Verluste (ca. 3,5 % -> 7 %), stabil 2012-2018 und 2019-2026;
    Juli 2024 lagen USDJPY/AUDJPY/GBPJPY 4-5 Wochen vor dem Unwind auf dem 95.-100. Perzentil.
Schwellen sind daraus abgeleitete, nicht optimierte Startwerte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CARRY_CONFIG = {
    "min_diff": 1.0,          # %-Punkte Leitzinsdifferenz, ab der ein Paar als Carry-Paar gilt
    "sma_days": 200,          # Trend: Kurs vs. 200-Tage-Durchschnitt
    "vol_days": 60,
    "cot_weeks": 156,         # Vergleichszeitraum Positionierung (3 Jahre)
    "crowd_warn": 0.80,       # ab hier: ueberfuellt (gelb)
    "crowd_extreme": 0.90,    # ab hier + Trend gebrochen: rot
    "vix_stress": 20.0,
    "vix_panic": 30.0,
    "vix_spike_days": 5,
    "vix_spike_pct": 30.0,
    "cb_warn_days": 7,        # Notenbank-Sitzung einer der beiden Waehrungen innerhalb von x Tagen
}

AMPEL_STYLE = {"GRÜN": ("#22c55e", "🟢"), "GELB": ("#eab308", "🟡"), "ROT": ("#ef4444", "🔴"),
               "KEIN CARRY": ("#64748b", "⚪")}


# ── reine Berechnung (testbar ohne Streamlit/Netzwerk) ───────────────────────
def vix_state(vix: pd.Series | None) -> dict:
    """Globales Carry-Klima aus dem VIX."""
    c = CARRY_CONFIG
    if vix is None or len(vix.dropna()) < 70:
        return {"level": None, "avg60": None, "spike_pct": None, "state": "unbekannt",
                "text": "Keine VIX-Daten."}
    v = vix.dropna()
    level, avg60 = float(v.iloc[-1]), float(v.iloc[-60:].mean())
    k = c["vix_spike_days"]
    spike = float(v.iloc[-1] / v.iloc[-1 - k] - 1) * 100
    if level >= c["vix_panic"] or spike >= c["vix_spike_pct"]:
        state, text = "Panik", "VIX-Sprung bzw. Panik-Niveau – klassisches Carry-Unwind-Umfeld."
    elif level >= c["vix_stress"]:
        state, text = "Stress", "Erhöhte Volatilität – Carry-Positionen verkleinern, keine neuen aufbauen."
    elif level < avg60:
        state, text = "Ruhig", "Volatilität niedrig und fallend – günstiges Umfeld für Carry."
    else:
        state, text = "Normal", "Volatilität normal, aber über dem 60-Tage-Schnitt – beobachten."
    return {"level": level, "avg60": avg60, "spike_pct": spike, "state": state, "text": text}


def cot_spread_percentile(hist_base: pd.DataFrame | None, hist_quote: pd.DataFrame | None,
                          base: str, quote: str, weeks: int | None = None) -> dict | None:
    """Perzentil der aktuellen Positionierung 'Spekulanten long Basis minus long Quote' im
    Vergleich zu den letzten `weeks` Wochen. USD = 0 (Waehrungs-Futures notieren gegen USD).
    hist_*: DataFrame mit report_date, net_pct."""
    weeks = weeks or CARRY_CONFIG["cot_weeks"]

    def series(h, cur):
        if cur == "USD":
            return None
        if h is None or len(h) == 0:
            return pd.Series(dtype=float)
        return h.sort_values("report_date").set_index("report_date")["net_pct"].astype(float)

    sb, sq = series(hist_base, base), series(hist_quote, quote)
    if sb is None and sq is None:
        return None
    if sb is None:
        spread = -sq
    elif sq is None:
        spread = sb
    else:
        spread = pd.concat([sb, sq], axis=1, join="inner").pipe(lambda d: d.iloc[:, 0] - d.iloc[:, 1])
    spread = spread.dropna()
    if len(spread) < 60:
        return None
    window = spread.iloc[-(weeks + 1):]
    last = float(window.iloc[-1])
    pct = float((window.iloc[:-1] < last).mean())
    return {"spread": last, "percentile": pct, "date": spread.index[-1],
            "history": spread.iloc[-(weeks + 1):]}


def pair_trend(close: pd.Series | None) -> dict | None:
    c = CARRY_CONFIG
    if close is None or len(close.dropna()) < c["sma_days"] + 5:
        return None
    s = close.dropna()
    sma = float(s.iloc[-c["sma_days"]:].mean())
    vol = float(s.pct_change().iloc[-c["vol_days"]:].std() * np.sqrt(252) * 100)
    return {"close": float(s.iloc[-1]), "sma": sma, "above": bool(s.iloc[-1] > sma),
            "dist_pct": float(s.iloc[-1] / sma - 1) * 100, "vol_pct": vol,
            "ret_1m": float(s.iloc[-1] / s.iloc[-22] - 1) * 100 if len(s) > 22 else None}


def carry_status(pair: str, rate_base: float | None, rate_quote: float | None, cycle_diff: float | None,
                 trend: dict | None, crowd: dict | None, vix: dict, days_to_cb: float | None) -> dict:
    """Ampel und Phase fuer ein Paar. Richtung = Seite mit dem hoeheren Zins."""
    c = CARRY_CONFIG
    base, quote = pair[:3], pair[3:]
    if rate_base is None or rate_quote is None:
        return {"pair": pair, "ampel": "KEIN CARRY", "phase": "keine Zinsdaten", "direction": None,
                "diff": None, "reasons": ["Keine Zinsentscheide im MT5-Kalender."], "crowd": None}
    diff = rate_base - rate_quote
    if abs(diff) < c["min_diff"]:
        return {"pair": pair, "ampel": "KEIN CARRY", "phase": f"Zinsdifferenz nur {diff:+.2f} %-Pkt.",
                "direction": None, "diff": diff, "reasons": [], "crowd": None}
    d = 1 if diff > 0 else -1
    direction = "LONG" if d > 0 else "SHORT"
    high, low = (base, quote) if d > 0 else (quote, base)

    crowd_c = None
    if crowd is not None:
        crowd_c = crowd["percentile"] if d > 0 else 1 - crowd["percentile"]
    trend_ok = None if trend is None else (trend["above"] if d > 0 else not trend["above"])
    widening = None if cycle_diff is None else cycle_diff * d > 0.05

    reasons, red, yellow = [], [], []
    if vix["state"] == "Panik":
        red.append("VIX-Sprung/Panik")
    elif vix["state"] == "Stress":
        yellow.append(f"VIX {vix['level']:.1f} ≥ {c['vix_stress']:.0f}")
    if crowd_c is not None:
        if crowd_c >= c["crowd_extreme"] and trend_ok is False:
            red.append(f"Positionierung extrem ({crowd_c * 100:.0f}. Perz.) und Trend gebrochen")
        elif crowd_c >= c["crowd_warn"]:
            yellow.append(f"Carry-Seite überfüllt ({crowd_c * 100:.0f}. Perz.)")
    if trend_ok is False:
        yellow.append(f"Kurs gegen Carry-Richtung (200-Tage-Linie)")
    if days_to_cb is not None and days_to_cb <= c["cb_warn_days"]:
        yellow.append(f"Notenbank-Sitzung in {days_to_cb:.0f} Tagen")

    if red:
        ampel, phase = "ROT", "Unwind-Gefahr"
    elif yellow:
        ampel = "GELB"
        phase = "Überfüllt" if crowd_c is not None and crowd_c >= c["crowd_warn"] else "Fragil"
    else:
        ampel = "GRÜN"
        phase = "Aufbau" if (widening and (crowd_c is None or crowd_c < 0.5)) else "Läuft"
    reasons = red + yellow
    if ampel == "GRÜN":
        reasons.append(f"{high}-Zins {abs(diff):.2f} %-Pkt. über {low}, Trend intakt"
                       + (", Zinsabstand wächst" if widening else ""))
    carry_vol = abs(diff) / trend["vol_pct"] if trend and trend["vol_pct"] > 0 else None
    return {"pair": pair, "ampel": ampel, "phase": phase, "direction": direction, "diff": diff,
            "high": high, "low": low, "crowd": crowd_c, "trend_ok": trend_ok, "widening": widening,
            "carry_vol": carry_vol, "days_to_cb": days_to_cb, "reasons": reasons}


def compute_carry(data: dict, fetch_close=None, now: pd.Timestamp | None = None) -> dict:
    """data = Ergebnis von macro_dashboard.compute_all. fetch_close(symbol, period) -> Series."""
    import macro_dashboard as md
    now = now or data["now"]
    cal = data["calendar"]
    vix = vix_state(data.get("closes", {}).get("vix"))
    rates = {cur: md.policy_rate_info(cal, cur, now) for cur in md.CURRENCIES}
    rows = []
    for pair in md.PAIRS:
        base, quote = pair[:3], pair[3:]
        ib, iq = rates.get(base), rates.get(quote)
        rb, rq = (ib or {}).get("rate"), (iq or {}).get("rate")
        cyc = (ib["change"] - iq["change"]) if ib and iq else None
        trend = None
        if rb is not None and rq is not None and abs(rb - rq) >= CARRY_CONFIG["min_diff"] and fetch_close:
            trend = pair_trend(fetch_close(f"{pair}=X", "2y"))
        cot = data["results"].get(pair, {}).get("pillars", {}).get("cot")
        hb = hq = None
        if cot is not None and cot.available:
            hb = (cot.extra.get("base") or {}).get("history")
            hq = (cot.extra.get("quote") or {}).get("history")
        crowd = cot_spread_percentile(hb, hq, base, quote)
        nxt = [i["next_date"] for i in (ib, iq) if i and i.get("next_date") is not None]
        days_cb = min((pd.Timestamp(t) - now).total_seconds() / 86400 for t in nxt) if nxt else None
        st_ = carry_status(pair, rb, rq, cyc, trend, crowd, vix, days_cb)
        st_["trend"] = trend
        st_["crowd_hist"] = crowd["history"] if crowd else None
        rows.append(st_)
    return {"vix": vix, "rows": rows}


# ── UI ───────────────────────────────────────────────────────────────────────
def render_carry_ampel(data: dict) -> None:
    import streamlit as st
    import macro_dashboard as md
    col = md._COL
    with st.spinner("Carry-Ampel: lade Kurse der Carry-Paare …"):
        res = compute_carry(data, md.fetch_daily_close)
    v = res["vix"]
    st.markdown("### 🌡️ Carry-Ampel")
    st.caption("Wann laufen Carry-Trades (Hochzins- gegen Niedrigzinswährung), wann drohen sie zu kippen? "
               "Kein Einstiegssignal – Filter für Richtung und Positionsgröße deiner H4/H1-Setups.")
    vcol = {"Ruhig": "#4ade80", "Normal": "#cbd5e1", "Stress": "#facc15", "Panik": "#f87171"}.get(v["state"], "#94a3b8")
    vtxt = (f"VIX {v['level']:.1f} · 60T-Schnitt {v['avg60']:.1f} · 5 Tage {v['spike_pct']:+.0f} %"
            if v["level"] is not None else "VIX n/a")
    rows = res["rows"]
    carry_rows = [r for r in rows if r["ampel"] != "KEIN CARRY"]
    n = {k: sum(r["ampel"] == k for r in carry_rows) for k in ("GRÜN", "GELB", "ROT")}
    st.markdown(
        f'<div style="background:{col["bg"]};border:1px solid {vcol};border-left:6px solid {vcol};'
        f'border-radius:10px;padding:12px 16px;margin:4px 0 10px 0">'
        f'<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:baseline">'
        f'<span style="font-size:.75rem;color:{col["muted"]};text-transform:uppercase">Carry-Klima</span>'
        f'<span style="font-size:1.35rem;font-weight:800;color:{vcol}">{v["state"].upper()}</span>'
        f'<span style="color:{col["muted"]}">{vtxt}</span>'
        f'<span style="color:{col["text"]}">🟢 {n["GRÜN"]} · 🟡 {n["GELB"]} · 🔴 {n["ROT"]} Carry-Paare</span></div>'
        f'<div style="color:#cbd5e1;margin-top:6px">{v["text"]}</div></div>', unsafe_allow_html=True)

    if not carry_rows:
        st.info("Kein Paar hat aktuell eine Leitzinsdifferenz ≥ "
                f"{CARRY_CONFIG['min_diff']:.1f} %-Punkte (oder es fehlen MT5-Zinsentscheide).")
    order = {"ROT": 0, "GELB": 1, "GRÜN": 2}
    carry_rows.sort(key=lambda r: (order[r["ampel"]], -abs(r["diff"])))
    table = []
    for r in carry_rows:
        t = r.get("trend") or {}
        table.append({
            "Ampel": f'{AMPEL_STYLE[r["ampel"]][1]} {r["ampel"]}',
            "Paar": r["pair"],
            "Carry-Richtung": f'{r["direction"]} ({r["high"]} > {r["low"]})',
            "Zinsdiff. %-Pkt.": round(abs(r["diff"]), 2),
            "Phase": r["phase"],
            "Positionierung (Perz.)": None if r["crowd"] is None else round(r["crowd"] * 100),
            "Trend 200T": "–" if r["trend_ok"] is None else ("✓ mit Carry" if r["trend_ok"] else "✗ gegen Carry"),
            "Carry/Vol": None if r["carry_vol"] is None else round(r["carry_vol"], 2),
            "Gründe": " · ".join(r["reasons"]),
        })
    if table:
        df = pd.DataFrame(table)

        def _style(val):
            for k, (c_, _) in AMPEL_STYLE.items():
                if isinstance(val, str) and val.endswith(k):
                    return f"color:{c_};font-weight:700"
            return ""
        styled = (df.style.map(_style, subset=["Ampel"])
                  .format({"Zinsdiff. %-Pkt.": "{:.2f}", "Carry/Vol": "{:.2f}",
                           "Positionierung (Perz.)": "{:.0f}"}, na_rep="–"))
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("Positionierung = wie stark Spekulanten (CFTC COT) im Vergleich zu den letzten 3 Jahren "
                   "auf der Carry-Seite stehen; ≥ 80 = überfüllt, ≥ 90 = extrem. Carry/Vol = Zinsdifferenz ÷ "
                   "Jahresvolatilität des Paares (höher = attraktiver).")

        pick = st.selectbox("Positionierung im Verlauf", [r["pair"] for r in carry_rows if r.get("crowd_hist") is not None]
                            or ["–"], key="carry_ampel_pick")
        r = next((x for x in carry_rows if x["pair"] == pick), None)
        if r is not None and r.get("crowd_hist") is not None:
            import plotly.graph_objects as go
            h = r["crowd_hist"] * (1 if r["direction"] == "LONG" else -1) * 100
            pct = h.rank(pct=True) * 100
            fig = go.Figure(go.Scatter(x=h.index, y=h.values, mode="lines", line=dict(color="#60a5fa"),
                                       name="Carry-Seite netto (% OI)"))
            hi = np.nanpercentile(h.values, CARRY_CONFIG["crowd_warn"] * 100)
            fig.add_hline(y=hi, line_dash="dash", line_color="#eab308",
                          annotation_text="80. Perzentil (überfüllt)", annotation_position="top left")
            fig.update_layout(template="plotly_dark", height=260, margin=dict(t=20, b=10, l=10, r=10),
                              yaxis_title="Spekulanten Basis − Quote (% OI)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{pick}: Netto-Positionierung der Spekulanten auf der Carry-Seite "
                       f"({r['high']} long / {r['low']} short), letzte 3 Jahre.")

    with st.expander("ℹ️ Wie die Carry-Ampel entscheidet (und was der Backtest dazu sagt)"):
        st.markdown(f"""
**Carry-Paar:** Leitzinsdifferenz (MT5-Zinsentscheide) ≥ {CARRY_CONFIG['min_diff']:.1f} %-Punkte. Richtung = Hochzinswährung long.

**🔴 ROT – Unwind-Gefahr:** VIX ≥ {CARRY_CONFIG['vix_panic']:.0f} oder VIX +{CARRY_CONFIG['vix_spike_pct']:.0f} % in {CARRY_CONFIG['vix_spike_days']} Tagen, **oder** Positionierung ≥ {CARRY_CONFIG['crowd_extreme']*100:.0f}. Perzentil und Kurs bereits unter der 200-Tage-Linie (gegen Carry).
**🟡 GELB – fragil/überfüllt:** VIX ≥ {CARRY_CONFIG['vix_stress']:.0f}, Positionierung ≥ {CARRY_CONFIG['crowd_warn']*100:.0f}. Perzentil, Trend gegen Carry oder Notenbank-Sitzung in ≤ {CARRY_CONFIG['cb_warn_days']} Tagen.
**🟢 GRÜN:** alles andere. Phase „Aufbau“, wenn der Zinsabstand zuletzt gewachsen ist und die Positionierung noch unter dem 50. Perzentil liegt, sonst „Läuft“.

**Was die Backtests (Nov. 2026) dazu zeigen:**
- Überfüllte Positionierung (≥ 90. Perz.) verdoppelte 2012–2026 die Quote starker 4-Wochen-Verluste von ca. 3,5 % auf ca. 7 % – in beiden Hälften des Zeitraums. Juli 2024 standen USDJPY, AUDJPY und GBPJPY 4–5 Wochen vor dem Yen-Unwind auf dem 95.–100. Perzentil.
- Trend- und VIX-Filter schützen vor Crashs (Carry 2008: −25 % → −1 bis −8 %), kosten in ruhigen Phasen aber Rendite. Die Sharpe Ratio steigt dadurch nicht.
- Die Ampel ist daher ein **Risiko-Filter**: bei Grün Setups in Carry-Richtung bevorzugen, bei Gelb keine neuen Carry-Trades und kleinere Größe, bei Rot die Carry-Seite meiden.

Schwellen sind Startwerte aus diesen Tests, nicht optimiert. Leitzins ≠ Broker-Swap. Kein Finanzrat.
""")
    st.divider()
