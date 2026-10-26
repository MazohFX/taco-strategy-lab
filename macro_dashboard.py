"""Makro-Dashboard pro Waehrungspaar (Horizont 1-4 Wochen).

Ersetzt die alte Seite "Extra: Makro & Sentiment". Fuenf Saeulen pro Paar, alle
aus Sicht der Basiswaehrung (+ = Paar steigt tendenziell):

  1. Zinsen & Notenbanken   (MT5-Zinsentscheide, Forex-Factory-Konsensus)
  2. Risikostimmung         (VIX, S&P 500, Nikkei) -- nur Paare mit JPY/CHF
  3. Daten-Ueberraschungen  (MT5 Surprise-Index Basis minus Quote)
  4. Positionierung (COT)   (CFTC Non-Commercials, 3-Jahres-Perzentil)
  5. Event-Risiko 4 Wochen  (Termine + JPY-Interventionsrisiko) -- NICHT im Score

Aufbau: reine Berechnungsfunktionen (ohne st.*) oben, testbar mit
test_macro_dashboard.py; UI-Funktionen unten. Abhaengigkeiten aus
taco_web_app.py (MT5-Kalender, Surprise-Index, COT-Marktnamen, KI-Analyse)
werden als ``deps``-Dict uebergeben, damit dieses Modul taco_web_app.py nicht
importieren muss (das wuerde die ganze App beim Import ausfuehren).

ALLE Gewichte und Schwellen in CONFIG sind UNVALIDIERTE STARTWERTE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# ── Konfiguration (unvalidierte Startwerte) ──────────────────────────────────
CONFIG = {
    "weights": {"zinsen": 0.35, "risiko": 0.25, "surprise": 0.25, "cot": 0.15},
    # Zinsen: Anteil Zinszyklus (Aenderungen letzte 6 Monate) vs. Zinsniveau (Carry)
    "rates_cycle_weight": 0.6,
    "rates_cycle_scale": 0.5,      # %-Punkte Zyklus-Differenz fuer tanh ~ 0.76
    "rates_level_scale": 2.0,      # %-Punkte Zinsdifferenz fuer tanh ~ 0.76
    "rates_lookback_days": 180,
    # Risiko
    "vix_change_scale_pct": 15.0,
    "equity_ret_scale_pct": 4.0,
    "equity_lookback_days": 20,
    "vix_change_lookback_days": 10,
    # Surprise
    "surprise_scale": 0.5,
    "surprise_trend_days": 20,
    # COT
    "cot_years": 3,
    "cot_trend_weeks": 4,
    "cot_z_scale": 1.5,
    "cot_extreme_high": 0.90,
    "cot_extreme_low": 0.10,
    "cot_extreme_damping": 0.5,
    # Gesamtfazit
    "neutral_band": 0.15,          # |Score| darunter = NEUTRAL
    "support_min": 0.15,           # ab hier zaehlt eine Saeule als "dafuer"
    "oppose_min": 0.30,            # ab hier zaehlt eine Saeule als "klar dagegen"
    # Event-Risiko / JPY-Intervention (Heuristik, keine offizielle Schwelle)
    "event_horizon_days": 28,
    "intervention_level_mid": 155.0,
    "intervention_level_high": 160.0,
    "intervention_move_mid_pct": 4.0,     # USDJPY-Anstieg in 20 Handelstagen
    "intervention_move_high_pct": 6.0,
    "mt5_stale_days": 3,
}

# Alle 28 Forex-Paare der 8 Hauptwaehrungen in Marktkonvention (EUR > GBP > AUD >
# NZD > USD > CAD > CHF > JPY bestimmt, welche Waehrung Basis ist).
_PAIR_ORDER = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
PAIRS = [a + b for i, a in enumerate(_PAIR_ORDER) for b in _PAIR_ORDER[i + 1:]]
# Waehrungs-Futures (CME): notieren die Waehrung in USD, z.B. 6S = CHF/USD.
# DXY = US-Dollar-Index (USD gegen Korb), eigene Logik in FUTURES.
CURRENCY_FUTURES = {"CHF": "CHFUSD", "EUR": "EURUSD", "AUD": "AUDUSD",
                    "NZD": "NZDUSD", "CAD": "CADUSD", "GBP": "GBPUSD",
                    "JPY": "JPYUSD"}
FUTURE_LIST = ["CHF", "EUR", "DXY", "AUD", "NZD", "CAD", "GBP", "JPY"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
SAFE_HAVENS = {"JPY", "CHF"}
# Risiko-Empfindlichkeit je Waehrung (+ = Risk-on-Waehrung, - = sicherer Hafen).
# Unvalidierte Faustregel; fuer ein Paar zaehlt Basis minus Quote.
RISK_BETA = {"AUD": 1.0, "NZD": 1.0, "CAD": 0.5, "GBP": 0.3, "EUR": 0.0,
             "USD": -0.5, "CHF": -0.7, "JPY": -1.0}
RISK_MIN_DIFF = 0.5   # darunter (z.B. AUDNZD, EURGBP) entfaellt die Saeule bei Forex-Paaren

RATE_DECISION_CODES = {
    "USD": "fed-interest-rate-decision", "EUR": "ecb-deposit-rate-decision",
    "GBP": "boe-interest-rate-decision", "AUD": "rba-interest-rate-decision",
    "NZD": "rbnz-interest-rate-decision", "CAD": "boc-interest-rate-decision",
    "CHF": "snb-interest-rate-decision", "JPY": "boj-interest-rate-decision",
}
CENTRAL_BANK_NAMES = {"USD": "Fed", "EUR": "EZB", "GBP": "BoE", "AUD": "RBA",
                      "NZD": "RBNZ", "CAD": "BoC", "CHF": "SNB", "JPY": "BoJ"}

# Futures/Indizes (wie auf der alten Seite). Vorzeichen = wie der jeweilige
# Treiber auf das Instrument wirkt (unvalidierte Faustregeln):
#   rates:    hawkishere Fed relativ zu den anderen Notenbanken
#   surprise: US-Daten besser als erwartet
#   risk:     Risk-on (0 = kein klarer Zusammenhang -> Saeule entfaellt)
FUTURES = {
    "Gold":       {"cot": "Gold",           "rates": -1, "surprise": -1, "risk": -1},
    "DXY":        {"cot": "DXY",            "rates": +1, "surprise": +1, "risk": -0.5},
    "Nasdaq 100": {"cot": "NQ Futures",     "rates": -1, "surprise": +1, "risk": +1},
    "S&P 500":    {"cot": "S&P500 Futures", "rates": -1, "surprise": +1, "risk": +1},
}
# Fuer Aktienindizes nur den VIX als Risiko-Input nehmen (S&P-/Nikkei-Returns
# waeren sonst der eigene Kurs).
EQUITY_INDICES = {"Nasdaq 100", "S&P 500"}
FUTURE_SYMBOLS = {"Gold": "GC=F", "DXY": "DX-Y.NYB", "Nasdaq 100": "^NDX", "S&P 500": "^GSPC",
                  "EUR": "6E=F", "GBP": "6B=F", "AUD": "6A=F", "NZD": "6N=F", "CAD": "6C=F", "CHF": "6S=F", "JPY": "6J=F"}
# Exakte, aktuell gepflegte CFTC-Marktnamen (geprueft Okt. 2026). Die Volltextsuche
# ueber COT_WATCHLIST trifft fuer NZD und DXY die alten, 2022 eingestellten Namen
# ("NEW ZEALAND DOLLAR", "U.S. DOLLAR INDEX") -> dort gab es keine aktuellen Daten.
COT_EXACT_MARKETS = {
    "EURO Futures": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "Pfund Futures": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "YEN Futures": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "AUD Futures": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "NZD Futures": "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CANADA Futures": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CHF Futures": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    "DXY": "USD INDEX - ICE FUTURES U.S.",
}
CURRENCY_COT_LABELS = {
    "USD": "DXY", "EUR": "EURO Futures", "GBP": "Pfund Futures", "JPY": "YEN Futures",
    "AUD": "AUD Futures", "NZD": "NZD Futures", "CAD": "CANADA Futures", "CHF": "CHF Futures",
}

PILLAR_LABELS = {
    "zinsen": "Zinsen & Notenbanken", "risiko": "Risikostimmung",
    "surprise": "Daten-Überraschungen", "cot": "Positionierung (COT)",
    "event": "Event-Risiko 4 Wochen",
}


def split_pair(pair: str) -> tuple[str, str]:
    return pair[:3], pair[3:]


@dataclass
class Pillar:
    key: str
    score: float | None
    reason: str
    available: bool
    base: str = ""
    quote: str = ""
    extra: dict = field(default_factory=dict)


def _tanh(x: float, scale: float) -> float:
    return float(np.tanh(x / scale)) if scale else 0.0


def _na(key: str, reason: str) -> Pillar:
    return Pillar(key=key, score=None, reason=reason, available=False)


# ── Saeule 1: Zinsen & Notenbanken ───────────────────────────────────────────
def policy_rate_info(calendar: pd.DataFrame, currency: str, now: pd.Timestamp,
                     lookback_days: int | None = None) -> dict | None:
    """Letzter Leitzins, Veraenderung ueber lookback und naechster Termin."""
    lookback_days = lookback_days or CONFIG["rates_lookback_days"]
    code = RATE_DECISION_CODES.get(currency)
    if calendar is None or calendar.empty or code is None:
        return None
    ev = calendar[calendar["event_code"] == code].sort_values("time_utc")
    done = ev[ev["actual"].notna() & (ev["time_utc"] <= now)]
    if done.empty:
        return None
    last = done.iloc[-1]
    before = done[done["time_utc"] <= last["time_utc"] - pd.Timedelta(days=lookback_days)]
    ref = float(before["actual"].iloc[-1]) if not before.empty else float(done["actual"].iloc[0])
    upcoming = ev[ev["actual"].isna() & (ev["time_utc"] > now)]
    nxt = upcoming["time_utc"].iloc[0] if not upcoming.empty else None
    return {"rate": float(last["actual"]), "change": float(last["actual"]) - ref,
            "last_date": last["time_utc"], "next_date": nxt}


def score_rates(pair: str, calendar: pd.DataFrame, now: pd.Timestamp,
                expected_change: Callable[[str], float | None] | None = None) -> Pillar:
    base, quote = split_pair(pair)
    ib, iq = policy_rate_info(calendar, base, now), policy_rate_info(calendar, quote, now)
    if ib is None or iq is None:
        missing = [c for c, i in ((base, ib), (quote, iq)) if i is None]
        return _na("zinsen", f"Keine MT5-Zinsentscheide für {', '.join(missing)}.")
    exp_b = expected_change(base) if expected_change else None
    exp_q = expected_change(quote) if expected_change else None
    cycle_diff = (ib["change"] + (exp_b or 0.0)) - (iq["change"] + (exp_q or 0.0))
    level_diff = ib["rate"] - iq["rate"]
    w = CONFIG["rates_cycle_weight"]
    score = w * _tanh(cycle_diff, CONFIG["rates_cycle_scale"]) + (1 - w) * _tanh(level_diff, CONFIG["rates_level_scale"])
    cb_b, cb_q = CENTRAL_BANK_NAMES[base], CENTRAL_BANK_NAMES[quote]
    if abs(cycle_diff) < 0.05:
        cyc = "Zinszyklus beider Notenbanken zuletzt gleich"
    elif cycle_diff > 0:
        cyc = f"{cb_b} zuletzt hawkisher als {cb_q}"
    else:
        cyc = f"{cb_q} zuletzt hawkisher als {cb_b}"
    reason = f"{cyc}; Zinsdifferenz {level_diff:+.2f} %-Pkt."
    expected_txt = []
    for cur, e in ((base, exp_b), (quote, exp_q)):
        if e:
            expected_txt.append(f"{CENTRAL_BANK_NAMES[cur]} Konsensus {e:+.2f}")
    if expected_txt:
        reason += " (" + ", ".join(expected_txt) + ")"
    return Pillar("zinsen", float(np.clip(score, -1, 1)), reason, True, base, quote,
                  {"base": ib, "quote": iq, "cycle_diff": cycle_diff, "level_diff": level_diff,
                   "exp_base": exp_b, "exp_quote": exp_q})


# ── Saeule 2: Risikostimmung ─────────────────────────────────────────────────
def risk_appetite(vix: pd.Series, spx: pd.Series, nikkei: pd.Series) -> dict | None:
    """Risk-on-Score in [-1, 1] (+ = Risk-on). Serien = Tagesschlusskurse."""
    parts, info = [], {}
    vix = vix.dropna() if vix is not None else pd.Series(dtype=float)
    if len(vix) >= 30:
        last = float(vix.iloc[-1])
        pct = float((vix <= last).mean())
        info["vix"], info["vix_pct"] = last, pct
        parts.append(-(pct - 0.5) * 2)
        n = CONFIG["vix_change_lookback_days"]
        if len(vix) > n:
            chg = (last / float(vix.iloc[-1 - n]) - 1) * 100
            info["vix_chg"] = chg
            parts.append(-_tanh(chg, CONFIG["vix_change_scale_pct"]))
    n = CONFIG["equity_lookback_days"]
    for name, s in (("spx", spx), ("nikkei", nikkei)):
        s = s.dropna() if s is not None else pd.Series(dtype=float)
        if len(s) > n:
            ret = (float(s.iloc[-1]) / float(s.iloc[-1 - n]) - 1) * 100
            info[f"{name}_ret"] = ret
            parts.append(_tanh(ret, CONFIG["equity_ret_scale_pct"]))
    if not parts:
        return None
    info["score"] = float(np.clip(np.mean(parts), -1, 1))
    return info


def risk_diff(pair: str) -> float:
    base, quote = split_pair(pair)
    return RISK_BETA.get(base, 0.0) - RISK_BETA.get(quote, 0.0)


def risk_relevant(pair: str, force: bool = False) -> bool:
    return force or abs(risk_diff(pair)) >= RISK_MIN_DIFF


def score_risk(pair: str, appetite: dict | None, force: bool = False) -> Pillar:
    base, quote = split_pair(pair)
    if not risk_relevant(pair, force):
        return _na("risiko", f"{base} und {quote} reagieren ähnlich auf die Risikostimmung – Säule entfällt.")
    if appetite is None:
        return _na("risiko", "VIX/Aktienkurse gerade nicht abrufbar.")
    sign = risk_diff(pair)
    raw = appetite["score"]
    mood = "Risk-on" if raw > 0.15 else "Risk-off" if raw < -0.15 else "gemischte Stimmung"
    parts = []
    if "vix" in appetite:
        parts.append(f"VIX {appetite['vix']:.1f} ({appetite['vix_pct'] * 100:.0f}. Perzentil 1J)")
    if "spx_ret" in appetite:
        parts.append(f"S&P 500 {appetite['spx_ret']:+.1f}% / 20T")
    reason = f"{mood}: " + ", ".join(parts)
    return Pillar("risiko", float(np.clip(sign * raw, -1, 1)), reason, True, base, quote,
                  {**appetite, "beta_diff": sign})


# ── Saeule 3: Daten-Ueberraschungen ──────────────────────────────────────────
def score_surprise(pair: str, surprise: Callable[[str], tuple[float | None, int]],
                   history: pd.DataFrame | None = None) -> Pillar:
    base, quote = split_pair(pair)
    sb, nb = surprise(base)
    sq, nq = surprise(quote)
    if sb is None or sq is None:
        missing = [c for c, s in ((base, sb), (quote, sq)) if s is None]
        return _na("surprise", f"Keine auswertbaren MT5-Releases für {', '.join(missing)}.")
    diff = sb - sq
    score = _tanh(diff, CONFIG["surprise_scale"])
    trend = None
    if history is not None and not history.empty and base in history and quote in history:
        d = (history[base] - history[quote]).dropna()
        n = CONFIG["surprise_trend_days"]
        if len(d) > n:
            trend = float(d.iloc[-1] - d.iloc[-1 - n])
    better = base if diff > 0 else quote
    if abs(diff) < 0.1:
        reason = f"Daten beider Seiten etwa wie erwartet ({base} {sb:+.2f}, {quote} {sq:+.2f})"
    else:
        reason = f"{better}-Daten überraschen stärker positiv ({base} {sb:+.2f}, {quote} {sq:+.2f})"
    if trend is not None and abs(trend) >= 0.1:
        reason += f"; Trend 20T {'zugunsten ' + base if trend > 0 else 'zugunsten ' + quote}"
    return Pillar("surprise", float(score), reason, True, base, quote,
                  {"base": sb, "quote": sq, "n_base": nb, "n_quote": nq, "diff": diff, "trend": trend})


# ── Saeule 4: Positionierung (COT) ───────────────────────────────────────────
def cot_currency_stats(hist: pd.DataFrame) -> dict | None:
    """hist: report_date, net_pct (Netto Non-Commercials / Open Interest)."""
    if hist is None or hist.empty or len(hist) < 30:
        return None
    s = hist.sort_values("report_date")["net_pct"].dropna()
    if len(s) < 30:
        return None
    last = float(s.iloc[-1])
    pct = float((s <= last).mean())
    k = CONFIG["cot_trend_weeks"]
    changes = s.diff(k).dropna()
    chg = float(s.iloc[-1] - s.iloc[-1 - k])
    std = float(changes.std()) if len(changes) > 5 else 0.0
    z = chg / std if std > 0 else 0.0
    return {"net_pct": last, "percentile": pct, "change": chg, "z": z,
            "score": _tanh(z, CONFIG["cot_z_scale"]),
            "date": hist["report_date"].max()}


def score_cot(pair: str, stats: Callable[[str], dict | None]) -> Pillar:
    base, quote = split_pair(pair)
    b, q = stats(base), stats(quote)
    if b is None or q is None:
        missing = [c for c, s in ((base, b), (quote, q)) if s is None]
        return _na("cot", f"Keine COT-Historie für {', '.join(missing)}.")
    score = (b["score"] - q["score"]) / 2
    hi, lo = CONFIG["cot_extreme_high"], CONFIG["cot_extreme_low"]
    flags = []
    crowded_up = crowded_down = False
    for cur, s, is_base in ((base, b, True), (quote, q, False)):
        if s["percentile"] >= hi:
            flags.append(f"{cur}-Longs überfüllt ({s['percentile'] * 100:.0f}. Perz.)")
            crowded_up |= is_base
            crowded_down |= not is_base
        elif s["percentile"] <= lo:
            flags.append(f"{cur}-Shorts überfüllt ({s['percentile'] * 100:.0f}. Perz.)")
            crowded_down |= is_base
            crowded_up |= not is_base
    if (score > 0 and crowded_up) or (score < 0 and crowded_down):
        score *= CONFIG["cot_extreme_damping"]
    if abs(score) < 0.1:
        reason = "Spekulanten verändern ihre Positionen kaum"
    else:
        reason = f"Spekulanten bauen in den letzten 4 Wochen Richtung {base if score > 0 else quote} auf"
    if flags:
        reason += "; ⚠️ " + ", ".join(flags) + " – Squeeze-Risiko"
    return Pillar("cot", float(np.clip(score, -1, 1)), reason, True, base, quote,
                  {"base": b, "quote": q, "flags": flags})


# ── Saeule 5: Event-Risiko ───────────────────────────────────────────────────
def intervention_level(usdjpy: pd.Series | None) -> dict | None:
    if usdjpy is None:
        return None
    s = usdjpy.dropna()
    if len(s) < 21:
        return None
    last = float(s.iloc[-1])
    move = (last / float(s.iloc[-21]) - 1) * 100
    c = CONFIG
    if last >= c["intervention_level_high"] or (last >= c["intervention_level_mid"] and move >= c["intervention_move_high_pct"]):
        level = "hoch"
    elif last >= c["intervention_level_mid"] or move >= c["intervention_move_mid_pct"]:
        level = "mittel"
    else:
        level = "niedrig"
    return {"usdjpy": last, "move_20d": move, "level": level}


def upcoming_events(calendar: pd.DataFrame, pair: str, now: pd.Timestamp) -> pd.DataFrame:
    base, quote = split_pair(pair)
    if calendar is None or calendar.empty:
        return pd.DataFrame()
    end = now + pd.Timedelta(days=CONFIG["event_horizon_days"])
    ev = calendar[calendar["currency"].isin([base, quote]) & (calendar["time_utc"] > now)
                  & (calendar["time_utc"] <= end) & (calendar["importance"] == "High")].copy()
    ev["notenbank"] = ev["event_code"].astype(str).isin(RATE_DECISION_CODES.values())
    return ev.sort_values("time_utc")


def score_event(pair: str, calendar: pd.DataFrame, now: pd.Timestamp,
                usdjpy: pd.Series | None = None) -> Pillar:
    base, quote = split_pair(pair)
    ev = upcoming_events(calendar, pair, now)
    covered_until = calendar["time_utc"].max() if calendar is not None and not calendar.empty else None
    cb = ev[ev["notenbank"]] if not ev.empty else ev
    interv = intervention_level(usdjpy) if "JPY" in (base, quote) else None
    level = "niedrig"
    reasons = []
    if not cb.empty:
        first = cb["time_utc"].iloc[0]
        days = (first - now).total_seconds() / 86400
        names = ", ".join(sorted({CENTRAL_BANK_NAMES[c] for c in cb["currency"]}))
        reasons.append(f"Notenbank-Sitzung ({names}) in {days:.0f} Tagen")
        level = "hoch" if days <= 7 else "mittel"
    n_high = len(ev)
    if n_high >= 6 and level == "niedrig":
        level = "mittel"
    reasons.append(f"{n_high} High-Impact-Termine")
    if interv:
        reasons.append(f"Interventionsrisiko {interv['level']} (USDJPY {interv['usdjpy']:.1f}, {interv['move_20d']:+.1f}% / 20T)")
        order = ["niedrig", "mittel", "hoch"]
        level = order[max(order.index(level), order.index(interv["level"]))]
    if covered_until is not None and covered_until < now + pd.Timedelta(days=CONFIG["event_horizon_days"]):
        reasons.append(f"Kalender reicht nur bis {covered_until:%d.%m.}")
    return Pillar("event", None, "; ".join(reasons), True, base, quote,
                  {"level": level, "events": ev, "intervention": interv, "covered_until": covered_until})


# ── Futures / Indizes ────────────────────────────────────────────────────────
def score_rates_future(name: str, calendar: pd.DataFrame, now: pd.Timestamp,
                       expected_change: Callable[[str], float | None] | None = None) -> Pillar:
    spec = FUTURES[name]
    infos = {c: policy_rate_info(calendar, c, now) for c in CURRENCIES}
    usd = infos.get("USD")
    if usd is None:
        return _na("zinsen", "Keine MT5-Zinsentscheide für die Fed.")

    def cyc(c):
        e = expected_change(c) if expected_change else None
        return infos[c]["change"] + (e or 0.0)

    others = [cyc(c) for c in CURRENCIES if c != "USD" and infos.get(c) is not None]
    if not others:
        return _na("zinsen", "Zu wenige Zinsentscheide anderer Notenbanken.")
    rel = cyc("USD") - float(np.mean(others))
    score = spec["rates"] * _tanh(rel, CONFIG["rates_cycle_scale"])
    if abs(rel) < 0.05:
        txt = "Fed im Gleichschritt mit den anderen Notenbanken"
    elif rel > 0:
        txt = "Fed zuletzt hawkisher als die anderen Notenbanken"
    else:
        txt = "Fed zuletzt dovisher als die anderen Notenbanken"
    effect = "Rückenwind" if score > 0.15 else "Gegenwind" if score < -0.15 else "neutral"
    reason = f"{txt} ({rel:+.2f} %-Pkt.) – für {name} {effect}"
    return Pillar("zinsen", float(np.clip(score, -1, 1)), reason, True, "USD", "",
                  {"usd": usd, "rel": rel, "others_mean": float(np.mean(others))})


def score_risk_future(name: str, appetite: dict | None, appetite_vix: dict | None) -> Pillar:
    sign = FUTURES[name]["risk"]
    if sign == 0:
        return _na("risiko", "Kein klarer Risk-on/Risk-off-Zusammenhang.")
    app = appetite_vix if name in EQUITY_INDICES else appetite
    if app is None:
        return _na("risiko", "VIX/Aktienkurse gerade nicht abrufbar.")
    raw = app["score"]
    mood = "Risk-on" if raw > 0.15 else "Risk-off" if raw < -0.15 else "gemischte Stimmung"
    parts = []
    if "vix" in app:
        parts.append(f"VIX {app['vix']:.1f} ({app['vix_pct'] * 100:.0f}. Perzentil 1J)")
    if "spx_ret" in app:
        parts.append(f"S&P 500 {app['spx_ret']:+.1f}% / 20T")
    return Pillar("risiko", float(sign * raw), f"{mood}: " + ", ".join(parts), True, extra=dict(app))


def score_surprise_future(name: str, surprise: Callable[[str], tuple[float | None, int]]) -> Pillar:
    s, n = surprise("USD")
    if s is None:
        return _na("surprise", "Keine auswertbaren US-Releases.")
    score = FUTURES[name]["surprise"] * _tanh(s, CONFIG["surprise_scale"])
    how = "besser" if s > 0.1 else "schlechter" if s < -0.1 else "etwa wie"
    reason = f"US-Daten {how} als erwartet (USD {s:+.2f}, {n} Releases)"
    return Pillar("surprise", float(score), reason, True, "USD", "", {"usd": s, "n": n})


def score_cot_future(name: str, stats: dict | None) -> Pillar:
    if stats is None:
        return _na("cot", "Keine COT-Historie verfügbar.")
    score = stats["score"]
    flags = []
    if stats["percentile"] >= CONFIG["cot_extreme_high"]:
        flags.append(f"Longs überfüllt ({stats['percentile'] * 100:.0f}. Perz.)")
        if score > 0:
            score *= CONFIG["cot_extreme_damping"]
    elif stats["percentile"] <= CONFIG["cot_extreme_low"]:
        flags.append(f"Shorts überfüllt ({stats['percentile'] * 100:.0f}. Perz.)")
        if score < 0:
            score *= CONFIG["cot_extreme_damping"]
    if abs(score) < 0.1:
        reason = "Spekulanten verändern ihre Positionen kaum"
    else:
        reason = f"Spekulanten bauen in den letzten 4 Wochen {'Longs' if score > 0 else 'Shorts'} auf"
    if flags:
        reason += "; ⚠️ " + ", ".join(flags) + " – Squeeze-Risiko"
    return Pillar("cot", float(np.clip(score, -1, 1)), reason, True, extra={"single": stats, "flags": flags})


def score_event_future(name: str, calendar: pd.DataFrame, now: pd.Timestamp) -> Pillar:
    # USD-Termine wie bei einem USD-Paar (USDUSD -> beide Seiten USD).
    p = score_event("USDUSD", calendar, now)
    p.base, p.quote = "USD", ""
    return p


def relevant_pillars(name: str, force_risk: bool = False) -> list[str]:
    if name in FUTURES:
        return [k for k in SCORED_PILLARS if k != "risiko" or FUTURES[name]["risk"] != 0]
    return [k for k in SCORED_PILLARS if k != "risiko" or risk_relevant(name, force_risk)]


# ── Gesamtfazit ──────────────────────────────────────────────────────────────
SCORED_PILLARS = ["zinsen", "risiko", "surprise", "cot"]


def overall(pillars: dict[str, Pillar], pair: str, force_risk: bool = False) -> dict:
    relevant = relevant_pillars(pair, force_risk)
    avail = {k: pillars[k].score for k in relevant if pillars.get(k) and pillars[k].score is not None}
    quality = f"{len(avail)}/{len(relevant)}"
    if not avail:
        return {"score": None, "label": "n/a", "quality": quality, "supporters": [], "opposers": []}
    w = CONFIG["weights"]
    total_w = sum(w[k] for k in avail)
    score = float(np.clip(sum(w[k] * v for k, v in avail.items()) / total_w, -1, 1))
    if abs(score) < CONFIG["neutral_band"]:
        return {"score": score, "label": "NEUTRAL", "quality": quality, "supporters": [], "opposers": []}
    d = np.sign(score)
    supporters = [k for k, v in avail.items() if np.sign(v) == d and abs(v) >= CONFIG["support_min"]]
    opposers = [k for k, v in avail.items() if np.sign(v) == -d and abs(v) >= CONFIG["oppose_min"]]
    if len(supporters) >= 2 and not opposers:
        label = "LONG" if d > 0 else "SHORT"
    else:
        label = "GEMISCHT"
    return {"score": score, "label": label, "quality": quality, "supporters": supporters, "opposers": opposers}


def summary_sentence(pair: str, pillars: dict[str, Pillar], verdict: dict) -> str:
    lbl = verdict["label"]
    names = lambda ks: ", ".join(PILLAR_LABELS[k] for k in ks)
    if lbl in ("LONG", "SHORT"):
        if pair in FUTURES:
            side = f"{pair} {'steigend' if lbl == 'LONG' else 'fallend'}"
        else:
            base, quote = split_pair(pair)
            side = base if lbl == "LONG" else quote
        return f"{names(verdict['supporters'])} sprechen für {side}; keine Säule stark dagegen."
    if lbl == "GEMISCHT":
        txt = "Die Säulen widersprechen sich"
        if verdict["supporters"]:
            txt += f": dafür {names(verdict['supporters'])}"
        if verdict["opposers"]:
            txt += f", dagegen {names(verdict['opposers'])}"
        return txt + "."
    if lbl == "NEUTRAL":
        return "Kein klarer fundamentaler Vorteil für eine Seite."
    return "Zu wenig Daten für ein Fazit."


# ═════════════════════════════════════════════════════════════════════════════
# Daten laden (Streamlit-Cache) und UI
# ═════════════════════════════════════════════════════════════════════════════
try:
    import streamlit as st
except Exception:  # pragma: no cover - Tests ohne Streamlit
    st = None

COT_SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


def _cache(ttl: int):
    if st is None:
        return lambda f: f
    return st.cache_data(ttl=ttl, show_spinner=False)


@_cache(60 * 60)
def fetch_daily_close(symbol: str, period: str = "1y") -> pd.Series:
    try:
        import yfinance as yf
        data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        s = data["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)


@_cache(24 * 60 * 60)
def fetch_cot_net_pct(market_name: str | None, years: int) -> pd.DataFrame:
    if not market_name:
        return pd.DataFrame()
    try:
        import requests
        start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%dT00:00:00.000")
        params = {
            "$where": f"market_and_exchange_names = '{market_name}' AND report_date_as_yyyy_mm_dd >= '{start}'",
            "$select": "report_date_as_yyyy_mm_dd,noncomm_positions_long_all,noncomm_positions_short_all,open_interest_all",
            "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": 2000,
        }
        r = requests.get(COT_SOCRATA_URL, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            return df
        df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
        for c in ("noncomm_positions_long_all", "noncomm_positions_short_all", "open_interest_all"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["net_pct"] = (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / df["open_interest_all"]
        return df[["report_date", "net_pct"]].dropna()
    except Exception:
        return pd.DataFrame()


def compute_all(deps: dict) -> dict:
    """Alle Paare berechnen. deps: siehe render_macro_dashboard."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    calendar = deps["load_calendar"]()
    history = deps["surprise_history"](365)
    expected = deps.get("expected_rate_change")

    closes = {k: fetch_daily_close(sym) for k, sym in
              (("vix", "^VIX"), ("spx", "^GSPC"), ("nikkei", "^N225"), ("usdjpy", "JPY=X"))}
    appetite = risk_appetite(closes["vix"], closes["spx"], closes["nikkei"])
    appetite_vix = risk_appetite(closes["vix"], None, None)

    cot_cache: dict[str, dict | None] = {}

    def cot_by_label(label: str) -> dict | None:
        if label not in cot_cache:
            if label in COT_EXACT_MARKETS:
                market = COT_EXACT_MARKETS[label]
            elif "cot_market_label" in deps:
                market = deps["cot_market_label"](label)
            else:  # aeltere Einbindung: nur Waehrungen
                cur = next((c for c, l in CURRENCY_COT_LABELS.items() if l == label), None)
                market = deps["cot_market"](cur) if cur else None
            hist = fetch_cot_net_pct(market, CONFIG["cot_years"])
            stats = cot_currency_stats(hist)
            if stats is not None:
                stats["history"] = hist
            cot_cache[label] = stats
        return cot_cache[label]

    def cot_stats(cur: str) -> dict | None:
        return cot_by_label(CURRENCY_COT_LABELS[cur])

    results = {}
    for pair in PAIRS:
        pillars = {
            "zinsen": score_rates(pair, calendar, now, expected),
            "risiko": score_risk(pair, appetite),
            "surprise": score_surprise(pair, deps["surprise_index"], history),
            "cot": score_cot(pair, cot_stats),
            "event": score_event(pair, calendar, now, closes["usdjpy"]),
        }
        results[pair] = {"pillars": pillars, "verdict": overall(pillars, pair), "code": pair}
    for cur, code in CURRENCY_FUTURES.items():
        pillars = {
            "zinsen": score_rates(code, calendar, now, expected),
            "risiko": score_risk(code, appetite, force=True),
            "surprise": score_surprise(code, deps["surprise_index"], history),
            # COT des Futures selbst (nicht Waehrung minus USD-Index)
            "cot": score_cot_future(cur, cot_stats(cur)),
            "event": score_event(code, calendar, now, closes["usdjpy"]),
        }
        results[cur] = {"pillars": pillars, "verdict": overall(pillars, code, force_risk=True), "code": code}
    for name in [n for n in FUTURE_LIST if n in FUTURES]:
        spec = FUTURES[name]
        pillars = {
            "zinsen": score_rates_future(name, calendar, now, expected),
            "risiko": score_risk_future(name, appetite, appetite_vix),
            "surprise": score_surprise_future(name, deps["surprise_index"]),
            "cot": score_cot_future(name, cot_by_label(spec["cot"])),
            "event": score_event_future(name, calendar, now),
        }
        results[name] = {"pillars": pillars, "verdict": overall(pillars, name), "code": name}
    return {"now": now, "calendar": calendar, "history": history, "closes": closes,
            "appetite": appetite, "results": results}


# ── UI-Bausteine ─────────────────────────────────────────────────────────────
_COL = {"pos": "#22c55e", "neg": "#ef4444", "neu": "#94a3b8", "warn": "#eab308",
        "bg": "#0f1724", "border": "rgba(148,163,184,.18)", "text": "#e2e8f0", "muted": "#94a3b8"}
_LABEL_COL = {"LONG": _COL["pos"], "SHORT": _COL["neg"], "GEMISCHT": _COL["warn"],
              "NEUTRAL": _COL["neu"], "n/a": "#475569"}
_EVENT_COL = {"niedrig": _COL["pos"], "mittel": _COL["warn"], "hoch": _COL["neg"]}


def _score_cell(score: float | None) -> str:
    if score is None:
        return "–"
    arrow = "▲" if score >= 0.15 else "▼" if score <= -0.15 else "●"
    return f"{arrow} {score:+.2f}"


def _cell_style(val: str) -> str:
    if not isinstance(val, str):
        return ""
    if val.startswith("▲") or val.startswith("LONG"):
        return "color:#4ade80;font-weight:600"
    if val.startswith("▼") or val.startswith("SHORT"):
        return "color:#f87171;font-weight:600"
    if val.startswith("GEMISCHT") or val == "mittel":
        return "color:#facc15;font-weight:600"
    if val == "hoch":
        return "color:#f87171;font-weight:600"
    return "color:#94a3b8"


def _bar_html(score: float | None) -> str:
    if score is None:
        return f'<div style="height:6px;background:#1e293b;border-radius:3px;margin:8px 0"></div>'
    pct = abs(score) * 50
    color = _COL["pos"] if score > 0 else _COL["neg"]
    left = 50 if score >= 0 else 50 - pct
    return (f'<div style="position:relative;height:6px;background:#1e293b;border-radius:3px;margin:8px 0">'
            f'<div style="position:absolute;left:50%;top:-2px;width:1px;height:10px;background:#475569"></div>'
            f'<div style="position:absolute;left:{left}%;width:{pct}%;height:6px;background:{color};border-radius:3px"></div></div>')


def _card(title: str, value: str, color: str, body: str, bar: str = "") -> str:
    return (f'<div style="background:{_COL["bg"]};border:1px solid {_COL["border"]};border-radius:10px;'
            f'padding:12px 14px;height:100%;min-height:150px">'
            f'<div style="font-size:.72rem;color:{_COL["muted"]};text-transform:uppercase;letter-spacing:.04em">{title}</div>'
            f'<div style="font-size:1.25rem;font-weight:700;color:{color};margin-top:2px">{value}</div>'
            f'{bar}<div style="font-size:.8rem;color:#cbd5e1;line-height:1.35">{body}</div></div>')


def _render_overview(data: dict, names: list[str], index_label: str = "Paar") -> None:
    rows = []
    for pair in names:
        r = data["results"][pair]
        p, v = r["pillars"], r["verdict"]
        rows.append({
            index_label: pair,
            "Fazit": f"{v['label']} ({v['score']:+.2f})" if v["score"] is not None else v["label"],
            "Zinsen": _score_cell(p["zinsen"].score),
            "Risiko": _score_cell(p["risiko"].score),
            "Überraschungen": _score_cell(p["surprise"].score),
            "COT": _score_cell(p["cot"].score),
            "Event-Risiko": p["event"].extra.get("level", "–"),
            "Daten": v["quality"],
        })
    df = pd.DataFrame(rows).set_index(index_label)
    st.dataframe(df.style.map(_cell_style), use_container_width=True, height=(min(len(df), 12) + 1) * 35 + 3)


def _render_verdict(title: str, r: dict, pair: str | None = None) -> None:
    pair = pair or title
    p, v = r["pillars"], r["verdict"]
    ev = p["event"].extra.get("level", "–")
    color = _LABEL_COL.get(v["label"], _COL["neu"])
    score_txt = f"{v['score']:+.2f}" if v["score"] is not None else "–"
    st.markdown(
        f'<div style="background:{_COL["bg"]};border:1px solid {color};border-left:6px solid {color};'
        f'border-radius:10px;padding:16px 20px;margin:6px 0 14px 0">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:14px">'
        f'<span style="font-size:1.6rem;font-weight:800;color:{_COL["text"]}">{title}</span>'
        f'<span style="font-size:1.6rem;font-weight:800;color:{color}">{v["label"]}</span>'
        f'<span style="color:{_COL["muted"]}">Score {score_txt} · Daten {v["quality"]} Säulen · '
        f'Event-Risiko <b style="color:{_EVENT_COL.get(ev, _COL["neu"])}">{ev}</b></span></div>'
        f'<div style="color:#cbd5e1;margin-top:6px">{summary_sentence(pair, p, v)}</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    for col, key in zip(cols, ["zinsen", "risiko", "surprise", "cot", "event"]):
        pil = p[key]
        if key == "event":
            lvl = pil.extra.get("level", "–")
            html = _card(PILLAR_LABELS[key], lvl.upper(), _EVENT_COL.get(lvl, _COL["neu"]), pil.reason)
        elif pil.score is None:
            html = _card(PILLAR_LABELS[key], "n/a", "#475569", pil.reason, _bar_html(None))
        else:
            c = _COL["pos"] if pil.score >= 0.15 else _COL["neg"] if pil.score <= -0.15 else _COL["neu"]
            html = _card(PILLAR_LABELS[key], f"{pil.score:+.2f}", c, pil.reason, _bar_html(pil.score))
        col.markdown(html, unsafe_allow_html=True)


def _plotly_layout(fig, height=320, **kw):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(t=30, b=20, l=50, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), **kw)
    return fig


def _tab_rates(pair: str, r: dict) -> None:
    pil = r["pillars"]["zinsen"]
    if not pil.available:
        st.info(pil.reason)
        return
    if pair in FUTURES:
        st.write(pil.reason)
        u = pil.extra["usd"]
        st.dataframe(pd.DataFrame([{
            "Fed-Leitzins": f"{u['rate']:.2f} %", "Änderung 6 Mon.": f"{u['change']:+.2f}",
            "Ø Änderung andere Notenbanken": f"{pil.extra['others_mean']:+.2f}",
            "Letzter Entscheid": f"{u['last_date']:%d.%m.%Y}",
            "Nächster Termin": f"{u['next_date']:%d.%m.%Y}" if u["next_date"] is not None else "nicht im Kalender",
        }]), use_container_width=True, hide_index=True)
        return
    rows = []
    for side in ("base", "quote"):
        cur = pil.base if side == "base" else pil.quote
        info = pil.extra[side]
        exp = pil.extra[f"exp_{side}"]
        rows.append({
            "Währung": cur, "Notenbank": CENTRAL_BANK_NAMES[cur],
            "Leitzins": f"{info['rate']:.2f} %",
            "Änderung 6 Mon.": f"{info['change']:+.2f}",
            "Letzter Entscheid": f"{info['last_date']:%d.%m.%Y}",
            "Nächster Termin": f"{info['next_date']:%d.%m.%Y}" if info["next_date"] is not None else "nicht im Kalender",
            "Konsensus nächster Entscheid": f"{exp:+.2f}" if exp else "–",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Zinsdifferenz {pil.extra['level_diff']:+.2f} %-Pkt. · Zyklus-Differenz {pil.extra['cycle_diff']:+.2f} %-Pkt. "
               "Konsensus nur, wenn der Entscheid diese Woche im Forex-Factory-Kalender steht.")


def _tab_risk(data: dict, pair: str, r: dict) -> None:
    import plotly.graph_objects as go
    pil = r["pillars"]["risiko"]
    st.write(pil.reason)
    vix = data["closes"]["vix"]
    if vix.empty:
        st.caption("VIX gerade nicht abrufbar.")
        return
    fig = go.Figure(go.Scatter(x=vix.index, y=vix.values, name="VIX", line=dict(color="#f59e0b")))
    fig.add_hline(y=float(vix.median()), line_dash="dot", line_color="#64748b",
                  annotation_text="Median 1J", annotation_position="top left")
    st.plotly_chart(_plotly_layout(fig, title="VIX (1 Jahr)"), use_container_width=True)


def _tab_surprise(data: dict, pair: str, r: dict) -> None:
    import plotly.graph_objects as go
    pil = r["pillars"]["surprise"]
    st.write(pil.reason)
    hist = data["history"]
    if pair in FUTURES:
        if hist is not None and not hist.empty and "USD" in hist:
            fig = go.Figure(go.Scatter(x=hist.index, y=hist["USD"], name="USD", line=dict(color="#3b82f6")))
            fig.add_hline(y=0, line_color="#475569")
            st.plotly_chart(_plotly_layout(fig, title="US-Surprise-Index (1 Jahr)"), use_container_width=True)
        return
    base, quote = split_pair(pair)
    if hist is None or hist.empty or base not in hist or quote not in hist:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist[base], name=base, line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist[quote], name=quote, line=dict(color="#a855f7")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist[base] - hist[quote], name=f"Differenz {base}−{quote}",
                             line=dict(color="#e2e8f0", width=2, dash="dot")))
    fig.add_hline(y=0, line_color="#475569")
    st.plotly_chart(_plotly_layout(fig, title="Surprise-Index (1 Jahr)"), use_container_width=True)


def _tab_cot(pair: str, r: dict) -> None:
    import plotly.graph_objects as go
    pil = r["pillars"]["cot"]
    st.write(pil.reason)
    if not pil.available:
        return
    if "single" in pil.extra:
        s = pil.extra["single"]
        h = s.get("history")
        if h is not None and not h.empty:
            fig = go.Figure(go.Scatter(x=h["report_date"], y=h["net_pct"] * 100, name=f"{pair} netto % OI",
                                       line=dict(color="#3b82f6")))
            fig.add_hline(y=0, line_color="#475569")
            st.plotly_chart(_plotly_layout(fig, title="Netto-Positionierung Spekulanten (% Open Interest, 3 Jahre)"),
                            use_container_width=True)
        st.dataframe(pd.DataFrame([{"Netto % OI": f"{s['net_pct'] * 100:+.1f}",
                                    "Perzentil 3J": f"{s['percentile'] * 100:.0f}",
                                    "Änderung 4 Wo.": f"{s['change'] * 100:+.1f}",
                                    "Stand": f"{s['date']:%d.%m.%Y}"}]),
                     use_container_width=True, hide_index=True)
        return
    fig = go.Figure()
    for side, color in (("base", "#3b82f6"), ("quote", "#a855f7")):
        s = pil.extra[side]
        h = s.get("history")
        cur = pil.base if side == "base" else pil.quote
        if h is not None and not h.empty:
            fig.add_trace(go.Scatter(x=h["report_date"], y=h["net_pct"] * 100, name=f"{cur} netto % OI",
                                     line=dict(color=color)))
    fig.add_hline(y=0, line_color="#475569")
    st.plotly_chart(_plotly_layout(fig, title="Netto-Positionierung Spekulanten (% Open Interest, 3 Jahre)"),
                    use_container_width=True)
    rows = []
    for side in ("base", "quote"):
        s = pil.extra[side]
        rows.append({"Währung": pil.base if side == "base" else pil.quote,
                     "Netto % OI": f"{s['net_pct'] * 100:+.1f}",
                     "Perzentil 3J": f"{s['percentile'] * 100:.0f}",
                     "Änderung 4 Wo.": f"{s['change'] * 100:+.1f}",
                     "Stand": f"{s['date']:%d.%m.%Y}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("USD über den ICE-Dollar-Index-Future. Perzentil > 90 bzw. < 10 = überfüllte Positionierung.")


def _tab_calendar(pair: str, r: dict, tz: str) -> None:
    pil = r["pillars"]["event"]
    st.write(pil.reason)
    ev = pil.extra.get("events")
    if ev is None or ev.empty:
        st.caption("Keine High-Impact-Termine im verfügbaren Kalender.")
        return
    local = ev["time_utc"].dt.tz_localize("UTC").dt.tz_convert(tz)
    show = pd.DataFrame({
        "Zeit (Berlin)": local.dt.strftime("%a %d.%m. %H:%M").values,
        "Währung": ev["currency"].values,
        "Event": [("🏦 " if nb else "") + e for e, nb in zip(ev["event"], ev["notenbank"])],
        "Forecast": ev["forecast"].values, "Previous": ev["previous"].values,
    })
    st.dataframe(show.style.format({"Forecast": "{:g}", "Previous": "{:g}"}, na_rep=""),
                 use_container_width=True, hide_index=True)


def _tab_method() -> None:
    w = CONFIG["weights"]
    st.markdown(f"""
**Horizont:** 1–4 Wochen. Alle Scores −1 … +1 aus Sicht der Basiswährung (+ = Paar steigt tendenziell).

**Zinsen & Notenbanken** – Leitzinsen aus den MT5-Zinsentscheiden. {CONFIG['rates_cycle_weight']:.0%} Zinszyklus
(Änderung der letzten 6 Monate, Basis minus Quote, plus Forex-Factory-Konsensus wenn der Entscheid diese Woche ansteht),
{1 - CONFIG['rates_cycle_weight']:.0%} Zinsniveau-Differenz (Carry). Markterwartungen für Sitzungen weiter in der Zukunft
(OIS-Pricing) und tägliche 2-jährige Renditen aller Länder gibt es nicht kostenlos – daher nicht enthalten.

**Risikostimmung** – VIX-Perzentil (1 Jahr) und 10-Tage-Veränderung, S&P 500 und Nikkei 20-Tage-Return.
Wirkung je Währung (Faustregel): AUD/NZD +1, CAD +0,5, GBP +0,3, EUR 0, USD −0,5, CHF −0,7, JPY −1. Für ein Paar zählt
Basis minus Quote (AUDJPY 2,0 = sehr risikoabhängig). Bei Forex-Paaren entfällt die Säule unter 0,5 (z. B. AUDNZD,
EURGBP, EURCHF); bei Futures und beim DXY (−0,5) ist sie immer aktiv.

**Daten-Überraschungen** – MT5-Surprise-Index (Actual vs. Forecast als z-Score, 90 Tage, Halbwertszeit 30 Tage),
Basis minus Quote.

**Positionierung (COT)** – Netto-Position der Non-Commercials in % des Open Interest. Score = Veränderung über
4 Wochen im Verhältnis zur üblichen Schwankung. Über 90. bzw. unter 10. Perzentil (3 Jahre) = überfüllt, Score wird halbiert.

**Futures** – Währungs-Futures (CHF, EUR, AUD, NZD, CAD, GBP, JPY) werden wie das Paar gegen den USD bewertet
(z. B. CHF-Future = CHF/USD), nur die COT-Säule nutzt die Positionierung im Future selbst. DXY: Fed-Zinszyklus relativ
zum Durchschnitt der anderen Notenbanken und US-Surprise-Index; COT aus dem Dollar-Index-Future.

**Event-Risiko** – fließt nicht in den Score. Hoch = Notenbank-Sitzung in ≤ 7 Tagen oder hohes Interventionsrisiko.
JPY-Interventionsrisiko ist eine Faustregel (USDJPY ≥ {CONFIG['intervention_level_mid']:.0f} oder ≥ {CONFIG['intervention_move_mid_pct']:.0f}% Anstieg in 20 Tagen).

**Gesamtfazit** – gewichteter Mittelwert: Zinsen {w['zinsen']:.0%}, Risiko {w['risiko']:.0%}, Überraschungen {w['surprise']:.0%},
COT {w['cot']:.0%} (fehlende Säulen → Rest hochgerechnet). LONG/SHORT nur, wenn mindestens 2 Säulen dafür und keine klar dagegen
sind, sonst GEMISCHT. |Score| < {CONFIG['neutral_band']} = NEUTRAL.

⚠️ **Alle Gewichte und Schwellen sind unvalidierte Startwerte.** Ob das Fazit die nächsten 1–4 Wochen tatsächlich
vorhersagt, ist noch nicht getestet. Kein Finanzrat.
""")


def render_macro_dashboard(deps: dict) -> None:
    """deps-Schluessel:
    load_calendar() -> DataFrame MT5-Kalender
    surprise_index(cur) -> (float|None, n)
    surprise_history(days) -> DataFrame (Index Datum, Spalten Waehrungen)
    exported_at() -> str|None
    cot_market(cur) -> CFTC-Marktname|None
    expected_rate_change(cur) -> float|None   (optional)
    render_ki(label, symbol)                   (optional)
    tz -> Anzeige-Zeitzone
    """
    st.markdown("## 🌐 Makro-Dashboard")
    st.caption("Fundamentale Richtung pro Währungspaar für die nächsten 1–4 Wochen. "
               "Unvalidierte Heuristik, kein Finanzrat – Details im Tab „Methodik“.")

    with st.spinner("Lade Makro-, Risiko- und COT-Daten …"):
        data = compute_all(deps)

    exported = deps["exported_at"]()
    cal = data["calendar"]
    stale_msg = ""
    if cal is not None and not cal.empty:
        last_actual = cal[cal["actual"].notna()]["time_utc"].max()
        if pd.notna(last_actual) and (data["now"] - last_actual).days > CONFIG["mt5_stale_days"]:
            stale_msg = (f"⚠️ MT5-Kalender veraltet: letzter Actual-Wert vom {last_actual:%d.%m.%Y}. "
                         "MT5 mit dem Export-EA laufen lassen und synchronisieren, sonst fehlen neue Daten.")
    st.caption(f"Stand MT5-Export: {exported or 'unbekannt'}")
    if stale_msg:
        st.warning(stale_msg)

    mode = st.radio("Ansicht", ["Forex-Paare", "Futures"], horizontal=True, key="macro_dash_mode")
    if mode == "Forex-Paare":
        choices, default, label = PAIRS, "GBPJPY", "Paar"
        st.markdown("#### Übersicht Forex-Paare (alle 28)")
    else:
        choices, default, label = FUTURE_LIST, "EUR", "Future"
        st.markdown("#### Übersicht Futures")
        st.caption("Währungs-Futures notieren in USD (z. B. CHF-Future = CHF gegen USD, LONG = CHF steigt). "
                   "DXY = US-Dollar gegen einen Währungskorb.")
    _render_overview(data, choices, label)

    name = st.selectbox("Markt im Detail", choices, index=choices.index(default), key=f"macro_dash_pick_{label}")
    r = data["results"][name]
    pair = r.get("code", name)
    _render_verdict(name, r, pair)

    tabs = st.tabs(["Zinsen & Notenbanken", "Risikostimmung", "Daten-Überraschungen",
                    "Positionierung (COT)", "Kalender 4 Wochen", "KI-Einschätzung", "Methodik"])
    with tabs[0]:
        _tab_rates(pair, r)
    with tabs[1]:
        _tab_risk(data, pair, r)
    with tabs[2]:
        _tab_surprise(data, pair, r)
    with tabs[3]:
        _tab_cot(pair, r)
    with tabs[4]:
        _tab_calendar(pair, r, deps.get("tz", "Europe/Berlin"))
    with tabs[5]:
        render_ki = deps.get("render_ki")
        if render_ki is None:
            st.caption("KI-Analyse nicht verfügbar.")
        else:
            st.caption("Qualitative Einschätzung inkl. Geopolitik über Gemini + Google-Suche. Fließt nicht in den Score.")
            render_ki(name, FUTURE_SYMBOLS.get(name, f"{name}=X"))
    with tabs[6]:
        _tab_method()
