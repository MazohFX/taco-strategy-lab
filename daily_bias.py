"""Daily & Weekly Bias: Richtung fuer Indizes, Gold und Forex-Futures.

Baustein fuer das Makro-Dashboard (macro_dashboard.py), nach dem Vorbild der
"Vault Engine" (payoutvault.one), aber mit festen, nachvollziehbaren Regeln statt
eines frei formulierenden Sprachmodells.

Maerkte:   Indizes & Gold: NQ, ES, GC
           Forex-Futures:  6E, 6B, 6J, 6A, 6N, 6C, 6S (Waehrung gegen USD) und DXY
Horizont:  Daily  = heutige Session      (Kipp-Marke: Vortageshoch/-tief)
           Weekly = laufende/naechste Woche (Kipp-Marke: Vorwochenhoch/-tief)

Komponenten (je -1..+1):
  1. Makro (1-4 Wochen)  Fazit-Score der Futures-Ansicht des Dashboards
                         (Zinsen/Notenbanken, Risikostimmung, Daten-Ueberraschungen, COT)
  2. Struktur            Kurs vs. Vortages-/Vorwochenrange, vs. Durchschnitt, Momentum (in ATR)
  3. Tape (Intermarket)  VIX, US-10J-Rendite, Dollar-Index, Oel, Gold, S&P 500, Bitcoin
  4. Schlagzeilen        Stichwort-Heuristik ueber RSS-Schlagzeilen der letzten 24 h

Dazu: News-Guard (Termine), historische Quote fuer die heutige Ausgangslage,
Backtest Struktur + Tape, im Weekly-Modus die Seasonality der Kalenderwoche
(nur Info, NICHT im Score).

ALLE Gewichte und Schwellen sind UNVALIDIERTE STARTWERTE.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

HORIZONS = {
    "Daily": {
        "weights": {"makro": 0.30, "struktur": 0.35, "tape": 0.25, "news": 0.10},
        "sma": 20, "atr": 14, "mom": 5, "years": 5,
        "tape_lag": 1, "tape_scale_mult": 1.0,
        "prev": "Vortag", "prev_gen": "Vortages", "avg_txt": "20-Tage-Schnitt", "mom_txt": "5-Tage-Momentum",
        "next_txt": "nächster Tag", "unit_txt": "Tage",
    },
    "Weekly": {
        "weights": {"makro": 0.45, "struktur": 0.35, "tape": 0.15, "news": 0.05},
        "sma": 10, "atr": 10, "mom": 4, "years": 15,
        "tape_lag": 5, "tape_scale_mult": 2.2,
        "prev": "Vorwoche", "prev_gen": "Vorwochen", "avg_txt": "10-Wochen-Schnitt", "mom_txt": "4-Wochen-Momentum",
        "next_txt": "nächste Woche", "unit_txt": "Wochen",
    },
}

BIAS_CONFIG = {
    "label_min": 3,            # |Score| (von 10) ab dem LONG/SHORT gilt
    "high_min": 6,             # |Score| fuer Sicherheit HOCH
    "support_min": 0.15,       # Komponente zaehlt als "dafuer"
    "oppose_min": 0.30,        # Komponente zaehlt als "klar dagegen"
    "guard_exit_min": 60,      # Exits planen x Minuten vor dem Termin
    "guard_flat_min": 15,      # keine neuen Trades x Minuten vor dem Termin
    "guard_history_n": 8,      # so viele fruehere Termine fuer die Reaktion
    "backtest_min_abs": 0.30,  # |Struktur+Tape-Score| ab dem ein historisches Signal zaehlt
    "fetch_years": 15,
    # Rueckwaertskompatibel (Daily)
    "weights": HORIZONS["Daily"]["weights"], "sma_days": 20, "atr_days": 14, "mom_days": 5, "history_years": 5,
}

# currency: Waehrung des Futures (fuer Termine/Notenbank); macro: Schluessel in data["results"]
ASSETS = {
    "NQ": {"name": "Nasdaq 100 Futures", "symbol": "NQ=F", "macro": "Nasdaq 100", "decimals": 2, "currency": "USD", "group": "idx"},
    "ES": {"name": "S&P 500 Futures", "symbol": "ES=F", "macro": "S&P 500", "decimals": 2, "currency": "USD", "group": "idx"},
    "GC": {"name": "Gold Futures", "symbol": "GC=F", "macro": "Gold", "decimals": 1, "currency": "USD", "group": "idx"},
    "6E": {"name": "Euro (EUR/USD)", "symbol": "6E=F", "macro": "EUR", "decimals": 5, "currency": "EUR", "group": "fx"},
    "6B": {"name": "Pfund (GBP/USD)", "symbol": "6B=F", "macro": "GBP", "decimals": 4, "currency": "GBP", "group": "fx"},
    "6J": {"name": "Yen (JPY/USD)", "symbol": "6J=F", "macro": "JPY", "decimals": 7, "currency": "JPY", "group": "fx"},
    "6A": {"name": "Aussie (AUD/USD)", "symbol": "6A=F", "macro": "AUD", "decimals": 5, "currency": "AUD", "group": "fx"},
    "6N": {"name": "Kiwi (NZD/USD)", "symbol": "6N=F", "macro": "NZD", "decimals": 5, "currency": "NZD", "group": "fx"},
    "6C": {"name": "Loonie (CAD/USD)", "symbol": "6C=F", "macro": "CAD", "decimals": 5, "currency": "CAD", "group": "fx"},
    "6S": {"name": "Franken (CHF/USD)", "symbol": "6S=F", "macro": "CHF", "decimals": 4, "currency": "CHF", "group": "fx"},
    "DXY": {"name": "US-Dollar-Index", "symbol": "DX-Y.NYB", "macro": "DXY", "decimals": 2, "currency": "USD", "group": "fx"},
}
GROUPS = {"Indizes & Gold": ["NQ", "ES", "GC"],
          "Forex-Futures": ["6E", "6B", "6J", "6A", "6N", "6C", "6S", "DXY"]}

# Tape-Treiber: Yahoo-Symbol, Einheit der Veraenderung, Skala fuer 1 Tag (tanh ~0.76 bei 1 Skala)
DRIVERS = {
    "VIX": {"symbol": "^VIX", "unit": "pct", "scale": 6.0, "label": "VIX"},
    "US10Y": {"symbol": "^TNX", "unit": "bp", "scale": 5.0, "label": "US-10J-Rendite"},
    "DXY": {"symbol": "DX-Y.NYB", "unit": "pct", "scale": 0.4, "label": "Dollar-Index"},
    "OIL": {"symbol": "CL=F", "unit": "pct", "scale": 2.0, "label": "Öl (WTI)"},
    "GOLD": {"symbol": "GC=F", "unit": "pct", "scale": 1.0, "label": "Gold"},
    "SPX": {"symbol": "^GSPC", "unit": "pct", "scale": 1.0, "label": "S&P 500"},
    "BTC": {"symbol": "BTC-USD", "unit": "pct", "scale": 3.0, "label": "Bitcoin"},
}
# Wie ein steigender Treiber auf den Markt wirkt (unvalidierte Faustregeln).
# Waehrungs-Futures notieren in USD: LONG 6J = Yen steigt gegen den Dollar.
SENSITIVITY = {
    "NQ": {"VIX": -1.0, "US10Y": -1.0, "DXY": -0.4, "BTC": +0.3},
    "ES": {"VIX": -1.0, "US10Y": -0.6, "DXY": -0.4, "BTC": +0.2},
    "GC": {"VIX": +0.4, "US10Y": -1.0, "DXY": -1.0, "OIL": +0.3},
    "6E": {"US10Y": -0.8, "VIX": -0.2, "SPX": +0.2},
    "6B": {"US10Y": -0.8, "VIX": -0.4, "SPX": +0.3},
    "6J": {"US10Y": -1.0, "VIX": +0.8, "SPX": -0.4},
    "6A": {"US10Y": -0.5, "VIX": -0.8, "SPX": +0.6, "GOLD": +0.4},
    "6N": {"US10Y": -0.5, "VIX": -0.8, "SPX": +0.6, "GOLD": +0.2},
    "6C": {"OIL": +0.8, "US10Y": -0.4, "VIX": -0.4, "SPX": +0.3},
    "6S": {"US10Y": -0.6, "VIX": +0.6, "GOLD": +0.3},
    "DXY": {"US10Y": +1.0, "VIX": +0.4, "SPX": -0.2},
}

# Schlagzeilen-Regeln: (Regex, Wirkung auf NQ/ES/GC, Label). Wirkung auf Forex-Futures je Label unten.
NEWS_RULES = [
    (r"cease-?fire|truce|peace (deal|talks|agreement)|de-?escalat", {"NQ": 1, "ES": 1, "GC": -1}, "Entspannung"),
    (r"escalat|missile|airstrike|air strike|invasion|attack|war\b|troops|drone strike|military", {"NQ": -1, "ES": -1, "GC": 1}, "Geopolitik"),
    (r"sanction|tension|conflict|nuclear", {"NQ": -1, "ES": -1, "GC": 1}, "Geopolitik"),
    (r"rate cut|cuts? rates|dovish|easing|pause hikes", {"NQ": 1, "ES": 1, "GC": 1}, "Fed dovish"),
    (r"rate hike|raises? rates|hawkish|higher for longer", {"NQ": -1, "ES": -1, "GC": -1}, "Fed hawkish"),
    (r"tariff|trade war|export (ban|curb|control)", {"NQ": -1, "ES": -1, "GC": 1}, "Zölle/Handel"),
    (r"(hot|hotter|sticky|accelerat\w*|higher[- ]than[- ]expected) inflation|inflation (jumps|surges|accelerat\w*|heats)", {"NQ": -1, "ES": -1, "GC": 0}, "Inflation heiß"),
    (r"(cool\w*|soft\w*|lower[- ]than[- ]expected) inflation|inflation (cools|eases|slows)", {"NQ": 1, "ES": 1, "GC": 0}, "Inflation kühl"),
    (r"outflow", {"NQ": -1, "ES": -1, "GC": 0}, "Fondsabflüsse"),
    (r"inflow", {"NQ": 1, "ES": 1, "GC": 0}, "Fondszuflüsse"),
    (r"sell-?off|plunge|slump|rout|tumble|crash", {"NQ": -1, "ES": -1, "GC": 0}, "Ausverkauf"),
    (r"record high|all-time high|rall(y|ies)|surge", {"NQ": 1, "ES": 1, "GC": 0}, "Rally"),
    (r"safe[- ]haven|gold (demand|buying)|central banks? (buy|buying) gold", {"NQ": 0, "ES": 0, "GC": 1}, "Sicherer Hafen"),
    (r"recession|layoffs|jobless claims (rise|jump)|weak (jobs|payrolls)", {"NQ": -1, "ES": -1, "GC": 1}, "Konjunktur schwach"),
    (r"shutdown|debt ceiling|downgrade", {"NQ": -1, "ES": -1, "GC": 1}, "US-Haushalt"),
]
_RISK_OFF = {"6J": 1, "6S": 1, "6A": -1, "6N": -1, "6C": -1, "DXY": 1}
_RISK_ON = {k: -v for k, v in _RISK_OFF.items()}
_USD_DOWN = {"6E": 1, "6B": 1, "6J": 1, "6A": 1, "6N": 1, "6C": 1, "6S": 1, "DXY": -1}
_USD_UP = {k: -v for k, v in _USD_DOWN.items()}
FX_LABEL_EFFECT = {
    "Entspannung": _RISK_ON, "Rally": _RISK_ON, "Fondszuflüsse": {},
    "Geopolitik": _RISK_OFF, "Sicherer Hafen": {"6J": 1, "6S": 1}, "Ausverkauf": _RISK_OFF,
    "Konjunktur schwach": {**_RISK_OFF, "DXY": 0}, "Fondsabflüsse": {},
    "Fed dovish": _USD_DOWN, "Fed hawkish": _USD_UP,
    "Inflation heiß": _USD_UP, "Inflation kühl": _USD_DOWN,
    "Zölle/Handel": {"6A": -1, "6N": -1, "6C": -1, "6J": 1, "6S": 1},
    "US-Haushalt": {**_USD_DOWN, "6A": 0, "6N": 0, "6C": 0},
}
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC,%5ENDX,GC%3DF,EURUSD%3DX,JPY%3DX&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=(stocks%20OR%20fed%20OR%20gold%20OR%20dollar%20OR%20geopolitics)%20when:1d&hl=en-US&gl=US&ceid=US:en",
]

LABEL_COL = {"LONG": "#22c55e", "SHORT": "#ef4444", "NEUTRAL": "#eab308"}
CONF_ORDER = ["NIEDRIG", "MITTEL", "HOCH"]
COMP_LABELS = {"makro": "Makro (1–4 Wo.)", "struktur": "Struktur", "tape": "Tape", "news": "Schlagzeilen"}


def _tanh(x: float, scale: float) -> float:
    return float(np.tanh(x / scale)) if scale else 0.0


def _fmt(v: float, dec: int) -> str:
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _cfg(horizon: str) -> dict:
    return HORIZONS[horizon]


# ── reine Berechnung (testbar ohne Streamlit/Netzwerk) ───────────────────────
def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Tageskerzen -> Wochenkerzen (Label = Freitag der Woche)."""
    if df is None or df.empty:
        return df
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return w


def split_running(df: pd.DataFrame, now_ny: pd.Timestamp) -> tuple[pd.DataFrame, float | None]:
    """Trennt die laufende Kerze ab. Rueckgabe: (abgeschlossene Kerzen, aktueller Kurs).
    Funktioniert fuer Tageskerzen (Label = Handelstag) und Wochenkerzen (Label = Freitag):
    Eine Kerze mit heutigem oder spaeterem Datum gilt als laufend, solange es vor 17:00 NY ist."""
    if df is None or df.empty:
        return df, None
    last_day = pd.Timestamp(df.index[-1]).normalize()
    today = now_ny.normalize().tz_localize(None) if now_ny.tzinfo else now_ny.normalize()
    hour = now_ny.hour + now_ny.minute / 60
    running = (last_day > today) or (last_day == today and hour < 17)
    if running and len(df) > 1:
        return df.iloc[:-1], float(df["Close"].iloc[-1])
    return df, float(df["Close"].iloc[-1])


def structure_state(done: pd.DataFrame, price: float, horizon: str = "Daily") -> dict | None:
    """Struktur-Score (-1..+1) aus abgeschlossenen Kerzen und aktuellem Kurs."""
    c = _cfg(horizon)
    if done is None or len(done) < max(c["sma"], c["atr"]) + c["mom"] + 1:
        return None
    a = float(atr(done, c["atr"]).iloc[-1])
    if not a or math.isnan(a):
        return None
    pdh, pdl = float(done["High"].iloc[-1]), float(done["Low"].iloc[-1])
    pdc = float(done["Close"].iloc[-1])
    sma = float(done["Close"].iloc[-c["sma"]:].mean())
    ref = float(done["Close"].iloc[-c["mom"]])
    pg = c["prev_gen"]
    if price > pdh:
        pos, pos_txt = 1.0, f"über dem {pg}hoch"
    elif price < pdl:
        pos, pos_txt = -1.0, f"unter dem {pg}tief"
    else:
        half = (pdh - pdl) / 2 or a
        pos = 0.5 * (price - (pdh + pdl) / 2) / half
        pos_txt = f"in der {pg}range (" + ("obere" if pos > 0 else "untere") + " Hälfte)"
    sma_s = _tanh((price - sma) / a, 1.5)
    mom_s = _tanh((price - ref) / a, 2.0)
    score = float(np.mean([pos, sma_s, mom_s]))
    dist_atr = (price - sma) / a
    sma_txt = (f"{'deutlich ' if abs(dist_atr) > 1.5 else ''}{'über' if dist_atr > 0 else 'unter'} dem "
               f"{c['avg_txt']} ({dist_atr:+.1f} ATR)")
    mom_txt = f"{c['mom_txt']} {'positiv' if price > ref else 'negativ'} ({(price - ref) / a:+.1f} ATR)"
    return {"score": score, "pos": pos, "sma": sma_s, "mom": mom_s, "pdh": pdh, "pdl": pdl, "pdc": pdc,
            "sma_level": sma, "atr": a, "price": price,
            "reason": f"Kurs {pos_txt}, {sma_txt}, {mom_txt}", "parts": [pos_txt, sma_txt, mom_txt]}


def driver_change(series: pd.Series | None, unit: str, lag: int = 1) -> float | None:
    if series is None:
        return None
    s = series.dropna()
    if len(s) < lag + 1:
        return None
    a, b = float(s.iloc[-1 - lag]), float(s.iloc[-1])
    if unit == "bp":
        return (b - a) * 100.0
    return (b / a - 1) * 100.0 if a else None


def tape_score(asset: str, changes: dict[str, float | None], horizon: str = "Daily") -> dict | None:
    sens = SENSITIVITY[asset]
    mult = _cfg(horizon)["tape_scale_mult"]
    num = den = 0.0
    items = []
    for k, w in sens.items():
        ch = changes.get(k)
        if ch is None or (isinstance(ch, float) and math.isnan(ch)):
            continue
        eff = w * _tanh(ch, DRIVERS[k]["scale"] * mult)
        num += eff
        den += abs(w)
        items.append((k, ch, eff))
    if den == 0:
        return None
    score = float(np.clip(num / den, -1, 1))
    items.sort(key=lambda x: -abs(x[2]))
    parts = []
    for k, ch, eff in items[:3]:
        unit = "Bp" if DRIVERS[k]["unit"] == "bp" else "%"
        word = "stützt" if eff > 0.05 else "belastet" if eff < -0.05 else "neutral"
        parts.append(f"{DRIVERS[k]['label']} {ch:+.1f} {unit} ({word})")
    since = "Seit Vortagesschluss" if horizon == "Daily" else "Letzte 5 Handelstage"
    return {"score": score, "items": items, "reason": f"{since}: " + ", ".join(parts)}


def tag_headline(title: str) -> dict:
    t = title.lower()
    effect = {k: 0 for k in ASSETS}
    labels = []
    for pattern, eff, label in NEWS_RULES:
        if not re.search(pattern, t):
            continue
        if label == "Geopolitik" and ("Geopolitik" in labels or "Entspannung" in labels):
            continue  # doppelt bzw. "ceasefire after attack" -> Entspannung zaehlt
        for k, v in eff.items():
            effect[k] += v
        for k, v in FX_LABEL_EFFECT.get(label, {}).items():
            effect[k] += v
        labels.append(label)
    return {"effect": {k: int(np.sign(v)) for k, v in effect.items()}, "labels": labels}


def news_scores(headlines: list[dict]) -> dict:
    """headlines: [{'title':..., 'source':..., 'time':...}] -> Score je Markt + getaggte Liste."""
    tagged = []
    tot = {k: 0 for k in ASSETS}
    for h in headlines:
        tg = tag_headline(h.get("title", ""))
        if not tg["labels"]:
            continue
        tagged.append({**h, **tg})
        for k in tot:
            tot[k] += tg["effect"][k]
    out = {}
    for k in tot:
        relevant = [x for x in tagged if x["effect"][k] != 0]
        if not relevant:
            out[k] = None
            continue
        top: dict[str, int] = {}
        for x in relevant:
            for lb in x["labels"]:
                top[lb] = top.get(lb, 0) + 1
        topic = ", ".join(f"{lb} ({n}×)" for lb, n in sorted(top.items(), key=lambda kv: -kv[1])[:3])
        out[k] = {"score": _tanh(tot[k], 3.0), "n": len(relevant),
                  "reason": f"{len(relevant)} relevante Schlagzeile{'n' if len(relevant) != 1 else ''} (24 h): {topic}"}
    return {"scores": out, "tagged": tagged}


def combine(comps: dict[str, dict | None], horizon: str = "Daily") -> dict:
    """comps: {'makro': {'score','reason'} | None, ...} -> Label, Score10, Sicherheit, Gruende."""
    c = BIAS_CONFIG
    avail = {k: v for k, v in comps.items() if v is not None and v.get("score") is not None}
    if not avail:
        return {"score": None, "score10": None, "label": "n/a", "confidence": "NIEDRIG",
                "reasons": [], "quality": f"0/{len(comps)}", "contrib": {}, "supporters": [], "opposers": []}
    w = _cfg(horizon)["weights"]
    tw = sum(w[k] for k in avail) or 1.0
    contrib = {k: w[k] * v["score"] / tw for k, v in avail.items()}
    score = float(np.clip(sum(contrib.values()), -1, 1))
    s10 = int(round(score * 10))
    label = "LONG" if s10 >= c["label_min"] else "SHORT" if s10 <= -c["label_min"] else "NEUTRAL"
    d = np.sign(score) if label != "NEUTRAL" else 0
    supp = [k for k, v in avail.items() if d and np.sign(v["score"]) == d and abs(v["score"]) >= c["support_min"]]
    opp = [k for k, v in avail.items() if d and np.sign(v["score"]) == -d and abs(v["score"]) >= c["oppose_min"]]
    if label == "NEUTRAL":
        conf = "MITTEL" if all(abs(v["score"]) < c["oppose_min"] for v in avail.values()) else "NIEDRIG"
    elif abs(s10) >= c["high_min"] and not opp and len(supp) >= 3:
        conf = "HOCH"
    elif not opp or (len(opp) == 1 and len(supp) >= 2):
        conf = "MITTEL"
    else:
        conf = "NIEDRIG"
    order = sorted(avail, key=lambda k: -abs(contrib[k]))
    if d:
        order = [k for k in order if np.sign(avail[k]["score"]) == d] + [k for k in order if np.sign(avail[k]["score"]) != d]
    reasons = [(k, avail[k]["reason"]) for k in order[:3]]
    return {"score": score, "score10": s10, "label": label, "confidence": conf, "reasons": reasons,
            "supporters": supp, "opposers": opp, "quality": f"{len(avail)}/{len(comps)}", "contrib": contrib}


def downgrade(conf: str, steps: int = 1) -> str:
    return CONF_ORDER[max(0, CONF_ORDER.index(conf) - steps)]


def flip_level(label: str, st_: dict | None, dec: int, horizon: str = "Daily") -> str:
    if st_ is None:
        return "–"
    pg = _cfg(horizon)["prev_gen"]
    pdh, pdl, price = st_["pdh"], st_["pdl"], st_["price"]
    if label == "LONG":
        lvl, txt = (pdh, f"Rückfall unter das {pg}hoch") if price > pdh else (pdl, f"Bruch unter das {pg}tief")
        return f"{txt} {_fmt(lvl, dec)}"
    if label == "SHORT":
        lvl, txt = (pdl, f"Rückkehr über das {pg}tief") if price < pdl else (pdh, f"Bruch über das {pg}hoch")
        return f"{txt} {_fmt(lvl, dec)}"
    return f"Long über {_fmt(pdh, dec)} ({pg}hoch), Short unter {_fmt(pdl, dec)} ({pg}tief)"


def horizon_end(now_utc: pd.Timestamp, horizon: str) -> pd.Timestamp:
    """Ende des Bewertungszeitraums in New-York-Zeit (17:00 NY = Futures-Settlement)."""
    ny = now_utc.tz_localize("UTC").tz_convert("America/New_York")
    end = ny.normalize() + pd.Timedelta(hours=17)
    if horizon == "Daily":
        if ny >= end or ny.weekday() >= 5:
            end = (ny.normalize() + pd.offsets.BDay(1)) + pd.Timedelta(hours=17)
        return end
    days_to_fri = (4 - ny.weekday()) % 7
    end = ny.normalize() + pd.Timedelta(days=days_to_fri, hours=17)
    if ny >= end:
        end = end + pd.Timedelta(days=7)
    return end


def relevant_events(calendar: pd.DataFrame | None, now_utc: pd.Timestamp, currencies: list[str],
                    horizon: str = "Daily") -> pd.DataFrame:
    """High-Impact-Termine der Waehrungen bis zum Ende des Horizonts (ab 2 h in der Vergangenheit)."""
    if calendar is None or calendar.empty:
        return pd.DataFrame()
    end_utc = horizon_end(now_utc, horizon).tz_convert("UTC").tz_localize(None)
    start = now_utc - pd.Timedelta(hours=2)
    ev = calendar[calendar["currency"].isin(currencies) & (calendar["importance"] == "High")
                  & (calendar["time_utc"] >= start) & (calendar["time_utc"] <= end_utc)].copy()
    ev["notenbank"] = ev["event_code"].astype(str).str.contains("interest-rate-decision|deposit-rate-decision")
    return ev.sort_values("time_utc")


def todays_us_events(calendar: pd.DataFrame | None, now_utc: pd.Timestamp) -> pd.DataFrame:
    return relevant_events(calendar, now_utc, ["USD"], "Daily")


def event_reactions(calendar: pd.DataFrame, event_code: str, ohlc: dict[str, pd.DataFrame],
                    before: pd.Timestamp, n: int) -> pd.DataFrame:
    """Tagesbewegung der Maerkte an den letzten n Terminen dieses Events (Close-zu-Close in %,
    Spanne in ATR)."""
    past = calendar[(calendar["event_code"] == event_code) & (calendar["time_utc"] < before)
                    & calendar["actual"].notna()].sort_values("time_utc").tail(n)
    rows = []
    for _, r in past.iterrows():
        day = (r["time_utc"].tz_localize("UTC").tz_convert("America/New_York")).normalize().tz_localize(None)
        row = {"Datum": day.strftime("%d.%m.%Y"), "Actual": r["actual"], "Forecast": r.get("forecast")}
        for k, df in ohlc.items():
            if df is None or df.empty or day not in df.index:
                row[k] = None
                continue
            i = df.index.get_loc(day)
            if i < 15:
                row[k] = None
                continue
            a = atr(df.iloc[: i], 14).iloc[-1]
            ret = (df["Close"].iloc[i] / df["Close"].iloc[i - 1] - 1) * 100
            rng = (df["High"].iloc[i] - df["Low"].iloc[i]) / a if a else np.nan
            row[k] = f"{ret:+.2f} % · {rng:.1f} ATR"
        rows.append(row)
    return pd.DataFrame(rows)


def _hist_structure(df: pd.DataFrame, horizon: str = "Daily") -> pd.DataFrame:
    """Struktur-Score je Kerze t (Schluss t gegen Range, Schnitt und ATR bis t-1), vektorisiert."""
    c = _cfg(horizon)
    close, hi, lo = df["Close"], df["High"].shift(1), df["Low"].shift(1)
    a = atr(df, c["atr"]).shift(1)
    half = ((hi - lo) / 2).replace(0, np.nan)
    pos = np.where(close > hi, 1.0, np.where(close < lo, -1.0, 0.5 * (close - (hi + lo) / 2) / half))
    sma = close.rolling(c["sma"]).mean().shift(1)
    sma_s = np.tanh((close - sma) / a / 1.5)
    mom_s = np.tanh((close - close.shift(c["mom"])) / a / 2.0)
    out = pd.DataFrame({"pos": pos, "sma": sma_s, "mom": mom_s}, index=df.index)
    out["struktur"] = out[["pos", "sma", "mom"]].mean(axis=1)
    out["above_sma"] = close > sma
    pg = c["prev_gen"]
    out["pos_bucket"] = np.where(close > hi, f"über {pg}hoch", np.where(close < lo, f"unter {pg}tief", f"in {pg}range"))
    out["atr"] = a
    return out


def _hist_tape(asset: str, index: pd.DatetimeIndex, drivers: dict[str, pd.Series], horizon: str = "Daily") -> pd.Series:
    mult = _cfg(horizon)["tape_scale_mult"]
    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    for k, w in SENSITIVITY[asset].items():
        s = drivers.get(k)
        if s is None or s.dropna().empty:
            continue
        s = s.dropna()
        if horizon == "Weekly":
            s = s.resample("W-FRI").last().dropna()
        ch = s.diff() * 100 if DRIVERS[k]["unit"] == "bp" else s.pct_change() * 100
        ch = ch.reindex(index)
        eff = w * np.tanh(ch / (DRIVERS[k]["scale"] * mult))
        num = num.add(eff.fillna(0))
        den = den.add(ch.notna() * abs(w))
    return (num / den.replace(0, np.nan)).clip(-1, 1)


def _binom_p(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p).pvalue)
    except Exception:  # Normal-Approximation
        z = (k - n * p) / math.sqrt(n * p * (1 - p))
        return float(math.erfc(abs(z) / math.sqrt(2)))


def history_stats(asset: str, done: pd.DataFrame, drivers: dict[str, pd.Series],
                  today_state: dict | None, horizon: str = "Daily") -> dict | None:
    """1) Quote fuer die heutige Ausgangslage (Lage zur Vorperioden-Range x Schnitt).
    2) Backtest Struktur+Tape: Vorzeichen an Kerze t -> Richtung Kerze t+1."""
    c = _cfg(horizon)
    min_len = 120 if horizon == "Daily" else 80
    if done is None or len(done) < min_len:
        return None
    if horizon == "Daily":
        done = done[done.index >= done.index[-1] - pd.DateOffset(years=c["years"])]
    w = c["weights"]
    h = _hist_structure(done, horizon)
    h["tape"] = _hist_tape(asset, done.index, drivers, horizon)
    tech_w = w["struktur"] + w["tape"]
    h["tech"] = np.where(h["tape"].notna(), (w["struktur"] * h["struktur"] + w["tape"] * h["tape"]) / tech_w, h["struktur"])
    nxt = done["Close"].shift(-1) / done["Close"] - 1
    h["next_up"] = nxt > 0
    h["next_ret"] = nxt * 100
    h["next_range"] = (done["High"].shift(-1) - done["Low"].shift(-1))
    h = h.dropna(subset=["struktur", "next_ret"])
    if h.empty:
        return None
    base_up = float(h["next_up"].mean())
    out = {"n_days": len(h), "base_up": base_up, "start": h.index[0], "end": h.index[-1]}
    if today_state is not None:
        pg = c["prev_gen"]
        bucket = (f"über {pg}hoch" if today_state["price"] > today_state["pdh"] else
                  f"unter {pg}tief" if today_state["price"] < today_state["pdl"] else f"in {pg}range")
        above = today_state["price"] > today_state["sma_level"]
        sub = h[(h["pos_bucket"] == bucket) & (h["above_sma"] == above)]
        if len(sub):
            k = int(sub["next_up"].sum())
            out["odds"] = {"bucket": bucket, "above_sma": above, "n": len(sub), "up": k / len(sub),
                           "avg_ret": float(sub["next_ret"].mean()), "avg_range": float(sub["next_range"].mean()),
                           "p": _binom_p(k, len(sub), base_up)}
    sig = h[h["tech"].abs() >= BIAS_CONFIG["backtest_min_abs"]]
    if len(sig):
        hit = (np.sign(sig["tech"]) == np.sign(sig["next_ret"])) & (sig["next_ret"] != 0)
        k = int(hit.sum())
        longs, shorts = sig[sig["tech"] > 0], sig[sig["tech"] < 0]
        out["backtest"] = {"n": len(sig), "hit": k / len(sig), "p": _binom_p(k, len(sig), 0.5),
                           "long_n": len(longs), "long_hit": float((longs["next_ret"] > 0).mean()) if len(longs) else None,
                           "short_n": len(shorts), "short_hit": float((shorts["next_ret"] < 0).mean()) if len(shorts) else None,
                           "avg_ret_signal": float((np.sign(sig["tech"]) * sig["next_ret"]).mean())}
    return out


def seasonality_week(daily: pd.DataFrame | None, now_ny: pd.Timestamp) -> dict | None:
    """Verhalten in derselben Kalenderwoche frueherer Jahre (nur Info, nicht im Score).
    Zielwoche = laufende Woche, am Wochenende bzw. nach Freitag 17:00 NY die naechste."""
    if daily is None or daily.empty:
        return None
    wk = to_weekly(daily)
    if len(wk) < 60:
        return None
    end = horizon_end(now_ny.tz_convert("UTC").tz_localize(None), "Weekly")
    target = int(end.isocalendar().week)
    year_now = int(end.isocalendar().year)
    ret = wk["Close"].pct_change() * 100
    iso = wk.index.isocalendar()
    mask = (iso["week"].astype(int) == target) & (iso["year"].astype(int) < year_now)
    r = ret[mask.values].dropna()
    if len(r) < 3:
        return None
    return {"week": target, "n": len(r), "up": float((r > 0).mean()), "avg": float(r.mean()),
            "median": float(r.median()), "p": _binom_p(int((r > 0).sum()), len(r), 0.5),
            "years": {int(i.isocalendar().year): round(float(v), 2) for i, v in r.items()}}


def asset_bias(asset: str, ohlc: pd.DataFrame | None, now_ny: pd.Timestamp, macro_score: float | None,
               macro_reason: str, changes: dict, news: dict | None, events: pd.DataFrame,
               vix_regime: str | None = None, horizon: str = "Daily", intervention: dict | None = None) -> dict:
    spec = ASSETS[asset]
    bars = ohlc
    if horizon == "Weekly" and ohlc is not None and not ohlc.empty:
        bars = to_weekly(ohlc)
    done, price = split_running(bars, now_ny) if bars is not None and not bars.empty else (None, None)
    st_ = structure_state(done, price, horizon) if price is not None else None
    tp = tape_score(asset, changes, horizon)
    comps = {
        "makro": None if macro_score is None else {"score": float(macro_score), "reason": macro_reason},
        "struktur": None if st_ is None else {"score": st_["score"], "reason": st_["reason"]},
        "tape": tp,
        "news": news,
    }
    res = combine(comps, horizon)
    notes = []
    ev = events
    if ev is not None and not ev.empty and "currency" in ev:
        ev = ev[ev["currency"].isin({"USD", spec["currency"]})]
    if ev is not None and not ev.empty:
        cb = ev["event_code"].astype(str).str.contains("interest-rate-decision|deposit-rate-decision")
        when = "heute" if horizon == "Daily" else "diese Woche"
        if cb.any():
            names = sorted(set(ev.loc[cb, "currency"])) if "currency" in ev else ["USD"]
            res["confidence"] = "NIEDRIG" if horizon == "Daily" else downgrade(res["confidence"])
            notes.append(f"Notenbank-Entscheid {when} ({', '.join(names)})")
        else:
            if horizon == "Daily":
                res["confidence"] = downgrade(res["confidence"])
            elif len(ev) >= 4:
                res["confidence"] = downgrade(res["confidence"])
            notes.append(f"{len(ev)} High-Impact-Termin(e) {when}")
    if vix_regime in ("Panik",):
        res["confidence"] = "NIEDRIG"
        notes.append("VIX-Panik-Regime")
    if asset == "6J" and intervention and intervention.get("level") in ("mittel", "hoch") and res["label"] == "SHORT":
        res["confidence"] = "NIEDRIG" if intervention["level"] == "hoch" else downgrade(res["confidence"])
        notes.append(f"Interventionsrisiko {intervention['level']} – Yen-Short gefährlich")
    res.update({"asset": asset, "name": spec["name"], "structure": st_, "tape": tp, "components": comps,
                "flip": flip_level(res["label"], st_, spec["decimals"], horizon), "notes": notes, "done": done,
                "price": price, "horizon": horizon})
    return res


# ── Daten laden ──────────────────────────────────────────────────────────────
def _ohlc_fetcher():
    import macro_dashboard as md

    @md._cache(30 * 60)
    def fetch_ohlc(symbol: str, years: int = 15) -> pd.DataFrame:
        _cache_version = 2  # noqa: F841
        try:
            import yfinance as yf
            start = (pd.Timestamp.today() - pd.DateOffset(years=years, days=10)).strftime("%Y-%m-%d")
            d = yf.download(symbol, start=start, interval="1d", progress=False, auto_adjust=False)
            if d is None or d.empty:
                d = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=False)
            if d is None or d.empty:
                return pd.DataFrame()
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d = d[["Open", "High", "Low", "Close"]].dropna()
            d = d[(d > 0).all(axis=1)]
            d.index = md._trading_dates(d.index)
            return d[~d.index.duplicated(keep="last")]
        except Exception:
            return pd.DataFrame()
    return fetch_ohlc


def _news_fetcher():
    import macro_dashboard as md

    @md._cache(20 * 60)
    def fetch_headlines() -> list[dict]:
        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime
        try:
            import requests
        except Exception:
            return []
        out, seen = [], set()
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
        for url in NEWS_FEEDS:
            try:
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (TACO Lab)"})
                r.raise_for_status()
                root = ET.fromstring(r.content)
            except Exception:
                continue
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                if not title:
                    continue
                key = re.sub(r"\W+", "", title.lower())[:80]
                if key in seen:
                    continue
                ts = None
                try:
                    ts = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate"))).tz_convert("UTC")
                except Exception:
                    pass
                if ts is not None and ts < cutoff:
                    continue
                seen.add(key)
                src = it.findtext("source") or ("Yahoo Finance" if "yahoo" in url else "Google News")
                out.append({"title": title, "source": src, "time": ts, "link": it.findtext("link")})
        return out
    return fetch_headlines


def compute_bias(data: dict, group: str = "Indizes & Gold", horizon: str = "Daily", fetch_ohlc=None,
                 fetch_close=None, fetch_news=None, now_utc: pd.Timestamp | None = None) -> dict:
    """data = Ergebnis von macro_dashboard.compute_all."""
    import macro_dashboard as md
    c = _cfg(horizon)
    now_utc = now_utc or data["now"]
    now_ny = now_utc.tz_localize("UTC").tz_convert("America/New_York")
    fetch_ohlc = fetch_ohlc or _ohlc_fetcher()
    fetch_close = fetch_close or md.fetch_daily_close
    years = BIAS_CONFIG["fetch_years"]
    keys = GROUPS[group]
    ohlc = {k: fetch_ohlc(ASSETS[k]["symbol"], years) for k in keys}
    drivers = {}
    needed = {d for k in keys for d in SENSITIVITY[k]}
    for k in needed:
        drivers[k] = fetch_close(DRIVERS[k]["symbol"], f"{years}y")
    changes = {k: driver_change(drivers.get(k), DRIVERS[k]["unit"], c["tape_lag"]) for k in DRIVERS if k in drivers}
    headlines = []
    if fetch_news is not False:
        try:
            headlines = (fetch_news or _news_fetcher())() or []
        except Exception:
            headlines = []
    news = news_scores(headlines)
    currencies = sorted({"USD"} | {ASSETS[k]["currency"] for k in keys})
    events = relevant_events(data.get("calendar"), now_utc, currencies, horizon)
    plan = data.get("plan") or {}
    regime = plan.get("regime") or {}
    out = {}
    for k in keys:
        spec = ASSETS[k]
        r = data.get("results", {}).get(spec["macro"]) or {}
        v = r.get("verdict") or {}
        msc = v.get("score")
        mreason = ""
        if msc is not None:
            try:
                mreason = f"Makro-Fazit {v.get('label', 'n/a')} ({msc:+.2f}): " + md.summary_sentence(
                    r.get("code", spec["macro"]), r["pillars"], v)
            except Exception:
                mreason = f"Makro-Fazit {v.get('label', 'n/a')} ({msc:+.2f})"
        res = asset_bias(k, ohlc[k], now_ny, msc, mreason, changes, news["scores"].get(k), events,
                         regime.get("effective"), horizon, plan.get("intervention"))
        res["history"] = history_stats(k, res["done"], drivers, res["structure"], horizon)
        res["seasonality"] = seasonality_week(ohlc[k], now_ny) if horizon == "Weekly" else None
        out[k] = res
    guard = []
    if horizon == "Daily":
        cal = data.get("calendar")
        for _, e in events.iterrows():
            react = event_reactions(cal, e["event_code"], ohlc, e["time_utc"], BIAS_CONFIG["guard_history_n"])
            guard.append({"event": e, "reactions": react})
    return {"assets": out, "changes": changes, "drivers": drivers, "news": news, "events": events,
            "guard": guard, "now_utc": now_utc, "headline_count": len(headlines), "group": group,
            "horizon": horizon}


def compute_daily_bias(data: dict, fetch_ohlc=None, fetch_close=None, fetch_news=None,
                       now_utc: pd.Timestamp | None = None) -> dict:
    """Rueckwaertskompatibel: Daily fuer NQ/ES/GC."""
    return compute_bias(data, "Indizes & Gold", "Daily", fetch_ohlc, fetch_close, fetch_news, now_utc)


# ── UI ───────────────────────────────────────────────────────────────────────
def _scale_bar(s10: int | None, color: str) -> str:
    if s10 is None:
        return '<div style="height:6px;background:#1e293b;border-radius:3px;margin:10px 0"></div>'
    pct = abs(s10) * 5
    left = 50 if s10 >= 0 else 50 - pct
    return (f'<div style="position:relative;height:6px;background:#1e293b;border-radius:3px;margin:10px 0">'
            f'<div style="position:absolute;left:50%;top:-3px;width:1px;height:12px;background:#475569"></div>'
            f'<div style="position:absolute;left:{left}%;width:{pct}%;height:6px;background:{color};border-radius:3px"></div></div>')


def _asset_card(r: dict, col: dict) -> str:
    color = LABEL_COL.get(r["label"], "#64748b")
    s10 = r["score10"]
    s10_txt = "–" if s10 is None else f"{s10:+d}/10"
    conf_col = {"HOCH": "#4ade80", "MITTEL": "#cbd5e1", "NIEDRIG": "#f87171"}[r["confidence"]]
    dec = ASSETS[r["asset"]]["decimals"]
    price = "" if r["price"] is None else f'<span style="color:{col["muted"]};font-size:.85rem">{_fmt(r["price"], dec)}</span>'
    reasons = "".join(f'<li style="margin:3px 0"><b style="color:{col["muted"]}">{COMP_LABELS[k]}:</b> {txt}</li>'
                      for k, txt in r["reasons"])
    notes = "".join(f'<div style="font-size:.78rem;color:#facc15;margin-top:4px">⚠️ {n}</div>' for n in r["notes"])
    seas = ""
    s = r.get("seasonality")
    if s:
        scol = "#4ade80" if s["avg"] > 0 else "#f87171"
        seas = (f'<div style="font-size:.76rem;color:{col["muted"]};margin-top:6px;border-top:1px dashed {col["border"]};'
                f'padding-top:5px">📅 Seasonality KW {s["week"]} (nur Info): in <b style="color:{col["text"]}">'
                f'{s["up"] * 100:.0f} %</b> von {s["n"]} Jahren höher, Ø <b style="color:{scol}">{s["avg"]:+.2f} %</b></div>')
    return (f'<div style="background:{col["bg"]};border:1px solid {col["border"]};border-top:3px solid {color};'
            f'border-radius:10px;padding:14px 16px;height:100%;margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<span style="font-size:1.5rem;font-weight:800;color:{col["text"]}">{r["asset"]}</span>'
            f'<span style="border:1px solid {color};color:{color};font-weight:800;padding:2px 10px;border-radius:6px;'
            f'font-size:.85rem">{r["label"]}</span></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:.72rem;color:{col["muted"]};'
            f'text-transform:uppercase;letter-spacing:.05em">{r["name"]} {price}</div>'
            f'{_scale_bar(s10, color)}'
            f'<div style="display:flex;justify-content:space-between;font-size:.78rem">'
            f'<span style="color:{col["muted"]}">SICHERHEIT <b style="color:{conf_col}">{r["confidence"]}</b></span>'
            f'<span style="color:{col["text"]};font-weight:700">{s10_txt}</span></div>'
            f'<ul style="font-size:.82rem;color:#cbd5e1;padding-left:18px;margin:10px 0 8px 0;line-height:1.35">{reasons}</ul>'
            f'<div style="background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.35);border-radius:6px;'
            f'padding:6px 9px;font-size:.8rem;color:#facc15">Kippt bei: {r["flip"]}</div>{notes}{seas}</div>')


def _render_guard_daily(res: dict, data: dict, col: dict, berlin: pd.Timestamp) -> None:
    import streamlit as st
    if res["guard"]:
        st.markdown("**🛡️ News-Guard – Termine heute**")
        by_time: dict = {}
        for g in res["guard"]:
            by_time.setdefault(g["event"]["time_utc"], []).append(g)
        for t_utc, group in by_time.items():
            t = t_utc.tz_localize("UTC").tz_convert("Europe/Berlin")
            exit_t = t - pd.Timedelta(minutes=BIAS_CONFIG["guard_exit_min"])
            flat_t = t - pd.Timedelta(minutes=BIAS_CONFIG["guard_flat_min"])
            status = "vorbei" if t < berlin else f"Exits planen ab {exit_t:%H:%M} · keine neuen Trades ab {flat_t:%H:%M}"
            names = []
            for g in group:
                e = g["event"]
                printed = ""
                if pd.notna(e.get("actual")):
                    printed = f" (Actual <b>{e['actual']:g}</b>" + (f" vs. {e['forecast']:g}" if pd.notna(e.get("forecast")) else "") + ")"
                names.append(f"{e['currency']} {e['event']}{printed}")
            st.markdown(
                f'<div style="background:{col["bg"]};border:1px solid {col["border"]};border-left:5px solid #ef4444;'
                f'border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:.88rem;color:#cbd5e1">'
                f'<b style="color:{col["text"]}">{t:%H:%M}</b> · {" · ".join(names)}<br>'
                f'<span style="color:#facc15">{status}</span></div>', unsafe_allow_html=True)
        for g in res["guard"]:
            e = g["event"]
            if not g["reactions"].empty:
                with st.expander(f"Reaktion an den letzten {len(g['reactions'])} Terminen: {e['currency']} {e['event']}"):
                    st.dataframe(g["reactions"], hide_index=True, use_container_width=True)
                    st.caption("Tagesbewegung Close-zu-Close und Tagesspanne in ATR(14) am Termin-Tag.")
    else:
        _calendar_hint(data, res, "heute")


def _calendar_hint(data: dict, res: dict, when: str) -> None:
    import streamlit as st
    cal = data.get("calendar")
    last = cal["time_utc"].max() if cal is not None and not cal.empty else None
    if last is not None and last < res["now_utc"]:
        st.warning(f"🛡️ News-Guard: Der MT5-Kalender reicht nur bis {last:%d.%m.%Y} – Termine fehlen. "
                   "Kalender synchronisieren.")
    else:
        st.caption(f"🛡️ News-Guard: {when} keine High-Impact-Termine der betroffenen Währungen im MT5-Kalender.")


def _render_guard_weekly(res: dict, data: dict) -> None:
    import streamlit as st
    ev = res["events"]
    if ev is None or ev.empty:
        _calendar_hint(data, res, "diese Woche")
        return
    st.markdown("**🛡️ Termine bis Freitag** (High Impact)")
    rows = []
    for _, e in ev.iterrows():
        t = e["time_utc"].tz_localize("UTC").tz_convert("Europe/Berlin")
        rows.append({"Zeit (Berlin)": t.strftime("%a %d.%m. %H:%M"), "Währung": e["currency"],
                     "Termin": ("🏦 " if e["notenbank"] else "") + str(e["event"]),
                     "Forecast": e.get("forecast"), "Actual": e.get("actual")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 height=min(len(rows), 10) * 35 + 38)
    st.caption("🏦 = Notenbank-Entscheid. Betroffene Märkte bekommen eine Stufe weniger Sicherheit.")


def render_daily_bias(data: dict) -> None:
    import streamlit as st
    import macro_dashboard as md
    col = md._COL
    st.markdown("### 🎯 Bias · Indizes, Gold & Forex-Futures")
    c1, c2 = st.columns([1, 2])
    horizon = c1.radio("Horizont", list(HORIZONS), horizontal=True, key="bias_horizon")
    group = c2.radio("Märkte", list(GROUPS), horizontal=True, key="bias_group")
    hc = _cfg(horizon)
    st.caption(("Richtung für die heutige Session" if horizon == "Daily" else "Richtung für die laufende bzw. nächste Woche")
               + " aus Makro (1–4 Wochen), Struktur, Tape und Schlagzeilen – nach Art der Vault Engine, aber mit "
               "festen Regeln und eigenem Backtest. Kontext für deine Setups, kein Einstiegssignal. Gewichte unvalidiert.")
    with st.spinner("Bias: lade Futures, Intermarket-Daten und Schlagzeilen …"):
        res = compute_bias(data, group, horizon)
    A = res["assets"]
    head = " · ".join(f'<span style="color:{col["text"]}">{k}</span> '
                      f'<span style="color:{LABEL_COL.get(A[k]["label"], "#64748b")}">{A[k]["label"]}</span>'
                      for k in A)
    berlin = res["now_utc"].tz_localize("UTC").tz_convert("Europe/Berlin")
    size = "1.7rem" if len(A) <= 3 else "1.15rem"
    st.markdown(f'<div style="text-align:center;font-size:{size};font-weight:800;margin:6px 0 2px 0">{head}</div>'
                f'<div style="text-align:center;color:{col["muted"]};font-size:.78rem;margin-bottom:10px">'
                f'{horizon} · Stand {berlin:%d.%m.%Y %H:%M} Berlin · aktualisiert sich bei jedem Seitenaufruf '
                f'(Kurse max. 30 Min. Cache)</div>', unsafe_allow_html=True)
    keys = list(A)
    per_row = 3 if len(keys) <= 3 else 4
    for i in range(0, len(keys), per_row):
        cols = st.columns(per_row)
        for c_, k in zip(cols, keys[i:i + per_row]):
            c_.markdown(_asset_card(A[k], col), unsafe_allow_html=True)

    if horizon == "Daily":
        _render_guard_daily(res, data, col, berlin)
    else:
        _render_guard_weekly(res, data)

    # Was bewegt den Markt
    keyset = set(keys)
    tagged = [h for h in res["news"]["tagged"] if any(v != 0 for k, v in h["effect"].items() if k in keyset)]
    with st.expander(f"📰 Was bewegt den Markt – {len(tagged)} von {res['headline_count']} Schlagzeilen (24 h) mit Bezug",
                     expanded=False):
        if not tagged:
            st.caption("Keine Schlagzeilen abrufbar oder keine mit erkanntem Bezug. Die Schlagzeilen-Komponente entfällt dann.")
        for h in tagged[:25]:
            tags = " ".join(
                f'<span style="border:1px solid {"#22c55e" if v > 0 else "#ef4444"};color:{"#4ade80" if v > 0 else "#f87171"};'
                f'border-radius:4px;padding:0 6px;font-size:.72rem;margin-left:4px">{k} {"BULL" if v > 0 else "BEAR"}</span>'
                for k, v in h["effect"].items() if v != 0 and k in keyset)
            when = h["time"].tz_convert("Europe/Berlin").strftime("%d.%m. %H:%M") if h.get("time") is not None else ""
            link = (f'<a href="{h["link"]}" target="_blank" style="color:#cbd5e1;text-decoration:none">{h["title"]}</a>'
                    if h.get("link") else h["title"])
            st.markdown(f'<div style="font-size:.84rem;margin:4px 0;color:#cbd5e1">{link} '
                        f'<span style="color:{col["muted"]};font-size:.74rem">· {h["source"]} {when} · {", ".join(h["labels"])}</span>{tags}</div>',
                        unsafe_allow_html=True)
        st.caption("Stichwort-Heuristik, keine KI – erkennt z. B. „escalation“, „rate cut“, „tariff“, „outflow“. "
                   "Für eine qualitative Einordnung den Tab „KI-Einschätzung“ unten nutzen.")

    # Komponenten + Historie
    with st.expander("🔬 Komponenten, Tape und historische Quoten", expanded=False):
        rows = []
        for k, r in A.items():
            row = {"Markt": k, "Bias": f'{r["label"]} ({r["score10"]:+d})' if r["score10"] is not None else "n/a",
                   "Sicherheit": r["confidence"], "Daten": r["quality"]}
            for ck, lab in COMP_LABELS.items():
                c_ = r["components"].get(ck)
                row[lab] = "–" if not c_ else f'{c_["score"]:+.2f}'
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        since = "seit Vortagesschluss" if horizon == "Daily" else "letzte 5 Handelstage"
        st.markdown(f"**Tape {since}:** " + " · ".join(
            f'{DRIVERS[k]["label"]} {v:+.2f} {"Bp" if DRIVERS[k]["unit"] == "bp" else "%"}'
            for k, v in res["changes"].items() if v is not None))
        for k, r in A.items():
            h = r.get("history")
            yrs = hc["years"] if horizon == "Daily" else BIAS_CONFIG["fetch_years"]
            st.markdown(f"**{k} – Historie (bis {yrs} Jahre, {'Tages' if horizon == 'Daily' else 'Wochen'}kerzen)**")
            if not h:
                st.caption("Zu wenig Kursdaten.")
                continue
            lines = [f"Basisquote: {h['base_up'] * 100:.0f} % aller {hc['unit_txt']} schlossen höher "
                     f"({h['n_days']} {hc['unit_txt']} ab {h['start']:%Y})."]
            o = h.get("odds")
            if o:
                sig = "" if o["p"] is None else (" – signifikant (p ≤ 0,05)" if o["p"] <= 0.05 else f" – nicht signifikant (p = {o['p']:.2f})")
                lines.append(f"Heutige Lage (**{o['bucket']}**, {'über' if o['above_sma'] else 'unter'} {hc['avg_txt']}): "
                             f"{hc['next_txt']} in **{o['up'] * 100:.0f} %** von {o['n']} Fällen höher, Ø {o['avg_ret']:+.2f} %, "
                             f"Ø Spanne {_fmt(o['avg_range'], ASSETS[k]['decimals'])}{sig}")
            b = h.get("backtest")
            if b:
                verdict = ("✅ Kante (p ≤ 0,05)" if b["p"] is not None and b["p"] <= 0.05 else
                           "🟡 grenzwertig (p ≤ 0,10)" if b["p"] is not None and b["p"] <= 0.10 else "❌ keine belegbare Kante")
                lh = "–" if b["long_hit"] is None else f"{b['long_hit'] * 100:.0f} %"
                sh = "–" if b["short_hit"] is None else f"{b['short_hit'] * 100:.0f} %"
                lines.append(f"Backtest Struktur + Tape (|Score| ≥ {BIAS_CONFIG['backtest_min_abs']:.2f}): Richtung "
                             f"({hc['next_txt']}) in **{b['hit'] * 100:.1f} %** von {b['n']} Signalen getroffen "
                             f"(Long {lh} / {b['long_n']}, Short {sh} / {b['short_n']}), Ø {b['avg_ret_signal']:+.3f} % je Signal · {verdict}")
            s = r.get("seasonality")
            if s:
                sp = "" if s["p"] is None else f", p = {s['p']:.2f}"
                yrs_txt = ", ".join(f"{y}: {v:+.1f} %" for y, v in sorted(s["years"].items())[-8:])
                lines.append(f"Seasonality KW {s['week']} (nur Info, nicht im Score): {s['up'] * 100:.0f} % von {s['n']} Jahren höher, "
                             f"Ø {s['avg']:+.2f} %, Median {s['median']:+.2f} %{sp} · letzte Jahre: {yrs_txt}")
            st.markdown("\n".join(f"- {x}" for x in lines))
        st.caption("Makro und Schlagzeilen sind im Backtest nicht enthalten (keine Historie). "
                   "Kerze t wird mit dem Schlusskurs bewertet und gegen die Richtung der nächsten Kerze geprüft – "
                   "keine Kosten, kein Stopp. Yahoo-Futures sind Endloskontrakte (Rollsprünge möglich).")

    with st.expander("📐 Methodik Bias", expanded=False):
        wd, ww = HORIZONS["Daily"]["weights"], HORIZONS["Weekly"]["weights"]
        st.markdown(f"""
**Score** = gewichteter Schnitt der verfügbaren Komponenten (je −1 … +1), ×10 gerundet.
Gewichte Daily: Makro {wd['makro']:.0%} · Struktur {wd['struktur']:.0%} · Tape {wd['tape']:.0%} · Schlagzeilen {wd['news']:.0%}.
Weekly: Makro {ww['makro']:.0%} · Struktur {ww['struktur']:.0%} · Tape {ww['tape']:.0%} · Schlagzeilen {ww['news']:.0%}.
Fehlt eine Komponente, werden die übrigen hochgewichtet.

- **Makro**: Fazit-Score der Futures-Ansicht dieses Dashboards (Notenbanken, Risikostimmung,
  Daten-Überraschungen, COT). Horizont 1–4 Wochen – im Weekly stärker gewichtet.
- **Struktur**: Lage zur Vortages- bzw. Vorwochenrange (über Hoch = +1, unter Tief = −1), Abstand zum
  20-Tage- bzw. 10-Wochen-Schnitt und 5-Tage- bzw. 4-Wochen-Momentum, gemessen in ATR.
- **Tape**: Veränderung seit Vortagesschluss (Daily) bzw. über 5 Handelstage (Weekly).
  NQ/ES: steigender VIX und Renditen belasten. Gold: Renditen und Dollar belasten.
  Forex-Futures (notieren gegen USD): steigende US-Renditen belasten alle außer DXY;
  JPY/CHF profitieren von steigendem VIX, AUD/NZD von steigenden Aktien (AUD auch Gold), CAD vom Öl.
- **Schlagzeilen**: RSS (Yahoo Finance, Google News) der letzten 24 h, Stichwort-Regeln
  (Geopolitik → Gold/JPY/CHF bull, Indizes/AUD/NZD bear; Fed dovish → Dollar bear usw.).

**Label**: ab ±{BIAS_CONFIG['label_min']} LONG/SHORT, sonst NEUTRAL.
**Sicherheit**: HOCH ab ±{BIAS_CONFIG['high_min']} mit ≥ 3 Komponenten dafür und keiner klar dagegen.
Daily: eine Stufe runter bei High-Impact-Terminen heute, NIEDRIG bei Notenbank-Entscheid (USD oder eigene
Währung). Weekly: eine Stufe runter bei Notenbank-Entscheid oder ≥ 4 High-Impact-Terminen in der Woche.
NIEDRIG bei VIX-Panik; Yen-Short wird bei Interventionsrisiko herabgestuft.
**Kipp-Marke**: Long über dem Vortages-/Vorwochenhoch → kippt beim Rückfall darunter, sonst unter dem Tief
(Short spiegelbildlich).
**Seasonality** (nur Weekly): wie sich der Markt in derselben Kalenderwoche früherer Jahre entwickelt hat –
reine Info, fließt NICHT in den Score.

**Bewusst nicht enthalten** (anders als bei der Vault Engine): Options-/Gamma-Level von SPY/QQQ und
Dark-Pool-Volumen – dafür gibt es keine verlässlichen kostenlosen Daten.
""")
    st.divider()
