"""Daily Bias: taegliche Richtung fuer NQ, ES und Gold (GC) – Horizont: die heutige Session.

Baustein fuer das Makro-Dashboard (macro_dashboard.py), nach dem Vorbild der
"Vault Engine" (payoutvault.one), aber mit festen, nachvollziehbaren Regeln statt
eines frei formulierenden Sprachmodells:

  1. Makro (1-4 Wochen)  Fazit-Score der Futures-Ansicht des Dashboards
                         (Zinsen/Fed, Risikostimmung, US-Daten-Ueberraschungen, COT)
  2. Struktur            Kurs vs. Vortageshoch/-tief, vs. 20-Tage-Durchschnitt,
                         5-Tage-Momentum (jeweils in ATR gemessen)
  3. Tape (Intermarket)  Veraenderung seit dem letzten Schlusskurs: VIX,
                         US-10J-Rendite, Dollar-Index, Oel, Bitcoin
  4. Schlagzeilen        Stichwort-Heuristik ueber RSS-Schlagzeilen der letzten
                         24 h (Geopolitik, Fed, Zoelle, Fondsflüsse ...), geringes Gewicht

Ausgabe je Markt: LONG/SHORT/NEUTRAL, Score -10..+10, Sicherheit, die drei
wichtigsten Gruende und die Marke, an der die Einschaetzung kippt. Dazu:
News-Guard fuer heutige US-High-Impact-Termine (mit der Reaktion von NQ/ES/GC an
den letzten Terminen), historische Quote fuer die heutige Ausgangslage und ein
Backtest von Struktur + Tape (Makro und Schlagzeilen lassen sich nicht
rueckwirkend rekonstruieren).

ALLE Gewichte und Schwellen sind UNVALIDIERTE STARTWERTE.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

BIAS_CONFIG = {
    "weights": {"makro": 0.30, "struktur": 0.35, "tape": 0.25, "news": 0.10},
    "label_min": 3,            # |Score| (von 10) ab dem LONG/SHORT gilt
    "high_min": 6,             # |Score| fuer Sicherheit HOCH
    "support_min": 0.15,       # Komponente zaehlt als "dafuer"
    "oppose_min": 0.30,        # Komponente zaehlt als "klar dagegen"
    "sma_days": 20,
    "atr_days": 14,
    "mom_days": 5,
    "history_years": 5,
    "guard_exit_min": 60,      # Exits planen x Minuten vor dem Termin
    "guard_flat_min": 15,      # keine neuen Trades x Minuten vor dem Termin
    "guard_history_n": 8,      # so viele fruehere Termine fuer die Reaktion
    "backtest_min_abs": 0.30,  # |Struktur+Tape-Score| ab dem ein historisches Signal zaehlt
}

ASSETS = {
    "NQ": {"name": "Nasdaq 100 Futures", "symbol": "NQ=F", "macro": "Nasdaq 100", "decimals": 2},
    "ES": {"name": "S&P 500 Futures", "symbol": "ES=F", "macro": "S&P 500", "decimals": 2},
    "GC": {"name": "Gold Futures", "symbol": "GC=F", "macro": "Gold", "decimals": 1},
}

# Tape-Treiber: Yahoo-Symbol, Einheit der Veraenderung, Skala (tanh ~0.76 bei 1 Skala)
DRIVERS = {
    "VIX": {"symbol": "^VIX", "unit": "pct", "scale": 6.0, "label": "VIX"},
    "US10Y": {"symbol": "^TNX", "unit": "bp", "scale": 5.0, "label": "US-10J-Rendite"},
    "DXY": {"symbol": "DX-Y.NYB", "unit": "pct", "scale": 0.4, "label": "Dollar-Index"},
    "OIL": {"symbol": "CL=F", "unit": "pct", "scale": 2.0, "label": "Öl (WTI)"},
    "BTC": {"symbol": "BTC-USD", "unit": "pct", "scale": 3.0, "label": "Bitcoin"},
}
# Wie ein steigender Treiber auf den Markt wirkt (unvalidierte Faustregeln)
SENSITIVITY = {
    "NQ": {"VIX": -1.0, "US10Y": -1.0, "DXY": -0.4, "BTC": +0.3},
    "ES": {"VIX": -1.0, "US10Y": -0.6, "DXY": -0.4, "BTC": +0.2},
    "GC": {"VIX": +0.4, "US10Y": -1.0, "DXY": -1.0, "OIL": +0.3},
}

# Schlagzeilen-Regeln: (Regex, Wirkung je Markt, Kurzlabel)
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
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC,%5ENDX,GC%3DF&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=(stocks%20OR%20fed%20OR%20gold%20OR%20geopolitics)%20when:1d&hl=en-US&gl=US&ceid=US:en",
]

LABEL_COL = {"LONG": "#22c55e", "SHORT": "#ef4444", "NEUTRAL": "#eab308"}
CONF_ORDER = ["NIEDRIG", "MITTEL", "HOCH"]
COMP_LABELS = {"makro": "Makro (1–4 Wo.)", "struktur": "Struktur", "tape": "Tape", "news": "Schlagzeilen"}


def _tanh(x: float, scale: float) -> float:
    return float(np.tanh(x / scale)) if scale else 0.0


def _fmt(v: float, dec: int) -> str:
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ── reine Berechnung (testbar ohne Streamlit/Netzwerk) ───────────────────────
def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def split_running(df: pd.DataFrame, now_ny: pd.Timestamp) -> tuple[pd.DataFrame, float | None]:
    """Trennt die laufende Tageskerze ab. Rueckgabe: (abgeschlossene Kerzen, aktueller Kurs).
    Futures-Session: 18:00 NY bis 17:00 NY. Eine Kerze mit heutigem (oder spaeterem) Datum
    gilt als laufend, solange es vor 17:00 NY ist; nach 18:00 gehoert eine Kerze mit
    morgigem Datum zur neuen Session."""
    if df is None or df.empty:
        return df, None
    last_day = pd.Timestamp(df.index[-1]).normalize()
    today = now_ny.normalize().tz_localize(None) if now_ny.tzinfo else now_ny.normalize()
    hour = now_ny.hour + now_ny.minute / 60
    running = (last_day > today) or (last_day == today and hour < 17)
    if running and len(df) > 1:
        return df.iloc[:-1], float(df["Close"].iloc[-1])
    return df, float(df["Close"].iloc[-1])


def structure_state(done: pd.DataFrame, price: float) -> dict | None:
    """Struktur-Score (-1..+1) aus abgeschlossenen Tageskerzen und aktuellem Kurs."""
    c = BIAS_CONFIG
    if done is None or len(done) < max(c["sma_days"], c["atr_days"]) + c["mom_days"] + 1:
        return None
    a = float(atr(done, c["atr_days"]).iloc[-1])
    if not a or math.isnan(a):
        return None
    pdh, pdl = float(done["High"].iloc[-1]), float(done["Low"].iloc[-1])
    pdc = float(done["Close"].iloc[-1])
    sma = float(done["Close"].iloc[-c["sma_days"]:].mean())
    ref = float(done["Close"].iloc[-c["mom_days"]])
    if price > pdh:
        pos, pos_txt = 1.0, "über dem Vortageshoch"
    elif price < pdl:
        pos, pos_txt = -1.0, "unter dem Vortagestief"
    else:
        half = (pdh - pdl) / 2 or a
        pos = 0.5 * (price - (pdh + pdl) / 2) / half
        pos_txt = "in der Vortagesrange (" + ("obere" if pos > 0 else "untere") + " Hälfte)"
    sma_s = _tanh((price - sma) / a, 1.5)
    mom_s = _tanh((price - ref) / a, 2.0)
    score = float(np.mean([pos, sma_s, mom_s]))
    dist_atr = (price - sma) / a
    sma_txt = (f"{'deutlich ' if abs(dist_atr) > 1.5 else ''}{'über' if dist_atr > 0 else 'unter'} dem "
               f"20-Tage-Schnitt ({dist_atr:+.1f} ATR)")
    mom_txt = f"5-Tage-Momentum {'positiv' if price > ref else 'negativ'} ({(price - ref) / a:+.1f} ATR)"
    return {"score": score, "pos": pos, "sma": sma_s, "mom": mom_s, "pdh": pdh, "pdl": pdl, "pdc": pdc,
            "sma_level": sma, "atr": a, "price": price,
            "reason": f"Kurs {pos_txt}, {sma_txt}, {mom_txt}",
            "parts": [pos_txt, sma_txt, mom_txt]}


def driver_change(series: pd.Series | None, unit: str) -> float | None:
    if series is None:
        return None
    s = series.dropna()
    if len(s) < 2:
        return None
    a, b = float(s.iloc[-2]), float(s.iloc[-1])
    if unit == "bp":
        return (b - a) * 100.0
    return (b / a - 1) * 100.0 if a else None


def tape_score(asset: str, changes: dict[str, float | None]) -> dict | None:
    sens = SENSITIVITY[asset]
    num = den = 0.0
    items = []
    for k, w in sens.items():
        ch = changes.get(k)
        if ch is None or (isinstance(ch, float) and math.isnan(ch)):
            continue
        eff = w * _tanh(ch, DRIVERS[k]["scale"])
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
    return {"score": score, "items": items, "reason": "Seit Vortagesschluss: " + ", ".join(parts)}


def tag_headline(title: str) -> dict:
    t = title.lower()
    effect = {"NQ": 0, "ES": 0, "GC": 0}
    labels = []
    for pattern, eff, label in NEWS_RULES:
        if re.search(pattern, t):
            if label == "Geopolitik" and "Geopolitik" in labels:
                continue
            if label == "Geopolitik" and "Entspannung" in labels:
                continue  # "ceasefire after attack" -> Entspannung zaehlt
            for k, v in eff.items():
                effect[k] += v
            labels.append(label)
    return {"effect": {k: int(np.sign(v)) for k, v in effect.items()}, "labels": labels}


def news_scores(headlines: list[dict]) -> dict:
    """headlines: [{'title':..., 'source':..., 'time':...}] -> Score je Markt + getaggte Liste."""
    tagged = []
    tot = {"NQ": 0, "ES": 0, "GC": 0}
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
        top = {}
        for x in relevant:
            for lb in x["labels"]:
                top[lb] = top.get(lb, 0) + 1
        topic = ", ".join(f"{lb} ({n}×)" for lb, n in sorted(top.items(), key=lambda kv: -kv[1])[:3])
        out[k] = {"score": _tanh(tot[k], 3.0), "n": len(relevant),
                  "reason": f"{len(relevant)} relevante Schlagzeile{'n' if len(relevant) != 1 else ''} (24 h): {topic}"}
    return {"scores": out, "tagged": tagged}


def combine(comps: dict[str, dict | None]) -> dict:
    """comps: {'makro': {'score','reason'} | None, ...} -> Label, Score10, Sicherheit, Gruende."""
    c = BIAS_CONFIG
    avail = {k: v for k, v in comps.items() if v is not None and v.get("score") is not None}
    if not avail:
        return {"score": None, "score10": None, "label": "n/a", "confidence": "NIEDRIG",
                "reasons": [], "quality": f"0/{len(comps)}", "contrib": {}}
    w = c["weights"]
    tw = sum(w[k] for k in avail)
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


def flip_level(label: str, st_: dict | None, dec: int) -> str:
    if st_ is None:
        return "–"
    pdh, pdl, price = st_["pdh"], st_["pdl"], st_["price"]
    if label == "LONG":
        lvl, txt = (pdh, "Rückfall unter das Vortageshoch") if price > pdh else (pdl, "Bruch unter das Vortagestief")
        return f"{txt} {_fmt(lvl, dec)}"
    if label == "SHORT":
        lvl, txt = (pdl, "Rückkehr über das Vortagestief") if price < pdl else (pdh, "Bruch über das Vortageshoch")
        return f"{txt} {_fmt(lvl, dec)}"
    return f"Long über {_fmt(pdh, dec)} (Vortageshoch), Short unter {_fmt(pdl, dec)} (Vortagestief)"


def todays_us_events(calendar: pd.DataFrame | None, now_utc: pd.Timestamp) -> pd.DataFrame:
    """US-High-Impact-Termine des heutigen New-Yorker Handelstags (bis 17:00 NY), noch nicht vorbei
    oder in den letzten 2 Stunden."""
    if calendar is None or calendar.empty:
        return pd.DataFrame()
    ny_now = now_utc.tz_localize("UTC").tz_convert("America/New_York")
    end = ny_now.normalize() + pd.Timedelta(hours=17)
    if ny_now >= end:
        end = end + pd.offsets.BDay(1)
    start = now_utc - pd.Timedelta(hours=2)
    end_utc = end.tz_convert("UTC").tz_localize(None)
    ev = calendar[(calendar["currency"] == "USD") & (calendar["importance"] == "High")
                  & (calendar["time_utc"] >= start) & (calendar["time_utc"] <= end_utc)].copy()
    return ev.sort_values("time_utc")


def event_reactions(calendar: pd.DataFrame, event_code: str, ohlc: dict[str, pd.DataFrame],
                    before: pd.Timestamp, n: int) -> pd.DataFrame:
    """Tagesbewegung von NQ/ES/GC an den letzten n Terminen dieses Events (Close-zu-Close in %,
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
            a = atr(df.iloc[: i], BIAS_CONFIG["atr_days"]).iloc[-1]
            ret = (df["Close"].iloc[i] / df["Close"].iloc[i - 1] - 1) * 100
            rng = (df["High"].iloc[i] - df["Low"].iloc[i]) / a if a else np.nan
            row[k] = f"{ret:+.2f} % · {rng:.1f} ATR"
        rows.append(row)
    return pd.DataFrame(rows)


def _hist_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Struktur-Score je Tag t (bewertet mit Schlusskurs t gegen Range t-1), vektorisiert."""
    c = BIAS_CONFIG
    # Wie live: Kurs (hier Schluss t) gegen Range, SMA und ATR der abgeschlossenen Kerzen bis t-1
    close, hi, lo = df["Close"], df["High"].shift(1), df["Low"].shift(1)
    a = atr(df, c["atr_days"]).shift(1)
    half = ((hi - lo) / 2).replace(0, np.nan)
    pos = np.where(close > hi, 1.0, np.where(close < lo, -1.0, 0.5 * (close - (hi + lo) / 2) / half))
    sma = close.rolling(c["sma_days"]).mean().shift(1)
    sma_s = np.tanh((close - sma) / a / 1.5)
    mom_s = np.tanh((close - close.shift(c["mom_days"])) / a / 2.0)
    out = pd.DataFrame({"pos": pos, "sma": sma_s, "mom": mom_s}, index=df.index)
    out["struktur"] = out[["pos", "sma", "mom"]].mean(axis=1)
    out["above_sma"] = close > sma
    out["pos_bucket"] = np.where(close > hi, "über Vortageshoch", np.where(close < lo, "unter Vortagestief", "in Vortagesrange"))
    out["atr"] = a
    return out


def _hist_tape(asset: str, index: pd.DatetimeIndex, drivers: dict[str, pd.Series]) -> pd.Series:
    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    for k, w in SENSITIVITY[asset].items():
        s = drivers.get(k)
        if s is None or s.dropna().empty:
            continue
        s = s.dropna()
        ch = s.diff() * 100 if DRIVERS[k]["unit"] == "bp" else s.pct_change() * 100
        ch = ch.reindex(index)
        eff = w * np.tanh(ch / DRIVERS[k]["scale"])
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
                  today_state: dict | None) -> dict | None:
    """1) Quote fuer die heutige Ausgangslage (Lage zur Vortagesrange x 20-Tage-Schnitt).
    2) Backtest Struktur+Tape: Vorzeichen am Tag t -> Richtung Tag t+1."""
    if done is None or len(done) < 120:
        return None
    w = BIAS_CONFIG["weights"]
    h = _hist_structure(done)
    h["tape"] = _hist_tape(asset, done.index, drivers)
    tech_w = w["struktur"] + w["tape"]
    h["tech"] = np.where(h["tape"].notna(), (w["struktur"] * h["struktur"] + w["tape"] * h["tape"]) / tech_w, h["struktur"])
    nxt = done["Close"].shift(-1) / done["Close"] - 1
    h["next_up"] = nxt > 0
    h["next_ret"] = nxt * 100
    h["next_range"] = (done["High"].shift(-1) - done["Low"].shift(-1))
    h = h.dropna(subset=["struktur", "next_ret"])
    base_up = float(h["next_up"].mean())
    out = {"n_days": len(h), "base_up": base_up, "start": h.index[0], "end": h.index[-1]}
    if today_state is not None:
        bucket = ("über Vortageshoch" if today_state["price"] > today_state["pdh"] else
                  "unter Vortagestief" if today_state["price"] < today_state["pdl"] else "in Vortagesrange")
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


def asset_bias(asset: str, ohlc: pd.DataFrame | None, now_ny: pd.Timestamp, macro_score: float | None,
               macro_reason: str, changes: dict, news: dict | None, events: pd.DataFrame,
               vix_regime: str | None = None) -> dict:
    spec = ASSETS[asset]
    done, price = split_running(ohlc, now_ny) if ohlc is not None and not ohlc.empty else (None, None)
    st_ = structure_state(done, price) if price is not None else None
    tp = tape_score(asset, changes)
    comps = {
        "makro": None if macro_score is None else {"score": float(macro_score), "reason": macro_reason},
        "struktur": None if st_ is None else {"score": st_["score"], "reason": st_["reason"]},
        "tape": tp,
        "news": news,
    }
    res = combine(comps)
    notes = []
    if events is not None and not events.empty:
        fomc = events["event_code"].astype(str).str.contains("fed-interest-rate-decision").any()
        res["confidence"] = "NIEDRIG" if fomc else downgrade(res["confidence"])
        notes.append("Fed-Zinsentscheid heute" if fomc else f"{len(events)} US-High-Impact-Termin(e) heute")
    if vix_regime in ("Panik",):
        res["confidence"] = "NIEDRIG"
        notes.append("VIX-Panik-Regime")
    res.update({"asset": asset, "name": spec["name"], "structure": st_, "tape": tp, "components": comps,
                "flip": flip_level(res["label"], st_, spec["decimals"]), "notes": notes, "done": done,
                "price": price})
    return res


# ── Daten laden ──────────────────────────────────────────────────────────────
def _ohlc_fetcher():
    import macro_dashboard as md

    @md._cache(30 * 60)
    def fetch_ohlc(symbol: str, years: int = 5) -> pd.DataFrame:
        _cache_version = 1  # noqa: F841
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


def compute_daily_bias(data: dict, fetch_ohlc=None, fetch_close=None, fetch_news=None,
                       now_utc: pd.Timestamp | None = None) -> dict:
    """data = Ergebnis von macro_dashboard.compute_all."""
    import macro_dashboard as md
    now_utc = now_utc or data["now"]
    now_ny = now_utc.tz_localize("UTC").tz_convert("America/New_York")
    fetch_ohlc = fetch_ohlc or _ohlc_fetcher()
    fetch_close = fetch_close or md.fetch_daily_close
    years = BIAS_CONFIG["history_years"]
    ohlc = {k: fetch_ohlc(v["symbol"], years) for k, v in ASSETS.items()}
    drivers = {}
    for k, d in DRIVERS.items():
        if k == "VIX" and data.get("closes", {}).get("vix") is not None:
            drivers[k] = data["closes"]["vix"]
        else:
            drivers[k] = fetch_close(d["symbol"], f"{years}y")
    changes = {k: driver_change(drivers.get(k), DRIVERS[k]["unit"]) for k in DRIVERS}
    headlines = []
    if fetch_news is not False:
        try:
            headlines = (fetch_news or _news_fetcher())() or []
        except Exception:
            headlines = []
    news = news_scores(headlines)
    events = todays_us_events(data.get("calendar"), now_utc)
    regime = (data.get("plan") or {}).get("regime") or {}
    out = {}
    for k, spec in ASSETS.items():
        r = data.get("results", {}).get(spec["macro"]) or {}
        v = r.get("verdict") or {}
        msc = v.get("score")
        mreason = (f"Makro-Fazit {v.get('label', 'n/a')} ({msc:+.2f}): "
                   + md.summary_sentence(spec["macro"], r["pillars"], v)) if msc is not None else ""
        res = asset_bias(k, ohlc[k], now_ny, msc, mreason, changes, news["scores"].get(k), events,
                         regime.get("effective"))
        res["history"] = history_stats(k, res["done"], drivers, res["structure"])
        out[k] = res
    guard = []
    cal = data.get("calendar")
    for _, e in events.iterrows():
        react = event_reactions(cal, e["event_code"], ohlc, e["time_utc"], BIAS_CONFIG["guard_history_n"])
        guard.append({"event": e, "reactions": react})
    return {"assets": out, "changes": changes, "drivers": drivers, "news": news, "events": events,
            "guard": guard, "now_utc": now_utc, "headline_count": len(headlines)}


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
    notes = ("".join(f'<div style="font-size:.78rem;color:#facc15;margin-top:4px">⚠️ {n}</div>' for n in r["notes"]))
    return (f'<div style="background:{col["bg"]};border:1px solid {col["border"]};border-top:3px solid {color};'
            f'border-radius:10px;padding:14px 16px;height:100%">'
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
            f'padding:6px 9px;font-size:.8rem;color:#facc15">Kippt bei: {r["flip"]}</div>{notes}</div>')


def render_daily_bias(data: dict) -> None:
    import streamlit as st
    import macro_dashboard as md
    col = md._COL
    st.markdown("### 🎯 Daily Bias · NQ · ES · Gold")
    st.caption("Richtung für die heutige Session aus Makro (1–4 Wochen), Struktur, Tape und Schlagzeilen – "
               "nach Art der Vault Engine, aber mit festen Regeln und eigenem Backtest. "
               "Kontext für deine Setups, kein Einstiegssignal. Gewichte unvalidiert.")
    with st.spinner("Daily Bias: lade Futures, Intermarket-Daten und Schlagzeilen …"):
        res = compute_daily_bias(data)
    A = res["assets"]
    head = " · ".join(f'<span style="color:{col["text"]}">{k}</span> '
                      f'<span style="color:{LABEL_COL.get(A[k]["label"], "#64748b")}">{A[k]["label"]}</span>'
                      for k in A)
    berlin = res["now_utc"].tz_localize("UTC").tz_convert("Europe/Berlin")
    st.markdown(f'<div style="text-align:center;font-size:1.7rem;font-weight:800;margin:6px 0 2px 0">{head}</div>'
                f'<div style="text-align:center;color:{col["muted"]};font-size:.78rem;margin-bottom:10px">'
                f'Stand {berlin:%d.%m.%Y %H:%M} Berlin · aktualisiert sich bei jedem Seitenaufruf (Kurse max. 30 Min. Cache)</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for c_, k in zip(cols, A):
        c_.markdown(_asset_card(A[k], col), unsafe_allow_html=True)

    # News-Guard
    st.markdown("")
    if res["guard"]:
        st.markdown("**🛡️ News-Guard – US-Termine heute**")
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
                names.append(f"{e['event']}{printed}")
            st.markdown(
                f'<div style="background:{col["bg"]};border:1px solid {col["border"]};border-left:5px solid #ef4444;'
                f'border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:.88rem;color:#cbd5e1">'
                f'<b style="color:{col["text"]}">{t:%H:%M}</b> · {" · ".join(names)}<br>'
                f'<span style="color:#facc15">{status}</span></div>', unsafe_allow_html=True)
        for g in res["guard"]:
            e = g["event"]
            if not g["reactions"].empty:
                with st.expander(f"So bewegten sich NQ/ES/GC an den letzten {len(g['reactions'])} Terminen: {e['event']}"):
                    st.dataframe(g["reactions"], hide_index=True, use_container_width=True)
                    st.caption("Tagesbewegung Close-zu-Close und Tagesspanne in ATR(14) am Termin-Tag.")
    else:
        cal = data.get("calendar")
        last = cal["time_utc"].max() if cal is not None and not cal.empty else None
        if last is not None and last < res["now_utc"]:
            st.warning(f"🛡️ News-Guard: Der MT5-Kalender reicht nur bis {last:%d.%m.%Y} – heutige Termine fehlen. "
                       "Kalender synchronisieren.")
        else:
            st.caption("🛡️ News-Guard: heute keine US-High-Impact-Termine im MT5-Kalender.")

    # Was bewegt den Markt
    tagged = res["news"]["tagged"]
    with st.expander(f"📰 Was bewegt den Markt – {len(tagged)} von {res['headline_count']} Schlagzeilen (24 h) mit Bezug",
                     expanded=False):
        if not tagged:
            st.caption("Keine Schlagzeilen abrufbar oder keine mit erkanntem Bezug. Die Schlagzeilen-Komponente entfällt dann.")
        for h in tagged[:25]:
            tags = " ".join(
                f'<span style="border:1px solid {"#22c55e" if v > 0 else "#ef4444"};color:{"#4ade80" if v > 0 else "#f87171"};'
                f'border-radius:4px;padding:0 6px;font-size:.72rem;margin-left:4px">{k} {"BULL" if v > 0 else "BEAR"}</span>'
                for k, v in h["effect"].items() if v != 0)
            when = h["time"].tz_convert("Europe/Berlin").strftime("%d.%m. %H:%M") if h.get("time") is not None else ""
            link = f'<a href="{h["link"]}" target="_blank" style="color:#cbd5e1;text-decoration:none">{h["title"]}</a>' if h.get("link") else h["title"]
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
        ch = res["changes"]
        st.markdown("**Tape seit Vortagesschluss:** " + " · ".join(
            f'{DRIVERS[k]["label"]} {v:+.2f} {"Bp" if DRIVERS[k]["unit"] == "bp" else "%"}'
            for k, v in ch.items() if v is not None))
        for k, r in A.items():
            h = r.get("history")
            st.markdown(f"**{k} – Historie ({BIAS_CONFIG['history_years']} Jahre Tagesdaten)**")
            if not h:
                st.caption("Zu wenig Kursdaten.")
                continue
            lines = [f"Basisquote: {h['base_up'] * 100:.0f} % aller Tage schlossen höher ({h['n_days']} Tage)."]
            o = h.get("odds")
            if o:
                sig = "" if o["p"] is None else (" – signifikant (p ≤ 0,05)" if o["p"] <= 0.05 else f" – nicht signifikant (p = {o['p']:.2f})")
                lines.append(f"Heutige Lage (**{o['bucket']}**, {'über' if o['above_sma'] else 'unter'} 20-Tage-Schnitt): "
                             f"nächster Tag in **{o['up'] * 100:.0f} %** von {o['n']} Fällen höher, Ø {o['avg_ret']:+.2f} %, "
                             f"Ø Spanne {_fmt(o['avg_range'], ASSETS[k]['decimals'])} Pkt.{sig}")
            b = h.get("backtest")
            if b:
                verdict = ("✅ Kante (p ≤ 0,05)" if b["p"] is not None and b["p"] <= 0.05 else
                           "🟡 grenzwertig (p ≤ 0,10)" if b["p"] is not None and b["p"] <= 0.10 else "❌ keine belegbare Kante")
                lh = "–" if b["long_hit"] is None else f"{b['long_hit'] * 100:.0f} %"
                sh = "–" if b["short_hit"] is None else f"{b['short_hit'] * 100:.0f} %"
                lines.append(f"Backtest Struktur + Tape (|Score| ≥ {BIAS_CONFIG['backtest_min_abs']:.2f}): Richtung des nächsten Tages "
                             f"in **{b['hit'] * 100:.1f} %** von {b['n']} Signalen getroffen (Long {lh} / {b['long_n']}, "
                             f"Short {sh} / {b['short_n']}), Ø {b['avg_ret_signal']:+.3f} % je Signal · {verdict}")
            st.markdown("\n".join(f"- {x}" for x in lines))
        st.caption("Makro und Schlagzeilen sind im Backtest nicht enthalten (keine Historie). "
                   "Tag t wird mit dem Schlusskurs bewertet und gegen die Richtung von Tag t+1 geprüft – "
                   "keine Kosten, kein Intraday-Stopp.")

    with st.expander("📐 Methodik Daily Bias", expanded=False):
        w = BIAS_CONFIG["weights"]
        st.markdown(f"""
**Score** = gewichteter Schnitt der verfügbaren Komponenten (je −1 … +1), ×10 gerundet:
Makro {w['makro']:.0%} · Struktur {w['struktur']:.0%} · Tape {w['tape']:.0%} · Schlagzeilen {w['news']:.0%}.
Fehlt eine Komponente, werden die übrigen hochgewichtet.

- **Makro**: Fazit-Score der Futures-Ansicht dieses Dashboards (Fed relativ zu anderen Notenbanken,
  Risikostimmung, US-Daten-Überraschungen, COT). Horizont 1–4 Wochen – gibt die Grundrichtung.
- **Struktur**: Lage zur Vortagesrange (über Hoch = +1, unter Tief = −1), Abstand zum 20-Tage-Schnitt
  und 5-Tage-Momentum, beides in ATR(14) gemessen.
- **Tape**: Veränderung seit dem letzten Schlusskurs. NQ/ES: steigender VIX und steigende Renditen
  belasten; Gold: steigende Renditen und stärkerer Dollar belasten, steigender VIX stützt leicht.
- **Schlagzeilen**: RSS (Yahoo Finance, Google News) der letzten 24 h, Stichwort-Regeln
  (Geopolitik → Gold bull / Indizes bear, Fed dovish → alles bull usw.).

**Label**: ab ±{BIAS_CONFIG['label_min']} LONG/SHORT, sonst NEUTRAL.
**Sicherheit**: HOCH ab ±{BIAS_CONFIG['high_min']} mit ≥ 3 Komponenten dafür und keiner klar dagegen;
eine Stufe runter bei US-High-Impact-Terminen heute, NIEDRIG bei Fed-Entscheid oder VIX-Panik.
**Kipp-Marke**: Long über dem Vortageshoch → kippt beim Rückfall darunter, sonst unter dem Vortagestief
(Short spiegelbildlich).

**Bewusst nicht enthalten** (anders als bei der Vault Engine): Options-/Gamma-Level von SPY/QQQ und
Dark-Pool-Volumen – dafür gibt es keine verlässlichen kostenlosen Daten.
""")
    st.divider()
