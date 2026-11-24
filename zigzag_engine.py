"""ZigZag Sweep Strategie (3 Positionen) — Python-Port des Pine-v6-Scripts "Zig Zag".

Regeln 1:1 aus dem Pine-Script:
  * Pivots: ta.pivothigh/pivotlow(legs, legs), legs = depth // 2 (bestätigt `legs` Bars später).
  * ZigZag-Level: Gegen-Pivot mit Abstand >= deviation % zum letzten Pivot → der letzte Pivot wird Level.
    50 %-Niveau m = Mitte zwischen Level und Gegen-Pivot (retraceValid).
  * Level wird "gültig", sobald der Kurs m erreicht; danach Sweep = High > Level (bzw. Low < Level).
  * Entry am Close des Sweep-Bars (process_orders_on_close), Richtung "continuation" (High-Sweep → Long)
    oder "reversal" (High-Sweep → Short).
  * Bis zu 3 Slots (z0/z1/z2) mit eigenem SL/TP/Time-Exit, nur gleichgerichtet (TradingView nettet —
    Gegensignale werden übersprungen). Time-Exit nach N Bars am Close; der Slot wird erst im nächsten
    Bar wieder frei (wie im Pine). SL/TP-Exit gibt den Slot im selben Bar frei.
  * Stop: % vom Close oder x Tages-ATR (ATR des VORTAGES, wie request.security(..., atr[1], lookahead_on)).
  * TP: nur Zeit oder x Tages-ATR (+ Time-Exit bleibt aktiv).

Abweichungen / Annahmen (Daten haben keine Tick-Reihenfolge):
  * Treffen SL und TP im selben Bar → SL zuerst (pessimistisch, wie DAX-Engine v2).
  * Eröffnet ein Bar jenseits von SL/TP (Gap) → Fill zum Open.
  * Tagesgrenze für die Tages-ATR auf Intraday-Daten: 22:00 UTC (FX-Tageswechsel, ohne Sommerzeit-Korrektur).

Rendite je Trade: R-Multiple × Risiko-% (risikobasiertes Sizing, wie qty = riskUSD / stopdist im Pine).
Kosten: Kommission % je Order (Entry + Exit, wie TradingView "Percent") + optionaler Spread in Pips.
"""
from __future__ import annotations

import gzip
import heapq
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_D1 = ROOT / "data" / "mt5"
DATA_ID = ROOT / "data" / "dukascopy_intraday"
TIMEFRAMES = ["D1", "H4", "H1", "M15"]

# Kosten-Presets aus dem Pine-Kommentar (commission_value in %, je Order)
COST_PRESETS = {"fx_major": 0.004, "fx_minor": 0.020, "metal": 0.0075}
_MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"}


def asset_class(symbol: str) -> str:
    s = symbol.upper()
    if s.startswith("XAU") or s.startswith("XAG"):
        return "metal"
    return "fx_major" if s in _MAJORS else "fx_minor"


def pip_size(symbol: str) -> float:
    s = symbol.upper()
    if s.startswith("XAU"):
        return 0.1
    if s.startswith("XAG"):
        return 0.01
    return 0.01 if s.endswith("JPY") else 0.0001


# ════════════════════════════════════════════════════════════════════════════
# Daten
# ════════════════════════════════════════════════════════════════════════════
def available_symbols(tf: str) -> list[str]:
    if tf == "D1":
        files = sorted(DATA_D1.glob("*.csv"))
        return [f.stem for f in files]
    out = set()
    for f in DATA_ID.glob(f"*_{tf}.csv*"):
        out.add(f.name.split("_")[0])
    return sorted(s for s in out if len(s) == 6 and s.isalpha())


def load_bars(symbol: str, tf: str) -> pd.DataFrame:
    """OHLC mit DatetimeIndex (naiv, UTC bei Intraday), Spalten open/high/low/close."""
    if tf == "D1":
        p = DATA_D1 / f"{symbol}.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df.index = pd.to_datetime(df["date"], format="%Y.%m.%d")
    else:
        p = DATA_ID / f"{symbol}_{tf}.csv.gz"
        if not p.exists():
            p = DATA_ID / f"{symbol}_{tf}.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df.index = pd.to_datetime(df["datetime"])
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    if tf == "D1":
        # MT5-Historie liefert ganz am Anfang teils Wochenbars → erst ab durchgehend täglichen Bars
        df = df[df.index.dayofweek < 5]
        gaps = df.index.to_series().diff().dt.days
        early = gaps[(gaps > 6) & (gaps.index < df.index[int(len(df) * 0.4)])]
        if len(early):
            df = df[df.index >= early.index[-1]]
    return df


# ════════════════════════════════════════════════════════════════════════════
# Indikatoren
# ════════════════════════════════════════════════════════════════════════════
def _rma_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """ta.atr (RMA, Start = SMA der ersten n True Ranges)."""
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    a = 1.0 / n
    for i in range(n, len(tr)):
        out[i] = out[i - 1] + a * (tr[i] - out[i - 1])
    return out


def daily_atr_prev(df: pd.DataFrame, tf: str, n: int) -> np.ndarray:
    """Tages-ATR des VORTAGES für jeden Bar (request.security(…, ta.atr(n)[1], lookahead_on))."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    chart_atr = _rma_atr(h, l, c, n)
    if tf == "D1":
        prev = np.concatenate([[np.nan], chart_atr[:-1]])
    else:
        day = (df.index + pd.Timedelta(hours=2)).normalize()
        d = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        datr = _rma_atr(d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(), n)
        prev_d = pd.Series(np.concatenate([[np.nan], datr[:-1]]), index=d.index)
        prev = prev_d.reindex(day).to_numpy()
    return np.where(np.isnan(prev), chart_atr, prev)


def pivots(h: np.ndarray, l: np.ndarray, legs: int) -> tuple[np.ndarray, np.ndarray]:
    """ta.pivothigh/pivotlow(legs, legs): Wert am BESTÄTIGUNGS-Bar i (Pivot liegt bei i-legs), sonst NaN.
    Pivot-High: Mitte > alle linken und >= alle rechten Highs (Pivot-Low spiegelbildlich)."""
    n = len(h)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    w = 2 * legs + 1
    if n < w:
        return ph, pl
    from numpy.lib.stride_tricks import sliding_window_view as swv
    H, L = swv(h, w), swv(l, w)
    ch, cl = H[:, legs], L[:, legs]
    okh = (ch > H[:, :legs].max(axis=1)) & (ch >= H[:, legs + 1:].max(axis=1))
    okl = (cl < L[:, :legs].min(axis=1)) & (cl <= L[:, legs + 1:].min(axis=1))
    idx = np.arange(len(ch)) + 2 * legs
    ph[idx[okh]] = ch[okh]
    pl[idx[okl]] = cl[okl]
    return ph, pl


# ════════════════════════════════════════════════════════════════════════════
# Schnelle "erster Bar ab start mit A >= x"-Suche (Blockmaxima)
# ════════════════════════════════════════════════════════════════════════════
class _Seeker:
    B = 512

    def __init__(self, a: np.ndarray):
        self.a = a
        n = len(a)
        nb = (n + self.B - 1) // self.B
        pad = np.full(nb * self.B, -np.inf)
        pad[:n] = a
        self.bmax = pad.reshape(nb, self.B).max(axis=1)

    def first(self, x: float, start: int, strict: bool) -> int:
        a, B, n = self.a, self.B, len(self.a)
        if start >= n:
            return n
        e = min((start // B + 1) * B, n)
        seg = a[start:e]
        hit = seg > x if strict else seg >= x
        if hit.any():
            return start + int(hit.argmax())
        bi = start // B + 1
        bm = self.bmax[bi:]
        hb = bm > x if strict else bm >= x
        if not hb.any():
            return n
        b = bi + int(hb.argmax())
        seg = a[b * B:min((b + 1) * B, n)]
        hit = seg > x if strict else seg >= x
        return b * B + int(hit.argmax())


# ════════════════════════════════════════════════════════════════════════════
# Vorberechnung je Datensatz (gecacht)
# ════════════════════════════════════════════════════════════════════════════
_PREP: dict = {}


def _prep(df: pd.DataFrame, tf: str, atr_len: int) -> dict:
    key = (tf, len(df), df.index[0], df.index[-1], float(df["close"].iloc[-1]), atr_len)
    if key in _PREP:
        return _PREP[key]
    if len(_PREP) > 6:
        _PREP.clear()
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    p = {"o": df["open"].to_numpy(float), "h": h, "l": l, "c": df["close"].to_numpy(float),
         "idx": df.index, "atrD": daily_atr_prev(df, tf, atr_len),
         "seek_h": _Seeker(h), "seek_nl": _Seeker(-l), "piv": {}, "levels": {}}
    _PREP[key] = p
    return p


def _levels(p: dict, depth: int, deviation: float, retrace: float, max_levels: int) -> list[tuple]:
    """Alle Level mit Sweep-Bar (oder None). Rückgabe sortiert nach (Sweep-Bar, Erstellungsreihenfolge):
    (sweep_bar, order, price, is_high)."""
    key = (depth, deviation, retrace, max_levels)
    if key in p["levels"]:
        return p["levels"][key]
    legs = int(depth / 2)
    if legs not in p["piv"]:
        p["piv"][legs] = pivots(p["h"], p["l"], legs)
    ph, pl = p["piv"][legs]
    h, l = p["h"], p["l"]
    n = len(h)
    created = []                     # (created_bar, price, is_high, m)
    last_px, last_hi = None, False
    for i in np.flatnonzero(~np.isnan(ph) | ~np.isnan(pl)):
        cands = []
        if not np.isnan(ph[i]):
            cands.append((ph[i], True))
        if not np.isnan(pl[i]):
            cands.append((pl[i], False))
        for px, ih in cands:
            if last_px is None:
                last_px, last_hi = px, ih
            elif ih == last_hi:
                if (ih and px > last_px) or (not ih and px < last_px):
                    last_px = px
            else:
                dev = (px - last_px) / last_px * 100.0 if ih else (last_px - px) / last_px * 100.0
                if dev >= deviation:
                    m = px + retrace * (last_px - px) if last_hi else px - retrace * (px - last_px)
                    created.append((int(i), float(last_px), bool(last_hi), float(m)))
                    last_px, last_hi = px, ih
    # Sweep-Bar je Level (unabhängig von Positionen)
    sh, snl = p["seek_h"], p["seek_nl"]
    sweep = []
    for c, price, ih, m in created:
        if ih:
            v = sh.first(m, c, strict=False)
            s = sh.first(price, v, strict=True) if v < n else n
        else:
            v = snl.first(-m, c, strict=False)
            s = snl.first(-price, v, strict=True) if v < n else n
        sweep.append(s)
    # maxLevels-Speichergrenze: beim Anlegen wird das älteste noch aktive Level verworfen
    active_heap, order_q, alive = [], deque(), [True] * len(created)
    n_active = 0
    for k, (c, _, _, _) in enumerate(created):
        while active_heap and active_heap[0][0] < c:          # vor Bar c verbraucht
            _, j = heapq.heappop(active_heap)
            if alive[j]:
                alive[j] = False
                n_active -= 1
        heapq.heappush(active_heap, (sweep[k], k))
        order_q.append(k)
        n_active += 1
        if n_active > max_levels:
            while order_q and not alive[order_q[0]]:
                order_q.popleft()
            j = order_q.popleft()
            alive[j] = False
            sweep[j] = n                                         # verworfen → nie gesweept
            n_active -= 1
    ev = sorted((sweep[k], k, created[k][1], created[k][2]) for k in range(len(created)) if sweep[k] < n)
    p["levels"][key] = ev
    return ev


# ════════════════════════════════════════════════════════════════════════════
# Backtest
# ════════════════════════════════════════════════════════════════════════════
DEFAULTS = {"tf": "H4", "symbol": "USDJPY", "dir_mode": "continuation", "deviation": 2.5, "depth": 10,
            "retrace": 0.5, "max_conc": 3, "max_levels": 300, "sl_mode": "atr", "sl_val": 1.5,
            "tp_mode": "time", "tp_atr": 1.0, "time_bars": 20, "atr_len": 14, "risk_pct": 1.0,
            "commission_pct": 0.004, "spread_pips": 0.0}


def backtest(df: pd.DataFrame, params: dict, return_stats: bool = False):
    """Engine-Signatur wie _momi_backtest_engine: (df, params) → (trades, equity)."""
    P = {**DEFAULTS, **params}
    p = _prep(df, P["tf"], int(P["atr_len"]))
    o, h, l, c, atrD, idx = p["o"], p["h"], p["l"], p["c"], p["atrD"], p["idx"]
    n = len(c)
    ev = _levels(p, int(P["depth"]), float(P["deviation"]), float(P["retrace"]), int(P["max_levels"]))
    T = int(P["time_bars"])
    max_conc = int(P["max_conc"])
    cont = P["dir_mode"] == "continuation"
    comm = float(P["commission_pct"]) / 100.0
    spread = float(P["spread_pips"]) * pip_size(P.get("symbol", "EURUSD"))
    risk = float(P["risk_pct"])

    rel = [0, 0, 0]          # Bar, ab dem der Slot wieder frei ist
    sdir = [0, 0, 0]
    trades = []
    skip_opp = skip_full = 0
    for s, _, price, ih in ev:
        go_long = ih if cont else (not ih)
        nd = 1 if go_long else -1
        free, n_open, cur = -1, 0, 0
        for si in range(3):
            if rel[si] <= s:
                if free == -1:
                    free = si
            else:
                n_open += 1
                cur = sdir[si]
        sd = c[s] * float(P["sl_val"]) / 100.0 if P["sl_mode"] == "pct" else atrD[s] * float(P["sl_val"])
        if not (sd > 0) or np.isnan(atrD[s]):
            continue
        if cur != 0 and cur != nd:
            skip_opp += 1
            continue
        if n_open >= max_conc or free == -1:
            skip_full += 1
            continue
        entry = c[s]
        stop = entry - nd * sd
        tp = entry + nd * atrD[s] * float(P["tp_atr"]) if P["tp_mode"] == "atr" else np.nan
        a, b = s + 1, min(s + T, n - 1)
        exit_px, exit_bar, reason = None, None, None
        if a <= b:
            oo, hh, ll = o[a:b + 1], h[a:b + 1], l[a:b + 1]
            if nd > 0:
                g_sl, h_sl = oo <= stop, ll <= stop
                g_tp = oo >= tp if not np.isnan(tp) else np.zeros(len(oo), bool)
                h_tp = hh >= tp if not np.isnan(tp) else g_tp
            else:
                g_sl, h_sl = oo >= stop, hh >= stop
                g_tp = oo <= tp if not np.isnan(tp) else np.zeros(len(oo), bool)
                h_tp = ll <= tp if not np.isnan(tp) else g_tp
            anyx = g_sl | h_sl | g_tp | h_tp
            if anyx.any():
                j = int(anyx.argmax())
                exit_bar = a + j
                if g_sl[j]:
                    exit_px, reason = oo[j], "SL"
                elif g_tp[j]:
                    exit_px, reason = oo[j], "TP"
                elif h_sl[j]:
                    exit_px, reason = stop, "SL"
                else:
                    exit_px, reason = tp, "TP"
        if exit_px is not None:
            rel[free] = exit_bar
        elif s + T <= n - 1:
            exit_bar, exit_px, reason = s + T, c[s + T], "Zeit"
            rel[free] = s + T + 1
        else:
            rel[free] = n + 10 ** 9      # bis Datenende offen → kein abgeschlossener Trade
            sdir[free] = nd
            continue
        sdir[free] = nd
        cost_r = (comm * (entry + exit_px) + spread) / sd
        r_mult = nd * (exit_px - entry) / sd - cost_r
        trades.append((idx[exit_bar], idx[s], "Long" if nd > 0 else "Short", free, entry, exit_px, reason,
                       exit_bar - s, r_mult))
    cols = ["Zeit", "Entry-Zeit", "Richtung", "Slot", "Entry", "Exit", "Grund", "Bars", "R"]
    tr = pd.DataFrame(trades, columns=cols)
    if tr.empty:
        tr = pd.DataFrame(columns=cols + ["Ret %", "Risiko %", "PnL $"])
        eq = pd.Series([10_000.0], index=[idx[0]])
    else:
        tr = tr.sort_values(["Zeit", "Entry-Zeit"], kind="stable").reset_index(drop=True)
        tr["Ret %"] = tr["R"] * risk
        tr["Risiko %"] = risk
        tr["PnL $"] = tr["Ret %"] * 100.0          # auf 10.000 Startkapital
        eq = pd.Series(10_000.0 * np.cumprod(1 + tr["Ret %"].to_numpy() / 100.0), index=tr["Zeit"])
        eq = pd.concat([pd.Series([10_000.0], index=[idx[0]]), eq])
    if return_stats:
        return tr, eq, {"levels_swept": len(ev), "skip_opposite": skip_opp, "skip_full": skip_full}
    return tr, eq
