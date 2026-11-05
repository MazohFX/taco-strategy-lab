"""Opening Range Breakout (NYSE 9:30 ET), Bestätigung per M15-Candle-Close.

Opening Range = High/Low der ersten Minuten ab 9:30 ET (15/30/60).
Signal = erste M15-Kerze, die ÜBER dem OR-High (Long) bzw. UNTER dem OR-Low (Short) schließt.
Einstieg zum Schlusskurs dieser Kerze. Nur das erste Signal pro Tag.
Stop-Varianten: andere OR-Seite, OR-Mitte, 0.10/0.25 x Daily-ATR.
Ausstieg: Schluss 16:00 ET (EOD) oder Ziel 1R/2R/3R (sonst EOD).
Stop wird konservativ zuerst geprüft, wenn Stop und Ziel in derselben Kerze liegen.
IS 2015-2020 / OOS 2021-2023 / Holdout 2024+.
"""
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

COST = {"USATECHIDXUSD": 2.0, "USA30IDXUSD": 3.0, "XAUUSD": 0.35}   # Spread+Slippage (Punkte, Roundtrip)


def load(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df["ts"] = ts
    df["date"] = ts.dt.date
    df["mins"] = ts.dt.hour * 60 + ts.dt.minute          # Minuten seit Mitternacht ET
    return df[["ts", "date", "mins", "open", "high", "low", "close"]]


def daily_context(df):
    rth = df[(df.mins >= 570) & (df.mins < 960)]
    d = rth.groupby("date").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    pc = d.c.shift()
    tr = np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs()))
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift()
    d["gap"] = (d.o - pc) / d.atr
    d["ema20"] = d.c.ewm(span=20, adjust=False).mean().shift()
    d["trend"] = np.sign(pc - d.ema20)
    return d


def trades_for_day(bars, or_min, last_entry, ctx, cost):
    """bars: M15-Kerzen eines Tages 9:30-16:00 ET (sortiert). Gibt Rohdaten des ersten Signals."""
    orb = bars[bars.mins < 570 + or_min]
    if len(orb) < or_min // 15:
        return None
    hi, lo = orb.high.max(), orb.low.min()
    after = bars[(bars.mins >= 570 + or_min)]
    if after.empty:
        return None
    c = after.close.values; m = after.mins.values
    sig = np.flatnonzero(((c > hi) | (c < lo)) & (m + 15 <= last_entry))
    if len(sig) == 0:
        return None
    i = sig[0]
    d = 1 if c[i] > hi else -1
    return dict(i=i, d=d, entry=c[i], hi=hi, lo=lo,
                H=after.high.values, L=after.low.values, C=c)


def simulate(t, stop_mode, target_R, atr, cost):
    d, e, i = t["d"], t["entry"], t["i"]
    if stop_mode == "OR_andere":
        s = t["lo"] if d > 0 else t["hi"]
    elif stop_mode == "OR_mitte":
        s = (t["hi"] + t["lo"]) / 2
    else:
        s = e - d * float(stop_mode) * atr
    risk = d * (e - s)
    if not risk > 0:
        return np.nan
    tgt = e + d * target_R * risk if target_R else None
    H, L, C = t["H"], t["L"], t["C"]
    for j in range(i + 1, len(C)):
        if (d > 0 and L[j] <= s) or (d < 0 and H[j] >= s):
            return -1.0 - cost / risk
        if tgt is not None and ((d > 0 and H[j] >= tgt) or (d < 0 and L[j] <= tgt)):
            return target_R - cost / risk
    return d * (C[-1] - e) / risk - cost / risk


def run(sym, path):
    df = load(path)
    ctx = daily_context(df)
    if sym in COST:
        cost = COST[sym]
    else:
        import levels as lv
        cost = lv.spread_price(sym) * 1.5   # Spread + etwas Slippage zur NY-Eröffnung
    rth = df[(df.mins >= 570) & (df.mins < 960)]
    rows = []
    OR = [15, 30, 60]; LAST = [660, 720, 960]
    STOPS = ["OR_andere", "OR_mitte", "0.10", "0.25"]; TGT = [0, 1, 2, 3]
    for date, bars in rth.groupby("date", sort=True):
        if date not in ctx.index or not np.isfinite(ctx.at[date, "atr"]):
            continue
        cx = ctx.loc[date]
        for orm, last in itertools.product(OR, LAST):
            t = trades_for_day(bars, orm, last, cx, cost)
            if t is None:
                continue
            width = (t["hi"] - t["lo"]) / cx.atr
            for st, tg in itertools.product(STOPS, TGT):
                r = simulate(t, st, tg, cx.atr, cost)
                rows.append((date, orm, last, st, tg, t["d"], r, width, cx.gap, cx.trend))
    res = pd.DataFrame(rows, columns=["date", "or_min", "last_entry", "stop", "target_R", "dir", "R",
                                      "or_width_atr", "gap_atr", "trend"])
    res["year"] = pd.to_datetime(res.date).dt.year
    res["per"] = np.where(res.year <= 2020, "IS", np.where(res.year <= 2023, "OOS", "HO"))
    res.insert(0, "sym", sym)
    return res


if __name__ == "__main__":
    out = []
    for p in sys.argv[1:]:
        sym = Path(p).name.split("_")[0]
        r = run(sym, p)
        print(sym, len(r), flush=True)
        out.append(r)
    pd.concat(out).to_parquet(Path(__file__).parent / "results" / "orb_trades.parquet")
