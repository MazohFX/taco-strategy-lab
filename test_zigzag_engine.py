"""Prüft zigzag_engine.backtest Trade für Trade gegen eine wörtliche Bar-für-Bar-Übersetzung des Pine-Scripts."""
"""Wörtliche Bar-für-Bar-Übersetzung des Pine-Scripts (langsam) als Referenz für zigzag_engine.backtest."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import zigzag_engine as ze


def reference(df, P):
    P = {**ze.DEFAULTS, **P}
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    atrD = ze.daily_atr_prev(df, P["tf"], int(P["atr_len"]))
    legs = int(P["depth"] / 2)
    ph, pl = ze.pivots(h, l, legs)
    levels = []  # dict price,isHigh,m,valid
    lastPx, lastHi = None, False
    slotBar = [None] * 3
    slotDir = [0] * 3
    openpos = {}  # slot -> dict(entry, stop, tp, dir, bar)
    pending_close = []  # slots to close at this bar close
    trades = []
    comm = P["commission_pct"] / 100
    spread = P["spread_pips"] * ze.pip_size(P.get("symbol", "EURUSD"))
    for i in range(n):
        # 1) Broker: SL/TP intrabar für offene Positionen (eröffnet vor diesem Bar)
        for si in list(openpos):
            ps = openpos[si]
            if ps["bar"] >= i:
                continue
            d, stop, tp = ps["dir"], ps["stop"], ps["tp"]
            px = rs = None
            if d > 0:
                if o[i] <= stop: px, rs = o[i], "SL"
                elif tp == tp and o[i] >= tp: px, rs = o[i], "TP"
                elif l[i] <= stop: px, rs = stop, "SL"
                elif tp == tp and h[i] >= tp: px, rs = tp, "TP"
            else:
                if o[i] >= stop: px, rs = o[i], "SL"
                elif tp == tp and o[i] <= tp: px, rs = o[i], "TP"
                elif h[i] >= stop: px, rs = stop, "SL"
                elif tp == tp and l[i] <= tp: px, rs = tp, "TP"
            if px is not None:
                trades.append((i, ps["bar"], d, px, ps))
                del openpos[si]
        # 2) Pivot → Level
        cands = []
        if not np.isnan(ph[i]): cands.append((ph[i], True))
        if not np.isnan(pl[i]): cands.append((pl[i], False))
        for px, ih in cands:
            if lastPx is None:
                lastPx, lastHi = px, ih
            elif ih == lastHi:
                if (ih and px > lastPx) or (not ih and px < lastPx):
                    lastPx = px
            else:
                dev = (px - lastPx) / lastPx * 100 if ih else (lastPx - px) / lastPx * 100
                if dev >= P["deviation"]:
                    m = px + P["retrace"] * (lastPx - px) if lastHi else px - P["retrace"] * (px - lastPx)
                    levels.append({"price": lastPx, "hi": lastHi, "m": m, "valid": False})
                    if len(levels) > P["max_levels"]:
                        levels.pop(0)
                    lastPx, lastHi = px, ih
        # 3) Slot-Sync
        for si in range(3):
            if slotBar[si] is not None and si not in openpos:
                slotBar[si] = None
                slotDir[si] = 0
        # 4) Time-Exit
        for si in range(3):
            eb = slotBar[si]
            if eb is not None and i - eb >= P["time_bars"] and si in openpos:
                ps = openpos.pop(si)
                trades.append((i, ps["bar"], ps["dir"], c[i], ps))
        # 5) Sweeps
        sd = c[i] * P["sl_val"] / 100 if P["sl_mode"] == "pct" else atrD[i] * P["sl_val"]
        k = 0
        while k < len(levels):
            lv = levels[k]
            consumed = False
            if not lv["valid"]:
                if (lv["hi"] and h[i] >= lv["m"]) or (not lv["hi"] and l[i] <= lv["m"]):
                    lv["valid"] = True
            if lv["valid"]:
                swept = h[i] > lv["price"] if lv["hi"] else l[i] < lv["price"]
                if swept:
                    consumed = True
                    goLong = (not lv["hi"]) if P["dir_mode"] == "reversal" else lv["hi"]
                    nd = 1 if goLong else -1
                    cur, nopen, free = 0, 0, -1
                    for si in range(3):
                        if slotBar[si] is None:
                            free = si if free == -1 else free
                        else:
                            nopen += 1
                            cur = slotDir[si]
                    if sd > 0 and not np.isnan(atrD[i]) and nopen < P["max_conc"] and free != -1 and (cur == 0 or cur == nd):
                        tp = c[i] + nd * atrD[i] * P["tp_atr"] if P["tp_mode"] == "atr" else float("nan")
                        openpos[free] = {"entry": c[i], "stop": c[i] - nd * sd, "tp": tp, "dir": nd, "bar": i, "sd": sd}
                        slotBar[free] = i
                        slotDir[free] = nd
            if consumed:
                levels.pop(k)
            else:
                k += 1
    rows = []
    for xb, eb, d, px, ps in trades:
        r = d * (px - ps["entry"]) / ps["sd"] - (comm * (ps["entry"] + px) + spread) / ps["sd"]
        rows.append((df.index[xb], df.index[eb], d, round(r, 9)))
    return pd.DataFrame(rows, columns=["Zeit", "Entry-Zeit", "dir", "R"]).sort_values(["Zeit", "Entry-Zeit"], kind="stable").reset_index(drop=True)



import pytest


@pytest.mark.parametrize("sym,tf", [("USDJPY", "D1"), ("AUDCAD", "H4")])
@pytest.mark.parametrize("P", [
    {"dir_mode": "continuation", "deviation": 2.5, "depth": 10, "sl_mode": "atr", "sl_val": 1.5, "tp_mode": "time", "time_bars": 20, "max_conc": 3},
    {"dir_mode": "reversal", "deviation": 0.8, "depth": 6, "sl_mode": "pct", "sl_val": 0.8, "tp_mode": "atr", "time_bars": 12, "max_conc": 2},
    {"dir_mode": "continuation", "deviation": 0.4, "depth": 14, "sl_mode": "atr", "sl_val": 1.5, "tp_mode": "atr", "time_bars": 40, "max_conc": 3, "max_levels": 20},
])
def test_fast_engine_matches_literal_pine_port(sym, tf, P):
    df = ze.load_bars(sym, tf)
    if df.empty:
        pytest.skip("keine Daten")
    P = {**P, "tf": tf, "symbol": sym, "spread_pips": 0.5}
    fast, _ = ze.backtest(df, P)
    ref = reference(df, P)
    assert len(fast) == len(ref) > 0
    assert (fast["Zeit"].values == ref["Zeit"].values).all()
    assert np.allclose(fast["R"].to_numpy(float), ref["R"].to_numpy(float))
