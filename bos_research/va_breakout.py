"""
Hypothese aus gianfx' echten Trades (Telegram, Feb-Okt 2026):
  BOS = Schlusskurs ausserhalb der Value Area (VAH/VAL) des laufenden Monats-Volumenprofils
        (Fixed Range ab Monats-Open bis zur Vorkerze, 70 % Value Area, Tick-Volumen).
  Fib 1 = Monatstief (Long) bzw. Monatshoch (Short) bis zum Bruch, Fib 0 = laufendes Extrem nach dem Bruch.
  Entry Limit 0.618, Stop Fib 1, 50 % bei -0.27, 50 % bei -0.62, Stop auf BE sobald Fib 0 ueberschritten.
  Neues Signal erst, wenn der Kurs wieder in der Value Area geschlossen hat.
IS 2015-2020 | OOS 2021-2023 | Holdout 2024-2026. Kein Repaint: Profil nur bis zur Vorkerze.
"""
import os, sys, time
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp

NB = 48


@njit(cache=True)
def value_area(h, l, v, mstart, min_bars):
    n = len(h); vah = np.full(n, np.nan); val = np.full(n, np.nan); poc = np.full(n, np.nan)
    prof = np.zeros(NB)
    for t in range(1, n):
        i0 = t - mstart[t]; i1 = t - 1
        if i1 - i0 + 1 < min_bars: continue
        hi = -1e18; lo = 1e18
        for i in range(i0, i1 + 1):
            if h[i] > hi: hi = h[i]
            if l[i] < lo: lo = l[i]
        if not hi > lo: continue
        w = (hi - lo) / NB
        for k in range(NB): prof[k] = 0.0
        for i in range(i0, i1 + 1):
            a = int((l[i] - lo) / w); b = int(np.ceil((h[i] - lo) / w))
            if a > NB - 1: a = NB - 1
            if b > NB: b = NB
            if b <= a: b = a + 1
            share = v[i] / (b - a)
            for k in range(a, b): prof[k] += share
        kp = 0; tot = 0.0
        for k in range(NB):
            tot += prof[k]
            if prof[k] > prof[kp]: kp = k
        lo_i = kp; hi_i = kp; acc = prof[kp]
        while acc < 0.7 * tot and (lo_i > 0 or hi_i < NB - 1):
            up = prof[hi_i + 1] if hi_i < NB - 1 else -1.0
            dn = prof[lo_i - 1] if lo_i > 0 else -1.0
            if up >= dn:
                hi_i += 1; acc += up
            else:
                lo_i -= 1; acc += dn
        vah[t] = lo + (hi_i + 1) * w; val[t] = lo + lo_i * w; poc[t] = lo + (kp + 0.5) * w
    return vah, val, poc


@njit(cache=True)
def run(h, l, c, vah, val, mstart, sp, entry_lvl, fill_bars, hold):
    n = len(h); out_t = []; out_d = []; out_R = []; out_f = []; out_rng = []; out_E = []
    armed_up = True; armed_dn = True
    for t in range(1, n):
        if np.isnan(vah[t]): continue
        if c[t] <= vah[t]: armed_up = True
        if c[t] >= val[t]: armed_dn = True
        for d in (1, -1):
            if d == 1 and not (armed_up and c[t] > vah[t]): continue
            if d == -1 and not (armed_dn and c[t] < val[t]): continue
            if d == 1: armed_up = False
            else: armed_dn = False
            i0 = t - mstart[t]
            origin = 1e18 if d == 1 else -1e18; E = -1e18 if d == 1 else 1e18
            for i in range(i0, t + 1):
                if d == 1:
                    if l[i] < origin: origin = l[i]
                    if h[i] > E: E = h[i]
                else:
                    if h[i] > origin: origin = h[i]
                    if l[i] < E: E = l[i]
            fill = -1
            for u in range(t + 1, min(n, t + 1 + fill_bars)):
                lvl = E - d * entry_lvl * abs(E - origin)
                adv = l[u] if d == 1 else h[u]
                if d * (adv - lvl) <= 0:
                    fill = u; break
                if d == 1:
                    if h[u] > E: E = h[u]
                else:
                    if l[u] < E: E = l[u]
            if fill < 0: continue
            rng = abs(E - origin); e = E - d * entry_lvl * rng; risk = abs(e - origin) + sp
            if rng <= 0: continue
            tp1 = E + d * 0.27 * rng; tp2 = E + d * 0.62 * rng
            stop = origin; be = False; half = False; R = 0.0; done = False
            last = min(n - 1, fill + hold)
            for u in range(fill, last + 1):
                adv = l[u] if d == 1 else h[u]; fav = h[u] if d == 1 else l[u]
                if d * (adv - stop) <= 0:
                    R += (0.5 if half else 1.0) * d * (stop - e) / risk; done = True; break
                if u > fill:
                    if not half and d * (fav - tp1) >= 0:
                        R += 0.5 * abs(tp1 - e) / risk; half = True
                    if half and d * (fav - tp2) >= 0:
                        R += 0.5 * abs(tp2 - e) / risk; done = True; break
                    if not be and d * (fav - E) > 0:
                        stop = e; be = True
            if not done:
                R += (0.5 if half else 1.0) * d * (c[last] - e) / risk
            out_t.append(t); out_d.append(d); out_R.append(R - sp / risk); out_f.append(fill)
            out_rng.append(rng); out_E.append(E)
    return out_t, out_d, out_R, out_f, out_rng, out_E


MIN_BARS = {"M15": 96 * 3, "H1": 24 * 3, "H4": 6 * 3}
FILL = {"M15": 96, "H1": 48, "H4": 30}
HOLD = {"M15": 96 * 5, "H1": 24 * 10, "H4": 6 * 20}

if __name__ == "__main__":
    tf = sys.argv[1]; root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for s in bp.FX:
        t0 = time.time(); df = bp.load(s, tf, root)
        if df is None: continue
        h, l, c, o, v = (df[x].values.astype(np.float64) for x in ["high", "low", "close", "open", "vol"])
        mstart = df.groupby(df.dt.dt.to_period("M")).cumcount().values.astype(np.int64)
        vah, val, poc = value_area(h, l, v, mstart, MIN_BARS[tf])
        a = bp.atr(df)
        opens = {nm: df.groupby(df.dt.dt.to_period(fq))["open"].transform("first").values
                 for nm, fq in [("D", "D"), ("W", "W-SUN"), ("M", "M"), ("Q", "Q"), ("Y", "Y")]}
        for ent in (0.618, 0.786):
            et, ed, eR, ef, erng, eE = run(h, l, c, vah, val, mstart, bp.spread(s), ent, FILL[tf], HOLD[tf])
            for t, d, R, f, rng, E in zip(et, ed, eR, ef, erng, eE):
                z = sorted([E - d * 0.618 * rng, E - d * 0.786 * rng])
                r = dict(sym=s, tf=tf, entry=ent, t=df.dt.iat[t], dir=d, R=R, hour=df.dt.iat[f].hour,
                         rng_atr=rng / a[t], sideD=int(d * (c[t] - opens["D"][t]) > 0),
                         sideW=int(d * (c[t] - opens["W"][t]) > 0), sideM=int(d * (c[t] - opens["M"][t]) > 0),
                         poc=int(z[0] <= poc[f] <= z[1]) if not np.isnan(poc[f]) else 0)
                for nm in opens: r[f"o{nm}"] = int(z[0] <= opens[nm][f] <= z[1])
                rows.append(r)
        print(tf, s, f"{time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(rows).to_parquet(f"{root}/bos_research/results/va_{tf}.parquet", index=False)
