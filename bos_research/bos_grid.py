"""
BOS-Raster: Welche objektive Strukturbruch-Definition funktioniert auf welchem Paar?

Getestet wird Tobis komplettes Setup (Fib 1 = Ursprung des Bruchbeins, Fib 0 = laufendes
Extrem bis zur Fuellung, Limit am Entry-Level, Stop Fib 1, drei gleiche Teile TP -0.27/-0.62/-1,
Break-even sobald Fib 0 ueberschritten ist) mit allen BOS-Definitionen aus dem Raster:

  Swing-Methode : ATR-ZigZag k = 1 / 1.5 / 2 / 3   |   Fraktal N = 3 / 5 / 8 Kerzen links+rechts
  Bruch         : Schlusskurs (close) | Docht (wick)
  Bruch-Verh.   : Bruchbein / Ruecklaufbein >= 1.0 (kein Filter) / 1.5 / 2.0   ("1:1, 1.5:1, 2:1")
  Entry         : 0.618 | 0.786
  Konfluenz     : alle Trades | nur wenn ein Y/Q/M/W-Open in der Zone 0.618-0.786 liegt

Zeitraeume (strikt getrennt):
  IS      2015-2020  -> hier wird je Paar die beste Definition ausgewaehlt
  OOS     2021-2023  -> erster unabhaengiger Test der ausgewaehlten Definition
  Holdout 2024-2026  -> zweiter unabhaengiger Test (nur fuer die ausgewaehlten Definitionen gelesen)

Alles ohne Repaint: Swings zaehlen erst ab der Kerze NACH ihrer Bestaetigung.
Aufruf: python3 bos_research/bos_grid.py --tf H4   (oder H1, M15)
"""
import argparse, os, sys, time, itertools
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp

IS_END, OOS_END = "2021-01-01", "2024-01-01"
SWINGS = [("ZZ", 1.0), ("ZZ", 1.5), ("ZZ", 2.0), ("ZZ", 3.0), ("FR", 3), ("FR", 5), ("FR", 8)]
BREAKS = ["close", "wick"]
RATIOS = [1.0, 1.5, 2.0]
ENTRIES = [0.618, 0.786]
CONFS = ["alle", "open"]
FILL_BARS, HOLD_BARS = 20, 60


@njit(cache=True)
def zigzag(h, l, a, k):
    n = len(h); typ = np.zeros(n, np.int8); ext = np.zeros(n, np.int64)
    pr = np.zeros(n); cf = np.zeros(n, np.int64); m = 0
    mode = 0; hi = h[0]; hi_i = 0; lo = l[0]; lo_i = 0
    for i in range(1, n):
        thr = k * a[i - 1]
        if np.isnan(thr): continue
        if mode >= 0:
            if h[i] > hi:
                hi = h[i]; hi_i = i
            elif hi - l[i] >= thr and (mode == 1 or hi_i > lo_i):
                typ[m] = 1; ext[m] = hi_i; pr[m] = hi; cf[m] = i; m += 1
                mode = -1; lo = l[i]; lo_i = i
                continue
        if mode <= 0:
            if l[i] < lo:
                lo = l[i]; lo_i = i
            elif h[i] - lo >= thr and (mode == -1 or lo_i > hi_i):
                typ[m] = -1; ext[m] = lo_i; pr[m] = lo; cf[m] = i; m += 1
                mode = 1; hi = h[i]; hi_i = i
    return typ[:m], ext[:m], pr[:m], cf[:m]


@njit(cache=True)
def fractal(h, l, N):
    n = len(h); typ = np.zeros(2 * n, np.int8); ext = np.zeros(2 * n, np.int64)
    pr = np.zeros(2 * n); cf = np.zeros(2 * n, np.int64); m = 0
    for i in range(N, n - N):
        ish = True; isl = True
        for j in range(i - N, i + N + 1):
            if j == i: continue
            if j < i:
                if h[j] >= h[i]: ish = False
                if l[j] <= l[i]: isl = False
            else:
                if h[j] > h[i]: ish = False
                if l[j] < l[i]: isl = False
        if ish:
            typ[m] = 1; ext[m] = i; pr[m] = h[i]; cf[m] = i + N; m += 1
        if isl:
            typ[m] = -1; ext[m] = i; pr[m] = l[i]; cf[m] = i + N; m += 1
    # nach Bestaetigungszeit sortieren (gleich wie Extrem-Reihenfolge, da cf = ext + N)
    return typ[:m], ext[:m], pr[:m], cf[:m]


@njit(cache=True)
def find_events(h, l, c, a, typ, ext, pr, cf, wick):
    """Je Bruch: t, dir, Ursprung-Index, Ursprung-Preis, gebrochenes Level, Bruch-Verhaeltnis."""
    n = len(h); m = len(typ)
    et = np.zeros(m, np.int64); ed = np.zeros(m, np.int8); eo = np.zeros(m, np.int64)
    eop = np.zeros(m); elv = np.zeros(m); erat = np.zeros(m); k = 0
    last_hi = -1; last_lo = -1; p = 0
    hi_broken = True; lo_broken = True
    for t in range(1, n):
        while p < m and cf[p] < t:
            if typ[p] == 1:
                last_hi = p; hi_broken = False
            else:
                last_lo = p; lo_broken = False
            p += 1
        if np.isnan(a[t]): continue
        for d in (1, -1):
            j = last_hi if d == 1 else last_lo
            if j < 0: continue
            if (hi_broken if d == 1 else lo_broken): continue
            lvl = pr[j]
            px = (h[t] if d == 1 else l[t]) if wick else c[t]
            if d * (px - lvl) <= 0: continue
            if d == 1: hi_broken = True
            else: lo_broken = True
            s = ext[j] + 1
            if s > t: continue
            oi = s
            for u in range(s, t + 1):
                if d == 1:
                    if l[u] < l[oi]: oi = u
                else:
                    if h[u] > h[oi]: oi = u
            op = l[oi] if d == 1 else h[oi]
            pull = d * (lvl - op)
            if pull <= 0: continue
            ext_px = h[t] if d == 1 else l[t]
            et[k] = t; ed[k] = d; eo[k] = oi; eop[k] = op; elv[k] = lvl
            erat[k] = d * (ext_px - op) / pull; k += 1
    return et[:k], ed[:k], eo[:k], eop[:k], elv[:k], erat[:k]


@njit(cache=True)
def simulate(h, l, c, et, ed, eo, eop, entry, sp, fill_bars, hold):
    m = len(et); n = len(h)
    R = np.full(m, np.nan); fidx = np.full(m, -1, np.int64); E0 = np.zeros(m); rk = np.zeros(m)
    tps = np.array([0.27, 0.62, 1.0])
    for q in range(m):
        t = et[q]; d = ed[q]; origin = eop[q]
        E = -1e18 if d == 1 else 1e18
        for u in range(eo[q], t + 1):
            if d == 1: E = max(E, h[u])
            else: E = min(E, l[u])
        fill = -1
        for u in range(t + 1, min(n, t + 1 + fill_bars)):
            lvl = E - d * entry * abs(E - origin)
            adv = l[u] if d == 1 else h[u]
            if d * (adv - origin) <= 0 and d * (adv - lvl) <= 0:
                fill = u; break          # Level und Stop auf derselben Kerze -> zaehlt als Fuellung, Stop folgt
            if d * (adv - lvl) <= 0:
                fill = u; break
            if d == 1: E = max(E, h[u])
            else: E = min(E, l[u])
        if fill < 0: continue
        rng = abs(E - origin); lvl = E - d * entry * rng; risk = abs(lvl - origin) + sp
        if risk <= 0: continue
        stop = origin; be = False; done = np.zeros(3, np.bool_); pnl = 0.0
        last = min(n - 1, fill + hold); u = fill
        for u in range(fill, last + 1):
            adv = l[u] if d == 1 else h[u]; fav = h[u] if d == 1 else l[u]
            if d * (adv - stop) <= 0:
                for kk in range(3):
                    if not done[kk]:
                        pnl += (d * (stop - lvl) / risk) / 3; done[kk] = True
                break
            if u > fill:
                for kk in range(3):
                    tp = E + d * tps[kk] * rng
                    if not done[kk] and d * (fav - tp) >= 0:
                        pnl += (abs(tp - lvl) / risk) / 3; done[kk] = True
                if not be and d * (fav - E) > 0:
                    stop = lvl; be = True
            if done[0] and done[1] and done[2]: break
        for kk in range(3):
            if not done[kk]: pnl += (d * (c[last] - lvl) / risk) / 3
        R[q] = pnl - sp / risk; fidx[q] = fill; E0[q] = E; rk[q] = rng
    return R, fidx, E0, rk


def period_opens(df):
    out = {}
    for name, freq in [("Y", "Y"), ("Q", "Q"), ("M", "M"), ("W", "W-SUN")]:
        out[name] = df.groupby(df.dt.dt.to_period(freq))["open"].transform("first").values
    return out


def run_pair(sym, tf, root):
    df = bp.load(sym, tf, root)
    if df is None or len(df) < 2000: return []
    h, l, c = (df[x].values.astype(np.float64) for x in ["high", "low", "close"])
    a = bp.atr(df).astype(np.float64); sp = bp.spread(sym); dts = df.dt.values
    op = period_opens(df); rows = []
    split = np.where(dts < np.datetime64(IS_END), 0, np.where(dts < np.datetime64(OOS_END), 1, 2))
    for sw, par in SWINGS:
        sv = zigzag(h, l, a, float(par)) if sw == "ZZ" else fractal(h, l, int(par))
        for br in BREAKS:
            et, ed, eo, eop, elv, erat = find_events(h, l, c, a, *sv, br == "wick")
            for entry in ENTRIES:
                R, fidx, E0, rk = simulate(h, l, c, et, ed, eo, eop, entry, sp, FILL_BARS, HOLD_BARS)
                ok = fidx >= 0
                fi = np.where(ok, fidx, 0)
                z1 = E0 - ed * 0.618 * rk; z2 = E0 - ed * 0.786 * rk
                lo_, hi_ = np.minimum(z1, z2), np.maximum(z1, z2)
                conf = np.zeros(len(et), bool)
                for nm in ["Y", "Q", "M", "W"]:
                    v = op[nm][fi]; conf |= (v >= lo_) & (v <= hi_)
                sp_ev = split[et]
                for rmin in RATIOS:
                    for cf_ in CONFS:
                        sel = ok & (erat >= rmin) & (conf if cf_ == "open" else True)
                        for s_ in (0, 1, 2):
                            r = R[sel & (sp_ev == s_)]
                            rows.append(dict(sym=sym, tf=tf, swing=f"{sw}{par:g}", brk=br, ratio=rmin,
                                             entry=entry, conf=cf_, split=["IS", "OOS", "HO"][s_],
                                             n=len(r), sumR=r.sum(), meanR=r.mean() if len(r) else np.nan,
                                             sdR=r.std(ddof=1) if len(r) > 1 else np.nan,
                                             win=(r > 0).mean() if len(r) else np.nan,
                                             sum_pos=r[r > 0].sum(), sum_neg=-r[r < 0].sum()))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tf", default="H4")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    outdir = os.path.join(a.root, "bos_research", "results"); os.makedirs(outdir, exist_ok=True)
    pairs = a.pairs.split(",") if a.pairs else bp.FX
    rows = []
    for s in pairs:
        t0 = time.time(); r = run_pair(s, a.tf, a.root); rows += r
        print(f"{a.tf} {s}: {len(r)} Zeilen, {time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{outdir}/grid_{a.tf}.csv.gz", index=False)


if __name__ == "__main__":
    main()
