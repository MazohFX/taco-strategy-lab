"""Events (Abprall / Durchbruch) und Barrier-Outcomes für alle Levels + Zufalls-Kontroll-Levels."""
import sys
import numpy as np
import pandas as pd
import numba as nb
from pathlib import Path
import levels as lv

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
KS = (0.5, 1.0)
HOR = {0.5: 96, 1.0: 192}          # M15-Bars (24 h / 48 h)
N_CTRL = 200
TOL = 0.10                          # Konfluenz-Toleranz in ATR
DIST_BINS = np.array([0, .1, .25, .5, .75, 1, 1.5, 2, 3, 5, np.inf])
TYPES = {0: "ABPRALL", 1: "BRUCH_M15", 2: "BRUCH_H1", 3: "BRUCH_H4"}


@nb.njit(cache=True)
def bounce_events(L, o, h, l, day, allowed):
    n = len(L)
    ei = np.empty(n, np.int64); es = np.empty(n, np.int8); k = 0
    marked = -1
    for i in range(1, n):
        Li = L[i]
        if np.isnan(Li) or day[i] == marked or not allowed[i]:
            continue
        if not (l[i] <= Li <= h[i]):
            continue
        marked = day[i]                      # erste Berührung des Tages
        if l[i - 1] > Li and o[i] >= Li:
            ei[k] = i; es[k] = 1; k += 1      # von oben → Support → Long
        elif h[i - 1] < Li and o[i] <= Li:
            ei[k] = i; es[k] = -1; k += 1     # von unten → Resistance → Short
    return ei[:k], es[:k]


@nb.njit(cache=True)
def break_events(L, c, day, ends, allowed):
    m = len(ends)
    ei = np.empty(m, np.int64); es = np.empty(m, np.int8); k = 0
    marked = -1
    for q in range(1, m):
        e = ends[q]; p = ends[q - 1]
        Le = L[e]
        if np.isnan(Le) or day[e] == marked or not allowed[e]:
            continue
        sp = c[p] - Le; sn = c[e] - Le
        if sp == 0 or sn == 0:
            continue
        if (sp > 0) != (sn > 0):
            marked = day[e]
            ei[k] = e; es[k] = 1 if sn > 0 else -1; k += 1
    return ei[:k], es[:k]


@nb.njit(cache=True)
def race(idx, entry, d, A, k, H, inclusive, h, l, c):
    n = len(c); m = len(idx)
    out = np.empty(m, np.float32)
    for q in range(m):
        i = idx[q]; e = entry[q]; dist = k * A[q]
        if not (dist > 0):
            out[q] = np.nan; continue
        T = e + d[q] * dist; S = e - d[q] * dist
        res = np.nan
        if inclusive:
            if (d[q] > 0 and l[i] <= S) or (d[q] < 0 and h[i] >= S):
                res = -1.0
        j = i + 1; jm = min(i + H, n - 1)
        while np.isnan(res) and j <= jm:
            if d[q] > 0:
                hs = l[j] <= S; ht = h[j] >= T
            else:
                hs = h[j] >= S; ht = l[j] <= T
            if hs:
                res = -1.0
            elif ht:
                res = 1.0
            j += 1
        if np.isnan(res):
            r = d[q] * (c[jm] - e) / dist
            res = min(1.0, max(-1.0, r))
        out[q] = res
    return out


def _events_for_level(Lrow, m, ends_by_tf):
    o, h, l, c, day = m["o"], m["h"], m["l"], m["c"], m["day"]
    res = []
    al = m["allowed"]
    bi, bs = bounce_events(Lrow, o, h, l, day, al)
    res.append((0, bi, bs, Lrow[bi]))
    for t, ends in ends_by_tf.items():
        ei, es = break_events(Lrow, c, day, ends, al)
        res.append((t, ei, es, c[ei]))
    return res


def run_pair(pair, seed=0):
    m = lv.load_m15(pair)
    names, Lmat, Ab, DOb, A, dO = lv.build_levels(pair, m)
    h, l, c, day = m["h"], m["l"], m["c"], m["day"]
    n = len(c)
    # Rollover-Fenster 16:45-18:00 NY: keine Events (Spread-Spitzen, Eröffnungs-Gaps)
    m["allowed"] = ~((m["ny_h"] == 17) | ((m["ny_h"] == 16) & (m["ny_m"] >= 45)))
    ends_by_tf = {1: np.arange(n, dtype=np.int64)}
    for t, hrs in [(2, 1), (3, 4)]:
        gi = lv._tf_index(m, hrs)
        ends_by_tf[t] = np.flatnonzero(np.r_[gi[1:] != gi[:-1], True]).astype(np.int64)
    year = pd.to_datetime(m["tdate"]).year.values.astype(np.int16)
    sp = lv.spread_price(pair)

    def outcomes(idx, side, entry, typ):
        a = Ab[idx]
        cols = {}
        for k in KS:
            cols[f"R{int(k*10):02d}"] = race(idx, entry.astype(np.float64), side.astype(np.float64), a, k,
                                           HOR[k], typ == 0, h, l, c)
            cols[f"C{int(k*10):02d}"] = (sp / (k * a)).astype(np.float32)
        return cols

    # --- echte Levels
    frames = []
    for li in range(len(names)):
        for typ, idx, side, entry in _events_for_level(Lmat[li], m, ends_by_tf):
            if len(idx) == 0:
                continue
            a = Ab[idx]
            ok = a > 0
            idx, side, entry = idx[ok], side[ok], entry[ok]
            lvl = Lmat[li, idx]
            dist = np.abs(lvl - DOb[idx]) / Ab[idx]
            f = dict(level=np.full(len(idx), li, np.int16), typ=np.full(len(idx), typ, np.int8),
                     bar=idx.astype(np.int32), day=day[idx].astype(np.int32), year=year[idx],
                     side=side, hb=m["hb"][idx].astype(np.int8),
                     dbin=(np.searchsorted(DIST_BINS, dist, side="right") - 1).astype(np.int8))
            f.update(outcomes(idx, side, entry, typ))
            # Konfluenz-Maske: andere Levels innerhalb TOL*ATR
            near = np.abs(Lmat[:, idx] - lvl[None, :]) <= TOL * Ab[idx][None, :]
            near[li] = False
            m0 = np.zeros(len(idx), np.uint64); m1 = np.zeros(len(idx), np.uint64)
            for j in np.flatnonzero(near.any(axis=1)):
                if j < 64:
                    m0 |= near[j].astype(np.uint64) << np.uint64(j)
                else:
                    m1 |= near[j].astype(np.uint64) << np.uint64(j - 64)
            f["m0"], f["m1"] = m0, m1
            frames.append(pd.DataFrame(f))
    ev = pd.concat(frames, ignore_index=True)
    ev.insert(0, "pair", pair)

    # --- Kontroll-Levels: DO + v*ATR, v ~ U(-5,5), täglich neu
    rng = np.random.default_rng(seed + sum(map(ord, pair)) * 7919)
    nd = len(A)
    cframes = []
    for cj in range(N_CTRL):
        v = rng.choice([-1.0, 1.0], nd) * 5.0 * rng.uniform(0, 1, nd) ** 2
        Lrow = (dO + v * A)[day]
        for typ, idx, side, entry in _events_for_level(Lrow, m, ends_by_tf):
            a = Ab[idx]; ok = a > 0
            idx, side, entry = idx[ok], side[ok], entry[ok]
            dist = np.abs(Lrow[idx] - DOb[idx]) / Ab[idx]
            f = dict(typ=np.full(len(idx), typ, np.int8), year=year[idx], side=side,
                     hb=m["hb"][idx].astype(np.int8),
                     dbin=(np.searchsorted(DIST_BINS, dist, side="right") - 1).astype(np.int8))
            f.update(outcomes(idx, side, entry, typ))
            cframes.append(pd.DataFrame(f))
    ct = pd.concat(cframes, ignore_index=True)
    ct.insert(0, "pair", pair)
    ev.to_parquet(OUT / f"ev_{pair}.parquet")
    ct.to_parquet(OUT / f"ct_{pair}.parquet")
    pd.Series(names).to_csv(OUT / "level_names.csv", index=False, header=False)
    dates = pd.DataFrame(dict(day=np.arange(nd), date=pd.to_datetime(pd.Series(m["tdate"]).groupby(day).first().values)))
    dates.to_parquet(OUT / f"dates_{pair}.parquet")
    return len(ev), len(ct)


if __name__ == "__main__":
    import time
    for p in sys.argv[1:]:
        t = time.time()
        ne, nc = run_pair(p)
        print(p, ne, nc, f"{time.time()-t:.0f}s", flush=True)
