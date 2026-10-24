"""
Tobis komplettes Setup als mechanischer Backtest.

Nach jedem bestaetigten BOS (bos_pipeline.events, ohne Repaint):
  Fib 1 = Ursprung des Impulsbeins (Swing), Fib 0 = laufendes Extrem bis zur Fuellung.
  Einstieg: Limit am 0.618 (Variante A) bzw. 0.786 (Variante B), verfaellt nach FILL_BARS
            oder wenn Fib 1 vorher erreicht wird.
  Konfluenz (gemessen mit Daten bis zur Vorkerze der Fuellung):
     - Monats-, 3-Monats-, Jahres-Open innerhalb der Zone 0.618-0.786
     - POC des Fixed Range Profile (Monatsbeginn -> Vorkerze) innerhalb der Zone
  Management: Stop Fib 1; drei gleiche Teile mit TP -0.27 / -0.62 / -1;
     Stop auf Break-even, sobald Fib 0 ueberschritten ist (ab der Folgekerze).
     Gleiche Kerze Stop + Ziel -> Stop. Timeout HOLD_BARS. Spread als Kosten.
Holdout ab bos_pipeline.HOLDOUT_START bleibt versiegelt.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp

FILL_BARS, HOLD_BARS = 20, 60
TPS = [0.27, 0.62, 1.0]
ZONE = (0.618, 0.786)
VP_BINS = 40


PERIODS = [("Y", "Y"), ("Q", "Q"), ("M", "M"), ("W", "W-SUN"), ("D", "D")]
CONF = ["M", "Q", "Y", "W", "D", "POC", "VAH", "VAL"]
VA_PCT = 0.68


def period_opens(df):
    dt = df.dt
    out = {}
    for name, freq in PERIODS:
        key = dt.dt.to_period(freq)
        out[name] = df.groupby(key)["open"].transform("first").values
    out["mstart"] = df.groupby(dt.dt.to_period("M")).cumcount().values   # Kerzen seit Monatsbeginn
    return out


def profile(h, l, v, i0, i1, nbins=VP_BINS):
    """Volumenprofil der Kerzen i0..i1 (inkl.): (POC, VAH, VAL). Volumen gleichmaessig ueber die Kerzenspanne."""
    hh, ll, vv = h[i0:i1 + 1], l[i0:i1 + 1], v[i0:i1 + 1]
    if len(hh) == 0: return (np.nan,) * 3
    hi, lo = hh.max(), ll.min()
    if not hi > lo: return (np.nan,) * 3
    w = (hi - lo) / nbins
    a = np.clip(((ll - lo) / w).astype(int), 0, nbins - 1)
    b = np.clip(np.ceil((hh - lo) / w).astype(int), 1, nbins)
    b = np.maximum(b, a + 1)
    diff = np.zeros(nbins + 1)
    np.add.at(diff, a, vv / (b - a)); np.add.at(diff, b, -vv / (b - a))
    prof = np.cumsum(diff)[:nbins]
    k = int(prof.argmax()); tot = prof.sum()
    lo_i, hi_i, acc = k, k, prof[k]
    while acc < VA_PCT * tot and (lo_i > 0 or hi_i < nbins - 1):    # Value Area um den POC erweitern
        up = prof[hi_i + 1] if hi_i < nbins - 1 else -1
        dn = prof[lo_i - 1] if lo_i > 0 else -1
        if up >= dn: hi_i += 1; acc += up
        else: lo_i -= 1; acc += dn
    return lo + (k + .5) * w, lo + (hi_i + 1) * w, lo + lo_i * w


def prev_period_profiles(df, h, l, v):
    """Je Kerze: POC/VAH/VAL der jeweils VORHERIGEN abgeschlossenen Periode (Jahr, Quartal, Monat)."""
    out = {}
    for name in ["Y", "Q", "M"]:
        key = df.dt.dt.to_period(name).values
        starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
        ends = np.r_[starts[1:] - 1, len(df) - 1]
        lv = [profile(h, l, v, s, e, 200) for s, e in zip(starts, ends)]
        arr = np.full((len(df), 3), np.nan)
        for j in range(1, len(starts)):
            arr[starts[j]:ends[j] + 1] = lv[j - 1]
        out[name] = arr
    return out


def run(sym, df, evs, entry):
    o, h, l, c, v = (df[x].values for x in ["open", "high", "low", "close", "vol"])
    a = bp.atr(df); n = len(df); sp = bp.spread(sym); op = period_opens(df)
    pp = prev_period_profiles(df, h, l, v); out = []
    for e in evs:
        t, d = e["t"], e["dir"]
        origin = c[t] - d * e["leg_atr"] * a[t]
        o_idx = t - e["bars_leg"]
        # --- Key Level: lag der Ursprung (Fib 1) an einem Level der hoeheren Zeitebene? ---
        ao = a[o_idx] if not np.isnan(a[o_idx]) else a[t]
        kl = {}
        prof_d, prof_t = np.inf, ""
        for per in ["Y", "Q", "M"]:
            best = np.inf
            for j, nm in enumerate(["POC", "VAH", "VAL"]):
                lvk = pp[per][o_idx, j]
                if np.isnan(lvk): continue
                dd = abs(origin - lvk) / ao
                best = min(best, dd)
                if dd < prof_d: prof_d, prof_t = dd, f"{per}-1 {nm}"
            kl[f"kl_prof_{per}"] = best
        open_d, open_t = np.inf, ""
        for per in ["Y", "Q", "M"]:
            dd = abs(origin - op[per][o_idx]) / ao
            kl[f"kl_open_{per}"] = dd
            if dd < open_d: open_d, open_t = dd, f"{per}-Open"
        E = h[o_idx:t + 1].max() if d == 1 else l[o_idx:t + 1].min()
        fill = None
        for u in range(t + 1, min(n, t + 1 + FILL_BARS)):
            lvl = E - d * entry * abs(E - origin)
            adv = l[u] if d == 1 else h[u]
            if d * (adv - lvl) <= 0: fill = u; break
            E = max(E, h[u]) if d == 1 else min(E, l[u])
        res = dict(sym=sym, dt=e["dt"], dir=d, impulse_atr=abs(E - origin) / a[t], ratio=e["ratio"],
                   hh=e["hh"], hl=e["hl"], filled=int(fill is not None),
                   kl_prof_dist=prof_d, kl_prof_type=prof_t, kl_open_dist=open_d, kl_open_type=open_t, **kl)
        if fill is None: out.append(res); continue
        rng = abs(E - origin)
        zn = sorted([E - d * ZONE[0] * rng, E - d * ZONE[1] * rng])        # 0.618 - 0.786
        zw = sorted([E - d * ZONE[0] * rng, origin])                       # 0.618 - 1
        u0 = fill - 1
        i0 = u0 - op["mstart"][u0]
        cur = profile(h, l, v, i0, u0) if u0 - i0 >= 2 else (np.nan,) * 3   # laufender Monat bis Vorkerze
        lv = {"M": op["M"][fill], "Q": op["Q"][fill], "Y": op["Y"][fill], "W": op["W"][fill], "D": op["D"][fill],
              "POC": cur[0], "VAH": cur[1], "VAL": cur[2]}
        for k_, p_ in lv.items():
            ok = not (p_ is None or np.isnan(p_))
            res[f"{k_}_n"] = int(ok and zn[0] <= p_ <= zn[1])
            res[f"{k_}_w"] = int(ok and zw[0] <= p_ <= zw[1])
        res.update(fill_dt=df.dt[fill], entry_px=E - d * entry * rng, stop_px=origin,
                   M=res["M_n"], Q=res["Q_n"], Y=res["Y_n"], POC=res["POC_n"])
        lvl = E - d * entry * rng; risk = abs(lvl - origin) + sp
        tps = [E + d * x * rng for x in TPS]
        stop, be, done, pnl = origin, False, [False] * 3, 0.0
        last = min(n - 1, fill + HOLD_BARS)
        for u in range(fill, last + 1):
            adv, fav = (l[u], h[u]) if d == 1 else (h[u], l[u])
            if d * (adv - stop) <= 0:                      # Stop / Break-even zuerst
                for k in range(3):
                    if not done[k]: pnl += (d * (stop - lvl) / risk) / 3; done[k] = True
                break
            if u > fill:
                for k in range(3):
                    if not done[k] and d * (fav - tps[k]) >= 0:
                        pnl += (abs(tps[k] - lvl) / risk) / 3; done[k] = True
                if not be and d * (fav - E) > 0: stop, be = lvl, True
            if all(done): break
        for k in range(3):
            if not done[k]: pnl += (d * (c[last] - lvl) / risk) / 3
        res.update(R=pnl - sp / risk, be=int(be), hold_bars=u - fill)
        out.append(res)
    return out


def summary(f, label):
    if len(f) == 0: return f"{label:<38} n=0"
    yrs = (f.dt.max() - f.dt.min()).days / 365 if len(f) > 1 else 1
    se = f.R.std() / np.sqrt(len(f))
    return (f"{label:<38} n={len(f):5d}  ({len(f)/max(yrs,1):5.1f}/Jahr)  Gewinn%={100*(f.R>0).mean():5.1f}  "
            f"E[R]={f.R.mean():+.3f} +/- {se:.3f}  Summe={f.R.sum():+7.1f}R")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tf", default="D1")
    ap.add_argument("--k", type=float, default=bp.K_ATR)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    outdir = os.path.join(a.root, "bos_research", "results"); rep = []
    data = {}
    for s in bp.FX:
        df = bp.load(s, a.tf, a.root)
        if df is None or len(df) < 500: continue
        data[s] = (df, bp.events(s, df, a.k))
    rep.append(f"# Komplettes Setup, {a.tf}, Swings {a.k:g}xATR, {len(data)} Paare, Holdout ab {bp.HOLDOUT_START} versiegelt")
    for entry in [0.618, 0.786]:
        rows = []
        for s, (df, evs) in data.items(): rows += run(s, df, evs, entry)
        ev = pd.DataFrame(rows); ev.to_csv(f"{outdir}/full_setup_{a.tf}_k{a.k:g}_e{entry}.csv", index=False)
        d = ev[(ev.dt < bp.HOLDOUT_START)]; f = d[d.filled == 1]
        rep += [f"\n## Einstieg {entry}   (BOS: {len(d)}, gefuellt: {len(f)})",
                summary(f, "Alle gefuellten Trades"),
                summary(f[f.M == 1], "Monats-Open in Zone"),
                summary(f[f.Q == 1], "3-Monats-Open in Zone"),
                summary(f[f.Y == 1], "Jahres-Open in Zone"),
                summary(f[(f.M + f.Q + f.Y) >= 1], "irgendein Open in Zone"),
                summary(f[f.POC == 1], "POC in Zone"),
                summary(f[(f.POC == 1) & ((f.M + f.Q + f.Y) >= 1)], "POC + Open in Zone (bestes Szenario)"),
                summary(f[(f.POC == 0) & ((f.M + f.Q + f.Y) == 0)], "keine Konfluenz"),
                summary(f[(f.hh == 1) & (f.hl == 1)], "nur mit dem Trend (HH+HL)")]
        best = f[(f.POC == 1) | ((f.M + f.Q + f.Y) >= 1)]
        rep.append("\nKonfluenz (POC oder Open) nach Impulsgroesse in ATR:")
        for lo, hi in [(0, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 99)]:
            rep.append(summary(best[best.impulse_atr.between(lo, hi, inclusive="left")], f"  Impuls {lo}-{hi} ATR"))
        rep.append("\nStabilitaet 'alle Trades' / 'POC oder Open':")
        for y0, y1 in [(2006, 2012), (2013, 2018), (2019, 2023)]:
            sel = lambda x: x[x.dt.dt.year.between(y0, y1)]
            rep += [summary(sel(f), f"  {y0}-{y1} alle"), summary(sel(best), f"  {y0}-{y1} POC oder Open")]
    txt = "\n".join(rep); open(f"{outdir}/report_full_setup_{a.tf}_k{a.k:g}.txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
