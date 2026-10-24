"""
BOS als Filter + Einstieg am 0.618-Retracement des Impulsbeins (Tobis Setup).

Ablauf je BOS (aus bos_pipeline.events, ohne Repaint):
  - Impulsbein: Ursprung (Fib 1) bis laufendes Extrem nach dem Bruch (Fib 0).
  - Limit-Order auf 0.618; das Extrem darf bis zur Fuellung weiterlaufen, der Level
    wird nur mit Kursen bis zur Vorkerze berechnet (kein Lookahead).
  - Order verfaellt nach FILL_BARS Kerzen oder wenn Fib 1 vorher erreicht wird.
  - Stop an Fib 1, Ziele -0.27 und -0.62 (Extension), Timeout HOLD_BARS.
  - Gleiche Kerze trifft Stop und Ziel -> Verlust (konservativ). Spread als Kosten.
Gruppierung nach Bruchgroesse in ATR des jeweiligen Paares.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp

FILL_BARS, HOLD_BARS, ENTRY = 20, 60, 0.618
TARGETS = {"-0.27": 0.27, "-0.62": 0.62}
LEG_BINS = [0, 1, 1.5, 2, 2.5, 3, 4, 6, np.inf]
DEPTH_BINS = [0, 0.25, 0.5, 1, 1.5, 2, np.inf]


def simulate(sym, df, evs):
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    a = bp.atr(df); n = len(df); sp = bp.spread(sym); out = []
    for e in evs:
        t, d = e["t"], e["dir"]
        origin = c[t] - d * e["leg_atr"] * a[t]
        o_idx = t - e["bars_leg"]
        E = h[o_idx:t + 1].max() if d == 1 else l[o_idx:t + 1].min()
        imp0 = abs(E - origin) / a[t]
        res = dict(e, impulse_atr=imp0, filled=0)
        fill_u = None
        for u in range(t + 1, min(n, t + 1 + FILL_BARS)):
            lvl = E - d * ENTRY * abs(E - origin)
            adv = l[u] if d == 1 else h[u]
            if d * (adv - origin) <= 0 and d * (adv - lvl) <= 0:
                fill_u = u; break                      # gefuellt und im selben Bar gestoppt
            if d * (adv - lvl) <= 0:
                fill_u = u; break
            E = max(E, h[u]) if d == 1 else min(E, l[u])
        if fill_u is None:
            out.append(res); continue
        lvl = E - d * ENTRY * abs(E - origin); risk = abs(lvl - origin) + sp
        res.update(filled=1, fill_bars=fill_u - t, impulse_fill_atr=abs(E - origin) / a[t])
        for name, ext in TARGETS.items():
            tgt = E + d * ext * abs(E - origin); rr = abs(tgt - lvl) / risk
            r = None
            for u in range(fill_u, min(n, fill_u + HOLD_BARS)):
                adv, fav = (l[u], h[u]) if d == 1 else (h[u], l[u])
                if d * (adv - origin) <= 0: r = -1.0; break
                if u > fill_u and d * (fav - tgt) >= 0: r = rr; break
            if r is None:
                u = min(n - 1, fill_u + HOLD_BARS - 1)
                r = max(-1.0, d * (c[u] - lvl) / risk)
            res[f"R{name}"] = r - sp / risk
            res[f"rr{name}"] = rr
        out.append(res)
    return out


def table(d, col, bins, label):
    g = d.groupby(pd.cut(d[col], bins), observed=True)
    f = d[d.filled == 1]; gf = f.groupby(pd.cut(f[col], bins), observed=True)
    t = pd.DataFrame({"BOS": g.size(), "Fill %": g.filled.mean() * 100, "Trades": gf.size(),
                      "Win% -0.27": gf["R-0.27"].apply(lambda s: (s > 0).mean() * 100),
                      "E[R] -0.27": gf["R-0.27"].mean(),
                      "Win% -0.62": gf["R-0.62"].apply(lambda s: (s > 0).mean() * 100),
                      "E[R] -0.62": gf["R-0.62"].mean(),
                      "SE": gf["R-0.27"].std() / np.sqrt(gf.size())})
    t.index = [f"{iv.left:g}-{iv.right:g}" for iv in t.index]
    return f"\n### {label}\n" + t.round(2).to_string()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tf", default="D1")
    ap.add_argument("--k", type=float, default=bp.K_ATR)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    rows = []
    for s in bp.FX:
        df = bp.load(s, a.tf, a.root)
        if df is None or len(df) < 500: continue
        rows += simulate(s, df, bp.events(s, df, a.k))
    ev = pd.DataFrame(rows)
    outdir = os.path.join(a.root, "bos_research", "results")
    ev.to_csv(f"{outdir}/fib618_{a.tf}_k{a.k:g}.csv", index=False)
    d = ev[ev.dt < bp.HOLDOUT_START]
    f = d[d.filled == 1]
    rep = [f"# BOS + 0.618-Einstieg, {a.tf}, ZigZag {a.k:g}xATR | Stop Fib 1 | Ziele -0.27 / -0.62",
           f"BOS: {len(d)}, gefuellt: {len(f)} ({len(f)/len(d):.0%}), Holdout ab {bp.HOLDOUT_START} versiegelt",
           f"Alle Trades: E[R] -0.27 = {f['R-0.27'].mean():.3f} (CRV {f['rr-0.27'].median():.2f}), "
           f"E[R] -0.62 = {f['R-0.62'].mean():.3f} (CRV {f['rr-0.62'].median():.2f})",
           table(d, "impulse_atr", LEG_BINS, "Impulsbein beim Bruch (Fib1 -> Extrem) in ATR"),
           table(d, "depth_atr", DEPTH_BINS, "Schluss jenseits des Swing-Levels in ATR"),
           table(d, "ratio", bp.BINS, "Bruchbein / uebliche Struktur"),
           table(d[(d.hh == 1) & (d.hl == 1)], "impulse_atr", LEG_BINS, "Impuls in ATR, nur mit dem Trend"),
           table(d[(d.hh == 0) | (d.hl == 0)], "impulse_atr", LEG_BINS, "Impuls in ATR, gegen Trend / CHoCH")]
    for y0, y1 in [(2006, 2012), (2013, 2018), (2019, 2023)]:
        rep.append(table(d[d.dt.dt.year.between(y0, y1)], "impulse_atr", LEG_BINS, f"Stabilitaet {y0}-{y1}"))
    txt = "\n".join(rep); open(f"{outdir}/report_fib618_{a.tf}_k{a.k:g}.txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
