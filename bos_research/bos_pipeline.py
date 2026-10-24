"""
BOS-Validierung: Welche Structure Breaks laufen weiter, welche nicht?

1. ATR-ZigZag ohne Repaint: Ein Swing gilt erst als bestaetigt, wenn der Kurs
   k*ATR vom Extrem zurueckgelaufen ist. Verwendet werden nur Swings, deren
   Bestaetigungskerze VOR der Bruchkerze liegt.
2. BOS: erster Schlusskurs ueber dem letzten bestaetigten Swing-Hoch (bzw. unter
   dem letzten Swing-Tief).
3. Merkmale je Bruch, alles ATR- oder strukturnormiert (paar- und TF-uebergreifend).
4. Labels (Triple Barrier, Einstieg zum Schlusskurs der Bruchkerze, Spread als Kosten):
   - cont : +1,5 ATR vor -1,5 ATR erreicht (reine Folgebewegung, 1 / 0, Timeout = NaN)
   - r1/r2: Ergebnis in R mit Stop am Ursprung des Bruchbeins, Ziel 1R bzw. 2R
5. Auswertung nach Strukturverhaeltnis + Walk-Forward-Modell (LightGBM vs. Logit).
   Daten ab HOLDOUT_START bleiben versiegelt und werden NICHT ausgewertet.

Aufruf:  python3 bos_research/bos_pipeline.py --tf D1
         python3 bos_research/bos_pipeline.py --tf H4   (nach dem Dukascopy-Import)
"""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HOLDOUT_START = "2024-01-01"
K_ATR = 2.0          # ZigZag-Umkehrschwelle in ATR (vor der Auswertung eingefroren)
ATR_N = 14
MED_N = 20           # Anzahl Beine fuer die Strukturgroesse
HORIZON = 30         # max. Kerzen bis Timeout
SYM_BAR = 1.5        # symmetrische Barriere in ATR
FX = ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","EURGBP","EURJPY",
      "EURAUD","EURNZD","EURCAD","EURCHF","GBPJPY","GBPAUD","GBPNZD","GBPCAD","GBPCHF",
      "AUDCAD","AUDCHF","AUDNZD","AUDJPY","NZDCAD","NZDCHF","NZDJPY","CADJPY","CADCHF","CHFJPY"]
SPREAD_PIPS = {"EURUSD":1.0,"USDJPY":1.2,"GBPUSD":1.4,"AUDUSD":1.4,"USDCHF":1.6,"USDCAD":1.6,
               "NZDUSD":1.8,"EURGBP":1.6,"EURJPY":1.8,"GBPJPY":2.6,"AUDJPY":2.0,"NZDJPY":2.4}
BINS = [0, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]


def pip(sym): return 0.01 if sym.endswith("JPY") else 0.0001
def spread(sym): return SPREAD_PIPS.get(sym, 2.8) * pip(sym)


def load(sym, tf, root):
    if tf == "D1":
        p = f"{root}/data/mt5/{sym}.csv"
        if not os.path.exists(p): return None
        df = pd.read_csv(p)
        df["dt"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
    else:
        p = f"{root}/data/dukascopy_intraday/{sym}_{tf}.csv"
        if not os.path.exists(p):
            p += ".gz"                      # komprimierte Version (Online-Repo)
            if not os.path.exists(p): return None
        df = pd.read_csv(p); df["dt"] = pd.to_datetime(df["datetime"])
    df = df[df["dt"].dt.weekday < 5] if tf == "D1" else df
    df = df[(df.high > df.low)].sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    vc = "tick_volume" if "tick_volume" in df else ("ticks" if "ticks" in df else None)
    df["vol"] = df[vc].astype(float) if vc else 1.0
    return df[["dt","open","high","low","close","vol"]]


def atr(df, n=ATR_N):
    pc = df.close.shift()
    tr = np.maximum(df.high - df.low, np.maximum((df.high - pc).abs(), (df.low - pc).abs()))
    return tr.ewm(alpha=1/n, adjust=False).mean().values


def zigzag(h, l, a, k):
    """Liste (typ, extrem_idx, preis, bestaetigt_idx). typ +1 = Hoch, -1 = Tief."""
    piv, mode = [], 0
    hi, hi_i, lo, lo_i = h[0], 0, l[0], 0
    for i in range(1, len(h)):
        thr = k * a[i-1]
        if np.isnan(thr): continue
        if mode >= 0:
            if h[i] > hi: hi, hi_i = h[i], i
            elif hi - l[i] >= thr and (mode == 1 or hi_i > lo_i):
                piv.append((1, hi_i, hi, i)); mode = -1; lo, lo_i = l[i], i; continue
        if mode <= 0:
            if l[i] < lo: lo, lo_i = l[i], i
            elif h[i] - lo >= thr and (mode == -1 or lo_i > hi_i):
                piv.append((-1, lo_i, lo, i)); mode = 1; hi, hi_i = h[i], i
    return piv


def events(sym, df, k=K_ATR):
    o, h, l, c = (df[x].values for x in ["open","high","low","close"])
    a = atr(df); n = len(df); sp = spread(sym)
    sma = pd.Series(c).rolling(200).mean().values
    atr_pct = pd.Series(a).rolling(250).rank(pct=True).values
    piv = zigzag(h, l, a, k)
    out, pi, conf = [], 0, []          # conf = bisher bestaetigte Pivots
    broken = set()
    for t in range(1, n - 1):
        while pi < len(piv) and piv[pi][3] < t:
            conf.append(piv[pi]); pi += 1
        if len(conf) < MED_N + 3 or np.isnan(a[t]) or np.isnan(sma[t]): continue
        for d in (1, -1):
            # letzter bestaetigter Swing in Bruchrichtung
            j = max(i for i in range(len(conf) - 3, len(conf)) if conf[i][0] == d)
            typ, ei, lvl, ci = conf[j]
            if (j, d) in broken: continue
            if not (d * (c[t] - lvl) > 0): continue
            broken.add((j, d))
            prev_same = next((p for p in reversed(conf[:j]) if p[0] == d), None)
            prev_opp = next((p for p in reversed(conf[:j]) if p[0] == -d), None)
            if prev_same is None or prev_opp is None: continue
            seg = slice(ei + 1, t + 1)
            origin = l[seg].min() if d == 1 else h[seg].max()
            o_idx = ei + 1 + (np.argmin(l[seg]) if d == 1 else np.argmax(h[seg]))
            leg = d * (c[t] - origin)
            prices = [p[2] for p in conf[-(MED_N + 1):]]
            med = np.median(np.abs(np.diff(prices)))
            prior_leg = abs(lvl - prev_opp[2])
            rng = h[t] - l[t]
            risk = leg + sp
            ev = dict(sym=sym, dt=df.dt[t], dir=d, t=t,
                      ratio=leg / med, leg_atr=leg / a[t], struct_atr=med / a[t],
                      depth_atr=d * (c[t] - lvl) / a[t], body_atr=d * (c[t] - o[t]) / a[t],
                      close_pos=((c[t] - l[t]) / rng if d == 1 else (h[t] - c[t]) / rng) if rng > 0 else .5,
                      bars_leg=t - o_idx, retrace=(d * (lvl - origin)) / prior_leg if prior_leg > 0 else np.nan,
                      hh=int(d * (lvl - prev_same[2]) > 0), hl=int(d * (origin - prev_opp[2]) > 0),
                      trend200=d * (c[t] - sma[t]) / a[t], atr_pct=atr_pct[t], risk_atr=risk / a[t])
            # ---- Labels ----
            e = c[t]; end = min(n - 1, t + HORIZON)
            up, dn = e + d * SYM_BAR * a[t], e - d * SYM_BAR * a[t]
            cont = np.nan; r1 = r2 = None; stop = origin
            t1, t2 = e + d * risk, e + d * 2 * risk
            for u in range(t + 1, end + 1):
                fav, adv = (h[u], l[u]) if d == 1 else (l[u], h[u])
                hit_stop = d * (adv - stop) <= 0
                if np.isnan(cont):
                    if d * (adv - dn) <= 0: cont = 0
                    elif d * (fav - up) >= 0: cont = 1
                if r1 is None:
                    if hit_stop: r1 = -1.0
                    elif d * (fav - t1) >= 0: r1 = 1.0
                if r2 is None:
                    if hit_stop: r2 = -1.0
                    elif d * (fav - t2) >= 0: r2 = 2.0
                if not np.isnan(cont) and r1 is not None and r2 is not None: break
            mtm = (d * (c[end] - e)) / risk
            ev.update(cont=cont, r1=(r1 if r1 is not None else max(-1, min(1, mtm))) - sp / risk,
                      r2=(r2 if r2 is not None else max(-1, min(2, mtm))) - sp / risk,
                      complete=int(end - t == HORIZON))
            out.append(ev)
    return out


FEATS = ["ratio","leg_atr","struct_atr","depth_atr","body_atr","close_pos","bars_leg",
         "retrace","hh","hl","trend200","atr_pct","risk_atr"]


def bucket_table(d, label):
    g = d.groupby(pd.cut(d.ratio, BINS), observed=True)
    t = pd.DataFrame({"n": g.size(), "P(Folge)": g.cont.mean(), "E[R] 1R": g.r1.mean(),
                      "E[R] 2R": g.r2.mean(), "Stop ATR": g.risk_atr.median()})
    t.index = [f"{iv.left:g}-{iv.right:g}x" for iv in t.index]
    return f"\n### {label}  (n={len(d)})\n" + t.round(3).to_string()


def walk_forward(d):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    d = d.dropna(subset=["cont"] + FEATS).copy(); d["year"] = d.dt.dt.year
    preds = []
    for y in range(d.year.min() + 5, int(HOLDOUT_START[:4])):
        tr, te = d[d.year < y - 0], d[d.year == y]
        # Purging: Ereignisse, deren Label-Horizont ins Testjahr reicht, raus
        tr = tr[tr.dt < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=45)]
        if len(te) < 30: continue
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(tr[FEATS], tr.cont)
        gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8,
                                min_child_samples=80, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, verbose=-1).fit(tr[FEATS], tr.cont)
        te = te.assign(p_lr=lr.predict_proba(te[FEATS])[:, 1], p_gb=gb.predict_proba(te[FEATS])[:, 1])
        preds.append(te)
    p = pd.concat(preds)
    lines = [f"Out-of-sample {p.year.min()}-{p.year.max()}, n={len(p)}, Basisrate P(Folge)={p.cont.mean():.3f}",
             f"AUC Logit {roc_auc_score(p.cont, p.p_lr):.3f} | AUC LightGBM {roc_auc_score(p.cont, p.p_gb):.3f}  (0.5 = Zufall)"]
    for col in ["p_lr", "p_gb"]:
        p["q"] = p.groupby("year")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        g = p.groupby("q")
        lines.append(f"\nQuintile nach {col} (0 = schlechteste, 4 = beste Prognose):\n" +
                     pd.DataFrame({"n": g.size(), "P(Folge)": g.cont.mean(), "E[R] 1R": g.r1.mean(),
                                   "E[R] 2R": g.r2.mean()}).round(3).to_string())
    yr = p.groupby("year").apply(lambda s: s[s.q == 4].r2.mean() - s.r2.mean())
    lines.append(f"\nTop-Quintil (LightGBM) minus Durchschnitt, E[R] 2R je Jahr:\n{yr.round(3).to_string()}")
    gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=80,
                            verbose=-1, importance_type="gain").fit(d[d.dt < HOLDOUT_START][FEATS],
                                                                    d[d.dt < HOLDOUT_START].cont)
    imp = pd.Series(gb.feature_importances_, FEATS).sort_values(ascending=False)
    lines.append("\nMerkmal-Wichtigkeit (Gain, normiert):\n" + (imp / imp.sum()).round(3).to_string())
    return "\n".join(lines), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="D1"); ap.add_argument("--k", type=float, default=K_ATR)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    outdir = os.path.join(a.root, "bos_research", "results"); os.makedirs(outdir, exist_ok=True)
    rows = []
    for s in FX:
        df = load(s, a.tf, a.root)
        if df is None or len(df) < 500: continue
        rows += events(s, df, a.k)
    ev = pd.DataFrame(rows)
    ev.to_csv(f"{outdir}/bos_events_{a.tf}_k{a.k:g}.csv", index=False)
    d = ev[(ev.dt < HOLDOUT_START) & (ev.complete == 1)]
    rep = [f"# BOS-Auswertung {a.tf}, ZigZag {a.k:g} x ATR({ATR_N}), Horizont {HORIZON} Kerzen",
           f"Paare: {ev.sym.nunique()}, Zeitraum {ev.dt.min():%Y-%m-%d} bis {ev.dt.max():%Y-%m-%d}",
           f"Ausgewertet (vor {HOLDOUT_START}): {len(d)} Brueche, Holdout versiegelt: {(ev.dt >= HOLDOUT_START).sum()}",
           f"Basis: P(Folge)={d.cont.mean():.3f}, E[R] 1R={d.r1.mean():.3f}, E[R] 2R={d.r2.mean():.3f}",
           "\n## Strukturverhaeltnis (Bruchbein / Median der letzten 20 Beine)",
           bucket_table(d, "Alle"),
           bucket_table(d[(d.hh == 1) & (d.hl == 1)], "Mit dem Trend (HH + HL)"),
           bucket_table(d[(d.hh == 0) | (d.hl == 0)], "Gegen den Trend / CHoCH")]
    for dec in ["2006-2012", "2013-2018", "2019-2023"]:
        y0, y1 = map(int, dec.split("-"))
        rep.append(bucket_table(d[d.dt.dt.year.between(y0, y1)], f"Stabilitaet {dec}"))
    wf, _ = walk_forward(d)
    rep += ["\n## Walk-Forward-Modell (Ziel: P(Folge))", wf]
    txt = "\n".join(rep)
    open(f"{outdir}/report_{a.tf}_k{a.k:g}.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
