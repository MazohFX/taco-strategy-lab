"""
Regelsuche ohne Datenschnueffeln: Jede Regel = Strategie-Familie x Swing x Bruchart x bis zu 3 Filter.
Auswahl NUR auf IS (2015-2020, gepoolt ueber alle 28 Paare - eine Regel fuer alle Paare),
danach unveraendert OOS (2021-2023) und Holdout (2024-2026).
Rollover 20-23 UTC ausgeschlossen (Spread dort real viel hoeher als modelliert).
"""
import sys, itertools, numpy as np, pandas as pd
tf = sys.argv[1]
d = pd.read_parquet(f"bos_research/results/ctx_{tf}.parquet")
d = d[~d.hour.isin([20, 21, 22, 23])].reset_index(drop=True)
d["split"] = np.where(d.t < "2021-01-01", 0, np.where(d.t < "2024-01-01", 1, 2))
F = {
    "Daily-Open-Seite": d.sideD == 1, "Wochen-Open-Seite": d.sideW == 1, "Monats-Open-Seite": d.sideM == 1,
    "HTF-BOS gleich": d.htf_bos == 1, "HTF-EMA200 gleich": d.htf_ema == 1, "Struktur HH/HL": d.hhl == 1,
    "Key Level <=0.25ATR": d.kl_dist <= 0.25, "Key Level <=0.1ATR": d.kl_dist <= 0.1,
    "Verdraengung >=0.5ATR": d.body >= 0.5, "Verdraengung >=1ATR": d.body >= 1.0,
    "Bruch-Verh >=1.5": d.ratio >= 1.5, "POC in Zone": d.get("poc", 0) == 1,
    "Open in Zone": (d[["W", "M", "Q", "Y"]].fillna(0).sum(axis=1) > 0) if "W" in d else False,
    "London/NY 7-16 UTC": d.hour.between(7, 16),
}
F = {k: np.asarray(v, bool) for k, v in F.items()}
rows = []
for fam in ["fib", "mom", "sweep"]:
    for rcol in (["R"] if fam == "fib" else ["R", "R2"]):
        base_f = (d.fam == fam).values
        R = d[rcol].values; sp = d.split.values
        for sw in d.swing.cat.categories:
            for br in (["close", "wick"] if fam == "fib" else (["close"] if fam == "mom" else ["sweep"])):
                b0 = base_f & (d.swing == sw).values & (d.brk == br).values & ~np.isnan(R)
                keys = [k for k in F if not (fam == "sweep" and k in ("Struktur HH/HL", "Bruch-Verh >=1.5", "POC in Zone", "Open in Zone"))
                        and not (fam == "mom" and k in ("POC in Zone", "Open in Zone"))]
                for kk in range(0, 4):
                    for combo in itertools.combinations(keys, kk):
                        m = b0.copy()
                        for c in combo: m &= F[c]
                        rec = dict(tf=tf, fam=fam, exit=("Fib-TPs" if fam == "fib" else ("1R" if rcol == "R" else "2R")),
                                   swing=sw, brk=br, filters=" + ".join(combo) or "-")
                        for s, nm in enumerate(["IS", "OOS", "HO"]):
                            r = R[m & (sp == s)]
                            rec[f"n_{nm}"] = len(r); rec[f"ER_{nm}"] = r.mean() if len(r) else np.nan
                            rec[f"PF_{nm}"] = r[r > 0].sum() / max(1e-9, -r[r < 0].sum()) if len(r) else np.nan
                            rec[f"WR_{nm}"] = (r > 0).mean() if len(r) else np.nan
                            if nm == "IS": rec["t_IS"] = r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 30 else np.nan
                        rows.append(rec)
res = pd.DataFrame(rows)
res.to_parquet(f"bos_research/results/rules_{tf}.parquet", index=False)
print(tf, "Regeln getestet:", len(res))
