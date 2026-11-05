"""Stufe 1: IS-FDR → OOS-Bestätigung → netto profitabel. Holdout wird hier NICHT angeschaut."""
import numpy as np
import pandas as pd
from scipy import stats
from analyze import TYPES, bh, label

RES = "results"


def stage1(path=f"{RES}/tests_pooled.parquet", min_n=300, q=0.05):
    t = pd.read_parquet(path)
    for c in ("level", "key", "pi"):
        if c in t:
            t[c] = t[c].fillna(-1)
    t = t[t.IS_count >= min_n].copy()
    t["p"] = 2 * stats.norm.sf(np.abs(t.IS_t)); t["q"] = bh(t.p.values)
    t["name"] = t.apply(label, axis=1); t["typ_name"] = t.typ.map(TYPES)
    sg = np.sign(t.IS_mean)
    t["dir"] = sg
    t["IS_net"] = sg * t.IS_R - t.IS_C
    t["OOS_net"] = sg * t.OOS_R - t.OOS_C
    t["sig_IS"] = t.q < q
    t["OOS_ok"] = t.sig_IS & (np.sign(t.OOS_mean) == sg) & (t.OOS_t * sg > 1.645) & (t.OOS_mean * sg >= t.IS_mean.abs() / 3)
    t["netto_ok"] = t.OOS_ok & (t.IS_net > 0) & (t.OOS_net > 0)
    return t


if __name__ == "__main__":
    t = stage1()
    print("Tests:", len(t), t.groupby("kind").size().to_dict())
    print("IS p<0.05: %.3f (Zufall erwartet 0.050)" % (t.p < 0.05).mean())
    print("IS FDR q<0.05:", t.sig_IS.sum(), "| + OOS bestätigt:", t.OOS_ok.sum(), "| + netto>0 IS&OOS:", t.netto_ok.sum())
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    cols = ["kind", "name", "typ_name", "k", "dir", "IS_count", "IS_mean", "IS_t", "OOS_count", "OOS_mean", "OOS_t",
            "IS_R", "OOS_R", "IS_C", "IS_net", "OOS_net"]
    print(t[t.OOS_ok].sort_values("IS_t", key=abs, ascending=False)[cols].round(3).head(80).to_string())
    print("\nNETTO-KANDIDATEN")
    print(t[t.netto_ok].sort_values("OOS_net", ascending=False)[cols].round(3).to_string())
    t.to_parquet(f"{RES}/stage1_all.parquet")
