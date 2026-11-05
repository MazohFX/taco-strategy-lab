"""Auswertung ORB: alle Varianten x Filter je Symbol, IS-FDR -> OOS-Bestätigung -> Holdout."""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

RES = Path(__file__).parent / "results"


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def add_filters(r):
    is_ = r[r.per == "IS"]
    q1, q2 = is_.or_width_atr.quantile([1 / 3, 2 / 3])
    f = {
        "alle": np.ones(len(r), bool),
        "nur_long": r.dir.values > 0,
        "nur_short": r.dir.values < 0,
        "OR_eng": r.or_width_atr.values <= q1,
        "OR_mittel": (r.or_width_atr.values > q1) & (r.or_width_atr.values <= q2),
        "OR_weit": r.or_width_atr.values > q2,
        "mit_Gap": np.sign(r.gap_atr.values) == r.dir.values,
        "gegen_Gap": np.sign(r.gap_atr.values) == -r.dir.values,
        "Gap>0.3ATR_mit": (np.abs(r.gap_atr.values) > 0.3) & (np.sign(r.gap_atr.values) == r.dir.values),
        "mit_Trend": r.trend.values == r.dir.values,
        "gegen_Trend": r.trend.values == -r.dir.values,
    }
    return f


def evaluate(trades):
    out = []
    for sym, r in trades.groupby("sym"):
        r = r.reset_index(drop=True)
        filt = add_filters(r)
        keys = ["or_min", "last_entry", "stop", "target_R"]
        for fname, mask in filt.items():
            sub = r[mask]
            g = sub.groupby(keys + ["per"]).R.agg(["count", "mean", "std"]).unstack("per")
            for (orm, last, st, tg), row in g.iterrows():
                d = dict(sym=sym, filter=fname, or_min=orm, last_entry=last, stop=st, target_R=tg)
                for per in ("IS", "OOS", "HO"):
                    n = row.get(("count", per), np.nan); m = row.get(("mean", per), np.nan); s = row.get(("std", per), np.nan)
                    d[f"{per}_n"] = n; d[f"{per}_R"] = m
                    d[f"{per}_t"] = m / (s / np.sqrt(n)) if n and n > 1 and s > 0 else np.nan
                out.append(d)
    t = pd.DataFrame(out)
    t = t[t.IS_n >= 100].copy()
    t["IS_p"] = stats.norm.sf(t.IS_t)                 # einseitig: Erwartungswert > 0 nach Kosten
    t["q"] = bh(t.IS_p.fillna(1).values)
    t["IS_sig"] = t.q < 0.05
    t["OOS_ok"] = t.IS_sig & (t.OOS_R > 0) & (t.OOS_t > 1.645) & (t.OOS_R >= t.IS_R / 3)
    return t


if __name__ == "__main__":
    tr = pd.read_parquet(RES / "orb_trades.parquet")
    t = evaluate(tr)
    t.to_parquet(RES / "orb_tests.parquet")
    pd.set_option("display.width", 250)
    print("Tests", len(t), "| IS p<0.05:", round((t.IS_p < 0.05).mean(), 3), "| FDR:", t.IS_sig.sum(), "| OOS ok:", t.OOS_ok.sum())
    cols = ["sym", "filter", "or_min", "last_entry", "stop", "target_R", "IS_n", "IS_R", "IS_t", "OOS_n", "OOS_R", "OOS_t"]
    print(t[t.OOS_ok].sort_values("IS_t", ascending=False)[cols].round(3).head(40).to_string())
    print(t.sort_values("IS_t", ascending=False)[cols].round(3).head(15).to_string())
