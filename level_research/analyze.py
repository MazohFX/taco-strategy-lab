"""Edge-Test aller Levels (einzeln + 2er-Konfluenz) gegen gematchte Zufalls-Levels.

Edge je Event = R - Baseline(Zufalls-Levels mit gleichem Paar, Typ, Seite, Abstand zum DO, Tageszeit, Periode).
SE geclustert nach Kalenderdatum (über alle Paare). IS 2015-2020, OOS 2021-2023, Holdout 2024+.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RES = Path(__file__).resolve().parent / "results"
NAMES = pd.read_csv(RES / "level_names.csv", header=None)[0].tolist()
TYPES = {0: "ABPRALL", 1: "BRUCH_M15", 2: "BRUCH_H1", 3: "BRUCH_H4"}
PAIRS = sorted(p.stem[3:] for p in RES.glob("ev_*.parquet"))


def period(year):
    return np.where(year <= 2020, 0, np.where(year <= 2023, 1, 2)).astype(np.int8)


def load_all():
    evs = []
    for pi, p in enumerate(PAIRS):
        ev = pd.read_parquet(RES / f"ev_{p}.parquet")
        ct = pd.read_parquet(RES / f"ct_{p}.parquet")
        dates = pd.read_parquet(RES / f"dates_{p}.parquet")
        ev["date"] = (dates["date"].values[ev["day"].values].astype("datetime64[D]").astype(np.int64)).astype(np.int32)
        ev["per"] = period(ev["year"].values); ct["per"] = period(ct["year"].values)
        cell = ["typ", "per", "side", "dbin", "hb"]
        for k in ("05", "10"):
            fine = ct.groupby(cell)[f"R{k}"].agg(["mean", "count"])
            coarse = ct.groupby(["typ", "per", "dbin"])[f"R{k}"].mean().rename("cm")
            ev = ev.merge(fine.rename(columns={"mean": f"bf{k}", "count": f"bn{k}"}), on=cell, how="left")
            ev = ev.merge(coarse.rename(f"bc{k}"), left_on=["typ", "per", "dbin"], right_index=True, how="left")
            base = np.where(ev[f"bn{k}"].fillna(0) >= 50, ev[f"bf{k}"], ev[f"bc{k}"])
            ev[f"E{k}"] = (ev[f"R{k}"] - base).astype(np.float32)
            ev.drop(columns=[f"bf{k}", f"bn{k}", f"bc{k}"], inplace=True)
        ev["pi"] = np.int8(pi)
        evs.append(ev.drop(columns=["pair", "day", "year", "hb", "dbin"]))
    return pd.concat(evs, ignore_index=True)


def cluster_stats(df, keys, val):
    """Mittelwert + datums-geclusterte SE je Gruppe."""
    g = df.groupby(keys + ["date"], observed=True)[val].agg(["sum", "count"]).reset_index()
    tot = g.groupby(keys, observed=True)[["sum", "count"]].sum()
    tot["mean"] = tot["sum"] / tot["count"]
    g = g.merge(tot["mean"], left_on=keys, right_index=True)
    g["dev2"] = (g["sum"] - g["mean"] * g["count"]) ** 2
    v = g.groupby(keys, observed=True)["dev2"].sum()
    tot["se"] = np.sqrt(v) / tot["count"]
    tot["t"] = tot["mean"] / tot["se"]
    return tot[["count", "mean", "se", "t"]]


def summarize(df, keys):
    """IS/OOS/HO-Statistik für Edge (E05,E10) und Netto-R in Edge-Richtung."""
    out = []
    for k in ("05", "10"):
        parts = []
        for per, tag in [(0, "IS"), (1, "OOS"), (2, "HO")]:
            sub = df[df["per"] == per]
            s = cluster_stats(sub, keys, f"E{k}").add_prefix(f"{tag}_")
            r = sub.groupby(keys, observed=True).agg(**{f"{tag}_R": (f"R{k}", "mean"), f"{tag}_C": (f"C{k}", "mean")})
            parts += [s, r]
        t = pd.concat(parts, axis=1)
        t["k"] = float(k) / 10
        out.append(t.reset_index())
    return pd.concat(out, ignore_index=True)


def confluence_summaries(ev, typ):
    """Konfluenz (Level a + b innerhalb TOL) – je Paar/Tag nur das früheste Event; chunkweise je a."""
    sub = ev[ev["typ"] == typ]
    lev = sub["level"].values.astype(np.int64)
    m0 = sub["m0"].values; m1 = sub["m1"].values
    keep = ["pi", "date", "bar", "per", "R05", "R10", "E05", "E10", "C05", "C10"]
    base = {c: sub[c].values for c in keep}
    bitcache = {}
    def bit(j):
        if j not in bitcache:
            b = ((m0 >> np.uint64(j)) & np.uint64(1)) if j < 64 else ((m1 >> np.uint64(j - 64)) & np.uint64(1))
            bitcache[j] = b.astype(bool)
        return bitcache[j]
    out = []; nrows = 0
    for a in range(len(NAMES)):
        isa = lev == a
        parts = []
        for j in range(a + 1, len(NAMES)):
            sel = np.flatnonzero(isa & bit(j))          # Event von a, b=j nahe
            if len(sel):
                parts.append((sel, j))
        sel_b = np.flatnonzero(bit(a) & (lev > a))      # Event von b>a, a nahe
        if len(sel_b):
            parts.append((sel_b, None))
        if not parts:
            continue
        idx = np.concatenate([p[0] for p in parts])
        keyb = np.concatenate([np.full(len(p[0]), p[1]) if p[1] is not None else lev[p[0]] for p in parts])
        f = {c: v[idx] for c, v in base.items()}
        f["key"] = (a * 128 + keyb).astype(np.int32)
        cf = pd.DataFrame(f).sort_values(["key", "pi", "date", "bar"]).drop_duplicates(["key", "pi", "date"])
        nrows += len(cf)
        s = summarize(cf, ["key"])
        s["typ"] = typ
        out.append(s)
    print("  conf rows", typ, nrows, flush=True)
    return pd.concat(out, ignore_index=True)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def label(row):
    if row["kind"] != "konfluenz":
        return NAMES[int(row["level"])]
    a, b = divmod(int(row["key"]), 128)
    return f"{NAMES[a]} + {NAMES[b]}"


if __name__ == "__main__":
    import time
    t0 = time.time()
    ev = load_all()
    print("events", len(ev), f"{time.time()-t0:.0f}s", flush=True)
    single = summarize(ev, ["level", "typ"]); single["kind"] = "einzeln"
    perpair = summarize(ev, ["pi", "level", "typ"]); perpair["kind"] = "je_paar"
    print("single done", f"{time.time()-t0:.0f}s", flush=True)
    conf = []
    for typ in TYPES:
        s = confluence_summaries(ev, typ)
        s["kind"] = "konfluenz"
        conf.append(s)
        print("conf", typ, f"{time.time()-t0:.0f}s", flush=True)
    allt = pd.concat([single] + conf, ignore_index=True)
    allt.to_parquet(RES / "tests_pooled.parquet")
    perpair.to_parquet(RES / "tests_perpair.parquet")
    print("saved", f"{time.time()-t0:.0f}s")
