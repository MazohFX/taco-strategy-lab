"""
Hypothesen-Scanner: testet alle Filterkombinationen der Checkliste automatisch und
trennt echte Kandidaten von Zufallstreffern.

Ablauf (Daten ab HOLDOUT_START bleiben versiegelt):
  1. Entdeckung   (erste ~60 % der Zeit): jede Kombination wird bewertet, einseitiger t-Test E[R] > 0.
     Benjamini-Hochberg-Korrektur ueber ALLE getesteten Kombinationen (FDR q).
  2. Bestaetigung (restliche ~40 %): nur die Ueberlebenden aus 1 werden auf unbekannten Daten
     erneut getestet, wieder BH-korrigiert.
  3. Robustheit: Anteil Paare und Jahre mit positivem E[R], Nachbar-Parameter (Swing-Schwelle).
Ergebnis: Rangliste mit Ampel. Eine Kombination ist nur dann ein Kandidat, wenn sie ALLE Stufen besteht.
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp  # noqa: E402
import full_setup as fs  # noqa: E402

K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
ENTRY_GRID = [0.618, 0.786]
KL_OPTIONS = {
    "kein Key Level": None,
    "Jahr-1 Profil": ["kl_prof_Y"], "Quartal-1 Profil": ["kl_prof_Q"], "Monat-1 Profil": ["kl_prof_M"],
    "Jahres-Open": ["kl_open_Y"], "Quartals-Open": ["kl_open_Q"], "Monats-Open": ["kl_open_M"],
    "Profil J/Q": ["kl_prof_Y", "kl_prof_Q"], "Opens J/Q": ["kl_open_Y", "kl_open_Q"],
    "Profil+Opens J/Q": ["kl_prof_Y", "kl_prof_Q", "kl_open_Y", "kl_open_Q"],
}
TOL_GRID = [0.25, 0.5, 1.0]
STRUCT = ["alle", "mit Trend", "CHoCH"]
CONF_NAMES = {"M": "Monats-Open", "Q": "3M-Open", "Y": "Jahres-Open", "W": "Wochen-Open", "D": "Tages-Open",
              "POC": "POC", "VAH": "VAH", "VAL": "VAL"}
IMP_MIN = [0.0, 2.0, 3.0, 4.0]


def p_one_sided(mean, sd, n):
    if n < 2 or not sd > 0:
        return 1.0
    t = mean / (sd / math.sqrt(n))
    return 0.5 * math.erfc(t / math.sqrt(2))


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: True = signifikant bei False Discovery Rate q."""
    p = np.asarray(p, float); m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p); ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    ok = ranked <= thresh
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    res = np.zeros(m, bool); res[order[:k]] = True
    return res


def load_trades(tf, root, pairs=None, ks=K_GRID, entries=ENTRY_GRID, progress=None):
    data = {}
    for s in (pairs or bp.FX):
        df = bp.load(s, tf, root)
        if df is not None and len(df) >= 500:
            data[s] = df
    out = {}
    combos = list(itertools.product(ks, entries)); i = 0
    for k in ks:
        evs = {s: bp.events(s, df, k) for s, df in data.items()}
        for e in entries:
            rows = []
            for s, df in data.items():
                rows += fs.run(s, df, evs[s], e)
            t = pd.DataFrame(rows)
            out[(k, e)] = t[(t.filled == 1) & (t.dt < bp.HOLDOUT_START)].reset_index(drop=True) if len(t) else t
            i += 1
            if progress: progress(i / len(combos), f"Swing {k} × ATR, Einstieg {e}")
    return out


def scan(trades: dict, split_date: str, min_n_disc=100, min_n_val=30, q=0.10):
    rows = []
    for (k, e), f in trades.items():
        if f is None or len(f) == 0:
            continue
        R = f.R.values; disc = (f.fill_dt < split_date).values; val = ~disc
        kl_m = {}
        for name, cols in KL_OPTIONS.items():
            if cols is None:
                kl_m[(name, None)] = np.ones(len(f), bool)
            else:
                for tol in TOL_GRID:
                    m = np.zeros(len(f), bool)
                    for c in cols:
                        m |= (f[c].values <= tol)
                    kl_m[(name, tol)] = m
        st_m = {"alle": np.ones(len(f), bool),
                "mit Trend": ((f.hh == 1) & (f.hl == 1)).values,
                "CHoCH": ((f.hh == 0) | (f.hl == 0)).values}
        conf_m = {("keine", None): np.ones(len(f), bool)}
        for z in ("n", "w"):
            cnt = f[[f"{c}_{z}" for c in CONF_NAMES]].sum(axis=1).values
            for mn in (1, 2, 3):
                conf_m[(f"mind. {mn} beliebige", z)] = cnt >= mn
            for c, lab in CONF_NAMES.items():
                conf_m[(lab, z)] = f[f"{c}_{z}"].values == 1
        imp_m = {x: f.impulse_atr.values >= x for x in IMP_MIN}
        for (kln, tol), m1 in kl_m.items():
            for stn, m2 in st_m.items():
                m12 = m1 & m2
                for imp, m3 in imp_m.items():
                    m123 = m12 & m3
                    if (m123 & disc).sum() < min_n_disc:
                        continue
                    for (cn, z), m4 in conf_m.items():
                        m = m123 & m4
                        md = m & disc
                        nd = int(md.sum())
                        if nd < min_n_disc:
                            continue
                        rd = R[md]; mv = m & val; rv = R[mv]
                        rows.append(dict(k=k, entry=e, key_level=kln, tol=tol, struktur=stn, impuls_min=imp,
                                         konfluenz=cn, zone={"n": "0.618–0.786", "w": "0.618–1", None: "–"}[z],
                                         n_disc=nd, er_disc=rd.mean(), p_disc=p_one_sided(rd.mean(), rd.std(ddof=1), nd),
                                         n_val=int(mv.sum()), er_val=rv.mean() if len(rv) else np.nan,
                                         p_val=p_one_sided(rv.mean(), rv.std(ddof=1), len(rv)) if len(rv) > 1 else 1.0,
                                         _mask_id=(k, e, kln, tol, stn, imp, cn, z)))
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    # identische Trade-Mengen (z. B. Impuls >= 0 und >= 2 bei Swing 2 x ATR) nur einmal zaehlen - einfachste Variante behalten
    res = res.drop_duplicates(subset=["k", "entry", "n_disc", "er_disc", "n_val", "er_val"], keep="first").reset_index(drop=True)
    res["bh_disc"] = bh(res.p_disc.values, q)
    surv = res.bh_disc & (res.n_val >= min_n_val)
    res["bh_val"] = False
    if surv.any():
        res.loc[surv, "bh_val"] = bh(res.loc[surv, "p_val"].values, q) & (res.loc[surv, "er_val"].values > 0)
    info = dict(n_tested=len(res), expected_false_at_05=round(0.05 * len(res)),
                n_raw_p05=int((res.p_disc < 0.05).sum()), n_bh_disc=int(res.bh_disc.sum()),
                n_bh_val=int(res.bh_val.sum()))
    return res, info


def mask_for(f, mid):
    k, e, kln, tol, stn, imp, cn, z = mid
    m = np.ones(len(f), bool)
    cols = KL_OPTIONS[kln]
    if cols is not None:
        mm = np.zeros(len(f), bool)
        for c in cols:
            mm |= (f[c].values <= tol)
        m &= mm
    if stn == "mit Trend":
        m &= ((f.hh == 1) & (f.hl == 1)).values
    elif stn == "CHoCH":
        m &= ((f.hh == 0) | (f.hl == 0)).values
    m &= f.impulse_atr.values >= imp
    if cn != "keine":
        if cn.startswith("mind."):
            mn = int(cn.split()[1])
            m &= f[[f"{c}_{z}" for c in CONF_NAMES]].sum(axis=1).values >= mn
        else:
            c = [c for c, lab in CONF_NAMES.items() if lab == cn][0]
            m &= f[f"{c}_{z}"].values == 1
    return m


def robustness(trades, res_row):
    mid = res_row["_mask_id"]; k, e = mid[0], mid[1]
    f = trades[(k, e)]; sel = f[mask_for(f, mid)]
    by_pair = sel.groupby("sym").R.mean()
    by_year = sel.groupby(sel.fill_dt.dt.year).R.mean()
    neigh = []
    for k2 in K_GRID:
        if abs(k2 - k) == 0.5 and (k2, e) in trades:
            f2 = trades[(k2, e)]; s2 = f2[mask_for(f2, (k2,) + mid[1:])]
            if len(s2) >= 30:
                neigh.append(s2.R.mean())
    return dict(pairs_pos=(by_pair > 0).mean() if len(by_pair) else np.nan, n_pairs=len(by_pair),
                years_pos=(by_year > 0).mean() if len(by_year) else np.nan, n_years=len(by_year),
                neighbours_pos=(np.mean([x > 0 for x in neigh]) if neigh else np.nan), sel=sel)


def verdict(r):
    if r.get("bh_disc") and r.get("bh_val") and r.get("pairs_pos", 0) >= 0.6 and r.get("years_pos", 0) >= 0.6 \
            and (np.isnan(r.get("neighbours_pos", np.nan)) or r.get("neighbours_pos") >= 0.5):
        return "🟢 Kandidat"
    if r.get("bh_disc") and r.get("er_val", -1) > 0:
        return "🟡 Bestätigung schwach"
    if r.get("bh_disc"):
        return "🔴 In Bestätigung durchgefallen"
    return "⚪ Nicht signifikant nach Korrektur"
