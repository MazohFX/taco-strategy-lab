"""IS-Top-N Walk-Forward — saubere Trennung von Auswahl (In-Sample) und Test (Out-of-Sample).

Verfahren (pro Fold):
  1. ALLE Grid-Varianten werden NUR auf dem IS-Fenster bewertet und nach dem Optimierungsziel sortiert.
  2. Die Top-N (z.B. 10) werden UNVERÄNDERT auf das folgende OOS-Fenster angewendet.
  3. OOS-Daten werden NIE zur Auswahl benutzt — nur gemessen.

Zusätzlich:
  - Endtest-Zeitraum (Holdout), der in der WFA komplett fehlt und nur einmal am Ende getestet wird.
  - PBO (Probability of Backtest Overfitting) per CSCV nach Bailey/Borwein/López de Prado/Zhu (2015).
  - Deflated Sharpe Ratio nach Bailey/López de Prado (2014).

Performance-Trick: jede Variante wird EINMAL über den ganzen Zeitraum vor dem Holdout gebacktestet.
IS-/OOS-Kennzahlen entstehen dann durch Zerschneiden der Trade-Liste nach Datum. Das ist erlaubt, weil
  - die Entry-Signale nur Vergangenheitsdaten nutzen (kein Look-ahead durch längere Historie) und
  - die Engine risikobasiert sized (qty ∝ Kapital) → "Ret %" je Trade ist unabhängig vom Kontostand.
Nebeneffekt: kein MA-Warm-up-Fehler am Anfang jedes Fensters.
"""
from __future__ import annotations

import math
from itertools import combinations, product
from typing import Callable

import numpy as np
import pandas as pd

NO_FILTER = "Kein Filter (nur Zeit)"


# ════════════════════════════════════════════════════════════════════════════
# Kennzahlen aus einer Liste von Trade-Renditen (in % des Kontos)
# ════════════════════════════════════════════════════════════════════════════
def metrics_from_rets(rets_pct: np.ndarray, years: float) -> dict:
    """Kennzahlen aus Trade-Renditen in % (z.B. 0.8 = +0,8 % aufs Konto), kompoundiert."""
    r = np.asarray(rets_pct, dtype=float) / 100.0
    n = len(r)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "sharpe": 0.0, "max_dd": 0.0, "total_ret": 0.0, "avg_ret": 0.0}
    gp = r[r > 0].sum()
    gl = -r[r <= 0].sum()
    pf = gp / gl if gl > 0 else 9.99
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = (eq / peak - 1).min() * 100
    sd = r.std(ddof=1) if n > 1 else 0.0
    tpy = n / years if years > 0.25 else n
    sharpe = r.mean() / sd * math.sqrt(tpy) if sd > 0 else 0.0
    return {"n": n, "wr": round(float((r > 0).mean() * 100), 1), "pf": round(float(pf), 3),
            "sharpe": round(float(sharpe), 3), "max_dd": round(float(dd), 2),
            "total_ret": round(float((eq[-1] - 1) * 100), 2), "avg_ret": round(float(r.mean() * 100), 3)}


# ════════════════════════════════════════════════════════════════════════════
# Folds
# ════════════════════════════════════════════════════════════════════════════
def build_folds(start: pd.Timestamp, end: pd.Timestamp, oos_months: int,
                mode: str = "anchored", is_months: int = 36) -> list[dict]:
    """mode="anchored": IS immer ab `start` (wachsend), erstes IS = is_months lang.
       mode="rolling":  IS = feste is_months vor jedem OOS-Fenster."""
    folds = []
    oos_start = start + pd.DateOffset(months=int(is_months))
    while True:
        oos_end = oos_start + pd.DateOffset(months=int(oos_months))
        if oos_end > end + pd.Timedelta(days=1):
            break
        is_start = start if mode == "anchored" else oos_start - pd.DateOffset(months=int(is_months))
        folds.append({"is_start": is_start, "is_end": oos_start, "oos_start": oos_start, "oos_end": oos_end})
        oos_start = oos_end
    return folds


# ════════════════════════════════════════════════════════════════════════════
# Haupt-Berechnung
# ════════════════════════════════════════════════════════════════════════════
def run_all_combos(df: pd.DataFrame, base_params: dict, combos: list[dict],
                   engine: Callable, progress_cb: Callable | None = None) -> dict:
    """Backtestet jede Kombination einmal auf df. Rückgabe: key -> DataFrame[Zeit, Ret %]."""
    out = {}
    for i, c in enumerate(combos):
        p = {**base_params, **c["params"]}
        try:
            tr, _ = engine(df.copy(), p)
        except Exception:
            tr = pd.DataFrame()
        if tr.empty or "Ret %" not in tr.columns:
            out[c["key"]] = pd.DataFrame({"Zeit": pd.Series(dtype="datetime64[ns]"), "Ret %": pd.Series(dtype=float)})
        else:
            out[c["key"]] = tr[["Zeit", "Ret %"]].reset_index(drop=True)
        if progress_cb and (i % 10 == 0 or i == len(combos) - 1):
            progress_cb((i + 1) / len(combos), f"Backtest Varianten {i+1}/{len(combos)} …")
    return out


def _slice(tr: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> np.ndarray:
    if tr.empty:
        return np.array([])
    m = (tr["Zeit"] >= a) & (tr["Zeit"] < b)
    return tr.loc[m, "Ret %"].to_numpy(dtype=float)


def topn_walk_forward(trades_by_key: dict, folds: list[dict], opt_metric: str,
                      min_trades: int, top_n: int) -> dict:
    """Pro Fold: Ranking nur nach IS, Top-N unverändert OOS testen.
    Zusätzlich wird für JEDE Variante die OOS-Kennzahl berechnet — aber ausschließlich, um zu messen,
    auf welchem OOS-Perzentil der IS-Sieger landet (Overfitting-Diagnose), nie zur Auswahl."""
    fold_rows, fold_details = [], []
    for fi, f in enumerate(folds):
        is_years = (f["is_end"] - f["is_start"]).days / 365.25
        oos_years = (f["oos_end"] - f["oos_start"]).days / 365.25
        is_list, oos_score = [], {}
        for key, tr in trades_by_key.items():
            r_is = _slice(tr, f["is_start"], f["is_end"])
            r_oos = _slice(tr, f["oos_start"], f["oos_end"])
            m_oos = metrics_from_rets(r_oos, oos_years)
            oos_score[key] = m_oos[opt_metric] if m_oos["n"] > 0 else -np.inf
            if len(r_is) < int(min_trades):
                continue
            m_is = metrics_from_rets(r_is, is_years)
            is_list.append((key, m_is, m_oos, r_oos))
        if not is_list:
            fold_rows.append({"fold": fi + 1, **f, "empty": True})
            fold_details.append([])
            continue
        is_list.sort(key=lambda x: (x[1][opt_metric], x[1]["n"]), reverse=True)
        top = is_list[:int(top_n)]
        # OOS-Perzentil des IS-Siegers unter ALLEN Varianten (0 = schlechteste, 1 = beste)
        _all = np.array([v for v in oos_score.values() if np.isfinite(v)])
        w_key, w_is, w_oos, w_r = top[0]
        w_s = oos_score[w_key]
        pct = float((_all < w_s).mean() + 0.5 * (_all == w_s).mean()) if len(_all) and np.isfinite(w_s) else np.nan
        top_oos_rets = [t[2]["total_ret"] for t in top]
        fold_rows.append({
            "fold": fi + 1, **f, "empty": False,
            "winner": w_key, "winner_is": w_is, "winner_oos": w_oos, "winner_oos_rets": w_r,
            "winner_oos_pct": pct,
            "top_avg_oos_ret": float(np.mean(top_oos_rets)),
            "top_avg_oos_pf": float(np.mean([t[2]["pf"] for t in top])),
            "top_share_profitable": float(np.mean([r > 0 for r in top_oos_rets])),
            "n_candidates": len(is_list),
        })
        fold_details.append([{"rank": i + 1, "key": t[0], "is": t[1], "oos": t[2], "oos_rets": t[3]}
                             for i, t in enumerate(top)])
    return {"folds": fold_rows, "details": fold_details}


def aggregate_setups(fold_details: list[list[dict]], oos_years_per_fold: float = 1.0) -> pd.DataFrame:
    """Pro Variante: wie oft in den IS-Top-N, Ø IS-Rang, gepoolte OOS-Trades NUR aus den Folds,
    in denen sie per IS ausgewählt wurde (= ehrliche OOS-Performance dieser Auswahl)."""
    agg: dict = {}
    for det in fold_details:
        for d in det:
            a = agg.setdefault(d["key"], {"n_top": 0, "ranks": [], "is_scores": [], "is_pf": [],
                                          "oos_rets": [], "fold_rets": []})
            a["n_top"] += 1
            a["ranks"].append(d["rank"])
            a["is_pf"].append(d["is"]["pf"])
            a["oos_rets"].append(d["oos_rets"])
            a["fold_rets"].append(d["oos"]["total_ret"])
    rows = []
    for key, a in agg.items():
        pooled = np.concatenate(a["oos_rets"]) if a["oos_rets"] else np.array([])
        m = metrics_from_rets(pooled, len(a["fold_rets"]) * oos_years_per_fold)
        is_pf = float(np.mean(a["is_pf"]))
        rows.append({"key": key, "n_top": a["n_top"], "avg_rank": float(np.mean(a["ranks"])),
                     "is_pf": is_pf, "oos_pf": m["pf"], "oos_wr": m["wr"], "oos_n": m["n"],
                     "oos_total_ret": m["total_ret"], "oos_sharpe": m["sharpe"], "oos_max_dd": m["max_dd"],
                     "oos_folds_pos": int(sum(r > 0 for r in a["fold_rets"])),
                     "oos_folds": len(a["fold_rets"]),
                     "pf_decay": (m["pf"] / is_pf) if is_pf > 0 else np.nan,
                     "pooled_rets": pooled})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["n_top", "oos_pf", "avg_rank"], ascending=[False, False, True]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# PBO per CSCV (Bailey, Borwein, López de Prado, Zhu 2015)
# ════════════════════════════════════════════════════════════════════════════
def monthly_matrix(trades_by_key: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Monatliche (log-)Renditen je Variante, Monate ohne Trade = 0."""
    months = pd.period_range(start, end - pd.Timedelta(days=1), freq="M")
    cols = {}
    for key, tr in trades_by_key.items():
        s = pd.Series(0.0, index=months)
        if not tr.empty:
            t = tr[(tr["Zeit"] >= start) & (tr["Zeit"] < end)]
            if not t.empty:
                lr = np.log1p(t["Ret %"].to_numpy(dtype=float) / 100)
                g = pd.Series(lr, index=t["Zeit"].dt.to_period("M")).groupby(level=0).sum()
                s.loc[g.index.intersection(months)] = g.loc[g.index.intersection(months)]
        cols[key] = s
    return pd.DataFrame(cols)


def cscv_pbo(M: pd.DataFrame, n_blocks: int = 10) -> dict:
    """M: T×N Matrix (Zeit × Varianten). Teilt T in n_blocks Blöcke, testet alle C(S, S/2)
    IS/OOS-Kombinationen: IS-Sieger (Sharpe) → relativer OOS-Rang ω → Logit λ.
    PBO = Anteil λ ≤ 0 (IS-Sieger landet OOS in der unteren Hälfte)."""
    X = M.to_numpy(dtype=float)
    T, N = X.shape
    if N < 2 or T < n_blocks * 2:
        return {"pbo": np.nan, "n_splits": 0, "logits": np.array([]), "n_variants": N}
    n_blocks -= n_blocks % 2
    blocks = np.array_split(np.arange(T), n_blocks)
    logits = []
    for is_ids in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in is_ids])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in is_ids])

        def _sr(A):
            sd = A.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                sr = np.where(sd > 0, A.mean(axis=0) / sd, -np.inf)
            return sr
        sr_is, sr_oos = _sr(X[is_idx]), _sr(X[oos_idx])
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1) + 1
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits), "logits": logits, "n_variants": N}


# ════════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ════════════════════════════════════════════════════════════════════════════
def deflated_sharpe(best_rets: np.ndarray, all_srs: np.ndarray, n_trials: int) -> dict:
    """best_rets: Perioden-Renditen (z.B. monatlich) des gewählten Setups.
    all_srs: nicht-annualisierte Sharpes ALLER getesteten Varianten (gleiche Periode).
    Gibt DSR = Wahrscheinlichkeit, dass die echte Sharpe > 0 ist, nach Korrektur für n_trials."""
    from scipy.stats import norm, skew, kurtosis
    r = np.asarray(best_rets, dtype=float)
    T = len(r)
    if T < 10 or r.std(ddof=1) == 0:
        return {"dsr": np.nan, "sr": np.nan, "sr0": np.nan}
    sr = r.mean() / r.std(ddof=1)
    srs = np.asarray([s for s in all_srs if np.isfinite(s)])
    var_sr = srs.var(ddof=1) if len(srs) > 1 else 0.0
    N = max(int(n_trials), 2)
    gamma = 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - gamma) * norm.ppf(1 - 1 / N) + gamma * norm.ppf(1 - 1 / (N * math.e)))
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
    return {"dsr": dsr, "sr": float(sr), "sr0": float(sr0)}


def build_combos(g_sl, g_tt, g_to, g_ma, g_fm, g_fill, g_adx, g_ed, g_xd) -> list[dict]:
    """Alle gültigen Kombinationen. Trail: nur 0/0 = 'aus', keine halben Kombis."""
    out = []
    for sl, tt, to, ma, fm, fill, adx, ed, xd in product(g_sl, g_tt, g_to, g_ma, g_fm, g_fill, g_adx, g_ed, g_xd):
        if ed == xd or (float(tt) > 0) != (float(to) > 0):
            continue
        if fm == NO_FILTER and ma != g_ma[0]:
            continue  # ohne MA-Filter ist die MA-Periode wirkungslos → keine identischen Duplikate
        key = (float(sl), float(tt), float(to), int(ma), fm, fill, float(adx), int(ed), int(xd))
        out.append({"key": key, "params": {"sl_pct": float(sl), "trail_trig": float(tt), "trail_off": float(to),
                                           "ma_period": int(ma), "filter_mode": fm, "fill_mode": fill,
                                           "adx_thresh": float(adx), "entry_dow": int(ed), "exit_dow": int(xd)}})
    return out


# ════════════════════════════════════════════════════════════════════════════
# Einfacher Split-Test: Top-N nach In-Sample → unverändert OOS → Holdout
# ════════════════════════════════════════════════════════════════════════════
def split_topn(trades_by_key: dict, start: pd.Timestamp, oos_start: pd.Timestamp, hold: pd.Timestamp,
               end: pd.Timestamp, opt_metric: str, min_trades: int, top_n: int) -> dict:
    """IS = [start, oos_start) · OOS = [oos_start, hold) · Holdout = [hold, end].
    Ranking ausschließlich nach IS. Für die Top-N werden OOS- und Holdout-Kennzahlen sowie die
    Trade-Liste (Datum, Ret %) über den GESAMTEN Zeitraum zurückgegeben (für die Equity-Kurve)."""
    end_x = end + pd.Timedelta(days=1)
    y_is = max((oos_start - start).days / 365.25, 0.01)
    y_oos = max((hold - oos_start).days / 365.25, 0.01)
    y_h = max((end_x - hold).days / 365.25, 0.01)
    all_is = []
    for key, tr in trades_by_key.items():
        r_is = _slice(tr, start, oos_start)
        m_is = metrics_from_rets(r_is, y_is)
        all_is.append({"key": key, "is": m_is, "eligible": len(r_is) >= int(min_trades)})
    ranked = sorted([a for a in all_is if a["eligible"]],
                    key=lambda a: (a["is"][opt_metric], a["is"]["n"]), reverse=True)
    for i, a in enumerate(ranked):
        a["rank"] = i + 1
    top = []
    for a in ranked[:int(top_n)]:
        tr = trades_by_key[a["key"]]
        t = tr[(tr["Zeit"] >= start) & (tr["Zeit"] < end_x)] if not tr.empty else tr
        top.append({
            "rank": a["rank"], "key": a["key"], "is": a["is"],
            "oos": metrics_from_rets(_slice(tr, oos_start, hold), y_oos),
            "hold": metrics_from_rets(_slice(tr, hold, end_x), y_h),
            "trades": [[pd.Timestamp(z).isoformat(), float(r)] for z, r in zip(t["Zeit"], t["Ret %"])] if not t.empty else [],
        })
    all_rows = sorted(all_is, key=lambda a: a.get("rank", 10**9))
    return {"start": start, "oos_start": oos_start, "hold": hold, "end": end, "top": top,
            "all": [{"key": a["key"], "rank": a.get("rank"), "is": a["is"]} for a in all_rows]}
