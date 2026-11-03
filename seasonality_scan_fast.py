"""
Schneller Seasonality-Muster-Scanner (vektorisiert).

Liefert exakt dieselben Zeilen wie die Schleife ueber _scan_patterns_cached in taco_web_app.py,
aber: Trade-Matrix, Drawdowns, Robustheit und Spaeteinstieg werden einmal pro Symbol als
NumPy-Matrizen berechnet statt pro Fenster und Jahr in Python-Schleifen.
Die Kennzahlen pro Kandidat (Mittelwert, Std, DD) nutzen dieselben NumPy-Aufrufe auf denselben
Arrays in derselben Reihenfolge wie das Original -> identische Werte.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _year_tables(year_data: dict, years: np.ndarray, max_td: int):
    """Pro Jahr: Index-Tabelle DOY->Handelstag, Close/ATR je DOY und Running-Min/Max fuer Drawdowns."""
    Y = len(years)
    I = np.zeros((Y, 370), dtype=np.int64)
    L = np.zeros(Y, dtype=np.int64)
    CL = np.full((Y, 370), np.nan)
    AT = np.full((Y, 370), np.nan)
    mins, maxs = [], []
    for yi, yr in enumerate(years):
        yd = year_data[int(yr)]
        doys = yd["doys"]; n = len(doys)
        idx = np.searchsorted(doys, np.arange(370))
        I[yi] = idx; L[yi] = n
        safe = np.minimum(idx, n - 1)
        CL[yi] = yd["closes"][safe]; AT[yi] = yd["atrs"][safe]
        lows, highs = yd["lows"], yd["highs"]
        K = max_td + 1
        mn = np.full((n + 1, K), np.inf); mx = np.full((n + 1, K), -np.inf)
        mn[:n, 0] = lows; mx[:n, 0] = highs
        for k in range(1, min(K, n)):
            mn[:n - k, k] = np.minimum(mn[:n - k, k - 1], lows[k:])
            mx[:n - k, k] = np.maximum(mx[:n - k, k - 1], highs[k:])
        mins.append(mn); maxs.append(mx)
    return I, L, CL, AT, mins, maxs


def scan_patterns_fast(
    year_data: dict,
    sorted_years: np.ndarray,
    end_year: int,
    lookback_years: int,
    holding_periods: list[int],
    directions: tuple[str, ...] | list[str],
    min_winrate: float,
    min_trades: int,
    compute_stars: Callable,
) -> list[dict]:
    y5, y10, y15, y20 = end_year - 4, end_year - 9, end_year - 14, end_year - 19
    year_start_primary = end_year - lookback_years + 1
    data_start_year = int(sorted_years.min())
    all_years_start = min(end_year - 20 + 1, data_start_year)
    has_5j, has_10j, has_15j, has_20j = (data_start_year <= y5, data_start_year <= y10,
                                         data_start_year <= y15, data_start_year <= y20)

    yrs = np.array([int(y) for y in sorted_years if all_years_start <= int(y) <= end_year and int(y) in year_data])
    if len(yrs) == 0:
        return []
    holds = [int(h) for h in holding_periods]
    max_td = int(max(len(year_data[int(y)]["doys"]) for y in yrs))
    max_td = min(max_td, max(holds) + 10)
    I, L, CL, AT, MN, MX = _year_tables(year_data, yrs, max_td)
    prim = yrs >= year_start_primary

    def pair(e: np.ndarray, x: np.ndarray):
        """Gueltigkeit, Long-Rendite, td fuer DOY-Paare (Broadcast ueber Jahre)."""
        ei = I[:, e]; xi = I[:, x]
        ok = (ei < L[:, None]) & (xi < L[:, None]) & (xi > ei)
        ep = CL[:, e]; xp = CL[:, x]
        return ok, (xp - ep) / ep, xi - ei

    # ---- Fenster in Original-Reihenfolge (entry, hold) ----
    E_list, X_list = [], []
    for entry_doy in range(1, 363):
        for h in holds:
            if entry_doy + h <= 366:
                E_list.append(entry_doy); X_list.append(entry_doy + h)
    E = np.array(E_list); X = np.array(X_list)
    ok, rl, td = pair(E, X)
    n_all = ok.sum(axis=0); n_prim = (ok & prim[:, None]).sum(axis=0)
    keep = (n_all >= min_trades) & (n_prim >= min_trades)

    okp = ok & prim[:, None]
    wins_l = (okp & (rl > 0)).sum(axis=0); wins_s = (okp & (-rl > 0)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        wr_l = wins_l / np.maximum(n_prim, 1); wr_s = wins_s / np.maximum(n_prim, 1)
    pass_l = keep & (wr_l >= min_winrate) if "long" in directions else np.zeros_like(keep)
    pass_s = keep & (wr_s >= min_winrate) if "short" in directions else np.zeros_like(keep)
    cand = np.flatnonzero(pass_l | pass_s)
    if len(cand) == 0:
        return []

    # ---- Robustheit (±3..±7) und Spaeteinstieg (+3..+7) vektorisiert ueber Kandidaten ----
    Ec = E[cand]; Xc = X[cand]
    rob_off = list(range(-7, -2)) + list(range(3, 8))
    alt = {}
    for off in sorted(set(rob_off) | set(range(3, 8))):
        ae = Ec + off
        aec = np.clip(ae, 0, 369)
        ok_a, r_a, _ = pair(aec, Xc)
        ok_a = ok_a & prim[:, None]
        n_a = ok_a.sum(axis=0)
        alt[off] = (ae, n_a, (ok_a & (r_a > 0)).sum(axis=0), (ok_a & (-r_a > 0)).sum(axis=0))

    _ref = pd.Timestamp(year=2000, month=1, day=1)
    _lab_cache: dict = {}

    def _label(doy: int, shift: bool) -> str:
        key = (doy, shift)
        if key not in _lab_cache:
            ts = _ref + pd.Timedelta(days=doy - 1)
            if shift:
                if ts.weekday() == 5: ts += pd.Timedelta(days=2)
                if ts.weekday() == 6: ts += pd.Timedelta(days=1)
            _lab_cache[key] = ts.strftime("%d. %b")
        return _lab_cache[key]

    rows: list[dict] = []
    for j, c in enumerate(cand):
        entry_doy, exit_doy = int(E[c]), int(X[c])
        okc = ok[:, c]
        ty = yrs[okc]
        ei = I[okc, entry_doy]; xi = I[okc, exit_doy]
        ep = CL[okc, entry_doy]; xp = CL[okc, exit_doy]
        long_ret_all = (xp - ep) / ep; short_ret_all = (ep - xp) / ep
        yi_idx = np.flatnonzero(okc)
        min_low = np.array([MN[yi][a + 1, b - a - 1] for yi, a, b in zip(yi_idx, ei, xi)])
        max_high = np.array([MX[yi][a + 1, b - a - 1] for yi, a, b in zip(yi_idx, ei, xi)])
        long_dd_all = (min_low - ep) / ep; short_dd_all = (ep - max_high) / ep
        atr_all = AT[okc, entry_doy]; td_all = xi - ei
        pmask = ty >= year_start_primary

        def stats(yr_start: int, dir_: str, min_t: int | None = None):
            m = ty >= yr_start
            if m.sum() < (min_t if min_t is not None else min_trades):
                return None
            rets = (long_ret_all if dir_ == "long" else short_ret_all)[m]
            dds = (long_dd_all if dir_ == "long" else short_dd_all)[m]
            nt = len(rets)
            wr = float((rets > 0).sum() / nt)
            avg_ret = rets.mean()
            std_ret = rets.std(ddof=1) if nt > 1 else 0.0
            return {"wr": wr, "nt": nt, "avg_ret": avg_ret, "std_ret": std_ret,
                    "avg_dd": dds.mean(), "max_dd": dds.min()}

        n_primary = int(pmask.sum())

        def wr_only(yr_start: int, dir_: str, min_t: int | None = None):
            m = ty >= yr_start
            nt_ = int(m.sum())
            if nt_ < (min_t if min_t is not None else min_trades):
                return np.nan
            rets = (long_ret_all if dir_ == "long" else short_ret_all)[m]
            return float((rets > 0).sum() / nt_)
        for dir_ in directions:
            if dir_ == "long" and not pass_l[c]: continue
            if dir_ == "short" and not pass_s[c]: continue
            st_ = stats(year_start_primary, dir_)
            if st_ is None or st_["wr"] < min_winrate:
                continue
            wr, avg_ret, std_ret, nt = st_["wr"], st_["avg_ret"], st_["std_ret"], st_["nt"]
            avg_dd, max_dd_val = st_["avg_dd"], st_["max_dd"]
            avg_td = np.mean(list(td_all[pmask]))
            sharpe = avg_ret / std_ret * np.sqrt(252 / max(avg_td, 1)) if std_ret > 1e-10 else np.nan
            sqn = avg_ret / std_ret * np.sqrt(nt) * 100 if std_ret > 1e-10 else np.nan

            wr_5j = wr_only(y5, dir_, min_t=3)
            wr_10j = wr_only(y10, dir_, min_t=max(3, int(10 * 0.6)))
            wr_15j = wr_only(y15, dir_, min_t=max(3, int(15 * 0.6)))
            wr_20j_raw = wr_only(y20, dir_)
            if not has_20j or np.isnan(wr_20j_raw):
                wr_20j_raw = wr_only(data_start_year, dir_)
            wr_20j = round(wr_20j_raw * 100, 1) if not np.isnan(wr_20j_raw) else np.nan
            wr_5j = round(wr_5j * 100, 1) if has_5j and not np.isnan(wr_5j) else np.nan
            wr_10j = round(wr_10j * 100, 1) if has_10j and not np.isnan(wr_10j) else np.nan
            wr_15j = round(wr_15j * 100, 1) if has_15j and not np.isnan(wr_15j) else np.nan

            ep_p = ep[pmask]; at_p = atr_all[pmask]
            avg_atr_pct_pre = float(np.mean([a / e for a, e in zip(at_p, ep_p) if e > 0]) * 100)
            atr_efficient = avg_atr_pct_pre > 0 and (abs(avg_ret) / (avg_atr_pct_pre / 100)) >= 0.4

            robust_wins = robust_total = 0
            for off in rob_off:
                ae, n_a, wl, ws = alt[off]
                if ae[j] < 1 or ae[j] > 365:
                    continue
                if n_a[j] >= min_trades:
                    robust_total += 1
                    alt_wr = (wl[j] if dir_ == "long" else ws[j]) / n_a[j]
                    if alt_wr >= min_winrate:
                        robust_wins += 1
            if robust_total == 0:
                robustheit = "—"
            else:
                rr = robust_wins / robust_total
                if rr >= 0.80 and atr_efficient: robustheit = "🟢 Stark"
                elif rr >= 0.80:                 robustheit = "✅ Robust"
                elif rr >= 0.60:                 robustheit = "✅ Robust"
                elif rr >= 0.40:                 robustheit = "⚠️ Sensitiv"
                else:                            robustheit = "❌ Fragil"

            avg_atr_pct = avg_atr_pct_pre
            try:
                entry_label = _label(entry_doy, True); exit_label = _label(exit_doy, True)
            except Exception:
                entry_label = f"DOY {entry_doy}"; exit_label = f"DOY {exit_doy}"
            hold = int(round(avg_td))

            best_late_wr = None; best_late_label = None
            for off in range(3, 8):
                ae, n_a, wl, ws = alt[off]
                if n_a[j] >= min_trades:
                    alt_wr2 = float((wl[j] if dir_ == "long" else ws[j]) / n_a[j] * 100)
                    if best_late_wr is None or alt_wr2 > best_late_wr:
                        best_late_wr = alt_wr2
                        try:
                            best_late_label = _label(int(ae[j]), False)
                        except Exception:
                            best_late_label = f"DOY {int(ae[j])}"

            rows.append({
                "Richtung": "Long" if dir_ == "long" else "Short",
                "Entry": entry_label,
                "Exit": exit_label,
                "Haltedauer (TD)": hold,
                "n (Jahre)": nt,
                "WR 5J %": wr_5j,
                "WR 10J %": wr_10j,
                "WR 15J %": wr_15j,
                "WR 20J %": wr_20j,
                "Ø Profit %": round(avg_ret * 100, 2),
                "Ø DD %": round(avg_dd * 100, 2),
                "Max DD %": round(max_dd_val * 100, 2),
                "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else np.nan,
                "SQN": round(sqn, 2) if not np.isnan(sqn) else np.nan,
                "Ø ATR %": round(avg_atr_pct, 3) if not np.isnan(avg_atr_pct) else np.nan,
                "Robustheit": robustheit,
                "⭐ Rating": compute_stars(wr, avg_ret, avg_dd, max_dd_val, sharpe, robustheit, avg_atr_pct,
                                           n_trades=n_primary),
                "Bester Späteinstieg": best_late_label or "—",
                "WR Späteinstieg %": round(best_late_wr, 1) if best_late_wr is not None else np.nan,
                "_entry_doy": entry_doy,
                "_exit_doy": exit_doy,
            })
    return rows
