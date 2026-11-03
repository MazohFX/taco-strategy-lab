"""
Seasonality Muster — Walk-Forward Validierung (Anchored/Expanding Window)

Eigenständiges Modul, das NUR die Kern-Logik enthält (keine UI).
Import in taco_web_app.py:
    from seasonality_wfa import run_seasonality_wfa, WFAResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

# ---------------------------------------------------------------------------
# Hilfsfunktionen (identisch zu taco_web_app.py — kein Import, um
# Zirkularabhängigkeit zu vermeiden)
# ---------------------------------------------------------------------------

# Minimales Start-In-Sample-Fenster (Jahre). Konstante, leicht auffindbar.
MIN_IS_YEARS: int = 10


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _wilcoxon_p(rets: np.ndarray) -> float:
    if len(rets) < 5:
        return float("nan")
    try:
        _, p = _scipy_stats.wilcoxon(rets, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Interne Scan-Hilfsfunktion (schlanke Version ohne Rating, Späteinstieg etc.)
# ---------------------------------------------------------------------------

def _build_year_data(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """
    Baut {year -> {doys, closes, highs, lows, atrs}} aus einem OHLC-DataFrame
    mit DatetimeIndex auf. ATR(14) wird per EWM berechnet.
    """
    _tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift(1)).abs(),
        "lc": (df["low"]  - df["close"].shift(1)).abs(),
    }).max(axis=1)
    df = df.copy()
    df["atr"] = _tr.ewm(span=14, adjust=False).mean()

    year_data: dict = {}
    for yr, grp in df.groupby(df.index.year):
        yr_doys = grp.index.dayofyear.values.astype(int)
        sort_idx = np.argsort(yr_doys)
        year_data[int(yr)] = {
            "doys":   yr_doys[sort_idx],
            "closes": grp["close"].values.astype(float)[sort_idx],
            "highs":  grp["high"].values.astype(float)[sort_idx],
            "lows":   grp["low"].values.astype(float)[sort_idx],
            "atrs":   grp["atr"].values.astype(float)[sort_idx],
        }
    sorted_years = np.array(sorted(year_data.keys()), dtype=int)
    return year_data, sorted_years


def _single_trade(year_data: dict, yr: int, entry_doy: int, exit_doy: int
                  ) -> tuple[float, float] | None:
    """
    Gibt (long_ret, short_ret) für ein einzelnes Jahr zurück, oder None.
    """
    yd = year_data.get(yr)
    if yd is None:
        return None
    doys = yd["doys"]
    ei = int(np.searchsorted(doys, entry_doy))
    xi = int(np.searchsorted(doys, exit_doy))
    if ei >= len(doys) or xi >= len(doys) or xi <= ei:
        return None
    ep = yd["closes"][ei]
    xp = yd["closes"][xi]
    return (xp - ep) / ep, (ep - xp) / ep


def _scan_is_window(
    year_data: dict,
    sorted_years: np.ndarray,
    is_years: np.ndarray,          # Welche Jahre zum IS-Fenster gehören
    holding_periods: list[int],
    directions: list[str],
    min_winrate: float,
    min_trades: int,
) -> list[dict]:
    """
    Führt den Muster-Scan auf exakt `is_years` aus (kein end_year-Heuristik).
    Gibt vereinfachte Kandidaten-Dicts zurück:
      {entry_doy, exit_doy, direction, is_winrate, is_n, is_avg_ret, p_val}
    """
    candidates: list[dict] = []
    is_set = set(is_years.tolist())

    for entry_doy in range(1, 363):
        for cal_hold in holding_periods:
            exit_doy = entry_doy + cal_hold
            if exit_doy > 366:
                continue

            # Alle IS-Jahre mit gültigem Trade
            trades_is: list[tuple[float, float]] = []
            for yr in is_years:
                t = _single_trade(year_data, int(yr), entry_doy, exit_doy)
                if t is not None:
                    trades_is.append(t)

            if len(trades_is) < min_trades:
                continue

            for dir_ in directions:
                rets = np.array([t[0] if dir_ == "long" else t[1] for t in trades_is])
                wins = int((rets > 0).sum())
                wr = wins / len(rets)
                if wr < min_winrate:
                    continue
                p_val = _wilcoxon_p(rets)
                candidates.append({
                    "entry_doy": entry_doy,
                    "exit_doy":  exit_doy,
                    "direction": dir_,
                    "is_winrate": wr,
                    "is_n": len(rets),
                    "is_avg_ret": float(rets.mean()),
                    "is_p_val": p_val,
                })

    return candidates


def _return_matrix(year_data: dict, sorted_years: np.ndarray, holding_periods: list[int]):
    """
    Rendite-Matrix R[jahr, entry_doy-1, hold_idx] (Long-Rendite, NaN = kein gültiger Trade).
    Exakt dieselbe Logik wie _single_trade (erster Handelstag >= DOY, Exit muss nach Entry liegen),
    aber einmal pro Symbol statt einmal pro Fold/Fenster/Jahr.
    """
    entries = np.arange(1, 363)
    holds = np.asarray(holding_periods, dtype=int)
    R = np.full((len(sorted_years), len(entries), len(holds)), np.nan)
    for yi, yr in enumerate(sorted_years):
        yd = year_data[int(yr)]
        doys, closes = yd["doys"], yd["closes"]
        idx = np.searchsorted(doys, np.arange(0, 368))           # idx[d] = erster Index mit doy >= d
        ei = idx[entries]
        for hi, h in enumerate(holds):
            xd = entries + h
            ok_d = xd <= 366
            xi = idx[np.minimum(xd, 367)]
            ok = ok_d & (ei < len(doys)) & (xi < len(doys)) & (xi > ei)
            ep = closes[np.minimum(ei, len(doys) - 1)]; xp = closes[np.minimum(xi, len(doys) - 1)]
            R[yi, :, hi] = np.where(ok, (xp - ep) / ep, np.nan)
    return entries, holds, R


# ---------------------------------------------------------------------------
# Datenklassen für das Ergebnis
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Ergebnis eines einzelnen OOS-Jahres (Folds)."""
    oos_year: int
    n_candidates: int                # Kandidaten, die in diesem Fold gefunden wurden
    trades: list[dict] = field(default_factory=list)
    # Jedes trade-Dict: {entry_doy, exit_doy, direction, is_winrate, oos_win, oos_ret}


@dataclass
class PatternWFAStats:
    """Aggregierte WFA-Statistik für eine Muster-Kombination."""
    symbol: str
    entry_doy: int
    exit_doy: int
    direction: str
    # Wie oft über alle Folds als IS-Kandidat gefunden
    n_folds_found: int
    # In wie vielen Folds war OOS-Daten vorhanden (Trade ausführbar)
    n_folds_oos: int
    # OOS-Winrate über alle Folds mit Trade
    oos_winrate: float | None
    oos_n_trades: int
    oos_avg_ret: float | None
    # Mittlere IS-Winrate in den Folds, in denen das Muster gefunden wurde
    avg_is_winrate: float | None
    # Stabilitätsbadge
    validated: bool          # True wenn n_folds_found >= min_folds_for_badge
                             # UND oos_winrate nicht signifikant unter IS fällt


@dataclass
class WFAResult:
    """Gesamtergebnis der Walk-Forward-Validierung für alle Symbole."""
    pattern_stats: list[PatternWFAStats]
    fold_results: dict[str, list[FoldResult]]   # symbol -> [FoldResult, ...]
    min_is_years: int
    n_folds: int


# ---------------------------------------------------------------------------
# Haupt-Funktion
# ---------------------------------------------------------------------------

def run_seasonality_wfa(
    symbol_dfs: dict[str, pd.DataFrame],   # {symbol -> OHLC-DataFrame mit DatetimeIndex}
    holding_periods: list[int],
    directions: list[str],
    min_winrate: float = 0.70,
    min_trades: int = 5,                   # Mindest-Trades im IS-Fenster
    min_is_years: int = MIN_IS_YEARS,
    min_folds_for_badge: int = 5,          # Stabilitätsschwelle für ✅ Badge
    progress_callback: Callable[[float, str], None] | None = None,
) -> WFAResult:
    """
    Anchored/Expanding-Window Walk-Forward-Validierung des Seasonality-Scanners.

    Ablauf:
      1. IS-Startfenster = erste `min_is_years` Jahre der verfügbaren Daten.
      2. Für jedes Fold-Jahr Y (erstes Jahr nach IS-Start bis letztes verfügbares Jahr):
         a. Scan nur auf den Jahren < Y.
         b. Kandidaten auf Jahr Y anwenden (keine erneute Optimierung).
         c. IS-Fenster für nächsten Fold um Y erweitern.
      3. Aggregation pro Muster: Stabilitätszähler + OOS-Winrate.

    Parameters
    ----------
    symbol_dfs
        OHLC-DataFrames mit DatetimeIndex, wie sie normalize_ohlc() liefert.
    holding_periods
        Liste von Kalenderhalteperioden (Tage) — identisch zum normalen Scanner.
    directions
        ["long"] / ["short"] / ["long", "short"].
    min_winrate
        Mindest-Winrate im IS-Fenster (0–1), z.B. 0.70.
    min_trades
        Mindest-Anzahl IS-Trades (je kleiner das IS-Fenster, desto restriktiver).
    min_is_years
        Start-In-Sample-Fenster in Jahren. Default 10.
    min_folds_for_badge
        Wie viele Folds ein Muster als IS-Kandidat erschienen sein muss, um
        als ✅ OOS-validiert zu gelten (zusätzlich zur OOS-Winrate-Bedingung).
    progress_callback
        Optionale Callback-Funktion (fraction: float, text: str) für UI-Updates.

    Returns
    -------
    WFAResult
        Enthält pro Symbol eine Liste von FoldResults und die aggregierten
        PatternWFAStats für alle Symbole.
    """
    all_pattern_stats: list[PatternWFAStats] = []
    all_fold_results: dict[str, list[FoldResult]] = {}
    total_symbols = len(symbol_dfs)

    for sym_idx, (symbol, df) in enumerate(symbol_dfs.items()):
        if progress_callback:
            progress_callback(sym_idx / max(total_symbols, 1),
                              f"WFA: {symbol} ({sym_idx+1}/{total_symbols})")

        year_data, sorted_years = _build_year_data(df)

        if len(sorted_years) < min_is_years + 1:
            # Nicht genug Daten für auch nur einen OOS-Fold
            all_fold_results[symbol] = []
            continue

        # Erster IS-Block: erste min_is_years Jahre
        first_is_end_year = sorted_years[min_is_years - 1]
        oos_years = sorted_years[sorted_years > first_is_end_year]

        if len(oos_years) == 0:
            all_fold_results[symbol] = []
            continue

        # Akkumulator: {(entry_doy, exit_doy, direction) -> {folds_found, oos_trades, is_wr_sum}}
        pattern_acc: dict[tuple, dict] = {}
        fold_results_sym: list[FoldResult] = []

        entries, holds, R = _return_matrix(year_data, sorted_years, holding_periods)
        valid = ~np.isnan(R)
        Rz = np.where(valid, R, 0.0)
        # kumulierte Summen über die Jahre -> jeder Fold (Jahre < Y) ist ein einziger Lookup
        c_n = np.cumsum(valid, axis=0); c_sum = np.cumsum(Rz, axis=0)
        c_wl = np.cumsum(valid & (R > 0), axis=0); c_ws = np.cumsum(valid & (R < 0), axis=0)

        for fold_i, oos_year in enumerate(oos_years):
            is_years = sorted_years[sorted_years < oos_year]
            if len(is_years) < min_is_years:
                continue
            k = len(is_years) - 1                     # sorted_years ist aufsteigend -> Index des letzten IS-Jahres
            yo = int(np.searchsorted(sorted_years, oos_year))
            n = c_n[k]
            fold_res = FoldResult(oos_year=int(oos_year), n_candidates=0)
            for dir_ in directions:
                wins = c_wl[k] if dir_ == "long" else c_ws[k]
                avg = (c_sum[k] if dir_ == "long" else -c_sum[k]) / np.maximum(n, 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    wr = np.where(n > 0, wins / np.maximum(n, 1), 0.0)
                cand = (n >= min_trades) & (wr >= min_winrate) & (n > 0)
                for ei_, hi_ in zip(*np.nonzero(cand)):
                    entry_doy = int(entries[ei_]); exit_doy = entry_doy + int(holds[hi_])
                    key = (entry_doy, exit_doy, dir_)
                    acc = pattern_acc.setdefault(key, {"folds_found": 0, "oos_wins": 0, "oos_n": 0,
                                                        "oos_ret_sum": 0.0, "is_wr_sum": 0.0})
                    acc["folds_found"] += 1
                    acc["is_wr_sum"] += float(wr[ei_, hi_])
                    r = R[yo, ei_, hi_]
                    oos_ret = oos_win = None
                    if not np.isnan(r):
                        oos_ret = float(r if dir_ == "long" else -r); oos_win = oos_ret > 0
                        acc["oos_n"] += 1; acc["oos_ret_sum"] += oos_ret; acc["oos_wins"] += int(oos_win)
                    fold_res.trades.append({"entry_doy": entry_doy, "exit_doy": exit_doy, "direction": dir_,
                                            "is_winrate": float(wr[ei_, hi_]), "oos_win": oos_win, "oos_ret": oos_ret})
                fold_res.n_candidates += int(cand.sum())
            fold_results_sym.append(fold_res)

        all_fold_results[symbol] = fold_results_sym

        # ---- Schritt 3: Aggregation pro Muster ----
        for (entry_doy, exit_doy, direction), acc in pattern_acc.items():
            n_found = acc["folds_found"]
            n_oos   = acc["oos_n"]
            oos_wr  = acc["oos_wins"] / n_oos if n_oos > 0 else None
            oos_avg = acc["oos_ret_sum"] / n_oos if n_oos > 0 else None
            avg_is_wr = acc["is_wr_sum"] / n_found if n_found > 0 else None

            # Badge-Bedingung:
            # 1. Muster war in >= min_folds_for_badge Folds als IS-Kandidat
            # 2. OOS-Winrate ist nicht signifikant schlechter als IS-Erwartung
            #    (Heuristik: OOS-WR >= IS-WR - 15 Prozentpunkte, bei mind. 3 OOS-Trades)
            validated = False
            if (n_found >= min_folds_for_badge
                    and n_oos >= 3
                    and oos_wr is not None
                    and avg_is_wr is not None
                    and oos_wr >= avg_is_wr - 0.15):
                validated = True

            all_pattern_stats.append(PatternWFAStats(
                symbol=symbol,
                entry_doy=entry_doy,
                exit_doy=exit_doy,
                direction=direction,
                n_folds_found=n_found,
                n_folds_oos=n_oos,
                oos_winrate=oos_wr,
                oos_n_trades=n_oos,
                oos_avg_ret=oos_avg,
                avg_is_winrate=avg_is_wr,
                validated=validated,
            ))

    if progress_callback:
        progress_callback(1.0, "WFA abgeschlossen")

    return WFAResult(
        pattern_stats=all_pattern_stats,
        fold_results=all_fold_results,
        min_is_years=min_is_years,
        n_folds=len(next(iter(all_fold_results.values()), [])),
    )


# ---------------------------------------------------------------------------
# Lookup-Hilfsfunktion (für die UI: "Hat dieses Ergebnis-Row ein WFA-Badge?")
# ---------------------------------------------------------------------------

def _badge_index(wfa_result: WFAResult) -> dict:
    """Einmal pro WFA-Ergebnis: (symbol, entry, exit, richtung) -> Badge (O(1)-Lookup statt Liste durchsuchen)."""
    idx = getattr(wfa_result, "_badge_idx", None)
    if idx is None:
        idx = {(ps.symbol, int(ps.entry_doy), int(ps.exit_doy), ps.direction):
               ("✅ OOS-validiert" if ps.validated else "⚠️ Nur IS") for ps in wfa_result.pattern_stats}
        try:
            object.__setattr__(wfa_result, "_badge_idx", idx)
        except Exception:
            pass
    return idx


def wfa_badge_for_row(
    wfa_result: WFAResult,
    symbol: str,
    entry_doy: int,
    exit_doy: int,
    direction: str,
) -> str:
    """
    Gibt den Badge-String für eine Muster-Row zurück:
      "✅ OOS-validiert"   — wenn validated == True
      "⚠️ Nur IS"          — wenn in WFA gefunden aber nicht validiert
      ""                   — wenn nicht in WFA-Ergebnis enthalten
    """
    return _badge_index(wfa_result).get((symbol, int(entry_doy), int(exit_doy), direction), "")


def wfa_badges_for_df(wfa_result: WFAResult, df: pd.DataFrame) -> list[str]:
    """Badges für ein ganzes Muster-DataFrame auf einmal (Spalten Symbol, Richtung, _entry_doy, _exit_doy)."""
    idx = _badge_index(wfa_result)
    if df.empty:
        return []
    return [idx.get((str(s), int(e), int(x), "long" if r == "Long" else "short"), "")
            for s, r, e, x in zip(df["Symbol"], df["Richtung"],
                                  df["_entry_doy"].fillna(0), df["_exit_doy"].fillna(0))]


# ---------------------------------------------------------------------------
# Quant-Bewertung (0–5 Sterne) — "Würde ich das als Quant-Trader handeln?"
# ---------------------------------------------------------------------------

QUANT_CRITERIA = {
    "★ OOS":         "Walk-Forward: ≥ 5 OOS-Trades, OOS-Winrate ≥ 65 % und OOS-Ø-Return > Kostenpuffer (0,1 %)",
    "★ Beide Hälften": "Erste UND zweite Hälfte der Historie je ≥ {half:.0%} Winrate (Muster war nicht nur zuletzt gut)",
    "★ Signifikanz":  "Binomial-p < {p:g} über alle Jahre — streng, weil je Symbol ~11.600 Fenster getestet werden",
    "★ Plateau":     "≥ {plat:.0%} der Nachbar-Fenster (Entry ±2, Haltedauer ±2 Tage) mit ≥ 60 % Winrate",
    "★ Ökonomie":    "Ø-Return ≥ {econ:.1%} und Median > 0 — lohnt sich nach Kosten wirklich",
}
QUANT_P, QUANT_HALF, QUANT_PLAT, QUANT_ECON = 0.0005, 0.70, 0.70, 0.004
QUANT_CRITERIA = {k: v.format(p=QUANT_P, half=QUANT_HALF, plat=QUANT_PLAT, econ=QUANT_ECON) for k, v in QUANT_CRITERIA.items()}


def _binom_p(k: int, n: int) -> float:
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _window_rets(year_data: dict, years: np.ndarray, entry_doy: int, exit_doy: int, direction: str) -> np.ndarray:
    out = []
    for yr in years:
        t = _single_trade(year_data, int(yr), entry_doy, exit_doy)
        if t is not None:
            out.append(t[0] if direction == "long" else t[1])
    return np.array(out, dtype=float)


def quant_rating_dataframe(
    wfa_result: WFAResult,
    symbol_dfs: dict[str, pd.DataFrame],
    cost_buffer: float = 0.001,          # 0,1 % Spread/Kommission je Trade
    min_stars: int = 0,
    dedup_days: int = 5,
) -> pd.DataFrame:
    """
    Bewertet jedes WFA-Muster mit 0–5 Sternen (je ein Stern pro erfülltem Kriterium aus QUANT_CRITERIA).
    Ohne "★ OOS" UND "★ Signifikanz" gibt es höchstens 3 Sterne. Schwellen an Zufallsdaten (gemischte Tagesrenditen) geeicht.
    Überlappende Fenster (gleiches Symbol/Richtung, Entry ≤ dedup_days auseinander) werden zu einem
    Setup zusammengefasst (das beste bleibt).
    Vektorisiert: alle Kennzahlen werden pro Symbol einmal als Matrix über alle (Entry, Haltedauer) berechnet.
    """
    rows = []
    by_sym: dict[str, list] = {}
    for ps in wfa_result.pattern_stats:
        if ps.symbol in symbol_dfs:
            by_sym.setdefault(ps.symbol, []).append(ps)
    max_hold = max((ps.exit_doy - ps.entry_doy for ps in wfa_result.pattern_stats), default=20)
    H = max_hold + 2
    binom_tab: list[list[float]] = []

    for sym, plist in by_sym.items():
        yd_, yrs_ = _build_year_data(symbol_dfs[sym])
        R = _return_matrix(yd_, yrs_, list(range(1, H + 1)))[2]          # (Y, 362, H)
        V = ~np.isnan(R); Rz = np.where(V, R, 0.0)
        n = V.sum(axis=0)
        cV = np.cumsum(V, axis=0)
        cWL = np.cumsum(V & (R > 0), axis=0); cWS = np.cumsum(V & (R < 0), axis=0)
        sumR = Rz.sum(axis=0)
        with np.errstate(all="ignore"):
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                med = np.nanmedian(R, axis=0)
        has8 = n >= 8
        wL, wS = cWL[-1], cWS[-1]
        okL = has8 & (wL >= 0.60 * n); okS = has8 & (wS >= 0.60 * n)
        # Plateau-Gitter: Anteil guter Nachbarn (Entry ±2, Haltedauer ±2) je Zelle
        E_, H_ = np.meshgrid(np.arange(1, 363), np.arange(1, H + 1), indexing="ij")
        nbN = np.zeros((362, H)); nbL = np.zeros((362, H)); nbS = np.zeros((362, H))
        for de in range(-2, 3):
            for dh in range(-2, 3):
                if de == 0 and dh == 0:
                    continue
                e2 = E_ + de; h2 = H_ + dh
                inb = (e2 >= 1) & (e2 <= 362) & (h2 >= 1) & (h2 <= H) & (e2 + h2 <= 366)
                e2c = np.clip(e2, 1, 362) - 1; h2c = np.clip(h2, 1, H) - 1
                m = inb & has8[e2c, h2c]
                nbN += m; nbL += m & okL[e2c, h2c]; nbS += m & okS[e2c, h2c]
        half = n // 2
        hidx = np.argmax(cV >= np.maximum(half, 1)[None], axis=0)
        n_folds_total = max(len(wfa_result.fold_results.get(sym, [])), 1)

        # ---- alle Muster dieses Symbols auf einmal ----
        ent = np.array([ps.entry_doy for ps in plist]); hold = np.array([ps.exit_doy - ps.entry_doy for ps in plist])
        lg = np.array([ps.direction == "long" for ps in plist])
        okm = (hold >= 1) & (hold <= H)
        ei = ent - 1; hi = np.clip(hold - 1, 0, H - 1)
        nn = n[ei, hi]
        okm &= nn >= 8
        wins = np.where(lg, wL[ei, hi], wS[ei, hi])
        hh = half[ei, hi]
        w1 = np.where(lg, cWL[hidx[ei, hi], ei, hi], cWS[hidx[ei, hi], ei, hi])
        w1 = np.where(hh > 0, w1, 0)
        sg = np.where(lg, 1.0, -1.0)
        avg = sumR[ei, hi] / np.maximum(nn, 1) * sg
        md = med[ei, hi] * sg
        nbn = nbN[ei, hi]
        plat = np.where(nbn > 0, np.where(lg, nbL[ei, hi], nbS[ei, hi]) / np.where(nbn > 0, nbn, 1), 0.0)
        idx_ok = np.flatnonzero(okm)
        if len(idx_ok) == 0:
            continue
        nk = nn[idx_ok].astype(int); wk = wins[idx_ok].astype(int); hk = hh[idx_ok].astype(int); w1k = w1[idx_ok].astype(int)
        nmax = int(nk.max())
        while len(binom_tab) <= nmax:
            m_ = len(binom_tab)
            binom_tab.append([_binom_p(k_, m_) for k_ in range(m_ + 1)])
        pval = np.array([binom_tab[a_][b_] for a_, b_ in zip(nk, wk)])
        with np.errstate(all="ignore"):
            wr_h1 = np.where(hk > 0, w1k / np.maximum(hk, 1), np.nan)
            wr_h2 = (wk - w1k) / (nk - hk)
        avg_k = sumR[ei[idx_ok], hi[idx_ok]] / nk * sg[idx_ok]
        md_k = md[idx_ok]; plateau = plat[idx_ok]
        oos_n = np.array([plist[k].oos_n_trades for k in idx_ok])
        oos_wr = np.array([plist[k].oos_winrate or 0 for k in idx_ok], dtype=float)
        oos_avg = np.array([plist[k].oos_avg_ret or 0 for k in idx_ok], dtype=float)
        c_oos = (oos_n >= 5) & (oos_wr >= 0.65) & (oos_avg > cost_buffer)
        c_hist = (wr_h1 >= QUANT_HALF) & (wr_h2 >= QUANT_HALF)
        c_sig = pval < QUANT_P
        c_plat = plateau >= QUANT_PLAT
        c_econ = (avg_k >= QUANT_ECON) & (md_k > 0)
        stars = c_oos.astype(int) + c_hist + c_sig + c_plat + c_econ
        stars = np.where(c_oos & c_sig, stars, np.minimum(stars, 3))
        score = stars - np.log10(np.maximum(pval, 1e-12)) / 10 + plateau / 10
        for j, k in enumerate(idx_ok):
            ps = plist[k]
            rows.append((sym, "Long" if lg[k] else "Short", ps.entry_doy, ps.exit_doy, float(score[j]),
                         (ps, int(nk[j]), int(wk[j]), int(hk[j]), int(w1k[j]), float(wr_h1[j]), float(wr_h2[j]),
                          float(avg_k[j]), float(plateau[j]), float(pval[j]), int(stars[j]),
                          bool(c_oos[j]), bool(c_hist[j]), bool(c_sig[j]), bool(c_plat[j]), bool(c_econ[j]),
                          n_folds_total)))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r[:5] for r in rows], columns=["Symbol", "Richtung", "_entry_doy", "_exit_doy", "_score"])
    df["_pl"] = [r[5] for r in rows]
    # Cluster-Dedup: überlappende Fenster = ein Setup
    df = df.sort_values(["Symbol", "Richtung", "_entry_doy"]).reset_index(drop=True)
    key_change = (df["Symbol"].ne(df["Symbol"].shift()) | df["Richtung"].ne(df["Richtung"].shift()))
    gap = df["_entry_doy"].diff().fillna(10**9) > dedup_days
    cid = (key_change | gap).cumsum()
    cluster_best = {c: i for c, i in df.groupby(cid, sort=False)["_score"].idxmax().items()}
    df = df.loc[sorted(cluster_best.values())]

    def _full_row(r) -> dict:
        ps, nk, wk, hk, w1k, wr_h1, wr_h2, avg_k, plateau, pval, stars, c_oos, c_hist, c_sig, c_plat, c_econ, nft = r["_pl"]
        hold = ps.exit_doy - ps.entry_doy
        start = pd.Timestamp("2001-01-01") + pd.Timedelta(days=ps.entry_doy - 1)
        end = start + pd.Timedelta(days=hold)
        return {
            "Symbol": r["Symbol"], "Richtung": r["Richtung"],
            "Fenster": f"{start:%d.%m.} – {end:%d.%m.}", "Tage": hold, "Quant ⭐": stars, "Jahre": nk,
            "WR gesamt %": round(wk / nk * 100, 1),
            "WR 1. Hälfte %": round(wr_h1 * 100, 1), "WR 2. Hälfte %": round(wr_h2 * 100, 1),
            "Wilson-Untergrenze %": round(_wilson_ci(wk, nk)[0] * 100, 1),
            "Ø Return gesamt %": round(avg_k * 100, 2),
            "OOS Trades": ps.oos_n_trades, "OOS WR %": round((ps.oos_winrate or 0) * 100, 1),
            "OOS Ø Return %": round((ps.oos_avg_ret or 0) * 100, 2),
            "Folds gefunden": f"{ps.n_folds_found}/{nft}", "Plateau %": round(plateau * 100),
            "p-Wert": round(pval, 4),
            "★ OOS": "✅" if c_oos else "—", "★ Beide Hälften": "✅" if c_hist else "—",
            "★ Signifikanz": "✅" if c_sig else "—", "★ Plateau": "✅" if c_plat else "—",
            "★ Ökonomie": "✅" if c_econ else "—",
            "_entry_doy": ps.entry_doy, "_exit_doy": ps.exit_doy, "_score": r["_score"],
        }
    df = pd.DataFrame([_full_row(r) for _, r in df.iterrows()])
    if df.empty:
        return df
    df = df[df["Quant ⭐"] >= min_stars]
    return df.sort_values(["Quant ⭐", "_score"], ascending=False).reset_index(drop=True)


def add_currency_clusters(df: pd.DataFrame, min_overlap: float = 0.5) -> pd.DataFrame:
    """
    Fasst Setups zu Clustern zusammen, die dieselbe Waehrungswette sind:
    gleiche Waehrung in gleicher Rolle (stark/schwach) und zeitlich ueberlappende Fenster
    (Ueberlappung >= min_overlap des kuerzeren Fensters). Beispiel: CADJPY Long + GBPJPY Long
    Ende Maerz = beide "JPY schwach" -> ein Cluster, Risiko nur einmal.
    """
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)
    legs = []
    for i, r in df.iterrows():
        sym = str(r["Symbol"]).upper()
        if len(sym) != 6:
            legs.append(set()); continue
        b, q = sym[:3], sym[3:]
        strong, weak = (b, q) if r["Richtung"] == "Long" else (q, b)
        legs.append({f"{strong} stark", f"{weak} schwach"})
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            common = legs[i] & legs[j]
            if not common:
                continue
            a0, a1 = df.at[i, "_entry_doy"], df.at[i, "_exit_doy"]
            b0, b1 = df.at[j, "_entry_doy"], df.at[j, "_exit_doy"]
            ov = min(a1, b1) - max(a0, b0)
            if ov > 0 and ov >= min_overlap * min(a1 - a0, b1 - b0):
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(df))]
    labels = {}
    for rt in set(roots):
        members = [i for i in range(len(df)) if roots[i] == rt]
        if len(members) == 1:
            labels[rt] = "—"
            continue
        common = set.intersection(*[legs[m] for m in members]) or set.union(*[legs[m] for m in members])
        s0 = min(df.at[m, "_entry_doy"] for m in members); s1 = max(df.at[m, "_exit_doy"] for m in members)
        d0 = (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(s0) - 1)).strftime("%d.%m.")
        d1 = (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(s1) - 1)).strftime("%d.%m.")
        labels[rt] = f"{' / '.join(sorted(common))} · {d0}–{d1} ({len(members)} Paare = 1 Wette)"
    df["Cluster"] = [labels[rt] for rt in roots]
    return df


def shuffled_symbol_dfs(symbol_dfs: dict[str, pd.DataFrame], seed: int = 7) -> dict[str, pd.DataFrame]:
    """
    Nullmodell für den Zufalls-Check: Tagesrenditen werden zufällig gemischt — gemeinsam für alle
    Symbole (gleiche Permutation je Kalendertag), damit die Korrelation zwischen Paaren erhalten bleibt.
    Volatilität und Drift bleiben gleich, echte Saisonalität ist zerstört.
    """
    rng = np.random.default_rng(seed)
    idx = sorted(set().union(*[set(d.index) for d in symbol_dfs.values()]))
    perm = rng.permutation(len(idx)); remap = dict(zip(idx, [idx[i] for i in perm]))
    out = {}
    for sym, d in symbol_dfs.items():
        d = d.sort_index()
        lr = np.log(d["close"]).diff().fillna(0.0)
        hl = np.log(d["high"]) - np.log(d["close"]); ll = np.log(d["low"]) - np.log(d["close"])
        src = pd.Index([remap[t] for t in d.index])
        ok = src.isin(d.index)
        r = np.where(ok, lr.reindex(src).values, 0.0)
        h_ = np.where(ok, hl.reindex(src).values, 0.0); l_ = np.where(ok, ll.reindex(src).values, 0.0)
        c = np.exp(np.log(d["close"].iloc[0]) + np.cumsum(r))
        out[sym] = pd.DataFrame({"open": c, "high": c * np.exp(h_), "low": c * np.exp(l_), "close": c}, index=d.index)
    return out


def wfa_results_to_dataframe(wfa_result: WFAResult) -> pd.DataFrame:
    """
    Konvertiert PatternWFAStats-Liste in ein DataFrame für die Anzeige.
    """
    rows = []
    for ps in wfa_result.pattern_stats:
        rows.append({
            "Symbol": ps.symbol,
            "Entry DOY": ps.entry_doy,
            "Exit DOY": ps.exit_doy,
            "Richtung": "Long" if ps.direction == "long" else "Short",
            "Folds (IS-Kandidat)": ps.n_folds_found,
            "Folds (OOS-Trade)": ps.n_folds_oos,
            "OOS Winrate %": round(ps.oos_winrate * 100, 1) if ps.oos_winrate is not None else None,
            "IS Winrate Ø %": round(ps.avg_is_winrate * 100, 1) if ps.avg_is_winrate is not None else None,
            "OOS Ø Return %": round(ps.oos_avg_ret * 100, 2) if ps.oos_avg_ret is not None else None,
            "OOS n Trades": ps.oos_n_trades,
            "Status": "✅ OOS-validiert" if ps.validated else "⚠️ Nur IS",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["Status", "OOS Winrate %"],
            ascending=[True, False]  # ✅ vor ⚠️ (alphabetisch), dann nach OOS-WR
        ).reset_index(drop=True)
    return df
