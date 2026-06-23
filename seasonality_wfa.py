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

        for fold_i, oos_year in enumerate(oos_years):
            # IS-Fenster = alle Jahre VOR oos_year
            is_years = sorted_years[sorted_years < oos_year]
            if len(is_years) < min_is_years:
                # Sollte durch Konstruktion nicht passieren, aber zur Sicherheit
                continue

            # ---- Schritt a: Scan auf IS-Jahren ----
            candidates = _scan_is_window(
                year_data, sorted_years, is_years,
                holding_periods, directions,
                min_winrate, min_trades,
            )

            fold_res = FoldResult(
                oos_year=int(oos_year),
                n_candidates=len(candidates),
            )

            # ---- Schritt b/c: Kandidaten auf OOS-Jahr anwenden ----
            for cand in candidates:
                key = (cand["entry_doy"], cand["exit_doy"], cand["direction"])
                t = _single_trade(year_data, int(oos_year),
                                  cand["entry_doy"], cand["exit_doy"])

                # Akkumulator aktualisieren (Muster wurde in diesem Fold gefunden)
                if key not in pattern_acc:
                    pattern_acc[key] = {
                        "folds_found": 0,
                        "oos_wins": 0,
                        "oos_n": 0,
                        "oos_ret_sum": 0.0,
                        "is_wr_sum": 0.0,
                    }
                acc = pattern_acc[key]
                acc["folds_found"] += 1
                acc["is_wr_sum"] += cand["is_winrate"]

                oos_ret: float | None = None
                oos_win: bool | None = None
                if t is not None:
                    raw_ret = t[0] if cand["direction"] == "long" else t[1]
                    oos_win = raw_ret > 0
                    oos_ret = raw_ret
                    acc["oos_n"] += 1
                    acc["oos_ret_sum"] += raw_ret
                    if oos_win:
                        acc["oos_wins"] += 1

                fold_res.trades.append({
                    "entry_doy": cand["entry_doy"],
                    "exit_doy":  cand["exit_doy"],
                    "direction": cand["direction"],
                    "is_winrate": cand["is_winrate"],
                    "oos_win": oos_win,
                    "oos_ret": oos_ret,
                })

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
    for ps in wfa_result.pattern_stats:
        if (ps.symbol == symbol
                and ps.entry_doy == entry_doy
                and ps.exit_doy == exit_doy
                and ps.direction == direction):
            if ps.validated:
                return "✅ OOS-validiert"
            return "⚠️ Nur IS"
    return ""


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
