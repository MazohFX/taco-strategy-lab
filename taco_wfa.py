"""
TACO Walk-Forward Analysis (WFA)
=================================
Portierung der Pine Script v5 Strategie "TACO Rebuild Asset Comparison Oscillator"
nach Python mit vollständiger Walk-Forward-Analyse über 5 Folds.

Datenquelle: yfinance
- Zuverlässig, kostenlos, lange Historien für Forex und DXY
- Forex-Paare über yfinance: z.B. EURUSD=X
- DXY: DX-Y.NYB (NYBOT Dollar Index)

Warum yfinance?
  - Keine API-Key nötig
  - Tägliche Daten ab ~1993 für DXY verfügbar
  - Einfache Installation: pip install yfinance pandas numpy matplotlib scipy
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats as scipy_stats

# =============================================================================
# KONFIGURATION — hier alle Parameter zentral änderbar
# =============================================================================

# Hauptasset (yfinance Ticker)
MAIN_ASSET = "EURUSD=X"

# Vergleichsbasket
USE_DXY    = True
USE_GOLD   = False
USE_BONDS  = False
USE_CUSTOM = False
CUSTOM_TICKER = "EURUSD=X"

# yfinance Ticker-Mapping für Referenzassets
TICKER_DXY   = "DX-Y.NYB"
TICKER_GOLD  = "GC=F"
TICKER_BONDS = "ZN=F"

# Gesamter Datenabruf-Zeitraum
DATA_START = "1995-01-01"
DATA_END   = "2010-12-31"

# Walk-Forward-Struktur
IS_YEARS   = 10
OOS_YEARS  = 1

# 5 Folds: IS-Startjahr, OOS-Jahr
FOLDS = [
    (1996, 2006),
    (1997, 2007),
    (1998, 2008),
    (1999, 2009),
    (2000, 2010),
]

# Fix-Parameter (werden NICHT optimiert)
SMOOTH_LENGTH    = 5
MODE             = "Ratio Z-Score"
TRADE_DIRECTION  = "Long & Short"
RISK_PCT         = 1.0
ENABLE_TP        = True
TP_MODE          = "Risk Reward"
FIXED_TP_PCT     = 1.30
EXIT_ON_ZERO     = False
ENABLE_TIME_EXIT = False
EXIT_AFTER_BARS  = 20
CONTRACT_PV      = 1.0
INITIAL_CAPITAL  = 10_000.0
COMMISSION_PCT   = 0.05

# Grid-Search Suchraum
GRID = {
    "cycleLength":  [5, 10, 15, 20, 30],
    "softness":     [0.8, 1.0, 1.35, 1.7, 2.2],
    "stopLossPct":  [0.3, 0.5, 0.65, 1.0, 1.5],
    "rrMultiple":   [1.0, 1.5, 2.0, 3.0],
    "level":        [50, 75, 90],
}

# Optimierungsziel: "profit_factor", "sharpe", "calmar"
OPTIMIZATION_METRIC = "profit_factor"

# Guardrails für die Optimierung
MIN_TRADES_IS  = 8
MAX_DD_IS      = 25.0


# =============================================================================
# DATEN LADEN
# =============================================================================

def load_data() -> dict[str, pd.Series]:
    tickers = {
        "asset": MAIN_ASSET,
        "dxy":   TICKER_DXY,
        "gold":  TICKER_GOLD,
        "bonds": TICKER_BONDS,
    }
    if USE_CUSTOM:
        tickers["custom"] = CUSTOM_TICKER

    data = {}
    for name, ticker in tickers.items():
        print(f"  Lade {name} ({ticker}) ...", end=" ")
        df = yf.download(ticker, start=DATA_START, end=DATA_END,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"KEINE DATEN — überspringe {ticker}")
            data[name] = pd.Series(dtype=float)
        else:
            s = df["Close"].squeeze()
            s = s.dropna()
            print(f"OK  ({s.index[0].date()} bis {s.index[-1].date()}, {len(s)} Bars)")
            data[name] = s

    return data


def align_data(data: dict) -> pd.DataFrame:
    frames = {k: v for k, v in data.items() if not v.empty}
    df = pd.DataFrame(frames)
    df = df.ffill().dropna()
    return df


# =============================================================================
# INDIKATOR-BERECHNUNG
# =============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rolling_stdev(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).std(ddof=1)


def f_tanh(x: pd.Series) -> pd.Series:
    clipped = x.clip(-10.0, 10.0)
    e = np.exp(2.0 * clipped)
    return (e - 1.0) / (e + 1.0)


def f_ratio_z(asset: pd.Series, comp: pd.Series, cycle_length: int) -> pd.Series:
    ratio = asset / comp
    mean  = ratio.rolling(cycle_length).mean()
    dev   = rolling_stdev(ratio, cycle_length)
    z = (ratio - mean) / dev.replace(0, np.nan)
    return z.fillna(0.0)


def f_spread_z(asset: pd.Series, comp: pd.Series, cycle_length: int) -> pd.Series:
    asset_ret = np.log(asset / asset.shift(1))
    comp_ret  = np.log(comp  / comp.shift(1))
    spread    = asset_ret - comp_ret
    spread_std = rolling_stdev(spread, cycle_length)
    z = ema(spread, cycle_length) / spread_std.replace(0, np.nan)
    return z.fillna(0.0)


def f_score(asset: pd.Series, comp: pd.Series, cycle_length: int) -> pd.Series:
    if MODE == "Ratio Z-Score":
        return f_ratio_z(asset, comp, cycle_length)
    else:
        return f_spread_z(asset, comp, cycle_length)


def compute_oscillator(df: pd.DataFrame,
                       cycle_length: int,
                       softness: float,
                       smooth_length: int) -> pd.Series:
    asset = df["asset"]
    scores = []

    if USE_DXY and "dxy" in df.columns:
        scores.append(f_score(asset, df["dxy"], cycle_length))
    if USE_GOLD and "gold" in df.columns:
        scores.append(f_score(asset, df["gold"], cycle_length))
    if USE_BONDS and "bonds" in df.columns:
        scores.append(f_score(asset, df["bonds"], cycle_length))
    if USE_CUSTOM and "custom" in df.columns:
        scores.append(f_score(asset, df["custom"], cycle_length))

    if not scores:
        scores.append(f_score(asset, df["dxy"], cycle_length))

    z = sum(scores) / len(scores)
    osc_raw = 100.0 * f_tanh(z / softness)
    osc = ema(osc_raw, smooth_length)
    return osc


# =============================================================================
# BACKTEST-ENGINE
# =============================================================================

@dataclass
class Trade:
    direction:   str
    entry_date:  pd.Timestamp
    entry_price: float
    stop_price:  float
    tp_price:    Optional[float]
    qty:         float
    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: str                    = ""
    pnl:         float                  = 0.0
    pnl_pct:     float                  = 0.0


def run_backtest(df: pd.DataFrame,
                 osc: pd.Series,
                 params: dict,
                 start_year: int,
                 end_year: int,
                 initial_capital: float = INITIAL_CAPITAL) -> tuple[list[Trade], pd.Series]:
    cycle_length  = params["cycleLength"]
    stop_loss_pct = params["stopLossPct"]
    rr_multiple   = params["rrMultiple"]
    upper_level   = params["level"]
    lower_level   = -params["level"]

    osc_prev = osc.shift(1)
    long_signal  = ((osc < lower_level) & (osc > osc_prev)) | \
                   ((osc_prev < lower_level) & (osc >= lower_level))
    short_signal = ((osc > upper_level) & (osc < osc_prev)) | \
                   ((osc_prev > upper_level) & (osc <= upper_level))

    mask = (osc.index.year >= start_year) & (osc.index.year <= end_year)
    long_signal  = long_signal  & mask
    short_signal = short_signal & mask

    allow_long  = TRADE_DIRECTION in ("Long & Short", "Long Only")
    allow_short = TRADE_DIRECTION in ("Long & Short", "Short Only")

    equity     = initial_capital
    trades     = []
    equity_ts  = pd.Series(index=osc.index, dtype=float)
    equity_ts.iloc[0] = equity

    pos: Optional[Trade] = None
    entry_bar_idx: Optional[int] = None

    bars   = osc.index.tolist()
    prices = df["asset"].reindex(osc.index).ffill()

    for i, date in enumerate(bars):
        close       = prices.iloc[i]
        current_osc = osc.iloc[i]

        if np.isnan(close) or np.isnan(current_osc):
            equity_ts.iloc[i] = equity
            continue

        if pos is not None:
            bar_in_trade = i - entry_bar_idx
            closed = False

            if pos.direction == "Long":
                if close <= pos.stop_price:
                    pos.exit_price  = pos.stop_price
                    pos.exit_reason = "Stop"
                    closed = True
                elif ENABLE_TP and pos.tp_price is not None and close >= pos.tp_price:
                    pos.exit_price  = pos.tp_price
                    pos.exit_reason = "TP"
                    closed = True
                elif EXIT_ON_ZERO:
                    osc_prev_val = osc.iloc[i-1] if i > 0 else current_osc
                    if osc_prev_val < 0 and current_osc >= 0:
                        pos.exit_price  = close
                        pos.exit_reason = "Zero Exit"
                        closed = True

            elif pos.direction == "Short":
                if close >= pos.stop_price:
                    pos.exit_price  = pos.stop_price
                    pos.exit_reason = "Stop"
                    closed = True
                elif ENABLE_TP and pos.tp_price is not None and close <= pos.tp_price:
                    pos.exit_price  = pos.tp_price
                    pos.exit_reason = "TP"
                    closed = True
                elif EXIT_ON_ZERO:
                    osc_prev_val = osc.iloc[i-1] if i > 0 else current_osc
                    if osc_prev_val > 0 and current_osc <= 0:
                        pos.exit_price  = close
                        pos.exit_reason = "Zero Exit"
                        closed = True

            if not closed and ENABLE_TIME_EXIT and bar_in_trade >= EXIT_AFTER_BARS:
                pos.exit_price  = close
                pos.exit_reason = "Time Exit"
                closed = True

            if not closed and date.year > end_year:
                pos.exit_price  = close
                pos.exit_reason = "End Year Exit"
                closed = True

            if closed:
                pos.exit_date = date
                if pos.direction == "Long":
                    raw_pnl = (pos.exit_price - pos.entry_price) * pos.qty * CONTRACT_PV
                else:
                    raw_pnl = (pos.entry_price - pos.exit_price) * pos.qty * CONTRACT_PV

                commission = (pos.entry_price * pos.qty * CONTRACT_PV * COMMISSION_PCT / 100.0 +
                              pos.exit_price  * pos.qty * CONTRACT_PV * COMMISSION_PCT / 100.0)
                pos.pnl     = raw_pnl - commission
                pos.pnl_pct = pos.pnl / equity * 100.0
                equity     += pos.pnl
                trades.append(pos)
                pos           = None
                entry_bar_idx = None

        if pos is None:
            risk_cash = equity * RISK_PCT / 100.0

            if long_signal.iloc[i] and allow_long:
                entry_price = close
                stop_price  = entry_price * (1.0 - stop_loss_pct / 100.0)
                risk_points = entry_price - stop_price
                if risk_points > 0:
                    qty      = risk_cash / (risk_points * CONTRACT_PV)
                    tp_rr    = entry_price + risk_points * rr_multiple
                    tp_fixed = entry_price * (1.0 + FIXED_TP_PCT / 100.0)
                    tp       = (tp_rr if TP_MODE == "Risk Reward" else tp_fixed) if ENABLE_TP else None
                    pos = Trade("Long", date, entry_price, stop_price, tp, qty)
                    entry_bar_idx = i

            elif short_signal.iloc[i] and allow_short:
                entry_price = close
                stop_price  = entry_price * (1.0 + stop_loss_pct / 100.0)
                risk_points = stop_price - entry_price
                if risk_points > 0:
                    qty      = risk_cash / (risk_points * CONTRACT_PV)
                    tp_rr    = entry_price - risk_points * rr_multiple
                    tp_fixed = entry_price * (1.0 - FIXED_TP_PCT / 100.0)
                    tp       = (tp_rr if TP_MODE == "Risk Reward" else tp_fixed) if ENABLE_TP else None
                    pos = Trade("Short", date, entry_price, stop_price, tp, qty)
                    entry_bar_idx = i

        equity_ts.iloc[i] = equity

    if pos is not None:
        last_price = prices.iloc[-1]
        pos.exit_date   = bars[-1]
        pos.exit_price  = last_price
        pos.exit_reason = "End of Period"
        if pos.direction == "Long":
            raw_pnl = (pos.exit_price - pos.entry_price) * pos.qty * CONTRACT_PV
        else:
            raw_pnl = (pos.entry_price - pos.exit_price) * pos.qty * CONTRACT_PV
        commission = (pos.entry_price * pos.qty * CONTRACT_PV * COMMISSION_PCT / 100.0 +
                      pos.exit_price  * pos.qty * CONTRACT_PV * COMMISSION_PCT / 100.0)
        pos.pnl     = raw_pnl - commission
        pos.pnl_pct = pos.pnl / equity * 100.0
        equity     += pos.pnl
        trades.append(pos)

    equity_ts = equity_ts.ffill()
    return trades, equity_ts


# =============================================================================
# KENNZAHLEN-BERECHNUNG
# =============================================================================

def compute_metrics(trades: list[Trade], equity_ts: pd.Series, initial_cap: float) -> dict:
    if not trades:
        return {
            "profit_factor": 0.0, "sharpe": 0.0, "calmar": 0.0,
            "num_trades": 0, "winrate": 0.0, "max_dd_pct": 0.0,
            "total_return_pct": 0.0,
        }

    wins  = [t.pnl for t in trades if t.pnl > 0]
    loses = [t.pnl for t in trades if t.pnl <= 0]

    gross_profit  = sum(wins)
    gross_loss    = abs(sum(loses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    winrate       = len(wins) / len(trades) * 100.0

    eq          = equity_ts.dropna()
    rolling_max = eq.cummax()
    dd          = (eq - rolling_max) / rolling_max * 100.0
    max_dd_pct  = abs(dd.min())

    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    total_days       = (eq.index[-1] - eq.index[0]).days
    years            = total_days / 365.25 if total_days > 0 else 1.0
    total_return_pct = (eq.iloc[-1] / initial_cap - 1) * 100.0
    ann_return       = (1 + total_return_pct / 100) ** (1 / years) - 1 if years > 0 else 0.0
    calmar           = ann_return * 100 / max_dd_pct if max_dd_pct > 0 else 0.0

    return {
        "profit_factor":    round(profit_factor, 3),
        "sharpe":           round(sharpe, 3),
        "calmar":           round(calmar, 3),
        "num_trades":       len(trades),
        "winrate":          round(winrate, 1),
        "max_dd_pct":       round(max_dd_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
    }


# =============================================================================
# AUFGABE 1 — GRID-SEARCH MIT ROBUSTER PLATEAU-AUSWAHL
# =============================================================================

def grid_search_dual(df: pd.DataFrame,
                     is_start: int,
                     is_end: int,
                     metric: str = OPTIMIZATION_METRIC
                     ) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    """
    Durchsucht alle Parameter-Kombinationen und gibt ZWEI Auswahlen zurück:
      - robust_params/metrics: Kombination mit bestem Nachbar-Durchschnitt
        (selbst + bis zu 8 Nachbarn ±1 in cycleLength/stopLossPct)
      - naive_params/metrics:  Kombination mit absolutem Einzelscore-Maximum
    Guardrails (MIN_TRADES_IS, MAX_DD_IS) werden als Vorfilter angewendet.
    """
    cycle_list = GRID["cycleLength"]
    sl_list    = GRID["stopLossPct"]
    n_cycle    = len(cycle_list)
    n_sl       = len(sl_list)

    # scores_map: (i_c, i_s, i_sl, i_rr, i_lv) -> score (nur gültige Kombinationen)
    scores_map  = {}
    results_map = {}  # selber Key -> (params, metrics)

    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))
    print(f"    Grid-Search: {len(combos)} Kombinationen ...")

    for combo in combos:
        params = dict(zip(keys, combo))
        i_c  = cycle_list.index(params["cycleLength"])
        i_s  = GRID["softness"].index(params["softness"])
        i_sl = sl_list.index(params["stopLossPct"])
        i_rr = GRID["rrMultiple"].index(params["rrMultiple"])
        i_lv = GRID["level"].index(params["level"])
        key  = (i_c, i_s, i_sl, i_rr, i_lv)

        osc = compute_oscillator(df, params["cycleLength"], params["softness"], SMOOTH_LENGTH)
        trades, eq_ts = run_backtest(df, osc, params,
                                     start_year=is_start, end_year=is_end,
                                     initial_capital=INITIAL_CAPITAL)
        m = compute_metrics(trades, eq_ts, INITIAL_CAPITAL)

        # Guardrails als Vorfilter
        if m["num_trades"] < MIN_TRADES_IS:
            continue
        if m["max_dd_pct"] > MAX_DD_IS:
            continue

        scores_map[key]  = m[metric]
        results_map[key] = (params.copy(), m.copy())

    if not scores_map:
        return None, None, None, None

    # ── Naive: absolutes Maximum ──────────────────────────────────────────────
    naive_key     = max(scores_map, key=lambda k: scores_map[k])
    naive_score   = scores_map[naive_key]
    naive_params, naive_metrics = results_map[naive_key]

    # ── Robust: bester Nachbar-Durchschnitt ──────────────────────────────────
    # Nachbarn: ±1 in cycleLength-Dim (i_c) und ±1 in stopLossPct-Dim (i_sl)
    robust_best_key = None
    robust_best_avg = -np.inf

    for key in scores_map:
        i_c, i_s, i_sl, i_rr, i_lv = key
        neighborhood = [scores_map[key]]  # selbst

        for dc in (-1, 0, 1):
            for dsl in (-1, 0, 1):
                if dc == 0 and dsl == 0:
                    continue
                nc  = i_c  + dc
                nsl = i_sl + dsl
                if 0 <= nc < n_cycle and 0 <= nsl < n_sl:
                    nb_key = (nc, i_s, nsl, i_rr, i_lv)
                    if nb_key in scores_map:
                        neighborhood.append(scores_map[nb_key])

        avg = float(np.mean(neighborhood))
        if avg > robust_best_avg:
            robust_best_avg = avg
            robust_best_key = key

    robust_score  = scores_map[robust_best_key]
    robust_params, robust_metrics = results_map[robust_best_key]

    # Abweichung robust vs. naiv loggen
    if naive_score != 0:
        divergence_pct = (naive_score - robust_score) / abs(naive_score) * 100.0
    else:
        divergence_pct = 0.0
    print(f"    Naiver Score ({metric}): {naive_score:.3f} | "
          f"Robuster Score: {robust_score:.3f} | "
          f"Robustheits-Opfer: {divergence_pct:.1f}%")

    return robust_params, robust_metrics, naive_params, naive_metrics


# Legacy-Wrapper (behält Kompatibilität zu eventuell externen Aufrufen)
def grid_search(df: pd.DataFrame, is_start: int, is_end: int,
                metric: str = OPTIMIZATION_METRIC) -> tuple[dict, dict]:
    robust_p, robust_m, _, _ = grid_search_dual(df, is_start, is_end, metric)
    return robust_p, robust_m


# =============================================================================
# WALK-FORWARD-ANALYSE (DUAL: ROBUST + NAIV)
# =============================================================================

def _concat_oos(oos_equities: list[pd.Series],
                fold_transitions: list[pd.Timestamp]) -> pd.Series:
    """Verkettet OOS-Equity-Stücke zu einer durchgehenden Kurve."""
    if not oos_equities:
        return pd.Series(dtype=float)
    combined = pd.concat(oos_equities)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def run_wfa_dual(df: pd.DataFrame
                 ) -> tuple[list[dict], list[dict], pd.Series, pd.Series, list[pd.Timestamp]]:
    """
    Führt die WFA für BEIDE Auswahlmodi (robust + naiv) in EINEM Durchlauf durch.
    Pro Fold wird grid_search_dual() einmalig aufgerufen (keine doppelte Rechenzeit).

    Rückgabe:
      robust_results, naive_results   — fold-Ergebnisse pro Modus
      robust_equity, naive_equity     — verkettete OOS-Equity-Kurven
      fold_transitions                — Datumsgrenzen zwischen Folds (für Markierungen)
    """
    robust_results   = []
    naive_results    = []
    robust_equities  = []
    naive_equities   = []
    fold_transitions = []
    running_cap_r    = INITIAL_CAPITAL
    running_cap_n    = INITIAL_CAPITAL

    for fold_idx, (is_start, oos_year) in enumerate(FOLDS):
        is_end = is_start + IS_YEARS - 1
        print(f"\n  Fold {fold_idx+1}: IS {is_start}–{is_end}  →  OOS {oos_year}")

        robust_p, robust_m_is, naive_p, naive_m_is = grid_search_dual(df, is_start, is_end)

        if robust_p is None:
            print(f"    WARNUNG: Keine gültige Kombination für Fold {fold_idx+1}!")
            continue

        print(f"    Robuste Params: {robust_p}")
        print(f"    Naive   Params: {naive_p}")
        print(f"    IS (robust): PF={robust_m_is['profit_factor']}  "
              f"Trades={robust_m_is['num_trades']}  MaxDD={robust_m_is['max_dd_pct']}%")

        for mode, params, m_is, equities, cap_ref in [
            ("robust", robust_p, robust_m_is, robust_equities, None),
            ("naive",  naive_p,  naive_m_is,  naive_equities,  None),
        ]:
            pass  # handled individually below to track running capitals

        # ── Robust OOS ────────────────────────────────────────────────────────
        osc_r = compute_oscillator(df, robust_p["cycleLength"], robust_p["softness"], SMOOTH_LENGTH)
        r_trades, r_eq = run_backtest(df, osc_r, robust_p,
                                      start_year=oos_year, end_year=oos_year,
                                      initial_capital=running_cap_r)
        r_metrics = compute_metrics(r_trades, r_eq, running_cap_r)

        mask_r = r_eq.index.year == oos_year
        r_eq_y = r_eq[mask_r]
        if not r_eq_y.empty:
            running_cap_r = r_eq_y.iloc[-1]
            robust_equities.append(r_eq_y)
            fold_transitions.append(r_eq_y.index[0])

        robust_results.append({
            "fold":       fold_idx + 1,
            "is_period":  f"{is_start}–{is_end}",
            "oos_year":   oos_year,
            "params":     robust_p,
            "is_metrics": robust_m_is,
            "oos_metrics": r_metrics,
            "oos_trades": r_trades,
        })

        # ── Naive OOS ────────────────────────────────────────────────────────
        osc_n = compute_oscillator(df, naive_p["cycleLength"], naive_p["softness"], SMOOTH_LENGTH)
        n_trades, n_eq = run_backtest(df, osc_n, naive_p,
                                      start_year=oos_year, end_year=oos_year,
                                      initial_capital=running_cap_n)
        n_metrics = compute_metrics(n_trades, n_eq, running_cap_n)

        mask_n = n_eq.index.year == oos_year
        n_eq_y = n_eq[mask_n]
        if not n_eq_y.empty:
            running_cap_n = n_eq_y.iloc[-1]
            naive_equities.append(n_eq_y)

        naive_results.append({
            "fold":       fold_idx + 1,
            "is_period":  f"{is_start}–{is_end}",
            "oos_year":   oos_year,
            "params":     naive_p,
            "is_metrics": naive_m_is,
            "oos_metrics": n_metrics,
            "oos_trades": n_trades,
        })

        print(f"    OOS robust: PF={r_metrics['profit_factor']}  "
              f"Trades={r_metrics['num_trades']}  MaxDD={r_metrics['max_dd_pct']}%")
        print(f"    OOS naive:  PF={n_metrics['profit_factor']}  "
              f"Trades={n_metrics['num_trades']}  MaxDD={n_metrics['max_dd_pct']}%")

    robust_equity = _concat_oos(robust_equities, fold_transitions)
    naive_equity  = _concat_oos(naive_equities, [])

    return robust_results, naive_results, robust_equity, naive_equity, fold_transitions


# =============================================================================
# AUFGABE 2 — GEPOOLTE OOS-STATISTIKEN + WILCOXON + WILSON CI
# =============================================================================

def wilson_ci(n_wins: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson Score Intervall für Binomialproportion."""
    if n_total == 0:
        return (0.0, 0.0)
    z     = scipy_stats.norm.ppf((1 + confidence) / 2)
    p_hat = n_wins / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denom
    return (center - margin, center + margin)


def compute_oos_pool_stats(fold_results: list[dict],
                           combined_equity: pd.Series) -> dict:
    """
    Berechnet Statistiken über ALLE gepoolten OOS-Trades.
    Wilcoxon Signed-Rank Test gegen H0: median PnL = 0.
    Wilson CI für Win Rate.
    Avg R = durchschnittlicher Trade PnL / mittleres Trade-Risiko (Risk-$).
    """
    all_trades: list[Trade] = []
    for r in fold_results:
        all_trades.extend(r["oos_trades"])

    if not all_trades:
        return {"error": "Keine OOS-Trades vorhanden"}

    pnl_list  = [t.pnl for t in all_trades]
    wins      = [p for p in pnl_list if p > 0]
    n_trades  = len(pnl_list)
    n_wins    = len(wins)

    gross_profit  = sum(wins)
    gross_loss    = abs(sum(p for p in pnl_list if p <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    winrate       = n_wins / n_trades * 100.0

    # Avg R: PnL / Risiko pro Trade (Risiko = Distanz Entry→SL × Qty × PV)
    r_values = []
    for t in all_trades:
        if t.direction == "Long":
            risk = (t.entry_price - t.stop_price) * t.qty * CONTRACT_PV
        else:
            risk = (t.stop_price - t.entry_price) * t.qty * CONTRACT_PV
        if risk > 0:
            r_values.append(t.pnl / risk)
    avg_r = float(np.mean(r_values)) if r_values else 0.0

    # Max Drawdown auf der verketteten Equity-Kurve
    if not combined_equity.empty:
        eq          = combined_equity.dropna()
        rolling_max = eq.cummax()
        dd          = (eq - rolling_max) / rolling_max * 100.0
        max_dd_pct  = abs(dd.min())
    else:
        max_dd_pct = 0.0

    # Wilcoxon Signed-Rank Test (PnL gegen 0, einseitig: Edge > 0)
    if n_trades >= 8:
        try:
            w_stat, w_p = scipy_stats.wilcoxon(pnl_list, alternative="greater")
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")
        print("    Hinweis: Zu wenige Trades für Wilcoxon-Test (< 8)")

    # Wilson Confidence Interval (95%) für Win Rate
    wi_lo, wi_hi = wilson_ci(n_wins, n_trades, confidence=0.95)

    return {
        "n_trades":     n_trades,
        "profit_factor": round(profit_factor, 3),
        "winrate_pct":  round(winrate, 1),
        "avg_r":        round(avg_r, 3),
        "max_dd_pct":   round(max_dd_pct, 2),
        "wilcoxon_stat": round(w_stat, 3) if not math.isnan(w_stat) else "n/a",
        "wilcoxon_p":   round(w_p, 4) if not math.isnan(w_p) else "n/a",
        "wilson_ci_lo": round(wi_lo * 100, 1),
        "wilson_ci_hi": round(wi_hi * 100, 1),
    }


# =============================================================================
# AUFGABE 3 — GLOBALER BENCHMARK (gesamter Zeitraum, kein Fold-Split)
# =============================================================================

def run_global_benchmark(df: pd.DataFrame) -> tuple[dict, dict, pd.Series]:
    """
    Einmalige Optimierung über den GESAMTEN Zeitraum (alle IS+OOS-Jahre),
    dann Backtest auf diesem gesamten Zeitraum mit den global besten Params.
    Klassischer In-Sample-Backtest ohne Out-of-Sample-Trennung.
    Das ist der geschönte Vergleich, der die Überanpassung sichtbar macht.
    """
    global_start = min(is_start for is_start, _ in FOLDS)
    global_end   = max(oos_year  for _,      oos_year in FOLDS)

    print(f"\n  Globaler Benchmark: Optimierung über {global_start}–{global_end} ...")

    best_params  = None
    best_metrics = None
    best_score   = -np.inf

    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))
    print(f"    Grid-Search: {len(combos)} Kombinationen ...")

    for combo in combos:
        params = dict(zip(keys, combo))
        osc    = compute_oscillator(df, params["cycleLength"], params["softness"], SMOOTH_LENGTH)
        trades, eq_ts = run_backtest(df, osc, params,
                                     start_year=global_start, end_year=global_end,
                                     initial_capital=INITIAL_CAPITAL)
        m = compute_metrics(trades, eq_ts, INITIAL_CAPITAL)

        if m["num_trades"] < MIN_TRADES_IS:
            continue
        if m["max_dd_pct"] > MAX_DD_IS:
            continue

        score = m[OPTIMIZATION_METRIC]
        if score > best_score or (score == best_score and
                                   (best_metrics is None or m["max_dd_pct"] < best_metrics["max_dd_pct"])):
            best_score   = score
            best_params  = params.copy()
            best_metrics = m.copy()

    if best_params is None:
        print("    WARNUNG: Kein gültiges globales Parameter-Set gefunden!")
        return None, None, pd.Series(dtype=float)

    print(f"    Globale beste Params: {best_params}")
    print(f"    IS-Gesamt-Kennzahlen: PF={best_metrics['profit_factor']}  "
          f"Trades={best_metrics['num_trades']}  MaxDD={best_metrics['max_dd_pct']}%")

    # Backtest auf dem GESAMTEN Zeitraum mit diesen Params
    osc = compute_oscillator(df, best_params["cycleLength"], best_params["softness"], SMOOTH_LENGTH)
    _, full_eq = run_backtest(df, osc, best_params,
                               start_year=global_start, end_year=global_end,
                               initial_capital=INITIAL_CAPITAL)

    mask = (full_eq.index.year >= global_start) & (full_eq.index.year <= global_end)
    full_eq = full_eq[mask]

    return best_params, best_metrics, full_eq


# =============================================================================
# OUTPUT: TABELLEN & CHART
# =============================================================================

def print_summary_table(robust_results: list[dict], naive_results: list[dict]) -> None:
    print("\n" + "="*130)
    print(f"{'ZUSAMMENFASSUNG WALK-FORWARD-ANALYSE (ROBUST vs. NAIV)':^130}")
    print("="*130)
    header = (f"{'Fold':>5} | {'IS-Zeitraum':>12} | {'OOS-Jahr':>8} | "
              f"{'IS PF':>6} | {'IS Trd':>6} | {'IS DD%':>6} || "
              f"{'R-OOS PF':>8} | {'R-OOS Trd':>9} | {'R-OOS DD%':>9} | "
              f"{'N-OOS PF':>8} | {'N-OOS Trd':>9} | {'N-OOS DD%':>9}")
    print(header)
    print("-"*130)

    for r, n in zip(robust_results, naive_results):
        im = r["is_metrics"]
        rm = r["oos_metrics"]
        nm = n["oos_metrics"]
        print(f"{r['fold']:>5} | {r['is_period']:>12} | {r['oos_year']:>8} | "
              f"{im['profit_factor']:>6.2f} | {im['num_trades']:>6} | {im['max_dd_pct']:>5.1f}% || "
              f"{rm['profit_factor']:>8.2f} | {rm['num_trades']:>9} | {rm['max_dd_pct']:>8.1f}% | "
              f"{nm['profit_factor']:>8.2f} | {nm['num_trades']:>9} | {nm['max_dd_pct']:>8.1f}%")

    print("="*130)


def print_pool_stats(label: str, stats: dict) -> None:
    print(f"\n  ── Gepoolte OOS-Statistiken ({label}) ──────────────────────────────")
    print(f"    Trades gesamt:   {stats.get('n_trades', 0)}")
    print(f"    Profit Factor:   {stats.get('profit_factor', 0)}")
    print(f"    Win Rate:        {stats.get('winrate_pct', 0)}%  "
          f"(Wilson 95% CI: {stats.get('wilson_ci_lo', 0)}%–{stats.get('wilson_ci_hi', 0)}%)")
    print(f"    Avg R:           {stats.get('avg_r', 0)}")
    print(f"    Max Drawdown:    {stats.get('max_dd_pct', 0)}%")
    print(f"    Wilcoxon-Test:   stat={stats.get('wilcoxon_stat', 'n/a')}  "
          f"p-Wert={stats.get('wilcoxon_p', 'n/a')}")
    wp = stats.get("wilcoxon_p", 1.0)
    if isinstance(wp, float):
        sig = "signifikant (p<0.05)" if wp < 0.05 else "NICHT signifikant (p≥0.05)"
        print(f"    → Edge ist {sig}")


def plot_three_curves(robust_equity:  pd.Series,
                      naive_equity:   pd.Series,
                      global_equity:  pd.Series,
                      fold_transitions: list[pd.Timestamp]) -> None:
    """
    Drei Equity-Kurven in einem Chart:
      1. Robust WFA OOS (Aufgabe 1+2) — steelblue
      2. Naive WFA OOS (bisheriges Verhalten) — darkorange
      3. Globaler IS-Backtest (Aufgabe 3, geschönt) — tomato gestrichelt
    Vertikale Linien an Fold-Übergängen.
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    if not robust_equity.empty:
        ax.plot(robust_equity.index, robust_equity.values,
                label="WFA OOS — Robuste Auswahl (Plateau)", color="steelblue", linewidth=2.2)

    if not naive_equity.empty:
        ax.plot(naive_equity.index, naive_equity.values,
                label="WFA OOS — Naive Auswahl (Einzelspitze)", color="darkorange",
                linewidth=1.8, linestyle=(0, (5, 2)))

    if not global_equity.empty:
        ax.plot(global_equity.index, global_equity.values,
                label="Globaler IS-Backtest (geschönt, kein OOS-Split)", color="tomato",
                linewidth=1.5, linestyle="--", alpha=0.85)

    # Fold-Übergänge markieren
    for i, ts in enumerate(fold_transitions):
        ax.axvline(ts, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.text(ts, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else INITIAL_CAPITAL * 0.97,
                f" F{i+1}→F{i+2}", fontsize=7, color="gray", rotation=90, va="bottom")

    ax.axhline(INITIAL_CAPITAL, color="black", linestyle=":", linewidth=1, label="Startkapital")

    ax.set_title(f"TACO WFA — Robuste vs. Naive vs. Globaler Benchmark | "
                 f"Asset: {MAIN_ASSET} | Basket: DXY={'✓' if USE_DXY else '✗'} "
                 f"Gold={'✓' if USE_GOLD else '✗'} Bonds={'✓' if USE_BONDS else '✗'}",
                 fontsize=11)
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Datum")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("taco_wfa_equity.png", dpi=150, bbox_inches="tight")
    print("\n  Chart gespeichert: taco_wfa_equity.png")
    plt.show()


def print_fazit(robust_results: list[dict],
                naive_results:  list[dict],
                robust_stats:   dict,
                naive_stats:    dict,
                robust_equity:  pd.Series,
                global_equity:  pd.Series) -> None:
    print("\n" + "="*80)
    print("FAZIT & ABSCHLUSSBERICHT")
    print("="*80)

    print(f"\n  (a) Optimierungsmetrik:  {OPTIMIZATION_METRIC}")

    # Vergleich robust vs. naiv OOS
    r_pfs = [r["oos_metrics"]["profit_factor"] for r in robust_results]
    n_pfs = [r["oos_metrics"]["profit_factor"] for r in naive_results]
    if r_pfs and n_pfs:
        print(f"\n  (b) Robuste vs. Naive Auswahl (OOS Profit Factors pro Fold):")
        for i, (rp, np_) in enumerate(zip(r_pfs, n_pfs)):
            diff = rp - np_
            print(f"      Fold {i+1}: robust={rp:.3f}  naiv={np_:.3f}  Δ={diff:+.3f}")
        avg_r_pf = np.mean(r_pfs)
        avg_n_pf = np.mean(n_pfs)
        print(f"      Ø Robust: {avg_r_pf:.3f}  |  Ø Naiv: {avg_n_pf:.3f}  |  "
              f"Δ Ø: {avg_r_pf - avg_n_pf:+.3f}")

    # Wilcoxon
    wp_r = robust_stats.get("wilcoxon_p", "n/a")
    wp_n = naive_stats.get("wilcoxon_p",  "n/a")
    print(f"\n  (c) Wilcoxon-Test (gepoolte OOS-Trades gegen H0: Median PnL=0, einseitig):")
    print(f"      Robust: p={wp_r}  |  Naiv: p={wp_n}")
    for label, wp in [("Robust", wp_r), ("Naiv", wp_n)]:
        if isinstance(wp, float):
            sig = "signifikant (p<0.05)" if wp < 0.05 else "NICHT signifikant (p≥0.05)"
            print(f"      → {label}-Edge ist {sig}")

    # Globaler Benchmark vs. WFA
    if not robust_equity.empty and not global_equity.empty:
        wfa_ret   = (robust_equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        glob_ret  = (global_equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        print(f"\n  WFA OOS Gesamtrendite (robust):     {wfa_ret:.1f}%")
        print(f"  Globaler IS-Backtest Gesamtrendite: {glob_ret:.1f}%")
        print(f"  Überanpassungs-Gap:                 {glob_ret - wfa_ret:+.1f}%")

    print("""
  BEKANNTE OFFENE EINSCHRÄNKUNGEN:
  ──────────────────────────────────────────────────────────────────
  1. Grid-Optimierung nur über cycleLength und stopLossPct (5×5 Ebene).
     Softness, RR-Multiple und Level werden weiterhin optimiert, aber
     die Nachbar-Glättung (Aufgabe 1) erstreckt sich bewusst nur auf die
     zwei Hauptparameter. Erweiterung auf alle 5 Dimensionen ist Phase 2
     (kombinatorische Explosion: 3^5 = 243 Nachbarn pro Kombination).
  2. Nur 5 Folds → zu wenig für statistische Aussagekraft. Empfehlung: ≥15 Folds.
  3. DXY-Daten via yfinance ab ~2004 — frühere Bars via ffill aufgefüllt.
  4. Slippage nicht modelliert (nur Commission 0.05% wie im Pine-Original).
""")


def print_trade_log(fold_results: list[dict], label: str = "") -> None:
    title = f"OOS TRADE LOG — {label}" if label else "OOS TRADE LOG"
    print("\n" + "="*120)
    print(f"{title:^120}")
    print("="*120)
    print(f"{'Fold':>5} | {'Richtg':>6} | {'Entry':>12} | {'Exit':>12} | "
          f"{'EntryPx':>10} | {'ExitPx':>9} | {'P&L':>9} | {'Grund':>15}")
    print("-"*120)
    for r in fold_results:
        for t in r["oos_trades"]:
            print(f"{r['fold']:>5} | {t.direction:>6} | {str(t.entry_date.date()):>12} | "
                  f"{str(t.exit_date.date()) if t.exit_date else 'offen':>12} | "
                  f"{t.entry_price:>10.5f} | {t.exit_price if t.exit_price else 0:>9.5f} | "
                  f"{t.pnl:>9.2f} | {t.exit_reason:>15}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  TACO Walk-Forward Analysis")
    print(f"  Asset:  {MAIN_ASSET}")
    print(f"  Basket: DXY={USE_DXY}  Gold={USE_GOLD}  Bonds={USE_BONDS}")
    print(f"  Folds:  {len(FOLDS)}  |  IS: {IS_YEARS}J  |  OOS: {OOS_YEARS}J")
    print(f"  Metrik: {OPTIMIZATION_METRIC}")
    print(f"  Grid:   {sum(len(v) for v in GRID.values())} Werte → "
          f"{len(list(product(*GRID.values())))} Kombinationen")
    print("=" * 65)

    print("\n[1/4] Lade Daten ...")
    raw_data = load_data()

    print("\n[2/4] Aligniere Daten ...")
    df = align_data(raw_data)
    print(f"  Gemeinsamer Zeitraum: {df.index[0].date()} bis {df.index[-1].date()}, "
          f"{len(df)} Handelstage")

    print("\n[3/4] Walk-Forward-Analyse (Robust + Naiv parallel) ...")
    robust_results, naive_results, robust_equity, naive_equity, fold_transitions = run_wfa_dual(df)

    print("\n[4/4] Globaler Benchmark (Aufgabe 3) ...")
    global_params, global_is_metrics, global_equity = run_global_benchmark(df)

    # ── Ausgabe ──────────────────────────────────────────────────────────────
    print_summary_table(robust_results, naive_results)

    robust_stats = compute_oos_pool_stats(robust_results, robust_equity)
    naive_stats  = compute_oos_pool_stats(naive_results,  naive_equity)

    print_pool_stats("Robust", robust_stats)
    print_pool_stats("Naiv",   naive_stats)

    print_trade_log(robust_results, label="Robuste Auswahl")

    print_fazit(robust_results, naive_results,
                robust_stats, naive_stats,
                robust_equity, global_equity)

    plot_three_curves(robust_equity, naive_equity, global_equity, fold_transitions)
