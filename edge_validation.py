"""
edge_validation.py

Wiederverwendbares Modul zur statistischen Bewertung einer Trade-Liste
(Edge-Validierung) und zur darauf basierenden Positionsgrößen-Empfehlung
(Kelly-Sizing) für PropFirm-konformes Trading.

Reine Berechnungslogik — keine UI-/Streamlit-Abhängigkeiten. Wird aktuell
vom Seasonality Muster Scanner in taco_web_app.py genutzt, ist aber generisch
gehalten, damit andere Scanner (z.B. Cycle Scanner) ihn ohne Anpassung
mitnutzen können.

Erwartetes Trade-DataFrame-Format:
    - Spalte "date" (oder "Jahr"/Timestamp, vom Aufrufer normalisiert)
    - Spalte "pnl" (Netto-PnL nach Kosten, in % oder Kontowährung — konsistent
      gewählt vom Aufrufer) und/oder "r_multiple"
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ─────────────────────────────────────────────────────────────────────────
# Funktion 1: Rolling Walk-Forward
# ─────────────────────────────────────────────────────────────────────────

def rolling_walk_forward(
    trades_df: pd.DataFrame,
    window_in: int,
    window_out: int,
    optimize_fn: Optional[Callable[[pd.DataFrame], dict]] = None,
    backtest_fn: Optional[Callable[[pd.DataFrame, dict], pd.DataFrame]] = None,
    date_col: str = "date",
) -> tuple[pd.DataFrame, list[dict]]:
    """Rollierendes Walk-Forward-Fenster: In-Sample optimieren (optional) ->
    Out-of-Sample blind testen -> nur OOS-Trades verketten.

    trades_df muss nach `date_col` aufsteigend sortierbar sein und mindestens
    eine PnL-Spalte ("pnl" oder "r_multiple") enthalten.

    window_in / window_out sind Fenstergrößen in Anzahl Trades (nicht Tagen).

    optimize_fn(in_sample_df) -> dict mit Parametern. Wenn None (z.B. bei
    Seasonality-Mustern ohne klassisches Parameter-Tuning), wird kein Tuning
    durchgeführt — stattdessen wird das Muster pro Fenster nur auf Konsistenz
    geprüft (Vorzeichen des mittleren PnL im In-Sample-Fenster wird als
    "Parameter" durchgereicht).

    backtest_fn(out_sample_df, params) -> DataFrame der OOS-Trades für dieses
    Fenster (inkl. "pnl"-Spalte). Wenn None, wird das out_sample_df
    unverändert als OOS-Ergebnis übernommen (kein echtes Backtesting nötig,
    z.B. wenn trades_df bereits realisierte Trades enthält).

    Rückgabe:
        (oos_equity_df, fold_params)
        oos_equity_df: verkettete OOS-Trades mit zusätzlicher "equity"-Spalte
                       (kumulierte Summe von "pnl")
        fold_params:   Liste der pro Fold genutzten Parameter-Dicts, inkl.
                       Fenstergrenzen (für späteren Plateau-Check; Plateau-
                       Logik selbst ist NICHT Teil dieser Funktion)
    """
    if trades_df.empty:
        return pd.DataFrame(), []

    df = trades_df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    step = window_out
    fold_params: list[dict] = []
    oos_chunks: list[pd.DataFrame] = []

    start = 0
    while start + window_in + window_out <= n:
        in_sample = df.iloc[start : start + window_in]
        out_sample = df.iloc[start + window_in : start + window_in + window_out]

        if optimize_fn is not None:
            params = optimize_fn(in_sample)
        else:
            pnl_col = "pnl" if "pnl" in in_sample.columns else "r_multiple"
            params = {"consistent_sign": float(np.sign(in_sample[pnl_col].mean()))}

        if backtest_fn is not None:
            oos_result = backtest_fn(out_sample, params)
        else:
            oos_result = out_sample.copy()

        fold_params.append({
            "fold_start_idx": start,
            "in_sample_start": in_sample[date_col].iloc[0] if len(in_sample) else None,
            "in_sample_end": in_sample[date_col].iloc[-1] if len(in_sample) else None,
            "out_sample_start": out_sample[date_col].iloc[0] if len(out_sample) else None,
            "out_sample_end": out_sample[date_col].iloc[-1] if len(out_sample) else None,
            "params": params,
        })
        oos_chunks.append(oos_result)
        start += step

    if not oos_chunks:
        return pd.DataFrame(), fold_params

    oos_df = pd.concat(oos_chunks, ignore_index=True)
    pnl_col = "pnl" if "pnl" in oos_df.columns else "r_multiple"
    oos_df["equity"] = oos_df[pnl_col].cumsum()
    return oos_df, fold_params


# ─────────────────────────────────────────────────────────────────────────
# Funktion 2: Edge-Bewertung
# ─────────────────────────────────────────────────────────────────────────

def _wilcoxon_p(pnl: list[float]) -> float:
    """Einseitiger Wilcoxon Signed-Rank Test (Edge > 0). Reuse der Logik aus
    taco_web_app._wf_pool_stats."""
    if len(pnl) >= 8:
        try:
            _, p = _scipy_stats.wilcoxon(pnl, alternative="greater")
            return float(p)
        except Exception:
            return float("nan")
    return float("nan")


def _wilson_ci(n_wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson-Score-Konfidenzintervall für eine Win-Rate. Reuse der Logik
    aus taco_web_app._wf_pool_stats."""
    if n == 0:
        return 0.0, 1.0
    z = _scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return center - margin, center + margin


def _sharpe_oos(pnl: pd.Series) -> float:
    std = pnl.std()
    if not std or std < 1e-10 or pd.isna(std):
        return float("nan")
    return float(pnl.mean() / std * math.sqrt(len(pnl)))


def evaluate_edge(
    trades_df: pd.DataFrame,
    min_trades: int = 200,
    alpha: float = 0.05,
    min_sharpe_oos: float = 1.0,
    pnl_col: str = "pnl",
) -> dict:
    """Bewertet eine Trade-Liste statistisch und liefert eine handelbar/
    grenzwertig/nicht-handelbar Einstufung.

    Status-Schwellen (fest definiert):
        p-Value:    handelbar <= 0.05, grenzwertig 0.05-0.10, nicht handelbar > 0.10
        Sharpe OOS: handelbar >= 1.0,  grenzwertig 0.7-1.0,  nicht handelbar < 0.7
        "handelbar":      beide Kriterien im handelbar-Bereich UND min_trades erfüllt
        "grenzwertig":    genau ein Kriterium im grenzwertig-Bereich, alle anderen ok
        "nicht handelbar": mehr als ein Kriterium verfehlt, ODER n_trades < min_trades,
                            ODER irgendein Kriterium im "nicht handelbar"-Bereich
    """
    reasons: list[str] = []

    if trades_df.empty or pnl_col not in trades_df.columns:
        return {
            "n_trades": 0,
            "p_value": float("nan"),
            "sharpe_oos": float("nan"),
            "wilson_ci": (0.0, 1.0),
            "passes_min_trades": False,
            "passes_significance": False,
            "passes_sharpe": False,
            "status": "nicht handelbar",
            "reasons": ["Keine Trades vorhanden."],
        }

    pnl = trades_df[pnl_col].astype(float)
    n = len(pnl)
    n_wins = int((pnl > 0).sum())

    p_value = _wilcoxon_p(pnl.tolist())
    sharpe = _sharpe_oos(pnl)
    wilson_ci = _wilson_ci(n_wins, n)

    passes_min_trades = n >= min_trades
    if not passes_min_trades:
        reasons.append(f"Nur {n} Trades (Minimum {min_trades}).")

    # p-Value Einstufung
    if math.isnan(p_value):
        p_level = "nicht handelbar"
        reasons.append("p-Value nicht berechenbar (zu wenige Trades für Wilcoxon-Test).")
    elif p_value <= alpha:
        p_level = "handelbar"
    elif p_value <= 0.10:
        p_level = "grenzwertig"
        reasons.append(f"p-Value {p_value:.4f} im Grenzbereich (0.05-0.10).")
    else:
        p_level = "nicht handelbar"
        reasons.append(f"p-Value {p_value:.4f} > 0.10 — kein signifikanter Edge.")

    # Sharpe Einstufung
    if math.isnan(sharpe):
        sharpe_level = "nicht handelbar"
        reasons.append("Sharpe OOS nicht berechenbar (Standardabweichung = 0).")
    elif sharpe >= min_sharpe_oos:
        sharpe_level = "handelbar"
    elif sharpe >= 0.7:
        sharpe_level = "grenzwertig"
        reasons.append(f"Sharpe OOS {sharpe:.2f} im Grenzbereich (0.7-1.0).")
    else:
        sharpe_level = "nicht handelbar"
        reasons.append(f"Sharpe OOS {sharpe:.2f} < 0.7.")

    passes_significance = p_level == "handelbar"
    passes_sharpe = sharpe_level == "handelbar"

    levels = [p_level, sharpe_level]
    n_not_handelbar = levels.count("nicht handelbar")
    n_grenzwertig = levels.count("grenzwertig")

    if not passes_min_trades or n_not_handelbar >= 1:
        status = "nicht handelbar"
    elif n_grenzwertig >= 1:
        status = "grenzwertig"
    else:
        status = "handelbar"

    return {
        "n_trades": n,
        "p_value": p_value,
        "sharpe_oos": sharpe,
        "wilson_ci": wilson_ci,
        "passes_min_trades": passes_min_trades,
        "passes_significance": passes_significance,
        "passes_sharpe": passes_sharpe,
        "status": status,
        "reasons": reasons,
    }


# ─────────────────────────────────────────────────────────────────────────
# Funktion 3: Kelly Position Sizing
# ─────────────────────────────────────────────────────────────────────────

def kelly_position_size(
    win_rate: float,
    reward_risk_ratio: float,
    account_balance: float,
    kelly_fraction: float = 0.25,
    max_risk_pct: float = 0.01,
    n_trades_used: Optional[int] = None,
    min_trades: int = 200,
) -> dict:
    """Berechnet eine Quarter-Kelly-Positionsgröße, hart gecappt auf
    max_risk_pct (PropFirm-konform, Default 1%).

    win_rate: 0..1
    reward_risk_ratio: b in der Kelly-Formel (Gewinn/Verlust-Verhältnis)
    n_trades_used / min_trades: wenn die Stichprobe kleiner als min_trades
        ist, wird eine Warnung statt stiller Berechnung ausgegeben.
    """
    if reward_risk_ratio <= 0:
        raise ValueError("reward_risk_ratio muss > 0 sein.")

    p = win_rate
    b = reward_risk_ratio
    f_kelly_raw = (p * b - (1 - p)) / b

    f_kelly_adjusted_unclamped = f_kelly_raw * kelly_fraction
    f_kelly_adjusted = max(0.0, f_kelly_adjusted_unclamped)

    capped = f_kelly_adjusted > max_risk_pct
    risk_pct_used = min(f_kelly_adjusted, max_risk_pct)

    result = {
        "f_kelly_raw": f_kelly_raw,
        "f_kelly_adjusted": f_kelly_adjusted,
        "risk_pct_used": risk_pct_used,
        "capped": capped,
        "risk_amount": risk_pct_used * account_balance,
        "warning": None,
    }

    if n_trades_used is not None and n_trades_used < min_trades:
        result["warning"] = (
            f"Win-Rate/Reward-Risk basieren auf nur {n_trades_used} Trades "
            f"(Minimum {min_trades}) — Sizing-Empfehlung ist statistisch unsicher."
        )

    return result
