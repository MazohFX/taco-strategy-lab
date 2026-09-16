"""
Fib 0.618 + BOS Backtest — Kern-Engine (Python-Portierung des Pine-Scripts
FIB618_Opens_MonthlyPOC_Strategy.pine, reduziert auf den validierbaren Kern).

Bewusst NICHT enthalten (siehe Pine-Version fuer die volle Fassung):
  - Opens-Confluence (Weekly/Monthly/Quarterly)
  - Monatliches Volumen-Profil (POC/VAH/VAL) + Confluence
  - Setup-Grading A+/A/B
  - Teil-Take-Profits / Break-Even-State-Machine
Grund: erst den Grund-Edge (Boden -> impulsiver Break of Structure ->
Fib-0.618-Retracement-Entry -> fester SL/TP) isoliert pruefen, bevor die
Confluence-Filter obendrauf kommen. Python statt Pine, weil sich das hier
interaktiv viel schneller debuggen laesst und die vorhandene Pepperstone-
CSV-Infrastruktur der App direkt wiederverwendet werden kann.

Alle Ergebnisse werden in R-Multiples gerechnet (kein syminfo.pointvalue /
keine Kontowaehrungs-Umrechnung noetig): R = (Exit-Preis - Entry-Preis) /
(Entry-Preis - SL-Preis), vorzeichenrichtig je Richtung. Das ist fuer einen
Kern-Validierungslauf robuster als eine $-P&L-Rechnung und direkt mit der
"Ø R-Multiple"/"Expectancy (R)"-Logik aus dem Pine-Script vergleichbar.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fib618Config:
    direction: str = "Beide"          # "Beide" | "Nur Long" | "Nur Short"

    # Struktur / Pivots
    left_bars: int = 5
    right_bars: int = 5
    atr_len: int = 14

    # Boden/Kompression vor BOS
    use_base_filter: bool = True
    base_lookback: int = 20
    atr_short_len: int = 5
    atr_long_len: int = 50
    compression_ratio: float = 0.8
    base_range_atr: float = 3.0

    # Impulsivitaet
    impulse_mode: str = "Beide"       # "Body" | "Displacement" | "Beide"
    bos_body_atr: float = 1.0
    close_pos_pct: float = 30.0
    impulse_atr: float = 3.0
    impulse_bars: int = 15

    # Entry / Order-Management
    pip_size: float = 0.0001
    sl_buffer_pips: float = 2.0
    tp_level: float = -0.62           # -0.27 | -0.62 | -1.0 (fester Full-TP)
    cancel_at_ext27: bool = True
    max_pending_bars: int = 72
    spread_pips: float = 1.0

    # Risiko
    risk_pct: float = 0.5
    compounding: bool = True


@dataclass
class Fib618Result:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: dict = field(default_factory=dict)


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(alpha=1 / length, adjust=False, min_periods=length).mean().to_numpy()


def _find_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int):
    n = len(high)
    piv_high = np.full(n, np.nan)
    piv_low = np.full(n, np.nan)
    for i in range(left, n - right):
        wh = high[i - left:i + right + 1]
        if high[i] == wh.max() and np.argmax(wh) == left:
            piv_high[i] = high[i]
        wl = low[i - left:i + right + 1]
        if low[i] == wl.min() and np.argmin(wl) == left:
            piv_low[i] = low[i]
    return piv_high, piv_low


def run_fib618_backtest(df: pd.DataFrame, cfg: Fib618Config) -> Fib618Result:
    """df braucht einen DatetimeIndex und Spalten open/high/low/close."""
    df = df.sort_index()
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    idx = df.index

    if n < max(cfg.left_bars + cfg.right_bars + 5, cfg.base_lookback + 5, cfg.atr_long_len + 5):
        return Fib618Result(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={"error": "Zu wenig Bars fuer diese Parameter."})

    atr = _wilder_atr(h, l, c, cfg.atr_len)
    atr_short = _wilder_atr(h, l, c, cfg.atr_short_len)
    atr_long = _wilder_atr(h, l, c, cfg.atr_long_len)
    roll_high = pd.Series(h).rolling(cfg.base_lookback).max().to_numpy()
    roll_low = pd.Series(l).rolling(cfg.base_lookback).min().to_numpy()

    piv_high, piv_low = _find_pivots(h, l, cfg.left_bars, cfg.right_bars)

    sl_buffer = cfg.sl_buffer_pips * cfg.pip_size
    spread = cfg.spread_pips * cfg.pip_size

    allow_long = cfg.direction in ("Beide", "Nur Long")
    allow_short = cfg.direction in ("Beide", "Nur Short")

    # Struktur-State
    swing_high_res = np.nan
    swing_low_sup = np.nan
    leg_low_since_high = np.nan
    leg_low_idx = -1
    leg_high_since_low = np.nan
    leg_high_idx = -1
    raw_break_long_prev = False
    raw_break_short_prev = False

    # Pending-Setup-State
    p_long_active = False
    p_long_sw_low = np.nan
    p_long_sw_high = np.nan
    p_long_bos_bar = -1

    p_short_active = False
    p_short_sw_high = np.nan
    p_short_sw_low = np.nan
    p_short_bos_bar = -1

    # Offene Position
    in_position = False
    pos_dir = None
    pos_entry = np.nan
    pos_sl = np.nan
    pos_tp = np.nan
    pos_entry_time = None

    trades = []

    for i in range(n):
        # ── Pivot-Tracking ───────────────────────────────────────────────
        if not np.isnan(piv_high[i]):
            if np.isnan(swing_high_res) or piv_high[i] != swing_high_res:
                leg_low_since_high = np.nan
                leg_low_idx = -1
            swing_high_res = piv_high[i]
        if not np.isnan(piv_low[i]):
            if np.isnan(swing_low_sup) or piv_low[i] != swing_low_sup:
                leg_high_since_low = np.nan
                leg_high_idx = -1
            swing_low_sup = piv_low[i]

        if not np.isnan(swing_high_res):
            if np.isnan(leg_low_since_high) or l[i] < leg_low_since_high:
                leg_low_since_high = l[i]
                leg_low_idx = i
        if not np.isnan(swing_low_sup):
            if np.isnan(leg_high_since_low) or h[i] > leg_high_since_low:
                leg_high_since_low = h[i]
                leg_high_idx = i

        # ── Boden/Kompressions-Filter ────────────────────────────────────
        base_ok = True
        if cfg.use_base_filter and i >= cfg.base_lookback:
            compression_ok = (not np.isnan(atr_short[i]) and not np.isnan(atr_long[i])
                               and atr_long[i] > 0 and atr_short[i] / atr_long[i] <= cfg.compression_ratio)
            range_ok = (not np.isnan(roll_high[i]) and not np.isnan(atr[i])
                        and (roll_high[i] - roll_low[i]) <= cfg.base_range_atr * atr[i])
            base_ok = compression_ok or range_ok
        elif cfg.use_base_filter:
            base_ok = False

        # ── Impulsivitaet ────────────────────────────────────────────────
        bar_range = h[i] - l[i]
        body = abs(c[i] - o[i])
        close_pos_up = (c[i] - l[i]) / bar_range if bar_range > 0 else 0.0
        close_pos_down = (h[i] - c[i]) / bar_range if bar_range > 0 else 0.0
        atr_i = atr[i] if not np.isnan(atr[i]) else 0.0

        body_ok_long = body >= cfg.bos_body_atr * atr_i and close_pos_up >= (1 - cfg.close_pos_pct / 100)
        body_ok_short = body >= cfg.bos_body_atr * atr_i and close_pos_down >= (1 - cfg.close_pos_pct / 100)
        disp_ok_long = (not np.isnan(leg_low_since_high) and (i - leg_low_idx) <= cfg.impulse_bars
                        and (c[i] - leg_low_since_high) >= cfg.impulse_atr * atr_i)
        disp_ok_short = (not np.isnan(leg_high_since_low) and (i - leg_high_idx) <= cfg.impulse_bars
                         and (leg_high_since_low - c[i]) >= cfg.impulse_atr * atr_i)

        if cfg.impulse_mode == "Body":
            impulsive_long, impulsive_short = body_ok_long, body_ok_short
        elif cfg.impulse_mode == "Displacement":
            impulsive_long, impulsive_short = disp_ok_long, disp_ok_short
        else:
            impulsive_long = body_ok_long and disp_ok_long
            impulsive_short = body_ok_short and disp_ok_short

        # ── Break of Structure (nur beim ersten Bruch) ──────────────────
        raw_break_long = (not np.isnan(swing_high_res)) and c[i] > swing_high_res
        raw_break_short = (not np.isnan(swing_low_sup)) and c[i] < swing_low_sup

        bos_long = (raw_break_long and not raw_break_long_prev and base_ok
                    and impulsive_long and not np.isnan(leg_low_since_high))
        bos_short = (raw_break_short and not raw_break_short_prev and base_ok
                     and impulsive_short and not np.isnan(leg_high_since_low))

        raw_break_long_prev = raw_break_long
        raw_break_short_prev = raw_break_short

        no_pos = not in_position

        if bos_long and allow_long and no_pos:
            p_long_active = True
            p_long_sw_low = leg_low_since_high
            p_long_sw_high = h[i]
            p_long_bos_bar = i
        if bos_short and allow_short and no_pos:
            p_short_active = True
            p_short_sw_high = leg_high_since_low
            p_short_sw_low = l[i]
            p_short_bos_bar = i

        if bos_short and p_long_active:
            p_long_active = False
        if bos_long and p_short_active:
            p_short_active = False

        # ── Long: Pending-Setup pflegen ──────────────────────────────────
        if p_long_active and no_pos:
            p_long_sw_high = max(p_long_sw_high, h[i])
            entry_l = p_long_sw_high - 0.618 * (p_long_sw_high - p_long_sw_low)
            ext_l = p_long_sw_high + 0.27 * (p_long_sw_high - p_long_sw_low)
            sl_raw_l = p_long_sw_low - sl_buffer

            cancel_ext = cfg.cancel_at_ext27 and h[i] >= ext_l
            cancel_inval = c[i] < p_long_sw_low
            cancel_time = (i - p_long_bos_bar) > cfg.max_pending_bars

            if cancel_ext or cancel_inval or cancel_time:
                p_long_active = False
            else:
                entry_fill = entry_l + spread
                # Fill nur wenn diese Bar das Level tatsaechlich beruehrt hat
                if l[i] <= entry_fill:
                    tp_l = p_long_sw_high - cfg.tp_level * (p_long_sw_high - p_long_sw_low) - spread
                    sl_l = sl_raw_l - spread
                    if tp_l > entry_fill > sl_l:
                        in_position = True
                        pos_dir = "Long"
                        pos_entry = entry_fill
                        pos_sl = sl_l
                        pos_tp = tp_l
                        pos_entry_time = idx[i]
                        p_long_active = False

        # ── Short: Pending-Setup pflegen ─────────────────────────────────
        if p_short_active and no_pos and not in_position:
            p_short_sw_low = min(p_short_sw_low, l[i])
            entry_s = p_short_sw_low + 0.618 * (p_short_sw_high - p_short_sw_low)
            ext_s = p_short_sw_low - 0.27 * (p_short_sw_high - p_short_sw_low)
            sl_raw_s = p_short_sw_high + sl_buffer

            cancel_ext = cfg.cancel_at_ext27 and l[i] <= ext_s
            cancel_inval = c[i] > p_short_sw_high
            cancel_time = (i - p_short_bos_bar) > cfg.max_pending_bars

            if cancel_ext or cancel_inval or cancel_time:
                p_short_active = False
            else:
                entry_fill = entry_s - spread
                if h[i] >= entry_fill:
                    tp_s = p_short_sw_low + cfg.tp_level * (p_short_sw_high - p_short_sw_low) + spread
                    sl_s = sl_raw_s + spread
                    if tp_s < entry_fill < sl_s:
                        in_position = True
                        pos_dir = "Short"
                        pos_entry = entry_fill
                        pos_sl = sl_s
                        pos_tp = tp_s
                        pos_entry_time = idx[i]
                        p_short_active = False

        # ── Offene Position managen (SL/TP-Check, Prioritaet: SL zuerst) ─
        if in_position:
            hit_sl = (h[i] >= pos_sl) if pos_dir == "Short" else (l[i] <= pos_sl)
            hit_tp = (l[i] <= pos_tp) if pos_dir == "Short" else (h[i] >= pos_tp)
            exit_price = None
            exit_reason = None
            # Konservative Annahme bei Ambiguitaet in derselben Kerze: SL zuerst
            if hit_sl:
                exit_price = pos_sl
                exit_reason = "SL"
            elif hit_tp:
                exit_price = pos_tp
                exit_reason = "TP"
            if exit_price is not None:
                risk = abs(pos_entry - pos_sl)
                raw_r = (exit_price - pos_entry) / risk if pos_dir == "Long" else (pos_entry - exit_price) / risk
                trades.append({
                    "entry_time": pos_entry_time, "exit_time": idx[i], "direction": pos_dir,
                    "entry": pos_entry, "sl": pos_sl, "tp": pos_tp, "exit": exit_price,
                    "exit_reason": exit_reason, "r_multiple": raw_r,
                })
                in_position = False
                pos_dir = None

    trades_df = pd.DataFrame(trades)
    equity_curve = _build_equity_curve(trades_df, cfg.risk_pct, cfg.compounding)
    stats = _compute_stats(trades_df, equity_curve)
    return Fib618Result(trades=trades_df, equity_curve=equity_curve, stats=stats)


def _build_equity_curve(trades_df: pd.DataFrame, risk_pct: float, compounding: bool) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["exit_time", "equity"])
    equity = 100.0
    rows = []
    for _, t in trades_df.iterrows():
        if compounding:
            equity *= (1 + t["r_multiple"] * risk_pct / 100)
        else:
            equity += 100.0 * t["r_multiple"] * risk_pct / 100
        rows.append({"exit_time": t["exit_time"], "equity": equity})
    return pd.DataFrame(rows)


def _compute_stats(trades_df: pd.DataFrame, equity_curve: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"trades": 0}
    wins = trades_df[trades_df["r_multiple"] > 0]
    losses = trades_df[trades_df["r_multiple"] <= 0]
    gross_win_r = wins["r_multiple"].sum()
    gross_loss_r = -losses["r_multiple"].sum()
    pf = gross_win_r / gross_loss_r if gross_loss_r > 0 else (np.inf if gross_win_r > 0 else 0.0)

    eq = equity_curve["equity"].to_numpy() if not equity_curve.empty else np.array([100.0])
    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / running_max * 100
    max_dd = dd.max() if len(dd) else 0.0

    return {
        "trades": len(trades_df),
        "winrate_pct": len(wins) / len(trades_df) * 100,
        "profit_factor": pf,
        "avg_r": trades_df["r_multiple"].mean(),
        "expectancy_r": trades_df["r_multiple"].mean(),
        "max_dd_pct": max_dd,
        "tp_hits": int((trades_df["exit_reason"] == "TP").sum()),
        "sl_hits": int((trades_df["exit_reason"] == "SL").sum()),
        "long_trades": int((trades_df["direction"] == "Long").sum()),
        "short_trades": int((trades_df["direction"] == "Short").sum()),
    }
