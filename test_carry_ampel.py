"""Tests fuer carry_ampel.py (synthetische Daten, kein Netzwerk)."""
import numpy as np
import pandas as pd

import carry_ampel as ca


def _vix(last, prev5=None, n=100, base=15.0):
    v = pd.Series(base, index=pd.bdate_range("2026-01-01", periods=n))
    if prev5 is not None:
        v.iloc[-6] = prev5
    v.iloc[-1] = last
    return v


def _hist(values):
    return pd.DataFrame({"report_date": pd.date_range("2023-01-03", periods=len(values), freq="7D"),
                         "net_pct": values})


def test_vix_state():
    assert ca.vix_state(_vix(12)).get("state") == "Ruhig"
    assert ca.vix_state(_vix(22, prev5=21)).get("state") == "Stress"
    assert ca.vix_state(_vix(31, prev5=30)).get("state") == "Panik"
    assert ca.vix_state(_vix(19, prev5=13)).get("state") == "Panik"     # +46 % in 5 Tagen
    assert ca.vix_state(None)["state"] == "unbekannt"


def test_cot_percentile_usd_and_cross():
    rng = np.random.default_rng(0)
    jpy = _hist(list(rng.normal(0, .1, 160)) + [-0.5])          # JPY-Shorts extrem
    p = ca.cot_spread_percentile(None, jpy, "USD", "JPY")        # USDJPY: USD(0) - JPY
    assert p["percentile"] > 0.95
    aud = _hist(list(rng.normal(0, .1, 161)))
    p2 = ca.cot_spread_percentile(aud, jpy, "AUD", "JPY")
    assert 0 <= p2["percentile"] <= 1
    assert ca.cot_spread_percentile(None, None, "USD", "USD") is None


def _trend(above):
    return {"above": above, "vol_pct": 10.0}


QUIET = {"state": "Ruhig", "level": 13.0}


def test_status_green_and_direction():
    r = ca.carry_status("USDJPY", 4.0, 0.5, 0.3, _trend(True), {"percentile": 0.3}, QUIET, 30)
    assert r["ampel"] == "GRÜN" and r["direction"] == "LONG" and r["phase"] == "Aufbau"
    # Carry-Richtung SHORT (Quote hat hoeheren Zins): Trend muss unter der SMA liegen
    r2 = ca.carry_status("EURUSD", 2.0, 4.0, -0.3, _trend(False), {"percentile": 0.7}, QUIET, 30)
    assert r2["direction"] == "SHORT" and r2["ampel"] == "GRÜN"
    assert abs(r2["crowd"] - 0.3) < 1e-9


def test_status_yellow_red_and_no_carry():
    crowded = ca.carry_status("AUDJPY", 4.0, 0.5, 0.0, _trend(True), {"percentile": 0.85}, QUIET, 30)
    assert crowded["ampel"] == "GELB" and crowded["phase"] == "Überfüllt"
    red = ca.carry_status("AUDJPY", 4.0, 0.5, 0.0, _trend(False), {"percentile": 0.95}, QUIET, 30)
    assert red["ampel"] == "ROT"
    panic = ca.carry_status("AUDJPY", 4.0, 0.5, 0.0, _trend(True), {"percentile": 0.2},
                            {"state": "Panik", "level": 35.0}, 30)
    assert panic["ampel"] == "ROT"
    cb = ca.carry_status("AUDJPY", 4.0, 0.5, 0.0, _trend(True), {"percentile": 0.2}, QUIET, 3)
    assert cb["ampel"] == "GELB"
    none = ca.carry_status("EURGBP", 2.0, 2.5, 0.0, None, None, QUIET, None)
    assert none["ampel"] == "KEIN CARRY"


def test_pair_trend():
    s = pd.Series(np.linspace(100, 120, 300), index=pd.bdate_range("2025-01-01", periods=300))
    t = ca.pair_trend(s)
    assert t["above"] and t["vol_pct"] >= 0
    assert ca.pair_trend(s.iloc[:50]) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
