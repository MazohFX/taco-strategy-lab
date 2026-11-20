"""Tests fuer daily_bias.py (synthetische Daten, kein Netzwerk)."""
import numpy as np
import pandas as pd

import daily_bias as db


def _ohlc(n=400, drift=0.0, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = start * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.005, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


def _ny(ts):
    return pd.Timestamp(ts, tz="America/New_York")


def test_split_running():
    df = _ohlc(30)
    last = df.index[-1]
    done, price = db.split_running(df, _ny(last + pd.Timedelta(hours=10)))   # 10:00 NY am selben Tag
    assert len(done) == 29 and price == df["Close"].iloc[-1]
    done2, _ = db.split_running(df, _ny(last + pd.Timedelta(hours=17, minutes=30)))  # nach Settlement
    assert len(done2) == 30


def test_structure_uptrend_positive():
    df = _ohlc(200, drift=0.004)
    done = df.iloc[:-1]
    price = float(done["High"].iloc[-1]) * 1.01   # ueber Vortageshoch
    st = db.structure_state(done, price)
    assert st["pos"] == 1.0 and st["score"] > 0.5
    st2 = db.structure_state(done, float(done["Low"].iloc[-1]) * 0.98)
    assert st2["pos"] == -1.0 and st2["score"] < st["score"]
    assert db.structure_state(done.iloc[:10], price) is None


def test_tape_signs():
    risk_off = {"VIX": 15.0, "US10Y": 8.0, "DXY": 0.6, "OIL": 0.0, "BTC": -4.0}
    assert db.tape_score("NQ", risk_off)["score"] < -0.5
    assert db.tape_score("GC", {"VIX": 10.0, "US10Y": -8.0, "DXY": -0.6})["score"] > 0.5
    assert db.tape_score("ES", {}) is None


def test_driver_change_units():
    s = pd.Series([4.00, 4.05])
    assert abs(db.driver_change(s, "bp") - 5.0) < 1e-9
    assert abs(db.driver_change(pd.Series([100, 101]), "pct") - 1.0) < 1e-9
    assert db.driver_change(pd.Series([1.0]), "pct") is None


def test_headline_tags():
    t = db.tag_headline("Russian forces escalate attacks in Donetsk")
    assert t["effect"] == {"NQ": -1, "ES": -1, "GC": 1}
    t2 = db.tag_headline("Ceasefire agreed after weeks of attacks")
    assert t2["effect"]["GC"] == -1 and "Entspannung" in t2["labels"]
    assert db.tag_headline("Apple unveils new iPhone colours")["labels"] == []
    ns = db.news_scores([{"title": "US equity funds post biggest weekly outflow"},
                         {"title": "Iran calls for pushback against U.S. sanctions"}])
    assert ns["scores"]["NQ"]["score"] < 0 and ns["scores"]["GC"]["score"] > 0


def test_combine_labels_and_confidence():
    strong = {k: {"score": -0.7, "reason": k} for k in ("makro", "struktur", "tape", "news")}
    r = db.combine(strong)
    assert r["label"] == "SHORT" and r["score10"] == -7 and r["confidence"] == "HOCH"
    mixed = {"makro": {"score": 0.8, "reason": "m"}, "struktur": {"score": -0.6, "reason": "s"},
             "tape": {"score": -0.5, "reason": "t"}, "news": None}
    r2 = db.combine(mixed)
    assert r2["quality"] == "3/4" and r2["label"] in ("NEUTRAL", "SHORT", "LONG")
    assert r2["confidence"] != "HOCH"
    assert db.combine({"makro": None})["label"] == "n/a"


def test_flip_levels():
    st = {"pdh": 110.0, "pdl": 100.0, "price": 112.0}
    assert "Vortageshoch" in db.flip_level("LONG", st, 1) and "110" in db.flip_level("LONG", st, 1)
    st["price"] = 105.0
    assert "Vortagestief" in db.flip_level("LONG", st, 1)
    assert "Long über" in db.flip_level("NEUTRAL", st, 1)


def test_events_and_reactions():
    now = pd.Timestamp("2026-11-20 12:00")          # UTC = 07:00 NY
    cal = pd.DataFrame({
        "event_code": ["nonfarm-payrolls", "nonfarm-payrolls", "nonfarm-payrolls", "ecb-x"],
        "event": ["NFP"] * 3 + ["EZB"],
        "currency": ["USD", "USD", "USD", "EUR"],
        "importance": ["High"] * 4,
        "time_utc": pd.to_datetime(["2026-11-20 13:30", "2026-10-02 12:30", "2026-09-04 12:30", "2026-11-20 13:00"]),
        "actual": [np.nan, 150.0, 120.0, np.nan], "forecast": [140.0, 130.0, 110.0, np.nan],
    })
    ev = db.todays_us_events(cal, now)
    assert len(ev) == 1 and ev["event"].iloc[0] == "NFP"
    df = _ohlc(500)
    df.index = pd.bdate_range("2025-01-01", periods=500)
    react = db.event_reactions(cal, "nonfarm-payrolls", {"NQ": df}, ev["time_utc"].iloc[0], 8)
    assert len(react) == 2 and "%" in str(react["NQ"].iloc[0])


def test_history_stats_runs():
    df = _ohlc(600, drift=0.001, seed=3)
    drivers = {"VIX": pd.Series(np.random.default_rng(1).normal(18, 2, 600), index=df.index),
               "US10Y": pd.Series(4 + np.cumsum(np.random.default_rng(2).normal(0, .03, 600)), index=df.index)}
    st = db.structure_state(df.iloc[:-1], float(df["Close"].iloc[-1]))
    h = db.history_stats("NQ", df.iloc[:-1], drivers, st)
    assert 0 < h["base_up"] < 1 and "odds" in h and "backtest" in h
    assert 0 <= h["backtest"]["hit"] <= 1 and h["backtest"]["p"] is not None


def test_asset_bias_end_to_end():
    df = _ohlc(300, drift=0.003, seed=5)
    now_ny = _ny(df.index[-1] + pd.Timedelta(hours=9))
    r = db.asset_bias("GC", df, now_ny, 0.5, "Makro bullish", {"VIX": 5.0, "US10Y": -6.0, "DXY": -0.5},
                      {"score": 0.6, "reason": "Geopolitik"}, pd.DataFrame(), "Normal")
    assert r["label"] == "LONG" and r["score10"] >= 3 and "Kippt" not in r["flip"]
    fomc = pd.DataFrame({"event_code": ["fed-interest-rate-decision"]})
    r2 = db.asset_bias("GC", df, now_ny, 0.5, "m", {}, None, fomc, "Normal")
    assert r2["confidence"] == "NIEDRIG" and r2["notes"]


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
