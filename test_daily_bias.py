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
    assert {k: t["effect"][k] for k in ("NQ", "ES", "GC")} == {"NQ": -1, "ES": -1, "GC": 1}
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


def test_weekly_bars_and_running():
    df = _ohlc(300)
    wk = db.to_weekly(df)
    assert (wk.index.dayofweek == 4).all() and wk["High"].iloc[-1] >= wk["Close"].iloc[-1]
    # Mittwoch 10:00 NY in der letzten Woche -> laufende Woche wird abgetrennt
    last_fri = wk.index[-1]
    done, price = db.split_running(wk, _ny(last_fri - pd.Timedelta(days=2) + pd.Timedelta(hours=10)))
    assert len(done) == len(wk) - 1 and price == wk["Close"].iloc[-1]
    st = db.structure_state(done, price, "Weekly")
    assert st is not None and "Vorwochen" in st["reason"]


def test_fx_tape_and_news():
    # Risk-off: VIX hoch, Aktien runter -> Yen bull, Aussie bear
    ch = {"VIX": 20.0, "SPX": -2.0, "US10Y": -6.0, "GOLD": 0.0}
    assert db.tape_score("6J", ch)["score"] > 0.5
    assert db.tape_score("6A", ch)["score"] < 0
    assert db.tape_score("6C", {"OIL": 4.0})["score"] > 0.5
    assert db.tape_score("DXY", {"US10Y": 8.0})["score"] > 0.5
    # Weekly-Skalen sind breiter -> gleicher Move wirkt schwaecher
    assert abs(db.tape_score("6J", ch, "Weekly")["score"]) < abs(db.tape_score("6J", ch)["score"])
    t = db.tag_headline("Missile attack escalates conflict")
    assert t["effect"]["6J"] == 1 and t["effect"]["6A"] == -1 and t["effect"]["GC"] == 1
    t2 = db.tag_headline("Fed signals rate cut in December")
    assert t2["effect"]["DXY"] == -1 and t2["effect"]["6E"] == 1


def test_horizon_end_and_weekly_events():
    now = pd.Timestamp("2026-11-18 12:00")   # Mittwoch
    end = db.horizon_end(now, "Weekly")
    assert end.weekday() == 4 and end.hour == 17
    sat = db.horizon_end(pd.Timestamp("2026-11-21 12:00"), "Weekly")
    assert sat.day == 27                       # naechster Freitag
    cal = pd.DataFrame({"event_code": ["boj-interest-rate-decision", "nonfarm-payrolls", "x"],
                        "event": ["BoJ", "NFP", "x"], "currency": ["JPY", "USD", "EUR"],
                        "importance": ["High", "High", "Low"],
                        "time_utc": pd.to_datetime(["2026-11-19 03:00", "2026-11-20 13:30", "2026-11-19 10:00"]),
                        "actual": [np.nan] * 3, "forecast": [np.nan] * 3})
    ev = db.relevant_events(cal, now, ["USD", "JPY", "EUR"], "Weekly")
    assert len(ev) == 2 and ev["notenbank"].sum() == 1


def test_seasonality_week():
    idx = pd.bdate_range("2010-01-01", "2026-11-13")
    rng = np.random.default_rng(7)
    c = 100 * np.exp(np.cumsum(rng.normal(0, .005, len(idx))))
    df = pd.DataFrame({"Open": c, "High": c * 1.002, "Low": c * .998, "Close": c}, index=idx)
    s = db.seasonality_week(df, _ny("2026-11-18 10:00"))
    assert s["week"] == 47 and s["n"] >= 10 and 0 <= s["up"] <= 1 and 2026 not in s["years"]


def test_compute_bias_groups_and_horizons():
    idx = pd.bdate_range(end="2026-11-18", periods=900)
    def ohlc(sym, years=15):
        rng = np.random.default_rng(abs(hash(sym)) % 997)
        c = 1.1 * np.exp(np.cumsum(rng.normal(0, .006, len(idx))))
        o = np.r_[c[0], c[:-1]]
        return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.002, "Low": np.minimum(o, c) * .998, "Close": c}, index=idx)
    def close(sym, period):
        rng = np.random.default_rng(len(sym))
        return pd.Series(10 * np.exp(np.cumsum(rng.normal(0, .01, len(idx)))), index=idx)
    data = {"now": pd.Timestamp("2026-11-18 14:00"), "calendar": pd.DataFrame(), "results": {},
            "plan": {"regime": {"effective": "Normal"}, "intervention": {"level": "hoch"}}}
    for group in db.GROUPS:
        for hz in db.HORIZONS:
            res = db.compute_bias(data, group, hz, fetch_ohlc=ohlc, fetch_close=close, fetch_news=False)
            assert set(res["assets"]) == set(db.GROUPS[group])
            r = next(iter(res["assets"].values()))
            assert r["label"] in ("LONG", "SHORT", "NEUTRAL") and r["history"] is not None
            if hz == "Weekly":
                assert "Vorwochen" in r["flip"] or r["flip"] == "–"


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
