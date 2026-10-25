"""Tests fuer macro_dashboard.py (synthetische Daten, kein Netzwerk)."""
import numpy as np
import pandas as pd

import macro_dashboard as md

NOW = pd.Timestamp("2026-10-20 12:00")


def _calendar():
    rows = []
    vid = 0

    def add(cur, t, actual, code=None, importance="High", event="Event", forecast=np.nan):
        nonlocal vid
        vid += 1
        rows.append({"value_id": vid, "event_id": hash(code or event) % 10**6,
                     "event_code": code or event.lower(), "currency": cur, "event": event,
                     "importance": importance, "time_utc": pd.Timestamp(t), "actual": actual,
                     "forecast": forecast, "previous": np.nan, "impact": ""})

    # GBP: 3.50 -> 4.00 (hawkish), JPY: 1.00 -> 1.00
    add("GBP", "2026-03-01", 3.50, md.RATE_DECISION_CODES["GBP"])
    add("GBP", "2026-09-17", 4.00, md.RATE_DECISION_CODES["GBP"])
    add("JPY", "2026-03-01", 1.00, md.RATE_DECISION_CODES["JPY"])
    add("JPY", "2026-09-18", 1.00, md.RATE_DECISION_CODES["JPY"])
    add("JPY", "2026-10-24", np.nan, md.RATE_DECISION_CODES["JPY"], event="BoJ Zinsentscheid")
    add("GBP", "2026-10-22", np.nan, event="UK CPI")
    return pd.DataFrame(rows)


def test_split_pair():
    assert md.split_pair("GBPJPY") == ("GBP", "JPY")


def test_rates_hawkish_base_positive():
    p = md.score_rates("GBPJPY", _calendar(), NOW)
    assert p.available and p.score > 0.3
    # Spiegelpaar hat umgekehrtes Vorzeichen
    q = md.score_rates("JPYGBP", _calendar(), NOW)
    assert np.isclose(p.score, -q.score)


def test_rates_expected_change_used():
    p0 = md.score_rates("GBPJPY", _calendar(), NOW)
    p1 = md.score_rates("GBPJPY", _calendar(), NOW, lambda c: 0.25 if c == "JPY" else None)
    assert p1.score < p0.score


def test_rates_missing_currency():
    p = md.score_rates("AUDJPY", _calendar(), NOW)
    assert not p.available and p.score is None and "AUD" in p.reason


def test_risk_sign_flips_with_jpy_side():
    idx = pd.date_range("2025-10-01", periods=260, freq="B")
    vix = pd.Series(np.linspace(30, 12, 260), index=idx)        # fallender VIX = Risk-on
    spx = pd.Series(np.linspace(100, 120, 260), index=idx)
    app = md.risk_appetite(vix, spx, spx)
    assert app["score"] > 0.3
    assert md.score_risk("GBPJPY", app).score > 0
    assert md.score_risk("USDJPY", app).score > 0
    assert md.score_risk("JPYGBP", app).score < 0
    assert not md.score_risk("EURUSD", app).available


def test_surprise_diff():
    vals = {"GBP": (0.8, 10), "JPY": (-0.4, 8)}
    p = md.score_surprise("GBPJPY", lambda c: vals[c])
    assert p.score > 0.9
    p2 = md.score_surprise("GBPJPY", lambda c: (None, 0) if c == "JPY" else vals[c])
    assert not p2.available


def _cot_hist(trend, end_level):
    dates = pd.date_range("2023-10-01", periods=160, freq="W-TUE")
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.02, len(dates)).cumsum() * 0.2
    s = base + np.linspace(0, trend, len(dates))
    s = s - s[-1] + end_level
    return pd.DataFrame({"report_date": dates, "net_pct": s})


def test_cot_extreme_dampens():
    hist_up = _cot_hist(0.0, 0.0)
    hist_up.loc[hist_up.index[-4:], "net_pct"] = [0.1, 0.2, 0.3, 0.5]   # starker Aufbau, Extrem
    stats = {"GBP": md.cot_currency_stats(hist_up), "JPY": md.cot_currency_stats(_cot_hist(0.0, 0.0))}
    assert stats["GBP"]["percentile"] >= 0.9
    p = md.score_cot("GBPJPY", lambda c: stats[c])
    raw = (stats["GBP"]["score"] - stats["JPY"]["score"]) / 2
    assert p.score > 0 and np.isclose(p.score, raw * md.CONFIG["cot_extreme_damping"])
    assert p.extra["flags"]


def test_intervention_levels():
    idx = pd.date_range("2026-08-01", periods=30, freq="B")
    calm = pd.Series(np.full(30, 145.0), index=idx)
    assert md.intervention_level(calm)["level"] == "niedrig"
    spike = pd.Series(np.linspace(150, 161, 30), index=idx)
    assert md.intervention_level(spike)["level"] == "hoch"


def test_event_central_bank_within_week_is_high():
    p = md.score_event("GBPJPY", _calendar(), NOW)
    assert p.extra["level"] == "hoch"
    assert p.score is None


def _pil(key, s):
    return md.Pillar(key, s, "", s is not None)


def test_overall_long_short_mixed():
    pills = {"zinsen": _pil("zinsen", 0.6), "risiko": _pil("risiko", 0.4),
             "surprise": _pil("surprise", 0.1), "cot": _pil("cot", 0.0)}
    assert md.overall(pills, "GBPJPY")["label"] == "LONG"
    pills["cot"] = _pil("cot", -0.8)
    assert md.overall(pills, "GBPJPY")["label"] == "GEMISCHT"
    pills = {k: _pil(k, -0.6) for k in md.SCORED_PILLARS}
    assert md.overall(pills, "GBPJPY")["label"] == "SHORT"


def test_overall_skips_risk_for_non_haven_pair_and_missing():
    pills = {"zinsen": _pil("zinsen", 0.5), "risiko": _pil("risiko", -1.0),
             "surprise": _pil("surprise", None), "cot": _pil("cot", 0.3)}
    v = md.overall(pills, "EURUSD")
    assert v["quality"] == "2/3" and v["label"] == "LONG"


def test_overall_none():
    pills = {k: _pil(k, None) for k in md.SCORED_PILLARS}
    assert md.overall(pills, "GBPJPY")["label"] == "n/a"


def test_futures_signs():
    cal = _calendar()
    # Fed fehlt im Test-Kalender -> n/a
    assert not md.score_rates_future("Gold", cal, NOW).available
    idx = pd.date_range("2025-10-01", periods=260, freq="B")
    vix = pd.Series(np.linspace(30, 12, 260), index=idx)
    app = md.risk_appetite(vix, None, None)
    assert md.score_risk_future("S&P 500", app, app).score > 0
    assert md.score_risk_future("Gold", app, app).score < 0
    assert not md.score_risk_future("DXY", app, app).available
    s = lambda c: (0.8, 10)
    assert md.score_surprise_future("DXY", s).score > 0
    assert md.score_surprise_future("Gold", s).score < 0
    assert md.relevant_pillars("DXY") == ["zinsen", "surprise", "cot"]


def test_futures_rates_relative_to_others():
    rows = []
    for i, (cur, a, b) in enumerate([("USD", 3.5, 4.0), ("EUR", 2.0, 2.0), ("JPY", 1.0, 1.0)]):
        for t, v in (("2026-03-01", a), ("2026-09-17", b)):
            rows.append({"event_code": md.RATE_DECISION_CODES[cur], "currency": cur, "event": "x",
                         "importance": "High", "time_utc": pd.Timestamp(t), "actual": v})
    cal = pd.DataFrame(rows)
    assert md.score_rates_future("DXY", cal, NOW).score > 0.3
    assert md.score_rates_future("Gold", cal, NOW).score < -0.3
