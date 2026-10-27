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
    assert not md.score_risk("AUDNZD", app).available      # gleiche Empfindlichkeit
    assert not md.score_risk("EURGBP", app).available
    assert md.score_risk("AUDUSD", app).score > 0              # Risk-on hilft AUD gegen USD
    assert md.score_risk("AUDJPY", app).score > md.score_risk("GBPJPY", app).score
    assert md.score_risk("CHFUSD", app, force=True).available  # Futures immer


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
    v = md.overall(pills, "EURGBP")
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
    assert md.score_risk_future("DXY", app, app).score < 0    # Risk-on schwaecht den Dollar
    s = lambda c: (0.8, 10)
    assert md.score_surprise_future("DXY", s).score > 0
    assert md.score_surprise_future("Gold", s).score < 0
    assert md.relevant_pillars("DXY") == ["zinsen", "risiko", "surprise", "cot"]
    assert "risiko" not in md.relevant_pillars("AUDNZD")
    assert "risiko" in md.relevant_pillars("CHFUSD", force_risk=True)


def test_futures_rates_relative_to_others():
    rows = []
    for i, (cur, a, b) in enumerate([("USD", 3.5, 4.0), ("EUR", 2.0, 2.0), ("JPY", 1.0, 1.0)]):
        for t, v in (("2026-03-01", a), ("2026-09-17", b)):
            rows.append({"event_code": md.RATE_DECISION_CODES[cur], "currency": cur, "event": "x",
                         "importance": "High", "time_utc": pd.Timestamp(t), "actual": v})
    cal = pd.DataFrame(rows)
    assert md.score_rates_future("DXY", cal, NOW).score > 0.3
    assert md.score_rates_future("Gold", cal, NOW).score < -0.3


def test_universe():
    assert len(md.PAIRS) == 28 and len(set(md.PAIRS)) == 28
    for p in ("AUDNZD", "CADCHF", "GBPJPY", "EURUSD", "USDJPY", "CHFJPY"):
        assert p in md.PAIRS
    assert set(md.FUTURE_LIST) == {"CHF", "EUR", "DXY", "AUD", "NZD", "CAD", "GBP", "JPY"}


def test_vix_link_and_symbols():
    idx = pd.date_range("2025-10-01", periods=250, freq="B")
    rng = np.random.default_rng(1)
    vret = rng.normal(0, 0.05, 250)
    vix = pd.Series(20 * np.cumprod(1 + vret), index=idx)
    price = pd.Series(100 * np.cumprod(1 - 0.1 * vret + rng.normal(0, 0.002, 250)), index=idx)
    link = md.vix_link(price, vix)
    assert link["corr"] < -0.8 and link["move_per_10pct_vix"] < 0
    assert md.price_symbol("JPYUSD") == ("JPY=X", True)
    assert md.price_symbol("AUDUSD") == ("AUDUSD=X", False)
    assert md.price_symbol("USDJPY") == ("JPY=X", False)
    assert md.price_symbol("DXY") == ("DX-Y.NYB", False)
    assert md.price_symbol("GBPJPY") == ("GBPJPY=X", False)


def test_vix_level_uses_long_history_and_absolute_zones():
    # 4 Jahre mit VIX ~22, dann 1 ruhiges Jahr ~14, heute 19.7
    idx = pd.date_range("2021-10-01", periods=5 * 252, freq="B")
    v = np.r_[np.full(4 * 252, 22.0), np.full(252 - 1, 14.0), [19.7]]
    vix = pd.Series(v, index=idx)
    long = md.risk_appetite(vix, None, None)
    short = md.risk_appetite(vix.iloc[-252:], None, None)
    assert long["vix_regime"] == "normal"
    assert long["vix_pct"] < 0.5 < short["vix_pct"]       # 5J: unterdurchschnittlich, 1J: "hoch"
    assert md.vix_regime(13) == "ruhig" and md.vix_regime(25) == "Stress" and md.vix_regime(35) == "Panik"


def test_vix_spikes_and_zone_advice():
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    v = np.full(60, 15.0); v[30:35] = [17, 19, 22, 26, 30]   # +100% in 5 Tagen
    spikes = md.vix_spike_periods(pd.Series(v, index=idx))
    assert len(spikes) == 1 and spikes[0][2] >= 30
    adv = md.zone_advice("AUDUSD", 1.5, "AUD")
    assert len(adv) == 4 and "fällt" in adv[3]["Für AUD"]
    adv_jpy = md.zone_advice("JPYUSD", -0.5, "JPY")
    assert "steigt" in adv_jpy[3]["Für JPY"]
    assert "kaum" in md.zone_advice("EURGBP", 0.1, "EURGBP")[2]["Für EURGBP"]


def test_vix_link_robust_to_bad_print():
    idx = pd.date_range("2025-10-01", periods=250, freq="B")
    rng = np.random.default_rng(2)
    vret = rng.normal(0, 0.05, 250)
    vix = pd.Series(20 * np.cumprod(1 + vret), index=idx)
    price = pd.Series(100 * np.cumprod(1 - 0.1 * vret + rng.normal(0, 0.002, 250)), index=idx)
    price.iloc[100] = price.iloc[100] * 5          # Fehlkurs
    assert md.vix_link(price, vix)["corr"] < -0.6


def test_trading_dates_align_fx_and_vix():
    fx = pd.DatetimeIndex(["2026-07-06 00:00", "2026-07-07 00:00"]).tz_localize("Europe/London")
    vix = pd.DatetimeIndex(["2026-07-06 00:00", "2026-07-07 00:00"]).tz_localize("America/New_York")
    utc_fx = fx.tz_convert("UTC")                       # 23:00 des Vortags
    want = pd.DatetimeIndex(["2026-07-06", "2026-07-07"])
    for idx in (fx, vix, utc_fx, utc_fx.tz_localize(None)):
        assert (md._trading_dates(idx) == want).all()


def test_fx_dates_shift_back_one_trading_day():
    idx = pd.DatetimeIndex(["2026-10-22 23:00", "2026-10-26 00:00", "2026-10-26 11:57"]).tz_localize("UTC")
    # Fr 23.10. 00:00 London -> Do 22.10.; Mo 26.10. 00:00 London (GMT) -> Fr 23.10.; Intraday bleibt Mo
    got = md._fx_trading_dates(idx)
    assert list(got.strftime("%Y-%m-%d")) == ["2026-10-22", "2026-10-23", "2026-10-26"]


def test_fx_dates_naive_index_also_shifted():
    idx = pd.DatetimeIndex(["2026-07-07", "2026-07-08", "2026-07-13"])      # Di, Mi, Mo
    assert list(md._fx_trading_dates(idx).strftime("%Y-%m-%d")) == ["2026-07-06", "2026-07-07", "2026-07-10"]


def _scores(d):
    return pd.DataFrame([{"currency": c, "score": v} for c, v in d.items()])


def test_watchlist_strong_vs_weak_and_convention():
    sc = _scores({"GBP": 0.6, "AUD": 0.4, "USD": 0.1, "EUR": 0.0, "CAD": -0.1, "CHF": -0.2, "NZD": -0.3, "JPY": -0.7})
    wl = md.build_watchlist(sc)
    assert wl[0]["pair"] == "GBPJPY" and wl[0]["direction"] == "LONG"
    assert len(wl) == 3
    pairs = [w["pair"] for w in wl]
    assert all(sum(c in p for p in pairs) <= 2 for c in md.CURRENCIES)
    # Marktkonvention: NZD stark, EUR schwach -> EURNZD SHORT
    wl2 = md.build_watchlist(_scores({"NZD": 0.8, "EUR": -0.8, **{c: 0.0 for c in ["GBP", "AUD", "USD", "CAD", "CHF", "JPY"]}}), top_n=1)
    assert wl2[0]["pair"] == "EURNZD" and wl2[0]["direction"] == "SHORT"
    assert md.build_watchlist(_scores({c: 0.0 for c in md.CURRENCIES})) == []


def test_regime_rules_and_blocks():
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    calm = pd.Series(np.full(30, 16.0), index=idx)
    r = md.trading_regime({"vix": 16.0, "score": 0.2}, calm)
    assert r["effective"] == "Normal" and r["risk_pct"] == 1.0
    spike = calm.copy(); spike.iloc[-1] = 22.0          # +37 % in 5 Tagen
    r2 = md.trading_regime({"vix": 22.0, "score": -0.3}, spike)
    assert r2["effective"] == "Panik" and r2["risk_pct"] == 0
    stress = md.trading_regime({"vix": 24.0, "score": -0.3}, pd.Series(np.full(30, 24.0), index=idx))
    empty = pd.DataFrame(columns=["currency", "time_utc", "importance", "event_code", "actual"])
    long_aud = md.trade_status({"pair": "AUDJPY", "direction": "LONG"}, stress, empty, NOW)
    short_aud = md.trade_status({"pair": "AUDJPY", "direction": "SHORT"}, stress, empty, NOW)
    assert long_aud["status"] == "gesperrt" and short_aud["status"] == "handelbar"


def test_central_bank_block_and_intervention():
    cal = _calendar()   # BoJ-Sitzung am 24.10., NOW = 20.10.
    normal = {"effective": "Normal", "mult": 1.0}
    st_far = md.trade_status({"pair": "GBPJPY", "direction": "LONG"}, normal, cal, NOW)
    assert st_far["status"] == "Hinweis" and "BoJ" in st_far["warns"][0]
    st_near = md.trade_status({"pair": "GBPJPY", "direction": "LONG"}, normal, cal, pd.Timestamp("2026-10-23 12:00"))
    assert st_near["status"] == "gesperrt"
    hi = {"level": "hoch"}
    assert "JPY-Interventionsrisiko hoch" in md.trade_status({"pair": "USDJPY", "direction": "LONG"}, normal, cal.iloc[:0], NOW, hi)["blocks"]
    assert md.trade_status({"pair": "USDJPY", "direction": "SHORT"}, normal, cal.iloc[:0], NOW, hi)["status"] == "handelbar"


def test_currency_scores_relative():
    cal = _calendar()
    sc = md.currency_scores(cal, NOW, lambda c: ({"GBP": 0.5, "JPY": -0.5}.get(c), 5), lambda c: None,
                            {"score": 0.5, "vix": 15})
    top, bottom = sc.iloc[0]["currency"], sc.dropna(subset=["score"]).iloc[-1]["currency"]
    assert top in ("GBP", "AUD", "NZD") and bottom in ("JPY", "CHF")


def test_crowded_flag_only_where_set():
    stats = {"GBP": {"score": 0.5, "percentile": 0.95}, "EUR": {"score": 0.1, "percentile": 0.5}}
    sc = md.currency_scores(_calendar(), NOW, lambda c: (None, 0), lambda c: stats.get(c), None)
    flags = dict(zip(sc["currency"], sc["crowded"]))
    assert flags["GBP"] == "Longs überfüllt"
    assert not isinstance(flags["EUR"], str) and not isinstance(flags["USD"], str)
