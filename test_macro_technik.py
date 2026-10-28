"""Tests fuer macro_technik.py (synthetisch, ohne Netzwerk)."""
import numpy as np
import pandas as pd

import macro_technik as mt


def _vix(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_features_zones_and_spike():
    v = np.r_[np.full(300, 14.0), np.full(5, 14.0), [25.0], np.full(20, 35.0)]
    f = mt.vix_features(_vix(v))
    assert f["zone"].iloc[10] == "Ruhig"
    assert f["zone_eff"].iloc[305] == "Panik"          # Stress + Sprung -> Panik
    assert f["app"].iloc[10] > 0 > f["app"].iloc[-1]   # ruhig = Risk-on, hoch = Risk-off


def test_annotate_no_lookahead_and_sides():
    v = np.r_[np.full(300, 13.0), np.full(300, 35.0)]
    feats = mt.vix_features(_vix(v))
    calm_day, panic_day = feats.index[250], feats.index[590]
    tr = pd.DataFrame([
        {"sym": "AUDJPY", "dir": 1, "fill_dt": calm_day, "R": 1.0},     # Risk-on + AUDJPY long = mit
        {"sym": "AUDJPY", "dir": 1, "fill_dt": panic_day, "R": -1.0},   # Risk-off + long = gegen
        {"sym": "AUDNZD", "dir": 1, "fill_dt": calm_day, "R": 0.5},     # gleiche Empfindlichkeit = neutral
    ])
    t = mt.annotate(tr, feats)
    assert (t["vix_day"] < t["decision_day"]).all()
    sides = dict(zip(zip(t.sym, t.fill_dt), t.side_a))
    assert sides[("AUDJPY", calm_day)] == "mit"
    assert sides[("AUDJPY", panic_day)] == "gegen"
    assert sides[("AUDNZD", calm_day)] == "neutral"


def test_measured_correlation_side():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    vret = rng.normal(0, 0.05, 400)
    vix = pd.Series(15 * np.cumprod(1 + vret), index=idx)
    price = pd.Series(np.cumprod(1 - 0.05 * vret + rng.normal(0, 0.002, 400)), index=idx)
    corr = mt.rolling_vix_corr(price, vix)
    assert corr.dropna().iloc[-1] < -0.5
    feats = mt.vix_features(vix)
    day = idx[-5]
    tr = pd.DataFrame([{"sym": "XXXYYY", "dir": 1, "fill_dt": day, "R": 0.0}])
    t = mt.annotate(tr, feats, {"XXXYYY": corr})
    app = t["app"].iloc[0]
    expected = "neutral" if abs(app) < mt.APP_NEUTRAL else ("mit" if app > 0 else "gegen")
    assert t["side_b"].iloc[0] == expected


def test_future_view_inverts_usd_base():
    tr = pd.DataFrame([{"sym": "USDCAD", "dir": 1, "R": 1.0}, {"sym": "AUDUSD", "dir": -1, "R": 1.0},
                       {"sym": "EURGBP", "dir": 1, "R": 1.0}])
    v = mt.to_future_view(tr)
    assert list(v["market"]) == ["CAD (USDCAD)", "AUD (AUDUSD)"]
    assert list(v["fdir"]) == [-1, -1]


def test_stats_bh_and_verdict():
    assert mt.stats(pd.Series([1.0, -1.0, 2.0]))["pf"] == 3.0
    q = mt.bh_adjust([0.01, 0.04, float("nan"), 0.5])
    assert abs(q[0] - 0.03) < 1e-9 and np.isnan(q[2]) and q[3] == 0.5
    d, p = mt.diff_test(pd.Series(np.full(50, 0.5) + np.r_[0.1, -0.1] .repeat(25)), pd.Series(np.full(50, -0.5) + np.r_[0.1, -0.1].repeat(25)))
    assert d > 0.9 and p < 1e-6
    assert mt.filter_verdict(100, 0.2, 0.01, 0.1)[0] == "🟢"
    assert mt.filter_verdict(100, -0.2, 0.01, -0.1)[0] == "🔴"
    assert mt.filter_verdict(10, 0.2, 0.01, 0.1)[0] == "⚪"


def test_load_vix_repo_file():
    v = mt.load_vix()
    assert len(v) > 5000 and v.index.min().year <= 2005
    assert abs(v.loc["2020-03-16"] - 82.69) < 0.01
