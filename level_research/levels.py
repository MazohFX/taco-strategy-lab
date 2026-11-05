"""Objektive Trading-Levels aus Dukascopy-M15 (UTC), ohne Lookahead.

Jedes Level ist ein Array der Länge n_bars: der Wert, der VOR Bar i bekannt ist
(NaN = nicht definiert). Handelstag = 17:00 New York Rollover.
"""
import numpy as np
import pandas as pd
import numba as nb
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "dukascopy_intraday"

SPREAD_PIPS = {
    "EURUSD": 0.8, "USDJPY": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.0, "USDCHF": 1.3,
    "USDCAD": 1.5, "NZDUSD": 1.5, "EURGBP": 1.2, "EURJPY": 1.5, "EURCHF": 1.6,
    "GBPJPY": 2.5, "AUDJPY": 1.8, "CADJPY": 2.0, "CHFJPY": 2.2, "NZDJPY": 2.2,
}


def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def spread_price(pair):
    return SPREAD_PIPS.get(pair, 2.5) * pip_size(pair)


def load_m15(pair):
    df = pd.read_csv(DATA / f"{pair}_M15.csv.gz")
    ts = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    ny = ts.dt.tz_convert("America/New_York")
    ldn = ts.dt.tz_convert("Europe/London")
    tdate = (ny + pd.Timedelta(hours=7)).dt.date
    out = dict(
        ts=ts.dt.tz_localize(None).values,
        o=df["open"].values.astype(np.float64), h=df["high"].values.astype(np.float64),
        l=df["low"].values.astype(np.float64), c=df["close"].values.astype(np.float64),
        v=np.maximum(df["ticks"].fillna(0).values.astype(np.float64), 1.0),
        utc_h=ts.dt.hour.values.astype(np.int64),
        ny_h=ny.dt.hour.values.astype(np.int64), ny_m=ny.dt.minute.values.astype(np.int64),
        ldn_h=ldn.dt.hour.values.astype(np.int64), ldn_m=ldn.dt.minute.values.astype(np.int64),
    )
    td = pd.to_datetime(pd.Series(tdate))
    out["tdate"] = td.values
    out["day"] = pd.factorize(td)[0].astype(np.int64)
    out["hb"] = (((out["ny_h"] + 7) % 24) // 4).astype(np.int64)  # 6 x 4h Buckets ab 17:00 NY
    return out


# --------------------------------------------------------------------------- numba helpers
@nb.njit(cache=True)
def _profile_days(day, h, l, v, n_days, A, bins_per_atr):
    """POC/VAH/VAL je Gruppe (Tag oder Woche); Volumen gleichmäßig über [low, high] verteilt."""
    poc = np.full(n_days, np.nan); vah = np.full(n_days, np.nan); val = np.full(n_days, np.nan)
    n = len(day)
    start = 0
    while start < n:
        g = day[start]; end = start
        while end < n and day[end] == g:
            end += 1
        a = A[g]
        if not np.isnan(a) and a > 0:
            lo = np.min(l[start:end]); hi = np.max(h[start:end])
            w = a / bins_per_atr
            nb_ = int((hi - lo) / w) + 1
            if nb_ > 5000:
                nb_ = 5000; w = (hi - lo) / 4999 + 1e-12
            hist = np.zeros(nb_)
            for i in range(start, end):
                b0 = int((l[i] - lo) / w); b1 = int((h[i] - lo) / w)
                if b1 >= nb_: b1 = nb_ - 1
                cnt = b1 - b0 + 1
                for b in range(b0, b1 + 1):
                    hist[b] += v[i] / cnt
            pb = np.argmax(hist)
            tot = hist.sum(); acc = hist[pb]; lo_b = pb; hi_b = pb
            while acc < 0.7 * tot and (lo_b > 0 or hi_b < nb_ - 1):
                up = hist[hi_b + 1] if hi_b < nb_ - 1 else -1.0
                dn = hist[lo_b - 1] if lo_b > 0 else -1.0
                if up >= dn:
                    hi_b += 1; acc += up
                else:
                    lo_b -= 1; acc += dn
            poc[g] = lo + (pb + 0.5) * w; vah[g] = lo + (hi_b + 1) * w; val[g] = lo + lo_b * w
        start = end
    return poc, vah, val


@nb.njit(cache=True)
def _naked_poc(poc, dH, dL, dO, lookback):
    n = len(poc)
    up = np.full(n, np.nan); dn = np.full(n, np.nan)
    for d in range(1, n):
        best_up = np.inf; best_dn = -np.inf
        for j in range(max(0, d - lookback), d):
            p = poc[j]
            if np.isnan(p):
                continue
            touched = False
            for k in range(j + 1, d):
                if dL[k] <= p <= dH[k]:
                    touched = True; break
            if touched:
                continue
            if p >= dO[d] and p < best_up: best_up = p
            if p < dO[d] and p > best_dn: best_dn = p
        if best_up < np.inf: up[d] = best_up
        if best_dn > -np.inf: dn[d] = best_dn
    return up, dn


@nb.njit(cache=True)
def _running_vwap(grp, tp, v):
    """VWAP und VW-Std je Gruppe, Wert vor Bar i (erste 4 Bars einer Gruppe NaN)."""
    n = len(grp)
    vw = np.full(n, np.nan); sd = np.full(n, np.nan)
    cv = 0.0; cpv = 0.0; cp2v = 0.0; cnt = 0
    for i in range(n):
        if i == 0 or grp[i] != grp[i - 1]:
            cv = 0.0; cpv = 0.0; cp2v = 0.0; cnt = 0
        if cnt >= 4:
            m = cpv / cv
            vw[i] = m
            var = cp2v / cv - m * m
            sd[i] = np.sqrt(var) if var > 0 else 0.0
        cv += v[i]; cpv += tp[i] * v[i]; cp2v += tp[i] * tp[i] * v[i]; cnt += 1
    return vw, sd


@nb.njit(cache=True)
def _fvg(dH, dL, dC):
    """Jüngste unaufgefüllte Daily-FVG: bull top (=low[j]) / bear bottom (=high[j]); bekannt ab j+1."""
    n = len(dH)
    bull = np.full(n, np.nan); bear = np.full(n, np.nan)
    bt = np.nan; bb = np.nan; rt = np.nan; rb = np.nan  # bull top/bottom, bear top/bottom
    for d in range(n):
        bull[d] = bt; bear[d] = rb
        # Füllung durch Tag d
        if not np.isnan(bt) and dL[d] <= bb:
            bt = np.nan; bb = np.nan
        if not np.isnan(rb) and dH[d] >= rt:
            rt = np.nan; rb = np.nan
        if d >= 2:
            if dL[d] > dH[d - 2]:
                bt = dL[d]; bb = dH[d - 2]
            if dH[d] < dL[d - 2]:
                rb = dH[d]; rt = dL[d - 2]
    return bull, bear


@nb.njit(cache=True)
def _fractals(H, L, avail_offset):
    """Letztes bestätigtes Williams-Fraktal (2/2). Wert für Bar t (bekannt vor t)."""
    n = len(H)
    fh = np.full(n, np.nan); fl = np.full(n, np.nan)
    ch = np.nan; cl = np.nan
    for t in range(n):
        # Fraktal an j=t-1-2 ist bestätigt nach Schluss von t-1 → gültig für t
        j = t - 3
        if j >= 2:
            if H[j] > H[j - 1] and H[j] > H[j - 2] and H[j] > H[j + 1] and H[j] > H[j + 2]:
                ch = H[j]
            if L[j] < L[j - 1] and L[j] < L[j - 2] and L[j] < L[j + 1] and L[j] < L[j + 2]:
                cl = L[j]
        fh[t] = ch; fl[t] = cl
    return fh, fl


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False, min_periods=span).mean().values


def _shift(x, k=1):
    y = np.full_like(x, np.nan, dtype=np.float64)
    y[k:] = x[:-k]
    return y


def _group_ohlc(key, o, h, l, c):
    k = pd.factorize(key)[0]
    s = pd.DataFrame(dict(k=k, o=o, h=h, l=l, c=c)).groupby("k").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    return k, s["o"].values, s["h"].values, s["l"].values, s["c"].values


def _tf_index(m, hours):
    """Gruppen-ID für H1/H4-Bars (UTC-ausgerichtet)."""
    t = pd.to_datetime(m["ts"])
    return pd.factorize(t.floor(f"{hours}h"))[0].astype(np.int64)


def build_levels(pair, m):
    n = len(m["c"]); day = m["day"]; nd = day.max() + 1
    o, h, l, c, v = m["o"], m["h"], m["l"], m["c"], m["v"]
    # Tages-OHLC
    df = pd.DataFrame(dict(day=day, o=o, h=h, l=l, c=c, v=v))
    g = df.groupby("day")
    dO = g.o.first().values; dH = g.h.max().values; dL = g.l.min().values; dC = g.c.last().values
    tdates = pd.to_datetime(pd.Series(m["tdate"]).groupby(day).first().values)
    prevC = _shift(dC)
    tr = np.maximum(dH - dL, np.maximum(np.abs(dH - prevC), np.abs(dL - prevC)))
    tr[0] = dH[0] - dL[0]
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values
    A = _shift(atr)                       # ATR bekannt zu Tagesbeginn
    D = {}                                # tägliche statische Levels (Länge nd)
    pH, pL, pC = _shift(dH), _shift(dL), _shift(dC)
    R = pH - pL
    D["DO"] = dO
    D["PDH"], D["PDL"] = pH, pL
    D["PD_F382"], D["PD_MID"], D["PD_F618"] = pL + 0.382 * R, pL + 0.5 * R, pL + 0.618 * R
    D["PD_X272U"], D["PD_X272D"] = pH + 0.272 * R, pL - 0.272 * R
    D["PD_X618U"], D["PD_X618D"] = pH + 0.618 * R, pL - 0.618 * R
    P = (pH + pL + pC) / 3
    D["PIV_P"], D["PIV_R1"], D["PIV_S1"] = P, 2 * P - pL, 2 * P - pH
    D["PIV_R2"], D["PIV_S2"] = P + R, P - R
    D["CAM_H3"], D["CAM_L3"] = pC + 1.1 * R / 4, pC - 1.1 * R / 4
    D["CAM_H4"], D["CAM_L4"] = pC + 1.1 * R / 2, pC - 1.1 * R / 2
    D["FPIV_R1"], D["FPIV_S1"] = P + 0.382 * R, P - 0.382 * R
    D["FPIV_R2"], D["FPIV_S2"] = P + 0.618 * R, P - 0.618 * R
    # Woche / Monat / Quartal / Jahr
    wk = (tdates - pd.to_timedelta(tdates.weekday, unit="D")).values
    for name, key in [("W", wk), ("M", tdates.to_period("M").astype(str).values),
                      ("Q", tdates.to_period("Q").astype(str).values), ("Y", tdates.year.values)]:
        k, go, gh, gl, gc = _group_ohlc(key, dO, dH, dL, dC)
        D[f"{name}O"] = go[k]
        ph, pl, pc = _shift(gh)[k], _shift(gl)[k], _shift(gc)[k]
        D[f"P{name}H"], D[f"P{name}L"] = ph, pl
        if name == "W":
            D["PWC"] = pc
            rr = ph - pl
            D["PW_F382"], D["PW_MID"], D["PW_F618"] = pl + 0.382 * rr, pl + 0.5 * rr, pl + 0.618 * rr
            pw = (ph + pl + pc) / 3
            D["WPIV_P"] = pw
        if name == "M":
            D["MPIV_P"] = (ph + pl + pc) / 3
    # Indikatoren D1 (Werte bis Vortag)
    for sp in (20, 50, 200):
        D[f"D_EMA{sp}"] = _shift(_ema(dC, sp))
    D["D_SMA200"] = _shift(pd.Series(dC).rolling(200).mean().values)
    m20 = pd.Series(dC).rolling(20).mean(); s20 = pd.Series(dC).rolling(20).std()
    D["D_BBU"], D["D_BBL"] = _shift((m20 + 2 * s20).values), _shift((m20 - 2 * s20).values)
    hh = lambda w: _shift(pd.Series(dH).rolling(w).max().values)
    ll = lambda w: _shift(pd.Series(dL).rolling(w).min().values)
    D["KIJUN26"] = (hh(26) + ll(26)) / 2
    D["DON20H"], D["DON20L"], D["DON55H"], D["DON55L"] = hh(20), ll(20), hh(55), ll(55)
    D["W52H"], D["W52L"] = hh(250), ll(250)
    # Struktur D1
    fh, fl = _fractals(dH, dL, 3)
    D["D_FRACH"], D["D_FRACL"] = fh, fl
    sw = fh - fl
    for f, nm in [(0.214, "SW_F786D"), (0.382, "SW_F382"), (0.5, "SW_F50"), (0.618, "SW_F618"), (0.786, "SW_F786")]:
        D[nm] = fl + f * sw
    D["FVG_BULL"], D["FVG_BEAR"] = _fvg(dH, dL, dC)
    # Runde Zahlen relativ zum Daily Open
    st = 100 * pip_size(pair)
    D["RN00_UP"], D["RN00_DN"] = np.ceil(dO / st) * st, np.floor(dO / st) * st
    D["RN50_UP"] = (np.floor(dO / st - 0.5) + 1.5) * st
    D["RN50_DN"] = (np.ceil(dO / st + 0.5) - 1.5) * st
    # ATR-Projektionen
    for f, nm in [(0.5, "05"), (1.0, "10")]:
        D[f"ADR{nm}_UP"], D[f"ADR{nm}_DN"] = dO + f * A, dO - f * A
    # Volumenprofil Vortag / Vorwoche
    poc, vah, val = _profile_days(day, h, l, v, nd, A, 50.0)
    D["PD_POC"], D["PD_VAH"], D["PD_VAL"] = _shift(poc), _shift(vah), _shift(val)
    nu, ndn = _naked_poc(poc, dH, dL, dO, 20)
    D["NPOC_UP"], D["NPOC_DN"] = nu, ndn
    wk_id = pd.factorize(wk)[0]
    wk_bar = wk_id[day]
    Aw = np.array([np.nanmean(A[wk_id == i]) if np.any(~np.isnan(A[wk_id == i])) else np.nan
                   for i in range(wk_id.max() + 1)])
    wpoc, wvah, wval = _profile_days(wk_bar, h, l, v, wk_id.max() + 1, Aw, 25.0)
    D["PW_POC"], D["PW_VAH"], D["PW_VAL"] = _shift(wpoc)[wk_id], _shift(wvah)[wk_id], _shift(wval)[wk_id]

    L = {k: val_[day] for k, val_ in D.items()}   # auf Bars broadcasten

    # intraday dynamische Levels
    tp = (h + l + c) / 3
    vw, sd = _running_vwap(day, tp, v)
    L["VWAP_D"] = vw
    L["VWAP_D_U1"], L["VWAP_D_L1"] = vw + sd, vw - sd
    L["VWAP_D_U2"], L["VWAP_D_L2"] = vw + 2 * sd, vw - 2 * sd
    pdv = pd.Series(vw).groupby(day).last().values  # (Wert vor letzter Bar ~ Tagesschluss-VWAP)
    L["PD_VWAP"] = _shift(pdv)[day]
    L["VWAP_W"], _ = _running_vwap(wk_bar, tp, v)
    mkey = pd.factorize(pd.Series(tdates.to_period("M").astype(str).values)[day].values)[0]
    L["VWAP_M"], _ = _running_vwap(mkey.astype(np.int64), tp, v)
    # H1 / H4 EMAs und H4-Fraktale
    for hrs, sp, nm in [(1, 200, "H1_EMA200"), (4, 50, "H4_EMA50"), (4, 200, "H4_EMA200")]:
        gi = _tf_index(m, hrs)
        gc_ = pd.Series(c).groupby(gi).last().values
        e = _shift(_ema(gc_, sp))
        L[nm] = e[gi]
    gi4 = _tf_index(m, 4)
    g4h = pd.Series(h).groupby(gi4).max().values; g4l = pd.Series(l).groupby(gi4).min().values
    f4h, f4l = _fractals(g4h, g4l, 3)
    L["H4_FRACH"], L["H4_FRACL"] = f4h[gi4], f4l[gi4]
    # Sessions
    first_of = lambda mask: _session_value(day, mask, o, nd)
    utc = m["utc_h"]
    asia = (utc >= 0) & (utc < 6)
    ah, al = _session_hl(day, asia, h, l, nd)
    valid_asia = utc >= 6
    L["ASIA_H"] = np.where(valid_asia & (utc < 21), ah[day], np.nan)
    L["ASIA_L"] = np.where(valid_asia & (utc < 21), al[day], np.nan)
    lo_open = (m["ldn_h"] == 8) & (m["ldn_m"] == 0)
    lov = first_of(lo_open)
    after_lo = (m["ldn_h"] > 8) | ((m["ldn_h"] == 8) & (m["ldn_m"] > 0))
    after_lo &= (m["ny_h"] < 17)
    L["LDN_OPEN"] = np.where(after_lo, lov[day], np.nan)
    orm = (m["ldn_h"] == 8)
    orh, orl = _session_hl(day, orm, h, l, nd)
    after_or = (m["ldn_h"] >= 9) & (m["ny_h"] < 17)
    L["LDN_ORH"] = np.where(after_or, orh[day], np.nan)
    L["LDN_ORL"] = np.where(after_or, orl[day], np.nan)
    ny8 = (m["ny_h"] == 8) & (m["ny_m"] == 0)
    nyv = first_of(ny8)
    after_ny = ((m["ny_h"] > 8) | ((m["ny_h"] == 8) & (m["ny_m"] > 0))) & (m["ny_h"] < 17)
    L["NY_OPEN"] = np.where(after_ny, nyv[day], np.nan)
    mid = (m["ny_h"] == 0) & (m["ny_m"] == 0)
    midv = first_of(mid)
    after_mid = (m["ny_h"] >= 0) & (m["ny_h"] < 17) & ~mid
    L["MIDNIGHT_OPEN"] = np.where(after_mid, midv[day], np.nan)

    names = sorted(L.keys())
    Lmat = np.vstack([np.asarray(L[k], dtype=np.float64) for k in names])
    return names, Lmat, A[day], dO[day], A, dO


def _session_value(day, mask, x, nd):
    out = np.full(nd, np.nan)
    idx = np.where(mask)[0]
    d = day[idx]
    first = np.unique(d, return_index=True)
    out[first[0]] = x[idx[first[1]]]
    return out


def _session_hl(day, mask, h, l, nd):
    oh = np.full(nd, np.nan); ol = np.full(nd, np.nan)
    if mask.any():
        s = pd.DataFrame(dict(d=day[mask], h=h[mask], l=l[mask])).groupby("d").agg(h=("h", "max"), l=("l", "min"))
        oh[s.index.values] = s.h.values; ol[s.index.values] = s.l.values
    return oh, ol
