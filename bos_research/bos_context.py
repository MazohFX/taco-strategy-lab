"""
BOS mit Kontext: Trade-Datensatz je Bruch inkl. aller objektiven Konfluenzen.

Pro Paar / TF / Swing-Methode / Bruchart werden drei Strategie-Familien erzeugt:
  fib     : Tobis Setup (Limit 0.618, Stop Fib 1, TP -0.27/-0.62/-1 zu je 1/3, BE ab Fib 0)
  mom     : Osler-Momentum - Markt-Einstieg am Schluss der Bruchkerze, Stop 1 ATR, Ziel 1R / 2R
  sweep   : Liquidity Sweep - Docht ueber/unter den Swing, Schluss zurueck im Bereich
            -> Gegenrichtung, Stop hinter dem Docht + 0.1 ATR, Ziel 1R / 2R
Merkmale (alle nur mit Daten bis zur Einstiegskerze):
  htf_bos  : Richtung des letzten Bruchs auf der hoeheren Zeitebene (ZZ2-Close), +1 = gleich, -1 = gegen
  htf_ema  : Schluss der letzten abgeschlossenen HTF-Kerze ueber/unter EMA200 (HTF), +1 = gleich
  hh_hl    : Bruch mit intakter Struktur (HH+HL bzw. LL+LH) = Trendfortsetzung, sonst CHoCH
  body_atr : Koerper der Bruchkerze in ATR (Verdraengung)
  ratio    : Bruchbein / Ruecklaufbein
  Y,Q,M,W,D: Open liegt in der Zone 0.618-0.786
  poc      : POC des laufenden Monats (Monatsbeginn bis Vorkerze) in der Zone
  hour     : UTC-Stunde der Einstiegskerze
  atr_pct  : ATR-Perzentil (250 Kerzen)
"""
import os, sys, time
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp
import bos_grid as g

HTF = {"M15": "H1", "H1": "H4", "H4": "D1"}
TF_MIN = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}
SWINGS = [("ZZ", 1.5), ("ZZ", 2.0), ("ZZ", 3.0), ("FR", 5), ("FR", 8)]


@njit(cache=True)
def events_j(h, l, c, a, typ, ext, pr, cf, wick):
    """wie bos_grid.find_events, zusaetzlich Index j des gebrochenen Swings."""
    n = len(h); m = len(typ)
    et = np.zeros(m, np.int64); ed = np.zeros(m, np.int8); eo = np.zeros(m, np.int64)
    eop = np.zeros(m); ej = np.zeros(m, np.int64); erat = np.zeros(m); k = 0
    last_hi = -1; last_lo = -1; p = 0; hb = True; lb = True
    for t in range(1, n):
        while p < m and cf[p] < t:
            if typ[p] == 1:
                last_hi = p; hb = False
            else:
                last_lo = p; lb = False
            p += 1
        if np.isnan(a[t]): continue
        for d in (1, -1):
            j = last_hi if d == 1 else last_lo
            if j < 0 or (hb if d == 1 else lb): continue
            lvl = pr[j]
            px = (h[t] if d == 1 else l[t]) if wick else c[t]
            if d * (px - lvl) <= 0: continue
            if d == 1: hb = True
            else: lb = True
            s = ext[j] + 1
            if s > t: continue
            oi = s
            for u in range(s, t + 1):
                if d == 1:
                    if l[u] < l[oi]: oi = u
                else:
                    if h[u] > h[oi]: oi = u
            op = l[oi] if d == 1 else h[oi]
            pull = d * (lvl - op)
            if pull <= 0: continue
            ext_px = h[t] if d == 1 else l[t]
            et[k] = t; ed[k] = d; eo[k] = oi; eop[k] = op; ej[k] = j
            erat[k] = d * (ext_px - op) / pull; k += 1
    return et[:k], ed[:k], eo[:k], eop[:k], ej[:k], erat[:k]


@njit(cache=True)
def sweeps(h, l, c, a, typ, pr, cf):
    """Docht ueber das letzte Swing-Hoch, Schluss darunter -> Short (d=-1); spiegelbildlich Long."""
    n = len(h); m = len(typ)
    et = np.zeros(m, np.int64); ed = np.zeros(m, np.int8); ex = np.zeros(m); k = 0
    last_hi = -1; last_lo = -1; p = 0; hu = True; lu = True
    for t in range(1, n):
        while p < m and cf[p] < t:
            if typ[p] == 1:
                last_hi = p; hu = False
            else:
                last_lo = p; lu = False
            p += 1
        if np.isnan(a[t]): continue
        if last_hi >= 0 and not hu:
            if c[t] > pr[last_hi]: hu = True              # echter Bruch -> Level verbraucht
            elif h[t] > pr[last_hi]:
                et[k] = t; ed[k] = -1; ex[k] = h[t]; k += 1; hu = True
        if last_lo >= 0 and not lu:
            if c[t] < pr[last_lo]: lu = True
            elif l[t] < pr[last_lo]:
                et[k] = t; ed[k] = 1; ex[k] = l[t]; k += 1; lu = True
    return et[:k], ed[:k], ex[:k]


@njit(cache=True)
def fixed_r(h, l, c, t0, d, entry, stop, tps, sp, hold):
    """Markt-Einstieg am Schluss t0; je Ziel in tps (R-Vielfache) ein Ergebnis in R."""
    m = len(t0); n = len(h); k = len(tps); R = np.full((m, k), np.nan)
    for q in range(m):
        e = entry[q]; s = stop[q]; dd = d[q]; risk = abs(e - s) + sp
        if risk <= 0 or abs(e - s) <= 0: continue
        for z in range(k):
            tp = e + dd * tps[z] * abs(e - s); res = np.nan
            last = min(n - 1, t0[q] + hold)
            for u in range(t0[q] + 1, last + 1):
                adv = l[u] if dd == 1 else h[u]; fav = h[u] if dd == 1 else l[u]
                if dd * (adv - s) <= 0:
                    res = -abs(e - s); break
                if dd * (fav - tp) >= 0:
                    res = tps[z] * abs(e - s); break
            if np.isnan(res): res = dd * (c[last] - e)
            R[q, z] = (res - sp) / risk
    return R


def load_any(sym, tf, root):
    if tf != "D1":
        return bp.load(sym, tf, root)
    h4 = bp.load(sym, "H4", root)
    if h4 is None: return None
    x = h4.set_index("dt").resample("D").agg({"open": "first", "high": "max", "low": "min",
                                              "close": "last", "vol": "sum"}).dropna().reset_index()
    return x


def htf_signals(sym, tf, root):
    """Zeitreihe (verfuegbar_ab, bos_dir, ema_dir) der hoeheren Zeitebene."""
    hd = load_any(sym, HTF[tf], root)
    hh, hl, hc = (hd[x].values.astype(np.float64) for x in ["high", "low", "close"])
    ha = bp.atr(hd).astype(np.float64)
    sv = g.zigzag(hh, hl, ha, 2.0)
    et, ed, *_ = events_j(hh, hl, hc, ha, *sv, False)
    close_t = hd.dt + pd.Timedelta(minutes=TF_MIN[HTF[tf]])
    bos = pd.Series(np.nan, index=range(len(hd))); bos.iloc[et] = ed
    ema = pd.Series(hc).ewm(span=200, adjust=False).mean().values
    s = pd.DataFrame({"avail": close_t, "htf_bos": bos.ffill().values,
                      "htf_ema": np.sign(hc - ema)})
    return s.sort_values("avail")


def month_poc(h, l, v, mstart, i1):
    i0 = i1 - mstart[i1]
    if i1 - i0 < 3: return np.nan
    return g_profile(h, l, v, i0, i1)


def g_profile(h, l, v, i0, i1, nb=40):
    hh, ll, vv = h[i0:i1 + 1], l[i0:i1 + 1], v[i0:i1 + 1]
    hi, lo = hh.max(), ll.min()
    if not hi > lo: return np.nan
    w = (hi - lo) / nb
    a = np.clip(((ll - lo) / w).astype(int), 0, nb - 1)
    b = np.maximum(np.clip(np.ceil((hh - lo) / w).astype(int), 1, nb), a + 1)
    diff = np.zeros(nb + 1)
    np.add.at(diff, a, vv / (b - a)); np.add.at(diff, b, -vv / (b - a))
    return lo + (np.cumsum(diff)[:nb].argmax() + .5) * w


def run_pair(sym, tf, root):
    df = bp.load(sym, tf, root)
    if df is None or len(df) < 2000: return []
    o, h, l, c, v = (df[x].values.astype(np.float64) for x in ["open", "high", "low", "close", "vol"])
    a = bp.atr(df).astype(np.float64); sp = bp.spread(sym); dts = df.dt
    atr_pct = pd.Series(a).rolling(250).rank(pct=True).values
    opens = {}
    for nm, fq in [("Y", "Y"), ("Q", "Q"), ("M", "M"), ("W", "W-SUN"), ("D", "D")]:
        opens[nm] = df.groupby(dts.dt.to_period(fq))["open"].transform("first").values
    mstart = df.groupby(dts.dt.to_period("M")).cumcount().values
    # ---- Key Levels (zur Zeit des Ursprungs bekannt) ----
    import full_setup as fs
    pp = fs.prev_period_profiles(df, h, l, v)
    lv_names, lv_cols = [], []
    for nm in ["Y", "Q", "M"]:
        for j, k in enumerate(["POC", "VAH", "VAL"]):
            lv_names.append(f"{nm}-1 {k}"); lv_cols.append(pp[nm][:, j])
    for nm in ["Y", "Q", "M", "W"]:
        lv_names.append(f"{nm}-Open"); lv_cols.append(opens[nm])
    for nm, fq in [("D", "D"), ("W", "W-SUN")]:
        key = dts.dt.to_period(fq)
        ph = df.groupby(key)["high"].max().shift(); pl = df.groupby(key)["low"].min().shift()
        lv_names += [f"PD{nm}H", f"PD{nm}L"]
        lv_cols += [key.map(ph).values.astype(float), key.map(pl).values.astype(float)]
    LV = np.column_stack(lv_cols)

    def kl_at(i, price):
        dd = np.abs(LV[i] - price) / a[i]
        if np.all(np.isnan(dd)): return np.nan, ""
        k = int(np.nanargmin(dd)); return dd[k], lv_names[k]

    def sides(d, i, px):
        return {f"side{nm}": int(d * (px - opens[nm][i]) > 0) for nm in ["D", "W", "M"]}
    hs = htf_signals(sym, tf, root)
    bar_open = pd.DataFrame({"t": dts}).reset_index()
    ctx = pd.merge_asof(bar_open.sort_values("t"), hs, left_on="t", right_on="avail",
                        direction="backward").sort_values("index")
    hb, he = ctx.htf_bos.values, ctx.htf_ema.values
    rows = []
    tps = np.array([1.0, 2.0])
    for sw, par in SWINGS:
        sv = g.zigzag(h, l, a, float(par)) if sw == "ZZ" else g.fractal(h, l, int(par))
        typ, ext, pr, cf = sv
        name = f"{sw}{par:g}"
        for br in ("close", "wick"):
            et, ed, eo, eop, ej, erat = events_j(h, l, c, a, *sv, br == "wick")
            if len(et) == 0: continue
            # Struktur HH/HL
            hhl = np.zeros(len(et), np.int8)
            for q in range(len(et)):
                j = ej[q]; d = ed[q]
                ps = [i for i in range(j - 1, max(-1, j - 6), -1) if typ[i] == d]
                po = [i for i in range(j - 1, max(-1, j - 6), -1) if typ[i] == -d]
                if ps and po:
                    hhl[q] = int(d * (pr[j] - pr[ps[0]]) > 0 and d * (eop[q] - pr[po[0]]) > 0)
            body = ed * (c[et] - o[et]) / a[et]
            base = dict(sym=sym, tf=tf, swing=name, brk=br)
            # ---- fib ----
            R, fidx, E0, rk = g.simulate(h, l, c, et, ed, eo, eop, 0.618, sp, g.FILL_BARS, g.HOLD_BARS)
            for q in np.flatnonzero(fidx >= 0):
                f = fidx[q]; d = int(ed[q]); E = E0[q]; rng = rk[q]
                z = sorted([E - d * 0.618 * rng, E - d * 0.786 * rng])
                r = dict(base, fam="fib", t=dts.iat[et[q]], fill=dts.iat[f], dir=d, R=R[q], R2=np.nan,
                         ratio=erat[q], body=body[q], hhl=hhl[q], leg_atr=rng / a[et[q]],
                         htf_bos=d * hb[f], htf_ema=d * he[f], hour=dts.iat[f].hour, atr_pct=atr_pct[f])
                for nm in opens: r[nm] = int(z[0] <= opens[nm][f] <= z[1])
                kd, kt = kl_at(eo[q], eop[q]); r.update(kl_dist=kd, kl_type=kt, **sides(d, et[q], c[et[q]]))
                pc = month_poc(h, l, v, mstart, f - 1)
                r["poc"] = int(not np.isnan(pc) and z[0] <= pc <= z[1])
                rows.append(r)
            # ---- mom (nur Close-Brueche) ----
            if br == "close":
                st = c[et] - ed * a[et]
                Rm = fixed_r(h, l, c, et, ed.astype(np.int64), c[et], st, tps, sp, 30)
                for q in range(len(et)):
                    t = et[q]; d = int(ed[q]); kd, kt = kl_at(eo[q], eop[q])
                    rows.append(dict(base, fam="mom", t=dts.iat[t], fill=dts.iat[t], dir=d, R=Rm[q, 0], R2=Rm[q, 1],
                                     ratio=erat[q], body=body[q], hhl=hhl[q], leg_atr=d * (c[t] - eop[q]) / a[t],
                                     htf_bos=d * hb[t], htf_ema=d * he[t], hour=dts.iat[t].hour, atr_pct=atr_pct[t],
                                     kl_dist=kd, kl_type=kt, **sides(d, t, c[t])))
        # ---- sweep ----
        st_, sd_, sx_ = sweeps(h, l, c, a, typ, pr, cf)
        if len(st_):
            stop = sx_ - sd_ * 0.1 * a[st_]
            Rs = fixed_r(h, l, c, st_, sd_.astype(np.int64), c[st_], stop, tps, sp, 30)
            for q in range(len(st_)):
                t = st_[q]; d = int(sd_[q]); kd, kt = kl_at(t, sx_[q])
                rows.append(dict(kl_dist=kd, kl_type=kt, **sides(d, t, c[t])) |
dict(sym=sym, tf=tf, swing=name, brk="sweep", fam="sweep", t=dts.iat[t], fill=dts.iat[t],
                                 dir=d, R=Rs[q, 0], R2=Rs[q, 1], ratio=np.nan, body=d * (c[t] - o[t]) / a[t],
                                 hhl=0, leg_atr=np.nan, htf_bos=d * hb[t], htf_ema=d * he[t],
                                 hour=dts.iat[t].hour, atr_pct=atr_pct[t]))
    return rows


if __name__ == "__main__":
    tf = sys.argv[1]; root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pairs = sys.argv[2].split(",") if len(sys.argv) > 2 else bp.FX
    if tf == "M15":                       # Speicher: weniger Varianten, pro Paar schreiben
        SWINGS[:] = [("ZZ", 2.0), ("ZZ", 3.0), ("FR", 8)]
    parts = []
    for s in pairs:
        t0 = time.time(); r = pd.DataFrame(run_pair(s, tf, root))
        for col in ["sym", "tf", "swing", "brk", "fam", "kl_type"]:
            if col in r: r[col] = r[col].astype("category")
        parts.append(r)
        print(f"{tf} {s}: {len(r)} Trades/Events, {time.time() - t0:.1f}s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(f"{root}/bos_research/results/ctx_{tf}.parquet", index=False)
