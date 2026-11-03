"""
Saisonfenster aus Tobis Excel: Einstieg ohne Bruch vs. mit BOS / POC / Value Area / Opens.
Gleiches Fenster, gleicher Ausstieg (Close am letzten Fenstertag), gleicher Stop (1,5 x Tages-ATR ab Einstieg).
Kennzahl in % (richtungsbereinigt, nach Spread) und in R (Stop = 1R).
Daten: Dukascopy H1/H4/M15, 2015 bis Okt. 2026.
"""
import os, sys, openpyxl
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp, bos_grid as g
from va_breakout import value_area

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Seasonality_Excel.xlsx"


def windows():
    wb = openpyxl.load_workbook(XL, data_only=True)
    out = []
    main = {"GBPNZD Januar bis Februar": ("GBPNZD", 1, "12.01", "31.01"),
            "USDCAD Februar bis März": ("USDCAD", 1, "16.02", "10.03"),
            "NZDCAD März bis April": ("NZDCAD", -1, "12.03", "08.04"),
            "EURUSD Mai bis Juni": ("EURUSD", 1, "13.05", "05.06"),
            "EURNZD Juli bis August": ("EURNZD", 1, "13.07", "20.08"),
            "NZDUSD September bis Oktober": ("NZDUSD", -1, "20.09", "12.10"),
            "EURUSD September ": ("EURUSD", -1, "12.09", "27.09"),
            "EURUSD November bis Dezember": ("EURUSD", 1, "23.11", "30.12")}
    for k, (p, d, s, e) in main.items():
        out.append(dict(src="Hauptfenster", name=k.strip(), pair=p, dir=d, s=s, e=e))
    seen = set()
    for r in wb["Tabelle1"].iter_rows(min_row=2, values_only=True):
        if not r[0]: continue
        a = r[0].split(); key = (a[0], a[1], a[3])
        if key in seen: continue
        seen.add(key); s, e = a[3].split("-")
        out.append(dict(src="Tabelle1", name=f"{a[0]} {a[1]} {a[3]}", pair=a[0], dir=1 if a[1] == "LONG" else -1, s=s, e=e))
    return pd.DataFrame(out)


def prep(sym, tf):
    df = bp.load(sym, tf, ROOT)
    h, l, c = (df[x].values.astype(np.float64) for x in ["high", "low", "close"])
    a = bp.atr(df).astype(np.float64)
    ev = g.find_events(h, l, c, a, *g.zigzag(h, l, a, 2.0), False)   # ZZ2, Close-Bruch
    return df, ev


def run():
    W = windows(); rows = []
    cache = {}
    for _, w in W.iterrows():
        p, d = w.pair, w.dir
        if p not in cache:
            h1 = bp.load(p, "H1", ROOT)
            if h1 is None: continue
            d1 = h1.set_index("dt").resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - d1.close.shift()).abs(), (d1.low - d1.close.shift()).abs()))
            d1["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean().shift()
            hh, ll, vv = (h1[x].values.astype(np.float64) for x in ["high", "low", "vol"])
            ms = h1.groupby(h1.dt.dt.to_period("M")).cumcount().values.astype(np.int64)
            vah, val, poc = value_area(hh, ll, vv, ms, 24 * 3)
            h1 = h1.assign(vah=vah, val=val, poc=poc,
                           wo=h1.groupby(h1.dt.dt.to_period("W-SUN")).open.transform("first"),
                           mo=h1.groupby(h1.dt.dt.to_period("M")).open.transform("first"))
            cache[p] = dict(h1=h1, d1=d1, bos={tf: prep(p, tf) for tf in ["M15", "H1", "H4"]})
        C = cache[p]; h1 = C["h1"]; sp = bp.spread(p)
        for y in range(2015, 2027):
            a = pd.Timestamp(f"{y}-{w.s[3:5]}-{w.s[:2]}"); b = pd.Timestamp(f"{y}-{w.e[3:5]}-{w.e[:2]}") + pd.Timedelta(days=1)
            if b <= a: b = b.replace(year=y + 1)
            x = h1[(h1.dt >= a) & (h1.dt < b)].reset_index(drop=True)
            if len(x) < 24 or b > h1.dt.iloc[-1]: continue
            atr = C["d1"].atr.asof(a)
            exit_px = x.close.iloc[-1]; mid = a + (b - a) / 2

            def result(i, px, name, stop=None):
                stop = px - d * 1.5 * atr if stop is None else stop
                seg = x.iloc[i:]
                hit = (seg.low <= stop) if d == 1 else (seg.high >= stop)
                ret = d * (exit_px - px) - sp
                ret_sl = (-abs(px - stop) - sp) if hit.any() else ret
                rows.append(dict(src=w.src, window=w["name"], pair=p, year=y, entry=name, filled=1,
                                 pct=ret / px, R=ret_sl / abs(px - stop), pct_sl=ret_sl / px))

            def nofill(name):
                rows.append(dict(src=w.src, window=w["name"], pair=p, year=y, entry=name, filled=0, pct=0.0, R=0.0, pct_sl=0.0))

            # A: Signal / Tages-Open am ersten Fenstertag
            result(0, x.open.iloc[0], "A Tages-Open Fensterstart")
            # B/C: erster BOS in Saisonrichtung (erste Fensterhaelfte)
            for tf in ["M15", "H1", "H4"]:
                df, (et, ed, eo, eop, elv, erat) = C["bos"][tf]
                t_ev = df.dt.values[et]
                m = (ed == d) & (t_ev >= np.datetime64(a)) & (t_ev < np.datetime64(mid))
                if not m.any():
                    nofill(f"B BOS {tf} Markt"); nofill(f"C BOS {tf} + 0.618"); continue
                q = np.flatnonzero(m)[0]; tb = pd.Timestamp(t_ev[q]) + pd.Timedelta(minutes={"M15": 15, "H1": 60, "H4": 240}[tf])
                i = int(np.searchsorted(x.dt.values, np.datetime64(tb)))
                if i >= len(x): nofill(f"B BOS {tf} Markt"); nofill(f"C BOS {tf} + 0.618"); continue
                origin = eop[q]
                result(i, x.open.iloc[i], f"B BOS {tf} Markt", stop=origin if d * (x.open.iloc[i] - origin) > 0 else None)
                E = df.high.values[eo[q]:et[q] + 1].max() if d == 1 else df.low.values[eo[q]:et[q] + 1].min()
                lvl = E - d * 0.618 * abs(E - origin)
                f = np.flatnonzero(((x.low <= lvl) if d == 1 else (x.high >= lvl)).values[i:])
                if len(f): result(i + f[0], lvl, f"C BOS {tf} + 0.618", stop=origin)
                else: nofill(f"C BOS {tf} + 0.618")
            # D: Limit an Levels (nur wenn Level auf der guenstigen Seite liegt)
            p0 = x.open.iloc[0]
            for nm, lv in [("D Limit Monats-POC", x.poc.iloc[0]), ("D Limit VAL/VAH (Long VAL, Short VAH)", x.val.iloc[0] if d == 1 else x.vah.iloc[0]),
                           ("D Limit Wochen-Open", x.wo.iloc[0]), ("D Limit Monats-Open", x.mo.iloc[0]),
                           ("D Limit Vortages-Tief/Hoch", (C["d1"].low if d == 1 else C["d1"].high).shift().asof(a))]:
                if np.isnan(lv) or d * (p0 - lv) <= 0:
                    result(0, p0, nm)                       # Level schon unterschritten -> wie A
                    continue
                f = np.flatnonzero(((x.low <= lv) if d == 1 else (x.high >= lv)).values)
                if len(f): result(f[0], lv, nm)
                else: nofill(nm)
            # E: Tages-Open als Retest: ab Tag 2 Limit am jeweiligen Tages-Open, sobald der Tag in Saisonrichtung eroeffnet
            days = x.groupby(x.dt.dt.date)
            done = False
            for k, (day, gday) in enumerate(days):
                if k == 0: continue
                o = gday.open.iloc[0]; prev = x[x.dt.dt.date < day].close.iloc[-1]
                touch = np.flatnonzero(((gday.low <= o) if d == 1 else (gday.high >= o)).values[1:])
                if len(touch) and d * (gday.close.iloc[-1] - o) >= -1e9:
                    i = gday.index[1 + touch[0]]; result(i, o, "E Retest Tages-Open (ab Tag 2)"); done = True; break
            if not done: nofill("E Retest Tages-Open (ab Tag 2)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = run()
    r.to_pickle(f"{ROOT}/bos_research/results/season_entries.pkl")
    print(len(r))
