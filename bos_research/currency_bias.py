"""
Wochen-Bias pro Waehrung als Filter vor dem Setup ("Bias fuer die Woche").

Waehrungsstaerke (rein preisbasiert, ohne Lookahead):
  Fuer jede der 8 Waehrungen = Mittel der log-Renditen aller 7 Paare, in denen sie vorkommt
  (Vorzeichen + als Basis, - als Quote), gemessen bis zum Schluss der VORWOCHE.
  Lookbacks: 1 Woche, 4 Wochen, 13 Wochen.
Risk-Regime (VIX, Vortag): risk-on = VIX < 20, risk-off = VIX >= 20; Risk-Beta wie im Pine-Indikator.

Filter je Trade (Richtung d auf BASE/QUOTE):
  stark_gg_schwach : Basis staerker als Quote (Long) bzw. schwaecher (Short)
  top2_vs_bottom2  : Basis unter den 2 staerksten UND Quote unter den 2 schwaechsten (bzw. umgekehrt)
  contrarian       : genau umgekehrt (Mean Reversion der Wochenstaerke)
  vix_beta         : risk-on -> nur in Richtung hoeherer Risk-Beta, risk-off -> nur Richtung Safe Haven
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bos_pipeline as bp, bos_context as bc

CUR = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
BETA = {"AUD": 1.0, "NZD": 1.0, "CAD": 0.5, "GBP": 0.3, "EUR": 0.0, "USD": -0.5, "CHF": -0.7, "JPY": -1.0}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weekly_strength():
    closes = {}
    for s in bp.FX:
        d = bc.load_any(s, "D1", ROOT)
        closes[s] = d.set_index("dt").close
    px = pd.DataFrame(closes).sort_index()
    wk = px.resample("W-FRI").last()                       # Freitags-Schluss
    out = {}
    for lb in (1, 4, 13):
        r = np.log(wk / wk.shift(lb))
        sc = pd.DataFrame(0.0, index=r.index, columns=CUR)
        for s in bp.FX:
            b, q = s[:3], s[3:]
            sc[b] += r[s] / 7; sc[q] -= r[s] / 7
        out[lb] = sc
    return out          # Index = Freitag der Messung; gilt ab der Folgewoche


def tag(df, strength, vix):
    df = df.copy()
    off = (df.t.dt.weekday - 4) % 7
    off = off.where(off != 0, 7)                            # Freitag selbst -> Vorwochen-Freitag
    df["week"] = (df.t - pd.to_timedelta(off, unit="D")).dt.normalize()
    base = df.sym.astype(str).str[:3]; quote = df.sym.astype(str).str[3:]
    for lb, sc in strength.items():
        sc = sc.copy(); sc.index = sc.index.normalize()
        rk = sc.rank(axis=1, ascending=False)               # 1 = staerkste
        sb = sc.reindex(df.week).values; rb = rk.reindex(df.week).values
        ci = {c: i for i, c in enumerate(CUR)}
        bi = base.map(ci).values; qi = quote.map(ci).values
        n = np.arange(len(df))
        diff = sb[n, bi] - sb[n, qi]
        df[f"st{lb}"] = np.sign(diff) * df.dir.values                 # +1 = Trade in Richtung der Staerke
        top = (rb[n, bi] <= 2) & (rb[n, qi] >= 7); bot = (rb[n, bi] >= 7) & (rb[n, qi] <= 2)
        df[f"tb{lb}"] = np.where(df.dir.values == 1, np.where(top, 1, np.where(bot, -1, 0)),
                                 np.where(bot, 1, np.where(top, -1, 0)))
    v = vix.copy(); v.index = v.index + pd.Timedelta(days=1)
    vv = v.reindex(df.t.dt.normalize(), method="ffill").values
    rb_ = (base.map(BETA) - quote.map(BETA)).values
    want = np.where(vv < 20, np.sign(rb_), -np.sign(rb_))
    df["vix"] = np.where(rb_ == 0, 0, want * df.dir.values)
    df["split"] = np.where(df.t < "2021-01-01", "IS", np.where(df.t < "2024-01-01", "OOS", "HO"))
    return df


def st(x):
    r = x.R
    return pd.Series({"n": len(r), "WR": (r > 0.05).mean(), "PF": r[r > 0].sum() / max(1e-9, -r[r < 0].sum()),
                      "ER": r.mean()})


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    S = weekly_strength()
    vix = pd.read_csv(f"{ROOT}/data/vix/VIX.csv", parse_dates=["date"]).set_index("date").close
    rows = []
    for tf in ["D1", "H4", "H1", "M15"]:
        sets = []
        if tf != "D1":
            c = pd.read_parquet(f"{ROOT}/bos_research/results/ctx_{tf}.parquet",
                                columns=["sym", "swing", "brk", "fam", "t", "dir", "R", "R2", "hour"])
            c = c[~c.hour.isin([20, 21, 22, 23])]
            sets.append(("Fib-Setup (Swing-BOS)", c[(c.fam == "fib") & (c.brk == "close")]))
            sets.append(("Momentum 2R", c[c.fam == "mom"].assign(R=lambda x: x.R2)))
            sets.append(("Liquidity Sweep 2R", c[c.fam == "sweep"].assign(R=lambda x: x.R2)))
            v = pd.read_parquet(f"{ROOT}/bos_research/results/va_{tf}.parquet")
            v = v[(v.entry == 0.618) & ~v.hour.isin([20, 21, 22, 23])]
            sets.append(("Value-Area-Ausbruch (gianfx)", v))
        for name, df in sets:
            df = tag(df.reset_index(drop=True), S, vix)
            for flt, m in [("ohne Filter", df.R.notna()),
                           ("Staerke 1W", df.st1 == 1), ("Staerke 4W", df.st4 == 1), ("Staerke 13W", df.st13 == 1),
                           ("Top2 vs Bottom2 1W", df.tb1 == 1), ("Top2 vs Bottom2 4W", df.tb4 == 1),
                           ("Top2 vs Bottom2 13W", df.tb13 == 1),
                           ("Contrarian 1W", df.st1 == -1), ("Contrarian 4W", df.st4 == -1),
                           ("Contrarian Top2/Bottom2 1W", df.tb1 == -1),
                           ("VIX Risk-Beta", df.vix == 1),
                           ("Staerke 4W + 13W + VIX", (df.st4 == 1) & (df.st13 == 1) & (df.vix == 1))]:
                x = df[m]
                rec = dict(tf=tf, setup=name, filter=flt)
                for s in ["IS", "OOS", "HO"]:
                    r = x.R[x.split == s].dropna()
                    rec[f"n_{s}"] = len(r); rec[f"WR_{s}"] = (r > 0.05).mean()
                    rec[f"PF_{s}"] = r[r > 0].sum() / max(1e-9, -r[r < 0].sum()); rec[f"ER_{s}"] = r.mean()
                rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(f"{ROOT}/bos_research/results/currency_bias.csv", index=False)
    show = res[["tf", "setup", "filter", "n_IS", "PF_IS", "ER_IS", "n_OOS", "PF_OOS", "ER_OOS", "n_HO", "PF_HO", "ER_HO", "WR_HO"]]
    print(show.round(2).to_string())
