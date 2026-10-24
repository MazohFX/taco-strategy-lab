"""
Dukascopy Intraday Importer (korrigierte Version) -> M15 / H1 / H4

Behebt zwei Fehler des alten dukascopy_intraday_import.py:
 1. Falscher Preis-Divisor: duka.normalize teilt JPY-Kurse durch 100000
    statt 1000 -> GBPJPY erschien als 1.79 statt 179.
 2. Verlorene Stunde: normalize(day, tokens) kennt die Stunde der .bi5-
    Datei nicht -> Zeitstempel falsch, nur ~8 Kerzen pro Tag.
Hier werden die .bi5-Dateien selbst dekodiert (20-Byte-Records, big endian:
ms-Offset, Ask, Bid, AskVol, BidVol). Keine Abhaengigkeit von "duka".

Ausgabe: data/dukascopy_intraday/<SYMBOL>_<TF>.csv (UTC, Bid-OHLC)
Spalten: datetime,open,high,low,close,ticks,spread
Resumable ueber .checkpoint.json im Ausgabeordner.

Start (im Ordner ~/Claude, im normalen Mac-Terminal):
  pip3 install requests
  python3 bos_research/dukascopy_import_fixed.py --start 2015-01-01
"""
import argparse, concurrent.futures, csv, json, lzma, os, struct, time
from datetime import date, datetime, timedelta, timezone
import requests, requests.adapters

DEFAULT_SYMBOLS = ["GBPJPY", "EURJPY", "AUDJPY", "NZDJPY", "USDJPY",
                   "EURUSD", "GBPUSD", "AUDUSD"]
URL = "https://www.dukascopy.com/datafeed/{s}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TFS = {"M15": 900, "H1": 3600, "H4": 14400}


def point(sym):
    return 1e3 if sym.endswith("JPY") or sym.startswith("XA") else 1e5


def fetch(sess, sym, day, hour):
    url = URL.format(s=sym, y=day.year, m=day.month - 1, d=day.day, h=hour)
    for i in range(10):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return b""
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** i))
    print(f"WARNUNG: Stunde fehlt nach 10 Versuchen: {url}", flush=True)
    return b""


def decode(raw, sym, day, hour):
    if not raw:
        return []
    data = lzma.decompress(raw)
    base = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp()
    p = point(sym)
    out = []
    for ms, ask, bid, av, bv in struct.iter_unpack(">3I2f", data):
        out.append((base + ms / 1000.0, ask / p, bid / p))
    return out


def day_ticks(sess, sym, day, workers):
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        raws = list(ex.map(lambda h: fetch(sess, sym, day, h), range(24)))
    ticks = []
    for h, raw in enumerate(raws):
        ticks.extend(decode(raw, sym, day, h))
    return ticks


CANDLE_URL = "https://www.dukascopy.com/datafeed/{s}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def fetch_url(sess, url):
    for i in range(10):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return b""
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** i))
    print(f"WARNUNG: fehlt nach 10 Versuchen: {url}", flush=True)
    return b""


def day_minutes(sess, sym, day):
    """1-Minuten-Bid-Kerzen eines Tages: Records (sek-Offset, O, C, L, H, Volumen), big endian."""
    raw = fetch_url(sess, CANDLE_URL.format(s=sym, y=day.year, m=day.month - 1, d=day.day))
    if not raw:
        return []
    data = lzma.decompress(raw)
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    p = point(sym); out = []
    for sec, o, c, l, h, v in struct.iter_unpack(">5if", data):
        if v <= 0 and o == c == l == h:
            continue                          # leere Minute (kein Handel)
        out.append((base + sec, o / p, h / p, l / p, c / p, v))
    return out


def aggregate_minutes(mins, sec):
    b = {}
    for ts, o, h, l, c, v in mins:
        k = int(ts) - int(ts) % sec
        x = b.get(k)
        if x is None:
            b[k] = [o, h, l, c, v, 0.0]
        else:
            x[1] = max(x[1], h); x[2] = min(x[2], l); x[3] = c; x[4] += v
    return b


def aggregate(ticks, sec):
    b = {}
    for ts, ask, bid in ticks:
        k = int(ts) - int(ts) % sec
        c = b.get(k)
        if c is None:
            b[k] = [bid, bid, bid, bid, 1, ask - bid]
        else:
            c[1] = max(c[1], bid); c[2] = min(c[2], bid); c[3] = bid
            c[4] += 1; c[5] += ask - bid
    return b


def write_day(out, sym, ticks):
    dig = 3 if point(sym) == 1e3 else 5
    for tf, sec in TFS.items():
        path = os.path.join(out, f"{sym}_{tf}.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["datetime", "open", "high", "low", "close", "ticks", "spread"])
            for k, c in sorted(aggregate(ticks, sec).items()):
                w.writerow([datetime.fromtimestamp(k, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            *(f"{x:.{dig}f}" for x in c[:4]), c[4], f"{c[5]/c[4]:.{dig}f}"])


def write_day_minutes(out, sym, mins):
    dig = 3 if point(sym) == 1e3 else 5
    for tf, sec in TFS.items():
        path = os.path.join(out, f"{sym}_{tf}.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["datetime", "open", "high", "low", "close", "ticks", "spread"])
            for k, c in sorted(aggregate_minutes(mins, sec).items()):
                w.writerow([datetime.fromtimestamp(k, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            *(f"{x:.{dig}f}" for x in c[:4]), round(c[4], 2), ""])


def _cp_path(out, sym):
    return os.path.join(out, f".checkpoint_{sym}.json")


def _load_cp(out, sym):
    p = _cp_path(out, sym)
    if os.path.exists(p):
        return json.load(open(p)).get(sym)
    old = os.path.join(out, ".checkpoint.json")          # alte gemeinsame Datei
    if os.path.exists(old):
        return json.load(open(old)).get(sym)
    return None


def import_symbol(sym, a, end):
    sess = requests.Session(); sess.headers.update(HEADERS)
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=a.workers, pool_maxsize=a.workers))
    last = _load_cp(a.out, sym)
    d = date.fromisoformat(last) + timedelta(days=1) if last else date.fromisoformat(a.start)
    t0, n = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        while d <= end:
            days = []
            while d <= end and len(days) < a.batch:
                if d.weekday() != 5:
                    days.append(d)
                d += timedelta(days=1)
            if a.mode == "ticks":
                jobs = {(dd, h): ex.submit(fetch, sess, sym, dd, h) for dd in days for h in range(24)}
            else:
                jobs = {dd: ex.submit(day_minutes, sess, sym, dd) for dd in days}
            for dd in days:                                  # streng chronologisch schreiben
                if a.mode == "ticks":
                    ticks = []
                    for h in range(24):
                        ticks.extend(decode(jobs[(dd, h)].result(), sym, dd, h))
                    write_day(a.out, sym, ticks)
                else:
                    write_day_minutes(a.out, sym, jobs[dd].result())
                json.dump({sym: dd.isoformat()}, open(_cp_path(a.out, sym), "w"))
            n += len(days)
            if days:
                print(f"[{sym}] {days[-1]}  ({(time.time()-t0)/n:.2f}s/Tag)", flush=True)
    print(f"[{sym}] fertig bis {end}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--out", default="data/dukascopy_intraday")
    ap.add_argument("--parallel", type=int, default=4, help="Paare gleichzeitig")
    ap.add_argument("--workers", type=int, default=12, help="parallele Downloads je Paar")
    ap.add_argument("--batch", type=int, default=20, help="Tage pro Paket je Paar")
    ap.add_argument("--mode", choices=["candles", "ticks"], default="candles",
                    help="candles = 1 Datei/Tag (schnell), ticks = 24 Dateien/Tag")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    end = date.fromisoformat(a.end)
    syms = a.symbols.split(",")
    with concurrent.futures.ThreadPoolExecutor(a.parallel) as px:
        for f in concurrent.futures.as_completed([px.submit(import_symbol, s, a, end) for s in syms]):
            f.result()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen. Fortschritt ist gespeichert - zum Fortsetzen einfach neu starten.", flush=True)
        os._exit(1)   # sofort beenden, ohne auf offene Downloads zu warten
