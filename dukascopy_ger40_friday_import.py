"""
Dukascopy GER40 (DAX-CFD) Freitags-Import -> M15-Kerzen (UTC).

Laedt fuer jeden Freitag die Ticks von Do 22:00 UTC bis Fr 21:59 UTC
(= kompletter Freitag in Berliner Zeit, Sommer und Winter) direkt aus dem
Dukascopy-Datafeed und baut daraus M15-Kerzen (OHLC aus BID, wie MT5).
H1 wird in der App aus M15 resampled.

Warum ein eigenes Script statt dukascopy_intraday_import.py:
  - duka.normalize() setzt alle Ticks auf die Tagesstunde 0 (Stunde fehlt)
  - Index-CFDs sind bei Dukascopy mit Faktor 1000 skaliert, nicht 100000

Ausgabe: data/mt5_intraday/GER40_FRI_M15.csv
  datetime_utc,open,high,low,close,tick_volume,spread
Resumable: bereits vorhandene Freitage werden uebersprungen.

    python3 dukascopy_ger40_friday_import.py --start 2014-01-01
"""

import argparse
import concurrent.futures
import csv
import lzma
import os
import struct
import time
from datetime import date, datetime, timedelta, timezone

import requests

SYMBOL = "DEUIDXEUR"
POINT = 1000.0
URL = "https://www.dukascopy.com/datafeed/{symbol}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36"}
OUT = "data/mt5_intraday/GER40_FRI_M15.csv"
BUCKET = 900


def fetch_hour(session, hour_start: datetime, attempts: int = 6):
    """Liefert Liste (epoch_sec, ask, bid) fuer eine UTC-Stunde."""
    url = URL.format(symbol=SYMBOL, y=hour_start.year, m=hour_start.month - 1,
                     d=hour_start.day, h=hour_start.hour)
    for i in range(attempts):
        try:
            r = session.get(url, timeout=(10, 45))
            if r.status_code == 404:
                return []
            if r.status_code == 200:
                if not r.content:
                    return []
                raw = lzma.decompress(r.content)
                base = hour_start.timestamp()
                out = []
                for ms, ask, bid, _av, _bv in struct.iter_unpack(">3I2f", raw):
                    out.append((base + ms / 1000.0, ask / POINT, bid / POINT))
                return out
        except (requests.exceptions.RequestException, lzma.LZMAError):
            pass
        time.sleep(min(2.0 * (i + 1), 15.0))
    print(f"  ! Stunde nicht ladbar: {url}", flush=True)
    return None  # Freitag wird dann ausgelassen und beim naechsten Lauf erneut versucht


def friday_hours(fri: date):
    thu = datetime(fri.year, fri.month, fri.day, tzinfo=timezone.utc) - timedelta(days=1)
    return [thu + timedelta(hours=h) for h in (22, 23)] + \
           [datetime(fri.year, fri.month, fri.day, h, tzinfo=timezone.utc) for h in range(22)]


def to_buckets(ticks):
    buckets = {}
    for ts, ask, bid in ticks:
        key = int(ts) - int(ts) % BUCKET
        b = buckets.get(key)
        if b is None:
            buckets[key] = [bid, bid, bid, bid, 1, ask - bid]
        else:
            if bid > b[1]:
                b[1] = bid
            if bid < b[2]:
                b[2] = bid
            b[3] = bid
            b[4] += 1
            b[5] += ask - bid
    return buckets


def existing_fridays():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            next(f, None)
            for line in f:
                dt = datetime.strptime(line[:16], "%Y-%m-%d %H:%M")
                # Kerzen ab Do 22:00 UTC gehoeren zum folgenden Freitag
                d = (dt + timedelta(hours=2)).date()
                done.add(d)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--batch", type=int, default=6, help="Freitage parallel")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    fri = start + timedelta(days=(4 - start.weekday()) % 7)
    done = existing_fridays()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new_file = not os.path.exists(OUT)

    session = requests.Session()
    session.headers.update(HEADERS)
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)

    t0, n = time.time(), 0
    with open(OUT, "a", newline="") as f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        w = csv.writer(f)
        if new_file:
            w.writerow(["datetime_utc", "open", "high", "low", "close", "tick_volume", "spread"])
        todo, skipped = [], []
        while fri <= end:
            if fri not in done:
                todo.append(fri)
            fri += timedelta(days=7)
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            futs = {d: [pool.submit(fetch_hour, session, hs) for hs in friday_hours(d)] for d in batch}
            for d in batch:
                parts = [fu.result() for fu in futs[d]]
                if any(p is None for p in parts):
                    skipped.append(d)
                    continue
                ticks = [t for p in parts for t in p]
                buckets = to_buckets(ticks)
                for key in sorted(buckets):
                    o, h, l, c, cnt, sp = buckets[key]
                    w.writerow([datetime.fromtimestamp(key, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                                f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}", cnt, f"{sp / cnt:.2f}"])
                n += 1
            f.flush()
            print(f"{batch[-1]} – {n}/{len(todo)} Freitage in {time.time() - t0:.0f}s", flush=True)
    print(f"fertig: {n} neue Freitage, {time.time() - t0:.0f}s", flush=True)
    if skipped:
        print(f"Unvollstaendig (erneut starten zum Nachladen): {', '.join(map(str, skipped))}", flush=True)


if __name__ == "__main__":
    main()
