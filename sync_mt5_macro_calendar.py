"""Synchronisiert den MT5-Wirtschaftskalender-Export (Export_MacroCalendar_MT5_CSV.mq5)
ins Repo, damit die Streamlit-Cloud-App die Actual-Werte lesen kann.

Quelle: <MT5 Common>/Files/TACO_macro_recent.csv + TACO_macro_history.csv (cp1252).
Ziel:   data/macro_calendar/mt5_recent.csv       (letzte 60 + naechste 14 Tage, klein)
        data/macro_calendar/mt5_history.csv.gz   (alles aelter als 60 Tage, max. 1x/Woche)
        data/macro_calendar/meta.json            (Zeitpunkt des letzten Exports)

Nur High/Medium-Impact plus KEEP_LOW_CODES (Low blaeht sonst das Repo auf).

Aufruf:  python sync_mt5_macro_calendar.py            -> nur Dateien schreiben
         python sync_mt5_macro_calendar.py --push     -> bei Aenderung commit + push
         python sync_mt5_macro_calendar.py --push --force-history
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "data" / "macro_calendar"
COMMON_FILES = (
    Path.home() / "Library/Application Support/net.metaquotes.wine.metatrader5"
    / "drive_c/users/user/AppData/Roaming/MetaQuotes/Terminal/Common/Files"
)
RECENT_DAYS = 60
HISTORY_MAX_AGE_DAYS = 7
KEEP_COLS = [
    "value_id", "event_id", "event_code", "currency", "country", "event", "importance",
    "unit", "time_utc", "period", "actual", "forecast", "previous", "revised_previous", "impact",
]


# Low-Impact-Releases, die trotzdem gebraucht werden: MetaQuotes stuft einige
# Kern-Kennzahlen je Land als "Low" ein (z.B. Arbeitslosenquote JP/CH/NZ, CPI y/y
# CA/CH/IT, Einzelhandel y/y) -- die App nutzt sie als aktuelle Quelle fuer die
# Makro-Indikatoren statt der teils seit Jahren eingefrorenen FRED/OECD-Reihen.
KEEP_LOW_CODES = {
    "consumer-price-index-yy", "cpi-yy", "national-consumer-price-index-yy",
    "unemployment-rate", "retail-sales-yy", "retail-sales-mm", "gdp-yy",
    "gross-domestic-product-yy", "jobs-to-applicants-ratio",
    "household-spending-mm", "household-spending-yy",
}


def load_export(name: str) -> pd.DataFrame:
    df = pd.read_csv(COMMON_FILES / name, encoding="cp1252")
    keep = df["importance"].isin(["High", "Medium"]) | df["event_code"].isin(KEEP_LOW_CODES)
    df = df[keep].copy()
    df["unit"] = df["unit"].str.replace("CALENDAR_UNIT_", "", regex=False).str.lower()
    df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y.%m.%d %H:%M")
    df["period"] = df["period"].astype(str).str.replace(".", "-", regex=False)
    df["event"] = df["event"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df[KEEP_COLS]


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--force-history", action="store_true")
    args = parser.parse_args()

    recent_src = COMMON_FILES / "TACO_macro_recent.csv"
    if not recent_src.exists():
        print(f"Kein Export gefunden: {recent_src} (laeuft der EA in MT5?)")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exported_at = datetime.fromtimestamp(recent_src.stat().st_mtime, tz=timezone.utc)
    cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=RECENT_DAYS))

    recent = load_export("TACO_macro_recent.csv")
    recent = recent[recent["time_utc"] >= cutoff].sort_values(["time_utc", "value_id"])
    recent.to_csv(OUT_DIR / "mt5_recent.csv", index=False, date_format="%Y-%m-%d %H:%M")

    hist_path = OUT_DIR / "mt5_history.csv.gz"
    hist_age_days = (
        (datetime.now().timestamp() - hist_path.stat().st_mtime) / 86400 if hist_path.exists() else None
    )
    if args.force_history or hist_age_days is None or hist_age_days > HISTORY_MAX_AGE_DAYS:
        history = load_export("TACO_macro_history.csv")
        history = history[history["time_utc"] < cutoff].sort_values(["time_utc", "value_id"])
        # mtime=0 -> byte-identische Datei bei gleichem Inhalt (keine Schein-Diffs in git)
        history.to_csv(hist_path, index=False, date_format="%Y-%m-%d %H:%M",
                       compression={"method": "gzip", "mtime": 0})
        print(f"Historie geschrieben: {len(history)} Zeilen")

    print(f"Recent geschrieben: {len(recent)} Zeilen, Export-Stand {exported_at:%Y-%m-%d %H:%M} UTC")

    data_changed = bool(git("status", "--porcelain", "--", "data/macro_calendar/mt5_recent.csv",
                            "data/macro_calendar/mt5_history.csv.gz").stdout.strip())
    if data_changed:
        (OUT_DIR / "meta.json").write_text(json.dumps({"exported_at_utc": exported_at.isoformat()}, indent=2))

    if not args.push:
        return 0
    if not data_changed:
        print("Keine Aenderung an den Daten -> kein Commit.")
        return 0
    git("add", "data/macro_calendar")
    msg = f"MT5 Makro-Kalender Sync {exported_at:%Y-%m-%d %H:%M} UTC"
    commit = git("commit", "-m", msg, "--", "data/macro_calendar")
    if commit.returncode != 0:
        print(commit.stdout, commit.stderr)
        return 1
    push = git("push", "origin", "main")
    if push.returncode != 0:
        # z.B. Remote hat neue Commits -> einmal rebasen und nochmal
        git("pull", "--rebase", "--autostash", "origin", "main")
        push = git("push", "origin", "main")
    print(push.stdout, push.stderr)
    return push.returncode


if __name__ == "__main__":
    sys.exit(main())
