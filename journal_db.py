"""
journal_db.py – Persistenz für das TACO-Lab-Trading-Journal.

Lokal:   SQLite-Datei  data/trading_journal.db  (kein Setup nötig)
Online:  Postgres (z. B. Supabase, kostenloser Plan reicht) – in .streamlit/secrets.toml:

    [journal]
    db_url = "postgresql+psycopg2://USER:PASSWORT@HOST:5432/postgres"

Warum Postgres online? Streamlit Community Cloud löscht das Dateisystem bei jedem
Neustart/Redeploy – eine SQLite-Datei (inkl. Screenshots) wäre dann weg.
Screenshots liegen komprimiert als Binärdaten direkt in der DB, damit alles an einem Ort ist.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import (Column, DateTime, Float, Integer, LargeBinary, MetaData,
                        String, Table, Text, and_, create_engine, delete, func,
                        insert, select, update)

metadata = MetaData()

accounts = Table(
    "tj_accounts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(120), nullable=False),
    Column("broker", String(60)),
    Column("platform", String(20)),
    Column("metaapi_account_id", String(80)),
    Column("region", String(40)),
    Column("start_balance", Float),
    Column("currency", String(10)),
    Column("sync_from", DateTime),
    Column("last_sync", DateTime),
)

trades = Table(
    "tj_trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("account_id", Integer, index=True),
    Column("source", String(20)),            # manual | metaapi | csv
    Column("external_id", String(80), index=True),  # Positions-ID beim Broker
    Column("symbol", String(30)),
    Column("direction", String(10)),         # Long | Short
    Column("order_type", String(30)),        # Buy Limit | Sell Limit | Market Order | ...
    Column("timeframe", String(10)),         # H4 | H1 | M15
    Column("entry_time", DateTime),
    Column("exit_time", DateTime),
    Column("entry_price", Float),
    Column("exit_price", Float),
    Column("stop_loss", Float),
    Column("tp1", Float),
    Column("tp2", Float),
    Column("tp3", Float),
    Column("volume", Float),
    Column("pnl", Float),                    # in Kontowährung (netto inkl. Kommission/Swap)
    Column("pnl_pct", Float),                # in % vom Kontostand
    Column("risk_pct", Float),
    Column("r_multiple", Float),
    Column("confluences", Text),             # JSON-Liste
    Column("checklist", Text),               # JSON-Dict {Regel: bool}
    Column("plan_followed", String(20)),     # Ja | Teilweise | Nein
    Column("rating", Integer),               # 0–5 Sterne
    Column("notes", Text),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
)

screenshots = Table(
    "tj_screenshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trade_id", Integer, index=True, nullable=False),
    Column("filename", String(200)),
    Column("mime", String(40)),
    Column("data", LargeBinary),
    Column("created_at", DateTime),
)

settings_tbl = Table(
    "tj_settings", metadata,
    Column("key", String(60), primary_key=True),
    Column("value", Text),
)

# Felder, die beim Broker-Sync überschrieben werden dürfen – alles andere
# (Kriterien, Checkliste, Notizen, Sterne …) bleibt beim erneuten Sync erhalten.
BROKER_FIELDS = (
    "symbol", "direction", "order_type", "entry_time", "exit_time", "entry_price",
    "exit_price", "stop_loss", "tp1", "volume", "pnl", "pnl_pct", "r_multiple",
)


# --------------------------------------------------------------------------- Engine
def _secret(section: str, key: str) -> Any:
    try:
        return st.secrets[section][key]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_engine():
    url = _secret("journal", "db_url")
    if not url:
        Path("data").mkdir(exist_ok=True)
        url = "sqlite:///data/trading_journal.db"
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def backend_name() -> str:
    return get_engine().dialect.name


# --------------------------------------------------------------------------- Helfer
def _clean(value: Any) -> Any:
    """NaN/NaT/numpy-Typen -> Python-Typen, damit SQLite & Postgres sie schlucken."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _row(data: dict, table: Table) -> dict:
    out = {}
    for k, v in data.items():
        if k not in table.c or k == "id":
            continue
        if k in ("confluences", "checklist") and not isinstance(v, (str, type(None))):
            v = json.dumps(v, ensure_ascii=False)
        out[k] = _clean(v)
    return out


def _loads(v: Any, default):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def compute_r(direction: str | None, entry: float | None, sl: float | None,
              exit_: float | None) -> float | None:
    """R-Multiple rein aus Preisen – unabhängig von Pip-Werten/Kontowährung."""
    if None in (direction, entry, sl, exit_):
        return None
    try:
        risk = (entry - sl) if direction == "Long" else (sl - entry)
        if risk <= 0:
            return None
        move = (exit_ - entry) if direction == "Long" else (entry - exit_)
        return round(move / risk, 2)
    except Exception:
        return None


# --------------------------------------------------------------------------- Konten
def list_accounts() -> pd.DataFrame:
    with get_engine().connect() as c:
        df = pd.read_sql(select(accounts).order_by(accounts.c.name), c)
    return df


def get_account(account_id: int) -> dict | None:
    with get_engine().connect() as c:
        r = c.execute(select(accounts).where(accounts.c.id == account_id)).mappings().first()
    return dict(r) if r else None


def save_account(data: dict, account_id: int | None = None) -> int:
    row = _row(data, accounts)
    with get_engine().begin() as c:
        if account_id:
            c.execute(update(accounts).where(accounts.c.id == account_id).values(**row))
            return account_id
        res = c.execute(insert(accounts).values(**row))
        return int(res.inserted_primary_key[0])


def delete_account(account_id: int) -> None:
    """Löscht nur das Konto. Trades bleiben erhalten (ohne Kontozuordnung)."""
    with get_engine().begin() as c:
        c.execute(update(trades).where(trades.c.account_id == account_id).values(account_id=None))
        c.execute(delete(accounts).where(accounts.c.id == account_id))


# --------------------------------------------------------------------------- Trades
TRADE_COLUMNS = [c.name for c in trades.c]


def load_trades() -> pd.DataFrame:
    with get_engine().connect() as c:
        df = pd.read_sql(select(trades), c)
    if df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    for col in ("entry_time", "exit_time", "created_at", "updated_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["confluences"] = df["confluences"].apply(lambda v: _loads(v, []))
    df["checklist"] = df["checklist"].apply(lambda v: _loads(v, {}))
    return df


def get_trade(trade_id: int) -> dict | None:
    with get_engine().connect() as c:
        r = c.execute(select(trades).where(trades.c.id == trade_id)).mappings().first()
    if not r:
        return None
    d = dict(r)
    d["confluences"] = _loads(d.get("confluences"), [])
    d["checklist"] = _loads(d.get("checklist"), {})
    return d


def save_trade(data: dict, trade_id: int | None = None) -> int:
    row = _row(data, trades)
    row["updated_at"] = datetime.now()
    with get_engine().begin() as c:
        if trade_id:
            c.execute(update(trades).where(trades.c.id == trade_id).values(**row))
            return trade_id
        row.setdefault("created_at", datetime.now())
        row.setdefault("source", "manual")
        res = c.execute(insert(trades).values(**row))
        return int(res.inserted_primary_key[0])


def delete_trade(trade_id: int) -> None:
    with get_engine().begin() as c:
        c.execute(delete(screenshots).where(screenshots.c.trade_id == trade_id))
        c.execute(delete(trades).where(trades.c.id == trade_id))


def upsert_broker_trade(account_id: int, external_id: str, fields: dict,
                        source: str = "metaapi") -> str:
    """Legt einen Broker-Trade an oder aktualisiert NUR die Broker-Felder.
    Rückgabe: 'neu' | 'aktualisiert'."""
    broker_part = {k: fields.get(k) for k in BROKER_FIELDS if k in fields}
    with get_engine().begin() as c:
        existing = c.execute(
            select(trades.c.id).where(and_(trades.c.account_id == account_id,
                                           trades.c.external_id == str(external_id)))
        ).first()
        if existing:
            row = _row(broker_part, trades)
            row["updated_at"] = datetime.now()
            c.execute(update(trades).where(trades.c.id == existing[0]).values(**row))
            return "aktualisiert"
        row = _row({**fields, "account_id": account_id, "external_id": str(external_id),
                    "source": source}, trades)
        row["created_at"] = row["updated_at"] = datetime.now()
        c.execute(insert(trades).values(**row))
        return "neu"


# --------------------------------------------------------------------------- Screenshots
def add_screenshot(trade_id: int, filename: str, mime: str, data: bytes) -> None:
    with get_engine().begin() as c:
        c.execute(insert(screenshots).values(trade_id=trade_id, filename=filename, mime=mime,
                                             data=data, created_at=datetime.now()))


def list_screenshots(trade_id: int) -> list[dict]:
    with get_engine().connect() as c:
        rows = c.execute(select(screenshots).where(screenshots.c.trade_id == trade_id)
                         .order_by(screenshots.c.id)).mappings().all()
    return [dict(r) for r in rows]


def delete_screenshot(shot_id: int) -> None:
    with get_engine().begin() as c:
        c.execute(delete(screenshots).where(screenshots.c.id == shot_id))


def screenshot_counts() -> dict[int, int]:
    with get_engine().connect() as c:
        rows = c.execute(select(screenshots.c.trade_id, func.count())
                         .group_by(screenshots.c.trade_id)).all()
    return {int(t): int(n) for t, n in rows}


# --------------------------------------------------------------------------- Einstellungen
def get_setting(key: str) -> str | None:
    with get_engine().connect() as c:
        r = c.execute(select(settings_tbl.c.value).where(settings_tbl.c.key == key)).first()
    return r[0] if r else None


def set_setting(key: str, value: str | None) -> None:
    with get_engine().begin() as c:
        c.execute(delete(settings_tbl).where(settings_tbl.c.key == key))
        if value:
            c.execute(insert(settings_tbl).values(key=key, value=value))
