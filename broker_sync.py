"""
broker_sync.py – Trades automatisch von MT4/MT5-Konten holen (Pepperstone, FTMO,
Vantage, FundedNext …) über MetaApi, plus CSV-Import als Fallback.

Warum MetaApi und nicht direkt "Login-Daten in die App"?
  * Die offizielle MetaTrader5-Python-Bibliothek läuft nur unter Windows mit
    installiertem Terminal – nicht auf deinem Mac und nicht auf Streamlit Cloud.
  * MetaApi verbindet sich in der Cloud mit deinem MT-Konto. Du hinterlegst dort
    einmalig Login + Server + INVESTOR-Passwort (nur Lesezugriff, kann nicht traden).
  * In TACO Lab steht danach nur noch die MetaApi-Konto-ID – kein Passwort.
  * Ein Konto ist bei MetaApi kostenlos, weitere Konten kosten Gebühren.

Token in .streamlit/secrets.toml:
    [metaapi]
    token = "DEIN_METAAPI_TOKEN"
"""
from __future__ import annotations

import io
import time as _time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import journal_db as db

DISPLAY_TZ = "Europe/Berlin"
METAAPI_REGIONS = ["new-york", "london", "singapore", "vint-hill"]
PAGE_LIMIT = 1000

ORDER_TYPE_MAP = {
    "ORDER_TYPE_BUY": "Market Order",
    "ORDER_TYPE_SELL": "Market Order",
    "ORDER_TYPE_BUY_LIMIT": "Buy Limit",
    "ORDER_TYPE_SELL_LIMIT": "Sell Limit",
    "ORDER_TYPE_BUY_STOP": "Buy Stop",
    "ORDER_TYPE_SELL_STOP": "Sell Stop",
    "ORDER_TYPE_BUY_STOP_LIMIT": "Buy Stop",
    "ORDER_TYPE_SELL_STOP_LIMIT": "Sell Stop",
}
TRADE_DEAL_TYPES = {"DEAL_TYPE_BUY", "DEAL_TYPE_SELL"}


class SyncError(Exception):
    pass


PROVISIONING = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts"


def connect_mt_account(token: str, name: str, login: str, password: str, server: str,
                       platform: str = "mt5", transaction_id: str | None = None,
                       max_wait: int = 100) -> dict:
    """Legt das MT4/MT5-Konto bei MetaApi an (Login + Server + Passwort).
    Das Passwort geht nur an MetaApi und wird im Journal NICHT gespeichert.
    Rueckgabe: {"id", "region", "transaction_id"} – oder SyncError mit Hinweis."""
    tx = transaction_id or uuid.uuid4().hex
    body = {"name": name, "login": str(login).strip(), "password": password, "server": server.strip(),
            "platform": platform.lower(), "magic": 0, "type": "cloud-g2", "reliability": "regular"}
    waited = 0
    while True:
        try:
            r = requests.post(PROVISIONING, json=body, timeout=60,
                              headers={"auth-token": token, "transaction-id": tx,
                                       "Content-Type": "application/json", "Accept": "application/json"})
        except requests.RequestException as e:
            raise SyncError(f"Keine Verbindung zu MetaApi: {e}") from e
        if r.status_code in (200, 201):
            acc_id = r.json().get("id")
            info = account_info(token, acc_id)
            return {"id": acc_id, "region": info.get("region") or "new-york", "transaction_id": tx}
        if r.status_code == 202:
            wait = int(r.headers.get("Retry-After", "20") or 20)
            if waited + wait > max_wait:
                err = SyncError("MetaApi erkennt noch die Broker-Einstellungen. Bitte in ca. 1 Minute "
                                "nochmal auf 'Verbinden' klicken (die Eingaben bleiben stehen).")
                err.transaction_id = tx
                raise err
            _time.sleep(min(wait, 30))
            waited += min(wait, 30)
            continue
        if r.status_code == 401:
            raise SyncError("MetaApi-Token ungültig. Bitte den Token prüfen.")
        try:
            msg = r.json().get("message") or r.text
        except Exception:
            msg = r.text
        if "server" in msg.lower() and ("not found" in msg.lower() or "unable" in msg.lower()):
            msg += " → Servername genau so eintragen wie im MT5-Login-Fenster (z. B. 'FTMO-Server3')."
        if "password" in msg.lower() or "authoriz" in msg.lower():
            msg += " → Login oder Passwort prüfen."
        raise SyncError(f"Verbindung fehlgeschlagen ({r.status_code}): {msg[:400]}")


def account_info(token: str, account_id: str) -> dict:
    try:
        r = requests.get(f"{PROVISIONING}/{account_id}", headers={"auth-token": token}, timeout=30)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def remove_mt_account(token: str, account_id: str) -> None:
    """Loescht das Konto bei MetaApi (damit keine Kosten weiterlaufen)."""
    try:
        requests.delete(f"{PROVISIONING}/{account_id}", headers={"auth-token": token}, timeout=30)
    except Exception:
        pass


# --------------------------------------------------------------------------- MetaApi REST
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _get_paged(token: str, region: str, account_id: str, kind: str,
               start: datetime, end: datetime) -> list[dict]:
    base = f"https://mt-client-api-v1.{region}.agiliumtrade.ai"
    url = f"{base}/users/current/accounts/{account_id}/{kind}/time/{_iso(start)}/{_iso(end)}"
    out, offset = [], 0
    while True:
        try:
            r = requests.get(url, headers={"auth-token": token, "Accept": "application/json"},
                             params={"offset": offset, "limit": PAGE_LIMIT}, timeout=60)
        except requests.RequestException as e:
            raise SyncError(f"Keine Verbindung zu MetaApi: {e}") from e
        if r.status_code == 401:
            raise SyncError("MetaApi-Token ungültig oder abgelaufen.")
        if r.status_code == 404:
            raise SyncError("Konto nicht gefunden – Konto-ID oder Region prüfen.")
        if r.status_code >= 400:
            msg = r.text[:300]
            if "not connected" in msg.lower() or "deploy" in msg.lower():
                msg += " → Konto im MetaApi-Dashboard auf 'deployed' stellen und kurz warten."
            raise SyncError(f"MetaApi-Fehler {r.status_code}: {msg}")
        batch = r.json() or []
        out.extend(batch)
        if len(batch) < PAGE_LIMIT:
            return out
        offset += PAGE_LIMIT


def _parse_time(value: str | None, tz: str = DISPLAY_TZ) -> datetime | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.tz_convert(ZoneInfo(tz)).tz_localize(None).to_pydatetime()


def _f(x) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _vwap(deals: list[dict]) -> float | None:
    vol = sum(_f(d.get("volume")) for d in deals)
    if vol <= 0:
        return None
    return sum(_f(d.get("price")) * _f(d.get("volume")) for d in deals) / vol


# --------------------------------------------------------------------------- Deals -> Trades
def build_trades_from_deals(deals: list[dict], orders: list[dict] | None = None,
                            start_balance: float | None = None,
                            tz: str = DISPLAY_TZ) -> list[dict]:
    """Fasst MT-Deals pro Position zu einem Trade zusammen.

    * Teil-Schließungen (TP1/TP2/TP3) -> volumengewichteter Exit, PnL summiert
    * pnl_pct = Netto-PnL / Kontostand bei Eröffnung
    * Kontostand wird aus allen Deals rekonstruiert (Einzahlung = Balance-Deal).
      Fehlt die Ersteinzahlung im Zeitraum, wird der Startsaldo des Kontos genutzt.
    """
    deals = sorted(deals, key=lambda d: d.get("time") or "")
    orders_by_id = {str(o.get("id")): o for o in (orders or [])}

    first_trade_time = next((d.get("time") for d in deals if d.get("type") in TRADE_DEAL_TYPES), None)
    first_balance_time = next((d.get("time") for d in deals if d.get("type") not in TRADE_DEAL_TYPES), None)
    has_initial_deposit = bool(first_balance_time) and (
        not first_trade_time or first_balance_time <= first_trade_time)
    balance = 0.0 if has_initial_deposit else float(start_balance or 0)

    pos = defaultdict(lambda: {"in": [], "out": [], "pnl": 0.0, "bal": None})
    for d in deals:
        amount = _f(d.get("profit")) + _f(d.get("commission")) + _f(d.get("swap"))
        if d.get("type") not in TRADE_DEAL_TYPES:
            balance += amount           # Einzahlung, Auszahlung, Credit …
            continue
        pid = str(d.get("positionId") or d.get("id"))
        p = pos[pid]
        if p["bal"] is None:
            p["bal"] = balance
        if d.get("entryType") == "DEAL_ENTRY_IN":
            p["in"].append(d)
        else:
            p["out"].append(d)
        p["pnl"] += amount
        balance += amount

    result = []
    for pid, p in pos.items():
        if not p["in"]:
            continue                    # Position vor dem Sync-Zeitraum eröffnet
        first = p["in"][0]
        direction = "Long" if first.get("type") == "DEAL_TYPE_BUY" else "Short"
        vol_in = sum(_f(d.get("volume")) for d in p["in"])
        vol_out = sum(_f(d.get("volume")) for d in p["out"])
        closed = vol_out >= vol_in * 0.999 and vol_out > 0

        order = orders_by_id.get(str(first.get("orderId")))
        order_type = ORDER_TYPE_MAP.get((order or {}).get("type"))
        sl = first.get("stopLoss") or (order or {}).get("stopLoss")
        tp = first.get("takeProfit") or (order or {}).get("takeProfit")

        entry_price = _vwap(p["in"])
        exit_price = _vwap(p["out"]) if p["out"] else None
        pnl = round(p["pnl"], 2) if closed else None
        base = p["bal"] or 0
        pnl_pct = round(pnl / base * 100, 3) if (closed and base > 0) else None

        result.append({
            "external_id": pid,
            "symbol": first.get("symbol"),
            "direction": direction,
            "order_type": order_type,
            "entry_time": _parse_time(first.get("time"), tz),
            "exit_time": _parse_time(p["out"][-1].get("time"), tz) if closed else None,
            "entry_price": entry_price,
            "exit_price": exit_price if closed else None,
            "stop_loss": _f(sl) or None,
            "tp1": _f(tp) or None,
            "volume": round(vol_in, 2),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "r_multiple": db.compute_r(direction, entry_price, _f(sl) or None,
                                       exit_price) if closed else None,
        })
    return sorted(result, key=lambda t: t["entry_time"] or datetime.min)


def sync_account(account: dict, token: str) -> dict:
    if not account.get("metaapi_account_id"):
        raise SyncError("Für dieses Konto ist keine MetaApi-Konto-ID hinterlegt.")
    start = account.get("sync_from") or (datetime.now() - timedelta(days=730))
    if isinstance(start, pd.Timestamp):
        start = start.to_pydatetime()
    end = datetime.now(timezone.utc) + timedelta(days=1)
    region = account.get("region")
    if not region:
        region = account_info(token, account["metaapi_account_id"]).get("region") or "new-york"
    acc_id = account["metaapi_account_id"].strip()

    deals = _get_paged(token, region, acc_id, "history-deals", start, end)
    try:
        orders = _get_paged(token, region, acc_id, "history-orders", start, end)
    except SyncError:
        orders = []                     # Ordertyp ist "nice to have"

    built = build_trades_from_deals(deals, orders, account.get("start_balance"))
    stats = {"neu": 0, "aktualisiert": 0}
    for t in built:
        stats[db.upsert_broker_trade(int(account["id"]), t["external_id"], t)] += 1
    db.save_account({"last_sync": datetime.now()}, int(account["id"]))
    return stats


# --------------------------------------------------------------------------- CSV-Import
CSV_TEMPLATE_COLUMNS = ["Ticket", "Symbol", "Richtung", "Ordertyp", "Zeitfenster",
                        "Einstieg", "Ausstieg", "Entry-Preis", "Exit-Preis", "Stop Loss",
                        "Lots", "Gewinn", "Gewinn %", "Risiko %"]

_ALIASES = {
    "external_id": ["ticket", "position", "positionid", "id", "order"],
    "symbol": ["symbol", "instrument", "pair", "markt"],
    "direction": ["richtung", "direction", "type", "side", "typ"],
    "order_type": ["ordertyp", "order type", "ordertype"],
    "timeframe": ["zeitfenster", "timeframe", "tf"],
    "entry_time": ["einstieg", "open time", "opentime", "entry time", "open", "eröffnung"],
    "exit_time": ["ausstieg", "close time", "closetime", "exit time", "close", "schließung"],
    "entry_price": ["entry-preis", "open price", "entry price", "openprice", "einstiegskurs"],
    "exit_price": ["exit-preis", "close price", "exit price", "closeprice", "ausstiegskurs"],
    "stop_loss": ["stop loss", "sl", "s/l", "stoploss"],
    "volume": ["lots", "volume", "size", "volumen"],
    "pnl": ["gewinn", "profit", "pnl", "p/l", "net profit"],
    "pnl_pct": ["gewinn %", "profit %", "pnl %", "return %", "ergebnis %"],
    "risk_pct": ["risiko %", "risk %", "risk"],
}


def csv_template() -> bytes:
    example = pd.DataFrame([[
        "123456", "GBPJPY", "Long", "Buy Limit", "H1", "2026-10-21 09:15", "2026-10-22 16:40",
        191.250, 192.100, 190.900, 0.8, 452.10, 0.45, 0.5]], columns=CSV_TEMPLATE_COLUMNS)
    return example.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def read_csv_flexible(file) -> pd.DataFrame:
    raw = file.read() if hasattr(file, "read") else file
    for sep, dec in ((";", ","), (",", "."), ("\t", ".")):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=dec)
            if df.shape[1] >= 3:
                return df
        except Exception:
            continue
    raise SyncError("CSV konnte nicht gelesen werden (Trennzeichen ; , oder Tab erwartet).")


def _parse_local(v) -> datetime | None:
    """ISO (2026-10-21 09:15) oder deutsch (21.10.2026 09:15)."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    s = str(v).strip()
    ts = pd.to_datetime(s, dayfirst=not s[:4].isdigit(), errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()


def parse_trades_csv(df: pd.DataFrame) -> list[dict]:
    cols = {c.strip().lower(): c for c in df.columns}
    mapping = {}
    for field, aliases in _ALIASES.items():
        for a in aliases:
            if a in cols:
                mapping[field] = cols[a]
                break
    if "symbol" not in mapping or "entry_time" not in mapping:
        raise SyncError("Spalten 'Symbol' und 'Einstieg' (bzw. 'Open Time') werden mindestens benötigt.")

    out = []
    for i, r in df.iterrows():
        t = {f: r[c] for f, c in mapping.items()}
        d = str(t.get("direction", "")).strip().lower()
        t["direction"] = "Short" if d in ("short", "sell", "verkauf") else "Long"
        for k in ("entry_time", "exit_time"):
            t[k] = _parse_local(t.get(k))
        for k in ("entry_price", "exit_price", "stop_loss", "volume", "pnl", "pnl_pct", "risk_pct"):
            if k in t:
                v = pd.to_numeric(str(t[k]).replace(",", "."), errors="coerce")
                t[k] = None if pd.isna(v) else float(v)
        t["symbol"] = str(t["symbol"]).strip().upper()
        t["external_id"] = str(t.get("external_id") or f"csv-{t['symbol']}-{t['entry_time']}-{i}")
        t["r_multiple"] = db.compute_r(t["direction"], t.get("entry_price"),
                                       t.get("stop_loss"), t.get("exit_price"))
        out.append(t)
    return out


def import_csv_trades(parsed: list[dict], account_id: int | None) -> dict:
    stats = {"neu": 0, "aktualisiert": 0}
    for t in parsed:
        if account_id:
            stats[db.upsert_broker_trade(account_id, t["external_id"], t, source="csv")] += 1
        else:
            db.save_trade({**t, "source": "csv"})
            stats["neu"] += 1
    return stats
