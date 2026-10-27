"""
trading_journal.py – Trading-Journal für TACO Lab (Stil TradeZella).

Einbinden in taco_web_app.py:
    from trading_journal import render_trading_journal
    ...
    elif seite == "📓 Trading-Journal":
        render_trading_journal()

Kriterien und Checkliste kannst du oben in CONFLUENCES / CHECKLIST anpassen –
bestehende Trades behalten ihre gespeicherten Häkchen.
"""
from __future__ import annotations

import calendar
import io
from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st
from PIL import Image

import broker_sync as bs
import journal_db as db

# =========================================================================== Konfiguration
CONFLUENCES = [
    "Monats-Open", "3-Monats-Open", "Jahres-Open", "Wochen-Open", "Tages-Open",
    "Volumen-Area / POC", "Weekly Area", "Break of Structure", "Fib 0.618", "Fib 0.786",
]
CHECKLIST = [
    "BOS klar bestätigt",
    "Entry in der Zone 0.618–0.786",
    "Mind. ein Open (Monat / 3M / Jahr) in der Zone",
    "POC (FRVP ab Monats-Open) in der Zone",
    "Stop Loss nach Plan gesetzt",
    "Risiko ≤ 1 %",
    "TPs nach Plan (-0.27 / -0.62 / -1)",
    "SL auf Break-even nach Bruch des Fib-0-Levels",
    "Keine spontane Anpassung während des Trades",
]
TIMEFRAMES = ["H4", "H1", "M15"]
ORDER_TYPES = ["Buy Limit", "Sell Limit", "Market Order", "Buy Stop", "Sell Stop"]
DIRECTIONS = ["Long", "Short"]
PLAN_OPTIONS = ["Ja", "Teilweise", "Nein"]
BROKERS = ["Pepperstone", "FTMO", "Vantage", "FundedNext", "Sonstiger"]
PLATFORMS = ["MT5", "MT4"]
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
          "September", "Oktober", "November", "Dezember"]
GREEN, RED, GREY = "#16a34a", "#dc2626", "#6b7280"


# =========================================================================== Kleine Helfer
def _stars(n) -> str:
    n = 0 if n is None or pd.isna(n) else int(n)
    return "★" * n + "☆" * (5 - n)


def _fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "–"
    return f"{v:+.2f} %"


def _color(v) -> str:
    if v is None or pd.isna(v) or v == 0:
        return GREY
    return GREEN if v > 0 else RED


def _button(label: str, key: str, **kw) -> bool:
    """Volle Breite – kompatibel mit alten und neuen Streamlit-Versionen."""
    try:
        return st.button(label, key=key, width="stretch", **kw)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, **kw)


def _secret(section: str, key: str):
    try:
        return st.secrets[section][key]
    except Exception:
        return None


def _compress_image(uploaded) -> tuple[bytes, str]:
    img = Image.open(uploaded)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((1920, 1920))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _check_password() -> bool:
    """Optionaler Schutz – deine Seite ist online. Passwort in secrets.toml:
    [journal]  password = "..." """
    pw = _secret("journal", "password")
    if not pw or st.session_state.get("tj_auth"):
        return True
    st.subheader("🔒 Trading-Journal")
    entered = st.text_input("Passwort", type="password", key="tj_pw")
    if entered:
        if entered == pw:
            st.session_state.tj_auth = True
            st.rerun()
        st.error("Falsches Passwort.")
    return False


def _prepare(df: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["close_day"] = df["exit_time"].fillna(df["entry_time"]).dt.date
    df["is_closed"] = df["exit_time"].notna() & df["pnl_pct"].notna()
    names = dict(zip(acc["id"], acc["name"])) if not acc.empty else {}
    df["account"] = df["account_id"].map(names).fillna("Ohne Konto")
    df["compliance"] = df["checklist"].apply(
        lambda c: (sum(bool(c.get(r)) for r in CHECKLIST) / len(CHECKLIST) * 100) if c else None)
    return df


def _trade_title(t: pd.Series, shots: dict) -> str:
    icon = "⏳" if not t["is_closed"] else ("🟢" if t["pnl_pct"] > 0 else "🔴" if t["pnl_pct"] < 0 else "⚪")
    r = f" · {t['r_multiple']:+.2f} R" if pd.notna(t.get("r_multiple")) else ""
    cam = f" · 📷 {shots[t['id']]}" if shots.get(t["id"]) else ""
    tf = f" · {t['timeframe']}" if t.get("timeframe") else ""
    when = t["entry_time"].strftime("%d.%m. %H:%M") if pd.notna(t["entry_time"]) else ""
    return (f"{icon} {t['symbol']} {t['direction'] or ''}{tf} · {_fmt_pct(t['pnl_pct'])}{r}"
            f" · {_stars(t['rating'])}{cam} · {when} · {t['account']}")


# =========================================================================== Trade-Formular
def _trade_form(trade: dict | None, key: str, acc: pd.DataFrame) -> None:
    t = trade or {}
    is_new = trade is None
    acc_opts = [None] + acc["id"].tolist()
    acc_names = {None: "Ohne Konto", **dict(zip(acc["id"], acc["name"]))}
    entry_dt = t.get("entry_time") if t.get("entry_time") is not None and not pd.isna(t.get("entry_time")) else None
    exit_dt = t.get("exit_time") if t.get("exit_time") is not None and not pd.isna(t.get("exit_time")) else None
    now = datetime.now().replace(second=0, microsecond=0)

    def num(field):
        v = t.get(field)
        return None if v is None or pd.isna(v) else float(v)

    if t.get("source") in ("metaapi", "csv"):
        st.caption("🔌 Vom Broker importiert – Preise/Zeiten/Ergebnis werden beim nächsten Sync "
                   "aktualisiert. Kriterien, Checkliste, Notizen und Sterne bleiben erhalten.")

    with st.form(f"tj_form_{key}", clear_on_submit=is_new):
        st.markdown("**Trade**")
        c = st.columns(5)
        account_id = c[0].selectbox("Konto", acc_opts, format_func=lambda i: acc_names.get(i, "?"),
                                    index=acc_opts.index(t.get("account_id")) if t.get("account_id") in acc_opts else 0,
                                    key=f"{key}_acc")
        symbol = c[1].text_input("Symbol", value=t.get("symbol") or "", placeholder="GBPJPY", key=f"{key}_sym")
        direction = c[2].selectbox("Richtung", DIRECTIONS,
                                   index=DIRECTIONS.index(t["direction"]) if t.get("direction") in DIRECTIONS else 0,
                                   key=f"{key}_dir")
        order_type = c[3].selectbox("Ordertyp", ORDER_TYPES,
                                    index=ORDER_TYPES.index(t["order_type"]) if t.get("order_type") in ORDER_TYPES else 0,
                                    key=f"{key}_ot")
        timeframe = c[4].selectbox("Entry-Zeitfenster", TIMEFRAMES,
                                   index=TIMEFRAMES.index(t["timeframe"]) if t.get("timeframe") in TIMEFRAMES else 1,
                                   key=f"{key}_tf")

        c = st.columns(5)
        e_date = c[0].date_input("Einstieg – Tag", value=(entry_dt or now).date(), format="DD.MM.YYYY", key=f"{key}_ed")
        e_time = c[1].time_input("Einstieg – Uhrzeit", value=(entry_dt or now).time(), step=60, key=f"{key}_et")
        x_date = c[2].date_input("Ausstieg – Tag", value=(exit_dt or now).date(), format="DD.MM.YYYY", key=f"{key}_xd")
        x_time = c[3].time_input("Ausstieg – Uhrzeit", value=(exit_dt or now).time(), step=60, key=f"{key}_xt")
        still_open = c[4].checkbox("Trade noch offen", value=(not is_new and exit_dt is None), key=f"{key}_open")

        c = st.columns(7)
        entry_price = c[0].number_input("Entry", value=num("entry_price"), format="%.5f", key=f"{key}_ep")
        stop_loss = c[1].number_input("Stop Loss", value=num("stop_loss"), format="%.5f", key=f"{key}_sl")
        tp1 = c[2].number_input("TP1 (-0.27)", value=num("tp1"), format="%.5f", key=f"{key}_tp1")
        tp2 = c[3].number_input("TP2 (-0.62)", value=num("tp2"), format="%.5f", key=f"{key}_tp2")
        tp3 = c[4].number_input("TP3 (-1)", value=num("tp3"), format="%.5f", key=f"{key}_tp3")
        exit_price = c[5].number_input("Exit (Ø)", value=num("exit_price"), format="%.5f", key=f"{key}_xp")
        volume = c[6].number_input("Lots", value=num("volume"), min_value=0.0, step=0.01, key=f"{key}_vol")

        c = st.columns(4)
        risk_pct = c[0].number_input("Risiko %", value=num("risk_pct"), min_value=0.0, step=0.05, format="%.2f", key=f"{key}_risk")
        pnl_pct = c[1].number_input("Ergebnis %", value=num("pnl_pct"), step=0.05, format="%.2f", key=f"{key}_pct",
                                    help="Gewinn/Verlust in % vom Kontostand. Leer lassen + Betrag + Startsaldo → wird berechnet.")
        pnl = c[2].number_input("Ergebnis (Betrag)", value=num("pnl"), step=1.0, format="%.2f", key=f"{key}_pnl")
        c[3].metric("R-Multiple", f"{t['r_multiple']:+.2f} R" if num("r_multiple") is not None else "–",
                    help="Wird aus Entry, SL und Exit berechnet.")

        st.markdown("**Einstiegskriterien (Konfluenzen)**")
        saved_conf = set(t.get("confluences") or [])
        cols = st.columns(5)
        conf_vals = {name: cols[i % 5].checkbox(name, value=name in saved_conf, key=f"{key}_c{i}")
                     for i, name in enumerate(CONFLUENCES)}

        st.markdown("**Checkliste – habe ich mich an den Plan gehalten?**")
        saved_chk = t.get("checklist") or {}
        cols = st.columns(3)
        chk_vals = {rule: cols[i % 3].checkbox(rule, value=bool(saved_chk.get(rule)), key=f"{key}_k{i}")
                    for i, rule in enumerate(CHECKLIST)}

        c = st.columns([2, 2, 3])
        plan = c[0].radio("Plan eingehalten?", PLAN_OPTIONS, horizontal=True,
                          index=PLAN_OPTIONS.index(t["plan_followed"]) if t.get("plan_followed") in PLAN_OPTIONS else 0,
                          key=f"{key}_plan")
        rating = c[1].select_slider("Bewertung", options=[0, 1, 2, 3, 4, 5], value=0 if t.get("rating") is None or pd.isna(t.get("rating")) else int(t["rating"]),
                                    format_func=_stars, key=f"{key}_rate")

        notes = st.text_area("Notizen", value=t.get("notes") or "", height=180, key=f"{key}_notes",
                             placeholder="Warum dieser Trade? Was lief gut/schlecht? Emotionen? Lektion?")
        uploads = st.file_uploader("Screenshots hinzufügen", type=["png", "jpg", "jpeg", "webp"],
                                   accept_multiple_files=True, key=f"{key}_up")

        submitted = st.form_submit_button("💾 Trade speichern" if is_new else "💾 Änderungen speichern",
                                          type="primary")

    if not submitted:
        return
    if not symbol.strip():
        st.error("Bitte ein Symbol angeben.")
        return

    entry_time = datetime.combine(e_date, e_time)
    exit_time = None if still_open else datetime.combine(x_date, x_time)
    if exit_time and exit_time < entry_time:
        st.error("Ausstieg liegt vor dem Einstieg.")
        return

    # Ergebnis % aus Betrag ableiten, falls nur der Betrag bekannt ist
    if pnl_pct is None and pnl is not None and account_id is not None:
        bal = acc.loc[acc["id"] == account_id, "start_balance"]
        if not bal.empty and pd.notna(bal.iloc[0]) and bal.iloc[0] > 0:
            pnl_pct = round(pnl / float(bal.iloc[0]) * 100, 3)

    data = dict(
        account_id=account_id, symbol=symbol.strip().upper(), direction=direction,
        order_type=order_type, timeframe=timeframe, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=None if still_open else exit_price, stop_loss=stop_loss,
        tp1=tp1, tp2=tp2, tp3=tp3, volume=volume, risk_pct=risk_pct,
        pnl_pct=None if still_open else pnl_pct, pnl=None if still_open else pnl,
        r_multiple=None if still_open else db.compute_r(direction, entry_price, stop_loss, exit_price),
        confluences=[k for k, v in conf_vals.items() if v],
        checklist=chk_vals, plan_followed=plan, rating=rating, notes=notes,
    )
    trade_id = db.save_trade(data, None if is_new else int(t["id"]))
    for f in uploads or []:
        blob, mime = _compress_image(f)
        db.add_screenshot(trade_id, f.name, mime, blob)
    st.toast("Trade gespeichert ✅")
    st.rerun()


def _screenshots(trade_id: int, key: str) -> None:
    shots = db.list_screenshots(trade_id)
    if not shots:
        st.caption("Noch keine Screenshots – über das Formular oben hinzufügen.")
        return
    st.markdown(f"**Screenshots ({len(shots)})** – zum Vergrößern auf das Bild fahren und ⛶ klicken")
    cols = st.columns(3)
    for i, s in enumerate(shots):
        with cols[i % 3]:
            st.image(bytes(s["data"]), caption=s["filename"])
            if st.button("🗑️ Entfernen", key=f"tj_rmshot_{s['id']}_{key}"):
                db.delete_screenshot(s["id"])
                st.rerun()


def _trade_detail(t: pd.Series, acc: pd.DataFrame, key: str) -> None:
    trade = db.get_trade(int(t["id"]))
    if not trade:
        st.warning("Trade nicht gefunden.")
        return
    _trade_form(trade, key, acc)
    _screenshots(int(t["id"]), key)
    with st.popover("🗑️ Trade löschen"):
        st.write("Trade inkl. Notizen und Screenshots endgültig löschen?")
        if st.button("Ja, löschen", key=f"tj_del_{key}", type="primary"):
            db.delete_trade(int(t["id"]))
            st.session_state.pop("tj_edit_id", None)
            st.toast("Trade gelöscht")
            st.rerun()


# =========================================================================== Kalender
def _shift_month(delta: int) -> None:
    m = st.session_state.tj_month
    y, mo = divmod(m.year * 12 + m.month - 1 + delta, 12)
    st.session_state.tj_month = date(y, mo + 1, 1)
    st.session_state.tj_day = None


def _select_day(d: date) -> None:
    st.session_state.tj_day = d


def _cell_html(d: date, info: dict | None, in_month: bool) -> str:
    opacity = "1" if in_month else "0.35"
    today = "box-shadow: inset 0 0 0 2px #3b82f6;" if d == date.today() else ""
    if not info:
        return (f"<div style='height:86px;border-radius:8px;padding:6px 8px;opacity:{opacity};"
                f"border:1px solid rgba(128,128,128,.25);{today}'>"
                f"<div style='font-size:.8rem;color:{GREY}'>{d.day}</div></div>")
    col = _color(info["pct"])
    bg = {GREEN: "rgba(22,163,74,.14)", RED: "rgba(220,38,38,.14)"}.get(col, "rgba(107,114,128,.12)")
    n = info["n"]
    wr = f"{info['wins']}/{info['closed']} W" if info["closed"] else "offen"
    return (f"<div style='height:86px;border-radius:8px;padding:6px 8px;opacity:{opacity};"
            f"background:{bg};border:1px solid {col};{today}'>"
            f"<div style='font-size:.8rem;color:{GREY}'>{d.day}</div>"
            f"<div style='font-weight:700;font-size:1.05rem;color:{col}'>{_fmt_pct(info['pct'])}</div>"
            f"<div style='font-size:.72rem;color:{GREY}'>{n} Trade{'s' if n != 1 else ''} · {wr}</div></div>")


def _render_calendar(df: pd.DataFrame, acc: pd.DataFrame, shots: dict) -> None:
    ss = st.session_state
    ss.setdefault("tj_month", date.today().replace(day=1))
    ss.setdefault("tj_day", None)
    month = ss.tj_month

    c = st.columns([1, 4, 1])
    c[0].button("◀ Vormonat", key="tj_prev", on_click=_shift_month, args=(-1,))
    c[1].markdown(f"<h3 style='text-align:center;margin:0'>{MONTHS[month.month - 1]} {month.year}</h3>",
                  unsafe_allow_html=True)
    c[2].button("Nächster ▶", key="tj_next", on_click=_shift_month, args=(1,))

    weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(month.year, month.month)
    daily: dict[date, dict] = {}
    if not df.empty:
        span = df[(df["close_day"] >= weeks[0][0]) & (df["close_day"] <= weeks[-1][-1])]
        for d, g in span.groupby("close_day"):
            closed = g[g["is_closed"]]
            daily[d] = {"pct": closed["pnl_pct"].sum() if not closed.empty else None,
                        "n": len(g), "closed": len(closed), "wins": int((closed["pnl_pct"] > 0).sum())}

    # Monats-KPIs
    mdf = df[(df["close_day"] >= month) & (df["close_day"] < (month + timedelta(days=32)).replace(day=1))] \
        if not df.empty else df
    closed = mdf[mdf["is_closed"]] if not mdf.empty else mdf
    k = st.columns(5)
    k[0].metric("Netto Monat", _fmt_pct(closed["pnl_pct"].sum()) if not closed.empty else "–")
    k[1].metric("Trades", len(mdf))
    k[2].metric("Trefferquote", f"{(closed['pnl_pct'] > 0).mean() * 100:.0f} %" if not closed.empty else "–")
    gains, losses = closed[closed["pnl_pct"] > 0]["pnl_pct"].sum(), -closed[closed["pnl_pct"] < 0]["pnl_pct"].sum()
    k[3].metric("Profit-Faktor", f"{gains / losses:.2f}" if losses > 0 else ("∞" if gains > 0 else "–"))
    k[4].metric("Ø R", f"{closed['r_multiple'].mean():+.2f}" if not closed.empty and closed["r_multiple"].notna().any() else "–")

    head = st.columns(8)
    for i, w in enumerate(WEEKDAYS + ["Woche"]):
        head[i].markdown(f"<div style='text-align:center;font-weight:600;color:{GREY}'>{w}</div>",
                         unsafe_allow_html=True)

    for wk in weeks:
        cols = st.columns(8)
        week_pct, week_n = 0.0, 0
        for i, d in enumerate(wk):
            info = daily.get(d)
            with cols[i]:
                st.markdown(_cell_html(d, info, d.month == month.month), unsafe_allow_html=True)
                if info:
                    _button("Öffnen", key=f"tj_day_{d.isoformat()}", on_click=_select_day, args=(d,),
                            type="primary" if ss.tj_day == d else "secondary")
                    if d.month == month.month:
                        week_pct += info["pct"] or 0
                        week_n += info["n"]
        with cols[7]:
            col = _color(week_pct)
            st.markdown(
                f"<div style='height:86px;border-radius:8px;padding:6px 8px;border:1px dashed {GREY};'>"
                f"<div style='font-size:.72rem;color:{GREY}'>KW {wk[0].isocalendar()[1]}</div>"
                f"<div style='font-weight:700;color:{col}'>{_fmt_pct(week_pct) if week_n else '–'}</div>"
                f"<div style='font-size:.72rem;color:{GREY}'>{week_n} Trades</div></div>",
                unsafe_allow_html=True)

    sel = ss.tj_day
    if sel:
        st.divider()
        h = st.columns([5, 1])
        h[0].subheader(f"Trades am {sel.strftime('%d.%m.%Y')} ({WEEKDAYS[sel.weekday()]})")
        h[1].button("✖ Schließen", key="tj_close_day", on_click=_select_day, args=(None,))
        day_df = df[df["close_day"] == sel].sort_values("entry_time")
        for _, t in day_df.iterrows():
            with st.expander(_trade_title(t, shots), expanded=len(day_df) == 1):
                _trade_detail(t, acc, key=f"cal{t['id']}")


# =========================================================================== Trade-Liste
def _render_list(df: pd.DataFrame, acc: pd.DataFrame, shots: dict) -> None:
    if df.empty:
        st.info("Noch keine Trades. Lege einen unter **➕ Neuer Trade** an oder verbinde ein Konto.")
        return
    view = df.sort_values("entry_time", ascending=False)
    table = pd.DataFrame({
        "ID": view["id"],
        "Einstieg": view["entry_time"].dt.strftime("%d.%m.%Y %H:%M"),
        "Ausstieg": view["exit_time"].dt.strftime("%d.%m.%Y %H:%M").fillna("offen"),
        "Konto": view["account"],
        "Symbol": view["symbol"],
        "Richtung": view["direction"],
        "Order": view["order_type"],
        "TF": view["timeframe"],
        "Ergebnis %": view["pnl_pct"].round(2),
        "R": view["r_multiple"],
        "Risiko %": view["risk_pct"],
        "Plan": view["plan_followed"],
        "Checkliste %": view["compliance"].round(0),
        "Sterne": view["rating"].apply(_stars),
        "Kriterien": view["confluences"].apply(lambda l: ", ".join(l)),
        "📷": view["id"].map(shots).fillna(0).astype(int),
    })
    st.dataframe(table, hide_index=True, height=420)
    st.download_button("⬇️ Als CSV exportieren", table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="trading_journal.csv", mime="text/csv")

    st.markdown("#### Trade öffnen / bearbeiten")
    ids = view["id"].tolist()
    indexed = view.set_index("id", drop=False)
    labels = {i: _trade_title(indexed.loc[i], shots) for i in ids}
    choice = st.selectbox("Trade", ids, format_func=lambda i: labels[i], key="tj_list_pick")
    if choice is not None:
        _trade_detail(view[view["id"] == choice].iloc[0], acc, key=f"list{choice}")


# =========================================================================== Auswertung
def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * ((p * (1 - p) + z ** 2 / (4 * n)) / n) ** 0.5
    return ((centre - margin) / denom * 100, (centre + margin) / denom * 100)


def _group_stats(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        n, wins = len(g), int((g["pnl_pct"] > 0).sum())
        lo, hi = _wilson(wins, n)
        gains, losses = g[g["pnl_pct"] > 0]["pnl_pct"].sum(), -g[g["pnl_pct"] < 0]["pnl_pct"].sum()
        rows.append({label: "–" if pd.isna(key) else key, "Trades": n,
                     "Trefferquote %": round(wins / n * 100, 1),
                     "95%-KI": f"{lo:.0f}–{hi:.0f} %",
                     "Ø Ergebnis %": round(g["pnl_pct"].mean(), 2),
                     "Summe %": round(g["pnl_pct"].sum(), 2),
                     "Ø R": round(g["r_multiple"].mean(), 2) if g["r_multiple"].notna().any() else None,
                     "Profit-Faktor": round(gains / losses, 2) if losses > 0 else None})
    return pd.DataFrame(rows).sort_values("Trades", ascending=False)


def _render_stats(df: pd.DataFrame) -> None:
    closed = df[df["is_closed"]].sort_values("exit_time") if not df.empty else df
    if closed.empty:
        st.info("Sobald abgeschlossene Trades mit Ergebnis % vorhanden sind, erscheint hier die Auswertung.")
        return
    wins, losses = closed[closed["pnl_pct"] > 0], closed[closed["pnl_pct"] < 0]
    equity = closed["pnl_pct"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    pf = wins["pnl_pct"].sum() / -losses["pnl_pct"].sum() if not losses.empty else float("inf")
    lo, hi = _wilson(len(wins), len(closed))

    k = st.columns(4)
    k[0].metric("Trades (geschlossen)", len(closed))
    k[1].metric("Trefferquote", f"{len(wins) / len(closed) * 100:.1f} %", help=f"95%-KI: {lo:.0f}–{hi:.0f} %")
    k[2].metric("Netto", _fmt_pct(closed["pnl_pct"].sum()))
    k[3].metric("Profit-Faktor", f"{pf:.2f}" if pf != float("inf") else "∞")
    k = st.columns(4)
    k[0].metric("Ø Gewinn", _fmt_pct(wins["pnl_pct"].mean()) if not wins.empty else "–")
    k[1].metric("Ø Verlust", _fmt_pct(losses["pnl_pct"].mean()) if not losses.empty else "–")
    k[2].metric("Erwartungswert / Trade", _fmt_pct(closed["pnl_pct"].mean()))
    k[3].metric("Max. Drawdown (kumuliert)", _fmt_pct(max_dd))

    if len(closed) < 30:
        st.caption(f"⚠️ Erst {len(closed)} Trades – Trefferquoten je Gruppe sind bei so wenigen Daten "
                   "noch sehr unsicher (siehe Konfidenzintervall).")

    st.markdown("#### Equity-Kurve (kumulierte %)")
    st.line_chart(pd.DataFrame({"Kumuliert %": equity.values}, index=closed["exit_time"].values))

    st.markdown("#### Welche Kriterien funktionieren?")
    ex = closed.explode("confluences")
    ex = ex[ex["confluences"].notna()]
    if not ex.empty:
        st.dataframe(_group_stats(ex, "confluences", "Kriterium"), hide_index=True)
    cnt = closed.assign(n_conf=closed["confluences"].apply(len))
    st.markdown("**Anzahl Konfluenzen im Trade**")
    st.dataframe(_group_stats(cnt, "n_conf", "Konfluenzen"), hide_index=True)

    c = st.columns(2)
    with c[0]:
        st.markdown("**Plan eingehalten?**")
        st.dataframe(_group_stats(closed, "plan_followed", "Plan"), hide_index=True)
        st.markdown("**Entry-Zeitfenster**")
        st.dataframe(_group_stats(closed, "timeframe", "TF"), hide_index=True)
        st.markdown("**Wochentag (Einstieg)**")
        wd = closed.assign(wd=closed["entry_time"].dt.weekday.map(dict(enumerate(WEEKDAYS))))
        st.dataframe(_group_stats(wd, "wd", "Tag"), hide_index=True)
    with c[1]:
        st.markdown("**Ordertyp**")
        st.dataframe(_group_stats(closed, "order_type", "Order"), hide_index=True)
        st.markdown("**Symbol**")
        st.dataframe(_group_stats(closed, "symbol", "Symbol"), hide_index=True)
        st.markdown("**Bewertung (Sterne)**")
        st.dataframe(_group_stats(closed.assign(s=closed["rating"].apply(_stars)), "s", "Sterne"), hide_index=True)


# =========================================================================== Konten & Sync
def _render_accounts(acc: pd.DataFrame) -> None:
    token = _secret("metaapi", "token") or st.session_state.get("tj_token")
    st.markdown("#### 🔌 Verbundene Konten")
    if token:
        st.caption("MetaApi-Token ist hinterlegt ✅")
    else:
        st.warning("Kein MetaApi-Token in den Secrets. Für einen Test kannst du ihn hier eingeben "
                   "(wird nicht gespeichert, nur für diese Sitzung).")
        t = st.text_input("MetaApi-Token", type="password", key="tj_token_in")
        if t:
            st.session_state.tj_token = t
            token = t

    if acc.empty:
        st.info("Noch kein Konto angelegt.")
    else:
        if token and acc["metaapi_account_id"].notna().any() and st.button("🔄 Alle Konten synchronisieren", type="primary"):
            for _, a in acc[acc["metaapi_account_id"].notna()].iterrows():
                _run_sync(a.to_dict(), token)
        for _, a in acc.iterrows():
            with st.container(border=True):
                c = st.columns([3, 2, 2, 1, 1])
                c[0].markdown(f"**{a['name']}**  \n{a['broker'] or ''} · {a['platform'] or ''} · {a['currency'] or ''}")
                c[1].caption(f"MetaApi-ID: {a['metaapi_account_id'] or '–'}  \nRegion: {a['region'] or '–'}")
                last = pd.to_datetime(a["last_sync"])
                c[2].caption(f"Startsaldo: {a['start_balance'] or '–'}  \n"
                             f"Letzter Sync: {last.strftime('%d.%m.%Y %H:%M') if pd.notna(last) else 'nie'}")
                if c[3].button("🔄 Sync", key=f"tj_sync_{a['id']}", disabled=not (token and a["metaapi_account_id"])):
                    _run_sync(a.to_dict(), token)
                with c[4].popover("🗑️"):
                    st.write("Konto entfernen? Die Trades bleiben im Journal.")
                    if st.button("Entfernen", key=f"tj_delacc_{a['id']}", type="primary"):
                        db.delete_account(int(a["id"]))
                        st.rerun()

    with st.expander("➕ Konto hinzufügen", expanded=acc.empty):
        with st.form("tj_add_acc", clear_on_submit=True):
            c = st.columns(3)
            name = c[0].text_input("Name", placeholder="FTMO 100k Phase 1")
            broker = c[1].selectbox("Broker / Prop-Firm", BROKERS)
            platform = c[2].selectbox("Plattform", PLATFORMS)
            c = st.columns(3)
            meta_id = c[0].text_input("MetaApi-Konto-ID (optional)",
                                      help="Leer lassen, wenn du nur manuell oder per CSV journalst.")
            region = c[1].selectbox("MetaApi-Region", bs.METAAPI_REGIONS)
            sync_from = c[2].date_input("Sync ab", value=date.today() - timedelta(days=365), format="DD.MM.YYYY",
                                        help="Am besten vor dem Kontostart, damit die Ersteinzahlung erkannt wird.")
            c = st.columns(3)
            start_balance = c[0].number_input("Startsaldo", value=100000.0, step=1000.0)
            currency = c[1].selectbox("Kontowährung", ["USD", "EUR", "GBP", "CHF"])
            if st.form_submit_button("Konto speichern", type="primary"):
                if not name.strip():
                    st.error("Bitte einen Namen angeben.")
                else:
                    db.save_account(dict(name=name.strip(), broker=broker, platform=platform,
                                         metaapi_account_id=meta_id.strip() or None, region=region,
                                         start_balance=start_balance, currency=currency,
                                         sync_from=datetime.combine(sync_from, time())))
                    st.toast("Konto gespeichert ✅")
                    st.rerun()

    with st.expander("ℹ️ So verbindest du Pepperstone, FTMO, Vantage & Co."):
        st.markdown("""
1. Kostenlosen Account bei **metaapi.cloud** anlegen → unter *API access* einen **Token** erzeugen.
2. Im MetaApi-Dashboard unter *Accounts* dein MT4/MT5-Konto hinzufügen: **Login, Server und das
   Investor-Passwort** (nur Lesezugriff – damit kann niemand traden). Das Master-Passwort brauchst du nicht.
   Investor-Passwort findest du bei FTMO/Pepperstone/Vantage im Kundenbereich bzw. setzt es im MT-Terminal.
3. Konto auf **deployed** stellen, die **Konto-ID** und **Region** kopieren und oben eintragen.
4. Token in `.streamlit/secrets.toml` (lokal) bzw. in den App-Secrets auf Streamlit Cloud eintragen:
   ```toml
   [metaapi]
   token = "DEIN_TOKEN"
   ```
Ein Konto ist bei MetaApi kostenlos, weitere kosten. Alternativ: CSV-Import unten oder manuell eintragen.
""")

    st.markdown("#### 📄 CSV-Import")
    c = st.columns([3, 2])
    up = c[0].file_uploader("CSV mit Trades", type=["csv", "txt"], key="tj_csv")
    c[1].download_button("⬇️ CSV-Vorlage", bs.csv_template(), file_name="journal_vorlage.csv", mime="text/csv")
    if up is not None:
        try:
            parsed = bs.parse_trades_csv(bs.read_csv_flexible(up))
            st.dataframe(pd.DataFrame(parsed).drop(columns=["external_id"], errors="ignore").head(20), hide_index=True)
            acc_opts = [None] + acc["id"].tolist()
            names = {None: "Ohne Konto", **dict(zip(acc["id"], acc["name"]))}
            target = st.selectbox("Zu Konto zuordnen", acc_opts, format_func=lambda i: names[i], key="tj_csv_acc")
            if st.button(f"{len(parsed)} Trades importieren", type="primary"):
                res = bs.import_csv_trades(parsed, target)
                st.toast(f"Import: {res['neu']} neu, {res['aktualisiert']} aktualisiert ✅")
                st.rerun()
        except bs.SyncError as e:
            st.error(str(e))


def _run_sync(account: dict, token: str) -> None:
    with st.spinner(f"Synchronisiere {account['name']} …"):
        try:
            res = bs.sync_account(account, token)
            st.success(f"{account['name']}: {res['neu']} neue, {res['aktualisiert']} aktualisierte Trades.")
        except bs.SyncError as e:
            st.error(f"{account['name']}: {e}")


# =========================================================================== Einstieg
def render_trading_journal() -> None:
    if not _check_password():
        return
    st.title("📓 Trading-Journal")

    acc = db.list_accounts()
    df = _prepare(db.load_trades(), acc)
    shots = db.screenshot_counts()

    if not df.empty and df["account"].nunique() > 1:
        options = ["Alle Konten"] + sorted(df["account"].unique().tolist())
        pick = st.selectbox("Konto-Filter", options, key="tj_acc_filter")
        if pick != "Alle Konten" and not df.empty:
            df = df[df["account"] == pick]

    tabs = st.tabs(["📅 Kalender", "➕ Neuer Trade", "📋 Alle Trades", "📊 Auswertung", "🔌 Konten & Import"])
    with tabs[0]:
        _render_calendar(df, acc, shots)
    with tabs[1]:
        _trade_form(None, "new", acc)
    with tabs[2]:
        _render_list(df, acc, shots)
    with tabs[3]:
        _render_stats(df)
    with tabs[4]:
        _render_accounts(acc)
        if db.backend_name() == "sqlite":
            st.caption("💾 Speicher: lokale SQLite-Datei (data/trading_journal.db). "
                       "Für die Online-Seite eine Postgres-DB in den Secrets eintragen, sonst gehen Daten beim Neustart verloren.")


if __name__ == "__main__":       # Standalone-Test:  streamlit run trading_journal.py
    st.set_page_config(page_title="Trading-Journal", page_icon="📓", layout="wide")
    render_trading_journal()
