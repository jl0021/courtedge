import datetime as dt
import shutil
import sqlite3

import pytest

from src import kalshi_reader, risk_gate, slate, vault_writer
from src.server import App
from tests.test_server import app  # noqa: F401  (module fixture: a small two-tour database)

NOW = dt.datetime(2026, 11, 17, 4, 0, tzinfo=dt.timezone.utc)
LATER = (NOW + dt.timedelta(hours=9)).isoformat().replace("+00:00", "Z")


def market(event, code, who, opponent, ask, size=100.0):
    return {"ticker": f"{event}-{code}", "event_ticker": event, "status": "active", "result": "",
            "yes_sub_title": who, "yes_ask_dollars": f"{ask:.4f}", "yes_ask_size_fp": f"{size:.2f}",
            "rules_primary": f"If {who} wins the {who.split()[-1]} vs {opponent.split()[-1]} professional tennis match "
                             f"in the 2026 Test Open Men Singles Round 1 after a ball has been played, then the market "
                             f"resolves to Yes."}


def match(tour, code, fav, dog, fav_ask, dog_ask):
    event = f"KX{tour}MATCH-26NOV17{code}"
    return [market(event, "F", fav, dog, fav_ask), market(event, "D", dog, fav, dog_ask)]


class FakeKalshi:
    """Answers the three GET endpoints the scanner uses, from in-memory data."""

    def __init__(self):
        self.atp = (match("ATP", "M1", "First1 Last1", "First12 Last12", 0.40, 0.62)
                    + match("ATP", "M2", "First2 Last2", "First11 Last11", 0.40, 0.62)
                    + match("ATP", "M3", "First3 Last3", "First10 Last10", 0.40, 0.62)
                    + match("ATP", "LIVE", "First4 Last4", "First9 Last9", 0.40, 0.62)
                    + match("ATP", "UNK", "First5 Last5", "Nobody Atall", 0.40, 0.62))
        self.totals = [{"ticker": "KXATPGTOTAL-26NOV17M1-22", "event_ticker": "KXATPGTOTAL-26NOV17M1",
                        "status": "active", "strike_type": "greater", "floor_strike": 21.5,
                        "yes_ask_dollars": "0.3000", "yes_ask_size_fp": "50.00"}]
        self.combos = []
        self.settled = {}
        self.calls = []

    def __call__(self, path, params):
        self.calls.append(path)
        if path == "/markets":
            series = params["series_ticker"]
            data = {"KXATPMATCH": self.atp, "KXATPGTOTAL": self.totals}.get(series, [])
            if series in kalshi_reader.COMBO_SERIES:
                data = self.combos
            return {"markets": data, "cursor": ""}
        if path == "/milestones":
            event = params["related_event_ticker"]
            status = "live" if "LIVE" in event else "not_started"
            return {"milestones": [{"details": {"main_game_event_ticker": event, "status": status, "best_of": "3",
                                                "tournament_name": "Test Open Men Singles"},
                                    "primary_event_tickers": [event], "start_date": LATER}]}
        if path.startswith("/markets/"):
            ticker = path.rsplit("/", 1)[1]
            result = self.settled.get(ticker)
            if result is None:
                return {"market": {"ticker": ticker, "status": "active", "result": ""}}
            if result == "void":
                return {"market": {"ticker": ticker, "status": "finalized", "result": "",
                                   "settlement_value_dollars": "0.5000"}}
            return {"market": {"ticker": ticker, "status": "finalized", "result": result}}
        raise AssertionError(f"unexpected request {path}")


@pytest.fixture
def setup(app, tmp_path):  # noqa: F811
    """A private copy of the shared database, with the Milestone 1 gate marked as passed."""
    path = tmp_path / "slate.db"
    shutil.copy(app.db_path, path)
    copy = App(db_path=path, archive=app.archive, write_vault=False)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO backtest_runs VALUES ('gate-ok', '2999-01-01T00:00:00', 'analytic', 600, 0.20, 0.01, "
                 "0.6, 0.66, 1, '{}')")
    conn.commit()
    slate.init_ledger(conn)
    yield conn, copy.models(), FakeKalshi()
    conn.close()


def test_scan_prices_and_filters_legs(setup):
    conn, models, kalshi = setup
    payload = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    legs = {leg["market_ticker"]: leg for leg in payload["legs"]}
    for code in ("M1", "M2", "M3"):
        fav = legs[f"KXATPMATCH-26NOV17{code}-F"]
        assert fav["qualifies"], fav["reasons"]
        assert fav["simulated_prob"] - fav["kalshi_ask"] >= risk_gate.MIN_LEG_EDGE
        assert fav["player_name"].startswith("First") and fav["best_of"] == 3
        dog = legs[f"KXATPMATCH-26NOV17{code}-D"]
        assert not dog["qualifies"] and any("leg edge" in r for r in dog["reasons"])
        assert fav["simulated_prob"] + dog["simulated_prob"] == pytest.approx(1.0)
    assert "match has started or finished" in legs["KXATPMATCH-26NOV17LIVE-F"]["reasons"]
    assert any("below 98%" in r for r in legs["KXATPMATCH-26NOV17UNK-F"]["reasons"])
    total = legs["KXATPGTOTAL-26NOV17M1-22"]
    assert total["kind"] == "total" and slate.TOTALS_DISABLED_REASON in total["reasons"]
    assert 0 < total["simulated_prob"] < 1
    assert payload["counts"]["qualifying_legs"] == 3


def test_combos_pass_the_gate_and_are_paper_logged(setup):
    conn, models, kalshi = setup
    payload = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    approved = [c for c in payload["combos"] if c["approved"]]
    assert approved
    for combo in approved:
        assert len(combo["legs"]) >= 2 and combo["fair_combo_prob"] >= risk_gate.MIN_COMBO_HIT_PROB
        assert combo["edge"] >= risk_gate.MIN_COMBO_EDGE
        assert len({leg["match_key"] for leg in combo["legs"]}) == len(combo["legs"])
        assert combo["price_source"] == slate.PRICE_SOURCE_PRODUCT
    rows = conn.execute("SELECT combo_id, price_source, stake_usd, status FROM paper_combos").fetchall()
    assert 1 <= len(rows) <= slate.MAX_NEW_COMBOS_PER_SCAN and all(r[2] == 1.0 and r[3] == "open" for r in rows)
    tickers = [r[0] for r in conn.execute("SELECT market_ticker FROM paper_legs")]
    assert len(tickers) == len(set(tickers))                          # no leg staked twice
    # A second scan finds the same legs but logs nothing new: they are already riding.
    again = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    assert again["counts"]["paper_trades_logged"] == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_combos").fetchone()[0] == len(rows)


def test_existing_kalshi_combo_market_sets_the_price(setup):
    conn, models, kalshi = setup
    legs = ["KXATPMATCH-26NOV17M1-F", "KXATPMATCH-26NOV17M2-F"]
    kalshi.combos = [{"ticker": "KXMVETEST-1", "status": "active", "yes_ask_dollars": "0.2000",
                      "yes_ask_size_fp": "25.00",
                      "mve_selected_legs": [{"market_ticker": t, "side": "yes", "event_ticker": t[:-2]} for t in legs]}]
    payload = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None, paper_trade=False)
    pair = next(c for c in payload["combos"] if {l["market_ticker"] for l in c["legs"]} == set(legs))
    assert pair["price_source"] == slate.PRICE_SOURCE_COMBO and pair["kalshi_combo_ticker"] == "KXMVETEST-1"
    assert pair["price"] == pytest.approx(0.20)
    others = [c for c in payload["combos"] if c is not pair]
    assert all(c["price_source"] == slate.PRICE_SOURCE_PRODUCT for c in others)
    assert payload["counts"]["kalshi_combo_markets"] == 1


def test_gate_must_be_open(setup):
    conn, models, kalshi = setup
    conn.execute("INSERT INTO backtest_runs VALUES ('gate-bad', '3000-01-01T00:00:00', 'analytic', 600, 0.25, 0.08, "
                 "0.7, 0.6, 0, '{}')")
    conn.commit()
    with pytest.raises(slate.GateClosed, match="failed the Milestone 1 gate"):
        slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    assert kalshi.calls == []                                          # Kalshi was never contacted


def test_vetoed_player_cannot_be_a_leg(setup):
    conn, models, kalshi = setup
    conn.execute("UPDATE player_status SET last_outcome = 'RET', last_match_won = 0, last_match_ret_wo = 1 "
                 "WHERE tour = 'ATP' AND player_id = 1")
    conn.commit()
    payload = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None, paper_trade=False)
    leg = next(l for l in payload["legs"] if l["market_ticker"] == "KXATPMATCH-26NOV17M1-F")
    assert not leg["qualifies"] and "lost last match by RET" in leg["reasons"]
    assert all("KXATPMATCH-26NOV17M1-F" not in [l["market_ticker"] for l in c["legs"]]
               for c in payload["combos"] if c["approved"])


def test_no_cash_no_paper_trades(setup, monkeypatch):
    conn, models, kalshi = setup
    monkeypatch.setattr(slate, "START_BANKROLL", 0.5)
    payload = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    assert payload["counts"]["combos_approved"] > 0 and payload["counts"]["paper_trades_logged"] == 0


def test_settlement_won_lost_void_and_bankroll(setup):
    conn, models, kalshi = setup
    slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    combo_id, price = conn.execute("SELECT combo_id, kalshi_combo_price FROM paper_combos ORDER BY combo_id").fetchone()
    tickers = [r[0] for r in conn.execute("SELECT market_ticker FROM paper_legs WHERE combo_id = ?", (combo_id,))]

    assert slate.settle(conn, fetch=kalshi, now=NOW)["still_open"] >= 1            # nothing settled yet
    kalshi.settled = {tickers[0]: "yes"}
    assert slate.settle(conn, fetch=kalshi, now=NOW)["won"] == 0                    # one leg is not a combo

    kalshi.settled = {t: "yes" for t in tickers}
    summary = slate.settle(conn, fetch=kalshi, now=NOW)
    assert summary["won"] == 1
    status, payout = conn.execute("SELECT status, payout_usd FROM paper_combos WHERE combo_id = ?", (combo_id,)).fetchone()
    assert status == "won" and payout == pytest.approx(round(1.0 / price, 4))
    book = slate.bankroll(conn)
    assert book["won"] == 1 and book["profit"] == pytest.approx(payout - 1.0, abs=1e-4)
    assert slate.settle(conn, fetch=kalshi, now=NOW)["won"] == 0                    # settled combos are not regraded


def test_a_losing_leg_settles_the_combo_at_once_and_void_refunds(setup):
    conn, models, kalshi = setup
    slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    ids = [r[0] for r in conn.execute("SELECT combo_id FROM paper_combos ORDER BY combo_id")]
    first = [r[0] for r in conn.execute("SELECT market_ticker FROM paper_legs WHERE combo_id = ?", (ids[0],))]
    kalshi.settled = {first[0]: "no"}
    assert slate.settle(conn, fetch=kalshi, now=NOW)["lost"] == 1
    assert conn.execute("SELECT status, payout_usd FROM paper_combos WHERE combo_id = ?", (ids[0],)).fetchone() == ("lost", 0.0)

    conn.execute("INSERT INTO paper_combos (created_at, slate_date, combo_key, n_legs, fair_combo_prob, "
                 "kalshi_combo_price, price_source, edge, stake_usd) VALUES ('t', '2026-11-17', 'void-test', 1, 0.5, "
                 "0.4, 'leg_product', 0.1, 1.0)")
    void_id = conn.execute("SELECT MAX(combo_id) FROM paper_combos").fetchone()[0]
    conn.execute("INSERT INTO paper_legs (combo_id, tour, match_key, market_ticker, kind, label, simulated_prob, kalshi_ask) "
                 "VALUES (?, 'ATP', 'k', 'VOID-LEG', 'winner', 'x', 0.5, 0.4)", (void_id,))
    kalshi.settled = {"VOID-LEG": "void"}
    slate.settle(conn, fetch=kalshi, now=NOW)
    assert conn.execute("SELECT status, payout_usd FROM paper_combos WHERE combo_id = ?", (void_id,)).fetchone() == ("void", 1.0)


def test_build_combos_limits_and_ordering():
    def leg(i, prob, ask, match=None):
        return slate.LegCandidate(tour="ATP", match_key=match or f"m{i}", event_ticker=f"e{i}", market_ticker=f"t{i}",
                                  kind="winner", label=f"P{i}", player=f"P{i}", opponent="O", tournament="",
                                  round_name="", kalshi_ask=ask, ask_size=10, simulated_prob=prob, edge=prob - ask,
                                  name_confidence=1.0, player_name=f"P{i}")
    legs = [leg(1, 0.80, 0.70), leg(2, 0.75, 0.65), leg(3, 0.60, 0.50), leg(4, 0.90, 0.80, match="m1")]
    combos = slate.build_combos(legs, [])
    keys = [c.key for c in combos]
    assert "t1|t4" not in keys                                         # same match
    assert all(c.fair_combo_prob >= risk_gate.MIN_COMBO_HIT_PROB for c in combos)
    assert combos == sorted(combos, key=lambda c: (not c.approved, -c.edge, -c.fair_combo_prob))
    assert "t1|t2|t3" in keys or 0.8 * 0.75 * 0.6 < 0.30
    assert len(slate.build_combos(legs, [], limit=1)) == 1


def test_old_placeholder_ledger_tables_are_replaced(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.execute("CREATE TABLE paper_combos (combo_id INTEGER PRIMARY KEY, created_at TEXT)")
    conn.execute("CREATE TABLE paper_legs (leg_id INTEGER PRIMARY KEY)")
    slate.init_ledger(conn)
    assert "combo_key" in {row[1] for row in conn.execute("PRAGMA table_info(paper_combos)")}


def test_vault_notes_after_scan_and_settle(setup, tmp_path):
    conn, models, kalshi = setup
    payload = slate.scan(conn, models, fetch=kalshi, now=NOW, log=lambda m: None)
    tickers = [r[0] for r in conn.execute("SELECT market_ticker FROM paper_legs WHERE combo_id = 1")]
    kalshi.settled = {t: "yes" for t in tickers}
    slate.settle(conn, fetch=kalshi, now=NOW)
    day = slate.local_date(NOW).isoformat()           # 2026-11-16 in California, 2026-11-17 in UTC
    written = vault_writer.write_all(tmp_path, conn, payload, slate.local_date(NOW))
    names = {p.relative_to(tmp_path).as_posix() for p in written}
    assert f"Daily_Slates/{day}.md" in names and "Post_Mortems/Paper_Ledger.md" in names
    assert f"Winners/{day} Combo 1.md" in names
    assert "Players/First1 Last1.md" in names
    for path in written:
        text = path.read_text()
        head = text.split("---\n")[1].splitlines()
        assert [line.split(":")[0] for line in head] == list(vault_writer.FRONTMATTER_KEYS), path
        assert text.startswith("---\n")
    slate_note = (tmp_path / f"Daily_Slates/{day}.md").read_text()
    assert "[[First1 Last1]]" in slate_note and "status: combos" in slate_note
    assert "leg_product" not in slate_note and "product of the legs' asks (estimate)" in slate_note
    audit = (tmp_path / "Post_Mortems/Paper_Ledger.md").read_text()
    assert "status: collecting" in audit and "to go" in audit
    # Writing again does not duplicate winner notes.
    assert not any("Winners" in p.as_posix() for p in vault_writer.write_all(tmp_path, conn, None, NOW.date()))


def run_job(application, name):
    import time
    assert application.start_job(name)
    deadline = time.time() + 60
    while application.job["state"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    return application.job


def test_dashboard_scan_and_settle_jobs(app, tmp_path):  # noqa: F811
    path = tmp_path / "dash.db"
    shutil.copy(app.db_path, path)
    dashboard = App(db_path=path, archive=app.archive, write_vault=True)
    dashboard.vault_dir = tmp_path / "vault"
    dashboard.kalshi_fetch = kalshi = FakeKalshi()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO backtest_runs VALUES ('bad', '2999-01-01T00:00:00', 'analytic', 600, 0.30, 0.09, "
                 "0.7, 0.6, 0, '{}')")
    conn.commit()
    job = run_job(dashboard, "scan")
    assert job["state"] == "error" and "gate" in job["message"]
    assert kalshi.calls == []                         # the gate stopped it before any market was read

    conn.execute("INSERT INTO backtest_runs VALUES ('good', '3000-01-01T00:00:00', 'analytic', 600, 0.20, 0.01, "
                 "0.6, 0.66, 1, '{}')")
    conn.commit()
    assert run_job(dashboard, "scan")["state"] == "done"
    view = dashboard.slate_view()
    assert view["gate"]["open"] and view["scan"]["counts"]["qualifying_legs"] == 3
    assert view["bankroll"]["open"] >= 1 and view["ledger"][0]["legs"]
    assert view["totals_enabled"] is False
    assert (tmp_path / "vault" / "Daily_Slates").is_dir() and (tmp_path / "vault" / "Players").is_dir()

    kalshi.settled = {leg["market_ticker"]: "no" for combo in view["ledger"] for leg in combo["legs"]}
    assert run_job(dashboard, "settle")["state"] == "done"
    assert dashboard.slate_view()["bankroll"]["lost"] == view["bankroll"]["open"]
    conn.close()
