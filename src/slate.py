"""Live slate: price Kalshi tennis legs with the simulator, build combos, keep the paper ledger.

Flow of one scan:
  1. Refuse to run unless the latest backtest passed the Milestone 1 gate.
  2. Read open match-winner and total-games markets (GET only, src/kalshi_reader.py).
  3. Keep matches Kalshi lists as not started, with both names resolved at 98%.
  4. Simulate each match 10,000 times and compare each leg with its Yes ask.
  5. Build combos from qualifying legs and price each one the hybrid way:
     Kalshi's own combo market when one exists, otherwise the product of the legs' asks.
  6. Run every combo through src/risk_gate.py and paper-log the best few.

Nothing here can place an order. A paper trade is a row in SQLite.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import zlib
from dataclasses import asdict, dataclass, field

from src import ingest, kalshi_reader, risk_gate, sim
from src.model import bo3_probability, serve_inputs

START_BANKROLL = 10.0
UNIT_STAKE = 1.0
MAX_NEW_COMBOS_PER_SCAN = 3
MAX_COMBOS_EVALUATED = 50_000
SHOW_COMBOS = 25

# The simulator's total-games numbers failed a calibration check on 2,875 archive matches
# (it predicts about 2.3 games too many per best-of-3 match and scores worse than a constant
# guess), so total-games legs are read and shown but never used in a combo.
TOTALS_ENABLED = False
TOTALS_DISABLED_REASON = "total-games model not calibrated"

PRICE_SOURCE_COMBO = "kalshi_combo"
PRICE_SOURCE_PRODUCT = "leg_product"

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_combos (
    combo_id        INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,
    slate_date      TEXT NOT NULL,
    combo_key       TEXT NOT NULL UNIQUE,          -- sorted leg tickers; the same combo is never logged twice
    n_legs          INTEGER NOT NULL,
    fair_combo_prob REAL NOT NULL,
    kalshi_combo_price REAL NOT NULL,
    price_source    TEXT NOT NULL,                 -- 'kalshi_combo' or 'leg_product'
    kalshi_combo_ticker TEXT,
    edge            REAL NOT NULL,
    stake_usd       REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',  -- open, won, lost, void
    payout_usd      REAL,
    settled_at      TEXT
);
CREATE TABLE IF NOT EXISTS paper_legs (
    leg_id          INTEGER PRIMARY KEY,
    combo_id        INTEGER NOT NULL REFERENCES paper_combos (combo_id),
    tour            TEXT NOT NULL,
    match_key       TEXT NOT NULL,
    market_ticker   TEXT NOT NULL,
    kind            TEXT NOT NULL,                 -- 'winner' (totals are not traded yet)
    label           TEXT NOT NULL,
    player_id       INTEGER,
    opponent_id     INTEGER,
    player_name     TEXT,
    opponent_name   TEXT,
    surface         TEXT,
    start_time      TEXT,
    simulated_prob  REAL NOT NULL,
    kalshi_ask      REAL NOT NULL,
    result          TEXT,                          -- yes, no, void; NULL while open
    settled_at      TEXT
);
CREATE TABLE IF NOT EXISTS kalshi_scans (
    scan_id     INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL                      -- JSON shown by the dashboard and the daily slate
);
"""


class GateClosed(RuntimeError):
    """Raised when the Milestone 1 gate has not passed, so live scanning stays off."""


@dataclass
class LegCandidate:
    tour: str
    match_key: str
    event_ticker: str
    market_ticker: str
    kind: str                     # 'winner' or 'total'
    label: str                    # "Carlos Alcaraz to win" / "Over 22.5 games"
    player: str                   # name as Kalshi writes it (match players for a total)
    opponent: str
    tournament: str
    round_name: str
    kalshi_ask: float | None
    ask_size: float
    player_id: int | None = None
    opponent_id: int | None = None
    player_name: str | None = None        # canonical archive name, used for [[wikilinks]]
    opponent_name: str | None = None
    name_confidence: float = 0.0
    surface: str = "Unknown"
    best_of: int = 3
    start_time: str | None = None
    simulated_prob: float | None = None
    edge: float | None = None
    last_result: str | None = None
    last_result_won: bool | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def qualifies(self) -> bool:
        return not self.reasons

    def to_gate_leg(self) -> risk_gate.Leg:
        return risk_gate.Leg(player=self.player_name or self.player, match_key=self.match_key,
                             simulated_prob=self.simulated_prob or 0.0, kalshi_ask=self.kalshi_ask or 0.0,
                             name_match_confidence=self.name_confidence, last_result=self.last_result,
                             last_result_won=self.last_result_won)


@dataclass
class ComboCandidate:
    legs: list[LegCandidate]
    fair_combo_prob: float
    price: float
    price_source: str
    kalshi_combo_ticker: str | None
    edge: float
    approved: bool
    reasons: list[str]

    @property
    def key(self) -> str:
        return combo_key(leg.market_ticker for leg in self.legs)


def local_date(moment: dt.datetime) -> dt.date:
    """The calendar date on this computer's clock. Daily slates follow your day, not UTC's."""
    return moment.astimezone().date()


def combo_key(tickers) -> str:
    return "|".join(sorted(tickers))


def init_ledger(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(paper_combos)")}
    if existing and "combo_key" not in existing:
        # Placeholder tables from Milestone 1 that were never written to.
        if conn.execute("SELECT COUNT(*) FROM paper_combos").fetchone()[0]:
            raise RuntimeError("paper_combos has rows in an old layout; back up data/courtedge.db and remove them")
        conn.execute("DROP TABLE IF EXISTS paper_legs")
        conn.execute("DROP TABLE paper_combos")
    conn.executescript(LEDGER_SCHEMA)
    conn.commit()


def gate_status(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT run_id, brier, max_calibration_gap, passed FROM backtest_runs "
                       "ORDER BY created_at DESC LIMIT 1").fetchone()
    if row is None:
        return {"open": False, "reason": "No backtest has been run yet."}
    run_id, brier, gap, passed = row
    if not passed:
        return {"open": False, "reason": f"The latest backtest failed the Milestone 1 gate (Brier {brier:.4f}, "
                                          f"largest calibration gap {gap:.1%})."}
    return {"open": True, "reason": f"Gate passed in run {run_id}."}


# --------------------------------------------------------------------------- pricing

def _player_status(conn: sqlite3.Connection, tour: str, player_id: int) -> tuple[str | None, bool | None]:
    row = conn.execute("SELECT last_outcome, last_match_won FROM player_status WHERE tour = ? AND player_id = ?",
                       (tour, player_id)).fetchone()
    if row is None:
        return None, None
    return row[0], None if row[1] is None else bool(row[1])


def _seed(key: str) -> int:
    """A fixed seed per match, so the same scan inputs always give the same probabilities."""
    return zlib.crc32(key.encode())


def simulate_match(model, id_a: int, id_b: int, surface: str, best_of: int, final_tb: int, seed: int):
    a, b = model.players[id_a], model.players[id_b]
    x, serve_total = model.features(id_a, id_b, surface, a.last_points, b.last_points)
    p_bo3 = bo3_probability(x, model.betas)
    serve_a, serve_b = serve_inputs(p_bo3, serve_total)
    return sim.simulate_match(serve_a, serve_b, best_of=best_of, final_set_tb_points=final_tb, seed=seed)


def price_legs(conn: sqlite3.Connection, models: dict, winners: list, totals: list, milestones: dict,
               now: dt.datetime) -> list[LegCandidate]:
    """Turn parsed Kalshi markets into priced, gate-checked legs."""
    surfaces = ingest.build_surface_index(conn)
    by_match: dict[str, list] = {}
    for market in winners:
        by_match.setdefault(market.match_key, []).append(market)
    totals_by_match: dict[str, list] = {}
    for market in totals:
        totals_by_match.setdefault(market.match_key, []).append(market)

    legs: list[LegCandidate] = []
    for key, pair in sorted(by_match.items()):
        first = pair[0]
        tour = first.tour
        milestone = milestones.get(first.event_ticker)
        tournament = first.tournament or (milestone.tournament if milestone else "")
        round_name = first.round_name or (milestone.round_name if milestone else "")
        match_reasons: list[str] = []
        if milestone is None or milestone.start is None:
            match_reasons.append("start time unknown")
        elif milestone.status != "not_started" or milestone.start <= now:
            match_reasons.append("match has started or finished")
        best_of = (milestone.best_of if milestone and milestone.best_of else
                   (5 if tour == "ATP" and kalshi_reader.is_grand_slam(tournament) else 3))
        start = milestone.start if milestone else None
        surface, _tier = ingest.infer_surface(surfaces, tour, tournament, "", (start or now).month)

        ids = {}
        for market in pair:
            hit = ingest.resolve_player(conn, market.player, tour)
            ids[market.player] = hit
        model = models.get(tour)
        result = None
        resolved = all(ids.values()) and model is not None and len(pair) == 2
        if resolved and all(hit.player_id in model.players for hit in ids.values()):
            a, b = pair[0], pair[1]
            result = simulate_match(model, ids[a.player].player_id, ids[b.player].player_id, surface, best_of,
                                    10 if kalshi_reader.is_grand_slam(tournament) else 7, _seed(key))
        for index, market in enumerate(pair):
            hit, opp = ids.get(market.player), ids.get(market.opponent)
            leg = LegCandidate(
                tour=tour, match_key=key, event_ticker=market.event_ticker, market_ticker=market.ticker,
                kind="winner", label=f"{(hit.canonical_name if hit else market.player)} to win", player=market.player,
                opponent=market.opponent, tournament=tournament, round_name=round_name, kalshi_ask=market.yes_ask,
                ask_size=market.ask_size, player_id=hit.player_id if hit else None,
                opponent_id=opp.player_id if opp else None, player_name=hit.canonical_name if hit else None,
                opponent_name=opp.canonical_name if opp else None,
                name_confidence=min(hit.confidence if hit else 0.0, opp.confidence if opp else 0.0),
                surface=surface, best_of=best_of, start_time=start.isoformat() if start else None,
                reasons=list(match_reasons))
            if hit is None or opp is None:
                missing = [name for name, h in ((market.player, hit), (market.opponent, opp)) if h is None]
                leg.reasons.append(f"name below 98% match: {', '.join(missing)}")
            if result is not None:
                leg.simulated_prob = result.p_a_win if index == 0 else result.p_b_win
            elif hit is not None and opp is not None:
                leg.reasons.append("no rating history for a player")
            if leg.kalshi_ask is None or leg.ask_size <= 0:
                leg.reasons.append("no Yes offer to buy")
            if hit is not None:
                leg.last_result, leg.last_result_won = _player_status(conn, tour, hit.player_id)
            if leg.simulated_prob is not None and leg.kalshi_ask is not None:
                leg.edge = leg.simulated_prob - leg.kalshi_ask
                # The locked per-leg checks: name confidence, last-match veto, +5% edge.
                leg.reasons.extend(r.split(": ", 1)[-1] for r in risk_gate.check_leg(leg.to_gate_leg()))
            legs.append(leg)

        for total in sorted(totals_by_match.get(key, []), key=lambda t: t.line):
            p_over = float((result.total_games > total.line).mean()) if result is not None else None
            leg = LegCandidate(
                tour=tour, match_key=key, event_ticker=total.event_ticker, market_ticker=total.ticker, kind="total",
                label=f"Over {total.line:g} games", player=pair[0].player, opponent=pair[0].opponent,
                tournament=tournament, round_name=round_name, kalshi_ask=total.yes_ask, ask_size=total.ask_size,
                surface=surface, best_of=best_of, start_time=start.isoformat() if start else None,
                simulated_prob=p_over, reasons=list(match_reasons))
            if p_over is not None and total.yes_ask is not None:
                leg.edge = p_over - total.yes_ask
            if not TOTALS_ENABLED:
                leg.reasons.append(TOTALS_DISABLED_REASON)
            legs.append(leg)
    return legs


# --------------------------------------------------------------------------- combos

def build_combos(legs: list[LegCandidate], combo_markets: list, limit: int = MAX_COMBOS_EVALUATED) -> list[ComboCandidate]:
    """Every combo of two or more qualifying legs on different matches with a fair hit chance of at least 30%.

    Price: Kalshi's own combo market when one with exactly these legs is open with an offer,
    otherwise the product of the legs' Yes asks. Every combo then goes through the locked gate.
    """
    pool = sorted((leg for leg in legs if leg.qualifies), key=lambda leg: -(leg.simulated_prob or 0.0))
    offers = {}
    for market in combo_markets:
        if market.yes_ask is not None and market.ask_size > 0 and market.status in ("active", "open"):
            best = offers.get(market.legs)
            if best is None or market.yes_ask < best.yes_ask:
                offers[market.legs] = market

    found: list[ComboCandidate] = []
    evaluated = 0

    def extend(start: int, chosen: list[LegCandidate], prob: float, matches: set) -> None:
        nonlocal evaluated
        for index in range(start, len(pool)):
            if evaluated >= limit:
                return
            leg = pool[index]
            if leg.match_key in matches:
                continue
            joint = prob * leg.simulated_prob
            if joint < risk_gate.MIN_COMBO_HIT_PROB - 1e-9:
                break             # the pool is sorted, so every later leg fails too
            combo = chosen + [leg]
            if len(combo) >= 2:
                evaluated += 1
                found.append(price_combo(combo, offers))
            extend(index + 1, combo, joint, matches | {leg.match_key})

    extend(0, [], 1.0, set())
    found.sort(key=lambda c: (not c.approved, -c.edge, -c.fair_combo_prob))
    return found


def price_combo(legs: list[LegCandidate], offers: dict) -> ComboCandidate:
    gate_legs = [leg.to_gate_leg() for leg in legs]
    offer = offers.get(frozenset((leg.market_ticker, "yes") for leg in legs))
    if offer is not None:
        price, source, ticker = offer.yes_ask, PRICE_SOURCE_COMBO, offer.ticker
    else:
        price, source, ticker = math.prod(leg.kalshi_ask for leg in legs), PRICE_SOURCE_PRODUCT, None
    decision = risk_gate.evaluate_combo(gate_legs, price)
    return ComboCandidate(legs=legs, fair_combo_prob=decision.fair_combo_prob, price=price, price_source=source,
                          kalshi_combo_ticker=ticker, edge=decision.combo_edge, approved=decision.approved,
                          reasons=decision.reasons)


# --------------------------------------------------------------------------- ledger

def bankroll(conn: sqlite3.Connection) -> dict:
    settled = conn.execute("SELECT COALESCE(SUM(payout_usd - stake_usd), 0) FROM paper_combos "
                           "WHERE status != 'open'").fetchone()[0]
    at_risk = conn.execute("SELECT COALESCE(SUM(stake_usd), 0) FROM paper_combos WHERE status = 'open'").fetchone()[0]
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM paper_combos GROUP BY status").fetchall())
    balance = START_BANKROLL + settled
    return {"start": START_BANKROLL, "balance": round(balance, 4), "at_risk": round(at_risk, 4),
            "available": round(balance - at_risk, 4), "profit": round(settled, 4), "unit": UNIT_STAKE,
            "open": counts.get("open", 0), "won": counts.get("won", 0), "lost": counts.get("lost", 0),
            "void": counts.get("void", 0)}


def log_paper_trades(conn: sqlite3.Connection, combos: list[ComboCandidate], now: dt.datetime) -> list[int]:
    """Paper-log the best approved combos: no repeats, no leg staked twice while open, $1 each while cash lasts."""
    logged: list[int] = []
    # Legs already riding in an open paper combo are not staked again in a new one.
    used_legs: set[str] = {row[0] for row in conn.execute(
        "SELECT l.market_ticker FROM paper_legs l JOIN paper_combos c USING (combo_id) WHERE c.status = 'open'")}
    known = {row[0] for row in conn.execute("SELECT combo_key FROM paper_combos")}
    for combo in combos:
        if len(logged) >= MAX_NEW_COMBOS_PER_SCAN or not combo.approved:
            continue
        tickers = {leg.market_ticker for leg in combo.legs}
        if combo.key in known or tickers & used_legs:
            continue
        if bankroll(conn)["available"] < UNIT_STAKE - 1e-9:
            break
        cursor = conn.execute(
            "INSERT INTO paper_combos (created_at, slate_date, combo_key, n_legs, fair_combo_prob, kalshi_combo_price, "
            "price_source, kalshi_combo_ticker, edge, stake_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(), local_date(now).isoformat(), combo.key, len(combo.legs), combo.fair_combo_prob, combo.price,
             combo.price_source, combo.kalshi_combo_ticker, combo.edge, UNIT_STAKE))
        combo_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO paper_legs (combo_id, tour, match_key, market_ticker, kind, label, player_id, opponent_id, "
            "player_name, opponent_name, surface, start_time, simulated_prob, kalshi_ask) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(combo_id, leg.tour, leg.match_key, leg.market_ticker, leg.kind, leg.label, leg.player_id,
              leg.opponent_id, leg.player_name, leg.opponent_name, leg.surface, leg.start_time, leg.simulated_prob,
              leg.kalshi_ask) for leg in combo.legs])
        logged.append(combo_id)
        used_legs |= tickers
        known.add(combo.key)
    conn.commit()
    return logged


def settle(conn: sqlite3.Connection, fetch=kalshi_reader.get_json, now: dt.datetime | None = None) -> dict:
    """Grade open paper trades from Kalshi's own settlement of each market.

    A combo loses as soon as one leg resolves No, wins when every leg resolves Yes, and is
    void (stake returned) if a leg was resolved to a fair price, such as a walkover.
    A combo priced from a Kalshi combo market is graded by that market when it has settled.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.isoformat()
    summary = {"checked": 0, "won": 0, "lost": 0, "void": 0, "still_open": 0, "errors": []}
    cache: dict[str, str | None] = {}

    def outcome(ticker: str) -> str | None:
        if ticker not in cache:
            try:
                cache[ticker] = kalshi_reader.settlement(kalshi_reader.get_market(ticker, fetch))
            except kalshi_reader.KalshiError as exc:
                summary["errors"].append(str(exc))
                cache[ticker] = None
        return cache[ticker]

    open_combos = conn.execute("SELECT combo_id, kalshi_combo_ticker, kalshi_combo_price, stake_usd FROM paper_combos "
                               "WHERE status = 'open'").fetchall()
    for combo_id, combo_ticker, combo_price, stake in open_combos:
        summary["checked"] += 1
        legs = conn.execute("SELECT leg_id, market_ticker, result FROM paper_legs WHERE combo_id = ?",
                            (combo_id,)).fetchall()
        results = []
        for leg_id, ticker, result in legs:
            if result is None:
                result = outcome(ticker)
                if result is not None:
                    conn.execute("UPDATE paper_legs SET result = ?, settled_at = ? WHERE leg_id = ?",
                                 (result, stamp, leg_id))
            results.append(result)
        status = None
        if combo_ticker:
            status = {"yes": "won", "no": "lost", "void": "void"}.get(outcome(combo_ticker))
        if status is None:
            if "no" in results:
                status = "lost"
            elif all(r is not None for r in results):
                status = "won" if all(r == "yes" for r in results) else "void"
        if status is None:
            summary["still_open"] += 1
            continue
        payout = {"won": stake / combo_price, "lost": 0.0, "void": stake}[status]
        conn.execute("UPDATE paper_combos SET status = ?, payout_usd = ?, settled_at = ? WHERE combo_id = ?",
                     (status, round(payout, 4), stamp, combo_id))
        summary[status] += 1
    conn.commit()
    return summary


def ledger(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    combos = conn.execute(
        "SELECT combo_id, created_at, slate_date, n_legs, fair_combo_prob, kalshi_combo_price, price_source, "
        "kalshi_combo_ticker, edge, stake_usd, status, payout_usd, settled_at FROM paper_combos "
        "ORDER BY combo_id DESC LIMIT ?", (limit,)).fetchall()
    names = ("combo_id", "created_at", "slate_date", "n_legs", "fair_combo_prob", "kalshi_combo_price", "price_source",
             "kalshi_combo_ticker", "edge", "stake_usd", "status", "payout_usd", "settled_at")
    out = []
    for row in combos:
        item = dict(zip(names, row))
        item["legs"] = [dict(zip(("tour", "label", "player_name", "opponent_name", "market_ticker", "surface",
                                  "start_time", "simulated_prob", "kalshi_ask", "result"), leg)) for leg in conn.execute(
            "SELECT tour, label, player_name, opponent_name, market_ticker, surface, start_time, simulated_prob, "
            "kalshi_ask, result FROM paper_legs WHERE combo_id = ? ORDER BY leg_id", (row[0],))]
        out.append(item)
    return out


# --------------------------------------------------------------------------- scan

def fetch_markets(fetch=kalshi_reader.get_json, log=print) -> dict:
    """Every open tennis market the scanner uses, plus each match's milestone."""
    winners, totals, combos, warnings = [], [], [], []
    for tour, series in kalshi_reader.MATCH_SERIES.items():
        log(f"Reading open {tour} match markets")
        winners += kalshi_reader.parse_winner_markets(kalshi_reader.list_markets(series, fetch=fetch), tour)
    for tour, series in kalshi_reader.TOTAL_SERIES.items():
        try:
            totals += kalshi_reader.parse_total_markets(kalshi_reader.list_markets(series, fetch=fetch), tour)
        except kalshi_reader.KalshiError as exc:
            warnings.append(f"total-games markets for {tour} unavailable: {exc}")
    tennis = {market.ticker for market in winners} | {market.ticker for market in totals}
    log("Checking Kalshi's existing combo markets")
    seen: set[str] = set()
    for series in kalshi_reader.COMBO_SERIES:
        try:
            for combo in kalshi_reader.parse_combo_markets(kalshi_reader.list_markets(series, fetch=fetch, max_pages=3)):
                if all(ticker in tennis for ticker, _side in combo.legs) and combo.ticker not in seen:
                    combos.append(combo)
                    seen.add(combo.ticker)
        except kalshi_reader.KalshiError as exc:
            warnings.append(f"combo markets in {series} unavailable, using leg products: {exc}")
    milestones = {}
    events = sorted({market.event_ticker for market in winners})
    log(f"Checking start times for {len(events)} matches")
    for event in events:
        try:
            milestones[event] = kalshi_reader.parse_milestone(kalshi_reader.get_milestone(event, fetch), event)
        except kalshi_reader.KalshiError as exc:
            warnings.append(f"start time for {event} unavailable: {exc}")
            milestones[event] = None
    return {"winners": winners, "totals": totals, "combos": combos, "milestones": milestones, "warnings": warnings}


def scan(conn: sqlite3.Connection, models: dict, fetch=kalshi_reader.get_json, now: dt.datetime | None = None,
         log=print, paper_trade: bool = True) -> dict:
    """One full scan. Returns the payload the dashboard and the daily slate note show."""
    now = now or dt.datetime.now(dt.timezone.utc)
    init_ledger(conn)
    gate = gate_status(conn)
    if not gate["open"]:
        raise GateClosed(gate["reason"] + " Live scanning stays off until it passes.")
    markets = fetch_markets(fetch, log)
    log("Simulating matches")
    legs = price_legs(conn, models, markets["winners"], markets["totals"], markets["milestones"], now)
    log("Building combos")
    combos = build_combos(legs, markets["combos"])
    logged = log_paper_trades(conn, combos, now) if paper_trade else []
    logged_keys = {row[0] for row in conn.execute(
        f"SELECT combo_key FROM paper_combos WHERE combo_id IN ({','.join('?' * len(logged))})", logged)} if logged else set()

    def leg_json(leg: LegCandidate) -> dict:
        item = asdict(leg)
        item["qualifies"] = leg.qualifies
        return item

    approved = [c for c in combos if c.approved]
    payload = {
        "created_at": now.isoformat(), "slate_date": local_date(now).isoformat(), "gate": gate,
        "warnings": markets["warnings"],
        "counts": {"matches": len({leg.match_key for leg in legs}), "legs": len(legs),
                   "qualifying_legs": sum(leg.qualifies for leg in legs), "combos_checked": len(combos),
                   "combos_approved": len(approved), "kalshi_combo_markets": len(markets["combos"]),
                   "paper_trades_logged": len(logged)},
        "legs": [leg_json(leg) for leg in sorted(legs, key=lambda l: (not l.qualifies, -(l.edge or -9)))],
        "combos": [{"legs": [leg_json(leg) for leg in c.legs], "fair_combo_prob": c.fair_combo_prob, "price": c.price,
                    "price_source": c.price_source, "kalshi_combo_ticker": c.kalshi_combo_ticker, "edge": c.edge,
                    "approved": c.approved, "reasons": c.reasons, "logged": c.key in logged_keys}
                   for c in (approved[:SHOW_COMBOS] or combos[:5])],
        "logged_combo_ids": logged,
    }
    conn.execute("INSERT INTO kalshi_scans (created_at, payload) VALUES (?, ?)", (now.isoformat(), json.dumps(payload)))
    conn.commit()
    return payload


def latest_scan(conn: sqlite3.Connection) -> dict | None:
    init_ledger(conn)
    row = conn.execute("SELECT payload FROM kalshi_scans ORDER BY scan_id DESC LIMIT 1").fetchone()
    return json.loads(row[0]) if row else None
