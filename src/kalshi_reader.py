"""Read-only client for Kalshi's public market data.

Every request in this module is an HTTP GET to the public trade API. There is no
code path that places, amends or cancels an order, and none that requests a quote.
tests/test_risk_gate.py fails if a write call ever appears under src/.

Endpoints used:
    GET /markets?series_ticker=...&status=open     open markets in a series (paginated)
    GET /markets/{ticker}                           one market, used to settle paper trades
    GET /milestones?related_event_ticker=...        a match's start time, status and best-of
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MATCH_SERIES = {"ATP": "KXATPMATCH", "WTA": "KXWTAMATCH"}
TOTAL_SERIES = {"ATP": "KXATPGTOTAL", "WTA": "KXWTAGTOTAL"}
# Kalshi's combo collections that accept tennis match legs. A combo market only exists here
# after someone has requested it, so most combos will not have one.
COMBO_SERIES = ("KXMVESPORTSMULTIGAMEEXTENDED", "KXMVECROSSCATEGORY", "KXMVECROSSCATEGORY-SHARD1")
PAGE_LIMIT = 1000
MAX_PAGES = 10
GRAND_SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")

_RULE = re.compile(r" vs (?P<b>.+?) professional tennis match in the (?:\d{4} )?(?P<tournament>.+?) "
                   r"(?:Men|Women)(?:'s)? Singles(?: (?P<round>.+?))?(?: after a ball has been played)?, then", re.I)


class KalshiError(RuntimeError):
    pass


def get_json(path: str, params: dict | None = None, timeout: int = 30) -> dict:
    """GET one public endpoint and return its JSON body."""
    url = BASE_URL + path + ("?" + urllib.parse.urlencode(params) if params else "")
    request = urllib.request.Request(url, method="GET", headers={
        "Accept": "application/json", "User-Agent": "CourtEdge read-only scanner"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, ValueError) as exc:
        raise KalshiError(f"GET {path} failed: {exc}") from exc


# --------------------------------------------------------------------------- requests

def list_markets(series_ticker: str, status: str | None = "open", fetch=get_json,
                 max_pages: int = MAX_PAGES) -> list[dict]:
    """Every market in a series, following the cursor."""
    markets, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series_ticker, "limit": PAGE_LIMIT}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        payload = fetch("/markets", params)
        markets.extend(payload.get("markets") or [])
        cursor = payload.get("cursor")
        if not cursor:
            break
    return markets


def get_market(ticker: str, fetch=get_json) -> dict:
    return (fetch(f"/markets/{urllib.parse.quote(ticker, safe='-_')}", None) or {}).get("market") or {}


def get_milestone(event_ticker: str, fetch=get_json) -> dict | None:
    payload = fetch("/milestones", {"related_event_ticker": event_ticker, "category": "Sports", "limit": 5})
    for milestone in payload.get("milestones") or []:
        details = milestone.get("details") or {}
        if details.get("main_game_event_ticker") == event_ticker or event_ticker in (milestone.get("primary_event_tickers") or []):
            return milestone
    return None


# --------------------------------------------------------------------------- parsing

def price(value) -> float | None:
    """A Kalshi dollar string as a probability. None when there is no usable offer."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 < number < 1.0 else None


def size(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def match_key(event_ticker: str, tour: str) -> str:
    """'KXATPGTOTAL-26NOV16MEDCOB' and 'KXATPMATCH-26NOV16MEDCOB' are the same match: 'ATP:26NOV16MEDCOB'."""
    return f"{tour}:{event_ticker.split('-', 1)[1] if '-' in event_ticker else event_ticker}"


def parse_rules(rules: str) -> tuple[str, str]:
    """Tournament and round from a market's rules text."""
    found = _RULE.search(rules or "")
    if not found:
        return "", ""
    return found.group("tournament").strip(), (found.group("round") or "").strip()


@dataclass(frozen=True)
class WinnerMarket:
    ticker: str
    event_ticker: str
    match_key: str
    tour: str
    player: str            # the player this market pays out on
    opponent: str
    yes_ask: float | None
    ask_size: float
    tournament: str
    round_name: str
    status: str


@dataclass(frozen=True)
class TotalMarket:
    ticker: str
    event_ticker: str
    match_key: str
    tour: str
    line: float            # pays Yes when total games played is above this
    yes_ask: float | None
    ask_size: float
    status: str


@dataclass(frozen=True)
class Milestone:
    event_ticker: str
    start: dt.datetime | None
    status: str            # Kalshi uses 'not_started' before the first ball
    best_of: int | None
    tournament: str
    round_name: str


@dataclass(frozen=True)
class ComboMarket:
    ticker: str
    legs: frozenset         # {(market_ticker, side)}
    yes_ask: float | None
    ask_size: float
    status: str


def parse_winner_markets(markets: list[dict], tour: str) -> list[WinnerMarket]:
    """Match-winner markets. Each event has one market per player; the other one names the opponent."""
    by_event: dict[str, list[dict]] = {}
    for market in markets:
        if market.get("event_ticker") and market.get("yes_sub_title"):
            by_event.setdefault(market["event_ticker"], []).append(market)
    parsed = []
    for event, pair in by_event.items():
        if len(pair) != 2:
            continue
        for market, other in ((pair[0], pair[1]), (pair[1], pair[0])):
            tournament, round_name = parse_rules(market.get("rules_primary", ""))
            parsed.append(WinnerMarket(
                ticker=market["ticker"], event_ticker=event, match_key=match_key(event, tour), tour=tour,
                player=market["yes_sub_title"].strip(), opponent=other["yes_sub_title"].strip(),
                yes_ask=price(market.get("yes_ask_dollars")), ask_size=size(market.get("yes_ask_size_fp")),
                tournament=tournament, round_name=round_name, status=market.get("status", "")))
    return parsed


def parse_total_markets(markets: list[dict], tour: str) -> list[TotalMarket]:
    parsed = []
    for market in markets:
        try:
            line = float(market["floor_strike"])
        except (KeyError, TypeError, ValueError):
            continue
        if market.get("strike_type") not in (None, "greater"):
            continue
        parsed.append(TotalMarket(
            ticker=market["ticker"], event_ticker=market.get("event_ticker", ""),
            match_key=match_key(market.get("event_ticker", ""), tour), tour=tour, line=line,
            yes_ask=price(market.get("yes_ask_dollars")), ask_size=size(market.get("yes_ask_size_fp")),
            status=market.get("status", "")))
    return parsed


def parse_milestone(milestone: dict | None, event_ticker: str) -> Milestone | None:
    if not milestone:
        return None
    details = milestone.get("details") or {}
    try:
        start = dt.datetime.fromisoformat(str(milestone.get("start_date")).replace("Z", "+00:00"))
    except ValueError:
        start = None
    try:
        best_of = int(details.get("best_of"))
    except (TypeError, ValueError):
        best_of = None
    return Milestone(event_ticker=event_ticker, start=start, status=str(details.get("status") or ""),
                     best_of=best_of if best_of in (3, 5) else None,
                     tournament=str(details.get("tournament_name") or ""), round_name=str(details.get("round") or ""))


def parse_combo_markets(markets: list[dict]) -> list[ComboMarket]:
    parsed = []
    for market in markets:
        legs = market.get("mve_selected_legs")
        if isinstance(legs, str):
            try:
                legs = json.loads(legs)
            except ValueError:
                continue
        if not legs:
            continue
        try:
            leg_set = frozenset((leg["market_ticker"], leg.get("side", "yes")) for leg in legs)
        except (KeyError, TypeError):
            continue
        parsed.append(ComboMarket(ticker=market.get("ticker", ""), legs=leg_set,
                                  yes_ask=price(market.get("yes_ask_dollars")),
                                  ask_size=size(market.get("yes_ask_size_fp")), status=market.get("status", "")))
    return parsed


def settlement(market: dict) -> str | None:
    """'yes', 'no' or 'void' once a market has settled; None while it is still open.

    'void' covers markets Kalshi resolved to a fair price, for example a walkover
    before a ball was played.
    """
    if not market or market.get("status") not in ("finalized", "settled", "determined"):
        return None
    result = str(market.get("result") or "").lower()
    if result in ("yes", "no"):
        return result
    value = market.get("settlement_value_dollars")
    if value in (None, ""):
        return None
    return {1.0: "yes", 0.0: "no"}.get(round(size(value), 4), "void")


def is_grand_slam(tournament: str) -> bool:
    name = tournament.lower()
    return any(slam in name for slam in GRAND_SLAMS)
