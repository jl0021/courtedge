import inspect
import json
import re
from pathlib import Path

import pytest

from src import kalshi_reader as k

FIXTURES = Path(__file__).parent / "fixtures"
SRC = Path(k.__file__).parent


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_winner_markets_from_a_real_listing():
    markets = k.parse_winner_markets(load("kalshi_open_markets.json")["markets"], "ATP")
    assert len(markets) == 8
    alcaraz = next(m for m in markets if m.player == "Carlos Alcaraz")
    assert alcaraz.opponent == "Daniil Medvedev" and alcaraz.match_key == "ATP:26NOV17ALCMED"
    assert alcaraz.yes_ask == pytest.approx(0.77) and alcaraz.ask_size == pytest.approx(1.0)
    assert (alcaraz.tournament, alcaraz.round_name) == ("ATP World Tour Finals", "Ivan Lendl Group")
    assert {m.match_key for m in markets} == {"ATP:26NOV17ALCMED", "ATP:26NOV17COBTIA", "ATP:26NOV17ZVESHE",
                                              "ATP:26NOV17FILJOD"}


def test_winner_markets_need_both_players():
    lone = load("kalshi_open_markets.json")["markets"][:1]
    assert k.parse_winner_markets(lone, "ATP") == []


def test_totals_and_match_key_link_to_the_winner_event():
    total = k.parse_total_markets(load("kalshi_totals_markets.json")["markets"], "ATP")[0]
    assert total.line == 22.5 and total.match_key == "ATP:26NOV16MEDCOB"
    assert total.yes_ask is None                      # settled: an ask of $1.00 is no offer
    assert k.match_key("KXATPMATCH-26NOV16MEDCOB", "ATP") == total.match_key


def test_milestone_gives_start_status_and_format():
    event = "KXATPMATCH-26NOV17ALCMED"
    milestone = k.parse_milestone(load("kalshi_milestone.json")["milestones"][0], event)
    assert milestone.status == "not_started" and milestone.best_of == 3
    assert milestone.start.isoformat() == "2026-11-18T13:00:00+00:00"
    assert k.parse_milestone(None, event) is None


def test_combo_markets_and_prices():
    combo = k.parse_combo_markets(load("kalshi_combo_markets.json")["markets"])[0]
    assert combo.legs == frozenset({("KXATPMATCH-26NOV17ZVESHE-ZVE", "yes"), ("KXATPMATCH-26NOV17FILJOD-JOD", "yes")})
    assert combo.yes_ask == pytest.approx(0.315)
    as_string = dict(load("kalshi_combo_markets.json")["markets"][0])
    as_string["mve_selected_legs"] = json.dumps(as_string["mve_selected_legs"])   # the API sometimes sends a string
    assert k.parse_combo_markets([as_string])[0].legs == combo.legs
    assert [k.price(v) for v in ("0.4500", "1.0000", "0.0000", None, "x")] == [0.45, None, None, None, None]


@pytest.mark.parametrize("market,expected", [
    ({"status": "finalized", "result": "yes"}, "yes"),
    ({"status": "finalized", "result": "no"}, "no"),
    ({"status": "finalized", "result": "", "settlement_value_dollars": "0.6200"}, "void"),
    ({"status": "finalized", "result": "", "settlement_value_dollars": "1.0000"}, "yes"),
    ({"status": "active", "result": ""}, None),
    ({}, None),
])
def test_settlement(market, expected):
    assert k.settlement(market) == expected


def test_settlement_of_a_real_market():
    assert k.settlement(load("kalshi_settled_market.json")["market"]) == "yes"


def test_rules_parsing_and_grand_slams():
    assert k.parse_rules("If X wins the Sinner vs Zverev professional tennis match in the 2026 Wimbledon Men's Singles "
                         "Final after a ball has been played, then the market resolves to Yes.") == ("Wimbledon", "Final")
    assert k.parse_rules("nothing useful") == ("", "")
    assert k.is_grand_slam("Roland Garros") and k.is_grand_slam("US Open") and not k.is_grand_slam("ATP World Tour Finals")


def test_list_markets_follows_the_cursor():
    pages = {None: {"markets": [{"ticker": "A"}], "cursor": "next"}, "next": {"markets": [{"ticker": "B"}], "cursor": ""}}
    calls = []

    def fetch(path, params):
        calls.append((path, dict(params)))
        return pages[params.get("cursor")]
    assert [m["ticker"] for m in k.list_markets("KXATPMATCH", fetch=fetch)] == ["A", "B"]
    assert calls[0] == ("/markets", {"series_ticker": "KXATPMATCH", "limit": k.PAGE_LIMIT, "status": "open"})


def test_get_json_sends_only_get_requests(monkeypatch):
    seen = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self): return b'{"markets": []}'

    def fake_urlopen(request, timeout=None):
        seen.append((request.get_method(), request.full_url, request.data))
        return Response()
    monkeypatch.setattr(k.urllib.request, "urlopen", fake_urlopen)
    k.list_markets("KXWTAMATCH")
    k.get_market("KXATPMATCH-26NOV16ALCTIA-ALC")
    k.get_milestone("KXATPMATCH-26NOV17ALCMED")
    assert [method for method, _, _ in seen] == ["GET", "GET", "GET"]
    assert all(data is None for _, _, data in seen)
    assert all(url.startswith(k.BASE_URL) for _, url, _ in seen)


def test_get_json_wraps_network_errors(monkeypatch):
    def offline(request, timeout=None):
        raise OSError("no route")
    monkeypatch.setattr(k.urllib.request, "urlopen", offline)
    with pytest.raises(k.KalshiError, match="no route"):
        k.get_json("/markets", {"series_ticker": "KXATPMATCH"})


def test_only_the_reader_talks_to_kalshi():
    host = "kalshi.com"
    for path in list(SRC.glob("*.py")) + [SRC / "web" / "index.html"]:
        if path.name != "kalshi_reader.py":
            assert host not in path.read_text().lower(), f"{path.name} reaches Kalshi directly"
    source = inspect.getsource(k)
    assert not re.search(r"method\s*=\s*['\"](POST|PUT|DELETE|PATCH)", source, re.I)
    assert "/portfolio" not in source and "/orders" not in source and "/communications" not in source
