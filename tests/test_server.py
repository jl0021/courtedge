import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from src import server
from tests.test_ingest import HEADER, row


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """A small two-season archive: 12 players per tour with a hidden skill order."""
    root = tmp_path_factory.mktemp("archive")
    rng = np.random.default_rng(5)
    for tour in ("atp", "wta"):
        (root / tour).mkdir()
        lines = {2024: [HEADER], 2025: [HEADER]}
        for week in range(60):
            year = 2024 if week < 30 else 2025
            date = f"{year}{1 + (week % 30) // 3:02d}{1 + (week % 3) * 7:02d}"
            for n in range(10):
                a, b = rng.choice(12, 2, replace=False) + 1
                winner, loser = (a, b) if rng.random() < 1 / (1 + np.exp((a - b) / 5)) else (b, a)
                score = "6-2 1-0 RET" if (week, n) == (59, 9) else "6-4 6-4"
                surface = ("Hard", "Clay", "Grass")[week % 3]
                lines[year].append(row(f"{year}-{week:03d}", date, n + 1, winner, f"First{winner} Last{winner}",
                                       loser, f"First{loser} Last{loser}", score, "R32", surface=surface))
        for year, rows in lines.items():
            (root / tour / f"{tour}_matches_{year}.csv").write_text("\n".join(rows) + "\n")
        (root / tour / f"{tour}_players.csv").write_text(
            "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
            + "".join(f"{i},First{i},Last{i},R,,USA,185,\n" for i in range(1, 13)))
    application = server.App(db_path=root / "test.db", archive=root, write_vault=False)
    assert not application.has_data()
    assert application.status()["ready"] is False
    application.fetch = lambda league, start, end: {"events": []}      # no network in tests
    assert application.start_job("setup")
    assert not application.start_job("backtest")            # one job at a time
    deadline = time.time() + 60
    while application.job["state"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert application.job["state"] == "done", application.job
    return application


@pytest.fixture(scope="module")
def base_url(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def send(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_status_and_performance(app):
    status = app.status()
    assert status["ready"] and 1100 < status["data"]["matches"] <= 1200 and status["data"]["players"] == 24
    n_scored = status["run"]["n_matches"]
    assert status["run"]["engine"] == "analytic" and 550 < n_scored <= 600
    assert status["run"]["passed"] == (status["run"]["brier_passed"] and status["run"]["calibration_passed"])
    assert status["limits"] == {"brier": 0.215, "calibration_gap": 0.035}
    perf = app.performance()
    assert {r["tour"] for r in perf["tables"]["by_tour"]} == {"ATP", "WTA"}
    assert sum(r["n"] for r in perf["tables"]["calibration"]) == n_scored
    assert perf["tables"]["lower"] == []
    assert perf["run"]["brier"] == pytest.approx(status["run"]["brier"])
    assert app.performance() is perf                          # cached per run


def test_players_and_search(app):
    players = app.players("ATP", "All")
    assert len(players) == 12 and players[0]["position"] == 1
    assert [p["elo"] for p in players] == sorted((p["elo"] for p in players), reverse=True)
    assert app.players("ATP", "All", "Last12")[0]["full_name"] == "First12 Last12"
    assert app.search("WTA", "last1")[0]["full_name"].startswith("First1")
    assert app.search("ATP", "x") == []
    flagged = [p for p in app.players("WTA", "All") if p["vetoed"]]
    assert len(flagged) == 1 and flagged[0]["last_outcome"] == "RET" and flagged[0]["last_match_won"] == 0


def test_simulate(app):
    result = app.simulate("ATP", 1, 12, "Hard", 3)
    assert result["a"]["name"] == "First1 Last1" and result["trials"] == 10_000
    assert result["p_a"] > 0.6                                # player 1 is the strongest
    assert result["p_a"] + result["p_b"] == pytest.approx(1.0)
    assert sum(result["set_scores"].values()) == pytest.approx(1.0)
    assert sum(g["prob"] for g in result["total_games"]) == pytest.approx(1.0)
    assert app.simulate("ATP", 1, 12, "Hard", 5)["p_a"] > result["p_a"] - 0.02
    assert app.simulate("ATP", 1, 12, "Hard", 3)["p_a"] == result["p_a"]          # repeatable
    assert app.simulate("ATP", 12, 1, "Hard", 3)["p_a"] == pytest.approx(result["p_b"], abs=0.03)
    for bad in (("ATP", 1, 1, "Hard", 3), ("ATP", 1, 999, "Hard", 3), ("XYZ", 1, 2, "Hard", 3), ("ATP", 1, 2, "Ice", 3)):
        with pytest.raises(ValueError):
            app.simulate(*bad)


def test_http_routes(app, base_url):
    status, content_type, body = get(base_url + "/")
    assert status == 200 and content_type.startswith("text/html") and b"CourtEdge" in body
    assert json.loads(get(base_url + "/api/status")[2])["ready"] is True
    assert json.loads(get(base_url + "/api/guardrails")[2])["MIN_COMBO_HIT_PROB"] == 0.30
    assert len(json.loads(get(base_url + "/api/players?tour=WTA&surface=Clay")[2])) > 0
    code, payload = send(base_url + "/api/simulate", {"tour": "WTA", "a": 2, "b": 9, "surface": "Clay", "best_of": 3})
    assert code == 200 and 0 < payload["p_a"] < 1
    code, payload = send(base_url + "/api/simulate", {"tour": "WTA", "a": 2, "b": 2, "surface": "Clay", "best_of": 3})
    assert code == 400 and "different" in payload["error"]
    assert send(base_url + "/api/jobs/place-order", {})[0] == 404
    with pytest.raises(urllib.error.HTTPError):
        get(base_url + "/api/nope")
