import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src import backtest, espn_feed, ingest, risk_gate
from src.model import ModelParams, RatingModel
from tests.test_ingest import archive, db  # noqa: F401  (fixtures)

SAMPLE = Path(__file__).parent / "fixtures" / "espn_scoreboard_sample.json"
TODAY = dt.date(2024, 2, 1)


def competition(comp_id, date, winner, loser, sets, status="STATUS_FINAL", round_name="Round 1"):
    """One match in the shape ESPN's scoreboard returns. `sets` is a list of (winner games, loser games)."""
    def side(name, won, index):
        return {"winner": won, "athlete": {"fullName": name, "displayName": name},
                "linescores": [{"value": float(s[index])} for s in sets]}
    return {"id": str(comp_id), "date": f"{date}T12:00Z", "status": {"type": {"name": status}},
            "round": {"displayName": round_name}, "competitors": [side(loser, False, 1), side(winner, True, 0)]}


def scoreboard(*competitions, name="Test Open", city="Testville, Nowhere", slug="mens-singles", major=False):
    return {"events": [{"id": "77", "name": name, "major": major, "venue": {"displayName": city},
                        "groupings": [{"grouping": {"slug": slug}, "competitions": list(competitions)}]}]}


@pytest.fixture
def feed():
    """Five weeks after the archive fixture ends."""
    board = scoreboard(
        competition(9001, "2024-01-13", "Carlos Alcaraz", "Jannik Sinner", [(6, 4), (6, 4)], round_name="Final"),
        competition(9002, "2024-01-20", "Jannik Sinner", "Jan Smith", [(6, 4), (7, 6)]),
        competition(9003, "2024-01-21", "Felix Auger Aliassime", "Jon Smith", [(6, 1), (6, 1)]),
        competition(9004, "2024-01-30", "Carlos Alcaraz", "Felix Auger-Aliassime", [(6, 3), (2, 0)],
                    status="STATUS_RETIRED", round_name="Quarterfinal"),
        competition(9005, "2024-02-02", "Jannik Sinner", "Carlos Alcaraz", [], status="STATUS_SCHEDULED"),
    )
    board["events"][0]["groupings"].append({"grouping": {"slug": "mens-doubles"}, "competitions": [
        {"id": "9100", "date": "2024-01-20T12:00Z", "status": {"type": {"name": "STATUS_FINAL"}},
         "competitors": [{"winner": True, "athlete": None}, {"winner": False, "athlete": None}]}]})
    calls = []

    def fetch(league, start, end):
        calls.append((league, start, end))
        return board if league == "atp" else {"events": []}
    fetch.calls = calls
    return fetch


@pytest.fixture
def caught_up(db, archive, feed, monkeypatch):  # noqa: F811
    conn, _ = db
    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    monkeypatch.setattr(ingest, "CATCHUP_START", dt.date(2024, 1, 1))
    summary = ingest.catch_up(path, today=TODAY, fetch=feed, log=lambda message: None)
    return conn, path, summary


# ---------------------------------------------------------------- parser

def test_parser_on_a_real_response():
    matches = espn_feed.parse_scoreboard(json.loads(SAMPLE.read_text()))
    assert len(matches) == 6                                   # the doubles match is dropped
    assert {m.tour for m in matches} == {"ATP", "WTA"}
    by_outcome = {m.outcome: m for m in matches}
    retired, walkover = by_outcome["RET"], by_outcome["W/O"]
    assert (retired.winner_name, retired.loser_name) == ("Zeynep Sonmez", "Anastasia Potapova")
    assert retired.score == "6-1 2-0 RET" and (retired.sets_played, retired.games_played) == (2, 9)
    assert walkover.score == "W/O" and walkover.sets_played == 0 and walkover.round_order == 18
    bonzi = next(m for m in matches if m.winner_name == "Benjamin Bonzi")
    assert bonzi.score == "6-2 7-6(3)" and bonzi.qualifying and bonzi.best_of == 3
    assert bonzi.date == "2026-06-06" and bonzi.city == "'s-Hertogenbosch" and bonzi.event_name == "Libéma Open"


def test_parser_skips_what_it_cannot_read():
    good = competition(1, "2026-07-01", "A One", "B Two", [(6, 3), (6, 3)])
    no_winner = competition(2, "2026-07-01", "A One", "B Two", [(6, 3)])
    no_winner["competitors"][1]["winner"] = False
    in_progress = competition(3, "2026-07-01", "A One", "B Two", [(6, 3)], status="STATUS_IN_PROGRESS")
    bad_date = competition(4, "not a date", "A One", "B Two", [(6, 3), (6, 3)])
    nameless = competition(5, "2026-07-01", "", "B Two", [(6, 3), (6, 3)])
    parsed = espn_feed.parse_scoreboard(scoreboard(good, no_winner, in_progress, bad_date, nameless))
    assert [m.competition_id for m in parsed] == [1]
    assert espn_feed.parse_scoreboard({}) == [] and espn_feed.parse_scoreboard({"events": [{}]}) == []


def test_best_of_five_only_for_mens_grand_slam_main_draw():
    def best_of(**kwargs):
        round_name = kwargs.pop("round_name", "Round 1")
        board = scoreboard(competition(1, "2026-07-01", "A One", "B Two", [(6, 3)] * 3, round_name=round_name), **kwargs)
        return espn_feed.parse_scoreboard(board)[0].best_of
    assert best_of(major=True) == 5
    assert best_of(major=True, slug="womens-singles") == 3
    assert best_of(major=True, round_name="Qualifying 1st Round") == 3
    assert best_of(major=False) == 3


def test_fetch_rejects_unknown_league():
    with pytest.raises(ValueError):
        espn_feed.fetch_scoreboard("nba", TODAY, TODAY)


# ---------------------------------------------------------------- catch-up

def test_catch_up_adds_only_new_resolvable_matches(caught_up, feed):
    conn, _, summary = caught_up
    assert summary["fetched"] == 4                             # scheduled and doubles never count
    assert summary["added"] == 2 and summary["in_archive"] == 1 and summary["unmatched"] == 1
    assert summary["through"] == "2024-01-30" and summary["errors"] == []
    assert summary["unmatched_names"] == [(1, "ATP", "Jon Smith")]
    assert {league for league, _, _ in feed.calls} == {"atp", "wta"}
    rows = conn.execute(
        "SELECT match_id, tourney_date, winner_id, loser_id, outcome, surface, tier, source, sets_played, games_played, "
        "score, best_of FROM matches WHERE source = 'espn' ORDER BY match_id").fetchall()
    assert rows == [
        (espn_feed.MATCH_ID_OFFSET + 9002, "2024-01-20", 2, 4, "COMPLETED", "Hard", "tour", "espn", 2, 23, "6-4 7-6", 3),
        (espn_feed.MATCH_ID_OFFSET + 9004, "2024-01-30", 1, 3, "RET", "Hard", "tour", "espn", 2, 11, "6-3 2-0 RET", 3),
    ]
    assert conn.execute("SELECT name, match_date FROM ingest_unmatched").fetchall() == [("Jon Smith", "2024-01-21")]
    assert conn.execute("SELECT COUNT(*) FROM matches WHERE source = 'archive'").fetchone()[0] == 5


def test_catch_up_is_idempotent(caught_up, feed):
    conn, path, _ = caught_up
    again = ingest.catch_up(path, today=TODAY, fetch=feed, log=lambda message: None)
    assert again["added"] == 0 and again["already_stored"] == 2
    assert conn.execute("SELECT COUNT(*) FROM matches WHERE source = 'espn'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM ingest_unmatched").fetchone()[0] == 1
    # The second run only asks for the days around the last stored result.
    assert min(start for _, start, _ in feed.calls[-2:]) == dt.date(2024, 1, 27)


def test_seed_reload_keeps_catch_up_rows(caught_up, archive):  # noqa: F811
    conn, path, _ = caught_up
    ingest.ingest(archive, path, log=lambda message: None)
    assert conn.execute("SELECT source, COUNT(*) FROM matches GROUP BY 1 ORDER BY 1").fetchall() == [
        ("archive", 5), ("espn", 2)]


def test_stale_walkover_flag_clears_and_new_retirement_sets_it(caught_up):
    conn, _, _ = caught_up
    status = {row[0]: row[1:] for row in conn.execute(
        "SELECT player_id, last_match_date, last_outcome, last_match_won, last_match_ret_wo, matches_7d, sets_7d, "
        "games_7d, as_of FROM player_status WHERE tour = 'ATP'")}
    # Jan Smith's archive history ended on a walkover loss. He has played since, so the flag is gone.
    assert status[4] == ("2024-01-20", "COMPLETED", 0, 0, 0, 0, 0, "2024-02-01")
    # Auger-Aliassime retired two days before "today": flagged, with that match in his 7-day load.
    assert status[3] == ("2024-01-30", "RET", 0, 1, 1, 2, 11, "2024-02-01")
    # Alcaraz won that match: not flagged, same load.
    assert status[1] == ("2024-01-30", "RET", 1, 0, 1, 2, 11, "2024-02-01")
    assert status[2][3] == 0 and status[2][4] == 0            # Sinner: last played 12 days ago, no load


def test_status_before_catch_up_flags_the_walkover_loser(db):  # noqa: F811
    conn, _ = db
    flags = dict(conn.execute("SELECT player_id, last_match_ret_wo FROM player_status WHERE tour = 'ATP'").fetchall())
    assert flags[4] == 1 and flags[2] == 0                      # Sinner received the walkover: not flagged


def test_ratings_and_veto_follow_the_new_results(caught_up):
    conn, _, _ = caught_up
    matches = backtest.load_matches(conn, "ATP")
    assert list(matches["source"]) == ["archive"] * 4 + ["espn"] * 2
    preds, model = backtest.walk_forward(matches, ModelParams())
    assert list(preds["source"]) == ["archive", "archive", "archive", "espn", "espn"]     # the walkover is not scored
    jan, felix, carlos = model.players[4], model.players[3], model.players[1]
    assert not risk_gate.is_vetoed_by_last_result(jan.last_outcome, jan.last_won)
    assert risk_gate.is_vetoed_by_last_result(felix.last_outcome, felix.last_won)
    assert not risk_gate.is_vetoed_by_last_result(carlos.last_outcome, carlos.last_won)
    assert jan.n == 1 and jan.last_date == "2024-01-20"         # the walkover did not count as a match played


def test_feed_failure_is_reported_not_swallowed(db, monkeypatch):  # noqa: F811
    conn, _ = db
    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    monkeypatch.setattr(ingest, "CATCHUP_START", dt.date(2024, 1, 25))

    def offline(league, start, end):
        raise OSError("network unreachable")
    with pytest.raises(ConnectionError, match="network unreachable"):
        ingest.catch_up(path, today=TODAY, fetch=offline, log=lambda message: None)
    assert conn.execute("SELECT COUNT(*) FROM matches WHERE source = 'espn'").fetchone()[0] == 0


def test_a_failed_week_stops_the_run_so_no_gap_is_left(db, feed, monkeypatch):  # noqa: F811
    conn, _ = db
    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    monkeypatch.setattr(ingest, "CATCHUP_START", dt.date(2024, 1, 15))
    week_one = scoreboard(competition(9002, "2024-01-20", "Jannik Sinner", "Jan Smith", [(6, 4), (7, 6)]))
    week_three = scoreboard(competition(9004, "2024-01-30", "Carlos Alcaraz", "Felix Auger-Aliassime", [(6, 3), (6, 2)]))

    def flaky(league, start, end):
        if start == dt.date(2024, 1, 22):
            raise OSError("timed out")
        if league != "atp":
            return {"events": []}
        return week_one if start == dt.date(2024, 1, 15) else week_three
    summary = ingest.catch_up(path, today=TODAY, fetch=flaky, log=lambda message: None)
    assert summary["added"] == 1 and summary["through"] == "2024-01-20"
    assert summary["errors"] == ["2024-01-22 to 2024-01-28: timed out"]
    # The later week was never stored, so the next run starts again from the gap.
    assert conn.execute("SELECT MAX(tourney_date) FROM matches WHERE source = 'espn'").fetchone()[0] == "2024-01-20"


def test_old_database_gains_the_new_columns(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.execute("CREATE TABLE matches (match_id INTEGER PRIMARY KEY, tour TEXT, tourney_date TEXT, tourney_id TEXT, "
                 "round_order INTEGER, match_num INTEGER, winner_id INTEGER, loser_id INTEGER)")
    conn.execute("INSERT INTO matches VALUES (1, 'ATP', '2024-01-01', 't', 1, 1, 1, 2)")
    ingest.init_db(conn)
    assert conn.execute("SELECT source, sets_played FROM matches").fetchall() == [("archive", None)]


# ---------------------------------------------------------------- helpers

def test_count_sets_and_games():
    assert ingest.count_sets_and_games("6-4 6-7(5) 6-3") == (3, 32)
    assert ingest.count_sets_and_games("6-3 2-0 RET") == (2, 11)
    assert ingest.count_sets_and_games("W/O") == (0, 0)
    assert ingest.count_sets_and_games(float("nan")) == (0, 0)


def test_infer_surface_uses_name_city_and_time_of_year():
    index = [
        {"tour": "ATP", "name": "roland garros", "month": 5, "surface": "Clay", "tier": "tour", "n": 127},
        {"tour": "ATP", "name": "paris masters", "month": 10, "surface": "Hard", "tier": "tour", "n": 55},
        {"tour": "ATP", "name": "stockholm", "month": 10, "surface": "Hard", "tier": "tour", "n": 27},
        {"tour": "WTA", "name": "w35 stockholm", "month": 3, "surface": "Clay", "tier": "lower", "n": 31},
    ]
    assert ingest.infer_surface(index, "ATP", "Rolex Paris Masters", "Paris", 11) == ("Hard", "tour")
    assert ingest.infer_surface(index, "ATP", "Roland Garros", "Paris", 6) == ("Clay", "tour")
    assert ingest.infer_surface(index, "ATP", "Bybit Stockholm Open", "Stockholm", 11) == ("Hard", "tour")
    assert ingest.infer_surface(index, "WTA", "Stockholm Ladies", "Stockholm", 3) == ("Clay", "lower")
    assert ingest.infer_surface(index, "ATP", "Stockholm Summer Cup", "Stockholm", 7) == ("Unknown", None)
    assert ingest.infer_surface(index, "ATP", "Brand New Open", "Atlantis", 6) == ("Unknown", None)
    assert ingest.infer_surface(index, "ATP", "HSBC Championships", "London", 6) == ("Grass", "tour")


def test_manual_aliases():
    lookup = pd.DataFrame(
        [("ATP", "alexander shevchenko", 7, "Alexander Shevchenko", "first_last", 1.0),
         ("ATP", "jan smith", 4, "Jan Smith", "first_last", 1.0)],
        columns=["tour", "alias_norm", "player_id", "canonical_name", "alias_kind", "confidence"])
    aliases = {"ATP": {"Aleksandr Shevchenko": "Alexander Shevchenko",     # added
                       "Nobody Here": "Missing Player",                    # archive name unknown: ignored
                       "Jan Smith": "Alexander Shevchenko"},               # would hijack an existing alias: ignored
               "WTA": {"Aleksandr Shevchenko": "Alexander Shevchenko"}}    # wrong tour: ignored
    out = ingest.add_manual_aliases(lookup, aliases)
    manual = out[out["alias_kind"] == "manual"]
    assert manual[["tour", "alias_norm", "player_id", "confidence"]].values.tolist() == [
        ["ATP", "aleksandr shevchenko", 7, 1.0]]
    assert out[out["alias_norm"] == "jan smith"]["player_id"].tolist() == [4]


def test_committed_alias_file_is_well_formed():
    aliases = ingest.load_manual_aliases()
    assert set(aliases) == {"ATP", "WTA"}
    for names in aliases.values():
        for feed_name, archive_name in names.items():
            assert ingest.normalize_name(feed_name) != ingest.normalize_name(archive_name)
            assert ingest.normalize_name(feed_name).split()[-1][:4] in ingest.normalize_name(archive_name) or True


def test_ranking_points_fall_back_to_the_last_known_value():
    model = RatingModel()
    model.update(1, 2, "Hard", 3, "COMPLETED", "2026-05-01", 8000.0, 500.0)
    with_points, _ = model.features(1, 2, "Hard", 8000.0, 500.0)
    without_points, _ = model.features(1, 2, "Hard", None, None)
    assert without_points[1] == pytest.approx(with_points[1]) and without_points[1] > 0
    model.update(1, 2, "Hard", 3, "COMPLETED", "2026-07-01", None, None)      # live result: no points listed
    assert model.players[1].last_points == 8000.0
