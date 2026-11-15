import sqlite3

import pandas as pd
import pytest

from src import ingest

HEADER = ("tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,winner_id,winner_seed,"
          "winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,"
          "loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,minutes,w_ace,w_df,w_svpt,w_1stIn,"
          "w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,"
          "l_bpSaved,l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points")


def row(tid, date, num, wid, wname, lid, lname, score, rnd, surface="Hard", level="A", best_of=3):
    return (f"{tid},Test Open,{surface},32,{level},{date},{num},{wid},,,{wname},R,185,ESP,22.0,{lid},,,{lname},R,188,"
            f"ITA,23.0,{score},{best_of},{rnd},90,5,2,70,45,33,14,10,3,4,3,3,72,44,28,12,10,4,7,3,5000,10,3000")


@pytest.fixture
def archive(tmp_path):
    atp, wta = tmp_path / "atp", tmp_path / "wta"
    atp.mkdir()
    wta.mkdir()
    (atp / "atp_matches_2024.csv").write_text("\n".join([
        HEADER,
        row("2024-001", "20240108", 2, 1, "Carlos Alcaraz", 2, "Jannik Sinner", "6-4 6-4", "F"),
        row("2024-001", "20240108", 1, 1, "Carlos Alcaraz", 3, "Felix Auger-Aliassime", "6-3 2-0 RET", "SF"),
        row("2024-002", "20240101", 1, 2, "Jannik Sinner", 4, "Jan Smith", "W/O", "R32", surface="clay"),
    ]) + "\n")
    (atp / "atp_matches_qual_chall_2024.csv").write_text("\n".join([
        HEADER,
        row("2024-001", "20240108", 9, 3, "Felix Auger-Aliassime", 5, "John Smith", "7-6(3) 6-2", "Q1"),
        # duplicate of a tour-level row: must be dropped
        row("2024-001", "20240108", 2, 1, "Carlos Alcaraz", 2, "Jannik Sinner", "6-4 6-4", "F"),
    ]) + "\n")
    (atp / "atp_players.csv").write_text(
        "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
        "1,Carlos,Alcaraz,R,20030505,ESP,183,Q1\n2,Jannik,Sinner,R,20010816,ITA,191,Q2\n"
        "3,Felix,Auger-Aliassime,R,20000808,CAN,193,Q3\n4,Jan,Smith,R,,USA,,\n5,John,Smith,R,,USA,,\n")
    (wta / "wta_matches_2024.csv").write_text("\n".join([
        HEADER, row("2024-W01", "20240108", 1, 1, "Iga Swiatek", 2, "Aryna Sabalenka", "6-1 6-2", "F", surface=""),
    ]) + "\n")
    (wta / "wta_players.csv").write_text(
        "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n1,Iga,Świątek,R,,POL,176,\n2,Aryna,Sabalenka,R,,BLR,182,\n")
    return tmp_path


@pytest.fixture
def db(archive, tmp_path):
    path = tmp_path / "test.db"
    summary = ingest.ingest(archive, path)
    conn = sqlite3.connect(path)
    yield conn, summary
    conn.close()


def test_normalize_name():
    assert ingest.normalize_name("Iga Świątek") == "iga swiatek"
    assert ingest.normalize_name("Félix Auger-Aliassime") == "felix auger aliassime"
    assert ingest.normalize_name("  O'Connell,  Christopher ") == "oconnell christopher"
    assert ingest.normalize_name(None) == ""


@pytest.mark.parametrize("score,expected", [
    ("6-4 6-4", "COMPLETED"), ("6-3 2-0 RET", "RET"), ("W/O", "W/O"), ("Walkover", "W/O"),
    ("6-4 1-0 DEF", "DEF"), ("3-2 Ret'd", "RET"), (float("nan"), "COMPLETED"),
])
def test_classify_outcome(score, expected):
    assert ingest.classify_outcome(score) == expected


def test_matches_are_deduplicated_and_ordered(db):
    conn, summary = db
    assert summary["matches"] == 5
    df = pd.read_sql_query("SELECT * FROM matches WHERE tour='ATP' ORDER BY match_id", conn)
    assert list(df["round"]) == ["R32", "Q1", "SF", "F"]            # by date, then round order
    assert list(df["outcome"]) == ["W/O", "COMPLETED", "RET", "COMPLETED"]
    assert df.loc[df["round"] == "F", "tier"].item() == "tour"       # tour copy kept over the duplicate
    assert df["tourney_date"].iloc[0] == "2024-01-01"


def test_surface_is_normalised(db):
    conn, _ = db
    surfaces = dict(conn.execute("SELECT round || tour, surface FROM matches").fetchall())
    assert surfaces["R32ATP"] == "Clay"
    assert surfaces["FWTA"] == "Unknown"


def test_player_ids_are_scoped_by_tour(db):
    conn, _ = db
    names = dict(conn.execute("SELECT tour, full_name FROM players WHERE player_id = 1").fetchall())
    assert names == {"ATP": "Carlos Alcaraz", "WTA": "Iga Świątek"}


def test_lookup_exact_and_variants(db):
    conn, _ = db
    for name in ("Carlos Alcaraz", "alcaraz carlos", "C. Alcaraz", "Alcaraz C."):
        match = ingest.resolve_player(conn, name, "ATP")
        assert match and match.player_id == 1 and match.confidence == 1.0
    assert ingest.resolve_player(conn, "Iga Swiatek", "WTA").player_id == 1
    assert ingest.resolve_player(conn, "Felix Auger Aliassime", "ATP").player_id == 3


def test_lookup_rejects_low_confidence_and_ambiguity(db):
    conn, _ = db
    assert ingest.resolve_player(conn, "Carlos Alcarez", "ATP") is None      # 93% similar: below 98%
    assert ingest.resolve_player(conn, "J. Smith", "ATP") is None            # Jan Smith or John Smith
    assert ingest.resolve_player(conn, "Jan Smith", "ATP").player_id == 4
    assert ingest.resolve_player(conn, "Carlos Alcaraz", "WTA") is None      # wrong tour
    assert ingest.resolve_player(conn, "", "ATP") is None


def test_lookup_confidence_floor_is_enforced_by_schema(db):
    conn, _ = db
    assert conn.execute("SELECT MIN(confidence) FROM player_lookup").fetchone()[0] >= 0.98
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO player_lookup VALUES ('ATP','x y',1,'Carlos Alcaraz','first_last',0.9)")


def test_forbidden_upstream_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ingest.ensure_archive(tmp_path / "JeffSackmann" / "tennis_atp")
