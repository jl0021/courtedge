"""Ingest ATP and WTA match history into SQLite, in two stages.

Stage 1, the seed: Aneeshers/tennis-sackmann-archive (an archival mirror of Jeff
Sackmann's tennis data, CC BY-NC-SA 4.0), seasons 2024-2026. The archive stops in
early June 2026. The upstream JeffSackmann/tennis_atp repository returns 404 and
must not be used.

Stage 2, the catch-up: results after the archive ends, from ESPN's public
scoreboard (see src/espn_feed.py). Every player name must resolve through
player_lookup at 98% confidence or the match is skipped and reported.

Usage:
    python -m src.ingest                       # downloads the needed files into data/raw/ if missing
    python -m src.ingest --archive /path/to/tennis-sackmann-archive
    python -m src.ingest --catch-up            # only pull results newer than what is stored
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import sqlite3
import subprocess
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import espn_feed

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "courtedge.db"
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "raw" / "tennis-sackmann-archive"
ARCHIVE_URL = "https://github.com/Aneeshers/tennis-sackmann-archive"
ARCHIVE_RAW_URL = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
FORBIDDEN_SOURCES = ("JeffSackmann/tennis_atp",)
ALIASES_PATH = REPO_ROOT / "config" / "name_aliases.json"

# The catch-up asks the live feed for everything from this date and drops what the archive already has.
CATCHUP_START = dt.date(2026, 5, 25)
CATCHUP_CHUNK_DAYS = 7
DUPLICATE_WINDOW_DAYS = 21
FATIGUE_WINDOW_DAYS = 7

SEASONS = (2024, 2025, 2026)
TOURS = ("ATP", "WTA")

# Locked guardrail: mirrored in vault/Hard_Guardrails.md and src/risk_gate.py.
MIN_NAME_MATCH_CONFIDENCE = 0.98

# Files per tour: (filename pattern, tier). "tour" is the main tour-level draw.
# "lower" is tour qualifying plus Challenger (ATP) or ITF (WTA) events. Lower-tier
# matches feed the ratings but are not part of the headline backtest score.
SOURCE_FILES = {
    "ATP": (("atp_matches_{year}.csv", "tour"), ("atp_matches_qual_chall_{year}.csv", "lower")),
    "WTA": (("wta_matches_{year}.csv", "tour"), ("wta_matches_qual_itf_{year}.csv", "lower")),
}

# Order of rounds within one tournament, earliest first.
ROUND_ORDER = {
    "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4,
    "RR": 10, "R128": 11, "R64": 12, "R32": 13, "R16": 14,
    "QF": 15, "SF": 16, "BR": 17, "F": 18,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    tour        TEXT NOT NULL,
    player_id   INTEGER NOT NULL,
    first_name  TEXT,
    last_name   TEXT,
    full_name   TEXT NOT NULL,
    hand        TEXT,
    ioc         TEXT,
    height_cm   REAL,
    PRIMARY KEY (tour, player_id)
);

-- Every name variant that maps to exactly one player. Lookups that cannot reach
-- 98% confidence return no player, and the leg is vetoed.
CREATE TABLE IF NOT EXISTS player_lookup (
    tour            TEXT NOT NULL,
    alias_norm      TEXT NOT NULL,
    player_id       INTEGER NOT NULL,
    canonical_name  TEXT NOT NULL,
    alias_kind      TEXT NOT NULL,
    confidence      REAL NOT NULL CHECK (confidence >= 0.98 AND confidence <= 1.0),
    PRIMARY KEY (tour, alias_norm),
    FOREIGN KEY (tour, player_id) REFERENCES players (tour, player_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY,
    tour            TEXT NOT NULL,
    season          INTEGER NOT NULL,
    tier            TEXT NOT NULL,          -- 'tour' or 'lower'
    tourney_id      TEXT NOT NULL,
    tourney_name    TEXT,
    tourney_level   TEXT,
    tourney_date    TEXT NOT NULL,          -- ISO date of the tournament's first day
    surface         TEXT NOT NULL,          -- Hard, Clay, Grass, Carpet, Unknown
    round           TEXT,
    round_order     INTEGER NOT NULL,
    match_num       INTEGER,
    best_of         INTEGER NOT NULL,
    winner_id       INTEGER NOT NULL,
    loser_id        INTEGER NOT NULL,
    winner_name     TEXT,
    loser_name      TEXT,
    winner_rank_points REAL,
    loser_rank_points  REAL,
    winner_rank     REAL,
    loser_rank      REAL,
    score           TEXT,
    outcome         TEXT NOT NULL,          -- COMPLETED, RET, W/O, DEF, ABD
    minutes         REAL,
    w_svpt REAL, w_1stWon REAL, w_2ndWon REAL,
    l_svpt REAL, l_1stWon REAL, l_2ndWon REAL,
    source          TEXT NOT NULL DEFAULT 'archive',   -- 'archive' (seed) or 'espn' (catch-up)
    sets_played     INTEGER,
    games_played    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_matches_order ON matches (tour, tourney_date, tourney_id, round_order, match_num);
CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches (tour, winner_id);
CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches (tour, loser_id);

-- Surface-segmented telemetry as of the end of the ingested history. Written by backtest.py.
CREATE TABLE IF NOT EXISTS player_surface_stats (
    tour            TEXT NOT NULL,
    player_id       INTEGER NOT NULL,
    surface         TEXT NOT NULL,          -- Hard, Clay, Grass, or All
    matches_played  INTEGER NOT NULL,
    elo             REAL NOT NULL,
    serve_pts_won   REAL,
    return_pts_won  REAL,
    last_tourney_date TEXT,
    last_outcome    TEXT,                   -- how the player's most recent match ended: COMPLETED, RET, W/O...
    last_match_won  INTEGER,                -- 1 if the player won that match, 0 if they lost it
    as_of           TEXT NOT NULL,
    PRIMARY KEY (tour, player_id, surface)
);

-- Where each player stands today: last result and workload over the last 7 days. Written by the catch-up.
CREATE TABLE IF NOT EXISTS player_status (
    tour            TEXT NOT NULL,
    player_id       INTEGER NOT NULL,
    last_match_date TEXT,
    last_outcome    TEXT,
    last_match_won  INTEGER,
    last_match_ret_wo INTEGER NOT NULL,     -- 1 if the player lost their last match by RET or W/O (vetoed)
    matches_7d      INTEGER NOT NULL,
    sets_7d         INTEGER NOT NULL,
    games_7d        INTEGER NOT NULL,
    as_of           TEXT NOT NULL,
    PRIMARY KEY (tour, player_id)
);

-- Live-feed matches skipped because a player name did not reach 98% confidence.
CREATE TABLE IF NOT EXISTS ingest_unmatched (
    match_id    INTEGER NOT NULL,
    tour        TEXT NOT NULL,
    name        TEXT NOT NULL,
    match_date  TEXT NOT NULL,
    PRIMARY KEY (match_id, name)
);

-- One row per match scored by the walk-forward backtest.
CREATE TABLE IF NOT EXISTS predictions (
    run_id      TEXT NOT NULL,
    match_id    INTEGER NOT NULL,
    engine      TEXT NOT NULL,
    p_winner    REAL NOT NULL,              -- probability the model gave to the eventual winner
    PRIMARY KEY (run_id, match_id)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    engine      TEXT NOT NULL,
    n_matches   INTEGER NOT NULL,
    brier       REAL NOT NULL,
    max_calibration_gap REAL NOT NULL,
    log_loss    REAL NOT NULL,
    accuracy    REAL NOT NULL,
    passed      INTEGER NOT NULL,
    params_json TEXT NOT NULL
);

-- Paper trades only. Nothing in this project places real orders.
CREATE TABLE IF NOT EXISTS paper_combos (
    combo_id        INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,
    slate_date      TEXT NOT NULL,
    n_legs          INTEGER NOT NULL,
    fair_combo_prob REAL NOT NULL,
    kalshi_combo_price REAL NOT NULL,
    edge            REAL NOT NULL,
    stake_usd       REAL NOT NULL DEFAULT 1.0,
    status          TEXT NOT NULL DEFAULT 'open',   -- open, won, lost, void
    settled_at      TEXT
);

CREATE TABLE IF NOT EXISTS paper_legs (
    leg_id          INTEGER PRIMARY KEY,
    combo_id        INTEGER NOT NULL REFERENCES paper_combos (combo_id),
    tour            TEXT NOT NULL,
    player_id       INTEGER NOT NULL,
    opponent_id     INTEGER,
    kalshi_ticker   TEXT,
    surface         TEXT,
    simulated_prob  REAL NOT NULL,
    kalshi_ask      REAL NOT NULL,
    result          TEXT                            -- yes, no, void
);
"""


# --------------------------------------------------------------------------- names

def normalize_name(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("-", " ").replace("'", "").replace(".", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_aliases(first: str, last: str) -> list[tuple[str, str]]:
    """Return (alias_norm, alias_kind) variants for one player."""
    first_n, last_n = normalize_name(first), normalize_name(last)
    aliases = []
    if first_n and last_n:
        aliases.append((f"{first_n} {last_n}", "first_last"))
        aliases.append((f"{last_n} {first_n}", "last_first"))
        aliases.append((f"{first_n[0]} {last_n}", "initial_last"))
        aliases.append((f"{last_n} {first_n[0]}", "last_initial"))
    elif last_n or first_n:
        aliases.append((last_n or first_n, "single"))
    return aliases


@dataclass(frozen=True)
class NameMatch:
    player_id: int
    canonical_name: str
    confidence: float


def resolve_player(conn: sqlite3.Connection, name: str, tour: str) -> NameMatch | None:
    """Map a free-text player name to a player_id.

    Returns None unless the match reaches MIN_NAME_MATCH_CONFIDENCE (0.98) and is
    unambiguous. A None result means the caller must veto the leg.
    """
    query = normalize_name(name)
    if not query:
        return None
    row = conn.execute(
        "SELECT player_id, canonical_name, confidence FROM player_lookup WHERE tour = ? AND alias_norm = ?",
        (tour, query),
    ).fetchone()
    if row:
        return NameMatch(row[0], row[1], row[2])

    # Fuzzy fallback against full-name aliases only. Initial-based aliases are too
    # short to compare safely.
    rows = conn.execute(
        "SELECT alias_norm, player_id, canonical_name FROM player_lookup "
        "WHERE tour = ? AND alias_kind IN ('first_last', 'last_first')",
        (tour,),
    ).fetchall()
    best: dict[int, tuple[float, str]] = {}
    for alias, player_id, canonical in rows:
        if abs(len(alias) - len(query)) > 1:
            continue        # a length gap of 2 or more cannot reach 98% similarity
        matcher = difflib.SequenceMatcher(None, query, alias)
        if matcher.real_quick_ratio() < MIN_NAME_MATCH_CONFIDENCE or matcher.quick_ratio() < MIN_NAME_MATCH_CONFIDENCE:
            continue
        ratio = matcher.ratio()
        if ratio > best.get(player_id, (0.0, ""))[0]:
            best[player_id] = (ratio, canonical)
    qualifying = [(ratio, pid, canonical) for pid, (ratio, canonical) in best.items()
                  if ratio >= MIN_NAME_MATCH_CONFIDENCE]
    if len(qualifying) != 1:
        return None  # no confident match, or two players both match
    ratio, player_id, canonical = qualifying[0]
    return NameMatch(player_id, canonical, round(ratio, 4))


# --------------------------------------------------------------------------- matches

def classify_outcome(score) -> str:
    """Classify how a match ended from its score string."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return "COMPLETED"
    text = str(score).upper()
    if "W/O" in text or "WALKOVER" in text:
        return "W/O"
    if "RET" in text:
        return "RET"
    if "DEF" in text:
        return "DEF"
    if "ABD" in text or "ABN" in text:
        return "ABD"
    return "COMPLETED"


_SET_PATTERN = re.compile(r"(\d+)-(\d+)")


def count_sets_and_games(score) -> tuple[int, int]:
    """Sets and games played, from a score string such as '6-4 6-7(5) 6-3'."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return 0, 0
    sets = [(int(a), int(b)) for a, b in _SET_PATTERN.findall(re.sub(r"\(\d+\)", "", str(score)))]
    return len(sets), sum(a + b for a, b in sets)


def normalize_surface(surface) -> str:
    if surface is None or (isinstance(surface, float) and pd.isna(surface)):
        return "Unknown"
    text = str(surface).strip().title()
    return text if text in ("Hard", "Clay", "Grass", "Carpet") else "Unknown"


def required_files(seasons=SEASONS, tours=TOURS) -> list[str]:
    """Archive-relative paths of every file the ingest reads (about 26 MB in total)."""
    files = []
    for tour in tours:
        folder = tour.lower()
        for pattern, _tier in SOURCE_FILES[tour]:
            files += [f"{folder}/{pattern.format(year=season)}" for season in seasons]
        files.append(f"{folder}/{folder}_players.csv")
    return files


def download_archive_files(archive: Path, seasons=SEASONS, force: bool = False, log=print) -> int:
    """Download only the needed CSV files from the archive. Returns how many were fetched."""
    fetched = 0
    for rel in required_files(seasons):
        target = archive / rel
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"Downloading {rel}")
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(f"{ARCHIVE_RAW_URL}/{rel}", timeout=60) as response:
            partial.write_bytes(response.read())
        partial.replace(target)
        fetched += 1
    return fetched


def ensure_archive(archive: Path, seasons=SEASONS, force_download: bool = False, log=print) -> Path:
    """Make sure the needed archive files are on disk.

    The default location (data/raw/) is managed: missing files are downloaded, and
    if direct downloads are blocked the archive is cloned with git instead. Any other
    path is treated as a copy the caller already has and is used as it is.
    """
    if any(bad.lower() in str(archive).lower() for bad in ("jeffsackmann",)):
        raise ValueError("JeffSackmann upstream repositories must not be used as a source")
    if archive.resolve() != DEFAULT_ARCHIVE.resolve():
        if not (archive / "atp").is_dir() or not (archive / "wta").is_dir():
            raise FileNotFoundError(f"{archive} does not contain atp/ and wta/ folders")
        return archive

    have_all = all((archive / rel).exists() for rel in required_files(seasons))
    if have_all and not force_download:
        return archive
    try:
        download_archive_files(archive, seasons, force=force_download, log=log)
        return archive
    except OSError as exc:
        if have_all:
            log(f"Could not refresh the archive ({exc}). Using the files already on disk.")
            return archive
        if archive.exists() and any(archive.iterdir()):
            raise
        log(f"Direct download failed ({exc}). Cloning the archive with git instead.")
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", ARCHIVE_URL, str(archive)],
        check=True,
    )
    subprocess.run(["git", "-C", str(archive), "sparse-checkout", "set", "atp", "wta"], check=True)
    return archive


def load_matches(archive: Path, seasons=SEASONS, tours=TOURS) -> pd.DataFrame:
    """Read and normalise every match file for the requested seasons."""
    frames = []
    for tour in tours:
        folder = archive / tour.lower()
        for pattern, tier in SOURCE_FILES[tour]:
            for season in seasons:
                path = folder / pattern.format(year=season)
                if not path.exists():
                    continue
                df = pd.read_csv(path, low_memory=False)
                df["tour"], df["season"], df["tier"] = tour, season, tier
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No match files found under {archive}")
    raw = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "tour": raw["tour"],
        "season": raw["season"].astype(int),
        "tier": raw["tier"],
        "tourney_id": raw["tourney_id"].astype(str),
        "tourney_name": raw["tourney_name"],
        "tourney_level": raw["tourney_level"].astype(str),
        "tourney_date": pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d"),
        "surface": raw["surface"].map(normalize_surface),
        "round": raw["round"],
        "round_order": raw["round"].map(ROUND_ORDER).fillna(9).astype(int),
        "match_num": pd.to_numeric(raw["match_num"], errors="coerce"),
        "best_of": pd.to_numeric(raw["best_of"], errors="coerce").fillna(3).astype(int),
        "winner_id": pd.to_numeric(raw["winner_id"], errors="coerce"),
        "loser_id": pd.to_numeric(raw["loser_id"], errors="coerce"),
        "winner_name": raw["winner_name"],
        "loser_name": raw["loser_name"],
        "winner_rank_points": pd.to_numeric(raw["winner_rank_points"], errors="coerce"),
        "loser_rank_points": pd.to_numeric(raw["loser_rank_points"], errors="coerce"),
        "winner_rank": pd.to_numeric(raw["winner_rank"], errors="coerce"),
        "loser_rank": pd.to_numeric(raw["loser_rank"], errors="coerce"),
        "score": raw["score"],
        "outcome": raw["score"].map(classify_outcome),
        "minutes": pd.to_numeric(raw["minutes"], errors="coerce"),
    })
    for col in ("w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"):
        out[col] = pd.to_numeric(raw[col], errors="coerce")
    out["source"] = "archive"
    counts = raw["score"].map(count_sets_and_games)
    out["sets_played"] = counts.map(lambda c: c[0])
    out["games_played"] = counts.map(lambda c: c[1])

    out = out.dropna(subset=["winner_id", "loser_id"])
    out["winner_id"] = out["winner_id"].astype(int)
    out["loser_id"] = out["loser_id"].astype(int)
    # The same match can appear in two files. Keep the tour-level copy.
    out = out.sort_values("tier", ascending=False, kind="stable")  # 'tour' before 'lower'
    out = out.drop_duplicates(subset=["tour", "tourney_id", "round", "winner_id", "loser_id"], keep="first")
    out = out.sort_values(
        ["tourney_date", "tour", "tourney_id", "round_order", "match_num"], kind="stable"
    ).reset_index(drop=True)
    out.insert(0, "match_id", out.index + 1)
    return out


def load_players(archive: Path, matches: pd.DataFrame) -> pd.DataFrame:
    """Players who appear in the ingested matches, joined to the archive's player tables."""
    frames = []
    for tour in matches["tour"].unique():
        subset = matches[matches["tour"] == tour]
        seen = pd.concat([
            subset[["winner_id", "winner_name"]].set_axis(["player_id", "match_name"], axis=1),
            subset[["loser_id", "loser_name"]].set_axis(["player_id", "match_name"], axis=1),
        ]).drop_duplicates("player_id")
        table_path = archive / tour.lower() / f"{tour.lower()}_players.csv"
        if table_path.exists():
            table = pd.read_csv(table_path, low_memory=False)
            table["player_id"] = pd.to_numeric(table["player_id"], errors="coerce")
            seen = seen.merge(table, on="player_id", how="left")
        else:
            seen = seen.assign(name_first=None, name_last=None, hand=None, ioc=None, height=None)
        # Fall back to splitting the match-file name when the player table has no entry.
        missing = seen["name_last"].isna()
        split = seen.loc[missing, "match_name"].fillna("").astype(str).str.strip().str.split(" ", n=1, expand=True)
        if missing.any():
            seen.loc[missing, "name_first"] = split[0]
            seen.loc[missing, "name_last"] = split[1] if split.shape[1] > 1 else ""
        seen["name_first"] = seen["name_first"].fillna("").astype(str).str.strip()
        seen["name_last"] = seen["name_last"].fillna("").astype(str).str.strip()
        seen["full_name"] = (seen["name_first"] + " " + seen["name_last"]).str.strip()
        seen["tour"] = tour
        frames.append(seen[["tour", "player_id", "name_first", "name_last", "full_name", "hand", "ioc", "height"]])
    players = pd.concat(frames, ignore_index=True)
    players = players.rename(columns={"name_first": "first_name", "name_last": "last_name", "height": "height_cm"})
    players["player_id"] = players["player_id"].astype(int)
    players["height_cm"] = pd.to_numeric(players["height_cm"], errors="coerce")
    return players


def build_player_lookup(players: pd.DataFrame) -> pd.DataFrame:
    """Build alias rows. An alias shared by two players in a tour is dropped as ambiguous."""
    rows = []
    for rec in players.itertuples(index=False):
        for alias, kind in name_aliases(rec.first_name, rec.last_name):
            rows.append((rec.tour, alias, rec.player_id, rec.full_name, kind, 1.0))
    lookup = pd.DataFrame(rows, columns=["tour", "alias_norm", "player_id", "canonical_name", "alias_kind", "confidence"])
    lookup = lookup.drop_duplicates(["tour", "alias_norm", "player_id"])
    owners = lookup.groupby(["tour", "alias_norm"])["player_id"].transform("nunique")
    lookup = lookup[owners == 1].drop_duplicates(["tour", "alias_norm"]).reset_index(drop=True)
    return add_manual_aliases(lookup, load_manual_aliases())


def load_manual_aliases(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Hand-confirmed spellings: {"ATP": {"Name in the live feed": "Name in the archive"}}."""
    path = path or ALIASES_PATH
    if not path.exists():
        return {}
    return {tour: names for tour, names in json.loads(path.read_text()).items() if not tour.startswith("_")}


def add_manual_aliases(lookup: pd.DataFrame, aliases: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Add hand-confirmed aliases. One is ignored if its archive name is unknown or it clashes with an existing alias."""
    index = {(row.tour, row.alias_norm): row for row in lookup.itertuples(index=False)}
    rows = []
    for tour, names in aliases.items():
        for feed_name, archive_name in names.items():
            target = index.get((tour, normalize_name(archive_name)))
            alias = normalize_name(feed_name)
            if target is None or not alias or (tour, alias) in index:
                continue
            rows.append((tour, alias, target.player_id, target.canonical_name, "manual", 1.0))
            index[(tour, alias)] = target
    if not rows:
        return lookup
    return pd.concat([lookup, pd.DataFrame(rows, columns=lookup.columns)], ignore_index=True)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Databases created before the catch-up stage lack these columns.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
    for column, ddl in (("source", "TEXT NOT NULL DEFAULT 'archive'"), ("sets_played", "INTEGER"),
                        ("games_played", "INTEGER")):
        if column not in existing:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {column} {ddl}")
    conn.commit()


def ingest(archive: Path = DEFAULT_ARCHIVE, db_path: Path = DEFAULT_DB, seasons=SEASONS,
           force_download: bool = False, log=print) -> dict:
    archive = ensure_archive(Path(archive), seasons, force_download=force_download, log=log)
    matches = load_matches(archive, seasons)
    players = load_players(archive, matches)
    lookup = build_player_lookup(players)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # Derived tables are rebuilt by backtest.py, so drop them to pick up schema changes.
        for table in ("player_surface_stats", "predictions", "backtest_runs"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        init_db(conn)
        conn.execute("DELETE FROM matches WHERE source = 'archive'")     # keep catch-up rows
        for table in ("player_lookup", "players"):
            conn.execute(f"DELETE FROM {table}")
        players.to_sql("players", conn, if_exists="append", index=False)
        lookup.to_sql("player_lookup", conn, if_exists="append", index=False)
        matches.to_sql("matches", conn, if_exists="append", index=False)
        refresh_player_status(conn)
        conn.commit()

    return {
        "matches": len(matches),
        "players": len(players),
        "lookup_aliases": len(lookup),
        "by_tour_tier": matches.groupby(["tour", "tier", "season"]).size().to_dict(),
        "date_range": (matches["tourney_date"].min(), matches["tourney_date"].max()),
    }


# --------------------------------------------------------------------------- stage 2: catch-up

def build_surface_index(conn: sqlite3.Connection) -> list[dict]:
    """Archive tournaments: normalised name, usual month, surface and tier. Used to place live events."""
    rows = conn.execute(
        "SELECT tour, tourney_name, CAST(substr(tourney_date, 6, 2) AS INTEGER), surface, tier, COUNT(*) "
        "FROM matches WHERE source = 'archive' AND surface != 'Unknown' AND tourney_name IS NOT NULL "
        "GROUP BY 1, 2, 3, 4, 5").fetchall()
    return [{"tour": tour, "name": normalize_name(name), "month": month, "surface": surface, "tier": tier, "n": n}
            for tour, name, month, surface, tier, n in rows if normalize_name(name)]


# Tour events the archive lists under a name that shares no word with the live feed's name or city.
EVENT_OVERRIDES = {
    "hsbc championships": ("Grass", "tour"),          # Queen's Club, London
    "millennium estoril open": ("Clay", "tour"),
}


def infer_surface(index: list[dict], tour: str, event_name: str, city: str, month: int) -> tuple[str, str | None]:
    """Surface and tier for a live event, from the archive's record of the same tournament.

    The live feed does not state the surface. A tournament is recognised when the
    archive's name appears in the event name or contains the host city, and the
    archive played it within two months of the same time of year. Returns
    ('Unknown', None) when nothing matches; ratings then use the player's overall level.
    """
    event, town = f" {normalize_name(event_name)} ", f" {normalize_name(city)} "
    for key, known in EVENT_OVERRIDES.items():
        if f" {key} " in event:
            return known
    best = None
    for entry in index:
        name = f" {entry['name']} "
        by_name = len(entry["name"]) >= 4 and name in event
        by_city = len(town.strip()) >= 4 and town in name
        if not (by_name or by_city):
            continue
        gap = min(abs(entry["month"] - month), 12 - abs(entry["month"] - month))
        if gap > 2:
            continue
        rank = (gap, entry["tour"] != tour, not by_name, -entry["n"])
        if best is None or rank < best[0]:
            best = (rank, entry)
    if best is None:
        return "Unknown", None
    return best[1]["surface"], best[1]["tier"]


def refresh_player_status(conn: sqlite3.Connection, today: dt.date | None = None) -> int:
    """Rebuild player_status: each player's last result and their workload over the last 7 days."""
    today = today or dt.date.today()
    since = (today - dt.timedelta(days=FATIGUE_WINDOW_DAYS)).isoformat()
    frame = pd.read_sql_query(
        "SELECT match_id, tour, tourney_date, round_order, outcome, winner_id, loser_id, "
        "COALESCE(sets_played, 0) AS sets_played, COALESCE(games_played, 0) AS games_played FROM matches", conn)
    conn.execute("DELETE FROM player_status")
    if frame.empty:
        return 0
    shared = ["match_id", "tour", "tourney_date", "round_order", "outcome", "sets_played", "games_played"]
    long = pd.concat([
        frame[shared + ["winner_id"]].rename(columns={"winner_id": "player_id"}).assign(won=1),
        frame[shared + ["loser_id"]].rename(columns={"loser_id": "player_id"}).assign(won=0),
    ]).sort_values(["tourney_date", "round_order", "match_id"], kind="stable")
    last = long.groupby(["tour", "player_id"]).tail(1).set_index(["tour", "player_id"])
    recent = long[(long["tourney_date"] > since) & (long["tourney_date"] <= today.isoformat()) & (long["outcome"] != "W/O")]
    load = recent.groupby(["tour", "player_id"]).agg(
        matches_7d=("match_id", "size"), sets_7d=("sets_played", "sum"), games_7d=("games_played", "sum"))
    status = last.join(load).fillna({"matches_7d": 0, "sets_7d": 0, "games_7d": 0}).reset_index()
    rows = [(r.tour, int(r.player_id), r.tourney_date, r.outcome, int(r.won),
             int(r.outcome in ("RET", "W/O") and r.won == 0), int(r.matches_7d), int(r.sets_7d), int(r.games_7d),
             today.isoformat()) for r in status.itertuples(index=False)]
    conn.executemany("INSERT INTO player_status VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def catch_up(db_path: Path = DEFAULT_DB, today: dt.date | None = None, fetch=espn_feed.fetch_scoreboard,
             log=print) -> dict:
    """Append results newer than what is stored, then refresh every player's status.

    Safe to run as often as you like: a match already stored is never added twice.
    A match is skipped, and counted in the summary, when either name fails the 98%
    lookup or when the archive already holds the same result.
    """
    today = today or dt.date.today()
    summary = {"fetched": 0, "added": 0, "already_stored": 0, "in_archive": 0, "unmatched": 0,
               "unmatched_names": [], "unknown_surface": 0, "through": None, "errors": []}
    with sqlite3.connect(db_path) as conn:
        init_db(conn)
        last = conn.execute("SELECT MAX(tourney_date) FROM matches WHERE source = 'espn'").fetchone()[0]
        start = max(CATCHUP_START, dt.date.fromisoformat(last) - dt.timedelta(days=3)) if last else CATCHUP_START

        found: dict[int, espn_feed.EspnMatch] = {}
        day = start
        while day <= today:
            end = min(day + dt.timedelta(days=CATCHUP_CHUNK_DAYS - 1), today)
            log(f"Fetching results {day.isoformat()} to {end.isoformat()}")
            chunk: dict[int, espn_feed.EspnMatch] = {}
            try:
                for league in espn_feed.LEAGUES:
                    for match in espn_feed.parse_scoreboard(fetch(league, day, end)):
                        chunk[match.competition_id] = match
            except (OSError, ValueError) as exc:
                # Stop at the first gap so stored results stay continuous. The next run resumes from here.
                summary["errors"].append(f"{day.isoformat()} to {end.isoformat()}: {exc}")
                break
            found.update(chunk)
            day = end + dt.timedelta(days=1)
        summary["fetched"] = len(found)

        stored = {row[0] for row in conn.execute("SELECT match_id FROM matches WHERE source = 'espn'")}
        archive_end = {tour: dt.date.fromisoformat(d) for tour, d in conn.execute(
            "SELECT tour, MAX(tourney_date) FROM matches WHERE source = 'archive' GROUP BY tour")}
        surfaces = build_surface_index(conn)
        resolved: dict[tuple[str, str], NameMatch | None] = {}
        unmatched: dict[tuple[str, str], int] = {}
        if "match_date" not in {row[1] for row in conn.execute("PRAGMA table_info(ingest_unmatched)")}:
            conn.execute("DROP TABLE ingest_unmatched")      # layout from an earlier version
            init_db(conn)

        def lookup(tour: str, name: str) -> NameMatch | None:
            key = (tour, name)
            if key not in resolved:
                resolved[key] = resolve_player(conn, name, tour)
            return resolved[key]

        log(f"Matching {len(found):,} results to players")
        rows = []
        for match in sorted(found.values(), key=lambda m: (m.date, m.competition_id)):
            match_id = espn_feed.MATCH_ID_OFFSET + match.competition_id
            if match_id in stored:
                summary["already_stored"] += 1
                continue
            winner, loser = lookup(match.tour, match.winner_name), lookup(match.tour, match.loser_name)
            if winner is None or loser is None:
                summary["unmatched"] += 1
                for name, hit in ((match.winner_name, winner), (match.loser_name, loser)):
                    if hit is None:
                        unmatched[(match.tour, name)] = unmatched.get((match.tour, name), 0) + 1
                        conn.execute("INSERT OR IGNORE INTO ingest_unmatched VALUES (?,?,?,?)",
                                     (match_id, match.tour, name, match.date))
                continue
            played = dt.date.fromisoformat(match.date)
            cutoff = archive_end.get(match.tour)
            if cutoff and played <= cutoff + dt.timedelta(days=DUPLICATE_WINDOW_DAYS):
                # Same players, same winner, same way of ending and same number of games, around the same
                # date: the archive already has this match. Archive dates are the tournament's first day.
                duplicate = conn.execute(
                    "SELECT 1 FROM matches WHERE source = 'archive' AND tour = ? AND winner_id = ? AND loser_id = ? "
                    "AND tourney_date BETWEEN ? AND ? AND outcome = ? AND COALESCE(games_played, 0) = ? LIMIT 1",
                    (match.tour, winner.player_id, loser.player_id,
                     (played - dt.timedelta(days=DUPLICATE_WINDOW_DAYS)).isoformat(),
                     (played + dt.timedelta(days=7)).isoformat(), match.outcome, match.games_played)).fetchone()
                if duplicate:
                    summary["in_archive"] += 1
                    continue
            surface, tier = infer_surface(surfaces, match.tour, match.event_name, match.city, played.month)
            if surface == "Unknown":
                summary["unknown_surface"] += 1
            if match.qualifying:
                tier = "lower"
            elif match.major:
                tier = "tour"
            rows.append((
                match_id, match.tour, played.year, tier or "lower", f"espn-{match.event_id}", match.event_name, "",
                match.date, surface, match.round_name, match.round_order, match.competition_id, match.best_of,
                winner.player_id, loser.player_id, winner.canonical_name, loser.canonical_name,
                match.score, match.outcome, "espn", match.sets_played, match.games_played,
            ))
        conn.executemany(
            "INSERT INTO matches (match_id, tour, season, tier, tourney_id, tourney_name, tourney_level, tourney_date, "
            "surface, round, round_order, match_num, best_of, winner_id, loser_id, winner_name, loser_name, score, "
            "outcome, source, sets_played, games_played) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        summary["added"] = len(rows)
        # A name that now resolves (for example after adding an alias) no longer belongs in the skipped list.
        for tour, name in conn.execute("SELECT DISTINCT tour, name FROM ingest_unmatched").fetchall():
            if (tour, name) not in unmatched and lookup(tour, name) is not None:
                conn.execute("DELETE FROM ingest_unmatched WHERE tour = ? AND name = ?", (tour, name))
        summary["unmatched_names"] = sorted(((count, tour, name) for (tour, name), count in unmatched.items()),
                                            reverse=True)[:25]
        refresh_player_status(conn, today)
        summary["through"] = conn.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]
        conn.commit()
    if summary["errors"] and not found:
        raise ConnectionError("Could not reach the live results feed: " + summary["errors"][0])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--catch-up", action="store_true", help="skip the seed and only pull newer results")
    parser.add_argument("--seed-only", action="store_true", help="load the archive without pulling newer results")
    args = parser.parse_args()
    if args.catch_up:
        print_catch_up(catch_up(args.db))
        return
    summary = ingest(args.archive, args.db)
    print(f"Ingested {summary['matches']:,} matches, {summary['players']:,} players, "
          f"{summary['lookup_aliases']:,} lookup aliases")
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    for key, count in sorted(summary["by_tour_tier"].items()):
        print(f"  {key[0]} {key[1]:5s} {key[2]}: {count:,}")
    if not args.seed_only:
        print_catch_up(catch_up(args.db))


def print_catch_up(summary: dict) -> None:
    print(f"Catch-up: {summary['added']:,} new results added, now current through {summary['through']}")
    print(f"  {summary['in_archive']:,} already in the archive, {summary['already_stored']:,} already stored, "
          f"{summary['unmatched']:,} skipped because a name did not reach 98% confidence")
    if summary["unknown_surface"]:
        print(f"  {summary['unknown_surface']:,} added with an unknown surface")
    for count, tour, name in summary["unmatched_names"][:10]:
        print(f"    unmatched: {name} ({tour}), {count} matches")
    for error in summary["errors"][:5]:
        print(f"  fetch problem: {error}")


if __name__ == "__main__":
    main()
