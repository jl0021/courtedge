"""Ingest 2024-2026 ATP and WTA match history into SQLite.

Source: Aneeshers/tennis-sackmann-archive (an archival mirror of Jeff Sackmann's
tennis data, CC BY-NC-SA 4.0). The upstream JeffSackmann/tennis_atp repository
returns 404 and must not be used.

Usage:
    python -m src.ingest                       # downloads the needed files into data/raw/ if missing
    python -m src.ingest --archive /path/to/tennis-sackmann-archive
"""

from __future__ import annotations

import argparse
import difflib
import re
import sqlite3
import subprocess
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "courtedge.db"
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "raw" / "tennis-sackmann-archive"
ARCHIVE_URL = "https://github.com/Aneeshers/tennis-sackmann-archive"
ARCHIVE_RAW_URL = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
FORBIDDEN_SOURCES = ("JeffSackmann/tennis_atp",)

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
    l_svpt REAL, l_1stWon REAL, l_2ndWon REAL
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
        if abs(len(alias) - len(query)) > 3:
            continue
        ratio = difflib.SequenceMatcher(None, query, alias).ratio()
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
    return lookup[owners == 1].drop_duplicates(["tour", "alias_norm"]).reset_index(drop=True)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
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
        for table in ("matches", "player_lookup", "players"):
            conn.execute(f"DELETE FROM {table}")
        players.to_sql("players", conn, if_exists="append", index=False)
        lookup.to_sql("player_lookup", conn, if_exists="append", index=False)
        matches.to_sql("matches", conn, if_exists="append", index=False)
        conn.commit()

    return {
        "matches": len(matches),
        "players": len(players),
        "lookup_aliases": len(lookup),
        "by_tour_tier": matches.groupby(["tour", "tier", "season"]).size().to_dict(),
        "date_range": (matches["tourney_date"].min(), matches["tourney_date"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    summary = ingest(args.archive, args.db)
    print(f"Ingested {summary['matches']:,} matches, {summary['players']:,} players, "
          f"{summary['lookup_aliases']:,} lookup aliases")
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    for key, count in sorted(summary["by_tour_tier"].items()):
        print(f"  {key[0]} {key[1]:5s} {key[2]}: {count:,}")


if __name__ == "__main__":
    main()
