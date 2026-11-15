"""Read-only client for ESPN's public tennis scoreboard.

Used to bring results up to date after the historical archive ends. The endpoint is
public but undocumented, so every field is read defensively: a match that cannot be
understood is skipped, never guessed.

What the feed provides: tournament, city, match date, round, both players' full names,
set scores, and whether the match ended normally, by retirement or by walkover.
What it does not provide: surface, ranking points, and serve statistics.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from dataclasses import dataclass

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{league}/scoreboard?dates={start}-{end}"
LEAGUES = ("atp", "wta")
TOUR_BY_GROUPING = {"mens-singles": "ATP", "womens-singles": "WTA"}
OUTCOME_BY_STATUS = {"STATUS_FINAL": "COMPLETED", "STATUS_RETIRED": "RET", "STATUS_WALKOVER": "W/O"}
ROUND_ORDER = {
    "qualifying 1st round": 1, "qualifying 2nd round": 2, "qualifying 3rd round": 3, "qualifying final": 3,
    "round robin": 10, "round 1": 11, "round 2": 12, "round 3": 13, "round 4": 14, "round of 16": 14,
    "quarterfinal": 15, "quarterfinals": 15, "semifinal": 16, "semifinals": 16, "final": 18,
}
MATCH_ID_OFFSET = 1_000_000_000      # keeps ESPN match ids clear of the archive's 1..N


@dataclass(frozen=True)
class EspnMatch:
    competition_id: int
    event_id: str
    event_name: str
    city: str
    major: bool
    tour: str
    date: str                 # ISO date (UTC) the match was played
    round_name: str
    round_order: int
    qualifying: bool
    outcome: str              # COMPLETED, RET or W/O
    winner_name: str
    loser_name: str
    score: str                # winner's games first, e.g. "6-4 6-7(5) 6-3"
    sets_played: int
    games_played: int
    best_of: int


def fetch_scoreboard(league: str, start: dt.date, end: dt.date, timeout: int = 60) -> dict:
    """One GET request for every match between two dates (inclusive)."""
    if league not in LEAGUES:
        raise ValueError(f"unknown league {league!r}")
    url = SCOREBOARD_URL.format(league=league, start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CourtEdge results reader)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _city(event: dict) -> str:
    """'Inalpi Arena, Turin, Italy' -> 'Turin'; 'Stockholm, Sweden' -> 'Stockholm'."""
    venue = (event.get("venue") or {}).get("displayName") or ""
    parts = [p.strip() for p in venue.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""


def _score(winner: dict, loser: dict) -> tuple[str, int, int]:
    sets = []
    games = 0
    for w, l in zip(winner.get("linescores") or [], loser.get("linescores") or []):
        try:
            wg, lg = int(w["value"]), int(l["value"])
        except (KeyError, TypeError, ValueError):
            continue
        tiebreak = l.get("tiebreak") if wg > lg else w.get("tiebreak")
        sets.append(f"{wg}-{lg}" + (f"({int(tiebreak)})" if tiebreak is not None else ""))
        games += wg + lg
    return " ".join(sets), len(sets), games


def parse_scoreboard(payload: dict) -> list[EspnMatch]:
    """Finished singles matches in a scoreboard response. Doubles and unfinished matches are dropped."""
    matches: list[EspnMatch] = []
    for event in payload.get("events") or []:
        for grouping in event.get("groupings") or []:
            tour = TOUR_BY_GROUPING.get((grouping.get("grouping") or {}).get("slug"))
            if tour is None:
                continue
            for comp in grouping.get("competitions") or []:
                outcome = OUTCOME_BY_STATUS.get(((comp.get("status") or {}).get("type") or {}).get("name"))
                competitors = comp.get("competitors") or []
                if outcome is None or len(competitors) != 2:
                    continue
                winners = [c for c in competitors if c.get("winner") is True]
                losers = [c for c in competitors if c.get("winner") is not True]
                if len(winners) != 1 or len(losers) != 1:
                    continue
                winner_name = ((winners[0].get("athlete") or {}).get("fullName")
                               or (winners[0].get("athlete") or {}).get("displayName"))
                loser_name = ((losers[0].get("athlete") or {}).get("fullName")
                              or (losers[0].get("athlete") or {}).get("displayName"))
                try:
                    competition_id = int(comp["id"])
                    date = dt.datetime.fromisoformat(comp["date"].replace("Z", "+00:00")).date().isoformat()
                except (KeyError, TypeError, ValueError):
                    continue
                if not winner_name or not loser_name:
                    continue
                round_name = ((comp.get("round") or {}).get("displayName") or "").strip()
                qualifying = round_name.lower().startswith("qualifying")
                major = bool(event.get("major"))
                score, sets_played, games_played = _score(winners[0], losers[0])
                if outcome == "W/O":
                    score, sets_played, games_played = "W/O", 0, 0
                elif outcome == "RET":
                    score = (score + " RET").strip()
                matches.append(EspnMatch(
                    competition_id=competition_id, event_id=str(event.get("id") or ""),
                    event_name=str(event.get("name") or ""), city=_city(event), major=major, tour=tour, date=date,
                    round_name=round_name, round_order=ROUND_ORDER.get(round_name.lower(), 9), qualifying=qualifying,
                    outcome=outcome, winner_name=winner_name, loser_name=loser_name, score=score,
                    sets_played=sets_played, games_played=games_played,
                    best_of=5 if (major and tour == "ATP" and not qualifying) else 3,
                ))
    return matches
