"""Write Obsidian notes into vault/.

Every note starts with YAML frontmatter holding exactly the keys the vault standard
requires (date, tour, surface, legs, combo_hit_prob, edge_pct, status), and players
are linked as [[First Last]]. Player notes are named "First Last.md" so those links
resolve in Obsidian.

    vault/Daily_Slates/YYYY-MM-DD.md      the day's qualifying combos and legs
    vault/Players/First Last.md           surface stats, 7-day workload, retirement flag
    vault/Winners/YYYY-MM-DD Combo N.md   one note per paper combo that won
    vault/Post_Mortems/Paper_Ledger.md    calibration audit of settled paper legs
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from pathlib import Path

import numpy as np

from src import risk_gate, slate

FRONTMATTER_KEYS = ("date", "tour", "surface", "legs", "combo_hit_prob", "edge_pct", "status")
_UNSAFE = re.compile(r'[\\/:*?"<>|#^\[\]]')


def yaml_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value == value else "null"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_+\-. ]+", text) and text.strip() == text and text.lower() not in ("null", "true", "false"):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def frontmatter(**fields) -> str:
    missing = [key for key in FRONTMATTER_KEYS if key not in fields]
    if missing:
        raise ValueError(f"frontmatter is missing {missing}")
    return "---\n" + "".join(f"{key}: {yaml_value(fields[key])}\n" for key in FRONTMATTER_KEYS) + "---\n"


def link(name: str | None) -> str:
    return f"[[{safe_name(name)}]]" if name else "(unmatched)"


def safe_name(name: str) -> str:
    return _UNSAFE.sub("", name).strip() or "Unknown"


def pct(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _tour_and_surface(legs: list[dict]) -> tuple[str, str]:
    tours = sorted({leg["tour"] for leg in legs}) or ["ATP+WTA"]
    surfaces = sorted({leg.get("surface") or "Unknown" for leg in legs}) or ["All"]
    return "+".join(tours), surfaces[0] if len(surfaces) == 1 else "Mixed"


def _leg_line(leg: dict) -> str:
    who = (f"{link(leg.get('player_name'))} to beat {link(leg.get('opponent_name'))}" if leg["kind"] == "winner"
           else f"{leg['label']}: {link(leg.get('player_name') or leg.get('player'))} vs "
                f"{link(leg.get('opponent_name') or leg.get('opponent'))}")
    edge = "–" if leg.get("edge") is None else f"{leg['edge'] * 100:+.1f} points"
    return (f"{who} ({leg['tour']}, {leg.get('tournament') or 'tournament unknown'}, {leg.get('surface')}) "
            f"model {pct(leg.get('simulated_prob'))}, ask {pct(leg.get('kalshi_ask'), 0)}, edge {edge}")


# --------------------------------------------------------------------------- daily slate

def write_daily_slate(vault: Path, payload: dict) -> Path:
    approved = [c for c in payload["combos"] if c["approved"]]
    best = approved[0] if approved else None
    all_legs = [leg for c in approved for leg in c["legs"]] or [leg for leg in payload["legs"] if leg["qualifies"]]
    tour, surface = _tour_and_surface(all_legs)
    head = frontmatter(
        date=payload["slate_date"], tour=tour, surface=surface, legs=len(best["legs"]) if best else 0,
        combo_hit_prob=round(best["fair_combo_prob"], 4) if best else None,
        edge_pct=round(best["edge"] * 100, 2) if best else None,
        status="combos" if approved else "no_qualifying_combos")
    counts = payload["counts"]
    lines = [head, f"# Slate {payload['slate_date']}", "",
             f"Scanned {dt.datetime.fromisoformat(payload['created_at']).astimezone():%Y-%m-%d %H:%M %Z}. "
             f"{counts['matches']} matches, "
             f"{counts['legs']} legs, {counts['qualifying_legs']} qualifying, {counts['combos_approved']} combos "
             f"approved, {counts['paper_trades_logged']} paper-logged. Rules: [[Hard_Guardrails]].", ""]
    for warning in payload.get("warnings") or []:
        lines.append(f"> [!warning] {warning}")
    lines += ["## Combos", ""]
    if not approved:
        lines += ["No combo passed the gate.", ""]
    for number, combo in enumerate(approved, 1):
        source = ("Kalshi combo market " + combo["kalshi_combo_ticker"] if combo["price_source"] == slate.PRICE_SOURCE_COMBO
                  else "product of the legs' asks (estimate)")
        lines += [f"### {number}. {len(combo['legs'])} legs{' — paper-logged' if combo['logged'] else ''}", "",
                  f"- Hit chance {pct(combo['fair_combo_prob'])}, price {pct(combo['price'])} from {source}, "
                  f"edge {combo['edge'] * 100:+.1f} points"]
        lines += [f"- {_leg_line(leg)}" for leg in combo["legs"]]
        lines.append("")
    lines += ["## Legs", "", "| Leg | Match | Model | Ask | Edge | Status |", "|---|---|---|---|---|---|"]
    for leg in payload["legs"]:
        match = f"{link(leg.get('player_name') or leg['player'])} vs {link(leg.get('opponent_name') or leg['opponent'])}"
        status = "qualifies" if leg["qualifies"] else "; ".join(leg["reasons"])
        edge = "–" if leg.get("edge") is None else f"{leg['edge'] * 100:+.1f}"
        lines.append(f"| {leg['label']} | {match} | {pct(leg.get('simulated_prob'))} | {pct(leg.get('kalshi_ask'), 0)} | "
                     f"{edge} | {status} |")
    return _write(vault / "Daily_Slates" / f"{payload['slate_date']}.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- players

def write_player_card(vault: Path, conn: sqlite3.Connection, tour: str, player_id: int, today: dt.date) -> Path | None:
    row = conn.execute("SELECT full_name, ioc, hand FROM players WHERE tour = ? AND player_id = ?",
                       (tour, player_id)).fetchone()
    if row is None:
        return None
    name, ioc, hand = row
    status = conn.execute("SELECT last_match_date, last_outcome, last_match_won, last_match_ret_wo, matches_7d, sets_7d, "
                          "games_7d, as_of FROM player_status WHERE tour = ? AND player_id = ?", (tour, player_id)).fetchone()
    stats = conn.execute("SELECT surface, matches_played, elo, serve_pts_won, return_pts_won FROM player_surface_stats "
                         "WHERE tour = ? AND player_id = ? ORDER BY surface = 'All' DESC, matches_played DESC",
                         (tour, player_id)).fetchall()
    vetoed = bool(status and status[3])
    main_surface = next((s[0] for s in stats if s[0] != "All"), "All")
    lines = [frontmatter(date=today.isoformat(), tour=tour, surface="All", legs=0, combo_hit_prob=None, edge_pct=None,
                         status="vetoed" if vetoed else "active"),
             f"# {name}", "", f"{tour}, {ioc or 'country unknown'}, {'right' if hand == 'R' else 'left' if hand == 'L' else 'unknown'}-handed. "
             f"Most matches on {main_surface}.", ""]
    if status:
        last_date, outcome, won, _flag, m7, s7, g7, as_of = status
        result = ("won" if won else "lost") + ("" if outcome == "COMPLETED" else f" by {outcome}")
        lines += ["## Status", "", f"- Last match {last_date}: {result}.",
                  f"- Last 7 days (to {as_of}): {m7} matches, {s7} sets, {g7} games." if m7 else
                  f"- Last 7 days (to {as_of}): no matches.",
                  "- **Vetoed**: lost the last match by retirement or walkover, so no combo may include this player."
                  if vetoed else "- Not vetoed.", ""]
    if stats:
        lines += ["## Ratings by surface", "", "| Surface | Matches | Rating | Serve points won | Return points won |",
                  "|---|---|---|---|---|"]
        lines += [f"| {s} | {n} | {elo:.0f} | {pct(sv)} | {pct(rt)} |" for s, n, elo, sv, rt in stats]
        lines.append("")
    recent = conn.execute(
        "SELECT tourney_date, tourney_name, round, winner_id, winner_name, loser_name, score FROM matches "
        "WHERE tour = ? AND (winner_id = ? OR loser_id = ?) ORDER BY tourney_date DESC, round_order DESC LIMIT 8",
        (tour, player_id, player_id)).fetchall()
    if recent:
        lines += ["## Recent matches", ""]
        lines += [f"- {d} {t} {r}: {'beat ' + link(l) if w == player_id else 'lost to ' + link(wn)} {score}"
                  for d, t, r, w, wn, l, score in recent]
    return _write(vault / "Players" / f"{safe_name(name)}.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- results

def write_winner(vault: Path, combo: dict) -> Path:
    tour, surface = _tour_and_surface(combo["legs"])
    lines = [frontmatter(date=combo["slate_date"], tour=tour, surface=surface, legs=combo["n_legs"],
                         combo_hit_prob=round(combo["fair_combo_prob"], 4), edge_pct=round(combo["edge"] * 100, 2),
                         status="won"),
             f"# Combo {combo['combo_id']} won", "",
             f"Paper stake ${combo['stake_usd']:.2f} at {pct(combo['kalshi_combo_price'])} "
             f"({'Kalshi combo market' if combo['price_source'] == slate.PRICE_SOURCE_COMBO else 'leg-product estimate'}), "
             f"paid ${combo['payout_usd']:.2f}. Logged {combo['created_at'][:16].replace('T', ' ')} UTC, settled "
             f"{(combo['settled_at'] or '')[:16].replace('T', ' ')} UTC.", "", "## Legs", ""]
    lines += [f"- {link(leg['player_name'])} to beat {link(leg['opponent_name'])}: model {pct(leg['simulated_prob'])}, "
              f"ask {pct(leg['kalshi_ask'], 0)}, result {leg['result']}" for leg in combo["legs"]]
    return _write(vault / "Winners" / f"{combo['slate_date']} Combo {combo['combo_id']}.md", "\n".join(lines) + "\n")


def write_ledger_audit(vault: Path, conn: sqlite3.Connection, today: dt.date) -> Path:
    book = slate.bankroll(conn)
    legs = conn.execute("SELECT simulated_prob, kalshi_ask, result FROM paper_legs WHERE result IN ('yes', 'no')").fetchall()
    n = len(legs)
    ready = n >= risk_gate.MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE
    lines = [frontmatter(date=today.isoformat(), tour="ATP+WTA", surface="All", legs=n, combo_hit_prob=None,
                         edge_pct=None, status="reviewable" if ready else "collecting"),
             "# Paper ledger audit", "",
             f"Bankroll ${book['balance']:.2f} (started ${book['start']:.2f}), ${book['at_risk']:.2f} at risk in "
             f"{book['open']} open combos. Settled: {book['won']} won, {book['lost']} lost, {book['void']} void, "
             f"profit ${book['profit']:+.2f}.", ""]
    if not n:
        lines += ["No legs have settled yet.", ""]
    else:
        p = np.array([row[0] for row in legs])
        ask = np.array([row[1] for row in legs])
        y = np.array([1.0 if row[2] == "yes" else 0.0 for row in legs])
        lines += ["## Calibration of settled legs", "",
                  f"- Settled legs: {n}. Model Brier {np.mean((p - y) ** 2):.4f}; Kalshi ask Brier {np.mean((ask - y) ** 2):.4f} "
                  "(lower is better).",
                  f"- Predicted hit rate {p.mean():.1%}, actual {y.mean():.1%}, average ask {ask.mean():.1%}.",
                  f"- [[Active_Playbook]] changes need {risk_gate.MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE} settled legs: "
                  + ("reached." if ready else f"{risk_gate.MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE - n} to go."), ""]
        bands = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
        lines += ["| Model chance | Legs | Predicted | Won |", "|---|---|---|---|"]
        for lo, hi in bands:
            mask = (p >= lo) & (p < hi)
            if mask.any():
                lines.append(f"| {lo:.0%}–{min(hi, 1):.0%} | {int(mask.sum())} | {p[mask].mean():.1%} | {y[mask].mean():.1%} |")
        lines.append("")
    return _write(vault / "Post_Mortems" / "Paper_Ledger.md", "\n".join(lines) + "\n")


def write_all(vault: Path, conn: sqlite3.Connection, payload: dict | None, today: dt.date) -> list[Path]:
    """The slate, cards for every player in a qualifying leg, new winners, and the audit."""
    written: list[Path] = []
    if payload:
        written.append(write_daily_slate(vault, payload))
        players = {(leg["tour"], pid) for leg in payload["legs"] if leg["qualifies"]
                   for pid in (leg.get("player_id"), leg.get("opponent_id")) if pid}
        players |= {(leg["tour"], pid) for combo in payload["combos"] if combo["approved"] for leg in combo["legs"]
                    for pid in (leg.get("player_id"), leg.get("opponent_id")) if pid}
        written += [path for tour, pid in sorted(players) if (path := write_player_card(vault, conn, tour, pid, today))]
    for combo in slate.ledger(conn, limit=1000):
        if combo["status"] == "won":
            target = vault / "Winners" / f"{combo['slate_date']} Combo {combo['combo_id']}.md"
            if not target.exists():
                written.append(write_winner(vault, combo))
    written.append(write_ledger_audit(vault, conn, today))
    return written
