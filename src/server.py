"""Local dashboard for CourtEdge.

    python -m src.server            # or ./start.sh, which also sets up the environment

Serves one page on http://127.0.0.1:8770. On first run it downloads the match data
and runs the backtest by itself. The server is read-only towards the outside world:
it downloads match data and never contacts Kalshi.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import datetime as dt

from src import backtest, ingest, kalshi_reader, risk_gate, sim, slate, vault_writer
from src.model import bo3_probability, serve_inputs

WEB_DIR = Path(__file__).resolve().parent / "web"
VAULT_DIR = ingest.REPO_ROOT / "vault"
DEFAULT_PORT = 8770
SURFACES = ("Hard", "Clay", "Grass")
ACTIVE_WINDOW_DAYS = 365

JOBS = {
    "setup": "Loading match data and running the backtest",
    "update": "Fetching new results and updating ratings",
    "startup": "Checking for new results",
    "scan": "Scanning Kalshi",
    "settle": "Settling paper trades",
    "backtest": "Running the fast backtest",
    "backtest-full": "Running the 10,000-trial simulator backtest",
}


class App:
    """Holds the database path, the single background job, and cached model state."""

    def __init__(self, db_path: Path = ingest.DEFAULT_DB, archive: Path = ingest.DEFAULT_ARCHIVE,
                 write_vault: bool = True):
        self.db_path = Path(db_path)
        self.archive = Path(archive)
        self.write_vault = write_vault
        self.lock = threading.Lock()
        self.job = {"name": None, "state": "idle", "message": "", "started": None, "finished": None}
        self._models = None
        self._performance_cache: tuple[str, dict] | None = None
        self.last_catch_up: dict | None = None
        self.fetch = None            # tests replace the live feed with a stub
        self.kalshi_fetch = kalshi_reader.get_json
        self.vault_dir = VAULT_DIR
        self.last_settle: dict | None = None

    # ---- database helpers ----
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def has_data(self) -> bool:
        if not self.db_path.exists():
            return False
        with self.connect() as conn:
            try:
                return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] > 0
            except sqlite3.OperationalError:
                return False

    def latest_run(self, conn: sqlite3.Connection) -> dict | None:
        try:
            row = conn.execute(
                "SELECT run_id, created_at, engine, n_matches, brier, max_calibration_gap, log_loss, accuracy, passed "
                "FROM backtest_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        run = dict(row)
        run["passed"] = bool(run["passed"])
        run["brier_passed"] = risk_gate.brier_gate_passed(run["brier"])
        run["calibration_passed"] = risk_gate.calibration_gate_passed(run["max_calibration_gap"])
        return run

    # ---- jobs ----
    def start_job(self, name: str) -> bool:
        if name not in JOBS:
            raise KeyError(name)
        with self.lock:
            if self.job["state"] == "running":
                return False
            self.job = {"name": name, "state": "running", "message": JOBS[name], "started": time.time(),
                        "finished": None}
        threading.Thread(target=self._run_job, args=(name,), daemon=True).start()
        return True

    def _say(self, message: str) -> None:
        self.job["message"] = message

    def _run_job(self, name: str) -> None:
        try:
            if name in ("scan", "settle"):
                self._run_trading_job(name)
                return
            if name == "setup":
                summary = ingest.ingest(self.archive, self.db_path, log=self._say)
                self._say(f"Loaded {summary['matches']:,} archive matches. Fetching newer results")
            if name in ("setup", "update", "startup"):
                self._models = None
                kwargs = {"fetch": self.fetch} if self.fetch else {}
                try:
                    result = ingest.catch_up(self.db_path, log=self._say, **kwargs)
                    self.last_catch_up = {k: result[k] for k in (
                        "added", "in_archive", "already_stored", "unmatched", "unknown_surface", "through")}
                    self.last_catch_up["unmatched_names"] = [
                        {"matches": count, "tour": tour, "name": player} for count, tour, player in result["unmatched_names"]]
                    self.last_catch_up["warning"] = result["errors"][0] if result["errors"] else None
                except ConnectionError as exc:
                    if name == "update":
                        raise
                    self.last_catch_up = {"warning": str(exc)}     # first run offline: continue on the archive alone
                self._say("Updating ratings and running the backtest")
            if name == "backtest-full":
                with self.connect() as conn:
                    total = conn.execute(
                        "SELECT COUNT(*) FROM matches WHERE tier = 'tour' AND season >= 2025 AND outcome != 'W/O'"
                    ).fetchone()[0]
                backtest.run(self.db_path, engine="sim", write_vault=self.write_vault,
                             progress=lambda n: n % 50 == 0 and self._say(f"Simulated {n:,} of {total:,} matches"))
            else:
                backtest.run(self.db_path, engine="analytic", write_vault=False, progress=False)
            self._performance_cache = None
            self.job.update(state="done", message="Finished", finished=time.time())
        except Exception as exc:  # shown in the dashboard's error bar
            traceback.print_exc()
            self.job.update(state="error", message=f"{type(exc).__name__}: {exc}", finished=time.time())

    def _run_trading_job(self, name: str) -> None:
        """Settle open paper trades, then (for a scan) read Kalshi and log new ones. GET requests only."""
        now = dt.datetime.now(dt.timezone.utc)
        with self.connect() as conn:
            slate.init_ledger(conn)
            self._say("Settling open paper trades")
            self.last_settle = slate.settle(conn, fetch=self.kalshi_fetch, now=now)
            payload = None
            if name == "scan":
                gate = slate.gate_status(conn)
                if not gate["open"]:
                    raise slate.GateClosed(gate["reason"] + " Run a backtest first.")
                self._say("Loading ratings")
                models = self.models()
                payload = slate.scan(conn, models, fetch=self.kalshi_fetch, now=now, log=self._say)
            if self.write_vault:
                self._say("Writing vault notes")
                vault_writer.write_all(self.vault_dir, conn, payload, slate.local_date(now))
        self.job.update(state="done", message="Finished", finished=time.time())

    def slate_view(self) -> dict:
        with self.connect() as conn:
            slate.init_ledger(conn)
            return {"scan": slate.latest_scan(conn), "gate": slate.gate_status(conn),
                    "bankroll": slate.bankroll(conn), "ledger": slate.ledger(conn, limit=50),
                    "last_settle": self.last_settle, "totals_enabled": slate.TOTALS_ENABLED,
                    "totals_reason": slate.TOTALS_DISABLED_REASON}

    # ---- API payloads ----
    def status(self) -> dict:
        payload = {
            "ready": False, "data": None, "run": None, "job": dict(self.job), "catch_up": self.last_catch_up,
            "limits": {"brier": risk_gate.MAX_BRIER_SCORE, "calibration_gap": risk_gate.MAX_CALIBRATION_GAP},
        }
        if self.job["started"]:
            payload["job"]["elapsed"] = (self.job["finished"] or time.time()) - self.job["started"]
        if not self.has_data():
            return payload
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*), MIN(tourney_date), MAX(tourney_date) FROM matches").fetchone()
            players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            live = conn.execute("SELECT COUNT(*) FROM matches WHERE source != 'archive'").fetchone()[0]
            skipped = conn.execute("SELECT COUNT(DISTINCT match_id) FROM ingest_unmatched").fetchone()[0]
            payload["data"] = {"matches": row[0], "first": row[1], "last": row[2], "players": players,
                               "live_matches": live, "skipped_matches": skipped}
            payload["run"] = self.latest_run(conn)
        payload["ready"] = payload["run"] is not None
        return payload

    def performance(self) -> dict:
        with self.connect() as conn:
            run = self.latest_run(conn)
            if run is None:
                return {"run": None, "tables": {}}
            if self._performance_cache and self._performance_cache[0] == run["run_id"]:
                return self._performance_cache[1]
            preds = pd.read_sql_query(
                "SELECT p.p_winner, m.tour, m.season, m.tier, m.surface, m.source FROM predictions p "
                "JOIN matches m USING (match_id) WHERE p.run_id = ? AND m.season >= 2025", conn, params=[run["run_id"]])
        archive = preds[preds["source"] == "archive"]
        live = preds[preds["source"] != "archive"]
        head = archive[archive["tier"] == "tour"]
        lower = archive[archive["tier"] == "lower"]

        def records(frame: pd.DataFrame) -> list[dict]:
            return json.loads(frame.to_json(orient="records"))

        tables = {
            "by_tour": records(backtest.summarize(head, ["tour"])),
            "by_season": records(backtest.summarize(head, ["tour", "season"])),
            "by_surface": records(backtest.summarize(head, ["surface"])),
            "calibration": records(backtest.calibration_table(head["p_winner"])),
            "lower": records(backtest.summarize(lower, ["tour"])) if len(lower) else [],
            "live": records(backtest.summarize(live, ["tour", "tier"])) if len(live) else [],
        }
        payload = {"run": run, "tables": tables}
        self._performance_cache = (run["run_id"], payload)
        return payload

    def players(self, tour: str, surface: str, query: str = "", limit: int = 100) -> list[dict]:
        min_matches = 10 if surface == "All" else 5
        with self.connect() as conn:
            latest = conn.execute("SELECT MAX(tourney_date) FROM matches WHERE tour = ?", (tour,)).fetchone()[0]
            if latest is None:
                return []
            cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=ACTIVE_WINDOW_DAYS)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT s.player_id, p.full_name, p.ioc, s.matches_played, s.elo, s.serve_pts_won, s.return_pts_won, "
                "s.last_tourney_date, s.last_outcome, s.last_match_won, "
                "COALESCE(st.matches_7d, 0) AS matches_7d, COALESCE(st.sets_7d, 0) AS sets_7d, "
                "COALESCE(st.games_7d, 0) AS games_7d, "
                "(SELECT COUNT(*) + 1 FROM player_surface_stats o WHERE o.tour = s.tour AND o.surface = s.surface "
                "   AND o.matches_played >= ? AND o.last_tourney_date >= ? AND o.elo > s.elo) AS position "
                "FROM player_surface_stats s JOIN players p ON p.tour = s.tour AND p.player_id = s.player_id "
                "LEFT JOIN player_status st ON st.tour = s.tour AND st.player_id = s.player_id "
                "WHERE s.tour = ? AND s.surface = ? AND s.matches_played >= ? AND s.last_tourney_date >= ? "
                "AND p.full_name LIKE ? ORDER BY s.elo DESC LIMIT ?",
                (min_matches, cutoff, tour, surface, min_matches, cutoff, f"%{query}%", limit)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            won = None if item["last_match_won"] is None else bool(item["last_match_won"])
            item["vetoed"] = risk_gate.is_vetoed_by_last_result(item["last_outcome"], won)
            out.append(item)
        return out

    def search(self, tour: str, query: str, limit: int = 12) -> list[dict]:
        if len(query.strip()) < 2:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT p.player_id, p.full_name, p.ioc FROM players p "
                "LEFT JOIN player_surface_stats s ON s.tour = p.tour AND s.player_id = p.player_id AND s.surface = 'All' "
                "WHERE p.tour = ? AND p.full_name LIKE ? ORDER BY COALESCE(s.elo, 0) DESC LIMIT ?",
                (tour, f"%{query.strip()}%", limit)).fetchall()
        return [dict(row) for row in rows]

    def models(self) -> dict:
        """Ratings as of the last ingested match, rebuilt on demand (a few seconds)."""
        with self.lock:
            if self._models is None:
                params = backtest.load_params()
                built = {}
                with self.connect() as conn:
                    for tour in ingest.TOURS:
                        _, built[tour] = backtest.walk_forward(backtest.load_matches(conn, tour), params[tour])
                self._models = built
            return self._models

    def simulate(self, tour: str, id_a: int, id_b: int, surface: str, best_of: int) -> dict:
        if tour not in ingest.TOURS or surface not in SURFACES or best_of not in (3, 5):
            raise ValueError("unknown tour, surface or format")
        if id_a == id_b:
            raise ValueError("pick two different players")
        model = self.models()[tour]
        if id_a not in model.players or id_b not in model.players:
            raise ValueError("no match history for one of these players")
        a, b = model.players[id_a], model.players[id_b]
        x, serve_total = model.features(id_a, id_b, surface, a.last_points, b.last_points)
        p_bo3 = bo3_probability(x, model.betas)
        serve_a, serve_b = serve_inputs(p_bo3, serve_total)
        # Seeded from the inputs so the same matchup always shows the same numbers.
        seed = (min(id_a, id_b) * 1_000_003 + max(id_a, id_b)) * 10 + SURFACES.index(surface) * 2 + (best_of == 5)
        result = sim.simulate_match(serve_a, serve_b, best_of=best_of, seed=seed)
        with self.connect() as conn:
            names = dict(conn.execute(
                "SELECT player_id, full_name FROM players WHERE tour = ? AND player_id IN (?, ?)",
                (tour, id_a, id_b)).fetchall())
            load = {row[0]: {"matches_7d": row[1], "sets_7d": row[2], "games_7d": row[3]} for row in conn.execute(
                "SELECT player_id, matches_7d, sets_7d, games_7d FROM player_status WHERE tour = ? AND player_id IN (?, ?)",
                (tour, id_a, id_b))}

        def side(player_id: int, state, serve: float) -> dict:
            return {
                "player_id": player_id, "name": names.get(player_id, str(player_id)),
                "elo": round(state.elo), "surface_elo": round(state.surf_elo.get(surface, state.elo)),
                "matches": state.n, "surface_matches": state.surf_n.get(surface, 0),
                "serve_point_prob": serve, "last_date": state.last_date, "last_outcome": state.last_outcome,
                "load": load.get(player_id, {"matches_7d": 0, "sets_7d": 0, "games_7d": 0}),
                "vetoed": risk_gate.is_vetoed_by_last_result(state.last_outcome, state.last_won),
            }

        games, counts = np.unique(result.total_games, return_counts=True)
        return {
            "tour": tour, "surface": surface, "best_of": best_of, "trials": result.n_trials,
            "a": side(id_a, a, serve_a), "b": side(id_b, b, serve_b),
            "p_a": result.p_a_win, "p_b": result.p_b_win, "standard_error": result.standard_error,
            "p_a_first_set": result.p_a_first_set, "mean_total_games": result.mean_total_games,
            "set_scores": result.set_score_probs,
            "total_games": [{"games": int(g), "prob": float(c) / result.n_trials} for g, c in zip(games, counts)],
        }

    @staticmethod
    def guardrails() -> dict:
        return {
            "MIN_COMBO_HIT_PROB": risk_gate.MIN_COMBO_HIT_PROB, "MIN_LEG_EDGE": risk_gate.MIN_LEG_EDGE,
            "MIN_COMBO_EDGE": risk_gate.MIN_COMBO_EDGE,
            "MIN_NAME_MATCH_CONFIDENCE": risk_gate.MIN_NAME_MATCH_CONFIDENCE,
            "VETO_LAST_RESULTS": list(risk_gate.VETO_LAST_RESULTS),
            "MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE": risk_gate.MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE,
            "MAX_BRIER_SCORE": risk_gate.MAX_BRIER_SCORE, "MAX_CALIBRATION_GAP": risk_gate.MAX_CALIBRATION_GAP,
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:   # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _guarded(self, fn) -> None:
            try:
                fn()
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            routes = {
                "/api/status": app.status,
                "/api/performance": app.performance,
                "/api/guardrails": app.guardrails,
                "/api/players": lambda: app.players(query.get("tour", "ATP"), query.get("surface", "All"),
                                                    query.get("q", "")),
                "/api/search": lambda: app.search(query.get("tour", "ATP"), query.get("q", "")),
                "/api/slate": app.slate_view,
            }
            if url.path in routes:
                return self._guarded(lambda: self._json(routes[url.path]()))
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if url.path.startswith("/api/jobs/"):
                name = url.path.rsplit("/", 1)[1]
                if name not in JOBS:
                    return self._json({"error": "unknown job"}, 404)
                started = app.start_job(name)
                return self._json({"started": started, "job": app.job}, 200 if started else 409)
            if url.path == "/api/simulate":
                return self._guarded(lambda: self._json(app.simulate(
                    str(body.get("tour")), int(body.get("a")), int(body.get("b")),
                    str(body.get("surface")), int(body.get("best_of", 3)))))
            self._json({"error": "not found"}, 404)

    return Handler


def serve(port: int = DEFAULT_PORT, open_browser: bool = True, app: App | None = None) -> None:
    app = app or App()
    httpd = None
    for candidate in range(port, port + 20):     # use the next free port if this one is busy
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), make_handler(app))
            break
        except OSError:
            continue
    if httpd is None:
        raise RuntimeError(f"No free port between {port} and {port + 19}")
    address = f"http://127.0.0.1:{httpd.server_address[1]}"
    if not app.has_data():
        app.start_job("setup")
    else:
        app.start_job("startup")     # pull anything new each time the dashboard opens
    print(f"CourtEdge dashboard: {address}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
