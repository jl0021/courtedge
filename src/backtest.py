"""Blind walk-forward backtest for Milestone 1.

Every match is predicted using only matches that were played before it, then the
match is folded into the model state. Model hyperparameters are tuned on the 2024
season only. The 2025-2026 seasons are never used for tuning, so their score is blind.

Usage:
    python -m src.backtest --tune        # tune on 2024, save config/model_params.json
    python -m src.backtest               # walk-forward with the 10k-trial simulator
    python -m src.backtest --engine analytic   # same model, exact chain instead of sampling
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src import sim
from src.ingest import DEFAULT_DB, REPO_ROOT, init_db
from src.model import ModelParams, RatingModel, bo3_probability, bo3_to_bo5, fit_betas, serve_inputs
from src.risk_gate import (CALIBRATION_BUCKETS, MAX_BRIER_SCORE, MAX_CALIBRATION_GAP, brier_gate_passed,
                           calibration_gate_passed, milestone1_gate_passed)

PARAMS_PATH = REPO_ROOT / "config" / "model_params.json"
VAULT_POST_MORTEMS = REPO_ROOT / "vault" / "Post_Mortems"

TUNE_SEASON = 2024
TEST_SEASONS = (2025, 2026)
TUNE_SCORE_FROM = "2024-03-01"      # let ratings warm up before scoring tuning candidates
MIN_FIT_ROWS = 300
DEFAULT_BETAS = np.array([1.0, 0.0, 0.0])
PROB_CLIP = (0.005, 0.995)

MATCH_COLUMNS = (
    "match_id, tour, season, tier, tourney_level, tourney_date, surface, best_of, outcome, "
    "winner_id, loser_id, winner_rank_points, loser_rank_points, "
    "w_svpt, w_1stWon, w_2ndWon, l_svpt, l_1stWon, l_2ndWon"
)


# --------------------------------------------------------------------------- metrics

def brier_score(p_winner) -> float:
    """Mean squared error of the probability given to the eventual winner."""
    p = np.asarray(p_winner, dtype=float)
    return float(np.mean((1.0 - p) ** 2))


def log_loss(p_winner) -> float:
    return float(-np.mean(np.log(np.asarray(p_winner, dtype=float))))


def accuracy(p_winner) -> float:
    p = np.asarray(p_winner, dtype=float)
    return float(np.mean(p > 0.5) + 0.5 * np.mean(p == 0.5))


def calibration_table(p_winner, bins: int = CALIBRATION_BUCKETS) -> pd.DataFrame:
    """Calibration of the favourite: predicted win rate against actual win rate."""
    p = np.asarray(p_winner, dtype=float)
    fav_prob = np.maximum(p, 1 - p)
    fav_won = (p > 0.5).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (fav_prob >= lo) & ((fav_prob < hi) | (hi == 1.0))
        if mask.any():
            rows.append({"band": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                         "predicted": float(fav_prob[mask].mean()), "actual": float(fav_won[mask].mean()),
                         "brier": float(np.mean((1 - p[mask]) ** 2))})
    table = pd.DataFrame(rows)
    table.insert(4, "gap", table["actual"] - table["predicted"])
    return table


def max_calibration_gap(p_winner) -> float:
    """Largest absolute gap between predicted and actual win rate across the buckets."""
    return float(calibration_table(p_winner)["gap"].abs().max())


# --------------------------------------------------------------------------- walk-forward

def load_matches(conn: sqlite3.Connection, tour: str, seasons=None) -> pd.DataFrame:
    sql = f"SELECT {MATCH_COLUMNS} FROM matches WHERE tour = ?"
    args: list = [tour]
    if seasons:
        sql += f" AND season IN ({','.join('?' * len(seasons))})"
        args += list(seasons)
    sql += " ORDER BY tourney_date, tourney_id, round_order, match_num, match_id"
    return pd.read_sql_query(sql, conn, params=args)


def _num(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def walk_forward(matches: pd.DataFrame, params: ModelParams, engine: str = "analytic",
                 sim_seasons=TEST_SEASONS, sim_tiers=("tour",), n_trials: int = sim.DEFAULT_TRIALS,
                 progress=False) -> tuple[pd.DataFrame, RatingModel]:
    """Predict then update, one match at a time, in chronological order.

    `matches` must already be sorted chronologically. Walkovers are not predicted,
    because no ball is played. Retirements are, because Kalshi settles them.

    The blend weights are refit at the start of each calendar month from earlier
    tour-level matches only. With engine="sim", matches in `sim_seasons` and
    `sim_tiers` are scored by the 10k-trial point-by-point simulator. All other
    matches use the exact solution of the same chain.
    """
    if engine not in ("analytic", "sim"):
        raise ValueError("engine must be 'analytic' or 'sim'")
    model = RatingModel(params)
    betas = DEFAULT_BETAS.copy()
    fit_x: list[np.ndarray] = []
    fit_y: list[float] = []
    current_month = None
    out = []
    n_simulated = 0

    for r in matches.itertuples(index=False):
        month = r.tourney_date[:7]
        if month != current_month:
            current_month = month
            if len(fit_y) >= MIN_FIT_ROWS:
                betas = fit_betas(np.array(fit_x), np.array(fit_y))

        w_pts, l_pts = _num(r.winner_rank_points), _num(r.loser_rank_points)
        if r.outcome != "W/O":
            a_is_winner = r.winner_id < r.loser_id
            if a_is_winner:
                x, serve_total = model.features(r.winner_id, r.loser_id, r.surface, w_pts, l_pts)
            else:
                x, serve_total = model.features(r.loser_id, r.winner_id, r.surface, l_pts, w_pts)
            p3 = bo3_probability(x, betas)
            use_sim = engine == "sim" and r.season in sim_seasons and r.tier in sim_tiers
            if use_sim:
                p_a_serve, p_b_serve = serve_inputs(p3, serve_total)
                result = sim.simulate_match(
                    p_a_serve, p_b_serve, best_of=int(r.best_of), n_trials=n_trials, seed=int(r.match_id),
                    final_set_tb_points=10 if r.tourney_level == "G" else 7,
                )
                p_a = result.p_a_win
                n_simulated += 1
                if callable(progress):
                    progress(n_simulated)
                elif progress and n_simulated % 1000 == 0:
                    print(f"    simulated {n_simulated:,} matches", flush=True)
            else:
                p_a = bo3_to_bo5(p3) if r.best_of == 5 else p3
            p_a = min(max(p_a, PROB_CLIP[0]), PROB_CLIP[1])
            out.append((r.match_id, r.season, r.tier, r.surface, r.best_of, r.outcome,
                        p_a if a_is_winner else 1.0 - p_a, "sim" if use_sim else "analytic"))
            if r.tier == "tour" and r.best_of == 3:
                fit_x.append(x)
                fit_y.append(1.0 if a_is_winner else 0.0)

        w_svpt, l_svpt = _num(r.w_svpt), _num(r.l_svpt)
        w_first, w_second = _num(r.w_1stWon), _num(r.w_2ndWon)
        l_first, l_second = _num(r.l_1stWon), _num(r.l_2ndWon)
        has_stats = None not in (w_svpt, l_svpt, w_first, w_second, l_first, l_second)
        model.update(
            r.winner_id, r.loser_id, r.surface, int(r.best_of), r.outcome, r.tourney_date, w_pts, l_pts,
            w_svpt if has_stats else None, (w_first + w_second) if has_stats else None,
            l_svpt if has_stats else None, (l_first + l_second) if has_stats else None,
        )

    if len(fit_y) >= MIN_FIT_ROWS:
        betas = fit_betas(np.array(fit_x), np.array(fit_y))   # weights for predicting matches still to come
    model.betas = betas
    preds = pd.DataFrame(out, columns=["match_id", "season", "tier", "surface", "best_of", "outcome",
                                       "p_winner", "engine"])
    return preds, model


# --------------------------------------------------------------------------- tuning (2024 only)

TUNE_GRID = {
    "k0": (150.0, 200.0, 250.0, 300.0),
    "k_shape": (0.3, 0.4, 0.5),
    "w_surface": (0.3, 0.5, 0.7),
    "init_slope": (100.0, 150.0, 200.0),
}


def tune_params(matches_2024: pd.DataFrame, verbose: bool = False) -> tuple[ModelParams, float]:
    """Grid-search Elo hyperparameters on the 2024 season only."""
    if (matches_2024["season"] != TUNE_SEASON).any():
        raise ValueError("tuning data must contain the 2024 season only")
    best_params, best_score = ModelParams(), math.inf
    keys = list(TUNE_GRID)
    for values in itertools.product(*(TUNE_GRID[k] for k in keys)):
        candidate = ModelParams(**dict(zip(keys, values)))
        preds, _ = walk_forward(matches_2024, candidate, engine="analytic")
        scored = preds[(preds["tier"] == "tour")
                       & preds["match_id"].isin(matches_2024.loc[matches_2024["tourney_date"] >= TUNE_SCORE_FROM, "match_id"])]
        score = brier_score(scored["p_winner"])
        if score < best_score:
            best_params, best_score = candidate, score
            if verbose:
                print(f"    {dict(zip(keys, values))} -> {score:.5f}", flush=True)
    return best_params, best_score


def load_params(path: Path = PARAMS_PATH) -> dict[str, ModelParams]:
    if not path.exists():
        return {"ATP": ModelParams(), "WTA": ModelParams()}
    saved = json.loads(path.read_text())
    return {tour: ModelParams(**values) for tour, values in saved["params"].items()}


# --------------------------------------------------------------------------- reporting

def summarize(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = preds.groupby(by)["p_winner"]
    return pd.DataFrame({
        "n": grouped.size(), "brier": grouped.apply(brier_score),
        "log_loss": grouped.apply(log_loss), "accuracy": grouped.apply(accuracy),
    }).reset_index()


def write_snapshot(conn: sqlite3.Connection, tour: str, model: RatingModel, as_of: str) -> None:
    rows = []
    for player_id, state in model.players.items():
        surfaces = ["All"] + sorted(state.surf_n)
        for surface in surfaces:
            mu = model._tour_serve_rate(surface if surface != "All" else "Hard")
            serve = state.serve.get(surface)
            ret = state.ret.get(surface)
            rows.append((
                tour, player_id, surface,
                state.n if surface == "All" else state.surf_n[surface],
                state.elo if surface == "All" else state.surf_elo[surface],
                serve[0] / serve[1] if serve and serve[1] > 0 else None,
                ret[0] / ret[1] if ret and ret[1] > 0 else None,
                state.last_date, state.last_outcome,
                None if state.last_won is None else int(state.last_won), as_of,
            ))
    conn.execute("DELETE FROM player_surface_stats WHERE tour = ?", (tour,))
    conn.executemany("INSERT INTO player_surface_stats VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)


def write_post_mortem(path: Path, run_id: str, engine: str, headline: pd.DataFrame, tables: dict[str, pd.DataFrame],
                      passed: bool, brier: float, gap: float) -> None:
    today = dt.date.today().isoformat()
    lines = [
        "---", f"date: {today}", "tour: ATP+WTA", "surface: All", "legs: 0", "combo_hit_prob: null",
        "edge_pct: null", f"status: {'passed' if passed else 'failed'}", "---", "",
        f"# Milestone 1 backtest ({today})", "",
        f"Blind walk-forward on the 2025-2026 seasons, tour-level main draws, engine `{engine}`. Run `{run_id}`.", "",
        f"**Milestone 1 gate: {'PASS' if passed else 'FAIL'}.**", "",
        f"- Brier score {brier:.4f}. Ceiling: below {MAX_BRIER_SCORE}. "
        f"{'Pass' if brier_gate_passed(brier) else 'Fail'}.",
        f"- Largest calibration gap {gap:.4f}. Limit: {MAX_CALIBRATION_GAP} in every bucket. "
        f"{'Pass' if calibration_gate_passed(gap) else 'Fail'}.", "",
        "See [[Hard_Guardrails]] for the gate.", "",
    ]
    for title, table in tables.items():
        lines += [f"## {title}", "", table.to_markdown(index=False, floatfmt=".4f"), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(db_path: Path = DEFAULT_DB, engine: str = "sim", tune: bool = False, n_trials: int = sim.DEFAULT_TRIALS,
        write_vault: bool = True, progress=True) -> dict:
    with sqlite3.connect(db_path) as conn:
        init_db(conn)
        if tune:
            tuned, scores = {}, {}
            for tour in ("ATP", "WTA"):
                print(f"Tuning {tour} on {TUNE_SEASON} only...", flush=True)
                tuned[tour], scores[tour] = tune_params(load_matches(conn, tour, [TUNE_SEASON]), verbose=True)
            PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
            PARAMS_PATH.write_text(json.dumps({
                "tuned_on_season": TUNE_SEASON, "scored_from": TUNE_SCORE_FROM,
                "tuning_brier": scores, "params": {t: p.to_dict() for t, p in tuned.items()},
            }, indent=2) + "\n")
        params = load_params()

        frames = []
        as_of = dt.date.today().isoformat()
        for tour in ("ATP", "WTA"):
            print(f"Walk-forward {tour} ({engine})...", flush=True)
            matches = load_matches(conn, tour)
            done_before = sum(int((f["engine"] == "sim").sum()) for f in frames)
            tour_progress = (lambda n, base=done_before: progress(base + n)) if callable(progress) else progress
            preds, model = walk_forward(matches, params[tour], engine=engine, n_trials=n_trials,
                                        progress=tour_progress)
            preds.insert(1, "tour", tour)
            frames.append(preds)
            write_snapshot(conn, tour, model, as_of)
        preds = pd.concat(frames, ignore_index=True)

        test = preds[preds["season"].isin(TEST_SEASONS)]
        headline_set = test[test["tier"] == "tour"]
        if headline_set.empty:
            raise RuntimeError("No tour-level 2025-2026 matches in the database. Run the ingest first.")
        brier = brier_score(headline_set["p_winner"])
        gap = max_calibration_gap(headline_set["p_winner"])
        passed = milestone1_gate_passed(brier, gap)
        se = float(np.std((1 - headline_set["p_winner"]) ** 2, ddof=1) / math.sqrt(len(headline_set)))

        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{engine}"
        conn.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
        conn.executemany(
            "INSERT INTO predictions (run_id, match_id, engine, p_winner) VALUES (?,?,?,?)",
            [(run_id, int(m), e, float(p)) for m, e, p in zip(preds["match_id"], preds["engine"], preds["p_winner"])],
        )
        conn.execute(
            "INSERT INTO backtest_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, dt.datetime.now(dt.timezone.utc).isoformat(), engine, len(headline_set), brier, gap,
             log_loss(headline_set["p_winner"]), accuracy(headline_set["p_winner"]), int(passed),
             json.dumps({t: p.to_dict() for t, p in params.items()})),
        )
        conn.commit()

    tables = {
        "Headline: tour-level main draw, 2025-2026": summarize(headline_set.assign(scope="ATP+WTA"), ["scope"]),
        "By tour": summarize(headline_set, ["tour"]),
        "By tour and season": summarize(headline_set, ["tour", "season"]),
        "By surface": summarize(headline_set, ["surface"]),
        "Calibration of the favourite": calibration_table(headline_set["p_winner"]),
        "Lower tier (qualifying, Challenger, ITF), 2025-2026, exact chain": summarize(test[test["tier"] == "lower"], ["tour"]),
    }
    for title, table in tables.items():
        print(f"\n{title}\n{table.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\nBrier {brier:.4f} (standard error {se:.4f}) on {len(headline_set):,} matches. "
          f"Ceiling < {MAX_BRIER_SCORE}: {'PASS' if brier_gate_passed(brier) else 'FAIL'}")
    print(f"Largest calibration gap {gap:.4f}. Limit {MAX_CALIBRATION_GAP}: "
          f"{'PASS' if calibration_gate_passed(gap) else 'FAIL'}")
    print(f"Milestone 1 gate: {'PASS' if passed else 'FAIL'}")

    if write_vault:
        write_post_mortem(VAULT_POST_MORTEMS / f"Backtest_{as_of}.md", run_id, engine, headline_set, tables, passed, brier, gap)
    return {"run_id": run_id, "brier": brier, "max_calibration_gap": gap, "standard_error": se, "n": len(headline_set), "passed": passed,
            "tables": tables}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--engine", choices=("sim", "analytic"), default="sim")
    parser.add_argument("--tune", action="store_true", help="re-tune hyperparameters on the 2024 season first")
    parser.add_argument("--trials", type=int, default=sim.DEFAULT_TRIALS)
    parser.add_argument("--no-vault", action="store_true")
    args = parser.parse_args()
    result = run(args.db, args.engine, args.tune, args.trials, write_vault=not args.no_vault)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
