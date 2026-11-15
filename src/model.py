"""Pre-match rating model that feeds the point-by-point simulator.

State is updated one match at a time, in chronological order, so a prediction can
only use matches that came before it. Three pre-match signals are combined:

  1. Elo (overall, blended with surface-specific Elo)
  2. ATP/WTA ranking points as published before the tournament
  3. Surface-segmented serve and return point-win rates

The combined signal is a best-of-3 win probability. `serve_inputs` turns that into
the two serve-point probabilities the Markov simulator needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from src import sim

LN10_OVER_400 = math.log(10.0) / 400.0
SURFACES = ("Hard", "Clay", "Grass")
N_FEATURES = 3


@dataclass(frozen=True)
class ModelParams:
    k0: float = 250.0              # Elo K numerator
    k_offset: float = 5.0
    k_shape: float = 0.4           # K = k0 / (matches + k_offset) ** k_shape
    w_surface: float = 0.5         # weight of surface Elo in the blended rating
    init_slope: float = 150.0      # Elo points per unit of ln(ranking points) for a new player
    unranked_points: float = 15.0  # ranking points assumed when a player has none listed
    ret_weight: float = 0.5        # K multiplier for matches that ended in a retirement
    stats_decay: float = 0.97      # per-match decay of serve/return point counts
    stats_prior_pts: float = 400.0 # shrinkage strength, in points

    def to_dict(self) -> dict:
        return asdict(self)


class PlayerState:
    __slots__ = ("elo", "n", "surf_elo", "surf_n", "serve", "ret", "last_outcome", "last_won", "last_date",
                 "last_points")

    def __init__(self, elo: float):
        self.elo = elo
        self.n = 0
        self.surf_elo: dict[str, float] = {}
        self.surf_n: dict[str, int] = {}
        self.serve: dict[str, list[float]] = {}   # surface or "All" -> [points won, points played]
        self.ret: dict[str, list[float]] = {}
        self.last_outcome: str | None = None
        self.last_won: bool | None = None         # whether the player won (advanced from) that match
        self.last_date: str | None = None
        self.last_points: float | None = None     # ranking points listed at the player's most recent match


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


_BO5_GRID: tuple[np.ndarray, np.ndarray] | None = None


def bo3_to_bo5(p3: float) -> float:
    """Convert a best-of-3 win probability to best-of-5 using the exact Markov chain."""
    global _BO5_GRID
    if _BO5_GRID is None:
        gaps = np.linspace(-0.3, 0.3, 241)
        x = np.array([sim.match_win_prob(0.63 + g, 0.63 - g, 3) for g in gaps])
        y = np.array([sim.match_win_prob(0.63 + g, 0.63 - g, 5) for g in gaps])
        _BO5_GRID = (x, y)
    return float(np.interp(p3, _BO5_GRID[0], _BO5_GRID[1]))


class RatingModel:
    """Walk-forward state for one tour."""

    def __init__(self, params: ModelParams = ModelParams()):
        self.params = params
        self.players: dict[int, PlayerState] = {}
        self.betas = np.array([1.0, 0.0, 0.0])        # blend weights, set by the walk-forward
        self.tour_serve: dict[str, list[float]] = {}   # surface -> [serve points won, serve points played]

    # ---- state access ----
    def _player(self, player_id: int, rank_points: float | None) -> PlayerState:
        state = self.players.get(player_id)
        if state is None:
            pts = rank_points if rank_points is not None and rank_points > 0 else self.params.unranked_points
            state = PlayerState(1500.0 + self.params.init_slope * math.log((pts + 10.0) / 500.0))
            self.players[player_id] = state
        return state

    def _tour_serve_rate(self, surface: str) -> float:
        won, played = self.tour_serve.get(surface, (0.0, 0.0))
        return (won + 0.62 * 5000.0) / (played + 5000.0)

    def _rate(self, table: dict[str, list[float]], surface: str, tour_mean: float) -> float:
        prior = self.params.stats_prior_pts
        won, played = table.get("All", (0.0, 0.0))
        overall = (won + prior * tour_mean) / (played + prior)
        won_s, played_s = table.get(surface, (0.0, 0.0))
        return (won_s + prior * overall) / (played_s + prior)

    def serve_expectations(self, a: PlayerState, b: PlayerState, surface: str) -> tuple[float, float]:
        """Expected serve-point win rates (A serving, B serving) from serve/return history."""
        mu = self._tour_serve_rate(surface)
        f_a = mu + (self._rate(a.serve, surface, mu) - mu) - (self._rate(b.ret, surface, 1 - mu) - (1 - mu))
        f_b = mu + (self._rate(b.serve, surface, mu) - mu) - (self._rate(a.ret, surface, 1 - mu) - (1 - mu))
        return f_a, f_b

    # ---- prediction ----
    def features(self, id_a: int, id_b: int, surface: str,
                 points_a: float | None, points_b: float | None) -> tuple[np.ndarray, float]:
        """Pre-match features for A against B, plus the expected serve total.

        Feature vector: [Elo log-odds, ln ranking-points ratio, serve-rate gap].
        Every term flips sign when A and B swap, so the model is symmetric.
        """
        p = self.params
        a, b = self._player(id_a, points_a), self._player(id_b, points_b)
        surf_a, surf_b = a.surf_elo.get(surface, a.elo), b.surf_elo.get(surface, b.elo)
        diff = (1 - p.w_surface) * (a.elo - b.elo) + p.w_surface * (surf_a - surf_b)
        pts_a = points_a if points_a is not None and points_a > 0 else p.unranked_points
        pts_b = points_b if points_b is not None and points_b > 0 else p.unranked_points
        f_a, f_b = self.serve_expectations(a, b, surface)
        x = np.array([diff * LN10_OVER_400, math.log(pts_a + 10.0) - math.log(pts_b + 10.0), f_a - f_b])
        total = min(max(f_a + f_b, 1.05), 1.45)
        return x, total

    # ---- update ----
    def update(self, winner_id: int, loser_id: int, surface: str, best_of: int, outcome: str,
               tourney_date: str, winner_points: float | None, loser_points: float | None,
               w_svpt: float | None = None, w_won: float | None = None,
               l_svpt: float | None = None, l_won: float | None = None) -> None:
        """Fold one finished match into the state. Call only after predicting it."""
        p = self.params
        w, l = self._player(winner_id, winner_points), self._player(loser_id, loser_points)
        w.last_outcome = l.last_outcome = outcome
        w.last_won, l.last_won = True, False
        if winner_points is not None:
            w.last_points = winner_points
        if loser_points is not None:
            l.last_points = loser_points
        w.last_date = l.last_date = tourney_date
        if outcome == "W/O":
            return  # no ball was played, so there is nothing to learn about form

        weight = p.ret_weight if outcome != "COMPLETED" else 1.0

        def expected(rating_w: float, rating_l: float) -> float:
            p3 = _sigmoid((rating_w - rating_l) * LN10_OVER_400)
            return bo3_to_bo5(p3) if best_of == 5 else p3

        k_w = weight * p.k0 / (w.n + p.k_offset) ** p.k_shape
        k_l = weight * p.k0 / (l.n + p.k_offset) ** p.k_shape
        surprise = 1.0 - expected(w.elo, l.elo)
        sw, sl = w.surf_elo.get(surface, w.elo), l.surf_elo.get(surface, l.elo)
        surf_surprise = 1.0 - expected(sw, sl)
        ksw = weight * p.k0 / (w.surf_n.get(surface, 0) + p.k_offset) ** p.k_shape
        ksl = weight * p.k0 / (l.surf_n.get(surface, 0) + p.k_offset) ** p.k_shape

        w.elo += k_w * surprise
        l.elo -= k_l * surprise
        w.surf_elo[surface] = sw + ksw * surf_surprise
        l.surf_elo[surface] = sl - ksl * surf_surprise
        w.n += 1
        l.n += 1
        w.surf_n[surface] = w.surf_n.get(surface, 0) + 1
        l.surf_n[surface] = l.surf_n.get(surface, 0) + 1

        if w_svpt and l_svpt and w_won is not None and l_won is not None and w_svpt > 0 and l_svpt > 0:
            decay = p.stats_decay
            for key in ("All", surface):
                for state, srv_won, srv_pts, ret_won, ret_pts in (
                    (w, w_won, w_svpt, l_svpt - l_won, l_svpt),
                    (l, l_won, l_svpt, w_svpt - w_won, w_svpt),
                ):
                    s = state.serve.setdefault(key, [0.0, 0.0])
                    s[0], s[1] = s[0] * decay + srv_won, s[1] * decay + srv_pts
                    r = state.ret.setdefault(key, [0.0, 0.0])
                    r[0], r[1] = r[0] * decay + ret_won, r[1] * decay + ret_pts
            t = self.tour_serve.setdefault(surface, [0.0, 0.0])
            t[0] += w_won + l_won
            t[1] += w_svpt + l_svpt


def bo3_probability(x: np.ndarray, betas: np.ndarray) -> float:
    return _sigmoid(float(x @ betas))


def fit_betas(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Logistic regression with no intercept (keeps the model symmetric), by Newton's method."""
    betas = np.array([1.0, 0.0, 0.0])
    penalty = ridge * np.eye(x.shape[1])
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-(x @ betas)))
        grad = x.T @ (p - y) + ridge * (betas - np.array([1.0, 0.0, 0.0]))
        hess = (x * (p * (1 - p))[:, None]).T @ x + penalty
        step = np.linalg.solve(hess, grad)
        betas = betas - step
        if np.abs(step).max() < 1e-8:
            break
    return betas


def serve_inputs(p_bo3: float, serve_total: float) -> tuple[float, float]:
    """Serve-point probabilities whose exact best-of-3 win probability equals `p_bo3`."""
    return sim.solve_serve_probs(p_bo3, serve_total, best_of=3)
