"""Point-by-point Markov tennis simulator.

A match is a Markov chain over (points, games, sets, server). The only inputs are
each player's probability of winning a point on their own serve. `simulate_match`
plays 10,000 full matches point by point. `match_win_prob` solves the same chain
exactly and is used to check the simulator and to calibrate its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

DEFAULT_TRIALS = 10_000
MIN_SERVE_PROB, MAX_SERVE_PROB = 0.05, 0.95


# --------------------------------------------------------------------------- simulation

@dataclass
class SimResult:
    p_a_win: float
    n_trials: int
    best_of: int
    set_score_probs: dict[str, float]        # e.g. {"2-0": 0.41, "2-1": 0.22, "1-2": ..., "0-2": ...}
    p_a_first_set: float
    mean_total_games: float
    total_games: np.ndarray = field(repr=False)

    @property
    def p_b_win(self) -> float:
        return 1.0 - self.p_a_win

    def prob_total_games_over(self, line: float) -> float:
        return float(np.mean(self.total_games > line))

    @property
    def standard_error(self) -> float:
        return float(np.sqrt(max(self.p_a_win * (1 - self.p_a_win), 1e-12) / self.n_trials))


def simulate_match(
    p_a_serve: float,
    p_b_serve: float,
    best_of: int = 3,
    n_trials: int = DEFAULT_TRIALS,
    seed: int | None = None,
    final_set_tb_points: int = 7,
) -> SimResult:
    """Simulate `n_trials` matches point by point.

    p_a_serve: probability player A wins a point when A serves.
    p_b_serve: probability player B wins a point when B serves.
    final_set_tb_points: tiebreak target in the deciding set (10 at Grand Slams).
    Every set uses a tiebreak at 6-6. The first server is chosen by coin toss per trial.
    """
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5")
    for p in (p_a_serve, p_b_serve):
        if not 0.0 < p < 1.0:
            raise ValueError("serve probabilities must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    n = n_trials
    sets_to_win = best_of // 2 + 1

    pts_a = np.zeros(n, dtype=np.int16)
    pts_b = np.zeros(n, dtype=np.int16)
    games_a = np.zeros(n, dtype=np.int16)
    games_b = np.zeros(n, dtype=np.int16)
    sets_a = np.zeros(n, dtype=np.int16)
    sets_b = np.zeros(n, dtype=np.int16)
    total_games = np.zeros(n, dtype=np.int16)
    in_tb = np.zeros(n, dtype=bool)
    server = rng.integers(0, 2, n).astype(np.int8)        # 0 = A serves the current game
    tb_first_server = np.zeros(n, dtype=np.int8)
    first_set_a = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    while True:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        tb = in_tb[idx]
        # In a tiebreak the serve changes after the first point, then every two points.
        tb_played = pts_a[idx] + pts_b[idx]
        tb_server = np.where(((tb_played + 1) // 2) % 2 == 0, tb_first_server[idx], 1 - tb_first_server[idx])
        srv = np.where(tb, tb_server, server[idx])
        p_a_point = np.where(srv == 0, p_a_serve, 1.0 - p_b_serve)
        a_won = rng.random(idx.size) < p_a_point
        pts_a[idx] += a_won
        pts_b[idx] += ~a_won

        pa, pb = pts_a[idx], pts_b[idx]
        deciding = (sets_a[idx] + sets_b[idx]) == best_of - 1
        target = np.where(tb, np.where(deciding, final_set_tb_points, 7), 4)
        game_a = (pa >= target) & (pa - pb >= 2)
        game_b = (pb >= target) & (pb - pa >= 2)
        game_over = game_a | game_b
        if not game_over.any():
            continue

        g = idx[game_over]
        won_a = game_a[game_over]
        was_tb = in_tb[g]
        games_a[g] += won_a
        games_b[g] += ~won_a
        total_games[g] += 1
        pts_a[g] = 0
        pts_b[g] = 0
        # Next game's server. After a tiebreak, the player who received first serves.
        server[g] = np.where(was_tb, 1 - tb_first_server[g], 1 - server[g])
        in_tb[g] = False

        ga, gb = games_a[g], games_b[g]
        set_a = was_tb & won_a | ~was_tb & (ga >= 6) & (ga - gb >= 2)
        set_b = was_tb & ~won_a | ~was_tb & (gb >= 6) & (gb - ga >= 2)
        start_tb = ~was_tb & (ga == 6) & (gb == 6)
        t = g[start_tb]
        in_tb[t] = True
        tb_first_server[t] = server[t]

        set_over = set_a | set_b
        if set_over.any():
            s = g[set_over]
            s_won_a = set_a[set_over]
            first = (sets_a[s] + sets_b[s]) == 0
            first_set_a[s[first]] = s_won_a[first]
            sets_a[s] += s_won_a
            sets_b[s] += ~s_won_a
            games_a[s] = 0
            games_b[s] = 0
            done = (sets_a[s] == sets_to_win) | (sets_b[s] == sets_to_win)
            active[s[done]] = False

    a_wins = sets_a == sets_to_win
    scores, counts = np.unique(sets_a.astype(np.int32) * 10 + sets_b, return_counts=True)
    set_score_probs = {f"{code // 10}-{code % 10}": float(count / n) for code, count in zip(scores, counts)}
    return SimResult(
        p_a_win=float(a_wins.mean()),
        n_trials=n,
        best_of=best_of,
        set_score_probs=set_score_probs,
        p_a_first_set=float(first_set_a.mean()),
        mean_total_games=float(total_games.mean()),
        total_games=total_games,
    )


# --------------------------------------------------------------------------- exact solution

def game_win_prob(p: float) -> float:
    """Probability the server wins a standard game, given point-win probability p."""
    q = 1.0 - p
    deuce = p * p / (p * p + q * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


def tiebreak_win_prob(p_a_serve: float, p_b_serve: float, target: int = 7, a_serves_first: bool = True) -> float:
    """Probability A wins a tiebreak to `target` points, win by two."""
    win_on_a = p_a_serve          # P(A wins point) on A's serve
    win_on_b = 1.0 - p_b_serve    # P(A wins point) on B's serve
    # From any tied score at target-1 or later, the next two points are one on each serve.
    both = win_on_a * win_on_b
    neither = (1 - win_on_a) * (1 - win_on_b)
    tied = both / (both + neither)

    @lru_cache(maxsize=None)
    def solve(a: int, b: int) -> float:
        if a >= target - 1 and b >= target - 1 and a == b:
            return tied
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        played = a + b
        first_serving = ((played + 1) // 2) % 2 == 0
        a_serving = first_serving == a_serves_first
        p = win_on_a if a_serving else win_on_b
        return p * solve(a + 1, b) + (1 - p) * solve(a, b + 1)

    return solve(0, 0)


def _set_outcomes(p_a: float, p_b: float, a_serves_first: bool, tb_target: int) -> dict[tuple[bool, bool], float]:
    """Distribution over (A wins the set, A serves first in the next set)."""
    hold_a, hold_b = game_win_prob(p_a), game_win_prob(p_b)
    outcomes: dict[tuple[bool, bool], float] = {}
    states = {(0, 0): 1.0}
    while states:
        nxt: dict[tuple[int, int], float] = {}
        for (ga, gb), prob in states.items():
            a_serving = a_serves_first == ((ga + gb) % 2 == 0)
            if ga == 6 and gb == 6:
                p_tb = tiebreak_win_prob(p_a, p_b, tb_target, a_serving)
                next_first = not a_serving      # receiver of the tiebreak's first point serves next
                outcomes[(True, next_first)] = outcomes.get((True, next_first), 0.0) + prob * p_tb
                outcomes[(False, next_first)] = outcomes.get((False, next_first), 0.0) + prob * (1 - p_tb)
                continue
            p_game_a = hold_a if a_serving else 1.0 - hold_b
            for a_won, p_branch in ((True, p_game_a), (False, 1.0 - p_game_a)):
                na, nb = ga + a_won, gb + (not a_won)
                weight = prob * p_branch
                finished_a = na >= 6 and na - nb >= 2
                finished_b = nb >= 6 and nb - na >= 2
                if finished_a or finished_b:
                    key = (finished_a, not a_serving)
                    outcomes[key] = outcomes.get(key, 0.0) + weight
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + weight
        states = nxt
    return outcomes


@lru_cache(maxsize=200_000)
def _match_win_prob_cached(p_a: float, p_b: float, best_of: int, final_tb: int) -> float:
    sets_to_win = best_of // 2 + 1
    regular = {first: _set_outcomes(p_a, p_b, first, 7) for first in (True, False)}
    deciding = regular if final_tb == 7 else {first: _set_outcomes(p_a, p_b, first, final_tb) for first in (True, False)}

    @lru_cache(maxsize=None)
    def solve(sa: int, sb: int, a_first: bool) -> float:
        if sa == sets_to_win:
            return 1.0
        if sb == sets_to_win:
            return 0.0
        table = deciding if sa + sb == best_of - 1 else regular
        return sum(prob * solve(sa + a_won, sb + (not a_won), next_first)
                   for (a_won, next_first), prob in table[a_first].items())

    return 0.5 * (solve(0, 0, True) + solve(0, 0, False))


def match_win_prob(p_a_serve: float, p_b_serve: float, best_of: int = 3, final_set_tb_points: int = 7) -> float:
    """Exact probability that A wins the match under the same rules as `simulate_match`."""
    return _match_win_prob_cached(round(p_a_serve, 5), round(p_b_serve, 5), best_of, final_set_tb_points)


def solve_serve_probs(p_match: float, serve_total: float = 1.26, best_of: int = 3) -> tuple[float, float]:
    """Find (p_a_serve, p_b_serve) that sum to `serve_total` and give A a match win probability of `p_match`.

    The total sets how serve-dominated the match is. The split sets who is favoured.
    """
    p_match = min(max(p_match, 1e-4), 1 - 1e-4)
    mean = serve_total / 2.0
    max_gap = min(mean - MIN_SERVE_PROB, MAX_SERVE_PROB - mean)
    lo, hi = -max_gap, max_gap
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if match_win_prob(mean + mid, mean - mid, best_of) < p_match:
            lo = mid
        else:
            hi = mid
    gap = (lo + hi) / 2.0
    return mean + gap, mean - gap
