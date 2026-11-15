import math

import pytest

from src import sim


def test_game_win_prob_known_values():
    assert sim.game_win_prob(0.5) == pytest.approx(0.5)
    assert sim.game_win_prob(0.6) == pytest.approx(0.7357, abs=1e-4)
    assert sim.game_win_prob(0.99) > 0.9999


def test_tiebreak_is_symmetric_and_fair():
    assert sim.tiebreak_win_prob(0.63, 0.63) == pytest.approx(0.5)
    a = sim.tiebreak_win_prob(0.68, 0.60, a_serves_first=True)
    b = sim.tiebreak_win_prob(0.60, 0.68, a_serves_first=False)
    assert a + b == pytest.approx(1.0)
    assert a > 0.5


@pytest.mark.parametrize("best_of", [3, 5])
def test_match_prob_symmetry(best_of):
    assert sim.match_win_prob(0.64, 0.64, best_of) == pytest.approx(0.5)
    p = sim.match_win_prob(0.67, 0.61, best_of)
    assert p + sim.match_win_prob(0.61, 0.67, best_of) == pytest.approx(1.0)


def test_best_of_five_favours_the_better_player_more():
    assert sim.match_win_prob(0.66, 0.62, 5) > sim.match_win_prob(0.66, 0.62, 3) > 0.5


@pytest.mark.parametrize("p_a,p_b,best_of,final_tb", [
    (0.65, 0.62, 3, 7), (0.60, 0.66, 3, 7), (0.65, 0.62, 5, 10), (0.70, 0.55, 5, 7), (0.63, 0.63, 3, 7),
])
def test_simulator_matches_exact_chain(p_a, p_b, best_of, final_tb):
    n = 40_000
    result = sim.simulate_match(p_a, p_b, best_of, n_trials=n, seed=7, final_set_tb_points=final_tb)
    exact = sim.match_win_prob(p_a, p_b, best_of, final_tb)
    tolerance = 4 * math.sqrt(exact * (1 - exact) / n) + 1e-4
    assert abs(result.p_a_win - exact) < tolerance


def test_simulator_defaults_and_outputs():
    result = sim.simulate_match(0.66, 0.61, best_of=3, seed=1)
    assert result.n_trials == 10_000 == sim.DEFAULT_TRIALS
    assert set(result.set_score_probs) == {"2-0", "2-1", "1-2", "0-2"}
    assert sum(result.set_score_probs.values()) == pytest.approx(1.0)
    wins = result.set_score_probs["2-0"] + result.set_score_probs["2-1"]
    assert wins == pytest.approx(result.p_a_win)
    assert 12 <= result.total_games.min() and result.total_games.max() <= 39
    assert result.prob_total_games_over(20.5) + (result.total_games <= 20).mean() == pytest.approx(1.0)
    assert 0 < result.standard_error < 0.006


def test_simulator_is_reproducible_with_a_seed():
    a = sim.simulate_match(0.64, 0.62, seed=42)
    b = sim.simulate_match(0.64, 0.62, seed=42)
    assert a.p_a_win == b.p_a_win and a.set_score_probs == b.set_score_probs


def test_simulator_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sim.simulate_match(0.6, 0.6, best_of=4)
    with pytest.raises(ValueError):
        sim.simulate_match(1.0, 0.6)


@pytest.mark.parametrize("target", [0.2, 0.5, 0.73, 0.95])
def test_solve_serve_probs_hits_target(target):
    p_a, p_b = sim.solve_serve_probs(target, serve_total=1.28, best_of=3)
    assert p_a + p_b == pytest.approx(1.28)
    assert sim.match_win_prob(p_a, p_b, 3) == pytest.approx(target, abs=1e-3)
