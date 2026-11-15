import numpy as np
import pandas as pd
import pytest

from src import backtest
from src.model import ModelParams, RatingModel, bo3_to_bo5, fit_betas


def synthetic_matches(n=1500, n_players=40, seed=0, season_split=True):
    """Random matches where a hidden skill decides the winner."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1.0, n_players)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(n):
        a, b = rng.choice(n_players, 2, replace=False)
        p_a = 1 / (1 + np.exp(-(skill[a] - skill[b])))
        winner, loser = (a, b) if rng.random() < p_a else (b, a)
        date = start + pd.Timedelta(days=int(i * 700 / n))
        rows.append({
            "match_id": i + 1, "tour": "ATP", "season": date.year if season_split else 2024, "tier": "tour",
            "tourney_level": "A", "tourney_date": date.strftime("%Y-%m-%d"), "surface": "Hard", "best_of": 3,
            "outcome": "COMPLETED", "winner_id": int(winner) + 1, "loser_id": int(loser) + 1,
            "winner_rank_points": 1000.0, "loser_rank_points": 1000.0,
            "w_svpt": 70.0, "w_1stWon": 32.0, "w_2ndWon": 14.0, "l_svpt": 72.0, "l_1stWon": 28.0, "l_2ndWon": 13.0,
        })
    return pd.DataFrame(rows)


def test_metrics():
    assert backtest.brier_score([0.5, 0.5]) == pytest.approx(0.25)
    assert backtest.brier_score([1.0, 0.0]) == pytest.approx(0.5)
    assert backtest.brier_score([0.8]) == pytest.approx(0.04)
    assert backtest.accuracy([0.9, 0.2, 0.6, 0.4]) == pytest.approx(0.5)
    assert backtest.log_loss([0.5]) == pytest.approx(np.log(2))


def test_walk_forward_learns_and_beats_coin_flip():
    preds, _ = backtest.walk_forward(synthetic_matches(), ModelParams())
    late = preds[preds["match_id"] > 750]
    assert backtest.brier_score(late["p_winner"]) < 0.23
    assert preds["p_winner"].between(0.005, 0.995).all()


def test_no_lookahead():
    """Changing the future must not change any earlier prediction."""
    matches = synthetic_matches()
    cut = 900
    altered = matches.copy()
    swap = altered["match_id"] > cut
    altered.loc[swap, ["winner_id", "loser_id"]] = altered.loc[swap, ["loser_id", "winner_id"]].to_numpy()
    altered.loc[swap, "w_svpt"] = 99.0
    base, _ = backtest.walk_forward(matches, ModelParams())
    changed, _ = backtest.walk_forward(altered, ModelParams())
    np.testing.assert_array_equal(base.loc[base["match_id"] <= cut, "p_winner"].to_numpy(),
                                  changed.loc[changed["match_id"] <= cut, "p_winner"].to_numpy())
    # The match's own result must not leak into its own prediction either.
    assert base.loc[base["match_id"] == cut + 1, "p_winner"].item() == pytest.approx(
        1 - changed.loc[changed["match_id"] == cut + 1, "p_winner"].item())


def test_walkovers_are_not_scored_and_do_not_move_ratings():
    matches = synthetic_matches(n=50)
    matches.loc[10, "outcome"] = "W/O"
    preds, model = backtest.walk_forward(matches, ModelParams())
    assert len(preds) == 49 and 11 not in set(preds["match_id"])
    fresh = RatingModel()
    fresh.update(1, 2, "Hard", 3, "W/O", "2024-01-01", 1000.0, 1000.0)
    assert fresh.players[1].elo == fresh.players[2].elo and fresh.players[1].n == 0
    assert fresh.players[1].last_outcome == fresh.players[2].last_outcome == "W/O"
    assert fresh.players[1].last_won is True and fresh.players[2].last_won is False


def test_calibration_gap():
    rng = np.random.default_rng(3)
    p_fav = rng.uniform(0.5, 1.0, 40_000)
    fav_won = rng.random(40_000) < p_fav
    calibrated = np.where(fav_won, p_fav, 1 - p_fav)
    assert backtest.max_calibration_gap(calibrated) < 0.02
    assert len(backtest.calibration_table(calibrated)) == 5
    overconfident = np.where(rng.random(40_000) < p_fav - 0.08, p_fav, 1 - p_fav)
    assert backtest.max_calibration_gap(overconfident) > 0.05


def test_model_is_symmetric():
    _, model = backtest.walk_forward(synthetic_matches(n=300), ModelParams())
    x_ab, total_ab = model.features(1, 2, "Hard", 800.0, 1200.0)
    x_ba, total_ba = model.features(2, 1, "Hard", 1200.0, 800.0)
    np.testing.assert_allclose(x_ab, -x_ba)
    assert total_ab == pytest.approx(total_ba)


def test_sim_engine_agrees_with_exact_chain():
    matches = synthetic_matches(n=400)
    exact, _ = backtest.walk_forward(matches, ModelParams(), engine="analytic")
    sampled, _ = backtest.walk_forward(matches, ModelParams(), engine="sim", sim_seasons=(2025,), n_trials=4000)
    used = sampled["engine"] == "sim"
    assert used.sum() > 50 and (sampled.loc[~used, "season"] == 2024).all()
    diff = (sampled.loc[used, "p_winner"] - exact.loc[used, "p_winner"]).abs()
    assert diff.max() < 0.04 and diff.mean() < 0.012


def test_tuning_refuses_test_seasons():
    with pytest.raises(ValueError):
        backtest.tune_params(synthetic_matches(n=50))          # contains 2025 rows
    assert backtest.TUNE_SEASON == 2024 and backtest.TEST_SEASONS == (2025, 2026)


def test_bo5_conversion_and_beta_fit():
    assert bo3_to_bo5(0.5) == pytest.approx(0.5, abs=1e-6)
    assert bo3_to_bo5(0.7) > 0.7 and bo3_to_bo5(0.3) < 0.3
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 3))
    true = np.array([1.5, 0.4, -0.2])
    y = (rng.random(5000) < 1 / (1 + np.exp(-(x @ true)))).astype(float)
    np.testing.assert_allclose(fit_betas(x, y, ridge=0.0), true, atol=0.12)
