"""Tests for parameter search, walk-forward and the multiple-testing maths.

The walk-forward tests matter more than they look. Their job is to prove that
no parameter ever sees the returns it is scored on -- which is the one bug that
would make every out-of-sample number in this repository a lie, and which is
invisible in the output because a leaked fit simply looks like a good result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics, validation  # noqa: E402


@pytest.fixture
def prices() -> pd.DataFrame:
    """Synthetic random walk, long enough for several walk-forward folds."""
    rng = np.random.default_rng(7)
    n = 1500
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    gap = rng.normal(0, 0.003, n)
    open_ = np.r_[close[0], close[:-1] * (1 + gap[1:])]
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.002, n)))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=idx
    )


GRID = {"fast": [5, 10, 20], "slow": [50, 100]}


class TestGrid:
    def test_expands_the_cartesian_product(self):
        assert len(validation.expand_grid(GRID)) == 6

    def test_rejects_combinations_the_strategy_would_refuse(self):
        assert validation.valid_params("ma_crossover", {"fast": 5, "slow": 50})
        assert not validation.valid_params("ma_crossover", {"fast": 50, "slow": 20})
        assert not validation.valid_params(
            "rsi_reversion", {"oversold": 70, "overbought": 30}
        )

    def test_invalid_combinations_do_not_inflate_the_trial_count(self, prices):
        # The multiple-testing penalty is driven by how many strategies were
        # actually evaluated. Counting combinations that could never have run
        # would make the search look wider than it was and, perversely, make
        # the deflated Sharpe more forgiving.
        grid = {"fast": [10, 50, 100], "slow": [20, 50, 100]}
        # Of the 9 combinations, only fast < slow is evaluable:
        # (10,20) (10,50) (10,100) (50,100).
        search = validation.grid_search(
            prices, "ma_crossover", grid, pip_size=0.0001, spread_pips=1.0
        )
        assert len(search) == 4
        assert len(validation.expand_grid(grid)) == 9
        assert (search["fast"] < search["slow"]).all()

    def test_returns_every_trial_not_just_the_winner(self, prices):
        search = validation.grid_search(
            prices, "ma_crossover", GRID, pip_size=0.0001, spread_pips=1.0
        )
        assert len(search) == 6
        assert search["sharpe"].is_monotonic_decreasing

    def test_params_come_back_as_native_python_numbers(self):
        coerced = validation.coerce_params(
            {"fast": np.float64(20.0), "threshold": np.float64(2.5)}
        )
        assert coerced == {"fast": 20, "threshold": 2.5}
        assert isinstance(coerced["fast"], int)


class TestWalkForward:
    def test_produces_the_expected_number_of_folds(self, prices):
        wf = validation.walk_forward(
            prices, "ma_crossover", GRID,
            pip_size=0.0001, spread_pips=1.0,
            train_days=500, test_days=250,
        )
        # (1500 - 500) // 250 == 4 complete folds
        assert wf.n_folds == 4

    def test_test_windows_are_contiguous_and_do_not_overlap(self, prices):
        wf = validation.walk_forward(
            prices, "ma_crossover", GRID,
            pip_size=0.0001, spread_pips=1.0,
            train_days=500, test_days=250,
        )
        starts = wf.folds["test_start"].tolist()
        ends = wf.folds["test_end"].tolist()
        for previous_end, next_start in zip(ends, starts[1:]):
            assert next_start > previous_end

    def test_training_block_always_precedes_its_test_block(self, prices):
        """The property that makes walk-forward meaningful.

        If a training window ever reached into or past its own test window,
        the reported out-of-sample Sharpe would be an in-sample Sharpe wearing
        a different label.
        """
        wf = validation.walk_forward(
            prices, "ma_crossover", GRID,
            pip_size=0.0001, spread_pips=1.0,
            train_days=500, test_days=250,
        )
        assert (wf.folds["train_end"] < wf.folds["test_start"]).all()

    def test_stitched_returns_cover_every_test_window_exactly_once(self, prices):
        wf = validation.walk_forward(
            prices, "ma_crossover", GRID,
            pip_size=0.0001, spread_pips=1.0,
            train_days=500, test_days=250,
        )
        assert len(wf.returns) == wf.n_folds * 250
        assert wf.returns.index.is_monotonic_increasing
        assert not wf.returns.index.has_duplicates

    def test_equity_is_compounded_from_the_stitched_returns(self, prices):
        wf = validation.walk_forward(
            prices, "ma_crossover", GRID,
            pip_size=0.0001, spread_pips=1.0,
            train_days=500, test_days=250, initial_equity=10_000.0,
        )
        expected = 10_000.0 * (1 + wf.returns).prod()
        assert wf.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_refuses_a_window_longer_than_the_data(self, prices):
        with pytest.raises(ValueError):
            validation.walk_forward(
                prices, "ma_crossover", GRID,
                pip_size=0.0001, spread_pips=1.0,
                train_days=1400, test_days=300,
            )

    def test_parameter_stability_is_reported(self, prices):
        wf = validation.walk_forward(
            prices, "ma_crossover", GRID,
            pip_size=0.0001, spread_pips=1.0,
            train_days=500, test_days=250,
        )
        assert 0.0 <= wf.parameter_stability <= 1.0


class TestExpectedMaxSharpe:
    def test_a_single_trial_carries_no_penalty(self):
        assert validation.expected_max_sharpe(1, 0.05) == 0.0

    def test_the_penalty_grows_with_the_number_of_trials(self):
        values = [validation.expected_max_sharpe(n, 0.05)
                  for n in (2, 10, 100, 1000, 10_000)]
        assert values == sorted(values)

    def test_the_penalty_scales_with_the_spread_of_trial_results(self):
        # A grid whose results barely differ cannot manufacture a good Sharpe
        # by chance; a grid with wildly varying results can.
        narrow = validation.expected_max_sharpe(100, 0.01)
        wide = validation.expected_max_sharpe(100, 0.10)
        assert wide > narrow * 5

    def test_no_variation_means_no_penalty(self):
        assert validation.expected_max_sharpe(100, 0.0) == 0.0


class TestNormalHelpers:
    @pytest.mark.parametrize(
        "p, expected",
        [(0.025, -1.959964), (0.5, 0.0), (0.975, 1.959964), (0.995, 2.575829)],
    )
    def test_inverse_cdf_matches_published_quantiles(self, p, expected):
        assert validation._norm_ppf(p) == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize(
        "x, expected",
        [(-3, 0.001350), (-1, 0.158655), (0, 0.5), (1, 0.841345), (2, 0.977250)],
    )
    def test_cdf_matches_published_values(self, x, expected):
        assert validation._norm_cdf(x) == pytest.approx(expected, abs=1e-6)


class TestDeflatedSharpe:
    def _returns(self, mean, n=1500, seed=3):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mean, 0.01, n))

    def test_a_genuine_edge_found_in_one_try_survives(self):
        report = validation.deflated_sharpe_ratio(self._returns(0.001), 1, 0.0)
        assert report["deflated_sharpe_ratio"] > 0.95

    def test_the_same_edge_may_not_survive_a_wide_search(self):
        # Identical returns, only the search width changes. This is the whole
        # argument: the result did not get worse, the evidence for it did.
        returns = self._returns(0.001)
        alone = validation.deflated_sharpe_ratio(returns, 1, 0.0)
        searched = validation.deflated_sharpe_ratio(returns, 500, 0.06)
        assert searched["deflated_sharpe_ratio"] < alone["deflated_sharpe_ratio"]

    def test_noise_does_not_survive(self):
        report = validation.deflated_sharpe_ratio(self._returns(0.0), 200, 0.05)
        assert report["deflated_sharpe_ratio"] < 0.5

    def test_reports_the_benchmark_it_had_to_beat(self):
        report = validation.deflated_sharpe_ratio(self._returns(0.001), 200, 0.05)
        assert report["benchmark_sharpe"] > 0
        assert report["benchmark_sharpe_annualised"] == pytest.approx(
            report["benchmark_sharpe"] * np.sqrt(metrics.TRADING_DAYS)
        )
        assert report["n_trials"] == 200

    def test_refuses_a_sample_too_short_to_mean_anything(self):
        with pytest.raises(ValueError):
            validation.deflated_sharpe_ratio(pd.Series([0.01, -0.01]), 1, 0.0)

    def test_multiple_testing_report_counts_the_grid_it_was_given(self, prices):
        search = validation.grid_search(
            prices, "ma_crossover", GRID, pip_size=0.0001, spread_pips=1.0
        )
        returns = pd.Series(
            np.random.default_rng(1).normal(0.0002, 0.01, 1000)
        )
        report = validation.multiple_testing_report(search, returns)
        assert report["n_trials"] == len(search)
        assert np.isfinite(report["trial_sharpe_std_annualised"])
