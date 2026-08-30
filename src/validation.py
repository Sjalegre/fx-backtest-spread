"""Validation: parameter search, walk-forward, and the multiple-testing bill.

A backtest that reports the best parameter set it found is reporting the
maximum of a sample, not an expectation. Search hard enough over a random
series and a Sharpe of 1.5 appears on its own. This module exists to put a
number on that, and to measure what survives when parameters are chosen
without seeing the data they are then judged on.

Three things happen here, in increasing order of how uncomfortable the answer
usually is:

1. **Grid search** over a parameter space, on training data only. Produces the
   sensitivity surface: whether the good result is a plateau you could have
   landed on by accident, or a spike you had to go looking for.
2. **Walk-forward**: re-optimise on a rolling window, trade the next window
   forward, never letting a parameter see the data it is evaluated on. The
   concatenated out-of-sample curve is the honest one.
3. **Deflated Sharpe**: given how many parameter sets were tried, how good
   would the best one look if none of them had any edge at all?
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine, metrics, strategies

# Euler-Mascheroni constant, used by the expected-maximum-Sharpe formula.
_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Parameter search
# ---------------------------------------------------------------------------

def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a parameter grid, as a list of kwargs dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def valid_params(strategy: str, params: dict) -> bool:
    """Reject combinations the strategy itself would refuse.

    Filtering here rather than catching the exception keeps the trial count
    honest: a combination that was never evaluable was never a trial, and
    counting it would understate the multiple-testing penalty by inflating N
    with combinations that could not have won.
    """
    if strategy == "ma_crossover":
        return params["fast"] < params["slow"]
    if strategy == "rsi_reversion":
        return params["oversold"] < params["overbought"]
    return True


def coerce_params(params: dict) -> dict:
    """Return parameters as native Python numbers.

    Parameters read back out of a results frame arrive as numpy scalars, and
    `Series.rolling(np.float64(50.0))` raises: a window must be a Python int.
    Coercing here rather than at each call site means every consumer -- the
    walk-forward loop, the report, a notebook -- gets the same clean values,
    and the chosen parameters print as `50` instead of `np.float64(50.0)`.
    """
    out = {}
    for key, value in params.items():
        number = float(value)
        out[key] = int(number) if number.is_integer() else number
    return out


def build_signal(prices: pd.DataFrame, strategy: str, params: dict,
                 allow_short: bool = True) -> pd.Series:
    """Instantiate one strategy's signal for one parameter set."""
    fn = strategies.REGISTRY[strategy]
    if strategy == "buy_and_hold":
        return fn(prices)
    return fn(prices, allow_short=allow_short, **coerce_params(params))


def grid_search(
    prices: pd.DataFrame,
    strategy: str,
    grid: dict[str, list],
    *,
    pip_size: float,
    spread_pips: float,
    objective: str = "sharpe",
    allow_short: bool = True,
    execution: str = "next_open",
    initial_equity: float = 10_000.0,
) -> pd.DataFrame:
    """Evaluate every valid parameter combination on `prices`.

    Returns one row per combination with the full metric set, sorted by
    `objective` descending. The whole frame is returned, not just the winner:
    the distribution of results is the interesting part, and it is what the
    deflated Sharpe needs.
    """
    rows = []
    for params in expand_grid(grid):
        if not valid_params(strategy, params):
            continue
        signal = build_signal(prices, strategy, params, allow_short)
        result = engine.run(
            prices, signal,
            pip_size=pip_size, spread_pips=spread_pips,
            execution=execution, initial_equity=initial_equity,
        )
        row = metrics.summarise(result)
        row.update(params)
        rows.append(row)

    if not rows:
        raise ValueError(f"{strategy}: no valid parameter combinations in grid")

    frame = pd.DataFrame(rows)
    return frame.sort_values(objective, ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Out-of-sample performance from rolling re-optimisation."""

    returns: pd.Series            # concatenated out-of-sample net returns
    equity: pd.Series
    folds: pd.DataFrame           # one row per fold: window, chosen params, both scores
    strategy: str
    n_trials_per_fold: int

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def parameter_stability(self) -> float:
        """Share of folds that re-selected the previous fold's parameters.

        A strategy whose optimal parameters change every fold has not found a
        property of the market; it has fitted the window. This number is the
        cheapest available warning of that.
        """
        if len(self.folds) < 2:
            return float("nan")
        cols = [c for c in self.folds.columns if c.startswith("param_")]
        if not cols:
            return float("nan")
        chosen = self.folds[cols]
        same = (chosen.iloc[1:].values == chosen.iloc[:-1].values).all(axis=1)
        return float(same.mean())


def walk_forward(
    prices: pd.DataFrame,
    strategy: str,
    grid: dict[str, list],
    *,
    pip_size: float,
    spread_pips: float,
    train_days: int = 756,
    test_days: int = 252,
    objective: str = "sharpe",
    allow_short: bool = True,
    execution: str = "next_open",
    initial_equity: float = 10_000.0,
) -> WalkForwardResult:
    """Roll a train/test window forward, re-optimising on each training block.

    Windows are counted in **bars**, not calendar days, so a fold always holds
    the same amount of information regardless of holidays.

    Each fold's parameters are chosen on its training block and then applied,
    unchanged, to the block that follows. No parameter ever sees the returns it
    is scored on. The signal is rebuilt on train+test together so that
    indicators needing a warm-up period do not start the test window blind,
    but only the test window's returns are kept.
    """
    n = len(prices)
    if train_days + test_days > n:
        raise ValueError(
            f"need at least {train_days + test_days} bars, got {n}"
        )

    combos = [p for p in expand_grid(grid) if valid_params(strategy, p)]
    common = dict(
        pip_size=pip_size, spread_pips=spread_pips,
        execution=execution, initial_equity=initial_equity,
    )

    oos_returns: list[pd.Series] = []
    fold_rows: list[dict] = []

    start = 0
    while start + train_days + test_days <= n:
        train = prices.iloc[start: start + train_days]
        # The signal is computed over the warm-up plus the test window so that
        # a 100-bar moving average is not undefined for the first 100 bars of
        # every fold. Returns are still taken from the test window alone.
        test_slice = prices.iloc[start: start + train_days + test_days]
        test_index = prices.index[start + train_days: start + train_days + test_days]

        scores = grid_search(
            train, strategy, grid,
            objective=objective, allow_short=allow_short, **common,
        )
        best = scores.iloc[0]
        params = coerce_params({k: best[k] for k in combos[0]}) if combos[0] else {}

        signal = build_signal(test_slice, strategy, params, allow_short)
        result = engine.run(test_slice, signal, **common)
        fold_returns = result.returns.loc[test_index]
        oos_returns.append(fold_returns)

        row = {
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test_index[0],
            "test_end": test_index[-1],
            "in_sample_sharpe": float(best[objective]),
            "out_of_sample_sharpe": metrics.sharpe(fold_returns),
            "out_of_sample_return": float((1 + fold_returns).prod() - 1),
            "n_trials": len(scores),
        }
        row.update({f"param_{k}": v for k, v in params.items()})
        fold_rows.append(row)

        start += test_days

    if not oos_returns:
        raise ValueError("no complete folds; shorten train_days or test_days")

    stitched = pd.concat(oos_returns)
    equity = initial_equity * (1 + stitched).cumprod()

    return WalkForwardResult(
        returns=stitched,
        equity=equity,
        folds=pd.DataFrame(fold_rows),
        strategy=strategy,
        n_trials_per_fold=len(combos),
    )


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, without pulling in SciPy.

    Acklam's rational approximation; accurate to about 1.15e-9 across the open
    interval, which is far beyond what is needed to report a Sharpe threshold.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > p_high:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected best Sharpe from `n_trials` strategies that all have no edge.

    This is the number that makes a backtest honest. Search a hundred
    parameter sets over a random walk and the best of them will show a
    positive Sharpe; this says how positive, so a result can be compared
    against what luck alone would have produced.

    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio". All Sharpes
    here are per-observation, not annualised.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    if trial_sharpe_std <= 0 or not np.isfinite(trial_sharpe_std):
        return 0.0

    return trial_sharpe_std * (
        (1 - _EULER_GAMMA) * _norm_ppf(1 - 1 / n_trials)
        + _EULER_GAMMA * _norm_ppf(1 - 1 / (n_trials * np.e))
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpe_std: float,
) -> dict:
    """Probability the observed Sharpe is real, given how hard we searched.

    Returns a dict rather than a bare float because the intermediate
    quantities are the argument: the benchmark the Sharpe had to beat, and the
    skew and kurtosis that decide how much a given Sharpe is worth.

    A DSR of 0.95 means: after accounting for `n_trials` attempts and for the
    shape of this return distribution, there is a 95% chance the true Sharpe
    is above zero. Below 0.95 the usual reading is that the result is not
    distinguishable from the best of a lucky search.
    """
    clean = returns.dropna()
    n_obs = len(clean)
    if n_obs < 3:
        raise ValueError("need at least 3 observations")

    sd = clean.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return {
            "sharpe_per_obs": float("nan"), "sharpe_annualised": float("nan"),
            "benchmark_sharpe": float("nan"), "deflated_sharpe_ratio": float("nan"),
            "n_trials": n_trials, "n_obs": n_obs,
            "skew": float("nan"), "kurtosis": float("nan"),
        }

    sr = float(clean.mean() / sd)
    skew = float(clean.skew())
    kurt = float(clean.kurtosis()) + 3.0   # pandas reports excess kurtosis

    sr_star = expected_max_sharpe(n_trials, trial_sharpe_std)

    # Variance of the Sharpe estimator under non-normal returns.
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0 or not np.isfinite(denom):
        dsr = float("nan")
    else:
        dsr = _norm_cdf((sr - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(denom))

    return {
        "sharpe_per_obs": sr,
        "sharpe_annualised": sr * np.sqrt(metrics.TRADING_DAYS),
        "benchmark_sharpe": sr_star,
        "benchmark_sharpe_annualised": sr_star * np.sqrt(metrics.TRADING_DAYS),
        "deflated_sharpe_ratio": dsr,
        "n_trials": n_trials,
        "n_obs": n_obs,
        "skew": skew,
        "kurtosis": kurt,
    }


def multiple_testing_report(
    search: pd.DataFrame,
    returns: pd.Series,
    objective: str = "sharpe",
) -> dict:
    """Deflate the best result of a grid search by the size of that search.

    `search` is the full frame from `grid_search` -- every combination, not
    just the winner. The spread of Sharpes across the grid is what estimates
    how much variation the search itself could generate.
    """
    trial_sharpes = search[objective].dropna()
    n_trials = int(len(trial_sharpes))
    # Convert the annualised Sharpes the metrics module reports back to
    # per-observation, which is the scale the deflation formula works on.
    per_obs_std = float(trial_sharpes.std(ddof=1) / np.sqrt(metrics.TRADING_DAYS))

    report = deflated_sharpe_ratio(returns, n_trials, per_obs_std)
    report["trial_sharpe_std_annualised"] = float(trial_sharpes.std(ddof=1))
    report["best_trial_sharpe_annualised"] = float(trial_sharpes.max())
    return report
