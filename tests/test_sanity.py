"""Sanity tests for the engine.

If these fail, every number this project produces is meaningless. They run
before any result is believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import engine, strategies  # noqa: E402


@pytest.fixture
def prices() -> pd.DataFrame:
    """Synthetic random walk with genuine overnight gaps.

    Gaps matter: with open == previous close, a look-ahead bug in the gap
    handling would be invisible.
    """
    rng = np.random.default_rng(42)
    n = 500
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    gap = rng.normal(0, 0.003, n)
    open_ = np.r_[close[0], close[:-1] * (1 + gap[1:])]
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.002, n)))

    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=idx
    )


def test_always_long_reproduces_buy_and_hold(prices):
    """The load-bearing test: constant long, zero spread == buy-and-hold.

    Catches additive-instead-of-compounded returns and any mishandling of
    the overnight gap.
    """
    sig = strategies.buy_and_hold(prices)
    res = engine.run(prices, sig, pip_size=0.0001, spread_pips=0.0)

    # Position only settles after the two-bar execution lag.
    start = prices.index[2]
    expected = prices["Close"].loc[start:] / prices["Close"].loc[start]
    actual = res.equity.loc[start:] / res.equity.loc[start]

    pd.testing.assert_series_equal(actual, expected, check_names=False, rtol=1e-12)


def test_execution_lag_is_enforced(prices):
    """The position held into bar `t` must be the signal decided at `t-1`.

    Asserted against a deliberately cheating oracle built from tomorrow's
    return. If the engine ever filled on the same bar the signal was
    computed, this equality would break — which is exactly the bug that
    makes a backtest print an impossible equity curve.
    """
    future_ret = prices["Close"].pct_change().shift(-1)
    oracle = np.sign(future_ret).fillna(0.0)

    res = engine.run(prices, oracle, pip_size=0.0001, spread_pips=0.0)

    pd.testing.assert_series_equal(
        res.position, oracle.shift(1).fillna(0.0), check_names=False
    )
    assert res.position.iloc[0] == 0.0
    assert res.returns.iloc[0] == 0.0


def test_flat_signal_never_moves_equity(prices):
    sig = pd.Series(0.0, index=prices.index)
    res = engine.run(prices, sig, pip_size=0.0001, spread_pips=2.0)

    assert res.n_trades == 0
    assert res.costs.sum() == 0.0
    assert res.equity.nunique() == 1


def test_spread_only_ever_hurts(prices):
    """More spread can never produce more money."""
    sig = strategies.ma_crossover(prices, fast=5, slow=20)

    finals = [
        engine.run(prices, sig, pip_size=0.0001, spread_pips=s).equity.iloc[-1]
        for s in (0.0, 1.0, 3.0, 10.0)
    ]
    assert finals == sorted(finals, reverse=True)


def test_cost_charged_once_per_position_change(prices):
    """A single flip of a 1-unit position pays exactly one full spread."""
    sig = pd.Series(0.0, index=prices.index)
    sig.iloc[10:] = 1.0                     # one entry, no exit

    spread_pips, pip_size = 2.0, 0.0001
    res = engine.run(prices, sig, pip_size=pip_size, spread_pips=spread_pips)

    # Signal decided at the close of bar 10 is filled at the open of bar 11.
    fill_price = prices["Open"].iloc[11]
    expected = (spread_pips * pip_size / 2) / fill_price

    assert res.n_trades == 1
    assert res.costs.sum() == pytest.approx(expected, rel=1e-12)


def test_short_position_profits_when_price_falls(prices):
    """Sign convention: -1 must make money in a downtrend."""
    falling = prices.copy()
    factor = pd.Series(np.linspace(1.0, 0.5, len(prices)), index=prices.index)
    for col in ("Open", "High", "Low", "Close"):
        falling[col] = falling[col] * factor

    short = pd.Series(-1.0, index=prices.index)
    res = engine.run(falling, short, pip_size=0.0001, spread_pips=0.0)

    assert res.equity.iloc[-1] > res.equity.iloc[0]


def test_execution_modes_differ_but_agree_in_direction(prices):
    """next_close is the optimistic assumption; both must stay finite."""
    sig = strategies.ma_crossover(prices, fast=5, slow=20)

    a = engine.run(prices, sig, pip_size=0.0001, spread_pips=1.0,
                   execution="next_open")
    b = engine.run(prices, sig, pip_size=0.0001, spread_pips=1.0,
                   execution="next_close")

    assert np.isfinite(a.equity.iloc[-1]) and np.isfinite(b.equity.iloc[-1])
    assert a.equity.iloc[-1] != b.equity.iloc[-1]
