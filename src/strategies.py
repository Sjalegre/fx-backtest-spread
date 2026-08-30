"""Signal generators.

Every function returns a Series of target positions (-1, 0, +1) indexed like
the price frame, computed **only from information available at that bar's
close**. The engine handles the execution lag; strategies must never shift
their own output.

Two strategies with opposite logic are included on purpose. Reporting only
the one that worked is the most common way a backtest ends up telling you
what you wanted to hear.
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    """Benchmark. Always long. Used by the sanity test."""
    return pd.Series(1.0, index=prices.index, name="buy_and_hold")


def ma_crossover(
    prices: pd.DataFrame, fast: int = 20, slow: int = 100, allow_short: bool = True
) -> pd.Series:
    """Trend following: long while the fast average leads the slow one."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")

    close = prices["Close"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()

    sig = pd.Series(0.0, index=close.index)
    sig[ma_fast > ma_slow] = 1.0
    sig[ma_fast < ma_slow] = -1.0 if allow_short else 0.0
    sig[ma_slow.isna()] = 0.0        # no position until both windows are full
    return sig.rename(f"ma_{fast}_{slow}")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, smoothed with an exponential mean."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - 100 / (1 + rs)).astype(float)


def rsi_reversion(
    prices: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    allow_short: bool = True,
) -> pd.Series:
    """Mean reversion: buy oversold, sell overbought, flat in between."""
    r = rsi(prices["Close"], period)

    sig = pd.Series(0.0, index=prices.index)
    sig[r < oversold] = 1.0
    sig[r > overbought] = -1.0 if allow_short else 0.0
    sig[r.isna()] = 0.0
    return sig.rename(f"rsi_{period}_{int(oversold)}_{int(overbought)}")


REGISTRY = {
    "buy_and_hold": buy_and_hold,
    "ma_crossover": ma_crossover,
    "rsi_reversion": rsi_reversion,
}
