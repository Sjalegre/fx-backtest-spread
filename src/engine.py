"""Backtest engine.

Design rules, in order of how much damage they prevent:

1. **No look-ahead.** A signal computed from the close of day `t` cannot be
   filled before day `t+1`. The engine enforces this by shifting positions,
   never by trusting the strategy to behave.

2. **The overnight gap belongs to the old position.** If you decide at the
   close of Monday and fill at Tuesday's open, then Monday-close→Tuesday-open
   is still held at the *previous* position. Collapsing this into a single
   close-to-close return silently hands the strategy the gap it never had.

3. **Returns compound, they do not add.** Segment returns are multiplied, so
   an always-long run reproduces buy-and-hold exactly. That is what
   `tests/test_sanity.py` checks.

4. **Crossing the spread costs money.** Every unit of position change pays
   half the spread on entry and half on exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series          # net, per bar
    position: pd.Series         # position held into each bar's close
    costs: pd.Series            # cost paid per bar, in return terms
    n_trades: int
    spread_pips: float
    execution: str

    @property
    def trade_dates(self) -> pd.DatetimeIndex:
        return self.costs.index[self.costs > 0]


def run(
    prices: pd.DataFrame,
    signal: pd.Series,
    *,
    pip_size: float,
    spread_pips: float = 0.0,
    execution: str = "next_open",
    initial_equity: float = 10_000.0,
) -> BacktestResult:
    """Backtest one signal series over one instrument.

    `signal` holds the target position (-1, 0 or +1) decided at each bar's
    close. It is shifted internally — do not pre-shift it.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution mode: {execution}")

    px = prices.loc[signal.index]
    half_spread = spread_pips * pip_size / 2.0

    if execution == "next_open":
        # Filled at the open of the following bar.
        pos_gap = signal.shift(2).fillna(0.0)      # held across the overnight gap
        pos_intraday = signal.shift(1).fillna(0.0)  # held from that open to close

        gap_ret = px["Open"] / px["Close"].shift(1) - 1.0
        intraday_ret = px["Close"] / px["Open"] - 1.0

        gross = (1 + pos_gap * gap_ret.fillna(0.0)) * (1 + pos_intraday * intraday_ret)
        traded = (pos_intraday - pos_gap).abs()
        cost = traded * half_spread / px["Open"]
        position = pos_intraday
    else:
        # Filled at the close of the bar on which the signal was computed.
        position = signal.shift(1).fillna(0.0)
        cc_ret = px["Close"].pct_change().fillna(0.0)

        gross = 1 + position * cc_ret
        traded = position.diff().abs().fillna(position.abs())
        cost = traded * half_spread / px["Close"].shift(1).bfill()

    net = gross * (1 - cost)
    equity = initial_equity * net.cumprod()

    return BacktestResult(
        equity=equity,
        returns=net - 1.0,
        position=position,
        costs=cost,
        n_trades=int((traded > 0).sum()),
        spread_pips=spread_pips,
        execution=execution,
    )


def sweep_spread(
    prices: pd.DataFrame,
    signal: pd.Series,
    *,
    pip_size: float,
    spreads: list[float],
    **kwargs,
) -> pd.DataFrame:
    """Re-run the same strategy across spread levels.

    This is the headline analysis: an edge that only exists at zero cost is
    not an edge, and this is where that shows up.
    """
    from .metrics import summarise

    rows = []
    for s in spreads:
        res = run(prices, signal, pip_size=pip_size, spread_pips=s, **kwargs)
        row = summarise(res)
        row["spread_pips"] = s
        rows.append(row)
    return pd.DataFrame(rows).set_index("spread_pips")
