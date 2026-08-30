"""Performance metrics.

Deliberately includes the unflattering ones. A report with CAGR and no
drawdown, or hit rate with no profit factor, is a sales pitch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe, risk-free assumed 0 unless given."""
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std(ddof=1)
    if downside == 0 or np.isnan(downside):
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity) / mdd) if mdd > 0 else float("nan")


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    return float((active > 0).mean()) if len(active) else float("nan")


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def exposure(position: pd.Series) -> float:
    """Share of bars with a position open."""
    return float((position != 0).mean())


def summarise(result) -> dict:
    """One row of results for a BacktestResult."""
    eq, ret = result.equity, result.returns
    return {
        "total_return": total_return(eq),
        "cagr": cagr(eq),
        "ann_vol": annual_volatility(ret),
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(eq),
        "hit_rate": hit_rate(ret),
        "profit_factor": profit_factor(ret),
        "exposure": exposure(result.position),
        "n_trades": result.n_trades,
        "total_cost": float(result.costs.sum()),
    }


def table(results: dict) -> pd.DataFrame:
    """`{label: BacktestResult}` -> comparison table."""
    return pd.DataFrame({k: summarise(v) for k, v in results.items()}).T
