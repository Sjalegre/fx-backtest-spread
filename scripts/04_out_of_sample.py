"""The out-of-sample window, spent once.

    python scripts/04_out_of_sample.py [--instrument EURUSD]

Parameters are chosen on the training window by the same grid search
`03_validation.py` runs, then applied unchanged to the block after
`split.train_end`. Nothing here is tuned. If the result is bad, it is
published bad -- re-running this script with different parameters until it
improves would convert the held-out window into another training set, which
is the exact failure the split exists to prevent.

Produces:
  reports/figures/<instrument>_out_of_sample.png
  reports/<instrument>_out_of_sample.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, engine, metrics, strategies, validation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="EURUSD")
    args = ap.parse_args()

    cfg = data.load_config()
    name = args.instrument
    spec = cfg["instruments"][name]
    grids = cfg["validation"]["grids"]
    train_end = cfg["split"]["train_end"]

    prices, _ = data.load(name, cfg)
    train = prices.loc[:train_end]

    common = dict(
        pip_size=spec["pip_size"],
        spread_pips=spec["typical_spread_pips"],
        execution=cfg["backtest"]["execution"],
        initial_equity=cfg["backtest"]["initial_equity"],
    )

    # The test window starts the bar after train_end, but the signal is built
    # over the whole series so indicators are warm when the window opens.
    # Only the test window's returns are kept.
    test_index = prices.loc[train_end:].index[1:]
    if len(test_index) < 60:
        raise SystemExit(f"{name}: only {len(test_index)} bars after {train_end}")

    print(f"=== {name} · out-of-sample · "
          f"{test_index[0].date()} -> {test_index[-1].date()} "
          f"({len(test_index)} bars) ===")
    print(f"    parameters fixed on training data ending {train_end}\n")

    rows = {}
    curves = {}

    hold_signal = strategies.buy_and_hold(prices)
    hold = engine.run(prices, hold_signal, **common)
    rows["buy_and_hold"] = metrics.summarise(
        _restrict(hold, test_index, common["initial_equity"])
    )
    curves["buy_and_hold"] = _equity(hold, test_index, common["initial_equity"])

    for strategy, grid in grids.items():
        search = validation.grid_search(
            train, strategy, grid,
            allow_short=cfg["backtest"]["allow_short"], **common,
        )
        params = validation.coerce_params(
            {k: search.iloc[0][k] for k in grid}
        )
        in_sample_sharpe = float(search.iloc[0]["sharpe"])

        signal = validation.build_signal(
            prices, strategy, params, cfg["backtest"]["allow_short"]
        )
        result = engine.run(prices, signal, **common)
        restricted = _restrict(result, test_index, common["initial_equity"])

        summary = metrics.summarise(restricted)
        summary["in_sample_sharpe"] = in_sample_sharpe
        summary["params"] = str(params)
        rows[strategy] = summary
        curves[strategy] = _equity(result, test_index, common["initial_equity"])

        print(f"  {strategy}  {params}")
        print(f"    in-sample Sharpe      {in_sample_sharpe:+.3f}")
        print(f"    out-of-sample Sharpe  {summary['sharpe']:+.3f}")
        print(f"    out-of-sample return  {summary['total_return']:+.2%}"
              f"  (CAGR {summary['cagr']:+.2%})")
        print(f"    max drawdown          {summary['max_drawdown']:+.2%}")
        # Printed next to Sharpe on purpose. A rule that is flat 99% of the
        # time can post an excellent Sharpe on a rounding error: the ratio is
        # scale-free, and cash has no volatility. Exposure is what says
        # whether there is a business here or just a very quiet number.
        print(f"    exposure              {summary['exposure']:.1%} of bars"
              f"  (annualised vol {summary['ann_vol']:.2%})")
        print(f"    trades                {summary['n_trades']}\n")

    table = pd.DataFrame(rows).T
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / f"{name}_out_of_sample.csv")

    print(table[["total_return", "cagr", "sharpe", "ann_vol",
                 "max_drawdown", "exposure", "n_trades"]]
          .astype(float).round(4).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, curve in curves.items():
        style = dict(linewidth=1.0, color="grey", linestyle="--") \
            if label == "buy_and_hold" else dict(linewidth=1.3)
        ax.plot(curve.index, curve.values, label=label, **style)
    ax.set_title(f"{name} — out-of-sample, parameters fixed before this window opened")
    ax.set_ylabel("Equity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_out_of_sample.png", dpi=140)
    plt.close(fig)

    print(f"\n[ok] figure -> {FIGURES / f'{name}_out_of_sample.png'}")


def _equity(result, index, initial_equity: float) -> pd.Series:
    """Re-base an equity curve to the start of the out-of-sample window."""
    returns = result.returns.loc[index]
    return initial_equity * (1 + returns).cumprod()


def _restrict(result, index, initial_equity: float):
    """A BacktestResult narrowed to the out-of-sample window.

    Rebuilt rather than sliced so the equity curve starts at the initial
    stake: a curve inherited mid-flight would make drawdown and total return
    measure the training window too.
    """
    returns = result.returns.loc[index]
    return engine.BacktestResult(
        equity=initial_equity * (1 + returns).cumprod(),
        returns=returns,
        position=result.position.loc[index],
        costs=result.costs.loc[index],
        n_trades=int((result.costs.loc[index] > 0).sum()),
        spread_pips=result.spread_pips,
        execution=result.execution,
    )


if __name__ == "__main__":
    main()
