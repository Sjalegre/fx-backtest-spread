"""Validation run: sensitivity, walk-forward, and the multiple-testing bill.

    python scripts/03_validation.py [--instrument EURUSD]

Three questions, in order:

  1. Is the best parameter set a plateau or a spike? (sensitivity heatmap)
  2. What survives when parameters never see the data they are judged on?
     (walk-forward)
  3. Given how many parameter sets were tried, how good would the best one
     look if none of them had any edge? (deflated Sharpe)

The out-of-sample window is still not touched here. Walk-forward is
out-of-sample *within* the training period; the held-out block after
`split.train_end` is spent once, by `04_out_of_sample.py`.

Produces:
  reports/figures/<instrument>_sensitivity.png
  reports/figures/<instrument>_walk_forward.png
  reports/<instrument>_walk_forward_folds.csv
  reports/<instrument>_multiple_testing.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, engine, metrics, validation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Which two parameters to draw the sensitivity surface over. RSI has three, so
# the third is held at whichever value scored best and the caption says so.
HEATMAP_AXES = {
    "ma_crossover": ("fast", "slow"),
    "rsi_reversion": ("oversold", "overbought"),
}


def sensitivity_heatmap(ax, search: pd.DataFrame, strategy: str) -> str:
    """Draw Sharpe across the parameter grid. Returns a caption note."""
    x_name, y_name = HEATMAP_AXES[strategy]
    frame = search
    note = ""

    held = [c for c in ("period",) if c in search.columns and c not in (x_name, y_name)]
    for col in held:
        best_value = search.iloc[0][col]
        frame = frame[frame[col] == best_value]
        note = f" ({col} = {int(best_value)}, the best-scoring value)"

    pivot = frame.pivot_table(index=y_name, columns=x_name, values="sharpe")
    # Symmetric colour scale about zero: a red/blue split at 0 is the whole
    # point, and letting matplotlib centre it on the data hides the sign.
    limit = float(np.nanmax(np.abs(pivot.values))) or 1.0

    mesh = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-limit, vmax=limit,
                     aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if abs(value) > limit * 0.55 else "black")
    return note, mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="EURUSD")
    args = ap.parse_args()

    cfg = data.load_config()
    name = args.instrument
    spec = cfg["instruments"][name]
    vcfg = cfg["validation"]
    grids = vcfg["grids"]

    prices, warnings = data.load(name, cfg)
    train = prices.loc[: cfg["split"]["train_end"]]
    for w in warnings:
        print(f"    [!] {w}")

    common = dict(
        pip_size=spec["pip_size"],
        spread_pips=spec["typical_spread_pips"],
        execution=cfg["backtest"]["execution"],
        initial_equity=cfg["backtest"]["initial_equity"],
        allow_short=cfg["backtest"]["allow_short"],
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---- 1. sensitivity ---------------------------------------------------
    print(f"\n=== {name} · parameter sensitivity · training window ===")
    searches = {}
    fig, axes = plt.subplots(1, len(grids), figsize=(6.2 * len(grids), 5))
    axes = np.atleast_1d(axes)

    for ax, (strategy, grid) in zip(axes, grids.items()):
        search = validation.grid_search(train, strategy, grid, **common)
        searches[strategy] = search
        best = search.iloc[0]
        params = validation.coerce_params({k: best[k] for k in grid})
        print(f"  {strategy}: {len(search)} trials, best Sharpe "
              f"{best['sharpe']:.3f} at {params}")
        print(f"    grid Sharpe range {search['sharpe'].min():.3f} to "
              f"{search['sharpe'].max():.3f}, "
              f"{(search['sharpe'] > 0).mean():.0%} of trials positive")

        note, mesh = sensitivity_heatmap(ax, search, strategy)
        ax.set_title(f"{strategy} — Sharpe across the grid{note}", fontsize=10)
        fig.colorbar(mesh, ax=ax, fraction=0.046, label="Sharpe")

    fig.suptitle(f"{name} — is the best result a plateau or a spike?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_sensitivity.png", dpi=140)
    plt.close(fig)

    # ---- 2. walk-forward --------------------------------------------------
    print(f"\n=== {name} · walk-forward · "
          f"{vcfg['walk_forward']['train_days']}-bar train / "
          f"{vcfg['walk_forward']['test_days']}-bar test ===")
    wf_results = {}
    for strategy, grid in grids.items():
        wf = validation.walk_forward(
            prices, strategy, grid,
            train_days=vcfg["walk_forward"]["train_days"],
            test_days=vcfg["walk_forward"]["test_days"],
            objective=vcfg["walk_forward"]["objective"],
            **common,
        )
        wf_results[strategy] = wf
        folds = wf.folds
        print(f"  {strategy}: {wf.n_folds} folds, {wf.n_trials_per_fold} trials each")
        print(f"    in-sample Sharpe  (mean over folds): "
              f"{folds['in_sample_sharpe'].mean():+.3f}")
        print(f"    out-of-sample Sharpe (mean over folds): "
              f"{folds['out_of_sample_sharpe'].mean():+.3f}")
        print(f"    stitched out-of-sample Sharpe: "
              f"{metrics.sharpe(wf.returns):+.3f}")
        print(f"    total out-of-sample return: "
              f"{(1 + wf.returns).prod() - 1:+.2%}")
        print(f"    parameter stability across folds: "
              f"{wf.parameter_stability:.0%}")
        folds.to_csv(REPORTS / f"{name}_{strategy}_walk_forward_folds.csv",
                     index=False)

    fig, (ax_eq, ax_sc) = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, wf in wf_results.items():
        ax_eq.plot(wf.equity.index, wf.equity.values, linewidth=1.2,
                   label=f"{strategy} (walk-forward)")

    # The benchmark is drawn over exactly the walk-forward window, not the
    # whole series. Walk-forward cannot start until its first training block
    # is full and stops at the last complete fold; letting buy-and-hold run
    # over a longer window would compare two different periods.
    wf_index = next(iter(wf_results.values())).equity.index
    hold_prices = prices.loc[wf_index[0]: wf_index[-1]]
    hold = engine.run(
        hold_prices,
        pd.Series(1.0, index=hold_prices.index),
        pip_size=spec["pip_size"], spread_pips=spec["typical_spread_pips"],
        execution=cfg["backtest"]["execution"],
        initial_equity=cfg["backtest"]["initial_equity"],
    )
    ax_eq.plot(hold.equity.index, hold.equity.values, linewidth=1.0,
               color="grey", linestyle="--", label="buy and hold")
    print(f"  buy and hold over the same window: "
          f"Sharpe {metrics.sharpe(hold.returns):+.3f}, "
          f"return {(1 + hold.returns).prod() - 1:+.2%}")
    ax_eq.set_title(f"{name} — out-of-sample equity, parameters re-optimised each fold")
    ax_eq.set_ylabel("Equity")
    ax_eq.legend(fontsize=8)
    ax_eq.grid(alpha=0.3)

    for strategy, wf in wf_results.items():
        ax_sc.scatter(wf.folds["in_sample_sharpe"],
                      wf.folds["out_of_sample_sharpe"],
                      s=38, alpha=0.85, label=strategy)
    lim = 2.5
    ax_sc.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8,
               linestyle=":", label="if the fit held up")
    ax_sc.axhline(0, color="black", linewidth=0.8)
    ax_sc.axvline(0, color="black", linewidth=0.8)
    ax_sc.set_xlabel("In-sample Sharpe (chosen on the training block)")
    ax_sc.set_ylabel("Out-of-sample Sharpe (the block that followed)")
    ax_sc.set_title("Every point below the diagonal is fitting, not forecasting")
    ax_sc.legend(fontsize=8)
    ax_sc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_walk_forward.png", dpi=140)
    plt.close(fig)

    # ---- 3. multiple testing ---------------------------------------------
    print(f"\n=== {name} · multiple-testing adjustment ===")
    rows = []
    for strategy, search in searches.items():
        best = search.iloc[0]
        grid = grids[strategy]
        signal = validation.build_signal(
            train, strategy, {k: best[k] for k in grid},
            cfg["backtest"]["allow_short"],
        )
        best_result = engine.run(
            train, signal,
            pip_size=spec["pip_size"],
            spread_pips=spec["typical_spread_pips"],
            execution=cfg["backtest"]["execution"],
            initial_equity=cfg["backtest"]["initial_equity"],
        )
        report = validation.multiple_testing_report(search, best_result.returns)
        report["strategy"] = strategy
        report["walk_forward_sharpe"] = metrics.sharpe(wf_results[strategy].returns)
        rows.append(report)

        print(f"  {strategy}:")
        print(f"    best in-sample Sharpe            "
              f"{report['sharpe_annualised']:+.3f}  (annualised)")
        print(f"    Sharpe luck alone would produce  "
              f"{report['benchmark_sharpe_annualised']:+.3f}  "
              f"after {report['n_trials']} trials")
        print(f"    deflated Sharpe ratio            "
              f"{report['deflated_sharpe_ratio']:.4f}  "
              f"({'survives' if report['deflated_sharpe_ratio'] > 0.95 else 'does not survive'} "
              f"the 0.95 threshold)")

    pd.DataFrame(rows).set_index("strategy").to_csv(
        REPORTS / f"{name}_multiple_testing.csv"
    )

    print(f"\n[ok] figures -> {FIGURES}")
    print(f"[ok] tables  -> {REPORTS}")


if __name__ == "__main__":
    main()
