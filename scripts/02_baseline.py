"""Baseline run: both strategies vs buy-and-hold, on the TRAINING window only.

    python scripts/02_baseline.py [--instrument EURUSD]

The test window is not touched here on purpose. Parameters get chosen on
training data; the out-of-sample window is spent once, at the end.

Produces:
  reports/figures/<instrument>_equity.png
  reports/figures/<instrument>_spread_sweep.png
  reports/<instrument>_baseline.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, engine, metrics, strategies  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"


def build_signals(prices, cfg):
    p = cfg["strategies"]
    short = cfg["backtest"]["allow_short"]
    return {
        "buy_and_hold": strategies.buy_and_hold(prices),
        "ma_crossover": strategies.ma_crossover(
            prices, p["ma_crossover"]["fast"], p["ma_crossover"]["slow"], short
        ),
        "rsi_reversion": strategies.rsi_reversion(
            prices,
            p["rsi_reversion"]["period"],
            p["rsi_reversion"]["oversold"],
            p["rsi_reversion"]["overbought"],
            short,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="EURUSD")
    args = ap.parse_args()

    cfg = data.load_config()
    name = args.instrument
    spec = cfg["instruments"][name]

    prices, warnings = data.load(name, cfg)
    train = prices.loc[: cfg["split"]["train_end"]]
    print(f"[i] {name}: {len(train)} training bars "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    for w in warnings:
        print(f"    [!] {w}")

    signals = build_signals(train, cfg)
    spread = spec["typical_spread_pips"]
    common = dict(
        pip_size=spec["pip_size"],
        execution=cfg["backtest"]["execution"],
        initial_equity=cfg["backtest"]["initial_equity"],
    )

    results = {
        label: engine.run(train, sig, spread_pips=spread, **common)
        for label, sig in signals.items()
    }

    summary = metrics.table(results)
    print(f"\n=== {name} · training window · spread {spread} pips ===")
    print(summary[["total_return", "cagr", "sharpe", "max_drawdown",
                   "hit_rate", "n_trades"]].round(3).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROOT / "reports" / f"{name}_baseline.csv")

    # --- equity curves -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=label, linewidth=1.2)
    ax.set_title(f"{name} — equity, training window (spread {spread} pips)")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_equity.png", dpi=140)
    plt.close(fig)

    # --- the headline chart: does the edge survive the spread? ---------
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, sig in signals.items():
        if label == "buy_and_hold":
            continue
        sweep = engine.sweep_spread(
            train, sig, spreads=cfg["spread_sweep"], **common
        )
        ax.plot(sweep.index, sweep["total_return"] * 100, marker="o", label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(spread, color="red", linestyle="--", linewidth=0.8,
               label=f"typical spread ({spread} pips)")
    ax.set_title(f"{name} — where the edge dies")
    ax.set_xlabel("Spread charged (pips)")
    ax.set_ylabel("Total return over training window (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_spread_sweep.png", dpi=140)
    plt.close(fig)

    print(f"\n[ok] figures -> {FIGURES}")


if __name__ == "__main__":
    main()
