"""Download and cache every instrument, then report data quality.

    python scripts/01_download_data.py [--refresh]

Run this first. The quality warnings it prints are not noise — they belong
in the report, because they bound what the backtest can honestly claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the CSV cache")
    args = ap.parse_args()

    cfg = data.load_config()

    for name in cfg["instruments"]:
        try:
            df, warnings = data.load(name, cfg, refresh=args.refresh)
        except Exception as exc:
            print(f"[x] {name}: {type(exc).__name__}: {exc}")
            continue

        print(f"[ok] {name:<8} {len(df):>5} bars  "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        for w in warnings:
            print(f"     [!] {w}")


if __name__ == "__main__":
    main()
