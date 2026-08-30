"""Download, cache and validate daily OHLC data.

Data is cached to CSV on first download so every later run is offline and
reproducible. Nothing downstream is allowed to silently accept broken data:
`validate` raises on the failures that quietly corrupt a backtest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["Open", "High", "Low", "Close"]


def load_config(path: Path | str | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (field, ticker); keep the field."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download(name: str, cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Fetch one instrument, using the CSV cache unless `refresh` is set."""
    import yfinance as yf

    cache_dir = ROOT / cfg["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.csv"

    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        spec = cfg["instruments"][name]
        raw = yf.download(
            spec["ticker"],
            start=cfg["data"]["start"],
            end=cfg["data"]["end"],
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if raw.empty:
            raise RuntimeError(f"{name}: yfinance returned no rows")
        df = _flatten(raw)[COLUMNS]
        df.to_csv(cache)

    df.index.name = "Date"
    return df[COLUMNS].astype(float)


def validate(name: str, df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality warnings. Raises on fatal problems.

    Bad data is the most common reason a backtest lies, so this runs before
    anything else and its output belongs in the report.
    """
    problems: list[str] = []

    if df.empty:
        raise ValueError(f"{name}: empty dataset")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index is not sorted ascending")
    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicated dates in index")

    nans = int(df[COLUMNS].isna().any(axis=1).sum())
    if nans:
        problems.append(f"{nans} rows with missing OHLC values (dropped)")

    non_positive = int((df[COLUMNS] <= 0).any(axis=1).sum())
    if non_positive:
        raise ValueError(f"{name}: {non_positive} rows with non-positive prices")

    bad_range = int(((df["High"] < df["Low"]) | (df["Close"] > df["High"]) |
                     (df["Close"] < df["Low"])).sum())
    if bad_range:
        problems.append(f"{bad_range} rows where OHLC bounds are inconsistent")

    # Flat bars are common in FX feeds on holidays; they are not tradable days.
    flat = int((df["High"] == df["Low"]).sum())
    if flat:
        problems.append(f"{flat} bars with High == Low (illiquid/holiday rows)")

    gaps = df.index.to_series().diff().dt.days
    long_gaps = int((gaps > 5).sum())
    if long_gaps:
        problems.append(f"{long_gaps} calendar gaps longer than 5 days")

    return problems


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows a backtest cannot use. Never fills forward: an invented
    price produces invented trades."""
    out = df.dropna(subset=COLUMNS)
    return out[~(out["High"] == out["Low"])]


def load(name: str, cfg: dict | None = None, refresh: bool = False):
    """Convenience: download + validate + clean. Returns (df, warnings)."""
    cfg = cfg or load_config()
    df = download(name, cfg, refresh=refresh)
    warnings = validate(name, df)
    return clean(df), warnings
