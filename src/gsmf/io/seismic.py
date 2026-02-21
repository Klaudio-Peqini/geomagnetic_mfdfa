from __future__ import annotations

from typing import Optional

import pandas as pd

from .time import ensure_datetime_index


def load_earthquake_catalog(
    path: str,
    time_col: str = "time",
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """Load an earthquake catalog CSV."""
    df = pd.read_csv(path)
    df = ensure_datetime_index(df, time_col=time_col, tz=tz, sort=True)
    return df


def events_to_counts(
    catalog: pd.DataFrame,
    freq: str = "1H",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    weight_col: Optional[str] = None,
) -> pd.Series:
    """Convert an event catalog into an evenly sampled count (or weighted) time series."""
    df = catalog
    if not isinstance(df.index, pd.DatetimeIndex):
        df = ensure_datetime_index(df)

    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()

    binned = df.loc[(df.index >= start) & (df.index <= end)]
    if weight_col is None:
        s = binned.copy()
        s["_w"] = 1.0
    else:
        if weight_col not in binned.columns:
            raise ValueError(f"weight_col '{weight_col}' not found")
        s = binned[[weight_col]].copy()
        s["_w"] = pd.to_numeric(s[weight_col], errors="coerce").fillna(0.0)

    out = s["_w"].resample(freq).sum().astype(float)
    out.name = "counts" if weight_col is None else f"sum_{weight_col}"
    return out
