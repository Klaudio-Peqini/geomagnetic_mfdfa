from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .time import ensure_datetime_index


def load_geomag_csv(
    path: str,
    time_col: str = "time",
    value_cols: Optional[Sequence[str]] = None,
    sep: str = ",",
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """Load a geomagnetic time series CSV."""
    df = pd.read_csv(path, sep=sep)
    df = ensure_datetime_index(df, time_col=time_col, tz=tz, sort=True)
    if value_cols is not None:
        keep = [c for c in value_cols if c in df.columns]
        if not keep:
            raise ValueError(f"None of value_cols found in CSV: {value_cols}")
        df = df[keep]
    return df
