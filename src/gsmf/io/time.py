from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ensure_datetime_index(
    df: pd.DataFrame,
    time_col: str = "time",
    tz: Optional[str] = None,
    sort: bool = True,
) -> pd.DataFrame:
    """Ensure `df` has a DatetimeIndex."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if time_col not in out.columns:
            raise ValueError(f"No DatetimeIndex and missing time_col='{time_col}'")
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.set_index(time_col)

    if tz is not None:
        if out.index.tz is None:
            out.index = out.index.tz_localize(tz)
        else:
            out.index = out.index.tz_convert(tz)

    if sort:
        out = out.sort_index()
    return out


def infer_sampling_period(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    """Infer a typical sampling period from a DatetimeIndex."""
    if len(index) < 3:
        return None
    diffs = np.diff(index.view("int64"))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    med = np.median(diffs)
    return pd.to_timedelta(int(med), unit="ns")
