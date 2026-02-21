from __future__ import annotations

from typing import Tuple
import numpy as np


def detrend_linear(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    p = np.polyfit(x, y, deg=1)
    trend = np.polyval(p, x)
    return y - trend, trend


def detrend_poly(y: np.ndarray, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    p = np.polyfit(x, y, deg=order)
    trend = np.polyval(p, x)
    return y - trend, trend
