from __future__ import annotations

from typing import Iterator, Tuple
import numpy as np


def sliding_windows(x: np.ndarray, window: int, step: int) -> Iterator[Tuple[int, np.ndarray]]:
    x = np.asarray(x)
    if window <= 1 or step <= 0:
        raise ValueError("window must be >1 and step must be >0")
    for start in range(0, len(x) - window + 1, step):
        yield start, x[start : start + window]
