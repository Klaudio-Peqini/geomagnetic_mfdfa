from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, List, Tuple

import numpy as np

from ..preprocessing.segmentation import sliding_windows
from ..multifractal.mfdfa import mfdfa, MFDFAResult


@dataclass
class WindowedMFDFAResult:
    starts: np.ndarray
    results: List[MFDFAResult]
    info: Dict[str, Any]


def windowed_mfdfa(
    x: np.ndarray,
    window: int,
    step: int,
    scales: Sequence[int],
    qs: Sequence[float] = tuple(np.linspace(-5, 5, 21)),
    poly_order: int = 2,
    fit_range: Optional[Tuple[int, int]] = None,
) -> WindowedMFDFAResult:
    """Run MFDFA over sliding windows."""
    x = np.asarray(x, dtype=float)
    starts = []
    out = []
    for start, seg in sliding_windows(x, window=window, step=step):
        try:
            res = mfdfa(seg, scales=scales, qs=qs, poly_order=poly_order, fit_range=fit_range)
        except Exception:
            continue
        starts.append(start)
        out.append(res)
    return WindowedMFDFAResult(starts=np.array(starts, dtype=int), results=out, info=dict(window=window, step=step))
