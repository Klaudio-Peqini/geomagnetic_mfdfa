from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class MFDFAResult:
    scales: np.ndarray
    qs: np.ndarray
    Fq: np.ndarray
    hq: np.ndarray
    hq_stderr: np.ndarray
    fit_range: Tuple[int, int]
    poly_order: int
    info: Dict[str, Any]


def _profile(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("Input series too short after removing non-finite values.")
    return np.cumsum(x - np.mean(x))


def _detrended_var(seg: np.ndarray, m: int) -> float:
    n = len(seg)
    t = np.arange(n, dtype=float)
    p = np.polyfit(t, seg, deg=m)
    fit = np.polyval(p, t)
    res = seg - fit
    return np.mean(res * res)


def mfdfa(
    x: np.ndarray,
    scales: Sequence[int],
    qs: Sequence[float] = tuple(np.linspace(-5, 5, 21)),
    poly_order: int = 2,
    fit_range: Optional[Tuple[int, int]] = None,
) -> MFDFAResult:
    """Multifractal Detrended Fluctuation Analysis (MFDFA)."""
    Y = _profile(x)
    scales = np.array(list(scales), dtype=int)
    if np.any(scales < poly_order + 2):
        raise ValueError("All scales must be >= poly_order+2.")
    qs = np.array(list(qs), dtype=float)
    qs = np.unique(qs)
    qs.sort()

    Fq = np.full((len(qs), len(scales)), np.nan, dtype=float)

    for j, s in enumerate(scales):
        Ns = len(Y) // s
        if Ns < 2:
            continue
        vars_ = []
        for v in range(Ns):
            seg = Y[v * s : (v + 1) * s]
            vars_.append(_detrended_var(seg, poly_order))
        for v in range(Ns):
            seg = Y[len(Y) - (v + 1) * s : len(Y) - v * s]
            vars_.append(_detrended_var(seg, poly_order))
        F2 = np.array(vars_, dtype=float)

        for i, q in enumerate(qs):
            if np.isclose(q, 0.0):
                Fq[i, j] = np.exp(0.5 * np.mean(np.log(F2 + 1e-300)))
            else:
                Fq[i, j] = (np.mean((F2 ** (q / 2.0))) + 1e-300) ** (1.0 / q)

    valid_scales = np.isfinite(Fq).all(axis=0)
    if fit_range is None:
        idx = np.where(valid_scales)[0]
        if len(idx) < 3:
            raise ValueError("Not enough valid scales to fit h(q).")
        i0, i1 = idx[0], idx[-1] + 1
    else:
        i0, i1 = fit_range

    log_s = np.log(scales[i0:i1].astype(float))
    hq = np.zeros(len(qs), dtype=float)
    hq_se = np.zeros(len(qs), dtype=float)

    for i, q in enumerate(qs):
        y = np.log(Fq[i, i0:i1])
        A = np.vstack([np.ones_like(log_s), log_s]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        h = coef[1]
        yhat = A @ coef
        resid = y - yhat
        dof = max(1, len(y) - 2)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        hq[i] = h
        hq_se[i] = float(np.sqrt(cov[1, 1]))

    return MFDFAResult(
        scales=scales,
        qs=qs,
        Fq=Fq,
        hq=hq,
        hq_stderr=hq_se,
        fit_range=(i0, i1),
        poly_order=poly_order,
        info=dict(n=len(Y)),
    )
