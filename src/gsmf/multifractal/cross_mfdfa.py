from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class CrossMFDFAResult:
    scales: np.ndarray
    qs: np.ndarray
    Fxy_q: np.ndarray
    hxy_q: np.ndarray
    hxy_stderr: np.ndarray
    fit_range: Tuple[int, int]
    poly_order: int
    info: Dict[str, Any]


def _profile(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("Input series too short after removing non-finite values.")
    return np.cumsum(x - np.mean(x))


def _detrended_residual(seg: np.ndarray, m: int) -> np.ndarray:
    n = len(seg)
    t = np.arange(n, dtype=float)
    p = np.polyfit(t, seg, deg=m)
    fit = np.polyval(p, t)
    return seg - fit


def cross_mfdfa(
    x: np.ndarray,
    y: np.ndarray,
    scales: Sequence[int],
    qs: Sequence[float] = tuple(np.linspace(-5, 5, 21)),
    poly_order: int = 2,
    fit_range: Optional[Tuple[int, int]] = None,
) -> CrossMFDFAResult:
    """Cross-MFDFA (MF-DCCA style) between two series."""
    X = _profile(x)
    Y = _profile(y)
    n = min(len(X), len(Y))
    X = X[:n]
    Y = Y[:n]

    scales = np.array(list(scales), dtype=int)
    qs = np.array(sorted(set(qs)), dtype=float)

    Fxy = np.full((len(qs), len(scales)), np.nan)

    for j, s in enumerate(scales):
        Ns = n // s
        if Ns < 2:
            continue
        covs = []
        for v in range(Ns):
            segx = X[v*s:(v+1)*s]
            segy = Y[v*s:(v+1)*s]
            rx = _detrended_residual(segx, poly_order)
            ry = _detrended_residual(segy, poly_order)
            covs.append(np.mean(rx * ry))
        for v in range(Ns):
            segx = X[n-(v+1)*s:n-v*s]
            segy = Y[n-(v+1)*s:n-v*s]
            rx = _detrended_residual(segx, poly_order)
            ry = _detrended_residual(segy, poly_order)
            covs.append(np.mean(rx * ry))

        F2 = np.array(covs, dtype=float)
        A = np.abs(F2) + 1e-300
        for i, q in enumerate(qs):
            if np.isclose(q, 0.0):
                Fxy[i, j] = np.exp(0.5 * np.mean(np.log(A)))
            else:
                Fxy[i, j] = (np.mean(A ** (q / 2.0))) ** (1.0 / q)

    valid_scales = np.isfinite(Fxy).all(axis=0)
    if fit_range is None:
        idx = np.where(valid_scales)[0]
        if len(idx) < 3:
            raise ValueError("Not enough valid scales to fit hxy(q).")
        i0, i1 = idx[0], idx[-1] + 1
    else:
        i0, i1 = fit_range

    log_s = np.log(scales[i0:i1].astype(float))
    hxy = np.zeros(len(qs))
    hxy_se = np.zeros(len(qs))
    for i, q in enumerate(qs):
        yv = np.log(Fxy[i, i0:i1])
        Areg = np.vstack([np.ones_like(log_s), log_s]).T
        coef, *_ = np.linalg.lstsq(Areg, yv, rcond=None)
        h = coef[1]
        yhat = Areg @ coef
        resid = yv - yhat
        dof = max(1, len(yv) - 2)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(Areg.T @ Areg)
        hxy[i] = h
        hxy_se[i] = float(np.sqrt(cov[1, 1]))

    return CrossMFDFAResult(
        scales=scales, qs=qs, Fxy_q=Fxy,
        hxy_q=hxy, hxy_stderr=hxy_se,
        fit_range=(i0, i1), poly_order=poly_order,
        info=dict(n=n),
    )
