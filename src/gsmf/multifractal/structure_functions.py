from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class StructureFunctionResult:
    lags: np.ndarray
    qs: np.ndarray
    Sq: np.ndarray
    zeta_q: np.ndarray
    zeta_stderr: np.ndarray
    fit_range: Tuple[int, int]
    info: Dict[str, Any]


def structure_functions(
    x: np.ndarray,
    lags: Sequence[int],
    qs: Sequence[float] = tuple(np.linspace(0.5, 6.0, 12)),
    fit_range: Optional[Tuple[int, int]] = None,
    absolute: bool = True,
) -> StructureFunctionResult:
    """Compute q-order structure functions S_q(ℓ) = <|x(t+ℓ)-x(t)|^q>."""
    x = np.asarray(x, dtype=float)
    lags = np.array(list(lags), dtype=int)
    qs = np.array(sorted(set(qs)), dtype=float)

    Sq = np.full((len(qs), len(lags)), np.nan)
    for j, ell in enumerate(lags):
        if ell <= 0 or ell >= len(x):
            continue
        dx = x[ell:] - x[:-ell]
        if absolute:
            dx = np.abs(dx)
        for i, q in enumerate(qs):
            Sq[i, j] = np.mean(dx**q)

    valid = np.isfinite(Sq).all(axis=0)
    if fit_range is None:
        idx = np.where(valid)[0]
        if len(idx) < 3:
            raise ValueError("Not enough valid lags to fit scaling exponents.")
        i0, i1 = idx[0], idx[-1] + 1
    else:
        i0, i1 = fit_range

    log_l = np.log(lags[i0:i1].astype(float))
    zeta = np.zeros(len(qs))
    zeta_se = np.zeros(len(qs))
    for i, q in enumerate(qs):
        y = np.log(Sq[i, i0:i1])
        A = np.vstack([np.ones_like(log_l), log_l]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        z = coef[1]
        yhat = A @ coef
        resid = y - yhat
        dof = max(1, len(y) - 2)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        zeta[i] = z
        zeta_se[i] = float(np.sqrt(cov[1, 1]))

    return StructureFunctionResult(
        lags=lags,
        qs=qs,
        Sq=Sq,
        zeta_q=zeta,
        zeta_stderr=zeta_se,
        fit_range=(i0, i1),
        info=dict(n=len(x)),
    )
