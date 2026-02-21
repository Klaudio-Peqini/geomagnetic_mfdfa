from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class SpectrumResult:
    qs: np.ndarray
    hq: np.ndarray
    tau_q: np.ndarray
    alpha: np.ndarray
    f_alpha: np.ndarray
    info: Dict[str, Any]


def legendre_spectrum(qs: np.ndarray, hq: np.ndarray) -> SpectrumResult:
    """Compute f(α) from h(q) via Legendre transform."""
    qs = np.asarray(qs, dtype=float)
    hq = np.asarray(hq, dtype=float)
    tau = qs * hq - 1.0
    alpha = np.gradient(tau, qs)
    f = qs * alpha - tau
    return SpectrumResult(qs=qs, hq=hq, tau_q=tau, alpha=alpha, f_alpha=f, info={})
