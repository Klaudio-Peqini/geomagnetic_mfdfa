from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, List

import numpy as np

from ..multifractal.mfdfa import mfdfa, MFDFAResult


def shuffle_surrogate(x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    y = x.copy()
    rng.shuffle(y)
    return y


def phase_randomized_surrogate(x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if rng is None:
        rng = np.random.default_rng()

    x0 = x - np.mean(x)
    X = np.fft.rfft(x0)
    mag = np.abs(X)
    phase = np.angle(X)

    if len(phase) > 2:
        phase[1:-1] = rng.uniform(0, 2*np.pi, size=len(phase)-2)

    Y = mag * np.exp(1j * phase)
    y = np.fft.irfft(Y, n=len(x0))
    return y + np.mean(x)


@dataclass
class SurrogateTestResult:
    original: MFDFAResult
    surrogates: List[MFDFAResult]
    metric_name: str
    metric_original: float
    metric_surrogates: np.ndarray
    p_value: float


def surrogate_mfdfa_test(
    x: np.ndarray,
    scales: Sequence[int],
    qs: Sequence[float] = tuple(np.linspace(-5, 5, 21)),
    poly_order: int = 2,
    fit_range: Optional[Tuple[int, int]] = None,
    n_surrogates: int = 50,
    surrogate: str = "phase",
    metric: str = "hq_range",
    seed: Optional[int] = None,
) -> SurrogateTestResult:
    rng = np.random.default_rng(seed)
    orig = mfdfa(x, scales=scales, qs=qs, poly_order=poly_order, fit_range=fit_range)

    def compute_metric(res: MFDFAResult) -> float:
        if metric == "hq_range":
            return float(np.nanmax(res.hq) - np.nanmin(res.hq))
        if metric == "h2":
            i = int(np.argmin(np.abs(res.qs - 2.0)))
            return float(res.hq[i])
        raise ValueError("Unknown metric")

    m0 = compute_metric(orig)
    surr_results = []
    ms = []
    for _ in range(n_surrogates):
        if surrogate == "phase":
            xs = phase_randomized_surrogate(x, rng=rng)
        elif surrogate == "shuffle":
            xs = shuffle_surrogate(x, rng=rng)
        else:
            raise ValueError("surrogate must be 'phase' or 'shuffle'")
        try:
            r = mfdfa(xs, scales=scales, qs=qs, poly_order=poly_order, fit_range=fit_range)
        except Exception:
            continue
        surr_results.append(r)
        ms.append(compute_metric(r))

    ms = np.array(ms, dtype=float)
    p = (np.sum(np.abs(ms) >= np.abs(m0)) + 1) / (len(ms) + 1)

    return SurrogateTestResult(
        original=orig,
        surrogates=surr_results,
        metric_name=metric,
        metric_original=m0,
        metric_surrogates=ms,
        p_value=float(p),
    )
