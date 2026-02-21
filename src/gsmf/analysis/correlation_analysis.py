from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np

Method = Literal["direct", "fft"]


@dataclass
class CorrelationResult:
    lags: np.ndarray
    corr: np.ndarray
    method: str
    demean: bool
    normalize: bool
    unbiased: bool


def autocorrelation(
    x: np.ndarray,
    max_lag: Optional[int] = None,
    method: Method = "fft",
    demean: bool = True,
    normalize: bool = True,
    unbiased: bool = False,
) -> CorrelationResult:
    """Autocorrelation function (ACF) for a 1D series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        raise ValueError("Series too short.")

    if demean:
        x = x - np.mean(x)

    if max_lag is None or max_lag > n - 1:
        max_lag = n - 1

    if method == "direct":
        corr_full = np.array([np.dot(x[: n - k], x[k:]) for k in range(max_lag + 1)], dtype=float)
    elif method == "fft":
        m = 1 << (2 * n - 1).bit_length()
        X = np.fft.rfft(x, n=m)
        S = X * np.conj(X)
        corr = np.fft.irfft(S, n=m)[: max_lag + 1]
        corr_full = corr.astype(float)
    else:
        raise ValueError("method must be 'direct' or 'fft'")

    if unbiased:
        denom = np.arange(n, n - max_lag - 1, -1, dtype=float)
    else:
        denom = float(n)
    corr_full = corr_full / denom

    if normalize and corr_full[0] != 0:
        corr_full = corr_full / corr_full[0]

    lags = np.arange(max_lag + 1, dtype=int)
    return CorrelationResult(lags=lags, corr=corr_full, method=method, demean=demean, normalize=normalize, unbiased=unbiased)


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: Optional[int] = None,
    method: Method = "fft",
    demean: bool = True,
    normalize: bool = True,
    unbiased: bool = False,
) -> CorrelationResult:
    """Cross-correlation C_xy(k) for k >= 0 (x with y shifted forward)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    if demean:
        x = x - np.mean(x)
        y = y - np.mean(y)

    if max_lag is None or max_lag > n - 1:
        max_lag = n - 1

    if method == "direct":
        corr_full = np.array([np.dot(x[: n - k], y[k:]) for k in range(max_lag + 1)], dtype=float)
    elif method == "fft":
        m = 1 << (2 * n - 1).bit_length()
        X = np.fft.rfft(x, n=m)
        Y = np.fft.rfft(y, n=m)
        corr = np.fft.irfft(X * np.conj(Y), n=m)[: max_lag + 1]
        corr_full = corr.astype(float)
    else:
        raise ValueError("method must be 'direct' or 'fft'")

    if unbiased:
        denom = np.arange(n, n - max_lag - 1, -1, dtype=float)
    else:
        denom = float(n)
    corr_full = corr_full / denom

    if normalize:
        sx = np.std(x)
        sy = np.std(y)
        if sx > 0 and sy > 0:
            corr_full = corr_full / (sx * sy)

    lags = np.arange(max_lag + 1, dtype=int)
    return CorrelationResult(lags=lags, corr=corr_full, method=method, demean=demean, normalize=normalize, unbiased=unbiased)


def acf_confint(n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Approx. white-noise confidence bounds for ACF: +/- z/sqrt(n)."""
    from math import sqrt
    z = 1.959963984540054
    if alpha != 0.05:
        z = 1.959963984540054
    b = z / sqrt(max(1, n))
    return (-b, b)
