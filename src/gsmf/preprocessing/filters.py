from __future__ import annotations

import numpy as np
from scipy import signal


def butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype="low")
    return signal.filtfilt(b, a, x)


def butter_highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype="high")
    return signal.filtfilt(b, a, x)


def butter_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    b, a = signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band")
    return signal.filtfilt(b, a, x)


def rolling_zscore(x: np.ndarray, window: int, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window < 3:
        raise ValueError("window must be >= 3")
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        w = x[i - window + 1 : i + 1]
        mu = np.nanmean(w)
        sd = np.nanstd(w)
        out[i] = (x[i] - mu) / (sd + eps)
    return out
