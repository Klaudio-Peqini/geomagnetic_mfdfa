from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..analysis.correlation_analysis import CorrelationResult, acf_confint


def plot_acf(
    res: CorrelationResult,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Lag (samples)",
    ylabel: str = "Autocorrelation",
    confint: bool = False,
    n_for_confint: Optional[int] = None,
    style: str = "k-",
    marker: Optional[str] = None,
    markersize: float = 3.0,
    linewidth: float = 1.2,
):
    """Plot an autocorrelation function.

    Parameters
    ----------
    res:
        Output of `gsmf.analysis.autocorrelation`.
    confint:
        If True, draw approximate white-noise confidence bounds.
    n_for_confint:
        Sample size for confint. Defaults to max lag + 1.
    style:
        Matplotlib style string for line (e.g., 'k-', 'm-').
    marker:
        Optional marker, e.g. '+' to mimic your figure.
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = res.lags
    y = res.corr

    if marker is None:
        ax.plot(x, y, style, linewidth=linewidth)
    else:
        ax.plot(x, y, style, marker=marker, markersize=markersize, linewidth=linewidth)

    if confint:
        n = int(n_for_confint) if n_for_confint is not None else int(np.max(x) + 1)
        lo, hi = acf_confint(n)
        ax.axhline(hi, linestyle="--", linewidth=1.0)
        ax.axhline(lo, linestyle="--", linewidth=1.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"ACF (method={res.method}, unbiased={res.unbiased})"
    ax.set_title(title)
    ax.grid(False)
    return ax


def plot_ccf(
    res: CorrelationResult,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Lag (samples)",
    ylabel: str = "Cross-correlation",
    style: str = "k-",
    marker: Optional[str] = None,
    markersize: float = 3.0,
    linewidth: float = 1.2,
):
    """Plot a cross-correlation function (lags >= 0)."""
    if ax is None:
        fig, ax = plt.subplots()

    x = res.lags
    y = res.corr

    if marker is None:
        ax.plot(x, y, style, linewidth=linewidth)
    else:
        ax.plot(x, y, style, marker=marker, markersize=markersize, linewidth=linewidth)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"CCF (method={res.method}, unbiased={res.unbiased})"
    ax.set_title(title)
    ax.grid(False)
    return ax
