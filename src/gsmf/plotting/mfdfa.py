from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..multifractal.mfdfa import MFDFAResult


def plot_mfdfa_Fq(
    res: MFDFAResult,
    ax: Optional[plt.Axes] = None,
    title: str = "MFDFA fluctuation function",
    xlabel: str = "Scale s",
    ylabel: str = "Fq(s)",
    show_legend: bool = True,
):
    """Log-log plot of Fq(s) vs scale for multiple q."""
    if ax is None:
        fig, ax = plt.subplots()

    s = res.scales.astype(float)
    for i, q in enumerate(res.qs):
        ax.loglog(s, res.Fq[i, :], label=f"q={q:g}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.grid(False)
    return ax


def plot_hq(
    res: MFDFAResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Generalized Hurst exponent h(q)",
    xlabel: str = "q",
    ylabel: str = "h(q)",
    with_errorbars: bool = True,
):
    """Plot generalized Hurst exponent h(q) (optionally with fit uncertainties)."""
    if ax is None:
        fig, ax = plt.subplots()

    if with_errorbars and res.hq_stderr is not None:
        ax.errorbar(res.qs, res.hq, yerr=res.hq_stderr, fmt="o-", linewidth=1.2, markersize=3)
    else:
        ax.plot(res.qs, res.hq, "o-", linewidth=1.2, markersize=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    return ax
