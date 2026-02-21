from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from ..multifractal.spectrum import SpectrumResult


def plot_spectrum(
    spec: SpectrumResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Multifractal spectrum f(α)",
    xlabel: str = "α",
    ylabel: str = "f(α)",
):
    """Plot multifractal spectrum f(α)."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(spec.alpha, spec.f_alpha, "o-", linewidth=1.2, markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    return ax
