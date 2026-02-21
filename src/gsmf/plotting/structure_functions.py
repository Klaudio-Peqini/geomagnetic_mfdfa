from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from ..multifractal.structure_functions import StructureFunctionResult


def plot_structure_functions(
    res: StructureFunctionResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Structure functions",
    xlabel: str = "Lag ℓ (samples)",
    ylabel: str = "S_q(ℓ)",
    show_legend: bool = True,
):
    """Log-log plot of S_q(ℓ) vs lag for multiple q."""
    if ax is None:
        fig, ax = plt.subplots()

    for i, q in enumerate(res.qs):
        ax.loglog(res.lags, res.Sq[i, :], label=f"q={q:g}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.grid(False)
    return ax
