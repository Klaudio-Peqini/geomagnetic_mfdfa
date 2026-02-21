"""Geomag-Seismic Multifractal (gsmf).

Core features:
- I/O helpers for geomagnetic and seismic time series
- preprocessing: filtering, segmentation, detrending
- multifractal analysis: MFDFA, structure functions, spectrum & cross-MFDFA
- analysis: correlation/autocorrelation, windowed analysis, surrogate tests
"""

from importlib.metadata import version as _version

__all__ = [
    "__version__",
]

try:
    __version__ = _version("gsmf")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# Convenience namespace (optional)
from . import plotting  # noqa: F401
