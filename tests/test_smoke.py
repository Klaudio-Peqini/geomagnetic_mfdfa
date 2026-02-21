import numpy as np
from gsmf.analysis import autocorrelation
from gsmf.multifractal import mfdfa, legendre_spectrum

def test_acf_smoke():
    x = np.random.default_rng(0).normal(size=1024)
    res = autocorrelation(x, max_lag=10, method="fft")
    assert res.corr.shape[0] == 11
    assert np.isfinite(res.corr[0])

def test_mfdfa_smoke():
    x = np.random.default_rng(0).normal(size=4096)
    scales = np.unique(np.logspace(np.log10(16), np.log10(512), 10).astype(int))
    r = mfdfa(x, scales=scales, qs=np.linspace(-3,3,13), poly_order=1)
    s = legendre_spectrum(r.qs, r.hq)
    assert s.alpha.shape == s.f_alpha.shape
