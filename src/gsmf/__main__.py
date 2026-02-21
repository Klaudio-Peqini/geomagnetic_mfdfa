from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from gsmf.io import load_earthquake_catalog, events_to_counts
from gsmf.analysis import autocorrelation
from gsmf.multifractal import mfdfa, legendre_spectrum


def main():
    p = argparse.ArgumentParser(description="gsmf quick CLI: autocorrelation + mfdfa on binned event series")
    p.add_argument("--catalog", type=str, default="data/seismic/eq_data_earthquake_reviewed_mag4.csv")
    p.add_argument("--freq", type=str, default="1D", help="binning frequency (e.g. 1H, 1D)")
    p.add_argument("--max-lag", type=int, default=365)
    p.add_argument("--out-prefix", type=str, default="results/tables/quick")
    args = p.parse_args()

    cat = load_earthquake_catalog(args.catalog)
    series = events_to_counts(cat, freq=args.freq)
    x = series.values.astype(float)

    acf = autocorrelation(x, max_lag=args.max_lag, method="fft")

    scales = np.unique(np.logspace(np.log10(16), np.log10(min(2048, max(64, len(x)//4))), 18).astype(int))
    m = mfdfa(x, scales=scales, qs=np.linspace(-5,5,21), poly_order=2)
    spec = legendre_spectrum(m.qs, m.hq)

    pd.DataFrame({"lag": acf.lags, "acf": acf.corr}).to_csv(args.out_prefix + "_acf.csv", index=False)
    pd.DataFrame({"q": m.qs, "hq": m.hq, "hq_stderr": m.hq_stderr}).to_csv(args.out_prefix + "_mfdfa_hq.csv", index=False)
    pd.DataFrame({"q": spec.qs, "tau": spec.tau_q, "alpha": spec.alpha, "f_alpha": spec.f_alpha}).to_csv(args.out_prefix + "_spectrum.csv", index=False)

    print("Wrote:")
    print(" -", args.out_prefix + "_acf.csv")
    print(" -", args.out_prefix + "_mfdfa_hq.csv")
    print(" -", args.out_prefix + "_spectrum.csv")


if __name__ == "__main__":
    main()
