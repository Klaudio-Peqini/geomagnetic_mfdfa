# Geomagnetic--Seismic Multifractal Analysis

## Overview

This repository provides a compact, research-oriented framework for
**multifractal analysis of geomagnetic time series** and their
**correlation with seismic activity**.

The core objective is to investigate whether the multifractal properties of
geomagnetic field fluctuations exhibit systematic variations before,
during, or after seismic events.

This project is motivated by the hypothesis that:

-   The lithosphere--atmosphere--ionosphere system may exhibit
    **scale-invariant signatures**
-   Seismic processes may induce **changes in intermittency, long-range
    correlations, or singularity spectra**
-   Multifractal measures may reveal subtle precursory dynamics not
    visible in standard spectral analysis

The package is designed to remain:

-   Compact
-   Reproducible
-   Modular
-   Physically interpretable

------------------------------------------------------------------------

## Scientific Motivation

The geomagnetic field is a complex signal influenced by:

-   Core dynamo processes
-   Ionospheric currents
-   Solar--terrestrial interactions
-   Lithospheric electromagnetic disturbances

Seismic processes are known to produce:

-   Electromagnetic emissions
-   ULF magnetic anomalies
-   Perturbations in ionospheric conductivity

Both geomagnetic and seismic time series exhibit:

-   Non-Gaussian statistics
-   Long-range correlations
-   Intermittency
-   Scaling laws

This repository explores these features through **multifractal
formalism**.

------------------------------------------------------------------------

## Core Questions

1.  Do geomagnetic time series show measurable changes in multifractal
    spectra before earthquakes?
2.  Can seismic time clustering be reflected in scaling exponents?
3.  Are geomagnetic and seismic datasets jointly multifractal?
4.  Does cross-correlated multifractal analysis (MF-X-DFA) reveal
    coupling?

------------------------------------------------------------------------

## Methodological Framework

The repository implements:

### 1. Preprocessing

-   Detrending
-   Filtering (ULF bands if needed)
-   Window segmentation
-   Stationarity checks

### 2. Multifractal Analysis

-   MF-DFA (Multifractal Detrended Fluctuation Analysis)
-   Structure functions
-   Generalized Hurst exponent $H(q)$
-   Mass exponent $τ(q)$
-   Singularity spectrum $f(α)$

### 3. Cross-Correlation Analysis

-   DCCA
-   MF-X-DFA
-   Time-lagged correlation structure
-   Joint scaling behavior

### 4. Statistical Validation

-   Surrogate testing
-   Shuffling tests
-   Significance assessment

------------------------------------------------------------------------

## Repository Structure (Planned)

    geomag-seismic-multifractal/

    │
    ├── data/
    │   ├── geomagnetic/
    │   └── seismic/
    │
    ├── preprocessing/
    │   ├── filters.py
    │   ├── segmentation.py
    │   └── detrending.py
    │
    ├── multifractal/
    │   ├── mfdaf.py
    │   ├── structure_functions.py
    │   ├── spectrum.py
    │   └── cross_mfdfa.py
    │
    ├── analysis/
    │   ├── correlation_analysis.py
    │   ├── windowed_analysis.py
    │   └── surrogate_tests.py
    │
    ├── notebooks/
    │   └── exploratory_analysis.ipynb
    │
    ├── tests/
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Input Data

### Geomagnetic Data

-   1 Hz or minute-resolution magnetic field components ($X, Y, Z$ or $H, D, Z$)
-   Local observatory data or INTERMAGNET format

### Seismic Data

-   Earthquake catalog
    -   Origin time
    -   Magnitude
    -   Depth
    -   Location

Optional: - Seismic energy release time series - Cumulative Benioff
strain

------------------------------------------------------------------------

## Outputs

The framework generates:

-   Generalized Hurst exponent curves
-   Multifractal spectra $f(α)$
-   Time-evolving scaling exponents
-   Cross-correlation scaling maps
-   Statistical confidence intervals

------------------------------------------------------------------------

## Theoretical Background (Compact)

For a time series x(t), MF-DFA computes:

$F_q(s) \sim s\^{h(q)}$

where: - s is scale - h(q) is generalized Hurst exponent

Mass exponent:

$τ(q) = qh(q) - 1$

Singularity spectrum:

$α = dτ/dq\$
$f(α) = qα - τ(q)$

Width of spectrum:

$Δα = α_{max} - α_{min}$

A wider $Δα$ implies stronger multifractality.

------------------------------------------------------------------------

## Research Workflow

1.  Load geomagnetic and seismic datasets
2.  Align time windows
3.  Compute MF-DFA for sliding windows
4.  Track evolution of:
    -   $H(2)$
    -   $Δα$
5.  Compare with seismic activity metrics
6.  Perform surrogate validation
7.  Interpret physical coupling mechanisms

------------------------------------------------------------------------

## Installation

``` bash
git clone https://github.com/<your-username>/geomag-seismic-multifractal.git
cd geomag-seismic-multifractal
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Dependencies

-   numpy
-   scipy
-   matplotlib
-   pandas
-   numba (optional acceleration)
-   tqdm

------------------------------------------------------------------------

## Reproducibility

All analyses are:

-   Deterministic
-   Seed-controlled
-   Designed for HPC compatibility
-   Suitable for batch processing

------------------------------------------------------------------------

## Long-Term Extensions

-   GPU acceleration
-   Real-time anomaly detection
-   Integration with geodynamo simulations
-   Coupling with ionospheric TEC data
-   Deep learning comparison with multifractal indicators

------------------------------------------------------------------------

## Intended Audience

-   Geophysicists
-   Space physics researchers
-   Seismologists
-   Nonlinear dynamics researchers
-   Statistical physicists
-   Time series analysts
-   Data scientists in general

------------------------------------------------------------------------

## License

To be defined.

------------------------------------------------------------------------

## Author

Klaudio Peqini\
Physicist \| Geomagnetic Field Modeling \| Nonlinear Dynamics
