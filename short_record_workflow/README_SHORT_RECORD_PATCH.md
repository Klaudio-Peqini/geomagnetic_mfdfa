# README addition: short-record geophysical workflow

You can paste the following section into the main `README.md`, preferably after the existing **Geomagnetic Workflow Example** section or before the seismic CLI examples.

---

## Short-record workflow for radon and geomagnetic/RPI series

For records with only a few thousand samples, the full seismic-style parameter scans should not be used directly. Short radon and geomagnetic/RPI records require conservative scale limits, narrow q-ranges, preprocessing diagnostics, and surrogate-based interpretation.

A practical rule is:

```text
smax ≈ N / 10
min_segments >= 8--10
q ∈ [-3, 3] for radon
q ∈ [-4, 4] for geomagnetic/RPI records
poly_order = 1 by default
```

### Radon activity

Radon activity should be treated as an environmentally modulated signal. The recommended workflow is:

```text
raw radon
→ hourly regularization and gap report
→ log-transform
→ remove diurnal component
→ regress against temperature, humidity, and diurnal harmonics
→ remove slow drift
→ analyze residuals and increments
→ compare against shuffled, phase-randomized, and IAAFT surrogates
```

Example:

```bash
python scripts/analyze_short_geophysical_series.py \
  --mode radon \
  --input data/radon/PE22111010052_LogData.txt \
  --out results_short/radon \
  --qmin -3 --qmax 3 --qstep 0.5 \
  --smin 8 \
  --min-segments 10 \
  --poly 1 \
  --series recommended \
  --n-surrogates 100
```

### Geomagnetic/RPI series

For short RPI records, compare the level series with a slow-trend anomaly and first differences:

```text
RPI(t)
→ check age spacing
→ normalize
→ remove slow trend with rolling anomaly
→ compute ΔRPI
→ analyze anomaly and increments
→ compare against shuffled, phase-randomized, and IAAFT surrogates
```

Example:

```bash
python scripts/analyze_short_geophysical_series.py \
  --mode geomag \
  --input data/geomagnetic/PADM2M.xlsx \
  --out results_short/PADM2M \
  --age-unit kyr \
  --qmin -4 --qmax 4 --qstep 0.5 \
  --smin 8 \
  --min-segments 10 \
  --poly 1 \
  --series recommended \
  --n-surrogates 100
```

For Excel input, install the optional reader:

```bash
pip install openpyxl
```

The most important output files are:

```text
scale_diagnostics.csv
mfdfa_hq.csv
mfdfa_spectrum.csv
surrogate_hq_quantiles.csv
surrogate_summary_quantiles.csv
short_record_quality_report.txt
summary_short_record_results.csv
```

Interpretation should be based on whether the cleaned/residual signal remains outside the surrogate bands. Do not claim physical multifractality from raw short records alone.
