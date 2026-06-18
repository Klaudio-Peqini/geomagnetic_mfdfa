# Short-record workflow for radon activity and geomagnetic/RPI time series

This document adds a conservative workflow for geophysical time series with only a few thousand samples. It is intended for records such as:

- hourly radon activity series of a few weeks to a few months;
- geomagnetic or paleomagnetic RPI/VADM series with roughly 2,000--4,000 samples.

The purpose is not to replace the existing seismic pipeline. Instead, this workflow prevents overinterpretation when MFDFA is applied to short, nonstationary, or environmentally forced records.

---

## Scientific motivation

For short records, MFDFA can produce visually attractive log--log plots and broad multifractal spectra even when the apparent scaling is mainly caused by finite-size effects, trends, seasonality, environmental forcing, or a heavy-tailed amplitude distribution. Therefore, the central question should not be simply:

> Is the signal multifractal?

The safer question is:

> Does the cleaned or residual signal show scaling behavior that remains stable over a conservative scale range and exceeds shuffled, phase-randomized, and IAAFT surrogate controls?

---

## Conservative parameter rules

For a record of length `N`, use:

```text
smax ≈ N / 10
min_segments >= 8--10
q ∈ [-3, 3] for radon
q ∈ [-4, 4] for geomagnetic/RPI records
poly_order = 1 by default
```

Avoid very broad q-ranges such as `[-10, 10]` or `[-30, 30]` for short radon or geomagnetic records. Large positive q overweights rare bursts, while large negative q overweights tiny local variances.

---

## Radon workflow

Radon activity should not be analyzed directly as a raw level series unless the goal is only exploratory visualization. A more defensible workflow is:

```text
Radon Eye log
→ parse time, radon, temperature, humidity
→ regularize to hourly grid
→ detect gaps
→ log-transform radon
→ remove diurnal component
→ regress against temperature, humidity, and diurnal harmonics
→ remove very slow drift with a rolling median
→ run MFDFA on residuals and increments
→ compare against shuffled, phase, and IAAFT surrogates
```

Recommended command:

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

The recommended radon series are:

- `log_radon_env_residual_drift_removed`
- `d_log_radon`

Use `--series all` to inspect all intermediate variants.

---

## Geomagnetic/RPI workflow

For a short RPI record, the level series may appear highly persistent simply because it is smooth or slowly varying. Always compare the level series with anomalies and increments:

```text
RPI(t)
→ check uniform age spacing
→ robust normalization
→ rolling anomaly after removing slow trend
→ first differences ΔRPI
→ MFDFA on anomaly and increments
→ surrogate comparison
```

Recommended command:

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

The recommended geomagnetic series are:

- `rpi_rolling_anomaly`
- `d_rpi`

Use `--series all` to also analyze `rpi_level`, but interpret it cautiously.

---

## Output files

For every analyzed series, the script writes:

```text
cleaned_series_used.csv
acf.csv
acf.png
mfdfa_Fq_matrix.csv
scale_diagnostics.csv
mfdfa_hq.csv
mfdfa_spectrum.csv
surrogate_hq_quantiles.csv
surrogate_summary_quantiles.csv
mfdfa_loglog_fits.png
hq_with_surrogates.png
spectrum.png
short_record_quality_report.txt
short_record_quality_report.json
```

At the run level, it writes:

```text
run_metadata.json
summary_short_record_results.csv
summary_short_record_results.json
READ_ME_FIRST.txt
```

---

## Interpretation checklist

Before claiming multifractality, check the following:

1. The fitted scale range spans at least about 0.7 decades.
2. The largest scale leaves at least 8--10 forward segments.
3. The `q≈2` fit has high `R²`, preferably above 0.95 and ideally above 0.98.
4. The result is not driven only by the raw level series.
5. The original `h(q)`, `Δh`, or `Δα` is clearly separated from surrogate bands.
6. Radon conclusions are based on corrected residuals or increments, not raw activity alone.
7. Geomagnetic conclusions compare RPI level, anomaly, and first differences.

Suggested language:

> The residual/increment series shows scale-dependent persistence and intermittency beyond the adopted surrogate controls.

Avoid stronger language such as:

> The raw radon or RPI series proves multifractality.

---

## Optional dependency for Excel input

The main repository dependencies are enough for CSV input. For Excel input, install:

```bash
pip install openpyxl
```

Alternatively, save the spreadsheet as CSV and run the same script on the CSV file.
