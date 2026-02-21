# Data

This repository is structured to keep **raw or lightly processed** inputs under `data/`.

## Contents (current)

### `data/seismic/`

- `eq_data_earthquake_reviewed_mag4.csv`
  - Earthquake catalogue (time, latitude, longitude, magnitude).
  - Columns: `time` (ISO string), `latitude`, `longitude`, `mag`.
- `eq_self_singlepair_19680831_W3350_perwin_selflag_singlepair.csv`
  - Example precomputed lag-correlation-like table.
  - Columns: `lag_hours`, `lag_days`, `n_valid_bins`, `corr_sign`, `match_frac`.

### `data/geomagnetic/`

- Placeholder directory for geomagnetic time series.
  - Expected format(s) supported by `gsmf.io.geomag`: CSV with a time column + one or more signal columns, or simple two-column (time, value).

## Notes

- Keep the original files unchanged; prefer adding derived series into `results/` or saving intermediate products in a separate folder.
