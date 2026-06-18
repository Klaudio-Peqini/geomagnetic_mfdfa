#!/usr/bin/env python3
"""
analyze_short_geophysical_series.py

Conservative short-record workflow for geomagnetic/RPI and radon activity time series.

The script is designed for records with O(10^3) samples, where standard MFDFA can
produce visually convincing but physically fragile spectra. It therefore enforces:

  * conservative q-ranges,
  * automatic scale selection using a minimum-segment rule,
  * diagnostics for the fitted scaling region,
  * preprocessing appropriate to radon and geomagnetic records,
  * shuffled, phase-randomized, and IAAFT surrogate comparisons.

Example, radon:

  python scripts/analyze_short_geophysical_series.py \
    --mode radon \
    --input data/radon/PE22111010052_LogData.txt \
    --out results_short/radon \
    --qmin -3 --qmax 3 --qstep 0.5 \
    --n-surrogates 100

Example, geomagnetic RPI:

  python scripts/analyze_short_geophysical_series.py \
    --mode geomag \
    --input data/geomagnetic/PADM2M.xlsx \
    --out results_short/PADM2M \
    --age-unit kyr \
    --qmin -4 --qmax 4 --qstep 0.5 \
    --n-surrogates 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def zscore(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    mu = np.nanmean(a)
    sig = np.nanstd(a)
    if not np.isfinite(sig) or sig <= 0:
        return a - mu
    return (a - mu) / sig


def finite_array(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    sx, sy = np.std(x[m]), np.std(y[m])
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def robust_numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, len(df) // 10):
            cols.append(str(c))
    return cols


def write_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 5.6),
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
    })


# -----------------------------------------------------------------------------
# Data loading and preprocessing: radon
# -----------------------------------------------------------------------------

RADON_LINE_RE = re.compile(
    r"^\s*(\d+)\)\s+"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s*°?C\s+"
    r"([-+]?\d+(?:\.\d+)?)\s*%\s*$"
)


def load_radon_eye_log(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    metadata: Dict[str, str] = {}
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if ":" in raw and not re.match(r"^\d+\)", raw):
                parts = raw.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                if key:
                    metadata[key] = val
            m = RADON_LINE_RE.match(raw)
            if m:
                rows.append({
                    "record_no": int(m.group(1)),
                    "time": pd.to_datetime(m.group(2), errors="coerce"),
                    "radon_bqm3": float(m.group(3)),
                    "temperature_c": float(m.group(4)),
                    "humidity_pct": float(m.group(5)),
                })
    if not rows:
        raise ValueError(f"No Radon Eye rows could be parsed from {path}")
    df = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df, metadata


def regularize_radon_hourly(df: pd.DataFrame, interpolate_limit: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy().sort_values("time")
    d["dt_hours"] = d["time"].diff().dt.total_seconds() / 3600.0
    gap_report = d.loc[d["dt_hours"].fillna(1.0) > 1.25, ["record_no", "time", "dt_hours"]].copy()

    hourly = (
        d.set_index("time")[["radon_bqm3", "temperature_c", "humidity_pct"]]
        .resample("1h")
        .mean()
    )
    hourly["was_missing"] = hourly["radon_bqm3"].isna()
    for col in ["radon_bqm3", "temperature_c", "humidity_pct"]:
        hourly[col] = hourly[col].interpolate("time", limit=interpolate_limit, limit_direction="both")
    hourly = hourly.dropna(subset=["radon_bqm3"]).copy()
    hourly.index.name = "time"
    return hourly.reset_index(), gap_report


def remove_diurnal_component(values: np.ndarray, time_index: pd.Series) -> np.ndarray:
    s = pd.Series(values, index=pd.to_datetime(time_index))
    hourly_mean = s.groupby(s.index.hour).transform("mean")
    return (s - hourly_mean + s.mean()).to_numpy(dtype=float)


def regress_environmental_radon(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    """Residual of log_radon against T, RH and diurnal harmonics."""
    t = pd.to_datetime(df["time"])
    y = np.log(np.maximum(pd.to_numeric(df["radon_bqm3"], errors="coerce"), 1e-9).to_numpy(dtype=float))
    temp = zscore(df["temperature_c"])
    hum = zscore(df["humidity_pct"])
    hour = t.dt.hour.to_numpy(dtype=float) + t.dt.minute.to_numpy(dtype=float) / 60.0
    sin1 = np.sin(2.0 * np.pi * hour / 24.0)
    cos1 = np.cos(2.0 * np.pi * hour / 24.0)
    X = np.column_stack([np.ones_like(y), temp, hum, sin1, cos1])
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
    fitted = X @ beta
    residual = y - fitted
    names = ["intercept", "temperature_z", "humidity_z", "diurnal_sin", "diurnal_cos"]
    coefs = {name: float(val) for name, val in zip(names, beta)}
    coefs["corr_raw_radon_temperature"] = safe_corr(df["radon_bqm3"], df["temperature_c"])
    coefs["corr_raw_radon_humidity"] = safe_corr(df["radon_bqm3"], df["humidity_pct"])
    coefs["corr_log_radon_temperature"] = safe_corr(y, df["temperature_c"])
    coefs["corr_log_radon_humidity"] = safe_corr(y, df["humidity_pct"])
    return residual, coefs


def rolling_anomaly(x: Sequence[float], window: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    trend = s.rolling(window=window, center=True, min_periods=max(5, window // 4)).median()
    trend = trend.interpolate(limit_direction="both")
    return (s - trend).to_numpy(dtype=float)


def prepare_radon(path: str | Path, out: Path, drift_window: int = 168) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict]:
    raw, meta = load_radon_eye_log(path)
    hourly, gaps = regularize_radon_hourly(raw)
    raw.to_csv(out / "radon_raw_parsed.csv", index=False)
    hourly.to_csv(out / "radon_regularized_hourly.csv", index=False)
    gaps.to_csv(out / "gap_report.csv", index=False)

    log_rn = np.log(np.maximum(hourly["radon_bqm3"].to_numpy(dtype=float), 1e-9))
    log_rn_diurnal_removed = remove_diurnal_component(log_rn, hourly["time"])
    env_resid, env_info = regress_environmental_radon(hourly)
    env_resid_drift_removed = rolling_anomaly(env_resid, drift_window)
    dlog = np.diff(log_rn, prepend=np.nan)

    hourly["log_radon"] = log_rn
    hourly["log_radon_diurnal_removed"] = log_rn_diurnal_removed
    hourly["log_radon_env_residual"] = env_resid
    hourly["log_radon_env_residual_drift_removed"] = env_resid_drift_removed
    hourly["d_log_radon"] = dlog
    hourly.to_csv(out / "radon_preprocessed_series.csv", index=False)

    series = {
        "log_radon": zscore(log_rn),
        "log_radon_diurnal_removed": zscore(log_rn_diurnal_removed),
        "log_radon_env_residual": zscore(env_resid),
        "log_radon_env_residual_drift_removed": zscore(env_resid_drift_removed),
        "d_log_radon": zscore(dlog[1:]),
    }
    info = {
        "metadata": meta,
        "n_raw": int(len(raw)),
        "n_regularized": int(len(hourly)),
        "first_time": str(hourly["time"].iloc[0]),
        "last_time": str(hourly["time"].iloc[-1]),
        "missing_bins_after_regularization": int(hourly["was_missing"].sum()),
        "environmental_regression": env_info,
        "recommended_units": "hourly samples; MFDFA scales are hours",
    }
    return hourly, series, info


# -----------------------------------------------------------------------------
# Data loading and preprocessing: geomagnetic/RPI
# -----------------------------------------------------------------------------


def read_table_auto(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt", ".dat"}:
        # Try common separators. If there is only one column, fall back to whitespace.
        df = pd.read_csv(p)
        if df.shape[1] == 1:
            df = pd.read_csv(p, sep=r"\s+", engine="python")
        return df
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p, sheet_name=sheet_name or 0)
        except ImportError as e:
            raise ImportError(
                "Reading Excel files requires openpyxl. Install it with: pip install openpyxl\n"
                "Alternatively, save the spreadsheet as CSV and use that CSV as input."
            ) from e
    raise ValueError(f"Unsupported file extension: {suffix}")


def choose_geomag_columns(df: pd.DataFrame, age_col: str = "auto", value_col: str = "auto") -> Tuple[str, str]:
    cols = list(map(str, df.columns))
    if age_col != "auto" and value_col != "auto":
        return age_col, value_col
    numeric = robust_numeric_columns(df)
    lower = {c: c.lower() for c in cols}
    if age_col == "auto":
        age_candidates = [c for c in numeric if any(k in lower[c] for k in ["age", "time", "kyr", "ka", "year"])]
        age = age_candidates[0] if age_candidates else numeric[0]
    else:
        age = age_col
    if value_col == "auto":
        value_candidates = [c for c in numeric if c != age and any(k in lower[c] for k in ["rpi", "vadm", "paleoint", "intensity", "value"])]
        if value_candidates:
            val = value_candidates[0]
        else:
            candidates = [c for c in numeric if c != age]
            if not candidates:
                raise ValueError("Could not identify a geomagnetic value column.")
            val = candidates[0]
    else:
        val = value_col
    return age, val


def prepare_geomag(
    path: str | Path,
    out: Path,
    age_col: str = "auto",
    value_col: str = "auto",
    sheet_name: Optional[str] = None,
    age_unit: str = "kyr",
    trend_window: int = 201,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict]:
    df0 = read_table_auto(path, sheet_name=sheet_name)
    df0.columns = [str(c).strip() for c in df0.columns]
    age, val = choose_geomag_columns(df0, age_col=age_col, value_col=value_col)
    df = df0[[age, val]].copy()
    df.columns = ["age", "value"]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("age").reset_index(drop=True)
    if len(df) < 64:
        raise ValueError("Geomagnetic series is too short after cleaning.")

    d_age = np.diff(df["age"].to_numpy())
    med_dt = float(np.median(d_age))
    uniform = bool(np.nanmax(np.abs(d_age - med_dt)) <= max(1e-9, 0.01 * abs(med_dt)))
    if not uniform:
        new_age = np.arange(float(df["age"].iloc[0]), float(df["age"].iloc[-1]) + 0.5 * med_dt, med_dt)
        new_val = np.interp(new_age, df["age"].to_numpy(), df["value"].to_numpy())
        df = pd.DataFrame({"age": new_age, "value": new_val})

    value = df["value"].to_numpy(dtype=float)
    dvalue = np.diff(value, prepend=np.nan)
    anomaly = rolling_anomaly(value, trend_window)
    df["d_value"] = dvalue
    df["rolling_anomaly"] = anomaly
    df.to_csv(out / "geomag_preprocessed_series.csv", index=False)

    series = {
        "rpi_level": zscore(value),
        "rpi_rolling_anomaly": zscore(anomaly),
        "d_rpi": zscore(dvalue[1:]),
    }
    dt_label = "samples"
    if age_unit.lower() in {"kyr", "ka"}:
        dt_label = f"{med_dt * 1000.0:g} years/sample"
    elif age_unit.lower() in {"yr", "year", "years"}:
        dt_label = f"{med_dt:g} years/sample"
    info = {
        "n_clean": int(len(df)),
        "age_column_used": age,
        "value_column_used": val,
        "age_min": float(df["age"].min()),
        "age_max": float(df["age"].max()),
        "median_age_step": med_dt,
        "age_unit": age_unit,
        "uniform_grid_detected": uniform,
        "scale_units": dt_label,
        "trend_window_samples": int(trend_window),
    }
    return df, series, info


# -----------------------------------------------------------------------------
# ACF, MFDFA, spectra, and surrogates
# -----------------------------------------------------------------------------


def autocorr_fft(x: Sequence[float], max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x = finite_array(x)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 finite samples for ACF.")
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return np.arange(min(max_lag, n - 1) + 1), np.full(min(max_lag, n - 1) + 1, np.nan)
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    acf = acf / np.arange(n, 0, -1, dtype=float)
    acf = acf / acf[0]
    max_lag = int(min(max_lag, n - 1))
    return np.arange(max_lag + 1), acf[:max_lag + 1]


def acf_summary(lags: np.ndarray, acf: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["acf_lag1"] = float(acf[1]) if len(acf) > 1 else float("nan")
    below_e = np.where(acf < 1.0 / math.e)[0]
    out["first_lag_below_1_over_e"] = float(below_e[0]) if below_e.size else float("nan")
    below_zero = np.where(acf < 0.0)[0]
    out["first_zero_crossing_lag"] = float(below_zero[0]) if below_zero.size else float("nan")
    out["white_noise_95_abs_threshold"] = float(1.96 / math.sqrt(max(len(acf), 1)))
    return out


def _poly_detrend(y: np.ndarray, order: int) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    deg = min(order, max(0, len(y) - 2))
    coeffs = np.polyfit(x, y, deg=deg)
    return y - np.polyval(coeffs, x)


def mfdfa(x: Sequence[float], scales: Iterable[int], qs: np.ndarray, poly_order: int = 1, var_floor: float = 1e-30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = finite_array(x)
    if x.size < 64:
        raise ValueError("Series too short for MFDFA after removing NaNs.")
    y = np.cumsum(x - np.mean(x))
    scales_arr = np.array(sorted(set(int(s) for s in scales if int(s) >= max(4, poly_order + 3) and int(s) <= x.size // 2)), dtype=int)
    if scales_arr.size == 0:
        raise ValueError("No valid MFDFA scales remain.")
    Fq = np.full((len(qs), len(scales_arr)), np.nan, dtype=float)
    nseg = np.full(len(scales_arr), 0, dtype=int)
    for j, s in enumerate(scales_arr):
        ns = x.size // s
        if ns < 2:
            continue
        vars_: List[float] = []
        for v in range(ns):
            seg = y[v * s:(v + 1) * s]
            vars_.append(float(np.mean(_poly_detrend(seg, poly_order) ** 2)))
        for v in range(ns):
            start = x.size - (v + 1) * s
            seg = y[start:start + s]
            vars_.append(float(np.mean(_poly_detrend(seg, poly_order) ** 2)))
        vars_arr = np.maximum(np.asarray(vars_, dtype=float), var_floor)
        nseg[j] = len(vars_arr)
        for i, q in enumerate(qs):
            if abs(q) < 1e-14:
                Fq[i, j] = np.exp(0.5 * np.mean(np.log(vars_arr)))
            else:
                val = np.mean(vars_arr ** (q / 2.0))
                if np.isfinite(val) and val > 0:
                    Fq[i, j] = val ** (1.0 / q)
    return scales_arr, Fq, nseg


def fit_hq(scales: np.ndarray, Fq: np.ndarray, qs: np.ndarray, fit_smin: Optional[int], fit_smax: Optional[int]) -> Tuple[pd.DataFrame, np.ndarray]:
    rows: List[Dict] = []
    masks = np.zeros_like(Fq, dtype=bool)
    log_s = np.log10(scales.astype(float))
    base = np.ones(scales.shape, dtype=bool)
    if fit_smin is not None:
        base &= scales >= fit_smin
    if fit_smax is not None:
        base &= scales <= fit_smax
    for i, q in enumerate(qs):
        y = Fq[i, :]
        m = base & np.isfinite(y) & (y > 0)
        n = int(m.sum())
        row: Dict[str, float] = {"q": float(q), "n_scales_used": n}
        if n < 3:
            row.update({"h": np.nan, "intercept": np.nan, "stderr_h": np.nan, "r2": np.nan, "fit_smin": np.nan, "fit_smax": np.nan, "fit_decades": np.nan})
        else:
            x = log_s[m]
            yy = np.log10(y[m])
            A = np.vstack([x, np.ones_like(x)]).T
            h, c = np.linalg.lstsq(A, yy, rcond=None)[0]
            pred = h * x + c
            resid = yy - pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            dof = max(n - 2, 1)
            s2 = ss_res / dof
            x_centered = x - np.mean(x)
            stderr = float(math.sqrt(s2 / np.sum(x_centered ** 2))) if np.sum(x_centered ** 2) > 0 else np.nan
            row.update({
                "h": float(h),
                "intercept": float(c),
                "stderr_h": stderr,
                "r2": float(r2),
                "fit_smin": int(scales[m][0]),
                "fit_smax": int(scales[m][-1]),
                "fit_decades": float(np.log10(scales[m][-1] / scales[m][0])),
            })
            masks[i, :] = m
        rows.append(row)
    return pd.DataFrame(rows), masks


def legendre_spectrum(qs: np.ndarray, hq: np.ndarray) -> pd.DataFrame:
    tau = qs * hq - 1.0
    alpha = np.gradient(tau, qs)
    falpha = qs * alpha - tau
    return pd.DataFrame({"q": qs, "h": hq, "tau": tau, "alpha": alpha, "f_alpha": falpha})


def spectrum_summary(spec: pd.DataFrame) -> Dict[str, float]:
    ok = spec[["alpha", "f_alpha"]].replace([np.inf, -np.inf], np.nan).dropna()
    out: Dict[str, float] = {}
    htable = spec[["q", "h"]].dropna()
    for q0 in [-4, -3, -2, 0, 2, 3, 4]:
        if not htable.empty:
            idx = (htable["q"] - q0).abs().idxmin()
            out[f"h_q{q0:g}"] = float(htable.loc[idx, "h"])
    if not ok.empty:
        out["delta_alpha"] = float(ok["alpha"].max() - ok["alpha"].min())
        imax = ok["f_alpha"].idxmax()
        out["alpha_peak"] = float(ok.loc[imax, "alpha"])
        out["f_alpha_peak"] = float(ok.loc[imax, "f_alpha"])
    else:
        out["delta_alpha"] = float("nan")
        out["alpha_peak"] = float("nan")
        out["f_alpha_peak"] = float("nan")
    if "h_q-4" in out and "h_q4" in out:
        out["delta_h_hminus4_h4"] = out["h_q-4"] - out["h_q4"]
    if "h_q-3" in out and "h_q3" in out:
        out["delta_h_hminus3_h3"] = out["h_q-3"] - out["h_q3"]
    return out


def default_scales(n: int, smin: int, smax: Optional[int], n_scales: int, min_segments: int) -> np.ndarray:
    if smax is None:
        smax = max(smin + 1, n // min_segments)
    smax = min(int(smax), max(smin + 1, n // 4))
    vals = np.unique(np.round(np.logspace(np.log10(smin), np.log10(smax), n_scales)).astype(int))
    vals = vals[(vals >= smin) & (vals <= smax)]
    # Keep only scales that leave at least min_segments forward segments.
    vals = vals[(n // vals) >= min_segments]
    if vals.size < 5:
        vals = np.array([s for s in range(smin, max(smin + 1, smax + 1)) if n // s >= min_segments], dtype=int)
    return vals


def shuffled_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.array(x, copy=True)
    rng.shuffle(y)
    return y


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    amp = np.abs(X)
    phases = np.angle(X)
    random_phases = rng.uniform(0, 2 * np.pi, size=len(X))
    random_phases[0] = phases[0]
    if n % 2 == 0:
        random_phases[-1] = phases[-1]
    y = np.fft.irfft(amp * np.exp(1j * random_phases), n=n)
    return (y - np.mean(y)) / (np.std(y) if np.std(y) > 0 else 1.0)


def iaaft_surrogate(x: np.ndarray, rng: np.random.Generator, n_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier-transform surrogate."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(max(1, n_iter)):
        Y = np.fft.rfft(y)
        phases = np.exp(1j * np.angle(Y))
        y = np.fft.irfft(target_amp * phases, n=n)
        ranks = np.argsort(np.argsort(y))
        y = sorted_x[ranks]
    return y


def surrogate_series(kind: str, x: np.ndarray, rng: np.random.Generator, iaaft_iter: int) -> np.ndarray:
    if kind == "shuffle":
        return shuffled_surrogate(x, rng)
    if kind == "phase":
        return phase_randomized_surrogate(x, rng)
    if kind == "iaaft":
        return iaaft_surrogate(x, rng, n_iter=iaaft_iter)
    raise ValueError(f"Unknown surrogate kind: {kind}")


# -----------------------------------------------------------------------------
# Plotting and reporting
# -----------------------------------------------------------------------------


def plot_series(df: pd.DataFrame, xcol: str, ycols: List[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    for y in ycols:
        if y in df.columns:
            ax.plot(df[xcol], df[y], label=y)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.savefig(path)
    plt.close(fig)


def plot_acf(lags: np.ndarray, acf: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(lags, acf, marker="o", markersize=2)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("lag [samples]")
    ax.set_ylabel("ACF")
    fig.savefig(path)
    plt.close(fig)


def plot_loglog_fits(scales: np.ndarray, Fq: np.ndarray, hq_df: pd.DataFrame, qs: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    pick_qs = [q for q in [-4, -2, 0, 2, 4] if np.any(np.isclose(qs, q))]
    if not pick_qs:
        pick_qs = [float(qs[0]), float(qs[len(qs)//2]), float(qs[-1])]
    for q in pick_qs:
        i = int(np.argmin(np.abs(qs - q)))
        y = Fq[i, :]
        ok = np.isfinite(y) & (y > 0)
        ax.plot(scales[ok], y[ok], marker="o", markersize=3, label=f"q={qs[i]:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("scale s [samples]")
    ax.set_ylabel(r"$F_q(s)$")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def plot_hq_with_surrogates(hq: pd.DataFrame, surr: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(hq["q"], hq["h"], marker="o", label="original")
    if not surr.empty:
        for kind, g in surr.groupby("surrogate_kind"):
            if {"q", "h_p05", "h_p50", "h_p95"}.issubset(g.columns):
                g = g.sort_values("q")
                ax.fill_between(g["q"].to_numpy(), g["h_p05"].to_numpy(), g["h_p95"].to_numpy(), alpha=0.18, label=f"{kind} 5–95%")
                ax.plot(g["q"], g["h_p50"], linestyle="--", linewidth=1.0, label=f"{kind} median")
    ax.set_title(title)
    ax.set_xlabel("q")
    ax.set_ylabel("h(q)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def plot_spectrum_with_surrogates(spec: pd.DataFrame, surr_summary: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(spec["alpha"], spec["f_alpha"], marker="o", label="original")
    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def quality_flag(n: int, hq_df: pd.DataFrame, scales: np.ndarray, min_segments: int) -> Tuple[str, List[str]]:
    notes: List[str] = []
    if scales.size == 0:
        return "not enough data", ["No valid scales available."]
    n_segments_smax = n // int(scales.max())
    fit_decades = float(np.log10(scales.max() / scales.min())) if scales.max() > scales.min() else 0.0
    if n < 1000:
        notes.append("Record length is below 1000 samples; MFDFA should be treated as exploratory.")
    if n_segments_smax < min_segments:
        notes.append(f"Largest scale leaves only {n_segments_smax} forward segments; require >= {min_segments}.")
    if fit_decades < 0.7:
        notes.append(f"Scaling interval spans only {fit_decades:.2f} decades; prefer >= 0.7.")
    q2_rows = hq_df.iloc[(hq_df["q"] - 2.0).abs().argsort()[:1]] if not hq_df.empty else pd.DataFrame()
    r2q2 = float(q2_rows["r2"].iloc[0]) if not q2_rows.empty and np.isfinite(q2_rows["r2"].iloc[0]) else float("nan")
    if not np.isfinite(r2q2) or r2q2 < 0.95:
        notes.append(f"The q≈2 log-log fit has R²={r2q2:.3g}; inspect the scaling range.")
    if not notes and n >= 1500 and fit_decades >= 0.8 and r2q2 >= 0.98:
        return "moderate-to-strong scaling evidence, pending surrogate interpretation", notes
    if len(notes) <= 1 and fit_decades >= 0.65:
        return "moderate scaling evidence", notes
    return "weak/exploratory scaling evidence", notes


def aggregate_surrogates(
    x: np.ndarray,
    qs: np.ndarray,
    scales: np.ndarray,
    poly_order: int,
    fit_smin: Optional[int],
    fit_smax: Optional[int],
    surrogate_kinds: List[str],
    n_surrogates: int,
    seed: int,
    iaaft_iter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    h_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []
    if n_surrogates <= 0 or not surrogate_kinds:
        return pd.DataFrame(), pd.DataFrame()
    for kind in surrogate_kinds:
        for i in range(n_surrogates):
            xs = surrogate_series(kind, x, rng, iaaft_iter=iaaft_iter)
            sc, Fq, _ = mfdfa(xs, scales, qs, poly_order=poly_order)
            hq, _ = fit_hq(sc, Fq, qs, fit_smin, fit_smax)
            spec = legendre_spectrum(qs, hq["h"].to_numpy(dtype=float))
            summ = spectrum_summary(spec)
            summ.update({"surrogate_kind": kind, "surrogate_id": i})
            summary_rows.append(summ)
            tmp = hq[["q", "h"]].copy()
            tmp["surrogate_kind"] = kind
            tmp["surrogate_id"] = i
            h_rows.append(tmp)
    h_all = pd.concat(h_rows, ignore_index=True) if h_rows else pd.DataFrame()
    if not h_all.empty:
        h_quant = (
            h_all.groupby(["surrogate_kind", "q"])["h"]
            .quantile([0.05, 0.50, 0.95])
            .unstack()
            .reset_index()
            .rename(columns={0.05: "h_p05", 0.5: "h_p50", 0.95: "h_p95"})
        )
    else:
        h_quant = pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        metric_cols = [c for c in summary.columns if c not in {"surrogate_kind", "surrogate_id"}]
        summary_quant = (
            summary.groupby("surrogate_kind")[metric_cols]
            .quantile([0.05, 0.50, 0.95])
            .unstack()
        )
        summary_quant.columns = [f"{a}_p{int(round(b*100)):02d}" for a, b in summary_quant.columns]
        summary_quant = summary_quant.reset_index()
    else:
        summary_quant = pd.DataFrame()
    return h_quant, summary_quant


def analyze_one_series(
    name: str,
    x0: Sequence[float],
    out: Path,
    qs: np.ndarray,
    smin: int,
    smax: Optional[int],
    n_scales: int,
    min_segments: int,
    poly_order: int,
    fit_smin: Optional[int],
    fit_smax: Optional[int],
    maxlag: Optional[int],
    surrogate_kinds: List[str],
    n_surrogates: int,
    seed: int,
    iaaft_iter: int,
) -> Dict:
    x = finite_array(x0)
    series_out = ensure_dir(out / name)
    n = int(len(x))
    scales = default_scales(n, smin=smin, smax=smax, n_scales=n_scales, min_segments=min_segments)
    if fit_smin is None:
        fit_smin_use = int(scales.min()) if scales.size else None
    else:
        fit_smin_use = fit_smin
    if fit_smax is None:
        fit_smax_use = int(scales.max()) if scales.size else None
    else:
        fit_smax_use = fit_smax

    pd.DataFrame({"sample": np.arange(n), "value": x}).to_csv(series_out / "cleaned_series_used.csv", index=False)

    lags, acf = autocorr_fft(x, max_lag=maxlag or min(300, n // 4))
    acf_df = pd.DataFrame({"lag": lags, "acf": acf})
    acf_df.to_csv(series_out / "acf.csv", index=False)
    plot_acf(lags, acf, series_out / "acf.png", f"ACF: {name}")

    sc, Fq, nseg = mfdfa(x, scales, qs, poly_order=poly_order)
    fq_df = pd.DataFrame(Fq, index=[f"q={q:g}" for q in qs], columns=[f"s={s}" for s in sc])
    fq_df.to_csv(series_out / "mfdfa_Fq_matrix.csv")
    scale_diag = pd.DataFrame({"scale": sc, "forward_segments": n // sc, "total_segments_used": nseg})
    scale_diag.to_csv(series_out / "scale_diagnostics.csv", index=False)

    hq, masks = fit_hq(sc, Fq, qs, fit_smin_use, fit_smax_use)
    hq.to_csv(series_out / "mfdfa_hq.csv", index=False)
    spec = legendre_spectrum(qs, hq["h"].to_numpy(dtype=float))
    spec.to_csv(series_out / "mfdfa_spectrum.csv", index=False)
    summ = spectrum_summary(spec)

    h_surr_quant, surr_summary_quant = aggregate_surrogates(
        x=x,
        qs=qs,
        scales=sc,
        poly_order=poly_order,
        fit_smin=fit_smin_use,
        fit_smax=fit_smax_use,
        surrogate_kinds=surrogate_kinds,
        n_surrogates=n_surrogates,
        seed=seed,
        iaaft_iter=iaaft_iter,
    )
    h_surr_quant.to_csv(series_out / "surrogate_hq_quantiles.csv", index=False)
    surr_summary_quant.to_csv(series_out / "surrogate_summary_quantiles.csv", index=False)

    plot_loglog_fits(sc, Fq, hq, qs, series_out / "mfdfa_loglog_fits.png", f"MFDFA fits: {name}")
    plot_hq_with_surrogates(hq, h_surr_quant, series_out / "hq_with_surrogates.png", f"h(q): {name}")
    plot_spectrum_with_surrogates(spec, surr_summary_quant, series_out / "spectrum.png", f"Spectrum: {name}")

    qflag, notes = quality_flag(n, hq, sc, min_segments=min_segments)
    out_summary = {
        "series": name,
        "n_samples": n,
        "qmin": float(qs.min()),
        "qmax": float(qs.max()),
        "n_q": int(len(qs)),
        "smin_used": int(sc.min()) if len(sc) else None,
        "smax_used": int(sc.max()) if len(sc) else None,
        "n_scales": int(len(sc)),
        "fit_smin": fit_smin_use,
        "fit_smax": fit_smax_use,
        "fit_decades": float(np.log10(sc.max() / sc.min())) if len(sc) and sc.max() > sc.min() else None,
        "min_forward_segments": int(np.min(n // sc)) if len(sc) else None,
        "acf_summary": acf_summary(lags, acf),
        "spectrum_summary_original": summ,
        "quality_flag": qflag,
        "quality_notes": notes,
        "surrogate_kinds": surrogate_kinds,
        "n_surrogates_per_kind": n_surrogates,
    }
    write_json(series_out / "short_record_quality_report.json", out_summary)
    with open(series_out / "short_record_quality_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Series: {name}\n")
        f.write(f"Quality flag: {qflag}\n")
        f.write(f"N samples: {n}\n")
        f.write(f"Scales: {out_summary['smin_used']} to {out_summary['smax_used']} samples, {out_summary['n_scales']} scales\n")
        f.write(f"Fit range: {fit_smin_use} to {fit_smax_use} samples\n")
        f.write("\nOriginal spectrum summary:\n")
        for k, v in summ.items():
            f.write(f"  {k}: {v}\n")
        if notes:
            f.write("\nWarnings / interpretation notes:\n")
            for note in notes:
                f.write(f"  - {note}\n")
        f.write("\nInterpretation rule:\n")
        f.write("  Treat h(q), Δh and Δα as meaningful only when they are stable over the scale range and exceed surrogate bands.\n")
    return out_summary


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Short-record MFDFA workflow for radon activity and geomagnetic/RPI series."
    )
    p.add_argument("--mode", choices=["radon", "geomag"], required=True)
    p.add_argument("--input", required=True, help="Input Radon Eye .txt log, geomagnetic CSV, or geomagnetic Excel file.")
    p.add_argument("--out", required=True, help="Output directory.")

    # MFDFA controls
    p.add_argument("--qmin", type=float, default=None, help="Minimum q. Defaults: -3 for radon, -4 for geomag.")
    p.add_argument("--qmax", type=float, default=None, help="Maximum q. Defaults: 3 for radon, 4 for geomag.")
    p.add_argument("--qstep", type=float, default=0.5)
    p.add_argument("--poly", type=int, default=1, help="Polynomial detrending order for MFDFA.")
    p.add_argument("--smin", type=int, default=None, help="Minimum scale in samples. Defaults: 8 radon, 8 geomag.")
    p.add_argument("--smax", type=int, default=None, help="Maximum scale in samples. Defaults to N/min_segments.")
    p.add_argument("--n-scales", type=int, default=16)
    p.add_argument("--min-segments", type=int, default=10, help="Require at least this many forward segments at the largest scale.")
    p.add_argument("--fit-smin", type=int, default=None)
    p.add_argument("--fit-smax", type=int, default=None)
    p.add_argument("--maxlag", type=int, default=None)

    # Surrogates
    p.add_argument("--n-surrogates", type=int, default=100)
    p.add_argument("--surrogates", nargs="+", default=["shuffle", "phase", "iaaft"], choices=["shuffle", "phase", "iaaft"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iaaft-iter", type=int, default=100)

    # Radon-specific
    p.add_argument("--radon-drift-window", type=int, default=168, help="Rolling-median window for radon drift removal, in hours.")

    # Geomagnetic-specific
    p.add_argument("--age-column", default="auto")
    p.add_argument("--value-column", default="auto")
    p.add_argument("--sheet-name", default=None)
    p.add_argument("--age-unit", default="kyr", choices=["kyr", "ka", "yr", "year", "years", "sample"])
    p.add_argument("--geomag-trend-window", type=int, default=201, help="Rolling median window for slow RPI anomaly.")

    p.add_argument(
        "--series",
        nargs="+",
        default=["recommended"],
        help=(
            "Series to analyze. Use 'recommended' or 'all'. "
            "Radon: log_radon, log_radon_diurnal_removed, log_radon_env_residual, "
            "log_radon_env_residual_drift_removed, d_log_radon. "
            "Geomag: rpi_level, rpi_rolling_anomaly, d_rpi."
        ),
    )
    return p.parse_args(argv)


def select_series(mode: str, available: Dict[str, np.ndarray], requested: List[str]) -> Dict[str, np.ndarray]:
    if "all" in requested:
        return available
    if "recommended" in requested:
        if mode == "radon":
            names = ["log_radon_env_residual_drift_removed", "d_log_radon"]
        else:
            names = ["rpi_rolling_anomaly", "d_rpi"]
        return {k: available[k] for k in names if k in available}
    missing = [k for k in requested if k not in available]
    if missing:
        raise ValueError(f"Requested series not available: {missing}. Available: {list(available)}")
    return {k: available[k] for k in requested}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_plot_style()
    out = ensure_dir(args.out)

    if args.mode == "radon":
        qmin = -3.0 if args.qmin is None else args.qmin
        qmax = 3.0 if args.qmax is None else args.qmax
        smin = 8 if args.smin is None else args.smin
        source_df, available, info = prepare_radon(args.input, out, drift_window=args.radon_drift_window)
        plot_series(
            source_df,
            xcol="time",
            ycols=["radon_bqm3", "temperature_c", "humidity_pct"],
            path=out / "radon_raw_overview.png",
            title="Radon, temperature, and humidity overview",
        )
    else:
        qmin = -4.0 if args.qmin is None else args.qmin
        qmax = 4.0 if args.qmax is None else args.qmax
        smin = 8 if args.smin is None else args.smin
        source_df, available, info = prepare_geomag(
            args.input,
            out,
            age_col=args.age_column,
            value_col=args.value_column,
            sheet_name=args.sheet_name,
            age_unit=args.age_unit,
            trend_window=args.geomag_trend_window,
        )
        plot_series(
            source_df,
            xcol="age",
            ycols=["value", "rolling_anomaly"],
            path=out / "geomag_overview.png",
            title="Geomagnetic/RPI overview",
        )

    qs = np.round(np.arange(qmin, qmax + 0.5 * args.qstep, args.qstep), 10)
    selected = select_series(args.mode, available, args.series)

    run_info = {
        "mode": args.mode,
        "input": str(args.input),
        "out": str(out),
        "preprocessing_info": info,
        "q_values": qs.tolist(),
        "parameters": {
            "poly": args.poly,
            "smin": smin,
            "smax": args.smax,
            "n_scales": args.n_scales,
            "min_segments": args.min_segments,
            "fit_smin": args.fit_smin,
            "fit_smax": args.fit_smax,
            "n_surrogates": args.n_surrogates,
            "surrogates": args.surrogates,
            "seed": args.seed,
        },
        "analyzed_series": list(selected.keys()),
    }
    write_json(out / "run_metadata.json", run_info)

    summaries: List[Dict] = []
    for name, x in selected.items():
        summary = analyze_one_series(
            name=name,
            x0=x,
            out=out,
            qs=qs,
            smin=smin,
            smax=args.smax,
            n_scales=args.n_scales,
            min_segments=args.min_segments,
            poly_order=args.poly,
            fit_smin=args.fit_smin,
            fit_smax=args.fit_smax,
            maxlag=args.maxlag,
            surrogate_kinds=args.surrogates,
            n_surrogates=args.n_surrogates,
            seed=args.seed,
            iaaft_iter=args.iaaft_iter,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame([
        {
            "series": s["series"],
            "n_samples": s["n_samples"],
            "smin_used": s["smin_used"],
            "smax_used": s["smax_used"],
            "fit_decades": s["fit_decades"],
            "min_forward_segments": s["min_forward_segments"],
            "quality_flag": s["quality_flag"],
            "h_q2": s["spectrum_summary_original"].get("h_q2", np.nan),
            "delta_alpha": s["spectrum_summary_original"].get("delta_alpha", np.nan),
            "delta_h_hminus3_h3": s["spectrum_summary_original"].get("delta_h_hminus3_h3", np.nan),
            "delta_h_hminus4_h4": s["spectrum_summary_original"].get("delta_h_hminus4_h4", np.nan),
        }
        for s in summaries
    ])
    summary_df.to_csv(out / "summary_short_record_results.csv", index=False)
    write_json(out / "summary_short_record_results.json", {"series": summaries})

    with open(out / "READ_ME_FIRST.txt", "w", encoding="utf-8") as f:
        f.write("Short-record geophysical MFDFA analysis\n")
        f.write("======================================\n\n")
        f.write("This workflow is intentionally conservative. For each analyzed series, inspect:\n")
        f.write("  - scale_diagnostics.csv\n")
        f.write("  - mfdfa_loglog_fits.png\n")
        f.write("  - hq_with_surrogates.png\n")
        f.write("  - surrogate_summary_quantiles.csv\n")
        f.write("  - short_record_quality_report.txt\n\n")
        f.write("Do not interpret Δα or Δh as physical multifractality unless the original series is stable over the scale range and separated from surrogate bands.\n")

    print(f"Analysis completed. Results written to: {out}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
