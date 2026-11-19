"""Drift metric selection, Population Stability Index (PSI) and Kolmogorov-Smirnov (KS)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .io import to_finite_floats

DEFAULT_PSI_EPSILON = 1e-6
DEFAULT_PSI_WATCH = 0.10
DEFAULT_PSI_ALERT = 0.25
DEFAULT_KS_ALERT_D = 0.10
MIN_MONITORING_SAMPLES = 100
KS_ALPHA_0_05_COEFF = 1.358
MISSING_CATEGORY_LABEL = "(ausente)"
PRIMARY_INPUT_VAR = "occupation"
PRIMARY_OUTPUT_VAR = "probability_above_50k"

VARIABLE_SPECS: dict[str, dict[str, object]] = {
    "age": {"role": "input", "kind": "integer", "domain_range": 100 - 17},
    "hours_per_week": {"role": "input", "kind": "integer", "domain_range": 99 - 1},
    "education": {"role": "input", "kind": "categorical", "domain_range": None},
    "workclass": {"role": "input", "kind": "categorical", "domain_range": None},
    "occupation": {"role": "input", "kind": "categorical", "domain_range": None},
    "marital_status": {"role": "input", "kind": "categorical", "domain_range": None},
    "probability_above_50k": {"role": "output", "kind": "continuous", "domain_range": None},
    "income_class_proxy": {"role": "output", "kind": "categorical", "domain_range": None},
}


def select_drift_method(
    kind: str,
    *,
    integer_range: int | None = None,
) -> Literal["KS", "PSI"]:
    """Choose KS for continuous/wide-integer variables (>100) and PSI otherwise."""
    if kind == "continuous":
        return "KS"
    if kind == "integer" and integer_range is not None and integer_range > 100:
        return "KS"
    return "PSI"


def normalize_categorical_value(
    value: object,
    *,
    missing_label: str = MISSING_CATEGORY_LABEL,
) -> str:
    """Normalize nullable categorical values into an explicit string bucket."""
    if value is None:
        return missing_label
    if isinstance(value, float) and math.isnan(value):
        return missing_label
    if pd.isna(value):
        return missing_label
    text = str(value).strip()
    return text if text else missing_label


def _psi_severity(
    psi_value: float,
    *,
    n_ref: int,
    n_prod: int,
    min_samples: int = MIN_MONITORING_SAMPLES,
    watch_threshold: float = DEFAULT_PSI_WATCH,
    alert_threshold: float = DEFAULT_PSI_ALERT,
) -> str:
    if n_ref < min_samples or n_prod < min_samples:
        return f"Inconclusivo (< {min_samples} amostras)"
    if psi_value < watch_threshold:
        return f"Estável (< {watch_threshold:.2f})"
    if psi_value < alert_threshold:
        return f"Atenção ({watch_threshold:.2f} a {alert_threshold:.2f})"
    return f"Alerta (>= {alert_threshold:.2f})"


def compute_psi(
    reference: Sequence[object],
    production: Sequence[object],
    *,
    variable: str = PRIMARY_INPUT_VAR,
    epsilon: float = DEFAULT_PSI_EPSILON,
    missing_label: str = MISSING_CATEGORY_LABEL,
    min_samples: int = MIN_MONITORING_SAMPLES,
    watch_threshold: float = DEFAULT_PSI_WATCH,
    alert_threshold: float = DEFAULT_PSI_ALERT,
) -> dict[str, object]:
    """Compute Population Stability Index (PSI) with explicit null bucket and smoothing."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")

    n_ref = len(reference)
    n_prod = len(production)
    if n_ref == 0 or n_prod == 0:
        return {
            "variable": variable,
            "method": "PSI",
            "status": "N/A",
            "psi": None,
            "severity": "N/A",
            "n_reference": n_ref,
            "n_production": n_prod,
            "epsilon": float(epsilon),
            "category_count": 0,
            "breakdown": [],
        }

    ref_norm = pd.Series(
        [normalize_categorical_value(v, missing_label=missing_label) for v in reference],
        dtype="object",
    )
    prod_norm = pd.Series(
        [normalize_categorical_value(v, missing_label=missing_label) for v in production],
        dtype="object",
    )
    ref_counts = ref_norm.value_counts().to_dict()
    prod_counts = prod_norm.value_counts().to_dict()

    categories = sorted((set(ref_counts) | set(prod_counts)) - {missing_label})
    if missing_label in ref_counts or missing_label in prod_counts:
        categories.append(missing_label)

    denom = 1.0 + len(categories) * epsilon
    breakdown: list[dict[str, object]] = []
    total_psi = 0.0
    for category in categories:
        c_ref = int(ref_counts.get(category, 0))
        c_prod = int(prod_counts.get(category, 0))
        p_ref_raw = c_ref / n_ref
        p_prod_raw = c_prod / n_prod
        p_ref_smooth = (p_ref_raw + epsilon) / denom
        p_prod_smooth = (p_prod_raw + epsilon) / denom
        contribution = (p_prod_smooth - p_ref_smooth) * math.log(p_prod_smooth / p_ref_smooth)
        total_psi += contribution
        breakdown.append(
            {
                "category": category,
                "ref_count": c_ref,
                "prod_count": c_prod,
                "ref_pct": p_ref_raw * 100.0,
                "prod_pct": p_prod_raw * 100.0,
                "delta_pp": (p_prod_raw - p_ref_raw) * 100.0,
                "psi_contribution": float(contribution),
            }
        )

    breakdown.sort(key=lambda row: float(row["psi_contribution"]), reverse=True)
    return {
        "variable": variable,
        "method": "PSI",
        "status": "OK",
        "psi": float(total_psi),
        "severity": _psi_severity(
            float(total_psi),
            n_ref=n_ref,
            n_prod=n_prod,
            min_samples=min_samples,
            watch_threshold=watch_threshold,
            alert_threshold=alert_threshold,
        ),
        "n_reference": n_ref,
        "n_production": n_prod,
        "epsilon": float(epsilon),
        "category_count": len(categories),
        "breakdown": breakdown,
    }


def compute_binned_psi(
    reference: Sequence[float | int],
    production: Sequence[float | int],
    *,
    variable: str,
    bins: int = 10,
    epsilon: float = DEFAULT_PSI_EPSILON,
    min_samples: int = MIN_MONITORING_SAMPLES,
    watch_threshold: float = DEFAULT_PSI_WATCH,
    alert_threshold: float = DEFAULT_PSI_ALERT,
) -> dict[str, object]:
    """Discretize bounded integer/numeric variables into reference bins and compute PSI."""
    ref_clean, _ = to_finite_floats(reference)
    prod_clean, _ = to_finite_floats(production)
    if not ref_clean or not prod_clean:
        return compute_psi([], [], variable=variable, epsilon=epsilon)

    ref_arr = np.asarray(ref_clean, dtype=float)
    quantiles = np.linspace(0.0, 1.0, max(2, bins) + 1)
    edges = np.unique(np.quantile(ref_arr, quantiles))
    if len(edges) < 2:
        val = int(round(ref_arr[0]))
        ref_labels = [str(val) for _ in ref_clean]
        prod_labels = [
            str(int(round(v))) if v == ref_arr[0] else ("<ref" if v < ref_arr[0] else ">ref")
            for v in prod_clean
        ]
        return compute_psi(ref_labels, prod_labels, variable=variable, epsilon=epsilon)

    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = []
    for i in range(len(edges) - 1):
        if i == 0:
            labels.append(f"<= {edges[i + 1]:.0f}")
        elif i == len(edges) - 2:
            labels.append(f"> {edges[i]:.0f}")
        else:
            labels.append(f"({edges[i]:.0f}, {edges[i + 1]:.0f}]")
    ref_binned = pd.cut(ref_clean, bins=edges, labels=labels, include_lowest=True).astype(str)
    prod_binned = pd.cut(prod_clean, bins=edges, labels=labels, include_lowest=True).astype(str)
    return compute_psi(
        ref_binned.tolist(),
        prod_binned.tolist(),
        variable=variable,
        epsilon=epsilon,
        min_samples=min_samples,
        watch_threshold=watch_threshold,
        alert_threshold=alert_threshold,
    )


def _ks_severity(
    ks_stat: float,
    *,
    n_ref: int,
    n_prod: int,
    min_samples: int = MIN_MONITORING_SAMPLES,
    alert_threshold: float = DEFAULT_KS_ALERT_D,
) -> str:
    if n_ref < min_samples or n_prod < min_samples:
        return f"Inconclusivo (< {min_samples} amostras)"
    if ks_stat < alert_threshold:
        return f"Estável (< {alert_threshold:.2f})"
    if ks_stat < 0.20:
        return f"Atenção ({alert_threshold:.2f} a 0.20)"
    return "Alerta (>= 0.20)"


def compute_ecdf_diff_curve(
    ref_arr: np.ndarray,
    prod_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sorted joint grid, ECDFs, and absolute pointwise difference |F_ref - F_prod|."""
    grid = np.unique(np.concatenate([ref_arr, prod_arr]))
    ecdf_ref = np.searchsorted(ref_arr, grid, side="right") / len(ref_arr)
    ecdf_prod = np.searchsorted(prod_arr, grid, side="right") / len(prod_arr)
    delta = np.abs(ecdf_ref - ecdf_prod)
    return grid, ecdf_ref, ecdf_prod, delta


def compute_ks(
    reference: Sequence[float],
    production: Sequence[float],
    *,
    variable: str = PRIMARY_OUTPUT_VAR,
    min_samples: int = MIN_MONITORING_SAMPLES,
    alert_threshold: float = DEFAULT_KS_ALERT_D,
) -> dict[str, object]:
    """Compute two-sample Kolmogorov-Smirnov distance, critical threshold, and summary."""
    ref_clean, _ = to_finite_floats(reference)
    prod_clean, _ = to_finite_floats(production)

    if not ref_clean or not prod_clean:
        return {
            "variable": variable,
            "method": "KS",
            "status": "N/A",
            "ks_statistic": None,
            "p_value": None,
            "critical_d_0_05": None,
            "n_eff": None,
            "significant_at_0_05": None,
            "severity": "N/A",
            "n_reference": len(ref_clean),
            "n_production": len(prod_clean),
            "max_diff_at": None,
            "ref_mean": None,
            "ref_median": None,
            "ref_p95": None,
            "prod_mean": None,
            "prod_median": None,
            "prod_p95": None,
            "median_shift": None,
        }

    ref_arr = np.sort(np.asarray(ref_clean, dtype=float))
    prod_arr = np.sort(np.asarray(prod_clean, dtype=float))
    ks_result = scipy_stats.ks_2samp(ref_arr, prod_arr, method="auto")

    grid, ecdf_ref, ecdf_prod, delta = compute_ecdf_diff_curve(ref_arr, prod_arr)
    max_idx = int(np.argmax(delta))

    n_ref = len(ref_clean)
    n_prod = len(prod_clean)
    n_eff = (n_ref * n_prod) / (n_ref + n_prod)
    critical_d_0_05 = KS_ALPHA_0_05_COEFF * math.sqrt((n_ref + n_prod) / (n_ref * n_prod))

    ref_med = float(np.median(ref_arr))
    prod_med = float(np.median(prod_arr))
    ks_stat = float(ks_result.statistic)
    p_val = float(ks_result.pvalue)

    return {
        "variable": variable,
        "method": "KS",
        "status": "OK",
        "ks_statistic": ks_stat,
        "p_value": p_val,
        "critical_d_0_05": float(critical_d_0_05),
        "n_eff": float(n_eff),
        "significant_at_0_05": bool(p_val < 0.05 or ks_stat > critical_d_0_05),
        "severity": _ks_severity(
            ks_stat,
            n_ref=n_ref,
            n_prod=n_prod,
            min_samples=min_samples,
            alert_threshold=alert_threshold,
        ),
        "n_reference": n_ref,
        "n_production": n_prod,
        "max_diff_at": float(grid[max_idx]),
        "max_diff_ecdf_ref": float(ecdf_ref[max_idx]),
        "max_diff_ecdf_prod": float(ecdf_prod[max_idx]),
        "ref_mean": float(np.mean(ref_arr)),
        "ref_median": ref_med,
        "ref_p95": float(np.quantile(ref_arr, 0.95)),
        "prod_mean": float(np.mean(prod_arr)),
        "prod_median": prod_med,
        "prod_p95": float(np.quantile(prod_arr, 0.95)),
        "median_shift": float(prod_med - ref_med),
    }


def compute_all_variable_drifts(
    ref_records: Sequence[Mapping[str, object]],
    prod_records: Sequence[Mapping[str, object]],
    *,
    epsilon: float = DEFAULT_PSI_EPSILON,
    min_samples: int = MIN_MONITORING_SAMPLES,
    psi_watch: float = DEFAULT_PSI_WATCH,
    psi_alert: float = DEFAULT_PSI_ALERT,
    ks_alert_d: float = DEFAULT_KS_ALERT_D,
) -> dict[str, object]:
    """Compute primary drift metrics plus supporting metrics for all logged fields."""
    ref_df = pd.DataFrame(list(ref_records))
    prod_df = pd.DataFrame(list(prod_records))

    def _col(frame: pd.DataFrame, col: str) -> list[object]:
        if frame.empty or col not in frame.columns:
            return []
        return frame[col].tolist()

    primary_input = compute_psi(
        _col(ref_df, PRIMARY_INPUT_VAR),
        _col(prod_df, PRIMARY_INPUT_VAR),
        variable=PRIMARY_INPUT_VAR,
        epsilon=epsilon,
        min_samples=min_samples,
        watch_threshold=psi_watch,
        alert_threshold=psi_alert,
    )
    primary_output = compute_ks(
        _col(ref_df, PRIMARY_OUTPUT_VAR),
        _col(prod_df, PRIMARY_OUTPUT_VAR),
        variable=PRIMARY_OUTPUT_VAR,
        min_samples=min_samples,
        alert_threshold=ks_alert_d,
    )

    supplementary: list[dict[str, object]] = []
    for name, spec in VARIABLE_SPECS.items():
        method = select_drift_method(
            str(spec["kind"]),
            integer_range=spec["domain_range"] if isinstance(spec["domain_range"], int) else None,
        )
        ref_vals = _col(ref_df, name)
        prod_vals = _col(prod_df, name)
        if spec["kind"] == "continuous":
            res = compute_ks(
                ref_vals,
                prod_vals,
                variable=name,
                min_samples=min_samples,
                alert_threshold=ks_alert_d,
            )
            metric_value = res["ks_statistic"]
        elif spec["kind"] == "integer":
            res = compute_binned_psi(
                ref_vals,
                prod_vals,
                variable=name,
                epsilon=epsilon,
                min_samples=min_samples,
                watch_threshold=psi_watch,
                alert_threshold=psi_alert,
            )
            metric_value = res["psi"]
        else:
            res = compute_psi(
                ref_vals,
                prod_vals,
                variable=name,
                epsilon=epsilon,
                min_samples=min_samples,
                watch_threshold=psi_watch,
                alert_threshold=psi_alert,
            )
            metric_value = res["psi"]

        supplementary.append(
            {
                "variable": name,
                "role": spec["role"],
                "kind": spec["kind"],
                "domain_range": spec["domain_range"],
                "method": method,
                "value": metric_value,
                "p_value": res.get("p_value"),
                "severity": res["severity"],
                "is_primary": name in {PRIMARY_INPUT_VAR, PRIMARY_OUTPUT_VAR},
            }
        )

    return {
        "primary_input": primary_input,
        "primary_output": primary_output,
        "supplementary": supplementary,
    }
