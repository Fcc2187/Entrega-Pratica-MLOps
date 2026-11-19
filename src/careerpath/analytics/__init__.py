"""Non-parametric latency statistics, drift metrics (PSI/KS) and shared Matplotlib figures."""

from __future__ import annotations

from .drift import (
    DEFAULT_KS_ALERT_D,
    DEFAULT_PSI_ALERT,
    DEFAULT_PSI_EPSILON,
    DEFAULT_PSI_WATCH,
    KS_ALPHA_0_05_COEFF,
    MIN_MONITORING_SAMPLES,
    MISSING_CATEGORY_LABEL,
    PRIMARY_INPUT_VAR,
    PRIMARY_OUTPUT_VAR,
    VARIABLE_SPECS,
    compute_all_variable_drifts,
    compute_binned_psi,
    compute_ecdf_diff_curve,
    compute_ks,
    compute_psi,
    normalize_categorical_value,
    select_drift_method,
)
from .io import (
    extract_inference_latencies,
    extract_numeric_column,
    extract_valid_prediction_records,
    load_jsonl_records,
)
from .latency import DEFAULT_SLA_MS, compute_latency_stats
from .plots import (
    PALETTE,
    build_ks_figure,
    build_latency_boxplot_figure,
    build_latency_overview_figure,
    build_latency_violin_figure,
    build_psi_figure,
)

__all__ = [
    "DEFAULT_KS_ALERT_D",
    "DEFAULT_PSI_ALERT",
    "DEFAULT_PSI_EPSILON",
    "DEFAULT_PSI_WATCH",
    "DEFAULT_SLA_MS",
    "KS_ALPHA_0_05_COEFF",
    "MIN_MONITORING_SAMPLES",
    "MISSING_CATEGORY_LABEL",
    "PALETTE",
    "PRIMARY_INPUT_VAR",
    "PRIMARY_OUTPUT_VAR",
    "VARIABLE_SPECS",
    "build_ks_figure",
    "build_latency_boxplot_figure",
    "build_latency_overview_figure",
    "build_latency_violin_figure",
    "build_psi_figure",
    "compute_all_variable_drifts",
    "compute_binned_psi",
    "compute_ecdf_diff_curve",
    "compute_ks",
    "compute_latency_stats",
    "compute_psi",
    "extract_inference_latencies",
    "extract_numeric_column",
    "extract_valid_prediction_records",
    "load_jsonl_records",
    "normalize_categorical_value",
    "select_drift_method",
]
