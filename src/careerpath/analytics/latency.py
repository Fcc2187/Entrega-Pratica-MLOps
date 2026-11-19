"""Non-parametric response-time statistics and SLA compliance."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from .io import to_finite_floats

DEFAULT_SLA_MS = 400.0


def compute_latency_stats(
    latencies_ms: Sequence[float],
    *,
    sla_ms: float = DEFAULT_SLA_MS,
) -> dict[str, object]:
    """Compute non-parametric response-time statistics and SLA compliance."""
    if not math.isfinite(sla_ms) or sla_ms <= 0:
        raise ValueError("sla_ms must be a finite positive number")

    raw_clean, dropped_non_finite = to_finite_floats(latencies_ms)
    clean: list[float] = []
    for val in raw_clean:
        if val < 0:
            raise ValueError("latency values must be non-negative")
        clean.append(val)

    if not clean:
        return {
            "count": 0,
            "dropped_non_finite": dropped_non_finite,
            "status": "N/A",
            "sla_threshold_ms": float(sla_ms),
            "sla_passed": None,
            "sla_margin_ms": None,
            "min_ms": None,
            "q1_ms": None,
            "median_ms": None,
            "q3_ms": None,
            "iqr_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "std_ms": None,
            "lower_fence_ms": None,
            "upper_fence_ms": None,
            "lower_whisker_ms": None,
            "upper_whisker_ms": None,
            "outlier_count": 0,
            "outliers_ms": [],
        }

    series = pd.Series(clean, dtype="float64")
    quantiles = series.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99], interpolation="linear")
    q1 = float(quantiles.loc[0.25])
    median = float(quantiles.loc[0.50])
    q3 = float(quantiles.loc[0.75])
    p90 = float(quantiles.loc[0.90])
    p95 = float(quantiles.loc[0.95])
    p99 = float(quantiles.loc[0.99])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inside = series.between(lower_fence, upper_fence)
    outliers = [float(x) for x in series.loc[~inside].tolist()]
    lower_whisker = float(series.loc[inside].min()) if inside.any() else float(series.min())
    upper_whisker = float(series.loc[inside].max()) if inside.any() else float(series.max())

    return {
        "count": len(clean),
        "dropped_non_finite": dropped_non_finite,
        "status": "OK",
        "sla_threshold_ms": float(sla_ms),
        "sla_passed": bool(p95 < sla_ms),
        "sla_margin_ms": float(sla_ms - p95),
        "min_ms": float(series.min()),
        "q1_ms": q1,
        "median_ms": median,
        "q3_ms": q3,
        "iqr_ms": float(iqr),
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": float(series.max()),
        "mean_ms": float(series.mean()),
        "std_ms": float(series.std(ddof=1)) if len(clean) > 1 else 0.0,
        "lower_fence_ms": float(lower_fence),
        "upper_fence_ms": float(upper_fence),
        "lower_whisker_ms": lower_whisker,
        "upper_whisker_ms": upper_whisker,
        "outlier_count": len(outliers),
        "outliers_ms": outliers,
    }
