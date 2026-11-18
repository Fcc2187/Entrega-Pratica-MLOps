"""Non-parametric latency statistics, drift metrics (PSI/KS) and shared Matplotlib figures."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DEFAULT_SLA_MS = 400.0
DEFAULT_PSI_EPSILON = 1e-6
DEFAULT_PSI_WATCH = 0.10
DEFAULT_PSI_ALERT = 0.25
DEFAULT_KS_ALERT_D = 0.10
MIN_MONITORING_SAMPLES = 100
KS_ALPHA_0_05_COEFF = 1.358
MISSING_CATEGORY_LABEL = "(ausente)"
PRIMARY_INPUT_VAR = "occupation"
PRIMARY_OUTPUT_VAR = "probability_above_50k"

PALETTE = {
    "ink": "#0f172a",
    "muted": "#475569",
    "grid": "#cbd5e1",
    "bg": "#f8fafc",
    "white": "#ffffff",
    "blue": "#3b82f6",
    "emerald": "#10b981",
    "violet": "#8b5cf6",
    "orange": "#f97316",
    "red": "#ef4444",
}

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


def _configure_matplotlib_env() -> None:
    """Ensure Matplotlib writes cache to a writable path before importing pyplot/seaborn."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    default_dir = Path.home() / ".matplotlib"
    try:
        default_dir.mkdir(parents=True, exist_ok=True)
        probe = default_dir / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "careerpath-mplconfig"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback)


def _import_plotting():
    _configure_matplotlib_env()
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    matplotlib.use("Agg", force=False)
    import seaborn as sns

    return Figure, FigureCanvasAgg, sns


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


def _to_finite_floats(values: Sequence[object]) -> tuple[list[float], int]:
    """Convert numeric sequences to finite floats, counting skipped non-finite/missing items."""
    clean: list[float] = []
    skipped = 0
    for raw in values:
        if raw is None or isinstance(raw, bool) or pd.isna(raw):
            skipped += 1
            continue
        val = float(raw)
        if not math.isfinite(val):
            skipped += 1
            continue
        clean.append(val)
    return clean, skipped


def compute_latency_stats(
    latencies_ms: Sequence[float],
    *,
    sla_ms: float = DEFAULT_SLA_MS,
) -> dict[str, object]:
    """Compute non-parametric response-time statistics and SLA compliance."""
    if not math.isfinite(sla_ms) or sla_ms <= 0:
        raise ValueError("sla_ms must be a finite positive number")

    raw_clean, dropped_non_finite = _to_finite_floats(latencies_ms)
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
    ref_clean, _ = _to_finite_floats(reference)
    prod_clean, _ = _to_finite_floats(production)
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
    ref_clean, _ = _to_finite_floats(reference)
    prod_clean, _ = _to_finite_floats(production)

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


def load_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Read UTF-8 JSONL file into a list of dicts, failing clearly if malformed."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{lineno}")
        records.append(obj)
    return records


def extract_inference_latencies(
    jsonl_path: Path,
    *,
    endpoint: str = "/predict",
    only_success: bool = False,
) -> list[float]:
    """Extract non-negative latency_ms values from monitoring JSONL events."""
    events = load_jsonl_records(jsonl_path)
    latencies: list[float] = []
    for event in events:
        if event.get("endpoint") != endpoint:
            continue
        if only_success and event.get("outcome") != "success":
            continue
        raw = event.get("latency_ms")
        if isinstance(raw, int | float) and not isinstance(raw, bool) and math.isfinite(raw):
            latencies.append(float(raw))
    return latencies


def extract_valid_prediction_records(
    jsonl_path: Path,
    *,
    endpoint: str = "/predict",
) -> list[dict[str, object]]:
    """Extract flat input+output records from complete successful inference events."""
    events = load_jsonl_records(jsonl_path)
    rows: list[dict[str, object]] = []
    for event in events:
        if event.get("endpoint") != endpoint:
            continue
        if event.get("outcome") != "success" or event.get("input_capture") != "complete":
            continue
        inputs = event.get("input")
        outputs = event.get("output")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        rows.append(
            {
                **inputs,
                **outputs,
                "latency_ms": event.get("latency_ms"),
                "timestamp": event.get("timestamp"),
                "request_id": event.get("request_id"),
            }
        )
    return rows


def _empty_figure(title: str, message: str) -> object:
    Figure, FigureCanvasAgg, _ = _import_plotting()
    fig = Figure(figsize=(8, 4.2), dpi=150)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(PALETTE["white"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(PALETTE["bg"])
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["muted"],
        transform=ax.transAxes,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", color=PALETTE["ink"], pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _style_axis(ax: object, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["ink"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["ink"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.45, color=PALETTE["grid"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_boxplot_panel(
    ax: object,
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
    *,
    title: str,
) -> None:
    box = ax.boxplot(
        list(latencies_ms),
        orientation="horizontal",
        widths=0.42,
        patch_artist=True,
        boxprops={
            "facecolor": PALETTE["blue"],
            "edgecolor": PALETTE["ink"],
            "alpha": 0.72,
            "linewidth": 1.3,
        },
        medianprops={"color": PALETTE["orange"], "linewidth": 2.2},
        whiskerprops={"color": PALETTE["ink"], "linewidth": 1.2},
        capprops={"color": PALETTE["ink"], "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": PALETTE["red"],
            "markeredgecolor": PALETTE["ink"],
            "alpha": 0.7,
        },
    )
    assert box is not None
    p95 = stats.get("p95_ms")
    p99 = stats.get("p99_ms")
    if isinstance(p95, float):
        ax.axvline(
            p95,
            color=PALETTE["emerald"],
            linestyle="--",
            linewidth=1.6,
            label=f"p95 = {p95:.2f} ms",
        )
    if isinstance(p99, float):
        ax.axvline(
            p99,
            color=PALETTE["violet"],
            linestyle=":",
            linewidth=1.6,
            label=f"p99 = {p99:.2f} ms",
        )
    sla = stats.get("sla_threshold_ms")
    sla_passed = stats.get("sla_passed")
    if isinstance(sla, float) and isinstance(sla_passed, bool):
        status = "APROVADO" if sla_passed else "VIOLADO"
        ax.text(
            0.98,
            0.94,
            f"SLA p95 < {sla:.0f} ms: {status}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["emerald"] if sla_passed else PALETTE["red"],
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": PALETTE["white"],
                "edgecolor": PALETTE["grid"],
            },
        )
    ax.set_yticks([])
    _style_axis(ax, title=title, xlabel="Tempo de resposta (ms)")
    ax.legend(loc="lower right", frameon=True, facecolor=PALETTE["white"], fontsize=8.5)


def _draw_violin_panel(
    ax: object,
    sns: object,
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
    *,
    title: str,
) -> None:
    series = pd.Series(latencies_ms, dtype="float64")
    sns.violinplot(
        x=series,
        ax=ax,
        color=PALETTE["violet"],
        inner="quartile",
        cut=0,
        linewidth=1.2,
    )
    for collection in ax.collections:
        collection.set_alpha(0.55)
    median = stats.get("median_ms")
    p95 = stats.get("p95_ms")
    if isinstance(median, float):
        ax.axvline(
            median,
            color=PALETTE["orange"],
            linestyle="-",
            linewidth=1.6,
            label=f"Mediana = {median:.2f} ms",
        )
    if isinstance(p95, float):
        ax.axvline(
            p95,
            color=PALETTE["emerald"],
            linestyle="--",
            linewidth=1.6,
            label=f"p95 = {p95:.2f} ms",
        )
    ax.set_yticks([])
    _style_axis(ax, title=title, xlabel="Tempo de resposta (ms)")
    ax.legend(loc="lower right", frameon=True, facecolor=PALETTE["white"], fontsize=8.5)


def build_latency_boxplot_figure(
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
    *,
    title_prefix: str = "POST /predict",
) -> object:
    """Build a standalone boxplot figure for endpoint response times."""
    if not latencies_ms or stats.get("count", 0) == 0:
        return _empty_figure(f"Boxplot — {title_prefix}", "Sem observações de latência.")
    Figure, FigureCanvasAgg, _ = _import_plotting()
    fig = Figure(figsize=(8.4, 4.0), dpi=150)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(PALETTE["white"])
    ax = fig.add_subplot(111)
    _draw_boxplot_panel(
        ax,
        latencies_ms,
        stats,
        title=f"Boxplot da latência — {title_prefix} (n={stats['count']})",
    )
    fig.tight_layout()
    return fig


def build_latency_violin_figure(
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
    *,
    title_prefix: str = "POST /predict",
) -> object:
    """Build a standalone violin plot figure for endpoint response times."""
    if not latencies_ms or stats.get("count", 0) == 0:
        return _empty_figure(f"Violino — {title_prefix}", "Sem observações de latência.")
    Figure, FigureCanvasAgg, sns = _import_plotting()
    fig = Figure(figsize=(8.4, 4.0), dpi=150)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(PALETTE["white"])
    ax = fig.add_subplot(111)
    _draw_violin_panel(
        ax,
        sns,
        latencies_ms,
        stats,
        title=f"Gráfico de violino da latência — {title_prefix} (n={stats['count']})",
    )
    fig.tight_layout()
    return fig


def build_latency_overview_figure(
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
    *,
    title_prefix: str = "POST /predict",
) -> object:
    """Build a combined boxplot + violin figure sharing the same horizontal latency scale."""
    if not latencies_ms or stats.get("count", 0) == 0:
        return _empty_figure(f"Latência — {title_prefix}", "Sem observações de latência.")
    Figure, FigureCanvasAgg, sns = _import_plotting()
    fig = Figure(figsize=(11.5, 4.5), dpi=150)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(PALETTE["white"])
    ax_box = fig.add_subplot(121)
    ax_violin = fig.add_subplot(122, sharex=ax_box)
    _draw_boxplot_panel(
        ax_box,
        latencies_ms,
        stats,
        title="Boxplot (quartis, bigodes Tukey e outliers)",
    )
    _draw_violin_panel(ax_violin, sns, latencies_ms, stats, title="Violino (densidade e quartis)")
    fig.suptitle(
        f"Distribuição do tempo de resposta — {title_prefix} (n={stats['count']})",
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.tight_layout()
    return fig


def build_psi_figure(psi_result: Mapping[str, object]) -> object:
    """Build grouped horizontal bar chart comparing Reference vs Production proportions."""
    breakdown = psi_result.get("breakdown")
    variable = str(psi_result.get("variable", PRIMARY_INPUT_VAR))
    if not isinstance(breakdown, list) or not breakdown:
        return _empty_figure(f"PSI — {variable}", "Dados insuficientes para calcular PSI.")

    Figure, FigureCanvasAgg, _ = _import_plotting()
    ordered = sorted(breakdown, key=lambda r: float(r["ref_pct"]) + float(r["prod_pct"]))
    categories = [str(r["category"]) for r in ordered]
    ref_pct = [float(r["ref_pct"]) for r in ordered]
    prod_pct = [float(r["prod_pct"]) for r in ordered]

    fig = Figure(figsize=(9.5, max(4.5, 0.36 * len(categories) + 1.8)), dpi=150)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(PALETTE["white"])
    ax = fig.add_subplot(111)

    y = np.arange(len(categories))
    height = 0.38
    ax.barh(y + height / 2, ref_pct, height=height, color=PALETTE["blue"], label="Referência (%)")
    ax.barh(y - height / 2, prod_pct, height=height, color=PALETTE["orange"], label="Produção (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    psi_val = psi_result.get("psi")
    severity = psi_result.get("severity", "")
    psi_label = f"{psi_val:.4f}" if isinstance(psi_val, float) else "N/A"
    _style_axis(
        ax,
        title=f"Comparação de frequências ({variable}) — PSI = {psi_label} [{severity}]",
        xlabel="Proporção (%)",
    )
    ax.legend(loc="lower right", frameon=True, facecolor=PALETTE["white"], fontsize=9)
    fig.tight_layout()
    return fig


def build_ks_figure(
    ref_values: Sequence[float],
    prod_values: Sequence[float],
    ks_result: Mapping[str, object],
) -> object:
    """Build ECDF (highlighting max gap D) + overlapping histogram for continuous output."""
    variable = str(ks_result.get("variable", PRIMARY_OUTPUT_VAR))
    ref_clean, _ = _to_finite_floats(ref_values)
    prod_clean, _ = _to_finite_floats(prod_values)
    ref_arr = np.sort(np.asarray(ref_clean, dtype=float))
    prod_arr = np.sort(np.asarray(prod_clean, dtype=float))
    if len(ref_arr) == 0 or len(prod_arr) == 0:
        return _empty_figure(f"KS — {variable}", "Dados insuficientes para calcular KS.")

    Figure, FigureCanvasAgg, _ = _import_plotting()
    fig = Figure(figsize=(14.5, 4.6), dpi=150)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(PALETTE["white"])

    grid, ecdf_ref, ecdf_prod, delta = compute_ecdf_diff_curve(ref_arr, prod_arr)

    ax_ecdf = fig.add_subplot(131)
    ax_ecdf.step(
        grid,
        ecdf_ref,
        where="post",
        color=PALETTE["blue"],
        linewidth=2.0,
        label="Referência",
    )
    ax_ecdf.step(
        grid,
        ecdf_prod,
        where="post",
        color=PALETTE["orange"],
        linewidth=2.0,
        label="Produção",
    )

    max_x = ks_result.get("max_diff_at")
    y_ref = ks_result.get("max_diff_ecdf_ref")
    y_prod = ks_result.get("max_diff_ecdf_prod")
    ks_stat = ks_result.get("ks_statistic")
    if isinstance(max_x, float) and isinstance(y_ref, float) and isinstance(y_prod, float):
        ax_ecdf.vlines(
            max_x,
            min(y_ref, y_prod),
            max(y_ref, y_prod),
            colors=PALETTE["red"],
            linestyles="--",
            linewidth=2.0,
            label=f"KS D = {ks_stat:.4f} em {max_x:.3f}",
        )
    _style_axis(
        ax_ecdf,
        title="ECDF (Referência vs Produção)",
        xlabel=variable,
        ylabel="Probabilidade acumulada",
    )
    ax_ecdf.legend(loc="lower right", frameon=True, facecolor=PALETTE["white"], fontsize=8)

    ax_delta = fig.add_subplot(132)
    ax_delta.step(
        grid,
        delta,
        where="post",
        color=PALETTE["violet"],
        linewidth=1.8,
        label=r"$\delta(x) = |F_{\mathrm{ref}} - F_{\mathrm{prod}}|$",
    )
    if isinstance(ks_stat, float):
        ax_delta.axhline(
            ks_stat,
            color=PALETTE["red"],
            linestyle="--",
            linewidth=1.5,
            label=f"D = {ks_stat:.4f}",
        )
        if isinstance(max_x, float):
            ax_delta.scatter([max_x], [ks_stat], color=PALETTE["red"], zorder=5, s=36)
    crit_d = ks_result.get("critical_d_0_05")
    if isinstance(crit_d, float):
        ax_delta.axhline(
            crit_d,
            color=PALETTE["emerald"],
            linestyle=":",
            linewidth=1.5,
            label=f"D crit (α=0,05) = {crit_d:.4f}",
        )
    _style_axis(
        ax_delta,
        title="Diferença absoluta δ(x)",
        xlabel=variable,
        ylabel="|F_ref(x) - F_prod(x)|",
    )
    top_delta = max(float(delta.max()), float(crit_d) if isinstance(crit_d, float) else 0.0, 0.04)
    ax_delta.set_ylim(0.0, top_delta * 1.32)
    ax_delta.legend(loc="upper right", frameon=True, facecolor=PALETTE["white"], fontsize=8)

    ax_hist = fig.add_subplot(133)
    bins = np.linspace(0.0, 1.0, 21)
    ax_hist.hist(
        ref_arr,
        bins=bins,
        density=True,
        alpha=0.55,
        color=PALETTE["blue"],
        edgecolor=PALETTE["white"],
        label="Referência",
    )
    ax_hist.hist(
        prod_arr,
        bins=bins,
        density=True,
        alpha=0.55,
        color=PALETTE["orange"],
        edgecolor=PALETTE["white"],
        label="Produção",
    )
    p_val = ks_result.get("p_value")
    p_label = f"{p_val:.4g}" if isinstance(p_val, float) else "N/A"
    _style_axis(
        ax_hist,
        title=f"Densidade comparada (p-valor = {p_label})",
        xlabel=variable,
        ylabel="Densidade",
    )
    ax_hist.legend(loc="upper right", frameon=True, facecolor=PALETTE["white"], fontsize=8.5)

    severity = ks_result.get("severity", "")
    ks_label = f"{ks_stat:.4f}" if isinstance(ks_stat, float) else "N/A"
    fig.suptitle(
        f"Teste de Kolmogorov-Smirnov ({variable}) — D = {ks_label} [{severity}]",
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.tight_layout()
    return fig
