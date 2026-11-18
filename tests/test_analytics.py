import json
import math
from pathlib import Path

from careerpath.analytics import (
    MISSING_CATEGORY_LABEL,
    PRIMARY_INPUT_VAR,
    PRIMARY_OUTPUT_VAR,
    build_ks_figure,
    build_latency_boxplot_figure,
    build_latency_overview_figure,
    build_latency_violin_figure,
    build_psi_figure,
    compute_all_variable_drifts,
    compute_ecdf_diff_curve,
    compute_ks,
    compute_latency_stats,
    compute_psi,
    extract_inference_latencies,
    extract_valid_prediction_records,
    select_drift_method,
)


def test_select_drift_method_follows_prompt_rules() -> None:
    assert select_drift_method("continuous") == "KS"
    assert select_drift_method("integer", integer_range=255) == "KS"
    assert select_drift_method("integer", integer_range=101) == "KS"
    assert select_drift_method("integer", integer_range=100) == "PSI"
    assert select_drift_method("integer", integer_range=83) == "PSI"
    assert select_drift_method("integer", integer_range=98) == "PSI"
    assert select_drift_method("categorical") == "PSI"


def test_compute_latency_stats_percentiles_sla_and_outliers() -> None:
    latencies = [float(i) for i in range(1, 100)] + [500.0]
    stats = compute_latency_stats(latencies, sla_ms=400.0)
    assert stats["count"] == 100
    assert stats["status"] == "OK"
    assert stats["sla_passed"] is True
    assert math.isclose(float(stats["median_ms"]), 50.5, rel_tol=1e-6)
    assert math.isclose(float(stats["p90_ms"]), 90.1, rel_tol=1e-6)
    assert math.isclose(float(stats["p95_ms"]), 95.05, rel_tol=1e-6)
    assert math.isclose(float(stats["p99_ms"]), 103.01, rel_tol=1e-6)
    assert stats["outlier_count"] == 1
    assert stats["outliers_ms"] == [500.0]

    strict = compute_latency_stats(latencies, sla_ms=90.0)
    assert strict["sla_passed"] is False


def test_compute_latency_stats_edge_cases() -> None:
    empty = compute_latency_stats([])
    assert empty["count"] == 0
    assert empty["status"] == "N/A"
    assert empty["sla_passed"] is None

    constant = compute_latency_stats([12.0, 12.0, 12.0, float("nan")])
    assert constant["count"] == 3
    assert constant["dropped_non_finite"] == 1
    assert constant["iqr_ms"] == 0.0
    assert constant["lower_whisker_ms"] == 12.0
    assert constant["upper_whisker_ms"] == 12.0
    assert constant["outlier_count"] == 0


def test_compute_psi_handles_nulls_and_zero_counts() -> None:
    identical = compute_psi(["Tech-support", None, "Sales"], ["Tech-support", None, "Sales"])
    assert identical["status"] == "OK"
    assert identical["psi"] == 0.0

    ref = ["Tech-support"] * 80 + [None] * 20
    prod = ["Tech-support"] * 40 + ["Exec-managerial"] * 50 + [None] * 10
    shifted = compute_psi(ref, prod, variable=PRIMARY_INPUT_VAR)
    assert shifted["status"] == "OK"
    assert math.isfinite(float(shifted["psi"])) and float(shifted["psi"]) > 0.25
    categories = {row["category"] for row in shifted["breakdown"]}
    assert categories == {"Tech-support", "Exec-managerial", MISSING_CATEGORY_LABEL}

    empty = compute_psi([], ["Sales"])
    assert empty["status"] == "N/A"
    assert empty["psi"] is None

    small = compute_psi(["Sales"] * 20, ["Tech-support"] * 20)
    assert small["severity"].startswith("Inconclusivo")


def test_compute_ks_detects_shift_and_handles_empty() -> None:
    same = compute_ks([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])
    assert same["status"] == "OK"
    assert same["ks_statistic"] == 0.0
    assert same["severity"].startswith("Inconclusivo")

    ref = [0.05 + 0.002 * i for i in range(100)]
    prod = [0.35 + 0.004 * i for i in range(100)]
    shifted = compute_ks(ref, prod, variable=PRIMARY_OUTPUT_VAR)
    assert shifted["status"] == "OK"
    assert float(shifted["ks_statistic"]) > 0.5
    assert float(shifted["p_value"]) < 0.001
    expected_crit = 1.358 * math.sqrt((100 + 100) / (100 * 100))
    assert math.isclose(float(shifted["critical_d_0_05"]), expected_crit, rel_tol=1e-6)
    assert shifted["severity"].startswith("Alerta")

    import numpy as np

    grid, ecdf_r, ecdf_p, delta = compute_ecdf_diff_curve(
        np.sort(np.asarray(ref)), np.sort(np.asarray(prod))
    )
    assert len(grid) == len(ecdf_r) == len(ecdf_p) == len(delta)
    assert math.isclose(float(delta.max()), float(shifted["ks_statistic"]), rel_tol=1e-9)

    empty = compute_ks([], [0.5])
    assert empty["status"] == "N/A"


def test_compute_all_variable_drifts_and_jsonl_filtering(tmp_path: Path) -> None:
    log_path = tmp_path / "inference.jsonl"
    events = [
        {
            "endpoint": "/health",
            "outcome": "success",
            "input_capture": "complete",
            "latency_ms": 1.0,
        },
        {
            "endpoint": "/predict",
            "outcome": "client_error",
            "input_capture": "complete",
            "latency_ms": 2.5,
            "input": {"age": -1},
            "output": None,
        },
        {
            "endpoint": "/predict",
            "outcome": "success",
            "input_capture": "truncated",
            "latency_ms": 5.0,
            "input": {},
            "output": {"probability_above_50k": 0.2, "income_class_proxy": "<=50K"},
        },
        {
            "endpoint": "/predict",
            "outcome": "success",
            "input_capture": "complete",
            "latency_ms": 10.0,
            "input": {
                "age": 35,
                "education": "Bachelors",
                "workclass": "Private",
                "occupation": "Tech-support",
                "marital_status": "Never-married",
                "hours_per_week": 40,
            },
            "output": {
                "income_class_proxy": "<=50K",
                "probability_above_50k": 0.14,
                "model_version": "adult-income-v1",
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    all_latencies = extract_inference_latencies(log_path)
    assert all_latencies == [2.5, 5.0, 10.0]
    valid = extract_valid_prediction_records(log_path)
    assert len(valid) == 1
    assert valid[0]["occupation"] == "Tech-support"

    drifts = compute_all_variable_drifts(valid * 20, valid * 20)
    assert drifts["primary_input"]["variable"] == "occupation"
    assert drifts["primary_output"]["variable"] == "probability_above_50k"
    assert len(drifts["supplementary"]) == 8


def test_chart_builders_render_png_for_populated_and_empty_inputs(tmp_path: Path) -> None:
    latencies = [5.0 + 0.1 * i for i in range(100)] + [85.0]
    stats = compute_latency_stats(latencies)
    psi_res = compute_psi(["Tech-support"] * 60 + [None] * 40, ["Sales"] * 70 + [None] * 30)
    ref_probs = [0.1 + 0.003 * i for i in range(100)]
    prod_probs = [0.25 + 0.004 * i for i in range(100)]
    ks_res = compute_ks(ref_probs, prod_probs)

    figs = {
        "box": build_latency_boxplot_figure(latencies, stats),
        "violin": build_latency_violin_figure(latencies, stats),
        "overview": build_latency_overview_figure(latencies, stats),
        "psi": build_psi_figure(psi_res),
        "ks": build_ks_figure(ref_probs, prod_probs, ks_res),
        "empty_box": build_latency_boxplot_figure([], compute_latency_stats([])),
        "empty_violin": build_latency_violin_figure([], compute_latency_stats([])),
        "empty_psi": build_psi_figure(compute_psi([], [])),
        "empty_ks": build_ks_figure([], [], compute_ks([], [])),
    }
    for name, fig in figs.items():
        out = tmp_path / f"{name}.png"
        fig.savefig(out, dpi=100)
        assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
