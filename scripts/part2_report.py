"""Generate non-parametric latency report, PSI/KS drift analysis, and PNG plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from careerpath.analytics import (
    DEFAULT_PSI_EPSILON,
    DEFAULT_SLA_MS,
    PRIMARY_OUTPUT_VAR,
    build_ks_figure,
    build_latency_boxplot_figure,
    build_latency_overview_figure,
    build_latency_violin_figure,
    build_psi_figure,
    compute_all_variable_drifts,
    compute_latency_stats,
    extract_inference_latencies,
    extract_numeric_column,
    load_jsonl_records,
)

DEFAULT_EVIDENCE_DIR = Path("docs/av2-p2/evidence")


def generate_part2_report(
    evidence_dir: Path,
    *,
    out_dir: Path | None = None,
    inference_log: Path | None = None,
    sla_ms: float = DEFAULT_SLA_MS,
    epsilon: float = DEFAULT_PSI_EPSILON,
) -> dict[str, object]:
    """Compute latency & drift metrics and export JSON summary plus PNG charts."""
    target_dir = out_dir or evidence_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    bench_records = load_jsonl_records(evidence_dir / "latency-benchmark.jsonl")
    ref_records = load_jsonl_records(evidence_dir / "reference-responses.jsonl")
    prod_records = load_jsonl_records(evidence_dir / "production-responses.jsonl")

    client_latencies = extract_numeric_column(bench_records, "client_latency_ms")
    client_latency_stats = compute_latency_stats(client_latencies, sla_ms=sla_ms)

    server_latency_stats: dict[str, object] | None = None
    if inference_log is not None and inference_log.is_file():
        server_latencies = extract_inference_latencies(inference_log)
        server_latency_stats = compute_latency_stats(server_latencies, sla_ms=sla_ms)

    drift = compute_all_variable_drifts(ref_records, prod_records, epsilon=epsilon)

    ref_probs = extract_numeric_column(ref_records, PRIMARY_OUTPUT_VAR)
    prod_probs = extract_numeric_column(prod_records, PRIMARY_OUTPUT_VAR)

    artifacts = {
        "latency_boxplot": target_dir / "latency_boxplot.png",
        "latency_violin": target_dir / "latency_violin.png",
        "latency_overview": target_dir / "latency_boxplot_violin.png",
        "drift_psi": target_dir / "drift_psi_occupation.png",
        "drift_ks": target_dir / "drift_ks_probability.png",
    }

    build_latency_boxplot_figure(client_latencies, client_latency_stats).savefig(
        artifacts["latency_boxplot"], dpi=150, facecolor="#ffffff"
    )
    build_latency_violin_figure(client_latencies, client_latency_stats).savefig(
        artifacts["latency_violin"], dpi=150, facecolor="#ffffff"
    )
    build_latency_overview_figure(client_latencies, client_latency_stats).savefig(
        artifacts["latency_overview"], dpi=150, facecolor="#ffffff"
    )
    build_psi_figure(drift["primary_input"]).savefig(
        artifacts["drift_psi"], dpi=150, facecolor="#ffffff"
    )
    build_ks_figure(ref_probs, prod_probs, drift["primary_output"]).savefig(
        artifacts["drift_ks"], dpi=150, facecolor="#ffffff"
    )

    report = {
        "latency_client": client_latency_stats,
        "latency_server_log": server_latency_stats,
        "drift": drift,
        "counts": {
            "benchmark_requests": len(bench_records),
            "reference_samples": len(ref_records),
            "production_samples": len(prod_records),
        },
        "charts": {name: str(path).replace("\\", "/") for name, path in artifacts.items()},
    }
    (target_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--inference-log", type=Path, default=None)
    parser.add_argument("--sla-ms", type=float, default=DEFAULT_SLA_MS)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_PSI_EPSILON)
    args = parser.parse_args(argv)

    try:
        report = generate_part2_report(
            args.evidence_dir,
            out_dir=args.out_dir,
            inference_log=args.inference_log,
            sla_ms=args.sla_ms,
            epsilon=args.epsilon,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Erro ao gerar relatório da Parte 2: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "counts": report["counts"],
                "latency_client": {
                    k: report["latency_client"][k]
                    for k in ("count", "median_ms", "p90_ms", "p95_ms", "p99_ms", "sla_passed")
                },
                "primary_input_psi": {
                    k: report["drift"]["primary_input"][k]
                    for k in ("variable", "psi", "severity")
                },
                "primary_output_ks": {
                    k: report["drift"]["primary_output"][k]
                    for k in ("variable", "ks_statistic", "p_value", "severity")
                },
                "charts": report["charts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
