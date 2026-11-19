"""Interactive Streamlit dashboard for CareerPath AV2 Part 2 latency and drift monitoring."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import streamlit as st

from careerpath.analytics import (
    DEFAULT_KS_ALERT_D,
    DEFAULT_PSI_ALERT,
    DEFAULT_PSI_WATCH,
    DEFAULT_SLA_MS,
    PRIMARY_INPUT_VAR,
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
DEFAULT_INFERENCE_LOG = Path("logs/inference.jsonl")


def _fmt_ms(value: object) -> str:
    return f"{value:.2f} ms" if isinstance(value, int | float) else "N/A"


def _fmt_num(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, int | float) else "N/A"


def _default_evidence_dir() -> Path:
    return Path(os.environ.get("CAREERPATH_PART2_EVIDENCE_DIR", "docs/av2-p2/evidence"))


def _file_stamp(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _bundle_signature(
    evidence_dir: Path,
    inference_log: Path | None,
) -> tuple[tuple[int, int] | None, ...]:
    return (
        _file_stamp(evidence_dir / "latency-benchmark.jsonl"),
        _file_stamp(evidence_dir / "reference-responses.jsonl"),
        _file_stamp(evidence_dir / "production-responses.jsonl"),
        _file_stamp(inference_log),
    )


@st.cache_data(show_spinner=False)
def _load_bundle(
    evidence_dir_str: str,
    inference_log_str: str | None,
    file_signature: tuple[tuple[int, int] | None, ...],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[float] | None,
]:
    del file_signature  # Included in cache key so edits on disk invalidate cached JSONL reads.
    evidence_dir = Path(evidence_dir_str)
    inference_log = Path(inference_log_str) if inference_log_str else None
    bench = load_jsonl_records(evidence_dir / "latency-benchmark.jsonl")
    reference = load_jsonl_records(evidence_dir / "reference-responses.jsonl")
    production = load_jsonl_records(evidence_dir / "production-responses.jsonl")
    server_latencies: list[float] | None = None
    if inference_log is not None and inference_log.is_file():
        server_latencies = extract_inference_latencies(inference_log)
    return bench, reference, production, server_latencies


def _latency_metric_rows(stats: Mapping[str, object], sla_ms: float) -> list[tuple[str, str]]:
    sla_passed = stats.get("sla_passed")
    sla_label = "N/A" if sla_passed is None else ("APROVADO" if sla_passed else "VIOLADO")
    return [
        ("Mínimo", _fmt_ms(stats["min_ms"])),
        ("Q1 (25%)", _fmt_ms(stats["q1_ms"])),
        ("Mediana (p50)", _fmt_ms(stats["median_ms"])),
        ("Q3 (75%)", _fmt_ms(stats["q3_ms"])),
        ("IQR (Q3 - Q1)", _fmt_ms(stats["iqr_ms"])),
        ("Bigode inferior (Tukey)", _fmt_ms(stats["lower_whisker_ms"])),
        ("Bigode superior (Tukey)", _fmt_ms(stats["upper_whisker_ms"])),
        ("p90", _fmt_ms(stats["p90_ms"])),
        ("p95", _fmt_ms(stats["p95_ms"])),
        ("p99", _fmt_ms(stats["p99_ms"])),
        ("Máximo", _fmt_ms(stats["max_ms"])),
        ("Outliers além de 1,5 × IQR", str(stats["outlier_count"])),
        (f"Status SLA (p95 < {sla_ms:.0f} ms)", sla_label),
    ]


def _render_latency_tab(
    bench_records: list[dict[str, object]],
    server_latencies: list[float] | None,
    sla_ms: float,
) -> None:
    latencies = extract_numeric_column(bench_records, "client_latency_ms")
    stats = compute_latency_stats(latencies, sla_ms=sla_ms)
    if stats["count"] == 0:
        st.info("Nenhuma observação de latência disponível no arquivo de benchmark.")
        return

    sla_ok = bool(stats["sla_passed"])
    sla_label = "APROVADO" if sla_ok else "VIOLADO"

    c1, c2, c3 = st.columns(3)
    c1.metric("Pedidos analisados", str(stats["count"]))
    c2.metric("Mediana (p50)", _fmt_ms(stats["median_ms"]))
    c3.metric("p90", _fmt_ms(stats["p90_ms"]))

    c4, c5 = st.columns(2)
    c4.metric(
        f"p95 (SLA < {sla_ms:.0f} ms)",
        _fmt_ms(stats["p95_ms"]),
        delta=f"{sla_label} (+{stats['sla_margin_ms']:.2f} ms)"
        if sla_ok
        else f"{sla_label} ({stats['sla_margin_ms']:.2f} ms)",
        delta_color="normal" if sla_ok else "inverse",
    )
    c5.metric("p99", _fmt_ms(stats["p99_ms"]))

    st.subheader("Boxplot e Gráfico de Violino (POST /predict)")
    overview_fig = build_latency_overview_figure(latencies, stats, title_prefix="POST /predict")
    st.pyplot(
        overview_fig,
        width="stretch",
        alt=(
            f"Boxplot e gráfico de violino da latência de {stats['count']} pedidos: "
            f"mediana {_fmt_ms(stats['median_ms'])}, p95 {_fmt_ms(stats['p95_ms'])}, "
            f"p99 {_fmt_ms(stats['p99_ms'])}."
        ),
    )

    with st.expander("Visualizar Boxplot e Gráfico de Violino separadamente"):
        col_left, col_right = st.columns(2)
        with col_left:
            st.pyplot(
                build_latency_boxplot_figure(latencies, stats),
                width="stretch",
                alt="Boxplot isolado da latência do endpoint /predict.",
            )
        with col_right:
            st.pyplot(
                build_latency_violin_figure(latencies, stats),
                width="stretch",
                alt="Gráfico de violino isolado da latência do endpoint /predict.",
            )

    summary_rows = [
        {"Métrica": label, "Cliente HTTP": value}
        for label, value in _latency_metric_rows(stats, sla_ms)
    ]
    if server_latencies:
        server_stats = compute_latency_stats(server_latencies, sla_ms=sla_ms)
        for row, (_, server_value) in zip(
            summary_rows, _latency_metric_rows(server_stats, sla_ms), strict=True
        ):
            row["Log Servidor (ASGI)"] = server_value

    st.subheader("Tabela de estatísticas não paramétricas")
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.markdown(
        "**Interpretação:** o boxplot resume quartis, dispersão interquartil (IQR) e valores "
        "extremos além dos bigodes de Tukey (`Q3 + 1,5 × IQR`), enquanto o gráfico de violino "
        "revela a densidade completa dos tempos de resposta (concentração próxima à mediana e "
        "cauda longa capturada por p95 e p99)."
    )


def _render_status_banner(title: str, severity: str) -> None:
    text = f"**{title}:** {severity}"
    if severity.startswith("Estável"):
        st.success(text)
    elif severity.startswith("Atenção"):
        st.warning(text)
    elif severity.startswith("Alerta"):
        st.error(text)
    else:
        st.info(text)


def _render_drift_tab(
    ref_records: list[dict[str, object]],
    prod_records: list[dict[str, object]],
    *,
    psi_watch: float = DEFAULT_PSI_WATCH,
    psi_alert: float = DEFAULT_PSI_ALERT,
    ks_alert_d: float = DEFAULT_KS_ALERT_D,
) -> None:
    drift = compute_all_variable_drifts(
        ref_records,
        prod_records,
        psi_watch=psi_watch,
        psi_alert=psi_alert,
        ks_alert_d=ks_alert_d,
    )
    psi_res = drift["primary_input"]
    ks_res = drift["primary_output"]

    b1, b2 = st.columns(2)
    with b1:
        _render_status_banner(
            f"Semáforo PSI ({PRIMARY_INPUT_VAR} = {_fmt_num(psi_res['psi'])})",
            str(psi_res["severity"]),
        )
    with b2:
        _render_status_banner(
            f"Semáforo KS ({PRIMARY_OUTPUT_VAR} D = {_fmt_num(ks_res['ks_statistic'])})",
            str(ks_res["severity"]),
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"PSI Entrada ({PRIMARY_INPUT_VAR})",
        _fmt_num(psi_res["psi"]),
        delta=str(psi_res["severity"]),
        delta_color="off",
    )
    k2.metric("Amostras (Ref / Prod)", f"{psi_res['n_reference']} / {psi_res['n_production']}")
    k3.metric(
        f"KS Saída ({PRIMARY_OUTPUT_VAR})",
        _fmt_num(ks_res["ks_statistic"]),
        delta=f"D crit (α=0,05): {_fmt_num(ks_res.get('critical_d_0_05'))}",
        delta_color="off",
    )
    k4.metric(
        "p-valor KS / Deslocamento mediana",
        f"{ks_res['p_value']:.4g}" if isinstance(ks_res["p_value"], float) else "N/A",
        delta=(
            f"Δ mediana: {ks_res['median_shift']:+.4f}"
            if isinstance(ks_res["median_shift"], float)
            else None
        ),
        delta_color="off",
    )

    st.caption(
        "Regra aplicada: variáveis contínuas ou inteiras com range > 100 usam KS "
        "(probability_above_50k); variáveis categóricas ou inteiras com range <= 100 "
        "(age: 83; hours_per_week: 98; occupation: categórica com null explícito) usam PSI."
    )
    with st.expander("Fórmulas e critérios de referência (PSI / Jeffreys e KS biamostral)"):
        st.markdown(
            "- **PSI (Divergência KL simetrizada / Jeffreys):** "
            r"$\mathrm{PSI} = \sum_i (p_{\mathrm{prod},i} - p_{\mathrm{ref},i}) "
            r"\ln\!\left(\frac{p_{\mathrm{prod},i}}{p_{\mathrm{ref},i}}\right)$"
            " (limiares padrão: `< 0,10` estável; `0,10–0,25` atenção; `>= 0,25` alerta).\n"
            "- **KS biamostral:** "
            r"$D_{n,m} = \sup_x |F_{\mathrm{ref},n}(x) - F_{\mathrm{prod},m}(x)|$"
            " e valor crítico estatístico "
            r"$D_{\mathrm{crit}}(\alpha) = c(\alpha)\sqrt{\frac{n+m}{n\cdot m}}$"
            " com $c(0{,}05) \\approx 1{,}358$ (alerta operacional configurável em `D`)."
        )

    st.subheader(f"Entrada monitorada: {PRIMARY_INPUT_VAR} (PSI)")
    st.pyplot(
        build_psi_figure(psi_res),
        width="stretch",
        alt=(
            f"Comparação percentual por categoria de {PRIMARY_INPUT_VAR} "
            "entre Referência e Produção."
        ),
    )
    if psi_res["breakdown"]:
        psi_df = pd.DataFrame(psi_res["breakdown"]).rename(
            columns={
                "category": "Categoria",
                "ref_count": "Qtd Ref",
                "prod_count": "Qtd Prod",
                "ref_pct": "Ref (%)",
                "prod_pct": "Prod (%)",
                "delta_pp": "Diferença (p.p.)",
                "psi_contribution": "Contribuição PSI",
            }
        )
        st.dataframe(psi_df, width="stretch", hide_index=True)

    st.subheader(f"Saída monitorada: {PRIMARY_OUTPUT_VAR} (KS)")
    ref_probs = extract_numeric_column(ref_records, PRIMARY_OUTPUT_VAR)
    prod_probs = extract_numeric_column(prod_records, PRIMARY_OUTPUT_VAR)
    st.pyplot(
        build_ks_figure(ref_probs, prod_probs, ks_res),
        width="stretch",
        alt=f"ECDF e densidade comparada de {PRIMARY_OUTPUT_VAR} destacando distância KS.",
    )

    ks_summary = pd.DataFrame(
        [
            {
                "Conjunto": "Referência",
                "N": ks_res["n_reference"],
                "Média": _fmt_num(ks_res["ref_mean"]),
                "Mediana": _fmt_num(ks_res["ref_median"]),
                "p95": _fmt_num(ks_res["ref_p95"]),
            },
            {
                "Conjunto": "Produção",
                "N": ks_res["n_production"],
                "Média": _fmt_num(ks_res["prod_mean"]),
                "Mediana": _fmt_num(ks_res["prod_median"]),
                "p95": _fmt_num(ks_res["prod_p95"]),
            },
        ]
    )
    st.dataframe(ks_summary, width="stretch", hide_index=True)

    st.subheader("Visão de apoio: todas as variáveis do contrato")
    supp_df = pd.DataFrame(drift["supplementary"]).rename(
        columns={
            "variable": "Variável",
            "role": "Papel",
            "kind": "Tipo",
            "domain_range": "Range inteiro",
            "method": "Métrica",
            "value": "Valor",
            "p_value": "p-valor",
            "severity": "Diagnóstico",
            "is_primary": "Principal",
        }
    )
    st.dataframe(supp_df, width="stretch", hide_index=True)


def _render_data_tab(
    ref_records: list[dict[str, object]],
    prod_records: list[dict[str, object]],
    evidence_dir: Path,
) -> None:
    preview_cols = [
        "age",
        "occupation",
        "education",
        "hours_per_week",
        "workclass",
        "marital_status",
        "income_class_proxy",
        "probability_above_50k",
    ]
    st.subheader("Amostras submetidas ao servidor BentoML")
    col1, col2 = st.columns(2)
    for col, label, records in (
        (col1, "Referência", ref_records),
        (col2, "Produção", prod_records),
    ):
        with col:
            st.markdown(f"**{label} (`{len(records)}` registros válidos)**")
            frame = pd.DataFrame(records)
            cols = [c for c in preview_cols if c in frame.columns] or list(frame.columns)
            st.dataframe(frame[cols].head(25), width="stretch", hide_index=True)

    st.subheader("Reprodução local pela linha de comando")
    st.code(
        "\n".join(
            [
                (
                    "uv run --frozen bentoml serve careerpath.service:CareerPathService "
                    "--host 127.0.0.1 --port 3000 --do-not-track"
                ),
                f"uv run --frozen python scripts/part2_collect.py --evidence-dir {evidence_dir}",
                f"uv run --frozen python scripts/part2_report.py --evidence-dir {evidence_dir}",
                "uv run --frozen streamlit run scripts/dashboard.py",
            ]
        ),
        language="bash",
    )


def main() -> None:
    st.set_page_config(
        page_title="CareerPath MLOps — Monitoramento AV2 Parte 2",
        layout="wide",
    )
    st.title("CareerPath MLOps — Monitoramento Estatístico (AV2 Parte 2)")
    st.caption(
        "Latência não paramétrica (p90, p95, p99, boxplot e violino) e detecção de drift "
        "(PSI e KS)."
    )

    default_dir = _default_evidence_dir()
    with st.sidebar:
        st.header("Configurações da análise")
        source_mode = st.radio(
            "Fonte dos dados",
            options=["Evidência versionada (offline)", "Diretório customizado"],
            index=0,
        )
        if source_mode == "Evidência versionada (offline)":
            evidence_dir = default_dir
            st.text_input("Diretório de evidências", value=str(evidence_dir), disabled=True)
        else:
            raw_dir = st.text_input("Diretório de evidências", value=str(default_dir))
            evidence_dir = Path(raw_dir.strip() or str(default_dir))

        raw_log = st.text_input(
            "Log do servidor JSONL (opcional)",
            value=str(DEFAULT_INFERENCE_LOG),
        )
        inference_log = Path(raw_log.strip()) if raw_log.strip() else None
        sla_ms = float(
            st.number_input(
                "SLA alvo para p95 (ms)",
                min_value=1.0,
                max_value=10000.0,
                value=DEFAULT_SLA_MS,
                step=25.0,
            )
        )
        psi_watch = float(
            st.number_input(
                "Limiar atenção PSI",
                min_value=0.01,
                max_value=1.0,
                value=DEFAULT_PSI_WATCH,
                step=0.05,
            )
        )
        psi_alert = float(
            st.number_input(
                "Limiar alerta PSI",
                min_value=max(psi_watch + 0.01, 0.05),
                max_value=2.0,
                value=max(DEFAULT_PSI_ALERT, psi_watch + 0.05),
                step=0.05,
            )
        )
        ks_alert_d = float(
            st.number_input(
                "Limiar alerta KS (D)",
                min_value=0.01,
                max_value=1.0,
                value=DEFAULT_KS_ALERT_D,
                step=0.02,
            )
        )
        if st.button("Recarregar análise", width="stretch"):
            _load_bundle.clear()

    expected = [
        evidence_dir / "latency-benchmark.jsonl",
        evidence_dir / "reference-responses.jsonl",
        evidence_dir / "production-responses.jsonl",
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        st.info(
            "Nenhuma coleta encontrada no diretório selecionado. "
            f"Arquivos ausentes: {', '.join(str(p) for p in missing)}. "
            "Inicie o servidor BentoML e execute `uv run --frozen python scripts/part2_collect.py`."
        )
        return

    try:
        with st.spinner("Calculando estatísticas de latência e drift..."):
            signature = _bundle_signature(evidence_dir, inference_log)
            bench_records, ref_records, prod_records, server_latencies = _load_bundle(
                str(evidence_dir),
                str(inference_log) if inference_log is not None else None,
                signature,
            )
    except ValueError as exc:
        st.error(f"Não foi possível ler os arquivos de monitoramento: {exc}")
        return

    tab_latency, tab_drift, tab_data = st.tabs(
        ["Latência & SLA", "Drift (PSI & KS)", "Dados & Reprodução"]
    )
    with tab_latency:
        _render_latency_tab(bench_records, server_latencies, sla_ms)
    with tab_drift:
        _render_drift_tab(
            ref_records,
            prod_records,
            psi_watch=psi_watch,
            psi_alert=psi_alert,
            ks_alert_d=ks_alert_d,
        )
    with tab_data:
        _render_data_tab(ref_records, prod_records, evidence_dir)


if __name__ == "__main__":
    main()
