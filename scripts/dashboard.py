"""Interactive Streamlit observability dashboard for CareerPath AV2 Part 2."""

from __future__ import annotations

import html
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLES_PATH = PROJECT_ROOT / "assets" / "styles.css"
DEFAULT_EVIDENCE_DIR = Path("docs/av2-p2/evidence")
DEFAULT_INFERENCE_LOG = Path("logs/inference.jsonl")

_ICON_BRAND = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">'
    '<rect x="1.5" y="9" width="3" height="5.5" rx="0.6" fill="currentColor"/>'
    '<rect x="6.5" y="5.5" width="3" height="9" rx="0.6" fill="currentColor"/>'
    '<rect x="11.5" y="1.5" width="3" height="13" rx="0.6" fill="currentColor"/>'
    "</svg>"
)

_ICON_WAVE = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<polyline points="1 9 4 9 6 3 9.5 13 11.5 9 15 9"/>'
    "</svg>"
)

_ICON_TAIL = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<line x1="1.5" y1="8" x2="10.5" y2="8"/>'
    '<rect x="3.5" y="4.5" width="4.5" height="7" rx="0.8"/>'
    '<circle cx="13.5" cy="8" r="1.4" fill="currentColor" stroke="none"/>'
    "</svg>"
)


def _fmt_ms(value: object) -> str:
    return f"{value:.2f} ms" if isinstance(value, int | float) else "N/A"


def _fmt_num(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, int | float) else "N/A"


def _severity_variant(severity: str) -> str:
    if severity.startswith("Estável"):
        return "ok"
    if severity.startswith("Atenção"):
        return "warn"
    if severity.startswith("Alerta"):
        return "alert"
    return "neutral"


def _inject_styles() -> None:
    if STYLES_PATH.is_file():
        css = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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


def _render_section_head(index: str, title: str, meta: str = "", *, first: bool = False) -> None:
    cls = "cp-section-head cp-section-head--first" if first else "cp-section-head"
    meta_html = f'<span class="cp-section-head__meta">{html.escape(meta)}</span>' if meta else ""
    st.markdown(
        f'<div class="{cls} cp-anim">'
        f'<h2 class="cp-section-head__title">'
        f'<span class="cp-section-head__index">{html.escape(index)}</span>'
        f"{html.escape(title)}</h2>"
        f"{meta_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_header(
    *,
    request_count: int | None,
    ref_count: int | None,
    prod_count: int | None,
    sla_ms: float,
    sla_passed: bool | None,
    p95_ms: float | None,
    model_version: str,
    server_count: int | None,
) -> None:
    if sla_passed is True:
        pill_cls = "cp-status-pill cp-status-pill--ok"
        pill_text = f"SLA p95 < {sla_ms:.0f} ms · APROVADO ({_fmt_ms(p95_ms)})"
    elif sla_passed is False:
        pill_cls = "cp-status-pill cp-status-pill--alert"
        pill_text = f"SLA p95 < {sla_ms:.0f} ms · VIOLADO ({_fmt_ms(p95_ms)})"
    else:
        pill_cls = "cp-status-pill"
        pill_text = f"SLA p95 < {sla_ms:.0f} ms · AGUARDANDO DADOS"

    chips = []
    if request_count is not None:
        chips.append(("Pedidos HTTP", f"{request_count}"))
    if ref_count is not None and prod_count is not None:
        chips.append(("Ref / Prod", f"{ref_count} / {prod_count}"))
    chips.append(("Modelo", model_version))
    if server_count:
        chips.append(("Log ASGI", f"{server_count} eventos"))

    chips_html = "".join(
        f'<span class="cp-meta-chip">'
        f'<span class="cp-meta-chip__label">{html.escape(label)}</span>'
        f"{html.escape(val)}</span>"
        for label, val in chips
    )
    st.markdown(
        '<header class="cp-header cp-anim">'
        '<div class="cp-header__left">'
        '<div class="cp-eyebrow">'
        "<span>CAREERPATH MLOPS</span>"
        '<span class="cp-eyebrow__sep">/</span>'
        "<span>AV2 PARTE 2</span>"
        '<span class="cp-eyebrow__sep">/</span>'
        "<span>POST /predict</span>"
        "</div>"
        '<h1 class="cp-title">Observabilidade de Latência e Drift</h1>'
        '<p class="cp-subtitle">'
        "Percentis não paramétricos, verificação de SLA e estabilidade entre Referência e Produção."
        "</p>"
        "</div>"
        '<div class="cp-header__right">'
        f'<div class="{pill_cls}">'
        '<span class="cp-live-dot" aria-hidden="true"></span>'
        f"<span>{html.escape(pill_text)}</span>"
        "</div>"
        f'<div class="cp-meta-row">{chips_html}</div>'
        "</div>"
        "</header>",
        unsafe_allow_html=True,
    )


def _render_panel_header(role: str, label: str, variable: str, severity: str) -> None:
    variant = _severity_variant(severity)
    st.markdown(
        '<div class="cp-panel-header">'
        '<div class="cp-panel-header__title">'
        f'<span class="cp-panel-header__role">{html.escape(role)}</span>'
        f'<span class="cp-panel-header__name">{html.escape(label)} '
        f"<code>{html.escape(variable)}</code></span>"
        "</div>"
        f'<span class="cp-badge cp-badge--{variant}">'
        '<span class="cp-badge__dot" aria-hidden="true"></span>'
        f"{html.escape(severity)}"
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_overview_tab(
    latencies: list[float],
    latency_stats: Mapping[str, object],
    server_latencies: list[float] | None,
    ref_records: list[dict[str, object]],
    prod_records: list[dict[str, object]],
    drift: Mapping[str, object],
    sla_ms: float,
) -> None:
    # ------------------------------------------------------------ Bloco 1: Latência & SLA
    _render_section_head(
        "01",
        "Latência e Conformidade de SLA",
        f"Endpoint POST /predict · n = {latency_stats['count']}",
        first=True,
    )

    if latency_stats["count"] == 0:
        st.info("Nenhuma observação de latência disponível no arquivo de benchmark.")
    else:
        sla_ok = bool(latency_stats["sla_passed"])
        sla_label = "APROVADO" if sla_ok else "VIOLADO"
        margin = float(latency_stats["sla_margin_ms"])
        margin_txt = (
            f"{sla_label} (+{margin:.2f} ms)" if sla_ok else f"{sla_label} ({margin:.2f} ms)"
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Mediana (p50)",
            _fmt_ms(latency_stats["median_ms"]),
            delta=f"IQR {_fmt_ms(latency_stats['iqr_ms'])}",
            delta_color="off",
        )
        k2.metric(
            "Percentil 90 (p90)",
            _fmt_ms(latency_stats["p90_ms"]),
            delta=f"Bigode sup. {_fmt_ms(latency_stats['upper_whisker_ms'])}",
            delta_color="off",
        )
        k3.metric(
            f"Percentil 95 · SLA < {sla_ms:.0f} ms",
            _fmt_ms(latency_stats["p95_ms"]),
            delta=margin_txt,
            delta_color="normal" if sla_ok else "inverse",
        )
        k4.metric(
            "Percentil 99 (p99)",
            _fmt_ms(latency_stats["p99_ms"]),
            delta=(
                f"{latency_stats['outlier_count']} outliers · "
                f"máx {latency_stats['max_ms']:.1f} ms"
            ),
            delta_color="off",
        )

        # -------------------------------------------------------- Bloco 2: Distribuição
        _render_section_head(
            "02",
            "Distribuição de Latência (Boxplot e Violino)",
            "Escala horizontal alinhada com limiar de SLA",
        )

        with st.container(border=True):
            overview_fig = build_latency_overview_figure(
                latencies, latency_stats, title_prefix="POST /predict"
            )
            st.pyplot(
                overview_fig,
                width="stretch",
                alt=(
                    f"Boxplot e violino da latência de {latency_stats['count']} pedidos: "
                    f"mediana {_fmt_ms(latency_stats['median_ms'])}, "
                    f"p95 {_fmt_ms(latency_stats['p95_ms'])}, "
                    f"p99 {_fmt_ms(latency_stats['p99_ms'])}."
                ),
            )

            q1 = _fmt_ms(latency_stats["q1_ms"])
            med = _fmt_ms(latency_stats["median_ms"])
            q3 = _fmt_ms(latency_stats["q3_ms"])
            iqr = _fmt_ms(latency_stats["iqr_ms"])
            whisker_hi = _fmt_ms(latency_stats["upper_whisker_ms"])
            max_ms = _fmt_ms(latency_stats["max_ms"])
            outliers = int(latency_stats["outlier_count"])
            st.markdown(
                '<div class="cp-note-grid">'
                '<div class="cp-note-card">'
                f'<div class="cp-note-card__title">{_ICON_WAVE}'
                "<span>Assimetria e dispersão central</span></div>"
                '<p class="cp-note-card__body">'
                f"Metade das respostas concentra-se entre <code>{q1}</code> (Q1) e "
                f"<code>{q3}</code> (Q3), com mediana em <code>{med}</code> e "
                f"<code>IQR = {iqr}</code>. O violino revela a densidade KDE espelhada, "
                "evidenciando a concentração central além do resumo de quartis."
                "</p>"
                "</div>"
                '<div class="cp-note-card">'
                f'<div class="cp-note-card__title">{_ICON_TAIL}'
                "<span>Cauda longa e outliers de Tukey</span></div>"
                '<p class="cp-note-card__body">'
                f"O bigode superior alcança <code>{whisker_hi}</code>, isolando "
                f"<code>{outliers}</code> observações extremas até <code>{max_ms}</code>. "
                "Por isso contratos de SLA são avaliados sobre <code>p95</code> em vez da média."
                "</p>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Gráficos isolados e tabela completa de estatísticas não paramétricas"):
            col_left, col_right = st.columns(2)
            with col_left:
                st.pyplot(
                    build_latency_boxplot_figure(latencies, latency_stats),
                    width="stretch",
                    alt="Boxplot isolado da latência do endpoint /predict.",
                )
            with col_right:
                st.pyplot(
                    build_latency_violin_figure(latencies, latency_stats),
                    width="stretch",
                    alt="Gráfico de violino isolado da latência do endpoint /predict.",
                )

            summary_rows = [
                {"Métrica": label, "Cliente HTTP": value}
                for label, value in _latency_metric_rows(latency_stats, sla_ms)
            ]
            if server_latencies:
                server_stats = compute_latency_stats(server_latencies, sla_ms=sla_ms)
                for row, (_, server_value) in zip(
                    summary_rows, _latency_metric_rows(server_stats, sla_ms), strict=True
                ):
                    row["Log Servidor (ASGI)"] = server_value
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    # ------------------------------------------------------------ Bloco 3: Drift (Espelhado)
    psi_res = drift["primary_input"]
    ks_res = drift["primary_output"]
    _render_section_head(
        "03",
        "Monitoramento de Drift (Entrada e Saída)",
        "Referência em cinza neutro · Produção em destaque cobalto",
    )

    col_in, col_out = st.columns(2, gap="medium")

    with col_in, st.container(border=True):
        _render_panel_header(
            "DATA DRIFT · ENTRADA (PSI)",
            "Entrada",
            PRIMARY_INPUT_VAR,
            str(psi_res["severity"]),
        )
        m1, m2 = st.columns(2)
        m1.metric(
            f"PSI ({PRIMARY_INPUT_VAR})",
            _fmt_num(psi_res["psi"]),
            delta=f"{psi_res['category_count']} cat. · null ativo",
            delta_color="off",
        )
        m2.metric(
            "Amostras (Ref / Prod)",
            f"{psi_res['n_reference']} / {psi_res['n_production']}",
            delta=f"ε = {psi_res['epsilon']:.0e}",
            delta_color="off",
        )
        st.pyplot(
            build_psi_figure(psi_res, compact=True),
            width="stretch",
            alt=(
                f"Comparação percentual por categoria de {PRIMARY_INPUT_VAR} "
                "entre Referência e Produção."
            ),
        )

    with col_out, st.container(border=True):
        _render_panel_header(
            "PREDICTION DRIFT · SAÍDA (KS)",
            "Saída",
            PRIMARY_OUTPUT_VAR,
            str(ks_res["severity"]),
        )
        m3, m4 = st.columns(2)
        m3.metric(
            "Distância KS (D)",
            _fmt_num(ks_res["ks_statistic"]),
            delta=f"D crit (α=0,05): {_fmt_num(ks_res.get('critical_d_0_05'))}",
            delta_color="off",
        )
        m4.metric(
            "p-valor KS",
            f"{ks_res['p_value']:.4g}" if isinstance(ks_res["p_value"], float) else "N/A",
            delta=(
                f"Δ mediana: {ks_res['median_shift']:+.4f}"
                if isinstance(ks_res["median_shift"], float)
                else None
            ),
            delta_color="off",
        )
        ref_probs = extract_numeric_column(ref_records, PRIMARY_OUTPUT_VAR)
        prod_probs = extract_numeric_column(prod_records, PRIMARY_OUTPUT_VAR)
        st.pyplot(
            build_ks_figure(ref_probs, prod_probs, ks_res, compact=True),
            width="stretch",
            alt=f"ECDF e densidade comparada de {PRIMARY_OUTPUT_VAR} destacando distância KS.",
        )

    with st.expander("Visualizar curva completa δ(x) = |F_ref(x) - F_prod(x)| em tela larga"):
        st.pyplot(
            build_ks_figure(ref_probs, prod_probs, ks_res, compact=False),
            width="stretch",
            alt=f"Painel triplo KS (ECDF, delta e densidade) de {PRIMARY_OUTPUT_VAR}.",
        )


def _render_report_tab(
    latency_stats: Mapping[str, object],
    server_latencies: list[float] | None,
    drift: Mapping[str, object],
    sla_ms: float,
) -> None:
    psi_res = drift["primary_input"]
    ks_res = drift["primary_output"]
    supp = drift["supplementary"]

    _render_section_head(
        "04",
        "Relatório Executivo e Conclusões",
        "Síntese metodológica, SLA e estabilidade do contrato",
        first=True,
    )

    alert_vars = sum(1 for row in supp if str(row["severity"]).startswith("Alerta"))
    watch_vars = sum(1 for row in supp if str(row["severity"]).startswith("Atenção"))
    sla_ok = bool(latency_stats["sla_passed"]) if latency_stats["sla_passed"] is not None else False
    sla_margin = latency_stats.get("sla_margin_ms")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Pedidos no benchmark", str(latency_stats["count"]))
    e2.metric(
        "Folga de SLA (p95)",
        _fmt_ms(sla_margin),
        delta="APROVADO" if sla_ok else "VIOLADO",
        delta_color="normal" if sla_ok else "inverse",
    )
    e3.metric(
        "Variáveis em alerta / atenção",
        f"{alert_vars} / {watch_vars}",
        delta=f"{len(supp)} variáveis avaliadas",
        delta_color="off",
    )
    e4.metric(
        "Deslocamento mediana (saída)",
        (
            f"{ks_res['median_shift']:+.4f}"
            if isinstance(ks_res["median_shift"], float)
            else "N/A"
        ),
        delta=f"Ref {_fmt_num(ks_res['ref_median'])} → Prod {_fmt_num(ks_res['prod_median'])}",
        delta_color="off",
    )

    server_note = ""
    if server_latencies:
        s_stats = compute_latency_stats(server_latencies, sla_ms=sla_ms)
        server_note = (
            f" No log interno ASGI (<code>n={s_stats['count']}</code>), a mediana registrada foi "
            f"<code>{_fmt_ms(s_stats['median_ms'])}</code> e o "
            f"<code>p95 = {_fmt_ms(s_stats['p95_ms'])}</code>; "
            "a diferença reflete serialização HTTP/JSON e stack TCP loopback no cliente."
        )

    sla_verdict = "APROVADO" if sla_ok else "VIOLADO"
    ks_sig = (
        "ultrapassa marginalmente o valor crítico estatístico a 5%"
        if ks_res.get("significant_at_0_05")
        else "permanece abaixo do valor crítico estatístico a 5%"
    )

    st.markdown(
        '<div class="cp-report-grid">'
        '<div class="cp-report-card">'
        '<div class="cp-report-card__eyebrow">01 · Desempenho e SLA</div>'
        '<h3 class="cp-report-card__title">Conclusão sobre tempo de resposta</h3>'
        '<p class="cp-report-card__text">'
        f"Nos <code>{latency_stats['count']}</code> pedidos HTTP enviados a "
        "<code>POST /predict</code>, "
        f"90% concluíram abaixo de <code>{_fmt_ms(latency_stats['p90_ms'])}</code>, 95% abaixo de "
        f"<code>{_fmt_ms(latency_stats['p95_ms'])}</code> e 99% abaixo de "
        f"<code>{_fmt_ms(latency_stats['p99_ms'])}</code>. O contrato de SLA "
        f"(<code>p95 &lt; {sla_ms:.0f} ms</code>) está <strong>{sla_verdict}</strong>."
        f"{server_note}"
        "</p>"
        '<p class="cp-report-card__text">'
        "O uso combinado de boxplot e gráfico de violino separa a dispersão interquartil "
        f"(<code>IQR = {_fmt_ms(latency_stats['iqr_ms'])}</code>) dos picos isolados na cauda "
        f"(<code>{latency_stats['outlier_count']}</code> outliers até "
        f"<code>{_fmt_ms(latency_stats['max_ms'])}</code>), evitando distorções da média."
        "</p>"
        "</div>"
        '<div class="cp-report-card">'
        '<div class="cp-report-card__eyebrow">02 · Estabilidade e Drift</div>'
        '<h3 class="cp-report-card__title">Conclusão sobre entrada e saída</h3>'
        '<p class="cp-report-card__text">'
        "Seguindo a regra de seleção (KS para variáveis contínuas ou inteiras com range &gt; 100; "
        f"PSI para as demais), a entrada <code>{PRIMARY_INPUT_VAR}</code> obteve "
        f"<strong>PSI = <code>{_fmt_num(psi_res['psi'])}</code></strong> "
        f"(<em>{html.escape(str(psi_res['severity']))}</em>) com tratamento explícito de valores "
        "nulos na categoria <code>(ausente)</code>."
        "</p>"
        '<p class="cp-report-card__text">'
        f"A saída contínua <code>{PRIMARY_OUTPUT_VAR}</code> registrou "
        f"<strong>KS D = <code>{_fmt_num(ks_res['ks_statistic'])}</code></strong> "
        f"(<code>D crit = {_fmt_num(ks_res.get('critical_d_0_05'))}</code>, "
        f"<code>p = {ks_res['p_value']:.4g}</code> — {ks_sig}), mantendo classificação operacional "
        f"<em>{html.escape(str(ks_res['severity']))}</em>."
        "</p>"
        "</div>"
        "</div>"
        '<div class="cp-callout">'
        "<strong>Nota metodológica:</strong> deslocamentos estatísticos de entrada (PSI) ou saída "
        "(KS) sinalizam mudança de população e justificam auditoria periódica dos rótulos reais, "
        "mas não comprovam isoladamente queda de acurácia do classificador."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Fórmulas de referência (PSI / Jeffreys e KS biamostral)"):
        st.markdown(
            "- **PSI (Divergência KL simetrizada / Jeffreys):** "
            r"$\mathrm{PSI} = \sum_i (p_{\mathrm{prod},i} - p_{\mathrm{ref},i}) "
            r"\ln\!\left(\frac{p_{\mathrm{prod},i}}{p_{\mathrm{ref},i}}\right)$"
            " (limiares padrão: `< 0,10` estável; `0,10–0,25` atenção; `>= 0,25` alerta).\n"
            "- **KS biamostral:** "
            r"$D_{n,m} = \sup_x |F_{\mathrm{ref},n}(x) - F_{\mathrm{prod},m}(x)|$"
            " e valor crítico "
            r"$D_{\mathrm{crit}}(\alpha) = c(\alpha)\sqrt{\frac{n+m}{n\cdot m}}$"
            r" com $c(0{,}05) \approx 1{,}358$."
        )

    col_tbl1, col_tbl2 = st.columns([1.35, 1.0], gap="medium")
    with col_tbl1:
        st.markdown(f"**Detalhamento por categoria — `{PRIMARY_INPUT_VAR}` (PSI)**")
        if psi_res["breakdown"]:
            psi_df = pd.DataFrame(psi_res["breakdown"]).rename(
                columns={
                    "category": "Categoria",
                    "ref_count": "Qtd Ref",
                    "prod_count": "Qtd Prod",
                    "ref_pct": "Ref (%)",
                    "prod_pct": "Prod (%)",
                    "delta_pp": "Δ (p.p.)",
                    "psi_contribution": "Contrib. PSI",
                }
            )
            st.dataframe(psi_df, width="stretch", hide_index=True)

    with col_tbl2:
        st.markdown(f"**Resumo estatístico — `{PRIMARY_OUTPUT_VAR}` (KS)**")
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

    st.markdown("**Visão complementar de todas as variáveis do contrato**")
    supp_df = pd.DataFrame(supp).rename(
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
    _render_section_head(
        "05",
        "Amostras Submetidas e Reprodução CLI",
        f"Diretório: {evidence_dir}",
        first=True,
    )
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
    col1, col2 = st.columns(2, gap="medium")
    for col, label, records in (
        (col1, "Conjunto de Referência", ref_records),
        (col2, "Conjunto de Produção", prod_records),
    ):
        with col:
            st.markdown(f"**{label}** (`{len(records)}` registros válidos)")
            frame = pd.DataFrame(records)
            cols = [c for c in preview_cols if c in frame.columns] or list(frame.columns)
            st.dataframe(frame[cols].head(25), width="stretch", hide_index=True)

    st.markdown("**Reprodução local pela linha de comando**")
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
        page_title="CareerPath MLOps — Observabilidade AV2 Parte 2",
        layout="wide",
        initial_sidebar_state="auto",
    )
    _inject_styles()

    default_dir = _default_evidence_dir()
    with st.sidebar:
        st.markdown(
            '<div class="cp-sidebar-brand">'
            f'<span class="cp-sidebar-brand__mark">{_ICON_BRAND}</span>'
            "<div>"
            '<div class="cp-sidebar-brand__name">CareerPath MLOps</div>'
            '<div class="cp-sidebar-brand__sub">Monitoramento · AV2 P2</div>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="cp-sidebar-group">Fonte de evidências</div>',
            unsafe_allow_html=True,
        )
        source_mode = st.radio(
            "Modo de carregamento",
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

        st.markdown(
            '<div class="cp-sidebar-group">Limiares operacionais</div>',
            unsafe_allow_html=True,
        )
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
        _render_header(
            request_count=None,
            ref_count=None,
            prod_count=None,
            sla_ms=sla_ms,
            sla_passed=None,
            p95_ms=None,
            model_version="adult-income-v1",
            server_count=None,
        )
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
        _render_header(
            request_count=None,
            ref_count=None,
            prod_count=None,
            sla_ms=sla_ms,
            sla_passed=None,
            p95_ms=None,
            model_version="adult-income-v1",
            server_count=None,
        )
        st.error(f"Não foi possível ler os arquivos de monitoramento: {exc}")
        return

    latencies = extract_numeric_column(bench_records, "client_latency_ms")
    latency_stats = compute_latency_stats(latencies, sla_ms=sla_ms)
    drift = compute_all_variable_drifts(
        ref_records,
        prod_records,
        psi_watch=psi_watch,
        psi_alert=psi_alert,
        ks_alert_d=ks_alert_d,
    )
    model_version = next(
        (
            str(r["model_version"])
            for r in ref_records
            if isinstance(r.get("model_version"), str) and r["model_version"]
        ),
        "adult-income-v1",
    )

    _render_header(
        request_count=int(latency_stats["count"]),
        ref_count=len(ref_records),
        prod_count=len(prod_records),
        sla_ms=sla_ms,
        sla_passed=(
            bool(latency_stats["sla_passed"])
            if latency_stats["sla_passed"] is not None
            else None
        ),
        p95_ms=latency_stats["p95_ms"] if isinstance(latency_stats["p95_ms"], float) else None,
        model_version=model_version,
        server_count=len(server_latencies) if server_latencies else None,
    )

    tab_overview, tab_report, tab_data = st.tabs(
        ["Visão Geral", "Relatório", "Dados & CLI"]
    )
    with tab_overview:
        _render_overview_tab(
            latencies,
            latency_stats,
            server_latencies,
            ref_records,
            prod_records,
            drift,
            sla_ms,
        )
    with tab_report:
        _render_report_tab(latency_stats, server_latencies, drift, sla_ms)
    with tab_data:
        _render_data_tab(ref_records, prod_records, evidence_dir)


if __name__ == "__main__":
    main()
