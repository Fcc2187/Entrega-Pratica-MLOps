"""Shared Matplotlib/Seaborn figure builders for CLI reports and Streamlit."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import PRIMARY_INPUT_VAR, PRIMARY_OUTPUT_VAR, compute_ecdf_diff_curve
from .io import to_finite_floats

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
    ref_clean, _ = to_finite_floats(ref_values)
    prod_clean, _ = to_finite_floats(prod_values)
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
