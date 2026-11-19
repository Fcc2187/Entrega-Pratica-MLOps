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

# Unified Taste Skill chart theme (white-link-blue + semantic data-only colors, zero violet).
CHART_THEME: dict[str, object] = {
    "bg": "none",
    "surface": "#f8f9fb",
    "white": "#ffffff",
    "ink": "#111111",
    "muted": "#525866",
    "grid": "#e2e6ee",
    "spine": "#cbd2dc",
    "reference": "#94a3b8",
    "reference_edge": "#64748b",
    "production": "#1a4be0",
    "production_soft": "#dbeafe",
    "ok": "#047857",
    "warn": "#d97706",
    "alert": "#dc2626",
    "font_sans": ["Instrument Sans", "DejaVu Sans", "Arial", "sans-serif"],
    "font_mono": ["DM Mono", "DejaVu Sans Mono", "Consolas", "monospace"],
}

# Backwards-compatible color palette mapping derived from CHART_THEME.
PALETTE: dict[str, str] = {
    "ink": str(CHART_THEME["ink"]),
    "muted": str(CHART_THEME["muted"]),
    "grid": str(CHART_THEME["grid"]),
    "bg": str(CHART_THEME["surface"]),
    "white": str(CHART_THEME["white"]),
    "blue": str(CHART_THEME["production"]),
    "emerald": str(CHART_THEME["ok"]),
    "orange": str(CHART_THEME["warn"]),
    "red": str(CHART_THEME["alert"]),
    "reference": str(CHART_THEME["reference"]),
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
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = list(CHART_THEME["font_sans"])
    matplotlib.rcParams["axes.edgecolor"] = str(CHART_THEME["spine"])
    matplotlib.rcParams["axes.linewidth"] = 0.9
    import seaborn as sns

    return Figure, FigureCanvasAgg, sns


def apply_chart_theme(
    fig: object,
    ax: object,
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    transparent: bool = True,
    grid_axis: str = "both",
) -> None:
    """Apply the unified CareerPath observability chart theme to a Matplotlib figure/axis."""
    if transparent:
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(PALETTE["white"])
        ax.set_facecolor(PALETTE["white"])

    ax.set_title(
        title,
        fontsize=10.8,
        fontweight="bold",
        color=PALETTE["ink"],
        loc="left",
        pad=11,
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.2, color=PALETTE["muted"], labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.2, color=PALETTE["muted"], labelpad=6)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8.6, length=3, width=0.8)
    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.75,
        alpha=0.75,
        color=PALETTE["grid"],
    )
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(str(CHART_THEME["spine"]))
        ax.spines[spine].set_linewidth(0.85)


def _empty_figure(title: str, message: str) -> object:
    Figure, FigureCanvasAgg, _ = _import_plotting()
    fig = Figure(figsize=(8, 4.0), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    apply_chart_theme(fig, ax, title=title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10.5,
        color=PALETTE["muted"],
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return fig


def _annotate_sla_line(
    ax: object,
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
) -> None:
    """Highlight the SLA threshold clearly on both Boxplot and Violin axes."""
    sla = stats.get("sla_threshold_ms")
    sla_passed = stats.get("sla_passed")
    if not isinstance(sla, float) or not isinstance(sla_passed, bool):
        return

    max_obs = max(latencies_ms)
    min_obs = min(latencies_ms)
    span = max(max_obs - min_obs, 1.0)
    sla_color = PALETTE["emerald"] if sla_passed else PALETTE["red"]
    status = "APROVADO" if sla_passed else "VIOLADO"

    if sla <= max_obs * 1.65:
        x_right = max(max_obs, sla) + span * 0.14
        x_left = max(0.0, min_obs - span * 0.08)
        ax.set_xlim(x_left, x_right)
        ax.axvspan(sla, x_right, color=PALETTE["red"], alpha=0.06, zorder=0)
        ax.axvline(
            sla,
            color=sla_color,
            linestyle="-",
            linewidth=2.0,
            zorder=4,
            label=f"SLA = {sla:.0f} ms ({status})",
        )
        ax.text(
            sla,
            0.95,
            f" SLA {sla:.0f} ms · {status}",
            transform=ax.get_xaxis_transform(),
            ha="right" if sla > (x_left + x_right) * 0.65 else "left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=sla_color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": PALETTE["white"],
                "edgecolor": sla_color,
                "linewidth": 0.9,
                "alpha": 0.94,
            },
        )
    else:
        # Keep the distribution legible while rendering an explicit right-edge SLA boundary line.
        x_left = max(0.0, min_obs - span * 0.08)
        x_right = max_obs + span * 0.22
        ax.set_xlim(x_left, x_right)
        marker_x = max_obs + span * 0.18
        ax.axvline(
            marker_x,
            color=sla_color,
            linestyle=(0, (4, 3)),
            linewidth=1.8,
            zorder=4,
            label=f"SLA = {sla:.0f} ms → ({status})",
        )
        margin = stats.get("sla_margin_ms")
        margin_txt = f" (+{margin:.1f} ms)" if isinstance(margin, float) else ""
        ax.text(
            marker_x - span * 0.03,
            0.94,
            f"SLA p95 < {sla:.0f} ms · {status}{margin_txt} →",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=sla_color,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": PALETTE["white"],
                "edgecolor": sla_color,
                "linewidth": 0.9,
                "alpha": 0.96,
            },
        )


def _draw_boxplot_panel(
    fig: object,
    ax: object,
    latencies_ms: Sequence[float],
    stats: Mapping[str, object],
    *,
    title: str,
) -> None:
    box = ax.boxplot(
        list(latencies_ms),
        orientation="horizontal",
        widths=0.38,
        patch_artist=True,
        boxprops={
            "facecolor": str(CHART_THEME["production_soft"]),
            "edgecolor": PALETTE["blue"],
            "alpha": 0.92,
            "linewidth": 1.4,
        },
        medianprops={"color": PALETTE["ink"], "linewidth": 2.2},
        whiskerprops={"color": PALETTE["muted"], "linewidth": 1.25},
        capprops={"color": PALETTE["muted"], "linewidth": 1.25},
        flierprops={
            "marker": "o",
            "markersize": 4.2,
            "markerfacecolor": PALETTE["red"],
            "markeredgecolor": PALETTE["white"],
            "markeredgewidth": 0.6,
            "alpha": 0.85,
        },
    )
    assert box is not None
    p95 = stats.get("p95_ms")
    p99 = stats.get("p99_ms")
    if isinstance(p95, float):
        ax.axvline(
            p95,
            color=PALETTE["blue"],
            linestyle="--",
            linewidth=1.6,
            label=f"p95 = {p95:.2f} ms",
        )
    if isinstance(p99, float):
        ax.axvline(
            p99,
            color=PALETTE["orange"],
            linestyle=":",
            linewidth=1.7,
            label=f"p99 = {p99:.2f} ms",
        )
    _annotate_sla_line(ax, latencies_ms, stats)
    ax.set_yticks([])
    apply_chart_theme(fig, ax, title=title, xlabel="Tempo de resposta (ms)", grid_axis="x")
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=str(CHART_THEME["spine"]),
        framealpha=0.94,
        fontsize=8.1,
    )


def _draw_violin_panel(
    fig: object,
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
        color=PALETTE["blue"],
        inner="quartile",
        cut=0,
        linewidth=1.15,
    )
    for collection in ax.collections:
        collection.set_alpha(0.24)
        collection.set_edgecolor(PALETTE["blue"])
    for line in ax.lines:
        line.set_color(PALETTE["ink"])
        line.set_alpha(0.75)

    median = stats.get("median_ms")
    p95 = stats.get("p95_ms")
    if isinstance(median, float):
        ax.axvline(
            median,
            color=PALETTE["ink"],
            linestyle="-",
            linewidth=1.7,
            label=f"Mediana = {median:.2f} ms",
        )
    if isinstance(p95, float):
        ax.axvline(
            p95,
            color=PALETTE["blue"],
            linestyle="--",
            linewidth=1.6,
            label=f"p95 = {p95:.2f} ms",
        )
    _annotate_sla_line(ax, latencies_ms, stats)
    ax.set_yticks([])
    apply_chart_theme(fig, ax, title=title, xlabel="Tempo de resposta (ms)", grid_axis="x")
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=str(CHART_THEME["spine"]),
        framealpha=0.94,
        fontsize=8.1,
    )


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
    fig = Figure(figsize=(8.4, 3.8), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _draw_boxplot_panel(
        fig,
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
    fig = Figure(figsize=(8.4, 3.8), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _draw_violin_panel(
        fig,
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
    fig = Figure(figsize=(12.2, 4.15), dpi=150)
    FigureCanvasAgg(fig)
    ax_box = fig.add_subplot(121)
    ax_violin = fig.add_subplot(122, sharex=ax_box)
    _draw_boxplot_panel(
        fig,
        ax_box,
        latencies_ms,
        stats,
        title=f"Boxplot · Quartis, Tukey e outliers ({title_prefix})",
    )
    _draw_violin_panel(
        fig,
        ax_violin,
        sns,
        latencies_ms,
        stats,
        title=f"Violino · Densidade KDE e quartis (n={stats['count']})",
    )
    fig.tight_layout(w_pad=2.2)
    return fig


def build_psi_figure(
    psi_result: Mapping[str, object],
    *,
    compact: bool = False,
) -> object:
    """Build grouped horizontal bar chart comparing Reference (grey) vs Production (accent)."""
    breakdown = psi_result.get("breakdown")
    variable = str(psi_result.get("variable", PRIMARY_INPUT_VAR))
    if not isinstance(breakdown, list) or not breakdown:
        return _empty_figure(f"PSI — {variable}", "Dados insuficientes para calcular PSI.")

    Figure, FigureCanvasAgg, _ = _import_plotting()
    ordered = sorted(breakdown, key=lambda r: float(r["ref_pct"]) + float(r["prod_pct"]))
    categories = [str(r["category"]) for r in ordered]
    ref_pct = [float(r["ref_pct"]) for r in ordered]
    prod_pct = [float(r["prod_pct"]) for r in ordered]
    deltas = [float(r["delta_pp"]) for r in ordered]

    figsize = (6.8, 6.2) if compact else (
        9.6,
        max(4.6, 0.36 * len(categories) + 1.8),
    )
    fig = Figure(figsize=figsize, dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    y = np.arange(len(categories))
    height = 0.37
    ax.barh(
        y + height / 2,
        ref_pct,
        height=height,
        color=PALETTE["reference"],
        alpha=0.82,
        label="Referência (%)",
    )
    ax.barh(
        y - height / 2,
        prod_pct,
        height=height,
        color=PALETTE["blue"],
        alpha=0.92,
        label="Produção (%)",
    )
    max_bar = max(max(ref_pct, default=1.0), max(prod_pct, default=1.0), 1.0)
    ax.set_xlim(0.0, max_bar * 1.24)
    for idx, (r_val, p_val, diff) in enumerate(zip(ref_pct, prod_pct, deltas, strict=True)):
        if abs(diff) >= 1.5:
            color = PALETTE["blue"] if diff > 0 else PALETTE["muted"]
            ax.text(
                max(r_val, p_val) + max_bar * 0.02,
                idx,
                f"{diff:+.1f} pp",
                va="center",
                ha="left",
                fontsize=7.6,
                fontweight="normal",
                color=color,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8.4)
    psi_val = psi_result.get("psi")
    psi_label = f"{psi_val:.4f}" if isinstance(psi_val, float) else "N/A"
    apply_chart_theme(
        fig,
        ax,
        title=f"Frequências por categoria ({variable}) · PSI = {psi_label}",
        xlabel="Proporção da amostra (%)",
        grid_axis="x",
    )
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=str(CHART_THEME["spine"]),
        framealpha=0.95,
        fontsize=8.3,
    )
    fig.tight_layout()
    return fig


def _draw_ks_ecdf_axis(
    fig: object,
    ax: object,
    grid: np.ndarray,
    ecdf_ref: np.ndarray,
    ecdf_prod: np.ndarray,
    ks_result: Mapping[str, object],
    variable: str,
) -> None:
    ax.fill_between(
        grid,
        ecdf_ref,
        ecdf_prod,
        step="post",
        color=PALETTE["blue"],
        alpha=0.09,
    )
    ax.step(
        grid,
        ecdf_ref,
        where="post",
        color=str(CHART_THEME["reference_edge"]),
        linewidth=1.9,
        linestyle="-",
        label="Referência",
    )
    ax.step(
        grid,
        ecdf_prod,
        where="post",
        color=PALETTE["blue"],
        linewidth=2.1,
        label="Produção",
    )
    max_x = ks_result.get("max_diff_at")
    y_ref = ks_result.get("max_diff_ecdf_ref")
    y_prod = ks_result.get("max_diff_ecdf_prod")
    ks_stat = ks_result.get("ks_statistic")
    if isinstance(max_x, float) and isinstance(y_ref, float) and isinstance(y_prod, float):
        ax.vlines(
            max_x,
            min(y_ref, y_prod),
            max(y_ref, y_prod),
            colors=PALETTE["red"],
            linestyles="--",
            linewidth=2.1,
            zorder=5,
            label=f"KS D = {ks_stat:.4f} em {max_x:.3f}",
        )
        ax.scatter(
            [max_x, max_x],
            [y_ref, y_prod],
            color=PALETTE["red"],
            s=22,
            zorder=6,
        )
    apply_chart_theme(
        fig,
        ax,
        title="ECDF · Referência vs Produção",
        xlabel=variable,
        ylabel="Prob. acumulada",
    )
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=str(CHART_THEME["spine"]),
        framealpha=0.95,
        fontsize=8.0,
    )


def _draw_ks_delta_axis(
    fig: object,
    ax: object,
    grid: np.ndarray,
    delta: np.ndarray,
    ks_result: Mapping[str, object],
    variable: str,
) -> None:
    ks_stat = ks_result.get("ks_statistic")
    max_x = ks_result.get("max_diff_at")
    crit_d = ks_result.get("critical_d_0_05")
    ax.fill_between(grid, 0.0, delta, step="post", color=PALETTE["blue"], alpha=0.10)
    ax.step(
        grid,
        delta,
        where="post",
        color=PALETTE["ink"],
        linewidth=1.6,
        label=r"$\delta(x) = |F_{\mathrm{ref}} - F_{\mathrm{prod}}|$",
    )
    if isinstance(ks_stat, float):
        ax.axhline(
            ks_stat,
            color=PALETTE["red"],
            linestyle="--",
            linewidth=1.5,
            label=f"D = {ks_stat:.4f}",
        )
        if isinstance(max_x, float):
            ax.scatter([max_x], [ks_stat], color=PALETTE["red"], zorder=5, s=34)
    if isinstance(crit_d, float):
        ax.axhline(
            crit_d,
            color=PALETTE["emerald"],
            linestyle=":",
            linewidth=1.6,
            label=f"D crit (α=0,05) = {crit_d:.4f}",
        )
    apply_chart_theme(
        fig,
        ax,
        title="Diferença absoluta δ(x)",
        xlabel=variable,
        ylabel="|F_ref(x) - F_prod(x)|",
    )
    top_delta = max(float(delta.max()), float(crit_d) if isinstance(crit_d, float) else 0.0, 0.04)
    ax.set_ylim(0.0, top_delta * 1.35)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=str(CHART_THEME["spine"]),
        framealpha=0.95,
        fontsize=8.0,
    )


def _draw_ks_density_axis(
    fig: object,
    ax: object,
    ref_arr: np.ndarray,
    prod_arr: np.ndarray,
    ks_result: Mapping[str, object],
    variable: str,
) -> None:
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        ref_arr,
        bins=bins,
        density=True,
        alpha=0.62,
        color=PALETTE["reference"],
        edgecolor=PALETTE["white"],
        linewidth=0.7,
        label="Referência",
    )
    ax.hist(
        prod_arr,
        bins=bins,
        density=True,
        alpha=0.45,
        color=PALETTE["blue"],
        edgecolor=PALETTE["blue"],
        linewidth=0.9,
        label="Produção",
    )
    p_val = ks_result.get("p_value")
    p_label = f"{p_val:.4g}" if isinstance(p_val, float) else "N/A"
    apply_chart_theme(
        fig,
        ax,
        title=f"Densidade comparada (p-valor = {p_label})",
        xlabel=variable,
        ylabel="Densidade",
    )
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=str(CHART_THEME["spine"]),
        framealpha=0.95,
        fontsize=8.0,
    )


def build_ks_figure(
    ref_values: Sequence[float],
    prod_values: Sequence[float],
    ks_result: Mapping[str, object],
    *,
    compact: bool = False,
) -> object:
    """Build ECDF (highlighting max gap D), delta curve, and overlapping density for output."""
    variable = str(ks_result.get("variable", PRIMARY_OUTPUT_VAR))
    ref_clean, _ = to_finite_floats(ref_values)
    prod_clean, _ = to_finite_floats(prod_values)
    ref_arr = np.sort(np.asarray(ref_clean, dtype=float))
    prod_arr = np.sort(np.asarray(prod_clean, dtype=float))
    if len(ref_arr) == 0 or len(prod_arr) == 0:
        return _empty_figure(f"KS — {variable}", "Dados insuficientes para calcular KS.")

    Figure, FigureCanvasAgg, _ = _import_plotting()
    grid, ecdf_ref, ecdf_prod, delta = compute_ecdf_diff_curve(ref_arr, prod_arr)

    if compact:
        # Stacked 2-row figure sized to mirror build_psi_figure(..., compact=True) side-by-side.
        fig = Figure(figsize=(6.8, 6.2), dpi=150)
        FigureCanvasAgg(fig)
        ax_ecdf = fig.add_subplot(211)
        ax_hist = fig.add_subplot(212)
        _draw_ks_ecdf_axis(fig, ax_ecdf, grid, ecdf_ref, ecdf_prod, ks_result, variable)
        _draw_ks_density_axis(fig, ax_hist, ref_arr, prod_arr, ks_result, variable)
        fig.tight_layout(h_pad=1.8)
        return fig

    fig = Figure(figsize=(14.4, 4.4), dpi=150)
    FigureCanvasAgg(fig)
    ax_ecdf = fig.add_subplot(131)
    ax_delta = fig.add_subplot(132)
    ax_hist = fig.add_subplot(133)

    _draw_ks_ecdf_axis(fig, ax_ecdf, grid, ecdf_ref, ecdf_prod, ks_result, variable)
    _draw_ks_delta_axis(fig, ax_delta, grid, delta, ks_result, variable)
    _draw_ks_density_axis(fig, ax_hist, ref_arr, prod_arr, ks_result, variable)

    ks_stat = ks_result.get("ks_statistic")
    severity = ks_result.get("severity", "")
    ks_label = f"{ks_stat:.4f}" if isinstance(ks_stat, float) else "N/A"
    fig.suptitle(
        f"Teste de Kolmogorov-Smirnov ({variable}) — D = {ks_label} [{severity}]",
        fontsize=11.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.tight_layout()
    return fig
