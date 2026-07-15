#!/usr/bin/env python3
"""Generate protocol-consistent EasyTSAD paper tables and vector figures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results" / "RENE"
TEMPLATE = ROOT.parent / "IEEE-conference-template"
FIGURES = TEMPLATE / "figures"
GENERATED = TEMPLATE / "generated"
DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")
PAPER_METRICS = ("point_f1_pa", "event_f1_pa_log")
LODO_METHOD = "TShape-zero_shot_balanced"
PATTERN_METHOD = "TShapeUniversalPattern-100"
PROPOSED_METHOD = "TShapeUniversalZeroPlus-100"
GUARD_METHOD = "TShape-zero_plus_pattern_balanced_residual_only_minmax"
FOCAL_EXCLUDED_METHODS = {"TimesFM-2.5-200M"}

COLORS = {
    "teal": "#007C83",
    "coral": "#E45756",
    "gold": "#D49300",
    "purple": "#6F4C9B",
    "green": "#3A923A",
    "blue": "#3274A1",
    "gray": "#6B7280",
    "light": "#E8ECEF",
    "ink": "#17212B",
}
DATASET_COLORS = dict(zip(DATASETS, ["#007C83", "#E45756", "#D49300", "#6F4C9B", "#3A923A", "#3274A1"]))


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.edgecolor": "#303840",
            "axes.linewidth": 0.65,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=dpi)
    plt.close(fig)


def draw_intro_story() -> None:
    """Two-column overview that keeps claim, repair, and evidence boundaries explicit."""
    fig, ax = plt.subplots(figsize=(7.16, 3.46))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def card(x, y, w, h, stage, title, body, color):
        patch = mpl.patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.009,rounding_size=0.018",
            facecolor="white",
            edgecolor=color,
            linewidth=1.25,
        )
        ax.add_patch(patch)
        ax.text(x + 0.015, y + h - 0.030, stage.upper(), color=color,
                fontsize=6.0, weight="bold", va="top")
        ax.text(x + 0.015, y + h - 0.084, title, color=COLORS["ink"],
                fontsize=7.4, weight="bold", va="top")
        ax.text(x + 0.015, y + h - 0.147, body, color=COLORS["gray"],
                fontsize=5.45, va="top", linespacing=1.18)

    def arrow(x0, x1, y=0.695):
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops={"arrowstyle": "-|>", "color": "#657487", "lw": 1.0},
        )

    ax.text(0.015, 0.968, "CLAIM -> STRESS TEST -> CONSTRUCTIVE REPAIR -> EXECUTABLE PRODUCT",
            color=COLORS["ink"], fontsize=7.0, weight="bold", va="top")
    xs = (0.015, 0.263, 0.511, 0.759)
    width = 0.222
    card(xs[0], 0.505, width, 0.385, "Original claim", "Target-trained TShape",
         "Per-series training\n100-epoch EasyTSAD recipe\n5 reported benchmarks",
         COLORS["teal"])
    card(xs[1], 0.505, width, 0.385, "Reuse test", "Direct TShape-Zero",
         "Leave-one-dataset-out\nno target labels/gradients\n6 full-series targets",
         COLORS["gold"])
    card(xs[2], 0.505, width, 0.385, "Algorithmic repair", "TShape-Zero+",
         "Equal-source pretraining\nnested Pattern Bank v2\nopen-box residual guard",
         COLORS["coral"])
    card(xs[3], 0.505, width, 0.385, "Engineering release", "Product contract",
         "Versioned checkpoint + CLI/Web\naligned score + attribution\nprovenance + failure evidence",
         COLORS["purple"])
    for x0, x1 in zip((0.237, 0.485, 0.733), (0.263, 0.511, 0.759)):
        arrow(x0, x1)

    evidence = [
        (0.015, 0.310, "EMPIRICAL BREADTH", "834 KPI series | 15.1M test points\n4 official foundation checkpoints\nall original TShape baseline families", COLORS["blue"]),
        (0.345, 0.310, "MECHANISM ISOLATION", "10-level Pattern Bank sweep\nmatched-guard and channel ablations\npaired bootstrap intervals", "#C43C64"),
        (0.675, 0.310, "SOFTWARE CONTRACT", "preprocessing + score semantics\ncoverage + checksums + resource card\nshared CLI/Web production scorer", COLORS["green"]),
    ]
    for x, w, heading, body, color in evidence:
        box = mpl.patches.FancyBboxPatch(
            (x, 0.075), w, 0.285,
            boxstyle="round,pad=0.009,rounding_size=0.016",
            facecolor=mpl.colors.to_rgba(color, 0.045),
            edgecolor=color,
            linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(x + 0.016, 0.325, heading, color=color, fontsize=5.9,
                weight="bold", va="top")
        ax.text(x + 0.016, 0.265, body, color=COLORS["ink"], fontsize=5.45,
                va="top", linespacing=1.22)

    ax.text(0.015, 0.018,
            "Claim boundary: we preserve TShape's target-trained result; the tested object is checkpoint-style reuse.",
            color=COLORS["gray"], fontsize=5.25, weight="bold", va="bottom")
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.008, top=0.992)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(FIGURES / f"fig01_intro_story.{suffix}", dpi=360,
                        bbox_inches=None, pad_inches=0)
    plt.close(fig)


def draw_zero_plus_framework() -> None:
    """Camera-ready 16:7 framework figure with explicit module I/O."""
    fig, ax = plt.subplots(figsize=(7.16, 3.13))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, body, color, fill=None, title_size=7.0, body_size=5.7):
        patch = mpl.patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            facecolor=fill or mpl.colors.to_rgba(color, 0.055),
            edgecolor=color,
            linewidth=1.05,
        )
        ax.add_patch(patch)
        ax.text(x + 0.012, y + h - 0.024, title, color=COLORS["ink"], fontsize=title_size, weight="bold", va="top")
        ax.text(x + 0.012, y + h - 0.070, body, color=COLORS["gray"], fontsize=body_size, va="top", linespacing=1.18)
        return patch

    def arrow(x0, y0, x1, y1, color=COLORS["ink"], style="-|>", lw=0.9):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": style, "lw": lw, "color": color, "shrinkA": 1, "shrinkB": 1})

    stages = [
        (0.015, 0.965, "A. TARGET-EXCLUDED PRETRAINING", COLORS["purple"]),
        (0.375, 0.965, "B. ZERO-SHOT SCORING", COLORS["teal"]),
        (0.75, 0.965, "C. PRODUCT OUTPUT", COLORS["coral"]),
    ]
    for x, y, label, color in stages:
        ax.text(x, y, label, fontsize=6.8, weight="bold", color=color, va="top")

    box(0.015, 0.65, 0.145, 0.235, "Real source bank", "five datasets D_-q\ntarget q excluded\n9,000 windows / source", COLORS["blue"])
    trace_x = np.linspace(0.032, 0.142, 52)
    for row, color in enumerate((COLORS["teal"], COLORS["gold"])):
        center = 0.672 + 0.020 * row
        ax.plot(trace_x, center + 0.008 * np.sin(np.linspace(0, (2.0 + row) * np.pi, len(trace_x)) + row), color=color, lw=0.75)

    box(0.015, 0.34, 0.145, 0.235, "Pattern Bank v2", "20 nested normal motifs\ncontext-only incidents\n15,000 windows\nfor denoising", COLORS["gold"], body_size=5.0)
    motif_x = np.linspace(0.032, 0.142, 48)
    ax.plot(motif_x, 0.365 + 0.011 * np.sin(np.linspace(0, 4 * np.pi, len(motif_x))), color=COLORS["gold"], lw=0.8)

    box(0.195, 0.57, 0.145, 0.205, "Balanced hybrid", "45k real + 15k synthetic\nfixed 60k-window budget\n5 epochs, three seeds", COLORS["purple"], body_size=5.25)
    box(0.195, 0.25, 0.145, 0.235, "Pattern-TShape", "patch CNN + local/global\nattention + gated head\ncheckpoint theta_PB(-q)", COLORS["purple"], body_size=5.2)
    arrow(0.16, 0.75, 0.195, 0.69, COLORS["blue"])
    arrow(0.16, 0.45, 0.195, 0.63, COLORS["gold"])
    arrow(0.267, 0.57, 0.267, 0.485, COLORS["purple"])

    box(0.375, 0.68, 0.125, 0.205, "Input", "unseen KPI history x^q\nno target label\nno target gradient", COLORS["teal"])
    input_x = np.linspace(0.39, 0.485, 60)
    signal = 0.695 + 0.008 * np.sin(np.linspace(0, 5 * np.pi, 60))
    signal[34:38] += np.array([0.002, 0.012, 0.006, -0.004])
    ax.plot(input_x, signal, color=COLORS["teal"], lw=0.9)
    box(0.535, 0.68, 0.17, 0.205, "Preprocessing P", "first difference + output pad\nhistory-fitted scaling\np=16 aligned windows", COLORS["blue"])
    arrow(0.50, 0.78, 0.535, 0.78, COLORS["teal"])

    box(0.375, 0.37, 0.145, 0.205, "Neural channel", "frozen theta_PB(-q)\nnext-value residual\nT_PB(t)", COLORS["purple"])
    box(0.56, 0.37, 0.145, 0.205, "Reliability guard", "rolling median M(t)\nspectral S(t) | MAD D(t)\nG=(.55M+.25S+.05D)/.85", COLORS["gold"], body_size=5.35)
    arrow(0.62, 0.68, 0.45, 0.575, COLORS["purple"])
    arrow(0.62, 0.68, 0.635, 0.575, COLORS["gold"])
    arrow(0.34, 0.365, 0.375, 0.47, COLORS["purple"])
    box(0.425, 0.11, 0.235, 0.16, "Channel normalization", "N(T_PB) and G remain separately inspectable", COLORS["teal"], title_size=6.6, body_size=5.5)
    arrow(0.447, 0.37, 0.50, 0.27, COLORS["purple"])
    arrow(0.635, 0.37, 0.59, 0.27, COLORS["gold"])

    fusion_policy = "\n".join(
        [
            r"$\alpha_q\in\{0,.05,\ldots,.50\}$",
            "selected on the other five datasets",
            r"future-unseen release: $\alpha=.05$",
        ]
    )
    fusion_equation = "\n".join(
        [
            r"$s^+_{t,q}=\alpha_qN(T^{PB}_{t,q})$",
            r"$\qquad +(1-\alpha_q)G_{t,q}$",
        ]
    )
    box(0.75, 0.68, 0.225, 0.205, "Source-only fusion policy", fusion_policy, COLORS["blue"])
    box(0.75, 0.40, 0.225, 0.18, "TShape-Zero+ score", fusion_equation, COLORS["coral"], title_size=7.1, body_size=5.5)
    arrow(0.66, 0.19, 0.75, 0.47, COLORS["teal"])
    arrow(0.86, 0.68, 0.86, 0.58, COLORS["blue"])

    box(
        0.75,
        0.11,
        0.225,
        0.18,
        "Open-box response",
        "aligned score | channel attribution\ntop-k indices | CLI | web | JSON",
        COLORS["green"],
        title_size=7.0,
        body_size=5.6,
    )
    arrow(0.86, 0.40, 0.86, 0.29, COLORS["green"])

    ax.text(0.015, 0.035, "Source histories -> checkpoint", fontsize=5.15, color=COLORS["purple"], weight="bold")
    ax.text(0.375, 0.035, "Unlabeled history -> aligned scores", fontsize=5.15, color=COLORS["teal"], weight="bold")
    ax.text(0.75, 0.035, "Labels choose neither weights nor alarms", fontsize=5.15, color=COLORS["gray"], weight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.012, top=0.988)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(
                FIGURES / f"fig_tshape_zero_plus_architecture_wide.{suffix}",
                dpi=360,
                bbox_inches=None,
                pad_inches=0,
            )
    plt.close(fig)


def draw_reuse_gap_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.82))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def panel(x: float, color: str, title: str, subtitle: str, shifted: bool) -> None:
        box = mpl.patches.FancyBboxPatch(
            (x, 0.49),
            0.43,
            0.47,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc="white",
            ec=color,
            lw=1.35,
        )
        ax.add_patch(box)
        ax.text(x + 0.025, 0.918, title, color=color, fontsize=6.8, weight="bold", va="top")
        ax.text(x + 0.025, 0.872, subtitle, color=COLORS["gray"], fontsize=5.25, va="top")
        trace_x = np.linspace(x + 0.028, x + 0.185, 38)
        for row in range(3):
            center = 0.805 - 0.058 * row
            phase = 0.7 * row
            if shifted:
                signal = 0.014 * np.sin(np.linspace(0, (4.5 + row) * np.pi, 38) + phase)
                if row == 1:
                    signal[19:23] += np.array([0.01, 0.035, 0.02, -0.012])
            else:
                signal = 0.014 * np.sin(np.linspace(0, (2.0 + 0.4 * row) * np.pi, 38) + phase)
            ax.plot(trace_x, center + signal, color=color, lw=0.85)
            ax.plot([trace_x[0], trace_x[-1]], [center - 0.025, center - 0.025], color="#DCE3E8", lw=0.35)
        grid_x, grid_y, cell = x + 0.235, 0.687, 0.029
        for i in range(5):
            for j in range(5):
                distance = abs(i - j)
                value = (0.80 - 0.14 * distance) if not shifted else (0.56 - 0.05 * distance + 0.14 * ((2 * i + j) % 3 == 0))
                rgba = mpl.colors.to_rgba(color, alpha=max(0.10, min(0.78, value)))
                ax.add_patch(mpl.patches.Rectangle((grid_x + j * cell, grid_y + (4 - i) * cell), cell, cell, fc=rgba, ec="white", lw=0.25))
        ax.text(grid_x + 0.072, 0.665, "attention map", ha="center", fontsize=4.65, color=COLORS["gray"])
        density_x = np.linspace(x + 0.045, x + 0.39, 100)
        if shifted:
            normal = np.exp(-0.5 * ((density_x - (x + 0.205)) / 0.057) ** 2)
            anomaly = np.exp(-0.5 * ((density_x - (x + 0.245)) / 0.067) ** 2)
        else:
            normal = np.exp(-0.5 * ((density_x - (x + 0.155)) / 0.042) ** 2)
            anomaly = np.exp(-0.5 * ((density_x - (x + 0.305)) / 0.043) ** 2)
        ax.plot(density_x, 0.535 + 0.070 * normal, color=COLORS["teal"], lw=1.0)
        ax.plot(density_x, 0.535 + 0.070 * anomaly, color=COLORS["purple"], lw=1.0, ls="--")
        ax.plot([density_x[0], density_x[-1]], [0.535, 0.535], color="#AEB8C2", lw=0.45)
        residual_label = "residual overlap" if shifted else "residual separation"
        ax.text(x + 0.217, 0.507, residual_label, ha="center", fontsize=4.65, color=COLORS["gray"])

    panel(0.01, COLORS["teal"], "SOURCE CHECKPOINT", "Recurring source KPI shapes", shifted=False)
    panel(0.56, COLORS["coral"], "UNSEEN TARGET SHIFT", "Changed trend, period, and noise", shifted=True)
    ax.annotate("", xy=(0.555, 0.77), xytext=(0.445, 0.77), arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": COLORS["ink"]})
    ax.text(0.50, 0.805, "same\nAPI", ha="center", va="center", fontsize=4.5, weight="bold", color=COLORS["ink"], bbox={"fc": "white", "ec": "none", "pad": 0.3})
    ax.text(0.50, 0.455, "SAME API  $\\ne$  SAME ANOMALY RANKING", ha="center", fontsize=6.0, weight="bold", color=COLORS["ink"])

    nodes = [
        (0.01, 0.21, COLORS["purple"], "Pattern-TShape", "equal sources\n+ nested bank v2"),
        (0.25, 0.23, COLORS["gold"], "Residual guard", "median | spectral\n| rolling MAD"),
        (0.51, 0.22, COLORS["teal"], "Normalize + fuse", "channel min-max\nsource-only $\\alpha$"),
        (0.76, 0.23, COLORS["green"], "Open-box output", "score trace\n+ attribution"),
    ]
    for x, width, color, title, body in nodes:
        node = mpl.patches.FancyBboxPatch(
            (x, 0.115),
            width,
            0.245,
            boxstyle="round,pad=0.010,rounding_size=0.022",
            fc=mpl.colors.to_rgba(color, 0.07),
            ec=color,
            lw=1.15,
        )
        ax.add_patch(node)
        ax.text(x + width / 2, 0.285, title, ha="center", fontsize=4.85, weight="bold", color=COLORS["ink"])
        ax.text(x + width / 2, 0.205, body, ha="center", va="center", fontsize=4.15, color=COLORS["gray"], linespacing=1.15)
    for start, end in ((0.22, 0.25), (0.48, 0.51), (0.73, 0.76)):
        ax.annotate("", xy=(end, 0.235), xytext=(start, 0.235), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": COLORS["ink"]})
    ax.text(0.50, 0.055, "Repair is inspectable: the learned channel and reliability floor remain separate.", ha="center", fontsize=5.2, color=COLORS["gray"])
    save_figure(fig, "fig02_easytsad_reuse_gap")


def grouped_series(
    rows: Iterable[Mapping[str, str]], method: str, metric: str
) -> Dict[Tuple[str, str], float]:
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        if row.get("method") != method:
            continue
        value = fnum(row.get(metric))
        if np.isfinite(value):
            values[(str(row["dataset"]), str(row["series"]))].append(value)
    return {key: float(np.mean(group)) for key, group in values.items()}


def ledger_map(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], Mapping[str, str]]:
    return {(str(row["method_id"]), str(row["dataset"])): row for row in rows}


def macro(ledger: Mapping[Tuple[str, str], Mapping[str, str]], method: str, field: str) -> float:
    return float(np.mean([fnum(ledger[(method, dataset)][field]) for dataset in DATASETS]))


def draw_protocol_radar(ledger: Sequence[Mapping[str, str]]) -> None:
    original_rows = read_rows(RESULTS / "original_reported_results.csv")
    original = {row["dataset"]: row for row in original_rows}
    replay_rows = read_rows(RESULTS / "tshape_faithful_reproduction_summary_metrics.csv")
    replay = {row["dataset"]: row for row in replay_rows}
    lookup = ledger_map(ledger)
    angles = np.linspace(0, 2 * np.pi, len(DATASETS), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    specs = [
        ("Event-F1 (F1-E)", "f1_event", "event_f1"),
        ("Point-F1 (F1)", "f1", "point_f1"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.92), subplot_kw={"projection": "polar"})
    styles = [
        ("Original reported", COLORS["blue"], "--", "o"),
        ("Reproduction / replay", COLORS["teal"], "-", "s"),
        ("Strict Pattern checkpoint", COLORS["coral"], "-", "^"),
    ]
    for panel, (ax, (title, original_field, ledger_field)) in enumerate(zip(axes, specs)):
        reported = [fnum(original.get(dataset, {}).get(original_field)) for dataset in DATASETS]
        replay_field = "event_f1_pa_log" if ledger_field == "event_f1" else "point_f1_pa"
        reproduced = [fnum(replay.get(dataset, {}).get(replay_field)) for dataset in DATASETS]
        direct = [fnum(lookup[(PATTERN_METHOD, dataset)][ledger_field]) for dataset in DATASETS]
        for values, (label, color, linestyle, marker) in zip((reported, reproduced, direct), styles):
            closed = np.r_[values, values[0]]
            ax.plot(closed_angles, closed, color=color, lw=1.45, ls=linestyle, marker=marker, ms=3.0, label=label)
            if label != "Original reported":
                ax.fill(closed_angles, closed, color=color, alpha=0.055)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(DATASETS)
        ax.tick_params(axis="x", labelsize=5.2, pad=0.5)
        ax.set_ylim(0.30, 1.0)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([".4", ".6", ".8", "1.0"])
        ax.tick_params(axis="y", labelsize=4.8, pad=0.0)
        ax.grid(color="#C9D1D8", lw=0.55, alpha=0.8)
        ax.spines["polar"].set_color("#98A3AD")
        ax.set_title(f"({chr(97 + panel)}) {title}", pad=7, fontsize=6.3, weight="bold")
        ax.text(angles[-1], 0.955, "N/A", color=COLORS["gray"], fontsize=4.6, ha="center", va="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=5.1, handlelength=1.5, columnspacing=0.7)
    aiops_replay = replay.get("AIOPS", {}).get("series", "0")
    fig.text(
        0.99,
        0.005,
        f"Original: Yahoo N/A; AIOPS reproduction: {aiops_replay}/29.",
        ha="right",
        color=COLORS["gray"],
        fontsize=4.4,
    )
    fig.subplots_adjust(wspace=0.56, bottom=0.28, top=0.82, left=0.04, right=0.96)
    save_figure(fig, "fig03_easytsad_protocol_radar")


def draw_overall(ledger_rows: Sequence[Mapping[str, str]]) -> None:
    lookup = ledger_map(ledger_rows)
    selected = [
        ("Rolling median", "rolling_median"),
        ("Chronos", "Chronos-Bolt-Tiny"),
        ("Timer", "Timer-base-84m"),
        ("TimesFM", "TimesFM-2.5-200M"),
        ("Sundial", "Sundial-base-128m"),
        ("FCVAE adapter", "FCVAEAdapter"),
        ("Pattern-only Zero", PATTERN_METHOD),
        ("Residual guard", GUARD_METHOD),
        ("TShape-Zero+", PROPOSED_METHOD),
    ]
    points = np.array([macro(lookup, method, "point_f1") for _, method in selected])
    events = np.array([macro(lookup, method, "event_f1") for _, method in selected])
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for yi, pvalue, evalue in zip(y, points, events):
        ax.plot([pvalue, evalue], [yi, yi], color="#CDD4DA", lw=1.5, zorder=1)
    ax.scatter(points, y, s=27, color=COLORS["teal"], label="Point-F1", zorder=3)
    ax.scatter(events, y, s=27, color=COLORS["coral"], marker="D", label="Event-F1", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in selected])
    ax.invert_yaxis()
    ax.set_xlim(0.35, 1.01)
    ax.set_xlabel("Six-dataset macro F1")
    ax.xaxis.grid(True, color="#E2E6EA", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.35, columnspacing=0.8)
    ax.set_title("Unified EasyTSAD comparison", loc="left", weight="bold")
    save_figure(fig, "fig06_easytsad_overall")


def draw_pattern_bank() -> None:
    rows = read_rows(RESULTS / "pattern_bank_hybrid_summary_metrics_submission_easytsad_pattern_hybrid_v2.csv")
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.72), sharex=True, sharey=False)
    for ax, dataset in zip(axes.flat, DATASETS):
        channel = [
            row
            for row in rows
            if row["dataset"] == dataset and row["method"].startswith("TShapeHybridPattern-")
        ]
        final = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["method"].startswith("TShapeHybridPatternZeroPlus-")
        ]
        channel.sort(key=lambda row: fnum(row.get("fraction")))
        final.sort(key=lambda row: fnum(row.get("fraction")))
        x = np.array([100 * fnum(row["fraction"]) for row in channel])
        channel_point = np.array([fnum(row["point_f1_pa"]) for row in channel])
        channel_event = np.array([fnum(row["event_f1_pa_log"]) for row in channel])
        final_point = np.array([fnum(row["point_f1_pa"]) for row in final])
        final_event = np.array([fnum(row["event_f1_pa_log"]) for row in final])
        ax.plot(x, final_point, color=COLORS["teal"], marker="o", ms=2.3, lw=1.15, label="Final P")
        ax.plot(x, final_event, color=COLORS["coral"], marker="s", ms=2.1, lw=1.1, label="Final E")
        ax.plot(x, channel_point, color=COLORS["teal"], lw=0.8, ls=":", alpha=0.8, label="Shape P")
        ax.plot(x, channel_event, color=COLORS["coral"], lw=0.8, ls=":", alpha=0.8, label="Shape E")
        ax.axvline(10, color=COLORS["purple"], lw=0.75, ls="--", alpha=0.75)
        ax.text(0.04, 0.05, "source-CV", transform=ax.transAxes, color=COLORS["purple"], fontsize=4.5, weight="bold")
        ax.set_title(dataset, fontsize=6.2, weight="bold", pad=2)
        ax.set_xticks([10, 50, 80, 100])
        ax.tick_params(axis="both", labelsize=5.2, pad=1)
        panel_values = np.concatenate([channel_point, channel_event, final_point, final_event])
        lo = max(0.0, float(np.nanmin(panel_values)) - 0.035)
        hi = min(1.01, float(np.nanmax(panel_values)) + 0.025)
        if hi - lo < 0.12:
            center = 0.5 * (hi + lo)
            lo, hi = max(0.0, center - 0.06), min(1.01, center + 0.06)
        ax.set_ylim(lo, hi)
        ax.yaxis.grid(True, color="#E3E7EA", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 1].set_xlabel("Pattern Bank coverage (\%)", fontsize=5.8)
    axes[0, 0].set_ylabel("EasyTSAD F1", fontsize=5.8)
    axes[1, 0].set_ylabel("EasyTSAD F1", fontsize=5.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=4.9, handlelength=1.4, columnspacing=0.55)
    fig.subplots_adjust(wspace=0.18, hspace=0.27, bottom=0.20, left=0.12, right=0.99, top=0.96)
    save_figure(fig, "fig08_easytsad_pattern_bank")


def draw_guard_lift(ledger_rows: Sequence[Mapping[str, str]]) -> None:
    lookup = ledger_map(ledger_rows)
    columns = [
        ("P: final-pattern", "point_f1", PATTERN_METHOD),
        ("E: final-pattern", "event_f1", PATTERN_METHOD),
        ("P: final-guard", "point_f1", GUARD_METHOD),
        ("E: final-guard", "event_f1", GUARD_METHOD),
    ]
    values = np.zeros((len(DATASETS), len(columns)))
    for i, dataset in enumerate(DATASETS):
        for j, (_, field, baseline) in enumerate(columns):
            values[i, j] = fnum(lookup[(PROPOSED_METHOD, dataset)][field]) - fnum(lookup[(baseline, dataset)][field])
    limit = max(0.01, float(np.max(np.abs(values))))
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=6.2, color="#111827")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([label for label, _, _ in columns], rotation=28, ha="right")
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title("What the guard fixes, and what TShape adds", loc="left", weight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("F1 difference", labelpad=2)
    save_figure(fig, "fig09_easytsad_guard_lift")


def draw_ablation() -> None:
    rows: List[Dict[str, str]] = []
    for filename in (
        "baseline_summary_metrics.csv",
        "tshape_summary_metrics_submission_easytsad_tshape_balanced_full3.csv",
        "pattern_bank_tshape_summary_metrics_submission_strict_zero_main_full.csv",
    ):
        rows.extend(read_rows(RESULTS / filename))
    latest = {(row["method"], row["dataset"]): row for row in rows}
    names = {
        "rolling_median": "Rolling median only",
        "spectral_residual": "Spectral residual only",
        GUARD_METHOD: "Residual guard only",
        LODO_METHOD: "LODO TShape diagnostic",
        PATTERN_METHOD: "Pattern-only strict checkpoint",
        PROPOSED_METHOD: "TShape-Zero+ ($\\alpha=0.15$)",
    }
    order = list(names)
    point, event = [], []
    for method in order:
        subset = [latest[(method, dataset)] for dataset in DATASETS]
        point.append(np.mean([fnum(row["point_f1_pa"]) for row in subset]))
        event.append(np.mean([fnum(row["event_f1_pa_log"]) for row in subset]))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.5, 2.85))
    for yi, pvalue, evalue in zip(y, point, event):
        ax.plot([pvalue, evalue], [yi, yi], color="#D7DCE1", lw=1.4)
    ax.scatter(point, y, s=24, color=COLORS["teal"], label="Point-F1", zorder=3)
    ax.scatter(event, y, s=24, marker="D", color=COLORS["coral"], label="Event-F1", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[method] for method in order])
    ax.invert_yaxis()
    finite = np.asarray(point + event, dtype=float)
    lo = max(0.0, float(np.nanmin(finite)) - 0.025)
    hi = min(1.01, float(np.nanmax(finite)) + 0.025)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Six-dataset macro F1")
    ax.xaxis.grid(True, color="#E3E7EA", lw=0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.7)
    ax.set_title("Fusion and channel ablation", loc="left", weight="bold")
    save_figure(fig, "fig10_easytsad_ablation")


def method_detail_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for filename in (
        "baseline_series_metrics.csv",
        "tshape_series_metrics_submission_easytsad_tshape_full3.csv",
        "tshape_series_metrics_submission_easytsad_tshape_balanced_full3.csv",
        "tshape_series_metrics_submission_easytsad_tshape_pattern_v2_full3.csv",
        "pattern_v2_zero_plus_fusion_cv_series.csv",
        "pattern_bank_sourcecv_series.csv",
        "pattern_bank_tshape_series_metrics_submission_strict_zero_main_full.csv",
    ):
        loaded = read_rows(RESULTS / filename)
        if filename == "tshape_series_metrics_submission_easytsad_tshape_balanced_full3.csv":
            loaded = [row for row in loaded if "pattern" not in row.get("method", "")]
        rows.extend(loaded)
    return rows


def draw_violin(detail: Sequence[Mapping[str, str]]) -> None:
    final = grouped_series(detail, PROPOSED_METHOD, "point_f1_pa")
    data = [[value for (dataset, _), value in final.items() if dataset == name] for name in DATASETS]
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.78)
    for body, dataset in zip(parts["bodies"], DATASETS):
        body.set_facecolor(DATASET_COLORS[dataset])
        body.set_edgecolor("white")
        body.set_alpha(0.75)
    parts["cmedians"].set_color(COLORS["ink"])
    parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks(np.arange(1, len(DATASETS) + 1))
    ax.set_xticklabels(DATASETS, rotation=25, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("TShape-Zero+ Point-F1")
    ax.yaxis.grid(True, color="#E3E7EA", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Series-level heterogeneity", loc="left", weight="bold")
    save_figure(fig, "fig11_easytsad_violin")


def draw_scatter(detail: Sequence[Mapping[str, str]]) -> None:
    direct = grouped_series(detail, PATTERN_METHOD, "event_f1_pa_log")
    final = grouped_series(detail, PROPOSED_METHOD, "event_f1_pa_log")
    ratios: Dict[Tuple[str, str], float] = {}
    for row in detail:
        key = (str(row.get("dataset")), str(row.get("series")))
        value = fnum(row.get("anomaly_ratio"))
        if np.isfinite(value):
            ratios[key] = value
    keys = sorted(set(direct) & set(final) & set(ratios))
    x = np.array([max(ratios[key], 1e-6) for key in keys])
    y = np.array([final[key] - direct[key] for key in keys])
    rho, pvalue = spearmanr(np.log10(x), y)
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    for dataset in DATASETS:
        mask = np.array([key[0] == dataset for key in keys])
        ax.scatter(x[mask], y[mask], s=8, alpha=0.48, color=DATASET_COLORS[dataset], label=dataset, edgecolors="none")
    ax.axhline(0, color="#4B5563", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Anomaly ratio (log scale)")
    ax.set_ylabel("Event-F1 lift: final - direct")
    ax.yaxis.grid(True, color="#E3E7EA", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.96, f"Spearman $\\rho$={rho:.2f}, $p$={pvalue:.2g}", transform=ax.transAxes, va="top", fontsize=6.3)
    ax.legend(frameon=False, ncol=3, loc="lower right", handletextpad=0.2, columnspacing=0.55)
    ax.set_title("Where the repair changes event ranking", loc="left", weight="bold")
    save_figure(fig, "fig12_easytsad_scatter")


def draw_attribution(detail: Sequence[Mapping[str, str]]) -> None:
    method_names = [
        ("Pattern-TShape", PATTERN_METHOD),
        ("Median", "rolling_median"),
        ("Spectral", "spectral_residual"),
        ("Guard", GUARD_METHOD),
    ]
    values = {
        label: {
            key: 0.5 * point + 0.5 * grouped_series(detail, method, "event_f1_pa_log").get(key, np.nan)
            for key, point in grouped_series(detail, method, "point_f1_pa").items()
        }
        for label, method in method_names
    }
    final_event = grouped_series(detail, PROPOSED_METHOD, "event_f1_pa_log")
    guard_event = grouped_series(detail, GUARD_METHOD, "event_f1_pa_log")
    fractions = np.zeros((len(DATASETS), len(method_names)))
    win_rates = np.zeros(len(DATASETS))
    for i, dataset in enumerate(DATASETS):
        keys = sorted(set.intersection(*(set(channel) for channel in values.values())))
        keys = [key for key in keys if key[0] == dataset]
        counts = np.zeros(len(method_names))
        for key in keys:
            scores = np.array([values[label][key] for label, _ in method_names])
            counts[int(np.nanargmax(scores))] += 1
        fractions[i] = counts / max(np.sum(counts), 1)
        paired = [key for key in keys if key in final_event and key in guard_event]
        win_rates[i] = np.mean([final_event[key] > guard_event[key] + 1e-12 for key in paired]) if paired else np.nan
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))
    channel_colors = [COLORS["purple"], COLORS["teal"], COLORS["gold"], COLORS["gray"]]
    for j, ((label, _), color) in enumerate(zip(method_names, channel_colors)):
        ax.bar(x, fractions[:, j], bottom=bottom, width=0.68, color=color, label=label, edgecolor="white", linewidth=0.3)
        bottom += fractions[:, j]
    ax.plot(x, win_rates, color=COLORS["coral"], marker="D", ms=3.2, lw=1.2, label="Final beats guard")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Series fraction")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E3E7EA", lw=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=0.55, handletextpad=0.25)
    ax.set_title("Best evidence channel and neural marginal value", loc="left", weight="bold")
    save_figure(fig, "fig13_easytsad_attribution")


def series_characteristics(dataset: str, series: str) -> Dict[str, float]:
    path = ROOT / "datasets" / "UTS" / dataset / series
    values = np.asarray(np.load(path / "test.npy"), dtype=np.float64).reshape(-1)
    labels = np.asarray(np.load(path / "test_label.npy"), dtype=np.int8).reshape(-1)
    if values.size > 4096:
        indices = np.linspace(0, values.size - 1, 4096, dtype=np.int64)
        values = values[indices]
    finite_fill = float(np.nanmedian(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0
    values = np.nan_to_num(values, nan=finite_fill, posinf=finite_fill, neginf=finite_fill)
    centered = values - np.mean(values)
    scale = float(np.std(centered))
    if scale < 1e-12 or values.size < 8:
        trend = periodicity = roughness = 0.0
    else:
        time_axis = np.linspace(-1.0, 1.0, values.size)
        trend = abs(float(np.corrcoef(time_axis, centered)[0, 1]))
        candidates = [lag for lag in (4, 8, 16, 24, 32, 48, 64, 96, 128) if lag < values.size // 3]
        correlations = []
        for lag in candidates:
            left, right = centered[:-lag], centered[lag:]
            denom = float(np.std(left) * np.std(right))
            correlations.append(float(np.mean(left * right) / denom) if denom > 1e-12 else 0.0)
        periodicity = max([0.0] + correlations)
        diff = np.diff(values)
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        roughness = float(np.median(np.abs(diff - np.median(diff))) / max(iqr, 1e-12))
    return {
        "Anomaly ratio": float(np.mean(labels)) if labels.size else 0.0,
        "Trend strength": trend,
        "Periodicity": periodicity,
        "Roughness": roughness,
    }


def draw_feature_attribution(detail: Sequence[Mapping[str, str]]) -> None:
    final_point = grouped_series(detail, PROPOSED_METHOD, "point_f1_pa")
    final_event = grouped_series(detail, PROPOSED_METHOD, "event_f1_pa_log")
    guard_point = grouped_series(detail, GUARD_METHOD, "point_f1_pa")
    guard_event = grouped_series(detail, GUARD_METHOD, "event_f1_pa_log")
    keys = sorted(set(final_point) & set(final_event) & set(guard_point) & set(guard_event))
    features = ("Anomaly ratio", "Trend strength", "Periodicity", "Roughness")
    groups = DATASETS + ("All",)
    values = np.full((len(features), len(groups)), np.nan, dtype=float)
    pvalues = np.full_like(values, np.nan)
    counts = np.zeros_like(values, dtype=int)
    feature_cache = {key: series_characteristics(*key) for key in keys}
    gains = {
        key: 0.5 * ((final_point[key] - guard_point[key]) + (final_event[key] - guard_event[key]))
        for key in keys
    }
    rows: List[Dict[str, object]] = []
    for j, group in enumerate(groups):
        selected = keys if group == "All" else [key for key in keys if key[0] == group]
        for i, feature in enumerate(features):
            x = np.array([feature_cache[key][feature] for key in selected], dtype=float)
            y = np.array([gains[key] for key in selected], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(finite)) >= 5 and np.unique(x[finite]).size > 1:
                rho, pvalue = spearmanr(x[finite], y[finite])
                values[i, j] = float(rho)
                pvalues[i, j] = float(pvalue)
                counts[i, j] = int(np.sum(finite))
            rows.append(
                {
                    "feature": feature,
                    "dataset": group,
                    "series": counts[i, j],
                    "spearman_rho": values[i, j],
                    "p_value": pvalues[i, j],
                }
            )
    out_path = RESULTS / "submission_easytsad_feature_attribution.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "dataset", "series", "spearman_rho", "p_value"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for i in range(len(features)):
        for j in range(len(groups)):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            color = COLORS["teal"] if value >= 0 else COLORS["coral"]
            ax.scatter(j, i, s=18 + 310 * abs(value), color=color, alpha=0.78, edgecolor="white", linewidth=0.5)
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5.0, color="white" if abs(value) > 0.22 else COLORS["ink"], weight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=28, ha="right")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(len(features) - 0.55, -0.55)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Which KPI traits expose neural marginal value?", loc="left", weight="bold")
    ax.text(0.99, 0.02, "teal: positive   coral: negative", transform=ax.transAxes, ha="right", fontsize=5.5, color=COLORS["gray"])
    save_figure(fig, "fig16_easytsad_feature_attribution")


def draw_training() -> None:
    runs = json.loads((RESULTS / "tshape_run_info_submission_easytsad_tshape_full3.json").read_text())
    target_runs = [run for run in runs if run.get("method") == "TShape-target_trained"]
    by_dataset: Dict[str, List[np.ndarray]] = defaultdict(list)
    for run in target_runs:
        loss = np.asarray(run.get("loss", []), dtype=float)
        if loss.size and loss[0] > 0:
            by_dataset[str(run["target"])].append(loss / loss[0])
    fig = plt.figure(figsize=(3.5, 4.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.48, wspace=0.18)
    ax = fig.add_subplot(grid[0, :])
    epochs = np.arange(1, 6)
    for dataset in DATASETS:
        traces = by_dataset.get(dataset, [])
        if not traces:
            continue
        for trace in traces:
            ax.plot(epochs, trace, color=DATASET_COLORS[dataset], alpha=0.16, lw=0.7)
        ax.plot(epochs, np.mean(np.stack(traces), axis=0), color=DATASET_COLORS[dataset], marker="o", ms=2.5, lw=1.25, label=dataset)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss / epoch-1 loss")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color="#E3E7EA", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right", columnspacing=0.6, handletextpad=0.25)
    ax.set_title("(a) Shared target-context optimization converges", loc="left", weight="bold")

    diagnostic_rows = read_rows(
        RESULTS / "target_diagnostic_summary_submission_easytsad_target_diagnostics.csv"
    )
    diagnostic = {
        (row["variant"], row["dataset"]): row
        for row in diagnostic_rows
    }
    variants = ("diff1_p16", "raw_p16", "diff1_p32")
    variant_labels = (r"$\Delta$, $p=16$", r"raw, $p=16$", r"$\Delta$, $p=32$")
    for panel, (metric, title) in enumerate(
        (("point_f1_pa", "(b) Point-F1"), ("event_f1_pa_log", "(c) Event-F1"))
    ):
        heat = np.array(
            [[fnum(diagnostic[(variant, dataset)].get(metric)) for dataset in DATASETS] for variant in variants],
            dtype=float,
        )
        hax = fig.add_subplot(grid[1, panel])
        image = hax.imshow(heat, cmap="YlGnBu", vmin=0.30, vmax=1.00, aspect="auto")
        for row_i in range(heat.shape[0]):
            for col_i in range(heat.shape[1]):
                value = heat[row_i, col_i]
                hax.text(
                    col_i,
                    row_i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.7,
                    color="white" if value > 0.72 else COLORS["ink"],
                )
        hax.set_xticks(range(len(DATASETS)))
        hax.set_xticklabels(DATASETS, rotation=55, ha="right", fontsize=5.0)
        hax.set_yticks(range(len(variants)))
        hax.set_yticklabels(variant_labels if panel == 0 else [], fontsize=5.6)
        hax.tick_params(length=0, pad=1)
        hax.spines[:].set_visible(False)
        hax.set_title(title, loc="left", weight="bold", fontsize=7.1)
    colorbar = fig.colorbar(image, ax=fig.axes[1:3], orientation="horizontal", fraction=0.055, pad=0.20)
    colorbar.ax.tick_params(labelsize=5.2, length=2)
    colorbar.set_label("EasyTSAD score", fontsize=6.0)
    save_figure(fig, "fig14_easytsad_training")


def write_efficiency_ledger() -> None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    checkpoint_specs = [
        ("TShape-Zero+", "TShape-zero_plus", ROOT / "models" / "tshape_zero_plus_release.pt", "released checkpoint"),
        ("Chronos-Bolt-Tiny", "Chronos-Bolt-Tiny", cache_root / "models--amazon--chronos-bolt-tiny", "downloaded checkpoint cache"),
        ("Timer-base-84M", "Timer-base-84m", cache_root / "models--thuml--timer-base-84m", "downloaded checkpoint cache"),
        ("Sundial-base-128M", "Sundial-base-128m", cache_root / "models--thuml--sundial-base-128m", "downloaded checkpoint cache"),
        ("TimesFM-2.5-200M", "TimesFM-2.5-200M", cache_root / "models--google--timesfm-2.5-200m-pytorch", "downloaded checkpoint cache"),
    ]

    def footprint(path: Path) -> int:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            return int(path.stat().st_size)
        resolved = {}
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            target = candidate.resolve()
            resolved[str(target)] = int(target.stat().st_size)
        return int(sum(resolved.values()))

    rows = []
    for method, source_method, path, scope in checkpoint_specs:
        size = footprint(path)
        rows.append(
            {
                "method": method,
                "source_method": source_method,
                "footprint_bytes": size,
                "footprint_mib": f"{size / (1024 * 1024):.6f}",
                "footprint_scope": scope,
            }
        )
    with (RESULTS / "submission_efficiency_frontier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "source_method", "footprint_bytes", "footprint_mib", "footprint_scope"],
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_efficiency(ledger_rows: Sequence[Mapping[str, str]]) -> None:
    lookup = ledger_map(ledger_rows)
    footprint_rows = read_rows(RESULTS / "submission_efficiency_frontier.csv")
    footprints = {row["source_method"]: fnum(row["footprint_mib"]) for row in footprint_rows}
    methods = [
        ("TShape-Zero+", PROPOSED_METHOD, "TShape-zero_plus"),
        ("Chronos", "Chronos-Bolt-Tiny", "Chronos-Bolt-Tiny"),
        ("Timer", "Timer-base-84m", "Timer-base-84m"),
        ("Sundial", "Sundial-base-128m", "Sundial-base-128m"),
        ("TimesFM", "TimesFM-2.5-200M", "TimesFM-2.5-200M"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for i, (label, method, footprint_id) in enumerate(methods):
        x = footprints[footprint_id]
        pvalue = macro(lookup, method, "point_f1")
        evalue = macro(lookup, method, "event_f1")
        color = [
            COLORS["purple"],
            COLORS["teal"],
            COLORS["blue"],
            COLORS["gold"],
            COLORS["coral"],
            COLORS["green"],
        ][i]
        ax.scatter(x, pvalue, s=42 + 100 * evalue, color=color, alpha=0.82, edgecolor="white", linewidth=0.7)
        ax.annotate(label, (x, pvalue), xytext=(3, 3), textcoords="offset points", fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlabel("Checkpoint footprint (MiB, log scale)")
    ax.set_ylabel("Six-dataset macro Point-F1")
    ax.xaxis.grid(True, color="#E3E7EA", lw=0.55)
    ax.yaxis.grid(True, color="#E3E7EA", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Effectiveness versus distribution footprint", loc="left", weight="bold")
    ax.text(0.98, 0.04, "Bubble area: Event-F1", transform=ax.transAxes, ha="right", color=COLORS["gray"], fontsize=6.0)
    save_figure(fig, "fig15_easytsad_efficiency")


def tex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def write_latex_assets(ledger_rows: Sequence[Mapping[str, str]]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    lookup = ledger_map(ledger_rows)
    methods: List[Tuple[str, str, str, str, str]] = []
    seen = set()
    for row in ledger_rows:
        if row["family"] == "Residual" or row["method_id"] in FOCAL_EXCLUDED_METHODS:
            continue
        method_id = row["method_id"]
        if method_id in seen:
            continue
        seen.add(method_id)
        methods.append(
            (row["family"], row["method"], method_id, row["status"], row["access"])
        )
    rankable_access = {"Frozen", "Strict"}
    ranks: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for dataset in DATASETS:
        for field in ("point_f1", "event_f1"):
            values = sorted(
                {
                    fnum(lookup[(method_id, dataset)][field])
                    for _, _, method_id, _, access in methods
                    if access in rankable_access
                    and np.isfinite(fnum(lookup[(method_id, dataset)][field]))
                },
                reverse=True,
            )
            ranks[(dataset, field)] = (
                values[0],
                values[1] if len(values) > 1 else float("nan"),
            )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r'\caption{Focal non-residual comparison under one EasyTSAD protocol (Point-F1 / Event-F1). Among the displayed official frozen foundation checkpoints and strict synthetic-only TShape checkpoints, \textcolor{red}{red bold} and \textcolor{blue}{blue underlined} mark the best and second-best score per metric. Adapter, adaptation, diagnostic, and ablation provenance is retained in the public result ledger.}',
        r"\label{tab:all-easytsad}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{0.91}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"Family & Method & AIOPS & NAB & TODS & UCR & WSD & Yahoo \\",
        r"\midrule",
    ]
    previous_family = None
    for family, display, method_id, status, access in methods:
        if previous_family is not None and family != previous_family:
            lines.append(r"\midrule")
        family_cell = tex_escape(family) if family != previous_family else ""
        method_cell = tex_escape(display)
        if method_id == PROPOSED_METHOD:
            method_cell = r"\textbf{" + method_cell + r"$^{\star}$}"
        cells = []
        for dataset in DATASETS:
            point = fnum(lookup[(method_id, dataset)]["point_f1"])
            event = fnum(lookup[(method_id, dataset)]["event_f1"])
            ptext = f"{point:.3f}"
            etext = f"{event:.3f}"
            if access in rankable_access:
                point_best, point_second = ranks[(dataset, "point_f1")]
                event_best, event_second = ranks[(dataset, "event_f1")]
                if math.isclose(point, point_best, abs_tol=5e-7):
                    ptext = r"\textcolor{red}{\textbf{" + ptext + "}}"
                elif math.isclose(point, point_second, abs_tol=5e-7):
                    ptext = r"\textcolor{blue}{\underline{" + ptext + "}}"
                if math.isclose(event, event_best, abs_tol=5e-7):
                    etext = r"\textcolor{red}{\textbf{" + etext + "}}"
                elif math.isclose(event, event_second, abs_tol=5e-7):
                    etext = r"\textcolor{blue}{\underline{" + etext + "}}"
            cells.append(f"{ptext} / {etext}")
        lines.append(
            f"{family_cell} & {method_cell} & "
            + " & ".join(cells)
            + r" \\"
        )
        previous_family = family
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\renewcommand{\arraystretch}{1.0}",
            r"\end{table*}",
        ]
    )
    (GENERATED / "easytsad_main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    original = {
        row["dataset"]: row
        for row in read_rows(RESULTS / "original_reported_results.csv")
    }
    reproduction = {
        row["dataset"]: row
        for row in read_rows(RESULTS / "tshape_faithful_reproduction_summary_metrics.csv")
    }
    reproduction_lines = [
        r"\begin{table}[t]",
        r"\caption{Claim-separated TShape evidence under the two EasyTSAD metrics. Cells are Point-F1 / Event-F1. Original values are transcribed; reproduction combines stored-score replay and protocol-faithful completion; Strict Zero is one synthetic-only Pattern checkpoint evaluated unchanged on every dataset.}",
        r"\label{tab:tshape-claim-boundary}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Original reported & Reproduction & Strict Zero \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        original_row = original[dataset]
        if np.isfinite(fnum(original_row.get("f1"))):
            original_cell = f"{fnum(original_row['f1']):.3f} / {fnum(original_row['f1_event']):.3f}"
        else:
            original_cell = "--"
        replay = reproduction[dataset]
        replay_cell = f"{fnum(replay['point_f1_pa']):.3f} / {fnum(replay['event_f1_pa_log']):.3f}"
        direct = lookup[(PATTERN_METHOD, dataset)]
        direct_cell = f"{fnum(direct['point_f1']):.3f} / {fnum(direct['event_f1']):.3f}"
        reproduction_lines.append(
            f"{dataset} & {original_cell} & {replay_cell} & {direct_cell} " + r"\\"
        )
    reproduction_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (GENERATED / "easytsad_reproduction_table.tex").write_text(
        "\n".join(reproduction_lines) + "\n", encoding="utf-8"
    )

    macros = {
        "DirectMacroPoint": macro(lookup, PATTERN_METHOD, "point_f1"),
        "DirectMacroEvent": macro(lookup, PATTERN_METHOD, "event_f1"),
        "PatternMacroPoint": macro(lookup, PATTERN_METHOD, "point_f1"),
        "PatternMacroEvent": macro(lookup, PATTERN_METHOD, "event_f1"),
        "FinalMacroPoint": macro(lookup, PROPOSED_METHOD, "point_f1"),
        "FinalMacroEvent": macro(lookup, PROPOSED_METHOD, "event_f1"),
        "GuardMacroPoint": macro(lookup, GUARD_METHOD, "point_f1"),
        "GuardMacroEvent": macro(lookup, GUARD_METHOD, "event_f1"),
    }
    run_info = json.loads(
        (RESULTS / "pattern_bank_tshape_run_info_submission_strict_zero_main_full.json").read_text(
            encoding="utf-8"
        )
    )
    training = run_info[0]
    macros["PatternTrainLossInitial"] = fnum(training["loss"][0])
    macros["PatternTrainLossFinal"] = fnum(training["loss"][-1])
    macros["PatternTrainEpochs"] = float(training["epochs_completed"])
    macro_lines = [f"\\newcommand{{\\{name}}}{{{value:.3f}}}" for name, value in macros.items()]
    for dataset in DATASETS:
        safe = dataset.replace("Yahoo", "Yahoo")
        for prefix, method in (("Direct", PATTERN_METHOD), ("Final", PROPOSED_METHOD)):
            macro_lines.append(f"\\newcommand{{\\{prefix}{safe}Point}}{{{fnum(lookup[(method, dataset)]['point_f1']):.3f}}}")
            macro_lines.append(f"\\newcommand{{\\{prefix}{safe}Event}}{{{fnum(lookup[(method, dataset)]['event_f1']):.3f}}}")
    (GENERATED / "easytsad_numbers.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    paired = read_rows(RESULTS / "submission_easytsad_paired_statistics.csv")
    paired_lookup = {
        (row["dataset"], row["metric"], row["comparator"]): row
        for row in paired
    }
    stat_lines = [
        r"\begin{table}[t]",
        r"\caption{Paired series-level repair effects. Cells are mean difference [bootstrap 95\% interval]. PB: strict Pattern-only TShape-Zero; G: matched residual-only guard.}",
        r"\label{tab:paired-effects}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & $\Delta$Point vs. PB & $\Delta$Event vs. PB & $\Delta$Point vs. G & $\Delta$Event vs. G \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        cells = []
        for metric, comparator in (
            ("Point-F1", "Pattern-only TShape-Zero"),
            ("Event-F1", "Pattern-only TShape-Zero"),
            ("Point-F1", "Residual-Only Guard"),
            ("Event-F1", "Residual-Only Guard"),
        ):
            row = paired_lookup[(dataset, metric, comparator)]
            mean = fnum(row["mean_difference"])
            lo = fnum(row["ci_low"])
            hi = fnum(row["ci_high"])
            cells.append(f"{mean:+.3f} [{lo:+.3f},{hi:+.3f}]")
        stat_lines.append(f"{dataset} & " + " & ".join(cells) + r" \\")
    stat_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    (GENERATED / "easytsad_paired_table.tex").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    ledger_rows = read_rows(RESULTS / "submission_easytsad_all_methods.csv")
    detail = method_detail_rows()
    draw_intro_story()
    draw_zero_plus_framework()
    draw_reuse_gap_mechanism()
    draw_protocol_radar(ledger_rows)
    draw_overall(ledger_rows)
    draw_pattern_bank()
    draw_guard_lift(ledger_rows)
    draw_ablation()
    draw_violin(detail)
    draw_scatter(detail)
    draw_attribution(detail)
    draw_feature_attribution(detail)
    draw_training()
    write_efficiency_ledger()
    draw_efficiency(ledger_rows)
    write_latex_assets(ledger_rows)
    print(f"Wrote EasyTSAD figures to {FIGURES}")
    print(f"Wrote LaTeX assets to {GENERATED}")


if __name__ == "__main__":
    main()
