#!/usr/bin/env python3
"""Create the strict zero-shot evaluation figures used by the IEEE paper."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from layyer import TShape_model  # noqa: E402
from tshape_zero_product import prepare_target_context  # noqa: E402


RESULTS = ROOT / "Results" / "RENE"
FIGURES = ROOT.parent / "IEEE-conference-template" / "figures"
DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")
PATTERN = "TShapeUniversalPattern-100"
FINAL = "TShapeUniversalZeroPlus-100"
GUARD = "TShape-zero_plus_pattern_balanced_residual_only_minmax"
SEED = 20260704
COLORS = {
    "ink": "#17212B",
    "muted": "#667085",
    "grid": "#DDE3E8",
    "teal": "#0B8A8F",
    "coral": "#E45B5B",
    "purple": "#7052A3",
    "gold": "#D99A00",
    "green": "#3D936A",
    "blue": "#3979A8",
}
DATASET_COLORS = dict(
    zip(DATASETS, ("#0B8A8F", "#E45B5B", "#D99A00", "#7052A3", "#3D936A", "#3979A8"))
)


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "axes.edgecolor": "#344054",
            "axes.linewidth": 0.65,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def save(fig: plt.Figure, stem: str, dpi: int = 360) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=dpi)
    plt.close(fig)


def _paper_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    *,
    title_size: float = 6.2,
    body_size: float = 4.8,
) -> None:
    patch = mpl.patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.009,rounding_size=0.018",
        facecolor=mpl.colors.to_rgba(color, 0.055),
        edgecolor=color,
        linewidth=1.05,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + h - 0.025, title, color=COLORS["ink"],
            fontsize=title_size, weight="bold", va="top")
    ax.text(x + 0.014, y + h - 0.078, body, color=COLORS["muted"],
            fontsize=body_size, va="top", linespacing=1.20)


def _paper_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str = "#667085") -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": color, "shrinkA": 1, "shrinkB": 1},
    )


def draw_intro_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.975, "From target training to a reusable detector", fontsize=7.2,
            weight="bold", color=COLORS["ink"], va="top")
    ax.text(0.02, 0.925, "claim boundary  |  stress test  |  repair  |  deployment",
            fontsize=4.8, color=COLORS["muted"], va="top")

    specs = (
        (0.02, 0.57, COLORS["teal"], "1  ORIGINAL CLAIM",
         "Target KPI + labels\nper-series TShape training\nstrong EasyTSAD F1 / F1-E"),
        (0.52, 0.57, COLORS["coral"], "2  REUSE STRESS TEST",
         "One frozen model meets unseen KPI\nshape, scale, and noise shift\nnormal / incident scores overlap"),
        (0.02, 0.20, COLORS["purple"], "3  ALGORITHMIC REPAIR",
         "Synthetic Pattern Bank -> TShape\nresidual guard under shift\nfixed open-box score fusion"),
        (0.52, 0.20, COLORS["green"], "4  PRODUCT CONTRACT",
         "0.138-MiB checkpoint + scorer\n834 series / 44 configurations\nCLI + web + channel attribution"),
    )
    for x, y, color, title, body in specs:
        _paper_box(ax, x, y, 0.46, 0.275, title, body, color, title_size=5.6, body_size=4.7)

    # Small visual glyphs make the story readable before the text is inspected.
    t = np.linspace(0, 1, 44)
    ax.plot(0.05 + 0.15 * t, 0.605 + 0.018 * np.sin(4 * np.pi * t), color=COLORS["teal"], lw=0.9)
    ax.plot(0.25 + 0.15 * t, 0.605 + 0.018 * np.sin(4 * np.pi * t + 0.35), color=COLORS["blue"], lw=0.9)
    shifted = 0.017 * np.sin(7 * np.pi * t)
    shifted[24:29] += np.array([0.0, 0.02, 0.055, 0.022, -0.006])
    ax.plot(0.55 + 0.34 * t, 0.605 + shifted, color=COLORS["coral"], lw=0.9)
    for index, color in enumerate((COLORS["purple"], COLORS["gold"], COLORS["teal"])):
        ax.plot(0.05 + 0.11 * t, 0.235 + 0.012 * index + 0.007 * np.sin((index + 2) * np.pi * t),
                color=color, lw=0.7)
    ax.add_patch(mpl.patches.Rectangle((0.56, 0.225), 0.13, 0.052, fc="white", ec=COLORS["green"], lw=0.7))
    ax.plot([0.575, 0.67], [0.245, 0.258], color=COLORS["green"], lw=0.8)
    ax.scatter([0.61, 0.65], [0.25, 0.255], s=5, color=COLORS["coral"], zorder=3)

    _paper_arrow(ax, 0.48, 0.705, 0.52, 0.705)
    _paper_arrow(ax, 0.75, 0.57, 0.75, 0.485)
    _paper_arrow(ax, 0.52, 0.335, 0.48, 0.335)
    ax.text(0.02, 0.085,
            "We do not refute target-trained TShape; we engineer and test the stronger frozen-checkpoint contract.",
            fontsize=4.85, color=COLORS["ink"], weight="bold", va="top", wrap=True)
    ax.plot([0.02, 0.98], [0.115, 0.115], color=COLORS["grid"], lw=0.7)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_intro_strict")


def draw_motivation_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.98, "An executable residual is not a transfer guarantee", fontsize=7.0,
            weight="bold", color=COLORS["ink"], va="top")

    def state_panel(x: float, color: str, title: str, shifted: bool) -> None:
        _paper_box(ax, x, 0.52, 0.45, 0.37, title, "", color, title_size=5.7)
        t = np.linspace(0, 1, 54)
        for row in range(3):
            baseline = 0.78 - 0.063 * row
            freq = 2.0 + 0.5 * row if not shifted else 4.4 + 0.8 * row
            signal = 0.014 * np.sin(freq * np.pi * t + row * 0.5)
            if shifted and row == 1:
                signal[30:35] += np.array([0.0, 0.025, 0.06, 0.025, -0.01])
            ax.plot(x + 0.025 + 0.18 * t, baseline + signal, color=color, lw=0.75)
        gx, gy, cell = x + 0.26, 0.675, 0.030
        for i in range(4):
            for j in range(4):
                value = 0.75 - 0.15 * abs(i - j) if not shifted else 0.30 + 0.45 * ((i + 2 * j) % 3 == 0)
                ax.add_patch(mpl.patches.Rectangle(
                    (gx + cell * j, gy + cell * (3 - i)), cell, cell,
                    fc=mpl.colors.to_rgba(color, max(0.10, min(0.78, value))), ec="white", lw=0.3))
        xx = np.linspace(x + 0.05, x + 0.40, 100)
        mu1, mu2 = ((x + 0.15, x + 0.31) if not shifted else (x + 0.20, x + 0.25))
        sig1, sig2 = ((0.040, 0.042) if not shifted else (0.060, 0.067))
        n = np.exp(-0.5 * ((xx - mu1) / sig1) ** 2)
        a = np.exp(-0.5 * ((xx - mu2) / sig2) ** 2)
        ax.plot(xx, 0.555 + 0.045 * n, color=COLORS["teal"], lw=0.9)
        ax.plot(xx, 0.555 + 0.045 * a, color=COLORS["purple"], lw=0.9, ls="--")
        ax.text(x + 0.225, 0.535, "separated" if not shifted else "overlap",
                ha="center", fontsize=4.3, color=COLORS["muted"])

    state_panel(0.02, COLORS["teal"], "TARGET-TRAINED KPI", False)
    state_panel(0.53, COLORS["coral"], "UNSEEN SHIFTED KPI", True)
    _paper_arrow(ax, 0.47, 0.71, 0.53, 0.71, COLORS["ink"])
    ax.text(0.50, 0.75, "same API", fontsize=4.2, weight="bold", ha="center",
            color=COLORS["ink"], bbox={"fc": "white", "ec": "none", "pad": 0.2})

    _paper_box(ax, 0.02, 0.19, 0.45, 0.22, "PATTERN COVERAGE",
               "20 generators x 8 variants\nrandom transforms + context incidents\n-> one benchmark-independent checkpoint",
               COLORS["purple"], title_size=5.7, body_size=4.55)
    _paper_box(ax, 0.53, 0.19, 0.45, 0.22, "RELIABILITY FLOOR",
               "rolling median + spectral + MAD\nfixed 85% guard budget under shift\n-> inspectable fallback evidence",
               COLORS["gold"], title_size=5.7, body_size=4.55)
    _paper_arrow(ax, 0.245, 0.52, 0.245, 0.41, COLORS["purple"])
    _paper_arrow(ax, 0.755, 0.52, 0.755, 0.41, COLORS["gold"])
    ax.text(0.50, 0.095, "TShape-Zero+ = 15% learned shape evidence + 85% residual guard",
            ha="center", fontsize=5.1, weight="bold", color=COLORS["ink"])
    ax.text(0.50, 0.045, "Both channels are returned, measured, and ablated separately.",
            ha="center", fontsize=4.65, color=COLORS["muted"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_motivation_strict")


def draw_framework_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.98, "TShape-Zero+ end-to-end contract", fontsize=7.2, weight="bold",
            color=COLORS["ink"], va="top")
    ax.text(0.02, 0.925, "OFFLINE: BENCHMARK-INDEPENDENT WEIGHT LEARNING", fontsize=5.0,
            weight="bold", color=COLORS["purple"], va="top")

    _paper_box(ax, 0.02, 0.63, 0.28, 0.23, "Pattern Bank", "20 dynamics x 8 variants\nrandomized transforms\n6 context-only incidents",
               COLORS["purple"], title_size=5.8, body_size=4.25)
    _paper_box(ax, 0.36, 0.63, 0.27, 0.23, "Denoising pairs", "60,000 generated windows\ncorrupt context -> clean target\nnested 10%-100% prefixes",
               COLORS["blue"], title_size=5.8, body_size=4.05)
    _paper_box(ax, 0.69, 0.63, 0.29, 0.23, "Frozen TShape", "patch CNN + dual attention\nmax 30 epochs; patience 3\n0.138-MiB strict checkpoint",
               COLORS["teal"], title_size=5.8, body_size=4.05)
    _paper_arrow(ax, 0.30, 0.745, 0.36, 0.745, COLORS["purple"])
    _paper_arrow(ax, 0.63, 0.745, 0.69, 0.745, COLORS["blue"])

    ax.plot([0.02, 0.98], [0.56, 0.56], color=COLORS["grid"], lw=0.8)
    ax.text(0.02, 0.525, "ONLINE: ONE FROZEN CHECKPOINT, NO TARGET FITTING", fontsize=5.0,
            weight="bold", color=COLORS["teal"], va="top")
    _paper_box(ax, 0.02, 0.22, 0.20, 0.23, "KPI input", "unseen history x\noptional unlabeled\ncalibration prefix",
               COLORS["teal"], title_size=5.6, body_size=4.25)
    _paper_box(ax, 0.27, 0.22, 0.20, 0.23, "Preprocess", "first difference\ntrain/history scale\np=16 aligned windows",
               COLORS["blue"], title_size=5.6, body_size=4.25)
    _paper_box(ax, 0.52, 0.34, 0.20, 0.12, "Neural residual", "frozen Pattern-TShape",
               COLORS["purple"], title_size=5.1, body_size=4.0)
    _paper_box(ax, 0.52, 0.19, 0.20, 0.12, "Residual Guard", "median / spectral / MAD",
               COLORS["gold"], title_size=5.1, body_size=3.75)
    _paper_box(ax, 0.77, 0.28, 0.21, 0.17, "Fixed fusion", "15% neural + 85% guard\ncomponent traces\nretained",
               COLORS["coral"], title_size=5.5, body_size=3.75)
    _paper_arrow(ax, 0.22, 0.335, 0.27, 0.335, COLORS["teal"])
    _paper_arrow(ax, 0.47, 0.335, 0.52, 0.40, COLORS["purple"])
    _paper_arrow(ax, 0.47, 0.335, 0.52, 0.25, COLORS["gold"])
    _paper_arrow(ax, 0.72, 0.40, 0.77, 0.38, COLORS["purple"])
    _paper_arrow(ax, 0.72, 0.25, 0.77, 0.33, COLORS["gold"])

    output = mpl.patches.FancyBboxPatch(
        (0.18, 0.035), 0.64, 0.085, boxstyle="round,pad=0.008,rounding_size=0.018",
        fc=mpl.colors.to_rgba(COLORS["green"], 0.06), ec=COLORS["green"], lw=1.0)
    ax.add_patch(output)
    ax.text(0.50, 0.092, "OPEN-BOX OUTPUT", fontsize=5.2, weight="bold", color=COLORS["green"], ha="center", va="center")
    ax.text(0.50, 0.057, "aligned input + fused/neural/guard scores + top-k indices  |  CLI and web",
            fontsize=4.25, color=COLORS["ink"], ha="center", va="center")
    _paper_arrow(ax, 0.875, 0.28, 0.79, 0.12, COLORS["green"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_framework_strict")


def draw_tshape_architecture() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.98, "TShape next-value residual scorer", fontsize=7.1, weight="bold",
            color=COLORS["ink"], va="top")

    _paper_box(ax, 0.02, 0.66, 0.23, 0.22, "Input window", "$x[t-p+1:t]$,  $p=16$",
               COLORS["teal"], title_size=5.6, body_size=4.4)
    t = np.linspace(0, 1, 16)
    signal = 0.695 + 0.035 * np.sin(5 * np.pi * t) + 0.012 * np.cos(9 * np.pi * t)
    ax.plot(0.045 + 0.18 * t, signal, color=COLORS["teal"], lw=0.9)
    ax.scatter(0.045 + 0.18 * t, signal, s=3.5, color=COLORS["teal"], zorder=3)

    _paper_box(ax, 0.31, 0.66, 0.31, 0.22, "Patch + shape encoder", "4 patches x 4  |  Conv1D + pool",
               COLORS["blue"], title_size=5.6, body_size=4.2)
    patch_colors = (COLORS["teal"], COLORS["gold"], COLORS["purple"], COLORS["green"])
    for i, color in enumerate(patch_colors):
        x = 0.335 + 0.065 * i
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (x, 0.685), 0.052, 0.050, boxstyle="round,pad=0.004,rounding_size=0.008",
            fc=mpl.colors.to_rgba(color, 0.18), ec=color, lw=0.7))
        ax.plot([x + 0.010, x + 0.025, x + 0.042],
                [0.704, 0.721 - 0.004 * i, 0.698 + 0.004 * i], color=color, lw=0.65)

    _paper_box(ax, 0.68, 0.66, 0.30, 0.22, "Positioned tokens", "shape evidence + patch position",
               COLORS["purple"], title_size=5.6, body_size=4.2)
    for i, color in enumerate(patch_colors):
        x = 0.715 + 0.058 * i
        ax.add_patch(mpl.patches.Circle((x, 0.707), 0.020, fc=mpl.colors.to_rgba(color, 0.22), ec=color, lw=0.7))
        ax.text(x, 0.707, str(i + 1), fontsize=3.8, ha="center", va="center", color=COLORS["ink"])
    _paper_arrow(ax, 0.25, 0.77, 0.31, 0.77, COLORS["teal"])
    _paper_arrow(ax, 0.62, 0.77, 0.68, 0.77, COLORS["blue"])

    ax.text(0.02, 0.59, "DUAL ATTENTION", fontsize=4.9, weight="bold", color=COLORS["purple"])
    _paper_box(ax, 0.02, 0.34, 0.29, 0.20, "Local attention", "within-patch shape",
               COLORS["blue"], title_size=5.4, body_size=4.0)
    _paper_box(ax, 0.36, 0.34, 0.29, 0.20, "Global attention", "cross-patch context",
               COLORS["purple"], title_size=5.4, body_size=4.0)
    local = np.eye(4) * 0.70 + np.array([[0,.12,.04,.03],[.10,0,.08,.03],[.04,.09,0,.12],[.03,.04,.10,0]])
    global_map = np.array([[.50,.18,.12,.35],[.15,.48,.30,.17],[.11,.33,.46,.22],[.38,.16,.24,.45]])
    for matrix, x, color in ((local, 0.185, COLORS["blue"]), (global_map, 0.525, COLORS["purple"])):
        cell = 0.026
        for i in range(4):
            for j in range(4):
                ax.add_patch(mpl.patches.Rectangle(
                    (x - 0.052 + j * cell, 0.350 + (3 - i) * cell), cell, cell,
                    fc=mpl.colors.to_rgba(color, 0.10 + 0.75 * float(matrix[i, j])), ec="white", lw=0.25))

    _paper_box(ax, 0.72, 0.34, 0.26, 0.20, "Gated fusion", "$gL + (1-g)G$\npatch evidence",
               COLORS["gold"], title_size=5.4, body_size=4.2)
    ax.add_patch(mpl.patches.Circle((0.845, 0.385), 0.032, fc=mpl.colors.to_rgba(COLORS["gold"], 0.20),
                                    ec=COLORS["gold"], lw=0.8))
    ax.plot([0.845, 0.863], [0.385, 0.408], color=COLORS["ink"], lw=0.8)
    _paper_arrow(ax, 0.83, 0.66, 0.22, 0.54, COLORS["blue"])
    _paper_arrow(ax, 0.83, 0.66, 0.50, 0.54, COLORS["purple"])
    _paper_arrow(ax, 0.31, 0.44, 0.72, 0.44, COLORS["blue"])
    _paper_arrow(ax, 0.65, 0.40, 0.72, 0.40, COLORS["purple"])

    _paper_box(ax, 0.10, 0.07, 0.30, 0.17, "Forecast head", "fused tokens -> $\\hat{x}[t+1]$",
               COLORS["teal"], title_size=5.5, body_size=4.3)
    _paper_box(ax, 0.53, 0.07, 0.37, 0.17, "Anomaly score", "$T[t+1]=(x[t+1]-\\hat{x}[t+1])^2$\nlarger residual -> more anomalous",
               COLORS["coral"], title_size=5.5, body_size=4.1)
    _paper_arrow(ax, 0.85, 0.34, 0.38, 0.24, COLORS["teal"])
    _paper_arrow(ax, 0.40, 0.155, 0.53, 0.155, COLORS["coral"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_tshape_architecture_strict")


def sweep_artifacts() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    summaries: list[dict[str, str]] = []
    runs: list[dict[str, object]] = []
    for pct in range(10, 100, 10):
        tag = f"submission_strict_zero_size_{pct}"
        summaries.extend(rows(RESULTS / f"pattern_bank_tshape_summary_metrics_{tag}.csv"))
        runs.extend(
            json.loads((RESULTS / f"pattern_bank_tshape_run_info_{tag}.json").read_text(encoding="utf-8"))
        )
    summaries.extend(
        rows(RESULTS / "pattern_bank_tshape_summary_metrics_submission_strict_zero_main_full.csv")
    )
    runs.extend(
        json.loads(
            (RESULTS / "pattern_bank_tshape_run_info_submission_strict_zero_main_full.json").read_text(
                encoding="utf-8"
            )
        )
    )
    return summaries, runs


def draw_pattern_training(runs: Sequence[Mapping[str, object]]) -> None:
    ordered = sorted(runs, key=lambda row: int(row["fraction_pct"]))
    cmap = mpl.colormaps["viridis"]
    fig, (ax, endpoint) = plt.subplots(
        2,
        1,
        figsize=(3.5, 3.12),
        gridspec_kw={"height_ratios": (2.0, 1.0), "hspace": 0.64},
    )
    padded = np.full((len(ordered), max(len(row["loss"]) for row in ordered)), np.nan)
    for index, run in enumerate(ordered):
        loss = np.asarray(run["loss"], dtype=float)
        padded[index, : len(loss)] = loss
        pct = int(run["fraction_pct"])
        color = cmap(index / max(1, len(ordered) - 1))
        label = f"{pct}%" if pct in {10, 50, 100} else None
        ax.plot(np.arange(1, len(loss) + 1), loss, color=color, lw=1.0, alpha=0.88, label=label)
        ax.scatter(len(loss), loss[-1], s=8, color=color, edgecolor="white", linewidth=0.3, zorder=3)
    epochs = np.arange(1, padded.shape[1] + 1)
    cross_size_mean = np.nanmean(padded, axis=0)
    valid_counts = np.sum(np.isfinite(padded), axis=0)
    cross_size_sem = np.nanstd(padded, axis=0, ddof=1) / np.sqrt(np.maximum(valid_counts, 1))
    ax.plot(
        epochs,
        cross_size_mean,
        color=COLORS["teal"],
        lw=0.85,
        ls="--",
        alpha=0.82,
        zorder=2,
    )
    ax.fill_between(
        epochs,
        cross_size_mean - cross_size_sem,
        cross_size_mean + cross_size_sem,
        color=COLORS["teal"],
        alpha=0.11,
        linewidth=0,
        label="cross-size mean $\pm$ SEM",
    )
    ax.set_ylabel("Training MSE")
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper right", handlelength=1.2, columnspacing=0.55)
    ax.set_title("(a) One synthetic curriculum, increasing sample budget", loc="left", weight="bold")

    sizes = np.array([int(run["fraction_pct"]) for run in ordered])
    best = np.array([float(run["best_training_loss"]) for run in ordered])
    windows = np.array([int(run["windows"]) for run in ordered])
    late_epoch_sd = np.array(
        [float(np.std(np.asarray(run["loss"], dtype=float)[-5:], ddof=1)) for run in ordered]
    )
    endpoint.plot(sizes, best, color=COLORS["purple"], lw=1.35, marker="o", ms=3.2)
    endpoint.fill_between(
        sizes,
        np.maximum(0.0, best - late_epoch_sd),
        best + late_epoch_sd,
        color=COLORS["purple"],
        alpha=0.10,
        linewidth=0,
    )
    for size, value, count in zip(sizes[[0, -1]], best[[0, -1]], windows[[0, -1]]):
        endpoint.annotate(
            f"{count // 1000}k windows",
            (size, value),
            xytext=(3 if size < 50 else -3, 5),
            textcoords="offset points",
            ha="left" if size < 50 else "right",
            fontsize=5.5,
            color=COLORS["muted"],
        )
    endpoint.set_xticks(np.arange(10, 101, 10))
    endpoint.set_xlabel("Pattern Bank size (% of 60k windows)")
    endpoint.set_ylabel("Best MSE")
    endpoint.grid(axis="y", color=COLORS["grid"], lw=0.55)
    endpoint.spines[["top", "right"]].set_visible(False)
    endpoint.set_title("(b) Endpoint optimization", loc="left", weight="bold")
    save(fig, "fig_strict_pattern_training")


def draw_pattern_effect(summaries: Sequence[Mapping[str, str]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.75), sharex=True)
    legend_handles = None
    for ax, dataset in zip(axes.flat, DATASETS):
        subset = [row for row in summaries if row.get("dataset") == dataset]
        pure = sorted(
            [row for row in subset if row.get("method", "").startswith("TShapeUniversalPattern-")],
            key=lambda row: fnum(row.get("fraction")),
        )
        final = sorted(
            [row for row in subset if row.get("method", "").startswith("TShapeUniversalZeroPlus-")],
            key=lambda row: fnum(row.get("fraction")),
        )
        if len(pure) != 10 or len(final) != 10:
            raise RuntimeError(f"Incomplete Pattern Bank curve for {dataset}: {len(pure)}/{len(final)}")
        x = np.array([100 * fnum(row["fraction"]) for row in pure])
        curves = (
            (np.array([fnum(row["point_f1_pa"]) for row in final]), COLORS["teal"], "o", "Zero+ Point-F1", "-"),
            (np.array([fnum(row["event_f1_pa_log"]) for row in final]), COLORS["coral"], "s", "Zero+ Event-F1", "-"),
            (np.array([fnum(row["point_f1_pa"]) for row in pure]), COLORS["teal"], None, "Pattern Point-F1", ":"),
            (np.array([fnum(row["event_f1_pa_log"]) for row in pure]), COLORS["coral"], None, "Pattern Event-F1", ":"),
        )
        all_values = []
        for values, color, marker, label, linestyle in curves:
            ax.plot(
                x,
                values,
                color=color,
                lw=1.15 if linestyle == "-" else 0.8,
                ls=linestyle,
                marker=marker,
                ms=2.5,
                alpha=1.0 if linestyle == "-" else 0.58,
                label=label,
            )
            all_values.extend(values.tolist())
        ax.fill_between(
            x,
            np.array([fnum(row["point_f1_pa"]) for row in pure]),
            np.array([fnum(row["point_f1_pa"]) for row in final]),
            color=COLORS["teal"],
            alpha=0.055,
        )
        lo, hi = min(all_values), max(all_values)
        margin = max(0.025, 0.08 * (hi - lo))
        ax.set_ylim(max(0.0, lo - margin), min(1.005, hi + margin))
        ax.set_xticks((10, 40, 70, 100))
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(dataset, pad=2, weight="bold", color=DATASET_COLORS[dataset])
        legend_handles = ax.get_legend_handles_labels()
    axes[1, 1].set_xlabel("Pattern Bank size (\%)")
    axes[0, 0].set_ylabel("EasyTSAD F1")
    axes[1, 0].set_ylabel("EasyTSAD F1")
    handles, labels = legend_handles
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=0.7,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.96, bottom=0.21, wspace=0.24, hspace=0.28)
    save(fig, "fig_strict_pattern_effect")


def grouped(
    source: Iterable[Mapping[str, str]], method: str, metric: str
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in source:
        if row.get("method") != method:
            continue
        seed = str(row.get("seed", "")).strip()
        if seed and int(float(seed)) != SEED:
            continue
        value = fnum(row.get(metric))
        if np.isfinite(value):
            values[(str(row["dataset"]), str(row["series"]))].append(value)
    return {key: float(np.mean(group)) for key, group in values.items()}


def draw_overall(ledger: Sequence[Mapping[str, str]]) -> None:
    """Summarize only directly deployable no-fit/frozen/strict methods."""
    lookup = {(row["method_id"], row["dataset"]): row for row in ledger}
    methods = (
        ("persistence", "Persistence"),
        ("rolling_mean", "Rolling mean"),
        ("rolling_median", "Rolling median"),
        ("spectral_residual", "Spectral residual"),
        ("Chronos-Bolt-Tiny", "Chronos"),
        ("Timer-base-84m", "Timer"),
        ("TimesFM-2.5-200M", "TimesFM"),
        ("Sundial-base-128m", "Sundial"),
        (PATTERN, "Pattern-only Zero"),
        (FINAL, "TShape-Zero+"),
    )
    values = []
    for method, label in methods:
        point = float(np.mean([fnum(lookup[(method, dataset)]["point_f1"]) for dataset in DATASETS]))
        event = float(np.mean([fnum(lookup[(method, dataset)]["event_f1"]) for dataset in DATASETS]))
        values.append((method, label, point, event))
    values.sort(key=lambda item: 0.5 * (item[2] + item[3]), reverse=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    y = np.arange(len(values))
    for index, (method, _label, point, event) in enumerate(values):
        emphasized = method in {PATTERN, FINAL}
        line_color = COLORS["teal"] if method == FINAL else COLORS["purple"] if method == PATTERN else "#C8D0D8"
        ax.plot([event, point], [index, index], color=line_color, lw=2.0 if emphasized else 1.1, zorder=1)
        ax.scatter(event, index, s=24 if emphasized else 17, marker="s", color=COLORS["coral"],
                   edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(point, index, s=28 if emphasized else 19, marker="o", color=COLORS["teal"],
                   edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _method, label, _point, _event in values])
    for tick, (method, _label, _point, _event) in zip(ax.get_yticklabels(), values):
        if method in {PATTERN, FINAL}:
            tick.set_weight("bold")
            tick.set_color(COLORS["teal"] if method == FINAL else COLORS["purple"])
    ax.invert_yaxis()
    all_scores = np.asarray([score for _method, _label, point, event in values for score in (point, event)])
    ax.set_xlim(max(0.0, float(np.min(all_scores)) - 0.018), min(1.0, float(np.max(all_scores)) + 0.018))
    ax.set_xlabel("Six-dataset macro EasyTSAD F1")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        mpl.lines.Line2D([], [], color=COLORS["teal"], marker="o", ls="", label="Point-F1"),
        mpl.lines.Line2D([], [], color=COLORS["coral"], marker="s", ls="", label="Event-F1"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        handletextpad=0.25,
        columnspacing=0.65,
    )
    ax.set_title("Strict N/F/S methods", loc="left", weight="bold")
    save(fig, "fig_strict_overall")


def strict_detail() -> list[dict[str, str]]:
    detail = rows(RESULTS / "pattern_bank_tshape_series_metrics_submission_strict_zero_main_full.csv")
    detail.extend(rows(RESULTS / "tshape_series_metrics_submission_easytsad_tshape_balanced_full3.csv"))
    detail.extend(rows(RESULTS / "baseline_series_metrics.csv"))
    return detail


def _violin(ax: plt.Axes, data: Sequence[np.ndarray], positions: np.ndarray, color: str) -> None:
    parts = ax.violinplot(data, positions=positions, widths=0.27, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_linewidth(0.45)
        body.set_alpha(0.56)
    for values, position in zip(data, positions):
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        ax.vlines(position, q1, q3, color=COLORS["ink"], lw=1.2, zorder=3)
        ax.scatter(position, median, s=8, color="white", edgecolor=COLORS["ink"], lw=0.55, zorder=4)


def draw_guard_distributions(detail: Sequence[Mapping[str, str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.45), sharex=True)
    x = np.arange(len(DATASETS), dtype=float)
    for ax, metric, title in (
        (axes[0], "point_f1_pa", "(a) Point-F1 paired lift"),
        (axes[1], "event_f1_pa_log", "(b) Event-F1 paired lift"),
    ):
        final = grouped(detail, FINAL, metric)
        pattern = grouped(detail, PATTERN, metric)
        guard = grouped(detail, GUARD, metric)
        pattern_delta = []
        guard_delta = []
        for dataset in DATASETS:
            kp = sorted(key for key in final if key[0] == dataset and key in pattern)
            kg = sorted(key for key in final if key[0] == dataset and key in guard)
            pattern_delta.append(np.asarray([final[key] - pattern[key] for key in kp], dtype=float))
            guard_delta.append(np.asarray([final[key] - guard[key] for key in kg], dtype=float))
        _violin(ax, pattern_delta, x - 0.16, COLORS["purple"])
        _violin(ax, guard_delta, x + 0.16, COLORS["gold"])
        ax.axhline(0, color="#475467", lw=0.75, ls="--")
        ax.set_ylabel("Zero+ lift")
        ax.set_ylim(-1.02, 1.02)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, loc="left", weight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DATASETS, rotation=22, ha="right")
    handles = [
        mpl.patches.Patch(fc=COLORS["purple"], alpha=0.56, label="vs. Pattern-only"),
        mpl.patches.Patch(fc=COLORS["gold"], alpha=0.56, label="vs. Guard-only"),
    ]
    axes[0].legend(handles=handles, frameon=False, ncol=2, loc="lower left")
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.15, hspace=0.27)
    save(fig, "fig_strict_guard_violin")


def draw_guard_bars(ledger: Sequence[Mapping[str, str]]) -> None:
    lookup = {(row["method_id"], row["dataset"]): row for row in ledger}
    methods = (
        (PATTERN, "Pattern-only", COLORS["purple"]),
        (GUARD, "Guard-only", COLORS["gold"]),
        (FINAL, "TShape-Zero+", COLORS["teal"]),
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
    x = np.arange(len(DATASETS))
    width = 0.23
    for ax, field, title in (
        (axes[0], "point_f1", "(a) Point-F1"),
        (axes[1], "event_f1", "(b) Event-F1"),
    ):
        for offset, (method, label, color) in zip((-width, 0, width), methods):
            values = [fnum(lookup[(method, dataset)][field]) for dataset in DATASETS]
            ax.bar(x + offset, values, width=width * 0.9, color=color, alpha=0.88, label=label)
        ax.set_ylim(0.30, 1.02)
        ax.set_ylabel("EasyTSAD F1")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, loc="left", weight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DATASETS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.005),
        columnspacing=0.6,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.19, hspace=0.30)
    save(fig, "fig_strict_guard_bars")


def draw_guard_ablation(ledger: Sequence[Mapping[str, str]]) -> None:
    summary = rows(
        RESULTS / "strict_zero_ablation_summary_metrics_submission_strict_zero_ablation.csv"
    )
    lookup = {(row["method_id"], row["dataset"]): row for row in ledger}

    def ledger_macro(method: str, field: str) -> float:
        return float(np.mean([fnum(lookup[(method, dataset)][field]) for dataset in DATASETS]))

    def ablation_macro(method: str, field: str) -> float:
        selected = [fnum(row[field]) for row in summary if row.get("method") == method]
        selected = [value for value in selected if np.isfinite(value)]
        if len(selected) != len(DATASETS):
            raise RuntimeError(f"Incomplete strict ablation for {method}/{field}: {len(selected)}")
        return float(np.mean(selected))

    alpha_methods = {
        0.00: "StrictGuardOnly",
        0.05: "StrictZeroPlusAlpha005",
        0.10: "StrictZeroPlusAlpha010",
        0.25: "StrictZeroPlusAlpha025",
        0.50: "StrictZeroPlusAlpha050",
    }
    alphas = np.asarray((0.00, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00))
    point = []
    event = []
    for alpha in alphas:
        if math.isclose(alpha, 0.15):
            point.append(ledger_macro(FINAL, "point_f1"))
            event.append(ledger_macro(FINAL, "event_f1"))
        elif math.isclose(alpha, 1.00):
            point.append(ledger_macro(PATTERN, "point_f1"))
            event.append(ledger_macro(PATTERN, "event_f1"))
        else:
            method = alpha_methods[float(alpha)]
            point.append(ablation_macro(method, "point_f1_pa"))
            event.append(ablation_macro(method, "event_f1_pa_log"))
    point = np.asarray(point)
    event = np.asarray(event)

    fig, (ax, pairs) = plt.subplots(
        2, 1, figsize=(3.5, 3.2), gridspec_kw={"height_ratios": (1.35, 1.0), "hspace": 0.48}
    )
    alpha_x = np.arange(len(alphas), dtype=float)
    ax.plot(alpha_x, point, color=COLORS["teal"], marker="o", ms=3.3, lw=1.35, label="Point-F1")
    ax.plot(alpha_x, event, color=COLORS["coral"], marker="s", ms=3.0, lw=1.35, label="Event-F1")
    ax.fill_between(alpha_x, event, point, color=COLORS["purple"], alpha=0.055)
    release_i = int(np.flatnonzero(np.isclose(alphas, 0.15))[0])
    ax.scatter([release_i, release_i], [point[release_i], event[release_i]], s=32, facecolor="white",
               edgecolor=COLORS["ink"], lw=0.75, zorder=4)
    ax.axvline(release_i, color=COLORS["ink"], lw=0.65, ls="--")
    ax.text(release_i + 0.10, 0.750, "release", fontsize=5.2, ha="left", va="center", color=COLORS["ink"])
    ax.set_xticks(alpha_x, ("0", ".05", ".10", ".15", ".25", ".50", "1"))
    ax.set_xlabel("Neural budget $\\alpha$  (guard $\\rightarrow$ Pattern)")
    ax.set_ylabel("Six-dataset macro F1")
    ax.set_ylim(min(event.min(), point.min()) - 0.018, max(event.max(), point.max()) + 0.012)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower left", handletextpad=0.25, columnspacing=0.65)
    ax.set_title("(a) Fixed-weight sensitivity", loc="left", weight="bold")

    pair_specs = (
        ("TShape + median", "StrictTShapeMedian"),
        ("TShape + spectral", "StrictTShapeSpectral"),
        ("Rank fusion (.15)", "StrictZeroPlusRank015"),
        ("Release (.15)", None),
    )
    y = np.arange(len(pair_specs))
    pvals, evals = [], []
    for _label, method in pair_specs:
        if method is None:
            pvals.append(ledger_macro(FINAL, "point_f1"))
            evals.append(ledger_macro(FINAL, "event_f1"))
        else:
            pvals.append(ablation_macro(method, "point_f1_pa"))
            evals.append(ablation_macro(method, "event_f1_pa_log"))
    for yi, pvalue, evalue in zip(y, pvals, evals):
        pairs.plot([evalue, pvalue], [yi, yi], color="#C8D0D8", lw=1.5, zorder=1)
    pairs.scatter(pvals, y, color=COLORS["teal"], s=22, zorder=3)
    pairs.scatter(evals, y, color=COLORS["coral"], marker="s", s=20, zorder=3)
    pairs.set_yticks(y, [label for label, _method in pair_specs])
    pairs.invert_yaxis()
    pairs.set_xlabel("Six-dataset macro F1")
    pairs.grid(axis="x", color=COLORS["grid"], lw=0.5)
    pairs.spines[["top", "right", "left"]].set_visible(False)
    pairs.tick_params(axis="y", length=0)
    pairs.set_title("(b) Normalization and channel pairs", loc="left", weight="bold")
    fig.subplots_adjust(left=0.28, right=0.99, top=0.97, bottom=0.12)
    save(fig, "fig_strict_guard_ablation")


def _series_characteristics(dataset: str, series: str) -> dict[str, float]:
    """Compute target descriptors without using them to fit or select the checkpoint."""
    path = ROOT / "datasets" / "UTS" / dataset / series
    values = np.asarray(np.load(path / "test.npy"), dtype=np.float64).reshape(-1)
    labels = np.asarray(np.load(path / "test_label.npy"), dtype=np.int8).reshape(-1)
    anomaly_ratio = float(np.mean(labels)) if labels.size else 0.0
    if values.size > 4096:
        indices = np.linspace(0, values.size - 1, 4096, dtype=np.int64)
        values = values[indices]
    finite = values[np.isfinite(values)]
    fill = float(np.median(finite)) if finite.size else 0.0
    values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
    centered = values - np.mean(values)
    scale = float(np.std(centered))
    if values.size < 8 or scale < 1e-12:
        trend = periodicity = roughness = 0.0
    else:
        time_axis = np.linspace(-1.0, 1.0, values.size)
        trend = abs(float(np.corrcoef(time_axis, centered)[0, 1]))
        lags = [lag for lag in (4, 8, 16, 24, 32, 48, 64, 96, 128) if lag < values.size // 3]
        correlations = []
        for lag in lags:
            left, right = centered[:-lag], centered[lag:]
            denominator = float(np.std(left) * np.std(right))
            correlations.append(float(np.mean(left * right) / denominator) if denominator > 1e-12 else 0.0)
        periodicity = max([0.0] + correlations)
        differences = np.diff(values)
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        roughness = float(
            np.median(np.abs(differences - np.median(differences))) / max(iqr, 1e-12)
        )
    return {
        "anomaly_ratio": anomaly_ratio,
        "trend_strength": trend,
        "periodicity": periodicity,
        "roughness": roughness,
    }


def draw_channel_attribution(detail: Sequence[Mapping[str, str]]) -> None:
    """Separate the residual reliability floor from TShape's marginal value."""
    channel_methods = (
        ("Pattern", PATTERN),
        ("Median", "rolling_median"),
        ("Spectral", "spectral_residual"),
        ("Guard", GUARD),
    )
    point = {label: grouped(detail, method, "point_f1_pa") for label, method in channel_methods}
    event = {label: grouped(detail, method, "event_f1_pa_log") for label, method in channel_methods}
    final_point = grouped(detail, FINAL, "point_f1_pa")
    final_event = grouped(detail, FINAL, "event_f1_pa_log")
    keys = sorted(
        set(final_point)
        & set(final_event)
        & set.intersection(*(set(values) for values in point.values()))
        & set.intersection(*(set(values) for values in event.values()))
    )
    if not keys:
        raise RuntimeError("No paired strict channel-attribution series")

    records: list[dict[str, object]] = []
    for dataset, series in keys:
        utilities = {
            label: 0.5 * (point[label][(dataset, series)] + event[label][(dataset, series)])
            for label, _method in channel_methods
        }
        guard_utility = utilities["Guard"]
        final_utility = 0.5 * (final_point[(dataset, series)] + final_event[(dataset, series)])
        features = _series_characteristics(dataset, series)
        records.append(
            {
                "dataset": dataset,
                "series": series,
                **features,
                "best_evidence_channel": max(utilities, key=utilities.get),
                "pattern_point_f1": point["Pattern"][(dataset, series)],
                "pattern_event_f1": event["Pattern"][(dataset, series)],
                "guard_point_f1": point["Guard"][(dataset, series)],
                "guard_event_f1": event["Guard"][(dataset, series)],
                "final_point_f1": final_point[(dataset, series)],
                "final_event_f1": final_event[(dataset, series)],
                "final_gain_vs_guard": final_utility - guard_utility,
            }
        )

    series_fields = (
        "dataset", "series", "anomaly_ratio", "trend_strength", "periodicity", "roughness",
        "best_evidence_channel", "pattern_point_f1", "pattern_event_f1", "guard_point_f1",
        "guard_event_f1", "final_point_f1", "final_event_f1", "final_gain_vs_guard",
    )
    with (RESULTS / "submission_strict_channel_attribution_easytsad.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=series_fields)
        writer.writeheader()
        writer.writerows(records)

    labels = tuple(label for label, _method in channel_methods)
    fractions = np.zeros((len(DATASETS), len(labels)), dtype=float)
    win_rates = np.zeros(len(DATASETS), dtype=float)
    summary_rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        selected = [record for record in records if record["dataset"] == dataset]
        for channel_index, label in enumerate(labels):
            fractions[dataset_index, channel_index] = np.mean(
                [record["best_evidence_channel"] == label for record in selected]
            )
        gains = np.asarray([record["final_gain_vs_guard"] for record in selected], dtype=float)
        win_rates[dataset_index] = float(np.mean(gains > 1e-12))
        summary_rows.append(
            {
                "dataset": dataset,
                "valid_series": len(selected),
                **{f"best_{label.lower()}_share": fractions[dataset_index, index]
                   for index, label in enumerate(labels)},
                "final_beats_guard_share": win_rates[dataset_index],
                "mean_final_gain_vs_guard": float(np.mean(gains)),
            }
        )
    summary_fields = (
        "dataset", "valid_series", "best_pattern_share", "best_median_share",
        "best_spectral_share", "best_guard_share", "final_beats_guard_share",
        "mean_final_gain_vs_guard",
    )
    with (RESULTS / "submission_strict_channel_attribution_summary_easytsad.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    feature_specs = (
        ("Anomaly ratio", "anomaly_ratio"),
        ("Trend", "trend_strength"),
        ("Periodicity", "periodicity"),
        ("Roughness", "roughness"),
    )
    groups = DATASETS + ("All",)
    correlations = np.full((len(feature_specs), len(groups)), np.nan, dtype=float)
    for group_index, group in enumerate(groups):
        selected = records if group == "All" else [record for record in records if record["dataset"] == group]
        gains = np.asarray([record["final_gain_vs_guard"] for record in selected], dtype=float)
        for feature_index, (_display, field) in enumerate(feature_specs):
            values = np.asarray([record[field] for record in selected], dtype=float)
            finite = np.isfinite(values) & np.isfinite(gains)
            if int(np.sum(finite)) >= 5 and np.unique(values[finite]).size > 1:
                correlations[feature_index, group_index] = float(spearmanr(values[finite], gains[finite]).statistic)

    fig, (bars, heat) = plt.subplots(
        2, 1, figsize=(3.5, 3.65), gridspec_kw={"height_ratios": (1.05, 1.0), "hspace": 0.90}
    )
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS), dtype=float)
    channel_colors = (COLORS["purple"], COLORS["teal"], COLORS["coral"], COLORS["gold"])
    for index, (label, color) in enumerate(zip(labels, channel_colors)):
        bars.bar(
            x, fractions[:, index], bottom=bottom, width=0.68, color=color, alpha=0.88,
            edgecolor="white", linewidth=0.35, label=label,
        )
        bottom += fractions[:, index]
    bars.plot(
        x, win_rates, color=COLORS["ink"], marker="D", ms=3.2, lw=1.15,
        label="Zero+ > guard",
    )
    bars.set_xticks(x, DATASETS, rotation=20, ha="right")
    bars.set_ylim(0, 1.04)
    bars.set_ylabel("Valid-series share")
    bars.grid(axis="y", color=COLORS["grid"], lw=0.45)
    bars.spines[["top", "right"]].set_visible(False)
    bars.set_title("(a) Best evidence channel and fusion win rate", loc="left", weight="bold")
    bars.legend(
        frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.31),
        columnspacing=0.55, handletextpad=0.25,
    )

    for feature_index in range(len(feature_specs)):
        for group_index in range(len(groups)):
            value = correlations[feature_index, group_index]
            if not np.isfinite(value):
                continue
            color = COLORS["teal"] if value >= 0 else COLORS["coral"]
            heat.scatter(
                group_index, feature_index, s=20 + 300 * abs(value), color=color,
                alpha=0.80, edgecolor="white", linewidth=0.45,
            )
            heat.text(
                group_index, feature_index, f"{value:+.2f}", ha="center", va="center",
                fontsize=4.8, weight="bold",
                color="white" if abs(value) > 0.24 else COLORS["ink"],
            )
    heat.set_xticks(range(len(groups)), groups, rotation=24, ha="right")
    heat.set_yticks(range(len(feature_specs)), [display for display, _field in feature_specs])
    heat.set_xlim(-0.55, len(groups) - 0.45)
    heat.set_ylim(len(feature_specs) - 0.55, -0.55)
    heat.spines[:].set_visible(False)
    heat.tick_params(length=0)
    heat.set_title(r"(b) KPI trait vs. neural marginal gain (Spearman $\rho$)", loc="left", weight="bold")
    heat.text(
        0.99, -0.27, "teal: positive   coral: negative", transform=heat.transAxes,
        ha="right", fontsize=5.2, color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.23, right=0.99, top=0.98, bottom=0.12)
    save(fig, "fig_strict_channel_attribution")


def draw_effectiveness_bubble(ledger: Sequence[Mapping[str, str]]) -> None:
    lookup = {(row["method_id"], row["dataset"]): row for row in ledger}
    footprint_rows = rows(RESULTS / "submission_efficiency_frontier.csv")
    footprint = {row["source_method"]: fnum(row["footprint_mib"]) for row in footprint_rows}
    specs = (
        ("TShape-Zero+", FINAL, "TShape-zero_plus", COLORS["purple"]),
        ("Chronos", "Chronos-Bolt-Tiny", "Chronos-Bolt-Tiny", COLORS["teal"]),
        ("Timer", "Timer-base-84m", "Timer-base-84m", COLORS["blue"]),
        ("TimesFM", "TimesFM-2.5-200M", "TimesFM-2.5-200M", COLORS["coral"]),
        ("Sundial", "Sundial-base-128m", "Sundial-base-128m", COLORS["gold"]),
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    offsets = {
        "TShape-Zero+": (5, 5),
        "Chronos": (5, 7),
        "Timer": (5, 5),
        "TimesFM": (-5, -13),
        "Sundial": (5, 5),
    }
    for label, method, footprint_id, color in specs:
        point = np.mean([fnum(lookup[(method, dataset)]["point_f1"]) for dataset in DATASETS])
        event = np.mean([fnum(lookup[(method, dataset)]["event_f1"]) for dataset in DATASETS])
        utility = 0.5 * (point + event)
        size = footprint[footprint_id]
        ax.scatter(
            size,
            utility,
            s=45 + 135 * event,
            color=color,
            alpha=0.84,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy = offsets[label]
        ax.annotate(
            label,
            (size, utility),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.4,
            ha="right" if dx < 0 else "left",
            va="top" if dy < 0 else "bottom",
            clip_on=False,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Checkpoint footprint (MiB, log scale)")
    ax.set_ylabel("Macro mean of Point-F1 and Event-F1")
    ax.grid(color=COLORS["grid"], lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    utilities = [
        0.5
        * (
            np.mean([fnum(lookup[(method, dataset)]["point_f1"]) for dataset in DATASETS])
            + np.mean([fnum(lookup[(method, dataset)]["event_f1"]) for dataset in DATASETS])
        )
        for _, method, _, _ in specs
    ]
    ax.set_ylim(min(utilities) - 0.010, max(utilities) + 0.008)
    ax.set_title("Checkpoint effectiveness--footprint frontier", loc="left", weight="bold")
    save(fig, "fig_strict_effectiveness_bubble")


def draw_attention() -> None:
    import torch

    checkpoint = ROOT / "models" / "tshape_zero_plus_release.pt"
    payload = torch.load(checkpoint, map_location="cpu")
    metadata = payload["metadata"]
    p = int(metadata.get("p", 16))
    model = TShape_model(p)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    case = json.loads((RESULTS / "product_demo_case_scores.json").read_text(encoding="utf-8"))
    values = np.asarray(case["input_values"], dtype=np.float32)
    prepared, _, _ = prepare_target_context(values, int(metadata.get("diff_order", 1)))
    top_indices = [int(row["index"]) for row in case["top_anomalies"][:5]]
    positions = [index - 1 for index in top_indices if index - 1 >= p]
    contexts = np.stack([prepared[position - p : position] for position in positions]).astype(np.float32)

    captured: dict[str, np.ndarray] = {}
    layer = model.transformer_layers[0]
    hooks = [
        layer.local_feature.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("local", output[1].detach().cpu().numpy())
        ),
        layer.global_attention.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("global", output[1].detach().cpu().numpy())
        ),
        layer.fusion_gate.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("gate", output.detach().cpu().numpy())
        ),
    ]
    with torch.no_grad():
        model(torch.from_numpy(contexts))
    for hook in hooks:
        hook.remove()

    global_attention = np.mean(captured["global"], axis=0)
    local_attention = np.mean(captured["local"], axis=0).reshape(8, 8, 8, 8).mean(axis=(1, 3))
    gate = np.mean(captured["gate"], axis=0).reshape(4, 8, 8).mean(axis=2)
    mean_context = np.mean(contexts, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.05))
    ax = axes[0, 0]
    ax.plot(np.arange(p), mean_context, color=COLORS["teal"], lw=1.35)
    for boundary in range(0, p + 1, 4):
        ax.axvline(boundary - 0.5, color=COLORS["grid"], lw=0.55)
    ax.scatter(np.arange(p), mean_context, s=7, color=COLORS["teal"], edgecolor="white", lw=0.3)
    ax.set_title("(a) Top-5 incident contexts", loc="left", weight="bold")
    ax.set_xlabel("Context index / four patches")
    ax.set_ylabel("Normalized value")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45)

    axes[0, 1].imshow(global_attention, cmap="YlGnBu", aspect="equal")
    for i in range(4):
        for j in range(4):
            axes[0, 1].text(
                j,
                i,
                f"{global_attention[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.2,
                color="white" if global_attention[i, j] > 0.42 else COLORS["ink"],
            )
    axes[0, 1].set_xticks(range(4), [f"P{i+1}" for i in range(4)])
    axes[0, 1].set_yticks(range(4), [f"P{i+1}" for i in range(4)])
    axes[0, 1].set_title("(b) Global patch attention", loc="left", weight="bold")
    axes[0, 1].tick_params(length=0)

    axes[1, 0].imshow(gate, cmap="PuBuGn", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set_xticks(range(8), [f"C{i+1}" for i in range(8)], rotation=45, ha="right")
    axes[1, 0].set_yticks(range(4), [f"P{i+1}" for i in range(4)])
    axes[1, 0].set_title("(c) Local/global fusion gate", loc="left", weight="bold")
    axes[1, 0].tick_params(length=0)
    axes[1, 0].text(
        0.5,
        -0.24,
        "0 = global   |   1 = local",
        transform=axes[1, 0].transAxes,
        ha="center",
        color=COLORS["muted"],
        fontsize=5.3,
    )

    axes[1, 1].imshow(local_attention, cmap="YlOrRd", aspect="equal")
    axes[1, 1].set_xticks((0, 3, 7), ("C1", "C4", "C8"))
    axes[1, 1].set_yticks((0, 3, 7), ("C1", "C4", "C8"))
    axes[1, 1].set_title("(d) Grouped local attention", loc="left", weight="bold")
    axes[1, 1].tick_params(length=0)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.11, wspace=0.38, hspace=0.45)
    save(fig, "fig_strict_attention")


def main() -> None:
    configure()
    ledger = rows(RESULTS / "submission_easytsad_all_methods.csv")
    detail = strict_detail()
    summaries, runs = sweep_artifacts()
    draw_intro_mechanism()
    draw_motivation_mechanism()
    draw_framework_mechanism()
    draw_tshape_architecture()
    draw_overall(ledger)
    draw_pattern_training(runs)
    draw_pattern_effect(summaries)
    draw_guard_distributions(detail)
    draw_guard_bars(ledger)
    draw_guard_ablation(ledger)
    draw_channel_attribution(detail)
    draw_effectiveness_bubble(ledger)
    draw_attention()
    print(f"Wrote strict zero-shot figures to {FIGURES}")


if __name__ == "__main__":
    main()
