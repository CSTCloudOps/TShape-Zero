#!/usr/bin/env python3
"""Draw the camera-ready Event-F1 and Point-F1 radar panels for Fig. 3."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Results" / "RENE" / "submission_tshape_dual_radar.csv"
DEFAULT_OUTPUT = ROOT / "paper" / "figures_submission" / "fig05_original_vs_zero"

DATASETS = ["AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo"]
INK = "#1f2430"
MUTED = "#64748b"
BLUE = "#2f6f9f"
TEAL = "#2a9d8f"
CORAL = "#e76f51"

METRICS = {
    "event": {
        "title": "(a) Event-F1",
        "reported": "original_reported_event_f1",
        "replayed": "reproduction_easytsad_event_f1_log",
        "zero": "zero_shot_easytsad_event_f1_log",
    },
    "point": {
        "title": "(b) Point-F1",
        "reported": "original_reported_point_f1",
        "replayed": "reproduction_easytsad_point_f1_pa",
        "zero": "zero_shot_easytsad_point_f1_pa",
    },
}

# Exact values plotted in Fig. 3. Yahoo was not part of the original paper's
# five-benchmark table, so its two paper-reported entries are intentionally N/A.
EMBEDDED_DATA = {
    "AIOPS": {
        "original_reported_event_f1": 0.804900,
        "reproduction_easytsad_event_f1_log": 0.818067,
        "zero_shot_easytsad_event_f1_log": 0.768656,
        "original_reported_point_f1": 0.926300,
        "reproduction_easytsad_point_f1_pa": 0.920105,
        "zero_shot_easytsad_point_f1_pa": 0.870431,
    },
    "NAB": {
        "original_reported_event_f1": 0.918600,
        "reproduction_easytsad_event_f1_log": 0.922756,
        "zero_shot_easytsad_event_f1_log": 0.877489,
        "original_reported_point_f1": 0.998200,
        "reproduction_easytsad_point_f1_pa": 0.998387,
        "zero_shot_easytsad_point_f1_pa": 0.998330,
    },
    "TODS": {
        "original_reported_event_f1": 0.856100,
        "reproduction_easytsad_event_f1_log": 0.835495,
        "zero_shot_easytsad_event_f1_log": 0.618372,
        "original_reported_point_f1": 0.908500,
        "reproduction_easytsad_point_f1_pa": 0.887597,
        "zero_shot_easytsad_point_f1_pa": 0.693578,
    },
    "UCR": {
        "original_reported_event_f1": 0.591500,
        "reproduction_easytsad_event_f1_log": 0.590152,
        "zero_shot_easytsad_event_f1_log": 0.363671,
        "original_reported_point_f1": 0.849300,
        "reproduction_easytsad_point_f1_pa": 0.849411,
        "zero_shot_easytsad_point_f1_pa": 0.618884,
    },
    "WSD": {
        "original_reported_event_f1": 0.913700,
        "reproduction_easytsad_event_f1_log": 0.901868,
        "zero_shot_easytsad_event_f1_log": 0.879338,
        "original_reported_point_f1": 0.982900,
        "reproduction_easytsad_point_f1_pa": 0.972524,
        "zero_shot_easytsad_point_f1_pa": 0.958089,
    },
    "Yahoo": {
        "original_reported_event_f1": None,
        "reproduction_easytsad_event_f1_log": 0.791030,
        "zero_shot_easytsad_event_f1_log": 0.665050,
        "original_reported_point_f1": None,
        "reproduction_easytsad_point_f1_pa": 0.807939,
        "zero_shot_easytsad_point_f1_pa": 0.675379,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the side-by-side EasyTSAD Event-F1 and Point-F1 radar plots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional EasyTSAD replay CSV. Omit this argument to use the exact "
            "values embedded in this script."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without an extension; PDF, SVG, and PNG are written.",
    )
    parser.add_argument("--dpi", type=int, default=360, help="PNG resolution.")
    return parser.parse_args()


def read_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["dataset"]: row for row in csv.DictReader(handle)}

    missing_datasets = [dataset for dataset in DATASETS if dataset not in rows]
    required_fields = {
        "dataset",
        *(field for metric in METRICS.values() for field in metric.values() if field != metric["title"]),
    }
    available_fields = set(next(iter(rows.values()), {}))
    missing_fields = sorted(required_fields - available_fields)
    if missing_datasets or missing_fields:
        raise ValueError(
            f"Invalid radar CSV: missing datasets={missing_datasets}, "
            f"missing fields={missing_fields}"
        )
    return rows


def numeric_values(rows: Dict[str, Dict[str, object]], field: str) -> np.ndarray:
    values: List[float] = []
    for dataset in DATASETS:
        value = rows[dataset][field]
        if value is None:
            values.append(np.nan)
        elif isinstance(value, str) and value.strip().upper() in {"N/A", "NA", "NAN", ""}:
            values.append(np.nan)
        else:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def closed(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("Reproduction and zero-shot radar series must be finite.")
    return np.r_[values, values[0]]


def draw_panel(
    fig: plt.Figure,
    position: Iterable[float],
    angles: np.ndarray,
    reported: np.ndarray,
    replayed: np.ndarray,
    zero_shot: np.ndarray,
) -> plt.Axes:
    axis = fig.add_axes(position, polar=True)
    axis.set_facecolor("white")
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles)
    axis.set_xticklabels(DATASETS, fontsize=5.4, color=INK)
    axis.tick_params(axis="x", pad=0)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=4.2, color=MUTED)
    axis.set_rlabel_position(12)
    axis.yaxis.grid(True, color="#cbd5e1", linewidth=0.5)
    axis.xaxis.grid(True, color="#dbe3eb", linewidth=0.5)
    axis.spines["polar"].set_color("#9aa9b7")
    axis.spines["polar"].set_linewidth(0.7)

    closed_angles = np.r_[angles, angles[0]]
    finite_reported = np.isfinite(reported)
    axis.plot(
        angles[finite_reported],
        reported[finite_reported],
        color=BLUE,
        linewidth=1.45,
        marker="o",
        markersize=2.1,
        label="Paper-reported",
    )
    axis.plot(
        closed_angles,
        closed(replayed),
        color=TEAL,
        linewidth=1.35,
        linestyle="--",
        marker="s",
        markersize=1.9,
        label="Our reproduction",
    )
    axis.fill(closed_angles, closed(replayed), color=TEAL, alpha=0.065)
    axis.plot(
        closed_angles,
        closed(zero_shot),
        color=CORAL,
        linewidth=1.35,
        linestyle="-.",
        marker="^",
        markersize=2.0,
        label="Direct Zero-shot",
    )
    axis.fill(closed_angles, closed(zero_shot), color=CORAL, alpha=0.05)

    for index in np.flatnonzero(~finite_reported):
        axis.text(
            angles[index],
            0.56,
            "N/A",
            ha="center",
            va="center",
            fontsize=4.5,
            color=BLUE,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#b8c7d4",
                "linewidth": 0.45,
            },
        )
    return axis


def draw(input_path: Optional[Path], output_stem: Path, dpi: int) -> None:
    rows = read_rows(input_path) if input_path is not None else EMBEDDED_DATA
    event = METRICS["event"]
    point = METRICS["point"]
    angles = np.linspace(0, 2 * np.pi, len(DATASETS), endpoint=False)

    fig = plt.figure(figsize=(3.55, 2.25), facecolor="white")
    event_axis = draw_panel(
        fig,
        [0.055, 0.235, 0.39, 0.62],
        angles,
        numeric_values(rows, event["reported"]),
        numeric_values(rows, event["replayed"]),
        numeric_values(rows, event["zero"]),
    )
    draw_panel(
        fig,
        [0.555, 0.235, 0.39, 0.62],
        angles,
        numeric_values(rows, point["reported"]),
        numeric_values(rows, point["replayed"]),
        numeric_values(rows, point["zero"]),
    )

    fig.text(0.25, 0.985, event["title"], ha="center", va="top", fontsize=6.6, weight="bold", color=INK)
    fig.text(0.75, 0.985, point["title"], ha="center", va="top", fontsize=6.6, weight="bold", color=INK)
    handles, labels = event_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=5.0,
        handlelength=1.7,
        columnspacing=0.65,
        frameon=False,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve() if args.input is not None else None
    draw(input_path, args.output.resolve(), args.dpi)
    source = input_path if input_path is not None else "embedded data"
    print(f"Data source: {source}")
    print(f"Wrote {args.output.resolve()}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
