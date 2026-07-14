#!/usr/bin/env python3
"""Export the compact EasyTSAD-only evidence bundle used by the strict paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results" / "RENE"
DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")
SEED = 20260704


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pattern_paths(percent: int) -> tuple[Path, Path, Path]:
    if percent == 100:
        tag = "submission_strict_zero_main_full"
    else:
        tag = f"submission_strict_zero_size_{percent}"
    return (
        RESULTS / f"pattern_bank_tshape_summary_metrics_{tag}.csv",
        RESULTS / f"pattern_bank_tshape_series_metrics_{tag}.csv",
        RESULTS / f"pattern_bank_tshape_run_info_{tag}.json",
    )


def export_pattern_sweep() -> None:
    summary_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for percent in range(10, 101, 10):
        summary_path, series_path, run_path = pattern_paths(percent)
        for path in (summary_path, series_path, run_path):
            if not path.exists():
                raise FileNotFoundError(path)
        runs = json.loads(run_path.read_text(encoding="utf-8"))
        if len(runs) != 1:
            raise RuntimeError(f"Expected one run for {percent}%, found {len(runs)}")
        run = runs[0]
        if int(run["seed"]) != SEED or int(run["fraction_pct"]) != percent:
            raise RuntimeError(f"Unexpected Pattern Bank provenance at {percent}%")
        losses = [float(value) for value in run["loss"]]
        for epoch, loss in enumerate(losses, start=1):
            training_rows.append(
                {
                    "fraction_pct": percent,
                    "synthetic_windows": int(run["windows"]),
                    "epoch": epoch,
                    "training_mse": f"{loss:.9f}",
                    "seed": SEED,
                }
            )
        manifest_rows.append(
            {
                "fraction_pct": percent,
                "synthetic_windows": int(run["windows"]),
                "epochs_completed": int(run["epochs_completed"]),
                "early_stopped": bool(run["early_stopped"]),
                "best_training_mse": f"{float(run['best_training_loss']):.9f}",
                "resolved_device": str(run.get("resolved_device", "mps")),
                "evaluated_series": int(run["eval_series"]),
                "eval_stride": int(run["eval_stride"]),
                "seed": SEED,
            }
        )
        summaries = read_csv(summary_path)
        if {row["dataset"] for row in summaries} != set(DATASETS):
            raise RuntimeError(f"Incomplete summary datasets at {percent}%")
        for row in summaries:
            method = row["method"]
            channel = "TShape-Zero+" if "ZeroPlus" in method else "Pattern-only TShape-Zero"
            summary_rows.append(
                {
                    "fraction_pct": percent,
                    "synthetic_windows": int(float(row["synthetic_windows"])),
                    "dataset": row["dataset"],
                    "channel": channel,
                    "point_f1": f"{float(row['point_f1_pa']):.6f}",
                    "event_f1": f"{float(row['event_f1_pa_log']):.6f}",
                    "point_f1_series_std": f"{float(row['point_f1_pa_std']):.6f}",
                    "event_f1_series_std": f"{float(row['event_f1_pa_log_std']):.6f}",
                    "valid_point_series": int(float(row["valid_point_pa_series"])),
                    "valid_event_series": int(float(row["valid_event_pa_series"])),
                    "all_normal_series": int(float(row["zero_anomaly_series"])),
                    "coverage_ratio": f"{float(row['coverage_ratio']):.6f}",
                    "seed": SEED,
                }
            )
        for row in read_csv(series_path):
            series_rows.append(
                {
                    "fraction_pct": percent,
                    "synthetic_windows": int(float(row["synthetic_windows"])),
                    "dataset": row["dataset"],
                    "series": row["series"],
                    "channel": "TShape-Zero+" if "ZeroPlus" in row["method"] else "Pattern-only TShape-Zero",
                    "points": int(float(row["points"])),
                    "anomaly_ratio": f"{float(row['anomaly_ratio']):.9f}",
                    "point_f1": row["point_f1_pa"],
                    "event_f1": row["event_f1_pa_log"],
                    "seed": SEED,
                }
            )

    summary_fields = (
        "fraction_pct", "synthetic_windows", "dataset", "channel", "point_f1", "event_f1",
        "point_f1_series_std", "event_f1_series_std", "valid_point_series",
        "valid_event_series", "all_normal_series", "coverage_ratio", "seed",
    )
    series_fields = (
        "fraction_pct", "synthetic_windows", "dataset", "series", "channel", "points",
        "anomaly_ratio", "point_f1", "event_f1", "seed",
    )
    training_fields = ("fraction_pct", "synthetic_windows", "epoch", "training_mse", "seed")
    manifest_fields = (
        "fraction_pct", "synthetic_windows", "epochs_completed", "early_stopped",
        "best_training_mse", "resolved_device", "evaluated_series", "eval_stride", "seed",
    )
    write_csv(RESULTS / "submission_strict_pattern_sweep_easytsad.csv", summary_rows, summary_fields)
    write_csv(RESULTS / "submission_strict_pattern_series_easytsad.csv", series_rows, series_fields)
    write_csv(RESULTS / "submission_strict_pattern_training.csv", training_rows, training_fields)
    write_csv(RESULTS / "submission_strict_pattern_manifest.csv", manifest_rows, manifest_fields)


def export_guard_ablation() -> None:
    source = read_csv(
        RESULTS / "strict_zero_ablation_summary_metrics_submission_strict_zero_ablation.csv"
    )
    rows: list[dict[str, object]] = []
    for row in source:
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "point_f1": row["point_f1_pa"],
                "event_f1": row["event_f1_pa_log"],
                "point_f1_series_std": row["point_f1_pa_std"],
                "event_f1_series_std": row["event_f1_pa_log_std"],
                "valid_point_series": row["valid_point_pa_series"],
                "valid_event_series": row["valid_event_pa_series"],
                "all_normal_series": row["zero_anomaly_series"],
                "coverage_ratio": row["coverage_ratio"],
                "seed": SEED,
            }
        )
    fields = (
        "dataset", "method", "point_f1", "event_f1", "point_f1_series_std",
        "event_f1_series_std", "valid_point_series", "valid_event_series",
        "all_normal_series", "coverage_ratio", "seed",
    )
    write_csv(RESULTS / "submission_strict_guard_ablation_easytsad.csv", rows, fields)


def main() -> None:
    export_pattern_sweep()
    export_guard_ablation()
    print("Exported strict EasyTSAD-only submission evidence.")


if __name__ == "__main__":
    main()
