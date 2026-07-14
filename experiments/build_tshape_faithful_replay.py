#!/usr/bin/env python3
"""Merge stored and newly re-executed TShape scores under EasyTSAD metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

import rene_experiments as re


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "Results" / "Scores" / "TShape" / "naive"
REEXECUTED = ROOT / "Results" / "Scores" / "TShapeReproduction" / "naive"
RESULTS = ROOT / "Results" / "RENE"
DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")


def write_rows(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    detail: List[Dict[str, object]] = []
    expected = {dataset: len(re.iter_series([dataset])) for dataset in DATASETS}
    for dataset in DATASETS:
        refs = {ref.name: ref for ref in re.iter_series([dataset])}
        for series, ref in sorted(refs.items()):
            reexecuted = REEXECUTED / dataset / f"{series}.npy"
            stored = ORIGINAL / dataset / f"{series}.npy"
            if reexecuted.exists():
                score_path = reexecuted
                provenance = "faithful local re-execution"
            elif stored.exists():
                score_path = stored
                provenance = "stored artifact replay"
            else:
                continue
            labels = np.asarray(np.load(ref.path / "test_label.npy"), dtype=np.int8).reshape(-1)
            scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
            offset = len(labels) - len(scores)
            if offset >= 0:
                labels = labels[offset:]
            else:
                scores = scores[-len(labels) :]
                offset = 0
            row = re.metric_row(
                dataset,
                series,
                "TShape-faithful-reproduction",
                labels,
                scores,
            )
            row.update(
                {
                    "seed": "stored" if provenance.startswith("stored") else "20260704",
                    "provenance": provenance,
                    "score_file": str(score_path.relative_to(ROOT)),
                    "label_offset": offset,
                }
            )
            detail.append(row)

    summary = re.aggregate_rows(detail, expected)
    provenance_counts: Dict[str, Dict[str, int]] = {}
    for row in detail:
        counts = provenance_counts.setdefault(str(row["dataset"]), {})
        key = str(row["provenance"])
        counts[key] = counts.get(key, 0) + 1
    for row in summary:
        counts = provenance_counts.get(str(row["dataset"]), {})
        row["stored_series"] = counts.get("stored artifact replay", 0)
        row["reexecuted_series"] = counts.get("faithful local re-execution", 0)

    detail_fields = [
        "dataset",
        "series",
        "method",
        "seed",
        "provenance",
        "score_file",
        "label_offset",
        "points",
        "anomaly_ratio",
        "point_f1_pa",
        "point_precision_pa",
        "point_recall_pa",
        "point_threshold_pa",
        "event_f1_pa_log",
        "event_precision_pa_log",
        "event_recall_pa_log",
        "event_threshold_pa_log",
    ]
    summary_fields = [
        "dataset",
        "method",
        "series",
        "rows",
        "seeds",
        "points",
        "valid_point_pa_series",
        "valid_event_pa_series",
        "zero_anomaly_series",
        "coverage_ratio",
        "stored_series",
        "reexecuted_series",
        "point_f1_pa",
        "point_f1_pa_std",
        "event_f1_pa_log",
        "event_f1_pa_log_std",
    ]
    detail_path = RESULTS / "tshape_faithful_reproduction_series_metrics.csv"
    summary_path = RESULTS / "tshape_faithful_reproduction_summary_metrics.csv"
    write_rows(detail_path, detail, detail_fields)
    write_rows(summary_path, summary, summary_fields)
    missing = {
        dataset: expected[dataset]
        - len({row["series"] for row in detail if row["dataset"] == dataset})
        for dataset in DATASETS
    }
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Missing series: {missing}")


if __name__ == "__main__":
    main()
