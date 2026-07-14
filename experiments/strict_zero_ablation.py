#!/usr/bin/env python3
"""Full-series fusion ablations for the frozen synthetic TShape checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from layyer import TShape_model
from rene_experiments import (
    OUT_ROOT,
    ROOT,
    aggregate_rows,
    baseline_scores,
    iter_series,
    load_prepared,
    metric_row,
    rank01,
    scale01,
    tshape_scores,
    write_csv,
)


DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")


def load_checkpoint(path: Path, device: str):
    import torch

    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    if not metadata.get("strict_zero_shot", False):
        raise ValueError("Fusion ablation requires a strict zero-shot checkpoint.")
    p = int(metadata.get("p", 16))
    model = TShape_model(p)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


def score_variants(neural: np.ndarray, test: np.ndarray, p: int) -> dict[str, np.ndarray]:
    median = baseline_scores(test, p, "rolling_median")
    spectral = baseline_scores(test, p, "spectral_residual")
    mad = baseline_scores(test, p, "rolling_mad")
    n_neural = scale01(neural)
    n_median = scale01(median)
    n_spectral = scale01(spectral)
    n_mad = scale01(mad)
    guard = (0.55 * n_median + 0.25 * n_spectral + 0.05 * n_mad) / 0.85

    variants = {
        "StrictGuardOnly": guard,
        "StrictTShapeMedian": 0.5 * n_neural + 0.5 * n_median,
        "StrictTShapeSpectral": 0.5 * n_neural + 0.5 * n_spectral,
        "StrictZeroPlusRank015": (
            0.15 * rank01(neural)
            + 0.55 * rank01(median)
            + 0.25 * rank01(spectral)
            + 0.05 * rank01(mad)
        ),
    }
    for alpha in (0.05, 0.10, 0.25, 0.50):
        variants[f"StrictZeroPlusAlpha{int(round(alpha * 100)):03d}"] = (
            alpha * n_neural + (1.0 - alpha) * guard
        )
    return {name: np.asarray(values, dtype=np.float32) for name, values in variants.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "strict_pattern_main" / "pattern_bank_100_seed_20260704.pt",
    )
    parser.add_argument("--targets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-series-per-target", type=int)
    parser.add_argument("--tag", default="submission_strict_zero_ablation")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    device = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, metadata = load_checkpoint(args.checkpoint, device)
    p = int(metadata.get("p", 16))
    diff_order = int(metadata.get("diff_order", 1))
    expected = {dataset: len(iter_series([dataset])) for dataset in args.targets}
    rows: list[dict[str, object]] = []
    started = time.time()
    evaluated = 0

    for dataset in args.targets:
        refs = iter_series([dataset])
        if args.max_series_per_target is not None:
            refs = refs[: args.max_series_per_target]
        for index, ref in enumerate(refs, start=1):
            prepared = load_prepared(ref, p=p, diff_order=diff_order)
            if prepared is None:
                continue
            neural = tshape_scores(model, prepared.test, p, args.eval_batch_size, 1)
            for method, scores in score_variants(neural, prepared.test, p).items():
                row = metric_row(dataset, ref.name, method, prepared.labels, scores)
                row["seed"] = args.seed
                rows.append(row)
            evaluated += 1
            if index == 1 or index == len(refs) or index % args.progress_every == 0:
                print(f"ablation {dataset}: {index}/{len(refs)} series", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"strict_zero_ablation_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"strict_zero_ablation_summary_metrics{suffix}.csv"
    run_path = OUT_ROOT / f"strict_zero_ablation_run_info{suffix}.json"
    fields = ["dataset", "series", "seed", "method", "points", "anomaly_ratio"]
    metric_fields = [key for key in rows[0] if key not in fields] if rows else []
    write_csv(detail_path, rows, fields + metric_fields)
    summary = aggregate_rows(rows, expected)
    summary_fields = list(summary[0]) if summary else ["dataset", "method"]
    write_csv(summary_path, summary, summary_fields)
    run_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "targets": list(args.targets),
                "strict_zero_shot": True,
                "checkpoint_training_corpus": metadata.get("training_corpus"),
                "checkpoint_sha256_scope": "recorded by release packager",
                "device": device,
                "eval_batch_size": args.eval_batch_size,
                "eval_series": evaluated,
                "seconds": time.time() - started,
                "methods": sorted({str(row["method"]) for row in rows}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {run_path}")


if __name__ == "__main__":
    main()
