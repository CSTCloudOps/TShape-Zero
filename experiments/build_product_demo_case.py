#!/usr/bin/env python3
"""Regenerate the fixed real-AIOPS web-demo case through the product scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rene_experiments import easytsad_pa_metrics
from tshape_zero_product import DEFAULT_CHECKPOINT, DEFAULT_DEMO_CASE, ROOT, score_array


DEFAULT_SERIES = "431a8542-c468-3988-a508-3afd06a218da"


def count_events(labels: np.ndarray) -> int:
    indices = np.flatnonzero(labels)
    if len(indices) == 0:
        return 0
    return 1 + int(np.sum(np.diff(indices) > 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--series", default=DEFAULT_SERIES)
    parser.add_argument("--start", type=int, default=95100)
    parser.add_argument("--end", type=int, default=95340)
    parser.add_argument("--output", type=Path, default=DEFAULT_DEMO_CASE)
    parser.add_argument("--allow-nonholdout", action="store_true")
    args = parser.parse_args()

    series_dir = ROOT / "datasets" / "UTS" / "AIOPS" / args.series
    values = np.asarray(np.load(series_dir / "test.npy"), dtype=np.float32).reshape(-1)[args.start : args.end]
    labels = np.asarray(np.load(series_dir / "test_label.npy"), dtype=np.int8).reshape(-1)[args.start : args.end]
    result = score_array(values, args.checkpoint, device="cpu")
    excluded_target = result["metadata"].get("excluded_target")
    strict_checkpoint = bool(result["metadata"].get("strict_zero_shot")) and not bool(
        result["metadata"].get("target_values_used_for_training", True)
    )
    if not strict_checkpoint and excluded_target != "AIOPS" and not args.allow_nonholdout:
        raise ValueError(
            "The paper case requires either a strict benchmark-free checkpoint "
            "or an AIOPS-holdout diagnostic checkpoint."
        )
    offset = int(result["score_index_offset"])
    scores = np.asarray(result["scores"], dtype=np.float64)
    tshape_scores = np.asarray(result["tshape_scores"], dtype=np.float64)
    guard_scores = np.asarray(result["residual_guard_scores"], dtype=np.float64)
    scored_labels = labels[offset : offset + len(scores)]
    zero_plus_metrics = easytsad_pa_metrics(scored_labels, scores)
    tshape_metrics = easytsad_pa_metrics(scored_labels, tshape_scores)
    guard_metrics = easytsad_pa_metrics(scored_labels, guard_scores)
    top = []
    for item in result["top_anomalies"]:
        local_index = int(item["index"])
        top.append(
            {
                **item,
                "global_index": args.start + local_index,
                "label": int(labels[local_index]),
            }
        )
    tshape_top = np.argsort(-tshape_scores)[: min(10, len(tshape_scores))]
    case = {
        "case_name": "AIOPS complex multi-event KPI window",
        "dataset": "AIOPS",
        "series": args.series,
        "window_start": args.start,
        "window_end": args.end,
        "input_points": int(len(values)),
        "anomaly_points": int(np.sum(labels)),
        "anomaly_events": count_events(labels),
        "scored_anomaly_points": int(np.sum(scored_labels)),
        "local_point_f1_pa": float(zero_plus_metrics["point_f1_pa"]),
        "local_event_f1_pa_log": float(zero_plus_metrics["event_f1_pa_log"]),
        "local_tshape_point_f1_pa": float(tshape_metrics["point_f1_pa"]),
        "local_tshape_event_f1_pa_log": float(tshape_metrics["event_f1_pa_log"]),
        "local_guard_point_f1_pa": float(guard_metrics["point_f1_pa"]),
        "local_guard_event_f1_pa_log": float(guard_metrics["event_f1_pa_log"]),
        "top10_label_hits": int(sum(item["label"] for item in top)),
        "tshape_top10_label_hits": int(np.sum(scored_labels[tshape_top])),
        "selection_note": "Selected after model freezing as a multi-event product illustration with changing baseline and at least four event phases; excluded from aggregate evaluation.",
        "description": "Real AIOPS KPI window with trend drift, burst/plateau incidents, recovery, and repeated escalation; no injected incidents.",
        "checkpoint_role": result["metadata"].get("checkpoint_role"),
        "checkpoint_excluded_target": excluded_target,
        "checkpoint_strict_zero_shot": strict_checkpoint,
        "checkpoint_training_corpus": result["metadata"].get("training_corpus"),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "fusion_alpha": result["fusion_alpha"],
        "calibration_scope": result["calibration_scope"],
    }
    payload = {
        "case": case,
        **case,
        "metadata": result["metadata"],
        "input_values": [float(value) for value in values],
        "labels": [int(value) for value in labels],
        "scores": result["scores"],
        "tshape_scores": result["tshape_scores"],
        "residual_guard_scores": result["residual_guard_scores"],
        "top_anomalies": top,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.output}: Point-F1={payload['local_point_f1_pa']:.3f}, "
        f"Event-F1={payload['local_event_f1_pa_log']:.3f}, "
        f"top10_hits={payload['top10_label_hits']}"
    )


if __name__ == "__main__":
    main()
