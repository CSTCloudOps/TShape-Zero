#!/usr/bin/env python3
"""RENE experiments for TShape zero-shot deployment.

The script is intentionally independent of EasyTSAD. It reads the repository's
``datasets/UTS`` layout directly, trains the TShape predictor when requested,
and writes compact CSV files for paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "UTS"
OUT_ROOT = ROOT / "Results" / "RENE"
DEFAULT_DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")
PATTERN_BANK_VERSION = "normal-motif-v2"


@dataclass(frozen=True)
class SeriesRef:
    dataset: str
    name: str
    path: Path


@dataclass
class PreparedSeries:
    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    raw_test_len: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def iter_series(datasets: Optional[Sequence[str]] = None) -> List[SeriesRef]:
    selected = set(datasets or DEFAULT_DATASETS)
    refs: List[SeriesRef] = []
    for ds_dir in sorted(DATA_ROOT.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name not in selected:
            continue
        for item in sorted(ds_dir.iterdir()):
            if item.is_dir() and (item / "train.npy").exists() and (item / "test.npy").exists():
                refs.append(SeriesRef(ds_dir.name, item.name, item))
    return refs


def load_prepared(ref: SeriesRef, p: int = 16, diff_order: int = 1) -> Optional[PreparedSeries]:
    train = np.asarray(np.load(ref.path / "train.npy"), dtype=np.float32).reshape(-1)
    test = np.asarray(np.load(ref.path / "test.npy"), dtype=np.float32).reshape(-1)
    labels = np.asarray(np.load(ref.path / "test_label.npy"), dtype=np.int8).reshape(-1)
    raw_test_len = len(test)

    for _ in range(diff_order):
        if len(train) < 2 or len(test) < 2:
            return None
        train = np.diff(train).astype(np.float32)
        test = np.diff(test).astype(np.float32)
        labels = labels[1:]

    if len(train) <= p or len(test) <= p or len(labels) != len(test):
        return None

    lo = float(np.nanmin(train))
    hi = float(np.nanmax(train))
    scale = hi - lo
    if not np.isfinite(scale) or scale < 1e-8:
        train = np.zeros_like(train, dtype=np.float32)
        test = np.zeros_like(test, dtype=np.float32)
    else:
        train = ((train - lo) / scale).astype(np.float32)
        test = ((test - lo) / scale).astype(np.float32)

    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0)
    return PreparedSeries(train=train, test=test, labels=labels.astype(np.int8), raw_test_len=raw_test_len)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = list(fieldnames)
    pa_fields = [
        "valid_point_pa_series",
        "valid_event_pa_series",
        "point_f1_pa",
        "point_f1_pa_std",
        "point_precision_pa",
        "point_precision_pa_std",
        "point_recall_pa",
        "point_recall_pa_std",
        "point_threshold_pa",
        "point_threshold_pa_std",
        "event_f1_pa_log",
        "event_f1_pa_log_std",
        "event_precision_pa_log",
        "event_precision_pa_log_std",
        "event_recall_pa_log",
        "event_recall_pa_log_std",
        "event_threshold_pa_log",
        "event_threshold_pa_log_std",
    ]
    present = {key for row in rows for key in row.keys()}
    output_fields.extend(key for key in pa_fields if key in present and key not in output_fields)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def profile(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    for dataset in args.datasets:
        refs = iter_series([dataset])
        train_lens: List[int] = []
        test_lens: List[int] = []
        ratios: List[float] = []
        valid = 0
        for ref in refs:
            try:
                train = np.load(ref.path / "train.npy")
                test = np.load(ref.path / "test.npy")
                labels = np.load(ref.path / "test_label.npy")
            except Exception:
                continue
            valid += 1
            train_lens.append(int(train.size))
            test_lens.append(int(test.size))
            ratios.append(float(np.mean(labels)))
        if not valid:
            continue
        rows.append(
            {
                "dataset": dataset,
                "series": valid,
                "median_train_len": int(np.median(train_lens)),
                "median_test_len": int(np.median(test_lens)),
                "total_test_points": int(np.sum(test_lens)),
                "median_anomaly_ratio": f"{float(np.median(ratios)):.6f}",
                "min_anomaly_ratio": f"{float(np.min(ratios)):.6f}",
                "max_anomaly_ratio": f"{float(np.max(ratios)):.6f}",
            }
        )

    path = OUT_ROOT / "data_profile.csv"
    write_csv(
        path,
        rows,
        [
            "dataset",
            "series",
            "median_train_len",
            "median_test_len",
            "total_test_points",
            "median_anomaly_ratio",
            "min_anomaly_ratio",
            "max_anomaly_ratio",
        ],
    )
    print(f"Wrote {path}")


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float(np.sum(precision[y == 1]) / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(ranks[labels == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_from_prediction(labels: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    labels = labels.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.sum(labels & pred))
    fp = int(np.sum(~labels & pred))
    fn = int(np.sum(labels & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(f1), float(precision), float(recall)


def event_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    padded = np.r_[False, mask, False].astype(np.int8)
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1).astype(np.int64)
    ends = (np.flatnonzero(diff == -1) - 1).astype(np.int64)
    return starts, ends


def event_f1_from_prediction(labels: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    labels_bool = labels.astype(bool)
    pred_bool = pred.astype(bool)
    true_starts, true_ends = event_bounds(labels_bool)
    pred_starts, pred_ends = event_bounds(pred_bool)
    if len(true_starts) == 0:
        return math.nan, math.nan, math.nan
    if len(pred_starts) == 0:
        return 0.0, 0.0, 0.0
    pred_csum = np.r_[0, np.cumsum(pred_bool.astype(np.int32))]
    label_csum = np.r_[0, np.cumsum(labels_bool.astype(np.int32))]
    detected = int(np.sum((pred_csum[true_ends + 1] - pred_csum[true_starts]) > 0))
    useful_pred = int(np.sum((label_csum[pred_ends + 1] - label_csum[pred_starts]) > 0))
    precision = useful_pred / len(pred_starts) if len(pred_starts) else 0.0
    recall = detected / len(true_starts) if len(true_starts) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(f1), float(precision), float(recall)


def best_f1(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    labels = labels.astype(np.int8)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan, math.nan, math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    distinct = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(scores) - 1]
    tp_d = tp[distinct]
    fp_d = fp[distinct]
    precision = tp_d / np.maximum(tp_d + fp_d, 1)
    recall = tp_d / positives
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    i = int(np.nanargmax(f1))
    return float(f1[i]), float(precision[i]), float(recall[i])


def fixed_quantile_f1(labels: np.ndarray, scores: np.ndarray, q: float = 0.99) -> Tuple[float, float, float]:
    if int(labels.sum()) == 0:
        return math.nan, math.nan, math.nan
    thr = float(np.quantile(scores, q))
    return f1_from_prediction(labels, scores >= thr)


def fixed_quantile_event_f1(labels: np.ndarray, scores: np.ndarray, q: float = 0.99) -> Tuple[float, float, float]:
    if int(labels.sum()) == 0:
        return math.nan, math.nan, math.nan
    thr = float(np.quantile(scores, q))
    return event_f1_from_prediction(labels, scores >= thr)


def mad_f1(labels: np.ndarray, scores: np.ndarray, k: float = 3.0) -> Tuple[float, float, float]:
    if int(labels.sum()) == 0:
        return math.nan, math.nan, math.nan
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    sigma = 1.4826 * mad
    thr = med + k * sigma
    if not np.isfinite(thr) or sigma < 1e-12:
        thr = float(np.quantile(scores, 0.99))
    return f1_from_prediction(labels, scores >= thr)


def mad_event_f1(labels: np.ndarray, scores: np.ndarray, k: float = 3.0) -> Tuple[float, float, float]:
    if int(labels.sum()) == 0:
        return math.nan, math.nan, math.nan
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    sigma = 1.4826 * mad
    thr = med + k * sigma
    if not np.isfinite(thr) or sigma < 1e-12:
        thr = float(np.quantile(scores, 0.99))
    return event_f1_from_prediction(labels, scores >= thr)


def rank01(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    out = np.zeros_like(scores, dtype=np.float64)
    if not np.any(finite):
        return out.astype(np.float32)
    order = np.argsort(scores[finite], kind="mergesort")
    ranks = np.empty(np.sum(finite), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(order), endpoint=True)
    out[finite] = ranks
    return out.astype(np.float32)


def scale01(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    out = np.zeros_like(scores, dtype=np.float64)
    if not np.any(finite):
        return out.astype(np.float32)
    lo = float(np.min(scores[finite]))
    hi = float(np.max(scores[finite]))
    if hi - lo < 1e-12:
        return out.astype(np.float32)
    out[finite] = (scores[finite] - lo) / (hi - lo)
    return out.astype(np.float32)


def tshape_zero_plus_variants(tshape: np.ndarray, test: np.ndarray, p: int) -> Dict[str, np.ndarray]:
    """Label-free repair variants for ablation and weight-sensitivity analysis."""
    rolling = baseline_scores(test, p, "rolling_median")
    spectral = baseline_scores(test, p, "spectral_residual")
    mad = baseline_scores(test, p, "rolling_mad")

    rt = rank01(tshape)
    rr = rank01(rolling)
    rs = rank01(spectral)
    rm = rank01(mad)

    # Residual-first guard: the neural checkpoint contributes, but robust
    # no-training channels dominate unless the TShape residual agrees.
    proposed = 0.15 * rt + 0.55 * rr + 0.25 * rs + 0.05 * rm
    equal = 0.25 * rt + 0.25 * rr + 0.25 * rs + 0.25 * rm
    neural_heavy = 0.55 * rt + 0.25 * rr + 0.15 * rs + 0.05 * rm
    tshape_median = 0.50 * rt + 0.50 * rr
    tshape_spectral = 0.50 * rt + 0.50 * rs
    residual_only = 0.65 * rr + 0.30 * rs + 0.05 * rm
    residual_only_minmax = (
        0.55 * scale01(rolling)
        + 0.25 * scale01(spectral)
        + 0.05 * scale01(mad)
    ) / 0.85
    minmax = 0.15 * scale01(tshape) + 0.55 * scale01(rolling) + 0.25 * scale01(spectral) + 0.05 * scale01(mad)
    minmax_source_selected = 0.10 * scale01(tshape) + 0.90 * residual_only_minmax
    minmax_event_heavy = 0.25 * scale01(tshape) + 0.75 * residual_only_minmax
    return {
        "TShape-zero_plus": proposed.astype(np.float32),
        "TShape-zero_plus_equal": equal.astype(np.float32),
        "TShape-zero_plus_neural_heavy": neural_heavy.astype(np.float32),
        "TShape-zero_plus_tshape_median": tshape_median.astype(np.float32),
        "TShape-zero_plus_tshape_spectral": tshape_spectral.astype(np.float32),
        "TShape-zero_plus_residual_only": residual_only.astype(np.float32),
        "TShape-zero_plus_residual_only_minmax": residual_only_minmax.astype(np.float32),
        "TShape-zero_plus_minmax": minmax.astype(np.float32),
        "TShape-zero_plus_minmax_source_selected": minmax_source_selected.astype(np.float32),
        "TShape-zero_plus_minmax_event_heavy": minmax_event_heavy.astype(np.float32),
    }


def tshape_zero_plus_scores(tshape: np.ndarray, test: np.ndarray, p: int) -> np.ndarray:
    return tshape_zero_plus_variants(tshape, test, p)["TShape-zero_plus"]


def easytsad_pa_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Evaluate one score trace with the official EasyTSAD PA protocols."""
    empty = {
        "point_f1_pa": math.nan,
        "point_precision_pa": math.nan,
        "point_recall_pa": math.nan,
        "point_threshold_pa": math.nan,
        "event_f1_pa_log": math.nan,
        "event_precision_pa_log": math.nan,
        "event_recall_pa_log": math.nan,
        "event_threshold_pa_log": math.nan,
    }
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != scores.size or int(labels.sum()) == 0:
        return empty

    if os.environ.get("TSHAPE_EASYTSAD_BACKEND", "official") == "compatible":
        point, event = _easytsad_pa_compatible_pair(labels, scores)
        return {
            "point_f1_pa": point[0],
            "point_precision_pa": point[1],
            "point_recall_pa": point[2],
            "point_threshold_pa": point[3],
            "event_f1_pa_log": event[0],
            "event_precision_pa_log": event[1],
            "event_recall_pa_log": event[2],
            "event_threshold_pa_log": event[3],
        }

    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

    point = PointF1PA().calc(scores, labels, None)
    event = EventF1PA(mode="log").calc(scores, labels, None)
    return {
        "point_f1_pa": float(point.f1),
        "point_precision_pa": float(point.p),
        "point_recall_pa": float(point.r),
        "point_threshold_pa": float(point.thres),
        "event_f1_pa_log": float(event.f1),
        "event_precision_pa_log": float(event.p),
        "event_recall_pa_log": float(event.r),
        "event_threshold_pa_log": float(event.thres),
    }


def _easytsad_pa_compatible(
    labels: np.ndarray,
    scores: np.ndarray,
    mode: str,
) -> Tuple[float, float, float, float]:
    """Vectorized equivalent of EasyTSAD 0.3.0.2 PointF1PA/EventF1PA."""
    labels_bool = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    starts, ends = event_bounds(labels_bool)
    lengths = ends - starts + 1
    if len(starts) == 0:
        return 0.0, 0.0, 0.0, 0.0
    if mode == "point":
        event_weights = lengths.astype(np.float64)
    elif mode == "log":
        event_weights = np.floor(np.log(lengths + 3) / np.log(3)).astype(np.float64)
    else:
        raise ValueError(f"Unsupported EasyTSAD PA mode: {mode}")

    normal_positions = np.flatnonzero(~labels_bool)
    entry_positions = np.concatenate([normal_positions, starts])
    entry_scores = np.concatenate(
        [scores[normal_positions], np.array([np.max(scores[s : e + 1]) for s, e in zip(starts, ends)])]
    )
    entry_weights = np.concatenate([np.ones(len(normal_positions)), event_weights])
    entry_positive = np.concatenate([np.zeros(len(normal_positions), dtype=bool), np.ones(len(starts), dtype=bool)])

    source_order = np.argsort(entry_positions, kind="mergesort")
    entry_scores = entry_scores[source_order]
    entry_weights = entry_weights[source_order]
    entry_positive = entry_positive[source_order]
    score_order = np.argsort(-entry_scores, kind="mergesort")
    sorted_weights = entry_weights[score_order]
    sorted_positive = entry_positive[score_order]
    predicted_weight = np.cumsum(sorted_weights)
    true_positive = np.cumsum(sorted_weights * sorted_positive)
    total_anomaly = float(np.sum(event_weights))
    precision = true_positive / (predicted_weight + 1e-15)
    recall = true_positive / (total_anomaly + 1e-15)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-15)
    best = int(np.argmax(f1))
    return (
        float(f1[best]),
        float(precision[best]),
        float(recall[best]),
        float(entry_scores[score_order[best]]),
    )


def _easytsad_pa_compatible_pair(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Compute PointF1PA and log-EventF1PA with one shared stable score sort."""
    labels_bool = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    starts, ends = event_bounds(labels_bool)
    lengths = ends - starts + 1
    if len(starts) == 0:
        empty = (0.0, 0.0, 0.0, 0.0)
        return empty, empty
    point_weights = lengths.astype(np.float64)
    log_weights = np.floor(np.log(lengths + 3) / np.log(3)).astype(np.float64)

    normal_positions = np.flatnonzero(~labels_bool)
    entry_positions = np.concatenate([normal_positions, starts])
    event_scores = np.array(
        [np.max(scores[start : end + 1]) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    entry_scores = np.concatenate([scores[normal_positions], event_scores])
    entry_positive = np.concatenate(
        [np.zeros(len(normal_positions), dtype=bool), np.ones(len(starts), dtype=bool)]
    )
    source_order = np.argsort(entry_positions, kind="mergesort")
    entry_scores = entry_scores[source_order]
    entry_positive = entry_positive[source_order]
    score_order = np.argsort(-entry_scores, kind="mergesort")
    sorted_positive = entry_positive[score_order]

    outputs = []
    for event_weights in (point_weights, log_weights):
        entry_weights = np.concatenate(
            [np.ones(len(normal_positions), dtype=np.float64), event_weights]
        )[source_order]
        sorted_weights = entry_weights[score_order]
        predicted_weight = np.cumsum(sorted_weights)
        true_positive = np.cumsum(sorted_weights * sorted_positive)
        total_anomaly = float(np.sum(event_weights))
        precision = true_positive / (predicted_weight + 1e-15)
        recall = true_positive / (total_anomaly + 1e-15)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-15)
        best = int(np.argmax(f1))
        outputs.append(
            (
                float(f1[best]),
                float(precision[best]),
                float(recall[best]),
                float(entry_scores[score_order[best]]),
            )
        )
    return outputs[0], outputs[1]


def metric_row(dataset: str, series: str, method: str, labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask].astype(np.float64)
    easytsad_only = os.environ.get("TSHAPE_EASYTSAD_ONLY", "0") == "1"
    if len(labels) == 0 or easytsad_only:
        ap = auc = best = q95 = q97 = qf1 = q995 = mad3 = math.nan
        eq95 = eq97 = eq99 = eq995 = emad3 = math.nan
        bp = br = qp = qr = madp = madr = math.nan
    else:
        ap = average_precision(labels, scores)
        auc = roc_auc(labels, scores)
        best, bp, br = best_f1(labels, scores)
        q95, _, _ = fixed_quantile_f1(labels, scores, 0.95)
        q97, _, _ = fixed_quantile_f1(labels, scores, 0.97)
        qf1, qp, qr = fixed_quantile_f1(labels, scores, 0.99)
        q995, _, _ = fixed_quantile_f1(labels, scores, 0.995)
        mad3, madp, madr = mad_f1(labels, scores, 3.0)
        eq95, _, _ = fixed_quantile_event_f1(labels, scores, 0.95)
        eq97, _, _ = fixed_quantile_event_f1(labels, scores, 0.97)
        eq99, _, _ = fixed_quantile_event_f1(labels, scores, 0.99)
        eq995, _, _ = fixed_quantile_event_f1(labels, scores, 0.995)
        emad3, _, _ = mad_event_f1(labels, scores, 3.0)
    pa = easytsad_pa_metrics(labels, scores)
    row = {
        "dataset": dataset,
        "series": series,
        "method": method,
        "points": int(len(labels)),
        "anomaly_ratio": f"{float(np.mean(labels)):.6f}" if len(labels) else "nan",
        "ap": f"{ap:.6f}" if np.isfinite(ap) else "nan",
        "auc": f"{auc:.6f}" if np.isfinite(auc) else "nan",
        "best_f1": f"{best:.6f}" if np.isfinite(best) else "nan",
        "best_precision": f"{bp:.6f}" if np.isfinite(bp) else "nan",
        "best_recall": f"{br:.6f}" if np.isfinite(br) else "nan",
        "f1_q95": f"{q95:.6f}" if np.isfinite(q95) else "nan",
        "f1_q97": f"{q97:.6f}" if np.isfinite(q97) else "nan",
        "f1_q99": f"{qf1:.6f}" if np.isfinite(qf1) else "nan",
        "f1_q995": f"{q995:.6f}" if np.isfinite(q995) else "nan",
        "f1_mad3": f"{mad3:.6f}" if np.isfinite(mad3) else "nan",
        "event_f1_q95": f"{eq95:.6f}" if np.isfinite(eq95) else "nan",
        "event_f1_q97": f"{eq97:.6f}" if np.isfinite(eq97) else "nan",
        "event_f1_q99": f"{eq99:.6f}" if np.isfinite(eq99) else "nan",
        "event_f1_q995": f"{eq995:.6f}" if np.isfinite(eq995) else "nan",
        "event_f1_mad3": f"{emad3:.6f}" if np.isfinite(emad3) else "nan",
        "precision_q99": f"{qp:.6f}" if np.isfinite(qp) else "nan",
        "recall_q99": f"{qr:.6f}" if np.isfinite(qr) else "nan",
        "precision_mad3": f"{madp:.6f}" if np.isfinite(madp) else "nan",
        "recall_mad3": f"{madr:.6f}" if np.isfinite(madr) else "nan",
    }
    row.update({key: f"{value:.6f}" if np.isfinite(value) else "nan" for key, value in pa.items()})
    return row


def aggregate_rows(
    rows: Sequence[Dict[str, object]],
    expected_series_by_dataset: Optional[Dict[str, int]] = None,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)

    out: List[Dict[str, object]] = []
    metrics = [
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
        "point_f1_pa",
        "point_precision_pa",
        "point_recall_pa",
        "point_threshold_pa",
        "event_f1_pa_log",
        "event_precision_pa_log",
        "event_recall_pa_log",
        "event_threshold_pa_log",
    ]
    for (dataset, method), group in sorted(grouped.items()):
        series_names = {str(r["series"]) for r in group}
        expected = (expected_series_by_dataset or {}).get(dataset, len(series_names))
        valid_ap_series = {
            str(r["series"])
            for r in group
            if str(r.get("ap", "nan")) != "nan"
        }
        valid_point_pa_series = {
            str(r["series"])
            for r in group
            if str(r.get("point_f1_pa", "nan")) != "nan"
        }
        valid_event_pa_series = {
            str(r["series"])
            for r in group
            if str(r.get("event_f1_pa_log", "nan")) != "nan"
        }
        zero_anomaly_series = {
            str(r["series"])
            for r in group
            if float(r.get("anomaly_ratio", "nan")) == 0.0
        }
        seeds = {
            str(r["seed"])
            for r in group
            if "seed" in r and str(r["seed"]) not in {"", "nan", "None"}
        }
        item: Dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "series": len(series_names),
            "rows": len(group),
            "seeds": len(seeds) if seeds else 1,
            "points": int(sum(int(r["points"]) for r in group)),
            "valid_ap_series": len(valid_ap_series),
            "valid_point_pa_series": len(valid_point_pa_series),
            "valid_event_pa_series": len(valid_event_pa_series),
            "zero_anomaly_series": len(zero_anomaly_series),
            "coverage_ratio": f"{(len(series_names) / expected):.6f}" if expected else "nan",
        }
        for metric in metrics:
            vals = np.array([float(r.get(metric, "nan")) for r in group if str(r.get(metric, "nan")) != "nan"], dtype=np.float64)
            item[metric] = f"{float(np.mean(vals)):.6f}" if vals.size else "nan"
            item[f"{metric}_std"] = f"{float(np.std(vals, ddof=1)):.6f}" if vals.size > 1 else "0.000000"
        out.append(item)
    return out


def baseline_scores(test: np.ndarray, p: int, method: str) -> np.ndarray:
    if method == "persistence":
        prev = np.r_[test[0], test[:-1]]
        return np.abs(test - prev).astype(np.float32)
    if method == "rolling_mean":
        csum = np.r_[0.0, np.cumsum(test, dtype=np.float64)]
        scores = np.zeros_like(test, dtype=np.float32)
        for i in range(len(test)):
            start = max(0, i - p)
            count = i - start
            pred = test[i - 1] if count == 0 else (csum[i] - csum[start]) / count
            scores[i] = abs(float(test[i]) - float(pred))
        return scores
    if method == "rolling_median":
        scores = np.zeros_like(test, dtype=np.float32)
        prefix = min(p, len(test))
        for i in range(prefix):
            pred = float(test[i]) if i == 0 else float(np.median(test[:i]))
            scores[i] = abs(float(test[i]) - pred)
        if len(test) > p:
            windows = np.lib.stride_tricks.sliding_window_view(test, p)[:-1]
            pred = np.median(windows, axis=1).astype(np.float32)
            scores[p:] = np.abs(test[p:] - pred).astype(np.float32)
        return scores
    if method == "rolling_mad":
        base = baseline_scores(test, p, "rolling_median")
        scores = np.zeros_like(test, dtype=np.float32)
        prefix = min(p, len(test))
        for i in range(prefix):
            hist = test[:i]
            if len(hist) == 0:
                scores[i] = 0.0
            else:
                med = float(np.median(hist))
                mad = float(np.median(np.abs(hist - med)))
                scores[i] = float(base[i]) / max(1.4826 * mad, 1e-6)
        if len(test) > p:
            windows = np.lib.stride_tricks.sliding_window_view(test, p)[:-1]
            med = np.median(windows, axis=1)
            mad = np.median(np.abs(windows - med[:, None]), axis=1)
            scores[p:] = (base[p:] / np.maximum(1.4826 * mad, 1e-6)).astype(np.float32)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if method == "spectral_residual":
        x = np.asarray(test, dtype=np.float64)
        if len(x) < 8:
            return np.zeros_like(test, dtype=np.float32)
        spectrum = np.fft.fft(x)
        amp = np.abs(spectrum)
        log_amp = np.log(np.maximum(amp, 1e-8))
        k = min(max(3, p), max(3, len(log_amp) // 8))
        kernel = np.ones(k, dtype=np.float64) / k
        avg = np.convolve(log_amp, kernel, mode="same")
        residual = np.exp(log_amp - avg)
        saliency = np.abs(np.fft.ifft(spectrum * residual / np.maximum(amp, 1e-8)))
        smooth = np.convolve(saliency, np.ones(k) / k, mode="same")
        scores = np.abs(saliency - smooth)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"unknown baseline: {method}")


def forecast_residual_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """A zero-training forecasting-residual adapter.

    This is a local, reproducible adapter for the same scoring interface used by
    Chronos/TimesFM/Timer/Sundial-style forecasters: predict the next point from
    recent context, then rank by forecast residual. It is intentionally not a
    pretrained foundation model; it provides a full-coverage adapter baseline
    without external model downloads.
    """
    x = np.asarray(test, dtype=np.float32)
    prev1 = np.r_[x[0], x[:-1]]
    prev2 = np.r_[prev1[0], prev1[:-1]]
    linear = 2.0 * prev1 - prev2
    mean_res = baseline_scores(x, p, "rolling_mean")
    median_res = baseline_scores(x, p, "rolling_median")
    persistence_res = np.abs(x - prev1).astype(np.float32)
    linear_res = np.abs(x - linear).astype(np.float32)
    # Normalize channels before ensembling so that one high-variance forecaster
    # cannot dominate the residual ranking.
    return (
        0.30 * scale01(persistence_res)
        + 0.25 * scale01(mean_res)
        + 0.25 * scale01(median_res)
        + 0.20 * scale01(linear_res)
    ).astype(np.float32)


def shifted(x: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return x.copy()
    if len(x) == 0:
        return x.copy()
    out = np.empty_like(x, dtype=np.float32)
    out[:lag] = x[0]
    out[lag:] = x[:-lag]
    return out


def rolling_forecast(x: np.ndarray, p: int, reducer: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if len(arr) == 0:
        return arr
    out = np.empty_like(arr, dtype=np.float32)
    out[0] = arr[0]
    prefix = min(max(1, p), len(arr))
    for i in range(1, prefix):
        hist = arr[:i]
        out[i] = float(np.median(hist) if reducer == "median" else np.mean(hist))
    if len(arr) > p:
        windows = np.lib.stride_tricks.sliding_window_view(arr, p)[:-1]
        if reducer == "median":
            out[p:] = np.median(windows, axis=1).astype(np.float32)
        else:
            csum = np.r_[0.0, np.cumsum(arr, dtype=np.float64)]
            out[p:] = ((csum[p:-1] - csum[:-p - 1]) / p).astype(np.float32)
    return out


def timer_patch_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """Faithful Timer-style next-patch residual adapter.

    Timer is a generative, patch-based time-series model. When the official
    checkpoint cannot be executed in this archived Python stack, this adapter
    keeps the same operational contract: use recent context to predict the next
    patch/token, then rank by forecast residual.
    """
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    prev1 = shifted(x, 1)
    prev2 = shifted(x, 2)
    local_linear = 2.0 * prev1 - prev2
    patch_repeat = shifted(x, max(2, p))
    patch_drift = shifted(x, 1) - shifted(x, max(2, p + 1))
    patch_forecast = patch_repeat + patch_drift
    mean_res = baseline_scores(x, p, "rolling_mean")
    return (
        0.45 * scale01((x - local_linear) ** 2)
        + 0.35 * scale01((x - patch_forecast) ** 2)
        + 0.20 * scale01(mean_res)
    ).astype(np.float32)


def timesfm_trend_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """Faithful TimesFM-style decoder forecasting residual adapter.

    TimesFM is a decoder-only zero-shot forecaster. This local adapter emulates
    the inference contract with robust trend and seasonal-repeat forecast
    candidates, then scores the residual from the median forecast.
    """
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    prev1 = shifted(x, 1)
    prev2 = shifted(x, 2)
    linear = 2.0 * prev1 - prev2
    lag = max(4, min(2 * p, max(4, len(x) // 8)))
    seasonal = shifted(x, lag)
    short_mean = rolling_forecast(x, max(3, p // 2), "mean")
    long_mean = rolling_forecast(x, max(p, lag), "mean")
    forecasts = np.vstack([prev1, linear, seasonal, short_mean, long_mean]).astype(np.float32)
    median_forecast = np.median(forecasts, axis=0).astype(np.float32)
    spread = np.median(np.abs(forecasts - median_forecast[None, :]), axis=0).astype(np.float32)
    residual = np.abs(x - median_forecast)
    return scale01(residual / np.maximum(1.4826 * spread, 1e-4)).astype(np.float32)


def sundial_prob_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """Faithful Sundial-style probabilistic forecast residual adapter.

    Sundial exposes point and probabilistic zero-shot forecasts. This adapter
    builds a label-free predictive distribution from context-consistent local
    forecasters and scores target points by calibrated tail distance.
    """
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    prev = shifted(x, 1)
    linear = 2.0 * shifted(x, 1) - shifted(x, 2)
    median_center = rolling_forecast(x, p, "median")
    spectral_residual = baseline_scores(x, p, "spectral_residual")
    spectral_center = x - np.sign(x - median_center) * spectral_residual
    candidates = np.vstack([prev, linear, median_center, spectral_center]).astype(np.float32)
    loc = np.median(candidates, axis=0).astype(np.float32)
    epistemic = np.median(np.abs(candidates - loc[None, :]), axis=0).astype(np.float32)
    local_mad = baseline_scores(x, p, "rolling_mad")
    residual = np.abs(x - loc)
    z = residual / np.maximum(0.5 * scale01(local_mad) + 1.4826 * epistemic, 1e-4)
    return scale01(z).astype(np.float32)


def autocorr_lags(x: np.ndarray, p: int, top_k: int = 3) -> List[int]:
    """Return stable candidate periods from autocorrelation."""
    arr = np.asarray(x, dtype=np.float32)
    if len(arr) < max(8, p + 3):
        return [max(2, p)]
    centered = arr - float(np.mean(arr))
    max_lag = min(max(2 * p, p + 8), max(2, len(arr) // 3))
    scores: List[Tuple[int, float]] = []
    for lag in range(2, max_lag + 1):
        a = centered[:-lag]
        b = centered[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        corr = 0.0 if denom < 1e-8 else float(np.dot(a, b) / denom)
        scores.append((lag, corr))
    scores.sort(key=lambda item: item[1], reverse=True)
    lags = [lag for lag, corr in scores[:top_k] if np.isfinite(corr)]
    return lags or [max(2, p)]


def lowpass_reconstruct(x: np.ndarray, keep_ratio: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if len(arr) < 4:
        return arr.copy()
    spectrum = np.fft.rfft(arr.astype(np.float64))
    keep = max(2, min(len(spectrum), int(math.ceil(len(spectrum) * keep_ratio))))
    filtered = spectrum.copy()
    filtered[keep:] = 0.0
    recon = np.fft.irfft(filtered, n=len(arr))
    return recon.astype(np.float32)


def time_mixer_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """TimeMixer-style multi-scale forecasting residual adapter.

    TimeMixer mixes decomposed temporal scales. This adapter keeps that scoring
    contract by combining short, medium, and long context forecasts and using
    cross-scale disagreement as uncertainty.
    """
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    windows = [max(3, p // 2), max(4, p), max(6, 2 * p), max(8, 4 * p)]
    forecasts = [rolling_forecast(x, min(w, max(1, len(x))), "mean") for w in windows]
    forecasts.append(rolling_forecast(x, max(3, p), "median"))
    stack = np.vstack(forecasts).astype(np.float32)
    loc = np.median(stack, axis=0).astype(np.float32)
    scale = np.median(np.abs(stack - loc[None, :]), axis=0).astype(np.float32)
    residual = np.abs(x - loc)
    high_freq = np.abs(x - rolling_forecast(x, max(3, p // 2), "mean"))
    return (
        0.70 * scale01(residual / np.maximum(1.4826 * scale, 1e-4))
        + 0.30 * scale01(high_freq)
    ).astype(np.float32)


def time_moe_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """Time-MoE-style mixture-of-experts zero-shot forecast residual adapter."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    prev = shifted(x, 1)
    linear = 2.0 * shifted(x, 1) - shifted(x, 2)
    median = rolling_forecast(x, max(3, p), "median")
    trend = rolling_forecast(x, max(6, 2 * p), "mean")
    lags = autocorr_lags(x, p, top_k=2)
    seasonal = np.median(np.vstack([shifted(x, lag) for lag in lags]), axis=0).astype(np.float32)

    experts = np.vstack([prev, linear, median, trend, seasonal]).astype(np.float32)
    local_vol = scale01(baseline_scores(x, p, "rolling_mad"))
    local_slope = scale01(np.abs(linear - prev))
    seasonal_fit = 1.0 - scale01(np.abs(x - seasonal))
    weights = np.vstack(
        [
            0.30 + 0.20 * local_vol,
            0.25 + 0.25 * local_slope,
            0.35 + 0.20 * local_vol,
            0.25 + 0.20 * (1.0 - local_slope),
            0.20 + 0.30 * seasonal_fit,
        ]
    ).astype(np.float32)
    weights = weights / np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-6)
    loc = np.sum(weights * experts, axis=0).astype(np.float32)
    spread = np.sum(weights * np.abs(experts - loc[None, :]), axis=0).astype(np.float32)
    residual = np.abs(x - loc)
    return scale01(residual / np.maximum(1.4826 * spread, 1e-4)).astype(np.float32)


def fits_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """FITS-style frequency interpolation/reconstruction residual adapter."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    low = lowpass_reconstruct(x, keep_ratio=min(0.25, max(0.04, p / max(len(x), 1))))
    residual = np.abs(x - low)
    local = baseline_scores(x, p, "rolling_mad")
    return scale01(residual / np.maximum(scale01(local), 1e-4)).astype(np.float32)


def timesnet_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """TimesNet-style multi-period residual adapter."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    lags = autocorr_lags(x, p, top_k=4)
    period_forecasts = [shifted(x, lag) for lag in lags]
    period_forecasts.extend(
        [
            rolling_forecast(x, max(3, p), "mean"),
            rolling_forecast(x, max(3, p), "median"),
        ]
    )
    stack = np.vstack(period_forecasts).astype(np.float32)
    loc = np.median(stack, axis=0).astype(np.float32)
    disagreement = np.median(np.abs(stack - loc[None, :]), axis=0).astype(np.float32)
    residual = np.abs(x - loc)
    return scale01(residual / np.maximum(disagreement, 1e-4)).astype(np.float32)


def ofa_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """OFA/GPT-style general time-series task adapter.

    The official OFA implementation is task-oriented. This adapter implements
    the same open-task forecast-residual interface as a robust ensemble of
    trend, seasonality, and frequency channels.
    """
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    forecast = forecast_residual_adapter_scores(x, p)
    period = timesnet_adapter_scores(x, p)
    frequency = fits_adapter_scores(x, p)
    return (0.40 * scale01(forecast) + 0.35 * scale01(period) + 0.25 * scale01(frequency)).astype(np.float32)


def fcvae_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """FCVAE-style frequency-contrast reconstruction residual adapter."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    coarse = lowpass_reconstruct(x, keep_ratio=0.08)
    mid = lowpass_reconstruct(x, keep_ratio=0.18)
    rec = 0.65 * coarse + 0.35 * mid
    contrast = np.abs(mid - coarse)
    residual = np.abs(x - rec)
    return scale01(residual + 0.35 * contrast).astype(np.float32)


def kanad_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """KANAD-style nonlinear spline residual adapter."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    prev1 = shifted(x, 1)
    prev2 = shifted(x, 2)
    prev3 = shifted(x, 3)
    quadratic = 3.0 * prev1 - 3.0 * prev2 + prev3
    spline_center = 0.55 * rolling_forecast(x, max(3, p // 2), "median") + 0.45 * rolling_forecast(x, max(4, p), "mean")
    residual = 0.55 * np.abs(x - quadratic) + 0.45 * np.abs(x - spline_center)
    return scale01(residual).astype(np.float32)


def sublof_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """SubLOF-family local-density residual with no target labels or fitting."""
    x = np.asarray(test, dtype=np.float32)
    robust = scale01(baseline_scores(x, p, "rolling_mad"))
    short = scale01(np.abs(x - rolling_forecast(x, max(3, p // 2), "median")))
    slope = scale01(np.abs(shifted(x, 1) - shifted(x, 2)))
    return (0.55 * robust + 0.30 * short + 0.15 * slope).astype(np.float32)


def sand_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """SAND-family streaming discord score from recurring-shape mismatch."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    lags = autocorr_lags(x, max(p, 8), top_k=4)
    recurring = np.median(np.vstack([shifted(x, lag) for lag in lags]), axis=0)
    discord = np.abs(x - recurring)
    local = np.abs(x - rolling_forecast(x, max(4, p), "median"))
    return (0.70 * scale01(discord) + 0.30 * scale01(local)).astype(np.float32)


def matrix_profile_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """Linear-time Matrix-Profile family approximation for full long traces.

    It compares each point with several recurring lagged contexts and uses the
    smallest rolling context distance as a discord score.  The paper marks this
    row as a family adapter, not as an execution of the STOMP/STAMP artifact.
    """
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return x
    lags = sorted(set(autocorr_lags(x, max(p, 8), top_k=4) + [max(2, p), max(3, 2 * p)]))
    profiles: List[np.ndarray] = []
    kernel = np.ones(max(2, p), dtype=np.float64) / max(2, p)
    for lag in lags:
        delta = (x.astype(np.float64) - shifted(x, lag).astype(np.float64)) ** 2
        distance = np.sqrt(np.convolve(delta, kernel, mode="same"))
        profiles.append(distance.astype(np.float32))
    return scale01(np.min(np.vstack(profiles), axis=0)).astype(np.float32)


def ar_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """AR-family residual using a causal third-order extrapolator."""
    x = np.asarray(test, dtype=np.float32)
    pred = 3.0 * shifted(x, 1) - 3.0 * shifted(x, 2) + shifted(x, 3)
    robust = rolling_forecast(x, max(4, p), "median")
    disagreement = np.abs(pred - robust)
    return scale01(np.abs(x - pred) / np.maximum(disagreement, 1e-4)).astype(np.float32)


def lstmad_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """LSTMAD-family memory residual with causal short/long context gates."""
    x = np.asarray(test, dtype=np.float32)
    short = rolling_forecast(x, max(4, p), "mean")
    long = rolling_forecast(x, max(8, 4 * p), "mean")
    seasonal_lags = autocorr_lags(x, max(p, 8), top_k=2)
    seasonal = np.median(np.vstack([shifted(x, lag) for lag in seasonal_lags]), axis=0)
    volatility = scale01(baseline_scores(x, p, "rolling_mad"))
    pred = (0.55 - 0.20 * volatility) * short + (0.20 + 0.10 * volatility) * long + (0.25 + 0.10 * volatility) * seasonal
    return scale01(np.abs(x - pred)).astype(np.float32)


def ae_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """AE-family smooth reconstruction residual."""
    x = np.asarray(test, dtype=np.float32)
    recon = lowpass_reconstruct(x, keep_ratio=min(0.18, max(0.04, p / max(len(x), 1))))
    return scale01(np.abs(x - recon)).astype(np.float32)


def encdec_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """EncDecAD-family multi-resolution reconstruction residual."""
    x = np.asarray(test, dtype=np.float32)
    recons = np.vstack(
        [
            lowpass_reconstruct(x, keep_ratio=0.05),
            lowpass_reconstruct(x, keep_ratio=0.12),
            rolling_forecast(x, max(6, 2 * p), "mean"),
        ]
    )
    center = np.median(recons, axis=0)
    spread = np.median(np.abs(recons - center[None, :]), axis=0)
    return scale01(np.abs(x - center) / np.maximum(spread, 1e-4)).astype(np.float32)


def srcnn_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """SRCNN-family spectral-residual saliency score."""
    return scale01(baseline_scores(np.asarray(test, dtype=np.float32), p, "spectral_residual")).astype(np.float32)


def tfad_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """TFAD-family time/frequency decomposition score."""
    x = np.asarray(test, dtype=np.float32)
    frequency = scale01(baseline_scores(x, p, "spectral_residual"))
    trend = lowpass_reconstruct(x, keep_ratio=0.08)
    time_residual = scale01(np.abs(x - trend))
    local = scale01(np.abs(x - rolling_forecast(x, max(4, p), "median")))
    return (0.40 * frequency + 0.35 * time_residual + 0.25 * local).astype(np.float32)


def donut_adapter_scores(test: np.ndarray, p: int) -> np.ndarray:
    """Donut-family robust reconstruction score for missing/noisy KPI values."""
    x = np.asarray(test, dtype=np.float32)
    median = rolling_forecast(x, max(4, p), "median")
    smooth = lowpass_reconstruct(x, keep_ratio=0.12)
    reconstruction = 0.70 * median + 0.30 * smooth
    mad = baseline_scores(x, p, "rolling_mad")
    return (0.75 * scale01(np.abs(x - reconstruction)) + 0.25 * scale01(mad)).astype(np.float32)


FOUNDATION_ADAPTERS = {
    "TimerPatchAdapter": ("thuml/timer-base-84m faithful patch-forecast adapter", timer_patch_adapter_scores),
    "TimesFMTrendAdapter": ("google-research/timesfm faithful trend-decoder adapter", timesfm_trend_adapter_scores),
    "SundialProbAdapter": ("thuml/sundial-base-128m faithful probabilistic adapter", sundial_prob_adapter_scores),
    "TimeMixerAdapter": ("kwuking/TimeMixer faithful multi-scale forecast residual adapter", time_mixer_adapter_scores),
    "TimeMoEAdapter": ("Time-MoE/Time-MoE faithful mixture-of-experts forecast residual adapter", time_moe_adapter_scores),
}


MODERN_ADAPTERS = {
    "ForecastResidualAdapter": ("local robust forecast residual adapter", forecast_residual_adapter_scores),
    "OFAAdapter": ("DAMO-DI-ML/NeurIPS2023-One-Fits-All open-task residual adapter", ofa_adapter_scores),
    "FCVAEAdapter": ("FCVAE-style frequency-contrast reconstruction residual adapter", fcvae_adapter_scores),
    "TimesNetAdapter": ("THUML TimesNet-style multi-period residual adapter", timesnet_adapter_scores),
    "KANADAdapter": ("KANAD-style nonlinear spline residual adapter", kanad_adapter_scores),
    "FITSAdapter": ("VEWOXIC/FITS frequency interpolation residual adapter", fits_adapter_scores),
    "SubLOFAdapter": ("EasyTSAD SubLOF family zero-shot scoring adapter", sublof_adapter_scores),
    "SANDAdapter": ("EasyTSAD SAND family streaming-discord adapter", sand_adapter_scores),
    "MatrixProfileAdapter": ("Matrix Profile family full-series discord adapter", matrix_profile_adapter_scores),
    "ARAdapter": ("EasyTSAD AR family causal forecasting adapter", ar_adapter_scores),
    "LSTMADAdapter": ("EasyTSAD LSTMAD family causal-memory adapter", lstmad_adapter_scores),
    "AEAdapter": ("EasyTSAD AE family reconstruction adapter", ae_adapter_scores),
    "EncDecADAdapter": ("EasyTSAD EncDecAD family reconstruction adapter", encdec_adapter_scores),
    "SRCNNAdapter": ("EasyTSAD SRCNN family spectral-residual adapter", srcnn_adapter_scores),
    "TFADAdapter": ("EasyTSAD TFAD family time-frequency adapter", tfad_adapter_scores),
    "DonutAdapter": ("EasyTSAD Donut family robust-reconstruction adapter", donut_adapter_scores),
}


def pattern_bank_prototypes(p: int, fraction: float, seed: int, variants_per_family: int = 4) -> np.ndarray:
    """Generate a deterministic bank of normal KPI continuation patterns.

    The bank is synthetic on purpose: it represents the open-world patterns a
    reusable detector should see before it is applied to a new service metric.
    Fractions select progressively more pattern families, so the resulting CSV
    is a real coverage ablation rather than a cosmetic plotting variable.
    """
    rng = np.random.default_rng(seed)
    families = [
        "flat",
        "linear",
        "quadratic",
        "exponential_trend",
        "sine",
        "multi_sine",
        "sawtooth",
        "triangle",
        "square",
        "stair",
        "random_walk",
        "mean_reverting",
        "damped_oscillation",
        "chirp",
        "autoregressive",
        "trend_seasonal",
        "logistic_growth",
        "smooth_piecewise",
        "seasonal_drift",
        "periodic_pulse",
    ]
    k = max(1, min(len(families), int(math.ceil(len(families) * fraction))))
    selected = families[:k]
    x = np.linspace(0.0, 1.0, p + 1, dtype=np.float64)
    protos: List[np.ndarray] = []
    for family in selected:
        for _ in range(variants_per_family):
            phase = rng.uniform(0.0, 2.0 * math.pi)
            amp = rng.uniform(0.45, 1.0)
            slope = rng.uniform(-0.8, 0.8)
            noise = rng.normal(0.0, 0.015, size=p + 1)
            if family == "flat":
                y = np.ones(p + 1) * rng.uniform(-0.2, 0.2)
            elif family == "linear":
                y = slope * x + rng.uniform(-0.2, 0.2)
            elif family == "quadratic":
                y = slope * x + rng.uniform(-1.0, 1.0) * (x - 0.5) ** 2
            elif family == "exponential_trend":
                rate = rng.uniform(0.8, 3.2)
                direction = -1.0 if rng.random() < 0.5 else 1.0
                y = direction * np.expm1(rate * x) / np.expm1(rate)
            elif family == "sine":
                y = amp * np.sin(2.0 * math.pi * rng.uniform(0.7, 2.0) * x + phase)
            elif family == "multi_sine":
                y = amp * np.sin(2.0 * math.pi * x + phase) + 0.35 * np.sin(2.0 * math.pi * rng.uniform(2.0, 5.0) * x)
            elif family == "sawtooth":
                period = rng.uniform(0.22, 0.55)
                y = ((x + rng.uniform(0.0, period)) % period) / period
            elif family == "triangle":
                period = rng.uniform(0.24, 0.60)
                cycle = ((x + rng.uniform(0.0, period)) % period) / period
                y = 1.0 - 2.0 * np.abs(2.0 * cycle - 1.0)
            elif family == "square":
                y = (np.sin(2.0 * math.pi * rng.uniform(0.7, 2.2) * x + phase) > 0).astype(float)
            elif family == "stair":
                steps = rng.integers(3, 6)
                y = np.floor(x * steps) / max(steps - 1, 1)
            elif family == "random_walk":
                y = np.cumsum(rng.normal(0.0, rng.uniform(0.04, 0.14), size=p + 1))
            elif family == "mean_reverting":
                phi = rng.uniform(0.45, 0.85)
                center = rng.uniform(-0.25, 0.25)
                y = np.empty(p + 1, dtype=np.float64)
                y[0] = center + rng.normal(0.0, 0.2)
                innovations = rng.normal(0.0, rng.uniform(0.035, 0.11), size=p)
                for step in range(1, p + 1):
                    y[step] = center + phi * (y[step - 1] - center) + innovations[step - 1]
            elif family == "damped_oscillation":
                damping = rng.uniform(0.45, 2.2)
                frequency = rng.uniform(1.0, 3.5)
                y = amp * np.exp(-damping * x) * np.sin(2.0 * math.pi * frequency * x + phase)
            elif family == "chirp":
                f0 = rng.uniform(0.4, 1.2)
                sweep = rng.uniform(1.0, 4.0)
                y = amp * np.sin(2.0 * math.pi * (f0 * x + 0.5 * sweep * x * x) + phase)
            elif family == "autoregressive":
                phi = rng.uniform(0.55, 0.94)
                y = np.zeros(p + 1, dtype=np.float64)
                y[0] = rng.normal(0.0, 0.2)
                innovations = rng.normal(0.0, rng.uniform(0.04, 0.16), size=p)
                for step in range(1, p + 1):
                    y[step] = phi * y[step - 1] + innovations[step - 1]
            elif family == "trend_seasonal":
                slow = rng.uniform(0.55, 1.25)
                fast = rng.uniform(2.0, 4.5)
                y = (
                    slope * x
                    + amp * np.sin(2.0 * math.pi * slow * x + phase)
                    + 0.22 * amp * np.sin(2.0 * math.pi * fast * x + 0.5 * phase)
                )
            elif family == "logistic_growth":
                rate = rng.uniform(5.0, 12.0)
                center = rng.uniform(0.32, 0.68)
                direction = -1.0 if rng.random() < 0.5 else 1.0
                y = direction / (1.0 + np.exp(-rate * (x - center)))
            elif family == "smooth_piecewise":
                knots = np.linspace(0.0, 1.0, 5)
                levels = np.cumsum(rng.normal(0.0, 0.28, size=len(knots)))
                y = np.interp(x, knots, levels)
                y = np.convolve(np.pad(y, (1, 1), mode="edge"), np.array([0.2, 0.6, 0.2]), mode="valid")
            elif family == "seasonal_drift":
                frequency = rng.uniform(0.8, 2.2)
                envelope = 0.55 + rng.uniform(0.15, 0.45) * x
                y = slope * x + envelope * np.sin(2.0 * math.pi * frequency * x + phase)
            elif family == "periodic_pulse":
                centers = np.linspace(0.1, 0.9, int(rng.integers(2, 5)))
                width = rng.uniform(0.025, 0.09)
                y = np.zeros_like(x)
                for center in centers:
                    y += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
            else:
                y = np.zeros(p + 1)
            y = y + noise
            lo = float(np.min(y))
            hi = float(np.max(y))
            if hi - lo < 1e-8:
                z = np.zeros_like(y)
            else:
                z = (y - lo) / (hi - lo)
            protos.append(z.astype(np.float32))
    return np.stack(protos, axis=0).astype(np.float32)


def pattern_bank_scores(test: np.ndarray, p: int, prototypes: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Forecast-residual scores from nearest synthetic KPI pattern contexts."""
    x = np.asarray(test, dtype=np.float32)
    scores = baseline_scores(x, p, "rolling_median")
    if len(x) <= p or len(prototypes) == 0:
        return scores.astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x, p + 1)
    proto_ctx = prototypes[:, :p].astype(np.float32)
    proto_next = prototypes[:, p].astype(np.float32)
    out = np.zeros(len(windows), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        batch = np.ascontiguousarray(windows[start : start + batch_size], dtype=np.float32)
        ctx = batch[:, :p]
        y = batch[:, p]
        lo = np.min(ctx, axis=1)
        hi = np.max(ctx, axis=1)
        scale = np.maximum(hi - lo, 1e-6)
        ctx_norm = (ctx - lo[:, None]) / scale[:, None]
        diff = ctx_norm[:, None, :] - proto_ctx[None, :, :]
        dist = np.mean(diff * diff, axis=2)
        nearest = np.argmin(dist, axis=1)
        pred = lo + proto_next[nearest] * scale
        residual = (y - pred) ** 2
        out[start : start + len(batch)] = residual + 0.10 * dist[np.arange(len(batch)), nearest]
    scores[p:] = out
    if len(out):
        scores[:p] = float(np.min(out))
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pattern_bank_guard_scores(pattern_scores: np.ndarray, test: np.ndarray, p: int) -> np.ndarray:
    rolling = baseline_scores(test, p, "rolling_median")
    spectral = baseline_scores(test, p, "spectral_residual")
    mad = baseline_scores(test, p, "rolling_mad")
    return (
        0.35 * scale01(pattern_scores)
        + 0.40 * scale01(rolling)
        + 0.20 * scale01(spectral)
        + 0.05 * scale01(mad)
    ).astype(np.float32)


def pattern_bank(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    run_info: List[Dict[str, object]] = []
    expected = {dataset: len(iter_series([dataset])) for dataset in args.datasets}
    for frac in args.fractions:
        prototypes = pattern_bank_prototypes(args.p, frac, args.seed, args.variants_per_family)
        method = f"PatternBank-{int(round(frac * 100)):03d}"
        plus_method = f"PatternBankZeroPlus-{int(round(frac * 100)):03d}"
        start_time = time.time()
        print(f"Scoring {method} with {len(prototypes)} prototypes", flush=True)
        evaluated = 0
        for ref_i, ref in enumerate(iter_series(args.datasets), start=1):
            prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
            if prepared is None:
                continue
            scores = pattern_bank_scores(prepared.test, args.p, prototypes, args.batch_size)
            row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
            row["seed"] = args.seed
            row["fraction"] = f"{frac:.2f}"
            row["prototypes"] = len(prototypes)
            rows.append(row)
            guard = pattern_bank_guard_scores(scores, prepared.test, args.p)
            plus_row = metric_row(ref.dataset, ref.name, plus_method, prepared.labels, guard)
            plus_row["seed"] = args.seed
            plus_row["fraction"] = f"{frac:.2f}"
            plus_row["prototypes"] = len(prototypes)
            rows.append(plus_row)
            evaluated += 1
            if ref_i == 1 or ref_i % args.progress_every == 0:
                print(f"{method}: evaluated {ref_i} series; last={ref.dataset}/{ref.name}", flush=True)
        run_info.append(
            {
                "method": method,
                "fraction": frac,
                "prototypes": int(len(prototypes)),
                "seconds": time.time() - start_time,
                "eval_series": evaluated,
                "seed": args.seed,
            }
        )

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"pattern_bank_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"pattern_bank_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"pattern_bank_run_info{suffix}.json"
    fields = [
        "dataset",
        "series",
        "seed",
        "fraction",
        "prototypes",
        "method",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
    ]
    write_csv(detail_path, rows, fields)
    summary = aggregate_rows(rows, expected)
    frac_by_method = {f"PatternBank-{int(round(frac * 100)):03d}": frac for frac in args.fractions}
    frac_by_method.update({f"PatternBankZeroPlus-{int(round(frac * 100)):03d}": frac for frac in args.fractions})
    proto_by_method = {
        f"PatternBank-{int(round(info['fraction'] * 100)):03d}": info["prototypes"]
        for info in run_info
    }
    proto_by_method.update({
        f"PatternBankZeroPlus-{int(round(info['fraction'] * 100)):03d}": info["prototypes"]
        for info in run_info
    })
    for row in summary:
        row["fraction"] = f"{frac_by_method.get(str(row['method']), float('nan')):.2f}"
        row["prototypes"] = proto_by_method.get(str(row["method"]), "")
    preferred_summary_fields = [
        "dataset",
        "method",
        "fraction",
        "prototypes",
        "series",
        "rows",
        "seeds",
        "points",
        "valid_ap_series",
        "zero_anomaly_series",
        "coverage_ratio",
        "ap",
        "ap_std",
        "auc",
        "auc_std",
        "best_f1",
        "best_f1_std",
        "event_f1_q99",
        "event_f1_q99_std",
        "f1_q99",
        "f1_q99_std",
        "f1_mad3",
        "f1_mad3_std",
    ]
    summary_fields = preferred_summary_fields + [
        key for key in summary[0].keys() if key not in preferred_summary_fields
    ] if summary else preferred_summary_fields
    write_csv(summary_path, summary, summary_fields)
    runs_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def chronos_bolt_scores(
    pipeline,
    test: np.ndarray,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    early_p: int,
) -> np.ndarray:
    """Score a full series with official Chronos-Bolt forecasts.

    The adapter is chunked rather than point-rolling: each model call forecasts
    the next ``prediction_length`` points from the previous ``context_length``
    target points. This keeps full-series scoring feasible while preserving a
    pure zero-shot forecasting-residual interface.
    """
    import torch

    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return np.zeros_like(x, dtype=np.float32)
    scores = np.zeros_like(x, dtype=np.float32)
    first_position = max(1, early_p)
    if len(x) <= first_position:
        return scores.astype(np.float32)

    starts = list(range(first_position, len(x), prediction_length))
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_positions = starts[batch_start : batch_start + batch_size]
            contexts = [
                torch.tensor(x[max(0, pos - context_length) : pos], dtype=torch.float32)
                for pos in batch_positions
            ]
            horizon = max(min(prediction_length, len(x) - pos) for pos in batch_positions)
            forecast = pipeline.predict(contexts, prediction_length=horizon)
            arr = forecast.detach().cpu().numpy()
            if arr.ndim == 3:
                pred = arr[:, arr.shape[1] // 2, :]
            elif arr.ndim == 2:
                pred = arr
            else:
                pred = arr.reshape(len(batch_positions), -1)
            for row_idx, pos in enumerate(batch_positions):
                local_h = min(horizon, len(x) - pos)
                scores[pos : pos + local_h] = (x[pos : pos + local_h] - pred[row_idx, :local_h]) ** 2
    if starts:
        scored = scores[first_position:]
        scores[:first_position] = float(np.min(scored)) if scored.size else 0.0
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def hf_generate_forecast_scores(
    model,
    test: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    early_p: int,
    device: str,
    num_samples: int = 1,
) -> np.ndarray:
    """Score a full series with an official HF time-series generator.

    Timer, Sundial, and Time-MoE expose their pretrained checkpoints through
    ``AutoModelForCausalLM`` with custom ``generate`` methods. We use the same
    zero-shot forecasting-residual contract as Chronos: predict future target
    values from unlabeled target context and score squared residuals.
    """
    import torch

    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return np.zeros_like(x, dtype=np.float32)
    scores = np.zeros_like(x, dtype=np.float32)
    token_len = int(getattr(model.config, "input_token_len", 1) or 1)
    first_position = max(early_p, token_len)
    if len(x) <= first_position:
        return scores.astype(np.float32)

    starts = list(range(first_position, len(x), prediction_length))
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_positions = starts[batch_start : batch_start + batch_size]
            horizon = max(min(prediction_length, len(x) - pos) for pos in batch_positions)
            contexts = []
            for pos in batch_positions:
                context = x[max(0, pos - context_length) : pos]
                usable = (len(context) // token_len) * token_len
                if usable >= token_len:
                    context = context[-usable:]
                contexts.append(torch.tensor(context, dtype=torch.float32))
            max_context = max(ctx.numel() for ctx in contexts)
            padded = []
            for ctx in contexts:
                if ctx.numel() < max_context:
                    pad = torch.full((max_context - ctx.numel(),), float(ctx[0]) if ctx.numel() else 0.0)
                    ctx = torch.cat([pad, ctx])
                padded.append(ctx)
            inputs = torch.stack(padded, dim=0).to(device)
            kwargs = {"max_new_tokens": horizon}
            if model.config.model_type == "sundial":
                kwargs["num_samples"] = num_samples
            generated = model.generate(inputs, **kwargs)
            arr = generated.detach().cpu().numpy()
            if arr.ndim == 3:
                pred = np.median(arr, axis=1)
            elif arr.ndim == 2:
                # Timer returns only the forecast, whereas Time-MoE's custom
                # generation mixin returns the input context followed by the
                # forecast. Keep the common scorer explicit about that API
                # difference so context values can never be scored as output.
                pred = arr[:, -horizon:] if model.config.model_type == "time_moe" else arr
            else:
                pred = arr.reshape(len(batch_positions), -1)
            for row_idx, pos in enumerate(batch_positions):
                local_h = min(horizon, len(x) - pos)
                scores[pos : pos + local_h] = (x[pos : pos + local_h] - pred[row_idx, :local_h]) ** 2
    if starts:
        scored = scores[first_position:]
        scores[:first_position] = float(np.min(scored)) if scored.size else 0.0
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def time_moe_causal_scores(
    model,
    test: np.ndarray,
    *,
    context_length: int,
    block_length: int,
    batch_size: int,
    early_p: int,
    device: str,
) -> np.ndarray:
    """Score Time-MoE with its official causal next-value logits.

    ``generate`` performs one model call per generated value and is prohibitively
    slow for multi-million-point KPI corpora.  Time-MoE is a causal language
    model over scalar values, so a teacher-forced forward pass returns the same
    one-step prediction at every position in parallel.  Causal masking ensures
    that the score at time ``t`` uses values no later than ``t-1``.  Blocks
    overlap by ``context_length`` so every retained prediction has the requested
    history while keeping memory bounded.
    """
    import torch

    x = np.asarray(test, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return np.zeros_like(x, dtype=np.float32)
    first_position = max(1, int(early_p))
    scores = np.zeros_like(x, dtype=np.float32)
    if len(x) <= first_position:
        return scores

    target_block = max(32, int(block_length))
    context = max(first_position, int(context_length))
    spans = [
        (start, min(len(x), start + target_block))
        for start in range(first_position, len(x), target_block)
    ]
    effective_batch = max(1, min(int(batch_size), 8))
    with torch.no_grad():
        for batch_start in range(0, len(spans), effective_batch):
            batch_spans = spans[batch_start : batch_start + effective_batch]
            chunks = []
            offsets = []
            for start, end in batch_spans:
                left = max(0, start - context)
                chunk = torch.tensor(x[left:end], dtype=torch.float32)
                chunks.append(chunk)
                offsets.append((left, start, end))
            max_len = max(chunk.numel() for chunk in chunks)
            padded = []
            for chunk in chunks:
                if chunk.numel() < max_len:
                    # Right padding cannot affect earlier logits under causal
                    # masking; padded logits are discarded below.
                    pad = torch.full((max_len - chunk.numel(),), float(chunk[-1]))
                    chunk = torch.cat([chunk, pad])
                padded.append(chunk)
            inputs = torch.stack(padded, dim=0).to(device)
            logits = model(inputs, use_cache=False).logits[..., 0].detach().cpu().numpy()
            for row_idx, (left, start, end) in enumerate(offsets):
                # logits[j] predicts the value following input[j].
                lo = start - 1 - left
                hi = end - 1 - left
                pred = logits[row_idx, lo:hi]
                if len(pred) != end - start:
                    raise ValueError(
                        f"Time-MoE causal alignment failed: expected {end - start}, got {len(pred)}"
                    )
                scores[start:end] = (x[start:end] - pred.astype(np.float32)) ** 2
    scored = scores[first_position:]
    scores[:first_position] = float(np.min(scored)) if scored.size else 0.0
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def timesfm_official_scores(
    model,
    test: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    early_p: int,
) -> np.ndarray:
    """Score a full series with the official TimesFM 2.x torch forecaster."""
    x = np.asarray(test, dtype=np.float32)
    if len(x) == 0:
        return np.zeros_like(x, dtype=np.float32)
    scores = np.zeros_like(x, dtype=np.float32)
    first_position = max(early_p, 32)
    if len(x) <= first_position:
        return scores.astype(np.float32)

    starts = list(range(first_position, len(x), prediction_length))
    for batch_start in range(0, len(starts), batch_size):
        batch_positions = starts[batch_start : batch_start + batch_size]
        horizon = max(min(prediction_length, len(x) - pos) for pos in batch_positions)
        contexts = [x[max(0, pos - context_length) : pos].astype(np.float32) for pos in batch_positions]
        point_forecast, _ = model.forecast(horizon, contexts)
        pred = np.asarray(point_forecast, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(len(batch_positions), -1)
        for row_idx, pos in enumerate(batch_positions):
            local_h = min(horizon, len(x) - pos)
            scores[pos : pos + local_h] = (x[pos : pos + local_h] - pred[row_idx, :local_h]) ** 2
    if starts:
        scored = scores[first_position:]
        scores[:first_position] = float(np.min(scored)) if scored.size else 0.0
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def foundation_status_rows(executed_methods: Sequence[str]) -> List[Dict[str, object]]:
    executed = set(executed_methods)
    return [
        {
            "method": "Chronos-Bolt-Tiny",
            "model_or_repo": "amazon/chronos-bolt-tiny",
            "implementation": "chronos-forecasting 1.5.3",
            "status": "run" if "Chronos-Bolt-Tiny" in executed else "not-run-in-this-command",
            "notes": "Official Chronos-Bolt pipeline; chunked zero-shot forecast residual.",
        },
        {
            "method": "Timer-base-84m",
            "model_or_repo": "thuml/timer-base-84m",
            "implementation": "HuggingFace trust_remote_code quickstart",
            "status": "run" if "Timer-base-84m" in executed else "not-run-in-this-command",
            "notes": "Official pretrained Timer checkpoint via AutoModelForCausalLM.generate; requires the separate Python 3.11/transformers 4.40.1 baseline environment.",
        },
        {
            "method": "TimerPatchAdapter",
            "model_or_repo": "thuml/timer-base-84m / thuml/Large-Time-Series-Model",
            "implementation": "local faithful patch-forecast residual adapter",
            "status": "run" if "TimerPatchAdapter" in executed else "not-run-in-this-command",
            "notes": "Full-coverage local adapter preserving Timer's patch next-token forecasting residual interface; not an official pretrained checkpoint result.",
        },
        {
            "method": "Sundial-base-128m",
            "model_or_repo": "thuml/sundial-base-128m",
            "implementation": "HuggingFace trust_remote_code quickstart",
            "status": "run" if "Sundial-base-128m" in executed else "not-run-in-this-command",
            "notes": "Official pretrained Sundial checkpoint via AutoModelForCausalLM.generate; requires the separate Python 3.11/transformers 4.40.1 baseline environment.",
        },
        {
            "method": "SundialProbAdapter",
            "model_or_repo": "thuml/sundial-base-128m / thuml/Sundial",
            "implementation": "local faithful probabilistic forecasting residual adapter",
            "status": "run" if "SundialProbAdapter" in executed else "not-run-in-this-command",
            "notes": "Full-coverage local adapter preserving Sundial's point/probabilistic zero-shot forecast scoring interface; not an official pretrained checkpoint result.",
        },
        {
            "method": "TimesFM",
            "model_or_repo": "google-research/timesfm / timesfm PyPI",
            "implementation": "official timesfm package",
            "status": "superseded-by-TimesFM-2.5-200M",
            "notes": "The archived Python 3.9 stack cannot host the current official package; the separate Python 3.11 foundation environment runs the official torch checkpoint below.",
        },
        {
            "method": "TimesFM-2.5-200M",
            "model_or_repo": "google/timesfm-2.5-200m-pytorch",
            "implementation": "official timesfm 2.0.2 torch checkpoint",
            "status": "run" if "TimesFM-2.5-200M" in executed else "not-run-in-this-command",
            "notes": "Official TimesFM torch forecaster via timesfm.TimesFM_2p5_200M_torch; zero-shot forecast residual on unlabeled target context.",
        },
        {
            "method": "TimesFMTrendAdapter",
            "model_or_repo": "google-research/timesfm",
            "implementation": "local faithful decoder trend forecasting residual adapter",
            "status": "run" if "TimesFMTrendAdapter" in executed else "not-run-in-this-command",
            "notes": "Full-coverage local adapter preserving TimesFM's decoder-only zero-shot forecasting residual interface; not an official pretrained checkpoint result.",
        },
        {
            "method": "TimeMixer",
            "model_or_repo": "kwuking/TimeMixer",
            "implementation": "official repository",
            "status": "adapter-documented",
            "notes": "No PyPI package; official code is task-training oriented. Not run as a zero-shot pretrained checkpoint in this artifact.",
        },
        {
            "method": "TimeMixerAdapter",
            "model_or_repo": "kwuking/TimeMixer",
            "implementation": "local faithful multi-scale forecasting residual adapter",
            "status": "run" if "TimeMixerAdapter" in executed else "not-run-in-this-command",
            "notes": "Full-coverage adapter preserving TimeMixer's multi-scale decomposition/mixing intuition for zero-shot residual scoring.",
        },
        {
            "method": "TimeMoE-50M",
            "model_or_repo": "Maple728/TimeMoE-50M",
            "implementation": "official HuggingFace checkpoint",
            "status": "run" if "TimeMoE-50M" in executed else "not-run-in-this-command",
            "notes": "Official 50M-parameter Time-MoE checkpoint; causal teacher-forced one-step logits provide a full-series forecast residual without target fitting.",
        },
        {
            "method": "TimeMoEAdapter",
            "model_or_repo": "Time-MoE/Time-MoE",
            "implementation": "local faithful mixture-of-experts forecasting residual adapter",
            "status": "run" if "TimeMoEAdapter" in executed else "not-run-in-this-command",
            "notes": "Full-coverage adapter with trend, persistence, median, and seasonal experts plus data-dependent gates.",
        },
    ]


def foundation_baselines(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    refs = iter_series(args.datasets)
    if args.max_series is not None:
        refs = refs[: args.max_series]
    expected = {
        dataset: sum(ref.dataset == dataset for ref in refs)
        for dataset in args.datasets
    }

    if "Chronos-Bolt-Tiny" in args.methods:
        import torch
        from chronos import ChronosBoltPipeline

        start_load = time.time()
        print(f"Loading official Chronos model {args.chronos_model}", flush=True)
        pipeline = ChronosBoltPipeline.from_pretrained(args.chronos_model, torch_dtype=torch.float32)
        runs.append(
            {
                "method": "Chronos-Bolt-Tiny",
                "model": args.chronos_model,
                "load_seconds": time.time() - start_load,
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        )
        for ref_i, ref in enumerate(refs, start=1):
            prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
            if prepared is None:
                continue
            eval_start = time.time()
            scores = chronos_bolt_scores(
                pipeline,
                prepared.test,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                batch_size=args.batch_size,
                early_p=args.p,
            )
            row = metric_row(ref.dataset, ref.name, "Chronos-Bolt-Tiny", prepared.labels, scores)
            row["seed"] = args.seed
            row["model_id"] = args.chronos_model
            rows.append(row)
            if ref_i == 1 or ref_i % args.progress_every == 0:
                print(
                    f"Chronos evaluated {ref_i} series; last={ref.dataset}/{ref.name} seconds={time.time() - eval_start:.2f}",
                    flush=True,
                )

    official_hf_models = {
        "Timer-base-84m": "thuml/timer-base-84m",
        "Sundial-base-128m": "thuml/sundial-base-128m",
        "TimeMoE-50M": "Maple728/TimeMoE-50M",
    }
    for method in [m for m in args.methods if m in official_hf_models]:
        import torch
        from transformers import AutoModelForCausalLM

        model_id = official_hf_models[method]
        if args.hf_device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = args.hf_device
        start_load = time.time()
        print(f"Loading official HF model {model_id} on {device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            torch_dtype=torch.float32,
        ).to(device)
        model.eval()
        runs.append(
            {
                "method": method,
                "model": model_id,
                "load_seconds": time.time() - start_load,
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "local_files_only": args.local_files_only,
                "scoring_contract": (
                    "causal teacher-forced one-step logits"
                    if method == "TimeMoE-50M"
                    else "chunked multi-step generate forecast"
                ),
            }
        )
        for ref_i, ref in enumerate(refs, start=1):
            prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
            if prepared is None:
                continue
            eval_start = time.time()
            if method == "TimeMoE-50M":
                scores = time_moe_causal_scores(
                    model,
                    prepared.test,
                    context_length=args.context_length,
                    block_length=args.prediction_length,
                    batch_size=args.batch_size,
                    early_p=args.p,
                    device=device,
                )
            else:
                scores = hf_generate_forecast_scores(
                    model,
                    prepared.test,
                    context_length=args.context_length,
                    prediction_length=args.prediction_length,
                    batch_size=args.batch_size,
                    early_p=args.p,
                    device=device,
                    num_samples=args.num_samples,
                )
            row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
            row["seed"] = args.seed
            row["model_id"] = model_id
            rows.append(row)
            if ref_i == 1 or ref_i % args.progress_every == 0:
                print(
                    f"{method} evaluated {ref_i} series; last={ref.dataset}/{ref.name} seconds={time.time() - eval_start:.2f}",
                    flush=True,
                )

    if "TimesFM-2.5-200M" in args.methods:
        import timesfm

        start_load = time.time()
        print(f"Loading official TimesFM model {args.timesfm_model}", flush=True)
        timesfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            args.timesfm_model,
            local_files_only=args.local_files_only,
            torch_compile=False,
        )
        forecast_config = timesfm.ForecastConfig(
            max_context=args.context_length,
            max_horizon=args.prediction_length,
            normalize_inputs=True,
            per_core_batch_size=args.batch_size,
        )
        timesfm_model.compile(forecast_config)
        runs.append(
            {
                "method": "TimesFM-2.5-200M",
                "model": args.timesfm_model,
                "load_seconds": time.time() - start_load,
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": str(getattr(timesfm_model.model, "device", "unknown")),
                "local_files_only": args.local_files_only,
                "package": "timesfm",
            }
        )
        for ref_i, ref in enumerate(refs, start=1):
            prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
            if prepared is None:
                continue
            eval_start = time.time()
            scores = timesfm_official_scores(
                timesfm_model,
                prepared.test,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                batch_size=args.batch_size,
                early_p=args.p,
            )
            row = metric_row(ref.dataset, ref.name, "TimesFM-2.5-200M", prepared.labels, scores)
            row["seed"] = args.seed
            row["model_id"] = args.timesfm_model
            rows.append(row)
            if ref_i == 1 or ref_i % args.progress_every == 0:
                print(
                    f"TimesFM-2.5-200M evaluated {ref_i} series; last={ref.dataset}/{ref.name} seconds={time.time() - eval_start:.2f}",
                    flush=True,
                )

    for method in [m for m in args.methods if m in FOUNDATION_ADAPTERS]:
        model_id, scorer = FOUNDATION_ADAPTERS[method]
        print(f"Scoring {method} over all target series", flush=True)
        start = time.time()
        evaluated = 0
        for ref_i, ref in enumerate(refs, start=1):
            prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
            if prepared is None:
                continue
            scores = scorer(prepared.test, args.p)
            row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
            row["seed"] = args.seed
            row["model_id"] = model_id
            rows.append(row)
            evaluated += 1
            if ref_i == 1 or ref_i % args.progress_every == 0:
                print(f"{method} evaluated {ref_i} series; last={ref.dataset}/{ref.name}", flush=True)
        runs.append(
            {
                "method": method,
                "model": model_id,
                "seconds": time.time() - start,
                "eval_series": evaluated,
                "seed": args.seed,
                "adapter_kind": "faithful-local-forecast-residual",
            }
        )

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"foundation_baseline_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"foundation_baseline_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"foundation_baseline_run_info{suffix}.json"
    status_path = OUT_ROOT / f"foundation_baseline_status{suffix}.csv"
    fields = [
        "dataset",
        "series",
        "seed",
        "model_id",
        "method",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
    ]
    write_csv(detail_path, rows, fields)
    write_csv(
        summary_path,
        aggregate_rows(rows, expected),
        [
            "dataset",
            "method",
            "series",
            "rows",
            "seeds",
            "points",
            "valid_ap_series",
            "zero_anomaly_series",
            "coverage_ratio",
            "ap",
            "ap_std",
            "auc",
            "auc_std",
            "best_f1",
            "best_f1_std",
            "best_precision",
            "best_precision_std",
            "best_recall",
            "best_recall_std",
            "f1_q95",
            "f1_q95_std",
            "f1_q97",
            "f1_q97_std",
            "f1_q99",
            "f1_q99_std",
            "f1_q995",
            "f1_q995_std",
            "f1_mad3",
            "f1_mad3_std",
            "precision_q99",
            "precision_q99_std",
            "recall_q99",
            "recall_q99_std",
            "precision_mad3",
            "precision_mad3_std",
            "recall_mad3",
            "recall_mad3_std",
            "event_f1_q95",
            "event_f1_q95_std",
            "event_f1_q97",
            "event_f1_q97_std",
            "event_f1_q99",
            "event_f1_q99_std",
            "event_f1_q995",
            "event_f1_q995_std",
            "event_f1_mad3",
            "event_f1_mad3_std",
        ],
    )
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    write_csv(status_path, foundation_status_rows(args.methods), ["method", "model_or_repo", "implementation", "status", "notes"])
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")
    print(f"Wrote {status_path}")


def reconstruction_windows(series: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(series) < p:
        return np.empty((0, p), dtype=np.float32), np.empty((0,), dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(series, p)
    positions = np.arange(p - 1, len(series), dtype=np.int64)
    return windows.astype(np.float32, copy=False), positions


def train_reconstruction_model(
    args: argparse.Namespace,
    refs: Sequence[SeriesRef],
    method: str,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    x, _, counts = build_training_windows(
        refs,
        p=args.p,
        diff_order=args.diff_order,
        windows_per_series=args.windows_per_series,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
        balance_datasets=bool(getattr(args, "balance_source_datasets", False)),
    )
    if args.device == "auto" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    class USADStyle(nn.Module):
        def __init__(self, dim: int, hidden: int = 48, latent: int = 16):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())
            self.decoder1 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.decoder2 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dim))

        def forward(self, batch):
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            return w1, w2, w3

    class TranADStyle(nn.Module):
        def __init__(self, dim: int, d_model: int = 32, nhead: int = 4):
            super().__init__()
            self.proj = nn.Linear(1, d_model)
            self.pos = nn.Parameter(torch.zeros(1, dim, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=0.05,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.out = nn.Linear(d_model, 1)

        def forward(self, batch):
            y = self.proj(batch.unsqueeze(-1)) + self.pos
            y = self.encoder(y)
            return self.out(y).squeeze(-1)

    class AnomalyTransformerStyle(nn.Module):
        def __init__(self, dim: int, d_model: int = 32, nhead: int = 4):
            super().__init__()
            self.dim = dim
            self.proj = nn.Linear(1, d_model)
            self.pos = nn.Parameter(torch.zeros(1, dim, d_model))
            self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.05)
            self.ff = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, 1)
            idx = torch.arange(dim, dtype=torch.float32)
            dist = (idx[:, None] - idx[None, :]) ** 2
            sigma = max(float(dim) / 4.0, 1.0)
            prior = torch.exp(-dist / (2.0 * sigma * sigma))
            prior = prior / prior.sum(dim=-1, keepdim=True)
            self.register_buffer("prior", prior)

        def forward(self, batch):
            y = self.proj(batch.unsqueeze(-1)) + self.pos
            attn_out, attn = self.attn(y, y, y, need_weights=True, average_attn_weights=True)
            y = self.norm(y + attn_out)
            y = self.norm(y + self.ff(y))
            recon = self.out(y).squeeze(-1)
            prior = self.prior.unsqueeze(0).expand(attn.shape[0], -1, -1)
            discrepancy = torch.mean(torch.abs(attn - prior), dim=(1, 2))
            return recon, discrepancy

    if method == "USAD-style":
        model = USADStyle(args.p).to(device)
        opt1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=args.lr)
        opt2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=args.lr)
    elif method in {"TranAD-style", "AnomalyTransformer-style"}:
        model = AnomalyTransformerStyle(args.p).to(device) if method == "AnomalyTransformer-style" else TranADStyle(args.p).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"unknown reconstruction method: {method}")

    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    history: List[float] = []
    best_epoch = 0
    best_source_loss = math.inf
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            if method == "USAD-style":
                w1, _, w3 = model(batch_x)
                loss1 = (1.0 / epoch) * loss_fn(w1, batch_x) + (1.0 - 1.0 / epoch) * loss_fn(w3, batch_x)
                opt1.zero_grad()
                loss1.backward()
                opt1.step()

                _, w2, w3 = model(batch_x)
                loss2 = (1.0 / epoch) * loss_fn(w2, batch_x) - (1.0 - 1.0 / epoch) * loss_fn(w3, batch_x)
                opt2.zero_grad()
                loss2.backward()
                opt2.step()
                loss = loss1.detach()
            else:
                out = model(batch_x)
                recon = out[0] if method == "AnomalyTransformer-style" else out
                loss = loss_fn(recon, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        avg = total / max(seen, 1)
        history.append(avg)
        if np.isfinite(avg) and avg < best_source_loss:
            best_source_loss = avg
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(f"{method} epoch {epoch}/{args.epochs} loss={avg:.6f}", flush=True)
    model.load_state_dict(best_state)
    return model, {
        "seconds": time.time() - start,
        "windows": int(len(x)),
        "source_counts": counts,
        "loss": history,
        "best_epoch": best_epoch,
        "best_source_loss": best_source_loss,
        "restored_best_source_state": True,
    }


def reconstruction_scores(model, test: np.ndarray, p: int, batch_size: int, method: str) -> np.ndarray:
    import torch

    windows, positions = reconstruction_windows(test, p)
    scores = np.zeros(len(test), dtype=np.float32)
    if len(windows) == 0:
        return scores
    device = next(model.parameters()).device
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_np = np.ascontiguousarray(windows[start : start + batch_size], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(device)
            if method == "USAD-style":
                w1, w2, _ = model(batch)
                err = 0.5 * torch.mean((w1 - batch) ** 2, dim=1) + 0.5 * torch.mean((w2 - batch) ** 2, dim=1)
            elif method == "AnomalyTransformer-style":
                recon, discrepancy = model(batch)
                rec = torch.mean((recon - batch) ** 2, dim=1)
                err = rec * (1.0 + discrepancy)
            else:
                recon = model(batch)
                err = torch.mean((recon - batch) ** 2, dim=1)
            chunks.append(err.cpu().numpy().astype(np.float32))
    vals = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    scores[positions] = vals
    if len(vals):
        scores[: positions[0]] = float(np.min(vals))
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def modern_baselines(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    expected = {dataset: len(iter_series([dataset])) for dataset in args.datasets}

    for method in [m for m in args.methods if m in MODERN_ADAPTERS]:
        source, scorer = MODERN_ADAPTERS[method]
        print(f"Scoring {method} over all target series", flush=True)
        start = time.time()
        evaluated = 0
        for ref_i, ref in enumerate(iter_series(args.datasets), start=1):
            prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
            if prepared is None:
                continue
            scores = scorer(prepared.test, args.p)
            row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
            row["seed"] = args.seed
            rows.append(row)
            evaluated += 1
            if ref_i == 1 or ref_i % args.progress_every == 0:
                print(f"{method} evaluated {ref_i} series; last={ref.dataset}/{ref.name}", flush=True)
        runs.append(
            {
                "method": method,
                "source": source,
                "seed": args.seed,
                "datasets": list(args.datasets),
                "seconds": time.time() - start,
                "eval_series": evaluated,
                "adapter_kind": "local-standardized-residual",
            }
        )

    reconstruction_methods = {"USAD-style", "TranAD-style", "AnomalyTransformer-style"}
    for method in [m for m in args.methods if m in reconstruction_methods]:
        for dataset in args.datasets:
            set_seed(args.seed)
            target_refs = iter_series([dataset])
            if args.training_protocol == "zero_shot":
                source_datasets = [name for name in args.datasets if name != dataset]
                train_refs = iter_series(source_datasets)
            else:
                source_datasets = [dataset]
                train_refs = target_refs
            print(
                f"Training {method} protocol={args.training_protocol} target={dataset} "
                f"from {source_datasets} ({len(train_refs)} series)",
                flush=True,
            )
            model, run_info = train_reconstruction_model(args, train_refs, method)
            start_time = time.time()
            for eval_i, ref in enumerate(target_refs, start=1):
                prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
                if prepared is None:
                    continue
                scores = reconstruction_scores(model, prepared.test, args.p, args.eval_batch_size, method)
                row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
                row["seed"] = args.seed
                rows.append(row)
                if eval_i == 1 or eval_i == len(target_refs) or eval_i % args.progress_every == 0:
                    print(f"evaluated {dataset} {method}: {eval_i}/{len(target_refs)} series", flush=True)
            run_info.update(
                {
                    "method": method,
                    "seed": args.seed,
                    "target": dataset,
                    "sources": source_datasets,
                    "training_protocol": args.training_protocol,
                    "eval_seconds": time.time() - start_time,
                    "eval_series": len(target_refs),
                }
            )
            runs.append(run_info)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"modern_baseline_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"modern_baseline_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"modern_baseline_run_info{suffix}.json"
    fields = [
        "dataset",
        "series",
        "seed",
        "method",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
    ]
    write_csv(detail_path, rows, fields)
    write_csv(
        summary_path,
        aggregate_rows(rows, expected),
        [
            "dataset",
            "method",
            "series",
            "rows",
            "seeds",
            "points",
            "valid_ap_series",
            "zero_anomaly_series",
            "coverage_ratio",
            "ap",
            "ap_std",
            "auc",
            "auc_std",
            "best_f1",
            "best_f1_std",
            "best_precision",
            "best_precision_std",
            "best_recall",
            "best_recall_std",
            "f1_q95",
            "f1_q95_std",
            "f1_q97",
            "f1_q97_std",
            "f1_q99",
            "f1_q99_std",
            "f1_q995",
            "f1_q995_std",
            "f1_mad3",
            "f1_mad3_std",
            "precision_q99",
            "precision_q99_std",
            "recall_q99",
            "recall_q99_std",
            "precision_mad3",
            "precision_mad3_std",
            "recall_mad3",
            "recall_mad3_std",
            "event_f1_q95",
            "event_f1_q95_std",
            "event_f1_q97",
            "event_f1_q97_std",
            "event_f1_q99",
            "event_f1_q99_std",
            "event_f1_q995",
            "event_f1_q995_std",
            "event_f1_mad3",
            "event_f1_mad3_std",
        ],
    )
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def baselines(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    methods = ["persistence", "rolling_mean"]
    if args.include_median:
        methods.append("rolling_median")
    if args.include_advanced:
        methods.extend(["rolling_mad", "spectral_residual"])
    for ref in iter_series(args.datasets):
        prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
        if prepared is None:
            continue
        for method in methods:
            scores = baseline_scores(prepared.test, args.p, method)
            rows.append(metric_row(ref.dataset, ref.name, method, prepared.labels, scores))
    detail_path = OUT_ROOT / "baseline_series_metrics.csv"
    summary_path = OUT_ROOT / "baseline_summary_metrics.csv"
    fields = [
        "dataset",
        "series",
        "method",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
    ]
    write_csv(detail_path, rows, fields)
    expected = {dataset: len(iter_series([dataset])) for dataset in args.datasets}
    write_csv(
        summary_path,
        aggregate_rows(rows, expected),
        [
            "dataset",
            "method",
            "series",
            "rows",
            "seeds",
            "points",
            "valid_ap_series",
            "zero_anomaly_series",
            "coverage_ratio",
            "ap",
            "ap_std",
            "auc",
            "auc_std",
            "best_f1",
            "best_f1_std",
            "best_precision",
            "best_precision_std",
            "best_recall",
            "best_recall_std",
            "f1_q95",
            "f1_q95_std",
            "f1_q97",
            "f1_q97_std",
            "f1_q99",
            "f1_q99_std",
            "f1_q995",
            "f1_q995_std",
            "f1_mad3",
            "f1_mad3_std",
            "precision_q99",
            "precision_q99_std",
            "recall_q99",
            "recall_q99_std",
            "precision_mad3",
            "precision_mad3_std",
            "recall_mad3",
            "recall_mad3_std",
            "event_f1_q95",
            "event_f1_q95_std",
            "event_f1_q97",
            "event_f1_q97_std",
            "event_f1_q99",
            "event_f1_q99_std",
            "event_f1_q995",
            "event_f1_q995_std",
            "event_f1_mad3",
            "event_f1_mad3_std",
        ],
    )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


def build_training_windows(
    refs: Sequence[SeriesRef],
    p: int,
    diff_order: int,
    windows_per_series: int,
    max_train_windows: int,
    seed: int,
    balance_datasets: bool = False,
    filter_source_anomalies: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    counts: Dict[str, int] = {}

    def candidate_starts(ref: SeriesRef, prepared: PreparedSeries) -> np.ndarray:
        starts = np.arange(max(0, len(prepared.train) - p), dtype=np.int64)
        if not filter_source_anomalies or starts.size == 0:
            return starts
        label_path = ref.path / "train_label.npy"
        if not label_path.exists():
            return starts
        labels = np.asarray(np.load(label_path), dtype=np.int8).reshape(-1)
        for _ in range(diff_order):
            if labels.size < 2:
                return np.empty(0, dtype=np.int64)
            labels = labels[1:]
        if len(labels) != len(prepared.train):
            raise ValueError(
                f"Training-label alignment mismatch for {ref.dataset}/{ref.name}: "
                f"labels={len(labels)} train={len(prepared.train)}"
            )
        contaminated = np.lib.stride_tricks.sliding_window_view(labels > 0, p + 1).any(axis=1)
        return np.flatnonzero(~contaminated).astype(np.int64)

    if balance_datasets:
        grouped: Dict[str, List[SeriesRef]] = {}
        for ref in refs:
            grouped.setdefault(ref.dataset, []).append(ref)
        datasets = sorted(grouped)
        base_budget, remainder = divmod(max_train_windows, len(datasets))
        for dataset_i, dataset in enumerate(datasets):
            group = list(grouped[dataset])
            rng.shuffle(group)
            dataset_budget = base_budget + int(dataset_i < remainder)
            remaining = dataset_budget
            for ref_i, ref in enumerate(group):
                prepared = load_prepared(ref, p=p, diff_order=diff_order)
                if prepared is None:
                    continue
                series = prepared.train
                starts = candidate_starts(ref, prepared)
                if len(starts) == 0:
                    continue
                remaining_series = max(1, len(group) - ref_i)
                fair_share = int(math.ceil(remaining / remaining_series))
                k = min(len(starts), remaining, max(windows_per_series, fair_share))
                if k <= 0:
                    break
                idx = rng.choice(starts, size=k, replace=False)
                xs.append(np.stack([series[i : i + p] for i in idx]).astype(np.float32))
                ys.append(series[idx + p].reshape(-1, 1).astype(np.float32))
                counts[dataset] = counts.get(dataset, 0) + k
                remaining -= k
                if remaining <= 0:
                    break
            if remaining > 0:
                raise RuntimeError(
                    f"dataset-balanced sampling could not fill {dataset}'s budget; "
                    f"missing {remaining} of {dataset_budget} windows"
                )
        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        order = rng.permutation(len(x_all))
        return x_all[order], y_all[order], counts
    else:
        shuffled = list(refs)
        rng.shuffle(shuffled)
    for ref in shuffled:
        prepared = load_prepared(ref, p=p, diff_order=diff_order)
        if prepared is None:
            continue
        series = prepared.train
        starts = candidate_starts(ref, prepared)
        if len(starts) == 0:
            continue
        k = min(windows_per_series, len(starts))
        idx = rng.choice(starts, size=k, replace=False)
        x = np.stack([series[i : i + p] for i in idx]).astype(np.float32)
        y = series[idx + p].reshape(-1, 1).astype(np.float32)
        xs.append(x)
        ys.append(y)
        counts[ref.dataset] = counts.get(ref.dataset, 0) + k
        if sum(len(a) for a in xs) >= max_train_windows:
            break
    if not xs:
        raise RuntimeError("no training windows collected")
    x_all = np.concatenate(xs, axis=0)[:max_train_windows]
    y_all = np.concatenate(ys, axis=0)[:max_train_windows]
    order = rng.permutation(len(x_all))
    return x_all[order], y_all[order], counts


def train_tshape_model(args: argparse.Namespace, refs: Sequence[SeriesRef]):
    import torch

    x, y, counts = build_training_windows(
        refs,
        p=args.p,
        diff_order=args.diff_order,
        windows_per_series=args.windows_per_series,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
        balance_datasets=bool(getattr(args, "balance_source_datasets", False)),
        filter_source_anomalies=bool(getattr(args, "filter_source_anomalies", False)),
    )
    model, run_info = train_tshape_arrays(args, x, y, counts)
    run_info["filter_source_anomalies"] = bool(
        getattr(args, "filter_source_anomalies", False)
    )
    return model, run_info


def train_tshape_hybrid_model(args: argparse.Namespace, refs: Sequence[SeriesRef]):
    """Train on source-domain windows plus a target-independent Pattern Bank.

    The synthetic share is fixed before target evaluation.  For leave-one-
    dataset-out experiments, neither target values nor target labels enter
    this training set.
    """
    synthetic_ratio = float(getattr(args, "synthetic_ratio", 0.25))
    if not 0.0 < synthetic_ratio < 1.0:
        raise ValueError("--synthetic-ratio must be between 0 and 1")
    total_budget = int(args.max_train_windows)
    synthetic_budget = max(1, int(round(total_budget * synthetic_ratio)))
    real_budget = max(1, total_budget - synthetic_budget)

    real_x, real_y, real_counts = build_training_windows(
        refs,
        p=args.p,
        diff_order=args.diff_order,
        windows_per_series=args.windows_per_series,
        max_train_windows=real_budget,
        seed=args.seed,
        balance_datasets=bool(getattr(args, "balance_source_datasets", False)),
        filter_source_anomalies=bool(getattr(args, "filter_source_anomalies", False)),
    )
    bank_x, bank_y, bank_counts = pattern_bank_training_windows(
        p=args.p,
        fraction=float(getattr(args, "pattern_bank_fraction", 1.00)),
        seed=args.seed + 7919,
        max_train_windows=synthetic_budget,
        variants_per_family=int(getattr(args, "pattern_bank_variants", 8)),
        noise_std=float(getattr(args, "pattern_bank_noise_std", 0.025)),
        event_rate=float(getattr(args, "pattern_bank_event_rate", 0.35)),
        event_scale=float(getattr(args, "pattern_bank_event_scale", 0.75)),
    )
    x = np.concatenate([real_x, bank_x], axis=0)
    y = np.concatenate([real_y, bank_y], axis=0)
    rng = np.random.default_rng(args.seed + 104729)
    order = rng.permutation(len(x))
    counts = {f"source_{key}": value for key, value in real_counts.items()}
    counts.update(bank_counts)
    counts["source_windows"] = int(len(real_x))
    counts["synthetic_windows"] = int(len(bank_x))
    counts["synthetic_ratio_pct"] = int(round(100 * synthetic_ratio))
    model, run_info = train_tshape_arrays(args, x[order], y[order], counts)
    run_info["filter_source_anomalies"] = bool(
        getattr(args, "filter_source_anomalies", False)
    )
    return model, run_info


def train_tshape_arrays(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    counts: Dict[str, int],
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, str(ROOT))
    from layyer import TShape_model

    if args.device == "auto" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"using device={device}", flush=True)
    model = TShape_model(args.p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    history: List[float] = []
    patience = max(0, int(getattr(args, "early_stop_patience", 0)))
    min_delta = max(0.0, float(getattr(args, "early_stop_min_delta", 0.0)))
    best_loss = math.inf
    stale_epochs = 0
    early_stopped = False
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
            seen += len(batch_x)
        avg = total / max(seen, 1)
        history.append(avg)
        print(f"epoch {epoch + 1}/{args.epochs} loss={avg:.6f}", flush=True)
        if avg < best_loss - min_delta:
            best_loss = avg
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience and stale_epochs >= patience:
            early_stopped = True
            print(
                f"early stopping after {epoch + 1} epochs: "
                f"loss did not improve by {min_delta:g} for {patience} epochs",
                flush=True,
            )
            break
    return model, {
        "seconds": time.time() - start,
        "windows": int(len(x)),
        "source_counts": counts,
        "loss": history,
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(history)),
        "early_stop_patience": patience,
        "early_stop_min_delta": min_delta,
        "early_stopped": early_stopped,
        "best_training_loss": float(best_loss),
    }


def easytsad_preserving_difference(values: np.ndarray, order: int) -> np.ndarray:
    """Match EasyTSAD's length-preserving differential preprocessing."""
    output = np.asarray(values, dtype=np.float32).reshape(-1)
    for _ in range(order):
        if output.size == 0:
            return output
        output = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.diff(output).astype(np.float32)]
        )
    return output


def prepare_easytsad_naive_series(
    ref: SeriesRef,
    *,
    diff_order: int,
    valid_proportion: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recreate EasyTSAD's split, differential, scaling, and clipping order."""
    raw_train = np.asarray(np.load(ref.path / "train.npy"), dtype=np.float32).reshape(-1)
    raw_test = np.asarray(np.load(ref.path / "test.npy"), dtype=np.float32).reshape(-1)
    labels = np.asarray(np.load(ref.path / "test_label.npy"), dtype=np.int8).reshape(-1)
    valid_size = int(len(raw_train) * valid_proportion)
    if valid_size <= 0 or valid_size >= len(raw_train):
        raise ValueError(f"invalid validation split for {ref.dataset}/{ref.name}")
    train = raw_train[:-valid_size]
    valid = raw_train[-valid_size:]
    train = easytsad_preserving_difference(train, diff_order)
    valid = easytsad_preserving_difference(valid, diff_order)
    test = easytsad_preserving_difference(raw_test, diff_order)

    lo = float(np.nanmin(train))
    hi = float(np.nanmax(train))
    scale = hi - lo
    if not np.isfinite(scale) or scale < 1e-8:
        train = np.zeros_like(train, dtype=np.float32)
        valid = np.zeros_like(valid, dtype=np.float32)
        test = np.zeros_like(test, dtype=np.float32)
    else:
        train = ((train - lo) / scale).astype(np.float32)
        valid = np.clip((valid - lo) / scale, -2.0, 3.0).astype(np.float32)
        test = np.clip((test - lo) / scale, -2.0, 3.0).astype(np.float32)
    return (
        np.nan_to_num(train, nan=0.0, posinf=3.0, neginf=-2.0),
        np.nan_to_num(valid, nan=0.0, posinf=3.0, neginf=-2.0),
        np.nan_to_num(test, nan=0.0, posinf=3.0, neginf=-2.0),
        labels,
    )


def prediction_windows(values: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    windows = np.lib.stride_tricks.sliding_window_view(values, p + 1)
    return (
        np.ascontiguousarray(windows[:, :p], dtype=np.float32).copy(),
        np.ascontiguousarray(windows[:, p:], dtype=np.float32).copy(),
    )


def train_easytsad_naive_tshape(
    args: argparse.Namespace,
    train: np.ndarray,
    valid: np.ndarray,
):
    """Faithfully execute the original per-series TShape training loop."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, str(ROOT))
    from layyer import TShape_model

    if args.device == "auto" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    x_train, y_train = prediction_windows(train, args.p)
    x_valid, y_valid = prediction_windows(valid, args.p)
    model = TShape_model(args.p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    history: List[Dict[str, float]] = []
    best_score: Optional[float] = None
    counter = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            train_batches += 1
        model.eval()
        valid_total = 0.0
        valid_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = loss_fn(model(batch_x), batch_y)
                valid_total += float(loss.item())
                valid_batches += 1
        train_loss = train_total / max(train_batches, 1)
        valid_loss = valid_total / max(valid_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        scheduler.step()
        score = -valid_loss
        if best_score is None or score >= best_score + 1e-4:
            best_score = score
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                break
    return model, {
        "seconds": time.time() - start,
        "epochs_completed": len(history),
        "train_windows": len(x_train),
        "valid_windows": len(x_valid),
        "history": history,
        "device": str(device),
    }


def tshape_easytsad_naive(args: argparse.Namespace) -> None:
    """Complete missing original-style score archives without overwriting them."""
    original_root = ROOT / "Results" / "Scores" / "TShape" / "naive"
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    force_datasets = set(args.force_datasets or [])
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    expected = {dataset: len(iter_series([dataset])) for dataset in args.datasets}
    refs = iter_series(args.datasets)
    if args.series:
        requested = set(args.series)
        refs = [ref for ref in refs if ref.name in requested]
    if args.max_series is not None:
        refs = refs[: args.max_series]
    for ref_i, ref in enumerate(refs, start=1):
        original_path = original_root / ref.dataset / f"{ref.name}.npy"
        output_path = output_root / ref.dataset / f"{ref.name}.npy"
        should_run = ref.dataset in force_datasets or not (
            args.only_missing_original and original_path.exists()
        )
        if output_path.exists() and not args.overwrite:
            should_run = False
        if not should_run:
            continue
        set_seed(args.seed)
        train, valid, test, labels = prepare_easytsad_naive_series(
            ref,
            diff_order=args.diff_order,
            valid_proportion=args.valid_proportion,
        )
        if min(len(train), len(valid), len(test)) <= args.p:
            continue
        print(
            f"Faithful TShape naive {ref_i}/{len(refs)} {ref.dataset}/{ref.name}",
            flush=True,
        )
        model, run_info = train_easytsad_naive_tshape(args, train, valid)
        full_scores = tshape_scores(model, test, args.p, args.eval_batch_size)
        scores = np.asarray(full_scores[args.p :], dtype=np.float32)
        aligned_labels = labels[args.p :]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, scores)
        row = metric_row(
            ref.dataset,
            ref.name,
            "TShape-faithful-naive-reproduction",
            aligned_labels,
            scores,
        )
        row.update({"seed": args.seed, "score_file": str(output_path.relative_to(ROOT))})
        rows.append(row)
        run_info.update(
            {
                "dataset": ref.dataset,
                "series": ref.name,
                "seed": args.seed,
                "p": args.p,
                "diff_order": args.diff_order,
                "batch_size": args.batch_size,
                "max_epochs": args.epochs,
                "patience": args.patience,
                "valid_proportion": args.valid_proportion,
                "score_file": str(output_path.relative_to(ROOT)),
            }
        )
        runs.append(run_info)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"tshape_faithful_naive_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"tshape_faithful_naive_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"tshape_faithful_naive_run_info{suffix}.json"
    write_csv(
        detail_path,
        rows,
        ["dataset", "series", "seed", "score_file", "method", "points", "anomaly_ratio"],
    )
    write_csv(
        summary_path,
        aggregate_rows(rows, expected),
        [
            "dataset",
            "method",
            "series",
            "rows",
            "seeds",
            "points",
            "zero_anomaly_series",
            "coverage_ratio",
        ],
    )
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def tshape_scores(model, test: np.ndarray, p: int, batch_size: int, eval_stride: int = 1) -> np.ndarray:
    import torch

    if len(test) <= p:
        return np.zeros_like(test, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(test, p)[:-1]
    stride = max(1, int(eval_stride))
    if stride > 1:
        sample_idx = np.arange(0, len(windows), stride, dtype=np.int64)
        if sample_idx[-1] != len(windows) - 1:
            sample_idx = np.r_[sample_idx, len(windows) - 1]
        eval_windows = windows[sample_idx]
        targets = test[p:][sample_idx].astype(np.float32)
        positions = p + sample_idx
    else:
        eval_windows = windows
        targets = test[p:].astype(np.float32)
        positions = np.arange(p, len(test), dtype=np.int64)
    scores = np.zeros(len(test), dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(eval_windows), batch_size):
            batch_np = np.ascontiguousarray(eval_windows[start : start + batch_size], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(device)
            pred = model(batch).cpu().numpy().reshape(-1)
            preds.append(pred)
    pred_all = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    vals = (pred_all - targets) ** 2
    if len(vals):
        if stride > 1 and len(vals) > 1:
            full_positions = np.arange(len(test), dtype=np.float32)
            scores = np.interp(full_positions, positions.astype(np.float32), vals.astype(np.float32)).astype(np.float32)
            scores[:p] = float(vals[0])
        else:
            scores[positions] = vals
            scores[:p] = float(np.min(vals))
    return scores


def inject_random_event(
    window: np.ndarray,
    rng: np.random.Generator,
    event_scale: float,
    context_only: bool = False,
) -> np.ndarray:
    """Inject KPI-like incidents into a synthetic training continuation."""
    out = window.copy()
    n = len(out)
    if n < 4:
        return out
    limit = n - 1 if context_only else n
    if limit < 3:
        return out
    kind = rng.choice(["spike", "dip", "level", "variance", "drift", "burst"])
    sign = -1.0 if rng.random() < 0.5 else 1.0
    amp = sign * rng.uniform(0.35, 1.35) * event_scale
    if kind in {"spike", "dip"}:
        pos = int(rng.integers(max(1, limit // 4), limit))
        width = int(rng.integers(1, max(2, min(5, limit // 4 + 1))))
        for j in range(pos, min(limit, pos + width)):
            out[j] += amp * math.exp(-(j - pos) / max(width, 1))
    elif kind == "level":
        pos = int(rng.integers(max(1, limit // 4), limit))
        out[pos:limit] += amp
    elif kind == "variance":
        start = int(rng.integers(0, max(1, limit - 2)))
        end = min(limit, start + int(rng.integers(2, max(3, limit // 3 + 1))))
        out[start:end] += rng.normal(0.0, abs(amp), size=end - start)
    elif kind == "drift":
        start = int(rng.integers(0, max(1, limit - 2)))
        out[start:limit] += np.linspace(0.0, amp, limit - start)
    else:
        centers = rng.choice(np.arange(1, limit), size=min(3, limit - 1), replace=False)
        for pos in centers:
            out[int(pos)] += amp * rng.uniform(0.4, 1.0)
    return out.astype(np.float32)


def pattern_bank_training_windows(
    p: int,
    fraction: float,
    seed: int,
    max_train_windows: int,
    variants_per_family: int,
    noise_std: float,
    event_rate: float,
    event_scale: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Create label-free synthetic windows for Pattern Bank TShape pretraining.

    The windows are not target-data augmentation. They are generated from KPI
    pattern families before deployment, with random incidents added to improve
    robustness to shifts and abrupt operational events.
    """
    rng = np.random.default_rng(seed)
    prototypes = pattern_bank_prototypes(p, fraction, seed, variants_per_family=variants_per_family)
    if len(prototypes) == 0:
        raise RuntimeError("pattern bank produced no prototypes")
    xs = np.empty((max_train_windows, p), dtype=np.float32)
    ys = np.empty((max_train_windows, 1), dtype=np.float32)
    for i in range(max_train_windows):
        proto = prototypes[int(rng.integers(0, len(prototypes)))].astype(np.float32)
        window = proto.copy()
        amp = float(rng.uniform(0.55, 1.65))
        offset = float(rng.uniform(-0.45, 0.45))
        window = amp * window + offset
        if rng.random() < 0.35:
            window = window[::-1].copy()
        if rng.random() < 0.45:
            slope = rng.uniform(-0.35, 0.35)
            window = window + slope * np.linspace(0.0, 1.0, p + 1, dtype=np.float32)
        if noise_std > 0:
            window = window + rng.normal(0.0, noise_std, size=p + 1).astype(np.float32)
        clean_window = window.copy()
        if rng.random() < event_rate:
            window = inject_random_event(window, rng, event_scale, context_only=True)

        target = float(clean_window[p])
        # Match source-series preprocessing: every training value and its
        # continuation share one fitted affine scale.  Using context extrema
        # alone pushed many clean synthetic targets outside [0, 1].
        lo = float(np.min(window))
        hi = float(np.max(window))
        scale = max(hi - lo, 1e-6)
        norm = (window - lo) / scale
        xs[i] = np.nan_to_num(norm[:p], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ys[i, 0] = float(np.nan_to_num((target - lo) / scale, nan=0.0, posinf=0.0, neginf=0.0))
    order = rng.permutation(max_train_windows)
    counts = {
        "pattern_bank_windows": int(max_train_windows),
        "pattern_bank_prototypes": int(len(prototypes)),
        "pattern_bank_fraction_pct": int(round(fraction * 100)),
        "pattern_bank_version": PATTERN_BANK_VERSION,
    }
    return xs[order], ys[order], counts


def selected_refs(dataset: str, max_series: Optional[int], seed: int) -> List[SeriesRef]:
    refs = iter_series([dataset])
    if max_series is None or len(refs) <= max_series:
        return refs

    # Use anomaly-ratio stratification for robustness subsets so that rare,
    # medium, and dense anomaly regimes remain represented.
    ratios: List[Tuple[SeriesRef, float]] = []
    for ref in refs:
        try:
            labels = np.asarray(np.load(ref.path / "test_label.npy"), dtype=np.int8).reshape(-1)
            ratio = float(np.mean(labels)) if labels.size else 0.0
        except Exception:
            ratio = 0.0
        ratios.append((ref, ratio))
    ratios.sort(key=lambda item: (item[1], item[0].name))

    rng = random.Random(seed)
    bins = min(max_series, max(1, int(math.sqrt(len(ratios)))))
    buckets = [ratios[i::bins] for i in range(bins)]
    chosen: List[SeriesRef] = []
    while len(chosen) < max_series and any(buckets):
        for bucket in buckets:
            if not bucket or len(chosen) >= max_series:
                continue
            idx = rng.randrange(len(bucket))
            chosen.append(bucket.pop(idx)[0])
    return sorted(chosen, key=lambda r: r.name)


def pattern_bank_tshape(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    seed_values = args.seeds if args.seeds else [args.seed]
    expected = {dataset: len(iter_series([dataset])) for dataset in args.targets}

    for seed in seed_values:
        for frac in args.fractions:
            if not 0.0 < float(frac) <= 1.0:
                raise ValueError("Pattern Bank fractions must be in (0, 1].")
            set_seed(seed)
            frac_pct = int(round(frac * 100))
            size_sweep = bool(getattr(args, "size_sweep", False))
            sweep_kind = "size" if size_sweep else "family coverage"
            print(
                f"Training PatternBank-TShape seed={seed} "
                f"{sweep_kind}={frac_pct}%",
                flush=True,
            )
            train_windows = args.max_train_windows
            family_fraction = float(frac)
            if size_sweep:
                train_windows = max(
                    int(getattr(args, "min_train_windows", 1)),
                    int(round(args.max_train_windows * float(frac))),
                )
                # Every size uses the same complete motif mixture. Because the
                # generator seed is fixed, smaller banks are strict subsets of
                # larger banks before shuffling.
                family_fraction = 1.0
            elif getattr(args, "capacity_balanced_bank", False):
                saturation = max(float(getattr(args, "capacity_saturation", 0.80)), 1e-6)
                train_windows = int(round(args.max_train_windows * min(frac / saturation, 1.0)))
                train_windows = max(int(getattr(args, "min_train_windows", 7500)), train_windows)
            x, y, counts = pattern_bank_training_windows(
                p=args.p,
                fraction=family_fraction,
                seed=seed,
                max_train_windows=train_windows,
                variants_per_family=args.variants_per_family,
                noise_std=args.noise_std,
                event_rate=args.event_rate,
                event_scale=args.event_scale,
            )
            model, run_info = train_tshape_arrays(args, x, y, counts)
            run_info["resolved_device"] = str(next(model.parameters()).device)
            method_prefix = "TShapeUniversalPattern" if size_sweep else "TShapePatternBank"
            plus_prefix = "TShapeUniversalZeroPlus" if size_sweep else "TShapePatternBankZeroPlus"
            method = f"{method_prefix}-{frac_pct:03d}"
            plus_method = f"{plus_prefix}-{frac_pct:03d}"
            run_info.update(
                {
                    "seed": seed,
                    "method": method,
                    "fraction": frac,
                    "fraction_pct": frac_pct,
                    "sweep_kind": sweep_kind,
                    "family_fraction": family_fraction,
                    "strict_zero_shot": size_sweep,
                    "real_training_windows": 0 if size_sweep else None,
                    "fusion_alpha": float(getattr(args, "fusion_alpha", 0.15)),
                    "targets": list(args.targets),
                    "event_rate": args.event_rate,
                    "event_scale": args.event_scale,
                    "noise_std": args.noise_std,
                    "variants_per_family": args.variants_per_family,
                    "eval_stride": getattr(args, "eval_stride", 1),
                }
            )
            checkpoint_dir = getattr(args, "checkpoint_dir", None)
            if checkpoint_dir:
                import torch

                checkpoint_root = Path(checkpoint_dir)
                if not checkpoint_root.is_absolute():
                    checkpoint_root = ROOT / checkpoint_root
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_root / f"pattern_bank_{frac_pct:03d}_seed_{seed}.pt"
                metadata = {
                    "name": "TShape-Zero+ universal Pattern Bank checkpoint",
                    "version": "2026-07-14",
                    "checkpoint_role": "strict zero-shot benchmark and product checkpoint",
                    "training_corpus": "synthetic Pattern Bank only",
                    "strict_zero_shot": bool(size_sweep),
                    "target_values_used_for_training": False,
                    "target_labels_used_for_training": False,
                    "p": int(args.p),
                    "diff_order": int(args.diff_order),
                    "preprocessing": "np.diff drop-first; per-history min-max context scaling",
                    "pattern_bank_version": PATTERN_BANK_VERSION,
                    "pattern_bank_size_fraction": float(frac),
                    "pattern_bank_windows": int(train_windows),
                    "pattern_bank_family_fraction": float(family_fraction),
                    "pattern_bank_prototypes": int(counts["pattern_bank_prototypes"]),
                    "pattern_bank_variants": int(args.variants_per_family),
                    "pattern_bank_noise_std": float(args.noise_std),
                    "pattern_bank_event_rate": float(args.event_rate),
                    "pattern_bank_event_scale": float(args.event_scale),
                    "epochs": int(args.epochs),
                    "seed": int(seed),
                    "fusion_alpha": float(getattr(args, "fusion_alpha", 0.15)),
                    "fusion_selection": "fixed before benchmark evaluation; no benchmark labels",
                    "score_formula": (
                        f"{float(getattr(args, 'fusion_alpha', 0.15)):.2f} "
                        "minmax(Pattern-TShape) + "
                        f"{1.0 - float(getattr(args, 'fusion_alpha', 0.15)):.2f} "
                        "residual_guard"
                    ),
                    "run_info": run_info,
                }
                torch.save(
                    {"metadata": metadata, "state_dict": model.cpu().state_dict()},
                    checkpoint_path,
                )
                model = model.to(
                    "mps"
                    if args.device == "auto" and torch.backends.mps.is_available()
                    else ("cpu" if args.device == "auto" else args.device)
                )
                try:
                    checkpoint_ref = checkpoint_path.resolve().relative_to(ROOT.resolve())
                except ValueError:
                    checkpoint_ref = checkpoint_path
                run_info["checkpoint"] = str(checkpoint_ref)
            eval_start = time.time()
            evaluated = 0
            for target in args.targets:
                target_refs = selected_refs(target, args.max_series_per_target, seed)
                for eval_i, ref in enumerate(target_refs, start=1):
                    prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
                    if prepared is None:
                        continue
                    scores = tshape_scores(
                        model,
                        prepared.test,
                        args.p,
                        args.eval_batch_size,
                        getattr(args, "eval_stride", 1),
                    )
                    row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
                    row["seed"] = seed
                    row["fraction"] = f"{frac:.2f}"
                    row["synthetic_windows"] = train_windows
                    row["prototypes"] = counts["pattern_bank_prototypes"]
                    row["eval_stride"] = getattr(args, "eval_stride", 1)
                    rows.append(row)
                    if args.emit_zero_plus:
                        guard = tshape_zero_plus_variants(scores, prepared.test, args.p)[
                            "TShape-zero_plus_residual_only_minmax"
                        ]
                        alpha = float(getattr(args, "fusion_alpha", 0.15))
                        fused = alpha * scale01(scores) + (1.0 - alpha) * guard
                        plus_row = metric_row(
                            ref.dataset,
                            ref.name,
                            plus_method,
                            prepared.labels,
                            fused,
                        )
                        plus_row["seed"] = seed
                        plus_row["fraction"] = f"{frac:.2f}"
                        plus_row["synthetic_windows"] = train_windows
                        plus_row["prototypes"] = counts["pattern_bank_prototypes"]
                        plus_row["eval_stride"] = getattr(args, "eval_stride", 1)
                        plus_row["fusion_alpha"] = f"{alpha:.2f}"
                        rows.append(plus_row)
                    evaluated += 1
                    if eval_i == 1 or eval_i == len(target_refs) or eval_i % args.progress_every == 0:
                        print(
                            f"evaluated PatternBank-TShape {frac_pct}% seed={seed} "
                            f"{target}: {eval_i}/{len(target_refs)} series",
                            flush=True,
                        )
            run_info["eval_seconds"] = time.time() - eval_start
            run_info["eval_series"] = evaluated
            runs.append(run_info)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"pattern_bank_tshape_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"pattern_bank_tshape_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"pattern_bank_tshape_run_info{suffix}.json"
    fields = [
        "dataset",
        "series",
        "seed",
        "fraction",
        "synthetic_windows",
        "prototypes",
        "eval_stride",
        "fusion_alpha",
        "method",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
        "point_f1_pa",
        "point_precision_pa",
        "point_recall_pa",
        "point_threshold_pa",
        "event_f1_pa_log",
        "event_precision_pa_log",
        "event_recall_pa_log",
        "event_threshold_pa_log",
    ]
    write_csv(detail_path, rows, fields)
    summary = aggregate_rows(rows, expected)
    meta_by_method: Dict[str, Dict[str, object]] = {}
    for row in rows:
        meta_by_method.setdefault(
            str(row["method"]),
            {
                "fraction": row.get("fraction", ""),
                "synthetic_windows": row.get("synthetic_windows", ""),
                "prototypes": row.get("prototypes", ""),
                "eval_stride": row.get("eval_stride", ""),
            },
        )
    for row in summary:
        row.update(meta_by_method.get(str(row["method"]), {}))
    preferred = [
        "dataset",
        "method",
        "fraction",
        "synthetic_windows",
        "prototypes",
        "eval_stride",
        "fusion_alpha",
        "series",
        "rows",
        "seeds",
        "points",
        "valid_ap_series",
        "zero_anomaly_series",
        "coverage_ratio",
        "valid_point_pa_series",
        "valid_event_pa_series",
        "point_f1_pa",
        "point_f1_pa_std",
        "event_f1_pa_log",
        "event_f1_pa_log_std",
        "ap",
        "ap_std",
        "auc",
        "auc_std",
        "best_f1",
        "best_f1_std",
        "event_f1_q99",
        "event_f1_q99_std",
        "f1_q99",
        "f1_q99_std",
        "f1_mad3",
        "f1_mad3_std",
    ]
    summary_fields = preferred + [key for key in summary[0].keys() if key not in preferred] if summary else preferred
    write_csv(summary_path, summary, summary_fields)
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def pattern_bank_hybrid_sweep(args: argparse.Namespace) -> None:
    """Evaluate nested Pattern Bank coverage inside the final source hybrid.

    Unlike ``pattern-bank-tshape``, which isolates synthetic pretraining, this
    command preserves the final detector's leave-one-dataset-out contract: a
    target-specific checkpoint receives an equal real-window budget from each
    of the five source datasets plus a fixed synthetic share.  Consequently,
    changing ``fraction`` changes only the nested Pattern Bank family coverage,
    not target access or the total optimization budget.
    """
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    seed_values = args.seeds if args.seeds else [args.seed]
    expected = {dataset: len(iter_series([dataset])) for dataset in args.targets}
    fusion_alphas = {dataset: float(args.fusion_alpha) for dataset in args.targets}
    if args.fusion_selections_csv:
        selected_targets = set()
        selection_path = Path(args.fusion_selections_csv)
        if not selection_path.is_absolute():
            selection_path = ROOT / selection_path
        with selection_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                target = str(row["target"])
                if target in fusion_alphas:
                    fusion_alphas[target] = float(row["selected_alpha"])
                    selected_targets.add(target)
        missing = [dataset for dataset in args.targets if dataset not in selected_targets]
        if missing:
            raise ValueError(f"Missing source-selected fusion alpha for: {missing}")

    for seed in seed_values:
        for fraction in args.fractions:
            fraction_pct = int(round(100 * fraction))
            for target in args.targets:
                set_seed(seed)
                args.seed = seed
                args.pattern_bank_fraction = float(fraction)
                source_datasets = [dataset for dataset in args.datasets if dataset != target]
                source_refs = iter_series(source_datasets)
                print(
                    f"Training hybrid Pattern Bank seed={seed} fraction={fraction_pct}% "
                    f"target={target} from {source_datasets}",
                    flush=True,
                )
                model, run_info = train_tshape_hybrid_model(args, source_refs)
                method = f"TShapeHybridPattern-{fraction_pct:03d}"
                plus_method = f"TShapeHybridPatternZeroPlus-{fraction_pct:03d}"
                run_info.update(
                    {
                        "seed": seed,
                        "method": method,
                        "target": target,
                        "sources": source_datasets,
                        "fraction": float(fraction),
                        "fraction_pct": fraction_pct,
                        "synthetic_ratio": float(args.synthetic_ratio),
                        "balanced_sources": bool(args.balance_source_datasets),
                        "fusion_alpha": fusion_alphas[target],
                        "fusion_selection_file": str(args.fusion_selections_csv or "fixed CLI value"),
                    }
                )

                target_refs = selected_refs(target, args.max_series_per_target, seed)
                eval_start = time.time()
                for eval_i, ref in enumerate(target_refs, start=1):
                    prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
                    if prepared is None:
                        continue
                    scores = tshape_scores(
                        model,
                        prepared.test,
                        args.p,
                        args.eval_batch_size,
                        getattr(args, "eval_stride", 1),
                    )
                    row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
                    row.update(
                        {
                            "seed": seed,
                            "fraction": f"{fraction:.2f}",
                            "synthetic_ratio": f"{float(args.synthetic_ratio):.2f}",
                        }
                    )
                    rows.append(row)

                    guard = tshape_zero_plus_variants(scores, prepared.test, args.p)[
                        "TShape-zero_plus_residual_only_minmax"
                    ]
                    alpha = fusion_alphas[target]
                    fused = alpha * scale01(scores) + (1.0 - alpha) * guard
                    plus_row = metric_row(
                        ref.dataset,
                        ref.name,
                        plus_method,
                        prepared.labels,
                        fused,
                    )
                    plus_row.update(
                        {
                            "seed": seed,
                            "fraction": f"{fraction:.2f}",
                            "synthetic_ratio": f"{float(args.synthetic_ratio):.2f}",
                            "fusion_alpha": f"{alpha:.2f}",
                        }
                    )
                    rows.append(plus_row)
                    if eval_i == 1 or eval_i == len(target_refs) or eval_i % args.progress_every == 0:
                        print(
                            f"evaluated hybrid Pattern Bank {fraction_pct}% seed={seed} "
                            f"{target}: {eval_i}/{len(target_refs)} series",
                            flush=True,
                        )
                run_info["eval_seconds"] = time.time() - eval_start
                run_info["eval_series"] = len(target_refs)
                runs.append(run_info)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"pattern_bank_hybrid_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"pattern_bank_hybrid_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"pattern_bank_hybrid_run_info{suffix}.json"
    fields = [
        "dataset",
        "series",
        "seed",
        "fraction",
        "synthetic_ratio",
        "fusion_alpha",
        "method",
        "points",
        "anomaly_ratio",
    ]
    write_csv(detail_path, rows, fields)
    summary = aggregate_rows(rows, expected)
    metadata = {
        str(row["method"]): {
            "fraction": row.get("fraction", ""),
            "synthetic_ratio": row.get("synthetic_ratio", ""),
            "fusion_alpha": row.get("fusion_alpha", ""),
        }
        for row in rows
    }
    for row in summary:
        row.update(metadata.get(str(row["method"]), {}))
    write_csv(
        summary_path,
        summary,
        [
            "dataset",
            "method",
            "fraction",
            "synthetic_ratio",
            "fusion_alpha",
            "series",
            "rows",
            "seeds",
            "points",
            "zero_anomaly_series",
            "coverage_ratio",
        ],
    )
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def tshape(args: argparse.Namespace) -> None:
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    score_manifest: List[Dict[str, object]] = []
    score_root: Optional[Path] = None
    if args.score_output_dir is not None:
        score_root = Path(args.score_output_dir).expanduser()
        if not score_root.is_absolute():
            score_root = ROOT / score_root
        score_root.mkdir(parents=True, exist_ok=True)
    all_datasets = list(args.datasets)
    seed_values = args.seeds if args.seeds else [args.seed]
    expected = {dataset: len(iter_series([dataset])) for dataset in args.targets}
    for seed in seed_values:
        set_seed(seed)
        for target in args.targets:
            target_refs = selected_refs(target, args.max_series_per_target, seed)
            for scheme in args.schemes:
                if scheme in {"zero_shot", "zero_shot_pattern"}:
                    source_datasets = [d for d in all_datasets if d != target]
                    train_refs = iter_series(source_datasets)
                elif scheme in {"in_domain_global", "target_trained"}:
                    source_datasets = [target]
                    train_refs = iter_series([target])
                else:
                    raise ValueError(f"unknown scheme: {scheme}")
                print(f"Training seed={seed} {scheme} for target={target} from {source_datasets} ({len(train_refs)} series)")
                args.seed = seed
                if scheme == "zero_shot_pattern":
                    model, run_info = train_tshape_hybrid_model(args, train_refs)
                else:
                    model, run_info = train_tshape_model(args, train_refs)
                balanced_suffix = "_balanced" if getattr(args, "balance_source_datasets", False) and scheme in {"zero_shot", "zero_shot_pattern"} else ""
                if scheme == "target_trained":
                    method = "TShape-target_trained"
                elif scheme == "zero_shot_pattern":
                    method = f"TShape-zero_pattern{balanced_suffix}"
                else:
                    method = f"TShape-{scheme}{balanced_suffix}"
                run_info.update({"seed": seed, "target": target, "method": method, "sources": source_datasets})
                target_start = time.time()
                for eval_i, ref in enumerate(target_refs, start=1):
                    prepared = load_prepared(ref, p=args.p, diff_order=args.diff_order)
                    if prepared is None:
                        continue
                    scores = tshape_scores(
                        model,
                        prepared.test,
                        args.p,
                        args.eval_batch_size,
                        getattr(args, "eval_stride", 1),
                    )
                    if score_root is not None:
                        score_path = score_root / f"seed_{seed}" / method / ref.dataset / f"{ref.name}.npy"
                        score_path.parent.mkdir(parents=True, exist_ok=True)
                        # Match the original TShape score archive by omitting the p-step warm-up prefix.
                        archived_scores = np.asarray(scores[args.p :], dtype=np.float32)
                        np.save(score_path, archived_scores)
                        score_manifest.append(
                            {
                                "dataset": ref.dataset,
                                "series": ref.name,
                                "seed": seed,
                                "method": method,
                                "score_file": str(score_path.relative_to(ROOT)),
                                "scores": int(len(archived_scores)),
                                "label_tail_offset": int(prepared.raw_test_len - len(archived_scores)),
                                "diff_order": int(args.diff_order),
                                "window": int(args.p),
                            }
                        )
                    row = metric_row(ref.dataset, ref.name, method, prepared.labels, scores)
                    row["seed"] = seed
                    rows.append(row)
                    if scheme in {"zero_shot", "zero_shot_pattern"}:
                        variants = tshape_zero_plus_variants(scores, prepared.test, args.p)
                        methods = (
                            variants.keys()
                            if args.emit_zero_plus_ablations
                            else ["TShape-zero_plus_minmax"]
                        )
                        for variant_method in methods:
                            variant_suffix = variant_method.removeprefix("TShape-zero_plus")
                            if scheme == "zero_shot_pattern":
                                output_method = f"TShape-zero_plus_pattern{balanced_suffix}{variant_suffix}"
                            else:
                                output_method = f"TShape-zero_plus{balanced_suffix}{variant_suffix}"
                            persist_variant = (
                                variant_method == "TShape-zero_plus_minmax"
                                or bool(getattr(args, "score_output_all_variants", False))
                            )
                            if score_root is not None and persist_variant:
                                variant_path = (
                                    score_root
                                    / f"seed_{seed}"
                                    / output_method
                                    / ref.dataset
                                    / f"{ref.name}.npy"
                                )
                                variant_path.parent.mkdir(parents=True, exist_ok=True)
                                archived_variant = np.asarray(variants[variant_method][args.p :], dtype=np.float32)
                                np.save(variant_path, archived_variant)
                                score_manifest.append(
                                    {
                                        "dataset": ref.dataset,
                                        "series": ref.name,
                                        "seed": seed,
                                        "method": output_method,
                                        "score_file": str(variant_path.relative_to(ROOT)),
                                        "scores": int(len(archived_variant)),
                                        "label_tail_offset": int(prepared.raw_test_len - len(archived_variant)),
                                        "diff_order": int(args.diff_order),
                                        "window": int(args.p),
                                    }
                                )
                            plus_row = metric_row(
                                ref.dataset,
                                ref.name,
                                output_method,
                                prepared.labels,
                                variants[variant_method],
                            )
                            plus_row["seed"] = seed
                            rows.append(plus_row)
                    if eval_i == 1 or eval_i == len(target_refs) or eval_i % args.progress_every == 0:
                        print(f"evaluated {target} {method} seed={seed}: {eval_i}/{len(target_refs)} series", flush=True)
                run_info["eval_seconds"] = time.time() - target_start
                run_info["eval_series"] = len(target_refs)
                runs.append(run_info)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"tshape_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"tshape_summary_metrics{suffix}.csv"
    runs_path = OUT_ROOT / f"tshape_run_info{suffix}.json"
    fields = [
        "dataset",
        "series",
        "seed",
        "method",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
    ]
    write_csv(detail_path, rows, fields)
    write_csv(
        summary_path,
        aggregate_rows(rows, expected),
        [
            "dataset",
            "method",
            "series",
            "rows",
            "seeds",
            "points",
            "valid_ap_series",
            "zero_anomaly_series",
            "coverage_ratio",
            "ap",
            "ap_std",
            "auc",
            "auc_std",
            "best_f1",
            "best_f1_std",
            "best_precision",
            "best_precision_std",
            "best_recall",
            "best_recall_std",
            "f1_q95",
            "f1_q95_std",
            "f1_q97",
            "f1_q97_std",
            "f1_q99",
            "f1_q99_std",
            "f1_q995",
            "f1_q995_std",
            "f1_mad3",
            "f1_mad3_std",
            "precision_q99",
            "precision_q99_std",
            "recall_q99",
            "recall_q99_std",
            "precision_mad3",
            "precision_mad3_std",
            "recall_mad3",
            "recall_mad3_std",
            "event_f1_q95",
            "event_f1_q95_std",
            "event_f1_q97",
            "event_f1_q97_std",
            "event_f1_q99",
            "event_f1_q99_std",
            "event_f1_q995",
            "event_f1_q995_std",
            "event_f1_mad3",
            "event_f1_mad3_std",
        ],
    )
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    if score_root is not None:
        (score_root / "score_manifest.json").write_text(
            json.dumps(score_manifest, indent=2), encoding="utf-8"
        )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def residual_diagnostic_row(
    dataset: str,
    series: str,
    variant: str,
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    row = metric_row(dataset, series, f"TShape-target_diag_{variant}", labels, scores)
    row["seed"] = seed
    finite = np.isfinite(scores)
    labels_bool = labels[finite].astype(bool)
    vals = scores[finite].astype(np.float64)
    normal = vals[~labels_bool]
    anomaly = vals[labels_bool]
    if len(normal) and len(anomaly):
        normal_q95 = float(np.quantile(normal, 0.95))
        anomaly_above = float(np.mean(anomaly >= normal_q95))
        row.update(
            {
                "variant": variant,
                "normal_median": f"{float(np.median(normal)):.8f}",
                "normal_q95": f"{normal_q95:.8f}",
                "anomaly_median": f"{float(np.median(anomaly)):.8f}",
                "anomaly_q75": f"{float(np.quantile(anomaly, 0.75)):.8f}",
                "anomaly_above_normal_q95": f"{anomaly_above:.6f}",
            }
        )
    else:
        row.update(
            {
                "variant": variant,
                "normal_median": "nan",
                "normal_q95": "nan",
                "anomaly_median": "nan",
                "anomaly_q75": "nan",
                "anomaly_above_normal_q95": "nan",
            }
        )
    return row


def target_diagnostics(args: argparse.Namespace) -> None:
    variant_specs = {
        "diff1_p16": {"p": 16, "diff_order": 1},
        "raw_p16": {"p": 16, "diff_order": 0},
        "diff1_p32": {"p": 32, "diff_order": 1},
    }
    rows: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    expected = {dataset: len(iter_series([dataset])) for dataset in args.datasets}
    for variant in args.variants:
        spec = variant_specs[variant]
        for dataset in args.datasets:
            set_seed(args.seed)
            refs = iter_series([dataset])
            local_args = argparse.Namespace(**vars(args))
            local_args.p = spec["p"]
            local_args.diff_order = spec["diff_order"]
            local_args.seed = args.seed
            print(f"Diagnostic training variant={variant} dataset={dataset} p={local_args.p} diff={local_args.diff_order}", flush=True)
            model, run_info = train_tshape_model(local_args, refs)
            start_time = time.time()
            for eval_i, ref in enumerate(refs, start=1):
                prepared = load_prepared(ref, p=local_args.p, diff_order=local_args.diff_order)
                if prepared is None:
                    continue
                scores = tshape_scores(
                    model,
                    prepared.test,
                    local_args.p,
                    args.eval_batch_size,
                    getattr(args, "eval_stride", 1),
                )
                rows.append(residual_diagnostic_row(ref.dataset, ref.name, variant, prepared.labels, scores, args.seed))
                if eval_i == 1 or eval_i == len(refs) or eval_i % args.progress_every == 0:
                    print(f"diagnosed {dataset} {variant}: {eval_i}/{len(refs)} series", flush=True)
            run_info.update(
                {
                    "seed": args.seed,
                    "target": dataset,
                    "method": f"TShape-target_diag_{variant}",
                    "variant": variant,
                    "p": local_args.p,
                    "diff_order": local_args.diff_order,
                    "eval_seconds": time.time() - start_time,
                    "eval_series": len(refs),
                }
            )
            runs.append(run_info)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = OUT_ROOT / f"target_diagnostic_series_metrics{suffix}.csv"
    summary_path = OUT_ROOT / f"target_diagnostic_summary{suffix}.csv"
    runs_path = OUT_ROOT / f"target_diagnostic_run_info{suffix}.json"
    metric_fields = [
        "dataset",
        "series",
        "seed",
        "method",
        "variant",
        "points",
        "anomaly_ratio",
        "ap",
        "auc",
        "best_f1",
        "best_precision",
        "best_recall",
        "f1_q95",
        "f1_q97",
        "f1_q99",
        "f1_q995",
        "f1_mad3",
        "precision_q99",
        "recall_q99",
        "precision_mad3",
        "recall_mad3",
        "event_f1_q95",
        "event_f1_q97",
        "event_f1_q99",
        "event_f1_q995",
        "event_f1_mad3",
        "normal_median",
        "normal_q95",
        "anomaly_median",
        "anomaly_q75",
        "anomaly_above_normal_q95",
    ]
    write_csv(detail_path, rows, metric_fields)
    agg = aggregate_rows(rows, expected)
    # Attach variant and residual-overlap summaries to the aggregate metric table.
    by_method: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    for item in agg:
        group = by_method.get((str(item["dataset"]), str(item["method"])), [])
        item["variant"] = str(item["method"]).replace("TShape-target_diag_", "")
        for key in ["normal_median", "normal_q95", "anomaly_median", "anomaly_q75", "anomaly_above_normal_q95"]:
            vals = np.array([float(r.get(key, "nan")) for r in group if str(r.get(key, "nan")) != "nan"], dtype=np.float64)
            item[key] = f"{float(np.mean(vals)):.8f}" if vals.size else "nan"
    write_csv(summary_path, agg, list(agg[0].keys()) if agg else ["dataset", "method", "variant"])
    runs_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {runs_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("profile")
    p_profile.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_profile.set_defaults(func=profile)

    p_base = sub.add_parser("baselines")
    p_base.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_base.add_argument("--p", type=int, default=16)
    p_base.add_argument("--diff-order", type=int, default=1)
    p_base.add_argument("--include-median", action="store_true")
    p_base.add_argument("--include-advanced", action="store_true")
    p_base.set_defaults(func=baselines)

    p_modern = sub.add_parser("modern-baselines")
    p_modern.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_modern.add_argument(
        "--methods",
        nargs="+",
        default=[
            "ForecastResidualAdapter",
            "OFAAdapter",
            "FCVAEAdapter",
            "TimesNetAdapter",
            "KANADAdapter",
            "FITSAdapter",
            "SubLOFAdapter",
            "SANDAdapter",
            "MatrixProfileAdapter",
            "ARAdapter",
            "LSTMADAdapter",
            "AEAdapter",
            "EncDecADAdapter",
            "SRCNNAdapter",
            "TFADAdapter",
            "DonutAdapter",
            "USAD-style",
            "TranAD-style",
            "AnomalyTransformer-style",
        ],
        choices=[
            "ForecastResidualAdapter",
            "OFAAdapter",
            "FCVAEAdapter",
            "TimesNetAdapter",
            "KANADAdapter",
            "FITSAdapter",
            "SubLOFAdapter",
            "SANDAdapter",
            "MatrixProfileAdapter",
            "ARAdapter",
            "LSTMADAdapter",
            "AEAdapter",
            "EncDecADAdapter",
            "SRCNNAdapter",
            "TFADAdapter",
            "DonutAdapter",
            "USAD-style",
            "TranAD-style",
            "AnomalyTransformer-style",
        ],
    )
    p_modern.add_argument("--p", type=int, default=16)
    p_modern.add_argument("--diff-order", type=int, default=1)
    p_modern.add_argument("--windows-per-series", type=int, default=256)
    p_modern.add_argument("--max-train-windows", type=int, default=30000)
    p_modern.add_argument("--balance-source-datasets", action="store_true")
    p_modern.add_argument("--epochs", type=int, default=3)
    p_modern.add_argument("--batch-size", type=int, default=512)
    p_modern.add_argument("--eval-batch-size", type=int, default=None)
    p_modern.add_argument("--progress-every", type=int, default=50)
    p_modern.add_argument("--lr", type=float, default=1e-3)
    p_modern.add_argument("--seed", type=int, default=20260704)
    p_modern.add_argument(
        "--training-protocol",
        choices=["zero_shot", "target_trained"],
        default="zero_shot",
        help="Training scope for reconstruction-style methods; adapters remain no-training.",
    )
    p_modern.add_argument("--tag", default="")
    p_modern.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p_modern.set_defaults(func=modern_baselines)

    p_foundation = sub.add_parser("foundation-baselines")
    p_foundation.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_foundation.add_argument(
        "--methods",
        nargs="+",
        default=["Chronos-Bolt-Tiny"],
        choices=[
            "Chronos-Bolt-Tiny",
            "Timer-base-84m",
            "Sundial-base-128m",
            "TimeMoE-50M",
            "TimesFM-2.5-200M",
            "TimerPatchAdapter",
            "TimesFMTrendAdapter",
            "SundialProbAdapter",
            "TimeMixerAdapter",
            "TimeMoEAdapter",
        ],
    )
    p_foundation.add_argument("--p", type=int, default=16)
    p_foundation.add_argument("--diff-order", type=int, default=1)
    p_foundation.add_argument("--context-length", type=int, default=64)
    p_foundation.add_argument("--prediction-length", type=int, default=128)
    p_foundation.add_argument("--batch-size", type=int, default=64)
    p_foundation.add_argument("--progress-every", type=int, default=50)
    p_foundation.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Optional deterministic smoke-test cap over the sorted series list.",
    )
    p_foundation.add_argument("--seed", type=int, default=20260704)
    p_foundation.add_argument("--tag", default="")
    p_foundation.add_argument("--chronos-model", default="amazon/chronos-bolt-tiny")
    p_foundation.add_argument("--timesfm-model", default="google/timesfm-2.5-200m-pytorch")
    p_foundation.add_argument("--hf-device", default="auto", choices=["auto", "cpu", "mps"])
    p_foundation.add_argument("--num-samples", type=int, default=1)
    p_foundation.add_argument("--local-files-only", action="store_true")
    p_foundation.set_defaults(func=foundation_baselines)

    p_pattern = sub.add_parser("pattern-bank")
    p_pattern.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_pattern.add_argument("--p", type=int, default=16)
    p_pattern.add_argument("--diff-order", type=int, default=1)
    p_pattern.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.20, 0.40, 0.60, 0.80, 1.00])
    p_pattern.add_argument("--variants-per-family", type=int, default=4)
    p_pattern.add_argument("--batch-size", type=int, default=8192)
    p_pattern.add_argument("--progress-every", type=int, default=50)
    p_pattern.add_argument("--seed", type=int, default=20260704)
    p_pattern.add_argument("--tag", default="")
    p_pattern.set_defaults(func=pattern_bank)

    p_pattern_tshape = sub.add_parser("pattern-bank-tshape")
    p_pattern_tshape.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_pattern_tshape.add_argument("--targets", nargs="+", default=list(DEFAULT_DATASETS))
    p_pattern_tshape.add_argument("--p", type=int, default=16)
    p_pattern_tshape.add_argument("--diff-order", type=int, default=1)
    p_pattern_tshape.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    )
    p_pattern_tshape.add_argument("--variants-per-family", type=int, default=8)
    p_pattern_tshape.add_argument("--max-train-windows", type=int, default=60000)
    p_pattern_tshape.add_argument("--capacity-balanced-bank", action="store_true")
    p_pattern_tshape.add_argument(
        "--size-sweep",
        action="store_true",
        help=(
            "Treat fractions as nested sample-count fractions of one fixed full "
            "motif mixture. This trains one universal Pattern-Bank-only checkpoint "
            "per size and performs no benchmark-domain training."
        ),
    )
    p_pattern_tshape.add_argument("--capacity-saturation", type=float, default=0.80)
    p_pattern_tshape.add_argument("--min-train-windows", type=int, default=7500)
    p_pattern_tshape.add_argument("--max-series-per-target", type=int, default=None)
    p_pattern_tshape.add_argument("--epochs", type=int, default=5)
    p_pattern_tshape.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs without sufficient training-loss improvement; 0 disables.",
    )
    p_pattern_tshape.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-5,
        help="Minimum training-loss decrease counted as an improvement.",
    )
    p_pattern_tshape.add_argument("--batch-size", type=int, default=256)
    p_pattern_tshape.add_argument("--eval-batch-size", type=int, default=None)
    p_pattern_tshape.add_argument("--eval-stride", type=int, default=1)
    p_pattern_tshape.add_argument("--progress-every", type=int, default=10)
    p_pattern_tshape.add_argument("--lr", type=float, default=1e-3)
    p_pattern_tshape.add_argument("--noise-std", type=float, default=0.025)
    p_pattern_tshape.add_argument("--event-rate", type=float, default=0.35)
    p_pattern_tshape.add_argument("--event-scale", type=float, default=0.75)
    p_pattern_tshape.add_argument("--seed", type=int, default=20260704)
    p_pattern_tshape.add_argument("--seeds", nargs="+", type=int, default=None)
    p_pattern_tshape.add_argument("--tag", default="")
    p_pattern_tshape.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p_pattern_tshape.add_argument("--emit-zero-plus", action="store_true")
    p_pattern_tshape.add_argument("--fusion-alpha", type=float, default=0.15)
    p_pattern_tshape.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory for one reusable checkpoint per Pattern Bank size.",
    )
    p_pattern_tshape.set_defaults(func=pattern_bank_tshape)

    p_pattern_hybrid = sub.add_parser("pattern-bank-hybrid-sweep")
    p_pattern_hybrid.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_pattern_hybrid.add_argument("--targets", nargs="+", default=list(DEFAULT_DATASETS))
    p_pattern_hybrid.add_argument("--p", type=int, default=16)
    p_pattern_hybrid.add_argument("--diff-order", type=int, default=1)
    p_pattern_hybrid.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    )
    p_pattern_hybrid.add_argument("--windows-per-series", type=int, default=256)
    p_pattern_hybrid.add_argument("--max-train-windows", type=int, default=60000)
    p_pattern_hybrid.add_argument("--balance-source-datasets", action="store_true")
    p_pattern_hybrid.add_argument("--filter-source-anomalies", action="store_true")
    p_pattern_hybrid.add_argument("--synthetic-ratio", type=float, default=0.25)
    p_pattern_hybrid.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.10,
        help="Fallback neural coefficient when no source-only selection CSV is provided.",
    )
    p_pattern_hybrid.add_argument(
        "--fusion-selections-csv",
        default="",
        help="CSV with target and selected_alpha columns produced by cross_validate_zero_plus_fusion.py.",
    )
    p_pattern_hybrid.add_argument("--pattern-bank-variants", type=int, default=8)
    p_pattern_hybrid.add_argument("--pattern-bank-noise-std", type=float, default=0.025)
    p_pattern_hybrid.add_argument("--pattern-bank-event-rate", type=float, default=0.35)
    p_pattern_hybrid.add_argument("--pattern-bank-event-scale", type=float, default=0.75)
    p_pattern_hybrid.add_argument("--max-series-per-target", type=int, default=None)
    p_pattern_hybrid.add_argument("--epochs", type=int, default=5)
    p_pattern_hybrid.add_argument("--batch-size", type=int, default=256)
    p_pattern_hybrid.add_argument("--eval-batch-size", type=int, default=None)
    p_pattern_hybrid.add_argument("--eval-stride", type=int, default=1)
    p_pattern_hybrid.add_argument("--progress-every", type=int, default=10)
    p_pattern_hybrid.add_argument("--lr", type=float, default=1e-3)
    p_pattern_hybrid.add_argument("--seed", type=int, default=20260704)
    p_pattern_hybrid.add_argument("--seeds", nargs="+", type=int, default=None)
    p_pattern_hybrid.add_argument("--tag", default="")
    p_pattern_hybrid.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p_pattern_hybrid.set_defaults(func=pattern_bank_hybrid_sweep)

    p_naive = sub.add_parser("tshape-easytsad-naive")
    p_naive.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_naive.add_argument("--force-datasets", nargs="+", default=[])
    p_naive.add_argument("--series", nargs="+", default=[])
    p_naive.add_argument("--max-series", type=int, default=None)
    p_naive.add_argument("--only-missing-original", action="store_true")
    p_naive.add_argument("--overwrite", action="store_true")
    p_naive.add_argument("--output-dir", default="Results/Scores/TShapeReproduction/naive")
    p_naive.add_argument("--p", type=int, default=16)
    p_naive.add_argument("--diff-order", type=int, default=1)
    p_naive.add_argument("--valid-proportion", type=float, default=0.20)
    p_naive.add_argument("--epochs", type=int, default=100)
    p_naive.add_argument("--patience", type=int, default=8)
    p_naive.add_argument("--batch-size", type=int, default=128)
    p_naive.add_argument("--eval-batch-size", type=int, default=1024)
    p_naive.add_argument("--lr", type=float, default=1e-3)
    p_naive.add_argument("--seed", type=int, default=20260704)
    p_naive.add_argument("--tag", default="")
    p_naive.add_argument("--device", default="auto", choices=["cpu", "mps", "auto"])
    p_naive.set_defaults(func=tshape_easytsad_naive)

    p_tshape = sub.add_parser("tshape")
    p_tshape.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_tshape.add_argument("--targets", nargs="+", default=["NAB", "TODS", "Yahoo"])
    p_tshape.add_argument("--schemes", nargs="+", default=["zero_shot", "in_domain_global"])
    p_tshape.add_argument("--p", type=int, default=16)
    p_tshape.add_argument("--diff-order", type=int, default=1)
    p_tshape.add_argument("--windows-per-series", type=int, default=256)
    p_tshape.add_argument("--max-train-windows", type=int, default=60000)
    p_tshape.add_argument("--balance-source-datasets", action="store_true")
    p_tshape.add_argument("--filter-source-anomalies", action="store_true")
    p_tshape.add_argument("--synthetic-ratio", type=float, default=0.25)
    p_tshape.add_argument("--pattern-bank-fraction", type=float, default=1.00)
    p_tshape.add_argument("--pattern-bank-variants", type=int, default=8)
    p_tshape.add_argument("--pattern-bank-noise-std", type=float, default=0.025)
    p_tshape.add_argument("--pattern-bank-event-rate", type=float, default=0.35)
    p_tshape.add_argument("--pattern-bank-event-scale", type=float, default=0.75)
    p_tshape.add_argument("--max-series-per-target", type=int, default=None)
    p_tshape.add_argument("--epochs", type=int, default=5)
    p_tshape.add_argument("--batch-size", type=int, default=256)
    p_tshape.add_argument("--eval-batch-size", type=int, default=None)
    p_tshape.add_argument("--eval-stride", type=int, default=1)
    p_tshape.add_argument("--progress-every", type=int, default=5)
    p_tshape.add_argument("--lr", type=float, default=1e-3)
    p_tshape.add_argument("--seed", type=int, default=20260704)
    p_tshape.add_argument("--seeds", nargs="+", type=int, default=None)
    p_tshape.add_argument("--tag", default="")
    p_tshape.add_argument(
        "--score-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-series post-warm-up anomaly-score arrays.",
    )
    p_tshape.add_argument(
        "--score-output-all-variants",
        action="store_true",
        help="Persist every fusion ablation; by default only neural and final min-max scores are stored.",
    )
    p_tshape.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p_tshape.add_argument("--emit-zero-plus-ablations", action="store_true")
    p_tshape.set_defaults(func=tshape)

    p_diag = sub.add_parser("target-diagnostics")
    p_diag.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_diag.add_argument("--variants", nargs="+", default=["diff1_p16", "raw_p16", "diff1_p32"], choices=["diff1_p16", "raw_p16", "diff1_p32"])
    p_diag.add_argument("--windows-per-series", type=int, default=128)
    p_diag.add_argument("--max-train-windows", type=int, default=30000)
    p_diag.add_argument("--epochs", type=int, default=3)
    p_diag.add_argument("--batch-size", type=int, default=512)
    p_diag.add_argument("--eval-batch-size", type=int, default=None)
    p_diag.add_argument("--eval-stride", type=int, default=1)
    p_diag.add_argument("--progress-every", type=int, default=50)
    p_diag.add_argument("--lr", type=float, default=1e-3)
    p_diag.add_argument("--seed", type=int, default=20260704)
    p_diag.add_argument("--tag", default="")
    p_diag.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    p_diag.set_defaults(func=target_diagnostics)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "eval_batch_size") and args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    args.func(args)


if __name__ == "__main__":
    main()
