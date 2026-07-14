#!/usr/bin/env python3
"""Build the paper's protocol-consistent EasyTSAD result ledger and statistics."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results" / "RENE"
DATASETS = ("AIOPS", "NAB", "TODS", "UCR", "WSD", "Yahoo")
METRICS = ("point_f1_pa", "event_f1_pa_log")
PRIMARY_SEED = 20260704
LODO_METHOD = "TShape-zero_shot_balanced"
PATTERN_METHOD = "TShapeUniversalPattern-100"
FIXED_FUSION_METHOD = "TShape-zero_plus_pattern_balanced_minmax"
PROPOSED_METHOD = "TShapeUniversalZeroPlus-100"
GUARD_METHOD = "TShape-zero_plus_pattern_balanced_residual_only_minmax"


METHODS = {
    "persistence": ("Residual", "Persistence", "no-training control"),
    "rolling_mean": ("Residual", "Rolling Mean", "no-training control"),
    "rolling_median": ("Residual", "Rolling Median", "no-training control"),
    "spectral_residual": ("Residual", "Spectral Residual", "no-training control"),
    "rolling_mad": ("Residual", "Rolling MAD", "no-training control"),
    "Chronos-Bolt-Tiny": ("Foundation", "Chronos-Bolt-Tiny", "official checkpoint"),
    "Timer-base-84m": ("Foundation", "Timer-base-84M", "official checkpoint"),
    "TimesFM-2.5-200M": ("Foundation", "TimesFM-2.5-200M", "official checkpoint"),
    "Sundial-base-128m": ("Foundation", "Sundial-base-128M", "official checkpoint"),
    "TimerPatchAdapter": ("Foundation", "TimerPatch", "family adapter"),
    "TimesFMTrendAdapter": ("Foundation", "TimesFMTrend", "family adapter"),
    "SundialProbAdapter": ("Foundation", "SundialProb", "family adapter"),
    "TimeMixerAdapter": ("Foundation", "TimeMixer", "family adapter"),
    "TimeMoEAdapter": ("Foundation", "TimeMoE", "family adapter"),
    "ForecastResidualAdapter": ("TSAD", "ForecastResidual", "family adapter"),
    "OFAAdapter": ("TSAD", "OFA", "family adapter"),
    "FCVAEAdapter": ("TSAD", "FCVAE", "family adapter"),
    "TimesNetAdapter": ("TSAD", "TimesNet", "family adapter"),
    "KANADAdapter": ("TSAD", "KANAD", "family adapter"),
    "FITSAdapter": ("TSAD", "FITS", "family adapter"),
    "SubLOFAdapter": ("TSAD", "SubLOF", "family adapter"),
    "SANDAdapter": ("TSAD", "SAND", "family adapter"),
    "MatrixProfileAdapter": ("TSAD", "Matrix Profile", "family adapter"),
    "ARAdapter": ("TSAD", "AR", "family adapter"),
    "LSTMADAdapter": ("TSAD", "LSTMAD", "family adapter"),
    "AEAdapter": ("TSAD", "AE", "family adapter"),
    "EncDecADAdapter": ("TSAD", "EncDecAD", "family adapter"),
    "SRCNNAdapter": ("TSAD", "SRCNN", "family adapter"),
    "TFADAdapter": ("TSAD", "TFAD", "family adapter"),
    "DonutAdapter": ("TSAD", "Donut", "family adapter"),
    "USAD-style": ("TSAD", "USAD", "source-pretrained style implementation"),
    "TranAD-style": ("TSAD", "TranAD", "source-pretrained style implementation"),
    "AnomalyTransformer-style": (
        "TSAD",
        "Anomaly Transformer",
        "source-pretrained style implementation",
    ),
    PATTERN_METHOD: (
        "TShape",
        "Pattern-only TShape-Zero",
        "strict Pattern-Bank checkpoint",
    ),
    PROPOSED_METHOD: (
        "TShape",
        "TShape-Zero+",
        "proposed strict checkpoint",
    ),
    LODO_METHOD: (
        "TShape",
        "LODO TShape adaptation",
        "source-adaptation diagnostic",
    ),
    "TShape-zero_pattern_balanced": ("TShape", "PB Hybrid", "source+Pattern-Bank pretraining"),
    "TShape-zero-pattern-sourcecv": (
        "TShape",
        "PB Source-CV",
        "source-adaptation diagnostic",
    ),
    FIXED_FUSION_METHOD: (
        "TShape",
        "Fixed 0.15 Fusion",
        "ablation",
    ),
    "TShape-zero-plus-pattern-cv": (
        "TShape",
        "PB100+Guard",
        "100%-bank ablation",
    ),
    "TShape-zero-plus-sourcecv": (
        "TShape",
        "LODO Zero+ diagnostic",
        "source-adaptation diagnostic",
    ),
    "TShape-target_trained": (
        "TShape",
        "Target-Context Diagnostic",
        "target-context diagnostic",
    ),
    GUARD_METHOD: (
        "TShape",
        "Residual-Only Guard",
        "ablation",
    ),
    "TShape-zero_plus_balanced_minmax": ("TShape", "Direct+Guard", "ablation"),
}


SUMMARY_FILES = (
    "baseline_summary_metrics.csv",
    "foundation_baseline_summary_metrics_submission_easytsad_chronos.csv",
    "foundation_baseline_summary_metrics_submission_easytsad_timer.csv",
    "foundation_baseline_summary_metrics_submission_easytsad_timesfm.csv",
    "foundation_baseline_summary_metrics_submission_easytsad_sundial.csv",
    "foundation_baseline_summary_metrics_submission_easytsad_foundation_adapters.csv",
    "modern_baseline_summary_metrics_submission_easytsad_modern_zero.csv",
    "modern_baseline_summary_metrics_submission_easytsad_modern_zero_balanced.csv",
    "modern_baseline_summary_metrics_submission_easytsad_usad_balanced_best.csv",
    "modern_baseline_summary_metrics_submission_easytsad_legacy_adapters.csv",
    "easytsad_tshape_zero_replay_summary.csv",
    "tshape_summary_metrics_submission_easytsad_tshape_full3.csv",
    "tshape_summary_metrics_submission_easytsad_tshape_balanced_full3.csv",
    "tshape_summary_metrics_submission_easytsad_tshape_pattern_v2_full3.csv",
    "pattern_v2_zero_plus_fusion_cv_summary.csv",
    "pattern_bank_sourcecv_summary.csv",
    "pattern_bank_tshape_summary_metrics_submission_strict_zero_main_full.csv",
)

DETAIL_FILES = (
    "baseline_series_metrics.csv",
    "foundation_baseline_series_metrics_submission_easytsad_chronos.csv",
    "foundation_baseline_series_metrics_submission_easytsad_timer.csv",
    "foundation_baseline_series_metrics_submission_easytsad_timesfm.csv",
    "foundation_baseline_series_metrics_submission_easytsad_sundial.csv",
    "foundation_baseline_series_metrics_submission_easytsad_foundation_adapters.csv",
    "modern_baseline_series_metrics_submission_easytsad_modern_zero.csv",
    "modern_baseline_series_metrics_submission_easytsad_modern_zero_balanced.csv",
    "modern_baseline_series_metrics_submission_easytsad_usad_balanced_best.csv",
    "modern_baseline_series_metrics_submission_easytsad_legacy_adapters.csv",
    "tshape_series_metrics_submission_easytsad_tshape_full3.csv",
    "tshape_series_metrics_submission_easytsad_tshape_balanced_full3.csv",
    "tshape_series_metrics_submission_easytsad_tshape_pattern_v2_full3.csv",
    "pattern_v2_zero_plus_fusion_cv_series.csv",
    "pattern_bank_sourcecv_series.csv",
    "easytsad_tshape_zero_replay_series.csv",
    "pattern_bank_tshape_series_metrics_submission_strict_zero_main_full.csv",
)


def access_class(status: str) -> str:
    """Return the benchmark-access contract used by the paper table."""
    if status == "no-training control":
        return "No-fit"
    if status == "official checkpoint":
        return "Frozen"
    if status in {"strict Pattern-Bank checkpoint", "proposed strict checkpoint"}:
        return "Strict"
    if status in {
        "source-pretrained style implementation",
        "source-pretrained",
        "source+Pattern-Bank pretraining",
        "source-adaptation diagnostic",
        "source-selected Pattern-Bank coverage",
    }:
        return "Adapt"
    if status == "family adapter":
        return "Adapter"
    if status in {"target-context diagnostic", "shared-harness target-context diagnostic"}:
        return "Target"
    return "Ablation"


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        sample = rng.choice(values, size=(count, len(values)), replace=True)
        means[start : start + count] = np.mean(sample, axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def single_seed_detail_summary(
    detail: Sequence[Dict[str, str]], method: str, dataset: str
) -> Dict[str, object]:
    candidates = [
        row
        for row in detail
        if row.get("method") == method and row.get("dataset") == dataset
    ]
    primary = [
        row
        for row in candidates
        if str(row.get("seed", "")).strip()
        and int(float(row["seed"])) == PRIMARY_SEED
    ]
    if primary:
        candidates = primary
    by_series: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_series[str(row["series"])].append(row)

    out: Dict[str, object] = {"series": len(by_series), "seeds": 1}
    for metric in METRICS:
        values = []
        for rows in by_series.values():
            series_values = np.asarray(
                [fnum(row.get(metric)) for row in rows], dtype=np.float64
            )
            series_values = series_values[np.isfinite(series_values)]
            if series_values.size:
                values.append(float(np.mean(series_values)))
        array = np.asarray(values, dtype=np.float64)
        out[metric] = float(np.mean(array)) if array.size else float("nan")
        out[f"{metric}_std"] = (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        )
        out[f"valid_{'point_pa' if metric == 'point_f1_pa' else 'event_pa'}_series"] = int(
            array.size
        )
    return out


def load_summary_ledger(detail: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    selected: Dict[Tuple[str, str], Dict[str, str]] = {}
    for filename in SUMMARY_FILES:
        for row in read_rows(RESULTS / filename):
            method = row.get("method", "")
            if method in METHODS and row.get("dataset") in DATASETS:
                selected[(row["dataset"], method)] = row
    missing = [(dataset, method) for method in METHODS for dataset in DATASETS if (dataset, method) not in selected]
    if missing:
        preview = ", ".join(f"{dataset}/{method}" for dataset, method in missing[:12])
        raise RuntimeError(f"Missing {len(missing)} main-table cells: {preview}")

    ledger: List[Dict[str, object]] = []
    for method, (family, display, status) in METHODS.items():
        for dataset in DATASETS:
            row = selected[(dataset, method)]
            seed_summary = single_seed_detail_summary(detail, method, dataset)
            if seed_summary["series"]:
                row = dict(row)
                row.update(seed_summary)
            ledger.append(
                {
                    "family": family,
                    "method": display,
                    "method_id": method,
                    "status": status,
                    "access": access_class(status),
                    "dataset": dataset,
                    "point_f1": f"{fnum(row['point_f1_pa']):.6f}",
                    "point_f1_std": f"{fnum(row.get('point_f1_pa_std')):.6f}",
                    "event_f1": f"{fnum(row['event_f1_pa_log']):.6f}",
                    "event_f1_std": f"{fnum(row.get('event_f1_pa_log_std')):.6f}",
                    "series": row.get("series", ""),
                    "valid_point_series": row.get("valid_point_pa_series", ""),
                    "valid_event_series": row.get("valid_event_pa_series", ""),
                    "coverage_ratio": row.get("coverage_ratio", ""),
                    "seeds": row.get("seeds", "1"),
                }
            )
    return ledger


def load_detail() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for filename in DETAIL_FILES:
        loaded = read_rows(RESULTS / filename)
        if filename == "easytsad_tshape_zero_replay_series.csv":
            # Archived replay rows use the original post-warm-up alignment and
            # belong only to the provenance figure, not the unified main table.
            loaded = []
        elif filename == "tshape_series_metrics_submission_easytsad_tshape_balanced_full3.csv":
            # Pattern Bank v2 is the final three-seed run. Keep source-only
            # direct TShape rows from this older file, but never average v1 and
            # v2 Pattern rows into an accidental six-seed result.
            loaded = [row for row in loaded if "pattern" not in row.get("method", "")]
        elif filename == "modern_baseline_series_metrics_submission_easytsad_modern_zero.csv":
            # The equal-source rerun below supersedes the earlier natural-mixture
            # versions of the three learned reconstruction baselines.
            loaded = [
                row
                for row in loaded
                if row.get("method") not in {"USAD-style", "TranAD-style", "AnomalyTransformer-style"}
            ]
        elif filename == "modern_baseline_series_metrics_submission_easytsad_modern_zero_balanced.csv":
            # USAD is superseded once more by the best-source-state rerun.
            loaded = [row for row in loaded if row.get("method") != "USAD-style"]
        rows.extend(loaded)
    return [row for row in rows if row.get("method") in METHODS]


def per_series_values(rows: Iterable[Dict[str, str]], method: str, dataset: str, metric: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("method") != method or row.get("dataset") != dataset:
            continue
        seed = str(row.get("seed", "")).strip()
        if seed and int(float(seed)) != PRIMARY_SEED:
            continue
        value = fnum(row.get(metric))
        if np.isfinite(value):
            grouped[row["series"]].append(value)
    return {series: float(np.mean(values)) for series, values in grouped.items()}


def paired_statistics(ledger: List[Dict[str, object]], detail: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in ledger:
        by_dataset[str(row["dataset"])].append(row)
    out: List[Dict[str, object]] = []
    allowed_external = {"no-training control", "official checkpoint"}
    for dataset_i, dataset in enumerate(DATASETS):
        for metric_i, (metric, paper_metric) in enumerate(
            (("point_f1_pa", "Point-F1"), ("event_f1_pa_log", "Event-F1"))
        ):
            field = "point_f1" if metric == "point_f1_pa" else "event_f1"
            external = [row for row in by_dataset[dataset] if row["status"] in allowed_external]
            best_external = max(external, key=lambda row: fnum(row[field]))
            comparisons = [
                ("Pattern-only TShape-Zero", PATTERN_METHOD),
                (
                    "Residual-Only Guard",
                    GUARD_METHOD,
                ),
                (str(best_external["method"]), str(best_external["method_id"])),
            ]
            for comparator_name, comparator_id in comparisons:
                proposed = per_series_values(
                    detail,
                    PROPOSED_METHOD,
                    dataset,
                    metric,
                )
                comparator = per_series_values(detail, comparator_id, dataset, metric)
                keys = sorted(set(proposed) & set(comparator))
                diffs = np.array([proposed[key] - comparator[key] for key in keys], dtype=np.float64)
                lo, hi = bootstrap_mean_ci(diffs, 20260714 + 101 * dataset_i + 17 * metric_i)
                non_ties = diffs[np.abs(diffs) > 1e-12]
                wins = int(np.sum(non_ties > 0))
                losses = int(np.sum(non_ties < 0))
                pvalue = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
                out.append(
                    {
                        "dataset": dataset,
                        "metric": paper_metric,
                        "comparator": comparator_name,
                        "series": len(keys),
                        "mean_difference": f"{float(np.mean(diffs)):.6f}",
                        "ci_low": f"{lo:.6f}",
                        "ci_high": f"{hi:.6f}",
                        "wins": wins,
                        "losses": losses,
                        "ties": int(len(diffs) - wins - losses),
                        "sign_test_p": f"{pvalue:.6g}",
                    }
                )
    return out


def main() -> None:
    detail = load_detail()
    ledger = load_summary_ledger(detail)
    stats = paired_statistics(ledger, detail)
    write_rows(
        RESULTS / "submission_easytsad_all_methods.csv",
        ledger,
        [
            "family",
            "method",
            "method_id",
            "status",
            "access",
            "dataset",
            "point_f1",
            "point_f1_std",
            "event_f1",
            "event_f1_std",
            "series",
            "valid_point_series",
            "valid_event_series",
            "coverage_ratio",
            "seeds",
        ],
    )
    write_rows(
        RESULTS / "submission_easytsad_paired_statistics.csv",
        stats,
        [
            "dataset",
            "metric",
            "comparator",
            "series",
            "mean_difference",
            "ci_low",
            "ci_high",
            "wins",
            "losses",
            "ties",
            "sign_test_p",
        ],
    )
    print(f"Wrote {RESULTS / 'submission_easytsad_all_methods.csv'}")
    print(f"Wrote {RESULTS / 'submission_easytsad_paired_statistics.csv'}")


if __name__ == "__main__":
    main()
