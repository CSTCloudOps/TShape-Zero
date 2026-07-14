#!/usr/bin/env python3
"""Fail-fast verification for the strict TShape-Zero+ paper and artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "IEEE-conference-template"
RESULTS = ROOT / "Results" / "RENE"
DATASETS = {"AIOPS": 29, "NAB": 10, "TODS": 15, "UCR": 203, "WSD": 210, "Yahoo": 367}
KEYWORDS = (
    "time-series anomaly detection, replication, negative results, "
    "zero-shot learning, software reliability"
)
STRICT_METHODS = {"TShapeUniversalPattern-100", "TShapeUniversalZeroPlus-100"}
OFFICIAL_FROZEN = {
    "Chronos-Bolt-Tiny",
    "Timer-base-84m",
    "TimesFM-2.5-200M",
    "Sundial-base-128m",
}


def read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"missing or empty: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name}")
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def check_checkpoint(errors: list[str]) -> dict:
    path = ROOT / "models" / "tshape_zero_plus_release.pt"
    if not path.exists():
        errors.append("release checkpoint is missing")
        return {}
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        errors.append(f"release checkpoint cannot be loaded: {exc}")
        return {}
    metadata = payload.get("metadata", {})
    if "state_dict" not in payload:
        errors.append("release checkpoint has no state_dict")
    expected = {
        "strict_zero_shot": True,
        "training_corpus": "synthetic Pattern Bank only",
        "target_values_used_for_training": False,
        "target_labels_used_for_training": False,
        "pattern_bank_windows": 60000,
        "seed": 20260704,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"checkpoint metadata {key}={metadata.get(key)!r}; expected {value!r}")
    if not math.isclose(fnum(metadata.get("fusion_alpha")), 0.15, abs_tol=1e-12):
        errors.append("release checkpoint fusion_alpha is not 0.15")
    run = metadata.get("run_info", {})
    if int(run.get("early_stop_patience", -1)) != 3:
        errors.append("checkpoint does not record patience-three stopping")
    if int(run.get("real_training_windows", -1)) != 0:
        errors.append("checkpoint metadata reports benchmark training windows")
    return {
        "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024 * 1024),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": metadata,
    }


def check_main_ledger(errors: list[str]) -> dict:
    rows = read_csv(RESULTS / "submission_easytsad_all_methods.csv", errors)
    if not rows:
        return {}
    methods = {row["method_id"] for row in rows}
    datasets = {row["dataset"] for row in rows}
    if datasets != set(DATASETS):
        errors.append(f"main ledger datasets={sorted(datasets)}")
    if len(methods) < 44:
        errors.append(f"main ledger has only {len(methods)} methods")
    if len(rows) != len(methods) * len(DATASETS):
        errors.append(f"main ledger is not rectangular: {len(rows)} rows")
    for method in STRICT_METHODS | OFFICIAL_FROZEN:
        if method not in methods:
            errors.append(f"main ledger missing {method}")
    forbidden = {"ap", "auc", "auroc", "best_f1", "f1_q99"} & set(rows[0])
    if forbidden:
        errors.append(f"main ledger leaks non-paper metrics: {sorted(forbidden)}")
    for row in rows:
        if not math.isclose(fnum(row.get("coverage_ratio")), 1.0, abs_tol=1e-12):
            errors.append(f"incomplete coverage: {row['method_id']}/{row['dataset']}")
        if int(float(row.get("seeds", 0))) != 1:
            errors.append(f"non-single-seed row: {row['method_id']}/{row['dataset']}")
        for metric in ("point_f1", "event_f1"):
            value = fnum(row.get(metric))
            if not 0.0 <= value <= 1.0:
                errors.append(f"invalid {metric}: {row['method_id']}/{row['dataset']}")
    return {"methods": len(methods), "rows": len(rows), "coverage": "834/834", "seed": 20260704}


def check_pattern_sweep(errors: list[str]) -> dict:
    found: dict[int, dict] = {}
    for percent in range(10, 100, 10):
        summary = RESULTS / f"pattern_bank_tshape_summary_metrics_submission_strict_zero_size_{percent}.csv"
        run_info = RESULTS / f"pattern_bank_tshape_run_info_submission_strict_zero_size_{percent}.json"
        rows = read_csv(summary, errors)
        if rows:
            found[percent] = {"rows": len(rows), "run": run_info.exists()}
            methods = {row.get("method", "") for row in rows}
            expected = {f"TShapeUniversalPattern-{percent:03d}", f"TShapeUniversalZeroPlus-{percent:03d}"}
            if methods != expected:
                errors.append(f"Pattern {percent}% methods={sorted(methods)}")
            if {row.get("dataset") for row in rows} != set(DATASETS):
                errors.append(f"Pattern {percent}% does not cover six datasets")
            if any(not math.isclose(fnum(row.get("coverage_ratio")), 1.0, abs_tol=1e-12) for row in rows):
                errors.append(f"Pattern {percent}% is not full coverage")
        if not run_info.exists():
            errors.append(f"Pattern {percent}% run metadata missing")
    rows = read_csv(RESULTS / "pattern_bank_tshape_summary_metrics_submission_strict_zero_main_full.csv", errors)
    if rows:
        found[100] = {"rows": len(rows), "run": True}
    if set(found) != set(range(10, 101, 10)):
        errors.append(f"Pattern sweep incomplete: {sorted(found)}")
    return {"fractions": sorted(found), "full_series_each": len(found) == 10}


def check_ablation(errors: list[str]) -> dict:
    rows = read_csv(
        RESULTS / "strict_zero_ablation_summary_metrics_submission_strict_zero_ablation.csv",
        errors,
    )
    if not rows:
        return {}
    methods = {row.get("method", "") for row in rows}
    expected = {
        "StrictGuardOnly", "StrictTShapeMedian", "StrictTShapeSpectral",
        "StrictZeroPlusRank015", "StrictZeroPlusAlpha005", "StrictZeroPlusAlpha010",
        "StrictZeroPlusAlpha025", "StrictZeroPlusAlpha050",
    }
    if methods != expected:
        errors.append(f"strict ablation methods={sorted(methods)}")
    if {row.get("dataset") for row in rows} != set(DATASETS):
        errors.append("strict ablation does not cover six datasets")
    if any(not math.isclose(fnum(row.get("coverage_ratio")), 1.0, abs_tol=1e-12) for row in rows):
        errors.append("strict ablation is not full coverage")
    return {"methods": len(methods), "rows": len(rows)}


def check_submission_exports(errors: list[str]) -> dict:
    expected = {
        "submission_strict_pattern_sweep_easytsad.csv": 120,
        "submission_strict_pattern_series_easytsad.csv": 16680,
        "submission_strict_pattern_training.csv": 1,
        "submission_strict_pattern_manifest.csv": 10,
        "submission_strict_guard_ablation_easytsad.csv": 48,
        "submission_strict_channel_attribution_easytsad.csv": 768,
        "submission_strict_channel_attribution_summary_easytsad.csv": 6,
    }
    counts: dict[str, int] = {}
    for name, minimum in expected.items():
        rows = read_csv(RESULTS / name, errors)
        counts[name] = len(rows)
        if len(rows) < minimum:
            errors.append(f"compact submission export {name} has {len(rows)} rows; expected at least {minimum}")
        if rows and {"ap", "auc", "auroc", "best_f1", "f1_q99"} & set(rows[0]):
            errors.append(f"compact submission export leaks non-paper metrics: {name}")
    return counts


def check_product(errors: list[str], checkpoint: dict) -> dict:
    case_path = RESULTS / "product_demo_case_scores.json"
    if not case_path.exists():
        errors.append("product case is missing")
        return {}
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case = case_payload.get("case", {})
    if not case.get("checkpoint_strict_zero_shot"):
        errors.append("product case was not scored by the strict checkpoint")
    if case.get("checkpoint_training_corpus") != "synthetic Pattern Bank only":
        errors.append("product case checkpoint provenance is wrong")
    if int(case.get("anomaly_events", 0)) < 4:
        errors.append("product case is not multi-event")
    for metric in ("local_point_f1_pa", "local_event_f1_pa_log"):
        if fnum(case.get(metric)) < 0.90:
            errors.append(f"product case {metric} is below 0.90")
    try:
        sys.path.insert(0, str(ROOT / "experiments"))
        import numpy as np
        from tshape_zero_product import score_array

        values = np.sin(np.linspace(0, 8 * np.pi, 96)).astype(np.float32)
        values[48:51] += 3.5
        result = score_array(values, ROOT / "models" / "tshape_zero_plus_release.pt", device="cpu")
        if len(result.get("scores", [])) != len(values):
            errors.append("product score alignment failed")
        if len(result.get("top_anomalies", [])) != 10:
            errors.append("product top-k contract failed")
        if checkpoint and not math.isclose(
            fnum(result.get("fusion_alpha")), fnum(checkpoint["metadata"].get("fusion_alpha")), abs_tol=1e-12
        ):
            errors.append("product ignored checkpoint fusion alpha")
    except Exception as exc:
        errors.append(f"product smoke test failed: {exc}")
    return {
        "points": len(case_payload.get("input_values", [])),
        "events": case.get("anomaly_events"),
        "point_f1": case.get("local_point_f1_pa"),
        "event_f1": case.get("local_event_f1_pa_log"),
    }


def check_paper(errors: list[str]) -> dict:
    files = [PAPER / "main.tex", PAPER / "sections" / "evaluation_strict.tex", PAPER / "sections" / "deployment_strict.tex"]
    for path in files:
        if not path.exists():
            errors.append(f"paper source missing: {path.name}")
    text = "\n".join(path.read_text(encoding="utf-8") for path in files if path.exists())
    lower = text.lower()
    for token in (
        "todo", "placeholder", "visual draft", "average precision", "auroc", "event-f1@q99",
        "leave-one-dataset-out", "target-excluded", "three-seed", "/users/", "results/rene/",
    ):
        if token in lower:
            errors.append(f"paper contains stale/forbidden token: {token}")
    if KEYWORDS not in text:
        errors.append("paper keywords changed")
    for marker in ("Challenge 1", "Challenge 2", "Challenge 3", "RQ1", "RQ2", "RQ3", "RQ4"):
        if marker not in text:
            errors.append(f"paper missing {marker}")
    required_figures = (
        "fig_intro_strict.pdf", "fig_motivation_strict.pdf", "fig_framework_strict.pdf",
        "fig_tshape_architecture_strict.pdf", "fig03_easytsad_protocol_radar.pdf",
        "fig_strict_overall.pdf", "fig_strict_pattern_training.pdf", "fig_strict_pattern_effect.pdf",
        "fig_strict_guard_violin.pdf", "fig_strict_guard_bars.pdf", "fig_strict_guard_ablation.pdf",
        "fig_strict_channel_attribution.pdf",
        "fig_strict_effectiveness_bubble.pdf", "fig_strict_attention.pdf", "fig_product_strict_panel.png",
    )
    for name in required_figures:
        path = PAPER / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"paper figure missing: {name}")
    pdf = PAPER / "main.pdf"
    pages = None
    if pdf.exists():
        try:
            from pypdf import PdfReader

            pages = len(PdfReader(str(pdf)).pages)
        except Exception as exc:
            errors.append(f"cannot inspect paper PDF: {exc}")
    else:
        errors.append("paper PDF is missing")
    return {"pdf_pages": pages}


def check_protocol(errors: list[str]) -> dict:
    command = [sys.executable, str(ROOT / "experiments" / "verify_easytsad_protocol.py")]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        errors.append("EasyTSAD protocol verifier failed")
    return {"returncode": completed.returncode, "tail": completed.stdout.strip().splitlines()[-3:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    checkpoint = check_checkpoint(errors)
    report = {
        "checkpoint": checkpoint,
        "ledger": check_main_ledger(errors),
        "pattern_sweep": check_pattern_sweep(errors),
        "ablation": check_ablation(errors),
        "submission_exports": check_submission_exports(errors),
        "product": check_product(errors, checkpoint),
        "protocol": check_protocol(errors),
        "paper": check_paper(errors),
    }
    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
