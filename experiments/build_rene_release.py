#!/usr/bin/env python3
"""Build a review-ready TShape-Zero+ RENE archive with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "IEEE-conference-template"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sanitized(value):
    """Remove machine-local absolute paths from copied JSON metadata."""
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), "<artifact-root>").replace(str(PAPER), "<paper-root>")
    return value


def copy_result(source: Path, destination: Path) -> None:
    if source.suffix.lower() in {".csv", ".txt"}:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            source.read_text(encoding="utf-8")
            .replace(str(ROOT), "<artifact-root>")
            .replace(str(PAPER), "<paper-root>"),
            encoding="utf-8",
        )
        return
    if source.suffix.lower() != ".json":
        copy_file(source, destination)
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(sanitized(payload), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release")
    args = parser.parse_args()
    package = args.output_dir / "TShape-Zero-RENE"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    core_files = [
        "README.md",
        "RENE_ARTIFACT_README.md",
        "MODEL_CARD.md",
        "FAILURE_CARD.md",
        "CITATION.cff",
        "LICENSE",
        "Makefile",
        "requirements_rene.txt",
        "requirements_foundation.txt",
        "layyer.py",
        "models/tshape_zero_plus_release.pt",
        "models/tshape_zero_plus_release.manifest.json",
        "experiments/README_RENE.md",
        "experiments/rene_experiments.py",
        "experiments/build_tshape_faithful_replay.py",
        "experiments/build_easytsad_submission_results.py",
        "experiments/make_easytsad_submission_assets.py",
        "experiments/make_strict_zero_submission_assets.py",
        "experiments/export_strict_submission_results.py",
        "experiments/strict_zero_ablation.py",
        "experiments/verify_easytsad_protocol.py",
        "experiments/plot_tshape_dual_radar.py",
        "experiments/tshape_zero_product.py",
        "experiments/package_strict_checkpoint.py",
        "experiments/build_product_demo_case.py",
        "experiments/verify_strict_release.py",
        "experiments/build_rene_release.py",
    ]
    for relative in core_files:
        source = ROOT / relative
        if not source.exists() and relative.startswith("IEEE-conference-template/"):
            source = ROOT.parent / relative
        copy_file(source, package / relative)
    copy_file(ROOT / "RENE_ARTIFACT_README.md", package / "ARTIFACT_GUIDE.md")

    result_dir = ROOT / "Results" / "RENE"
    result_names = {
        "data_profile.csv",
        "baseline_series_metrics.csv",
        "baseline_summary_metrics.csv",
        "dataset_manifest.csv",
        "original_reported_results.csv",
        "product_demo_case_scores.json",
        "strict_release_verification.json",
        "tshape_faithful_reproduction_series_metrics.csv",
        "tshape_faithful_reproduction_summary_metrics.csv",
    }
    result_names.update(path.name for path in result_dir.glob("*easytsad*.csv"))
    result_names.update(path.name for path in result_dir.glob("*easytsad*.json"))
    result_names.update(path.name for path in result_dir.glob("pattern_v2_*.csv"))
    result_names.update(path.name for path in result_dir.glob("pattern_bank_sourcecv*"))
    result_names.update(path.name for path in result_dir.glob("pattern_bank_tshape_*submission_strict_zero*.csv"))
    result_names.update(path.name for path in result_dir.glob("pattern_bank_tshape_*submission_strict_zero*.json"))
    result_names.update(path.name for path in result_dir.glob("strict_zero_ablation_*submission_strict_zero_ablation.*"))
    result_names.update(path.name for path in result_dir.glob("submission_strict_*.csv"))
    result_names.update(path.name for path in result_dir.glob("submission_efficiency_frontier.csv"))
    result_names = {
        name
        for name in result_names
        if "smoke" not in name and "extra_seeds" not in name
    }
    for name in sorted(result_names):
        source = result_dir / name
        if source.exists():
            copy_result(source, package / "Results" / "RENE" / name)

    for name in ["main.tex", "references.bib", "IEEEtran.cls", "main.pdf"]:
        if (PAPER / name).exists():
            copy_file(PAPER / name, package / "paper" / "IEEE" / name)
    for name in ("evaluation_strict.tex", "deployment_strict.tex"):
        copy_file(PAPER / "sections" / name, package / "paper" / "IEEE" / "sections" / name)

    # Ship only camera-ready evidence. The workspace intentionally retains
    # exploratory figures, but stale AP-era plots must not leak into the RENE
    # archive and be mistaken for current protocol evidence.
    figure_stems = (
        "fig_intro_strict",
        "fig_tshape_architecture_strict",
        "fig_motivation_strict",
        "fig03_easytsad_protocol_radar",
        "fig_framework_strict",
        "fig_strict_overall",
        "fig_strict_pattern_training",
        "fig_strict_pattern_effect",
        "fig_strict_guard_violin",
        "fig_strict_guard_bars",
        "fig_strict_guard_ablation",
        "fig_strict_channel_attribution",
        "fig_strict_effectiveness_bubble",
        "fig_strict_attention",
        "fig_product_strict_panel",
    )
    for stem in figure_stems:
        copied = False
        for suffix in (".pdf", ".svg", ".png"):
            source = PAPER / "figures" / f"{stem}{suffix}"
            if source.exists():
                copy_file(source, package / "paper" / "IEEE" / "figures" / source.name)
                copied = True
        if not copied:
            raise FileNotFoundError(PAPER / "figures" / stem)

    for name in (
        "easytsad_numbers.tex",
        "easytsad_reproduction_table.tex",
        "easytsad_main_table.tex",
        "easytsad_paired_table.tex",
    ):
        copy_file(PAPER / "generated" / name, package / "paper" / "IEEE" / "generated" / name)

    entries = [path for path in sorted(package.rglob("*")) if path.is_file()]
    checksum_lines = [f"{sha256(path)}  {path.relative_to(package).as_posix()}" for path in entries]
    (package / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive = args.output_dir / "TShape-Zero-RENE.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zf.write(path, Path(package.name) / path.relative_to(package))
    print(f"Wrote {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
