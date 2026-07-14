#!/usr/bin/env python3
"""Package the evaluated synthetic Pattern checkpoint as the product artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "models" / "strict_pattern_main" / "pattern_bank_100_seed_20260704.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "tshape_zero_plus_release.pt",
    )
    args = parser.parse_args()

    import torch

    payload = torch.load(args.source, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    if not metadata.get("strict_zero_shot"):
        raise ValueError("Source checkpoint is not marked strict_zero_shot.")
    if metadata.get("training_corpus") != "synthetic Pattern Bank only":
        raise ValueError("Source checkpoint was not trained only on the synthetic Pattern Bank.")
    if metadata.get("target_values_used_for_training") or metadata.get("target_labels_used_for_training"):
        raise ValueError("Source checkpoint metadata reports benchmark training access.")

    alpha = float(metadata.get("fusion_alpha", 0.15))
    if abs(alpha - 0.15) > 1e-12:
        raise ValueError(f"Expected fusion_alpha=0.15, found {alpha}")
    metadata.update(
        {
            "name": "TShape-Zero+ strict universal checkpoint",
            "checkpoint_role": "released out-of-the-box checkpoint",
            "output_alignment": "prepend a minimum-score warm-up value per differencing order",
            "residual_guard_formula": (
                "(0.55 minmax(rolling-median) + 0.25 minmax(spectral-residual) + "
                "0.05 minmax(rolling-MAD)) / 0.85"
            ),
            "score_formula": "0.15 minmax(Pattern-TShape) + 0.85 residual_guard",
            "calibration": "label-free score ranking; caller chooses an alarm policy or top-k",
            "repository": "https://github.com/CSTCloudOps/TShape-Zero",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state_dict": payload["state_dict"]}, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "artifact": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "strict_zero_shot": True,
        "training_corpus": metadata["training_corpus"],
        "seed": metadata.get("seed"),
        "pattern_bank_windows": metadata.get("pattern_bank_windows"),
        "fusion_alpha": alpha,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
