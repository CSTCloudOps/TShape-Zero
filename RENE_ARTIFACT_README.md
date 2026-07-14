# TShape-Zero+ RENE Artifact

This artifact supports the paper:

**TShape-Zero+: An Out-of-the-Box TShape for Zero-Shot KPI Anomaly Detection**

The contribution is an artifact-reuse stress test plus a constructive repair.
It does not claim that the original TShape paper promised zero-shot transfer.
It preserves the original target-trained evidence and evaluates a stronger
product contract: one frozen model must score an unseen KPI without benchmark
weight fitting.

## Strict Model Contract

The released 0.138-MiB checkpoint is trained only on a generated Pattern Bank:

- 20 temporal generators and eight randomized variants;
- 60,000 generated continuation windows;
- context-only spike, dip, level, variance, drift, and burst incidents;
- one seed, 20260704;
- at most 30 epochs with patience-three stopping;
- zero benchmark values and zero benchmark labels used for weight training.

At scoring time, TShape-Zero+ returns the Pattern-TShape residual and the
Residual Guard separately. The fixed score is:

~~~text
Zero+ = 0.15 * minmax(Pattern-TShape)
      + 0.55 * minmax(rolling-median)
      + 0.25 * minmax(spectral-residual)
      + 0.05 * minmax(rolling-MAD)
~~~

The complete historical sequence may provide unlabeled affine scaling context.
No label, gradient step, or target-specific checkpoint selection is used.

## Evidence Included

- original-paper values and protocol-faithful TShape reproduction;
- six datasets: AIOPS, NAB, TODS, UCR, WSD, and Yahoo;
- 834/834 series and 15,126,509 raw test observations;
- one EasyTSAD protocol: PointF1PA and EventF1PA(mode="log");
- one-seed, 44-method by six-dataset main ledger;
- four official frozen forecasting checkpoints;
- all original TShape baseline families with explicit execution status;
- full 10%-100% Pattern Bank size sweep;
- full-series fusion, normalization, and channel ablations;
- paired series bootstrap intervals and sign-test counts;
- strict checkpoint, CLI, local web scorer, and complex real AIOPS case;
- versioned model card and explicit failure card;
- vector paper figures and IEEE source.

Rows marked N, F, and S are directly deployment-comparable no-fit, official
frozen, and strict synthetic-only methods. Rows marked X are family adapters;
A rows are adaptation diagnostics; T and B rows are diagnostics and ablations.
Only N/F/S rows participate in red/blue ranking.

## Five-Minute Data-Free Check

~~~bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements_rene.txt

TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
.venv/bin/python experiments/verify_easytsad_protocol.py

.venv/bin/python experiments/tshape_zero_product.py score \
  --checkpoint models/tshape_zero_plus_release.pt \
  --values "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
~~~

The protocol command compares the optimized evaluator with EasyTSAD 0.3.0.2
on boundary and randomized tied-score traces. The product command must return
24 aligned scores, separate neural/guard channels, and ten ranked indices.

Launch the interactive product with:

~~~bash
.venv/bin/python experiments/tshape_zero_product.py serve \
  --checkpoint models/tshape_zero_plus_release.pt \
  --host 127.0.0.1 --port 8787
~~~

Then open http://127.0.0.1:8787. The default example is a real 240-point AIOPS
window with drift and five incident phases. It is an interface demonstration,
not an additional aggregate result.

## Dataset Layout

Raw third-party datasets are not silently redistributed. Place each series at:

~~~text
datasets/UTS/<dataset>/<series>/train.npy
datasets/UTS/<dataset>/<series>/test.npy
datasets/UTS/<dataset>/<series>/test_label.npy
~~~

The dataset manifest records expected counts and checksums. All-normal series
remain in coverage and runtime accounting and are excluded only where F1 is
undefined.

## Full Reproduction

The canonical command guide is experiments/README_RENE.md. It documents:

1. environment and dataset checks;
2. original TShape score replay and faithful completion;
3. no-fit residual controls;
4. official Chronos, Timer, TimesFM, and Sundial checkpoints;
5. strict synthetic Pattern Bank training and the ten-level full-series sweep;
6. Residual Guard weight, rank, and pairwise-channel ablations;
7. the 44x6 ledger, paired statistics, and all figures;
8. product scoring and release verification.

Official TimeMoE-50M exceeded the predeclared ten-minute optional-checkpoint
screen and is omitted rather than reported on a partial subset. The separately
marked TimeMoE family adapter remains in the diagnostic ledger.

## Resource Expectations

On the study machine (Apple M3, 16 GB), the final 60,000-window checkpoint
trains in 128 seconds. Dense stride-one scoring of all 834 series takes 291
seconds on MPS. The complete ten-level sweep used 756 training seconds and
3,085 evaluation seconds (64.0 minutes total); every fraction is restartable
and can run in an independent CPU/MPS queue. Official foundation checkpoints
require their own downloaded caches; checkpoint footprints are reported in the
paper rather than hidden in artifact setup.

## Final Verification

After all dataset-dependent runs, figure generation, and IEEE compilation:

~~~bash
TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
.venv/bin/python experiments/verify_strict_release.py \
  --json Results/RENE/strict_release_verification.json
~~~

The verifier fails on an incomplete dataset, non-single-seed row, missing
method, absent Pattern fraction, partial ablation, changed keywords, stale LODO
language, local path leakage in the paper, missing figure, checkpoint metadata
mismatch, product misalignment, protocol mismatch, or missing PDF.

## Public Entry Point

Source and release: https://github.com/CSTCloudOps/TShape-Zero

ISSRE 2026 RENE does not use double-anonymous review, so this public artifact
may retain authorship metadata. The authoring draft can be longer for internal
editing, but the submitted PDF must be trimmed to the official 10 body pages
plus at most 2 reference-only pages. Dataset licenses must still be respected
before publishing an archival release.
