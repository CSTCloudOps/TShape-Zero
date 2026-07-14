# TShape-Zero+ RENE Reproduction Guide

This artifact supports the ISSRE RENE study of target-trained TShape,
frozen-checkpoint reuse, and the TShape-Zero+ repair. The paper reports exactly
two EasyTSAD metrics for every score stream:

- `PointF1PA()` (Point-F1, or F1)
- `EventF1PA(mode="log")` (Event-F1, or F1-E)

The unified paper ledger does not mix these values with AP, AUROC, quantile
thresholds, or a separately implemented event metric.

## 1. Evidence Boundary

The artifact keeps five evidence types separate:

1. `Original reported`: values transcribed from the original TShape paper.
2. `Faithful reproduction`: stored per-series scores plus protocol-faithful
   completion of missing AIOPS/TODS runs.
3. `Target-context diagnostic`: a shared, fixed-budget implementation used to
   diagnose preprocessing and training-scope sensitivity.
4. `Strict zero-shot evaluation`: one immutable checkpoint is trained only on
   generated Pattern Bank windows and then applied unchanged to all six targets.
5. `Product checkpoint`: the exact strict checkpoint used in the aggregate
   evaluation, packaged with fixed fusion, alignment, and provenance metadata.

Official downloaded checkpoints, local family adapters, and adaptation
diagnostics are marked separately in the result ledger. Adapter
values are executed results, not numbers claimed by the corresponding papers.

## 2. Environments

The base environment is Python 3.9:

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements_rene.txt
```

Official Timer, Sundial, and TimesFM checkpoints use the isolated
Python 3.11 environment because their released dependency pins conflict:

```bash
python3.11 -m venv .venv311_foundation
.venv311_foundation/bin/pip install -r requirements_foundation.txt
```

The full runs used Apple Silicon MPS for TShape. Chronos, Timer, Sundial,
TimesFM, and deterministic adapters used CPU. Every run-info JSON
records the device and execution contract.

## 3. Data Layout

Place each univariate series under:

```text
datasets/UTS/<dataset>/<series>/train.npy
datasets/UTS/<dataset>/<series>/test.npy
datasets/UTS/<dataset>/<series>/test_label.npy
```

The six dataset names are `AIOPS`, `NAB`, `TODS`, `UCR`, `WSD`, and `Yahoo`.
The complete local artifact contains 834 series and 15,126,509 raw test
observations. Drop-first differencing leaves 15,125,675 evaluated positions;
the product adds one non-alarming warm-up score per series so output indices
remain aligned with the raw upload.
Dataset licenses may prevent redistribution; the release provides the expected
layout and checksum manifest instead of silently repackaging third-party data.

## 4. Fast Verification

Run the data-free metric contract check:

```bash
TSHAPE_EASYTSAD_ONLY=1 \
TSHAPE_EASYTSAD_BACKEND=compatible \
.venv/bin/python experiments/verify_easytsad_protocol.py
```

It evaluates 16 edge-case traces against EasyTSAD 0.3.0.2. Point-F1,
Event-F1, precision, recall, and selected thresholds must match exactly.

Run a product smoke test:

```bash
.venv/bin/python experiments/tshape_zero_product.py score \
  --checkpoint models/tshape_zero_plus_release.pt \
  --values "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
```

## 5. Profiles and No-Training Controls

```bash
TSHAPE_EASYTSAD_ONLY=1 .venv/bin/python experiments/rene_experiments.py profile
TSHAPE_EASYTSAD_ONLY=1 .venv/bin/python experiments/rene_experiments.py \
  baselines --include-median
```

These commands produce the dataset profile and persistence, rolling-mean,
rolling-median, spectral-residual, and rolling-MAD scores over all 834 series.

## 6. TShape Reproduction and Diagnostics

Complete the original per-series score archive without overwriting immutable
stored arrays:

```bash
TSHAPE_EASYTSAD_ONLY=1 .venv/bin/python experiments/rene_experiments.py \
  tshape-easytsad-naive \
  --datasets AIOPS NAB TODS UCR WSD Yahoo \
  --only-missing-original --force-datasets TODS \
  --epochs 100 --patience 5 --batch-size 128 --eval-batch-size 4096 \
  --device mps --tag submission_easytsad_naive_completion

TSHAPE_EASYTSAD_ONLY=1 .venv/bin/python \
  experiments/build_tshape_faithful_replay.py
```

Run the full-series target-context preprocessing diagnostic:

```bash
TSHAPE_EASYTSAD_ONLY=1 .venv/bin/python experiments/rene_experiments.py \
  target-diagnostics \
  --datasets AIOPS NAB TODS UCR WSD Yahoo \
  --variants diff1_p16 raw_p16 diff1_p32 \
  --epochs 5 --batch-size 256 --eval-batch-size 4096 --device mps \
  --tag submission_easytsad_target_diagnostics
```

## 7. Strict Pattern Bank TShape-Zero

The paper's zero-shot model is not trained on the other five benchmark
datasets. One checkpoint is trained only on generated Pattern Bank windows and
then applied unchanged to all six targets. The full release checkpoint uses
60,000 windows; every experiment uses seed 20260704.

Run the 100% checkpoint and full-series evaluation:

~~~bash
TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
.venv/bin/python experiments/rene_experiments.py pattern-bank-tshape \
  --targets AIOPS NAB TODS UCR WSD Yahoo \
  --fractions 1.00 --size-sweep \
  --max-train-windows 60000 --min-train-windows 6000 \
  --epochs 30 --early-stop-patience 3 --early-stop-min-delta 0.00001 \
  --batch-size 256 --eval-batch-size 2048 --eval-stride 1 \
  --device auto --emit-zero-plus --fusion-alpha 0.15 \
  --checkpoint-dir models/strict_pattern_main \
  --seed 20260704 --tag submission_strict_zero_main_full
~~~

Run the nested 10%-90% size sweep. Each fraction is an independent,
restartable full-series job; 100% is supplied by the preceding command.

~~~bash
for pct in 10 20 30 40 50 60 70 80 90
do
  frac=$(printf "0.%02d" "$pct")
  TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
  .venv/bin/python experiments/rene_experiments.py pattern-bank-tshape \
    --targets AIOPS NAB TODS UCR WSD Yahoo \
    --fractions "$frac" --size-sweep \
    --max-train-windows 60000 --min-train-windows 6000 \
    --epochs 30 --early-stop-patience 3 --early-stop-min-delta 0.00001 \
    --batch-size 256 --eval-batch-size 2048 --eval-stride 1 \
    --device auto --emit-zero-plus --fusion-alpha 0.15 \
    --checkpoint-dir models/strict_pattern_size \
    --seed 20260704 --tag "submission_strict_zero_size_$pct"
done
~~~

The 10% run uses 6,000 windows; each level adds a strict prefix of 6,000
generated windows until 60,000. All levels use the same 20-generator mixture,
incident process, and seed. A checkpoint is never chosen by a benchmark F1.

## 8. Residual Guard Ablation

The matched guard, neural-budget sweep, rank normalization, and pairwise
TShape+median/TShape+spectral variants reuse the frozen 100% checkpoint:

~~~bash
TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
.venv/bin/python experiments/strict_zero_ablation.py \
  --targets AIOPS NAB TODS UCR WSD Yahoo \
  --device cpu --eval-batch-size 16384 \
  --seed 20260704 --tag submission_strict_zero_ablation
~~~

The release formula is fixed before benchmark inspection:

~~~text
Zero+ = 0.15 * minmax(Pattern-TShape)
      + 0.55 * minmax(rolling-median)
      + 0.25 * minmax(spectral-residual)
      + 0.05 * minmax(rolling-MAD)
~~~

The ablation evaluates alpha in {0, .05, .10, .15, .25, .50, 1}, a
rank-normalized .15 variant, and the two pairwise channel combinations.
Alpha=0 is the exactly matched guard and alpha=1 is Pattern-only TShape.

## 9. Modern and Foundation Baselines

No-fit controls are regenerated over all 834 series with:

~~~bash
TSHAPE_EASYTSAD_ONLY=1 .venv/bin/python experiments/rene_experiments.py \
  baselines --include-median
~~~

The ledger contains four official forecasting checkpoints used without target
training: Chronos-Bolt-Tiny, Timer-base-84M, TimesFM-2.5-200M, and
Sundial-base-128M. Timer, TimesFM, and Sundial use the isolated Python 3.11
environment described above. Their exact output tags are
submission_easytsad_timer, submission_easytsad_timesfm, and
submission_easytsad_sundial; Chronos uses submission_easytsad_chronos.

Official TimeMoE-50M exceeded the predeclared ten-minute optional-checkpoint
screen and is omitted instead of being reported on a partial subset.
TimeMixer, TimeMoE, and anomaly-model rows marked X are executable family
adapters. USAD, TranAD, and Anomaly Transformer rows marked A are adaptation
diagnostics. They broaden the stress test but are not presented as official
frozen-checkpoint reproductions. The main table ranks only no-fit, official
frozen, and strict synthetic-only rows.

## 10. Unified Ledger, Statistics, and Figures

~~~bash
.venv/bin/python experiments/build_easytsad_submission_results.py
.venv/bin/python experiments/make_easytsad_submission_assets.py
.venv/bin/python experiments/export_strict_submission_results.py
.venv/bin/python experiments/make_strict_zero_submission_assets.py
~~~

The generated main ledger is rectangular: 44 methods by six datasets, one
seed, and full 834/834-series coverage. The paper-facing ledger contains only
Point-F1 and Event-F1. Paired effects use series as the analysis unit and report
a 10,000-resample bootstrap interval plus exact sign-test counts. The interval
describes series heterogeneity under the fixed run; it is not a multi-seed
stability interval.

Vector PDF and SVG figures are generated with white backgrounds. The strict
asset generator fails if any of the ten Pattern Bank sizes or any full
ablation row is missing.

## 11. Product Checkpoint and Web Case

The distributed model is an exact state-dict copy of the strict 100% checkpoint.
Its metadata records synthetic-only training, zero benchmark training windows,
the fixed 0.15 fusion budget, preprocessing, alignment, stopping rule, and
model hash. It is not retrained on the six evaluation datasets.

Score pasted values or launch the local web product:

~~~bash
.venv/bin/python experiments/tshape_zero_product.py score \
  --checkpoint models/tshape_zero_plus_release.pt \
  --values "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"

.venv/bin/python experiments/tshape_zero_product.py serve \
  --checkpoint models/tshape_zero_plus_release.pt \
  --host 127.0.0.1 --port 8787
~~~

The web UI returns the aligned input, fused score, Pattern-TShape score,
Residual Guard score, and top-ranked indices. Its default example is a real
240-point, five-event AIOPS window. Labels are optional retrospective evidence
and never enter scoring. The case is illustrative and excluded from aggregate
claims.

## 12. Fail-Fast Verification

After experiments, table generation, paper compilation, and screenshot
regeneration, run:

~~~bash
TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
.venv/bin/python experiments/verify_strict_release.py \
  --json Results/RENE/strict_release_verification.json
~~~

The verifier checks the synthetic-only checkpoint contract, 0.15 fusion
weight, one-seed 44x6 ledger, 100% coverage, all ten Pattern Bank sizes,
full-series ablations, exact EasyTSAD compatibility, the complex product case,
required figures, unchanged keywords, stale LODO language, local path leakage,
and the compiled PDF. A missing or shortened result fails rather than silently
reducing the study.
