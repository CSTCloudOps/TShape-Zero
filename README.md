# TShape-Zero+

TShape-Zero+ upgrades the target-trained TShape artifact into a reviewable,
out-of-the-box scorer for unseen univariate KPI histories. It combines:

- one 0.138-MiB TShape checkpoint trained only on a generated Pattern Bank;
- 20 temporal generators, eight randomized variants, and context-only incidents;
- a transparent rolling-median, spectral-residual, and rolling-MAD guard;
- a fixed 0.15/0.85 neural/guard reliability budget;
- aligned score output, channel attribution, CLI, and web API.

This repository contains the ISSRE RENE replication and artifact-reuse stress
test. It preserves the original TShape target-trained claim and evaluates the
stronger frozen-checkpoint reuse requirement separately. No benchmark value or
label is used to train the released checkpoint.

## Quick Start

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements_rene.txt

make smoke
make serve
```

Open `http://127.0.0.1:8787` to paste a history and inspect the aligned
TShape-Zero+ score, neural channel, residual guard, and top-ranked indices.

## Evaluation Protocol

Every paper result uses the same EasyTSAD implementations:

- `PointF1PA()` (Point-F1 / F1)
- `EventF1PA(mode="log")` (Event-F1 / F1-E)

The six-dataset study covers AIOPS, NAB, TODS, UCR, WSD, and Yahoo: 834 series
and 15,126,509 test points. One fixed seed is used throughout. Four official
forecasting checkpoints, local family adapters, adaptation diagnostics, and
no-fit controls are explicitly separated in the result ledger.

## Full Reproduction

See [`experiments/README_RENE.md`](experiments/README_RENE.md) for the exact
environment, data layout, faithful TShape replay, four official foundation
checkpoint commands, strict 10--100% Pattern Bank sweep, full-series fusion
ablation, paired statistics, paper figures, product checkpoint, and release
verification.

The review contract is summarized in [`MODEL_CARD.md`](MODEL_CARD.md) and the
known operational boundaries are recorded in
[`FAILURE_CARD.md`](FAILURE_CARD.md). The complete tiered reproduction guide is
[`RENE_ARTIFACT_README.md`](RENE_ARTIFACT_README.md).

## Artifact Links

- Source and release: <https://github.com/CSTCloudOps/TShape-Zero>
- Local web interface: `http://127.0.0.1:8787` after running the serve command

## Original TShape

The original target-trained work is **TShape: Rescuing Machine Learning Models
from Complex Shapelet Anomalies**. TShape-Zero+ does not claim that the original
paper promised zero-shot transfer; it studies and repairs a stronger artifact
reuse contract.
