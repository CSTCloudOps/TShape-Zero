# TShape-Zero+ Model Card

## Model

TShape-Zero+ is a univariate, batch time-series anomaly scorer. The release
contains one compact TShape next-value predictor and a deterministic Residual
Guard. The same checkpoint is used for every target series.

## Intended Use

- retrospective triage of software-service KPI histories;
- score generation before target labels or target-specific training exist;
- reproducible comparison under EasyTSAD `PointF1PA()` and
  `EventF1PA(mode="log")`;
- inspection of neural and residual evidence through the CLI or web interface.

It is not an automated paging policy, a causal streaming detector, or a root
cause analysis system.

## Training Provenance

- architecture: TShape with context length 16 and four size-4 patches;
- training corpus: 60,000 generated Pattern Bank continuation windows;
- temporal coverage: 20 normal generators and eight randomized variants;
- perturbations: context-only spike, dip, level shift, variance change, drift,
  and burst operators;
- optimization: Adam, batch 256, at most 30 epochs, patience 3;
- seed: 20260704;
- benchmark values used to fit weights: none;
- benchmark labels used to fit or select weights: none.

The machine-readable checkpoint manifest records the exact model digest and
training metadata.

The 20 generators are flat, linear, quadratic, exponential trend, sine,
multi-sine, sawtooth, triangle, square, stair, random walk, mean reverting,
damped oscillation, chirp, autoregressive, trend-seasonal, logistic growth,
smooth piecewise, seasonal drift, and periodic pulse.

## Input and Output

Input is one finite univariate history and, optionally, a separate unlabeled
calibration prefix. The shared scorer applies first differencing, affine
scaling, and index alignment. It returns:

- the fused TShape-Zero+ score;
- the Pattern-TShape score;
- the matched Residual Guard score;
- aligned raw indices and top-ranked suspicious points.

The fixed fusion is:

```text
0.15 * minmax(Pattern-TShape)
+ 0.55 * minmax(rolling-median residual)
+ 0.25 * minmax(spectral residual)
+ 0.05 * minmax(rolling-MAD residual)
```

## Evaluation Scope

The release was evaluated with one seed on all 834 series from AIOPS, NAB,
TODS, UCR, WSD, and Yahoo. Results use only the two EasyTSAD metrics named
above. All-normal test series remain in coverage accounting and are excluded
only where F1 is undefined.

## Ethical and Operational Notes

The model does not infer incident severity, affected users, or causality. A
human operator should inspect the original KPI, channel attribution, and
service context before acting. See `FAILURE_CARD.md` for explicit boundaries.
