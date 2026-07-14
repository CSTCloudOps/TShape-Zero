# TShape-Zero+ Failure Card

This document records known failure boundaries of the released scorer. It is
part of the product contract rather than a promise of universal detection.

## Evidence Boundaries

- The full study uses one training seed. Paired bootstrap intervals measure
  cross-series heterogeneity, not optimizer-seed variability.
- The Residual Guard supplies most average robustness. Pattern-TShape adds
  bounded event-level value on some KPI families and is not indispensable on
  every series.
- UCR remains the clearest transfer boundary: rare, heterogeneous events can
  favor persistence or spectral evidence over the fused checkpoint.
- Public foundation checkpoints may have unknown overlap with public benchmark
  corpora. Their rows demonstrate executable reuse, not guaranteed clean-room
  pretraining.

## Scoring Boundaries

- The default normalization uses the complete uploaded history. It supports
  retrospective batch triage and is not causal streaming normalization.
- The product returns continuous scores. EasyTSAD's best F1 threshold uses
  labels and is evaluation-only; it is never presented as an online threshold.
- Histories shorter than the 16-point context cannot provide the documented
  neural score and are rejected with an actionable error.
- Constant, nearly constant, non-finite, or extremely quantized inputs can
  collapse affine score ranges. The scorer sanitizes finite input and exposes
  channels, but an operator should verify the raw trace.
- Abrupt scale changes can make the residual channels dominate. This is an
  intentional reliability floor, not evidence that TShape caused the alarm.

## Deployment Boundaries

- The artifact is univariate and does not model causal or cross-service
  dependencies.
- It has not been validated as a safety-critical autonomous controller.
- Dataset-specific labels, weights, or checkpoint selection must not be added
  while retaining the strict zero-shot claim.
- Streaming use requires a rolling or source-calibrated normalization contract
  and an independently validated alarm policy.

## Operator Response

For every high score, inspect the fused, neural, and guard channels together.
If only the guard fires, first check level, variance, or spectral drift. If the
neural channel fires beyond the guard, inspect repeated local/global shapes.
If channels disagree persistently, retain the score stream for diagnosis and
fall back to an explicitly validated service-specific detector.
