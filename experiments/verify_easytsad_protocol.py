#!/usr/bin/env python3
"""Check the vectorized PA evaluator against EasyTSAD 0.3.0.2 without datasets."""

from __future__ import annotations

import numpy as np

from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

import rene_experiments as re


def boundary_traces():
    """Return deterministic edge cases for event boundaries, ties, and normal data."""
    labels = [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0],
    ]
    scores = [
        [0.1, 0.2, 0.8, 0.7, 0.3, 0.0],
        [0.6, 0.2, 0.1, 0.7, 0.8, 0.9],
        [0.9, 0.8, 0.3, 0.2, 0.1, 0.0],
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.4, 0.7, 0.7, 0.4, 0.7, 0.7],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        [0.5, 0.4, 0.5, 0.4, 0.5, 0.4],
        [0.9, 0.1, 0.2, 0.8, 0.3, 0.7],
        [0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
        [0.3, 0.8, 0.6, 0.4, 0.2, 0.9],
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.9],
        [np.nan, 0.1, 0.9, 0.8, 0.7, 0.2],
        [0.1, 1.0, 0.1, 0.1, 1.0, 0.1],
        [-1.0, -0.5, 0.0, 0.5, 1.0, 0.2],
        [1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9],
    ]
    traces = []
    for label, score in zip(labels, scores):
        label_array = np.asarray(label, dtype=np.int8)
        score_array = np.asarray(score, dtype=np.float64)
        if np.any(~np.isfinite(score_array)):
            score_array = np.nan_to_num(score_array, nan=-1.0)
        traces.append((label_array, score_array))
    return traces


def randomized_traces(count: int = 200):
    rng = np.random.default_rng(20260704)
    traces = []
    for index in range(count):
        length = int(rng.integers(24, 193))
        labels = np.zeros(length, dtype=np.int8)
        if index % 25:
            for _ in range(int(rng.integers(1, 6))):
                start = int(rng.integers(0, length))
                width = int(rng.integers(1, min(24, length - start) + 1))
                labels[start : start + width] = 1
        scores = rng.normal(size=length)
        if index % 4 == 0:
            scores = np.round(scores, 1)
        if index % 7 == 0:
            scores[labels == 1] += 0.7
        traces.append((labels, scores.astype(np.float64)))
    return traces


def assert_match(labels: np.ndarray, scores: np.ndarray) -> None:
    for protocol, mode in ((PointF1PA(), "point"), (EventF1PA(mode="log"), "log")):
        official = protocol.calc(scores, labels, None)
        compatible = re._easytsad_pa_compatible(labels, scores, mode)
        expected = np.asarray([official.f1, official.p, official.r, official.thres])
        np.testing.assert_allclose(compatible, expected, rtol=0.0, atol=1e-12)


def main() -> None:
    boundaries = boundary_traces()
    randomized = randomized_traces()
    for labels, scores in boundaries + randomized:
        assert_match(labels, scores)
    print(
        "EasyTSAD PA compatibility verified on "
        f"{len(boundaries)} boundary and {len(randomized)} randomized traces."
    )


if __name__ == "__main__":
    main()
