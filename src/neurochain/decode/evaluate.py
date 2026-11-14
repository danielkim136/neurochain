"""Held-out evaluation shared by every decoder.

Three pieces:

* :func:`held_out_predictions` trains on all but one contiguous block of
  trials and predicts that block, for each block in turn. Every bin is
  predicted exactly once, by a decoder that never saw it.
* :func:`r2_score` pools those predictions into one R².
* :func:`bootstrap_r2` puts a confidence interval on it by resampling trials.
  Trials, not bins, are resampled, because neighbouring bins are strongly
  correlated and would make the interval far too narrow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from neurochain.recording import FloatArray, IntArray


class Decoder(Protocol):
    """Anything that can be fitted to features and behaviour and then predict."""

    def fit(self, features: FloatArray, targets: FloatArray, groups: IntArray) -> None:
        """Fit to training data."""

    def predict(self, features: FloatArray, groups: IntArray) -> FloatArray:
        """Return decoded behaviour for new features."""


def standardise(train: FloatArray, *others: FloatArray) -> tuple[FloatArray, ...]:
    """Scale features by the training set's mean and standard deviation."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0) + 1e-9
    return tuple((x - mean) / scale for x in (train, *others))


def fit_and_predict(
    make_decoder: Callable[[], Decoder],
    train: tuple[FloatArray, FloatArray, IntArray],
    test_features: FloatArray,
    test_groups: IntArray,
) -> FloatArray:
    """Fit a fresh decoder on one set and predict another."""
    features, targets, groups = train
    x_train, x_test = standardise(features, test_features)
    decoder = make_decoder()
    decoder.fit(x_train, targets, groups)
    return decoder.predict(x_test, test_groups)


def held_out_predictions(
    make_decoder: Callable[[], Decoder],
    features: FloatArray,
    targets: FloatArray,
    groups: IntArray,
    *,
    n_folds: int = 5,
) -> FloatArray:
    """Predict every bin with a decoder trained on the other folds.

    Folds are contiguous blocks of whole trials, in recording order.

    Returns:
        ``(n_bins, n_targets)`` predictions.
    """
    predictions = np.empty_like(targets, dtype=np.float64)
    for block in np.array_split(np.unique(groups), n_folds):
        test = np.isin(groups, block)
        predictions[test] = fit_and_predict(
            make_decoder,
            (features[~test], targets[~test], groups[~test]),
            features[test],
            groups[test],
        )
    return predictions


def r2_score(targets: FloatArray, predictions: FloatArray) -> float:
    """Return R², averaged over the behaviour's dimensions."""
    residual = np.sum((targets - predictions) ** 2, axis=0)
    total = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(1.0 - residual / total))


def _resampled_scores(
    targets: FloatArray, predictions: FloatArray, groups: IntArray, n_boot: int, seed: int
) -> FloatArray:
    """Return R² on ``n_boot`` resamples of the trials, ``(n_boot,)``.

    Per-trial sums are computed once, so a thousand resamples cost almost
    nothing. The same seed gives the same trials, which is what makes a paired
    comparison of two decoders possible.
    """
    labels, index = np.unique(groups, return_inverse=True)
    n_groups = labels.size

    def per_group(values: FloatArray) -> FloatArray:
        out = np.zeros((n_groups, values.shape[1]))
        np.add.at(out, index, values)
        return out

    count = np.bincount(index, minlength=n_groups).astype(np.float64)[:, np.newaxis]
    sum_y = per_group(targets)
    sum_y2 = per_group(targets**2)
    sum_res = per_group((targets - predictions) ** 2)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    n = count[draws].sum(axis=1)
    total = sum_y2[draws].sum(axis=1) - sum_y[draws].sum(axis=1) ** 2 / n
    return np.asarray(np.mean(1.0 - sum_res[draws].sum(axis=1) / total, axis=1), dtype=np.float64)


def _interval(scores: FloatArray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(scores, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_r2(
    targets: FloatArray,
    predictions: FloatArray,
    groups: IntArray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Return R² with a confidence interval from resampling trials.

    Each resample draws trials with replacement and recomputes R² on their
    bins.

    Returns:
        ``(r2, low, high)``.
    """
    scores = _resampled_scores(targets, predictions, groups, n_boot, seed)
    return (r2_score(targets, predictions), *_interval(scores, level))


def bootstrap_r2_difference(
    targets: FloatArray,
    predictions: FloatArray,
    reference: FloatArray,
    groups: IntArray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Return ``R²(predictions) - R²(reference)`` with a paired interval.

    Both sets of predictions are scored on the same resampled trials, so
    trial-to-trial difficulty, which moves both scores together, cancels. The
    interval on the difference is therefore much narrower than the two separate
    intervals suggest. If it excludes zero, the two differ.

    Returns:
        ``(difference, low, high)``.
    """
    a = _resampled_scores(targets, predictions, groups, n_boot, seed)
    b = _resampled_scores(targets, reference, groups, n_boot, seed)
    difference = r2_score(targets, predictions) - r2_score(targets, reference)
    return (difference, *_interval(a - b, level))
