"""Ridge regression on lagged features, scored by blocked cross-validation.

A linear readout is the baseline every other decoder has to beat, and it is
enough to compare two feature sets fairly: same model, same folds, different
inputs.

Folds are contiguous blocks of whole trials. Neighbouring time bins are highly
correlated, so shuffling bins into folds would leak the test set into training
and inflate every score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurochain.recording import FloatArray, IntArray


def lagged(features: FloatArray, groups: IntArray, n_lags: int) -> FloatArray:
    """Stack each row with its ``n_lags - 1`` predecessors from the same group.

    Args:
        features: ``(n_bins, n_features)``.
        groups: ``(n_bins,)`` label of the trial each bin belongs to. A lag that
            would reach into another trial is filled with zeros.
        n_lags: Number of time steps per row, the current one included.

    Returns:
        ``(n_bins, n_features * n_lags)``.
    """
    columns = []
    for lag in range(n_lags):
        shifted = np.roll(features, lag, axis=0)
        crossed = np.roll(groups, lag) != groups
        crossed[:lag] = True
        shifted[crossed] = 0.0
        columns.append(shifted)
    return np.concatenate(columns, axis=1)


def fit_ridge(x: FloatArray, y: FloatArray, alpha: float) -> tuple[FloatArray, float]:
    """Fit ``y = x @ w + b`` with an L2 penalty ``alpha * n * |w|**2``.

    Inputs are expected to be standardised. Returns ``(weights, intercept)``.
    """
    intercept = float(np.mean(y))
    gram = x.T @ x + alpha * x.shape[0] * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x.T @ (y - intercept))
    return np.asarray(weights, dtype=np.float64), intercept


def r_squared(y: FloatArray, prediction: FloatArray) -> float:
    """Return the coefficient of determination, ``1 - SS_res / SS_tot``."""
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - residual / total


@dataclass(frozen=True)
class CrossValidation:
    """Scores of a blocked cross-validation.

    Attributes:
        r2_folds: Coefficient of determination on each held-out fold.
        correlation_folds: Pearson correlation on each held-out fold.
    """

    r2_folds: FloatArray
    correlation_folds: FloatArray

    @property
    def r2(self) -> float:
        """Mean held-out R²."""
        return float(np.mean(self.r2_folds))

    @property
    def r2_sem(self) -> float:
        """Standard error of the mean R² across folds."""
        return float(np.std(self.r2_folds, ddof=1) / np.sqrt(self.r2_folds.size))

    @property
    def correlation(self) -> float:
        """Mean held-out correlation."""
        return float(np.mean(self.correlation_folds))


def blocked_cross_validation(
    features: FloatArray,
    target: FloatArray,
    groups: IntArray,
    *,
    n_folds: int = 5,
    n_lags: int = 5,
    alpha: float = 0.01,
) -> CrossValidation:
    """Score a lagged ridge readout on held-out blocks of trials.

    Features are standardised with the training fold's mean and standard
    deviation only.

    Args:
        features: ``(n_bins, n_features)``.
        target: ``(n_bins,)`` the quantity to decode, or ``(n_bins, n_targets)``
            to decode several at once; scores are then averaged over targets
            within each fold.
        groups: ``(n_bins,)`` trial label of each bin.
        n_folds: Number of contiguous blocks of trials.
        n_lags: Time steps of history given to the readout.
        alpha: Ridge penalty.
    """
    x = lagged(features, groups, n_lags)
    y = target if target.ndim == 2 else target[:, np.newaxis]
    blocks = np.array_split(np.unique(groups), n_folds)
    r2, correlation = [], []
    for block in blocks:
        test = np.isin(groups, block)
        mean = x[~test].mean(axis=0)
        scale = x[~test].std(axis=0) + 1e-9
        x_train = (x[~test] - mean) / scale
        intercept = y[~test].mean(axis=0)
        gram = x_train.T @ x_train + alpha * x_train.shape[0] * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x_train.T @ (y[~test] - intercept))
        prediction = (x[test] - mean) / scale @ weights + intercept
        r2.append(np.mean([r_squared(y[test][:, d], prediction[:, d]) for d in range(y.shape[1])]))
        correlation.append(
            np.mean([np.corrcoef(prediction[:, d], y[test][:, d])[0, 1] for d in range(y.shape[1])])
        )
    return CrossValidation(
        np.asarray(r2, dtype=np.float64), np.asarray(correlation, dtype=np.float64)
    )


@dataclass
class RidgeDecoder:
    """The lagged ridge readout as a decoder object.

    Attributes:
        n_lags: Bins of history per prediction, the current one included.
        alpha: Ridge penalty.
    """

    n_lags: int = 5
    alpha: float = 0.01

    def fit(self, features: FloatArray, targets: FloatArray, groups: IntArray) -> None:
        """Fit the weights on lagged features."""
        x = lagged(features, groups, self.n_lags)
        self._intercept = targets.mean(axis=0)
        gram = x.T @ x + self.alpha * x.shape[0] * np.eye(x.shape[1])
        self._weights = np.linalg.solve(gram, x.T @ (targets - self._intercept))

    def predict(self, features: FloatArray, groups: IntArray) -> FloatArray:
        """Return decoded behaviour, ``(n_bins, n_targets)``."""
        prediction = lagged(features, groups, self.n_lags) @ self._weights + self._intercept
        return np.asarray(prediction, dtype=np.float64)


@dataclass
class WienerDecoder:
    """A ridge readout with long history and a penalty chosen by cross-validation.

    The same linear map as :class:`RidgeDecoder`, but tuned the way the other
    decoders are: it sees ``n_lags`` bins of history (about a second), and the
    penalty that keeps that many weights from fitting noise is chosen on
    contiguous blocks of the training trials. In the BCI literature a linear
    filter over lagged features is called a Wiener filter.

    Choosing the penalty costs one eigendecomposition per block: with
    ``G = V diag(s) V'`` the ridge solution for every penalty is
    ``V diag(1 / (s + penalty)) V' b``.

    Attributes:
        n_lags: Bins of history, including the current bin.
        alphas: Penalties to choose between, as in :func:`fit_ridge`.
        selection_folds: Blocks of training trials used to choose the penalty.
        alpha: The penalty in use (set by :meth:`fit`).
    """

    n_lags: int = 20
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    selection_folds: int = 3
    alpha: float = 0.0
    _weights: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _intercept: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)

    def fit(self, features: FloatArray, targets: FloatArray, groups: IntArray) -> None:
        """Choose the penalty on held-back training trials, then fit on all of them."""
        x = lagged(features, groups, self.n_lags)
        self._intercept = targets.mean(axis=0)
        y = targets - self._intercept
        labels = np.unique(groups)
        blocks = [
            np.isin(groups, block)
            for block in np.array_split(labels, min(self.selection_folds, len(labels)))
        ]
        grams = [x[held].T @ x[held] for held in blocks]
        moments = [x[held].T @ y[held] for held in blocks]
        gram, moment = np.sum(grams, axis=0), np.sum(moments, axis=0)
        if len(self.alphas) > 1:
            errors = np.zeros(len(self.alphas))
            for held, held_gram, held_moment in zip(blocks, grams, moments, strict=True):
                values, vectors = np.linalg.eigh(gram - held_gram)
                projected = vectors.T @ (moment - held_moment)
                n_train = len(x) - int(held.sum())
                for position, alpha in enumerate(self.alphas):
                    weights = vectors @ (projected / (values + alpha * n_train)[:, np.newaxis])
                    errors[position] += float(np.sum((x[held] @ weights - y[held]) ** 2))
            self.alpha = self.alphas[int(np.argmin(errors))]
        else:
            self.alpha = self.alphas[0]
        penalty = self.alpha * len(x) * np.eye(x.shape[1])
        self._weights = np.linalg.solve(gram + penalty, moment)

    def predict(self, features: FloatArray, groups: IntArray) -> FloatArray:
        """Return decoded behaviour, ``(n_bins, n_targets)``."""
        return np.asarray(
            lagged(features, groups, self.n_lags) @ self._weights + self._intercept,
            dtype=np.float64,
        )
