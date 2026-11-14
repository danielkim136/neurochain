"""A Kalman filter decoder for behaviour from neural features.

The filter assumes two linear relations with Gaussian noise (Wu et al., 2006):

* **Behaviour has momentum:** ``x[t] = A x[t-1] + w``, with ``w ~ N(0, W)``.
* **Features encode behaviour:** ``y[t] = C x[t] + q``, with ``q ~ N(0, Q)``.

At each step it predicts the state from the last one, then corrects the
prediction by a gain ``K`` times the part of the features the prediction did
not explain. ``A``, ``W``, ``C`` and ``Q`` are fitted by least squares.

Because the model does not change over time, the gain converges to a constant
within a few steps. This implementation computes that steady-state gain once
and then filters with it, which is also how an embedded decoder would run: one
matrix-vector product per time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurochain.recording import FloatArray, IntArray


def steady_state_gain(
    a: FloatArray, w: FloatArray, c: FloatArray, q: FloatArray, *, tol: float = 1e-10
) -> FloatArray:
    """Iterate the Riccati recursion to the constant Kalman gain.

    Uses the information form, ``P = (I + P_pred C' Q^-1 C)^-1 P_pred`` and
    ``K = P C' Q^-1``. The only large inverse is of ``Q``, done once, and
    ``P_pred`` itself is never inverted, so a state with an exact relation
    between its parts (displacement is the sum of velocity) causes no trouble.

    Args:
        a: State transition, ``(k, k)``.
        w: Process noise covariance, ``(k, k)``.
        c: Observation matrix, ``(d, k)``.
        q: Observation noise covariance, ``(d, d)``.
        tol: Largest change in the gain at which iteration stops.
    """
    q_inverse_c = np.linalg.solve(q, c)
    information = c.T @ q_inverse_c
    p = w.copy()
    gain = np.zeros((a.shape[0], c.shape[0]))
    for _ in range(1000):
        p_pred = a @ p @ a.T + w
        p = np.linalg.solve(np.eye(a.shape[0]) + p_pred @ information, p_pred)
        new_gain = p @ q_inverse_c.T
        if np.max(np.abs(new_gain - gain)) < tol:
            return np.asarray(new_gain, dtype=np.float64)
        gain = new_gain
    return np.asarray(gain, dtype=np.float64)


def delayed(features: FloatArray, groups: IntArray, lag: int) -> FloatArray:
    """Return features delayed by ``lag`` bins within each trial.

    Row ``t`` holds the features of bin ``t - lag``, so behaviour is paired with
    the neural activity that preceded it. Rows that would reach into another
    trial are zero.
    """
    if lag == 0:
        return features
    out = np.roll(features, lag, axis=0)
    crossed = np.roll(groups, lag) != groups
    crossed[:lag] = True
    out[crossed] = 0.0
    return out


def running_integral(values: FloatArray, groups: IntArray) -> FloatArray:
    """Return the running sum of ``values`` inside each trial.

    For a velocity this is proportional to the displacement since the trial
    began; the constant (the bin width) does not matter to a linear model.
    """
    total = np.cumsum(values, axis=0)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    lengths = np.diff(np.r_[starts, len(groups)])
    before = np.vstack([np.zeros((1, values.shape[1])), total])[starts]
    return np.asarray(total - np.repeat(before, lengths, axis=0), dtype=np.float64)


@dataclass
class KalmanDecoder:
    """A steady-state Kalman filter whose state is the behaviour itself.

    Two details follow the standard practice for motor decoding (Wu et al.,
    2006):

    * **The state carries displacement as well as velocity.** Neurons are tuned
      to where the hand is as well as how fast it moves, so the state is
      ``[velocity, displacement since the trial began]``. Only velocity is
      returned.
    * **The features are delayed.** Motor cortex leads movement by tens of
      milliseconds and the filter's observation model is instantaneous, so the
      features are delayed by a whole number of bins. The delay is chosen by
      cross-validation over contiguous blocks of the training trials.

    Attributes:
        regularisation: Share of the mean feature variance added to the
            diagonal of ``Q``, which keeps it invertible when channels are
            nearly collinear.
        lags: Delays, in bins, to choose between.
        integral_state: Whether the state carries the displacement.
        selection_folds: Blocks of training trials used to choose the delay.
        lag: The delay in use (set by :meth:`fit`).
    """

    regularisation: float = 1.0e-3
    lags: tuple[int, ...] = (0, 1, 2, 3, 4)
    integral_state: bool = True
    selection_folds: int = 4
    lag: int = 0
    _a: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _c: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _gain: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _state_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _feature_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _n_targets: int = field(default=0, repr=False)

    @property
    def n_states(self) -> int:
        """Size of the filter's state vector (after :meth:`fit`)."""
        return int(self._a.shape[0])

    def fit(self, features: FloatArray, targets: FloatArray, groups: IntArray) -> None:
        """Choose the delay on held-back training trials, then fit on all of them.

        Args:
            features: ``(n_bins, n_features)``.
            targets: ``(n_bins, n_targets)`` behaviour.
            groups: ``(n_bins,)`` trial label; transitions across trials are not
                used to fit the dynamics.
        """
        if len(self.lags) > 1:
            errors = np.zeros(len(self.lags))
            labels = np.unique(groups)
            for block in np.array_split(labels, min(self.selection_folds, len(labels))):
                held = np.isin(groups, block)
                for position, lag in enumerate(self.lags):
                    self.lag = lag
                    self._fit(features[~held], targets[~held], groups[~held])
                    prediction = self.predict(features[held], groups[held])
                    errors[position] += float(np.sum((prediction - targets[held]) ** 2))
            self.lag = self.lags[int(np.argmin(errors))]
        else:
            self.lag = self.lags[0]
        self._fit(features, targets, groups)

    def _fit(self, features: FloatArray, targets: FloatArray, groups: IntArray) -> None:
        """Fit the two linear models and the steady-state gain at the current delay."""
        self._n_targets = targets.shape[1]
        state = targets
        if self.integral_state:
            state = np.hstack([targets, running_integral(targets, groups)])
        features = delayed(features, groups, self.lag)
        self._state_mean = state.mean(axis=0)
        self._feature_mean = features.mean(axis=0)
        x = state - self._state_mean
        y = features - self._feature_mean
        within = groups[1:] == groups[:-1]
        previous, current = x[:-1][within], x[1:][within]
        k = x.shape[1]
        tiny = 1e-9 * float(np.trace(x.T @ x)) / k * np.eye(k)
        self._a = np.linalg.solve(previous.T @ previous + tiny, previous.T @ current).T
        process = current - previous @ self._a.T
        w = process.T @ process / len(process) + 1e-12 * np.eye(k)
        self._c = np.linalg.solve(x.T @ x + tiny, x.T @ y).T
        residual = y - x @ self._c.T
        q = residual.T @ residual / len(residual)
        q += self.regularisation * float(np.trace(q)) / q.shape[0] * np.eye(q.shape[0])
        self._gain = steady_state_gain(self._a, w, self._c, q)

    def predict(self, features: FloatArray, groups: IntArray) -> FloatArray:
        """Filter the features and return the decoded behaviour, ``(n_bins, n_targets)``.

        The state restarts from the training mean at the first bin of each
        trial.
        """
        y = delayed(features, groups, self.lag) - self._feature_mean
        k = self.n_states
        out = np.empty((len(y), k))
        # x[t] = M x[t-1] + K y[t], with M = (I - K C) A.
        transition = (np.eye(k) - self._gain @ self._c) @ self._a
        driven = y @ self._gain.T
        state = np.zeros(k)
        previous_group = None
        for t in range(len(y)):
            if groups[t] != previous_group:
                state = np.zeros(k)
                previous_group = groups[t]
            state = transition @ state + driven[t]
            out[t] = state
        return (out + self._state_mean)[:, : self._n_targets]
