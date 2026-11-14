import numpy as np

from neurochain.decode.evaluate import r2_score
from neurochain.decode.kalman import KalmanDecoder, delayed, running_integral, steady_state_gain


def simulate(
    seed: int, *, n_trials: int = 60, length: int = 50, lead: int = 0, noise: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth velocity and ten channels that encode it ``lead`` bins early."""
    rng = np.random.default_rng(seed)
    n = n_trials * length
    velocity = np.zeros(n + lead)
    for t in range(1, n + lead):
        velocity[t] = 0.9 * velocity[t - 1] + rng.standard_normal()
    tuning = rng.standard_normal(10)
    features = np.outer(velocity[lead:], tuning) + noise * rng.standard_normal((n, 10))
    return features, velocity[:n, None], np.repeat(np.arange(n_trials), length)


def test_steady_state_gain_matches_the_textbook_recursion() -> None:
    rng = np.random.default_rng(0)
    a = np.array([[0.9, 0.1], [0.0, 0.8]])
    w = np.diag([0.2, 0.1])
    c = rng.standard_normal((6, 2))
    q = np.diag(rng.uniform(0.5, 1.5, 6))
    p = w.copy()
    for _ in range(500):
        p_pred = a @ p @ a.T + w
        gain = p_pred @ c.T @ np.linalg.inv(c @ p_pred @ c.T + q)
        p = (np.eye(2) - gain @ c) @ p_pred
    np.testing.assert_allclose(steady_state_gain(a, w, c, q), gain, atol=1e-8)


def test_steady_state_gain_survives_a_singular_process_noise() -> None:
    """Displacement is an exact sum of velocity, so its noise is not independent."""
    a = np.array([[0.9, 0.0], [0.9, 1.0]])
    w = np.array([[1.0, 1.0], [1.0, 1.0]])
    c = np.random.default_rng(1).standard_normal((5, 2))
    gain = steady_state_gain(a, w, c, np.eye(5))
    assert np.all(np.isfinite(gain))


def test_delayed_shifts_within_trials_only() -> None:
    features = np.arange(1.0, 7.0)[:, None]
    groups = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(delayed(features, groups, 1)[:, 0], [0, 1, 2, 0, 4, 5])
    np.testing.assert_array_equal(delayed(features, groups, 2)[:, 0], [0, 0, 1, 0, 0, 4])
    assert delayed(features, groups, 0) is features


def test_running_integral_restarts_each_trial() -> None:
    values = np.ones((6, 1))
    groups = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(running_integral(values, groups)[:, 0], [1, 2, 3, 1, 2, 3])


def test_filter_decodes_a_linear_gaussian_system() -> None:
    features, targets, groups = simulate(0, noise=2.0)
    train = groups < 45
    decoder = KalmanDecoder()
    decoder.fit(features[train], targets[train], groups[train])
    prediction = decoder.predict(features[~train], groups[~train])
    assert prediction.shape == targets[~train].shape
    assert r2_score(targets[~train], prediction) > 0.85
    assert decoder.n_states == 2


def test_filter_beats_the_instantaneous_readout_when_channels_are_noisy() -> None:
    """The momentum prior is what the filter adds over a per-bin regression."""
    features, targets, groups = simulate(2, noise=6.0)
    train = groups < 45
    decoder = KalmanDecoder(lags=(0,), integral_state=False)
    decoder.fit(features[train], targets[train], groups[train])
    filtered = r2_score(targets[~train], decoder.predict(features[~train], groups[~train]))
    design = np.c_[features[train], np.ones(train.sum())]
    weights, *_ = np.linalg.lstsq(design, targets[train], rcond=None)
    instant = r2_score(targets[~train], np.c_[features[~train], np.ones((~train).sum())] @ weights)
    assert filtered > instant + 0.05


def test_delay_is_found_by_cross_validation() -> None:
    for lead in (0, 2):
        features, targets, groups = simulate(3, lead=lead)
        decoder = KalmanDecoder()
        decoder.fit(features, targets, groups)
        assert decoder.lag == lead


def test_a_single_delay_is_used_as_given() -> None:
    features, targets, groups = simulate(4)
    decoder = KalmanDecoder(lags=(3,))
    decoder.fit(features, targets, groups)
    assert decoder.lag == 3


def test_state_restarts_at_each_trial() -> None:
    features, targets, groups = simulate(5)
    decoder = KalmanDecoder()
    decoder.fit(features, targets, groups)
    last = groups == groups.max()
    together = decoder.predict(features, groups)[last]
    alone = decoder.predict(features[last], groups[last])
    np.testing.assert_allclose(together, alone)
