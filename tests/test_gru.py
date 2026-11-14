import numpy as np
import pytest

from neurochain.decode.evaluate import r2_score
from neurochain.decode.gru import GruDecoder
from neurochain.decode.ridge import RidgeDecoder


@pytest.fixture(scope="module")
def nonlinear() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eight channels follow a smooth latent; the target is its square."""
    rng = np.random.default_rng(0)
    n_trials, length = 80, 40
    n = n_trials * length
    latent = np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same") * 3
    features = np.outer(latent, rng.standard_normal(8)) + 0.3 * rng.standard_normal((n, 8))
    targets = (latent**2)[:, None]
    return features, targets, np.repeat(np.arange(n_trials), length)


def small(seed: int = 0) -> GruDecoder:
    return GruDecoder(hidden=16, epochs=40, patience=40, dropout=0.0, chunk=40, seed=seed)


def test_gru_learns_what_a_linear_readout_cannot(
    nonlinear: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    features, targets, groups = nonlinear
    train = groups < 60
    gru, ridge = small(), RidgeDecoder(n_lags=3)
    scores = []
    for decoder in (gru, ridge):
        decoder.fit(features[train], targets[train], groups[train])
        prediction = decoder.predict(features[~train], groups[~train])
        assert prediction.shape == targets[~train].shape
        scores.append(r2_score(targets[~train], prediction))
    assert scores[0] > 0.6
    assert scores[1] < 0.2


def test_gru_is_reproducible_and_seeded(
    nonlinear: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    features, targets, groups = nonlinear
    use = groups < 20
    outputs = []
    for seed in (0, 0, 1):
        decoder = GruDecoder(hidden=8, epochs=3, seed=seed)
        decoder.fit(features[use], targets[use], groups[use])
        outputs.append(decoder.predict(features[use], groups[use]))
    np.testing.assert_allclose(outputs[0], outputs[1])
    assert not np.allclose(outputs[0], outputs[2])


def test_gru_is_causal_and_restarts_each_trial(
    nonlinear: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    features, targets, groups = nonlinear
    use = groups < 20
    decoder = GruDecoder(hidden=8, epochs=3)
    decoder.fit(features[use], targets[use], groups[use])
    trial = groups == 30
    before = decoder.predict(features[trial], groups[trial])
    changed = features[trial].copy()
    changed[25:] += 5.0
    after = decoder.predict(changed, groups[trial])
    np.testing.assert_allclose(before[:25], after[:25], atol=1e-6)
    assert not np.allclose(before[25:], after[25:])
    pair = (groups == 29) | (groups == 30)
    np.testing.assert_allclose(
        decoder.predict(features[pair], groups[pair])[40:], before, atol=1e-6
    )
