from dataclasses import dataclass, field

import numpy as np

from neurochain.decode.evaluate import (
    bootstrap_r2,
    bootstrap_r2_difference,
    fit_and_predict,
    held_out_predictions,
    r2_score,
    standardise,
)
from neurochain.decode.ridge import RidgeDecoder, WienerDecoder


@dataclass
class Spy:
    """Predicts the training mean and remembers which trials it was shown."""

    seen: list[set[int]] = field(default_factory=list)
    mean: float = 0.0

    def fit(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> None:
        self.seen.append(set(groups.tolist()))
        self.mean = float(targets.mean())

    def predict(self, features: np.ndarray, groups: np.ndarray) -> np.ndarray:
        self.seen.append(set(groups.tolist()))
        return np.full((len(features), 1), self.mean)


def linear(seed: int = 0, n_trials: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_trials * 25
    features = rng.standard_normal((n, 6))
    targets = features @ rng.standard_normal((6, 2)) + 0.5 * rng.standard_normal((n, 2))
    return features, targets, np.repeat(np.arange(n_trials), 25)


def test_standardise_uses_the_training_statistics() -> None:
    train = np.array([[0.0], [2.0]])
    scaled_train, scaled_other = standardise(train, np.array([[4.0]]))
    np.testing.assert_allclose(scaled_train[:, 0], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(scaled_other[:, 0], [3.0], atol=1e-6)


def test_no_decoder_sees_the_trials_it_predicts() -> None:
    features, targets, groups = linear()
    spy = Spy()
    predictions = held_out_predictions(lambda: spy, features, targets[:, :1], groups, n_folds=4)
    assert predictions.shape == (len(groups), 1)
    assert len(spy.seen) == 8
    predicted: set[int] = set()
    for trained, tested in zip(spy.seen[::2], spy.seen[1::2], strict=True):
        assert not trained & tested
        assert max(tested) - min(tested) == len(tested) - 1  # a contiguous block
        predicted |= tested
    assert predicted == set(groups.tolist())


def test_r2_score_limits() -> None:
    _, targets, _ = linear()
    assert r2_score(targets, targets) == 1.0
    assert abs(r2_score(targets, np.tile(targets.mean(axis=0), (len(targets), 1)))) < 1e-12


def test_ridge_decoder_fits_through_the_shared_interface() -> None:
    features, targets, groups = linear()
    train = groups < 30
    prediction = fit_and_predict(
        lambda: RidgeDecoder(n_lags=1),
        (features[train], targets[train], groups[train]),
        features[~train],
        groups[~train],
    )
    assert r2_score(targets[~train], prediction) > 0.9


def test_bootstrap_interval_brackets_the_score_and_is_seeded() -> None:
    _, targets, groups = linear()
    predictions = targets + np.random.default_rng(1).standard_normal(targets.shape)
    r2, low, high = bootstrap_r2(targets, predictions, groups, n_boot=400, seed=3)
    assert r2 == r2_score(targets, predictions)
    assert low < r2 < high
    assert bootstrap_r2(targets, predictions, groups, n_boot=400, seed=3) == (r2, low, high)
    assert bootstrap_r2(targets, predictions, groups, n_boot=400, seed=4) != (r2, low, high)


def test_bootstrap_interval_narrows_with_more_trials() -> None:
    widths = []
    for n_trials in (20, 320):
        _, targets, groups = linear(2, n_trials)
        predictions = targets + np.random.default_rng(5).standard_normal(targets.shape)
        _, low, high = bootstrap_r2(targets, predictions, groups, n_boot=400)
        widths.append(high - low)
    assert widths[1] < 0.5 * widths[0]


def test_paired_difference_is_tighter_than_two_separate_intervals() -> None:
    _, targets, groups = linear(3)
    rng = np.random.default_rng(6)
    # Trials differ in difficulty: a shared error that both decoders make.
    shared = np.repeat(rng.uniform(0.2, 2.0, groups.max() + 1), 25)[:, None]
    shared = shared * rng.standard_normal(targets.shape)
    good = targets + shared
    worse = good + 0.4 * rng.standard_normal(targets.shape)
    difference, low, high = bootstrap_r2_difference(targets, good, worse, groups, n_boot=400)
    assert difference > 0
    assert low > 0
    _, good_low, good_high = bootstrap_r2(targets, good, groups, n_boot=400)
    assert high - low < 0.5 * (good_high - good_low)
    assert bootstrap_r2_difference(targets, good, good, groups, n_boot=50) == (0.0, 0.0, 0.0)


def test_wiener_filter_with_one_penalty_is_the_ridge_readout() -> None:
    features, targets, groups = linear()
    ridge, wiener = RidgeDecoder(n_lags=3, alpha=0.5), WienerDecoder(n_lags=3, alphas=(0.5,))
    for decoder in (ridge, wiener):
        decoder.fit(features, targets, groups)
    assert wiener.alpha == 0.5
    np.testing.assert_allclose(
        wiener.predict(features, groups), ridge.predict(features, groups), atol=1e-8
    )


def test_wiener_filter_raises_its_penalty_when_weights_outnumber_the_evidence() -> None:
    """Few trials and long history: the smallest penalty overfits and is not chosen."""
    rng = np.random.default_rng(7)
    n_trials, length = 16, 25
    features = rng.standard_normal((n_trials * length, 12))
    targets = features[:, :1] + 2.0 * rng.standard_normal((n_trials * length, 1))
    groups = np.repeat(np.arange(n_trials), length)
    train = groups < 12
    tuned, loose = WienerDecoder(n_lags=10), WienerDecoder(n_lags=10, alphas=(1e-4,))
    for decoder in (tuned, loose):
        decoder.fit(features[train], targets[train], groups[train])
    assert tuned.alpha > 0.01
    score = [
        r2_score(targets[~train], d.predict(features[~train], groups[~train]))
        for d in (tuned, loose)
    ]
    assert score[0] > score[1] + 0.1
