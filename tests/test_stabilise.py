"""The running stabiliser and the latent alignment, on synthetic data."""

import numpy as np
import pytest

from neurochain.align import aligned_axes, principal_axes, procrustes_rotation
from neurochain.stabilise import (
    normalise,
    operations_per_bin,
    running_average,
    running_statistics,
)


def test_running_average_matches_the_register_recursion() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3)) + np.array([0.0, 5.0, -2.0])
    start = np.array([1.0, 2.0, 3.0])
    shift = 4
    expected = np.empty_like(x)
    register = start.copy()
    for n in range(x.shape[0]):
        expected[n] = register  # the value before this bin is used
        register = register + (x[n] - register) / 2**shift
    assert np.allclose(running_average(x, start, shift), expected)


def test_running_statistics_are_causal() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 2))
    changed = x.copy()
    changed[120:] += 50.0
    mean, mad = np.zeros(2), np.ones(2)
    a = running_statistics(x, mean, mad, 3)
    b = running_statistics(changed, mean, mad, 3)
    # Nothing before bin 120, nor the estimate used at bin 120, can differ.
    assert np.array_equal(a[0][:121], b[0][:121])
    assert np.array_equal(a[1][:121], b[1][:121])
    assert not np.allclose(a[0][121:], b[0][121:])


def test_modes_agree_when_nothing_has_changed() -> None:
    rng = np.random.default_rng(2)
    x = 3.0 + 2.0 * rng.standard_normal((20_000, 4))
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mad = np.abs(x - mean).mean(axis=0)
    fixed = normalise(x, mean, scale, mad, mode="fixed")
    assert np.allclose(fixed.mean(axis=0), 0.0, atol=1e-6)
    for mode in ("mean", "zscore"):
        running = normalise(x, mean, scale, mad, mode=mode, shift=10)  # type: ignore[arg-type]
        assert np.corrcoef(fixed.ravel(), running.ravel())[0, 1] > 0.99
        assert running.std() == pytest.approx(1.0, abs=0.05)


def test_zscore_undoes_an_offset_and_a_gain_change_and_mean_only_the_offset() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((30_000, 2))
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mad = np.abs(x - mean).mean(axis=0)
    aged = 0.5 * x + 4.0  # half the modulation on a shifted level
    tail = slice(10_000, None)  # after the registers have settled (shift 8: 256 bins)
    fixed = normalise(aged, mean, scale, mad, mode="fixed")
    centred = normalise(aged, mean, scale, mad, mode="mean", shift=8)
    zscored = normalise(aged, mean, scale, mad, mode="zscore", shift=8)
    assert abs(fixed[tail].mean()) > 3.0
    assert abs(centred[tail].mean()) < 0.05
    assert centred[tail].std() == pytest.approx(0.5, rel=0.05)
    assert abs(zscored[tail].mean()) < 0.05
    assert zscored[tail].std() == pytest.approx(1.0, rel=0.08)


def test_gain_is_limited_on_a_channel_that_has_gone_quiet() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((20_000, 1))
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mad = np.abs(x - mean).mean(axis=0)
    quiet = 0.01 * x
    out = normalise(quiet, mean, scale, mad, mode="zscore", shift=6, max_gain=4.0)
    assert out[5000:].std() == pytest.approx(0.04, rel=0.1)  # 4 times 0.01, not 1


def test_gated_registers_keep_two_conditions_apart() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((40_000, 1))
    mean, scale = np.zeros(1), np.ones(1)
    mad = np.full(1, np.sqrt(2.0 / np.pi))
    gate = (np.arange(x.shape[0]) // 20) % 2  # alternate every 20 bins
    shifted = x + 6.0 * gate[:, np.newaxis]
    tail = slice(20_000, None)
    plain = normalise(shifted, mean, scale, mad, mode="zscore", shift=8)
    gated = normalise(shifted, mean, scale, mad, mode="zscore", shift=8, gate=gate)
    # One set of registers settles between the two levels and leaves the step in.
    assert abs(plain[tail][gate[tail] == 1].mean() - plain[tail][gate[tail] == 0].mean()) > 1.0
    assert abs(gated[tail][gate[tail] == 1].mean()) < 0.1
    assert abs(gated[tail][gate[tail] == 0].mean()) < 0.1


def test_warm_start_and_bad_arguments() -> None:
    x = np.full((50, 1), 7.0)
    mean, scale, mad = np.zeros(1), np.ones(1), np.ones(1)
    cold = normalise(x, mean, scale, mad, mode="mean", shift=10)
    warm = normalise(x, mean, scale, mad, mode="mean", shift=10, start=(np.full(1, 7.0), mad))
    assert cold[0, 0] == pytest.approx(7.0)
    assert np.allclose(warm, 0.0)
    with pytest.raises(ValueError, match="unknown mode"):
        normalise(x, mean, scale, mad, mode="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shift"):
        running_average(x, mean, -1)


def test_operation_counts() -> None:
    assert operations_per_bin("fixed") == 1
    assert operations_per_bin("mean") == 3
    assert operations_per_bin("zscore", quotient_bits=12) == 17


def test_principal_axes_are_orthonormal_and_ordered() -> None:
    rng = np.random.default_rng(6)
    latents = rng.standard_normal((2000, 3)) * np.array([5.0, 2.0, 1.0])
    mixing = np.linalg.qr(rng.standard_normal((12, 3)))[0]
    x = latents @ mixing.T + 0.05 * rng.standard_normal((2000, 12))
    axes = principal_axes(x, 3)
    assert axes.shape == (12, 3)
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-9)
    assert np.allclose(np.abs(axes.T @ mixing), np.eye(3), atol=0.05)
    with pytest.raises(ValueError):
        principal_axes(x, 13)


def test_procrustes_recovers_a_rotation() -> None:
    rng = np.random.default_rng(7)
    reference = np.linalg.qr(rng.standard_normal((20, 4)))[0]
    rotation = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    found = procrustes_rotation(reference @ rotation.T, reference)
    assert np.allclose(found, rotation, atol=1e-9)
    assert np.allclose(found.T @ found, np.eye(4), atol=1e-9)


def test_alignment_restores_the_latents_after_channels_change() -> None:
    # Day 2 records the same three latent variables through electrodes whose
    # gains have changed and most of which now see other neurons. A fixed
    # projection reads garbage on the changed channels; re-finding the axes
    # does not.
    rng = np.random.default_rng(8)
    mixing = rng.standard_normal((40, 3))
    day1 = rng.standard_normal((4000, 3)) * np.array([3.0, 2.0, 1.0])
    day2 = rng.standard_normal((4000, 3)) * np.array([3.0, 2.0, 1.0])
    changed = mixing * rng.uniform(0.5, 1.5, size=(40, 1))
    changed[:24] = rng.standard_normal((24, 3))  # 24 channels pick up new neurons
    x1 = day1 @ mixing.T + 0.3 * rng.standard_normal((4000, 40))
    x2 = day2 @ changed.T + 0.3 * rng.standard_normal((4000, 40))
    reference = principal_axes(x1, 3)
    readout = np.linalg.lstsq(x1 @ reference, day1, rcond=None)[0]

    def error(axes: np.ndarray) -> float:
        return float(np.mean((x2 @ axes @ readout - day2) ** 2) / np.mean(day2**2))

    assert error(reference) > 0.3
    assert error(aligned_axes(x2, reference)) < 0.5 * error(reference)
