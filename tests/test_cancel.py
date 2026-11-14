"""Blanking and the adaptive template canceller."""

import numpy as np
import pytest

from neurochain.stim import cancel

PERIOD = 20
TAPS = 16
SHAPE = np.array([0, 900, 300, -400, -150, -60, 20, 10, 4, 2, 1, 0, 0, 0, 0, 0], dtype=float)


def record(
    n_pulses: int = 400, noise: float = 5.0, seed: int = 0, drift: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise plus the same artifact after every pulse; returns codes, noise codes, onsets."""
    rng = np.random.default_rng(seed)
    n = n_pulses * PERIOD + 40
    background = rng.normal(0.0, noise, n)
    onsets = 20 + PERIOD * np.arange(n_pulses)
    x = background.copy()
    for p, onset in enumerate(onsets):
        x[onset : onset + TAPS] += SHAPE * (1.0 + drift * p / n_pulses)
    return np.rint(x).astype(np.int64), np.rint(background).astype(np.int64), onsets


def test_blank_fills_and_masks() -> None:
    codes = np.arange(1, 21, dtype=np.int64)
    zeroed, valid = cancel.blank(codes, np.array([5, 15]), 3)
    assert zeroed[5:8].tolist() == [0, 0, 0] and zeroed[8] == 9
    assert valid.sum() == 14 and not valid[15:18].any()
    held, _ = cancel.blank(codes, np.array([5, 18]), 3, fill="hold")
    assert held[5:8].tolist() == [5, 5, 5]
    assert held[18:].tolist() == [18, 18]
    assert codes[5] == 6, "the input is not modified"


def test_template_removes_a_repeating_artifact() -> None:
    codes, background, onsets = record()
    out, valid = cancel.cancel_template(
        codes, onsets, n_taps=TAPS, shift=4, code_min=-2048, code_max=2047
    )
    values = out / 4.0
    assert not valid[onsets[0] : onsets[0] + TAPS].any(), "the first pulse loads the template"
    late = np.zeros(codes.size, dtype=bool)
    late[onsets[50] :] = True
    window = np.zeros(codes.size, dtype=bool)
    for onset in onsets:
        window[onset : onset + TAPS] = True
    inside = late & window & valid
    residual = (values - background)[inside]
    # What is left is the template's own noise: variance mu / (2 - mu) of the background's.
    expected = 5.0 * np.sqrt(cancel.noise_gain(4) - 1.0)
    assert residual.std() == pytest.approx(expected, rel=0.15)
    assert abs(residual.mean()) < 0.2
    assert np.array_equal(values[late & ~window], codes[late & ~window])
    assert np.abs(codes[inside]).max() > 800 and np.abs(values[inside]).max() < 40


@pytest.mark.parametrize("shift", [1, 3, 5])
def test_noise_gain_formula(shift: int) -> None:
    codes, _, onsets = record(n_pulses=3000, noise=20.0, seed=shift)
    out, _ = cancel.cancel_template_float(
        codes.astype(float), onsets, n_taps=TAPS, rate=2.0**-shift, low=-1e9, high=1e9
    )
    plain = codes.astype(float)
    quiet = np.concatenate([np.arange(o + 11, o + TAPS) for o in onsets[200:]])
    ratio = out[quiet].var() / plain[quiet].var()
    assert ratio == pytest.approx(cancel.noise_gain(shift), rel=0.05)
    assert cancel.noise_gain(1) == pytest.approx(4.0 / 3.0)


def test_integer_canceller_matches_floating_point() -> None:
    codes, _, onsets = record()
    fixed, valid = cancel.cancel_template(
        codes, onsets, n_taps=TAPS, shift=4, code_min=-2048, code_max=2047, frac_bits=6
    )
    floating, valid_float = cancel.cancel_template_float(
        codes.astype(float), onsets, n_taps=TAPS, rate=1 / 16, low=-2048, high=2047
    )
    assert np.array_equal(valid, valid_float)
    assert np.abs(fixed / 4.0 - floating).max() < 0.6
    coarse, _ = cancel.cancel_template(
        codes, onsets, n_taps=TAPS, shift=4, code_min=-2048, code_max=2047, out_frac_bits=0
    )
    assert np.abs(coarse - floating).std() > 1.5 * np.abs(fixed / 4.0 - floating).std()


def test_clipped_samples_are_blanked_not_subtracted() -> None:
    codes, _, onsets = record()
    clipped = np.clip(codes, -512, 511)
    out, valid = cancel.cancel_template(
        clipped, onsets, n_taps=TAPS, shift=4, code_min=-512, code_max=511
    )
    at_rail = onsets[10:] + 1
    assert not valid[at_rail].any()
    assert not out[at_rail].any()
    assert valid[onsets[10:] + 2].all()


def test_adaptation_follows_a_drifting_artifact() -> None:
    codes, background, onsets = record(n_pulses=600, drift=0.5)
    adaptive, _ = cancel.cancel_template(
        codes, onsets, n_taps=TAPS, shift=3, code_min=-4096, code_max=4095
    )
    frozen = codes.astype(float)
    for onset in onsets:
        frozen[onset : onset + TAPS] -= SHAPE
    last = slice(onsets[-100], onsets[-1] + TAPS)
    error_adaptive = (adaptive / 4.0 - background)[last]
    error_frozen = (frozen - background)[last]
    assert np.sqrt(np.mean(error_adaptive**2)) < 0.1 * np.sqrt(np.mean(error_frozen**2))


def test_batches_share_one_set_of_onsets() -> None:
    codes, _, onsets = record(n_pulses=50)
    stacked = np.stack([codes, 2 * codes])
    out, valid = cancel.cancel_template(
        stacked, onsets, n_taps=TAPS, shift=4, code_min=-5000, code_max=5000
    )
    single, single_valid = cancel.cancel_template(
        codes, onsets, n_taps=TAPS, shift=4, code_min=-5000, code_max=5000
    )
    assert np.array_equal(out[0], single) and np.array_equal(valid[0], single_valid)
    assert out.shape == stacked.shape


def test_canceller_rejects_bad_settings() -> None:
    codes, _, onsets = record(n_pulses=5)
    with pytest.raises(ValueError, match="out_frac_bits"):
        cancel.cancel_template(
            codes, onsets, n_taps=TAPS, shift=4, code_min=-9, code_max=9, frac_bits=1
        )


def test_masked_mean_abs() -> None:
    x = np.array([1.0, -3.0, 5.0, -7.0, 2.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0])
    valid = np.ones(12, dtype=bool)
    valid[[1, 3]] = False
    valid[4:8] = False
    sbp, share = cancel.masked_mean_abs(x, valid, 4)
    assert sbp.tolist() == [3.0, 3.0, 9.0], "an empty bin repeats the one before"
    assert share.tolist() == [0.5, 0.0, 1.0]
    full, _ = cancel.masked_mean_abs(x, np.ones(12, dtype=bool), 4)
    assert full.tolist() == [4.0, 2.0, 9.0]


def test_hold_last_valid() -> None:
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [9.0, 8.0, 7.0, 6.0, 5.0]])
    valid = np.array([[True, False, False, True, False], [False, True, True, False, True]])
    held = cancel.hold_last_valid(x, valid)
    assert held[0].tolist() == [1.0, 1.0, 1.0, 4.0, 4.0]
    assert held[1].tolist() == [9.0, 8.0, 7.0, 7.0, 5.0]
