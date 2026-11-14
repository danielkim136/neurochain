import numpy as np
import pytest

from neurochain.adc import Adc
from neurochain.bridge import (
    degrade_crossings,
    degrade_sbp,
    sbp_floor,
    sbp_noise_power_ratio,
)
from neurochain.frontend.electrode import UTAH_PT_IN_VITRO
from neurochain.frontend.model import FrontEnd, design_lna

FREQS = np.linspace(0.0, 15_000.0, 1501)
FLAT = np.full(FREQS.size, 100.0 / 7200.0)  # 10 µV rms over a 7.2 kHz band


def test_thinning_keeps_the_right_share_and_adds_the_right_rate() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(4.0, size=(20_000, 3)).astype(float)
    assert degrade_crossings(counts, 1.0, 0.0, 0.05, rng) == pytest.approx(counts)
    thinned = degrade_crossings(counts, 0.4, 0.0, 0.05, rng)
    assert thinned.mean() == pytest.approx(0.4 * counts.mean(), rel=0.02)
    assert np.all(thinned <= counts)
    noisy = degrade_crossings(counts, 1.0, 10.0, 0.05, rng)
    assert (noisy - counts).mean() == pytest.approx(10.0 * 0.05, rel=0.03)
    with pytest.raises(ValueError, match="between 0 and 1"):
        degrade_crossings(counts, 1.2, 0.0, 0.05, rng)


def test_thinning_poisson_counts_leaves_a_root_p_correlation() -> None:
    rng = np.random.default_rng(1)
    counts = rng.poisson(6.0, size=(200_000, 1)).astype(float)
    thinned = degrade_crossings(counts, 0.5, 0.0, 0.05, rng)
    assert np.corrcoef(counts[:, 0], thinned[:, 0])[0, 1] == pytest.approx(np.sqrt(0.5), abs=0.01)


def test_sbp_is_unchanged_without_added_noise() -> None:
    rng = np.random.default_rng(0)
    sbp = 10.0 + rng.random((100, 2))
    out = degrade_sbp(sbp, 1.0, np.array([10.0, 10.0]), n_effective=70.0, rng=rng)
    np.testing.assert_array_equal(out, sbp)
    with pytest.raises(ValueError, match="less noise"):
        degrade_sbp(sbp, 0.9, np.array([10.0, 10.0]), n_effective=70.0, rng=rng)


def test_added_noise_raises_the_level_in_quadrature_and_adds_jitter() -> None:
    rng = np.random.default_rng(0)
    sbp = np.full((100_000, 1), 12.0)
    floor = np.array([10.0])
    out = degrade_sbp(sbp, 2.0, floor, n_effective=70.0, rng=rng)
    assert out.mean() == pytest.approx(np.sqrt(12.0**2 + 10.0**2), rel=1e-3)
    assert out.std() == pytest.approx(10.0 * np.sqrt((np.pi / 2 - 1) / 70.0), rel=0.02)


def test_floor_is_recovered_from_noise_only_bins() -> None:
    # Each bin is the mean of |x| over 70 independent Gaussian samples.
    rng = np.random.default_rng(0)
    rho = 12.5
    sbp = np.abs(rho * rng.standard_normal((5000, 2, 70))).mean(axis=2)
    estimate = sbp_floor(sbp, n_effective=70.0)
    np.testing.assert_allclose(estimate, rho * np.sqrt(2 / np.pi), rtol=0.02)
    # Without the correction the 5th percentile would read 15 % low.
    assert np.percentile(sbp, 5, axis=0).mean() < 0.88 * estimate.mean()


def frontend(i_total_a: float, *, narrow: bool) -> FrontEnd:
    lna = (
        design_lna(f_high_hz=1000.0, f_low_hz=300.0, i_total_a=i_total_a)
        if narrow
        else design_lna(i_total_a=i_total_a)
    )
    return FrontEnd(lna, UTAH_PT_IN_VITRO)


def test_noise_ratio_grows_as_current_and_bits_fall() -> None:
    adc = Adc(fs_hz=2000.0, bits=10)
    high = sbp_noise_power_ratio(frontend(10e-6, narrow=True), adc, FREQS, FLAT)
    low = sbp_noise_power_ratio(frontend(0.1e-6, narrow=True), adc, FREQS, FLAT)
    coarse = sbp_noise_power_ratio(
        frontend(10e-6, narrow=True), Adc(fs_hz=2000.0, bits=5), FREQS, FLAT
    )
    assert 1.0 < high < low
    assert coarse > high
    # A hundredth of the current is a hundred times the OTA's noise power; the
    # feedback resistor's and the quantiser's shares do not change, so the total
    # added noise grows by less.
    assert 10.0 < (low - 1.0) / (high - 1.0) < 100.0


def test_digital_band_counts_only_in_band_noise() -> None:
    adc = Adc(fs_hz=30_000.0, bits=10)
    wide = frontend(1e-6, narrow=False)
    ratio = sbp_noise_power_ratio(wide, adc, FREQS, FLAT, digital_band_hz=(300.0, 1000.0))
    background = 700.0 * 100.0 / 7200.0
    front_end = (wide.band_noise(300.0, 1000.0).total_v * 1e6) ** 2
    quantisation = adc.lsb_uv**2 / 12 * 700.0 / 15_000.0
    assert ratio == pytest.approx(1.0 + (front_end + quantisation) / background, rel=0.01)
