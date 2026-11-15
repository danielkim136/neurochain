"""The aging model: impedance rise, noise that follows it, amplitude decline."""

import numpy as np
import pytest

from neurochain.aging import WEEKS_PER_MONTH, AgingModel, encapsulated
from neurochain.frontend.electrode import UTAH_PT_IN_VITRO
from neurochain.frontend.physics import BODY_TEMPERATURE_K


@pytest.mark.parametrize("mode", ["series", "interface"])
@pytest.mark.parametrize("factor", [1.5, 2.48, 8.0])
def test_encapsulation_hits_the_impedance_ratio_at_1_khz(mode: str, factor: float) -> None:
    aged = encapsulated(UTAH_PT_IN_VITRO, factor, mode)  # type: ignore[arg-type]
    assert aged.magnitude_ohm(1000.0) == pytest.approx(
        factor * UTAH_PT_IN_VITRO.magnitude_ohm(1000.0), rel=1e-6
    )


def test_a_sheath_only_changes_the_series_resistance() -> None:
    aged = encapsulated(UTAH_PT_IN_VITRO, 2.48, "series")
    assert aged.rs_ohm > UTAH_PT_IN_VITRO.rs_ohm
    assert (aged.rct_ohm, aged.q_s_sn, aged.n) == (
        UTAH_PT_IN_VITRO.rct_ohm,
        UTAH_PT_IN_VITRO.q_s_sn,
        UTAH_PT_IN_VITRO.n,
    )


def test_interface_scaling_keeps_the_phase() -> None:
    aged = encapsulated(UTAH_PT_IN_VITRO, 3.0, "interface")
    for freq in (100.0, 1000.0, 5000.0):
        assert aged.phase_deg(freq) == pytest.approx(UTAH_PT_IN_VITRO.phase_deg(freq))


def test_a_sheath_is_the_noisier_way_to_raise_the_impedance() -> None:
    # Thermal noise follows the real part of Z; a series resistor is all real.
    noise = {
        mode: encapsulated(UTAH_PT_IN_VITRO, 2.48, mode).thermal_noise_rms_v(  # type: ignore[arg-type]
            300.0, 7500.0, BODY_TEMPERATURE_K
        )
        for mode in ("series", "interface")
    }
    fresh = UTAH_PT_IN_VITRO.thermal_noise_rms_v(300.0, 7500.0, BODY_TEMPERATURE_K)
    assert noise["series"] > noise["interface"] > fresh
    # Scaling every element by k scales Re{Z} by k, so the noise by sqrt(k).
    assert noise["interface"] == pytest.approx(fresh * np.sqrt(2.48), rel=1e-3)


def test_factor_one_returns_the_electrode_and_below_one_is_refused() -> None:
    assert encapsulated(UTAH_PT_IN_VITRO, 1.0) is UTAH_PT_IN_VITRO
    with pytest.raises(ValueError, match="raises"):
        encapsulated(UTAH_PT_IN_VITRO, 0.5)


def test_encapsulation_is_95_percent_done_at_its_stated_time() -> None:
    model = AgingModel(impedance_factor_final=3.0, encapsulation_weeks=2.0)
    assert model.impedance_factor(0.0) == 1.0
    assert model.settled_share(2.0) == pytest.approx(1.0 - np.exp(-3.0))
    assert model.impedance_factor(2.0) == pytest.approx(1.0 + 2.0 * (1.0 - np.exp(-3.0)))
    assert model.impedance_factor(52.0) == pytest.approx(3.0)
    assert model.background_factor(52.0) == 1.0
    assert model.electrode(UTAH_PT_IN_VITRO, 52.0).magnitude_ohm(1000.0) == pytest.approx(
        3.0 * UTAH_PT_IN_VITRO.magnitude_ohm(1000.0), rel=1e-6
    )


def test_a_channel_at_the_mean_rate_loses_the_stated_share_each_month() -> None:
    model = AgingModel(amplitude_decline_per_month=0.024, decline_spread=0.0)
    scale = model.amplitude_scale(np.array([0.0, WEEKS_PER_MONTH, 12 * WEEKS_PER_MONTH]), 5, seed=3)
    assert scale.shape == (3, 5)
    assert np.allclose(scale[0], 1.0)
    assert np.allclose(scale[1], 0.976)
    assert np.allclose(scale[2], 0.976**12)


def test_channels_differ_decline_monotonically_and_keep_their_rate() -> None:
    model = AgingModel()
    weeks = np.array([0.0, 13.0, 39.0, 104.0])
    scale = model.amplitude_scale(weeks, 400, seed=1)
    assert np.all(np.diff(scale, axis=0) <= 0.0)
    assert np.all((scale > 0.0) & (scale <= 1.0))
    assert scale[-1].std() > 0.1
    # The same seed gives the same channels the same rates, whatever the weeks.
    again = model.amplitude_scale(np.array([39.0]), 400, seed=1)
    assert np.array_equal(again[0], scale[2])
    rates = model.decline_rates_per_week(20_000, seed=2)
    expected = -np.log(1.0 - 0.024) / WEEKS_PER_MONTH
    assert rates.mean() == pytest.approx(expected, rel=0.03)
    assert rates.std() / rates.mean() == pytest.approx(1.0, rel=0.05)


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError):
        AgingModel(impedance_factor_final=0.9)
    with pytest.raises(ValueError):
        AgingModel(amplitude_decline_per_month=1.0)
    with pytest.raises(ValueError):
        AgingModel(decline_spread=-1.0)
