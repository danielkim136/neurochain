"""The large-signal amplifier model, on its own and against ngspice."""

import numpy as np
import pytest

from neurochain.analysis import u3
from neurochain.frontend import spice
from neurochain.frontend.emulator import reference_gain
from neurochain.frontend.model import FrontEnd
from neurochain.frontend.saturation import (
    SaturatingAmplifier,
    pulse_kernel,
    pulse_response_uv,
    recovery_time_s,
)
from neurochain.power import PowerModel
from neurochain.stim.pulse import Pulse

needs_ngspice = pytest.mark.skipif(not spice.ngspice_available(), reason="ngspice not installed")


@pytest.fixture(scope="module")
def wide() -> FrontEnd:
    return u3.wideband_frontend(10e-6, PowerModel())


@pytest.fixture(scope="module")
def narrow() -> FrontEnd:
    return u3.narrowband_frontend(0.25e-6, PowerModel())


@pytest.mark.parametrize("which", ["wide", "narrow"])
def test_small_signals_see_the_linear_gain(which: str, request: pytest.FixtureRequest) -> None:
    frontend: FrontEnd = request.getfixturevalue(which)
    amplifier = SaturatingAmplifier.from_frontend(frontend)
    f0 = 600.0
    t = np.arange(0.0, 0.03, 2e-6)
    v_out, v_x = amplifier.simulate(t, 1e-5 * np.sin(2 * np.pi * f0 * t), t)
    settled = t > 0.02
    measured = np.sqrt(2.0) * v_out[settled].std() / 1e-5
    expected = float(np.abs(frontend.without_electrode().signal_gain(np.array([f0])))[0])
    assert measured == pytest.approx(expected, rel=0.01)
    # A virtual ground: the summing node barely moves.
    assert np.abs(v_x).max() < 1e-4


def test_output_stops_at_the_rails_and_the_summing_node_floats(narrow: FrontEnd) -> None:
    amplifier = SaturatingAmplifier.from_frontend(narrow)
    t = np.arange(0.0, 4e-4, 1e-6)
    v_out, v_x = amplifier.simulate(np.array([0.0, 2e-6]), np.array([0.0, 0.05]), t)
    assert np.abs(v_out).max() == pytest.approx(amplifier.swing_v, rel=0.01)
    # Open loop: the node sits near C1 v_in / C_sum instead of at zero.
    assert v_x[-1] > 0.03
    assert amplifier.linear_input_v == pytest.approx(8e-3)


def test_charge_moved_during_saturation_leaks_at_the_slow_pole(wide: FrontEnd) -> None:
    """After a one-sided pulse that rails the output, the offset decays with R2 C2."""
    amplifier = SaturatingAmplifier.from_frontend(wide)
    tau = amplifier.r2_ohm * amplifier.c2_f
    assert -1.0 / amplifier.poles().real.max() == pytest.approx(tau, rel=0.02)
    t_bp = np.array([0.0, 2e-6, 3e-4, 3.02e-4])
    v_bp = np.array([0.0, 0.03, 0.03, 0.0])
    v_out, _ = amplifier.simulate(t_bp, v_bp, np.array([0.5, 1.5]) * tau)
    assert abs(v_out[0]) > 1e-4
    assert v_out[1] / v_out[0] == pytest.approx(np.exp(-1.0), rel=0.02)


def test_a_balanced_pulse_leaves_an_offset_only_if_it_saturates(wide: FrontEnd) -> None:
    """Linear: the two phases cancel. Railed: they do not, and the offset lasts."""
    amplifier = SaturatingAmplifier.from_frontend(wide)
    t_bp, wave = Pulse(imbalance=0.0).breakpoints()
    gain = reference_gain(wide)
    late = np.array([0.05])
    linear = pulse_response_uv(amplifier, t_bp, wave, 3000.0, late, gain=gain)[0]
    railed = pulse_response_uv(amplifier, t_bp, wave, 30_000.0, late, gain=gain)[0]
    assert abs(linear) < 0.05
    assert 3.0 < abs(railed) < 20.0


def test_the_narrow_amplifier_rings(narrow: FrontEnd) -> None:
    """Its two corners are close: the closed loop has a complex pole pair."""
    amplifier = SaturatingAmplifier.from_frontend(narrow)
    poles = amplifier.poles()
    assert np.all(poles.real < 0) and abs(poles[0].imag) > 0
    assert -1e3 / poles[0].real == pytest.approx(0.41, abs=0.01)
    assert abs(poles[0].imag) / (2 * np.pi) == pytest.approx(271.0, abs=3.0)
    t_bp, wave = Pulse(imbalance=0.0).breakpoints()
    t = np.arange(8e-4, 6e-3, 1e-5)
    y = pulse_response_uv(amplifier, t_bp, wave, 3000.0, t, gain=reference_gain(narrow))
    assert np.count_nonzero(np.diff(np.sign(y))) >= 2


def test_the_narrow_chain_recovers_more_slowly(wide: FrontEnd, narrow: FrontEnd) -> None:
    pulse = Pulse()
    t_bp, wave = pulse.breakpoints()
    t = np.arange(0.0, 0.02, 2e-6)
    recovery = {}
    for name, frontend in (("wide", wide), ("narrow", narrow)):
        y = pulse_response_uv(
            SaturatingAmplifier.from_frontend(frontend), t_bp, wave, 3000.0, t,
            gain=reference_gain(frontend),
        )  # fmt: skip
        recovery[name] = recovery_time_s(t, y, pulse.duration_s, 13.0)
    assert recovery["narrow"] > 5 * recovery["wide"]
    assert 1e-3 < recovery["narrow"] < 4e-3


def test_recovery_time_definition() -> None:
    t = np.arange(10) * 1e-3
    y = np.array([0, 50, 50, 20, 14, 12, 5, 1, 0, 0], dtype=float)
    assert recovery_time_s(t, y, 2e-3, 13.0) == pytest.approx(2e-3)
    assert recovery_time_s(t, y, 2e-3, 100.0) == 0.0
    assert recovery_time_s(t, y, 6e-3, 13.0) == 0.0


def test_kernel_phases_interleave(narrow: FrontEnd) -> None:
    amplifier = SaturatingAmplifier.from_frontend(narrow)
    t_bp, wave = Pulse().breakpoints()
    gain = reference_gain(narrow)
    one = pulse_kernel(amplifier, t_bp, wave, 3000.0, 2000.0, gain=gain)
    four = pulse_kernel(amplifier, t_bp, wave, 3000.0, 2000.0, gain=gain, n_phases=4)
    assert one.shape[0] == 1 and four.shape[0] == 4
    n = min(one.shape[1], four.shape[1])
    assert four[0, :n] == pytest.approx(one[0, :n], abs=0.05)
    assert one[0, 0] == pytest.approx(0.0, abs=1e-6)
    # Trimmed where the response has died away.
    assert np.abs(one[0, -1]) < 0.1
    assert one.shape[1] < 60


def test_pulses_superpose_when_the_tail_has_died(narrow: FrontEnd) -> None:
    """Two saturating pulses 5 ms apart give the sum of two single responses."""
    amplifier = SaturatingAmplifier.from_frontend(narrow)
    t_bp, wave = Pulse(imbalance=0.0).breakpoints()
    gain = reference_gain(narrow)
    t = np.arange(0.0, 0.012, 1e-5)
    single = pulse_response_uv(amplifier, t_bp, wave, 30_000.0, t, gain=gain)
    both = pulse_response_uv(
        amplifier,
        np.concatenate([t_bp, t_bp + 5e-3]),
        np.concatenate([wave, wave]),
        30_000.0,
        t,
        gain=gain,
    )
    shifted = np.interp(t - 5e-3, t, single, left=0.0)
    late = t > 5.8e-3
    assert np.abs(both - single - shifted)[late].max() < 1.0


@needs_ngspice
@pytest.mark.parametrize("which", ["wide", "narrow"])
@pytest.mark.parametrize("amplitude_mv", [3.0, 30.0])
def test_model_matches_ngspice(
    which: str, amplitude_mv: float, request: pytest.FixtureRequest
) -> None:
    frontend: FrontEnd = request.getfixturevalue(which)
    ideal = frontend.without_electrode()
    pulse = Pulse()
    t_bp, wave = pulse.breakpoints()
    gain = reference_gain(frontend)
    t_spice, v_out, _ = spice.simulate_transient(
        ideal, t_bp, wave * amplitude_mv * 1e-3, t_stop_s=0.02, max_step_s=5e-6
    )
    y_spice = -v_out / gain * 1e6
    t = np.arange(0.0, 0.02, 2e-6)
    y = pulse_response_uv(
        SaturatingAmplifier.from_frontend(frontend), t_bp, wave, amplitude_mv * 1e3, t, gain=gain
    )
    after = t > pulse.duration_s + 1e-4
    assert np.abs(np.interp(t, t_spice, y_spice) - y)[after].max() < 2.0
    assert recovery_time_s(t_spice, y_spice, pulse.duration_s, 13.0) == pytest.approx(
        recovery_time_s(t, y, pulse.duration_s, 13.0), abs=2e-5
    )
    # The rails hold in both.
    assert np.abs(y_spice).max() == pytest.approx(np.abs(y).max(), rel=0.01)


def test_transient_netlist_contents(narrow: FrontEnd) -> None:
    text = spice.transient_netlist(
        narrow, np.array([0.0, 1e-6]), np.array([0.0, 1e-3]), t_stop_s=1e-3, max_step_s=1e-6
    )
    assert "tanh(" in text and "Bclamp" in text and "PWL(0 0 1e-06 0.001)" in text
    assert "Rs tissue iface" in text
    assert "Rs tissue" not in spice.transient_netlist(
        narrow.without_electrode(),
        np.array([0.0, 1e-6]),
        np.array([0.0, 1e-3]),
        t_stop_s=1e-3,
        max_step_s=1e-6,
    )
