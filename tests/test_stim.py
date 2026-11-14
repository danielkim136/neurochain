"""The stimulation pulse, its schedule and how it spreads through tissue."""

import numpy as np
import pytest

from neurochain.stim import propagation as prop
from neurochain.stim.pulse import Pulse, pulse_train, random_schedule


def test_pulse_charge_and_duration() -> None:
    pulse = Pulse()
    assert pulse.charge_per_phase_nc == pytest.approx(12.0)
    assert pulse.second_phase_s == pytest.approx(400e-6)
    assert pulse.duration_s == pytest.approx(700e-6)
    assert pulse.residual_charge_nc == pytest.approx(0.12)


@pytest.mark.parametrize("imbalance", [0.0, 0.01, 0.1])
def test_pulse_returns_all_its_charge_once_the_discharge_is_counted(imbalance: float) -> None:
    pulse = Pulse(imbalance=imbalance)
    t, w = pulse.breakpoints(tail_points=400)
    net = np.trapezoid(w, t) / pulse.phase_s
    # Finite edges and the truncated tail leave a fraction of a percent.
    assert abs(net) < 2e-3


def test_pulse_without_discharge_leaves_the_imbalance() -> None:
    pulse = Pulse(imbalance=0.05)
    t, w = pulse.breakpoints()
    main = t <= pulse.duration_s + pulse.edge_s
    assert np.trapezoid(w[main], t[main]) / pulse.phase_s == pytest.approx(-0.05, abs=5e-3)


def test_waveform_levels() -> None:
    pulse = Pulse(imbalance=0.0)
    w = pulse.waveform(np.array([-1e-4, 1e-4, 2.5e-4, 5e-4, 2e-3]))
    assert w == pytest.approx([0.0, -1.0, 0.0, 0.5, 0.0])


def test_pulse_rejects_impossible_timing() -> None:
    with pytest.raises(ValueError, match="longer than an edge"):
        Pulse(gap_s=1e-6)
    with pytest.raises(ValueError, match="second_ratio"):
        Pulse(second_ratio=1.5)


def test_schedule_is_seeded_and_on_the_clock() -> None:
    a = random_schedule(20.0, rate_hz=100.0, seed=3, start_s=2.0)
    b = random_schedule(20.0, rate_hz=100.0, seed=3, start_s=2.0)
    c = random_schedule(20.0, rate_hz=100.0, seed=4, start_s=2.0)
    assert np.array_equal(a.onsets_s, b.onsets_s)
    assert not np.array_equal(a.trains_s, c.trains_s)
    assert a.onsets_s.min() >= 2.0
    assert a.trains_s.max() <= 20.0
    assert a.period_s == pytest.approx(0.01)
    ticks = a.onsets_s * 2000.0
    assert np.allclose(ticks, np.rint(ticks), atol=1e-6)


def test_schedule_bins_match_the_trains() -> None:
    schedule = random_schedule(30.0, rate_hz=100.0, seed=0)
    on = schedule.bins_on(600, 0.05)
    assert on.sum() * 0.05 == pytest.approx(np.diff(schedule.trains_s, axis=1).sum())
    assert schedule.samples_on(60_000, 2000.0).mean() == pytest.approx(on.mean())
    # Every pulse lies in a stimulated bin.
    assert on[(schedule.onsets_s / 0.05 + 1e-9).astype(int)].all()


def test_period_is_rounded_to_the_clock() -> None:
    schedule = random_schedule(10.0, rate_hz=300.0, seed=0)
    assert schedule.period_s == pytest.approx(3.5e-3)


def test_jitter_moves_onsets_off_the_clock() -> None:
    schedule = random_schedule(10.0, rate_hz=100.0, seed=0, jitter_s=100e-6)
    ticks = schedule.onsets_s * 2000.0
    assert np.abs(ticks - np.rint(ticks)).max() > 0.05
    assert np.abs(ticks - np.rint(ticks)).max() <= 0.2 + 1e-9


def test_pulse_train_places_the_kernel_at_each_onset() -> None:
    kernel = np.array([[1.0, 2.0, 3.0]])
    out = pulse_train(kernel, np.array([0.002, 0.010]), 1000.0, 16)
    expected = np.zeros(16)
    expected[2:5] = [1, 2, 3]
    expected[10:13] = [1, 2, 3]
    assert out == pytest.approx(expected, abs=1e-12)


def test_pulse_train_picks_the_phase_of_an_onset_between_samples() -> None:
    kernels = np.array([[1.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    # An onset a quarter of a sample before sample 4: the first sample is 1/4 late.
    out = pulse_train(kernels, np.array([3.75e-3]), 1000.0, 8)
    assert out[4] == pytest.approx(10.0)
    assert np.count_nonzero(np.abs(out) > 1e-9) == 1


def test_monopole_matches_the_hand_calculation() -> None:
    # 60 µA at 1 mm in 0.3 S/m: 60e-6 / (4 pi 0.3 1e-3) = 15.9 mV.
    assert prop.monopole_uv(60.0, 1000.0)[()] == pytest.approx(15_915.5, rel=1e-4)
    assert prop.monopole_uv(60.0, 2000.0)[()] == pytest.approx(15_915.5 / 2, rel=1e-4)


def test_dipole_falls_as_the_square_and_matches_two_sources() -> None:
    far = np.array([4000.0, 8000.0])
    approx = prop.dipole_far_field_uv(60.0, 400.0, far)
    assert approx[0] / approx[1] == pytest.approx(4.0)
    points = np.column_stack([far, np.zeros(2), np.zeros(2)])
    pair = np.array([[200.0, 0.0, 0.0], [-200.0, 0.0, 0.0]])
    exact = prop.potential_uv(points, pair, np.array([60.0, -60.0]))
    assert exact == pytest.approx(approx, rel=0.01)
    assert np.all(exact < prop.monopole_uv(60.0, far) / 5)


def test_utah_array_layout() -> None:
    sites = prop.utah_positions_um()
    assert sites.shape == (96, 3)
    assert prop.utah_positions_um(drop_corners=False).shape == (100, 3)
    assert sites.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    assert np.diff(np.unique(sites[:, 0])) == pytest.approx(400.0)


def test_a_nearby_reference_removes_the_common_part() -> None:
    sites = prop.utah_positions_um()
    stim = np.array([20_000.0, 0.0, 0.0])
    distant = prop.artifact_amplitude_uv(sites, stim, 60.0)
    local = prop.artifact_amplitude_uv(sites, stim, 60.0, reference_um=np.array([[0.0, 0.0, 0.0]]))
    assert np.abs(local).max() < 0.15 * np.abs(distant).min()
    bipolar = prop.artifact_amplitude_uv(
        sites, stim, 60.0, return_um=stim + np.array([400.0, 0.0, 0.0])
    )
    assert np.abs(bipolar).max() < 0.05 * np.abs(distant).min()
