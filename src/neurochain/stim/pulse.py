"""The stimulation pulse, the pulse schedule, and the artifact a schedule produces.

**The pulse.** Intracortical microstimulation uses a biphasic current pulse: a
cathodic phase that excites, a short gap, then an anodic phase that takes the
charge back out so no net charge is left on the electrode. The default here is
the waveform of Flesher et al. (2016): 200 µs cathodic, 100 µs gap, then an
anodic phase at half the current for twice as long.

**Charge balance.** Two current sources never match exactly. If the second
phase returns only ``1 - imbalance`` of the charge, the rest sits on the
stimulating electrode's double layer. A stimulator then shorts the electrode to
its return after the pulse (passive discharge), and the leftover charge flows
back through the tissue as a decaying exponential with the time constant of the
electrode's capacitance and the tissue resistance. That slow current is part of
the artifact, so it is part of the waveform here.

**The waveform is piecewise linear** (a list of breakpoints), because that is
what both ngspice (a ``PWL`` source) and the Python amplifier model integrate
exactly. The same breakpoints go to both.

**Superposition.** Tissue is resistive at these frequencies, so the artifact at
a site is the current waveform times one number. The response of the amplifier
to one pulse (a *kernel*) is computed once; a train is the sum of shifted
kernels. :func:`pulse_train` does that sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from neurochain.recording import FloatArray

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Pulse:
    """A biphasic, cathodic-first current pulse.

    Attributes:
        amplitude_ua: Current of the first (cathodic) phase, microamps.
        phase_s: Duration of the first phase.
        gap_s: Gap between the phases.
        second_ratio: Current of the second phase as a fraction of the first.
            Its duration is ``phase_s / second_ratio``, so the two phases carry
            the same nominal charge.
        imbalance: Fraction of the first phase's charge that the second phase
            fails to return. 0.01 is an assumed current-source mismatch of 1 %.
        discharge_tau_s: Time constant of the passive discharge that returns
            the leftover charge. 1 ms is an assumption: about 20 nF of
            electrode capacitance (20 nC moves the electrode about 1 V) through
            about 50 kΩ of tissue and switch.
        edge_s: Rise and fall time of each current edge.
    """

    amplitude_ua: float = 60.0
    phase_s: float = 200.0e-6
    gap_s: float = 100.0e-6
    second_ratio: float = 0.5
    imbalance: float = 0.01
    discharge_tau_s: float = 1.0e-3
    edge_s: float = 2.0e-6

    def __post_init__(self) -> None:
        """Validate the timing."""
        if min(self.phase_s, self.gap_s, self.discharge_tau_s) <= self.edge_s:
            raise ValueError("phase, gap and discharge time must be longer than an edge")
        if not 0.0 < self.second_ratio <= 1.0 or not 0.0 <= self.imbalance < 1.0:
            raise ValueError("second_ratio must be in (0, 1] and imbalance in [0, 1)")

    @property
    def second_phase_s(self) -> float:
        """Duration of the second (anodic) phase."""
        return self.phase_s / self.second_ratio

    @property
    def duration_s(self) -> float:
        """Time from the start of the first phase to the end of the second."""
        return self.phase_s + self.gap_s + self.second_phase_s

    @property
    def charge_per_phase_nc(self) -> float:
        """Charge of the first phase in nanocoulombs."""
        return self.amplitude_ua * self.phase_s * 1.0e3

    @property
    def residual_charge_nc(self) -> float:
        """Charge left on the electrode after the second phase, nanocoulombs."""
        return self.imbalance * self.charge_per_phase_nc

    def breakpoints(self, tail_points: int = 48) -> tuple[FloatArray, FloatArray]:
        """Return the waveform as ``(time_s, current / amplitude)`` breakpoints.

        The first phase is -1 (cathodic). The discharge tail, if any, is
        sampled at ``tail_points`` points over eight time constants and joined
        by straight lines.
        """
        t1 = self.phase_s
        t2 = t1 + self.gap_s
        t3 = t2 + self.second_phase_s
        e = self.edge_s
        second = self.second_ratio * (1.0 - self.imbalance)
        tail0 = self.imbalance * self.phase_s / self.discharge_tau_s
        times = [0.0, e, t1, t1 + e, t2, t2 + e, t3, t3 + e]
        values = [0.0, -1.0, -1.0, 0.0, 0.0, second, second, tail0]
        if tail0 > 0.0:
            step = 8.0 * self.discharge_tau_s / tail_points
            for k in range(1, tail_points + 1):
                times.append(t3 + e + k * step)
                values.append(tail0 * float(np.exp(-k * step / self.discharge_tau_s)))
            times.append(times[-1] + step)
            values.append(0.0)
        return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)

    def waveform(self, time_s: FloatArray) -> FloatArray:
        """Return ``current / amplitude`` at the given times after pulse onset."""
        t, v = self.breakpoints()
        return np.asarray(np.interp(time_s, t, v, left=0.0, right=0.0), dtype=np.float64)


@dataclass(frozen=True)
class Schedule:
    """When the stimulator fires.

    Attributes:
        onsets_s: Onset time of every pulse, in order.
        trains_s: ``(n_trains, 2)`` start and stop time of each pulse train.
        period_s: Time between pulses inside a train.
    """

    onsets_s: FloatArray
    trains_s: FloatArray
    period_s: float

    def bins_on(self, n_bins: int, bin_s: float) -> BoolArray:
        """Return which bins lie wholly inside a train."""
        on = np.zeros(n_bins, dtype=bool)
        for start, stop in self.trains_s:
            on[round(start / bin_s) : round(stop / bin_s)] = True
        return on

    def samples_on(self, n_samples: int, fs_hz: float) -> BoolArray:
        """Return which samples lie inside a train."""
        on = np.zeros(n_samples, dtype=bool)
        for start, stop in self.trains_s:
            on[round(start * fs_hz) : round(stop * fs_hz)] = True
        return on

    def onset_samples(self, fs_hz: float) -> NDArray[np.int64]:
        """Return the first sample at or after each onset, at ``fs_hz``."""
        return np.ceil(self.onsets_s * fs_hz - 1e-6).astype(np.int64)


def random_schedule(
    duration_s: float,
    *,
    rate_hz: float,
    seed: int | list[int],
    start_s: float = 0.0,
    train_s: tuple[float, float] = (0.5, 1.5),
    pause_s: tuple[float, float] = (0.5, 1.5),
    grid_s: float = 0.05,
    clock_hz: float = 2000.0,
    jitter_s: float = 0.0,
) -> Schedule:
    """Draw pulse trains separated by pauses, as sensory feedback would arrive.

    Train and pause lengths are uniform in their ranges and rounded to
    ``grid_s`` (the feature bin), so a bin is either wholly stimulated or not.
    Inside a train the period is a whole number of ticks of ``clock_hz``: the
    stimulator runs off the ADC's clock, so every pulse lands on the same phase
    of the sampling. ``jitter_s`` breaks that on purpose.

    Args:
        duration_s: Length of the record.
        rate_hz: Requested pulse rate. The realised rate is ``1 / period_s``.
        seed: Seed of the generator.
        start_s: No stimulation before this time.
        train_s: Shortest and longest train.
        pause_s: Shortest and longest pause.
        grid_s: Train boundaries fall on multiples of this.
        clock_hz: Clock the pulse period is counted in.
        jitter_s: Each onset is moved by a uniform draw from ``±jitter_s``.
    """
    rng = np.random.default_rng(seed)
    period = max(1, round(clock_hz / rate_hz)) / clock_hz
    trains: list[tuple[float, float]] = []
    onsets: list[float] = []
    t = np.ceil(start_s / grid_s - 1e-9) * grid_s
    while True:
        t += max(1, round(rng.uniform(*pause_s) / grid_s)) * grid_s
        length = max(1, round(rng.uniform(*train_s) / grid_s)) * grid_s
        if t + length > duration_s:
            break
        trains.append((t, t + length))
        onsets += [t + k * period for k in range(int(np.floor(length / period + 1e-9)))]
        t += length
    times = np.asarray(onsets, dtype=np.float64)
    if jitter_s > 0.0:
        times = times + rng.uniform(-jitter_s, jitter_s, size=times.size)
    return Schedule(
        onsets_s=times,
        trains_s=np.asarray(trains, dtype=np.float64).reshape(-1, 2),
        period_s=period,
    )


def pulse_train(
    kernels: FloatArray, onsets_s: FloatArray, fs_hz: float, n_samples: int
) -> FloatArray:
    """Sum one pulse response per onset into a record sampled at ``fs_hz``.

    Args:
        kernels: ``(n_phases, n_kernel)``. Row ``p`` is the response to one
            pulse sampled at ``(m + p / n_phases) / fs_hz`` after its onset, so
            the row is chosen by where the onset falls between two samples.
            With one row every onset is taken to fall on a sample.
        onsets_s: Pulse onset times.
        fs_hz: Sample rate of the output.
        n_samples: Length of the output.
    """
    n_phases = kernels.shape[0]
    fine = np.rint(np.asarray(onsets_s) * fs_hz * n_phases).astype(np.int64)
    first = -(-fine // n_phases)
    phase = first * n_phases - fine
    keep = (first >= 0) & (first < n_samples)
    out = np.zeros(n_samples, dtype=np.float64)
    for p in np.unique(phase[keep]):
        impulses = np.bincount(first[keep & (phase == p)], minlength=n_samples).astype(np.float64)
        out += signal.fftconvolve(impulses, kernels[p])[:n_samples]
    return out
