"""ADC codes to run the RTL and its Python reference on.

Three sources, all returning a :class:`Stimulus`:

* :func:`neuralink_stimulus`: real recordings. The Neuralink clips are the only
  real broadband in the project long enough for ten minutes. They have no
  voltage scale, which does not matter here: digital logic sees integers, and
  bit-exactness is a statement about integers.
* :func:`synthetic_stimulus`: seeded noise and artifacts, for tests that must
  run without downloaded data.
* :func:`worst_case_stimulus`: codes chosen to drive every register to the
  largest value it can reach.

How real clips become 8-bit codes at 2 kS/s (a stand-in for U3's narrowband
front end, stated on every figure):

1. Each clip's 10-bit codes pass a causal second-order Butterworth band-pass,
   300 Hz to 1 kHz, the amplifier's pass band. The filter starts in its steady
   state for the clip's first sample, so a clip's DC level causes no transient.
2. The result is sampled at 2 kS/s by linear interpolation between the 19.5 kHz
   samples. Nothing else limits the band, so what the band-pass lets through
   above 1 kHz aliases, as it would in the implant.
3. It is divided by 4: the same converter range digitised with 8 bits instead
   of 10. No gain is invented.
4. Each channel is its own seeded shuffle of the clips, joined end to end.
5. Stimulation artifacts (U5's pulse responses of the same front end, in units
   of the 8-bit LSB) are added before rounding, and the sum is clipped to the
   converter's range, so large artifacts rail the ADC as they would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from neurochain.data import neuralink
from neurochain.features.sbp import SBP_BAND_HZ
from neurochain.recording import FloatArray
from neurochain.rtl.spec import ChainSpec
from neurochain.stim.pulse import Schedule, random_schedule

Codes8 = NDArray[np.int8]


@dataclass(frozen=True)
class Stimulus:
    """Codes for every channel and the pulse onsets they share.

    Attributes:
        codes: ``(n_channels, n_samples)`` signed ADC codes.
        onsets: Sample index of each stimulation pulse onset.
        fs_hz: Sample rate per channel.
        info: What the codes are made of (dataset, scale, pulse counts).
    """

    codes: Codes8
    onsets: NDArray[np.int64]
    fs_hz: float
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Samples per channel."""
        return int(self.codes.shape[1])

    @property
    def seconds(self) -> float:
        """Duration of the record."""
        return self.n_samples / self.fs_hz

    def trigger(self) -> NDArray[np.uint8]:
        """Return one flag per sample: a pulse starts here."""
        flags = np.zeros(self.n_samples, dtype=np.uint8)
        flags[self.onsets[(self.onsets >= 0) & (self.onsets < self.n_samples)]] = 1
        return flags

    def head(self, n_samples: int) -> Stimulus:
        """Return the first ``n_samples`` of every channel."""
        keep = self.onsets[self.onsets < n_samples]
        return Stimulus(self.codes[:, :n_samples], keep, self.fs_hz, self.info)

    def window_fraction(self, n_taps: int) -> float:
        """Return the share of samples that fall inside a pulse window."""
        inside = sum(min(int(n) + n_taps, self.n_samples) - int(n) for n in self.onsets)
        return inside / self.n_samples


def band_limit(
    codes: FloatArray,
    fs_hz: float,
    *,
    out_fs_hz: float,
    band_hz: tuple[float, float] = SBP_BAND_HZ,
) -> FloatArray:
    """Band-pass a clip and sample it at ``out_fs_hz`` (steps 1 and 2 above).

    Args:
        codes: One clip, in the units of its converter.
        fs_hz: Its sample rate.
        out_fs_hz: Rate of the returned signal.
        band_hz: Pass band of the stand-in amplifier.

    Returns:
        The band-limited clip at ``out_fs_hz``, same units as ``codes``.
    """
    b, a = signal.butter(2, band_hz, btype="bandpass", fs=fs_hz)
    state = signal.lfilter_zi(b, a) * codes[0]
    filtered, _ = signal.lfilter(b, a, codes, zi=state)
    n_out = int(codes.size * out_fs_hz / fs_hz)
    position = np.arange(n_out) * (fs_hz / out_fs_hz)
    return np.asarray(np.interp(position, np.arange(codes.size), filtered), dtype=np.float64)


def _streams(clips: list[FloatArray], n_channels: int, n_samples: int, seed: int) -> FloatArray:
    """Join a seeded shuffle of the clips into each channel's stream."""
    rng = np.random.default_rng([seed, 0])
    out = np.empty((n_channels, n_samples), dtype=np.float64)
    for channel in range(n_channels):
        filled = 0
        while filled < n_samples:
            for index in rng.permutation(len(clips)):
                piece = clips[index][: n_samples - filled]
                out[channel, filled : filled + piece.size] = piece
                filled += piece.size
                if filled == n_samples:
                    break
    return out


def _add_artifacts(
    analog_lsb: FloatArray,
    schedule: Schedule,
    kernels_lsb: list[FloatArray],
    fs_hz: float,
    seed: int,
) -> NDArray[np.int64]:
    """Add one pulse response per onset, in place; return each train's level per channel.

    Every train draws, for every channel, one of the kernels: the artifact a
    channel sees depends on its distance from the stimulating electrode, and
    here it also changes from train to train, so the template has to re-adapt.
    """
    n_channels, n_samples = analog_lsb.shape
    onsets = schedule.onset_samples(fs_hz)
    n_trains = schedule.trains_s.shape[0]
    levels = np.random.default_rng([seed, 2]).integers(
        0, len(kernels_lsb), size=(n_trains, n_channels)
    )
    if onsets.size == 0:
        return np.asarray(levels, dtype=np.int64)
    length = max(k.size for k in kernels_lsb)
    padded = np.zeros((len(kernels_lsb), length))
    for i, kernel in enumerate(kernels_lsb):
        padded[i, : kernel.size] = kernel
    train = np.searchsorted(schedule.trains_s[:, 0], schedule.onsets_s + 1e-9, side="right") - 1
    index = onsets[:, np.newaxis] + np.arange(length)
    inside = index < n_samples
    for channel in range(n_channels):
        values = padded[levels[train, channel]]
        np.add.at(analog_lsb[channel], index[inside], values[inside])
    return np.asarray(levels, dtype=np.int64)


def _quantise(analog_lsb: FloatArray, spec: ChainSpec) -> Codes8:
    """Round half up and clip to the converter's range."""
    return np.clip(np.floor(analog_lsb + 0.5), spec.code_min, spec.code_max).astype(np.int8)


def _describe(codes: Codes8, schedule: Schedule, spec: ChainSpec) -> dict[str, Any]:
    """Return the facts about a stimulus that figures and tables quote."""
    n_samples = codes.shape[1]
    on = schedule.samples_on(n_samples, spec.fs_hz)
    quiet = codes[:, ~on].astype(np.float64)
    railed = (codes <= spec.code_min) | (codes >= spec.code_max)
    return {
        "n_channels": int(codes.shape[0]),
        "n_samples": int(n_samples),
        "seconds": n_samples / spec.fs_hz,
        "n_pulses": int(schedule.onsets_s.size),
        "n_trains": int(schedule.trains_s.shape[0]),
        "pulse_period_samples": round(schedule.period_s * spec.fs_hz),
        "stimulated_fraction": float(on.mean()),
        "railed_fraction": float(railed.mean()),
        "quiet_rms_lsb": float(np.sqrt(np.mean(quiet**2))) if quiet.size else 0.0,
        "quiet_mean_abs_lsb": float(np.mean(np.abs(quiet))) if quiet.size else 0.0,
    }


def u5_kernels_lsb(period_s: float) -> tuple[list[FloatArray], list[float]]:
    """Return U5's pulse responses of the SBP chain in 8-bit LSB, and their amplitudes in mV.

    The amplitudes are U5's grid (0.1 to 100 mV at the electrode). The LSB is
    the chain's nominal one, ±1 mV over 8 bits; it fixes how large an artifact
    is relative to a code and nothing else.
    """
    from neurochain.analysis import u5  # heavy import, needed only here

    config = u5.U5Config()
    chain = u5.chains(config)["sbp"]
    kernels = u5.build_artifacts(config, period_s).kernels["sbp"]
    return [k[0] / chain.adc.lsb_uv for k in kernels], list(config.amplitudes_mv)


def neuralink_stimulus(
    spec: ChainSpec,
    seconds: float,
    seed: int,
    *,
    root: Path | None = None,
    rate_hz: float = 100.0,
    start_s: float = 1.0,
) -> Stimulus:
    """Build codes from the Neuralink clips, with stimulation artifacts added.

    Args:
        spec: The chain; sets channel count, resolution and sample rate.
        seconds: Length of the record per channel.
        seed: Seed for the clip order, the pulse trains and the artifact sizes.
        root: Directory of the clips; defaults to the downloader's.
        rate_hz: Pulse rate inside a train.
        start_s: No stimulation before this time.
    """
    paths = neuralink.list_clips(root)
    clips: list[FloatArray] = []
    raw_s = 0.0
    for path in paths:
        pcm, fs_hz = neuralink.read_pcm(path)
        raw = neuralink.int16_to_adc_code(pcm).astype(np.float64)
        raw_s += raw.size / fs_hz
        clips.append(band_limit(raw, fs_hz, out_fs_hz=spec.fs_hz) / 4.0)
    n_samples = round(seconds * spec.fs_hz)
    analog = _streams(clips, spec.n_channels, n_samples, seed)
    schedule = random_schedule(
        seconds,
        rate_hz=rate_hz,
        seed=[seed, 1],
        start_s=start_s,
        grid_s=spec.bin_s,
        clock_hz=spec.fs_hz,
    )
    kernels, amplitudes_mv = u5_kernels_lsb(schedule.period_s)
    _add_artifacts(analog, schedule, kernels, spec.fs_hz, seed)
    codes = _quantise(analog, spec)
    info = {
        "source": "neuralink",
        "n_clips": len(clips),
        "unique_seconds": raw_s,
        "clips_per_channel": n_samples / float(np.mean([c.size for c in clips])),
        "band_hz": list(SBP_BAND_HZ),
        "scale": "10-bit codes divided by 4",
        "artifact_amplitudes_mv": amplitudes_mv,
        "pulse_rate_hz": 1.0 / schedule.period_s,
        "seed": seed,
        **_describe(codes, schedule, spec),
    }
    return Stimulus(codes, schedule.onset_samples(spec.fs_hz), spec.fs_hz, info)


def synthetic_stimulus(
    spec: ChainSpec,
    seconds: float,
    seed: int,
    *,
    noise_lsb: float = 1.4,
    peaks_lsb: tuple[float, ...] = (0.0, 6.0, 60.0, 600.0, 1500.0),
) -> Stimulus:
    """Build codes from Gaussian noise plus artifacts: no downloaded data.

    The artifact is a short damped oscillation scaled to each of ``peaks_lsb``;
    the larger ones rail the converter.
    """
    rng = np.random.default_rng([seed, 3])
    n_samples = round(seconds * spec.fs_hz)
    analog = rng.normal(0.0, noise_lsb, size=(spec.n_channels, n_samples))
    schedule = random_schedule(
        seconds,
        rate_hz=100.0,
        seed=[seed, 1],
        start_s=0.1,
        train_s=(0.2, 0.6),
        pause_s=(0.1, 0.3),
        grid_s=spec.bin_s,
        clock_hz=spec.fs_hz,
    )
    shape = np.array([0.0, 1.0, 0.07, -0.21, -0.085, -0.015, 0.001])
    _add_artifacts(analog, schedule, [peak * shape for peak in peaks_lsb], spec.fs_hz, seed)
    codes = _quantise(analog, spec)
    info = {"source": "synthetic", "seed": seed, **_describe(codes, schedule, spec)}
    return Stimulus(codes, schedule.onset_samples(spec.fs_hz), spec.fs_hz, info)


def worst_case_stimulus(spec: ChainSpec, n_pulses: int = 40, seed: int = 0) -> Stimulus:
    """Build codes that push the canceller and the filter to their largest outputs.

    Pulses come every ``n_taps`` samples, so every sample lies in a window.
    Channel 0 loads the template with one extreme and then presents the other,
    with signs that follow the filter's impulse response backwards: the input
    that attains ``sum |h|``. The other channels hold seeded random extremes,
    including codes on the rails, which are blanked.
    """
    rng = np.random.default_rng([seed, 4])
    taps = spec.n_taps
    n_samples = taps * n_pulses
    b, a = spec.highpass.float_coefficients()
    impulse = np.zeros(taps)
    impulse[0] = 1.0
    signs = np.where(signal.lfilter(b, a, impulse)[::-1] >= 0.0, 1, -1)
    extremes = np.array([spec.code_min, spec.code_min + 1, spec.code_max - 1, spec.code_max])
    codes = rng.choice(extremes, size=(spec.n_channels, n_samples))
    first = np.where(signs > 0, spec.code_min, spec.code_max)
    second = np.where(signs > 0, spec.code_max - 1, spec.code_min + 1)
    codes[0] = np.concatenate([first if p % 2 == 0 else second for p in range(n_pulses)])
    onsets = np.arange(n_pulses, dtype=np.int64) * taps
    info = {"source": "worst case", "n_channels": spec.n_channels, "n_samples": n_samples}
    return Stimulus(codes.astype(np.int8), onsets, spec.fs_hz, info)
