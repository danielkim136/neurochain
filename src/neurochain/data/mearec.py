"""Loader for the MEArec ground-truth recording.

MEArec (Buccino and Einevoll, 2021) simulates extracellular recordings from
biophysically detailed cortical neurons, so every spike's time and source is
known. ``scripts/make_mearec.py`` generates one recording and stores it as a
plain ``.npz``; this module reads it and needs neither MEArec nor NEURON.

The file holds two signals built from the same spike trains:

* ``clean_uv``: the spikes of the simulated units and nothing else.
* ``background_uv``: MEArec's far-neuron background, hundreds of distant
  neurons too small to detect. This is the "biological noise" that dominates a
  real recording.

Keeping them apart lets later stages set the background level exactly, and add
front-end noise from the U1 model on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurochain.dsp.filters import SPIKE_BAND_HZ
from neurochain.dsp.spectra import band_power, welch_psd
from neurochain.paths import data_dir
from neurochain.recording import FloatArray, IntArray, Recording

DEFAULT_FILE = "recording_seed0.npz"


def mearec_dir() -> Path:
    """Return the directory generated MEArec files live in."""
    return data_dir() / "mearec"


@dataclass(frozen=True)
class MearecRecording:
    """A simulated multi-channel recording with ground-truth spike times.

    Attributes:
        fs_hz: Sampling rate.
        clean_uv: ``(n_channels, n_samples)`` spikes only, in microvolts.
        background_uv: ``(n_channels, n_samples)`` far-neuron background,
            scaled so each channel has an RMS of 1 µV inside the spike band
            (300 Hz to 7.5 kHz). More than half of its power lies below 300 Hz,
            so its broadband RMS is larger.
        spike_samples: One array per unit of spike sample indices.
        templates_uv: ``(n_units, n_channels, n_template_samples)`` mean
            waveform of each unit on each channel.
        cell_types: MEArec's cell type of each unit (``"E"`` or ``"I"``).
    """

    fs_hz: float
    clean_uv: FloatArray
    background_uv: FloatArray
    spike_samples: tuple[IntArray, ...]
    templates_uv: FloatArray
    cell_types: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.clean_uv.shape[0])

    @property
    def n_units(self) -> int:
        """Number of simulated units."""
        return len(self.spike_samples)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return float(self.clean_uv.shape[1] / self.fs_hz)

    @property
    def amplitudes_uv(self) -> FloatArray:
        """Depth of each unit's trough on each channel, ``(n_units, n_channels)``, positive."""
        return np.asarray(-self.templates_uv.min(axis=2), dtype=np.float64)

    def signal_uv(self, channel: int, background_rms_uv: float) -> FloatArray:
        """Return one channel with the background scaled to a chosen RMS.

        Args:
            channel: Channel index.
            background_rms_uv: Spike-band RMS of the far-neuron background to
                add; 0 gives the clean spikes. The Nason in-vivo channel
                measures about 13 µV on this scale.
        """
        return np.asarray(
            self.clean_uv[channel] + background_rms_uv * self.background_uv[channel],
            dtype=np.float64,
        )

    def recording(self, channel: int, background_rms_uv: float) -> Recording:
        """Return one channel as a :class:`Recording` in microvolts."""
        return Recording(
            samples=self.signal_uv(channel, background_rms_uv)[np.newaxis, :],
            fs_hz=self.fs_hz,
            units="µV",
            source="mearec",
            meta={"channel": channel, "background_rms_uv": background_rms_uv},
        )

    def units_above(self, channel: int, min_amplitude_uv: float) -> IntArray:
        """Return the indices of units whose trough on a channel is at least this deep."""
        return np.flatnonzero(self.amplitudes_uv[:, channel] >= min_amplitude_uv).astype(np.int64)

    def spikes_of(self, units: IntArray) -> IntArray:
        """Return the merged, sorted spike samples of the given units."""
        if units.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate([self.spike_samples[int(u)] for u in units]))


def load_mearec(path: Path | None = None) -> MearecRecording:
    """Load a recording written by ``scripts/make_mearec.py``.

    Args:
        path: The ``.npz`` file. Defaults to ``data/mearec/recording_seed0.npz``.

    Raises:
        FileNotFoundError: If the file is absent, with the command that makes it.
    """
    path = path or mearec_dir() / DEFAULT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make mearec` to generate it")
    with np.load(path, allow_pickle=False) as stored:
        background = stored["background_uv"].astype(np.float64)
        fs_hz = float(stored["fs_hz"])
        freqs, psd = welch_psd(background, fs_hz, resolution_hz=10.0)
        rms = np.sqrt(band_power(freqs, psd, *SPIKE_BAND_HZ))[:, np.newaxis]
        spike_unit = stored["spike_unit"]
        spike_samples = stored["spike_samples"]
        n_units = int(stored["templates_uv"].shape[0])
        return MearecRecording(
            fs_hz=fs_hz,
            clean_uv=stored["clean_uv"].astype(np.float64),
            background_uv=background / rms,
            spike_samples=tuple(
                np.sort(spike_samples[spike_unit == u]).astype(np.int64) for u in range(n_units)
            ),
            templates_uv=stored["templates_uv"].astype(np.float64),
            cell_types=tuple(str(c) for c in stored["cell_types"]),
        )


def synthetic_recording(
    n_channels: int = 2, duration_s: float = 8.0, fs_hz: float = 30_000.0, seed: int = 0
) -> MearecRecording:
    """Build a small stand-in with the real recording's structure, for tests.

    One 100 µV unit per channel firing at about 12 Hz, and a white background.
    It needs neither MEArec nor a generated file.
    """
    from neurochain.data.synthetic import make_synthetic_recording, spike_template

    rng = np.random.default_rng(seed)
    template, _ = spike_template(fs_hz)
    clean, spikes, templates = [], [], []
    for channel in range(n_channels):
        recording, truth = make_synthetic_recording(
            duration_s=duration_s,
            fs_hz=fs_hz,
            amplitudes=(100.0,),
            rates_hz=(12.0,),
            noise_rms=0.0,
            lfp_rms=0.0,
            seed=seed * 1000 + channel,
        )
        clean.append(recording.channel(0))
        spikes.append(truth.spike_samples[0])
        row = np.zeros((n_channels, template.size))
        row[channel] = 100.0 * template
        templates.append(row)
    background = rng.standard_normal((n_channels, round(duration_s * fs_hz)))
    in_band = (SPIKE_BAND_HZ[1] - SPIKE_BAND_HZ[0]) / (fs_hz / 2.0)
    return MearecRecording(
        fs_hz=fs_hz,
        clean_uv=np.stack(clean),
        background_uv=background / np.sqrt(in_band),
        spike_samples=tuple(spikes),
        templates_uv=np.stack(templates),
        cell_types=("E",) * n_channels,
    )
