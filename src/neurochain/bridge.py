"""Carrying a front end's effect onto feature datasets that have no broadband.

No public dataset in this project has both broadband voltages and behaviour.
The front end and ADC act on voltages; the decoders need behaviour. This module
is the bridge between them.

The idea: measure what a chain configuration does to features on data with
ground truth (MEArec), express that as a small number of unit-free quantities,
and apply the same change to recorded features.

* **Threshold crossings.** A worse chain misses real spikes and adds false
  ones. So recorded counts are thinned (each event kept with probability
  ``p_keep``) and Poisson false positives are added.
* **Spiking-band power.** SBP is the mean absolute value of a band signal, and
  for independent sources mean-square values add. A worse chain adds noise
  power, so the recorded SBP is raised in quadrature by a multiple of each
  channel's own noise floor, and each bin's average becomes noisier.

Both operations are relative to the chain that made the recording, which is
taken to be as good as the project's reference chain. A configuration can
therefore be made worse than the recording, never better.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from neurochain.adc import Adc
from neurochain.frontend.model import FrontEnd
from neurochain.recording import FloatArray


def degrade_crossings(
    counts: FloatArray,
    p_keep: float,
    extra_rate_hz: float,
    bin_s: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Thin recorded crossing counts and add false positives.

    Args:
        counts: ``(n_bins, n_channels)`` recorded counts.
        p_keep: Probability that a recorded event survives the worse chain.
        extra_rate_hz: False positives added per channel per second.
        bin_s: Bin width in seconds.
        rng: Random generator.
    """
    if not 0.0 <= p_keep <= 1.0:
        raise ValueError("p_keep must be between 0 and 1")
    kept = rng.binomial(np.rint(counts).astype(np.int64), p_keep)
    added = rng.poisson(extra_rate_hz * bin_s, size=counts.shape) if extra_rate_hz > 0 else 0
    return np.asarray(kept + added, dtype=np.float64)


def sbp_floor(sbp: FloatArray, n_effective: float, percentile: float = 5.0) -> FloatArray:
    """Estimate each channel's noise-only SBP level from its quietest bins.

    A low percentile over time picks out bins with little spiking. But each bin
    is an average of a finite number of samples, so even pure noise scatters
    around its true level with a relative standard deviation of
    ``sqrt((pi/2 - 1) / n_effective)``; the 5th percentile sits 1.645 of those
    below the level. The estimate is corrected for that.

    Args:
        sbp: ``(n_bins, n_channels)`` recorded SBP.
        n_effective: Independent samples averaged per bin.
        percentile: Which low percentile to read.
    """
    relative_sd = np.sqrt((np.pi / 2.0 - 1.0) / n_effective)
    z = float(stats.norm.ppf(percentile / 100.0))
    low = np.percentile(sbp, percentile, axis=0)
    return np.asarray(low / (1.0 + z * relative_sd), dtype=np.float64)


def degrade_sbp(
    sbp: FloatArray,
    noise_power_ratio: float,
    floor: FloatArray,
    *,
    n_effective: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Change recorded SBP the way a noisier chain would.

    Two things happen when noise power is added to a band signal whose mean
    absolute value is being averaged:

    * **The level rises.** Mean-square values add, so
      ``sbp' = sqrt(sbp**2 + (ratio - 1) * floor**2)``.
    * **Each average gets noisier.** For Gaussian noise of RMS ``rho``,
      ``|x|`` has mean ``rho * sqrt(2/pi)`` and variance
      ``rho**2 * (1 - 2/pi)``. Averaging ``n_effective`` independent samples
      leaves a standard deviation of ``floor * sqrt((ratio - 1) * (pi/2 - 1) /
      n_effective)`` from the added noise, in the feature's own units.

    Args:
        sbp: ``(n_bins, n_channels)`` recorded SBP, any unit.
        noise_power_ratio: Total over background noise power; at least 1, and 1
            leaves the feature unchanged.
        floor: ``(n_channels,)`` noise-only SBP level, same unit as ``sbp``.
        n_effective: Independent samples averaged per bin, about
            ``2 * bandwidth * bin duration``.
        rng: Random generator.
    """
    if noise_power_ratio < 1.0:
        raise ValueError("a chain cannot have less noise than the background alone")
    added = noise_power_ratio - 1.0
    if added == 0.0:
        return np.asarray(sbp, dtype=np.float64).copy()
    raised = np.sqrt(sbp**2 + added * floor**2)
    jitter = floor * np.sqrt(added * (np.pi / 2.0 - 1.0) / n_effective)
    return np.asarray(raised + jitter * rng.standard_normal(sbp.shape), dtype=np.float64)


def sbp_noise_power_ratio(
    frontend: FrontEnd,
    adc: Adc,
    background_freqs_hz: FloatArray,
    background_psd_uv2: FloatArray,
    *,
    digital_band_hz: tuple[float, float] | None = None,
) -> float:
    """Return total over background noise power in a chain's SBP signal.

    Two cases:

    * **Narrowband chain** (``digital_band_hz`` is ``None``). The amplifier's
      own first-order skirts are the only filter, and the ADC samples directly.
      Sampling folds every frequency into the first Nyquist zone, so all of the
      amplifier's output noise and all of the background that the amplifier
      passes end up in the feature. The skirts are gentle, which matters: the
      amplifier's white noise has an effective bandwidth well beyond the
      nominal band.
    * **Wideband chain with a digital band-pass** (``digital_band_hz`` given).
      The feature only sees the band, so noise is integrated across it.

    Quantisation adds ``step**2 / 12``, spread from 0 to the ADC's Nyquist
    frequency.

    Args:
        frontend: Electrode and amplifier.
        adc: The converter.
        background_freqs_hz: Frequency axis of the neural background's PSD.
        background_psd_uv2: That PSD in µV²/Hz at the electrode.
        digital_band_hz: Pass band of a digital filter after the ADC, if any.
    """
    if digital_band_hz is not None:
        low, high = digital_band_hz
        in_band = (background_freqs_hz >= low) & (background_freqs_hz <= high)
        background = float(np.trapezoid(background_psd_uv2[in_band], background_freqs_hz[in_band]))
        front_end = (frontend.band_noise(low, high).total_v * 1e6) ** 2
        quantisation = adc.lsb_uv**2 / 12.0 * min(1.0, (high - low) / (adc.fs_hz / 2.0))
        return 1.0 + (front_end + quantisation) / background
    centre = float(np.sqrt(frontend.f_low_hz * frontend.f_high_hz))
    gain_sq = float(np.abs(frontend.signal_gain(np.array([centre])))[0]) ** 2
    positive = background_freqs_hz > 0
    freqs = background_freqs_hz[positive]
    shaped = background_psd_uv2[positive] * np.abs(frontend.signal_gain(freqs)) ** 2 / gain_sq
    background = float(np.trapezoid(shaped, freqs))
    wide = np.geomspace(1.0e-2, 1.0e7, 4000)
    front_end = float(np.trapezoid(frontend.output_noise(wide).total, wide)) / gain_sq * 1e12
    return 1.0 + (front_end + adc.lsb_uv**2 / 12.0) / background
