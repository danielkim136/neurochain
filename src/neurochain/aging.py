"""A small model of how a chronically implanted electrode ages.

Three things change in the months after implantation, and each enters the
sensing chain at a different place:

* **Encapsulation raises the impedance.** Glial tissue grows around the shank.
  At 1 kHz the impedance of Utah arrays in monkey cortex rose 2.48 times in the
  first two weeks and then held for months (Barrese et al., 2013). The model
  puts that rise in series with the electrode, as a resistive sheath, and
  solves the sheath resistance from the measured ratio.
* **The noise follows the impedance.** A resistor generates thermal noise of
  density ``4kT Re{Z}`` (Nyquist; ELEC2133), so nothing extra is assumed: the
  aged electrode goes through U1's front-end model and the noise comes out.
* **Spikes get smaller.** Neurons die or are pushed away. The largest spike on
  each channel lost 2.4 % of its amplitude per month on average (Chestek et
  al., 2011). Channels differ, so each gets its own rate.

Every number here is an assumption taken from one paper about one kind of
array. ``scripts/u6_key_number.py`` changes each of them, and the conclusions
of U6 are the ones that survive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import optimize

from neurochain.frontend.electrode import Electrode
from neurochain.recording import FloatArray

WEEKS_PER_MONTH = 365.25 / 12.0 / 7.0
REFERENCE_HZ = 1000.0

Encapsulation = Literal["series", "interface"]


def encapsulated(
    electrode: Electrode, impedance_factor: float, mode: Encapsulation = "series"
) -> Electrode:
    """Return the electrode with its 1 kHz impedance multiplied by a factor.

    Args:
        electrode: The fresh electrode.
        impedance_factor: Ratio of aged to fresh ``|Z|`` at 1 kHz; at least 1.
        mode: Where the rise is put. ``"series"`` adds a resistive tissue
            sheath to ``Rs`` and leaves the metal interface alone, which is
            what encapsulation is and is the worst case for noise, because all
            of the added impedance is real. ``"interface"`` scales every
            element, as if the exposed area had shrunk: the phase is kept and
            less of the rise is resistive.

    Raises:
        ValueError: If the factor is below 1.
    """
    if impedance_factor < 1.0:
        raise ValueError("the model only raises the impedance (factor >= 1)")
    if impedance_factor == 1.0:
        return electrode
    name = f"{electrode.name}, |Z| x{impedance_factor:.2f}"
    if mode == "interface":
        return replace(
            electrode,
            rs_ohm=electrode.rs_ohm * impedance_factor,
            rct_ohm=electrode.rct_ohm * impedance_factor,
            q_s_sn=electrode.q_s_sn / impedance_factor,
            name=name,
        )
    target_ohm = impedance_factor * electrode.magnitude_ohm(REFERENCE_HZ)

    def mismatch(extra_ohm: float) -> float:
        trial = replace(electrode, rs_ohm=electrode.rs_ohm + extra_ohm)
        return trial.magnitude_ohm(REFERENCE_HZ) - target_ohm

    extra_ohm = float(optimize.brentq(mismatch, 0.0, target_ohm))
    return replace(electrode, rs_ohm=electrode.rs_ohm + extra_ohm, name=name)


@dataclass(frozen=True)
class AgingModel:
    """Parameters of the aging model. All of them are assumptions.

    Attributes:
        impedance_factor_final: Ratio of settled to fresh ``|Z|`` at 1 kHz
            (2.48: Barrese et al., 2013, 305 to 755 kOhm on Utah arrays).
        encapsulation_weeks: Weeks after which 95 % of the rise has happened
            (2: the same paper).
        encapsulation: Where the rise is put (see :func:`encapsulated`).
        amplitude_decline_per_month: Mean share of spike amplitude lost each
            month (0.024: Chestek et al., 2011).
        decline_spread: Coefficient of variation of the decline rate across
            channels. Rates are gamma distributed; 1 makes them exponential,
            so some channels barely change and a few fade fast. 0 gives every
            channel the mean rate.
        background_factor_final: Ratio of settled to fresh neural background
            (RMS). 1 by default: the far-neuron background is assumed not to
            change. Barrese et al. saw recorded noise rise 1.53 times in the
            first month, which the key-number script can switch on.
    """

    impedance_factor_final: float = 2.48
    encapsulation_weeks: float = 2.0
    encapsulation: Encapsulation = "series"
    amplitude_decline_per_month: float = 0.024
    decline_spread: float = 1.0
    background_factor_final: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.impedance_factor_final < 1.0 or self.background_factor_final <= 0.0:
            raise ValueError("impedance factor must be >= 1 and background factor > 0")
        if not 0.0 <= self.amplitude_decline_per_month < 1.0:
            raise ValueError("amplitude_decline_per_month must be in [0, 1)")
        if self.decline_spread < 0.0 or self.encapsulation_weeks <= 0.0:
            raise ValueError("decline_spread must be >= 0 and encapsulation_weeks > 0")

    def settled_share(self, weeks: float) -> float:
        """Return the share of encapsulation complete after ``weeks`` (0 to 1).

        A first-order approach, ``1 - exp(-t / tau)``, with ``tau`` a third of
        ``encapsulation_weeks`` so that 95 % is reached there.
        """
        return float(1.0 - np.exp(-3.0 * max(weeks, 0.0) / self.encapsulation_weeks))

    def impedance_factor(self, weeks: float) -> float:
        """Return aged over fresh ``|Z|`` at 1 kHz after ``weeks``."""
        return 1.0 + (self.impedance_factor_final - 1.0) * self.settled_share(weeks)

    def background_factor(self, weeks: float) -> float:
        """Return aged over fresh neural background RMS after ``weeks``."""
        return 1.0 + (self.background_factor_final - 1.0) * self.settled_share(weeks)

    def electrode(self, fresh: Electrode, weeks: float) -> Electrode:
        """Return the electrode as it is after ``weeks``."""
        return encapsulated(fresh, self.impedance_factor(weeks), self.encapsulation)

    def decline_rates_per_week(self, n_channels: int, seed: int) -> FloatArray:
        """Draw each channel's amplitude decay rate, in 1/week.

        The mean rate reproduces ``amplitude_decline_per_month`` for a channel
        that has exactly the mean rate.
        """
        mean = -np.log(1.0 - self.amplitude_decline_per_month) / WEEKS_PER_MONTH
        if self.decline_spread == 0.0 or mean == 0.0:
            return np.full(n_channels, mean, dtype=np.float64)
        shape = 1.0 / self.decline_spread**2
        rng = np.random.default_rng(seed)
        return np.asarray(rng.gamma(shape, mean / shape, size=n_channels), dtype=np.float64)

    def amplitude_scale(self, weeks: FloatArray, n_channels: int, seed: int) -> FloatArray:
        """Return each channel's spike amplitude relative to week 0.

        Args:
            weeks: Times to evaluate, in weeks.
            n_channels: Number of channels.
            seed: Seed for the per-channel rates. The same seed gives the same
                channels their rates, whatever ``weeks`` is.

        Returns:
            ``(n_weeks, n_channels)`` factors in (0, 1], each channel decaying
            exponentially at its own rate.
        """
        rates = self.decline_rates_per_week(n_channels, seed)
        times = np.asarray(weeks, dtype=np.float64)[:, np.newaxis]
        return np.asarray(np.exp(-rates[np.newaxis, :] * times), dtype=np.float64)
