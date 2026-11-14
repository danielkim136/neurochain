"""Per-channel power of a sensing chain: amplifier, ADC, digital logic, radio.

Each stage is a one-line model with a stated constant:

* **Amplifier:** supply voltage times supply current. Current buys noise
  through the NEF: ``e_n = NEF * sqrt(4kT * U_T / I)`` (U1).
* **ADC:** the Walden figure of merit, ``P = FoM * 2**bits * fs``. Every extra
  bit doubles the power; so does doubling the sample rate.
* **Digital logic:** an energy per multiply, times multiplies per sample, times
  the sample rate.
* **Radio:** an energy per bit times the bit rate actually sent.

The constants are assumptions, stated in :class:`PowerModel` with their
sources. The conclusions that matter are the ones that survive changing them
by a factor of ten, which ``scripts/u3_key_number.py`` lets you try.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Feature = Literal["crossings", "sbp", "both", "raw"]

WIDEBAND_HZ = (1.0, 7500.0)
NARROWBAND_HZ = (300.0, 1000.0)
SBP_FS_HZ = 2000.0


@dataclass(frozen=True)
class PowerModel:
    """Technology constants of the power model.

    Attributes:
        v_supply_v: Supply voltage of the amplifier.
        nef: Noise efficiency factor of the amplifier (Harrison and Charles,
            2003, measured 4.0).
        adc_fom_j: Walden figure of merit, joules per conversion step. 50 fJ is
            about ten times the 4.4 fJ of van Elzakker et al. (2010), leaving
            margin for a converter shared across an implant.
        energy_per_multiply_j: Energy of one fixed-point multiply-accumulate.
            1 pJ is an order-of-magnitude assumption; U7 replaces it with a
            synthesised estimate.
        energy_per_bit_j: Radio energy per transmitted bit. 10 pJ is the
            near-term target of Sen et al. (2026).
    """

    v_supply_v: float = 1.8
    nef: float = 4.0
    adc_fom_j: float = 50.0e-15
    energy_per_multiply_j: float = 1.0e-12
    energy_per_bit_j: float = 10.0e-12


@dataclass(frozen=True)
class ChainConfig:
    """One point in the design space.

    Attributes:
        feature: What the implant computes and sends. ``"crossings"``: counts
            of threshold crossings from a wideband chain. ``"sbp"``:
            spiking-band power from a narrowband chain sampled at 2 kS/s.
            ``"both"``: both features from one wideband chain. ``"raw"``: every
            sample, with nothing computed on the implant.
        i_total_a: Amplifier supply current.
        bits: ADC resolution.
        fs_hz: ADC sample rate (fixed at 2 kS/s for ``"sbp"``).
        bin_s: Interval at which a feature value is sent.
    """

    feature: Feature
    i_total_a: float
    bits: int = 10
    fs_hz: float = 30_000.0
    bin_s: float = 0.05

    def __post_init__(self) -> None:
        """Pin the sample rate of the narrowband chain."""
        if self.feature == "sbp" and self.fs_hz != SBP_FS_HZ:
            object.__setattr__(self, "fs_hz", SBP_FS_HZ)

    @property
    def band_hz(self) -> tuple[float, float]:
        """Pass band of the amplifier this configuration needs."""
        return NARROWBAND_HZ if self.feature == "sbp" else WIDEBAND_HZ

    @property
    def multiplies_per_sample(self) -> int:
        """Multiplies the on-implant logic performs per ADC sample.

        A second-order high-pass built as ``b0 * (1 - z**-1)**2`` over two
        feedback taps is three multiplies. Computing SBP from a wideband stream
        adds a band-pass of two such sections.
        """
        return {"crossings": 3, "sbp": 3, "both": 9, "raw": 0}[self.feature]

    @property
    def bit_rate_bps(self) -> float:
        """Bits per second sent over the radio for this channel.

        A crossing count per bin fits in 4 bits and an SBP value in 12. Raw
        streaming sends every ADC sample.
        """
        if self.feature == "raw":
            return self.fs_hz * self.bits
        bits_per_bin = {"crossings": 4, "sbp": 12, "both": 16}[self.feature]
        return bits_per_bin / self.bin_s

    @property
    def label(self) -> str:
        """Short description for tables."""
        return (
            f"{self.feature}, {self.i_total_a * 1e6:g} µA, {self.bits} b, {self.fs_hz / 1e3:g} kS/s"
        )


@dataclass(frozen=True)
class PowerBreakdown:
    """Power of each stage of one channel, in watts."""

    lna_w: float
    adc_w: float
    dsp_w: float
    link_w: float

    @property
    def total_w(self) -> float:
        """Total power per channel."""
        return self.lna_w + self.adc_w + self.dsp_w + self.link_w

    def as_microwatts(self) -> dict[str, float]:
        """Return each stage and the total in microwatts."""
        return {
            "lna_uw": self.lna_w * 1e6,
            "adc_uw": self.adc_w * 1e6,
            "dsp_uw": self.dsp_w * 1e6,
            "link_uw": self.link_w * 1e6,
            "total_uw": self.total_w * 1e6,
        }


def chain_power(config: ChainConfig, model: PowerModel | None = None) -> PowerBreakdown:
    """Return the per-channel power of a configuration."""
    model = model or PowerModel()
    return PowerBreakdown(
        lna_w=model.v_supply_v * config.i_total_a,
        adc_w=model.adc_fom_j * (1 << config.bits) * config.fs_hz,
        dsp_w=model.energy_per_multiply_j * config.multiplies_per_sample * config.fs_hz,
        link_w=model.energy_per_bit_j * config.bit_rate_bps,
    )
