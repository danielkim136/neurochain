"""Parameters, word widths and the integer reference of the SBP chain.

The chain is U3's winning feature path with U5's canceller in front::

    8-bit code -> template canceller -> 300 Hz high-pass -> |.| -> sum over 50 ms

Each stage already has an integer Python version (:mod:`neurochain.stim.cancel`,
:mod:`neurochain.fixedpoint`). :class:`ChainSpec` strings them together, and
that composition is the specification: the RTL is correct when its outputs equal
:meth:`ChainSpec.reference` on every sample and every bin.

Hardware also needs something Python integers never ask for: a width for every
wire. Python's integers grow as needed; a register does not. The widths here
are worst-case bounds derived from the parameters, not peaks seen on some data,
so no input can overflow them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from neurochain.adc import IntCodes
from neurochain.fixedpoint import (
    FixedHighpass,
    design_fixed_highpass,
    fixed_highpass,
    fixed_masked_sbp,
)
from neurochain.stim.cancel import BoolArray, cancel_template


def csd_digits(coefficient: int) -> tuple[int, int]:
    """Recode ``|coefficient|`` into canonical signed digits.

    Returns:
        ``(plus, minus)``: two bit masks with ``|coefficient| == plus - minus``
        and no two adjacent bits set across them, the form with the fewest
        non-zero digits. ``rtl/nc_const_mult.sv`` computes the same masks at
        elaboration; each set bit is one shifted term to add or subtract.
    """
    magnitude = abs(coefficient)
    half = magnitude >> 1
    three_halves = magnitude + half
    carry = half ^ three_halves
    return three_halves & carry, half & carry


@dataclass(frozen=True)
class ChainReference:
    """Outputs of the integer reference for a block of channels.

    Attributes:
        cancelled: ``(n_channels, n_samples)`` canceller output, scaled by
            ``2**out_frac_bits`` (quarter-LSB steps by default).
        valid: Mask of the same shape, false on blanked samples.
        filtered: ``(n_channels, n_samples)`` high-pass output, same scale.
        sums: ``(n_channels, n_bins)`` sum of ``|filtered|`` over valid samples.
        counts: ``(n_channels, n_bins)`` number of valid samples.
        accumulator_peak: Largest magnitude the filter's accumulator reached.
    """

    cancelled: IntCodes
    valid: BoolArray
    filtered: IntCodes
    sums: IntCodes
    counts: IntCodes
    accumulator_peak: int


@dataclass(frozen=True)
class ChainSpec:
    """Everything that fixes the hardware.

    Attributes:
        n_channels: Channels sharing the datapath.
        code_bits: ADC resolution.
        fs_hz: Sample rate per channel.
        corner_hz: High-pass corner.
        n_taps: Samples after each pulse the template covers.
        template_frac_bits: Fractional bits of the stored template.
        out_frac_bits: Fractional bits the canceller hands to the filter.
        adapt_shift: Template step size is ``2**-adapt_shift``.
        coef_frac_bits: Fractional bits of the filter coefficients.
        bin_s: Feature bin.
        y_bits_override: Force the filter's output width, to see what a
            narrower register does; ``None`` uses the worst-case width.
        shift_add: Build the filter's constant multiplications from shifts and
            adds (true) or write ``*`` and leave them to the synthesiser.
    """

    n_channels: int = 96
    code_bits: int = 8
    fs_hz: float = 2000.0
    corner_hz: float = 300.0
    n_taps: int = 16
    template_frac_bits: int = 6
    out_frac_bits: int = 2
    adapt_shift: int = 4
    coef_frac_bits: int = 8
    bin_s: float = 0.05
    y_bits_override: int | None = None
    shift_add: bool = True

    def __post_init__(self) -> None:
        """Reject parameters the RTL does not support."""
        if self.n_channels < 1 or self.n_taps < 2:
            raise ValueError("need at least one channel and two template taps")
        if not 0 <= self.out_frac_bits <= self.template_frac_bits:
            raise ValueError("need 0 <= out_frac_bits <= template_frac_bits")
        if self.adapt_shift < 0 or self.coef_frac_bits < 1:
            raise ValueError("need adapt_shift >= 0 and coef_frac_bits >= 1")

    # ------------------------------------------------------------------ parameters

    @property
    def code_min(self) -> int:
        """Most negative ADC code."""
        return -(1 << (self.code_bits - 1))

    @property
    def code_max(self) -> int:
        """Most positive ADC code."""
        return (1 << (self.code_bits - 1)) - 1

    @property
    def samples_per_bin(self) -> int:
        """Samples of one channel in a feature bin."""
        return round(self.bin_s * self.fs_hz)

    @cached_property
    def highpass(self) -> FixedHighpass:
        """The quantised filter. No guard bits: the canceller's output carries them."""
        return design_fixed_highpass(
            self.fs_hz, self.corner_hz, frac_bits=self.coef_frac_bits, guard_bits=0
        )

    # ---------------------------------------------------------------- worst cases

    @property
    def cancel_bound(self) -> int:
        """Largest magnitude the canceller can output, in its scaled units.

        Inside a window the output is ``((x << F) - T + half) >> drop`` with the
        template ``T`` anywhere in the code range and ``x`` one step inside it
        (a code on a rail is blanked). Outside a window it is ``x << out_frac``.
        """
        drop = self.template_frac_bits - self.out_frac_bits
        half = (1 << (drop - 1)) if drop else 0
        span = ((1 << self.code_bits) - 2) << self.template_frac_bits
        return max((span + half) >> drop, -self.code_min << self.out_frac_bits)

    @cached_property
    def filter_gains(self) -> tuple[float, float]:
        """Return ``(sum |h[n]|, sum |g[n]|)`` of the quantised filter.

        ``h`` is the impulse response from input to output and ``g`` that of
        the recursion alone (``1 / A(z)``). A bounded input ``|x| <= X`` gives
        ``|y| <= X * sum|h|``, reached by the input that follows the signs of
        ``h`` backwards; rounding injects at most half a step into the
        recursion each sample, which adds at most ``sum|g| / 2``.
        """
        b, a = self.highpass.float_coefficients()
        impulse = np.zeros(4096)
        impulse[0] = 1.0
        h = signal.lfilter(b, a, impulse)
        g = signal.lfilter([1.0], a, impulse)
        return float(np.abs(h).sum()), float(np.abs(g).sum())

    @property
    def y_bound(self) -> int:
        """Largest magnitude the filter's output can reach."""
        h_gain, loop_gain = self.filter_gains
        return int(np.floor(h_gain * self.cancel_bound + 0.5 * loop_gain))

    @property
    def sum_bound(self) -> int:
        """Largest value a bin's accumulator can reach."""
        return self.samples_per_bin * self.y_bound

    # --------------------------------------------------------------------- widths

    @property
    def template_bits(self) -> int:
        """Width of a stored template tap (signed)."""
        return self.code_bits + self.template_frac_bits

    @property
    def cancel_bits(self) -> int:
        """Width of the canceller's output (signed)."""
        return self.code_bits + self.out_frac_bits + 1

    @property
    def y_bits_needed(self) -> int:
        """Signed width that holds :attr:`y_bound`."""
        return self.y_bound.bit_length() + 1

    @property
    def y_bits(self) -> int:
        """Width of the filter's output register (signed)."""
        return self.y_bits_needed if self.y_bits_override is None else self.y_bits_override

    @property
    def accumulator_bits(self) -> int:
        """Width of the filter's accumulator: the output plus the coefficient fraction."""
        return self.y_bits + self.coef_frac_bits

    @property
    def sum_bits(self) -> int:
        """Width of the bin accumulator (unsigned)."""
        return self.sum_bound.bit_length()

    @property
    def count_bits(self) -> int:
        """Width of the per-bin sample counters (unsigned)."""
        return self.samples_per_bin.bit_length()

    @property
    def tap_index_bits(self) -> int:
        """Width of the position inside a pulse window."""
        return (self.n_taps - 1).bit_length()

    @property
    def state_bits(self) -> dict[str, int]:
        """Bits of state per channel, by the memory that holds them."""
        return {
            "control": self.tap_index_bits + 1,
            "template": self.n_taps * self.template_bits,
            "filter": 2 * self.cancel_bits + 2 * self.y_bits,
            "bin": self.sum_bits + 2 * self.count_bits,
        }

    @property
    def adders_per_sample(self) -> int:
        """Additions the datapath performs on one sample, counted from the RTL.

        Canceller: error, rounding, template update. Filter: two for the second
        difference, then one per shifted term of the three coefficients (the
        rounding constant takes the place of the first term's missing adder).
        Bin: magnitude, sum, two counters. Control: the tap index.
        """
        filt = self.highpass
        terms = sum(
            bin(plus).count("1") + bin(minus).count("1")
            for plus, minus in map(csd_digits, (filt.b0, filt.a1, filt.a2))
        )
        return 3 + 2 + terms + 4 + 1

    def memory_bits_per_sample(self, window_fraction: float) -> float:
        """Return the memory bits read plus written for one sample of one channel.

        Every sample reads and writes the control, filter and bin words. The
        template is touched one tap at a time and only inside a pulse window.

        Args:
            window_fraction: Share of samples that fall inside a pulse window.
        """
        bits = self.state_bits
        always = bits["control"] + bits["filter"] + bits["bin"]
        return 2.0 * (always + window_fraction * self.template_bits)

    def verilog_parameters(self) -> dict[str, int]:
        """Return the parameters of ``nc_sbp_chain`` that realise this specification."""
        filt = self.highpass
        return {
            "N_CH": self.n_channels,
            "CODE_W": self.code_bits,
            "N_TAPS": self.n_taps,
            "TPL_FRAC": self.template_frac_bits,
            "OUT_FRAC": self.out_frac_bits,
            "ADAPT_SHIFT": self.adapt_shift,
            "COEF_FRAC": self.coef_frac_bits,
            "B0": filt.b0,
            "A1": filt.a1,
            "A2": filt.a2,
            "Y_W": self.y_bits,
            "BIN_LEN": self.samples_per_bin,
            "SUM_W": self.sum_bits,
            "SHIFT_ADD": int(self.shift_add),
        }

    # ------------------------------------------------------------------ reference

    def check_onsets(self, onsets: NDArray[np.int64]) -> None:
        """Raise if two pulses are closer than the template is long.

        The Python canceller and the RTL then mean different things (the first
        reprocesses the overlap, the second restarts the window), so the case
        is outside the specification.
        """
        if onsets.size > 1 and int(np.diff(onsets).min()) < self.n_taps:
            raise ValueError(f"pulses must be at least {self.n_taps} samples apart")

    def reference(
        self, codes: NDArray[np.integer[Any]], onsets: NDArray[np.int64]
    ) -> ChainReference:
        """Run the integer Python chain: the specification the RTL must match.

        Args:
            codes: ``(n_channels, n_samples)`` or ``(n_samples,)`` signed ADC
                codes.
            onsets: Sample index of each pulse onset, shared by the channels.
        """
        self.check_onsets(onsets)
        rows = np.atleast_2d(np.asarray(codes, dtype=np.int64))
        cancelled, valid = cancel_template(
            rows,
            onsets,
            n_taps=self.n_taps,
            shift=self.adapt_shift,
            code_min=self.code_min,
            code_max=self.code_max,
            frac_bits=self.template_frac_bits,
            out_frac_bits=self.out_frac_bits,
        )
        filtered = np.empty_like(cancelled)
        peak = 0
        for i, row in enumerate(cancelled):
            filtered[i], row_peak = fixed_highpass(row, self.highpass)
            peak = max(peak, row_peak)
        sums, counts = fixed_masked_sbp(filtered, valid, self.samples_per_bin)
        return ChainReference(cancelled, valid, filtered, sums, counts, peak)
