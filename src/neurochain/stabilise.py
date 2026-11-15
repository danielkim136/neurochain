"""A causal per-channel stabiliser: running mean and running scale.

A decoder standardises each feature with two constants stored when it was
fitted, a mean and a scale. When the electrode ages those constants go stale:
the channel's level moves and its modulation shrinks, and the decoder reads
the change as behaviour.

The stabiliser replaces the stored constants with running estimates. Per
channel and per bin it keeps two registers:

    mean  <- mean  + (x - mean) / 2**shift
    mad   <- mad   + (|x - mean| - mad) / 2**shift

Each is a first-order IIR low-pass, ``y[n] = (1 - a) y[n-1] + a x[n]`` with
``a = 2**-shift`` (ELEC3104), so the "division" is a wire shift and there is no
multiplier (ELEC2141). The scale is tracked as a mean absolute deviation, which
needs no squaring and no square root. A bin is normalised with the registers
as they were *before* it arrived, the way a register's output is used in the
clock cycle before it is updated.

The registers start from the constants the decoder was fitted with, so a
session in which nothing has changed is normalised as before.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from neurochain.recording import FloatArray, IntArray

Mode = Literal["fixed", "mean", "zscore"]
MODES: tuple[Mode, ...] = ("fixed", "mean", "zscore")

_TINY = 1.0e-9


def running_average(x: FloatArray, start: FloatArray, shift: int) -> FloatArray:
    """Return the running average as it stood *before* each bin.

    Args:
        x: ``(n_bins, n_channels)`` values in time order.
        start: ``(n_channels,)`` value of the register before the first bin.
        shift: The update is ``(x - average) >> shift``; the time constant is
            ``2**shift`` bins.

    Returns:
        ``(n_bins, n_channels)``; row ``n`` uses bins ``0 .. n-1`` only.
    """
    if shift < 0:
        raise ValueError("shift must be non-negative")
    alpha = 2.0 ** (-shift)
    start = np.asarray(start, dtype=np.float64)
    after, _ = signal.lfilter(
        [alpha], [1.0, alpha - 1.0], x, axis=0, zi=((1.0 - alpha) * start)[np.newaxis, :]
    )
    return np.concatenate((start[np.newaxis, :], after[:-1]), axis=0)


def running_statistics(
    x: FloatArray, mean: FloatArray, mad: FloatArray, shift: int
) -> tuple[FloatArray, FloatArray]:
    """Return the running mean and mean absolute deviation before each bin."""
    centre = running_average(x, mean, shift)
    spread = running_average(np.abs(x - centre), mad, shift)
    return centre, spread


def normalise(
    x: FloatArray,
    mean: FloatArray,
    scale: FloatArray,
    mad: FloatArray,
    *,
    mode: Mode,
    shift: int = 10,
    max_gain: float = 4.0,
    gate: IntArray | None = None,
    start: tuple[FloatArray, FloatArray] | None = None,
) -> FloatArray:
    """Standardise features with fixed or running constants.

    Args:
        x: ``(n_bins, n_channels)`` features in time order.
        mean: ``(n_channels,)`` mean stored when the decoder was fitted.
        scale: ``(n_channels,)`` standard deviation stored with it.
        mad: ``(n_channels,)`` mean absolute deviation stored with it; the
            running scale is ``scale * running_mad / mad``, so an unchanged
            channel keeps its stored scale.
        mode: ``"fixed"``: the stored constants. ``"mean"``: a running mean and
            the stored scale. ``"zscore"``: running mean and running scale.
        shift: Time constant of the registers, ``2**shift`` bins.
        max_gain: Largest factor by which the running scale may amplify a
            channel whose spread has shrunk. Without a limit a channel that
            has gone nearly silent would be blown up to full size, noise and
            all.
        gate: Optional ``(n_bins,)`` integer label. Each label keeps its own
            registers, which only see their own bins; a stimulator that knows
            when it is on can supply it.
        start: Optional ``(mean, mad)`` the registers hold before the first
            bin. By default they hold the stored constants (a cold start: the
            implant was last used when the decoder was fitted).

    Returns:
        ``(n_bins, n_channels)`` standardised features.
    """
    x = np.asarray(x, dtype=np.float64)
    safe_scale = scale + _TINY
    if mode == "fixed":
        return np.asarray((x - mean) / safe_scale, dtype=np.float64)
    if mode not in ("mean", "zscore"):
        raise ValueError(f"unknown mode {mode!r}")
    out = np.empty_like(x)
    labels = np.zeros(x.shape[0], dtype=np.int64) if gate is None else np.asarray(gate)
    start_mean, start_mad = (mean, mad) if start is None else start
    for label in np.unique(labels):
        rows = labels == label
        centre, spread = running_statistics(x[rows], start_mean, start_mad, shift)
        if mode == "mean":
            out[rows] = (x[rows] - centre) / safe_scale
            continue
        ratio = np.maximum((spread + _TINY) / (mad + _TINY), 1.0 / max_gain)
        out[rows] = (x[rows] - centre) / (safe_scale * ratio)
    return out


def operations_per_bin(mode: Mode, quotient_bits: int = 12) -> int:
    """Return the add-or-subtract steps the stabiliser needs per channel per bin.

    * Running mean: one subtraction and one shifted addition, plus the
      subtraction that centres the feature.
    * Running scale: an absolute difference and a shifted addition, plus one
      division, done bit-serially as one subtract-and-shift per quotient bit.

    None of them is a multiplication. U6 charges each as a full multiply of the
    power model, as U5 does for its canceller, so the cost is an upper bound.
    """
    if mode == "fixed":
        return 1
    if mode == "mean":
        return 3
    return 5 + quotient_bits
