"""Removing a stimulation artifact from ADC codes: blanking and template subtraction.

The implant fires the stimulator itself, so it knows when every pulse starts.
Both methods use that and nothing else.

**Blanking** throws away a fixed window after each pulse. It needs no
arithmetic and cannot leave a residue, but the window has to cover the slowest
recovery, and everything in it is lost, neural signal included.

**Adaptive template subtraction** exploits the artifact being the same after
every pulse. It keeps a running average of the last pulses' waveforms, one
value per sample position ``k`` after the trigger, and subtracts it::

    e[k]  = x[k] - T[k]                 (output)
    T[k] += (x[k] - T[k]) >> shift      (update)

This is the LMS algorithm of ELEC3104 in its smallest form. The reference
input is an impulse at the trigger, so tap ``k`` only ever sees the ``k``-th
sample, the gradient is the error itself, and the step size ``mu = 2**-shift``
is a bit shift: no multiplier anywhere. The update is also a one-pole low-pass
filter along the pulse index with a memory of about ``2**shift`` pulses, which
is what lets the template follow an artifact that drifts as electrodes age.

Two limits follow from the arithmetic:

* The average contains a little of the neural signal from recent pulses, so
  subtracting it adds noise: the output's noise power is ``1 + mu / (2 - mu)``
  times the input's. A larger ``shift`` costs less noise and adapts slower.
* A sample that hit the ADC's rail holds no information to recover. Those
  samples are still blanked, one at a time, by looking at the code.

All functions work along the last axis, so a batch of conditions can share one
set of pulse times.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurochain.adc import IntCodes
from neurochain.recording import FloatArray

BoolArray = NDArray[np.bool_]
Fill = Literal["zero", "hold"]


def _windows(onsets: NDArray[np.int64], length: int, n_samples: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` of the window after each onset that lies in the record."""
    return [
        (int(n), min(int(n) + length, n_samples)) for n in onsets.tolist() if 0 <= n < n_samples
    ]


def blank(
    codes: IntCodes, onsets: NDArray[np.int64], n_blank: int, *, fill: Fill = "zero"
) -> tuple[IntCodes, BoolArray]:
    """Discard ``n_blank`` samples from each pulse onset.

    Args:
        codes: ``(..., n_samples)`` ADC codes.
        onsets: Sample index of each pulse onset.
        n_blank: Samples to discard per pulse.
        fill: What the discarded samples are replaced with before any filter:
            ``"zero"`` for a band-limited signal, ``"hold"`` (the last good
            sample, as a sample-and-hold would) for one that carries an offset.

    Returns:
        ``(codes, valid)``: the codes with the windows filled, and a mask that
        is false inside them.
    """
    out = np.array(codes, dtype=np.int64)
    valid = np.ones(out.shape, dtype=bool)
    for start, stop in _windows(onsets, n_blank, out.shape[-1]):
        if fill == "hold" and start > 0:
            out[..., start:stop] = out[..., start - 1 : start]
        else:
            out[..., start:stop] = 0
        valid[..., start:stop] = False
    return out, valid


def cancel_template(
    codes: IntCodes,
    onsets: NDArray[np.int64],
    *,
    n_taps: int,
    shift: int,
    code_min: int,
    code_max: int,
    frac_bits: int = 6,
    out_frac_bits: int = 2,
) -> tuple[IntCodes, BoolArray]:
    """Subtract an adaptive template after each pulse, in integer arithmetic.

    The template is stored with ``frac_bits`` fractional bits so that a shift
    by ``shift`` does not throw the update away. The first pulse loads the
    template and is blanked, since there is nothing to subtract yet. Samples at
    either end of the ADC's range are blanked: the artifact there is unknown.

    The output keeps ``out_frac_bits`` of the template's fractional bits, so it
    is scaled by ``2**out_frac_bits``. The signal here is about one LSB RMS,
    and rounding the template to a whole LSB before subtracting would add a
    twelfth of an LSB squared of noise to it. The high-pass filter that follows
    shifts its input left by two guard bits anyway (U2); the canceller fills
    those bits with information instead of zeros.

    This function is the reference the U7 RTL has to match bit for bit.

    Args:
        codes: ``(..., n_samples)`` signed ADC codes.
        onsets: Sample index of each pulse onset; windows must not overlap.
        n_taps: Samples after each onset the template covers.
        shift: Adaptation shift; the step size is ``2**-shift``.
        code_min: Most negative ADC code.
        code_max: Most positive ADC code.
        frac_bits: Fractional bits of the stored template.
        out_frac_bits: Fractional bits kept in the output, at most ``frac_bits``.

    Returns:
        ``(codes, valid)``: the cancelled codes scaled by ``2**out_frac_bits``
        (0 where blanked) and a mask that is false on blanked samples.
    """
    if shift < 0 or not 0 <= out_frac_bits <= frac_bits:
        raise ValueError("need shift >= 0 and 0 <= out_frac_bits <= frac_bits")
    drop = frac_bits - out_frac_bits
    out = np.array(codes, dtype=np.int64) << out_frac_bits
    valid = np.ones(out.shape, dtype=bool)
    template = np.zeros((*out.shape[:-1], n_taps), dtype=np.int64)
    half = (1 << (drop - 1)) if drop else 0
    loaded = False
    for start, stop in _windows(onsets, n_taps, out.shape[-1]):
        x = np.asarray(codes[..., start:stop], dtype=np.int64)
        taps = template[..., : stop - start]
        if not loaded:
            taps[...] = x << frac_bits
            out[..., start:stop] = 0
            valid[..., start:stop] = False
            loaded = True
            continue
        clipped = (x <= code_min) | (x >= code_max)
        error = ((x << frac_bits) - taps + half) >> drop
        out[..., start:stop] = np.where(clipped, 0, error)
        valid[..., start:stop] = ~clipped
        taps += ((x << frac_bits) - taps) >> shift
    return out, valid


def cancel_template_float(
    x: FloatArray,
    onsets: NDArray[np.int64],
    *,
    n_taps: int,
    rate: float,
    low: float,
    high: float,
) -> tuple[FloatArray, BoolArray]:
    """Floating-point twin of :func:`cancel_template`, with step size ``rate``.

    ``low`` and ``high`` are the values at which the converter clips.
    """
    out = np.array(x, dtype=np.float64)
    valid = np.ones(out.shape, dtype=bool)
    template = np.zeros((*out.shape[:-1], n_taps), dtype=np.float64)
    loaded = False
    for start, stop in _windows(onsets, n_taps, out.shape[-1]):
        segment = np.asarray(x[..., start:stop], dtype=np.float64)
        taps = template[..., : stop - start]
        if not loaded:
            taps[...] = segment
            out[..., start:stop] = 0.0
            valid[..., start:stop] = False
            loaded = True
            continue
        clipped = (segment <= low) | (segment >= high)
        out[..., start:stop] = np.where(clipped, 0.0, segment - taps)
        valid[..., start:stop] = ~clipped
        taps += rate * (segment - taps)
    return out, valid


def hold_last_valid(x: FloatArray, valid: BoolArray) -> FloatArray:
    """Replace each invalid sample with the last valid one before it.

    For a signal that still carries an offset, where a zero would be a step.
    Samples before the first valid one are left as they are.
    """
    index = np.where(valid, np.arange(x.shape[-1]), 0)
    np.maximum.accumulate(index, axis=-1, out=index)
    return np.asarray(np.take_along_axis(x, index, axis=-1), dtype=x.dtype)


def masked_mean_abs(
    x: FloatArray | IntCodes, valid: BoolArray, samples_per_bin: int
) -> tuple[FloatArray, FloatArray]:
    """Return the mean absolute value of the valid samples in each bin.

    This is spiking-band power when some samples have been blanked: the sum of
    ``|x|`` over the samples that survived, divided by how many did. A bin with
    no valid sample repeats the previous bin's value.

    Returns:
        ``(sbp, valid_fraction)``, each ``(..., n_bins)``.
    """
    n_bins = x.shape[-1] // samples_per_bin
    shape = (*x.shape[:-1], n_bins, samples_per_bin)
    keep = valid[..., : n_bins * samples_per_bin].reshape(shape)
    magnitude = np.abs(np.asarray(x[..., : n_bins * samples_per_bin], dtype=np.float64))
    total = (magnitude.reshape(shape) * keep).sum(axis=-1)
    count = keep.sum(axis=-1)
    sbp = total / np.maximum(count, 1)
    if (count == 0).any():
        index = np.where(count > 0, np.arange(n_bins), 0)
        np.maximum.accumulate(index, axis=-1, out=index)
        sbp = np.take_along_axis(sbp, index, axis=-1)
    return np.asarray(sbp, dtype=np.float64), np.asarray(count / samples_per_bin, dtype=np.float64)


def noise_gain(shift: int) -> float:
    """Return the factor by which template subtraction raises noise power.

    The template is an exponentially weighted average of past samples of the
    same noise, with variance ``mu / (2 - mu)`` times the noise's, and it is
    independent of the present sample. Variances add.
    """
    mu = 2.0**-shift
    return 1.0 + mu / (2.0 - mu)
