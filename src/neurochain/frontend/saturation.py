"""The amplifier when the input is too large: saturation and recovery.

Everything in U1 to U4 treats the amplifier as linear and its rails as a hard
clip. A stimulation artifact is hundreds of times larger than a spike, and what
matters is not only that the output hits a rail but what state the amplifier is
left in afterwards. This module is the large-signal model.

**The circuit** is U1's capacitive-feedback amplifier (``C1`` in, ``C2`` and
``R2`` in feedback, an OTA driving ``CL``), with two nonlinearities added:

* **The differential pair.** In weak inversion a differential pair steers its
  tail current as ``I_max * tanh(gm * v / I_max)``. For small ``v`` this is
  ``gm * v``, the small-signal model of ELEC2133; for large ``v`` the output
  current stops growing at ``I_max``, which is where slew rate comes from.
* **The rails.** The output cannot pass ``±swing``. A stiff conductance that
  only conducts beyond the swing holds it there.

**Why saturation has memory.** A charge-balanced pulse has no DC content, so a
linear high-pass has nothing to remember: the charge the first phase moves
through ``R2`` is taken back by the second, and the two cancel. A rail breaks
that. While the output sits on a rail the feedback loop is open, the summing
node is no longer a virtual ground, and the current through ``R2`` is set by
the rail voltage instead of by the input. A 60 mV phase and a 30 mV phase both
give the same 0.8 V at the output, so a phase twice as long now moves twice
the charge and the cancellation is gone. When the input returns to zero the
difference is still on the summing node, the output is offset by about
``q / C2``, and the amplifier works it off at the pace of its slowest pole:

* With a 1 Hz corner the two poles are far apart (1 Hz and 7.5 kHz) and the
  slow one is ``R2 C2``: 160 ms.
* With a 300 Hz corner and a 1 kHz bandwidth the two corners are so close that
  the closed loop has a complex pole pair (:meth:`SaturatingAmplifier.poles`):
  the tail is a damped oscillation near 270 Hz whose envelope falls with a time
  constant of 0.41 ms.

**State equations.** With ``v_p`` the voltage on the electrode side of ``C1``,
``v_x`` the summing node and ``v_o`` the output, and ``C_s = C1 + C2 + Cin``::

    q   = C_s v_x - C1 v_p - C2 v_o              (charge on the summing node)
    dq/dt = (v_o - v_x) / R2
    s   = C_e v_o - (C2/C_s) q - (C2 C1/C_s) v_p,   C_e = CL + C2 - C2**2/C_s
    ds/dt = -[ v_o/Ro + (v_o - v_x)/R2 + i_ota(v_x) + i_clamp(v_o) ]

Choosing ``s`` this way keeps the derivative of the input out of the equations,
so a step at the input is handled exactly. The same equations, element for
element, are what :func:`neurochain.frontend.spice.transient_netlist` gives to
ngspice, and the two are compared in the tests.

The source is ideal here (the electrode's impedance is left out). It is a few
percent of ``1 / (s C1)`` in the band; the SPICE run with the electrode in
place measures what that omission costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from neurochain.frontend.model import FrontEnd
from neurochain.recording import FloatArray


@dataclass(frozen=True)
class SaturatingAmplifier:
    """Large-signal model of the capacitive-feedback amplifier.

    Attributes:
        c1_f: Input coupling capacitor.
        c2_f: Feedback capacitor.
        c_in_f: OTA input capacitance.
        c_load_f: Load capacitor.
        r2_ohm: Feedback resistance.
        r_out_ohm: OTA output resistance.
        gm_s: Small-signal transconductance.
        i_max_a: Largest output current: the tail current of the input pair.
        swing_v: Output excursion at which the rails take over.
        clamp_s: Conductance of the rail clamp beyond the swing. 1 mS is a
            stand-in for an output transistor leaving saturation; any value
            much larger than ``gm`` gives the same waveforms.
    """

    c1_f: float
    c2_f: float
    c_in_f: float
    c_load_f: float
    r2_ohm: float
    r_out_ohm: float
    gm_s: float
    i_max_a: float
    swing_v: float
    clamp_s: float = 1.0e-3

    @classmethod
    def from_frontend(cls, frontend: FrontEnd) -> SaturatingAmplifier:
        """Take every value from the linear model, so the two agree for small signals.

        The tail current is half the supply current: a current-mirror OTA
        splits the supply into the pair and its mirrored copy.
        """
        lna, temperature_k = frontend.lna, frontend.temperature_k
        return cls(
            c1_f=lna.c1_f,
            c2_f=lna.c2_f,
            c_in_f=lna.c_in_f,
            c_load_f=lna.c_load_f,
            r2_ohm=lna.r2_ohm,
            r_out_ohm=lna.r_out_ohm(temperature_k),
            gm_s=lna.gm_s(temperature_k),
            i_max_a=lna.i_total_a / 2.0,
            swing_v=lna.output_swing_v,
        )

    @property
    def c_sum_f(self) -> float:
        """Total capacitance on the summing node."""
        return self.c1_f + self.c2_f + self.c_in_f

    @property
    def c_eff_f(self) -> float:
        """Capacitance the OTA's output current sees."""
        return self.c_load_f + self.c2_f - self.c2_f**2 / self.c_sum_f

    def poles(self) -> NDArray[np.complex128]:
        """Return the two small-signal poles of the closed loop, in rad/s.

        The roots of the network's characteristic equation with an ideal
        source: ``(s C_s + G2)(s (CL + C2) + Go + G2) - (s C2 + G2)(s C2 + G2 - gm)``.
        A complex pair means the amplifier rings after a disturbance.
        """
        g_o, g_2 = 1.0 / self.r_out_ohm, 1.0 / self.r2_ohm
        left = np.polymul([self.c_sum_f, g_2], [self.c_load_f + self.c2_f, g_o + g_2])
        right = np.polymul([self.c2_f, g_2], [self.c2_f, g_2 - self.gm_s])
        return np.asarray(np.roots(np.polysub(left, right)), dtype=np.complex128)

    @property
    def linear_input_v(self) -> float:
        """Largest slow input the amplifier passes without touching a rail."""
        return self.swing_v * self.c2_f / self.c1_f

    def simulate(
        self, t_bp_s: FloatArray, v_bp_v: FloatArray, t_eval_s: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Integrate the amplifier's response to a piecewise-linear input.

        The amplifier starts at rest. The input is zero before the first
        breakpoint and holds its last value after the last one.

        Args:
            t_bp_s: Breakpoint times, increasing, starting at or after 0.
            v_bp_v: Input voltage at each breakpoint.
            t_eval_s: Increasing times at which to report the output.

        Returns:
            ``(v_out_v, v_x_v)`` at ``t_eval_s``: the output and the summing
            node. The amplifier inverts, so a positive input drives the output
            negative.
        """
        cs, ce = self.c_sum_f, self.c_eff_f
        c1, c2 = self.c1_f, self.c2_f
        k_a, k_p = c2 / ce, c2 * c1 / (cs * ce)
        g_o, g_2 = 1.0 / self.r_out_ohm, 1.0 / self.r2_ohm
        gm, i_max, swing, clamp = self.gm_s, self.i_max_a, self.swing_v, self.clamp_s

        def nodes(a: float, b: float, v_p: float) -> tuple[float, float]:
            v_o = b + k_a * a + k_p * v_p
            return v_o, a + (c1 * v_p + c2 * v_o) / cs

        def rhs(t: float, y: list[float], t0: float, v0: float, slope: float) -> list[float]:
            v_o, v_x = nodes(y[0], y[1], v0 + slope * (t - t0))
            over = max(v_o - swing, 0.0) + min(v_o + swing, 0.0)
            current = (
                v_o * g_o + (v_o - v_x) * g_2 + i_max * math.tanh(gm * v_x / i_max) + clamp * over
            )
            return [(v_o - v_x) * g_2 / cs, -current / ce]

        t_eval = np.asarray(t_eval_s, dtype=np.float64)
        times = np.asarray(t_bp_s, dtype=np.float64)
        values = np.asarray(v_bp_v, dtype=np.float64)
        end = max(float(t_eval[-1]), float(times[-1])) + 1.0e-9
        edges = np.concatenate(([0.0], times[times > 0.0], [end]))
        levels = np.interp(edges, times, values, left=0.0)
        v_out = np.zeros_like(t_eval)
        v_x = np.zeros_like(t_eval)
        state = [0.0, 0.0]
        for i in range(edges.size - 1):
            t0, t1 = float(edges[i]), float(edges[i + 1])
            if t1 <= t0:
                continue
            slope = (float(levels[i + 1]) - float(levels[i])) / (t1 - t0)
            solution = integrate.solve_ivp(
                rhs,
                (t0, t1),
                state,
                method="LSODA",
                args=(t0, float(levels[i]), slope),
                dense_output=True,
                rtol=1.0e-9,
                atol=1.0e-13,
            )
            if not solution.success:  # pragma: no cover - the solver has not failed in use
                raise RuntimeError(f"amplifier integration failed: {solution.message}")
            inside = (t_eval >= t0) & (t_eval < t1)
            if inside.any():
                a, b = solution.sol(t_eval[inside])
                v_p = float(levels[i]) + slope * (t_eval[inside] - t0)
                v_o = b + k_a * a + k_p * v_p
                v_out[inside] = v_o
                v_x[inside] = a + (c1 * v_p + c2 * v_o) / cs
            state = [float(solution.y[0, -1]), float(solution.y[1, -1])]
        return v_out, v_x


def pulse_response_uv(
    amplifier: SaturatingAmplifier,
    t_bp_s: FloatArray,
    unit_waveform: FloatArray,
    amplitude_uv: float,
    t_eval_s: FloatArray,
    *,
    gain: float,
) -> FloatArray:
    """Return the amplifier's output for one pulse, referred to the input in µV.

    The output is divided by ``-gain``: the amplifier inverts, and the rest of
    the project refers everything to the electrode with a positive gain.

    Args:
        amplifier: The large-signal model.
        t_bp_s: Breakpoint times of the pulse waveform.
        unit_waveform: Waveform at each breakpoint, with the first phase -1.
        amplitude_uv: Artifact amplitude at the electrode: the size of the
            first phase, in microvolts.
        t_eval_s: Times after pulse onset at which to report.
        gain: Gain used to refer the output to the input.
    """
    v_out, _ = amplifier.simulate(t_bp_s, unit_waveform * amplitude_uv * 1.0e-6, t_eval_s)
    return np.asarray(-v_out / gain * 1.0e6, dtype=np.float64)


def pulse_kernel(
    amplifier: SaturatingAmplifier,
    t_bp_s: FloatArray,
    unit_waveform: FloatArray,
    amplitude_uv: float,
    fs_hz: float,
    *,
    gain: float,
    n_phases: int = 1,
    max_s: float = 1.2,
    floor_uv: float = 0.02,
) -> FloatArray:
    """Return the sampled response to one pulse, for :func:`neurochain.stim.pulse.pulse_train`.

    Returns:
        ``(n_phases, n_kernel)`` in microvolts referred to the input. Row ``p``
        holds the response at ``(m + p / n_phases) / fs_hz`` after the onset.
        The kernel ends where the response has fallen below ``floor_uv`` for
        good, or at ``max_s``.
    """
    fine_hz = fs_hz * n_phases
    t = np.arange(round(max_s * fine_hz)) / fine_hz
    y = pulse_response_uv(amplifier, t_bp_s, unit_waveform, amplitude_uv, t, gain=gain)
    above = np.flatnonzero(np.abs(y) > floor_uv)
    length = 1 if above.size == 0 else min(int(above[-1]) // n_phases + 2, y.size // n_phases)
    return np.asarray(y[: length * n_phases].reshape(length, n_phases).T, dtype=np.float64)


def recovery_time_s(
    t_s: FloatArray, y_uv: FloatArray, pulse_end_s: float, threshold_uv: float
) -> float:
    """Return how long after the pulse ends the output stays outside ``±threshold_uv``.

    Zero means the output was already inside the threshold when the pulse
    ended. The answer is the last reported time above the threshold, so it is
    only as fine as ``t_s``.
    """
    above = np.flatnonzero(np.abs(y_uv) > threshold_uv)
    if above.size == 0:
        return 0.0
    return max(0.0, float(t_s[above[-1]]) - pulse_end_s)
