"""How a stimulation current reaches a recording electrode.

Below a few kilohertz cortex behaves as a resistor: Logothetis et al. (2007)
measured an impedance that is flat with frequency and nearly the same in every
direction. So the potential follows the current instantly, with no filtering,
and the only question is how large it is.

For a point current source ``I`` in a uniform medium of conductivity ``sigma``
the potential at distance ``r`` is the resistive analogue of a point charge::

    V = I / (4 pi sigma r)

* **Monopolar** stimulation returns its current through a distant electrode.
  The artifact falls as ``1/r``.
* **Bipolar** stimulation returns it through a neighbour: a source and a sink a
  distance ``d`` apart, a current dipole. Far away the two nearly cancel and
  the artifact falls as ``I d cos(theta) / (4 pi sigma r**2)``.

An amplifier records the difference between its electrode and a reference. The
reference here is distant unless a position is given, which is the worst case:
none of the artifact is common to both.

Units: positions in micrometres, currents in microamps, potentials in
microvolts.
"""

from __future__ import annotations

import numpy as np

from neurochain.recording import FloatArray

GREY_MATTER_S_PER_M = 0.3
"""Assumed conductivity of cortex. Results scale as ``1 / sigma``."""

UTAH_PITCH_UM = 400.0
ELECTRODE_RADIUS_UM = 25.0
"""Closest approach used when a site coincides with a source."""


def monopole_uv(
    current_ua: float, distance_um: FloatArray | float, sigma_s_per_m: float = GREY_MATTER_S_PER_M
) -> FloatArray:
    """Return the potential of a point source at a distance, in microvolts."""
    r_m = np.maximum(np.asarray(distance_um, dtype=np.float64), ELECTRODE_RADIUS_UM) * 1.0e-6
    return np.asarray(current_ua / (4.0 * np.pi * sigma_s_per_m * r_m), dtype=np.float64)


def dipole_far_field_uv(
    current_ua: float,
    separation_um: float,
    distance_um: FloatArray | float,
    *,
    cos_theta: float = 1.0,
    sigma_s_per_m: float = GREY_MATTER_S_PER_M,
) -> FloatArray:
    """Return the far-field potential of a source and sink pair, in microvolts.

    Valid when ``distance_um`` is several times ``separation_um``.
    """
    r_m = np.asarray(distance_um, dtype=np.float64) * 1.0e-6
    moment = current_ua * separation_um * 1.0e-6
    return np.asarray(moment * cos_theta / (4.0 * np.pi * sigma_s_per_m * r_m**2), dtype=np.float64)


def potential_uv(
    points_um: FloatArray,
    sources_um: FloatArray,
    currents_ua: FloatArray,
    sigma_s_per_m: float = GREY_MATTER_S_PER_M,
) -> FloatArray:
    """Return the potential at each point from several point sources.

    Args:
        points_um: ``(n_points, 3)`` positions.
        sources_um: ``(n_sources, 3)`` positions of the current sources.
        currents_ua: ``(n_sources,)`` current of each; negative for a sink.
        sigma_s_per_m: Conductivity of the medium.
    """
    points = np.atleast_2d(np.asarray(points_um, dtype=np.float64))
    sources = np.atleast_2d(np.asarray(sources_um, dtype=np.float64))
    distance = np.linalg.norm(points[:, np.newaxis, :] - sources[np.newaxis, :, :], axis=2)
    each = monopole_uv(1.0, distance, sigma_s_per_m) * np.asarray(currents_ua, dtype=np.float64)
    return np.asarray(each.sum(axis=1), dtype=np.float64)


def utah_positions_um(
    n_side: int = 10, pitch_um: float = UTAH_PITCH_UM, *, drop_corners: bool = True
) -> FloatArray:
    """Return electrode tip positions of a Utah array, row by row, centred on the origin.

    A 10 by 10 array has 96 wired electrodes: the four corners are not
    connected. With ``drop_corners`` they are left out.
    """
    axis = (np.arange(n_side) - (n_side - 1) / 2.0) * pitch_um
    x, y = np.meshgrid(axis, axis)
    sites = np.column_stack([x.ravel(), y.ravel(), np.zeros(n_side * n_side)])
    if drop_corners:
        corners = {0, n_side - 1, n_side * (n_side - 1), n_side * n_side - 1}
        sites = sites[[i for i in range(n_side * n_side) if i not in corners]]
    return np.asarray(sites, dtype=np.float64)


def artifact_amplitude_uv(
    sites_um: FloatArray,
    stim_um: FloatArray,
    current_ua: float,
    *,
    return_um: FloatArray | None = None,
    reference_um: FloatArray | None = None,
    sigma_s_per_m: float = GREY_MATTER_S_PER_M,
) -> FloatArray:
    """Return the artifact at each recording site per unit of the pulse waveform.

    Multiply by the pulse's ``current / amplitude`` waveform to get the
    potential over time.

    Args:
        sites_um: ``(n_sites, 3)`` recording electrodes.
        stim_um: Position of the stimulating electrode.
        current_ua: Pulse amplitude.
        return_um: Position of the return electrode; ``None`` for a distant
            return (monopolar).
        reference_um: Position of the recording reference; ``None`` for a
            distant reference, which sees none of the artifact.
        sigma_s_per_m: Conductivity of the medium.
    """
    sources = [np.asarray(stim_um, dtype=np.float64)]
    currents = [current_ua]
    if return_um is not None:
        sources.append(np.asarray(return_um, dtype=np.float64))
        currents.append(-current_ua)
    stacked = np.stack(sources)
    weights = np.asarray(currents, dtype=np.float64)
    amplitude = potential_uv(sites_um, stacked, weights, sigma_s_per_m)
    if reference_um is not None:
        amplitude = amplitude - potential_uv(reference_um, stacked, weights, sigma_s_per_m)[0]
    return amplitude
