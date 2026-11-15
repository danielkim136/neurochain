"""A linear latent alignment: keep the decoder, re-find its input space.

Population activity in motor cortex moves in far fewer dimensions than there
are electrodes. If the decoder reads those few latent variables instead of the
channels, then on a later day the job is to find the same variables again from
whatever the electrodes now record. This module does that with two pieces of
MATH2099:

* **Principal axes.** The directions of largest variance of the standardised
  features are the leading right singular vectors of the data matrix (the
  eigenvectors of its covariance). Projecting on the first ``d`` gives ``d``
  latent variables.
* **Orthogonal Procrustes.** The later day's axes span nearly the same space
  but in a different basis (order, sign and mixture of the axes are arbitrary).
  The rotation ``O`` that minimises ``||A_new O - A_ref||`` over orthogonal
  matrices is ``U V^T`` from the singular value decomposition
  ``A_new^T A_ref = U S V^T``.

Nothing here uses behaviour: the later day's axes come from unlabelled
features, so this is recalibration without asking the user to do anything.
It is the linear idea behind Degenhart et al. (2020); Karpowicz et al. (2025)
align nonlinear dynamics instead and do better on hard cases.
"""

from __future__ import annotations

import numpy as np

from neurochain.recording import FloatArray


def principal_axes(features: FloatArray, n_latents: int) -> FloatArray:
    """Return the leading principal axes of a feature matrix.

    Args:
        features: ``(n_bins, n_channels)`` features, already standardised.
        n_latents: Number of axes to keep.

    Returns:
        ``(n_channels, n_latents)`` with orthonormal columns, ordered by
        variance explained.

    Raises:
        ValueError: If more axes are asked for than the data can have.
    """
    if not 0 < n_latents <= min(features.shape):
        raise ValueError("n_latents must be between 1 and min(n_bins, n_channels)")
    centred = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return np.asarray(vt[:n_latents].T, dtype=np.float64)


def procrustes_rotation(axes: FloatArray, reference: FloatArray) -> FloatArray:
    """Return the orthogonal matrix that best maps one set of axes onto another.

    Args:
        axes: ``(n_channels, d)`` axes to rotate.
        reference: ``(n_channels, d)`` axes to match.

    Returns:
        ``(d, d)`` orthogonal ``O`` minimising the Frobenius norm of
        ``axes @ O - reference``.
    """
    u, _, vt = np.linalg.svd(axes.T @ reference)
    return np.asarray(u @ vt, dtype=np.float64)


def aligned_axes(features: FloatArray, reference: FloatArray) -> FloatArray:
    """Return a new day's principal axes, rotated onto the reference day's.

    Args:
        features: ``(n_bins, n_channels)`` standardised, unlabelled features
            from the new day (a calibration block is enough).
        reference: ``(n_channels, d)`` principal axes of the day the decoder
            was fitted on.

    Returns:
        ``(n_channels, d)``: project the new day's features on these and the
        latent variables line up with the reference day's.
    """
    axes = principal_axes(features, reference.shape[1])
    return np.asarray(axes @ procrustes_rotation(axes, reference), dtype=np.float64)
