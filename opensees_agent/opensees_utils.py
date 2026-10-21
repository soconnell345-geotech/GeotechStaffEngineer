"""
Shared OpenSees utilities for all analysis types.

Provides import guard, model initialization, and spectral analysis
helpers used by every OpenSees analysis wrapper.
"""

import math
import numpy as np


def import_ops():
    """Import openseespy with a helpful error message.

    Returns
    -------
    module
        The ``openseespy.opensees`` module.

    Raises
    ------
    ImportError
        If openseespy is not installed.
    """
    try:
        import openseespy.opensees as ops
        return ops
    except ImportError:
        raise ImportError(
            "openseespy is required for OpenSees analyses. "
            "Install with: pip install openseespy"
        )


def has_opensees() -> bool:
    """Return True if openseespy is importable."""
    try:
        import openseespy.opensees  # noqa: F401
        return True
    except (ImportError, RuntimeError):
        return False


def fresh_model(ndm: int = 2, ndf: int = 3):
    """Wipe and initialize a new OpenSees model.

    Parameters
    ----------
    ndm : int
        Number of spatial dimensions (1, 2, or 3).
    ndf : int
        Number of degrees of freedom per node.

    Returns
    -------
    module
        The ``openseespy.opensees`` module, ready to use.
    """
    ops = import_ops()
    ops.wipe()
    ops.model('basic', '-ndm', ndm, '-ndf', ndf)
    return ops


def compute_response_spectrum(accel_g, dt, periods=None, damping=0.05):
    """Compute the pseudo-acceleration response spectrum of a record, every
    period by Nigam and Jennings (1969) (GM-272).

    The record is taken as linear between its samples and each
    oscillator's response to it is solved in closed form, step by step: at
    the points it is the response to that piecewise-linear record, with no
    integration error. A period shorter than 20 record steps uses equal
    substeps of at most a twentieth of the period on the same straight
    lines; between points the largest response is read off the cubic
    through the displacement and velocity at both ends of each step.
    numpy and scipy; does not require openseespy.

    The code is ``geotech_common.response_spectrum``, which seismic_signals'
    response spectrum uses too, so the two give the same Sa (GM-274).

    Parameters
    ----------
    accel_g : array_like
        Ground acceleration time history in g.
    dt : float
        Time step (s).
    periods : array_like, optional
        Spectral periods (s).  Default: 0.01 to 10 s (100 points, log-spaced).
    damping : float
        Damping ratio, at least 0 and below 1.  Default 0.05 (5%).

    Returns
    -------
    periods : numpy.ndarray
        Spectral periods (s).
    Sa : numpy.ndarray
        Pseudo-spectral acceleration (g).
    """
    from geotech_common.response_spectrum import pseudo_spectral_accel
    if periods is None:
        periods = np.logspace(-2, 1, 100)
    periods = np.asarray(periods, dtype=float)
    return periods, pseudo_spectral_accel(accel_g, dt, periods, damping)


#: Order of the Butterworth low-pass in :func:`lowpass` (GM-264).
LOWPASS_ORDER = 4


def lowpass(accel_g, dt, corner_hz, order=LOWPASS_ORDER):
    """The record low-passed by a Butterworth filter of ``order`` with its
    corner at ``corner_hz``, run forward and then backward, so the result
    has no phase shift (GM-264).

    Returns the record unchanged where the corner is at or above the
    record's Nyquist frequency (it carries nothing above the corner) or
    the record is too short for the backward pass's padding.
    """
    from scipy.signal import butter, sosfiltfilt
    accel = np.asarray(accel_g, dtype=float)
    if corner_hz >= 0.5 / dt:
        return accel.copy()
    sos = butter(order, corner_hz, btype="low", fs=1.0 / dt, output="sos")
    if len(accel) <= 3 * (2 * len(sos) + 1):
        return accel.copy()
    return sosfiltfilt(sos, accel)


def clean_numpy(obj):
    """Convert numpy types to plain Python for JSON serialization.

    Handles ndarray → list, numpy scalars → float/int/bool,
    and NaN → None recursively.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_numpy(v) for v in obj]
    return obj
