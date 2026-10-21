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


#: Every spectral period is integrated by Nigam and Jennings (1969)
#: (GM-272): the record is taken as linear between its samples and the
#: oscillator's response to it is solved in closed form, step by step. A
#: period shorter than this many record steps is integrated on the same
#: straight lines split into equal substeps of at most T / 20, and between
#: points the largest response is read off the cubic through the
#: displacement and velocity at both ends of each step (Hermite). Against
#: scipy's lsim (the same piecewise-linear record through the matrix
#: exponential, read every T / 400 or dt / 50, whichever is finer) on
#: three site response records and their inputs, 27 periods from 0.01 to
#: 4 s: within 0.004 %. Before, Newmark's average acceleration at the
#: record's step from ten steps a period up (in substeps below ten, GM-264)
#: was within 11 % (-10.7 % at 0.066 s on a 0.005 s record, +10.4 % at
#: 0.125 s on a 0.01 s one), and read the peak only at the points.
_POINTS_PER_PERIOD = 20


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
    accel = np.asarray(accel_g, dtype=float)
    g = 9.81  # m/s^2

    if periods is None:
        periods = np.logspace(-2, 1, 100)
    periods = np.asarray(periods, dtype=float)
    if not 0.0 <= damping < 1.0:
        raise ValueError(f"damping is a ratio of critical, at least 0 and "
                         f"below 1; got {damping}")

    Sa = np.zeros(len(periods))
    p_record = -accel * g  # effective force = -m * ag (unit mass)
    finer = {1: p_record}

    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = np.max(np.abs(accel))
            continue

        # Substeps for a period shorter than _POINTS_PER_PERIOD steps, on
        # the same straight lines between the record's samples
        n_sub = max(1, math.ceil(_POINTS_PER_PERIOD * dt / T - 1e-9))
        if n_sub not in finer:
            finer[n_sub] = _substeps(p_record, n_sub)
        h = dt / n_sub
        omega = 2.0 * math.pi / T
        u, v = _nigam_jennings(finer[n_sub], h, omega, damping)
        # Pseudo-spectral acceleration: Sa = omega^2 * Sd
        Sa[i] = omega ** 2 * _peak_between_points(u, v, h) / g

    return periods, Sa


def _substeps(p, n):
    """``p`` with ``n - 1`` points put evenly on the straight line between
    each pair of its samples."""
    f = np.arange(n) / n
    return np.append((p[:-1, None] + np.diff(p)[:, None] * f).ravel(), p[-1:])


def _nigam_jennings(p, h, omega, damping):
    """Displacement and velocity at every point of a unit-mass oscillator of
    circular frequency ``omega`` and damping ratio ``damping``, at rest at
    the first point, under the force ``p`` taken as linear between points
    ``h`` apart: the closed-form step of Nigam and Jennings (1969),

        u[j+1] = A u[j] + B v[j] + C p[j] + D p[j+1]
        v[j+1] = A1 u[j] + B1 v[j] + C1 p[j] + D1 p[j+1],

    run as the two-step recursion it implies for each (scipy ``lfilter``)."""
    from scipy.signal import lfilter, lfiltic
    k = omega * omega
    r = math.sqrt(1.0 - damping * damping)
    wd = omega * r  # damped circular frequency
    e = math.exp(-damping * omega * h)
    s, c = math.sin(wd * h), math.cos(wd * h)
    zr = damping / r
    zh = 2.0 * damping / (omega * h)
    A = e * (zr * s + c)
    B = e * s / wd
    C = (zh + e * (((1.0 - 2.0 * damping ** 2) / (wd * h) - zr) * s
                   - (1.0 + zh) * c)) / k
    D = (1.0 - zh + e * ((2.0 * damping ** 2 - 1.0) / (wd * h) * s
                         + zh * c)) / k
    A1 = -e * omega / r * s
    B1 = e * (c - zr * s)
    C1 = (-1.0 / h + e * ((omega / r + zr / h) * s + c / h)) / k
    D1 = (1.0 - e * (zr * s + c)) / (k * h)

    u = np.zeros(len(p))
    v = np.zeros(len(p))
    if len(p) < 2:
        return u, v
    u[1] = C * p[0] + D * p[1]
    v[1] = C1 * p[0] + D1 * p[1]
    if len(p) > 2:
        a = [1.0, -(A + B1), A * B1 - B * A1]
        for x, b in ((u, [D, C - B1 * D + B * D1, B * C1 - B1 * C]),
                     (v, [D1, C1 - A * D1 + A1 * D, A1 * C - A * C1])):
            zi = lfiltic(b, a, [x[1], 0.0], [p[1], p[0]])
            x[2:] = lfilter(b, a, p[2:], zi=zi)[0]
    return u, v


def _peak_between_points(u, v, h):
    """The largest |u| over the record, between its points too: on each
    step of ``h`` the cubic through u and v at both ends (Hermite), at its
    turning points inside the step."""
    if len(u) == 0:
        return 0.0
    peak = float(np.max(np.abs(u)))
    u0, u1, m0, m1 = u[:-1], u[1:], h * v[:-1], h * v[1:]
    # the cubic u0 + m0 s + c2 s^2 + c3 s^3 over the step (s from 0 to 1)
    c2 = 3.0 * (u1 - u0) - 2.0 * m0 - m1
    c3 = 2.0 * (u0 - u1) + m0 + m1
    # its turning points, roots of 3 c3 s^2 + 2 c2 s + m0
    disc = c2 * c2 - 3.0 * c3 * m0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -(c2 + np.copysign(np.sqrt(disc), c2))
        for s in (m0 / q, q / (3.0 * c3)):
            inside = (s > 0.0) & (s < 1.0)  # false where nan or inf
            if np.any(inside):
                s = s[inside]
                cubic = u0[inside] + s * (m0[inside] + s * (
                    c2[inside] + s * c3[inside]))
                peak = max(peak, float(np.max(np.abs(cubic))))
    return peak


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
