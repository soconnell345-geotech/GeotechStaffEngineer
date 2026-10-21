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


#: A spectral period shorter than this many record steps is integrated on
#: the record linearly interpolated to a step of at most T / 20
#: (_SUBSTEPS_PER_PERIOD) (GM-264): Newmark's average acceleration method
#: in substeps over the piecewise-linear record, the excitation Nigam and
#: Jennings (1969) integrate exactly between samples. At the record's
#: step, average acceleration lengthens the period by about
#: (pi^2 / 12) (dt / T)^2 and the largest response falls between steps:
#: at one step a period (0.01 s on a 0.01 s record) Sa(0.01 s) came out
#: 0.40 g on a surface record peaking at 0.45 g, and 0.61 g on one peaking
#: at 0.43 g. At T / 20 it is within 3 % of the exact response to the
#: piecewise-linear record (scipy lsim, first-order hold, read at T / 400)
#: on the site response records tried, and within 0.3 % at 0.01 s.
#: Periods of ten steps and longer keep the record's step, as before;
#: there it is within about 5 % (one 0.05 s ordinate on a 0.005 s record).
_MIN_STEPS_PER_PERIOD = 10
_SUBSTEPS_PER_PERIOD = 20


def compute_response_spectrum(accel_g, dt, periods=None, damping=0.05):
    """Compute pseudo-acceleration response spectrum via Newmark-beta SDOF.

    Pure numpy implementation — does not require openseespy. A period
    shorter than 10 record steps is integrated in substeps of at most a
    twentieth of the period on the record linearly interpolated between
    its samples (the piecewise-linear excitation of Nigam and Jennings
    1969; GM-264); longer periods use the record's own step.

    Parameters
    ----------
    accel_g : array_like
        Ground acceleration time history in g.
    dt : float
        Time step (s).
    periods : array_like, optional
        Spectral periods (s).  Default: 0.01 to 10 s (100 points, log-spaced).
    damping : float
        Damping ratio.  Default 0.05 (5%).

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

    Sa = np.zeros(len(periods))
    p_record = -accel * g  # effective force = -m * ag (unit mass)
    finer = {1: (p_record, dt)}

    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = np.max(np.abs(accel))
            continue

        # Substeps for a period shorter than _MIN_STEPS_PER_PERIOD steps,
        # on the record interpolated linearly between its samples
        n_sub = 1
        if T < _MIN_STEPS_PER_PERIOD * dt * (1.0 - 1e-9):
            n_sub = math.ceil(_SUBSTEPS_PER_PERIOD * dt / T - 1e-9)
        if n_sub not in finer:
            t_rec = np.arange(len(p_record)) * dt
            t_sub = np.arange((len(p_record) - 1) * n_sub + 1) * (dt / n_sub)
            finer[n_sub] = (np.interp(t_sub, t_rec, p_record), dt / n_sub)
        p_ground, h = finer[n_sub]
        Sa[i] = _newmark_peak_sa(p_ground, h, T, damping, g)

    return periods, Sa


def _newmark_peak_sa(p_ground, dt, T, damping, g):
    """Pseudo-spectral acceleration (g) of a unit-mass oscillator of period
    ``T`` under the effective force history ``p_ground`` (step ``dt``),
    by Newmark average acceleration."""
    omega = 2.0 * math.pi / T
    k = omega ** 2  # unit mass m=1
    c = 2.0 * damping * omega

    # Newmark average acceleration (gamma=0.5, beta=0.25)
    # Unconditionally stable for linear systems
    gamma_n, beta_n = 0.5, 0.25
    dt2 = dt * dt

    # Precompute constants
    a0 = 1.0 / (beta_n * dt2)
    a1 = gamma_n / (beta_n * dt)
    a2 = 1.0 / (beta_n * dt)
    a3 = 1.0 / (2.0 * beta_n) - 1.0
    a4 = gamma_n / beta_n - 1.0
    a5 = dt * (gamma_n / (2.0 * beta_n) - 1.0)

    k_eff = k + a0 + a1 * c

    u, v, a_resp = 0.0, 0.0, 0.0
    # Initial acceleration from equilibrium: m*a + c*v + k*u = p(0)
    a_resp = p_ground[0]  # m=1, u=v=0 at t=0

    max_abs_u = 0.0

    for j in range(1, len(p_ground)):
        # Effective force at next step
        dp_eff = (p_ground[j]
                  + (a0 * u + a2 * v + a3 * a_resp)       # mass terms
                  + c * (a1 * u + a4 * v + a5 * a_resp))  # damping terms

        u_new = dp_eff / k_eff
        a_new = a0 * (u_new - u) - a2 * v - a3 * a_resp
        v_new = v + dt * ((1.0 - gamma_n) * a_resp + gamma_n * a_new)

        u = u_new
        v = v_new
        a_resp = a_new

        abs_u = abs(u)
        if abs_u > max_abs_u:
            max_abs_u = abs_u

    # Pseudo-spectral acceleration: Sa = omega^2 * Sd
    return omega ** 2 * max_abs_u / g


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
