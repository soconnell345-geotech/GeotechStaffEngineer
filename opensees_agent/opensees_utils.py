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
    except ImportError:
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
    """Compute pseudo-acceleration response spectrum via Newmark-beta SDOF.

    Pure numpy implementation — does not require openseespy.

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
    p_ground = -accel * g  # effective force = -m * ag (unit mass)

    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = np.max(np.abs(accel))
            continue

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

        for j in range(1, len(accel)):
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
        Sa[i] = omega ** 2 * max_abs_u / g

    return periods, Sa


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
