"""
Response spectrum analysis using eqsig (Nigam-Jennings algorithm).

Provides exact piecewise-linear response spectrum computation,
which is more accurate than Newmark-beta for a given time step.
"""

import numpy as np

from seismic_signals_agent.signal_utils import import_eqsig, _G
from seismic_signals_agent.results import ResponseSpectrumResult


def _validate_spectrum_inputs(periods, damping):
    """Validate response spectrum parameters.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if periods is not None:
        periods = np.asarray(periods, dtype=float)
        if len(periods) == 0:
            raise ValueError("periods must be a non-empty array")
        if np.any(periods <= 0):
            raise ValueError("All periods must be positive")
    if not (0 < damping < 1):
        raise ValueError(f"damping must be between 0 and 1 (exclusive), got {damping}")


def analyze_response_spectrum(
    motion=None,
    accel_history=None,
    dt=None,
    periods=None,
    damping=0.05,
) -> ResponseSpectrumResult:
    """Compute response spectrum using Nigam-Jennings algorithm (eqsig).

    Parameters
    ----------
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse').
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    periods : array_like, optional
        Spectral periods (s). Default: logspace(-2, 1, 200).
    damping : float, optional
        Damping ratio (decimal). Default: 0.05.

    Returns
    -------
    ResponseSpectrumResult
        Result container with spectrum data and plot methods.
    """
    _validate_spectrum_inputs(periods, damping)

    # Resolve ground motion (lazy import — no openseespy dependency)
    from opensees_agent.ground_motions import validate_motion_input
    accel_g, dt_val = validate_motion_input(motion, accel_history, dt)

    # Determine motion name
    if motion is not None:
        motion_name = motion
    else:
        motion_name = "custom"

    # Default periods
    if periods is None:
        periods = np.logspace(-2, 1, 200)
    else:
        periods = np.asarray(periods, dtype=float)

    # Sort periods ascending
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]

    # Import eqsig and create AccSignal (convert g -> m/s²)
    eqsig = import_eqsig()
    acc = eqsig.AccSignal(accel_g * _G, dt_val)

    # Compute response spectrum
    acc.generate_response_spectrum(periods, xi=damping)
    Sa_m_s2 = acc.s_a  # spectral acceleration in m/s²
    Sa_g = Sa_m_s2 / _G

    # Time array
    time = np.arange(len(accel_g)) * dt_val

    return ResponseSpectrumResult(
        motion_name=motion_name,
        n_points=len(accel_g),
        duration_s=float(time[-1]),
        dt_s=float(dt_val),
        pga_g=float(np.max(np.abs(accel_g))),
        pgv_m_per_s=float(np.max(np.abs(acc.velocity))),
        pgd_m=float(np.max(np.abs(acc.displacement))),
        damping=damping,
        periods=periods,
        Sa_g=Sa_g,
        time=time,
        accel_g=accel_g,
    )
