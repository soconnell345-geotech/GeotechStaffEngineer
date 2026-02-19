"""
Rotated spectral acceleration (RotD50/RotD100) using pyrotd.

Computes orientation-independent spectral acceleration from two orthogonal
horizontal components of ground motion, per Boore (2010).
"""

import numpy as np

from seismic_signals_agent.signal_utils import import_pyrotd
from seismic_signals_agent.results import RotDSpectrumResult


def _validate_rotd_inputs(accel_a, accel_b, periods, damping, percentiles):
    """Validate RotD spectrum parameters.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if accel_a is None:
        raise ValueError("Component A acceleration is required")
    if accel_b is None:
        raise ValueError("Component B acceleration is required")
    if periods is not None:
        periods = np.asarray(periods, dtype=float)
        if len(periods) == 0:
            raise ValueError("periods must be a non-empty array")
        if np.any(periods <= 0):
            raise ValueError("All periods must be positive")
    if not (0 < damping < 1):
        raise ValueError(f"damping must be between 0 and 1 (exclusive), got {damping}")
    if percentiles is not None:
        for p in percentiles:
            if not (0 <= p <= 100):
                raise ValueError(f"Each percentile must be between 0 and 100, got {p}")


def analyze_rotd_spectrum(
    motion_a=None,
    accel_history_a=None,
    motion_b=None,
    accel_history_b=None,
    dt=None,
    periods=None,
    damping=0.05,
    percentiles=None,
) -> RotDSpectrumResult:
    """Compute rotated spectral acceleration (RotD) using pyrotd.

    Two orthogonal horizontal components are required. Each can be specified
    as a built-in motion name or a custom acceleration array. Both components
    must share the same time step.

    Parameters
    ----------
    motion_a : str, optional
        Built-in motion name for component A.
    accel_history_a : array_like, optional
        Custom acceleration for component A (g).
    motion_b : str, optional
        Built-in motion name for component B.
    accel_history_b : array_like, optional
        Custom acceleration for component B (g).
    dt : float, optional
        Time step (s). Required for custom motions; ignored for built-in.
    periods : array_like, optional
        Spectral periods (s). Default: logspace(-2, 1, 200).
    damping : float, optional
        Damping ratio (decimal). Default: 0.05.
    percentiles : list of int, optional
        Percentiles to compute. Default: [0, 50, 100].

    Returns
    -------
    RotDSpectrumResult
        Result container with RotD spectra and plot methods.
    """
    # Resolve ground motions
    from opensees_agent.ground_motions import validate_motion_input

    # Component A
    accel_a_g, dt_a = validate_motion_input(motion_a, accel_history_a, dt)
    name_a = motion_a if motion_a is not None else "custom_a"

    # Component B
    accel_b_g, dt_b = validate_motion_input(motion_b, accel_history_b, dt)
    name_b = motion_b if motion_b is not None else "custom_b"

    # Validate after resolving motions
    _validate_rotd_inputs(accel_a_g, accel_b_g, periods, damping, percentiles)

    # Ensure same time step
    if abs(dt_a - dt_b) > 1e-10:
        raise ValueError(
            f"Both components must have the same time step. "
            f"Component A: dt={dt_a}, Component B: dt={dt_b}")
    dt_val = dt_a

    # Pad shorter record with zeros to match length
    len_a = len(accel_a_g)
    len_b = len(accel_b_g)
    if len_a > len_b:
        accel_b_g = np.pad(accel_b_g, (0, len_a - len_b))
    elif len_b > len_a:
        accel_a_g = np.pad(accel_a_g, (0, len_b - len_a))
    n_points = len(accel_a_g)

    # Default periods and percentiles
    if periods is None:
        periods = np.logspace(-2, 1, 200)
    else:
        periods = np.asarray(periods, dtype=float)
    if percentiles is None:
        percentiles = [0, 50, 100]

    # Sort periods ascending
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]

    # Convert periods to frequencies for pyrotd
    osc_freqs = 1.0 / periods

    # Import pyrotd and compute
    pyrotd = import_pyrotd()
    result = pyrotd.calc_rotated_spec_accels(
        dt_val, accel_a_g, accel_b_g,
        osc_freqs, osc_damping=damping,
        percentiles=percentiles,
    )

    # Extract RotD arrays from recarray
    # pyrotd returns a flat recarray with fields: osc_freq, percentile, spec_accel, angle
    # One record per (osc_freq, percentile) combination. Filter by percentile.
    rotd0 = np.array([])
    rotd50 = np.array([])
    rotd100 = np.array([])

    for pctl in percentiles:
        mask = result.percentile == pctl
        # Sort by osc_freq descending (= period ascending) to match our period array
        sa_vals = np.asarray(result[mask].spec_accel, dtype=float)
        if pctl == 0:
            rotd0 = sa_vals
        elif pctl == 50:
            rotd50 = sa_vals
        elif pctl == 100:
            rotd100 = sa_vals

    return RotDSpectrumResult(
        motion_a_name=name_a,
        motion_b_name=name_b,
        n_points=n_points,
        dt_s=float(dt_val),
        pga_a_g=float(np.max(np.abs(accel_a_g))),
        pga_b_g=float(np.max(np.abs(accel_b_g))),
        damping=damping,
        percentiles=percentiles,
        periods=periods,
        rotd0=rotd0,
        rotd50=rotd50,
        rotd100=rotd100,
    )
