"""
Response spectrum of one ground motion record, every period by Nigam and
Jennings (1969) with sub-steps.

The record is taken as linear between its samples and each oscillator's
response to it is solved in closed form, in sub-steps of at most a
twentieth of the period below 20 record steps a period, the peak read
between points (``geotech_common.response_spectrum``, the OpenSees site
response's code too, so the two tools give the same Sa; GM-274). PGV and
PGD come from eqsig's integration of the record.

Until GM-274 the spectrum was eqsig's: the same step, but the whole record
at one step no finer than a quarter of the record's, the peak read only at
its points and the PGA below six of them, so up to 2.5 % low below ten
record steps a period.
"""

import numpy as np

from geotech_common.response_spectrum import pseudo_spectral_accel
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
    target_pga_g=None,
    periods=None,
    damping=0.05,
) -> ResponseSpectrumResult:
    """Compute the response spectrum, every period by Nigam and Jennings
    (1969) with sub-steps (GM-274); PGV and PGD by eqsig.

    Parameters
    ----------
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse').
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    target_pga_g : float, optional
        Scale the motion so its PGA equals this value (g) before the
        analysis (built-in motions have fixed PGAs). Default None.
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
    from opensees_agent.ground_motions import (
        validate_motion_input, motion_label)
    accel_g, dt_val = validate_motion_input(
        motion, accel_history, dt, target_pga_g=target_pga_g)

    # Determine motion name
    motion_name = motion_label(motion, target_pga_g)

    # Default periods
    if periods is None:
        periods = np.logspace(-2, 1, 200)
    else:
        periods = np.asarray(periods, dtype=float)

    # Sort periods ascending
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]

    # Import eqsig and create AccSignal (convert g -> m/s²): PGV and PGD
    eqsig = import_eqsig()
    acc = eqsig.AccSignal(accel_g * _G, dt_val)

    # The spectrum, the OpenSees site response's code (GM-274)
    Sa_g = pseudo_spectral_accel(accel_g, dt_val, periods, damping)

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
