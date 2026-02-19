"""
Signal processing (filtering + baseline correction) using eqsig.

Applies Butterworth bandpass filtering and/or polynomial baseline correction
to an acceleration time history. Returns processed acceleration, velocity,
and displacement.
"""

import numpy as np

from seismic_signals_agent.signal_utils import import_eqsig, _G
from seismic_signals_agent.results import SignalProcessingResult


def _validate_processing_inputs(bandpass, baseline_order):
    """Validate signal processing parameters.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if bandpass is None and baseline_order is None:
        raise ValueError(
            "At least one of 'bandpass' or 'baseline_order' must be specified")
    if bandpass is not None:
        if not (hasattr(bandpass, '__len__') and len(bandpass) == 2):
            raise ValueError("bandpass must be [f_low, f_high] in Hz")
        f_low, f_high = bandpass
        if f_low <= 0:
            raise ValueError(f"bandpass f_low must be positive, got {f_low}")
        if f_high <= f_low:
            raise ValueError(
                f"bandpass f_high ({f_high}) must be greater than f_low ({f_low})")
    if baseline_order is not None:
        if not isinstance(baseline_order, int) or baseline_order < 0:
            raise ValueError(
                f"baseline_order must be a non-negative integer, got {baseline_order}")


def analyze_signal_processing(
    motion=None,
    accel_history=None,
    dt=None,
    bandpass=None,
    baseline_order=None,
) -> SignalProcessingResult:
    """Process an acceleration time history using eqsig.

    Applies Butterworth bandpass filtering and/or polynomial baseline
    correction. Returns processed acceleration, velocity, and displacement.

    Parameters
    ----------
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse').
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    bandpass : list of float, optional
        Bandpass frequencies [f_low, f_high] in Hz.
    baseline_order : int, optional
        Polynomial order for baseline correction (0, 1, 2, ...).

    Returns
    -------
    SignalProcessingResult
        Result container with original and processed signals.
    """
    _validate_processing_inputs(bandpass, baseline_order)

    # Resolve ground motion
    from opensees_agent.ground_motions import validate_motion_input
    accel_g, dt_val = validate_motion_input(motion, accel_history, dt)

    # Determine motion name
    if motion is not None:
        motion_name = motion
    else:
        motion_name = "custom"

    # Store original PGA
    pga_original = float(np.max(np.abs(accel_g)))

    # Import eqsig and create AccSignal (convert g -> m/s²)
    eqsig = import_eqsig()
    acc = eqsig.AccSignal(accel_g * _G, dt_val)

    # Apply bandpass filter first (standard seismological practice)
    if bandpass is not None:
        acc.butter_pass(list(bandpass))

    # Then apply baseline correction
    if baseline_order is not None:
        acc.remove_poly(poly_fit=baseline_order)

    # Extract processed results (convert m/s² back to g for acceleration)
    processed_accel_g = acc.values / _G
    velocity = acc.velocity      # m/s (native eqsig units)
    displacement = acc.displacement  # m (native eqsig units)

    # Time array
    time = np.arange(len(accel_g)) * dt_val

    return SignalProcessingResult(
        motion_name=motion_name,
        n_points=len(accel_g),
        dt_s=float(dt_val),
        bandpass_hz=list(bandpass) if bandpass is not None else [],
        baseline_order=baseline_order if baseline_order is not None else -1,
        pga_original_g=pga_original,
        pga_processed_g=float(np.max(np.abs(processed_accel_g))),
        pgv_processed_m_per_s=float(np.max(np.abs(velocity))),
        pgd_processed_m=float(np.max(np.abs(displacement))),
        time=time,
        accel_original_g=accel_g.copy(),
        accel_processed_g=processed_accel_g,
        velocity_m_per_s=np.asarray(velocity, dtype=float),
        displacement_m=np.asarray(displacement, dtype=float),
    )
