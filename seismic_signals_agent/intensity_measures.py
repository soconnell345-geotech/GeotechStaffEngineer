"""
Earthquake intensity measure calculations using eqsig.

Computes Arias intensity, significant duration, CAV, bracketed duration,
PGA, PGV, and PGD from an acceleration time history.
"""

import numpy as np

from seismic_signals_agent.signal_utils import import_eqsig, _G
from seismic_signals_agent.results import IntensityMeasuresResult


def _validate_intensity_inputs(sig_dur_start, sig_dur_end):
    """Validate intensity measure parameters.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not (0 < sig_dur_start < 1):
        raise ValueError(
            f"sig_dur_start must be between 0 and 1 (exclusive), got {sig_dur_start}")
    if not (0 < sig_dur_end < 1):
        raise ValueError(
            f"sig_dur_end must be between 0 and 1 (exclusive), got {sig_dur_end}")
    if sig_dur_start >= sig_dur_end:
        raise ValueError(
            f"sig_dur_start ({sig_dur_start}) must be less than "
            f"sig_dur_end ({sig_dur_end})")


def analyze_intensity_measures(
    motion=None,
    accel_history=None,
    dt=None,
    sig_dur_start=0.05,
    sig_dur_end=0.95,
) -> IntensityMeasuresResult:
    """Compute earthquake intensity measures using eqsig.

    Parameters
    ----------
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse').
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    sig_dur_start : float, optional
        Husid start fraction for significant duration. Default: 0.05.
    sig_dur_end : float, optional
        Husid end fraction for significant duration. Default: 0.95.

    Returns
    -------
    IntensityMeasuresResult
        Result container with intensity measures and plot methods.
    """
    _validate_intensity_inputs(sig_dur_start, sig_dur_end)

    # Resolve ground motion
    from opensees_agent.ground_motions import validate_motion_input
    accel_g, dt_val = validate_motion_input(motion, accel_history, dt)

    # Determine motion name
    if motion is not None:
        motion_name = motion
    else:
        motion_name = "custom"

    # Import eqsig and create AccSignal (convert g -> m/s²)
    eqsig = import_eqsig()
    import eqsig.im as eqsig_im
    acc = eqsig.AccSignal(accel_g * _G, dt_val)

    # Compute intensity measures
    arias_cum = eqsig_im.calc_arias_intensity(acc)
    arias_total = float(arias_cum[-1])

    sig_dur = float(eqsig_im.calc_sig_dur(acc, start=sig_dur_start, end=sig_dur_end))
    cav_result = eqsig_im.calc_cav(acc)
    # calc_cav returns cumulative array; take final value for total CAV
    cav = float(cav_result[-1]) if hasattr(cav_result, '__len__') else float(cav_result)

    # Bracketed duration — may not be available in all eqsig versions
    try:
        brac_dur = float(eqsig_im.calc_brac(acc))
    except (AttributeError, TypeError):
        brac_dur = 0.0

    # Time array
    time = np.arange(len(accel_g)) * dt_val

    return IntensityMeasuresResult(
        motion_name=motion_name,
        n_points=len(accel_g),
        duration_s=float(time[-1]),
        dt_s=float(dt_val),
        pga_g=float(np.max(np.abs(accel_g))),
        pgv_m_per_s=float(np.max(np.abs(acc.velocity))),
        pgd_m=float(np.max(np.abs(acc.displacement))),
        arias_intensity_m_per_s=arias_total,
        significant_duration_s=sig_dur,
        sig_dur_start=sig_dur_start,
        sig_dur_end=sig_dur_end,
        cav_m_per_s=cav,
        bracketed_duration_s=brac_dur,
        arias_cumulative=np.asarray(arias_cum, dtype=float),
        time=time,
        accel_g=accel_g,
    )
