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


#: Why the bracketed duration is None without a threshold (GM-325).
NO_THRESHOLD = (
    "not computed: the bracketed duration is the time between the first "
    "and the last exceedance of a threshold acceleration, and no threshold "
    "was given (bracketed_threshold_g, in g). There is no default: no "
    "reference in the library prints one.")


def bracketed_duration(accel_g, dt, threshold_g):
    """The bracketed duration of a record and, when there is none, why.

    The time between the first and the last sample whose absolute
    acceleration exceeds ``threshold_g`` (the definition eqsig's
    ``calc_brac_dur`` uses; computed here so it does not depend on the
    eqsig version).

    Until GM-325 ``analyze_intensity_measures`` called an eqsig function
    that does not exist (``calc_brac``), caught the error and returned
    0.0 for every record.

    Returns
    -------
    (duration_s, note) : (float or None, str or None)
        ``duration_s`` is None when it cannot be computed (no threshold,
        or a record that never exceeds it), and ``note`` then says why;
        otherwise ``note`` is None.
    """
    if threshold_g is None:
        return None, NO_THRESHOLD
    if threshold_g <= 0:
        raise ValueError(
            f"bracketed_threshold_g must be positive (g), got {threshold_g}")
    a = np.abs(np.asarray(accel_g, dtype=float))
    over = np.nonzero(a > threshold_g)[0]
    if len(over) == 0:
        return None, (
            f"not computed: the record never exceeds the threshold of "
            f"{threshold_g:g} g (its peak is {float(a.max()):.4g} g), so "
            "there is no first or last exceedance to measure between.")
    return float((over[-1] - over[0]) * dt), None


def analyze_intensity_measures(
    motion=None,
    accel_history=None,
    dt=None,
    target_pga_g=None,
    sig_dur_start=0.05,
    sig_dur_end=0.95,
    bracketed_threshold_g=None,
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
    target_pga_g : float, optional
        Scale the motion so its PGA equals this value (g) before the
        analysis (built-in motions have fixed PGAs). Default None.
    sig_dur_start : float, optional
        Husid start fraction for significant duration. Default: 0.05.
    sig_dur_end : float, optional
        Husid end fraction for significant duration. Default: 0.95.
    bracketed_threshold_g : float, optional
        Threshold acceleration (g) for the bracketed duration, the time
        between the first and the last sample whose absolute acceleration
        exceeds it. No default (:data:`NO_THRESHOLD`): without it the
        bracketed duration is None and the result says why.

    Returns
    -------
    IntensityMeasuresResult
        Result container with intensity measures and plot methods.
    """
    _validate_intensity_inputs(sig_dur_start, sig_dur_end)

    # Resolve ground motion
    from opensees_agent.ground_motions import (
        validate_motion_input, motion_label)
    accel_g, dt_val = validate_motion_input(
        motion, accel_history, dt, target_pga_g=target_pga_g)

    # Determine motion name
    motion_name = motion_label(motion, target_pga_g)

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

    # Bracketed duration: the time between the first and the last
    # exceedance of a threshold acceleration (GM-325).
    brac_dur, brac_note = bracketed_duration(accel_g, dt_val,
                                             bracketed_threshold_g)

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
        bracketed_threshold_g=(None if bracketed_threshold_g is None
                               else float(bracketed_threshold_g)),
        bracketed_duration_note=brac_note,
        arias_cumulative=np.asarray(arias_cum, dtype=float),
        time=time,
        accel_g=accel_g,
    )
