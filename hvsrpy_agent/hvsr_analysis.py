"""
HVSR analysis wrapper using hvsrpy.

Computes Horizontal-to-Vertical Spectral Ratio from 3-component
seismograms to identify site resonant frequency and amplification.
"""

import numpy as np

from hvsrpy_agent.hvsrpy_utils import import_hvsrpy
from hvsrpy_agent.results import HvsrResult


def _validate_hvsr_inputs(ns, ew, vt, dt, window_length_s, smoothing_bandwidth,
                          distribution, horizontal_method):
    """Validate HVSR analysis inputs."""
    if len(ns) == 0 or len(ew) == 0 or len(vt) == 0:
        raise ValueError("All three components (ns, ew, vt) must be non-empty")
    if len(ns) != len(ew) or len(ns) != len(vt):
        raise ValueError(
            f"All components must have the same length: "
            f"ns={len(ns)}, ew={len(ew)}, vt={len(vt)}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if window_length_s <= 0:
        raise ValueError(f"window_length_s must be positive, got {window_length_s}")

    total_duration = len(ns) * dt
    if window_length_s > total_duration:
        raise ValueError(
            f"window_length_s ({window_length_s}s) exceeds total duration "
            f"({total_duration:.1f}s). Need at least 1 window."
        )

    if smoothing_bandwidth <= 0:
        raise ValueError(f"smoothing_bandwidth must be positive, got {smoothing_bandwidth}")
    if distribution not in ("lognormal", "normal"):
        raise ValueError(f"distribution must be 'lognormal' or 'normal', got '{distribution}'")

    valid_methods = {
        "geometric_mean", "arithmetic_mean", "quadratic_mean",
        "squared_average", "total_horizontal_energy", "vector_summation",
        "maximum_horizontal_value",
    }
    if horizontal_method not in valid_methods:
        raise ValueError(
            f"horizontal_method must be one of {sorted(valid_methods)}, "
            f"got '{horizontal_method}'"
        )


def analyze_hvsr(
    ns,
    ew,
    vt,
    dt,
    window_length_s=60.0,
    filter_hz=None,
    smoothing_operator="konno_and_ohmachi",
    smoothing_bandwidth=40,
    freq_min=0.2,
    freq_max=50.0,
    n_freq=200,
    horizontal_method="geometric_mean",
    distribution="lognormal",
    rejection_n_std=2.0,
    rejection_max_iterations=50,
    degrees_from_north=0.0,
) -> HvsrResult:
    """Compute HVSR from 3-component seismogram arrays.

    Parameters
    ----------
    ns : array-like
        North-south component amplitudes (any unit, e.g. m/s, counts).
    ew : array-like
        East-west component amplitudes (same unit as ns).
    vt : array-like
        Vertical component amplitudes (same unit as ns).
    dt : float
        Sampling interval (seconds).
    window_length_s : float
        Time window length for splitting (seconds). Default 60.
    filter_hz : list of 2 floats or None
        Butterworth bandpass corners [f_low, f_high] in Hz.
        Use None for no filter. Default None.
    smoothing_operator : str
        Smoothing method. Default 'konno_and_ohmachi'.
    smoothing_bandwidth : float
        Smoothing bandwidth parameter. Default 40 (for Konno-Ohmachi).
    freq_min : float
        Minimum frequency for output (Hz). Default 0.2.
    freq_max : float
        Maximum frequency for output (Hz). Default 50.0.
    n_freq : int
        Number of frequency points (log-spaced). Default 200.
    horizontal_method : str
        Method to combine horizontal components. Default 'geometric_mean'.
    distribution : str
        Statistical distribution: 'lognormal' or 'normal'. Default 'lognormal'.
    rejection_n_std : float
        Number of standard deviations for frequency-domain window rejection.
        Set to 0 to skip rejection. Default 2.0.
    rejection_max_iterations : int
        Maximum iterations for rejection convergence. Default 50.
    degrees_from_north : float
        Sensor orientation (degrees clockwise from north). Default 0.

    Returns
    -------
    HvsrResult
        HVSR results including f0, A0, T0, SESAME criteria, and curves.
    """
    ns_arr = np.asarray(ns, dtype=float)
    ew_arr = np.asarray(ew, dtype=float)
    vt_arr = np.asarray(vt, dtype=float)

    _validate_hvsr_inputs(
        ns_arr, ew_arr, vt_arr, dt, window_length_s, smoothing_bandwidth,
        distribution, horizontal_method
    )

    hvsrpy = import_hvsrpy()
    from hvsrpy import TimeSeries, SeismicRecording3C, settings

    # Build recording from arrays
    ts_ns = TimeSeries(ns_arr, dt)
    ts_ew = TimeSeries(ew_arr, dt)
    ts_vt = TimeSeries(vt_arr, dt)
    rec = SeismicRecording3C(
        ns=ts_ns, ew=ts_ew, vt=ts_vt,
        degrees_from_north=degrees_from_north,
    )

    # Preprocessing settings
    pre = settings.HvsrPreProcessingSettings()
    pre.window_length_in_seconds = window_length_s
    pre.detrend = "linear"
    if filter_hz is not None:
        pre.filter_corner_frequencies_in_hz = list(filter_hz)

    # Processing settings
    proc = settings.HvsrTraditionalProcessingSettings()
    proc.smoothing = dict(
        operator=smoothing_operator,
        bandwidth=smoothing_bandwidth,
        center_frequencies_in_hz=np.geomspace(freq_min, freq_max, n_freq),
    )
    proc.method_to_combine_horizontals = horizontal_method

    # Run pipeline
    records = hvsrpy.preprocess([rec], pre)
    hvsr = hvsrpy.process(records, proc)

    # Optional frequency-domain window rejection (need >3 windows)
    if rejection_n_std > 0 and hvsr.n_curves > 3:
        try:
            hvsrpy.frequency_domain_window_rejection(
                hvsr, n=rejection_n_std,
                max_iterations=rejection_max_iterations,
                distribution_fn=distribution,
                distribution_mc=distribution,
            )
        except Exception:
            pass  # rejection may fail with too few windows

    # Extract results
    n_valid = int(sum(hvsr.valid_window_boolean_mask))

    # mean_fn_frequency can return NaN if no valid peaks
    f0_raw = hvsr.mean_fn_frequency(distribution=distribution)
    f0 = float(f0_raw) if not np.isnan(f0_raw) else 0.0
    A0_raw = hvsr.mean_fn_amplitude(distribution=distribution)
    A0 = float(A0_raw) if not np.isnan(A0_raw) else 0.0
    T0 = 1.0 / f0 if f0 > 0 else 0.0

    mean_curve = hvsr.mean_curve(distribution=distribution)
    # Replace any NaN in mean curve with 0
    if mean_curve is not None:
        mean_curve = np.where(np.isnan(mean_curve), 0.0, mean_curve)

    # std_curve requires >1 valid window
    if n_valid > 1:
        try:
            f0_std = float(hvsr.std_fn_frequency(distribution=distribution))
            A0_std = float(hvsr.std_fn_amplitude(distribution=distribution))
            std_curve_arr = hvsr.std_curve(distribution=distribution)
            upper = hvsr.nth_std_curve(n=1, distribution=distribution)
            lower = hvsr.nth_std_curve(n=-1, distribution=distribution)
            # Clean NaN
            if np.any(np.isnan(f0_std)):
                f0_std = 0.0
            if np.any(np.isnan(A0_std)):
                A0_std = 0.0
            std_curve_arr = np.where(np.isnan(std_curve_arr), 0.0, std_curve_arr)
            upper = np.where(np.isnan(upper), mean_curve, upper)
            lower = np.where(np.isnan(lower), mean_curve, lower)
        except (ValueError, RuntimeError):
            f0_std = 0.0
            A0_std = 0.0
            std_curve_arr = np.zeros_like(mean_curve)
            upper = mean_curve.copy()
            lower = mean_curve.copy()
    else:
        f0_std = 0.0
        A0_std = 0.0
        std_curve_arr = np.zeros_like(mean_curve)
        upper = mean_curve.copy()
        lower = mean_curve.copy()

    # SESAME criteria
    from hvsrpy.sesame import reliability, clarity
    try:
        rel = reliability(
            window_length_s, n_valid,
            hvsr.frequency, mean_curve, std_curve_arr, verbose=0,
        )
        cla = clarity(
            hvsr.frequency, mean_curve, std_curve_arr, f0_std, verbose=0,
        )
    except Exception:
        rel = [0.0, 0.0, 0.0]
        cla = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return HvsrResult(
        f0_hz=f0,
        A0=A0,
        T0_s=T0,
        f0_std_hz=f0_std,
        A0_std=A0_std,
        n_windows=hvsr.n_curves,
        n_valid_windows=n_valid,
        window_length_s=window_length_s,
        distribution=distribution,
        smoothing_operator=smoothing_operator,
        horizontal_method=horizontal_method,
        sesame_reliability=[float(x) for x in rel],
        sesame_clarity=[float(x) for x in cla],
        frequency=hvsr.frequency,
        mean_curve=mean_curve,
        std_curve=std_curve_arr,
        upper_curve=upper,
        lower_curve=lower,
    )
