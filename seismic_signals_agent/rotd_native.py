"""
Rotated pseudo-spectral acceleration (RotDnn) with numpy only.

The fallback for ``analyze_rotd_spectrum`` when pyrotd cannot be imported
(pyrotd 0.6.x imports ``pkg_resources``, gone from setuptools 81+; live
smoke G10, 2026-10-08). It follows the same published procedure as pyrotd
0.6 (A. Kottke, MIT), so the two agree to round-off
(``tests/test_seismic_signals.py::TestRotDNative`` checks it against pyrotd
where pyrotd can be loaded):

1. Each component's single-degree-of-freedom pseudo-acceleration response
   is computed in the frequency domain, with the transfer function
   ``H(w) = -w0^2 / (w^2 - w0^2 - 2i*zeta*w0*w)`` applied to the Fourier
   spectrum of the record. When the oscillator frequency is high relative
   to the record's Nyquist frequency the spectrum is zero-padded so the
   response is resolved with at least ``max_freq_ratio`` points per
   oscillator cycle (the amplitude is rescaled for the longer inverse FFT).
2. The two responses are combined at every rotation angle
   ``theta = 0, 1, ..., 179 deg``: ``r(t) = a(t) cos(theta) + b(t) sin(theta)``,
   and the peak |r| is taken at each angle.
3. RotDnn is the nn-th percentile (linear interpolation) of those peaks
   over the angles: RotD0 the minimum, RotD50 the median, RotD100 the
   maximum (Boore 2010).

Only the time steps where the vector sum sqrt(a^2 + b^2) reaches 0.7 x the
smaller of the two component peaks are rotated: no rotation can peak
anywhere else (Stewart et al. 2017, PEER 2017/09, Eqs 2.4-2.5).

References
----------
Boore, D.M. (2010). Orientation-independent, nongeometric-mean measures of
seismic intensity from two horizontal components of motion. BSSA 100(4),
1830-1835.
Stewart, J.P. et al. (2017). PEER Report 2017/09, Eqs 2.4-2.5 (the reduced
rotation set; as cited by pyrotd 0.6).
"""

import numpy as np


def _psa_response(freqs, fourier_amp, damping, osc_freq, max_freq_ratio):
    """Pseudo-acceleration time history of one SDOF oscillator (g)."""
    w = 2.0 * np.pi * freqs
    w0 = 2.0 * np.pi * osc_freq
    h = -(w0 ** 2) / (w ** 2 - w0 ** 2 - 2.0j * damping * w0 * w)
    n = len(fourier_amp)
    m = max(n, int(max_freq_ratio * osc_freq / freqs[1]))
    scale = float(m) / float(n)
    return scale * np.fft.irfft(fourier_amp * h, 2 * (m - 1))


def rotated_spectral_accels(dt, accel_a, accel_b, periods, damping=0.05,
                            percentiles=(0, 50, 100), angles=None,
                            max_freq_ratio=5.0):
    """RotDnn pseudo-spectral accelerations of two orthogonal components.

    Parameters
    ----------
    dt : float
        Time step (s), shared by both records.
    accel_a, accel_b : array_like
        Acceleration histories (g) of equal length.
    periods : array_like
        Oscillator periods (s), > 0.
    damping : float
        Oscillator damping ratio (decimal).
    percentiles : sequence of float
        Percentiles over the rotation angles (0 = RotD0, 50 = RotD50,
        100 = RotD100).
    angles : array_like, optional
        Rotation angles (deg). Default 0, 1, ..., 179.
    max_freq_ratio : float
        Minimum oscillator-to-record frequency resolution ratio (pyrotd's
        default, 5).

    Returns
    -------
    dict
        {percentile: numpy.ndarray of Sa (g), one per period, in the order
        of ``periods``}.
    """
    accel_a = np.asarray(accel_a, dtype=float)
    accel_b = np.asarray(accel_b, dtype=float)
    if accel_a.shape != accel_b.shape:
        raise ValueError("Both components must have the same length")
    periods = np.asarray(periods, dtype=float)
    pct = np.asarray(list(percentiles), dtype=float)
    ang = np.radians(np.arange(0, 180, 1) if angles is None
                     else np.asarray(angles, dtype=float))
    coeffs = np.c_[np.cos(ang), np.sin(ang)]

    fa = np.fft.rfft(accel_a)
    fb = np.fft.rfft(accel_b)
    freqs = np.linspace(0.0, 1.0 / (2.0 * dt), num=fa.size)

    out = np.zeros((len(pct), len(periods)))
    for j, period in enumerate(periods):
        f0 = 1.0 / period
        ra = _psa_response(freqs, fa, damping, f0, max_freq_ratio)
        rb = _psa_response(freqs, fb, damping, f0, max_freq_ratio)
        pair = np.vstack([ra, rb])
        level = 0.7 * min(np.abs(ra).max(), np.abs(rb).max())
        keep = np.sqrt(ra ** 2 + rb ** 2) >= level
        peaks = np.abs(coeffs @ pair[:, keep]).max(axis=1)
        out[:, j] = np.percentile(peaks, pct, method="linear")
    return {float(p): out[i] for i, p in enumerate(pct)}
