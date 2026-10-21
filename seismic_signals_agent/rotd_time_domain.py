"""
Rotated pseudo-spectral accelerations (RotDnn) from time-domain oscillator
responses (GM-275).

For each period, each horizontal component's oscillator displacement
history comes from the shared Nigam and Jennings (1969) code
(``geotech_common.response_spectrum.oscillator_response``): the record
linear between its samples, in sub-steps of at most a twentieth of the
period below 20 record steps a period, over the record's length. Then, as
Boore (2010) defines RotD:

1. The two histories are combined at every rotation angle,
   ``r(t) = u_a(t) cos(theta) + u_b(t) sin(theta)``,
   ``theta = 0, 1, ..., 179 deg``.
2. The peak |r| is taken at each angle, between points too: off the cubic
   through r and its velocity at both ends of each step, as the response
   spectrum reads its peak.
3. RotDnn is the nn-th percentile of those peaks over the angles (linear
   interpolation): RotD0 the least, RotD50 the median, RotD100 the
   largest. Sa = (2 pi / T)^2 * peak / g.

So RotD100 of a component paired with zero is that component's spectrum
from ``analyze_response_spectrum``.

Only the steps that can hold an angle's peak are rotated, and none that
can is left out: on a step no projection of w = (u_a, u_b) exceeds
max(|w0|, |w1|) + 4/27 (|h w'0| + |h w'1|) (the cubic's bound). Each
angle's peak starts at its largest projection over the points of largest
|w|; the steps are then rotated in blocks, largest bound first, and an
angle stops once its peak reaches the bound of the steps left.

Until GM-275 the responses came from the FFT, pyrotd 0.6's procedure (or
``rotd_native``, a numpy copy of it): with the record unpadded the response
wrapped round its end, and below ten record steps a period the record was
read as band-limited; against scipy's lsim -8 to +30 % at long periods and
up to +61 % at short ones (``DESIGN.md``, GM-274 and GM-275).

References
----------
Boore, D.M. (2010). Orientation-independent, nongeometric-mean measures of
seismic intensity from two horizontal components of motion. BSSA 100(4),
1830-1835. (Not in the reference library.)
Nigam, N.C. and Jennings, P.C. (1969). Calculation of response spectra
from strong-motion earthquake records. BSSA 59(2), 909-922.
"""

import math

import numpy as np

from geotech_common.response_spectrum import G, oscillator_response

#: Angles x steps rotated at once, at most (small enough to stay in the
#: processor's cache; a million took up to three times as long), and the
#: steps of the first block (half of it; the blocks double)
_BLOCK = 32_768
_FIRST_BLOCK = 128

#: The points of largest |w| whose projections bound each angle's peak
#: from below
_PROBE = 2000

#: What the result says about the method (``engine_note``)
METHOD_NOTE = (
    "Each component's oscillator response by Nigam and Jennings (1969), "
    "the record linear between samples, in sub-steps of at most a "
    "twentieth of the period; the two combined at every angle, the peak "
    "at each angle read between samples, RotDnn the nn-th percentile of "
    "the peaks over the angles (Boore 2010).")


def rotated_spectral_accels(dt, accel_a, accel_b, periods, damping=0.05,
                            percentiles=(0, 50, 100), angles=None):
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
    cos, sin = np.cos(ang), np.sin(ang)

    out = np.zeros((len(pct), len(periods)))
    for j, period in enumerate(periods):
        ua, va, h = oscillator_response(accel_a, dt, period, damping)
        ub, vb, _ = oscillator_response(accel_b, dt, period, damping)
        peaks = rotated_peaks(ua, va, ub, vb, h, cos, sin)
        omega = 2.0 * math.pi / period
        out[:, j] = omega ** 2 * np.percentile(peaks, pct,
                                               method="linear") / G
    return {float(p): out[i] for i, p in enumerate(pct)}


def rotated_peaks(ua, va, ub, vb, h, cos, sin):
    """The largest |u_a cos + u_b sin| over the record at each angle,
    between points too (the cubic through it and its velocity at both
    ends of each step of ``h``); one per angle."""
    n = len(ua)
    if n < 2:  # at rest at the first point
        return np.zeros(len(cos))
    mag = np.hypot(ua, ub)
    # every angle's peak is at least its largest projection among the
    # points of largest |w|
    top = np.argpartition(mag, n - min(n, _PROBE))[n - min(n, _PROBE):]
    peaks = np.max(np.abs(np.outer(cos, ua[top]) + np.outer(sin, ub[top])),
                   axis=1)
    # on a step no projection exceeds this (the cubic's bound): the steps
    # that can raise some angle's peak, largest bound first
    hv = h * np.hypot(va, vb)
    bound = (np.maximum(mag[:-1], mag[1:])
             + (4.0 / 27.0) * (hv[:-1] + hv[1:]))
    steps = np.nonzero(bound >= peaks.min() * (1.0 - 1e-9))[0]
    steps = steps[np.argsort(-bound[steps], kind="stable")]

    # rotate them in growing blocks; an angle is done once its peak
    # reaches the bound of the steps left
    active = np.arange(len(cos))
    start, size = 0, _FIRST_BLOCK
    while start < len(steps) and len(active):
        size = min(2 * size, max(_FIRST_BLOCK, _BLOCK // len(active)))
        k = steps[start:start + size]
        c, s = cos[active, None], sin[active, None]
        u0 = c * ua[k] + s * ub[k]
        u1 = c * ua[k + 1] + s * ub[k + 1]
        m0 = h * (c * va[k] + s * vb[k])
        m1 = h * (c * va[k + 1] + s * vb[k + 1])
        ends = np.maximum(np.abs(u0), np.abs(u1))
        got = np.maximum(peaks[active], ends.max(axis=1))
        # between its ends the cubic rises at most 4/27 (|m0| + |m1|)
        rows, cols = np.nonzero(
            ends + (4.0 / 27.0) * (np.abs(m0) + np.abs(m1)) > got[:, None])
        if len(rows):
            np.maximum.at(got, rows, _turning_peaks(
                u0[rows, cols], u1[rows, cols], m0[rows, cols],
                m1[rows, cols]))
        peaks[active] = got
        start += size
        if start < len(steps):
            left = bound[steps[start]] * (1.0 + 1e-9)
            active = active[peaks[active] < left]
    return peaks


def _turning_peaks(u0, u1, m0, m1):
    """The largest |u| at a turning point inside each step (0 where it has
    none) of the cubic u0 + m0 s + c2 s^2 + c3 s^3 (s from 0 to 1) through
    u and its slope at both ends, as ``peak_between_points`` reads it."""
    peak = np.zeros(len(u0))
    c2 = 3.0 * (u1 - u0) - 2.0 * m0 - m1
    c3 = 2.0 * (u0 - u1) + m0 + m1
    # its turning points, roots of 3 c3 s^2 + 2 c2 s + m0
    disc = c2 * c2 - 3.0 * c3 * m0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -(c2 + np.copysign(np.sqrt(disc), c2))
        for s in (m0 / q, q / (3.0 * c3)):
            inside = (s > 0.0) & (s < 1.0)  # false where nan or inf
            cubic = np.abs(u0 + s * (m0 + s * (c2 + s * c3)))
            peak = np.where(inside, np.maximum(peak, cubic), peak)
    return peak
