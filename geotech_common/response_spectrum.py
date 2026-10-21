"""
The pseudo-acceleration response spectrum of a ground motion record, every
period by Nigam and Jennings (1969).

One code for the tools that compute a spectrum from a record (GM-274): the
OpenSees site response (``opensees_agent.opensees_utils.
compute_response_spectrum``, GM-272) and the seismic-signals response
spectrum (``seismic_signals_agent.analyze_response_spectrum``), so the two
give the same Sa for the same record. numpy and scipy only.

The record is taken as linear between its samples and each oscillator's
response to it is solved in closed form, step by step. A period shorter
than ``POINTS_PER_PERIOD`` record steps is integrated on the same straight
lines split into equal sub-steps of at most T / 20, and between points the
largest response is read off the cubic through the displacement and
velocity at both ends of each step (Hermite). Against scipy's lsim (the
same piecewise-linear record through the matrix exponential, read every
T / 400 or dt / 50, whichever is finer) on three site response records and
their inputs, 27 periods from 0.01 to 4 s: within 0.004 % (GM-272).

Until GM-274 the seismic-signals spectrum came from eqsig, whose step is
this one but which integrates the whole record at the larger of the
shortest period / 20 and a quarter of the record's step, reads the peak
only at its points and gives the PGA below six of them: up to 2.5 % low
below ten record steps a period, within 0.04 % from ten up.

Reference
---------
Nigam, N.C. and Jennings, P.C. (1969). Calculation of response spectra
from strong-motion earthquake records. BSSA 59(2), 909-922.
"""

import math

import numpy as np

#: Sub-steps a period below which the record's straight lines are split
#: (GM-272)
POINTS_PER_PERIOD = 20

#: m/s^2 in one g (the tools' convention)
G = 9.81


def pseudo_spectral_accel(accel_g, dt, periods, damping=0.05):
    """Pseudo-spectral acceleration Sa (g) of a record at each period,
    every period by Nigam and Jennings (1969), in sub-steps of at most a
    twentieth of the period below 20 record steps, the peak read between
    points.

    At the points the response is the closed-form response to the record
    taken as linear between its samples, with no integration error; a
    period of 0 or less gives the PGA.

    Parameters
    ----------
    accel_g : array_like
        Ground acceleration time history (g).
    dt : float
        Time step (s).
    periods : array_like
        Spectral periods (s), in any order.
    damping : float
        Damping ratio, at least 0 and below 1. Default 0.05 (5 %).

    Returns
    -------
    numpy.ndarray
        Sa (g), one per period, in the order of ``periods``.
    """
    accel = np.asarray(accel_g, dtype=float)
    periods = np.asarray(periods, dtype=float)
    if not 0.0 <= damping < 1.0:
        raise ValueError(f"damping is a ratio of critical, at least 0 and "
                         f"below 1; got {damping}")

    Sa = np.zeros(len(periods))
    p_record = -accel * G  # effective force = -m * ag (unit mass)
    finer = {1: p_record}

    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = np.max(np.abs(accel))
            continue

        # Sub-steps for a period shorter than POINTS_PER_PERIOD steps, on
        # the same straight lines between the record's samples
        n_sub = max(1, math.ceil(POINTS_PER_PERIOD * dt / T - 1e-9))
        if n_sub not in finer:
            finer[n_sub] = substeps(p_record, n_sub)
        h = dt / n_sub
        omega = 2.0 * math.pi / T
        u, v = nigam_jennings(finer[n_sub], h, omega, damping)
        # Pseudo-spectral acceleration: Sa = omega^2 * Sd
        Sa[i] = omega ** 2 * peak_between_points(u, v, h) / G

    return Sa


def substeps(p, n):
    """``p`` with ``n - 1`` points put evenly on the straight line between
    each pair of its samples."""
    f = np.arange(n) / n
    return np.append((p[:-1, None] + np.diff(p)[:, None] * f).ravel(), p[-1:])


def nigam_jennings(p, h, omega, damping):
    """Displacement and velocity at every point of a unit-mass oscillator of
    circular frequency ``omega`` and damping ratio ``damping``, at rest at
    the first point, under the force ``p`` taken as linear between points
    ``h`` apart: the closed-form step of Nigam and Jennings (1969),

        u[j+1] = A u[j] + B v[j] + C p[j] + D p[j+1]
        v[j+1] = A1 u[j] + B1 v[j] + C1 p[j] + D1 p[j+1],

    run as the two-step recursion it implies for each (scipy ``lfilter``)."""
    from scipy.signal import lfilter, lfiltic
    k = omega * omega
    r = math.sqrt(1.0 - damping * damping)
    wd = omega * r  # damped circular frequency
    e = math.exp(-damping * omega * h)
    s, c = math.sin(wd * h), math.cos(wd * h)
    zr = damping / r
    zh = 2.0 * damping / (omega * h)
    A = e * (zr * s + c)
    B = e * s / wd
    C = (zh + e * (((1.0 - 2.0 * damping ** 2) / (wd * h) - zr) * s
                   - (1.0 + zh) * c)) / k
    D = (1.0 - zh + e * ((2.0 * damping ** 2 - 1.0) / (wd * h) * s
                         + zh * c)) / k
    A1 = -e * omega / r * s
    B1 = e * (c - zr * s)
    C1 = (-1.0 / h + e * ((omega / r + zr / h) * s + c / h)) / k
    D1 = (1.0 - e * (zr * s + c)) / (k * h)

    u = np.zeros(len(p))
    v = np.zeros(len(p))
    if len(p) < 2:
        return u, v
    u[1] = C * p[0] + D * p[1]
    v[1] = C1 * p[0] + D1 * p[1]
    if len(p) > 2:
        a = [1.0, -(A + B1), A * B1 - B * A1]
        for x, b in ((u, [D, C - B1 * D + B * D1, B * C1 - B1 * C]),
                     (v, [D1, C1 - A * D1 + A1 * D, A1 * C - A * C1])):
            zi = lfiltic(b, a, [x[1], 0.0], [p[1], p[0]])
            x[2:] = lfilter(b, a, p[2:], zi=zi)[0]
    return u, v


def peak_between_points(u, v, h):
    """The largest |u| over the record, between its points too: on each
    step of ``h`` the cubic through u and v at both ends (Hermite), at its
    turning points inside the step."""
    if len(u) == 0:
        return 0.0
    peak = float(np.max(np.abs(u)))
    u0, u1, m0, m1 = u[:-1], u[1:], h * v[:-1], h * v[1:]
    # the cubic u0 + m0 s + c2 s^2 + c3 s^3 over the step (s from 0 to 1)
    c2 = 3.0 * (u1 - u0) - 2.0 * m0 - m1
    c3 = 2.0 * (u0 - u1) + m0 + m1
    # its turning points, roots of 3 c3 s^2 + 2 c2 s + m0
    disc = c2 * c2 - 3.0 * c3 * m0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = -(c2 + np.copysign(np.sqrt(disc), c2))
        for s in (m0 / q, q / (3.0 * c3)):
            inside = (s > 0.0) & (s < 1.0)  # false where nan or inf
            if np.any(inside):
                s = s[inside]
                cubic = u0[inside] + s * (m0[inside] + s * (
                    c2[inside] + s * c3[inside]))
                peak = max(peak, float(np.max(np.abs(cubic))))
    return peak
