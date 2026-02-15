"""
Limit equilibrium methods for factor of safety computation.

Three methods in order of rigor:
1. Fellenius (OMS) — moment equilibrium only, no iteration
2. Bishop's Simplified — moment + vertical force equilibrium, iterative
3. Spencer — full force + moment equilibrium, two-variable iteration

All units SI: kPa, kN/m, degrees, meters.

References:
    Fellenius (1927)
    Bishop (1955) — Geotechnique, Vol. 5, pp. 7-17
    Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
    Duncan, Wright & Brandon (2014) — Chapters 6-7
"""

import math
import warnings
from typing import List, Tuple, Optional

from slope_stability.slices import Slice
from slope_stability.slip_surface import CircularSlipSurface


_FOS_MAX = 999.9  # sentinel for zero or negative driving


def fellenius_fos(slices: List[Slice],
                  slip: CircularSlipSurface) -> float:
    """Ordinary Method of Slices (Fellenius) factor of safety.

    FOS = sum[c'*dl + (W*cos(alpha) - u*dl)*tan(phi')] /
          sum[W*sin(alpha) + kh*W*(yc - z_centroid)/R]

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface
        Circle geometry (for seismic moment arms).

    Returns
    -------
    float
        Factor of safety.

    Notes
    -----
    Conservative: typically 10-60% lower than rigorous methods.
    No iteration required. Used as initial guess for Bishop.
    """
    resisting = 0.0
    driving = 0.0

    for s in slices:
        phi_rad = math.radians(s.phi)
        dl = s.base_length
        W = s.weight + s.surcharge_force

        # Resisting: shear strength along base
        N_prime = W * math.cos(s.alpha) - s.pore_pressure * dl
        resisting += s.c * dl + max(N_prime, 0.0) * math.tan(phi_rad)

        # Driving moment about circle center:
        # Moment arm = (x_mid - xc), equivalent to R*sin(alpha)
        driving += W * (s.x_mid - slip.xc) / slip.radius

        # Seismic: horizontal force * vertical arm about center
        # Use (z_centroid - yc) so seismic adds to driving in same direction as gravity
        if s.seismic_force != 0:
            driving += s.seismic_force * (s.z_centroid - slip.yc) / slip.radius

    # Driving can be negative for clockwise rotation (slope down L-to-R)
    # FOS is always positive: use absolute value
    driving = abs(driving)
    if driving <= 0:
        return _FOS_MAX

    return resisting / driving


def bishop_fos(slices: List[Slice],
               slip: CircularSlipSurface,
               tol: float = 1e-4,
               max_iter: int = 50,
               fos_initial: Optional[float] = None) -> float:
    """Bishop's Simplified Method factor of safety.

    FOS = sum[(c'*b + (W - u*b)*tan(phi')) / m_alpha] /
          sum[W*sin(alpha)]

    where m_alpha = cos(alpha) + sin(alpha)*tan(phi')/FOS

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface
        Circle geometry.
    tol : float
        Convergence tolerance on FOS. Default 1e-4.
    max_iter : int
        Maximum iterations. Default 50.
    fos_initial : float, optional
        Initial FOS guess. If None, uses Fellenius result.

    Returns
    -------
    float
        Factor of safety.

    Notes
    -----
    - Satisfies moment equilibrium and vertical force equilibrium
    - Most commonly used method in practice
    - Typically converges in 5-10 iterations
    """
    if fos_initial is None:
        fos_initial = fellenius_fos(slices, slip)
        if fos_initial >= _FOS_MAX:
            fos_initial = 1.5

    # Precompute driving sum (denominator — independent of FOS)
    # Use moment-arm formulation: W*(x_mid - xc)/R for numerical stability
    driving = 0.0
    for s in slices:
        W = s.weight + s.surcharge_force
        driving += W * (s.x_mid - slip.xc) / slip.radius
        if s.seismic_force != 0:
            driving += s.seismic_force * (s.z_centroid - slip.yc) / slip.radius

    # FOS is always positive: use absolute value of driving moment
    driving = abs(driving)
    if driving <= 0:
        return _FOS_MAX

    fos = fos_initial
    for iteration in range(max_iter):
        resisting = 0.0
        for s in slices:
            phi_rad = math.radians(s.phi)
            tan_phi = math.tan(phi_rad)
            W = s.weight + s.surcharge_force
            b = s.width

            m_alpha = math.cos(s.alpha) + math.sin(s.alpha) * tan_phi / fos
            if abs(m_alpha) < 1e-10:
                m_alpha = 1e-10

            numerator = s.c * b + (W - s.pore_pressure * b) * tan_phi
            resisting += numerator / m_alpha

        fos_new = resisting / driving
        if abs(fos_new - fos) < tol:
            return fos_new
        fos = fos_new

    warnings.warn(
        f"Bishop did not converge after {max_iter} iterations "
        f"(last FOS={fos:.4f})"
    )
    return fos


def spencer_fos(slices: List[Slice],
                slip: CircularSlipSurface,
                tol: float = 1e-4,
                max_iter: int = 100) -> Tuple[float, float]:
    """Spencer's Method factor of safety.

    Satisfies both force and moment equilibrium simultaneously.
    Assumes interslice forces inclined at constant angle theta.

    Finds (FOS, theta) such that FOS_moment(theta) = FOS_force(theta).

    Both equations use Spencer's m_alpha:
        m_alpha = cos(alpha - theta) + sin(alpha - theta)*tan(phi')/FOS

    When theta=0 this reduces to Bishop. For c-phi soils, the converged
    theta is typically non-zero (±5-15°).

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface
        Circle geometry.
    tol : float
        Convergence tolerance. Default 1e-4.
    max_iter : int
        Maximum iterations for outer theta loop. Default 100.

    Returns
    -------
    (FOS, theta) : tuple of (float, float)
        Factor of safety and interslice force angle (degrees).

    Notes
    -----
    Algorithm: secant method to find theta where FOS_m(theta) = FOS_f(theta).
    For circular surfaces, Spencer is typically within 2-5% of Bishop.

    References
    ----------
    Spencer (1967), Geotechnique, Vol. 17, pp. 11-26
    xslope formulation: m_alpha = cos(alpha-theta) + sin(alpha-theta)*tan(phi')/F
    """
    # Get Bishop FOS as starting guess
    fos_guess = bishop_fos(slices, slip)
    if fos_guess >= _FOS_MAX:
        return (_FOS_MAX, 0.0)

    # Precompute driving moment (independent of theta and FOS)
    driving_moment = 0.0
    for s in slices:
        W = s.weight + s.surcharge_force
        driving_moment += W * (s.x_mid - slip.xc) / slip.radius
        if s.seismic_force != 0:
            driving_moment += s.seismic_force * (s.z_centroid - slip.yc) / slip.radius
    driving_moment = abs(driving_moment)
    if driving_moment <= 0:
        return (_FOS_MAX, 0.0)

    def _fos_moment(theta_rad, fos_est):
        """Moment equilibrium FOS for given theta.

        Fm = sum[(c'*b + (W - u*b)*tan(phi')) / m_alpha] / sum[W*sin(alpha)]

        where m_alpha = cos(alpha - theta) + sin(alpha - theta)*tan(phi')/Fm
        (Spencer's m_alpha, NOT Bishop's).
        """
        fos_m = fos_est
        for _ in range(50):
            resisting = 0.0
            for s in slices:
                phi_rad = math.radians(s.phi)
                tan_phi = math.tan(phi_rad)
                W = s.weight + s.surcharge_force
                b = s.width

                # Spencer m_alpha: uses (alpha - theta)
                diff = s.alpha - theta_rad
                m_alpha = math.cos(diff) + math.sin(diff) * tan_phi / fos_m
                if abs(m_alpha) < 1e-10:
                    m_alpha = 1e-10

                resisting += (s.c * b + (W - s.pore_pressure * b) * tan_phi) / m_alpha

            fos_new = resisting / driving_moment
            if abs(fos_new - fos_m) < tol * 0.1:
                return fos_new
            fos_m = fos_new
        return fos_m

    def _fos_force(theta_rad, fos_est):
        """Force equilibrium FOS for given theta.

        Ff = sum[(c'*b + (W - u*b)*tan(phi')) / m_alpha] / sum[W*d_alpha]

        where m_alpha = cos(alpha - theta) + sin(alpha - theta)*tan(phi')/Ff
        and   d_alpha = sin(alpha) + cos(alpha)*tan(theta)
        """
        tan_theta = math.tan(theta_rad)

        fos_f = fos_est
        for _ in range(50):
            total_resist = 0.0
            total_drive = 0.0
            for s in slices:
                phi_rad = math.radians(s.phi)
                tan_phi = math.tan(phi_rad)
                W = s.weight + s.surcharge_force
                b = s.width
                sin_a = math.sin(s.alpha)
                cos_a = math.cos(s.alpha)

                # Spencer m_alpha: uses (alpha - theta)
                diff = s.alpha - theta_rad
                m_alpha = math.cos(diff) + math.sin(diff) * tan_phi / fos_f
                if abs(m_alpha) < 1e-10:
                    m_alpha = 1e-10

                total_resist += (s.c * b + (W - s.pore_pressure * b) * tan_phi) / m_alpha

                # Force driving: horizontal projection with interslice angle
                d_alpha = sin_a + cos_a * tan_theta
                total_drive += W * d_alpha
                if s.seismic_force != 0:
                    n_alpha = cos_a - sin_a * tan_theta
                    total_drive += s.seismic_force * n_alpha

            total_drive = abs(total_drive)
            if total_drive <= 0:
                return _FOS_MAX

            fos_new = total_resist / total_drive
            if abs(fos_new - fos_f) < tol * 0.1:
                return fos_new
            fos_f = fos_new

        return fos_f

    # Secant method on f(theta) = FOS_m(theta) - FOS_f(theta)
    theta_a = 0.0
    theta_b = math.radians(10.0)

    fm_a = _fos_moment(theta_a, fos_guess)
    ff_a = _fos_force(theta_a, fos_guess)
    fa = fm_a - ff_a

    for iteration in range(max_iter):
        fm_b = _fos_moment(theta_b, fos_guess)
        ff_b = _fos_force(theta_b, fos_guess)
        fb = fm_b - ff_b

        if abs(fb) < tol:
            fos_final = 0.5 * (fm_b + ff_b)
            return (fos_final, math.degrees(theta_b))

        # Secant update
        if abs(fb - fa) < 1e-12:
            break
        theta_c = theta_b - fb * (theta_b - theta_a) / (fb - fa)

        # Clamp theta to reasonable range (-30 to +30 degrees)
        theta_c = max(math.radians(-30), min(math.radians(30), theta_c))

        theta_a, fa = theta_b, fb
        theta_b = theta_c
        fos_guess = 0.5 * (fm_b + ff_b)

    # Return best estimate
    fos_final = 0.5 * (_fos_moment(theta_b, fos_guess) +
                        _fos_force(theta_b, fos_guess))
    return (fos_final, math.degrees(theta_b))
