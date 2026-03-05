"""
Limit equilibrium methods for factor of safety computation.

Three methods in order of rigor:
1. Fellenius (OMS) — moment equilibrium only, no iteration
2. Bishop's Simplified — moment + vertical force equilibrium, iterative
3. Spencer — full force + moment equilibrium, two-variable iteration

Spencer supports both circular and noncircular slip surfaces.
Bishop and Fellenius require circular surfaces (moment equilibrium
about circle center).

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


_FOS_MAX = 999.9  # sentinel for zero or negative driving


# Morgenstern-Price interslice force functions
def _f_constant(x_norm):
    """Constant f(x) = 1 (equivalent to Spencer)."""
    return 1.0


def _f_half_sine(x_norm):
    """Half-sine f(x) = sin(pi * x_norm), x_norm in [0, 1]."""
    return math.sin(math.pi * x_norm)


def _f_trapezoidal(x_norm):
    """Trapezoidal f(x): ramps up in first quarter, flat middle, ramps down last quarter."""
    if x_norm < 0.25:
        return 4.0 * x_norm
    elif x_norm > 0.75:
        return 4.0 * (1.0 - x_norm)
    return 1.0


_MP_FUNCTIONS = {
    "constant": _f_constant,
    "half_sine": _f_half_sine,
    "trapezoidal": _f_trapezoidal,
}


def fellenius_fos(slices: List[Slice],
                  slip) -> float:
    """Ordinary Method of Slices (Fellenius) factor of safety.

    For circular surfaces:
        FOS = sum[c'*dl + (W*cos(alpha) - u*dl)*tan(phi')] /
              sum[W*(x_mid-xc)/R + kh*W*(yc-z_centroid)/R]

    For noncircular surfaces:
        FOS = sum[c'*dl + (W*cos(alpha) - u*dl)*tan(phi')] /
              sum[W*sin(alpha) + kh*W*cos(alpha)]

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface or PolylineSlipSurface
        Slip surface geometry.

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
    is_circular = getattr(slip, 'is_circular', True)

    gravity_driving = 0.0
    seismic_driving = 0.0
    crack_water_driving = 0.0

    for s in slices:
        phi_rad = math.radians(s.phi)
        dl = s.base_length
        W = s.weight + s.surcharge_force

        # Resisting: shear strength along base
        N_prime = W * math.cos(s.alpha) - s.pore_pressure * dl
        resisting += s.c * dl + max(N_prime, 0.0) * math.tan(phi_rad)

        if is_circular:
            # Circular: moment-arm formulation W*(x_mid - xc)/R
            gravity_driving += W * (s.x_mid - slip.xc) / slip.radius
            if s.seismic_force != 0:
                seismic_driving += s.seismic_force * (slip.yc - s.z_centroid) / slip.radius
            if s.crack_water_force != 0:
                crack_water_driving += s.crack_water_force * (slip.yc - s.crack_water_z) / slip.radius
        else:
            # Noncircular: use W*sin(alpha) directly
            gravity_driving += W * math.sin(s.alpha)
            if s.seismic_force != 0:
                seismic_driving += s.seismic_force * math.cos(s.alpha)
            if s.crack_water_force != 0:
                crack_water_driving += s.crack_water_force * math.cos(s.alpha)

    # Driving = gravity magnitude + seismic magnitude + crack water.
    # Separated so seismic always increases driving regardless of slope direction.
    driving = abs(gravity_driving) + abs(seismic_driving) + abs(crack_water_driving)
    if driving <= 0:
        return _FOS_MAX

    return resisting / driving


def bishop_fos(slices: List[Slice],
               slip,
               tol: float = 1e-4,
               max_iter: int = 50,
               fos_initial: Optional[float] = None) -> float:
    """Bishop's Simplified Method factor of safety.

    Requires circular slip surfaces only (moment equilibrium about
    circle center).

    FOS = sum[(c'*b + (W - u*b)*tan(phi')) / m_alpha] /
          sum[W*sin(alpha)]

    where m_alpha = cos(alpha) + sin(alpha)*tan(phi')/FOS

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface
        Circle geometry. Must be circular.
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

    Raises
    ------
    ValueError
        If slip surface is not circular.

    Notes
    -----
    - Satisfies moment equilibrium and vertical force equilibrium
    - Most commonly used method in practice
    - Typically converges in 5-10 iterations
    """
    if not getattr(slip, 'is_circular', True):
        raise ValueError(
            "Bishop's method requires a circular slip surface. "
            "Use Spencer's method for noncircular surfaces."
        )

    if fos_initial is None:
        fos_initial = fellenius_fos(slices, slip)
        if fos_initial >= _FOS_MAX:
            fos_initial = 1.5

    # Precompute driving sum (denominator — independent of FOS)
    # Use moment-arm formulation: W*(x_mid - xc)/R for numerical stability
    gravity_driving = 0.0
    seismic_driving = 0.0
    crack_water_driving = 0.0
    for s in slices:
        W = s.weight + s.surcharge_force
        gravity_driving += W * (s.x_mid - slip.xc) / slip.radius
        if s.seismic_force != 0:
            seismic_driving += s.seismic_force * (slip.yc - s.z_centroid) / slip.radius
        if s.crack_water_force != 0:
            crack_water_driving += s.crack_water_force * (slip.yc - s.crack_water_z) / slip.radius

    # Separated so seismic always increases driving regardless of slope direction
    driving = abs(gravity_driving) + abs(seismic_driving) + abs(crack_water_driving)
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
                slip,
                tol: float = 1e-4,
                max_iter: int = 100) -> Tuple[float, float]:
    """Spencer's Method factor of safety.

    Satisfies both force and moment equilibrium simultaneously.
    Assumes interslice forces inclined at constant angle theta.

    Works for both circular and noncircular slip surfaces.
    For noncircular surfaces, this is the primary recommended method.

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface or PolylineSlipSurface
        Slip surface geometry (circular or noncircular).
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
    For circular surfaces: uses moment-arm formulation (same as existing code).
    For noncircular surfaces: uses force-based formulation with W*sin(alpha)
    for driving. The secant iteration on theta finds where FOS_moment = FOS_force.

    References
    ----------
    Spencer (1967), Geotechnique, Vol. 17, pp. 11-26
    Duncan, Wright & Brandon (2014), Chapter 7
    """
    is_circular = getattr(slip, 'is_circular', True)

    # Get initial FOS guess from Fellenius
    fos_guess = fellenius_fos(slices, slip)
    if fos_guess >= _FOS_MAX:
        fos_guess = 1.5

    # Precompute driving moment
    if is_circular:
        gravity_moment = 0.0
        seismic_moment = 0.0
        crack_water_moment = 0.0
        for s in slices:
            W = s.weight + s.surcharge_force
            gravity_moment += W * (s.x_mid - slip.xc) / slip.radius
            if s.seismic_force != 0:
                seismic_moment += s.seismic_force * (slip.yc - s.z_centroid) / slip.radius
            if s.crack_water_force != 0:
                crack_water_moment += s.crack_water_force * (slip.yc - s.crack_water_z) / slip.radius
        driving_moment = abs(gravity_moment) + abs(seismic_moment) + abs(crack_water_moment)
    else:
        # Noncircular: use W*sin(alpha) for driving
        gravity_moment = 0.0
        seismic_moment = 0.0
        crack_water_moment = 0.0
        for s in slices:
            W = s.weight + s.surcharge_force
            gravity_moment += W * math.sin(s.alpha)
            if s.seismic_force != 0:
                seismic_moment += s.seismic_force * math.cos(s.alpha)
            if s.crack_water_force != 0:
                crack_water_moment += s.crack_water_force * math.cos(s.alpha)
        driving_moment = abs(gravity_moment) + abs(seismic_moment) + abs(crack_water_moment)

    if driving_moment <= 0:
        return (_FOS_MAX, 0.0)

    def _fos_moment(theta_rad, fos_est):
        """Moment equilibrium FOS for given theta."""
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
        """Force equilibrium FOS for given theta."""
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
                if s.crack_water_force != 0:
                    n_alpha = cos_a - sin_a * tan_theta
                    total_drive += s.crack_water_force * n_alpha

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


def morgenstern_price_fos(slices: List[Slice],
                          slip,
                          f_interslice: str = "half_sine",
                          tol: float = 1e-4,
                          max_iter: int = 100) -> Tuple[float, float]:
    """Morgenstern-Price / GLE factor of safety.

    Generalizes Spencer by using a variable interslice force function f(x)
    scaled by lambda. When f(x) = constant, this reduces to Spencer.

    Solves for both FOS and lambda simultaneously using secant iteration:
    find lambda where FOS_moment(lambda) = FOS_force(lambda).

    Works for both circular and noncircular slip surfaces.

    Parameters
    ----------
    slices : list of Slice
        Slice data from build_slices().
    slip : CircularSlipSurface or PolylineSlipSurface
        Slip surface geometry.
    f_interslice : str
        Interslice force function: 'constant', 'half_sine', 'trapezoidal'.
        Default 'half_sine'.
    tol : float
        Convergence tolerance. Default 1e-4.
    max_iter : int
        Maximum iterations for outer lambda loop. Default 100.

    Returns
    -------
    (FOS, lambda) : tuple of (float, float)
        Factor of safety and interslice force scaling factor.

    References
    ----------
    Morgenstern & Price (1965), Geotechnique, Vol. 15, pp. 79-93
    Duncan, Wright & Brandon (2014), Chapter 7
    """
    f_func = _MP_FUNCTIONS.get(f_interslice, _f_half_sine)
    is_circular = getattr(slip, 'is_circular', True)
    n = len(slices)

    # Compute normalized x positions for f(x)
    x_vals = [s.x_mid for s in slices]
    x_lo = min(x_vals)
    x_hi = max(x_vals)
    x_span = x_hi - x_lo if x_hi > x_lo else 1.0
    # f(x) evaluated at slice boundaries (between slices)
    f_vals = []
    for i in range(n - 1):
        x_norm = ((x_vals[i] + x_vals[i + 1]) / 2.0 - x_lo) / x_span
        f_vals.append(f_func(x_norm))

    # Get initial FOS from Fellenius
    fos_guess = fellenius_fos(slices, slip)
    if fos_guess >= _FOS_MAX:
        fos_guess = 1.5

    # Precompute driving for moment equilibrium
    if is_circular:
        gravity_driving = 0.0
        seismic_driving = 0.0
        crack_water_driving = 0.0
        for s in slices:
            W = s.weight + s.surcharge_force
            gravity_driving += W * (s.x_mid - slip.xc) / slip.radius
            if s.seismic_force != 0:
                seismic_driving += s.seismic_force * (slip.yc - s.z_centroid) / slip.radius
            if s.crack_water_force != 0:
                crack_water_driving += s.crack_water_force * (slip.yc - s.crack_water_z) / slip.radius
        driving_moment = abs(gravity_driving) + abs(seismic_driving) + abs(crack_water_driving)
    else:
        gravity_driving = 0.0
        seismic_driving = 0.0
        crack_water_driving = 0.0
        for s in slices:
            W = s.weight + s.surcharge_force
            gravity_driving += W * math.sin(s.alpha)
            if s.seismic_force != 0:
                seismic_driving += s.seismic_force * math.cos(s.alpha)
            if s.crack_water_force != 0:
                crack_water_driving += s.crack_water_force * math.cos(s.alpha)
        driving_moment = abs(gravity_driving) + abs(seismic_driving) + abs(crack_water_driving)

    if driving_moment <= 0:
        return (_FOS_MAX, 0.0)

    def _fos_moment(lam, fos_est):
        """Moment equilibrium FOS for given lambda."""
        fos_m = fos_est
        for _ in range(50):
            resisting = 0.0
            for i, s in enumerate(slices):
                phi_rad = math.radians(s.phi)
                tan_phi = math.tan(phi_rad)
                W = s.weight + s.surcharge_force
                b = s.width

                # Effective theta for this slice via lambda * f(x)
                # At slice boundaries: theta_i = atan(lambda * f_i)
                # Average over left and right boundary
                if i == 0:
                    theta_eff = math.atan(lam * f_vals[0]) if f_vals else 0.0
                elif i == n - 1:
                    theta_eff = math.atan(lam * f_vals[-1]) if f_vals else 0.0
                else:
                    theta_eff = math.atan(lam * 0.5 * (f_vals[i - 1] + f_vals[i]))

                diff = s.alpha - theta_eff
                m_alpha = math.cos(diff) + math.sin(diff) * tan_phi / fos_m
                if abs(m_alpha) < 1e-10:
                    m_alpha = 1e-10

                resisting += (s.c * b + (W - s.pore_pressure * b) * tan_phi) / m_alpha

            fos_new = resisting / driving_moment
            if abs(fos_new - fos_m) < tol * 0.1:
                return fos_new
            fos_m = fos_new
        return fos_m

    def _fos_force(lam, fos_est):
        """Force equilibrium FOS for given lambda."""
        fos_f = fos_est
        for _ in range(50):
            total_resist = 0.0
            total_drive = 0.0
            for i, s in enumerate(slices):
                phi_rad = math.radians(s.phi)
                tan_phi = math.tan(phi_rad)
                W = s.weight + s.surcharge_force
                b = s.width
                sin_a = math.sin(s.alpha)
                cos_a = math.cos(s.alpha)

                if i == 0:
                    theta_eff = math.atan(lam * f_vals[0]) if f_vals else 0.0
                elif i == n - 1:
                    theta_eff = math.atan(lam * f_vals[-1]) if f_vals else 0.0
                else:
                    theta_eff = math.atan(lam * 0.5 * (f_vals[i - 1] + f_vals[i]))

                tan_theta = math.tan(theta_eff)

                diff = s.alpha - theta_eff
                m_alpha = math.cos(diff) + math.sin(diff) * tan_phi / fos_f
                if abs(m_alpha) < 1e-10:
                    m_alpha = 1e-10

                total_resist += (s.c * b + (W - s.pore_pressure * b) * tan_phi) / m_alpha

                d_alpha = sin_a + cos_a * tan_theta
                total_drive += W * d_alpha
                if s.seismic_force != 0:
                    n_alpha = cos_a - sin_a * tan_theta
                    total_drive += s.seismic_force * n_alpha
                if s.crack_water_force != 0:
                    n_alpha = cos_a - sin_a * tan_theta
                    total_drive += s.crack_water_force * n_alpha

            total_drive = abs(total_drive)
            if total_drive <= 0:
                return _FOS_MAX

            fos_new = total_resist / total_drive
            if abs(fos_new - fos_f) < tol * 0.1:
                return fos_new
            fos_f = fos_new

        return fos_f

    # Secant method on g(lambda) = FOS_m(lambda) - FOS_f(lambda)
    lam_a = 0.0
    lam_b = 0.3

    fm_a = _fos_moment(lam_a, fos_guess)
    ff_a = _fos_force(lam_a, fos_guess)
    ga = fm_a - ff_a

    for _ in range(max_iter):
        fm_b = _fos_moment(lam_b, fos_guess)
        ff_b = _fos_force(lam_b, fos_guess)
        gb = fm_b - ff_b

        if abs(gb) < tol:
            fos_final = 0.5 * (fm_b + ff_b)
            return (fos_final, lam_b)

        if abs(gb - ga) < 1e-12:
            break
        lam_c = lam_b - gb * (lam_b - lam_a) / (gb - ga)

        # Clamp lambda to reasonable range
        lam_c = max(-2.0, min(2.0, lam_c))

        lam_a, ga = lam_b, gb
        lam_b = lam_c
        fos_guess = 0.5 * (fm_b + ff_b)

    fos_final = 0.5 * (_fos_moment(lam_b, fos_guess) +
                        _fos_force(lam_b, fos_guess))
    return (fos_final, lam_b)
