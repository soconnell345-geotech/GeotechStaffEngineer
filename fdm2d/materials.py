"""
Constitutive models for 2D explicit FDM (plane strain).

Standalone implementation — no imports from fem2d.

Provides:
- Linear elastic D-matrix (plane strain)
- Bulk/shear moduli and P-wave speed
- Mohr-Coulomb return mapping (2D in-plane Mohr circle)

Stress convention: tension-positive (compression is negative).
Stress vector: [sigma_xx, sigma_yy, tau_xy].
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Linear Elastic
# ---------------------------------------------------------------------------

def elastic_D(E, nu):
    """Plane-strain elastic constitutive matrix (3x3).

    Parameters
    ----------
    E : float — Young's modulus (kPa).
    nu : float — Poisson's ratio.

    Returns
    -------
    D : (3, 3) array
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ])


def bulk_shear_moduli(E, nu):
    """Compute bulk modulus K and shear modulus G.

    Parameters
    ----------
    E : float — Young's modulus (kPa).
    nu : float — Poisson's ratio.

    Returns
    -------
    K : float — bulk modulus (kPa).
    G : float — shear modulus (kPa).
    """
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return K, G


def wave_speed(K, G, rho):
    """P-wave speed for critical timestep calculation.

    Parameters
    ----------
    K : float — bulk modulus (kPa = kN/m²).
    G : float — shear modulus (kPa).
    rho : float — mass density (kN·s²/m⁴ = kg/m³ × 1e-3 if kN units).

    Returns
    -------
    vp : float — P-wave speed (m/s).
    """
    return math.sqrt((K + 4.0 * G / 3.0) / rho)


# ---------------------------------------------------------------------------
# Mohr-Coulomb return mapping (2D in-plane Mohr circle)
# ---------------------------------------------------------------------------

def mc_return_mapping(sigma_trial, E, nu, c, phi_deg, psi_deg=0.0):
    """Mohr-Coulomb return mapping for plane strain.

    Works directly in [sxx, syy, txy] space using the in-plane Mohr circle.
    Returns only (sigma_new, yielded) — no tangent D_ep (explicit solver
    doesn't need it).

    Tension-positive convention: compression is negative.

    MC yield (in-plane Mohr circle, tension-positive):
        f = q + p*sin(phi) - c*cos(phi)
    where p = (sxx+syy)/2, q = sqrt(((sxx-syy)/2)^2 + txy^2).

    Parameters
    ----------
    sigma_trial : (3,) array — trial stress [sigma_xx, sigma_yy, tau_xy].
    E, nu : float — elastic properties.
    c : float — cohesion (kPa).
    phi_deg : float — friction angle (degrees).
    psi_deg : float — dilation angle (degrees), default 0.

    Returns
    -------
    sigma_new : (3,) array — returned stress.
    yielded : bool — True if plastic correction was applied.
    """
    phi = math.radians(phi_deg)
    psi = math.radians(psi_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_psi = math.sin(psi)

    sxx, syy, txy = sigma_trial

    # In-plane Mohr circle invariants
    p = (sxx + syy) / 2.0
    d_xx = (sxx - syy) / 2.0
    d_xy = txy
    q = math.sqrt(d_xx ** 2 + d_xy ** 2)

    # Yield check
    f_trial = q + p * sin_phi - c * cos_phi

    if f_trial <= 0.0:
        return sigma_trial.copy(), False

    # --- Plastic return ---
    if q < 1e-15:
        # Hydrostatic tension exceeds apex
        if sin_phi > 1e-10:
            p_apex = c * cos_phi / sin_phi
        else:
            p_apex = p
        return np.array([p_apex, p_apex, 0.0]), True

    if sin_psi < 1e-15:
        # Non-dilative (psi=0): mean stress p unchanged, only q reduces
        q_new = c * cos_phi - p * sin_phi

        if q_new <= 0.0:
            # Apex return
            if sin_phi > 1e-10:
                p_apex = c * cos_phi / sin_phi
            else:
                p_apex = p
            return np.array([p_apex, p_apex, 0.0]), True

        # Scale deviatoric components
        r = q_new / q
        sxx_new = p + d_xx * r
        syy_new = p - d_xx * r
        txy_new = d_xy * r
    else:
        # General non-associated flow
        D3 = elastic_D(E, nu)
        n_vec = np.array([
            d_xx / (2.0 * q) + sin_phi / 2.0,
            -d_xx / (2.0 * q) + sin_phi / 2.0,
            d_xy / q,
        ])
        m_vec = np.array([
            d_xx / (2.0 * q) + sin_psi / 2.0,
            -d_xx / (2.0 * q) + sin_psi / 2.0,
            d_xy / q,
        ])

        Dm = D3 @ m_vec
        nDm = n_vec @ Dm
        if abs(nDm) < 1e-30:
            return sigma_trial.copy(), False

        d_lambda = f_trial / nDm
        sigma_new = sigma_trial - d_lambda * Dm

        # Check for apex
        sn_xx, sn_yy, sn_xy = sigma_new
        p_new = (sn_xx + sn_yy) / 2.0
        q_new = math.sqrt(((sn_xx - sn_yy) / 2.0) ** 2 + sn_xy ** 2)
        f_check = q_new + p_new * sin_phi - c * cos_phi

        if q_new < 1e-10 or f_check > 1.0:
            if sin_phi > 1e-10:
                p_apex = c * cos_phi / sin_phi
            else:
                p_apex = p
            return np.array([p_apex, p_apex, 0.0]), True

        return sigma_new, True

    sigma_new = np.array([sxx_new, syy_new, txy_new])
    return sigma_new, True
