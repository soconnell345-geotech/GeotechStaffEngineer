"""
Constitutive models for 2D plane-strain FEM.

Provides:
- Linear elastic D-matrix (plane strain)
- Mohr-Coulomb elastoplastic return mapping (2D in-plane Mohr circle)
- Drucker-Prager smooth approximation

Stress convention: tension-positive (compression is negative).
Stress vector: [sigma_xx, sigma_yy, tau_xy] for in-plane (3-component)
or [sigma_xx, sigma_yy, sigma_zz, tau_xy] for full (4-component).
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


def elastic_D_4(E, nu):
    """Plane-strain elastic constitutive matrix (4x4) including sigma_zz.

    Stress/strain vectors: [xx, yy, zz, xy].
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, nu, 0.0],
        [nu, 1.0 - nu, nu, 0.0],
        [nu, nu, 1.0 - nu, 0.0],
        [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ])


# ---------------------------------------------------------------------------
# Mohr-Coulomb return mapping (2D in-plane Mohr circle)
# ---------------------------------------------------------------------------

def mc_return_mapping(sigma_trial_3, E, nu, c, phi_deg, psi_deg=0.0,
                      sigma_t=0.0):
    """Mohr-Coulomb return mapping for plane strain.

    Works directly in the 2D [sxx, syy, txy] stress space using the
    in-plane Mohr circle criterion. This avoids inconsistency between
    the 3D principal return and the plane-strain szz constraint.

    Tension-positive convention: compression is negative.

    MC yield (in-plane Mohr circle, tension-positive):
        f = q + p*sin(phi) - c*cos(phi)
    where p = (sxx+syy)/2, q = sqrt(((sxx-syy)/2)^2 + txy^2).

    Parameters
    ----------
    sigma_trial_3 : (3,) array — trial stress [sigma_xx, sigma_yy, tau_xy].
    E, nu : float — elastic properties.
    c : float — cohesion (kPa).
    phi_deg : float — friction angle (degrees).
    psi_deg : float — dilation angle (degrees), default 0.
    sigma_t : float — tension cutoff (kPa), default 0.

    Returns
    -------
    sigma_new : (3,) array — returned stress.
    D_ep : (3, 3) array — tangent modulus (elastic approximation).
    yielded : bool — True if plastic correction was applied.
    """
    phi = math.radians(phi_deg)
    psi = math.radians(psi_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_psi = math.sin(psi)

    D3 = elastic_D(E, nu)

    sxx, syy, txy = sigma_trial_3

    # In-plane Mohr circle invariants
    p = (sxx + syy) / 2.0          # mean in-plane stress
    d_xx = (sxx - syy) / 2.0       # deviatoric normal
    d_xy = txy                      # deviatoric shear
    q = math.sqrt(d_xx ** 2 + d_xy ** 2)  # Mohr circle radius

    # Yield check (tension-positive):
    # f = q + p*sin(phi) - c*cos(phi)
    # p < 0 for compression → confining pressure adds strength
    f_trial = q + p * sin_phi - c * cos_phi

    if f_trial <= 0.0:
        return sigma_trial_3.copy(), D3.copy(), False

    # --- Plastic return ---
    # Target q on the yield surface at current p (for psi=0)
    # or at adjusted p (for psi>0)

    if q < 1e-15:
        # Hydrostatic tension exceeds apex — return to apex
        if sin_phi > 1e-10:
            p_apex = c * cos_phi / sin_phi  # MC apex in tension-positive p-q space
        else:
            p_apex = p
        return np.array([p_apex, p_apex, 0.0]), D3.copy(), True

    if sin_psi < 1e-15:
        # Non-dilative (psi=0): mean stress p unchanged, only q reduces
        q_new = c * cos_phi - p * sin_phi

        if q_new <= 0.0:
            # Apex return: all deviatoric stress removed
            if sin_phi > 1e-10:
                p_apex = c * cos_phi / sin_phi  # MC apex in tension-positive p-q space
            else:
                p_apex = p
            return np.array([p_apex, p_apex, 0.0]), D3.copy(), True

        # Scale deviatoric components
        r = q_new / q
        sxx_new = p + d_xx * r
        syy_new = p - d_xx * r
        txy_new = d_xy * r
    else:
        # General non-associated flow: use stress-space return mapping
        # Flow direction m = dg/d_sigma, yield normal n = df/d_sigma
        # g = q + p*sin(psi), f = q + p*sin(phi)
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
            return sigma_trial_3.copy(), D3.copy(), False

        d_lambda = f_trial / nDm
        sigma_new = sigma_trial_3 - d_lambda * Dm

        # Check for apex (q becomes negative after return)
        sn_xx, sn_yy, sn_xy = sigma_new
        p_new = (sn_xx + sn_yy) / 2.0
        q_new = math.sqrt(((sn_xx - sn_yy) / 2.0) ** 2 + sn_xy ** 2)
        f_check = q_new + p_new * sin_phi - c * cos_phi

        if q_new < 1e-10 or f_check > 1.0:
            # Apex return
            if sin_phi > 1e-10:
                p_apex = c * cos_phi / sin_phi  # MC apex in tension-positive p-q space
            else:
                p_apex = p
            return np.array([p_apex, p_apex, 0.0]), D3.copy(), True

        return sigma_new, D3.copy(), True

    sigma_new = np.array([sxx_new, syy_new, txy_new])
    return sigma_new, D3.copy(), True


# ---------------------------------------------------------------------------
# Drucker-Prager (plane-strain matching)
# ---------------------------------------------------------------------------

def drucker_prager_params(c, phi_deg):
    """Compute DP parameters matched to MC for plane strain.

    Parameters
    ----------
    c : float — cohesion (kPa).
    phi_deg : float — friction angle (degrees).

    Returns
    -------
    alpha : float — DP yield surface parameter.
    k : float — DP yield surface parameter.
    """
    tan_phi = math.tan(math.radians(phi_deg))
    denom = math.sqrt(9.0 + 12.0 * tan_phi ** 2)
    alpha = tan_phi / denom
    k = 3.0 * c / denom
    return alpha, k
