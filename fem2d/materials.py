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
# Hardening Soil model (shear hardening, Schanz et al. 1999)
# ---------------------------------------------------------------------------

def _inplane_principals(sigma_3):
    """Compute in-plane principal stresses from [sxx, syy, txy].

    Returns (s1, s2) with s1 >= s2 (algebraic ordering, tension-positive).
    """
    sxx, syy, txy = sigma_3
    p = (sxx + syy) / 2.0
    R = math.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)
    return p + R, p - R


def hs_return_mapping(sigma_trial_3, state, E50_ref, Eur_ref, m, p_ref,
                      R_f, nu, c, phi_deg, psi_deg=0.0):
    """Hardening Soil return mapping for plane strain (shear hardening only).

    Hyperbolic deviatoric response with stress-dependent stiffness and
    Mohr-Coulomb failure envelope.  Unload/reload uses the stiffer E_ur.

    Theory: Schanz, Vermeer & Bonnier (1999).

    Parameters
    ----------
    sigma_trial_3 : (3,) array — trial stress [sigma_xx, sigma_yy, tau_xy].
    state : dict — internal state variables.
        'gamma_p_s' : float — accumulated plastic shear strain.
        'sigma_prev' : (3,) array — stress at end of last converged step.
        'loading' : bool — True if primary loading in previous step.
    E50_ref : float — secant stiffness at 50% strength at p_ref (kPa).
    Eur_ref : float — unload/reload stiffness at p_ref (kPa).
    m : float — power-law exponent for stress dependency (typically 0.5-1.0).
    p_ref : float — reference confining pressure (kPa, positive).
    R_f : float — failure ratio q_f/q_a (typically 0.9).
    nu : float — Poisson's ratio.
    c : float — cohesion (kPa).
    phi_deg : float — friction angle (degrees).
    psi_deg : float — dilation angle (degrees), default 0.

    Returns
    -------
    sigma_new : (3,) array — returned stress.
    D_tang : (3, 3) array — tangent modulus.
    yielded : bool — True if MC failure was reached.
    state_new : dict — updated state variables.
    """
    phi = math.radians(phi_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    # --- Confining stress (minor principal, compression-positive) ---
    s1, s2 = _inplane_principals(sigma_trial_3)
    # In tension-positive convention, confining = most compressive = min
    sigma3_comp = -min(s1, s2)  # convert to compression-positive for HS eqs
    if sigma3_comp < 0.0:
        sigma3_comp = 0.0  # clamp to zero if in tension

    # --- Stress-dependent stiffness ---
    denom = c * cos_phi + p_ref * sin_phi
    if denom < 1e-10:
        denom = 1e-10
    numer = c * cos_phi + sigma3_comp * sin_phi
    if numer < 1e-10:
        numer = 1e-10
    stress_ratio = (numer / denom) ** m

    E_50 = max(E50_ref * stress_ratio, E50_ref * 0.01)
    E_ur = max(Eur_ref * stress_ratio, Eur_ref * 0.01)

    # --- Deviatoric stress and failure ---
    sxx, syy, txy = sigma_trial_3
    p_mean = (sxx + syy) / 2.0
    q = math.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)

    # MC failure deviatoric stress (tension-positive Mohr circle):
    # q_f = c*cos(phi) - p*sin(phi)  (this is the MC yield: f = q - q_f)
    q_f = c * cos_phi - p_mean * sin_phi
    if q_f < 1e-10:
        q_f = 1e-10

    # Asymptotic deviatoric stress
    q_a = q_f / max(R_f, 0.01)

    # --- Detect loading vs unloading ---
    sigma_prev = state.get('sigma_prev', np.zeros(3))
    sxx_p, syy_p, txy_p = sigma_prev
    q_prev = math.sqrt(((sxx_p - syy_p) / 2.0) ** 2 + txy_p ** 2)
    gamma_p_s = state.get('gamma_p_s', 0.0)

    is_loading = q >= q_prev - 1e-10

    # --- Build tangent stiffness ---
    if is_loading and q > 1e-10:
        # Primary loading: hyperbolic tangent
        # E_t = E_50 * (1 - R_f * q / q_f)^2
        ratio = R_f * q / q_f
        if ratio >= 1.0:
            # At or beyond failure — use very soft tangent
            E_t = E_50 * 0.001
        else:
            E_t = E_50 * (1.0 - ratio) ** 2
        E_t = max(E_t, E_50 * 0.001)  # floor

        # Update plastic shear strain
        # eps_s = q_a * q / (E_50 * (q_a - q))  for q < q_a
        if q < q_a - 1e-10:
            eps_s_total = q_a * q / (E_50 * (q_a - q))
        else:
            eps_s_total = q_a / E_50 * 10.0  # cap
        eps_s_elastic = q / E_ur
        gamma_p_s_new = max(eps_s_total - eps_s_elastic, gamma_p_s)
    else:
        # Unload/reload: use E_ur (stiffer)
        E_t = E_ur
        gamma_p_s_new = gamma_p_s

    # Build D-matrix with effective tangent modulus
    D_tang = _build_D_planestrain(E_t, nu)

    # --- Check MC failure — if exceeded, return to yield surface ---
    f_trial = q + p_mean * sin_phi - c * cos_phi
    if f_trial > 0.0:
        sigma_new, D_ep, _ = mc_return_mapping(
            sigma_trial_3, E_ur, nu, c, phi_deg, psi_deg)
        state_new = {
            'gamma_p_s': gamma_p_s_new,
            'sigma_prev': sigma_new.copy(),
            'loading': is_loading,
        }
        return sigma_new, D_ep, True, state_new

    # Elastic / sub-yield
    state_new = {
        'gamma_p_s': gamma_p_s_new,
        'sigma_prev': sigma_trial_3.copy(),
        'loading': is_loading,
    }
    return sigma_trial_3.copy(), D_tang, False, state_new


def _build_D_planestrain(E, nu):
    """Build plane-strain D-matrix (3x3) from E and nu."""
    if E < 1e-10:
        E = 1e-10
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ])


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
