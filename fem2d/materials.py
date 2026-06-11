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


# ---------------------------------------------------------------------------
# 3D principal-stress Mohr-Coulomb return mapping (vectorized, plane strain)
# ---------------------------------------------------------------------------
# Theory: de Souza Neto, Peric & Owen (2008) ch. 8; Clausen, Damkilde &
# Andersen (2006/07). Tension-positive. The full 4-component plane-strain
# stress [sxx, syy, szz, txy] is decomposed into three principal stresses
# (two in-plane eigenvalues + szz), sorted s1 >= s2 >= s3, returned to the
# MC surface in principal-stress space (main plane, extension/compression
# edge via Koiter, or apex), then rebuilt in the xy frame using the
# unchanged trial principal directions.

def _mc_edge_coeffs(lam, G, sphi, spsi):
    """Diagonal and off-diagonal n.D.m coefficients for MC plane/edges."""
    # d_aa = n_a . D_p . m_a  (same for any single MC plane)
    d_diag = 4.0 * (lam * sphi * spsi + G * (1.0 + sphi * spsi))
    # extension edge (planes share s3): n_a . D_p . m_b
    d_ext = 4.0 * lam * sphi * spsi + 2.0 * G * (1.0 - sphi) * (1.0 - spsi)
    # compression edge (planes share s1): n_a . D_p . m_b
    d_comp = 4.0 * lam * sphi * spsi + 2.0 * G * (1.0 + sphi) * (1.0 + spsi)
    return d_diag, d_ext, d_comp


def mc_return_principal(sig4, E, nu, c, phi_deg, psi_deg=0.0, want_tangent=True):
    """Vectorized 3D principal-stress Mohr-Coulomb return mapping.

    Parameters
    ----------
    sig4 : (N, 4) array — trial stresses [sxx, syy, szz, txy],
        tension-positive.
    E, nu, c, phi_deg, psi_deg : float or (N,) arrays — material params.
    want_tangent : bool — if True, also build the in-plane continuum
        elastoplastic tangent (3x3 per point).

    Returns
    -------
    sig4_new : (N, 4) array — returned stresses.
    Dep : (N, 3, 3) array or None — in-plane tangent mapping
        [d_exx, d_eyy, d_gxy] -> [d_sxx, d_syy, d_txy] (eps_zz = 0).
        Continuum tangent with fixed-principal-direction rotation
        (documented simplification of the Clausen consistent tangent).
    yielded : (N,) bool array.
    region : (N,) int array — 0 elastic, 1 main plane, 2 extension edge,
        3 compression edge, 4 apex.
    """
    sig4 = np.atleast_2d(np.asarray(sig4, dtype=float))
    N = len(sig4)
    E = np.broadcast_to(np.asarray(E, dtype=float), (N,))
    nu = np.broadcast_to(np.asarray(nu, dtype=float), (N,))
    c = np.broadcast_to(np.asarray(c, dtype=float), (N,))
    phi = np.radians(np.broadcast_to(np.asarray(phi_deg, dtype=float), (N,)))
    psi = np.radians(np.broadcast_to(np.asarray(psi_deg, dtype=float), (N,)))
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    spsi = np.sin(psi)

    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    sxx, syy, szz, txy = sig4[:, 0], sig4[:, 1], sig4[:, 2], sig4[:, 3]

    # --- in-plane principal decomposition ---
    p_in = 0.5 * (sxx + syy)
    dd = 0.5 * (sxx - syy)
    R = np.sqrt(dd ** 2 + txy ** 2)
    safe = R > 1e-12
    c2t = np.where(safe, np.where(safe, dd, 1.0) / np.where(safe, R, 1.0), 1.0)
    s2t = np.where(safe, txy / np.where(safe, R, 1.0), 0.0)
    sa = p_in + R   # in-plane major
    sb = p_in - R   # in-plane minor

    # --- sort the three principal values descending ---
    s3v = np.column_stack([sa, sb, szz])         # (N, 3)
    order = np.argsort(-s3v, axis=1, kind='stable')  # indices into [sa,sb,szz]
    s_sorted = np.take_along_axis(s3v, order, axis=1)
    s1, s2, s3 = s_sorted[:, 0], s_sorted[:, 1], s_sorted[:, 2]

    # --- yield check ---
    f_tr = (s1 - s3) + (s1 + s3) * sphi - 2.0 * c * cphi
    yielded = f_tr > 1e-10
    region = np.zeros(N, dtype=int)

    s_new_sorted = s_sorted.copy()

    if np.any(yielded):
        idx = np.where(yielded)[0]
        lam_y, G_y = lam[idx], G[idx]
        sphi_y, spsi_y = sphi[idx], spsi[idx]
        cphi_y, c_y = cphi[idx], c[idx]
        f_y = f_tr[idx]
        s_tr = s_sorted[idx]

        d_diag, d_ext, d_comp = _mc_edge_coeffs(lam_y, G_y, sphi_y, spsi_y)

        # D_p @ m vectors (m = flow direction of each plane)
        # plane a: f(s1, s3); m_a = [1+spsi, 0, -(1-spsi)]
        Dm_a = np.column_stack([
            2.0 * lam_y * spsi_y + 2.0 * G_y * (1.0 + spsi_y),
            2.0 * lam_y * spsi_y,
            2.0 * lam_y * spsi_y - 2.0 * G_y * (1.0 - spsi_y),
        ])
        # plane b_ext: f(s2, s3); m = [0, 1+spsi, -(1-spsi)]
        Dm_be = np.column_stack([
            2.0 * lam_y * spsi_y,
            2.0 * lam_y * spsi_y + 2.0 * G_y * (1.0 + spsi_y),
            2.0 * lam_y * spsi_y - 2.0 * G_y * (1.0 - spsi_y),
        ])
        # plane b_comp: f(s1, s2); m = [1+spsi, -(1-spsi), 0]
        Dm_bc = np.column_stack([
            2.0 * lam_y * spsi_y + 2.0 * G_y * (1.0 + spsi_y),
            2.0 * lam_y * spsi_y - 2.0 * G_y * (1.0 - spsi_y),
            2.0 * lam_y * spsi_y,
        ])

        # --- 1) main-plane return ---
        dl = f_y / d_diag
        s_main = s_tr - dl[:, None] * Dm_a
        tol_o = 1e-9 * np.maximum(1.0, np.abs(s_tr).max(axis=1))
        ok_main = ((s_main[:, 0] >= s_main[:, 1] - tol_o) &
                   (s_main[:, 1] >= s_main[:, 2] - tol_o))

        s_ret = s_main.copy()
        reg = np.ones(len(idx), dtype=int)

        # --- 2) edge returns where main plane invalid ---
        need_edge = ~ok_main
        if np.any(need_edge):
            je = np.where(need_edge)[0]
            # extension edge if s1' < s2' (returned violates top ordering)
            is_ext = s_main[je, 0] < s_main[je, 1] - tol_o[je]

            # f values of the second plane at trial state
            s1e, s2e, s3e = s_tr[je, 0], s_tr[je, 1], s_tr[je, 2]
            f_a = f_y[je]
            f_b_ext = (s2e - s3e) + (s2e + s3e) * sphi_y[je] \
                - 2.0 * c_y[je] * cphi_y[je]
            f_b_comp = (s1e - s2e) + (s1e + s2e) * sphi_y[je] \
                - 2.0 * c_y[je] * cphi_y[je]
            f_b = np.where(is_ext, f_b_ext, f_b_comp)

            dgg = d_diag[je]
            off = np.where(is_ext, d_ext[je], d_comp[je])
            det = dgg ** 2 - off ** 2
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            dl_a = (dgg * f_a - off * f_b) / det
            dl_b = (dgg * f_b - off * f_a) / det

            Dm_b = np.where(is_ext[:, None], Dm_be[je], Dm_bc[je])
            s_edge = s_tr[je] - dl_a[:, None] * Dm_a[je] \
                - dl_b[:, None] * Dm_b

            ok_edge = ((dl_a >= -1e-12) & (dl_b >= -1e-12) &
                       (s_edge[:, 0] >= s_edge[:, 1] - tol_o[je]) &
                       (s_edge[:, 1] >= s_edge[:, 2] - tol_o[je]))

            s_ret[je] = np.where(ok_edge[:, None], s_edge, s_ret[je])
            reg[je] = np.where(ok_edge, np.where(is_ext, 2, 3), reg[je])

            # --- 3) apex where edge invalid ---
            need_apex = ~ok_edge
            if np.any(need_apex):
                ja = je[need_apex]
                sphi_a = np.maximum(sphi_y[ja], 1e-10)
                apex = c_y[ja] * cphi_y[ja] / sphi_a
                s_ret[ja] = apex[:, None]
                reg[ja] = 4

        s_new_sorted[idx] = s_ret
        region[idx] = reg

    # --- unsort back to [sa', sb', szz'] positions ---
    s_unsorted = np.empty_like(s_new_sorted)
    np.put_along_axis(s_unsorted, order, s_new_sorted, axis=1)
    sa_n, sb_n, szz_n = s_unsorted[:, 0], s_unsorted[:, 1], s_unsorted[:, 2]

    # --- rebuild xy stresses from unchanged principal directions ---
    p_n = 0.5 * (sa_n + sb_n)
    r_n = 0.5 * (sa_n - sb_n)
    sig4_new = np.column_stack([
        p_n + r_n * c2t,
        p_n - r_n * c2t,
        szz_n,
        r_n * s2t,
    ])

    Dep = None
    if want_tangent:
        Dep = _mc_tangent_inplane(
            N, lam, G, sphi, spsi, region, order, c2t, s2t,
            d_diag_all=_mc_edge_coeffs(lam, G, sphi, spsi))

    return sig4_new, Dep, yielded, region


def _mc_tangent_inplane(N, lam, G, sphi, spsi, region, order, c2t, s2t,
                        d_diag_all):
    """Continuum elastoplastic tangent, rotated to the xy frame.

    Built in the principal frame (axes: in-plane major a, in-plane minor b,
    z), with the elastic shear modulus retained for the in-plane shear
    component (fixed-principal-direction simplification), then rotated by
    the principal angle and condensed to the plane-strain 3x3 (row/col of
    eps_zz removed since d_eps_zz = 0).
    """
    d_diag, d_ext, d_comp = d_diag_all

    # Elastic principal-space matrix entries
    # D_p = lam * ones + 2G * I  (3x3 normal components)
    Dp = (lam[:, None, None] * np.ones((1, 3, 3)) +
          2.0 * G[:, None, None] * np.eye(3)[None])

    Dep_p = Dp.copy()  # (N,3,3) in SORTED principal coords

    plastic = region > 0
    if np.any(plastic):
        # n / m vectors per region in sorted coords
        def _nm(sv, idx0, idx2):
            v = np.zeros((len(sv), 3))
            v[:, idx0] = 1.0 + sv
            v[:, idx2] = -(1.0 - sv)
            return v

        for r in (1, 2, 3, 4):
            jr = np.where(region == r)[0]
            if len(jr) == 0:
                continue
            if r == 4:
                # apex: near-zero stiffness (keep small fraction for SPD)
                Dep_p[jr] = 1e-6 * Dp[jr]
                continue
            n_a = _nm(sphi[jr], 0, 2)
            m_a = _nm(spsi[jr], 0, 2)
            Dm_a = np.einsum('nij,nj->ni', Dp[jr], m_a)
            Dn_a = np.einsum('nij,nj->ni', Dp[jr], n_a)
            if r == 1:
                Dep_p[jr] = Dp[jr] - \
                    np.einsum('ni,nj->nij', Dm_a, Dn_a) / d_diag[jr][:, None, None]
            else:
                if r == 2:   # extension edge: second plane f(s2, s3)
                    n_b = _nm(sphi[jr], 1, 2)
                    m_b = _nm(spsi[jr], 1, 2)
                    off = d_ext[jr]
                else:        # compression edge: second plane f(s1, s2)
                    n_b = _nm(sphi[jr], 0, 1)
                    m_b = _nm(spsi[jr], 0, 1)
                    off = d_comp[jr]
                Dm_b = np.einsum('nij,nj->ni', Dp[jr], m_b)
                Dn_b = np.einsum('nij,nj->ni', Dp[jr], n_b)
                det = d_diag[jr] ** 2 - off ** 2
                det = np.where(np.abs(det) < 1e-30, 1e-30, det)
                ia = d_diag[jr] / det
                ib = -off / det
                # Dep = Dp - [Dm_a Dm_b] inv(A) [Dn_a Dn_b]^T
                Dep_p[jr] = Dp[jr] - (
                    ia[:, None, None] * np.einsum('ni,nj->nij', Dm_a, Dn_a) +
                    ib[:, None, None] * np.einsum('ni,nj->nij', Dm_a, Dn_b) +
                    ib[:, None, None] * np.einsum('ni,nj->nij', Dm_b, Dn_a) +
                    ia[:, None, None] * np.einsum('ni,nj->nij', Dm_b, Dn_b))

    # --- unsort rows/cols from sorted coords back to (a, b, z) ---
    # order maps sorted position -> original axis index; build permutation
    inv = np.empty_like(order)
    np.put_along_axis(inv, order, np.arange(3)[None].repeat(N, 0), axis=1)
    # Dep_abz[i,j] = Dep_p[inv[i], inv[j]]
    Dep_abz = np.take_along_axis(
        np.take_along_axis(Dep_p, inv[:, :, None], axis=1),
        inv[:, None, :], axis=2)

    # --- build 4x4 in principal frame (a, b, z, gamma_ab) and rotate ---
    D4p = np.zeros((N, 4, 4))
    D4p[:, :3, :3] = Dep_abz
    D4p[:, 3, 3] = G  # elastic shear stiffness (fixed-direction approx)

    # Strain rotation matrix T_e: eps_principal = T_e @ eps_xy
    # (eps vector [exx, eyy, ezz, gxy]; angle theta from xy to principal)
    cc = 0.5 * (1.0 + c2t)   # cos^2
    ss = 0.5 * (1.0 - c2t)   # sin^2
    sc = 0.5 * s2t           # sin*cos
    T = np.zeros((N, 4, 4))
    T[:, 0, 0] = cc
    T[:, 0, 1] = ss
    T[:, 0, 3] = sc
    T[:, 1, 0] = ss
    T[:, 1, 1] = cc
    T[:, 1, 3] = -sc
    T[:, 2, 2] = 1.0
    T[:, 3, 0] = -2.0 * sc
    T[:, 3, 1] = 2.0 * sc
    T[:, 3, 3] = cc - ss

    # sigma_xy = T^T sigma_principal (contragredience) -> D_xy = T^T D4p T
    D4 = np.einsum('nji,njk,nkl->nil', T, D4p, T)

    # Condense to in-plane 3x3: drop eps_zz row/col (d eps_zz = 0)
    keep = [0, 1, 3]
    return D4[:, keep][:, :, keep]
