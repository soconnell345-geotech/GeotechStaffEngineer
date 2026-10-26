"""
Modulus-reduction and damping curves for the site-response materials
(GM-80), and the fit of the OpenSees multi-yield backbone to one of them.

The multi-yield materials (PressureDependMultiYield02 for sand,
PressureIndependMultiYield for clay) build their yield surfaces either on
their own hyperbolic backbone (from Gmax, the strength and a peak shear
strain of 10 %) or on a user-defined backbone: a negative noYieldSurf
followed by pairs of shear strain and secant G/Gmax at the material's
reference pressure. With user-defined pairs the material takes its
strength from the LAST pair (OpenSees source,
PressureDependMultiYield02::setUpSurfaces and
PressureIndependMultiYield::setUpSurfaces: tau_max = Gref * G/Gmax *
gamma; sand M = sqrt(3) tau_max / p_ref and sin(phi) = 3M / (6 + M); clay
c = sqrt(3)/2 tau_max), so :func:`fit_backbone` follows the chosen curve
up to the layer's own strength and stops there: the layer keeps the
friction angle or su it was given.

Curves offered:
  - ``opensees``: the material's own hyperbolic backbone (the default,
    unchanged).
  - ``darendeli``: Darendeli (2001), modified hyperbolic with a reference
    strain set by PI, OCR and the mean effective stress.
  - ``menq``: Menq (2003), modified hyperbolic for sands and gravels with
    a reference strain set by Cu and the mean effective stress.
  - ``user``: a table of G/Gmax against shear strain (and, optionally,
    damping against strain).

The two published curves are cited (GEC 5, FHWA-NHI-16-072, Sec. 8.8,
p. 8-23; FEMA P-2082, Part 2, Ch. 21 commentary, p. 515) but their
equations are not printed in the reference library; the constants below
are those of the pystrata implementation (Kottke), which
``pystrata_agent`` already runs, and the tests check them against it.

The multi-yield materials damp by hysteresis on Masing's rules: they cannot
take a damping curve. A curve's damping is therefore reported beside the
Masing damping of the fitted backbone (:func:`masing_damping`), and the
small-strain damping is the Rayleigh term (``damping``).

References:
    Darendeli, M.B. (2001). Development of a New Family of Normalized
    Modulus Reduction and Material Damping Curves. Ph.D. Dissertation,
    University of Texas at Austin.

    Menq, F.Y. (2003). Dynamic Properties of Sandy and Gravelly Soils.
    Ph.D. Dissertation, University of Texas at Austin.
"""

import math

import numpy as np

from geotech_common.numpy_compat import trapezoid

#: Curve names a layer's ``curve`` may take.
CURVES = ("opensees", "darendeli", "menq", "user")

P_ATM_KPA = 101.325  # pystrata's KPA_TO_ATM = kilo / atm

# Strains (decimal) at which a published or tabled curve is sampled for the
# backbone: 1e-6 to 10 %, the module's peak shear strain.
_FIT_STRAINS = np.logspace(-6.0, -1.0, 31)


def _masing_scaling(n_cycles):
    """Darendeli's scaling of the Masing damping (b), as in pystrata."""
    return 0.6329 - 0.00566 * math.log(n_cycles)


def darendeli_params(sigma_m_kPa, plas_index=0.0, ocr=1.0, freq_hz=1.0,
                     n_cycles=10.0):
    """Reference strain (decimal), curvature, minimum damping (decimal)
    and Masing scaling of Darendeli (2001)."""
    s = sigma_m_kPa / P_ATM_KPA
    gamma_r = ((0.0352 + 0.0010 * plas_index * ocr ** 0.3246)
               * s ** 0.3483) / 100.0
    d_min = ((0.8005 + 0.0129 * plas_index * ocr ** -0.1069)
             * s ** -0.2889 * (1.0 + 0.2919 * math.log(freq_hz))) / 100.0
    return gamma_r, 0.9190, d_min, _masing_scaling(n_cycles)


def menq_params(sigma_m_kPa, uniformity_coeff, diam_mean, n_cycles=10.0):
    """Reference strain (decimal), curvature, minimum damping (decimal)
    and Masing scaling of Menq (2003); ``diam_mean`` is D50 in mm."""
    s = sigma_m_kPa / P_ATM_KPA
    cu = uniformity_coeff
    gamma_r = (0.12 * cu ** -0.6 * s ** (0.5 * cu ** -0.15)) / 100.0
    d_min = (0.55 * cu ** 0.1 * diam_mean ** -0.3 * s ** -0.08) / 100.0
    return gamma_r, 0.86 + 0.1 * math.log10(s), d_min, \
        _masing_scaling(n_cycles)


def modified_hyperbolic(strains, gamma_r, curvature, d_min, b):
    """G/Gmax and damping (both decimal) of a modified hyperbolic curve
    at ``strains`` (decimal): G/Gmax = 1 / (1 + (gamma/gamma_r)^a), and
    damping = b (G/Gmax)^0.1 D_Masing + D_min, the Masing term corrected
    from the hyperbolic to the modified hyperbolic shape (as pystrata)."""
    g = np.asarray(strains, dtype=float)
    ggmax = 1.0 / (1.0 + (g / gamma_r) ** curvature)
    gp, rp = g * 100.0, gamma_r * 100.0
    d_a1 = (100.0 / math.pi) * (
        4.0 * (gp - rp * np.log((gp + rp) / rp)) / (gp ** 2 / (gp + rp))
        - 2.0)
    a = curvature
    c1 = -1.1143 * a ** 2 + 1.8618 * a + 0.2523
    c2 = 0.0805 * a ** 2 - 0.0710 * a - 0.0095
    c3 = -0.0005 * a ** 2 + 0.0002 * a + 0.0003
    d_masing = c1 * d_a1 + c2 * d_a1 ** 2 + c3 * d_a1 ** 3
    damping = np.maximum.accumulate(b * d_masing * ggmax ** 0.1 / 100.0)
    return ggmax, damping + d_min


def table(curve, value_key, layer_label):
    """(strains decimal, values) of a user table
    ``{"strain_pct": [...], value_key: [...]}``, checked."""
    if not isinstance(curve, dict):
        raise ValueError(
            f"{layer_label}: a curve table is a dict with 'strain_pct' and "
            f"'{value_key}' lists, got {type(curve).__name__}")
    for key in ("strain_pct", value_key):
        if key not in curve:
            raise ValueError(f"{layer_label}: the curve table needs "
                             f"'{key}' (have {sorted(curve)})")
    s = np.asarray(curve["strain_pct"], dtype=float)
    v = np.asarray(curve[value_key], dtype=float)
    if s.ndim != 1 or s.shape != v.shape or len(s) < 2:
        raise ValueError(
            f"{layer_label}: 'strain_pct' and '{value_key}' must be lists "
            f"of the same length, at least 2")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(v))):
        raise ValueError(f"{layer_label}: the curve table holds a value "
                         f"that is not a number")
    if np.any(s <= 0) or np.any(np.diff(s) <= 0):
        raise ValueError(f"{layer_label}: 'strain_pct' must be positive "
                         f"and rise from one point to the next")
    return s / 100.0, v


def table_ggmax(curve, layer_label):
    """G/Gmax of a user table at the fit strains: interpolated in log
    strain, held at the first value below the table and extended beyond
    it at the last point's shear stress (perfectly plastic)."""
    s, v = table(curve, "G_Gmax", layer_label)
    if np.any(v <= 0) or np.any(v > 1.0 + 1e-9):
        raise ValueError(f"{layer_label}: 'G_Gmax' must be in (0, 1]")
    if np.any(np.diff(v) > 1e-12):
        raise ValueError(f"{layer_label}: 'G_Gmax' must not rise with "
                         f"strain")
    if np.any(np.diff(s * v) <= 0):
        raise ValueError(
            f"{layer_label}: the shear stress G/Gmax x strain of the table "
            f"must rise from one point to the next (a falling stress is a "
            f"softening backbone, which the multi-yield materials refuse)")
    return s, v


def ggmax_at(strains, curve_strains, curve_values):
    """Interpolate G/Gmax in log strain; below the table the first value,
    beyond it the last point's stress held (tau = const)."""
    g = np.asarray(strains, dtype=float)
    out = np.interp(np.log(g), np.log(curve_strains), curve_values)
    beyond = g > curve_strains[-1]
    out[beyond] = curve_values[-1] * curve_strains[-1] / g[beyond]
    return out


def fit_backbone(ggmax_fn, g_max, tau_f):
    """User-defined backbone pairs for a multi-yield material.

    ``ggmax_fn(strains) -> G/Gmax``; ``g_max`` the reference shear modulus
    (kPa); ``tau_f`` the layer's shear strength at the reference pressure
    (kPa). The curve is followed at 31 strains from 1e-6 to 10 % until its
    shear stress reaches ``tau_f``; the last pair is put exactly at
    ``tau_f``, so the material keeps the layer's strength. If the curve is
    still below ``tau_f`` at 10 %, the last pair is ``tau_f`` at 10 %.

    Returns ``(pairs, info)``: ``pairs`` a flat list strain, G/Gmax, ...
    (strains decimal, stress strictly rising) and ``info`` the strain at
    which the strength was reached and how.
    """
    strains = _FIT_STRAINS
    ratios = np.asarray(ggmax_fn(strains), dtype=float)
    taus = g_max * ratios * strains
    pts = []
    reached = None
    for k, (g, r, t) in enumerate(zip(strains, ratios, taus)):
        if t >= tau_f:
            lo = strains[k - 1] if k else strains[0] * 1e-3
            hi = g
            for _ in range(80):  # bisect in log strain for tau = tau_f
                mid = math.sqrt(lo * hi)
                if g_max * float(ggmax_fn(np.array([mid]))[0]) * mid \
                        < tau_f:
                    lo = mid
                else:
                    hi = mid
            reached = hi
            break
        pts.append((float(g), float(r)))
    if reached is None:
        how = "the curve stays below the strength to 10 %: last pair " \
              "set at the strength at 10 %"
        last = float(strains[-1])
        pts = [p for p in pts if p[0] < last]
    else:
        how = "the curve reaches the strength"
        last = reached
        pts = [p for p in pts if p[0] < last * (1.0 - 1e-9)]
    pts.append((last, tau_f / (g_max * last)))
    # Stress must rise strictly from pair to pair (the material exits
    # otherwise): drop any pair whose stress does not.
    clean = []
    for g, r in pts:
        if clean and g_max * r * g <= g_max * clean[-1][1] * clean[-1][0]:
            continue
        clean.append((g, r))
    if len(clean) < 2:
        raise ValueError(
            "The curve reaches the layer's strength at a strain below "
            "1e-6: the strength or Gmax is out of range for this curve.")
    flat = []
    for g, r in clean:
        flat.extend((g, r))
    return flat, {"strength_strain_pct": round(last * 100.0, 4),
                  "n_points": len(clean), "how": how}


def masing_damping(pairs, g_max, strains):
    """Masing damping (decimal) of a piecewise-linear backbone at
    ``strains`` (decimal): D = (4 A / (tau gamma) - 2) / pi, with A the
    area under the backbone to gamma (Kramer 1996, Sec. 6.4: the loop of
    a Masing material is the backbone scaled by two)."""
    g_pts = np.concatenate(([0.0], np.asarray(pairs[0::2], dtype=float)))
    t_pts = np.concatenate(
        ([0.0], g_max * np.asarray(pairs[1::2], dtype=float)
         * np.asarray(pairs[0::2], dtype=float)))
    out = []
    for g in np.atleast_1d(strains):
        if g > g_pts[-1]:
            gg = np.append(g_pts, g)
            tt = np.append(t_pts, t_pts[-1])
        else:
            m = g_pts < g
            gg = np.append(g_pts[m], g)
            tt = np.append(t_pts[m], np.interp(g, g_pts, t_pts))
        area = trapezoid(tt, gg)
        tau = tt[-1]
        out.append(max(0.0, (4.0 * area / (tau * g) - 2.0) / math.pi))
    return np.array(out)


def sand_strength(phi_deg, p_ref):
    """Shear strength (kPa) the sand material takes at ``p_ref``: the
    inverse of the source's user-backbone relation, tau = M p / sqrt(3),
    M = 6 sin(phi) / (3 - sin(phi))."""
    s = math.sin(math.radians(phi_deg))
    return 6.0 * s / (3.0 - s) * p_ref / math.sqrt(3.0)


def hyperbolic_ggmax(g_max, tau_f, peak_strain=0.1):
    """G/Gmax of the material's own hyperbolic backbone (the ``opensees``
    curve): tau = G gamma / (1 + gamma / gamma_r) reaching tau_f at the
    peak shear strain (the source's refStrain), for reporting."""
    gamma_r = peak_strain * tau_f / (g_max * peak_strain - tau_f)
    return lambda g: 1.0 / (1.0 + np.asarray(g) / gamma_r)
