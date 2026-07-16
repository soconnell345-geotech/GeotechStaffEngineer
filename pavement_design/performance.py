"""Environmental serviceability loss and performance-period estimation
(AASHTO 1993 Section 2.1.4 / 3.1.3 / Appendix G).

Roadbed swelling and frost heave consume part of the design serviceability
budget over the analysis period; the design functions reduce the
traffic-available dPSI accordingly (Step 4 of the Table 3.1 procedure).
``estimate_performance_period`` runs the full printed Table 3.1 iteration
for a DESIGNED section: at a trial period the environmental loss shrinks
the traffic budget, Figure 3.1/3.7 in reverse gives the allowable
cumulative ESALs at the section's fixed structural capacity, and the
traffic-growth curve converts that back to a calendar period, iterated to
convergence.

All physics from ``geotech_references.aashto_1993.environmental``.
UNITS: US customary; swelling spec {vr_in, ps_pct, theta}, frost spec
{phi_mm_day, pf_pct, delta_psi_max}.
"""

import math

from geotech_references.aashto_1993 import equations as _eq

from .common import add_ref

try:  # Appendix G environmental-loss module (refs follow-on)
    from geotech_references.aashto_1993 import environmental as _env_mod
except ImportError:  # pragma: no cover - depends on refs version
    _env_mod = None


def _require_env_module():
    if _env_mod is None:
        raise NotImplementedError(
            "Swelling/frost-heave serviceability loss requires "
            "geotech_references.aashto_1993.environmental, which is not "
            "available in this install."
        )


def resolve_environmental_loss(design_period_yr, swelling, frost,
                               references, notes):
    """Total environmental dPSI at the analysis period; None if no spec.

    ``swelling`` = {vr_in, ps_pct, theta} (Figure G.4);
    ``frost`` = {phi_mm_day, pf_pct, delta_psi_max} (Figure G.8).
    Returns the reference-module result dict (delta_psi_sw / delta_psi_fh /
    delta_psi_total) augmented with the design period used.
    """
    if not swelling and not frost:
        return None
    _require_env_module()
    if design_period_yr is None or design_period_yr <= 0:
        raise ValueError(
            "design_period_yr (> 0, the analysis period in years) is "
            "required when swelling/frost specs are given -- environmental "
            "serviceability loss accumulates over time."
        )
    r = _env_mod.total_environmental_loss(design_period_yr,
                                          swelling=swelling, frost=frost)
    add_ref(references, r.get("reference"))
    # Keep the input specs in the block so downstream consumers (plots,
    # performance iteration) can re-evaluate the loss-vs-time curve.
    r = dict(r, swelling_spec=dict(swelling) if swelling else None,
             frost_spec=dict(frost) if frost else None)
    notes.append(
        f"Environmental serviceability loss at {design_period_yr} yr: "
        f"dPSI_sw = {r.get('delta_psi_sw', 0)}, dPSI_fh = "
        f"{r.get('delta_psi_fh', 0)}, total = {r['delta_psi_total']} "
        "(subtracted from the design dPSI before the traffic solve, "
        "Table 3.1 Step 4)."
    )
    return dict(r, design_period_yr=design_period_yr)


def _time_from_w18(w18, base_year_w18, growth_rate_pct):
    """Years to accumulate a cumulative W18 under compound growth
    (inverse of the growth-factor relation; g = 0 -> W18/base)."""
    g = growth_rate_pct / 100.0
    if w18 <= 0:
        return 0.0
    if g == 0:
        return w18 / base_year_w18
    return math.log(1.0 + w18 * g / base_year_w18) / math.log(1.0 + g)


def estimate_performance_period(
    pavement_type,
    delta_psi_design,
    base_year_w18,
    growth_rate_pct=0.0,
    swelling=None,
    frost=None,
    max_performance_period_yr=20.0,
    initial_trial_yr=None,
    # flexible section capacity (Figure 3.1 held at the as-built SN):
    sn=None,
    mr_psi=None,
    # rigid section capacity (Figure 3.7 held at the as-built D):
    d_in=None,
    sc_psi=None,
    ec_psi=None,
    j=None,
    cd=None,
    k_pci=None,
    pt=2.5,
    # reliability (both):
    zr=None,
    so=None,
) -> dict:
    """Predicted performance period of a designed section under
    swelling/frost heave (the printed Table 3.1 iteration).

    Parameters mirror the design functions but hold the STRUCTURE fixed
    (``sn`` + ``mr_psi`` for flexible; ``d_in`` + rigid inputs for rigid)
    and describe traffic by its first-year design-lane ESALs
    (``base_year_w18``) plus compound growth, so allowable cumulative
    traffic converts to calendar years in closed form.

    Returns the reference worksheet dict (per-iteration rows, converged,
    performance_period_yr) plus the inputs echoed.
    """
    _require_env_module()
    if not swelling and not frost:
        raise ValueError(
            "Provide a swelling and/or frost spec -- without environmental "
            "loss the performance period is set by traffic alone."
        )
    if base_year_w18 is None or base_year_w18 <= 0:
        raise ValueError(f"base_year_w18 must be > 0, got {base_year_w18}")
    if zr is None or so is None:
        raise ValueError(
            "zr and so are required (use the same reliability inputs as "
            "the design; Table 3.1 Step 5 note)."
        )

    if pavement_type == "flexible":
        if sn is None or mr_psi is None:
            raise ValueError(
                "Flexible performance period needs sn (as-built) and mr_psi."
            )

        def w18_fn(dpsi_traffic):
            return _eq.flexible_w18_from_sn(sn, zr, so, dpsi_traffic,
                                            mr_psi)["w18"]
    elif pavement_type == "rigid":
        missing = [n for n, v in (("d_in", d_in), ("sc_psi", sc_psi),
                                  ("ec_psi", ec_psi), ("j", j), ("cd", cd),
                                  ("k_pci", k_pci)) if v is None]
        if missing:
            raise ValueError(
                f"Rigid performance period needs {missing} (the as-built "
                "slab and the design k/J/Cd)."
            )

        def w18_fn(dpsi_traffic):
            return _eq.rigid_w18_from_d(d_in, zr, so, dpsi_traffic, sc_psi,
                                        cd, j, ec_psi, k_pci, pt=pt)["w18"]
    else:
        raise ValueError(
            f"pavement_type must be 'flexible' or 'rigid', got "
            f"'{pavement_type}'"
        )

    r = _env_mod.performance_period_iteration(
        delta_psi_design,
        w18_fn,
        lambda w18: _time_from_w18(w18, base_year_w18, growth_rate_pct),
        initial_trial_yr=initial_trial_yr,
        max_performance_period_yr=max_performance_period_yr,
        swelling=swelling,
        frost=frost,
    )
    out = dict(r)
    out.update({
        "pavement_type": pavement_type,
        "base_year_w18": base_year_w18,
        "growth_rate_pct": growth_rate_pct,
    })
    return out
