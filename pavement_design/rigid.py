"""AASHTO 1993 rigid pavement design (Part II, Ch 3; Figure 3.7).

Orchestrates the digitized guide equations into a complete rigid (PCC slab)
design: reliability (ZR/So) -> serviceability loss -> effective modulus of
subgrade reaction k (direct value, the simplified MR/19.4 relation, or the
full Section 3.2 composite-k worksheet) -> required slab thickness D
(Figure 3.7 solve) -> practical rounding -> forward adequacy check. When
the composite-k procedure is used, the seasonal relative-damage step
depends on the slab thickness itself, so k and D are iterated to
convergence.

UNITS: US customary (psi, pci, inches, ESALs) -- documented exception to
the repo SI rule; see DESIGN.md.
"""

from geotech_references.aashto_1993 import equations as _eq
from geotech_references.aashto_1993 import tables as _tb

from .common import (add_ref, midpoint_range, resolve_delta_psi,
                     resolve_reliability, resolve_so, round_up)
from .results import RigidPavementResult

try:  # Section 3.2 composite-k worksheet (built as a follow-on module)
    from geotech_references.aashto_1993 import composite_k as _ck_mod
except ImportError:  # pragma: no cover - depends on refs version
    _ck_mod = None


def _resolve_j(j, pavement_type, shoulder_type, load_transfer_devices,
               references, notes):
    if j is not None:
        if j <= 0:
            raise ValueError(f"j must be > 0, got {j}")
        return float(j), "user-specified"
    r = _tb.load_transfer_coefficient_j(pavement_type, shoulder_type,
                                        load_transfer_devices)
    add_ref(references, r["reference"])
    val = midpoint_range(r["j_min"], r["j_max"])
    notes.append(
        f"J = {val} (midpoint of Table 2.6 range {r['j_min']}-{r['j_max']} "
        f"for {r['pavement_type']}, {r['shoulder_type']} shoulder, "
        f"dowels={r['load_transfer_devices']})."
    )
    return val, (f"Table 2.6 midpoint ({r['j_min']}-{r['j_max']})")


def _resolve_cd(cd, drainage_quality, pct_saturation_time, references, notes):
    if cd is not None:
        if cd <= 0:
            raise ValueError(f"cd must be > 0, got {cd}")
        return float(cd), "user-specified"
    if drainage_quality is not None:
        pct = pct_saturation_time or "1-5%"
        if pct_saturation_time is None:
            notes.append(
                "pct_saturation_time defaulted to '1-5%' for the Cd lookup."
            )
        r = _tb.drainage_cd_rigid(drainage_quality, pct)
        add_ref(references, r["reference"])
        val = midpoint_range(r["cd_min"], r["cd_max"])
        notes.append(
            f"Cd = {val} (midpoint of Table 2.5 range "
            f"{r['cd_min']}-{r['cd_max']}, quality={r['quality']}, {pct})."
        )
        return val, f"Table 2.5 midpoint ({r['cd_min']}-{r['cd_max']})"
    notes.append(
        "Cd defaulted to 1.0 (AASHO Road Test drainage conditions, "
        "Table 2.5 'fair')."
    )
    return 1.0, "default 1.0 (AASHO Road Test conditions)"


def _resolve_k(k_pci, mr_psi, composite_k, slab_d_in, references, notes):
    """Effective k for one trial slab thickness; returns (k, basis dict)."""
    supplied = [x is not None for x in (k_pci, mr_psi, composite_k)]
    if sum(supplied) != 1:
        raise ValueError(
            "Provide exactly one of k_pci (direct), mr_psi (simplified "
            "k = MR/19.4, slab on roadbed with no subbase), or composite_k "
            "(the Section 3.2 worksheet spec)."
        )
    if k_pci is not None:
        if k_pci <= 0:
            raise ValueError(f"k_pci must be > 0, got {k_pci}")
        return float(k_pci), {"basis": "direct", "k_pci": float(k_pci)}
    if mr_psi is not None:
        r = _eq.modulus_subgrade_reaction_simple(mr_psi)
        add_ref(references, r["reference"])
        notes.append(f"k basis: {r['note']}")
        return r["k_pci"], {"basis": "simple_mr_over_19.4",
                            "mr_psi": mr_psi, "k_pci": r["k_pci"]}
    if _ck_mod is None:
        raise NotImplementedError(
            "The composite-k procedure requires "
            "geotech_references.aashto_1993.composite_k, which is not "
            "available in this install; give k_pci directly or use mr_psi "
            "for the simplified no-subbase relation."
        )
    spec = dict(composite_k)
    spec["slab_d_in"] = slab_d_in
    r = _ck_mod.effective_modulus_subgrade_reaction(**spec)
    add_ref(references, r.get("reference"))
    basis = {"basis": "composite_section_3.2", "slab_d_in": slab_d_in}
    basis.update({key: val for key, val in r.items()
                  if key not in ("equation",)})
    # The DESIGN k is the loss-of-support-corrected value (Figure 3.6,
    # the worksheet's final step); fall back to the uncorrected effective
    # k when no LS correction applies (ls=0).
    k_design = r.get("k_corrected_for_loss_of_support_pci")
    if k_design is None:
        k_design = r["effective_k_pci"]
    basis["design_k_pci"] = k_design
    return k_design, basis


def design_rigid_pavement(
    w18,
    sc_psi,
    ec_psi,
    reliability_pct=None,
    zr=None,
    so=None,
    pt=2.5,
    po=None,
    delta_psi=None,
    j=None,
    pavement_type="plain_jointed_jrcp",
    shoulder_type="asphalt",
    load_transfer_devices=True,
    cd=None,
    drainage_quality=None,
    pct_saturation_time=None,
    k_pci=None,
    mr_psi=None,
    composite_k=None,
    slab_thickness_in=None,
    thickness_increment_in=0.5,
    max_k_iterations=12,
    swelling=None,
    frost=None,
    design_period_yr=None,
) -> RigidPavementResult:
    """Complete AASHTO 1993 rigid pavement design or adequacy check.

    Parameters
    ----------
    w18 : float
        Design-lane cumulative 18-kip ESALs.
    sc_psi : float
        Mean 28-day PCC modulus of rupture (third-point loading), psi.
    ec_psi : float
        PCC elastic modulus, psi.
    reliability_pct, zr, so : float, optional
        Reliability inputs; So defaults to 0.35 (midpoint of the rigid
        0.30-0.40 range).
    pt, po, delta_psi : float, optional
        Serviceability (po defaults to 4.5; pt default 2.5). pt also enters
        the Figure 3.7 strength-term exponent.
    j : float, optional
        Load transfer coefficient directly; otherwise resolved from
        Table 2.6 via pavement_type ('plain_jointed_jrcp' | 'crcp'),
        shoulder_type ('asphalt' | 'tied_pcc'), and load_transfer_devices
        (midpoint of the printed range).
    cd : float, optional
        Drainage coefficient directly; otherwise Table 2.5 via
        drainage_quality + pct_saturation_time (midpoint), else 1.0.
    k_pci, mr_psi, composite_k : optional (exactly one)
        Effective modulus of subgrade reaction: direct value; simplified
        k = MR/19.4 (slab directly on roadbed, no subbase); or the full
        Section 3.2 worksheet spec (dict of keyword arguments for
        ``geotech_references.aashto_1993.composite_k.
        effective_modulus_subgrade_reaction``, e.g. seasonal entries,
        subbase thickness/modulus, depth to rigid foundation, loss of
        support -- ``slab_d_in`` is injected and iterated by this
        function).
    slab_thickness_in : float, optional
        Give a slab thickness to run a CHECK instead of a design.
    thickness_increment_in : float, optional
        Practical rounding increment for the designed slab (default 0.5 in,
        rounded UP).
    max_k_iterations : int, optional
        Cap on the composite-k <-> D fixed-point iteration.
    swelling, frost : dict, optional
        Roadbed swelling {vr_in, ps_pct, theta} (Figure G.4) and/or frost
        heave {phi_mm_day, pf_pct, delta_psi_max} (Figure G.8) specs; the
        environmental loss at ``design_period_yr`` is subtracted from the
        design dPSI before the slab solve (Table 3.1 Step 4).
    design_period_yr : float, optional
        Analysis period, years -- required when swelling/frost are given.

    Returns
    -------
    RigidPavementResult
    """
    if w18 is None or w18 <= 0:
        raise ValueError(f"w18 must be > 0, got {w18}")

    references = []
    notes = []
    warnings = []

    reliability_pct, zr_val, ref, n = resolve_reliability(reliability_pct, zr)
    add_ref(references, ref)
    notes.extend(n)
    so_val, ref, n = resolve_so("rigid", so)
    add_ref(references, ref)
    notes.extend(n)
    dpsi, po_val, pt_val, ref, n = resolve_delta_psi("rigid", delta_psi, po, pt)
    add_ref(references, ref)
    notes.extend(n)
    from .performance import resolve_environmental_loss
    environmental = resolve_environmental_loss(design_period_yr, swelling,
                                               frost, references, notes)
    if environmental is not None:
        dpsi_env = environmental["delta_psi_total"]
        if dpsi_env >= dpsi:
            raise ValueError(
                f"Environmental serviceability loss ({dpsi_env}) consumes "
                f"the entire design dPSI ({dpsi}) at {design_period_yr} yr "
                "-- shorten the analysis period, mitigate the roadbed "
                "(see the guide's Appendix G options), or raise po/lower pt."
            )
        if dpsi_env > 0.5 * dpsi:
            warnings.append(
                f"Environmental loss ({dpsi_env}) exceeds half the design "
                f"dPSI ({dpsi}); the guide recommends considering roadbed "
                "mitigation before designing around swelling/frost of this "
                "magnitude."
            )
        dpsi_traffic = round(dpsi - dpsi_env, 4)
        environmental = dict(environmental, delta_psi_design=dpsi,
                             delta_psi_traffic=dpsi_traffic)
        dpsi = dpsi_traffic
    j_val, _ = _resolve_j(j, pavement_type, shoulder_type,
                          load_transfer_devices, references, notes)
    cd_val, _ = _resolve_cd(cd, drainage_quality, pct_saturation_time,
                            references, notes)

    mode = "check" if slab_thickness_in is not None else "design"
    # The composite-k relative-damage step depends on D, so iterate a
    # fixed point: k(D) -> D(k) until the designed D stabilizes. For
    # direct/simple k the first pass converges immediately.
    d_trial = (float(slab_thickness_in) if mode == "check" else 9.0)
    iterations = 0
    k_val = None
    k_basis = None
    d_required = None
    for iterations in range(1, max_k_iterations + 1):
        k_val, k_basis = _resolve_k(k_pci, mr_psi, composite_k, d_trial,
                                    references, notes)
        if mode == "check":
            d_required_r = _eq.rigid_d_from_w18(
                w18, zr_val, so_val, dpsi, sc_psi, cd_val, j_val, ec_psi,
                k_val, pt=pt_val)
            add_ref(references, d_required_r["reference"])
            d_required = d_required_r["d"]
            break
        d_required_r = _eq.rigid_d_from_w18(
            w18, zr_val, so_val, dpsi, sc_psi, cd_val, j_val, ec_psi, k_val,
            pt=pt_val)
        add_ref(references, d_required_r["reference"])
        d_required = d_required_r["d"]
        d_new = max(round_up(d_required, thickness_increment_in), 4.0)
        if composite_k is None or abs(d_new - d_trial) < 0.25:
            d_trial = d_new
            break
        d_trial = d_new
    else:
        warnings.append(
            f"Composite-k <-> D iteration did not settle within "
            f"{max_k_iterations} passes; last D = {d_trial} in used."
        )

    if mode == "check":
        d_provided = float(slab_thickness_in)
    else:
        d_provided = d_trial
        if composite_k is not None and iterations > 1:
            notes.append(
                f"Composite-k iterated with the slab thickness "
                f"({iterations} passes)."
            )

    fwd = _eq.rigid_w18_from_d(d_provided, zr_val, so_val, dpsi, sc_psi,
                               cd_val, j_val, ec_psi, k_val, pt=pt_val)
    add_ref(references, fwd["reference"])
    w18_capacity = fwd["w18"]
    adequate = (d_provided >= d_required - 1e-9) and (w18_capacity >= w18)

    return RigidPavementResult(
        mode=mode,
        w18=w18,
        reliability_pct=reliability_pct,
        zr=zr_val,
        so=so_val,
        po=po_val,
        pt=pt_val,
        delta_psi=dpsi,
        sc_psi=sc_psi,
        ec_psi=ec_psi,
        j=j_val,
        cd=cd_val,
        k_pci=k_val,
        k_basis=k_basis,
        d_required_in=d_required,
        d_provided_in=d_provided,
        w18_capacity=w18_capacity,
        adequate=adequate,
        environmental=environmental,
        iterations=iterations,
        notes=notes,
        warnings=warnings,
        references=references,
    )
