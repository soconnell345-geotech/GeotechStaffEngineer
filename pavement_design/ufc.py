"""UFC 3-250-01 (2016) roads/parking pavement design — the DoD alternative
to the AASHTO 1993 procedures, plus cross-guide comparison.

The UFC method characterizes traffic as equivalent passes of an 18-kip
single-axle, dual-tire load (controlling-vehicle procedure, Guide Ch 4 /
Appendix G) and designs flexible sections by the Corps CBR method (Figure
E-1 cover curves) and rigid slabs by the Westergaard-based Figure F-1
family (+ Eq 13-1 on stabilized foundations). All physics comes from
``geotech_references.ufc_pavement`` (Appendix-G-anchored digitization);
chart-read tolerances propagate to the results.

COMPARISON BASIS (``compare_flexible_pavement_methods``): both guides
accept "passes of an 18-kip single axle" as the traffic number — AASHTO's
W18 by definition (LEF = 1.0), the UFC's E-1/F-1 curves natively. The two
are the same PHYSICAL count but different DAMAGE MODELS (AASHTO
serviceability loss vs Corps CBR-beta), so identical inputs legitimately
give different sections; the comparison states every cross-guide
assumption (CBR<->Mr correlation, reliability present only in AASHTO) in
``notes`` rather than hiding them.

UNITS: US customary throughout (CBR %, psi, pci, inches, 18-kip passes).
"""

from geotech_references.fhwa_pavements import equations as _fhwa
from geotech_references.ufc_pavement import equations as _ueq
from geotech_references.ufc_pavement import tables as _utb

from .common import add_ref, round_up


def _cbr_to_mr(cbr, references, notes):
    r = _fhwa.resilient_modulus_from_cbr(cbr)
    add_ref(references, r["reference"])
    notes.append(
        f"CBR {cbr} -> Mr {r['mr']:.0f} psi via {r['equation']} "
        "(the preferred FHWA-NHI-05-037 correlation; override with a "
        "direct modulus if you have one)."
    )
    return r["mr"]


def _mr_to_cbr(mr_psi, notes):
    cbr = (mr_psi / 2555.0) ** (1.0 / 0.64)
    notes.append(
        f"Mr {mr_psi:.0f} psi -> CBR {cbr:.1f} by inverting "
        "Mr = 2555*CBR^0.64 (FHWA-NHI-05-037 Table 5-34)."
    )
    return cbr


def design_flexible_pavement_ufc(
    passes_18kip,
    cbr_subgrade,
    cbr_base=80.0,
    cbr_subbase=None,
    frost=None,
    thickness_increment_in=0.5,
) -> dict:
    """UFC 3-250-01 flexible (CBR-method) design for roads/parking.

    Layered cover logic (Guide Ch 5-9 / Figure E-1): the thickness of
    material above ANY layer must equal the E-1 curve read at that layer's
    CBR and the design passes. Surface and base minimums come from Table
    7-2. Mixed traffic reduces to ``passes_18kip`` via the reference
    ``mixed_traffic_equivalent_esal`` (controlling-vehicle procedure).

    Parameters
    ----------
    passes_18kip : float
        Equivalent passes of the 18-kip single-axle, dual-tire load over
        the design life (design-lane).
    cbr_subgrade : float
        Subgrade design CBR (%).
    cbr_base, cbr_subbase : float, optional
        Base course design CBR (default 80, Table 7-1 range) and optional
        subbase CBR (omit for a 2-layer surface+base section).
    frost : dict, optional
        Seasonal-frost spec (reduced-subgrade-strength method, Ch 19):
        {'uscs_class': ..., 'finer_than_0_02mm_pct': ...}. If the soil is
        frost-susceptible, the frost support index replaces the subgrade
        CBR and the governing (thicker) section is reported.
    thickness_increment_in : float, optional
        Round-UP increment (default 0.5 in).

    Returns
    -------
    dict with the layered section, both non-frost/frost solutions when a
    frost spec is given, chart tolerances, and full provenance.
    """
    if passes_18kip is None or passes_18kip <= 0:
        raise ValueError(f"passes_18kip must be > 0, got {passes_18kip}")
    if cbr_subgrade is None or cbr_subgrade <= 0:
        raise ValueError(f"cbr_subgrade must be > 0, got {cbr_subgrade}")

    references = []
    notes = []
    warnings = []

    def _cover_over(cbr_layer, label):
        """E-1 cover requirement above a layer of the given CBR; 0.0 when
        the CBR is off the high end of the chart (no structural cover
        needed — Table 7-2 minimums govern instead)."""
        try:
            r = _utb.figure_e1_flexible_thickness(cbr=cbr_layer,
                                                  passes=passes_18kip)
        except ValueError:
            notes.append(
                f"E-1 cover over the {label} (CBR {cbr_layer}) is off the "
                "high-CBR end of the curve — no structural cover required; "
                "Table 7-2 minimum thicknesses govern that interface."
            )
            return 0.0
        add_ref(references, r["reference"])
        if "tolerance" in r:
            warnings.append(f"E-1 ({label}): {r['tolerance']}")
        return r["thickness_in"]

    def _section(cbr_sg, label):
        total_r = _utb.figure_e1_flexible_thickness(cbr=cbr_sg,
                                                    passes=passes_18kip)
        add_ref(references, total_r["reference"])
        if "tolerance" in total_r:
            warnings.append(f"E-1 ({label}): {total_r['tolerance']}")
        t_total = total_r["thickness_in"]

        mins = _utb.table_7_2_min_thickness(esal=passes_18kip,
                                            base_cbr=cbr_base)
        add_ref(references, mins["reference"])
        min_surf = mins.get("surface_in") or 0.0
        min_base = mins.get("base_in") or 0.0

        surface = round_up(max(_cover_over(cbr_base, "base"), min_surf),
                           thickness_increment_in)

        if cbr_subbase is not None:
            base = round_up(
                max(_cover_over(cbr_subbase, "subbase") - surface,
                    min_base), thickness_increment_in)
            subbase = round_up(max(t_total - surface - base, 0.0),
                               thickness_increment_in)
            layers = [
                {"layer": "asphalt_surface", "thickness_in": surface},
                {"layer": "base", "cbr": cbr_base, "thickness_in": base},
                {"layer": "subbase", "cbr": cbr_subbase,
                 "thickness_in": subbase},
            ]
        else:
            base = round_up(max(t_total - surface, min_base),
                            thickness_increment_in)
            layers = [
                {"layer": "asphalt_surface", "thickness_in": surface},
                {"layer": "base", "cbr": cbr_base, "thickness_in": base},
            ]
        provided = sum(l["thickness_in"] for l in layers)
        return {
            "design_cbr_subgrade": cbr_sg,
            "required_total_thickness_in": t_total,
            "provided_total_thickness_in": round(provided, 2),
            "layers": layers,
        }

    result = {
        "method": "UFC 3-250-01 (2016) CBR procedure",
        "passes_18kip": passes_18kip,
        "section": _section(cbr_subgrade, "subgrade"),
    }

    if frost:
        fc = _utb.table_19_2_frost_classification(**frost)
        add_ref(references, fc["reference"])
        group = fc.get("frost_group")
        result["frost_group"] = group
        if group and str(group).upper() not in ("NFS", "PFS", "NONE"):
            fsi = _utb.table_19_3_frost_support_index(group)
            add_ref(references, fsi["reference"])
            cbr_frost = fsi.get("soil_support_index")
            result["frost_section"] = _section(cbr_frost, "frost FSI")
            governs = (result["frost_section"]
                       ["provided_total_thickness_in"]
                       > result["section"]["provided_total_thickness_in"])
            result["frost_governs"] = governs
            notes.append(
                f"Seasonal frost (group {group}): reduced-subgrade-strength "
                f"method, frost support index {cbr_frost} used in place of "
                f"CBR {cbr_subgrade}; "
                + ("FROST GOVERNS." if governs else "non-frost governs.")
            )
        else:
            notes.append(f"Frost classification: {group} — no frost "
                         "adjustment required.")

    result.update({"notes": notes, "warnings": warnings,
                   "references": references})
    return result


def design_rigid_pavement_ufc(
    passes_18kip,
    flexural_strength_psi,
    k_pci=None,
    subgrade=None,
    stabilized_foundation=None,
    thickness_increment_in=0.5,
) -> dict:
    """UFC 3-250-01 rigid (plain concrete) design for roads/parking.

    Slab thickness from the Figure F-1 family (flexural strength R x
    modulus of subgrade reaction k x equivalent 18-kip passes), reduced
    per Eq 13-1 when placed on a bound/stabilized foundation
    (``stabilized_foundation`` = {'ef_psi': ..., 'hs_in': ...}).
    ``k_pci`` directly, or ``subgrade`` = {'uscs_group': ...,
    'moisture_pct': ...} for the Table 10-1 lookup.
    """
    if passes_18kip is None or passes_18kip <= 0:
        raise ValueError(f"passes_18kip must be > 0, got {passes_18kip}")
    references = []
    notes = []
    warnings = []

    if k_pci is None:
        if not subgrade:
            raise ValueError(
                "Provide k_pci directly or subgrade={'uscs_group', "
                "'moisture_pct'} for the Table 10-1 lookup."
            )
        kr = _utb.table_10_1_k_subgrade(**subgrade)
        add_ref(references, kr["reference"])
        k_pci = kr.get("k_pci") or kr.get("k_pci_mid")
        notes.append(f"k = {k_pci} pci from Table 10-1 for {subgrade}.")

    f1 = getattr(_utb, "figure_f1_rigid_thickness", None)
    if f1 is None:
        raise NotImplementedError(
            "The Figure F-1 rigid design curve is not yet digitized in "
            "geotech_references.ufc_pavement — the UFC rigid path needs it. "
            "(In progress; the flexible UFC path and Eq 13-1 are available.)"
        )
    hd_r = f1(flexural_strength_psi=flexural_strength_psi, k_pci=k_pci,
              passes=passes_18kip)
    add_ref(references, hd_r["reference"])
    if "tolerance" in hd_r:
        warnings.append(f"F-1: {hd_r['tolerance']}")
    hd = hd_r["thickness_in"]

    result = {
        "method": "UFC 3-250-01 (2016) rigid procedure",
        "passes_18kip": passes_18kip,
        "flexural_strength_psi": flexural_strength_psi,
        "k_pci": k_pci,
        "hd_required_in": hd,
    }
    if stabilized_foundation:
        ho_r = _ueq.plain_concrete_thickness_on_stabilized_foundation(
            hd_in=hd, ef_psi=stabilized_foundation["ef_psi"],
            hs_in=stabilized_foundation["hs_in"])
        add_ref(references, ho_r["reference"])
        result["ho_on_stabilized_in"] = ho_r["ho_in"]
        result["slab_provided_in"] = round_up(ho_r["ho_in"],
                                              thickness_increment_in)
        notes.append(
            f"Eq 13-1 stabilized-foundation reduction: hd {hd} -> ho "
            f"{ho_r['ho_in']} in (Ef {stabilized_foundation['ef_psi']:,} "
            f"psi, hs {stabilized_foundation['hs_in']} in)."
        )
    else:
        result["slab_provided_in"] = round_up(hd, thickness_increment_in)

    result.update({"notes": notes, "warnings": warnings,
                   "references": references})
    return result


def compare_flexible_pavement_methods(
    passes_18kip,
    cbr_subgrade=None,
    mr_psi=None,
    reliability_pct=95.0,
    so=0.45,
    pt=2.5,
    cbr_base=80.0,
    cbr_subbase=30.0,
    eac_psi=400000.0,
    ebs_psi=None,
    esb_psi=None,
) -> dict:
    """Run BOTH guides on a shared 18-kip traffic basis and compare.

    Traffic: ``passes_18kip`` is used as AASHTO W18 directly (the 18-kip
    single axle has LEF = 1.0 by definition) and as the UFC E-1 pass
    count — the same physical count under two different damage models.
    Subgrade: give ``cbr_subgrade`` (Mr derived via the FHWA correlation)
    or ``mr_psi`` (CBR derived by inversion); both derivations are stated
    in ``notes``. AASHTO layer moduli default from the layer CBRs via the
    same correlation unless given directly.
    """
    from .flexible import PavementLayer, design_flexible_pavement

    if (cbr_subgrade is None) == (mr_psi is None):
        raise ValueError("Give exactly one of cbr_subgrade or mr_psi.")
    notes = []
    references = []
    if cbr_subgrade is not None:
        mr = _cbr_to_mr(cbr_subgrade, references, notes)
        cbr = cbr_subgrade
    else:
        mr = mr_psi
        cbr = _mr_to_cbr(mr_psi, notes)

    ebs = ebs_psi or _cbr_to_mr(cbr_base, references, notes)
    esb = esb_psi or _cbr_to_mr(cbr_subbase, references, notes)

    aashto = design_flexible_pavement(
        w18=passes_18kip, reliability_pct=reliability_pct, so=so, pt=pt,
        mr_psi=mr,
        layers=[
            PavementLayer("asphalt", modulus_psi=eac_psi),
            PavementLayer("granular_base", modulus_psi=ebs),
            PavementLayer("granular_subbase", modulus_psi=esb),
        ])
    ufc = design_flexible_pavement_ufc(
        passes_18kip=passes_18kip, cbr_subgrade=cbr,
        cbr_base=cbr_base, cbr_subbase=cbr_subbase)

    aashto_total = sum(l["thickness_in"] for l in aashto.layers)
    ufc_total = ufc["section"]["provided_total_thickness_in"]
    notes.extend([
        "Traffic basis: the SAME 18-kip single-axle pass count feeds both "
        "guides (AASHTO LEF(18k single) = 1.0 by definition); the guides' "
        "damage models differ, so differing sections are expected, not an "
        "error.",
        f"Reliability exists only in AASHTO (R = {reliability_pct}%, So = "
        f"{so}); the UFC procedure is deterministic.",
        "AASHTO layer split uses the Figure 3.2 modulus cascade; the UFC "
        "split uses CBR cover curves — layer-by-layer thicknesses are not "
        "directly interchangeable even when totals are close.",
    ])
    return {
        "traffic_18kip_passes": passes_18kip,
        "aashto_1993": {
            "sn_required": aashto.sn_required,
            "layers": [{"layer": l["layer_type"],
                        "thickness_in": l["thickness_in"]}
                       for l in aashto.layers],
            "total_thickness_in": round(aashto_total, 2),
            "reliability_pct": reliability_pct,
        },
        "ufc_3_250_01": {
            "layers": ufc["section"]["layers"],
            "total_thickness_in": ufc_total,
            "required_total_thickness_in":
                ufc["section"]["required_total_thickness_in"],
        },
        "delta_total_thickness_in": round(ufc_total - aashto_total, 2),
        "notes": notes,
        "ufc_warnings": ufc.get("warnings", []),
        "aashto_warnings": aashto.warnings,
        "references": references,
    }
