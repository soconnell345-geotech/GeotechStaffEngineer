"""UFC 3-250-01 (2016) roads/parking pavement design — the DoD alternative
to the AASHTO 1993 procedures, plus cross-guide comparison.

The UFC method characterizes traffic as equivalent passes of an 18-kip
single-axle, dual-tire load (controlling-vehicle procedure, Guide Ch 4 /
Appendix G) and designs flexible sections by the Corps CBR method (Figure
E-1 cover curves) and rigid slabs by the Westergaard-based Figure F-1
family (+ Eq 13-1 on stabilized foundations). Figure E-1 is read off the
print in ``ufc_e1`` (GM-298); the other charts and tables come from
``geotech_references.ufc_pavement`` (Appendix-G-anchored digitization),
whose chart-read tolerances propagate to the results.

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
from .ufc_e1 import figure_e1_thickness as _e1
from .ufc_f1 import (PARA_13_3_REF, TABLE_10_1_NOTE_REF,
                     figure_f1_thickness as _f1)

# UFC 3-250-01 p. 20 (pdf_page 39): the drainage criteria require "a
# minimum of 4 in (100 mm) of drainage layer and 4 in (100 mm) of subbase
# (separation) course for most pavements"; Appendix G-3 (printed 259-260)
# ends with a 4 in subbase where its cover arithmetic left 4.4 in.
_MIN_SUBBASE_IN = 4.0
_MIN_SUBBASE_REF = "UFC 3-250-01, para 7-5 (pdf_page 39, printed 20)"
# Para 13-3 (pdf_page 56, printed 37): "The minimum thickness of plain
# concrete is 6 in (150 mm)."
_MIN_PLAIN_CONCRETE_IN = 6.0


def _cbr_to_mr(cbr, references, notes):
    r = _fhwa.resilient_modulus_from_cbr(cbr)
    add_ref(references, r["reference"])
    notes.append(
        f"CBR {cbr} -> Mr {r['mr']:.0f} psi via {r['equation']} "
        "(the preferred FHWA-NHI-05-037 correlation; override with a "
        "direct modulus if you have one)."
    )
    return r["mr"]


# AASHTO 1993 Figures 2.6 (base, pdf_page 108, printed II-19) and 2.7
# (subbase, pdf_page 110, printed II-21) set a CBR scale and a modulus
# scale side by side, read straight across. Heights (scan pixels, ~305
# dpi) of each labelled tick, read off the print (GM-298). The guide's
# AASHO Road Test materials sit on them: base CBR 100 with EBS 30,000 psi,
# subbase CBR 30 with ESB 15,000 psi (printed II-20).
_FIG_CBR_SCALES = {
    "granular_base": (
        "AASHTO 1993 Guide, Figure 2.6 (pdf_page 108, printed II-19)",
        ((100, 1467.0), (70, 1527.5), (60, 1561.0), (50, 1618.5),
         (40, 1675.0), (30, 1746.5), (20, 1897.0)),
        ((40000, 1256.5), (30000, 1463.5), (25000, 1589.0),
         (20000, 1730.5), (15000, 1897.5))),
    "granular_subbase": (
        "AASHTO 1993 Guide, Figure 2.7 (pdf_page 110, printed II-21)",
        ((100, 1246.0), (70, 1309.5), (50, 1338.5), (40, 1374.5),
         (30, 1442.5), (20, 1533.0), (10, 1620.0), (5, 1809.5)),
        ((20000, 1261.5), (15000, 1432.5), (14000, 1477.5),
         (13000, 1521.5), (12000, 1563.5), (11000, 1598.0),
         (10000, 1639.5))),
}


def _cbr_to_layer_modulus(cbr, layer_type, references, notes):
    """A granular base's or subbase's modulus from its CBR, read across
    the guide's own Figure 2.6 / 2.7 scales."""
    import math

    import numpy as np

    ref, cbr_ticks, mod_ticks = _FIG_CBR_SCALES[layer_type]
    which = "base" if layer_type == "granular_base" else "subbase"
    lo, hi = cbr_ticks[-1][0], cbr_ticks[0][0]
    if not lo <= cbr <= hi:
        raise ValueError(
            f"{which} CBR {cbr} is off the printed CBR scale of {ref} "
            f"({lo}-{hi}); give the {which} modulus directly."
        )
    c_log = [math.log10(c) for c, _ in reversed(cbr_ticks)]
    c_y = [y for _, y in reversed(cbr_ticks)]
    y = float(np.interp(math.log10(cbr), c_log, c_y))
    m_y = [yy for _, yy in mod_ticks]
    # The two scales' end ticks sit up to ~15 px apart where the guide
    # lines them up (CBR 100 with 20,000 psi on Figure 2.7).
    if y > m_y[-1] + 20 or y < m_y[0] - 20:
        raise ValueError(
            f"{which} CBR {cbr} reads past the end of the modulus scale of "
            f"{ref}; give the {which} modulus directly."
        )
    y = min(max(y, m_y[0]), m_y[-1])
    mod = 10 ** float(np.interp(y, m_y, [math.log10(m) for m, _ in
                                         mod_ticks]))
    mod = round(mod, -2)
    add_ref(references, ref)
    symbol = "EBS" if which == "base" else "ESB"
    figure = "Figure 2.6" if which == "base" else "Figure 2.7"
    notes.append(
        f"{which.capitalize()} CBR {cbr} -> {symbol} {mod:,.0f} psi read "
        f"across the AASHTO guide's {figure} (its own CBR and modulus "
        "scales); the NCHRP 1-37A correlation Mr = 2555*CBR^0.64 "
        f"(FHWA-NHI-05-037 Table 5-34) would give {2555 * cbr ** 0.64:,.0f}"
        " psi. Give the modulus directly to use another."
    )
    return mod


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
    7-2; a subbase is at least 4 in (p. 20, as Appendix G-3 ends). Mixed traffic reduces to ``passes_18kip`` via the reference
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
        the CBR is off the end of the curve (no structural cover
        needed — Table 7-2 minimums govern instead)."""
        r = _e1(cbr=cbr_layer, passes=passes_18kip)
        add_ref(references, r["reference"])
        if r["thickness_in"] <= 0.0:
            notes.append(
                f"E-1 cover over the {label} (CBR {cbr_layer}) is off the "
                "high-CBR end of the curve — no structural cover required; "
                "Table 7-2 minimum thicknesses govern that interface."
            )
        return r["thickness_in"]

    def _section(cbr_sg, label):
        total_r = _e1(cbr=cbr_sg, passes=passes_18kip)
        add_ref(references, total_r["reference"])
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
            if subbase < _MIN_SUBBASE_IN:
                add_ref(references, _MIN_SUBBASE_REF)
                notes.append(
                    f"Subbase {subbase} in from the cover curves raised to "
                    f"the {_MIN_SUBBASE_IN:g} in the drainage criteria ask "
                    "of a subbase (separation) course 'for most pavements' "
                    "(UFC 3-250-01 p. 20; Appendix G-3 ends the same way). "
                    "Leave cbr_subbase out for a surface-and-base section."
                )
                subbase = _MIN_SUBBASE_IN
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
    notes.insert(0, "Figure E-1: " + _e1(cbr=cbr_subgrade,
                                         passes=passes_18kip)["tolerance"]
                 + ".")

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
        elif str(group).upper() == "PFS":
            warnings.append(
                "Frost classification PFS: Table 19-2 footnote *** prints "
                "it as \"possibly frost-susceptible, but requires "
                "laboratory test to determine frost design soils "
                "classification\". No frost section is computed; give the "
                "frost group the laboratory test sets."
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

    Slab thickness from Figure F-1 read off the print (``ufc_f1``:
    flexural strength R x modulus of subgrade reaction k x equivalent
    18-kip passes), reduced per Eq 13-1 when placed on a bound/stabilized
    foundation (``stabilized_foundation`` = {'ef_psi': ..., 'hs_in': ...}),
    rounded up to the increment and to the 6 in minimum of para 13-3.
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
        k_pci = kr["k_psi_in"]
        notes.append(f"k = {k_pci:g} pci from Table 10-1 for {subgrade}.")

    hd_r = _f1(flexural_psi=flexural_strength_psi, k_pci=k_pci,
               passes=passes_18kip)
    add_ref(references, hd_r["reference"])
    if hd_r["k_pci"] != k_pci:
        add_ref(references, TABLE_10_1_NOTE_REF)
    notes.append(f"Figure F-1: {hd_r['tolerance']}.")
    notes.extend(hd_r["notes"])
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
        slab = round_up(ho_r["ho_in"], thickness_increment_in)
        notes.append(
            f"Eq 13-1 stabilized-foundation reduction: hd {hd} -> ho "
            f"{ho_r['ho_in']} in (Ef {stabilized_foundation['ef_psi']:,} "
            f"psi, hs {stabilized_foundation['hs_in']} in)."
        )
    else:
        slab = round_up(hd, thickness_increment_in)
    if slab < _MIN_PLAIN_CONCRETE_IN:
        add_ref(references, PARA_13_3_REF)
        notes.append(
            f"Slab {slab} in raised to the {_MIN_PLAIN_CONCRETE_IN:g} in "
            "minimum thickness of plain concrete (para 13-3)."
        )
        slab = _MIN_PLAIN_CONCRETE_IN
    result["slab_provided_in"] = slab

    result.update({"notes": notes, "warnings": warnings,
                   "references": references})
    return result


def ufc_mixed_traffic(
    vehicles,
    representative_cbr,
) -> dict:
    """Automated UFC mixed-traffic reduction (Ch 4 / Table G-1 procedure).

    Takes the raw vehicle mix and does the whole controlling-vehicle
    procedure from the digitized Appendix E curves: per-vehicle required
    thickness at the representative subgrade CBR (Table 4-1 category
    value), controlling vehicle = largest thickness, allowable passes of
    each other vehicle at the controlling thickness by curve inversion,
    then the reference ``mixed_traffic_equivalent_esal`` roll-up.

    Parameters
    ----------
    vehicles : list of dict
        Each: {'vehicle': 'E-1'|'E-2'..'E-31' or a registry slug (e.g.
        'truck_5_axle'), 'design_passes': float, 'name': optional label}.
        Use 'E-1' for the 18-kip ESAL reference vehicle.
    representative_cbr : float
        Representative subgrade CBR for the equivalency step (Table 4-1
        category A-D value — NOT the site CBR; the final design re-enters
        the curve at the real site CBR).

    Returns
    -------
    dict — the reference roll-up result (controlling vehicle, per-vehicle
    equivalent passes, total) plus the per-vehicle worksheet. When the
    controlling vehicle is E-1, ``total_equivalent_passes`` feeds
    ``design_flexible_pavement_ufc`` directly.
    """
    if not vehicles:
        raise ValueError("Provide at least one vehicle dict.")
    references = []
    notes = []

    def _thickness(vehicle, passes):
        if str(vehicle).strip().lower() in ("e-1", "e1", "18kip",
                                            "18_kip_esal"):
            r = _e1(cbr=representative_cbr, passes=passes)
        else:
            r = _utb.figure_e_vehicle_thickness(vehicle,
                                                cbr=representative_cbr,
                                                passes=passes)
        add_ref(references, r["reference"])
        return r["thickness_in"]

    def _allowable(vehicle, thickness_in):
        key = str(vehicle).strip().lower()
        if key in ("e-1", "e1", "18kip", "18_kip_esal"):
            # Invert E-1 by bisection over log10(passes).
            import math as _math
            lo, hi = 2.0, 8.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                try:
                    t = _e1(cbr=representative_cbr,
                            passes=10 ** mid)["thickness_in"]
                except ValueError:
                    hi = mid
                    continue
                if t < thickness_in:
                    lo = mid
                else:
                    hi = mid
            return 10 ** (0.5 * (lo + hi))
        r = _utb.figure_e_vehicle_allowable_passes(
            vehicle, cbr=representative_cbr, thickness_in=thickness_in)
        clamped = r.get("clamped")
        if clamped and "above_max_passes" in str(clamped):
            # The controlling thickness exceeds this vehicle's entire curve
            # family — its allowable passes are effectively UNLIMITED, so
            # it contributes ~nothing to the equivalent total (the guide's
            # own Table G-1 convention; see the reference docstring).
            notes.append(
                f"{vehicle}: controlling thickness exceeds its whole curve "
                "family — allowable passes treated as unlimited "
                "(negligible equivalent contribution, per Table G-1)."
            )
            return float("inf")
        if clamped:
            notes.append(
                f"{vehicle}: allowable passes at the controlling thickness "
                f"clamped at the curve-family bound "
                f"({r['allowable_passes']:,.0f})."
            )
        return r["allowable_passes"]

    worksheet = []
    for v in vehicles:
        t = _thickness(v["vehicle"], v["design_passes"])
        worksheet.append({
            "name": v.get("name", str(v["vehicle"])),
            "vehicle": v["vehicle"],
            "design_passes": v["design_passes"],
            "required_thickness_in": t,
        })
    controlling = max(worksheet, key=lambda w: w["required_thickness_in"])
    for w in worksheet:
        if w is controlling:
            w["allowable_passes_at_controlling"] = None
        else:
            w["allowable_passes_at_controlling"] = _allowable(
                w["vehicle"], controlling["required_thickness_in"])

    roll = _ueq.mixed_traffic_equivalent_esal([
        {"name": w["name"], "design_passes": w["design_passes"],
         "required_thickness_in": w["required_thickness_in"],
         "allowable_passes_at_controlling":
             w["allowable_passes_at_controlling"]}
        for w in worksheet])
    add_ref(references, roll.get("reference"))
    out = dict(roll)
    out.update({
        "representative_cbr": representative_cbr,
        "worksheet": worksheet,
        "controlling_vehicle": controlling["name"],
        "notes": notes,
        "references": references,
    })
    return out


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
    in ``notes``. The AASHTO base and subbase moduli default from their
    CBRs read across the guide's own Figure 2.6 / 2.7 scales (base CBR
    20-100, subbase CBR about 9-100) unless given directly; the
    correlation above, applied to a base, gives a modulus the guide's
    scales put at a higher CBR (CBR 80: 42,200 psi against 28,200).
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

    ebs = ebs_psi or _cbr_to_layer_modulus(cbr_base, "granular_base",
                                           references, notes)
    esb = esb_psi or _cbr_to_layer_modulus(cbr_subbase, "granular_subbase",
                                           references, notes)

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
