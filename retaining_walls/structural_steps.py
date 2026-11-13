"""
Calc-package steps for the cantilever wall's structural design (GM-312):
the sections that follow the stability sections when
``retaining_wall_package`` runs with wall_type ``"cantilever_reinforced"``.

Every step shows its equation, the numbers put in and the printed page,
in SI with the US customary value beside it (the manual, USACE EM
1110-2-2104 (1 November 2023), is in US customary units and the section
design is worked in them).
"""

from typing import List

from calc_package.data_model import (
    CalcSection, CalcStep, CheckItem, InputItem, TableData,
)
from geotech_common.units import (
    kN_per_m_to_kip_per_ft, kNm_per_m_to_kipft_per_ft,
)
from retaining_walls import structural_section as S

EM = "EM 1110-2-2104 (2023)"

REFERENCES = [
    "USACE EM 1110-2-2104, Strength Design for Reinforced Concrete "
    "Hydraulic Structures, 1 November 2023.",
    "FHWA NHI-06-089, Soils and Foundations Reference Manual, Volume II, "
    "2006: Sec. 10.5.3 (p. 10-41) and Fig. 10-17 (p. 10-39).",
    "FHWA GEC-10 (FHWA-NHI-18-024), Drilled Shafts, 2018: Table 6-2 "
    "(p. 6-3), weights and dimensions of deformed bars.",
    "FHWA GEC-6 (FHWA-SA-02-054), Shallow Foundations: Sec. 6.6 "
    "(pp. 121-122), the base pressure.",
]


def _opt(value, fmt="g", unit=""):
    return "not given" if value is None else f"{value:{fmt}}{unit}"


def input_items(result, analysis) -> List[InputItem]:
    """The structural inputs, after the wall's own."""
    m = result.materials
    els = result.elements
    items = [
        InputItem("f'c", "Concrete compressive strength",
                  f"{m['fc_MPa']:g} MPa ({m['fc_psi']:,.0f} psi)", ""),
        InputItem("fy", "Reinforcement yield strength",
                  f"{m['fy_MPa']:g} MPa ({m['fy_psi']:,.0f} psi)", ""),
        InputItem("joint spacing", "Length between control joints along "
                  "the wall", f"{m['joint_spacing_m']:g}", "m"),
        InputItem("beta1", "Stress block factor (balanced steel ratio "
                  "only)", f"{m['beta1']:.3f}", ""),
        InputItem("n", "Modular ratio Es / Ec (service steel stress)",
                  _opt(m["modular_ratio"]), ""),
        InputItem("aggregate", "Largest coarse aggregate",
                  _opt(m["max_aggregate_mm"], unit=" mm"), ""),
        InputItem("cover, stem", "Clear cover to the stem's bars"
                  + ("" if els["stem"]["cover_given"] else " (Table 2-1)"),
                  f"{els['stem']['cover_mm']:g}", "mm"),
    ]
    for name, label in (("heel", "top"), ("toe", "bottom")):
        if name in els:
            items.append(InputItem(
                f"cover, base {label}",
                f"Clear cover to the base slab's {label} bars"
                + ("" if els[name]["cover_given"] else " (Table 2-1)"),
                f"{els[name]['cover_mm']:g}", "mm"))
    if m.get("lap_length_mm") is not None:
        items.append(InputItem("lap", "Lap length supplied by the "
                               "engineer (not checked)",
                               f"{m['lap_length_mm']:g}", "mm"))
    return items


def _us_m(M):
    return f"{kNm_per_m_to_kipft_per_ft(M):.2f} kip-ft/ft"


def _us_v(V):
    return f"{kN_per_m_to_kip_per_ft(V):.2f} kips/ft"


def _case_table(el, title) -> TableData:
    rows = []
    for c in el["load_cases"]:
        rows.append([
            f"{c['case']}, {c['column']}",
            c["M_factors"], f"{c['M_kNm_per_m']:.1f}",
            f"{c['M_kipft_per_ft']:.2f}", c["V_factors"],
            f"{c['V_kN_per_m']:.1f}", f"{c['V_kip_per_ft']:.2f}"])
    return TableData(
        title=title,
        headers=["Case, column", "Combination for M", "M (kN-m/m)",
                 "M (kip-ft/ft)", "Combination for V", "V (kN/m)",
                 "V (kips/ft)"],
        rows=rows,
        notes=(f"{EM} Table E-2 (p. 93). Serviceability: the unfactored "
               "results times the single load factor of Table 3-4 (p. 26; "
               "para. 3-1f, pp. 13-14). Strength: factored loads, the "
               "larger of the D and EV factor pairs (para. 3-3d, p. 22). "
               "The largest moment and the largest shear govern."))


def _section_steps(el) -> list:
    """Flexure, shear and the checks of one element, from its section
    design in US customary units."""
    sec = el["_section"]
    fc, fy = sec_fc_fy(el)
    d, h = sec["d_in"], sec["h_in"]
    Mu = sec["Mu_kipft"]
    items = [
        CalcStep(
            title="Effective depth",
            equation="d = h - cover - d_b / 2",
            substitution=(f"d = {h:.2f} - {sec['cover_in']:.2f} - "
                          f"{sec['bar_diameter_in']:.3f} / 2 in."),
            result_name="d",
            result_value=f"{d:.2f} in. ({el['d_mm']:.0f} mm)",
            reference=f"{EM} para. 2-3 (p. 4); D-3 Step 4 (p. 75), D-5 "
                      "Step 3(d) (p. 84)",
            notes="The main bars are the outer layer of their face."),
        CalcStep(
            title="Required nominal moment",
            equation="M_n = M_u / &phi;, &phi; = 0.90",
            substitution=f"M_n = {Mu:.2f} &times; 12 / 0.90",
            result_name="M_n",
            result_value=f"{sec['Mn_required_kin']:.1f} kip-in./ft",
            reference=f"{EM} D-3 Step 1 (p. 74); &phi; from D-5 Step 3(b) "
                      "(p. 84)"),
    ]
    if sec["Ku"] is not None:
        items += [
            CalcStep(
                title="Steel required for the moment",
                equation="K_u = 1 - &radic;(1 - M_n / (0.425 f'c b "
                         "d&sup2;)); A_s = 0.85 f'c K_u b d / f_y",
                substitution=(
                    f"K_u = 1 - &radic;(1 - {sec['Mn_required_kin']:.1f} "
                    f"/ (0.425 &times; {fc / 1000:.3f} &times; 12 &times; "
                    f"{d:.2f}&sup2;)) = {sec['Ku']:.4f}; A_s = 0.85 "
                    f"&times; {fc / 1000:.3f} &times; {sec['Ku']:.4f} "
                    f"&times; 12 &times; {d:.2f} / {fy / 1000:.0f}"),
                result_name="A_s (flexure)",
                result_value=(f"{sec['As_flexure_in2']:.3f} sq in/ft "
                              f"({el['As_flexure_mm2_per_m']:.0f} mm2/m)"),
                reference=f"{EM} Eq. D-8, D-9 (p. 72), no axial load"),
        ]
    else:
        items.append(
            "The section is too thin for the moment: the radicand of Eq. "
            "D-8 (p. 72) is negative. Thicken the section.")
    items += [
        CalcStep(
            title="Least steel: shrinkage and temperature",
            equation="A_s,min = 0.003 A_g / 2 per face, not less than No. "
                     "4 at 12 in.",
            substitution=f"A_s,min = 0.003 &times; {h:.2f} &times; 12 / 2",
            result_name="A_s,min",
            result_value=(f"{sec['As_min_in2']:.3f} sq in/ft "
                          f"({el['As_minimum_mm2_per_m']:.0f} mm2/m)"),
            reference=f"{EM} para. 2-9g (p. 8) and 2-9a (pp. 6-7)",
            notes="The manual prints no other minimum for flexural "
                  "steel."),
        CalcStep(
            title="Bar chosen",
            equation="A_s = A_bar &times; 12 / s",
            substitution=(f"{sec['bar_size']} (A_bar = "
                          f"{sec['bar_area_in2']:.2f} sq in, d_b = "
                          f"{sec['bar_diameter_in']:.3f} in.) at "
                          f"{sec['spacing_in']:g} in."),
            result_name="A_s (provided)",
            result_value=(f"{sec['As_provided_in2']:.3f} sq in/ft "
                          f"({el['As_provided_mm2_per_m']:.0f} mm2/m): "
                          f"{sec['bar_size']} @ "
                          f"{el['bar']['spacing_mm']:g} mm"),
            reference="Bar areas: FHWA GEC-10 Table 6-2 (p. 6-3). "
                      f"Spacing: {EM} para. 2-6b (p. 5)",
            notes="Chosen by " + sec["chosen_by"] + "."),
        CalcStep(
            title="Design moment strength",
            equation="a = A_s f_y / (0.85 f'c b); &phi;M_n = &phi; A_s "
                     "f_y (d - a/2)",
            substitution=(
                f"a = {sec['As_provided_in2']:.3f} &times; {fy:,.0f} / "
                f"(0.85 &times; {fc:,.0f} &times; 12) = {sec['a_in']:.3f} "
                f"in.; &phi;M_n = 0.90 &times; {sec['As_provided_in2']:.3f}"
                f" &times; {fy / 1000:.0f} &times; ({d:.2f} - "
                f"{sec['a_in']:.3f}/2) / 12"),
            result_name="&phi;M_n",
            result_value=(f"{sec['phi_Mn_kipft']:.2f} kip-ft/ft "
                          f"({el['phi_Mn_kNm_per_m']:.1f} kN-m/m)"),
            reference=f"{EM} Eq. B-49, B-50 (p. 51)"),
        CalcStep(
            title="Steel ratio and the balanced ratio",
            equation="&rho; = A_s / (b d); &rho;_b = 0.85 &beta;1 (f'c / "
                     "f_y) [87,000 / (87,000 + f_y)]",
            substitution=(
                f"&rho; = {sec['As_provided_in2']:.3f} / (12 &times; "
                f"{d:.2f}) = {sec['rho']:.5f}; &rho;_b = 0.85 &times; "
                f"{sec['beta1']:.3f} &times; ({fc:,.0f} / {fy:,.0f}) "
                f"&times; 87,000 / (87,000 + {fy:,.0f})"),
            result_name="&rho;_b",
            result_value=(f"{sec['rho_b']:.5f} (0.50 &rho;_b = "
                          f"{0.5 * sec['rho_b']:.5f}, 0.25 &rho;_b = "
                          f"{0.25 * sec['rho_b']:.5f})"),
            reference=f"{EM} pp. 53 and 75; para. 3-6 and 3-4b(2) (p. 27)",
            notes="&beta;1 is ACI 318-19's (para. 4-1d(4), p. 30); the "
                  "manual prints 0.85 at 4,000 psi and 0.80 at 5,000 "
                  "psi."),
        CalcStep(
            title="Design shear strength (no shear reinforcement)",
            equation="V_c = 2 &radic;f'c b d (f'c in psi); &phi;V_c = "
                     "0.75 V_c",
            substitution=(f"V_c = 2 &radic;{fc:,.0f} &times; 12 &times; "
                          f"{d:.2f} / 1000 = {sec['Vc_kip']:.2f} kips/ft"),
            result_name="&phi;V_c",
            result_value=(f"{sec['phi_Vc_kip']:.2f} kips/ft "
                          f"({el['phi_Vc_kN_per_m']:.1f} kN/m)"),
            reference=f"{EM} Eq. 5-1 (p. 35), without the axial term as "
                      "D-5 Step 3(h) (p. 85); &phi; from p. 84",
            notes=(f"The printed SI form, 0.17 &radic;f'c b d (MPa, mm), "
                   f"gives {el['phi_Vc_SI_form_kN_per_m']:.1f} kN/m, "
                   "about 2 % more; the US form is used, as the manual's "
                   "examples use it.")),
    ]
    for srow in sec["service"]:
        if srow["fs_psi"] is None:
            continue
        if srow["stress_basis"] == "actual":
            items.append(CalcStep(
                title=f"Service steel stress, case {srow['case']} "
                      f"({srow['category']} loads)",
                equation="b x&sup2; / 2 = n A_s (d - x); f_s = M / [A_s "
                         "(d - x/3)]",
                substitution=(f"x = {srow['x_in']:.2f} in.; f_s = "
                              f"{srow['M_kipft']:.2f} &times; 12 / "
                              f"[{sec['As_provided_in2']:.3f} &times; "
                              f"({d:.2f} - {srow['x_in']:.2f}/3)]"),
                result_name="f_s",
                result_value=f"{srow['fs_psi'] / 1000:.2f} ksi",
                reference=f"{EM} D-3 Step 5 (pp. 75-76), D-5 Step 5 "
                          "(p. 86); limit Table 3-3 (p. 26)"))
        items.append(CalcStep(
            title=f"Crack-control spacing, case {srow['case']} "
                  f"({srow['category']} loads)",
            equation="s_max = 15 (40,000 / f_s) - 2.5 c_c, c_c = the "
                     "lesser of 2.5 in. and the cover",
            substitution=(f"s_max = 15 &times; 40,000 / "
                          f"{srow['fs_psi']:,.0f} - 2.5 &times; "
                          f"{min(2.5, sec['cover_in']):.2f}"),
            result_name="s_max",
            result_value=f"{srow['s_max_in']:.1f} in.",
            reference=f"{EM} para. 2-6b (p. 5); the formula as printed in "
                      "D-3 Step 5(b) (p. 76)",
            notes=("On the actual service stress."
                   if srow["stress_basis"] == "actual" else
                   "On the 35 ksi service stress limit of Table 3-3: no "
                   "modular ratio was given for the actual stress "
                   "(para. 2-6b(3) allows either).")))
    for c in sec["checks"]:
        items.append(_check_item(c))
    return items


def sec_fc_fy(el):
    """(f'c, fy) in psi of the element's design."""
    return el["_fc_psi"], el["_fy_psi"]


def _check_item(c) -> CheckItem:
    def num(v):
        return float(v) if isinstance(v, (int, float)) else 0.0
    lower_is_capacity = c["name"].startswith(S.LEAST_ALLOWED_CHECKS)
    unit = c["unit"]
    label = c["name"]
    if lower_is_capacity:
        # demand is the least value allowed; capacity what is provided
        return CheckItem(description=label + f" ({c['reference']})",
                         demand=round(num(c["demand"]), 4),
                         demand_label="least allowed",
                         capacity=round(num(c["capacity"]), 4),
                         capacity_label="provided", unit=unit,
                         passes=c["passes"])
    return CheckItem(description=label + f" ({c['reference']})",
                     demand=round(num(c["demand"]), 5),
                     demand_label="demand",
                     capacity=round(num(c["capacity"]), 5),
                     capacity_label="limit", unit=unit, passes=c["passes"])


def _stem_section(result) -> CalcSection:
    el, loads = result.elements["stem"], result.loads
    rows = []
    for key, name in (("EH", "Earth pressure, without the live load (EH)"),
                      ("ES", "Live-load surcharge (ES)"),
                      ("Hs", "Water (Hs)")):
        V, M = loads.stem[key]
        if key != "EH" and V == 0:
            continue
        rows.append([name, f"{V:.2f}", f"{(M / V if V else 0):.3f}",
                     f"{M:.2f}", _us_v(V), _us_m(M)])
    t = loads.thrust
    items = [
        f"Section: {el['section']}. {el['section_basis']}.",
        "The stem carries the horizontal component of the active "
        f"pressure (K_a = {t['Ka']:.4f}, the wall analysis's) and the "
        "water pressure on the vertical plane through the back of its "
        f"base, over {loads.h_stem_plane:.2f} m. The surcharge is taken "
        "from the back of the stem (FHWA NHI-06-089 Sec. 10.5.3, p. "
        "10-41: 'the surcharge loads within the width of the base slab "
        "are considered for the structural design of the wall stem'). "
        "The stem's weight and the thrust's vertical component are left "
        f"out, as {EM} D-5 Step 2 neglects the stem's weight (p. 84).",
        TableData(
            title="Unfactored horizontal loads on the stem, at its base",
            headers=["Load", "Force (kN/m)", "Arm above the base of the "
                     "stem (m)", "Moment (kN-m/m)", "Force (US)",
                     "Moment (US)"],
            rows=rows),
        _case_table(el, "Stem: load combinations at the base of the stem"),
        CalcStep(
            title="Factored moment and shear",
            equation="M_u, V_u = the largest of the combinations",
            substitution=(f"M_u from case "
                          f"{el['governing_moment_case']['case']} "
                          f"({el['governing_moment_case']['column']}: "
                          f"{el['governing_moment_case']['factors']})"),
            result_name="M_u",
            result_value=(f"{el['Mu_kNm_per_m']:.1f} kN-m/m "
                          f"({el['Mu_kipft_per_ft']:.2f} kip-ft/ft); V_u = "
                          f"{el['Vu_kN_per_m']:.1f} kN/m "
                          f"({el['Vu_kip_per_ft']:.2f} kips/ft)"),
            reference=f"{EM} Table E-2 (p. 93)"),
    ] + _section_steps(el)
    return CalcSection(title="Structural Design: Stem", items=items)


def _pressure_words(bp) -> str:
    return (f"V = {bp['V_kN_per_m']:.1f} kN/m at "
            f"{bp['resultant_from_toe_m']:.3f} m from the toe (e = "
            f"{bp['eccentricity_m']:.3f} m), {bp['shape'].replace('_', ' ')}"
            f": q_toe = {bp['q_toe_kPa']:.1f} kPa, q_heel = "
            f"{bp['q_heel_kPa']:.1f} kPa")


def _slab_section(result, name) -> CalcSection:
    el = result.elements[name]
    gm = el["governing_moment_case"]
    row = max(el["_rows"], key=lambda r: r["M"])
    p = row["state"]["parts"]
    f = p["factors"]
    fwords = (f"load factors D {f['D']:g}, EV {f['EV']:g}, EH {f['EH']:g}, "
              f"ES {f['ES']:g}, Hs {f['Hs']:g}")
    if row["column"] == "serviceability":
        lf = S.SINGLE_LOAD_FACTOR[row["category"]]
        fwords = (f"unfactored loads; the result is multiplied by the "
                  f"single load factor {lf:g}")
    else:
        lf = 1.0
    items = [
        f"Section: {el['section']}. {el['section_basis']}.",
        "Base pressure for structural design: the trapezoid, or the "
        "triangle when the resultant is outside the middle third (FHWA "
        "NHI-06-089 Fig. 10-17, p. 10-39; FHWA GEC-6 Sec. 6.6, pp. "
        f"121-122), from the loads of each combination ({EM} para. 3-1f, "
        "pp. 13-14).",
        _case_table(el, f"{name.capitalize()}: load combinations"),
        f"Governing moment: case {gm['case']} ({gm['column']}: "
        f"{gm['factors']}), {fwords}. " + _pressure_words(
            gm["base_pressure"]) + ".",
    ]
    if name == "toe":
        items.append(CalcStep(
            title="Moment at the front face of the stem",
            equation="M = &int; q(x) (L_toe - x) dx - w L_toe&sup2; / 2",
            substitution=(
                "upward: base pressure"
                + (" and uplift" if result.loads.h_w > 0 else "")
                + f" {p['up_force']:.2f} kN/m, moment {p['up_moment']:.2f}"
                f" kN-m/m (q at the face {p['q_at_face']:.1f} kPa); "
                f"downward: slab and soil over the toe w = "
                f"{p['w_down']:.2f} kPa over {p['length']:.2f} m, moment "
                f"{p['down_moment']:.2f} kN-m/m"
                + (f"; &times; {lf:g}" if lf != 1.0 else "")),
            result_name="M",
            result_value=(f"{row['M']:.1f} kN-m/m ({_us_m(row['M'])}), "
                          "tension in the bottom"),
            reference="statics on the toe; the section is not printed in "
                      "a held reference",
            notes="The shear is taken at d from the face of the stem "
                  f"({EM} para. 5-3, p. 36): "
                  + (f"{p['shear_section']:.3f} m from the toe."
                     if p["shear_section"] > 0 else
                     "d is longer than the toe, so no shear section falls "
                     "on it and V_u = 0.")))
    else:
        items.append(CalcStep(
            title="Moment at the back face of the stem",
            equation="M = w L_heel&sup2; / 2 + W_wedge (2 L_heel / 3) + "
                     "P_v L_heel - &int; q(x) (x - x_back) dx",
            substitution=(
                "downward: slab, soil"
                + (", surcharge" if f["ES"] or result.loads.q_dead else "")
                + f" w = {p['w_down']:.2f} kPa over {p['length']:.2f} m"
                + (f", backfill wedge {p['wedge']:.2f} kN/m"
                   if p["wedge"] else "")
                + (f", thrust's vertical component {p['tip']:.2f} kN/m at "
                   "the end of the heel" if p["tip"] else "")
                + f": {p['down_force']:.2f} kN/m, moment "
                f"{p['down_moment']:.2f} kN-m/m; upward: base pressure"
                + (" and uplift" if result.loads.h_w > 0 else "")
                + f" {p['up_force']:.2f} kN/m, moment {p['up_moment']:.2f}"
                f" kN-m/m (q at the face {p['q_at_face']:.1f} kPa)"
                + (f"; &times; {lf:g}" if lf != 1.0 else "")),
            result_name="M",
            result_value=(f"{row['M']:.1f} kN-m/m ({_us_m(row['M'])}), "
                          "tension in the top"),
            reference="statics on the heel; the section is not printed in "
                      "a held reference",
            notes=f"The shear is taken at the same face ({EM} para. 5-3, "
                  "p. 36)."))
    items.append(CalcStep(
        title="Factored moment and shear",
        equation="M_u, V_u = the largest of the combinations",
        substitution=(f"V_u from case "
                      f"{el['governing_shear_case']['case']} "
                      f"({el['governing_shear_case']['column']}: "
                      f"{el['governing_shear_case']['factors']})"),
        result_name="M_u",
        result_value=(f"{el['Mu_kNm_per_m']:.1f} kN-m/m "
                      f"({el['Mu_kipft_per_ft']:.2f} kip-ft/ft); V_u = "
                      f"{el['Vu_kN_per_m']:.1f} kN/m "
                      f"({el['Vu_kip_per_ft']:.2f} kips/ft)"),
        reference=f"{EM} Table E-2 (p. 93)"))
    items += _section_steps(el)
    if "reverse_moment" in el:
        items.append("Note: " + el["reverse_moment"]["note"] + " ("
                     f"{el['reverse_moment']['M_kNm_per_m']:.1f} kN-m/m).")
    for w in el["warnings"]:
        items.append("Warning: " + w + ".")
    return CalcSection(title=f"Structural Design: {name.capitalize()}",
                       items=items)


def sections(result, analysis) -> List[CalcSection]:
    """The structural sections: basis and load cases, stem, toe, heel,
    then the steel, the checks and what was not designed."""
    m = result.materials
    for el in result.elements.values():
        el["_fc_psi"], el["_fy_psi"] = m["fc_psi"], m["fy_psi"]
    basis = CalcSection(
        title="Structural Design: Basis and Load Cases",
        items=[
            f"Design basis: {result.basis}. The design is worked in the "
            "manual's US customary units on a 12 in. strip of wall and "
            "reported in both systems (conversion factors: its Glossary "
            "Section III, p. 129).",
            TableData(
                title="Load cases for an earth retaining wall (Table E-2, "
                      "p. 93)",
                headers=["Case", "Designed", "Serviceability", "Strength"],
                rows=[[f"{c['case']}: {c['description']} "
                       f"({c['category']})",
                       "yes" if c["designed"] else
                       "no" + (f" ({c['why_not']})" if c.get("why_not")
                               else ""),
                       c.get("serviceability", "-"),
                       c.get("strength", "-")]
                      for c in result.load_cases],
                notes="D: the concrete. EV: the soil over the heel and "
                      "the toe and a dead-load surcharge on the heel. EH: "
                      "the active thrust without the live load. ES: the "
                      "live-load surcharge. Hs: the water's thrust and "
                      "the uplift. The strength factor 1.5 on EH assumes "
                      "the active state: Table E-2 note b (p. 93): 'The "
                      "load factors shown for EH assume that sufficient "
                      "movement occurs for a design condition using "
                      "active and passive earth pressures. This depends "
                      "on the expected movement of the structure.' The "
                      "at-rest basis the manual also prints (EH driving "
                      "1.35, para. 3-3d, p. 22) is not built."),
            CalcStep(
                title="Strength reduction factors",
                equation="&phi; = 0.90 (bending), 0.75 (shear)",
                substitution="as the manual's example prints them",
                result_name="&phi;", result_value="0.90, 0.75",
                reference=f"{EM} D-5 Step 3(b) (p. 84): 'For shear; phi = "
                          "0.75 (ACI 318-19). For bending; phi = 0.90 (ACI "
                          "318-19)'"),
        ])
    out = [basis, _stem_section(result)]
    for name in ("toe", "heel"):
        if name in result.elements:
            out.append(_slab_section(result, name))

    bars = result.reinforcement["bars"]
    steel = [
        TableData(
            title="Bar schedule",
            headers=["Mark", "Where", "Runs", "Role", "Bar and spacing",
                     "Cover (mm)", "As required / provided (mm2/m)",
                     "End"],
            rows=[[b["mark"], f"{b['element']}, {b['face']}",
                   b["runs"].replace("_", " "),
                   "flexure" if b["role"] == "flexure"
                   else "temperature and shrinkage",
                   f"{b['size']} @ {b['spacing_mm']:g} mm",
                   f"{b['cover_mm']:g}",
                   ("-" if b["As_required_mm2_per_m"] is None
                    else f"{b['As_required_mm2_per_m']:.0f}")
                   + f" / {b['As_provided_mm2_per_m']:.0f}",
                   "straight" if b["end"] is None else
                   "standard 90-degree hook into the " + b["end"]["into"]]
                  for b in bars],
            notes="Temperature and shrinkage steel: "
                  f"{EM} para. 2-9 and Table 2-3 (pp. 6-8), half in each "
                  "face, not less than No. 4 at 12 in. nor more than No. 9 "
                  "at 12 in.; along the wall by the joint spacing, up the "
                  "stem and across the base 0.003 of the gross area. No "
                  "bar is cut off or lapped."),
    ]
    d = result.development
    if d["computed"]:
        db = result.elements["stem"]["_section"]["bar_diameter_in"]
        steel.append(CalcStep(
            title="Stem bars: development length of the standard hook",
            equation="l_dh = [f_y &psi;_e &psi;_c &psi;_r &psi;_o / (55 "
                     "&lambda; &radic;f'c)] d_b^1.5, &psi;_c = f'c / "
                     "15,000 + 0.6",
            substitution=(f"l_dh = [{m['fy_psi']:,.0f} &times; 1.0 &times; "
                          f"{d['psi_c']:.3f} &times; 1.0 &times; 1.0 / (55 "
                          f"&times; 1.0 &times; &radic;{m['fc_psi']:,.0f})]"
                          f" &times; {db:.3f}^1.5"),
            result_name="l_dh",
            result_value=(f"{d['ldh_in']:.1f} in. ({d['ldh_mm']:.0f} mm); "
                          f"available {d['available_in']:.1f} in. "
                          f"({d['available_mm']:.0f} mm)"),
            reference=d["reference"],
            notes="Uncoated bars, no confining reinforcement, side cover "
                  "of 6 bar diameters or more, normal weight concrete, as "
                  "the example takes them. " + d["confirm"] + " Available: "
                  + d["available_basis"] + "."))
    else:
        steel.append("Stem bars' hook development length: not computed. "
                     + d.get("why_not", ""))
    for c in result.checks:
        steel.append(_check_item(c))
    out.append(CalcSection(
        title="Structural Design: Reinforcement and Development",
        items=steel))
    out.append(CalcSection(
        title="Structural Design: Not Designed",
        items=[
            "The design stops short of the following. Each is the "
            "engineer's to complete.",
            TableData(title="Not designed",
                      headers=["What", "Why"],
                      rows=[[n["what"], n["why"]]
                            for n in result.not_designed])]))
    return out
