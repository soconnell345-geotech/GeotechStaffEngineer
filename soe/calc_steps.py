"""
Calculation package steps for support of excavation (SOE) walls.

Three walls, each the SOE tool's own result (BACKLOG GM-41):

* ``cantilever`` -- ``analyze_cantilever_excavation``: Caltrans T&S Manual
  Simplified Method (Sec 7-5.02) on layered Rankine pressures with water;
* ``braced`` -- ``analyze_braced_excavation``: Terzaghi-Peck apparent
  pressure envelope, tributary support loads, free earth support below the
  lowest support;
* ``fhwa_anchored`` -- ``fhwa_apparent_pressure_anchored_wall``: the FHWA
  GEC-4 apparent-pressure envelope and tributary anchor loads (free earth
  support about a single anchor).

Every number printed is the result's (or the inputs the run used); nothing
here works a value out again (GM-141 / GM-143). An equation is shown with
the values the result carries; where the result does not carry an
intermediate, the step says what it is instead of re-deriving it.
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64
from geotech_common.log_spiral import source_label

DISPLAY_NAME = "Support of Excavation Wall Analysis"

REFERENCES = [
    "California Department of Transportation (2025). Trenching and Shoring "
    "Manual, Section 7-5.02 (cantilever walls, Simplified Method) and "
    "Chapter 8 (anchored walls, Example 8-1).",
    "FHWA-IF-99-015 (1999). Geotechnical Engineering Circular No. 4: Ground "
    "Anchors and Anchored Systems, Section 5.2 (apparent pressure envelopes, "
    "Figure 24), Figure 34 (anchor loads, hinge method), Equation 20 and "
    "Figure 39 (anchor loads and moments, tributary area method).",
    "Terzaghi, K. and Peck, R.B. (1967). Soil Mechanics in Engineering "
    "Practice, 2nd ed. Wiley.",
    "Peck, R.B. (1969). Deep Excavations and Tunneling in Soft Ground. "
    "Proc. 7th ICSMFE, State-of-the-Art Volume.",
    "AASHTO LRFD Bridge Design Specifications, Article 3.11.5.6 (D = 1.2 D0 "
    "for cantilever walls).",
]


# ---------------------------------------------------------------------------
# which wall
# ---------------------------------------------------------------------------

def _wall_type(result, analysis) -> str:
    wt = analysis.get("wall_type") if isinstance(analysis, dict) else None
    if wt:
        return wt
    name = type(result).__name__
    if name == "CantileverExcavationResult":
        return "cantilever"
    if name == "BracedExcavationResult":
        return "braced"
    return "fhwa_anchored"


def _geometry(analysis):
    if isinstance(analysis, dict):
        return analysis.get("geometry")
    return analysis


def _get(analysis, key, default=None):
    return analysis.get(key, default) if isinstance(analysis, dict) \
        else default


def _f(x, nd=2):
    return "not found" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Values Used
# ---------------------------------------------------------------------------

def get_input_summary(result, analysis) -> List[InputItem]:
    """The inputs the analysis used."""
    wt = _wall_type(result, analysis)
    if wt == "fhwa_anchored":
        return _fhwa_inputs(result, analysis)
    items = [InputItem("Wall", "Wall type",
                       "Cantilever (unbraced)" if wt == "cantilever"
                       else "Braced / anchored (multi-level)")]
    g = _geometry(analysis)
    H = result.excavation_depth
    items.append(InputItem("H", "Excavation depth", H, "m"))
    if g is not None:
        items.append(InputItem("q", "Surcharge on the retained side",
                               g.surcharge, "kPa"))
        items.append(InputItem(
            "GWT", "Water table, retained side (below top of wall)",
            "none" if g.gwt_depth is None else g.gwt_depth,
            "" if g.gwt_depth is None else "m"))
        gwt_e = getattr(g, "gwt_depth_excavation", None)
        if gwt_e is None:
            gwt_e = g.gwt_depth
        items.append(InputItem(
            "GWT_e", "Water table, excavation side (not above the base)",
            "none" if gwt_e is None else max(gwt_e, H),
            "" if gwt_e is None else "m"))
        if wt == "braced":
            for k, s in enumerate(g.support_levels, 1):
                items.append(InputItem(
                    f"S{k}", f"Support {k} ({s.support_type}), depth",
                    s.depth, "m"))
    if wt == "cantilever":
        items.append(InputItem("FS_p", "Factor of safety on passive pressure",
                               result.FOS_passive))
        items.append(InputItem("D/D0", "Embedment increase",
                               result.embedment_increase))
    else:
        items.append(InputItem("FS", "Embedment factor of safety, "
                               "MR = FS x MD", result.embedment_FS))
        inc = _get(analysis, "embedment_increase")
        if inc is not None:
            items.append(InputItem("D/D_FS", "Embedment increase", inc))
    fy = _get(analysis, "Fy")
    if fy is not None:
        items.append(InputItem("Fy", "Steel yield strength", fy, "MPa"))
    return items


def _fhwa_inputs(result, analysis):
    u = analysis if isinstance(analysis, dict) else {}
    items = [
        InputItem("Wall", "Wall type", "Anchored, FHWA apparent pressure"),
        InputItem("H", "Wall height (excavation depth)", u.get("H"), "m"),
        InputItem("z_a", "Anchor depths",
                  ", ".join(f"{d:g}" for d in u.get("anchor_depths", [])),
                  "m"),
        InputItem("γ", "Soil unit weight", u.get("gamma"), "kN/m³"),
        InputItem("φ'", "Soil friction angle", u.get("phi"), "deg"),
        InputItem("q", "Surcharge", u.get("surcharge"), "kPa"),
        InputItem("s", "Horizontal anchor spacing", result.get("spacing_m"),
                  "m"),
        InputItem("i", "Anchor inclination below horizontal",
                  result.get("inclination_deg"), "deg"),
    ]
    if u.get("Ka") is not None:
        items.append(InputItem("Ka", "Active coefficient (given)", u["Ka"]))
    if u.get("Kp") is not None:
        items.append(InputItem("Kp", "Passive coefficient (given)", u["Kp"]))
    if result.get("n_anchors") == 1:
        items.append(InputItem("FS", "Embedment factor of safety",
                               result.get("FOS_embedment",
                                          u.get("FOS_embedment"))))
    return items


# ---------------------------------------------------------------------------
# Calculation steps
# ---------------------------------------------------------------------------

def get_calc_steps(result, analysis) -> List[CalcSection]:
    wt = _wall_type(result, analysis)
    if wt == "cantilever":
        return _cantilever_steps(result, analysis)
    if wt == "braced":
        return _braced_steps(result, analysis)
    return _fhwa_steps(result, analysis)


def _log_spiral(layers_used) -> bool:
    """Whether the passive pressure took the log-spiral Kp (GM-242)."""
    return any("passive_wall_friction_deg" in l for l in layers_used)


def _layer_table(layers_used, where: str = "") -> TableData:
    spiral = _log_spiral(layers_used)
    return TableData(
        title="Soil layers and earth pressure coefficients",
        headers=["Layer", "Top (m)", "Bottom (m)", "γ (kN/m³)", "φ' (deg)",
                 "c (kPa)", "Ka", "Kp"]
        + (["δ passive (deg)"] if spiral else []),
        rows=[[k, f"{l['top_m']:.2f}", f"{l['bottom_m']:.2f}",
               f"{l['unit_weight']:.1f}", f"{l['friction_angle']:.1f}",
               f"{l['cohesion']:.1f}", f"{l['Ka']:.4f}", f"{l['Kp']:.4f}"]
              + ([f"{l['passive_wall_friction_deg']:.1f}"] if spiral else [])
              for k, l in enumerate(layers_used, 1)],
        notes=("Ka = tan²(45° − φ'/2) (Rankine); Kp the log-spiral "
               "coefficient at the wall friction δ on the passive side below "
               "the excavation (passive_theory 'log_spiral'; Rankine's "
               "tan²(45° + φ'/2) where δ = 0; see Choices)"
               if spiral else
               "Ka = tan²(45° − φ'/2), Kp = tan²(45° + φ'/2) (Rankine)")
        + f", as the analysis applied them{where}; the last layer "
          "continues below its bottom.")


def _cantilever_steps(r, analysis) -> List[CalcSection]:
    sections = []
    pressures = []
    if r.layers_used:
        pressures.append(_layer_table(r.layers_used))
        l0 = next((l for l in r.layers_used
                   if l["top_m"] <= r.excavation_depth < l["bottom_m"]),
                  r.layers_used[-1])
        phi = l0["friction_angle"]
        pressures.append(CalcStep(
            title="Active coefficient, embedment layer",
            equation="K<sub>a</sub> = tan²(45° − φ'/2)",
            substitution=f"K<sub>a</sub> = tan²(45° − {phi:g}°/2)",
            result_name="Ka", result_value=r.Ka,
            reference="Rankine (1857)"))
        delta = l0.get("passive_wall_friction_deg") or 0.0
        if delta > 0:
            # the log-spiral Kp (GM-242): its source is in the choices
            pressures.append(CalcStep(
                title="Passive coefficient, embedment layer",
                equation="K<sub>p</sub> log spiral, vertical face, level "
                         "ground in front",
                substitution=f"K<sub>p</sub> at φ' = {phi:g}°, δ = "
                             f"{delta:g}°",
                result_name="Kp", result_value=r.Kp,
                reference=source_label(next(
                    (c["chosen"] for c in r.choices
                     if "log_spiral_source" in c["what"]),
                    "ufc_3_220_20"))))
        else:
            pressures.append(CalcStep(
                title="Passive coefficient, embedment layer",
                equation="K<sub>p</sub> = tan²(45° + φ'/2)",
                substitution=f"K<sub>p</sub> = tan²(45° + {phi:g}°/2)",
                result_name="Kp", result_value=r.Kp,
                reference="Rankine (1857)"))
    pressures.append(
        "Active pressure on the retained side: σa = Ka·σ'v − 2c·√Ka (no "
        "tension), plus the net water pressure u(retained) − u(excavation). "
        "Passive pressure on the excavation side below H: σp = (Kp·σ'v + "
        f"2c·√Kp) / FS, FS = {r.FOS_passive:g} (Caltrans T&S 7-5.02, p. 7-11, "
        "step 1).")
    if r.pressure_points:
        pressures.append(TableData(
            title="Pressures at the break depths",
            headers=["z (m)", "at", "σ'v retained (kPa)", "σa (kPa)",
                     "u net (kPa)", "σ'v excavation (kPa)",
                     "σp / FS (kPa)", "net (kPa)"],
            rows=[[f"{p['z_m']:.2f}", p["side"],
                   f"{p['sv_retained_kPa']:.2f}", f"{p['active_kPa']:.2f}",
                   f"{p['water_net_kPa']:.2f}",
                   f"{p['sv_excavation_kPa']:.2f}",
                   f"{p['passive_over_FS_kPa']:.2f}", f"{p['net_kPa']:.2f}"]
                  for p in r.pressure_points],
            notes="Net = σa + u net − σp/FS (toward the excavation positive); "
                  "the last row is the rotation point O."))
    sections.append(CalcSection(title="Earth Pressures", items=pressures))

    emb = []
    if not r.converged:
        emb.append("No depth D0 within 6H balances the moments about O: the "
                   "wall cannot stand as a cantilever. " + (r.notes[0]
                                                           if r.notes else ""))
        sections.append(CalcSection(title="Embedment", items=emb))
        return sections
    H, D0 = r.excavation_depth, r.embedment_converged_m
    emb.append(CalcStep(
        title="Depth to the rotation point O (moments about O)",
        equation="Σ M<sub>O</sub>(driving) = Σ M<sub>O</sub>(passive / FS)",
        substitution=f"solved for D<sub>0</sub> below H = {H:g} m "
                     "(Caltrans T&S 7-5.02, p. 7-11, steps 1-2)",
        result_name="D0", result_value=D0, result_unit="m",
        reference="Caltrans T&S Manual 7-5.02, Figure 7-11, p. 7-11"))
    emb.append(CalcStep(
        title="Net horizontal force at O",
        equation="F<sub>O</sub> = ∫ driving − ∫ passive/FS, 0 to O",
        substitution="from the pressure diagram above",
        result_name="F_O", result_value=r.net_force_at_O_kN_per_m,
        result_unit="kN/m",
        reference="Caltrans T&S 7-5.02 step 4",
        notes="Negative: the passive side carries the balance below O "
              "(the toe force). Positive: increase the embedment."))
    emb.append(CalcStep(
        title="Required embedment",
        equation="D = (D/D<sub>0</sub>) · D<sub>0</sub>",
        substitution=f"D = {r.embedment_increase:g} × {D0:.3f}",
        result_name="D", result_value=r.required_embedment_m, result_unit="m",
        reference="Caltrans T&S 7-5.02, p. 7-11 (D = 1.2 D0, AASHTO 3.11.5.6)"))
    emb.append(CalcStep(
        title="Total wall length",
        equation="L = H + D",
        substitution=f"L = {H:g} + {r.required_embedment_m:.2f}",
        result_name="L", result_value=r.total_wall_length_m,
        result_unit="m"))
    sections.append(CalcSection(title="Embedment", items=emb))
    sections.append(CalcSection(title="Wall Demands",
                                items=_demand_items(r, analysis)))
    return sections


def _demand_items(r, analysis) -> list:
    items = []
    M = r.max_moment_kNm_per_m
    if M is None:
        items.append("The maximum moment, shear and required section are not "
                     "found (no moment balance; see the notes).")
        return items
    items.append(CalcStep(
        title="Maximum bending moment",
        equation="M<sub>max</sub> = max |∫ V dz|",
        substitution=f"at z = {r.max_moment_depth_m:.2f} m below the top "
                     "of the wall",
        result_name="M_max", result_value=M, result_unit="kN·m/m"))
    items.append(CalcStep(
        title="Maximum shear",
        equation="V<sub>max</sub> = max |∫ p dz|",
        substitution="from the same pressure diagram",
        result_name="V_max", result_value=r.max_shear_kN_per_m,
        result_unit="kN/m"))
    fy = _get(analysis, "Fy")
    if fy is not None:
        items.append(CalcStep(
            title="Required section modulus (ASD)",
            equation="S<sub>x</sub> = M<sub>max</sub> / (0.66 F<sub>y</sub>)",
            substitution=f"S<sub>x</sub> = {M:.2f} / (0.66 × {fy:g} × 1000)"
                         " × 10⁶",
            result_name="Sx", result_value=r.required_Sx_cm3,
            result_unit="cm³/m"))
    return items


_ENVELOPE = {
    "sand": ("p = 0.65 K<sub>a</sub> γ H",
             "Terzaghi & Peck (1967); FHWA GEC-4 Eq. 10b, p. 51"),
    "soft_clay": ("p = (1 − m·4c<sub>u</sub>/(γH)) γ H, "
                  "K<sub>a,app</sub> ≥ 0.25, m = 1",
                  "Peck (1969); FHWA GEC-4 5.2.6"),
    "stiff_clay": ("p = (0.2 + 0.2 N/4) γ H ≤ 0.4 γ H, N = γH/c<sub>u</sub>"
                   " (trapezoid, 0.25H-0.75H)",
                   "Peck (1969); FHWA GEC-4 5.2.5"),
}


def _braced_steps(r, analysis) -> List[CalcSection]:
    sections = []
    ap = r.apparent_pressure or {}
    H = r.excavation_depth
    items = []
    t = r.apparent_pressure_type
    eq, ref = _ENVELOPE.get(t, ("p (apparent envelope)", "FHWA GEC-4 5.2.3"))
    if t == "sand":
        sub = (f"p = 0.65 × {ap.get('Ka', 0):.4f} × {ap.get('gamma_avg', 0):.2f}"
               f" × {H:g}")
    elif ap.get("cu_avg") is not None:
        sub = (f"γ = {ap.get('gamma_avg', 0):.2f} kN/m³, c<sub>u</sub> = "
               f"{ap['cu_avg']:.1f} kPa, H = {H:g} m, N = "
               f"{ap.get('stability_number')}")
    else:
        sub = f"H = {H:g} m"
    items.append(CalcStep(
        title=f"Apparent pressure envelope ({t.replace('_', ' ')}, "
              f"{ap.get('shape', '')})",
        equation=eq, substitution=sub,
        result_name="p_max", result_value=r.max_apparent_pressure_kPa,
        result_unit="kPa", reference=ref))
    k_q = "K<sub>a</sub>" if t == "sand" else "1.0"
    items.append(CalcStep(
        title="Surcharge pressure added over the excavated height",
        equation=f"p<sub>s</sub> = {k_q} · q",
        substitution="(GEC-4 5.2.4: surcharge and water added explicitly)",
        result_name="p_s", result_value=r.surcharge_pressure_kPa,
        result_unit="kPa", reference="FHWA GEC-4 5.2.4, p. 52"))
    items.append(
        "Water pressure below the retained-side water table is "
        + ("added to the envelope (sand envelopes are effective stress)."
           if r.water_pressure_included else
           "not added (no water table above the base, or a total-stress clay "
           "envelope)."))
    sections.append(CalcSection(title="Design Pressure", items=items))

    loads = []
    if r.spans:
        loads.append(TableData(
            title="Spans and tributary loads",
            headers=["Span", "Top (m)", "Bottom (m)", "Load (kN/m)",
                     "Centroid below top (m)", "To support above (kN/m)",
                     "To support below (kN/m)"],
            rows=[[k, f"{s['z_top_m']:.2f}", f"{s['z_bot_m']:.2f}",
                   f"{s['load_kN_per_m']:.2f}",
                   f"{s['centroid_below_top_m']:.3f}",
                   f"{s['to_upper_kN_per_m']:.2f}",
                   f"{s['to_lower_kN_per_m']:.2f}"]
                  for k, s in enumerate(r.spans, 1)],
            notes="The hinge method of FHWA GEC-4 Figure 34 (p. 67): the "
                  "span above the first support is a cantilever (all to "
                  "that support); interior spans split by their load's "
                  "centroid; the bottom span goes wholly to the lowest "
                  "support, with no subgrade reaction."))
    for k, rx in enumerate(r.support_reactions, 1):
        loads.append(CalcStep(
            title=f"Support {k} load ({rx['type']}, z = {rx['depth_m']:g} m)",
            equation="R<sub>i</sub> = Σ tributary span loads",
            substitution="sum of the span shares above",
            result_name=f"R{k}", result_value=rx["load_kN_per_m"],
            result_unit="kN/m",
            reference="FHWA GEC-4 Figure 34, p. 67 (hinge method)"))
    sections.append(CalcSection(title="Support Loads", items=loads))

    fe = r.free_earth or {}
    emb = []
    if r.layers_used:
        emb.append(_layer_table(
            r.layers_used, " in the free earth body below H (active and "
            "water on the retained side, passive on the excavation side; "
            "above H the body carries the design pressure)"))
    if r.required_embedment_m is None:
        emb.append("Free earth support found no moment balance within 6H; "
                   "the embedment is not found. "
                   + (r.notes[0] if r.notes else ""))
    else:
        emb.append(CalcStep(
            title="Embedment, free earth support about the lowest support",
            equation="M<sub>R</sub> = FS · M<sub>D</sub>",
            substitution=f"FS = {r.embedment_FS:g}; Rankine active, net water "
                         "and passive pressure below H",
            result_name="D_FS", result_value=fe.get("embedment_D_m"),
            result_unit="m", reference="Caltrans T&S Chapter 8; Example 8-1, p. 8-11"))
        if fe.get("embedment_D_FS1_m") is not None:
            emb.append(CalcStep(
                title="Embedment at FS = 1 (the body for loads and moments)",
                equation="M<sub>R</sub> = M<sub>D</sub>",
                substitution="same body",
                result_name="D'", result_value=fe["embedment_D_FS1_m"],
                result_unit="m"))
        inc = _get(analysis, "embedment_increase", 1.0)
        emb.append(CalcStep(
            title="Required embedment",
            equation="D = (D/D<sub>FS</sub>) · D<sub>FS</sub>",
            substitution=f"D = {inc:g} × {fe.get('embedment_D_m', 0):.3f}",
            result_name="D", result_value=r.required_embedment_m,
            result_unit="m"))
        emb.append(CalcStep(
            title="Total wall length", equation="L = H + D",
            substitution=f"L = {H:g} + {r.required_embedment_m:.2f}",
            result_name="L", result_value=r.total_wall_length_m,
            result_unit="m"))
    if fe.get("lowest_support_load_from_below_kN_per_m") is not None:
        emb.append(CalcStep(
            title="Load on the lowest support from the wall below it (FS = 1)",
            equation="T = ∫ driving − ∫ passive, lowest support to tip",
            substitution="free earth support body",
            result_name="T_below",
            result_value=fe["lowest_support_load_from_below_kN_per_m"],
            result_unit="kN/m",
            notes="Not added to the tabulated support load, which takes the "
                  "whole bottom span; see the notes."))
    sections.append(CalcSection(title="Embedment", items=emb))
    sections.append(CalcSection(title="Wall Demands",
                                items=_demand_items(r, analysis)))
    return sections


def _fhwa_steps(r, analysis) -> List[CalcSection]:
    sections = []
    u = analysis if isinstance(analysis, dict) else {}
    H = u.get("H")
    items = [CalcStep(
        title="Active coefficient",
        equation="K<sub>a</sub> = tan²(45° − φ'/2)" if u.get("Ka") is None
        else "K<sub>a</sub> (given)",
        substitution=f"φ' = {u.get('phi')}°",
        result_name="Ka", result_value=r["Ka"],
        reference="Rankine; FHWA GEC-4 5.2.4")]
    h1, hn = r.get("H1_m"), r.get("Hn1_m")
    single = r.get("n_anchors") == 1
    items.append(CalcStep(
        title="Total apparent earth load",
        equation="P<sub>T</sub> = 0.65 K<sub>a</sub> γ H²",
        substitution=f"P<sub>T</sub> = 0.65 × {r['Ka']:.4f} × "
                     f"{u.get('gamma')} × {H}²",
        result_name="PT", result_value=r["PT_total_kN_per_m"],
        result_unit="kN/m", reference="FHWA GEC-4 Figure 24, p. 51",
        notes="The earth load alone; the surcharge pressure below acts over "
              "H in addition and is in the anchor loads."))
    items.append(CalcStep(
        title="Apparent pressure ordinate",
        equation=("p<sub>e</sub> = P<sub>T</sub> / (2H/3)" if single else
                  "p<sub>e</sub> = P<sub>T</sub> / "
                  "(H − H<sub>1</sub>/3 − H<sub>n+1</sub>/3)"),
        substitution=(f"p<sub>e</sub> = {r['PT_total_kN_per_m']} / "
                      + (f"(2 × {H} / 3)" if single else
                         f"({H} − {h1:g}/3 − {hn:g}/3)")),
        result_name="pe", result_value=r["pe_kPa"], result_unit="kPa",
        reference=("FHWA GEC-4 Figure 24(a), p. 51 (one anchor level: "
                   "H − H1/3 − Hn+1/3 = 2H/3)" if single else
                   "FHWA GEC-4 Figure 24(b), p. 51")))
    items.append(CalcStep(
        title="Surcharge pressure", equation="p<sub>s</sub> = K<sub>a</sub> q",
        substitution=f"{r['Ka']:.4f} × {u.get('surcharge')}",
        result_name="ps", result_value=r["ps_kPa"], result_unit="kPa",
        reference="FHWA GEC-4 5.2.4, p. 52; 5.2.10.1"))
    sections.append(CalcSection(title="Apparent Earth Pressure", items=items))

    loads = []
    for k, a in enumerate(r.get("anchors", []), 1):
        loads.append(CalcStep(
            title=f"Anchor {k} horizontal load (z = {a['depth_m']:g} m)",
            equation="T<sub>H,i</sub> (tributary area)" if r["n_anchors"] > 1
            else "T<sub>H</sub> (free earth support about the anchor)",
            substitution="GEC-4 tributary area formulas" if r["n_anchors"]
            > 1 else "moments about the anchor, FS = 1",
            result_name=f"TH{k}", result_value=a["TH_kN_per_m"],
            result_unit="kN/m",
            reference="FHWA GEC-4 Figure 39, p. 80 (tributary area)"
            if r["n_anchors"] > 1 else
            "Caltrans T&S 8-4, Example 8-1 (free earth support)"))
        loads.append(CalcStep(
            title=f"Anchor {k} design load",
            equation="DL = T<sub>H</sub> · s / cos i",
            substitution=f"DL = {a['TH_kN_per_m']:.2f} × "
                         f"{r['spacing_m']:g} / cos {r['inclination_deg']:g}°",
            result_name=f"DL{k}", result_value=a["design_load_kN"],
            result_unit="kN", reference="FHWA GEC-4 Eq. 20, p. 67"))
    if r.get("subgrade_reaction_kN_per_m") is not None:
        loads.append(CalcStep(
            title="Subgrade reaction at the excavation base",
            equation="R", substitution="tributary share below the lowest "
            "anchor", result_name="R",
            result_value=r["subgrade_reaction_kN_per_m"], result_unit="kN/m",
            reference="FHWA GEC-4 Figure 39, p. 80 (R)"))
    sections.append(CalcSection(title="Anchor Loads", items=loads))

    dem = []
    if r.get("embedment_D_m") is not None:
        dem.append(CalcStep(
            title="Embedment (free earth support, single anchor)",
            equation="M<sub>R</sub> = FS · M<sub>D</sub>",
            substitution=f"FS = {r.get('FOS_embedment')}; Kp = "
                         f"{r.get('Kp')}",
            result_name="D", result_value=r["embedment_D_m"],
            result_unit="m", reference="Caltrans T&S Example 8-1, p. 8-11"))
        if r.get("embedment_D_FS1_m") is not None:
            dem.append(CalcStep(
                title="Embedment at FS = 1", equation="M<sub>R</sub> = "
                "M<sub>D</sub>", substitution="same body", result_name="D'",
                result_value=r["embedment_D_FS1_m"], result_unit="m"))
    if r.get("max_moment_kN_m_per_m") is not None:
        dem.append(CalcStep(
            title="Maximum wall moment",
            equation="M<sub>max</sub>",
            substitution=("the largest of the span moments, 13/54 H1² p and "
                          "p l²/10 (tributary area method)"
                          if r["n_anchors"] > 1 else
                          f"at zero shear, z = {r.get('zero_shear_depth_m')} m"),
            result_name="M_max", result_value=r["max_moment_kN_m_per_m"],
            result_unit="kN·m/m",
            reference="FHWA GEC-4 Figure 39, p. 80" if r["n_anchors"] > 1
            else "Caltrans T&S 8-4, Example 8-1"))
    if dem:
        sections.append(CalcSection(title="Embedment and Wall Moment",
                                    items=dem))
    return sections


# ---------------------------------------------------------------------------
# Checks are carried as notes; figures
# ---------------------------------------------------------------------------

def get_figures(result, analysis) -> List[FigureData]:
    wt = _wall_type(result, analysis)
    if wt == "fhwa_anchored":
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if wt == "cantilever":
        if not result.diagram:
            return []
        series = [("", result.diagram, "p_net", "-", "#1f4e79")]
        title = "Net pressure, shear and moment (to O)"
    else:
        series = [(lab, d, key, ls, col) for lab, d, key, ls, col in (
            ("Spans above the excavation", result.diagram, "p", "-",
             "#1f4e79"),
            ("Free earth body below the lowest support (FS = 1)",
             result.free_earth_diagram, "p_net", "--", "#b35900"))
            if d]
        if not series:
            return []
        title = "Design pressure, shear and moment"
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    for lab, d, key, ls, col in series:
        for ax, k in zip(axes, (key, "V", "M")):
            ax.plot(d[k], d["z"], ls=ls, lw=1.5, color=col,
                    label=(lab or None) if ax is axes[0] else None)
    for ax, xl in zip(axes, ("Pressure (kPa)", "Shear (kN/m)",
                             "Moment (kN·m/m)")):
        ax.axvline(0, color="0.6", lw=0.8)
        ax.axhline(result.excavation_depth, color="0.4", ls=":", lw=0.8)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Depth below top of wall (m)")
    axes[0].invert_yaxis()
    if len(series) > 1:
        fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.08 if len(series) > 1 else 0, 1, 1))
    b64 = figure_to_base64(fig)
    plt.close(fig)
    return [FigureData(
        title=title, image_base64=b64,
        caption=("The analysis's own diagram; the dotted line is the "
                 f"excavation base, H = {result.excavation_depth:g} m."),
        width_percent=90)]


def get_tables(result, analysis) -> list:
    """The wall's diagram (pressure, shear and moment along the wall) and,
    for a braced wall, its spans; for the FHWA wall its anchors."""
    from geotech_common.tables import CalcTable
    wt = _wall_type(result, analysis)
    tables = []
    if wt == "fhwa_anchored":
        rows = [[a["depth_m"], a["TH_kN_per_m"], a["design_load_kN"]]
                for a in result.get("anchors", [])]
        if rows:
            tables.append(CalcTable(
                title="Anchor loads",
                columns=[("Depth", "m"), ("T_H", "kN/m"),
                         ("Design load DL", "kN")],
                rows=rows, note="FHWA GEC-4 apparent pressure, tributary "
                "area (free earth support for a single anchor)",
                name="anchor_loads"))
        return tables

    def diagram_rows(d, pkey):
        n = len(d["z"])
        step = max(1, n // 200)
        keep = sorted(set(range(0, n, step)) | {n - 1})
        return [[d["z"][i], d[pkey][i], d["V"][i], d["M"][i]] for i in keep]

    cols = [("Depth z", "m"), ("Pressure", "kPa"), ("Shear V", "kN/m"),
            ("Moment M", "kN·m/m")]
    if wt == "cantilever":
        if result.diagram:
            tables.append(CalcTable(
                title="Net pressure, shear and moment along the wall",
                columns=cols, rows=diagram_rows(result.diagram, "p_net"),
                note="Caltrans T&S 7-5.02 Simplified Method, top of wall to "
                     "the rotation point O (every point the analysis "
                     "integrated, thinned to about 200)",
                name="wall_diagram"))
        return tables
    if result.spans:
        tables.append(CalcTable(
            title="Spans and tributary loads",
            columns=[("Top", "m"), ("Bottom", "m"), ("Load", "kN/m"),
                     ("Centroid below top", "m"),
                     ("To support above", "kN/m"),
                     ("To support below", "kN/m")],
            rows=[[s["z_top_m"], s["z_bot_m"], s["load_kN_per_m"],
                   s["centroid_below_top_m"], s["to_upper_kN_per_m"],
                   s["to_lower_kN_per_m"]] for s in result.spans],
            note="Tributary area method above the excavation",
            name="spans", printed=False))
    if result.diagram:
        tables.append(CalcTable(
            title="Design pressure, shear and moment above the excavation",
            columns=cols, rows=diagram_rows(result.diagram, "p"),
            note="Apparent envelope with surcharge and water, span by span",
            name="span_diagram"))
    if result.free_earth_diagram:
        tables.append(CalcTable(
            title="Free earth body below the lowest support (FS = 1)",
            columns=cols,
            rows=diagram_rows(result.free_earth_diagram, "p_net"),
            note="Net pressure (driving minus passive), shear and moment "
                 "with the support load at the lowest support",
            name="free_earth_diagram"))
    return tables
