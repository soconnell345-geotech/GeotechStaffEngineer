"""Calc-package template for pavement_design (AASHTO 1993).

Renders FlexiblePavementResult / RigidPavementResult into Mathcad-style
sections for ``calc_package.generate_calc_package``. UNITS: US customary
(the 1993 Guide's native units) -- unit strings say so explicitly.
"""

from calc_package.data_model import (CalcSection, CalcStep, CheckItem,
                                     FigureData, InputItem, TableData)

from .results import FlexiblePavementResult, RigidPavementResult

DISPLAY_NAME = "AASHTO 1993 Pavement Structural Design"

REFERENCES = [
    "AASHTO, Guide for Design of Pavement Structures, American Association "
    "of State Highway and Transportation Officials, Washington, D.C., 1993.",
]

_FLEX_EQ = ("log<sub>10</sub>(W<sub>18</sub>) = Z<sub>R</sub>&middot;S<sub>o</sub> + "
            "9.36&middot;log<sub>10</sub>(SN+1) &minus; 0.20 + "
            "log<sub>10</sub>(&Delta;PSI/2.7)/[0.40 + 1094/(SN+1)<sup>5.19</sup>] + "
            "2.32&middot;log<sub>10</sub>(M<sub>R</sub>) &minus; 8.07")

_RIGID_EQ = ("log<sub>10</sub>(W<sub>18</sub>) = Z<sub>R</sub>&middot;S<sub>o</sub> + "
             "7.35&middot;log<sub>10</sub>(D+1) &minus; 0.06 + "
             "log<sub>10</sub>(&Delta;PSI/3.0)/[1 + 1.624&times;10<sup>7</sup>/(D+1)<sup>8.46</sup>] + "
             "(4.22 &minus; 0.32&middot;p<sub>t</sub>)&middot;log<sub>10</sub>{S'<sub>c</sub>&middot;C<sub>d</sub>&middot;"
             "(D<sup>0.75</sup> &minus; 1.132) / [215.63&middot;J&middot;(D<sup>0.75</sup> &minus; "
             "18.42/(E<sub>c</sub>/k)<sup>0.25</sup>)]}")


def _is_ufc_flexible(result):
    return isinstance(result, dict) and "section" in result


def _is_ufc_rigid(result):
    return isinstance(result, dict) and "hd_required_in" in result


def _is_comparison(result):
    return isinstance(result, dict) and "aashto_1993" in result


def _ufc_input_summary(result):
    items = [InputItem("method", "Design basis",
                       result.get("method", "UFC 3-250-01"), "")]
    if "passes_18kip" in result:
        items.append(InputItem("passes", "18-kip single-axle passes",
                               f"{result['passes_18kip']:,.0f}", ""))
    if _is_ufc_flexible(result):
        items.append(InputItem("CBR", "Subgrade design CBR",
                               result["section"]["design_cbr_subgrade"],
                               "%"))
    if _is_ufc_rigid(result):
        items.extend([
            InputItem("R", "Concrete flexural strength",
                      result["flexural_strength_psi"], "psi"),
            InputItem("k", "Modulus of subgrade reaction",
                      result["k_pci"], "pci"),
        ])
    items.append(InputItem("units", "Unit system",
                           "US customary (UFC 3-250-01 native)", ""))
    return items


def _comparison_input_summary(result):
    return [
        InputItem("basis", "Comparison basis",
                  "Both guides on the SAME 18-kip single-axle pass count "
                  "(AASHTO LEF = 1.0 by definition)", ""),
        InputItem("passes", "18-kip passes / AASHTO W18",
                  f"{result['traffic_18kip_passes']:,.0f}", ""),
        InputItem("R_aashto", "AASHTO reliability",
                  result["aashto_1993"].get("reliability_pct"), "%"),
        InputItem("units", "Unit system", "US customary", ""),
    ]


def _ufc_flexible_sections(result):
    sec = result["section"]
    rows = [[l["layer"], l.get("cbr", "-"), l["thickness_in"]]
            for l in sec["layers"]]
    items = [
        CalcStep(
            title="Required total thickness (Figure E-1)",
            equation="t = E-1(CBR, passes)  [Corps CBR method cover curve]",
            substitution=(f"CBR = {sec['design_cbr_subgrade']}, passes = "
                          f"{result['passes_18kip']:,.0f}"),
            result_name="t required",
            result_value=sec["required_total_thickness_in"],
            result_unit="in",
            reference="UFC 3-250-01, Figure E-1 (Appendix E)",
        ),
        TableData(title="Layered section (cover cascade + Table 7-2 minimums)",
                  headers=["Layer", "CBR", "D (in)"], rows=rows,
                  notes=f"Provided total "
                        f"{sec['provided_total_thickness_in']} in."),
        CheckItem(
            description="Total thickness adequacy",
            demand=sec["required_total_thickness_in"],
            demand_label="t required (in)",
            capacity=sec["provided_total_thickness_in"],
            capacity_label="t provided (in)", unit="in",
            passes=(sec["provided_total_thickness_in"]
                    >= sec["required_total_thickness_in"] - 0.51),
        ),
    ]
    sections = [CalcSection(title="UFC 3-250-01 Flexible Design (CBR method)",
                            items=items)]
    if result.get("frost_section"):
        fs = result["frost_section"]
        sections.append(CalcSection(
            title="Seasonal Frost (reduced subgrade strength, Ch 19)",
            items=[
                TableData(
                    title=f"Frost section (group {result.get('frost_group')}"
                          f", FSI {fs['design_cbr_subgrade']})",
                    headers=["Layer", "CBR", "D (in)"],
                    rows=[[l["layer"], l.get("cbr", "-"),
                           l["thickness_in"]] for l in fs["layers"]],
                    notes=("FROST GOVERNS." if result.get("frost_governs")
                           else "Non-frost section governs.")),
            ]))
    return sections


def _ufc_rigid_sections(result):
    items = [
        CalcStep(
            title="Required slab thickness (Figure F-1)",
            equation="hd = F-1(R, k, passes)",
            substitution=(f"R = {result['flexural_strength_psi']:,.0f} psi, "
                          f"k = {result['k_pci']:,.0f} pci, passes = "
                          f"{result['passes_18kip']:,.0f}"),
            result_name="hd required",
            result_value=result["hd_required_in"],
            result_unit="in",
            reference="UFC 3-250-01, Figure F-1 (Appendix F)",
        ),
    ]
    if "ho_on_stabilized_in" in result:
        items.append(CalcStep(
            title="Stabilized-foundation reduction (Eq 13-1)",
            equation="ho = [hd^1.4 - (0.0063*Ef^(1/3)*hs)^1.4]^(1/1.4)",
            substitution=f"hd = {result['hd_required_in']} in",
            result_name="ho",
            result_value=result["ho_on_stabilized_in"],
            result_unit="in",
            reference="UFC 3-250-01, Eq. 13-1",
        ))
    items.append(CheckItem(
        description="Slab thickness adequacy",
        demand=result.get("ho_on_stabilized_in",
                          result["hd_required_in"]),
        demand_label="required (in)",
        capacity=result["slab_provided_in"],
        capacity_label="slab provided (in)", unit="in",
        passes=result["slab_provided_in"] >= result.get(
            "ho_on_stabilized_in", result["hd_required_in"]) - 0.51,
    ))
    return [CalcSection(title="UFC 3-250-01 Rigid Design", items=items)]


def _comparison_sections(result):
    a, u = result["aashto_1993"], result["ufc_3_250_01"]
    rows = []
    n = max(len(a["layers"]), len(u["layers"]))
    for i in range(n):
        al = a["layers"][i] if i < len(a["layers"]) else {}
        ul = u["layers"][i] if i < len(u["layers"]) else {}
        rows.append([al.get("layer", "-"), al.get("thickness_in", "-"),
                     ul.get("layer", "-"), ul.get("thickness_in", "-")])
    rows.append(["TOTAL", a["total_thickness_in"],
                 "TOTAL", u["total_thickness_in"]])
    return [CalcSection(
        title="Method Comparison — AASHTO 1993 vs UFC 3-250-01",
        items=[
            TableData(
                title="Sections side by side (same 18-kip traffic)",
                headers=["AASHTO layer", "D (in)", "UFC layer", "D (in)"],
                rows=rows,
                notes=(f"AASHTO SN required {a.get('sn_required')}; "
                       f"delta (UFC - AASHTO) = "
                       f"{result['delta_total_thickness_in']} in.")),
        ] + list(result.get("notes", [])))]


def get_input_summary(result, analysis=None):
    if _is_comparison(result):
        return _comparison_input_summary(result)
    if _is_ufc_flexible(result) or _is_ufc_rigid(result):
        return _ufc_input_summary(result)
    items = [
        InputItem("W18", "Design-lane 18-kip ESALs", f"{result.w18:,.0f}", ""),
        InputItem("R", "Design reliability",
                  result.reliability_pct if result.reliability_pct is not None
                  else "(ZR given directly)", "%"),
        InputItem("ZR", "Standard normal deviate", result.zr, ""),
        InputItem("So", "Overall standard deviation", result.so, ""),
        InputItem("po", "Initial serviceability", result.po, ""),
        InputItem("pt", "Terminal serviceability", result.pt, ""),
        InputItem("dPSI", "Design serviceability loss", result.delta_psi, ""),
    ]
    if isinstance(result, FlexiblePavementResult):
        items.append(InputItem("MR", "Effective roadbed resilient modulus",
                               f"{result.effective_mr_psi:,.0f}", "psi"))
    else:
        items.extend([
            InputItem("S'c", "PCC modulus of rupture",
                      f"{result.sc_psi:,.0f}", "psi"),
            InputItem("Ec", "PCC elastic modulus",
                      f"{result.ec_psi:,.0f}", "psi"),
            InputItem("J", "Load transfer coefficient", result.j, ""),
            InputItem("Cd", "Drainage coefficient", result.cd, ""),
            InputItem("k", f"Effective modulus of subgrade reaction "
                      f"({result.k_basis.get('basis', '')})",
                      result.k_pci, "pci"),
        ])
    items.append(InputItem("units", "Unit system",
                           "US customary (AASHTO 1993 native)", ""))
    return items


def _flexible_sections(res: FlexiblePavementResult):
    sections = []

    # SN solves over each foundation.
    steps = []
    for row in res.sn_stack:
        steps.append(CalcStep(
            title=f"Required SN over {row['over']}",
            equation=_FLEX_EQ,
            substitution=(f"solve for SN with W<sub>18</sub> = {res.w18:,.0f}, "
                          f"Z<sub>R</sub> = {res.zr}, S<sub>o</sub> = {res.so}, "
                          f"&Delta;PSI = {res.delta_psi}, M<sub>R</sub> = "
                          f"{row['foundation_mr_psi']:,.0f} psi"),
            result_name=f"SN ({row['over']})",
            result_value=row["sn_required"],
            reference="AASHTO 1993 Guide, Figure 3.1 / 3.2",
        ))
    sections.append(CalcSection(
        title="Required Structural Number", items=steps))

    # Layer table.
    headers = ["Layer", "a", "basis", "m", "D (in)", "cumulative SN"]
    rows = []
    for lay in res.layers:
        rows.append([
            lay["layer_type"], lay["a"], lay["a_basis"], lay["m"],
            lay.get("thickness_in", "-"), lay.get("sn_cumulative", "-"),
        ])
    sections.append(CalcSection(
        title="Layered Section (Figure 3.2 split)",
        items=[
            TableData(title="Layer coefficients and thicknesses",
                      headers=headers, rows=rows,
                      notes="Thicknesses rounded UP; drainage m applies to "
                            "unbound layers only (Table 2.4)."),
            CalcStep(
                title="Structural number of the section",
                equation=("SN = a<sub>1</sub>D<sub>1</sub> + a<sub>2</sub>D<sub>2</sub>m<sub>2</sub> + "
                          "a<sub>3</sub>D<sub>3</sub>m<sub>3</sub>"),
                substitution=" + ".join(
                    f"{lay['a']}&times;{lay['thickness_in']}"
                    + (f"&times;{lay['m']}" if lay['m'] != 1.0 else "")
                    for lay in res.layers),
                result_name="SN provided",
                result_value=res.sn_provided,
                reference="AASHTO 1993 Guide, Section 3.1.4",
            ),
        ]))

    checks = [
        CheckItem(
            description="Structural number adequacy",
            demand=res.sn_required, demand_label="SN required",
            capacity=res.sn_provided, capacity_label="SN provided",
            unit="", passes=res.sn_provided >= res.sn_required - 1e-9,
        ),
        CheckItem(
            description="Forward traffic check (Figure 3.1, forward)",
            demand=res.w18, demand_label="W18 design",
            capacity=res.w18_capacity, capacity_label="W18 capacity",
            unit="ESALs", passes=res.w18_capacity >= res.w18,
        ),
    ]
    sections.append(CalcSection(title="Adequacy Checks", items=checks))
    return sections


def _rigid_sections(res: RigidPavementResult):
    k_note = ", ".join(f"{k}={v}" for k, v in res.k_basis.items()
                       if not isinstance(v, (list, dict)))
    steps = [
        CalcStep(
            title="Required slab thickness",
            equation=_RIGID_EQ,
            substitution=(f"solve for D with W<sub>18</sub> = {res.w18:,.0f}, "
                          f"Z<sub>R</sub> = {res.zr}, S<sub>o</sub> = {res.so}, "
                          f"&Delta;PSI = {res.delta_psi}, S'<sub>c</sub> = "
                          f"{res.sc_psi:,.0f} psi, C<sub>d</sub> = {res.cd}, "
                          f"J = {res.j}, E<sub>c</sub> = {res.ec_psi:,.0f} psi, "
                          f"k = {res.k_pci} pci, p<sub>t</sub> = {res.pt}"),
            result_name="D required",
            result_value=res.d_required_in,
            result_unit="in",
            reference="AASHTO 1993 Guide, Figure 3.7",
            notes=f"k basis: {k_note}" + (
                f"; composite-k iterated with D ({res.iterations} passes)"
                if res.iterations > 1 else ""),
        ),
        CalcStep(
            title="Slab thickness provided",
            equation="D provided (rounded up / as checked)",
            substitution=f"D = {res.d_provided_in} in",
            result_name="D provided",
            result_value=res.d_provided_in,
            result_unit="in",
        ),
    ]
    checks = [
        CheckItem(
            description="Slab thickness adequacy",
            demand=res.d_required_in, demand_label="D required (in)",
            capacity=res.d_provided_in, capacity_label="D provided (in)",
            unit="in", passes=res.d_provided_in >= res.d_required_in - 1e-9,
        ),
        CheckItem(
            description="Forward traffic check (Figure 3.7, forward)",
            demand=res.w18, demand_label="W18 design",
            capacity=res.w18_capacity, capacity_label="W18 capacity",
            unit="ESALs", passes=res.w18_capacity >= res.w18,
        ),
    ]
    return [
        CalcSection(title="Slab Thickness Design (Figure 3.7)", items=steps),
        CalcSection(title="Adequacy Checks", items=checks),
    ]


def get_calc_steps(result, analysis=None):
    if _is_comparison(result):
        return _comparison_sections(result)
    if _is_ufc_flexible(result):
        sections = _ufc_flexible_sections(result)
        notes = list(result.get("notes", [])) + [
            f"WARNING: {w}" for w in result.get("warnings", [])]
        if notes:
            sections.append(CalcSection(title="Basis and Assumptions",
                                        items=notes))
        return sections
    if _is_ufc_rigid(result):
        sections = _ufc_rigid_sections(result)
        notes = list(result.get("notes", [])) + [
            f"WARNING: {w}" for w in result.get("warnings", [])]
        if notes:
            sections.append(CalcSection(title="Basis and Assumptions",
                                        items=notes))
        return sections
    if isinstance(result, FlexiblePavementResult):
        sections = _flexible_sections(result)
    elif isinstance(result, RigidPavementResult):
        sections = _rigid_sections(result)
    else:
        raise TypeError(
            f"Unsupported result type for pavement_design calc package: "
            f"{type(result).__name__}"
        )
    env = getattr(result, "environmental", None)
    if env:
        env_steps = [
            CalcStep(
                title="Environmental serviceability loss (swelling / frost heave)",
                equation=("&Delta;PSI<sub>SW</sub> = 0.00335&middot;V<sub>R</sub>&middot;P<sub>s</sub>&middot;"
                          "(1 &minus; e<sup>&minus;&theta;t</sup>);  "
                          "&Delta;PSI<sub>FH</sub> = 0.01&middot;P<sub>F</sub>&middot;&Delta;PSI<sub>MAX</sub>&middot;"
                          "(1 &minus; e<sup>&minus;0.02&middot;&phi;t</sup>)"),
                substitution=(f"at t = {env.get('design_period_yr')} yr: "
                              f"&Delta;PSI<sub>SW</sub> = {env.get('delta_psi_sw', 0)}, "
                              f"&Delta;PSI<sub>FH</sub> = {env.get('delta_psi_fh', 0)}"),
                result_name="dPSI environmental",
                result_value=env.get("delta_psi_total"),
                reference="AASHTO 1993 Guide, Appendix G (Figures G.4/G.8) / Figure 2.2",
            ),
            CalcStep(
                title="Traffic-available serviceability loss (Table 3.1 Step 4)",
                equation="&Delta;PSI<sub>TR</sub> = &Delta;PSI &minus; &Delta;PSI<sub>SW,FH</sub>",
                substitution=(f"{env.get('delta_psi_design')} &minus; "
                              f"{env.get('delta_psi_total')}"),
                result_name="dPSI traffic",
                result_value=env.get("delta_psi_traffic"),
                reference="AASHTO 1993 Guide, Table 3.1 (pdf_page 123, printed II-34)",
            ),
        ]
        sections.insert(0, CalcSection(
            title="Environmental Serviceability Loss", items=env_steps))
    basis = list(result.notes) + [f"WARNING: {w}" for w in result.warnings]
    if basis:
        sections.append(CalcSection(title="Basis and Assumptions",
                                    items=basis))
    if result.references:
        sections.append(CalcSection(
            title="Guide Provenance",
            items=["Digitized values traced to: "
                   + "; ".join(result.references)]))
    return sections


def get_figures(result, analysis=None):
    """Design-chart figures: the guide's nomograph relationships re-plotted
    from the digitized equations with the design solution overlaid, plus the
    section diagram and (when applicable) seasonal-MR and environmental-loss
    plots. Empty list when matplotlib is unavailable."""
    try:
        from calc_package.renderer import figure_to_base64
        import matplotlib.pyplot as plt
        from . import plots as _plots
    except ImportError:
        return []

    figures = []

    def _add(fig, title, caption, width=80):
        if fig is None:
            return
        figures.append(FigureData(
            title=title, image_base64=figure_to_base64(fig, dpi=150),
            caption=caption, width_percent=width))
        plt.close(fig)

    if _is_comparison(result):
        _add(_plots.plot_method_comparison(result),
             "Method comparison",
             "AASHTO 1993 vs UFC 3-250-01 sections on the same 18-kip "
             "traffic (two design bases, differences expected).", 70)
        return figures
    if _is_ufc_flexible(result):
        _add(_plots.plot_ufc_flexible_design_chart(result),
             "UFC flexible design chart",
             "Computed Figure E-1 (Corps CBR method) with the design point "
             "and provided thickness overlaid.")
        return figures
    if _is_ufc_rigid(result):
        _add(_plots.plot_ufc_rigid_design_chart(result),
             "UFC rigid design chart",
             "Computed Figure F-1 slice at the design flexural strength "
             "and k, with the required and provided slab overlaid.")
        return figures
    if isinstance(result, FlexiblePavementResult):
        _add(_plots.plot_flexible_design_chart(result),
             "Flexible design chart",
             "Computed Figure 3.1: W18 capacity vs SN from the digitized "
             "design equation for each foundation modulus, with the "
             "required-SN solutions and the provided SN overlaid.")
        _add(_plots.plot_layer_section(result),
             "Designed section",
             "Layered section with per-course thickness, layer coefficient, "
             "and drainage m.", 65)
        _add(_plots.plot_seasonal_mr(result),
             "Effective roadbed MR",
             "Seasonal MR and relative damage uf (Figure 2.3/2.4 "
             "procedure) with the effective design MR.")
    elif isinstance(result, RigidPavementResult):
        _add(_plots.plot_rigid_design_chart(result),
             "Rigid design chart",
             "Computed Figure 3.7: W18 capacity vs slab thickness from the "
             "digitized design equation, with the required and provided D "
             "overlaid.")
    _add(_plots.plot_environmental_loss(result),
         "Environmental serviceability loss",
         "Computed Figure 2.2: swelling / frost-heave serviceability loss "
         "vs time (Appendix G equations), the analysis period, and the "
         "design dPSI budget.")
    return figures
