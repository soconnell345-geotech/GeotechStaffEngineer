"""Calc-package template for pavement_design (AASHTO 1993).

Renders FlexiblePavementResult / RigidPavementResult into Mathcad-style
sections for ``calc_package.generate_calc_package``. UNITS: US customary
(the 1993 Guide's native units) -- unit strings say so explicitly.
"""

from calc_package.data_model import (CalcSection, CalcStep, CheckItem,
                                     InputItem, TableData)

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


def get_input_summary(result, analysis=None):
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
    return []
