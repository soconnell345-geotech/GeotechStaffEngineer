"""Rectangular RC section analysis via concreteproperties.

Interface units: dimensions mm, strengths MPa (library-native N/mm);
moments converted to kN*m and axial forces to kN on output.

Material model defaults (all overridable; see DESIGN.md):
- Ec = 4700*sqrt(f'c) MPa (ACI 318-19 Eq. 19.2.2.1.b, normal weight)
- rectangular stress block: alpha = 0.85, gamma = ACI beta1
  (0.85 down to 0.65 by 0.05 per 7 MPa above 28 MPa), eps_cu = 0.003
- modulus of rupture fr = 0.62*sqrt(f'c) MPa (ACI 318-19 Eq. 19.2.3.1)
- steel: elastic-perfectly-plastic, Es = 200 GPa

Capacities are NOMINAL (no phi factors applied).
"""

import math

from concrete_props_agent.concrete_utils import import_concreteproperties
from concrete_props_agent.results import RCSectionResult


def aci_beta1(fc_MPa: float) -> float:
    """ACI 318-19 Table 22.2.2.4.3 equivalent-stress-block beta1 (metric):
    0.85 for f'c <= 28; linear to 55; 0.65 for f'c >= 55."""
    if fc_MPa >= 55.0:
        return 0.65
    return max(0.65, min(0.85, 0.85 - 0.05 * (fc_MPa - 28.0) / 7.0))


def _bar_area(dia_mm: float) -> float:
    return math.pi * dia_mm ** 2 / 4.0


def analyze_rc_rectangle(
    b_mm: float,
    h_mm: float,
    fc_MPa: float,
    fy_MPa: float,
    n_bot: int,
    dia_bot_mm: float,
    cover_mm: float = 40.0,
    n_top: int = 0,
    dia_top_mm: float = 0.0,
    ec_MPa: float = None,
    es_MPa: float = 200e3,
    include_interaction: bool = False,
    n_interaction_points: int = 24,
) -> RCSectionResult:
    """Analyze a rectangular reinforced-concrete section.

    Parameters
    ----------
    b_mm, h_mm : float
        Width and overall depth (mm).
    fc_MPa : float
        Concrete cylinder strength f'c (MPa).
    fy_MPa : float
        Bar yield strength (MPa).
    n_bot, dia_bot_mm : int, float
        Bottom (tension for sagging) bar count and diameter.
    cover_mm : float
        CLEAR cover to the bar surface (both faces). Default 40.
    n_top, dia_top_mm : int, float
        Optional top steel.
    ec_MPa : float, optional
        Concrete modulus; default ACI 4700*sqrt(f'c).
    es_MPa : float
        Steel modulus. Default 200,000 MPa.
    include_interaction : bool
        Also compute the N-M interaction diagram. Default False.
    n_interaction_points : int
        Interaction diagram resolution. Default 24.

    Returns
    -------
    RCSectionResult
    """
    for name, v in (("b_mm", b_mm), ("h_mm", h_mm), ("fc_MPa", fc_MPa),
                    ("fy_MPa", fy_MPa), ("dia_bot_mm", dia_bot_mm)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be a positive number, got {v!r}")
    if n_bot < 1:
        raise ValueError(f"n_bot must be >= 1, got {n_bot}")
    if n_top and (dia_top_mm <= 0):
        raise ValueError("dia_top_mm required when n_top > 0")
    if cover_mm <= 0 or 2 * cover_mm + dia_bot_mm >= h_mm:
        raise ValueError("cover_mm inconsistent with section depth")

    (ConcreteSection, Concrete, SteelBar, ssp,
     concrete_rectangular_section) = import_concreteproperties()

    if ec_MPa is None:
        ec_MPa = 4700.0 * math.sqrt(fc_MPa)
    fr = 0.62 * math.sqrt(fc_MPa)
    gamma = aci_beta1(fc_MPa)

    concrete = Concrete(
        name=f"{fc_MPa:.0f} MPa",
        density=2.4e-6,
        stress_strain_profile=ssp.ConcreteLinear(elastic_modulus=ec_MPa),
        ultimate_stress_strain_profile=ssp.RectangularStressBlock(
            compressive_strength=fc_MPa, alpha=0.85, gamma=gamma,
            ultimate_strain=0.003),
        flexural_tensile_strength=fr,
        colour="lightgrey",
    )
    steel = SteelBar(
        name=f"fy {fy_MPa:.0f}",
        density=7.85e-6,
        stress_strain_profile=ssp.SteelElasticPlastic(
            yield_strength=fy_MPa, elastic_modulus=es_MPa,
            fracture_strain=0.05),
        colour="grey",
    )

    area_bot = _bar_area(dia_bot_mm)
    area_top = _bar_area(dia_top_mm) if n_top else 0.0
    geom = concrete_rectangular_section(
        b=b_mm, d=h_mm,
        dia_top=dia_top_mm if n_top else 0,
        area_top=area_top, n_top=n_top, c_top=cover_mm if n_top else 0,
        dia_bot=dia_bot_mm, area_bot=area_bot, n_bot=n_bot, c_bot=cover_mm,
        n_circle=4, conc_mat=concrete, steel_mat=steel)
    sec = ConcreteSection(geom)

    gross = sec.get_gross_properties()
    # transformed gross Ixx about the centroid
    tr = sec.get_transformed_gross_properties(elastic_modulus=ec_MPa)
    ixx_gross = float(tr.ixx_c)

    cracked = sec.calculate_cracked_properties(theta=0)
    cracked.calculate_transformed_properties(elastic_modulus=ec_MPa)
    ixx_cr = float(cracked.ixx_c_cr)
    m_cr = float(cracked.m_cr) / 1e6

    ult_pos = sec.ultimate_bending_capacity(theta=0)
    mn_pos = abs(float(ult_pos.m_x)) / 1e6

    mn_neg = None
    if n_top:
        ult_neg = sec.ultimate_bending_capacity(theta=math.pi)
        mn_neg = abs(float(ult_neg.m_x)) / 1e6

    interaction = []
    if include_interaction:
        mi = sec.moment_interaction_diagram(
            theta=0, n_points=n_interaction_points, progress_bar=False)
        n_list, m_list = mi.get_results_lists(moment="m_x")
        interaction = [(float(n) / 1e3, float(m) / 1e6)
                       for n, m in zip(n_list, m_list)]

    d_eff = h_mm - cover_mm - dia_bot_mm / 2.0
    return RCSectionResult(
        b_mm=b_mm, h_mm=h_mm, fc_MPa=fc_MPa, fy_MPa=fy_MPa, ec_MPa=ec_MPa,
        as_bot_mm2=n_bot * area_bot, as_top_mm2=n_top * area_top,
        d_eff_mm=d_eff,
        gross_area_mm2=float(gross.total_area),
        ixx_gross_mm4=ixx_gross,
        ixx_cracked_mm4=ixx_cr,
        m_cr_kNm=m_cr,
        mn_pos_kNm=mn_pos,
        mn_neg_kNm=mn_neg,
        interaction=interaction,
    )
