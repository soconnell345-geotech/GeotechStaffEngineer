"""
Reinforced concrete section design for a wall or slab strip, per unit
length of wall, on USACE EM 1110-2-2104, "Strength Design for Reinforced
Concrete Hydraulic Structures" (1 November 2023) (GM-312).

The manual is in US customary units, and so is this file: inches, square
inches per foot of wall, psi, kips per foot and kip-ft per foot, on a
design strip b = 12 in. :mod:`retaining_walls.structural` converts.

Every formula and limit here is cited to the printed page it was read
from. Two values the manual takes from ACI 318-19 without printing a rule
are used as its examples print them (the calls are in
:data:`PHI_BASIS` and :data:`BETA1_BASIS`); nothing else is from ACI.

The equations themselves are the reference library's
(``geotech_references.em_2104``), a shared library outside the rule
against cross-module imports, as ``pavement_design`` calls
``geotech_references.aashto_1993``.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

from geotech_references.em_2104 import design as _em_design
from geotech_references.em_2104 import flexure_axial as _em_flexure
from geotech_references.em_2104 import shear as _em_shear

EM = "EM 1110-2-2104 (1 November 2023)"

#: The design strip, one foot of wall (in.).
B_IN = 12.0

#: Deformed bars: diameter (in.) and area (sq in), FHWA GEC-10
#: (FHWA-NHI-18-024) Table 6-2, "Weights and dimensions of deformed bars
#: (customary)", p. 6-3. No. 4 to No. 11: bars above No. 11 "must not be
#: lap spliced" (EM 1110-2-2104 para. 2-8b, p. 6) and the manual's wall
#: examples use No. 6 and No. 11 (pp. 75, 85).
BARS = {
    "No. 4": (0.500, 0.20),
    "No. 5": (0.625, 0.31),
    "No. 6": (0.750, 0.44),
    "No. 7": (0.875, 0.60),
    "No. 8": (1.000, 0.79),
    "No. 9": (1.128, 1.00),
    "No. 10": (1.270, 1.27),
    "No. 11": (1.410, 1.56),
}
BAR_SIZES = tuple(BARS)
BARS_BASIS = ("FHWA GEC-10 (FHWA-NHI-18-024) Table 6-2, p. 6-3: nominal "
              "diameter and cross-sectional area of deformed bars")

#: Strength reduction factors (call 1 of the GM-312 brief).
PHI_FLEXURE = 0.90
PHI_SHEAR = 0.75
PHI_BASIS = (
    f"{EM} takes the resistance factor 'from ACI 318-19' (Eq. 3-1, p. 21) "
    "and prints the values only in its floodwall example, D-5 Step 3(b), "
    "p. 84: 'For shear; phi = 0.75 (ACI 318-19). For bending; phi = 0.90 "
    "(ACI 318-19)'")

BETA1_BASIS = (
    f"{EM} para. 4-1d(4), p. 30: 'Factor beta1 will be taken as specified "
    "in ACI 318-19'. The manual prints no rule, only 0.85 at f'c = 4,000 "
    "psi (pp. 52, 75) and 0.80 at 5,000 psi (p. 86); the value used is the "
    "reference library's aci_beta1 (ACI 318-19's, not printed in the "
    "manual), which gives those two. beta1 enters only the balanced steel "
    "ratio")

#: Maximum bar spacing, centre to centre (in.): EM 1110-2-2104 para. 2-6b,
#: p. 5, "must not exceed 12 in. (30 cm)".
MAX_SPACING_IN = 12.0
#: The spacings tried when the engineer gives none: 12 in., then 6 in., as
#: the manual's examples use them (D-3, p. 75: "No. 6 bars at 12-in. on
#: center"; D-5, p. 85: "No. 11 bars at 6 in. center to center").
DEFAULT_SPACINGS_IN = (12.0, 6.0)

#: Table 3-3, p. 26: maximum service stress in the reinforcement, flexure
#: and shear (psi).
SERVICE_STRESS_LIMIT_PSI = {"usual": 25000.0, "unusual": 35000.0}
#: Table 3-4, p. 26: single load factors, flexure and shear, fy = 60,000
#: psi (414 MPa) only.
SINGLE_LOAD_FACTOR = {"usual": 2.2, "unusual": 1.6}


def beta1_default(fc_psi: float) -> float:
    """beta1 where the engineer gives none (:data:`BETA1_BASIS`)."""
    return _em_flexure.aci_beta1(fc_psi, unit="psi")


def balanced_ratio(fc_psi: float, fy_psi: float, beta1: float) -> float:
    """The balanced steel ratio as EM 1110-2-2104 prints it (C-2, p. 53;
    D-3 Step 4, p. 75): rho_b = 0.85 beta1 (f'c / fy) [87,000 / (87,000 +
    fy)], stresses in psi."""
    return 0.85 * beta1 * fc_psi / fy_psi * 87000.0 / (87000.0 + fy_psi)


def hook_development_length(db_in: float, fc_psi: float,
                            fy_psi: float) -> Dict[str, Any]:
    """The development length of a standard hook as EM 1110-2-2104 prints
    it in D-5 Step 6 (pp. 86-87), "From ACI 318-19": ldh = [fy psi_e psi_c
    psi_r psi_o / (55 lambda sqrt(f'c))] db^1.5, with the factors the
    example takes: psi_e = 1.0 (uncoated bars), psi_c = f'c / 15,000 + 0.6
    "for f'c < 6,000 psi", psi_r = 1.0 (no confining reinforcement),
    psi_o = 1.0 ("for no. 11 and smaller bars with a side cover >= 6 db")
    and lambda = 1.0 (normal weight concrete).

    Returns ``{"ldh_in", "psi_c", "computed", "why"}``; at f'c of 6,000 psi
    or more the manual prints no psi_c, so ``ldh_in`` is None and ``why``
    says so."""
    if fc_psi >= 6000.0:
        return {"ldh_in": None, "psi_c": None, "computed": False,
                "why": (f"{EM} D-5 Step 6 (p. 87) prints the concrete "
                        "strength factor psi_c = f'c/15,000 + 0.6 'for "
                        f"f'c < 6,000 psi' only; at f'c = {fc_psi:,.0f} "
                        "psi it prints none, so the hook development "
                        "length is not computed")}
    psi_c = fc_psi / 15000.0 + 0.6
    ldh = fy_psi * psi_c / (55.0 * math.sqrt(fc_psi)) * db_in ** 1.5
    return {"ldh_in": ldh, "psi_c": psi_c, "computed": True, "why": ""}


def temperature_shrinkage_ratio(joint_spacing_ft: float) -> float:
    """Table 2-3, p. 7: the minimum temperature and shrinkage
    reinforcement ratio (Grade 60) by the length between control joints:
    0.003 below 30 ft (9 m), 0.004 from 30 to 40 ft (9 to 12 m), 0.005
    above 40 ft (12 m)."""
    if joint_spacing_ft < 30.0:
        return 0.003
    if joint_spacing_ft <= 40.0:
        return 0.004
    return 0.005


#: Para. 2-9a (pp. 6-7): "Generally ... no less than No. 4 bars at 12 in.
#: (30 cm) in each face"; para. 2-9b (p. 7): "need not exceed the area
#: equivalent to No. 9 bars at 12 in. (30 cm) in each face" (sq in/ft).
TS_FLOOR_IN2 = BARS["No. 4"][1]
TS_CAP_IN2 = BARS["No. 9"][1]


def temperature_shrinkage_area(ratio: float, thickness_in: float) -> float:
    """Temperature and shrinkage steel in EACH face (sq in per ft), para.
    2-9a and 2-9b (pp. 6-7): the ratio times the gross area, "half in
    each face", not less than No. 4 at 12 in. and not more than No. 9 at
    12 in."""
    area = ratio * thickness_in * B_IN / 2.0
    return min(max(area, TS_FLOOR_IN2), TS_CAP_IN2)


def table_2_1_cover_in(thickness_in: float,
                       against_ground: bool = False) -> Optional[float]:
    """Table 2-1, p. 5: minimum clear cover (in.). Unformed surfaces in
    contact with the foundation: 4 in. (10 cm). Formed and screeded
    surfaces not subject to cavitation or abrasion: 4 in. (10 cm) for a
    section 24 in. (61 cm) thick or more, 3 in. (7.5 cm) for one over 12
    in. (30 cm) and under 24 in.; for 12 in. or less the table says
    "According to ACI 318-19" and this returns None."""
    if against_ground:
        return 4.0
    if thickness_in >= 24.0:
        return 4.0
    if thickness_in > 12.0:
        return 3.0
    return None


def service_steel_stress(M_kipft: float, As_in2: float, d_in: float,
                         modular_ratio: float) -> Dict[str, float]:
    """The service stress in the tension steel of a cracked, singly
    reinforced strip, as D-3 Step 5 and D-5 Step 5 work it (pp. 75-76,
    86): the neutral axis depth x from b x^2 / 2 = n As (d - x), then fs =
    M / [As (d - x/3)] (p. 86). Returns ``{"x_in", "fs_psi"}``."""
    nAs = modular_ratio * As_in2
    x = (-nAs + math.sqrt(nAs ** 2 + 2.0 * B_IN * nAs * d_in)) / B_IN
    fs_ksi = M_kipft * 12.0 / (As_in2 * (d_in - x / 3.0))
    return {"x_in": x, "fs_psi": fs_ksi * 1000.0}


def crack_control_spacing(fs_psi: float, cover_in: float) -> float:
    """The maximum bar spacing for crack control as the manual's examples
    print it, "according to ACI 318-19" (D-3 Step 5(b), p. 76; D-5 Step
    5(b), p. 86): s_max = 15 (40,000 / fs) - 2.5 cc (in.), with cc the
    lesser of 2.5 in. and the actual clear cover (para. 2-6b(1), p. 5)."""
    cc = min(2.5, cover_in)
    return 15.0 * (40000.0 / fs_psi) - 2.5 * cc


def _check(name: str, demand, capacity, unit: str, passes: bool,
           reference: str, required: bool = True,
           note: str = "") -> Dict[str, Any]:
    c = {"name": name, "demand": demand, "capacity": capacity,
         "unit": unit, "passes": bool(passes), "required": required,
         "reference": reference}
    if note:
        c["note"] = note
    return c


def _trial(size: str, spacing_in: float, *, Mu_kipft: float, Vu_kip: float,
           h_in: float, cover_in: float, fc_psi: float, fy_psi: float,
           beta1: float, As_min_in2: float, d_in: Optional[float],
           service_moments: Sequence[Dict[str, Any]],
           modular_ratio: Optional[float],
           max_aggregate_in: Optional[float]) -> Dict[str, Any]:
    """One bar size at one spacing on the section: the steel, the
    capacities and every check."""
    db, Ab = BARS[size]
    d = (h_in - cover_in - db / 2.0) if d_in is None else d_in
    if d <= 0:
        raise ValueError(
            f"the section is {h_in:.2f} in. thick with {cover_in:.2f} in. "
            f"of cover: no effective depth is left for a {size} bar")
    fc_ksi, fy_ksi = fc_psi / 1000.0, fy_psi / 1000.0

    # Eq. D-8, D-9 (p. 72), no axial load
    Mn_kin = Mu_kipft * 12.0 / PHI_FLEXURE
    try:
        sized = _em_design.design_singly_reinforced(
            max(Mn_kin, 0.0), 0.0, d, h_in, B_IN, fc_ksi, fy_ksi)
        Ku, As_flex = sized["ku"], sized["as_required"]
        too_thin = False
    except ValueError:
        Ku, As_flex, too_thin = None, math.inf, True
    As_req = max(As_flex, As_min_in2)
    As_prov = Ab * B_IN / spacing_in

    # Eq. B-49, B-50 (p. 51)
    a = As_prov * fy_psi / (0.85 * fc_psi * B_IN)
    phi_Mn = PHI_FLEXURE * As_prov * fy_psi * (d - a / 2.0) / 12000.0
    rho = As_prov / (B_IN * d)
    rho_b = balanced_ratio(fc_psi, fy_psi, beta1)

    # Eq. 5-1 (p. 35), no axial load, as D-5 Step 3(h) works it (p. 85);
    # the US form on psi and inches, and the printed SI form beside it
    Vc = _em_shear.shear_capacity_one_way_slab(
        fc_psi, 0.0, B_IN * h_in, B_IN, d, unit="us")["vc"] / 1000.0
    phi_Vc = PHI_SHEAR * Vc
    Vc_si_N = _em_shear.shear_capacity_one_way_slab(
        fc_psi * 6.894757e-3, 0.0, B_IN * h_in * 645.16, B_IN * 25.4,
        d * 25.4, unit="si")["vc"]
    phi_Vc_si_form = PHI_SHEAR * Vc_si_N / 4448.222

    checks = [
        _check("flexure, phi Mn >= Mu", Mu_kipft, phi_Mn, "kip-ft/ft",
               phi_Mn >= Mu_kipft - 1e-9,
               f"{EM} Eq. B-49, B-50 (p. 51), phi = 0.90 (p. 84)"),
        _check("steel provided >= steel required", As_req, As_prov,
               "sq in/ft", As_prov >= As_req - 1e-9,
               f"{EM} Eq. D-8, D-9 (p. 72); not less than the shrinkage "
               "and temperature steel (para. 2-9g, p. 8)",
               note=("the section is too thin for the moment: Eq. D-8's "
                     "radicand is negative" if too_thin else "")),
        _check("shear, phi Vc >= Vu (no shear reinforcement)", Vu_kip,
               phi_Vc, "kips/ft", phi_Vc >= Vu_kip - 1e-9,
               f"{EM} Eq. 5-1 and para. 5-2b (p. 35), phi = 0.75 (p. 84)"),
        _check("steel ratio <= 0.50 rho_b", rho, 0.50 * rho_b, "",
               rho <= 0.50 * rho_b + 1e-12,
               f"{EM} para. 3-6 (p. 27); rho_b as printed on pp. 53, 75"),
        _check("steel ratio <= 0.25 rho_b (recommended)", rho,
               0.25 * rho_b, "", rho <= 0.25 * rho_b + 1e-12,
               f"{EM} para. 3-4b(2) (p. 27): 'It is recommended to "
               "generally design within this limit'", required=False),
        _check("bar spacing <= 12 in.", spacing_in, MAX_SPACING_IN, "in.",
               spacing_in <= MAX_SPACING_IN + 1e-9,
               f"{EM} para. 2-6b (p. 5)"),
    ]
    clear = spacing_in - db
    least = 1.5 * db
    words = "1.5 bar diameters"
    if max_aggregate_in is not None and 1.5 * max_aggregate_in > least:
        least, words = 1.5 * max_aggregate_in, "1.5 x the largest aggregate"
    checks.append(_check(
        f"clear distance between bars >= {words}", least, clear, "in.",
        clear >= least - 1e-9, f"{EM} para. 2-6a (p. 4)"))
    least_cover, cover_words = 2.5 * db, "2.5 bar diameters"
    if max_aggregate_in is not None and 1.5 * max_aggregate_in > least_cover:
        least_cover = 1.5 * max_aggregate_in
        cover_words = "1.5 x the largest aggregate"
    checks.append(_check(
        f"clear cover >= {cover_words}", least_cover, cover_in, "in.",
        cover_in >= least_cover - 1e-9, f"{EM} Table 2-1, Note (p. 5)"))

    # Serviceability: the reinforcement stress (Table 3-3) where a modular
    # ratio is given, and the crack-control spacing (para. 2-6b)
    service = []
    for sm in service_moments:
        cat, M = sm["category"], sm["M_kipft"]
        row = {"case": sm.get("case"), "category": cat, "M_kipft": M,
               "fs_psi": None, "x_in": None, "s_max_in": None,
               "stress_basis": None}
        limit = SERVICE_STRESS_LIMIT_PSI[cat]
        if modular_ratio is not None and M > 0:
            st = service_steel_stress(M, As_prov, d, modular_ratio)
            row.update(fs_psi=st["fs_psi"], x_in=st["x_in"],
                       stress_basis="actual")
            checks.append(_check(
                f"service steel stress, {cat} loads (case {row['case']})",
                st["fs_psi"] / 1000.0, limit / 1000.0, "ksi",
                st["fs_psi"] <= limit + 1e-6,
                f"{EM} Table 3-3 (p. 26); fs as D-5 Step 5 works it "
                "(p. 86)"))
            row["s_max_in"] = crack_control_spacing(st["fs_psi"], cover_in)
        elif modular_ratio is None and cat == "unusual":
            row.update(fs_psi=limit, stress_basis="limit")
            row["s_max_in"] = crack_control_spacing(limit, cover_in)
        if row["s_max_in"] is not None:
            how = ("the actual service stress" if row["stress_basis"]
                   == "actual" else "the 35 ksi service stress limit of "
                   "Table 3-3 (no modular ratio given for the actual "
                   "stress)")
            checks.append(_check(
                f"crack-control spacing, {cat} loads (case {row['case']})",
                spacing_in, row["s_max_in"], "in.",
                spacing_in <= row["s_max_in"] + 1e-9,
                f"{EM} para. 2-6b (p. 5) with s_max = 15 (40,000 / fs) - "
                f"2.5 cc as printed in D-3 Step 5(b) (p. 76), on {how}"))
        service.append(row)

    selectable = all(c["passes"] for c in checks
                     if c["required"] and (
                         c["name"].startswith(("steel provided",
                                               "service steel stress",
                                               "crack-control"))))
    return {
        "bar_size": size, "bar_diameter_in": db, "bar_area_in2": Ab,
        "spacing_in": spacing_in, "d_in": d, "h_in": h_in,
        "cover_in": cover_in, "Mu_kipft": Mu_kipft, "Vu_kip": Vu_kip,
        "Mn_required_kin": Mn_kin, "Ku": Ku,
        "As_flexure_in2": None if too_thin else As_flex,
        "As_min_in2": As_min_in2,
        "As_required_in2": None if too_thin else As_req,
        "As_provided_in2": As_prov, "a_in": a, "phi_Mn_kipft": phi_Mn,
        "rho": rho, "rho_b": rho_b, "beta1": beta1,
        "Vc_kip": Vc, "phi_Vc_kip": phi_Vc,
        "phi_Vc_SI_form_kip": phi_Vc_si_form,
        "service": service, "checks": checks, "too_thin": too_thin,
        "selectable": selectable,
        "passes": all(c["passes"] for c in checks if c["required"]),
    }


def design_section(Mu_kipft: float, Vu_kip: float, h_in: float,
                   cover_in: float, fc_psi: float, fy_psi: float = 60000.0,
                   *, beta1: Optional[float] = None,
                   As_min_in2: float = 0.0,
                   bar_size: Optional[str] = None,
                   spacing_in: Optional[float] = None,
                   d_in: Optional[float] = None,
                   service_moments: Sequence[Dict[str, Any]] = (),
                   modular_ratio: Optional[float] = None,
                   max_aggregate_in: Optional[float] = None
                   ) -> Dict[str, Any]:
    """Design a singly reinforced strip, one foot of wall wide, for a
    factored moment and shear (EM 1110-2-2104 App. D-2, pp. 70-72, as its
    examples D-3 and D-5 work it, pp. 73-76 and 84-87).

    Parameters
    ----------
    Mu_kipft, Vu_kip : float
        Factored moment (kip-ft/ft) and shear (kips/ft) at the section.
    h_in, cover_in : float
        Section thickness and clear cover to the tension steel (in.).
    fc_psi, fy_psi : float
        Concrete strength and steel yield (psi).
    beta1 : float, optional
        None: :func:`beta1_default` (:data:`BETA1_BASIS`).
    As_min_in2 : float
        The least steel (sq in/ft): the shrinkage and temperature steel
        for the face (para. 2-9g, p. 8).
    bar_size, spacing_in : optional
        The bar ("No. 4" to "No. 11") and its spacing (in.) where the
        engineer fixes them. Left out, the choice follows the manual's
        examples: the smallest bar that gives the steel at 12 in.; if No.
        11 does not, the smallest at 6 in. A spacing that fails the
        crack-control spacing or a service stress is passed over.
    d_in : float, optional
        The effective depth (in.) where it is fixed (the examples round
        it); None: h - cover - bar diameter / 2, the primary steel in the
        outer layer (D-3 Step 4, p. 75; D-5 Step 3(d), p. 84).
    service_moments : sequence of dict
        ``{"case", "category" ("usual" or "unusual"), "M_kipft"}``: the
        unfactored moments for the serviceability checks.
    modular_ratio : float, optional
        n = Es / Ec for the actual service stress. The manual prints no
        rule for it (its examples take 29,000 / 3,605 at 4,000 psi, p.
        75), so it is the engineer's. Without it no stress is computed
        and the crack-control spacing for unusual loads is on the 35 ksi
        limit (para. 2-6b(3), p. 5).
    max_aggregate_in : float, optional
        The largest aggregate (in.), for the clear distance and the cover
        (para. 2-6a, p. 4; Table 2-1 Note, p. 5).

    Returns
    -------
    dict
        The chosen trial (:func:`_trial`), with ``"chosen_by"``.
    """
    if bar_size is not None and bar_size not in BARS:
        raise ValueError(f"bar size must be one of {list(BARS)}, got "
                         f"{bar_size!r}")
    if spacing_in is not None and spacing_in <= 0:
        raise ValueError(f"bar spacing must be positive, got {spacing_in}")
    if beta1 is None:
        beta1 = beta1_default(fc_psi)
    sizes = (bar_size,) if bar_size is not None else BAR_SIZES
    spacings = ((spacing_in,) if spacing_in is not None
                else DEFAULT_SPACINGS_IN)
    kw = dict(Mu_kipft=Mu_kipft, Vu_kip=Vu_kip, h_in=h_in,
              cover_in=cover_in, fc_psi=fc_psi, fy_psi=fy_psi, beta1=beta1,
              As_min_in2=As_min_in2, d_in=d_in,
              service_moments=service_moments,
              modular_ratio=modular_ratio,
              max_aggregate_in=max_aggregate_in)
    chosen = last = None
    for s in spacings:
        for size in sizes:
            last = _trial(size, s, **kw)
            if last["selectable"]:
                chosen = last
                break
        if chosen is not None:
            break
    if chosen is None:
        chosen = last
    given = [n for n, v in (("size", bar_size), ("spacing", spacing_in))
             if v is not None]
    chosen["chosen_by"] = (
        "the engineer's bar size and spacing" if len(given) == 2 else
        f"the engineer's bar {given[0]}; the other by the manual's "
        "examples" if given else
        "the manual's examples: the smallest bar that gives the steel at "
        "12 in.; if No. 11 does not, at 6 in. (D-3, p. 75; D-5, p. 85)")
    return chosen


def temperature_shrinkage_bar(As_in2: float) -> Dict[str, Any]:
    """The smallest bar at 12 in. that gives the temperature and shrinkage
    steel of one face (para. 2-9a, pp. 6-7: No. 4 at 12 in. at least)."""
    for size in BAR_SIZES:
        db, Ab = BARS[size]
        if Ab >= As_in2 - 1e-9:
            return {"bar_size": size, "bar_diameter_in": db,
                    "bar_area_in2": Ab, "spacing_in": 12.0,
                    "As_required_in2": As_in2, "As_provided_in2": Ab}
    raise ValueError(f"no bar gives {As_in2:.2f} sq in/ft at 12 in.")


__all__: List[str] = [
    "BARS", "BAR_SIZES", "PHI_FLEXURE", "PHI_SHEAR", "PHI_BASIS",
    "BETA1_BASIS", "SINGLE_LOAD_FACTOR", "SERVICE_STRESS_LIMIT_PSI",
    "beta1_default", "balanced_ratio", "hook_development_length",
    "temperature_shrinkage_ratio", "temperature_shrinkage_area",
    "table_2_1_cover_in", "service_steel_stress", "crack_control_spacing",
    "design_section", "temperature_shrinkage_bar",
]
