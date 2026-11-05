"""
LRFD resistance factors for drilled shaft design per AASHTO.

References:
    AASHTO LRFD Bridge Design Specifications, 9th Ed., Table 10.5.5.2.4-1
    FHWA GEC-10 (FHWA-NHI-18-024), Table 8-4, p. 8-14
"""

from typing import Any, Dict, Optional

from drilled_shaft.results import DrillShaftResult


# Resistance factors for drilled shafts as GEC-10 Table 8-4 (p. 8-14) prints
# them. Side resistance in rock (Eq. 10-21) is 0.50 in the table; the table
# governs over the 0.55 in the text of p. 10-21 (owner, GM-278).
RESISTANCE_FACTORS = {
    "side_cohesive": 0.45,
    "side_cohesionless": 0.55,
    "side_rock": 0.50,
    "tip_cohesive": 0.40,
    "tip_cohesionless": 0.50,
    "tip_rock": 0.50,
    "uplift_cohesive": 0.35,
    "uplift_cohesionless": 0.45,
    "uplift_rock": 0.40,
}


def get_resistance_factor(component: str) -> float:
    """Get the AASHTO LRFD resistance factor for a component.

    Parameters
    ----------
    component : str
        One of: "side_cohesive", "side_cohesionless", "side_rock",
        "tip_cohesive", "tip_cohesionless", "tip_rock",
        "uplift_cohesive", "uplift_cohesionless", "uplift_rock".

    Returns
    -------
    float
        Resistance factor phi.
    """
    if component not in RESISTANCE_FACTORS:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Valid: {list(RESISTANCE_FACTORS.keys())}"
        )
    return RESISTANCE_FACTORS[component]


def apply_lrfd(result: DrillShaftResult,
               tip_soil_type: Optional[str] = None) -> Dict[str, Any]:
    """Apply LRFD resistance factors to drilled shaft results.

    Parameters
    ----------
    result : DrillShaftResult
        ASD analysis results.
    tip_soil_type : str, optional
        Soil type at tip: "cohesive", "cohesionless", or "rock". Left out,
        the soil type of the base the analysis computed
        (``result.base_steps``); given, it must be that one, since each
        base factor of Table 8-4 goes with its own method (GM-244 review:
        a tip in rock was factored at clay's 0.40 by the old default
        "cohesive"). A result with no base steps needs it.

    Returns
    -------
    dict
        Factored resistances with keys:
        - phi_Qs_clay_kN, phi_Qs_sand_kN, phi_Qs_rock_kN
        - phi_Qt_kN (with the base in rock and the shaft's concrete_fc
          given, the smaller of the geotechnical 0.50 Q_tip and the
          concrete's bearing 0.70 [0.85 f'c A1 m], GEC-10 p. 10-25 and
          Eq. 12-7; then also phi_Qt_geotechnical_kN,
          phi_Pn_concrete_bearing_kN and phi_Qt_governed_by)
        - phi_Qn_kN (total factored resistance)
    """
    actual = (result.base_steps or {}).get("soil_type")
    if tip_soil_type is None:
        if actual is None:
            raise ValueError(
                "tip_soil_type is needed: the result does not say what soil "
                "the base is in")
        tip_soil_type = actual
    elif actual is not None and tip_soil_type != actual:
        raise ValueError(
            f"tip_soil_type '{tip_soil_type}' is not the soil the base was "
            f"computed in ('{actual}'): GEC-10 Table 8-4 (p. 8-14) gives "
            "each base resistance its own factor by the method used (0.40 "
            "cohesive, 0.50 cohesionless and rock). Leave tip_soil_type "
            "out to take the analysis's own")
    phi_Qs_clay = result.Q_side_clay * RESISTANCE_FACTORS["side_cohesive"]
    phi_Qs_sand = result.Q_side_sand * RESISTANCE_FACTORS["side_cohesionless"]
    phi_Qs_rock = result.Q_side_rock * RESISTANCE_FACTORS["side_rock"]

    tip_key = f"tip_{tip_soil_type}"
    if tip_key not in RESISTANCE_FACTORS:
        raise ValueError(
            f"unknown tip_soil_type '{tip_soil_type}'. Allowed: "
            "['cohesive', 'cohesionless', 'rock']")
    phi_Qt = result.Q_tip * RESISTANCE_FACTORS[tip_key]
    # GEC-10 p. 10-25: with the base in rock the factored base is the
    # smaller of the geotechnical and the concrete's bearing (Eq. 12-7,
    # phi 0.70), where concrete_fc was given (GM-233)
    concrete = (result.base_steps or {}).get("concrete_bearing")
    extra = {}
    if concrete is not None and tip_key == "tip_rock":
        extra = {"phi_Qt_geotechnical_kN": round(phi_Qt, 1),
                 "phi_Pn_concrete_bearing_kN": round(concrete["phi_Pn_kN"], 1),
                 "phi_concrete_bearing": concrete["phi_concrete"]}
        if concrete["phi_Pn_kN"] < phi_Qt:
            phi_Qt = concrete["phi_Pn_kN"]
            extra["phi_Qt_governed_by"] = "concrete bearing (Eq. 12-7)"
        else:
            extra["phi_Qt_governed_by"] = "geotechnical"

    phi_Qn = phi_Qs_clay + phi_Qs_sand + phi_Qs_rock + phi_Qt

    return {**extra,
        "phi_Qs_clay_kN": round(phi_Qs_clay, 1),
        "phi_Qs_sand_kN": round(phi_Qs_sand, 1),
        "phi_Qs_rock_kN": round(phi_Qs_rock, 1),
        "phi_Qt_kN": round(phi_Qt, 1),
        "phi_Qn_kN": round(phi_Qn, 1),
        "phi_side_clay": RESISTANCE_FACTORS["side_cohesive"],
        "phi_side_sand": RESISTANCE_FACTORS["side_cohesionless"],
        "phi_side_rock": RESISTANCE_FACTORS["side_rock"],
        "phi_tip": RESISTANCE_FACTORS[tip_key],
    }
