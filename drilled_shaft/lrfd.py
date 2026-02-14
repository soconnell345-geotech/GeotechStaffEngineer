"""
LRFD resistance factors for drilled shaft design per AASHTO.

References:
    AASHTO LRFD Bridge Design Specifications, 9th Ed., Table 10.5.5.2.4-1
    FHWA GEC-10, Table 13-1
"""

from typing import Dict, Any

from drilled_shaft.results import DrillShaftResult


# AASHTO resistance factors for drilled shafts
RESISTANCE_FACTORS = {
    "side_cohesive": 0.45,
    "side_cohesionless": 0.55,
    "side_rock": 0.55,
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
               tip_soil_type: str = "cohesive") -> Dict[str, Any]:
    """Apply LRFD resistance factors to drilled shaft results.

    Parameters
    ----------
    result : DrillShaftResult
        ASD analysis results.
    tip_soil_type : str, optional
        Soil type at tip: "cohesive", "cohesionless", or "rock".
        Default "cohesive".

    Returns
    -------
    dict
        Factored resistances with keys:
        - phi_Qs_clay_kN, phi_Qs_sand_kN, phi_Qs_rock_kN
        - phi_Qt_kN
        - phi_Qn_kN (total factored resistance)
    """
    phi_Qs_clay = result.Q_side_clay * RESISTANCE_FACTORS["side_cohesive"]
    phi_Qs_sand = result.Q_side_sand * RESISTANCE_FACTORS["side_cohesionless"]
    phi_Qs_rock = result.Q_side_rock * RESISTANCE_FACTORS["side_rock"]

    tip_key = f"tip_{tip_soil_type}"
    phi_Qt = result.Q_tip * RESISTANCE_FACTORS.get(tip_key, 0.40)

    phi_Qn = phi_Qs_clay + phi_Qs_sand + phi_Qs_rock + phi_Qt

    return {
        "phi_Qs_clay_kN": round(phi_Qs_clay, 1),
        "phi_Qs_sand_kN": round(phi_Qs_sand, 1),
        "phi_Qs_rock_kN": round(phi_Qs_rock, 1),
        "phi_Qt_kN": round(phi_Qt, 1),
        "phi_Qn_kN": round(phi_Qn, 1),
        "phi_side_clay": RESISTANCE_FACTORS["side_cohesive"],
        "phi_side_sand": RESISTANCE_FACTORS["side_cohesionless"],
        "phi_side_rock": RESISTANCE_FACTORS["side_rock"],
        "phi_tip": RESISTANCE_FACTORS.get(tip_key, 0.40),
    }
