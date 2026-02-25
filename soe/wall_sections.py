"""
Steel section databases and selection for SOE wall design.

Includes HP shapes (soldier piles), sheet pile sections, and W shapes
(wales/struts) with manufacturer data.

All section properties in US customary (in, in², in³, in⁴, lb/ft) as
commonly published. Conversion to SI provided where needed.

References:
    AISC Steel Construction Manual, 16th Edition
    Nucor Skyline Sheet Pile Catalog
    ArcelorMittal Piling Handbook
"""

from typing import Optional, Dict, List


# ============================================================================
# HP Sections (soldier piles) — AISC 16th Ed
# ============================================================================
# Properties: d (in), bf (in), tf (in), tw (in), A (in²),
#             Ix (in⁴), Sx (in³), Zx (in³), weight (lb/ft)

_HP_SECTIONS = {
    "HP8x36": {
        "d": 8.02, "bf": 8.155, "tf": 0.445, "tw": 0.445,
        "A": 10.6, "Ix": 119, "Sx": 29.8, "Zx": 33.6, "weight": 36,
    },
    "HP10x42": {
        "d": 9.70, "bf": 10.075, "tf": 0.420, "tw": 0.415,
        "A": 12.4, "Ix": 210, "Sx": 43.4, "Zx": 48.3, "weight": 42,
    },
    "HP10x57": {
        "d": 9.99, "bf": 10.225, "tf": 0.565, "tw": 0.565,
        "A": 16.8, "Ix": 294, "Sx": 58.8, "Zx": 66.5, "weight": 57,
    },
    "HP12x53": {
        "d": 11.78, "bf": 12.045, "tf": 0.435, "tw": 0.435,
        "A": 15.5, "Ix": 393, "Sx": 66.8, "Zx": 74.0, "weight": 53,
    },
    "HP12x63": {
        "d": 11.94, "bf": 12.125, "tf": 0.515, "tw": 0.515,
        "A": 18.4, "Ix": 472, "Sx": 79.1, "Zx": 88.3, "weight": 63,
    },
    "HP12x74": {
        "d": 12.13, "bf": 12.215, "tf": 0.610, "tw": 0.605,
        "A": 21.8, "Ix": 569, "Sx": 93.8, "Zx": 105, "weight": 74,
    },
    "HP12x84": {
        "d": 12.28, "bf": 12.295, "tf": 0.685, "tw": 0.685,
        "A": 24.6, "Ix": 650, "Sx": 106, "Zx": 119, "weight": 84,
    },
    "HP14x73": {
        "d": 13.61, "bf": 14.585, "tf": 0.505, "tw": 0.505,
        "A": 21.4, "Ix": 729, "Sx": 107, "Zx": 118, "weight": 73,
    },
    "HP14x89": {
        "d": 13.83, "bf": 14.695, "tf": 0.615, "tw": 0.615,
        "A": 26.1, "Ix": 904, "Sx": 131, "Zx": 146, "weight": 89,
    },
    "HP14x102": {
        "d": 14.01, "bf": 14.785, "tf": 0.705, "tw": 0.705,
        "A": 30.0, "Ix": 1050, "Sx": 150, "Zx": 168, "weight": 102,
    },
    "HP14x117": {
        "d": 14.21, "bf": 14.885, "tf": 0.805, "tw": 0.805,
        "A": 34.4, "Ix": 1220, "Sx": 172, "Zx": 194, "weight": 117,
    },
}


# ============================================================================
# Sheet Pile Sections — Nucor Skyline / ArcelorMittal
# ============================================================================
# Properties per meter of wall: Sx (in³/ft → cm³/m), Ix (in⁴/ft → cm⁴/m),
# width (mm per pair), weight (kg/m² of wall)

_SHEET_PILE_SECTIONS = {
    "PZ22": {
        "Sx_in3_per_ft": 18.1, "Ix_in4_per_ft": 84.4,
        "width_mm": 559, "weight_kg_per_m2": 107,
        "Sx_cm3_per_m": 974, "Ix_cm4_per_m": 11520,
    },
    "PZ27": {
        "Sx_in3_per_ft": 30.2, "Ix_in4_per_ft": 184.2,
        "width_mm": 457, "weight_kg_per_m2": 131,
        "Sx_cm3_per_m": 1625, "Ix_cm4_per_m": 25150,
    },
    "PZ35": {
        "Sx_in3_per_ft": 48.5, "Ix_in4_per_ft": 361.2,
        "width_mm": 575, "weight_kg_per_m2": 161,
        "Sx_cm3_per_m": 2610, "Ix_cm4_per_m": 49310,
    },
    "PZ40": {
        "Sx_in3_per_ft": 60.7, "Ix_in4_per_ft": 490.8,
        "width_mm": 500, "weight_kg_per_m2": 186,
        "Sx_cm3_per_m": 3265, "Ix_cm4_per_m": 67020,
    },
    "AZ12-770": {
        "Sx_in3_per_ft": 15.8, "Ix_in4_per_ft": 72.8,
        "width_mm": 770, "weight_kg_per_m2": 78,
        "Sx_cm3_per_m": 850, "Ix_cm4_per_m": 9940,
    },
    "AZ18-700": {
        "Sx_in3_per_ft": 28.2, "Ix_in4_per_ft": 155.0,
        "width_mm": 700, "weight_kg_per_m2": 106,
        "Sx_cm3_per_m": 1517, "Ix_cm4_per_m": 21160,
    },
    "AZ26-700": {
        "Sx_in3_per_ft": 41.5, "Ix_in4_per_ft": 286.0,
        "width_mm": 700, "weight_kg_per_m2": 134,
        "Sx_cm3_per_m": 2233, "Ix_cm4_per_m": 39050,
    },
}


# ============================================================================
# W Sections (common wale/strut sizes) — AISC 16th Ed
# ============================================================================

_W_SECTIONS = {
    "W14x22": {"d": 13.74, "A": 6.49, "Ix": 199, "Sx": 29.0, "Zx": 33.2, "weight": 22},
    "W14x30": {"d": 13.84, "A": 8.85, "Ix": 291, "Sx": 42.0, "Zx": 47.3, "weight": 30},
    "W14x43": {"d": 13.66, "A": 12.6, "Ix": 428, "Sx": 62.6, "Zx": 69.6, "weight": 43},
    "W14x61": {"d": 13.89, "A": 17.9, "Ix": 640, "Sx": 92.1, "Zx": 102, "weight": 61},
    "W14x82": {"d": 14.31, "A": 24.0, "Ix": 881, "Sx": 123, "Zx": 139, "weight": 82},
    "W21x44": {"d": 20.66, "A": 13.0, "Ix": 843, "Sx": 81.6, "Zx": 95.4, "weight": 44},
    "W21x62": {"d": 20.99, "A": 18.3, "Ix": 1330, "Sx": 127, "Zx": 144, "weight": 62},
    "W21x83": {"d": 21.43, "A": 24.3, "Ix": 1830, "Sx": 171, "Zx": 196, "weight": 83},
    "W24x55": {"d": 23.57, "A": 16.2, "Ix": 1350, "Sx": 114, "Zx": 134, "weight": 55},
    "W24x76": {"d": 23.92, "A": 22.4, "Ix": 2100, "Sx": 176, "Zx": 200, "weight": 76},
    "W24x104": {"d": 24.06, "A": 30.6, "Ix": 3100, "Sx": 258, "Zx": 289, "weight": 104},
}


# ============================================================================
# Section selection functions
# ============================================================================

def select_hp_section(required_Sx_cm3: float,
                      Fy_MPa: float = 345.0) -> Optional[Dict]:
    """Select the lightest HP section with adequate section modulus.

    Parameters
    ----------
    required_Sx_cm3 : float
        Required section modulus (cm³).
    Fy_MPa : float
        Steel yield strength (MPa). Default 345.

    Returns
    -------
    dict or None
        Selected section with all properties plus "name" and
        "Sx_cm3" (converted). None if no section is adequate.
    """
    # Convert required Sx from cm³ to in³ (1 in³ = 16.387 cm³)
    required_Sx_in3 = required_Sx_cm3 / 16.387

    best = None
    for name, props in _HP_SECTIONS.items():
        if props["Sx"] >= required_Sx_in3:
            if best is None or props["weight"] < best["weight"]:
                best = {**props, "name": name, "Sx_cm3": props["Sx"] * 16.387}

    return best


def select_sheet_pile(required_Sx_cm3_per_m: float) -> Optional[Dict]:
    """Select the lightest sheet pile section with adequate section modulus.

    Parameters
    ----------
    required_Sx_cm3_per_m : float
        Required section modulus per meter of wall (cm³/m).

    Returns
    -------
    dict or None
        Selected section with all properties plus "name".
        None if no section is adequate.
    """
    best = None
    for name, props in _SHEET_PILE_SECTIONS.items():
        if props["Sx_cm3_per_m"] >= required_Sx_cm3_per_m:
            if best is None or props["weight_kg_per_m2"] < best["weight_kg_per_m2"]:
                best = {**props, "name": name}

    return best


def select_w_section(required_Sx_cm3: float) -> Optional[Dict]:
    """Select the lightest W section for wales/struts.

    Parameters
    ----------
    required_Sx_cm3 : float
        Required section modulus (cm³).

    Returns
    -------
    dict or None
        Selected section. None if no section is adequate.
    """
    required_Sx_in3 = required_Sx_cm3 / 16.387

    best = None
    for name, props in _W_SECTIONS.items():
        if props["Sx"] >= required_Sx_in3:
            if best is None or props["weight"] < best["weight"]:
                best = {**props, "name": name, "Sx_cm3": props["Sx"] * 16.387}

    return best


def check_flexural_demand(Sx_cm3: float, M_demand_kNm: float,
                          Fy_MPa: float = 345.0) -> Dict:
    """Check flexural utilization ratio for a steel section.

    Parameters
    ----------
    Sx_cm3 : float
        Section modulus (cm³).
    M_demand_kNm : float
        Maximum bending moment demand (kN·m).
    Fy_MPa : float
        Steel yield strength (MPa). Default 345.

    Returns
    -------
    dict
        "Fb_MPa" (allowable bending stress), "fb_MPa" (actual stress),
        "utilization_ratio", "adequate" (bool).
    """
    Fb = 0.66 * Fy_MPa  # ASD allowable bending stress
    Sx_m3 = Sx_cm3 * 1e-6  # cm³ to m³
    M_Nm = M_demand_kNm * 1000.0  # kN·m to N·m
    fb = M_Nm / Sx_m3 / 1e6 if Sx_m3 > 0 else float("inf")  # MPa

    ratio = fb / Fb if Fb > 0 else float("inf")

    return {
        "Fb_MPa": round(Fb, 1),
        "fb_MPa": round(fb, 1),
        "utilization_ratio": round(ratio, 3),
        "adequate": ratio <= 1.0,
    }


def list_hp_sections() -> List[str]:
    """Return sorted list of available HP section names."""
    return sorted(_HP_SECTIONS.keys())


def list_sheet_pile_sections() -> List[str]:
    """Return sorted list of available sheet pile section names."""
    return sorted(_SHEET_PILE_SECTIONS.keys())
