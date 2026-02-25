"""
Ground anchor design for anchored excavation walls.

Implements design procedures from FHWA-IF-99-015 (GEC-4) and
PTI DC35.1 for prestressed ground anchors including:
- Unbonded (free) length to extend past the active wedge
- Bond length from grout-ground interface capacity
- Tendon selection (strand count and bar size)
- Test load requirements (proof, performance, extended creep)

All units SI: kN, m, mm, MPa, degrees.

References:
    FHWA-IF-99-015: Ground Anchors and Anchored Systems (GEC-4)
    PTI DC35.1: Recommendations for Prestressed Rock and Soil Anchors
    BS 8081: Code of Practice for Ground Anchorages
"""

import math
from typing import Dict, Any, Optional

from soe.results import AnchorDesignResult


# ============================================================================
# Nominal bond stress by soil/rock type (GEC-4 Table 4)
# ============================================================================
# Values in kPa — representative ultimate bond stress at the
# grout-ground interface for straight-shaft pressure-grouted anchors.
# These are nominal values; actual values depend on installation method,
# grout pressure, and ground conditions.

_BOND_STRESS_TABLE = {
    # Cohesionless soils
    "sand_loose": {"bond_stress_kPa": 70, "range": (50, 100),
                   "description": "Loose sand (SPT N < 10)"},
    "sand_medium": {"bond_stress_kPa": 145, "range": (100, 190),
                    "description": "Medium dense sand (SPT N 10-30)"},
    "sand_dense": {"bond_stress_kPa": 250, "range": (190, 310),
                   "description": "Dense sand (SPT N 30-50)"},
    "gravel": {"bond_stress_kPa": 310, "range": (250, 380),
               "description": "Sand-gravel mixtures"},

    # Cohesive soils
    "clay_stiff": {"bond_stress_kPa": 50, "range": (30, 70),
                   "description": "Stiff clay (cu 50-100 kPa)"},
    "clay_hard": {"bond_stress_kPa": 100, "range": (70, 130),
                  "description": "Hard clay (cu > 100 kPa)"},

    # Rock
    "rock_soft": {"bond_stress_kPa": 700, "range": (350, 1050),
                  "description": "Soft shale, sandstone, marl"},
    "rock_medium": {"bond_stress_kPa": 1050, "range": (700, 1400),
                    "description": "Medium limestone, schist"},
    "rock_hard": {"bond_stress_kPa": 1750, "range": (1400, 2100),
                  "description": "Hard granite, basalt, gneiss"},
}


# ============================================================================
# Strand and bar data (PTI DC35.1)
# ============================================================================
# Prestressing strand properties

_STRAND_DATA = {
    "strand_13mm": {
        "diameter_mm": 12.7,
        "area_mm2": 98.7,
        "ultimate_kN": 183.7,  # Grade 270 (1860 MPa)
        "description": "0.5 in (12.7 mm) Grade 270 strand",
    },
    "strand_15mm": {
        "diameter_mm": 15.2,
        "area_mm2": 140.0,
        "ultimate_kN": 260.7,  # Grade 270 (1860 MPa)
        "description": "0.6 in (15.2 mm) Grade 270 strand",
    },
}

# Prestressing bar properties (ASTM A722, Grade 150)
_BAR_DATA = {
    "bar_26mm": {
        "diameter_mm": 26,
        "area_mm2": 548,
        "ultimate_kN": 568,  # Grade 150 (1035 MPa)
        "description": "1 in (26 mm) Grade 150 bar",
    },
    "bar_32mm": {
        "diameter_mm": 32,
        "area_mm2": 819,
        "ultimate_kN": 848,
        "description": "1.25 in (32 mm) Grade 150 bar",
    },
    "bar_36mm": {
        "diameter_mm": 36,
        "area_mm2": 1019,
        "ultimate_kN": 1054,
        "description": "1.375 in (36 mm) Grade 150 bar",
    },
    "bar_44mm": {
        "diameter_mm": 44,
        "area_mm2": 1548,
        "ultimate_kN": 1601,
        "description": "1.75 in (44 mm) Grade 150 bar",
    },
}


# ============================================================================
# Unbonded (free) length
# ============================================================================

def compute_unbonded_length(
    anchor_depth: float,
    H: float,
    phi_deg: float,
    anchor_angle_deg: float = 15.0,
    min_free_length: float = 4.5,
) -> float:
    """Compute minimum unbonded (free) length for a ground anchor.

    The unbonded length must extend beyond the active failure wedge
    so that the bond zone is entirely in stable ground. Per GEC-4,
    the minimum unbonded length is also 4.5 m (or 3.0 m for bars)
    to allow adequate load transfer and limit lock-off losses.

    Parameters
    ----------
    anchor_depth : float
        Depth of anchor head from ground surface (m).
    H : float
        Total excavation depth (m).
    phi_deg : float
        Friction angle of retained soil (degrees).
    anchor_angle_deg : float
        Anchor inclination below horizontal (degrees). Default 15.
    min_free_length : float
        Minimum free length per code (m). Default 4.5 (GEC-4 for strand).

    Returns
    -------
    float
        Required unbonded length (m).
    """
    if anchor_depth <= 0 or H <= 0:
        raise ValueError("anchor_depth and H must be positive")

    # Active wedge angle from vertical = 45 - phi/2
    # The failure plane extends from the base of the wall at angle
    # (45 + phi/2) from horizontal
    wedge_angle_rad = math.radians(45.0 + phi_deg / 2.0)
    anchor_angle_rad = math.radians(anchor_angle_deg)

    # Horizontal distance from wall to active wedge at anchor depth
    # The active wedge starts at the excavation base and goes up at
    # angle (45+phi/2) from horizontal
    depth_below_anchor = H - anchor_depth
    if depth_below_anchor <= 0:
        # Anchor is at or below excavation level
        horiz_to_wedge = 0.0
    else:
        # Horizontal distance from wall base to where wedge intersects
        # anchor elevation
        horiz_to_wedge = depth_below_anchor / math.tan(wedge_angle_rad)

    # Length along anchor direction to reach past the wedge
    # Add 1.5 m buffer past the active wedge per GEC-4
    if anchor_angle_rad > 0:
        length_to_wedge = horiz_to_wedge / math.cos(anchor_angle_rad)
    else:
        length_to_wedge = horiz_to_wedge

    # Add 1.5 m past the wedge
    length_past_wedge = length_to_wedge + 1.5

    # Apply minimum free length
    return max(length_past_wedge, min_free_length)


# ============================================================================
# Bond length
# ============================================================================

def compute_bond_length(
    design_load_kN: float,
    bond_stress_kPa: float,
    drill_diameter_mm: float = 150.0,
    FOS_bond: float = 2.0,
    min_bond_length: float = 3.0,
    max_bond_length: float = 12.0,
) -> float:
    """Compute required bond (anchor) length in the grout zone.

    Bond length is determined by the grout-ground interface capacity:
        Lb = FOS * T / (pi * DDH * tau)

    where T is the design load, DDH is the drill hole diameter,
    and tau is the ultimate bond stress.

    Parameters
    ----------
    design_load_kN : float
        Design anchor load (kN).
    bond_stress_kPa : float
        Ultimate bond stress at grout-ground interface (kPa).
    drill_diameter_mm : float
        Drill hole diameter (mm). Default 150.
    FOS_bond : float
        Factor of safety on bond capacity. Default 2.0 (GEC-4).
    min_bond_length : float
        Minimum bond length (m). Default 3.0 per GEC-4.
    max_bond_length : float
        Maximum effective bond length (m). Default 12.0 per GEC-4.
        Beyond ~12 m, bond efficiency decreases significantly.

    Returns
    -------
    float
        Required bond length (m).
    """
    if design_load_kN <= 0:
        raise ValueError("Design load must be positive")
    if bond_stress_kPa <= 0:
        raise ValueError("Bond stress must be positive")
    if drill_diameter_mm <= 0:
        raise ValueError("Drill diameter must be positive")

    drill_diameter_m = drill_diameter_mm / 1000.0
    perimeter = math.pi * drill_diameter_m  # m

    # Required bond length
    Lb = (FOS_bond * design_load_kN) / (perimeter * bond_stress_kPa)

    # Apply limits
    Lb = max(Lb, min_bond_length)

    return round(Lb, 2)


# ============================================================================
# Tendon selection
# ============================================================================

def select_tendon(
    design_load_kN: float,
    lock_off_pct: float = 0.80,
    max_load_pct: float = 0.60,
    tendon_type: str = "strand_15mm",
) -> Dict[str, Any]:
    """Select tendon size and strand count for a ground anchor.

    Per PTI DC35.1, the maximum lock-off load should not exceed
    60% of the guaranteed ultimate tensile strength (GUTS) for
    permanent anchors or 80% for temporary anchors.

    The test load (typically 1.33× or 1.50× design load) must
    also not exceed 80% GUTS.

    Parameters
    ----------
    design_load_kN : float
        Design anchor load (kN).
    lock_off_pct : float
        Lock-off load as fraction of design load. Default 0.80
        (80% of design load, common practice).
    max_load_pct : float
        Maximum allowable load as fraction of GUTS. Default 0.60
        for permanent anchors. Use 0.80 for temporary.
    tendon_type : str
        "strand_13mm", "strand_15mm", or a bar type from _BAR_DATA.
        Default "strand_15mm" (0.6 in Grade 270).

    Returns
    -------
    dict
        "tendon_type", "n_strands" or "bar_size", "ultimate_capacity_kN",
        "lock_off_kN", "proof_test_kN", "performance_test_kN",
        "max_test_load_pct_GUTS".
    """
    if design_load_kN <= 0:
        raise ValueError("Design load must be positive")

    lock_off_kN = design_load_kN * lock_off_pct
    proof_test_kN = design_load_kN * 1.33  # PTI proof test = 133% DL
    performance_test_kN = design_load_kN * 1.50  # PTI performance = 150% DL

    # Maximum test load governs tendon sizing
    max_test_load = performance_test_kN  # 150% DL is the highest

    if tendon_type in _STRAND_DATA:
        strand = _STRAND_DATA[tendon_type]
        strand_capacity = strand["ultimate_kN"]

        # Required capacity: max_test_load / max_load_pct must not
        # exceed total GUTS
        # n * strand_capacity * max_load_pct >= max_test_load
        # But for permanent: design_load / (n * strand_capacity) <= 0.60
        required_GUTS = max_test_load / 0.80  # test load < 80% GUTS
        required_GUTS_dl = design_load_kN / max_load_pct  # DL < 60% GUTS

        governing_GUTS = max(required_GUTS, required_GUTS_dl)
        n_strands = math.ceil(governing_GUTS / strand_capacity)
        n_strands = max(n_strands, 1)

        total_GUTS = n_strands * strand_capacity
        max_test_pct = max_test_load / total_GUTS

        return {
            "tendon_type": tendon_type,
            "description": strand["description"],
            "n_strands": n_strands,
            "strand_capacity_kN": strand_capacity,
            "total_GUTS_kN": round(total_GUTS, 1),
            "design_load_kN": round(design_load_kN, 1),
            "lock_off_kN": round(lock_off_kN, 1),
            "proof_test_kN": round(proof_test_kN, 1),
            "performance_test_kN": round(performance_test_kN, 1),
            "design_load_pct_GUTS": round(design_load_kN / total_GUTS * 100, 1),
            "max_test_load_pct_GUTS": round(max_test_pct * 100, 1),
        }

    elif tendon_type in _BAR_DATA:
        bar = _BAR_DATA[tendon_type]
        bar_capacity = bar["ultimate_kN"]

        allowable = bar_capacity * max_load_pct
        if design_load_kN > allowable:
            # Bar is inadequate, try to find a larger bar
            selected_bar = None
            for bar_name in sorted(_BAR_DATA.keys(),
                                   key=lambda k: _BAR_DATA[k]["ultimate_kN"]):
                if _BAR_DATA[bar_name]["ultimate_kN"] * max_load_pct >= design_load_kN:
                    selected_bar = bar_name
                    break
            if selected_bar is None:
                return {
                    "tendon_type": tendon_type,
                    "error": "No single bar adequate — use strand tendon",
                    "design_load_kN": round(design_load_kN, 1),
                    "max_bar_capacity_kN": round(
                        max(b["ultimate_kN"] for b in _BAR_DATA.values()), 1
                    ),
                }
            bar = _BAR_DATA[selected_bar]
            tendon_type = selected_bar
            bar_capacity = bar["ultimate_kN"]

        max_test_pct = max_test_load / bar_capacity

        return {
            "tendon_type": tendon_type,
            "description": bar["description"],
            "n_strands": 1,
            "bar_capacity_kN": round(bar_capacity, 1),
            "total_GUTS_kN": round(bar_capacity, 1),
            "design_load_kN": round(design_load_kN, 1),
            "lock_off_kN": round(lock_off_kN, 1),
            "proof_test_kN": round(proof_test_kN, 1),
            "performance_test_kN": round(performance_test_kN, 1),
            "design_load_pct_GUTS": round(design_load_kN / bar_capacity * 100, 1),
            "max_test_load_pct_GUTS": round(max_test_pct * 100, 1),
        }

    else:
        raise ValueError(
            f"Unknown tendon_type '{tendon_type}'. "
            f"Available: {list(_STRAND_DATA.keys()) + list(_BAR_DATA.keys())}"
        )


# ============================================================================
# Complete anchor design
# ============================================================================

def design_ground_anchor(
    design_load_kN: float,
    anchor_depth: float,
    excavation_depth: float,
    phi_deg: float,
    soil_type: str = "sand_medium",
    anchor_angle_deg: float = 15.0,
    drill_diameter_mm: float = 150.0,
    tendon_type: str = "strand_15mm",
    FOS_bond: float = 2.0,
    lock_off_pct: float = 0.80,
    max_load_pct: float = 0.60,
    bond_stress_kPa: Optional[float] = None,
) -> AnchorDesignResult:
    """Design a complete ground anchor per GEC-4 and PTI.

    Computes unbonded length, bond length, total anchor length,
    and selects the tendon (strand count or bar size).

    Parameters
    ----------
    design_load_kN : float
        Design anchor load per anchor (kN).
    anchor_depth : float
        Depth of anchor head from ground surface (m).
    excavation_depth : float
        Total excavation depth H (m).
    phi_deg : float
        Friction angle of retained soil (degrees).
    soil_type : str
        Soil/rock type for bond stress lookup. See _BOND_STRESS_TABLE.
        Default "sand_medium".
    anchor_angle_deg : float
        Anchor inclination below horizontal (degrees). Default 15.
    drill_diameter_mm : float
        Drill hole diameter (mm). Default 150.
    tendon_type : str
        Tendon type for selection. Default "strand_15mm".
    FOS_bond : float
        Factor of safety on bond. Default 2.0.
    lock_off_pct : float
        Lock-off as fraction of design load. Default 0.80.
    max_load_pct : float
        Max allowable load as fraction of GUTS. Default 0.60 (permanent).
    bond_stress_kPa : float or None
        Override bond stress (kPa). If None, looked up from soil_type.

    Returns
    -------
    AnchorDesignResult
    """
    if design_load_kN <= 0:
        raise ValueError("Design load must be positive")
    if anchor_depth <= 0:
        raise ValueError("Anchor depth must be positive")
    if excavation_depth <= 0:
        raise ValueError("Excavation depth must be positive")

    # Bond stress
    if bond_stress_kPa is not None:
        tau = bond_stress_kPa
    elif soil_type in _BOND_STRESS_TABLE:
        tau = _BOND_STRESS_TABLE[soil_type]["bond_stress_kPa"]
    else:
        raise ValueError(
            f"Unknown soil_type '{soil_type}'. "
            f"Available: {list(_BOND_STRESS_TABLE.keys())}"
        )

    # Unbonded length
    unbonded_length = compute_unbonded_length(
        anchor_depth, excavation_depth, phi_deg, anchor_angle_deg
    )

    # Bond length
    bond_length = compute_bond_length(
        design_load_kN, tau, drill_diameter_mm, FOS_bond
    )

    # Total anchor length
    total_length = unbonded_length + bond_length

    # Tendon selection
    tendon = select_tendon(
        design_load_kN, lock_off_pct, max_load_pct, tendon_type
    )

    # Notes
    notes = []
    if bond_length > 12.0:
        notes.append(
            f"Bond length {bond_length:.1f} m exceeds 12 m — "
            "bond efficiency decreases; consider higher capacity "
            "soil zone or post-grouting"
        )
    if unbonded_length < 4.5:
        notes.append(
            f"Unbonded length {unbonded_length:.1f} m is less than "
            "4.5 m minimum — check active wedge geometry"
        )

    return AnchorDesignResult(
        design_load_kN=round(design_load_kN, 1),
        anchor_depth_m=round(anchor_depth, 2),
        anchor_angle_deg=anchor_angle_deg,
        unbonded_length_m=round(unbonded_length, 2),
        bond_length_m=round(bond_length, 2),
        total_length_m=round(total_length, 2),
        bond_stress_kPa=round(tau, 1),
        soil_type=soil_type,
        drill_diameter_mm=drill_diameter_mm,
        tendon=tendon,
        proof_test_kN=round(design_load_kN * 1.33, 1),
        performance_test_kN=round(design_load_kN * 1.50, 1),
        notes=notes,
    )


# ============================================================================
# Lookup helpers
# ============================================================================

def list_bond_stress_types() -> Dict[str, str]:
    """Return available soil/rock types and their bond stress descriptions."""
    return {k: v["description"] for k, v in _BOND_STRESS_TABLE.items()}


def get_bond_stress(soil_type: str) -> Dict[str, Any]:
    """Get bond stress data for a soil/rock type."""
    if soil_type not in _BOND_STRESS_TABLE:
        raise ValueError(
            f"Unknown soil_type '{soil_type}'. "
            f"Available: {list(_BOND_STRESS_TABLE.keys())}"
        )
    return {**_BOND_STRESS_TABLE[soil_type], "soil_type": soil_type}
