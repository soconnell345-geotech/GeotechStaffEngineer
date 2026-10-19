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
# Nominal bond stress by soil/rock type
# ============================================================================
# Values in kPa — representative ultimate bond stress at the
# grout-ground interface for straight-shaft pressure-grouted anchors.
# These are nominal values; actual values depend on installation method,
# grout pressure, and ground conditions.
#
# Provenance (GM-185, 2026-10-19): not traced to a printed table. This
# comment used to cite "GEC-4 Table 4", which is GEC-4's table of design
# steps. The medium sand row's numbers (145, range 100-190) are GEC-4
# Table 6's ultimate transfer LOADS for sand in kN/m (p. 71), used here as
# a bond stress in kPa over the hole's perimeter; GEC-4 Table 7 (p. 73,
# after PTI 1996) prints average ultimate bond stresses by anchor type and
# ground. The anchor result's choices say so.

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

# Prestressing bar properties (ASTM A722, Grade 150): GEC-4 Table 9 (p. 77),
# its metric rows as printed. The 15 mm strand is GEC-4 Table 10 (p. 78);
# the 13 mm strand is not in GEC-4 (ASTM A416). Until the GM-182/185 review
# the 32 mm bar carried 819 mm2 / 848 kN and the 1-3/4 in bar 1548 mm2 /
# 1601 kN, neither traced; Table 9 prints 806 / 835 and 1716 / 1779.
_BAR_DATA = {
    "bar_26mm": {
        "diameter_mm": 26,
        "area_mm2": 548,
        "ultimate_kN": 568,  # Grade 150 (1035 MPa)
        "description": "1 in (26 mm) Grade 150 bar",
    },
    "bar_32mm": {
        "diameter_mm": 32,
        "area_mm2": 806,
        "ultimate_kN": 835,
        "description": "1.25 in (32 mm) Grade 150 bar",
    },
    "bar_36mm": {
        "diameter_mm": 36,
        "area_mm2": 1019,
        "ultimate_kN": 1055,
        "description": "1.375 in (36 mm) Grade 150 bar",
    },
    "bar_44mm": {
        "diameter_mm": 44,
        "area_mm2": 1716,
        "ultimate_kN": 1779,
        "description": "1.75 in (44 mm; GEC-4 Table 9 prints 45 mm) "
                       "Grade 150 bar",
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

#: GEC-4 Sec. 5.3.8 (p. 77): the maximum test load shall not exceed 80 %
#: of the SMTS.
_TEST_LOAD_LIMIT = 0.80

#: The test load as a multiple of the design load (GM-185). GEC-4: "The
#: test load may vary from 120 to 150 percent of the design load with 133
#: percent being commonly used for permanent applications and 120 percent
#: being commonly used for temporary applications. A test load of 150
#: percent may be used for anchors in potentially creeping soils or when an
#: independent reference cannot be established for the dial gauge" (Sec.
#: 7.4.2.2, p. 143); the performance and proof test schedules both end at
#: 1.33 DL (Tables 21 and 22, pp. 144, 147), and "the test load for an
#: anchor for a temporary support of excavation application may be set at
#: 1.20 DL" (Table 22); permanent anchors "are typically loaded to 133
#: percent" (p. 76). The default is 1.33, the schedules' test load, though
#: this module mostly serves temporary support of excavation: 1.20 is
#: GEC-4's permission, not its procedure, and the higher test load is the
#: stricter check. At max_load_pct 0.60 the two size the same tendon (the
#: design-load limit governs at both). Until GM-185 the tendon was sized
#: for a 1.50 DL performance test, fixed.
TEST_LOAD_FACTOR = 1.33


def select_tendon(
    design_load_kN: float,
    lock_off_pct: float = 0.80,
    max_load_pct: float = 0.60,
    tendon_type: str = "strand_15mm",
    test_load_factor: float = TEST_LOAD_FACTOR,
) -> Dict[str, Any]:
    """Select tendon size and strand count for a ground anchor.

    GEC-4 Sec. 5.3.8 (p. 77): the design load shall not exceed 60 % of
    the specified minimum tensile strength (SMTS), the lock-off load 70 %
    and the maximum test load 80 %. The maximum test load is the test
    load of the proof and performance tests, ``test_load_factor`` x the
    design load (default 1.33, :data:`TEST_LOAD_FACTOR`). GEC-4 works both
    cases on p. 77: at 133 % the test-load limit allows 0.80/1.33 = 0.60
    SMTS, the design-load limit; at 150 % it allows 0.80/1.50 = 0.53 SMTS
    and governs. Both limits are checked for strand and bar tendons; until
    GM-160 a bar was held to the design-load limit only. The lock-off load
    (at most 100 % of the design load) is under 70 % SMTS whenever the
    design load is under 60 %.

    Parameters
    ----------
    design_load_kN : float
        Design anchor load (kN).
    lock_off_pct : float
        Lock-off load as fraction of design load. Default 0.80
        (80% of design load, common practice). Reported only
        (``lock_off_kN``); it does not size the tendon.
    max_load_pct : float
        Design-load limit as a fraction of SMTS. Default 0.60 (GEC-4
        p. 77). It governs the tendon when it is below
        0.80 / test_load_factor, the design load the 80 % test-load limit
        allows (0.6015 at 1.33, so the default governs; 0.53 at 1.50).
    tendon_type : str
        "strand_13mm", "strand_15mm", or a bar type from _BAR_DATA.
        Default "strand_15mm" (0.6 in Grade 270).
    test_load_factor : float
        The test load as a multiple of the design load: the maximum load
        of the proof and performance tests (GEC-4 Sec. 7.4.2.2, p. 143:
        1.20 to 1.50, 1.33 common for permanent anchors, 1.20 for
        temporary ones, 1.50 in potentially creeping soils). Default 1.33.

    Returns
    -------
    dict
        "tendon_type", "n_strands" or "bar_size", "ultimate_capacity_kN",
        "lock_off_kN", "test_load_factor", "test_load_kN" (also as
        "proof_test_kN" and "performance_test_kN": both tests go to the
        test load), "max_test_load_pct_GUTS", and "governed_by": which
        GEC-4 limit sized the tendon.
    """
    if design_load_kN <= 0:
        raise ValueError("Design load must be positive")
    if test_load_factor < 1.0:
        raise ValueError(
            f"test_load_factor {test_load_factor:g} is below 1.0: the test "
            "load is a multiple of the design load (GEC-4 Sec. 7.4.2.2, "
            "p. 143: 1.20 to 1.50)")

    lock_off_kN = design_load_kN * lock_off_pct
    # GEC-4 Sec. 7.4: the proof and the performance test both go to the
    # test load (Table 22, p. 147; Sec. 7.4.2.2, p. 143)
    max_test_load = design_load_kN * test_load_factor
    proof_test_kN = performance_test_kN = max_test_load
    # The SMTS each GEC-4 limit needs (Sec. 5.3.8, p. 77)
    need_test = max_test_load / _TEST_LOAD_LIMIT
    need_design = design_load_kN / max_load_pct
    governed_by = (
        f"maximum test load ({test_load_factor:.2f} DL) <= "
        f"{_TEST_LOAD_LIMIT:.2f} SMTS (GEC-4 Sec. 5.3.8, p. 77)"
        if need_test >= need_design else
        f"design load <= {max_load_pct:.2f} SMTS (max_load_pct; GEC-4 "
        f"Sec. 5.3.8, p. 77, prints 0.60)")
    governing_GUTS = max(need_test, need_design)

    if tendon_type in _STRAND_DATA:
        strand = _STRAND_DATA[tendon_type]
        strand_capacity = strand["ultimate_kN"]

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
            "test_load_factor": test_load_factor,
            "test_load_kN": round(max_test_load, 1),
            "proof_test_kN": round(proof_test_kN, 1),
            "performance_test_kN": round(performance_test_kN, 1),
            "design_load_pct_GUTS": round(design_load_kN / total_GUTS * 100, 1),
            "max_test_load_pct_GUTS": round(max_test_pct * 100, 1),
            "governed_by": governed_by,
        }

    elif tendon_type in _BAR_DATA:
        bar = _BAR_DATA[tendon_type]
        bar_capacity = bar["ultimate_kN"]

        if bar_capacity < governing_GUTS:
            # Bar is inadequate, try to find a larger bar
            selected_bar = None
            for bar_name in sorted(_BAR_DATA.keys(),
                                   key=lambda k: _BAR_DATA[k]["ultimate_kN"]):
                if _BAR_DATA[bar_name]["ultimate_kN"] >= governing_GUTS:
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
            "test_load_factor": test_load_factor,
            "test_load_kN": round(max_test_load, 1),
            "proof_test_kN": round(proof_test_kN, 1),
            "performance_test_kN": round(performance_test_kN, 1),
            "design_load_pct_GUTS": round(design_load_kN / bar_capacity * 100, 1),
            "max_test_load_pct_GUTS": round(max_test_pct * 100, 1),
            "governed_by": governed_by,
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
    test_load_factor: float = TEST_LOAD_FACTOR,
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
        Design-load limit as a fraction of SMTS. Default 0.60 (GEC-4
        Sec. 5.3.8, p. 77); the 80 % test-load limit governs where it
        allows less, 0.80 / test_load_factor (see :func:`select_tendon`).
    bond_stress_kPa : float or None
        Override bond stress (kPa). If None, looked up from soil_type.
    test_load_factor : float
        The test load as a multiple of the design load (GEC-4 Sec.
        7.4.2.2, p. 143). Default 1.33 (:data:`TEST_LOAD_FACTOR`).

    Returns
    -------
    AnchorDesignResult
        With ``choices`` (GM-185): the method, the test load, the SMTS
        limit, the bond stress and its factor of safety, each given or
        default, with its basis and the other options.
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
        design_load_kN, lock_off_pct, max_load_pct, tendon_type,
        test_load_factor=test_load_factor,
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
        test_load_factor=test_load_factor,
        test_load_kN=round(design_load_kN * test_load_factor, 1),
        proof_test_kN=round(design_load_kN * test_load_factor, 1),
        performance_test_kN=round(design_load_kN * test_load_factor, 1),
        notes=notes,
        choices=anchor_choices(
            soil_type, bond_stress_kPa, tau, FOS_bond, max_load_pct,
            test_load_factor, tendon.get("governed_by", "")),
    )


#: The test loads GEC-4 names (Sec. 7.4.2.2, p. 143; Table 22, p. 147).
_TEST_LOADS = (
    (1.20, "1.20 DL, temporary applications; Table 22 permits it for a "
           "temporary support of excavation (test_load_factor 1.2)"),
    (1.33, "1.33 DL, permanent applications; the test load of Tables 21 "
           "and 22 (test_load_factor 1.33)"),
    (1.50, "1.50 DL, potentially creeping soils, or no independent "
           "reference for the dial gauge (test_load_factor 1.5)"),
)


def anchor_choices(soil_type: str, bond_stress_kPa: Optional[float],
                   tau: float, FOS_bond: float, max_load_pct: float,
                   test_load_factor: float, governed_by: str) -> list:
    """What :func:`design_ground_anchor` chose (GM-185), as a result's
    ``choices`` (``geotech_common.choices``)."""
    from geotech_common.choices import MODULE, choice, how_given
    out = [
        choice("anchor design method",
               "GEC-4 (FHWA-IF-99-015) Sec. 5.3", MODULE,
               "unbonded length to 1.5 m past the active wedge from the "
               "excavation base at 45 + phi/2, at least 4.5 m (GEC-4 Sec. "
               "5.3.4, pp. 65-68: at least 4.5 m for strand and 3 m for "
               "bar tendons, extended H/5 or 1.5 m behind the critical "
               "potential failure surface); bond length L_b = FS_bond x "
               "DL / (pi d tau), at least 3 m; the tendon to the SMTS "
               "limits of Sec. 5.3.8 (p. 77)"),
        choice("test load", f"{test_load_factor:g} x design load",
               how_given(test_load_factor, TEST_LOAD_FACTOR,
                         name="test_load_factor"),
               "the maximum load of the proof and performance tests; "
               "GEC-4 Sec. 7.4.2.2 (p. 143): 120 to 150 percent of the "
               "design load, 133 percent commonly used for permanent and "
               "120 percent for temporary applications, 150 percent for "
               "potentially creeping soils; Table 21 (p. 144) and Table "
               "22 (p. 147): the performance and proof tests to 1.33 DL, "
               "which may be set at 1.20 DL for a temporary support of "
               "excavation. The default is 1.33, the tables' test load and "
               "the stricter check; at max_load_pct 0.60, 1.20 sizes the "
               "same tendon. The tendon carries the test load at no more "
               "than 0.80 SMTS (Sec. 5.3.8, p. 77)",
               [text for f, text in _TEST_LOADS
                if abs(f - test_load_factor) > 1e-9]),
        choice("design load limit (fraction of SMTS)", max_load_pct,
               how_given(max_load_pct, 0.60, name="max_load_pct"),
               "GEC-4 Sec. 5.3.8 (p. 77): the design load at most 0.60 "
               "SMTS, the lock-off load 0.70 and the maximum test load "
               f"0.80; a {test_load_factor:g} DL test load allows "
               f"{0.80 / test_load_factor:.2f} SMTS. Sized by: "
               f"{governed_by}"),
    ]
    if bond_stress_kPa is not None:
        out.append(choice(
            "ultimate bond stress", f"{tau:g} kPa",
            how_given(bond_stress_kPa, None, name="bond_stress_kPa"),
            "bond_stress_kPa as given; GEC-4 Table 7 (p. 73, after PTI "
            "1996) prints average ultimate bond stresses by anchor type "
            "and ground"))
    else:
        row = _BOND_STRESS_TABLE[soil_type]
        lo, hi = row["range"]
        out.append(choice(
            "ultimate bond stress",
            f"{tau:g} kPa for soil_type '{soil_type}' "
            f"({row['description']}; range {lo:g}-{hi:g} kPa)",
            how_given(soil_type, "sand_medium", name="soil_type"),
            "the module's presumptive value, not traced to a printed "
            "table (medium sand's 145, range 100-190, are GEC-4 Table 6's "
            "ultimate transfer loads for sand in kN/m, p. 71, not "
            "stresses); GEC-4 Table 7 (p. 73, after PTI 1996) prints "
            "average ultimate bond stresses by anchor type and ground, "
            "e.g. pressure-grouted fine-medium sand, medium dense to "
            "dense, 0.08-0.38 MPa",
            ["bond_stress_kPa from GEC-4 Table 7 (p. 73) for the anchor "
             "type and ground",
             "bond_stress_kPa from preproduction load tests (GEC-4 Sec. "
             "5.3.6, p. 75)"]))
    out.append(choice(
        "factor of safety on bond", FOS_bond,
        how_given(FOS_bond, 2.0, name="FOS_bond"),
        "on the ultimate load transfer; GEC-4 Sec. 5.3.6: 2.0 in soil "
        "(p. 71) and in weak rock, 3.0 in competent rock (p. 74)",
        ["3.0 for competent rock (FOS_bond 3)"] if FOS_bond != 3.0
        else ["2.0 for soil and weak rock (FOS_bond 2)"]))
    return out


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
