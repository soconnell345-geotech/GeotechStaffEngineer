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
# Presumptive ultimate load transfer by ground (GEC-4 Tables 6 and 8)
# ============================================================================
# GM-191 (2026-10-19): every row is a printed row, in the quantity printed:
# the ultimate LOAD TRANSFER per metre of bond length (kN/m), used per metre
# (bond length = FS x design load / load transfer), never multiplied by a
# perimeter.
#
# GEC-4 Table 6 (p. 71): "Presumptive ultimate values of load transfer for
# preliminary design of small diameter straight shaft gravity-grouted
# ground anchors in soil" (SPT values corrected for overburden pressure);
# "The maximum allowable anchor design load in soil may be determined by
# multiplying the bond length by the ultimate transfer load and dividing by
# a factor of safety of 2.0."
# GEC-4 Table 8 (p. 74): "Presumptive ultimate values of load transfer for
# preliminary design of ground anchors in rock"; "The maximum allowable
# anchor design load in competent rock may be determined by multiplying the
# bond length by the ultimate transfer load and dividing by a factor of
# safety of 3.0."
#
# GEC-4 Table 7 (p. 73, after PTI 1996) prints average ultimate bond
# STRESSES (MPa) instead; one given as bond_stress_kPa is multiplied by the
# perimeter of the drill hole ("capacity estimates should be based on the
# as-drilled hole diameter", p. 72).
#
# Until GM-191 this was a table of "bond stresses" (kPa) multiplied by the
# hole's perimeter and traced to no table: medium sand's 145 (100-190) is
# Table 6's sand row in kN/m, loose sand's 70 and hard clay's 100 (70-130)
# are Table 6's sand-and-silt row, stiff clay's 50 (30-70) is the middle of
# Table 7's gravity-grouted range for cohesive soil, hard rock's 1750
# (1400-2100) is the middle of Table 7's dolomitic limestone range; dense
# sand's 250, gravel's 310 and soft and medium rock's 700 and 1050 match no
# printed row. Over a 150 mm hole (perimeter 0.471 m) a kN/m value read as
# kPa gave 47 % of the load transfer, so a bond length 2.1 times Table 6's:
# conservative below a 318 mm hole (where pi x d = 1 m) and unconservative
# above. The rows kept their names where the name is a printed row (sand
# loose, medium dense, dense); the others are in _REMOVED_TYPES.

_TABLE_6 = "GEC-4 Table 6 (p. 71)"
_TABLE_8 = "GEC-4 Table 8 (p. 74)"

_LOAD_TRANSFER_TABLE = {
    # Table 6, soil
    "sand_gravel_loose": {"load_transfer_kN_per_m": 145, "ground": "soil",
                          "description": "Sand and gravel, loose (SPT 4-10)",
                          "source": _TABLE_6},
    "sand_gravel_medium": {"load_transfer_kN_per_m": 220, "ground": "soil",
                           "description": "Sand and gravel, medium dense "
                                          "(SPT 11-30)",
                           "source": _TABLE_6},
    "sand_gravel_dense": {"load_transfer_kN_per_m": 290, "ground": "soil",
                          "description": "Sand and gravel, dense (SPT 31-50)",
                          "source": _TABLE_6},
    "sand_loose": {"load_transfer_kN_per_m": 100, "ground": "soil",
                   "description": "Sand, loose (SPT 4-10)",
                   "source": _TABLE_6},
    "sand_medium": {"load_transfer_kN_per_m": 145, "ground": "soil",
                    "description": "Sand, medium dense (SPT 11-30)",
                    "source": _TABLE_6},
    "sand_dense": {"load_transfer_kN_per_m": 190, "ground": "soil",
                   "description": "Sand, dense (SPT 31-50)",
                   "source": _TABLE_6},
    "sand_silt_loose": {"load_transfer_kN_per_m": 70, "ground": "soil",
                        "description": "Sand and silt, loose (SPT 4-10)",
                        "source": _TABLE_6},
    "sand_silt_medium": {"load_transfer_kN_per_m": 100, "ground": "soil",
                         "description": "Sand and silt, medium dense "
                                        "(SPT 11-30)",
                         "source": _TABLE_6},
    "sand_silt_dense": {"load_transfer_kN_per_m": 130, "ground": "soil",
                        "description": "Sand and silt, dense (SPT 31-50)",
                        "source": _TABLE_6},
    "silt_clay_stiff": {"load_transfer_kN_per_m": 30, "ground": "soil",
                        "description": "Silt-clay mixture with low plasticity "
                                       "or fine micaceous sand or silt "
                                       "mixtures, stiff (SPT 10-20)",
                        "source": _TABLE_6},
    "silt_clay_hard": {"load_transfer_kN_per_m": 60, "ground": "soil",
                       "description": "Silt-clay mixture with low plasticity "
                                      "or fine micaceous sand or silt "
                                      "mixtures, hard (SPT 21-40)",
                       "source": _TABLE_6},
    # Table 8, rock
    "granite_basalt": {"load_transfer_kN_per_m": 730, "ground": "rock",
                       "description": "Granite or basalt",
                       "source": _TABLE_8},
    "dolomitic_limestone": {"load_transfer_kN_per_m": 580, "ground": "rock",
                            "description": "Dolomitic limestone",
                            "source": _TABLE_8},
    "soft_limestone": {"load_transfer_kN_per_m": 440, "ground": "rock",
                       "description": "Soft limestone", "source": _TABLE_8},
    "sandstone": {"load_transfer_kN_per_m": 440, "ground": "rock",
                  "description": "Sandstone", "source": _TABLE_8},
    "slates_hard_shales": {"load_transfer_kN_per_m": 360, "ground": "rock",
                           "description": "Slates and hard shales",
                           "source": _TABLE_8},
    "soft_shales": {"load_transfer_kN_per_m": 150, "ground": "rock",
                    "description": "Soft shales", "source": _TABLE_8},
}

#: The rock rows (GEC-4 Table 8): their factor of safety on bond defaults
#: to 3.0 (p. 74).
ROCK_TYPES = tuple(k for k, v in _LOAD_TRANSFER_TABLE.items()
                   if v["ground"] == "rock")

#: The rows GM-191 removed, with what each was and where to look instead.
_REMOVED_TYPES = {
    "gravel": "'gravel' (sand-gravel mixtures, 310 kPa) was removed (GM-191): "
              "no printed row has it. GEC-4 Table 6 (p. 71) prints sand and "
              "gravel by density: 'sand_gravel_loose' 145, "
              "'sand_gravel_medium' 220, 'sand_gravel_dense' 290 kN/m",
    "clay_stiff": "'clay_stiff' (50 kPa, the middle of GEC-4 Table 7's "
                  "gravity-grouted range for cohesive soil) was removed "
                  "(GM-191). Table 6 (p. 71) prints only silt-clay mixtures "
                  "with low plasticity: 'silt_clay_stiff' 30 kN/m; for a "
                  "clay give bond_stress_kPa from Table 7 (p. 73; "
                  "gravity-grouted 30-70 kPa, pressure-grouted by clay)",
    "clay_hard": "'clay_hard' (100 kPa; its 70-130 is Table 6's sand-and-silt "
                 "row in kN/m) was removed (GM-191). Table 6 (p. 71) prints "
                 "'silt_clay_hard' 60 kN/m (silt-clay mixture with low "
                 "plasticity, hard); for a clay give bond_stress_kPa from "
                 "Table 7 (p. 73)",
    "rock_soft": "'rock_soft' (soft shale, sandstone, marl, 700 kPa) was "
                 "removed (GM-191): no printed row has it. GEC-4 Table 8 "
                 "(p. 74) prints 'soft_shales' 150 and 'sandstone' 440 kN/m; "
                 "Table 7 (p. 73) prints bond stresses for bond_stress_kPa",
    "rock_medium": "'rock_medium' (medium limestone, schist, 1050 kPa) was "
                   "removed (GM-191): no printed row has it. GEC-4 Table 8 "
                   "(p. 74) prints 'soft_limestone' 440, "
                   "'dolomitic_limestone' 580 and 'slates_hard_shales' "
                   "360 kN/m",
    "rock_hard": "'rock_hard' (hard granite, basalt, gneiss, 1750 kPa: the "
                 "middle of Table 7's dolomitic limestone range) was removed "
                 "(GM-191). GEC-4 Table 8 (p. 74) prints 'granite_basalt' "
                 "730 kN/m; Table 7 (p. 73) prints granite and basalt "
                 "1.7-3.1 MPa as a bond stress",
}


def _row(soil_type: str) -> Dict[str, Any]:
    """The load-transfer row of ``soil_type``; a removed row or an unknown
    name raises, the removed one saying where it went."""
    if soil_type in _LOAD_TRANSFER_TABLE:
        return _LOAD_TRANSFER_TABLE[soil_type]
    if soil_type in _REMOVED_TYPES:
        raise ValueError(_REMOVED_TYPES[soil_type])
    raise ValueError(
        f"Unknown soil_type '{soil_type}'. "
        f"Available: {list(_LOAD_TRANSFER_TABLE.keys())}"
    )


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

#: GEC-4 Sec. 5.3.4 (p. 67): "The minimum unbonded length for rock and
#: soil ground anchors is 4.5 m for strand tendons and 3 m for bar
#: tendons" (the same in the guide specification, App. E Part 3 C, p. E-15).
MIN_UNBONDED_STRAND_M = 4.5
MIN_UNBONDED_BAR_M = 3.0


def distance_behind_failure_surface(H: float) -> float:
    """How far behind the critical potential failure surface the unbonded
    length extends: the greater of 1.5 m and H/5 (m). GEC-4 App. E Part 3
    C (p. E-15): "The free stressing length (unbonded length) shall extend
    at least 1.5 m or 20 percent of the height of the wall, whichever is
    greater, behind the critical failure surface"; Sec. 5.3.2 (p. 65) and
    5.3.4 (p. 68) give "H/5 or 1.5 m"."""
    return max(1.5, H / 5.0)


def compute_unbonded_length(
    anchor_depth: float,
    H: float,
    phi_deg: float,
    anchor_angle_deg: float = 15.0,
    min_free_length: float = MIN_UNBONDED_STRAND_M,
) -> float:
    """Compute minimum unbonded (free) length for a ground anchor.

    GEC-4 (FHWA-IF-99-015): for walls in cohesionless soils the critical
    potential failure surface "can be assumed to extend up from the corner
    of the excavation at an angle of 45 + phi/2 from the horizontal" (Sec.
    5.3.2, p. 65), and the unbonded length extends the greater of 1.5 m and
    H/5 behind it (:func:`distance_behind_failure_surface`; App. E, p.
    E-15). The distance is measured normal to the surface and the unbonded
    length runs along the anchor to that line, as GEC-4's design example
    draws it (App. A, Figure A-3, p. A-8: H 10 m, phi 33, anchors at 15
    deg from 2.5 m and 6.25 m deep, 5.7 m and 3.9 m). It is at least
    ``min_free_length`` (Sec. 5.3.4, p. 67: 4.5 m for strand, 3 m for bar
    tendons).

    Until GM-192 (2026-10-19) the distance behind the surface was 1.5 m
    whatever the wall height, and the anchor was measured to the surface at
    the anchor head's level (longer than the anchor's own run to it).

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
        Minimum unbonded length (m). Default 4.5, GEC-4's for strand
        tendons (3.0 for bar tendons).

    Returns
    -------
    float
        Required unbonded length (m).
    """
    if anchor_depth <= 0 or H <= 0:
        raise ValueError("anchor_depth and H must be positive")

    beta = math.radians(45.0 + phi_deg / 2.0)   # surface from horizontal
    theta = math.radians(anchor_angle_deg)       # anchor below horizontal
    clear = distance_behind_failure_surface(H)

    # The limit line is the failure surface moved back `clear`, measured
    # normal to it: horizontally from the wall at depth z it lies
    #   x(z) = clear / sin(beta) + (H - z) / tan(beta).
    # Along the anchor, x = s cos(theta) at depth z = anchor_depth +
    # s sin(theta), so
    #   s = (clear / sin(beta) + (H - anchor_depth) / tan(beta))
    #       / (cos(theta) + sin(theta) / tan(beta)).
    # An anchor head at or below the excavation level is clear of the
    # surface from the start.
    above_base = max(H - anchor_depth, 0.0)
    to_limit = ((clear / math.sin(beta) + above_base / math.tan(beta))
                / (math.cos(theta) + math.sin(theta) / math.tan(beta)))

    return max(to_limit, min_free_length)


# ============================================================================
# Bond length
# ============================================================================

def compute_bond_length(
    design_load_kN: float,
    bond_stress_kPa: Optional[float] = None,
    drill_diameter_mm: float = 150.0,
    FOS_bond: float = 2.0,
    min_bond_length: float = 3.0,
    max_bond_length: float = 12.0,
    load_transfer_kN_per_m: Optional[float] = None,
) -> float:
    """Compute required bond (anchor) length in the grout zone.

    From an ultimate load transfer per metre of bond length q (kN/m, GEC-4
    Tables 6 and 8, pp. 71, 74), used as printed (GM-191):
        Lb = FOS * T / q
    or from an average ultimate bond stress tau (kPa, GEC-4 Table 7, p. 73)
    over the perimeter of the drill hole:
        Lb = FOS * T / (pi * DDH * tau)
    where T is the design load and DDH the drill hole diameter. Give one
    of ``load_transfer_kN_per_m`` and ``bond_stress_kPa``.

    Parameters
    ----------
    design_load_kN : float
        Design anchor load (kN).
    bond_stress_kPa : float, optional
        Average ultimate bond stress at the grout-ground interface (kPa).
    drill_diameter_mm : float
        Drill hole diameter (mm), with ``bond_stress_kPa`` only. Default
        150.
    FOS_bond : float
        Factor of safety on the ultimate load transfer. Default 2.0, GEC-4's
        for soil (p. 71; 3.0 in competent rock, p. 74).
    min_bond_length : float
        Minimum bond length (m). Default 3.0; :func:`design_ground_anchor`
        passes GEC-4's (App. E Part 3 B, p. E-15: 4.5 m in soil, 4.5 m for
        strand and 3 m for bar tendons in rock).
    max_bond_length : float
        Not applied; GEC-4 p. 71: soil anchor bond lengths are "typically
        4.5 to 12 m" and longer ones add little capacity.
    load_transfer_kN_per_m : float, optional
        Ultimate load transfer per metre of bond length (kN/m).

    Returns
    -------
    float
        Required bond length (m).
    """
    if design_load_kN <= 0:
        raise ValueError("Design load must be positive")
    if (bond_stress_kPa is None) == (load_transfer_kN_per_m is None):
        raise ValueError("Give one of load_transfer_kN_per_m (kN/m) and "
                         "bond_stress_kPa (kPa)")
    if load_transfer_kN_per_m is not None:
        if load_transfer_kN_per_m <= 0:
            raise ValueError("Load transfer must be positive")
        q = load_transfer_kN_per_m
    else:
        if bond_stress_kPa <= 0:
            raise ValueError("Bond stress must be positive")
        if drill_diameter_mm <= 0:
            raise ValueError("Drill diameter must be positive")
        # kPa over the hole's perimeter (m) is kN per metre of bond length
        q = bond_stress_kPa * math.pi * drill_diameter_mm / 1000.0

    # Required bond length
    Lb = (FOS_bond * design_load_kN) / q

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
    drill_diameter_mm: Optional[float] = None,
    tendon_type: str = "strand_15mm",
    FOS_bond: Optional[float] = None,
    lock_off_pct: float = 0.80,
    max_load_pct: float = 0.60,
    bond_stress_kPa: Optional[float] = None,
    test_load_factor: float = TEST_LOAD_FACTOR,
    load_transfer_kN_per_m: Optional[float] = None,
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
        The ground of the bond zone, a row of GEC-4 Table 6 (soil, p. 71)
        or Table 8 (rock, p. 74): its presumptive ultimate load transfer
        (kN/m) is used per metre of bond length (GM-191), and its ground
        sets FOS_bond's default and the minimum bond length. Default
        "sand_medium" (sand, medium dense, 145 kN/m). See
        :func:`list_bond_stress_types`.
    anchor_angle_deg : float
        Anchor inclination below horizontal (degrees). Default 15.
    drill_diameter_mm : float, optional
        Drill hole diameter (mm): the perimeter a given ``bond_stress_kPa``
        acts on. Default 150 there (GEC-4 p. 70: "generally less than 150
        mm"); not used with a load transfer per metre.
    tendon_type : str
        Tendon type for selection. Default "strand_15mm".
    FOS_bond : float, optional
        Factor of safety on the ultimate load transfer. Default 2.0 in soil
        (GEC-4 p. 71) and 3.0 in rock (competent rock, p. 74), by
        ``soil_type``'s ground.
    lock_off_pct : float
        Lock-off as fraction of design load. Default 0.80.
    max_load_pct : float
        Design-load limit as a fraction of SMTS. Default 0.60 (GEC-4
        Sec. 5.3.8, p. 77); the 80 % test-load limit governs where it
        allows less, 0.80 / test_load_factor (see :func:`select_tendon`).
    bond_stress_kPa : float or None
        Average ultimate bond stress (kPa; GEC-4 Table 7, p. 73), used in
        place of ``soil_type``'s load transfer over the perimeter of the
        drill hole.
    test_load_factor : float
        The test load as a multiple of the design load (GEC-4 Sec.
        7.4.2.2, p. 143). Default 1.33 (:data:`TEST_LOAD_FACTOR`).
    load_transfer_kN_per_m : float or None
        Ultimate load transfer per metre of bond length (kN/m), used in
        place of ``soil_type``'s, e.g. as verified by preproduction
        performance tests (GEC-4 p. 75).

    Returns
    -------
    AnchorDesignResult
        With ``choices`` (GM-185, GM-191/192): the method, the test load,
        the SMTS limit, the load transfer and its factor of safety, the
        distance behind the failure surface and the minimum unbonded and
        bond lengths, each given, default or the module's, with its basis
        and the other options.
    """
    if design_load_kN <= 0:
        raise ValueError("Design load must be positive")
    if anchor_depth <= 0:
        raise ValueError("Anchor depth must be positive")
    if excavation_depth <= 0:
        raise ValueError("Excavation depth must be positive")
    if bond_stress_kPa is not None and load_transfer_kN_per_m is not None:
        raise ValueError(
            "Give bond_stress_kPa (a stress, kPa, over the drill hole's "
            "perimeter) or load_transfer_kN_per_m (kN per metre of bond "
            "length), not both")

    # The ground: its row gives the load transfer and, beside a given one,
    # still says soil or rock for the factor of safety and the minimum
    # bond length.
    row = _row(soil_type)
    ground = row["ground"]
    bar = tendon_type in _BAR_DATA

    # Ultimate load transfer per metre of bond length (kN/m)
    diameter = None
    if bond_stress_kPa is not None:
        if bond_stress_kPa <= 0:
            raise ValueError("Bond stress must be positive")
        diameter = 150.0 if drill_diameter_mm is None else drill_diameter_mm
        if diameter <= 0:
            raise ValueError("Drill diameter must be positive")
        q = bond_stress_kPa * math.pi * diameter / 1000.0
    elif load_transfer_kN_per_m is not None:
        if load_transfer_kN_per_m <= 0:
            raise ValueError("Load transfer must be positive")
        q = load_transfer_kN_per_m
    else:
        q = row["load_transfer_kN_per_m"]

    fs = FOS_bond if FOS_bond is not None else (
        3.0 if ground == "rock" else 2.0)
    if fs <= 0:
        raise ValueError("FOS_bond must be positive")

    min_bond = _min_bond_length(ground, bar)
    min_unbonded = MIN_UNBONDED_BAR_M if bar else MIN_UNBONDED_STRAND_M

    # Unbonded length
    unbonded_length = compute_unbonded_length(
        anchor_depth, excavation_depth, phi_deg, anchor_angle_deg,
        min_free_length=min_unbonded,
    )

    # Bond length
    bond_length = compute_bond_length(
        design_load_kN, FOS_bond=fs, min_bond_length=min_bond,
        load_transfer_kN_per_m=q,
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
    if ground == "soil" and bond_length > 12.0:
        notes.append(
            f"Bond length {bond_length:.1f} m exceeds 12 m: GEC-4 p. 71, "
            "soil anchor bond lengths are typically 4.5 to 12 m, and "
            "significant increases in capacity beyond about 12 m cannot "
            "be achieved unless specialized methods are used to transfer "
            "load towards the end of the anchor; consider post-grouting, "
            "a better bond zone or a lower design load"
        )
    if ground == "rock" and bond_length > 10.0:
        notes.append(
            f"Bond length {bond_length:.1f} m exceeds 10 m: GEC-4 p. 74, "
            "rock anchor bond lengths typically range from 3 to 10 m"
        )
    if drill_diameter_mm is not None and bond_stress_kPa is None:
        notes.append(
            "drill_diameter_mm is not used: a load transfer per metre "
            "(GEC-4 Table 6 or 8, or load_transfer_kN_per_m) does not "
            "depend on the hole; give bond_stress_kPa for a stress over "
            "the hole's perimeter"
        )

    return AnchorDesignResult(
        design_load_kN=round(design_load_kN, 1),
        anchor_depth_m=round(anchor_depth, 2),
        anchor_angle_deg=anchor_angle_deg,
        unbonded_length_m=round(unbonded_length, 2),
        bond_length_m=round(bond_length, 2),
        total_length_m=round(total_length, 2),
        load_transfer_kN_per_m=round(q, 1),
        FOS_bond=fs,
        bond_stress_kPa=(None if bond_stress_kPa is None
                         else round(bond_stress_kPa, 1)),
        soil_type=soil_type,
        drill_diameter_mm=diameter,
        tendon=tendon,
        test_load_factor=test_load_factor,
        test_load_kN=round(design_load_kN * test_load_factor, 1),
        proof_test_kN=round(design_load_kN * test_load_factor, 1),
        performance_test_kN=round(design_load_kN * test_load_factor, 1),
        notes=notes,
        choices=anchor_choices(
            soil_type, bond_stress_kPa, load_transfer_kN_per_m, q,
            FOS_bond, fs, max_load_pct, test_load_factor,
            tendon.get("governed_by", ""), diameter=diameter,
            drill_diameter_mm=drill_diameter_mm, bar=bar,
            H=excavation_depth),
    )


def _min_bond_length(ground: str, bar: bool) -> float:
    """GEC-4 App. E Part 3 B (p. E-15): "The minimum bond length shall be
    4.5 m for strand tendons in rock and 3 m for bar tendons in rock. The
    minimum bond length shall be 4.5 m for strand and bar tendons in
    soil." (m)"""
    return 3.0 if (ground == "rock" and bar) else 4.5


#: The test loads GEC-4 names (Sec. 7.4.2.2, p. 143; Table 22, p. 147).
_TEST_LOADS = (
    (1.20, "1.20 DL, temporary applications; Table 22 permits it for a "
           "temporary support of excavation (test_load_factor 1.2)"),
    (1.33, "1.33 DL, permanent applications; the test load of Tables 21 "
           "and 22 (test_load_factor 1.33)"),
    (1.50, "1.50 DL, potentially creeping soils, or no independent "
           "reference for the dial gauge (test_load_factor 1.5)"),
)

#: The factors of safety on bond GEC-4 gives (Sec. 5.3.6).
_BOND_FS = (
    (2.0, "2.0 in soil (p. 71), and in weak rocks such as clay shales, "
          "intermediate geomaterials of unconfined compressive strength 0.5 "
          "to 5.0 MPa (p. 74) (FOS_bond 2)"),
    (3.0, "3.0 in competent rock (p. 74) (FOS_bond 3)"),
)


def anchor_choices(soil_type: str, bond_stress_kPa: Optional[float],
                   load_transfer_kN_per_m: Optional[float], q: float,
                   FOS_bond: Optional[float], fs: float,
                   max_load_pct: float, test_load_factor: float,
                   governed_by: str, *, diameter: Optional[float] = None,
                   drill_diameter_mm: Optional[float] = None,
                   bar: bool = False, H: float = 0.0) -> list:
    """What :func:`design_ground_anchor` chose (GM-185, GM-191/192), as a
    result's ``choices`` (``geotech_common.choices``)."""
    from geotech_common.choices import MODULE, choice, how_given
    row = _LOAD_TRANSFER_TABLE[soil_type]
    ground = row["ground"]
    out = [
        choice("anchor design method",
               "GEC-4 (FHWA-IF-99-015) Sec. 5.3", MODULE,
               "unbonded length from the anchor head to a line behind the "
               "critical potential failure surface, which rises from the "
               "corner of the excavation at 45 + phi/2 from the horizontal "
               "(Sec. 5.3.2, p. 65, for walls in cohesionless soils), the "
               "distance measured normal to it as GEC-4's design example "
               "draws it (App. A, Figure A-3, p. A-8); bond length L_b = FS "
               "x DL / ultimate load transfer per metre (pp. 71, 74); the "
               "tendon to the SMTS limits of Sec. 5.3.8 (p. 77)"),
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
    by_row = (f"soil_type '{soil_type}': {row['load_transfer_kN_per_m']:g} "
              f"kN/m, {row['description']} ({row['source']})")
    given_transfer = ("load_transfer_kN_per_m, e.g. as verified by "
                      "preproduction performance tests (GEC-4 p. 75)")
    given_stress = ("bond_stress_kPa from GEC-4 Table 7 (p. 73, after PTI "
                    "1996), an average ultimate bond stress by anchor type "
                    "and ground, over the perimeter of the drill hole "
                    "(drill_diameter_mm)")
    if bond_stress_kPa is not None:
        out.append(choice(
            "ultimate load transfer",
            f"{bond_stress_kPa:g} kPa x pi x {diameter:g} mm = {q:.1f} kN/m",
            how_given(bond_stress_kPa, None, name="bond_stress_kPa"),
            "bond_stress_kPa as given, an average ultimate bond stress "
            "(GEC-4 Table 7, p. 73, after PTI 1996), over the perimeter of "
            "the drill hole: \"capacity estimates should be based on the "
            "as-drilled hole diameter\" (p. 72)",
            [given_transfer, by_row]))
        out.append(choice(
            "drill hole diameter", f"{diameter:g} mm",
            how_given(drill_diameter_mm, None, name="drill_diameter_mm"),
            "the perimeter the bond stress acts on; GEC-4 p. 70: the drill "
            "hole diameter \"is generally less than 150 mm, except for "
            "hollow stem augered anchors that are typically approximately "
            "300 mm in diameter\""))
    elif load_transfer_kN_per_m is not None:
        out.append(choice(
            "ultimate load transfer", f"{q:g} kN/m",
            how_given(load_transfer_kN_per_m, None,
                      name="load_transfer_kN_per_m"),
            "load_transfer_kN_per_m as given, per metre of bond length",
            [by_row, given_stress]))
    else:
        table = ("small diameter straight shaft gravity-grouted ground "
                 "anchors in soil" if ground == "soil"
                 else "ground anchors in rock")
        caveat = ("SPT values corrected for overburden pressure"
                  if ground == "soil" else
                  "lower values may be recommended after input from a "
                  "geologist, especially if the rock mass strength is "
                  "controlled by discontinuities (p. 74)")
        out.append(choice(
            "ultimate load transfer",
            f"{q:g} kN/m for soil_type '{soil_type}' "
            f"({row['description']}; {row['source']})",
            how_given(soil_type, "sand_medium", name="soil_type"),
            f"{row['source']}: presumptive ultimate values of load transfer "
            f"for preliminary design of {table}, per metre of bond length; "
            f"{caveat}. The drill hole diameter does not enter it (until "
            "GM-191 the module read kN/m values as kPa and multiplied them "
            "by the hole's perimeter)",
            [given_transfer, given_stress]))
    out.append(choice(
        "factor of safety on bond", fs,
        how_given(FOS_bond, None, name="FOS_bond"),
        "on the ultimate load transfer; GEC-4 Sec. 5.3.6: 2.0 in soil "
        "(p. 71); 3.0 in competent rock, \"to account for uncertainties "
        "associated with potential discontinuities in the rock mass\", and "
        "2.0 in weak rocks such as clay shales (p. 74). The default follows "
        f"soil_type: '{soil_type}' is {ground}",
        [text for f, text in _BOND_FS if abs(f - fs) > 1e-9]))
    clear = distance_behind_failure_surface(H)
    out.append(choice(
        "distance behind the critical potential failure surface",
        f"{clear:.2f} m (the greater of 1.5 m and H/5)", MODULE,
        "GEC-4 App. E Part 3 C (p. E-15): \"The free stressing length "
        "(unbonded length) shall extend at least 1.5 m or 20 percent of "
        "the height of the wall, whichever is greater, behind the critical "
        "failure surface\"; Sec. 5.3.2 (p. 65) and 5.3.4 (p. 68): \"a "
        "minimum distance of H/5 or 1.5 m\". Until GM-192, 1.5 m whatever "
        "the wall height"))
    out.append(choice(
        "minimum unbonded length",
        f"{MIN_UNBONDED_BAR_M if bar else MIN_UNBONDED_STRAND_M:g} m "
        f"({'bar' if bar else 'strand'} tendon)", MODULE,
        "GEC-4 Sec. 5.3.4 (p. 67): \"The minimum unbonded length for rock "
        "and soil ground anchors is 4.5 m for strand tendons and 3 m for "
        "bar tendons\", to prevent significant reductions in load from "
        "seating losses. Until GM-192, 4.5 m for bars too"))
    out.append(choice(
        "minimum bond length",
        f"{_min_bond_length(ground, bar):g} m ({ground}, "
        f"{'bar' if bar else 'strand'} tendon)", MODULE,
        "GEC-4 App. E Part 3 B (p. E-15): \"The minimum bond length shall "
        "be 4.5 m for strand tendons in rock and 3 m for bar tendons in "
        "rock. The minimum bond length shall be 4.5 m for strand and bar "
        "tendons in soil.\" Until GM-192, 3 m in every ground"))
    return out


# ============================================================================
# Lookup helpers
# ============================================================================

def list_bond_stress_types() -> Dict[str, str]:
    """The ``soil_type`` values and the printed row each is: GEC-4 Table 6
    (soil) and Table 8 (rock), presumptive ultimate load transfer per metre
    of bond length (GM-191)."""
    return {k: f"{v['description']}: {v['load_transfer_kN_per_m']:g} kN/m "
               f"({v['source']})"
            for k, v in _LOAD_TRANSFER_TABLE.items()}


def get_bond_stress(soil_type: str) -> Dict[str, Any]:
    """The load-transfer row of a ``soil_type``: ``load_transfer_kN_per_m``
    (kN per metre of bond length, GEC-4 Table 6 or 8), ``ground``,
    ``description``, ``source``. Kept under its old name; the table has
    held loads per metre, not stresses, since GM-191."""
    return {**_row(soil_type), "soil_type": soil_type}
