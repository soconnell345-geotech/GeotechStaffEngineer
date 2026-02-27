"""
Freestanding wall and fence wind load analysis per ASCE 7-22 Chapter 29.3.

Functions:
    get_Cf_freestanding_wall  - Net force coefficient Cf (Figure 29.3-1)
    analyze_freestanding_wall_wind - Full wall wind analysis
    analyze_fence_wind        - Fence analysis with porosity reduction
"""

from wind_loads.wind_pressure import compute_velocity_pressure
from wind_loads.results import FreestandingWallWindResult

# ---------------------------------------------------------------------------
# Cf data from ASCE 7-22 Figure 29.3-1
# ---------------------------------------------------------------------------

# Case A: wall on ground (clearance_ratio = 0, i.e., h/s = 0)
_CF_CASE_A = [
    (1.0, 1.30),
    (2.0, 1.40),
    (5.0, 1.55),
    (10.0, 1.70),
    (40.0, 1.75),
]

# Case C: wall with clearance (h >= s, i.e., clearance_ratio >= 1.0)
_CF_CASE_C = [
    (1.0, 1.80),
    (2.0, 1.85),
    (5.0, 1.90),
    (10.0, 1.95),
    (40.0, 2.00),
]


def _interp(table, x):
    """Linear interpolation in a sorted (x, y) table with clamping."""
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def get_Cf_freestanding_wall(
    B_over_s: float,
    clearance_ratio: float = 0.0,
) -> float:
    """Look up net force coefficient Cf for a freestanding wall (Figure 29.3-1).

    Interpolates between Case A (wall on ground) and Case C (elevated wall)
    based on the clearance ratio h/s.

    Parameters
    ----------
    B_over_s : float
        Ratio of wall length B to wall height s. Must be > 0.
    clearance_ratio : float
        Ratio of clearance height h to wall height s (h/s). 0 for wall
        on ground, >= 1.0 for fully elevated. Default 0.0.

    Returns
    -------
    float
        Net force coefficient Cf (dimensionless).

    Raises
    ------
    ValueError
        If B_over_s <= 0 or clearance_ratio < 0.
    """
    if B_over_s <= 0:
        raise ValueError(f"B_over_s must be > 0, got {B_over_s}")
    if clearance_ratio < 0:
        raise ValueError(f"clearance_ratio must be >= 0, got {clearance_ratio}")

    cf_a = _interp(_CF_CASE_A, B_over_s)
    cf_c = _interp(_CF_CASE_C, B_over_s)

    # Linear interpolation between Case A and Case C
    ratio = min(clearance_ratio, 1.0)
    Cf = cf_a + (cf_c - cf_a) * ratio
    return Cf


def analyze_freestanding_wall_wind(
    V: float,
    wall_height: float,
    wall_length: float,
    exposure_category: str,
    Kzt: float = 1.0,
    Kd: float = 0.85,
    Ke: float = 1.0,
    G: float = 0.85,
    clearance_height: float = 0.0,
) -> FreestandingWallWindResult:
    """Analyze wind loads on a solid freestanding wall per ASCE 7-22 Ch 29.3.

    Computes velocity pressure at the top of the wall, applies gust-effect
    factor and net force coefficient, and returns forces and overturning moment.

    Parameters
    ----------
    V : float
        Basic wind speed (m/s). Must be > 0.
    wall_height : float
        Wall height s (m). Must be > 0.
    wall_length : float
        Wall length B (m). Must be > 0.
    exposure_category : str
        Exposure category: 'B', 'C', or 'D'.
    Kzt : float
        Topographic factor. Default 1.0.
    Kd : float
        Wind directionality factor. Default 0.85 (Table 26.6-1).
    Ke : float
        Ground elevation factor. Default 1.0.
    G : float
        Gust-effect factor. Default 0.85 (rigid structures).
    clearance_height : float
        Clearance from ground to bottom of wall (m). Default 0.0.

    Returns
    -------
    FreestandingWallWindResult
        Complete wind analysis results.

    Raises
    ------
    ValueError
        If any input is out of valid range.
    """
    if wall_height <= 0:
        raise ValueError(f"wall_height must be > 0, got {wall_height}")
    if wall_length <= 0:
        raise ValueError(f"wall_length must be > 0, got {wall_length}")
    if G <= 0:
        raise ValueError(f"G must be > 0, got {G}")
    if clearance_height < 0:
        raise ValueError(f"clearance_height must be >= 0, got {clearance_height}")

    s = wall_height
    B = wall_length

    # Height to top of wall
    z_top = clearance_height + s

    # Velocity pressure at top of wall
    vp = compute_velocity_pressure(V, z_top, exposure_category, Kzt, Kd, Ke)

    # Force coefficient
    B_over_s = B / s
    clearance_ratio = clearance_height / s
    Cf = get_Cf_freestanding_wall(B_over_s, clearance_ratio)

    # Design wind pressure
    p = vp.qz_Pa * G * Cf  # Pa

    # Force per unit length
    f = p * s / 1000.0  # kN/m

    # Total force
    F = f * B  # kN

    # Overturning moment per unit length about base
    # Moment arm is from base to centroid of pressure on wall
    moment_arm = clearance_height + s / 2.0
    M = f * moment_arm  # kN*m/m

    return FreestandingWallWindResult(
        velocity_pressure_Pa=vp.qz_Pa,
        velocity_pressure_kPa=vp.qz_kPa,
        wind_pressure_Pa=p,
        wind_pressure_kPa=p / 1000.0,
        force_per_unit_length_kN_m=f,
        total_force_kN=F,
        overturning_moment_kNm_per_m=M,
        Kz=vp.Kz,
        Kzt=vp.Kzt,
        Kd=vp.Kd,
        Ke=vp.Ke,
        G=G,
        Cf=Cf,
        B_over_s=B_over_s,
        V_m_s=V,
        wall_height_m=s,
        wall_length_m=B,
        exposure_category=vp.exposure_category,
        clearance_height_m=clearance_height,
        solidity_ratio=1.0,
    )


def analyze_fence_wind(
    V: float,
    fence_height: float,
    fence_length: float,
    solidity_ratio: float,
    exposure_category: str,
    Kzt: float = 1.0,
    Kd: float = 0.85,
    Ke: float = 1.0,
    G: float = 0.85,
    clearance_height: float = 0.0,
) -> FreestandingWallWindResult:
    """Analyze wind loads on a fence per ASCE 7-22 Ch 29.3, Note 4.

    For porous fences, the effective force coefficient is reduced by the
    solidity ratio (epsilon = solid_area / gross_area).

    Parameters
    ----------
    V : float
        Basic wind speed (m/s). Must be > 0.
    fence_height : float
        Fence height s (m). Must be > 0.
    fence_length : float
        Fence length B (m). Must be > 0.
    solidity_ratio : float
        Ratio of solid area to gross area (0 < epsilon <= 1.0).
        1.0 = solid wall, ~0.5 = typical chain-link.
    exposure_category : str
        Exposure category: 'B', 'C', or 'D'.
    Kzt : float
        Topographic factor. Default 1.0.
    Kd : float
        Wind directionality factor. Default 0.85.
    Ke : float
        Ground elevation factor. Default 1.0.
    G : float
        Gust-effect factor. Default 0.85.
    clearance_height : float
        Clearance from ground to bottom of fence (m). Default 0.0.

    Returns
    -------
    FreestandingWallWindResult
        Complete wind analysis results with porosity reduction applied.

    Raises
    ------
    ValueError
        If any input is out of valid range.
    """
    if fence_height <= 0:
        raise ValueError(f"fence_height must be > 0, got {fence_height}")
    if fence_length <= 0:
        raise ValueError(f"fence_length must be > 0, got {fence_length}")
    if solidity_ratio <= 0 or solidity_ratio > 1.0:
        raise ValueError(
            f"solidity_ratio must be in (0, 1.0], got {solidity_ratio}"
        )
    if G <= 0:
        raise ValueError(f"G must be > 0, got {G}")
    if clearance_height < 0:
        raise ValueError(f"clearance_height must be >= 0, got {clearance_height}")

    s = fence_height
    B = fence_length

    # Height to top of fence
    z_top = clearance_height + s

    # Velocity pressure at top of fence
    vp = compute_velocity_pressure(V, z_top, exposure_category, Kzt, Kd, Ke)

    # Force coefficient (solid wall basis)
    B_over_s = B / s
    clearance_ratio = clearance_height / s
    Cf_solid = get_Cf_freestanding_wall(B_over_s, clearance_ratio)

    # Porosity reduction per Figure 29.3-1 Note 4
    Cf_effective = Cf_solid * solidity_ratio

    # Design wind pressure
    p = vp.qz_Pa * G * Cf_effective  # Pa

    # Force per unit length
    f = p * s / 1000.0  # kN/m

    # Total force
    F = f * B  # kN

    # Overturning moment per unit length about base
    moment_arm = clearance_height + s / 2.0
    M = f * moment_arm  # kN*m/m

    return FreestandingWallWindResult(
        velocity_pressure_Pa=vp.qz_Pa,
        velocity_pressure_kPa=vp.qz_kPa,
        wind_pressure_Pa=p,
        wind_pressure_kPa=p / 1000.0,
        force_per_unit_length_kN_m=f,
        total_force_kN=F,
        overturning_moment_kNm_per_m=M,
        Kz=vp.Kz,
        Kzt=vp.Kzt,
        Kd=vp.Kd,
        Ke=vp.Ke,
        G=G,
        Cf=Cf_effective,
        B_over_s=B_over_s,
        V_m_s=V,
        wall_height_m=s,
        wall_length_m=B,
        exposure_category=vp.exposure_category,
        clearance_height_m=clearance_height,
        solidity_ratio=solidity_ratio,
    )
