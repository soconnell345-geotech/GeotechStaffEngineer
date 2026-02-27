"""
Velocity pressure calculations per ASCE 7-22 Chapter 26.

Functions:
    compute_Kz       - Velocity pressure exposure coefficient (Table 26.10-1)
    compute_Kzt      - Topographic factor (Section 26.8)
    compute_Ke       - Ground elevation factor (Table 26.9-1)
    compute_velocity_pressure - Complete velocity pressure qz (Eq. 26.10-1)
"""

import math

from wind_loads.results import VelocityPressureResult

# ---------------------------------------------------------------------------
# Exposure constants (Table 26.10-1, Note 1)
# ---------------------------------------------------------------------------

_EXPOSURE_CONSTANTS = {
    "B": {"alpha": 7.0, "zg": 365.76},   # zg = 1200 ft
    "C": {"alpha": 9.5, "zg": 274.32},   # zg = 900 ft
    "D": {"alpha": 11.5, "zg": 213.36},  # zg = 700 ft
}

# Minimum height for Kz evaluation (Section 26.10.1: 4.6 m = 15 ft)
_Z_MIN = 4.6


def compute_Kz(z: float, exposure_category: str) -> float:
    """Compute velocity pressure exposure coefficient Kz (Table 26.10-1).

    Parameters
    ----------
    z : float
        Height above ground level (m). Must be > 0. Values below 4.6 m
        are evaluated at z = 4.6 m per ASCE 7-22 Section 26.10.1.
    exposure_category : str
        Exposure category: 'B', 'C', or 'D'.

    Returns
    -------
    float
        Kz value (dimensionless).

    Raises
    ------
    ValueError
        If z <= 0 or exposure_category is invalid.
    """
    if z <= 0:
        raise ValueError(f"Height z must be > 0, got {z}")

    cat = exposure_category.upper().strip()
    if cat not in _EXPOSURE_CONSTANTS:
        raise ValueError(
            f"Invalid exposure_category '{exposure_category}'. "
            f"Must be 'B', 'C', or 'D'."
        )

    alpha = _EXPOSURE_CONSTANTS[cat]["alpha"]
    zg = _EXPOSURE_CONSTANTS[cat]["zg"]

    # Clamp z to minimum 4.6 m per Section 26.10.1
    z_eval = max(z, _Z_MIN)

    # Cap z at zg (gradient height)
    z_eval = min(z_eval, zg)

    Kz = 2.01 * (z_eval / zg) ** (2.0 / alpha)
    return Kz


def compute_Kzt(
    hill_shape: str = "none",
    H_hill: float = 0.0,
    Lh: float = 1.0,
    x_distance: float = 0.0,
    z_height: float = 0.0,
) -> float:
    """Compute topographic factor Kzt (ASCE 7-22 Section 26.8, Eq. 26.8-1).

    For flat terrain, returns 1.0. For hills/ridges/escarpments, computes
    Kzt = (1 + K1*K2*K3)^2 using Figure 26.8-1 parameters.

    Parameters
    ----------
    hill_shape : str
        Topographic feature: 'none' (flat), '2d_ridge', '2d_escarpment',
        or '3d_hill'. Default 'none'.
    H_hill : float
        Height of hill/escarpment (m). Must be >= 0.
    Lh : float
        Distance from crest to half-height on upwind side (m). Must be > 0.
    x_distance : float
        Horizontal distance from crest (m). Upwind is positive.
    z_height : float
        Height above local ground surface (m). Must be >= 0.

    Returns
    -------
    float
        Kzt value (dimensionless, >= 1.0).

    Raises
    ------
    ValueError
        If invalid hill_shape or parameters out of range.
    """
    shape = hill_shape.lower().strip()

    if shape == "none" or shape == "flat":
        return 1.0

    if H_hill < 0:
        raise ValueError(f"H_hill must be >= 0, got {H_hill}")
    if Lh <= 0:
        raise ValueError(f"Lh must be > 0, got {Lh}")
    if z_height < 0:
        raise ValueError(f"z_height must be >= 0, got {z_height}")

    if H_hill == 0:
        return 1.0

    # K1 factor — depends on H/Lh ratio and shape
    H_over_Lh = H_hill / Lh

    # K1 values from Figure 26.8-1 (simplified linear for H/Lh <= 0.5)
    if shape == "2d_ridge":
        K1 = min(1.04 * H_over_Lh, 0.72)
    elif shape == "2d_escarpment":
        K1 = min(0.75 * H_over_Lh, 0.36)
    elif shape == "3d_hill":
        K1 = min(0.95 * H_over_Lh, 0.37)
    else:
        raise ValueError(
            f"Invalid hill_shape '{hill_shape}'. "
            f"Must be 'none', '2d_ridge', '2d_escarpment', or '3d_hill'."
        )

    # K2 factor — horizontal attenuation
    # mu values from Figure 26.8-1
    if shape == "2d_ridge":
        mu = 1.0 / 1.5  # 1/mu = 1.5
    elif shape == "2d_escarpment":
        mu = 1.0 / 1.5
    else:  # 3d_hill
        mu = 1.0 / 1.5

    K2 = max(1.0 - abs(x_distance) / (mu * Lh), 0.0)

    # K3 factor — vertical attenuation
    # gamma values from Figure 26.8-1
    if shape == "2d_ridge":
        gamma_val = 3.0
    elif shape == "2d_escarpment":
        gamma_val = 2.5
    else:  # 3d_hill
        gamma_val = 4.0

    K3 = math.exp(-gamma_val * z_height / Lh)

    Kzt = (1.0 + K1 * K2 * K3) ** 2
    return Kzt


def compute_Ke(elevation_m: float = 0.0) -> float:
    """Compute ground elevation factor Ke (ASCE 7-22 Table 26.9-1).

    Parameters
    ----------
    elevation_m : float
        Ground elevation above sea level (m). Negative values allowed
        (below sea level areas). Default 0.0 (sea level, Ke=1.0).

    Returns
    -------
    float
        Ke value (dimensionless). Always <= 1.0 for positive elevations.
    """
    # Eq. from Table 26.9-1 note: Ke = e^(-0.0000362 * ze)
    # where ze is ground elevation in meters
    Ke = math.exp(-0.0000362 * elevation_m)
    return Ke


def compute_velocity_pressure(
    V: float,
    z: float,
    exposure_category: str,
    Kzt: float = 1.0,
    Kd: float = 0.85,
    Ke: float = 1.0,
) -> VelocityPressureResult:
    """Compute velocity pressure qz per ASCE 7-22 Eq. 26.10-1.

    qz = 0.613 * Kz * Kzt * Kd * Ke * V^2  [Pa, V in m/s]

    Parameters
    ----------
    V : float
        Basic wind speed (m/s). Must be > 0.
    z : float
        Height above ground level (m). Must be > 0.
    exposure_category : str
        Exposure category: 'B', 'C', or 'D'.
    Kzt : float
        Topographic factor. Default 1.0 (flat terrain).
    Kd : float
        Wind directionality factor. Default 0.85 (freestanding walls,
        Table 26.6-1).
    Ke : float
        Ground elevation factor. Default 1.0 (sea level).

    Returns
    -------
    VelocityPressureResult
        Complete velocity pressure results with all coefficients.

    Raises
    ------
    ValueError
        If V <= 0, z <= 0, Kzt < 1.0, Kd <= 0, Ke <= 0, or invalid exposure.
    """
    if V <= 0:
        raise ValueError(f"Wind speed V must be > 0, got {V}")
    if z <= 0:
        raise ValueError(f"Height z must be > 0, got {z}")
    if Kzt < 1.0:
        raise ValueError(f"Kzt must be >= 1.0, got {Kzt}")
    if Kd <= 0:
        raise ValueError(f"Kd must be > 0, got {Kd}")
    if Ke <= 0:
        raise ValueError(f"Ke must be > 0, got {Ke}")

    Kz = compute_Kz(z, exposure_category)

    # ASCE 7-22 Eq. 26.10-1 (SI units)
    qz = 0.613 * Kz * Kzt * Kd * Ke * V * V

    return VelocityPressureResult(
        qz_Pa=qz,
        qz_kPa=qz / 1000.0,
        V_m_s=V,
        z_m=z,
        exposure_category=exposure_category.upper().strip(),
        Kz=Kz,
        Kzt=Kzt,
        Kd=Kd,
        Ke=Ke,
    )
