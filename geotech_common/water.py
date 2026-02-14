"""
Water-related utilities for geotechnical calculations.

Provides the unit weight of water and pore pressure calculations.
"""

# Unit weight of water in kN/m³
GAMMA_W: float = 9.81


def pore_pressure(depth_below_gwt: float, gamma_w: float = GAMMA_W) -> float:
    """Compute hydrostatic pore water pressure.

    Parameters
    ----------
    depth_below_gwt : float
        Depth below the groundwater table (m). Negative values return 0.
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default is 9.81.

    Returns
    -------
    float
        Pore water pressure u (kPa).
    """
    if depth_below_gwt <= 0.0:
        return 0.0
    return gamma_w * depth_below_gwt


def effective_unit_weight(gamma_total: float, gamma_w: float = GAMMA_W) -> float:
    """Compute effective (buoyant) unit weight for submerged soil.

    Parameters
    ----------
    gamma_total : float
        Total (saturated) unit weight (kN/m³).
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default is 9.81.

    Returns
    -------
    float
        Effective unit weight gamma' (kN/m³).
    """
    return gamma_total - gamma_w
