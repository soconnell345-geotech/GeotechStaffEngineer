"""
Beta (Effective Stress) Method for axial pile capacity.

GEC-12 Sec. 7.2.1.3.4 (pp. 255-258): fs = beta sigma'vo (Eq. 7-19) and
qp = Nt sigma'p (Eq. 7-20), beta and Nt from local experience (Steps 2a,
5a) or, without it, from Table 7-9 by phi' (Steps 2b, 5b). GEC-12 prints
no formula for beta or Nt from phi' alone, so the module has none (GM-282):
Table 7-9 is the only automatic route.

Applicable to any soil type (sand, clay, silt).

All units are SI: kPa, meters, kN.

References:
    Fellenius, B.H. (1991) — "Pile foundations"
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Section 7.2.1.3.4
"""

from axial_pile.nordlund import _note


def skin_friction_beta(sigma_v: float, beta: float,
                       pile_perimeter: float,
                       layer_thickness: float) -> float:
    """Compute skin friction using the beta method.

    Qs_layer = beta * sigma_v' * perimeter * thickness

    Parameters
    ----------
    sigma_v : float
        Effective vertical stress at center of layer (kPa).
    beta : float
        Beta coefficient.
    pile_perimeter : float
        Pile perimeter (m).
    layer_thickness : float
        Layer thickness (m).

    Returns
    -------
    float
        Skin friction from this layer (kN).
    """
    fs = beta * sigma_v
    return fs * pile_perimeter * layer_thickness


def end_bearing_beta(sigma_v_tip: float, Nt: float,
                     tip_area: float) -> float:
    """Compute end bearing using the effective stress method.

    Qt = Nt * sigma_v' * At

    Parameters
    ----------
    sigma_v_tip : float
        Effective vertical stress at pile tip (kPa).
    Nt : float
        Bearing capacity factor. Typical values:
        Sand: Nt = 30-150 (function of phi)
        Clay: Nt = 3-9
    tip_area : float
        Pile tip area (m²).

    Returns
    -------
    float
        End bearing capacity (kN).
    """
    return Nt * sigma_v_tip * tip_area


#: GEC-12 Table 7-9 (after Fellenius 2014, printed p. 256): the approximate
#: range of beta and Nt by soil type, with the phi' range each goes with:
#: soil type -> ((phi' low, high) deg, (beta low, high), (Nt low, high)).
TABLE_7_9 = {
    "clay": ((25.0, 30.0), (0.15, 0.35), (3.0, 30.0)),
    "silt": ((28.0, 34.0), (0.25, 0.50), (20.0, 40.0)),
    "sand": ((32.0, 40.0), (0.30, 0.90), (30.0, 150.0)),
    "gravel": ((35.0, 45.0), (0.35, 0.80), (60.0, 300.0)),
}


def table_7_9(soil: str, phi_deg: float, notes=None) -> tuple:
    """(beta, Nt) from GEC-12 Table 7-9 for ``soil`` ("clay", "silt",
    "sand" or "gravel") at phi' (degrees), as Steps 2b and 5b read it ("use
    Table 7-9 to estimate beta coefficient from phi' angle"): linear in
    phi' across the soil type's phi' range, low beta and Nt at its low
    phi'. A phi' outside the range takes the range's end, with a note."""
    if soil not in TABLE_7_9:
        raise ValueError(f"Table 7-9 soil type must be one of "
                         f"{list(TABLE_7_9)}, got '{soil}'")
    (p0, p1), (b0, b1), (n0, n1) = TABLE_7_9[soil]
    phi = min(max(phi_deg, p0), p1)
    if phi != phi_deg:
        text = (f"phi' = {phi_deg:g} deg is outside GEC-12 Table 7-9's "
                f"{soil} range ({p0:g}-{p1:g} deg); beta and Nt were taken "
                f"at {phi:g} deg, the range's end.")
        _note(notes, text)
    w = (phi - p0) / (p1 - p0)
    return b0 + w * (b1 - b0), n0 + w * (n1 - n0)
