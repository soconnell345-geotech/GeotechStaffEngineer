"""
Empirical HS / HSsmall constitutive parameter estimation for sands.

Brinkgreve, Engin & Engin (2010) published closed-form expressions for
every Hardening Soil (small-strain) model parameter as a function of
relative density alone, calibrated against high-quality laboratory data
on Toyoura, Ham River, Hostun and Ticino sands. They are intended as a
first estimate — site-specific testing governs for final design.

Reference:
    Brinkgreve, R.B.J., Engin, E. & Engin, H.K. (2010). "Validation of
    empirical formulas to derive model parameters for sands." Numerical
    Methods in Geotechnical Engineering (NUMGE 2010).
"""

__all__ = ["estimate_hs_parameters_sand"]


def estimate_hs_parameters_sand(relative_density_pct: float) -> dict:
    """HS / HSsmall parameters for sand from relative density.

    gamma_unsat = 15 + 4 Dr/100          [kN/m3]
    gamma_sat   = 19 + 1.6 Dr/100        [kN/m3]
    E50_ref     = 60e3 Dr/100            [kPa]
    Eoed_ref    = 60e3 Dr/100            [kPa]
    Eur_ref     = 180e3 Dr/100           [kPa]
    G0_ref      = 60e3 + 68e3 Dr/100     [kPa]
    m           = 0.7 - Dr/320           [-]
    gamma_07    = 1e-4 (2 - Dr/100)      [-]
    phi_eff     = 28 + 12.5 Dr/100       [deg]
    psi         = -2 + 12.5 Dr/100       [deg]
    Rf          = 1 - Dr/800             [-]

    Parameters
    ----------
    relative_density_pct : float
        Relative density Dr in PERCENT (10-100).

    Returns
    -------
    dict
        Keys: gamma_unsat_kN_m3, gamma_sat_kN_m3, E50_ref_kPa,
        Eoed_ref_kPa, Eur_ref_kPa, G0_ref_kPa, m, gamma_07, phi_eff_deg,
        psi_deg, Rf. Reference stress for the stiffness values is 100 kPa.

    Notes
    -----
    Feed E50_ref / Eur_ref / m / Rf / phi / psi to the fem2d HS material
    (``fem2d`` hardening-soil model); G0_ref and gamma_07 are the
    small-strain extension (HSsmall) parameters.
    """
    if not 10.0 <= relative_density_pct <= 100.0:
        raise ValueError(
            f"relative_density_pct outside the calibrated 10-100 range: "
            f"{relative_density_pct}")
    dr = relative_density_pct / 100.0
    return {
        "gamma_unsat_kN_m3": 15.0 + 4.0 * dr,
        "gamma_sat_kN_m3": 19.0 + 1.6 * dr,
        "E50_ref_kPa": 60e3 * dr,
        "Eoed_ref_kPa": 60e3 * dr,
        "Eur_ref_kPa": 180e3 * dr,
        "G0_ref_kPa": 60e3 + 68e3 * dr,
        "m": 0.7 - relative_density_pct / 320.0,
        "gamma_07": 1e-4 * (2.0 - dr),
        "phi_eff_deg": 28.0 + 12.5 * dr,
        "psi_deg": -2.0 + 12.5 * dr,
        "Rf": 1.0 - relative_density_pct / 800.0,
    }
