"""
Property-variability knowledge base: published COV ranges as DATA with
provenance.

Datasets
--------
1. ``duncan_2000``  — Duncan (2000) Table 3 "Values of COV for geotechnical
   properties and in situ tests" (sources Harr 1984; Kulhawy 1992; Lacasse &
   Nadim 1997; Benson et al. 1999; Duncan 2000). Verified against the
   published paper.
2. ``tc304_2021``   — ISSMGE TC304 (2021) "State-of-the-art review of
   inherent variability and uncertainty in geotechnical properties and
   models", Tables 1.2 (clay), 1.3 (sand), 1.4 (rock): site-specific COV
   range and mean per property (combined Appendix + EPRI TR-105000 data).
3. ``transformation`` — transformation (correlation-model) uncertainty,
   Phoon & Kulhawy (1999b) examples as quoted in UFC 3-220-20 sec. 7-3.1.3.
4. ``measurement``  — total in-situ test variability ranges (Kulhawy 1992;
   Harr 1984; Kulhawy 1992), as reproduced in Duncan (2000) Table 3.

All COV values are stored in PERCENT, as published. Combine components with
:func:`reliability.stats.combined_cov` (which takes fractions —
divide by 100).

References
----------
Duncan, J.M. (2000). "Factors of safety and reliability in geotechnical
    engineering." J. Geotech. Geoenviron. Eng., 126(4), 307-316, Table 3 (p. 310; table number verified in-hand 2026-07-18, module_work/wiki_verification/duncan_2000_cov.md).
Phoon, K.K. & Kulhawy, F.H. (1999a, b). Can. Geotech. J., 36(4), 612-624 and
    625-639.
ISSMGE-TC304 (2021). State-of-the-art review of inherent variability and
    uncertainty in geotechnical properties and models. ISSMGE TC304 report,
    March 2021 (Ching & Schweckendiek, eds.).
UFC 3-220-20 (2025). Foundations and Earth Structures, ch. 7, sec. 7-3.1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CovEntry:
    """One published COV guidance row (COV values in PERCENT)."""
    property: str            # canonical key, e.g. "su", "phi", "gamma"
    label: str               # human-readable property / test description
    cov_min_pct: float
    cov_max_pct: float
    cov_mean_pct: Optional[float]   # published mean where available
    category: str            # inherent | site_specific | total_test | transformation
    soil_type: Optional[str]        # clay | sand | rock | None (general)
    test: Optional[str]      # associated test where applicable
    source: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _e(prop, label, lo, hi, mean, cat, soil, test, source):
    return CovEntry(prop, label, lo, hi, mean, cat, soil, test, source)


_DUNCAN = "Duncan (2000) Table 3"
_TC304_CLAY = "ISSMGE-TC304 (2021) Table 1.2 (clay, site-specific)"
_TC304_SAND = "ISSMGE-TC304 (2021) Table 1.3 (sand, site-specific)"
_TC304_ROCK = "ISSMGE-TC304 (2021) Table 1.4 (rock/rock mass, site-specific)"
_UFC_TRANS = ("Phoon & Kulhawy (1999b), quoted in UFC 3-220-20 "
              "sec. 7-3.1.3")
_KT = ("Harr (1984); Kulhawy (1992); reproduced in "
       "Duncan (2000) Table 3")

COV_DATABASE: List[CovEntry] = [
    # --- Duncan (2000) Table 3 -------------------------------------------
    _e("gamma", "Unit weight", 3, 7, None, "inherent", None, None,
       _DUNCAN + " [Harr 1984; Kulhawy 1992]"),
    _e("gamma_b", "Buoyant unit weight", 0, 10, None, "inherent", None, None,
       _DUNCAN + " [Lacasse & Nadim 1997; Duncan 2000]"),
    _e("phi", "Effective stress friction angle", 2, 13, None, "inherent",
       None, None, _DUNCAN + " [Harr 1984; Kulhawy 1992]"),
    _e("su", "Undrained shear strength", 13, 40, None, "inherent", None,
       None, _DUNCAN + " [Harr 1984; Kulhawy 1992; Lacasse & Nadim 1997]"),
    _e("su_ratio", "Undrained strength ratio su/sigma'v", 5, 15, None,
       "inherent", None, None,
       _DUNCAN + " [Lacasse & Nadim 1997; Duncan 2000]"),
    _e("Cc", "Compression index", 10, 37, None, "inherent", None, None,
       _DUNCAN + " [Harr 1984; Kulhawy 1992; Duncan 2000]"),
    _e("pc", "Preconsolidation pressure", 10, 35, None, "inherent", None,
       None, _DUNCAN + " [Harr 1984; Lacasse & Nadim 1997; Duncan 2000]"),
    _e("k_sat", "Coefficient of permeability, saturated clay", 68, 90, None,
       "inherent", "clay", None, _DUNCAN + " [Harr 1984; Duncan 2000]"),
    _e("k_unsat", "Coefficient of permeability, partly saturated clay",
       130, 240, None, "inherent", "clay", None,
       _DUNCAN + " [Harr 1984; Benson et al. 1999]"),
    _e("cv", "Coefficient of consolidation", 33, 68, None, "inherent",
       "clay", None, _DUNCAN + " [Duncan 2000]"),
    # --- in-situ tests (total test variability) ---------------------------
    _e("N", "Standard penetration test blow count", 15, 45, None,
       "total_test", None, "SPT", _KT),
    _e("qc", "Electric cone penetration test tip resistance", 5, 15, None,
       "total_test", None, "CPT (electric)", _KT),
    _e("qc", "Mechanical cone penetration test tip resistance", 15, 37, None,
       "total_test", None, "CPT (mechanical)", _KT),
    _e("q_dmt", "Dilatometer test tip resistance", 5, 15, None,
       "total_test", None, "DMT", _KT),
    _e("su", "Vane shear test undrained strength", 10, 20, None,
       "total_test", None, "VST", _KT),
    # --- ISSMGE TC304 (2021) site-specific COV, clay (Table 1.2) ----------
    _e("LL", "Liquid limit", 3.4, 39, 15.6, "site_specific", "clay", "lab",
       _TC304_CLAY),
    _e("PL", "Plastic limit", 2.9, 38.1, 13.5, "site_specific", "clay",
       "lab", _TC304_CLAY),
    _e("PI", "Plasticity index", 6.5, 57, 23.5, "site_specific", "clay",
       "lab", _TC304_CLAY),
    _e("w", "Natural water content", 3.5, 46, 15.3, "site_specific", "clay",
       "lab", _TC304_CLAY),
    _e("LI", "Liquidity index", 5.8, 88, 24.5, "site_specific", "clay",
       "lab", _TC304_CLAY),
    _e("OCR", "Overconsolidation ratio", 1.2, 39, 17.8, "site_specific",
       "clay", None, _TC304_CLAY),
    _e("Cc", "Compression index", 18.1, 47.3, 35.6, "site_specific", "clay",
       "oedometer", _TC304_CLAY),
    _e("phi", "Friction angle (clay)", 10, 50, 21.3, "site_specific",
       "clay", None, _TC304_CLAY),
    _e("su", "Undrained shear strength (clay)", 6, 56, 28.2,
       "site_specific", "clay", None, _TC304_CLAY),
    _e("su_ratio", "Undrained strength ratio su/sigma'v", 3.2, 39.4, 20.8,
       "site_specific", "clay", None, _TC304_CLAY),
    _e("St", "Sensitivity", 12.4, 63.4, 30.8, "site_specific", "clay",
       None, _TC304_CLAY),
    _e("qt", "Corrected cone resistance (clay)", 2, 17, 7.9,
       "site_specific", "clay", "CPTu", _TC304_CLAY),
    _e("N", "SPT blow count (clay)", 15.9, 57, 30.7, "site_specific",
       "clay", "SPT", _TC304_CLAY),
    _e("K0", "At-rest earth pressure coefficient (clay)", 2.4, 22, 13.5,
       "site_specific", "clay", None, _TC304_CLAY),
    # --- ISSMGE TC304 (2021) site-specific COV, sand (Table 1.3) ----------
    _e("e", "Void ratio (sand)", 7, 19.9, 11.1, "site_specific", "sand",
       "lab", _TC304_SAND),
    _e("phi", "Friction angle (sand)", 4.2, 12.5, 7.9, "site_specific",
       "sand", None, _TC304_SAND),
    _e("qc", "Cone tip resistance (sand)", 17, 81, 39.7, "site_specific",
       "sand", "CPT", _TC304_SAND),
    _e("N", "SPT blow count (sand)", 18.4, 62, 34.3, "site_specific",
       "sand", "SPT", _TC304_SAND),
    _e("N160", "Corrected SPT (N1)60 (sand)", 16.5, 38.8, 32.2,
       "site_specific", "sand", "SPT", _TC304_SAND),
    _e("K0", "At-rest earth pressure coefficient (sand)", 25.8, 36.9, 33.1,
       "site_specific", "sand", None, _TC304_SAND),
    # --- ISSMGE TC304 (2021) site-specific COV, rock (Table 1.4) ----------
    _e("gamma", "Unit weight (rock)", 0.4, 21.5, 5.2, "site_specific",
       "rock", "lab", _TC304_ROCK),
    _e("sigma_ci", "Uniaxial compressive strength (intact rock)", 5.7,
       108.4, 33.8, "site_specific", "rock", "UCS", _TC304_ROCK),
    _e("Ei", "Intact rock Young's modulus", 3.8, 73.7, 33.4,
       "site_specific", "rock", "lab", _TC304_ROCK),
    _e("RQD", "Rock quality designation", 4.8, 114.8, 29.9,
       "site_specific", "rock", "core logging", _TC304_ROCK),
    _e("RMR", "Rock mass rating", 4.7, 46.8, 21.3, "site_specific", "rock",
       None, _TC304_ROCK),
    _e("GSI", "Geological strength index", 3.0, 57.0, 19.9,
       "site_specific", "rock", None, _TC304_ROCK),
    _e("Is50", "Point load index Is50", 5.1, 91.5, 34.4, "site_specific",
       "rock", "point load", _TC304_ROCK),
    # --- transformation uncertainty (Phoon & Kulhawy 1999b via UFC) -------
    _e("su", "su correlated from corrected vane shear", 7.5, 15, None,
       "transformation", "clay", "VST", _UFC_TRANS),
    _e("su", "su correlated from corrected CPT tip resistance", 29, 35,
       None, "transformation", "clay", "CPT", _UFC_TRANS),
    _e("su", "su correlated from SPT", 15, 15, 15, "transformation",
       "clay", "SPT", _UFC_TRANS),
]

# alias -> canonical property key
ALIASES: Dict[str, str] = {
    "friction_angle": "phi", "phi_prime": "phi", "effective_friction_angle":
    "phi", "undrained_shear_strength": "su", "cu": "su", "unit_weight":
    "gamma", "buoyant_unit_weight": "gamma_b", "water_content": "w",
    "natural_water_content": "w", "liquid_limit": "LL", "plastic_limit":
    "PL", "plasticity_index": "PI", "liquidity_index": "LI", "spt": "N",
    "spt_n": "N", "blow_count": "N", "cpt": "qc", "cone_resistance": "qc",
    "tip_resistance": "qc", "compression_index": "Cc",
    "preconsolidation_pressure": "pc", "permeability": "k_sat",
    "hydraulic_conductivity": "k_sat", "coefficient_of_consolidation": "cv",
    "undrained_strength_ratio": "su_ratio", "sensitivity": "St",
    "void_ratio": "e", "ucs": "sigma_ci", "uniaxial_compressive_strength":
    "sigma_ci", "youngs_modulus_rock": "Ei", "k0": "K0",
}

CATEGORIES = ("inherent", "site_specific", "total_test", "transformation")


def list_properties() -> List[str]:
    """Sorted list of canonical property keys with guidance available."""
    return sorted({e.property for e in COV_DATABASE})


def cov_guidance(property: str,
                 soil_type: Optional[str] = None,
                 test: Optional[str] = None,
                 category: Optional[str] = None) -> List[CovEntry]:
    """Published COV guidance rows for a property.

    Parameters
    ----------
    property : str
        Canonical key ('phi', 'su', 'gamma', 'N', 'qc', ...) or a common
        alias ('friction_angle', 'undrained_shear_strength', 'spt', ...).
    soil_type : str, optional
        Filter: 'clay', 'sand' or 'rock'.
    test : str, optional
        Filter by associated test (substring match, case-insensitive),
        e.g. 'SPT', 'CPT', 'VST'.
    category : str, optional
        One of 'inherent', 'site_specific', 'total_test', 'transformation'.

    Returns
    -------
    list of CovEntry
        COV values in percent, each with its published source.

    Examples
    --------
    >>> rows = cov_guidance('phi', soil_type='sand')
    >>> rows[0].cov_mean_pct
    7.9
    """
    key = ALIASES.get(property.lower(), None)
    if key is None:
        # exact canonical match (case-sensitive keys like 'N', 'LL')
        canon = {e.property for e in COV_DATABASE}
        if property in canon:
            key = property
        elif property.upper() in canon:
            key = property.upper()
        elif property.lower() in canon:
            key = property.lower()
        else:
            raise ValueError(
                f"No COV guidance for '{property}'. Known properties: "
                f"{list_properties()}; aliases: {sorted(ALIASES)}.")
    if category is not None and category not in CATEGORIES:
        raise ValueError(
            f"category must be one of {CATEGORIES}, got '{category}'.")
    out = [e for e in COV_DATABASE if e.property == key]
    if soil_type is not None:
        out = [e for e in out
               if e.soil_type is None or e.soil_type == soil_type.lower()]
    if test is not None:
        out = [e for e in out
               if e.test is not None and test.lower() in e.test.lower()]
    if category is not None:
        out = [e for e in out if e.category == category]
    return out
