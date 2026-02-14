"""
Hammer models and database for wave equation analysis.

Provides data classes for impact hammers (single-acting, diesel, hydraulic)
and a built-in database of common hammers used in North American practice.

All units are SI: kN, m, seconds, kg.

References:
    FHWA GEC-12, Chapter 12
    WEAP87 Manual (FHWA, Goble & Rausche)
    Manufacturer specifications (Vulcan, Delmag, ICE, APE)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Hammer:
    """Impact hammer for pile driving.

    Parameters
    ----------
    name : str
        Hammer model designation.
    ram_weight : float
        Ram weight (kN).
    stroke : float
        Rated stroke height (m).
    efficiency : float
        Energy transfer efficiency (0 to 1). Accounts for losses in
        the hammer mechanism. Typical: 0.67 for air/steam, 0.80 for
        diesel, 0.95 for hydraulic.
    hammer_type : str
        One of 'single_acting', 'diesel', 'hydraulic'.
    rated_energy : float, optional
        Rated energy (kN-m). If None, computed as ram_weight * stroke.
    ram_stiffness : float, optional
        Ram stiffness (kN/m). If None, treated as rigid ram.
    """
    name: str
    ram_weight: float
    stroke: float
    efficiency: float = 0.67
    hammer_type: str = "single_acting"
    rated_energy: Optional[float] = None
    ram_stiffness: Optional[float] = None

    @property
    def ram_mass(self) -> float:
        """Ram mass (kg)."""
        return self.ram_weight / 9.81 * 1000  # kN -> N -> kg

    @property
    def energy(self) -> float:
        """Rated energy (kN-m)."""
        if self.rated_energy is not None:
            return self.rated_energy
        return self.ram_weight * self.stroke

    @property
    def impact_velocity(self) -> float:
        """Ram velocity at impact (m/s).

        v = sqrt(2 * g * stroke * efficiency)
        For diesel, use rated energy instead of W*h.
        """
        import math
        if self.rated_energy is not None:
            # v = sqrt(2 * E_rated * eff / m)
            # E_rated in kN-m, m in kg
            E_joules = self.rated_energy * 1000 * self.efficiency  # kN-m -> N-m
            return math.sqrt(2 * E_joules / self.ram_mass)
        # Single-acting: free fall
        g = 9.81  # m/s^2
        return math.sqrt(2 * g * self.stroke * self.efficiency)


# ─── Built-in Hammer Database ───────────────────────────────────────

_HAMMER_DB = {
    # Vulcan single-acting air/steam hammers
    "Vulcan 06": Hammer("Vulcan 06", 28.9, 0.914, 0.67, "single_acting"),
    "Vulcan 08": Hammer("Vulcan 08", 35.6, 0.914, 0.67, "single_acting"),
    "Vulcan 010": Hammer("Vulcan 010", 44.5, 0.914, 0.67, "single_acting"),
    "Vulcan 012": Hammer("Vulcan 012", 53.4, 0.991, 0.67, "single_acting"),
    "Vulcan 014": Hammer("Vulcan 014", 62.3, 0.914, 0.67, "single_acting"),
    "Vulcan 020": Hammer("Vulcan 020", 89.0, 0.914, 0.67, "single_acting"),
    "Vulcan 060": Hammer("Vulcan 060", 267.0, 0.914, 0.67, "single_acting"),

    # Delmag diesel hammers
    "Delmag D12": Hammer("Delmag D12", 12.2, 2.50, 0.80, "diesel",
                         rated_energy=30.5),
    "Delmag D19-32": Hammer("Delmag D19-32", 19.0, 2.80, 0.80, "diesel",
                            rated_energy=53.2),
    "Delmag D30-32": Hammer("Delmag D30-32", 29.4, 2.80, 0.80, "diesel",
                            rated_energy=82.3),
    "Delmag D46-32": Hammer("Delmag D46-32", 45.1, 2.80, 0.80, "diesel",
                            rated_energy=126.3),
    "Delmag D62-22": Hammer("Delmag D62-22", 61.5, 2.50, 0.80, "diesel",
                            rated_energy=153.8),

    # ICE hydraulic hammers
    "ICE I-30V2": Hammer("ICE I-30V2", 29.4, 1.22, 0.95, "hydraulic"),
    "ICE I-46V2": Hammer("ICE I-46V2", 45.1, 1.22, 0.95, "hydraulic"),
}


def get_hammer(name: str) -> Hammer:
    """Look up a hammer from the built-in database.

    Parameters
    ----------
    name : str
        Hammer name (e.g. 'Vulcan 010', 'Delmag D30-32').

    Returns
    -------
    Hammer

    Raises
    ------
    KeyError
        If hammer name not found.
    """
    if name not in _HAMMER_DB:
        available = ", ".join(sorted(_HAMMER_DB.keys()))
        raise KeyError(f"Hammer '{name}' not found. Available: {available}")
    return _HAMMER_DB[name]


def list_hammers() -> list:
    """List all available hammers in the database.

    Returns
    -------
    list of str
        Sorted hammer names.
    """
    return sorted(_HAMMER_DB.keys())
