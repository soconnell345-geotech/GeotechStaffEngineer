"""
Hammer models and database for wave equation analysis.

Provides data classes for impact hammers (single-acting, diesel, hydraulic)
and a built-in database of common hammers used in North American practice,
with the ram weights, strokes and rated energies of GEC-12 Vol 2 Appendix C,
"Pile Hammer Information" (pp. 519-541), converted to SI.

All units are SI: kN, m, seconds, kg.

Every hammer is a rigid ram striking at v = sqrt(2 g h efficiency): GEC-12
Vol 2 Eq. 15-1 (p. 358) with the stroke times the hammer efficiency, the
"portion of the potential ram energy that is available in the form of
kinetic ram energy" (Sec. 12.6.2, p. 205). A diesel hammer's combustion
is not modelled.

References:
    FHWA GEC-12 Vol 2, Chapters 12 and 15, and Appendix C (pp. 519-541)
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
        Hammer efficiency (0 to 1): the share of the potential ram energy
        that reaches the ram as kinetic energy at impact (GEC-12 Vol 2
        p. 205). GEC-12 Vol 2 prints 0.67 for an air hammer and 0.80 for
        an open end diesel (Example 6, p. 185), and "relatively high (say
        0.95)" for a hydraulic hammer whose ram kinetic energy is
        measured and displayed (p. 205).
    hammer_type : str
        One of 'single_acting', 'diesel', 'hydraulic': a label only.
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

# Ram weight, stroke and rated energy as GEC-12 Vol 2 Appendix C prints
# them (kips, ft, kip-ft; hammer number and page in each comment), in kN, m
# and kN-m. Efficiencies: 0.67 air/steam, 0.80 open end diesel (GEC-12 Vol 2
# Example 6, p. 185). Before GM-294 the Vulcan 08 and 010 strokes were
# 0.914 m, and the diesels' strokes and rated energies (and the D19-32 and
# D62-22 rams) were not the print's.
_HAMMER_DB = {
    # Vulcan single-acting air/steam hammers (ECH; Nos. 206-215, pp. 522-523)
    "Vulcan 06": Hammer("Vulcan 06", 28.9, 0.914, 0.67, "single_acting"),    # 6.5 k, 3.00 ft
    "Vulcan 08": Hammer("Vulcan 08", 35.6, 0.991, 0.67, "single_acting"),    # 8 k, 3.25 ft
    "Vulcan 010": Hammer("Vulcan 010", 44.5, 0.991, 0.67, "single_acting"),  # 10 k, 3.25 ft
    "Vulcan 012": Hammer("Vulcan 012", 53.4, 0.991, 0.67, "single_acting"),  # 12 k, 3.25 ft
    "Vulcan 014": Hammer("Vulcan 014", 62.3, 0.914, 0.67, "single_acting"),  # 14 k, 3.00 ft
    "Vulcan 020": Hammer("Vulcan 020", 89.0, 0.914, 0.67, "single_acting"),  # 20 k, 3.00 ft
    "Vulcan 060": Hammer("Vulcan 060", 266.9, 0.914, 0.67, "single_acting"), # 60 k, 3.00 ft

    # Delmag open end diesel hammers (OED)
    "Delmag D12": Hammer("Delmag D12", 12.2, 2.505, 0.80, "diesel",
                         rated_energy=30.7),     # No. 3, p. 519: 2.75 k, 8.22 ft, 22.61 k-ft
    "Delmag D19-32": Hammer("Delmag D19-32", 17.8, 3.234, 0.80, "diesel",
                            rated_energy=57.5),  # No. 40, p. 520: 4.00 k, 10.61 ft, 42.44 k-ft
    "Delmag D30-32": Hammer("Delmag D30-32", 29.4, 3.484, 0.80, "diesel",
                            rated_energy=102.3), # No. 15, p. 519: 6.60 k, 11.43 ft, 75.44 k-ft
    "Delmag D46-32": Hammer("Delmag D46-32", 45.1, 3.673, 0.80, "diesel",
                            rated_energy=165.7), # No. 26, p. 519: 10.14 k, 12.05 ft, 122.19 k-ft
    "Delmag D62-22": Hammer("Delmag D62-22", 60.8, 3.673, 0.80, "diesel",
                            rated_energy=223.2), # No. 30, p. 519: 13.66 k, 12.05 ft, 164.60 k-ft

    # ICE I-series open end diesels, with stepwise fuel pump settings (GEC-12
    # Vol 2 Sec. 12.6.3, p. 206). Listed as hydraulic, 1.22 m stroke and 0.95
    # efficiency, before GM-294.
    "ICE I-30V2": Hammer("ICE I-30V2", 29.4, 3.505, 0.80, "diesel",
                         rated_energy=103.1),    # No. 1504, p. 536: 6.61 k, 11.50 ft, 76.05 k-ft
    "ICE I-46V2": Hammer("ICE I-46V2", 45.1, 3.600, 0.80, "diesel",
                         rated_energy=162.4),    # No. 1506, p. 536: 10.14 k, 11.81 ft, 119.77 k-ft
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
