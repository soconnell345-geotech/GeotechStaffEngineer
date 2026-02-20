"""
Result dataclasses for geolysis_agent.

All results follow the standard pattern:
  - Frozen dataclass with default values
  - summary() -> str for human-readable output
  - to_dict() -> dict for JSON serialization
  - Optional plot methods
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationResult:
    """
    Result from USCS or AASHTO soil classification.

    Attributes
    ----------
    system : str
        Classification system ('uscs' or 'aashto').
    symbol : str
        Classification symbol (e.g., 'SW-SC', 'A-7-6(20)').
    description : str
        Verbal description of soil type.
    group_index : str or None
        AASHTO group index (None for USCS).
    liquid_limit : float or None
        Liquid limit (%).
    plastic_limit : float or None
        Plastic limit (%).
    plasticity_index : float or None
        Plasticity index (%).
    fines : float or None
        Fines content (% passing #200 sieve).
    sand : float or None
        Sand content (% passing #4, retained on #200). USCS only.
    """

    system: str = ""
    symbol: str = ""
    description: str = ""
    group_index: str = None
    liquid_limit: float = None
    plastic_limit: float = None
    plasticity_index: float = None
    fines: float = None
    sand: float = None

    def summary(self):
        """Return human-readable summary."""
        lines = [
            f"{self.system.upper()} Soil Classification",
            f"  Symbol: {self.symbol}",
            f"  Description: {self.description}",
        ]

        if self.group_index is not None:
            lines.append(f"  Group Index: {self.group_index}")

        lines.append("\nIndex Properties:")
        if self.liquid_limit is not None:
            lines.append(f"  Liquid Limit: {self.liquid_limit:.1f}%")
        if self.plastic_limit is not None:
            lines.append(f"  Plastic Limit: {self.plastic_limit:.1f}%")
        if self.plasticity_index is not None:
            lines.append(f"  Plasticity Index: {self.plasticity_index:.1f}%")
        if self.fines is not None:
            lines.append(f"  Fines Content: {self.fines:.1f}%")
        if self.sand is not None:
            lines.append(f"  Sand Content: {self.sand:.1f}%")

        return "\n".join(lines)

    def to_dict(self):
        """Return dict for JSON serialization."""
        return {
            "system": self.system,
            "symbol": self.symbol,
            "description": self.description,
            "group_index": self.group_index,
            "liquid_limit": self.liquid_limit,
            "plastic_limit": self.plastic_limit,
            "plasticity_index": self.plasticity_index,
            "fines": self.fines,
            "sand": self.sand,
        }


# ---------------------------------------------------------------------------
# SPTCorrectionResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SPTCorrectionResult:
    """
    Result from SPT N-value correction.

    Attributes
    ----------
    recorded_n : int
        Field-recorded SPT N-value.
    n60 : float
        Energy-corrected N-value (N60).
    n1_60 : float
        Overburden-corrected N-value (N1_60).
    n_corrected : float
        Final corrected N-value (after dilatancy if applied).
    energy_percentage : float
        Energy ratio used (decimal, 0-1).
    hammer_type : str
        Hammer type used in correction.
    sampler_type : str
        Sampler type used in correction.
    opc_method : str
        Overburden pressure correction method.
    eop_kpa : float
        Effective overburden pressure (kPa).
    dilatancy_applied : bool
        True if dilatancy correction was applied.
    """

    recorded_n: int = 0
    n60: float = 0.0
    n1_60: float = 0.0
    n_corrected: float = 0.0
    energy_percentage: float = 0.6
    hammer_type: str = ""
    sampler_type: str = ""
    opc_method: str = ""
    eop_kpa: float = 0.0
    dilatancy_applied: bool = False

    def summary(self):
        """Return human-readable summary."""
        lines = [
            "SPT N-Value Correction",
            f"  Recorded N: {self.recorded_n}",
            f"  Energy-corrected (N60): {self.n60:.1f}",
            f"  Overburden-corrected (N1_60): {self.n1_60:.1f}",
            f"  Final corrected N: {self.n_corrected:.1f}",
            "",
            "Correction Parameters:",
            f"  Energy ratio: {self.energy_percentage * 100:.0f}%",
            f"  Hammer type: {self.hammer_type}",
            f"  Sampler type: {self.sampler_type}",
            f"  Overburden method: {self.opc_method}",
            f"  Effective overburden pressure: {self.eop_kpa:.1f} kPa",
            f"  Dilatancy correction: {'Yes' if self.dilatancy_applied else 'No'}",
        ]
        return "\n".join(lines)

    def to_dict(self):
        """Return dict for JSON serialization."""
        return {
            "recorded_n": self.recorded_n,
            "n60": self.n60,
            "n1_60": self.n1_60,
            "n_corrected": self.n_corrected,
            "energy_percentage": self.energy_percentage,
            "hammer_type": self.hammer_type,
            "sampler_type": self.sampler_type,
            "opc_method": self.opc_method,
            "eop_kpa": self.eop_kpa,
            "dilatancy_applied": self.dilatancy_applied,
        }


# ---------------------------------------------------------------------------
# BearingCapacityResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BearingCapacityResult:
    """
    Result from bearing capacity analysis.

    Attributes
    ----------
    method : str
        Analysis method (e.g., 'bowles', 'vesic').
    bc_type : str
        Type of bearing capacity ('allowable_spt' or 'ultimate').
    bearing_capacity_kpa : float
        Bearing capacity (kPa). Ultimate or allowable depending on bc_type.
    allowable_load_kn : float or None
        Allowable load (kN). Only for allowable_spt type.
    depth_m : float
        Foundation depth (m).
    width_m : float
        Foundation width (m).
    shape : str
        Foundation shape.
    n_c : float or None
        Bearing capacity factor Nc (ultimate only).
    n_q : float or None
        Bearing capacity factor Nq (ultimate only).
    n_gamma : float or None
        Bearing capacity factor Nγ (ultimate only).
    factor_of_safety : float or None
        Factor of safety (ultimate only).
    corrected_spt_n : float or None
        Corrected SPT N-value (allowable_spt only).
    settlement_mm : float or None
        Tolerable settlement (mm, allowable_spt only).
    allowable_bearing_capacity_kpa : float or None
        Allowable BC from ultimate (ultimate only).
    """

    method: str = ""
    bc_type: str = ""
    bearing_capacity_kpa: float = 0.0
    allowable_load_kn: float = None
    depth_m: float = 0.0
    width_m: float = 0.0
    shape: str = ""
    n_c: float = None
    n_q: float = None
    n_gamma: float = None
    factor_of_safety: float = None
    corrected_spt_n: float = None
    settlement_mm: float = None
    allowable_bearing_capacity_kpa: float = None

    def summary(self):
        """Return human-readable summary."""
        lines = [
            f"Bearing Capacity Analysis ({self.method.upper()})",
            f"  Type: {self.bc_type.replace('_', ' ').title()}",
        ]

        if self.bc_type == "allowable_spt":
            lines.extend([
                f"  Allowable Bearing Capacity: {self.bearing_capacity_kpa:.1f} kPa",
                f"  Allowable Load: {self.allowable_load_kn:.1f} kN" if self.allowable_load_kn is not None else "",
                "",
                "Input Parameters:",
                f"  Corrected SPT N: {self.corrected_spt_n:.1f}" if self.corrected_spt_n is not None else "",
                f"  Tolerable Settlement: {self.settlement_mm:.1f} mm" if self.settlement_mm is not None else "",
                f"  Foundation Depth: {self.depth_m:.2f} m",
                f"  Foundation Width: {self.width_m:.2f} m",
                f"  Foundation Shape: {self.shape}",
            ])
        else:  # ultimate
            lines.extend([
                f"  Ultimate Bearing Capacity: {self.bearing_capacity_kpa:.1f} kPa",
                f"  Allowable Bearing Capacity: {self.allowable_bearing_capacity_kpa:.1f} kPa"
                if self.allowable_bearing_capacity_kpa else "",
                "",
                "Bearing Capacity Factors:",
                f"  Nc: {self.n_c:.2f}" if self.n_c is not None else "",
                f"  Nq: {self.n_q:.2f}" if self.n_q is not None else "",
                f"  Nγ: {self.n_gamma:.2f}" if self.n_gamma is not None else "",
                "",
                "Input Parameters:",
                f"  Foundation Depth: {self.depth_m:.2f} m",
                f"  Foundation Width: {self.width_m:.2f} m",
                f"  Foundation Shape: {self.shape}",
                f"  Factor of Safety: {self.factor_of_safety:.1f}"
                if self.factor_of_safety else "",
            ])

        return "\n".join([line for line in lines if line])

    def to_dict(self):
        """Return dict for JSON serialization."""
        return {
            "method": self.method,
            "bc_type": self.bc_type,
            "bearing_capacity_kpa": self.bearing_capacity_kpa,
            "allowable_load_kn": self.allowable_load_kn,
            "depth_m": self.depth_m,
            "width_m": self.width_m,
            "shape": self.shape,
            "n_c": self.n_c,
            "n_q": self.n_q,
            "n_gamma": self.n_gamma,
            "factor_of_safety": self.factor_of_safety,
            "corrected_spt_n": self.corrected_spt_n,
            "settlement_mm": self.settlement_mm,
            "allowable_bearing_capacity_kpa": self.allowable_bearing_capacity_kpa,
        }
