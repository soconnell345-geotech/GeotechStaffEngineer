"""
Result dataclasses for SOE wall analysis.

Each result includes summary() for human-readable output and to_dict()
for LLM agent consumption.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ============================================================================
# Stability check result
# ============================================================================

@dataclass
class StabilityCheckResult:
    """Result from a single stability check (basal heave, blowout, piping).

    Attributes
    ----------
    check_type : str
        "basal_heave_terzaghi", "basal_heave_bjerrum_eide",
        "bottom_blowout", or "piping".
    FOS : float
        Computed factor of safety.
    FOS_required : float
        Minimum required factor of safety.
    passes : bool
        True if FOS >= FOS_required.
    resistance : float
        Resisting quantity (numerator of FOS).
    demand : float
        Driving quantity (denominator of FOS).
    parameters : dict
        Input parameters and intermediate values for transparency.
    notes : list
        Warnings or special conditions.
    """
    check_type: str = ""
    FOS: float = 0.0
    FOS_required: float = 1.5
    passes: bool = False
    resistance: float = 0.0
    demand: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary string."""
        status = "PASS" if self.passes else "FAIL"
        lines = [
            "=" * 60,
            f"  STABILITY CHECK: {self.check_type.upper().replace('_', ' ')}",
            "=" * 60,
            "",
            f"  Factor of Safety:   FOS = {self.FOS:.3f}  [{status}]",
            f"  Required FOS:             {self.FOS_required:.2f}",
            f"  Resistance:         {self.resistance:.2f}",
            f"  Demand:             {self.demand:.2f}",
        ]
        if self.parameters:
            lines.append("")
            lines.append("  Parameters:")
            for k, v in self.parameters.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM agent consumption."""
        return {
            "check_type": self.check_type,
            "FOS": round(self.FOS, 3),
            "FOS_required": self.FOS_required,
            "passes": self.passes,
            "resistance": round(self.resistance, 2),
            "demand": round(self.demand, 2),
            "parameters": self.parameters,
            "notes": self.notes,
        }


# ============================================================================
# Anchor design result
# ============================================================================

@dataclass
class AnchorDesignResult:
    """Result from ground anchor design per GEC-4/PTI.

    Attributes
    ----------
    design_load_kN : float
        Design anchor load (kN).
    anchor_depth_m : float
        Depth of anchor head from ground surface (m).
    anchor_angle_deg : float
        Anchor inclination below horizontal (degrees).
    unbonded_length_m : float
        Required free (unbonded) length (m).
    bond_length_m : float
        Required bond length in grout zone (m).
    total_length_m : float
        Total anchor length = unbonded + bond (m).
    bond_stress_kPa : float
        Ultimate bond stress used (kPa).
    soil_type : str
        Soil/rock type used for bond stress.
    drill_diameter_mm : float
        Drill hole diameter (mm).
    tendon : dict
        Tendon selection details (type, strand count, capacities).
    proof_test_kN : float
        Proof test load = 1.33 × design load (kN).
    performance_test_kN : float
        Performance test load = 1.50 × design load (kN).
    notes : list
        Warnings or special conditions.
    """
    design_load_kN: float = 0.0
    anchor_depth_m: float = 0.0
    anchor_angle_deg: float = 15.0
    unbonded_length_m: float = 0.0
    bond_length_m: float = 0.0
    total_length_m: float = 0.0
    bond_stress_kPa: float = 0.0
    soil_type: str = ""
    drill_diameter_mm: float = 150.0
    tendon: Dict[str, Any] = field(default_factory=dict)
    proof_test_kN: float = 0.0
    performance_test_kN: float = 0.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 60,
            "  GROUND ANCHOR DESIGN RESULTS",
            "=" * 60,
            "",
            f"  Design load:          {self.design_load_kN:.1f} kN",
            f"  Anchor depth:         {self.anchor_depth_m:.2f} m",
            f"  Anchor angle:         {self.anchor_angle_deg:.1f} deg",
            "",
            f"  Unbonded length:      {self.unbonded_length_m:.2f} m",
            f"  Bond length:          {self.bond_length_m:.2f} m",
            f"  Total anchor length:  {self.total_length_m:.2f} m",
            "",
            f"  Bond stress:          {self.bond_stress_kPa:.1f} kPa"
            f"  ({self.soil_type})",
            f"  Drill diameter:       {self.drill_diameter_mm:.0f} mm",
        ]
        if self.tendon:
            lines.append("")
            lines.append("  Tendon:")
            desc = self.tendon.get("description", "")
            if desc:
                lines.append(f"    Type: {desc}")
            n = self.tendon.get("n_strands", 0)
            if n > 0:
                lines.append(f"    Strands/bars: {n}")
            guts = self.tendon.get("total_GUTS_kN", 0)
            if guts > 0:
                lines.append(f"    Total GUTS: {guts:.1f} kN")
            dl_pct = self.tendon.get("design_load_pct_GUTS", 0)
            if dl_pct > 0:
                lines.append(f"    DL / GUTS: {dl_pct:.1f}%")

        lines.extend([
            "",
            f"  Proof test load:       {self.proof_test_kN:.1f} kN (133% DL)",
            f"  Performance test load: {self.performance_test_kN:.1f} kN (150% DL)",
        ])
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  Note: {note}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM agent consumption."""
        return {
            "design_load_kN": self.design_load_kN,
            "anchor_depth_m": self.anchor_depth_m,
            "anchor_angle_deg": self.anchor_angle_deg,
            "unbonded_length_m": self.unbonded_length_m,
            "bond_length_m": self.bond_length_m,
            "total_length_m": self.total_length_m,
            "bond_stress_kPa": self.bond_stress_kPa,
            "soil_type": self.soil_type,
            "drill_diameter_mm": self.drill_diameter_mm,
            "tendon": self.tendon,
            "proof_test_kN": self.proof_test_kN,
            "performance_test_kN": self.performance_test_kN,
            "notes": self.notes,
        }


# ============================================================================
# Braced excavation result
# ============================================================================

@dataclass
class BracedExcavationResult:
    """Results from a multi-level braced excavation analysis.

    Attributes
    ----------
    excavation_depth : float
        Total depth of excavation H (m).
    n_support_levels : int
        Number of bracing levels.
    apparent_pressure_type : str
        "sand", "soft_clay", or "stiff_clay".
    max_apparent_pressure_kPa : float
        Peak ordinate of the apparent pressure envelope (kPa).
    support_reactions : list
        List of dicts with "depth_m", "load_kN_per_m", "type".
    max_moment_kNm_per_m : float
        Maximum wall bending moment (kN·m/m of wall).
    max_moment_depth_m : float
        Approximate depth of maximum moment (m).
    max_shear_kN_per_m : float
        Maximum wall shear (kN/m of wall).
    required_embedment_m : float
        Required embedment below excavation (m).
    total_wall_length_m : float
        H + embedment (m).
    required_Sx_cm3 : float
        Required section modulus for ASD (cm³).
    """
    excavation_depth: float = 0.0
    n_support_levels: int = 0
    apparent_pressure_type: str = ""
    max_apparent_pressure_kPa: float = 0.0
    support_reactions: List[Dict[str, Any]] = field(default_factory=list)
    max_moment_kNm_per_m: float = 0.0
    max_moment_depth_m: float = 0.0
    max_shear_kN_per_m: float = 0.0
    required_embedment_m: float = 0.0
    total_wall_length_m: float = 0.0
    required_Sx_cm3: float = 0.0

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 60,
            "  BRACED EXCAVATION ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Excavation depth:        H = {self.excavation_depth:.2f} m",
            f"  Number of support levels:    {self.n_support_levels}",
            f"  Apparent pressure type:      {self.apparent_pressure_type}",
            f"  Max apparent pressure:       {self.max_apparent_pressure_kPa:.2f} kPa",
            "",
            "  Support Reactions:",
        ]
        for rxn in self.support_reactions:
            lines.append(
                f"    z = {rxn['depth_m']:.2f} m  |  "
                f"{rxn['load_kN_per_m']:.1f} kN/m  |  {rxn['type']}"
            )
        lines.extend([
            "",
            f"  Max bending moment:  {self.max_moment_kNm_per_m:.2f} kN·m/m"
            f"  (at z = {self.max_moment_depth_m:.2f} m)",
            f"  Max shear:           {self.max_shear_kN_per_m:.2f} kN/m",
            f"  Required Sx (ASD):   {self.required_Sx_cm3:.1f} cm³",
            "",
            f"  Required embedment:  {self.required_embedment_m:.2f} m",
            f"  Total wall length:   {self.total_wall_length_m:.2f} m",
            "",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM agent consumption."""
        return {
            "excavation_depth_m": self.excavation_depth,
            "n_support_levels": self.n_support_levels,
            "apparent_pressure_type": self.apparent_pressure_type,
            "max_apparent_pressure_kPa": self.max_apparent_pressure_kPa,
            "support_reactions": self.support_reactions,
            "max_moment_kNm_per_m": self.max_moment_kNm_per_m,
            "max_moment_depth_m": self.max_moment_depth_m,
            "max_shear_kN_per_m": self.max_shear_kN_per_m,
            "required_embedment_m": self.required_embedment_m,
            "total_wall_length_m": self.total_wall_length_m,
            "required_Sx_cm3": self.required_Sx_cm3,
        }


@dataclass
class CantileverExcavationResult:
    """Results from a cantilever (unbraced) excavation wall analysis.

    Attributes
    ----------
    excavation_depth : float
        Total depth of excavation H (m).
    FOS_passive : float
        Factor of safety on passive resistance.
    Ka : float
        Active earth pressure coefficient.
    Kp : float
        Passive earth pressure coefficient.
    required_embedment_m : float
        Required embedment below excavation (m).
    total_wall_length_m : float
        H + embedment (m).
    max_moment_kNm_per_m : float
        Maximum wall bending moment (kN·m/m).
    max_shear_kN_per_m : float
        Maximum wall shear (kN/m).
    required_Sx_cm3 : float
        Required section modulus for ASD (cm³).
    """
    excavation_depth: float = 0.0
    FOS_passive: float = 1.5
    Ka: float = 0.0
    Kp: float = 0.0
    required_embedment_m: float = 0.0
    total_wall_length_m: float = 0.0
    max_moment_kNm_per_m: float = 0.0
    max_shear_kN_per_m: float = 0.0
    required_Sx_cm3: float = 0.0

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 60,
            "  CANTILEVER EXCAVATION ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Excavation depth:   H = {self.excavation_depth:.2f} m",
            f"  FOS (passive):          {self.FOS_passive:.2f}",
            f"  Ka = {self.Ka:.4f}    Kp = {self.Kp:.4f}",
            "",
            f"  Max bending moment: {self.max_moment_kNm_per_m:.2f} kN·m/m",
            f"  Max shear:          {self.max_shear_kN_per_m:.2f} kN/m",
            f"  Required Sx (ASD):  {self.required_Sx_cm3:.1f} cm³",
            "",
            f"  Required embedment: {self.required_embedment_m:.2f} m",
            f"  Total wall length:  {self.total_wall_length_m:.2f} m",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM agent consumption."""
        return {
            "excavation_depth_m": self.excavation_depth,
            "FOS_passive": self.FOS_passive,
            "Ka": self.Ka,
            "Kp": self.Kp,
            "required_embedment_m": self.required_embedment_m,
            "total_wall_length_m": self.total_wall_length_m,
            "max_moment_kNm_per_m": self.max_moment_kNm_per_m,
            "max_shear_kN_per_m": self.max_shear_kN_per_m,
            "required_Sx_cm3": self.required_Sx_cm3,
        }
