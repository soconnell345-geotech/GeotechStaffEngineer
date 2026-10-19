"""
Result dataclasses for SOE wall analysis.

Each result includes summary() for human-readable output and to_dict()
for LLM agent consumption.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

#: Printed in summary() where free earth support found no moment balance.
_NOT_FOUND = "not found (no moment balance; see notes)"


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
    test_load_factor : float
        The test load as a multiple of the design load (GEC-4 Sec.
        7.4.2.2, p. 143; default 1.33; GM-185).
    test_load_kN : float
        Test load = test_load_factor × design load (kN): the maximum load
        of the proof and the performance test.
    proof_test_kN : float
        Proof test load, the test load (kN).
    performance_test_kN : float
        Performance test load, the test load (kN). Until GM-185 1.50 ×
        design load, fixed.
    notes : list
        Warnings or special conditions.
    choices : list of dict
        What the design chose (GM-185, ``geotech_common.choices``;
        ``anchor_design.anchor_choices``).
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
    test_load_factor: float = 1.33
    test_load_kN: float = 0.0
    proof_test_kN: float = 0.0
    performance_test_kN: float = 0.0
    notes: List[str] = field(default_factory=list)
    choices: List[Dict[str, Any]] = field(default_factory=list)

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
            f"  Test load:             {self.test_load_kN:.1f} kN "
            f"({self.test_load_factor:g} DL; proof and performance tests)",
        ])
        if self.choices:
            from geotech_common.choices import choices_lines
            lines.extend(choices_lines(self.choices))
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
            "test_load_factor": self.test_load_factor,
            "test_load_kN": self.test_load_kN,
            "proof_test_kN": self.proof_test_kN,
            "performance_test_kN": self.performance_test_kN,
            "notes": self.notes,
            "choices": self.choices,
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
    converged : bool
        False when free earth support found no moment balance within 6H:
        the embedment (and, with no FS = 1 balance, the maximum moment,
        shear and section) are then None and the first note says why.
    """
    excavation_depth: float = 0.0
    n_support_levels: int = 0
    apparent_pressure_type: str = ""
    max_apparent_pressure_kPa: float = 0.0
    support_reactions: List[Dict[str, Any]] = field(default_factory=list)
    max_moment_kNm_per_m: Optional[float] = 0.0
    max_moment_depth_m: Optional[float] = 0.0
    max_shear_kN_per_m: Optional[float] = 0.0
    required_embedment_m: Optional[float] = 0.0
    total_wall_length_m: Optional[float] = 0.0
    required_Sx_cm3: Optional[float] = 0.0
    embedment_FS: float = 1.5
    surcharge_pressure_kPa: float = 0.0
    water_pressure_included: bool = False
    free_earth: Dict[str, Any] = field(default_factory=dict)
    converged: bool = True
    notes: List[str] = field(default_factory=list)
    # What the analysis used, for its calc package (BACKLOG GM-41); not in
    # to_dict, so the tool's output is as before.
    #: The apparent-pressure envelope (select_apparent_pressure's dict, with
    #: the surcharge and water terms the design pressure adds).
    apparent_pressure: Dict[str, Any] = field(default_factory=dict, repr=False)
    #: Per layer: top_m, bottom_m, unit_weight, friction_angle, cohesion,
    #: Ka, Kp (the Rankine coefficients of the free earth body below the
    #: excavation, from the profile embedment_detail solves).
    layers_used: List[Dict[str, Any]] = field(default_factory=list,
                                              repr=False)
    #: One entry per span above the excavation: z_top_m, z_bot_m,
    #: load_kN_per_m, centroid_below_top_m, to_upper_kN_per_m,
    #: to_lower_kN_per_m (the tributary split).
    spans: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    #: The design pressure, shear and moment over the spans above the
    #: excavation: {"z", "p", "V", "M"}, V and M in the free earth body's
    #: sign (push toward the excavation positive).
    diagram: Optional[Dict[str, Any]] = field(default=None, repr=False)
    #: The free-earth body below the lowest support (FS = 1): {"z", "p_net",
    #: "V", "M"}, or None.
    free_earth_diagram: Optional[Dict[str, Any]] = field(default=None,
                                                         repr=False)

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
        if self.max_moment_kNm_per_m is None:
            lines.extend([
                "",
                f"  Max bending moment:  {_NOT_FOUND}",
                f"  Required Sx (ASD):   {_NOT_FOUND}",
            ])
        else:
            lines.extend([
                "",
                f"  Max bending moment:  {self.max_moment_kNm_per_m:.2f} kN·m/m"
                f"  (at z = {self.max_moment_depth_m:.2f} m)",
                f"  Max shear:           {self.max_shear_kN_per_m:.2f} kN/m",
                f"  Required Sx (ASD):   {self.required_Sx_cm3:.1f} cm³",
            ])
        if self.required_embedment_m is None:
            lines.extend([
                "",
                f"  Required embedment:  {_NOT_FOUND}"
                f"  (free earth support, FS = {self.embedment_FS:g})",
                "",
            ])
        else:
            lines.extend([
                "",
                f"  Required embedment:  {self.required_embedment_m:.2f} m"
                f"  (free earth support, FS = {self.embedment_FS:g})",
                f"  Total wall length:   {self.total_wall_length_m:.2f} m",
                "",
            ])
        lines.extend(f"  Note: {n}" for n in self.notes)
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM agent consumption."""
        return {
            "excavation_depth_m": self.excavation_depth,
            "n_support_levels": self.n_support_levels,
            "apparent_pressure_type": self.apparent_pressure_type,
            "max_apparent_pressure_kPa": self.max_apparent_pressure_kPa,
            "surcharge_pressure_kPa": self.surcharge_pressure_kPa,
            "water_pressure_included": self.water_pressure_included,
            "support_reactions": self.support_reactions,
            "max_moment_kNm_per_m": self.max_moment_kNm_per_m,
            "max_moment_depth_m": self.max_moment_depth_m,
            "max_shear_kN_per_m": self.max_shear_kN_per_m,
            "required_embedment_m": self.required_embedment_m,
            "embedment_FS": self.embedment_FS,
            "total_wall_length_m": self.total_wall_length_m,
            "required_Sx_cm3": self.required_Sx_cm3,
            "free_earth": self.free_earth,
            "converged": self.converged,
            "notes": self.notes,
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
    embedment_converged_m : float
        D0, the depth of the rotation point O below the excavation (m).
    embedment_increase : float
        required_embedment_m / D0 (1.2 per Caltrans / AASHTO 3.11.5.6).
    max_moment_depth_m : float
        Depth of the maximum moment below the top of the wall (m).
    net_force_at_O_kN_per_m : float
        Driving minus resisting force to O; positive means embedment must
        increase (Caltrans 7-5.02 step 4).
    converged : bool
        False when no D0 within 6H balances the moments about O: the wall
        cannot stand as a cantilever, every quantity above except Ka and Kp
        is None, and the first note says why.
    notes : list of str
        Method and applicability notes.

    Ka and Kp are those of the layer the wall is embedded in (at the
    excavation level); every depth uses its own layer's coefficients.
    """
    excavation_depth: float = 0.0
    FOS_passive: float = 1.5
    Ka: float = 0.0
    Kp: float = 0.0
    required_embedment_m: Optional[float] = 0.0
    total_wall_length_m: Optional[float] = 0.0
    max_moment_kNm_per_m: Optional[float] = 0.0
    max_shear_kN_per_m: Optional[float] = 0.0
    required_Sx_cm3: Optional[float] = 0.0
    embedment_converged_m: Optional[float] = 0.0
    embedment_increase: float = 1.2
    max_moment_depth_m: Optional[float] = 0.0
    net_force_at_O_kN_per_m: Optional[float] = 0.0
    converged: bool = True
    notes: List[str] = field(default_factory=list)
    # What the analysis used, for its calc package (BACKLOG GM-41); not in
    # to_dict, so the tool's output is as before.
    #: Per layer: top_m, bottom_m, unit_weight, friction_angle, cohesion,
    #: Ka, Kp (the coefficients the pressures used).
    layers_used: List[Dict[str, Any]] = field(default_factory=list,
                                              repr=False)
    #: The pressures at the wall's break depths (layer bottoms, water
    #: tables, H, O): z_m, sv_retained_kPa, active_kPa, water_net_kPa,
    #: sv_excavation_kPa, passive_over_FS_kPa, net_kPa.
    pressure_points: List[Dict[str, Any]] = field(default_factory=list,
                                                  repr=False)
    #: Net pressure, shear and moment, top of wall to O: {"z", "p_net",
    #: "V", "M"}, or None without a moment balance.
    diagram: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 60,
            "  CANTILEVER EXCAVATION ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Excavation depth:   H = {self.excavation_depth:.2f} m",
            f"  FOS (passive):          {self.FOS_passive:.2f}",
            f"  Ka = {self.Ka:.4f}    Kp = {self.Kp:.4f}  (embedment layer)",
            "",
        ]
        if self.required_embedment_m is None:
            lines.extend([
                f"  Required embedment: {_NOT_FOUND}",
                "  Moment, shear and section: not reported",
                "",
            ])
        else:
            lines.extend([
                f"  Max bending moment: {self.max_moment_kNm_per_m:.2f} kN·m/m"
                f"  (at z = {self.max_moment_depth_m:.2f} m)",
                f"  Max shear:          {self.max_shear_kN_per_m:.2f} kN/m",
                f"  Required Sx (ASD):  {self.required_Sx_cm3:.1f} cm³",
                "",
                f"  Required embedment: {self.required_embedment_m:.2f} m"
                f"  (= {self.embedment_increase:g} x D0 = "
                f"{self.embedment_increase:g} x {self.embedment_converged_m:.2f})",
                f"  Total wall length:  {self.total_wall_length_m:.2f} m",
                "",
            ])
        lines.extend(f"  Note: {n}" for n in self.notes)
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM agent consumption."""
        return {
            "excavation_depth_m": self.excavation_depth,
            "FOS_passive": self.FOS_passive,
            "Ka": self.Ka,
            "Kp": self.Kp,
            "required_embedment_m": self.required_embedment_m,
            "embedment_converged_m": self.embedment_converged_m,
            "embedment_increase": self.embedment_increase,
            "total_wall_length_m": self.total_wall_length_m,
            "max_moment_kNm_per_m": self.max_moment_kNm_per_m,
            "max_moment_depth_m": self.max_moment_depth_m,
            "max_shear_kN_per_m": self.max_shear_kN_per_m,
            "net_force_at_O_kN_per_m": self.net_force_at_O_kN_per_m,
            "required_Sx_cm3": self.required_Sx_cm3,
            "method": "Caltrans T&S Manual 7-5.02 Simplified Method, layered "
                      "Rankine pressures with water",
            "converged": self.converged,
            "notes": self.notes,
        }
