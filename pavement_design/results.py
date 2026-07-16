"""Result dataclasses for the pavement_design module.

UNITS: US customary throughout (psi, pci, inches, kips, 18-kip ESALs) --
this module is a documented exception to the repo's SI convention because
the AASHTO 1993 Guide is US-customary native (see DESIGN.md and the
``geotech_references.aashto_1993`` precedent).
"""

from dataclasses import dataclass, field


def _fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


@dataclass
class DesignTrafficResult:
    """Design-lane cumulative 18-kip ESALs over the performance period."""

    w18_design_lane: float
    w18_two_way_total: float
    base_year_w18_two_way: float
    growth_rate_pct: float
    design_period_yr: float
    growth_factor: float
    directional_factor: float
    lane_factor: float
    lef_basis: str  # 'closed_form' | 'digitized_table' | 'truck_factors' | 'direct'
    axle_breakdown: list = field(default_factory=list)
    vehicle_breakdown: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "w18_design_lane": self.w18_design_lane,
            "w18_two_way_total": self.w18_two_way_total,
            "base_year_w18_two_way": self.base_year_w18_two_way,
            "growth_rate_pct": self.growth_rate_pct,
            "design_period_yr": self.design_period_yr,
            "growth_factor": self.growth_factor,
            "directional_factor": self.directional_factor,
            "lane_factor": self.lane_factor,
            "lef_basis": self.lef_basis,
            "axle_breakdown": self.axle_breakdown,
            "vehicle_breakdown": self.vehicle_breakdown,
            "notes": self.notes,
            "references": self.references,
        }

    def summary(self) -> str:
        lines = [
            "AASHTO 1993 design traffic (18-kip ESALs)",
            f"  Base-year two-way W18: {_fmt(self.base_year_w18_two_way, 0)}/yr",
            f"  Growth: {self.growth_rate_pct}%/yr over {self.design_period_yr} yr "
            f"(growth factor {_fmt(self.growth_factor)})",
            f"  Two-way total W18: {_fmt(self.w18_two_way_total, 0)}",
            f"  x DD {self.directional_factor} x DL {self.lane_factor} -> "
            f"design-lane W18 = {_fmt(self.w18_design_lane, 0)}",
            f"  LEF basis: {self.lef_basis}",
        ]
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


@dataclass
class FlexiblePavementResult:
    """AASHTO 1993 flexible pavement design (Part II, Ch 3, Figure 3.1/3.2)."""

    mode: str  # 'design' or 'check'
    w18: float
    reliability_pct: float
    zr: float
    so: float
    po: float
    pt: float
    delta_psi: float
    effective_mr_psi: float
    sn_required: float          # over the roadbed (the overall design SN)
    sn_provided: float          # from the final (rounded / supplied) section
    layers: list                # per-layer dicts (type, a, m, thickness_in, basis, ...)
    sn_stack: list              # SN required over each interface, top-down
    w18_capacity: float         # forward-check capacity of the provided section
    adequate: bool
    environmental: dict = None  # swelling/frost dPSI block (None if not used)
    effective_mr_detail: dict = None  # seasonal uf worksheet (if monthly given)
    minimums_applied: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "w18": self.w18,
            "reliability_pct": self.reliability_pct,
            "zr": self.zr,
            "so": self.so,
            "po": self.po,
            "pt": self.pt,
            "delta_psi": self.delta_psi,
            "effective_mr_psi": self.effective_mr_psi,
            "sn_required": self.sn_required,
            "sn_provided": self.sn_provided,
            "layers": self.layers,
            "sn_stack": self.sn_stack,
            "w18_capacity": self.w18_capacity,
            "adequate": self.adequate,
            "environmental": self.environmental,
            "effective_mr_detail": self.effective_mr_detail,
            "minimums_applied": self.minimums_applied,
            "notes": self.notes,
            "warnings": self.warnings,
            "references": self.references,
        }

    def summary(self) -> str:
        lines = [
            f"AASHTO 1993 flexible pavement {self.mode}",
            f"  W18 = {_fmt(self.w18, 0)} ESALs | R = {self.reliability_pct}% "
            f"(ZR = {self.zr}) | So = {self.so}",
            f"  dPSI = {self.delta_psi} (po {self.po} -> pt {self.pt}) | "
            f"effective roadbed MR = {_fmt(self.effective_mr_psi, 0)} psi",
            f"  SN required = {_fmt(self.sn_required)} | SN provided = "
            f"{_fmt(self.sn_provided)}",
        ]
        for lay in self.layers:
            lines.append(
                f"    {lay['layer_type']}: D = {_fmt(lay.get('thickness_in'))} in "
                f"(a = {_fmt(lay.get('a'), 3)}, m = {_fmt(lay.get('m'), 2)})"
            )
        lines.append(
            f"  Forward check: section carries W18 = {_fmt(self.w18_capacity, 0)} "
            f"-> {'ADEQUATE' if self.adequate else 'NOT ADEQUATE'}"
        )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


@dataclass
class RigidPavementResult:
    """AASHTO 1993 rigid pavement design (Part II, Ch 3, Figure 3.7)."""

    mode: str  # 'design' or 'check'
    w18: float
    reliability_pct: float
    zr: float
    so: float
    po: float
    pt: float
    delta_psi: float
    sc_psi: float
    ec_psi: float
    j: float
    cd: float
    k_pci: float
    k_basis: dict               # how k was obtained (direct/simple/composite + trail)
    d_required_in: float        # unrounded solve
    d_provided_in: float        # rounded-up (design) or supplied (check)
    w18_capacity: float
    adequate: bool
    environmental: dict = None  # swelling/frost dPSI block (None if not used)
    iterations: int = 0         # composite-k <-> D iterations (0 if k not D-dependent)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "w18": self.w18,
            "reliability_pct": self.reliability_pct,
            "zr": self.zr,
            "so": self.so,
            "po": self.po,
            "pt": self.pt,
            "delta_psi": self.delta_psi,
            "sc_psi": self.sc_psi,
            "ec_psi": self.ec_psi,
            "j": self.j,
            "cd": self.cd,
            "k_pci": self.k_pci,
            "k_basis": self.k_basis,
            "d_required_in": self.d_required_in,
            "d_provided_in": self.d_provided_in,
            "w18_capacity": self.w18_capacity,
            "adequate": self.adequate,
            "environmental": self.environmental,
            "iterations": self.iterations,
            "notes": self.notes,
            "warnings": self.warnings,
            "references": self.references,
        }

    def summary(self) -> str:
        lines = [
            f"AASHTO 1993 rigid pavement {self.mode}",
            f"  W18 = {_fmt(self.w18, 0)} ESALs | R = {self.reliability_pct}% "
            f"(ZR = {self.zr}) | So = {self.so}",
            f"  dPSI = {self.delta_psi} (po {self.po} -> pt {self.pt})",
            f"  Sc' = {_fmt(self.sc_psi, 0)} psi | Ec = {_fmt(self.ec_psi, 0)} psi "
            f"| J = {self.j} | Cd = {self.cd}",
            f"  k = {_fmt(self.k_pci, 1)} pci ({self.k_basis.get('basis', '?')})",
            f"  D required = {_fmt(self.d_required_in)} in | D provided = "
            f"{_fmt(self.d_provided_in)} in",
            f"  Forward check: slab carries W18 = {_fmt(self.w18_capacity, 0)} "
            f"-> {'ADEQUATE' if self.adequate else 'NOT ADEQUATE'}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)
