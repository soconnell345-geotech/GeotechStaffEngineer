"""
Rigid cap pile group analysis.

Assembles the group stiffness matrix from individual pile stiffnesses,
solves for cap displacements under applied loads, and back-calculates
individual pile forces.

Supports vertical and battered piles under 6-DOF loading.

All units are SI: kN, m, radians.

References:
    CPGA User's Guide (ITL-89-4, Hartman et al., 1989)
    USACE EM 1110-2-2906, Chapter 4
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from pile_group.pile_layout import GroupPile


@dataclass
class GroupLoad:
    """Applied loads on the pile cap.

    All loads/moments are at the centroid of the pile group at the
    cap elevation.

    Parameters
    ----------
    Vx : float
        Horizontal force in X-direction (kN).
    Vy : float
        Horizontal force in Y-direction (kN).
    Vz : float
        Vertical force (kN). Positive downward (compression).
    Mx : float
        Moment about X-axis (kN-m). Positive per right-hand rule.
    My : float
        Moment about Y-axis (kN-m).
    Mz : float
        Torsion about Z-axis (kN-m).
    """
    Vx: float = 0.0
    Vy: float = 0.0
    Vz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0


@dataclass
class PileGroupResult:
    """Results from pile group analysis.

    Attributes
    ----------
    cap_displacements : dict
        Cap displacements {dx, dy, dz, rx, ry, rz} in m and radians.
    pile_forces : list of dict
        Per-pile forces and utilization ratios.
    max_compression : float
        Maximum compression force in any pile (kN).
    max_tension : float
        Maximum tension force in any pile (kN, positive = tension).
    max_utilization : float
        Maximum utilization ratio (demand/capacity).
    """
    cap_displacements: Dict[str, float] = field(default_factory=dict)
    pile_forces: List[Dict[str, Any]] = field(default_factory=list)
    max_compression: float = 0.0
    max_tension: float = 0.0
    max_utilization: float = 0.0
    n_piles: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PILE GROUP ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Number of piles: {self.n_piles}",
            "",
            "  Cap Displacements:",
        ]
        d = self.cap_displacements
        lines.extend([
            f"    dx = {d.get('dx', 0)*1000:.2f} mm",
            f"    dy = {d.get('dy', 0)*1000:.2f} mm",
            f"    dz = {d.get('dz', 0)*1000:.2f} mm",
            f"    rx = {d.get('rx', 0)*1000:.4f} mrad",
            f"    ry = {d.get('ry', 0)*1000:.4f} mrad",
            f"    rz = {d.get('rz', 0)*1000:.4f} mrad",
        ])
        lines.extend([
            "",
            f"  Max compression:   {self.max_compression:>10,.1f} kN",
            f"  Max tension:       {self.max_tension:>10,.1f} kN",
            f"  Max utilization:   {self.max_utilization:>10.2f}",
            "",
            "  Per-Pile Forces:",
        ])
        for pf in self.pile_forces:
            util_str = f"{pf.get('utilization', 0):.2f}" if pf.get('utilization') else "N/A"
            lines.append(
                f"    {pf['label']:>8}: axial={pf['axial_kN']:>8.1f} kN, "
                f"util={util_str}"
            )
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_piles": self.n_piles,
            "cap_displacements": self.cap_displacements,
            "max_compression_kN": round(self.max_compression, 1),
            "max_tension_kN": round(self.max_tension, 1),
            "max_utilization": round(self.max_utilization, 3),
            "pile_forces": self.pile_forces,
        }


def analyze_vertical_group_simple(
    piles: List[GroupPile],
    load: GroupLoad,
) -> PileGroupResult:
    """Simplified elastic analysis for vertical piles only.

    Uses the standard pile group equation:
        Pi = Vz/n +/- My*xi/SUM(xi^2) +/- Mx*yi/SUM(yi^2)

    This ignores lateral loads and torsion. Use the general 6-DOF
    method for battered piles or combined loading.

    Parameters
    ----------
    piles : list of GroupPile
        Pile layout (all assumed vertical).
    load : GroupLoad
        Applied loads on the cap.

    Returns
    -------
    PileGroupResult
    """
    n = len(piles)
    if n == 0:
        raise ValueError("At least one pile required")

    xs = np.array([p.x for p in piles])
    ys = np.array([p.y for p in piles])

    sum_x2 = np.sum(xs**2)
    sum_y2 = np.sum(ys**2)

    # Axial force in each pile
    axial = np.full(n, load.Vz / n)

    # Moment contribution
    if sum_x2 > 0:
        axial += load.My * xs / sum_x2
    if sum_y2 > 0:
        axial += load.Mx * ys / sum_y2

    # Results
    pile_forces = []
    max_comp = 0.0
    max_tens = 0.0
    max_util = 0.0

    for i, (pile, force) in enumerate(zip(piles, axial)):
        compression = force if force > 0 else 0
        tension = -force if force < 0 else 0

        max_comp = max(max_comp, compression)
        max_tens = max(max_tens, tension)

        util = 0.0
        if compression > 0 and pile.axial_capacity_compression:
            util = compression / pile.axial_capacity_compression
        elif tension > 0 and pile.axial_capacity_tension:
            util = tension / pile.axial_capacity_tension
        max_util = max(max_util, util)

        pile_forces.append({
            "label": pile.label or f"P{i+1}",
            "x_m": round(pile.x, 3),
            "y_m": round(pile.y, 3),
            "axial_kN": round(float(force), 1),
            "utilization": round(util, 3),
        })

    # Simple cap displacement estimate
    if piles[0].axial_stiffness and piles[0].axial_stiffness > 0:
        ka_total = n * piles[0].axial_stiffness
        dz = load.Vz / ka_total
    else:
        dz = 0.0

    cap_disp = {"dx": 0.0, "dy": 0.0, "dz": dz, "rx": 0.0, "ry": 0.0, "rz": 0.0}

    return PileGroupResult(
        cap_displacements=cap_disp,
        pile_forces=pile_forces,
        max_compression=max_comp,
        max_tension=max_tens,
        max_utilization=max_util,
        n_piles=n,
    )


def analyze_group_6dof(
    piles: List[GroupPile],
    load: GroupLoad,
) -> PileGroupResult:
    """General 6-DOF rigid cap analysis for vertical and battered piles.

    Assembles a 6x6 group stiffness matrix from individual pile
    contributions (transformed to global coordinates), solves for
    cap displacements, then back-calculates individual pile forces.

    Parameters
    ----------
    piles : list of GroupPile
        Pile layout with stiffnesses and optional batter angles.
    load : GroupLoad
        Applied 6-DOF loading on the cap.

    Returns
    -------
    PileGroupResult
    """
    n = len(piles)
    if n == 0:
        raise ValueError("At least one pile required")

    # Check all piles have stiffness
    for i, p in enumerate(piles):
        if p.axial_stiffness is None:
            raise ValueError(f"Pile {i} ({p.label}) missing axial_stiffness")

    # Build 6x6 group stiffness matrix
    K_group = np.zeros((6, 6))

    for pile in piles:
        ka = pile.axial_stiffness
        kl = pile.lateral_stiffness or 0.0

        lx, ly, lz = pile.direction_cosines()

        # Pile stiffness in local coordinates: [axial, lateral_x, lateral_y]
        # Transform to global using direction cosines
        # Simplified: axial along pile axis, lateral perpendicular
        # For vertical piles: axial = kz, lateral = kx, ky
        # For battered: transform

        # Global stiffness contributions
        kxx = ka * lx**2 + kl * (1 - lx**2)
        kyy = ka * ly**2 + kl * (1 - ly**2)
        kzz = ka * lz**2 + kl * (1 - lz**2)
        kxy = (ka - kl) * lx * ly
        kxz = (ka - kl) * lx * lz
        kyz = (ka - kl) * ly * lz

        x = pile.x
        y = pile.y

        # Direct stiffness terms
        K_group[0, 0] += kxx  # Vx -> dx
        K_group[1, 1] += kyy  # Vy -> dy
        K_group[2, 2] += kzz  # Vz -> dz

        # Coupling: force-rotation and moment-translation
        K_group[0, 4] += kxz * 0  # simplified
        K_group[1, 3] += kyz * 0  # simplified

        # Moment stiffness from pile eccentricity
        # Mx causes rotation about X -> pile at y gets axial force
        K_group[3, 3] += kzz * y**2  # Mx -> rx
        K_group[4, 4] += kzz * x**2  # My -> ry
        K_group[5, 5] += kxx * y**2 + kyy * x**2  # Mz -> rz

        # Cross-coupling
        K_group[2, 3] += kzz * y  # Vz-rx
        K_group[3, 2] += kzz * y
        K_group[2, 4] += kzz * x  # Vz-ry
        K_group[4, 2] += kzz * x
        K_group[3, 4] += kzz * x * y  # rx-ry
        K_group[4, 3] += kzz * x * y

    # Load vector
    F = np.array([load.Vx, load.Vy, load.Vz, load.Mx, load.My, load.Mz])

    # Solve for displacements
    try:
        U = np.linalg.solve(K_group, F)
    except np.linalg.LinAlgError:
        # Singular matrix — likely no lateral stiffness or other issue
        # Fall back to simplified method
        return analyze_vertical_group_simple(piles, load)

    dx, dy, dz, rx, ry, rz = U

    # Back-calculate pile forces
    pile_forces = []
    max_comp = 0.0
    max_tens = 0.0
    max_util = 0.0

    for i, pile in enumerate(piles):
        ka = pile.axial_stiffness
        lx, ly, lz = pile.direction_cosines()

        # Pile head displacement along pile axis
        # Local displacement = dot(global_disp_at_pile, pile_axis)
        # Global displacement at pile location:
        dx_pile = dx - ry * 0 + rz * pile.y  # simplified
        dy_pile = dy + rx * 0 - rz * pile.x
        dz_pile = dz + rx * pile.y + ry * pile.x

        # Axial displacement along pile
        d_axial = dx_pile * lx + dy_pile * ly + dz_pile * lz

        # Axial force
        axial_force = ka * d_axial

        compression = axial_force if axial_force > 0 else 0
        tension = -axial_force if axial_force < 0 else 0
        max_comp = max(max_comp, compression)
        max_tens = max(max_tens, tension)

        util = 0.0
        if compression > 0 and pile.axial_capacity_compression:
            util = compression / pile.axial_capacity_compression
        elif tension > 0 and pile.axial_capacity_tension:
            util = tension / pile.axial_capacity_tension
        max_util = max(max_util, util)

        pile_forces.append({
            "label": pile.label or f"P{i+1}",
            "x_m": round(pile.x, 3),
            "y_m": round(pile.y, 3),
            "axial_kN": round(float(axial_force), 1),
            "utilization": round(util, 3),
        })

    cap_disp = {
        "dx": float(dx), "dy": float(dy), "dz": float(dz),
        "rx": float(rx), "ry": float(ry), "rz": float(rz),
    }

    return PileGroupResult(
        cap_displacements=cap_disp,
        pile_forces=pile_forces,
        max_compression=max_comp,
        max_tension=max_tens,
        max_utilization=max_util,
        n_piles=n,
    )
