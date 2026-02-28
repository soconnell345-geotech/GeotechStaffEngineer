"""
Soil nail reinforcement for slope stability analysis.

Computes nail-slip-surface intersections, pullout and tensile capacities,
and resisting forces/moments for use in limit equilibrium FOS calculations.

Each nail contributes a resisting force T (kN/m run) that adds to the
FOS numerator (moment methods) or modifies force equilibrium (Spencer).

References:
    Lazarte et al. (2003) — FHWA GEC-7: Soil Nail Walls
    Byrne et al. (1998) — FHWA-SA-96-069R: Soil Nailing Manual
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SoilNail:
    """A single soil nail defined by head position and geometry.

    Parameters
    ----------
    x_head : float
        Nail head x-position at slope face (m).
    z_head : float
        Nail head elevation (m).
    length : float
        Total nail length (m).
    inclination : float
        Degrees below horizontal (positive = downward). Default 15.
    bar_diameter : float
        Nail bar diameter (mm). Default 25 (#8 bar).
    drill_hole_diameter : float
        Drill hole diameter (mm). Default 150.
    fy : float
        Bar yield strength (MPa, Grade 60). Default 420.
    bond_stress : float
        Ultimate bond stress along grout-soil interface (kPa). Default 100.
    spacing_h : float
        Horizontal spacing between nails (m). Default 1.5.
    """
    x_head: float
    z_head: float
    length: float
    inclination: float = 15.0
    bar_diameter: float = 25.0
    drill_hole_diameter: float = 150.0
    fy: float = 420.0
    bond_stress: float = 100.0
    spacing_h: float = 1.5

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"Nail length must be positive, got {self.length}")
        if self.inclination < -90 or self.inclination > 90:
            raise ValueError(
                f"Nail inclination must be between -90 and 90 degrees, "
                f"got {self.inclination}"
            )
        if self.bar_diameter <= 0:
            raise ValueError(
                f"Bar diameter must be positive, got {self.bar_diameter}"
            )
        if self.drill_hole_diameter <= 0:
            raise ValueError(
                f"Drill hole diameter must be positive, "
                f"got {self.drill_hole_diameter}"
            )
        if self.fy <= 0:
            raise ValueError(f"Yield strength must be positive, got {self.fy}")
        if self.bond_stress <= 0:
            raise ValueError(
                f"Bond stress must be positive, got {self.bond_stress}"
            )
        if self.spacing_h <= 0:
            raise ValueError(
                f"Horizontal spacing must be positive, got {self.spacing_h}"
            )

    @property
    def tip_x(self) -> float:
        """X-coordinate of nail tip."""
        beta = math.radians(self.inclination)
        return self.x_head + self.length * math.cos(beta)

    @property
    def tip_z(self) -> float:
        """Z-coordinate of nail tip."""
        beta = math.radians(self.inclination)
        return self.z_head - self.length * math.sin(beta)

    @property
    def tensile_capacity_kN(self) -> float:
        """Tensile capacity of the bar (kN).

        T_tens = fy * pi * (d/2)^2
        fy in MPa = kN/mm^2 * 1000, d in mm -> result in kN.
        """
        area_mm2 = math.pi * (self.bar_diameter / 2.0) ** 2
        return self.fy * area_mm2 / 1000.0  # MPa * mm^2 = N -> /1000 = kN


@dataclass
class NailContribution:
    """Result of computing a single nail's contribution to slope stability.

    Attributes
    ----------
    nail_index : int
        Index of the nail in the input list.
    x_intersect : float
        X-coordinate where nail crosses slip circle (m).
    z_intersect : float
        Z-coordinate where nail crosses slip circle (m).
    length_behind : float
        Nail length behind (beyond) the slip surface (m).
    T_pullout : float
        Pullout capacity per meter of wall run (kN/m).
    T_tensile : float
        Tensile capacity per meter of wall run (kN/m).
    T_design : float
        Design force = min(T_pullout, T_tensile) (kN/m).
    beta_rad : float
        Nail inclination angle (radians, positive = downward).
    moment_arm : float
        Perpendicular distance from circle center to nail force line (m).
    resisting_moment : float
        T_design * moment_arm / R contribution to moment equilibrium.
    force_h : float
        Horizontal force component T * cos(beta) (kN/m).
    force_v : float
        Vertical force component T * sin(beta) (kN/m, positive downward).
    """
    nail_index: int = 0
    x_intersect: float = 0.0
    z_intersect: float = 0.0
    length_behind: float = 0.0
    T_pullout: float = 0.0
    T_tensile: float = 0.0
    T_design: float = 0.0
    beta_rad: float = 0.0
    moment_arm: float = 0.0
    resisting_moment: float = 0.0
    force_h: float = 0.0
    force_v: float = 0.0


def nail_circle_intersection(
    nail: SoilNail, xc: float, yc: float, radius: float
) -> Optional[Tuple[float, float, float]]:
    """Find where a nail intersects a circular slip surface.

    The nail is a line segment from (x_head, z_head) extending at
    inclination beta (positive downward from horizontal).

    Line parametric: P(t) = (x_h + t*cos(beta), z_h - t*sin(beta)) for t in [0, length]
    Circle: (x - xc)^2 + (z - yc)^2 = R^2

    We want the intersection on the lower arc (z < yc) with 0 < t <= length.

    Parameters
    ----------
    nail : SoilNail
        The nail to test.
    xc, yc : float
        Circle center coordinates.
    radius : float
        Circle radius.

    Returns
    -------
    (x_int, z_int, t) or None
        Intersection point and parametric distance, or None if no intersection.
    """
    beta = math.radians(nail.inclination)
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)

    # Direction vector: (cos_b, -sin_b) (x increases, z decreases for positive beta)
    dx0 = nail.x_head - xc
    dz0 = nail.z_head - yc

    # Quadratic: t^2 + 2*B*t + C = 0
    # where a_coeff=1 (unit direction), B = dx0*cos_b - dz0*sin_b, C = dx0^2 + dz0^2 - R^2
    a_coeff = 1.0  # cos_b^2 + sin_b^2
    b_coeff = 2.0 * (dx0 * cos_b - dz0 * sin_b)
    c_coeff = dx0 ** 2 + dz0 ** 2 - radius ** 2

    discriminant = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)
    t2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)

    # We need t > 0 (ahead of nail head) and t <= length
    # Prefer the first intersection the nail encounters (smaller valid t)
    # Also require intersection on lower arc (z_int <= yc)
    best = None
    for t in (t1, t2):
        if t < 1e-6 or t > nail.length + 1e-6:
            continue
        x_int = nail.x_head + t * cos_b
        z_int = nail.z_head - t * sin_b
        # Must be on lower arc (below center)
        if z_int > yc + 1e-6:
            continue
        if best is None or t < best[2]:
            best = (x_int, z_int, t)

    return best


def compute_nail_contribution(
    nail: SoilNail, idx: int, xc: float, yc: float, radius: float
) -> Optional[NailContribution]:
    """Compute a single nail's contribution to the FOS calculation.

    Parameters
    ----------
    nail : SoilNail
        The nail.
    idx : int
        Index of this nail in the nail list.
    xc, yc : float
        Circle center.
    radius : float
        Circle radius.

    Returns
    -------
    NailContribution or None
        None if the nail does not intersect the slip circle.
    """
    intersection = nail_circle_intersection(nail, xc, yc, radius)
    if intersection is None:
        return None

    x_int, z_int, t_int = intersection
    beta = math.radians(nail.inclination)

    # Length behind the slip surface (the anchored portion)
    length_behind = nail.length - t_int
    if length_behind < 0:
        length_behind = 0.0

    # Pullout capacity: T_pull = bond_stress * pi * DDH * L_behind
    # bond_stress in kPa, DDH in mm -> convert to m
    ddh_m = nail.drill_hole_diameter / 1000.0
    T_pullout_total = nail.bond_stress * math.pi * ddh_m * length_behind  # kN per nail

    # Tensile capacity
    T_tensile_total = nail.tensile_capacity_kN  # kN per nail

    # Design force per meter of wall run
    T_pull_per_m = T_pullout_total / nail.spacing_h
    T_tens_per_m = T_tensile_total / nail.spacing_h
    T_design = min(T_pull_per_m, T_tens_per_m)

    # Moment arm: perpendicular distance from circle center to the nail
    # force line at the intersection point.
    # Force direction: (cos_b, -sin_b)
    # Perpendicular distance = cross product of (P_int - Center) x Direction
    # d_perp = (x_int - xc)*(-sin_b) - (z_int - yc)*cos_b
    d_perp = -(x_int - xc) * math.sin(beta) - (z_int - yc) * math.cos(beta)

    # Nail always resists (acts against rotation), use abs
    moment_arm = abs(d_perp)

    # Resisting moment contribution: T * d_perp / R
    resisting_moment = T_design * moment_arm / radius

    # Force components
    force_h = T_design * math.cos(beta)
    force_v = T_design * math.sin(beta)  # positive = downward

    return NailContribution(
        nail_index=idx,
        x_intersect=x_int,
        z_intersect=z_int,
        length_behind=length_behind,
        T_pullout=T_pull_per_m,
        T_tensile=T_tens_per_m,
        T_design=T_design,
        beta_rad=beta,
        moment_arm=moment_arm,
        resisting_moment=resisting_moment,
        force_h=force_h,
        force_v=force_v,
    )


def compute_all_nail_contributions(
    nails: List[SoilNail], xc: float, yc: float, radius: float
) -> List[NailContribution]:
    """Compute contributions for all nails that intersect the slip circle.

    Parameters
    ----------
    nails : list of SoilNail
        All nails defined for the slope.
    xc, yc : float
        Circle center.
    radius : float
        Circle radius.

    Returns
    -------
    list of NailContribution
        Only nails that actually intersect the circle.
    """
    contributions = []
    for i, nail in enumerate(nails):
        contrib = compute_nail_contribution(nail, i, xc, yc, radius)
        if contrib is not None:
            contributions.append(contrib)
    return contributions


def total_nail_resisting(contributions: List[NailContribution]) -> float:
    """Total nail resisting moment equivalent (sum of T*d/R).

    This is added directly to the FOS numerator for moment-based methods
    (Fellenius, Bishop).

    Parameters
    ----------
    contributions : list of NailContribution

    Returns
    -------
    float
        Total resisting force equivalent (kN/m).
    """
    return sum(c.resisting_moment for c in contributions)


def nail_force_components(
    contributions: List[NailContribution],
) -> Tuple[float, float]:
    """Total horizontal and vertical nail force components.

    Used for Spencer's force equilibrium.

    Parameters
    ----------
    contributions : list of NailContribution

    Returns
    -------
    (F_h, F_v) : tuple of float
        Total horizontal force (kN/m) and vertical force (kN/m).
        F_v positive = downward.
    """
    f_h = sum(c.force_h for c in contributions)
    f_v = sum(c.force_v for c in contributions)
    return (f_h, f_v)
