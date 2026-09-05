"""
CGPR #56 downdrag method family.

Implements the single-pile drag load / downdrag methods and the pile-group
methods cataloged in "Downdrag and Drag Load on Piles" by Greenfield & Filz,
Virginia Tech Center for Geotechnical Practice and Research report CGPR #56
(February 2009):

- Endo method (CGPR #56 Section 3.2.1; Endo et al. 1969): assumed neutral
  plane at a fraction of the embedment depth, drag load by integration of the
  skin friction above it.
- Poulos hand approximation (CGPR #56 Section 3.2.2; Poulos 1997a):
  two-layer (consolidating layer over bearing layer) closed-form neutral
  plane and settlement estimate.
- Fellenius method (CGPR #56 Section 3.2.3; Hannigan et al. 1997): neutral
  plane at the intersection of the load and resistance curves with fully
  mobilized skin and toe resistance; settlement from the free-field profile
  plus a 2:1 equivalent footing at the neutral plane.
- PILENEG procedure (CGPR #56 Section 3.2.4; Briaud & Tucker 1997): partially
  mobilized toe resistance via an elastic pile movement envelope intersected
  with the free-field settlement profile. This implements the calculation
  procedure as documented in CGPR #56 — not the original MS-DOS program.
- Rigid block method for pile groups (CGPR #56 Section 4.3.1; Terzaghi &
  Peck 1967, Broms 1976).
- Drag load reduction coefficient for pile groups (CGPR #56 Section 4.3.2;
  Jeong & Briaud 1994).

A convenience runner ``downdrag_method_comparison`` executes every
single-pile method the supplied inputs allow and tabulates the results.

All equations are dimensionally consistent: use any one consistent unit set
(SI — m, kN, kPa — is the package convention; the module's validation tests
use the report's kips/ft/psf units directly).

Skin friction and settlement inputs are piecewise-linear profiles given as
``[(depth, value), ...]`` point lists; a repeated depth encodes a jump
(e.g. the skin-friction discontinuity at a layer boundary).

Validation
----------
The detailed worked example of CGPR #56 Section 3.4 (a 100 ft, 16 in square
concrete pile through fill and OC clay into dense sand, group load
2000 kips / 9 piles, water table drop 5 ft -> 15 ft) is reproduced in
``downdrag/tests/test_cgpr56.py``. Achieved vs published (Table 3.7):

===========  =============  =========  ================================
Method       Neutral plane  Max force  Pile settlement
===========  =============  =========  ================================
Endo         75.0 / 75.0    522 / 522  (not computed by method)
Poulos       73.6 / 73.6    542 / 542  0.0454 / 0.046 ft
Fellenius    78.4 / 78.4    541 / 541  0.118  / 0.118 ft
PILENEG      62.4 / 62.4    454 / 453  0.049  / 0.049 ft
===========  =============  =========  ================================

(computed / published; PILENEG values with the report's 10-ft trial-depth
grid — the default dense grid moves the neutral plane to 62.6 ft because
the report linearly interpolates its coarse pile-movement envelope.)

References
----------
- Greenfield, M.L. and Filz, G.M. (2009). "Downdrag and Drag Load on
  Piles." Virginia Tech CGPR #56.
- Endo, M. et al. (1969), Poulos, H.G. (1997a), Hannigan et al. (1997),
  Briaud, J.-L. and Tucker, L. (1997), Broms (1976), Jeong & Briaud (1994)
  as cited therein.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Piecewise-linear profile helpers
# ---------------------------------------------------------------------------

def _check_profile(points, name):
    """Validate and normalize a [(depth, value), ...] profile."""
    if not points or len(points) < 1:
        raise ValueError(f"{name}: at least one (depth, value) point required")
    pts = [(float(z), float(v)) for z, v in points]
    for i in range(1, len(pts)):
        if pts[i][0] < pts[i - 1][0]:
            raise ValueError(
                f"{name}: depths must be non-decreasing "
                f"(got {pts[i - 1][0]} then {pts[i][0]})"
            )
    return pts


def profile_value(points, z):
    """Evaluate a piecewise-linear profile at depth z.

    A repeated depth encodes a jump; at the jump depth itself the value
    approached from below (the deeper side) is returned. Beyond the profile
    ends the end values extend as constants.
    """
    pts = _check_profile(points, "profile")
    if z <= pts[0][0]:
        return pts[0][1]
    if z >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        z0, v0 = pts[i]
        z1, v1 = pts[i + 1]
        if z0 <= z < z1:
            if z == z0 and i + 1 < len(pts) and pts[i + 1][0] == z0:
                continue  # take the deeper side of a jump
            if z1 == z0:
                continue
            return v0 + (v1 - v0) * (z - z0) / (z1 - z0)
    return pts[-1][1]


def profile_integral(points, a, b):
    """Integrate a piecewise-linear profile from depth a to depth b (a <= b).

    Exact (trapezoidal over the linear segments). End values extend as
    constants beyond the profile's depth range.
    """
    if b < a:
        raise ValueError(f"profile_integral: b ({b}) < a ({a})")
    if b == a:
        return 0.0
    pts = _check_profile(points, "profile")
    total = 0.0
    # Constant extension before the first point
    if a < pts[0][0]:
        z_end = min(b, pts[0][0])
        total += pts[0][1] * (z_end - a)
    # Constant extension past the last point
    if b > pts[-1][0]:
        z_start = max(a, pts[-1][0])
        total += pts[-1][1] * (b - z_start)
    for i in range(len(pts) - 1):
        z0, v0 = pts[i]
        z1, v1 = pts[i + 1]
        if z1 <= z0:
            continue  # jump (zero width)
        lo = max(a, z0)
        hi = min(b, z1)
        if hi <= lo:
            continue
        vlo = v0 + (v1 - v0) * (lo - z0) / (z1 - z0)
        vhi = v0 + (v1 - v0) * (hi - z0) / (z1 - z0)
        total += 0.5 * (vlo + vhi) * (hi - lo)
    return total


# ---------------------------------------------------------------------------
# Free-field / equivalent-footing consolidation settlement profile
# ---------------------------------------------------------------------------

def consolidation_settlement_profile(
    layers,
    p0_profile,
    dp_profile,
    sublayer_thickness=None,
    eq_footing=None,
    split_depths=(),
):
    """Compute a settlement-vs-depth profile from consolidation parameters.

    Mirrors the sublayer tabulation used in CGPR #56 Section 3.4 (Tables 3.3
    and 3.5): strain from log-linear compression at each sublayer midpoint,
    settlement accumulated upward from the bottom of the deepest settling
    layer, defined at sublayer tops.

    Parameters
    ----------
    layers : list of dict
        Settling layers, each with keys ``z_top``, ``z_bot``, ``C_er``
        (modified recompression index Cr/(1+e0)), and optionally ``C_ec``
        (modified compression index) and ``p_c`` (preconsolidation
        pressure). With only ``C_er`` the layer is treated as remaining
        overconsolidated for the whole stress increment (the CGPR #56
        example case); with ``C_ec`` and ``p_c`` the standard OC/NC cases
        apply; with ``C_ec`` and no ``p_c`` the layer is normally
        consolidated.
    p0_profile : list of (depth, stress)
        Initial vertical effective stress profile.
    dp_profile : float or list of (depth, stress)
        Background stress change causing settlement (fill, water table
        change, ...). A float applies uniformly.
    sublayer_thickness : float, optional
        Sublayer discretization within each layer. Default: layer
        thickness / 10.
    eq_footing : dict, optional
        Equivalent footing adding a 2:1 load-spread stress increment below
        depth ``depth``: keys ``load``, ``width``, ``depth`` and optionally
        ``length`` (default = width). delta_p = load /
        ((width + (z - depth)) * (length + (z - depth))) for z > depth.
    split_depths : iterable of float
        Extra sublayer boundaries to insert (e.g. the neutral plane).

    Returns
    -------
    list of (depth, settlement)
        Piecewise-linear settlement profile from the surface (depth 0,
        full settlement — soil above the settling zone rides down with it)
        to the bottom of the deepest settling layer (settlement 0).
    """
    p0_pts = _check_profile(p0_profile, "p0_profile")
    if isinstance(dp_profile, (int, float)):
        dp_pts = [(0.0, float(dp_profile))]
    else:
        dp_pts = _check_profile(dp_profile, "dp_profile")

    eq = None
    if eq_footing is not None:
        eq = {
            "load": float(eq_footing["load"]),
            "width": float(eq_footing["width"]),
            "length": float(eq_footing.get("length") or eq_footing["width"]),
            "depth": float(eq_footing["depth"]),
        }

    entries = []  # (z_top, z_bot, dH)
    for lay in layers:
        z_top, z_bot = float(lay["z_top"]), float(lay["z_bot"])
        if z_bot <= z_top:
            raise ValueError(f"layer z_bot ({z_bot}) must exceed z_top ({z_top})")
        C_er = float(lay["C_er"])
        C_ec = lay.get("C_ec")
        p_c = lay.get("p_c")
        dz = sublayer_thickness or (z_bot - z_top) / 10.0

        # Sublayer boundaries: regular grid plus any requested splits
        bounds = []
        z = z_top
        while z < z_bot - 1e-9:
            bounds.append(z)
            z += dz
        bounds.append(z_bot)
        for sd in list(split_depths) + ([eq["depth"]] if eq else []):
            if z_top < sd < z_bot and all(abs(sd - b) > 1e-9 for b in bounds):
                bounds.append(sd)
        bounds = sorted(bounds)

        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            mid = 0.5 * (lo + hi)
            p0 = profile_value(p0_pts, mid)
            dp = profile_value(dp_pts, mid)
            if eq and mid > eq["depth"]:
                dz_f = mid - eq["depth"]
                denom = (eq["width"] + dz_f) * (eq["length"] + dz_f)
                dp += eq["load"] / denom
            pf = p0 + dp
            strain = _log_strain(C_er, C_ec, p_c, p0, pf)
            entries.append((lo, hi, strain * (hi - lo)))

    if not entries:
        return [(0.0, 0.0)]

    entries.sort(key=lambda e: e[0])
    # Accumulate settlement upward from the bottom
    profile = []
    s = 0.0
    profile.append((entries[-1][1], 0.0))
    for lo, hi, dH in reversed(entries):
        s += dH
        profile.append((lo, s))
    profile.reverse()
    if profile[0][0] > 0.0:
        profile.insert(0, (0.0, profile[0][1]))
    return profile


def _log_strain(C_er, C_ec, p_c, p0, pf):
    """Vertical strain from log-linear (re)compression indices."""
    if pf <= p0 or p0 <= 0:
        return 0.0
    if C_ec is None:
        return C_er * math.log10(pf / p0)
    C_ec = float(C_ec)
    if p_c is None or float(p_c) <= p0:
        return C_ec * math.log10(pf / p0)
    p_c = float(p_c)
    if pf <= p_c:
        return C_er * math.log10(pf / p0)
    return C_er * math.log10(p_c / p0) + C_ec * math.log10(pf / p_c)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EndoResult:
    """Endo method result (CGPR #56 Section 3.2.1)."""
    neutral_plane_depth: float
    drag_load: float
    max_force: float
    Q_static: float
    neutral_plane_basis: str

    def to_dict(self):
        return {
            "method": "endo",
            "neutral_plane_depth": self.neutral_plane_depth,
            "drag_load": self.drag_load,
            "max_force": self.max_force,
            "Q_static": self.Q_static,
            "neutral_plane_basis": self.neutral_plane_basis,
            "notes": "Drag load estimate only; the Endo method does not "
                     "compute pile settlement. CGPR #56 Section 3.2.1.",
        }

    def summary(self):
        return (
            f"Endo method (CGPR #56 3.2.1): neutral plane at "
            f"{self.neutral_plane_depth:.1f} ({self.neutral_plane_basis}), "
            f"drag load {self.drag_load:.1f}, max pile force {self.max_force:.1f}"
        )


@dataclass
class PoulosResult:
    """Poulos hand approximation result (CGPR #56 Section 3.2.2)."""
    neutral_plane_depth: float
    z_max: float
    drag_load: float
    max_force: float
    Q_static: float
    fs_consolidating: float
    fs_bearing: float
    bearing_fully_mobilized: bool
    pile_settlement: Optional[float]
    soil_settlement_at_np: Optional[float]
    elastic_compression: Optional[float]
    s_equivalent: Optional[float]

    def to_dict(self):
        return {
            "method": "poulos",
            "neutral_plane_depth": self.neutral_plane_depth,
            "z_max": self.z_max,
            "drag_load": self.drag_load,
            "max_force": self.max_force,
            "Q_static": self.Q_static,
            "fs_consolidating": self.fs_consolidating,
            "fs_bearing": self.fs_bearing,
            "bearing_fully_mobilized": self.bearing_fully_mobilized,
            "pile_settlement": self.pile_settlement,
            "soil_settlement_at_np": self.soil_settlement_at_np,
            "elastic_compression": self.elastic_compression,
            "s_equivalent": self.s_equivalent,
        }

    def summary(self):
        s = (
            f"Poulos hand approximation (CGPR #56 3.2.2): neutral plane at "
            f"{self.neutral_plane_depth:.1f} (z_max {self.z_max:.1f}), "
            f"max pile force {self.max_force:.1f}"
        )
        if self.pile_settlement is not None:
            s += f", pile head settlement {self.pile_settlement:.4f}"
        return s


@dataclass
class FelleniusCgpr56Result:
    """Fellenius method result per the CGPR #56 formulation (Section 3.2.3)."""
    neutral_plane_depth: float
    drag_load: float
    max_force: float
    Q_static: float
    toe_resistance: float
    pile_in_failure: bool
    pile_settlement: Optional[float]
    soil_settlement_at_np: Optional[float]
    elastic_compression: Optional[float]
    surface_settlement: Optional[float]
    includes_pile_load_transfer: bool
    depths: List[float] = field(default_factory=list, repr=False)
    load_curve: List[float] = field(default_factory=list, repr=False)
    resistance_curve: List[float] = field(default_factory=list, repr=False)

    def to_dict(self):
        return {
            "method": "fellenius_cgpr56",
            "neutral_plane_depth": self.neutral_plane_depth,
            "drag_load": self.drag_load,
            "max_force": self.max_force,
            "Q_static": self.Q_static,
            "toe_resistance": self.toe_resistance,
            "pile_in_failure": self.pile_in_failure,
            "pile_settlement": self.pile_settlement,
            "soil_settlement_at_np": self.soil_settlement_at_np,
            "elastic_compression": self.elastic_compression,
            "surface_settlement": self.surface_settlement,
            "includes_pile_load_transfer": self.includes_pile_load_transfer,
        }

    def summary(self):
        s = (
            f"Fellenius method, CGPR #56 formulation (3.2.3): neutral plane "
            f"at {self.neutral_plane_depth:.1f}, max pile force "
            f"{self.max_force:.1f} (drag load {self.drag_load:.1f})"
        )
        if self.pile_settlement is not None:
            s += f", pile head settlement {self.pile_settlement:.4f}"
        if self.pile_in_failure:
            s += " — WARNING: load exceeds resistance everywhere (failure)"
        return s


@dataclass
class PilenegResult:
    """PILENEG procedure result as documented in CGPR #56 Section 3.2.4."""
    neutral_plane_depth: float
    drag_load: float
    max_force: float
    Q_static: float
    toe_load: float
    toe_capacity: float
    toe_fully_mobilized: bool
    pile_settlement: Optional[float]
    soil_settlement_at_np: Optional[float]
    elastic_compression: Optional[float]
    trial_depths: List[float] = field(default_factory=list, repr=False)
    envelope: List[float] = field(default_factory=list, repr=False)
    trial_toe_loads: List[float] = field(default_factory=list, repr=False)

    def to_dict(self):
        d = {
            "method": "pileneg",
            "neutral_plane_depth": self.neutral_plane_depth,
            "drag_load": self.drag_load,
            "max_force": self.max_force,
            "Q_static": self.Q_static,
            "toe_load": self.toe_load,
            "toe_capacity": self.toe_capacity,
            "toe_fully_mobilized": self.toe_fully_mobilized,
            "pile_settlement": self.pile_settlement,
            "soil_settlement_at_np": self.soil_settlement_at_np,
            "elastic_compression": self.elastic_compression,
        }
        if self.toe_fully_mobilized:
            d["notes"] = (
                "Elastic bearing assumption invalid (toe load reached the "
                "bearing capacity); CGPR #56 3.2.4 recommends the Fellenius "
                "method in this case."
            )
        return d

    def summary(self):
        s = (
            f"PILENEG procedure (CGPR #56 3.2.4): neutral plane at "
            f"{self.neutral_plane_depth:.1f}, max pile force "
            f"{self.max_force:.1f}, toe load {self.toe_load:.1f} of "
            f"{self.toe_capacity:.1f} capacity"
        )
        if self.pile_settlement is not None:
            s += f", pile head settlement {self.pile_settlement:.4f}"
        if self.toe_fully_mobilized:
            s += " — toe fully mobilized: use the Fellenius method instead"
        return s


@dataclass
class RigidBlockResult:
    """Rigid block pile-group result (CGPR #56 Section 4.3.1)."""
    perimeter_pile_max_force: float
    interior_pile_max_force: float
    static_per_pile: float

    def to_dict(self):
        return {
            "method": "rigid_block_group",
            "perimeter_pile_max_force": self.perimeter_pile_max_force,
            "interior_pile_max_force": self.interior_pile_max_force,
            "static_per_pile": self.static_per_pile,
        }

    def summary(self):
        return (
            f"Rigid block method (CGPR #56 4.3.1): perimeter pile max force "
            f"{self.perimeter_pile_max_force:.1f}, interior pile "
            f"{self.interior_pile_max_force:.1f} (static per pile "
            f"{self.static_per_pile:.1f})"
        )


@dataclass
class GroupReductionResult:
    """Drag load reduction coefficient result (CGPR #56 Section 4.3.2)."""
    reduction_factor: float
    location: str
    s_over_d: float
    max_force_group_pile: float
    max_force_single: float
    static_per_pile: float

    def to_dict(self):
        return {
            "method": "drag_load_reduction_group",
            "reduction_factor": self.reduction_factor,
            "location": self.location,
            "s_over_d": self.s_over_d,
            "max_force_group_pile": self.max_force_group_pile,
            "max_force_single": self.max_force_single,
            "static_per_pile": self.static_per_pile,
        }

    def summary(self):
        return (
            f"Drag load reduction (CGPR #56 4.3.2, Jeong & Briaud 1994): "
            f"A = {self.reduction_factor:.2f} ({self.location} pile, "
            f"s/d = {self.s_over_d:.2f}), group pile max force "
            f"{self.max_force_group_pile:.1f} vs single-pile "
            f"{self.max_force_single:.1f}"
        )


@dataclass
class MethodComparisonResult:
    """Multi-method single-pile downdrag comparison (CGPR #56 Section 3.4.6)."""
    rows: List[dict]
    results: dict = field(repr=False, default_factory=dict)

    def to_dict(self):
        return {
            "comparison_table": self.rows,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

    def summary(self):
        lines = ["Downdrag method comparison (CGPR #56 methods):",
                 f"{'Method':<18}{'Neutral plane':>14}{'Max force':>12}"
                 f"{'Pile settlement':>17}"]
        for r in self.rows:
            if r.get("error"):
                lines.append(f"{r['method']:<18}  skipped: {r['error']}")
                continue
            zn = f"{r['neutral_plane_depth']:.1f}"
            fm = f"{r['max_force']:.0f}"
            st = ("-" if r.get("pile_settlement") is None
                  else f"{r['pile_settlement']:.4f}")
            lines.append(f"{r['method']:<18}{zn:>14}{fm:>12}{st:>17}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endo method — CGPR #56 Section 3.2.1
# ---------------------------------------------------------------------------

# Table 3.1 (Little 1994 and Endo et al. 1969): neutral plane depth as a
# fraction of pile embedment for piles driven through consolidating clay.
ENDO_NEUTRAL_PLANE_RATIOS = {
    # Floating pile, almost no toe resistance (bearing on soft/medium clay)
    "floating": 0.67,
    # Bearing on stiff but flexible materials (loose/medium dense sand, silt)
    "stiff_flexible": 0.75,
    # Fully end bearing (stiff sand or rock)
    "end_bearing": 1.0,
}


def endo_method(
    Q_static,
    pile_length,
    pile_perimeter,
    skin_friction_profile,
    bearing_condition=None,
    neutral_plane_ratio=None,
    neutral_plane_depth=None,
):
    """Endo method drag load estimate (CGPR #56 Section 3.2.1, Eqs 3.4/3.5).

    The neutral plane is assumed (Table 3.1) rather than computed; the drag
    load is the integral of the fully mobilized skin friction from the pile
    head to the assumed neutral plane. Estimate/check-level accuracy only;
    no settlement is computed.

    Provide exactly one of ``bearing_condition`` (a key of
    ``ENDO_NEUTRAL_PLANE_RATIOS``: 'floating', 'stiff_flexible',
    'end_bearing'), ``neutral_plane_ratio`` (fraction of embedment), or
    ``neutral_plane_depth``.
    """
    fs = _check_profile(skin_friction_profile, "skin_friction_profile")
    given = [x is not None
             for x in (bearing_condition, neutral_plane_ratio, neutral_plane_depth)]
    if sum(given) != 1:
        raise ValueError(
            "endo_method: provide exactly one of bearing_condition, "
            "neutral_plane_ratio, or neutral_plane_depth"
        )
    if bearing_condition is not None:
        if bearing_condition not in ENDO_NEUTRAL_PLANE_RATIOS:
            raise ValueError(
                f"endo_method: bearing_condition must be one of "
                f"{sorted(ENDO_NEUTRAL_PLANE_RATIOS)} (got '{bearing_condition}')"
            )
        ratio = ENDO_NEUTRAL_PLANE_RATIOS[bearing_condition]
        z_n = ratio * pile_length
        basis = f"Table 3.1 '{bearing_condition}' ratio {ratio}"
    elif neutral_plane_ratio is not None:
        if not 0.0 < neutral_plane_ratio <= 1.0:
            raise ValueError(
                f"endo_method: neutral_plane_ratio must be in (0, 1] "
                f"(got {neutral_plane_ratio})"
            )
        z_n = neutral_plane_ratio * pile_length
        basis = f"assumed ratio {neutral_plane_ratio}"
    else:
        if not 0.0 < neutral_plane_depth <= pile_length:
            raise ValueError(
                f"endo_method: neutral_plane_depth must be in "
                f"(0, pile_length] (got {neutral_plane_depth})"
            )
        z_n = neutral_plane_depth
        basis = "assumed depth"

    F_negative = pile_perimeter * profile_integral(fs, 0.0, z_n)  # Eq 3.4
    return EndoResult(
        neutral_plane_depth=z_n,
        drag_load=F_negative,
        max_force=Q_static + F_negative,  # Eq 3.5
        Q_static=Q_static,
        neutral_plane_basis=basis,
    )


# ---------------------------------------------------------------------------
# Poulos hand approximation — CGPR #56 Section 3.2.2
# ---------------------------------------------------------------------------

def poulos_method(
    Q_static,
    pile_length,
    pile_perimeter,
    pile_area,
    pile_E,
    depth_to_bearing_layer,
    toe_bearing_capacity,
    skin_friction_profile=None,
    toe_area=None,
    fs_consolidating=None,
    fs_bearing=None,
    settlement_profile=None,
    s_equivalent=None,
):
    """Poulos hand approximation (CGPR #56 Section 3.2.2, Eqs 3.6-3.9).

    Two-layer idealization: a consolidating layer (thickness L1 =
    ``depth_to_bearing_layer``) over an incompressible bearing layer the
    pile penetrates by L2 = pile_length - L1. Average skin frictions may be
    supplied directly (``fs_consolidating`` / ``fs_bearing``) or are
    computed as depth-weighted averages of ``skin_friction_profile``.

    The maximum possible neutral plane depth (Eq 3.6) is

        z_max = 1/2 * [L1 + L2*(fs2/fs1) + (qb*Ab - Q_static)/(fs1*P)]

    capped at L1 (Eq 3.7). Max force from Eq 3.8 uses fs1 regardless of
    neutral plane depth. Pile head settlement (Eq 3.9): free-field
    settlement at the neutral plane (requires ``settlement_profile``) plus
    elastic compression when the bearing layer is fully mobilized
    (z_max < L1); otherwise the settlement of an equivalent pile/footing in
    the bearing layer (``s_equivalent``, supplied by the caller from an
    equivalent-footing analysis) plus elastic compression.
    """
    L1 = float(depth_to_bearing_layer)
    L2 = pile_length - L1
    if L2 < 0:
        raise ValueError(
            f"poulos_method: depth_to_bearing_layer ({L1}) exceeds "
            f"pile_length ({pile_length})"
        )
    Ab = toe_area if toe_area is not None else pile_area

    if fs_consolidating is None or fs_bearing is None:
        if skin_friction_profile is None:
            raise ValueError(
                "poulos_method: provide skin_friction_profile or both "
                "fs_consolidating and fs_bearing"
            )
        fs = _check_profile(skin_friction_profile, "skin_friction_profile")
        if fs_consolidating is None:
            fs_consolidating = profile_integral(fs, 0.0, L1) / L1
        if fs_bearing is None:
            fs_bearing = (profile_integral(fs, L1, pile_length) / L2
                          if L2 > 0 else 0.0)
    if fs_consolidating <= 0:
        raise ValueError("poulos_method: fs_consolidating must be positive")

    # Eq 3.6
    z_max = 0.5 * (
        L1
        + L2 * fs_bearing / fs_consolidating
        + (toe_bearing_capacity * Ab - Q_static)
        / (fs_consolidating * pile_perimeter)
    )
    # Eq 3.7
    fully_mobilized = z_max > L1
    z_n = L1 if fully_mobilized else max(z_max, 0.0)

    # Eq 3.8 — uses fs1 regardless of neutral plane depth
    drag_load = fs_consolidating * pile_perimeter * z_n
    max_force = Q_static + drag_load

    # Eq 3.9 settlement
    pile_settlement = None
    s_n = None
    delta_elastic = None
    if pile_area > 0 and pile_E > 0 and z_n > 0:
        Q_avg = 0.5 * (Q_static + max_force)
        delta_elastic = Q_avg * z_n / (pile_area * pile_E)
    if fully_mobilized:
        if s_equivalent is not None and delta_elastic is not None:
            pile_settlement = s_equivalent + delta_elastic
    else:
        if settlement_profile is not None and delta_elastic is not None:
            sp = _check_profile(settlement_profile, "settlement_profile")
            s_n = profile_value(sp, z_n)
            pile_settlement = s_n + delta_elastic

    return PoulosResult(
        neutral_plane_depth=z_n,
        z_max=z_max,
        drag_load=drag_load,
        max_force=max_force,
        Q_static=Q_static,
        fs_consolidating=fs_consolidating,
        fs_bearing=fs_bearing,
        bearing_fully_mobilized=fully_mobilized,
        pile_settlement=pile_settlement,
        soil_settlement_at_np=s_n,
        elastic_compression=delta_elastic,
        s_equivalent=s_equivalent if fully_mobilized else None,
    )


# ---------------------------------------------------------------------------
# Fellenius method (CGPR #56 formulation) — Section 3.2.3
# ---------------------------------------------------------------------------

def fellenius_method_cgpr56(
    Q_static,
    pile_length,
    pile_perimeter,
    pile_area,
    pile_E,
    toe_bearing_capacity,
    skin_friction_profile,
    toe_area=None,
    settlement_profile=None,
    consolidation=None,
    eq_footing_width=None,
    eq_footing_length=None,
    eq_footing_load=None,
    n_curve_points=101,
):
    """Fellenius neutral plane method as formulated in CGPR #56 Section 3.2.3.

    Neutral plane at the intersection of the load curve
    Q(z) = Q_static + P*int_0^z fs dz and the resistance curve
    R(z) = qb*Ab + P*int_z^L fs dz (Eq 3.10), both with fully mobilized
    skin and toe resistance. Note this formulation, following Hannigan et
    al. (1997), does NOT include the pile self-weight in the load curve
    (unlike this package's ``DowndragAnalysis``).

    Settlement (steps 6-8): pile head settlement = soil settlement at the
    neutral plane + elastic compression above it, where the soil settlement
    includes the pile-soil load transfer modeled as a 2:1 equivalent
    footing at the neutral plane. Two input modes:

    - ``consolidation`` dict (keys ``layers``, ``p0_profile``,
      ``dp_profile``, optional ``sublayer_thickness`` — see
      ``consolidation_settlement_profile``) plus ``eq_footing_width``
      (and optional ``eq_footing_length`` / ``eq_footing_load``, defaults
      pile width x width / Q_static): the settlement profile is recomputed
      with the equivalent footing at the neutral plane (report Table 3.5).
      For a pile group analyzed as in the report's example, pass the
      group's equivalent footing width and total group load.
    - ``settlement_profile`` points: free-field only; the load-transfer
      component is then NOT included (flagged in the result).
    """
    fs = _check_profile(skin_friction_profile, "skin_friction_profile")
    Ab = toe_area if toe_area is not None else pile_area
    Qb = toe_bearing_capacity * Ab

    def load_curve(z):
        return Q_static + pile_perimeter * profile_integral(fs, 0.0, z)

    def resistance_curve(z):
        return Qb + pile_perimeter * profile_integral(fs, z, pile_length)

    pile_in_failure = False
    d0 = load_curve(0.0) - resistance_curve(0.0)
    dL = load_curve(pile_length) - resistance_curve(pile_length)
    if d0 >= 0:
        # Load exceeds total resistance: geotechnical failure condition
        # (CGPR #56 3.2.3 step 5)
        z_n = 0.0
        pile_in_failure = True
    elif dL <= 0:
        z_n = pile_length  # curves never cross: neutral plane at the toe
    else:
        lo, hi = 0.0, pile_length
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if load_curve(mid) - resistance_curve(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10 * pile_length:
                break
        z_n = 0.5 * (lo + hi)

    max_force = load_curve(z_n)
    drag_load = max_force - Q_static

    # Settlement
    s_n = None
    surface_settlement = None
    includes_transfer = False
    if consolidation is not None:
        width = eq_footing_width
        if width is None:
            raise ValueError(
                "fellenius_method_cgpr56: eq_footing_width is required when "
                "consolidation parameters are given (pile or group width "
                "for the 2:1 equivalent footing at the neutral plane)"
            )
        eq = {
            "load": eq_footing_load if eq_footing_load is not None else Q_static,
            "width": width,
            "length": eq_footing_length or width,
            "depth": z_n,
        }
        prof = consolidation_settlement_profile(
            layers=consolidation["layers"],
            p0_profile=consolidation["p0_profile"],
            dp_profile=consolidation["dp_profile"],
            sublayer_thickness=consolidation.get("sublayer_thickness"),
            eq_footing=eq,
        )
        s_n = profile_value(prof, z_n)
        surface_settlement = prof[0][1]
        includes_transfer = True
    elif settlement_profile is not None:
        sp = _check_profile(settlement_profile, "settlement_profile")
        s_n = profile_value(sp, z_n)
        surface_settlement = sp[0][1]

    delta_elastic = None
    pile_settlement = None
    if pile_area > 0 and pile_E > 0:
        Q_avg = 0.5 * (Q_static + max_force)
        delta_elastic = Q_avg * z_n / (pile_area * pile_E)
        if s_n is not None:
            pile_settlement = s_n + delta_elastic

    # Sampled curves for plotting / inspection
    depths = [pile_length * i / (n_curve_points - 1)
              for i in range(n_curve_points)]
    return FelleniusCgpr56Result(
        neutral_plane_depth=z_n,
        drag_load=drag_load,
        max_force=max_force,
        Q_static=Q_static,
        toe_resistance=Qb,
        pile_in_failure=pile_in_failure,
        pile_settlement=pile_settlement,
        soil_settlement_at_np=s_n,
        elastic_compression=delta_elastic,
        surface_settlement=surface_settlement,
        includes_pile_load_transfer=includes_transfer,
        depths=depths,
        load_curve=[load_curve(z) for z in depths],
        resistance_curve=[resistance_curve(z) for z in depths],
    )


# ---------------------------------------------------------------------------
# PILENEG procedure — CGPR #56 Section 3.2.4
# ---------------------------------------------------------------------------

def pileneg_procedure(
    Q_static,
    pile_length,
    pile_perimeter,
    pile_area,
    pile_E,
    toe_bearing_capacity,
    pile_width,
    bearing_E,
    skin_friction_profile,
    settlement_profile,
    bearing_nu=0.3,
    toe_area=None,
    trial_depths=None,
    n_grid=401,
):
    """PILENEG calculation procedure as documented in CGPR #56 Section 3.2.4.

    Implements the method of the PILENEG program (Briaud & Tucker 1997) as
    described in the report — partially mobilized, linear-elastic /
    perfectly-plastic toe resistance:

    1. For each trial neutral plane depth z, solve static equilibrium for
       the toe load Q_toe(z) with fully mobilized skin friction (negative
       above z, positive below).
    2. Pile movement envelope (Eqs 3.11-3.13): delta_n = delta_elastic +
       delta_punch, with delta_punch = (pi/4)*(1 - nu^2)*Q_toe*D/(Ab*Es)
       (elastic half-space, center of a circular load; use the pile width
       for D on non-circular piles) and delta_elastic =
       (Q_toe + 0.5*F_positive)*(L - z)/(A*E).
    3. The neutral plane is where the envelope intersects the free-field
       settlement profile.
    4. Validity check: if Q_toe at the neutral plane exceeds qb*Ab, the
       elastic bearing assumption is invalid and the report recommends the
       Fellenius method (flagged in the result).

    ``trial_depths`` reproduces the report's coarse-grid workflow (linear
    interpolation between trial points on both curves); the default is a
    dense ``n_grid``-point envelope, which relocates the crossing slightly
    (the Section 3.4 example moves from 62.4 ft to 62.6 ft).
    """
    fs = _check_profile(skin_friction_profile, "skin_friction_profile")
    sp = _check_profile(settlement_profile, "settlement_profile")
    Ab = toe_area if toe_area is not None else pile_area
    toe_capacity = toe_bearing_capacity * Ab

    def F_top(z):
        return Q_static + pile_perimeter * profile_integral(fs, 0.0, z)

    def F_pos(z):
        return pile_perimeter * profile_integral(fs, z, pile_length)

    def Q_toe(z):
        return F_top(z) - F_pos(z)

    def envelope(z):
        qt = Q_toe(z)
        # Eq 3.13; a toe load <= 0 produces no punching penetration
        punch = 0.0
        if qt > 0:
            punch = (math.pi / 4.0) * (1.0 - bearing_nu**2) * qt * pile_width \
                / (Ab * bearing_E)
        # Eq 3.12
        elastic = (qt + 0.5 * F_pos(z)) * (pile_length - z) \
            / (pile_area * pile_E)
        return elastic + punch

    if trial_depths is not None:
        zs = sorted(float(z) for z in trial_depths)
        if len(zs) < 2:
            raise ValueError("pileneg_procedure: at least 2 trial_depths required")
    else:
        zs = [pile_length * i / (n_grid - 1) for i in range(n_grid)]

    env = [envelope(z) for z in zs]
    soil = [profile_value(sp, z) for z in zs]

    # Intersection: soil settlement exceeds the envelope above the neutral
    # plane; find the first sign change of (envelope - soil).
    z_n = None
    for i in range(len(zs) - 1):
        d0 = env[i] - soil[i]
        d1 = env[i + 1] - soil[i + 1]
        if d0 < 0 <= d1:
            frac = -d0 / (d1 - d0) if d1 != d0 else 0.0
            z_n = zs[i] + frac * (zs[i + 1] - zs[i])
            break
    if z_n is None:
        if env[0] - soil[0] >= 0:
            z_n = zs[0]  # pile moves more than the soil everywhere: no NSF
        else:
            z_n = pile_length  # curves never cross: neutral plane at the toe

    toe_load = Q_toe(z_n)
    fully_mobilized = toe_load > toe_capacity
    max_force = F_top(z_n)
    drag_load = max_force - Q_static
    s_n = profile_value(sp, z_n)

    delta_elastic = None
    pile_settlement = None
    if pile_area > 0 and pile_E > 0:
        Q_avg = 0.5 * (Q_static + max_force)
        delta_elastic = Q_avg * z_n / (pile_area * pile_E)
        pile_settlement = s_n + delta_elastic

    return PilenegResult(
        neutral_plane_depth=z_n,
        drag_load=drag_load,
        max_force=max_force,
        Q_static=Q_static,
        toe_load=toe_load,
        toe_capacity=toe_capacity,
        toe_fully_mobilized=fully_mobilized,
        pile_settlement=pile_settlement,
        soil_settlement_at_np=s_n,
        elastic_compression=delta_elastic,
        trial_depths=list(zs),
        envelope=env,
        trial_toe_loads=[Q_toe(z) for z in zs],
    )


# ---------------------------------------------------------------------------
# Pile groups — CGPR #56 Section 4.3
# ---------------------------------------------------------------------------

def rigid_block_method(
    Q_static_group,
    n_piles,
    spacing,
    neutral_plane_depth,
    cu_average,
    delta_q,
):
    """Rigid block pile-group method (CGPR #56 Section 4.3.1, Eq 4.1).

    Terzaghi & Peck (1967) / Broms (1976): the pile-soil block settles as a
    unit; perimeter piles carry the block-perimeter shear (average undrained
    strength ``cu_average`` over the depth to the neutral plane on a
    tributary width of one pile spacing) and interior piles carry the
    increased surface stress ``delta_q`` (fills, water table change) over
    their tributary area:

        perimeter pile: F_max = Q_static_group/n + s * z_n * cu
        interior pile:  F_max = Q_static_group/n + s^2 * delta_q

    The group neutral plane is assumed at the single-pile neutral plane
    depth. May exceed single-pile drag loads at large spacing or strength —
    cross-check with the drag load reduction coefficient (Section 4.3.2).
    """
    if n_piles < 1:
        raise ValueError(f"rigid_block_method: n_piles must be >= 1 (got {n_piles})")
    static_per_pile = Q_static_group / n_piles
    perimeter = static_per_pile + spacing * neutral_plane_depth * cu_average
    interior = static_per_pile + spacing**2 * delta_q
    return RigidBlockResult(
        perimeter_pile_max_force=perimeter,
        interior_pile_max_force=interior,
        static_per_pile=static_per_pile,
    )


# Table 4.1 (Jeong & Briaud 1994): drag load reduction factor A at pile
# spacings of 2.5 and 5 diameters.
_GROUP_REDUCTION_TABLE = {
    "interior": (0.15, 0.5),
    "side": (0.4, 0.8),
    "corner": (0.5, 0.9),
}


def drag_load_reduction_factor(s_over_d, location):
    """Drag load reduction factor A (CGPR #56 Table 4.1 / Figure 4.2).

    Linear interpolation between s/d = 2.5 and 5; per Figure 4.2 the
    conservative extensions are the 2.5-diameter value for closer spacings
    and A = 1.0 (no reduction) for spacings beyond 5 diameters.
    """
    if location not in _GROUP_REDUCTION_TABLE:
        raise ValueError(
            f"drag_load_reduction_factor: location must be one of "
            f"{sorted(_GROUP_REDUCTION_TABLE)} (got '{location}')"
        )
    if s_over_d <= 0:
        raise ValueError(f"drag_load_reduction_factor: s_over_d must be "
                         f"positive (got {s_over_d})")
    a25, a5 = _GROUP_REDUCTION_TABLE[location]
    if s_over_d <= 2.5:
        return a25
    if s_over_d > 5.0:
        return 1.0
    return a25 + (a5 - a25) * (s_over_d - 2.5) / 2.5


def drag_load_reduction_method(
    F_max_single,
    Q_static_group,
    n_piles,
    location,
    s_over_d=None,
    spacing=None,
    pile_diameter=None,
):
    """Group drag load via reduction coefficient (CGPR #56 Section 4.3.2).

    Applies Jeong & Briaud's (1994) FEM-derived reduction factor A to the
    single-pile drag load (Eq 4.2):

        F_max,group pile = A * (F_max,single - Q_static_group/n)
                           + Q_static_group/n

    ``F_max_single`` comes from any Section 3 single-pile method. Provide
    ``s_over_d`` directly or both ``spacing`` and ``pile_diameter``.
    Applicable to uniform pile groups of roughly 9-25 piles (reduction
    ceases to improve beyond 25 piles).
    """
    if s_over_d is None:
        if spacing is None or pile_diameter is None:
            raise ValueError(
                "drag_load_reduction_method: provide s_over_d or both "
                "spacing and pile_diameter"
            )
        s_over_d = spacing / pile_diameter
    if n_piles < 1:
        raise ValueError(f"drag_load_reduction_method: n_piles must be >= 1 "
                         f"(got {n_piles})")
    A = drag_load_reduction_factor(s_over_d, location)
    static_per_pile = Q_static_group / n_piles
    F_group = A * (F_max_single - static_per_pile) + static_per_pile
    return GroupReductionResult(
        reduction_factor=A,
        location=location,
        s_over_d=s_over_d,
        max_force_group_pile=F_group,
        max_force_single=F_max_single,
        static_per_pile=static_per_pile,
    )


# ---------------------------------------------------------------------------
# Convenience runner — all applicable single-pile methods on one input set
# ---------------------------------------------------------------------------

def downdrag_method_comparison(
    Q_static,
    pile_length,
    pile_perimeter,
    pile_area,
    pile_E,
    toe_bearing_capacity,
    skin_friction_profile,
    toe_area=None,
    settlement_profile=None,
    consolidation=None,
    endo_bearing_condition=None,
    endo_neutral_plane_ratio=None,
    depth_to_bearing_layer=None,
    poulos_s_equivalent=None,
    pile_width=None,
    bearing_E=None,
    bearing_nu=0.3,
    eq_footing_width=None,
    eq_footing_length=None,
    eq_footing_load=None,
    pileneg_trial_depths=None,
):
    """Run every applicable CGPR #56 single-pile downdrag method and tabulate.

    Always runs the Fellenius method (CGPR #56 formulation). Additionally
    runs, when their extra inputs are present:

    - Endo: needs ``endo_bearing_condition`` (Table 3.1 key) or
      ``endo_neutral_plane_ratio``.
    - Poulos: needs ``depth_to_bearing_layer``.
    - PILENEG: needs ``pile_width``, ``bearing_E``, and a settlement
      profile (``settlement_profile``, or computed free-field from
      ``consolidation``).

    Methods whose inputs are missing appear in the table with a skip
    reason. See the individual method functions for parameter meanings;
    ``consolidation``/``eq_footing_*`` feed the Fellenius settlement step
    and, via the derived free-field profile, Poulos and PILENEG.
    """
    rows = []
    results = {}

    # Free-field settlement profile: given directly, or derived from the
    # consolidation parameters (no equivalent footing).
    free_field = settlement_profile
    if free_field is None and consolidation is not None:
        free_field = consolidation_settlement_profile(
            layers=consolidation["layers"],
            p0_profile=consolidation["p0_profile"],
            dp_profile=consolidation["dp_profile"],
            sublayer_thickness=consolidation.get("sublayer_thickness"),
        )

    def add(name, fn):
        try:
            res = fn()
        except Exception as e:  # surface per-method input problems as rows
            rows.append({"method": name, "error": str(e)})
            return
        results[name] = res
        d = res.to_dict()
        rows.append({
            "method": name,
            "neutral_plane_depth": d["neutral_plane_depth"],
            "max_force": d["max_force"],
            "drag_load": d.get("drag_load"),
            "pile_settlement": d.get("pile_settlement"),
        })

    if endo_bearing_condition is not None or endo_neutral_plane_ratio is not None:
        add("endo", lambda: endo_method(
            Q_static=Q_static, pile_length=pile_length,
            pile_perimeter=pile_perimeter,
            skin_friction_profile=skin_friction_profile,
            bearing_condition=endo_bearing_condition,
            neutral_plane_ratio=endo_neutral_plane_ratio,
        ))
    else:
        rows.append({"method": "endo",
                     "error": "endo_bearing_condition (or "
                              "endo_neutral_plane_ratio) not provided"})

    if depth_to_bearing_layer is not None:
        add("poulos", lambda: poulos_method(
            Q_static=Q_static, pile_length=pile_length,
            pile_perimeter=pile_perimeter, pile_area=pile_area,
            pile_E=pile_E, depth_to_bearing_layer=depth_to_bearing_layer,
            toe_bearing_capacity=toe_bearing_capacity,
            skin_friction_profile=skin_friction_profile, toe_area=toe_area,
            settlement_profile=free_field, s_equivalent=poulos_s_equivalent,
        ))
    else:
        rows.append({"method": "poulos",
                     "error": "depth_to_bearing_layer not provided"})

    add("fellenius_cgpr56", lambda: fellenius_method_cgpr56(
        Q_static=Q_static, pile_length=pile_length,
        pile_perimeter=pile_perimeter, pile_area=pile_area, pile_E=pile_E,
        toe_bearing_capacity=toe_bearing_capacity,
        skin_friction_profile=skin_friction_profile, toe_area=toe_area,
        settlement_profile=settlement_profile, consolidation=consolidation,
        eq_footing_width=eq_footing_width, eq_footing_length=eq_footing_length,
        eq_footing_load=eq_footing_load,
    ))

    if pile_width is not None and bearing_E is not None and free_field is not None:
        add("pileneg", lambda: pileneg_procedure(
            Q_static=Q_static, pile_length=pile_length,
            pile_perimeter=pile_perimeter, pile_area=pile_area,
            pile_E=pile_E, toe_bearing_capacity=toe_bearing_capacity,
            pile_width=pile_width, bearing_E=bearing_E,
            skin_friction_profile=skin_friction_profile,
            settlement_profile=free_field, bearing_nu=bearing_nu,
            toe_area=toe_area, trial_depths=pileneg_trial_depths,
        ))
    else:
        rows.append({"method": "pileneg",
                     "error": "pile_width, bearing_E and a settlement "
                              "profile are required"})

    return MethodComparisonResult(rows=rows, results=results)
