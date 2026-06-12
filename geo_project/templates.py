"""Parametric Project generators — "describe it in words" starting points.

Each template returns a :class:`~geo_project.schema.Project` whose GEOMETRY
is fully built (provenance='template') and whose stratigraphy carries
named layers with EMPTY materials (gamma defaults to 18 kN/m3, logged in the
assumption ledger; strengths are left unset so validate() flags exactly what
the requested analyses still need). The agent fills materials at the
materials stage with the human, citing cov_lookup / project sources.

Conventions (all SI, x left → right, z = elevation in m):

* The slope DESCENDS left → right (crest on the left, toe on the right).
* ``slope_ratio`` is H:V, e.g. 2.0 means 2H:1V (flatter = larger).
* Margins are flat runs added beyond the crest and toe so slip-circle
  search grids have room (defaults scale with H).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from geo_project.schema import (
    Geometry,
    Layer,
    Material,
    Project,
    ProjectMeta,
)


def _default_margin(H: float) -> float:
    """Default flat crest/toe margin: 1.5x the slope height, min 5 m."""
    return max(1.5 * H, 5.0)


def _make_layers(layer_names: List[str], bottoms: List[float],
                 project: Project) -> None:
    """Append layers with default-gamma materials + ledger entries."""
    for name, bot in zip(layer_names, bottoms):
        project.stratigraphy.append(Layer(
            name=name,
            material=Material(gamma=18.0),
            bottom_elevation=float(bot),
        ))
        project.add_assumption(
            f"stratigraphy[{name}].material.gamma", 18.0,
            source="placeholder default — set real unit weight at the "
                   "materials stage",
        )


def simple_slope(H: float, slope_ratio: float = 2.0,
                 crest_margin: Optional[float] = None,
                 toe_margin: Optional[float] = None,
                 foundation_depth: float = 0.0,
                 name: str = "Simple slope") -> Project:
    """A single planar slope from crest to toe, descending left → right.

    Use for: "a 10 m high 2H:1V slope", cuts/fills with one face.

    Parameters
    ----------
    H : float
        Slope height, crest to toe (m).
    slope_ratio : float
        Horizontal-to-vertical ratio (2.0 = 2H:1V). Must be > 0.
    crest_margin, toe_margin : float, optional
        Flat ground beyond the crest / toe (m). Default max(1.5*H, 5).
    foundation_depth : float
        Thickness of a distinct foundation layer below the toe (m).
        0 (default) = single layer extending H below the toe; > 0 adds a
        second 'Foundation' layer of that thickness.
    name : str
        Project name.

    Returns
    -------
    Project
        Geometry (provenance='template') + named layers with placeholder
        materials; toe elevation is z=0, crest at z=H.
    """
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")
    if slope_ratio <= 0:
        raise ValueError(f"slope_ratio must be positive, got {slope_ratio}")
    cm = _default_margin(H) if crest_margin is None else float(crest_margin)
    tm = _default_margin(H) if toe_margin is None else float(toe_margin)

    x_crest = cm
    x_toe = cm + slope_ratio * H
    surface = [(0.0, H), (x_crest, H), (x_toe, 0.0), (x_toe + tm, 0.0)]

    p = Project(
        meta=ProjectMeta(
            name=name,
            description=(f"Template simple_slope: H={H:g} m, "
                         f"{slope_ratio:g}H:1V, crest margin {cm:g} m, "
                         f"toe margin {tm:g} m, foundation_depth="
                         f"{foundation_depth:g} m")),
        geometry=Geometry(surface_points=surface, provenance="template"),
    )
    if foundation_depth > 0:
        _make_layers(["Slope soil", "Foundation"],
                     [0.0, -float(foundation_depth)], p)
    else:
        _make_layers(["Slope soil"], [-H], p)
        p.add_assumption(
            "stratigraphy[Slope soil].bottom_elevation", -H,
            source="template default: section extends H below the toe")
    return p


def benched_slope(H: float, slope_ratio: float = 2.0, n_benches: int = 1,
                  bench_width: float = 4.0,
                  crest_margin: Optional[float] = None,
                  toe_margin: Optional[float] = None,
                  foundation_depth: float = 0.0,
                  name: str = "Benched slope") -> Project:
    """A slope descending in equal lifts with flat benches between them.

    Use for: "a 15 m cut in three benches with 4 m wide benches".

    Parameters
    ----------
    H : float
        TOTAL height crest to toe (m); split equally across the lifts.
    slope_ratio : float
        H:V ratio of each lift face.
    n_benches : int
        Number of INTERMEDIATE benches (>= 0; 0 degenerates to
        simple_slope). Lifts = n_benches + 1.
    bench_width : float
        Width of each flat bench (m).
    crest_margin, toe_margin, foundation_depth, name
        As in :func:`simple_slope`.
    """
    if n_benches < 0:
        raise ValueError(f"n_benches must be >= 0, got {n_benches}")
    if n_benches == 0:
        return simple_slope(H, slope_ratio, crest_margin, toe_margin,
                            foundation_depth, name=name)
    if H <= 0 or slope_ratio <= 0 or bench_width < 0:
        raise ValueError("H and slope_ratio must be positive, "
                         "bench_width non-negative")
    cm = _default_margin(H) if crest_margin is None else float(crest_margin)
    tm = _default_margin(H) if toe_margin is None else float(toe_margin)

    n_lifts = n_benches + 1
    dz = H / n_lifts
    dx_face = slope_ratio * dz

    pts: List[Tuple[float, float]] = [(0.0, H), (cm, H)]
    x = cm
    z = H
    for lift in range(n_lifts):
        x += dx_face
        z -= dz
        pts.append((x, z))
        if lift < n_benches:
            x += bench_width
            pts.append((x, z))
    pts.append((x + tm, 0.0))

    p = Project(
        meta=ProjectMeta(
            name=name,
            description=(f"Template benched_slope: H={H:g} m total, "
                         f"{n_lifts} lifts at {slope_ratio:g}H:1V, "
                         f"{n_benches} bench(es) {bench_width:g} m wide")),
        geometry=Geometry(surface_points=pts, provenance="template"),
    )
    if foundation_depth > 0:
        _make_layers(["Slope soil", "Foundation"],
                     [0.0, -float(foundation_depth)], p)
    else:
        _make_layers(["Slope soil"], [-H], p)
    return p


def embankment_on_foundation(H: float, crest_width: float = 10.0,
                             slope_ratio: float = 2.0,
                             foundation_depth: float = 10.0,
                             margin: Optional[float] = None,
                             symmetric: bool = False,
                             name: str = "Embankment on foundation"
                             ) -> Project:
    """A fill embankment on level ground over a foundation layer.

    Use for: "a 6 m embankment with a 12 m crest on soft clay".

    The embankment is its own layer ('Embankment fill'); the original
    ground (z=0 down to -foundation_depth) is the 'Foundation' layer —
    the usual soft-ground stability setup.

    Parameters
    ----------
    H : float
        Embankment height above original ground (m). Crest at z=H.
    crest_width : float
        Flat crest width (m).
    slope_ratio : float
        Side-slope H:V ratio.
    foundation_depth : float
        Foundation layer thickness below original ground (m).
    margin : float, optional
        Flat original ground beyond each toe (m). Default max(1.5*H, 5).
    symmetric : bool
        True → model BOTH side slopes (full embankment); False (default) →
        model the left half only from the crest centerline (half-section,
        standard for symmetric problems).
    """
    if H <= 0 or crest_width < 0 or slope_ratio <= 0 or foundation_depth <= 0:
        raise ValueError("H, slope_ratio, foundation_depth must be positive; "
                         "crest_width non-negative")
    mg = _default_margin(H) if margin is None else float(margin)
    dx_face = slope_ratio * H

    if symmetric:
        x = 0.0
        pts = [(x, 0.0)]
        x += mg
        pts.append((x, 0.0))
        x += dx_face
        pts.append((x, H))
        x += crest_width
        pts.append((x, H))
        x += dx_face
        pts.append((x, 0.0))
        x += mg
        pts.append((x, 0.0))
        # left-to-right ascending then descending — keep as is (x increasing)
    else:
        # Half-section: crest centerline on the LEFT, slope descends to toe.
        half_crest = crest_width / 2.0
        pts = [(0.0, H), (half_crest, H), (half_crest + dx_face, 0.0),
               (half_crest + dx_face + mg, 0.0)]

    p = Project(
        meta=ProjectMeta(
            name=name,
            description=(f"Template embankment_on_foundation: H={H:g} m, "
                         f"crest {crest_width:g} m, {slope_ratio:g}H:1V, "
                         f"foundation {foundation_depth:g} m, "
                         f"{'full' if symmetric else 'half'}-section")),
        geometry=Geometry(surface_points=pts, provenance="template"),
    )
    _make_layers(["Embankment fill", "Foundation"],
                 [0.0, -float(foundation_depth)], p)
    return p


def cut_with_berm(H_upper: float, H_lower: float, berm_width: float,
                  slope_ratio_upper: float = 2.0,
                  slope_ratio_lower: float = 2.0,
                  crest_margin: Optional[float] = None,
                  toe_margin: Optional[float] = None,
                  foundation_depth: float = 0.0,
                  name: str = "Cut with berm") -> Project:
    """A two-stage cut with a stabilizing berm bench between the slopes.

    Use for: "an 8 m upper cut, a 6 m berm, then a 4 m lower slope".
    Differs from :func:`benched_slope` by allowing DIFFERENT heights and
    ratios above and below the berm.

    Parameters
    ----------
    H_upper, H_lower : float
        Heights of the upper and lower slope faces (m). Toe at z=0,
        berm at z=H_lower, crest at z=H_lower+H_upper.
    berm_width : float
        Flat berm width (m).
    slope_ratio_upper, slope_ratio_lower : float
        H:V ratios of the two faces.
    crest_margin, toe_margin, foundation_depth, name
        As in :func:`simple_slope`.
    """
    if H_upper <= 0 or H_lower <= 0 or berm_width < 0:
        raise ValueError("H_upper/H_lower must be positive, berm_width "
                         "non-negative")
    if slope_ratio_upper <= 0 or slope_ratio_lower <= 0:
        raise ValueError("slope ratios must be positive")
    H = H_upper + H_lower
    cm = _default_margin(H) if crest_margin is None else float(crest_margin)
    tm = _default_margin(H) if toe_margin is None else float(toe_margin)

    z_crest = H
    z_berm = H_lower
    x = cm
    pts = [(0.0, z_crest), (x, z_crest)]
    x += slope_ratio_upper * H_upper
    pts.append((x, z_berm))
    x += berm_width
    pts.append((x, z_berm))
    x += slope_ratio_lower * H_lower
    pts.append((x, 0.0))
    pts.append((x + tm, 0.0))

    p = Project(
        meta=ProjectMeta(
            name=name,
            description=(f"Template cut_with_berm: upper {H_upper:g} m at "
                         f"{slope_ratio_upper:g}H:1V, berm {berm_width:g} m "
                         f"at z={z_berm:g}, lower {H_lower:g} m at "
                         f"{slope_ratio_lower:g}H:1V")),
        geometry=Geometry(surface_points=pts, provenance="template"),
    )
    if foundation_depth > 0:
        _make_layers(["Slope soil", "Foundation"],
                     [0.0, -float(foundation_depth)], p)
    else:
        _make_layers(["Slope soil"], [-H], p)
    return p


#: Registry the setup agent's ``project_new`` tool dispatches on.
TEMPLATES = {
    "simple_slope": simple_slope,
    "benched_slope": benched_slope,
    "embankment_on_foundation": embankment_on_foundation,
    "cut_with_berm": cut_with_berm,
}


__all__ = ["simple_slope", "benched_slope", "embankment_on_foundation",
           "cut_with_berm", "TEMPLATES"]
