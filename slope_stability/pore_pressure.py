"""
Pore pressure on a slice base, as printed (GM-140 step 5a).

One function, ``base_pore_pressure``, gives the pore pressure every slice
base carries; ``slices.build_slices`` calls it for every method, every
family of surfaces and every caller (a single surface, the searches, the
rapid-drawdown stages, the yield acceleration, FOSM and Monte Carlo).

The rule, by the kind of water input and the kind of strength the base's
layer has:

1. A water line (``SlopeGeometry.gwt_points``), read as the caller says it
   is (``SlopeGeometry.water_line_kind``; owner call OWN-104):

   * ``'piezometric'`` (default): a piezometric line, the height water
     rises to in standpipes (UFC 3-220-10 Sec. 7-5.1, p. 355, method 2).
     u = gamma_w x the vertical depth of the base's mid-point below the
     line. Never corrected.
   * ``'water_table'``: a water table (phreatic surface, the line of
     u = 0) with seepage parallel to it. "When the water table is sloping,
     use Equation 6-16 to calculate the water pressure on the base of the
     slice: u = h_w gamma_w cos^2(alpha_w)", alpha_w the slope of the
     water table from the horizontal and h_w the depth from the ground
     water surface to the base (FHWA NHI-06-088 Eq. 6-16, p. 6-14); "Water
     table or phreatic surface: can be corrected to approximate the
     piezometric surface based on the slope of the surface above the
     specific point" (UFC 3-220-10 Sec. 7-5.1, p. 355, method 1). alpha_w
     is taken on the stretch of the line above the base's mid-point. Where
     the line is above the ground there (ponded water) nothing is
     corrected: the pond is hydrostatic.

   The uncorrected head is the larger pore pressure and so the lower
   factor of safety; the correction holds only for seepage parallel to a
   straight water table, a statement about the seepage that the caller
   makes by naming the line a water table.

2. ``ru`` on the base's layer, used where the line gives no pressure at
   the base (and no pore-pressure grid is given): u = ru x the total
   overburden stress at the base, the weight of the soil column above it
   per unit width, each soil at its own unit weight (the column step 1
   weighs, ``SlopeGeometry.column_at``). EM 1110-2-1902 C-7, p. C-29,
   defines ru = u / (gamma z) for a uniform slope only; the sum over a
   layered column is this module's reading of that definition (no printed
   rule for a layered column was found). A surcharge and ponded water are
   not counted: gamma z is the soil's weight. (Water ponded over the base
   puts the base under the line, where the line governs and ru is not
   used.)

3. A pore-pressure grid (``pore_pressure_points``) governs the base
   pressure wherever it is given, in place of the line and of ru; the line
   then still gives the ponded water's loads and the saturated unit
   weights below it.

4. A layer analysed in total stresses carries no pore pressure on its
   base, whatever the water input: "For total stress analyses, pore water
   pressures are set to zero" (EM 1110-2-1902 Table 2-1, p. 2-2); "the
   pore water pressure should be set to zero" (2-2c, p. 2-4). Total-stress
   layers here: Mohr-Coulomb with
   ``analysis_mode='undrained'`` (cu), and the 'shansep' and 'anisotropic'
   undrained strengths. A Hoek-Brown layer is an effective-stress
   strength. Table 2-1's note 2 (p. 2-2): total-stress envelopes with
   phi > 0 apply only to partially saturated soils; a total-stress layer
   given phi > 0 that lies under water is warned of
   (``water_statements``).

``water_statements`` gives the entries of a result's ``choices`` and the
warnings that go with these rules.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from geotech_common.water import GAMMA_W

#: What ``SlopeGeometry.gwt_points`` may be (``water_line_kind``).
PIEZOMETRIC = "piezometric"
WATER_TABLE = "water_table"
WATER_LINE_KINDS = (PIEZOMETRIC, WATER_TABLE)

# A stretch of the water line flatter than this (rise over run) is level:
# cos^2 of its slope differs from 1 by less than 1e-12.
_LEVEL = 1e-6


@dataclass
class BasePorePressure:
    """The pore pressure at one slice base (``base_pore_pressure``).

    Attributes
    ----------
    u : float
        Pore pressure the methods subtract on the base (kPa): 0 on a
        total-stress layer.
    u_water : float
        The water pressure at the base before the total-stress rule
        (kPa): what the line, ru or the grid gives.
    source : str
        Where ``u_water`` came from: 'none', 'line' (the full vertical
        head), 'line_cos2' (FHWA Eq. 6-16), 'ru' or 'grid'.
    alpha_w : float
        Slope of the water line above the base's mid-point (radians, 0
        where level or where there is no line).
    total_stress : bool
        The base's layer is analysed in total stresses.
    """
    u: float = 0.0
    u_water: float = 0.0
    source: str = "none"
    alpha_w: float = 0.0
    total_stress: bool = False


def is_total_stress(layer) -> bool:
    """Whether the layer's strength is a total-stress (undrained)
    strength: Mohr-Coulomb with ``analysis_mode='undrained'``, 'shansep'
    or 'anisotropic'. Its base carries no pore pressure (EM 1110-2-1902
    2-2c, p. 2-4)."""
    model = getattr(layer, "strength_model", "mohr_coulomb")
    if model in ("shansep", "anisotropic"):
        return True
    return model == "mohr_coulomb" and layer.analysis_mode == "undrained"


def water_line_slope(geom, x: float) -> float:
    """alpha_w (radians, >= 0): the slope from the horizontal of the water
    line's stretch above x. Level beyond the line's ends (the line is
    carried on level there) and without a line. At a vertex of the line,
    the flatter of the two stretches that meet there (the larger pore
    pressure)."""
    pts = geom.gwt_points
    if not pts or len(pts) < 2:
        return 0.0
    if x < pts[0][0] or x > pts[-1][0]:
        return 0.0
    best = None
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        if x1 <= x0 or x < x0 or x > x1:
            continue
        a = math.atan(abs(z1 - z0) / (x1 - x0))
        best = a if best is None else min(best, a)
    if best is None:
        return 0.0
    if x == pts[0][0] or x == pts[-1][0]:
        return 0.0            # an end: level on the other side
    return best


def base_pore_pressure(geom, x_mid: float, z_base: float, z_top: float,
                       base_layer, overburden: float,
                       pp_interp=None, gwt_elev=None) -> BasePorePressure:
    """The pore pressure on the base of a slice whose base mid-point is
    (``x_mid``, ``z_base``) under ground at ``z_top``, in ``base_layer``
    (the module docstring gives the rule and its print).

    Parameters
    ----------
    geom : SlopeGeometry
    x_mid, z_base, z_top : float
        The base's mid-point and the ground above it (m).
    base_layer : SlopeSoilLayer
        The layer that holds the base's mid-point.
    overburden : float
        Total overburden stress at the base (kPa): the soil column's
        weight per unit width, each soil at its own unit weight, without
        surcharge or ponded water. Used by ru.
    pp_interp : callable, optional
        The pore-pressure grid's interpolator u(x, z), when
        ``geom.pore_pressure_points`` is given.
    gwt_elev : float, optional
        The water line's elevation at ``x_mid`` when the caller already
        has it; looked up otherwise.
    """
    if gwt_elev is None:
        gwt_elev = geom.gwt_elevation_at(x_mid)
    out = BasePorePressure(total_stress=is_total_stress(base_layer))

    # 1. the water line
    if gwt_elev is not None and gwt_elev > z_base:
        head = gwt_elev - z_base
        out.u_water = GAMMA_W * head
        out.source = "line"
        if (getattr(geom, "water_line_kind", PIEZOMETRIC) == WATER_TABLE
                and gwt_elev <= z_top):
            # FHWA NHI-06-088 Eq. 6-16, p. 6-14: u = h_w gamma_w cos^2(a_w)
            a_w = water_line_slope(geom, x_mid)
            if math.tan(a_w) > _LEVEL:
                out.alpha_w = a_w
                out.u_water = GAMMA_W * head * math.cos(a_w) ** 2
                out.source = "line_cos2"

    # 2. ru where the line gives nothing (EM 1110-2-1902 C-7, p. C-29,
    #    read for a layered column as ru x the total overburden stress)
    if pp_interp is None and out.u_water <= 0 and base_layer.ru > 0:
        out.u_water = base_layer.ru * overburden
        out.source = "ru"

    # 3. a pore-pressure grid governs the base wherever it is given
    if pp_interp is not None:
        out.u_water = pp_interp(x_mid, z_base)
        out.source = "grid"
        out.alpha_w = 0.0

    # 4. no pore pressure on a total-stress base (EM 2-2c, p. 2-4)
    out.u = 0.0 if out.total_stress else out.u_water
    return out


def pond_edges(geom) -> List[float]:
    """The x where ponded water ends: where the water line comes down to
    the ground (above it on one side, at or below it on the other). A
    slice boundary falls at each (account D16). FHWA NHI-06-088 p. 6-14
    locates the sides of the slices "at the intersection of the ground
    water level and the circle" and their tops "at breaks in the slope";
    the pond's edge is the intersection of the water level with the
    ground, where the load on a slice's top changes, and it is cut here
    as an edge of a surcharge is, so that each slice's top is wholly
    under the pond or wholly out of it."""
    pts = geom.gwt_points
    if not pts:
        return []
    x0, x1 = geom.surface_points[0][0], geom.surface_points[-1][0]
    xs = {x0, x1}
    xs.update(p[0] for p in geom.surface_points)
    xs.update(p[0] for p in pts if x0 < p[0] < x1)
    xs = sorted(xs)
    span = max(x1 - x0, 1.0)
    tol = 1e-9 * span
    d = [geom.gwt_elevation_at(x) - geom.ground_elevation_at(x) for x in xs]
    out = []
    for i in range(len(xs) - 1):
        a, b = d[i], d[i + 1]
        if (a > tol) == (b > tol):
            continue
        # ponded on one side only: the edge is where the depth reaches 0
        if abs(a - b) <= 0.0:
            continue
        t = a / (a - b)
        t = min(max(t, 0.0), 1.0)
        out.append(xs[i] + t * (xs[i + 1] - xs[i]))
    return out


def _line_slopes_below_ground(geom) -> bool:
    """Whether the water line has a sloping stretch that lies at or below
    the ground somewhere (so the two readings of the line differ)."""
    pts = geom.gwt_points
    if not pts or len(pts) < 2:
        return False
    x_lo, x_hi = geom.surface_points[0][0], geom.surface_points[-1][0]
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        if x1 <= x0 or abs(z1 - z0) / (x1 - x0) <= _LEVEL:
            continue
        a, b = max(x0, x_lo), min(x1, x_hi)
        if b <= a:
            continue
        xs = [a, b] + [p[0] for p in geom.surface_points if a < p[0] < b]
        for x in xs:
            if geom.gwt_elevation_at(x) <= geom.ground_elevation_at(x) + 1e-9:
                return True
        # between its ends and the ground's breaks both are straight, so
        # the line is below the ground somewhere only if it is at one of
        # these points
    return False


def _layer_under_water(geom, layer) -> bool:
    """Whether ``layer`` holds ground below the water line anywhere (at
    the vertices of the ground, the line and the layer boundaries, and
    midway between them)."""
    if not geom.gwt_points:
        return False
    x_lo, x_hi = geom.surface_points[0][0], geom.surface_points[-1][0]
    xs = {x_lo, x_hi}
    xs.update(p[0] for p in geom.surface_points)
    xs.update(p[0] for p in geom.gwt_points if x_lo < p[0] < x_hi)
    for L in geom.soil_layers:
        for pts in (L.top_boundary_points, L.bottom_boundary_points):
            xs.update(p[0] for p in (pts or ()) if x_lo < p[0] < x_hi)
    xs = sorted(xs)
    xs += [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:])]
    for x in xs:
        zw = geom.gwt_elevation_at(x)
        for held, z_bot, _ in geom.column_at(x):
            if held is layer and z_bot < zw:
                return True
    return False


_FULL_HEAD = ("the full vertical head below the line "
              "(water_line_kind 'piezometric')")
_COS2 = ("u = h_w gamma_w cos^2(alpha_w), alpha_w the slope of the line "
         "above each base (water_line_kind 'water_table')")


def water_statements(geom) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(choices, warnings): what a result says about the pore pressure it
    used (``geotech_common.choices``; the module docstring gives the rules
    and their print).

    * With a water line that slopes below the ground: which reading of the
      line was used (OWN-104), its premise and the other reading. A level
      line reads the same either way and has no entry.
    * With a pore-pressure grid: that the grid governs the base pressure
      and what the line still gives.
    * With ``ru`` on a layer: how it is applied in a layered column.
    * With a total-stress layer under water: that its bases carry no pore
      pressure, and a warning when it was given phi > 0 (EM 1110-2-1902
      Table 2-1 note 2, p. 2-2).
    """
    from geotech_common.choices import MODULE, choice, how_given
    choices: List[Dict[str, Any]] = []
    notes: List[str] = []
    kind = getattr(geom, "water_line_kind", PIEZOMETRIC)
    has_grid = geom.pore_pressure_points is not None
    has_line = bool(geom.gwt_points)

    if has_grid:
        choices.append(choice(
            "pore pressure on slice bases",
            "interpolated from the pore-pressure grid "
            "(pore_pressure_points)", MODULE,
            "the grid governs the base pressure wherever it is given, in "
            "place of the water line and of ru; the water line "
            "(gwt_points), when given, still gives the ponded water's "
            "loads and the saturated unit weights below it"))
    elif has_line and _line_slopes_below_ground(geom):
        corrected = kind == WATER_TABLE
        if corrected:
            basis = (
                "FHWA NHI-06-088 Eq. 6-16 (p. 6-14); UFC 3-220-10 Sec. "
                "7-5.1 (p. 355): a water table corrected to the "
                "piezometric surface by its slope. Premise: seepage "
                "parallel to a straight water table. Not applied where "
                "the line is above the ground (ponded water). The "
                "uncorrected head gives the lower factor of safety")
        else:
            basis = (
                "the line is read as a piezometric line (measured heads), "
                "which is never corrected. If it is a water table with "
                "seepage parallel to it, FHWA NHI-06-088 Eq. 6-16 "
                "(p. 6-14) and UFC 3-220-10 Sec. 7-5.1 (p. 355) print "
                "u = h_w gamma_w cos^2(alpha_w); the uncorrected head "
                "used here gives the lower factor of safety")
        choices.append(choice(
            "pore pressure under the sloping water line",
            _COS2 if corrected else _FULL_HEAD,
            how_given(kind, PIEZOMETRIC, name="water_line_kind"),
            basis, [_FULL_HEAD, _COS2]))

    if not has_grid and any(L.ru > 0 for L in geom.soil_layers):
        choices.append(choice(
            "pore pressure from ru",
            "u = ru x the total overburden stress at the base: the soil "
            "column above it, each soil at its own unit weight, without "
            "surcharge or ponded water", MODULE,
            "EM 1110-2-1902 C-7 (p. C-29) defines ru = u / (gamma z) for "
            "a uniform slope; the sum over a layered column is this "
            "module's reading of that definition. ru is used where the "
            "water line gives no pressure at the base"))

    wet = []
    for L in geom.soil_layers:
        if not is_total_stress(L):
            continue
        if has_grid or L.ru > 0 or _layer_under_water(geom, L):
            wet.append(L)
    if wet:
        names = ", ".join(f"'{L.name}'" for L in wet)
        choices.append(choice(
            "pore pressure on bases in total-stress layers",
            f"none ({names})", MODULE,
            'for total stress analyses "pore water pressures are set to '
            'zero" (EM 1110-2-1902 Table 2-1, p. 2-2), and "the pore water '
            'pressure should be set to zero" (2-2c, p. 2-4)'))
        for L in wet:
            if L.phi > 0:
                notes.append(
                    f"Layer '{L.name}' is analysed in total stresses "
                    f"(undrained strength) and was given phi = {L.phi:g} "
                    "deg, and it lies under water or was given a pore "
                    "pressure. Total-stress envelopes with phi > 0 apply "
                    "only to partially saturated soils (EM 1110-2-1902 "
                    "Table "
                    "2-1, note 2, p. 2-2). This module takes a total-stress "
                    "layer's strength with phi = 0 and subtracts no pore "
                    "pressure on its base (EM 2-2c, p. 2-4): the phi given "
                    "is not used. For an effective-stress analysis give "
                    "c_prime and phi with analysis_mode 'drained'.")
    return choices, notes


def add_water_statements(result, geom) -> None:
    """Put ``water_statements(geom)`` on a search's result: its entries
    after the search's own ``choices`` and its warnings in ``warnings``,
    and take the same entries off the search's critical surface, so they
    are said once, by the search."""
    choices, notes = water_statements(geom)
    result.choices = list(result.choices or []) + choices
    for n in notes:
        if n not in result.warnings:
            result.warnings.append(n)
    critical = getattr(result, "critical", None)
    if critical is not None and getattr(critical, "choices", None):
        whats = {c["what"] for c in choices}
        critical.choices = [c for c in critical.choices
                            if c["what"] not in whats]
        critical.warnings = [w for w in critical.warnings if w not in notes]
