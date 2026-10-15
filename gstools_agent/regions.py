"""
Plan regions (polygons with reference levels) and the cell lattice they are
integrated on.

Shared by :func:`gstools_agent.analyze_kriging` (``regions`` on the kriged
mean) and :func:`gstools_agent.analyze_conditional_simulation` (``regions`` on
every realization). A region is ``{name, polygon: [[x, y], ...], level}``;
``level`` is one reference value or a list of them (a sweep).

The integrated quantities over the cells whose CENTRE lies inside the polygon:

* ``above``: volume = sum(max(z - level, 0) * A_cell), area = sum(A_cell
  where z > level)  -- e.g. rock above an excavation level;
* ``below``: volume = sum(max(level - z, 0) * A_cell), area = sum(A_cell
  where z < level)  -- the mirror case, e.g. fill up to a grade.
"""

import math

import numpy as np

QUANTITIES = ("above", "below")

_REGION_KEYS = {"name", "polygon", "level", "levels"}

#: "Nice" cell sizes (m): 1, 2, 2.5, 5 times a power of ten.
_NICE = (1.0, 2.0, 2.5, 5.0)


def shoelace_area(poly) -> float:
    """Plan area of a simple polygon (any vertex order)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) -
                           np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly):
    """Area centroid of a simple polygon (falls back to the vertex mean)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return float(cx), float(cy)


def _finite_number(v, what):
    if isinstance(v, bool):
        raise ValueError(f"{what} must be a number, got {v!r}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {v!r}") from None
    if not math.isfinite(f):
        raise ValueError(f"{what} must be finite, got {v!r}")
    return f


def parse_regions(regions, *, method="conditional_simulation"):
    """Validate ``regions`` and return a list of
    ``{"name", "polygon" (n x 2 array), "levels" (list of float)}``."""
    if isinstance(regions, dict):
        regions = [regions]
    if not isinstance(regions, (list, tuple)) or not regions:
        raise ValueError(
            f"{method}: regions must be a non-empty list of "
            "{name, polygon: [[x, y], ...], level} dicts")
    out, seen = [], set()
    for i, reg in enumerate(regions):
        if not isinstance(reg, dict):
            raise ValueError(f"{method}: regions[{i}] must be a dict with "
                             "name, polygon and level")
        unknown = sorted(set(reg) - _REGION_KEYS)
        if unknown:
            raise ValueError(
                f"{method}: regions[{i}] has unknown key(s) {unknown}; valid "
                f"keys are name, polygon, level (one number or a list)")
        name = str(reg.get("name") or f"region_{i + 1}").strip()
        if name in seen:
            raise ValueError(f"{method}: two regions are named '{name}'; "
                             "give each a different name")
        seen.add(name)
        poly = reg.get("polygon")
        if not isinstance(poly, (list, tuple)) or len(poly) < 3:
            raise ValueError(
                f"{method}: region '{name}' needs a polygon of at least 3 "
                "[x, y] vertices")
        pts = []
        for j, v in enumerate(poly):
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(f"{method}: region '{name}' vertex {j} must "
                                 f"be [x, y], got {v!r}")
            pts.append((_finite_number(v[0], f"region '{name}' vertex {j} x"),
                        _finite_number(v[1], f"region '{name}' vertex {j} y")))
        if len(pts) > 3 and pts[0] == pts[-1]:
            pts = pts[:-1]                       # closed ring given
        p = np.asarray(pts, dtype=float)
        if len({tuple(r) for r in p.tolist()}) < 3:
            raise ValueError(f"{method}: region '{name}' polygon has fewer "
                             "than 3 distinct vertices")
        if shoelace_area(p) <= 0.0:
            raise ValueError(f"{method}: region '{name}' polygon has zero "
                             "area")
        lev = reg.get("level", reg.get("levels"))
        if lev is None:
            raise ValueError(f"{method}: region '{name}' needs a level (one "
                             "number or a list of numbers)")
        levs = lev if isinstance(lev, (list, tuple)) else [lev]
        if not levs:
            raise ValueError(f"{method}: region '{name}' level list is empty")
        levels = [_finite_number(v, f"region '{name}' level") for v in levs]
        out.append({"name": name, "polygon": p, "levels": levels})
    return out


def nice_floor(value: float) -> float:
    """The largest 1/2/2.5/5 x 10^k not above ``value``."""
    if value <= 0:
        raise ValueError("cell size must be positive")
    k = math.floor(math.log10(value))
    best = None
    for kk in (k - 1, k):
        for m in _NICE:
            c = m * 10.0 ** kk
            if c <= value * (1 + 1e-9):
                best = c if best is None else max(best, c)
    return float(best)


def nice_ceil(value: float) -> float:
    """The smallest 1/2/2.5/5 x 10^k not below ``value``."""
    if value <= 0:
        raise ValueError("cell size must be positive")
    k = math.floor(math.log10(value))
    for kk in (k, k + 1):
        for m in _NICE:
            c = m * 10.0 ** kk
            if c >= value * (1 - 1e-9):
                return float(c)
    return float(10.0 ** (k + 1))


#: The automatic cell size gives at most about this many cells in the
#: regions (between a quarter of it and it, by the 1/2/2.5/5 steps).
AUTO_TARGET_CELLS = 3000


def auto_cell_size(regions, target_cells=AUTO_TARGET_CELLS) -> float:
    """The smallest round cell size giving at most about ``target_cells``
    cells inside the regions together."""
    total = sum(shoelace_area(r["polygon"]) for r in regions)
    return nice_ceil(math.sqrt(total / float(target_cells)))


def build_lattice(regions, cell_size, margin=None):
    """Cell-centred lattice over the union of the region polygons plus a
    margin. Cell EDGES are aligned to the regions' lowest x and y, so a
    rectangle drawn on round coordinates (a building footprint) is covered
    by whole cells and its edges do not run through a row of centres.

    Returns a dict with ``gx`` / ``gy`` (cell-centre coordinates), ``cell``,
    ``margin``, and the flattened centre coordinates ``px`` / ``py`` (x varies
    slowest: index = i * ny + j, matching ``np.meshgrid(..., indexing='ij')``).
    """
    cell = float(cell_size)
    allp = np.vstack([r["polygon"] for r in regions])
    xmin, ymin = allp.min(axis=0)
    xmax, ymax = allp.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if margin is None:
        margin = max(2.0 * cell, 0.05 * span)
    margin = float(margin)
    k = math.ceil(margin / cell - 1e-9)
    x0 = float(xmin) - k * cell
    y0 = float(ymin) - k * cell
    nx = max(int(math.ceil((xmax + margin - x0) / cell - 1e-9)), 1)
    ny = max(int(math.ceil((ymax + margin - y0) / cell - 1e-9)), 1)
    gx = x0 + (np.arange(nx) + 0.5) * cell
    gy = y0 + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return {"gx": gx, "gy": gy, "cell": cell, "margin": margin,
            "nx": nx, "ny": ny, "px": X.ravel(), "py": Y.ravel(),
            "x_min": float(x0), "x_max": float(x0 + nx * cell),
            "y_min": float(y0), "y_max": float(y0 + ny * cell)}


def region_masks(regions, px, py, cell=None):
    """Boolean mask per region: cell centres inside its polygon.

    ``matplotlib.path.Path.contains_points`` handles concave (L, C, U
    shaped) simple polygons. A centre exactly ON an edge (a polygon drawn on
    round coordinates often runs through a row of centres) is decided by a
    half-open rule: the test point is nudged by a tiny oblique offset, so of
    two parallel edges one takes the row and the other does not, and a
    rectangle drawn on cell centres gets exactly its own area."""
    from matplotlib.path import Path
    span = float(cell) if cell else max(
        float(np.ptp(px)) if len(px) else 1.0, 1.0)
    eps = 1e-7 * span
    pts = np.column_stack([np.asarray(px, dtype=float) + eps,
                           np.asarray(py, dtype=float) + 0.618 * eps])
    return [Path(r["polygon"]).contains_points(pts) for r in regions]


def integrate(z, levels, cell_area, quantity="above"):
    """Volume and area above (or below) each level.

    ``z`` is (n_cells,) or (n_cells, n_real). Returns ``(volume, area)``,
    each (n_levels,) or (n_levels, n_real)."""
    z = np.asarray(z, dtype=float)
    lv = np.asarray(levels, dtype=float)
    shape = (-1,) + (1,) * z.ndim
    if quantity == "above":
        d = z[None, ...] - lv.reshape(shape)
    elif quantity == "below":
        d = lv.reshape(shape) - z[None, ...]
    else:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    vol = np.clip(d, 0.0, None).sum(axis=1) * cell_area
    area = (d > 0).sum(axis=1) * cell_area
    return vol, area


def pointwise_gaussian(mu, sd, level, quantity="above"):
    """Per cell, from the kriging mean ``mu`` and standard deviation ``sd``
    of a Gaussian field: the probability that the value is beyond the level,
    and the expected excess beyond it,

        above: p = Phi(d), E[max(z - L, 0)] = sd phi(d) + (mu - L) Phi(d),
               d = (mu - L) / sd;
        below: the same with d = (L - mu) / sd.

    These are pointwise expectations: summed over cells they give the
    EXPECTED total (expectation is linear), not the probability that a total
    exceeds a value -- that needs the joint distribution (conditional
    simulation). Where sd = 0 (at an observation) the value is known."""
    from scipy.special import ndtr
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    diff = (mu - level) if quantity == "above" else (level - mu)
    p = (diff > 0).astype(float)
    e = np.clip(diff, 0.0, None)
    pos = sd > 0
    d = diff[pos] / sd[pos]
    p[pos] = ndtr(d)
    e[pos] = (sd[pos] * np.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
              + diff[pos] * p[pos])
    return p, e


def region_geometry(regions, masks, cell_area, obs_x, obs_y):
    """Per-region polygon area, the summed cell area (discretisation check),
    the centroid, and the distance from the centroid to the nearest
    observation."""
    out = []
    for reg, m in zip(regions, masks):
        a_poly = shoelace_area(reg["polygon"])
        n = int(m.sum())
        a_cells = n * cell_area
        cx, cy = polygon_centroid(reg["polygon"])
        d = np.hypot(np.asarray(obs_x) - cx, np.asarray(obs_y) - cy)
        k = int(np.argmin(d))
        out.append({
            "polygon_area": a_poly,
            "n_cells": n,
            "cell_area_sum": a_cells,
            "discretisation_ratio": (a_cells / a_poly) if a_poly > 0
            else float("nan"),
            "centroid": [cx, cy],
            "nearest_observation_distance": float(d[k]),
            "nearest_observation_index": k,
        })
    return out
