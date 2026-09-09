"""Cross-section property analysis.

All dimensions in mm (documented structural-section exception to the
toolkit's metre convention).

The geometric properties are exact closed-form integrals over the section
outline (see ``polygon_props``), and the torsion/warping constants come from
the published closed-form solutions (see ``torsion``) -- there is no mesh and
no third-party geometry library in this path. Circular shapes are handled
analytically; the rounded corners of RHS and I-section fillets are integrated
as finely-sampled arcs, which is exact to ~1e-6 of the section area.
"""

import math

from section_props_agent import torsion as _t
from section_props_agent.polygon_props import (
    plastic_modulus_x, plastic_modulus_y, polygon_is_simple, region_properties,
    ring_perimeter,
)
from section_props_agent.results import SectionPropertiesResult

#: shape name -> required dimension parameters (mm)
SECTION_SHAPES = {
    "rectangle": ("d", "b"),
    "circle": ("d",),
    "chs": ("d", "t"),
    "rhs": ("d", "b", "t"),
    "i_section": ("d", "b", "t_f", "t_w"),
}

_ARC_SEGMENTS = 48          # per quarter turn
_DEFAULT_N_ACROSS = 150     # polygon torsion grid (Richardson-extrapolated)


# ------------------------------------------------------------- ring builders

def _arc(cx, cy, r, a0_deg, a1_deg, n=_ARC_SEGMENTS, skip_first=True):
    """Sample a circular arc from a0 to a1 (degrees, anti-clockwise if a1>a0)."""
    pts = []
    start = 1 if skip_first else 0
    for i in range(start, n + 1):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _rounded_rect(x0, y0, w, h, r):
    """Anti-clockwise ring of a rectangle with four rounded corners."""
    r = max(0.0, min(r, 0.5 * min(w, h)))
    if r <= 0.0:
        return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    pts = [(x0 + r, y0), (x0 + w - r, y0)]
    pts += _arc(x0 + w - r, y0 + r, r, 270.0, 360.0)
    pts.append((x0 + w, y0 + h - r))
    pts += _arc(x0 + w - r, y0 + h - r, r, 0.0, 90.0)
    pts.append((x0 + r, y0 + h))
    pts += _arc(x0 + r, y0 + h - r, r, 90.0, 180.0)
    pts.append((x0, y0 + r))
    pts += _arc(x0 + r, y0 + r, r, 180.0, 270.0)
    return pts


def _i_section_ring(d, b, t_f, t_w, r):
    """Anti-clockwise ring of an I-section with concave root fillets.

    Origin at the bottom-left of the bounding box, matching the shape's
    construction origin.
    """
    xl = (b - t_w) / 2.0          # web left face
    xr = (b + t_w) / 2.0          # web right face
    r = max(0.0, min(r, min(xl, (d - 2.0 * t_f) / 2.0)))

    pts = [(0.0, 0.0), (b, 0.0), (b, t_f)]
    if r > 0:
        pts.append((xr + r, t_f))
        pts += _arc(xr + r, t_f + r, r, 270.0, 180.0)     # concave, clockwise
    else:
        pts.append((xr, t_f))
    if r > 0:
        pts.append((xr, d - t_f - r))
        pts += _arc(xr + r, d - t_f - r, r, 180.0, 90.0)
        pts.append((b, d - t_f))
    else:
        pts.append((xr, d - t_f))
        pts.append((b, d - t_f))
    pts += [(b, d), (0.0, d), (0.0, d - t_f)]
    if r > 0:
        pts.append((xl - r, d - t_f))
        pts += _arc(xl - r, d - t_f - r, r, 90.0, 0.0)
        pts.append((xl, t_f + r))
        pts += _arc(xl - r, t_f + r, r, 0.0, -90.0)
        pts.append((0.0, t_f))
    else:
        pts += [(xl, d - t_f), (xl, t_f), (0.0, t_f)]
    return pts


def _build_rings(shape, dims):
    """Return ``[(points, sign), ...]`` for a parametric shape, in mm."""
    if shape == "rectangle":
        b, d = dims["b"], dims["d"]
        return [([(0.0, 0.0), (b, 0.0), (b, d), (0.0, d)], 1)]
    if shape == "rhs":
        b, d, t = dims["b"], dims["d"], dims["t"]
        r_out = dims.get("r_out", 2.0 * t)
        if t >= 0.5 * min(b, d):
            raise ValueError(
                f"rhs wall thickness t={t} is too large for {b}x{d}")
        r_in = max(r_out - t, 0.0)
        return [(_rounded_rect(0.0, 0.0, b, d, r_out), 1),
                (_rounded_rect(t, t, b - 2.0 * t, d - 2.0 * t, r_in), -1)]
    if shape == "i_section":
        d, b, t_f, t_w = dims["d"], dims["b"], dims["t_f"], dims["t_w"]
        if 2.0 * t_f >= d:
            raise ValueError(
                f"i_section flanges (2 x {t_f}) do not fit in depth d={d}")
        if t_w >= b:
            raise ValueError(
                f"i_section web t_w={t_w} does not fit in width b={b}")
        return [(_i_section_ring(d, b, t_f, t_w, dims.get("r", 0.0)), 1)]
    raise ValueError(
        f"Unknown shape '{shape}'. Available: {sorted(SECTION_SHAPES)} "
        "(or use analyze_polygon_section for arbitrary outlines).")


# --------------------------------------------------------- circular analytics

def _circular_result(shape, d_o, d_i, warping):
    """Exact properties of a solid circle (d_i = 0) or a CHS."""
    area = math.pi * (d_o ** 2 - d_i ** 2) / 4.0
    i = math.pi * (d_o ** 4 - d_i ** 4) / 64.0
    z = i / (d_o / 2.0)
    s = (d_o ** 3 - d_i ** 3) / 6.0
    j = _t.circle_torsion(d_o) if d_i == 0.0 else _t.chs_torsion(
        d_o, (d_o - d_i) / 2.0)
    return SectionPropertiesResult(
        shape=shape,
        area_mm2=area,
        perimeter_mm=math.pi * d_o,
        cx_mm=0.0, cy_mm=0.0,
        ixx_mm4=i, iyy_mm4=i, ixy_mm4=0.0,
        zxx_plus_mm3=z, zxx_minus_mm3=z, zyy_plus_mm3=z, zyy_minus_mm3=z,
        sxx_mm3=s, syy_mm3=s,
        rx_mm=math.sqrt(i / area), ry_mm=math.sqrt(i / area),
        j_mm4=j if warping else 0.0,
        # a circular section does not warp
        gamma_mm6=0.0 if warping else None,
        i11_mm4=i, i22_mm4=i, phi_deg=0.0,
    )


# ------------------------------------------------------------- the assembler

def _result_from_rings(shape_name, rings, j, gamma):
    props = region_properties(rings)
    sxx, _ = plastic_modulus_x(rings)
    syy, _ = plastic_modulus_y(rings)
    outer = rings[0][0]
    return SectionPropertiesResult(
        shape=shape_name,
        area_mm2=props["area"],
        perimeter_mm=ring_perimeter(outer),
        cx_mm=props["cx"], cy_mm=props["cy"],
        ixx_mm4=props["ixx"], iyy_mm4=props["iyy"], ixy_mm4=props["ixy"],
        zxx_plus_mm3=props["zxx_plus"], zxx_minus_mm3=props["zxx_minus"],
        zyy_plus_mm3=props["zyy_plus"], zyy_minus_mm3=props["zyy_minus"],
        sxx_mm3=sxx, syy_mm3=syy,
        rx_mm=props["rx"], ry_mm=props["ry"],
        j_mm4=j, gamma_mm6=gamma,
        i11_mm4=props["i11"], i22_mm4=props["i22"], phi_deg=props["phi_deg"],
    )


def _shape_torsion(shape, dims):
    """(J, warping constant) for a parametric shape; gamma None if not defined."""
    if shape == "rectangle":
        return _t.rectangle_torsion(dims["b"], dims["d"]), None
    if shape == "rhs":
        return _t.rhs_torsion(dims["d"], dims["b"], dims["t"],
                              dims.get("r_out")), None
    if shape == "i_section":
        r = dims.get("r", 0.0)
        return (_t.i_section_torsion(dims["d"], dims["b"], dims["t_f"],
                                     dims["t_w"], r),
                _t.i_section_warping(dims["d"], dims["b"], dims["t_f"],
                                     dims["t_w"], r))
    raise ValueError(f"no torsion solution registered for shape '{shape}'")


def analyze_section(shape, mesh_size=None, warping=True,
                    **dims) -> SectionPropertiesResult:
    """Compute cross-section properties for a parametric shape.

    Parameters
    ----------
    shape : str
        One of ``rectangle`` (d, b), ``circle`` (d), ``chs`` (d, t),
        ``rhs`` (d, b, t[, r_out]), ``i_section`` (d, b, t_f, t_w[, r]).
    mesh_size : float, optional
        Retained for backwards compatibility. The geometric properties are
        now exact integrals and need no mesh; the value is only used as a
        target cell area (mm^2) for the polygon torsion solve.
    warping : bool
        Compute the torsion constant J and, where it is defined, the warping
        constant. Default True.
    **dims
        Shape dimensions in mm (see per-shape lists above; d = overall
        depth/diameter, b = width, t = wall thickness, t_f/t_w =
        flange/web thickness, r = root radius).

    Returns
    -------
    SectionPropertiesResult
    """
    if shape not in SECTION_SHAPES:
        raise ValueError(
            f"Unknown shape '{shape}'. Available: {sorted(SECTION_SHAPES)}")
    missing = [k for k in SECTION_SHAPES[shape] if k not in dims]
    if missing:
        raise ValueError(
            f"shape '{shape}' requires dimensions {SECTION_SHAPES[shape]}; "
            f"missing {missing}")
    for k, v in dims.items():
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"dimension '{k}' must be a positive number, got {v!r}")

    if shape == "circle":
        return _circular_result("circle", float(dims["d"]), 0.0, warping)
    if shape == "chs":
        d_o = float(dims["d"])
        t = float(dims["t"])
        if 2.0 * t >= d_o:
            raise ValueError(f"chs wall t={t} is too thick for d={d_o}")
        return _circular_result("chs", d_o, d_o - 2.0 * t, warping)

    rings = _build_rings(shape, dims)
    j, gamma = (0.0, None)
    if warping:
        j, gamma = _shape_torsion(shape, dims)
    return _result_from_rings(shape, rings, j, gamma)


def analyze_polygon_section(points, mesh_size=None,
                            warping=True) -> SectionPropertiesResult:
    """Compute cross-section properties for an arbitrary closed polygon.

    Parameters
    ----------
    points : list of (x, y)
        Polygon vertices in mm, in order (closed automatically).
    mesh_size : float, optional
        Target cell area (mm^2) for the torsion solve. Auto-chosen if
        omitted; the geometric properties are exact and unaffected.
    warping : bool
        Solve for the torsion constant J. Default True. The warping constant
        is not defined for an arbitrary outline and is always None here.

    Returns
    -------
    SectionPropertiesResult
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 polygon points, got {len(points)}")
    pts = [(float(x), float(y)) for x, y in points]
    if not polygon_is_simple(pts):
        raise ValueError("polygon is invalid (self-intersecting or zero area)")

    rings = [(pts, 1)]
    j = 0.0
    if warping:
        n_across = _DEFAULT_N_ACROSS
        if mesh_size:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            cell = math.sqrt(float(mesh_size))
            if cell > 0:
                n_across = max(20, min(400, int(round(span / cell))))
        j = _t.polygon_torsion_fd(pts, n_across=n_across)
    return _result_from_rings("polygon", rings, j, None)
