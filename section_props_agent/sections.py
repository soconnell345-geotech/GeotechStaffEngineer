"""Cross-section property analysis via sectionproperties.

All dimensions in mm (documented structural-section exception to the
toolkit's metre convention). The library is unit-agnostic; feeding mm
yields mm-based results directly — no conversion applied.
"""

import math

from section_props_agent.results import SectionPropertiesResult
from section_props_agent.section_utils import import_sectionproperties

#: shape name -> required dimension parameters (mm)
SECTION_SHAPES = {
    "rectangle": ("d", "b"),
    "circle": ("d",),
    "chs": ("d", "t"),
    "rhs": ("d", "b", "t"),
    "i_section": ("d", "b", "t_f", "t_w"),
}

_MIN_MESH_ELEMENTS = 8.0   # mesh_sizes target: area / this


def _build_geometry(shape, dims, library):
    if shape == "rectangle":
        return library.rectangular_section(d=dims["d"], b=dims["b"])
    if shape == "circle":
        return library.circular_section(d=dims["d"], n=64)
    if shape == "chs":
        return library.circular_hollow_section(d=dims["d"], t=dims["t"], n=64)
    if shape == "rhs":
        r_out = dims.get("r_out", 2.0 * dims["t"])
        return library.rectangular_hollow_section(
            d=dims["d"], b=dims["b"], t=dims["t"], r_out=r_out, n_r=8)
    if shape == "i_section":
        r = dims.get("r", 0.0)
        return library.i_section(
            d=dims["d"], b=dims["b"], t_f=dims["t_f"], t_w=dims["t_w"],
            r=r, n_r=8 if r > 0 else 1)
    raise ValueError(
        f"Unknown shape '{shape}'. Available: {sorted(SECTION_SHAPES)} "
        "(or use analyze_polygon_section for arbitrary outlines).")


def _analyze_geometry(shape_name, geom, mesh_size=None, warping=True):
    Section, _ = import_sectionproperties()
    if mesh_size is None:
        # heuristic: ~a few hundred elements — accurate J/warping without
        # multi-second meshes for typical member sections
        import numpy as np  # noqa: F401  (sectionproperties dep, always present)
        area_guess = geom.calculate_area()
        mesh_size = max(area_guess / 400.0, 1.0)
    geom.create_mesh(mesh_sizes=[mesh_size])
    sec = Section(geometry=geom)
    sec.calculate_geometric_properties()
    gamma = None
    j = 0.0
    if warping:
        sec.calculate_warping_properties()
        j = float(sec.get_j())
        gamma = float(sec.get_gamma())
    sec.calculate_plastic_properties()

    area = float(sec.get_area())
    perimeter = float(sec.get_perimeter())
    cx, cy = (float(v) for v in sec.get_c())
    ixx, iyy, ixy = (float(v) for v in sec.get_ic())
    zxx_p, zxx_m, zyy_p, zyy_m = (float(v) for v in sec.get_z())
    sxx, syy = (float(v) for v in sec.get_s())
    rx, ry = (float(v) for v in sec.get_rc())
    i11, i22 = (float(v) for v in sec.get_ip())
    phi = float(sec.get_phi())

    return SectionPropertiesResult(
        shape=shape_name, area_mm2=area, perimeter_mm=perimeter,
        cx_mm=cx, cy_mm=cy,
        ixx_mm4=ixx, iyy_mm4=iyy, ixy_mm4=ixy,
        zxx_plus_mm3=zxx_p, zxx_minus_mm3=zxx_m,
        zyy_plus_mm3=zyy_p, zyy_minus_mm3=zyy_m,
        sxx_mm3=sxx, syy_mm3=syy,
        rx_mm=rx, ry_mm=ry,
        j_mm4=j, gamma_mm6=gamma,
        i11_mm4=i11, i22_mm4=i22, phi_deg=phi,
    )


def analyze_section(shape, mesh_size=None, warping=True,
                    **dims) -> SectionPropertiesResult:
    """Compute cross-section properties for a parametric shape.

    Parameters
    ----------
    shape : str
        One of ``rectangle`` (d, b), ``circle`` (d), ``chs`` (d, t),
        ``rhs`` (d, b, t[, r_out]), ``i_section`` (d, b, t_f, t_w[, r]).
    mesh_size : float, optional
        FE mesh target element area (mm^2). Auto-chosen if omitted.
    warping : bool
        Run the (slower) warping analysis for J and the warping constant.
        Default True.
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
    _, library = import_sectionproperties()
    geom = _build_geometry(shape, dims, library)
    return _analyze_geometry(shape, geom, mesh_size=mesh_size, warping=warping)


def analyze_polygon_section(points, mesh_size=None,
                            warping=True) -> SectionPropertiesResult:
    """Compute cross-section properties for an arbitrary closed polygon.

    Parameters
    ----------
    points : list of (x, y)
        Polygon vertices in mm, in order (closed automatically).
    mesh_size : float, optional
        FE mesh target element area (mm^2). Auto-chosen if omitted.
    warping : bool
        Run the warping analysis (J, warping constant). Default True.

    Returns
    -------
    SectionPropertiesResult
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 polygon points, got {len(points)}")
    from sectionproperties.pre import Geometry
    from shapely import Polygon as _ShapelyPolygon
    poly = _ShapelyPolygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid or poly.area <= 0:
        raise ValueError("polygon is invalid (self-intersecting or zero area)")
    geom = Geometry(geom=poly)
    return _analyze_geometry("polygon", geom, mesh_size=mesh_size,
                             warping=warping)
