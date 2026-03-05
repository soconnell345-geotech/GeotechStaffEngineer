"""Slope stability adapter — flat dict → analyze_slope/search → dict."""

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import analyze_slope, search_critical_surface


def _build_geometry(params: dict) -> SlopeGeometry:
    soil_layers = [
        SlopeSoilLayer(
            name=d["name"], top_elevation=d["top_elevation"],
            bottom_elevation=d["bottom_elevation"], gamma=d["gamma"],
            gamma_sat=d.get("gamma_sat"), phi=d.get("phi", 0.0),
            c_prime=d.get("c_prime", 0.0), cu=d.get("cu", 0.0),
            analysis_mode=d.get("analysis_mode", "drained"),
        ) for d in params["soil_layers"]
    ]
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    gwt_points = [tuple(pt) for pt in params["gwt_points"]] if params.get("gwt_points") else None
    surcharge_x_range = tuple(params["surcharge_x_range"]) if params.get("surcharge_x_range") else None
    return SlopeGeometry(
        surface_points=surface_points, soil_layers=soil_layers,
        gwt_points=gwt_points, surcharge=params.get("surcharge", 0.0),
        surcharge_x_range=surcharge_x_range,
        reinforcement_force=params.get("reinforcement_force", 0.0),
        reinforcement_elevation=params.get("reinforcement_elevation"),
        kh=params.get("kh", 0.0),
    )


def _run_analyze_slope(params: dict) -> dict:
    geom = _build_geometry(params)
    slip_surface = None
    if params.get("slip_points") is not None:
        from slope_stability.slip_surface import PolylineSlipSurface
        slip_surface = PolylineSlipSurface(points=[tuple(pt) for pt in params["slip_points"]])
    result = analyze_slope(
        geom=geom, xc=params.get("xc"), yc=params.get("yc"),
        radius=params.get("radius"), slip_surface=slip_surface,
        method=params.get("method", "bishop"),
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        include_slice_data=params.get("include_slice_data", False),
        compare_methods=params.get("compare_methods", False),
    )
    return result.to_dict()


def _run_search_critical_surface(params: dict) -> dict:
    geom = _build_geometry(params)
    x_range = tuple(params["x_range"]) if params.get("x_range") else None
    y_range = tuple(params["y_range"]) if params.get("y_range") else None
    x_entry_range = tuple(params["x_entry_range"]) if params.get("x_entry_range") else None
    x_exit_range = tuple(params["x_exit_range"]) if params.get("x_exit_range") else None
    result = search_critical_surface(
        geom=geom, x_range=x_range, y_range=y_range,
        nx=params.get("nx", 10), ny=params.get("ny", 10),
        method=params.get("method", "bishop"),
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        surface_type=params.get("surface_type", "circular"),
        x_entry_range=x_entry_range, x_exit_range=x_exit_range,
        n_trials=params.get("n_trials", 500), n_points=params.get("n_points", 5),
        seed=params.get("seed"),
    )
    return result.to_dict()


METHOD_REGISTRY = {
    "analyze_slope": _run_analyze_slope,
    "search_critical_surface": _run_search_critical_surface,
}

METHOD_INFO = {
    "analyze_slope": {
        "category": "Slope Stability",
        "brief": "Single slip surface analysis (Fellenius/Bishop/Spencer). Circular or noncircular.",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,y], ...] array."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of {name, top_elevation, bottom_elevation, gamma, phi, c_prime, analysis_mode} dicts."},
            "xc": {"type": "float", "required": False, "description": "Circle center x (m). Required for circular."},
            "yc": {"type": "float", "required": False, "description": "Circle center y (m). Required for circular."},
            "radius": {"type": "float", "required": False, "description": "Circle radius (m). Required for circular."},
            "slip_points": {"type": "array", "required": False, "description": "Polyline slip surface [[x,y],...] for noncircular. Use instead of xc/yc/radius."},
            "method": {"type": "str", "required": False, "default": "bishop", "description": "fellenius/bishop/spencer."},
            "gwt_points": {"type": "array", "required": False, "description": "Groundwater table [[x,y],...]."},
            "include_slice_data": {"type": "bool", "required": False, "default": False, "description": "Include per-slice forces and stresses."},
            "compare_methods": {"type": "bool", "required": False, "default": False, "description": "Compare all 3 methods."},
        },
        "returns": {"FOS": "Factor of safety.", "method": "Method used.", "n_slices": "Number of slices."},
    },
    "search_critical_surface": {
        "category": "Slope Stability",
        "brief": "Search for critical slip surface (minimum FOS) via grid or random search.",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface [[x,y],...]."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of soil layer dicts."},
            "surface_type": {"type": "str", "required": False, "default": "circular", "description": "circular or noncircular."},
            "x_range": {"type": "array", "required": False, "description": "[xmin, xmax] for circle center search."},
            "y_range": {"type": "array", "required": False, "description": "[ymin, ymax] for circle center search."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y."},
            "method": {"type": "str", "required": False, "default": "bishop", "description": "fellenius/bishop/spencer."},
            "n_trials": {"type": "int", "required": False, "default": 500, "description": "Random trials for noncircular search."},
        },
        "returns": {"min_FOS": "Minimum factor of safety found.", "critical": "Critical surface parameters.", "n_surfaces_evaluated": "Number of surfaces checked."},
    },
}
