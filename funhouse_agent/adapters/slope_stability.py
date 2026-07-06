"""Slope stability adapter — flat dict → analyze_slope/search/probabilistic → dict."""

from funhouse_agent.adapters import reject_unknown_params, require_keys, require_params
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import (
    analyze_slope, search_critical_surface, compare_methods_table,
    infinite_slope_fos,
)

_METHODS = ["fellenius", "bishop", "janbu", "spencer", "morgenstern_price",
            "gle"]
_WATER_CONDITIONS = ["dry", "seepage_parallel", "ru"]
_F_INTERSLICE = ["constant", "half_sine", "clipped_sine", "trapezoidal"]
_SURFACE_TYPES = ["circular", "entry_exit", "noncircular", "noncircular_de",
                  "pso", "weak_layer"]
_STRENGTH_MODELS = ["mohr_coulomb", "shansep", "hoek_brown"]


# Top-level geometry params consumed by _build_geometry, and the trial-surface
# spec params — shared by every method's reject_unknown_params valid set.
_GEOM_PARAMS = (
    "surface_points", "soil_layers", "gwt_points", "surcharge",
    "surcharge_x_range", "reinforcement_force", "reinforcement_elevation",
    "kh", "nails", "anchors", "geosynthetics", "tension_crack_depth",
    "tension_crack_water_depth",
)
_SURFACE_PARAMS = ("xc", "yc", "radius", "slip_points")


def _check_choice(value, allowed, *, name: str, method: str):
    if value not in allowed:
        raise ValueError(
            f"{method}: invalid {name} '{value}'. Allowed: {allowed}.")
    return value


def _build_geometry(params: dict, *, method: str) -> SlopeGeometry:
    require_params(params, ["surface_points", "soil_layers"], method=method)
    soil_layers = []
    for i, d in enumerate(params["soil_layers"]):
        require_keys(d, ["top_elevation", "bottom_elevation", "gamma"],
                     method=method, item_label="soil_layers[]")
        strength_model = d.get("strength_model", "mohr_coulomb")
        _check_choice(strength_model, _STRENGTH_MODELS,
                      name="soil_layers[].strength_model", method=method)
        bbp = d.get("bottom_boundary_points")
        soil_layers.append(SlopeSoilLayer(
            name=d.get("name", f"layer_{i + 1}"), top_elevation=d["top_elevation"],
            bottom_elevation=d["bottom_elevation"], gamma=d["gamma"],
            gamma_sat=d.get("gamma_sat"), phi=d.get("phi", 0.0),
            c_prime=d.get("c_prime", 0.0), cu=d.get("cu", 0.0),
            analysis_mode=d.get("analysis_mode", "drained"),
            ru=d.get("ru", 0.0),
            bottom_boundary_points=[tuple(p) for p in bbp] if bbp else None,
            strength_model=strength_model,
            shansep_S=d.get("shansep_S", 0.22),
            shansep_m=d.get("shansep_m", 0.8),
            ocr=d.get("ocr", 1.0),
            su_min=d.get("su_min", 0.0),
            hb_sigci=d.get("hb_sigci", 0.0),
            hb_gsi=d.get("hb_gsi", 50.0),
            hb_mi=d.get("hb_mi", 10.0),
            hb_D=d.get("hb_D", 0.0),
        ))
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    gwt_points = [tuple(pt) for pt in params["gwt_points"]] if params.get("gwt_points") else None
    surcharge_x_range = tuple(params["surcharge_x_range"]) if params.get("surcharge_x_range") else None

    nails = None
    if params.get("nails"):
        from slope_stability.nails import SoilNail
        nails = []
        for d in params["nails"]:
            require_keys(d, ["x_head", "z_head", "length"], method=method,
                         item_label="nails[]")
            nails.append(SoilNail(
                x_head=d["x_head"], z_head=d["z_head"], length=d["length"],
                inclination=d.get("inclination", 15.0),
                bar_diameter=d.get("bar_diameter", 25.0),
                drill_hole_diameter=d.get("drill_hole_diameter", 150.0),
                fy=d.get("fy", 420.0),
                bond_stress=d.get("bond_stress", 100.0),
                spacing_h=d.get("spacing_h", 1.5),
            ))
    anchors = None
    if params.get("anchors"):
        from slope_stability.reinforcement import Anchor
        anchors = []
        for d in params["anchors"]:
            require_keys(d, ["x_head", "z_head", "length", "T_allow"],
                         method=method, item_label="anchors[]")
            anchors.append(Anchor(
                x_head=d["x_head"], z_head=d["z_head"], length=d["length"],
                T_allow=d["T_allow"],
                inclination=d.get("inclination", 15.0),
            ))
    geosynthetics = None
    if params.get("geosynthetics"):
        from slope_stability.reinforcement import Geosynthetic
        geosynthetics = []
        for d in params["geosynthetics"]:
            require_keys(d, ["elevation", "T_allow"], method=method,
                         item_label="geosynthetics[]")
            geosynthetics.append(Geosynthetic(
                elevation=d["elevation"], T_allow=d["T_allow"],
                x_start=d.get("x_start"), x_end=d.get("x_end"),
            ))

    return SlopeGeometry(
        surface_points=surface_points, soil_layers=soil_layers,
        gwt_points=gwt_points, surcharge=params.get("surcharge", 0.0),
        surcharge_x_range=surcharge_x_range,
        reinforcement_force=params.get("reinforcement_force", 0.0),
        reinforcement_elevation=params.get("reinforcement_elevation"),
        kh=params.get("kh", 0.0),
        nails=nails, anchors=anchors, geosynthetics=geosynthetics,
        tension_crack_depth=params.get("tension_crack_depth", 0.0),
        tension_crack_water_depth=params.get("tension_crack_water_depth", 0.0),
    )


def _slip_surface_from(params: dict):
    if params.get("slip_points") is not None:
        from slope_stability.slip_surface import PolylineSlipSurface
        return PolylineSlipSurface(points=[tuple(pt) for pt in params["slip_points"]])
    return None


def _run_analyze_slope(params: dict) -> dict:
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + (
            "method", "f_interslice", "n_slices", "tol",
            "include_slice_data", "compare_methods"),
        method="analyze_slope")
    geom = _build_geometry(params, method="analyze_slope")
    method = _check_choice(params.get("method", "bishop"), _METHODS,
                           name="method", method="analyze_slope")
    f_interslice = _check_choice(
        params.get("f_interslice", "half_sine"), _F_INTERSLICE,
        name="f_interslice", method="analyze_slope")
    result = analyze_slope(
        geom=geom, xc=params.get("xc"), yc=params.get("yc"),
        radius=params.get("radius"), slip_surface=_slip_surface_from(params),
        method=method, f_interslice=f_interslice,
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        include_slice_data=params.get("include_slice_data", False),
        compare_methods=params.get("compare_methods", False),
    )
    return result.to_dict()


def _run_compare_methods(params: dict) -> dict:
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + ("f_interslice", "n_slices", "tol"),
        method="compare_methods_table")
    geom = _build_geometry(params, method="compare_methods_table")
    f_interslice = _check_choice(
        params.get("f_interslice", "half_sine"), _F_INTERSLICE,
        name="f_interslice", method="compare_methods_table")
    tab = compare_methods_table(
        geom, xc=params.get("xc"), yc=params.get("yc"),
        radius=params.get("radius"), slip_surface=_slip_surface_from(params),
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        f_interslice=f_interslice,
    )
    return tab


def _run_search_critical_surface(params: dict) -> dict:
    reject_unknown_params(
        params,
        _GEOM_PARAMS + (
            "surface_type", "method", "x_range", "y_range", "x_entry_range",
            "x_exit_range", "nx", "ny", "n_trials", "n_points", "n_slices",
            "tol", "seed"),
        method="search_critical_surface")
    geom = _build_geometry(params, method="search_critical_surface")
    method = _check_choice(params.get("method", "bishop"), _METHODS,
                           name="method", method="search_critical_surface")
    surface_type = _check_choice(
        params.get("surface_type", "circular"), _SURFACE_TYPES,
        name="surface_type", method="search_critical_surface")
    x_range = tuple(params["x_range"]) if params.get("x_range") else None
    y_range = tuple(params["y_range"]) if params.get("y_range") else None
    x_entry_range = tuple(params["x_entry_range"]) if params.get("x_entry_range") else None
    x_exit_range = tuple(params["x_exit_range"]) if params.get("x_exit_range") else None
    result = search_critical_surface(
        geom=geom, x_range=x_range, y_range=y_range,
        nx=params.get("nx", 10), ny=params.get("ny", 10),
        method=method,
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        surface_type=surface_type,
        x_entry_range=x_entry_range, x_exit_range=x_exit_range,
        n_trials=params.get("n_trials", 500), n_points=params.get("n_points", 5),
        seed=params.get("seed"),
    )
    return result.to_dict()


def _run_fosm(params: dict) -> dict:
    from slope_stability.probabilistic import fosm_fos
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + ("variables", "method", "n_slices",
                                          "tol"),
        method="fosm_fos")
    geom = _build_geometry(params, method="fosm_fos")
    require_params(params, ["variables"], method="fosm_fos")
    method = _check_choice(params.get("method", "bishop"), _METHODS,
                           name="method", method="fosm_fos")
    result = fosm_fos(
        geom, params["variables"],
        xc=params.get("xc"), yc=params.get("yc"), radius=params.get("radius"),
        slip_surface=_slip_surface_from(params),
        method=method, n_slices=params.get("n_slices", 30),
        tol=params.get("tol", 1e-4),
    )
    return result.to_dict()


def _run_monte_carlo(params: dict) -> dict:
    from slope_stability.probabilistic import monte_carlo_fos
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + (
            "variables", "method", "n", "seed", "n_slices", "tol",
            "research_surface"),
        method="monte_carlo_fos")
    geom = _build_geometry(params, method="monte_carlo_fos")
    require_params(params, ["variables"], method="monte_carlo_fos")
    method = _check_choice(params.get("method", "bishop"), _METHODS,
                           name="method", method="monte_carlo_fos")
    result = monte_carlo_fos(
        geom, params["variables"],
        xc=params.get("xc"), yc=params.get("yc"), radius=params.get("radius"),
        slip_surface=_slip_surface_from(params),
        method=method, n=params.get("n", 1000), seed=params.get("seed"),
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        research_surface=params.get("research_surface", False),
    )
    return result.to_dict()


def _run_infinite_slope(params: dict) -> dict:
    reject_unknown_params(
        params,
        ("slope_angle", "phi", "gamma", "c", "depth", "water_condition",
         "gamma_w", "ru", "water_depth"),
        method="infinite_slope_fos")
    require_params(params, ["slope_angle", "phi", "gamma"],
                   method="infinite_slope_fos")
    water = _check_choice(params.get("water_condition", "dry"),
                          _WATER_CONDITIONS, name="water_condition",
                          method="infinite_slope_fos")
    result = infinite_slope_fos(
        slope_angle=params["slope_angle"], phi=params["phi"],
        gamma=params["gamma"], c=params.get("c", 0.0),
        depth=params.get("depth", 1.0), water_condition=water,
        gamma_w=params.get("gamma_w", 9.81), ru=params.get("ru", 0.0),
        water_depth=params.get("water_depth", 0.0),
    )
    return result.to_dict()


METHOD_REGISTRY = {
    "analyze_slope": _run_analyze_slope,
    "search_critical_surface": _run_search_critical_surface,
    "compare_methods_table": _run_compare_methods,
    "infinite_slope_fos": _run_infinite_slope,
    "fosm_fos": _run_fosm,
    "monte_carlo_fos": _run_monte_carlo,
}

_GEOMETRY_PARAMS = {
    "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,y], ...] array (x increasing)."},
    "soil_layers": {"type": "array", "required": True, "description": "Array of soil-layer dicts: {top_elevation, bottom_elevation, gamma (all required); name, gamma_sat, phi, c_prime, cu, analysis_mode ('drained'|'undrained'), ru, bottom_boundary_points optional}. Per-layer strength_model: 'mohr_coulomb' (default), 'shansep' (su = shansep_S * ocr^shansep_m * sigma'_v; fields shansep_S, shansep_m, ocr, su_min) or 'hoek_brown' (Generalized Hoek-Brown; fields hb_sigci kPa, hb_gsi, hb_mi, hb_D)."},
    "gwt_points": {"type": "array", "required": False, "description": "Groundwater table [[x,y],...]. If above the ground surface, ponded water is auto-detected (water weight + horizontal hydrostatic thrust applied as external loads)."},
    "kh": {"type": "float", "required": False, "default": 0.0, "description": "Horizontal pseudo-static seismic coefficient (acts on soil weight only)."},
    "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Vertical surcharge (kPa)."},
    "surcharge_x_range": {"type": "array", "required": False, "description": "[x_start, x_end] extent of the surcharge."},
    "tension_crack_depth": {"type": "float", "required": False, "default": 0.0, "description": "Tension crack depth at the crest (m)."},
    "tension_crack_water_depth": {"type": "float", "required": False, "default": 0.0, "description": "Water depth in the tension crack (m)."},
    "nails": {"type": "array", "required": False, "description": "Soil nails (per metre of slope run): [{x_head, z_head, length required; inclination deg below horizontal=15, bar_diameter mm=25, drill_hole_diameter mm=150, fy MPa=420, bond_stress kPa=100, spacing_h m=1.5}]. Capacity = min(pullout behind slip surface, bar tensile)/spacing_h (FHWA GEC-7)."},
    "anchors": {"type": "array", "required": False, "description": "Tieback anchors: [{x_head, z_head, length, T_allow kN/m required; inclination=15}]. Full T_allow applied when the bond zone crosses the slip surface."},
    "geosynthetics": {"type": "array", "required": False, "description": "Horizontal geosynthetic layers: [{elevation, T_allow kN/m required; x_start, x_end optional}]."},
    "reinforcement_force": {"type": "float", "required": False, "default": 0.0, "description": "Single equivalent horizontal reinforcement force (kN/m). Simpler alternative to nails/anchors."},
    "reinforcement_elevation": {"type": "float", "required": False, "description": "Elevation of the equivalent reinforcement force (m)."},
}

_SURFACE_SPEC_PARAMS = {
    "xc": {"type": "float", "required": False, "description": "Circle center x (m). Required for circular."},
    "yc": {"type": "float", "required": False, "description": "Circle center y (m). Required for circular."},
    "radius": {"type": "float", "required": False, "description": "Circle radius (m). Required for circular."},
    "slip_points": {"type": "array", "required": False, "description": "Polyline slip surface [[x,y],...] for noncircular. Use instead of xc/yc/radius."},
}

_VARIABLES_PARAM = {
    "variables": {"type": "object", "required": True, "description": "Random variables: {'phi': {cov: 0.1}, 'cu:Clay': {mean: 30, std: 5, dist: 'lognormal'}, ...}. Keys are layer parameters (phi, c_prime, cu, gamma, ...) optionally scoped ':LayerName'; each spec needs cov or std (mean defaults to the layer value); dist is 'normal' (default) or 'lognormal'."},
}

METHOD_INFO = {
    "analyze_slope": {
        "category": "Slope Stability",
        "brief": "Analyze ONE specified slip surface (Fellenius/Bishop/Janbu+f0/Spencer/Morgenstern-Price/GLE). Requires a trial circle (xc/yc/radius) or slip_points. If you do NOT already have a specific trial surface, use search_critical_surface instead — it auto-finds the critical surface and avoids 'circle does not intersect the ground surface' errors. Supports reinforcement (nails/anchors/geosynthetics), ponded water (auto), SHANSEP and Hoek-Brown layer strengths, pseudo-static kh, tension cracks.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            "method": {"type": "str", "required": False, "default": "bishop", "allowed_values": _METHODS, "description": "Limit-equilibrium method. 'janbu' reports the f0-corrected FOS; 'spencer'/'morgenstern_price'/'gle' use the rigorous Fredlund-Krahn GLE engine (interslice E/X integration)."},
            "f_interslice": {"type": "str", "required": False, "default": "half_sine", "allowed_values": _F_INTERSLICE, "description": "Interslice force function f(x) for morgenstern_price/gle ('constant' reproduces Spencer)."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
            "include_slice_data": {"type": "bool", "required": False, "default": False, "description": "Include the per-slice force table (W, N', S_mob, u*l, alpha; interslice E/X and line of thrust for rigorous methods)."},
            "compare_methods": {"type": "bool", "required": False, "default": False, "description": "Also report FOS for every method on this surface."},
        },
        "returns": {"FOS": "Factor of safety.", "method": "Method used.", "lambda_mp": "Interslice scaling (M-P/GLE).", "theta_spencer_deg": "Spencer interslice angle.", "FOS_janbu_corrected": "Janbu f0-corrected FOS.", "slice_data": "Per-slice force table (include_slice_data=true).", "thrust_line": "Line of thrust [[x,z],...] (rigorous methods + include_slice_data)."},
    },
    "compare_methods_table": {
        "category": "Slope Stability",
        "brief": "Run ALL limit-equilibrium methods (OMS, Bishop, Janbu uncorr/corr, Spencer, Morgenstern-Price) on ONE slip surface and return a side-by-side FOS table (Fredlund & Krahn 1977 style method comparison).",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            "f_interslice": {"type": "str", "required": False, "default": "half_sine", "allowed_values": _F_INTERSLICE, "description": "Interslice function for the Morgenstern-Price row."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
        },
        "returns": {"rows": "[{method, FOS, detail}, ...] per method.", "surface": "Surface description.", "summary": "Formatted text table."},
    },
    "infinite_slope_fos": {
        "category": "Slope Stability",
        "brief": "Infinite-slope (planar translational) factor of safety in closed form (Duncan-Wright): FOS = [c' + (gamma*z*cos^2(beta) - u)*tan(phi')] / (gamma*z*sin(beta)*cos(beta)). For the shallow surface-parallel failure mechanism; the depth cancels for cohesionless soil (FOS = tan(phi')/tan(beta) dry, or gamma'/gamma * that for seepage at the surface). Use analyze_slope/search_critical_surface for circular/noncircular mechanisms.",
        "parameters": {
            "slope_angle": {"type": "float", "required": True, "description": "Slope inclination beta (degrees from horizontal), 0-90."},
            "phi": {"type": "float", "required": True, "description": "Effective friction angle (degrees)."},
            "gamma": {"type": "float", "required": True, "description": "Total (moist/saturated) unit weight (kN/m3)."},
            "c": {"type": "float", "required": False, "default": 0.0, "description": "Effective cohesion (kPa). Default 0 (cohesionless)."},
            "depth": {"type": "float", "required": False, "default": 1.0, "description": "Slip-plane depth z below the surface (m). Cancels for c'=0; matters when c'>0."},
            "water_condition": {"type": "str", "required": False, "default": "dry", "allowed_values": _WATER_CONDITIONS, "description": "'dry' (u=0); 'seepage_parallel' (steady seepage parallel to the slope, phreatic surface at depth water_depth, u=gamma_w*(z-water_depth)*cos^2 beta); 'ru' (u=ru*gamma*z)."},
            "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Unit weight of water (kN/m3), for seepage_parallel."},
            "ru": {"type": "float", "required": False, "default": 0.0, "description": "Pore-pressure ratio, for water_condition='ru'."},
            "water_depth": {"type": "float", "required": False, "default": 0.0, "description": "Depth of the phreatic surface below the ground (m), for seepage_parallel. 0 = water table at the surface."},
        },
        "returns": {"FOS": "Factor of safety.", "normal_stress_kPa": "sigma_n on the slip plane.", "shear_stress_kPa": "Driving shear tau.", "pore_pressure_kPa": "u on the slip plane."},
    },
    "search_critical_surface": {
        "category": "Slope Stability",
        "brief": "Search for the critical slip surface (minimum FOS): circular centre-grid, entry-exit arcs, random noncircular polylines, differential-evolution noncircular refinement, PSO, or weak-layer-biased search.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            "surface_type": {"type": "str", "required": False, "default": "circular", "allowed_values": _SURFACE_TYPES, "description": "Search strategy. 'entry_exit' grids circular arcs between entry/exit windows; 'noncircular_de' refines random polylines with scipy differential evolution."},
            "method": {"type": "str", "required": False, "default": "bishop", "allowed_values": _METHODS, "description": "Limit-equilibrium method (noncircular searches use spencer when 'bishop' is requested)."},
            "x_range": {"type": "array", "required": False, "description": "[xmin, xmax] for circle center search."},
            "y_range": {"type": "array", "required": False, "description": "[ymin, ymax] for circle center search."},
            "x_entry_range": {"type": "array", "required": False, "description": "[xmin, xmax] allowed slip-surface entry window on the ground."},
            "x_exit_range": {"type": "array", "required": False, "description": "[xmin, xmax] allowed exit window."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x (or entry divisions for entry_exit)."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y (or exit divisions for entry_exit)."},
            "n_trials": {"type": "int", "required": False, "default": 500, "description": "Random trials for noncircular searches."},
            "n_points": {"type": "int", "required": False, "default": 5, "description": "Polyline vertices for noncircular searches."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices per trial."},
            "seed": {"type": "int", "required": False, "description": "Random seed for reproducible noncircular/PSO/DE searches."},
        },
        "returns": {"critical": "Critical surface result (FOS, geometry).", "n_surfaces_evaluated": "Number of surfaces checked."},
    },
    "fosm_fos": {
        "category": "Slope Stability",
        "brief": "FOSM / Taylor-series reliability of the FOS on a fixed slip surface (Duncan 2000): central finite differences at +/- 1 sigma per variable -> COV_F, beta (normal + lognormal), probability of failure, per-variable variance contributions.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            **_VARIABLES_PARAM,
            "method": {"type": "str", "required": False, "default": "bishop", "allowed_values": _METHODS, "description": "Limit-equilibrium method for each FOS evaluation."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
        },
        "returns": {"FOS_mean_values": "FOS at mean (most-likely) values.", "COV_F": "Coefficient of variation of FOS.", "beta_lognormal": "Lognormal reliability index (Duncan 2000).", "pf_lognormal": "Probability of failure.", "variable_variance_pct": "Per-variable share of Var[FOS]."},
    },
    "monte_carlo_fos": {
        "category": "Slope Stability",
        "brief": "Monte Carlo FOS distribution on a fixed slip surface (optionally re-searching the critical surface per realization): samples layer parameters (normal/lognormal), returns pf, FOS statistics and histogram.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            **_VARIABLES_PARAM,
            "method": {"type": "str", "required": False, "default": "bishop", "allowed_values": _METHODS, "description": "Limit-equilibrium method per realization."},
            "n": {"type": "int", "required": False, "default": 1000, "description": "Number of realizations."},
            "seed": {"type": "int", "required": False, "description": "Random seed for reproducibility."},
            "research_surface": {"type": "bool", "required": False, "default": False, "description": "Re-search the critical surface for every realization (slow; default keeps the surface fixed, standard practice)."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
        },
        "returns": {"pf": "Probability of failure (FOS < 1).", "FOS_mean": "Mean FOS.", "FOS_std": "Std of FOS.", "beta_lognormal": "Lognormal reliability index from the sample moments.", "histogram_bins": "Histogram bin edges.", "histogram_counts": "Histogram counts."},
    },
}
