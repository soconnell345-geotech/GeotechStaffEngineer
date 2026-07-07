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
_NEWMARK_POLARITY = ["downslope", "rectified"]
_CRACK_SIDES = ["entry", "exit"]
_CRACK_MODELS = ["strength", "truncation"]


# Top-level geometry params consumed by _build_geometry, and the trial-surface
# spec params — shared by every method's reject_unknown_params valid set.
_GEOM_PARAMS = (
    "surface_points", "soil_layers", "gwt_points", "surcharge",
    "surcharge_x_range", "reinforcement_force", "reinforcement_elevation",
    "kh", "nails", "anchors", "geosynthetics", "stabilizing_piles",
    "tension_crack_depth", "tension_crack_water_depth",
    "tension_crack_side", "tension_crack_model", "pore_pressure_points",
    "surcharges",
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
            R_c=d.get("R_c", 0.0),
            R_phi=d.get("R_phi"),
        ))
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    gwt_points = [tuple(pt) for pt in params["gwt_points"]] if params.get("gwt_points") else None
    surcharge_x_range = tuple(params["surcharge_x_range"]) if params.get("surcharge_x_range") else None
    crack_side = _check_choice(params.get("tension_crack_side", "entry"),
                               _CRACK_SIDES, name="tension_crack_side",
                               method=method)
    crack_model = _check_choice(params.get("tension_crack_model", "strength"),
                                _CRACK_MODELS, name="tension_crack_model",
                                method=method)
    pore_pressure_points = (
        [tuple(p) for p in params["pore_pressure_points"]]
        if params.get("pore_pressure_points") else None)
    surcharges = None
    if params.get("surcharges"):
        surcharges = []
        for z in params["surcharges"]:
            require_keys(z, ["pressure", "x_start", "x_end"], method=method,
                         item_label="surcharges[]")
            surcharges.append((z["pressure"], z["x_start"], z["x_end"]))

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

    stabilizing_piles = None
    if params.get("stabilizing_piles"):
        from slope_stability.reinforcement import StabilizingPile
        stabilizing_piles = []
        for d in params["stabilizing_piles"]:
            require_keys(d, ["x"], method=method, item_label="stabilizing_piles[]")
            _check_choice(d.get("support_convention", "active"),
                          ["active", "passive"],
                          name="stabilizing_piles[].support_convention",
                          method=method)
            stabilizing_piles.append(StabilizingPile(
                x=d["x"], shear_capacity=d.get("shear_capacity"),
                spacing=d.get("spacing", 1.0), z_head=d.get("z_head"),
                z_toe=d.get("z_toe"), ito_matsui=d.get("ito_matsui", False),
                diameter=d.get("diameter"), c=d.get("c"), phi=d.get("phi"),
                gamma=d.get("gamma"),
                force_direction=d.get("force_direction", "horizontal"),
                support_convention=d.get("support_convention", "active"),
            ))

    return SlopeGeometry(
        surface_points=surface_points, soil_layers=soil_layers,
        gwt_points=gwt_points, surcharge=params.get("surcharge", 0.0),
        surcharge_x_range=surcharge_x_range,
        reinforcement_force=params.get("reinforcement_force", 0.0),
        reinforcement_elevation=params.get("reinforcement_elevation"),
        kh=params.get("kh", 0.0),
        nails=nails, anchors=anchors, geosynthetics=geosynthetics,
        stabilizing_piles=stabilizing_piles,
        tension_crack_depth=params.get("tension_crack_depth", 0.0),
        tension_crack_water_depth=params.get("tension_crack_water_depth", 0.0),
        tension_crack_side=crack_side, tension_crack_model=crack_model,
        pore_pressure_points=pore_pressure_points, surcharges=surcharges,
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


_RD_METHODS = ["corps_2stage", "duncan_3stage"]
_RD_STAGE3_NORMAL = ["fellenius", "gle"]
_RD_SURFACE_TYPES = ["circular", "entry_exit", "noncircular", "noncircular_de"]


def _run_rapid_drawdown(params: dict) -> dict:
    from slope_stability.analysis import rapid_drawdown_fos
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + (
            "drawdown_from_elevation", "drawdown_to_elevation", "method",
            "f_interslice", "n_slices", "tol", "stage1_phreatic_points",
            "stage3_effective_normal"),
        method="rapid_drawdown_fos")
    geom = _build_geometry(params, method="rapid_drawdown_fos")
    require_params(params, ["drawdown_from_elevation", "drawdown_to_elevation"],
                   method="rapid_drawdown_fos")
    method = _check_choice(params.get("method", "duncan_3stage"), _RD_METHODS,
                           name="method", method="rapid_drawdown_fos")
    f_int = _check_choice(params.get("f_interslice", "constant"), _F_INTERSLICE,
                          name="f_interslice", method="rapid_drawdown_fos")
    stage3 = _check_choice(
        params.get("stage3_effective_normal", "fellenius"), _RD_STAGE3_NORMAL,
        name="stage3_effective_normal", method="rapid_drawdown_fos")
    result = rapid_drawdown_fos(
        geom, params["drawdown_from_elevation"], params["drawdown_to_elevation"],
        xc=params.get("xc"), yc=params.get("yc"), radius=params.get("radius"),
        slip_surface=_slip_surface_from(params),
        method=method, f_interslice=f_int,
        n_slices=params.get("n_slices", 50), tol=params.get("tol", 1e-4),
        stage1_phreatic_points=params.get("stage1_phreatic_points"),
        stage3_effective_normal=stage3,
    )
    return result.to_dict()


def _run_search_rapid_drawdown(params: dict) -> dict:
    from slope_stability.rapid_drawdown import search_rapid_drawdown
    reject_unknown_params(
        params,
        _GEOM_PARAMS + (
            "drawdown_from_elevation", "drawdown_to_elevation", "method",
            "surface_type", "x_range", "y_range", "x_entry_range",
            "x_exit_range", "nx", "ny", "n_trials", "n_points", "seed",
            "n_slices", "tol", "f_interslice", "stage1_phreatic_points",
            "stage3_effective_normal"),
        method="search_rapid_drawdown")
    geom = _build_geometry(params, method="search_rapid_drawdown")
    require_params(params, ["drawdown_from_elevation", "drawdown_to_elevation"],
                   method="search_rapid_drawdown")
    method = _check_choice(params.get("method", "corps_2stage"), _RD_METHODS,
                           name="method", method="search_rapid_drawdown")
    surface_type = _check_choice(
        params.get("surface_type", "circular"), _RD_SURFACE_TYPES,
        name="surface_type", method="search_rapid_drawdown")
    f_int = _check_choice(params.get("f_interslice", "constant"), _F_INTERSLICE,
                          name="f_interslice", method="search_rapid_drawdown")
    stage3 = _check_choice(
        params.get("stage3_effective_normal", "fellenius"), _RD_STAGE3_NORMAL,
        name="stage3_effective_normal", method="search_rapid_drawdown")
    x_range = tuple(params["x_range"]) if params.get("x_range") else None
    y_range = tuple(params["y_range"]) if params.get("y_range") else None
    x_entry_range = tuple(params["x_entry_range"]) if params.get("x_entry_range") else None
    x_exit_range = tuple(params["x_exit_range"]) if params.get("x_exit_range") else None
    result = search_rapid_drawdown(
        geom, params["drawdown_from_elevation"], params["drawdown_to_elevation"],
        method=method, surface_type=surface_type,
        x_range=x_range, y_range=y_range,
        nx=params.get("nx", 10), ny=params.get("ny", 10),
        x_entry_range=x_entry_range, x_exit_range=x_exit_range,
        n_trials=params.get("n_trials", 500), n_points=params.get("n_points", 5),
        seed=params.get("seed"), n_slices=params.get("n_slices", 50),
        tol=params.get("tol", 1e-4), f_interslice=f_int,
        stage1_phreatic_points=params.get("stage1_phreatic_points"),
        stage3_effective_normal=stage3,
    )
    return result.to_dict()


def _run_yield_acceleration(params: dict) -> dict:
    from slope_stability.newmark import yield_acceleration
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + ("method", "n_slices", "tol",
                                          "kh_max"),
        method="yield_acceleration")
    geom = _build_geometry(params, method="yield_acceleration")
    method = _check_choice(params.get("method", "spencer"), _METHODS,
                           name="method", method="yield_acceleration")
    result = yield_acceleration(
        geom, xc=params.get("xc"), yc=params.get("yc"),
        radius=params.get("radius"), slip_surface=_slip_surface_from(params),
        method=method, n_slices=params.get("n_slices", 50),
        tol=params.get("tol", 1e-4), kh_max=params.get("kh_max", 2.0))
    return result.to_dict()


def _run_newmark_displacement(params: dict) -> dict:
    from slope_stability.newmark import newmark_displacement
    reject_unknown_params(
        params, ("ky", "accel", "dt", "accel_in_g", "polarity"),
        method="newmark_displacement")
    require_params(params, ["ky", "accel", "dt"],
                   method="newmark_displacement")
    polarity = _check_choice(params.get("polarity", "downslope"),
                             _NEWMARK_POLARITY, name="polarity",
                             method="newmark_displacement")
    result = newmark_displacement(
        ky=params["ky"], accel=params["accel"], dt=params["dt"],
        accel_in_g=params.get("accel_in_g", False), polarity=polarity)
    return result.to_dict()


def _run_newmark_jibson2007(params: dict) -> dict:
    from slope_stability.newmark import newmark_jibson2007
    reject_unknown_params(params, ("ky", "amax"), method="newmark_jibson2007")
    require_params(params, ["ky", "amax"], method="newmark_jibson2007")
    result = newmark_jibson2007(ky=params["ky"], amax=params["amax"])
    return result.to_dict()


METHOD_REGISTRY = {
    "analyze_slope": _run_analyze_slope,
    "search_critical_surface": _run_search_critical_surface,
    "compare_methods_table": _run_compare_methods,
    "infinite_slope_fos": _run_infinite_slope,
    "rapid_drawdown_fos": _run_rapid_drawdown,
    "search_rapid_drawdown": _run_search_rapid_drawdown,
    "yield_acceleration": _run_yield_acceleration,
    "newmark_displacement": _run_newmark_displacement,
    "newmark_jibson2007": _run_newmark_jibson2007,
    "fosm_fos": _run_fosm,
    "monte_carlo_fos": _run_monte_carlo,
}

_GEOMETRY_PARAMS = {
    "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,y], ...] array (x increasing)."},
    "soil_layers": {"type": "array", "required": True, "description": "Array of soil-layer dicts: {top_elevation, bottom_elevation, gamma (all required); name, gamma_sat, phi, c_prime, cu, analysis_mode ('drained'|'undrained'), ru, bottom_boundary_points optional}. Per-layer strength_model: 'mohr_coulomb' (default), 'shansep' (su = shansep_S * ocr^shansep_m * sigma'_v; fields shansep_S, shansep_m, ocr, su_min) or 'hoek_brown' (Generalized Hoek-Brown; fields hb_sigci kPa, hb_gsi, hb_mi, hb_D). For rapid_drawdown_fos, low-permeability layers also take the total-stress R-envelope R_c (kPa) and R_phi (deg); R_phi omitted/null => free-draining."},
    "gwt_points": {"type": "array", "required": False, "description": "Groundwater table [[x,y],...]. If above the ground surface, ponded water is auto-detected (water weight + horizontal hydrostatic thrust applied as external loads)."},
    "kh": {"type": "float", "required": False, "default": 0.0, "description": "Horizontal pseudo-static seismic coefficient (acts on soil weight only)."},
    "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Vertical surcharge (kPa). A single loaded zone; use 'surcharges' for several distinct loaded areas."},
    "surcharge_x_range": {"type": "array", "required": False, "description": "[x_start, x_end] extent of the single 'surcharge'."},
    "surcharges": {"type": "array", "required": False, "description": "Several distinct surcharge zones (bench + crest loads, etc.): [{pressure kPa, x_start, x_end}, ...]. Each zone's pressure is summed at any x it covers, ON TOP OF the single 'surcharge'/'surcharge_x_range'. Represent a linearly-varying (trapezoidal) load as its mean uniform pressure or as several thin zones."},
    "tension_crack_depth": {"type": "float", "required": False, "default": 0.0, "description": "Tension crack depth at the crest (m)."},
    "tension_crack_water_depth": {"type": "float", "required": False, "default": 0.0, "description": "Water depth in the tension crack (m)."},
    "tension_crack_side": {"type": "str", "required": False, "default": "entry", "allowed_values": _CRACK_SIDES, "description": "Which crest end the tension crack is on: 'entry' (default, low-x / slip-surface entry) or 'exit' (high-x). Put it on whichever side the crest is on; no need to mirror the slope."},
    "tension_crack_model": {"type": "str", "required": False, "default": "strength", "allowed_values": _CRACK_MODELS, "description": "Tension-crack mechanism: 'strength' (default) keeps the cracked wedge as zero-shear-strength driving soil; 'truncation' removes it from the sliding mass (mass ends at the vertical crack face, as Slide2/UTEXAS do). Truncation is less conservative."},
    "pore_pressure_points": {"type": "array", "required": False, "description": "Discrete pore-pressure field as [[x, z, u], ...] triples (u in kPa) -- a flow-net / TIN sampling. When given, the base pore pressure at each slice is interpolated from this field (linear on the Delaunay triangulation, nearest-node fallback outside the hull, suction clamped to 0), OVERRIDING the gwt_points piezometric line and per-layer ru. The ponded-water buttress still comes from gwt_points, so set BOTH for a reservoir over a flow-net field."},
    "nails": {"type": "array", "required": False, "description": "Soil nails (per metre of slope run): [{x_head, z_head, length required; inclination deg below horizontal=15, bar_diameter mm=25, drill_hole_diameter mm=150, fy MPa=420, bond_stress kPa=100, spacing_h m=1.5}]. Capacity = min(pullout behind slip surface, bar tensile)/spacing_h (FHWA GEC-7)."},
    "anchors": {"type": "array", "required": False, "description": "Tieback anchors: [{x_head, z_head, length, T_allow kN/m required; inclination=15}]. Full T_allow applied when the bond zone crosses the slip surface."},
    "geosynthetics": {"type": "array", "required": False, "description": "Horizontal geosynthetic layers: [{elevation, T_allow kN/m required; x_start, x_end optional}]."},
    "stabilizing_piles": {"type": "array", "required": False, "description": "Single-row stabilizing/micro piles crossing the slope (Ito & Matsui 1975 or a specified shear): [{x required; spacing m=1.0 (D1); z_head, z_toe optional}]. Set the resistance EITHER by shear_capacity (kN per pile; per-metre force = shear_capacity/spacing) OR ito_matsui=true with diameter (m; D2=spacing-diameter) and optional c/phi/gamma (default: soil layer) giving the Ito-Matsui plastic lateral force integrated from head to slip surface. force_direction 'horizontal' (default) or 'normal'. support_convention 'active' (default; the force reduces the driving moment, Slide2 Method A) or 'passive' (added to the resisting side, Method B; more conservative -- affects the circular moment methods Fellenius/Bishop). Stabilizing force at the slip crossing."},
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
    "variables": {"type": "object", "required": True, "description": "Random variables: {'phi': {cov: 0.1}, 'cu:Clay': {mean: 30, std: 5, dist: 'lognormal'}, ...}. Keys are layer parameters (phi, c_prime, cu, gamma, gamma_sat) optionally scoped ':LayerName'; each spec needs cov or std (mean defaults to the layer value); dist is 'normal' (default) or 'lognormal'. Use gamma_sat (not dry gamma) for the unit weight of a submerged slope. A depth-varying undrained-strength law su(z)=a+b*(datum_z-z) is ONE correlated (a,b) variable applied coherently across layers — give an entry with a 'law':'linear_su' key: {'a':{mean,std|cov}, 'b':{mean,std|cov}, 'rho_ab':0..1, 'datum_z': elevation where su=a, 'z_ref':'mid'|'top'|'bottom', 'su_min': floor, 'layers': names|null}. A std-0 component (e.g. fixed intercept) drops out."},
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
    "rapid_drawdown_fos": {
        "category": "Slope Stability",
        "brief": "Rapid-drawdown FOS on a SPECIFIED slip surface: USACE/Army-Corps 2-stage or Duncan-Wright-Wong 3-stage. Low-permeability layers respond undrained after fast reservoir drawdown; give them the total-stress R-envelope (soil_layers[].R_c, R_phi) alongside c_prime/phi (R_phi=null => free-draining). Stage 1 = full-pool effective stresses; stage 2 = undrained strength from the R and effective envelopes (Kc-interpolated for 3-stage); stage 3 (3-stage) substitutes the drained strength where lower; final FOS is solved at the drawn-down pool.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            "drawdown_from_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation BEFORE drawdown (m)."},
            "drawdown_to_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation AFTER drawdown (m); must be below drawdown_from_elevation."},
            "method": {"type": "str", "required": False, "default": "duncan_3stage", "allowed_values": _RD_METHODS, "description": "'duncan_3stage' (Duncan-Wright-Wong, Kc-interpolated undrained strength + drained substitution) or 'corps_2stage' (USACE combined R/effective envelope)."},
            "f_interslice": {"type": "str", "required": False, "default": "constant", "allowed_values": _F_INTERSLICE, "description": "GLE interslice function ('constant' = Spencer)."},
            "n_slices": {"type": "int", "required": False, "default": 50, "description": "Number of slices."},
            "stage1_phreatic_points": {"type": "array", "required": False, "description": "Optional steady-seepage phreatic surface [[x,z],...] for the STAGE-1 consolidation stresses only. Default (omitted) uses a flat full-pool phreatic (hydrostatic to the reservoir) -- the conservative no-through-seepage bound. Supply the flow-net/Casagrande phreatic line (declining from the pool level at the upstream face through the dam) to reproduce the steady-seepage condition Slide2/EM 1110-2-1902 use, which raises the mobilized undrained strengths and the FOS."},
            "stage3_effective_normal": {"type": "str", "required": False, "default": "fellenius", "allowed_values": _RD_STAGE3_NORMAL, "description": "3-stage only: basis for the STAGE-3 drained-substitution effective normal stress. 'fellenius' (default) uses the W*cos(a)/l - u estimate (historical); 'gle' uses the rigorous GLE drawn-down normal (consistent with stage 1), which removes spurious drained substitutions and raises the FOS toward the published Duncan-Wright-Wong value (e.g. Slide2 #96: 1.27 -> 1.37). Inert for the 2-stage method."},
        },
        "returns": {"FOS": "Drawdown factor of safety.", "stage1_FOS": "Full-pool (pre-drawdown) FOS.", "n_undrained_slices": "Slices treated undrained.", "n_drained_substituted": "Stage-3 drained substitutions (3-stage)."},
    },
    "search_rapid_drawdown": {
        "category": "Slope Stability",
        "brief": "SEARCH for the critical (minimum-FOS) slip surface under RAPID-DRAWDOWN strengths (Corps 2-stage / Duncan-Wright-Wong 3-stage). Like search_critical_surface but the per-trial FOS is the rapid-drawdown solve; supports circular and noncircular searches. Use this (not rapid_drawdown_fos) when you do NOT already have a specific trial surface. Low-permeability layers need the R-envelope (soil_layers[].R_c, R_phi).",
        "parameters": {
            **_GEOMETRY_PARAMS,
            "drawdown_from_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation BEFORE drawdown (m)."},
            "drawdown_to_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation AFTER drawdown (m); must be below drawdown_from_elevation."},
            "method": {"type": "str", "required": False, "default": "corps_2stage", "allowed_values": _RD_METHODS, "description": "Rapid-drawdown stage method: 'corps_2stage' (default) or 'duncan_3stage'."},
            "surface_type": {"type": "str", "required": False, "default": "circular", "allowed_values": _RD_SURFACE_TYPES, "description": "Search strategy: 'circular' (centre grid), 'entry_exit' (arcs between entry/exit windows), 'noncircular' (random polylines), 'noncircular_de' (differential-evolution refinement)."},
            "x_range": {"type": "array", "required": False, "description": "[xmin, xmax] for the circle-centre grid (circular)."},
            "y_range": {"type": "array", "required": False, "description": "[ymin, ymax] for the circle-centre grid (circular)."},
            "x_entry_range": {"type": "array", "required": False, "description": "[xmin, xmax] allowed slip-surface entry window."},
            "x_exit_range": {"type": "array", "required": False, "description": "[xmin, xmax] allowed exit window."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x (or entry divisions for entry_exit). Each trial is several LE solves -- keep modest."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y (or exit divisions for entry_exit)."},
            "n_trials": {"type": "int", "required": False, "default": 500, "description": "Random trials for noncircular searches."},
            "n_points": {"type": "int", "required": False, "default": 5, "description": "Polyline vertices for noncircular searches."},
            "n_slices": {"type": "int", "required": False, "default": 50, "description": "Number of slices per trial."},
            "seed": {"type": "int", "required": False, "description": "Random seed for reproducible noncircular searches."},
            "f_interslice": {"type": "str", "required": False, "default": "constant", "allowed_values": _F_INTERSLICE, "description": "GLE interslice function ('constant' = Spencer)."},
            "stage1_phreatic_points": {"type": "array", "required": False, "description": "Optional steady-seepage stage-1 phreatic surface [[x,z],...] (see rapid_drawdown_fos)."},
            "stage3_effective_normal": {"type": "str", "required": False, "default": "fellenius", "allowed_values": _RD_STAGE3_NORMAL, "description": "3-stage stage-3 drained-substitution normal basis: 'fellenius' (default) or 'gle' (rigorous). See rapid_drawdown_fos."},
        },
        "returns": {"FOS": "Minimum drawdown factor of safety found.", "method": "Stage method used.", "surface_type": "Search strategy.", "n_surfaces_evaluated": "Number of surfaces checked.", "search": "Rich search result (critical surface geometry + diagnostics).", "drawdown_detail": "Stage-level detail on the winning surface (stage-1 FOS, drained substitutions)."},
    },
    "yield_acceleration": {
        "category": "Slope Stability",
        "brief": "Yield (critical) seismic coefficient ky for a SPECIFIED slip surface: the horizontal pseudo-static coefficient kh at which the factor of safety = 1.0 (Newmark critical acceleration, ay = ky*g). Found by bisection on the module's pseudo-static FOS. Feed ky into newmark_displacement or newmark_jibson2007. Provide the surface as xc/yc/radius or slip_points.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            "method": {"type": "str", "required": False, "default": "spencer", "allowed_values": _METHODS, "description": "Limit-equilibrium method for the pseudo-static FOS."},
            "n_slices": {"type": "int", "required": False, "default": 50, "description": "Number of slices."},
            "kh_max": {"type": "float", "required": False, "default": 2.0, "description": "Upper bound of the kh bracket; if the surface is still stable at kh_max the result is non-converged with ky=kh_max."},
        },
        "returns": {"ky": "Yield seismic coefficient (fraction of g).", "ay_m_s2": "Yield acceleration = ky*g (m/s^2).", "FOS_static": "Pseudo-static FOS at kh=0.", "converged": "Whether ky was bracketed."},
    },
    "newmark_displacement": {
        "category": "Slope Stability",
        "brief": "Newmark rigid-block permanent DOWNSLOPE displacement by double integration of an earthquake acceleration time history against the yield coefficient ky (no upslope rebound). polarity='downslope' (default) is the standard Newmark/Jibson single-block convention (only the destabilizing polarity drives the block); 'rectified' uses the absolute record so both polarities drive it (conservative, ~2x for a symmetric record). Get ky from yield_acceleration. For a quick estimate without a record, use newmark_jibson2007.",
        "parameters": {
            "ky": {"type": "float", "required": True, "description": "Yield seismic coefficient (fraction of g); ay = ky*g."},
            "accel": {"type": "array", "required": True, "description": "Ground acceleration time history (equally spaced), in m/s^2 (or in g if accel_in_g=true). Sign convention is downslope-positive when polarity='downslope'."},
            "dt": {"type": "float", "required": True, "description": "Time step of the record (s)."},
            "accel_in_g": {"type": "bool", "required": False, "default": False, "description": "True if 'accel' is in units of g rather than m/s^2."},
            "polarity": {"type": "str", "required": False, "default": "downslope", "allowed_values": _NEWMARK_POLARITY, "description": "'downslope' (default; standard Newmark/Jibson — integrate only destabilizing-polarity exceedances of the signed record) or 'rectified' (integrate the absolute record so both polarities drive the block; conservative, orientation-independent)."},
        },
        "returns": {"displacement_m": "Permanent downslope displacement (m).", "displacement_cm": "Same in cm.", "n_exceedances": "Steps exceeding ay.", "duration_s": "Record duration.", "polarity": "Polarity convention used."},
    },
    "newmark_jibson2007": {
        "category": "Slope Stability",
        "brief": "Jibson (2007) regression estimate of Newmark displacement from the critical-acceleration ratio ky/amax (Eq. 6): log10 D[cm] = 0.215 + log10[(1-ky/amax)^2.341 (ky/amax)^-1.438], sigma=0.51 log10. Needs no time history; requires 0 < ky < amax.",
        "parameters": {
            "ky": {"type": "float", "required": True, "description": "Yield / critical seismic coefficient (fraction of g)."},
            "amax": {"type": "float", "required": True, "description": "Peak ground acceleration (fraction of g). Must exceed ky."},
        },
        "returns": {"displacement_cm": "Estimated Newmark displacement (cm).", "displacement_m": "Same in m.", "amax_g": "PGA used."},
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
        "brief": "FOSM / Taylor-series reliability of the FOS on a fixed slip surface (Duncan 2000): central finite differences at +/- 1 sigma per variable -> COV_F, beta (normal + lognormal), probability of failure, per-variable variance contributions. Supports both independent per-layer scalar variables AND a correlated depth-varying su-gradient law (see 'variables').",
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
