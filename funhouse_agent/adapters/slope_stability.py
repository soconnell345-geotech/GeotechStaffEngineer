"""Slope stability adapter — flat dict → analyze_slope/search/probabilistic → dict."""

# check_nested_keys lived here until GM-58 moved it to
# funhouse_agent.adapters for every adapter; imported here so older imports
# of it from this module keep working.
from funhouse_agent.adapters import (
    check_nested_keys, declare_applies_when, declare_when_absent, field_keys,
    refuse_given, reject_unknown_params, require_keys, require_params,
    without_unused,
)
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import (
    analyze_slope, search_critical_surface, compare_methods_table,
    infinite_slope_fos,
)
from slope_stability.search import SEARCH_TIME_BUDGET_S, TIME_BUDGET_MAX_S
from slope_stability.methods import M_ALPHA_RULES, NEGATIVE_TERM_RULES
from slope_stability.search import LOW_M_ALPHA_ACTIONS
from slope_stability.search_supplements import headline_report

_METHODS = ["fellenius", "bishop", "janbu", "spencer", "morgenstern_price",
            "gle"]
_WATER_CONDITIONS = ["dry", "seepage_parallel", "ru"]
_F_INTERSLICE = ["constant", "half_sine", "clipped_sine", "trapezoidal"]
_SURFACE_TYPES = ["circular", "entry_exit", "noncircular", "noncircular_de",
                  "pso", "weak_layer"]
_STRENGTH_MODELS = ["mohr_coulomb", "shansep", "hoek_brown", "anisotropic"]
_NEWMARK_POLARITY = ["downslope", "rectified"]
_CRACK_SIDES = ["crest", "entry", "exit"]
_CRACK_MODELS = ["truncation", "strength"]
_CRACK_DEPTH_BASES = ["given", "rankine", "min_fos"]
_REVERSE_CURVATURE = ["tension_crack", "discard"]
_WATER_LINE_KINDS = ["piezometric", "water_table"]
_ASSISTING_SUPPORT = ["slack", "as_given"]
_FORCE_DIRECTIONS = ["along_reinforcement", "along_slip_surface"]
_PILE_FORCE_DIRECTIONS = ["horizontal", "along_slip_surface"]
_PULLOUT_BASES = ["dwb2014", "gec11", "given", "not_checked"]


# Top-level geometry params consumed by _build_geometry, and the trial-surface
# spec params — shared by every method's reject_unknown_params valid set.
_GEOM_PARAMS = (
    "surface_points", "soil_layers", "gwt_points", "surcharge",
    "surcharge_x_range", "reinforcement_force", "reinforcement_elevation",
    "kh", "nails", "anchors", "geosynthetics", "stabilizing_piles",
    "tension_crack_depth", "tension_crack_water_depth",
    "tension_crack_side", "tension_crack_model", "pore_pressure_points",
    "surcharges", "reverse_curvature", "tension_crack_depth_basis",
    "water_line_kind", "assisting_support",
)
_SURFACE_PARAMS = ("xc", "yc", "radius", "slip_points", "arc_x_start",
                   "arc_x_end")


#: The most time a slope search may be given through the agent (GM-123): a
#: third of the agent's 30-minute turn, as the SALib tools allow. A larger
#: value is refused, not clipped, so the agent knows what it got.
_TIME_BUDGET_MAX_S = TIME_BUDGET_MAX_S


def _time_budget(params: dict, method: str) -> float:
    """The search's ``time_budget_s`` (GM-123): `SEARCH_TIME_BUDGET_S` when
    left out (the library itself has no budget unless given one), above 0
    and at most `_TIME_BUDGET_MAX_S`. Used by every tool that runs a slope
    search for the agent (this adapter, calc_package, the project run)."""
    value = params.get("time_budget_s", SEARCH_TIME_BUDGET_S)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{method}: time_budget_s must be a number of "
                         f"seconds, got {value!r}.") from None
    if not 0.0 < value <= _TIME_BUDGET_MAX_S:
        raise ValueError(
            f"{method}: time_budget_s must be above 0 and at most "
            f"{_TIME_BUDGET_MAX_S:.0f} s (a third of the agent's 30-minute "
            f"turn), got {value:g}.")
    return value


def budget_is_default(choices, params: dict) -> None:
    """A search's time budget is the tool's default when the agent gave
    none (GM-123): say so in its ``choices`` entry (the library sees the
    tool's default as given)."""
    if "time_budget_s" in params:
        return
    for c in choices or ():
        if c.get("what") == "time budget of the search":
            c["how"] = "default"


def _check_choice(value, allowed, *, name: str, method: str):
    if value not in allowed:
        raise ValueError(
            f"{method}: invalid {name} '{value}'. Allowed: {allowed}.")
    return value


def _force_direction(d: dict, kind: str, method: str) -> str:
    """A support's ``force_direction``, checked (GM-140 step 8)."""
    return _check_choice(d.get("force_direction", "along_reinforcement"),
                         _FORCE_DIRECTIONS,
                         name=f"{kind}[].force_direction", method=method)


def _support_convention(d: dict, kind: str, method: str) -> str:
    """A support's 'active' (default, Method A) or 'passive' (Method B)."""
    return _check_choice(d.get("support_convention", "active"),
                         ["active", "passive"],
                         name=f"{kind}[].support_convention", method=method)


# The keys a nested dict may carry: the dataclass's own fields, which
# `_build_geometry` passes through one for one.
_LAYER_KEYS = field_keys(SlopeSoilLayer)

# Words for a layer's strength or weight that the generic near-miss matcher
# cannot place. Suggestions only, never renamed: 'c' and 'cohesion' could be
# the drained c' or the undrained cu, so both are named and the caller picks.
_LAYER_KEY_HINTS = {
    "c": ("c_prime", "cu"), "cohesion": ("c_prime", "cu"),
    "ceff": ("c_prime",), "ceffective": ("c_prime",),
    "effectivecohesion": ("c_prime",),
    "su": ("cu",), "su0": ("cu",), "undrainedstrength": ("cu",),
    "undrainedshearstrength": ("cu",),
    "phiprime": ("phi",), "phieff": ("phi",), "frictionangle": ("phi",),
    "unitweight": ("gamma",), "gammatotal": ("gamma",),
    "top": ("top_elevation",), "bottom": ("bottom_elevation",),
}

# Said when a did-you-mean offers both strengths ('c', 'cohesion').
_LAYER_KEY_NOTES = {
    ("c_prime", "cu"): ("(c_prime is the drained effective cohesion; cu is "
                        "the undrained strength, used with "
                        "analysis_mode='undrained'.)"),
}

# Keys of the probabilistic `variables` specs and `correlations` dicts, as
# slope_stability.probabilistic reads them ('source' is the citation the
# slope calc package prints beside a variable).
_VARIABLE_KEYS = ("mean", "std", "cov", "dist", "source")
_LAW_KEYS = ("law", "a", "b", "rho_ab", "datum_z", "z_ref", "su_min",
             "layers", "source")
_LAW_PAIR_KEYS = ("mean", "std", "cov")
_CORRELATION_KEYS = ("var1", "var2", "key1", "key2", "a", "b", "rho",
                     "rho_ab")


def check_variable_specs(params: dict, *, method: str) -> None:
    """Key check for the probabilistic `variables` specs and the dict form
    of `correlations` (a misspelt 'dist' or 'rho' was silently ignored)."""
    variables = params.get("variables")
    if isinstance(variables, dict):
        for name, spec in variables.items():
            if isinstance(spec, dict) and "law" in spec:
                check_nested_keys(spec, _LAW_KEYS, method=method,
                                  label=f"variables['{name}'] (a strength law)")
                for sub in ("a", "b"):
                    if isinstance(spec.get(sub), dict):
                        check_nested_keys(
                            spec[sub], _LAW_PAIR_KEYS, method=method,
                            label=f"variables['{name}']['{sub}']")
            else:
                check_nested_keys(spec, _VARIABLE_KEYS, method=method,
                                  label=f"variables['{name}']")
    for i, c in enumerate(params.get("correlations") or []):
        if isinstance(c, dict):
            check_nested_keys(c, _CORRELATION_KEYS, method=method,
                              label=f"correlations[{i}]")


def build_soil_layer(d, i: int, *, method: str,
                     required=("top_elevation", "bottom_elevation", "gamma"),
                     ) -> SlopeSoilLayer:
    """One SlopeSoilLayer from a layer dict: unknown keys refused (with a
    did-you-mean), required keys checked, every field the layer supports
    passed through. Shared by this adapter and the slope calc packages so
    both read a layer the same way."""
    name = d.get("name") if isinstance(d, dict) else None
    check_nested_keys(
        d, _LAYER_KEYS, method=method, hints=_LAYER_KEY_HINTS,
        notes=_LAYER_KEY_NOTES,
        label=f"soil_layers[{i}]" + (f" ('{name}')" if name else ""))
    require_keys(d, list(required), method=method, item_label="soil_layers[]")
    strength_model = d.get("strength_model", "mohr_coulomb")
    _check_choice(strength_model, _STRENGTH_MODELS,
                  name="soil_layers[].strength_model", method=method)
    bbp = d.get("bottom_boundary_points")
    tbp = d.get("top_boundary_points")
    return SlopeSoilLayer(
        name=d.get("name", f"layer_{i + 1}"), top_elevation=d["top_elevation"],
        bottom_elevation=d["bottom_elevation"], gamma=d["gamma"],
        gamma_sat=d.get("gamma_sat"), phi=d.get("phi", 0.0),
        c_prime=d.get("c_prime", 0.0), cu=d.get("cu", 0.0),
        analysis_mode=d.get("analysis_mode", "drained"),
        ru=d.get("ru", 0.0),
        bottom_boundary_points=[tuple(p) for p in bbp] if bbp else None,
        top_boundary_points=[tuple(p) for p in tbp] if tbp else None,
        strength_model=strength_model,
        shansep_S=d.get("shansep_S", 0.22),
        shansep_m=d.get("shansep_m", 0.8),
        ocr=d.get("ocr", 1.0),
        su_min=d.get("su_min", 0.0),
        hb_sigci=d.get("hb_sigci", 0.0),
        hb_gsi=d.get("hb_gsi", 50.0),
        hb_mi=d.get("hb_mi", 10.0),
        hb_D=d.get("hb_D", 0.0),
        su_active=d.get("su_active", 0.0),
        su_dss=d.get("su_dss", 0.0),
        su_passive=d.get("su_passive", 0.0),
        R_c=d.get("R_c", 0.0),
        R_phi=d.get("R_phi"),
        R_d=d.get("R_d", 0.0),
        R_psi=d.get("R_psi"),
    )


def _build_geometry(params: dict, *, method: str) -> SlopeGeometry:
    require_params(params, ["surface_points", "soil_layers"], method=method)
    soil_layers = [build_soil_layer(d, i, method=method)
                   for i, d in enumerate(params["soil_layers"])]
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    gwt_points = [tuple(pt) for pt in params["gwt_points"]] if params.get("gwt_points") else None
    surcharge_x_range = tuple(params["surcharge_x_range"]) if params.get("surcharge_x_range") else None
    crack_side = _check_choice(params.get("tension_crack_side", "crest"),
                               _CRACK_SIDES, name="tension_crack_side",
                               method=method)
    reverse_curvature = _check_choice(
        params.get("reverse_curvature", "tension_crack"), _REVERSE_CURVATURE,
        name="reverse_curvature", method=method)
    water_line_kind = _check_choice(
        params.get("water_line_kind", "piezometric"), _WATER_LINE_KINDS,
        name="water_line_kind", method=method)
    crack_model = _check_choice(params.get("tension_crack_model",
                                           "truncation"),
                                _CRACK_MODELS, name="tension_crack_model",
                                method=method)
    crack_basis = _check_choice(
        params.get("tension_crack_depth_basis", "given"), _CRACK_DEPTH_BASES,
        name="tension_crack_depth_basis", method=method)
    pore_pressure_points = (
        [tuple(p) for p in params["pore_pressure_points"]]
        if params.get("pore_pressure_points") else None)
    surcharges = None
    if params.get("surcharges"):
        surcharges = []
        for i, z in enumerate(params["surcharges"]):
            check_nested_keys(z, ("pressure", "x_start", "x_end"),
                              method=method, label=f"surcharges[{i}]")
            require_keys(z, ["pressure", "x_start", "x_end"], method=method,
                         item_label="surcharges[]")
            surcharges.append((z["pressure"], z["x_start"], z["x_end"]))

    nails = None
    if params.get("nails"):
        from slope_stability.nails import SoilNail
        nails = []
        for i, d in enumerate(params["nails"]):
            check_nested_keys(d, field_keys(SoilNail), method=method,
                              label=f"nails[{i}]")
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
                support_convention=_support_convention(d, "nails", method),
                facing_capacity=d.get("facing_capacity"),
                force_direction=_force_direction(d, "nails", method),
            ))
    anchors = None
    if params.get("anchors"):
        from slope_stability.reinforcement import Anchor
        anchors = []
        for i, d in enumerate(params["anchors"]):
            check_nested_keys(d, field_keys(Anchor), method=method,
                              label=f"anchors[{i}]")
            require_keys(d, ["x_head", "z_head", "length", "T_allow"],
                         method=method, item_label="anchors[]")
            anchors.append(Anchor(
                x_head=d["x_head"], z_head=d["z_head"], length=d["length"],
                T_allow=d["T_allow"],
                inclination=d.get("inclination", 15.0),
                support_convention=_support_convention(d, "anchors", method),
                force_direction=_force_direction(d, "anchors", method),
            ))
    geosynthetics = None
    if params.get("geosynthetics"):
        from slope_stability.reinforcement import Geosynthetic
        geosynthetics = []
        for i, d in enumerate(params["geosynthetics"]):
            check_nested_keys(d, field_keys(Geosynthetic), method=method,
                              label=f"geosynthetics[{i}]")
            require_keys(d, ["elevation", "T_allow"], method=method,
                         item_label="geosynthetics[]")
            geosynthetics.append(Geosynthetic(
                elevation=d["elevation"], T_allow=d["T_allow"],
                x_start=d.get("x_start"), x_end=d.get("x_end"),
                support_convention=_support_convention(d, "geosynthetics",
                                                       method),
                F_star=d.get("F_star"), alpha=d.get("alpha"),
                pullout_factor=d.get("pullout_factor"),
                pullout_basis=(None if d.get("pullout_basis") is None else
                               _check_choice(
                                   d["pullout_basis"], _PULLOUT_BASES,
                                   name="geosynthetics[].pullout_basis",
                                   method=method)),
                development_length=d.get("development_length"),
                anchored_end=(None if d.get("anchored_end") is None else
                              _check_choice(
                                  d["anchored_end"], ["start", "end"],
                                  name="geosynthetics[].anchored_end",
                                  method=method)),
                coverage_ratio=d.get("coverage_ratio", 1.0),
                force_direction=_force_direction(d, "geosynthetics", method),
            ))

    stabilizing_piles = None
    if params.get("stabilizing_piles"):
        from slope_stability.reinforcement import StabilizingPile
        stabilizing_piles = []
        for i, d in enumerate(params["stabilizing_piles"]):
            check_nested_keys(d, field_keys(StabilizingPile), method=method,
                              label=f"stabilizing_piles[{i}]")
            require_keys(d, ["x"], method=method, item_label="stabilizing_piles[]")
            _support_convention(d, "stabilizing_piles", method)
            stabilizing_piles.append(StabilizingPile(
                x=d["x"], shear_capacity=d.get("shear_capacity"),
                spacing=d.get("spacing", 1.0), z_head=d.get("z_head"),
                z_toe=d.get("z_toe"), ito_matsui=d.get("ito_matsui", False),
                diameter=d.get("diameter"), c=d.get("c"), phi=d.get("phi"),
                gamma=d.get("gamma"),
                force_direction=_check_choice(
                    d.get("force_direction", "horizontal"),
                    _PILE_FORCE_DIRECTIONS,
                    name="stabilizing_piles[].force_direction",
                    method=method),
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
        tension_crack_depth_basis=crack_basis,
        reverse_curvature=reverse_curvature,
        pore_pressure_points=pore_pressure_points, surcharges=surcharges,
        water_line_kind=water_line_kind,
        assisting_support=_check_choice(
            params.get("assisting_support", "slack"), _ASSISTING_SUPPORT,
            name="assisting_support", method=method),
    )


def _with_water_statements(out: dict, geom) -> dict:
    """``out`` with what the pore pressure on the slice bases was taken as
    (GM-140 step 5a; ``slope_stability.pore_pressure.water_statements``),
    for the results that do not carry it themselves (the yield
    acceleration, FOSM and Monte Carlo): the reading of a sloping water
    line, ru in a layered column and total-stress layers in ``choices``,
    and their warnings in ``warnings``."""
    from slope_stability.pore_pressure import water_statements
    choices, notes = water_statements(geom)
    if choices:
        out["choices"] = list(out.get("choices") or []) + choices
    if notes:
        out["warnings"] = list(out.get("warnings") or []) + notes
    return out


def _slip_surface_from(params: dict):
    if params.get("slip_points") is not None:
        from slope_stability.slip_surface import PolylineSlipSurface
        return PolylineSlipSurface(points=[tuple(pt) for pt in params["slip_points"]])
    a, b = params.get("arc_x_start"), params.get("arc_x_end")
    if a is not None or b is not None:
        # the two ends of the arc the circle stands for (GM-343)
        if a is None or b is None or any(
                params.get(k) is None for k in ("xc", "yc", "radius")):
            raise ValueError(
                "arc_x_start and arc_x_end name one arc of a circle: give "
                "both, with xc, yc and radius.")
        from slope_stability.slip_surface import CircularSlipSurface
        return CircularSlipSurface(params["xc"], params["yc"],
                                   params["radius"], a, b)
    return None


def _run_analyze_slope(params: dict) -> dict:
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + (
            "method", "f_interslice", "n_slices", "tol",
            "include_slice_data", "compare_methods", "m_alpha_rule",
            "negative_friction_term"),
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
        m_alpha_rule=_check_choice(
            params.get("m_alpha_rule", "frictional"), M_ALPHA_RULES,
            name="m_alpha_rule", method="analyze_slope"),
        negative_friction_term=_check_choice(
            params.get("negative_friction_term", "as_printed"),
            NEGATIVE_TERM_RULES, name="negative_friction_term",
            method="analyze_slope"),
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
            "tol", "seed", "time_budget_s", "bishop_screen", "m_alpha_rule",
            "low_m_alpha", "negative_friction_term"),
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
    low_m_alpha = _check_choice(
        params.get("low_m_alpha", "flag"), LOW_M_ALPHA_ACTIONS,
        name="low_m_alpha", method="search_critical_surface")
    result = search_critical_surface(
        geom=geom, x_range=x_range, y_range=y_range,
        nx=params.get("nx", 10), ny=params.get("ny", 10),
        method=method,
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        surface_type=surface_type,
        x_entry_range=x_entry_range, x_exit_range=x_exit_range,
        n_trials=params.get("n_trials", 500), n_points=params.get("n_points", 5),
        seed=params.get("seed"),
        time_budget_s=_time_budget(params, "search_critical_surface"),
        bishop_screen=bool(params.get("bishop_screen", False)),
        m_alpha_rule=_check_choice(
            params.get("m_alpha_rule", "frictional"), M_ALPHA_RULES,
            name="m_alpha_rule", method="search_critical_surface"),
        low_m_alpha=low_m_alpha,
        negative_friction_term=_check_choice(
            params.get("negative_friction_term", "as_printed"),
            NEGATIVE_TERM_RULES, name="negative_friction_term",
            method="search_critical_surface"),
    )
    budget_is_default(result.choices, params)
    out = result.to_dict()
    # the headline, its flags, the lowest surface with no flag and the
    # option that refuses flagged surfaces, as fields and in one sentence
    # (GM-140 step 3; its review's ruling 5)
    out.update(headline_report(result, low_m_alpha=low_m_alpha))
    return out


def _run_fosm(params: dict) -> dict:
    from slope_stability.probabilistic import fosm_fos
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + ("variables", "method", "n_slices",
                                          "tol", "correlations"),
        method="fosm_fos")
    geom = _build_geometry(params, method="fosm_fos")
    require_params(params, ["variables"], method="fosm_fos")
    check_variable_specs(params, method="fosm_fos")
    method = _check_choice(params.get("method", "bishop"), _METHODS,
                           name="method", method="fosm_fos")
    result = fosm_fos(
        geom, params["variables"],
        xc=params.get("xc"), yc=params.get("yc"), radius=params.get("radius"),
        slip_surface=_slip_surface_from(params),
        method=method, n_slices=params.get("n_slices", 30),
        tol=params.get("tol", 1e-4),
        correlations=params.get("correlations"),
    )
    return _with_water_statements(result.to_dict(), geom)


def _run_monte_carlo(params: dict) -> dict:
    from slope_stability.probabilistic import monte_carlo_fos
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + (
            "variables", "method", "n", "seed", "n_slices", "tol",
            "research_surface", "correlations", "time_budget_s"),
        method="monte_carlo_fos")
    geom = _build_geometry(params, method="monte_carlo_fos")
    require_params(params, ["variables"], method="monte_carlo_fos")
    check_variable_specs(params, method="monte_carlo_fos")
    method = _check_choice(params.get("method", "bishop"), _METHODS,
                           name="method", method="monte_carlo_fos")
    result = monte_carlo_fos(
        geom, params["variables"],
        xc=params.get("xc"), yc=params.get("yc"), radius=params.get("radius"),
        slip_surface=_slip_surface_from(params),
        method=method, n=params.get("n", 1000), seed=params.get("seed"),
        n_slices=params.get("n_slices", 30), tol=params.get("tol", 1e-4),
        research_surface=params.get("research_surface", False),
        search_kwargs=({"time_budget_s": _time_budget(params,
                                                      "monte_carlo_fos")}
                       if params.get("research_surface") else None),
        correlations=params.get("correlations"),
    )
    return _with_water_statements(result.to_dict(), geom)


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


_RD_METHODS = ["corps_2stage", "duncan_3stage", "lowe_karafiath"]
_RD_SURFACE_TYPES = ["circular", "entry_exit", "noncircular", "noncircular_de"]


def _run_rapid_drawdown(params: dict) -> dict:
    from slope_stability.analysis import rapid_drawdown_fos
    reject_unknown_params(
        params,
        _GEOM_PARAMS + _SURFACE_PARAMS + (
            "drawdown_from_elevation", "drawdown_to_elevation", "method",
            "f_interslice", "n_slices", "tol", "stage1_phreatic_points",
            "drawdown_phreatic_points"),
        method="rapid_drawdown_fos")
    geom = _build_geometry(params, method="rapid_drawdown_fos")
    require_params(params, ["drawdown_from_elevation", "drawdown_to_elevation"],
                   method="rapid_drawdown_fos")
    method = _check_choice(params.get("method", "duncan_3stage"), _RD_METHODS,
                           name="method", method="rapid_drawdown_fos")
    f_int = _check_choice(params.get("f_interslice", "constant"), _F_INTERSLICE,
                          name="f_interslice", method="rapid_drawdown_fos")
    result = rapid_drawdown_fos(
        geom, params["drawdown_from_elevation"], params["drawdown_to_elevation"],
        xc=params.get("xc"), yc=params.get("yc"), radius=params.get("radius"),
        slip_surface=_slip_surface_from(params),
        method=method, f_interslice=f_int,
        n_slices=params.get("n_slices", 50), tol=params.get("tol", 1e-4),
        stage1_phreatic_points=params.get("stage1_phreatic_points"),
        drawdown_phreatic_points=params.get("drawdown_phreatic_points"),
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
            "drawdown_phreatic_points", "time_budget_s", "low_m_alpha"),
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
        drawdown_phreatic_points=params.get("drawdown_phreatic_points"),
        time_budget_s=_time_budget(params, "search_rapid_drawdown"),
        low_m_alpha=_check_choice(
            params.get("low_m_alpha", "flag"), LOW_M_ALPHA_ACTIONS,
            name="low_m_alpha", method="search_rapid_drawdown"),
    )
    if result.search is not None:
        budget_is_default(result.search.choices, params)
    return result.to_dict()


def _run_yield_acceleration(params: dict) -> dict:
    from slope_stability.newmark import yield_acceleration
    params = without_unused(params, YIELD_REFUSED)
    refuse_given(params, YIELD_REFUSED, method="yield_acceleration")
    reject_unknown_params(
        params,
        tuple(k for k in _GEOM_PARAMS if k not in YIELD_REFUSED)
        + _SURFACE_PARAMS + ("method", "n_slices", "tol", "kh_max"),
        method="yield_acceleration")
    geom = _build_geometry(params, method="yield_acceleration")
    method = _check_choice(params.get("method", "spencer"), _METHODS,
                           name="method", method="yield_acceleration")
    result = yield_acceleration(
        geom, xc=params.get("xc"), yc=params.get("yc"),
        radius=params.get("radius"), slip_surface=_slip_surface_from(params),
        method=method, n_slices=params.get("n_slices", 50),
        tol=params.get("tol", 1e-4), kh_max=params.get("kh_max", 2.0))
    return _with_water_statements(result.to_dict(), geom)


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


def _run_bray_travasarou_2007(params: dict) -> dict:
    from slope_stability.newmark import bray_travasarou_2007
    reject_unknown_params(
        params, ("ky", "ts", "sa_1p5ts", "magnitude", "rigid"),
        method="bray_travasarou_2007")
    require_params(params, ["ky", "ts", "sa_1p5ts", "magnitude"],
                   method="bray_travasarou_2007")
    result = bray_travasarou_2007(
        ky=params["ky"], ts=params["ts"], sa_1p5ts=params["sa_1p5ts"],
        magnitude=params["magnitude"], rigid=params.get("rigid"))
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
    "bray_travasarou_2007": _run_bray_travasarou_2007,
    "fosm_fos": _run_fosm,
    "monte_carlo_fos": _run_monte_carlo,
}

_GEOMETRY_PARAMS = {
    "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,y], ...] array (x increasing)."},
    "soil_layers": {"type": "array", "required": True, "description": "Array of soil-layer dicts: {top_elevation, bottom_elevation, gamma (all required); name, gamma_sat, phi, c_prime, cu, analysis_mode ('drained'|'undrained'), ru optional; bottom_boundary_points and top_boundary_points ([[x,z],...], sorted by x) optional, a sloping bottom or top in place of the flat elevation}. Per-layer strength_model: 'mohr_coulomb' (default), 'shansep' (su = shansep_S * ocr^shansep_m * sigma'_v; fields shansep_S, shansep_m, ocr, su_min) 'hoek_brown' (Generalized Hoek-Brown; fields hb_sigci kPa, hb_gsi, hb_mi, hb_D) or 'anisotropic' (undrained su varies with the slice-base inclination alpha; fields su_active [kPa, ACTIVE/crest zone, alpha>=+45deg], su_dss [direct simple shear, alpha=0], su_passive [PASSIVE/toe zone, alpha<=-45deg] — su_active>=su_dss>=su_passive for K>1 clays; Casagrande-Carrillo sin(2alpha) interpolation; standard toe-at-low-x slope). For rapid_drawdown_fos and search_rapid_drawdown, a layer that does not drain freely also takes the total-stress R envelope R_c (kPa) and R_phi (deg), from which the strength of soil consolidated at K_c = 1 is derived by EM 1110-2-1902 Eqs. G-5 and G-6, or that envelope as measured, R_d (kPa) and R_psi (deg): tau_ff = R_d + sigma'_c tan(R_psi) (not for 'corps_2stage', which needs R_c and R_phi); a layer with neither R_phi nor R_psi is free-draining. One material per point: where two layers cover the same ground the one listed FIRST holds it, for weight and strength (reported in layer_overlaps); ground above the deepest layer's bottom that no layer covers is refused."},
    "gwt_points": {"type": "array", "required": False, "description": "Water line [[x,z],...]: by default a piezometric line (the pore pressure at a slice base is gamma_w x the full vertical depth below it; see water_line_kind). If above the ground surface, ponded water is auto-detected (water weight + horizontal hydrostatic thrust applied as external loads). No pore pressure is subtracted on the base of a slice in a total-stress layer (analysis_mode 'undrained', 'shansep', 'anisotropic'; EM 1110-2-1902 2-2c)."},
    "water_line_kind": {"type": "str", "required": False, "default": "piezometric", "allowed_values": _WATER_LINE_KINDS, "description": "What gwt_points is. 'piezometric' (default): a piezometric line (measured heads); u = gamma_w x the full vertical depth below it, never corrected. 'water_table': a water table with seepage parallel to it; u = h_w gamma_w cos^2(alpha_w), alpha_w the slope of the line above each slice base (FHWA NHI-06-088 Eq. 6-16, p. 6-14; UFC 3-220-10 Sec. 7-5.1), not applied where the line is above the ground (ponded water). The two differ only under a sloping line; the uncorrected head gives the lower factor of safety. The result's choices names the reading used."},
    "kh": {"type": "float", "required": False, "default": 0.0, "description": "Horizontal pseudo-static seismic coefficient (acts on soil weight only)."},
    "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Vertical surcharge (kPa). A single loaded zone; use 'surcharges' for several distinct loaded areas."},
    "surcharge_x_range": {"type": "array", "required": False, "description": "[x_start, x_end] extent of the single 'surcharge'."},
    "surcharges": {"type": "array", "required": False, "description": "Several distinct surcharge zones (bench + crest loads, etc.): [{pressure kPa, x_start, x_end}, ...]. Each zone's pressure is summed at any x it covers, ON TOP OF the single 'surcharge'/'surcharge_x_range'. Represent a linearly-varying (trapezoidal) load as its mean uniform pressure or as several thin zones."},
    "tension_crack_depth": {"type": "float", "required": False, "default": 0.0, "description": "Given depth of a tension crack at the upper end of the slip surface (m); 0 = no crack. Leave at 0 when tension_crack_depth_basis is 'rankine' or 'min_fos'."},
    "tension_crack_depth_basis": {"type": "str", "required": False, "default": "given", "allowed_values": _CRACK_DEPTH_BASES, "description": "How the crack's depth is set. 'given' (default): tension_crack_depth. 'rankine': the depth over which Rankine's active pressures are negative with the developed strengths, d = 2 c_D / (gamma tan(45 - phi_D/2)), c_D = c/F, tan(phi_D) = tan(phi)/F (EM 1110-2-1902 Eq. C-36, p. C-36; Eqs. 2-5, 2-6), F the surface's own factor of safety, the solution repeated until the depth stops changing; printed for total stresses and one soil, so it stops at the bottom of the soil at the crack, and the scan is used instead where that soil is drained with water in the cracked depth or has no c and phi. 'min_fos': the depth of minimum factor of safety from a scan of depths (EM 2-2f(4), p. 2-6; C-10c(1)): the first minimum going down in 20 equal steps to the lowest point of the surface's upper half; a surface whose factor of safety is still falling at the last depth gets no number."},
    "tension_crack_water_depth": {"type": "float", "required": False, "default": 0.0, "description": "Water depth in the tension crack (m); a depth above the crack's own fills it. Its thrust, 0.5 gamma_w z_w^2 at a third of its depth, acts on the vertical boundary at the crack."},
    "tension_crack_side": {"type": "str", "required": False, "default": "crest", "allowed_values": _CRACK_SIDES, "description": "Which end of the slip surface the crack is at: 'crest' (default) the end where the ground is higher (EM 1110-2-1902 2-2f(3), p. 2-6: near the crest of the slope), whichever way the slope faces; 'entry' (low x) or 'exit' (high x) gives the end."},
    "tension_crack_model": {"type": "str", "required": False, "default": "truncation", "allowed_values": _CRACK_MODELS, "description": "What the crack does: 'truncation' (default, as printed) ends the slip surface at the bottom of the crack, leaves out the soil upslope of it and ends the slices at a vertical boundary there (EM 1110-2-1902 2-2f(3), p. 2-6; C-10c, p. C-36; Slide v6 Tutorial 16); 'strength' keeps the cracked soil as driving weight without strength (the default before GM-140; no printed source; a lower factor of safety)."},
    "reverse_curvature": {"type": "str", "required": False, "default": "tension_crack", "allowed_values": _REVERSE_CURVATURE, "description": "A circle whose centre is below the ground at its upper end would turn back over itself (reverse curvature). The tension crack is a condition of the analysis, set by the engineer (tension_crack_depth, or tension_crack_depth_basis 'rankine' / 'min_fos'), and the same on every trial surface: a circle's shape never switches it on. 'tension_crack' (default): with a crack asked for, such a circle is closed by that crack where its overhang lies within the crack's depth and discarded where it runs deeper; with no crack asked for, every such circle is discarded (not an admissible surface; naming a crack admits those within its depth). 'discard': every such circle is discarded. A search counts the surfaces the crack rule refuses by reason and locates them (crack_refused); a discarded circle given to a single-surface method gets no factor of safety (an error saying why). A circle overhanging at its lower end is always discarded."},
    "pore_pressure_points": {"type": "array", "required": False, "description": "Discrete pore-pressure field as [[x, z, u], ...] triples (u in kPa) -- a flow-net / TIN sampling. When given, the base pore pressure at each slice is interpolated from this field (linear on the Delaunay triangulation, nearest-node fallback outside the hull, suction clamped to 0), OVERRIDING the gwt_points piezometric line and per-layer ru. The ponded-water buttress still comes from gwt_points, so set BOTH for a reservoir over a flow-net field."},
    "nails": {"type": "array", "required": False, "description": "Soil nails (per metre of slope run): [{x_head, z_head, length required; inclination deg below horizontal=15, bar_diameter mm=25, drill_hole_diameter mm=150, fy MPa=420, bond_stress kPa=100, spacing_h m=1.5, support_convention, facing_capacity kN per nail, force_direction}]. Load at the slip surface = the least of the pullout of the length behind the surface, the bar and, when facing_capacity is given, the facing with the pullout of the length in front of the surface, / spacing_h (FHWA GEC 7 Eq. 5.2, p. 105). With no facing_capacity that front term is not checked and the result says so. The bar, the bond and the facing are taken in the measure of support_convention (allowable when 'active', ultimate when 'passive')."},
    "anchors": {"type": "array", "required": False, "description": "Tieback anchors: [{x_head, z_head, length, T_allow kN/m required; inclination=15, support_convention, force_direction}]. Full T_allow applied when the bond zone crosses the slip surface."},
    "geosynthetics": {"type": "array", "required": False, "description": "Horizontal geosynthetic layers: [{elevation, T_allow kN/m required; x_start, x_end, support_convention, F_star, alpha, pullout_factor, pullout_basis, development_length, anchored_end, coverage_ratio, force_direction optional}]. Every support (nails, anchors, geosynthetics, piles) is a known force where it crosses the slip surface, acting on the slice whose base it crosses (on a slice boundary: the slice holding the part of the support inside the sliding mass). support_convention: 'active' (default; Duncan, Wright and Brandon 2014 Method A, Eqs. 8.1 and 8.2: the force is not divided by F and reduces the driving side -- give the ALLOWABLE force) or 'passive' (Method B, Eqs. 8.3 and 8.4: divided by F with the soil strength -- give the ULTIMATE force; lower FOS for the same force). Bishop's simplified method takes the forces by Eq. 6.78 (the moment, and the vertical component in the slice's vertical equilibrium); the Ordinary Method of Slices gives no number where a support crosses the surface (no reference prints it with known forces). force_direction: 'along_reinforcement' (default: the support's own orientation) or 'along_slip_surface' (Sec. 8.6). Pullout: the force a layer carries at a crossing is the lesser of T_allow and its pullout resistance. Give F_star and alpha (the pullout resistance factor and the scale effect factor of the fill and the product; no default) and, for an active layer, pullout_factor (what the pullout is divided by; no default: the book's Table 8.4 prints 1.3, 1.5 and 2.0): pullout_basis 'dwb2014' (default with F_star and alpha) takes 2 alpha F* gamma z integrated from each end of the layer (the book's Eqs. 8.8 and 8.9; ends x_start / x_end, else where the layer meets the ground or the model's edge); 'gec11' takes it behind the surface only, on the effective stress, times coverage_ratio, and counts nothing with less than 1 m behind the surface (GEC 11 Eq. 9-9 and p. 9-14; its factor is 1.5 for granular and 2 for cohesive soils, p. 9-5). Or give development_length (m): the force rises linearly from zero at an end to T_allow over that length. anchored_end 'start' (the -x end) or 'end': an end fixed to a facing, no pullout measured from it. With none of these the pullout is NOT checked (the full T_allow at every crossing) and the result says so. All methods."},
    "stabilizing_piles": {"type": "array", "required": False, "description": "Single-row stabilizing/micro piles crossing the slope (Ito & Matsui 1975 or a specified shear): [{x required; spacing m=1.0 (D1); z_head, z_toe optional}]. Set the resistance EITHER by shear_capacity (kN per pile; per-metre force = shear_capacity/spacing) OR ito_matsui=true with diameter (m; D2=spacing-diameter) and optional c/phi/gamma (default: soil layer) giving the Ito-Matsui plastic lateral force integrated from head to slip surface. force_direction 'horizontal' (default: the pile's shear, against the sliding) or 'along_slip_surface' (parallel to the slip surface at the crossing); 'normal' is no longer offered (no reference prints it). support_convention 'active' (default) or 'passive' as for geosynthetics. Stabilizing force at the slip crossing."},
    "assisting_support": {"type": "str", "required": False, "default": "slack", "allowed_values": _ASSISTING_SUPPORT, "description": "A support whose pull would assist the sliding (its force's component along the slip surface where it crosses points down the slope: a steep nail crossing the steep upper part of a surface, or the toe-side crossing of a layer the surface crosses twice). 'slack' (default): it counts nothing on that surface and the result names it (supports_not_counted). 'as_given': its force is applied as it stands. The owner's call (OWN-59); no reference prints one."},
    "reinforcement_force": {"type": "float", "required": False, "default": 0.0, "description": "Single equivalent horizontal reinforcement force (kN/m), an allowable force (Method A) acting at reinforcement_elevation, which must be given with it (a force with no elevation is refused). Simpler alternative to nails/anchors; its pullout is not checked."},
    "reinforcement_elevation": {"type": "float", "required": False, "description": "Elevation of the equivalent reinforcement force (m). Required with reinforcement_force."},
}

_SURFACE_SPEC_PARAMS = {
    "xc": {"type": "float", "required": False, "description": "Circle center x (m). Required for circular."},
    "yc": {"type": "float", "required": False, "description": "Circle center y (m). Required for circular."},
    "radius": {"type": "float", "required": False, "description": "Circle radius (m). Required for circular. A circle that meets the ground at more than two points is split there: each arc below the ground is analysed as a slip surface of its own and the result is the lowest, named in arc (the module's rule; no reference prints one). An arc with no net driving moment has no factor of safety."},
    "arc_x_start": {"type": "float", "required": False, "description": "With arc_x_end: the x (m) of the two ends of the one arc of the circle to analyse, as a result's arc.x_start and arc.x_end give them (use them to analyse a search's governing surface again). Each must be a point where that arc meets the ground, or a ground corner it passes under by less than 1 % of its greatest depth (a toe circle written to a few decimals). Left out, a circle with one arc below the ground is that arc, and one with several is analysed arc by arc."},
    "arc_x_end": {"type": "float", "required": False, "description": "The arc's higher-x end (m); see arc_x_start."},
    "slip_points": {"type": "array", "required": False, "description": "Polyline slip surface [[x,y],...] for noncircular. Use instead of xc/yc/radius."},
}

_VARIABLES_PARAM = {
    "variables": {"type": "object", "required": True, "description": "Random variables: {'phi': {cov: 0.1}, 'cu:Clay': {mean: 30, std: 5, dist: 'lognormal'}, ...}. Keys are layer parameters (phi, c_prime, cu, gamma, gamma_sat, ru) optionally scoped ':LayerName'; each spec needs cov or std (mean defaults to the layer value); dist is 'normal' (default) or 'lognormal'. Use gamma_sat (not dry gamma) for the unit weight of a submerged slope. A depth-varying undrained-strength law su(z)=a+b*(datum_z-z) is ONE correlated (a,b) variable applied coherently across layers — give an entry with a 'law':'linear_su' key: {'a':{mean,std|cov}, 'b':{mean,std|cov}, 'rho_ab':0..1, 'datum_z': elevation where su=a, 'z_ref':'mid'|'top'|'bottom', 'su_min': floor, 'layers': names|null}. A std-0 component (e.g. fixed intercept) drops out."},
}

_CORRELATIONS_PARAM = {
    "correlations": {"type": "array", "required": False, "description": "Optional correlated scalar pairs: [['c_prime','phi',-0.5], ...] (or [{'var1','var2','rho'}]). Each references two SCALAR variable keys from 'variables' (optionally scoped ':LayerName') and a correlation rho in [-1,1]. FOSM adds the Taylor cross-term 2*rho*(dF1/2)*(dF2/2); Monte Carlo draws the pair from a bivariate normal. Omit for independent variables (default). A c'-phi' negative correlation typically lowers the FOS variance."},
}


# rapid_drawdown_fos's doc must fit the deep agent's reference budget (16,000
# characters; test_wave1_geotech): these shared inputs point to analyze_slope's
# full text instead of repeating it there.
_SEE_ANALYZE_SLOPE = {
    _k: {**_GEOMETRY_PARAMS[_k], "description": (
        "As in analyze_slope (describe_method analyze_slope gives the full "
        "text): " + _s)}
    for _k, _s in (
        ("reverse_curvature", "how a circle that turns back over itself is "
         "treated; a tension crack is a condition of the analysis, the same "
         "on every trial surface."),
        ("tension_crack_depth_basis", "'given' (tension_crack_depth), "
         "'rankine' (EM 1110-2-1902 Eq. C-36, p. C-36) or 'min_fos'."),
        ("tension_crack_model", "'truncation' (default, as printed: EM "
         "1110-2-1902 2-2f(3), p. 2-6; C-10c, p. C-36) or 'strength'."),
        ("geosynthetics", "horizontal layers [{elevation, T_allow kN/m; "
         "x_start, x_end, support_convention optional}], each acting at the "
         "base of the slice it crosses; its pullout is not checked here "
         "(see analyze_slope's geosynthetics for the pullout inputs)."),
        ("stabilizing_piles", "a single row [{x; spacing; z_head, z_toe "
         "optional}], resisting by shear_capacity or by ito_matsui=true with "
         "diameter."),
    )
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
            "m_alpha_rule": {"type": "str", "required": False, "default": "frictional", "allowed_values": ["frictional", "every_slice"], "description": "Which slices the m_alpha < 0.2 check looks at (Whitman and Bailey 1967, p. 487). 'frictional': slices with phi' > 0; on a phi' = 0 slice m_alpha = cos(a) does not depend on F and the strength does not depend on the normal force (the module's own reasoning, no printed source). 'every_slice': every slice."},
            "negative_friction_term": {"type": "str", "required": False, "default": "as_printed", "allowed_values": list(NEGATIVE_TERM_RULES), "description": "Bishop's simplified method and the Ordinary Method of Slices on a slice where the pore-water force on the base exceeds the slice's weight. 'as_printed' (default): the frictional term W - u b (Bishop 1955 Eq. 13, p. 10) or W cos(alpha) - u l (Eq. 9) is used as printed, negative. 'zero': it is set to zero there, the module's rule before GM-140 step 3, with no printed source. Either way the result names the slices (acceptability, check 'negative_frictional_term'), as Whitman and Bailey (1967, p. 487) ask."},
        },
        "returns": {"arc": "Present when the circle meets the ground at more than two points, or its arc's ends were given: which arc of the circle this result is ({x_start, x_end, index, n_arcs, arcs, description}) and, where the arcs were each analysed, governs, others (each other arc with its value, or not_scored and the reason) and not_scored (the count by reason: no_driving, not_converged, refused). Tell the user which arc the result is.", "FOS": "Factor of safety: the method's own converged solution on this surface. Null when the method has none there (see method_solved and warnings): nothing stands in for the method. One exception, stated in method_used: with method 'bishop' on a circle with no supports and no seismic coefficient, where Bishop's value is below the Ordinary Method of Slices value for the same circle, FOS is the Ordinary Method of Slices value (Duncan 1996, p. 580); FOS_bishop then holds Bishop's own value and FOS_fellenius the Ordinary Method of Slices value.", "flags": "The printed acceptability checks this surface does not meet (present when acceptability is): low_m_alpha, bishop_below_oms, negative_frictional_term, base_tension, side_tension, thrust_outside. The surface is analysed whatever they find; report the flags to the user with the factor of safety.", "acceptability": "The findings of the printed acceptability checks (present when there are any), each {check, flag, source, text, and the slices or slice sides with their values}: m_alpha below 0.2 on a slice (Whitman and Bailey 1967, p. 487); Bishop's value against the Ordinary Method of Slices value for the circle (Duncan 1996, p. 580; with supports or a seismic coefficient the comparison is not made and the finding says why); a negative frictional term (Bishop 1955 Eqs. 9 and 13); and for spencer / morgenstern_price / gle a negative effective normal force on a slice base, a tensile interslice force on a slice side, and the line of thrust outside the sliding mass (EM 1110-2-1902 C-10a, pp. C-33 to C-34).", "method": "Method requested.", "method_solved": "False when the method has no solution on this surface: Spencer/Morgenstern-Price/GLE find no lambda at which the moment and force factors of safety agree, Bishop's or Janbu's iteration does not converge, or the surface has no driving moment or force. FOS is then null and warnings says why. Report that there is no factor of safety by this method on this surface; do not give another method's value as its own.", "method_used": "What produced FOS: the method itself; the Ordinary Method of Slices value where Bishop's was below it (see FOS); or null when the method has no solution.", "methods_without_solution": "Methods (the one asked for, or compare_methods values) with no solution here; their FOS is null.", "warnings": "What the engineer must check on this surface: no equilibrium solution, the tension crack it was solved with (a reverse-curvature circle closed by one says so), supports Bishop/OMS count little of (with Spencer's FOS on the circle), and each printed acceptability check not met (with EM 1110-2-1902 C-10b's remedies for a steep exit; none is applied). Report these to the user.", "tension_crack": "The crack the surface was solved with (present when there is one): depth_m and depth_basis (given / rankine / min_fos), side, model, where it is (x_m, where the surface reaches the crack's bottom and the slices end; ground_z_m the ground there; base_z_m the crack's bottom), water_depth_m; for a Rankine depth the F it was taken at and whether it stopped at the bottom of its soil; for a scanned depth the depths tried; reverse_curvature and overhang_depth_m when it closes a reverse-curvature circle.", "lambda_mp": "Interslice scaling (M-P/GLE).", "theta_spencer_deg": "Spencer interslice angle.", "FOS_janbu_corrected": "Janbu f0-corrected FOS.", "slice_data": "Per-slice force table (include_slice_data=true).", "thrust_line": "Line of thrust [[x,z],...] (rigorous methods + include_slice_data), clipped to the sliding mass for display; the acceptability finding 'thrust_outside' says where the solved line left the mass.", "layer_overlaps": "Where a layer loses ground to one listed before it (present when any does): layer, held_by, x_from, x_to, max_overlap_m, statement."},
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
        "returns": {"rows": "[{method, FOS, detail}, ...] per method (a method with no solution on the surface has no row and no number; methods_without_solution names it and warnings says why).", "methods_without_solution": "Methods with no solution on this surface (present when there are any).", "surface": "Surface description.", "summary": "Formatted text table.", "warnings": "What the engineer must check on this surface (present when there is any). Report these to the user.", "layer_overlaps": "Where a layer loses ground to one listed before it (present when any does): layer, held_by, x_from, x_to, max_overlap_m, statement."},
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
        "brief": "Rapid drawdown on a SPECIFIED slip surface, as EM 1110-2-1902 Appendix G prints it: the improved three-stage procedure (Duncan, Wright and Wong), Lowe and Karafiath's two-stage procedure, or the Corps of Engineers' 1970 two-stage procedure. Layers that do not drain freely respond undrained after a fast drawdown; give them the R envelope (soil_layers[].R_c, R_phi) or the measured R_d, R_psi beside c_prime and phi (neither => free-draining). Stage 1: effective stresses before drawdown give sigma'_c and tau_c on each slice base (Eqs. G-1, G-2). Stage 2: solved after drawdown with the undrained strength tau_ff as cohesion and phi = 0 (Eq. G-12 between the K_c = 1 strength of Eqs. G-5, G-6, G-9 and the effective-stress strength, K_c by Eq. G-11; the Corps procedure uses the lower of the R envelope as entered and the effective-stress envelope). Three-stage only: the drained strength from the second stage's normal force (Eqs. G-13 to G-15) is compared slice by slice, and a third stage is solved only if a slice's drained strength is lower. Every stage's factor of safety is returned; FOS is that of the last stage computed, and null where a stage has no solution. Every stage is solved by Spencer's method (the GLE engine).",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            **_SEE_ANALYZE_SLOPE,
            "drawdown_from_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation BEFORE drawdown (m)."},
            "drawdown_to_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation AFTER drawdown (m); must be below drawdown_from_elevation."},
            "method": {"type": "str", "required": False, "default": "duncan_3stage", "allowed_values": _RD_METHODS, "description": "'duncan_3stage' (the improved procedure, EM G-3: the answer is the third stage's factor of safety, or the second's where no slice's drained strength is lower), 'lowe_karafiath' (the same first and second stages without the drained check: the answer is the second stage's, so never below duncan_3stage), or 'corps_2stage' (the Corps 1970 procedure, EM G-2: the lower of the R envelope with R_c and R_phi as entered and the effective-stress envelope; no third stage). The manual's own example of the 1970 procedure was computed by the Modified Swedish Method at a fixed side-force inclination (3.49, then 1.35); Spencer's method, used here, gives 1.24 on it, and the result says so."},
            "f_interslice": {"type": "str", "required": False, "default": "constant", "allowed_values": _F_INTERSLICE, "description": "GLE interslice function ('constant' = Spencer)."},
            "n_slices": {"type": "int", "required": False, "default": 50, "description": "Number of slices."},
            "stage1_phreatic_points": {"type": "array", "required": False, "description": "The seepage (phreatic) line BEFORE drawdown, [[x,z],...], for the first stage's pore pressures (EM 1110-2-1902 Table 2-1: steady seepage). Omitted: a level pool at drawdown_from_elevation with hydrostatic pressures below it, the assumption of the manual's own example (G-4b, p. G-9), which the result names as assumed. The line should start at the pool level on the upstream face so the pool still loads the face."},
            "drawdown_phreatic_points": {"type": "array", "required": False, "description": "The piezometric line AFTER drawdown, [[x,z],...]: the pore pressures on free-draining slices in the second stage, in the drained check (Eq. G-14) and on the drained slices of the third stage. Omitted: a level pool at drawdown_to_elevation, as in the manual's example, which the result names as assumed."},
        },
        "returns": {"arc": "Present when the circle meets the ground at more than two points, or its arc's ends were given: which arc of the circle this result is ({x_start, x_end, index, n_arcs, arcs, description}) and, where the arcs were each analysed, governs, others (each other arc with its value, or not_scored and the reason) and not_scored (the count by reason: no_driving, not_converged, refused). Tell the user which arc the result is.", "FOS": "Drawdown factor of safety: that of the last stage computed (answer_stage); null where a stage has no solution (no_solution says which and why).", "answer_stage": "2 or 3: the stage FOS is from.", "stage1_FOS": "First stage (before drawdown).", "stage2_FOS": "Second stage (after drawdown, undrained strengths).", "stage3_FOS": "Third stage; null when not computed (two-stage procedures, or no slice's drained strength lower).", "stage_method": "The method of slices that solved every stage.", "water_before": "The water before drawdown: the line given, or the level pool assumed.", "water_after": "The water after drawdown: the line given, or the level lowered pool assumed.", "n_undrained_slices": "Slices that do not drain freely.", "m_alpha_min_stage1": "Least m_alpha of the first stage at its own FOS (a warning says so below 0.2).", "m_alpha_min": "Least m_alpha of the last stage computed at its FOS.", "n_drained_substituted": "Slices on c', phi' and the after-drawdown pore pressure in the third stage.", "slices": "Per slice, low x to high x, the columns of EM Figs. G-10b and G-11b (kPa; N in kN/m): x_left, x_right, sigma_c, tau_c, Kc, Kf, tau_ff_Kc1, tau_ff_Kcf, tau_ff, N_stage2, u_after, sigma_d, s_d, strength_used; 'columns' explains each.", "choices": "The procedure, the method of slices, the strength rules and the water assumed, each with its printed basis.", "layer_overlaps": "Where a layer loses ground to one listed before it (present when any does): layer, held_by, x_from, x_to, max_overlap_m, statement."},
    },
    "search_rapid_drawdown": {
        "category": "Slope Stability",
        "brief": "SEARCH for the critical (minimum-FOS) slip surface under RAPID DRAWDOWN (Corps 1970 two-stage / Lowe and Karafiath two-stage / improved three-stage, EM 1110-2-1902 Appendix G). Like search_critical_surface, but every trial is scored by the same stage-by-stage procedure as rapid_drawdown_fos, and a trial on which a stage has no solution is not scored; supports circular and noncircular searches. Use this (not rapid_drawdown_fos) when you do NOT already have a specific trial surface. Layers that do not drain freely need the R envelope (soil_layers[].R_c, R_phi) or the measured R_d, R_psi.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            "drawdown_from_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation BEFORE drawdown (m)."},
            "drawdown_to_elevation": {"type": "float", "required": True, "description": "Reservoir surface elevation AFTER drawdown (m); must be below drawdown_from_elevation."},
            "method": {"type": "str", "required": False, "default": "corps_2stage", "allowed_values": _RD_METHODS, "description": "Rapid-drawdown stage method: 'corps_2stage' (default), 'lowe_karafiath', or 'duncan_3stage'. See rapid_drawdown_fos."},
            "surface_type": {"type": "str", "required": False, "default": "circular", "allowed_values": _RD_SURFACE_TYPES, "description": "Search strategy: 'circular' (centre grid), 'entry_exit' (arcs between entry/exit windows), 'noncircular' (random polylines), 'noncircular_de' (differential-evolution refinement)."},
            "x_range": {"type": "array", "required": False, "description": "[xmin, xmax] for the circle-centre grid (circular)."},
            "y_range": {"type": "array", "required": False, "description": "[ymin, ymax] for the circle-centre grid (circular)."},
            "x_entry_range": {"type": "array", "required": False, "description": "[xmin, xmax] window for the slip surface's ENTRY: its low-x end, whichever way the slope faces (the toe side of a slope rising to the right, the crest side of one falling to the right)."},
            "x_exit_range": {"type": "array", "required": False, "description": "[xmin, xmax] window for the slip surface's EXIT: its high-x end. Windows given the other way round (entry wholly right of exit) are swapped, and the result's warnings say so."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x (or entry divisions for entry_exit). Each trial is several LE solves -- keep modest."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y (or exit divisions for entry_exit)."},
            "n_trials": {"type": "int", "required": False, "default": 500, "description": "Random trials for noncircular searches."},
            "n_points": {"type": "int", "required": False, "default": 5, "description": "Polyline vertices for noncircular searches."},
            "n_slices": {"type": "int", "required": False, "default": 50, "description": "Number of slices per trial."},
            "seed": {"type": "int", "required": False, "description": "Random seed for reproducible noncircular searches."},
            "f_interslice": {"type": "str", "required": False, "default": "constant", "allowed_values": _F_INTERSLICE, "description": "GLE interslice function ('constant' = Spencer)."},
            "stage1_phreatic_points": {"type": "array", "required": False, "description": "The seepage line before drawdown, [[x,z],...]; omitted, a level pool at drawdown_from_elevation, named in the result as assumed (see rapid_drawdown_fos)."},
            "drawdown_phreatic_points": {"type": "array", "required": False, "description": "The piezometric line after drawdown, [[x,z],...]; omitted, a level pool at drawdown_to_elevation, named in the result as assumed (see rapid_drawdown_fos)."},
            "low_m_alpha": {"type": "str", "required": False, "default": "flag", "allowed_values": list(LOW_M_ALPHA_ACTIONS), "description": "A trial surface with m_alpha < 0.2 on a drawn-down slice at the drawdown FOS: 'flag' (default), kept and counted as flagged in search.trial_flags (Whitman and Bailey 1967, p. 487, print a warning, not a rejection); 'refuse', left out and counted in search.n_rejected.m_alpha, as before GM-140 step 3."},
            "time_budget_s": {"type": "float", "required": False, "default": SEARCH_TIME_BUDGET_S, "description": "Time budget of the whole search in seconds (at most 600), as search_critical_surface: each trial is two or three GLE solves, so the main search scores coarse first and stops at 80 % of the budget, and search.warnings and search.search_ranges.time_budget say what was covered."},
        },
        "returns": {"FOS": "Minimum drawdown factor of safety found.", "method": "Stage method used.", "surface_type": "Search strategy.", "n_surfaces_evaluated": "Number of surfaces checked.", "search": "Rich search result (critical surface geometry + diagnostics), with search_ranges, governing_location, refused_lowest and choices as search_critical_surface returns them.", "drawdown_detail": "The winning surface stage by stage, as rapid_drawdown_fos returns it: each stage's factor of safety, the stage the answer is from, the water assumed, the per-slice columns and the choices (layer_overlaps where a layer loses ground to one listed before it)."},
    },
    "yield_acceleration": {
        "category": "Slope Stability",
        "brief": "Yield (critical) seismic coefficient ky for a SPECIFIED slip surface: the horizontal pseudo-static coefficient kh at which the factor of safety = 1.0 (Newmark critical acceleration, ay = ky*g). Found by bisection on the module's pseudo-static FOS. Feed ky into newmark_displacement or newmark_jibson2007. Provide the surface as xc/yc/radius or slip_points.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            **_SURFACE_SPEC_PARAMS,
            "method": {"type": "str", "required": False, "default": "spencer", "allowed_values": _METHODS, "description": "Limit-equilibrium method for the pseudo-static FOS."},
            "n_slices": {"type": "int", "required": False, "default": 50, "description": "Number of slices."},
            "kh_max": {"type": "float", "required": False, "default": 2.0, "description": "Upper bound of the kh bracket; if the surface is still stable at kh_max the result is non-converged with ky=kh_max. Where the method has no solution at kh_max the upper end is halved until it has one."},
        },
        "returns": {"arc": "Present when the circle meets the ground at more than two points, or its arc's ends were given: which arc of the circle this result is ({x_start, x_end, index, n_arcs, arcs, description}) and, where the arcs were each analysed, governs, others (each other arc with its value, or not_scored and the reason) and not_scored (the count by reason: no_driving, not_converged, refused). Tell the user which arc the result is.", "ky": "Yield seismic coefficient (fraction of g). Null when the method has no solution on the surface at a trial kh the bisection needs: every trial kh is solved by the method itself, and nothing stands in for it (see reason).", "ay_m_s2": "Yield acceleration = ky*g (m/s^2); null with ky.", "FOS_static": "Pseudo-static FOS at kh=0 (null when the method has no solution there).", "converged": "Whether ky was bracketed.", "reason": "Present when ky is null: the method, the kh without a solution, and why.", "no_solution_kh": "That kh.", "ky_bounds": "[lower, upper] kh at which the method solved with F > 1 and F <= 1 (an end is null where there is none)."},
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
    "bray_travasarou_2007": {
        "category": "Slope Stability",
        "brief": "Bray & Travasarou (2007) simplified seismic DEVIATORIC slope displacement — a fully-coupled stick-slip sliding-block regression that captures sliding-mass flexibility (via the fundamental period Ts) and spectral demand at 1.5*Ts, which the rigid-block Jibson/integration models don't. Returns P(D=0) (prob. of negligible <=1 cm displacement) and the median non-zero displacement (16/84% via sigma_ln=0.66). A period-dependent cross-check for Jibson.",
        "parameters": {
            "ky": {"type": "float", "required": True, "description": "Yield / critical seismic coefficient (fraction of g)."},
            "ts": {"type": "float", "required": True, "description": "Initial fundamental period of the potential sliding mass (s); ~4H/Vs (deep/wide) or 2.6H/Vs (surface). Ts<0.05 s auto-uses the rigid-block branch (then sa_1p5ts is the PGA)."},
            "sa_1p5ts": {"type": "float", "required": True, "description": "5%-damped elastic spectral acceleration at the degraded period 1.5*Ts (g)."},
            "magnitude": {"type": "float", "required": True, "description": "Moment magnitude Mw."},
            "rigid": {"type": "bool", "required": False, "description": "Force the rigid (True) / flexible (False) branch; default auto-selects rigid when ts<0.05 s."},
        },
        "returns": {"displacement_cm": "Median deviatoric displacement (cm).", "displacement_16pct_cm": "16th-percentile displacement.", "displacement_84pct_cm": "84th-percentile displacement.", "p_zero": "P(D=0): probability of negligible (<=1 cm) displacement.", "displacement_m": "Median in m."},
    },
    "search_critical_surface": {
        "category": "Slope Stability",
        "brief": "Search for the critical slip surface (minimum FOS): circular centre-grid, entry-exit arcs, random noncircular polylines, differential-evolution noncircular refinement, PSO, or weak-layer-biased search.",
        "parameters": {
            **_GEOMETRY_PARAMS,
            "surface_type": {"type": "str", "required": False, "default": "circular", "allowed_values": _SURFACE_TYPES, "description": "Search strategy. 'entry_exit' grids circular arcs between entry/exit windows; 'noncircular_de' refines random polylines with scipy differential evolution."},
            "method": {"type": "str", "required": False, "default": "bishop", "allowed_values": _METHODS, "description": "Limit-equilibrium method (noncircular searches use spencer when 'bishop' is requested). Cost per trial surface, measured on four pinned slopes: fellenius about the same as bishop, janbu about twice bishop, and spencer, morgenstern_price and gle each about 30 to 40 times bishop at 30 slices and 40 to 70 times at 60 slices. A whole search costs less than that ratio where it scans with Bishop (the centre grid's refinement, the toe circles): the default 10 x 10 centre grid took 2-3 s with bishop and 14-16 s with spencer, a 10 x 10 or 12 x 12 entry_exit grid 2 s and 60-70 s, a 30 x 30 entry_exit grid at 60 slices 21 s and 972 s; the noncircular searches, scored with Spencer at their defaults (500 trials, 5 points), took 6-7 s (noncircular), 3-7 s (weak_layer) and 10-12 s (noncircular_de), 41 s where differential evolution scored 2,325 surfaces (development laptop; the app's host up to twice as long). On the circles of these slopes bishop and spencer gave FOS within 0.3 % of each other. Which method to use is the engineer's call; bishop_screen keeps the method and scans with Bishop."},
            "x_range": {"type": "array", "required": False, "description": "[xmin, xmax] for circle center search."},
            "y_range": {"type": "array", "required": False, "description": "[ymin, ymax] for circle center search."},
            "x_entry_range": {"type": "array", "required": False, "description": "[xmin, xmax] window for the slip surface's ENTRY: its low-x end, whichever way the slope faces (the toe side of a slope rising to the right, the crest side of one falling to the right)."},
            "x_exit_range": {"type": "array", "required": False, "description": "[xmin, xmax] window for the slip surface's EXIT: its high-x end. Windows given the other way round (entry wholly right of exit) are swapped, and the result's warnings say so."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x (or entry divisions for entry_exit)."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y (or exit divisions for entry_exit)."},
            "n_trials": {"type": "int", "required": False, "default": 500, "description": "Random trials for noncircular searches."},
            "n_points": {"type": "int", "required": False, "default": 5, "description": "Polyline vertices for noncircular searches."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices per trial."},
            "seed": {"type": "int", "required": False, "description": "Random seed for reproducible noncircular/PSO/DE searches."},
            "time_budget_s": {"type": "float", "required": False, "default": SEARCH_TIME_BUDGET_S, "description": "Time budget of the whole search in seconds (at most 600). The main search scores its centres or entry/exit pairs coarse first and stops at 80 % of the budget; the result is then the lowest surface found, and warnings and search_ranges.time_budget say what was scored of how many, at what spacing, the run time projected for the whole search, and what was not run. Raise it for a long search you mean to wait for."},
            "bishop_screen": {"type": "bool", "required": False, "default": False, "description": "circular and entry_exit searches with spencer / morgenstern_price / gle: score every centre or pair with Bishop's simplified method, then re-score the lowest (at least 10, every one within 5 % of the lowest, at most 30) with the method itself; the reported FOS is the method's own. On four pinned slopes it found the same surfaces and FOS as the unscreened search in 35-60 % of the time on the default centre grid, 15-30 % on 10 x 10 and 12 x 12 entry_exit grids and 3 % on a 30 x 30 entry_exit grid at 60 slices (33 s against 972 s). Not applied where supports are modelled; noncircular searches are not screened."},
            "m_alpha_rule": {"type": "str", "required": False, "default": "frictional", "allowed_values": ["frictional", "every_slice"], "description": "Which slices the m_alpha < 0.2 check looks at (Whitman and Bailey 1967, p. 487). 'frictional': slices with phi' > 0; on a phi' = 0 slice m_alpha = cos(a) does not depend on F and the strength does not depend on the normal force (the module's own reasoning, no printed source). 'every_slice': every slice."},
            "low_m_alpha": {"type": "str", "required": False, "default": "flag", "allowed_values": list(LOW_M_ALPHA_ACTIONS), "description": "What the search does with a trial surface that has m_alpha < 0.2 on a slice. 'flag' (default): the surface is kept and scored, and the result says so with the surface and the slice (Whitman and Bailey 1967, p. 487, print a warning and an instruction to examine the result, not a rejection). 'refuse': it is left out and counted (n_rejected.m_alpha), as the module did before GM-140 step 3 and as Slide2 does with its 'Check malpha < 0.2' setting on."},
            "negative_friction_term": {"type": "str", "required": False, "default": "as_printed", "allowed_values": list(NEGATIVE_TERM_RULES), "description": "Bishop's simplified method and the Ordinary Method of Slices on a slice where the pore-water force on the base exceeds the slice's weight. 'as_printed' (default): the frictional term W - u b (Bishop 1955 Eq. 13, p. 10) or W cos(alpha) - u l (Eq. 9) is used as printed, negative. 'zero': it is set to zero there, the module's rule before GM-140 step 3, with no printed source. Either way the result names the slices (acceptability, check 'negative_frictional_term'), as Whitman and Bailey (1967, p. 487) ask."},
        },
        "returns": {"critical": "Critical surface result (FOS, geometry): the lowest of the surfaces the method itself solved. A surface the method has no solution on is not scored by another solver. With method 'bishop', on a circle with no supports and no seismic coefficient where Bishop's value is below the Ordinary Method of Slices value, the Ordinary Method of Slices value is taken (Duncan 1996, p. 580): critical.FOS_bishop and critical.FOS_fellenius give both and critical.method_used says so.", "n_surfaces_evaluated": "Number of surfaces checked.", "circles_split": "Present when any trial circle met the ground at more than two points: {arcs: circles}, the number of such circles by the number of arcs below the ground each gave. Each arc inside the entry / exit windows was scored as a trial surface of its own (the module's rule; no reference prints one) and the circle's score is its lowest arc's; arcs with no driving moment are counted in no_solution under no_driving. critical.arc says which arc the governing surface is; pass its x_start and x_end as arc_x_start and arc_x_end to analyse that surface again.", "arcs_outside_windows": "Present when the entry / exit windows (given, or the automatic ones) left arcs of split circles out: {family: arcs}, not scored. The toe-circle families score only the arc that ends at the toe or face point.", "no_solution": "Trial surfaces left without a solution, by family and method: [{family, method, n, not_converged, no_driving, below_0_05}] (present when there are any). They were not scored. not_converged: the method's own solve did not converge on them; when the governing family lost trials this way warnings says so, because a lower surface among them cannot be ruled out (UFC 3-220-10 p. 362). Tell the user. no_driving: surfaces with no net driving moment or force (held, for example by a support), which have no factor of safety. below_0_05: surfaces the method solved with a value below 0.05, which a search does not take as a factor of safety.", "n_rejected": "Trial surfaces refused, by reason (present when any were): geometry, nonconverged, jagged, ground_crossings (noncircular surfaces below the ground over more than one stretch: they leave the ground and come back into it, so they bound more than one sliding mass; a circle is not refused for it, see circles_split), and m_alpha (only with low_m_alpha 'refuse': m_alpha below 0.2 on a slice with phi' > 0, or any slice with m_alpha_rule 'every_slice'; by default such surfaces are kept and flagged, see trial_flags).", "trial_flags": "The printed acceptability checks on the scored trials, by family and method: [{family, method, n_scored, n_flagged, by_check}]. A flagged trial is kept and scored.", "critical_flags": "The checks the governing surface does not meet (empty when it meets them all); critical.acceptability gives each finding with the slice and the value. Report them to the user with the factor of safety.", "lowest_unflagged": "Present when the governing surface carries a flag: the lowest factor of safety among the surfaces the search scored by the same method that carry none ({FOS, family, method, and the surface}), or FOS null with a note when it scored none. The headline (critical) stays the lowest solved surface; give the user both.", "refuse_option": "Present when the governing surface carries a flag: the option that leaves surfaces with its flags out of the search, or that there is none (low_m_alpha='refuse' leaves out surfaces with m_alpha below 0.2; no option leaves out the other flags).", "summary": "One sentence: the headline (the lowest solved surface, its FOS, family and method), and when it carries a flag, its flags, the lowest scored surface with no flag (or none found) and refuse_option. Neither number alone is safe: give the user this sentence's numbers together.", "screen_refusals": "Noncircular trial surfaces refused by each of the module's own screens, which are not found in print: slice_count and span, applied before any solve, and jagged_below_1_5 (a FOS below 1.5 on a jagged surface), applied with method janbu or fellenius, which no printed check of EM 1110-2-1902 C-10a covers, so the surfaces it refuses carry no flag: {n, n_flagged}; present when any were. With spencer, morgenstern_price or gle (and bishop, whose noncircular surfaces Spencer scores) a low factor of safety on a jagged surface, or a solution that held about one moment axis or one slicing only, is no longer refused: such a surface is scored, kept and flagged, and can be the headline, with lowest_unflagged beside it.", "governing_surface": "Kind of surface that governs: 'circular' / 'noncircular' (the main search), 'toe_circle', 'circle_above_toe' (lower end on the face above the toe), or 'infinite_slope' (shallow planar sliding where c'=0 soil forms the face). A circular or entry_exit search also scores toe circles and circles passing above the toe, and every search scores shallow planar surfaces where c'=0 soil forms the face; the lowest is reported.", "candidates": "Minimum FOS of each surface family scored.", "infinite_slope_checks": "Each stretch of face where c'=0 soil is at the surface: slope angle, phi', tan(phi')/tan(beta), and the planar-surface FOS (or why it was not evaluated).", "warnings": "What the engineer must check on this surface: no equilibrium solution, the tension crack it was solved with (a reverse-curvature circle closed by one says so), supports Bishop/OMS count little of (with Spencer's FOS on the circle), and each printed acceptability check not met (with EM 1110-2-1902 C-10b's remedies for a steep exit; none is applied). Report these to the user.", "search_ranges": "The entry/exit windows and centre ranges used, which were set automatically from the slope (its toe, crest, height, depth to the base) and where the model's end cut them: what the search covered; time_budget when the budget stopped the search (what was scored, of how many, the projected run time, what was not run).", "governing_location": "Where the governing surface lies: its ends, lowest point, greatest depth below the ground, the layers along its base, and its position against the slope's toe and crest in multiples of H.", "refused_lowest": "No longer returned: a noncircular surface whose solution held about one moment axis or one slicing only is now scored, kept and flagged (trial_flags, critical_flags).", "crack_refused": "Trial surfaces the tension-crack rule refused, by reason (present when any were): reverse_curvature_no_crack (reverse-curvature circles with no crack asked for; naming a crack admits those whose overhang lies within its depth), reverse_curvature_deeper_than_crack (overhang deeper than the crack asked for), reverse_curvature_no_crack_at_end (overhang at the lower end, or at the end away from the crack), reverse_curvature_discard_asked (reverse_curvature='discard'), within_cracked_depth (the surface nowhere passes below the crack's depth, measured below the ground above it, so it is no slip surface) and scan_no_minimum (the scan for the crack depth of minimum F found no minimum). Each: n, what, x_end_range_m (where their upper ends are) and the first few with centre, radius, depths and why; choices gives the counts. The critical surface's own crack is in critical.tension_crack.", "choices": "What the search chose and how (given, default or module), with basis and alternatives: the search and its method, the surface families scored (one it runs that scored none, and why), the search ranges, the m_alpha check, the time budget, any Bishop screen.", "layer_overlaps": "Where a layer loses ground to one listed before it (present when any does): layer, held_by, x_from, x_to, max_overlap_m, statement."},
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
            **_CORRELATIONS_PARAM,
        },
        "returns": {"FOS_mean_values": "FOS at mean (most-likely) values. Every evaluation is the method's own solution on the surface: where the method has none at the mean values or at a value one standard deviation off, there is no FOSM result and the error names the variable, the value and why.", "COV_F": "Coefficient of variation of FOS.", "beta_lognormal": "Lognormal reliability index (Duncan 2000).", "pf_lognormal": "Probability of failure.", "variable_variance_pct": "Per-variable share of Var[FOS] (with a 'corr(k1,k2)' entry per correlated pair)."},
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
            "time_budget_s": {"type": "float", "required": False, "default": SEARCH_TIME_BUDGET_S, "description": "With research_surface: the time budget of EACH realization's search, in seconds (at most 600), as search_critical_surface. It is per search, not for the run: n realizations may take up to n times it, so keep n small."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
            **_CORRELATIONS_PARAM,
        },
        "returns": {"pf": "Probability of failure (FOS < 1), over the realizations in which the method has a solution.", "n_realizations": "Realizations with a solution (the statistics are over these).", "n_no_solution": "Realizations in which the method has no solution on the surface (present when there are any): counted and left out of the statistics, never taken as a sample.", "pf_bounds": "With n_no_solution: [lower, upper] probability of failure, counting none of those realizations as a failure and all of them. Report the bounds with pf.", "no_solution_note": "With n_no_solution: the statement to pass on to the user. The rule is the module's own; no printed rule was found for it.", "FOS_mean": "Mean FOS.", "FOS_std": "Std of FOS.", "beta_lognormal": "Lognormal reliability index from the sample moments.", "histogram_bins": "Histogram bin edges.", "histogram_counts": "Histogram counts."},
    },
}

# The solver tolerance these methods pass on (accepted, and now documented,
# since dispatch refuses an undocumented name: GM-110).
for _m in ("analyze_slope", "compare_methods_table", "rapid_drawdown_fos",
           "search_rapid_drawdown", "yield_acceleration",
           "search_critical_surface", "fosm_fos", "monte_carlo_fos"):
    METHOD_INFO[_m]["parameters"]["tol"] = {
        "type": "float", "required": False, "default": 1e-4,
        "description": "Convergence tolerance for the iterative methods."}

# When each default applies (GM-119), so a result's defaults_used lists only
# what the run used: the crack's water, side and model to a crack; reverse
# curvature to circles; each search setting to the searches that read it
# (nx and ny the centre grid, the entry/exit pairs and the swarm; n_trials
# the random polylines, the swarm and the weak-layer search; n_points every
# noncircular search; the Bishop screen circles with a rigorous method);
# the M-P/GLE interslice function to those methods; a Monte Carlo search
# budget to re-searched surfaces.
_CRACK = [{"tension_crack_depth": "set"},
          {"tension_crack_depth_basis": ["rankine", "min_fos"]}]
_CIRCLE_GIVEN = {"xc": "set"}
_SEARCH_CIRCLES = {"surface_type": ["circular", "entry_exit"]}
_WATER = {"water_condition": ["seepage_parallel"]}


def _crack_and(extra):
    out = {k: _CRACK for k in ("tension_crack_water_depth",
                               "tension_crack_side", "tension_crack_model")}
    out["tension_crack_depth_basis"] = {"tension_crack_depth": "set"}
    out.update(extra)
    return out


#: yield_acceleration finds kh itself (BACKLOG GM-113): a kh given beside
#: the geometry was read into it and then overwritten at every trial, so
#: it changed nothing. 0, no seismic coefficient, passes.
YIELD_REFUSED = {"kh": {
    "reason": ("yield_acceleration finds kh itself: it is the coefficient "
               "at which the factor of safety is 1.0 (ky), found by "
               "bisection between 0 and kh_max, so a given kh is not used. "
               "Leave kh out; for the factor of safety at a given kh, call "
               "analyze_slope with that kh"),
    "accepts": [0, 0.0]}}
METHOD_INFO["yield_acceleration"]["parameters"].pop("kh")
METHOD_INFO["yield_acceleration"]["refused_parameters"] = YIELD_REFUSED

#: The three-stage drained check has one printed basis (GM-140 step 9, EM
#: 1110-2-1902 Eqs. G-13 to G-15); a caller who still names the removed
#: input is told the rule.
from slope_stability.rapid_drawdown import STAGE3_NORMAL_RULE  # noqa: E402
RD_REFUSED = {"stage3_effective_normal": {"reason": STAGE3_NORMAL_RULE,
                                          "accepts": []}}
for _m in ("rapid_drawdown_fos", "search_rapid_drawdown"):
    METHOD_INFO[_m]["refused_parameters"] = RD_REFUSED

declare_applies_when(METHOD_INFO, {
    "analyze_slope": _crack_and({
        "reverse_curvature": _CIRCLE_GIVEN,
        "f_interslice": [{"method": ["morgenstern_price", "gle"]},
                         {"compare_methods": [True]}],
        "negative_friction_term": [{"method": ["bishop", "fellenius"]},
                                   {"compare_methods": [True]}],
        "m_alpha_rule": {"method": ["bishop", "janbu", "spencer",
                                    "morgenstern_price", "gle"]}}),
    "compare_methods_table": _crack_and({
        "reverse_curvature": _CIRCLE_GIVEN}),
    "rapid_drawdown_fos": _crack_and({
        "reverse_curvature": _CIRCLE_GIVEN}),
    "search_rapid_drawdown": _crack_and({
        "reverse_curvature": _SEARCH_CIRCLES,
        "nx": _SEARCH_CIRCLES, "ny": _SEARCH_CIRCLES,
        "n_trials": {"surface_type": ["noncircular"]},
        "n_points": {"surface_type": ["noncircular", "noncircular_de"]}}),
    "yield_acceleration": _crack_and({
        "reverse_curvature": _CIRCLE_GIVEN}),
    "search_critical_surface": _crack_and({
        "reverse_curvature": _SEARCH_CIRCLES,
        "nx": {"surface_type": ["circular", "entry_exit", "pso"]},
        "ny": {"surface_type": ["circular", "entry_exit", "pso"]},
        "n_trials": {"surface_type": ["noncircular", "pso", "weak_layer"]},
        "n_points": {"surface_type": ["noncircular", "noncircular_de", "pso",
                                      "weak_layer"]},
        "bishop_screen": {**_SEARCH_CIRCLES,
                          "method": ["spencer", "morgenstern_price", "gle"]},
        "negative_friction_term": {"method": ["bishop", "fellenius"]}}),
    "fosm_fos": _crack_and({"reverse_curvature": _CIRCLE_GIVEN}),
    "monte_carlo_fos": _crack_and({
        "time_budget_s": {"research_surface": [True]}}),
    "infinite_slope_fos": {
        "depth": [{"c": "set"}, {**_WATER, "water_depth": "set"}],
        "gamma_w": _WATER, "water_depth": _WATER,
        "ru": {"water_condition": ["ru"]}},
})

# What a left-out input with no default stands for (BACKLOG GM-165): the
# water table, the surcharge's extent, the search's windows and grid, the
# random seed, independent variables, the stage-1 phreatic surface and the
# Bray-Travasarou branch. Soil nails, anchors, geosynthetics and
# stabilizing piles left out are simply not there.
_NO_WATER = ("no water table",
             "no pore water pressure: effective stress equals total stress")
_FROM_WATER = ("from the water table",
               "hydrostatic below gwt_points; none without a water table")
_WHOLE_SURFACE = ("the whole ground surface",
                  "the single surcharge acts over the entire surface")
_NO_ZONES = ("no surcharge zones", "only the single surcharge, if any")
_WINDOW = ("set from the slope's geometry",
           "entry and exit windows from the slope's toe, crest, height and "
           "depth to the base; the result lists what the search covered")
_GRID = ("set from the slope's geometry",
         "the circle-centre grid over the windows, from the slope's height "
         "and depth to the base")
_NOT_SEEDED = ("not seeded", "the trial surfaces are random and can differ "
               "from run to run; give seed to repeat a run")
_INDEPENDENT = ("independent variables",
                "no correlation between the random variables")
_POOL = ("a level pool at drawdown_from_elevation",
         "hydrostatic below the pool before drawdown: the assumption of "
         "the manual's own example (EM 1110-2-1902 G-4b, p. G-9); seepage "
         "analyses would often be appropriate")
_LOW_POOL = ("a level pool at drawdown_to_elevation",
             "hydrostatic below the lowered pool after drawdown, as in the "
             "manual's example (EM 1110-2-1902 G-4b, p. G-9)")
_LOADS = {"surcharge_x_range": _WHOLE_SURFACE, "surcharges": _NO_ZONES}
_GROUND = {"gwt_points": _NO_WATER, "pore_pressure_points": _FROM_WATER,
           **_LOADS}
declare_when_absent(METHOD_INFO, {
    "analyze_slope": _GROUND,
    "compare_methods_table": _GROUND,
    "yield_acceleration": _GROUND,
    "search_critical_surface": {
        **_GROUND, "x_entry_range": _WINDOW, "x_exit_range": _WINDOW,
        "x_range": _GRID, "y_range": _GRID, "seed": _NOT_SEEDED},
    "fosm_fos": {**_GROUND, "correlations": _INDEPENDENT},
    "monte_carlo_fos": {**_GROUND, "correlations": _INDEPENDENT,
                        "seed": _NOT_SEEDED},
    "rapid_drawdown_fos": {**_LOADS, "stage1_phreatic_points": _POOL,
                           "drawdown_phreatic_points": _LOW_POOL},
    "search_rapid_drawdown": {
        **_LOADS, "stage1_phreatic_points": _POOL,
        "drawdown_phreatic_points": _LOW_POOL,
        "x_entry_range": _WINDOW, "x_exit_range": _WINDOW,
        "x_range": _GRID, "y_range": _GRID, "seed": _NOT_SEEDED},
    "bray_travasarou_2007": {
        "rigid": ("chosen by the period ts",
                  "the rigid branch when ts < 0.05 s, else the flexible "
                  "branch")},
})
for _m, _info in METHOD_INFO.items():
    _p = _info["parameters"]
    for _name, _when in (
            ("surcharge_x_range", {"surcharge": "set"}),
            ("gwt_points", {"pore_pressure_points": "unset"}),
            ("pore_pressure_points", {"gwt_points": "set"})):
        if _name in _p and _p[_name].get("when_absent"):
            _p[_name]["applies_when"] = _when
# The reading of the water line (GM-140 step 5a) applies where a line
# gives the pore pressure: gwt_points without a pore-pressure grid, or,
# in a rapid drawdown (whose pools are level), a stage-1 seepage line.
for _m, _info in METHOD_INFO.items():
    _p = _info["parameters"]
    if "water_line_kind" in _p:
        _p["water_line_kind"] = {
            **_p["water_line_kind"],
            "applies_when": (
                {"stage1_phreatic_points": "set"}
                if "stage1_phreatic_points" in _p else
                {"gwt_points": "set", "pore_pressure_points": "unset"})}
# What a support whose pull assists the sliding counts (GM-140 step 8)
# applies where there is a support.
for _m, _info in METHOD_INFO.items():
    _p = _info["parameters"]
    if "assisting_support" in _p:
        _p["assisting_support"] = {
            **_p["assisting_support"],
            "applies_when": [{_k: "set"} for _k in (
                "nails", "anchors", "geosynthetics", "stabilizing_piles",
                "reinforcement_force") if _k in _p]}
for _m, _seeded in (
        ("search_critical_surface",
         ["noncircular", "noncircular_de", "pso", "weak_layer"]),
        ("search_rapid_drawdown", ["noncircular", "noncircular_de"])):
    _p = METHOD_INFO[_m]["parameters"]
    _p["seed"]["applies_when"] = {"surface_type": _seeded}
    for _name in ("x_range", "y_range"):
        _p[_name]["applies_when"] = {
            "surface_type": ["circular", "entry_exit", "pso"]}

# The keys each item of a list parameter carries, so a key given at the
# top level is refused with a hint naming its list (BACKLOG GM-113).
from funhouse_agent.adapters import item_keys_for, merge_item_keys  # noqa: E402,F401
ITEM_KEYS = item_keys_for(METHOD_INFO, "soil_layers", _LAYER_KEYS)
