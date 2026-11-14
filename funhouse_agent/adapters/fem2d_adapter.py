"""FEM2D adapter — flat dict -> fem2d high-level API -> dict."""

import numpy as np

from funhouse_agent.adapters import (
    check_nested_list, clean_result, declare_applies_when,
    reject_unknown_params, require_keys, require_params,
)


# Keys a soil_layers[] dict may carry: what fem2d reads from it (and the
# top_elevation and model that dxf_import's build_fem_inputs writes into
# each layer). Anything else was read with .get and a module default and
# silently left out: a layer's 'cohesion' ran with c = 0, 'unit_weight' at
# gamma = 18 and 'E_kPa' at E = 30000 (GM-58).
_LAYER_KEYS = ("name", "top_elevation", "bottom_elevation", "E", "nu", "c",
               "phi", "psi", "gamma", "model")
_HS_KEYS = ("E50_ref", "Eur_ref", "m", "p_ref", "R_f")
_EXCAVATION_LAYER_KEYS = _LAYER_KEYS + ("K0", "k") + _HS_KEYS
_CONSOLIDATION_LAYER_KEYS = _LAYER_KEYS + ("fill",)
# fem2d_staged's nested items, as the adapter and analyze_staged read them.
_MATERIAL_KEYS = ("name", "E", "nu", "c", "phi", "psi", "model") + _HS_KEYS
_PHASE_KEYS = ("name", "active_soil_groups", "active_beam_ids",
               "surface_loads", "gwt", "n_steps", "reset_displacements")
_BEAM_KEYS = ("node_i", "node_j", "EA", "EI", "weight_per_m")
_STRUT_KEYS = ("depth", "stiffness")
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "cohesion": ("c",), "cprime": ("c",), "cu": ("c",), "su": ("c",),
    "frictionangle": ("phi",), "phiprime": ("phi",),
    "dilation": ("psi",), "dilationangle": ("psi",),
    "unitweight": ("gamma",), "poisson": ("nu",), "poissonsratio": ("nu",),
    "modulus": ("E",), "youngsmodulus": ("E",), "es": ("E",),
    "top": ("top_elevation",), "bottom": ("bottom_elevation",),
    "permeability": ("k",), "hydraulicconductivity": ("k",),
    "groups": ("active_soil_groups",), "activegroups": ("active_soil_groups",),
    "loads": ("surface_loads",), "beams": ("active_beam_ids",),
    "steps": ("n_steps",), "nsteps": ("n_steps",), "k": ("stiffness",),
    "weight": ("weight_per_m",),
}
_LAYER_KEY_NOTES = {
    ("c",): ("(fem2d has one cohesion, c: for an undrained layer give the "
             "undrained strength as c with phi = 0.)"),
}


def _require_layer_elevations(soil_layers, *, method, valid=_LAYER_KEYS):
    """fem2d builds strata from each layer's bottom_elevation; everything else
    (E, nu, c, phi, psi, gamma) has module defaults, so an unknown key is
    refused first (``valid``: the keys this analysis reads)."""
    check_nested_list(soil_layers, valid, method=method, label="soil_layers",
                      hints=_LAYER_KEY_HINTS, notes=_LAYER_KEY_NOTES)
    for sl in soil_layers:
        require_keys(sl, ["bottom_elevation"], method=method,
                     item_label="soil_layers[]")


def _require_choice(value, allowed, *, name, method):
    """Single-sourced enum/choice validation for adapter parameters.

    Replaces the per-parameter inline ``if value not in (...): raise`` blocks
    (element_type / srm_field / n_gp / consolidation_scheme) with one check so
    the allowed set lives in exactly one place per call site.
    """
    if value not in allowed:
        raise ValueError(
            f"{method}: {name} must be one of {list(allowed)}, got {value!r}.")


def _require_positive(value, *, name, method, hint=""):
    """Reject a non-positive value for an input that must be a positive physical
    dimension (e.g. a domain depth). A negative or zero value silently builds a
    degenerate FE mesh and returns non-converged garbage rather than erroring —
    so guard it with a clear message. ``None`` (use the default) is allowed.
    """
    if value is not None and value <= 0:
        raise ValueError(
            f"{method}: '{name}' must be a positive distance in metres; got "
            f"{value}. A negative or zero value makes the FE domain degenerate "
            f"and the solve will not converge.{hint}")


def _normalize_gwt(gwt):
    """Coerce a groundwater-table polyline [[x, y], ...] to an ndarray; pass a
    scalar elevation (or an already-array value) through unchanged.

    Single-sources the identical GWT-argument coercion previously copy-pasted
    into the slope-SRM, excavation and staged adapters.
    """
    if isinstance(gwt, list) and len(gwt) > 0 and isinstance(gwt[0], (list, tuple)):
        return np.array(gwt)
    return gwt

def _package_model(soil_layers, gwt) -> dict:
    """What the fem2d calc package prints of a layered model beside its
    result: the layers as given (its materials table) and the water table
    in words. A layer value the call left out is left out here too."""
    model = {"soil_layers": [dict(sl) for sl in soil_layers]}
    if gwt is None:
        model["gwt_description"] = "No water table (no pore water pressure)"
    elif isinstance(gwt, (int, float)):
        model["gwt_description"] = f"Water table at elevation {gwt:g} m"
    else:
        model["gwt_description"] = (
            "Water table along the polyline "
            + ", ".join(f"({float(x):g}, {float(z):g})" for x, z in gwt))
    return model


def _run_analyze_gravity(params: dict) -> dict:
    from fem2d import analyze_gravity
    _valid = ("width", "depth", "gamma", "E", "nu", "nx", "ny", "t")
    reject_unknown_params(params, _valid, method="fem2d_gravity")
    require_params(params, ["width", "depth", "gamma", "E", "nu"],
                   method="fem2d_gravity", valid=_valid)
    _require_positive(params.get("depth"), name="depth", method="fem2d_gravity")
    result = analyze_gravity(
        width=params["width"],
        depth=params["depth"],
        gamma=params["gamma"],
        E=params["E"],
        nu=params["nu"],
        nx=params.get("nx", 20),
        ny=params.get("ny", 10),
        t=params.get("t", 1.0),
    )
    return clean_result(result.to_dict())


def _run_analyze_foundation(params: dict) -> dict:
    from fem2d import analyze_foundation
    _valid = ("B", "q", "depth", "E", "nu", "gamma", "nx", "ny", "t")
    reject_unknown_params(params, _valid, method="fem2d_foundation")
    require_params(params, ["B", "q", "depth", "E", "nu"],
                   method="fem2d_foundation", valid=_valid)
    _require_positive(params.get("depth"), name="depth",
                      method="fem2d_foundation")
    result = analyze_foundation(
        B=params["B"],
        q=params["q"],
        depth=params["depth"],
        E=params["E"],
        nu=params["nu"],
        gamma=params.get("gamma", 0.0),
        nx=params.get("nx", 30),
        ny=params.get("ny", 15),
        t=params.get("t", 1.0),
    )
    return clean_result(result.to_dict())


def _run_analyze_footing_capacity(params: dict) -> dict:
    from fem2d import analyze_footing_capacity
    _valid = ("B", "c", "phi", "gamma", "E", "nu", "psi", "surcharge",
              "q_max", "n_load_steps", "domain_depth", "domain_half_width",
              "nx", "ny", "element_type", "max_iter", "tol", "q_applied",
              "footing", "base", "settlement_max", "mesh")
    reject_unknown_params(params, _valid, method="fem2d_footing_capacity")
    require_params(params, ["B", "c"], method="fem2d_footing_capacity",
                   valid=_valid)
    element_type = params.get("element_type", "t6")
    _require_choice(element_type, ("t6", "cst"),
                    name="element_type", method="fem2d_footing_capacity")
    footing = params.get("footing", "flexible")
    _require_choice(footing, ("flexible", "rigid"),
                    name="footing", method="fem2d_footing_capacity")
    base = params.get("base", "rough")
    _require_choice(base, ("rough", "smooth"),
                    name="base", method="fem2d_footing_capacity")
    mesh = params.get("mesh", "uniform")
    _require_choice(mesh, ("uniform", "graded"),
                    name="mesh", method="fem2d_footing_capacity")
    result = analyze_footing_capacity(
        B=params["B"], c=params["c"],
        phi=params.get("phi", 0.0), gamma=params.get("gamma", 0.0),
        E=params.get("E", 1e5), nu=params.get("nu", 0.3),
        psi=params.get("psi", 0.0), surcharge=params.get("surcharge", 0.0),
        q_max=params.get("q_max"), n_load_steps=params.get("n_load_steps", 45),
        domain_depth=params.get("domain_depth"),
        domain_half_width=params.get("domain_half_width"),
        nx=params.get("nx", 40), ny=params.get("ny", 20),
        element_type=element_type,
        max_iter=params.get("max_iter"), tol=params.get("tol", 1e-4),
        q_applied=params.get("q_applied"),
        footing=footing, base=base,
        settlement_max=params.get("settlement_max"), mesh=mesh,
    )
    out = {
        "q_ult_kPa": result.q_ult_kPa,
        "footing": result.footing,
        "base": result.base,
        "mesh": result.mesh,
        "Nc_backfigured": result.Nc_backfigured,
        "bearing_FOS": result.bearing_FOS,
        "bearing_capacity_factors": result.bearing_capacity_factors,
        "q_ult_estimate_kPa": result.q_ult_estimate_kPa,
        "q_max_kPa": result.q_max_kPa,
        "n_steps_converged": result.n_steps_converged,
        "collapse_load_fraction": result.collapse_load_fraction,
        "collapse_bracketed": result.collapse_bracketed,
        "surcharge_kPa": result.surcharge_kPa,
        "max_displacement_m": result.max_displacement_m,
        "choices": result.choices,
    }
    if result.n_steps_settled:
        out["n_steps_settled"] = result.n_steps_settled
    if result.footing == "rigid":
        out["settlement_max_m"] = result.settlement_max_m
        out["pressure_settlement"] = result.pressure_settlement
    if result.warnings:
        out["warnings"] = list(result.warnings)
    return clean_result(out)


def run_fem2d_slope_srm(params: dict, *, method="fem2d_slope_srm"):
    """The fem2d_slope_srm tool's run: (the result object, what the calc package
    prints of the model beside it)."""
    from fem2d import analyze_slope_srm
    _valid = ("surface_points", "soil_layers", "nx", "ny", "srf_tol",
              "n_load_steps", "t", "gamma_w", "max_iter", "tol",
              "element_type", "srm_field", "blowup_factor", "srf_range",
              "n_gp", "depth", "x_extend", "gwt", "layer_polylines",
              "mesh_check")
    reject_unknown_params(params, _valid, method=method)
    require_params(params, ["surface_points", "soil_layers"],
                   method=method, valid=_valid)
    _require_layer_elevations(params["soil_layers"], method=method)
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    soil_layers = params["soil_layers"]

    element_type = params.get("element_type", "t6")
    _require_choice(element_type, ("t6", "cst"),
                    name="element_type", method=method)
    srm_field = params.get("srm_field", "c_phi")
    _require_choice(srm_field, ("c_phi", "c", "phi"),
                    name="srm_field", method=method)
    n_gp = params.get("n_gp")
    if n_gp is not None:
        _require_choice(int(n_gp), (3, 6),
                        name="n_gp", method=method)
    _require_positive(
        params.get("depth"), name="depth", method=method,
        hint=" Omit 'depth' for the 2*H default. To compare FOS across several "
             "meshes (a mesh-refinement / mesh-consistency study), use the "
             "fem2d_srm_mesh_refinement_study method instead of driving this per mesh.")

    kwargs = dict(
        surface_points=surface_points,
        soil_layers=soil_layers,
        nx=params.get("nx", 30),
        ny=params.get("ny", 15),
        srf_tol=params.get("srf_tol", 0.02),
        n_load_steps=params.get("n_load_steps", 2),
        t=params.get("t", 1.0),
        gamma_w=params.get("gamma_w", 9.81),
        max_iter=params.get("max_iter", 1000),
        tol=params.get("tol", 1e-5),
        element_type=element_type,
        srm_field=srm_field,
        blowup_factor=params.get("blowup_factor", 15.0),
        srf_range=tuple(params.get("srf_range", (0.5, 3.0))),
        n_gp=int(n_gp) if n_gp is not None else None,
        mesh_check=bool(params.get("mesh_check", False)),
    )
    if "depth" in params:
        kwargs["depth"] = params["depth"]
    if "x_extend" in params:
        kwargs["x_extend"] = params["x_extend"]
    if "gwt" in params:
        kwargs["gwt"] = _normalize_gwt(params["gwt"])
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]

    result = analyze_slope_srm(**kwargs)
    return result, _package_model(soil_layers, params.get("gwt"))


def _run_analyze_slope_srm(params: dict) -> dict:
    result = run_fem2d_slope_srm(params)[0]
    out = clean_result(result.to_dict())
    out["fos_basis"] = getattr(result, "fos_basis", None)
    out["n_srf_trials"] = getattr(result, "n_srf_trials", None)
    if getattr(result, "mesh_check", None):
        out["mesh_check"] = result.mesh_check
    curve = getattr(result, "srf_curve", None)
    if curve is not None:
        srf, dim = curve
        out["srf_curve"] = {
            "srf": [round(float(v), 4) for v in srf],
            "dimensionless_displacement": [round(float(v), 4) for v in dim],
        }
    return out


def _run_srm_mesh_refinement_study(params: dict) -> dict:
    from fem2d import srm_mesh_refinement_study
    method = "fem2d_srm_mesh_refinement_study"
    _valid = ("surface_points", "soil_layers", "meshes", "published",
              "conv_tol", "depth", "x_extend", "element_type", "srm_field",
              "srf_tol", "n_load_steps", "t", "gamma_w", "max_iter", "tol",
              "blowup_factor", "srf_range", "n_gp", "gwt", "layer_polylines")
    reject_unknown_params(params, _valid, method=method)
    require_params(params, ["surface_points", "soil_layers", "meshes"],
                   method=method, valid=_valid)
    _require_layer_elevations(params["soil_layers"], method=method)
    meshes = [tuple(int(v) for v in m) for m in params["meshes"]]
    if len(meshes) < 2 or any(len(m) != 2 or min(m) < 1 for m in meshes):
        raise ValueError(
            f"{method}: 'meshes' must be at least two [nx, ny] pairs of "
            "positive integers, coarse to fine (three or more give the "
            "Richardson estimate).")
    element_type = params.get("element_type", "t6")
    _require_choice(element_type, ("t6", "cst"),
                    name="element_type", method=method)
    srm_field = params.get("srm_field", "c_phi")
    _require_choice(srm_field, ("c_phi", "c", "phi"),
                    name="srm_field", method=method)
    n_gp = params.get("n_gp")
    if n_gp is not None:
        _require_choice(int(n_gp), (3, 6), name="n_gp", method=method)
    _require_positive(params.get("depth"), name="depth", method=method)
    _require_positive(params.get("conv_tol"), name="conv_tol", method=method)

    kwargs = {"element_type": element_type, "srm_field": srm_field}
    for name in ("depth", "x_extend", "published", "conv_tol", "srf_tol",
                 "n_load_steps", "t", "gamma_w", "max_iter", "tol",
                 "blowup_factor", "layer_polylines"):
        if name in params:
            kwargs[name] = params[name]
    if "srf_range" in params:
        kwargs["srf_range"] = tuple(params["srf_range"])
    if n_gp is not None:
        kwargs["n_gp"] = int(n_gp)
    if "gwt" in params:
        kwargs["gwt"] = _normalize_gwt(params["gwt"])

    result = srm_mesh_refinement_study(
        [tuple(pt) for pt in params["surface_points"]],
        params["soil_layers"], meshes, **kwargs)
    return clean_result(result.to_dict())


def _run_analyze_local_fos(params: dict) -> dict:
    from fem2d import analyze_slope_srm
    _valid = ("surface_points", "soil_layers", "nx", "ny", "srf_tol",
              "n_load_steps", "t", "gamma_w", "max_iter", "tol",
              "element_type", "srm_field", "blowup_factor", "srf_range",
              "n_gp", "depth", "x_extend", "gwt", "layer_polylines",
              "local_fos_cap")
    reject_unknown_params(params, _valid, method="fem2d_local_fos")
    require_params(params, ["surface_points", "soil_layers"],
                   method="fem2d_local_fos", valid=_valid)
    _require_layer_elevations(params["soil_layers"], method="fem2d_local_fos")
    _require_positive(
        params.get("depth"), name="depth", method="fem2d_local_fos",
        hint=" Omit 'depth' for the 2*H default.")
    element_type = params.get("element_type", "t6")
    _require_choice(element_type, ("t6", "cst"),
                    name="element_type", method="fem2d_local_fos")
    # t, gamma_w, max_iter, tol and n_gp reach the analysis as they do for
    # fem2d_slope_srm, with the same defaults (BACKLOG GM-113)
    n_gp = params.get("n_gp")
    if n_gp is not None:
        _require_choice(int(n_gp), (3, 6),
                        name="n_gp", method="fem2d_local_fos")
    kwargs = dict(
        surface_points=[tuple(pt) for pt in params["surface_points"]],
        soil_layers=params["soil_layers"],
        nx=params.get("nx", 30), ny=params.get("ny", 15),
        srf_tol=params.get("srf_tol", 0.02),
        n_load_steps=params.get("n_load_steps", 2),
        t=params.get("t", 1.0),
        gamma_w=params.get("gamma_w", 9.81),
        max_iter=params.get("max_iter", 1000),
        tol=params.get("tol", 1e-5),
        n_gp=int(n_gp) if n_gp is not None else None,
        element_type=element_type,
        srm_field=params.get("srm_field", "c_phi"),
        blowup_factor=params.get("blowup_factor", 15.0),
        srf_range=tuple(params.get("srf_range", (0.5, 3.0))),
        compute_local_fos=True,
        local_fos_cap=params.get("local_fos_cap", 10.0),
    )
    if "depth" in params:
        kwargs["depth"] = params["depth"]
    if "x_extend" in params:
        kwargs["x_extend"] = params["x_extend"]
    if "gwt" in params:
        kwargs["gwt"] = _normalize_gwt(params["gwt"])
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]
    result = analyze_slope_srm(**kwargs)
    out = result.local_fos.to_dict()
    out["FOS"] = result.FOS
    out["fos_basis"] = getattr(result, "fos_basis", None)
    out["converged"] = result.converged
    out["warnings"] = list(result.warnings)
    return clean_result(out)


def run_fem2d_excavation(params: dict, *, method="fem2d_excavation"):
    """The fem2d_excavation tool's run: (the result object, what the calc package
    prints of the model beside it)."""
    from fem2d import analyze_excavation
    _valid = ("width", "depth", "wall_depth", "soil_layers", "wall_EI",
              "wall_EA", "nx", "ny", "t", "n_steps", "gamma_w", "max_iter",
              "tol", "gwt", "struts", "layer_polylines", "lifts",
              "wall_contact", "excavation_water", "water_level_inside",
              "interface_delta", "interface_ratio", "interface_adhesion")
    reject_unknown_params(params, _valid, method=method)
    require_params(params, ["width", "depth", "wall_depth", "soil_layers",
                            "wall_EI", "wall_EA"],
                   method=method, valid=_valid)
    _require_layer_elevations(params["soil_layers"], method=method,
                              valid=_EXCAVATION_LAYER_KEYS)
    check_nested_list(params.get("struts"), _STRUT_KEYS,
                      method=method, label="struts",
                      hints=_LAYER_KEY_HINTS)
    _require_positive(params.get("depth"), name="depth",
                      method=method)
    kwargs = dict(
        width=params["width"],
        depth=params["depth"],
        wall_depth=params["wall_depth"],
        soil_layers=params["soil_layers"],
        wall_EI=params["wall_EI"],
        wall_EA=params["wall_EA"],
        nx=params.get("nx", 30),
        ny=params.get("ny", 15),
        t=params.get("t", 1.0),
        n_steps=params.get("n_steps", 10),
        gamma_w=params.get("gamma_w", 9.81),
        max_iter=params.get("max_iter", 1000),
        tol=params.get("tol", 1e-5),
    )
    if "gwt" in params:
        kwargs["gwt"] = _normalize_gwt(params["gwt"])
    if "struts" in params:
        kwargs["struts"] = params["struts"]
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]
    if "lifts" in params:
        kwargs["lifts"] = params["lifts"]
    if "wall_contact" in params:
        _require_choice(params["wall_contact"],
                        ("smooth", "bonded", "interface"),
                        name="wall_contact", method=method)
        kwargs["wall_contact"] = params["wall_contact"]
    for name in ("interface_delta", "interface_ratio", "interface_adhesion"):
        if params.get(name) is not None:
            kwargs[name] = params[name]
    if "excavation_water" in params:
        _require_choice(params["excavation_water"],
                        ("dewatered", "full", "level"),
                        name="excavation_water", method=method)
        kwargs["excavation_water"] = params["excavation_water"]
    if params.get("water_level_inside") is not None:
        kwargs["water_level_inside"] = params["water_level_inside"]

    result = analyze_excavation(**kwargs)
    return result, _package_model(params["soil_layers"], params.get("gwt"))


def _run_analyze_excavation(params: dict) -> dict:
    return clean_result(run_fem2d_excavation(params)[0].to_dict())


def run_fem2d_seepage(params: dict, *, method="fem2d_seepage"):
    """The fem2d_seepage tool's run: (the result object, what the calc package
    prints of the model beside it)."""
    from fem2d import analyze_seepage
    _valid = ("nodes", "elements", "k", "head_bcs", "t", "gamma_w")
    reject_unknown_params(params, _valid, method=method)
    require_params(params, ["nodes", "elements", "k", "head_bcs"],
                   method=method, valid=_valid)
    nodes = np.array(params["nodes"])
    elements = np.array(params["elements"])
    k = params["k"]
    if isinstance(k, list):
        k = np.array(k)
    head_bcs = [(int(nid), float(val)) for nid, val in params["head_bcs"]]

    result = analyze_seepage(
        nodes=nodes,
        elements=elements,
        k=k,
        head_bcs=head_bcs,
        t=params.get("t", 1.0),
        gamma_w=params.get("gamma_w", 9.81),
    )
    return result, {}


def _run_analyze_seepage(params: dict) -> dict:
    return clean_result(run_fem2d_seepage(params)[0].to_dict())


def run_fem2d_consolidation(params: dict, *, method="fem2d_consolidation"):
    """The fem2d_consolidation tool's run: (the result object, what the calc package
    prints of the model beside it)."""
    from fem2d import analyze_consolidation
    _valid = ("width", "depth", "soil_layers", "k", "load_q", "time_points",
              "gwt", "gamma_w", "nx", "ny", "t", "n_w", "layer_polylines",
              "consolidation_scheme", "theta", "reset_displacements_after",
              "startup_steps")
    reject_unknown_params(params, _valid, method=method)
    require_params(params, ["width", "depth", "soil_layers", "k", "load_q",
                            "time_points"],
                   method=method, valid=_valid)
    _require_layer_elevations(params["soil_layers"],
                              method=method,
                              valid=_CONSOLIDATION_LAYER_KEYS)
    _require_positive(params.get("depth"), name="depth",
                      method=method)
    scheme = params.get("consolidation_scheme", "staggered")
    _require_choice(scheme, ("staggered", "monolithic"),
                    name="consolidation_scheme", method=method)
    kwargs = dict(
        width=params["width"],
        depth=params["depth"],
        soil_layers=params["soil_layers"],
        k=params["k"],
        load_q=params["load_q"],
        time_points=params["time_points"],
        gwt=params.get("gwt", 0.0),
        gamma_w=params.get("gamma_w", 9.81),
        nx=params.get("nx", 10),
        ny=params.get("ny", 20),
        t=params.get("t", 1.0),
        n_w=params.get("n_w", 2.2e6),
        consolidation_scheme=scheme,
        theta=params.get("theta", 1.0),
        startup_steps=params.get("startup_steps", 2),
    )
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]
    if params.get("reset_displacements_after") is not None:
        kwargs["reset_displacements_after"] = params["reset_displacements_after"]

    result = analyze_consolidation(**kwargs)
    return result, _package_model(params["soil_layers"], kwargs["gwt"])


def _run_analyze_consolidation(params: dict) -> dict:
    return clean_result(run_fem2d_consolidation(params)[0].to_dict())


def run_fem2d_staged(params: dict, *, method="fem2d_staged"):
    """The fem2d_staged tool's run: (the result object, what the calc package
    prints of the model beside it)."""
    from fem2d import (
        analyze_staged, ConstructionPhase,
        generate_rect_mesh, detect_boundary_nodes,
        assign_element_groups, create_wall_elements,
    )

    _valid = ("nodes", "elements", "material_props", "gamma",
              "element_groups", "phases", "t", "max_iter", "tol", "gamma_w",
              "beam_elements", "reset_displacements_after")
    reject_unknown_params(params, _valid, method=method)
    require_params(params, ["nodes", "elements", "material_props", "gamma",
                            "element_groups", "phases"],
                   method=method, valid=_valid)
    check_nested_list(params["material_props"], _MATERIAL_KEYS,
                      method=method, label="material_props",
                      hints=_LAYER_KEY_HINTS, notes=_LAYER_KEY_NOTES)
    check_nested_list(params["phases"], _PHASE_KEYS, method=method,
                      label="phases", hints=_LAYER_KEY_HINTS)
    check_nested_list(params.get("beam_elements"), _BEAM_KEYS,
                      method=method, label="beam_elements",
                      hints=_LAYER_KEY_HINTS)
    nodes = np.array(params["nodes"])
    elements = np.array(params["elements"])
    material_props = params["material_props"]
    gamma = params["gamma"]
    if isinstance(gamma, list):
        gamma = np.array(gamma)

    # Build bc_nodes from mesh
    bc_nodes = detect_boundary_nodes(nodes)

    # Element groups
    element_groups = params["element_groups"]

    # Build ConstructionPhase objects
    phases = []
    for pd in params["phases"]:
        gwt = _normalize_gwt(pd.get("gwt"))
        phases.append(ConstructionPhase(
            name=pd.get("name", "Phase"),
            active_soil_groups=pd.get("active_soil_groups", []),
            active_beam_ids=pd.get("active_beam_ids"),
            surface_loads=pd.get("surface_loads"),
            gwt=gwt,
            n_steps=pd.get("n_steps", 5),
            reset_displacements=pd.get("reset_displacements", False),
        ))

    kwargs = dict(
        nodes=nodes,
        elements=elements,
        material_props=material_props,
        gamma=gamma,
        bc_nodes=bc_nodes,
        element_groups=element_groups,
        phases=phases,
        t=params.get("t", 1.0),
        max_iter=params.get("max_iter", 500),
        tol=params.get("tol", 1e-5),
        gamma_w=params.get("gamma_w", 9.81),
        reset_displacements_after=params.get("reset_displacements_after"),
    )

    # Beam elements if provided
    if "beam_elements" in params:
        from fem2d.elements import BeamElement
        beam_elems = []
        for bd in params["beam_elements"]:
            require_keys(bd, ["node_i", "node_j", "EA", "EI"],
                         method=method, item_label="beam_elements[]")
            beam_elems.append(BeamElement(
                node_i=bd["node_i"],
                node_j=bd["node_j"],
                EA=bd["EA"],
                EI=bd["EI"],
                weight_per_m=bd.get("weight_per_m", 0.0),
            ))
        kwargs["beam_elements"] = beam_elems

    result = analyze_staged(**kwargs)
    return result, {}


def _run_analyze_staged(params: dict) -> dict:
    return clean_result(run_fem2d_staged(params)[0].to_dict())


def _run_hs_parameters(params: dict) -> dict:
    from fem2d.hs_correlations import estimate_hs_parameters_sand
    _valid = ("relative_density_pct",)
    reject_unknown_params(params, _valid, method="fem2d_hs_parameters")
    require_params(params, ["relative_density_pct"],
                   method="fem2d_hs_parameters", valid=_valid)
    return {k: round(v, 6) for k, v in
            estimate_hs_parameters_sand(params["relative_density_pct"]).items()}


METHOD_REGISTRY = {
    "fem2d_hs_parameters": _run_hs_parameters,
    "fem2d_gravity": _run_analyze_gravity,
    "fem2d_foundation": _run_analyze_foundation,
    "fem2d_footing_capacity": _run_analyze_footing_capacity,
    "fem2d_slope_srm": _run_analyze_slope_srm,
    "fem2d_srm_mesh_refinement_study": _run_srm_mesh_refinement_study,
    "fem2d_local_fos": _run_analyze_local_fos,
    "fem2d_excavation": _run_analyze_excavation,
    "fem2d_seepage": _run_analyze_seepage,
    "fem2d_consolidation": _run_analyze_consolidation,
    "fem2d_staged": _run_analyze_staged,
}

METHOD_INFO = {
    "fem2d_hs_parameters": {
        "category": "FEM 2D",
        "brief": "Estimate HS/HSsmall model parameters for sand from relative density (Brinkgreve et al. 2010).",
        "parameters": {
            "relative_density_pct": {"type": "float", "required": True, "description": "Relative density Dr in PERCENT (10-100)."},
        },
        "returns": {"E50_ref_kPa": "Reference secant stiffness (kPa).", "Eur_ref_kPa": "Unload-reload stiffness (kPa).", "phi_eff_deg": "Friction angle (deg).", "psi_deg": "Dilation angle (deg).", "m": "Stress-dependency exponent."},
    },
    "fem2d_gravity": {
        "category": "FEM 2D",
        "brief": "Elastic gravity analysis of a rectangular soil column (plane strain FEM).",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Domain width (m)."},
            "depth": {"type": "float", "required": True, "description": "Domain depth (m)."},
            "gamma": {"type": "float", "required": True, "description": "Unit weight (kN/m3)."},
            "E": {"type": "float", "required": True, "description": "Young's modulus (kPa)."},
            "nu": {"type": "float", "required": True, "description": "Poisson's ratio."},
            "nx": {"type": "int", "required": False, "default": 20, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Mesh divisions in y."},
            "t": {"type": "float", "required": False, "default": 1.0, "description": "Out-of-plane thickness of the plane-strain slice (m). Every result is per metre run and does not depend on it (the soil, walls, struts and joints all scale with it); leave it at 1."},
        },
        "returns": {
            "max_displacement_m": "Maximum displacement magnitude.",
            "max_sigma_yy_kPa": "Maximum vertical stress.",
            "converged": "Whether analysis converged.",
        },
    },
    "fem2d_foundation": {
        "category": "FEM 2D",
        "brief": "Elastic strip foundation analysis on a half-space (plane strain FEM).",
        "parameters": {
            "B": {"type": "float", "required": True, "description": "Foundation width (m)."},
            "q": {"type": "float", "required": True, "description": "Applied pressure (kPa, positive downward)."},
            "depth": {"type": "float", "required": True, "description": "Domain depth (m)."},
            "E": {"type": "float", "required": True, "description": "Young's modulus (kPa)."},
            "nu": {"type": "float", "required": True, "description": "Poisson's ratio."},
            "gamma": {"type": "float", "required": False, "default": 0.0, "description": "Soil unit weight (kN/m3)."},
            "nx": {"type": "int", "required": False, "default": 30, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 15, "description": "Mesh divisions in y."},
        },
        "returns": {
            "max_displacement_m": "Maximum displacement magnitude.",
            "max_sigma_yy_kPa": "Maximum vertical stress.",
        },
    },
    "fem2d_footing_capacity": {
        "category": "FEM 2D",
        "brief": "Ultimate bearing capacity of a strip footing by FEM (plane-strain, the FE analogue of a bearing-capacity calc): footing='flexible' (default) ramps a uniform pressure to collapse, validated vs Prandtl Nc=5.14; footing='rigid' pushes the footing down as one body (GEC-6 p. 55: Eq. 5-7 assumes a rigid footing) and reads q from the reactions, but the Prandtl clay check read high and fell as the mesh was refined (348, 306 kPa on uniform 20x10, 40x20, 288, 273 on graded ones, vs 257), and for c-phi and c = 0 no mesh-converged value was found. On a frictional soil the flexible q_ult is a lower bound that a larger max_iter can raise (a warning says so). Use element_type='t6' (CST locks and never collapses).",
        "parameters": {
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "c": {"type": "float", "required": True, "description": "Cohesion / undrained shear strength (kPa)."},
            "phi": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle (deg). Default 0 (Prandtl / undrained)."},
            "gamma": {"type": "float", "required": False, "default": 0.0, "description": "Soil unit weight (kN/m3). Applied in full (with any surcharge) before the footing is loaded, so the footing acts on soil already stressed by its own weight. Default 0 = classical weightless mechanism (the validated basis). With c = 0 and no surcharge the uniform footing pressure fails at its edges at a load set by the mesh (halved when the elements are halved): not a bearing capacity."},
            "E": {"type": "float", "required": False, "default": 1e5, "description": "Young's modulus (kPa)."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio."},
            "psi": {"type": "float", "required": False, "default": 0.0, "description": "Dilation angle (deg)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform surcharge q beside the footing (kPa), e.g. an embedded footing's overburden gamma*Df (GEC-6 Eq. 5-1). Applied on the ground surface: with the soil's weight over the whole surface first, then held beside the footing while the footing pressure rises from q, so it resists through the Nq mechanism. q_ult is the gross pressure."},
            "q_max": {"type": "float", "required": False, "description": "Top of the load ramp (kPa). Default 1.6x the closed-form estimate c*Nc+surcharge*Nq+0.5*gamma*B*Ngamma. Pass explicitly for a c=phi=0 footing."},
            "n_load_steps": {"type": "int", "required": False, "default": 45, "description": "Load increments; collapse-load resolution = (q_max - surcharge)/n_load_steps. Rigid: the number of equal settlement steps to settlement_max."},
            "nx": {"type": "int", "required": False, "default": 40, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 20, "description": "Mesh divisions in y."},
            "element_type": {"type": "str", "required": False, "default": "t6", "allowed_values": ["t6", "cst"], "description": "Soil element. t6 (recommended): collapse load within ~2% of Prandtl. cst LOCKS on the isochoric bearing mechanism and never collapses — do not use for capacity."},
            "q_applied": {"type": "float", "required": False, "description": "Working/design pressure (kPa). If given, bearing_FOS = q_ult/q_applied is reported."},
            "footing": {"type": "str", "required": False, "default": "flexible", "allowed_values": ["flexible", "rigid"], "description": "flexible (default): a uniform pressure ramped to collapse, the validated basis (Prandtl within ~2%). rigid: the footing's nodes pushed down together in n_load_steps settlement steps to settlement_max, q = their vertical reactions / B, q_ult the largest q reached. The default stays flexible on the mesh study (fem2d VALIDATION.md section 3): rigid reads high on these uniform meshes (Prandtl 348, 306 kPa on 20x10, 40x20 vs 257.1) and gives no plateau on a cohesionless soil."},
            "base": {"type": "str", "required": False, "default": "rough", "allowed_values": ["rough", "smooth"], "description": "Rigid footing only. rough (default): footing nodes held horizontally, no slip (GEC-6 pp. 49-50: the wedge under the footing moves down with it; pp. 99-100: concrete cast on the ground takes tan delta = tan phi, on clay c_a = c_u). smooth: free to slide."},
            "mesh": {"type": "str", "required": False, "default": "uniform", "allowed_values": ["uniform", "graded"], "description": "uniform (default): evenly spaced grid lines. graded: the same nx x ny elements graded towards the footing's edges and the ground surface (the elements there 1/4 the uniform size; this module's choice), nx even. Mesh study (fem2d VALIDATION.md section 3, B = 2 m): graded brings the rigid footing nearer Prandtl's 257.1 (c = 50: 287.8, 273.1, 267.9, 265.3 kPa on 20x10 to 80x40, against 347.9 to 283.1 uniform) and the flexible one to 255.9 already on 20x10. graded is for undrained clay (phi = 0) only and is refused when phi > 0: on frictional soils it read far lower (c = 20, phi = 25, flexible: 151.9 against 455.6 kPa uniform on 40x20). Not the default."},
            "max_iter": {"type": "int", "required": False, "description": "Iteration ceiling per load step. Default 1000 x max(1, nx*ny/800): 1000 on the default 40x20 mesh and coarser ones, more on finer ones (80x40: 4000), where the iteration converges more slowly (80x40 read 217.0 kPa at 1000 and 455.6, the 40x20 value, at 4000). On a frictional soil the flexible q_ult can rise with it (a warning says so)."},
            "settlement_max": {"type": "float", "required": False, "description": "Rigid footing only: the settlement the footing is pushed to (m). Default 10 q_est*B*(1-nu^2)/E, a settlement scale times 10, this module's choice (not printed in a reference), sized so q levelled off on the mesh study's cohesive cases; raise it when collapse_bracketed is False."},
        },
        "returns": {
            "q_ult_kPa": "Ultimate (gross) bearing pressure, including the surcharge: flexible, the last converged load level; rigid, the largest footing pressure reached.",
            "footing": "The footing model used ('flexible' or 'rigid').",
            "base": "Rigid: 'rough' or 'smooth'; null when flexible.",
            "mesh": "The mesh used ('uniform' or 'graded').",
            "n_steps_settled": "Flexible, when any: of the converged steps, those that reached max_iter with the displacements standing still (slow convergence, not collapse) and were accepted.",
            "pressure_settlement": "Rigid only: [[settlement_m, q_kPa], ...] after each converged step.",
            "settlement_max_m": "Rigid only: the settlement the footing was pushed to (m).",
            "choices": "The footing model, the mesh and, rigid, its base, each with how it was set and its basis.",
            "Nc_backfigured": "q_ult/c for a pure-cohesion footing (compare to Prandtl 5.14); null otherwise.",
            "bearing_FOS": "q_ult/q_applied when q_applied is given, else null.",
            "bearing_capacity_factors": "Closed-form {Nc, Nq, Ngamma} reference (Prandtl-Reissner + Vesic Ngamma).",
            "collapse_bracketed": "Flexible: False if the footing carried the full q_max (q_ult is then a lower bound — raise q_max). Rigid: True when q levelled off (rose < 1% over the last tenth of the settlement run); False means q_ult is a lower bound of this mesh's answer.",
            "warnings": "Flexible, on a cohesionless soil with no surcharge: q_ult is set by the mesh and is not a bearing capacity to design from. Flexible, phi > 0, when the run stopped at max_iter: q_ult is a lower bound of this mesh's answer and can rise with max_iter. Rigid: always, that on the Prandtl clay check q_ult read high and fell as the mesh was refined and that c-phi and cohesionless soils have no mesh-converged value; and when q did not level off (raise n_load_steps or max_iter if a step failed to converge, settlement_max if q was still rising). Quote them with q_ult.",
        },
    },
    "fem2d_slope_srm": {
        "category": "FEM 2D",
        "brief": "Slope stability FOS via Strength Reduction Method (plane strain FEM).",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,z],...] array."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of dicts with 'name','bottom_elevation','E','nu','c','phi','psi','gamma'."},
            "depth": {"type": "float", "required": False, "description": "Depth below lowest surface (m). Default the larger of 2*H and 10 m."},
            "nx": {"type": "int", "required": False, "default": 30, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 15, "description": "Mesh divisions in y."},
            "x_extend": {"type": "float", "required": False, "description": "Horizontal margin (m): a number adds 30 % of it on each side, a [left, right] pair gives each side in metres. Default 0.3 max(0.5*H, depth, 5) each side; over undrained (phi = 0) soil below the toe the level ground is carried to twice that soil's depth each side, with nx raised to keep the element size (Taylor's deep circle; the result's notes say so). Pass 0 when the profile already has margins."},
            "srf_tol": {"type": "float", "required": False, "default": 0.02, "description": "SRF bisection tolerance (0.01 for benchmark-grade runs)."},
            "mesh_check": {"type": "bool", "required": False, "default": False, "description": "Also run a mesh 1.5x finer each way and report both FOS (mesh_check), warning when they differ by more than 5 %. About 3x the run time."},
            "gwt": {"type": "float|array", "required": False, "description": "Groundwater table: float elevation, or [[x,y],...] polyline."},
            "element_type": {"type": "str", "required": False, "default": "t6", "allowed_values": ["t6", "cst"], "description": "Soil element. t6 (quadratic, recommended): collapse loads within a few % of published benchmarks. cst locks (overpredicts FOS) - elastic work only."},
            "srm_field": {"type": "str", "required": False, "default": "c_phi", "allowed_values": ["c_phi", "c", "phi"], "description": "Strengths reduced by the SRF: both (default), cohesion only, or friction only."},
            "blowup_factor": {"type": "float", "required": False, "default": 15.0, "description": "Secondary failure criterion: dimensionless-displacement blowup threshold (x the value at the lowest stable SRF). null disables (pure Griffiths-Lane non-convergence)."},
            "srf_range": {"type": "array", "required": False, "default": [0.5, 3.0], "description": "[min, max] SRF search range."},
            "n_gp": {"type": "int", "required": False, "allowed_values": [3, 6], "description": "T6 Gauss rule override (default 3)."},
            "max_iter": {"type": "int", "required": False, "default": 1000, "description": "Iteration ceiling per load step (Griffiths-Lane non-convergence criterion)."},
            "layer_polylines": {"type": "dict", "required": False, "description": "{layer_name: [[x,y],...]} bottom-boundary polylines for non-horizontal layering."},
        },
        "returns": {
            "FOS": "Factor of safety from SRM.",
            "fos_basis": "Failure criterion that fixed the FOS: nonconvergence | blowup | range_exhausted (= FOS capped at srf_range max).",
            "n_srf_trials": "Number of SRF trials run.",
            "srf_curve": "SRF vs dimensionless displacement E*dmax/(gamma*H^2) for converged trials (plotting / failure-onset review).",
            "max_displacement_m": "Maximum displacement at the last stable SRF.",
            "converged": "Whether SRM bracketing succeeded.",
            "warnings": "Plain-language cautions: slope covered by too few elements, search range that did not bracket failure, finer mesh that disagrees, side boundaries inside the deep circle of an undrained foundation (extend the level ground), an undrained layer with Poisson's ratio below 0.45 (use 0.49), a soil layer with no elements, a cohesionless face above tan(phi)/tan(beta). Quote them with the factor of safety.",
            "mesh_check": "With mesh_check: {mesh, FOS, finer_mesh, FOS_finer_mesh, change_pct}.",
            "notes": "What the model did by default, e.g. the level ground carried wider (and nx raised) over a deep undrained foundation.",
        },
    },
    "fem2d_srm_mesh_refinement_study": {
        "category": "FEM 2D",
        "brief": "Slope SRM factor of safety on a sequence of meshes (coarse to fine): per-mesh table, change between meshes, Richardson estimate, convergence flag. Runs fem2d_slope_srm once per mesh.",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,z],...] array (as fem2d_slope_srm)."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of dicts with 'name','bottom_elevation','E','nu','c','phi','psi','gamma' (as fem2d_slope_srm)."},
            "meshes": {"type": "array", "required": True, "description": "Mesh densities [[nx, ny], ...], coarse to fine; at least two, three or more give the Richardson estimate. Each is one SRM run, so the run time is their sum."},
            "published": {"type": "float", "required": False, "description": "Reference or published FOS, used only for the per-mesh error column."},
            "conv_tol": {"type": "float", "required": False, "default": 0.03, "description": "Relative FOS change between the two finest meshes below which the study reads converged."},
            "depth": {"type": "float", "required": False, "description": "Depth below lowest surface (m); held for every mesh. Default as fem2d_slope_srm."},
            "x_extend": {"type": "float", "required": False, "description": "Horizontal margin (m); held for every mesh. Default as fem2d_slope_srm."},
            "element_type": {"type": "str", "required": False, "default": "t6", "allowed_values": ["t6", "cst"], "description": "Soil element, held for every mesh (t6 recommended)."},
            "srm_field": {"type": "str", "required": False, "default": "c_phi", "allowed_values": ["c_phi", "c", "phi"], "description": "Strengths reduced by the SRF, held for every mesh."},
            "srf_tol": {"type": "float", "required": False, "default": 0.02, "description": "SRF bisection tolerance for every mesh."},
            "gwt": {"type": "float|array", "required": False, "description": "Groundwater table: float elevation, or [[x,y],...] polyline."},
            "blowup_factor": {"type": "float", "required": False, "default": 15.0, "description": "Dimensionless-displacement blowup threshold, as fem2d_slope_srm."},
            "srf_range": {"type": "array", "required": False, "default": [0.5, 3.0], "description": "[min, max] SRF search range."},
            "n_gp": {"type": "int", "required": False, "allowed_values": [3, 6], "description": "T6 Gauss rule override (default 3)."},
            "max_iter": {"type": "int", "required": False, "default": 1000, "description": "Iteration ceiling per load step."},
            "layer_polylines": {"type": "dict", "required": False, "description": "{layer_name: [[x,y],...]} bottom-boundary polylines for non-horizontal layering."},
        },
        "returns": {
            "levels": "Mesh table, coarse to fine: nx, ny, n_elements, n_nodes, FOS, fos_basis, converged (bracketing), n_srf_trials, wall_time_s, rel_change (from the previous mesh), error_vs_published.",
            "fos_finest": "FOS on the finest mesh.",
            "fos_coarsest": "FOS on the coarsest mesh.",
            "fos_richardson": "Mesh-independent estimate from the three finest meshes, or null when they are not monotonic, fewer than three, or the observed order is out of range.",
            "observed_order": "Observed order of convergence used for the extrapolation.",
            "fos_estimate": "The Richardson value where there is one, else the finest-mesh FOS.",
            "converged": "True when the change between the two finest meshes is below conv_tol.",
            "conv_tol": "Tolerance used for converged.",
            "published": "The reference FOS, if given.",
            "element_type": "Echo of the element used for every mesh.",
            "srm_field": "Echo of the strengths reduced for every mesh.",
        },
    },
    "fem2d_local_fos": {
        "category": "FEM 2D",
        "brief": "Local (pointwise) factor-of-safety map for a slope: runs the SRM then evaluates local_FOS = Mohr-Coulomb available shear / mobilized shear at each element using the ORIGINAL strengths. At the critical SRF the minimum local FOS ~ the global SRM FOS and the low-FOS band is the failure mass. Returns summary stats (the full field is for plotting).",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,z],...] array."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of dicts with 'name','bottom_elevation','E','nu','c','phi','psi','gamma' (same as fem2d_slope_srm)."},
            "depth": {"type": "float", "required": False, "description": "Depth below lowest surface (m). Default the larger of 2*H and 10 m."},
            "nx": {"type": "int", "required": False, "default": 30, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 15, "description": "Mesh divisions in y."},
            "x_extend": {"type": "float", "required": False, "description": "Horizontal margin (m). Pass 0 when the profile already has margins."},
            "srf_tol": {"type": "float", "required": False, "default": 0.02, "description": "SRF bisection tolerance."},
            "element_type": {"type": "str", "required": False, "default": "t6", "allowed_values": ["t6", "cst"], "description": "Soil element. t6 recommended (cst locks)."},
            "srm_field": {"type": "str", "required": False, "default": "c_phi", "allowed_values": ["c_phi", "c", "phi"], "description": "Strengths reduced by the SRF."},
            "n_gp": {"type": "int", "required": False, "allowed_values": [3, 6], "description": "T6 Gauss rule override (default 3)."},
            "max_iter": {"type": "int", "required": False, "default": 1000, "description": "Iteration ceiling per load step (Griffiths-Lane non-convergence criterion)."},
            "local_fos_cap": {"type": "float", "required": False, "default": 10.0, "description": "Upper cap on local FOS where the mobilized shear is ~0 (deep isotropic stress)."},
        },
        "returns": {
            "min_local_fos": "Minimum local factor of safety (~ the global FOS at the critical SRF).",
            "min_location_xy": "[x, y] of the minimum-FOS point (on the slip surface, typically the toe).",
            "global_fos": "The global SRM factor of safety.",
            "frac_below_1": "Fraction of elements with local FOS < 1 (inadmissible).",
            "frac_below_1_5": "Fraction with local FOS < 1.5 (extent of the near-critical mobilized mass).",
            "FOS": "Global SRM FOS (alias of global_fos).",
        },
    },
    "fem2d_excavation": {
        "category": "FEM 2D",
        "brief": "Staged excavation in front of an embedded wall (plane strain FEM): in-situ K0 stresses, then lifts with struts installed as the excavation reaches them; half of a symmetric excavation.",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Full excavation width (m); the model is half of a symmetric excavation, its right boundary the centre line. For a one-sided cut give at least twice the passive zone in front of the wall."},
            "depth": {"type": "float", "required": True, "description": "Final excavation depth (m)."},
            "wall_depth": {"type": "float", "required": True, "description": "Wall tip depth below the ground (m); must exceed depth."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of soil property dicts (E, nu, c, phi, psi, gamma = total unit weight, bottom_elevation; optional K0, else Jaky 1 - sin phi'; optional k = hydraulic conductivity m/s for the seepage of a dewatered excavation, every layer or none: a small k below the tip models a wall cut off into an impervious layer)."},
            "wall_EI": {"type": "float", "required": True, "description": "Wall flexural stiffness (kN*m2/m)."},
            "wall_EA": {"type": "float", "required": True, "description": "Wall axial stiffness (kN/m)."},
            "struts": {"type": "array", "required": False, "description": "Array of {depth (m, above the final excavation level), stiffness (kN/m per m)}; each installed stress-free once the excavation reaches its level; no preload."},
            "lifts": {"type": "array", "required": False, "description": "Excavation depths (m) at the end of each lift; default one lift to each strut level, then to depth."},
            "wall_contact": {"type": "str", "required": False, "default": "smooth", "allowed_values": ["smooth", "bonded", "interface"], "description": "'smooth' (default): no wall friction (Rankine). 'bonded': the wall shares the soil's nodes and hangs the retained soil (thrust below active). 'interface': wall friction delta between 0 and phi' (Mohr-Coulomb joints between wall and soil); give interface_delta or interface_ratio."},
            "interface_delta": {"type": "float", "required": False, "description": "For wall_contact='interface': the wall friction angle delta (deg, 0-45), the same on every layer. UFC 3-220-20 (2025) Table 4-1, p. 184: steel sheet piles against clean sand 17, clean gravel 22, silty sand 14; formed concrete against clean sand 17-22. Give this or interface_ratio."},
            "interface_ratio": {"type": "float", "required": False, "description": "For wall_contact='interface', in place of interface_delta: delta/phi' (0-1) of the layer at each joint."},
            "interface_adhesion": {"type": "float", "required": False, "default": 0.0, "description": "For wall_contact='interface': adhesion c_a (kPa) on the wall (UFC 3-220-20 Table 4-1 gives it for clayey soils by undrained strength). Default 0 (friction only)."},
            "gwt": {"type": "float|array", "required": False, "description": "Groundwater table elevation (m, the ground is 0, negative below it): the in-situ table, held outside the excavation. What happens inside where it is above the excavation level is excavation_water."},
            "excavation_water": {"type": "str", "required": False, "default": "dewatered", "allowed_values": ["dewatered", "full", "level"], "description": "Water inside the excavation where the table is above the excavation level. 'dewatered' (default, the construction case: UFC 3-220-20 p. 97, groundwater control below the table; Caltrans T&S Example 7-2): lowered to the excavation level at each lift, steady seepage under the wall's tip, the wall carries the net water pressure (GEC 4 sec. 5.2.9). 'full': kept full of water to the table (dug in the wet; the water on the two faces balances). 'level': lowered to water_level_inside."},
            "water_level_inside": {"type": "float", "required": False, "description": "For excavation_water='level': depth (m below the ground) of the water inside the excavation, between the table and the excavation depth."},
            "n_steps": {"type": "int", "required": False, "default": 10, "description": "Load steps in each stage."},
            "max_iter": {"type": "int", "required": False, "default": 1000, "description": "Iteration ceiling per load step."},
            "layer_polylines": {"type": "dict", "required": False, "description": "{layer_name: [[x,y],...]} bottom-boundary polylines for non-horizontal layering."},
        },
        "returns": {
            "max_displacement_m": "Largest movement since the in-situ stage.",
            "max_beam_moment_kNm_per_m": "Largest wall bending moment at the end of the last stage only.",
            "max_beam_shear_kN_per_m": "Largest wall shear force at the end of the last stage only.",
            "wall_envelope": "Largest wall moment and shear over ALL stages, with the stage each occurs in, its depth below the ground (moment_depth_m, shear_depth_m) and whether that is above or below the excavation level: the values to design the wall for (an earlier lift, before a strut is installed, can govern). Quote these depths; do not infer them.",
            "strut_forces": "Per strut: force = stiffness x movement since installation (positive holds the wall back).",
            "excavation_stages": "Per lift: wall head and largest wall u_x (positive toward the excavation), M_max, V_max, strut forces; with a water table also the water inside, the pore pressure and total head at the wall's tip, the net water pressure at the excavation level, the upward gradient in front of the wall and the critical gradient.",
            "excavation_water": "The water condition modelled inside the excavation.",
            "wall_water_pressures": "Pore pressure on each face of the wall and the net, per wall node, at the end of the last stage.",
            "notes": "Method, K0 per layer, sign conventions; the water condition with its basis and the net water pressure beside the no-seepage and GEC 4 values.",
            "wall_interface": "With wall_contact='interface': per face (retained side, excavation side) the effective normal thrust and the soil's shear on the wall (+ up) summed over the joints, the wall friction angle mobilized and how many joints slip.",
            "choices": "The wall contact and, with 'interface', the wall friction and adhesion; then for each soil layer its soil model (Mohr-Coulomb, Hardening Soil or linear elastic) and the K0 of its in-situ stresses (the layer's own, Jaky 1 - sin phi', or nu/(1-nu) for phi = 0); each with how it was set and its basis.",
        },
    },
    "fem2d_seepage": {
        "category": "FEM 2D",
        "brief": "Steady-state seepage analysis (Laplace equation, CST elements).",
        "parameters": {
            "nodes": {"type": "array", "required": True, "description": "Node coordinates [[x,y],...]."},
            "elements": {"type": "array", "required": True, "description": "CST connectivity [[n1,n2,n3],...]."},
            "k": {"type": "float|array", "required": True, "description": "Hydraulic conductivity (m/s). Scalar or per-element."},
            "head_bcs": {"type": "array", "required": True, "description": "Dirichlet BCs [[node_id, head_value],...]."},
            "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Unit weight of water (kN/m3)."},
            "t": {"type": "float", "required": False, "default": 1.0, "description": "Out-of-plane thickness of the plane-strain slice (m). No result depends on it: the heads and pore pressures do not, and the flow is reported per metre run. Leave it at 1."},
        },
        "returns": {
            "max_head_m": "Maximum total head.",
            "min_head_m": "Minimum total head.",
            "flow_m3_per_s_per_m": "The flow through the section per metre run (m3/s/m): the inflow across the fixed-head boundaries, summed from the nodal reactions at the nodes where water enters. Boundaries with no head given are impervious.",
            "outflow_m3_per_s_per_m": "The flow leaving across the fixed-head boundaries (m3/s/m).",
            "flow_balance": "Inflow against outflow: the mismatch |in - out| / in, its tolerance (1e-6) and whether the check passed.",
            "flow_boundaries": "Per prescribed head value: the head (m), its node count and the net flow entering there (m3/s/m; negative = leaving).",
            "flow_measured_on": "Which boundary set the flow is measured on.",
        },
    },
    "fem2d_consolidation": {
        "category": "FEM 2D",
        "brief": "Coupled Biot consolidation analysis (settlement + pore pressure dissipation).",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Domain width (m)."},
            "depth": {"type": "float", "required": True, "description": "Domain depth (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "Soil property dicts, each with 'E' (kPa), 'nu', 'gamma' (kN/m3) AND 'bottom_elevation' (m; 'top_elevation' optional, defaults to the domain top / the layer above). Add 'fill': true to a layer PLACED during construction: its self-weight then moves from the initial stage into the load stage (monolithic: applied undrained with the load, so it consolidates). Fill layers must be the top layers; they may be placed under water (gwt above their base): below gwt a fill loads the soil by its buoyant unit weight gamma - gamma_w, the water standing on the soil in place before placement being part of the initial stage."},
            "k": {"type": "float", "required": True, "description": "Hydraulic conductivity (m/s) for 'staggered'; for 'monolithic' pass the MOBILITY m^2/(kPa.s)."},
            "load_q": {"type": "float", "required": True, "description": "Surface load (kPa, positive downward)."},
            "time_points": {"type": "array", "required": True, "description": "OUTPUT times since the load is applied (seconds, >= 0). t = 0 (the loading instant) is always reported first; U and settlement come back at each of these times. The monolithic scheme integrates on a finer internal schedule between them, so a sparse list (e.g. [1e3, 1e6, 1e7]) is fine."},
            "gwt": {"type": "float", "required": False, "default": 0.0, "description": "GWT elevation (m)."},
            "n_w": {"type": "float", "required": False, "default": 2.2e6, "description": "For 'monolithic': the Biot modulus M (kPa) (~2.2e6 for near-incompressible water). Governs the undrained load response p0. Ignored by 'staggered'."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Mesh divisions in x (horizontal)."},
            "ny": {"type": "int", "required": False, "default": 20, "description": "Mesh divisions in y (vertical)."},
            "t": {"type": "float", "required": False, "default": 1.0, "description": "Out-of-plane thickness of the plane-strain slice (m). Every result is per metre run and does not depend on it (the soil, walls, struts and joints all scale with it); leave it at 1."},
            "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Unit weight of water (kN/m3)."},
            "layer_polylines": {"type": "array", "required": False, "description": "Optional explicit layer-boundary polylines [[[x,z],...],...] for non-horizontal layering."},
            "consolidation_scheme": {"type": "string", "required": False, "default": "staggered",
                                     "allowed_values": ["staggered", "monolithic"],
                                     "description": "Biot solver: 'staggered' (default, sequential split; drained at every step, so U = 1 and NO consolidation transient under the load) or 'monolithic' (coupled u-p, Taylor-Hood T6/T3, reproduces the load-induced undrained response p0 and the Terzaghi transient; use it for U and settlement versus time). For 'monolithic', pass k as the MOBILITY m^2/(kPa.s) and n_w as the Biot modulus M (kPa)."},
            "theta": {"type": "float", "required": False, "default": 1.0,
                      "description": "Time-integration parameter for the monolithic scheme (1.0 backward Euler; 0.5 Crank-Nicolson, more accurate). Range [0.5, 1.0]."},
            "startup_steps": {"type": "int", "required": False, "default": 2,
                              "description": "Monolithic scheme with theta < 1: the first startup_steps time steps are each taken as two fully implicit half-steps (Rannacher start-up), so Crank-Nicolson does not ring at the drained boundary. 0 = plain theta-stepping from the first step."},
            "reset_displacements_after": {"type": "string", "required": False,
                                          "allowed_values": ["initial", "start"], "value_aliases": {"first": "initial", "0": "initial", "none": "start"}, "any_case": "strip",
                                          "description": "Which stage the reported settlements are measured from. The analysis has two stages: 0 'initial' (self-weight of the soil in place, drained, with the hydrostatic water table and any ponded water) and 1 'load' (load_q + any 'fill' layers, then consolidation). 'initial' = from the end of the initial stage, i.e. the load stage alone (common FE practice; the DEFAULT when omitted). 'start' = cumulative, self-weight settlement included. Omitting it lets the default apply and the result then carries a 'judgment' record if the choice matters."},
        },
        "returns": {
            "n_nodes": "Mesh nodes.",
            "n_elements": "Mesh elements.",
            "n_time_steps": "Number of reported times (len(time_s)).",
            "converged": "Whether analysis converged.",
            "max_settlement_m": "Maximum surface settlement (m, most negative = downward), measured from displacement_reference (default: the load stage alone).",
            "max_excess_pore_pressure_kPa": "Peak excess pore pressure over the run (kPa).",
            "degree_of_consolidation": "Degree of consolidation U (0-1) at the final time.",
            "time_s": "The reported times (s since loading): 0 first, then each requested time_points value.",
            "degree_of_consolidation_by_time": "U (0-1) at each time in time_s, same order.",
            "surface_settlement_m_by_time": "Surface settlement (m, negative = down) at each time in time_s, measured from displacement_reference (default: the load stage alone), in both schemes.",
            "max_excess_pore_pressure_kPa_by_time": "Peak |excess pore pressure| (kPa) at each time in time_s.",
            "final_drained_settlement_m": "Monolithic: the drained end-state settlement of the LOAD stage (m), the U = 1 reference.",
            "displacement_reference": "{reset_after_stage, measured_from, chosen_by ('default' or 'user')}: the stage the reported settlements start from.",
            "stages": "Per construction stage: [0] 'initial' (self-weight) and [1] 'load' — each with surface_settlement_m (this stage's own) and cumulative_surface_settlement_m, delta_displacement / cumulative_displacement extremes {settlement_m (negative = down), heave_m, horizontal_m, total_m, and *_at_xy}; the load stage also per time: surface_settlement_m_by_time (this stage), cumulative_surface_settlement_m_by_time, degree_of_consolidation_by_time.",
            "judgment": "Present only when the default reference stage was applied AND it changes the reported settlement materially: {question, options [{name, source, assumptions, applies, result}], used, why} — the self-weight movement counts or not depending on the construction sequence; say which basis the reported number uses.",
            "scheme": "'staggered' or 'monolithic'.",
            "notes": "Basis and caveats: the time origin, gwt as an elevation, the displacement reference, what the excess pore pressure includes, the early-time mesh limit; for 'staggered', that it is drained at every step (U = 1, no transient).",
            "choices": "The water table elevation and where it lies, the fill's unit weight (buoyant below the water table), and with theta < 1 the time-stepping start-up, each with how it was set and its basis.",
        },
    },
    "fem2d_staged": {
        "category": "FEM 2D",
        "brief": "Staged construction analysis (multi-phase, activate/deactivate soil groups and beams); reports each stage's own movement, the cumulative movement and the movement since a reference stage.",
        "parameters": {
            "nodes": {"type": "array", "required": True, "description": "Node coordinates [[x,y],...]."},
            "elements": {"type": "array", "required": True, "description": "Element connectivity [[n1,n2,n3],...]."},
            "material_props": {"type": "array", "required": True, "description": "Per-element material dicts."},
            "gamma": {"type": "float|array", "required": True, "description": "Unit weight. Scalar or per-element."},
            "element_groups": {"type": "dict", "required": True, "description": "Group name -> list of element indices."},
            "phases": {"type": "array", "required": True, "description": "Array of phase dicts: name, active_soil_groups, active_beam_ids, surface_loads [[edges, qx, qy]], gwt, n_steps, reset_displacements (true = report this phase's movements from its START, PLAXIS 'reset displacements to zero'; reporting only). Phase 0 is normally the initial (gravity) stage; soil PLACED during construction (fill) is a group first activated in a later phase, so its self-weight movement falls in that phase."},
            "beam_elements": {"type": "array", "required": False, "description": "Array of beam element dicts."},
            "reset_displacements_after": {"type": "int|string", "required": False,
                                          "description": "The stage at whose END displacement_since_reference is zeroed: a stage index, a stage name, 'initial' (= stage 0) or 'start' (no reset: cumulative). DEFAULT when omitted (and no phase sets reset_displacements): after stage 0, taken as the initial gravity/K0 stage, so construction movements exclude self-weight (common FE practice); the result then carries a 'judgment' record if the choice matters."},
        },
        "returns": {
            "n_phases": "Number of phases completed.",
            "converged": "Whether all phases converged.",
            "phases": "Per-phase results. max_displacement_m/_x_m/_y_m are CUMULATIVE from the start (gravity included). delta_displacement = this stage's own movement; cumulative_displacement = from the start; displacement_since_reference = from reference_stage. Each is {settlement_m (most negative uy, negative = down), heave_m (most positive uy), horizontal_m (ux of largest magnitude, signed +x), total_m (largest |u|), and *_at_xy}, over the nodes of the stage's active elements.",
            "displacement_reference": "{reset_after_stages, measured_from, chosen_by ('default' or 'user')}: where displacement_since_reference starts.",
            "judgment": "Present only when the default reference stage was applied AND it changes the reported movement materially: {question, options [{name, source, assumptions, applies, result}], used, why}. Say which basis a reported movement uses.",
            "notes": "Basis of each displacement key.",
        },
    },
}


# ---------------------------------------------------------------------------
# METHOD_INFO parameter backfill (eval CON-1 class)
# ---------------------------------------------------------------------------
# Several handlers accept optional controls (Biot modulus n_w, mesh nx/ny,
# thickness t, unit weight of water gamma_w, Newton max_iter/tol, SRM knobs)
# that were historically absent from their METHOD_INFO ``parameters`` schema, so
# describe_method did not advertise them and the agent could not discover how to
# set them (the CON-1 miss: no discoverable n_w → wrong Biot modulus). This
# backfill declares the missing-but-accepted params so they are discoverable.
# Docs-only: handler behavior (which already accepts these) is unchanged; a
# ``setdefault`` never overrides a hand-written declaration.
_FEM_PARAM_DECLS = {
    "t": {"type": "float", "required": False, "default": 1.0,
          "description": "Out-of-plane thickness of the plane-strain slice "
          "(m). Every result is per metre run and does not depend on it "
          "(the soil, walls, struts and joints all scale with it); leave it "
          "at 1."},
    "gamma_w": {"type": "float", "required": False, "default": 9.81,
                "description": "Unit weight of water (kN/m3)."},
    "nx": {"type": "int", "required": False,
           "description": "Mesh divisions in x (horizontal)."},
    "ny": {"type": "int", "required": False,
           "description": "Mesh divisions in y (vertical)."},
    "max_iter": {"type": "int", "required": False,
                 "description": "Max Newton-Raphson iterations per load step "
                 "(raise for a hard-to-converge model)."},
    "tol": {"type": "float", "required": False,
            "description": "Convergence tolerance (residual norm)."},
    "n_load_steps": {"type": "int", "required": False,
                     "description": "Number of load increments applied before "
                     "the analysis / SRF sweep."},
    "domain_depth": {"type": "float", "required": False,
                     "description": "FE domain depth below the footing (m). "
                     "Positive; default auto-sized."},
    "domain_half_width": {"type": "float", "required": False,
                          "description": "FE domain half-width (m). Positive; "
                          "default auto-sized."},
    "gwt": {"type": "float|array", "required": False,
            "description": "Groundwater table: scalar elevation (m) or "
            "[[x,z],...] polyline."},
    "srf_range": {"type": "array", "required": False,
                  "description": "[min, max] strength-reduction-factor search "
                  "bracket (default (0.5, 3.0))."},
    "blowup_factor": {"type": "float", "required": False, "default": 15.0,
                      "description": "Displacement blow-up factor that flags "
                      "SRM non-convergence."},
    "n_gp": {"type": "int", "required": False, "allowed_values": [3, 6],
             "description": "Gauss points per T6 element (3 or 6)."},
    "layer_polylines": {"type": "array", "required": False,
                        "description": "Explicit layer-boundary polylines for "
                        "non-horizontal layering."},
}
_FEM_PARAM_BACKFILL = {
    "fem2d_foundation": ["t"],
    "fem2d_footing_capacity": ["domain_depth", "domain_half_width", "tol"],
    "fem2d_slope_srm": ["gamma_w", "n_load_steps", "t", "tol"],
    "fem2d_srm_mesh_refinement_study": ["gamma_w", "n_load_steps", "t", "tol"],
    "fem2d_local_fos": ["blowup_factor", "gamma_w", "gwt", "layer_polylines",
                        "n_load_steps", "srf_range", "t", "tol"],
    "fem2d_excavation": ["gamma_w", "max_iter", "nx", "ny", "t", "tol"],
    "fem2d_seepage": ["t"],
    "fem2d_staged": ["gamma_w", "max_iter", "t", "tol"],
}
for _m, _names in _FEM_PARAM_BACKFILL.items():
    _pp = METHOD_INFO[_m]["parameters"]
    for _n in _names:
        _pp.setdefault(_n, dict(_FEM_PARAM_DECLS[_n]))

# When each default applies (GM-119): the Biot modulus and the time
# integration parameter to the monolithic consolidation scheme.
_WATER = [{"gwt": "set"}, {"gwt": [0]}]
declare_applies_when(METHOD_INFO, {
    # GM-305: the load ramp is the flexible footing's, the base and the
    # settlement run the rigid footing's
    "fem2d_footing_capacity": {
        "q_max": {"footing": ["flexible"]},
        "base": {"footing": ["rigid"]},
        "settlement_max": {"footing": ["rigid"]}},
    "fem2d_consolidation": {
        name: {"consolidation_scheme": ["monolithic"]}
        for name in ("n_w", "theta")} | {
        "startup_steps": {"consolidation_scheme": ["monolithic"],
                          "theta": [0.5]}},
    # the wall's joints (GM-72); since GM-184 the water's unit weight, and
    # the excavation's water inside, only with a water table ("set" is
    # not 0, so a table at the ground surface is named)
    "fem2d_excavation": {
        "interface_adhesion": {"wall_contact": ["interface"]},
        "gamma_w": _WATER, "excavation_water": _WATER},
    "fem2d_local_fos": {"gamma_w": _WATER},
    "fem2d_slope_srm": {"gamma_w": _WATER},
    "fem2d_srm_mesh_refinement_study": {"gamma_w": _WATER}})

# The values the handlers pass when these are left out (BACKLOG GM-165),
# and what a left-out input with no default stands for: the load ramp,
# the model's extent, the water table, the Gauss rule, the lifts and the
# stage the reported movements are measured from.
for _m, _defaults in {
        "fem2d_footing_capacity": {"tol": 1e-4},
        "fem2d_slope_srm": {"n_load_steps": 2, "tol": 1e-5},
        "fem2d_srm_mesh_refinement_study": {"n_load_steps": 2, "tol": 1e-5},
        "fem2d_local_fos": {"n_load_steps": 2, "srf_range": [0.5, 3.0],
                            "max_iter": 1000, "tol": 1e-5},
        "fem2d_excavation": {"nx": 30, "ny": 15, "tol": 1e-5},
        "fem2d_staged": {"max_iter": 500, "tol": 1e-5}}.items():
    for _n, _v in _defaults.items():
        METHOD_INFO[_m]["parameters"][_n]["default"] = _v
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_WATER = ("no water table", "no pore water pressure")
_DEPTH = ("the larger of 2 H and 10 m",
          "the model's depth below the lowest ground surface")
_X_EXTEND = ("0.3 max(0.5 H, depth, 5 m) each side",
             "the model's margin beyond the slope; wider over deep undrained "
             "soil, to reach Taylor's deep circle")
_GAUSS = ("the 3-point Gauss rule", "for T6 elements")
declare_when_absent(METHOD_INFO, {
    "fem2d_footing_capacity": {
        "q_max": ("1.6 times the closed-form estimate",
                  "the load ramp tops at 1.6 (c Nc + q Nq + 0.5 gamma B "
                  "Ngamma)"),
        "domain_depth": ("5 B", "the mesh extends five footing widths "
                         "below the footing"),
        "domain_half_width": ("5 B", "the mesh extends five footing widths "
                              "to the side"),
        "max_iter": ("1000 x max(1, nx ny / 800) iterations a load step",
                     "1000 on the default 40x20 mesh, scaled with the "
                     "element count on finer meshes, where the iteration "
                     "converges more slowly (80x40: 4000); this module's "
                     "choice (fem2d VALIDATION.md section 3)"),
        "settlement_max": ("10 q_est B (1 - nu^2) / E",
                           "a settlement scale times 10, this module's "
                           "choice (not printed), sized so q levelled off "
                           "on the mesh study's cohesive cases; raise it "
                           "when collapse_bracketed is False")},
    "fem2d_slope_srm": {"depth": _DEPTH, "x_extend": _X_EXTEND,
                        "gwt": _NO_WATER, "n_gp": _GAUSS},
    "fem2d_srm_mesh_refinement_study": {
        "depth": _DEPTH, "x_extend": _X_EXTEND, "gwt": _NO_WATER,
        "n_gp": _GAUSS},
    "fem2d_local_fos": {"depth": _DEPTH, "x_extend": _X_EXTEND,
                        "gwt": _NO_WATER, "n_gp": _GAUSS},
    "fem2d_consolidation": {
        "reset_displacements_after": (
            "from the end of the initial stage",
            "the load stage alone: the settlements leave out self-weight "
            "(common FE practice); the result carries a judgment record "
            "where the choice matters")},
    "fem2d_staged": {
        "reset_displacements_after": (
            "after stage 0",
            "the initial gravity / K0 stage: construction movements leave "
            "out self-weight (common FE practice); the result carries a "
            "judgment record where the choice matters")},
    "fem2d_excavation": {
        "gwt": ("no water table",
                "no water pressure on the wall and no seepage"),
        "lifts": ("one lift to each strut level, then to the final depth",
                  "the excavation sequence the module takes; give lifts for "
                  "the sequence planned")},
})

# The keys each item of a list parameter carries, so a key given at the
# top level is refused with a hint naming its list (BACKLOG GM-113).
from funhouse_agent.adapters import item_keys_for, merge_item_keys  # noqa: E402,F401
ITEM_KEYS = merge_item_keys(
    item_keys_for(METHOD_INFO, "soil_layers", _LAYER_KEYS),
    item_keys_for(METHOD_INFO, "soil_layers", _EXCAVATION_LAYER_KEYS,
                  ("fem2d_excavation",)),
    item_keys_for(METHOD_INFO, "soil_layers", _CONSOLIDATION_LAYER_KEYS,
                  ("fem2d_consolidation",)),
    item_keys_for(METHOD_INFO, "material_props", _MATERIAL_KEYS))
