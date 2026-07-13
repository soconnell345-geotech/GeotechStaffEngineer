"""FEM2D adapter — flat dict -> fem2d high-level API -> dict."""

import numpy as np

from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_keys, require_params,
)



def _require_layer_elevations(soil_layers, *, method):
    """fem2d builds strata from each layer's bottom_elevation; everything else
    (E, nu, c, phi, psi, gamma) has module defaults."""
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
              "nx", "ny", "element_type", "max_iter", "tol", "q_applied")
    reject_unknown_params(params, _valid, method="fem2d_footing_capacity")
    require_params(params, ["B", "c"], method="fem2d_footing_capacity",
                   valid=_valid)
    element_type = params.get("element_type", "t6")
    _require_choice(element_type, ("t6", "cst"),
                    name="element_type", method="fem2d_footing_capacity")
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
        max_iter=params.get("max_iter", 1000), tol=params.get("tol", 1e-4),
        q_applied=params.get("q_applied"),
    )
    out = {
        "q_ult_kPa": result.q_ult_kPa,
        "Nc_backfigured": result.Nc_backfigured,
        "bearing_FOS": result.bearing_FOS,
        "bearing_capacity_factors": result.bearing_capacity_factors,
        "q_ult_estimate_kPa": result.q_ult_estimate_kPa,
        "q_max_kPa": result.q_max_kPa,
        "n_steps_converged": result.n_steps_converged,
        "collapse_load_fraction": result.collapse_load_fraction,
        "collapse_bracketed": result.collapse_bracketed,
        "max_displacement_m": result.max_displacement_m,
    }
    return clean_result(out)


def _run_analyze_slope_srm(params: dict) -> dict:
    from fem2d import analyze_slope_srm
    _valid = ("surface_points", "soil_layers", "nx", "ny", "srf_tol",
              "n_load_steps", "t", "gamma_w", "max_iter", "tol",
              "element_type", "srm_field", "blowup_factor", "srf_range",
              "n_gp", "depth", "x_extend", "gwt", "layer_polylines")
    reject_unknown_params(params, _valid, method="fem2d_slope_srm")
    require_params(params, ["surface_points", "soil_layers"],
                   method="fem2d_slope_srm", valid=_valid)
    _require_layer_elevations(params["soil_layers"], method="fem2d_slope_srm")
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    soil_layers = params["soil_layers"]

    element_type = params.get("element_type", "t6")
    _require_choice(element_type, ("t6", "cst"),
                    name="element_type", method="fem2d_slope_srm")
    srm_field = params.get("srm_field", "c_phi")
    _require_choice(srm_field, ("c_phi", "c", "phi"),
                    name="srm_field", method="fem2d_slope_srm")
    n_gp = params.get("n_gp")
    if n_gp is not None:
        _require_choice(int(n_gp), (3, 6),
                        name="n_gp", method="fem2d_slope_srm")
    _require_positive(
        params.get("depth"), name="depth", method="fem2d_slope_srm",
        hint=" Omit 'depth' for the 2*H default. To compare FOS across several "
             "meshes (a mesh-refinement / mesh-consistency study), use the "
             "srm_mesh_refinement_study method instead of driving this per mesh.")

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
    out = clean_result(result.to_dict())
    out["fos_basis"] = getattr(result, "fos_basis", None)
    out["n_srf_trials"] = getattr(result, "n_srf_trials", None)
    curve = getattr(result, "srf_curve", None)
    if curve is not None:
        srf, dim = curve
        out["srf_curve"] = {
            "srf": [round(float(v), 4) for v in srf],
            "dimensionless_displacement": [round(float(v), 4) for v in dim],
        }
    return out


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
    kwargs = dict(
        surface_points=[tuple(pt) for pt in params["surface_points"]],
        soil_layers=params["soil_layers"],
        nx=params.get("nx", 30), ny=params.get("ny", 15),
        srf_tol=params.get("srf_tol", 0.02),
        n_load_steps=params.get("n_load_steps", 2),
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
    return clean_result(out)


def _run_analyze_excavation(params: dict) -> dict:
    from fem2d import analyze_excavation
    _valid = ("width", "depth", "wall_depth", "soil_layers", "wall_EI",
              "wall_EA", "nx", "ny", "t", "n_steps", "gamma_w", "max_iter",
              "tol", "gwt", "struts", "layer_polylines")
    reject_unknown_params(params, _valid, method="fem2d_excavation")
    require_params(params, ["width", "depth", "wall_depth", "soil_layers",
                            "wall_EI", "wall_EA"],
                   method="fem2d_excavation", valid=_valid)
    _require_layer_elevations(params["soil_layers"], method="fem2d_excavation")
    _require_positive(params.get("depth"), name="depth",
                      method="fem2d_excavation")
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
        max_iter=params.get("max_iter", 100),
        tol=params.get("tol", 1e-5),
    )
    if "gwt" in params:
        kwargs["gwt"] = _normalize_gwt(params["gwt"])
    if "struts" in params:
        kwargs["struts"] = params["struts"]
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]

    result = analyze_excavation(**kwargs)
    return clean_result(result.to_dict())


def _run_analyze_seepage(params: dict) -> dict:
    from fem2d import analyze_seepage
    _valid = ("nodes", "elements", "k", "head_bcs", "t", "gamma_w")
    reject_unknown_params(params, _valid, method="fem2d_seepage")
    require_params(params, ["nodes", "elements", "k", "head_bcs"],
                   method="fem2d_seepage", valid=_valid)
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
    return clean_result(result.to_dict())


def _run_analyze_consolidation(params: dict) -> dict:
    from fem2d import analyze_consolidation
    _valid = ("width", "depth", "soil_layers", "k", "load_q", "time_points",
              "gwt", "gamma_w", "nx", "ny", "t", "n_w", "layer_polylines",
              "consolidation_scheme", "theta")
    reject_unknown_params(params, _valid, method="fem2d_consolidation")
    require_params(params, ["width", "depth", "soil_layers", "k", "load_q",
                            "time_points"],
                   method="fem2d_consolidation", valid=_valid)
    _require_layer_elevations(params["soil_layers"], method="fem2d_consolidation")
    _require_positive(params.get("depth"), name="depth",
                      method="fem2d_consolidation")
    scheme = params.get("consolidation_scheme", "staggered")
    _require_choice(scheme, ("staggered", "monolithic"),
                    name="consolidation_scheme", method="fem2d_consolidation")
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
    )
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]

    result = analyze_consolidation(**kwargs)
    return clean_result(result.to_dict())


def _run_analyze_staged(params: dict) -> dict:
    from fem2d import (
        analyze_staged, ConstructionPhase,
        generate_rect_mesh, detect_boundary_nodes,
        assign_element_groups, create_wall_elements,
    )

    _valid = ("nodes", "elements", "material_props", "gamma",
              "element_groups", "phases", "t", "max_iter", "tol", "gamma_w",
              "beam_elements")
    reject_unknown_params(params, _valid, method="fem2d_staged")
    require_params(params, ["nodes", "elements", "material_props", "gamma",
                            "element_groups", "phases"],
                   method="fem2d_staged", valid=_valid)
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
        max_iter=params.get("max_iter", 100),
        tol=params.get("tol", 1e-5),
        gamma_w=params.get("gamma_w", 9.81),
    )

    # Beam elements if provided
    if "beam_elements" in params:
        from fem2d.elements import BeamElement
        beam_elems = []
        for bd in params["beam_elements"]:
            require_keys(bd, ["node_i", "node_j", "EA", "EI"],
                         method="fem2d_staged", item_label="beam_elements[]")
            beam_elems.append(BeamElement(
                node_i=bd["node_i"],
                node_j=bd["node_j"],
                EA=bd["EA"],
                EI=bd["EI"],
                weight_per_m=bd.get("weight_per_m", 0.0),
            ))
        kwargs["beam_elements"] = beam_elems

    result = analyze_staged(**kwargs)
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "fem2d_gravity": _run_analyze_gravity,
    "fem2d_foundation": _run_analyze_foundation,
    "fem2d_footing_capacity": _run_analyze_footing_capacity,
    "fem2d_slope_srm": _run_analyze_slope_srm,
    "fem2d_local_fos": _run_analyze_local_fos,
    "fem2d_excavation": _run_analyze_excavation,
    "fem2d_seepage": _run_analyze_seepage,
    "fem2d_consolidation": _run_analyze_consolidation,
    "fem2d_staged": _run_analyze_staged,
}

METHOD_INFO = {
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
            "t": {"type": "float", "required": False, "default": 1.0, "description": "Out-of-plane thickness (m)."},
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
        "brief": "Ultimate bearing capacity of a rigid strip footing by FEM load-control to collapse (plane-strain, the FE analogue of a bearing-capacity calc). Validated vs Prandtl Nc=5.14. Use element_type='t6' (CST locks and never collapses).",
        "parameters": {
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "c": {"type": "float", "required": True, "description": "Cohesion / undrained shear strength (kPa)."},
            "phi": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle (deg). Default 0 (Prandtl / undrained)."},
            "gamma": {"type": "float", "required": False, "default": 0.0, "description": "Soil unit weight (kN/m3), applied as a body force ramped with the load. Default 0 = classical weightless mechanism (the validated basis)."},
            "E": {"type": "float", "required": False, "default": 1e5, "description": "Young's modulus (kPa)."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio."},
            "psi": {"type": "float", "required": False, "default": 0.0, "description": "Dilation angle (deg)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform surcharge q beside the footing (kPa) — used to size the load ramp and the reported factors."},
            "q_max": {"type": "float", "required": False, "description": "Top of the load ramp (kPa). Default 1.6x the closed-form estimate c*Nc+surcharge*Nq+0.5*gamma*B*Ngamma. Pass explicitly for a c=phi=0 footing."},
            "n_load_steps": {"type": "int", "required": False, "default": 45, "description": "Load increments; collapse-load resolution = q_max/n_load_steps."},
            "nx": {"type": "int", "required": False, "default": 40, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 20, "description": "Mesh divisions in y."},
            "element_type": {"type": "str", "required": False, "default": "t6", "allowed_values": ["t6", "cst"], "description": "Soil element. t6 (recommended): collapse load within ~2% of Prandtl. cst LOCKS on the isochoric bearing mechanism and never collapses — do not use for capacity."},
            "q_applied": {"type": "float", "required": False, "description": "Working/design pressure (kPa). If given, bearing_FOS = q_ult/q_applied is reported."},
        },
        "returns": {
            "q_ult_kPa": "Ultimate bearing pressure (last converged load level).",
            "Nc_backfigured": "q_ult/c for a pure-cohesion footing (compare to Prandtl 5.14); null otherwise.",
            "bearing_FOS": "q_ult/q_applied when q_applied is given, else null.",
            "bearing_capacity_factors": "Closed-form {Nc, Nq, Ngamma} reference (Prandtl-Reissner + Vesic Ngamma).",
            "collapse_bracketed": "False if the footing carried the full q_max (q_ult is then a lower bound — raise q_max).",
        },
    },
    "fem2d_slope_srm": {
        "category": "FEM 2D",
        "brief": "Slope stability FOS via Strength Reduction Method (plane strain FEM).",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,z],...] array."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of dicts with 'name','bottom_elevation','E','nu','c','phi','psi','gamma'."},
            "depth": {"type": "float", "required": False, "description": "Depth below lowest surface (m). Default 2*H."},
            "nx": {"type": "int", "required": False, "default": 30, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 15, "description": "Mesh divisions in y."},
            "x_extend": {"type": "float", "required": False, "description": "Horizontal extension (m). Default 2*width."},
            "srf_tol": {"type": "float", "required": False, "default": 0.02, "description": "SRF bisection tolerance (0.01 for benchmark-grade runs)."},
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
        },
    },
    "fem2d_local_fos": {
        "category": "FEM 2D",
        "brief": "Local (pointwise) factor-of-safety map for a slope: runs the SRM then evaluates local_FOS = Mohr-Coulomb available shear / mobilized shear at each element using the ORIGINAL strengths. At the critical SRF the minimum local FOS ~ the global SRM FOS and the low-FOS band is the failure mass. Returns summary stats (the full field is for plotting).",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "Ground surface as [[x,z],...] array."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of dicts with 'name','bottom_elevation','E','nu','c','phi','psi','gamma' (same as fem2d_slope_srm)."},
            "depth": {"type": "float", "required": False, "description": "Depth below lowest surface (m). Default 2*H."},
            "nx": {"type": "int", "required": False, "default": 30, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 15, "description": "Mesh divisions in y."},
            "x_extend": {"type": "float", "required": False, "description": "Horizontal margin (m). Pass 0 when the profile already has margins."},
            "srf_tol": {"type": "float", "required": False, "default": 0.02, "description": "SRF bisection tolerance."},
            "element_type": {"type": "str", "required": False, "default": "t6", "allowed_values": ["t6", "cst"], "description": "Soil element. t6 recommended (cst locks)."},
            "srm_field": {"type": "str", "required": False, "default": "c_phi", "allowed_values": ["c_phi", "c", "phi"], "description": "Strengths reduced by the SRF."},
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
        "brief": "Braced excavation analysis with sheet pile wall (plane strain FEM).",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Excavation width (m)."},
            "depth": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "wall_depth": {"type": "float", "required": True, "description": "Total wall depth below surface (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "Array of soil property dicts."},
            "wall_EI": {"type": "float", "required": True, "description": "Wall flexural stiffness (kN*m2/m)."},
            "wall_EA": {"type": "float", "required": True, "description": "Wall axial stiffness (kN/m)."},
            "struts": {"type": "array", "required": False, "description": "Array of {depth, stiffness} for struts."},
            "gwt": {"type": "float|array", "required": False, "description": "Groundwater table."},
            "n_steps": {"type": "int", "required": False, "default": 10, "description": "Number of excavation steps."},
            "layer_polylines": {"type": "dict", "required": False, "description": "{layer_name: [[x,y],...]} bottom-boundary polylines for non-horizontal layering."},
        },
        "returns": {
            "max_displacement_m": "Maximum displacement.",
            "max_beam_moment_kNm_per_m": "Maximum wall bending moment.",
            "max_beam_shear_kN_per_m": "Maximum wall shear force.",
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
        },
        "returns": {
            "max_head_m": "Maximum total head.",
            "min_head_m": "Minimum total head.",
            "total_flow_m3_per_s_per_m": "Total seepage flow rate.",
        },
    },
    "fem2d_consolidation": {
        "category": "FEM 2D",
        "brief": "Coupled Biot consolidation analysis (settlement + pore pressure dissipation).",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Domain width (m)."},
            "depth": {"type": "float", "required": True, "description": "Domain depth (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "Soil property dicts, each with 'E' (kPa), 'nu', 'gamma' (kN/m3) AND 'bottom_elevation' (m; 'top_elevation' optional, defaults to the domain top / the layer above)."},
            "k": {"type": "float", "required": True, "description": "Hydraulic conductivity (m/s) for 'staggered'; for 'monolithic' pass the MOBILITY m^2/(kPa.s)."},
            "load_q": {"type": "float", "required": True, "description": "Surface load (kPa, positive downward)."},
            "time_points": {"type": "array", "required": True, "description": "Time points (seconds) for output."},
            "gwt": {"type": "float", "required": False, "default": 0.0, "description": "GWT elevation (m)."},
            "n_w": {"type": "float", "required": False, "default": 2.2e6, "description": "For 'monolithic': the Biot modulus M (kPa) (~2.2e6 for near-incompressible water). Governs the undrained load response p0. Ignored by 'staggered'."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Mesh divisions in x (horizontal)."},
            "ny": {"type": "int", "required": False, "default": 20, "description": "Mesh divisions in y (vertical)."},
            "t": {"type": "float", "required": False, "default": 1.0, "description": "Out-of-plane thickness (m) for plane-strain assembly."},
            "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Unit weight of water (kN/m3)."},
            "layer_polylines": {"type": "array", "required": False, "description": "Optional explicit layer-boundary polylines [[[x,z],...],...] for non-horizontal layering."},
            "consolidation_scheme": {"type": "string", "required": False, "default": "staggered",
                                     "allowed_values": ["staggered", "monolithic"],
                                     "description": "Biot solver: 'staggered' (default, sequential split) or 'monolithic' (coupled u-p, Taylor-Hood T6/T3, reproduces the load-induced undrained response p0 and the Terzaghi transient). For 'monolithic', pass k as the MOBILITY m^2/(kPa.s) and n_w as the Biot modulus M (kPa)."},
            "theta": {"type": "float", "required": False, "default": 1.0,
                      "description": "Time-integration parameter for the monolithic scheme (1.0 backward Euler; 0.5 Crank-Nicolson, more accurate). Range [0.5, 1.0]."},
        },
        "returns": {
            "max_settlement_m": "Maximum settlement.",
            "degree_of_consolidation": "Degree of consolidation at final time.",
            "converged": "Whether analysis converged.",
        },
    },
    "fem2d_staged": {
        "category": "FEM 2D",
        "brief": "Staged construction analysis (multi-phase, activate/deactivate soil groups and beams).",
        "parameters": {
            "nodes": {"type": "array", "required": True, "description": "Node coordinates [[x,y],...]."},
            "elements": {"type": "array", "required": True, "description": "Element connectivity [[n1,n2,n3],...]."},
            "material_props": {"type": "array", "required": True, "description": "Per-element material dicts."},
            "gamma": {"type": "float|array", "required": True, "description": "Unit weight. Scalar or per-element."},
            "element_groups": {"type": "dict", "required": True, "description": "Group name -> list of element indices."},
            "phases": {"type": "array", "required": True, "description": "Array of phase dicts with name, active_soil_groups, etc."},
            "beam_elements": {"type": "array", "required": False, "description": "Array of beam element dicts."},
        },
        "returns": {
            "n_phases": "Number of phases completed.",
            "converged": "Whether all phases converged.",
            "phases": "Per-phase results.",
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
          "description": "Out-of-plane thickness (m) for plane-strain assembly."},
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
    "fem2d_footing_capacity": ["domain_depth", "domain_half_width", "max_iter", "tol"],
    "fem2d_slope_srm": ["gamma_w", "n_load_steps", "t", "tol"],
    "fem2d_local_fos": ["blowup_factor", "gamma_w", "gwt", "layer_polylines",
                        "max_iter", "n_gp", "n_load_steps", "srf_range", "t", "tol"],
    "fem2d_excavation": ["gamma_w", "max_iter", "nx", "ny", "t", "tol"],
    "fem2d_seepage": ["t"],
    "fem2d_staged": ["gamma_w", "max_iter", "t", "tol"],
}
for _m, _names in _FEM_PARAM_BACKFILL.items():
    _pp = METHOD_INFO[_m]["parameters"]
    for _n in _names:
        _pp.setdefault(_n, dict(_FEM_PARAM_DECLS[_n]))
