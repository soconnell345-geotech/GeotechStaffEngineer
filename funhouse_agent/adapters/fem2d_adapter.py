"""FEM2D adapter — flat dict -> fem2d high-level API -> dict."""

import numpy as np

from funhouse_agent.adapters import clean_result


def _run_analyze_gravity(params: dict) -> dict:
    from fem2d import analyze_gravity
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


def _run_analyze_slope_srm(params: dict) -> dict:
    from fem2d import analyze_slope_srm
    surface_points = [tuple(pt) for pt in params["surface_points"]]
    soil_layers = params["soil_layers"]

    kwargs = dict(
        surface_points=surface_points,
        soil_layers=soil_layers,
        nx=params.get("nx", 30),
        ny=params.get("ny", 15),
        srf_tol=params.get("srf_tol", 0.02),
        n_load_steps=params.get("n_load_steps", 10),
        t=params.get("t", 1.0),
        gamma_w=params.get("gamma_w", 9.81),
        max_iter=params.get("max_iter", 100),
        tol=params.get("tol", 1e-5),
    )
    if "depth" in params:
        kwargs["depth"] = params["depth"]
    if "x_extend" in params:
        kwargs["x_extend"] = params["x_extend"]
    if "gwt" in params:
        gwt = params["gwt"]
        if isinstance(gwt, list) and len(gwt) > 0 and isinstance(gwt[0], (list, tuple)):
            kwargs["gwt"] = np.array(gwt)
        else:
            kwargs["gwt"] = gwt
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]

    result = analyze_slope_srm(**kwargs)
    return clean_result(result.to_dict())


def _run_analyze_excavation(params: dict) -> dict:
    from fem2d import analyze_excavation
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
        gwt = params["gwt"]
        if isinstance(gwt, list) and len(gwt) > 0 and isinstance(gwt[0], (list, tuple)):
            kwargs["gwt"] = np.array(gwt)
        else:
            kwargs["gwt"] = gwt
    if "struts" in params:
        kwargs["struts"] = params["struts"]
    if "layer_polylines" in params:
        kwargs["layer_polylines"] = params["layer_polylines"]

    result = analyze_excavation(**kwargs)
    return clean_result(result.to_dict())


def _run_analyze_seepage(params: dict) -> dict:
    from fem2d import analyze_seepage
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
        gwt = pd.get("gwt")
        if isinstance(gwt, list) and len(gwt) > 0 and isinstance(gwt[0], (list, tuple)):
            gwt = np.array(gwt)
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
    "fem2d_slope_srm": _run_analyze_slope_srm,
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
            "srf_tol": {"type": "float", "required": False, "default": 0.02, "description": "SRF bisection tolerance."},
            "gwt": {"type": "float|array", "required": False, "description": "Groundwater table: float elevation, or [[x,y],...] polyline."},
        },
        "returns": {
            "FOS": "Factor of safety from SRM.",
            "max_displacement_m": "Maximum displacement at failure.",
            "converged": "Whether SRM converged.",
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
            "soil_layers": {"type": "array", "required": True, "description": "Soil property dicts with 'E','nu','gamma'."},
            "k": {"type": "float", "required": True, "description": "Hydraulic conductivity (m/s)."},
            "load_q": {"type": "float", "required": True, "description": "Surface load (kPa, positive downward)."},
            "time_points": {"type": "array", "required": True, "description": "Time points (seconds) for output."},
            "gwt": {"type": "float", "required": False, "default": 0.0, "description": "GWT elevation (m)."},
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
