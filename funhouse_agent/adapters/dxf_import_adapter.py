"""DXF import adapter — discover layers, parse geometry, build slope/FEM inputs."""

from funhouse_agent.adapters import (
    check_nested_keys, check_nested_list, clean_result, declare_when_absent,
    reject_unknown_params, require_params,
)

# The keys this adapter reads from each nested dict: the LayerMapping
# fields, the nail_defaults build_slope_geometry reads (SoilNail's own
# names), and the soil_properties of a slope (SoilPropertyAssignment) and of
# an FEM model (FEMSoilPropertyAssignment). Anything else was read with .get
# and a default and silently left out (GM-58).
_MAPPING_KEYS = ("surface", "soil_boundaries", "water_table", "nails",
                 "annotations")
_NAIL_DEFAULT_KEYS = ("inclination", "bar_diameter", "drill_hole_diameter",
                      "fy", "bond_stress", "spacing_h")
_SOIL_PROPERTY_KEYS = {
    "slope": ("name", "gamma", "gamma_sat", "phi", "c_prime", "cu",
              "analysis_mode"),
    "fem": ("name", "gamma", "phi", "c", "E", "nu", "psi", "model",
            "hs_params"),
}


def _slope_params(geom) -> dict:
    """A SlopeGeometry as the slope_stability adapter's own parameters
    (surface_points, soil_layers, gwt_points and any nails), so the result is
    the next call's input: build_slope_geometry's own docstring says the
    geometry is "ready for analyze_slope() or search_critical_surface()".
    SlopeGeometry has no to_dict, so this adapter raised AttributeError on
    every real run; only mocked tests had called it."""
    import dataclasses

    def _pts(points):
        return [[float(x), float(z)] for x, z in points]

    layers = []
    for lyr in geom.soil_layers:
        d = {"name": lyr.name, "top_elevation": lyr.top_elevation,
             "bottom_elevation": lyr.bottom_elevation, "gamma": lyr.gamma,
             "phi": lyr.phi, "c_prime": lyr.c_prime, "cu": lyr.cu,
             "analysis_mode": lyr.analysis_mode}
        if lyr.gamma_sat is not None:
            d["gamma_sat"] = lyr.gamma_sat
        if lyr.bottom_boundary_points:
            d["bottom_boundary_points"] = _pts(lyr.bottom_boundary_points)
        layers.append(d)
    out = {"surface_points": _pts(geom.surface_points),
           "soil_layers": layers,
           "gwt_points": _pts(geom.gwt_points) if geom.gwt_points else None}
    if geom.nails:
        out["nails"] = [dataclasses.asdict(n) for n in geom.nails]
    return out


def check_soil_properties(items, kind, *, method):
    """Refuse a soil_properties[] key a ``kind`` ('slope' or 'fem')
    assignment does not take (GM-58): every one was read with .get and a
    default, so a slope layer's 'c' ran with c' = 0 and an FEM layer's
    'unit_weight' at gamma = 18. The did-you-means are the slope and fem2d
    adapters' own. Shared with the pdf_import adapter, which builds the same
    assignments."""
    if kind == "slope":
        from funhouse_agent.adapters.slope_stability import (
            _LAYER_KEY_HINTS as hints, _LAYER_KEY_NOTES as notes,
        )
    else:
        from funhouse_agent.adapters.fem2d_adapter import (
            _LAYER_KEY_HINTS as hints, _LAYER_KEY_NOTES as notes,
        )
    check_nested_list(items, _SOIL_PROPERTY_KEYS[kind], method=method,
                      label="soil_properties", hints=hints, notes=notes)


#: What a soil_properties[] entry must carry, by kind and case, and what it
#: may leave out with the value that then stands in (listed in the result's
#: ``defaults_used``). A drawing gives geometry only: dxf_import's own
#: documents say unit weight, strength and stiffness "must always come from
#: the user", so none of those is assumed (GM-327; they were gamma 18,
#: phi 0, c 0, cu 0, E 30000 and nu 0.3 with nothing said).
_SOIL_REQUIRED_TEXT = {
    "slope": ("a drained layer needs gamma and phi (c_prime is 0 when left "
              "out, and listed); an undrained layer (analysis_mode="
              "'undrained') needs gamma and cu"),
    "fem": ("every layer needs gamma, E and nu; a layer with a strength "
            "model (mc or hs) also needs phi (c and psi are 0 when left "
            "out, and listed)"),
}


def soil_assignments(items, kind, *, method):
    """``(kwargs_list, defaults_used)`` for the ``kind`` ('slope' or 'fem')
    soil_properties ``items``: one dict of constructor arguments per layer
    (SoilPropertyAssignment / FEMSoilPropertyAssignment), and every value
    that stood in for an input a layer left out, keyed
    ``soil_properties[<name>].<input>``.

    A layer without its unit weight, its strength for its case, or (FEM)
    its stiffness is refused in one message that names every missing input
    of every layer. Shared with the pdf_import adapter."""
    check_soil_properties(items, kind, method=method)
    out, used, missing = [], {}, []
    for i, sp in enumerate(items or []):
        name = sp.get("name", "")
        label = f"soil_properties[{name or i}]"
        need = ["gamma"]
        kw = {"name": name}

        def default(key, value):
            if sp.get(key) is None:
                used[f"{label}.{key}"] = value
                return value
            return sp[key]

        if kind == "slope":
            mode = default("analysis_mode", "drained")
            kw["analysis_mode"] = mode
            if mode == "undrained":
                need.append("cu")
                kw["phi"] = sp.get("phi", 0.0)
                kw["c_prime"] = sp.get("c_prime", 0.0)
            else:
                need.append("phi")
                kw["c_prime"] = default("c_prime", 0.0)
                kw["cu"] = sp.get("cu", 0.0)
            # left out, the layer is weighed at gamma below the water
            # table too (SlopeSoilLayer's own rule); the output carries that
            # value as gamma_sat, so it is listed as the value it took
            kw["gamma_sat"] = sp.get("gamma_sat")
            if sp.get("gamma_sat") is None and sp.get("gamma") is not None:
                used[f"{label}.gamma_sat"] = sp["gamma"]
        else:
            need += ["E", "nu"]
            model = default("model", "mc")
            kw["model"] = model
            kw["hs_params"] = sp.get("hs_params")
            if model == "elastic":
                kw["phi"] = sp.get("phi", 0.0)
                kw["c"] = sp.get("c", 0.0)
                kw["psi"] = sp.get("psi", 0.0)
            else:
                need.append("phi")
                kw["c"] = default("c", 0.0)
                kw["psi"] = default("psi", 0.0)
        lacking = [k for k in need if sp.get(k) is None]
        if lacking:
            missing.append(f"{label} has no {', '.join(lacking)}")
            continue
        for k in need:
            kw[k] = sp[k]
        out.append(kw)
    if missing:
        raise ValueError(
            f"{method}: {'; '.join(missing)}. A drawing gives geometry "
            f"only, so unit weight, strength and stiffness come from the "
            f"user and nothing is assumed: {_SOIL_REQUIRED_TEXT[kind]}.")
    return out, used


def nail_defaults_used(nail_lines, nail_defaults) -> dict:
    """The nail properties build_slope_geometry fills in for the drawing's
    nails because ``nail_defaults`` left them out, keyed
    ``nail_defaults.<input>`` (empty when the drawing has no nails)."""
    if not nail_lines:
        return {}
    from dxf_import.converter import NAIL_PROPERTY_DEFAULTS
    given = nail_defaults or {}
    return {f"nail_defaults.{k}": v
            for k, v in NAIL_PROPERTY_DEFAULTS.items()
            if given.get(k) is None}


def _with_defaults(result: dict, used: dict) -> dict:
    """``result`` with ``defaults_used`` when anything stood in."""
    if used:
        result["defaults_used"] = used
    return result


def _run_discover_layers(params):
    from dxf_import import discover_layers

    reject_unknown_params(params, ("file_path",), method="discover_layers")
    require_params(params, ["file_path"], method="discover_layers")
    filepath = params.get("file_path")
    result = discover_layers(filepath=filepath)
    return clean_result(result.to_dict())


def _run_parse_dxf_geometry(params):
    from dxf_import import parse_dxf_geometry, LayerMapping

    reject_unknown_params(params, ("file_path", "layer_mapping", "units",
                                   "flip_y", "ignore_header_units"),
                          method="parse_geometry")
    require_params(params, ["file_path", "layer_mapping"],
                   method="parse_geometry",
                   valid=["file_path", "layer_mapping", "units", "flip_y",
                          "ignore_header_units"])
    filepath = params.get("file_path")
    # No default: left out, the drawing's own header units are used, and a
    # drawing whose header states none is refused (GM-327).
    units = params.get("units")
    flip_y = params.get("flip_y", False)

    # Build LayerMapping from flat dict params
    mapping_dict = params.get("layer_mapping", {})
    check_nested_keys(mapping_dict, _MAPPING_KEYS,
                      method="parse_geometry", label="layer_mapping",
                      hints={"water": ("water_table",), "gwt": ("water_table",),
                             "groundwater": ("water_table",),
                             "ground": ("surface",),
                             "boundaries": ("soil_boundaries",),
                             "text": ("annotations",)})
    mapping = LayerMapping(
        surface=mapping_dict.get("surface", ""),
        soil_boundaries=mapping_dict.get("soil_boundaries", {}),
        water_table=mapping_dict.get("water_table"),
        nails=mapping_dict.get("nails"),
        annotations=mapping_dict.get("annotations"),
    )

    result = parse_dxf_geometry(
        filepath=filepath,
        layer_mapping=mapping,
        units=units,
        flip_y=flip_y,
        ignore_header_units=bool(params.get("ignore_header_units", False)),
    )
    return clean_result(result.to_dict())


def _run_build_slope_geometry(params):
    from dxf_import import build_slope_geometry, SoilPropertyAssignment
    from dxf_import.results import DxfParseResult

    reject_unknown_params(params, ("parse_result", "soil_properties",
                                   "nail_defaults"),
                          method="build_slope_geometry")
    require_params(params, ["parse_result", "soil_properties"],
                   method="build_slope_geometry",
                   valid=["parse_result", "soil_properties", "nail_defaults"])
    # Reconstruct DxfParseResult from dict
    pr = params.get("parse_result", {})
    surface_points = [
        (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
        for p in pr.get("surface_points", [])
    ]
    boundary_profiles = {}
    for name, pts in pr.get("boundary_profiles", {}).items():
        boundary_profiles[name] = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pts
        ]
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pr["gwt_points"]
        ]
    nail_lines = pr.get("nail_lines", [])
    text_annotations = pr.get("text_annotations", [])

    parse_result = DxfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units_used=pr.get("units_used", "m"),
        warnings=pr.get("warnings", []),
    )

    # Build SoilPropertyAssignment list: a missing unit weight or strength
    # is refused, and what stood in for the rest is listed (GM-327)
    if params.get("nail_defaults") is not None:
        check_nested_keys(params["nail_defaults"], _NAIL_DEFAULT_KEYS,
                          method="build_slope_geometry",
                          label="nail_defaults")
    kwargs_list, used = soil_assignments(params.get("soil_properties"),
                                         "slope",
                                         method="build_slope_geometry")
    soil_props = [SoilPropertyAssignment(**kw) for kw in kwargs_list]

    nail_defaults = params.get("nail_defaults")
    used.update(nail_defaults_used(nail_lines, nail_defaults))

    geom = build_slope_geometry(
        parse_result=parse_result,
        soil_properties=soil_props,
        nail_defaults=nail_defaults,
    )
    return clean_result(_with_defaults(_slope_params(geom), used))


def _run_build_fem_inputs(params):
    from dxf_import import build_fem_inputs, FEMSoilPropertyAssignment
    from dxf_import.results import DxfParseResult

    reject_unknown_params(params, ("parse_result", "soil_properties"),
                          method="build_fem_inputs")
    require_params(params, ["parse_result", "soil_properties"],
                   method="build_fem_inputs",
                   valid=["parse_result", "soil_properties"])
    # Reconstruct DxfParseResult from dict
    pr = params.get("parse_result", {})
    surface_points = [
        (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
        for p in pr.get("surface_points", [])
    ]
    boundary_profiles = {}
    for name, pts in pr.get("boundary_profiles", {}).items():
        boundary_profiles[name] = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pts
        ]
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pr["gwt_points"]
        ]

    parse_result = DxfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=pr.get("nail_lines", []),
        text_annotations=pr.get("text_annotations", []),
        units_used=pr.get("units_used", "m"),
        warnings=pr.get("warnings", []),
    )

    # Build FEMSoilPropertyAssignment list: a missing unit weight,
    # stiffness or strength is refused, and what stood in for the rest is
    # listed (GM-327)
    kwargs_list, used = soil_assignments(params.get("soil_properties"),
                                         "fem", method="build_fem_inputs")
    soil_props = [FEMSoilPropertyAssignment(**kw) for kw in kwargs_list]

    result = build_fem_inputs(
        parse_result=parse_result,
        soil_properties=soil_props,
    )
    return clean_result(_with_defaults(result, used))


METHOD_REGISTRY = {
    "discover_layers": _run_discover_layers,
    "parse_geometry": _run_parse_dxf_geometry,
    "build_slope_geometry": _run_build_slope_geometry,
    "build_fem_inputs": _run_build_fem_inputs,
}

METHOD_INFO = {
    "discover_layers": {
        "category": "DXF Import",
        "brief": "Inventory all DXF layers with entity counts, types, and bounding boxes.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to a DXF file."},
        },
        "returns": {
            "n_layers": "Number of layers with entities.",
            "n_total_entities": "Total entity count.",
            "units_hint": "Units detected from DXF header.",
            "layers": "List of layer info dicts (name, n_entities, entity_types, sample_texts, bbox).",
        },
    },
    "parse_geometry": {
        "category": "DXF Import",
        "brief": "Extract slope geometry coordinates from mapped DXF layers.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to DXF file."},
            "layer_mapping": {"type": "dict", "required": True, "description": "Mapping with keys: surface (str), soil_boundaries ({layer: soil_name}), water_table (str), nails (str), annotations ([str])."},
            "units": {"type": "str", "required": False, "allowed_values": ["m", "mm", "cm", "ft", "in"], "description": "Units the drawing is drawn in. Left out, the units the drawing's own header states ($INSUNITS, discover_layers' units_hint) are used; a drawing whose header states none is refused until units is given. Given, they are compared with the header, and a difference is refused unless ignore_header_units is true. Nothing is assumed."},
            "flip_y": {"type": "bool", "required": False, "default": False, "description": "Negate Y values for downward-positive drawings."},
            "ignore_header_units": {"type": "bool", "required": False, "default": False, "description": "True: use units even though the drawing's header states other units (a header left at a template's value is common); the warnings then name both."},
        },
        "returns": {
            "surface_points": "Ground surface as [{x, z}, ...], in metres.",
            "boundary_profiles": "Soil boundaries: {name: [{x, z}, ...]}, in metres.",
            "gwt_points": "GWT profile or null.",
            "nail_lines": "Nail geometry list (only when the drawing has nails), in metres. Only LINE entities are read as nails; anything else on the nail layer is counted in warnings.",
            "text_annotations": "Text labels (only when there are any): [{text, x, y, layer}], x and y in metres like the geometry.",
            "units_used": "The units the coordinates were converted from.",
            "units_source": "Where units_used came from: given, read from the drawing's header, or given against the header.",
            "header_units": "The units the drawing's own header states, or null.",
            "warnings": "Any parsing warnings: a mapped layer not found, a layer with no geometry, entities left out, a header ignored.",
        },
    },
    "build_slope_geometry": {
        "category": "DXF Import",
        "brief": "Assemble SlopeGeometry from DXF parse result and soil properties.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Output from parse_dxf_geometry (surface_points, boundary_profiles, etc.)."},
            "soil_properties": {"type": "list", "required": True, "description": "List of dicts: {name, gamma, gamma_sat, phi, c_prime, cu, analysis_mode}. A drawing gives geometry only, so no unit weight or strength is assumed: a drained layer must give gamma and phi, an undrained layer (analysis_mode='undrained') gamma and cu; a layer without them is refused. Left out and listed in defaults_used: c_prime 0, analysis_mode 'drained', gamma_sat = the layer's gamma (one unit weight above and below the water table)."},
            "nail_defaults": {"type": "dict", "required": False, "description": "Properties for the nails the drawing has: {bar_diameter mm, drill_hole_diameter mm, fy MPa, bond_stress kPa, spacing_h m}. A drawing gives each nail's head, length and inclination only. A property left out takes slope_stability's own nail default (25, 150, 420, 100, 1.5) and is listed in defaults_used; the grout-ground bond_stress depends on the ground, so give it."},
        },
        "returns": {
            "surface_points": "Ground surface profile [[x, z], ...].",
            "soil_layers": "Assembled soil layers with geometry and properties, as slope_stability's soil_layers.",
            "gwt_points": "GWT profile [[x, z], ...] or null.",
            "nails": "Soil nails from the DXF nail lines, as slope_stability's nails (only when there are any). The whole result is slope_stability.analyze_slope / search_critical_surface input.",
            "defaults_used": "Only when a value stood in for an input left out: {'soil_properties[<name>].<input>': value, 'nail_defaults.<input>': value}. Take it out before passing the result on to slope_stability.",
        },
    },
    "build_fem_inputs": {
        "category": "DXF Import",
        "brief": "Convert DXF parse result to fem2d input format with FEM soil properties.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Output from parse_dxf_geometry (surface_points, boundary_profiles, etc.)."},
            "soil_properties": {"type": "list", "required": True, "description": "List of dicts: {name, gamma, phi, c, E, nu, psi, model, hs_params}. A drawing gives geometry only, so no unit weight, stiffness or strength is assumed: every layer must give gamma, E and nu, and a layer with a strength model (mc or hs) phi; a layer without them is refused. Left out and listed in defaults_used: c 0, psi 0, model 'mc'."},
        },
        "returns": {
            "surface_points": "Ground surface profile.",
            "soil_layers": "FEM layer dicts with geometry and material properties.",
            "gwt": "GWT array or null.",
            "boundary_polylines": "Boundary polylines for layer assignment.",
            "defaults_used": "Only when a value stood in for an input left out: {'soil_properties[<name>].<input>': value}.",
        },
    },
}

# A left-out ``units`` is a case, not a value (GM-165): the drawing's own.
declare_when_absent(METHOD_INFO, {
    "parse_geometry": {
        "units": ("the units the drawing's header states ($INSUNITS)",
                  "the result's units_used and units_source say which; a "
                  "drawing whose header states none is refused")},
})
