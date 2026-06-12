"""DXF export adapter — write cross-section geometry to DXF files."""

import base64
import os

from funhouse_agent.adapters import reject_unknown_params, require_params

# Geometry params shared by both export methods.
_GEOMETRY_PARAMS = ("surface_points", "boundary_profiles", "gwt_points",
                    "nail_lines", "text_annotations", "units")


def _run_export_geometry_to_dxf(params):
    from dxf_export import export_to_dxf

    reject_unknown_params(params, _GEOMETRY_PARAMS + ("output_path",),
                          method="export_geometry_to_dxf")
    require_params(params, ["output_path"], method="export_geometry_to_dxf",
                   valid=_GEOMETRY_PARAMS + ("output_path",))
    params = dict(params)  # pops below must not mutate the caller's dict
    output_path = params.pop("output_path")

    # Convert boundary_profiles values from list-of-dicts to list-of-tuples
    boundaries = params.pop("boundary_profiles", None)
    if boundaries:
        boundaries = {
            name: [(p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
                   for p in pts]
            for name, pts in boundaries.items()
        }

    # Convert surface_points from list-of-dicts to list-of-tuples
    surface = params.pop("surface_points", None)
    if surface:
        surface = [(p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
                   for p in surface]

    # Convert gwt_points
    gwt = params.pop("gwt_points", None)
    if gwt:
        gwt = [(p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
               for p in gwt]

    # nail_lines can stay as list-of-dicts
    nail_lines = params.pop("nail_lines", None)

    # Convert text_annotations
    text_annotations = params.pop("text_annotations", None)

    units = params.pop("units", "m")

    result = export_to_dxf(
        filepath=output_path,
        surface_points=surface,
        boundary_profiles=boundaries,
        gwt_points=gwt,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units=units,
    )
    out = result.to_dict()
    # Verify the write on the REAL filesystem so callers (LLM agents) never
    # need a separate ls/read to trust the save — agent-side filesystem tools
    # may be sandboxed/virtual and cannot see this file.
    abs_path = os.path.abspath(output_path)
    file_exists = os.path.isfile(abs_path)
    out["filepath"] = abs_path
    out["file_exists"] = file_exists
    out["file_size_bytes"] = os.path.getsize(abs_path) if file_exists else 0
    if not file_exists:
        out["error"] = (
            f"DXF export ran but no file was found at '{abs_path}' after "
            "writing. The target location may not be writable from this "
            "process; retry with a local path (e.g. under /tmp)."
        )
    return out


def _run_export_to_dxf_bytes(params):
    from dxf_export import to_dxf_bytes

    reject_unknown_params(params, _GEOMETRY_PARAMS,
                          method="export_to_dxf_bytes")
    params = dict(params)

    # Convert geometry inputs same as above
    boundaries = params.pop("boundary_profiles", None)
    if boundaries:
        boundaries = {
            name: [(p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
                   for p in pts]
            for name, pts in boundaries.items()
        }

    surface = params.pop("surface_points", None)
    if surface:
        surface = [(p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
                   for p in surface]

    gwt = params.pop("gwt_points", None)
    if gwt:
        gwt = [(p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
               for p in gwt]

    nail_lines = params.pop("nail_lines", None)
    text_annotations = params.pop("text_annotations", None)
    units = params.pop("units", "m")

    dxf_bytes = to_dxf_bytes(
        surface_points=surface,
        boundary_profiles=boundaries,
        gwt_points=gwt,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units=units,
    )
    return {
        "dxf_base64": base64.b64encode(dxf_bytes).decode("ascii"),
        "size_bytes": len(dxf_bytes),
    }


METHOD_REGISTRY = {
    "export_geometry_to_dxf": _run_export_geometry_to_dxf,
    "export_to_dxf_bytes": _run_export_to_dxf_bytes,
}

METHOD_INFO = {
    "export_geometry_to_dxf": {
        "category": "Export",
        "brief": "Export cross-section geometry to a DXF file.",
        "parameters": {
            "output_path": {"type": "str", "required": True, "description": "Output file path for the DXF file."},
            "surface_points": {"type": "list", "required": False, "description": "Ground surface as [[x,z],...] or [{x,z},...]"},
            "boundary_profiles": {"type": "dict", "required": False, "description": "Soil boundaries: {name: [[x,z],...]}"},
            "gwt_points": {"type": "list", "required": False, "description": "Groundwater table as [[x,z],...]"},
            "nail_lines": {"type": "list", "required": False, "description": "Nail lines: [{x_head, z_head, x_tip, z_tip},...]"},
            "text_annotations": {"type": "list", "required": False, "description": "Text labels: [{text, x, y},...]"},
            "units": {"type": "str", "required": False, "default": "m", "allowed_values": ["m", "mm", "cm", "ft", "in"], "description": "Coordinate units."},
        },
        "returns": {
            "filepath": "Absolute path of the saved DXF file.",
            "file_exists": "True if the file was verified on disk after writing. "
                           "Trust this field — do NOT try to verify the file with "
                           "agent-side filesystem tools (they may be sandboxed).",
            "file_size_bytes": "Size of the saved file (0 if file_exists is false).",
            "n_layers": "Number of layers created.",
            "n_entities": "Total entity count.",
        },
    },
    "export_to_dxf_bytes": {
        "category": "Export",
        "brief": "Export geometry to DXF as base64-encoded bytes (no file save).",
        "parameters": {
            "surface_points": {"type": "list", "required": False, "description": "Ground surface as [[x,z],...] or [{x,z},...]"},
            "boundary_profiles": {"type": "dict", "required": False, "description": "Soil boundaries: {name: [[x,z],...]}"},
            "gwt_points": {"type": "list", "required": False, "description": "Groundwater table as [[x,z],...]"},
            "nail_lines": {"type": "list", "required": False, "description": "Nail lines: [{x_head, z_head, x_tip, z_tip},...]"},
            "text_annotations": {"type": "list", "required": False, "description": "Text labels: [{text, x, y},...]"},
            "units": {"type": "str", "required": False, "default": "m", "allowed_values": ["m", "mm", "cm", "ft", "in"], "description": "Coordinate units."},
        },
        "returns": {
            "dxf_base64": "Base64-encoded DXF file content.",
            "size_bytes": "Size of DXF content in bytes.",
        },
    },
}
