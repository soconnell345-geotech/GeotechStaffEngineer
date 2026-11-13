"""DXF export adapter — write cross-section geometry to DXF files."""

import base64
import os

from funhouse_agent.adapters import (
    check_nested_list, reject_unknown_params, require_params,
)

# Geometry params shared by both export methods.
_GEOMETRY_PARAMS = ("surface_points", "boundary_profiles", "gwt_points",
                    "nail_lines", "text_annotations", "units")

# Keys of the nail_lines / text_annotations dicts, as dxf_export writes them
# (plus the 'layer' dxf_import's parse_geometry gives each annotation, so a
# parsed section exports as it came). An annotation's other keys were read
# with .get and silently left out: one placed by 'z' (the section's own name
# for elevation) landed at y = 0 (GM-58).
_NAIL_KEYS = ("x_head", "z_head", "x_tip", "z_tip")
_TEXT_KEYS = ("text", "x", "y", "height", "layer")
_KEY_HINTS = {"z": ("y",), "label": ("text",), "size": ("height",),
              "textheight": ("height",)}


def _check_items(params, method):
    check_nested_list(params.get("nail_lines"), _NAIL_KEYS, method=method,
                      label="nail_lines", hints=_KEY_HINTS)
    check_nested_list(params.get("text_annotations"), _TEXT_KEYS,
                      method=method, label="text_annotations",
                      hints=_KEY_HINTS)


def _run_export_geometry_to_dxf(params):
    from dxf_export import export_to_dxf

    reject_unknown_params(params, _GEOMETRY_PARAMS + ("output_path",),
                          method="export_geometry_to_dxf")
    require_params(params, ["output_path"], method="export_geometry_to_dxf",
                   valid=_GEOMETRY_PARAMS + ("output_path",))
    _check_items(params, "export_geometry_to_dxf")
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
    # ``output_path`` too, as every other writer reports it: it is the key
    # the app recognises a written file by (webapp/output_capture.py), and a
    # DXF reported only as ``filepath`` was never delivered (brief 5, N2).
    out["output_path"] = abs_path
    out["file_exists"] = file_exists
    out["file_size_bytes"] = os.path.getsize(abs_path) if file_exists else 0
    if not file_exists:
        out["error"] = (
            f"DXF export ran but no file was found at '{abs_path}' after "
            "writing. The target location may not be writable from this "
            "process; retry with a bare file name (e.g. 'section.dxf'), "
            "which lands in the working folder."
        )
    return out


def _run_export_to_dxf_bytes(params):
    from dxf_export import to_dxf_bytes

    reject_unknown_params(params, _GEOMETRY_PARAMS,
                          method="export_to_dxf_bytes")
    _check_items(params, "export_to_dxf_bytes")
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
            "output_path": {"type": "str", "required": True, "description": "A file name for the DXF, e.g. 'section.dxf'; in the app it is written into this conversation's folder and attached to the conversation."},
            "surface_points": {"type": "list", "required": False, "description": "Ground surface as [[x,z],...] or [{x,z},...]"},
            "boundary_profiles": {"type": "dict", "required": False, "description": "Soil boundaries: {name: [[x,z],...]}"},
            "gwt_points": {"type": "list", "required": False, "description": "Groundwater table as [[x,z],...]"},
            "nail_lines": {"type": "list", "required": False, "description": "Nail lines: [{x_head, z_head, x_tip, z_tip},...]"},
            "text_annotations": {"type": "list", "required": False, "description": "Text labels: [{text, x, y},...]"},
            "units": {"type": "str", "required": False, "default": "m", "allowed_values": ["m", "mm", "cm", "ft", "in"], "value_aliases": {"meters": "m", "millimeters": "mm", "centimeters": "cm", "feet": "ft", "inches": "in"}, "any_case": "strip", "description": "Coordinate units."},
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
            "units": {"type": "str", "required": False, "default": "m", "allowed_values": ["m", "mm", "cm", "ft", "in"], "value_aliases": {"meters": "m", "millimeters": "mm", "centimeters": "cm", "feet": "ft", "inches": "in"}, "any_case": "strip", "description": "Coordinate units."},
        },
        "returns": {
            "dxf_base64": "Base64-encoded DXF file content.",
            "size_bytes": "Size of DXF content in bytes.",
        },
    },
}


# What a left-out input with no default stands for (BACKLOG GM-165):
# no groundwater line.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
declare_when_absent(METHOD_INFO, {
    method: {"gwt_points": ("no groundwater line",
                            "none is written to the drawing")}
    for method in ("export_geometry_to_dxf", "export_to_dxf_bytes")})


# ---------------------------------------------------------------------------
# Drawing sheets (BACKLOG GM-313): one typed description -> a DXF with real
# dimensions, layers, hatches and leaders, and the same drawing as a PDF
# page and a PNG. Data in, never code; the stamp, layers, text heights,
# border and title block are the code's.
# ---------------------------------------------------------------------------

_SHEET_SIZES = ["letter", "tabloid", "a4", "a3"]
_SCALES = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]


def _sheet_result(result, output_path):
    """A written sheet's result as every writer reports a file."""
    out = result.to_dict()
    out["output_path"] = out["filepath"]
    files = [out["filepath"], out["pdf_path"], out["png_path"]]
    out["file_exists"] = all(os.path.isfile(p) for p in files)
    out["file_size_bytes"] = (os.path.getsize(out["filepath"])
                              if os.path.isfile(out["filepath"]) else 0)
    out["stamp"] = "CALCULATION SKETCH - NOT FOR CONSTRUCTION"
    if not out["file_exists"]:
        out["error"] = (
            f"The sheet was drawn but its files were not found at "
            f"'{os.path.splitext(out['filepath'])[0]}.*' after writing. "
            "Retry with a bare file name (e.g. 'wall_section.dxf'), which "
            "lands in the working folder.")
    return out


def _sheet_name(output_path, method):
    """``output_path`` checked: a file name ending in .dxf (or no ending)."""
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(f"{method}: output_path must be a file name, "
                         "e.g. 'wall_section.dxf'.")
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in ("", ".dxf"):
        raise ValueError(
            f"{method}: output_path names the DXF ('<name>.dxf'); the PDF "
            f"and the PNG are written beside it as '<name>.pdf' and "
            f"'<name>.png'. Got '{os.path.basename(output_path)}'.")
    return output_path


def _run_draw_wall_section(params):
    from dxf_export import wall_section_drawing, write_sheet

    valid = ("description", "output_path", "sheet_size", "scale",
             "title_block")
    reject_unknown_params(params, valid, method="draw_wall_section")
    require_params(params, ["description", "output_path"],
                   method="draw_wall_section", valid=valid)
    path = _sheet_name(params["output_path"], "draw_wall_section")
    drawing = wall_section_drawing(
        params["description"],
        sheet={"size": params.get("sheet_size") or "letter",
               "scale": params.get("scale")},
        title_block=params.get("title_block"))
    return _sheet_result(write_sheet(drawing, path), path)


def _run_draw_sheet(params):
    from dxf_export import Drawing, write_sheet

    valid = ("drawing", "output_path")
    reject_unknown_params(params, valid, method="draw_sheet")
    require_params(params, ["drawing", "output_path"], method="draw_sheet",
                   valid=valid)
    path = _sheet_name(params["output_path"], "draw_sheet")
    drawing = Drawing.from_dict(params["drawing"])
    return _sheet_result(write_sheet(drawing, path), path)


_SHEET_RETURNS = {
    "output_path": "The DXF file (also 'filepath'): model space, "
                   "millimetres, real DIMENSION, HATCH and LEADER entities "
                   "on fixed layers.",
    "pdf_path": "The same drawing as a one-page vector PDF of the sheet's "
                "size, drawn from the DXF (lettering as outlines, not "
                "selectable text).",
    "png_path": "The same sheet as a PNG (200 dpi), drawn from the PDF.",
    "file_exists": "True if all three files were verified on disk. Trust "
                   "this field — do NOT verify with agent-side filesystem "
                   "tools (they may be sandboxed).",
    "sheet": "Sheet size used (landscape), with sheet_width_mm and "
             "sheet_height_mm.",
    "scale": "The sheet's scale as '1:N' (scale_n is N); scale_chosen_by "
             "says 'code' or 'given'.",
    "counts": "outlines, hatches, lines, bars, bar_dots, dimensions, "
              "callouts, notes, table_rows drawn.",
    "stamp": "The stamp printed on the sheet; no input removes it.",
    "warnings": "What the reader should know, e.g. bars drawn wider than "
                "to scale so they print, a hook length that does not fit "
                "the base, a title-block field cut to fit.",
}

METHOD_REGISTRY.update({
    "draw_wall_section": _run_draw_wall_section,
    "draw_sheet": _run_draw_sheet,
})

METHOD_INFO.update({
    "draw_wall_section": {
        "category": "Drawing sheets",
        "brief": "Draw a reinforced cantilever retaining wall section "
                 "(concrete, bars, dimensions, callouts, bar schedule) as "
                 "a DXF, a PDF sheet and a PNG, from a "
                 "reinforced_wall_section description.",
        "description": (
            "Fixed code draws the section from the description: it designs "
            "nothing and computes no steel. Drawn: the concrete outline to "
            "scale with a concrete hatch; the ground in front (at "
            "embedment_depth) and behind (from the top of the stem at "
            "backfill_slope_deg); every bar set; dimensions (real DIMENSION "
            "entities, millimetres, the number is the entity's own "
            "measurement) for wall height, stem height, base thickness, "
            "base width, toe, stem thickness top and bottom, heel, a shear "
            "key, and each cover; one callout per bar set (its mark and "
            "callout text); a bar schedule; the notes; a title block, a "
            "scale bar and the stamp 'CALCULATION SKETCH - NOT FOR "
            "CONSTRUCTION'. Rules: an in_section stem bar runs the full "
            "height of the stem and into the base, down to the base's "
            "bottom steel; an in_section base bar runs the full width of "
            "the base; along_wall bars are dots at their spacing, inside "
            "the in_section bars of the same face; no bar is cut off or "
            "lapped. A hooked stem bar is turned toward the opposite face "
            "of the stem (a back-face bar toward the toe), the correct "
            "orientation of USACE EM 1110-2-2104 (2023) Fig. 2-1; the "
            "hook's bend and tail are schematic, not detailed. The PDF and "
            "the PNG are drawn from the DXF itself."),
        "parameters": {
            "description": {
                "type": "dict", "required": True,
                "description": (
                    "A 'reinforced_wall_section/1' description (unknown "
                    "keys are refused by name): schema; title; geometry "
                    "{wall_height (underside of base to top of stem), "
                    "base_width, toe_length, heel_length, "
                    "stem_thickness_top, stem_thickness_base, "
                    "base_thickness, stem_batter 'back'|'front', "
                    "embedment_depth, backfill_slope_deg, has_shear_key, "
                    "key_depth, key_width, key_distance_from_toe} in "
                    "metres and degrees; materials {fc_MPa, fy_MPa}; bars: "
                    "a list of bar sets, each {mark (unique), element "
                    "'stem'|'toe'|'heel'|'base', face (stem: 'back' = "
                    "retained side, 'front', 'both'; base: 'top', "
                    "'bottom', 'both'), runs 'in_section' (drawn as a "
                    "line) | 'along_wall' (drawn as dots), role "
                    "'flexure'|'temperature_shrinkage', size (e.g. "
                    "'No. 6'), diameter_mm, spacing_mm, cover_mm (clear "
                    "cover from the named face), end (null, or {type: "
                    "'standard_hook_90', into: 'base', ldh_mm} on an "
                    "in_section stem bar), callout (the text printed "
                    "beside the mark), As_required_mm2_per_m, "
                    "As_provided_mm2_per_m}; notes: a list of lines; "
                    "basis: one line naming the design manual.")},
            "output_path": {
                "type": "str", "required": True,
                "description": (
                    "A file name for the DXF, e.g. 'wall_section.dxf'; "
                    "'<name>.pdf' and '<name>.png' are written beside it. "
                    "In the app all three go into this conversation's "
                    "folder and are attached to the conversation.")},
            "sheet_size": {
                "type": "str", "required": False, "default": "letter",
                "allowed_values": list(_SHEET_SIZES),
                "description": (
                    "Sheet size, always landscape. 'letter' matches a "
                    "calc package's pages; a tall wall draws larger on "
                    "'tabloid' or 'a3'.")},
            "scale": {
                "type": "int", "required": False,
                "allowed_values": list(_SCALES),
                "description": (
                    "N of the scale 1:N. Left out, the code takes the "
                    "largest standard scale at which the section with its "
                    "dimensions and callouts fits the sheet. A given "
                    "scale that does not fit is refused, with the one "
                    "that does.")},
            "title_block": {
                "type": "dict", "required": False,
                "description": (
                    "Title-block fields, all optional text: title, "
                    "project, project_number, drawn_by, checked_by, date, "
                    "drawing_number. Unknown keys are refused. The scale "
                    "and sheet size it prints are the code's.")},
        },
        "returns": dict(_SHEET_RETURNS),
    },
    "draw_sheet": {
        "category": "Drawing sheets",
        "brief": "Draw a general drawing sheet (outlines with hatches, "
                 "lines, bars, dimensions, callouts, notes, a table, a "
                 "title block) as a DXF, a PDF sheet and a PNG, from a "
                 "typed description.",
        "description": (
            "Fixed code draws exactly what the description holds; it "
            "takes data, never code. Coordinates are millimetres at true "
            "size, x right and y up; the code chooses the scale and lays "
            "out the sheet. Fixed by the code and closed to the "
            "description: layer names, colours and line weights; text "
            "heights (2.5 mm or more on the sheet); the border, title "
            "block layout and scale bar; and the stamp 'CALCULATION "
            "SKETCH - NOT FOR CONSTRUCTION' on every sheet. A dimension's "
            "number is the DIMENSION entity's own measurement of its two "
            "points, in whole millimetres: a description cannot type a "
            "number over it. The PDF and the PNG are drawn from the DXF "
            "itself."),
        "parameters": {
            "drawing": {
                "type": "dict", "required": True,
                "description": (
                    "The drawing (unknown keys are refused by name). "
                    "outlines: [{points [[x,y],...] (closed polygon), "
                    "layer 'concrete'|'soil'|'structure'|'general', hatch "
                    "'none'|'concrete'|'earth'|'gravel'}]. lines: "
                    "[{points, layer 'general'|'ground'|'hidden'|"
                    "'centre'}]. bars: [{points, diameter_mm, kind 'path' "
                    "(centreline, drawn as a heavy line diameter_mm wide) "
                    "| 'dots' (each point a bar seen end-on), "
                    "bend_radius_mm (corners of a path; 0 = sharp)}]. "
                    "dimensions: [{p1, p2, direction 'horizontal'|"
                    "'vertical'|'aligned', offset_mm (where the dimension "
                    "line sits, mm ON THE SHEET: horizontal + above / - "
                    "below the points; vertical + right / - left), suffix "
                    "(e.g. ' mm'), second_unit null|'in'|'ft'|'ft-in'|'m' "
                    "(the same measurement in brackets)}]. callouts: "
                    "[{at [x,y] (arrow), text_at [x,y] (where the leader "
                    "lands; text is set beyond it), text, mark (optional, "
                    "drawn in a circle), width_mm (wrap width on the "
                    "sheet, default 45)}]. notes: [text]. table: {title, "
                    "headers [..], rows [[..]]} (at most 8 columns). "
                    "title_block: {title, project, project_number, "
                    "drawn_by, checked_by, date, drawing_number}. sheet: "
                    "{size 'letter' (default)|'tabloid'|'a4'|'a3', always "
                    "landscape; scale N of 1:N from 1, 2, 5, 10, 20, 25, "
                    "50, 100, 200, 250, 500, 1000, or left out for the "
                    "largest that fits}.")},
            "output_path": {
                "type": "str", "required": True,
                "description": (
                    "A file name for the DXF, e.g. 'detail.dxf'; "
                    "'<name>.pdf' and '<name>.png' are written beside it. "
                    "In the app all three go into this conversation's "
                    "folder and are attached to the conversation.")},
        },
        "returns": dict(_SHEET_RETURNS),
    },
})

declare_when_absent(METHOD_INFO, {
    "draw_wall_section": {
        "scale": ("chosen by the code",
                  "the largest standard scale at which the section with "
                  "its dimensions and callouts fits the sheet"),
        "title_block": ("the description's title, the other fields blank",
                        "the title block prints the description's title"),
    }})
