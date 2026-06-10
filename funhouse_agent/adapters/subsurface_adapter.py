"""Subsurface characterization adapter — flat dict -> subsurface_characterization API -> dict.

Supports DIGGS XML ingestion via parse_diggs with a session cache so that
subsequent plot calls can reference the parsed site by key instead of
re-sending the entire data dict.
"""

import numpy as np

from funhouse_agent.adapters import clean_result

# Module-level cache: site_key -> SiteModel.  Populated by parse_diggs,
# consumed by any plot method that receives site_key instead of site_data.
_site_cache: dict = {}


def _resolve_site(params: dict):
    """Return a SiteModel from either site_key (cached) or site_data (dict)."""
    from subsurface_characterization import load_site_from_dict

    site_key = params.get("site_key")
    if site_key:
        if site_key not in _site_cache:
            available = sorted(_site_cache.keys()) or ["(none)"]
            raise KeyError(
                f"site_key '{site_key}' not found in cache. "
                f"Available keys: {available}. Call parse_diggs or load_site first."
            )
        return _site_cache[site_key]

    site_data = params.get("site_data")
    if site_data:
        return load_site_from_dict(site_data)

    raise ValueError("Provide either 'site_key' (from parse_diggs/load_site) or 'site_data' dict.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _run_parse_diggs(params: dict) -> dict:
    from subsurface_characterization import parse_diggs

    filepath = params.get("file_path")
    content = params.get("content")
    result = parse_diggs(filepath=filepath, content=content)

    # Cache the SiteModel for subsequent calls
    site_key = result.site.project_name or "site"
    # Avoid collisions by appending a counter
    base_key = site_key
    counter = 1
    while site_key in _site_cache:
        site_key = f"{base_key}_{counter}"
        counter += 1
    _site_cache[site_key] = result.site

    # Build per-investigation summary including available parameters
    inv_summaries = []
    for inv in result.site.investigations:
        param_names = sorted({m.parameter for m in inv.measurements})
        inv_summaries.append({
            "investigation_id": inv.investigation_id,
            "investigation_type": inv.investigation_type,
            "x": inv.x,
            "y": inv.y,
            "elevation_m": inv.elevation_m,
            "total_depth_m": inv.total_depth_m,
            "gwl_depth_m": inv.gwl_depth_m,
            "n_measurements": len(inv.measurements),
            "n_lithology": len(inv.lithology),
            "parameters": param_names,
        })

    return clean_result({
        "site_key": site_key,
        "project_name": result.site.project_name,
        "n_investigations": result.n_investigations,
        "n_measurements": result.n_measurements,
        "n_lithology_intervals": result.n_lithology_intervals,
        "warnings": result.warnings,
        "investigations": inv_summaries,
    })


def _run_load_site(params: dict) -> dict:
    from subsurface_characterization import load_site_from_dict
    site = load_site_from_dict(params["site_data"])

    # Cache the SiteModel
    site_key = site.project_name or "site"
    base_key = site_key
    counter = 1
    while site_key in _site_cache:
        site_key = f"{base_key}_{counter}"
        counter += 1
    _site_cache[site_key] = site

    return {
        "site_key": site_key,
        "project_name": site.project_name,
        "n_investigations": len(site.investigations),
        "investigations": [
            {
                "investigation_id": inv.investigation_id,
                "investigation_type": inv.investigation_type,
                "x": inv.x,
                "y": inv.y,
                "elevation_m": inv.elevation_m,
                "total_depth_m": inv.total_depth_m,
                "n_measurements": len(inv.measurements),
                "n_lithology": len(inv.lithology),
            }
            for inv in site.investigations
        ],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _run_plot_parameter_vs_depth(params: dict) -> dict:
    from subsurface_characterization import plot_parameter_vs_depth
    site = _resolve_site(params)
    result = plot_parameter_vs_depth(
        site=site,
        parameter=params["parameter"],
        color_by=params.get("color_by", "investigation"),
        use_elevation=params.get("use_elevation", False),
        show_trend=params.get("show_trend", False),
        show_bands=params.get("show_bands", False),
        band_sigma=params.get("band_sigma", 1.0),
        group_trends_by=params.get("group_trends_by", ""),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


def _run_plot_atterberg_limits(params: dict) -> dict:
    from subsurface_characterization import plot_atterberg_limits
    site = _resolve_site(params)
    result = plot_atterberg_limits(
        site=site,
        use_elevation=params.get("use_elevation", False),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


def _run_plot_multi_parameter(params: dict) -> dict:
    from subsurface_characterization import plot_multi_parameter
    site = _resolve_site(params)
    result = plot_multi_parameter(
        site=site,
        parameters=params["parameters"],
        use_elevation=params.get("use_elevation", False),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


def _run_plot_plan_view(params: dict) -> dict:
    from subsurface_characterization import plot_plan_view
    site = _resolve_site(params)
    result = plot_plan_view(
        site=site,
        color_by=params.get("color_by", "type"),
        label_field=params.get("label_field", "id"),
        parameter_for_color=params.get("parameter_for_color", ""),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


def _run_plot_cross_section(params: dict) -> dict:
    from subsurface_characterization import plot_cross_section
    site = _resolve_site(params)
    result = plot_cross_section(
        site=site,
        investigation_ids=params["investigation_ids"],
        use_elevation=params.get("use_elevation", True),
        annotate_parameter=params.get("annotate_parameter", ""),
        column_width=params.get("column_width", 0.3),
        show_gwl=params.get("show_gwl", True),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


# ---------------------------------------------------------------------------
# Format adapters — optional, dependency-backed file ingest/validate
# (folded in from the former pygef_agent / ags4_agent / pydiggs_agent modules)
# ---------------------------------------------------------------------------

def _run_parse_cpt(params: dict) -> dict:
    from subsurface_characterization.formats.gef import has_pygef, parse_cpt_file

    if not has_pygef():
        return {"error": "pygef is not installed. Install with: pip install pygef"}

    result = parse_cpt_file(
        file_path=params.get("file_path"),
        engine=params.get("engine", "auto"),
        index=params.get("index", 0),
    )
    return clean_result(result.to_dict())


def _run_parse_borehole(params: dict) -> dict:
    from subsurface_characterization.formats.gef import has_pygef, parse_bore_file

    if not has_pygef():
        return {"error": "pygef is not installed. Install with: pip install pygef"}

    result = parse_bore_file(
        file_path=params.get("file_path"),
        engine=params.get("engine", "auto"),
        index=params.get("index", 0),
    )
    return clean_result(result.to_dict())


def _run_read_ags4(params: dict) -> dict:
    from subsurface_characterization.formats.ags4 import has_ags4, read_ags4

    if not has_ags4():
        return {"error": "python-ags4 is not installed. Install with: pip install python-ags4"}

    result = read_ags4(
        filepath=params.get("file_path"),
        content=params.get("content"),
        encoding=params.get("encoding", "utf-8"),
        include_data=params.get("include_data", True),
        convert_numeric=params.get("convert_numeric", True),
    )
    return clean_result(result.to_dict())


def _run_validate_ags4(params: dict) -> dict:
    from subsurface_characterization.formats.ags4 import has_ags4, validate_ags4

    if not has_ags4():
        return {"error": "python-ags4 is not installed. Install with: pip install python-ags4"}

    result = validate_ags4(
        filepath=params.get("file_path"),
        content=params.get("content"),
        encoding=params.get("encoding", "utf-8"),
    )
    return clean_result(result.to_dict())


def _run_validate_diggs_schema(params: dict) -> dict:
    from subsurface_characterization.formats.diggs_validation import (
        has_pydiggs, validate_diggs_schema,
    )

    if not has_pydiggs():
        return {"error": "pydiggs is not installed. Install with: pip install pydiggs"}

    result = validate_diggs_schema(
        filepath=params.get("file_path"),
        content=params.get("content"),
        schema_version=params.get("schema_version", "2.6"),
    )
    return clean_result(result.to_dict())


def _run_validate_diggs_dictionary(params: dict) -> dict:
    from subsurface_characterization.formats.diggs_validation import (
        has_pydiggs, validate_diggs_dictionary,
    )

    if not has_pydiggs():
        return {"error": "pydiggs is not installed. Install with: pip install pydiggs"}

    result = validate_diggs_dictionary(
        filepath=params.get("file_path"),
        content=params.get("content"),
    )
    return clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _run_compute_trend(params: dict) -> dict:
    from subsurface_characterization import compute_trend
    depths = np.asarray(params["depths"], dtype=float)
    values = np.asarray(params["values"], dtype=float)
    result = compute_trend(
        depths=depths,
        values=values,
        trend_type=params.get("trend_type", "linear"),
        parameter=params.get("parameter", ""),
        group_label=params.get("group_label", ""),
    )
    return clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "parse_diggs": _run_parse_diggs,
    "load_site": _run_load_site,
    "plot_parameter_vs_depth": _run_plot_parameter_vs_depth,
    "plot_atterberg_limits": _run_plot_atterberg_limits,
    "plot_multi_parameter": _run_plot_multi_parameter,
    "plot_plan_view": _run_plot_plan_view,
    "plot_cross_section": _run_plot_cross_section,
    "compute_trend": _run_compute_trend,
    # Format adapters (folded-in pygef / ags4 / pydiggs)
    "parse_cpt": _run_parse_cpt,
    "parse_bore": _run_parse_borehole,
    "read_ags4": _run_read_ags4,
    "validate_ags4": _run_validate_ags4,
    "validate_diggs_schema": _run_validate_diggs_schema,
    "validate_diggs_dictionary": _run_validate_diggs_dictionary,
}

_SITE_KEY_DOC = "Key returned by parse_diggs or load_site (preferred). Alternative to site_data."
_SITE_DATA_DOC = "Nested dict with project_name, investigations. Alternative to site_key."

METHOD_INFO = {
    "parse_diggs": {
        "category": "Subsurface Characterization",
        "brief": "Parse DIGGS 2.6/2.5.a XML into site model. Returns site_key for use in subsequent plot calls.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file (DBFS, workspace, or local). Provide file_path, content, or attachment_key."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
            "attachment_key": {"type": "str", "required": False, "description": "Key of a DIGGS file uploaded via the chat widget. The file bytes are decoded to XML automatically."},
        },
        "returns": {
            "site_key": "Key to reference this site in subsequent calls (plot_*, compute_trend).",
            "project_name": "Project name from DIGGS file.",
            "n_investigations": "Number of borings/CPTs/etc extracted.",
            "n_measurements": "Total number of test measurements (SPT, Atterberg, etc).",
            "n_lithology_intervals": "Total lithology intervals.",
            "warnings": "Any parsing warnings.",
            "investigations": "Per-investigation summary with available parameters.",
        },
    },
    "load_site": {
        "category": "Subsurface Characterization",
        "brief": "Load site model from nested dict structure. Returns site_key for subsequent plot calls.",
        "parameters": {
            "site_data": {"type": "dict", "required": True, "description": "Nested dict with project_name, investigations [{investigation_id, x, y, elevation_m, measurements, lithology}]."},
        },
        "returns": {
            "site_key": "Key to reference this site in subsequent calls.",
            "project_name": "Site project name.",
            "n_investigations": "Number of investigation locations.",
            "investigations": "Summary of each investigation.",
        },
    },
    "plot_parameter_vs_depth": {
        "category": "Subsurface Characterization",
        "brief": "Scatter plot of a soil parameter vs depth (Plotly interactive).",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "parameter": {"type": "str", "required": True, "description": "Parameter name to plot (e.g., 'N_spt', 'cu_kPa', 'wn_pct')."},
            "color_by": {"type": "str", "required": False, "default": "investigation", "allowed_values": ["investigation", "uscs", "none"], "description": "Color coding."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation instead of depth."},
            "show_trend": {"type": "bool", "required": False, "default": False, "description": "Overlay linear trendline."},
            "show_bands": {"type": "bool", "required": False, "default": False, "description": "Overlay prediction bands."},
            "band_sigma": {"type": "float", "required": False, "default": 1.0, "description": "Number of sigma for prediction bands."},
            "group_trends_by": {"type": "str", "required": False, "default": "", "description": "Group trends by 'uscs' for separate trends per soil class."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Number of data points plotted.",
            "n_investigations": "Number of investigations shown.",
        },
    },
    "plot_atterberg_limits": {
        "category": "Subsurface Characterization",
        "brief": "Plot Atterberg limits (LL-PL bracket) with natural moisture content overlay.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Number of data points plotted.",
        },
    },
    "plot_multi_parameter": {
        "category": "Subsurface Characterization",
        "brief": "Side-by-side subplots of multiple parameters with shared depth axis.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "parameters": {"type": "array", "required": True, "description": "Parameter names for each panel (e.g., ['N_spt','cu_kPa','wn_pct'])."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Total data points across all panels.",
            "parameters": "Parameters plotted.",
        },
    },
    "plot_plan_view": {
        "category": "Subsurface Characterization",
        "brief": "Plan view map of investigation locations with optional parameter-based coloring.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "color_by": {"type": "str", "required": False, "default": "type", "allowed_values": ["type", "parameter"], "description": "Color mode: 'type' (investigation type) or 'parameter' (avg value)."},
            "label_field": {"type": "str", "required": False, "default": "id", "description": "Label: 'id', 'depth_to_rock', 'gwl', 'fill_thickness', or parameter name."},
            "parameter_for_color": {"type": "str", "required": False, "default": "", "description": "Parameter name when color_by='parameter'."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_investigations": "Number of investigation locations shown.",
        },
    },
    "plot_cross_section": {
        "category": "Subsurface Characterization",
        "brief": "Cross-section profile view with USCS-colored lithology columns along a transect.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "investigation_ids": {"type": "array", "required": True, "description": "Ordered list of investigation IDs defining the transect (e.g., ['B-1','B-2','B-3'])."},
            "use_elevation": {"type": "bool", "required": False, "default": True, "description": "If True, Y-axis is elevation (default). False for depth."},
            "annotate_parameter": {"type": "str", "required": False, "default": "", "description": "Parameter to annotate next to columns (e.g., 'N_spt')."},
            "column_width": {"type": "float", "required": False, "default": 0.3, "description": "Width of lithology columns (fraction of spacing)."},
            "show_gwl": {"type": "bool", "required": False, "default": True, "description": "Show groundwater level dashed line."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_investigations": "Number of investigations in the cross-section.",
        },
    },
    "compute_trend": {
        "category": "Subsurface Characterization",
        "brief": "Fit linear or log-linear depth-value trend with statistics (R-squared, COV).",
        "parameters": {
            "depths": {"type": "array", "required": True, "description": "Depth values (m)."},
            "values": {"type": "array", "required": True, "description": "Measurement values."},
            "trend_type": {"type": "str", "required": False, "default": "linear", "allowed_values": ["linear", "log_linear"], "description": "Trend type."},
            "parameter": {"type": "str", "required": False, "default": "", "description": "Parameter name for labeling."},
            "group_label": {"type": "str", "required": False, "default": "", "description": "Group label (e.g., USCS class)."},
        },
        "returns": {
            "slope": "Regression slope.",
            "intercept": "Regression intercept.",
            "r_squared": "Coefficient of determination.",
            "std_residual": "Standard deviation of residuals.",
            "cov": "Coefficient of variation.",
        },
    },
    # -----------------------------------------------------------------------
    # Format adapters — file ingest / validation (optional, dependency-backed)
    # -----------------------------------------------------------------------
    "parse_cpt": {
        "category": "File Import",
        "brief": "Parse a CPT file (GEF or BRO-XML, via pygef) into depth/qc/fs/u2 arrays (kPa).",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to CPT file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "gef", "xml"], "description": "Parser engine."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_points": "Number of data points.",
            "alias": "Test ID or filename.",
            "final_depth_m": "Final penetration depth (m).",
            "gwl_m": "Groundwater level (m below surface) or null.",
            "depth_m": "Depth array (m).",
            "q_c_kPa": "Cone tip resistance array (kPa).",
            "f_s_kPa": "Sleeve friction array (kPa).",
            "u_2_kPa": "Pore pressure u2 array (kPa).",
            "Rf_pct": "Friction ratio array (%).",
        },
    },
    "parse_bore": {
        "category": "File Import",
        "brief": "Parse a borehole file (GEF or BRO-XML, via pygef) into layer descriptions.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to borehole file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "gef", "xml"], "description": "Parser engine."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_layers": "Number of soil layers.",
            "alias": "Borehole ID or filename.",
            "final_depth_m": "Total bore depth (m).",
            "gwl_m": "Groundwater level (m below surface) or null.",
            "layers": "List of layer dicts with top_m, bottom_m, soil_name, soil_code.",
        },
    },
    "read_ags4": {
        "category": "File Import",
        "brief": "Read and parse an AGS4 geotechnical data file (via python-ags4) into structured tables.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to AGS4 file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "Raw AGS4 content as string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8", "description": "File encoding."},
            "include_data": {"type": "bool", "required": False, "default": True, "description": "Include all table data in result."},
            "convert_numeric": {"type": "bool", "required": False, "default": True, "description": "Convert numeric columns from text."},
        },
        "returns": {
            "filepath": "Source file path or '<string>'.",
            "n_groups": "Number of AGS4 groups (tables) found.",
            "group_names": "Names of all groups.",
            "group_row_counts": "Row counts per group.",
            "tables": "Dict of group_name to list of row dicts (if include_data=True).",
        },
    },
    "validate_ags4": {
        "category": "File Validation",
        "brief": "Validate an AGS4 file against AGS4 rules (via python-ags4).",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to AGS4 file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "Raw AGS4 content as string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8", "description": "File encoding."},
        },
        "returns": {
            "filepath": "Source file path.",
            "n_errors": "Number of errors found.",
            "n_warnings": "Number of warnings found.",
            "n_fyi": "Number of FYI messages.",
            "is_valid": "True if no errors (warnings/FYI acceptable).",
            "errors": "Error details grouped by rule number.",
        },
    },
    "validate_diggs_schema": {
        "category": "File Validation",
        "brief": "Validate DIGGS XML against XSD schema (v2.6 or v2.5.a, via pydiggs). For DIGGS data EXTRACTION use parse_diggs instead.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
            "schema_version": {"type": "str", "required": False, "default": "2.6", "allowed_values": ["2.6", "2.5.a"], "description": "DIGGS schema version."},
        },
        "returns": {
            "source": "Filename or 'content'.",
            "check_type": "Always 'schema'.",
            "schema_version": "Schema version validated against.",
            "is_valid": "Whether validation passed.",
            "n_errors": "Number of validation errors.",
            "errors": "List of error messages.",
        },
    },
    "validate_diggs_dictionary": {
        "category": "File Validation",
        "brief": "Validate DIGGS propertyClass values against DIGGS dictionary (via pydiggs).",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
        },
        "returns": {
            "source": "Filename or 'content'.",
            "check_type": "Always 'dictionary'.",
            "is_valid": "Whether all propertyClass values are valid.",
            "n_errors": "Number of undefined properties.",
            "errors": "List of undefined property messages.",
        },
    },
}
