"""
Subsurface Characterization Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions:
  1. subsurface_char_agent           - Run analysis / generate plot
  2. subsurface_char_list_methods    - Browse available methods
  3. subsurface_char_describe_method - Get detailed parameter docs

All plot methods return JSON with "html" key containing self-contained HTML.
"""

import json

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn


# ---------------------------------------------------------------------------
# Method wrappers
# ---------------------------------------------------------------------------

def _run_parse_diggs(params):
    from subsurface_characterization import parse_diggs
    content = params.get("content")
    filepath = params.get("filepath")
    result = parse_diggs(filepath=filepath, content=content)
    return result.to_dict()


def _run_load_site(params):
    from subsurface_characterization import load_site_from_dict
    data = params.get("data", {})
    site = load_site_from_dict(data)
    return site.to_dict()


def _run_plot_parameter_vs_depth(params):
    from subsurface_characterization import load_site_from_dict, plot_parameter_vs_depth
    data = params.pop("site_data", {})
    site = load_site_from_dict(data)
    result = plot_parameter_vs_depth(site, **params)
    return result.to_dict()


def _run_plot_atterberg_limits(params):
    from subsurface_characterization import load_site_from_dict, plot_atterberg_limits
    data = params.pop("site_data", {})
    site = load_site_from_dict(data)
    result = plot_atterberg_limits(site, **params)
    return result.to_dict()


def _run_plot_multi_parameter(params):
    from subsurface_characterization import load_site_from_dict, plot_multi_parameter
    data = params.pop("site_data", {})
    site = load_site_from_dict(data)
    result = plot_multi_parameter(site, **params)
    return result.to_dict()


def _run_plot_plan_view(params):
    from subsurface_characterization import load_site_from_dict, plot_plan_view
    data = params.pop("site_data", {})
    site = load_site_from_dict(data)
    result = plot_plan_view(site, **params)
    return result.to_dict()


def _run_plot_cross_section(params):
    from subsurface_characterization import load_site_from_dict, plot_cross_section
    data = params.pop("site_data", {})
    site = load_site_from_dict(data)
    result = plot_cross_section(site, **params)
    return result.to_dict()


def _run_compute_trend(params):
    from subsurface_characterization import load_site_from_dict, compute_grouped_trends
    import numpy as np
    data = params.pop("site_data", None)

    if data:
        site = load_site_from_dict(data)
        parameter = params.get("parameter", "N_spt")
        group_by = params.get("group_by", "uscs")
        trend_type = params.get("trend_type", "linear")
        results = compute_grouped_trends(site, parameter, group_by=group_by, trend_type=trend_type)
        return {label: t.to_dict() for label, t in results.items()}
    else:
        from subsurface_characterization import compute_trend
        depths = np.array(params.get("depths", []))
        values = np.array(params.get("values", []))
        trend_type = params.get("trend_type", "linear")
        parameter = params.get("parameter", "")
        result = compute_trend(depths, values, trend_type=trend_type, parameter=parameter)
        return result.to_dict()


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
}

METHOD_INFO = {
    "parse_diggs": {
        "category": "Data Loading",
        "brief": "Parse DIGGS 2.6/2.5.a XML into SiteModel.",
        "description": (
            "Parse a DIGGS XML file or string into a structured SiteModel. "
            "Extracts borings, lithology, SPT, Atterberg limits, moisture, GWL. "
            "Auto-detects DIGGS 2.6 vs 2.5.a namespace."
        ),
        "reference": "DIGGS 2.6 Schema (diggsml.org)",
        "parameters": {
            "filepath": {
                "type": "str",
                "required": False,
                "description": "Path to DIGGS XML file.",
            },
            "content": {
                "type": "str",
                "required": False,
                "description": "DIGGS XML string content. Provide either filepath or content.",
            },
        },
        "returns": {
            "project_name": "Project name from DIGGS",
            "n_investigations": "Number of borings/investigations extracted",
            "n_measurements": "Total point measurements",
            "n_lithology_intervals": "Total lithology intervals",
            "warnings": "List of parse warnings",
            "site": "Full SiteModel dict",
        },
        "related": {
            "subsurface_char.load_site": "Alternative: load from dict/CSV",
            "pydiggs.validate_diggs_file": "Validate DIGGS schema compliance",
        },
        "typical_workflow": (
            "1. parse_diggs(content=xml_string)\n"
            "2. Use returned site dict for plot methods\n"
            "3. plot_parameter_vs_depth(site_data=site, parameter='N_spt')"
        ),
        "common_mistakes": [
            "Providing neither filepath nor content",
            "Using pydiggs for extraction (it only validates, cannot extract data)",
        ],
    },
    "load_site": {
        "category": "Data Loading",
        "brief": "Create SiteModel from nested dict structure.",
        "description": (
            "Build a SiteModel from a dict with project_name and investigations array. "
            "Each investigation has id, type, coordinates, measurements, and lithology."
        ),
        "reference": "subsurface_characterization module docs",
        "parameters": {
            "data": {
                "type": "dict",
                "required": True,
                "description": (
                    "Nested dict with keys: project_name, investigations[]. "
                    "Each investigation: investigation_id, investigation_type, x, y, "
                    "elevation_m, total_depth_m, gwl_depth_m, measurements[], lithology[]."
                ),
            },
        },
        "returns": {
            "project_name": "Project name",
            "n_investigations": "Number of investigations",
            "investigations": "List of investigation dicts with all data",
        },
        "related": {
            "subsurface_char.parse_diggs": "Alternative: load from DIGGS XML",
        },
        "typical_workflow": (
            "1. Build dict with boring/CPT data\n"
            "2. load_site(data=my_dict)\n"
            "3. Use returned site dict for plot methods"
        ),
        "common_mistakes": [
            "Forgetting to nest measurements inside investigations",
            "Using wrong parameter names (use N_spt, not N)",
        ],
    },
    "plot_parameter_vs_depth": {
        "category": "Visualization",
        "brief": "XY scatter of parameter vs depth/elevation.",
        "description": (
            "Create an interactive scatter plot of any subsurface parameter vs depth. "
            "Color by investigation, USCS class, or none. Optional trendline with "
            "prediction bands (Phoon & Kulhawy 1999). Returns HTML for rendering."
        ),
        "reference": "Phoon & Kulhawy (1999), CGJ 36(4)",
        "parameters": {
            "site_data": {
                "type": "dict",
                "required": True,
                "description": "SiteModel dict (from parse_diggs or load_site).",
            },
            "parameter": {
                "type": "str",
                "required": True,
                "description": "Parameter name (e.g., 'N_spt', 'cu_kPa', 'qc_kPa').",
            },
            "color_by": {
                "type": "str",
                "required": False,
                "default": "investigation",
                "choices": ["investigation", "uscs", "none"],
                "description": "Color points by investigation ID, USCS class, or single color.",
            },
            "use_elevation": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "If true, Y-axis is elevation; if false, depth (inverted).",
            },
            "show_trend": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "Overlay best-fit trendline.",
            },
            "show_bands": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "Overlay ±σ prediction bands.",
            },
            "band_sigma": {
                "type": "float",
                "required": False,
                "default": 1.0,
                "description": "Number of σ for bands.",
            },
            "group_trends_by": {
                "type": "str",
                "required": False,
                "default": "",
                "choices": ["", "uscs"],
                "description": "If 'uscs', compute separate trends per soil type.",
            },
        },
        "returns": {
            "plot_type": "'parameter_vs_depth'",
            "title": "Plot title",
            "n_investigations": "Number of investigations shown",
            "n_data_points": "Number of data points plotted",
            "parameters": "Parameters shown",
            "html": "Self-contained HTML string for rendering",
            "trend_results": "Trend analysis results (if computed)",
        },
        "related": {
            "subsurface_char.plot_multi_parameter": "Side-by-side panels",
            "subsurface_char.compute_trend": "Standalone trend analysis",
        },
        "typical_workflow": (
            "1. parse_diggs or load_site to get site_data\n"
            "2. plot_parameter_vs_depth(site_data=site, parameter='N_spt', show_trend=true)\n"
            "3. Render HTML in browser/notebook"
        ),
        "common_mistakes": [
            "Forgetting site_data parameter",
            "Using non-standard parameter names",
        ],
    },
    "plot_atterberg_limits": {
        "category": "Visualization",
        "brief": "LL/PL bracket plot with natural moisture overlay.",
        "description": (
            "Plot Atterberg limits as horizontal brackets (PL to LL) at each depth, "
            "with natural moisture content shown as diamond markers. Colored by investigation."
        ),
        "reference": "ASTM D4318",
        "parameters": {
            "site_data": {
                "type": "dict",
                "required": True,
                "description": "SiteModel dict.",
            },
            "use_elevation": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "If true, Y-axis is elevation.",
            },
        },
        "returns": {
            "plot_type": "'atterberg_limits'",
            "html": "Self-contained HTML string",
            "n_data_points": "Number of bracket pairs plotted",
        },
        "related": {
            "subsurface_char.plot_parameter_vs_depth": "Single parameter scatter",
            "geolysis.classify_uscs": "USCS classification from Atterberg + gradation",
        },
        "typical_workflow": (
            "1. Load site with LL_pct, PL_pct, wn_pct measurements\n"
            "2. plot_atterberg_limits(site_data=site)\n"
            "3. Check if wn falls between PL and LL"
        ),
        "common_mistakes": [
            "No LL_pct or PL_pct data at matching depths",
        ],
    },
    "plot_multi_parameter": {
        "category": "Visualization",
        "brief": "Side-by-side subplots with shared Y-axis.",
        "description": (
            "Create multi-panel plot with one subplot per parameter, all sharing "
            "the same depth/elevation axis. Useful for comparing N_spt, cu, wn side by side."
        ),
        "reference": "",
        "parameters": {
            "site_data": {
                "type": "dict",
                "required": True,
                "description": "SiteModel dict.",
            },
            "parameters": {
                "type": "list[str]",
                "required": True,
                "description": "List of parameter names for each panel.",
            },
            "use_elevation": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "If true, Y-axis is elevation.",
            },
        },
        "returns": {
            "plot_type": "'multi_parameter'",
            "html": "Self-contained HTML string",
            "n_data_points": "Total data points across all panels",
        },
        "related": {
            "subsurface_char.plot_parameter_vs_depth": "Single parameter version",
        },
        "typical_workflow": (
            "1. plot_multi_parameter(site_data=site, parameters=['N_spt', 'cu_kPa', 'wn_pct'])\n"
            "2. Compare profiles side by side"
        ),
        "common_mistakes": [
            "Empty parameters list",
        ],
    },
    "plot_plan_view": {
        "category": "Visualization",
        "brief": "Plan view map of investigation locations.",
        "description": (
            "XY scatter of investigation locations with marker shapes by type "
            "(boring=circle, CPT=triangle, test pit=square). Color by type or "
            "by average parameter value. Labels for ID, GWL, depth to rock, etc."
        ),
        "reference": "",
        "parameters": {
            "site_data": {
                "type": "dict",
                "required": True,
                "description": "SiteModel dict.",
            },
            "color_by": {
                "type": "str",
                "required": False,
                "default": "type",
                "choices": ["type", "parameter"],
                "description": "Color by investigation type or parameter value.",
            },
            "label_field": {
                "type": "str",
                "required": False,
                "default": "id",
                "choices": ["id", "depth_to_rock", "gwl", "fill_thickness"],
                "description": "What to label each point with.",
            },
            "parameter_for_color": {
                "type": "str",
                "required": False,
                "default": "",
                "description": "If color_by='parameter', which parameter to use.",
            },
        },
        "returns": {
            "plot_type": "'plan_view'",
            "html": "Self-contained HTML string",
            "n_investigations": "Number of investigation points",
        },
        "related": {
            "subsurface_char.plot_cross_section": "Vertical profile between selected points",
        },
        "typical_workflow": (
            "1. plot_plan_view(site_data=site, color_by='type', label_field='id')\n"
            "2. Identify boring locations\n"
            "3. Select borings for cross-section"
        ),
        "common_mistakes": [
            "No coordinates in data (all at 0,0)",
        ],
    },
    "plot_cross_section": {
        "category": "Visualization",
        "brief": "Cross-section profile with lithology columns.",
        "description": (
            "Vertical lithology columns at selected investigation locations along a section line. "
            "X-axis is cumulative distance, Y-axis is elevation. USCS-colored layers, "
            "ground surface line, optional GWL dashed line and parameter annotations. "
            "Phase 1: NO interpreted connections between borings (raw data only)."
        ),
        "reference": "",
        "parameters": {
            "site_data": {
                "type": "dict",
                "required": True,
                "description": "SiteModel dict.",
            },
            "investigation_ids": {
                "type": "list[str]",
                "required": True,
                "description": "Ordered list of investigation IDs defining the section line.",
            },
            "use_elevation": {
                "type": "bool",
                "required": False,
                "default": True,
                "description": "If true, Y-axis is elevation; if false, depth (inverted).",
            },
            "annotate_parameter": {
                "type": "str",
                "required": False,
                "default": "",
                "description": "Show parameter values next to columns (e.g., 'N_spt').",
            },
            "show_gwl": {
                "type": "bool",
                "required": False,
                "default": True,
                "description": "Show groundwater level dashed line.",
            },
        },
        "returns": {
            "plot_type": "'cross_section'",
            "html": "Self-contained HTML string",
            "n_investigations": "Number of investigations in section",
            "n_data_points": "Number of lithology intervals shown",
        },
        "related": {
            "subsurface_char.plot_plan_view": "Select boring locations first",
            "subsurface_char.plot_parameter_vs_depth": "Detailed single-parameter view",
        },
        "typical_workflow": (
            "1. plot_plan_view to identify boring layout\n"
            "2. plot_cross_section(site_data=site, investigation_ids=['B-1','B-2','B-3'])\n"
            "3. Annotate with SPT: annotate_parameter='N_spt'"
        ),
        "common_mistakes": [
            "Less than 2 investigation IDs (need at least 2 for a section)",
            "IDs not in site data (check investigation_ids match)",
        ],
    },
    "compute_trend": {
        "category": "Statistics",
        "brief": "Linear/log-linear trend regression with COV.",
        "description": (
            "Fit depth-value trend line using least squares. Computes slope, intercept, "
            "R², standard deviation of residuals, and coefficient of variation (COV). "
            "Can group by USCS class or investigation. Based on Phoon & Kulhawy (1999)."
        ),
        "reference": "Phoon & Kulhawy (1999), CGJ 36(4), 612-624",
        "parameters": {
            "site_data": {
                "type": "dict",
                "required": False,
                "description": "SiteModel dict (for grouped trends). Omit for raw arrays.",
            },
            "parameter": {
                "type": "str",
                "required": False,
                "default": "N_spt",
                "description": "Parameter to analyze (when using site_data).",
            },
            "group_by": {
                "type": "str",
                "required": False,
                "default": "uscs",
                "choices": ["uscs", "investigation"],
                "description": "How to group data for separate trends.",
            },
            "trend_type": {
                "type": "str",
                "required": False,
                "default": "linear",
                "choices": ["linear", "log_linear"],
                "description": "'linear' or 'log_linear'.",
            },
            "depths": {
                "type": "list[float]",
                "required": False,
                "description": "Raw depth array (when not using site_data).",
            },
            "values": {
                "type": "list[float]",
                "required": False,
                "description": "Raw value array (when not using site_data).",
            },
        },
        "returns": {
            "parameter": "Parameter name",
            "n_points": "Number of data points",
            "trend_type": "linear or log_linear",
            "slope": "Regression slope",
            "intercept": "Regression intercept",
            "r_squared": "Coefficient of determination",
            "std_residual": "Standard deviation of residuals",
            "cov": "Coefficient of variation",
        },
        "related": {
            "subsurface_char.plot_parameter_vs_depth": "Plot with trend overlay",
        },
        "typical_workflow": (
            "1. compute_trend(site_data=site, parameter='cu_kPa', group_by='uscs')\n"
            "2. Check R² and COV per soil type\n"
            "3. Compare COV to Phoon & Kulhawy typical ranges"
        ),
        "common_mistakes": [
            "Less than 2 data points (need at least 2 for regression)",
            "log_linear with zero or negative values",
        ],
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def subsurface_char_agent(method: str, parameters_json: str) -> str:
    """Subsurface characterization agent.

    Parse DIGGS XML, load site data, and generate interactive Plotly visualizations.
    Plot methods return JSON with "html" key for rendering.

    Call subsurface_char_list_methods() first to see available methods,
    then subsurface_char_describe_method() for parameter details.

    Parameters:
        method: Analysis method name.
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with results or error message.
    """
    try:
        params = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def subsurface_char_list_methods(category: str = "") -> str:
    """List available subsurface characterization methods.

    Parameters:
        category: Optional filter ('Data Loading', 'Visualization', 'Statistics').

    Returns:
        JSON string with categorized method list.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]

    if not result:
        cats = sorted(set(i["category"] for i in METHOD_INFO.values()))
        return json.dumps({
            "error": f"No methods found for category '{category}'. Available: {', '.join(cats)}"
        })
    return json.dumps(result)


@function
def subsurface_char_describe_method(method: str) -> str:
    """Get detailed documentation for a subsurface characterization method.

    Parameters:
        method: Method name to describe.

    Returns:
        JSON string with full method documentation.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
