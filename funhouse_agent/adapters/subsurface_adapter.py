"""Subsurface characterization adapter — flat dict -> subsurface_characterization API -> dict."""

import numpy as np

from funhouse_agent.adapters import clean_result


def _run_load_site(params: dict) -> dict:
    from subsurface_characterization import load_site_from_dict
    site = load_site_from_dict(params["site_data"])
    return {
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


def _run_plot_parameter_vs_depth(params: dict) -> dict:
    from subsurface_characterization import load_site_from_dict, plot_parameter_vs_depth
    site = load_site_from_dict(params["site_data"])
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
    from subsurface_characterization import load_site_from_dict, plot_atterberg_limits
    site = load_site_from_dict(params["site_data"])
    result = plot_atterberg_limits(
        site=site,
        use_elevation=params.get("use_elevation", False),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


def _run_plot_multi_parameter(params: dict) -> dict:
    from subsurface_characterization import load_site_from_dict, plot_multi_parameter
    site = load_site_from_dict(params["site_data"])
    result = plot_multi_parameter(
        site=site,
        parameters=params["parameters"],
        use_elevation=params.get("use_elevation", False),
        title=params.get("title", ""),
    )
    output_format = params.get("output_format", "metadata")
    return clean_result(result.to_dict(output_format=output_format))


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


METHOD_REGISTRY = {
    "load_site": _run_load_site,
    "plot_parameter_vs_depth": _run_plot_parameter_vs_depth,
    "plot_atterberg_limits": _run_plot_atterberg_limits,
    "plot_multi_parameter": _run_plot_multi_parameter,
    "compute_trend": _run_compute_trend,
}

METHOD_INFO = {
    "load_site": {
        "category": "Subsurface Characterization",
        "brief": "Load site model from nested dict structure and return investigation summary.",
        "parameters": {
            "site_data": {"type": "dict", "required": True, "description": "Nested dict with project_name, investigations [{investigation_id, x, y, elevation_m, measurements, lithology}]."},
        },
        "returns": {
            "project_name": "Site project name.",
            "n_investigations": "Number of investigation locations.",
            "investigations": "Summary of each investigation.",
        },
    },
    "plot_parameter_vs_depth": {
        "category": "Subsurface Characterization",
        "brief": "Scatter plot of a soil parameter vs depth (Plotly interactive).",
        "parameters": {
            "site_data": {"type": "dict", "required": True, "description": "Site data dict (same format as load_site)."},
            "parameter": {"type": "str", "required": True, "description": "Parameter name to plot (e.g., 'N_spt', 'cu_kPa', 'wn_pct')."},
            "color_by": {"type": "str", "required": False, "default": "investigation", "description": "Color coding: investigation/uscs/none."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation instead of depth."},
            "show_trend": {"type": "bool", "required": False, "default": False, "description": "Overlay linear trendline."},
            "show_bands": {"type": "bool", "required": False, "default": False, "description": "Overlay prediction bands."},
            "band_sigma": {"type": "float", "required": False, "default": 1.0, "description": "Number of sigma for prediction bands."},
            "group_trends_by": {"type": "str", "required": False, "default": "", "description": "Group trends by 'uscs' for separate trends per soil class."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "description": "Output: metadata/html/json."},
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
            "site_data": {"type": "dict", "required": True, "description": "Site data dict."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "description": "Output: metadata/html/json."},
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
            "site_data": {"type": "dict", "required": True, "description": "Site data dict."},
            "parameters": {"type": "array", "required": True, "description": "Parameter names for each panel (e.g., ['N_spt','cu_kPa','wn_pct'])."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "description": "Output: metadata/html/json."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Total data points across all panels.",
            "parameters": "Parameters plotted.",
        },
    },
    "compute_trend": {
        "category": "Subsurface Characterization",
        "brief": "Fit linear or log-linear depth-value trend with statistics (R-squared, COV).",
        "parameters": {
            "depths": {"type": "array", "required": True, "description": "Depth values (m)."},
            "values": {"type": "array", "required": True, "description": "Measurement values."},
            "trend_type": {"type": "str", "required": False, "default": "linear", "description": "Trend type: linear or log_linear."},
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
}
