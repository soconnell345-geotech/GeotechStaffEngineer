"""
subsurface_characterization — Subsurface investigation data visualization.

The single data-I/O home for subsurface investigation data: native DIGGS parsing,
interactive Plotly visualizations, and trend statistics, PLUS optional,
dependency-backed format adapters (``formats`` subpackage) for GEF/BRO-XML CPT &
borehole files (pygef), AGS4 (python-ags4), and DIGGS schema/dictionary validation
(pydiggs).

Interactive Plotly visualizations for geotechnical subsurface data.
Supports DIGGS XML, CSV, dict, and CPTParseResult input formats.
All outputs are self-contained HTML (``fig.to_html()``).

Public API
----------
Data Model:
    SiteModel : Top-level site container
    Investigation : Single investigation location
    PointMeasurement : Discrete measurement at depth
    LithologyInterval : Soil/rock layer interval

Loaders:
    load_site_from_dict : Create SiteModel from nested dict
    load_site_from_csv : Create SiteModel from CSV files
    load_cpt_to_investigation : Bridge a CPTParseResult (from the GEF format adapter)
    parse_diggs : Parse DIGGS 2.6 XML → SiteModel (native, no external dependency)

Format adapters (``formats`` subpackage — optional, dependency-backed):
    GEF / BRO-XML (pygef):
        parse_cpt_file, parse_bore_file, has_pygef,
        CPTParseResult, BoreParseResult
    AGS4 (python-ags4):
        read_ags4, validate_ags4, has_ags4,
        AGS4ReadResult, AGS4ValidationResult
    DIGGS validation (pydiggs):
        validate_diggs_schema, validate_diggs_dictionary, has_pydiggs,
        DiggValidationResult

Plots (all return PlotResult with Plotly figure):
    plot_parameter_vs_depth : XY scatter of parameter vs depth
    plot_atterberg_limits : LL/PL bracket plot with wn overlay
    plot_multi_parameter : Side-by-side subplots
    plot_plan_view : Plan view map of locations
    plot_cross_section : Cross-section profile view

Statistics:
    compute_trend : Linear/log-linear trend regression
    compute_grouped_trends : Separate trends per soil type
    TrendAnalysisResult : Trend analysis result dataclass

Results:
    PlotResult : Visualization result with HTML export
    DiggsParseResult : DIGGS parse result with metadata
"""

from subsurface_characterization.site_model import (
    SiteModel,
    Investigation,
    PointMeasurement,
    LithologyInterval,
    STANDARD_PARAMETERS,
)

from subsurface_characterization.csv_loader import (
    load_site_from_dict,
    load_site_from_csv,
    load_cpt_to_investigation,
)

from subsurface_characterization.diggs_parser import parse_diggs

from subsurface_characterization.plots_xy import (
    plot_parameter_vs_depth,
    plot_atterberg_limits,
    plot_multi_parameter,
)

from subsurface_characterization.plots_plan import plot_plan_view

from subsurface_characterization.plots_profile import plot_cross_section

from subsurface_characterization.statistics import (
    compute_trend,
    compute_grouped_trends,
    TrendAnalysisResult,
)

from subsurface_characterization.results import PlotResult, DiggsParseResult

# Format adapters — optional, dependency-backed (pygef / python-ags4 / pydiggs).
# Importing the names is cheap (the heavy third-party libs are lazy-imported only
# when a parse/validate function is actually called).
from subsurface_characterization.formats import (
    # GEF / BRO-XML (pygef)
    parse_cpt_file,
    parse_bore_file,
    has_pygef,
    CPTParseResult,
    BoreParseResult,
    # AGS4 (python-ags4)
    read_ags4,
    validate_ags4,
    has_ags4,
    AGS4ReadResult,
    AGS4ValidationResult,
    # DIGGS validation (pydiggs)
    validate_diggs_schema,
    validate_diggs_dictionary,
    has_pydiggs,
    DiggValidationResult,
)

__all__ = [
    # Data model
    "SiteModel",
    "Investigation",
    "PointMeasurement",
    "LithologyInterval",
    "STANDARD_PARAMETERS",
    # Loaders
    "load_site_from_dict",
    "load_site_from_csv",
    "load_cpt_to_investigation",
    "parse_diggs",
    # Plots
    "plot_parameter_vs_depth",
    "plot_atterberg_limits",
    "plot_multi_parameter",
    "plot_plan_view",
    "plot_cross_section",
    # Statistics
    "compute_trend",
    "compute_grouped_trends",
    "TrendAnalysisResult",
    # Results
    "PlotResult",
    "DiggsParseResult",
    # Format adapters — GEF / BRO-XML (pygef)
    "parse_cpt_file",
    "parse_bore_file",
    "has_pygef",
    "CPTParseResult",
    "BoreParseResult",
    # Format adapters — AGS4 (python-ags4)
    "read_ags4",
    "validate_ags4",
    "has_ags4",
    "AGS4ReadResult",
    "AGS4ValidationResult",
    # Format adapters — DIGGS validation (pydiggs)
    "validate_diggs_schema",
    "validate_diggs_dictionary",
    "has_pydiggs",
    "DiggValidationResult",
]
