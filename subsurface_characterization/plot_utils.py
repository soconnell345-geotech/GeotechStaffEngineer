"""
Shared Plotly utilities for subsurface characterization plots.

Provides USCS color/marker maps, lazy Plotly import, and HTML export.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# USCS color and marker maps
# ---------------------------------------------------------------------------

USCS_COLORS = {
    "GW": "#D2691E",   # brown
    "GP": "#CD853F",   # peru
    "GM": "#C4A882",   # tan
    "GC": "#A0522D",   # sienna
    "SW": "#FFD700",   # gold
    "SP": "#FFA500",   # orange
    "SM": "#F0E68C",   # khaki
    "SC": "#DAA520",   # goldenrod
    "ML": "#90EE90",   # light green
    "CL": "#32CD32",   # lime green
    "OL": "#556B2F",   # dark olive
    "MH": "#00CED1",   # dark turquoise
    "CH": "#008B8B",   # dark cyan
    "OH": "#2F4F4F",   # dark slate gray
    "PT": "#4B0082",   # indigo
    "R":  "#808080",   # gray (rock)
    "":   "#B0B0B0",   # default gray
}

USCS_MARKERS = {
    "GW": "circle",
    "GP": "circle-open",
    "GM": "diamond",
    "GC": "diamond-open",
    "SW": "square",
    "SP": "square-open",
    "SM": "triangle-up",
    "SC": "triangle-up-open",
    "ML": "cross",
    "CL": "x",
    "OL": "star",
    "MH": "hexagon",
    "CH": "pentagon",
    "OH": "star-triangle-up",
    "PT": "star-diamond",
    "R":  "octagon",
    "":   "circle",
}

# Investigation type markers
INVESTIGATION_MARKERS = {
    "boring": "circle",
    "cpt": "triangle-up",
    "test_pit": "square",
    "monitoring_well": "diamond",
}

# Default color sequence for investigations
INVESTIGATION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def get_plotly():
    """Lazy import of plotly.graph_objects."""
    import plotly.graph_objects as go
    return go


def get_subplots():
    """Lazy import of plotly.subplots.make_subplots."""
    from plotly.subplots import make_subplots
    return make_subplots


def figure_to_html(fig, full_html: bool = True) -> str:
    """Convert Plotly figure to self-contained HTML string.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure.
    full_html : bool
        If True, include full HTML document wrapper.

    Returns
    -------
    str
        HTML string.
    """
    return fig.to_html(full_html=full_html, include_plotlyjs=True)


def classify_measurement_by_uscs(
    measurement,
    investigation,
) -> str:
    """Look up USCS classification at measurement depth.

    Parameters
    ----------
    measurement : PointMeasurement
        The measurement to classify.
    investigation : Investigation
        The investigation containing lithology.

    Returns
    -------
    str
        USCS code at measurement depth, or '' if not found.
    """
    return investigation.uscs_at_depth(measurement.depth_m)


def apply_standard_layout(fig, title: str = "", y_label: str = "Depth (m)"):
    """Apply standard engineering layout to a Plotly figure.

    White background, grid lines, professional font.
    """
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig
