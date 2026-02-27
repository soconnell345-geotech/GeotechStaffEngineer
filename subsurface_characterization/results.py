"""
Result dataclasses for subsurface characterization module.

PlotResult : Wraps a Plotly figure with metadata and HTML export
DiggsParseResult : Result of parsing a DIGGS XML file
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class PlotResult:
    """Result of a subsurface visualization.

    Attributes
    ----------
    plot_type : str
        Type of plot (e.g., 'parameter_vs_depth', 'plan_view', 'cross_section').
    title : str
        Plot title.
    n_investigations : int
        Number of investigations shown.
    n_data_points : int
        Number of data points plotted.
    parameters : list of str
        Parameters shown in the plot.
    figure : object
        Plotly go.Figure object.
    trend_results : list
        TrendAnalysisResult objects if trends were computed.
    """

    plot_type: str = ""
    title: str = ""
    n_investigations: int = 0
    n_data_points: int = 0
    parameters: List[str] = field(default_factory=list)
    figure: Any = None
    trend_results: list = field(default_factory=list)

    def to_html(self, full_html: bool = True) -> str:
        """Export figure as self-contained HTML string."""
        if self.figure is None:
            return "<p>No figure available</p>"
        return self.figure.to_html(full_html=full_html, include_plotlyjs=True)

    def summary(self) -> str:
        """Text summary of the plot."""
        lines = [
            "=" * 60,
            f"  PLOT: {self.title}",
            "=" * 60,
            f"  Type: {self.plot_type}",
            f"  Investigations: {self.n_investigations}",
            f"  Data points: {self.n_data_points}",
            f"  Parameters: {', '.join(self.parameters)}",
        ]
        if self.trend_results:
            lines.append(f"  Trends computed: {len(self.trend_results)}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (includes HTML for transport)."""
        d = {
            "plot_type": self.plot_type,
            "title": self.title,
            "n_investigations": self.n_investigations,
            "n_data_points": self.n_data_points,
            "parameters": self.parameters,
            "html": self.to_html(),
        }
        if self.trend_results:
            d["trend_results"] = [
                t.to_dict() if hasattr(t, "to_dict") else str(t)
                for t in self.trend_results
            ]
        return d


@dataclass
class DiggsParseResult:
    """Result of parsing a DIGGS XML file.

    Attributes
    ----------
    site : SiteModel
        Parsed site model.
    n_investigations : int
        Number of investigations extracted.
    n_measurements : int
        Total number of point measurements.
    n_lithology_intervals : int
        Total number of lithology intervals.
    warnings : list of str
        Any warnings generated during parsing.
    """

    site: Any = None
    n_investigations: int = 0
    n_measurements: int = 0
    n_lithology_intervals: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Text summary of parse results."""
        name = self.site.project_name if self.site else "Unknown"
        lines = [
            "=" * 60,
            f"  DIGGS PARSE RESULT: {name}",
            "=" * 60,
            f"  Investigations: {self.n_investigations}",
            f"  Measurements: {self.n_measurements}",
            f"  Lithology intervals: {self.n_lithology_intervals}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "project_name": self.site.project_name if self.site else "",
            "n_investigations": self.n_investigations,
            "n_measurements": self.n_measurements,
            "n_lithology_intervals": self.n_lithology_intervals,
            "warnings": self.warnings,
            "site": self.site.to_dict() if self.site else None,
        }
