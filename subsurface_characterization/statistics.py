"""
Trend regression and variance analysis for subsurface data.

Reference: Phoon & Kulhawy (1999), Canadian Geotechnical Journal.
Provides linear and log-linear trend fitting with prediction bands.

Functions
---------
compute_trend : Fit depth-value trend line
compute_grouped_trends : Separate trends per soil type
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from subsurface_characterization.site_model import SiteModel


@dataclass
class TrendAnalysisResult:
    """Result of trend regression analysis.

    Attributes
    ----------
    parameter : str
        Parameter name (e.g., 'N_spt', 'cu_kPa').
    n_points : int
        Number of data points used.
    trend_type : str
        'linear' or 'log_linear'.
    slope : float
        Slope of regression line.
    intercept : float
        Intercept of regression line.
    r_squared : float
        Coefficient of determination.
    std_residual : float
        Standard deviation of residuals.
    cov : float
        Coefficient of variation (std_residual / mean_value).
    group_label : str
        Grouping label (e.g., USCS class) if grouped.
    """

    parameter: str = ""
    n_points: int = 0
    trend_type: str = "linear"
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    std_residual: float = 0.0
    cov: float = 0.0
    group_label: str = ""

    def predict(self, depth: float) -> float:
        """Predict value at given depth using trend.

        Parameters
        ----------
        depth : float
            Depth (m) at which to predict.

        Returns
        -------
        float
            Predicted value.
        """
        if self.trend_type == "log_linear":
            if depth <= 0:
                depth = 0.01
            return np.exp(self.slope * np.log(depth) + self.intercept)
        return self.slope * depth + self.intercept

    def band(self, depth: float, n_sigma: float = 1.0) -> tuple:
        """Prediction band at given depth.

        Parameters
        ----------
        depth : float
            Depth (m).
        n_sigma : float
            Number of standard deviations for band width.

        Returns
        -------
        tuple
            (lower, upper) prediction band values.
        """
        predicted = self.predict(depth)
        half_width = n_sigma * self.std_residual
        return (predicted - half_width, predicted + half_width)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "parameter": self.parameter,
            "n_points": self.n_points,
            "trend_type": self.trend_type,
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 6),
            "r_squared": round(self.r_squared, 4),
            "std_residual": round(self.std_residual, 4),
            "cov": round(self.cov, 4),
            "group_label": self.group_label,
        }

    def summary(self) -> str:
        """Text summary."""
        label = f" ({self.group_label})" if self.group_label else ""
        return (
            f"{self.parameter}{label}: {self.trend_type}, n={self.n_points}, "
            f"R²={self.r_squared:.3f}, COV={self.cov:.3f}"
        )


def compute_trend(
    depths: np.ndarray,
    values: np.ndarray,
    trend_type: str = "linear",
    parameter: str = "",
    group_label: str = "",
) -> TrendAnalysisResult:
    """Fit depth-value trend line.

    Parameters
    ----------
    depths : array-like
        Depth values (m).
    values : array-like
        Measurement values.
    trend_type : str
        'linear' or 'log_linear'.
    parameter : str
        Parameter name for labeling.
    group_label : str
        Group label (e.g., USCS class).

    Returns
    -------
    TrendAnalysisResult
    """
    depths = np.asarray(depths, dtype=float)
    values = np.asarray(values, dtype=float)

    n = len(depths)
    if n < 2:
        return TrendAnalysisResult(
            parameter=parameter,
            n_points=n,
            trend_type=trend_type,
            group_label=group_label,
        )

    if trend_type == "log_linear":
        # Log-linear: ln(value) = slope * ln(depth) + intercept
        mask = (depths > 0) & (values > 0)
        if mask.sum() < 2:
            return TrendAnalysisResult(
                parameter=parameter, n_points=n,
                trend_type=trend_type, group_label=group_label,
            )
        log_d = np.log(depths[mask])
        log_v = np.log(values[mask])
        coeffs = np.polyfit(log_d, log_v, 1)
        slope, intercept = coeffs[0], coeffs[1]
        predicted_log = np.polyval(coeffs, log_d)
        residuals = log_v - predicted_log
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        std_residual = float(np.std(values[mask] - np.exp(predicted_log), ddof=1)) if mask.sum() > 2 else 0.0
        mean_val = float(np.mean(values[mask]))
    else:
        # Linear: value = slope * depth + intercept
        coeffs = np.polyfit(depths, values, 1)
        slope, intercept = coeffs[0], coeffs[1]
        predicted = np.polyval(coeffs, depths)
        residuals = values - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        std_residual = float(np.std(residuals, ddof=1)) if n > 2 else 0.0
        mean_val = float(np.mean(values))

    cov = std_residual / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

    return TrendAnalysisResult(
        parameter=parameter,
        n_points=n,
        trend_type=trend_type,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        std_residual=std_residual,
        cov=cov,
        group_label=group_label,
    )


def compute_grouped_trends(
    site: SiteModel,
    parameter: str,
    group_by: str = "uscs",
    trend_type: str = "linear",
) -> Dict[str, TrendAnalysisResult]:
    """Compute separate trends per soil type or investigation.

    Parameters
    ----------
    site : SiteModel
        Site model with investigations.
    parameter : str
        Parameter name to analyze.
    group_by : str
        'uscs' to group by USCS classification, 'investigation' by investigation_id.
    trend_type : str
        'linear' or 'log_linear'.

    Returns
    -------
    dict
        {group_label: TrendAnalysisResult}
    """
    groups: Dict[str, tuple] = {}  # label -> ([depths], [values])

    for inv in site.investigations:
        for m in inv.get_measurements(parameter):
            if group_by == "uscs":
                label = inv.uscs_at_depth(m.depth_m) or "Unknown"
            elif group_by == "investigation":
                label = inv.investigation_id
            else:
                label = "All"

            if label not in groups:
                groups[label] = ([], [])
            groups[label][0].append(m.depth_m)
            groups[label][1].append(m.value)

    results = {}
    for label, (depths, values) in groups.items():
        results[label] = compute_trend(
            np.array(depths), np.array(values),
            trend_type=trend_type,
            parameter=parameter,
            group_label=label,
        )

    return results
