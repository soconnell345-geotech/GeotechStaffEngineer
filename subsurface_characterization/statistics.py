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
        Number of data points the fit USED (the points given, less
        ``n_dropped``).
    n_dropped : int
        Points given and left out of the fit: a point whose depth or value
        is missing (NaN), and, in a log-linear fit, a point at depth <= 0 or
        with a value <= 0 (ln(depth) and ln(value) exist only for positive
        numbers).
    dropped_reason : str
        Why the dropped points were dropped ("" when none were).
    fitted : bool
        False only on a group that ``compute_grouped_trends`` could not fit
        (fewer than two usable points); its slope, intercept, R-squared,
        residual and COV are then None in ``to_dict`` and
        ``not_fitted_reason`` says why. ``compute_trend`` itself refuses
        such data.
    not_fitted_reason : str
        Why a group was not fitted ("" when it was).
    scatter_defined : bool
        False when the fit used exactly two points: a line through two
        points fits both exactly, so R-squared is 1 by construction and the
        residual scatter and COV are not defined (None in ``to_dict``).
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
    n_dropped: int = 0
    dropped_reason: str = ""
    fitted: bool = True
    not_fitted_reason: str = ""
    scatter_defined: bool = True

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
        if not self.fitted:
            raise ValueError(
                f"No trend was fitted ({self.not_fitted_reason}), so there "
                f"is nothing to predict from.")
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
        """Convert to JSON-serializable dict.

        A number that was not computed is None, never 0: every fit value on
        a group that was not fitted, and the residual scatter and COV of a
        line through exactly two points.
        """
        scatter = self.fitted and self.scatter_defined
        d = {
            "parameter": self.parameter,
            "n_points": self.n_points,
            "n_dropped": self.n_dropped,
            "trend_type": self.trend_type,
            "fitted": self.fitted,
            "slope": round(self.slope, 6) if self.fitted else None,
            "intercept": round(self.intercept, 6) if self.fitted else None,
            "r_squared": round(self.r_squared, 4) if self.fitted else None,
            "std_residual": round(self.std_residual, 4) if scatter else None,
            "cov": round(self.cov, 4) if scatter else None,
            "group_label": self.group_label,
        }
        notes = []
        if self.n_dropped:
            notes.append(
                f"{self.n_dropped} of {self.n_points + self.n_dropped} "
                f"points left out of the fit: {self.dropped_reason}")
        if not self.fitted:
            notes.append(f"Not fitted: {self.not_fitted_reason}")
        elif not self.scatter_defined:
            notes.append(
                "The fit used exactly 2 points: the line passes through "
                "both, so R-squared is 1 by construction and the residual "
                "scatter and COV are not defined (null).")
        if notes:
            d["notes"] = notes
        return d

    def summary(self) -> str:
        """Text summary."""
        label = f" ({self.group_label})" if self.group_label else ""
        if not self.fitted:
            return (f"{self.parameter}{label}: not fitted "
                    f"({self.not_fitted_reason})")
        dropped = (f", {self.n_dropped} dropped" if self.n_dropped else "")
        cov = (f"COV={self.cov:.3f}" if self.scatter_defined
               else "COV not defined (2 points)")
        return (
            f"{self.parameter}{label}: {self.trend_type}, n={self.n_points}"
            f"{dropped}, R²={self.r_squared:.3f}, {cov}"
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
        ``n_points`` is the number of points the fit used and ``n_dropped``
        the number left out (a missing depth or value; log-linear also
        points at depth <= 0 or with a value <= 0).

    Raises
    ------
    ValueError
        With fewer than 2 points (log-linear: fewer than 2 points with a
        positive depth and a positive value) there is no line to fit, and
        none is returned.
    """
    depths = np.asarray(depths, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()

    n = len(depths)
    if len(values) != n:
        raise ValueError(
            f"depths has {n} entries and values has {len(values)}; a trend "
            f"needs one value per depth.")
    if n < 2:
        raise ValueError(
            f"A trend line needs at least 2 points; {n} given"
            f"{_for_label(parameter, group_label)}. No trend is fitted.")

    # A point whose depth or value is missing (NaN: a null reading, as a
    # parsed CPT file gives a void) cannot enter any fit; one NaN made
    # every fitted number NaN.
    reasons = []
    finite = np.isfinite(depths) & np.isfinite(values)
    n_missing = n - int(finite.sum())
    if n_missing:
        reasons.append(f"{n_missing} with a missing (null or non-finite) "
                       f"depth or value")
        depths, values = depths[finite], values[finite]
    n_finite = len(depths)
    if n_finite < 2:
        raise ValueError(
            f"A trend line needs at least 2 points; {n_finite} of the {n} "
            f"given have both a depth and a value"
            f"{_for_label(parameter, group_label)}. No trend is fitted.")

    if trend_type == "log_linear":
        # Log-linear: ln(value) = slope * ln(depth) + intercept
        mask = (depths > 0) & (values > 0)
        n_used = int(mask.sum())
        if n_finite - n_used:
            reasons.append(
                f"{n_finite - n_used} at depth <= 0 or with a value <= 0 (a "
                f"log-linear fit takes ln(depth) and ln(value), which exist "
                f"only for positive numbers)")
        if n_used < 2:
            raise ValueError(
                f"A log-linear trend needs at least 2 points with depth > 0 "
                f"and value > 0; {n_used} of the {n} given qualify"
                f"{_for_label(parameter, group_label)}. No trend is fitted.")
        log_d = np.log(depths[mask])
        log_v = np.log(values[mask])
        coeffs = np.polyfit(log_d, log_v, 1)
        slope, intercept = coeffs[0], coeffs[1]
        predicted_log = np.polyval(coeffs, log_d)
        residuals = log_v - predicted_log
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        std_residual = float(np.std(values[mask] - np.exp(predicted_log), ddof=1)) if n_used > 2 else 0.0
        mean_val = float(np.mean(values[mask]))
    else:
        # Linear: value = slope * depth + intercept
        n_used = n_finite
        coeffs = np.polyfit(depths, values, 1)
        slope, intercept = coeffs[0], coeffs[1]
        predicted = np.polyval(coeffs, depths)
        residuals = values - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        std_residual = float(np.std(residuals, ddof=1)) if n_used > 2 else 0.0
        mean_val = float(np.mean(values))

    cov = std_residual / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

    return TrendAnalysisResult(
        parameter=parameter,
        n_points=n_used,
        trend_type=trend_type,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        std_residual=std_residual,
        cov=cov,
        group_label=group_label,
        n_dropped=n - n_used,
        dropped_reason="; ".join(reasons),
        scatter_defined=n_used > 2,
    )


def _for_label(parameter: str, group_label: str) -> str:
    """' for <parameter> (<group>)', or '' when neither is named."""
    if not parameter and not group_label:
        return ""
    group = f" ({group_label})" if group_label else ""
    return f" for {parameter or 'the data'}{group}"


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
        {group_label: TrendAnalysisResult}. A group with too few usable
        points for a line is kept in the dict with ``fitted=False`` and the
        reason in ``not_fitted_reason``; its fit values are None in
        ``to_dict``.
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
        try:
            results[label] = compute_trend(
                np.array(depths), np.array(values),
                trend_type=trend_type,
                parameter=parameter,
                group_label=label,
            )
        except ValueError as exc:
            results[label] = TrendAnalysisResult(
                parameter=parameter,
                n_points=len(depths),
                trend_type=trend_type,
                group_label=label,
                fitted=False,
                not_fitted_reason=str(exc),
                scatter_defined=False,
            )

    return results
