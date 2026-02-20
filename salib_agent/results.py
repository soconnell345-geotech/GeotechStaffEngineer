"""
Result dataclasses for SALib agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SobolResult:
    """Results from Sobol sensitivity analysis.

    Attributes
    ----------
    n_samples : int
        Number of model evaluations.
    n_vars : int
        Number of input variables.
    var_names : list of str
        Variable names.
    S1 : list of float
        First-order Sobol indices (main effect of each variable).
    S1_conf : list of float
        95% confidence intervals for S1.
    ST : list of float
        Total-order Sobol indices (including interactions).
    ST_conf : list of float
        95% confidence intervals for ST.
    S2 : list of list of float or None
        Second-order interaction indices (if calc_second_order=True).
    """
    n_samples: int = 0
    n_vars: int = 0
    var_names: List[str] = field(default_factory=list)
    S1: List[float] = field(default_factory=list)
    S1_conf: List[float] = field(default_factory=list)
    ST: List[float] = field(default_factory=list)
    ST_conf: List[float] = field(default_factory=list)
    S2: Optional[List] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SOBOL SENSITIVITY ANALYSIS RESULTS",
            "=" * 60,
            f"  Samples:    {self.n_samples}",
            f"  Variables:  {self.n_vars}",
            "",
            "  Variable        S1          ST",
            "  " + "-" * 44,
        ]
        for i, name in enumerate(self.var_names):
            s1 = self.S1[i] if i < len(self.S1) else float('nan')
            st = self.ST[i] if i < len(self.ST) else float('nan')
            lines.append(f"  {name:16s}  {s1:8.4f}    {st:8.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "n_samples": self.n_samples,
            "n_vars": self.n_vars,
            "var_names": list(self.var_names),
            "S1": [float(x) for x in self.S1],
            "S1_conf": [float(x) for x in self.S1_conf],
            "ST": [float(x) for x in self.ST],
            "ST_conf": [float(x) for x in self.ST_conf],
        }
        if self.S2 is not None:
            # S2 is a matrix — convert to nested lists
            if isinstance(self.S2, np.ndarray):
                d["S2"] = self.S2.tolist()
            else:
                d["S2"] = self.S2
        return d

    def plot_sensitivity(self, ax=None, show=True, **kwargs):
        """Bar chart comparing S1 and ST for each variable."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        setup_engineering_plot(ax, "Sobol Sensitivity Indices", "Variable", "Index")

        x = np.arange(self.n_vars)
        width = 0.35
        ax.bar(x - width / 2, self.S1, width, yerr=self.S1_conf,
               label='S1 (first-order)', capsize=3)
        ax.bar(x + width / 2, self.ST, width, yerr=self.ST_conf,
               label='ST (total)', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(self.var_names, rotation=45, ha='right')
        ax.legend(fontsize=8)

        if show:
            plt.show()
        return ax


@dataclass
class MorrisResult:
    """Results from Morris elementary effects screening.

    Attributes
    ----------
    n_trajectories : int
        Number of trajectories sampled.
    n_vars : int
        Number of input variables.
    var_names : list of str
        Variable names.
    mu_star : list of float
        Mean of absolute elementary effects (importance measure).
    sigma : list of float
        Standard deviation of elementary effects (interaction/nonlinearity).
    mu_star_conf : list of float
        95% confidence intervals for mu_star.
    """
    n_trajectories: int = 0
    n_vars: int = 0
    var_names: List[str] = field(default_factory=list)
    mu_star: List[float] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)
    mu_star_conf: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  MORRIS SENSITIVITY SCREENING RESULTS",
            "=" * 60,
            f"  Trajectories:  {self.n_trajectories}",
            f"  Variables:     {self.n_vars}",
            "",
            "  Variable        mu*         sigma",
            "  " + "-" * 44,
        ]
        for i, name in enumerate(self.var_names):
            mu = self.mu_star[i] if i < len(self.mu_star) else float('nan')
            sig = self.sigma[i] if i < len(self.sigma) else float('nan')
            lines.append(f"  {name:16s}  {mu:8.4f}    {sig:8.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_trajectories": self.n_trajectories,
            "n_vars": self.n_vars,
            "var_names": list(self.var_names),
            "mu_star": [float(x) for x in self.mu_star],
            "sigma": [float(x) for x in self.sigma],
            "mu_star_conf": [float(x) for x in self.mu_star_conf],
        }

    def plot_screening(self, ax=None, show=True, **kwargs):
        """Morris mu*/sigma scatter plot for factor screening."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        setup_engineering_plot(ax, "Morris Screening", "mu* (importance)", "sigma (nonlinearity/interactions)")

        ax.scatter(self.mu_star, self.sigma, s=80, zorder=5)
        for i, name in enumerate(self.var_names):
            ax.annotate(name, (self.mu_star[i], self.sigma[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        if show:
            plt.show()
        return ax
