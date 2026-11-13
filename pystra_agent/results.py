"""
Result dataclasses for pystra reliability analysis.
"""

from dataclasses import dataclass, field
from typing import Optional



def _f(value, spec):
    """``value`` formatted, or 'not reported' for None."""
    return "not reported" if value is None else format(value, spec)


@dataclass
class FormResult:
    """
    Results from First Order Reliability Method (FORM) analysis.

    FORM approximates the probability of failure using a first-order Taylor series
    expansion of the limit state function at the design point (most probable failure
    point). The reliability index beta is the shortest distance from the origin to
    the failure surface in standard normal space.

    Attributes:
        beta: Reliability index (dimensionless). Higher values indicate lower
            probability of failure. Typical values: 2-4 for structures, 3-5 for
            critical infrastructure.
        pf: Probability of failure. Related to beta by pf = Phi(-beta) where Phi
            is the standard normal CDF.
        alpha: Sensitivity factors (dict mapping variable name to float). Indicates
            contribution of each variable's uncertainty to overall reliability.
            Sum of squares equals 1.0. Sign indicates whether variable is resistance
            (+) or load (-).
        design_point_x: Design point in physical space (dict mapping variable name
            to float). Most probable failure point.
        design_point_u: Design point in standard normal space (dict mapping variable
            name to float). Shortest distance point from origin to failure surface.
        n_iterations: Number of iterations to convergence
        n_function_calls: Limit state function evaluations: pystra's own
            count where it keeps one (2.x), otherwise an estimate
            (iterations x variables); see n_function_calls_counted
        n_function_calls_counted: True when n_function_calls is pystra's
            count, False when it is the estimate
        converged: pystra's own convergence flag (2.x); on 1.x, which keeps
            no flag, whether it stopped before its iteration limit. Never
            inferred from the sign of beta: a converged run whose mean
            point is in the failure domain has beta < 0
        converged_source: Which of the two the flag is
        message: pystra's status message
        warnings: What the run could not give (an unconverged run has no
            index or design point on pystra 2.x; an estimate in place of a
            count). Empty for a clean run
        limit_state_expr: String representation of limit state function
        n_variables: Number of random variables in the analysis

    ``beta``, ``pf``, ``alpha`` and the design point are None / empty when
    pystra reports none (an unconverged run on pystra 2.x); ``warnings``
    then says so.

    Methods:
        summary() -> str: Human-readable summary
        to_dict() -> dict: Convert to dictionary (JSON-serializable)
        plot_importance(ax, show) -> ax: Bar chart of sensitivity factors
    """

    beta: float = 0.0
    pf: float = 1.0
    alpha: dict = field(default_factory=dict)
    design_point_x: dict = field(default_factory=dict)
    design_point_u: dict = field(default_factory=dict)
    n_iterations: int = 0
    n_function_calls: int = 0
    converged: bool = False
    limit_state_expr: str = ""
    n_variables: int = 0
    n_function_calls_counted: bool = False
    converged_source: str = ""
    message: str = ""
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        """
        Generate human-readable summary of FORM results.

        Returns:
            Multi-line string with key results and design point
        """
        lines = [
            "FORM Analysis Results",
            "=" * 60,
            f"Reliability Index (beta):     {_f(self.beta, '.4f')}",
            f"Probability of Failure (Pf):  {_f(self.pf, '.6e')}",
            f"Converged:                    {self.converged}",
            f"Iterations:                   {self.n_iterations}",
            f"Function Calls:               {self.n_function_calls}"
            + ("" if self.n_function_calls_counted else " (estimate)"),
            f"Number of Variables:          {self.n_variables}",
            "",
            f"Limit State Function: {self.limit_state_expr}",
            "",
        ]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        if self.warnings:
            lines.append("")

        if self.alpha:
            lines.append("Sensitivity Factors (alpha):")
            lines.append("-" * 40)
            for name, val in sorted(self.alpha.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {name:20s}: {val:8.4f}")
            lines.append("")

        if self.design_point_x:
            lines.append("Design Point (Physical Space):")
            lines.append("-" * 40)
            for name, val in self.design_point_x.items():
                u_val = self.design_point_u.get(name, 0.0)
                lines.append(f"  {name:20s}: {val:12.4f}  (u = {u_val:8.4f})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dict with all result fields
        """
        return {
            "beta": self.beta,
            "pf": self.pf,
            "alpha": self.alpha.copy(),
            "design_point_x": self.design_point_x.copy(),
            "design_point_u": self.design_point_u.copy(),
            "n_iterations": self.n_iterations,
            "n_function_calls": self.n_function_calls,
            "n_function_calls_counted": self.n_function_calls_counted,
            "converged": self.converged,
            "converged_source": self.converged_source,
            "message": self.message,
            "warnings": list(self.warnings),
            "limit_state_expr": self.limit_state_expr,
            "n_variables": self.n_variables,
        }

    def plot_importance(self, ax=None, show: bool = True):
        """
        Plot bar chart of sensitivity factor magnitudes.

        Args:
            ax: Matplotlib axes (creates new figure if None)
            show: Whether to display the plot

        Returns:
            Matplotlib axes object
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot

        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        setup_engineering_plot(
            ax, title='FORM Sensitivity Analysis',
            xlabel='|Sensitivity Factor|', ylabel='Variable'
        )

        if not self.alpha:
            ax.text(0.5, 0.5, "No sensitivity factors available",
                    ha='center', va='center', transform=ax.transAxes)
            return ax

        # Sort by absolute value
        items = sorted(self.alpha.items(), key=lambda x: abs(x[1]), reverse=True)
        names = [name for name, _ in items]
        values = [val for _, val in items]
        abs_values = [abs(val) for val in values]

        # Color bars by sign (positive = resistance, negative = load)
        colors = ['green' if v > 0 else 'red' for v in values]

        bars = ax.barh(names, abs_values, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel('|Sensitivity Factor|', fontsize=11, weight='bold')
        ax.set_ylabel('Variable', fontsize=11, weight='bold')
        ax.set_title('FORM Sensitivity Analysis', fontsize=12, weight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            label = f'{val:+.3f}'
            ax.text(width, bar.get_y() + bar.get_height()/2, f'  {label}',
                    va='center', fontsize=9)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Resistance (+)'),
            Patch(facecolor='red', alpha=0.7, label='Load (-)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()

        if show:
            plt.show()

        return ax


@dataclass
class SormResult:
    """
    Results from Second Order Reliability Method (SORM) analysis.

    SORM improves upon FORM by including second-order curvature information
    of the failure surface at the design point. This provides more accurate
    probability estimates for nonlinear limit state functions.

    Attributes:
        beta_form: FORM reliability index (first-order approximation)
        beta_breitung: SORM reliability index using Breitung's formula
        pf_breitung: SORM probability of failure (Breitung)
        kappa: Principal curvatures of the failure surface at the design point
        pf_form: FORM probability of failure (for comparison)
        alpha: Sensitivity factors from FORM (inherited)
        design_point_x: Design point in physical space (inherited)
        design_point_u: Design point in standard normal space (inherited)
        n_iterations: Number of iterations (inherited)
        n_function_calls: Limit state evaluations, FORM and SORM together:
            pystra's count (2.x) or an estimate (see FormResult)
        converged: FORM converged AND the second-order fit gave a value
            (both pystra's own flags; see FormResult.converged)
        form_converged, sorm_converged: The two flags apart
        sorm_failed: True when there is no second-order result.
            ``beta_breitung`` and ``pf_breitung`` are then None, never the
            FORM values under the SORM names
        sorm_message: pystra's message for the second-order stage
        warnings: What the run could not give
        limit_state_expr: String representation of limit state (inherited)
        n_variables: Number of random variables (inherited)

    Methods:
        summary() -> str: Human-readable summary
        to_dict() -> dict: Convert to dictionary (JSON-serializable)
    """

    beta_form: float = 0.0
    beta_breitung: Optional[float] = None
    pf_breitung: Optional[float] = None
    kappa: list = field(default_factory=list)
    pf_form: float = 1.0
    alpha: dict = field(default_factory=dict)
    design_point_x: dict = field(default_factory=dict)
    design_point_u: dict = field(default_factory=dict)
    n_iterations: int = 0
    n_function_calls: int = 0
    converged: bool = False
    limit_state_expr: str = ""
    n_variables: int = 0
    n_function_calls_counted: bool = False
    form_converged: bool = False
    sorm_converged: bool = False
    sorm_failed: bool = True
    converged_source: str = ""
    message: str = ""
    sorm_message: str = ""
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        """
        Generate human-readable summary of SORM results.

        Returns:
            Multi-line string with key results and comparison to FORM
        """
        lines = [
            "SORM Analysis Results",
            "=" * 60,
            "First-Order (FORM):",
            f"  Reliability Index (beta):   {_f(self.beta_form, '.4f')}",
            f"  Probability of Failure:     {_f(self.pf_form, '.6e')}",
            "",
            "Second-Order (SORM - Breitung):"
            + ("  FAILED, no second-order result" if self.sorm_failed else ""),
            f"  Reliability Index (beta):   {_f(self.beta_breitung, '.4f')}",
            f"  Probability of Failure:     {_f(self.pf_breitung, '.6e')}",
            "",
            f"Converged:                    {self.converged} "
            f"(FORM {self.form_converged}, SORM {self.sorm_converged})",
            f"Iterations:                   {self.n_iterations}",
            f"Function Calls:               {self.n_function_calls}"
            + ("" if self.n_function_calls_counted else " (estimate)"),
            f"Number of Variables:          {self.n_variables}",
            "",
            f"Limit State Function: {self.limit_state_expr}",
            "",
        ]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        if self.warnings:
            lines.append("")

        if self.kappa:
            lines.append("Principal Curvatures:")
            lines.append("-" * 40)
            for i, k in enumerate(self.kappa, 1):
                lines.append(f"  kappa_{i}: {k:12.6e}")
            lines.append("")

        if self.alpha:
            lines.append("Sensitivity Factors (alpha):")
            lines.append("-" * 40)
            for name, val in sorted(self.alpha.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {name:20s}: {val:8.4f}")
            lines.append("")

        if self.design_point_x:
            lines.append("Design Point (Physical Space):")
            lines.append("-" * 40)
            for name, val in self.design_point_x.items():
                u_val = self.design_point_u.get(name, 0.0)
                lines.append(f"  {name:20s}: {val:12.4f}  (u = {u_val:8.4f})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dict with all result fields
        """
        return {
            "beta_form": self.beta_form,
            "beta_breitung": self.beta_breitung,
            "pf_breitung": self.pf_breitung,
            "pf_form": self.pf_form,
            "kappa": self.kappa.copy() if self.kappa else [],
            "alpha": self.alpha.copy(),
            "design_point_x": self.design_point_x.copy(),
            "design_point_u": self.design_point_u.copy(),
            "n_iterations": self.n_iterations,
            "n_function_calls": self.n_function_calls,
            "n_function_calls_counted": self.n_function_calls_counted,
            "converged": self.converged,
            "form_converged": self.form_converged,
            "sorm_converged": self.sorm_converged,
            "sorm_failed": self.sorm_failed,
            "converged_source": self.converged_source,
            "message": self.message,
            "sorm_message": self.sorm_message,
            "warnings": list(self.warnings),
            "limit_state_expr": self.limit_state_expr,
            "n_variables": self.n_variables,
        }


@dataclass
class MonteCarloResult:
    """
    Results from Crude Monte Carlo simulation for reliability analysis.

    Monte Carlo estimates the failure probability, with no approximation of g,
    by sampling the joint distribution of random variables and counting failures.
    Accuracy improves with number of samples but computational cost is high.

    Attributes:
        beta: Reliability index, computed as Phi^-1(1 - pf) where Phi^-1 is
            the inverse standard normal CDF
        pf: Probability of failure, estimated as n_failures / n_samples
        n_samples: Monte Carlo samples run
        n_samples_requested: Samples asked for (the same, unless pystra
            ran a different number, which ``warnings`` then says)
        n_failures: Number of samples where limit state <= 0 (failure)
        cov_pf: Coefficient of variation of pf estimate,
            sqrt((1 - pf) / (n pf)). None when no sample failed (it is
            undefined there, not zero)
        pf_upper95_no_failures: With no failure, the upper end of the
            two-sided 95 % Clopper-Pearson interval on pf, 1 - 0.025^(1/n);
            None otherwise
        seed: The seed used; None means fresh entropy (not repeatable)
        warnings: What the run could not give (no failures, no seed)

    ``beta`` is None when no sample, or every sample, failed.
        limit_state_expr: String representation of limit state function
        n_variables: Number of random variables in the analysis

    Methods:
        summary() -> str: Human-readable summary
        to_dict() -> dict: Convert to dictionary (JSON-serializable)
    """

    beta: Optional[float] = 0.0
    pf: float = 1.0
    n_samples: int = 0
    n_failures: int = 0
    cov_pf: Optional[float] = 0.0
    limit_state_expr: str = ""
    n_variables: int = 0
    n_samples_requested: int = 0
    pf_upper95_no_failures: Optional[float] = None
    seed: Optional[int] = None
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        """
        Generate human-readable summary of Monte Carlo results.

        Returns:
            Multi-line string with key results and sampling statistics
        """
        lines = [
            "Monte Carlo Simulation Results",
            "=" * 60,
            f"Reliability Index (beta):     {_f(self.beta, '.4f')}",
            f"Probability of Failure (Pf):  {self.pf:.6e}",
            f"COV of Pf Estimate:           {_f(self.cov_pf, '.4f')}",
            "",
            f"Total Samples:                {self.n_samples:,}"
            + ("" if self.seed is None else f"  (seed {self.seed})"),
            f"Number of Failures:           {self.n_failures:,}",
            f"Number of Variables:          {self.n_variables}",
            "",
            f"Limit State Function: {self.limit_state_expr}",
            "",
        ]

        # Add guidance on COV
        if self.cov_pf is not None and self.cov_pf > 0:
            if self.cov_pf < 0.05:
                status = "Excellent (COV < 0.05)"
            elif self.cov_pf < 0.10:
                status = "Good (COV < 0.10)"
            elif self.cov_pf < 0.20:
                status = "Fair (COV < 0.20)"
            else:
                status = "Poor (COV > 0.20) - Consider more samples"
            lines.append(f"Estimate Quality: {status}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dict with all result fields
        """
        return {
            "beta": self.beta,
            "pf": self.pf,
            "n_samples": self.n_samples,
            "n_samples_requested": self.n_samples_requested or self.n_samples,
            "n_failures": self.n_failures,
            "cov_pf": self.cov_pf,
            "pf_upper95_no_failures": self.pf_upper95_no_failures,
            "seed": self.seed,
            "warnings": list(self.warnings),
            "limit_state_expr": self.limit_state_expr,
            "n_variables": self.n_variables,
        }
