"""
Calculation package steps for the reliability engines (BACKLOG GM-41).

One package for the four engines of the ``reliability`` tools -- FOSM
(Taylor series), the Rosenblueth point estimate method, Monte Carlo
simulation and FORM (Hasofer-Lind / Rackwitz-Fiessler) -- each rendered
from the engine's own result and the inputs its run used: the performance
function, every random variable with its distribution, the correlations,
then the moments of g, the reliability index and the probability of failure.
Nothing is worked out again here (GM-141 / GM-143).
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Reliability Analysis"

REFERENCES = [
    "UFC 3-220-20 (2025). Foundations and Earth Structures, Section 7-4.2 "
    "(reliability analysis): Eq. 7-7 and 7-8 (p. 533), Eq. 7-10 to 7-12 "
    "(p. 535), Eq. 7-13 and 7-14 (p. 536).",
    "Duncan, J.M. (2000). Factors of safety and reliability in geotechnical "
    "engineering. J. Geotech. Geoenviron. Eng., 126(4), 307-316.",
    "Rosenblueth, E. (1975). Point estimates for probability moments. "
    "Proc. Natl. Acad. Sci. USA, 72(10), 3812-3814.",
    "Hasofer, A.M. and Lind, N.C. (1974). Exact and invariant second-moment "
    "code format. J. Eng. Mech. Div., ASCE, 100(EM1), 111-121.",
    "Baecher, G.B. and Christian, J.T. (2003). Reliability and Statistics in "
    "Geotechnical Engineering. Wiley.",
]

_ENGINE_NAMES = {
    "fosm": "First-order second-moment (Taylor series, central differences)",
    "pem": "Point estimate method (Rosenblueth)",
    "monte_carlo": "Monte Carlo simulation",
    "form": "First-order reliability method (HL-RF)",
}


def _engine(result, analysis) -> str:
    if isinstance(analysis, dict) and analysis.get("engine"):
        return analysis["engine"]
    return {"FOSMResult": "fosm", "PEMResult": "pem",
            "MonteCarloResult": "monte_carlo",
            "FORMResult": "form"}.get(type(result).__name__, "fosm")


def _threshold(result) -> float:
    return 1.0 if result.convention == "fos" else 0.0


def _sig(x, n=4):
    """A value as printed: n significant figures (None stays None)."""
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return x
    return float(f"{x:.{n}g}")


def _num(x, fmt=".4g"):
    return "—" if x is None else format(x, fmt)


# ---------------------------------------------------------------------------
# Values Used
# ---------------------------------------------------------------------------

def get_input_summary(result, analysis) -> List[InputItem]:
    a = analysis if isinstance(analysis, dict) else {}
    eng = _engine(result, a)
    items = [
        InputItem("Engine", "Method", _ENGINE_NAMES.get(eng, eng)),
        InputItem("g", "Performance function", a.get("g_expression", "")),
        InputItem("Conv.", "Convention",
                  "factor of safety (failure at g < 1)"
                  if result.convention == "fos"
                  else "margin (failure at g < 0)"),
        InputItem("n_var", "Random variables", len(a.get("variables", []))),
        InputItem("ρ", "Correlation",
                  "none (independent)" if not a.get("correlation")
                  else "given (see the variable table)"),
    ]
    if eng == "pem":
        items.append(InputItem("Scheme", "Point scheme", a.get("scheme")))
    if eng == "monte_carlo":
        items.extend([
            InputItem("N", "Realizations", a.get("n")),
            InputItem("Sampling", "Sampling", a.get("sampling")),
            InputItem("Seed", "Random seed",
                      "none" if a.get("seed") is None else a.get("seed")),
        ])
    if eng == "form":
        items.append(InputItem("it_max", "Maximum iterations",
                               a.get("max_iterations")))
    return items


def _variable_table(a) -> TableData:
    rows = []
    for v in a.get("variables", []):
        rows.append([v["name"], v["dist"], _num(v["mean"]), _num(v["std"]),
                     _num(v["cov"], ".3f"), _num(v.get("lower")),
                     _num(v.get("upper")), _num(v.get("mode"))])
    return TableData(
        title="Random variables",
        headers=["Variable", "Distribution", "Mean", "Std. dev.", "COV",
                 "Lower", "Upper", "Mode"],
        rows=rows,
        notes="As the engine read them: a COV given is turned into the "
              "standard deviation (std = COV × |mean|) and the reverse.")


def _correlation_table(a):
    corr = a.get("correlation")
    if not corr:
        return None
    rows = []
    if isinstance(corr, dict):
        for k, v in corr.items():
            rows.append([str(k), _num(v, ".3f")])
    elif isinstance(corr, (list, tuple)):
        # a full matrix in variable order, the only list the engines take
        # (reliability.variables.build_correlation)
        names = [v["name"] for v in a.get("variables", [])]
        for i, e in enumerate(corr):
            rows.append([names[i] if i < len(names) else str(i),
                         ", ".join(_num(float(x), ".3f") for x in e)])
    return TableData(title="Correlation", headers=["Pair / row", "ρ"],
                     rows=rows, notes="As given to the engine.")


# ---------------------------------------------------------------------------
# Calculation steps
# ---------------------------------------------------------------------------

def get_calc_steps(result, analysis) -> List[CalcSection]:
    a = analysis if isinstance(analysis, dict) else {}
    eng = _engine(result, a)
    inputs = [_variable_table(a)]
    corr = _correlation_table(a)
    if corr is not None:
        inputs.append(corr)
    sections = [CalcSection(title="Random Variables", items=inputs)]
    if eng == "fosm":
        sections.append(CalcSection(title="FOSM (Taylor Series)",
                                    items=_fosm_steps(result)))
    elif eng == "pem":
        sections.append(CalcSection(title="Point Estimate Method",
                                    items=_pem_steps(result)))
    elif eng == "monte_carlo":
        sections.append(CalcSection(title="Monte Carlo Simulation",
                                    items=_mc_steps(result, a)))
    else:
        sections.append(CalcSection(title="FORM",
                                    items=_form_steps(result)))
    sections[-1].items.append(
        "β and Pf are what the assumed model implies: g as given "
        "and the variables' distributions and correlations above"
        + (", and g taken as linear at the design point" if eng == "form"
           else ", and for Φ(−β) the distribution named for g") + ". They are "
        "not a measured frequency of failure; UFC 3-220-20 7-4.2 (p. 533) "
        "calls β a relative measure of reliability.")
    return sections


def _beta_steps(r, ref_normal: str) -> list:
    thr = _threshold(r)
    out = []
    if r.convention == "fos":
        eq = "β = (μ<sub>g</sub> − 1) / σ<sub>g</sub>"
        sub = f"β = ({r.g_mean:.4f} − 1) / {r.g_std:.4f}"
    else:
        eq = "β = μ<sub>g</sub> / σ<sub>g</sub>"
        sub = f"β = {r.g_mean:.4f} / {r.g_std:.4f}"
    out.append(CalcStep(
        title="Reliability index (g normal)", equation=eq, substitution=sub,
        result_name="β", result_value=_sig(r.beta_normal),
        reference=ref_normal,
        notes="" if thr == 0 else "g = F − 1 in the form of Eq. 7-7."))
    pf_n = getattr(r, "pf_normal", None)
    if pf_n is not None:
        out.append(CalcStep(
            title="Probability of failure (g normal)",
            equation="P<sub>f</sub> = Φ(−β)",
            substitution=f"P<sub>f</sub> = Φ(−{r.beta_normal:.3f})",
            result_name="Pf", result_value=_sig(pf_n),
            reference="UFC 3-220-20 7-4.2"))
    if r.beta_lognormal is not None:
        out.append(CalcStep(
            title="Reliability index (factor of safety lognormal)",
            equation="β<sub>LN</sub> = [ln μ<sub>F</sub> − 0.5 ln(1 + "
                     "COV<sub>F</sub>²)] / [ln(1 + COV<sub>F</sub>²)]<sup>0.5"
                     "</sup>",
            substitution=f"μ<sub>F</sub> = {r.g_mean:.4f}, COV<sub>F</sub> = "
                         f"{r.g_cov:.4f}",
            result_name="β_LN", result_value=_sig(r.beta_lognormal),
            reference="UFC 3-220-20 Eq. 7-12, p. 535; Duncan (2000)"))
        out.append(CalcStep(
            title="Probability of failure (lognormal)",
            equation="P<sub>f</sub> = Φ(−β<sub>LN</sub>)",
            substitution=f"P<sub>f</sub> = Φ(−{r.beta_lognormal:.3f})",
            result_name="Pf_LN", result_value=_sig(r.pf_lognormal)))
    return out


def _fosm_steps(r) -> list:
    items = [CalcStep(
        title="Mean of g at the mean values",
        equation="μ<sub>g</sub> = g(μ<sub>X1</sub>, …, μ<sub>Xn</sub>)",
        substitution="g evaluated with every variable at its mean",
        result_name="μ_g", result_value=_sig(r.g_mean),
        reference="UFC 3-220-20 Eq. 7-8, p. 533")]
    rows = [[k, f"{d:.4f}", f"{d / 2:.4f}",
             f"{r.variance_contributions_pct.get(k, 0):.1f}"]
            for k, d in r.variable_deltas.items()]
    items.append(TableData(
        title="Central differences, one standard deviation either side",
        headers=["Variable", "Δg = g(μ+σ) − g(μ−σ)", "Δg / 2",
                 "Share of variance (%)"],
        rows=rows,
        notes="UFC 3-220-20 Eq. 7-10 (p. 535); the share is (Δg/2)² over "
              "their sum (Duncan 2000)."))
    items.append(CalcStep(
        title="Standard deviation of g",
        equation="σ<sub>g</sub> = [Σ (Δg<sub>i</sub>/2)²"
                 + (" + 2 Σ ρ<sub>ij</sub> (Δg<sub>i</sub>/2)(Δg<sub>j</sub>/2)"
                    if r.correlated else "") + "]<sup>0.5</sup>",
        substitution="from the table above",
        result_name="σ_g", result_value=_sig(r.g_std),
        reference="UFC 3-220-20 Eq. 7-11, p. 535"))
    items.append(CalcStep(
        title="Coefficient of variation of g",
        equation="COV<sub>g</sub> = σ<sub>g</sub> / μ<sub>g</sub>",
        substitution=f"COV<sub>g</sub> = {r.g_std:.4f} / {r.g_mean:.4f}",
        result_name="COV_g", result_value=_sig(r.g_cov)))
    items.extend(_beta_steps(r, "UFC 3-220-20 Eq. 7-7, p. 533"))
    items.append(f"g evaluated {r.n_g_calls} times (1 + 2n).")
    return items


def _pem_steps(r) -> list:
    items = [CalcStep(
        title="Mean of g over the 2ⁿ points",
        equation="μ<sub>g</sub> ≈ Σ P<sub>i</sub> g(X)<sub>i</sub>",
        substitution=f"{r.n_points} points, each variable at μ ± σ "
                     f"({r.scheme})",
        result_name="μ_g", result_value=_sig(r.g_mean),
        reference="UFC 3-220-20 Eq. 7-13, p. 536; Rosenblueth (1975)"),
        CalcStep(
        title="Standard deviation of g",
        equation="σ<sub>g</sub>² ≈ Σ P<sub>i</sub> g(X)<sub>i</sub>² − "
                 "(Σ P<sub>i</sub> g(X)<sub>i</sub>)²",
        substitution="same points",
        result_name="σ_g", result_value=_sig(r.g_std),
        reference="UFC 3-220-20 Eq. 7-14, p. 536"),
        CalcStep(
        title="Coefficient of variation of g",
        equation="COV<sub>g</sub> = σ<sub>g</sub> / μ<sub>g</sub>",
        substitution=f"COV<sub>g</sub> = {r.g_std:.4f} / {r.g_mean:.4f}",
        result_name="COV_g", result_value=_sig(r.g_cov))]
    items.extend(_beta_steps(r, "UFC 3-220-20 Eq. 7-7, p. 533"))
    return items


def _mc_steps(r, a) -> list:
    thr = _threshold(r)
    items = [CalcStep(
        title="Probability of failure (count)",
        equation=f"P<sub>f</sub> = N(g &lt; {thr:g}) / N",
        substitution=f"P<sub>f</sub> = {r.n_failed} / {r.n}",
        result_name="Pf", result_value=_sig(r.pf),
        reference="Baecher & Christian (2003)",
        notes=f"95 % confidence interval {r.pf_ci95[0]:.3g} to "
              f"{r.pf_ci95[1]:.3g}.")]
    items.append(TableData(
        title="Statistics of g over the realizations",
        headers=["Mean", "Std. dev.", "COV", "Min", "Median", "Max"],
        rows=[[f"{r.g_mean:.4f}", f"{r.g_std:.4f}", f"{r.g_cov:.4f}",
               f"{r.g_min:.4f}", f"{r.g_median:.4f}", f"{r.g_max:.4f}"]],
        notes="Percentiles: " + ", ".join(
            f"{k} {v:.4f}" for k, v in r.percentiles.items())))
    items.extend(_beta_steps(r, "UFC 3-220-20 Eq. 7-7, p. 533"))
    items.append("The indices from the moments of g are printed beside the "
                 "counted probability, which takes no distribution for g "
                 "(only the variables' own).")
    return items


def _form_steps(r) -> list:
    items = [CalcStep(
        title="Reliability index (design point)",
        equation="β = min ‖u‖ subject to g(u) = "
                 f"{_threshold(r):g}",
        substitution=f"HL-RF iteration, {r.n_iterations} iterations, "
                     + ("converged" if r.converged else "NOT converged"),
        result_name="β", result_value=_sig(r.beta),
        reference="Hasofer & Lind (1974); Baecher & Christian (2003)"),
        CalcStep(
        title="Probability of failure",
        equation="P<sub>f</sub> = Φ(−β)",
        substitution=f"P<sub>f</sub> = Φ(−{r.beta:.4f})",
        result_name="Pf", result_value=_sig(r.pf))]
    rows = [[k, f"{r.design_point.get(k, float('nan')):.4f}",
             f"{alpha:.4f}", f"{100 * alpha * alpha:.1f}"]
            for k, alpha in r.alphas.items()]
    items.append(TableData(
        title="Design point and sensitivities",
        headers=["Variable", "Design point x*", "α", "α² (%)"],
        rows=rows,
        notes=f"g at the design point = {r.g_at_design_point:.4g}."))
    if not r.converged:
        items.append("FORM did not converge: β and Pf are not reliable.")
    return items


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------

def get_figures(result, analysis) -> List[FigureData]:
    eng = _engine(result, analysis)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if eng == "monte_carlo" and result.histogram_counts:
        bins, counts = result.histogram_bins, result.histogram_counts
        fig, ax = plt.subplots(figsize=(7, 4))
        widths = [b - a for a, b in zip(bins[:-1], bins[1:])]
        ax.bar(bins[:-1], counts, width=widths, align="edge",
               color="#9db9d5", edgecolor="#1f4e79")
        ax.axvline(_threshold(result), color="#b30000", lw=1.5,
                   label=f"failure, g = {_threshold(result):g}")
        ax.set_xlabel("g")
        ax.set_ylabel("Realizations")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        b64 = figure_to_base64(fig)
        plt.close(fig)
        return [FigureData(
            title="Distribution of g", image_base64=b64,
            caption=f"{result.n} realizations; {result.n_failed} below the "
                    "failure line.", width_percent=70)]
    shares = (getattr(result, "variance_contributions_pct", None)
              or ({k: 100 * a * a for k, a in result.alphas.items()}
                  if eng == "form" else None))
    if shares:
        names = list(shares)
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))
        ax.barh(names, [shares[n] for n in names], color="#1f4e79")
        ax.set_xlabel("Share (%)")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        b64 = figure_to_base64(fig)
        plt.close(fig)
        what = ("variance of g" if eng == "fosm" else "α² at the design point")
        return [FigureData(
            title="Which variable matters", image_base64=b64,
            caption=f"Each variable's share of the {what}.",
            width_percent=65)]
    return []


def get_tables(result, analysis) -> list:
    """The random variables (CSV) and, for Monte Carlo, the histogram and
    the convergence of Pf."""
    from geotech_common.tables import CalcTable
    a = analysis if isinstance(analysis, dict) else {}
    tables = []
    rows = [[v["name"], v["dist"], v["mean"], v["std"], v["cov"],
             v.get("lower"), v.get("upper"), v.get("mode")]
            for v in a.get("variables", [])]
    if rows:
        tables.append(CalcTable(
            title="Random variables",
            columns=["Variable", "Distribution", "Mean", "Std. dev.", "COV",
                     "Lower", "Upper", "Mode"],
            rows=rows, note="As the engine read them", name="variables",
            printed=False))
    if _engine(result, a) == "monte_carlo":
        b, c = result.histogram_bins, result.histogram_counts
        if c:
            tables.append(CalcTable(
                title="Histogram of g",
                columns=["Bin from", "Bin to", "Realizations"],
                rows=[[b[i], b[i + 1], c[i]] for i in range(len(c))],
                note="Monte Carlo realizations of g", name="histogram",
                printed=False))
        if result.convergence:
            tables.append(CalcTable(
                title="Convergence of Pf",
                columns=["Realizations", "Pf"],
                rows=[[int(n), p] for n, p in result.convergence],
                note="Running probability of failure", name="convergence"))
    return tables
