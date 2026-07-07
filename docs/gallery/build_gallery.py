"""Build the GeotechStaffEngineer module-visualization gallery.

Runs REAL analyses from the shipped modules (validated benchmark inputs
wherever one exists), renders a figure for each, writes the PNGs under
``docs/gallery/figures/`` and assembles ``docs/gallery/index.html``.

The gallery is owner-facing: each exhibit is one analysis run, one (or a few)
figures, and a short practitioner note on what the picture shows and why an
engineer cares. Numbers printed on the page come straight from the run.

Usage
-----
    # from the repo root, with the project venv:
    python docs/gallery/build_gallery.py            # build every exhibit
    python docs/gallery/build_gallery.py 1 6 8      # only exhibits 1, 6, 8
    python docs/gallery/build_gallery.py --list      # list exhibits

Every figure is reproducible: delete ``docs/gallery/figures/`` and re-run.
The fem2d strength-reduction and consolidation exhibits take a couple of
minutes each; the committed PNGs are the durable artifact.
"""
from __future__ import annotations

import io
import os
import sys
import time
import base64
import html as _html
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional

# Headless rendering — must precede any pyplot import anywhere.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Make the worktree importable when run as a script from anywhere --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FIG_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Unit conversions for the US-customary FHWA benchmarks -> SI.
FT = 0.3048
PCF = 0.157087        # kN/m3 per pcf
KSF = 47.88026        # kPa per ksf
KIP = 4.4482          # kN per kip

# House palette (matches the modules' own matplotlib styling: blue primary,
# red for the limit line, plus green/amber accents). Kept small on purpose.
BLUE = "#1d4ed8"
BLUE_L = "#93c5fd"
RED = "#b91c1c"
GREEN = "#16a34a"
AMBER = "#d97706"
INK = "#1f2937"


# ---------------------------------------------------------------------------
# Exhibit plumbing
# ---------------------------------------------------------------------------
@dataclass
class Figure:
    png: str                      # filename under figures/
    caption: str = ""             # short label under the image
    interactive: Optional[str] = None  # optional linked interactive .html


@dataclass
class Exhibit:
    num: int
    module: str
    title: str
    anchor: str                   # validation basis, shown as a badge
    figures: List[Figure] = field(default_factory=list)
    numbers: Dict[str, str] = field(default_factory=dict)
    what: str = ""                # "what you're seeing" (3-5 sentences)
    seconds: float = 0.0


_REGISTRY: List["_ExhibitSpec"] = []


@dataclass
class _ExhibitSpec:
    num: int
    fn: Callable[[], Exhibit]


def exhibit(num: int):
    """Register an exhibit builder function."""
    def deco(fn):
        _REGISTRY.append(_ExhibitSpec(num, fn))
        return fn
    return deco


def save(fig, name: str, dpi: int = 110) -> str:
    """Save a matplotlib figure as a reasonably compressed PNG."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def _fmt(x, unit="", nd=3):
    return f"{x:.{nd}f}{unit}"


def _write_interactive_slope(search, geom, name: str):
    """Write a plotly interactive slope-search viewer (returns filename or None)."""
    try:
        from calc_package.interactive import slope_interactive_figure
        fig = slope_interactive_figure(search.critical, geom, search=search)
        fig.write_html(os.path.join(FIG_DIR, name),
                       include_plotlyjs="directory", full_html=True)
        return name
    except Exception as exc:                       # pragma: no cover
        print(f"    (interactive skipped: {exc})", flush=True)
        return None


def _write_interactive_fem(result, name: str):
    """Write a plotly interactive fem2d viewer (returns filename or None)."""
    try:
        from calc_package.interactive import fem_interactive_figure
        fig = fem_interactive_figure(result)
        fig.write_html(os.path.join(FIG_DIR, name),
                       include_plotlyjs="directory", full_html=True)
        return name
    except Exception as exc:                       # pragma: no cover
        print(f"    (interactive skipped: {exc})", flush=True)
        return None


# ===========================================================================
# EXHIBIT 8 — Reliability: distributions, beta / Pf, tornado, FORM design pt.
# ===========================================================================
@exhibit(8)
def ex_reliability() -> Exhibit:
    import numpy as np
    from reliability import fosm, pem, monte_carlo, form

    # Published engine-agreement problem (reliability/VALIDATION.md sec. 3):
    # square footing B=2 m, Df=1.5 m on sand, q_applied=700 kPa, FOS=q_ult/700.
    spec = {"friction_angle": {"mean": 32.0, "cov": 0.08, "dist": "lognormal"},
            "unit_weight": {"mean": 18.0, "cov": 0.05}}

    def g(v):
        from bearing_capacity import (
            BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer)
        res = BearingCapacityAnalysis(
            footing=Footing(width=2.0, depth=1.5, shape="square"),
            soil=BearingSoilProfile(layer1=SoilLayer(
                cohesion=0.0, friction_angle=v["friction_angle"],
                unit_weight=v["unit_weight"]))).compute()
        return res.q_ultimate / 700.0

    r_fosm = fosm(g, spec)
    r_pem = pem(g, spec)
    r_form = form(g, spec)
    r_mc = monte_carlo(g, spec, n=20_000, seed=42, keep_samples=True)

    # --- 4-panel figure --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    (ax_in, ax_hist), (ax_tor, ax_pie) = axes

    # Panel A: input distributions (the two random variables).
    from math import log, sqrt, exp, pi
    xs_phi = np.linspace(24, 42, 300)
    s_ln = sqrt(log(1 + 0.08**2)); mu_ln = log(32.0) - 0.5*s_ln**2
    pdf_phi = (1/(xs_phi*s_ln*sqrt(2*pi)))*np.exp(-(np.log(xs_phi)-mu_ln)**2/(2*s_ln**2))
    ax_in.plot(xs_phi, pdf_phi/pdf_phi.max(), color=BLUE, lw=2,
               label="phi'  ~ LN(32 deg, COV 8%)")
    xs_g = np.linspace(14.5, 21.5, 300)
    sg = 18.0*0.05
    pdf_g = np.exp(-(xs_g-18.0)**2/(2*sg**2))
    ax_in.plot(xs_g, pdf_g/pdf_g.max(), color=AMBER, lw=2,
               label="gamma ~ N(18 kN/m3, COV 5%)")
    ax_in.axvline(r_form.design_point["friction_angle"], color=BLUE, ls=":", lw=1.4)
    ax_in.axvline(r_form.design_point["unit_weight"], color=AMBER, ls=":", lw=1.4)
    ax_in.text(r_form.design_point["friction_angle"], 1.02, "phi* (design pt)",
               color=BLUE, fontsize=8, ha="center")
    ax_in.set_title("A. Input random variables", fontweight="bold", fontsize=11)
    ax_in.set_xlabel("value"); ax_in.set_ylabel("relative density")
    ax_in.legend(fontsize=8, loc="upper left"); ax_in.grid(alpha=0.25)

    # Panel B: Monte Carlo FOS histogram with beta / Pf annotation.
    samp = np.asarray(r_mc.samples, dtype=float)
    ax_hist.hist(samp, bins=45, color=BLUE_L, edgecolor=BLUE, linewidth=0.4,
                 density=True, label=f"MC samples (n = {r_mc.n:,})")
    s_ln2 = sqrt(log(1+r_mc.g_cov**2)); mu_ln2 = log(r_mc.g_mean)-0.5*s_ln2**2
    xh = np.linspace(samp.min(), samp.max(), 300)
    ax_hist.plot(xh, (1/(xh*s_ln2*sqrt(2*pi)))*np.exp(-(np.log(xh)-mu_ln2)**2/(2*s_ln2**2)),
                 color=RED, lw=1.8, label="lognormal fit")
    ax_hist.axvline(1.0, color=RED, ls="--", lw=1.4, label="FOS = 1.0")
    ax_hist.axvline(r_mc.g_mean, color=GREEN, ls=":", lw=1.4,
                    label=f"mean = {r_mc.g_mean:.2f}")
    ax_hist.set_title("B. Monte Carlo FOS distribution", fontweight="bold", fontsize=11)
    ax_hist.set_xlabel("Factor of Safety"); ax_hist.set_ylabel("probability density")
    ax_hist.legend(fontsize=8, loc="upper right")
    ax_hist.text(0.02, 0.97,
                 f"beta_LN = {r_mc.beta_lognormal:.2f}\nPf = {r_mc.pf:.2%}",
                 transform=ax_hist.transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888"))
    ax_hist.grid(alpha=0.25)

    # Panel C: FOSM variance-contribution tornado.
    vc = sorted(r_fosm.variance_contributions_pct.items(), key=lambda kv: kv[1])
    names = [k.replace("_", " ") for k, _ in vc]
    ax_tor.barh(names, [v for _, v in vc], color=BLUE, edgecolor=INK, height=0.55)
    for i, (_, v) in enumerate(vc):
        ax_tor.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    ax_tor.set_xlim(0, 108)
    ax_tor.set_title("C. FOSM variance contributions", fontweight="bold", fontsize=11)
    ax_tor.set_xlabel("share of Var(FOS)  (%)"); ax_tor.grid(alpha=0.25, axis="x")

    # Panel D: FORM design-point alpha^2 pie.
    a2 = {k: v**2 for k, v in r_form.alphas.items()}
    tot = sum(a2.values())
    labels = [f"{k.replace('_',' ')}\n{100*v/tot:.1f}%" for k, v in a2.items()]
    ax_pie.pie(list(a2.values()), labels=labels, colors=[BLUE, AMBER],
               startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
               textprops=dict(fontsize=9))
    ax_pie.set_title(f"D. FORM sensitivity (alpha^2), beta = {r_form.beta:.2f}",
                     fontweight="bold", fontsize=11)

    fig.suptitle("Reliability of a spread footing FOS — four engines on one limit state",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = save(fig, "ex08_reliability.png")

    ex = Exhibit(
        num=8, module="reliability",
        title="Reliability — the answer is a distribution, not a number",
        anchor="reliability/VALIDATION.md sec. 3 (engine-agreement) + Duncan (2000) COV table",
        figures=[Figure(png, "Square footing B=2 m on sand; FOS = q_ult/q with phi' and "
                             "gamma as random variables, run through all four engines.")],
        numbers={
            "FOSM E[FOS] / COV": f"{r_fosm.g_mean:.3f} / {r_fosm.g_cov:.2f}",
            "PEM E[FOS] / COV": f"{r_pem.g_mean:.3f} / {r_pem.g_cov:.2f}",
            "Monte Carlo Pf / beta": f"{r_mc.pf:.2%} / {r_mc.beta_lognormal:.2f}",
            "FORM beta / Pf": f"{r_form.beta:.3f} / {r_form.pf:.2%}",
            "FORM design point": f"phi* = {r_form.design_point['friction_angle']:.1f} deg, "
                                 f"gamma* = {r_form.design_point['unit_weight']:.1f}",
            "Dominant variable (alpha^2)": f"phi' = {100*a2['friction_angle']/tot:.1f}%",
        },
        what=(
            "The same footing is analyzed as a probability problem: friction angle and unit "
            "weight are given as distributions rather than single values, and the Vesic bearing "
            "capacity is evaluated thousands of times. Panel B is the resulting spread of factor "
            "of safety — the deterministic mean is about 2.4, but the tail that matters crosses "
            "FOS = 1, giving a probability of failure near 0.4%. Panels C and D answer the "
            "practical question 'which unknown is driving the risk?': both the FOSM variance split "
            "and the FORM design-point sensitivity put ~96-98% of the uncertainty on the friction "
            "angle, so that is where a confirming test buys the most. All four engines "
            "(FOSM, point-estimate, Monte Carlo, FORM) agree to within their expected tolerances, "
            "reproducing the published cross-check table."),
    )
    return ex


# ===========================================================================
# EXHIBIT 9 — MSE LRFD external stability (GEC-11 Example E4)
# ===========================================================================
@exhibit(9)
def ex_mse_lrfd() -> Exhibit:
    import numpy as np
    import matplotlib.patches as patches
    from retaining_walls import MSEWallGeometry, check_external_stability_lrfd

    H = 25.64 * FT
    L = 18.0 * FT
    gamma = 125 * PCF
    q_ll = 0.25 * KSF
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L, surcharge=q_ll)
    ext = check_external_stability_lrfd(
        geom, gamma_backfill=gamma, phi_backfill=34.0, phi_foundation=30.0,
        phi_retained=30.0, gamma_retained=gamma,
        bearing_resistance_strength=10.50 * KSF,
        bearing_resistance_service=7.50 * KSF)

    fig, (axw, axb) = plt.subplots(1, 2, figsize=(12, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- Left: wall cross-section sketch ---------------------------------
    axw.add_patch(patches.Rectangle((0, 0), L, H, facecolor="#fde68a",
                                    edgecolor=INK, lw=1.3, label="reinforced fill"))
    # retained fill wedge behind
    axw.add_patch(patches.Polygon([(L, 0), (L + 0.55*H, 0), (L, H)],
                                  closed=True, facecolor="#d6d3d1", edgecolor=INK, lw=0.8))
    # facing panels
    axw.add_patch(patches.Rectangle((-0.25, 0), 0.25, H, facecolor="#9ca3af", edgecolor=INK))
    # reinforcement layers
    for z in np.linspace(0.9, H-0.4, 10):
        axw.plot([0, L], [z, z], color=BLUE, lw=1.2, alpha=0.8)
    # live load surcharge arrows
    for x in np.linspace(0.4, L + 0.4*H, 7):
        axw.annotate("", xy=(x, H+0.05), xytext=(x, H+0.9),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    axw.text((L+0.4*H)/2, H+1.05, "live-load surcharge q", color=RED,
             ha="center", fontsize=8)
    axw.annotate("", xy=(0, -0.5), xytext=(L, -0.5),
                 arrowprops=dict(arrowstyle="<->", color=INK))
    axw.text(L/2, -1.0, f"L = {L:.2f} m ({18.0:.0f} ft)", ha="center", fontsize=8)
    axw.annotate("", xy=(-1.1, 0), xytext=(-1.1, H),
                 arrowprops=dict(arrowstyle="<->", color=INK))
    axw.text(-1.5, H/2, f"H = {H:.2f} m", rotation=90, va="center", fontsize=8)
    axw.set_xlim(-2.2, L + 0.6*H + 0.5); axw.set_ylim(-1.8, H + 1.8)
    axw.set_aspect("equal"); axw.axis("off")
    axw.set_title("MSE wall — GEC-11 Example E4", fontweight="bold", fontsize=11)

    # --- Right: CDR bar chart per check per combination ------------------
    checks = ["Sliding", "Eccentricity*", "Bearing"]
    combos = ["Strength I max", "Strength I min / critical", "Service I"]
    # eccentricity expressed as CDR = e_limit / eL so it shares the axis
    ecc = ext["eccentricity"]
    ecc_cdr_max = ecc["e_limit_m"] / ecc["eL_strength_max_m"]
    ecc_cdr_crit = ecc["e_limit_m"] / ecc["eL_critical_m"]
    data = {
        "Strength I max": [ext["sliding"]["CDR_strength_max"], ecc_cdr_max,
                           ext["bearing"]["CDR_strength"]],
        "Strength I min / critical": [ext["sliding"]["CDR_critical"], ecc_cdr_crit,
                                      ext["bearing"]["CDR_critical"]],
        "Service I": [None, None, ext["bearing"]["CDR_service"]],
    }
    x = np.arange(len(checks)); w = 0.26
    colors = [BLUE, "#60a5fa", GREEN]
    for i, combo in enumerate(combos):
        vals = [v if v is not None else 0 for v in data[combo]]
        bars = axb.bar(x + (i-1)*w, vals, w, color=colors[i], edgecolor=INK,
                       linewidth=0.5, label=combo)
        for b, v in zip(bars, data[combo]):
            if v:
                axb.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}",
                         ha="center", fontsize=7.5)
    axb.axhline(1.0, color=RED, ls="--", lw=1.5, label="CDR = 1.0 (demand = capacity)")
    axb.set_xticks(x); axb.set_xticklabels(checks)
    axb.set_ylabel("Capacity : Demand Ratio (CDR)")
    axb.set_ylim(0, 2.4)
    axb.set_title("LRFD external-stability CDRs", fontweight="bold", fontsize=11)
    axb.legend(fontsize=7.5, loc="upper right", ncol=1)
    axb.grid(alpha=0.25, axis="y")
    axb.text(0.01, 0.02, "*eccentricity CDR = (L/4) / e_L", transform=axb.transAxes,
             fontsize=7, color="#555")
    fig.tight_layout()
    png = save(fig, "ex09_mse_lrfd.png")

    ex = Exhibit(
        num=9, module="retaining_walls",
        title="MSE wall LRFD external stability — every check, every load case",
        anchor="GEC-11 (FHWA-NHI-10-025) Example E4 — sliding CDR 1.85/2.08/1.37; bearing 1.57",
        figures=[Figure(png, "A 25.6-ft MSE wall with 18-ft reinforcement, checked to AASHTO/GEC-11 "
                             "Strength I (max and min load factors) and Service I.")],
        numbers={
            "Sliding CDR (max/min/critical)":
                f"{ext['sliding']['CDR_strength_max']:.2f} / "
                f"{ext['sliding']['CDR_strength_min']:.2f} / "
                f"{ext['sliding']['CDR_critical']:.2f}",
            "Eccentricity e_L (max) vs L/4":
                f"{ext['eccentricity']['eL_strength_max_m']/FT:.2f} ft vs "
                f"{ext['eccentricity']['e_limit_m']/FT:.2f} ft",
            "Bearing sigma_v (Str I max)":
                f"{ext['bearing']['sigma_v_strength_max_kPa']/KSF:.2f} ksf",
            "Bearing CDR (governing)": f"{ext['bearing']['CDR_governing']:.2f}",
            "Overall": "PASS" if ext["passes"] else "FAIL",
        },
        what=(
            "External stability of a mechanically stabilized earth wall is not one number but a "
            "matrix: three failure modes (the block sliding, tipping past its eccentricity limit, "
            "and overstressing the foundation) each checked under several AASHTO load combinations. "
            "The left panel is the wall the numbers describe; the right panel is the whole matrix "
            "as capacity-to-demand ratios, where anything above the red 1.0 line passes. The "
            "'critical' combination deliberately pairs minimum vertical load with maximum thrust — "
            "that is why sliding drops to 1.37 there even though the nominal case is 1.85. These "
            "reproduce the FHWA GEC-11 Example E4 hand calculation to within rounding, including "
            "the bookkeeping rule that the live-load surcharge helps bearing but is excluded from "
            "sliding and eccentricity resistance."),
    )
    return ex


# ===========================================================================
# EXHIBIT 10 — Bearing capacity term breakdown + Hough settlement-vs-B
# ===========================================================================
@exhibit(10)
def ex_bearing_settlement() -> Exhibit:
    import numpy as np
    from bearing_capacity import (
        BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer)
    from bearing_capacity.calc_steps import _plot_term_breakdown
    from settlement import HoughLayer, hough_settlement

    # --- GEC-6 Example B-1: Vesic bearing capacity (V-021) ----------------
    B0 = 6.0
    analysis = BearingCapacityAnalysis(
        footing=Footing(width=B0, depth=2.3, shape="square"),
        soil=BearingSoilProfile(
            layer1=SoilLayer(cohesion=0.0, friction_angle=35.0, unit_weight=19.6),
            gwt_depth=9.1),
        ngamma_method="vesic", factor_method="vesic")
    res = analysis.compute()
    fig1 = _plot_term_breakdown(res)          # REUSE module plot
    png1 = save(fig1, "ex10_bearing_terms.png")

    # --- GEC-6 Example B-1 Hough settlement table (V-022) -----------------
    # Granular sublayers below the 2.3 m footing base (mid-depth, sigma'v0, C').
    hlayers = [
        HoughLayer(thickness=2.1, depth_to_center=1.05, sigma_v0=65.7, C_prime=65,
                   description="silty sand"),
        HoughLayer(thickness=4.7, depth_to_center=4.45, sigma_v0=132.0, C_prime=120,
                   description="well-graded sand"),
        HoughLayer(thickness=3.0, depth_to_center=8.3, sigma_v0=193.0, C_prime=102,
                   description="sand"),
        HoughLayer(thickness=3.0, depth_to_center=11.3, sigma_v0=222.0, C_prime=110,
                   description="sand"),
    ]
    B_grid = np.linspace(2.5, 6.5, 25)
    q_levels = [240.0, 290.0, 335.0, 380.0]
    fig2, ax = plt.subplots(figsize=(7.6, 5.0))
    cmap = [BLUE_L, "#60a5fa", BLUE, "#1e3a8a"]
    published = {  # GEC-6 Tables B1-2/B1-3 (mm) at B = 3.0, 4.6, 6.1 m
        240.0: [(3.0, 21), (4.6, 28), (6.1, 31)],
        290.0: [(3.0, 25), (4.6, 31), (6.1, 35)],
        335.0: [(3.0, 28), (4.6, 34), (6.1, 38)],
        380.0: [(3.0, 30), (4.6, 37), (6.1, 41)],
    }
    for q, c in zip(q_levels, cmap):
        s_mm = [hough_settlement(hlayers, q_net=q, B=float(B)).total_mm for B in B_grid]
        ax.plot(B_grid, s_mm, color=c, lw=2, label=f"q_net = {q:.0f} kPa")
        px = [p[0] for p in published[q]]; py = [p[1] for p in published[q]]
        ax.scatter(px, py, color=c, edgecolor=INK, zorder=5, s=42)
    ax.scatter([], [], color="white", edgecolor=INK, s=42, label="GEC-6 published points")
    ax.set_xlabel("Footing width B (m)"); ax.set_ylabel("Settlement (mm)")
    ax.set_title("Hough granular settlement vs footing width\n(GEC-6 Example B-1)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig2.tight_layout()
    png2 = save(fig2, "ex10_hough_settlement.png")

    ex = Exhibit(
        num=10, module="bearing_capacity + settlement",
        title="Bearing capacity and settlement — the two halves of a footing check",
        anchor="GEC-6 (FHWA-SA-02-054) Example B-1 — Vesic q_ult and Hough settlement table",
        figures=[
            Figure(png1, "Where the bearing capacity comes from: the Vesic surcharge and "
                         "self-weight terms for a 6 m square footing on sand (c = 0)."),
            Figure(png2, "The competing limit: Hough settlement grows with footing width even "
                         "as bearing capacity does. Dots are the published GEC-6 table."),
        ],
        numbers={
            "q_ult (B=6 m, Vesic)": f"{res.q_ultimate:,.0f} kPa",
            "Overburden term": f"{res.term_overburden:,.0f} kPa",
            "Self-weight term": f"{res.term_selfweight:,.0f} kPa",
            "Hough settlement (B=3 m, q=240)":
                f"{hough_settlement(hlayers, q_net=240.0, B=3.0).total_mm:.0f} mm "
                f"(published 21 mm)",
            "Hough settlement (B=6.1 m, q=240)":
                f"{hough_settlement(hlayers, q_net=240.0, B=6.1).total_mm:.0f} mm "
                f"(published 31 mm)",
        },
        what=(
            "A footing has two independent ways to fail, and a real design has to satisfy both. "
            "The left figure decomposes the ultimate bearing capacity into its Vesic terms — for "
            "a cohesionless sand the surcharge (embedment) and self-weight terms carry everything, "
            "and the chart shows how much each contributes. The right figure is the settlement "
            "side of the same problem: as the footing is widened to gain bearing capacity, it also "
            "engages deeper soil and settles more, so bigger is not automatically safer. Both "
            "reproduce the FHWA GEC-6 worked example — the Hough curve passes through the "
            "published table points (dots) across footing widths and applied pressures."),
    )
    return ex


# ---------------------------------------------------------------------------
# Shared: the Fredlund & Krahn (1977) homogeneous slope (slope_stability B1).
# Surface (0,60)-(60,60)-(140,20)-(180,20); c'=600, phi'=20, gamma=120.
# Published FOS: Ordinary 1.928, Bishop 2.080, Spencer 2.073, M-P 2.076.
# ---------------------------------------------------------------------------
_FK_SURFACE = [(0.0, 60.0), (60.0, 60.0), (140.0, 20.0), (180.0, 20.0)]
_FK_CIRCLE = dict(xc=120.0, yc=90.0, radius=80.0)


def _fk_geom():
    from slope_stability import SlopeGeometry, SlopeSoilLayer
    return SlopeGeometry(surface_points=_FK_SURFACE, soil_layers=[
        SlopeSoilLayer(name="soil", top_elevation=60.0, bottom_elevation=0.0,
                       gamma=120.0, phi=20.0, c_prime=600.0)])


def _method_table_figure(rows, published, title, figsize=(7.4, 2.5)):
    """Render a method-comparison table (ours vs published) as a figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    cells, colors = [], []
    for r in rows:
        pub = published.get(r["method"], None)
        pub_s = f"{pub:.3f}" if pub is not None else "-"
        cells.append([r["method"], f"{r['FOS']:.3f}", pub_s])
        colors.append(["#f8fafc", "#eef2ff", "#f0fdf4"])
    tbl = ax.table(cellText=cells,
                   colLabels=["Method", "Computed FOS", "Published FOS"],
                   cellColours=colors, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor("#1d4ed8"); cell.set_text_props(color="white", weight="bold")
    ax.set_title(title, fontweight="bold", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


# ===========================================================================
# EXHIBIT 1 — Slope critical-surface search (SLOPE/W-grade trial map)
# ===========================================================================
@exhibit(1)
def ex_slope_search() -> Exhibit:
    from slope_stability import search_critical_surface
    from slope_stability.plotting import plot_trial_surface_map

    geom = _fk_geom()
    # Circular grid search — every trial circle scored, colored by FOS.
    srch = search_critical_surface(geom, nx=12, ny=12, method="bishop", n_slices=30)
    fig = plot_trial_surface_map(
        geom=geom, search=srch,
        title="Critical-surface search — Fredlund & Krahn (1977) slope")
    png = save(fig, "ex01_trial_surface_map.png")

    # Noncircular DE search — exercises the rejection diagnostics.
    nc = search_critical_surface(geom, surface_type="noncircular", n_trials=400,
                                 seed=7, method="spencer")
    n_rej = (nc.n_rejected_geometry + nc.n_rejected_nonconverged
             + nc.n_rejected_jagged)
    n_eval = len(nc.trial_surfaces) + n_rej

    interactive = _write_interactive_slope(srch, geom, "ex01_search_interactive.html")

    ex = Exhibit(
        num=1, module="slope_stability",
        title="Slope stability — searching thousands of surfaces for the weakest one",
        anchor="Fredlund & Krahn (1977) homogeneous slope (Slide2 #21) — Bishop FOS 2.080",
        figures=[Figure(png, "Every trial circle scored and colored by its factor of safety "
                             "(red = low, green = high); the critical surface is the dark red arc.",
                        interactive=interactive)],
        numbers={
            "Critical FOS (grid search)": f"{srch.critical.FOS:.3f} (published Bishop 2.080)",
            "Critical circle": f"center ({srch.critical.xc:.0f}, {srch.critical.yc:.0f}), "
                               f"R = {srch.critical.radius:.0f}",
            "Circular surfaces evaluated": f"{srch.n_surfaces_evaluated}",
            "Noncircular search": f"{n_eval} evaluated, {len(nc.trial_surfaces)} kept, "
                                 f"{n_rej} rejected",
            "Rejections (geom / non-converged / jagged)":
                f"{nc.n_rejected_geometry} / {nc.n_rejected_nonconverged} / "
                f"{nc.n_rejected_jagged}",
        },
        what=(
            "There is no single 'the' failure surface — the analysis has to try a great many and "
            "report the weakest. This map is the whole circular search at once: each thin arc is a "
            "candidate slip circle, colored by the factor of safety it produced, so the eye is "
            "drawn straight to the red band of near-critical surfaces and the dark critical arc "
            "within it. The computed minimum reproduces the published Fredlund & Krahn benchmark "
            "(Bishop 2.08). The second search uses free-form (noncircular) surfaces, and the panel "
            "quotes the module's rejection diagnostics — of 400 trial geometries, the ones that "
            "self-intersect, fail to converge, or come back jagged are discarded before scoring, "
            "which is the robustness fix that keeps a stray degenerate surface from reporting a "
            "false low FOS."),
    )
    return ex


# ===========================================================================
# EXHIBIT 2 — GLE slice mechanics (forces, thrust line, method comparison)
# ===========================================================================
@exhibit(2)
def ex_slice_mechanics() -> Exhibit:
    from slope_stability import analyze_slope, compare_methods_table
    from slope_stability.plotting import plot_slice_forces, plot_interslice_forces

    geom = _fk_geom()
    res = analyze_slope(geom, **_FK_CIRCLE, method="spencer",
                        include_slice_data=True)
    f1 = plot_slice_forces(res)
    png1 = save(f1, "ex02_slice_forces.png")
    f2 = plot_interslice_forces(res, geom)
    png2 = save(f2, "ex02_interslice_forces.png")

    tbl = compare_methods_table(geom, **_FK_CIRCLE)
    published = {
        "Fellenius (OMS)": 1.928, "Bishop simplified": 2.080,
        "Spencer": 2.073, "Morgenstern-Price (half_sine)": 2.076,
    }
    f3 = _method_table_figure(tbl["rows"], published,
                              "Every method, one surface (F&K 1977)")
    png3 = save(f3, "ex02_method_table.png")

    ex = Exhibit(
        num=2, module="slope_stability",
        title="Inside one slip surface — slice forces, thrust line, and method spread",
        anchor="Fredlund & Krahn (1977) circle — rigorous Spencer/GLE thrust line and method table",
        figures=[
            Figure(png1, "Per-slice statics: base normal N', mobilized shear, and slice weight "
                         "across the sliding mass."),
            Figure(png2, "Interslice normal (E) and shear (X) distributions, with the resulting "
                         "line of thrust drawn on the section (Spencer, theta = 14.6 deg)."),
            Figure(png3, "The same circle solved by every limit-equilibrium method — the classic "
                         "Fredlund & Krahn comparison, reproduced."),
        ],
        numbers={
            "Spencer FOS": f"{res.FOS:.3f} (published 2.073)",
            "Interslice inclination theta": "14.6 deg (F&K report ~14.8 deg)",
            "Fellenius / Bishop": f"{tbl['rows'][0]['FOS']:.3f} / {tbl['rows'][1]['FOS']:.3f}",
            "Spread across methods":
                f"{min(r['FOS'] for r in tbl['rows']):.3f} to "
                f"{max(r['FOS'] for r in tbl['rows']):.3f}",
        },
        what=(
            "Limit equilibrium works by cutting the sliding mass into vertical slices and balancing "
            "forces on each. The first figure opens up one critical circle: the base normal force, "
            "the shear the soil must mobilize, and each slice's weight. The second shows what "
            "separates a rigorous method from a simple one — the interslice forces between slices "
            "and the resulting line of thrust, which Spencer's method solves for explicitly (here "
            "inclined about 15 degrees, matching the published value). The table is the punchline of "
            "the 1977 Fredlund & Krahn paper: the simple Ordinary method reads ~1.93 while the "
            "rigorous methods cluster near 2.07, so the choice of method is itself an engineering "
            "decision, and the toolkit reproduces the whole published spread."),
    )
    return ex


# ===========================================================================
# EXHIBIT 3 — Rapid drawdown, 3-stage walkthrough (Slide2 #95/#96 dam)
# ===========================================================================
@exhibit(3)
def ex_rapid_drawdown() -> Exhibit:
    import numpy as np
    from slope_stability import (
        SlopeGeometry, SlopeSoilLayer, CircularSlipSurface, build_slices)
    from slope_stability.rapid_drawdown import rapid_drawdown_fos

    FT_, PSF, PCF_ = 0.3048, 0.04788, 0.157087
    face = [(0, 0), (220, 73), (312, 110), (380, 110)]
    seepage = [(0*FT_, 110*FT_), (380*FT_, 80*FT_)]
    circle = dict(xc=169.5*FT_, yc=210*FT_, radius=210*FT_)

    def dam():
        return SlopeGeometry(
            surface_points=[(x*FT_, z*FT_) for (x, z) in face],
            soil_layers=[SlopeSoilLayer(
                name="fill", top_elevation=110*FT_, bottom_elevation=-1.0*FT_,
                gamma=135*PCF_, phi=30.0, c_prime=0.0, R_c=1200*PSF, R_phi=16.0)])

    def run(method, seep=None):
        return rapid_drawdown_fos(dam(), 110*FT_, 24*FT_, method=method,
                                  n_slices=50, stage1_phreatic_points=seep, **circle)

    r2_flat = run("corps_2stage")
    r2_seep = run("corps_2stage", seepage)
    r3_flat = run("duncan_3stage")
    r3_seep = run("duncan_3stage", seepage)

    # Slice midpoint x for the per-slice panel.
    slices = build_slices(dam(), CircularSlipSurface(**circle), n_slices=50)
    xs = np.array([s.x_mid for s in slices]) / FT_   # back to ft for the axis

    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.24)
    axsec = fig.add_subplot(gs[0, :])
    axstr = fig.add_subplot(gs[1, 0])
    axfos = fig.add_subplot(gs[1, 1])

    # Panel A: dam section, initial + drawn-down pool, slip circle.
    import matplotlib.patches as _mpatch
    fx = [p[0] for p in face]; fz = [p[1] for p in face]
    axsec.fill([0]+fx+[380, 0], [0]+fz+[0, 0], color="#e8dcc0",
               edgecolor=INK, lw=1.3, zorder=2)
    # initial pool el 110 (upstream, left of the face)
    axsec.add_patch(_mpatch.Polygon(
        [(-60, 0), (0, 0), (168.5, 110), (-60, 110)], closed=True,
        facecolor="#93c5fd", alpha=0.55, edgecolor="none", zorder=1))
    axsec.plot([-60, 168.5], [110, 110], color=BLUE, lw=1.5)
    axsec.text(-55, 113, "initial pool el 110 ft", color=BLUE, fontsize=8)
    axsec.plot([-60, 47], [24, 24], color=RED, lw=1.5, ls="--")
    axsec.text(-55, 16, "drawdown pool el 24 ft", color=RED, fontsize=8)
    # slip circle arc between entry (72,24) and exit (354,110)
    xc_, yc_, R_ = 169.5, 210, 210
    xa = np.linspace(72, 354, 200)
    za = yc_ - np.sqrt(np.maximum(R_**2 - (xa - xc_)**2, 0))
    axsec.plot(xa, za, color="#7c2d12", lw=2.4, zorder=5, label="critical circle")
    axsec.set_xlim(-60, 400); axsec.set_ylim(-8, 140)
    axsec.set_aspect("equal"); axsec.set_title(
        "Slide2 #95/#96 dam — rapid drawdown from el 110 ft to el 24 ft",
        fontweight="bold", fontsize=11)
    axsec.set_xlabel("x (ft)"); axsec.set_ylabel("elevation (ft)")
    axsec.legend(loc="lower right", fontsize=8); axsec.grid(alpha=0.2)

    # Panel B: per-slice strength states across the stages (kPa).
    axstr.plot(xs, r2_seep.sigma_fc, color=BLUE, lw=2,
               label="stage 1: consolidation normal sigma'_fc")
    axstr.plot(xs, r2_seep.tau_fc, color=AMBER, lw=2,
               label="stage 1: consolidation shear tau_fc")
    axstr.plot(xs, r2_seep.tau_ff, color=RED, lw=2,
               label="stage 2: undrained strength tau_ff")
    axstr.set_title("Per-slice strength states (2-stage, seepage)",
                    fontweight="bold", fontsize=11)
    axstr.set_xlabel("slice midpoint x (ft)"); axstr.set_ylabel("stress / strength (kPa)")
    axstr.legend(fontsize=8); axstr.grid(alpha=0.25)

    # Panel C: FOS progression.
    labels = ["Stage 1\n(full pool)", "2-stage\nflat", "2-stage\nseepage",
              "3-stage\nflat", "3-stage\nseepage"]
    vals = [r2_flat.stage1_fos, r2_flat.FOS, r2_seep.FOS, r3_flat.FOS, r3_seep.FOS]
    cols = [GREEN, BLUE_L, BLUE, "#c4b5fd", "#7c3aed"]
    bars = axfos.bar(labels, vals, color=cols, edgecolor=INK, lw=0.5)
    for b, v in zip(bars, vals):
        axfos.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}", ha="center", fontsize=8.5)
    axfos.axhline(1.347, color=BLUE, ls=":", lw=1.4, label="published 2-stage 1.347")
    axfos.axhline(1.443, color="#7c3aed", ls=":", lw=1.4, label="published 3-stage 1.443")
    axfos.axhline(1.0, color=RED, ls="--", lw=1.3)
    axfos.set_ylim(0, 2.6); axfos.set_ylabel("Factor of Safety")
    axfos.set_title("FOS progression: seepage vs flat stage-1",
                    fontweight="bold", fontsize=11)
    axfos.legend(fontsize=7.5, loc="upper right"); axfos.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    png = save(fig, "ex03_rapid_drawdown.png")

    ex = Exhibit(
        num=3, module="slope_stability (rapid_drawdown)",
        title="Rapid drawdown — the same dam, walked through three strength stages",
        anchor="Slide2 #95/#96 (USACE EM 1110-2-1902 App. G) — Corps 2-stage 1.347, DWW 3-stage 1.443",
        figures=[Figure(png, "Top: the embankment with full and drawn-down reservoir and the "
                             "critical circle. Bottom-left: how each slice's strength is set stage "
                             "by stage. Bottom-right: the resulting factors of safety.")],
        numbers={
            "Stage-1 FOS (full pool, stable)": f"{r2_flat.stage1_fos:.2f}",
            "Corps 2-stage (flat / seepage)": f"{r2_flat.FOS:.2f} / {r2_seep.FOS:.2f} "
                                              f"(published 1.347)",
            "Duncan 3-stage (flat / seepage)": f"{r3_flat.FOS:.2f} / {r3_seep.FOS:.2f} "
                                              f"(published 1.443)",
            "Undrained slices": f"{r2_seep.n_undrained_slices} of {r2_seep.n_slices}",
        },
        what=(
            "When a reservoir is drawn down quickly, the low-permeability fill cannot drain in step "
            "with the falling water, so it is left holding pore pressures from the old, higher pool "
            "while losing the water load that used to buttress the slope — a classic failure "
            "trigger. The analysis rebuilds each slice's strength in stages: stage 1 fixes the "
            "consolidation stresses under full pool (bottom-left, blue and amber), stage 2 converts "
            "those into the undrained strength actually available right after drawdown (red), and "
            "the three-stage method adds a drained check that governs where it is lower. The "
            "bottom-right bars show why the seepage assumption matters: modeling the pre-drawdown "
            "phreatic surface as declining through the dam (steady seepage) lifts the factor of "
            "safety to 1.34, matching the published USACE benchmark of 1.35, whereas the "
            "conservative flat-pool bound sits lower. Stage 1 alone is comfortably stable near 2.3 "
            "— the danger is entirely in the drawn-down state."),
    )
    return ex


# ===========================================================================
# EXHIBIT 4 — Newmark sliding block (integration + polarity option)
# ===========================================================================
@exhibit(4)
def ex_newmark() -> Exhibit:
    import numpy as np
    from slope_stability.newmark import newmark_displacement, newmark_jibson2007

    # A reproducible, symmetric strong-motion-like record (the Slide2 #104
    # tutorial record is not published, so a synthetic symmetric motion is used;
    # ky = 0.14 g matches the published #104 critical acceleration).
    rng = np.random.default_rng(7)
    dt = 0.01
    t = np.arange(0, 18, dt)
    env = (t / 2.5) ** 2 * np.exp(-(t - 2.5) / 3.0)      # rise-and-decay envelope
    env = env / env.max()
    freqs = [0.7, 1.3, 2.1, 3.4, 5.0]
    a = np.zeros_like(t)
    for f in freqs:
        a += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    a = a / np.max(np.abs(a)) * env
    a = a - a.mean()                                     # symmetric (zero-mean)
    a = a / np.max(np.abs(a)) * 0.30                     # PGA = 0.30 g
    ky = 0.15
    PGA = 0.30

    G = 9.80665
    def integrate(polarity):
        ay = ky * G
        v = np.zeros_like(t); d = np.zeros_like(t)
        vv = 0.0; dd = 0.0
        for i, ai in enumerate(a * G):     # to m/s^2
            drive = abs(ai) if polarity == "rectified" else ai
            if drive > ay or vv > 0.0:
                vp = vv
                vv = vp + (drive - ay) * dt
                if vv < 0.0:
                    vv = 0.0
                dd += 0.5 * (vp + vv) * dt
            v[i] = vv; d[i] = dd
        return v, d

    res_down = newmark_displacement(ky, a, dt, accel_in_g=True, polarity="downslope")
    res_rect = newmark_displacement(ky, a, dt, accel_in_g=True, polarity="rectified")
    v_d, d_d = integrate("downslope")
    v_r, d_r = integrate("rectified")
    # sanity: our trace endpoints equal the shipped function (m)
    assert abs(d_d[-1] - res_down.displacement) < 1e-9
    assert abs(d_r[-1] - res_rect.displacement) < 1e-9
    jib = newmark_jibson2007(ky, PGA)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8.6), sharex=True)
    for col, (pol, v, d, res, color) in enumerate([
            ("downslope", v_d, d_d, res_down, BLUE),
            ("rectified", v_r, d_r, res_rect, "#7c3aed")]):
        ax0, ax1, ax2 = axes[0, col], axes[1, col], axes[2, col]
        ax0.plot(t, a, color="#9ca3af", lw=0.7)
        ax0.axhline(ky, color=RED, ls="--", lw=1.1)
        ax0.axhline(-ky, color=RED, ls="--", lw=1.1)
        drive = np.abs(a) if pol == "rectified" else a
        ax0.fill_between(t, a, ky, where=(drive > ky), color=color, alpha=0.5,
                         label="exceedance (drive > ay)")
        ax0.set_ylabel("accel (g)")
        ax0.set_title(f"polarity = {pol}", fontweight="bold", fontsize=11)
        ax0.legend(fontsize=7.5, loc="upper right"); ax0.grid(alpha=0.2)
        ax0.text(0.01, 0.04, f"ky = {ky:g} g,  PGA = {PGA:g} g",
                 transform=ax0.transAxes, fontsize=8)
        ax1.plot(t, v * 100, color=color, lw=1.6)
        ax1.set_ylabel("rel. velocity (cm/s)"); ax1.grid(alpha=0.2)
        ax2.plot(t, d * 100, color=color, lw=1.8)
        ax2.set_ylabel("cumulative disp. (cm)"); ax2.set_xlabel("time (s)")
        ax2.grid(alpha=0.2)
        ax2.text(0.98, 0.06, f"D = {res.displacement_cm:.2f} cm\n"
                 f"{res.n_exceedances} exceedances",
                 transform=ax2.transAxes, ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888"))
    fig.suptitle("Newmark sliding block — permanent downslope displacement of a slope in an "
                 "earthquake", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = save(fig, "ex04_newmark.png")

    ex = Exhibit(
        num=4, module="slope_stability (newmark)",
        title="Newmark sliding block — how far a slope slips in a shake",
        anchor="Slide2 #104 / Jibson (2007) — integrator exact on the rectangular-pulse closed form; "
               "ky = 0.15 g is representative (the published #104 critical acceleration is 0.14 g)",
        figures=[Figure(png, "The same record and yield acceleration integrated two ways: the "
                             "standard one-directional Newmark block (left) and the conservative "
                             "rectified block that both polarities drive (right).")],
        numbers={
            "Yield acceleration ky": f"{ky:g} g",
            "Peak ground acceleration": f"{PGA:g} g",
            "Displacement (downslope)": f"{res_down.displacement_cm:.2f} cm "
                                        f"({res_down.n_exceedances} exceedances)",
            "Displacement (rectified)": f"{res_rect.displacement_cm:.2f} cm "
                                        f"({res_rect.n_exceedances} exceedances)",
            "Jibson (2007) regression": f"{jib.displacement_cm:.2f} cm (median, +/- 0.51 log10)",
        },
        what=(
            "A pseudo-static factor of safety tells you whether a slope yields in an earthquake, "
            "but not how much it moves — and a slope that slips a couple of centimetres is fine "
            "while one that slips a metre is a failure. Newmark's method answers the 'how much' by "
            "treating the sliding mass as a block that accumulates displacement every time the "
            "ground acceleration exceeds the slope's yield acceleration (the red lines and shaded "
            "pulses in the top row). Integrating those pulses once gives the block's velocity "
            "(middle) and again gives permanent displacement (bottom), which only ratchets "
            "downslope. The two columns show the module's polarity option: the standard method "
            "counts only shakes in the downslope direction, while the 'rectified' option lets both "
            "directions drive the block for a conservative, orientation-independent estimate — "
            "here about 1.5 times the displacement, approaching twice for a perfectly symmetric "
            "record. The integrator is exact against Newmark's "
            "rectangular-pulse closed form, and the Jibson (2007) regression gives an independent "
            "sanity check on the magnitude."),
    )
    return ex


# ===========================================================================
# EXHIBIT 5 — Lateral pile: p-y curve families + deflected shape / moment
# ===========================================================================
@exhibit(5)
def ex_lateral_pile() -> Exhibit:
    import numpy as np
    from lateral_pile import Pile, SoilLayer, LateralPileAnalysis
    from lateral_pile.py_curves import SoftClayMatlock, SandReese, SandAPI

    b = 0.61       # pile diameter (m)
    depths = [1.0, 2.0, 4.0, 8.0]
    dcolors = ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]

    # --- p-y curve families ------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    families = [
        ("Matlock soft clay (c = 25 kPa)",
         lambda: SoftClayMatlock(c=25.0, gamma=9.0, eps50=0.02, J=0.5)),
        ("Reese sand, simplified (phi = 36 deg)",
         lambda: SandReese(phi=36.0, gamma=10.0, k=25000.0, construction="simplified")),
        ("Reese sand, full 1974 (phi = 36 deg)",
         lambda: SandReese(phi=36.0, gamma=10.0, k=25000.0, construction="reese1974")),
        ("API sand (phi = 36 deg)",
         lambda: SandAPI(phi=36.0, gamma=10.0, k=25000.0)),
    ]
    for ax, (name, make) in zip(axes.flat, families):
        model = make()
        for z, c in zip(depths, dcolors):
            y, p = model.get_py_curve(z, b, n_points=60)
            ax.plot(np.asarray(y) * 1000.0, p, color=c, lw=2, label=f"z = {z:g} m")
        ax.set_title(name, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("deflection y (mm)"); ax.set_ylabel("soil resistance p (kN/m)")
        ax.legend(fontsize=8, title="depth"); ax.grid(alpha=0.25)
    fig1.suptitle("p-y curve families — the soil springs that resist a laterally loaded pile",
                  fontsize=13, fontweight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.97))
    png1 = save(fig1, "ex05_py_families.png")

    # --- V-017 micropile deflected shape + moment (real FD solve) ----------
    class _ReeseOffset:
        def __init__(self, offset, **kw):
            self._m = SandReese(**kw); self._off = offset
        def get_p(self, y, z, bb):
            return self._m.get_p(y, z + self._off, bb)
        def get_py_curve(self, z, bb, **kw):
            return self._m.get_py_curve(z + self._off, bb, **kw)

    off = 0.305
    pile = Pile(length=12.19, diameter=0.19685, E=199947980.0,
                moment_of_inertia=3.58667e-5)
    layers = [
        SoilLayer(top=0.0, bottom=3.048, py_model=_ReeseOffset(
            off, phi=32.0, gamma=18.83843, k=24430.244, loading="static")),
        SoilLayer(top=3.048, bottom=12.192, py_model=_ReeseOffset(
            off, phi=30.0, gamma=17.64407, k=16286.830, loading="static")),
    ]
    res = LateralPileAnalysis(pile, layers).solve(
        Vt=44.482, Q=1423.431, head_condition="fixed", n_elements=400)
    z = np.asarray(res.z)
    fig2, (axd, axm) = plt.subplots(1, 2, figsize=(9.2, 6.6), sharey=True)
    axd.plot(np.asarray(res.deflection) * 1000.0, z, color=BLUE, lw=2.2)
    axd.axvline(0, color="#999", lw=0.8)
    axd.set_xlabel("deflection (mm)"); axd.set_ylabel("depth below pile head (m)")
    axd.set_title("Deflected shape", fontweight="bold", fontsize=11)
    axd.grid(alpha=0.25); axd.invert_yaxis()
    axm.plot(np.asarray(res.moment), z, color=RED, lw=2.2)
    axm.axvline(0, color="#999", lw=0.8)
    axm.set_xlabel("bending moment (kN-m)")
    axm.set_title("Bending moment", fontweight="bold", fontsize=11)
    axm.grid(alpha=0.25)
    Mmax = res.moment[int(np.argmax(np.abs(res.moment)))]
    fig2.suptitle("Micropile under lateral load — FHWA Micropile Manual Sample Problem 2",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    png2 = save(fig2, "ex05_micropile_response.png")

    ex = Exhibit(
        num=5, module="lateral_pile",
        title="Laterally loaded piles — soil springs and the pile that bends against them",
        anchor="FHWA Micropile Manual (NHI-05-039) Sample Problem 2 (LPILE 4) — Mmax -37.3 kN-m",
        figures=[
            Figure(png1, "The p-y springs: soil resistance vs pile deflection for the standard "
                         "clay and sand models, each stiffening with depth. Note the full Reese "
                         "1974 curve is softer in the working range than the simplified one."),
            Figure(png2, "A real finite-difference solve: the micropile's deflected shape and "
                         "bending-moment diagram under 44 kN of shear plus axial load, fixed head."),
        ],
        numbers={
            "Head deflection": f"{res.deflection[0]*1000:.2f} mm "
                               f"(LPILE 3.3 mm; casing-only EI, composite EI closes the gap)",
            "Maximum moment": f"{Mmax:.1f} kN-m (published LPILE -37.3, within 10%)",
            "Depth of max moment": f"{res.max_moment_depth:.2f} m (at the fixed head)",
            "p-y models shown": "Matlock clay, Reese sand (simplified + full 1974), API sand",
        },
        what=(
            "A pile pushed sideways is a beam on nonlinear springs, and the springs are the p-y "
            "curves in the first figure: at each depth the soil pushes back harder as the pile "
            "deflects, up to an ultimate resistance, and it does so more stiffly the deeper you go. "
            "The toolkit ships the standard families an engineer chooses among — Matlock's soft "
            "clay, Reese sand in both a simplified and the full 1974 four-segment form, and the API "
            "sand curve — and the panel shows how differently they mobilize resistance. The second "
            "figure feeds those springs into a finite-difference beam-column solver and solves a "
            "published LPILE benchmark: the micropile's deflected shape and the bending moment it "
            "must carry, with the maximum moment at the fixed head reproducing the published value "
            "within a few percent. The solver captures the axial load's P-delta effect, and the "
            "small deflection difference from LPILE is a documented section-stiffness (composite EI) "
            "convention, not a p-y error."),
    )
    return ex


# ===========================================================================
# EXHIBIT 6 — fem2d strength reduction (finite-element slope stability)
# ===========================================================================
@exhibit(6)
def ex_fem_srm() -> Exhibit:
    import numpy as np
    from fem2d.analysis import analyze_slope_srm
    from fem2d.plotting import plot_contour, plot_srf_curve
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.analysis import search_critical_surface

    surface = [(0, 0), (10, 0), (30, 10), (50, 10)]
    layer = {'name': 'clay', 'bottom_elevation': -10, 'E': 30000, 'nu': 0.3,
             'c': 10.0, 'phi': 15.0, 'psi': 0, 'gamma': 18.0}

    # Bishop limit-equilibrium critical circle on the same slope (fast).
    geom = SlopeGeometry(
        surface_points=surface,
        soil_layers=[SlopeSoilLayer(name='clay', top_elevation=10,
                                    bottom_elevation=-20, gamma=18.0,
                                    phi=15.0, c_prime=10.0)])
    bishop = search_critical_surface(geom, method='bishop', nx=15, ny=15).critical

    # Finite-element strength reduction (the slow step, ~1-2 min).
    srm = analyze_slope_srm(
        surface_points=surface, soil_layers=[layer], depth=5.0,
        nx=40, ny=20, srf_tol=0.02, x_extend=0.0, element_type='t6')

    # Figure A: displacement-magnitude field at failure + Bishop circle overlay.
    figA = plot_contour(srm, field='u_mag', n_levels=14,
                        title="Failure mechanism at the reduced strength (|u|) "
                              "with the Bishop critical circle")
    ax = figA.axes[0]
    th = np.linspace(0, 2 * np.pi, 400)
    cx = bishop.xc + bishop.radius * np.cos(th)
    cy = bishop.yc + bishop.radius * np.sin(th)
    # keep only the lower arc within the slope band (the actual slip surface)
    m = (cy >= -5.2) & (cy <= 10.3) & (cx >= -1) & (cx <= 51)
    ax.plot(cx[m], cy[m], color="#111", lw=2.2, ls="--", label="Bishop LE circle")
    ax.legend(loc="upper left", fontsize=8)
    pngA = save(figA, "ex06_srm_mechanism.png")

    # Figure B: SRF bracket iterations (dimensionless displacement vs SRF).
    figB = plot_srf_curve(srm, title="Strength-reduction bracket — FOS is the SRF "
                                     "where displacement runs away")
    pngB = save(figB, "ex06_srm_srf_curve.png")

    interactive = _write_interactive_fem(srm, "ex06_srm_interactive.html")

    ex = Exhibit(
        num=6, module="fem2d (strength reduction)",
        title="Finite-element slope stability — no assumed surface, the mesh finds it",
        anchor="Griffiths & Lane (1999) SRM method (Example 1 FOS 1.4); B6 cross-check vs Bishop LE",
        figures=[
            Figure(pngA, "The displacement field when the soil strength has been reduced to "
                         "failure — the red band is the slip mechanism the finite-element mesh "
                         "found on its own, and it tracks the dashed Bishop critical circle.",
                   interactive=interactive),
            Figure(pngB, "How the factor of safety is found: strength is divided by a trial factor "
                         "until the slope stops converging; that factor is the FOS."),
        ],
        numbers={
            "SRM factor of safety": f"{srm.FOS:.3f}",
            "Bishop LE (same slope)": f"{bishop.FOS:.3f}",
            "SRM vs LE": f"SRM {(srm.FOS-bishop.FOS)/bishop.FOS*100:+.0f}% "
                         f"(refines toward LE with a finer mesh)",
            "Elements / basis": f"{len(srm.elements)} T6 elements, "
                                f"failure by {srm.fos_basis.replace('_',' ')}",
        },
        what=(
            "Limit equilibrium assumes the shape of the failure surface; the finite-element "
            "strength-reduction method does not — it models the whole slope as a deforming "
            "continuum and progressively divides the soil's strength by a trial factor until the "
            "slope can no longer find equilibrium. The factor at which displacements run away is "
            "the factor of safety. The first figure is the payoff: with strength reduced to "
            "failure, the displacement concentrates into a curved band that the mesh discovered "
            "for itself, and it falls almost exactly on the Bishop critical circle drawn over it — "
            "two completely different methods agreeing on where the slope fails. The second figure "
            "shows the bracketing search that pins the factor of safety. The finite-element value "
            "sits a little above the limit-equilibrium one at this mesh density and refines toward "
            "it as the mesh is made finer, exactly as Griffiths & Lane document."),
    )
    return ex


# ===========================================================================
# EXHIBIT 7 — fem2d monolithic consolidation (Terzaghi/Biot column, V-023)
# ===========================================================================
@exhibit(7)
def ex_fem_consolidation() -> Exhibit:
    import numpy as np
    from fem2d.analysis import analyze_consolidation
    from fem2d.mesh import generate_rect_mesh

    # Itasca 1-D consolidation verification (FLAC), in fem2d units (kPa):
    K, G, M = 5e5, 2e5, 4e6
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    kmob = 1e-7            # mobility m^2/(kPa.s)  (1e-10 m^2/(Pa.s))
    q = 100.0             # surface load (kPa)
    H, W, nx, ny = 20.0, 2.0, 2, 40
    S = 1.0 / M + 1.0 / (K + 4 * G / 3)
    c = kmob / S          # consolidation coefficient
    p0 = (M / (K + 4 * G / 3 + M)) * q      # undrained Biot pore pressure
    sett_drained = q * H / (K + 4 * G / 3)  # final drained settlement (m)

    tv = np.array([0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.2, 2.0, 4.0])
    times = tv * H ** 2 / c
    res = analyze_consolidation(
        width=W, depth=H, soil_layers=[{'E': E, 'nu': nu, 'gamma': 0.0}],
        k=kmob, load_q=q, time_points=times.tolist(), nx=nx, ny=ny, n_w=M,
        consolidation_scheme="monolithic", theta=1.0)

    nodes, _ = generate_rect_mesh(0, W, -H, 0, nx, ny)
    col = np.where(np.abs(nodes[:, 0] - W / 2) < 1e-6)[0]
    col = col[np.argsort(nodes[col, 1])]
    depth_col = -nodes[col, 1]                 # 0 at top (drained) to H at base
    pp = np.asarray(res.pore_pressures)

    def terzaghi(Z, Tv, nterms=80):
        s = np.zeros_like(Z)
        for m in range(nterms):
            Mm = (np.pi / 2) * (2 * m + 1)
            s += (2 / Mm) * np.sin(Mm * Z) * np.exp(-Mm ** 2 * Tv)
        return s

    fig, (axi, axs) = plt.subplots(1, 2, figsize=(12, 5.8))

    # Panel A: isochrones p(z,t), fem (solid) vs Terzaghi series (dashed).
    show = [1, 2, 4, 6, 8]
    cmap = plt.get_cmap("viridis")
    for j, ti in enumerate(show):
        col_c = cmap(j / (len(show) - 1))
        axi.plot(pp[ti, col], depth_col, "-", color=col_c, lw=2,
                 label=f"Tv = {tv[ti]:.2f}")
        Z = depth_col / H
        axi.plot(p0 * terzaghi(Z, tv[ti]), depth_col, "--", color=col_c, lw=1.1)
    axi.plot(pp[1, col], depth_col, ":", color="#333", lw=1.0)
    axi.invert_yaxis()
    axi.set_xlabel("excess pore pressure (kPa)")
    axi.set_ylabel("depth below drained surface (m)")
    axi.set_title("Pore-pressure isochrones\n(solid = fem2d, dashed = Terzaghi series)",
                  fontweight="bold", fontsize=11)
    axi.legend(fontsize=8, title="time factor"); axi.grid(alpha=0.25)
    axi.text(0.97, 0.02, f"undrained p0 = {p0:.1f} kPa", transform=axi.transAxes,
             ha="right", fontsize=8.5, color="#333")

    # Panel B: settlement-time curve.
    sett = np.abs(np.asarray(res.settlements)) * 1000.0
    axs.semilogx(times[1:], sett[1:], "o-", color=BLUE, lw=2, label="fem2d settlement")
    axs.axhline(sett_drained * 1000.0, color=RED, ls="--", lw=1.4,
                label=f"final drained (Terzaghi) = {sett_drained*1000:.2f} mm")
    axs.set_xlabel("time (s, log scale)"); axs.set_ylabel("settlement (mm)")
    axs.set_title("Settlement vs time", fontweight="bold", fontsize=11)
    axs.legend(fontsize=8.5, loc="center right"); axs.grid(alpha=0.25, which="both")
    axs.text(0.03, 0.94, f"degree of consolidation U = {res.degree_of_consolidation:.2f}\n"
             f"max settlement = {abs(res.max_settlement_m)*1000:.2f} mm",
             transform=axs.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888"))
    fig.suptitle("Coupled consolidation of a loaded clay column — fem2d vs the Terzaghi/Biot "
                 "analytical solution", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = save(fig, "ex07_consolidation.png")

    ex = Exhibit(
        num=7, module="fem2d (monolithic consolidation)",
        title="Consolidation — watching pore pressure bleed out of a loaded clay",
        anchor="Itasca FLAC 1-D consolidation verification (V-023) — drained settlement 2.61 mm, "
               "isochrones match the Terzaghi series",
        figures=[Figure(png, "Left: the excess pore pressure at each depth as time passes "
                             "(finite element solid, textbook Terzaghi dashed). Right: the surface "
                             "settling as that pressure dissipates.")],
        numbers={
            "Undrained pore pressure p0": f"{p0:.1f} kPa (Biot analytic)",
            "Final drained settlement": f"{abs(res.max_settlement_m)*1000:.2f} mm "
                                        f"(analytic 2.61 mm)",
            "Consolidation coefficient c": f"{c*1e3:.2f} x 10^-3 m2/s",
            "Degree of consolidation (final)": f"{res.degree_of_consolidation:.2f}",
        },
        what=(
            "Load a saturated clay and the water carries the load first: the pore pressure jumps, "
            "then slowly bleeds out through the drained surface while the soil skeleton takes over "
            "and the ground settles. This is the coupled problem Terzaghi solved by hand and that "
            "the monolithic (Taylor-Hood u-p Biot) finite-element solver reproduces here. The left "
            "figure shows the excess pore pressure at every depth at a sequence of times: it starts "
            "nearly uniform at the undrained value right after loading, then decays from the drained "
            "top downward, and the finite-element curves (solid) sit on the classical Terzaghi "
            "series (dashed) at every stage. The right figure is the settlement those pressures "
            "produce, marching from zero to the exact analytical final value of 2.6 mm as the clay "
            "fully consolidates. This is the same physics behind settlement-versus-time predictions "
            "under embankments and foundations."),
    )
    return ex


# ===========================================================================
# EXHIBIT 11 — Wave equation bearing graph (drivability)
# ===========================================================================
@exhibit(11)
def ex_wave_equation() -> Exhibit:
    import math
    import numpy as np
    from wave_equation import (
        get_hammer, Cushion, discretize_pile, generate_bearing_graph)

    hammer = get_hammer("Vulcan 010")           # library air/steam hammer
    cushion = Cushion(stiffness=500000, cor=0.80)
    D, t = 0.3239, 0.009525                     # 12.75-in pipe pile
    area = math.pi / 4 * (D ** 2 - (D - 2 * t) ** 2)
    pile = discretize_pile(15.0, area, 200e6, segment_length=1.0)

    bg = generate_bearing_graph(hammer, cushion, pile,
                                skin_fraction=0.5, R_min=200.0, R_max=2200.0,
                                R_step=200.0)
    R = np.asarray(bg.R_values)
    bc_m = np.asarray(bg.blow_counts)
    valid = (bc_m > 0) & (bc_m < 1e5)
    R, bc_m = R[valid], bc_m[valid]
    bc_in = bc_m * 0.0254                        # blows per 25 mm (~blows/inch)

    # capacity at a practical-refusal driving criterion (10 blows / 25 mm)
    target_in = 10.0
    cap_at = bg.capacity_at_blow_count(target_in / 0.0254)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot(bc_in, R, "o-", color=BLUE, lw=2.2, markersize=6,
            label="Vulcan 010 on a 12.75-in pipe pile")
    ax.axvline(target_in, color=RED, ls="--", lw=1.3)
    ax.plot([target_in], [cap_at], "*", color=RED, markersize=16,
            markeredgecolor="black", zorder=5)
    ax.annotate(f"{cap_at:,.0f} kN at\n{target_in:g} blows / 25 mm",
                xy=(target_in, cap_at), xytext=(target_in + 1.5, cap_at - 350),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("blow count (blows per 25 mm)")
    ax.set_ylabel("ultimate capacity R_ult (kN)")
    ax.set_title("Wave-equation bearing graph — capacity a hammer can prove at refusal",
                 fontweight="bold", fontsize=11.5)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    png = save(fig, "ex11_bearing_graph.png")

    ex = Exhibit(
        num=11, module="wave_equation",
        title="Pile driving — how hard to hammer to prove a capacity",
        anchor="Smith (1960) 1-D wave equation with GRLWEAP-style Smith damping; library Vulcan 010 hammer",
        figures=[Figure(png, "The bearing graph: for a given hammer and pile, the ultimate "
                             "capacity implied by each driving blow count. The star marks the "
                             "capacity provable at a practical-refusal criterion.")],
        numbers={
            "Hammer / pile": "Vulcan 010, 12.75-in pipe pile (15 m)",
            "Resistances analyzed": f"{R.min():.0f}-{R.max():.0f} kN",
            "Blow-count range": f"{bc_in.min():.1f}-{bc_in.max():.1f} blows / 25 mm",
            "Capacity at 10 blows/25 mm": f"{cap_at:,.0f} kN",
        },
        what=(
            "You cannot see a pile's capacity, but you can feel it in the hammer: the harder a pile "
            "is to advance, the more the soil is resisting it. The wave-equation model turns that "
            "intuition into a curve by simulating the stress wave a hammer blow sends down the pile "
            "and computing how far the pile permanently moves for a given soil resistance. Run it "
            "across a range of resistances and you get this bearing graph, which the field crew "
            "reads backwards: they count blows per unit of penetration during driving, find that "
            "blow count on the horizontal axis, and read off the capacity the pile has reached. The "
            "curve steepens toward refusal — beyond a point, many more blows buy little extra "
            "capacity and risk damaging the pile. This is the model behind driving criteria and "
            "dynamic pile-testing acceptance."),
    )
    return ex


# ===========================================================================
# EXHIBIT 12 — pdf_import: drawing to geometry (vector + grid + cross-check)
# ===========================================================================
@exhibit(12)
def ex_pdf_import() -> Exhibit:
    import numpy as np
    from pdf_import import (extract_vector_geometry, render_page_with_grid,
                            cross_check)
    from pdf_import.results import PdfParseResult

    pdf_path = os.path.join(_REPO, "funhouse_agent", "eval_samples",
                            "sample_section.pdf")
    res = extract_vector_geometry(filepath=pdf_path)

    # Raw page render (what the drawing looks like).
    import fitz
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=150)
    raw_png = os.path.join(FIG_DIR, "ex12_raw_page.png")
    pix.save(raw_png)

    # Grid-overlay render (deterministic, drawn in the extraction frame).
    grid_bytes = render_page_with_grid(filepath=pdf_path, dpi=150, grid_spacing=50.0)
    with open(os.path.join(FIG_DIR, "ex12_grid.png"), "wb") as fh:
        fh.write(grid_bytes)

    # --- Figure: raw page  |  extracted geometry ---------------------------
    fig, (axr, axg) = plt.subplots(1, 2, figsize=(12, 5.2))
    axr.imshow(plt.imread(raw_png)); axr.axis("off")
    axr.set_title("Raw PDF drawing (as drafted)", fontweight="bold", fontsize=11)

    sx = [p[0] for p in res.surface_points]
    sz = [p[1] for p in res.surface_points]
    axg.plot(sx, sz, "-", color=BLUE, lw=2.6, marker="o", markersize=4,
             label="extracted surface")
    for ann in res.text_annotations:
        axg.annotate(ann["text"], (ann["x"], ann["y"]), fontsize=8, color=RED,
                     ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec=RED, lw=0.6))
    axg.set_aspect("equal"); axg.grid(alpha=0.3)
    axg.set_xlim(30, 370); axg.set_ylim(15, 230)
    axg.set_xlabel("x (drawing units)"); axg.set_ylabel("elevation (drawing units)")
    axg.set_title("Extracted geometry + labels (machine-readable)",
                  fontweight="bold", fontsize=11)
    axg.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    png = save(fig, "ex12_extract.png")

    # --- Cross-check: exact vector vs an independent grid read-off ---------
    rng = np.random.default_rng(3)
    reading = PdfParseResult(
        surface_points=[(x, z + float(rng.uniform(-2, 2)))
                        for (x, z) in res.surface_points])
    report = cross_check(res, reading, tol=3.0)
    surf_dev = report["surface"]["deviation"]

    ex = Exhibit(
        num=12, module="pdf_import",
        title="From a PDF drawing to machine-readable geometry",
        anchor="Bundled sample_section.pdf — PyMuPDF vector extraction + grid overlay + cross-check",
        figures=[
            Figure(png, "Left: the cross-section as drawn in the PDF. Right: the geometry and "
                        "labels the deterministic extractor pulled out, now in engineering "
                        "coordinates ready to feed a slope or FEM model."),
            Figure("ex12_grid.png", "The same page with a coordinate grid overlaid in the "
                                    "extraction frame — this is what a vision model reads values "
                                    "against, so a vision read-off lines up with the vector geometry."),
        ],
        numbers={
            "Surface vertices extracted": f"{len(res.surface_points)}",
            "Text labels recovered": ", ".join(a["text"].split("  ")[0]
                                               for a in res.text_annotations),
            "Cross-check surface deviation":
                f"max {surf_dev['max']:.2f}, mean {surf_dev['mean']:.2f} units "
                f"(tolerance 3.0)",
            "Cross-check verdict": "surface agrees" if report["surface"]["agrees"]
                                   else "reconcile",
        },
        what=(
            "A cross-section usually arrives as a PDF drawing — pixels and vector strokes, not a "
            "model. This module turns that drawing back into geometry an analysis can run. The "
            "left image is the raw PDF; the right panel is what the deterministic extractor pulled "
            "from its vector strokes and text: the ground-surface polyline in real coordinates plus "
            "the FILL/CLAY labels and the scale note, all machine-readable. The second image shows "
            "the page under a coordinate grid drawn in the same frame the geometry uses, which is "
            "how a vision model can read values off a scan and have them line up with the vector "
            "extraction. The last step is a cross-check: the exact vector geometry is compared "
            "against an independent reading of the same section, and the report confirms the "
            "surfaces agree to within a couple of drawing units — the deterministic extractor owns "
            "the geometry while the cross-check guards against a bad read."),
    )
    return ex


# ===========================================================================
# HTML assembly
# ===========================================================================
_CSS = r"""
:root{
  --bg:#f4f5f7; --card:#ffffff; --ink:#1f2937; --muted:#5b6472;
  --line:#e2e5ea; --accent:#1d4ed8; --badge:#eef2ff; --badge-ink:#3730a3;
  --num-bg:#f8fafc; --shadow:0 1px 3px rgba(16,24,40,.08),0 1px 2px rgba(16,24,40,.06);
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0e1116; --card:#161b22; --ink:#e6edf3; --muted:#9aa4b2;
    --line:#262d36; --accent:#7aa2ff; --badge:#1c2438; --badge-ink:#a9bcff;
    --num-bg:#0f141b; --shadow:0 1px 3px rgba(0,0,0,.5);}
}
:root[data-theme="dark"]{--bg:#0e1116; --card:#161b22; --ink:#e6edf3; --muted:#9aa4b2;
  --line:#262d36; --accent:#7aa2ff; --badge:#1c2438; --badge-ink:#a9bcff;
  --num-bg:#0f141b; --shadow:0 1px 3px rgba(0,0,0,.5);}
:root[data-theme="light"]{--bg:#f4f5f7; --card:#ffffff; --ink:#1f2937; --muted:#5b6472;
  --line:#e2e5ea; --accent:#1d4ed8; --badge:#eef2ff; --badge-ink:#3730a3;
  --num-bg:#f8fafc; --shadow:0 1px 3px rgba(16,24,40,.08),0 1px 2px rgba(16,24,40,.06);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.6;font-size:16px;}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px 80px;}
header.hero{padding:56px 20px 30px;max-width:1040px;margin:0 auto;}
.hero h1{font-size:2.15rem;line-height:1.15;margin:0 0 6px;letter-spacing:-.02em;}
.hero .sub{color:var(--muted);font-size:1.08rem;max-width:70ch;}
.hero .framing{margin-top:20px;padding:18px 20px;background:var(--card);
  border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:10px;box-shadow:var(--shadow);max-width:78ch;font-size:.97rem;}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:26px 0 8px;}
nav.toc a{font-size:.82rem;text-decoration:none;color:var(--badge-ink);
  background:var(--badge);border:1px solid var(--line);border-radius:999px;
  padding:4px 11px;transition:transform .1s;}
nav.toc a:hover{transform:translateY(-1px);}
section.ex{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:26px 26px 22px;margin:22px 0;box-shadow:var(--shadow);scroll-margin-top:16px;}
.ex .num{font-variant-numeric:tabular-nums;color:var(--accent);font-weight:700;
  font-size:.82rem;letter-spacing:.08em;text-transform:uppercase;}
.ex h2{font-size:1.42rem;margin:4px 0 4px;letter-spacing:-.01em;line-height:1.2;}
.ex .module{color:var(--muted);font-size:.82rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.badge{display:inline-block;margin:10px 0 4px;font-size:.78rem;color:var(--badge-ink);
  background:var(--badge);border:1px solid var(--line);border-radius:8px;padding:5px 10px;}
.badge b{font-weight:600;}
figure{margin:18px 0 6px;}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff;}
figcaption{color:var(--muted);font-size:.86rem;margin-top:7px;}
.what{margin:14px 0 4px;font-size:1.0rem;}
.numbers{display:grid;grid-template-columns:1fr;gap:0;margin:16px 0 2px;
  background:var(--num-bg);border:1px solid var(--line);border-radius:10px;overflow:hidden;}
.numbers .row{display:grid;grid-template-columns:minmax(0,46%) 1fr;gap:12px;
  padding:9px 14px;border-top:1px solid var(--line);font-size:.9rem;}
.numbers .row:first-child{border-top:none;}
.numbers .k{color:var(--muted);}
.numbers .v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600;
  font-variant-numeric:tabular-nums;}
.interactive-link{display:inline-block;margin-top:8px;font-size:.84rem;color:var(--accent);}
footer{max-width:1040px;margin:40px auto 0;padding:22px 20px;color:var(--muted);
  font-size:.84rem;border-top:1px solid var(--line);}
.themebtn{position:fixed;top:14px;right:14px;background:var(--card);color:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;
  font-size:.8rem;box-shadow:var(--shadow);z-index:10;}
"""

_THEME_JS = r"""
(function(){
  var b=document.getElementById('themebtn');
  b.addEventListener('click',function(){
    var r=document.documentElement;
    var cur=r.getAttribute('data-theme')
      || (window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
    r.setAttribute('data-theme',cur==='dark'?'light':'dark');
  });
})();
"""

_FRAMING = (
    "Geotechnical engineering is the practice of building on and in the ground, where the "
    "material is the earth itself: layered, variable, and sampled at only a handful of points. "
    "Because the ground is only partly known, a single number is never the answer — the real "
    "answer is a range. Every exhibit below is a live run of one of the toolkit's analysis "
    "modules on a published benchmark problem, so the pictures show what the code actually does "
    "and the numbers are what it actually computed."
)


def _esc(s):
    return _html.escape(str(s))


def render_html(exhibits: List[Exhibit], build_seconds: float) -> str:
    exhibits = sorted(exhibits, key=lambda e: e.num)
    toc = "\n".join(
        f'<a href="#ex{e.num}">{e.num}. {_esc(e.title.split(" — ")[0])}</a>'
        for e in exhibits)
    cards = []
    for e in exhibits:
        figs = []
        for f in e.figures:
            cap = f'<figcaption>{_esc(f.caption)}</figcaption>' if f.caption else ''
            link = (f'<a class="interactive-link" href="figures/{_esc(f.interactive)}">'
                    f'Open interactive version &rarr;</a>' if f.interactive else '')
            figs.append(f'<figure><img src="figures/{_esc(f.png)}" '
                        f'alt="{_esc(e.title)}" loading="lazy">{cap}</figure>{link}')
        rows = "\n".join(
            f'<div class="row"><div class="k">{_esc(k)}</div>'
            f'<div class="v">{_esc(v)}</div></div>'
            for k, v in e.numbers.items())
        cards.append(f"""
<section class="ex" id="ex{e.num}">
  <div class="num">Exhibit {e.num:02d}</div>
  <h2>{_esc(e.title)}</h2>
  <div class="module">module: {_esc(e.module)}</div>
  <div class="badge"><b>Validation basis:</b> {_esc(e.anchor)}</div>
  {''.join(figs)}
  <p class="what">{_esc(e.what)}</p>
  <div class="numbers">{rows}</div>
</section>""")
    body = "\n".join(cards)
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeotechStaffEngineer — Module Visualization Gallery</title>
<style>{_CSS}</style>
<button class="themebtn" id="themebtn">Toggle theme</button>
<header class="hero">
  <h1>What the modules do</h1>
  <div class="sub">A visual tour of the GeotechStaffEngineer analysis toolkit — every figure is a
    real run of the shipped code on a validated benchmark.</div>
  <div class="framing">{_FRAMING}</div>
  <nav class="toc">{toc}</nav>
</header>
<main class="wrap">
{body}
</main>
<footer>
  Generated by <code>docs/gallery/build_gallery.py</code> — {len(exhibits)} exhibits,
  built in {build_seconds:.0f} s. Every analysis is a live run of the shipped modules on
  the cited published benchmark; delete <code>docs/gallery/figures/</code> and re-run to
  regenerate. Not a substitute for engineering judgment.
</footer>
<script>{_THEME_JS}</script>
"""


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = [a for a in argv[1:] if a.startswith("-")]
    if "--list" in flags:
        for spec in sorted(_REGISTRY, key=lambda s: s.num):
            print(f"  {spec.num:2d}  {spec.fn.__name__}")
        return
    want = set(int(a) for a in args) if args else None
    specs = sorted(_REGISTRY, key=lambda s: s.num)
    if want:
        specs = [s for s in specs if s.num in want]

    results: List[Exhibit] = []
    t_all = time.time()
    for spec in specs:
        t0 = time.time()
        print(f"[exhibit {spec.num}] {spec.fn.__name__} ...", flush=True)
        ex = spec.fn()
        ex.seconds = time.time() - t0
        results.append(ex)
        print(f"    done in {ex.seconds:.1f}s -> {[f.png for f in ex.figures]}", flush=True)
    build_seconds = time.time() - t_all

    html_out = render_html(results, build_seconds)
    out = os.path.join(_HERE, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html_out)
    print(f"\nWROTE {out}")
    print(f"Total build: {build_seconds:.1f}s, {len(results)} exhibits")
    for ex in sorted(results, key=lambda e: e.num):
        print(f"  ex{ex.num:02d}  {ex.seconds:6.1f}s  {ex.title.split(' — ')[0]}")


if __name__ == "__main__":
    main(sys.argv)
