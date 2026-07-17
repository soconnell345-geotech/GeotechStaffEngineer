"""Plots for pavement_design results (matplotlib -> calc-package figures).

The design charts re-plot the guide's nomograph relationships from the
DIGITIZED equations (exact, not chart traces) in the guide's own axes, and
overlay the design solution — i.e. a computed Figure 3.1 / 3.7 with the
design point marked, which is what a reviewer wants to see instead of a
hand-traced nomograph. All figures return matplotlib Figure objects;
callers own closing them. matplotlib import is deferred so the module
imports cleanly without it.

UNITS: US customary, matching the module.
"""

import math

from geotech_references.aashto_1993 import equations as _eq

try:
    from geotech_references.aashto_1993 import environmental as _env_mod
except ImportError:  # pragma: no cover
    _env_mod = None


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_flexible_design_chart(result):
    """Computed Figure 3.1: W18 capacity vs SN for each design foundation
    modulus, with the required-SN solution points and the provided SN.

    ``result`` : FlexiblePavementResult
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sn_grid = [0.5 + 0.05 * i for i in range(int((10.0 - 0.5) / 0.05) + 1)]
    for row in result.sn_stack:
        mr = row["foundation_mr_psi"]
        w18s = []
        for sn in sn_grid:
            try:
                w18s.append(_eq.flexible_w18_from_sn(
                    sn, result.zr, result.so, result.delta_psi, mr)["w18"])
            except ValueError:
                w18s.append(float("nan"))
        ax.plot(w18s, sn_grid, lw=1.4,
                label=f"over {row['over']} (MR = {mr:,.0f} psi)")
        ax.plot([result.w18], [row["sn_required"]], "o", ms=6, color="k",
                zorder=5)
        ax.annotate(f"SN = {row['sn_required']:.2f}",
                    (result.w18, row["sn_required"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.axvline(result.w18, color="0.4", ls=":", lw=1)
    ax.axhline(result.sn_provided, color="tab:green", ls="--", lw=1.2,
               label=f"SN provided = {result.sn_provided:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Design ESALs, W18 (18-kip, log scale)")
    ax.set_ylabel("Structural Number, SN")
    ax.set_title("AASHTO 1993 flexible design equation (computed Figure 3.1)\n"
                 f"ZR = {result.zr}, So = {result.so}, "
                 f"ΔPSI = {result.delta_psi}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(result.w18 / 300.0, result.w18 * 300.0)
    fig.tight_layout()
    return fig


def plot_rigid_design_chart(result):
    """Computed Figure 3.7: W18 capacity vs slab thickness D, with the
    required and provided D marked.

    ``result`` : RigidPavementResult
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    d_grid = [4.0 + 0.1 * i for i in range(int((16.0 - 4.0) / 0.1) + 1)]
    w18s = []
    for d in d_grid:
        try:
            w18s.append(_eq.rigid_w18_from_d(
                d, result.zr, result.so, result.delta_psi, result.sc_psi,
                result.cd, result.j, result.ec_psi, result.k_pci,
                pt=result.pt)["w18"])
        except ValueError:
            w18s.append(float("nan"))
    ax.plot(w18s, d_grid, lw=1.6, color="tab:blue",
            label=f"k = {result.k_pci:,.0f} pci, J = {result.j}, "
                  f"Cd = {result.cd}")
    ax.plot([result.w18], [result.d_required_in], "o", ms=7, color="k",
            zorder=5)
    ax.annotate(f"D required = {result.d_required_in:.2f} in",
                (result.w18, result.d_required_in),
                textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.axvline(result.w18, color="0.4", ls=":", lw=1)
    ax.axhline(result.d_provided_in, color="tab:green", ls="--", lw=1.2,
               label=f"D provided = {result.d_provided_in} in")
    ax.set_xscale("log")
    ax.set_xlabel("Design ESALs, W18 (18-kip, log scale)")
    ax.set_ylabel("Slab thickness, D (in)")
    ax.set_title("AASHTO 1993 rigid design equation (computed Figure 3.7)\n"
                 f"ZR = {result.zr}, So = {result.so}, "
                 f"ΔPSI = {result.delta_psi}, S'c = {result.sc_psi:,.0f} "
                 f"psi, Ec = {result.ec_psi:,.0f} psi")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(result.w18 / 300.0, result.w18 * 300.0)
    fig.tight_layout()
    return fig


_LAYER_COLORS = {
    "asphalt": "0.25",
    "granular_base": "#c8a165",
    "granular_subbase": "#e0cba8",
    "cement_treated_base": "#9fb4c7",
    "bituminous_treated_base": "#7a6a5f",
}


def plot_layer_section(result):
    """Cross-section bar diagram of the flexible section (to-scale depths,
    per-layer a / m / D annotations, roadbed below).

    ``result`` : FlexiblePavementResult
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    top = 0.0
    total = sum(lay.get("thickness_in") or 0 for lay in result.layers)
    for lay in result.layers:
        d = lay.get("thickness_in") or 0
        color = _LAYER_COLORS.get(lay["layer_type"], "0.7")
        ax.bar(0.5, -d, width=0.75, bottom=-top, color=color,
               edgecolor="k", lw=0.8)
        label = (f"{lay['layer_type'].replace('_', ' ')}  "
                 f"D = {d:g} in  (a = {lay['a']:g}"
                 + (f", m = {lay['m']:g}" if lay['m'] != 1.0 else "") + ")")
        ax.text(0.92, -(top + d / 2), label, va="center", fontsize=9)
        top += d
    # Roadbed block.
    ax.bar(0.5, -0.35 * max(total, 1), width=0.75, bottom=-top,
           color="#b7a189", edgecolor="k", lw=0.8, hatch="//")
    ax.text(0.92, -(top + 0.18 * max(total, 1)),
            f"roadbed soil  MR = {result.effective_mr_psi:,.0f} psi",
            va="center", fontsize=9)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-(top + 0.45 * max(total, 1)), 1.0)
    ax.axis("off")
    ax.set_title(
        f"Designed section — SN provided = {result.sn_provided:.2f} "
        f"(required {result.sn_required:.2f})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_seasonal_mr(result):
    """Seasonal roadbed MR and relative damage uf (Figure 2.3/2.4 style),
    with the effective MR line. Requires ``result.effective_mr_detail``.
    """
    detail = result.effective_mr_detail
    if not detail:
        return None
    plt = _plt()
    months = list(range(1, len(detail["monthly_mr_psi"]) + 1))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.step(months, detail["monthly_mr_psi"], where="mid", lw=1.6,
             color="tab:blue", label="seasonal MR")
    ax1.axhline(result.effective_mr_psi, color="tab:red", ls="--", lw=1.4,
                label=f"effective MR = {result.effective_mr_psi:,.0f} psi")
    ax1.set_xlabel("Seasonal increment (month)")
    ax1.set_ylabel("Roadbed resilient modulus, MR (psi)")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.bar(months, detail["uf_values"], alpha=0.25, color="tab:gray",
            label="relative damage uf")
    ax2.set_ylabel("Relative damage, uf")
    ax1.set_title("Effective roadbed MR from seasonal relative damage "
                  "(Figure 2.3/2.4 procedure)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ufc_flexible_design_chart(ufc_result):
    """Computed Figure E-1: thickness vs CBR curves at pass levels around
    the design traffic, with the design point overlaid.

    ``ufc_result`` : dict from ``design_flexible_pavement_ufc``.
    """
    from geotech_references.ufc_pavement import tables as utb
    plt = _plt()
    passes = ufc_result["passes_18kip"]
    sec = ufc_result["section"]
    cbr_design = sec["design_cbr_subgrade"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cbrs = [1.5 + 0.25 * i for i in range(int((40 - 1.5) / 0.25) + 1)]
    for p_lvl, style in ((passes / 10, ":"), (passes, "-"),
                         (passes * 10, "--")):
        ts = []
        for c in cbrs:
            try:
                ts.append(utb.figure_e1_flexible_thickness(
                    cbr=c, passes=p_lvl)["thickness_in"])
            except ValueError:
                ts.append(float("nan"))
        ax.plot(cbrs, ts, style, lw=1.4 if p_lvl == passes else 1.0,
                label=f"{p_lvl:,.0f} passes"
                      + (" (design)" if p_lvl == passes else ""))
    ax.plot([cbr_design], [sec["required_total_thickness_in"]], "o", ms=7,
            color="k", zorder=5)
    ax.annotate(
        f"CBR {cbr_design}, t = {sec['required_total_thickness_in']} in",
        (cbr_design, sec["required_total_thickness_in"]),
        textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(sec["provided_total_thickness_in"], color="tab:green",
               ls="--", lw=1.2,
               label=f"provided = {sec['provided_total_thickness_in']} in")
    ax.set_xscale("log")
    ax.set_xlabel("CBR (log scale)")
    ax.set_ylabel("Total thickness above layer (in)")
    ax.set_title("UFC 3-250-01 flexible design curve "
                 "(computed Figure E-1, 18-kip single axle)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_ufc_rigid_design_chart(ufc_result):
    """Computed Figure F-1 slice: slab thickness vs passes at the design
    flexural strength and k, with the design point overlaid.

    ``ufc_result`` : dict from ``design_rigid_pavement_ufc``.
    """
    from geotech_references.ufc_pavement import tables as utb
    f1 = getattr(utb, "figure_f1_rigid_thickness", None)
    if f1 is None:
        return None
    plt = _plt()
    passes = ufc_result["passes_18kip"]
    r_psi = ufc_result["flexural_strength_psi"]
    k = ufc_result["k_pci"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    import math as _m
    pgrid = [10 ** (3 + 0.15 * i)
             for i in range(int((7.7 - 3) / 0.15) + 1)]
    ts = []
    for p in pgrid:
        try:
            ts.append(f1(flexural_psi=r_psi, k_psi_in=k,
                         passes=p)["thickness_in"])
        except ValueError:
            ts.append(float("nan"))
    ax.plot(pgrid, ts, lw=1.6,
            label=f"R = {r_psi:,.0f} psi, k = {k:,.0f} pci")
    ax.plot([passes], [ufc_result["hd_required_in"]], "o", ms=7, color="k",
            zorder=5)
    ax.annotate(f"hd = {ufc_result['hd_required_in']} in",
                (passes, ufc_result["hd_required_in"]),
                textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.axhline(ufc_result["slab_provided_in"], color="tab:green", ls="--",
               lw=1.2,
               label=f"slab provided = {ufc_result['slab_provided_in']} in")
    ax.set_xscale("log")
    ax.set_xlabel("18-kip single-axle passes (log scale)")
    ax.set_ylabel("Slab thickness (in)")
    ax.set_title("UFC 3-250-01 rigid design (computed Figure F-1 slice)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_method_comparison(compare_result):
    """Side-by-side layer bars: AASHTO 1993 vs UFC 3-250-01 sections.

    ``compare_result`` : dict from ``compare_flexible_pavement_methods``.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = ["0.25", "#c8a165", "#e0cba8", "#9fb4c7"]
    for x, (label, layers) in enumerate((
            ("AASHTO 1993", compare_result["aashto_1993"]["layers"]),
            ("UFC 3-250-01", compare_result["ufc_3_250_01"]["layers"]))):
        top = 0.0
        for i, lay in enumerate(layers):
            d = lay["thickness_in"]
            ax.bar(x, -d, width=0.55, bottom=-top,
                   color=colors[i % len(colors)], edgecolor="k", lw=0.8)
            ax.text(x, -(top + d / 2),
                    f"{lay['layer'].replace('_', ' ')}\n{d:g} in",
                    ha="center", va="center", fontsize=8, color="w"
                    if i == 0 else "k")
            top += d
        ax.text(x, -(top + 1.0), f"{label}\ntotal {top:g} in", ha="center",
                va="top", fontsize=9)
    ax.set_xlim(-0.6, 1.6)
    ax.margins(y=0.15)
    ax.set_xticks([])
    ax.set_ylabel("Depth below surface (in)")
    ax.set_title("Method comparison — same 18-kip traffic, two design bases")
    fig.tight_layout()
    return fig


def plot_environmental_loss(result):
    """Environmental serviceability loss vs time (Figure 2.2 style):
    swelling / frost / total curves, the analysis period, and the design
    dPSI budget. Requires ``result.environmental`` (with the input specs).
    """
    env = getattr(result, "environmental", None)
    if not env or _env_mod is None:
        return None
    swelling = env.get("swelling_spec")
    frost = env.get("frost_spec")
    if not swelling and not frost:
        return None
    t_design = env["design_period_yr"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    n = 60
    t_max = 1.4 * t_design
    ts = [t_max * i / n for i in range(1, n + 1)]
    sw, fh, tot = [], [], []
    for t in ts:
        r = _env_mod.total_environmental_loss(t, swelling=swelling,
                                              frost=frost)
        sw.append(r.get("delta_psi_sw", 0.0))
        fh.append(r.get("delta_psi_fh", 0.0))
        tot.append(r["delta_psi_total"])
    if swelling:
        ax.plot(ts, sw, lw=1.4, color="tab:brown",
                label="swelling ΔPSI_SW (Fig G.4)")
    if frost:
        ax.plot(ts, fh, lw=1.4, color="tab:cyan",
                label="frost heave ΔPSI_FH (Fig G.8)")
    ax.plot(ts, tot, lw=1.8, color="k", label="total environmental loss")
    ax.axvline(t_design, color="0.4", ls=":", lw=1.2)
    ax.annotate(f"analysis period = {t_design:g} yr\n"
                f"ΔPSI_env = {env['delta_psi_total']:g}",
                (t_design, env["delta_psi_total"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
    budget = env.get("delta_psi_design")
    if budget:
        ax.axhline(budget, color="tab:red", ls="--", lw=1.2,
                   label=f"design ΔPSI budget = {budget:g}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Serviceability loss, ΔPSI")
    ax.set_title("Environmental serviceability loss vs time "
                 "(computed Figure 2.2)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, t_max)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig
