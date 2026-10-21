"""
Figures for a conditional simulation: the exceedance curve of the totals and
a plan map of the probability that the field is above (below) a level.

Each figure is drawn TWICE from the same arrays: a matplotlib PNG (the report
copy) and a Plotly figure (the interactive twin the app renders from a
``.plotly.json`` sidecar), the way ``profile_figure.plot_data`` does it.
matplotlib is used through its object API (``Figure`` + the Agg canvas), so
no global pyplot state is touched. The map is drawn by the module's one
plan-map path, :mod:`gstools_agent.plan_map`, which the kriging map shares.
"""

import io

import numpy as np

#: A raster map up to this many lattice cells; beyond, cells as markers.
#: ``_what`` names z in words (owner, 2026-10-15: "need to define z").
from gstools_agent.plan_map import MAX_RASTER as _MAX_RASTER, _what

_TEXT = "#222222"
_GRID = "#dddddd"
_PALETTE = ["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2", "#e08a1e",
            "#3b8ea5", "#7f7f7f", "#a05d2c"]
#: Curves drawn on one exceedance figure; the rest are listed in the JSON.
MAX_CURVES = 12


def _new_figure(width_in, height_in, dpi):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def _png(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    return buf.getvalue()


def _fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.1f}"
    return f"{v:.3g}"


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=_TEXT, fontsize=11)
    ax.set_xlabel(xlabel, color=_TEXT)
    ax.set_ylabel(ylabel, color=_TEXT)
    ax.tick_params(colors=_TEXT)
    for s in ax.spines.values():
        s.set_color("#888888")


def _plotly_layout(fig, title, xlabel, ylabel, legend_below=True):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font={"color": _TEXT},
        title={"text": title, "font": {"size": 15, "color": _TEXT}},
        legend=({"orientation": "h", "x": 0.0, "y": -0.2,
                 "yanchor": "top", "font": {"color": _TEXT}}
                if legend_below else {"font": {"color": _TEXT}}),
        margin={"l": 70, "r": 30, "t": 80, "b": 110},
    )
    axis = {"showgrid": True, "gridcolor": _GRID, "zeroline": False,
            "showline": True, "linecolor": "#888888",
            "title_font": {"color": _TEXT}, "tickfont": {"color": _TEXT}}
    fig.update_xaxes(title_text=xlabel, **axis)
    fig.update_yaxes(title_text=ylabel, **axis)


# ---------------------------------------------------------------------------
# (a) Exceedance curve
# ---------------------------------------------------------------------------

def exceedance_curves(result, max_curves=MAX_CURVES):
    """The arrays both renderings draw: per region/level, totals at 301
    quantiles and their exceedance probability, plus P50 / P90."""
    q = np.linspace(0.0, 1.0, 301)
    out = []
    for name, lv in result.rl_pairs[:max_curves]:
        v = result.totals(name, lv, "volume")
        xs = np.quantile(v, q)
        ps = 1.0 - q
        if xs[0] > 0:                 # P(V > v) = 1 below the smallest total
            xs = np.concatenate([[0.0], xs])
            ps = np.concatenate([[1.0], ps])
        out.append({"label": f"{name} @ {lv:g}", "x": xs, "p": ps,
                    "P50": float(np.percentile(v, 50)),
                    "P90": float(np.percentile(v, 90)),
                    "p_any": float(np.mean(v > 0))})
    return out


def exceedance_figure(result, dpi=150, width_in=7.0, height_in=4.8,
                      value_name=None, value_unit=""):
    """Exceedance curve P(V > v) per region/level, P50 and P90 marked.

    Returns ``(png_bytes, plotly_figure_or_None, n_curves_drawn)``."""
    q = result.quantity
    curves = exceedance_curves(result)
    xlabel = (f"Volume where the {_what(value_name)} is {q} the level, "
              f"v (m³)")
    ylabel = "Probability of exceedance, P(V > v)"
    title = (f"Exceedance curve of the volume {q} the level "
             f"({result.n_kept:,} realizations)")

    fig = _new_figure(width_in, height_in, dpi)
    ax = fig.add_subplot(111)
    for i, c in enumerate(curves):
        col = _PALETTE[i % len(_PALETTE)]
        lab = (f"{c['label']}: P50 {_fmt(c['P50'])}, "
               f"P90 {_fmt(c['P90'])} m³")
        ax.plot(c["x"], c["p"], color=col, lw=1.6, label=lab)
        ax.plot([c["P50"], c["P90"]], [0.5, 0.1], "o", color=col, ms=5,
                mec="white", mew=0.8)
        if len(curves) <= 3:
            for val, pp, tag in ((c["P50"], 0.5, "P50"),
                                 (c["P90"], 0.1, "P90")):
                ax.annotate(f"{tag} {_fmt(val)}", (val, pp),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=8, color=_TEXT)
    ax.axhline(0.5, color="#999999", lw=0.7, ls=":")
    ax.axhline(0.1, color="#999999", lw=0.7, ls=":")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0.0)
    ax.grid(True, color=_GRID, lw=0.7)
    _style_axes(ax, title, xlabel, ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
              frameon=False, fontsize=8, labelcolor=_TEXT)
    png = _png(fig, dpi)

    pfig = None
    try:
        import plotly.graph_objects as go
    except Exception:                                  # noqa: BLE001
        go = None
    if go is not None:
        pfig = go.Figure()
        for i, c in enumerate(curves):
            col = _PALETTE[i % len(_PALETTE)]
            pfig.add_trace(go.Scatter(
                x=c["x"].tolist(), y=c["p"].tolist(), mode="lines",
                name=c["label"], line={"color": col, "width": 1.8},
                hovertemplate=("v %{x:,.1f} m³<br>P(V > v) %{y:.3f}"
                               f"<extra>{c['label']}</extra>")))
            pfig.add_trace(go.Scatter(
                x=[c["P50"], c["P90"]], y=[0.5, 0.1], mode="markers+text",
                text=[f"P50 {_fmt(c['P50'])}", f"P90 {_fmt(c['P90'])}"],
                textposition="top right", showlegend=False,
                marker={"color": col, "size": 9,
                        "line": {"color": "white", "width": 1}},
                textfont={"color": _TEXT, "size": 11},
                hovertemplate=("%{text} m³"
                               f"<extra>{c['label']}</extra>")))
        for yy in (0.5, 0.1):
            pfig.add_hline(y=yy, line_color="#999999", line_width=0.8,
                           line_dash="dot")
        _plotly_layout(pfig, title, xlabel, ylabel)
        pfig.update_yaxes(range=[0, 1.02])
        pfig.update_xaxes(rangemode="tozero")
    return png, pfig, len(curves)


# ---------------------------------------------------------------------------
# (b) Plan map of P(z > level)
# ---------------------------------------------------------------------------

def _map_points(result, region=None, level=None):
    names = [g["name"] for g in result.regions_geom]
    region = region if region is not None else names[0]
    ri = names.index(region)
    level = (float(level) if level is not None
             else float(result.regions_geom[ri]["levels"][0]))
    li = result.levels_all.index(level)
    p_cells = result.exceed_counts[li] / float(result.n_kept)
    return region, level, p_cells


def probability_map_arrays(result, region=None, level=None, outlines=()):
    """What both renderings draw: the probability at every simulated cell
    (all regions, the chosen region's level) on the lattice, and the view:
    the simulated cells plus a margin and the ``outlines`` (parsed, see
    :func:`gstools_agent.plan_map.parse_outlines`), widened to take in the
    observations and bounds that are near (within one view diagonal)."""
    from gstools_agent.plan_map import plan_view

    region, level, p_cells = _map_points(result, region, level)
    lat = result.lattice
    ny = int(lat["ny"])
    cell = float(result.cell_size)
    idx = np.flatnonzero(result.lattice_union)
    ii, jj = idx // ny, idx % ny
    i0, i1, j0, j1 = ii.min(), ii.max(), jj.min(), jj.max()
    gx = lat["x_min"] + (np.arange(i0, i1 + 1) + 0.5) * cell
    gy = lat["y_min"] + (np.arange(j0, j1 + 1) + 0.5) * cell
    raster = None
    if gx.size * gy.size <= _MAX_RASTER:
        raster = np.full((gy.size, gx.size), np.nan)
        raster[jj - j0, ii - i0] = p_cells
    core_x = [gx[0] - cell / 2, gx[-1] + cell / 2]
    core_y = [gy[0] - cell / 2, gy[-1] + cell / 2]
    for o in outlines:
        core_x += o["xy"][:, 0].tolist()
        core_y += o["xy"][:, 1].tolist()
    bnds = list(result.bounds_input or [])
    xr, yr, near, bnear = plan_view(
        core_x, core_y, max(float(lat.get("margin", 0.0)), 2 * cell),
        result.obs_x, result.obs_y, [b["x"] for b in bnds],
        [b["y"] for b in bnds])
    return {
        "region": region, "level": level, "gx": gx, "gy": gy,
        "raster": raster, "cells_x": result.cell_x, "cells_y": result.cell_y,
        "p_cells": p_cells, "cell": cell, "xlim": xr, "ylim": yr,
        "obs_mask": near, "n_obs_off": int((~near).sum()),
        "bounds": [b for b, k in zip(bnds, bnear) if k],
    }


def probability_map_figure(result, region=None, level=None, dpi=150,
                           width_in=7.0, value_name=None, value_unit="",
                           north_deg=None, outlines=None):
    """Plan map of the probability that the value (``value_name``: what z
    is, e.g. 'top of hard rock elevation') is above (or below) the level at
    each simulated cell, drawn as a plan (owner, 2026-10-15) through the
    module's one plan-map path (:mod:`gstools_agent.plan_map`): to scale,
    no axes or grid, a scale bar and a north arrow (``north_deg`` clockwise
    from up, as the source plan shows it; up when not given), with the
    region outlines, the ``outlines`` given (a property line drawn
    dash-dot, plan outlines), the named observations and the bounds.
    ``width_in`` is kept for callers; the plan is 7 in wide. Returns
    ``(png_bytes, plotly_figure_or_None, meta)``."""
    from gstools_agent.plan_map import parse_outlines, plan_map_figure

    extra = parse_outlines(outlines)
    a = probability_map_arrays(result, region, level, extra)
    word = "above" if result.quantity == "above" else "below"
    rel = ">" if result.quantity == "above" else "<"
    what = _what(value_name)
    unit = str(value_unit or "").strip()
    lev = f"{a['level']:g}" + (f" {unit}" if unit else "")
    sub = (f", {a['n_obs_off']} observation(s) beyond the map"
           if a["n_obs_off"] else "")
    title = (f"Probability that the {what} is {word} {lev}\n"
             f"{a['region']}, {result.n_kept:,} realizations{sub}")
    panel = {"title": title, "kind": "raster", "gx": a["gx"], "gy": a["gy"],
             "z": a["raster"], "cell": a["cell"], "cells_x": a["cells_x"],
             "cells_y": a["cells_y"], "values": a["p_cells"],
             "cmap": "YlOrRd", "vmin": 0.0, "vmax": 1.0,
             "cbar_label": "Probability", "hover": f"P({what} {rel} {lev})"}
    outl = [{"name": g["name"], "xy": np.asarray(g["polygon"], float),
             "closed": True, "kind": "region",
             "main": g["name"] == a["region"]}
            for g in result.regions_geom] + extra
    png, pfig, north_note = plan_map_figure(
        [panel], view=(a["xlim"], a["ylim"]), title=title,
        observations={"x": result.obs_x, "y": result.obs_y,
                      "names": list(result.obs_names),
                      "values": result.obs_values, "mask": a["obs_mask"],
                      "value_label": what, "unit": unit},
        outlines=outl, bounds=a["bounds"], north_deg=north_deg, dpi=dpi)
    meta = {"region": a["region"], "level": a["level"],
            "observations_beyond_map": a["n_obs_off"],
            "value_name": what, "north_note": north_note}
    return png, pfig, meta
