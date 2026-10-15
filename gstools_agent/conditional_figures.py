"""
Figures for a conditional simulation: the exceedance curve of the totals and
a plan map of the probability that the field is above (below) a level.

Each figure is drawn TWICE from the same arrays: a matplotlib PNG (the report
copy) and a Plotly figure (the interactive twin the app renders from a
``.plotly.json`` sidecar), the way ``profile_figure.plot_data`` does it.
matplotlib is used through its object API (``Figure`` + the Agg canvas), so
no global pyplot state is touched.
"""

import io
import math

import numpy as np

_TEXT = "#222222"
_GRID = "#dddddd"
_MAP_BG = "#f4f4f4"
_OBS = "#1f4e79"
_BOUND = "#7a1f1f"
_PALETTE = ["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2", "#e08a1e",
            "#3b8ea5", "#7f7f7f", "#a05d2c"]
_OUTLINES = ["#111111", "#3b8ea5", "#8064a2", "#4f8f3a", "#e08a1e",
             "#7f7f7f"]
#: Curves drawn on one exceedance figure; the rest are listed in the JSON.
MAX_CURVES = 12
#: A raster map up to this many lattice cells; beyond, cells as markers.
_MAX_RASTER = 100000


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


def exceedance_figure(result, dpi=150, width_in=7.0, height_in=4.8):
    """Exceedance curve P(V > v) per region/level, P50 and P90 marked.

    Returns ``(png_bytes, plotly_figure_or_None, n_curves_drawn)``."""
    q = result.quantity
    curves = exceedance_curves(result)
    xlabel = f"Volume {q} the level, v (m³)"
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


def probability_map_arrays(result, region=None, level=None):
    """What both renderings draw: the probability at every simulated cell
    (all regions, the chosen region's level) on the lattice, the outlines,
    the observations and bounds, and the view."""
    region, level, p_cells = _map_points(result, region, level)
    lat = result.lattice
    nx, ny = int(lat["nx"]), int(lat["ny"])
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
    # the view: the simulated cells plus a margin, widened to take in the
    # observations and bounds that are near (within one view diagonal)
    x_lo, x_hi = gx[0] - cell / 2, gx[-1] + cell / 2
    y_lo, y_hi = gy[0] - cell / 2, gy[-1] + cell / 2
    pad = max(float(lat.get("margin", 0.0)), 2 * cell)
    x_lo, x_hi, y_lo, y_hi = x_lo - pad, x_hi + pad, y_lo - pad, y_hi + pad
    diag = math.hypot(x_hi - x_lo, y_hi - y_lo)
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    near = np.hypot(result.obs_x - cx, result.obs_y - cy) <= diag
    bnds = list(result.bounds_input or [])
    bnear = [b for b in bnds if math.hypot(b["x"] - cx, b["y"] - cy) <= diag]
    xs = np.concatenate([[x_lo, x_hi], result.obs_x[near],
                         [b["x"] for b in bnear]])
    ys = np.concatenate([[y_lo, y_hi], result.obs_y[near],
                         [b["y"] for b in bnear]])
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    extra = 0.04 * span
    return {
        "region": region, "level": level, "gx": gx, "gy": gy,
        "raster": raster, "cells_x": result.cell_x, "cells_y": result.cell_y,
        "p_cells": p_cells, "cell": cell,
        "xlim": (xs.min() - extra, xs.max() + extra),
        "ylim": (ys.min() - extra, ys.max() + extra),
        "obs_mask": near, "n_obs_off": int((~near).sum()),
        "bounds": bnear,
    }


def probability_map_figure(result, region=None, level=None, dpi=150,
                           width_in=7.0):
    """Plan map of P(z > level) (or < for 'below') over the simulated cells,
    drawn to scale, with the region outlines, the named observations and the
    bounds. Returns ``(png_bytes, plotly_figure_or_None, meta)``."""
    a = probability_map_arrays(result, region, level)
    rel = ">" if result.quantity == "above" else "<"
    ctitle = f"P(z {rel} {a['level']:g})"
    title = (f"Probability that z {rel} {a['level']:g} "
             f"({a['region']}, {result.n_kept:,} realizations)")
    xlabel, ylabel = "x (m)", "y (m)"
    names = list(result.obs_names)
    ox, oy, ov = result.obs_x, result.obs_y, result.obs_values
    m = a["obs_mask"]

    (x0, x1), (y0, y1) = a["xlim"], a["ylim"]
    aspect = (y1 - y0) / max(x1 - x0, 1e-9)
    height_in = min(max(5.2 * aspect + 2.0, 4.0), 10.0)
    fig = _new_figure(width_in, height_in, dpi)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_MAP_BG)
    if a["raster"] is not None:
        c = a["cell"]
        im = ax.imshow(a["raster"], origin="lower", cmap="YlOrRd",
                       vmin=0.0, vmax=1.0, interpolation="nearest",
                       extent=(a["gx"][0] - c / 2, a["gx"][-1] + c / 2,
                               a["gy"][0] - c / 2, a["gy"][-1] + c / 2))
    else:
        im = ax.scatter(a["cells_x"], a["cells_y"], c=a["p_cells"],
                        cmap="YlOrRd", vmin=0.0, vmax=1.0, marker="s",
                        s=4, linewidths=0)
    for k, g in enumerate(result.regions_geom):
        poly = np.vstack([g["polygon"], g["polygon"][:1]])
        ax.plot(poly[:, 0], poly[:, 1], color=_OUTLINES[k % len(_OUTLINES)],
                lw=2.0 if g["name"] == a["region"] else 1.3,
                ls="-" if g["name"] == a["region"] else "--",
                label=f"Region: {g['name']}")
    ax.scatter(ox[m], oy[m], marker="^", s=55, color=_OBS,
               edgecolors="white", linewidths=0.8, zorder=5,
               label="Observations")
    for k in np.flatnonzero(m):
        ax.annotate(names[k], (ox[k], oy[k]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color=_TEXT,
                    zorder=6)
    if a["bounds"]:
        bxs = [b["x"] for b in a["bounds"]]
        bys = [b["y"] for b in a["bounds"]]
        ax.scatter(bxs, bys, marker="X", s=50, color=_BOUND,
                   edgecolors="white", linewidths=0.6, zorder=5,
                   label="Bounds (inequality data)")
        for b in a["bounds"]:
            ax.annotate(b["name"], (b["x"], b["y"]), xytext=(5, -10),
                        textcoords="offset points", fontsize=8,
                        color=_BOUND, zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")                        # 1 m across = 1 m up
    ax.grid(True, color="white", lw=0.6)
    sub = (f"\n{a['n_obs_off']} observation(s) beyond the map"
           if a["n_obs_off"] else "")
    _style_axes(ax, title + sub, xlabel, ylabel)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(ctitle, color=_TEXT)
    cb.ax.tick_params(colors=_TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, fontsize=8, labelcolor=_TEXT)
    png = _png(fig, dpi)

    pfig = None
    try:
        import plotly.graph_objects as go
    except Exception:                                  # noqa: BLE001
        go = None
    if go is not None:
        pfig = go.Figure()
        cbar = {"title": {"text": ctitle, "font": {"color": _TEXT}},
                "tickfont": {"color": _TEXT}}
        if a["raster"] is not None:
            z = [[None if not np.isfinite(v) else float(v) for v in row]
                 for row in a["raster"]]
            pfig.add_trace(go.Heatmap(
                z=z, x=a["gx"].tolist(), y=a["gy"].tolist(),
                colorscale="YlOrRd", zmin=0.0, zmax=1.0, colorbar=cbar,
                name=ctitle, showscale=True,
                hovertemplate=("x %{x:.1f} m<br>y %{y:.1f} m<br>"
                               + ctitle + " %{z:.2f}<extra></extra>")))
        else:
            pfig.add_trace(go.Scattergl(
                x=a["cells_x"].tolist(), y=a["cells_y"].tolist(),
                mode="markers", name=ctitle, showlegend=False,
                marker={"color": a["p_cells"].tolist(),
                        "colorscale": "YlOrRd", "cmin": 0.0, "cmax": 1.0,
                        "symbol": "square", "size": 4, "colorbar": cbar},
                hovertemplate=("x %{x:.1f} m<br>y %{y:.1f} m<br>"
                               + ctitle + " %{marker.color:.2f}"
                               "<extra></extra>")))
        for k, g in enumerate(result.regions_geom):
            poly = np.vstack([g["polygon"], g["polygon"][:1]])
            main = g["name"] == a["region"]
            pfig.add_trace(go.Scatter(
                x=poly[:, 0].tolist(), y=poly[:, 1].tolist(), mode="lines",
                name=f"Region: {g['name']}", hoverinfo="name",
                line={"color": _OUTLINES[k % len(_OUTLINES)],
                      "width": 2.5 if main else 1.6,
                      "dash": "solid" if main else "dash"}))
        idx = np.flatnonzero(m)
        pfig.add_trace(go.Scatter(
            x=ox[idx].tolist(), y=oy[idx].tolist(), mode="markers+text",
            name="Observations", text=[names[k] for k in idx],
            textposition="top right", textfont={"color": _TEXT, "size": 11},
            marker={"symbol": "triangle-up", "size": 11, "color": _OBS,
                    "line": {"color": "white", "width": 1}},
            hovertext=[f"{names[k]}<br>value {ov[k]:g}<br>x {ox[k]:g} m, "
                       f"y {oy[k]:g} m" for k in idx],
            hoverinfo="text"))
        if a["bounds"]:
            pfig.add_trace(go.Scatter(
                x=[b["x"] for b in a["bounds"]],
                y=[b["y"] for b in a["bounds"]], mode="markers+text",
                name="Bounds (inequality data)",
                text=[b["name"] for b in a["bounds"]],
                textposition="bottom right",
                textfont={"color": _BOUND, "size": 11},
                marker={"symbol": "x", "size": 10, "color": _BOUND},
                hovertext=[
                    f"{b['name']}<br>"
                    + (f"max {b['max']:g}" if b["max"] is not None else "")
                    + (" " if b["max"] is not None and b["min"] is not None
                       else "")
                    + (f"min {b['min']:g}" if b["min"] is not None else "")
                    + f"<br>x {b['x']:g} m, y {b['y']:g} m"
                    for b in a["bounds"]],
                hoverinfo="text"))
        _plotly_layout(pfig, title, xlabel, ylabel)
        pfig.update_layout(plot_bgcolor=_MAP_BG)
        pfig.update_xaxes(range=list(a["xlim"]), showgrid=False)
        pfig.update_yaxes(range=list(a["ylim"]), showgrid=False,
                          scaleanchor="x", scaleratio=1)
    meta = {"region": a["region"], "level": a["level"],
            "observations_beyond_map": a["n_obs_off"]}
    return png, pfig, meta
