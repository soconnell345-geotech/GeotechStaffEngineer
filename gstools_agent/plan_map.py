"""
Plan maps of a field over a site, drawn as drawings (owner, 2026-10-15): to
scale, with no axes, axis numbers, grid or frame; a scale bar and a north
arrow (``geotech_common.plan_furniture``, the furniture
``profile_figure.plot_data(plan_view=True)`` draws); the outlines given (a
property line dash-dot, as drafted); the observations marked and named; and
what the colours mean said in words, with units.

One code path for every plan map this module draws:

* the probability map of a conditional simulation (one plan,
  :func:`gstools_agent.conditional_figures.probability_map_figure`);
* the kriging map (:func:`kriging_map_figure`): the kriged mean and the
  kriging standard deviation side by side, at the same scale.

Each is drawn twice from the same arrays: a matplotlib PNG (the report copy)
and a Plotly figure (the interactive twin the app shows from a
``.plotly.json`` sidecar). matplotlib is used through its object API, so no
global pyplot state is touched.

The view and the band below it that holds the scale bar and north arrow are
sized in POINTS as printed (as ``plot_data``'s plans are, BACKLOG GM-147), so
the furniture reads the same on a long flat site as on a tall narrow one.
"""

import io
import math

import numpy as np

_INK = "#222222"
_OBS = "#1f4e79"
_BOUND = "#7a1f1f"
_OUTLINES = ["#111111", "#3b8ea5", "#8064a2", "#4f8f3a", "#e08a1e",
             "#7f7f7f"]
_CONTOUR_LINE = "#4d4d4d"

#: Margins round the drawing, in points as printed: the band below holds the
#: scale bar and, at its right, the north arrow (``plot_data``'s layout).
_PT = {"side": 14.0, "above": 12.0, "below": 58.0}
_BAR_DROP = 30.0
_BAR_HEIGHT = 4.0
_ARROW_IN = 12.0
_ARROW_DROP = 32.0
_BAR_TO_ARROW = 44.0
#: Figure layout, inches.
_WIDTH_IN = 7.0
_SIDE_IN = 0.1
_GAP_IN = 0.3
_CBAR_IN = 0.95             #: room a colour scale takes beside its plan
_TITLE_LINE_IN = 0.24
_PANEL_TITLE_IN = 0.28
_FOOT_IN = 0.55             #: the legend and the caption below
#: A raster map up to this many cells; beyond, cells as markers.
MAX_RASTER = 100000
#: No contour label within this many points of an observation (its name).
_LABEL_CLEAR_PT = 18.0
#: Two plans stack (one above the other) when the site is this much wider
#: than tall; otherwise they sit side by side.
_STACK_ASPECT = 0.55


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

def plan_view(core_x, core_y, pad, obs_x=(), obs_y=(), extra_x=(),
              extra_y=()):
    """The part of the site a map shows: the box round ``core`` (the field
    drawn and the outlines) plus ``pad``, widened to take in the
    observations and ``extra`` points (bounds) that are near -- within one
    view diagonal of its centre. Returns ``((x0, x1), (y0, y1), obs_mask,
    extra_mask)``; an observation off the map is counted by the caller."""
    cx_, cy_ = np.asarray(core_x, float), np.asarray(core_y, float)
    x_lo, x_hi = cx_.min() - pad, cx_.max() + pad
    y_lo, y_hi = cy_.min() - pad, cy_.max() + pad
    diag = math.hypot(x_hi - x_lo, y_hi - y_lo)
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    ox, oy = np.asarray(obs_x, float), np.asarray(obs_y, float)
    ex, ey = np.asarray(extra_x, float), np.asarray(extra_y, float)
    near = np.hypot(ox - cx, oy - cy) <= diag
    enear = np.hypot(ex - cx, ey - cy) <= diag
    xs = np.concatenate([[x_lo, x_hi], ox[near], ex[enear]])
    ys = np.concatenate([[y_lo, y_hi], oy[near], ey[enear]])
    return ((float(xs.min()), float(xs.max())),
            (float(ys.min()), float(ys.max())), near, enear)


def _layout(view, n_panels, colorbar):
    """Where everything goes. The plans are as wide as the figure allows;
    their scale (data units per printed point) follows, then the margins,
    the band below and the figure's height."""
    from geotech_common.plan_furniture import nice_length

    (x0, x1), (y0, y1) = view
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    stacked = n_panels > 1 and dy / dx < _STACK_ASPECT
    ncols = 1 if (n_panels == 1 or stacked) else n_panels
    nrows = n_panels if stacked else 1
    cbar = _CBAR_IN if colorbar else 0.0
    w_in = ((_WIDTH_IN - 2 * _SIDE_IN - (ncols - 1) * _GAP_IN) / ncols
            - cbar)
    w_pt = w_in * 72.0
    length = nice_length(dx)
    side, above, below = _PT["side"], _PT["above"], _PT["below"]
    s = dx / max(w_pt - 2 * side, 0.2 * w_pt)
    for _ in range(20):                 # the bar may reach left of the site
        bx = x1 - (_ARROW_IN + _BAR_TO_ARROW) * s - length
        left = min(x0 - side * s, bx - side * s)
        new = (x1 + side * s - left) / w_pt
        if abs(new - s) <= 1e-9 * s:
            break
        s = new
    bx = x1 - (_ARROW_IN + _BAR_TO_ARROW) * s - length
    xlim = (min(x0 - side * s, bx - side * s), x1 + side * s)
    ylim = (y0 - below * s, y1 + above * s)
    # stacked plans: only the lowest carries the band (one scale bar and
    # north arrow serve both: they are drawn at the same scale)
    ylim_plain = (y0 - side * s, y1 + above * s)
    return {"xlim": xlim, "ylim": ylim, "ylim_plain": ylim_plain,
            "view_y": (y0, y1), "scale": s, "length": length,
            "bar_x": bx, "bar_y": y0 - _BAR_DROP * s,
            "bar_h": _BAR_HEIGHT * s,
            "arrow": (x1 - _ARROW_IN * s, y0 - _ARROW_DROP * s),
            "ncols": ncols, "nrows": nrows, "w_in": w_in,
            "h_in": (ylim[1] - ylim[0]) / s / 72.0,
            "h_plain_in": (ylim_plain[1] - ylim_plain[0]) / s / 72.0,
            "cbar_in": cbar}


def _panel_ylim(lay, k, n):
    """The y range of plan ``k`` of ``n``: with the band below when it is
    the lowest of stacked plans, or when the plans sit side by side."""
    if lay["nrows"] > 1 and k < n - 1:
        return lay["ylim_plain"], lay["h_plain_in"]
    return lay["ylim"], lay["h_in"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def contour_levels(z, bands=8):
    """Round contour levels over the finite values of ``z``: a 1, 2, 2.5 or
    5 x 10^k interval giving about ``bands`` bands. Returns ``(levels,
    step)``; a flat field gives one band about its value and ``step`` 0."""
    from gstools_agent.regions import nice_ceil

    v = np.asarray(z, float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-9 * max(abs(hi), 1.0):
        half = max(abs(hi) * 1e-3, 1e-6)
        return np.array([lo - half, hi + half]), 0.0
    step = nice_ceil((hi - lo) / float(bands))
    a = math.floor(lo / step + 1e-9) * step
    b = math.ceil(hi / step - 1e-9) * step
    return np.arange(a, b + step / 2, step), step


def _colormaps(name, top=1.0):
    """The matplotlib colormap ``name``, cut to its lighter part (0 to
    ``top``) so contour labels and names stay legible on it, and the same
    colours as a Plotly colorscale: one colouring for both renderings."""
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap, to_hex

    base = colormaps[name]
    cm = (base if top >= 1.0 else
          ListedColormap(base(np.linspace(0.0, top, 256)),
                         name=f"{name}_light"))
    return cm, [[float(t), to_hex(cm(t))] for t in np.linspace(0, 1, 11)]


def _decimals(step):
    """Decimals that print every multiple of ``step`` exactly."""
    if step <= 0:
        return 2
    for d in range(0, 7):
        if abs(round(step, d) - step) <= 1e-9 * step:
            return d
    return 6


def _halo(width=2.4):
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground="white")]


def _closed(poly):
    p = np.asarray(poly, float)
    return np.vstack([p, p[:1]])


def outline_style(o, k_region):
    """How an outline is drawn: ``kind`` 'region' (the regions totalled;
    the first solid and heavier, the rest dashed), 'property_line' (dash-dot,
    the drafting convention) or 'outline' (thin solid grey)."""
    kind = o.get("kind", "outline")
    if kind == "region":
        main = o.get("main", False)
        return {"color": _OUTLINES[k_region % len(_OUTLINES)],
                "lw": 2.0 if main else 1.3, "mpl_ls": "-" if main else "--",
                "dash": "solid" if main else "dash"}
    if kind == "property_line":
        return {"color": "#444444", "lw": 1.2, "mpl_ls": "-.",
                "dash": "dashdot"}
    return {"color": "#777777", "lw": 1.0, "mpl_ls": "-", "dash": "solid"}


def parse_outlines(outlines):
    """``outlines`` as given to a tool -> ``[{name, xy, closed, kind}]``.
    Each item is ``{name, polygon: [[x, y], ...]}`` (closed) or ``{name,
    line: [[x, y], ...]}`` (open, e.g. a wall or a road edge), with ``kind``
    'property_line' (drawn dash-dot) or 'outline' (default)."""
    if outlines is None:
        return []
    if isinstance(outlines, dict):
        outlines = [outlines]
    if not isinstance(outlines, (list, tuple)):
        raise ValueError("outlines must be a list of {name, polygon} or "
                         "{name, line} dicts")
    out = []
    for i, o in enumerate(outlines):
        if not isinstance(o, dict):
            raise ValueError(f"outlines[{i}] must be a dict with name and "
                             f"polygon (or line)")
        unknown = sorted(set(o) - {"name", "polygon", "line", "kind"})
        if unknown:
            raise ValueError(f"outlines[{i}] has unknown key(s) {unknown}; "
                             f"valid keys are name, polygon, line, kind")
        kind = o.get("kind") or "outline"
        if kind not in ("outline", "property_line"):
            raise ValueError(f"outlines[{i}] kind must be 'outline' or "
                             f"'property_line', got '{kind}'")
        pts = o.get("polygon") if o.get("polygon") is not None else o.get(
            "line")
        closed = o.get("polygon") is not None
        try:
            xy = np.asarray(pts, dtype=float)
        except (TypeError, ValueError):
            xy = None
        if (xy is None or xy.ndim != 2 or xy.shape[1] != 2
                or xy.shape[0] < (3 if closed else 2)
                or not np.all(np.isfinite(xy))):
            raise ValueError(
                f"outlines[{i}] needs polygon: [[x, y], ...] (3 points or "
                f"more) or line: [[x, y], ...] (2 or more), finite numbers")
        name = str(o.get("name") or ("Property line"
                                     if kind == "property_line"
                                     else "Outline"))
        out.append({"name": name, "xy": xy, "closed": closed, "kind": kind})
    return out


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------

def plan_map_figure(panels, *, view, title, observations, outlines=(),
                    bounds=(), caption=None, north_deg=None, units="m",
                    dpi=150):
    """Draw one or two plans of a field over the same view.

    ``panels``: per plan ``{title, kind, gx, gy, z, cmap, hover}`` --
    ``kind`` 'raster' (cells of size ``cell`` centred on ``gx`` x ``gy``,
    ``z`` (ny, nx) with NaN off the field, ``vmin`` / ``vmax`` and a colour
    scale named ``cbar_label``; or ``cells_x`` / ``cells_y`` / ``values``
    when too large for a raster) or 'contour' (``z`` (ny, nx) on the grid
    nodes, filled bands between labelled contour lines; the labels are the
    scale). ``hover`` names the value in the interactive chart, with
    ``hover_unit``.
    ``observations``: ``{x, y, names, values, mask, value_label, unit}``,
    drawn where ``mask``. ``outlines``: ``[{name, xy, closed, kind[,
    main]}]`` (see :func:`outline_style`). ``bounds``: ``[{name, x, y, max,
    min}]``. ``caption``: a line under the figure.

    Returns ``(png_bytes, plotly_figure_or_None, north_note)``."""
    from geotech_common import plan_furniture as pf

    angle, north_note = pf.north_or_default(north_deg)
    lay = _layout(view, len(panels),
                  any(p.get("cbar_label") for p in panels))
    png = _draw_png(panels, lay, title, observations, outlines, bounds,
                    caption, angle, units, dpi)
    pfig = _draw_plotly(panels, lay, title, observations, outlines, bounds,
                        caption, angle, units)
    return png, pfig, north_note


def _draw_png(panels, lay, title, obs, outlines, bounds, caption, angle,
              units, dpi):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from geotech_common import plan_furniture as pf

    n = len(panels)
    multi = n > 1
    ptitle = _PANEL_TITLE_IN if multi else 0.0
    top = len(str(title).splitlines()) * _TITLE_LINE_IN + 0.1
    n_cap = len(str(caption).splitlines()) if caption else 0
    foot = _FOOT_IN + 0.18 * n_cap
    nrows, ncols = lay["nrows"], lay["ncols"]
    row_h = [max(_panel_ylim(lay, k, n)[1] for k in range(n)
                 if k // ncols == r) for r in range(nrows)]
    fig_h = top + sum(row_h) + nrows * ptitle + (nrows - 1) * 0.15 + foot
    fig = Figure(figsize=(_WIDTH_IN, fig_h), dpi=dpi)
    FigureCanvasAgg(fig)
    if multi:
        fig.text(_SIDE_IN / _WIDTH_IN, 1.0 - 0.08 / fig_h, title,
                 ha="left", va="top", fontsize=11, color=_INK)

    handles = {}
    ox, oy = np.asarray(obs["x"], float), np.asarray(obs["y"], float)
    mask = np.asarray(obs["mask"], bool)
    names = list(obs["names"])
    for k, p in enumerate(panels):
        r, c = divmod(k, ncols)
        ylim, h_in = _panel_ylim(lay, k, n)
        left = _SIDE_IN + c * (lay["w_in"] + lay["cbar_in"] + _GAP_IN)
        bottom = (fig_h - top - sum(row_h[:r + 1]) - (r + 1) * ptitle
                  - r * 0.15)
        ax = fig.add_axes([left / _WIDTH_IN, bottom / fig_h,
                           lay["w_in"] / _WIDTH_IN, h_in / fig_h])
        ax.set_facecolor("white")
        cmap, _scale = _colormaps(p["cmap"], p.get("cmap_top", 1.0))
        if p["kind"] == "raster":
            if p.get("z") is not None:
                cs_ = p["cell"]
                im = ax.imshow(p["z"], origin="lower", cmap=cmap,
                               vmin=p["vmin"], vmax=p["vmax"],
                               interpolation="nearest",
                               extent=(p["gx"][0] - cs_ / 2,
                                       p["gx"][-1] + cs_ / 2,
                                       p["gy"][0] - cs_ / 2,
                                       p["gy"][-1] + cs_ / 2))
            else:
                im = ax.scatter(p["cells_x"], p["cells_y"], c=p["values"],
                                cmap=cmap, vmin=p["vmin"], vmax=p["vmax"],
                                marker="s", s=4, linewidths=0)
            if p.get("cbar_label"):     # the scale beside the site, not
                y0v, y1v = lay["view_y"]          # the band below it
                f0 = (y0v - ylim[0]) / (ylim[1] - ylim[0])
                f1 = (y1v - ylim[0]) / (ylim[1] - ylim[0])
                cax = fig.add_axes([(left + lay["w_in"] + 0.12) / _WIDTH_IN,
                                    (bottom + f0 * h_in) / fig_h,
                                    0.14 / _WIDTH_IN,
                                    (f1 - f0) * h_in / fig_h])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label(p["cbar_label"], color=_INK)
                cb.ax.tick_params(colors=_INK, labelsize=8)
                cb.outline.set_visible(False)
        else:
            levels, step = contour_levels(p["z"], p.get("bands", 8))
            ax.contourf(p["gx"], p["gy"], p["z"], levels=levels, cmap=cmap)
            if step > 0:
                cs = ax.contour(p["gx"], p["gy"], p["z"], levels=levels,
                                colors=_CONTOUR_LINE, linewidths=0.6,
                                linestyles="solid")
                d = _decimals(step)
                lab = cs.levels if len(cs.levels) <= 10 else cs.levels[::2]
                # labels on the lines, haloed; none on the small rings
                # round an observation, where they would sit on its name
                clear = _LABEL_CLEAR_PT * lay["scale"]
                for t in ax.clabel(cs, levels=lab, inline=False, fontsize=7,
                                   fmt=lambda v, d=d: f"{v:.{d}f}"):
                    tx, ty = t.get_position()
                    if mask.any() and float(np.min(np.hypot(
                            ox[mask] - tx, oy[mask] - ty))) < clear:
                        t.remove()
                        continue
                    t.set_color(_INK)
                    t.set_path_effects(_halo(2.0))
        ax.set_title(p["title"] if multi else title, loc="left",
                     fontsize=9.5 if multi else 11, color=_INK,
                     pad=4 if multi else 6)
        reg_k = 0
        for o in outlines:
            st = outline_style(o, reg_k)
            reg_k += o.get("kind") == "region"
            xy = _closed(o["xy"]) if o["closed"] else o["xy"]
            ax.plot(xy[:, 0], xy[:, 1], color=st["color"], lw=st["lw"],
                    ls=st["mpl_ls"], zorder=4)
            handles.setdefault(o["name"], Line2D(
                [], [], color=st["color"], lw=st["lw"], ls=st["mpl_ls"]))
        ax.scatter(ox[mask], oy[mask], marker="^", s=55, color=_OBS,
                   edgecolors="white", linewidths=0.8, zorder=5)
        for i in np.flatnonzero(mask):
            if not names[i]:          # unnamed: marked, never given a name
                continue
            ax.annotate(names[i], (ox[i], oy[i]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=_INK,
                        zorder=6, path_effects=_halo())
        handles.setdefault(obs.get("legend", "Observations"), Line2D(
            [], [], ls="", marker="^", ms=7, color=_OBS, mec="white"))
        if bounds:
            ax.scatter([b["x"] for b in bounds], [b["y"] for b in bounds],
                       marker="X", s=50, color=_BOUND, edgecolors="white",
                       linewidths=0.6, zorder=5)
            for b in bounds:
                ax.annotate(b["name"], (b["x"], b["y"]), xytext=(5, -10),
                            textcoords="offset points", fontsize=8,
                            color=_BOUND, zorder=6, path_effects=_halo())
            handles.setdefault("Bounds (inequality data)", Line2D(
                [], [], ls="", marker="X", ms=7, color=_BOUND, mec="white"))
        ax.set_xlim(*lay["xlim"])
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")      # 1 m across = 1 m up
        ax.set_autoscale_on(False)
        ax.set_axis_off()             # a plan: the scale bar gives the scale
        if k == n - 1:                # one scale bar and arrow: same scale
            pf.mpl_scale_bar(ax, lay["bar_x"], lay["bar_y"], lay["length"],
                             lay["bar_h"], units, fontsize=7.5,
                             text_color=_INK)
            pf.mpl_north_arrow(ax, lay["arrow"], angle)
    fig.legend(list(handles.values()), list(handles.keys()),
               loc="lower center",
               bbox_to_anchor=(0.5, (0.12 + 0.18 * n_cap) / fig_h),
               ncol=min(4, len(handles)), frameon=False, fontsize=8,
               labelcolor=_INK)
    if caption:
        fig.text(_SIDE_IN / _WIDTH_IN, 0.06 / fig_h, caption, ha="left",
                 va="bottom", fontsize=7.5, color="#555555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    return buf.getvalue()


def _draw_plotly(panels, lay, title, obs, outlines, bounds, caption, angle,
                 units):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:                                  # noqa: BLE001
        return None
    from geotech_common import plan_furniture as pf

    n = len(panels)
    nrows, ncols = lay["nrows"], lay["ncols"]
    if n == 1:
        fig = go.Figure()
    else:
        heights = [_panel_ylim(lay, r * ncols, n)[1] for r in range(nrows)]
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=[p["title"] for p in panels],
                            horizontal_spacing=0.06, vertical_spacing=0.08,
                            row_heights=heights if nrows > 1 else None)
    ox, oy = np.asarray(obs["x"], float), np.asarray(obs["y"], float)
    mask = np.asarray(obs["mask"], bool)
    names = list(obs["names"])
    ov = np.asarray(obs["values"], float)
    vlabel, vunit = obs.get("value_label", "value"), obs.get("unit", "")
    idx = np.flatnonzero(mask)
    for k, p in enumerate(panels):
        rc = ({} if n == 1 else
              {"row": k // ncols + 1, "col": k % ncols + 1})
        first = k == 0
        _cm, scale = _colormaps(p["cmap"], p.get("cmap_top", 1.0))
        hover = (f"x %{{x:.1f}} {units}<br>y %{{y:.1f}} {units}<br>"
                 f"{p['hover']} %{{z:.2f}}"
                 + (f" {p['hover_unit']}" if p.get("hover_unit") else "")
                 + "<extra></extra>")
        if p["kind"] == "raster":
            cbar = {"title": {"text": p.get("cbar_label", ""),
                              "font": {"color": _INK}},
                    "tickfont": {"color": _INK}}
            if p.get("z") is not None:
                z = [[None if not np.isfinite(v) else float(v) for v in row]
                     for row in p["z"]]
                fig.add_trace(go.Heatmap(
                    z=z, x=list(map(float, p["gx"])),
                    y=list(map(float, p["gy"])), colorscale=scale,
                    zmin=p["vmin"], zmax=p["vmax"], colorbar=cbar,
                    name=p.get("cbar_label", p["title"]),
                    showscale=bool(p.get("cbar_label")),
                    hovertemplate=hover), **rc)
            else:
                fig.add_trace(go.Scattergl(
                    x=list(map(float, p["cells_x"])),
                    y=list(map(float, p["cells_y"])), mode="markers",
                    name=p.get("cbar_label", p["title"]), showlegend=False,
                    marker={"color": list(map(float, p["values"])),
                            "colorscale": scale, "cmin": p["vmin"],
                            "cmax": p["vmax"], "symbol": "square",
                            "size": 4, "colorbar": cbar},
                    hovertemplate=hover.replace("%{z", "%{marker.color")),
                    **rc)
        else:
            levels, step = contour_levels(p["z"], p.get("bands", 8))
            z = [[None if not np.isfinite(v) else float(v) for v in row]
                 for row in p["z"]]
            fig.add_trace(go.Contour(
                z=z, x=list(map(float, p["gx"])),
                y=list(map(float, p["gy"])), colorscale=scale,
                showscale=False, name=p["title"],
                contours={"start": float(levels[0]),
                          "end": float(levels[-1]),
                          "size": float(step or (levels[-1] - levels[0])),
                          "coloring": "fill", "showlabels": step > 0,
                          "labelfont": {"size": 10, "color": _INK},
                          "labelformat": f".{_decimals(step)}f"},
                line={"width": 0.6, "color": _CONTOUR_LINE},
                hovertemplate=hover), **rc)
        reg_k = 0
        for o in outlines:
            st = outline_style(o, reg_k)
            reg_k += o.get("kind") == "region"
            xy = _closed(o["xy"]) if o["closed"] else o["xy"]
            fig.add_trace(go.Scatter(
                x=xy[:, 0].tolist(), y=xy[:, 1].tolist(), mode="lines",
                name=o["name"], legendgroup=o["name"], showlegend=first,
                hoverinfo="name",
                line={"color": st["color"], "width": st["lw"] + 0.5,
                      "dash": st["dash"]}), **rc)
        fig.add_trace(go.Scatter(
            x=ox[idx].tolist(), y=oy[idx].tolist(), mode="markers+text",
            name=obs.get("legend", "Observations"), legendgroup="obs",
            showlegend=first, text=[names[i] for i in idx],
            textposition="top right", textfont={"color": _INK, "size": 11},
            marker={"symbol": "triangle-up", "size": 11, "color": _OBS,
                    "line": {"color": "white", "width": 1}},
            hovertext=[(f"{names[i]}<br>" if names[i] else "")
                       + f"{vlabel} {ov[i]:g}"
                       + (f" {vunit}" if vunit else "")
                       + f"<br>x {ox[i]:g} {units}, y {oy[i]:g} {units}"
                       for i in idx],
            hoverinfo="text"), **rc)
        if bounds:
            fig.add_trace(go.Scatter(
                x=[b["x"] for b in bounds], y=[b["y"] for b in bounds],
                mode="markers+text", name="Bounds (inequality data)",
                legendgroup="bounds", showlegend=first,
                text=[b["name"] for b in bounds],
                textposition="bottom right",
                textfont={"color": _BOUND, "size": 11},
                marker={"symbol": "x", "size": 10, "color": _BOUND},
                hovertext=[
                    f"{b['name']}<br>"
                    + (f"max {b['max']:g}" if b["max"] is not None else "")
                    + (" " if b["max"] is not None and b["min"] is not None
                       else "")
                    + (f"min {b['min']:g}" if b["min"] is not None else "")
                    + f"<br>x {b['x']:g} {units}, y {b['y']:g} {units}"
                    for b in bounds],
                hoverinfo="text"), **rc)
    text = title.replace("\n", "<br>")
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white",
        plot_bgcolor="white", font={"color": _INK},
        title={"text": text, "x": 0.0, "xanchor": "left",
               "font": {"size": 15, "color": _INK}},
        legend={"orientation": "h", "x": 0.0, "y": -0.04, "yanchor": "top",
                "font": {"color": _INK}},
        margin={"l": 20, "r": 30, "t": 40 + 22 * len(text.split("<br>")),
                "b": 70 if not caption else 100})
    for k in range(n):
        sx = "" if k == 0 else str(k + 1)
        fig.layout[f"xaxis{sx}"].update(range=list(lay["xlim"]),
                                        visible=False)
        fig.layout[f"yaxis{sx}"].update(range=list(_panel_ylim(lay, k,
                                                               n)[0]),
                                        visible=False,
                                        scaleanchor=f"x{sx}",
                                        scaleratio=1)
    for a in fig.layout.annotations:          # the subplot titles: left
        a.update(font={"size": 12, "color": _INK})
    sx = "" if n == 1 else str(n)
    pf.plotly_scale_bar(fig, lay["bar_x"], lay["bar_y"], lay["length"],
                        lay["bar_h"], units, text_color=_INK,
                        xref=f"x{sx}", yref=f"y{sx}")
    pf.plotly_north_arrow(fig, lay["arrow"], angle, xref=f"x{sx}",
                          yref=f"y{sx}")
    if caption:
        fig.add_annotation(text=caption.replace("\n", "<br>"),
                           xref="paper", yref="paper", x=0.0, y=-0.02,
                           xanchor="left", yanchor="top", showarrow=False,
                           align="left", yshift=-40,
                           font={"size": 10, "color": "#555555"})
    return fig


# ---------------------------------------------------------------------------
# The kriging map
# ---------------------------------------------------------------------------

def _what(value_name):
    """What z is, in words (owner, 2026-10-15: "need to define z")."""
    return (str(value_name).strip() if value_name else "") or "value"


def kriging_map_figure(result, names=None, outlines=None, value_name=None,
                       value_unit="", north_deg=None, dpi=150):
    """Plan map of a kriging (:func:`gstools_agent.analyze_kriging`): the
    kriged mean and the kriging standard deviation side by side, at the same
    scale, as contours labelled with their values over the grid, with the
    observations named, the regions and the ``outlines`` (a property line,
    plan outlines; see :func:`parse_outlines`) drawn, a scale bar and a north
    arrow (``north_deg`` clockwise from up, as the source plan shows it; up
    when not given). ``value_name`` says what the values are (e.g. 'top of
    rock elevation'), ``value_unit`` their unit.

    Returns ``(png_bytes, plotly_figure_or_None, meta)``."""
    gx = np.asarray(result.grid_x, float)
    gy = np.asarray(result.grid_y, float)
    if gx.size < 2 or gy.size < 2:
        raise ValueError(f"a kriging map needs a grid of at least 2 x 2 "
                         f"points; this one is {gx.size} x {gy.size}")
    if result.obs_x is None:
        raise ValueError("this kriging result carries no observations "
                         "(made before figures existed); krige again")
    what = _what(value_name)
    unit = str(value_unit or "").strip()
    u = f" ({unit})" if unit else ""
    u2 = f" {unit}²" if unit else ""
    z = np.asarray(result.field, float).T          # (ny, nx)
    sd = np.sqrt(np.clip(np.asarray(result.krige_variance, float), 0.0,
                         None)).T
    n = int(result.n_data)
    names = [str(s) for s in names] if names is not None else [""] * n
    if len(names) != n:
        raise ValueError(f"names must be a list with one name per "
                         f"observation ({n})")
    outl = [dict(o, kind="region", main=(k == 0))
            for k, o in enumerate(
                {"name": g["name"], "xy": np.asarray(g["polygon"], float),
                 "closed": True} for g in (result.regions_geom or []))]
    outl += parse_outlines(outlines)
    core_x = [gx[0], gx[-1]] + [float(v) for o in outl
                                for v in o["xy"][:, 0]]
    core_y = [gy[0], gy[-1]] + [float(v) for o in outl
                                for v in o["xy"][:, 1]]
    xr, yr, near, _ = plan_view(core_x, core_y, 0.0, result.obs_x,
                                result.obs_y)
    # the title names z; the plans name what each shows of it. The
    # standard deviation's contours are fewer: it rises from the
    # observations, and rings of labels round every one would crowd them.
    panels = [
        {"title": f"Kriged mean{u}", "kind": "contour", "gx": gx, "gy": gy,
         "z": z, "cmap": "YlOrBr", "cmap_top": 0.55, "bands": 8,
         "hover": f"kriged mean {what}", "hover_unit": unit},
        {"title": f"Kriging standard deviation{u}", "kind": "contour",
         "gx": gx, "gy": gy, "z": sd, "cmap": "PuBu", "cmap_top": 0.55,
         "bands": 5, "hover": "kriging standard deviation",
         "hover_unit": unit},
    ]
    fit = ""
    if getattr(result, "variogram_fitted", False):
        fit = ", fitted to the empirical variogram" + (
            "" if getattr(result, "nugget_fitted", True)
            else " with the nugget held")
    off = int((~near).sum())
    off = (f" {off} observation(s) beyond the map, more than the map's "
           f"diagonal from its centre." if off else "")
    caption = (f"{result.kriging_type.capitalize()} kriging of {n} "
               f"observations, {result.model_type} covariance model: "
               f"variance {result.variance:.3g}{u2}, correlation length "
               f"{result.len_scale:.3g} m, nugget {result.nugget:.3g}{u2}"
               f"{fit}.\nThe standard deviation is of the value at a "
               f"point (nugget included).{off}")
    title = f"Kriging of the {what}"
    png, pfig, north_note = plan_map_figure(
        panels, view=(xr, yr), title=title,
        observations={"x": result.obs_x, "y": result.obs_y,
                      "names": names, "values": result.obs_values,
                      "mask": near, "value_label": what, "unit": unit},
        outlines=outl, caption=caption, north_deg=north_deg, dpi=dpi)
    meta = {"value_name": what, "north_note": north_note,
            "observations_beyond_map": int((~near).sum())}
    return png, pfig, meta


__all__ = ["plan_view", "plan_map_figure", "kriging_map_figure",
           "parse_outlines", "contour_levels", "MAX_RASTER"]
