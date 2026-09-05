"""
Matplotlib drawing for subsurface profile schematics.

The vertical scale is TRUE (elevations are plotted to scale); the horizontal
direction is schematic — a 1-D profile has no width, so the band is drawn on a
unit-width axis and a pile/footing is drawn at an exaggerated width so it is
visible.  The figure says so in its footer.

matplotlib is an optional dependency and is imported lazily via
``geotech_common.plotting.get_pyplot`` (project pattern), so this module
imports cleanly without it.

Colour/hatch palette is the house engineering-report palette (the muted earth
tones used by ``slope_stability.plotting``).  Every band also carries its NAME
as text and a distinct hatch, so a layer is never identified by colour alone.
"""

import io

#: Engineering-report palette (muted earth tones) — house convention.
LAYER_COLORS = ('#e8d9b0', '#cfa97c', '#a9c3a4', '#c9b8a6',
                '#dfd0a2', '#b3c4cf', '#d2c3ab', '#bfd3bf')
LAYER_HATCHES = ('', '..', '//', '\\\\', 'xx', '--', '++', 'oo')

_GWT_COLOR = '#1f6fbf'
_POND_COLOR = '#9ecbff'
_FILL_COLOR = '#d8cba6'
_FILL_HATCH = '///'
_EDGE_COLOR = '#777777'
_CONCRETE_FACE = '#9aa3ad'
_CONCRETE_EDGE = '#2f3640'
_NOTE_COLOR = '#444444'
_MUTED = '#888888'

# Horizontal geometry of the schematic (axes data units; the band spans 0..1).
_X0, _X1 = 0.0, 1.0
#: Drawn width of a foundation, as a fraction of the section width.
_FOUNDATION_WIDTH = {"pile": 0.055, "micropile": 0.045, "drilled_shaft": 0.075,
                     "wall": 0.10, "footing": 0.34}
#: Horizontal centre of the foundation overlay.
_FOUNDATION_X = {"footing": 0.50}
_FOUNDATION_X_DEFAULT = 0.62

#: A band shorter than this fraction of the plot height gets an outside label.
_MIN_INSIDE_LABEL = 0.045

#: Text sitting on a hatched band needs a light backing to stay legible.
_TEXT_BBOX = dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1.4)


def _get_plt():
    from geotech_common.plotting import get_pyplot
    return get_pyplot()


def _fmt_elev(value: float, ground: float, axis: str) -> str:
    """Format an elevation for the vertical axis / labels."""
    if axis == "depth":
        return f"{ground - value:.1f}"
    return f"{value:.1f}"


def _thin_labels(values, span: float, min_gap_frac: float = 0.028):
    """Drop tick values that would print on top of the previous one."""
    kept = []
    for v in sorted(set(values), reverse=True):
        if not kept or abs(kept[-1] - v) >= min_gap_frac * span:
            kept.append(v)
    return kept


def _draw_band(ax, top, bottom, color, hatch, zorder=1, alpha=0.85):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (_X0, bottom), _X1 - _X0, top - bottom, facecolor=color, alpha=alpha,
        hatch=hatch, edgecolor=_EDGE_COLOR, linewidth=0.6, zorder=zorder))


def _label_band(ax, band, y_span, index_text=""):
    """Name a band inside it, or outside with a leader when it is too thin."""
    mid = 0.5 * (band["top"] + band["bottom"])
    name = band["name"]
    if band.get("settling"):
        name += " (settling)"
    text = name if not index_text else f"{index_text} {name}"
    desc = band.get("description") or ""

    if band["thickness"] / y_span >= _MIN_INSIDE_LABEL:
        if desc:
            # Name in bold above the band's mid-height, description below it.
            ax.text(0.035, mid, text, ha="left", va="bottom", fontsize=8.0,
                    fontweight="bold", color="#222222", zorder=9,
                    bbox=_TEXT_BBOX)
            ax.text(0.035, mid, desc, ha="left", va="top", fontsize=7.2,
                    color="#555555", zorder=9, bbox=_TEXT_BBOX)
        else:
            ax.text(0.035, mid, text, ha="left", va="center", fontsize=8.0,
                    fontweight="bold", color="#222222", zorder=9,
                    bbox=_TEXT_BBOX)
    else:
        label = f"{text} — {desc}" if desc else text
        ax.annotate(label, xy=(0.30, mid), xytext=(1.06, mid),
                    ha="left", va="center", fontsize=7.5, color="#222222",
                    zorder=9,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=_MUTED,
                                    shrinkA=0, shrinkB=0))


def _draw_settling_arrows(ax, band, y_span):
    """Down-arrows marking a compressible/settling stratum."""
    if band["thickness"] / y_span < 0.05:
        return
    y0 = band["top"] - 0.18 * band["thickness"]
    y1 = band["bottom"] + 0.18 * band["thickness"]
    for x in (0.76, 0.82):
        ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#7a5c2e"))


def _draw_water(ax, water, ground, axis, y_span, top):
    shown = min(water, top)
    ax.plot([_X0, _X1], [shown, shown], ls="--", lw=1.4, color=_GWT_COLOR,
            zorder=6)
    ax.plot([0.90], [shown], marker="v", markersize=8, color=_GWT_COLOR,
            zorder=7)
    value = _fmt_elev(water, ground, axis)
    label = (f"GWT {value} m depth" if axis == "depth"
             else f"GWT El. {value} m")
    # Right-aligned: the left of the section carries the layer names.
    ax.text(0.99, shown + 0.012 * y_span, label, ha="right", va="bottom",
            fontsize=8.0, color=_GWT_COLOR, fontweight="bold", zorder=7,
            bbox=_TEXT_BBOX)
    if water > top:
        ax.add_patch(_rect(top, water, _POND_COLOR, "", alpha=0.45, zorder=1))


def _rect(bottom_top_a, bottom_top_b, color, hatch, alpha=0.85, zorder=1):
    from matplotlib.patches import Rectangle
    lo, hi = sorted((bottom_top_a, bottom_top_b))
    return Rectangle((_X0, lo), _X1 - _X0, hi - lo, facecolor=color,
                     alpha=alpha, hatch=hatch, edgecolor="none", zorder=zorder)


def _draw_surcharge(ax, surcharge, top, y_span):
    y_arrow = top + 0.055 * y_span
    ax.plot([0.06, 0.94], [y_arrow, y_arrow], color=_NOTE_COLOR, lw=1.0,
            zorder=7)
    for i in range(7):
        x = 0.10 + i * (0.80 / 6.0)
        ax.annotate("", xy=(x, top), xytext=(x, y_arrow), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                    color=_NOTE_COLOR))
    ax.text(0.5, y_arrow + 0.012 * y_span, surcharge["label"], ha="center",
            va="bottom", fontsize=8.5, fontweight="bold", color=_NOTE_COLOR,
            zorder=7)


def _draw_foundation(ax, found, ground, axis, y_span, label_offset=0.018):
    from matplotlib.patches import Rectangle

    ftype = found["type"]
    width = _FOUNDATION_WIDTH.get(ftype, 0.06)
    cx = _FOUNDATION_X.get(ftype, _FOUNDATION_X_DEFAULT)
    x_left = cx - 0.5 * width

    ax.add_patch(Rectangle(
        (x_left, found["tip"]), width, found["length"],
        facecolor=_CONCRETE_FACE, edgecolor=_CONCRETE_EDGE, linewidth=1.1,
        zorder=10))

    size_txt = (f"{found['size']:g} m " + ("wide" if ftype == "footing"
                                           else "dia."))
    head_label = f"{found['label']} ({size_txt})"
    ax.text(cx, found["head"] + label_offset * y_span, head_label,
            ha="center", va="bottom", fontsize=8.2, fontweight="bold",
            color=_CONCRETE_EDGE, zorder=11)

    # Tip elevation callout, on the opposite side from the layer labels.
    tip_value = _fmt_elev(found["tip"], ground, axis)
    tip_txt = (f"Tip {tip_value} m depth" if axis == "depth"
               else f"Tip El. {tip_value} m")
    ax.plot([x_left + width, x_left + width + 0.05], [found["tip"]] * 2,
            color=_CONCRETE_EDGE, lw=0.8, zorder=11)
    ax.text(x_left + width + 0.06, found["tip"], tip_txt, ha="left",
            va="center", fontsize=7.5, color=_CONCRETE_EDGE, zorder=11)


def _draw_annotations(ax, notes, y_span):
    for note in notes:
        y = note["elevation"]
        ax.plot([_X0, _X1], [y, y], ls=":", lw=0.9, color="#555555",
                alpha=0.9, zorder=9)
        if note["side"] == "left":
            # Clear of the elevation tick labels on the left spine.
            ax.annotate(note["text"], xy=(_X0, y), xytext=(-0.15, y),
                        ha="right", va="center", fontsize=8.0, zorder=11,
                        color="#333333",
                        arrowprops=dict(arrowstyle="-", lw=0.8,
                                        color="#555555"))
        else:
            ax.annotate(note["text"], xy=(_X1, y), xytext=(1.04, y),
                        ha="left", va="center", fontsize=8.0, zorder=11,
                        color="#333333",
                        arrowprops=dict(arrowstyle="-", lw=0.8,
                                        color="#555555"))


def build_profile_figure(resolved: dict, *, title: str = "Subsurface Profile",
                         width_in: float = 6.5, height_in: float = 8.0):
    """Draw a resolved profile and return the matplotlib ``(fig, ax)``.

    The caller owns the figure (and must close it).  ``resolved`` is the dict
    returned by :func:`profile_figure.geometry.resolve_profile`.
    """
    plt = _get_plt()

    ground = resolved["ground"]
    axis = resolved["axis"]
    top = resolved["top"]
    base = resolved["base"]
    found = resolved["foundation"]
    water = resolved["water"]

    y_lo = min([base] + ([found["tip"]] if found else [])
               + [n["elevation"] for n in resolved["annotations"]])
    y_hi = max([top] + ([found["head"]] if found else [])
               + ([water] if water is not None else [])
               + [n["elevation"] for n in resolved["annotations"]])
    y_span = max(y_hi - y_lo, 1e-6)
    pad_lo = 0.05 * y_span
    pad_hi = 0.05 * y_span
    if resolved["surcharge"]:
        pad_hi += 0.11 * y_span
    if found:
        pad_hi += 0.03 * y_span

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Bands: fill (above ground) then the soil layers.
    interfaces = []
    if resolved["fill"]:
        f = resolved["fill"]
        _draw_band(ax, f["top"], f["bottom"], f.get("color") or _FILL_COLOR,
                   f.get("hatch") or _FILL_HATCH, zorder=2)
        _label_band(ax, f, y_span)
        interfaces.append(f["top"])
        if f.get("settling"):
            _draw_settling_arrows(ax, f, y_span)

    for i, band in enumerate(resolved["layers"]):
        color = band.get("color") or LAYER_COLORS[i % len(LAYER_COLORS)]
        hatch = band.get("hatch")
        if hatch is None:
            hatch = LAYER_HATCHES[i % len(LAYER_HATCHES)]
        _draw_band(ax, band["top"], band["bottom"], color, hatch, zorder=2)
        _label_band(ax, band, y_span)
        if band.get("settling"):
            _draw_settling_arrows(ax, band, y_span)
        interfaces.extend((band["top"], band["bottom"]))

    # Ground surface (heavy) — the datum every depth is measured from.
    ax.plot([_X0, _X1], [ground, ground], color="black", lw=1.9, zorder=6)
    if resolved["fill"]:
        ax.text(0.99, ground - 0.012 * y_span, "Original ground",
                ha="right", va="top", fontsize=7.2, color="#333333", zorder=7)

    if water is not None and water >= base - 1e-9:
        _draw_water(ax, water, ground, axis, y_span, top)
    if resolved["surcharge"]:
        _draw_surcharge(ax, resolved["surcharge"], top, y_span)
    if found:
        # Lift the head label clear of the surcharge arrow band when present.
        _draw_foundation(ax, found, ground, axis, y_span,
                         label_offset=0.115 if resolved["surcharge"] else 0.018)
    if resolved["annotations"]:
        _draw_annotations(ax, resolved["annotations"], y_span)

    # Axes: ticks AT the layer interfaces (the engineering-relevant numbers),
    # no horizontal scale.
    ticks = _thin_labels(interfaces, y_span)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_fmt_elev(t, ground, axis) for t in ticks],
                       fontsize=8)
    ax.set_ylabel("Depth below ground (m)" if axis == "depth"
                  else "Elevation (m)", fontsize=9.5)
    ax.set_xticks([])
    ax.set_xlim(_X0 - 0.02, _X1 + 0.02)
    ax.set_ylim(y_lo - pad_lo, y_hi + pad_hi)
    ax.grid(axis="y", alpha=0.25, linestyle=":", linewidth=0.7)
    ax.set_axisbelow(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12)

    fig.text(0.99, 0.012, "Schematic — vertical scale only; widths exaggerated",
             ha="right", va="bottom", fontsize=7, color=_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig, ax


def figure_to_png(fig, dpi: int = 150) -> bytes:
    """Render a matplotlib figure to PNG bytes (white background, tight box)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data
