"""
Shared matplotlib helpers for geotechnical plotting.

matplotlib is an optional dependency. All functions in this module
should be called only when matplotlib is available.
"""

from typing import Optional


def get_pyplot():
    """Import and return matplotlib.pyplot, or raise ImportError.

    Returns
    -------
    module
        matplotlib.pyplot module.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def setup_engineering_plot(ax, title: str, xlabel: str, ylabel: str,
                           grid: bool = True) -> None:
    """Apply standard engineering plot formatting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    grid : bool, optional
        Whether to show grid. Default True.
    """
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def key_under_axes(fig, axes, *, handles=None, labels=None, note=None,
                   fontsize=8, ncol=2):
    """One key for ``axes`` (an axes or a list of them, twins included)
    under them, without a frame, and the figure laid out to leave it its
    room. ``note`` (a line or two the figure states, such as its factors
    of safety) heads the key.

    Inside the axes a key or a note's box covers the drawing, or the
    drawing runs through it, once the figure is drawn for the width it
    prints at (5 in, lettering 7 pt or more: BACKLOG GM-186, GM-194);
    under the axes it covers nothing. The figure grows by the key's
    height, so the drawing keeps the room it had (a four-line key took a
    third of a chart's height). Call it when the figure is drawn, in
    place of ``tight_layout``.

    ``handles`` and ``labels`` default to the axes' labelled artists, one
    entry per label. Returns the key, or None when it has nothing to say.
    """
    from geotech_common.layout_ink import _lay_out
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    if handles is None:
        handles, labels = [], []
        for ax in axes:
            for h, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(h)
                    labels.append(label)
    elif labels is None:
        labels = [h.get_label() for h in handles]
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    key = None
    if handles or note:
        key = fig.legend(handles, labels, loc='lower center',
                         fontsize=fontsize, frameon=False,
                         ncol=max(1, min(ncol, len(handles))),
                         title=note, title_fontsize=fontsize)
        try:
            tall = (key.get_window_extent(fig.canvas.get_renderer()).height
                    / fig.dpi)
        except Exception:                                  # noqa: BLE001
            tall = 0.0
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h + tall)
    _lay_out(fig)
    return key


def halo(texts, colour='white'):
    """A thin ring of ``colour`` (the background) round the letters of each
    of ``texts``, so a line, an arrow or hatching under lettering that must
    sit on the drawing (contour values among flow arrows, a value on its
    curve) stops short of it, as on a drawing (BACKLOG GM-194). Use it
    where no clear place exists; a label that can sit clear should."""
    from matplotlib import patheffects
    for text in texts:
        text.set_path_effects([patheffects.withStroke(linewidth=2.5,
                                                      foreground=colour)])
