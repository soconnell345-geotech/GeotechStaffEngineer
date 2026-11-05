"""The two matplotlib helpers every chart in the app draws through.

:func:`get_pyplot` (matplotlib imported where it is first needed) and
:func:`figure_to_png` (a figure to PNG bytes that carry their smallest
lettering). Moved here from ``geotech_common.plotting`` and
``profile_figure.plotting`` (INF-31, the app split's step 6), which import
them from here, so the old names are the same functions.
"""

import io


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


def figure_to_png(fig, dpi: int = 150) -> bytes:
    """Render a matplotlib figure to PNG bytes (white background, tight box)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    data = buf.read()
    buf.close()
    # the PNG carries its smallest lettering, so a PDF built from it can
    # tell how large that prints (GM-168)
    from funhouse_agent.charts.layout_ink import png_with_lettering
    return png_with_lettering(fig, data, dpi)
