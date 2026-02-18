"""
Helper for embedding DM7 figure plots into calculation packages.

Wraps any DM7 ``plot_figure_*()`` function call into a FigureData object
ready for inclusion in a calc package section.

Usage
-----
    from calc_package.dm7_figures import dm7_figure

    fig_data = dm7_figure(
        plot_figure_5_16,
        caption="Consolidation curve for Tv = 0.20",
        Tv=0.20,
    )
    # fig_data is a FigureData ready to append to a CalcSection.items list
"""

from calc_package.data_model import FigureData
from calc_package.renderer import figure_to_base64


def dm7_figure(plot_func, *, title=None, caption="", width_percent=85,
               dpi=150, **kwargs):
    """Call a DM7 plot function and return a FigureData for calc packages.

    Parameters
    ----------
    plot_func : callable
        A DM7 ``plot_figure_*()`` function (e.g. ``plot_figure_5_16``).
    title : str, optional
        Figure title.  Defaults to the plot function's docstring first line.
    caption : str, optional
        Caption shown below the figure in the calc package.
    width_percent : int, optional
        Display width as percentage of page width.  Default 85.
    dpi : int, optional
        Image resolution.  Default 150.
    **kwargs
        Passed through to *plot_func* (query parameters, etc.).
        ``show=False`` is always forced so the plot does not pop up.

    Returns
    -------
    FigureData
        Ready for inclusion in a CalcSection.

    Raises
    ------
    ImportError
        If matplotlib is not available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Call the plot function with show=False
    kwargs["show"] = False
    ax = plot_func(**kwargs)
    fig = ax.get_figure()

    # Convert to base64
    b64 = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)

    # Build title from docstring if not provided
    if title is None:
        doc = getattr(plot_func, "__doc__", "") or ""
        first_line = doc.strip().split("\n")[0].strip()
        # Strip "Reproduce " prefix if present
        if first_line.startswith("Reproduce "):
            first_line = first_line[len("Reproduce "):]
        # Strip trailing period
        title = first_line.rstrip(".")

    return FigureData(
        title=title,
        image_base64=b64,
        caption=caption,
        width_percent=width_percent,
    )
