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
