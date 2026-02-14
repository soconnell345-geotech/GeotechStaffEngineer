"""
Result visualization for wave equation analysis.

Provides plotting functions for force-time histories, bearing graphs,
and stress envelopes.

All units are SI: kN, m, seconds.
"""

from typing import Optional

import numpy as np

from wave_equation.time_integration import BlowResult
from wave_equation.bearing_graph import BearingGraphResult


def plot_blow_result(result: BlowResult, ax=None):
    """Plot force and velocity at pile head vs time.

    Parameters
    ----------
    result : BlowResult
        Single blow simulation result.
    ax : matplotlib axes, optional
        If None, creates a new figure.

    Returns
    -------
    ax : matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    t_ms = result.time * 1000  # s -> ms

    ax.plot(t_ms, result.pile_head_force, 'b-', label='Force (kN)')
    ax2 = ax.twinx()
    ax2.plot(t_ms, result.pile_head_velocity, 'r--', label='Velocity (m/s)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Force (kN)', color='b')
    ax2.set_ylabel('Velocity (m/s)', color='r')
    ax.set_title(f'Pile Head Response (Rult = {result.R_ultimate:.0f} kN)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_bearing_graph(bg: BearingGraphResult, ax=None,
                       show_stresses: bool = False):
    """Plot bearing graph (capacity vs blow count).

    Parameters
    ----------
    bg : BearingGraphResult
        Bearing graph results.
    ax : matplotlib axes, optional
    show_stresses : bool
        If True, add secondary axis showing max driving stresses.

    Returns
    -------
    ax : matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    valid = bg.blow_counts < 1e5
    ax.plot(bg.blow_counts[valid], bg.R_values[valid], 'bo-',
            linewidth=2, markersize=6)
    ax.set_xlabel('Blow Count (blows/m)')
    ax.set_ylabel('Ultimate Resistance (kN)')
    ax.set_title('Wave Equation Bearing Graph')
    ax.grid(True, alpha=0.3)

    if show_stresses:
        ax2 = ax.twinx()
        ax2.plot(bg.blow_counts[valid], bg.max_comp_stresses[valid] / 1000,
                 'r--', label='Comp Stress (MPa)')
        ax2.plot(bg.blow_counts[valid], bg.max_tens_stresses[valid] / 1000,
                 'g--', label='Tens Stress (MPa)')
        ax2.set_ylabel('Driving Stress (MPa)')
        ax2.legend()

    return ax
