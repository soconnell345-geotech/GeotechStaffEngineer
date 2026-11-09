"""The chart stack both pages draw with (INF-31, the app split's step 6;
in the shared core since step 11 (b), INF-32, with the old
``funhouse_agent.charts`` names as aliases).

``plot_data`` and the plotting box draw through these modules, which load
nothing of the geotechnical toolkit:

* :mod:`~funhouse_agent.charts.data_plot`: the x/y data plot behind
  ``plot_data`` (was ``profile_figure.data_plot``);
* :mod:`~funhouse_agent.charts.plot_tool`: the ``plot_data`` method itself,
  its parameters and its call (was half of the geotech
  ``profile_figure`` adapter);
* :mod:`~funhouse_agent.charts.layout_ink`: the layout check by drawn
  pixels (was ``geotech_common.layout_ink``);
* :mod:`~funhouse_agent.charts.plan_furniture`: the scale bar and north
  arrow (was ``geotech_common.plan_furniture``);
* :mod:`~funhouse_agent.charts.plotting`: ``get_pyplot`` and
  ``figure_to_png``;
* :mod:`~funhouse_agent.charts.figure_examples`: the figure helper's worked
  examples (was ``profile_figure.figure_examples``; OWN-77 (b)).

The old names are the same modules (or functions), so every existing import
keeps working. The standard chart lines of the reference library are not
here: they are the geotechnical app's, which hands them in with
:func:`register_reference_charts` (``plot_data``'s ``reference_chart`` and
the plotting box's ``chart_lines``). Nothing here imports them.

This registration is the core's chart-lines hook (plan section 2.8, OWN-77
(b)). It takes a provider OBJECT rather than an
:mod:`aec_agent_core.hooks` string, because the provider is four things at
once (a module the box loads by file, an overlay, a parameter entry and a
result note), and the geotechnical builds hand it in again on every build.
GSE fills it from
:func:`funhouse_agent.adapters.profile_figure_adapter.register_reference_charts`
(on that adapter's import, and from each geotech build and, until step 13,
the review build). With nothing registered the stack draws as before,
without ``reference_chart`` and without ``chart_lines`` in the box.
"""

from typing import Any, Optional

#: The reference charts registered by the app that has them (None: none).
_REFERENCE_CHARTS: Optional[Any] = None


def register_reference_charts(provider: Optional[Any]) -> None:
    """Hand in the standard chart lines (``None`` takes them away).

    ``provider`` has ``chart_lines``, a module with ``chart_names()``,
    ``chart_lines(name, x_range=, y_range=)`` and ``line_value(name, line,
    x)`` that the plotting box can load from its file (``__file__``);
    ``overlay(params) -> (params, n_added, said)``, what ``plot_data``'s
    ``reference_chart`` draws under the data; ``parameter``, that argument's
    entry in the method's parameters; and ``returns``, what its result says
    of it. The geotechnical app registers the reference library's
    (:func:`funhouse_agent.adapters.profile_figure_adapter.register_reference_charts`).
    """
    global _REFERENCE_CHARTS
    _REFERENCE_CHARTS = provider


def reference_charts() -> Optional[Any]:
    """The registered reference charts, or None."""
    return _REFERENCE_CHARTS
