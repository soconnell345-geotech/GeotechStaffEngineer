"""The figure test set (BACKLOG GM-314): about twenty figure requests drawn
by the geotech page with model-written plot code OFF (the baseline: the
figure helper and the tool figures only) and ON, each run's figures marked
against the figure checklist by a vision model, compared item by item, and
laid side by side for the owner's look.

No measurement of figure quality existed, and none with and without plot
code. The suite never calls a model in its own tests; the paid run happens
on a host that has the models (``module_work/gm314/
FOUNDRY_SECTION_FIGURE_SET.md``).

* :mod:`~funhouse_agent.figure_eval.request_set`: the requests (on GM-328's
  sample inputs) and the data folder they travel in;
* :mod:`~funhouse_agent.figure_eval.checklist`: the items and the vision
  check;
* :mod:`~funhouse_agent.figure_eval.compare`: better / same / worse;
* :mod:`~funhouse_agent.figure_eval.report`: RESULTS.md, results.json and
  the side-by-side page;
* :mod:`~funhouse_agent.figure_eval.runner`: :func:`score_figure_set` and
  the offline dry run.

The names below are loaded on first use, so ``python -m`` on a submodule
runs it once.
"""

import importlib

_WHERE = {"score_figure_set": "runner", "dry_run": "runner",
          "ARM_ENV": "runner", "ARMS": "runner", "prepare": "request_set",
          "load_data": "request_set", "check_run": "checklist",
          "ITEMS": "checklist"}


def __getattr__(name):
    mod = _WHERE.get(name)
    if mod is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)


__all__ = sorted(_WHERE)
