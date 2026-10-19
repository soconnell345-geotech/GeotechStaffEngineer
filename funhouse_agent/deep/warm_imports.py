"""Imports finished before an agent's tools run side by side (BACKLOG GM-172).

LangGraph runs the tool calls of one model turn at once, in threads. In live
wave w13 (F18 [45]/[48]) a calc helper drew a calc package's figures and a
subsurface profile in the same turn; both tools made the process's first
``import matplotlib.pyplot`` at once, and one failed with "partially
initialized module 'matplotlib.pyplot'" (Python's import lock hands a module
still being imported to a second thread when the two imports wait on each
other). The retry worked; the failed call cost a model call.

:func:`warm_pyplot` makes that first import, and the Agg backend's, once,
under a lock, when the agent is built -- before any tool runs.
"""

from __future__ import annotations

import sys
import threading

_LOCK = threading.Lock()
_warm = False


def warm_pyplot() -> bool:
    """Import ``matplotlib.pyplot`` and its backend once, choosing Agg when
    nothing else chose a backend (``MPLBACKEND``, ``matplotlib.use``).
    Returns True when pyplot is imported; never raises."""
    global _warm
    if _warm:
        return True
    with _LOCK:
        if _warm:
            return True
        try:
            import matplotlib
            fresh = "matplotlib.pyplot" not in sys.modules
            if fresh:
                chosen = getattr(matplotlib.rcParams,
                                 "_get_backend_or_none", lambda: None)()
                if chosen is None:
                    # Spelt as the tools spell it, so their own
                    # matplotlib.use("Agg") finds it set and never reaches
                    # into pyplot (the w13 failure did).
                    matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if fresh:
                # The backend module too (pyplot loads it on the first
                # figure); a new pyplot has no figure to close.
                plt.switch_backend(matplotlib.get_backend())
        except Exception:  # noqa: BLE001 - tools then import it themselves
            return False
        _warm = True
        return True


__all__ = ["warm_pyplot"]
