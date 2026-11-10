"""Hooks: what an app hands the core in place of an import of its own code.

The core imports neither app (BACKLOG INF-32; plan
``module_work/APP_SPLIT_PLAN.md`` section 2.7 B). Where a core module used
to reach into an app (a Key Vault lookup by the sign-in name, the reference
library's folders, a geotech reader), it now asks for a hook by name, and
the app fills the hook::

    from aec_agent_core.hooks import set_hook
    set_hook("prompter.resolve_for", "mypage.keys:resolve_for")

A hook is a callable, or a ``"module:attribute"`` string that is imported
the first time the hook is asked for (so an app can fill its hooks from its
package's ``__init__`` without importing anything there). An unfilled hook
is ``None``, and the core module that asks says what it does without it.

The hooks step 11 (a) defines (each documented where it is asked for):

- ``prompter.get_setting``, ``prompter.resolve_for``,
  ``prompter.tester_keys_possible``: :mod:`aec_agent_core.tinyapps_engine`
  (the Prompter connection; the page hands in how its settings are read and
  whose key serves a caller);
- ``vision_tools.reference_read_roots``,
  ``vision_tools.read_reference_figure``,
  ``vision_tools.view_worked_example_source``:
  :mod:`aec_agent_core.vision_tools` (an app's own reference library: the
  folders the read tools may also read, and two readers of it).

Step 11 (c)'s:

- ``held_records.record_fields``: :mod:`aec_agent_core.deep.held_records`
  (the shapes of the records a stopped helper held);
- ``cited_documents.reference_modules``:
  :mod:`aec_agent_core.cited_documents` (the app's reference modules);
- ``review_eval.page_host``, ``review_eval.open_tasks``,
  ``review_eval.document_dirs``, ``review_eval.fetch_document``,
  ``review_eval.fixture``: the review suite's framework
  (:mod:`aec_agent_core.review_eval`: the page whose turn a task runs on,
  the app's tasks, and where their documents are found).

Step 13's (INF-33):

- ``app.details``: :mod:`aec_agent_core.app_details` (which application
  runs on the core: its name, distribution and package, for the run
  record's version, the default Word and markup authors and the runtime
  messages; filled with :func:`aec_agent_core.app_details.hand_in`).

Step 11 (b)'s chart lines (an app's standard chart lines for ``plot_data``
and the plotting box) are handed in through
:func:`aec_agent_core.charts.register_reference_charts`, a registration
that takes a provider object, not through this table.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Callable, Dict, Optional, Union

__all__ = ["set_hook", "clear_hook", "get_hook", "hooks"]

Target = Union[str, Callable[..., Any]]

_LOCK = threading.Lock()
#: hook name -> what the app handed in (a callable or "module:attribute").
_HOOKS: Dict[str, Target] = {}
#: hook name -> the callable a string target resolved to.
_RESOLVED: Dict[str, Callable[..., Any]] = {}


def _check(name: object, target: object) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"a hook's name must be a non-empty string: {name!r}")
    if isinstance(target, str):
        module, sep, attr = target.partition(":")
        if not (sep and module and attr
                and all(p.isidentifier() for p in module.split("."))
                and all(p.isidentifier() for p in attr.split("."))):
            raise ValueError(
                f"hook {name!r}: a string target is 'module:attribute', "
                f"got {target!r}")
    elif not callable(target):
        raise ValueError(f"hook {name!r}: the target must be callable or a "
                         f"'module:attribute' string, got {target!r}")


def set_hook(name: str, target: Target) -> None:
    """Fill (or re-fill) the hook ``name`` with ``target``."""
    _check(name, target)
    with _LOCK:
        _HOOKS[name] = target
        _RESOLVED.pop(name, None)


def clear_hook(name: str) -> None:
    """Empty the hook ``name`` (no error if it was empty)."""
    with _LOCK:
        _HOOKS.pop(name, None)
        _RESOLVED.pop(name, None)


def get_hook(name: str) -> Optional[Callable[..., Any]]:
    """The callable filling ``name``, or None when nothing fills it.

    A string target is imported here, the first time; an import error is
    raised to the caller (which decides whether the hook is optional).
    """
    with _LOCK:
        target = _HOOKS.get(name)
        resolved = _RESOLVED.get(name)
    if target is None:
        return None
    if resolved is not None:
        return resolved
    if callable(target):
        return target
    module, _, attr = target.partition(":")
    value: Any = importlib.import_module(module)
    for part in attr.split("."):
        value = getattr(value, part)
    with _LOCK:
        if _HOOKS.get(name) is target:
            _RESOLVED[name] = value
    return value


def hooks() -> Dict[str, Target]:
    """A copy of what fills each hook (for tests and diagnostics)."""
    with _LOCK:
        return dict(_HOOKS)
