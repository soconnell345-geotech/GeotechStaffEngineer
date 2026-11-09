"""aec-agent-core: what document agents run below their chat pages.

The app split's core (BACKLOG INF-32; ``module_work/APP_SPLIT_PLAN.md``).
Since step 11 (a) it holds the model connections and the vision calls, and
:mod:`aec_agent_core.hooks`, where an app hands in what the core must not
import; since (b), the figure helper, the plotting box and the chart stack
(:mod:`aec_agent_core.charts`, where an app hands in its standard chart
lines); (c) brings the guards and records, and Word and Excel. This module
is the alias helper each move uses, so a moved module's old import path
keeps working until step 16 takes the aliases out:

    # in the old package's __init__.py, e.g. funhouse_agent/__init__.py
    from aec_agent_core import install_aliases
    install_aliases({"funhouse_agent.vision_tools": "aec_agent_core.vision_tools"})

From then on ``import funhouse_agent.vision_tools`` (or ``from
funhouse_agent import vision_tools``) gives the very module object that
``aec_agent_core.vision_tools`` is, with no stub file left at the old path.
So git follows the move as a rename, and a test that patches by the old
name (``monkeypatch.setattr("funhouse_agent.vision_tools.x", ...)``)
patches the real module, which a re-exporting stub would silently not do.

An alias also covers what lies under the name: with the table above,
``funhouse_agent.vision_tools.sub`` is ``aec_agent_core.vision_tools.sub``.

The alias is LAZY: nothing is imported until the old name is looked up (an
import, or ``importlib.util.find_spec``, which imports the target). The module
keeps its own ``__name__``, ``__spec__`` and ``__file__`` (the new ones).
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from typing import Mapping

__version__ = "1.0.0.dev1"

__all__ = ["install_aliases", "remove_aliases", "aliases", "resolve_alias",
           "__version__"]

_LOCK = threading.Lock()
#: old dotted name -> new dotted name, for every installed alias.
_TABLE: dict[str, str] = {}


def _valid_name(name: object) -> bool:
    return (isinstance(name, str) and bool(name)
            and all(part.isidentifier() for part in name.split(".")))


def resolve_alias(fullname: str) -> str | None:
    """The new name an old dotted name stands for, or None if none does.

    The longest matching alias wins, so ``a.b`` -> ``x`` and ``a.b.c`` ->
    ``y`` send ``a.b.c.d`` to ``y.d``.
    """
    with _LOCK:
        table = dict(_TABLE)
    name = fullname
    rest: list[str] = []
    while name:
        new = table.get(name)
        if new is not None:
            return ".".join([new, *reversed(rest)])
        name, _, last = name.rpartition(".")
        rest.append(last)
    return None


class _AliasLoader(importlib.abc.Loader):
    """Hands the import system the already-imported real module."""

    def __init__(self, target: str):
        self._target = target

    def create_module(self, spec):
        return importlib.import_module(self._target)

    def exec_module(self, module):
        # The import system has just set module.__spec__ to the ALIAS spec
        # (it always does, even on an existing module); put the module's own
        # back, so reload, pickle by reference and find_spec see the real
        # module. Nothing else on the module was overwritten: __name__,
        # __loader__, __package__ and __file__ were already set.
        spec = _REAL_SPECS.get(self._target)
        if spec is not None and module is sys.modules.get(self._target):
            module.__spec__ = spec


#: The real spec of every module an alias handed out, kept before the import
#: system overwrote it (see _AliasLoader.exec_module).
_REAL_SPECS: dict[str, object] = {}


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Finds an old (aliased) name and answers with the new module."""

    def find_spec(self, fullname, path=None, target=None):
        new = resolve_alias(fullname)
        if new is None or new == fullname:
            return None
        module = importlib.import_module(new)
        _REAL_SPECS.setdefault(new, module.__spec__)
        is_pkg = hasattr(module, "__path__")
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(new), is_package=is_pkg)


_FINDER = _AliasFinder()


def _ensure_finder() -> None:
    # First in line, so a stale file left at an old path cannot shadow the
    # moved module.
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def install_aliases(table: Mapping[str, str]) -> None:
    """Make each old dotted name import as the module of its new name.

    ``table`` maps old name -> new name, both full dotted module names.
    Installing the same pair twice is harmless; pointing an old name at a
    different new name, aliasing a name to itself, or aliasing a name that
    is already imported as a DIFFERENT module raises ``ValueError``, because
    each would leave two copies of the code or an alias that never applies.
    """
    pairs = dict(table)
    for old, new in pairs.items():
        if not (_valid_name(old) and _valid_name(new)):
            raise ValueError(f"not a dotted module name: {old!r} -> {new!r}")
        if old == new or new.startswith(old + ".") or old.startswith(new + "."):
            raise ValueError(f"an alias cannot point into itself: {old} -> {new}")
    with _LOCK:
        for old, new in pairs.items():
            have = _TABLE.get(old)
            if have is not None and have != new:
                raise ValueError(
                    f"{old} is already an alias of {have}, not {new}")
            loaded = sys.modules.get(old)
            if loaded is not None and loaded is not sys.modules.get(new):
                raise ValueError(
                    f"{old} is already imported as its own module; an alias "
                    f"to {new} would leave two copies")
        _TABLE.update(pairs)
        _ensure_finder()


def remove_aliases(*olds: str) -> None:
    """Take aliases out (all of them when no name is given).

    A module already imported by an old name stays in ``sys.modules`` under
    it; this only stops new imports from resolving. For tests, and for
    step 16.
    """
    with _LOCK:
        for old in (olds or tuple(_TABLE)):
            _TABLE.pop(old, None)
        if not _TABLE and _FINDER in sys.meta_path:
            sys.meta_path.remove(_FINDER)


def aliases() -> dict[str, str]:
    """A copy of the installed aliases, old name -> new name."""
    with _LOCK:
        return dict(_TABLE)
