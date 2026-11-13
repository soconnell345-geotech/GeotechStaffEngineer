"""The Python warnings a method raises reach the agent (BACKLOG GM-316).

WHY. A module method says something is wrong with ``warnings.warn``: a pile
tip below the soil profile, soil layers that stop short of a pile toe, a
cracked-section iteration that did not converge, Cr greater than Cc. Until
GM-316 those went to stderr and the agent never saw them, because nothing
caught them around the handler call in :func:`aec_agent_core.method_call.
run_method`.

HOW. One ``warnings.showwarning`` hook, installed once (and put back on top
when a ``catch_warnings(record=True)`` block has swapped it out for the
length of the block, as pytest does round every test). The hook appends to
a list held in a :class:`contextvars.ContextVar` that :func:`collecting` sets
round the handler call, so two calls running at once in different threads
never see each other's warnings. Outside a call the hook hands every warning
to the hook it replaced, unchanged. ``warnings.catch_warnings`` is NOT put
round the call: it swaps process-wide state while helpers run tool calls at
the same time.

WHICH WARNINGS (:func:`reaches_agent`). A warning raised while a handler
runs reaches the agent when

- the code that issued it, or the code it is attributed to (its
  ``stacklevel``), is in one of the app's calculation packages, handed in
  through the hook ``method_call.warning_packages`` (a callable returning
  module-name prefixes; GSE fills it in ``funhouse_agent/__init__.py``), or
  is :mod:`aec_agent_core.method_call` itself (a module's ``stacklevel`` set
  one too high lands on the frame that called the handler); or
- it is numpy's floating-point RuntimeWarning (overflow, invalid value,
  divide by zero), wherever it is attributed: it means a NaN or an infinity
  in a calculation;

and never when it is a DeprecationWarning, PendingDeprecationWarning,
FutureWarning, ImportWarning, ResourceWarning, SyntaxWarning, BytesWarning
or EncodingWarning, or was issued by a plotting library (matplotlib,
plotly, PIL, kaleido, seaborn), which attributes its layout warnings to the
caller. Any other third-party warning (scipy's own, attributed to scipy) is
left to stderr as before.

REPEATS. Python shows a warning once per code location
(``__warningregistry__``), so a second call would lose it. For a warning
that reaches the agent, the hook takes its key back out of that registry,
so the next call raising it reaches the hook again; nothing else's registry
is touched and no filter is added. Within one call a warning is handed on to
the replaced hook (stderr) once per location, as before.

WHAT THE AGENT GETS (:func:`add_to_result`). The message text only, no file
paths or line numbers; at most :data:`MAX_WARNINGS` a call, then
"and N more". A dict result gets them under ``warnings``, after any entries
the module already returns there, without duplicates, the key moved to the
front so a result cut for size (a tail cut, or the key map's preview) keeps
them. A result that is not a dict is returned unchanged: the dispatch's
methods all return dicts, and a library caller of :func:`run_method` that
gets a list back keeps a list.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
import sys
import threading
import warnings
from typing import Iterable, List, Optional, Tuple

__all__ = ["collecting", "add_to_result", "reaches_agent", "MAX_WARNINGS",
           "ensure_hook"]

#: The most warnings one call returns; the rest are counted.
MAX_WARNINGS = 20

#: Categories that never reach the agent (library and interpreter notices).
_NOISE = (DeprecationWarning, PendingDeprecationWarning, FutureWarning,
          ImportWarning, ResourceWarning, SyntaxWarning, BytesWarning,
          EncodingWarning)

#: Libraries whose own warnings never reach the agent, though they name
#: the caller's line (matplotlib's ``_api.warn_external``).
_PLOTTING = ("matplotlib", "mpl_toolkits", "plotly", "PIL", "kaleido",
             "seaborn")

#: The core's own module a handler is called from.
_CORE_ALWAYS = ("aec_agent_core.method_call",)

#: numpy's floating-point error warnings (``np.seterr`` "warn").
_NUMPY_FP = re.compile(
    r"^(overflow|invalid value|divide by zero) encountered\b")

#: The modules a warning passes through on its way to ``showwarning``.
_WARNING_MODULES = frozenset({"warnings", "_py_warnings"})

_HOOK_NAME = "method_call.warning_packages"


class _Bucket:
    """One call's warnings."""

    __slots__ = ("texts", "passed", "parent")

    def __init__(self, parent: Optional["_Bucket"]):
        self.texts: List[str] = []
        self.passed: set = set()
        self.parent = parent

    def add(self, text: str) -> None:
        if text not in self.texts:
            self.texts.append(text)


_BUCKET: contextvars.ContextVar = contextvars.ContextVar(
    "aec_call_warnings", default=None)
#: Set while one of these hooks runs, so a hook it hands on to that is
#: another of them (a hook installed over a hook installed over ours) only
#: hands on.
_IN_HOOK: contextvars.ContextVar = contextvars.ContextVar(
    "aec_call_warnings_in_hook", default=False)
_LOCK = threading.Lock()


def _packages() -> Tuple[str, ...]:
    """The module-name prefixes whose warnings reach the agent: the core's
    own plus the app's, from the hook (an unfilled or failing hook adds
    none)."""
    try:
        from aec_agent_core.hooks import get_hook
        fn = get_hook(_HOOK_NAME)
        extra = tuple(str(p) for p in (fn() if fn else ()) if p)
    except Exception:                                  # noqa: BLE001
        extra = ()
    return _CORE_ALWAYS + extra


def _under(name: Optional[str], prefixes: Iterable[str]) -> bool:
    if not name:
        return False
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _module_of(frame) -> Optional[str]:
    try:
        return frame.f_globals.get("__name__")
    except Exception:                                  # noqa: BLE001
        return None


def _frames(frame, filename: str, lineno: int):
    """(issuing frame, attributed frame) for a warning whose hook was
    called from ``frame``: the first frame outside the warnings machinery
    and these hooks, and the frame at the warning's file and line (its
    ``stacklevel``) at or above it, or None."""
    f = frame
    while f is not None and (_module_of(f) in _WARNING_MODULES
                             or f.f_code is _HOOK_CODE):
        f = f.f_back
    issuer = f
    attributed = None
    while f is not None:
        if f.f_code.co_filename == filename and f.f_lineno == lineno:
            attributed = f
            break
        f = f.f_back
    return issuer, attributed


def reaches_agent(text: str, category, issuer_module: Optional[str],
                  attributed_module: Optional[str],
                  packages: Optional[Iterable[str]] = None) -> bool:
    """Whether a warning raised while a handler runs reaches the agent (the
    rule in this module's docstring). ``packages`` defaults to the core's
    own module plus the app's (the hook)."""
    try:
        if issubclass(category, _NOISE):
            return False
    except TypeError:
        return False
    if _under(issuer_module, _PLOTTING):
        return False
    prefixes = _packages() if packages is None else tuple(packages)
    if _under(issuer_module, prefixes) or _under(attributed_module, prefixes):
        return True
    return issubclass(category, RuntimeWarning) and bool(
        _NUMPY_FP.match(text or ""))


def _make_hook(previous):
    """A ``showwarning`` that records a warning for the call running in
    this context, if any, and hands on to ``previous``."""

    def showwarning(message, category, filename, lineno, file=None,
                    line=None):
        bucket = _BUCKET.get()
        if bucket is None or _IN_HOOK.get():
            return previous(message, category, filename, lineno, file, line)
        token = _IN_HOOK.set(True)
        try:
            text = str(message)
            issuer, attributed = _frames(sys._getframe(1), filename, lineno)
            if reaches_agent(text, category, _module_of(issuer),
                             _module_of(attributed)):
                bucket.add(text)
                # The next call raising it reaches this hook again: the
                # default action's key (text, category, lineno) out of the
                # registry of the module it is attributed to.
                registry = (attributed.f_globals.get("__warningregistry__")
                            if attributed is not None else None)
                if isinstance(registry, dict):
                    registry.pop((text, category, lineno), None)
                where = (text, category, filename, lineno)
                if where in bucket.passed:
                    return None
                bucket.passed.add(where)
            return previous(message, category, filename, lineno, file, line)
        finally:
            _IN_HOOK.reset(token)

    showwarning._aec_call_warnings = True
    showwarning._aec_previous = previous
    return showwarning


_HOOK_CODE = _make_hook(lambda *a: None).__code__


def ensure_hook() -> None:
    """Install the hook over whatever ``warnings.showwarning`` is, unless it
    is one of these hooks already."""
    if getattr(warnings.showwarning, "_aec_call_warnings", False):
        return
    with _LOCK:
        current = warnings.showwarning
        if not getattr(current, "_aec_call_warnings", False):
            warnings.showwarning = _make_hook(current)


@contextlib.contextmanager
def collecting():
    """Record the warnings that reach the agent while the block runs, in
    this context only; yields the bucket (``.texts``). A block inside
    another passes its warnings on to the outer one when it ends."""
    ensure_hook()
    bucket = _Bucket(_BUCKET.get())
    token = _BUCKET.set(bucket)
    try:
        yield bucket
    finally:
        _BUCKET.reset(token)
        if bucket.parent is not None:
            for text in bucket.texts:
                bucket.parent.add(text)


def capped(texts: List[str]) -> List[str]:
    """``texts`` cut to :data:`MAX_WARNINGS`, the rest counted."""
    if len(texts) <= MAX_WARNINGS:
        return list(texts)
    return list(texts[:MAX_WARNINGS]) + [
        f"and {len(texts) - MAX_WARNINGS} more"]


def add_to_result(result, texts: List[str]):
    """``result`` with ``texts`` (one call's warnings) added under
    ``warnings`` (this module's docstring); unchanged when there are none or
    it is not a dict."""
    if not texts or not isinstance(result, dict):
        return result
    have = result.get("warnings")
    if isinstance(have, list):
        merged = list(have)
    elif have is None:
        merged = []
    else:
        merged = [have]
    known = {str(w).strip() for w in merged}
    new = [t for t in texts if t.strip() not in known]
    if not new:
        return result
    merged.extend(capped(new))
    out = {"warnings": merged}
    out.update((k, v) for k, v in result.items() if k != "warnings")
    return out
