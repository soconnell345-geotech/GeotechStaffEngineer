"""The Python warnings a method raises reach the agent (BACKLOG GM-316).

WHY. A module method says something is wrong with ``warnings.warn``: a pile
tip below the soil profile, soil layers that stop short of a pile toe, a
cracked-section iteration that did not converge, Cr greater than Cc. Until
GM-316 those went to stderr and the agent never saw them, because nothing
caught them around the handler call in :func:`aec_agent_core.method_call.
run_method`.

HOW. Two pieces, both installed once when :mod:`aec_agent_core.method_call`
is imported (and again by each call if something replaced them), both
writing to a list held in a :class:`contextvars.ContextVar` that
:func:`collecting` sets round the handler call, so two calls running at once
in different threads never see each other's warnings:

- a thin wrapper round ``warnings.warn``. While a call is collecting, a
  warning the app's packages raise (the rule below, less its numpy part) is
  recorded for that call BEFORE the process's filters and once-per-location
  registry act, and then handed to the original ``warn`` unchanged (same
  arguments, its ``stacklevel`` one deeper for the wrapper's own frame, so
  it is attributed, filtered and shown exactly as before). So no filter in
  the process stops the agent getting it: not an "ignore everything" filter
  left at the front of ``warnings.filters`` by two ``catch_warnings`` blocks
  that interleaved across threads (``catch_warnings`` swaps process-wide
  state; when A enters, B enters, A exits, B exits, B puts back A's copy,
  ignore filter and all, for the life of the process: GM-316 review A1),
  not a library's own ``catch_warnings`` block that happens to be open in
  another thread, and not the registry having seen the same text at the
  same line outside a call. Outside a call the wrapper only hands on.
- a ``warnings.showwarning`` hook (put back on top when a
  ``catch_warnings(record=True)`` block has swapped it out, as pytest does
  round every test), for what does not pass through the Python ``warn``:
  numpy's floating-point warnings, raised from C, and code that bound
  ``warn`` to a name of its own before the wrapper was installed. These DO
  depend on the filters: an "ignore" filter in front drops them. Outside a
  call the hook hands every warning to the hook it replaced, unchanged; a
  warning the wrapper already recorded is handed on unchanged too.

``warnings.catch_warnings`` is NOT put round the call, for the reason
above.

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

REPEATS. The wrapper records a warning on every call that raises it, the
registry notwithstanding, and leaves the registry alone: on stderr a
warning the wrapper recorded is shown once per location for the life of
the process, as before GM-316, and a ``simplefilter("always")`` recorder
(``pytest.warns``) sees every occurrence. For a warning only the hook sees
(numpy's), Python's once-per-location registry would hide it from a second
call, so the hook takes its key back out of the registry of the module it
is attributed to; within one call such a warning is handed on to the
replaced hook (stderr) once per location. Nothing else's registry is
touched and no filter is added. Like any replaced ``showwarning``, the hook
does not pass on a ResourceWarning's ``source`` (its allocation
traceback) to Python's default one.

THREADS. A worker a handler starts with ``contextvars.copy_context().run``
(the core's vision tools do) runs in the call's context, so its warnings
are recorded while the call runs; one that finishes after the call has
returned loses its warnings to the agent (stderr keeps them). A plain
``threading.Thread`` or pool worker has no context of the call, so its
warnings stay on stderr only (on a free-threaded 3.14 build, whose threads
inherit the context, they are recorded too).

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
    """One call's warnings: their texts in order, without repeats, and
    the module-name prefixes that count as the app's for this call."""

    __slots__ = ("texts", "seen", "passed", "parent", "prefixes")

    def __init__(self, parent: Optional["_Bucket"]):
        self.texts: List[str] = []
        self.seen: set = set()
        self.passed: set = set()
        self.parent = parent
        self.prefixes: Tuple[str, ...] = _packages()

    def add(self, text: str) -> None:
        if text not in self.seen:
            self.seen.add(text)
            self.texts.append(text)


_BUCKET: contextvars.ContextVar = contextvars.ContextVar(
    "aec_call_warnings", default=None)
#: Set while one of these hooks runs, so a hook it hands on to that is
#: another of them (a hook installed over a hook installed over ours) only
#: hands on.
_IN_HOOK: contextvars.ContextVar = contextvars.ContextVar(
    "aec_call_warnings_in_hook", default=False)
#: Set while the ``warn`` wrapper hands a warning it has recorded on to the
#: original ``warn``: the ``showwarning`` hook then only hands it on.
_RECORDED: contextvars.ContextVar = contextvars.ContextVar(
    "aec_call_warnings_recorded", default=False)
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
                             or f.f_code is _HOOK_CODE
                             or f.f_code is _WARN_CODE):
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
    prefixes = _packages() if packages is None else tuple(packages)
    if _is_own(category, issuer_module, attributed_module, prefixes):
        return True
    try:
        if issubclass(category, _NOISE):
            return False
    except TypeError:
        return False
    if _under(issuer_module, _PLOTTING):
        return False
    return issubclass(category, RuntimeWarning) and bool(
        _NUMPY_FP.match(text or ""))


def _is_own(category, issuer_module: Optional[str],
            attributed_module: Optional[str], prefixes) -> bool:
    """The first part of the rule: issued by, or attributed to, one of the
    ``prefixes`` (the app's packages and the core's own module), and not a
    noise category or a plotting library's."""
    try:
        if issubclass(category, _NOISE):
            return False
    except TypeError:
        return False
    if _under(issuer_module, _PLOTTING):
        return False
    return _under(issuer_module, prefixes) or _under(attributed_module,
                                                     prefixes)


def _category_and_text(message, category):
    """``(category, text)`` of a ``warnings.warn`` call as Python reads
    them, or None where ``warn`` would refuse the call."""
    if isinstance(message, Warning):
        return message.__class__, str(message)
    if category is None:
        category = UserWarning
    if not (isinstance(category, type) and issubclass(category, Warning)):
        return None
    return category, str(message)


def _skipped(frame, skip) -> bool:
    return bool(skip) and frame.f_code.co_filename.startswith(skip)


def _warned_frame(caller, stacklevel, skip):
    """The frame ``warnings.warn`` attributes a warning to when ``caller``
    calls it with ``stacklevel`` and ``skip_file_prefixes`` ``skip``
    (stacklevel 1 is ``caller``; frames in a skipped file are passed
    over), or None past the top of the stack."""
    n = stacklevel if isinstance(stacklevel, int) else 1
    if skip:
        n = max(n, 2)
    f = caller
    while n > 1 and f is not None:
        f = f.f_back
        while f is not None and _skipped(f, skip):
            f = f.f_back
        n -= 1
    return f


def _deeper(stacklevel, skip, caller):
    """The ``stacklevel`` to hand the original ``warn`` from the wrapper's
    frame so it attributes the warning to the frame it would have from
    ``caller``: one deeper, except where Python passes over ``caller``
    itself (a ``skip_file_prefixes`` naming its file)."""
    if not isinstance(stacklevel, int):
        return stacklevel                  # the original refuses it as before
    try:
        if skip:
            stacklevel = max(stacklevel, 2)
            return stacklevel if _skipped(caller, skip) else stacklevel + 1
    except Exception:                                  # noqa: BLE001
        pass
    return max(stacklevel, 1) + 1


def _record(bucket: _Bucket, message, category, stacklevel, skip,
            caller) -> bool:
    """Record a ``warnings.warn`` call made from ``caller`` for the call
    ``bucket`` collects, when the rule counts it as the app's own; whether
    it did."""
    got = _category_and_text(message, category)
    if got is None:
        return False
    category, text = got
    issuer = _module_of(caller)
    prefixes = bucket.prefixes
    if not _is_own(category, issuer, None, prefixes):
        attributed = _warned_frame(caller, stacklevel, skip)
        if not _is_own(category, issuer, _module_of(attributed), prefixes):
            return False
    bucket.add(text)
    return True


def _make_warn(original):
    """A ``warnings.warn`` that records an app's warning for the call
    running in this context, if any, before any filter acts, then hands
    the call to ``original`` unchanged (its ``stacklevel`` one deeper for
    this frame)."""

    def warn(message, category=None, stacklevel=1, source=None, **kwargs):
        caller = sys._getframe(1)
        skip = kwargs.get("skip_file_prefixes") or ()
        level = _deeper(stacklevel, skip, caller)
        bucket = _BUCKET.get()
        if bucket is None or _IN_HOOK.get() or _RECORDED.get():
            return original(message, category, level, source, **kwargs)
        try:
            recorded = _record(bucket, message, category, stacklevel, skip,
                               caller)
        except Exception:                              # noqa: BLE001
            recorded = False
        if not recorded:
            return original(message, category, level, source, **kwargs)
        token = _RECORDED.set(True)
        try:
            return original(message, category, level, source, **kwargs)
        finally:
            _RECORDED.reset(token)

    warn._aec_call_warnings = True
    warn._aec_original = original
    warn.__doc__ = getattr(original, "__doc__", None)
    return warn


_WARN_CODE = _make_warn(lambda *a, **k: None).__code__


def _make_hook(previous):
    """A ``showwarning`` that records a warning for the call running in
    this context, if any, and hands on to ``previous``."""

    def showwarning(message, category, filename, lineno, file=None,
                    line=None):
        bucket = _BUCKET.get()
        if bucket is None or _IN_HOOK.get() or _RECORDED.get():
            return previous(message, category, filename, lineno, file, line)
        token = _IN_HOOK.set(True)
        try:
            text = str(message)
            issuer, attributed = _frames(sys._getframe(1), filename, lineno)
            if reaches_agent(text, category, _module_of(issuer),
                             _module_of(attributed), bucket.prefixes):
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
    """Install the ``warnings.warn`` wrapper over whatever ``warnings.warn``
    is, and the hook over whatever ``warnings.showwarning`` is, each unless
    it is one of these already. ``catch_warnings`` never swaps ``warn``, so
    the wrapper stays once installed; the hook may be swapped out by a
    ``catch_warnings`` block and is put back by the next call."""
    if (getattr(warnings.warn, "_aec_call_warnings", False)
            and getattr(warnings.showwarning, "_aec_call_warnings", False)):
        return
    with _LOCK:
        if not getattr(warnings.warn, "_aec_call_warnings", False):
            warnings.warn = _make_warn(warnings.warn)
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


def so_far() -> List[str]:
    """The warnings recorded so far in the call this code runs in (the
    innermost :func:`collecting` block, with what the blocks inside it that
    have ended passed on), in order; empty outside a call. A calc package
    prints them before it is written (GM-316 review B4)."""
    bucket = _BUCKET.get()
    return list(bucket.texts) if bucket is not None else []


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
