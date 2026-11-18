"""A summary that fails keeps the history (BACKLOG TA-24, the summarizer).

deepagents' summarizer -- the one ``create_deep_agent`` builds into the main
agent, the general-purpose helper and every helper spec -- replaces the
older part of a long conversation with a summary written by a model call of
its own. That call sits outside the agents' busy ladder
(:mod:`aec_agent_core.deep.busy_retry`), and when it fails the two library
generations go wrong in different ways:

* **langchain 1.3.14 and older** catches the error and hands back
  ``"Error generating summary: <error>"`` AS the summary. deepagents
  puts that line in place of the older history, stores it in the state
  (``_summarization_event``) and hands it to the model at every later call:
  the turn goes on, having silently lost its earlier work.
* **langchain 1.3.15 and newer** (1.4 with it) asks again on its own
  (``Runnable.with_retry``) and then raises. deepagents does not catch it,
  so the model call, and the turn, end on it.

The deepagents version does not decide which: deepagents 0.6.8 runs on
either side of that boundary (installed today it resolves langchain 1.3.18,
and raises), deepagents 0.7.x only on the newer side. The guard reads the
behaviour, not a version.

The guard gives both the same outcome: **a summary that fails is not
used.** The library's own wrap is asked again for that one call with its
summarizing switched off, so the model is handed exactly what it would have
been handed had the history stayed under the threshold -- the history as it
was -- and nothing is written to the state; the library tries to summarize
again at the next TURN. A ``summary_failed`` event goes on the record
(:data:`EVENT`), once per failure. If that call is itself too long for the
model, the library summarizes regardless of the threshold; if that summary
fails too, the call ends on the error, and the history is still kept.

**Not again in the same turn.** The library writes the older history to a
file BEFORE it makes the summary, and asks the summary call several times
with a back-off; a failure that lasts would repeat both at every model
call. So once a summary has failed, the later model calls of that turn are
handed the history as it is from the start: no summary request, no
back-off, no new history file. The turn is the core's own
(:class:`aec_agent_core.deep.keep_work.TurnScope`, the scope a host opens
around one turn and LangGraph copies into every thread and task of it), so
two conversations on one agent at the same moment stay apart; inside a turn
the mark is per summarizer (the main agent's, each helper type's) and per
helper run. The next turn has a new scope and tries again. Each call that
was over the threshold and was not summarized for this reason is counted on
the scope (``summaries_skipped`` on the record's ``turn_end``). The one
exception is the library's own: a call the model refuses as too long is
summarized regardless, in any call. Outside a turn scope (a library call)
there is no turn to remember a failure in, and every call tries.

**How a failure is read.** langchain 1.3.14 and older has no constant or
code for it, only the text of its ``except`` branch: the summary's exact
prefix :data:`ERROR_PREFIX`. langchain 1.3.15 and newer by its code path:
an exception raised from ``_create_summary``.

**Where it stands.** On the library's class, not in the middleware list:
the summarizer is built inside ``create_deep_agent``, where no middleware
of ours reaches (ours run inside its wrap, after the summary is made and
before its state write). :func:`install` wraps the class's summary,
trigger and wrap methods once, in place. A call that does not summarize,
and a summary that succeeds, run the library's code unchanged. A deepagents
without those methods is left alone.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
from typing import Optional

from aec_agent_core.deep.tool_guards import GraphBubbleUp

#: How langchain 1.3.14 and older hands back a summary whose model call failed
#: (``SummarizationMiddleware._create_summary``: ``f"Error generating
#: summary: {e!s}"``).
ERROR_PREFIX = "Error generating summary:"

#: The record's event for a summary that failed and was not used.
EVENT = "summary_failed"

#: Set on the class once it is guarded.
_MARK = "_aec_summary_guard"

#: The summarizer whose summary is being made under the guard.
_guarding: contextvars.ContextVar = contextvars.ContextVar(
    "aec_summary_guarding", default=None)
#: ``(summarizer, skipped)``: the summarizer asked with its summarizing
#: switched off. ``skipped`` is ``None`` for the call whose summary just
#: failed, and a list for a later call of the same turn: it gets one entry
#: when that call was over the threshold.
_holding: contextvars.ContextVar = contextvars.ContextVar(
    "aec_summary_holding", default=None)


class SummaryFailed(Exception):
    """A guarded summary failed. ``error`` is the exception (langchain
    1.3.15 and newer) or ``None`` (1.3.14 and older hands back only its
    error line);
    ``source`` is the summarizer whose summary it was."""

    def __init__(self, source, error: Optional[BaseException] = None):
        super().__init__("The earlier part of the conversation could not "
                         "be summarized; it is kept as it was.")
        self.source = source
        self.error = error


def failed_summary(summary) -> bool:
    """True when ``summary`` is langchain's error line, not a summary."""
    return isinstance(summary, str) and summary.startswith(ERROR_PREFIX)


def _event(name: str, data: dict) -> None:
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data)
    except Exception:  # noqa: BLE001 - the record never fails a call
        pass


def _record(failed: SummaryFailed) -> None:
    """The record's plain entry. The provider's error text is left out (it
    may carry a key or a host name); its class name is kept."""
    data = {"history": "kept",
            "what": "the summary of the earlier conversation failed; the "
                    "model was handed the history as it was"}
    if failed.error is not None:
        data["error_type"] = type(failed.error).__name__
    _event(EVENT, data)


def _checked(self, make):
    """Run ``make()`` (the library's summary) and raise
    :class:`SummaryFailed` when it failed, either way."""
    try:
        summary = make()
    except GraphBubbleUp:
        raise
    except Exception as exc:  # noqa: BLE001 - newer langchain raises
        raise SummaryFailed(self, exc) from exc
    if failed_summary(summary):
        raise SummaryFailed(self)
    return summary


def _guard_create(orig):
    @functools.wraps(orig)
    def _create_summary(self, *args, **kwargs):
        if _guarding.get() is not self:
            return orig(self, *args, **kwargs)
        return _checked(self, lambda: orig(self, *args, **kwargs))
    return _create_summary


def _guard_acreate(orig):
    @functools.wraps(orig)
    async def _acreate_summary(self, *args, **kwargs):
        if _guarding.get() is not self:
            return await orig(self, *args, **kwargs)
        try:
            summary = await orig(self, *args, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - newer langchain raises
            raise SummaryFailed(self, exc) from exc
        if failed_summary(summary):
            raise SummaryFailed(self)
        return summary
    return _acreate_summary


def _guard_trigger(orig):
    """A trigger (``_should_summarize``, ``_over_budget``) that says "no"
    while its summarizer is asked with its summarizing switched off. In a
    later call of a turn whose summary failed, what the library's own
    trigger would have said is noted, for the count of calls not
    summarized."""
    @functools.wraps(orig)
    def trigger(self, *args, **kwargs):
        held = _holding.get()
        if held is None or held[0] is not self:
            return orig(self, *args, **kwargs)
        skipped = held[1]
        if skipped is not None and not skipped:
            try:
                if orig(self, *args, **kwargs):
                    skipped.append(True)
            except Exception:  # noqa: BLE001 - the count never fails a call
                pass
        return False
    return trigger


def _again_failed(self, failed: SummaryFailed):
    """The call asked again failed to summarize too: end on the error."""
    if failed.source is self and failed.error is not None:
        raise failed.error from None
    raise failed


def _turn_and_helper():
    """The running turn's scope and the helper run this call belongs to
    (``None``: the main agent); ``(None, None)`` outside a turn."""
    try:
        from aec_agent_core.deep import keep_work
        return keep_work.current_turn(), keep_work.current_helper()
    except Exception:  # noqa: BLE001 - no turn: every call tries
        return None, None


def _failed_this_turn(self) -> bool:
    scope, helper = _turn_and_helper()
    try:
        return scope is not None and scope.summary_failed(self, helper)
    except Exception:  # noqa: BLE001 - a scope of another shape
        return False


def _note_failed(self) -> None:
    scope, helper = _turn_and_helper()
    try:
        if scope is not None:
            scope.note_summary_failed(self, helper)
    except Exception:  # noqa: BLE001
        pass


def _note_skipped(skipped) -> None:
    if not skipped:
        return
    scope, _ = _turn_and_helper()
    try:
        if scope is not None:
            scope.note_summary_skipped()
    except Exception:  # noqa: BLE001
        pass


def _first_try(self):
    """What the hold pass starts with: ``None`` when this call may try its
    summary first, else the list that notes a call not summarized."""
    return [] if _failed_this_turn(self) else None


def _guard_wrap(orig):
    @functools.wraps(orig)
    def wrap_model_call(self, request, handler):
        skipped = _first_try(self)
        if skipped is None:
            token = _guarding.set(self)
            try:
                return orig(self, request, handler)
            except SummaryFailed as failed:
                if failed.source is not self:
                    raise
                first = failed
            finally:
                _guarding.reset(token)
            _record(first)
            _note_failed(self)
        hold, token = _holding.set((self, skipped)), _guarding.set(self)
        try:
            return orig(self, request, handler)
        except SummaryFailed as failed:
            _again_failed(self, failed)
        finally:
            _guarding.reset(token)
            _holding.reset(hold)
            _note_skipped(skipped)
    return wrap_model_call


def _guard_awrap(orig):
    @functools.wraps(orig)
    async def awrap_model_call(self, request, handler):
        skipped = _first_try(self)
        if skipped is None:
            token = _guarding.set(self)
            try:
                return await orig(self, request, handler)
            except SummaryFailed as failed:
                if failed.source is not self:
                    raise
                first = failed
            finally:
                _guarding.reset(token)
            _record(first)
            _note_failed(self)
        hold, token = _holding.set((self, skipped)), _guarding.set(self)
        try:
            return await orig(self, request, handler)
        except SummaryFailed as failed:
            _again_failed(self, failed)
        finally:
            _guarding.reset(token)
            _holding.reset(hold)
            _note_skipped(skipped)
    return awrap_model_call


#: The methods guarded, and how; all but ``_over_budget`` (deepagents 0.7
#: only) must be there.
_GUARDS = (("wrap_model_call", _guard_wrap),
           ("awrap_model_call", _guard_awrap),
           ("_create_summary", _guard_create),
           ("_acreate_summary", _guard_acreate),
           ("_should_summarize", _guard_trigger))
_OPTIONAL = (("_over_budget", _guard_trigger),)


logger = logging.getLogger(__name__)

#: The library's summarizer class, as the guard imports it.
_LIBRARY_CLASS = "deepagents.middleware.summarization.SummarizationMiddleware"

#: The one warning of a process has been written.
_warned = False


def _warn_absent(missing: str) -> None:
    """Say once per process, through ``logging``, that the guard is absent
    and what was missing. The app starts all the same."""
    global _warned
    if _warned:
        return
    _warned = True
    try:
        logger.warning(
            "The summary guard is not installed: %s. A summary that fails "
            "is left to the agent library (BACKLOG TA-24).", missing)
    except Exception:  # noqa: BLE001 - a warning never stops a build
        pass


def _library_class():
    """deepagents' summarizer class; ``None`` without deepagents (nothing
    to guard, nothing said) or when deepagents imports but the class is
    not where the guard knows it (said once)."""
    try:
        import deepagents  # noqa: F401
    except Exception:  # noqa: BLE001 - no deepagents: nothing to guard
        return None
    try:
        from deepagents.middleware.summarization import (
            SummarizationMiddleware)
        return SummarizationMiddleware
    except Exception as exc:  # noqa: BLE001 - the library moved its class
        _warn_absent(f"{_LIBRARY_CLASS} could not be imported "
                     f"({type(exc).__name__})")
        return None


#: Held over the check and the patching, so that two first builds at the
#: same moment (two testers' first turns after a restart) cannot both find
#: the class unguarded and wrap it twice.
_INSTALL_LOCK = threading.Lock()


def install(cls=None) -> bool:
    """Guard the summarizer class ``cls`` (deepagents' by default) once, in
    place. True when it is guarded; False, and nothing changed, when the
    class is missing or lacks a method the guard needs. When deepagents is
    there and its class is not the shape the guard knows, one warning per
    process says what was missing (``logging``, this module's logger)."""
    cls = cls if cls is not None else _library_class()
    if cls is None:
        return False
    with _INSTALL_LOCK:
        if cls.__dict__.get(_MARK):
            return True
        missing = [name for name, _ in _GUARDS
                   if not callable(getattr(cls, name, None))]
        if missing:
            _warn_absent(
                f"{getattr(cls, '__module__', '?')}."
                f"{getattr(cls, '__qualname__', cls)} has no "
                + ", ".join(missing))
            return False
        for name, guard in _GUARDS + _OPTIONAL:
            method = getattr(cls, name, None)
            if callable(method):
                setattr(cls, name, guard(method))
        setattr(cls, _MARK, True)
        return True


__all__ = ["ERROR_PREFIX", "EVENT", "SummaryFailed", "failed_summary",
           "install"]
