"""A summary that fails keeps the history (BACKLOG TA-24, the summarizer).

deepagents' summarizer -- the one ``create_deep_agent`` builds into the main
agent, the general-purpose helper and every helper spec -- replaces the
older part of a long conversation with a summary written by a model call of
its own. That call sits outside the agents' busy ladder
(:mod:`aec_agent_core.deep.busy_retry`), and when it fails the two library
generations go wrong in different ways:

* **langchain before 1.4** (deepagents 0.6.x) catches the error and hands
  back ``"Error generating summary: <error>"`` AS the summary. deepagents
  puts that line in place of the older history, stores it in the state
  (``_summarization_event``) and hands it to the model at every later call:
  the turn goes on, having silently lost its earlier work.
* **langchain 1.4** (deepagents 0.7.x) asks again on its own
  (``Runnable.with_retry``) and then raises. deepagents does not catch it,
  so the model call, and the turn, end on it.

The guard gives both the same outcome: **a summary that fails is not
used.** The library's own wrap is asked again for that one call with its
summarizing switched off, so the model is handed exactly what it would have
been handed had the history stayed under the threshold -- the history as it
was -- and nothing is written to the state; the library tries to summarize
again at its next call. A ``summary_failed`` event goes on the record
(:data:`EVENT`). If that call is itself too long for the model, the library
summarizes regardless of the threshold; if that summary fails too, the call
ends on the error, and the history is still kept.

**How a failure is read.** langchain before 1.4 has no constant or code for
it, only the text of its ``except`` branch: the summary's exact prefix
:data:`ERROR_PREFIX`. langchain 1.4 by its code path: an exception raised
from ``_create_summary``.

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
from typing import Optional

from aec_agent_core.deep.tool_guards import GraphBubbleUp

#: How langchain before 1.4 hands back a summary whose model call failed
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
#: The summarizer asked again with its summarizing switched off.
_holding: contextvars.ContextVar = contextvars.ContextVar(
    "aec_summary_holding", default=None)


class SummaryFailed(Exception):
    """A guarded summary failed. ``error`` is the exception (langchain 1.4)
    or ``None`` (langchain before 1.4 hands back only its error line);
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
    except Exception as exc:  # noqa: BLE001 - langchain 1.4's failure
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
        except Exception as exc:  # noqa: BLE001 - langchain 1.4's failure
            raise SummaryFailed(self, exc) from exc
        if failed_summary(summary):
            raise SummaryFailed(self)
        return summary
    return _acreate_summary


def _guard_trigger(orig):
    """A trigger (``_should_summarize``, ``_over_budget``) that says "no"
    while its summarizer is asked again after a failed summary."""
    @functools.wraps(orig)
    def trigger(self, *args, **kwargs):
        if _holding.get() is self:
            return False
        return orig(self, *args, **kwargs)
    return trigger


def _again_failed(self, failed: SummaryFailed):
    """The call asked again failed to summarize too: end on the error."""
    if failed.source is self and failed.error is not None:
        raise failed.error from None
    raise failed


def _guard_wrap(orig):
    @functools.wraps(orig)
    def wrap_model_call(self, request, handler):
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
        hold, token = _holding.set(self), _guarding.set(self)
        try:
            return orig(self, request, handler)
        except SummaryFailed as failed:
            _again_failed(self, failed)
        finally:
            _guarding.reset(token)
            _holding.reset(hold)
    return wrap_model_call


def _guard_awrap(orig):
    @functools.wraps(orig)
    async def awrap_model_call(self, request, handler):
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
        hold, token = _holding.set(self), _guarding.set(self)
        try:
            return await orig(self, request, handler)
        except SummaryFailed as failed:
            _again_failed(self, failed)
        finally:
            _guarding.reset(token)
            _holding.reset(hold)
    return awrap_model_call


#: The methods guarded, and how; all but ``_over_budget`` (deepagents 0.7
#: only) must be there.
_GUARDS = (("wrap_model_call", _guard_wrap),
           ("awrap_model_call", _guard_awrap),
           ("_create_summary", _guard_create),
           ("_acreate_summary", _guard_acreate),
           ("_should_summarize", _guard_trigger))
_OPTIONAL = (("_over_budget", _guard_trigger),)


def _library_class():
    try:
        from deepagents.middleware.summarization import (
            SummarizationMiddleware)
        return SummarizationMiddleware
    except Exception:  # noqa: BLE001 - no deepagents: nothing to guard
        return None


def install(cls=None) -> bool:
    """Guard the summarizer class ``cls`` (deepagents' by default) once, in
    place. True when it is guarded; False, and nothing changed, when the
    class is missing or lacks a method the guard needs."""
    cls = cls if cls is not None else _library_class()
    if cls is None:
        return False
    if cls.__dict__.get(_MARK):
        return True
    if not all(callable(getattr(cls, name, None)) for name, _ in _GUARDS):
        return False
    for name, guard in _GUARDS + _OPTIONAL:
        method = getattr(cls, name, None)
        if callable(method):
            setattr(cls, name, guard(method))
    setattr(cls, _MARK, True)
    return True


__all__ = ["ERROR_PREFIX", "EVENT", "SummaryFailed", "failed_summary",
           "install"]
