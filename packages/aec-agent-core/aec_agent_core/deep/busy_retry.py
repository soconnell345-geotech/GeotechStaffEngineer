"""A busy model is waited for, not the end of an agent (BACKLOG TA-20).

Field session 2026-10-14: 8 rate-limit refusals. The 5 that hit vision side
calls were asked again by the vision engine's own ladder and all read; the
3 that hit the agents' OWN calls (two helpers and the primary) each ended
their agent, 10-12 s after the call started -- the client's quick retries
and nothing else. The last one ended a 51-minute turn.

:class:`AgentBusyRetry` wraps every model call of an agent (the primary and
each sub-agent; :func:`funhouse_agent.deep.agent.build_deep_agent` puts it
outermost among the app's model middleware). When the call is refused as
busy -- a rate limit, an overload, a 5xx, a dropped connection, read off the
error's type, status and body (:func:`funhouse_agent.deep.vision_engine.
busy_kind`) -- it waits as the provider's ``retry-after`` asks (at most
:data:`MAX_RETRY_AFTER_S`), else 15, 30, 60 and 120 s with some jitter,
and asks again: :data:`DEFAULT_TRIES` tries in all. It works the same on
every engine (the Prompter key route's ``ChatOpenAI``, the Funhouse bridge,
Foundry), because it sits in the agent, not the client; the client's own
quick retries still run inside each try.

* **Never a spent budget.** An AI budget or quota that is used up is not
  busy (:func:`funhouse_agent.error_text.budget_exhausted`): asking again
  cannot fix it, so it is raised at once (as the Prompter client's
  ``no_retry_when_budget_spent`` hook already makes the client do).
* **A bound per turn.** All the time a turn spends on busy refusals -- the
  failed tries and the waits, of every agent in the turn added together --
  is held under ``GEOTECH_BUSY_TURN_MAX_S`` (default 900 s;
  :class:`funhouse_agent.deep.keep_work.TurnScope`), so a dead endpoint
  ends the turn instead of holding it for ever.
* **On the record.** Each wait is a ``model_busy`` event (the kind, the
  HTTP status, the ``retry-after`` asked, the wait, the try) in the
  activity log; an error that outlasts the ladder carries
  ``geotech_busy`` (tries, seconds waited) for the turn's error text.
* **Paced.** Each try first waits for room on its key under the token pacer
  when one is set (:mod:`funhouse_agent.deep.pacing`, BACKLOG TA-22), and
  the pacer is told the tokens the provider counted.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Optional

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

from funhouse_agent.deep.tool_guards import GraphBubbleUp, _ai_of

#: The waits between tries when the provider names none, in seconds.
LADDER_S = (15.0, 30.0, 60.0, 120.0)

#: The longest wait honoured from a ``retry-after`` header.
MAX_RETRY_AFTER_S = 120.0

#: Tries in all (the first and one per rung of the ladder).
TRIES_ENV = "GEOTECH_AGENT_BUSY_TRIES"
DEFAULT_TRIES = len(LADDER_S) + 1

#: The wait (tests replace them).
_sleep = time.sleep
_asleep = asyncio.sleep


def tries() -> int:
    raw = (os.environ.get(TRIES_ENV) or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_TRIES
    except ValueError:
        return DEFAULT_TRIES


def busy(exc: BaseException) -> Optional[str]:
    """The kind of busy refusal ``exc`` is, or ``None`` (including a spent
    budget, never retried)."""
    try:
        from funhouse_agent.deep.vision_engine import busy_kind
        return busy_kind(exc)
    except Exception:  # noqa: BLE001 - unclassifiable: not retried
        return None


def wait_s(exc: BaseException, attempt: int) -> float:
    """How long to wait before try ``attempt + 2``: the provider's
    ``retry-after`` (at most :data:`MAX_RETRY_AFTER_S`), else the ladder's
    rung, +/- 15 % so agents refused together do not return together."""
    try:
        from funhouse_agent.deep.vision_engine import _retry_after_s
        after = _retry_after_s(exc)
    except Exception:  # noqa: BLE001
        after = None
    if after is not None:
        return min(max(after, 1.0), MAX_RETRY_AFTER_S)
    rung = LADDER_S[min(attempt, len(LADDER_S) - 1)]
    return rung * random.uniform(0.85, 1.15)


def refusal_facts(exc: BaseException) -> dict:
    """What a refusal said, for the record: HTTP status, ``retry-after``,
    and the rate-limit headers a provider sends (``x-ratelimit-*``)."""
    out: dict = {}
    try:
        from funhouse_agent.deep.vision_engine import (_chain, _retry_after_s,
                                                       _status_of)
        for e in _chain(exc):
            status = _status_of(e)
            if status is not None:
                out.setdefault("status", status)
            headers = getattr(getattr(e, "response", None), "headers", None)
            if headers is not None:
                try:
                    for k, v in headers.items():
                        lk = str(k).lower()
                        if lk.startswith("x-ratelimit") or lk in (
                                "retry-after", "retry-after-ms"):
                            out.setdefault("headers", {})[lk] = str(v)
                except Exception:  # noqa: BLE001
                    pass
        after = _retry_after_s(exc)
        if after is not None:
            out["retry_after_s"] = round(after, 3)
    except Exception:  # noqa: BLE001
        pass
    return out


def _event(name: str, data: dict) -> None:
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data)
    except Exception:  # noqa: BLE001 - logging never fails a call
        pass


def _annotate(exc: BaseException, n: int, waited: float, kind: str,
              why: str) -> None:
    try:
        exc.geotech_busy = {"tries": n, "waited_s": round(waited, 1),
                            "kind": kind, "why": why}
    except Exception:  # noqa: BLE001 - a frozen exception type
        pass


def _estimate(request) -> int:
    from funhouse_agent.deep.pacing import estimate_tokens
    try:
        return estimate_tokens(getattr(request, "messages", None) or (),
                               getattr(request, "tools", None) or (),
                               getattr(request, "system_message", None))
    except Exception:  # noqa: BLE001
        return 0


def _settle(pacer, ticket, response) -> None:
    if pacer is None:
        return
    try:
        from funhouse_agent.deep.pacing import usage_tokens
        pacer.settle(ticket, usage_tokens(_ai_of(response)))
    except Exception:  # noqa: BLE001
        pass


class AgentBusyRetry(AgentMiddleware):
    """Ask a busy model again on a ladder (see the module docstring)."""

    def __init__(self, max_tries: Optional[int] = None):
        super().__init__()
        self._tries = max_tries

    def _n(self) -> int:
        return tries() if self._tries is None else max(1, int(self._tries))

    def _decide(self, exc, attempt: int, n: int, spent: float,
                waited: float) -> Optional[float]:
        """The wait before the next try, or ``None`` to raise ``exc``."""
        kind = busy(exc)
        if kind is None:
            return None
        if attempt >= n:
            _annotate(exc, attempt, waited, kind, "the ladder ran out")
            return None
        wait = wait_s(exc, attempt - 1)
        from funhouse_agent.deep.keep_work import current_turn
        scope = current_turn()
        if scope is not None and not scope.spend_busy(spent + wait):
            _annotate(exc, attempt, waited, kind,
                      "the turn's time for a busy model ran out")
            return None
        _event("model_busy", {"kind": kind, "attempt": attempt,
                              "tries": n, "wait_s": round(wait, 1),
                              **refusal_facts(exc)})
        return wait

    def wrap_model_call(self, request, handler):
        from funhouse_agent.deep.pacing import pace
        n = self._n()
        attempt, waited = 0, 0.0
        while True:
            pacer, ticket = pace(getattr(request, "model", None),
                                 _estimate(request))
            t0 = time.monotonic()
            try:
                response = handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                attempt += 1
                wait = self._decide(exc, attempt, n,
                                    time.monotonic() - t0, waited)
                if wait is None:
                    raise
                _sleep(wait)
                waited += wait
                continue
            _settle(pacer, ticket, response)
            return response

    async def awrap_model_call(self, request, handler):
        from funhouse_agent.deep.pacing import pace
        n = self._n()
        attempt, waited = 0, 0.0
        while True:
            pacer, ticket = await asyncio.to_thread(
                pace, getattr(request, "model", None), _estimate(request))
            t0 = time.monotonic()
            try:
                response = await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                attempt += 1
                wait = self._decide(exc, attempt, n,
                                    time.monotonic() - t0, waited)
                if wait is None:
                    raise
                await _asleep(wait)
                waited += wait
                continue
            _settle(pacer, ticket, response)
            return response


def retry_policy() -> dict:
    """The waiting, pacing and helper policy in force, for the record
    (BACKLOG EV-14: the field session's record named the model only)."""
    out: dict = {"agent_busy_tries": tries(), "agent_ladder_s": list(LADDER_S),
                 "agent_max_retry_after_s": MAX_RETRY_AFTER_S}
    try:
        from funhouse_agent.deep import keep_work, pacing, vision_engine
        from funhouse_agent.deep.limits import helper_max_model_calls
        out.update({
            "busy_turn_max_s": keep_work.TurnScope().busy_limit_s,
            "max_helpers": keep_work.max_helpers(),
            "helper_max_model_calls": helper_max_model_calls(),
            "tokens_per_minute": pacing.tokens_per_minute() or None,
            "vision_busy_tries": vision_engine.busy_tries(),
            "vision_max_inflight": vision_engine._int_env(
                vision_engine.INFLIGHT_ENV,
                vision_engine.DEFAULT_MAX_INFLIGHT),
            "vision_call_timeout_s": vision_engine.call_timeout_s(),
        })
    except Exception:  # noqa: BLE001 - a record, never a failure
        pass
    return out


def engine_facts(model) -> dict:
    """Which client a chat model is, for the record: its class, model
    name, the client's own retries and timeout, whether streamed replies
    carry usage. Never a key or an address."""
    if model is None:
        return {}
    out: dict = {"class": type(model).__name__}
    for attr, key in (("model_name", "model"), ("model", "model"),
                      ("max_retries", "client_max_retries"),
                      ("request_timeout", "client_timeout_s"),
                      ("timeout", "client_timeout_s"),
                      ("stream_usage", "stream_usage"),
                      ("disable_streaming", "streaming_disabled"),
                      ("streaming_enabled", "streaming")):
        v = getattr(model, attr, None)
        if key not in out and isinstance(v, (str, int, float, bool)) \
                and not isinstance(v, type):
            out[key] = v
    return out


__all__ = ["AgentBusyRetry", "LADDER_S", "MAX_RETRY_AFTER_S", "TRIES_ENV",
           "DEFAULT_TRIES", "tries", "busy", "wait_s", "refusal_facts",
           "retry_policy", "engine_facts"]
