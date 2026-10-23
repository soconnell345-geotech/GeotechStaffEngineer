"""Model access for the report-ingest passes, and what a run cost.

Two passes sit between planlens' rule-based page labels and any reader:
:mod:`report_ingest.triage` and :mod:`report_ingest.label_review`. Both are
written against the small :class:`Engine` protocol below rather than against
an SDK, so the same code runs on the Claude API here, on the cluster's
Prompter, and on whatever chat model the app itself was built with --
Foundry's own model classes among them (:class:`ChatModelEngine`, BACKLOG
RI-03). The protocol is one method::

    complete(messages, system=..., tools=..., images=..., output_format=...)

and the reply it returns carries the model's text, the tool calls it wants
run, the parsed structured output when one was asked for, and the tokens the
call spent.

MESSAGES ARE NEUTRAL, NOT ANTHROPIC. A message is ``{"role", "content"}``
where content is a string or a list of the blocks :func:`text_block`,
:func:`image_block`, :func:`tool_use_block`, :func:`tool_result_block` and
:func:`opaque_block` build. The engine translates them to and from its
provider's shape, in both directions, so a pass can append an assistant turn
straight back into its message list without knowing what a provider block
looks like. ``opaque`` is how that works for blocks a pass must carry but
must not read or rewrite -- a thinking block has to be echoed back unchanged
to the model that produced it, and is therefore carried, not inspected.

COST. Every call is metered into a :class:`CostMeter`: calls, input and
output tokens, cache reads and writes, reasoning tokens, wall clock, and
dollars -- and every request that came back with an error instead of a
reply, with its kind, its attempt number and whether the engine sent it
again (BACKLOG RI-21), one row per request in :attr:`CostMeter.per_call`.
Dollars come
from the model that ANSWERED -- ``response.model``, recorded as
``served_by`` -- priced in :data:`MODEL_PRICES` for the Claude API and in
:data:`PROMPTER_PRICES` for the owner's Funhouse deployments; a capability
tier is an alias with no price of its own, so a call whose deployment is
also unpriced reports tokens and nothing else. The owner's rule for this
work is that accuracy on key content beats tokens, so the meter exists to
REPORT what a run cost, not to cut a pass short.

The ``anthropic`` package is imported inside :class:`ClaudeEngine` and
nowhere else, so importing :mod:`report_ingest` costs an app nothing and
adds no dependency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "Engine", "Reply", "ToolCall", "Usage", "CostMeter", "ClaudeEngine",
    "engine_for", "ChatModelEngine", "ANSWER_TOOL", "CUT_REASONS",
    "DEFAULT_MAX_TOKENS", "CallCeilingReached",
    "PrompterEngine", "MODEL_PRICES", "PROMPTER_MODELS", "PROMPTER_PRICES",
    "price_for", "strict_schema", "error_kind", "FAILURE_KINDS",
    "text_block", "image_block", "tool_use_block", "tool_result_block",
    "opaque_block", "user", "assistant",
    "image_media_type", "IMAGE_SIGNATURES",
]

#: List price per MILLION tokens, ``(input, output)``, as published for the
#: Claude API. Kept here rather than fetched: a scorecard has to be able to
#: say what a run cost months later, and a price that moved under it would
#: rewrite history. Update deliberately, with the date, when it changes.
#: (Checked 2026-09-16.)
MODEL_PRICES: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}
#: Writing a prompt-cache entry costs more than plain input; reading one
#: costs a tenth. Both are multipliers on the model's INPUT price.
CACHE_WRITE_RATE = 1.25
CACHE_READ_RATE = 0.10

#: Funhouse publishes models by CAPABILITY TIER, not by name, and the model
#: behind a tier changes without notice. These are the three aliases, and
#: they are what a cluster run should record -- ``response.model`` says which
#: deployment actually served it, and the scorecard keeps that too.
#:
#: A TIER still has no price of its own: :meth:`Usage.dollars` returns 0.0
#: for one rather than inventing a number. What IS priced is the deployment
#: that answered -- see :data:`PROMPTER_PRICES`.
PROMPTER_MODELS: Tuple[str, ...] = (
    "funhouse-gpt-low", "funhouse-gpt-medium", "funhouse-gpt-high",
)

#: Per MILLION tokens, ``(input, output)``, keyed by the DEPLOYMENT that
#: served a call -- ``response.model``, which this package records as
#: ``served_by`` -- and never by the tier that was asked for. **These are the
#: owner's own rates from the Funhouse budget page, read 2026-09-18.** A tier
#: is an alias and the deployment behind it changes without notice, so a run
#: priced by the tier would be priced by a name that means something
#: different next month; pricing by what actually answered is the only
#: version of this that stays true. An embedding model has no output price,
#: and 0.0 is its real one.
#:
#: Update deliberately, with the date, the way :data:`MODEL_PRICES` is
#: updated: a scorecard has to be able to say what a run cost months later,
#: and a price that moved under it would rewrite history.
PROMPTER_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-5.4-2026-03-05": (2.50, 15.00),
    "gpt-5.1-2025-11-13": (1.25, 10.00),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "text-embedding-ada-002": (0.10, 0.00),
}


def _on_foundry() -> bool:
    """Palantir Foundry: its SDK is installed there and nowhere else this
    package runs. Foundry bills under the organisation's enrollment, with no
    per-token rate, so Funhouse's rates must not price a run there (a
    Foundry run of 2026-10-02 printed "$5.85" from them)."""
    import importlib.util
    try:
        return importlib.util.find_spec("palantir_models") is not None
    except (ImportError, ValueError):
        return False


def price_for(model: str) -> Optional[Tuple[float, float]]:
    """``(input, output)`` per million tokens for a model, or ``None``.

    Both tables are consulted -- the Claude list prices this package
    develops against and the owner's Funhouse deployment rates -- because a
    run is metered by whatever name came back on the reply. The Funhouse
    rates apply only off Foundry (:func:`_on_foundry`): the same deployment
    name answers there, at no per-token price.
    """
    if not model:
        return None
    price = MODEL_PRICES.get(model)
    if price is None and not _on_foundry():
        price = PROMPTER_PRICES.get(model)
    return price


#: The output cap a call gets when it names none, on the engines that keep a
#: cap of their own (:class:`ClaudeEngine`, :class:`PrompterEngine`;
#: :class:`ChatModelEngine` takes its model's own). BACKLOG RI-19: it was
#: 16,000, and on a reasoning model the cap holds the reasoning as well as
#: the answer, so one long report's label findings stopped there
#: (``stop_reason 'length'``) and the report failed with them
#: (``module_work/harness_theory/report_ingest.md`` section 8.2, P1). 32,000
#: is the cap every engine the app builds sends
#: (``webapp.engine_config.DEFAULT_MAX_TOKENS``), chosen there within each
#: production model's documented output limit (GPT-5.1, 5.4 and Sol: 128K;
#: GPT-4.1: 32,768; Claude Opus 4.x and later: 32,000 or more). A cap costs
#: nothing until it is used. What happens when a reply still reaches it:
#: :attr:`Reply.cut_off`, and the README's "At the output cap".
DEFAULT_MAX_TOKENS = 32000

#: Finish reasons that mean a reply stopped at its output cap, as OpenAI
#: (``length``), Anthropic (``max_tokens``) and a Responses status
#: (``incomplete``, ``max_output_tokens``) say it.
CUT_REASONS = frozenset({"length", "max_tokens", "max_output_tokens",
                         "incomplete"})


def _reason(value: Any) -> str:
    """A finish reason as lower-case text, an SDK enum's included
    (``FinishReason.LENGTH`` is ``length``)."""
    return str(getattr(value, "value", value) or "").rsplit(".", 1)[-1] \
        .strip().lower()


# -- neutral message blocks -------------------------------------------------

def text_block(text: str) -> Dict[str, Any]:
    """Words for the model to read."""
    return {"type": "text", "text": str(text)}


#: What the leading bytes of a picture say it is. A provider is told the
#: media type explicitly (Claude) or through a data-URI prefix (OpenAI), and
#: a JPEG announced as a PNG is refused by both, so the bytes -- not the
#: caller -- are what decides.
IMAGE_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def image_media_type(data: bytes) -> str:
    """``"image/png"`` or ``"image/jpeg"``, read off the bytes themselves.

    Anything unrecognised is called a PNG, which is what every picture in
    this package was until the pages started travelling as JPEG: an engine
    that mislabels one unfamiliar byte string is a bad request the provider
    names, while an engine that raises here would take down a whole run over
    a picture it could perfectly well have sent.
    """
    head = bytes(data[:8])
    for signature, media in IMAGE_SIGNATURES:
        if head.startswith(signature):
            return media
    return "image/png"


def image_block(data: bytes, detail: Optional[str] = None) -> Dict[str, Any]:
    """A picture for the model to look at -- PNG or JPEG.

    Bytes, not a path: the pages these passes render are held in memory and
    never written to disk, because the corpus is private. **The key is still
    called ``png`` and now holds either**, because every pass and every test
    that builds or reads a block names it; what a block holds is decided by
    :func:`image_media_type` reading the bytes, so a caller cannot mislabel
    one. A page renders as a PNG and is sent as a JPEG
    (:func:`report_ingest.vision_labels.encode_for_vision`): a scanned page
    at 100 dpi is a few hundred kilobytes of PNG, and thirty-six of them
    base64-encoded in one request is what the gateway refuses.

    ``detail`` is OpenAI's ``image_url.detail`` -- ``"low"``, ``"high"`` or
    ``"auto"``. It is carried only when it is set, so a call that does not
    ask for one is byte-for-byte the request it was before. ``"low"`` is the
    lever that matters for a whole-report call: the provider then looks at
    one 512 px tile and charges about 85 tokens instead of tiling the page.
    The Claude engine has no such key and ignores it.
    """
    block: Dict[str, Any] = {"type": "image", "png": bytes(data)}
    if detail:
        block["detail"] = str(detail)
    return block


def tool_use_block(call_id: str, name: str,
                   arguments: Dict[str, Any]) -> Dict[str, Any]:
    """One tool call, as the model asked for it."""
    return {"type": "tool_use", "id": call_id, "name": name,
            "input": dict(arguments)}


def tool_result_block(call_id: str, content: Any,
                      is_error: bool = False) -> Dict[str, Any]:
    """One tool's result.

    ``content`` is a string or a list of blocks, so a tool that answers with
    a picture (``render_page``) returns one the same way a tool that answers
    with text does.
    """
    return {"type": "tool_result", "tool_use_id": call_id,
            "content": content, "is_error": bool(is_error)}


def opaque_block(data: Any) -> Dict[str, Any]:
    """A provider block a pass carries but never reads.

    Thinking above all: it must go back to the model that produced it
    exactly as it arrived.
    """
    return {"type": "opaque", "data": data}


def user(*content: Any) -> Dict[str, Any]:
    return {"role": "user", "content": list(content)}


def assistant(*content: Any) -> Dict[str, Any]:
    return {"role": "assistant", "content": list(content)}


class CallCeilingReached(RuntimeError):
    """A model call refused, before anything was sent, because the run's
    ceiling of model calls was reached (``subagent.CallCeiling``).

    Here rather than beside the ceiling so that the graph, which only sees
    engines, can tell a step the ceiling cut short from one that failed: a
    step cut short is not saved as finished, and asking again reads it again.
    """


# -- what a call returns ----------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """One tool the model wants run."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Tokens one call spent."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: The output tokens the model spent reasoning before it answered, as the
    #: provider reports them: a PART of ``output_tokens``, not in addition,
    #: and priced with them. None where the provider does not say (the Claude
    #: API counts thinking in the output and reports no split), which is not
    #: the same as a reported 0.
    reasoning_tokens: Optional[int] = None

    def dollars(self, model: str) -> float:
        """What this call cost at list price, or 0.0 for an unpriced model.

        ``model`` is a model, not a tier: a Funhouse tier is an alias with no
        price of its own and gets 0.0, while the deployment behind it
        (``served_by``) is priced in :data:`PROMPTER_PRICES`.
        """
        price = price_for(model)
        if price is None:
            return 0.0
        din, dout = price[0] / 1e6, price[1] / 1e6
        return (self.input_tokens * din
                + self.cache_read_tokens * din * CACHE_READ_RATE
                + self.cache_write_tokens * din * CACHE_WRITE_RATE
                + self.output_tokens * dout)


@dataclass
class Reply:
    """One model turn: what it said, what it wants run, what it parsed."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    parsed: Any = None
    stop_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    seconds: float = 0.0
    #: The assistant turn in neutral blocks, to append straight back into a
    #: message list. Carries thinking blocks as ``opaque``.
    content: List[Dict[str, Any]] = field(default_factory=list)
    #: What the engine did with a structured answer that a reviewer of the
    #: run may want to know: unknown keys it dropped, an answer it did not
    #: take and why. Empty on the ordinary call.
    notes: List[str] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def cut_off(self) -> bool:
        """Whether the reply stopped at its output cap: its end is missing,
        and a structured answer cut there does not parse (BACKLOG RI-19)."""
        return _reason(self.stop_reason) in CUT_REASONS


#: How a request that came back with an error is filed (BACKLOG RI-21):
#: the provider's "too many requests"; a parameter the deployment refused;
#: a request too large for the gateway or the model's context; no answer
#: in time;
#: nothing back at all (Prompter's ``chat()`` hands back None); anything
#: else.
FAILURE_KINDS: Tuple[str, ...] = ("rate_limit", "parameter", "too_large",
                                  "timeout", "no_reply", "other")


def _message_of(error: Any) -> str:
    """An error's text, or ``""`` when reading it fails: recording an error
    must never raise in its place."""
    try:
        return str(error) if error is not None else ""
    except Exception:                       # noqa: BLE001 - never raise here
        return ""


def _count(value: Any) -> Optional[int]:
    """A token count as a provider sent it, or None when it is not one (a
    missing value, ``"n/a"``, a negative, a bool)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value == value else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def error_kind(error: Any) -> str:
    """Which of :data:`FAILURE_KINDS` an error is, read off its class name
    and its text the way :meth:`PrompterEngine.is_rate_limit` reads a rate
    limit. A prompt past the model's context is ``too_large`` (Claude says
    "prompt is too long", OpenAI "maximum context length")."""
    name = type(error).__name__.lower() if isinstance(error, BaseException) \
        else ""
    text = _message_of(error).lower()
    if ("ratelimit" in name or "too many requests" in text
            or "rate limit" in text or "rate_limit" in text
            or "error code: 429" in text):
        return "rate_limit"
    if ("too large" in text or "error code: 413" in text
            or "toolarge" in name or "prompt is too long" in text
            or "maximum context length" in text
            or "context_length_exceeded" in text):
        return "too_large"
    if "timeout" in name or "timed out" in text or "timeout" in text:
        return "timeout"
    if "unsupported parameter" in text or "unsupported value" in text:
        return "parameter"
    if "returned nothing" in text or "no reply" in text:
        return "no_reply"
    return "other"


def _first_line(error: Any) -> str:
    """An error's first line, short: what the cost record keeps of it."""
    text = _message_of(error)
    if isinstance(error, BaseException):
        text = (f"{type(error).__name__}: "
                f"{text or '(its message could not be read)'}")
    lines = text.strip().splitlines()
    return (lines[0] if lines else "")[:160]


def _usage_in_error(error: Any) -> Optional[Usage]:
    """The tokens an error response says it spent, when its body carries a
    ``usage`` (an OpenAI-shaped error body may); None when it does not,
    which is nearly always, or when what it carries is not a count."""
    try:
        body = getattr(error, "body", None)
        if not isinstance(body, dict):
            return None
        raw = body.get("usage")
        if raw is None and isinstance(body.get("error"), dict):
            raw = body["error"].get("usage")
        if not isinstance(raw, dict):
            return None
        spent_in = _count(raw.get("prompt_tokens"))
        if spent_in is None:
            spent_in = _count(raw.get("input_tokens"))
        spent_out = _count(raw.get("completion_tokens"))
        if spent_out is None:
            spent_out = _count(raw.get("output_tokens"))
    except Exception:                       # noqa: BLE001 - never raise here
        return None
    if not (spent_in or spent_out):
        return None
    return Usage(input_tokens=spent_in or 0, output_tokens=spent_out or 0)


class CostMeter:
    """What a run has spent so far, per model and in total.

    A call is priced by the DEPLOYMENT that served it when that deployment
    is priced (:data:`PROMPTER_PRICES`), and otherwise by the model that was
    asked for (:data:`MODEL_PRICES`). A Funhouse tier is in neither table, so
    a run on a tier whose deployment is also unpriced reports tokens and a
    flat 0.0 -- tokens, not dollars, exactly as before. :attr:`unpriced_calls`
    counts those, so a report can say how much of a total is really priced
    rather than printing a dollar figure that quietly covers half the run.

    EVERY REQUEST IS RECORDED, not only those that came back with a reply
    (BACKLOG RI-21). :attr:`calls` counts replies, as it always has;
    :attr:`failed` counts requests that came back with an error instead --
    a rate limit, a refused parameter, a body too large, a timeout, nothing
    at all (:data:`FAILURE_KINDS`) -- and :attr:`retried` how many of those
    the engine itself sent again. Requests sent = ``calls + failed``.
    :attr:`reasoning_tokens` adds up what the replies that SAID reported;
    :attr:`reasoning_reported` counts those replies, so a total over calls
    that never said is not read as "no reasoning". :attr:`per_call` holds
    one row per request in the order they went out: the call it belongs to
    and its attempt number, what came back (a reply and its stop reason, or
    the error's kind and first line), its tokens and the reasoning tokens
    among them, the wait before a retry, and what the engine changed from
    then on (``adapted``; all of them, once each, in :attr:`adaptations`).
    ``seconds`` is wall clock, waits included: a reply carries its whole
    call's, and so does an error that ended a call; an error that was sent
    again carries its wait instead.

    Recording never stands in the way of the call it records: an error
    whose text or body will not read is recorded with what can be read, and
    anything else that goes wrong while recording is logged and dropped, so
    the engine raises the provider's error, or returns the reply, as it
    would with no meter at all.

    What this meter cannot see is a retry INSIDE a provider's own client --
    the Claude SDK's, or a LangChain chat model's -- or in a host's adapter
    in front of it: only what reaches the engine is recorded.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.reasoning_tokens = 0
        self.reasoning_reported = 0
        self.seconds = 0.0
        self.dollars = 0.0
        self.unpriced_calls = 0
        self.by_model: Dict[str, Dict[str, Any]] = {}
        self.failed = 0
        self.retried = 0
        self.failed_by_kind: Dict[str, int] = {}
        self.waited_seconds = 0.0
        self.adaptations: List[str] = []
        self.per_call: List[Dict[str, Any]] = []
        self._call_no = 0

    def _model_row(self, model: str) -> Dict[str, Any]:
        return self.by_model.setdefault(
            model, {"calls": 0, "failed": 0, "input_tokens": 0,
                    "output_tokens": 0, "cache_read_tokens": 0,
                    "cache_write_tokens": 0, "reasoning_tokens": 0,
                    "seconds": 0.0, "dollars": 0.0, "priced_as": []})

    def _tokens(self, row: Dict[str, Any], usage: Usage) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_tokens",
                    "cache_write_tokens", "reasoning_tokens"):
            value = int(getattr(usage, key, 0) or 0)
            setattr(self, key, getattr(self, key) + value)
            row[key] = row.get(key, 0) + value

    def _entry(self, model: str, served_by: Optional[str], attempt: int,
               usage: Optional[Usage], seconds: float,
               adapted: Sequence[str]) -> Dict[str, Any]:
        """One :attr:`per_call` row; attempt 1 starts a new call."""
        if int(attempt) <= 1 or not self._call_no:
            self._call_no += 1
        entry: Dict[str, Any] = {"call": self._call_no,
                                 "attempt": max(1, int(attempt)),
                                 "model": model}
        if served_by:
            entry["served_by"] = served_by
        for key in ("input_tokens", "output_tokens", "cache_read_tokens",
                    "cache_write_tokens"):
            value = int(getattr(usage, key, 0) or 0) if usage else 0
            if value:
                entry[key] = value
        thought = getattr(usage, "reasoning_tokens", None) if usage else None
        if thought is not None:
            entry["reasoning_tokens"] = int(thought)
        if seconds:
            entry["seconds"] = round(float(seconds), 2)
        lessons = [str(a) for a in adapted or () if a]
        if lessons:
            entry["adapted"] = lessons
            for lesson in lessons:
                if lesson not in self.adaptations:
                    self.adaptations.append(lesson)
        self.per_call.append(entry)
        return entry

    def add(self, model: str, usage: Usage, seconds: float,
            served_by: Optional[str] = None, *, attempt: int = 1,
            stop: str = "", adapted: Sequence[str] = ()) -> Dict[str, Any]:
        """A reply. Returns its :attr:`per_call` row, for a note the engine
        has only after reading the reply."""
        self.calls += 1
        self.seconds += float(seconds)
        priced_as = served_by if price_for(served_by or "") else model
        cost = usage.dollars(priced_as)
        self.dollars += cost
        if price_for(priced_as or "") is None:
            self.unpriced_calls += 1
        row = self._model_row(model)
        row["calls"] += 1
        self._tokens(row, usage)
        if getattr(usage, "reasoning_tokens", None) is not None:
            self.reasoning_reported += 1
        row["seconds"] += float(seconds)
        row["dollars"] += cost
        if priced_as and priced_as not in row["priced_as"]:
            row["priced_as"].append(priced_as)
        try:
            entry = self._entry(model, served_by or model, attempt, usage,
                                seconds, adapted)
            entry["outcome"] = "reply"
            if stop:
                entry["stop"] = _reason(stop)
            return entry
        except Exception:                   # noqa: BLE001 - see the class
            logging.getLogger(__name__).warning(
                "the cost record could not keep a row for a reply from %s",
                model, exc_info=True)
            return {}

    def fail(self, model: str, error: Any, *, kind: Optional[str] = None,
             seconds: float = 0.0, attempt: int = 1, retried: bool = False,
             waited: float = 0.0, adapted: Sequence[str] = (),
             served_by: Optional[str] = None,
             usage: Optional[Usage] = None) -> Dict[str, Any]:
        """A request that came back with an error instead of a reply.

        ``retried``: the engine sent it again (``waited`` seconds first);
        ``seconds``: the wall clock of a call this error ENDED, waits
        included -- left 0 on one that was sent again, whose time the call's
        last row carries. ``usage``: what the error response said it spent,
        when it said (:func:`_usage_in_error` looks). Never raises: the
        caller is about to raise the provider's error, and a failure here
        must not take its place.
        """
        try:
            kind = kind if kind in FAILURE_KINDS else error_kind(error)
            usage = usage or _usage_in_error(error)
            said = _first_line(error)
            waited = float(waited or 0.0)
            seconds = float(seconds or 0.0)
            self.failed += 1
            self.retried += 1 if retried else 0
            self.failed_by_kind[kind] = self.failed_by_kind.get(kind, 0) + 1
            self.waited_seconds += waited
            self.seconds += seconds
            row = self._model_row(model)
            row["failed"] += 1
            row["seconds"] += seconds
            if usage is not None:
                self._tokens(row, usage)
            entry = self._entry(model, served_by, attempt, usage, seconds,
                                adapted)
            entry.update({"outcome": "error", "error_kind": kind,
                          "error": said})
            if retried:
                entry["retried"] = True
            if waited:
                entry["waited"] = round(waited, 1)
            return entry
        except Exception:                   # noqa: BLE001 - see the docstring
            logging.getLogger(__name__).warning(
                "the cost record could not record a failed request to %s",
                model, exc_info=True)
            return {}

    @property
    def priced(self) -> bool:
        """Whether every call this meter saw had a price."""
        return self.calls > 0 and self.unpriced_calls == 0

    def _rows(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """:attr:`per_call` as a run file keeps it: the model and the
        deployment that answered most of the replies once, in
        ``per_call_defaults``; a row names its own only where they differ,
        and its attempt only past the first. A reply row with no
        ``served_by`` was served by the default; an error row never names
        one (nothing served it)."""
        pairs = Counter((e.get("model"), e.get("served_by"))
                        for e in self.per_call if e.get("served_by"))
        if not pairs:
            pairs = Counter((e.get("model"), None) for e in self.per_call)
        model, served = pairs.most_common(1)[0][0] if pairs else (None, None)
        defaults = {key: value for key, value in (("model", model),
                                                  ("served_by", served))
                    if value}
        rows = []
        for entry in self.per_call:
            row = {key: value for key, value in entry.items()
                   if not (key in defaults and value == defaults[key])}
            if row.get("attempt") == 1:
                del row["attempt"]
            rows.append(row)
        return defaults, rows

    def to_dict(self) -> Dict[str, Any]:
        defaults, rows = self._rows()
        return {
            "calls": self.calls,
            "failed": self.failed,
            "retried": self.retried,
            "failed_by_kind": dict(self.failed_by_kind),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_reported": self.reasoning_reported,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "seconds": round(self.seconds, 1),
            "waited_seconds": round(self.waited_seconds, 1),
            "dollars": round(self.dollars, 4),
            "unpriced_calls": self.unpriced_calls,
            "adaptations": list(self.adaptations),
            "by_model": {m: {k: (round(v, 4) if isinstance(v, float)
                                 else list(v) if isinstance(v, list) else v)
                             for k, v in row.items()}
                         for m, row in self.by_model.items()},
            "per_call_defaults": defaults,
            "per_call": rows,
        }

    def summary(self) -> str:
        money = (f", ${self.dollars:.3f}" if self.dollars
                 else " (no priced deployment: tokens, not dollars)")
        failed = (f"; {self.failed} request(s) failed, {self.retried} sent "
                  f"again" if self.failed else "")
        return (f"{self.calls} calls, {self.input_tokens:,} in "
                f"(+{self.cache_read_tokens:,} cached), "
                f"{self.output_tokens:,} out, {self.seconds:.0f} s{money}"
                f"{failed}")


class Engine(Protocol):
    """What the two passes need from a model: one method, one reply.

    An implementation MUST accept neutral blocks and return neutral blocks,
    so a pass never sees a provider's shape.
    """

    #: What a scorecard records as the model that produced a run.
    name: str

    def complete(self, messages: Sequence[Dict[str, Any]], *,
                 system: Optional[str] = None,
                 tools: Optional[Sequence[Dict[str, Any]]] = None,
                 images: Optional[Sequence[bytes]] = None,
                 output_format: Any = None,
                 max_tokens: Optional[int] = None) -> Reply:
        ...


# -- the development engine -------------------------------------------------

class ClaudeEngine:
    """:class:`Engine` over the Claude API, for development and scoring.

    The credential comes from the environment (``ANTHROPIC_API_KEY``, or an
    ``ant auth login`` profile); it is never an argument, never logged and
    never printed -- only its presence is checked. ``model`` is recorded on
    every reply so a scorecard can say which model produced a number.

    Thinking is left at the model's own default (adaptive on Opus 5 and
    Sonnet 5, where it is the only on-mode), and thinking blocks come back as
    opaque blocks that a pass carries into the next turn untouched, as
    continuing with the same model requires.
    """

    def __init__(self, model: str = "claude-opus-5", *,
                 meter: Optional[CostMeter] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_retries: int = 4,
                 timeout: float = 900.0,
                 auto_cache: bool = True,
                 client: Any = None) -> None:
        self.model = str(model)
        self.name = self.model
        self.meter = meter if meter is not None else CostMeter()
        self.max_tokens = int(max_tokens)
        #: Cache the request's prefix. The label review resends its whole
        #: conversation every turn, and the brief alone is the document's
        #: entire ledger -- tens of thousands of tokens on a long report,
        #: unchanged from the first turn to the last. Caching it turns those
        #: resends into cache reads at a tenth the price. The meter counts
        #: them separately, so a run can show how much it saved.
        self.auto_cache = bool(auto_cache)
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:                         # pragma: no cover
            raise ImportError(
                "report_ingest.engine.ClaudeEngine needs the 'anthropic' "
                "package, which is optional: nothing in the app imports it"
            ) from exc
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "no Claude credential in the environment; set "
                "ANTHROPIC_API_KEY or run 'ant auth login'")
        self._client = anthropic.Anthropic(max_retries=int(max_retries),
                                           timeout=float(timeout))

    # -- translation -------------------------------------------------------
    @staticmethod
    def _to_provider_block(block: Any) -> Any:
        if isinstance(block, str):
            return {"type": "text", "text": block}
        kind = block.get("type")
        if kind == "text":
            return {"type": "text", "text": block["text"]}
        if kind == "image":
            import base64
            # The media type is read off the bytes: the ``png`` key holds a
            # PNG or a JPEG, and a JPEG announced as a PNG is refused.
            return {"type": "image",
                    "source": {"type": "base64",
                               "media_type": image_media_type(block["png"]),
                               "data": base64.standard_b64encode(
                                   block["png"]).decode("ascii")}}
        if kind == "tool_use":
            return {"type": "tool_use", "id": block["id"],
                    "name": block["name"], "input": block.get("input") or {}}
        if kind == "tool_result":
            content = block["content"]
            if not isinstance(content, str):
                content = [ClaudeEngine._to_provider_block(b) for b in content]
            out = {"type": "tool_result", "tool_use_id": block["tool_use_id"],
                   "content": content}
            if block.get("is_error"):
                out["is_error"] = True
            return out
        if kind == "opaque":
            return block["data"]
        raise ValueError(f"unknown block type {kind!r}")

    @classmethod
    def _to_provider(cls, messages: Sequence[Dict[str, Any]]
                     ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
            else:
                out.append({"role": m["role"],
                            "content": [cls._to_provider_block(b)
                                        for b in content]})
        return out

    @staticmethod
    def _from_provider(content: Sequence[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for block in content:
            kind = getattr(block, "type", None)
            if kind == "text":
                out.append(text_block(block.text))
            elif kind == "tool_use":
                out.append(tool_use_block(block.id, block.name,
                                          dict(block.input or {})))
            else:
                # Thinking, and anything else the provider adds: carried back
                # exactly as it arrived, never read.
                raw = (block.model_dump(exclude_none=True)
                       if hasattr(block, "model_dump") else block)
                out.append(opaque_block(raw))
        return out

    # -- the protocol ------------------------------------------------------
    def complete(self, messages: Sequence[Dict[str, Any]], *,
                 system: Optional[str] = None,
                 tools: Optional[Sequence[Dict[str, Any]]] = None,
                 images: Optional[Sequence[bytes]] = None,
                 output_format: Any = None,
                 max_tokens: Optional[int] = None) -> Reply:
        msgs = [dict(m) for m in messages]
        if images:
            if not msgs or msgs[-1]["role"] != "user":
                msgs.append(user())
            content = msgs[-1]["content"]
            if isinstance(content, str):
                content = [text_block(content)]
            msgs[-1] = {"role": "user",
                        "content": list(content)
                        + [image_block(p) for p in images]}
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_tokens or self.max_tokens),
            "messages": self._to_provider(msgs),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = list(tools)
        started = time.time()
        try:
            if output_format is not None:
                # ``parse`` builds the output format itself and takes no
                # top-level ``cache_control``; the same instruction reaches
                # the API through the request body.
                if self.auto_cache:
                    kwargs["extra_body"] = {
                        "cache_control": {"type": "ephemeral"}}
                response = self._client.messages.parse(
                    output_format=output_format, **kwargs)
            else:
                if self.auto_cache:
                    kwargs["cache_control"] = {"type": "ephemeral"}
                response = self._client.messages.create(**kwargs)
        except Exception as exc:
            # Recorded, then raised as before: a failed call is a call
            # (BACKLOG RI-21). The SDK's own retries happen inside it.
            self.meter.fail(self.model, exc, seconds=time.time() - started)
            raise
        seconds = time.time() - started

        raw = response.usage
        usage = Usage(
            input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
            cache_read_tokens=int(
                getattr(raw, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(
                getattr(raw, "cache_creation_input_tokens", 0) or 0),
        )
        self.meter.add(self.model, usage, seconds,
                       stop=str(getattr(response, "stop_reason", "") or ""))
        blocks = self._from_provider(response.content)
        return Reply(
            text="".join(b["text"] for b in blocks if b["type"] == "text"),
            tool_calls=[ToolCall(b["id"], b["name"], b["input"])
                        for b in blocks if b["type"] == "tool_use"],
            parsed=getattr(response, "parsed_output", None),
            stop_reason=str(getattr(response, "stop_reason", "") or ""),
            usage=usage,
            model=self.model,
            seconds=seconds,
            content=blocks,
        )


# -- the production engine --------------------------------------------------

#: JSON-schema keywords OpenAI's strict mode refuses. The schema is checked
#: before the call, and a refused schema is a Bad Request with no token
#: spent -- which is how every reader call of the first cluster run
#: (2026-09-18) died with "0 model calls": pydantic had emitted ``default``
#: for every optional field and ``minItems``/``maxItems`` for every bounded
#: list. Each is dropped, and where it carried a limit the model should
#: still know about, the limit is folded into the field's description. The
#: Python gates in every reader enforce the real limits either way.
STRICT_UNSUPPORTED: Tuple[str, ...] = (
    "default", "minItems", "maxItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
    "maxLength", "pattern", "format", "uniqueItems", "patternProperties",
    "minProperties", "maxProperties", "examples", "unevaluatedItems",
    "unevaluatedProperties", "contains", "minContains", "maxContains",
    "propertyNames", "prefixItems", "additionalItems",
)


def _constraint_hint(node: Dict[str, Any]) -> str:
    """The limits a node carries, as words for its description."""
    bits: List[str] = []
    lo, hi = node.get("minimum"), node.get("maximum")
    if lo is not None and hi is not None:
        bits.append(f"between {lo} and {hi}")
    elif lo is not None:
        bits.append(f"at least {lo}")
    elif hi is not None:
        bits.append(f"at most {hi}")
    if node.get("exclusiveMinimum") is not None:
        bits.append(f"greater than {node['exclusiveMinimum']}")
    if node.get("exclusiveMaximum") is not None:
        bits.append(f"less than {node['exclusiveMaximum']}")
    for lo_key, hi_key, unit in (("minItems", "maxItems", "items"),
                                 ("minLength", "maxLength", "characters")):
        a, b = node.get(lo_key), node.get(hi_key)
        if a is not None and b is not None:
            bits.append(f"{a} to {b} {unit}")
        elif a is not None:
            bits.append(f"at least {a} {unit}")
        elif b is not None:
            bits.append(f"at most {b} {unit}")
    if node.get("pattern"):
        bits.append(f"matching {node['pattern']}")
    if node.get("format"):
        bits.append(str(node["format"]))
    return "; ".join(bits)


def strict_schema(model: Any) -> Dict[str, Any]:
    """A pydantic model's JSON schema, in the shape strict mode demands.

    OpenAI's ``json_schema`` response format with ``strict: true`` requires
    every object to set ``additionalProperties: false`` and to list EVERY
    property in ``required``. Pydantic emits neither for a field with a
    default, so the schema is walked and both are imposed. ``$defs`` and
    ``$ref`` are left alone -- strict mode understands them -- and so is
    ``anyOf``, which is how an optional field's null arm arrives.

    Strict mode also REFUSES a schema that carries any keyword it does not
    implement (:data:`STRICT_UNSUPPORTED`): pydantic's ``default`` on every
    optional field, ``minItems``/``maxItems`` on a bounded list, numeric
    bounds, string patterns. Those are removed here; a limit they expressed
    is appended to the field's description so the model still reads it,
    and a single-value ``const`` becomes a one-entry ``enum``.
    """
    import copy

    schema = copy.deepcopy(model.model_json_schema())

    #: Keys whose value is a MAP of names to schemas; each value is walked.
    maps = ("properties", "$defs", "definitions")
    #: Keys whose value is itself a schema, or a list of them.
    schemas = ("items", "anyOf", "allOf", "oneOf", "prefixItems", "not",
               "additionalItems")

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        tuple_hint = ""
        if isinstance(node.get("prefixItems"), list):
            # A fixed-length tuple (a four-number bbox) is tuple validation,
            # which strict mode does not implement. It becomes a plain array
            # whose items may be any of the tuple's arm types, and the
            # length rides in the description; pydantic still checks the
            # tuple when the answer is parsed.
            arms = node.pop("prefixItems")
            distinct = [a for i, a in enumerate(arms) if a not in arms[:i]]
            node["items"] = distinct[0] if len(distinct) == 1 else {
                "anyOf": distinct}
            node.pop("minItems", None)
            node.pop("maxItems", None)
            kinds = ", ".join(str(a.get("type", "value")) for a in arms)
            tuple_hint = f"exactly {len(arms)} items: {kinds}"
        hint = _constraint_hint(node)
        if tuple_hint:
            hint = tuple_hint + (f"; {hint}" if hint else "")
        for key in STRICT_UNSUPPORTED:
            node.pop(key, None)
        if "const" in node:
            node["enum"] = [node.pop("const")]
        if hint:
            description = str(node.get("description") or "").rstrip()
            node["description"] = (f"{description} ({hint})" if description
                                   else f"({hint})")
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties") or {})
        for key, value in node.items():
            if key in maps and isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif key in schemas:
                walk(value)

    walk(schema)
    return schema


class PrompterEngine:
    """:class:`Engine` over Funhouse's Prompter, for the cluster.

    This is the engine the numbers come from. The app runs in Funhouse
    against OpenAI models through Prompter, so a score measured on Claude
    measures a model that will never do the work.

    It takes the live ``fh_prompter`` object the owner already has in the
    notebook. No credential is read, stored or passed: authentication is
    whatever that object was built with.

    HOW IT DIFFERS FROM THE CLAUDE ENGINE, because the wire format differs
    and the difference is not cosmetic:

    * A tool result is its own ``role="tool"`` message rather than a block
      inside the next user turn, and each one must follow its assistant turn
      immediately.
    * **A tool message's content must be a string.** The label review's
      ``render_page`` and ``contact_sheet`` answer with a PICTURE, and there
      is nowhere in a tool message to put one. So an image-bearing result is
      sent as a tool message saying the picture follows, and the picture
      itself rides in a user message straight after the tool messages. The
      model sees both; the ordering rules are respected.
    * Structured output is ``response_format`` with a strict JSON schema
      (:func:`strict_schema`), not a parsed pydantic object, so the reply's
      ``parsed`` is built here by validating the JSON that comes back.
    * There is no explicit prompt cache to ask for. Azure reports what it
      cached of its own accord in ``prompt_tokens_details.cached_tokens``,
      and that is recorded, but nothing here can widen it.

    THE RAW CLIENT, AND WHAT IT COSTS. A multi-turn tool loop needs
    assistant-with-tool-calls and ``role="tool"`` messages, and Prompter's
    own ``chat()`` only ever sends one system and one user message. So the
    loop drives ``prompter.client`` directly, exactly as the app's
    ``NativeToolEngine`` does. Two consequences, both deliberate and both in
    the README: the SDK's ``@check_budget`` guard does not run on those
    calls, and on a backend where ``prompter.client`` is ``None`` (Grok)
    there is no loop at all. Single-shot calls with no tools go through
    ``prompter.chat()`` instead, which keeps the budget guard and the SDK's
    logging -- so triage is always budgeted, and only the review is not.

    PARAMETERS A MODEL WILL NOT TAKE. The tiers are aliases and the model
    behind one changes without notice. A reasoning-class deployment refuses
    ``max_tokens`` (it wants ``max_completion_tokens``) and refuses any
    ``temperature`` but its default, which is what stopped the first cluster
    run on 2026-09-18. The first call that meets such a refusal is retried
    ONCE with the parameter renamed or dropped, exactly as the app's
    Databricks bridge does, and the lesson is kept on the engine
    (:attr:`token_key`, :attr:`send_temperature`) so every later call sends
    the accepted form first. ``prompter.chat()`` swallows the provider's
    error and hands back ``None``; that case is retried on the raw client
    with the same adaptation, and because ``chat()`` always sends a
    temperature of its own, the engine stops offering it for the rest of the
    run once it has failed. :attr:`adaptations` lists what was learned.

    EVERY ATTEMPT IS IN THE COST RECORD (BACKLOG RI-21): each refusal, each
    rate-limited request and each empty ``chat()`` is a failed request on
    :attr:`meter`, with its kind, its attempt number within the call, the
    wait before it was sent again and the lesson it taught; the reply that
    follows carries the attempt it came on and its reasoning tokens.
    """

    #: What a scorecard records when the tier's deployment is unknown.
    name: str

    def __init__(self, prompter: Any, model: str = "funhouse-gpt-medium", *,
                 meter: Optional[CostMeter] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 prefer_chat_for_single_calls: bool = True) -> None:
        self._prompter = prompter
        self.model = str(model)
        self.name = self.model
        self.meter = meter if meter is not None else CostMeter()
        self.max_tokens = int(max_tokens)
        self.prefer_chat = bool(prefer_chat_for_single_calls)
        #: The deployment that actually served the last call. Funhouse tiers
        #: are aliases and the model behind one changes without notice, so a
        #: run records what answered as well as what it asked for.
        self.served_by: Optional[str] = None
        #: Which token-cap key the deployment accepts, and whether it takes a
        #: temperature at all. Both start at the classic OpenAI form and are
        #: corrected by the first refusal (see the class docstring).
        self.token_key: str = "max_tokens"
        self.send_temperature: bool = True
        #: One line per lesson learned from a refusal, oldest first.
        self.adaptations: List[str] = []
        #: How a rate-limited call waits; a test replaces it.
        self._sleep = time.sleep
        #: The attempt the call under way is on, and whether any request of
        #: it has gone out yet: the cost record's (BACKLOG RI-21).
        self._attempt = 1
        self._sent = False

    def _sent_again(self, error: Any, kind: str, *, waited: float = 0.0,
                    lesson: str = "") -> None:
        """Record a request that came back with an error and is about to be
        sent again; the next request is the call's next attempt."""
        self.meter.fail(self.model, error, kind=kind, attempt=self._attempt,
                        retried=True, waited=waited,
                        adapted=[lesson] if lesson else ())
        self._attempt += 1

    @staticmethod
    def adjust_for_param_error(request: Dict[str, Any],
                               message: str) -> Optional[Dict[str, Any]]:
        """The retry request with the parameter the model refused renamed or
        dropped, or ``None`` when the error is not about a parameter."""
        low = (message or "").lower()
        new = dict(request)
        changed = False
        if "temperature" in low and "temperature" in new:
            new.pop("temperature", None)
            changed = True
        if (("max_completion_tokens" in low or "max_tokens" in low)
                and "max_tokens" in new):
            new["max_completion_tokens"] = new.pop("max_tokens")
            changed = True
        return new if changed else None

    #: Seconds to wait before each retry of a rate-limited call, in order.
    #: The provider's per-minute limit on a tier is shared with everything
    #: else the office runs on it, and the SDK's client retries only once
    #: with a sub-second pause, so the first full cluster run (2026-09-18)
    #: lost 10 of 38 label reviews and 5 of 15 logs to 429s. A retry-after
    #: header, when the provider sends one, is honoured instead.
    RATE_LIMIT_WAITS: Tuple[float, ...] = (15.0, 30.0, 60.0, 120.0, 120.0,
                                           120.0)

    @staticmethod
    def is_rate_limit(exc: BaseException) -> bool:
        """Whether an exception is the provider saying 'too many requests'."""
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        return ("ratelimit" in name or "too many requests" in text
                or "rate limit" in text or "rate_limit" in text
                or "error code: 429" in text)

    @staticmethod
    def _retry_after(exc: BaseException) -> Optional[float]:
        """The provider's own wait, when its response carried one."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
            return float(value) if value is not None else None
        except (TypeError, ValueError, AttributeError):
            return None

    def _create_adaptive(self, client: Any, kwargs: Dict[str, Any]) -> Any:
        """One call on the raw client, retried with each refused parameter
        fixed in turn -- the backend reports one refusal at a time -- and
        every fix remembered for every later call. A rate-limited call
        waits (:attr:`RATE_LIMIT_WAITS`, or the provider's retry-after)
        and tries again; the waits are metered as wall clock like any
        other second the call took."""
        create = client.chat.completions.create
        request = dict(kwargs)
        waits = list(self.RATE_LIMIT_WAITS)
        for _ in range(3 + len(waits)):
            try:
                self._sent = True
                return create(**request)
            except Exception as exc:                # noqa: BLE001 - inspected
                if self.is_rate_limit(exc) and waits:
                    wait = self._retry_after(exc) or waits.pop(0)
                    self.adaptations.append(
                        f"{self.model}: rate limited; waited {wait:.0f} s")
                    logging.getLogger(__name__).info(
                        "prompter engine rate limited on %s; waiting %.0f s",
                        self.model, wait)
                    self._sent_again(exc, "rate_limit", waited=wait)
                    self._sleep(wait)
                    continue
                adjusted = self.adjust_for_param_error(request, str(exc))
                if adjusted is None:
                    raise                   # complete() records it
                if "temperature" in request and "temperature" not in adjusted:
                    self.send_temperature = False
                if "max_completion_tokens" in adjusted:
                    self.token_key = "max_completion_tokens"
                first_line = str(exc).strip().splitlines()[0][:160]
                lesson = (f"{self.model}: refused "
                          f"{sorted(set(request) - set(adjusted))} -- "
                          f"{first_line}")
                self.adaptations.append(lesson)
                logging.getLogger(__name__).info(
                    "prompter engine adapted a request: %s", lesson)
                self._sent_again(exc, "parameter", lesson=lesson)
                request = adjusted
        self._sent = True
        return create(**request)

    @property
    def client(self) -> Any:
        """Prompter's own OpenAI client, or None on a backend without one."""
        get = getattr(self._prompter, "get_openai_instance", None)
        if callable(get):
            return get()
        return getattr(self._prompter, "client", None)

    # -- translation -------------------------------------------------------
    @staticmethod
    def _content(blocks: Sequence[Any]) -> Any:
        """Neutral blocks as OpenAI user content."""
        out: List[Dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, str):
                out.append({"type": "text", "text": block})
                continue
            kind = block.get("type")
            if kind == "text":
                out.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                import base64
                # The data-URI prefix is read off the bytes, for the same
                # reason the Claude engine reads the media type off them.
                media = image_media_type(block["png"])
                b64 = base64.b64encode(block["png"]).decode("ascii")
                url: Dict[str, Any] = {
                    "url": f"data:{media};base64,{b64}"}
                if block.get("detail"):
                    # Only when it was asked for: a request that never sets
                    # it gets the provider's own default, as before.
                    url["detail"] = str(block["detail"])
                out.append({"type": "image_url", "image_url": url})
            else:
                raise ValueError(
                    f"{kind!r} cannot go in a user message; tool calls and "
                    f"tool results are separate messages here")
        if len(out) == 1 and out[0]["type"] == "text":
            return out[0]["text"]
        return out

    @classmethod
    def _to_provider(cls, messages: Sequence[Dict[str, Any]],
                     system: Optional[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            blocks = list(content)
            if role == "assistant":
                text = "".join(b["text"] for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text")
                calls = [b for b in blocks
                         if isinstance(b, dict) and b.get("type") == "tool_use"]
                turn: Dict[str, Any] = {"role": "assistant",
                                        "content": text or None}
                if calls:
                    turn["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": json.dumps(
                                          c.get("input") or {})}}
                        for c in calls]
                out.append(turn)
                continue
            # A user turn: tool results become their own messages, and any
            # picture among them follows in a user message of its own.
            results = [b for b in blocks
                       if isinstance(b, dict) and b.get("type") == "tool_result"]
            plain = [b for b in blocks
                     if not (isinstance(b, dict)
                             and b.get("type") == "tool_result")]
            pictures: List[Dict[str, Any]] = []
            for result in results:
                body = result["content"]
                if isinstance(body, str):
                    text = body
                else:
                    words = [b["text"] for b in body
                             if isinstance(b, dict) and b.get("type") == "text"]
                    images = [b for b in body
                              if isinstance(b, dict) and b.get("type") == "image"]
                    text = " ".join(words)
                    if images:
                        text = ((text + " ") if text else "") + (
                            "[the picture follows in the next message]")
                        pictures.append(text_block(
                            f"Picture for tool call {result['tool_use_id']}"
                            + (f": {' '.join(words)}" if words else "")))
                        pictures.extend(images)
                out.append({"role": "tool",
                            "tool_call_id": result["tool_use_id"],
                            "content": text or "(no content)"})
            trailing = plain + pictures
            if trailing:
                out.append({"role": "user", "content": cls._content(trailing)})
        return out

    # -- the protocol ------------------------------------------------------
    def complete(self, messages: Sequence[Dict[str, Any]], *,
                 system: Optional[str] = None,
                 tools: Optional[Sequence[Dict[str, Any]]] = None,
                 images: Optional[Sequence[bytes]] = None,
                 output_format: Any = None,
                 max_tokens: Optional[int] = None) -> Reply:
        msgs = [dict(m) for m in messages]
        if images:
            if not msgs or msgs[-1]["role"] != "user":
                msgs.append(user())
            content = msgs[-1]["content"]
            if isinstance(content, str):
                content = [text_block(content)]
            msgs[-1] = {"role": "user",
                        "content": list(content)
                        + [image_block(p) for p in images]}

        response_format = None
        if output_format is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(output_format, "__name__", "answer"),
                    "schema": strict_schema(output_format),
                    "strict": True,
                },
            }

        openai_messages = self._to_provider(msgs, system)
        simple = (not tools
                  and len(openai_messages) <= (2 if system else 1)
                  and all(m["role"] in ("system", "user")
                          for m in openai_messages))
        started = time.time()
        cap = int(max_tokens or self.max_tokens)
        raw = None
        tried_chat = False
        self._attempt, self._sent = 1, False
        #: How the call ended, where this engine knows better than the
        #: error's text: nothing back is ``no_reply`` with or without a client.
        ended_kind: Optional[str] = None
        try:
            if self.prefer_chat and simple and hasattr(self._prompter, "chat"):
                # Keeps the SDK's budget guard and logging on every call
                # that does not need a tool loop -- triage, above all.
                last = openai_messages[-1]
                tried_chat = self._sent = True
                raw = self._prompter.chat(
                    user=last["content"], system=system or
                    "You are a helpful assistant.",
                    model=self.model, temperature=0, return_raw=True,
                    **{self.token_key: cap},
                    **({"response_format": response_format}
                       if response_format else {}))
                if raw is None:
                    # The SDK logs the provider's refusal and hands back
                    # None. A parameter the deployment does not take is the
                    # usual reason, and the raw client can adapt to that, so
                    # it is tried before giving up; the SDK's stashed reason
                    # travels either way.
                    reason = getattr(self._prompter, "_last_chat_error",
                                     None)
                    if self.client is None:
                        ended_kind = "no_reply"
                        raise RuntimeError(
                            "Prompter returned nothing; the call failed or "
                            f"was refused (the SDK's reason: {reason!r})")
                    lesson = (f"{self.model}: chat() returned nothing "
                              f"({_message_of(reason)[:160]!r}); retried on "
                              f"the raw client")
                    self.adaptations.append(lesson)
                    self._sent_again(reason if reason is not None
                                     else "chat() returned nothing",
                                     "no_reply", lesson=lesson)
            if raw is None:
                client = self.client
                if client is None:
                    raise RuntimeError(
                        "this Prompter backend exposes no OpenAI client, so "
                        "it cannot run a tool loop; the label review needs "
                        "one")
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": openai_messages,
                    self.token_key: cap,
                }
                if self.send_temperature:
                    kwargs["temperature"] = 0
                if tools:
                    kwargs["tools"] = [{"type": "function", "function": {
                        "name": t["name"], "description": t["description"],
                        "parameters": t["input_schema"]}} for t in tools]
                    kwargs["tool_choice"] = "auto"
                if response_format:
                    kwargs["response_format"] = response_format
                raw = self._create_adaptive(client, kwargs)
                if tried_chat:
                    # chat() failed where the raw client did not, and chat()
                    # always sends its own temperature, so it is not offered
                    # again this run.
                    self.prefer_chat = False
            if raw is None:
                ended_kind = "no_reply"
                raise RuntimeError(
                    "Prompter returned nothing; the call failed or was "
                    "refused (the SDK stashes the reason on the prompter "
                    "object)")
        except Exception as exc:
            # The request that ended the call, recorded before it is raised
            # as it always was. Nothing is recorded when nothing went out (a
            # backend with no client for a tool loop).
            if self._sent:
                self.meter.fail(self.model, exc, kind=ended_kind,
                                attempt=self._attempt,
                                seconds=time.time() - started)
            raise
        seconds = time.time() - started
        return self._reply(raw, output_format, seconds)

    def _reply(self, raw: Any, output_format: Any, seconds: float) -> Reply:
        if raw is None:
            raise RuntimeError(
                "Prompter returned nothing; the call failed or was refused "
                "(the SDK stashes the reason on the prompter object)")
        self.served_by = getattr(raw, "model", None) or self.model
        usage_raw = getattr(raw, "usage", None)
        details = getattr(usage_raw, "prompt_tokens_details", None)
        thought = getattr(usage_raw, "completion_tokens_details", None)
        usage = Usage(
            input_tokens=int(getattr(usage_raw, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage_raw, "completion_tokens", 0) or 0),
            cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            reasoning_tokens=_count(getattr(thought, "reasoning_tokens",
                                            None)),
        )
        choice = raw.choices[0] if getattr(raw, "choices", None) else None
        # Metered by what ANSWERED, not by the tier that was asked for: the
        # tier is an alias with no price and the deployment behind it has
        # one (:data:`PROMPTER_PRICES`).
        self.meter.add(self.model, usage, seconds, served_by=self.served_by,
                       attempt=self._attempt,
                       stop=str(getattr(choice, "finish_reason", "") or ""))

        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "") if message else ""
        blocks: List[Dict[str, Any]] = []
        if text:
            blocks.append(text_block(text))
        calls: List[ToolCall] = []
        for call in (getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            try:
                arguments = json.loads(getattr(function, "arguments", "") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(call.id, function.name, arguments))
            blocks.append(tool_use_block(call.id, function.name, arguments))

        parsed = None
        if output_format is not None and text:
            try:
                parsed = output_format.model_validate_json(text)
            except Exception:                       # noqa: BLE001 - reported
                try:
                    parsed = output_format.model_validate(json.loads(text))
                except Exception:                   # noqa: BLE001
                    parsed = None
        return Reply(
            text=text,
            tool_calls=calls,
            parsed=parsed,
            stop_reason=str(getattr(choice, "finish_reason", "") or ""),
            usage=usage,
            model=self.served_by or self.model,
            seconds=seconds,
            content=blocks,
        )


# -- the app's own chat model -----------------------------------------------

#: The tool a structured answer comes back through on
#: :class:`ChatModelEngine`. A pass's own tool may not take this name.
ANSWER_TOOL = "answer"

#: What the model is told when a structured answer is asked for.
_ANSWER_NOTE = (
    "Give your answer by calling the `answer` tool, with the whole answer "
    "as its arguments, not by writing it out as text.")
_ANSWER_NOTE_WITH_TOOLS = (
    "Use the other tools to look further when you need to. Give your answer "
    "by calling the `answer` tool, with the whole answer as its arguments, "
    "not by writing it out as text.")

#: The result sent back for a tool call nothing answered: the answer tool's
#: own call (the call WAS the answer), or a call the pass left out. Every
#: provider refuses a tool call with no result after it. The answer call's
#: result says what became of the answer: taken; not taken because the pass
#: ran the look asked for in the same turn; or not taken, and why.
_ANSWER_RECEIVED = "Answer received."
_ANSWER_AFTER_LOOK = (
    "Not taken: you asked to look first, and the look was run. Call "
    "`answer` again with the whole answer once you have what you looked for.")
_ANSWER_NOT_TAKEN = (
    "Not taken: {why}. Call `answer` again with the whole answer.")
_NOT_ANSWERED = "(this call was not run)"
#: Why an answer cut at the output cap is not taken.
_CUT_ANSWER = ("the reply stopped at its output limit, so the answer in it "
               "is incomplete; keep the answer shorter")

#: The fields a LangChain chat model keeps its per-reply output cap in.
_CAP_FIELDS = ("max_tokens", "max_output_tokens", "max_completion_tokens")


class ChatModelEngine:
    """:class:`Engine` over any LangChain chat model: the app's own (RI-03).

    The app builds its agent on a LangChain chat model -- on Foundry,
    ``webapp.palantir_sdk_engine``'s classes over ``palantir_models``, GPT
    or Claude; elsewhere ``ChatOpenAI``, ``ChatAnthropic`` or the Prompter
    bridge -- and this engine reads a report on that same model, so the
    ingest runs wherever the page does. Nothing here imports a provider:
    the model object does the talking, and its own retries and limits
    apply.

    HOW A CALL GOES OUT:

    * the pass's tools are bound with ``bind_tools``, as OpenAI function
      specs;
    * **a structured answer comes back as a call to the** ``answer`` **tool**,
      whose arguments are the answer: Foundry's model classes take no
      ``response_format``. They take no ``tool_choice`` either (the Claude
      class refuses a named one), so the call cannot be forced on the wire:
      the answer tool is offered beside the pass's own tools (alone when
      there are none), its schema is :func:`strict_schema`'s, and the system
      prompt says to call it. An answer written out as JSON text is parsed
      too. A reply with neither comes back with ``parsed`` None, as on
      every engine, and the pass's own re-ask or QA entry follows;
    * a picture goes in as an ``image_url`` block with a data URI, the way
      the app's vision tools send one (``detail`` only when the block asks);
      a tool's picture rides in a user message straight after the tool
      results, as on :class:`PrompterEngine`, because a tool message holds
      text on every route;
    * the assistant turn comes back with the provider's own message carried
      as an ``opaque`` block, so a Claude thinking block or a call id goes
      back exactly as it arrived; the answer call, which no pass answers,
      gets a result of its own on the next request saying what became of
      the answer, and a call whose arguments did not parse is dropped from
      the turn that goes back.

    WHAT IS TAKEN AS AN ANSWER. A reply cut at its output cap gives no
    answer, whatever its arguments say, and neither do empty arguments:
    Foundry's classes hand back ``{}`` for a call cut mid-arguments, and many
    reading schemas accept ``{}``, so it would pass as an empty reading. Both
    come back with ``parsed`` None and the pass's own re-ask or QA entry
    follows. Foundry sends tools without strict mode, so a model may add a
    key the schema forbids: unknown keys are dropped, named in
    :attr:`Reply.notes`, and the answer is validated again.

    ONE CALL, ONE MODEL REQUEST. No retry happens in here, so the ingest's
    :class:`~report_ingest.subagent.CallCeiling`, the pass's own count and
    :attr:`meter` all see the same number (the meter's replies and failed
    requests together: a request that raised is recorded as failed before
    it is raised). Usage is read off the reply's ``usage_metadata``, the
    cached input apart from the rest and the reasoning tokens counted.

    THE OUTPUT CAP (BACKLOG RI-19) is the model's own: no cap is sent unless
    a call names one, so every call kind gets what the host built the model
    with -- 32,000 tokens by default on the app
    (``webapp.engine_config.DEFAULT_MAX_TOKENS``, ``GEOTECH_WEBAPP_MAX_TOKENS``
    to change), within the model's own documented limit. A call that names a
    cap gets it on a copy of the model, never above the model's own. A reply
    that stops at the cap says so (:attr:`Reply.cut_off`) and is logged.
    """

    def __init__(self, chat_model: Any, *,
                 meter: Optional[CostMeter] = None,
                 max_tokens: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        self.chat_model = chat_model
        self.name = str(name or _chat_model_name(chat_model)
                        or type(chat_model).__name__)
        self.model = self.name
        self.meter = meter if meter is not None else CostMeter()
        #: The model's own per-reply cap, as the host built it; None when it
        #: carries none (the provider's default then applies).
        self.output_cap: Optional[int] = _own_cap(chat_model)
        #: A cap for every call that names none; None: the model's own.
        self.max_tokens: Optional[int] = int(max_tokens) if max_tokens else None
        #: The model that answered the last call, as the reply names it.
        self.served_by: Optional[str] = None
        self._capped: Dict[int, Any] = {}
        #: What this engine changed on the fly, oldest first: a cap it could
        #: not set on a copy of the model. Each lesson also rides on the
        #: cost record of the call it was learned on (BACKLOG RI-21).
        self.adaptations: List[str] = []

    # -- the cap -----------------------------------------------------------
    def _cap_for(self, asked: Optional[int]) -> Optional[int]:
        """The cap to send, or None for the model's own."""
        wanted = asked or self.max_tokens
        if not wanted:
            return None
        wanted = int(wanted)
        if self.output_cap and wanted >= self.output_cap:
            return None
        return wanted

    def _model_for(self, cap: Optional[int]) -> Any:
        """The model with its cap held to ``cap``: a copy made once per cap
        that shares the original's client; the model itself for None or for
        a model that keeps no cap a copy can change."""
        if cap is None:
            return self.chat_model
        if cap in self._capped:
            return self._capped[cap]
        made = self.chat_model
        fields = getattr(type(self.chat_model), "model_fields", None)
        copy = getattr(self.chat_model, "model_copy", None)
        if isinstance(fields, dict) and callable(copy):
            for key in _CAP_FIELDS:
                if key in fields:
                    try:
                        made = copy(update={key: cap})
                    except Exception as exc:        # noqa: BLE001
                        made = self.chat_model
                        self.adaptations.append(
                            f"{self.name}: a cap of {cap} could not be set "
                            f"on a copy of the model "
                            f"({_first_line(exc)}); the model's own cap "
                            f"applies")
                    break
        self._capped[cap] = made
        return made

    # -- translation -------------------------------------------------------
    @staticmethod
    def _user_content(blocks: Sequence[Any]) -> Any:
        """Neutral blocks as LangChain user content."""
        import base64

        out: List[Dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, str):
                out.append({"type": "text", "text": block})
                continue
            kind = block.get("type")
            if kind == "text":
                out.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                media = image_media_type(block["png"])
                b64 = base64.b64encode(block["png"]).decode("ascii")
                url: Dict[str, Any] = {"url": f"data:{media};base64,{b64}"}
                if block.get("detail"):
                    url["detail"] = str(block["detail"])
                out.append({"type": "image_url", "image_url": url})
            elif kind == "opaque":
                continue                # another engine's block: not ours
            else:
                raise ValueError(
                    f"{kind!r} cannot go in a user message here; tool calls "
                    f"and tool results are messages of their own")
        if len(out) == 1 and out[0]["type"] == "text":
            return out[0]["text"]
        return out

    @classmethod
    def _to_langchain(cls, messages: Sequence[Dict[str, Any]],
                      system: Optional[str]) -> List[Any]:
        from langchain_core.messages import (AIMessage, HumanMessage,
                                             SystemMessage, ToolMessage)

        out: List[Any] = []
        verdicts: Dict[str, str] = {}
        if system:
            out.append(SystemMessage(content=system))
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "system":
                out.append(SystemMessage(content=content if isinstance(
                    content, str) else cls._user_content(content)))
                continue
            if role == "assistant":
                if isinstance(content, str):
                    out.append(AIMessage(content=content))
                    continue
                held = next((b for b in content
                             if isinstance(b, dict)
                             and b.get("type") == "opaque"
                             and isinstance(b.get("data"), AIMessage)), None)
                if held is not None:
                    out.append(held["data"])
                    verdicts.update(held.get("answers") or {})
                    continue
                text = "".join(b["text"] for b in content
                               if isinstance(b, dict)
                               and b.get("type") == "text")
                calls = [{"name": b["name"], "args": dict(b.get("input") or {}),
                          "id": b["id"], "type": "tool_call"}
                         for b in content if isinstance(b, dict)
                         and b.get("type") == "tool_use"]
                out.append(AIMessage(content=text, tool_calls=calls))
                continue
            if isinstance(content, str):
                out.append(HumanMessage(content=content))
                continue
            blocks = list(content)
            results = [b for b in blocks if isinstance(b, dict)
                       and b.get("type") == "tool_result"]
            plain = [b for b in blocks if not (isinstance(b, dict)
                                               and b.get("type") == "tool_result")]
            pictures: List[Dict[str, Any]] = []
            for result in results:
                body = result["content"]
                if isinstance(body, str):
                    text = body
                else:
                    words = [b["text"] for b in body
                             if isinstance(b, dict) and b.get("type") == "text"]
                    images = [b for b in body
                              if isinstance(b, dict) and b.get("type") == "image"]
                    text = " ".join(words)
                    if images:
                        text = ((text + " ") if text else "") + (
                            "[the picture follows in the next message]")
                        pictures.append(text_block(
                            f"Picture for tool call {result['tool_use_id']}"
                            + (f": {' '.join(words)}" if words else "")))
                        pictures.extend(images)
                out.append(ToolMessage(
                    content=text or "(no content)",
                    tool_call_id=result["tool_use_id"],
                    status="error" if result.get("is_error") else "success"))
            trailing = plain + pictures
            if trailing:
                body = cls._user_content(trailing)
                if body:
                    out.append(HumanMessage(content=body))
        return _answer_every_call(out, verdicts)

    # -- the protocol ------------------------------------------------------
    def complete(self, messages: Sequence[Dict[str, Any]], *,
                 system: Optional[str] = None,
                 tools: Optional[Sequence[Dict[str, Any]]] = None,
                 images: Optional[Sequence[bytes]] = None,
                 output_format: Any = None,
                 max_tokens: Optional[int] = None) -> Reply:
        msgs = [dict(m) for m in messages]
        if images:
            if not msgs or msgs[-1]["role"] != "user":
                msgs.append(user())
            content = msgs[-1]["content"]
            if isinstance(content, str):
                content = [text_block(content)]
            msgs[-1] = {"role": "user",
                        "content": list(content)
                        + [image_block(p) for p in images]}

        offered = [_function_spec(t) for t in (tools or [])]
        if output_format is not None:
            if any(t["function"]["name"] == ANSWER_TOOL for t in offered):
                raise ValueError(
                    f"a pass tool may not be named {ANSWER_TOOL!r}: that is "
                    f"how this engine returns a structured answer")
            offered.append(_answer_spec(output_format))
            note = _ANSWER_NOTE_WITH_TOOLS if tools else _ANSWER_NOTE
            system = f"{system}\n\n{note}" if system else note

        learned = len(self.adaptations)
        model = self._model_for(self._cap_for(max_tokens))
        runnable = model.bind_tools(offered) if offered else model
        request = self._to_langchain(msgs, system)
        started = time.time()
        try:
            message = runnable.invoke(request)
        except Exception as exc:
            # Recorded, then raised as before: a failed call is a call
            # (BACKLOG RI-21). The model's own retries happen inside it.
            self.meter.fail(self.name, exc, seconds=time.time() - started,
                            adapted=self.adaptations[learned:])
            raise
        seconds = time.time() - started
        return self._reply(message, output_format, seconds,
                           adapted=self.adaptations[learned:])

    def _reply(self, message: Any, output_format: Any,
               seconds: float, adapted: Sequence[str] = ()) -> Reply:
        meta = getattr(message, "response_metadata", None) or {}
        self.served_by = str(meta.get("model_name") or meta.get("model")
                             or "") or self.name
        usage = _usage_of(message)
        incomplete = meta.get("incomplete_details")
        stop = _reason(meta.get("finish_reason") or meta.get("stop_reason")
                       or (incomplete.get("reason")
                           if isinstance(incomplete, dict) else "")
                       or meta.get("status"))
        entry = self.meter.add(self.name, usage, seconds,
                               served_by=self.served_by, stop=stop,
                               adapted=adapted)

        # A call whose arguments did not parse (a reply cut mid-call) is kept
        # by ChatOpenAI as text on ``invalid_tool_calls``, and sent back as a
        # call if left there. It is read for an answer once, then dropped
        # from the turn that goes back: nothing can run it.
        spoiled = [c.get("args") for c in
                   (getattr(message, "invalid_tool_calls", None) or [])
                   if c.get("name") == ANSWER_TOOL]
        if getattr(message, "invalid_tool_calls", None):
            message = message.model_copy(update={"invalid_tool_calls": []})
        message = _with_call_ids(message)
        text = _text_of(getattr(message, "content", ""))
        calls = list(getattr(message, "tool_calls", None) or [])
        answers = [c for c in calls if output_format is not None
                   and c.get("name") == ANSWER_TOOL]
        wanted = [c for c in calls if c not in answers]
        cut = stop in CUT_REASONS

        parsed = None
        notes: List[str] = []
        #: answer call id -> "" (a whole answer) or why it was not taken
        verdicts: Dict[str, str] = {}
        if output_format is not None:
            offered = ([(c["id"], c.get("args")) for c in answers]
                       + [(None, args) for args in spoiled])
            for call_id, args in offered:
                if cut:
                    answer, dropped, why = None, [], _CUT_ANSWER
                else:
                    answer, dropped, why = _validate(output_format, args)
                if dropped:
                    notes.append("unknown keys dropped from the answer: "
                                 + ", ".join(dropped))
                if answer is None:
                    notes.append(f"an answer was not taken: {why}")
                elif parsed is None:
                    parsed = answer
                if call_id:
                    verdicts[call_id] = "" if answer is not None else why
            if parsed is None and text and not cut:
                answer, dropped, _why = _validate(output_format,
                                                  _json_in(text))
                if answer is not None:
                    parsed = answer
                    if dropped:
                        notes.append("unknown keys dropped from the answer: "
                                     + ", ".join(dropped))
        if notes:
            logging.getLogger(__name__).warning("%s: %s", self.name,
                                                "; ".join(notes))
            # What the engine did with the answer, on the call's cost row.
            entry["notes"] = list(notes)

        blocks: List[Dict[str, Any]] = []
        if text:
            blocks.append(text_block(text))
        blocks.extend(tool_use_block(c["id"], c.get("name", ""),
                                     dict(c.get("args") or {}))
                      for c in calls)
        held = opaque_block(message)
        if verdicts:
            held["answers"] = verdicts
        blocks.append(held)
        reply = Reply(
            text=text,
            tool_calls=[ToolCall(c["id"], c.get("name", ""),
                                 dict(c.get("args") or {})) for c in wanted],
            parsed=parsed,
            stop_reason=stop,
            usage=usage,
            model=self.served_by,
            seconds=seconds,
            content=blocks,
            notes=notes,
        )
        if reply.cut_off:
            logging.getLogger(__name__).warning(
                "%s: a reply stopped at its output cap (%s tokens out); a "
                "structured answer cut there does not parse",
                self.name, usage.output_tokens)
        return reply


def _chat_model_name(model: Any) -> str:
    """What a chat model calls itself: its model name, id or API name."""
    for key in ("model_name", "model", "model_api_name", "model_id",
                "deployment_name"):
        try:
            value = getattr(model, key, None)
        except Exception:                                # noqa: BLE001
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _own_cap(model: Any) -> Optional[int]:
    """The per-reply output cap a chat model was built with, or None."""
    for key in _CAP_FIELDS:
        try:
            value = getattr(model, key, None)
        except Exception:                                # noqa: BLE001
            continue
        if isinstance(value, int) and not isinstance(value, bool) \
                and value > 0:
            return value
    return None


def _function_spec(tool: Dict[str, Any]) -> Dict[str, Any]:
    """A pass's tool (``name``, ``description``, ``input_schema``) as an
    OpenAI function spec, which every LangChain ``bind_tools`` takes."""
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""),
        "parameters": tool.get("input_schema")
        or {"type": "object", "properties": {}}}}


def _answer_spec(output_format: Any) -> Dict[str, Any]:
    """The answer tool for one structured answer: the answer's own schema,
    in the shape strict mode takes, as the tool's parameters."""
    what = getattr(output_format, "__name__", "answer")
    return {"type": "function", "function": {
        "name": ANSWER_TOOL,
        "description": (f"Your answer, as a {what}. Call this once, with the "
                        f"whole answer as the arguments."),
        "parameters": strict_schema(output_format)}}


def _answer_every_call(messages: List[Any],
                       verdicts: Optional[Dict[str, str]] = None
                       ) -> List[Any]:
    """``messages`` with a result after every tool call that had none -- the
    answer tool's own call above all, which a pass never answers -- each
    straight after the results that are there, before anything else.

    The answer call's result says what became of it: ``verdicts`` (call id
    -> why it was not taken, from :meth:`ChatModelEngine._reply`) names an
    answer that was not whole; an answer given in the same turn as a look
    the pass then ran was not taken; otherwise it was.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    verdicts = verdicts or {}

    out: List[Any] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        out.append(message)
        i += 1
        if not isinstance(message, AIMessage):
            continue
        wanted = [(c.get("id"), c.get("name"))
                  for c in (message.tool_calls or []) if c.get("id")]
        if not wanted:
            continue
        answered = set()
        while i < len(messages) and isinstance(messages[i], ToolMessage):
            answered.add(messages[i].tool_call_id)
            out.append(messages[i])
            i += 1
        looked = any(call_id in answered for call_id, name in wanted
                     if name != ANSWER_TOOL)
        for call_id, name in wanted:
            if call_id in answered:
                continue
            status = "success"
            if name != ANSWER_TOOL:
                content = _NOT_ANSWERED
            elif verdicts.get(call_id):
                content = _ANSWER_NOT_TAKEN.format(why=verdicts[call_id])
                status = "error"
            elif looked:
                content = _ANSWER_AFTER_LOOK
            else:
                content = _ANSWER_RECEIVED
            out.append(ToolMessage(content=content, tool_call_id=call_id,
                                   status=status))
    return out


def _with_call_ids(message: Any) -> Any:
    """The reply with an id on every tool call: a result has to name the
    call it answers, and not every route returns one."""
    calls = list(getattr(message, "tool_calls", None) or [])
    if not calls or all(c.get("id") for c in calls):
        return message
    import uuid
    fixed = [dict(c, id=c.get("id") or f"call_{uuid.uuid4().hex[:12]}")
             for c in calls]
    return message.model_copy(update={"tool_calls": fixed})


def _text_of(content: Any) -> str:
    """A reply's words: a string, or the text blocks of a list (thinking
    and tool blocks are not words)."""
    if isinstance(content, str):
        return content
    out: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(str(block.get("text") or ""))
    return "".join(out)


def _usage_of(message: Any) -> Usage:
    """A reply's tokens. LangChain counts cache reads and writes INTO the
    input; here they are apart, as :class:`Usage` keeps them. The reasoning
    tokens are LangChain's ``output_token_details.reasoning`` or the
    provider's own count in the response metadata (Chat Completions'
    ``completion_tokens_details``, Responses' ``output_tokens_details``),
    the larger where both say, and None where neither does (Claude reports
    no split)."""
    meta = getattr(message, "response_metadata", None) or {}
    raw = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(raw, dict):
        raw = {}
    um = getattr(message, "usage_metadata", None) or {}
    if not um:
        um = {"input_tokens": raw.get("prompt_tokens")
              or raw.get("input_tokens") or 0,
              "output_tokens": raw.get("completion_tokens")
              or raw.get("output_tokens") or 0}
    details = um.get("input_token_details") or {}
    read = int(details.get("cache_read") or 0)
    made = int(details.get("cache_creation") or 0)
    total = int(um.get("input_tokens") or 0)
    said = []
    out_details = um.get("output_token_details")
    if isinstance(out_details, dict):
        said.append(_count(out_details.get("reasoning")))
    for key in ("completion_tokens_details", "output_tokens_details"):
        part = raw.get(key)
        if isinstance(part, dict):
            said.append(_count(part.get("reasoning_tokens")))
    counted = [n for n in said if n is not None]
    thought = max(counted) if counted else None
    return Usage(input_tokens=max(0, total - read - made),
                 output_tokens=int(um.get("output_tokens") or 0),
                 cache_read_tokens=read, cache_write_tokens=made,
                 reasoning_tokens=thought)


def _json_in(text: str) -> Any:
    """The JSON object a reply wrote out as text, fenced or bare, or None."""
    body = str(text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = body.find("{"), body.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(body[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items()
                if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


def _validate(output_format: Any, args: Any
              ) -> Tuple[Any, List[str], str]:
    """``args`` as an ``output_format``: ``(answer, keys dropped, why not)``,
    the answer None when there is none.

    * Empty arguments are no answer. Foundry's classes hand back ``{}`` for
      a call cut mid-arguments, and many reading schemas accept ``{}``.
    * Strict mode's schema makes every field required, so a model may send
      null for a field that has a default: tried again with the nulls left
      out, letting the defaults stand.
    * Foundry sends tools without strict mode, so a model may add a key the
      schema forbids: the unknown keys are dropped and named, and the
      answer validated again. Any other error stands.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return None, [], "its arguments are not JSON"
    if not isinstance(args, dict):
        return None, [], "it is not a JSON object"
    if not args:
        return None, [], "it arrived empty"
    why = ""
    for candidate in (args, _without_nulls(args)):
        dropped: List[str] = []
        for _ in range(4):
            try:
                return output_format.model_validate(candidate), dropped, ""
            except Exception as exc:                     # noqa: BLE001
                errors = _errors_of(exc)
                why = ("it does not match the schema ("
                       + _error_text(exc, errors) + ")")
                extra = [tuple(e.get("loc") or ()) for e in errors
                         if e.get("type") == "extra_forbidden"]
                if not extra:
                    break
                candidate, removed = _drop_keys(candidate, extra)
                if not removed:
                    break
                dropped.extend(removed)
    return None, [], why


def _errors_of(exc: Exception) -> List[Dict[str, Any]]:
    """A pydantic validation error's list of errors; empty for anything
    else."""
    try:
        return list(exc.errors())                        # type: ignore[attr-defined]
    except Exception:                                    # noqa: BLE001
        return []


def _error_text(exc: Exception, errors: Sequence[Dict[str, Any]]) -> str:
    """The first few validation errors as ``where: what``."""
    if not errors:
        return f"{type(exc).__name__}: {exc}"[:300]
    parts = [f"{'.'.join(str(p) for p in e.get('loc') or ()) or 'answer'}: "
             f"{e.get('msg', '')}" for e in errors[:3]]
    more = len(errors) - 3
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")


def _drop_keys(value: Dict[str, Any], locations: Sequence[Tuple[Any, ...]]
               ) -> Tuple[Dict[str, Any], List[str]]:
    """A copy of ``value`` without the key at each location, and the dotted
    paths of the keys that were there to drop."""
    import copy

    out = copy.deepcopy(value)
    removed: List[str] = []
    for loc in locations:
        if not loc:
            continue
        node: Any = out
        for step in loc[:-1]:
            try:
                node = node[step]
            except (KeyError, IndexError, TypeError):
                node = None
                break
        if isinstance(node, dict) and loc[-1] in node:
            del node[loc[-1]]
            removed.append(".".join(str(p) for p in loc))
    return out, removed


# ---------------------------------------------------------------------------
# finding the engine the app is already holding
# ---------------------------------------------------------------------------

def engine_for(obj: Any, model: Optional[str] = None,
               meter: Optional[CostMeter] = None) -> Optional["Engine"]:
    """The ingest engine for whatever the host handed the agent, or None.

    The app builds its agents around its own engines -- ``NativeToolEngine``
    and ``PrompterBridgeEngine`` on the cluster, a LangChain chat model on
    every page, which the page wraps for its vision tools
    (``LangChainVisionEngine``). This finds the one to read a report on:

    * something that already satisfies the :class:`Engine` protocol comes
      back unchanged;
    * an app engine wrapping a Prompter (``prompter`` or ``_prompter``)
      becomes a :class:`PrompterEngine` over that Prompter;
    * a live Prompter itself becomes one directly;
    * a LangChain chat model -- itself, or the one inside the wrapper the
      page puts round it (``model`` / ``_model``) -- becomes a
      :class:`ChatModelEngine` over it (BACKLOG RI-03; on Foundry, the
      page's own ``palantir_models`` model), unless it holds a Prompter,
      which keeps the Prompter's engine on the page's own tier (the chat
      model's, else the one selected on the Prompter). A chat model is the one
      model the page holds: asked for a different one (``model``, a
      Funhouse tier for the page voter), there is none, and the caller
      falls back to the one engine it has;
    * anything else comes back **None**, which is an answer. A caller that
      cannot get an engine says so rather than falling back to a model nobody
      asked for: quietly reading a 400-page report on something else would
      be a bill and a number that belong to nobody.
    """
    if obj is None:
        return None
    if isinstance(obj, (ClaudeEngine, PrompterEngine, ChatModelEngine)):
        return obj
    complete = getattr(obj, "complete", None)
    if callable(complete):
        try:
            import inspect
            parameters = inspect.signature(complete).parameters
        except (TypeError, ValueError):           # a C-level callable
            parameters = {}
        if "output_format" in parameters:
            return obj                            # already one of ours
    for name in ("prompter", "_prompter"):
        inner = getattr(obj, name, None)
        if inner is not None and _looks_like_prompter(inner):
            # A chat model holding one is the page's: its own tier.
            tier = _page_tier(obj, inner) if _is_chat_model(obj) else ""
            return PrompterEngine(inner, model or tier or PROMPTER_MODELS[1],
                                  meter=meter)
    if _looks_like_prompter(obj):
        return PrompterEngine(obj, model or PROMPTER_MODELS[1], meter=meter)
    chat = _chat_model_in(obj)
    if chat is not None:
        for name in ("prompter", "_prompter"):
            inner = getattr(chat, name, None)
            if inner is not None and _looks_like_prompter(inner):
                return PrompterEngine(
                    inner, model or _page_tier(chat, inner)
                    or PROMPTER_MODELS[1], meter=meter)
        if model and model != _chat_model_name(chat):
            return None
        return ChatModelEngine(chat, meter=meter)
    return None


def _page_tier(chat: Any, prompter: Any) -> str:
    """The tier the page's own model calls: the one it was built with, else
    the one selected on the Prompter (``PrompterChatModel`` reads
    ``prompter.chat_model`` at every call), else none."""
    for value in (getattr(chat, "model", None),
                  getattr(prompter, "chat_model", None)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_chat_model(obj: Any) -> bool:
    """A LangChain chat model, by what it offers: it can be invoked and can
    have tools bound."""
    return (not isinstance(obj, str)
            and callable(getattr(obj, "invoke", None))
            and callable(getattr(obj, "bind_tools", None)))


def _chat_model_in(obj: Any) -> Any:
    """``obj`` if it is a chat model, else the chat model the wrapper holds
    (``LangChainVisionEngine.model``), else None."""
    if _is_chat_model(obj):
        return obj
    for name in ("model", "_model"):
        try:
            inner = getattr(obj, name, None)
        except Exception:                                # noqa: BLE001
            continue
        if inner is not None and _is_chat_model(inner):
            return inner
    return None


def _looks_like_prompter(obj: Any) -> bool:
    """Whether this object is a Funhouse Prompter, by what it offers.

    By capability rather than by class: the SDK's Prompter is imported in
    several places under several names and this package imports it nowhere.
    """
    return callable(getattr(obj, "chat", None)) and hasattr(obj, "client")
