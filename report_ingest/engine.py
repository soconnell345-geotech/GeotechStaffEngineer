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
:func:`pdf_block` goes out on :class:`ChatModelEngine` alone, and only on a
route that takes one (:func:`takes_pdf`); every engine refuses it elsewhere.

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
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "Engine", "Reply", "ToolCall", "Usage", "CostMeter", "ClaudeEngine",
    "RequestPacer", "shared_pacer", "pace_engines", "estimate_request",
    "LARGE_REQUEST_FRACTION", "PACER_WINDOW_SECONDS",
    "engine_for", "ChatModelEngine", "ANSWER_TOOL", "CUT_REASONS",
    "DEFAULT_MAX_TOKENS", "CallCeilingReached",
    "PrompterEngine", "MODEL_PRICES", "PROMPTER_MODELS", "PROMPTER_PRICES",
    "price_for", "strict_schema", "error_kind", "FAILURE_KINDS",
    "is_rate_limit", "RATE_LIMIT_WORDS",
    "text_block", "image_block", "tool_use_block", "tool_result_block",
    "opaque_block", "user", "assistant",
    "image_media_type", "IMAGE_SIGNATURES",
    "IMAGE_TOKENS_EST", "CHARS_PER_TOKEN_EST", "PIXELS_PER_TOKEN_EST",
    "PDF_PAGE_PICTURE_TOKENS_EST", "estimate_tokens", "image_size",
    "picture_tokens_est", "pdf_tokens_est", "schema_tokens_est",
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


def pdf_block(data: bytes) -> Dict[str, Any]:
    """A PDF for the model to read: its text layer and its pages' pictures
    both, where the provider reads a PDF that way (Anthropic's ``document``
    block; BACKLOG RI-86, RI-87). Bytes held in memory, like a picture.

    Only :class:`ChatModelEngine` sends one, and only when its
    :attr:`~ChatModelEngine.accepts_pdf` is on; a route that cannot carry
    it refuses it before the request. Put it first in its turn, ahead of
    the words, as Anthropic advises.
    """
    return {"type": "pdf", "data": bytes(data)}


def _no_pdf_in(result_body: Sequence[Any]) -> None:
    """Refuse a PDF inside a tool result: a tool message holds text on
    every route, so it would be lost there."""
    if any(isinstance(b, dict) and b.get("type") == "pdf"
           for b in result_body):
        raise ValueError(
            "a PDF goes in a user turn, not in a tool result: a tool "
            "message holds text on every route, and the PDF would be lost")


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


# -- sizing a request before it is sent (BACKLOG RI-110, RI-121) ------------

#: What one page picture is counted as when a request is sized against a
#: per-request token limit (BACKLOG RI-110). A deliberately high figure for
#: the log reader's pictures: a letter page at its 110 dpi is about 1,500
#: tokens on Claude (width x height / 750) and a handful of tiles on GPT.
IMAGE_TOKENS_EST = 1700
#: Characters per token when words are sized.
CHARS_PER_TOKEN_EST = 3.5
#: Pixels per token of a picture on Claude: width x height / 750 (the vision
#: page of Claude's API documentation). :func:`picture_tokens_est`.
PIXELS_PER_TOKEN_EST = 750
#: What the page picture of one page of a PDF document block is counted as
#: (:func:`pdf_tokens_est`, BACKLOG RI-121): the most one picture costs on
#: Claude's high-resolution tier (planlens' ``claude-hires`` budget, 2,576 px
#: and 4,784 patches of 28 px, the size the model-first pictures route draws
#: a page at for Claude 5). The API draws each PDF page as a picture and does
#: not say at what size, so the count is the largest it can be: a request
#: counted too large is split, one counted too small is refused.
PDF_PAGE_PICTURE_TOKENS_EST = 4784


def _blocks_of(message: Dict[str, Any]) -> List[Any]:
    content = message.get("content")
    return list(content) if isinstance(content, list) else [content]


def image_size(data: bytes) -> Optional[Tuple[int, int]]:
    """``(width, height)`` in pixels, read off a PNG's or a JPEG's own
    header, or None when the bytes are neither."""
    data = bytes(data or b"")
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" \
            and len(data) >= 24:
        return (int.from_bytes(data[16:20], "big"),
                int.from_bytes(data[20:24], "big"))
    if data[:2] != b"\xff\xd8":
        return None
    at = 2
    while at + 4 <= len(data):
        if data[at] != 0xFF:
            return None
        marker = data[at + 1]
        if marker == 0xFF:                       # a fill byte
            at += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            at += 2                              # a marker with no length
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if at + 9 > len(data):
                return None
            return (int.from_bytes(data[at + 7:at + 9], "big"),
                    int.from_bytes(data[at + 5:at + 7], "big"))
        at += 2 + int.from_bytes(data[at + 2:at + 4], "big")
    return None


def picture_tokens_est(block: Dict[str, Any]) -> int:
    """One picture block's input tokens, by its own size (BACKLOG RI-121):
    its pixels at :data:`PIXELS_PER_TOKEN_EST` a token, and never less than
    :data:`IMAGE_TOKENS_EST`, which is also the count when the size cannot
    be read. On GPT, which counts a picture in tiles, this is too many: a
    request is then split sooner than it needs to be."""
    size = image_size(block.get("png") or b"")
    if not size:
        return IMAGE_TOKENS_EST
    width, height = size
    return max(IMAGE_TOKENS_EST, -(-width * height // PIXELS_PER_TOKEN_EST))


def pdf_tokens_est(data: bytes) -> int:
    """One PDF document block's input tokens (BACKLOG RI-121): the API reads
    each page as both its text layer and a picture of the page, so each page
    counts one picture (:data:`PDF_PAGE_PICTURE_TOKENS_EST`) and the words
    of its text layer at :data:`CHARS_PER_TOKEN_EST` a token. A scanned page
    with no text layer is its picture alone. Bytes no PDF reader opens count
    one page picture."""
    try:
        import pymupdf

        with pymupdf.open(stream=bytes(data), filetype="pdf") as doc:
            words = sum(len(page.get_text()) for page in doc)
            return int(doc.page_count * PDF_PAGE_PICTURE_TOKENS_EST
                       + words / CHARS_PER_TOKEN_EST)
    except Exception:                    # sized, not read: never stop a run
        return PDF_PAGE_PICTURE_TOKENS_EST


def estimate_tokens(messages: Sequence[Dict[str, Any]],
                    picture: Any = None) -> int:
    """A rough count of the input tokens a conversation costs (BACKLOG
    RI-110): pictures at :data:`IMAGE_TOKENS_EST` each, or at
    ``picture(block)`` when given (:func:`picture_tokens_est`); words at
    :data:`CHARS_PER_TOKEN_EST` a token; a PDF document block at
    :func:`pdf_tokens_est`."""
    total = 0.0
    for message in messages:
        for block in _blocks_of(message):
            if isinstance(block, dict) and block.get("type") == "image":
                total += picture(block) if picture else IMAGE_TOKENS_EST
            elif isinstance(block, dict) and block.get("type") == "pdf":
                total += pdf_tokens_est(block.get("data") or b"")
            elif isinstance(block, dict) and "text" in block:
                total += len(str(block["text"])) / CHARS_PER_TOKEN_EST
            elif block:
                total += len(str(block)) / CHARS_PER_TOKEN_EST
    return int(total)


def schema_tokens_est(schema: Any) -> int:
    """The input tokens an answer schema costs when it is offered as the
    answer tool (BACKLOG RI-121): its JSON schema's words at
    :data:`CHARS_PER_TOKEN_EST` a token; 0 for no schema."""
    if schema is None:
        return 0
    try:
        shown = (schema.model_json_schema() if hasattr(
            schema, "model_json_schema") else schema)
        return int(len(json.dumps(shown, default=str)) / CHARS_PER_TOKEN_EST)
    except Exception:                    # sized, not read: never stop a run
        return 0


def _flat_for_size(messages: Sequence[Dict[str, Any]]
                   ) -> List[Dict[str, Any]]:
    """The messages with each tool result's own blocks put in its place and
    each opaque block as its text, so :func:`estimate_tokens` sizes a
    picture a tool returned by its pixels rather than by the text of its
    bytes."""
    out: List[Dict[str, Any]] = []
    for message in messages:
        blocks: List[Any] = []
        for block in _blocks_of(message):
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "tool_result":
                body = block.get("content")
                blocks.extend(body if isinstance(body, list)
                              else [str(body or "")])
            elif kind == "opaque":
                blocks.append(str(block.get("data") or ""))
            else:
                blocks.append(block)
        out.append({"role": message.get("role"), "content": blocks})
    return out


def estimate_request(messages: Sequence[Dict[str, Any]], *,
                     system: Optional[str] = None,
                     tools: Any = None,
                     output_format: Any = None) -> Optional[int]:
    """A request's input tokens, estimated before it is sent (BACKLOG
    RI-126, RI-127): its messages by :func:`estimate_tokens` (pictures by
    their pixels, :func:`picture_tokens_est`, a tool's pictures and words
    as their own blocks), the system prompt's and the tool specs' words at
    :data:`CHARS_PER_TOKEN_EST` a token, and the answer schema
    (:func:`schema_tokens_est`). None when it cannot be sized: an estimate
    never stops a request."""
    try:
        total = estimate_tokens(_flat_for_size(messages),
                                picture=picture_tokens_est)
        words = len(system or "")
        if tools:
            words += len(json.dumps(list(tools), default=str))
        return int(total + words / CHARS_PER_TOKEN_EST
                   + schema_tokens_est(output_format))
    except Exception:                       # noqa: BLE001 - sized, not read
        return None


def _utc_iso(seconds: float) -> str:
    """An epoch time as UTC ISO 8601 to the millisecond, ``Z`` for UTC."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(seconds), timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


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


#: What a rate limit says of itself, lower case, in the words a provider
#: puts its own message in (:func:`_own_words`). Read off the WORDS, never a
#: 400 status alone (BACKLOG RI-139): Foundry's proxy sends a rate limit as
#: a 400, and a 400 that says none of these is a refused request like any
#: other.
RATE_LIMIT_WORDS: Tuple[str, ...] = ("too many requests", "rate limit",
                                     "rate_limit", "throttl")

#: The fields of an error body a provider writes its own words in, under
#: ``error`` or at the top (OpenAI's ``error.message`` / ``code`` /
#: ``type``, Anthropic's the same).
_BODY_WORD_FIELDS: Tuple[str, ...] = ("message", "code", "type")


def _status_of(error: Any) -> Optional[int]:
    """The HTTP status an error carries as ``status_code``, on itself or on
    its ``response``; None when it carries none."""
    for holder in (error, getattr(error, "response", None)):
        try:
            code = getattr(holder, "status_code", None)
        except Exception:                   # noqa: BLE001 - never raise here
            continue
        if isinstance(code, int) and not isinstance(code, bool):
            return code
    return None


def _own_words(error: Any) -> str:
    """What an error says of itself, lower case, for :func:`is_rate_limit`
    (BACKLOG RI-139): its message, and the fields of its body a provider
    writes its own words in (:data:`_BODY_WORD_FIELDS`, under ``error`` or
    at the top; ``error`` itself where it is a string).

    Never the rest of a body: a validation error can echo the request back
    ("field required ... the outlet valve was throttled"), and an SDK's
    message carries the whole body after the status ("Error code: 422 -
    {...}"). So where the body is structured, the message is read only up
    to the first ``{`` or ``[``; a body that is plain text is read as its
    first line, 200 characters at most, and the message without that body.
    ``""`` where nothing will read."""
    try:
        text = _message_of(error)
        try:
            body = getattr(error, "body", None)
        except Exception:                   # noqa: BLE001 - never raise here
            body = None
        parts: List[str] = []
        if isinstance(body, (dict, list)):
            cut = [i for i in (text.find("{"), text.find("[")) if i >= 0]
            parts.append(text[:min(cut)] if cut else text)
            if isinstance(body, dict):
                inner = body.get("error")
                if isinstance(inner, str):
                    parts.append(inner)
                for holder in (inner, body):
                    if isinstance(holder, dict):
                        parts.extend(str(holder[key])
                                     for key in _BODY_WORD_FIELDS
                                     if isinstance(holder.get(key),
                                                   (str, int)))
        elif isinstance(body, str) and body.strip():
            own = body.strip()
            parts.append(text.replace(own, " "))
            parts.append(own.splitlines()[0][:200])
        else:
            parts.append(text)
        return " ".join(parts).lower()
    except Exception:                       # noqa: BLE001 - never raise here
        return ""


def is_rate_limit(error: Any) -> bool:
    """Whether an error is the provider saying 'too many requests': its
    class name (``RateLimitError``, a host's ``RateLimitGaveUp``), a 429
    status (``status_code``, or "Error code: 429" in its message), or
    :data:`RATE_LIMIT_WORDS` in its own words (:func:`_own_words`), whatever
    the status. Never raises."""
    name = type(error).__name__.lower() if isinstance(error, BaseException) \
        else ""
    if "ratelimit" in name or _status_of(error) == 429:
        return True
    said = _own_words(error)
    if "error code: 429" in said:
        return True
    return any(word in said for word in RATE_LIMIT_WORDS)


def error_kind(error: Any) -> str:
    """Which of :data:`FAILURE_KINDS` an error is, read off its class name
    and its text; a rate limit as :func:`is_rate_limit` reads one. A prompt
    past the model's context is ``too_large`` (Claude says "prompt is too
    long", OpenAI "maximum context length")."""
    name = type(error).__name__.lower() if isinstance(error, BaseException) \
        else ""
    text = _message_of(error).lower()
    if is_rate_limit(error):
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


#: Held while a :class:`CostMeter` records a call, so that pages described
#: on several threads at once (BACKLOG RI-73) never lose a count. One lock
#: for every meter: a record is a few additions, and a lock on the instance
#: would not survive the copies some callers take of a meter.
_METER_LOCK = threading.RLock()


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

    EVERY ROW SAYS WHEN (BACKLOG RI-126): ``started``, the moment its
    request went out, as UTC ISO 8601 to the millisecond, and ``duration``,
    the seconds from then until it came back -- the request's own, without
    the waits before it, and with any retries a provider's client made
    inside it. An engine passes both; a row recorded without them is dated
    by the meter's clock when it was recorded, less its seconds. A failed
    request's row carries its ``input_tokens`` as the error said, or, where
    the error said nothing, as the engine estimated them before sending
    (:func:`estimate_request`), marked ``input_tokens_estimated``; an
    estimate is never added to the totals. A request held back by a
    :class:`RequestPacer` (BACKLOG RI-127) carries the seconds it was held,
    ``paced``, and :attr:`paced_seconds` adds them up (in :meth:`to_dict`
    only when there were any); they are in the call's ``seconds`` too. A
    call that was held and whose request the route then refused to send
    (BACKLOG RI-139) is no request: it is not in ``calls`` or ``failed``,
    but its hold is a row of its own (:meth:`unsent`) and in
    :attr:`paced_seconds`.

    CALLS ON SEVERAL THREADS AT ONCE (BACKLOG RI-139): an engine that sends
    a call's request again passes the call's number (``call_no``) with each
    later attempt, so the attempt is filed under its own call even when
    another call's rows were recorded in between.

    Recording never stands in the way of the call it records: an error
    whose text or body will not read is recorded with what can be read, and
    anything else that goes wrong while recording is logged and dropped, so
    the engine raises the provider's error, or returns the reply, as it
    would with no meter at all.

    What this meter cannot see is a retry INSIDE a provider's own client --
    the Claude SDK's, or a LangChain chat model's -- or in a host's adapter
    in front of it: only what reaches the engine is recorded.
    """

    def __init__(self, clock: Any = None) -> None:
        #: Dates a row recorded without a start time; a test replaces it.
        self._clock = clock or time.time
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
        self.paced_seconds = 0.0
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
               adapted: Sequence[str], *, started: Optional[float] = None,
               duration: Optional[float] = None,
               paced: float = 0.0,
               call_no: Optional[int] = None) -> Dict[str, Any]:
        """One :attr:`per_call` row; attempt 1 starts a new call. ``started``
        is the epoch time the request went out and ``duration`` its own
        seconds (BACKLOG RI-126); without them the row is dated by
        :attr:`_clock`, less ``seconds``. ``call_no``: the call a later
        attempt belongs to, as its first row was numbered (BACKLOG RI-139);
        without it a later attempt belongs to the call begun last."""
        if call_no is not None:
            number = int(call_no)
        else:
            if int(attempt) <= 1 or not self._call_no:
                self._call_no += 1
            number = self._call_no
        entry: Dict[str, Any] = {"call": number,
                                 "attempt": max(1, int(attempt)),
                                 "model": model}
        if served_by:
            entry["served_by"] = served_by
        took = float(duration if duration is not None else (seconds or 0.0))
        when = (float(started) if started is not None
                else float(self._clock()) - took)
        entry["started"] = _utc_iso(when)
        entry["duration"] = round(max(0.0, took), 2)
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
        if paced:
            entry["paced"] = round(float(paced), 1)
            self.paced_seconds += float(paced)
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
            stop: str = "", adapted: Sequence[str] = (),
            started: Optional[float] = None,
            duration: Optional[float] = None,
            paced: float = 0.0,
            call_no: Optional[int] = None) -> Dict[str, Any]:
        """A reply. Returns its :attr:`per_call` row, for a note the engine
        has only after reading the reply. Safe from several threads at once
        (:data:`_METER_LOCK`). ``started``, ``duration``, ``paced``,
        ``call_no``: see :meth:`_entry`."""
        with _METER_LOCK:
            return self._add(model, usage, seconds, served_by,
                             attempt=attempt, stop=stop, adapted=adapted,
                             started=started, duration=duration,
                             paced=paced, call_no=call_no)

    def _add(self, model: str, usage: Usage, seconds: float,
             served_by: Optional[str] = None, *, attempt: int = 1,
             stop: str = "", adapted: Sequence[str] = (),
             started: Optional[float] = None,
             duration: Optional[float] = None,
             paced: float = 0.0,
             call_no: Optional[int] = None) -> Dict[str, Any]:
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
                                seconds, adapted, started=started,
                                duration=duration, paced=paced,
                                call_no=call_no)
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
             usage: Optional[Usage] = None,
             estimated_in: Optional[int] = None,
             started: Optional[float] = None,
             duration: Optional[float] = None,
             paced: float = 0.0,
             call_no: Optional[int] = None) -> Dict[str, Any]:
        """A request that came back with an error: :meth:`_fail`, under
        :data:`_METER_LOCK`."""
        with _METER_LOCK:
            return self._fail(model, error, kind=kind, seconds=seconds,
                              attempt=attempt, retried=retried,
                              waited=waited, adapted=adapted,
                              served_by=served_by, usage=usage,
                              estimated_in=estimated_in, started=started,
                              duration=duration, paced=paced,
                              call_no=call_no)

    def _fail(self, model: str, error: Any, *, kind: Optional[str] = None,
              seconds: float = 0.0, attempt: int = 1, retried: bool = False,
              waited: float = 0.0, adapted: Sequence[str] = (),
              served_by: Optional[str] = None,
              usage: Optional[Usage] = None,
              estimated_in: Optional[int] = None,
              started: Optional[float] = None,
              duration: Optional[float] = None,
              paced: float = 0.0,
              call_no: Optional[int] = None) -> Dict[str, Any]:
        """A request that came back with an error instead of a reply.

        ``retried``: the engine sent it again (``waited`` seconds first);
        ``seconds``: the wall clock of a call this error ENDED, waits
        included -- left 0 on one that was sent again, whose time the call's
        last row carries. ``usage``: what the error response said it spent,
        when it said (:func:`_usage_in_error` looks). ``estimated_in``: the
        engine's estimate of the request's input tokens (BACKLOG RI-126),
        written on the row, marked as an estimate, where the error said no
        input count; never added to the totals. ``started``, ``duration``,
        ``paced``, ``call_no``: see :meth:`_entry`. Never raises: the caller
        is about to raise the provider's error, and a failure here must not
        take its place.
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
                                adapted, started=started, duration=duration,
                                paced=paced, call_no=call_no)
            if "input_tokens" not in entry and estimated_in is not None:
                entry["input_tokens"] = int(estimated_in)
                entry["input_tokens_estimated"] = True
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

    def unsent(self, model: str, error: Any, *, paced: float,
               started: Optional[float] = None) -> Dict[str, Any]:
        """A call a :class:`RequestPacer` held whose request the route then
        refused to send (BACKLOG RI-139): a PDF block a route cannot send,
        a backend with no client for a tool loop.

        No request went out, so it is no call and no failed request:
        ``calls``, ``failed``, the tokens and the seconds are untouched. The
        hold was spent all the same, so it is recorded as a sent call's is:
        a row of its own (``outcome: "not_sent"``, its ``paced``, ``started``
        when the hold ended, and the refusal's first line) and the seconds
        in :attr:`paced_seconds`. A call nothing held records nothing, as
        before. Never raises."""
        if not paced:
            return {}
        try:
            with _METER_LOCK:
                entry = self._entry(model, None, 1, None, 0.0, (),
                                    started=started, duration=0.0,
                                    paced=paced)
                entry.update({"outcome": "not_sent",
                              "error": _first_line(error)})
                return entry
        except Exception:                   # noqa: BLE001 - see the docstring
            logging.getLogger(__name__).warning(
                "the cost record could not record a held request to %s",
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
        # Only when a pacer held a request (BACKLOG RI-127): a run with no
        # pacer writes the cost block it always wrote.
        paced = ({"paced_seconds": round(self.paced_seconds, 1)}
                 if self.paced_seconds else {})
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
            **paced,
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


# -- pacing to a per-minute token limit (BACKLOG RI-127) --------------------

#: A request is LARGE, for :class:`RequestPacer`, when its estimated input
#: is above this fraction of the per-minute budget.
LARGE_REQUEST_FRACTION = 0.5

#: The window a per-minute limit counts over, in seconds.
PACER_WINDOW_SECONDS = 60.0

#: Seconds to wait before each retry of a rate-limited request, in order:
#: :attr:`PrompterEngine.RATE_LIMIT_WAITS`, and the step a
#: ``retry-after: 0`` takes (:meth:`RequestPacer.backoff`).
RATE_LIMIT_LADDER: Tuple[float, ...] = (15.0, 30.0, 60.0, 120.0, 120.0,
                                        120.0)


def _retry_after_of(exc: Any) -> Optional[float]:
    """The provider's own wait, when its error response carried a
    ``retry-after`` header in seconds; None when it did not, or it will not
    read as a number of seconds of 0 or more (a date, a negative, an
    infinity). A ``0`` reads as 0.0, not as None (BACKLOG RI-139):
    :meth:`RequestPacer.backoff` says what wait it gets."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        if value is None:
            value = headers.get("Retry-After")
        if value is None:
            return None
        seconds = float(value)
    except (TypeError, ValueError, AttributeError):
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")) \
            or seconds < 0.0:
        return None
    return seconds


class RequestPacer:
    """Holds requests back so a large one does not fill a per-minute token
    limit for the next (BACKLOG RI-127).

    Foundry run 5 (2026-10-26) showed the limit that stopped a log item is
    per minute, not per request: two requests of about 118 k and 127 k input
    tokens each went through after a few rate-limit retries, and the 6 k
    request that followed waited 72 s over seven. So:

    * after a LARGE request -- estimated input (:func:`estimate_request`)
      above ``large_fraction`` of ``tokens_per_minute`` -- comes back, the
      next request to the same model waits until ``window`` seconds have
      passed since then (:meth:`wait`). The window counts from when the
      large request came BACK, not from when it went out: the provider
      counted its tokens at some moment in between, which the engine cannot
      see when the provider's own client retried inside the request;
    * a rate-limited request whose response carries no ``retry-after``
      waits at least a whole window before it is sent again
      (:meth:`backoff`), so a retry is not spent inside the window that
      refused it; and the next request to that model waits too
      (:meth:`hold`). A ``retry-after`` the provider sends is honoured as
      it is. A ``0`` takes the engine's own ladder step instead (BACKLOG
      RI-139): a provider stuck on 0 would otherwise be sent every retry
      back to back, and the next request at once; the step is the wait
      the ladder gives a rate limit, without the window's minimum, which
      is for a provider that names no wait at all.

    Each model has its own window: a large request on one model does not
    hold a request on another. One pacer serves every engine it is given,
    and :func:`shared_pacer` hands the same one to every run in a process
    with the same settings, so the window carries from one report, and one
    ``ingest_report`` call, to the next. Safe from several threads at once;
    a wait is spent outside the lock.

    ``clock`` and ``sleep`` are ``time.time`` and ``time.sleep`` unless a
    test gives others. :attr:`held` and :attr:`held_seconds` count the
    requests held back and the seconds they waited.
    """

    def __init__(self, tokens_per_minute: int, *,
                 large_fraction: float = LARGE_REQUEST_FRACTION,
                 window: float = PACER_WINDOW_SECONDS,
                 clock: Any = None, sleep: Any = None) -> None:
        if isinstance(tokens_per_minute, bool) or int(tokens_per_minute) <= 0:
            raise ValueError(f"tokens_per_minute must be a positive number "
                             f"of tokens; got {tokens_per_minute!r}")
        if not 0.0 < float(large_fraction) <= 1.0:
            raise ValueError(f"large_fraction must be above 0 and at most 1; "
                             f"got {large_fraction!r}")
        if float(window) <= 0.0:
            raise ValueError(f"window must be a positive number of seconds; "
                             f"got {window!r}")
        self.tokens_per_minute = int(tokens_per_minute)
        self.large_fraction = float(large_fraction)
        self.window = float(window)
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        #: model -> the epoch time before which its next request is held.
        self._until: Dict[str, float] = {}
        self.held = 0
        self.held_seconds = 0.0

    @property
    def large_tokens(self) -> int:
        """The estimated input above which a request is large."""
        return int(self.tokens_per_minute * self.large_fraction)

    def is_large(self, estimate: Optional[int]) -> bool:
        return estimate is not None and int(estimate) > self.large_tokens

    def _hold_until(self, model: str, until: float) -> None:
        with self._lock:
            key = str(model)
            self._until[key] = max(self._until.get(key, 0.0), float(until))

    def wait(self, model: str) -> float:
        """Before a request to ``model``: wait out what is left of a window
        a large request or a rate limit opened. Returns the seconds waited
        (0.0 when nothing held it)."""
        with self._lock:
            left = self._until.get(str(model), 0.0) - float(self._clock())
        if left <= 0.0:
            return 0.0
        self._sleep(left)
        with self._lock:
            self.held += 1
            self.held_seconds += left
        return left

    def sent(self, model: str, estimate: Optional[int]) -> None:
        """A request to ``model`` came back (a reply or an error): a large
        one holds the next request until a window from now."""
        if self.is_large(estimate):
            self._hold_until(model, float(self._clock()) + self.window)

    def backoff(self, retry_after: Optional[float],
                ladder: float = 0.0) -> float:
        """The wait before a rate-limited request is sent again: the
        provider's ``retry-after`` when it named a wait; for a ``0``, the
        engine's own step (``ladder``, or the ladder's first step when the
        engine has none), without the window's minimum (BACKLOG RI-139);
        for none, the engine's step and never less than a whole window."""
        if retry_after is None:
            return max(float(ladder or 0.0), self.window)
        if float(retry_after) <= 0.0:
            return float(ladder) if ladder else RATE_LIMIT_LADDER[0]
        return float(retry_after)

    def hold(self, model: str, seconds: float) -> None:
        """A rate limit on ``model``: its next request waits ``seconds``
        from now (:meth:`backoff`'s answer)."""
        self._hold_until(model, float(self._clock()) + float(seconds))

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens_per_minute": self.tokens_per_minute,
                "large_fraction": self.large_fraction,
                "large_tokens": self.large_tokens, "window": self.window,
                "held": self.held, "held_seconds": round(self.held_seconds, 1)}


_SHARED_PACERS: Dict[Tuple[int, float, float], RequestPacer] = {}
_SHARED_LOCK = threading.Lock()


def shared_pacer(tokens_per_minute: int, *,
                 large_fraction: float = LARGE_REQUEST_FRACTION,
                 window: float = PACER_WINDOW_SECONDS) -> RequestPacer:
    """The process's one :class:`RequestPacer` for these settings, made on
    first use: a per-minute limit belongs to the deployment, not to one
    report, so the window a report's last large request opened holds the
    next report's first request too."""
    key = (int(tokens_per_minute), float(large_fraction), float(window))
    with _SHARED_LOCK:
        if key not in _SHARED_PACERS:
            _SHARED_PACERS[key] = RequestPacer(
                key[0], large_fraction=key[1], window=key[2])
        return _SHARED_PACERS[key]


def pace_engines(engines: Sequence[Any], pacer: Optional[RequestPacer]
                 ) -> List[Any]:
    """Put ``pacer`` on each engine here that has none, looking through the
    wrappers a run puts round an engine (each keeps it as ``_inner``).
    Returns the engines it was put on, so the caller can take it off again
    (``engine.pacer = None``); an engine that already has a pacer keeps it,
    and anything that is not one of this module's engines is left alone."""
    put: List[Any] = []
    if pacer is None:
        return put
    for engine in engines:
        # A few wrappers deep at most: a stand-in that makes up any
        # attribute asked for would otherwise be followed for ever.
        for _ in range(8):
            if engine is None or isinstance(
                    engine, (ClaudeEngine, PrompterEngine, ChatModelEngine)):
                break
            engine = getattr(engine, "_inner", None)
        if isinstance(engine, (ClaudeEngine, PrompterEngine,
                               ChatModelEngine)) and engine.pacer is None \
                and not any(engine is e for e in put):
            engine.pacer = pacer
            put.append(engine)
    return put


class _Request:
    """One call's clock, its size and its pacing, for the cost record
    (BACKLOG RI-126) and the engine's pacer (RI-127).

    Made as a call begins: with a pacer, it waits out any window first.
    ``began`` is when the call began (the call's ``seconds`` count from it,
    waits included), ``sent`` when the request under way went out (an
    engine that sends again calls :meth:`send`), ``paced`` the seconds the
    pacer held it. The size is estimated once, and only when asked: on a
    failure, or for a pacer -- with neither, nothing is estimated.

    For an engine that sends a call's request again (:class:`PrompterEngine`)
    it is also the call's own record (BACKLOG RI-139): ``attempt``, the
    attempt the call is on; ``out``, whether any of its requests has gone
    out (:meth:`send`); ``number``, the call number its first cost row
    was filed under (:meth:`keep`), which its later attempts are filed
    under; and ``ladder_wait``, the last wait it took on the engine's
    ladder, which a ``retry-after: 0`` on its last request holds the
    model's next request for (:meth:`came_back`). It is made for one call
    and handed down that call's path, never kept on the engine, because
    calls on several threads at once share one engine (``range_workers``
    above 1) and one call's clock, size or attempt must not land on
    another's row."""

    def __init__(self, engine: Any, model: str, size: Any) -> None:
        self.pacer: Optional[RequestPacer] = getattr(engine, "pacer", None)
        self.clock = getattr(engine, "_clock", None) or time.time
        self.model = str(model)
        self._sizer = size
        self._sized = False
        self._size: Optional[int] = None
        self.attempt = 1
        self.out = False
        self.number: Optional[int] = None
        self.ladder_wait: Optional[float] = None
        self.began = float(self.clock())
        self.paced = self.pacer.wait(self.model) if self.pacer else 0.0
        self.sent = float(self.clock()) if self.paced else self.began
        self._paced_told = False

    def keep(self, entry: Optional[Dict[str, Any]]) -> None:
        """The call's number off its first cost row (an empty row, from a
        record that failed, leaves it unset)."""
        if self.number is None and entry and entry.get("call") is not None:
            self.number = int(entry["call"])

    def take_paced(self) -> float:
        """:attr:`paced` the first time, 0.0 after: the hold is on the row
        of the request it held, not on the retries after it."""
        if self._paced_told:
            return 0.0
        self._paced_told = True
        return self.paced

    def size(self) -> Optional[int]:
        if not self._sized:
            self._sized = True
            try:
                self._size = self._sizer()
            except Exception:               # noqa: BLE001 - sized, not read
                self._size = None
        return self._size

    def send(self) -> None:
        """Another request of this call goes out now."""
        self.out = True
        self.sent = float(self.clock())

    def took(self) -> float:
        """The request under way's own seconds so far."""
        return max(0.0, float(self.clock()) - self.sent)

    def came_back(self, error: Any = None) -> None:
        """Tell the pacer a request came back: a large one, or a rate limit,
        holds the next request to this model. A rate limit that named a
        wait of 0 holds it for the last step this call took on the ladder,
        or the ladder's first step (BACKLOG RI-139); one that named none
        for a window, as before. Never raises."""
        if self.pacer is None:
            return
        try:
            if error is not None and error_kind(error) == "rate_limit":
                named = _retry_after_of(error)
                step = (self.ladder_wait or 0.0) if named == 0.0 else 0.0
                self.pacer.hold(self.model, self.pacer.backoff(named, step))
            self.pacer.sent(self.model, self.size())
        except Exception:                   # noqa: BLE001 - never in the way
            logging.getLogger(__name__).warning(
                "the pacer could not record a request to %s", self.model,
                exc_info=True)


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
                 client: Any = None,
                 pacer: Optional[RequestPacer] = None) -> None:
        self.model = str(model)
        self.name = self.model
        self.meter = meter if meter is not None else CostMeter()
        self.max_tokens = int(max_tokens)
        #: Paces requests to a per-minute token limit (BACKLOG RI-127);
        #: None, the default, sends each request as it comes.
        self.pacer: Optional[RequestPacer] = pacer
        #: The clock a call is timed by; a test replaces it.
        self._clock = time.time
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
        call = _Request(self, self.model, lambda: estimate_request(
            msgs, system=system, tools=tools, output_format=output_format))
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
            self.meter.fail(self.model, exc,
                            seconds=self._clock() - call.began,
                            estimated_in=call.size(), started=call.sent,
                            duration=call.took(), paced=call.paced)
            call.came_back(exc)
            raise
        seconds = self._clock() - call.began
        took = call.took()
        call.came_back()

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
                       stop=str(getattr(response, "stop_reason", "") or ""),
                       started=call.sent, duration=took, paced=call.paced)
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
                 prefer_chat_for_single_calls: bool = True,
                 pacer: Optional[RequestPacer] = None) -> None:
        self._prompter = prompter
        self.model = str(model)
        self.name = self.model
        self.meter = meter if meter is not None else CostMeter()
        self.max_tokens = int(max_tokens)
        self.prefer_chat = bool(prefer_chat_for_single_calls)
        #: Paces requests to a per-minute token limit (BACKLOG RI-127);
        #: None, the default, sends each request as it comes and waits on
        #: :attr:`RATE_LIMIT_WAITS` alone.
        self.pacer: Optional[RequestPacer] = pacer
        #: The clock a call is timed by; a test replaces it. A call's own
        #: clock, size, attempt and number live on its :class:`_Request`,
        #: handed down the call's path and never kept here (BACKLOG RI-139).
        self._clock = time.time
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

    def _sent_again(self, call: _Request, error: Any, kind: str, *,
                    waited: float = 0.0, lesson: str = "") -> None:
        """Record a request of ``call`` that came back with an error and is
        about to be sent again; the next request is the call's next
        attempt, filed under the same call (BACKLOG RI-21, RI-139)."""
        call.keep(self.meter.fail(
            self.model, error, kind=kind, attempt=call.attempt,
            retried=True, waited=waited, adapted=[lesson] if lesson else (),
            call_no=call.number, **self._timing(call)))
        call.attempt += 1

    @staticmethod
    def _timing(call: "_Request") -> Dict[str, Any]:
        """A failed request's size, start and duration for its cost row
        (BACKLOG RI-126); the pacer's hold rides on the call's first row."""
        return {"estimated_in": call.size(), "started": call.sent,
                "duration": call.took(), "paced": call.take_paced()}

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
    RATE_LIMIT_WAITS: Tuple[float, ...] = RATE_LIMIT_LADDER

    @staticmethod
    def is_rate_limit(exc: BaseException) -> bool:
        """Whether an exception is the provider saying 'too many requests'
        (:func:`is_rate_limit`: its name, a 429, or the words in its message
        or its body, whatever the status; BACKLOG RI-139)."""
        return is_rate_limit(exc)

    @staticmethod
    def _retry_after(exc: BaseException) -> Optional[float]:
        """The provider's own wait, when its response carried one."""
        return _retry_after_of(exc)

    def _create_adaptive(self, client: Any, kwargs: Dict[str, Any],
                         call: _Request) -> Any:
        """One call on the raw client, retried with each refused parameter
        fixed in turn -- the backend reports one refusal at a time -- and
        every fix remembered for every later call. A rate-limited call
        waits (:attr:`RATE_LIMIT_WAITS`, or the provider's retry-after)
        and tries again; the waits are metered as wall clock like any
        other second the call took. With a :attr:`pacer`, a wait the
        provider did not name is never shorter than the pacer's window
        (:meth:`RequestPacer.backoff`), and the model's next request is
        held as long (BACKLOG RI-127); a retry-after of 0 takes the
        ladder's step, without the window's minimum (BACKLOG RI-139).
        ``call`` is this call's own record, which every request of it is
        timed and filed by."""
        create = client.chat.completions.create
        request = dict(kwargs)
        waits = list(self.RATE_LIMIT_WAITS)
        for _ in range(3 + len(waits)):
            try:
                call.send()
                return create(**request)
            except Exception as exc:                # noqa: BLE001 - inspected
                if self.is_rate_limit(exc) and waits:
                    named = self._retry_after(exc)
                    # A named wait of 0 takes the ladder's step, paced or
                    # not (BACKLOG RI-139): a provider stuck on 0 gets the
                    # ladder's waits and tries, not its retries back to
                    # back. With a pacer, a wait named by nobody is never
                    # under the window; a 0 is not forced up to it.
                    wait = named or waits.pop(0)
                    if self.pacer is not None:
                        wait = self.pacer.backoff(named, wait)
                        self.pacer.hold(self.model, wait)
                    if not named:
                        call.ladder_wait = wait
                    self.adaptations.append(
                        f"{self.model}: rate limited; waited {wait:.0f} s")
                    logging.getLogger(__name__).info(
                        "prompter engine rate limited on %s; waiting %.0f s",
                        self.model, wait)
                    self._sent_again(call, exc, "rate_limit", waited=wait)
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
                self._sent_again(call, exc, "parameter", lesson=lesson)
                request = adjusted
        call.send()
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
                    _no_pdf_in(body)
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
        # This call's own record, handed down its path: calls on several
        # threads at once share this engine (BACKLOG RI-139).
        call = _Request(self, self.model, lambda: estimate_request(
            msgs, system=system, tools=tools, output_format=output_format))
        cap = int(max_tokens or self.max_tokens)
        raw = None
        tried_chat = False
        #: How the call ended, where this engine knows better than the
        #: error's text: nothing back is ``no_reply`` with or without a client.
        ended_kind: Optional[str] = None
        try:
            if self.prefer_chat and simple and hasattr(self._prompter, "chat"):
                # Keeps the SDK's budget guard and logging on every call
                # that does not need a tool loop -- triage, above all.
                last = openai_messages[-1]
                tried_chat = True
                call.send()
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
                    self._sent_again(call, reason if reason is not None
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
                raw = self._create_adaptive(client, kwargs, call)
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
            # as it always was. Nothing went out (a backend with no client
            # for a tool loop): no request is recorded, only a pacer's hold
            # before it (BACKLOG RI-139).
            if call.out:
                self.meter.fail(self.model, exc, kind=ended_kind,
                                attempt=call.attempt, call_no=call.number,
                                seconds=self._clock() - call.began,
                                **self._timing(call))
                call.came_back(exc)
            else:
                self.meter.unsent(self.model, exc, paced=call.take_paced(),
                                  started=call.sent)
            raise
        seconds = self._clock() - call.began
        took = call.took()
        call.came_back()
        return self._reply(raw, output_format, seconds,
                           timing={"started": call.sent, "duration": took,
                                   "paced": call.take_paced()},
                           call=call)

    def _reply(self, raw: Any, output_format: Any, seconds: float,
               timing: Optional[Dict[str, Any]] = None,
               call: Optional[_Request] = None) -> Reply:
        if raw is None:
            raise RuntimeError(
                "Prompter returned nothing; the call failed or was refused "
                "(the SDK stashes the reason on the prompter object)")
        # Read into a local: another thread's call may set the engine's
        # attribute before this one is metered (BACKLOG RI-139).
        served = getattr(raw, "model", None) or self.model
        self.served_by = served
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
        self.meter.add(self.model, usage, seconds, served_by=served,
                       attempt=call.attempt if call is not None else 1,
                       call_no=call.number if call is not None else None,
                       stop=str(getattr(choice, "finish_reason", "") or ""),
                       **(timing or {}))

        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "") if message else ""
        blocks: List[Dict[str, Any]] = []
        if text:
            blocks.append(text_block(text))
        calls: List[ToolCall] = []
        for asked in (getattr(message, "tool_calls", None) or []):
            function = getattr(asked, "function", None)
            try:
                arguments = json.loads(getattr(function, "arguments", "") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(asked.id, function.name, arguments))
            blocks.append(tool_use_block(asked.id, function.name, arguments))

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
            model=served,
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

#: How much of an answer's raw arguments a call's cost row keeps
#: (``answer_args``) when the engine unwrapped it, dropped keys from it, set
#: aside an object in it that named its fields, or did not take it. Only run
#: files keep the rows.
RAW_ANSWER_CHARS = 20_000

#: The fields a LangChain chat model keeps its per-reply output cap in.
_CAP_FIELDS = ("max_tokens", "max_output_tokens", "max_completion_tokens")

#: Lets a PDF block go out to Claude through the Foundry Anthropic proxy
#: (BACKLOG RI-87). Off by default: no document block had been sent through
#: the proxy when this was written. Set it only once
#: :func:`report_ingest.model_first.probe_pdf_route` has read page 2 there.
#: Read when an engine is built.
FOUNDRY_CLAUDE_PDF_ENV = "GEOTECH_FOUNDRY_CLAUDE_PDF"

#: The ``_llm_type`` of the Foundry proxy's ``ChatAnthropic``
#: (``webapp.engine_config.foundry_chat_anthropic_class``).
FOUNDRY_PROXY_LLM_TYPE = "anthropic-chat-foundry-proxy"

#: Where ``ChatAnthropic`` reaches Anthropic's own API.
ANTHROPIC_API_HOST = "api.anthropic.com"


def takes_pdf(chat_model: Any) -> bool:
    """Whether a :func:`pdf_block` can go out on this chat model's route.

    * ``ChatAnthropic`` on Anthropic's own API: yes. langchain_anthropic
      sends LangChain's ``file`` block as a ``document`` block, which is
      how the plain baseline sent its PDFs (RI-86).
    * The Foundry proxy's ``ChatAnthropic``: only with
      :data:`FOUNDRY_CLAUDE_PDF_ENV` on.
    * Anything else: no. ``ChatAnthropic`` at any other address is a proxy
      nobody has tried, and so is one with no address of its own (its
      client then takes ``ANTHROPIC_BASE_URL``); the ``palantir_models``
      classes carry text and pictures only; the Prompter and OpenAI
      classes are untried.

    Read off the class and its ``_llm_type``, so nothing is imported.
    """
    if not any(c.__name__ == "ChatAnthropic"
               and c.__module__.startswith("langchain_anthropic")
               for c in type(chat_model).__mro__):
        return False
    try:
        kind = chat_model._llm_type
    except Exception:                                    # noqa: BLE001
        return False
    if kind == FOUNDRY_PROXY_LLM_TYPE:
        return (os.environ.get(FOUNDRY_CLAUDE_PDF_ENV) or "").strip() \
            .lower() in ("1", "on", "true", "yes")
    if kind != "anthropic-chat":
        return False
    from urllib.parse import urlparse

    url = getattr(chat_model, "anthropic_api_url", None)
    return bool(url) and urlparse(str(url)).hostname == ANTHROPIC_API_HOST


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
    * a PDF (:func:`pdf_block`, in a user turn) goes in as LangChain's
      ``file`` block, which langchain_anthropic sends as a ``document``
      block, and only when :attr:`accepts_pdf` is on (:func:`takes_pdf`
      unless the caller says). Otherwise the call raises ``ValueError``
      before any request: a PDF that vanished on the way would leave the
      model answering from the words alone. A PDF in a tool result is
      refused the same way, here and on :class:`PrompterEngine`. A call
      with no PDF goes out exactly as it did before PDFs were added;
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
    :attr:`Reply.notes`, and the answer is validated again; if what is left
    once they are dropped is empty or has every field at its default, there
    is no answer. An answer sent as the only key, one the schema does not
    have, is unwrapped and the wrapper named (:func:`_validate`). An unknown
    key BESIDE fields of the answer is a stray, however much it looks like
    the answer: dropped, never merged in, and named when it held an object
    naming fields of the answer. When the engine changed an answer, did not
    take it, or set such an object aside, the call's cost row keeps the raw
    arguments (``answer_args``).

    ONE CALL, ONE MODEL REQUEST. No retry happens in here, so the ingest's
    :class:`~report_ingest.subagent.CallCeiling`, the pass's own count and
    :attr:`meter` all see the same number (the meter's replies and failed
    requests together: a request that raised is recorded as failed before
    it is raised). A call refused before anything went out (a PDF on a
    route that takes none) is no request, and the meter does not count it.
    Usage is read off the reply's ``usage_metadata``, the cached input
    apart from the rest and the reasoning tokens counted.

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
                 name: Optional[str] = None,
                 accepts_pdf: Optional[bool] = None,
                 pacer: Optional[RequestPacer] = None) -> None:
        self.chat_model = chat_model
        self.name = str(name or _chat_model_name(chat_model)
                        or type(chat_model).__name__)
        self.model = self.name
        self.meter = meter if meter is not None else CostMeter()
        #: Paces requests to a per-minute token limit (BACKLOG RI-127);
        #: None, the default, sends each request as it comes. This engine
        #: still sends each call once: a rate limit that comes out of the
        #: model holds the NEXT request instead (:meth:`RequestPacer.hold`).
        self.pacer: Optional[RequestPacer] = pacer
        #: The clock a call is timed by; a test replaces it.
        self._clock = time.time
        #: Whether a :func:`pdf_block` may go out on this model's route:
        #: :func:`takes_pdf` unless the caller says.
        self.accepts_pdf: bool = (takes_pdf(chat_model) if accepts_pdf is None
                                  else bool(accepts_pdf))
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

    def _model_for(self, cap: Optional[int],
                   lessons: Optional[List[str]] = None) -> Any:
        """The model with its cap held to ``cap``: a copy made once per cap
        that shares the original's client; the model itself for None or for
        a model that keeps no cap a copy can change. A lesson learned here
        goes on :attr:`adaptations` and on ``lessons``, the call's own list
        for its cost row: calls on several threads share the engine, so a
        slice of :attr:`adaptations` could hold another call's (BACKLOG
        RI-139)."""
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
                        lesson = (f"{self.name}: a cap of {cap} could not be "
                                  f"set on a copy of the model "
                                  f"({_first_line(exc)}); the model's own "
                                  f"cap applies")
                        self.adaptations.append(lesson)
                        if lessons is not None:
                            lessons.append(lesson)
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
            elif kind == "pdf":
                # Whether the route takes one is decided in complete().
                out.append({"type": "file", "mime_type": "application/pdf",
                            "base64": base64.b64encode(
                                block["data"]).decode("ascii")})
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
                    _no_pdf_in(body)
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
        if not self.accepts_pdf and any(
                isinstance(b, dict) and b.get("type") == "pdf"
                for m in msgs if not isinstance(m["content"], str)
                for b in m["content"]):
            raise ValueError(
                f"{self.name} ({type(self.chat_model).__name__}) is not known "
                f"to take a PDF, so none is sent (accepts_pdf is off). On the "
                f"Foundry Anthropic proxy, {FOUNDRY_CLAUDE_PDF_ENV}=1 turns "
                f"it on once report_ingest.model_first.probe_pdf_route has "
                f"read page 2 there; elsewhere send the pages as pictures")

        offered = [_function_spec(t) for t in (tools or [])]
        if output_format is not None:
            if any(t["function"]["name"] == ANSWER_TOOL for t in offered):
                raise ValueError(
                    f"a pass tool may not be named {ANSWER_TOOL!r}: that is "
                    f"how this engine returns a structured answer")
            offered.append(_answer_spec(output_format))
            note = _ANSWER_NOTE_WITH_TOOLS if tools else _ANSWER_NOTE
            system = f"{system}\n\n{note}" if system else note

        # This call's own lessons, for its row (BACKLOG RI-139).
        learned: List[str] = []
        model = self._model_for(self._cap_for(max_tokens), learned)
        runnable = model.bind_tools(offered) if offered else model
        request = self._to_langchain(msgs, system)
        call = _Request(self, self.name, lambda: estimate_request(
            msgs, system=system, tools=offered))
        try:
            message = runnable.invoke(request)
        except Exception as exc:
            # Recorded, then raised as before: a failed call is a call
            # (BACKLOG RI-21). The model's own retries happen inside it.
            # A refusal that says nothing went out (``nothing_sent``: the
            # palantir_models Claude class refusing a PDF) is no request;
            # a pacer's hold before it is recorded all the same (BACKLOG
            # RI-139).
            if not getattr(exc, "nothing_sent", False):
                self.meter.fail(self.name, exc,
                                seconds=self._clock() - call.began,
                                adapted=list(learned),
                                estimated_in=call.size(), started=call.sent,
                                duration=call.took(), paced=call.paced)
                call.came_back(exc)
            else:
                self.meter.unsent(self.name, exc, paced=call.paced,
                                  started=call.sent)
            raise
        seconds = self._clock() - call.began
        took = call.took()
        call.came_back()
        return self._reply(message, output_format, seconds,
                           adapted=list(learned),
                           timing={"started": call.sent, "duration": took,
                                   "paced": call.paced})

    def _reply(self, message: Any, output_format: Any,
               seconds: float, adapted: Sequence[str] = (),
               timing: Optional[Dict[str, Any]] = None) -> Reply:
        meta = getattr(message, "response_metadata", None) or {}
        # A local first: calls on several threads share the engine
        # (BACKLOG RI-139).
        served = str(meta.get("model_name") or meta.get("model")
                     or "") or self.name
        self.served_by = served
        usage = _usage_of(message)
        incomplete = meta.get("incomplete_details")
        stop = _reason(meta.get("finish_reason") or meta.get("stop_reason")
                       or (incomplete.get("reason")
                           if isinstance(incomplete, dict) else "")
                       or meta.get("status"))
        entry = self.meter.add(self.name, usage, seconds,
                               served_by=served, stop=stop,
                               adapted=adapted, **(timing or {}))

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
        #: the raw arguments of each answer the engine changed or refused
        raw: List[str] = []
        #: answer call id -> "" (a whole answer) or why it was not taken
        verdicts: Dict[str, str] = {}
        if output_format is not None:
            offered = ([(c["id"], c.get("args")) for c in answers]
                       + [(None, args) for args in spoiled])
            for call_id, args in offered:
                strays: List[Tuple[str, List[str]]] = []
                if cut:
                    answer, dropped, why, wrappers = None, [], _CUT_ANSWER, []
                else:
                    answer, dropped, why, wrappers = _validate(output_format,
                                                               args)
                    strays = _stray_objects(output_format, args)
                notes.extend(_answer_notes(answer, dropped, why, wrappers,
                                           strays))
                if wrappers or dropped or strays or answer is None:
                    raw.append(_raw_args(args))
                if answer is not None and parsed is None:
                    parsed = answer
                if call_id:
                    verdicts[call_id] = "" if answer is not None else why
            if parsed is None and text and not cut:
                found = _json_in(text)
                answer, dropped, why, wrappers = _validate(output_format,
                                                           found)
                # Text that is no answer at all is a reply, not a refusal;
                # an answer written out as text is noted like a call's. JSON
                # in the words of a turn that asked to look ({"page": 4}) is
                # an answer only if it was taken or names a field of one.
                named = _names_a_field(output_format, found)
                tried = answer is not None or dropped or wrappers or named
                if tried and (answer is not None or not wanted or named):
                    strays = _stray_objects(output_format, found)
                    notes.extend(_answer_notes(answer, dropped, why,
                                               wrappers, strays))
                    if wrappers or dropped or strays or answer is None:
                        raw.append(_raw_args(found))
                    parsed = answer
        if notes:
            logging.getLogger(__name__).warning("%s: %s", self.name,
                                                "; ".join(notes))
            # What the engine did with the answer, on the call's cost row.
            entry["notes"] = list(notes)
        if raw:
            # The arguments as they came, so the next odd answer can be read
            # off the run file rather than guessed at.
            entry["answer_args"] = raw

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
            model=served,
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
              ) -> Tuple[Any, List[str], str, List[str]]:
    """``args`` as an ``output_format``: ``(answer, keys dropped, why not,
    wrappers taken off)``, the answer None when there is none.

    * Empty arguments are no answer. Foundry's classes hand back ``{}`` for
      a call cut mid-arguments, and many reading schemas accept ``{}``.
    * An answer may come wrapped: ONE key the schema does not have, holding
      the answer as an object or as JSON text of one (:func:`_unwrapped`).
      On the local Claude narrative run (2026-10-24) 9 of 20 chunk answers
      came as ``{"arguments": {...}}``. The key is taken off and the value
      validated instead. A key the schema has is never taken off. BESIDE
      other keys an unknown key is a stray, whatever it holds: models add
      maps of their own (``"evidence": {"has_logs": "pages 5-20"}``,
      ``"units": {"total_depth": "ft"}``) that name the answer's fields with
      other values, so it is never merged in and never refuses the answer
      (:func:`_stray_objects` names it).
    * Strict mode's schema makes every field required, so a model may send
      null for a field that has a default: tried again with the nulls left
      out, letting the defaults stand.
    * Foundry sends tools without strict mode, so a model may add a key the
      schema forbids: the unknown keys are dropped and named, and the
      answer validated again. Any other error stands.
    * What is left once unknown keys are dropped -- or passed over, on a
      schema that ignores them -- must still say something. Left empty, or
      with every field at its default, it is no answer: the keys dropped
      may have held all of it, and many reading schemas accept ``{}``.
      ON PURPOSE this also refuses an answer that truly has nothing to say
      when it carries a stray key: ``{"comments": [], "summary": "none"}``
      for the reconciler's comments, or ``{"title": null, "reasoning":
      "..."}`` for a bound document's identity. The schemas that build a
      blank answer from ``{}`` (the narrative reading and its summary, the
      bound identity, the conflict comments, the scale labels) cannot tell
      that from an answer lost to its wrapper; the model is told why, and
      the pass's own re-ask or warning follows.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return None, [], "its arguments are not JSON", []
    if not isinstance(args, dict):
        return None, [], "it is not a JSON object", []
    if not args:
        return None, [], "it arrived empty", []
    args, wrappers = _unwrapped(output_format, args)
    why = ""
    for candidate in (args, _without_nulls(args)):
        dropped: List[str] = []
        for _ in range(4):
            try:
                answer = output_format.model_validate(candidate)
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
                if not candidate:
                    return None, dropped, _emptied(dropped), wrappers
                continue
            passed = _passed_over(output_format, candidate)
            if (dropped or passed) and _says_nothing(output_format,
                                                     candidate, answer):
                return None, dropped, _emptied(dropped, passed), wrappers
            return answer, dropped, "", wrappers
    return None, [], why, wrappers


def _emptied(dropped: Sequence[str], passed: Sequence[str] = ()) -> str:
    """Why an answer that lost everything to its unknown keys is not
    taken: the keys dropped (a schema that forbids them) and the keys passed
    over (one that ignores them)."""
    said = []
    if dropped:
        said.append(f"{', '.join(dropped)} were dropped")
    if passed:
        said.append(f"{', '.join(passed)} were passed over")
    return "nothing was left once the unknown keys " + " and ".join(said)


def _answer_notes(answer: Any, dropped: Sequence[str], why: str,
                  wrappers: Sequence[str],
                  strays: Sequence[Tuple[str, Sequence[str]]] = ()
                  ) -> List[str]:
    """What the engine did with one answer, for :attr:`Reply.notes`: one
    note for its wrappers however deep, one for each stray key that held an
    object naming fields of the answer, one for the keys dropped, one if it
    was not taken."""
    notes = []
    if wrappers:
        notes.append("the answer arrived wrapped in "
                     + ", then ".join(repr(w) for w in wrappers)
                     + ("; unwrapped" if answer is not None
                        else "; unwrapped, and still not taken"))
    for key, named in strays:
        shown = ", ".join(named[:5]) + (", ..." if len(named) > 5 else "")
        notes.append(f"the unknown key {key!r} held an object naming fields "
                     f"of the answer ({shown}); set aside, not merged")
    if dropped:
        notes.append("unknown keys dropped from the answer: "
                     + ", ".join(dropped))
    if answer is None:
        notes.append(f"an answer was not taken: {why}")
    return notes


def _raw_args(args: Any, limit: int = RAW_ANSWER_CHARS) -> str:
    """An answer's arguments as they came, as JSON text cut to ``limit``
    characters."""
    if isinstance(args, str):
        body = args
    else:
        try:
            body = json.dumps(args, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            body = repr(args)
    if len(body) > limit:
        body = (body[:limit]
                + f" ... [cut at {limit:,} of {len(body):,} characters]")
    return body


def _field_names(output_format: Any) -> Optional[set]:
    """The keys a schema takes at its top: its fields' names and string
    aliases; None for a schema that does not say."""
    fields = getattr(output_format, "model_fields", None)
    if not isinstance(fields, dict):
        return None
    names = set()
    for name, info in fields.items():
        names.add(name)
        for alias in (getattr(info, "alias", None),
                      getattr(info, "validation_alias", None)):
            if isinstance(alias, str):
                names.add(alias)
    return names


def _object_in(value: Any) -> Optional[Dict[str, Any]]:
    """A non-empty object, or JSON text of one, as a dict; else None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value if isinstance(value, dict) and value else None


def _unwrapped(output_format: Any, args: Dict[str, Any]
               ) -> Tuple[Dict[str, Any], List[str]]:
    """``args`` with any wrapper taken off, and the wrappers' names.

    A wrapper is the ONLY key, one the schema does not have, holding a
    non-empty object or JSON text of one. Taken off at most three deep; a
    key the schema has is never taken off, whatever it holds. A key beside
    others is never a wrapper (:func:`_stray_objects`).
    """
    names = _field_names(output_format)
    wrappers: List[str] = []
    if names is None:
        return args, wrappers
    for _ in range(3):
        if len(args) != 1:
            break
        ((key, value),) = args.items()
        inner = None if key in names else _object_in(value)
        if inner is None:
            break
        wrappers.append(str(key))
        args = inner
    return args, wrappers


def _stray_objects(output_format: Any, args: Any
                   ) -> List[Tuple[str, List[str]]]:
    """The unknown keys beside the answer's fields (any wrapper taken off)
    that held an object naming fields of the answer, each with the fields
    it named.

    They are strays all the same -- dropped or passed over, never merged
    in, never a reason to refuse the answer: a map of the model's own
    (evidence, units, sources) names the answer's fields with other values.
    Named, so a run's check can show an answer that may have been hidden
    beside an id.
    """
    names = _field_names(output_format)
    value = _object_in(args)
    if not names or value is None:
        return []
    value = _unwrapped(output_format, value)[0]
    out: List[Tuple[str, List[str]]] = []
    for key, held in value.items():
        inner = None if key in names else _object_in(held)
        named = sorted(names.intersection(inner)) if inner else []
        if named:
            out.append((str(key), named))
    return out


def _names_a_field(output_format: Any, value: Any) -> bool:
    """Whether a JSON object names a field of the schema, at its top or
    inside a wrapper."""
    names = _field_names(output_format)
    if not names or not isinstance(value, dict):
        return False
    inner = _unwrapped(output_format, value)[0]
    return bool(names.intersection(value) or names.intersection(inner))


def _passed_over(output_format: Any, candidate: Dict[str, Any]) -> List[str]:
    """The top-level keys a schema that ignores unknown keys passed over."""
    names = _field_names(output_format)
    if names is None:
        return []
    return [str(k) for k in candidate if k not in names]


def _says_nothing(output_format: Any, candidate: Dict[str, Any],
                  answer: Any) -> bool:
    """Whether an answer that validated holds nothing: empty, or every field
    at its default. A schema with a required field always says something."""
    if not candidate:
        return True
    try:
        blank = output_format()
        return answer.model_dump() == blank.model_dump()
    except Exception:                                    # noqa: BLE001
        return False


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
