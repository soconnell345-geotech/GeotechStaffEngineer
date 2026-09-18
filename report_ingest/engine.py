"""Model access for the report-ingest passes, and what a run cost.

Two passes sit between planlens' rule-based page labels and any reader:
:mod:`report_ingest.triage` and :mod:`report_ingest.label_review`. Both are
written against the small :class:`Engine` protocol below rather than against
an SDK, so the same code runs on the Claude API here and on the cluster's
Prompter later. The protocol is one method::

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
output tokens, cache reads and writes, wall clock, and dollars at the list
prices in :data:`MODEL_PRICES`. The owner's rule for this work is that
accuracy on key content beats tokens, so the meter exists to REPORT what a
run cost, not to cut a pass short.

The ``anthropic`` package is imported inside :class:`ClaudeEngine` and
nowhere else, so importing :mod:`report_ingest` costs an app nothing and
adds no dependency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "Engine", "Reply", "ToolCall", "Usage", "CostMeter", "ClaudeEngine",
    "engine_for",
    "PrompterEngine", "MODEL_PRICES", "PROMPTER_MODELS", "strict_schema",
    "text_block", "image_block", "tool_use_block", "tool_result_block",
    "opaque_block", "user", "assistant",
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
#: There is no published per-token price for a tier, so a cluster run reports
#: TOKENS, not dollars: :meth:`Usage.dollars` returns 0.0 for a model it has
#: no price for rather than inventing one. Spend is read from Funhouse's own
#: budget endpoint.
PROMPTER_MODELS: Tuple[str, ...] = (
    "funhouse-gpt-low", "funhouse-gpt-medium", "funhouse-gpt-high",
)

DEFAULT_MAX_TOKENS = 16000


# -- neutral message blocks -------------------------------------------------

def text_block(text: str) -> Dict[str, Any]:
    """Words for the model to read."""
    return {"type": "text", "text": str(text)}


def image_block(png: bytes) -> Dict[str, Any]:
    """A PNG for the model to look at.

    Bytes, not a path: the pages these passes render are held in memory and
    never written to disk, because the corpus is private.
    """
    return {"type": "image", "png": bytes(png)}


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

    def dollars(self, model: str) -> float:
        """What this call cost at list price, or 0.0 for an unpriced model."""
        price = MODEL_PRICES.get(model)
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

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class CostMeter:
    """What a run has spent so far, per model and in total."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.seconds = 0.0
        self.dollars = 0.0
        self.by_model: Dict[str, Dict[str, Any]] = {}

    def add(self, model: str, usage: Usage, seconds: float) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.seconds += float(seconds)
        cost = usage.dollars(model)
        self.dollars += cost
        row = self.by_model.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "seconds": 0.0, "dollars": 0.0})
        row["calls"] += 1
        row["input_tokens"] += usage.input_tokens
        row["output_tokens"] += usage.output_tokens
        row["cache_read_tokens"] += usage.cache_read_tokens
        row["cache_write_tokens"] += usage.cache_write_tokens
        row["seconds"] += float(seconds)
        row["dollars"] += cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "seconds": round(self.seconds, 1),
            "dollars": round(self.dollars, 4),
            "by_model": {m: {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()}
                         for m, row in self.by_model.items()},
        }

    def summary(self) -> str:
        return (f"{self.calls} calls, {self.input_tokens:,} in "
                f"(+{self.cache_read_tokens:,} cached), "
                f"{self.output_tokens:,} out, {self.seconds:.0f} s, "
                f"${self.dollars:.3f}")


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
            return {"type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
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
        if output_format is not None:
            # ``parse`` builds the output format itself and takes no
            # top-level ``cache_control``; the same instruction reaches the
            # API through the request body.
            if self.auto_cache:
                kwargs["extra_body"] = {"cache_control": {"type": "ephemeral"}}
            response = self._client.messages.parse(output_format=output_format,
                                                   **kwargs)
        else:
            if self.auto_cache:
                kwargs["cache_control"] = {"type": "ephemeral"}
            response = self._client.messages.create(**kwargs)
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
        self.meter.add(self.model, usage, seconds)
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
                return create(**request)
            except Exception as exc:                # noqa: BLE001 - inspected
                if self.is_rate_limit(exc) and waits:
                    wait = self._retry_after(exc) or waits.pop(0)
                    self.adaptations.append(
                        f"{self.model}: rate limited; waited {wait:.0f} s")
                    logging.getLogger(__name__).info(
                        "prompter engine rate limited on %s; waiting %.0f s",
                        self.model, wait)
                    self._sleep(wait)
                    continue
                adjusted = self.adjust_for_param_error(request, str(exc))
                if adjusted is None:
                    raise
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
                request = adjusted
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
                b64 = base64.b64encode(block["png"]).decode("ascii")
                out.append({"type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"}})
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
        if self.prefer_chat and simple and hasattr(self._prompter, "chat"):
            # Keeps the SDK's budget guard and logging on every call that
            # does not need a tool loop -- triage, above all.
            last = openai_messages[-1]
            tried_chat = True
            raw = self._prompter.chat(
                user=last["content"], system=system or
                "You are a helpful assistant.",
                model=self.model, temperature=0, return_raw=True,
                **{self.token_key: cap},
                **({"response_format": response_format}
                   if response_format else {}))
            if raw is None:
                # The SDK logs the provider's refusal and hands back None.
                # A parameter the deployment does not take is the usual
                # reason, and the raw client can adapt to that, so it is
                # tried before giving up; the SDK's stashed reason travels
                # either way.
                reason = getattr(self._prompter, "_last_chat_error", None)
                if self.client is None:
                    raise RuntimeError(
                        "Prompter returned nothing; the call failed or was "
                        f"refused (the SDK's reason: {reason!r})")
                self.adaptations.append(
                    f"{self.model}: chat() returned nothing "
                    f"({str(reason)[:160]!r}); retried on the raw client")
        if raw is None:
            client = self.client
            if client is None:
                raise RuntimeError(
                    "this Prompter backend exposes no OpenAI client, so it "
                    "cannot run a tool loop; the label review needs one")
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
        usage = Usage(
            input_tokens=int(getattr(usage_raw, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage_raw, "completion_tokens", 0) or 0),
            cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        )
        self.meter.add(self.model, usage, seconds)

        choice = raw.choices[0] if getattr(raw, "choices", None) else None
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


# ---------------------------------------------------------------------------
# finding the engine the app is already holding
# ---------------------------------------------------------------------------

def engine_for(obj: Any, model: Optional[str] = None,
               meter: Optional[CostMeter] = None) -> Optional["Engine"]:
    """The ingest engine for whatever the host handed the agent, or None.

    The app builds its agents around its own engines -- ``NativeToolEngine``
    and ``PrompterBridgeEngine`` on the cluster, a LangChain chat model
    locally -- and every one of them is holding the live Prompter this
    package's :class:`PrompterEngine` needs. This digs it out:

    * something that already satisfies the :class:`Engine` protocol comes
      back unchanged;
    * an app engine wrapping a Prompter (``prompter`` or ``_prompter``)
      becomes a :class:`PrompterEngine` over that Prompter;
    * a live Prompter itself becomes one directly;
    * anything else comes back **None**, which is an answer. A caller that
      cannot get an engine says so rather than falling back to a model nobody
      asked for: the app runs on Funhouse's OpenAI tiers, and quietly reading
      a 400-page report on something else would be a bill and a number that
      belong to nobody.
    """
    if obj is None:
        return None
    if isinstance(obj, (ClaudeEngine, PrompterEngine)):
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
            return PrompterEngine(inner, model or PROMPTER_MODELS[1],
                                  meter=meter)
    if _looks_like_prompter(obj):
        return PrompterEngine(obj, model or PROMPTER_MODELS[1], meter=meter)
    return None


def _looks_like_prompter(obj: Any) -> bool:
    """Whether this object is a Funhouse Prompter, by what it offers.

    By capability rather than by class: the SDK's Prompter is imported in
    several places under several names and this package imports it nowhere.
    """
    return callable(getattr(obj, "chat", None)) and hasattr(obj, "client")
