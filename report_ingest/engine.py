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
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "Engine", "Reply", "ToolCall", "Usage", "CostMeter", "ClaudeEngine",
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

def strict_schema(model: Any) -> Dict[str, Any]:
    """A pydantic model's JSON schema, in the shape strict mode demands.

    OpenAI's ``json_schema`` response format with ``strict: true`` requires
    every object to set ``additionalProperties: false`` and to list EVERY
    property in ``required``. Pydantic emits neither for a field with a
    default, so the schema is walked and both are imposed. ``$defs`` and
    ``$ref`` are left alone -- strict mode understands them -- and so is
    ``anyOf``, which is how an optional field's null arm arrives.
    """
    import copy

    schema = copy.deepcopy(model.model_json_schema())

    #: Keys whose value is a MAP of names to schemas; each value is walked.
    maps = ("properties", "$defs", "definitions", "patternProperties")
    #: Keys whose value is itself a schema, or a list of them.
    schemas = ("items", "anyOf", "allOf", "oneOf", "prefixItems", "not",
               "additionalItems", "contains")

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
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
        if self.prefer_chat and simple and hasattr(self._prompter, "chat"):
            # Keeps the SDK's budget guard and logging on every call that
            # does not need a tool loop -- triage, above all.
            last = openai_messages[-1]
            raw = self._prompter.chat(
                user=last["content"], system=system or
                "You are a helpful assistant.",
                model=self.model, temperature=0, return_raw=True,
                max_tokens=int(max_tokens or self.max_tokens),
                **({"response_format": response_format}
                   if response_format else {}))
        else:
            client = self.client
            if client is None:
                raise RuntimeError(
                    "this Prompter backend exposes no OpenAI client, so it "
                    "cannot run a tool loop; the label review needs one")
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": 0,
                "max_tokens": int(max_tokens or self.max_tokens),
            }
            if tools:
                kwargs["tools"] = [{"type": "function", "function": {
                    "name": t["name"], "description": t["description"],
                    "parameters": t["input_schema"]}} for t in tools]
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format
            raw = client.chat.completions.create(**kwargs)
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
