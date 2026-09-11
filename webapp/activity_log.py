"""Always-on activity log — every tool call, tool result and model call of a
turn, across the primary agent AND its sub-agents, written to
``<conversation>/activity.jsonl``.

Owner feedback 2026-09-11: "Ensure tool calls for all agents and
agent-to-agent convos are tracked in the SharePoint archive of the
conversations, regardless of whether the 'show turn details' box is ticked."

What the archive held before this: the user's text, the assistant's final
answer and artifact paths (``transcript.jsonl``), plus — only when the
"Show turn details" box was ticked — an 80-character one-liner per PRIMARY
tool call in ``trace.jsonl``. Sub-agent exchanges (the ``calc`` and
``references`` agents) appeared nowhere: ``core.stream_turn`` streams the
graph without ``subgraphs=True``, so their internals never enter the stream
at all, only the primary's ``task(...)`` call and its compact reply.

This module does not touch the stream. It is a LangChain
:class:`BaseCallbackHandler` attached to the run config; callbacks propagate
into sub-agent invocations (that is already how ``turn_tokens`` sums their
usage), so one handler sees everything. It is ALWAYS on — independent of the
trace toggle — and the SharePoint mirror carries the file up with the rest
of the conversation directory.

Record shape (one JSON object per line)::

    {"turn": 3, "ts": 1757600000.1, "t": 12.34, "agent": "calc",
     "event": "tool_end", "name": "call_agent", "run_id": "...",
     "parent_run_id": "...", "result": "...", "truncated": false,
     "duration_s": 0.41}

``event`` is one of ``turn_start``, ``tool_start``, ``tool_end``,
``tool_error``, ``model_start``, ``model_end``, ``model_error``,
``turn_end``. ``agent`` is ``primary`` or the sub-agent's name, attributed
by nesting: a ``task`` tool call pushes its ``subagent_type`` on a stack
until that call ends, and everything nested inside belongs to it. No
deepagents internals are read, so this stays correct across versions.

Tool arguments are logged in full (JSON), tool results and errors in full up
to :data:`DEFAULT_MAX_CHARS` (then cut, with ``truncated: true`` and the full
length recorded). Model prompts are NOT logged — every call carries the
~17 KB system prompt and ``messages.json`` already holds the conversation —
only per-call usage and the number of tool calls the model requested.

A logging failure must never fail a turn: every handler swallows its own
exceptions and ``raise_error`` is left False.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:                                      # pragma: no cover
    class BaseCallbackHandler:                         # type: ignore[no-redef]
        """Fallback so the module imports without langchain (tests, tools)."""
        raise_error = False

FILE_NAME = "activity.jsonl"

#: Cap on a logged tool result / error text. Generous on purpose: the point
#: is a faithful archive, and a calc dump is a few KB. The cut is marked.
DEFAULT_MAX_CHARS = 32_000

#: Name of the deepagents delegation tool whose nesting defines sub-agents.
TASK_TOOL = "task"

PRIMARY = "primary"


def activity_path(conv_dir: str) -> str:
    return os.path.join(conv_dir, FILE_NAME)


def _cap(text: Any, max_chars: int) -> tuple:
    """``(text, truncated, full_len)`` with ``text`` cut at ``max_chars``."""
    s = text if isinstance(text, str) else _to_text(text)
    n = len(s)
    if n > max_chars:
        return s[:max_chars] + f"\n…[truncated: {n} chars total]", True, n
    return s, False, n


def _to_text(obj: Any) -> str:
    """Best-effort readable text for a tool output / message content."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    content = getattr(obj, "content", None)      # ToolMessage / Command-ish
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
        if parts:
            return "\n".join(parts)
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:                                  # noqa: BLE001
        return str(obj)


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:                                  # noqa: BLE001
        return str(obj)


def _usage_from_response(response: Any) -> Optional[dict]:
    """Pull ``{input_tokens, output_tokens, total_tokens}`` off an LLMResult
    (``usage_metadata`` on the generated message, else ``llm_output``)."""
    try:
        gens = getattr(response, "generations", None) or []
        for row in gens:
            for g in row or []:
                msg = getattr(g, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if um:
                    return {k: um.get(k) for k in
                            ("input_tokens", "output_tokens", "total_tokens")
                            if k in um}
        lo = getattr(response, "llm_output", None) or {}
        tu = lo.get("token_usage") or lo.get("usage")
        if isinstance(tu, dict):
            return {"input_tokens": tu.get("prompt_tokens",
                                           tu.get("input_tokens")),
                    "output_tokens": tu.get("completion_tokens",
                                            tu.get("output_tokens")),
                    "total_tokens": tu.get("total_tokens")}
    except Exception:                                  # noqa: BLE001
        pass
    return None


def _n_tool_calls(response: Any) -> int:
    try:
        n = 0
        for row in getattr(response, "generations", None) or []:
            for g in row or []:
                msg = getattr(g, "message", None)
                n += len(getattr(msg, "tool_calls", None) or [])
        return n
    except Exception:                                  # noqa: BLE001
        return 0


class ActivityLogger(BaseCallbackHandler):
    """Append every tool/model event of one turn to ``activity.jsonl``.

    Construct one per turn (``turn`` = the 1-based user-turn index), pass it
    in ``config["callbacks"]``, and call :meth:`turn_start` /
    :meth:`turn_end` around the run so the file is self-describing.
    """

    raise_error = False
    run_inline = True

    def __init__(self, conv_dir: str, turn: int = 0, *,
                 max_chars: int = DEFAULT_MAX_CHARS,
                 clock=time.time):
        super().__init__()
        self.conv_dir = conv_dir
        self.turn = int(turn)
        self.max_chars = int(max_chars)
        self._clock = clock
        self._t0 = clock()
        self._stack: list = []            # [(run_id, subagent_name), ...]
        self._names: dict = {}            # run_id -> tool name
        self._starts: dict = {}           # run_id -> start time
        self.records_written = 0
        self.last_error: Optional[str] = None

    # -- attribution ------------------------------------------------------
    @property
    def agent(self) -> str:
        return self._stack[-1][1] if self._stack else PRIMARY

    # -- writer -----------------------------------------------------------
    def _write(self, rec: dict) -> None:
        try:
            now = self._clock()
            rec = {"turn": self.turn, "ts": round(now, 3),
                   "t": round(now - self._t0, 3), **rec}
            os.makedirs(self.conv_dir, exist_ok=True)
            with open(activity_path(self.conv_dir), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str)
                         + "\n")
            self.records_written += 1
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    # -- turn envelope ----------------------------------------------------
    def turn_start(self, prompt: Optional[str] = None,
                   model: Optional[str] = None) -> None:
        self._write({"agent": PRIMARY, "event": "turn_start",
                     "prompt": (prompt or "")[:500], "model": model})

    def turn_end(self, *, turn_tokens: int = 0, error: Optional[str] = None,
                 answer_chars: int = 0) -> None:
        self._write({"agent": PRIMARY, "event": "turn_end",
                     "duration_s": round(self._clock() - self._t0, 3),
                     "turn_tokens": turn_tokens, "answer_chars": answer_chars,
                     "error": error})

    # -- tools ------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, metadata=None, inputs=None, **kwargs):
        try:
            name = (kwargs.get("name") or (serialized or {}).get("name")
                    or "tool")
            rid = str(run_id)
            self._names[rid] = name
            self._starts[rid] = self._clock()
            args = inputs if inputs is not None else input_str
            rec = {"agent": self.agent, "event": "tool_start", "name": name,
                   "run_id": rid,
                   "parent_run_id": str(parent_run_id) if parent_run_id else None,
                   "args": _json_safe(args)}
            if name == TASK_TOOL:
                sub = None
                if isinstance(inputs, dict):
                    sub = inputs.get("subagent_type")
                sub = str(sub or "subagent")
                rec["subagent"] = sub
                self._write(rec)
                self._stack.append((rid, sub))
                return
            self._write(rec)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _tool_finish(self, event: str, run_id, payload_key: str, payload,
                     parent_run_id=None) -> None:
        rid = str(run_id)
        name = self._names.pop(rid, "tool")
        t_start = self._starts.pop(rid, None)
        # Attribute the RESULT to the agent that made the call: a task's
        # result belongs to the primary, so pop before attributing.
        if self._stack and self._stack[-1][0] == rid:
            sub = self._stack.pop()[1]
            agent = self.agent
        else:
            sub = None
            agent = self.agent
        text, truncated, full_len = _cap(payload, self.max_chars)
        rec = {"agent": agent, "event": event, "name": name, "run_id": rid,
               "parent_run_id": str(parent_run_id) if parent_run_id else None,
               payload_key: text, "truncated": truncated,
               "chars": full_len,
               "duration_s": (round(self._clock() - t_start, 3)
                              if t_start is not None else None)}
        if sub:
            rec["subagent"] = sub
        self._write(rec)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        try:
            self._tool_finish("tool_end", run_id, "result", output,
                              parent_run_id)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        try:
            self._tool_finish("tool_error", run_id, "error",
                              f"{type(error).__name__}: {error}", parent_run_id)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    # -- model calls ------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id,
                            parent_run_id=None, tags=None, metadata=None,
                            **kwargs):
        try:
            rid = str(run_id)
            self._starts[rid] = self._clock()
            n_msgs = sum(len(m) for m in (messages or []))
            model = None
            try:
                model = ((serialized or {}).get("kwargs") or {}).get("model") \
                    or ((serialized or {}).get("kwargs") or {}).get("model_name") \
                    or (serialized or {}).get("name")
            except Exception:                          # noqa: BLE001
                model = None
            self._write({"agent": self.agent, "event": "model_start",
                         "run_id": rid,
                         "parent_run_id": (str(parent_run_id)
                                           if parent_run_id else None),
                         "model": model, "n_messages": n_msgs})
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        try:
            rid = str(run_id)
            t_start = self._starts.pop(rid, None)
            self._write({"agent": self.agent, "event": "model_end",
                         "run_id": rid,
                         "parent_run_id": (str(parent_run_id)
                                           if parent_run_id else None),
                         "usage": _usage_from_response(response),
                         "n_tool_calls": _n_tool_calls(response),
                         "duration_s": (round(self._clock() - t_start, 3)
                                        if t_start is not None else None)})
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        try:
            rid = str(run_id)
            t_start = self._starts.pop(rid, None)
            text, truncated, _n = _cap(f"{type(error).__name__}: {error}",
                                       self.max_chars)
            self._write({"agent": self.agent, "event": "model_error",
                         "run_id": rid,
                         "parent_run_id": (str(parent_run_id)
                                           if parent_run_id else None),
                         "error": text, "truncated": truncated,
                         "duration_s": (round(self._clock() - t_start, 3)
                                        if t_start is not None else None)})
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"


def load(conv_dir: str, turn: Optional[int] = None) -> list:
    """Records in ``activity.jsonl`` (oldest first), optionally one turn's.
    Malformed lines are skipped; missing file → ``[]``."""
    out: list = []
    try:
        with open(activity_path(conv_dir), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if turn is None or rec.get("turn") == turn:
                    out.append(rec)
    except OSError:
        return []
    return out


__all__ = ["ActivityLogger", "activity_path", "load", "FILE_NAME",
           "DEFAULT_MAX_CHARS", "TASK_TOOL", "PRIMARY"]
