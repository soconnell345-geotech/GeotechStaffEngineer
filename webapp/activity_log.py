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
``turn_end``, ``coverage_gate`` (the note the coverage gate gave the
model, with ``GEOTECH_COVERAGE`` on), the app events of
:data:`APP_EVENTS` (``cited_pages``: the answer's page citations checked
against the pages opened; ``cited_documents``: the documents an answer or a
writer cites, checked against what was looked up; ``judgment_check``: the
answer's "Judgment calls" list checked for calls with no source, and
whether it was sent back), and -- since 2026-10-09 (live smoke
wave 1, G11) -- ``tool_delivered`` (what the model RECEIVED from a tool call,
when the agent's middleware changed the tool's own output) and
``tool_refused`` (a call no tool ran: an unknown tool name, an
interception). ``agent`` is ``primary`` or the sub-agent's name, attributed
by run ancestry: each event's parent-run chain is followed up to the ``task``
call it ran inside (whose ``subagent_type`` names the sub-agent), or to the
top of the turn (primary). Until 2026-09-15 this was a stack of open ``task``
calls, which mislabelled two sub-agents running at once -- in the 2026-09-15
SOE session every call of a parallel calc/general-purpose pair was filed under
the second, and the two task results swapped names (field feedback N15). No
deepagents internals are read, so this stays correct across versions.

Tool arguments are logged in full (JSON), tool results and errors in full up
to :data:`DEFAULT_MAX_CHARS` (then cut, with ``truncated: true`` and the full
length recorded). Model prompts are NOT logged — every call carries the
~17 KB system prompt and ``messages.json`` already holds the conversation —
only per-call usage and the number of tool calls the model requested, and
(since 2026-10-07) what the model SAID: ``model_end`` carries ``text`` (its
own words that call, up to :data:`MODEL_TEXT_CHARS`) and ``reasoning`` (any
reasoning summary the provider returned). Without them a review of a run
could see every tool call and never the model's account of why (the owner,
reviewing the first live checks of 5.32.0).

Since 2026-10-21 (BACKLOG EV-06, EV-19) the record also says:

* on ``turn_end``: ``graph_steps``, the highest LangGraph step the
  primary's graph reached (what the step cap counts; per pass in
  ``graph_steps_by_pass`` when the turn ran more than one), ``step_cap`` and
  any ``step_allowance`` (:meth:`ActivityLogger.note_step_cap`),
  ``system_prompt`` (a :func:`fingerprint` and the length of the system
  message the primary was sent) and ``tools_offered_sha`` with
  ``tools_offered`` (the tool names the primary was offered: from its model
  calls where the client shows them, ``tools_offered_from: "calls"``, else
  the build's list, ``"build"``); the names are written when the list
  changes (and once per process), the sha every turn: a turn without them
  was offered the list of the latest ``turn_end`` with the same sha;
* on the ``model_end`` (and ``model_error``) of a vision side call:
  ``vision``, its label (``kind``, ``page``, ``tile``, ``image_px``,
  ``detail``) with what was sent: ``prompt`` (whole up to
  :data:`SIDE_PROMPT_CHARS`, else that much and a ``prompt_sha``),
  ``prompt_chars``, ``detail_sent`` and ``images``;
* ``tool_delivered`` with ``stage: "graph"``: a result the agent's graph
  changed after the app's own middleware, as the model got it -- deepagents
  moving a large result to ``/large_tool_results/`` and sending a notice and
  a preview instead (``moved_to``: the path it names);
* ``turn_files`` (after ``turn_end``): each file the turn wrote or listed,
  its ``bytes``, a ``sha`` of its content (its ``mtime`` instead over
  :data:`HASH_MAX_BYTES`) and ``how`` (``new``, ``rewritten``,
  ``listed``);
* ``process_start``: written before the first record a process adds to a
  conversation's log (``process`` is a token unique to the process, with
  its ``pid`` and when it started), so an app restart between turns shows;
* ``restored`` (the conversation was brought back from the mirror or a
  zip: ``webapp.sharepoint_store``, ``webapp.conversation_restore``), held
  by :func:`note_restored` and written before the next ``turn_start`` with
  the time of the restore;
* ``page_error``, written at once by :func:`append_event`: a page rerun
  raised (``webapp.page_errors``).

Hashes and sizes, never file contents, images or keys.

A logging failure must never fail a turn: every handler swallows its own
exceptions and ``raise_error`` is left False.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

#: One lock per activity file. Tool calls run in parallel threads (a model
#: step that asks for ten page looks runs ten tools at once, each with its
#: vision call), and every event is an append to the same file. Unlocked,
#: two long lines written at once interleave and both records are lost:
#: found 2026-10-08 when a suite run's file held the tail of one record
#: glued to another, and a parallel page look had no tool_end at all.
_FILE_LOCKS: dict = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = _FILE_LOCKS[key] = threading.Lock()
        return lock

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

#: App events written as they come: ``helper_kept`` (a stopped helper's
#: work went to a file: ``path``, ``why``) and ``helper_queued`` (a helper
#: waited ``queued_s`` for a place under the cap of helpers at once;
#: ``funhouse_agent.deep.keep_work``); ``model_busy`` (an agent's model
#: call was refused as busy and is asked again after ``wait_s``: the
#: ``kind``, HTTP ``status``, ``retry_after_s``, any ``x-ratelimit-*``
#: ``headers``, ``attempt`` of ``tries``; ``funhouse_agent.deep.
#: busy_retry``); ``pacer_wait`` (a call waited ``waited_s`` for room under
#: the tokens-a-minute pacer; ``funhouse_agent.deep.pacing``);
#: ``cited_pages`` (the pages the turn's answer cites, checked against the
#: pages any tool or helper opened in the conversation: ``never_opened``,
#: ``partly_opened``... and what was done, ``action``; ``funhouse_agent.
#: cited_pages``, BACKLOG EV-18); ``cited_documents`` (the published
#: documents the turn's answer, or a file a writer was given, cites,
#: checked against what the conversation looked up: ``not_looked_up``,
#: ``where`` (``answer`` or the writer), ``action``; ``funhouse_agent.
#: cited_documents``, BACKLOG EV-21).
APP_EVENTS = frozenset({"helper_kept", "helper_queued", "model_busy",
                        "pacer_wait", "cited_pages", "cited_documents",
                        "judgment_check"})

PRIMARY = "primary"


#: Where a payload too long for its record is kept whole, beside the log
#: (BACKLOG EV-14: the field session's 50,529-character helper answer was
#: cut at 32,000 and a helper's summary at 8,000; the record could not say
#: what the primary had been given). The record names the file in its
#: ``full`` / ``text_full`` / ``reasoning_full`` field.
SIDECAR_DIR = "activity_full"


def activity_path(conv_dir: str) -> str:
    return os.path.join(conv_dir, FILE_NAME)


#: Who writes the events that are not an agent's: a restore, a page error,
#: the process start.
APP = "app"

#: Most characters of a vision side call's prompt kept on its record; a
#: longer prompt keeps this much and a :func:`fingerprint` (EV-06).
SIDE_PROMPT_CHARS = 300

#: Most files listed on one ``turn_files`` record (``n_files`` says how many
#: there were).
MAX_TURN_FILES = 200

#: Largest file :func:`file_facts` hashes; a larger one (a 200 MB scanned
#: PDF a tool names) gets its ``mtime`` instead, so the record never holds
#: up the end of a turn reading it.
HASH_MAX_BYTES = 32 * 1024 * 1024


def fingerprint(data: Any) -> str:
    """The first 16 hex digits of the SHA-256 of ``data`` (text or bytes):
    two records with the same fingerprint carried the same text. Too short
    to look like a key."""
    if not isinstance(data, (bytes, bytearray)):
        data = str("" if data is None else data).encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()[:16]


def file_facts(path: str, base: Optional[str] = None) -> Optional[dict]:
    """``{"path", "bytes", "sha"}`` of a file: its path relative to ``base``
    ('/' separators; the file name alone when it lies outside), its size and
    a :func:`fingerprint` of its content -- or, over
    :data:`HASH_MAX_BYTES`, its ``mtime`` in place of the ``sha``. ``None``
    when it cannot be read."""
    try:
        st = os.stat(path)
        if st.st_size > HASH_MAX_BYTES:
            h, size = None, int(st.st_size)
        else:
            h, size = hashlib.sha256(), 0
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
                    size += len(block)
    except OSError:
        return None
    rel = os.path.basename(path)
    if base:
        try:
            r = os.path.relpath(os.path.abspath(path), os.path.abspath(base))
            if not r.startswith(".."):
                rel = r.replace(os.sep, "/")
        except ValueError:                             # another drive
            pass
    if h is None:
        return {"path": rel, "bytes": size, "mtime": round(st.st_mtime, 3)}
    return {"path": rel, "bytes": size, "sha": h.hexdigest()[:16]}


#: This process, as the ``process_start`` record names it: a token unique
#: to it (a pid can repeat across container restarts), its pid, and when it
#: started (psutil's word where installed, else when this module loaded).
_PROCESS_TOKEN = uuid.uuid4().hex[:12]
_IMPORTED_AT = time.time()
_PROCESS_NOTED: set = set()

#: The ``tools_offered_sha`` whose list this process last wrote into each
#: conversation's log: the list is written again only when it changes.
_TOOLS_NOTED: dict = {}


def _log_key(conv_dir: str) -> str:
    return os.path.normcase(os.path.abspath(conv_dir))


def _process_started() -> float:
    try:
        import psutil
        return float(psutil.Process().create_time())
    except Exception:                                  # noqa: BLE001
        return _IMPORTED_AT


def _app_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("geotech-staff-engineer")
    except Exception:                                  # noqa: BLE001
        return None


def _process_record(conv_dir: str) -> Optional[dict]:
    """The ``process_start`` record, the first time this process writes to
    ``conv_dir``'s log; ``None`` after that."""
    key = _log_key(conv_dir)
    with _FILE_LOCKS_GUARD:
        if key in _PROCESS_NOTED:
            return None
        _PROCESS_NOTED.add(key)
    return {"agent": APP, "event": "process_start",
            "process": _PROCESS_TOKEN, "pid": os.getpid(),
            "process_started": round(_process_started(), 3),
            "app_version": _app_version()}


def _append(conv_dir: str, recs: list) -> int:
    """Append whole lines to the log, under its lock. Returns how many."""
    os.makedirs(conv_dir, exist_ok=True)
    lines = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                    for r in recs)
    path = activity_path(conv_dir)
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(lines)
    return len(recs)


#: App events waiting for the next turn's log, by conversation folder
#: (:func:`note_restored`).
_PENDING: dict = {}


def note_restored(conv_dir: str, **fields) -> None:
    """Keep a ``restored`` event (the folder came back from the mirror or a
    zip: ``source``, ``files``, ...) for the conversation's next turn,
    whose log writes it before ``turn_start`` with the time of the restore.
    Held rather than appended at once so the restored record stays the
    mirror's copy byte for byte (and re-mirrors nothing) until a turn adds
    to it. The restored log may not carry this process's ``process_start``
    or its last tool list, so the next turn writes both again. Never
    raises."""
    try:
        key = _log_key(conv_dir)
        rec = {"ts": round(time.time(), 3), "t": None, "agent": APP,
               "event": "restored",
               **{k: _json_safe(v) for k, v in fields.items()
                  if k not in ("turn", "ts", "t", "agent", "event")}}
        with _FILE_LOCKS_GUARD:
            _PENDING.setdefault(key, []).append(rec)
            _PROCESS_NOTED.discard(key)
            _TOOLS_NOTED.pop(key, None)
    except Exception:                                  # noqa: BLE001
        pass


def _take_pending(conv_dir: str) -> list:
    key = _log_key(conv_dir)
    with _FILE_LOCKS_GUARD:
        return _PENDING.pop(key, [])


def append_event(conv_dir: str, event: str, turn: Optional[int] = None,
                 **fields) -> bool:
    """Write one app event outside a turn (``page_error``):
    ``{"turn", "ts", "agent": "app", "event", **fields}``, after the
    process's ``process_start`` when it is the process's first record
    there. Never raises; ``True`` when written."""
    try:
        now = round(time.time(), 3)
        recs = []
        proc = _process_record(conv_dir)
        if proc is not None:
            recs.append({"turn": turn, "ts": now, **proc})
        recs.append({"turn": turn, "ts": now, "agent": APP, "event": event,
                     **{k: _json_safe(v) for k, v in fields.items()
                        if k not in ("turn", "ts", "agent", "event")}})
        _append(conv_dir, recs)
        return True
    except Exception:                                  # noqa: BLE001
        return False


def _helper_fields() -> dict:
    """``task_id`` and ``helper`` of the ``task`` helper this event runs
    inside (``funhouse_agent.deep.keep_work``), else nothing. Field session
    2026-10-14: every helper's events said only ``general-purpose``, and 607
    of 916 vision calls could not be tied to a helper."""
    try:
        from funhouse_agent.deep.keep_work import current_helper
        rec = current_helper()
    except Exception:                                  # noqa: BLE001
        return {}
    if rec is None:
        return {}
    brief = " ".join(str(rec.brief or "").split())
    return {"task_id": rec.task_id,
            "helper": (brief[:80] + " ...") if len(brief) > 80 else brief}


def _side_call_fields() -> Optional[dict]:
    """The label of the vision side call running here (``kind``, ``page``,
    ``tile``, ``image_px``, ``detail``, ``queued_s``), if any."""
    try:
        from funhouse_agent.deep.vision_engine import current_side_call
        info = current_side_call()
    except Exception:                                  # noqa: BLE001
        return None
    return dict(info) if info else None


def _refusal_fields(error) -> dict:
    """A model error's HTTP status, ``retry-after`` and ``x-ratelimit-*``
    headers, and the busy ladder's tries when it gave up."""
    out: dict = {}
    try:
        from funhouse_agent.deep.busy_retry import refusal_facts
        out.update(refusal_facts(error))
    except Exception:                                  # noqa: BLE001
        pass
    busy = getattr(error, "geotech_busy", None)
    if isinstance(busy, dict):
        out["busy_ladder"] = busy
    return out


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
    # A LangGraph Command (a sub-agent's ``task`` result is
    # ``Command(update={"messages": [ToolMessage(...)], ...})``): log what
    # its messages SAY, not the object's repr (live smoke wave 1, A15e).
    update = getattr(obj, "update", None)
    if isinstance(update, dict) and not hasattr(obj, "content"):
        msgs = update.get("messages")
        if isinstance(msgs, (list, tuple)):
            texts = [_to_text(m) for m in msgs]
            texts = [t for t in texts if t]
            if texts:
                return "\n".join(texts)
    if isinstance(obj, (list, tuple)) and obj and all(
            hasattr(o, "content") or hasattr(o, "update") for o in obj):
        return "\n".join(t for t in (_to_text(o) for o in obj) if t)
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


#: Most characters of a model's own words kept per call (its text, and its
#: reasoning summary, each). A tool result's cap is larger: this is the
#: model's narration, not a document.
MODEL_TEXT_CHARS = 8_000


def _model_output(response: Any) -> tuple:
    """``(text, reasoning)`` of a chat model's reply, as far as the provider
    returns them: plain content, text blocks, and reasoning / thinking blocks
    or ``reasoning_content`` / ``reasoning`` in ``additional_kwargs``."""
    texts, thoughts = [], []
    try:
        for row in getattr(response, "generations", None) or []:
            for g in row or []:
                msg = getattr(g, "message", None)
                content = (getattr(msg, "content", None) if msg is not None
                           else getattr(g, "text", None))
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            texts.append(block)
                        elif isinstance(block, dict):
                            kind = str(block.get("type") or "")
                            if kind in ("reasoning", "thinking"):
                                summ = block.get("summary")
                                if isinstance(summ, list):
                                    thoughts.extend(
                                        str(s.get("text") if isinstance(s, dict)
                                            else s) for s in summ)
                                thoughts.append(str(block.get("thinking")
                                                    or block.get("reasoning")
                                                    or ""))
                            elif "text" in block:
                                texts.append(str(block.get("text") or ""))
                extra = getattr(msg, "additional_kwargs", None) or {}
                for key in ("reasoning_content", "reasoning"):
                    value = extra.get(key)
                    if isinstance(value, dict):
                        value = value.get("summary") or value.get("text")
                    if isinstance(value, list):
                        value = " ".join(str(v.get("text") if isinstance(v, dict)
                                             else v) for v in value)
                    if value:
                        thoughts.append(str(value))
    except Exception:                                  # noqa: BLE001
        pass
    return ("\n".join(t for t in texts if t).strip(),
            "\n".join(t for t in thoughts if t).strip())


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


def _tool_messages(obj: Any) -> list:
    """The tool messages in a graph node's output (a ToolMessage, a
    ``{"messages": [...]}`` update, a Command, or a list of them)."""
    if getattr(obj, "type", None) == "tool" and hasattr(obj, "tool_call_id"):
        return [obj]
    if isinstance(obj, dict):
        return _tool_messages(obj.get("messages"))
    if isinstance(obj, (list, tuple)):
        out: list = []
        for o in obj:
            out.extend(_tool_messages(o))
        return out
    update = getattr(obj, "update", None)                # a Command
    if isinstance(update, dict):
        return _tool_messages(update.get("messages"))
    return []


def _delivered_text(content: Any) -> str:
    """A message's content as the delivered-result event writes it
    (``funhouse_agent.deep.delivered_log``), so the two compare."""
    try:
        from funhouse_agent.deep.delivered_log import _text
        return _text(content)
    except Exception:                                  # noqa: BLE001
        return _to_text(content)


_MOVED = (re.compile(r"at this path:\s*(\S+)"),
          re.compile(r"(\S*/large_tool_results/\S+)"))


def _moved_to(text: str) -> Optional[str]:
    """The scratch path a moved result's notice names, else ``None``."""
    for pattern in _MOVED:
        m = pattern.search(text or "")
        if m:
            return m.group(1).rstrip(".,;:)'\"`")
    return None


def _side_prompt(messages: Any) -> dict:
    """What a vision side call sent besides its image: the prompt text
    (whole up to :data:`SIDE_PROMPT_CHARS`, else that much and a
    ``prompt_sha`` of the whole), ``prompt_chars``, the ``detail`` each image
    was sent with (``detail_sent``) and how many ``images``. Never the
    image; anything in the text shaped like a key is masked."""
    texts: list = []
    details: list = []
    images = 0
    for batch in messages or []:
        for m in batch or []:
            content = getattr(m, "content", m)
            if isinstance(content, str):
                texts.append(content)
                continue
            for b in content if isinstance(content, list) else []:
                if isinstance(b, str):
                    texts.append(b)
                elif isinstance(b, dict):
                    kind = str(b.get("type") or "")
                    if "image" in kind:
                        images += 1
                        url = b.get("image_url")
                        d = (url.get("detail") if isinstance(url, dict)
                             else None) or b.get("detail")
                        if d:
                            details.append(str(d))
                    elif isinstance(b.get("text"), str):
                        texts.append(b["text"])
    prompt = "\n".join(t for t in texts if t)
    try:
        from funhouse_agent.error_text import mask_secrets
        shown = mask_secrets(prompt)
    except Exception:  # noqa: BLE001 - no excerpt rather than a raw one
        shown = ""
    out: dict = {"prompt": shown[:SIDE_PROMPT_CHARS],
                 "prompt_chars": len(prompt), "images": images}
    if len(prompt) > SIDE_PROMPT_CHARS:
        out["prompt_sha"] = fingerprint(prompt)
    if details:
        out["detail_sent"] = (details[0] if len(set(details)) == 1
                              else sorted(set(details)))
    return out


def _tool_name(spec: Any) -> Optional[str]:
    """A tool's name from the spec a model call was sent (OpenAI's
    ``{"function": {"name"}}``, Anthropic's ``{"name"}``, or a tool)."""
    if isinstance(spec, dict):
        fn = spec.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        return str(spec["name"]) if spec.get("name") else None
    name = getattr(spec, "name", None)
    return str(name) if name else None


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
        self._active: dict = {}           # open task run_id -> subagent name
        self._task_sub: dict = {}         # every task run_id -> subagent name
        self._parent: dict = {}           # run_id -> parent run_id (or None)
        self._names: dict = {}            # run_id -> tool name
        self._starts: dict = {}           # run_id -> start time
        self._sources: dict = {}          # model run_id -> lc_source
        self._call_ids: dict = {}         # tool run_id -> tool_call_id
        self._outputs: dict = {}          # tool_call_id -> the tool's output
        self._vision: dict = {}           # tool run_id -> [calls, queued_s]
        self._lock_wait: dict = {}        # tool run_id -> document lock wait
        self._side: dict = {}             # model run_id -> side call facts
        self._seen: dict = {}             # tool_call_id -> text as delivered
        self._tool_nodes: set = set()     # run_ids of graph "tools" nodes
        self._passes: list = []           # highest primary step, per pass
        self._system: dict = {}           # primary system prompt sha -> chars
        self._tools_sent: set = set()     # tool names the primary was sent
        self._build_tools: Optional[list] = None
        self.step_cap: Optional[int] = None
        self.step_allowance: Optional[int] = None
        self._lock = threading.Lock()
        self.records_written = 0
        self.last_error: Optional[str] = None

    # -- attribution ------------------------------------------------------
    @property
    def agent(self) -> str:
        """The most recently opened sub-agent still running, else primary
        (a best guess; records use :meth:`_agent_of`)."""
        return next(reversed(self._active.values())) if self._active else PRIMARY

    def _note(self, run_id, parent_run_id) -> None:
        self._parent[str(run_id)] = str(parent_run_id) if parent_run_id else None

    def _agent_of(self, parent_run_id) -> str:
        """The agent whose run contains a run with this parent: walk up to a
        ``task`` call (that sub-agent) or to the top (primary)."""
        pid = str(parent_run_id) if parent_run_id else None
        seen = set()
        while pid and pid not in seen:
            if pid in self._task_sub:
                return self._task_sub[pid]
            seen.add(pid)
            if pid not in self._parent:       # a run we never saw start
                return self.agent
            pid = self._parent[pid]
        return PRIMARY

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       metadata=None, **kwargs):
        # Graph and node runs sit between a task call and the sub-agent's own
        # model/tool runs; their parentage is needed, and they carry the
        # graph's step count (EV-06).
        try:
            self._note(run_id, parent_run_id)
            if parent_run_id is None:
                # a graph run at the top: one pass of the turn (an
                # auto-continue or a checker's retry is another)
                self._passes.append(0)
                return
            md = metadata if isinstance(metadata, dict) else {}
            step = md.get("langgraph_step")
            if not isinstance(step, int):
                return
            if kwargs.get("name") == "tools" \
                    and md.get("langgraph_node") == "tools":
                self._tool_nodes.add(str(run_id))
            # The primary's own graph: a helper's nodes run in a namespace
            # nested under the task call ("tools:<id>|model:<id>").
            if "|" not in str(md.get("langgraph_checkpoint_ns") or "") \
                    and self._agent_of(parent_run_id) == PRIMARY:
                if not self._passes:
                    self._passes.append(0)
                self._passes[-1] = max(self._passes[-1], step)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        """A graph ``tools`` node has finished: its messages are what the
        model will be given. Where one differs from the text last seen for
        that call (the tool's own output, or as the app's middleware
        delivered it), the agent's graph changed it -- deepagents moves a
        large result to ``/large_tool_results/`` and sends a notice and a
        preview instead -- and it is logged as ``tool_delivered`` with
        ``stage: "graph"`` (EV-19; field session 2026-10-15: the record
        kept the first 32,000 characters of a 277,385-character result,
        never the notice the model read)."""
        rid = str(run_id)
        if rid not in self._tool_nodes:
            return
        try:
            self._tool_nodes.discard(rid)
            for msg in _tool_messages(outputs):
                self._graph_delivered(msg, rid)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._tool_nodes.discard(str(run_id))

    def _graph_delivered(self, msg, node_rid: str) -> None:
        call_id = str(getattr(msg, "tool_call_id", "") or "")
        if not call_id:
            return
        if call_id in self._seen:
            prev, now = self._seen.pop(call_id), _delivered_text(msg.content)
        elif call_id in self._outputs:
            prev, now = self._outputs.pop(call_id), _to_text(msg)
        else:
            return                       # nothing seen to compare with
        if prev == now:
            return
        text, truncated, n, whole = self._cap_whole(
            now, self.max_chars, f"delivered_{getattr(msg, 'name', '') or ''}",
            call_id)
        rec = {"agent": self._agent_of(node_rid), "event": "tool_delivered",
               "name": getattr(msg, "name", None), "tool_call_id": call_id,
               "status": getattr(msg, "status", None), "stage": "graph",
               "text": text, "truncated": truncated, "chars": n,
               "chars_before": len(prev)}
        moved = _moved_to(now)
        if moved:
            rec["moved_to"] = moved
        if whole:
            rec["full"] = whole
        self._write(rec)

    def note_step_cap(self, cap: Optional[int],
                      allowance: Optional[int] = None) -> None:
        """The step cap the turn's run was given, and the ordinary cap it
        travels with when the extraction cap raised it (``webapp.core.
        stream_turn``); written on ``turn_end`` beside ``graph_steps``."""
        try:
            self.step_cap = int(cap) if cap else None
            self.step_allowance = int(allowance) if allowance else None
        except (TypeError, ValueError):
            pass

    def _tool_of(self, parent_run_id) -> Optional[str]:
        """The run id of the tool call a run sits inside (the nearest open
        tool run up its parent chain), else ``None``."""
        pid = str(parent_run_id) if parent_run_id else None
        seen = set()
        while pid and pid not in seen:
            if pid in self._names:
                return pid
            seen.add(pid)
            pid = self._parent.get(pid)
        return None

    def _cap_whole(self, text: Any, max_chars: int, label: str,
                   run_id: Any) -> tuple:
        """``(text, truncated, full_len, sidecar)``: ``text`` cut at
        ``max_chars`` as :func:`_cap` does, the whole of it written to
        :data:`SIDECAR_DIR` when cut (``sidecar`` its name relative to the
        conversation folder, else ``None``)."""
        whole = text if isinstance(text, str) else _to_text(text)
        cut, truncated, n = _cap(whole, max_chars)
        if not truncated:
            return cut, False, n, None
        try:
            folder = os.path.join(self.conv_dir, SIDECAR_DIR)
            os.makedirs(folder, exist_ok=True)
            safe = "".join(ch if ch.isalnum() else "-" for ch in label)[:40]
            name = f"turn{self.turn}_{safe}_{str(run_id or '')[:8]}.txt"
            with open(os.path.join(folder, name), "w",
                      encoding="utf-8") as fh:
                fh.write(whole)
            return cut, True, n, f"{SIDECAR_DIR}/{name}"
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return cut, True, n, None

    # -- writer -----------------------------------------------------------
    def _write(self, rec: dict) -> None:
        try:
            now = self._clock()
            rec = {"turn": self.turn, "ts": round(now, 3),
                   "t": round(now - self._t0, 3), **_helper_fields(), **rec}
            os.makedirs(self.conv_dir, exist_ok=True)
            line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            path = activity_path(self.conv_dir)
            # One whole line at a time: parallel tools write here at once.
            with _lock_for(path):
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line)
                self.records_written += 1
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    # -- turn envelope ----------------------------------------------------
    def turn_start(self, prompt: Optional[str] = None,
                   model: Optional[str] = None,
                   note: Optional[str] = None,
                   engine: Optional[dict] = None) -> None:
        """``note`` is what the app put in front of the user's message for
        this turn (the working-folder note, ``core.working_files_note``):
        part of what the model was shown, so it belongs in the record.
        ``engine`` is which client served the turn and the retry, pacing
        and helper policy in force (``webapp.core.engine_record``)."""
        proc = _process_record(self.conv_dir)
        if proc is not None:
            self._write(proc)
        for held in _take_pending(self.conv_dir):    # a restore since
            self._write(held)
        rec = {"agent": PRIMARY, "event": "turn_start",
               "prompt": (prompt or "")[:500], "model": model}
        if engine:
            rec["engine"] = engine
            tools = (engine.get("build") or {}).get("tools") \
                if isinstance(engine.get("build"), dict) else None
            if isinstance(tools, list):
                self._build_tools = [str(t) for t in tools]
        if note:
            rec["context_note"] = str(note)[:DEFAULT_MAX_CHARS]
        self._write(rec)

    def turn_end(self, *, turn_tokens: int = 0, error: Optional[str] = None,
                 answer_chars: int = 0,
                 error_detail: Optional[str] = None) -> None:
        """``error`` is what the user was shown; ``error_detail`` the
        provider's own text when that was NOT shown (a spent AI budget:
        its JSON and request id are for the owner, live smoke 2c, D2).
        Also the turn's graph steps against its cap, the fingerprint of
        the primary's system prompt and the tools it was offered
        (EV-06; see the module docstring)."""
        rec = {"agent": PRIMARY, "event": "turn_end",
               "duration_s": round(self._clock() - self._t0, 3),
               "turn_tokens": turn_tokens, "answer_chars": answer_chars,
               "error": error}
        if error_detail:
            rec["error_detail"] = str(error_detail)[:DEFAULT_MAX_CHARS]
        try:
            rec.update(self._turn_facts())
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
        written = self.records_written
        self._write(rec)
        if "tools_offered" in rec and self.records_written > written:
            with _FILE_LOCKS_GUARD:
                _TOOLS_NOTED[_log_key(self.conv_dir)] = \
                    rec.get("tools_offered_sha")

    def _turn_facts(self) -> dict:
        out: dict = {}
        passes = [s for s in self._passes if s]
        if passes:
            out["graph_steps"] = max(passes)
            if len(passes) > 1:
                out["graph_steps_by_pass"] = passes
        if self.step_cap:
            out["step_cap"] = self.step_cap
        if self.step_allowance:
            out["step_allowance"] = self.step_allowance
        prompts = [{"sha": sha, "chars": n}
                   for sha, n in self._system.items()]
        if prompts:
            out["system_prompt"] = prompts[0] if len(prompts) == 1 else prompts
        names = sorted(self._tools_sent or self._build_tools or ())
        if names:
            # The names themselves only when this process has not already
            # written the same list into this log (a long conversation
            # repeated them every turn); the sha says which list it was.
            sha = fingerprint(json.dumps(names))
            out["tools_offered_sha"] = sha
            out["tools_offered_from"] = ("calls" if self._tools_sent
                                         else "build")
            with _FILE_LOCKS_GUARD:
                same = _TOOLS_NOTED.get(_log_key(self.conv_dir)) == sha
            if not same:
                out["tools_offered"] = names
        return out

    def turn_files(self, files: list) -> None:
        """The files the turn wrote or listed, each ``{"path", "bytes",
        "sha", "how"}`` (:func:`file_facts`; ``how`` is ``new``,
        ``rewritten`` or ``listed``), written after ``turn_end`` by the
        app's turn job (EV-19; field session 2026-10-15: a file rewritten
        under its own name looked like two outputs). Nothing when the turn
        wrote nothing."""
        files = [f for f in files or () if isinstance(f, dict)]
        if not files:
            return
        rec = {"agent": PRIMARY, "event": "turn_files",
               "files": files[:MAX_TURN_FILES], "n_files": len(files)}
        self._write(rec)

    # -- tools ------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, metadata=None, inputs=None, **kwargs):
        try:
            name = (kwargs.get("name") or (serialized or {}).get("name")
                    or "tool")
            rid = str(run_id)
            self._names[rid] = name
            self._starts[rid] = self._clock()
            self._note(rid, parent_run_id)
            if kwargs.get("tool_call_id"):
                self._call_ids[rid] = str(kwargs["tool_call_id"])
            args = inputs if inputs is not None else input_str
            rec = {"agent": self._agent_of(parent_run_id), "event": "tool_start",
                   "name": name, "run_id": rid,
                   "parent_run_id": str(parent_run_id) if parent_run_id else None,
                   "args": _json_safe(args)}
            if name == TASK_TOOL:
                sub = None
                if isinstance(inputs, dict):
                    sub = inputs.get("subagent_type")
                sub = str(sub or "subagent")
                rec["subagent"] = sub
                self._write(rec)
                self._task_sub[rid] = sub
                self._active[rid] = sub
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
        # result belongs to whoever delegated it, found through its parent.
        self._active.pop(rid, None)
        sub = self._task_sub.get(rid)
        if parent_run_id is None:
            parent_run_id = self._parent.get(rid)
        agent = self._agent_of(parent_run_id)
        text, truncated, full_len, sidecar = self._cap_whole(
            payload, self.max_chars, f"{event}_{name}", rid)
        rec = {"agent": agent, "event": event, "name": name, "run_id": rid,
               "parent_run_id": str(parent_run_id) if parent_run_id else None,
               payload_key: text, "truncated": truncated,
               "chars": full_len,
               "duration_s": (round(self._clock() - t_start, 3)
                              if t_start is not None else None)}
        if sidecar:
            rec["full"] = sidecar
        if sub:
            rec["subagent"] = sub
        # Time the tool spent waiting rather than working (EV-14): its
        # vision calls' waits for a slot, its waits for the document lock.
        with self._lock:
            vision = self._vision.pop(rid, None)
            lock_wait = self._lock_wait.pop(rid, None)
        if vision:
            rec["vision_calls"] = vision[0]
            rec["vision_queued_s"] = round(vision[1], 3)
        if lock_wait:
            rec["lock_wait_s"] = round(lock_wait, 3)
        self._write(rec)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        try:
            call_id = self._call_ids.pop(str(run_id), None)
            if call_id:
                self._outputs[call_id] = _to_text(output)
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
            self._note(rid, parent_run_id)
            n_msgs = sum(len(m) for m in (messages or []))
            model = None
            try:
                model = ((serialized or {}).get("kwargs") or {}).get("model") \
                    or ((serialized or {}).get("kwargs") or {}).get("model_name") \
                    or (serialized or {}).get("name")
            except Exception:                          # noqa: BLE001
                model = None
            rec = {"agent": self._agent_of(parent_run_id),
                   "event": "model_start",
                   "run_id": rid,
                   "parent_run_id": (str(parent_run_id)
                                     if parent_run_id else None),
                   "model": model, "n_messages": n_msgs}
            # langchain's own internal calls say what they are (the context
            # summarizer: ``lc_source: summarization``) -- not a reply.
            source = (metadata or {}).get("lc_source") \
                if isinstance(metadata, dict) else None
            if source:
                rec["source"] = str(source)
                self._sources[rid] = str(source)
            # A vision side call: what it looked at, and the tool it ran in
            # (EV-14; field session 2026-10-14: a bare model_start with
            # n_messages 1).
            side = _side_call_fields()
            if side:
                rec["vision"] = side
                # what was sent with the image, for its model_end (EV-06)
                self._side[rid] = {**side, **_side_prompt(messages)}
                tool_rid = self._tool_of(parent_run_id)
                if tool_rid:
                    rec["tool"] = self._names.get(tool_rid)
                    rec["tool_run_id"] = tool_rid
                    with self._lock:
                        acc = self._vision.setdefault(tool_rid, [0, 0.0])
                        acc[0] += 1
                        acc[1] += float(side.get("queued_s") or 0.0)
            elif rec["agent"] == PRIMARY and not source:
                self._primary_call(messages, kwargs)
            try:
                from webapp import http_trace
                http_trace.begin()
            except Exception:                          # noqa: BLE001
                pass
            self._write(rec)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _primary_call(self, messages, kwargs: dict) -> None:
        """Note what a primary model call was sent: a fingerprint of its
        system message, and the names of the tools it was offered where the
        client puts them in its invocation parameters (an OpenAI-style
        client does; a client that keeps them on itself does not, and the
        build's list stands in)."""
        try:
            first = messages[0][0] if messages and messages[0] else None
            if getattr(first, "type", None) == "system":
                text = _to_text(first)
                with self._lock:
                    self._system.setdefault(fingerprint(text), len(text))
            params = kwargs.get("invocation_params")
            tools = params.get("tools") if isinstance(params, dict) else None
            if isinstance(tools, (list, tuple)):
                names = {n for n in (_tool_name(t) for t in tools) if n}
                with self._lock:
                    self._tools_sent.update(names)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _side_fields(self, rid: str) -> dict:
        """A vision side call's label and what it sent (taken at its
        ``model_start``), with any facts the call added since."""
        sent = self._side.pop(rid, None)
        if sent is None:
            return {}
        return {"vision": {**(_side_call_fields() or {}), **sent}}

    @staticmethod
    def _http_fields(require_more: bool = True) -> dict:
        """The HTTP tries of the call that just ended on this thread
        (``webapp.http_trace``): kept when there was more than one, or one
        that was not a 200."""
        try:
            from webapp import http_trace
            tries = http_trace.take()
        except Exception:                              # noqa: BLE001
            return {}
        if not tries:
            return {}
        if require_more and len(tries) == 1 and tries[0].get("status") == 200:
            return {}
        return {"http_tries": len(tries), "http": tries}

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        try:
            rid = str(run_id)
            t_start = self._starts.pop(rid, None)
            parent = parent_run_id or self._parent.get(rid)
            record = {"agent": self._agent_of(parent), "event": "model_end",
                      "run_id": rid,
                      "parent_run_id": (str(parent_run_id)
                                        if parent_run_id else None),
                      "usage": _usage_from_response(response),
                      "n_tool_calls": _n_tool_calls(response),
                      "duration_s": (round(self._clock() - t_start, 3)
                                     if t_start is not None else None)}
            source = self._sources.pop(rid, None)
            if source:
                record["source"] = source
            text, reasoning = _model_output(response)
            if text:
                (record["text"], record["text_truncated"], _n,
                 whole) = self._cap_whole(text, MODEL_TEXT_CHARS,
                                          f"model_text_{source or ''}", rid)
                if whole:
                    record["text_full"] = whole
            if reasoning:
                (record["reasoning"], _t, _n, whole) = self._cap_whole(
                    reasoning, MODEL_TEXT_CHARS, "model_reasoning", rid)
                if whole:
                    record["reasoning_full"] = whole
            record.update(self._side_fields(rid))
            record.update(self._http_fields())
            self._write(record)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None,
                        **kwargs):
        """App events inside a run. ``coverage_gate``: the note the coverage
        gate put in front of the model (``funhouse_agent.deep.
        coverage_tools``) -- part of what the model was shown, so it belongs
        in the record."""
        try:
            if name == "tool_delivered":
                self._delivered(data, run_id)
                return
            if name == "document_lock_wait":
                # Added to the tool call's own tool_end (``lock_wait_s``).
                waited = float((data or {}).get("waited_s") or 0.0) \
                    if isinstance(data, dict) else 0.0
                tool_rid = (str(run_id) if str(run_id) in self._names
                            else self._tool_of(run_id))
                if tool_rid and waited > 0:
                    with self._lock:
                        self._lock_wait[tool_rid] = \
                            self._lock_wait.get(tool_rid, 0.0) + waited
                return
            if name in APP_EVENTS:
                rec = {"agent": self._agent_of(run_id), "event": name}
                if isinstance(data, dict):
                    rec.update({k: _json_safe(v) for k, v in data.items()
                                if k not in ("agent", "event")})
                self._write(rec)
                return
            if name != "coverage_gate":
                return
            note = (data or {}).get("note") if isinstance(data, dict) else data
            text, truncated, _n = _cap(note, self.max_chars)
            self._write({"agent": PRIMARY, "event": "coverage_gate",
                         "run_id": str(run_id), "text": text,
                         "truncated": truncated})
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _delivered(self, data, run_id) -> None:
        """A tool result as the model received it (``funhouse_agent.deep.
        delivered_log``): logged when no tool ran (``tool_refused`` -- an
        unknown tool name, a guard's interception) or when what the model
        got differs from the tool's own output (``tool_delivered``: a note
        added, a result replaced). Identical results are not logged twice."""
        if not isinstance(data, dict):
            return
        call_id = str(data.get("tool_call_id") or "")
        content = str(data.get("content") or "")
        own = self._outputs.pop(call_id, None) if call_id else None
        if call_id:
            # what the graph's own change, if any, is measured against
            self._seen[call_id] = content
        if own is not None and own == content:
            return
        refused = bool(data.get("refused")) or own is None
        text, truncated, n, whole = self._cap_whole(
            content, self.max_chars, f"delivered_{data.get('name') or ''}",
            call_id or run_id)
        rec = {"agent": self._agent_of(run_id),
               "event": "tool_refused" if refused else "tool_delivered",
               "name": data.get("name"), "tool_call_id": call_id or None,
               "status": data.get("status"), "text": text,
               "truncated": truncated, "chars": n}
        if whole:
            rec["full"] = whole
        self._write(rec)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        try:
            rid = str(run_id)
            t_start = self._starts.pop(rid, None)
            text, truncated, _n = _cap(f"{type(error).__name__}: {error}",
                                       self.max_chars)
            parent = parent_run_id or self._parent.get(rid)
            # What the refusal said (EV-14: the record held only the
            # exception's text): its status, retry-after, rate-limit
            # headers, and each HTTP try inside the call.
            self._write({"agent": self._agent_of(parent), "event": "model_error",
                         "run_id": rid,
                         "parent_run_id": (str(parent_run_id)
                                           if parent_run_id else None),
                         "error": text, "truncated": truncated,
                         "duration_s": (round(self._clock() - t_start, 3)
                                        if t_start is not None else None),
                         **self._side_fields(rid),
                         **_refusal_fields(error),
                         **self._http_fields(require_more=False)})
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


__all__ = ["ActivityLogger", "activity_path", "load", "append_event",
           "note_restored", "file_facts", "fingerprint", "FILE_NAME",
           "DEFAULT_MAX_CHARS", "TASK_TOOL", "PRIMARY", "APP"]
