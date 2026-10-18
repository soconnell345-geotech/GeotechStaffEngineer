"""Work a turn has done is kept when an agent stops (BACKLOG TA-19, TA-21).

Field session 2026-10-14 (report to DIGGS, 51 minutes): four ``task``
helpers held extracted data when they stopped -- one 21 borings, 53,000
characters -- and handed back two apologies and two error strings. A
helper's context dies with it: deepagents' ``task`` tool returns the
helper's last message, and an exception returns nothing at all. The turn
itself then ended on a rate-limit refusal and kept nothing either.

This module holds what one turn shares across its agents and what each
helper did:

* :class:`TurnScope` -- one per turn, in a context variable that LangGraph
  copies into every thread a tool or a sub-agent runs on. It carries the
  full text of the primary's very large tool results (deepagents swaps a
  result over ~80,000 characters for a pointer into its scratch space, which
  dies with the turn), the helper files written this turn, and the turn's
  bound on waiting for a busy model (:mod:`funhouse_agent.deep.busy_retry`).
* :class:`HelperRecorder` -- on every sub-agent: records each model reply
  (its words and its tool calls with their arguments in full) and each tool
  result, as they happen, into the running helper's :class:`HelperRecord`.
* :class:`KeepWork` -- on the primary: wraps every ``task`` call. When the
  helper raises, reaches its model-call budget, or hands back a short answer
  after reading a lot (:func:`gave_up`), everything it read and wrote goes to
  ``kept/helper_<what>_<id>.md`` in the conversation's working folder, and
  the primary is told the file name and given a one-paragraph summary made in
  code (no model call: the usual reason a helper stopped is a busy model).
  It also caps the helpers that run at once in one conversation
  (:func:`max_helpers`; BACKLOG TA-22).

Nothing here changes a tool's result except a ``task`` result that kept a
file or whose helper saved files (their names are added, BACKLOG RI-31), and
nothing here ever fails a call.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

from langchain_core.messages import ToolMessage

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

from funhouse_agent.deep.tool_guards import GraphBubbleUp, _ai_of, _text

#: The folder, inside the conversation's working folder, that kept work goes
#: into. Its files get download cards like any file a turn writes, and the
#: next turn's note of the conversation's files lists them.
KEPT_DIR = "kept"

#: The delegation tool whose helpers are kept.
TASK_TOOL = "task"

#: Helpers whose budget stop or short answer keeps a file. Other sub-agents
#: (the references consult, the reviewer) answer briefly BY DESIGN after
#: reading reference text, so only an error keeps theirs.
KEEP_ON_GIVE_UP = frozenset({"general-purpose"})

#: A helper's work is worth a file once it has read this many characters of
#: tool results (below it there is little to lose, and a card would be
#: clutter).
KEEP_MIN_READ_CHARS = 20_000

#: An answer shorter than this, after reading at least
#: :data:`KEEP_MIN_READ_CHARS`, counts as giving up (the field session's two
#: apologies were 366 and 268 characters, after 413,000 and about 1,000,000
#: characters of reading).
SHORT_ANSWER_CHARS = 2_000

#: The primary's tool results at least this long are kept whole on the turn
#: scope (deepagents evicts results of 80,000 characters or more from the
#: conversation into its scratch space, which a turn's error loses).
FULL_RESULT_MIN_CHARS = 60_000


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------

#: Seconds a turn may spend, in all, waiting on a busy model (rate limit,
#: overload) before it gives up: a dead endpoint must not hold a turn for
#: ever. Waits of helpers running at once are added together.
BUSY_TURN_MAX_ENV = "GEOTECH_BUSY_TURN_MAX_S"
DEFAULT_BUSY_TURN_MAX_S = 900.0


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


class TurnScope:
    """What one turn shares across its agents (see the module docstring)."""

    def __init__(self, busy_limit_s: Optional[float] = None):
        self.started = time.time()
        self.busy_limit_s = (_float_env(BUSY_TURN_MAX_ENV,
                                        DEFAULT_BUSY_TURN_MAX_S)
                             if busy_limit_s is None else float(busy_limit_s))
        self.busy_spent_s = 0.0
        self.busy_events = 0
        self.full_results: dict = {}          # tool_call_id -> full text
        self.kept_files: List[str] = []       # helper files written
        # the figures drawn this turn, by any helper or tool, and how often
        # (funhouse_agent.plot_box.figures; BACKLOG GM-151)
        self.figures: dict = {}
        self._lock = threading.Lock()

    def note_full_result(self, call_id: Optional[str], text: str) -> None:
        if not call_id or not isinstance(text, str):
            return
        with self._lock:
            self.full_results[str(call_id)] = text

    def note_kept_file(self, path: str) -> None:
        with self._lock:
            if path not in self.kept_files:
                self.kept_files.append(path)

    def spend_busy(self, seconds: float) -> bool:
        """Whether the turn may still spend ``seconds`` on a busy model; if
        so, they are counted. ``busy_limit_s <= 0`` means no bound."""
        with self._lock:
            if self.busy_limit_s > 0 and \
                    self.busy_spent_s + seconds > self.busy_limit_s:
                return False
            self.busy_spent_s += max(0.0, float(seconds))
            self.busy_events += 1
            return True


_TURN: "contextvars.ContextVar[Optional[TurnScope]]" = contextvars.ContextVar(
    "geotech_turn_scope", default=None)


def current_turn() -> Optional[TurnScope]:
    """The running turn's scope, or ``None`` outside one (a library call)."""
    return _TURN.get()


@contextmanager
def turn_scope(scope: Optional[TurnScope] = None) -> Iterator[TurnScope]:
    """``with turn_scope() as scope:`` -- the scope of one turn, seen by
    every agent, tool and helper the run starts in this context."""
    scope = scope if scope is not None else TurnScope()
    token = _TURN.set(scope)
    try:
        yield scope
    finally:
        try:
            _TURN.reset(token)
        except (ValueError, RuntimeError):     # closed from another context
            pass


# ---------------------------------------------------------------------------
# One helper
# ---------------------------------------------------------------------------

class HelperRecord:
    """Everything one ``task`` helper did, in order."""

    def __init__(self, task_id: str, subagent: str, brief: str):
        self.task_id = str(task_id or "")
        self.subagent = str(subagent or "subagent")
        self.brief = str(brief or "")
        self.started = time.time()
        self.steps: list = []        # {"kind": "model"|"result", ...}
        self.model_calls = 0
        self.read_chars = 0
        self.arg_chars = 0
        self.budget_reached = False
        self.error: Optional[BaseException] = None
        # its figures' draws when no turn is open (BACKLOG GM-151)
        self.figures: dict = {}
        self._lock = threading.Lock()

    def add_model(self, message) -> None:
        if message is None:
            return
        calls = []
        for c in getattr(message, "tool_calls", None) or []:
            args = c.get("args") if isinstance(c, dict) else None
            calls.append({"name": c.get("name") or "tool",
                          "id": c.get("id") or "",
                          "args": args if args is not None else {}})
        text = _text(getattr(message, "content", ""))
        with self._lock:
            self.model_calls += 1
            for c in calls:
                self.arg_chars += len(_json(c["args"]))
            self.steps.append({"kind": "model", "text": text,
                               "calls": calls})

    def add_result(self, name: str, call_id: str, text: str,
                   status: Optional[str] = None) -> None:
        with self._lock:
            self.read_chars += len(text or "")
            self.steps.append({"kind": "result", "name": name or "tool",
                               "id": call_id or "", "text": text or "",
                               "status": status})

    @property
    def n_results(self) -> int:
        return sum(1 for s in self.steps if s["kind"] == "result")

    def last_words(self) -> str:
        for s in reversed(self.steps):
            if s["kind"] == "model" and s["text"].strip():
                return s["text"].strip()
        return ""


_HELPER: "contextvars.ContextVar[Optional[HelperRecord]]" = \
    contextvars.ContextVar("geotech_helper", default=None)


def current_helper() -> Optional[HelperRecord]:
    """The ``task`` helper running in this context, or ``None`` (the
    primary, or a library call)."""
    return _HELPER.get()


def note_budget_reached() -> None:
    """Called by a model-call budget when it forces its last call
    (:class:`funhouse_agent.deep.limits.ModelCallBudgetMiddleware`)."""
    rec = _HELPER.get()
    if rec is not None:
        rec.budget_reached = True


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(obj)


def _result_messages(result) -> list:
    from funhouse_agent.deep.delivered_log import _messages_of
    return _messages_of(result)


class HelperRecorder(AgentMiddleware):
    """Record a sub-agent's model replies and tool results into the running
    helper's :class:`HelperRecord` (see the module docstring). Put it just
    inside the retries, so a retried call is recorded once, and outside the
    tool guards, so a result is recorded as the helper received it."""

    def wrap_model_call(self, request, handler):
        rec = _HELPER.get()
        try:
            response = handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if rec is not None:
                rec.error = exc
            raise
        if rec is not None:
            try:
                rec.add_model(_ai_of(response))
            except Exception:  # noqa: BLE001 - recording never fails a call
                pass
        return response

    async def awrap_model_call(self, request, handler):
        rec = _HELPER.get()
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if rec is not None:
                rec.error = exc
            raise
        if rec is not None:
            try:
                rec.add_model(_ai_of(response))
            except Exception:  # noqa: BLE001
                pass
        return response

    def _record(self, request, result) -> None:
        rec = _HELPER.get()
        if rec is None:
            return
        try:
            call = getattr(request, "tool_call", None) or {}
            for msg in _result_messages(result):
                rec.add_result(call.get("name") or getattr(msg, "name", None),
                               call.get("id") or getattr(msg, "tool_call_id",
                                                         ""),
                               _text(msg.content),
                               getattr(msg, "status", None))
        except Exception:  # noqa: BLE001
            pass

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        self._record(request, result)
        return result

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        self._record(request, result)
        return result


# ---------------------------------------------------------------------------
# The kept file
# ---------------------------------------------------------------------------

def gave_up(rec: HelperRecord, answer: str) -> Optional[str]:
    """Why a helper that RETURNED should have its work kept, or ``None``:
    it reached its model-call budget, or its answer is short after a lot of
    reading. Only for :data:`KEEP_ON_GIVE_UP` helpers."""
    if rec.subagent not in KEEP_ON_GIVE_UP:
        return None
    if rec.read_chars < KEEP_MIN_READ_CHARS:
        return None
    if rec.budget_reached:
        return ("it reached its model-call budget, so its answer was "
                "written from what it had by then")
    n = len((answer or "").strip())
    if n < SHORT_ANSWER_CHARS:
        return (f"its answer ({n:,} characters) is much shorter than what it "
                f"had read ({rec.read_chars:,} characters)")
    return None


_PAGE_KEYS = re.compile(r"page", re.IGNORECASE)


def _pages_named(rec: HelperRecord) -> str:
    """The page numbers the helper's calls named (any argument whose name
    holds 'page'), as written."""
    seen: list = []
    for s in rec.steps:
        if s["kind"] != "model":
            continue
        for c in s["calls"]:
            args = c["args"] if isinstance(c["args"], dict) else {}
            for k, v in args.items():
                if not _PAGE_KEYS.search(str(k)):
                    continue
                vals = v if isinstance(v, (list, tuple)) else [v]
                for one in vals:
                    if isinstance(one, (int, str)) and str(one).strip() and \
                            str(one) not in seen:
                        seen.append(str(one))
    if not seen:
        return ""
    shown = seen[:30]
    more = len(seen) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more else "")


def saved_files(rec: HelperRecord) -> List[str]:
    """The files the helper saved, as their results name them (each tool
    result that reports a ``saved`` file without an error), in order."""
    out: List[str] = []
    for s in rec.steps:
        if s["kind"] != "result" or s.get("status") == "error":
            continue
        text = s.get("text") or ""
        if '"saved"' not in text:
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        name = data.get("saved") if isinstance(data, dict) else None
        if isinstance(name, str) and name and not data.get("error") \
                and name not in out:
            out.append(name)
    return out


def saved_note(names: List[str]) -> str:
    """What the primary is told about the files a helper saved (BACKLOG
    RI-31: the primary retyped a helper's 29,000 characters of records)."""
    shown = ", ".join(f"'{n}'" for n in names[:10])
    more = len(names) - 10
    return (f"[Files this helper saved, in the working folder: {shown}"
            + (f" and {more} more" if more > 0 else "")
            + ". Give a file to the next tool by its name (write_diggs reads "
            "records from investigations_file / lab_tests_file) -- never "
            "retype what it holds.]")


def summary_paragraph(rec: HelperRecord, why: str) -> str:
    """One paragraph, made in code, of what a stopped helper did."""
    calls = Counter()
    errors = 0
    biggest = (0, "")
    for s in rec.steps:
        if s["kind"] == "model":
            for c in s["calls"]:
                calls[c["name"]] += 1
                n = len(_json(c["args"]))
                if n > biggest[0]:
                    biggest = (n, c["name"])
        elif s.get("status") == "error":
            errors += 1
    n_calls = sum(calls.values())
    parts = [f"The {rec.subagent} helper's work was kept because {why}."]
    if n_calls:
        used = ", ".join(f"{name} x{k}" for name, k in calls.most_common(8))
        parts.append(f"It made {rec.model_calls} model calls and {n_calls} "
                     f"tool calls ({used})"
                     + (f", {errors} of which came back as errors"
                        if errors else "") + ".")
    else:
        parts.append(f"It made {rec.model_calls} model calls and no tool "
                     "calls.")
    pages = _pages_named(rec)
    if pages:
        parts.append(f"Pages its calls named: {pages}.")
    files = saved_files(rec)
    if files:
        parts.append("Files it saved: " + ", ".join(files[:10])
                     + (f" and {len(files) - 10} more" if len(files) > 10
                        else "") + ".")
    parts.append(f"It had read {rec.read_chars:,} characters of tool results"
                 + (f"; its largest call ({biggest[1]}) carried "
                    f"{biggest[0]:,} characters of arguments, which are kept "
                    "too" if biggest[0] >= 2_000 else "") + ".")
    last = " ".join(rec.last_words().split())
    if last:
        if len(last) > 300:
            last = last[:300] + " ..."
        parts.append(f"Its last words: \"{last}\"")
    return " ".join(parts)


def _slug(text: str, words: int = 6, room: int = 40) -> str:
    w = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())[:words]
    s = "-".join(w)[:room].strip("-")
    return s or "helper"


def kept_folder() -> str:
    """The folder kept work goes into: ``kept/`` in the working folder the
    host bound to this turn, else in the temp folder (a library caller
    with no working folder)."""
    base = None
    try:
        from funhouse_agent._fileio import host_output_dir
        base = host_output_dir()
    except Exception:  # noqa: BLE001
        base = None
    if not base:
        base = os.path.join(tempfile.gettempdir(), "geotech_kept")
    return os.path.join(base, KEPT_DIR)


def _relative(path: str) -> str:
    """``path`` as the agent names it: relative to the working folder."""
    try:
        from funhouse_agent._fileio import host_output_dir
        base = host_output_dir()
        if base:
            rel = os.path.relpath(path, base)
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")
    except Exception:  # noqa: BLE001
        pass
    return path


def helper_file_text(rec: HelperRecord, why: str,
                     answer: Optional[str] = None) -> str:
    """The kept file: what the helper was asked, why it was kept, the
    summary, its answer, an index of its calls with where each result
    starts in the file, then every step in full."""
    head = [
        "# What a helper read and wrote before it stopped",
        "",
        f"Helper: {rec.subagent} (task call {rec.task_id or 'unknown'})",
        f"Kept because: {why}.",
        "",
        "## What it was asked",
        rec.brief.strip() or "(no brief recorded)",
        "",
        "## Summary",
        summary_paragraph(rec, why),
    ]
    if answer and answer.strip():
        head += ["", "## Its answer", answer.strip()]
    head += ["", "## Index (character offsets count from the start of this "
             "file; read_text_file with that offset starts there)"]

    body: list = []
    marks: list = []                 # (label, offset in body, length)
    pos = 0

    def put(text: str) -> None:
        nonlocal pos
        body.append(text)
        pos += len(text)

    step = 0
    pending: dict = {}
    for s in rec.steps:
        if s["kind"] == "model":
            step += 1
            put(f"\n\n## Step {step}\n")
            if s["text"].strip():
                put(f"\n### Its own words\n{s['text'].strip()}\n")
            for i, c in enumerate(s["calls"], 1):
                args = _json(c["args"])
                put(f"\n### Call {step}.{i}: {c['name']} "
                    f"(id {c['id'] or '?'})\nArguments: {args}\n")
                pending[c["id"]] = f"{step}.{i} {c['name']}"
        else:
            label = pending.pop(s["id"], None) or f"{s['name']}"
            status = " (an ERROR)" if s.get("status") == "error" else ""
            title = (f"\n### Result of {label}{status}: "
                     f"{len(s['text']):,} characters\n")
            put(title)
            marks.append((label, pos, len(s["text"]), status))
            put(s["text"] + "\n")
    body_text = "".join(body)
    head_text = "\n".join(head) + "\n"

    def index(offset: int) -> str:
        if not marks:
            return "(no tool results)\n"
        return "".join(
            f"- result of {label}{status}: {n:,} characters at offset "
            f"{offset + at:,}\n" for label, at, n, status in marks)

    # The index holds offsets into the file that contains it: settle them.
    offset = len(head_text)
    for _ in range(5):
        new = len(head_text) + len(index(offset))
        if new == offset:
            break
        offset = new
    return head_text + index(offset) + body_text


def write_helper_file(rec: HelperRecord, why: str,
                      answer: Optional[str] = None) -> Optional[str]:
    """Write the kept file; its absolute path, or ``None`` if it could not
    be written."""
    try:
        folder = kept_folder()
        os.makedirs(folder, exist_ok=True)
        tail = re.sub(r"[^A-Za-z0-9]", "", rec.task_id)[-6:] or \
            str(int(time.time()))[-6:]
        path = os.path.join(folder,
                            f"helper_{_slug(rec.brief)}_{tail}.md")
        text = helper_file_text(rec, why, answer)
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        scope = current_turn()
        if scope is not None:
            scope.note_kept_file(path)
        return path
    except Exception:  # noqa: BLE001 - keeping never fails the call
        return None


def kept_note(rec: HelperRecord, path: str, why: str) -> str:
    """What the primary is told about a kept file, after the helper's own
    answer."""
    return (f"[Kept: {why}. Everything this helper read and wrote "
            f"({rec.n_results} tool results, {rec.read_chars:,} characters) "
            f"is in the file '{_relative(path)}' in the working folder. Read "
            "it with read_text_file -- its index gives the offset of each "
            "result -- or name it in the next helper's brief, instead of "
            "reading the same pages again.]\n"
            f"Summary: {summary_paragraph(rec, why)}")


# ---------------------------------------------------------------------------
# Helpers at once (BACKLOG TA-22)
# ---------------------------------------------------------------------------

#: Most ``task`` helpers that run at once in one conversation; the rest wait
#: for a free place, in the order they were asked for. ``0`` = no cap. The
#: field session started five within 12 ms on one key; the vision slots,
#: the document lock and the rate limit then served them in turn anyway.
MAX_HELPERS_ENV = "GEOTECH_MAX_HELPERS"
DEFAULT_MAX_HELPERS = 2


def max_helpers() -> int:
    raw = (os.environ.get(MAX_HELPERS_ENV) or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_HELPERS
    except ValueError:
        return DEFAULT_MAX_HELPERS


class _HelperPlaces:
    """At most ``limit`` helpers at once, served first come first served."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._cond = threading.Condition()
        self._running = 0
        self._queue: list = []

    def acquire(self) -> float:
        """Wait for a place; the seconds waited."""
        t0 = time.monotonic()
        with self._cond:
            me = object()
            self._queue.append(me)
            try:
                while not (self._running < self.limit
                           and self._queue[0] is me):
                    self._cond.wait()
            except BaseException:
                self._queue.remove(me)
                self._cond.notify_all()
                raise
            self._queue.pop(0)
            self._running += 1
            self._cond.notify_all()
        return time.monotonic() - t0

    def release(self) -> None:
        with self._cond:
            self._running = max(0, self._running - 1)
            self._cond.notify_all()


def _event(name: str, data: dict) -> None:
    """A custom event for the activity log (``webapp.activity_log``)."""
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data)
    except Exception:  # noqa: BLE001 - logging never fails a call
        pass


def _conversation() -> str:
    try:
        from funhouse_agent.deep.vision_engine import conversation_key
        return conversation_key()
    except Exception:  # noqa: BLE001
        return ""


class KeepWork(AgentMiddleware):
    """On the primary: keep a stopped helper's work (see the module
    docstring), cap the helpers running at once, and keep the full text of
    very large tool results on the turn scope. Put it just inside
    :class:`~funhouse_agent.deep.tool_guards.ToolErrorGuard`, so a helper's
    error reaches it first."""

    def __init__(self, max_at_once: Optional[int] = None):
        super().__init__()
        self._limit = max_at_once
        self._places: dict = {}
        self._lock = threading.Lock()

    def _places_for(self) -> Optional[_HelperPlaces]:
        limit = max_helpers() if self._limit is None else int(self._limit)
        if limit <= 0:
            return None
        key = _conversation()
        with self._lock:
            places = self._places.get(key)
            if places is None or places.limit != limit:
                places = self._places[key] = _HelperPlaces(limit)
            return places

    # -- results ----------------------------------------------------------
    @staticmethod
    def _note_full(request, result) -> None:
        scope = current_turn()
        if scope is None:
            return
        try:
            call = getattr(request, "tool_call", None) or {}
            for msg in _result_messages(result):
                text = _text(msg.content)
                if len(text) >= FULL_RESULT_MIN_CHARS:
                    scope.note_full_result(
                        call.get("id") or getattr(msg, "tool_call_id", None),
                        text)
        except Exception:  # noqa: BLE001
            pass

    def _stopped(self, request, rec: HelperRecord, exc: BaseException):
        """A helper that raised: keep its work and say so, as the tool
        error the primary reads."""
        from funhouse_agent.error_text import error_line
        call = getattr(request, "tool_call", None) or {}
        out = {"error": f"task failed: the {rec.subagent} helper stopped "
                        f"on an error: {error_line(exc)}"}
        if rec.read_chars >= KEEP_MIN_READ_CHARS or rec.arg_chars >= 2_000:
            why = f"it stopped on an error ({error_line(exc, 160)})"
            path = write_helper_file(rec, why)
            if path:
                out["kept"] = _relative(path)
                out["summary"] = summary_paragraph(rec, why)
                out["hint"] = (
                    "What the helper had read and written is kept in that "
                    "file (its index gives the offset of each result). Read "
                    "it with read_text_file, or name it in a new helper's "
                    "brief, instead of reading the same pages again. If the "
                    "error was a busy model, wait before asking again.")
                _event("helper_kept", {"task_id": rec.task_id,
                                       "subagent": rec.subagent,
                                       "path": out["kept"], "why": why,
                                       "read_chars": rec.read_chars})
        if "kept" not in out:
            out["summary"] = summary_paragraph(
                rec, "it stopped on an error before it had read much")
            out["hint"] = ("Nothing it read was worth keeping. Try again "
                           "once if the cause looks passing (wait first for "
                           "a busy model), or another way; tell the user "
                           "plainly what could not be done.")
        try:
            from funhouse_agent.error_text import budget_exhausted, BUDGET_HINT
            if budget_exhausted(exc) is not None:
                out["hint"] = BUDGET_HINT
        except Exception:  # noqa: BLE001
            pass
        return ToolMessage(content=json.dumps(out, ensure_ascii=False),
                           name=TASK_TOOL, tool_call_id=call.get("id") or "",
                           status="error")

    def _finished(self, rec: HelperRecord, result):
        """A helper that returned: keep its work when it gave up."""
        msgs = _result_messages(result)
        answer = "\n".join(_text(m.content) for m in msgs)
        why = gave_up(rec, answer)
        if why is None:
            # A helper that saved files: their names reach the primary in
            # code, whether or not its reply named them (RI-31).
            files = saved_files(rec)
            return _with_note(result, saved_note(files)) if files else result
        path = write_helper_file(rec, why, answer)
        if not path:
            return result
        _event("helper_kept", {"task_id": rec.task_id,
                               "subagent": rec.subagent,
                               "path": _relative(path), "why": why,
                               "read_chars": rec.read_chars})
        note = kept_note(rec, path, why)
        return _with_note(result, note)

    # -- the wrapper ------------------------------------------------------
    def _begin(self, request):
        call = getattr(request, "tool_call", None) or {}
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        rec = HelperRecord(call.get("id") or "", args.get("subagent_type"),
                           args.get("description"))
        places = self._places_for()
        waited = places.acquire() if places is not None else 0.0
        if waited >= 0.5:
            _event("helper_queued", {"task_id": rec.task_id,
                                     "subagent": rec.subagent,
                                     "queued_s": round(waited, 3),
                                     "max_at_once": places.limit})
        token = _HELPER.set(rec)
        return rec, places, token

    @staticmethod
    def _end(places, token) -> None:
        if places is not None:
            places.release()
        try:
            _HELPER.reset(token)
        except (ValueError, RuntimeError):
            pass

    def wrap_tool_call(self, request, handler):
        call = getattr(request, "tool_call", None) or {}
        if call.get("name") != TASK_TOOL:
            result = handler(request)
            self._note_full(request, result)
            return result
        rec, places, token = self._begin(request)
        try:
            try:
                result = handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001 - kept and reported
                rec.error = exc
                return self._stopped(request, rec, exc)
            result = self._finished(rec, result)
            self._note_full(request, result)
            return result
        finally:
            self._end(places, token)

    async def awrap_tool_call(self, request, handler):
        call = getattr(request, "tool_call", None) or {}
        if call.get("name") != TASK_TOOL:
            result = await handler(request)
            self._note_full(request, result)
            return result
        rec, places, token = self._begin(request)
        try:
            try:
                result = await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001
                rec.error = exc
                return self._stopped(request, rec, exc)
            result = self._finished(rec, result)
            self._note_full(request, result)
            return result
        finally:
            self._end(places, token)


def _with_note(result, note: str):
    """``result`` (a ToolMessage or a Command carrying one) with ``note``
    after its text."""
    def add(msg):
        return msg.model_copy(update={
            "content": (_text(msg.content).rstrip() + "\n\n" + note)})
    if isinstance(result, ToolMessage):
        return add(result)
    update = getattr(result, "update", None)
    if isinstance(update, dict) and isinstance(update.get("messages"),
                                               (list, tuple)):
        msgs = [add(m) if isinstance(m, ToolMessage) else m
                for m in update["messages"]]
        try:
            import dataclasses
            return dataclasses.replace(result,
                                       update={**update, "messages": msgs})
        except Exception:  # noqa: BLE001 - the answer goes back unchanged
            return result
    return result


__all__ = ["TurnScope", "turn_scope", "current_turn", "HelperRecord",
           "current_helper", "HelperRecorder", "KeepWork",
           "note_budget_reached", "gave_up", "summary_paragraph",
           "saved_files", "saved_note",
           "helper_file_text", "write_helper_file", "kept_note",
           "kept_folder", "max_helpers", "KEPT_DIR", "KEEP_ON_GIVE_UP",
           "KEEP_MIN_READ_CHARS", "SHORT_ANSWER_CHARS",
           "FULL_RESULT_MIN_CHARS", "MAX_HELPERS_ENV", "DEFAULT_MAX_HELPERS",
           "BUSY_TURN_MAX_ENV", "DEFAULT_BUSY_TURN_MAX_S"]
