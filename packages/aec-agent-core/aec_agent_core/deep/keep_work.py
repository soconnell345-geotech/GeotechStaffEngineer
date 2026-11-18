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
  The boring-log and lab records such a helper held and had not saved --
  in its answer or one of its calls, found by their shape
  (:mod:`funhouse_agent.deep.held_records`) -- go to
  ``kept/helper_<what>_<id>_records.json`` beside it, named in the same
  note, so the primary passes that file to ``write_diggs`` instead of
  retyping them (BACKLOG RI-31).
  It also caps the helpers that run at once in one conversation
  (:func:`max_helpers`; BACKLOG TA-22).
  A page can name its READING helpers (``KeepWork(readers=...)``; BACKLOG
  DR-46). Each of their returns keeps the whole record in ``kept/`` too, as
  ``reader_<what>_<id>.md`` (gave up: ``helper_...`` as above), names the
  file in the result, and adds the open items the helper's own tools
  flagged (:func:`open_items`: what was not read), whether or not its
  answer named them. Only a helper that stopped or gave up is noted on the
  turn (:attr:`TurnScope.kept_files`).
  Document Review names its reader; the geotech page names none, so its
  helpers get what they always got. The minimal build's helper is not a
  ``task`` call this wraps: it adds the same open items to its own answer,
  from a record made of its messages (:func:`record_of_messages`; DR-50).

Nothing here changes a tool's result except a ``task`` result that kept a
file, whose helper saved files (their names are added, BACKLOG RI-31) or
whose helper is a named reader (DR-46), and nothing here ever fails a call.
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

from aec_agent_core.deep.tool_guards import GraphBubbleUp, _ai_of, _text

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


class TurnStopped(Exception):
    """Raised before a model call, a wait or a pass of a turn the user has
    asked to stop (BACKLOG TA-37). An ``Exception``, so a helper that meets
    it has its work kept like any helper that stopped
    (:class:`KeepWork`), and the primary's next model call raises it out of
    the run, where the host keeps the turn's work as it keeps an
    interrupted turn's."""

    def __init__(self, message: str = "the turn was stopped by the user"):
        super().__init__(message)


#: What a look refused at the turn's image ceiling says, in its tool's
#: result (BACKLOG TA-43).
IMAGE_CEILING_ERROR = (
    "this turn reached its ceiling of {limit} image calls, so this image "
    "was NOT read. Finish with what you have: answer from what was already "
    "read, and say what was left unread.")

#: What each look's result says while a ceiling is set (TA-43).
IMAGE_CALLS_NOTE = ("this look made {calls} image call{s}; this turn has "
                    "made {used} of its ceiling of {limit}")
IMAGE_REFUSED_NOTE = "; {refused} more were refused at the ceiling"
IMAGE_NEAR_NOTE = (". The ceiling is near: past it a look is refused, so "
                   "look only where it matters and finish with what you "
                   "have")
#: The same, once the ceiling is met: a note further down the same result
#: may still suggest a zoom or tiles, and that look would be refused.
IMAGE_MET_NOTE = (". The ceiling is met: any further look this turn is "
                  "refused, whatever a note in this result suggests, so "
                  "finish with what you have and say what was left unread")

#: The running turn's status line while a ceiling is set (TA-43).
IMAGE_STATUS = "image calls this turn: {used} of {limit}"

#: The line under the answer of a turn whose ceiling refused a look (TA-43).
IMAGE_CEILING_TOLD = (
    "_This turn reached its ceiling of {limit} image calls: {refused} "
    "image call{s} after that {were} not made, so the answer rests on what "
    "was read before. Ask to continue for more._")


class ImageBudgetSpent(Exception):
    """Raised before an image call past the turn's ceiling
    (``GEOTECH_TURN_MAX_IMAGE_CALLS``, BACKLOG TA-43): the image is not
    sent, and the tool reports :data:`IMAGE_CEILING_ERROR`. The turn goes
    on."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        super().__init__(IMAGE_CEILING_ERROR.format(limit=self.limit))


def _image_limit() -> int:
    try:
        from aec_agent_core.deep.limits import turn_max_image_calls
        return turn_max_image_calls()
    except Exception:  # noqa: BLE001 - no ceiling, as before
        return 0


class TurnScope:
    """What one turn shares across its agents (see the module docstring).

    ``stop_event`` (a :class:`threading.Event`, optional): set by the host
    when the user asks to stop the turn; every agent's next model call,
    every wait for a busy model or a helper place, and the host's next pass
    then raise :class:`TurnStopped` (BACKLOG TA-37). ``None`` (the default)
    = no stop, and nothing here waits differently.

    ``whole_turn_budget``: the primary agent's model-call budget is counted
    over the whole turn, all its passes, with the continuations of a reply
    cut off at the output limit counted too (BACKLOG TA-38). Off by default,
    so a host that does not ask keeps the per-pass count."""

    def __init__(self, busy_limit_s: Optional[float] = None,
                 stop_event: Optional[threading.Event] = None,
                 whole_turn_budget: bool = False):
        self.started = time.time()
        self.stop_event = stop_event
        self.whole_turn_budget = bool(whole_turn_budget)
        # the primary's model calls this turn, all passes (whole_turn_budget)
        self.primary_model_calls = 0
        # continuation calls of the primary's cut-off replies (whole turn)
        self.primary_extra_calls = 0
        # the primary's budget, noted by its budget middleware
        self.primary_budget: Optional[int] = None
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
        # the primary agent's model-call budget forced its last call
        # (note_budget_reached): the turn's answer is final
        self.budget_reached = False
        # the images sent to the vision model for a separate reading this
        # turn (the vision engine's analyze_image), every agent and helper
        # together; those refused at the ceiling; the ceiling
        # (GEOTECH_TURN_MAX_IMAGE_CALLS, 0 = none) (BACKLOG TA-43). Images
        # shown inline to the agent's own model (inline images, show_me)
        # are not counted, and the ceiling does not apply to them.
        self.image_calls = 0
        self.image_refused = 0
        self.image_limit = _image_limit()
        # what the lean framework prompt did at the primary's model calls
        # (GEOTECH_REVIEW_LEAN_PROMPT, BACKLOG DR-29; note_lean_prompt):
        # None when the switch is off
        self.lean_prompt: Optional[dict] = None
        # the summarizers whose summary failed this turn, each with the
        # helper run it failed in (None = not in a helper), and the model
        # calls that were over the summary threshold after that and were
        # not summarized (aec_agent_core.deep.summary_guard; BACKLOG TA-24)
        self.summaries_failed: dict = {}
        self.summaries_skipped = 0
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

    @property
    def stop_requested(self) -> bool:
        """The user asked to stop this turn (TA-37)."""
        ev = self.stop_event
        return bool(ev is not None and ev.is_set())

    def request_stop(self) -> None:
        """Ask the turn to stop (a scope made without a ``stop_event`` gets
        one)."""
        with self._lock:
            if self.stop_event is None:
                self.stop_event = threading.Event()
        self.stop_event.set()

    def spend_busy(self, seconds: float) -> bool:
        """Whether the turn may still spend ``seconds`` on a busy model; if
        so, they are counted. ``busy_limit_s <= 0`` means no bound. A turn
        the user asked to stop may not wait at all (TA-37)."""
        if self.stop_requested:
            return False
        with self._lock:
            if self.busy_limit_s > 0 and \
                    self.busy_spent_s + seconds > self.busy_limit_s:
                return False
            self.busy_spent_s += max(0.0, float(seconds))
            self.busy_events += 1
            return True

    def note_summary_failed(self, summarizer, helper=None) -> None:
        """``summarizer``'s summary failed in this turn, in the helper run
        ``helper`` (``None`` = the main agent, or no helper record): it is
        not tried again there this turn (TA-24)."""
        with self._lock:
            # the objects are kept, so their ids stay theirs for the turn
            self.summaries_failed[(id(summarizer), id(helper))] = (
                summarizer, helper)

    def summary_failed(self, summarizer, helper=None) -> bool:
        """Whether ``summarizer``'s summary already failed this turn in
        the helper run ``helper`` (TA-24)."""
        with self._lock:
            return (id(summarizer), id(helper)) in self.summaries_failed

    def note_summary_skipped(self) -> None:
        """A model call over the summary threshold was not summarized,
        because the summary failed earlier this turn (TA-24)."""
        with self._lock:
            self.summaries_skipped += 1

    def take_image_call(self, look: Optional[dict] = None) -> None:
        """Count one image call on the turn (and on ``look``, the tool call
        it is made for), or raise :class:`ImageBudgetSpent` when the turn's
        ceiling is reached (TA-43)."""
        with self._lock:
            refused = 0 < self.image_limit <= self.image_calls
            field = "refused" if refused else "calls"
            if refused:
                self.image_refused += 1
            else:
                self.image_calls += 1
            if look is not None:
                look[field] = look.get(field, 0) + 1
        if refused:
            raise ImageBudgetSpent(self.image_limit)


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


def stop_requested() -> bool:
    """Whether the user asked to stop the running turn (TA-37); ``False``
    outside a turn or where the turn has no stop."""
    scope = _TURN.get()
    return bool(scope is not None and scope.stop_requested)


def check_stop() -> None:
    """Raise :class:`TurnStopped` when the user asked to stop the running
    turn; nothing otherwise (TA-37). Called before each model call of every
    agent and before each wait."""
    if stop_requested():
        raise TurnStopped()


# ---------------------------------------------------------------------------
# The turn's image calls (BACKLOG TA-43)
# ---------------------------------------------------------------------------

#: The image calls of the tool call running here (:func:`counting_look`);
#: the tiles' threads run in copies of its context and share the dict.
_LOOK: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar(
    "geotech_look_calls", default=None)


def take_image_call() -> None:
    """Called once before each image is sent to the vision model for a
    separate reading (a whole page, a tile, a zoom, a candidate checked):
    counted on the running turn, or :class:`ImageBudgetSpent` raised past
    the turn's ceiling. Outside a turn nothing is counted. An image shown
    inline to the agent's own model does not come here."""
    scope = _TURN.get()
    if scope is not None:
        scope.take_image_call(_LOOK.get())


@contextmanager
def counting_look() -> Iterator[dict]:
    """``with counting_look() as look:`` -- ``look["calls"]`` and
    ``look["refused"]``, the image calls one tool call made and had
    refused."""
    look = {"calls": 0, "refused": 0}
    token = _LOOK.set(look)
    try:
        yield look
    finally:
        try:
            _LOOK.reset(token)
        except (ValueError, RuntimeError):
            pass


def _ceiling() -> Optional[TurnScope]:
    scope = _TURN.get()
    return scope if scope is not None and scope.image_limit > 0 else None


def with_image_calls(raw: str, look: Optional[dict]) -> str:
    """A tool's result with its image calls against the turn's ceiling, as
    an ``image_calls`` key first (or a first line, for a result that is not
    a JSON object), warning from :data:`limits.IMAGE_WARN_SHARE` of the
    ceiling. ``raw`` itself, unchanged, when no ceiling is set or the call
    made no image call."""
    scope = _ceiling()
    if scope is None or not look or not look.get("calls"):
        return raw
    try:
        from aec_agent_core.deep.limits import IMAGE_WARN_SHARE
        calls, used, limit = look["calls"], scope.image_calls, \
            scope.image_limit
        note = IMAGE_CALLS_NOTE.format(calls=calls,
                                       s="" if calls == 1 else "s",
                                       used=used, limit=limit)
        if look.get("refused"):
            note += IMAGE_REFUSED_NOTE.format(refused=look["refused"])
        if used >= limit:
            note += IMAGE_MET_NOTE
        elif used >= IMAGE_WARN_SHARE * limit:
            note += IMAGE_NEAR_NOTE
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
        if isinstance(data, dict):
            return json.dumps({"image_calls": note, **data})
        return f"[image calls] {note}.\n\n{raw}"
    except Exception:  # noqa: BLE001 - a note never fails a tool
        return raw


def image_calls_status(shown: list) -> Optional[str]:
    """The running turn's status line, ``image calls this turn: k of N``,
    when a ceiling is set and the count moved since ``shown[0]`` (which is
    then updated); ``None`` otherwise."""
    scope = _ceiling()
    if scope is None or scope.image_calls == shown[0]:
        return None
    shown[0] = scope.image_calls
    return IMAGE_STATUS.format(used=scope.image_calls,
                               limit=scope.image_limit)


def image_ceiling_met() -> bool:
    """Whether the running turn has a ceiling on its image calls and has
    reached it, so that any further look this turn would be refused.
    ``False`` with no ceiling set and outside a turn."""
    scope = _ceiling()
    return scope is not None and scope.image_calls >= scope.image_limit


def image_ceiling_told(scope) -> Optional[str]:
    """The line under the answer of a turn whose ceiling refused a look,
    once; ``None`` for any other turn."""
    refused = getattr(scope, "image_refused", 0)
    if not refused or getattr(scope, "image_limit", 0) <= 0:
        return None
    return IMAGE_CEILING_TOLD.format(limit=scope.image_limit,
                                     refused=refused,
                                     s="" if refused == 1 else "s",
                                     were="was" if refused == 1 else "were")


def image_call_facts(scope) -> dict:
    """The turn's image calls for its record: ``image_calls`` always (a
    scope was open), ``image_limit`` when a ceiling was set, and
    ``image_calls_refused`` when it refused any."""
    if scope is None:
        return {}
    out = {"image_calls": int(getattr(scope, "image_calls", 0))}
    if getattr(scope, "image_limit", 0) > 0:
        out["image_limit"] = int(scope.image_limit)
    if getattr(scope, "image_refused", 0):
        out["image_calls_refused"] = int(scope.image_refused)
    return out


def lean_prompt_facts(scope) -> dict:
    """What the lean framework prompt did this turn, for its record
    (``lean_prompt``; BACKLOG DR-29); ``{}`` when it did not run (its
    switch off, or no scope)."""
    value = getattr(scope, "lean_prompt", None) if scope is not None \
        else None
    return {"lean_prompt": dict(value)} if isinstance(value, dict) else {}


def summary_facts(scope) -> dict:
    """For the turn's record: ``summaries_skipped``, the model calls that
    were over the summary threshold and were not summarized because the
    summary had failed earlier in the turn; nothing when there were none
    (:mod:`aec_agent_core.deep.summary_guard`, BACKLOG TA-24)."""
    skipped = int(getattr(scope, "summaries_skipped", 0) or 0)
    return {"summaries_skipped": skipped} if skipped else {}


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
        # its budget and the continuation calls of its cut-off replies,
        # counted in a turn whose budget is the whole turn's (TA-38)
        self.budget: Optional[int] = None
        self.extra_model_calls = 0
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
    (:class:`funhouse_agent.deep.limits.ModelCallBudgetMiddleware`).

    A helper's budget is noted on its record; the primary's on the turn, so
    the host does not start another pass with a fresh budget."""
    rec = _HELPER.get()
    if rec is not None:
        rec.budget_reached = True
        return
    scope = _TURN.get()
    if scope is not None:
        scope.budget_reached = True


def note_lean_prompt(outcome: dict) -> None:
    """What the lean framework prompt did at one of the primary's model
    calls (:class:`aec_agent_core.deep.lean_prompt.LeanFrameworkPrompt`,
    BACKLOG DR-29), kept on the running turn for its record. A call where
    it was not applied is kept over any later call; nothing is kept in a
    helper or outside a turn."""
    if _HELPER.get() is not None or not isinstance(outcome, dict):
        return
    scope = _TURN.get()
    if scope is None:
        return
    with scope._lock:
        kept = getattr(scope, "lean_prompt", None)
        if isinstance(kept, dict) and not kept.get("applied"):
            return
        scope.lean_prompt = dict(outcome)


# ---------------------------------------------------------------------------
# A budget counted over the whole turn (BACKLOG TA-38)
# ---------------------------------------------------------------------------

def _whole_turn() -> Optional[TurnScope]:
    scope = _TURN.get()
    return scope if scope is not None and scope.whole_turn_budget else None


def note_budget(limit: Optional[int]) -> None:
    """A model-call budget says its limit before a call: on the running
    helper's record, or on the turn for the primary. Only in a turn whose
    budget is the whole turn's."""
    if _whole_turn() is None or not limit:
        return
    rec = _HELPER.get()
    if rec is not None:
        rec.budget = int(limit)
    else:
        _TURN.get().primary_budget = int(limit)


def note_primary_model_call() -> None:
    """The primary agent made a model call (its budget middleware counts it
    once per call, after the call); counted on the turn so a later pass
    goes on from it."""
    scope = _whole_turn()
    if scope is None or _HELPER.get() is not None:
        return
    with scope._lock:
        scope.primary_model_calls += 1


def turn_model_calls() -> Optional[int]:
    """The primary's model calls over the whole turn so far, or ``None``
    (a helper, or a turn that does not count over its passes)."""
    scope = _whole_turn()
    if scope is None or _HELPER.get() is not None:
        return None
    return scope.primary_model_calls


def extra_model_calls() -> int:
    """Continuation calls of cut-off replies counted against the running
    agent's budget (0 where none are counted)."""
    if _whole_turn() is None:
        return 0
    rec = _HELPER.get()
    if rec is not None:
        return rec.extra_model_calls
    return _TURN.get().primary_extra_calls


def take_extra_model_call() -> bool:
    """Whether the running agent may make one more call to continue a reply
    cut off at the output limit; if so, it is counted against its budget.
    One call is always left for the forced last reply: a continuation is
    refused when, after it, the next call would not fit. ``True`` where no
    budget is known (nothing counted)."""
    scope = _whole_turn()
    if scope is None:
        return True
    rec = _HELPER.get()
    if rec is not None:
        if rec.budget is None:
            return True
        if rec.budget_reached:
            return False
        # this call is not on the record yet (HelperRecorder records it
        # once it returns): done + this one + extra + next >= budget
        if rec.model_calls + 1 + rec.extra_model_calls + 1 >= rec.budget:
            return False
        rec.extra_model_calls += 1
        return True
    if scope.primary_budget is None:
        return True
    with scope._lock:
        if scope.budget_reached:
            return False
        if (scope.primary_model_calls + 1 + scope.primary_extra_calls + 1
                >= scope.primary_budget):
            return False
        scope.primary_extra_calls += 1
        return True


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(obj)


def _result_messages(result) -> list:
    from aec_agent_core.deep.delivered_log import _messages_of
    return _messages_of(result)


class HelperRecorder(AgentMiddleware):
    """Record a sub-agent's model replies and tool results into the running
    helper's :class:`HelperRecord` (see the module docstring). Put it just
    inside the retries, so a retried call is recorded once, and outside the
    tool guards, so a result is recorded as the helper received it."""

    def wrap_model_call(self, request, handler):
        rec = _HELPER.get()
        check_stop()                       # TA-37: a stopped turn
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
        check_stop()
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


def record_of_messages(messages, subagent: str = "",
                       brief: str = "") -> HelperRecord:
    """A :class:`HelperRecord` made after the fact from a finished helper's
    own conversation: its model replies with their tool calls and its tool
    results, in order (the brief and any other message are skipped).

    For a helper that does not run as a ``task`` call :class:`KeepWork`
    wraps, so nothing recorded it as it went: the minimal build's helper
    (BACKLOG DR-50). Its final state holds each result as the helper
    received it (the inline images are added to a request, never to the
    state), so :func:`open_items` reads this record as it reads one
    :class:`HelperRecorder` kept. Setting no running helper leaves
    everything that reads :func:`current_helper` as it was."""
    rec = HelperRecord("", subagent, brief)
    for m in messages or ():
        if isinstance(m, ToolMessage):
            rec.add_result(getattr(m, "name", None),
                           getattr(m, "tool_call_id", None) or "",
                           _text(m.content), getattr(m, "status", None))
        elif getattr(m, "type", "") == "ai":
            rec.add_model(m)
    return rec


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


#: The hook through which an app words what follows the file names in
#: :func:`saved_note` (BACKLOG DR-42): ``hint(names) -> str``. Unfilled,
#: failing or empty, the note keeps :data:`DEFAULT_SAVED_FILES_HINT`.
SAVED_FILES_HINT_HOOK = "keep_work.saved_files_hint"

#: What follows the file names when no app words it: the geotech page's
#: wording since RI-31, unchanged.
DEFAULT_SAVED_FILES_HINT = (
    "Give a file to the next tool by its name (write_diggs reads "
    "records from investigations_file / lab_tests_file) -- never "
    "retype what it holds.")


def _saved_files_hint(names: List[str]) -> str:
    try:
        from aec_agent_core.hooks import get_hook
        hook = get_hook(SAVED_FILES_HINT_HOOK)
        hint = hook(list(names)) if hook is not None else None
    except Exception:  # noqa: BLE001 - a hook never breaks the note
        hint = None
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return DEFAULT_SAVED_FILES_HINT


def saved_note(names: List[str]) -> str:
    """What the primary is told about the files a helper saved (BACKLOG
    RI-31: the primary retyped a helper's 29,000 characters of records).
    The sentence after the names is the app's (:data:`SAVED_FILES_HINT_HOOK`;
    BACKLOG DR-42: the default names a tool only the geotech page has)."""
    shown = ", ".join(f"'{n}'" for n in names[:10])
    more = len(names) - 10
    return (f"[Files this helper saved, in the working folder: {shown}"
            + (f" and {more} more" if more > 0 else "")
            + ". " + _saved_files_hint(names) + "]")


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
    with no working folder). On a shared host with no folder bound it
    raises :class:`~funhouse_agent._fileio.UnboundWorkingDir`, and nothing
    is kept: the temp folder is every user's (the callers catch it)."""
    from aec_agent_core._fileio import UnboundWorkingDir, host_output_dir
    base = None
    try:
        base = host_output_dir()
    except UnboundWorkingDir:
        raise
    except Exception:  # noqa: BLE001
        base = None
    if not base:
        base = os.path.join(tempfile.gettempdir(), "geotech_kept")
    return os.path.join(base, KEPT_DIR)


def _relative(path: str) -> str:
    """``path`` as the agent names it: relative to the working folder."""
    try:
        from aec_agent_core._fileio import host_output_dir
        base = host_output_dir()
        if base:
            rel = os.path.relpath(path, base)
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")
    except Exception:  # noqa: BLE001
        pass
    return path


def _stem(rec: HelperRecord, kind: str = "helper") -> str:
    """The kept files' common name for this helper, ``<kind>_<what>_<id>``
    (made once, so its ``.md`` and its records file match). ``kind`` is
    ``helper`` (it stopped or gave up) or :data:`READER_STEM`."""
    stem = getattr(rec, "_kept_stem", None)
    if not stem:
        tail = re.sub(r"[^A-Za-z0-9]", "", rec.task_id)[-6:] or \
            str(int(time.time()))[-6:]
        stem = f"{kind}_{_slug(rec.brief)}_{tail}"
        rec._kept_stem = stem
    return stem


def helper_file_text(rec: HelperRecord, why: str,
                     answer: Optional[str] = None,
                     records: Optional[dict] = None,
                     items: Optional[List[str]] = None,
                     title: Optional[str] = None) -> str:
    """The kept file: what the helper was asked, why it was kept, the
    summary, its answer, the open items its tools flagged (``items``,
    :func:`open_items`, all of them), an index of its calls with where each
    result starts in the file, then every step in full. ``records``: the
    records file saved beside it (:func:`save_held_records`), named at the
    top. ``title``: the heading (a reader kept every time did not stop)."""
    head = [
        f"# {title or 'What a helper read and wrote before it stopped'}",
        "",
        f"Helper: {rec.subagent} (task call {rec.task_id or 'unknown'})",
        f"Kept because: {why}.",
    ]
    if records:
        from aec_agent_core.deep.held_records import describe
        head.append(f"Records it held ({describe(records)}) are saved as "
                    f"'{records['name']}': pass that file to write_diggs.")
    head += [
        "",
        "## What it was asked",
        rec.brief.strip() or "(no brief recorded)",
        "",
        "## Summary",
        summary_paragraph(rec, why),
    ]
    if answer and answer.strip():
        head += ["", "## Its answer", answer.strip()]
    if items:
        head += ["", f"## {OPEN_ITEMS_HEADING}", *items]
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


def _write_kept(path: str, text: str, note: bool = True) -> str:
    """Write a kept file; ``note``: count it on the turn as the work of a
    helper that stopped (a reader's record kept every time is not)."""
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
    scope = current_turn()
    if scope is not None and note:
        scope.note_kept_file(path)
    return path


def write_helper_file(rec: HelperRecord, why: str,
                      answer: Optional[str] = None,
                      records: Optional[dict] = None,
                      items: Optional[List[str]] = None,
                      title: Optional[str] = None,
                      reader: bool = False) -> Optional[str]:
    """Write the kept file; its absolute path, or ``None`` if it could not
    be written. ``reader``: a reader's record kept every time
    (:data:`READER_STEM`, not noted on the turn)."""
    try:
        folder = kept_folder()
        os.makedirs(folder, exist_ok=True)
        stem = _stem(rec, READER_STEM if reader else "helper")
        path = os.path.join(folder, f"{stem}.md")
        return _write_kept(path, helper_file_text(rec, why, answer, records,
                                                  items, title),
                           note=not reader)
    except Exception:  # noqa: BLE001 - keeping never fails the call
        return None


def save_held_records(rec: HelperRecord,
                      answer: Optional[str] = None) -> Optional[dict]:
    """Save the records a stopped helper held and had not saved (BACKLOG
    RI-31; :func:`funhouse_agent.deep.held_records.held_records`) as
    ``kept/<stem>_records.json``, in the shape write_diggs reads.
    ``{"path", "name", "investigations", "lab_tests", "source"}``, or
    ``None`` when it held none or the file could not be written."""
    try:
        from aec_agent_core.deep.held_records import (file_content,
                                                      held_records)
        held = held_records(rec, answer)
        if not held:
            return None
        folder = kept_folder()
        os.makedirs(folder, exist_ok=True)
        path = _write_kept(os.path.join(folder, f"{_stem(rec)}_records.json"),
                           file_content(held))
        out = {**held, "path": path, "name": _relative(path)}
        _event("helper_records_kept", {
            "task_id": rec.task_id, "subagent": rec.subagent,
            "path": out["name"], "source": held.get("source"),
            "investigations": len(held.get("investigations") or ()),
            "lab_tests": len(held.get("lab_tests") or ())})
        return out
    except Exception:  # noqa: BLE001 - keeping never fails the call
        return None


def kept_note(rec: HelperRecord, path: str, why: str,
              records: Optional[dict] = None) -> str:
    """What the primary is told about a kept file, after the helper's own
    answer (and the records file saved beside it, when there is one)."""
    note = (f"[Kept: {why}. Everything this helper read and wrote "
            f"({rec.n_results} tool results, {rec.read_chars:,} characters) "
            f"is in the file '{_relative(path)}' in the working folder. Read "
            "it with read_text_file -- its index gives the offset of each "
            "result -- or name it in the next helper's brief, instead of "
            "reading the same pages again.]\n"
            f"Summary: {summary_paragraph(rec, why)}")
    if records:
        note += "\n" + _records_note(records)
    return note


def _records_note(records: dict) -> str:
    """What the primary is told about a records file; a plainer line if
    the full note cannot be made (keeping never fails the call)."""
    try:
        from aec_agent_core.deep.held_records import records_note
        return records_note(records["name"], records)
    except Exception:  # noqa: BLE001
        return (f"[Records this helper held are saved as "
                f"'{records.get('name')}' in the working folder: pass that "
                "file to write_diggs -- never retype them.]")


# ---------------------------------------------------------------------------
# A reading helper's boundary (BACKLOG DR-46)
# ---------------------------------------------------------------------------
#
# Trace review of a 40-sheet whole-set review (2026-11-13): the main agent
# split the set into reading helpers by page range; the plans and the
# schedules they must agree with fell to different helpers, and what each
# read (the tags, the rows) and each look that came back cut off or partly
# unread died with it. The reader's prompt now asks for what must be
# compared with other pages; this keeps, in code, what the prompt cannot
# guarantee: the whole record, named, and the tools' own flags.

#: Why a named reader's record is kept when it did not give up.
READER_KEPT_WHY = ("it is a reading helper: its whole record is kept every "
                   "time, so what it read stays open to the agent that asked")

#: The kept file's heading for such a reader.
READER_KEPT_TITLE = "What a reading helper read and wrote"

#: The heading of the open items, in the kept file and in the result.
OPEN_ITEMS_HEADING = "Open items its tools flagged"

#: Words of open items added to a reader's result at most. Past it the
#: rest are counted, and every one is in the kept file under
#: :data:`OPEN_ITEMS_HEADING`.
OPEN_ITEMS_MAX_WORDS = 300

#: The argument names that say which file a call read, in order.
_SOURCE_KEYS = ("attachment_key", "path", "file_path", "source", "document",
                "file", "name")

#: The looking tools of a PDF page (:mod:`aec_agent_core.vision_tools`); a
#: zoom is told apart by its box.
_LOOK_TOOLS = frozenset({"analyze_pdf_page", "render_region"})
_ZOOM_TOOLS = frozenset({"render_region"})

#: The text reads of the pages a helper asked for, whose result says when
#: it was cut at its size limit and where to go on from (BACKLOG DR-50):
#: the core's ``read_pdf_text`` and planlens' ``read_document``.
#: ``read_text_file`` is left out: it returns as many characters as were
#: asked for from an offset, so a part of a file is what it was asked for
#: (a reader opens a kept file at the offsets its index gives).
_TEXT_READ_TOOLS = frozenset({"read_pdf_text", "read_document"})

#: Where a cut text read says to go on from: its ``truncated_note`` names
#: the next pages to ask for as ``pdf_pages='N-M'``
#: (:func:`aec_agent_core.vision_tools._dispatch_read_pdf_text`).
_NEXT_PDF_PAGE = re.compile(r"pdf_pages\s*=\s*['\"]?(\d+)")

#: The stem of a reader's record kept every time (a helper that stopped or
#: gave up keeps ``helper_``), so the page can tell them apart: the files
#: note labels it, and it makes no card under the reply.
READER_STEM = "reader"

#: The order open items are listed in, most serious first, so the cap of
#: :data:`OPEN_ITEMS_MAX_WORDS` cuts the least serious.
_FLAG_ORDER = ("error", "overview_not_read", "tiles_not_read", "cut_off",
               "text_cut", "scanned", "edge_note")


def _clause(text: Any, room: int = 140) -> str:
    """The first clause of a tool's note (up to its first ``"; "``), on one
    line, at most ``room`` characters."""
    s = " ".join(str(text or "").split()).lstrip("! ")
    s = s.split("; ", 1)[0].rstrip(" .")
    return s if len(s) <= room else s[:room].rstrip() + " ..."


def _result_dict(text: str) -> Optional[dict]:
    """A tool result's JSON object, or ``None`` (plain text). A result that
    opens with a line of its own before the object is read from the
    object's start."""
    s = (text or "").strip()
    if not s:
        return None
    for start in (0, s.find("\n{") + 1):
        if start < 0 or (start == 0 and not s.startswith("{")):
            continue
        try:
            data = json.loads(s[start:])
        except ValueError:
            continue
        return data if isinstance(data, dict) else None
    return None


def _calls_by_id(rec: HelperRecord) -> dict:
    out = {}
    for s in rec.steps:
        if s["kind"] == "model":
            for c in s["calls"]:
                if c.get("id"):
                    out[c["id"]] = c
    return out


def _source_of(args: dict) -> str:
    for k in _SOURCE_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _page_of(data: Optional[dict], args: dict) -> Optional[int]:
    """The page as a viewer counts it: the result's ``pdf_page``, else the
    call's ``pdf_page``, else its 0-based ``page`` + 1."""
    for v, plus in (((data or {}).get("pdf_page"), 0),
                    (args.get("pdf_page"), 0), (args.get("page"), 1)):
        if isinstance(v, bool):
            continue
        try:
            return int(v) + plus
        except (TypeError, ValueError):
            continue
    return None


def _look_flags(data: dict) -> dict:
    """``{flag: short words}`` for what one look says it did not read -- its
    own fields, never its prose. Left out: things found only in the tiles
    (``found`` rows marked ``tiles_only``; the normal result of tiling, which
    the tool says to treat as found) and the ``legibility`` caution. In the
    saved runs of the 40-sheet review the first were on 87-97 % of the
    readers' looks and the second on about 30 % of zooms, so they would
    bury the holes; the kept file holds both."""
    out = {}
    if data.get("overview_not_read"):
        out["overview_not_read"] = "the whole-page view was not read"
    if data.get("tiles_not_read"):
        out["tiles_not_read"] = ("part not read in close-up: "
                                 + _clause(data["tiles_not_read"]))
    if data.get("cut_off"):
        head = str(data["cut_off"]).partition(":")[0].strip()
        where = f" ({head})" if head and len(head) < 80 else ""
        out["cut_off"] = ("the vision answer was cut off at its output "
                          f"limit{where}, so the end of that reading is "
                          "missing")
    if data.get("edge_note"):
        out["edge_note"] = _clause(data["edge_note"])
    return out


def _ask_of(args: dict) -> str:
    """A look's question as the helper sent it, on one line (``""`` when it
    sent none and the tool's default was asked)."""
    return " ".join(str(args.get("prompt") or "").split())


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_spans(value: Any) -> List[tuple]:
    """``[(first, last), ...]`` of a page list (``[0, 1, 5]``) or a range
    string (``"0-2,5"``), as given; ``[]`` when it is neither."""
    if isinstance(value, (list, tuple)):
        return [(n, n) for n in map(_int_or_none, value) if n is not None]
    spans = []
    for a, b in re.findall(r"(\d+)(?:\s*-\s*(\d+))?", str(value or "")):
        spans.append((int(a), int(b or a)))
    return spans


def _page_set(value: Any) -> set:
    """The pages of :func:`_page_spans`, one by one."""
    return {n for a, b in _page_spans(value) for n in range(a, b + 1)}


def _ranges(pages: Any) -> str:
    """``[3, 4, 5, 9]`` -> ``"3-5,9"``."""
    out: List[str] = []
    for n in sorted(pages):
        if out and n == int(out[-1].rpartition("-")[2]) + 1:
            out[-1] = out[-1].partition("-")[0] + f"-{n}"
        else:
            out.append(str(n))
    return ",".join(out)


def _pdf_text_asked(args: dict, data: dict) -> Optional[set]:
    """The pages (as a viewer counts them) a ``read_pdf_text`` call asked
    for, resolved by the tool's own reading of its arguments
    (:func:`aec_agent_core.vision_tools._read_pdf_text_pages` and
    ``_parse_pages``; no page argument is its first ``max_pages`` pages),
    or ``None`` when that cannot be said."""
    try:
        from aec_agent_core import vision_tools as tools
        pages, one_based, refused = tools._read_pdf_text_pages(args)
        if refused is not None:
            return None
        default_n = (_int_or_none(args.get("max_pages"))
                     or tools._PDF_TEXT_DEFAULT_PAGES)
        n_total = _int_or_none(data.get("n_pages_total")) or 100_000
        found, _note = tools._parse_pages(pages, n_total, default_n,
                                          one_based=one_based)
    except Exception:  # noqa: BLE001 - unsure: the cut stays listed
        return None
    return {n + 1 for n in found}


def _text_cut(tool: str, args: dict, data: dict) -> Optional[dict]:
    """What a text read whose result says it was cut at its size limit
    left unread, else ``None`` (BACKLOG DR-50), in the tool's own page
    numbers: ``owed`` maps each page not read yet to its first line not
    read (``None`` for a file read by lines), ``where`` is the tool's own
    words for where to go on from, and ``stuck`` is true when the result
    names no place to go on from (then the cut stays listed).

    * ``read_pdf_text``: ``truncated``; its ``truncated_note`` names the
      page to ask for next (``pdf_pages='N-M'``, N the page cut short or
      the one after the last read whole). Owed: the pages the call asked
      for (:func:`_pdf_text_asked`) from N on, counted as a viewer does;
      it reads a page from its top.
    * ``read_document``: a ``next`` cursor (``pages`` from 0, and
      ``start_line`` on the first of them). Owed: the cursor's pages, its
      first from ``start_line``; a Word, Excel or DXF file's cursor has
      ``start_line`` only."""
    if tool == "read_pdf_text":
        if not data.get("truncated"):
            return None
        note = str(data.get("truncated_note") or "")
        where = (f"the tool's note: {_clause(note, 160)}" if note.strip()
                 else "")
        found = _NEXT_PDF_PAGE.search(note)
        if not found:
            return {"owed": {}, "where": where, "stuck": True}
        nxt = int(found.group(1))
        asked = _pdf_text_asked(args, data)
        owed = {n: 0 for n in (asked or ()) if n >= nxt} or {nxt: 0}
        return {"owed": owed, "where": where, "stuck": asked is None}
    if tool == "read_document":
        nxt = data.get("next")
        if not isinstance(nxt, dict) or not nxt:
            return None
        where = f"the tool's cursor to go on from: next={_json(nxt)}"
        pages = sorted(_page_set(nxt.get("pages")))
        line = _int_or_none(nxt.get("start_line"))
        if pages:
            owed = {n: 0 for n in pages}
            owed[pages[0]] = line or 0
            return {"owed": owed, "where": where, "stuck": False}
        if line is None:
            return {"owed": {}, "where": where, "stuck": True}
        return {"owed": {None: line}, "where": where, "stuck": False}
    return None


def _text_read_on(tool: str, args: dict, data: dict, owed: dict) -> None:
    """Take from ``owed`` (:func:`_text_cut`) what a later read of the same
    file by the same tool (``args``, ``data``) read: a page it returned
    whole from at or before the first line owed there. The page it was
    itself cut on counts read up to its own cut, which it carries as a new
    cut. Unsure is unread: the rest stays owed."""
    nxt = data.get("next") if isinstance(data.get("next"), dict) else {}
    if tool == "read_pdf_text":
        returned = {n for n in map(_int_or_none,
                                   data.get("pdf_pages_returned") or ())
                    if n is not None}
        if data.get("truncated") and returned:
            # the page cut short: the one its note names, else the last
            found = _NEXT_PDF_PAGE.search(str(data.get("truncated_note")
                                              or ""))
            returned.discard(int(found.group(1)) if found
                             else max(returned))
        for n in [n for n in owed if n in returned]:
            del owed[n]
        return
    start = _int_or_none(args.get("start_line")) or 0
    cut_pages = sorted(_page_set(nxt.get("pages")))
    cut_page = cut_pages[0] if cut_pages else None
    cut_line = _int_or_none(nxt.get("start_line"))
    returned = _page_set(data.get("pages_returned"))
    for n, line in list(owed.items()):
        if n is None:
            # a file read by lines: from at or before the line, with text
            if (start > line or cut_pages
                    or not str(data.get("text") or "").strip()):
                continue
        elif n not in returned or (n == min(returned) and start > line):
            continue
        if n is None and nxt or n is not None and n == cut_page:
            if cut_line is not None and cut_line > line:
                owed[n] = cut_line
        else:
            del owed[n]


def _text_cut_words(tool: str, state: dict) -> tuple:
    """``(label page or None, where)`` of a cut text read's open item:
    the tool's own words while nothing of it has been read since, else
    what is still owed, as the tool's own arguments."""
    owed, last = state["owed"], state["last"]
    pages = sorted(n for n in owed if n is not None)
    label = (pages[0] + (1 if tool == "read_document" else 0)
             if pages else None)
    if owed == last["owed"]:
        return label, last["where"]
    if tool == "read_pdf_text":
        parts = [f"pdf_pages='{_ranges(pages)}'"] if pages else []
    else:
        runs: List[list] = []        # [first, last, line]
        for n in pages:
            if runs and n == runs[-1][1] + 1 and not owed[n]:
                runs[-1][1] = n
            else:
                runs.append([n, n, owed[n]])
        parts = [f"pages='{_ranges(range(a, b + 1))}'"
                 + (f", start_line={line}" if line else "")
                 for a, b, line in runs]
        if None in owed:
            parts.append(f"start_line={owed[None]}")
    where = ("still to read, in the tool's own numbers: " + "; ".join(parts)
             if parts else "")
    if state["stuck"] and last["where"]:
        where = "; ".join(x for x in (where, last["where"]) if x)
    return label, where


def _box4(value: Any) -> Optional[tuple]:
    """``(x0, y0, x1, y1)``, ordered, of a box given as four numbers, else
    ``None``."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    nums = []
    for v in value:
        if isinstance(v, bool):
            return None
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            return None
    x0, y0, x1, y1 = nums
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return (max(a[0], b[0]) < min(a[2], b[2])
            and max(a[1], b[1]) < min(a[3], b[3]))


def _failed_zoom_box(args: dict) -> tuple:
    """``(box, bad_box)`` of a zoom that failed (DR-50). ``box``: the box it
    asked for in page points, its ``bbox`` of four numbers, or its
    ``view`` + ``image_box`` turned into page points by the function the
    tool uses. ``bad_box``: the failure was in the box's arguments, as
    ``_dispatch_render_region`` checks them before it renders: a ``bbox``
    with a ``view`` or ``image_box`` too, a ``view`` + ``image_box`` that
    function refuses, or a ``bbox`` that is not four numbers. ``(None,
    False)``: it named no box (a whole page or image), or that cannot be
    said."""
    bbox, view, image_box = (args.get(k) for k in ("bbox", "view",
                                                   "image_box"))
    if bbox is not None:
        if view is not None or image_box is not None:
            return None, True
        box = _box4(bbox)
        return box, box is None
    if view is None and image_box is None:
        return None, False
    try:
        from aec_agent_core import vision_view
        box = _box4(vision_view.image_box_to_page(view or [],
                                                  image_box or []))
    except (TypeError, ValueError):
        return None, True
    except Exception:  # noqa: BLE001 - unsure: no redo settles it
        return None, False
    return box, False


def _zoom_place(args: dict) -> Optional[str]:
    """A zoom's place in the key of its flags, as it was given: its
    ``bbox``, else its ``view`` + ``image_box``; ``None`` for a zoom on a
    whole page or image. Only a later zoom of the same place settles a flag
    by its key (DR-50); before, every ``view`` + ``image_box`` zoom on a
    page shared one key, so any later good one there dropped a failed
    one."""
    if args.get("bbox") is not None:
        return _json(args.get("bbox"))
    if args.get("view") is not None or args.get("image_box") is not None:
        return _json({"view": args.get("view"),
                      "image_box": args.get("image_box")})
    return None


def open_items(rec: HelperRecord) -> List[str]:
    """The open items a helper's own tools flagged, one line per file and
    page, the most serious first: a call that failed; a look whose
    whole-page view or tiles were not read, or whose answer was cut off; a
    text read cut at its size limit, with where the tool says to go on from
    (:func:`_text_cut`; DR-50); pages a text read found with no text layer;
    lettering that ran past a zoom's edges (:func:`_look_flags`). Made from
    the record's tool results (the same results the page readings keep),
    never from the helper's words: how well pages were read, not what a
    reading asked to check.

    A flag is dropped when the helper read the thing again: a later result
    of the same tool on the same file and page (for a zoom, the same place,
    given the same way: :func:`_zoom_place`) without it; for a page with no
    text layer, a later look at it.

    DR-50, where it is not clear that a later call replaced an earlier one,
    the earlier one stays listed:

    * a failed zoom is also settled by a redo: a zoom on the same file and
      page that asked the same question (a failed zoom that gave no
      question of its own stays), called in a model step after the
      failure came back (never one sent with it), that did not fail. When
      the failed zoom had a box (:func:`_failed_zoom_box`), the redo's box
      (its ``bbox``, else its result's) must overlap it, and a redo
      settles only the failures it overlaps; when the failure was in the
      box's arguments, any such redo settles it.
    * a cut text read (:func:`_text_cut`) stays until later reads of the
      same file by the same tool have, between them, read every page it
      left (:func:`_text_read_on`); a later read cut again carries its
      own cut with what is left. Its item names what is still to read in
      the tool's own numbers (:func:`_text_cut_words`).

    The helper may still have settled an item in its own words; its kept
    file shows everything."""
    calls = _calls_by_id(rec)
    flagged: dict = {}           # (key, flag) -> (label, words)
    failed_zooms: dict = {}      # a failed zoom's key -> (ask, box, bad, at)
    cuts: dict = {}              # (tool, file) -> a cut text read's state
    issued: dict = {}            # call id -> the index of its model step
    by_id: dict = {}             # call id -> the call, as last issued
    for i, s in enumerate(rec.steps):
        if s["kind"] == "model":
            for c in s["calls"]:
                if c.get("id"):
                    issued[c["id"]], by_id[c["id"]] = i, c
            continue
        if s["kind"] != "result":
            continue
        call = by_id.get(s.get("id")) or calls.get(s.get("id")) or {}
        called_at = issued.get(s.get("id"))
        tool = str(call.get("name") or s.get("name") or "tool")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        data = _result_dict(s.get("text") or "")
        source = _source_of(args)
        page = _page_of(data, args)
        box = _zoom_place(args) if tool in _ZOOM_TOOLS else None
        key = (tool, source, page, box)
        label = " ".join(x for x in (
            source or tool, f"page {page}" if page is not None else "",
            "(a zoom)" if tool in _ZOOM_TOOLS else "") if x)
        error = (data or {}).get("error")
        failed = s.get("status") == "error" or bool(error)
        if failed:
            flags = {"error": f"{tool} failed: "
                              + _clause(error or s.get("text"), 120)}
            if tool in _ZOOM_TOOLS:
                failed_zooms[key] = (_ask_of(args),
                                     *_failed_zoom_box(args), i)
        else:
            flags = _look_flags(data) if data else {}
            if tool in _LOOK_TOOLS and page is not None:
                for k in list(flagged):
                    (k_tool, k_source, k_page, _b), _flag = k
                    if (k_tool == "(text layer)" and k_source == source
                            and k_page == page):
                        flagged.pop(k)
            ask = _ask_of(args)
            if (tool in _ZOOM_TOOLS and page is not None and ask
                    and called_at is not None):
                # a failed zoom redone (DR-50)
                redo_box = (_box4(args.get("bbox"))
                            or _box4((data or {}).get("bbox")))
                for k in list(flagged):
                    (k_tool, k_source, k_page, _b), flag = k
                    if (flag != "error" or k_tool != tool
                            or k_source != source or k_page != page
                            or k[0] not in failed_zooms):
                        continue
                    f_ask, f_box, bad_box, f_at = failed_zooms[k[0]]
                    if f_ask != ask or called_at <= f_at:
                        continue
                    if (bad_box if f_box is None
                            else (redo_box is not None
                                  and _boxes_overlap(f_box, redo_box))):
                        flagged.pop(k)
        for k in [k for k in flagged if k[0] == key and k[1] not in flags]:
            flagged.pop(k)
        for flag, words in flags.items():
            flagged[(key, flag)] = (label, words)
        for p in (data or {}).get("scanned_pages") or () if not flags else ():
            try:
                n = int(p) + 1
            except (TypeError, ValueError):
                continue
            k = (("(text layer)", source, n, None), "scanned")
            flagged[k] = (" ".join(x for x in (source, f"page {n}") if x),
                          "no text layer, so a text read saw nothing there "
                          "(only a look reads it)")
        if tool in _TEXT_READ_TOOLS and data and not failed:
            # a text read cut at its size limit (DR-50): what it left is
            # owed until later reads of the same file by the same tool
            # have read it all
            text_source = source or str(args.get("handle") or "")
            state = cuts.get((tool, text_source))
            if state is not None:
                _text_read_on(tool, args, data, state["owed"])
            cut = _text_cut(tool, args, data)
            if cut is not None:
                if state is None:
                    state = cuts[(tool, text_source)] = {
                        "owed": {}, "stuck": False}
                for n, line in cut["owed"].items():
                    state["owed"][n] = min(state["owed"].get(n, line), line)
                state["stuck"] = state["stuck"] or cut["stuck"]
                state["last"] = cut
            fkey = ((("(text cut)", tool), text_source, None, None),
                    "text_cut")
            if state is not None and (state["owed"] or state["stuck"]):
                n, where = _text_cut_words(tool, state)
                flagged[fkey] = (
                    " ".join(x for x in (
                        text_source or tool,
                        f"page {n}" if n is not None else "") if x),
                    f"a text read ({tool}) was cut at its size limit, so "
                    "the rest of the text it asked for was not read"
                    + (f" ({where})" if where else ""))
            elif state is not None:
                cuts.pop((tool, text_source))
                flagged.pop(fkey, None)
    rank = {f: i for i, f in enumerate(_FLAG_ORDER)}
    lines: dict = {}             # label -> [rank, [words]]
    for (_key, flag), (label, words) in flagged.items():
        entry = lines.setdefault(label, [len(rank), []])
        entry[0] = min(entry[0], rank.get(flag, len(rank)))
        entry[1].append((rank.get(flag, len(rank)), words))
    ordered = sorted(lines.items(), key=lambda kv: kv[1][0])
    return [f"- {label}: " + "; ".join(w for _r, w in sorted(entry[1],
                                                             key=lambda x: x[0]))
            for label, entry in ordered]


def open_items_note(items: List[str], kept_name: Optional[str]) -> str:
    """What the primary is told of a reader's open items, at most
    :data:`OPEN_ITEMS_MAX_WORDS` words of them; the rest are counted and
    left to the kept file."""
    shown: List[str] = []
    words = 0
    for line in items:
        n = len(line.split())
        if shown and words + n > OPEN_ITEMS_MAX_WORDS:
            break
        shown.append(line)
        words += n
    more = len(items) - len(shown)
    tail = ""
    if more:
        tail = (f"\n{more} more open item(s) are not shown here"
                + (f"; every one is in '{kept_name}' under "
                   f"'{OPEN_ITEMS_HEADING}'." if kept_name else "."))
    return (f"[{OPEN_ITEMS_HEADING}, listed in code from its record whether "
            "or not its answer names them (it may have settled some in its "
            "own words):\n" + "\n".join(shown) + tail + "]")


def reader_note(rec: HelperRecord, path: str) -> str:
    """What the primary is told about a reader's kept record."""
    return (f"[This reading helper's whole record ({rec.n_results} tool "
            f"results, {rec.read_chars:,} characters) is kept in the file "
            f"'{_relative(path)}' in the working folder. Read it with "
            "read_text_file -- its index gives the offset of each result -- "
            "to compare what it read with other pages, instead of reading "
            "the same pages again.]")


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

    def acquire(self, stoppable: bool = False) -> float:
        """Wait for a place; the seconds waited. ``stoppable``: the turn can
        be stopped by the user, so the wait looks every half second and
        raises :class:`TurnStopped` once asked (TA-37)."""
        t0 = time.monotonic()
        with self._cond:
            me = object()
            self._queue.append(me)
            try:
                while not (self._running < self.limit
                           and self._queue[0] is me):
                    if stoppable:
                        check_stop()
                        self._cond.wait(0.5)
                    else:
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
        from aec_agent_core.deep.vision_engine import conversation_key
        return conversation_key()
    except Exception:  # noqa: BLE001
        return ""


class KeepWork(AgentMiddleware):
    """On the primary: keep a stopped helper's work (see the module
    docstring), cap the helpers running at once, and keep the full text of
    very large tool results on the turn scope. Put it just inside
    :class:`~funhouse_agent.deep.tool_guards.ToolErrorGuard`, so a helper's
    error reaches it first.

    ``readers``: the names of the page's reading helpers (BACKLOG DR-46).
    Each of their returns keeps the whole record and adds its open items
    (see the module docstring); none named, every helper is kept as
    before."""

    def __init__(self, max_at_once: Optional[int] = None,
                 readers=()):
        super().__init__()
        self._limit = max_at_once
        self.readers = frozenset(str(n) for n in (readers or ()))
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
        from aec_agent_core.error_text import error_line
        call = getattr(request, "tool_call", None) or {}
        out = {"error": f"task failed: the {rec.subagent} helper stopped "
                        f"on an error: {error_line(exc)}"}
        # The records it held go to a file of their own, whatever it read
        # (BACKLOG RI-31).
        records = save_held_records(rec)
        if records:
            out["records_file"] = records["name"]
            out["records"] = _records_note(records)
        if rec.read_chars >= KEEP_MIN_READ_CHARS or rec.arg_chars >= 2_000:
            why = f"it stopped on an error ({error_line(exc, 160)})"
            path = write_helper_file(rec, why, records=records)
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
            from aec_agent_core.error_text import budget_exhausted, BUDGET_HINT
            if budget_exhausted(exc) is not None:
                out["hint"] = BUDGET_HINT
        except Exception:  # noqa: BLE001
            pass
        return ToolMessage(content=json.dumps(out, ensure_ascii=False),
                           name=TASK_TOOL, tool_call_id=call.get("id") or "",
                           status="error")

    def _finished(self, rec: HelperRecord, result):
        """A helper that returned: keep its work when it gave up, or every
        time for a named reader (DR-46)."""
        msgs = _result_messages(result)
        answer = "\n".join(_text(m.content) for m in msgs)
        why = gave_up(rec, answer)
        if rec.subagent in self.readers and rec.n_results:
            return self._reader_returned(rec, result, answer, why)
        if why is None:
            # A helper that saved files: their names reach the primary in
            # code, whether or not its reply named them (RI-31).
            files = saved_files(rec)
            return _with_note(result, saved_note(files)) if files else result
        records = save_held_records(rec, answer)
        path = write_helper_file(rec, why, answer, records)
        if not path:
            return (_with_note(result, _records_note(records)) if records
                    else result)
        _event("helper_kept", {"task_id": rec.task_id,
                               "subagent": rec.subagent,
                               "path": _relative(path), "why": why,
                               "read_chars": rec.read_chars})
        note = kept_note(rec, path, why, records)
        return _with_note(result, note)

    def _reader_returned(self, rec: HelperRecord, result, answer: str,
                         why: Optional[str]):
        """A named reader that returned: its whole record goes to
        ``kept/``, the file is named in the result, and the open items its
        tools flagged are added (DR-46). One that gave up is kept as any
        helper that gave up is, with its open items added."""
        try:
            items = open_items(rec)
        except Exception:  # noqa: BLE001 - keeping never fails the call
            items = []
        records = save_held_records(rec, answer) if why else None
        path = write_helper_file(
            rec, why or READER_KEPT_WHY, answer, records, items,
            None if why else READER_KEPT_TITLE, reader=not why)
        notes: List[str] = []
        if path:
            _event("helper_kept", {"task_id": rec.task_id,
                                   "subagent": rec.subagent,
                                   "path": _relative(path),
                                   "why": why or READER_KEPT_WHY,
                                   "read_chars": rec.read_chars,
                                   "open_items": len(items)})
            notes.append(kept_note(rec, path, why, records) if why
                         else reader_note(rec, path))
        elif records:
            notes.append(_records_note(records))
        files = saved_files(rec)
        if files and not why:
            notes.append(saved_note(files))
        if items:
            notes.append(open_items_note(
                items, _relative(path) if path else None))
        return _with_note(result, "\n".join(notes)) if notes else result

    # -- the wrapper ------------------------------------------------------
    def _begin(self, request):
        call = getattr(request, "tool_call", None) or {}
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        rec = HelperRecord(call.get("id") or "", args.get("subagent_type"),
                           args.get("description"))
        places = self._places_for()
        scope = current_turn()
        stoppable = scope is not None and scope.stop_event is not None
        if stoppable:
            check_stop()                   # TA-37: before the helper starts
        waited = (places.acquire(stoppable=stoppable) if places is not None
                  else 0.0)
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
           "TurnStopped", "stop_requested", "check_stop", "note_budget",
           "note_primary_model_call", "turn_model_calls",
           "extra_model_calls", "take_extra_model_call",
           "current_helper", "HelperRecorder", "record_of_messages",
           "KeepWork",
           "note_budget_reached", "gave_up", "summary_paragraph",
           "saved_files", "saved_note", "SAVED_FILES_HINT_HOOK",
           "DEFAULT_SAVED_FILES_HINT", "save_held_records",
           "helper_file_text", "write_helper_file", "kept_note",
           "kept_folder", "max_helpers", "open_items",
           "open_items_note", "reader_note", "READER_KEPT_WHY",
           "READER_STEM",
           "OPEN_ITEMS_HEADING", "OPEN_ITEMS_MAX_WORDS", "KEPT_DIR",
           "KEEP_ON_GIVE_UP",
           "KEEP_MIN_READ_CHARS", "SHORT_ANSWER_CHARS",
           "FULL_RESULT_MIN_CHARS", "MAX_HELPERS_ENV", "DEFAULT_MAX_HELPERS",
           "BUSY_TURN_MAX_ENV", "DEFAULT_BUSY_TURN_MAX_S",
           "ImageBudgetSpent", "IMAGE_CEILING_ERROR", "take_image_call",
           "counting_look", "with_image_calls", "image_calls_status",
           "image_ceiling_met",
           "image_ceiling_told", "image_call_facts",
           "note_lean_prompt", "lean_prompt_facts"]
