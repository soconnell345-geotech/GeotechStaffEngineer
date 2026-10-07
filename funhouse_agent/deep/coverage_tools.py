"""Coverage in code: the recorder, the gate and the tools (switch-gated).

Behind ``GEOTECH_COVERAGE`` (and ``GEOTECH_REVIEW_CHECKLIST`` for the
checklist tool), both OFF by default (:mod:`funhouse_agent.review_flags`).
The ledger itself, with no LangChain in it, is :mod:`funhouse_agent.coverage`.

* :class:`CoverageRecorder` - a middleware on the primary agent AND every
  helper. After each tool call it hands the call's name, arguments and
  result to the ledger, which works out from them which pages were read as
  text or looked at. Nothing the model SAYS it read counts; only the calls.
  This is the layer the activity log works at (it sees every tool call of
  every agent); a middleware rather than a callback because the ledger has
  to be the same object the gate and the tools read, bound when the agent is
  built.
* :class:`CoverageGate` - the recorder plus the gate, on the primary only.
  When the model replies with no tool call (it is trying to finish) and the
  turn took data out of a document - it opened a coverage task with
  ``document_coverage`` or wrote a data file with ``write_diggs`` (then every
  page but the cover, contents and tabs counts), or read pages of a data
  group (then every log, lab and field-test page counts) - and such pages
  nobody read remain, the model is told ONCE, with the list, and goes on: it
  may read them, mark them skipped with a reason, or say why not. It never
  loops: one note per user turn (an auto-continue of the same turn counts
  as the same turn), and it stands down when the turn is nearly out of steps
  or model calls, so it can never cost a turn its answer. The precedent is
  :class:`funhouse_agent.deep.limits.ModelCallBudgetMiddleware`, which acts at
  the end in the same way.
* ``document_coverage`` and ``report_checklist`` - the tools. The agent learns
  them from their own descriptions; no prompt rule names them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, ToolMessage

try:
    from langchain.agents.middleware import AgentMiddleware, hook_config
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware, hook_config

from funhouse_agent import coverage as _cov
from funhouse_agent import review_flags

#: Fewest graph steps that must be left in the turn for the gate to speak:
#: reading the pages it lists takes a few model calls of ~6 steps each, and a
#: GraphRecursionError would cost the turn its answer.
MIN_STEPS_FOR_GATE = 12

#: The host step cap a build with the gate asks for at least (the webapp
#: raises its per-turn cap to an agent's ``geotech_min_recursion_limit``):
#: the app's default of 50 plus room for the reads the gate may ask for.
COVERAGE_STEP_FLOOR = 80

#: What the web app's auto-continue sends (``webapp.core.CONTINUE_NUDGE``;
#: a test keeps the two equal): a continuation of the SAME user turn, not a
#: new one. Kept here so the library never imports the web app.
CONTINUE_NUDGE = "Continue — complete the action you just stated."

DOCUMENT_COVERAGE_DESCRIPTION = (
    "Coverage of one document for a task that takes data out of it or "
    "reviews all of it: every boring log and laboratory sheet of a report, a "
    "report into DIGGS or a table, a whole-set review. Call it when such a "
    "task starts: it lists the document's pages by what they are "
    "(exploration logs, laboratory sheets, plans and figures, calculations, "
    "text...), read from the document's own structure, and from then on the "
    "app records every page any tool or helper reads or looks at. Call it "
    "again to mark pages `extracted` (their data is in your output) or "
    "`skipped` with a `reason`, and before you answer, to get the coverage "
    "to state in your answer as counts ('laboratory sheets 10 of 23 read; "
    "not read: PDF pages 96-103'). A page with no text layer counts as read "
    "only once it has been looked at. `source`: the document's handle, "
    "attachment key or path. Pages are 0-based, as in every tool; the "
    "*_pdf lists are what a reader cites.")

REPORT_CHECKLIST_DESCRIPTION = (
    "The report-review checklist for a geotechnical report (a DRAFT the "
    "owner is still marking up). It runs the checks the app can make itself "
    "- coverage of the logs and laboratory sheets, the explorations named in "
    "the text against the logs, the laboratory sheets against the logs, and "
    "the cross-checks of data written out as DIGGS - and returns the items "
    "that need your judgement: campaigns and dates, groundwater, locations "
    "and datum, consistency, currency of the codes cited, plausibility, "
    "recommendations. Use it when reviewing a report or checking data taken "
    "out of one, and report against every item. `source`: the document's "
    "handle, attachment key or path.")


def _text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b)
                         for b in content)
    return "" if content is None else str(content)


def turn_key(messages) -> str:
    """Which USER turn a run belongs to: the number of user messages and a
    hash of the latest. The gate's own note and the web app's auto-continue
    nudge are not user turns, so a continuation pass is the same turn."""
    nudge = CONTINUE_NUDGE.strip()
    real: List[str] = []
    for m in messages or []:
        kind = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict)
                                            else None)
        if kind not in ("human", "user"):
            continue
        text = _text(m if not isinstance(m, dict) else m.get("content"))
        if text.startswith(_cov.GATE_PREFIX) or text.strip() == nudge:
            continue
        real.append(text)
    if not real:
        return "0:"
    return f"{len(real)}:{hashlib.sha1(real[-1].encode('utf-8', 'replace')).hexdigest()[:12]}"


def _steps_left() -> Optional[int]:
    """Graph steps left in this run, when LangGraph says."""
    try:
        from langgraph.config import get_config
        cfg = get_config() or {}
        limit = cfg.get("recursion_limit")
        step = (cfg.get("metadata") or {}).get("langgraph_step")
        if isinstance(limit, int) and isinstance(step, int):
            return limit - step
    except Exception:  # noqa: BLE001 - outside a graph run
        return None
    return None


def _emit(note: str) -> None:
    """Put the gate's note in the activity log (``coverage_gate`` event), so
    a review of the run can see what the model was told."""
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event("coverage_gate", {"note": note})
    except Exception:  # noqa: BLE001 - logging must never cost the turn
        pass


class CoverageRecorder(AgentMiddleware):
    """Feed every finished tool call of this agent to the ledger."""

    def __init__(self, ledger: "_cov.CoverageLedger", agent: str = "primary"):
        super().__init__()
        self.ledger = ledger
        self.agent = agent

    def _record(self, request, result) -> None:
        try:
            call = getattr(request, "tool_call", None) or {}
            if not isinstance(result, ToolMessage):
                return               # a Command (e.g. task): its helper records
            self.ledger.record_call(str(call.get("name") or ""),
                                    call.get("args") or {}, _text(result),
                                    agent=self.agent)
        except Exception as exc:  # noqa: BLE001 - never fail a tool call
            self.ledger.last_error = f"{type(exc).__name__}: {exc}"

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        self._record(request, result)
        return result

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        self._record(request, result)
        return result


class CoverageGate(CoverageRecorder):
    """The recorder, plus the once-per-turn gate (primary agent only)."""

    def __init__(self, ledger: "_cov.CoverageLedger", agent: str = "primary",
                 *, budget: Optional[int] = None, checklist: bool = False,
                 min_steps: int = MIN_STEPS_FOR_GATE):
        super().__init__(ledger, agent)
        self.budget = budget
        self.checklist = checklist
        self.min_steps = int(min_steps)

    def before_agent(self, state, runtime):
        try:
            self.ledger.begin_turn(turn_key(state.get("messages")))
        except Exception:  # noqa: BLE001
            pass
        return None

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def _extra(self, key: str) -> List[str]:
        if not self.checklist:
            return []
        try:
            from funhouse_agent import report_checklist
            lines: List[str] = []
            for doc_key in self.ledger.armed(key):
                found = report_checklist.failed_lines(self.ledger, doc_key)
                if found:
                    lines.append(f"Checks the app ran on "
                                 f"{self.ledger.docs[doc_key]['name']} "
                                 "(report checklist, DRAFT) that failed:")
                    lines += found
            return lines
        except Exception:  # noqa: BLE001 - the checklist is extra
            return []

    def _gate(self, state) -> Optional[Dict[str, Any]]:
        try:
            msgs = state.get("messages") or []
            last = msgs[-1] if msgs else None
            if last is None or getattr(last, "type", "") != "ai" \
                    or getattr(last, "tool_calls", None):
                return None
            key = turn_key(msgs)
            if self.ledger.gate_fired(key):
                return None
            if self.budget:
                used = int(state.get("run_model_call_count", 0) or 0)
                if used >= self.budget - 1:
                    return None      # the budget's forced last answer
            note = self.ledger.gate_note(key, answer=_text(last),
                                         extra=self._extra(key))
            if not note:
                return None
            room = _steps_left()
            if room is not None and room < self.min_steps:
                self.ledger.note_skipped(key, f"only {room} graph steps "
                                              "were left in the turn")
                return None
            self.ledger.note_fired(key, note)
            _emit(note)
            return {"messages": [HumanMessage(content=note)],
                    "jump_to": "model"}
        except Exception as exc:  # noqa: BLE001 - never cost the answer
            self.ledger.last_error = f"gate: {type(exc).__name__}: {exc}"
            return None

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        return self._gate(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        return self._gate(state)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _ref(source: str):
    """A handle when ``source`` is the handle of an open document."""
    from funhouse_agent import document_tools
    s = str(source or "").strip()
    if s.startswith("doc_") and document_tools.document_entry(s) is not None:
        return ("handle", s)
    return ("source", s)


def _fit(out: Dict[str, Any], max_chars: int) -> str:
    """JSON within ``max_chars``: the item list is shortened first, then
    dropped; the groups and the statement always stay."""
    text = json.dumps(out, ensure_ascii=False)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    items = list(out.get("items") or [])
    while items and len(text) > max_chars:
        items = items[: max(0, len(items) // 2)]
        out = dict(out, items=items,
                   items_note="work items shortened to fit; the groups are "
                              "complete")
        text = json.dumps(out, ensure_ascii=False)
    if len(text) > max_chars:
        out.pop("items", None)
        text = json.dumps(out, ensure_ascii=False)
    return text


def make_coverage_tool(ledger: "_cov.CoverageLedger",
                       max_result_chars: int = 8000):
    """The ``document_coverage`` tool over ``ledger``."""
    from langchain_core.tools import StructuredTool

    def document_coverage(source: str, extracted: Any = None,
                          skipped: Any = None, reason: str = "") -> str:
        try:
            key = ledger.resolve(_ref(source))
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return json.dumps({"error": f"could not open '{source}': "
                                        f"{type(exc).__name__}: {exc}"})
        if key is None:
            return json.dumps({"error": f"'{source}' is not an open document "
                                        "handle, an attachment key or a "
                                        "readable path"})
        n = ledger.docs[key]["inventory"].n_pages
        skip_pages = _cov.parse_pages(skipped, n)
        if skip_pages and not str(reason or "").strip():
            return json.dumps({"error": "say why the pages are skipped: "
                                        "pass a reason"})
        ledger.declare(key)
        marked: Dict[str, str] = {}
        got = ledger.mark(key, _cov.parse_pages(extracted, n), "extracted")
        if got:
            marked["extracted"] = _cov.compact(got)
        got = ledger.mark(key, skip_pages, "skipped", str(reason or ""))
        if got:
            marked["skipped"] = _cov.compact(got)
        out = ledger.report(key)
        if marked:
            out = {"marked": marked, **out}
        return _fit(out, max_result_chars)

    return StructuredTool.from_function(
        document_coverage, name="document_coverage",
        description=DOCUMENT_COVERAGE_DESCRIPTION)


def make_checklist_tool(ledger: "_cov.CoverageLedger",
                        max_result_chars: int = 16000):
    """The ``report_checklist`` tool over ``ledger``."""
    from langchain_core.tools import StructuredTool

    def report_checklist(source: str) -> str:
        from funhouse_agent import report_checklist as _rc
        try:
            key = ledger.resolve(_ref(source))
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"could not open '{source}': "
                                        f"{type(exc).__name__}: {exc}"})
        if key is None:
            return json.dumps({"error": f"'{source}' is not an open document "
                                        "handle, an attachment key or a "
                                        "readable path"})
        try:
            out = _rc.run_checklist(ledger, key)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"the checklist failed: "
                                        f"{type(exc).__name__}: {exc}"})
        text = json.dumps(out, ensure_ascii=False)
        if max_result_chars > 0 and len(text) > max_result_chars:
            for row in out.get("code_checks") or []:
                row.pop("entries", None)
            text = json.dumps(out, ensure_ascii=False)[:max_result_chars]
        return text

    return StructuredTool.from_function(
        report_checklist, name="report_checklist",
        description=REPORT_CHECKLIST_DESCRIPTION)


# ---------------------------------------------------------------------------
# What a builder attaches
# ---------------------------------------------------------------------------

class CoverageKit:
    """Everything one agent build needs, or nothing when both switches are
    off: ``tools`` for the primary, ``gate`` (or a plain recorder) for the
    primary's middleware, and :meth:`recorder` for each helper."""

    def __init__(self, ledger: Optional["_cov.CoverageLedger"],
                 tools: List[Any], gate_on: bool, checklist_on: bool):
        self.ledger = ledger
        self.tools = tools
        self.gate_on = gate_on
        self.checklist_on = checklist_on

    @property
    def on(self) -> bool:
        return self.ledger is not None

    def primary(self, budget: Optional[int] = None):
        if self.ledger is None:
            return None
        if self.gate_on:
            return CoverageGate(self.ledger, "primary", budget=budget,
                                checklist=self.checklist_on)
        return CoverageRecorder(self.ledger, "primary")

    def recorder(self, agent: str):
        if self.ledger is None:
            return None
        return CoverageRecorder(self.ledger, agent)


def coverage_kit(attachments: Optional[Dict[str, bytes]] = None,
                 folder: Optional[str] = None,
                 max_result_chars: int = 8000) -> CoverageKit:
    """Read the switches and build the kit (an empty kit when both are off).

    ``folder`` is the conversation's folder: the ledger is kept there as
    ``coverage.json`` and picked up again when the agent is rebuilt.
    """
    gate_on = review_flags.coverage()
    checklist_on = review_flags.checklist()
    if not (gate_on or checklist_on):
        return CoverageKit(None, [], False, False)
    ledger = _cov.CoverageLedger(path=_cov.ledger_path(folder),
                                 attachments=attachments)
    cap = max_result_chars if max_result_chars > 0 else 0
    tools: List[Any] = []
    if gate_on:
        tools.append(make_coverage_tool(ledger, max(cap, 8000) if cap else 0))
    if checklist_on:
        tools.append(make_checklist_tool(ledger, max(cap, 16000) if cap else 0))
    return CoverageKit(ledger, tools, gate_on, checklist_on)


__all__ = ["CoverageRecorder", "CoverageGate", "CoverageKit", "coverage_kit",
           "make_coverage_tool", "make_checklist_tool", "turn_key",
           "DOCUMENT_COVERAGE_DESCRIPTION", "REPORT_CHECKLIST_DESCRIPTION",
           "MIN_STEPS_FOR_GATE", "COVERAGE_STEP_FLOOR"]
