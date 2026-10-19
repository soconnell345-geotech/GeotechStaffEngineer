"""Model-call (round) budget for the references sub-agent — the v5.1 cost fix.

The rc7 Funhouse run showed 26/76 reference-consult questions burning **69% of
all tokens** (4.6M / 6.6M), almost entirely *input* tokens: the references
sub-agent re-sends its accumulated context on every internal model call, so an
unbounded number of internal rounds compounds quadratically. rc7 capped each
*result's* size (``max_result_chars``); this caps the *number of model calls*
per consult.

Mechanism
---------
deepagents 0.6.8 ships no call-limit middleware of its own, but its
``SubAgent`` spec accepts a ``middleware`` list that ``create_deep_agent``
appends to the sub-agent's default stack — and langchain provides
:class:`~langchain.agents.middleware.ModelCallLimitMiddleware` (per-run model
call counting with a graceful ``exit_behavior="end"``).
:class:`ModelCallBudgetMiddleware` subclasses it to *degrade gracefully* in two
stages instead of cutting the consult off with a stock "limits exceeded" note:

1. **Forced final turn** — on the LAST budgeted call, ``wrap_model_call``
   strips the tools from the model request and appends a "summarize what you
   have now and answer" instruction, so the budgeted final call produces a
   real answer from the evidence already gathered.
2. **Hard backstop** — if the model STILL emits tool calls on that final turn
   (it cannot execute them meaningfully, but a misbehaving model might try),
   the inherited ``before_model`` check fires at the budget and jumps to the
   end; the injected message is replaced with a clear "budget reached —
   re-delegate with a narrower question" note instead of the stock error text.

Neither stage raises: the parent agent always receives a final message.

A budget can also warn ahead (``warn_remaining``): from that many calls
left (the one being made included) down to the last but one, each call
carries a note saying how many remain (:data:`LOW_BUDGET_NOTE`); the last
is the forced final turn. BACKLOG GM-101: calc helpers that ran out
of calls before building their PDF (live waves w8b and w13) were never told
the end was near.
"""

from typing import Optional

from langchain.agents.middleware import (AgentMiddleware,
                                        ModelCallLimitMiddleware, hook_config)
from langchain_core.messages import AIMessage, HumanMessage

#: Default model-call budget per references consult (one ``task`` delegation).
#: Picked from the rc7 run data: good reference answers complete in a few
#: rounds (median question spend was modest); the token burners were consults
#: looping well past that. Raised 8 -> 10 on 2026-10-09: in live smoke wave 1
#: (G5) 17 of 18 consults stopped at exactly 8, their forced answers listing
#: items "not searched", about one call each having gone to scratch-file
#: tools the consult cannot use (now hidden from it: :class:`HideTools`).
#: 10 calls = up to 9 tool rounds + the final answer. Measure against 12.
DEFAULT_REFERENCES_MAX_MODEL_CALLS = 10

#: Default model-call budget per ``reviewer`` delegation. The reviewer had
#: none: in a field session (2026-10-14, app 5.33.0) one delegation ran for
#: over 2.5 hours without returning. Its work is reference lookups, so it
#: gets the references consult's budget.
DEFAULT_REVIEWER_MAX_MODEL_CALLS = DEFAULT_REFERENCES_MAX_MODEL_CALLS

#: deepagents' scratch-filesystem tools a references consult has no use for:
#: it never writes, and an empty scratch space answers nothing about the
#: library. ``read_file`` is kept -- an evicted large tool result is read
#: back with it. ``delete`` is deepagents 0.7.x's (BACKLOG INF-10).
REFERENCE_HIDDEN_TOOLS = frozenset({"ls", "glob", "grep", "write_file",
                                    "edit_file", "delete", "execute"})

#: deepagents' tools no agent on the geotech page can use (BACKLOG GM-176):
#: ``execute`` is its shell, which our in-memory backend cannot run, and
#: ``delete`` (deepagents 0.7.x) works on that scratch space only. Live
#: wave w13b: F55j's calc helper called ``execute`` with no command; the
#: review page and the references helper already hid both.
UNRUNNABLE_TOOLS = frozenset({"execute", "delete"})

#: deepagents' system-prompt section about a tool, by the tool's name: it
#: goes with the tool. deepagents adds "## Execute Tool `execute`" only
#: where its backend can run a shell, so on ours there is none to take.
_TOOL_SECTIONS = {"execute": "Execute Tool"}

#: deepagents' advice, inside another tool's result, to call a tool by
#: name: it goes with the tool too. ``read_file``'s note on a cut long line
#: says "use execute(command='jq . <file>') to pretty-print it" (deepagents
#: 0.6.8 and 0.7.13).
_TOOL_ADVICE = {"execute": (r"For example, if this is JSON, use "
                            r"execute\(command='[^']*'\)[^.]*\.[ \t]*")}


def _without_sections(text: str, titles) -> str:
    """``text`` without each ``## <title>`` section (to the next heading or
    the end); unchanged when none is there."""
    import re
    out = str(text or "")
    for title in titles:
        out = re.sub(r"(?ms)^## " + re.escape(title) + r"\b.*?(?=^## |\Z)",
                     "", out)
    if out == text:
        return text
    return re.sub(r"\n{3,}", "\n\n", out)


class HideTools(AgentMiddleware):
    """Keep the named tools out of the model's view (they stay registered,
    so nothing that refers to them breaks). deepagents attaches its
    filesystem tools to every sub-agent; this takes the ones a sub-agent
    must not spend calls on off the menu, with deepagents' prompt section
    about a hidden tool where it has one, out of an unknown-tool error's
    list of tools (BACKLOG GM-176: that list names every REGISTERED tool,
    and F18j's named 7 hidden ones), and out of deepagents' advice in
    another tool's result (``read_file``'s "use execute(...)")."""

    def __init__(self, names):
        super().__init__()
        self.hidden = frozenset(names or ())

    def _filtered(self, request):
        if not self.hidden:
            return request
        tools = [t for t in (request.tools or [])
                 if (t.get("name") if isinstance(t, dict)
                     else getattr(t, "name", None)) not in self.hidden]
        request = request.override(tools=tools)
        titles = [_TOOL_SECTIONS[n] for n in sorted(self.hidden)
                  if n in _TOOL_SECTIONS]
        sm = getattr(request, "system_message", None)
        if not titles or sm is None:
            return request
        content = sm.content
        if isinstance(content, str):
            new = _without_sections(content, titles)
            if new is content:
                return request
            return request.override(
                system_message=sm.model_copy(update={"content": new}))
        if isinstance(content, list):
            blocks, changed = [], False
            for b in content:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    t = _without_sections(b["text"], titles)
                    if t is not b["text"]:
                        changed = True
                        if not t.strip():
                            continue
                        b = {**b, "text": t}
                blocks.append(b)
            if changed:
                return request.override(
                    system_message=sm.model_copy(update={"content": blocks}))
        return request

    def wrap_model_call(self, request, handler):
        return handler(self._filtered(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._filtered(request))

    def _unadvised(self, result):
        """A tool's result without deepagents' advice to call a hidden tool
        (:data:`_TOOL_ADVICE`); the very same message when it has none."""
        import re
        content = result.content
        for name in sorted(self.hidden):
            if name in _TOOL_ADVICE and f"{name}(" in content:
                content = re.sub(_TOOL_ADVICE[name], "", content)
        if content == result.content:
            return result
        return result.model_copy(update={"content": content})

    def _unlisted(self, request, result):
        """An unknown-tool error lists every REGISTERED tool; the hidden ones
        are taken out of that list too (F31's listed them as the page's). A
        real tool's result loses advice to call a hidden one."""
        import re

        from langchain_core.messages import ToolMessage
        if not isinstance(result, ToolMessage) or \
                not isinstance(result.content, str) or not self.hidden:
            return result
        if getattr(request, "tool", None) is not None:
            return self._unadvised(result)
        m = re.search(r"try one of \[([^\]]*)\]", result.content)
        if not m:
            return result
        kept = [n.strip() for n in m.group(1).split(",")
                if n.strip() and n.strip().strip("'\"") not in self.hidden]
        return result.model_copy(update={"content": (
            result.content[:m.start(1)] + ", ".join(kept)
            + result.content[m.end(1):])})

    def wrap_tool_call(self, request, handler):
        return self._unlisted(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._unlisted(request, await handler(request))

#: Injected as a user message on the last budgeted model call (with tools
#: stripped) so the sub-agent spends its final call answering, not searching.
FINAL_TURN_NUDGE = (
    "[Round budget reached] This is your FINAL turn for this consult — no "
    "more tool calls are available. Using ONLY the information already "
    "gathered above, answer the original question now: give the specific "
    "value(s)/provision asked for with the exact citation(s) "
    "(reference + section/table/figure). If something could not be "
    "confirmed, say so in one line rather than guessing."
)

#: Added (as a user message, tools kept) to each call from ``warn_remaining``
#: calls left down to the last but one; ``{left}`` counts the call it is on
#: (BACKLOG GM-101).
LOW_BUDGET_NOTE = (
    "[Model-call budget] {left} model calls left, this one included; the "
    "last is your reply, with no tools. Use them to finish and check what "
    "the task asked for; leave anything else and say in your reply what "
    "you left.")

#: Backstop final message if the model ignores the forced final turn and the
#: hard limit fires. Replaces the stock "Model call limits exceeded" text with
#: something the PARENT agent can act on.
BUDGET_EXHAUSTED_MESSAGE = (
    "The references consult reached its model-call budget before a final "
    "summary could be written, so its findings may be incomplete. If more "
    "detail is needed, delegate to the references sub-agent again with a "
    "narrower, more specific question (one value / one provision at a time)."
)


#: The model-call budget of a ``general-purpose`` helper on the geotech page
#: (field session 2026-10-14, BACKLOG TA-22: helpers ran with none; the
#: longest made about 20 calls). The last budgeted call answers from what
#: the helper has, and the helper's reading is kept in a file
#: (:mod:`funhouse_agent.deep.keep_work`). ``0`` or ``off`` = no budget.
HELPER_MAX_MODEL_CALLS_ENV = "GEOTECH_HELPER_MAX_MODEL_CALLS"
DEFAULT_HELPER_MAX_MODEL_CALLS = 30

HELPER_FINAL_NUDGE = (
    "[Model-call budget reached] This is your FINAL turn: no more tool "
    "calls are available. Answer now from what you have gathered: the data "
    "itself, in full (every value you extracted, as structured as you can), "
    "then what you did not get to and why."
)

HELPER_EXHAUSTED = (
    "The helper reached its model-call budget before it could write an "
    "answer. Give the rest of the job to a new helper with a narrower brief "
    "(fewer pages or items)."
)


def helper_max_model_calls() -> int:
    """The general-purpose helper's budget (``0`` = none)."""
    import os
    raw = (os.environ.get(HELPER_MAX_MODEL_CALLS_ENV) or "").strip().lower()
    if raw in ("off", "none", "no", "false"):
        return 0
    try:
        return max(0, int(raw)) if raw else DEFAULT_HELPER_MAX_MODEL_CALLS
    except ValueError:
        return DEFAULT_HELPER_MAX_MODEL_CALLS


class ModelCallBudgetMiddleware(ModelCallLimitMiddleware):
    """A per-run model-call budget that ends with an answer, not an error.

    Subclasses langchain's :class:`ModelCallLimitMiddleware` (which contributes
    the per-run call counting via ``after_model`` and the graceful
    jump-to-end backstop via ``before_model`` with ``exit_behavior="end"``)
    and adds the forced summarize-and-answer final turn via
    ``wrap_model_call``.

    Parameters
    ----------
    max_model_calls : int
        Budget of model calls per run (per ``task`` delegation for a
        sub-agent). Must be >= 1 — the builder simply omits this middleware to
        disable the cap.
    final_turn_nudge : str, optional
        Instruction appended (as a user message, with tools stripped) on the
        last budgeted call.
    exhausted_message : str, optional
        Final AI message injected if the hard backstop fires.
    warn_remaining : int, optional
        From this many calls left (the one being made included) down to
        the last but one, each call carries ``low_budget_note`` saying how
        many remain. ``None`` (the default), ``0`` or ``1`` gives no
        warning.
    low_budget_note : str, optional
        The warning, with ``{left}`` for the calls left (this one included).
    """

    def __init__(
        self,
        max_model_calls: int = DEFAULT_REFERENCES_MAX_MODEL_CALLS,
        *,
        final_turn_nudge: str = FINAL_TURN_NUDGE,
        exhausted_message: str = BUDGET_EXHAUSTED_MESSAGE,
        warn_remaining: Optional[int] = None,
        low_budget_note: str = LOW_BUDGET_NOTE,
    ) -> None:
        if max_model_calls < 1:
            raise ValueError(
                "max_model_calls must be >= 1 (omit the middleware to disable "
                "the budget)"
            )
        # exit_behavior="end" = degrade gracefully, NEVER raise to the user.
        super().__init__(run_limit=int(max_model_calls), exit_behavior="end")
        self.final_turn_nudge = final_turn_nudge
        self.exhausted_message = exhausted_message
        self.warn_remaining = max(0, int(warn_remaining or 0))
        self.low_budget_note = low_budget_note

    # -- forced final turn ---------------------------------------------------

    def _calls_left(self, request) -> Optional[int]:
        """Calls left in budget, the one about to be made included."""
        if self.run_limit is None:
            return None
        state = request.state or {}
        return self.run_limit - state.get("run_model_call_count", 0)

    def _is_final_budgeted_call(self, request) -> bool:
        """True when the call about to be made is the LAST one in budget.
        A ``task`` helper's record notes it: a helper whose answer was
        forced by its budget has its reading kept in a file
        (:mod:`funhouse_agent.deep.keep_work`, BACKLOG TA-21)."""
        left = self._calls_left(request)
        final = left is not None and left <= 1
        if final:
            _note_budget_reached()
        return final

    def _force_final_answer(self, request):
        """Strip tools + append the summarize-now instruction."""
        return request.override(
            tools=[],
            tool_choice=None,
            messages=[
                *request.messages,
                HumanMessage(content=self.final_turn_nudge),
            ],
        )

    def _warned(self, request):
        """``request`` with the low-budget note added from
        :attr:`warn_remaining` calls left down to the last but one."""
        left = self._calls_left(request)
        if not self.warn_remaining or left is None \
                or not 1 < left <= self.warn_remaining:
            return request
        return request.override(messages=[
            *request.messages,
            HumanMessage(content=self.low_budget_note.format(left=left)),
        ])

    def _budgeted(self, request):
        if self._is_final_budgeted_call(request):
            return self._force_final_answer(request)
        return self._warned(request)

    def wrap_model_call(self, request, handler):
        """On the last budgeted call, force a tool-less final-answer turn;
        near it, say how many calls are left."""
        return handler(self._budgeted(request))

    async def awrap_model_call(self, request, handler):
        """Async mirror of :meth:`wrap_model_call`."""
        return await handler(self._budgeted(request))

    # -- hard backstop (inherited check, friendlier message) ------------------

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """Inherited limit check, but with an actionable final message.

        The base class injects "Model call limits exceeded: run limit (N/N)"
        when it jumps to the end; as a sub-agent that text would be returned
        verbatim to the parent. Swap in :attr:`exhausted_message` so the
        parent gets a usable directive (re-delegate narrower) instead.
        """
        result = super().before_model(state, runtime)
        if isinstance(result, dict) and result.get("jump_to") == "end":
            _note_budget_reached()
            result = dict(result)
            result["messages"] = [AIMessage(content=self.exhausted_message)]
        return result


def _note_budget_reached() -> None:
    """Tell the running ``task`` helper's record that its budget ended it."""
    try:
        from funhouse_agent.deep.keep_work import note_budget_reached
        note_budget_reached()
    except Exception:  # noqa: BLE001 - a note never fails a call
        pass


__all__ = [
    "ModelCallBudgetMiddleware",
    "helper_max_model_calls",
    "HELPER_MAX_MODEL_CALLS_ENV",
    "DEFAULT_HELPER_MAX_MODEL_CALLS",
    "HELPER_FINAL_NUDGE",
    "HELPER_EXHAUSTED",
    "HideTools",
    "REFERENCE_HIDDEN_TOOLS",
    "DEFAULT_REFERENCES_MAX_MODEL_CALLS",
    "FINAL_TURN_NUDGE",
    "LOW_BUDGET_NOTE",
    "BUDGET_EXHAUSTED_MESSAGE",
]
