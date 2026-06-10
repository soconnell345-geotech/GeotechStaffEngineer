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
"""

from langchain.agents.middleware import ModelCallLimitMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage

#: Default model-call budget per references consult (one ``task`` delegation).
#: Picked from the rc7 run data: good reference answers complete in a few
#: rounds (median question spend was modest); the token burners were consults
#: looping well past that. 8 calls = up to 7 tool rounds + the final answer.
DEFAULT_REFERENCES_MAX_MODEL_CALLS = 8

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

#: Backstop final message if the model ignores the forced final turn and the
#: hard limit fires. Replaces the stock "Model call limits exceeded" text with
#: something the PARENT agent can act on.
BUDGET_EXHAUSTED_MESSAGE = (
    "The references consult reached its model-call budget before a final "
    "summary could be written, so its findings may be incomplete. If more "
    "detail is needed, delegate to the references sub-agent again with a "
    "narrower, more specific question (one value / one provision at a time)."
)


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
    """

    def __init__(
        self,
        max_model_calls: int = DEFAULT_REFERENCES_MAX_MODEL_CALLS,
        *,
        final_turn_nudge: str = FINAL_TURN_NUDGE,
        exhausted_message: str = BUDGET_EXHAUSTED_MESSAGE,
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

    # -- forced final turn ---------------------------------------------------

    def _is_final_budgeted_call(self, request) -> bool:
        """True when the call about to be made is the LAST one in budget."""
        state = request.state or {}
        run_count = state.get("run_model_call_count", 0)
        return self.run_limit is not None and run_count >= self.run_limit - 1

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

    def wrap_model_call(self, request, handler):
        """On the last budgeted call, force a tool-less final-answer turn."""
        if self._is_final_budgeted_call(request):
            request = self._force_final_answer(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        """Async mirror of :meth:`wrap_model_call`."""
        if self._is_final_budgeted_call(request):
            request = self._force_final_answer(request)
        return await handler(request)

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
            result = dict(result)
            result["messages"] = [AIMessage(content=self.exhausted_message)]
        return result


__all__ = [
    "ModelCallBudgetMiddleware",
    "DEFAULT_REFERENCES_MAX_MODEL_CALLS",
    "FINAL_TURN_NUDGE",
    "BUDGET_EXHAUSTED_MESSAGE",
]
