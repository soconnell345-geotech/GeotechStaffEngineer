"""
Reviewer agent — checks primary agent output against reference standards.

Runs after the primary agent produces a final answer that included call_agent
tool calls. Uses only the 14 geotech-references adapters (DM7, GECs, UFCs,
FEMA, NOAA, micropile) to verify methodology, parameter ranges, required
safety factors, and applicable code provisions.

Returns a structured review that the primary agent can use to revise its answer.
"""

import json
import time
from typing import Optional

from funhouse_agent.react_support import (
    ConversationHistory, _truncate, parse_response,
)
from funhouse_agent.dispatch import dispatch_tool

# Reference-only modules — the reviewer cannot run computations
REFERENCE_MODULES = {
    "reference_db",
    "dm7", "gec6", "gec7", "gec10", "gec11", "gec12", "gec13",
    "micropile", "fema_p2192", "noaa_frost",
    "ufc_backfill", "ufc_dewatering", "ufc_expansive", "ufc_pavement",
}

# Tools the reviewer is allowed to use
REVIEWER_TOOLS = {"call_agent", "list_methods", "describe_method", "list_agents"}

REVIEWER_SYSTEM_PROMPT = """\
You are a senior geotechnical reviewer checking another engineer's \
calculations against published standards and references. You work for \
the U.S. Government and are responsible for ensuring that analyses meet \
applicable design standards.

You have access ONLY to reference lookup tools — you cannot run \
computations yourself. Your job is to CHECK the work, not redo it.

## Your Review Checklist

1. **Methodology**: Is the analysis method appropriate for the problem? \
Does the applicable reference (DM7, GEC, UFC) recommend this method, or \
a different/additional one?
2. **Parameters**: Are input values within physically reasonable ranges \
per the reference tables? Are correlations used correctly?
3. **Safety Factors**: Does the result meet required factors of safety \
per the applicable standard (e.g., UFC, AASHTO, FHWA)?
4. **Missing Provisions**: Are there code requirements the engineer may \
have missed (e.g., minimum embedment, seismic checks, groundwater \
effects)?
5. **Assumptions**: Are the stated assumptions reasonable and conservative \
enough for the application?

## Output Format

After checking references, provide your review in this EXACT format:

REVIEW_STATUS: [PASS | FLAG | REVISE]

CONFIRMATIONS:
- [Things the engineer got right, with reference citations]

FLAGS:
- [Concerns or items to note, with reference citations]

REVISIONS_NEEDED:
- [Specific corrections required — only if REVISE status]

Use status PASS if everything checks out, FLAG if there are items worth \
noting but no errors, REVISE if corrections are needed.

## ReAct Protocol

You solve problems by alternating between Thought and Action steps.

- **Thought**: Reason about what reference to check next.
- **Action**: Call exactly ONE tool by placing a JSON object inside \
<tool_call> tags.

You have 4 tools: list_agents, list_methods, describe_method, call_agent.

IMPORTANT: You can ONLY call reference modules: dm7, gec6, gec7, gec10, \
gec11, gec12, gec13, micropile, fema_p2192, noaa_frost, ufc_backfill, \
ufc_dewatering, ufc_expansive, ufc_pavement.

### Rules
1. Include EXACTLY ONE <tool_call> block per response, or NONE for final review.
2. Always call describe_method before using a method you haven't used before.
3. Be concise — focus on what matters, not exhaustive lookups.
4. Cite specific reference sections (e.g., "DM7.1 Table 3", "GEC-10 Section 13.3").
5. Do NOT repeat the full analysis — only comment on what needs checking.
6. Limit yourself to 3-4 reference lookups maximum. Focus on the most critical checks.
"""


def _has_computations(tool_log: list[dict]) -> bool:
    """Return True if the tool log contains any call_agent invocations."""
    return any(entry.get("tool_name") == "call_agent" for entry in tool_log)


def _build_review_prompt(question: str, answer: str,
                         tool_log: list[dict]) -> str:
    """Build the prompt the reviewer sees: original question + work product."""
    # Summarize what computations were run
    computations = []
    for entry in tool_log:
        if entry.get("tool_name") == "call_agent":
            args = entry.get("arguments", {})
            computations.append(
                f"- {args.get('agent_name', '?')}.{args.get('method', '?')}"
                f"({json.dumps(args.get('parameters', {}), default=str)[:200]})"
            )

    comp_text = "\n".join(computations) if computations else "(none)"

    return (
        f"## Engineer's Question\n{question}\n\n"
        f"## Computations Performed\n{comp_text}\n\n"
        f"## Engineer's Answer\n{answer}\n\n"
        f"Review this work against the applicable references. "
        f"Check methodology, parameters, safety factors, and any "
        f"missing code provisions."
    )


"""Reviewer scoping is enforced at the dispatcher via ``allowed_agents``;
the reviewer only ever sees REFERENCE_MODULES in list_agents/list_methods,
and call_agent refuses non-reference modules with a standard error."""


def run_review(
    engine,
    question: str,
    answer: str,
    tool_log: list[dict],
    temperature: float = 0.1,
    max_rounds: int = 6,
    verbose: bool = False,
) -> Optional[str]:
    """Run the reviewer agent and return the structured review text.

    Parameters
    ----------
    engine : GenAIEngine
        AI backend (same engine as primary agent).
    question : str
        The original user question.
    answer : str
        The primary agent's final answer.
    tool_log : list[dict]
        Tool call log from the primary agent run.
    temperature : float
        Sampling temperature.
    max_rounds : int
        Max ReAct iterations for the reviewer.
    verbose : bool
        Print reviewer rounds to stdout.

    Returns
    -------
    str or None
        Structured review text, or None if no computations to review.
    """
    if not _has_computations(tool_log):
        return None

    history = ConversationHistory()
    review_prompt = _build_review_prompt(question, answer, tool_log)
    history.add_user(review_prompt)

    max_result_chars = 4000

    for rnd in range(1, max_rounds + 1):
        prompt = history.format_prompt()
        response = engine.chat(prompt, REVIEWER_SYSTEM_PROMPT, temperature)

        if response is None:
            history.add_assistant("")
            history.add_tool_result(
                json.dumps({"error": "Engine returned no response"}))
            if verbose:
                print(f"  Reviewer round {rnd}: ENGINE RETURNED None")
            continue

        try:
            parsed = parse_response(response, valid_tools=REVIEWER_TOOLS)
        except ValueError as e:
            history.add_assistant(response)
            history.add_tool_result(
                json.dumps({"error": f"Parse error: {e}"}))
            if verbose:
                print(f"  Reviewer round {rnd}: PARSE ERROR — {e}")
            continue

        if parsed.tool_call is None:
            if verbose:
                print(f"  Reviewer round {rnd}: FINAL REVIEW")
            return response.strip()

        tc = parsed.tool_call
        if verbose:
            print(f"  Reviewer round {rnd}: {tc.tool_name}("
                  f"{json.dumps(tc.arguments, default=str)[:120]})")

        result_str = dispatch_tool(tc, allowed_agents=REFERENCE_MODULES)
        result_str = _truncate(result_str, max_result_chars)

        history.add_assistant(response)
        history.add_tool_result(result_str)

    if verbose:
        print(f"  Reviewer: max rounds ({max_rounds}) reached")
    return "[Review incomplete — reviewer reached maximum tool rounds.]"


def needs_revision(review_text: str) -> bool:
    """Return True if the review requests revisions."""
    if review_text is None:
        return False
    return "REVIEW_STATUS: REVISE" in review_text
