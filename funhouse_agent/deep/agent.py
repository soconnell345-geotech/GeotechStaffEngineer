"""``build_deep_agent`` — the v5.0 deepagents port of GeotechAgent.

Wires the v1 tool surface and sub-agent design onto
``deepagents.create_deep_agent`` (deepagents 0.6.8):

* **Primary agent** — scoped to ``ANALYSIS_MODULES`` (its direct tool surface
  stays small, mirroring v1). It reaches the reference modules ONLY by
  delegating to the ``references`` sub-agent via the deepagents-native
  ``task`` tool — that is the v5 replacement for v1's ``consult_references``.
* **references sub-agent** — the reference librarian. Scoped to
  ``REFERENCE_MODULES``, prompted with ``reviewer.CONSULTANT_ROLE``, and
  given the ``read_reference_figure`` vision tool so it can read values off
  design charts (DM7/GEC/UFC/micropile/FEMA).
* **reviewer sub-agent** — checks the primary's work against the references.
  Scoped to ``REFERENCE_MODULES``, prompted with
  ``reviewer.REVIEWER_DEEP_PROMPT`` (the v1 prompt minus its text-tag protocol).

The real ``create_deep_agent`` signature (deepagents 0.6.8) targeted here::

    create_deep_agent(model=None, tools=None, *, system_prompt=None,
                      middleware=(), subagents=None, ...)

Sub-agents follow the ``SubAgent`` TypedDict
(``{name, description, system_prompt, tools?, model?, ...}``).
"""

from typing import Callable, Optional

from deepagents import create_deep_agent

# What both pages' builds do the same way moved to funhouse_agent.deep.assembly
# (INF-31 step 8); these names are the same objects, for the callers that
# import them from here.
from funhouse_agent.deep import assembly as _assembly
from funhouse_agent.deep.assembly import (  # noqa: F401
    _AUTO_SUMMARIZATION_TRIGGER,
    _DEFAULT_SUMMARIZATION_KEEP,
    _GENERAL_PURPOSE_SPEC,
    _SUMMARIZATION_FRACTION,
    _SUMMARIZATION_TOKENS_FALLBACK,
    _TodoMiddleware,
    HELPER_PARTIAL_RULE,
    MEMORIES_AGENTS_FILE,
    MEMORIES_ROUTE,
    GeotechSummarizationMiddleware,
    _join_prompts,
    _memories_namespace,
    _merge_tools,
    _model_has_token_profile,
    _resolve_summarization_trigger,
    _save_tool,
    build_memory_backend,
    build_summarization_middleware,
)

from funhouse_agent.dispatch import ANALYSIS_MODULES, REFERENCE_MODULES
from funhouse_agent.reviewer import CONSULTANT_ROLE, REVIEWER_DEEP_PROMPT
from funhouse_agent.system_prompt import (REFERENCE_MODULES_CLAUSE,
                                          REFERENCES_HELPER_CLAUSE)

from funhouse_agent.deep.limits import (
    DEFAULT_REFERENCES_MAX_MODEL_CALLS,
    DEFAULT_REVIEWER_MAX_MODEL_CALLS,
    HELPER_EXHAUSTED,
    HELPER_FINAL_NUDGE,
    JUDGMENT_REFERENCES_MAX_MODEL_CALLS,
    JUDGMENT_REFERENCES_WARN_REMAINING,
    REFERENCE_HIDDEN_TOOLS,
    UNRUNNABLE_TOOLS,
    HideTools,
    ModelCallBudgetMiddleware,
    helper_max_model_calls,
)
from funhouse_agent.deep.busy_retry import AgentBusyRetry
from funhouse_agent.deep.scratch_guard import ScratchFilesystemGuard
from funhouse_agent.deep.unknown_tool import UnknownToolHint
from funhouse_agent.deep.delivered_log import DeliveredToolResults
from funhouse_agent.deep.keep_work import HelperRecorder, KeepWork
from funhouse_agent.deep.result_map import LargeResultKeyMap
from funhouse_agent.deep.tool_guards import (HideScratchFilesystem,
                                             OutputLimitGuard, ToolErrorGuard,
                                             auth_retry)
from funhouse_agent.deep.calculate_tool import make_calculate_tool

from funhouse_agent.deep.prompt import (
    REPORT_INGEST_NUDGE,
    REPORT_LIBRARY_NUDGE,
    build_domain_prompt,
    judgment_calls_section,
)
from funhouse_agent.deep.setup_agent import build_setup_subagent
from funhouse_agent.deep.tools import (
    DEFAULT_MAX_RESULT_CHARS,
    GEOTECH_VISION_TOOLS,
    make_core_tools,
    make_vision_tools,
)
from funhouse_agent.deep.vision_engine import LangChainVisionEngine


#: Concision instruction appended to the references sub-agent's framing. The
#: sub-agent's reference-text dumps are the main token-bloat source, so it is
#: told to return only the asked-for value(s) plus the citation rather than
#: pasting long chapter passages — this shrinks both its returned text and what
#: the primary then re-reads each round.
_REFERENCES_CONCISION = (
    "\n\nBe concise. Return ONLY the specific value(s)/provision asked for "
    "with the exact citation (reference + section/table/figure). Do NOT paste "
    "long chapter-text passages or restate full sections — a few sentences "
    "plus the citation is ideal."
)

#: Added after the concision rule with ``GEOTECH_JUDGMENT_CALLS`` on (BACKLOG
#: GM-116): one value where the references give several hides a choice that
#: is the user's to make.
_REFERENCES_OPTIONS = (
    "\n\nWhere the references give more than one method or value for the "
    "question, return each as a cited option with when it applies, and say "
    "which fits the question as asked; the choice is the user's. Otherwise "
    "return only what was asked. Asked about several choices at once, "
    "answer each in a line or two with its citation, or say no reference "
    "was found for it."
)

#: Added after the options rule where the main agent asks the consult about
#: its judgment calls (``GEOTECH_JUDGMENT_CALLS`` on the geotech page; BACKLOG
#: GM-187): a line per call, and the two lookups that find a section, so its
#: tighter budget goes on lookups. In live waves w13b and w14 each such
#: consult spent one to three of its calls listing the library's modules and
#: methods before the first search. A search per call: ``reference_search``
#: matches every plain word of its query, so one search over several calls
#: finds little.
_REFERENCES_JUDGMENT_CALLS = (
    "\n\nAsked about a list of judgment calls, answer each in one line: the "
    "reference and the section, table, figure or page your lookups returned "
    "that supports the choice or its alternative, or \"no reference "
    "found\". Look up with `call_agent` on `reference_db`: "
    "`reference_search` (`query`, optional `reference`, `chapter`, `limit`), "
    "one per judgment call, several at once, then `reference_get` "
    "(`reference`, `section_id`) for a section you cite; there is no need "
    "to list the modules or methods first."
)

#: The consult's model-call budget, stated to it (live smoke wave 1, G5).
_REFERENCES_BUDGET = (
    " You have {n} model calls for this consult, the last of which must be "
    "your answer: make independent lookups in the SAME turn (several tool "
    "calls at once) rather than one per turn."
)


# ---------------------------------------------------------------------------
# Sub-agent descriptions (what the primary uses to decide when to delegate)
# ---------------------------------------------------------------------------

_REFERENCES_DESCRIPTION = (
    "The geotechnical reference librarian. Delegate to it whenever you need a "
    "code provision, a recommended method, a parameter range, a required "
    "factor of safety, or a value read off a design chart. It searches the "
    "design references (NAVFAC DM7, the FHWA GEC series, UFC, micropile, FEMA, "
    "NOAA — chapter text, tables, equations, and figure charts), cites the "
    "specific reference and section/table/figure number, and reads chart "
    "values off the actual figures. You do NOT have the reference modules "
    "directly — delegating to this sub-agent is the only way to reach them."
)

_REVIEWER_DESCRIPTION = (
    "A senior geotechnical reviewer. Delegate to it only AFTER you have "
    "produced a draft answer that involved calculations, to check the methodology, "
    "parameter ranges, required factors of safety, and any missing code "
    "provisions against the published references (DM7, GECs, UFCs, FEMA). It "
    "returns a structured PASS / FLAG / REVISE review with citations; it does "
    "not run computations itself. It sees only what you write in its brief: "
    "it cannot open documents, pages or files, so it cannot check extracted "
    "values against a source."
)


def build_references_subagent(
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    max_model_calls: Optional[int] = DEFAULT_REFERENCES_MAX_MODEL_CALLS,
    extra_tools=None,
    extra_system_prompt: Optional[str] = None,
    judgment_consult: bool = False,
) -> dict:
    """Build the ``references`` sub-agent spec (reference librarian).

    Scoped to ``REFERENCE_MODULES`` core tools plus the ``read_reference_figure``
    vision tool (so it can read values off design charts). The vision tool
    needs an engine to actually call the model; offline it returns a clear
    "vision not available" error rather than raising.

    Parameters
    ----------
    engine, attachments, save_fn
        Forwarded to :func:`~funhouse_agent.deep.tools.make_vision_tools` for
        the figure read-off tool.
    max_result_chars : int, optional
        Tool-result size cap forwarded to ``make_core_tools`` /
        ``make_vision_tools`` (default ``8000``). This bounds what is re-sent
        each round. The sub-agent's framing is also extended with a concision
        instruction so it returns only the asked-for value(s) plus the
        citation.
    reference_result_chars : int, optional
        LARGER cap for reference reads (``call_agent`` on a reference module +
        ``read_reference_figure``) — the reference text is the payload.
        Defaults to ``DEFAULT_REFERENCE_RESULT_CHARS`` (16000); see
        :func:`~funhouse_agent.deep.tools.make_core_tools`.
    max_model_calls : int, optional
        The v5.1 round budget — THE key cost lever (26/76 reference questions
        burned 69% of all tokens in rc7 because the sub-agent re-sent its
        accumulated context across unbounded internal model calls). Attaches a
        :class:`~funhouse_agent.deep.limits.ModelCallBudgetMiddleware` to the
        sub-agent so each consult gets at most this many model calls; the last
        budgeted call is forced to summarize-and-answer from what was already
        gathered (graceful — a final answer is always returned, never an
        error). Defaults to ``8``; ``None``/``0`` disables the budget.
    extra_tools, extra_system_prompt
        Appended to the sub-agent's tools and prompt (the web app hands it
        ``record_feedback`` — a figure whose source PDF is missing is found
        HERE). Defaults leave the spec unchanged.
    judgment_consult : bool, optional
        The main agent's prompt asks it to consult this helper about its
        judgment calls (``GEOTECH_JUDGMENT_CALLS`` on the geotech page; see
        :func:`build_deep_agent`). With the switch on, the budget is then at
        most :data:`~funhouse_agent.deep.limits.JUDGMENT_REFERENCES_MAX_MODEL_CALLS`,
        warned two calls ahead, and the prompt asks for one line per call
        and names the two lookups (BACKLOG GM-187). Off by default: the spec
        as before.
    """
    tools = make_core_tools(
        allowed_agents=REFERENCE_MODULES,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
        attachments=attachments,
    ) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        include={"read_reference_figure"},
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    ) + [make_calculate_tool()]
    if extra_tools:
        tools = list(tools) + list(extra_tools)
    system_prompt = CONSULTANT_ROLE + _REFERENCES_CONCISION
    from funhouse_agent import review_flags as _flags
    warn = None
    if _flags.judgment_calls():
        system_prompt += _REFERENCES_OPTIONS
        if judgment_consult:
            system_prompt += _REFERENCES_JUDGMENT_CALLS
            if max_model_calls:
                max_model_calls = min(int(max_model_calls),
                                      JUDGMENT_REFERENCES_MAX_MODEL_CALLS)
                warn = JUDGMENT_REFERENCES_WARN_REMAINING
    if max_model_calls:
        system_prompt += _REFERENCES_BUDGET.format(n=int(max_model_calls))
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    spec = {
        "name": "references",
        "description": _REFERENCES_DESCRIPTION,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    # The consult never writes and an empty scratch space answers nothing:
    # deepagents' ls/glob/grep cost about one call a consult in live smoke
    # wave 1 (G5), so they are kept off its menu.
    middleware = [HideTools(REFERENCE_HIDDEN_TOOLS)]
    if max_model_calls:
        # deepagents appends SubAgent-spec middleware to the sub-agent's
        # default stack (see create_deep_agent's subagent processing).
        middleware.insert(0, ModelCallBudgetMiddleware(
            max_model_calls, warn_remaining=warn))
    spec["middleware"] = middleware
    return spec


# ---------------------------------------------------------------------------
# A2: calc sub-agent — context isolation for tool-heavy calculation work
# ---------------------------------------------------------------------------
# Measured (module_work/A2_CONTEXT_DESIGN.md): the ~10k/turn of calc-package /
# method-dump / reference output is what makes a persist+replay chat grow ~84%
# more expensive over 10 turns. Delegating the tool-heavy calc to a sub-agent
# keeps that bulky trace in the sub-agent's own context; only a compact result
# (values + units + method + saved-file path) returns to the main thread — which
# is what replays forward. NO DATA LOSS: the sub-agent saves the full payload to
# a file, so the detail stays retrievable on demand.

#: The calc sub-agent's model-call budget (a calc chain — multiple methods +
#: a package — runs longer than a reference lookup, so a bit above the 8 that
#: bounds the references consultant). ``None``/``0`` disables the budget.
DEFAULT_CALC_MAX_MODEL_CALLS = 16

#: The calc helper's budget, stated to it as the references helper's is
#: (BACKLOG GM-101; live wave w13: both calc helpers ran out of calls, one
#: before html_to_pdf, one before its PDF, and were never told the budget).
_CALC_BUDGET = (
    "\nCALL BUDGET: you have {n} model calls for this task, the last of "
    "which must be your reply. Make independent calls in the SAME turn "
    "(several tool calls at once). Build the deliverable and check it "
    "(html_to_pdf's page_contents, read_pdf_text for its numbers) with "
    "calls to spare.")

#: The calc helper is told how many calls are left from this many calls
#: left (that call included) down to its last but one (BACKLOG GM-101).
CALC_WARN_REMAINING = 4

_CALC_DESCRIPTION = (
    "The calculation engine. Delegate TOOL-HEAVY calculation work to it: running "
    "an analysis method (bearing capacity, settlement, slope stability, FEM, "
    "liquefaction, downdrag, …), a parameter sweep, or building a calc package. "
    "It runs the method(s), returns the KEY numeric results (governing value(s) + "
    "units + the method used + any FoS/utilization) and the saved calc-package / "
    "plot path, and keeps the bulky intermediate tool output in its OWN context "
    "instead of yours — so a long chat does not carry every calc dump forward."
)

#: The calc sub-agent's OWN preamble. Until 2026-09-11 the spec prefixed
#: ``_CALC_FRAMING`` with ``reviewer.CONSULTANT_FRAMING`` — the reference-
#: LIBRARIAN text ("Do NOT perform engineering calculations — reference lookup
#: only ... Question: ") — so the agent that builds every calc package was
#: first told not to calculate, then given a compactness brief, and never the
#: figure rules (those live in the PRIMARY prompt, which deepagents does not
#: pass down to sub-agents). Owner feedback: "calc packages tend to lack
#: figures". This preamble names what the calc agent actually has.
_CALC_PREAMBLE = (
    "You are the CALCULATION ENGINE of a geotechnical agent. You have the "
    "analysis modules (call_agent / list_methods / describe_method), the "
    "`calc_package` module (canned *_package reports, render_figures, "
    "render_tables, html_to_pdf), the `profile_figure` module (subsurface_profile schematic, "
    "plot_data data plots), the reference modules' equation, table and chart "
    "functions (DM7, GEC, UFC, micropile...), the `calculate` tool for a "
    "stated formula no method computes, save_file, and read access to the "
    "working folder "
    "(list_files, read_pdf_text, read_text_file). You run the numbers AND "
    "build the "
    "deliverable — report and figures — for the delegating agent."
)

#: The figure rules for a deliverable, in the calc agent's own prompt.
_CALC_FIGURES = (
    "FIGURES ARE PART OF EVERY DELIVERABLE. A calc package or report with no "
    "figure is incomplete unless the analysis genuinely has nothing to draw. "
    "Default content, in this order: (1) a SUBSURFACE PROFILE whenever you "
    "have layers — call_agent('profile_figure', 'subsurface_profile', "
    "{layers, water table, fill/surcharge, the foundation, callouts such as "
    "the neutral plane}) and paste the returned html_img_tag; (2) the "
    "ANALYSIS'S OWN FIGURES — a canned *_package already includes them (slope "
    "section + trial-surface map, p-y curves, settlement plots, wall pressure "
    "diagrams, pavement charts), so PREFER the canned package when one fits; "
    "for a bespoke html_to_pdf report get the same figures as PNGs with "
    "call_agent('calc_package', 'render_figures', {package: '<x>_package', "
    "...that package's inputs}) and paste each html_img_tag; (3) PLOTS OF "
    "DATA — SPT/CPT vs depth, settlement vs time, load vs displacement, a "
    "sensitivity sweep or method comparison — with call_agent("
    "'profile_figure', 'plot_data', {series: [{x, y, label}], xlabel, ylabel, "
    "depth_axis}). Report skeleton for a bespoke report: Inputs -> Subsurface "
    "profile figure -> Method -> Results (with result figures and data plots) "
    "-> Checks. Every figure is a saved PNG whose html_img_tag html_to_pdf "
    "embeds for you; never write '[image]' or an inline <svg> (html_to_pdf "
    "refuses those), nor a coloured table standing in for a figure "
    "(html_to_pdf does not check for one, and it draws no backgrounds, so "
    "the colours never reach the PDF). If the "
    "figure the task needs cannot be made with these tools, say so in the "
    "reply and record it with record_feedback when that tool is available."
)

_CALC_FRAMING = (
    "\n\nRun the requested analysis method(s) with the given inputs, then "
    "reply with a COMPACT result the delegating agent can carry forward "
    "WITHOUT re-running the calc: the governing value(s) with units, the "
    "method/standard used, the key inputs, any factor of safety / "
    "utilization, and the saved paths (report AND each figure). Do NOT paste "
    "the full method dump or calc-package text into your reply.\n"
    + _CALC_FIGURES + "\n"
    "NO DATA LOSS: whenever you produce a large result (a calc package, a full "
    "method dump, a big table, a plot), SAVE it to a file — pass an output_path to "
    "the tool, or use save_file — and include the saved path in your reply, so the "
    "full detail can be re-read on demand. Never discard a result you were asked "
    "to compute: summarize it and point to the saved file.\n"
    "REAL DISK ONLY for deliverables: any built-in scratch filesystem "
    "(write_file/read_file/ls) is a SANDBOX — files written there are NOT on the "
    "real disk and the user can never see them. Every file meant for the user MUST "
    "be produced by a tool that returns a verified real path: pass output_path to "
    "the analysis/package tool, or use save_file. If no canned *_package method "
    "fits the analysis, compose the report as self-contained HTML yourself and "
    "render it with calc_package method html_to_pdf (figures as the "
    "html_img_tag of a saved PNG, or a base64 PNG/JPEG data URI — not SVG); "
    "save_file the HTML too if an HTML copy is wanted.\n"
    "NEVER WRITE PLACEHOLDERS: every number, table row and label in a "
    "deliverable comes from a tool result or from the delegation text. If you "
    "are asked to rebuild, reformat or reuse earlier results and the numbers "
    "are not in the delegation, open the earlier file first (read_text_file "
    "for HTML/TXT/CSV, read_pdf_text for a PDF). If you still do not have "
    "them, stop and reply saying exactly what is missing — never write 'per "
    "prior analysis', 'see previous table' or labels you made up, and never "
    "describe a case differently from how it was defined.\n"
    "GIVEN OR ASSUMED: a deliverable keeps the inputs the delegation says "
    "the user gave apart from the assumptions (the delegation's, yours and "
    "a tool's defaults_used). A value the user's words only imply is "
    "assumed. The delegation's assumptions and choices are the delegating "
    "agent's, not the user's: give each assumption its engineering basis, "
    "or \"assumed, not given\", never \"stated by the engineer\". Never "
    "print an assumption as given. A *_package takes `assumptions` for the "
    "inputs you pass that the user did not give.\n"
    "LAYER NAMES: keep soil layer names exactly as the delegation gives them. "
    "When asked to change a property of named layers, change only those, and "
    "say in your reply which layers (name and depth range) you changed.\n"
    "SOURCE DOCUMENTS: you HAVE read access to the working folder — list_files "
    "shows what the user supplied, read_pdf_text reads a PDF and read_text_file "
    "a text file (the scratch read_file sees neither). When a delegation "
    "references source documents (a profile PDF, a reference report), CONSULT "
    "them before writing any provenance statement. NEVER write that sources "
    "were unavailable without having run list_files to check; if a document "
    "truly is not there, name what you looked for and where."
)

#: Added to the calc helper's prompt when ``plot_code`` is offered.
_CALC_PLOT_CODE = (
    "\nA FIGURE NONE OF THESE DRAW (a wall with its pressure diagrams, "
    "panels sharing an elevation axis, a section with layers filled): "
    "write it "
    "with plot_code (see its description), look at the PNG with "
    "analyze_image (what it shows against the data, then the lettering), "
    "redraw at most twice, then paste its html_img_tag.")

#: Added to the calc helper's prompt behind the figure switches (BACKLOG
#: GM-190; live wave w14 TRACE_REVIEW §8 fix 10): a look the helper made at
#: a page of its own PDF found a figure it could not read, and the PDF was
#: handed back as it was, with the fault named. Each draw's ``next`` asks
#: for a redraw on what a look at the figure finds; nothing asked the same
#: of a look at the built report.
_CALC_LOOK_FIXES = (
    "\nAFTER A LOOK: when a look at a figure, or at a page of a PDF you "
    "built, finds something a reader could not read or would read wrongly, "
    "fix it before handing back: redraw the figure or give it the page "
    "width, build the report again and look again, at most twice. Hand "
    "back a fault only when these tools cannot fix it, and name it in your "
    "reply.")

#: Added to the calc helper's prompt with ``GEOTECH_JUDGMENT_CALLS`` on
#: (BACKLOG GM-116): its compact reply left out the choices it made, so the
#: main agent had none to relay. GM-155 (live wave w12): no call carried a
#: reference, and the lists padded with rounding and how totals are
#: reported; a tool's ``choices`` already carry a basis. Wave w13: the
#: helper's "rests on judgment" reached the answer as it stood, and the
#: delegating agent looked nothing up; a call with no basis now says so,
#: for the delegating agent's one consult.
_CALC_JUDGMENT_CALLS = (
    "\nJUDGMENT CALLS: the engineering judgment is the user's, and the "
    "delegating agent puts your choices in front of them. End your reply "
    "with a short \"Judgment calls\" list of every engineering choice the "
    "delegation did not make for you, a tool's defaults and choices "
    "included where the result turns on them -- a method or analysis "
    "type, a parameter value, factor of "
    "safety or load factor, drained or undrained, which stratum governs, "
    "search or mesh settings, how you read ambiguous data -- each giving "
    "what you chose, why (the basis the tool's `choices` give, by name and "
    "the section or page it gives, else \"no basis in hand\"), the "
    "alternative you set "
    "aside and what would change the result (a number only from a tool "
    "run). Leave out rounding and presentation.")

#: Added after ``_CALC_JUDGMENT_CALLS`` (BACKLOG GM-181; GM-155 w13b §8
#: fixes 2 and 10): a package's Given Inputs hold only what its call names
#: in ``given`` (GM-167), so the helper passes the brief's GIVEN block
#: there; and a bespoke calc PDF printed none of the consult's citations.
_CALC_GIVEN_AND_CITATIONS = (
    "\nGIVEN IN A PACKAGE: pass a *_package `given` naming each input the "
    "delegation says the user gave, with the user's words as the "
    "delegation quotes them ({input name: the user's words}); a value "
    "worked out from those words, or chosen, goes in `assumptions` with "
    "its basis.\n"
    "CITATIONS: print each citation the delegation passes beside the "
    "method or assumption it supports: in a report you compose "
    "(html_to_pdf), next to it; in a *_package, in that assumption's "
    "basis.")

#: Appended to the PRIMARY agent's system prompt when the calc sub-agent is on,
#: nudging it to delegate tool-heavy calc rather than run those tools inline.
_CALC_DELEGATION_NUDGE = (
    "CONTEXT DISCIPLINE: a `calc` sub-agent is available. For any tool-heavy "
    "calculation — running an analysis method, a parameter sweep, or building a "
    "calc package — DELEGATE it to `calc` instead of running those tools yourself, "
    "so the bulky intermediate output stays out of this conversation. `calc` "
    "returns the key values + units + method and the saved file path; carry those "
    "forward and do the reasoning and final synthesis here. (A prior calc's values "
    "stay in this conversation, so re-use them rather than recomputing.) When "
    "you delegate a calc package or report, PASS THE DATA THE FIGURES NEED — "
    "the layer stack with names/thicknesses/properties, the water table, the "
    "foundation geometry, any depth-wise data (SPT, CPT, su) — and SAY which "
    "figures you expect (subsurface profile, the analysis figures, data plots). "
    "The calc agent draws only what it is handed; a report delegated as "
    "'q_ult for B=2 m, phi=30' comes back without a profile figure because it "
    "was never given the profile. "
    "The calc agent has NO memory of this conversation: a redo, rebuild or "
    "reformat must carry the numbers themselves (paste the table) or the real "
    "path of the earlier file. When it reports a rebuilt deliverable, open the "
    "file (read_pdf_text) and check the numbers are in it before you tell the "
    "user it is fixed. When you ask for a change to named soil layers, pass the "
    "layer list with those names and say exactly which layers to change. "
    "Keep what the user gave apart from what you assumed: a value the "
    "user's words only imply is assumed. An assumed value goes in the "
    "brief under assumptions, with its engineering basis, never among the "
    "user's inputs, and never reads as something found or measured."
)

#: Added to the delegation nudge with ``GEOTECH_JUDGMENT_CALLS`` on (BACKLOG
#: GM-181; GM-155 w13b §8 fix 2): a calc package's Given Inputs hold only
#: the inputs its call names in ``given`` (GM-167), so the brief carries the
#: user's own words for the helper to pass there.
_CALC_DELEGATION_GIVEN = (
    " In a brief to `calc`, a GIVEN block quotes the user's own words for "
    "each input they stated, and asks that a calc package pass those "
    "inputs as `given` ({input name: the user's words}); a value you "
    "derived from those words, or chose, goes under assumptions with its "
    "basis, for the package's `assumptions`.")

#: And where the references helper is built (w13b §8 fix 10): a bespoke
#: calc PDF printed only the request's and the tools' citations, because
#: the consult ran beside the delegation and the brief never carried its
#: citations.
_CALC_DELEGATION_CITATIONS = (
    " For a deliverable that rests on judgment calls you can see coming (a "
    "method, a value the user did not give), ask the `references` helper "
    "before you delegate it, and pass its citations in the brief, each "
    "beside the method or assumption it supports.")


def calc_delegation_nudge(judgment_calls: bool = False,
                          references: bool = True) -> str:
    """The primary's nudge to delegate to ``calc``; with the judgment calls
    on (``GEOTECH_JUDGMENT_CALLS``, the geotech prompt) it also asks for the
    brief's GIVEN block and ``given``, and, where the references helper is
    built, for the consult's citations in the brief."""
    text = _CALC_DELEGATION_NUDGE
    if judgment_calls:
        text += _CALC_DELEGATION_GIVEN
        if references:
            text += _CALC_DELEGATION_CITATIONS
    return text


def build_calc_subagent(
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    allowed_agents=None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    max_model_calls: Optional[int] = DEFAULT_CALC_MAX_MODEL_CALLS,
    extra_tools=None,
    extra_system_prompt: Optional[str] = None,
) -> dict:
    """Build the ``calc`` sub-agent spec (A2 context isolation).

    ``extra_tools`` / ``extra_system_prompt`` are appended to the sub-agent's
    tools and prompt (the web app hands it the ``record_feedback`` tool —
    "there is no tool for this" is most often discovered HERE, while building
    a deliverable). Defaults leave the spec unchanged.

    Scoped to the analysis modules (``allowed_agents``, default
    :data:`ANALYSIS_MODULES`) plus ``save_file`` so it can persist the full
    payload. Mirrors :func:`build_references_subagent`: a concise, no-data-loss
    framing plus a :class:`~funhouse_agent.deep.limits.ModelCallBudgetMiddleware`
    budget. The sub-agent runs the tool-heavy calc in its OWN context and returns
    only a compact result (values + units + method + saved-file path) to the
    delegating agent.
    """
    allowed = frozenset(ANALYSIS_MODULES if allowed_agents is None
                        else allowed_agents)
    if allowed == frozenset(ANALYSIS_MODULES):
        # The full agent's calc engine also runs the reference modules'
        # equation, table and chart FUNCTIONS (DM7, GEC, UFC, micropile...):
        # in live smoke wave 1 (G3) it could not reach gec8's downdrag
        # equation or the micropile tables and hand-calculated instead. The
        # two pure search modules stay with the references librarian.
        allowed = allowed | (frozenset(REFERENCE_MODULES)
                             - {"reference_db", "figure_db"})
    tools = make_core_tools(
        allowed_agents=allowed,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
        attachments=attachments,
    ) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        # save_file for persisting the calc package, PLUS read access to the
        # working folder. Field feedback 2026-09-04 (a downdrag session): with
        # save_file alone the calc sub-agent could not open the source PDFs
        # the user supplied and wrote "source PDFs weren't available" into a
        # delivered report — a false provenance claim. Isolation is about
        # keeping bulky OUTPUT out of the primary context, not blinding the
        # sub-agent to the job's inputs.
        # write_xlsx: a sweep or a results table as a spreadsheet (it is
        # only built where openpyxl imports). analyze_image: the helper that
        # draws the figures can look at them (field session 2026-10-15: it
        # tried read_file on its PNG, was refused, and the figure went out
        # unseen). analyze_pdf_page: and at the pages of the PDF it builds
        # (GM-175; wave w13b F55j: asked to look at its pages, it had no
        # page tool and called execute, then analyze_image on the PDF).
        include={"save_file", "list_files", "read_pdf_text", "read_text_file",
                 "write_xlsx", "analyze_image", "analyze_pdf_page"},
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    ) + [make_calculate_tool()]
    system_prompt = _CALC_PREAMBLE + _CALC_FRAMING
    if max_model_calls:     # the budget, stated (GM-101; unswitched)
        system_prompt = system_prompt + _CALC_BUDGET.format(
            n=int(max_model_calls))
    # Plotting code in the locked box, behind GEOTECH_PLOT_CODE (GM-98; the
    # owner on calc packages' figures, 2026-10-15).
    from funhouse_agent.deep.calc_figures import (CALC_REDRAW_NOTE,
                                                  FigureLooks)
    from funhouse_agent.deep.figure_helper import _plot_code_tool
    description = _CALC_DESCRIPTION
    plot_code = _plot_code_tool()
    if plot_code is not None:
        tools = list(tools) + [plot_code]
        system_prompt = system_prompt + _CALC_PLOT_CODE
        description = description + (
            " It draws the package's figures too, in code (plot_code) where "
            "plot_data cannot: a wall or pile to scale with its diagrams "
            "beside it, panels sharing one elevation axis."
            + CALC_REDRAW_NOTE)
    from funhouse_agent import review_flags as _flags
    figure_looks = _flags.figure_helper() or _flags.plot_code()
    if figure_looks:
        system_prompt = system_prompt + _CALC_LOOK_FIXES
    if _flags.judgment_calls():
        system_prompt = (system_prompt + _CALC_JUDGMENT_CALLS
                         + _CALC_GIVEN_AND_CITATIONS)
    if extra_tools:
        tools = list(tools) + list(extra_tools)
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    spec = {
        "name": "calc",
        "description": description,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    if max_model_calls:
        spec["middleware"] = [ModelCallBudgetMiddleware(
            max_model_calls, warn_remaining=CALC_WARN_REMAINING)]
    if figure_looks:
        # The figures a module method draws ask for a look, as plot_code's
        # do, and html_to_pdf names the ones no look read (live wave w12,
        # F18j: five figures in a package, no look; GM-151); behind the
        # figure switches. build_deep_agent moves it just inside
        # DeliveredToolResults.
        spec["middleware"] = list(spec.get("middleware") or []) + [
            FigureLooks()]
    return spec


def build_reviewer_subagent(
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    extra_tools=None,
    extra_system_prompt: Optional[str] = None,
    max_model_calls: Optional[int] = DEFAULT_REVIEWER_MAX_MODEL_CALLS,
) -> dict:
    """Build the ``reviewer`` sub-agent spec.

    Scoped to ``REFERENCE_MODULES`` core tools (reference lookup only — the
    reviewer checks work, it does not recompute).

    Parameters
    ----------
    max_model_calls : int, optional
        Per-delegation model-call budget; the last call answers from what
        was gathered. ``None``/``0`` disables it.
    max_result_chars : int, optional
        Tool-result size cap forwarded to ``make_core_tools`` (default
        ``8000``), so the reviewer's reference lookups are bounded too.
    reference_result_chars : int, optional
        Larger cap for the reviewer's reference reads; see
        :func:`build_references_subagent`.
    extra_tools, extra_system_prompt
        Appended to the sub-agent's tools and prompt (e.g. the web app's
        ``record_feedback``). Defaults leave the spec unchanged.
    """
    tools = make_core_tools(
        allowed_agents=REFERENCE_MODULES,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    )
    if extra_tools:
        tools = list(tools) + list(extra_tools)
    system_prompt = REVIEWER_DEEP_PROMPT
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    spec = {
        "name": "reviewer",
        "description": _REVIEWER_DESCRIPTION,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    if max_model_calls:
        spec["middleware"] = [ModelCallBudgetMiddleware(max_model_calls)]
    return spec


def build_primary_tools(
    allowed_agents=None,
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    markup_author: Optional[str] = None,
    description_overrides: Optional[dict] = None,
    description_rewrites: Optional[dict] = None,
    plot_reference_charts: bool = True,
    geotech_vision_tools: bool = True,
) -> list:
    """Build the primary agent's tool list.

    Core dispatch tools scoped to ``allowed_agents`` (defaults to
    ``ANALYSIS_MODULES``) plus the vision/file-output tools. The reference
    modules are intentionally absent — the primary reaches them via the
    ``references`` sub-agent (the ``task`` tool).

    Parameters
    ----------
    max_result_chars : int, optional
        Tool-result size cap forwarded to ``make_core_tools`` /
        ``make_vision_tools`` (default ``8000``, mirroring v1).
    reference_result_chars : int, optional
        Larger cap for reference reads (here only ``read_reference_figure``,
        since the primary's ``call_agent`` scope excludes the reference
        modules); see :func:`~funhouse_agent.deep.tools.make_core_tools`.
    description_overrides : dict, optional
        ``{tool name: description}`` for the vision/file tools, forwarded to
        ``make_vision_tools`` (the Document Review page's wording).
    description_rewrites, plot_reference_charts
        Forwarded to ``make_vision_tools``: phrases swapped inside built
        descriptions, and whether ``plot_data`` offers the
        soil-classification charts (the Document Review page: no; BACKLOG
        RI-34).
    geotech_vision_tools : bool, optional
        Whether the geotech page's two reference readers
        (:data:`~funhouse_agent.deep.tools.GEOTECH_VISION_TOOLS`) are added
        to the vision kit's tools (the Document Review page: no; DR-04).
    """
    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES
    # An EMPTY scope (the document-review page) means no analysis modules at
    # all, so the four dispatch tools are left out rather than offered over
    # an empty catalog — an agent told it can `list_agents` and finding none
    # spends turns discovering that.
    core_tools = make_core_tools(
        allowed_agents=allowed_agents,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
        attachments=attachments,
    ) if allowed_agents else []
    from funhouse_agent import review_flags as _flags
    return core_tools + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
        markup_author=markup_author,
        description_overrides=description_overrides,
        description_rewrites=description_rewrites,
        plot_reference_charts=plot_reference_charts,
        added_tools=GEOTECH_VISION_TOOLS if geotech_vision_tools else None,
        # behind the figure switches the main agent's own redraw can retire
        # a figure any helper drew this turn (GM-151)
        plot_data_replaces=_flags.figure_helper() or _flags.plot_code(),
    ) + [make_calculate_tool()]


#: The model calls one report-ingest delegation may spend, every pass, reader
#: and bound report included, enforced inside its graph
#: (``report_ingest.subagent.CallCeiling``): deepagents uses a
#: CompiledSubAgent's runnable as provided and never reads a spec's
#: middleware for one (BACKLOG RI-02). Above the 60-897 calls a whole report
#: took on Foundry; ``report_ingest.graph.Budgets`` still caps each reader.
DEFAULT_REPORT_INGEST_MAX_MODEL_CALLS = 1000

#: The library sub-agent's ceilings, both enforced in its graph for the same
#: reason: queries per answer, and model calls, the last made without tools.
DEFAULT_REPORT_LIBRARY_MAX_TOOL_CALLS = 8
DEFAULT_REPORT_LIBRARY_MAX_MODEL_CALLS = 10


def _report_ingest_engine_factory(engine, model=None):
    """A factory for the ingest's engine on whatever ``engine`` holds: the
    page's own chat model (``report_ingest.engine.ChatModelEngine``, how it
    runs on Foundry; BACKLOG RI-03) or a live Prompter.

    Called once per ingest run so each run gets its own cost meter. Returns
    None when there is nothing to read a report with, and the sub-agent then
    says so rather than reading it on something else. ``model`` names a tier
    (the page voter's cheap one), as the tool asks; a chat model is not one,
    and the voter then runs on the ingest's own engine.
    """
    def factory():
        from report_ingest.engine import engine_for
        return engine_for(engine, model=model)
    return factory


def _resolve_ingest_source(source, attachments):
    """An attachment key or a path, resolved the way the document tools do."""
    from funhouse_agent import document_tools as _document_tools
    return _document_tools.resolve_document_source(source, attachments)


def _resolved_or_none(source, attachments):
    """The readable file behind an ingest source, or None when it cannot be
    resolved (the folder name then falls back to the file name alone)."""
    try:
        return _resolve_ingest_source(source, attachments)
    except Exception:                                # noqa: BLE001
        return None


def build_deep_agent(
    model,
    *,
    allowed_agents=None,
    reference_mode: str = "anytime",
    extra_system_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra_tools=None,
    save_fn: Optional[Callable] = None,
    engine=None,
    attachments=None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    references_max_model_calls: Optional[int] = DEFAULT_REFERENCES_MAX_MODEL_CALLS,
    markup_author: Optional[str] = None,
    enable_calc_subagent: bool = False,
    calc_extra_tools=None,
    calc_extra_system_prompt: Optional[str] = None,
    calc_max_model_calls: Optional[int] = DEFAULT_CALC_MAX_MODEL_CALLS,
    subagent_extra_tools=None,
    subagent_extra_system_prompt: Optional[str] = None,
    enable_setup_agent: bool = False,
    setup_store=None,
    setup_render_dir: Optional[str] = None,
    enable_report_ingest: Optional[bool] = None,
    report_ingest_out_dir: Optional[str] = None,
    report_ingest_budgets=None,
    report_ingest_db: Optional[str] = None,
    report_ingest_max_model_calls: Optional[int] =
    DEFAULT_REPORT_INGEST_MAX_MODEL_CALLS,
    enable_report_library: bool = False,
    library_root: Optional[str] = None,
    library_db: Optional[str] = None,
    library_max_tool_calls: Optional[int] = None,
    store=None,
    checkpointer=None,
    enable_memory: bool = False,
    enable_summarization: bool = False,
    summarization_model=None,
    summarization_trigger=_AUTO_SUMMARIZATION_TRIGGER,
    summarization_keep=_DEFAULT_SUMMARIZATION_KEEP,
    review_page: bool = False,
    working_dir: Optional[str] = None,
    coverage_dir: Optional[str] = None,
    subagent_models: Optional[dict] = None,
    **kwargs,
):
    """Construct the deepagents port of the geotech agent.

    ``subagent_models`` gives a helper its own chat model, by the helper's
    name (``{"references": sonnet}``; BACKLOG INF-41). A helper not named,
    and every helper when it is ``None``, runs on ``model`` as before; a
    compiled helper (report ingest, the report library) keeps the model it
    was built with, and a name this build has no helper for is ignored. The
    host decides the map (``webapp.helper_models``); the compiled agent's
    ``geotech_build["helper_models"]`` records the result.

    ``coverage_dir`` is the conversation's folder, where the coverage ledger
    (``coverage.json``) is kept when ``GEOTECH_COVERAGE`` is on
    (:mod:`funhouse_agent.deep.coverage_tools`); ``None`` keeps it in
    memory. With the switch off (the default) it is unused and the build is
    unchanged. ``GEOTECH_REVIEW_CHECKLIST`` keeps no ledger since RI-34: its
    checks read report ingest's record (:mod:`report_ingest.checklist`).

    ``review_page=True`` hands the build to the Document Review page's own
    (:func:`funhouse_agent.review_build.build_review_page_agent`, INF-31
    step 8; the page builds itself there since), for callers from before,
    until step 15; any geotechnical part such a caller asks for (modules,
    reference or calc helpers, the domain prompt) is made here and handed
    over. ``working_dir`` (the conversation's working folder) is that
    build's (the lean route binds its findings ledger and digests to it) and
    is ignored by this one.

    By default this agent already has — courtesy of deepagents — *planning*
    (the ``write_todos`` tool, via ``TodoListMiddleware``) and an in-thread
    *scratch filesystem* (``ls`` / ``read_file`` / ``write_file`` /
    ``edit_file``, via ``FilesystemMiddleware`` on an ephemeral
    :class:`StateBackend`). Those are ON with no configuration. The Phase-3
    parameters below add the *cross-session* and *durability* capabilities.

    Parameters
    ----------
    model : str | BaseChatModel
        The chat model for ``create_deep_agent`` (e.g. a
        ``langchain_anthropic.ChatAnthropic`` instance, a ``"provider:model"``
        string, or a fake chat model for offline tests).
    allowed_agents : iterable of str, optional
        Direct-call scope for the PRIMARY agent. Defaults to
        ``ANALYSIS_MODULES`` (keeps the primary's tool surface small; reference
        access is delegated to the ``references`` sub-agent). Pass an explicit
        set to override.
    reference_mode : str
        Retained from v1 for call-site compatibility. In v5 reference access is
        always available through the ``references`` sub-agent (the ``task``
        tool), so this currently only gates whether that sub-agent is attached:
        ``"off"`` omits it, anything else attaches it. Defaults to
        ``"anytime"``.
    extra_system_prompt : str, optional
        An extra block appended to the domain system prompt. Lets a caller
        specialize the agent without forking the prompt machinery — e.g. the
        seismic reviewer (``funhouse_agent.reviewers.make_seismic_reviewer_deep``)
        injects its review-mode checklist here. Default ``None`` leaves the
        prompt unchanged.
    system_prompt : str, optional
        REPLACES the geotechnical domain prompt outright (``extra_system_prompt``
        still appends to it). The document-review page passes
        :func:`~funhouse_agent.review_prompt.build_document_review_prompt` here
        with ``allowed_agents=()`` and ``reference_mode="off"`` — a different
        agent over the same document, vision and file tools. Default ``None``
        keeps the geotechnical prompt.
    markup_author : str, optional
        Who ``annotate_document`` signs review comments as when a call names
        nobody — on a multi-user host the signed-in person's
        ``Identity.markup_author``. ``None`` = the deployment-wide default.
    extra_tools : list, optional
        Additional LangChain tools appended to the PRIMARY agent's tool list
        (after the standard meta/vision/file tools). Lets a host wire
        deployment-specific capabilities — e.g. the webapp's SharePoint
        file tools — without touching the module catalog. Default ``None``
        adds nothing. Pair with ``extra_system_prompt`` so the agent knows
        the tools exist.
    save_fn : callable, optional
        ``(path, content) -> saved_path`` for ``save_file``. Defaults to local
        filesystem write.
    engine : GenAIEngine, optional
        Vision engine for ``analyze_image`` / ``analyze_pdf_page`` /
        ``read_reference_figure``. If ``None`` AND ``model`` is a chat-model
        object (not a ``"provider:model"`` string), it is default-wrapped with
        :class:`~funhouse_agent.deep.vision_engine.LangChainVisionEngine` so the
        vision tools route through the SAME model object — one model drives both
        text and vision, no separate Anthropic/OpenAI SDK client needed. Pass an
        explicit ``engine`` (a Funhouse engine / ``ClaudeEngine`` /
        ``NativeToolEngine``) to override; it is used unchanged. ``None`` with a
        string ``model`` leaves vision unwired (those tools return a clear error
        if invoked).
    attachments : dict, optional
        ``{key: bytes}`` of attached files for the vision tools.
    max_result_chars : int, optional
        Single knob capping the size of EVERY tool result fed back to the model
        — the primary's tools AND both sub-agents' (references, reviewer). It is
        forwarded to ``build_primary_tools`` and both sub-agent builders.
        Defaults to ``8000`` (mirrors v1's ``GeotechAgent._max_result_chars``);
        set ``0`` to disable truncation. This is the token-bloat guard: the
        references sub-agent's large reference-text dumps would otherwise be
        re-sent on every round and compound the input-token cost.
    reference_result_chars : int, optional
        LARGER cap for reference reads (``call_agent`` on a reference module
        inside the references/reviewer sub-agents, plus
        ``read_reference_figure``), where the reference text IS the payload.
        Defaults to 16000 (never below ``max_result_chars``); numeric/calc
        ``call_agent`` results are uncapped (they are tiny). Disabled along
        with everything else when ``max_result_chars <= 0``.
    references_max_model_calls : int, optional
        Per-consult model-call budget for the ``references`` sub-agent — the
        v5.1 single biggest cost lever (rc7: 26/76 reference questions burned
        69% of all tokens via unbounded internal rounds). The last budgeted
        call is forced to summarize-and-answer from what was gathered, so a
        final answer is ALWAYS returned (never an error). Defaults to ``8``;
        ``None``/``0`` disables the budget.
    enable_calc_subagent : bool
        Attach the ``calc`` sub-agent (A2 context isolation) and nudge the primary
        to delegate tool-heavy calculation to it, so the bulky calc-package /
        method-dump trace stays in the sub-agent's context instead of the main
        conversation (which is what a persist+replay chat re-sends each turn).
        The sub-agent returns a compact result (values + units + method + saved
        file path) and saves the full payload to a file (no data loss). OFF by
        default (additive / default-preserving — the library and the eval suite
        are unchanged); the web app turns it ON per conversation.
    calc_extra_tools : list, optional
        Extra tools for the ``calc`` sub-agent ONLY, on top of
        ``subagent_extra_tools`` (a tool present in both is attached once).
    calc_extra_system_prompt : str, optional
        Appended to the ``calc`` sub-agent's prompt, after
        ``subagent_extra_system_prompt``.
    calc_max_model_calls : int, optional
        Per-delegation model-call budget for the ``calc`` sub-agent (as
        ``references_max_model_calls`` is for references). Defaults to ``16``;
        ``None``/``0`` disables the budget.
    subagent_extra_tools : list, optional
        Extra tools for EVERY sub-agent this builder attaches — ``references``,
        ``reviewer``, ``calc``, ``model_setup``. Each of those declares its own
        tool list, so ``extra_tools`` never reaches them (it does reach
        deepagents' built-in ``general-purpose`` sub-agent, which inherits the
        primary's tools). A host capability meant for the whole team — the web
        app's ``record_feedback`` — goes in both.
    subagent_extra_system_prompt : str, optional
        Appended to each of those sub-agents' prompts.
    enable_setup_agent : bool
        Attach the ``model_setup`` sub-agent (staged LE/FEM model building
        with human confirmation gates and echo-back renders — see
        :mod:`funhouse_agent.deep.setup_agent`). OFF by default: model setup
        is an opt-in, session-scoped workflow whose tools carry mutable
        per-build state (the ProjectStore), so hosts that only ask
        calculation questions never pay for the extra tool surface.
    setup_store : ProjectStore, optional
        The shared project state for the setup sub-agent (created fresh when
        omitted). Pass your own to inspect ``setup_store.project`` from the
        host. Only used when ``enable_setup_agent`` is on.
    setup_render_dir : str, optional
        Directory the setup sub-agent writes echo-back PNGs to (default
        ``model_setup_renders/``). Only used when ``enable_setup_agent`` is
        on.
    store : BaseStore, optional
        A LangGraph store (e.g. ``InMemoryStore`` / a Postgres store) for
        cross-session / persistent memory. Passing a store turns ``enable_memory``
        on automatically — files the agent writes under ``/memories/`` are routed
        to this store and therefore survive across threads, while ordinary scratch
        files stay ephemeral in per-thread state. Forwarded to
        ``create_deep_agent(store=...)``.
    checkpointer : Checkpointer, optional
        A LangGraph checkpointer (e.g. ``InMemorySaver`` / a Postgres saver) for
        durable, resumable threads. With a checkpointer, an interrupted run can be
        resumed and a thread's full state (messages, todos, scratch files) is
        persisted under its ``thread_id``. Forwarded to
        ``create_deep_agent(checkpointer=...)``.
    enable_memory : bool
        Turn on the persistent ``/memories/`` filesystem route. When on (or when a
        ``store`` is given), the filesystem backend becomes
        ``CompositeBackend(default=StateBackend(), routes={"/memories/":
        StoreBackend()})`` and a deepagents ``memory=[/memories/AGENTS.md]`` source
        is registered so any durable project context the agent has saved is loaded
        into the system prompt at the start of every thread. Requires a ``store``
        at runtime for the route to actually persist (``create_deep_agent`` wires
        the store into ``StoreBackend`` via ``get_store()``). Defaults to ``False``.
    enable_summarization : bool
        Append a second, explicitly-configured
        :class:`GeotechSummarizationMiddleware` so long sessions auto-compact at
        our chosen threshold. NOTE: deepagents already attaches a summarizer with
        model-aware defaults; this adds a custom-tuned one alongside it (they share
        state and cannot collide because the subclass reports a distinct name).
        Defaults to ``False``.
    summarization_model : str | BaseChatModel, optional
        Model used to *write* the summaries when ``enable_summarization`` is on.
        Defaults to the primary ``model``. (If ``model`` is a ``"provider:model"``
        string, that string is used as-is.)
    summarization_trigger : ContextSize, optional
        When to summarize. Defaults to auto: ``("fraction", 0.8)`` (compact at
        ~80% of the model's input window) when the model publishes its window,
        else an absolute ``("tokens", 170000)`` fallback for profile-less models
        (string specs / fakes) so construction never fails.
    summarization_keep : ContextSize
        How much recent history to keep verbatim after a summarization. Defaults
        to ``("messages", 8)`` so the latest exchanges survive intact.
    **kwargs
        Forwarded verbatim to ``create_deep_agent`` (e.g. ``name``, ``debug``,
        ``permissions``).

    Returns
    -------
    CompiledStateGraph
        The compiled deep agent.

    Notes
    -----
    With NO Phase-3 arguments this builds exactly as before: no ``store``, no
    ``checkpointer``, the default ephemeral ``StateBackend`` filesystem, and only
    the deepagents-default summarizer. Planning (``write_todos``) and the scratch
    filesystem tools are present regardless because deepagents attaches them by
    default.
    """
    # The first pyplot import, before tools run side by side (GM-172).
    from funhouse_agent.deep.warm_imports import warm_pyplot
    warm_pyplot()
    # The reference library's chart lines, handed to the chart stack before
    # any tool is described (plot_data's reference_chart, the box's
    # chart_lines; INF-31 step 6). Both pages are built here until step 8.
    from funhouse_agent.deep.tools import register_reference_charts
    register_reference_charts()
    if review_page:
        return _review_page_build(locals())

    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES

    # A store implies persistent memory: there is no point handing in a store
    # without routing /memories/ to it.
    if store is not None:
        enable_memory = True

    # Default-wrap the model for vision when no explicit engine is given and the
    # model is a chat-model object (not a "provider:model" string). This lets a
    # single model object satisfy both text and vision; an explicit engine
    # (Funhouse engine / ClaudeEngine / NativeToolEngine) is used unchanged.
    if engine is None and not isinstance(model, str):
        engine = LangChainVisionEngine(model)

    # ONE shared attachments dict for the whole agent: the primary vision tools
    # AND the references sub-agent close over the SAME object, and it is exposed
    # on the returned agent (below) so a notebook FileUpload can add files after
    # the build reaches every tool. (Previously a None default made each
    # make_vision_tools create its own empty dict — added files reached nothing.)
    attachments = {} if attachments is None else attachments

    tools = build_primary_tools(
        allowed_agents=allowed_agents,
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
        markup_author=markup_author,
    )
    if extra_tools:
        tools = list(tools) + list(extra_tools)

    # ``system_prompt`` REPLACES the geotechnical domain prompt — the
    # document-review page is a different agent over the same tools, not the
    # geotech agent with a suffix. ``extra_system_prompt`` still appends.
    # Whether the prompt asks for a judgment-calls list after one consult
    # of the references helper: the turn-end check (GM-174) acts only then.
    judgment_consult = False
    judgment_prompt = False
    if system_prompt is None:
        system_prompt = build_domain_prompt(allowed_agents,
                                            memory_enabled=enable_memory)
        if reference_mode == "off":
            # no references helper to run a reference's function through
            system_prompt = system_prompt.replace(
                REFERENCES_HELPER_CLAUSE, REFERENCE_MODULES_CLAUSE)
        from funhouse_agent import review_flags as _flags
        if _flags.judgment_calls():
            # The judgment calls put in front of the user (GM-116), behind
            # GEOTECH_JUDGMENT_CALLS, on by default (GM-226). The geotech
            # prompt only: a page that brings its own prompt is unchanged.
            # With the chat's references off there is no helper to consult
            # and the block says so (the user's choice wins).
            system_prompt += "\n\n" + judgment_calls_section(
                references=reference_mode != "off")
            judgment_consult = reference_mode != "off"
            judgment_prompt = True
    if extra_system_prompt:
        # Appended AFTER the domain prompt (mirrors GeotechAgent.system_prompt_extra)
        # so a caller can re-cast the agent — e.g. the seismic reviewer's review-mode
        # checklist (funhouse_agent.reviewers.make_seismic_reviewer_deep). Default
        # None leaves the prompt unchanged.
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    if enable_calc_subagent:
        # A2: nudge the primary to delegate tool-heavy calc to the `calc`
        # sub-agent so the bulky trace stays out of the main conversation.
        # With the judgment calls on, the brief also carries the user's own
        # words for a package's `given` and the consult's citations
        # (GM-181, GM-155 w13b fix 10).
        system_prompt = system_prompt + "\n\n" + calc_delegation_nudge(
            judgment_calls=judgment_prompt,
            references=reference_mode != "off")
    # The whole-report ingest: one primary tool and one sub-agent, both OFF by
    # default. The tool hides itself on a planlens too old to serve the ingest
    # (the page roles and the log grid), so the pair is only really on when
    # the deployment can run it. Unless the caller says, it follows
    # GEOTECH_REPORT_INGEST, never on the Document Review page (RI-35); with
    # GEOTECH_REVIEW_CHECKLIST on as well, the checklist's code checks run on
    # each record the ingest writes.
    from funhouse_agent import review_flags as _review_flags
    if enable_report_ingest is None:
        enable_report_ingest = _review_flags.report_ingest()
    ingest_checklist = _review_flags.checklist()
    ingest_tools = []
    if enable_report_ingest:
        from funhouse_agent.deep.tools import make_report_ingest_tool
        ingest_tools = make_report_ingest_tool(
            engine=engine, attachments=attachments,
            out_dir=report_ingest_out_dir, budgets=report_ingest_budgets,
            db_path=report_ingest_db, max_result_chars=max_result_chars,
            max_model_calls=report_ingest_max_model_calls,
            checklist=ingest_checklist)
        if ingest_tools:
            tools = list(tools) + list(ingest_tools)
            system_prompt = system_prompt + "\n\n" + REPORT_INGEST_NUDGE
            # DR-11: the calculations an ingested report claims, recomputed
            # through the modules; it reads the record the ingest writes, so
            # it comes and goes with the ingest tool.
            from report_ingest.recompute import make_recompute_tool
            cap = max(max_result_chars, 16000) if max_result_chars > 0 else 0
            tools = list(tools) + [make_recompute_tool(cap)]
    # The report LIBRARY: the same shape, the other half of the job. Its
    # feature detection is the FOLDER rather than a package version — a
    # deployment either has reports read into it or does not, and a library
    # agent over an empty folder would only teach the primary to ask it
    # things nothing can answer.
    library_tools = []
    if enable_report_library:
        from funhouse_agent.deep.tools import make_report_library_tool
        library_tools = make_report_library_tool(
            model=model, library_root=library_root, db_path=library_db,
            max_result_chars=max_result_chars,
            max_tool_calls=library_max_tool_calls)
        if library_tools:
            tools = list(tools) + list(library_tools)
            system_prompt = system_prompt + "\n\n" + REPORT_LIBRARY_NUDGE
    # Coverage in code (GEOTECH_COVERAGE, OFF by default): the ledger every
    # agent of this build reports its tool calls to, the gate on the primary,
    # the tool. No prompt text: the tool describes itself. On the review page
    # only a declared coverage task holds a turn (BACKLOG RI-34).
    from funhouse_agent.deep.coverage_tools import coverage_kit
    coverage = coverage_kit(attachments=attachments, folder=coverage_dir,
                            max_result_chars=max_result_chars,
                            review_page=False)
    if coverage.tools:
        tools = list(tools) + list(coverage.tools)
    # The geotechnical report review checklist (GEOTECH_REVIEW_CHECKLIST, OFF
    # by default): report_ingest's, run on the record report ingest wrote;
    # the geotech page only (BACKLOG RI-34).
    if ingest_checklist:
        from report_ingest.checklist import make_checklist_tool
        cap = max(max_result_chars, 16000) if max_result_chars > 0 else 0
        tools = list(tools) + [make_checklist_tool(
            cap, ingest_tool=bool(ingest_tools))]
    # Every primary tool refuses an argument it does not take, by name,
    # whichever module built it (brief 5, N3: analyze_pdf_page(pages=12)
    # silently looked at page 0). See funhouse_agent/deep/strict_args.py.
    from funhouse_agent.deep.strict_args import strict_tools
    tools = strict_tools(tools)

    subagents = []
    if reference_mode != "off":
        subagents.append(
            build_references_subagent(
                engine=engine,
                attachments=attachments,
                save_fn=save_fn,
                max_result_chars=max_result_chars,
                reference_result_chars=reference_result_chars,
                max_model_calls=references_max_model_calls,
                extra_tools=subagent_extra_tools,
                extra_system_prompt=subagent_extra_system_prompt,
                judgment_consult=judgment_consult,
            )
        )
        subagents.append(
            build_reviewer_subagent(
                max_result_chars=max_result_chars,
                reference_result_chars=reference_result_chars,
                extra_tools=subagent_extra_tools,
                extra_system_prompt=subagent_extra_system_prompt,
            )
        )
    if enable_calc_subagent:                                   # A2 context isolation
        subagents.append(
            build_calc_subagent(
                engine=engine,
                attachments=attachments,
                save_fn=save_fn,
                allowed_agents=allowed_agents,
                max_result_chars=max_result_chars,
                reference_result_chars=reference_result_chars,
                max_model_calls=calc_max_model_calls,
                extra_tools=_merge_tools(subagent_extra_tools, calc_extra_tools),
                extra_system_prompt=_join_prompts(subagent_extra_system_prompt,
                                                  calc_extra_system_prompt),
            )
        )
    from funhouse_agent import review_flags as _flags
    figure_tasks = None
    if _flags.figure_helper() or _flags.plot_code():
        # The figure helper (OWN-61 option B, GM-92): draws, looks, redraws
        # to a checklist; behind GEOTECH_FIGURE_HELPER, on by default
        # (GM-227).
        # GEOTECH_PLOT_CODE brings it too: plotting code needs a helper the
        # main agent hands figures to (live wave w8, F55: without one the
        # main agent drew a wall with plot_data five times and ran out of
        # steps).
        from funhouse_agent.deep.figure_helper import (FIGURE_PRIMARY_NOTE,
                                                       FigureTasks,
                                                       build_figure_subagent)
        _fig = build_figure_subagent(
            engine=engine, attachments=attachments, save_fn=save_fn,
            max_result_chars=max_result_chars,
            reference_result_chars=reference_result_chars,
            extra_tools=subagent_extra_tools,
            extra_system_prompt=subagent_extra_system_prompt)
        if _fig is not None:
            subagents.append(_fig)
            # The main agent hands figures to it and does not look again
            # at what it looked at; each brief carries the nearest worked
            # example and each result what the helper's looks found (live
            # wave w9, GM-107).
            system_prompt = system_prompt + "\n\n" + FIGURE_PRIMARY_NOTE
            figure_tasks = FigureTasks(with_code=any(
                getattr(t, "name", None) == "plot_code"
                for t in _fig["tools"]))
    if ingest_tools:
        # A CompiledSubAgent: its own small graph, no model of its own, and a
        # compact structured_response as the tool result. Attached only when
        # the tool was built, so the primary is never told to delegate to a
        # sub-agent that cannot run here.
        from funhouse_agent.deep.tools import VISION_TIER
        from funhouse_agent.deep.tools import report_ingest_out_dir as _out
        from report_ingest.subagent import build_report_ingest_subagent
        subagents.append(
            build_report_ingest_subagent(
                _report_ingest_engine_factory(engine),
                # The page voter on the cheap tier, as the tool runs it.
                vision_engine_factory=_report_ingest_engine_factory(
                    engine, model=VISION_TIER),
                # The conversation's own folder, resolved when the report is
                # read (the folder is bound per turn), as the tool does.
                out_dir_factory=(
                    (lambda named: _out(named, _resolved_or_none(
                        named, attachments)))
                    if report_ingest_out_dir is None
                    else (lambda _source: report_ingest_out_dir)),
                budgets=report_ingest_budgets,
                db_path=report_ingest_db,
                resolve_source=lambda source: _resolve_ingest_source(
                    source, attachments),
                max_model_calls=report_ingest_max_model_calls,
                checklist=ingest_checklist,
            )
        )
    if library_tools:
        # The library's twin CompiledSubAgent: a model with the query layer
        # bound as tools, answering only from the records. Attached only when
        # the tool was built, so the primary is never told to delegate to a
        # library that is not there.
        from report_ingest.library_agent import build_report_library_subagent
        subagents.append(
            build_report_library_subagent(
                model,
                library_root=library_root,
                db_path=library_db,
                max_tool_calls=(library_max_tool_calls
                                or DEFAULT_REPORT_LIBRARY_MAX_TOOL_CALLS),
                max_result_chars=max_result_chars,
                max_model_calls=DEFAULT_REPORT_LIBRARY_MAX_MODEL_CALLS,
            )
        )
    if enable_setup_agent:
        subagents.append(
            build_setup_subagent(
                store=setup_store,
                render_dir=setup_render_dir,
                max_result_chars=max_result_chars,
                extra_tools=subagent_extra_tools,
                extra_system_prompt=subagent_extra_system_prompt,
            )
        )

    # The guards on every helper, deepagents' general-purpose helper with
    # them, a helper's own model and the coverage recorders: the assembly
    # both pages share (funhouse_agent.deep.assembly, INF-31 step 8). An
    # agent with call_agent also gets the unknown-tool hint, scoped to the
    # modules that agent reaches; the calc helper's FigureLooks notes go
    # just inside DeliveredToolResults.
    from funhouse_agent.deep.calc_figures import FigureLooks
    helper_models = _assembly.guard_helpers(
        subagents, tools, allowed_agents=allowed_agents,
        call_scope=_call_scopes(allowed_agents), note_types=(FigureLooks,),
        unknown_tool_hint=UnknownToolHint,
        subagent_models=subagent_models, coverage=coverage)

    # The geotech page: deepagents' shell and delete cannot run on this
    # backend (GM-176; wave w13b F55j called execute with no command).
    create_kwargs = _assembly.create_kwargs_for(
        kwargs, model=model, store=store, checkpointer=checkpointer,
        enable_memory=enable_memory,
        enable_summarization=enable_summarization,
        summarization_model=summarization_model,
        summarization_trigger=summarization_trigger,
        summarization_keep=summarization_keep,
        figure_tasks=figure_tasks, allowed_agents=allowed_agents,
        unknown_tool_hint=UnknownToolHint,
        page_middleware=[HideTools(UNRUNNABLE_TOOLS)], coverage=coverage)
    # create_deep_agent is looked up here, so a test that patches this
    # module's name for it sees the call.
    agent = _assembly.compile_agent(
        create_deep_agent, model=model, tools=tools,
        system_prompt=system_prompt, subagents=subagents,
        create_kwargs=create_kwargs)
    return _assembly.record_build(
        agent, attachments=attachments, model=model, tools=tools,
        subagents=subagents, system_prompt=system_prompt,
        helper_models=helper_models, judgment_consult=judgment_consult,
        coverage=coverage)


def _call_scopes(allowed_agents) -> dict:
    """The modules each helper reaches, for its unknown-tool hint."""
    _calc_scope = frozenset(allowed_agents or ())
    if _calc_scope == frozenset(ANALYSIS_MODULES):      # as build_calc_subagent
        _calc_scope |= (frozenset(REFERENCE_MODULES)
                        - {"reference_db", "figure_db"})
    return {"references": REFERENCE_MODULES,
            "reviewer": REFERENCE_MODULES,
            "calc": _calc_scope,
            "general-purpose": allowed_agents}


def _review_page_build(args: dict):
    """``build_deep_agent(review_page=True)``: the Document Review page's own
    build (:func:`funhouse_agent.review_build.build_review_page_agent`),
    with whatever geotechnical part the caller asked for made here, as this
    builder made it, and placed where it placed it (INF-31 step 8; for
    callers from before, until step 15)."""
    from funhouse_agent import review_build
    a = dict(args)
    kwargs = a.pop("kwargs", None) or {}
    for name in ("review_page",):
        a.pop(name, None)
    model = a.pop("model")
    allowed_agents = a["allowed_agents"]
    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES
    reference_mode = a["reference_mode"]
    wants_parts = (bool(allowed_agents) or reference_mode != "off"
                   or a["enable_calc_subagent"] or a["enable_setup_agent"]
                   or a["system_prompt"] is None)

    def geotech_parts(engine, attachments, enable_memory):
        parts = {"allowed_agents": allowed_agents,
                 "call_scope": _call_scopes(allowed_agents),
                 "unknown_tool_hint": UnknownToolHint,
                 "note_types": ()}
        from funhouse_agent.deep.calc_figures import FigureLooks
        parts["note_types"] = (FigureLooks,)
        parts["core_tools"] = make_core_tools(
            allowed_agents=allowed_agents,
            max_result_chars=a["max_result_chars"],
            reference_result_chars=a["reference_result_chars"],
            attachments=attachments,
        ) if allowed_agents else []
        if a["system_prompt"] is None:
            prompt = build_domain_prompt(allowed_agents,
                                         memory_enabled=enable_memory)
            if reference_mode == "off":
                prompt = prompt.replace(REFERENCES_HELPER_CLAUSE,
                                        REFERENCE_MODULES_CLAUSE)
            parts["domain_prompt"] = prompt
        if a["enable_calc_subagent"]:
            parts["calc_nudge"] = calc_delegation_nudge(
                judgment_calls=False, references=reference_mode != "off")
        before = []
        if reference_mode != "off":
            before.append(build_references_subagent(
                engine=engine, attachments=attachments, save_fn=a["save_fn"],
                max_result_chars=a["max_result_chars"],
                reference_result_chars=a["reference_result_chars"],
                max_model_calls=a["references_max_model_calls"],
                extra_tools=a["subagent_extra_tools"],
                extra_system_prompt=a["subagent_extra_system_prompt"],
                judgment_consult=False))
            before.append(build_reviewer_subagent(
                max_result_chars=a["max_result_chars"],
                reference_result_chars=a["reference_result_chars"],
                extra_tools=a["subagent_extra_tools"],
                extra_system_prompt=a["subagent_extra_system_prompt"]))
        if a["enable_calc_subagent"]:
            before.append(build_calc_subagent(
                engine=engine, attachments=attachments, save_fn=a["save_fn"],
                allowed_agents=allowed_agents,
                max_result_chars=a["max_result_chars"],
                reference_result_chars=a["reference_result_chars"],
                max_model_calls=a["calc_max_model_calls"],
                extra_tools=_merge_tools(a["subagent_extra_tools"],
                                         a["calc_extra_tools"]),
                extra_system_prompt=_join_prompts(
                    a["subagent_extra_system_prompt"],
                    a["calc_extra_system_prompt"])))
        parts["subagents_before_figure"] = before
        parts["subagents_after_figure"] = [build_setup_subagent(
            store=a["setup_store"], render_dir=a["setup_render_dir"],
            max_result_chars=a["max_result_chars"],
            extra_tools=a["subagent_extra_tools"],
            extra_system_prompt=a["subagent_extra_system_prompt"],
        )] if a["enable_setup_agent"] else []
        return parts

    keep = ("engine", "attachments", "save_fn", "extra_tools",
            "extra_system_prompt", "system_prompt", "markup_author",
            "max_result_chars", "reference_result_chars",
            "subagent_extra_tools", "subagent_extra_system_prompt", "store",
            "checkpointer", "enable_memory", "enable_summarization",
            "summarization_model", "summarization_trigger",
            "summarization_keep", "working_dir", "coverage_dir",
            "subagent_models", "reference_mode", "enable_calc_subagent")
    return review_build.build_review_page_agent(
        model, allowed_agents=allowed_agents,
        geotech_parts=geotech_parts if wants_parts else None,
        # looked up here, so a test that patches this module's name for
        # create_deep_agent sees the call
        create=lambda **kw: create_deep_agent(**kw),
        **{k: a[k] for k in keep}, **kwargs)


#: The Document Review page's pieces of this build, moved with it to
#: funhouse_agent.review_build (INF-31 step 8); still answered here.
_MOVED_TO_REVIEW_BUILD = ("review_page_budget", "REVIEW_PAGE_STEPS_PER_CALL",
                          "REVIEW_READER_DESCRIPTION", "GEOTECH_ONLY_TOOLS",
                          "_BUDGET_OFF", "_review_page_descriptions",
                          "_review_page_rewrites", "_review_reader_subagent")


def __getattr__(name):
    if name in _MOVED_TO_REVIEW_BUILD:
        import importlib
        return getattr(importlib.import_module("funhouse_agent.review_build"),
                       name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_memory_demo(model, store, checkpointer=None, *, verbose: bool = True) -> bool:
    """Prove cross-session ``/memories/`` persistence end-to-end (REAL model).

    Builds a memory-enabled agent on ``model`` + ``store`` and runs TWO separate
    threads:

    * **Thread 1** is told a durable fact and asked to save it under
      ``/memories/`` (e.g. ``write_file('/memories/AGENTS.md', ...)``).
    * **Thread 2** — a *new* ``thread_id`` sharing the SAME ``store`` — is asked
      to recall the fact. Because ``/memories/`` is store-backed, thread 2 can
      read what thread 1 wrote even though the two threads share no message
      history.

    .. warning::
       This makes REAL LLM API calls. It is NOT part of the offline suite and is
       never executed by it — it exists so a notebook / live run can confirm the
       agent actually *uses* ``/memories/`` across sessions (the offline tests
       only prove the backend round-trip and the wiring).

    Parameters
    ----------
    model : BaseChatModel
        A real tool-calling chat model.
    store : BaseStore
        A LangGraph store shared across both threads (e.g. ``InMemoryStore``).
    checkpointer : Checkpointer, optional
        Optional durable checkpointer for the threads.
    verbose : bool
        Print the recall answer.

    Returns
    -------
    bool
        ``True`` if thread 2's answer references the fact saved in thread 1.
    """
    agent = build_deep_agent(
        model=model, store=store, checkpointer=checkpointer, enable_memory=True,
    )
    fact = "The project design groundwater table is at 3.5 m below grade."
    keyword = "3.5"

    agent.invoke(
        {"messages": [{"role": "user", "content": (
            f"Remember this for the whole project and save it to your memory: {fact}"
        )}]},
        config={"configurable": {"thread_id": "demo-thread-1"}},
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": (
            "What is the project design groundwater table depth? "
            "Check your memory if needed."
        )}]},
        config={"configurable": {"thread_id": "demo-thread-2"}},
    )

    messages = result.get("messages", []) if isinstance(result, dict) else []
    final = ""
    for m in reversed(messages):
        content = getattr(m, "content", "")
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in (content or [])
        )
        if text.strip():
            final = text
            break

    recalled = keyword in final
    if verbose:
        print("Thread-2 recall answer:", final)
        print("RESULT:", "PASS" if recalled else "FAIL")
    return recalled


__all__ = [
    "build_deep_agent",
    "build_primary_tools",
    "build_references_subagent",
    "build_calc_subagent",
    "build_reviewer_subagent",
    "build_setup_subagent",
    "build_memory_backend",
    "build_summarization_middleware",
    "run_memory_demo",
    "GeotechSummarizationMiddleware",
    "MEMORIES_ROUTE",
    "MEMORIES_AGENTS_FILE",
]
