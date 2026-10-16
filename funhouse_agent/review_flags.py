"""Switches for the Document Review harness changes of 2026-09-26.

The review of the Document Review page (plan: "Document Review harness —
big-picture review", 2026-09-26) found that every recent change had been judged
on the one example that prompted it. So each behaviour change from that review
ships behind a switch here, OFF by default, and the review suite
(:mod:`funhouse_agent.review_eval`) runs the same tasks with the switches off
and on from ONE install. A switch is turned on by default only once the suite
says it helps.

Every switch is read from the environment AT CALL TIME, so a notebook (or the
suite) can flip one between two runs without rebuilding anything but the agent.

``GEOTECH_REVIEW_AGENT``
    ``legacy`` (default) builds the review page's agent the way it has been
    built since 5.26: the geotechnical builder with the modules subtracted.
    ``lean`` builds it on its own (:mod:`funhouse_agent.deep.review_agent`):
    only the reading, looking and writing tools, no coding-agent prompt or
    scratch-filesystem tools, a reading sub-agent that carries the review
    rules, and a model-call budget that ends a long turn with an answer.
``GEOTECH_VISION_TEXT_CONTEXT``
    ``1`` gives every page/region vision call the PDF's own text-layer lines
    inside the view it is looking at (exact strings, with positions), or says
    there are none.
``GEOTECH_VISION_STRUCTURED``
    ``1`` asks every page/region vision call to end with a machine-readable
    list of what it located, and returns those locations as boxes in PDF
    points the agent can pass straight back to ``render_region``.
``GEOTECH_VISION_INLINE``
    ``1`` lets the reasoning model look: the page/region tools stop making the
    separate one-shot vision call and the newest images are shown to the main
    model itself at its next call (lean agent only). ``analyze_image`` keeps
    its one-shot vision call, exactly as released in 5.30, unless
    ``GEOTECH_REVIEW_OVERVIEW`` is on too (see there).
``GEOTECH_REVIEW_SWEEP``
    ``1`` offers the ``sweep_pages`` tool (lean agent only): one question asked
    of every page in a range, in parallel, answered per page with citations.
``GEOTECH_REVIEW_FINDINGS``
    ``1`` offers the findings tools (lean agent only): ``record_finding``,
    ``list_findings`` and ``findings_report``. Each finding is kept in
    ``findings.json`` in the conversation's working folder in the shared
    format of :mod:`funhouse_agent.review_findings`, its quotes checked
    against the pages it cites, and the Word comment log and marked-up PDFs
    are rendered from it (plan S1.5, ``module_work/REVIEW_ARCHITECTURE.md``).
``GEOTECH_REVIEW_OVERVIEW``
    ``1`` makes the orientation turn after an upload ask for the contact
    sheets explicitly when the new uploads run past
    ``GEOTECH_REVIEW_OVERVIEW_PAGES`` pages in all (default 20), and to skip
    them at or below it (plan S1.2). Only with BOTH this and
    ``GEOTECH_VISION_INLINE`` on is the lean agent's ``analyze_image`` of an
    image FILE (a contact sheet) shown to the model itself, like the inline
    page tools; either one alone leaves ``analyze_image`` as released.
``GEOTECH_REVIEW_GEOMETRY``
    ``1`` offers the drawing-geometry tools (lean agent only):
    ``drawing_callouts``, ``drawing_dimensions``, ``title_block`` and
    ``revision_clouds`` (:mod:`funhouse_agent.deep.geometry_tools`). They
    find leaders, dimension lines, the title block and revision clouds from
    a sheet's line-work and return boxes to zoom on - "geometry says where,
    vision says what" (plan S1.1).
``GEOTECH_REVIEW_DIGEST``
    ``1`` offers the digest tools (lean agent only; its reading helper gets
    them too): ``document_inventory``, ``digest_search``, ``digest_pages``
    and ``digest_references`` (:mod:`funhouse_agent.deep.digest_tools`) over
    the FREE layer of each upload's digest (:mod:`funhouse_agent.
    review_digest`: page map, text index, references, built by code in
    seconds and kept in the working folder) - plan shape 2, Stage A. Its
    small/large hint uses ``GEOTECH_REVIEW_SHAPE1_PAGES`` (default 20).
``GEOTECH_COVERAGE``
    ``1`` keeps a ledger of every page any tool or helper reads or looks at
    (:mod:`funhouse_agent.coverage`, derived from the tool calls, kept in
    the conversation's ``coverage.json``), offers ``document_coverage`` (the
    document's pages by what they are, what was read, coverage as counts,
    pages marked extracted or skipped) and adds the gate: a turn that takes
    data out of a document and tries to finish with data pages unread is
    told once, with the list. BOTH pages and every agent build that reads
    documents (the geotech page, the legacy and lean review agents); not the
    minimal agent (plan W4, ``module_work/COVERAGE_AND_CHECKLIST.md``).
``GEOTECH_REVIEW_CHECKLIST``
    ``1`` offers ``report_checklist``: the report-review checklist (DRAFT,
    :mod:`funhouse_agent.report_checklist`) with the checks code can make
    run from the ledger and the data's cross-checks, and the judgement items
    returned for the agent to report against. With ``GEOTECH_COVERAGE`` on
    too, a failed code check joins the gate's note.
``GEOTECH_VISION_TURN_RETRY``
    ``1`` makes a look whose image goes as the page shows it (no turn asked
    for, none read off the text layer) end with one tagged line when most of
    its lettering does not read left to right (``TURNED: bottom-to-top``,
    ``top-to-bottom``, ``upside-down``). That view is then read ONCE more,
    turned so its lettering reads upright, and that reading is the answer.
    When looks at a page keep finding it turned (two or more, and more of the
    page's looked-at area turned than not), later looks at the page are sent
    turned from the start. Both pages; page and region looks and tiles
    (BACKLOG VIS-01: the sheets Sol misread sideways in Foundry brief 5 have
    no text layer to say which way their lettering runs).
``GEOTECH_REVIEW_KEPT_READINGS``
    ``1`` offers ``kept_readings`` on the Document Review page (and to its
    reading helper): the conversation's kept page readings, listed by
    reference and returned whole by document and page, read-only. The
    geotech page always offers it; on the review page it waits for the
    suite (arm ``kept_readings``; BACKLOG TA-28).
``GEOTECH_FIGURE_HELPER``
    ``1`` adds the ``figure`` helper (:mod:`funhouse_agent.deep.figure_helper`)
    to the agent: it draws a figure with ``plot_data`` from a figure
    checklist (after Tufte) and worked examples, looks at the PNG and
    redraws (owner's choice, 2026-10-15, OWN-61 option B). Both pages; not
    the lean review agent (BACKLOG GM-92).
``GEOTECH_PLOT_CODE``
    ``1`` offers ``plot_code`` to the figure helper and the ``calc`` helper,
    and brings the figure helper with it (live wave w8: without a helper
    to hand figures to, the main agent never reached the box):
    matplotlib or Plotly code the model writes, run in a locked box (its own
    process, no network, files only in the conversation folder, imports
    limited to the plotting and number libraries, a memory ceiling and a
    generous time limit; :mod:`funhouse_agent.plot_box`). The owner,
    2026-10-15: "I've been frustrated by the calc packages' lack of
    creative figures... This would solve it." Settings (cleared with the
    switches): ``GEOTECH_PLOT_CODE_SECONDS``, ``GEOTECH_PLOT_CODE_MAX_SECONDS``,
    ``GEOTECH_PLOT_CODE_MEMORY_MB``, ``GEOTECH_PLOT_CODE_AT_ONCE``. Both
    pages (BACKLOG GM-98).
``GEOTECH_STEP_CAP_ANSWER``
    ``1``: a turn that reaches its step cap (LangGraph's recursion limit)
    answers from what it gathered, in one last model call with no tools
    (``webapp.core.answer_at_step_cap``), instead of ending on an error; the
    raw error goes to the activity log. Both pages (BACKLOG TA-33; live wave
    w8 F55 ended on a raw GraphRecursionError and five chart cards).
``GEOTECH_CITED_PAGES``
    What the app does when an answer cites pages that no tool or helper read
    or looked at in the conversation (:mod:`funhouse_agent.cited_pages`; the
    check itself always runs and is kept in ``activity.jsonl`` as a
    ``cited_pages`` event). ``note`` (or ``1``) appends a short note to the
    answer listing them; ``retry`` tells the model once, before the answer
    goes out, and the answer it writes after that note is the one delivered
    (with the note appended if it still cites such pages). Off: the answer is
    left as the model wrote it. Both pages (BACKLOG EV-18).
``GEOTECH_CITED_DOCUMENTS``
    What the app does when an answer or a deliverable cites a published
    document (a GEC, AASHTO LRFD, a UFC, an ASTM standard...) or an author
    and year that nothing in the conversation looked up: no references
    result, no document opened, no worked example, no calculation's own
    result and none of the user's own messages names it
    (:mod:`funhouse_agent.cited_documents`; the answer's check always runs
    and is kept in ``activity.jsonl`` as a ``cited_documents`` event).
    ``note`` (or ``1``) appends a short note to the answer naming them, and
    a calc package, Word file, text report or markup note written with such
    a citation gets ``cited_not_looked_up`` and one plain line in its tool
    result (the file is written all the same); ``retry`` does the same and
    also tells the model once, before the answer goes out, and the answer it
    writes after that note is the one delivered. Off: answers and results
    are left as they are. Both pages (BACKLOG EV-21; live wave w8b, F55).
``GEOTECH_JUDGMENT_CALLS``
    ``1`` tells the geotech agent that the engineering judgment is the
    user's: an answer that rests on a judgment call (a method, a value the
    user did not give, drained or undrained, which stratum governs, search
    or mesh settings, how ambiguous data was read) ends with a short
    "Judgment calls" list, each with its reason and reference, the
    alternative set aside and what would change the answer. The ``calc``
    helper's reply lists the calls it made, and the references helper
    returns a reference's several methods or values as cited options. The
    owner, 2026-10-16: "make sure the agent is communicating its
    engineering judgements to the user. User's judgement is the one that
    matters." The geotech page only: the Document Review page has its own
    prompt and the figure helper is unchanged (BACKLOG GM-116, EV-26).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Optional

AGENT_ENV = "GEOTECH_REVIEW_AGENT"
VISION_TEXT_ENV = "GEOTECH_VISION_TEXT_CONTEXT"
VISION_STRUCTURED_ENV = "GEOTECH_VISION_STRUCTURED"
VISION_INLINE_ENV = "GEOTECH_VISION_INLINE"
SWEEP_ENV = "GEOTECH_REVIEW_SWEEP"
FINDINGS_ENV = "GEOTECH_REVIEW_FINDINGS"
OVERVIEW_ENV = "GEOTECH_REVIEW_OVERVIEW"
GEOMETRY_ENV = "GEOTECH_REVIEW_GEOMETRY"
DIGEST_ENV = "GEOTECH_REVIEW_DIGEST"
COVERAGE_ENV = "GEOTECH_COVERAGE"
CHECKLIST_ENV = "GEOTECH_REVIEW_CHECKLIST"
VISION_TURN_RETRY_ENV = "GEOTECH_VISION_TURN_RETRY"
KEPT_READINGS_ENV = "GEOTECH_REVIEW_KEPT_READINGS"
CITED_PAGES_ENV = "GEOTECH_CITED_PAGES"
CITED_DOCUMENTS_ENV = "GEOTECH_CITED_DOCUMENTS"
FIGURE_HELPER_ENV = "GEOTECH_FIGURE_HELPER"
PLOT_CODE_ENV = "GEOTECH_PLOT_CODE"
STEP_CAP_ANSWER_ENV = "GEOTECH_STEP_CAP_ANSWER"
JUDGMENT_CALLS_ENV = "GEOTECH_JUDGMENT_CALLS"

ALL_ENVS = (AGENT_ENV, VISION_TEXT_ENV, VISION_STRUCTURED_ENV,
            VISION_INLINE_ENV, SWEEP_ENV, FINDINGS_ENV, OVERVIEW_ENV,
            GEOMETRY_ENV, DIGEST_ENV, COVERAGE_ENV, CHECKLIST_ENV,
            VISION_TURN_RETRY_ENV, KEPT_READINGS_ENV, CITED_PAGES_ENV,
            FIGURE_HELPER_ENV, PLOT_CODE_ENV, STEP_CAP_ANSWER_ENV,
            CITED_DOCUMENTS_ENV, JUDGMENT_CALLS_ENV)

#: Not a switch but the overview's setting: above this many pages in the
#: new uploads (all of them together) the orientation asks for contact sheets.
OVERVIEW_PAGES_ENV = "GEOTECH_REVIEW_OVERVIEW_PAGES"
DEFAULT_OVERVIEW_PAGES = 20

#: The switches' SETTINGS (numbers the switched behaviour reads, not
#: switches): the overview's page threshold, the digest's small/large hint,
#: how many inline images each call carries and the lean agent's model-call
#: budget. :func:`switches` clears them too unless the arm names them, so an
#: arm is exactly what it says. Named here as strings so this module imports
#: nothing.
SETTINGS_ENVS = (OVERVIEW_PAGES_ENV, "GEOTECH_REVIEW_SHAPE1_PAGES",
                 "GEOTECH_VISION_INLINE_KEEP",
                 "GEOTECH_REVIEW_MAX_MODEL_CALLS",
                 "GEOTECH_PLOT_CODE_SECONDS", "GEOTECH_PLOT_CODE_MAX_SECONDS",
                 "GEOTECH_PLOT_CODE_MEMORY_MB", "GEOTECH_PLOT_CODE_AT_ONCE")

_ON = ("1", "true", "yes", "on")


def _on(env: str) -> bool:
    return str(os.environ.get(env, "")).strip().lower() in _ON


def lean_agent() -> bool:
    """Whether the review page builds its own lean agent."""
    return str(os.environ.get(AGENT_ENV, "")).strip().lower() == "lean"


def minimal_agent() -> bool:
    """Whether the review page builds the looking-only agent
    (:mod:`funhouse_agent.deep.minimal_agent`) — a measuring stick, not a
    product: page and zoom tools, helpers, and the two output tools."""
    return str(os.environ.get(AGENT_ENV, "")).strip().lower() == "minimal"


def vision_text_context() -> bool:
    """Whether vision calls are given the text layer inside their view."""
    return _on(VISION_TEXT_ENV)


def vision_structured() -> bool:
    """Whether vision calls return located findings as page-point boxes."""
    return _on(VISION_STRUCTURED_ENV)


def vision_inline() -> bool:
    """Whether the main model is shown the images itself (lean agent only)."""
    return _on(VISION_INLINE_ENV)


def sweep() -> bool:
    """Whether the ``sweep_pages`` tool is offered (lean agent only)."""
    return _on(SWEEP_ENV)


def findings() -> bool:
    """Whether the findings tools are offered (lean agent only)."""
    return _on(FINDINGS_ENV)


def overview() -> bool:
    """Whether the orientation turn decides on contact sheets by page count."""
    return _on(OVERVIEW_ENV)


def geometry() -> bool:
    """Whether the drawing-geometry tools are offered (lean agent only)."""
    return _on(GEOMETRY_ENV)


def digest() -> bool:
    """Whether the digest tools are offered (lean agent only)."""
    return _on(DIGEST_ENV)


def coverage() -> bool:
    """Whether the coverage ledger, ``document_coverage`` and the gate are
    on (both pages)."""
    return _on(COVERAGE_ENV)


def checklist() -> bool:
    """Whether ``report_checklist`` is offered (both pages)."""
    return _on(CHECKLIST_ENV)


def kept_readings_on_review_page() -> bool:
    """Whether the Document Review page offers ``kept_readings`` (the
    conversation's kept page readings, TA-28). The geotech page always
    does; on the testers' page it waits for the suite (arm
    ``kept_readings``)."""
    return _on(KEPT_READINGS_ENV)


def vision_turn_retry() -> bool:
    """Whether a look whose lettering is found turned is read again upright,
    and a page found turned is sent turned (both pages)."""
    return _on(VISION_TURN_RETRY_ENV)


def figure_helper() -> bool:
    """Whether the agent gets the ``figure`` helper (both pages)."""
    return _on(FIGURE_HELPER_ENV)


def step_cap_answer() -> bool:
    """Whether a turn stopped at its step cap answers from what it gathered
    (one last call with no tools; both pages)."""
    return _on(STEP_CAP_ANSWER_ENV)


def plot_code() -> bool:
    """Whether the figure and ``calc`` helpers get ``plot_code``, the locked
    box for plotting code, and the figure helper is built (both pages)."""
    return _on(PLOT_CODE_ENV)


def judgment_calls() -> bool:
    """Whether the geotech agent ends an answer with its judgment calls, the
    ``calc`` helper reports the calls it made and the references helper
    returns a reference's several methods or values as cited options."""
    return _on(JUDGMENT_CALLS_ENV)


def cited_pages() -> str:
    """What an answer citing pages no tool opened gets: ``"off"`` (the
    default: recorded only), ``"note"`` (a note appended) or ``"retry"``
    (the model told once, then a note if still needed); both pages."""
    value = str(os.environ.get(CITED_PAGES_ENV, "")).strip().lower()
    if value == "retry":
        return "retry"
    if value == "note" or value in _ON:
        return "note"
    return "off"


def cited_documents() -> str:
    """What an answer or a deliverable citing a document nothing in the
    conversation looked up gets: ``"off"`` (the default: the answer's check
    recorded only), ``"note"`` (a note on the answer, a line in the
    writer's result) or ``"retry"`` (as ``note``, and the model told once
    before the answer goes out); both pages."""
    value = str(os.environ.get(CITED_DOCUMENTS_ENV, "")).strip().lower()
    if value == "retry":
        return "retry"
    if value == "note" or value in _ON:
        return "note"
    return "off"


def overview_pages() -> int:
    """The page count above which the orientation asks for contact sheets
    (``GEOTECH_REVIEW_OVERVIEW_PAGES``, default 20)."""
    try:
        return max(1, int(os.environ.get(OVERVIEW_PAGES_ENV,
                                         DEFAULT_OVERVIEW_PAGES)))
    except (TypeError, ValueError):
        return DEFAULT_OVERVIEW_PAGES


#: Named configurations the review suite compares. ``baseline`` is exactly
#: what the page does with no switch set. Each later arm adds one change, so
#: a gain or a loss is attributable to the change that was added.
ARMS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    "lean": {AGENT_ENV: "lean"},
    "grounded": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
                 VISION_STRUCTURED_ENV: "1"},
    "inline": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
               VISION_STRUCTURED_ENV: "1", VISION_INLINE_ENV: "1"},
    "sweep": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
              VISION_STRUCTURED_ENV: "1", SWEEP_ENV: "1"},
    # Only the orientation turn differs: run it with orientation=True.
    "overview": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
                 VISION_STRUCTURED_ENV: "1", OVERVIEW_ENV: "1"},
    "geometry": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
                 VISION_STRUCTURED_ENV: "1", GEOMETRY_ENV: "1"},
    "digest": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
               VISION_STRUCTURED_ENV: "1", DIGEST_ENV: "1"},
    # The looking-only agent (owner, 2026-10-02): which tools carry weight?
    "minimal": {AGENT_ENV: "minimal"},
    # Coverage in code (plan W4, 2026-10-08). On the DEFAULT agent of each
    # page, so it compares straight against `baseline`: the ledger, the
    # coverage tool and the gate; then the same plus the report checklist.
    "coverage": {COVERAGE_ENV: "1"},
    "checklist": {COVERAGE_ENV: "1", CHECKLIST_ENV: "1"},
    # Turned lettering read again upright (VIS-01, 2026-10-14). On the
    # DEFAULT agent, so it compares straight against `baseline`.
    "turn_retry": {VISION_TURN_RETRY_ENV: "1"},
    # The kept page readings as a tool on the review page (TA-28,
    # 2026-10-15). On the DEFAULT agent, so it compares against `baseline`.
    "kept_readings": {KEPT_READINGS_ENV: "1"},
    # Cited pages no tool opened (EV-18, 2026-10-15): a note on the answer,
    # or the model told once. On the DEFAULT agent, against `baseline`.
    "cited_pages_note": {CITED_PAGES_ENV: "note"},
    "cited_pages_retry": {CITED_PAGES_ENV: "retry"},
    # The figure helper (GM-92, 2026-10-15). On the DEFAULT agent.
    "figure_helper": {FIGURE_HELPER_ENV: "1"},
    # Plotting code in a locked box (GM-98, 2026-10-15) for the figure and
    # calc helpers; it brings the figure helper. On the DEFAULT agent.
    "plot_code": {PLOT_CODE_ENV: "1"},
    # A turn at its step cap answers from what it gathered (TA-33).
    "step_cap_answer": {STEP_CAP_ANSWER_ENV: "1"},
    # Cited documents nothing looked up (EV-21, 2026-10-15): a note on the
    # answer and a line in a writer's result, or the model told once too.
    # On the DEFAULT agent, against `baseline`.
    "cited_documents_note": {CITED_DOCUMENTS_ENV: "note"},
    "cited_documents_retry": {CITED_DOCUMENTS_ENV: "retry"},
    # Judgment calls put in front of the user (GM-116, 2026-10-16). On the
    # DEFAULT agent. The geotech page only: on the review page's tasks this
    # arm builds the same agent as `baseline`.
    "judgment_calls": {JUDGMENT_CALLS_ENV: "1"},
}


@contextmanager
def switches(values: Optional[Mapping[str, str]]) -> Iterator[None]:
    """Set exactly ``values`` for the block: every switch (:data:`ALL_ENVS`)
    and every switch setting (:data:`SETTINGS_ENVS`) NOT named is unset, so
    an arm is what it says and nothing leaks in from the notebook's own
    environment. Any other variable the arm names (a budget, a vision
    policy) is set too. The previous environment is restored afterwards."""
    values = dict(values or {})
    keys = set(ALL_ENVS) | set(SETTINGS_ENVS) | set(values)
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            if k in values and values[k] not in (None, ""):
                os.environ[k] = str(values[k])
            else:
                os.environ.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def describe() -> str:
    """One line naming the switches that are on (for diagnostics and logs)."""
    on = [f"{k}={os.environ[k]}" for k in ALL_ENVS
          if str(os.environ.get(k, "")).strip()]
    return ", ".join(on) if on else "none (legacy review page)"


__all__ = ["AGENT_ENV", "VISION_TEXT_ENV", "VISION_STRUCTURED_ENV",
           "VISION_INLINE_ENV", "SWEEP_ENV", "FINDINGS_ENV", "OVERVIEW_ENV",
           "GEOMETRY_ENV", "DIGEST_ENV", "COVERAGE_ENV", "CHECKLIST_ENV",
           "VISION_TURN_RETRY_ENV", "KEPT_READINGS_ENV", "CITED_PAGES_ENV",
           "FIGURE_HELPER_ENV", "PLOT_CODE_ENV", "STEP_CAP_ANSWER_ENV",
           "CITED_DOCUMENTS_ENV", "JUDGMENT_CALLS_ENV",
           "OVERVIEW_PAGES_ENV",
           "DEFAULT_OVERVIEW_PAGES", "ALL_ENVS", "SETTINGS_ENVS", "ARMS",
           "lean_agent",
           "vision_text_context", "vision_structured", "vision_inline",
           "sweep", "findings", "overview", "geometry", "digest",
           "coverage", "checklist", "vision_turn_retry",
           "kept_readings_on_review_page", "figure_helper", "plot_code",
           "step_cap_answer", "judgment_calls",
           "cited_pages", "cited_documents",
           "overview_pages", "switches", "describe"]
