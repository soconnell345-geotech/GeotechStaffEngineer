"""Switches for the Document Review harness changes of 2026-09-26.

The review of the Document Review page (plan: "Document Review harness —
big-picture review", 2026-09-26) found that every recent change had been judged
on the one example that prompted it. So each behaviour change from that review
ships behind a switch here, OFF by default, and the review suite
(:mod:`funhouse_agent.review_eval`) runs the same tasks with the switches off
and on from ONE install. A switch is turned on by default only once the suite
says it helps, or the owner rules it on: on 2026-10-20 the figure helper,
plotting code and judgment calls (:data:`DEFAULT_ON_ENVS`; ``0`` turns each
off).

Every switch is read from the environment AT CALL TIME, so a notebook (or the
suite) can flip one between two runs without rebuilding anything but the agent.

``GEOTECH_REVIEW_AGENT``
    Unset (default) builds the review page's own agent
    (:mod:`funhouse_agent.review_build`). ``minimal`` builds the
    looking-only agent (:mod:`funhouse_agent.deep.minimal_agent`), a
    measuring stick the suites compare against, not a product. (``lean``,
    the page's own lean build, was removed with its switches in BACKLOG
    DR-39: no suite run showed a gain; ``module_work/dr39/CHECK.md``.)
    (The vision switches ``GEOTECH_VISION_TEXT_CONTEXT``,
    ``GEOTECH_VISION_STRUCTURED`` and ``GEOTECH_VISION_TURN_RETRY``, and
    ``GEOTECH_VISION_PATCH_ALIGN`` in :mod:`funhouse_agent.vision_view`,
    were removed in BACKLOG DR-40 for the same reason, with
    ``GEOTECH_COVERAGE`` (the coverage ledger, ``document_coverage`` and the
    gate; run 7: 39/42 either way, at +47 % input);
    ``module_work/dr40/CHECK.md``.)
``GEOTECH_REVIEW_CHECKLIST``
    ``1`` offers ``report_checklist`` on the GeotechStaffEngineer page: the
    geotechnical report review checklist (DRAFT, OWN-31;
    :mod:`report_ingest.checklist`) with the checks code can make run on the
    record report ingest wrote, and the judgement items returned for the
    agent to report against. Never on the Document Review page (BACKLOG
    RI-34: every geotechnical report question lives in report_ingest).
    With ``GEOTECH_REPORT_INGEST`` on too, its code checks also run on every
    record the ingest writes (BACKLOG RI-35).
``GEOTECH_REPORT_INGEST``
    ``1`` offers report ingest on the GeotechStaffEngineer page: the
    ``report_ingest`` tool and the sub-agent of the same name, which read a
    whole geotechnical report into its record (``report.record.json``), its
    summary, its library page and its DIGGS file, written to
    ``report_ingest/<file name>`` in the conversation's own folder; with
    ``GEOTECH_REVIEW_CHECKLIST`` on, the checklist's code checks run on that
    record and are written beside it (``report.checklist.json``). The
    ingest runs on an engine dug out of the app's own (a Funhouse Prompter);
    where there is none it says so rather than reading the report on
    something else (BACKLOG RI-03). Never on the Document Review page
    (BACKLOG RI-34, RI-35).
``GEOTECH_FIGURE_HELPER``
    ON by default (owner, 2026-10-15, GM-92: "on everywhere"; BACKLOG
    GM-227); ``0`` with ``GEOTECH_PLOT_CODE=0`` leaves it out (plotting
    code brings it). Adds the ``figure`` helper
    (:mod:`funhouse_agent.deep.figure_helper`)
    to the agent: it draws a figure with ``plot_data`` from a figure
    checklist (after Tufte) and worked examples, looks at the PNG and
    redraws (owner's choice, 2026-10-15, OWN-61 option B). Both pages
    (BACKLOG GM-92).
``GEOTECH_PLOT_CODE``
    ON by default (owner, 2026-10-15, GM-92; BACKLOG GM-227); ``0`` takes
    it away (and the figure helper too only with
    ``GEOTECH_FIGURE_HELPER=0``). Offers ``plot_code`` to the figure helper
    and the ``calc`` helper,
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
    ON by default (owner, 2026-10-20, OWN-65: "This cost ($0.2 to 0.53) is
    low compared to the added quality"; BACKLOG GM-226); ``0`` turns it
    off. The chat's "Consult references" control governs the consult: with
    references off no references helper is built, each call names a
    ``choices`` basis or says it rests on judgment, and nothing is sent
    back at turn end; a lower "Reference consult budget" caps the consult
    too. Tells the geotech agent that the engineering judgment is the
    user's: an answer that rests on a judgment call (a method, a value the
    user did not give, drained or undrained, which stratum governs, search
    or mesh settings, how ambiguous data was read) ends with a short
    "Judgment calls" list, each with its reason and reference, the
    alternative set aside and what would change the answer. The ``calc``
    helper's reply lists the calls it made, and the references helper
    returns a reference's several methods or values as cited options. An
    answer whose list has calls that name no source, in a turn where no
    ``references`` task ran, is sent back once asking for one consult
    (:mod:`funhouse_agent.judgment_check`, GM-174). The
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
CHECKLIST_ENV = "GEOTECH_REVIEW_CHECKLIST"
REPORT_INGEST_ENV = "GEOTECH_REPORT_INGEST"
CITED_PAGES_ENV = "GEOTECH_CITED_PAGES"
CITED_DOCUMENTS_ENV = "GEOTECH_CITED_DOCUMENTS"
FIGURE_HELPER_ENV = "GEOTECH_FIGURE_HELPER"
PLOT_CODE_ENV = "GEOTECH_PLOT_CODE"
STEP_CAP_ANSWER_ENV = "GEOTECH_STEP_CAP_ANSWER"
JUDGMENT_CALLS_ENV = "GEOTECH_JUDGMENT_CALLS"

ALL_ENVS = (AGENT_ENV, CHECKLIST_ENV, CITED_PAGES_ENV,
            FIGURE_HELPER_ENV, PLOT_CODE_ENV, STEP_CAP_ANSWER_ENV,
            CITED_DOCUMENTS_ENV, JUDGMENT_CALLS_ENV, REPORT_INGEST_ENV)

#: The switches' SETTINGS (numbers the switched behaviour reads, not
#: switches): how many inline images each call of the minimal build carries,
#: the review page's model-call budget and the plotting box's limits.
#: :func:`switches` clears them too unless the arm names them, so an arm is
#: exactly what it says. Named here as strings so this module imports
#: nothing.
SETTINGS_ENVS = ("GEOTECH_VISION_INLINE_KEEP",
                 "GEOTECH_REVIEW_MAX_MODEL_CALLS",
                 "GEOTECH_PLOT_CODE_SECONDS", "GEOTECH_PLOT_CODE_MAX_SECONDS",
                 "GEOTECH_PLOT_CODE_MEMORY_MB", "GEOTECH_PLOT_CODE_AT_ONCE")

_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")

#: The switches that are ON when their variable is unset, by the owner's
#: ruling rather than the suite's (2026-10-20): the judgment calls (OWN-65,
#: BACKLOG GM-226) and the figure helper and plotting code (GM-92, GM-227).
#: ``0`` (or ``false``/``no``/``off``) turns each off.
DEFAULT_ON_ENVS = (FIGURE_HELPER_ENV, PLOT_CODE_ENV, JUDGMENT_CALLS_ENV)


def _on(env: str) -> bool:
    return str(os.environ.get(env, "")).strip().lower() in _ON


def _on_by_default(env: str) -> bool:
    """On unless the variable says ``0``/``false``/``no``/``off``."""
    return str(os.environ.get(env, "")).strip().lower() not in _OFF


def minimal_agent() -> bool:
    """Whether the review page builds the looking-only agent
    (:mod:`funhouse_agent.deep.minimal_agent`) — a measuring stick, not a
    product: page and zoom tools, helpers, and the two output tools."""
    return str(os.environ.get(AGENT_ENV, "")).strip().lower() == "minimal"


def checklist() -> bool:
    """Whether ``report_checklist`` is offered (the GeotechStaffEngineer
    page only)."""
    return _on(CHECKLIST_ENV)


def report_ingest() -> bool:
    """Whether the GeotechStaffEngineer page offers report ingest (the tool
    and the sub-agent; never the Document Review page)."""
    return _on(REPORT_INGEST_ENV)


def figure_helper() -> bool:
    """Whether the agent gets the ``figure`` helper (both pages; on unless
    ``GEOTECH_FIGURE_HELPER=0``)."""
    return _on_by_default(FIGURE_HELPER_ENV)


def step_cap_answer() -> bool:
    """Whether a turn stopped at its step cap answers from what it gathered
    (one last call with no tools; both pages)."""
    return _on(STEP_CAP_ANSWER_ENV)


def plot_code() -> bool:
    """Whether the figure and ``calc`` helpers get ``plot_code``, the locked
    box for plotting code, and the figure helper is built (both pages; on
    unless ``GEOTECH_PLOT_CODE=0``)."""
    return _on_by_default(PLOT_CODE_ENV)


def judgment_calls() -> bool:
    """Whether the geotech agent ends an answer with its judgment calls, the
    ``calc`` helper reports the calls it made, the references helper
    returns a reference's several methods or values as cited options, and
    an answer with unsourced calls and no consult is sent back once (on
    unless ``GEOTECH_JUDGMENT_CALLS=0``; with the chat's references off
    there is no consult and nothing is sent back)."""
    return _on_by_default(JUDGMENT_CALLS_ENV)


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


#: Named configurations the review suite compares. ``baseline`` is exactly
#: what the page does with no switch set, the switches of
#: :data:`DEFAULT_ON_ENVS` on among it. Each later arm adds one change (or,
#: for a switch on by default, takes it away), so a gain or a loss is
#: attributable to the change.
ARMS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    # The lean build and the arms on it (grounded, inline, sweep, overview,
    # geometry, digest) and kept_readings were removed with their switches
    # in BACKLOG DR-39 (module_work/dr39/CHECK.md).
    # The looking-only agent (owner, 2026-10-02): which tools carry weight?
    "minimal": {AGENT_ENV: "minimal"},
    # The report checklist (plan W4, 2026-10-08), which reaches only the
    # geotech page since RI-34. On the DEFAULT agent of each page, so it
    # compares straight against `baseline`. (It set GEOTECH_COVERAGE too
    # until the coverage ledger was removed in DR-40; the checklist reads
    # report ingest's record, not the ledger.)
    "checklist": {CHECKLIST_ENV: "1"},
    # Cited pages no tool opened (EV-18, 2026-10-15): a note on the answer,
    # or the model told once. On the DEFAULT agent, against `baseline`.
    "cited_pages_note": {CITED_PAGES_ENV: "note"},
    "cited_pages_retry": {CITED_PAGES_ENV: "retry"},
    # The figure helper (GM-92, 2026-10-15). On the DEFAULT agent. On by
    # default since GM-227, so this arm is `baseline`.
    "figure_helper": {FIGURE_HELPER_ENV: "1"},
    # Plotting code in a locked box (GM-98, 2026-10-15) for the figure and
    # calc helpers; it brings the figure helper. On the DEFAULT agent. On by
    # default since GM-227, so this arm is `baseline`.
    "plot_code": {PLOT_CODE_ENV: "1"},
    # GM-227 taken away: no figure helper and no plotting code (the page as
    # it was before 2026-10-20); and the figure helper without plotting code.
    "figure_helper_off": {FIGURE_HELPER_ENV: "0", PLOT_CODE_ENV: "0"},
    "plot_code_off": {PLOT_CODE_ENV: "0"},
    # A turn at its step cap answers from what it gathered (TA-33).
    "step_cap_answer": {STEP_CAP_ANSWER_ENV: "1"},
    # Cited documents nothing looked up (EV-21, 2026-10-15): a note on the
    # answer and a line in a writer's result, or the model told once too.
    # On the DEFAULT agent, against `baseline`.
    "cited_documents_note": {CITED_DOCUMENTS_ENV: "note"},
    "cited_documents_retry": {CITED_DOCUMENTS_ENV: "retry"},
    # Judgment calls put in front of the user (GM-116, 2026-10-16). On the
    # DEFAULT agent. The geotech page only: on the review page's tasks this
    # arm builds the same agent as `baseline`. On by default since GM-226,
    # so this arm is `baseline`, and `judgment_calls_off` takes it away.
    "judgment_calls": {JUDGMENT_CALLS_ENV: "1"},
    "judgment_calls_off": {JUDGMENT_CALLS_ENV: "0"},
}


@contextmanager
def switches(values: Optional[Mapping[str, str]]) -> Iterator[None]:
    """Set exactly ``values`` for the block: every switch (:data:`ALL_ENVS`)
    and every switch setting (:data:`SETTINGS_ENVS`) NOT named is unset (a
    switch of :data:`DEFAULT_ON_ENVS` unset is on), so an arm is what it
    says and nothing leaks in from the notebook's own
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
    """One line naming the switches that are set, and each switch on by
    default as ``=1 (default)`` while unset (for diagnostics and logs)."""
    out = []
    for k in ALL_ENVS:
        value = str(os.environ.get(k, "")).strip()
        if value:
            out.append(f"{k}={value}")
        elif k in DEFAULT_ON_ENVS:
            out.append(f"{k}=1 (default)")
    return ", ".join(out)


__all__ = ["AGENT_ENV", "CHECKLIST_ENV", "CITED_PAGES_ENV",
           "FIGURE_HELPER_ENV", "PLOT_CODE_ENV", "STEP_CAP_ANSWER_ENV",
           "CITED_DOCUMENTS_ENV", "JUDGMENT_CALLS_ENV", "REPORT_INGEST_ENV",
           "DEFAULT_ON_ENVS", "ALL_ENVS", "SETTINGS_ENVS", "ARMS",
           "minimal_agent",
           "checklist", "report_ingest",
           "figure_helper", "plot_code",
           "step_cap_answer", "judgment_calls",
           "cited_pages", "cited_documents",
           "switches", "describe"]
