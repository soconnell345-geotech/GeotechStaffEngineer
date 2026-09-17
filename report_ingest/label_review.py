"""Pass 0c: a model reviews the rule labels with the whole report in view.

:func:`report_ingest.triage.triage` decided what the document is. This pass
decides what each PAGE is, by checking planlens' rule labels against
everything the report says about itself and by looking at the pages the
rules were least sure of.

WHY A LOOP AND NOT A CALL. The rules' misses are not spread evenly. The
out-of-sample measurement sorted every one of them into four classes, and
three of the four are answerable only by opening the page:

* the label came from the appendix tab rather than the page (tag
  ``tab-declares``), and the appendix holds more than one kind of thing;
* the tab named several things and the page named none (``tab-ambiguous``,
  which the rules report as ``other`` with candidates);
* the page has no readable text, so no rule could fire at all.

So the review gets tools -- the page's text, the page as a picture, a contact
sheet, the outline -- and a budget that scales with the document. It spends
them on the pages the rules flagged, then walks the contact sheets once as a
gut check.

WHAT THE MODEL RETURNS, AND WHAT IT DOES NOT. It returns the CHANGES it
wants made, each with a reason and the tool that showed it, the structure it
reconciled, and the pages it could not settle. It does not return a label
for every page: :attr:`Review.final_labels` is the rules' labels with the
changes applied, in Python. A model re-emitting 455 page-to-label pairs can
drop a page or shift a number silently, and the scorecard would score the
slip rather than the judgement. Every change is visible, and a wrong
"correction" shows up in the scorecard as one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from report_ingest.engine import (
    Engine, image_block, text_block, tool_result_block, user,
)

__all__ = [
    "LABEL_DEFINITIONS", "Review", "ReviewFindings", "review_labels",
    "budget_for", "REVIEW_SYSTEM", "REVIEW_TOOLS",
]

#: One line per label. The vocabulary is planlens' ``ROLES``; these are the
#: definitions a reviewer needs to tell the neighbouring ones apart, which is
#: where the rules' errors actually live (a location plan against a figure, a
#: field infiltration test against a laboratory sheet).
LABEL_DEFINITIONS: Dict[str, str] = {
    "narrative": "the report's own prose: findings, discussion, "
                 "recommendations, and the tables inside that prose",
    "figure": "a drawn or plotted figure that is not a site plan and not a "
              "subsurface profile",
    "plan": "a plan view of the site showing where the explorations are",
    "profile": "a subsurface cross-section or fence diagram along a line",
    "boring_log": "the log of one drilled boring: depths, samples, blow "
                  "counts, descriptions",
    "test_pit_log": "the log of one excavated test pit or trench",
    "cpt_log": "a cone penetration test sounding: tip, sleeve, pore pressure "
               "against depth",
    "dcp_log": "a dynamic cone penetrometer record",
    "lab_test": "a laboratory test result sheet or a table of laboratory "
                "results",
    "field_test": "a test performed in the field that is not an exploration "
                  "log: infiltration, percolation, permeability, density",
    "calculation": "a calculation printout, worksheet or program output",
    "appended_report": "a page of a different, complete report bound inside "
                       "this one",
    "photos": "site or core photographs",
    "divider": "an appendix tab or fly sheet that names what follows",
    "cover": "a title page of the report or of a volume",
    "letter": "a transmittal or cover letter",
    "toc": "a table of contents, or a list of figures, tables or appendices",
    "other": "none of the above, or not decidable from the page",
}

#: Rules at or below this confidence are the rules saying they are unsure.
LOW_CONFIDENCE = 0.65
#: Evidence tags that mean the label did not come from the page itself.
WEAK_TAGS: Tuple[str, ...] = ("tab-declares", "tab-ambiguous", "page-shape",
                              "between-pages-of-one-log", "run-continuation")
#: Pages per contact sheet, and the dpi a single page is rendered at. 80 dpi
#: is a letter page at about 950 px on its long side: enough to read a log's
#: title block and column headers, not enough to read a lab sheet's fine
#: print, which is what ``read_page`` is for.
CONTACT_SHEET_PAGES = 48
PAGE_DPI = 80.0
#: Floor on the tool-call budget, and how many calls each page buys above it.
MIN_BUDGET = 60
BUDGET_PER_PAGE = 0.25
#: Ceiling on one page's text, so a dense narrative page cannot eat the
#: context a hundred spot-checks need.
MAX_PAGE_CHARS = 14000


def budget_for(n_pages: int) -> int:
    """Tool calls this review may spend: ``max(60, 0.25 x pages)``."""
    return max(MIN_BUDGET, int(BUDGET_PER_PAGE * int(n_pages)))


# -- the tools --------------------------------------------------------------

REVIEW_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_page",
        "description": (
            "The text of one page as the document readers see it: every text "
            "line in reading order, the tables, and a note where the text is "
            "not what the page shows. Use this to settle what a page IS when "
            "the page has readable text."),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer",
                         "description": "0-based PDF page index"},
            },
            "required": ["page"],
            "additionalProperties": False,
        },
    },
    {
        "name": "render_page",
        "description": (
            "One page as a picture. Use it when the page has no reliable "
            "text (text_ok=N in the ledger), when the text and the ledger "
            "disagree, or when the layout rather than the words decides what "
            "the page is - a log form, a plan, a profile, a photo sheet."),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer",
                         "description": "0-based PDF page index"},
            },
            "required": ["page"],
            "additionalProperties": False,
        },
    },
    {
        "name": "contact_sheet",
        "description": (
            "A grid of page thumbnails, 48 pages per sheet, each labelled "
            "with its 0-based page index and its kind. Walk every sheet once "
            "as a gut check: a run of pages that all look alike but carry "
            "different labels, or a page that looks nothing like its "
            "neighbours, is where the rules went wrong."),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_page": {
                    "type": "integer",
                    "description": "0-based index of the first page on the "
                                   "sheet; sheets start at 0, 48, 96, ..."},
            },
            "required": ["start_page"],
            "additionalProperties": False,
        },
    },
    {
        "name": "outline",
        "description": (
            "What the document prints about itself: the contents list, the "
            "lists of figures, tables and appendices, every divider's text, "
            "the figure captions and the narrative's section headings. Given "
            "to you already; call this to see it again in full."),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


class _Tools:
    """The tool implementations, over one open document."""

    def __init__(self, doc, outline_text: str) -> None:
        self.doc = doc
        self.outline_text = outline_text

    def _page(self, arguments: Dict[str, Any]) -> int:
        page = int(arguments["page"])
        if not 0 <= page < self.doc.n_pages:
            raise IndexError(
                f"page {page} is outside this document (0 to "
                f"{self.doc.n_pages - 1})")
        return page

    def read_page(self, arguments: Dict[str, Any]) -> Any:
        from planlens.document.advice import page_advice
        from planlens.tools.formatting import render_page

        page = self._page(arguments)
        pc = self.doc.page(page, tables=True)
        summary = self.doc.summary(page)
        rows = render_page(pc, summary, include_tables=True,
                           include_markups=True, with_locations=False,
                           advice=page_advice(summary, pc))
        text = "\n".join(rows)
        if len(text) > MAX_PAGE_CHARS:
            text = (text[:MAX_PAGE_CHARS]
                    + "\n[page text cut at the size limit; render_page to see "
                      "the rest of the layout]")
        return text

    def render_page(self, arguments: Dict[str, Any]) -> Any:
        page = self._page(arguments)
        png, info = self.doc.render(page, dpi=PAGE_DPI)
        return [text_block(f"page {page}, rendered at {info['dpi']:.0f} dpi"),
                image_block(png)]

    def contact_sheet(self, arguments: Dict[str, Any]) -> Any:
        start = int(arguments["start_page"])
        if not 0 <= start < self.doc.n_pages:
            raise IndexError(
                f"start_page {start} is outside this document (0 to "
                f"{self.doc.n_pages - 1})")
        start -= start % CONTACT_SHEET_PAGES
        end = min(start + CONTACT_SHEET_PAGES, self.doc.n_pages) - 1
        sheets = self.doc.render_thumbnails(pages=f"{start}-{end}",
                                            per_sheet=CONTACT_SHEET_PAGES)
        if not sheets:
            raise ValueError(f"no pages between {start} and {end}")
        png, info = sheets[0]
        return [text_block(f"pages {start} to {end}; {info['legend']}"),
                image_block(png)]

    def outline(self, arguments: Dict[str, Any]) -> Any:
        return self.outline_text

    def run(self, name: str, arguments: Dict[str, Any]) -> Tuple[Any, bool]:
        """``(content, is_error)`` -- a tool mistake is an answer, not a stop."""
        handler = getattr(self, name, None)
        if handler is None or name not in {t["name"] for t in REVIEW_TOOLS}:
            return (f"unknown tool {name!r}; the tools are "
                    f"{[t['name'] for t in REVIEW_TOOLS]}", True)
        try:
            return handler(arguments), False
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            return f"{type(exc).__name__}: {exc}", True


# -- what the model returns -------------------------------------------------

class LabelChange(BaseModel):
    """One page whose label the review wants changed."""

    page: int = Field(description="0-based PDF page index")
    from_label: str = Field(description="the label the rules gave it")
    to_label: str = Field(
        description="the label it should have; one of the vocabulary")
    reason: str = Field(description="why, in 25 words or fewer")
    evidence: str = Field(
        description="what showed it: read_page, render_page, contact_sheet, "
                    "outline, or ledger")


class StructureEntry(BaseModel):
    """One section or appendix, and the pages it turned out to occupy."""

    title: str = Field(description="the title as the document prints it")
    pages: str = Field(
        description="0-based PDF page range, e.g. '24-58' or '3,7-9'")


class UnresolvedPage(BaseModel):
    """A page the review could not settle."""

    page: int = Field(description="0-based PDF page index")
    why: str = Field(description="what stopped you, in 20 words or fewer")


class ReviewFindings(BaseModel):
    """The review's answer: what to change, what the structure is, what is
    still open."""

    changes: List[LabelChange] = Field(
        description="every label you want changed; empty when the rules were "
                    "right everywhere you looked")
    structure: List[StructureEntry] = Field(
        description="the sections and appendices reconciled against the "
                    "pages they actually occupy")
    unresolved: List[UnresolvedPage] = Field(
        description="pages you could not settle, with why")
    notes: str = Field(
        description="anything a reader of this report should know, in 60 "
                    "words or fewer; empty string when there is nothing")


@dataclass
class Review:
    """The reviewed labels, and everything needed to audit them."""

    final_labels: Dict[int, str]
    changes: List[Dict[str, Any]]
    structure: List[Dict[str, Any]]
    unresolved: List[Dict[str, Any]]
    rules_labels: Dict[int, str] = field(default_factory=dict)
    notes: str = ""
    tool_calls: int = 0
    budget: int = 0
    stopped_on_budget: bool = False
    model_calls: int = 0
    model: str = ""
    cost: Dict[str, Any] = field(default_factory=dict)
    rejected_changes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_labels": {str(k): v for k, v in
                             sorted(self.final_labels.items())},
            "changes": [dict(c) for c in self.changes],
            "rejected_changes": [dict(c) for c in self.rejected_changes],
            "structure": [dict(s) for s in self.structure],
            "unresolved": [dict(u) for u in self.unresolved],
            "notes": self.notes,
            "tool_calls": self.tool_calls,
            "budget": self.budget,
            "stopped_on_budget": self.stopped_on_budget,
            "model_calls": self.model_calls,
            "model": self.model,
            "cost": dict(self.cost),
        }


REVIEW_SYSTEM = """\
You are reviewing the page labels a rule engine put on one engineering
report. The rules read each page on its own; you have the whole report. Your
job is to correct the labels that the report's own context shows are wrong,
and to leave alone the ones that are right.

What the rules get wrong, measured on reports like this one:

1. A label inherited from an appendix tab when the page says otherwise. The
   tag [tab-declares] on a ledger line means exactly this: the label is the
   TAB talking, not the page. An appendix with logs, lab sheets and photos
   mixed together is where this bites.
2. A tab that names several things and a page that names none: the rules
   emit 'other' with candidates, tagged [tab-ambiguous]. Those pages are
   nearly always one of the candidates.
3. No appendix tab anywhere, so nothing inherits and everything falls back
   to the page's shape. A typed data report with no dividers had every lab
   sheet called narrative.
4. A page with no readable text (text_ok=N and di=N). No rule can fire.
   Render it and label it from the picture.
5. A single-page cue misfiring: a distribution list called a divider, an
   aerial photograph called photos when it is the site plan, a flood map
   called a calculation.

How to work.

- Start with the pages the brief lists as the rules' weak spots. Those are
  where the errors are.
- Reconcile the contents and appendix lists against page ranges. If the
  figure list says Figure 2 is the boring location plan, the page whose
  caption reads Figure 2 is 'plan'. If Appendix C is "Laboratory Test
  Results", a form page inside it the rules left as 'other' is almost
  certainly 'lab_test'.
- Walk every contact sheet once, in order. It is the cheapest way to see a
  run of pages that all look alike but carry different labels.
- Spot-check every low-confidence page, and any page whose label contradicts
  its neighbours or the appendix it sits in.
- A page with text_ok=N and di=N must be rendered, not read. Label it from
  the picture.
- Prefer read_page when a page has text; it is cheaper and more exact than
  looking. Look when the layout, not the words, decides.
- Do not change a label you have not checked. A confident wrong correction
  is worse than leaving a rule label alone: every change you make is scored
  against a hand label, and a wrong one counts against you.
- You have a tool-call budget. Spend it on the flagged pages first, then the
  contact sheets, then anything still nagging. You will be told when it runs
  low.

When you are done, you will be asked for your findings in a fixed shape:
the changes, the structure you reconciled, and the pages you could not
settle. Report a page only once.
"""

_FINAL_INSTRUCTION = (
    "Now give your findings. Include every label you want changed (page, "
    "the label the rules gave it, the label it should have, why in 25 words "
    "or fewer, and which tool showed it), the sections and appendices with "
    "the page ranges they actually occupy, and any page you could not "
    "settle. If the rules were right everywhere you looked, return no "
    "changes."
)


def _weak_spots(roles: Sequence[Any], summaries: Sequence[Any],
                no_dividers: bool) -> List[str]:
    """The pages the review should look at first, and why, one line each."""
    by_page = {s.page: s for s in summaries}
    label = {r.page: r.role for r in roles}
    out: List[str] = []
    for r in roles:
        s = by_page.get(r.page)
        tag = (r.evidence or {}).get("tag") or ""
        why: List[str] = []
        if tag == "tab-declares":
            why.append("label came from the appendix tab, not the page")
        elif tag == "tab-ambiguous":
            cands = (r.evidence or {}).get("candidates") or []
            why.append("tab names several things, page names none; "
                       "candidates " + ", ".join(cands))
        elif tag == "page-shape":
            why.append("guessed from the page's shape alone")
        if r.confidence <= LOW_CONFIDENCE and not why:
            why.append(f"low confidence {r.confidence:.2f}")
        if s is not None and (not s.text_reliable or s.n_text_chars == 0):
            why.append("no readable text: render it")
        prev_label = label.get(r.page - 1)
        next_label = label.get(r.page + 1)
        if (prev_label is not None and next_label is not None
                and prev_label == next_label and r.role != prev_label):
            why.append(f"both neighbours are {prev_label}")
        if why:
            out.append(f"p{r.page:03d} {r.role}: " + "; ".join(why))
    if no_dividers:
        out.insert(0, "THE DOCUMENT PRINTS NO APPENDIX TAB ANYWHERE. Nothing "
                      "inherited a label, so every label below page one is "
                      "the page's own shape. Check widely, not just the "
                      "flagged pages.")
    return out


def _brief(ledger: str, outline_text: str, profile: Any,
           weak: Sequence[str], budget: int, n_pages: int) -> str:
    vocab = "\n".join(f"  {name}: {text}"
                      for name, text in LABEL_DEFINITIONS.items())
    profile_text = (json.dumps(profile.to_dict(), indent=2, default=str)
                    if hasattr(profile, "to_dict")
                    else json.dumps(profile, indent=2, default=str))
    weak_text = "\n".join(weak) if weak else (
        "none: every page named itself and the rules were confident "
        "throughout. Still walk the contact sheets.")
    return "\n".join([
        f"This document has {n_pages} pages. Your tool-call budget is "
        f"{budget}.",
        "",
        "THE LABEL VOCABULARY",
        vocab,
        "",
        "WHAT TRIAGE DECIDED THIS DOCUMENT IS",
        profile_text,
        "",
        "WHAT THE DOCUMENT PRINTS ABOUT ITSELF",
        outline_text,
        "",
        "THE RULES' WEAK SPOTS: CHECK THESE FIRST",
        weak_text,
        "",
        "THE PAGE LEDGER, ONE LINE PER PAGE",
        ledger,
    ])


def review_labels(doc, roles=None, outline=None, profile=None, budget=None, *,
                  engine: Engine, max_model_calls: int = 60) -> Review:
    """Run pass 0c over one open document and return the reviewed labels.

    ``budget`` is the tool-call ceiling; it defaults to
    :func:`budget_for` of the page count. ``max_model_calls`` is a
    belt-and-braces stop on the loop itself, so a model that answers without
    spending tools cannot spin.
    """
    from planlens.document.roles import document_outline, page_ledger, page_roles
    from report_ingest.triage import outline_text as _outline_text

    if roles is None:
        roles = page_roles(doc)
    if outline is None:
        outline = document_outline(doc)
    if budget is None:
        budget = budget_for(doc.n_pages)
    budget = int(budget)

    summaries = list(doc.page_map())
    rules_labels = {r.page: r.role for r in roles}
    outline_text = _outline_text(outline)
    weak = _weak_spots(roles, summaries, bool(outline.no_dividers))
    brief = _brief("\n".join(page_ledger(doc, roles)), outline_text, profile,
                   weak, budget, doc.n_pages)

    tools = _Tools(doc, outline_text)
    messages: List[Dict[str, Any]] = [user(text_block(brief))]
    calls = 0
    model_calls = 0
    stopped_on_budget = False
    # This pass's OWN cost. The engine's meter may be shared with triage or
    # with a whole scorecard run, so it cannot answer "what did the review
    # cost on this report".
    spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0}

    def _charge(reply) -> None:
        spent["calls"] += 1
        spent["input_tokens"] += reply.usage.input_tokens
        spent["output_tokens"] += reply.usage.output_tokens
        spent["cache_read_tokens"] += reply.usage.cache_read_tokens
        spent["seconds"] += reply.seconds
        spent["dollars"] += reply.usage.dollars(reply.model)

    while True:
        reply = engine.complete(messages, system=REVIEW_SYSTEM,
                                tools=REVIEW_TOOLS)
        model_calls += 1
        _charge(reply)
        messages.append({"role": "assistant", "content": reply.content})
        if not reply.tool_calls:
            break
        results: List[Dict[str, Any]] = []
        for call in reply.tool_calls:
            if calls >= budget:
                stopped_on_budget = True
                results.append(tool_result_block(
                    call.id,
                    "tool-call budget spent. Stop looking and report your "
                    "findings from what you have seen.", is_error=True))
                continue
            calls += 1
            content, is_error = tools.run(call.name, call.arguments)
            results.append(tool_result_block(call.id, content,
                                             is_error=is_error))
        left = budget - calls
        if 0 < left <= max(5, budget // 10):
            results.append(text_block(
                f"{left} tool calls left of {budget}. Finish what you are "
                f"checking and get ready to report."))
        messages.append({"role": "user", "content": results})
        if model_calls >= max_model_calls:
            stopped_on_budget = stopped_on_budget or calls >= budget
            break

    messages.append(user(text_block(_FINAL_INSTRUCTION)))
    final = engine.complete(messages, output_format=ReviewFindings,
                            system=REVIEW_SYSTEM)
    model_calls += 1
    _charge(final)
    found = final.parsed
    if found is None:
        raise RuntimeError(
            f"the label review returned no structured findings (stop_reason "
            f"{final.stop_reason!r})")

    labels = dict(rules_labels)
    changes: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen: set = set()
    for c in found.changes:
        row = {"page": int(c.page), "from": c.from_label, "to": c.to_label,
               "reason": c.reason, "evidence": c.evidence}
        if c.page not in rules_labels:
            row["rejected"] = "no such page in this document"
            rejected.append(row)
            continue
        if c.to_label not in LABEL_DEFINITIONS:
            row["rejected"] = f"{c.to_label!r} is not in the vocabulary"
            rejected.append(row)
            continue
        if c.page in seen:
            row["rejected"] = "the page was already changed once"
            rejected.append(row)
            continue
        # The model's idea of the old label is not authoritative; the rules
        # are. Recording both is how a confused change shows up later.
        row["from"] = rules_labels[c.page]
        if row["from"] == c.to_label:
            row["rejected"] = "no change: it already has that label"
            rejected.append(row)
            continue
        seen.add(c.page)
        labels[c.page] = c.to_label
        changes.append(row)

    spent["seconds"] = round(spent["seconds"], 1)
    spent["dollars"] = round(spent["dollars"], 4)
    return Review(
        final_labels=labels,
        changes=changes,
        structure=[s.model_dump() for s in found.structure],
        unresolved=[u.model_dump() for u in found.unresolved],
        rules_labels=rules_labels,
        notes=found.notes,
        tool_calls=calls,
        budget=budget,
        stopped_on_budget=stopped_on_budget,
        model_calls=model_calls,
        model=final.model or getattr(engine, "name", ""),
        cost=spent,
        rejected_changes=rejected,
    )
