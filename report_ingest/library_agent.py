"""The library as a sub-agent: questions ACROSS reports, answered with pages.

The ingester reads one report. This answers questions about all of them --
"which of these sites was called potentially liquefiable", "what did each
firm recommend", "which report and page prints that bearing pressure" --
and it answers them ONLY from the records the ingester already wrote.

WHY IT IS A SUB-AGENT RATHER THAN A TOOL. A question about a library is
several queries: list the reports that match, pull the field from each, go
and find the one that phrased it differently. Done on the primary agent that
is ten tool results of tables in the main conversation, and the primary
would still have to decide what to ask next. Done here it is one delegation
and what comes back is a paragraph with its citations.

WHAT IT MAY SAY. Nothing that is not in a tool result. Every fact carries
``(report id, page)``; a question the library cannot answer is said to be
unanswerable rather than answered from what a geotechnical engineer would
usually expect; a report that was never ingested is not in the library and
saying so is the right answer. The tools cannot reach a PDF, the filesystem
or the network -- they read the records, so there is nothing else for an
answer to come from.

THE CEILINGS ARE PYTHON, not middleware. deepagents uses a CompiledSubAgent's
runnable AS PROVIDED and never reads a spec's middleware for one -- the same
note as :mod:`report_ingest.subagent` -- so the limits that actually hold are
in this module: :data:`MAX_TOOL_CALLS` queries per answer, and every query's
own rows and characters capped before the result reaches the model.

THE MODEL IS THE APP'S. Unlike the ingest, which runs readers on its own
engine one bounded call at a time, this sub-agent is a model with tools and
the model is whatever the host built the agent with.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

__all__ = [
    "build_report_library_subagent", "build_report_library_graph",
    "LibraryAnswer", "LIBRARY_TOOL_SPECS", "run_query", "QueryResult",
    "library_available", "SUBAGENT_NAME", "SUBAGENT_DESCRIPTION",
    "LIBRARY_SYSTEM_PROMPT", "MAX_TOOL_CALLS", "MAX_ROWS_PER_CALL",
    "MAX_RESULT_CHARS",
]

#: The name the primary delegates to, and the tool's own name.
SUBAGENT_NAME = "report_library"

SUBAGENT_DESCRIPTION = (
    "Answers questions ACROSS the geotechnical reports that have already "
    "been read into records -- which reports name a post or a phase, what "
    "each one recommended, where a value is printed, where two of them "
    "disagree, what the library holds. It reads the records and nothing "
    "else, cites the report id and the page behind every fact, and says so "
    "when the library holds no answer. It does NOT read new PDFs: a report "
    "that has not been ingested is not in the library, and `report_ingest` "
    "is what puts it there."
)

#: How many queries one answer may run. A library question is two or three
#: queries; eight is room to follow a thread and a hard stop on a loop.
MAX_TOOL_CALLS = 8
#: How many rows one query result may carry into the model's context.
MAX_ROWS_PER_CALL = 25
#: And how many characters, whichever is hit first.
MAX_RESULT_CHARS = 4000
#: How many model turns the graph will take. Each turn is one model call and
#: at most one batch of queries, so this bounds the run even if the model
#: asks for one query at a time.
MAX_TURNS = MAX_TOOL_CALLS + 2

LIBRARY_SYSTEM_PROMPT = """\
You answer questions about a LIBRARY of geotechnical reports that have
already been read into records. You have tools that query those records.

THE ONE RULE: every fact in your answer comes from a tool result in this
conversation. Not from what a geotechnical report usually says, not from the
wording of the question, not from what you know about the ground anywhere.
If the tools did not return it, you do not know it.

CITE EVERY FACT as `(report id, page)` -- for example `(L02, p23)` -- using
the report id and the page the tool result printed beside that value. A fact
whose tool result carried no page is cited with the report id alone:
`(L02)`. Never write a page number a tool result did not print.

WHEN THE LIBRARY DOES NOT HOLD THE ANSWER, say so plainly, name what you
looked for and which tools you tried, and stop. That is a complete answer.
Do not fill the space with what the answer probably is. A report nobody has
ingested is not in the library; say that rather than guessing at it.

END with one line per thing you could not settle, each beginning `Gap:`.
Write no `Gap:` lines when the tools answered the question in full.

HOW TO WORK.
- `library_stats` and `list_reports` are where a question about the library
  as a whole starts; `list_reports` filters by post, property type, phase,
  firm, document type, a date range and whether a report holds a kind of
  exploration or test.
- `facts` pulls the narrative fields out of one report and `compare` puts
  one field side by side across several -- those are the fastest answers to
  "what did each of them say about X" and they carry the pages.
- `find` searches the text of every record, page and summary; `where_is`
  answers "which report and page prints this".
- `explorations`, `lab_summary` and `calculations` are the detail of one
  report: the holes and their depths, the tests and their values, what was
  worked out.
- `disagreements` lists what a person should look at: the places where two
  readings of the same report disagree, a count does not add up, or a page
  could not be read.

A REPORT BOUND INSIDE ANOTHER ONE is a report of this library in its own
right, with its own id. Its borings are ITS borings, not its parent's; when
you name one, say which report it was bound inside.

Keep the answer short. A reader wants the answer and the pages to check it
against, not a summary of the library.
"""


# ---------------------------------------------------------------------------
# the tools: JSON-schema specs, and one dispatcher
# ---------------------------------------------------------------------------

def _string(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}


#: The query layer as tool specifications, in JSON Schema. Plain dicts rather
#: than framework objects so this module imports nothing from LangChain until
#: a graph is actually built; :func:`_langchain_tools` converts them.
LIBRARY_TOOL_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "library_stats",
        "description": "What the library holds: how many reports, over what "
                       "years, by document type, status and confidence, the "
                       "firms and posts that appear, and the totals of "
                       "explorations, laboratory tests and calculations. "
                       "Start here when the question is about the library "
                       "rather than about a report.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_reports",
        "description": "The reports of the library that match every filter "
                       "given; with no filters, all of them. Each row "
                       "carries the report id, the title, the firm, the "
                       "post, the phase, the date, the page count and what "
                       "the record holds.",
        "parameters": {
            "type": "object",
            "properties": {
                "post": _string("the post, matched as a substring"),
                "property_type": _string("the property type"),
                "phase": _string("the project phase"),
                "firm": _string("the geotechnical engineer's firm"),
                "document_type": _string(
                    "geotechnical report, environmental report, "
                    "recommendation letter, report addendum, report "
                    "appendix or figure(s), partial report, other"),
                "date_from": _string("earliest report date, YYYY-MM-DD"),
                "date_to": _string("latest report date, YYYY-MM-DD"),
                "has_kind": _string(
                    "only reports holding this kind: an exploration kind "
                    "(boring, test_pit, cpt, dcp), a laboratory test kind "
                    "(atterberg, gradation, triaxial, chemical ...), "
                    "'calculations', or 'bound' for a report with another "
                    "report bound inside it"),
            },
            "required": [],
        },
    },
    {
        "name": "find",
        "description": "Search the whole library -- the records, the library "
                       "pages and the summaries. Returns the report, the "
                       "pages behind the hit and a snippet. Use it for "
                       "anything phrased in the report's own words rather "
                       "than as a field.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": _string("what to search for"),
                "k": {"type": "integer",
                      "description": "how many hits, 1 to 25 (default 8)"},
                "section": _string(
                    "narrow to one kind of content: document, general, "
                    "natural_hazards, exploration, lab, calculation, qa, "
                    "bound, page, summary"),
                "report_id": _string("narrow to one report"),
            },
            "required": ["text"],
        },
    },
    {
        "name": "where_is",
        "description": "Which report and which page prints this. The same "
                       "search as `find`, shaped as places to look.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": _string("the words to locate"),
                "k": {"type": "integer",
                      "description": "how many places (default 8)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "facts",
        "description": "What one report answered, for the 37 narrative "
                       "fields the reading fills in, each with its pages and "
                       "the quote it came from. Also its counts and its QA "
                       "summary. A field the report did not answer is listed "
                       "as not answered rather than returned empty.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": _string("the report"),
                "fields": {
                    "type": "array", "items": {"type": "string"},
                    "description": "the field names wanted, e.g. siteClass, "
                                   "recommendedFoundations, "
                                   "liquefactionPotential, postName, "
                                   "bearingCapacity; omit for every field "
                                   "the report answered"},
            },
            "required": ["report_id"],
        },
    },
    {
        "name": "compare",
        "description": "One narrative field across several reports, as a "
                       "table with the pages. The reports that did not "
                       "answer the field are named.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": _string("the narrative field to compare"),
                "report_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "the reports to compare; omit for every "
                                   "report in the library"},
            },
            "required": ["field"],
        },
    },
    {
        "name": "explorations",
        "description": "One report's explorations: the holes, pits and "
                       "soundings with their depths, layers, samples, driven "
                       "records and water levels, each with its pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": _string("the report"),
                "kind": _string("boring, test_pit, cpt or dcp; omit for all"),
            },
            "required": ["report_id"],
        },
    },
    {
        "name": "lab_summary",
        "description": "One report's laboratory testing: how many of each "
                       "kind, and each test with the hole and sample it came "
                       "from, its depth, its standard, its values and its "
                       "pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": _string("the report"),
                "kind": _string("one test kind; omit for all"),
            },
            "required": ["report_id"],
        },
    },
    {
        "name": "calculations",
        "description": "What one report worked out: each calculation "
                       "printout with the program, the method, what it was "
                       "for, what it concluded and its pages.",
        "parameters": {
            "type": "object",
            "properties": {"report_id": _string("the report")},
            "required": ["report_id"],
        },
    },
    {
        "name": "disagreements",
        "description": "The quality-assurance entries a person should look "
                       "at: two readings that disagree, a count that does "
                       "not add up, a value out of range, a page that could "
                       "not be read. For one report, or for the whole "
                       "library.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": _string("one report; omit for the whole "
                                     "library"),
            },
            "required": [],
        },
    },
)

#: The tool names, for a caller that wants to check the surface.
TOOL_NAMES: Tuple[str, ...] = tuple(spec["name"] for spec in
                                    LIBRARY_TOOL_SPECS)


@dataclass
class QueryResult:
    """One query's answer, as the model sees it and as Python checks it."""

    #: The compact text the model is given.
    text: str = ""
    #: ``(report id, page or None)`` for everything the query returned. What
    #: the answer's citations are checked against.
    rows: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    #: The reports this query touched.
    reports: List[str] = field(default_factory=list)
    #: True when the query ran and found nothing. A gap, not an error.
    empty: bool = False
    error: str = ""


def _cite(report: str, pages: Sequence[Any]) -> str:
    """``(L02, p7, p8)``, or ``(L02)`` where the record carried no page."""
    numbers = [f"p{page}" for page in (pages or ()) if isinstance(page, int)]
    return f"({report}{', ' + ', '.join(numbers) if numbers else ''})"


def _rows_of(report: str, pages: Sequence[Any]
             ) -> List[Tuple[str, Optional[int]]]:
    numbers = [page for page in (pages or ()) if isinstance(page, int)]
    return [(report, page) for page in numbers] or [(report, None)]


def run_query(library: Any, name: str, arguments: Optional[Dict[str, Any]]
              = None, *, max_rows: int = MAX_ROWS_PER_CALL,
              max_chars: int = MAX_RESULT_CHARS) -> QueryResult:
    """Run one library query and render it as compact, cited text.

    The whole deterministic half of this sub-agent: no model is involved, so
    every tool the model can call is testable on its own and the answer's
    citations are checkable against what came back.
    """
    args = dict(arguments or {})
    if name not in TOOL_NAMES:
        return QueryResult(error=f"there is no library query called "
                                 f"{name!r}; the queries are "
                                 f"{', '.join(TOOL_NAMES)}",
                           text=f"no such query: {name}")
    handler = _HANDLERS[name]
    try:
        answer = handler(library, args, max_rows)
    except Exception as exc:                     # one query, not the answer
        return QueryResult(
            error=f"{type(exc).__name__}: {exc}",
            text=f"the query failed: {type(exc).__name__}: {exc}")
    if len(answer.text) > max_chars:
        answer.text = answer.text[:max_chars].rstrip() + "\n... (cut)"
    return answer


def _q_library_stats(library: Any, args: Dict[str, Any],
                     max_rows: int) -> QueryResult:
    stats = library.library_stats()
    years = stats.get("years") or {}
    totals = stats.get("totals") or {}
    lines = [
        f"The library holds {stats['reports']} report(s) over "
        f"{stats['pages']} page(s)"
        + (f", {years['first']} to {years['last']}"
           if years.get("first") else "") + ".",
        f"{stats.get('bound_inside_another', 0)} of them were bound inside "
        f"another report.",
        f"Totals: {totals.get('explorations', 0)} exploration(s), "
        f"{totals.get('lab_tests', 0)} laboratory test(s), "
        f"{totals.get('calculations', 0)} calculation(s), "
        f"{totals.get('qa_entries', 0)} QA entr(ies) of which "
        f"{totals.get('needing_review', 0)} want a person.",
    ]
    for label in ("by_document_type", "by_status", "by_confidence", "firms",
                  "posts", "property_types", "phases", "kinds"):
        counts = stats.get(label) or {}
        if counts:
            lines.append(
                f"{label.replace('_', ' ')}: "
                + ", ".join(f"{key} ({value})" for key, value
                            in list(counts.items())[:max_rows]))
    if stats.get("could_not_be_read"):
        lines.append("could not be read: " + ", ".join(
            row["report"] for row in stats["could_not_be_read"]))
    return QueryResult(text="\n".join(lines),
                       empty=not stats.get("reports"))


def _q_list_reports(library: Any, args: Dict[str, Any],
                    max_rows: int) -> QueryResult:
    answer = library.list_reports(
        post=args.get("post", ""), property_type=args.get("property_type", ""),
        phase=args.get("phase", ""), firm=args.get("firm", ""),
        date_from=args.get("date_from", ""), date_to=args.get("date_to", ""),
        document_type=args.get("document_type", ""),
        has_kind=args.get("has_kind", ""), limit=max_rows)
    rows, reports, lines = [], [], []
    for row in answer["reports"]:
        reports.append(row["report"])
        rows += _rows_of(row["report"], [])
        held = (f"{row['explorations']} exploration(s), "
                f"{row['lab_tests']} lab test(s), "
                f"{row['calculations']} calculation(s)")
        detail = " | ".join(x for x in (
            row["title"], row["firm"], row["post"], row["phase"],
            row["document_type"], row["date"] or str(row["year"] or ""),
            f"{row['n_pages']} pp", held,
            (f"bound inside {row['bound_inside']}"
             if row.get("bound_inside") else "")) if x)
        lines.append(f"{row['report']}: {detail} {_cite(row['report'], [])}")
    if not lines:
        filters = answer.get("filters") or {}
        return QueryResult(
            text="No report in the library matches "
                 + (", ".join(f"{k}={v!r}" for k, v in filters.items())
                    if filters else "that") + ".",
            empty=True)
    head = f"{answer['n']} report(s) match" + (
        " (showing the first %d)" % len(lines) if answer["truncated"] else "")
    return QueryResult(text=head + ":\n" + "\n".join(lines), rows=rows,
                       reports=reports)


def _hits_text(hits: Sequence[Dict[str, Any]]) -> Tuple[
        str, List[Tuple[str, Optional[int]]], List[str]]:
    rows: List[Tuple[str, Optional[int]]] = []
    reports: List[str] = []
    lines: List[str] = []
    for index, hit in enumerate(hits, start=1):
        rows += _rows_of(hit["report"], hit.get("pages") or [])
        reports.append(hit["report"])
        lines.append(f"{index}. {_cite(hit['report'], hit.get('pages') or [])}"
                     f" {hit.get('section', '')}/{hit.get('subject', '')}"
                     f" -- {hit.get('snippet', '')}")
    return "\n".join(lines), rows, reports


def _q_find(library: Any, args: Dict[str, Any],
            max_rows: int) -> QueryResult:
    answer = library.find(str(args.get("text", "")),
                          k=min(int(args.get("k") or 8), max_rows),
                          section=str(args.get("section") or ""),
                          report_id=str(args.get("report_id") or ""))
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]), error=str(
            answer["error"]), empty=True)
    if not answer["hits"]:
        return QueryResult(
            text=f"Nothing in the library matches {answer['query']!r}.",
            empty=True)
    text, rows, reports = _hits_text(answer["hits"])
    return QueryResult(text=f"{answer['n']} hit(s) for "
                            f"{answer['query']!r}:\n" + text,
                       rows=rows, reports=reports)


def _q_where_is(library: Any, args: Dict[str, Any],
                max_rows: int) -> QueryResult:
    answer = library.where_is(str(args.get("text", "")),
                              k=min(int(args.get("k") or 8), max_rows))
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]),
                           error=str(answer["error"]), empty=True)
    places = answer.get("locations") or []
    if not places:
        return QueryResult(
            text=f"No page in the library prints {answer['query']!r}.",
            empty=True)
    rows, reports, lines = [], [], []
    for place in places:
        pages = [place["page"]] if place.get("page") is not None else []
        rows += _rows_of(place["report"], pages)
        reports.append(place["report"])
        lines.append(f"{_cite(place['report'], pages)} "
                     f"{place.get('section', '')}/{place.get('subject', '')}"
                     f" -- {place.get('snippet', '')}")
    return QueryResult(text=f"{answer['query']!r} is printed here:\n"
                            + "\n".join(lines), rows=rows, reports=reports)


def _q_facts(library: Any, args: Dict[str, Any],
             max_rows: int) -> QueryResult:
    fields = args.get("fields")
    answer = library.facts(args.get("report_id", ""),
                           fields if isinstance(fields, (list, tuple))
                           else None)
    if answer.get("error"):
        known = answer.get("reports_in_the_library") or []
        return QueryResult(
            text=str(answer["error"]) + (
                "\nThe library holds: " + ", ".join(known) if known else ""),
            error=str(answer["error"]), empty=True)
    rid = answer["report"]
    rows, lines = [], [f"{rid} -- {answer.get('title', '')}"]
    for row in answer["fields"][:max_rows]:
        rows += _rows_of(rid, row.get("pages") or [])
        quote = f' "{row["quote"]}"' if row.get("quote") else ""
        lines.append(f"{row['field']}: {row['value']} "
                     f"{_cite(rid, row.get('pages') or [])}{quote}")
    if not answer["fields"]:
        lines.append("This report answered none of the fields asked for.")
    if answer.get("not_answered"):
        lines.append("Not answered by this report: "
                     + ", ".join(answer["not_answered"][:max_rows]))
    if answer.get("not_a_field"):
        lines.append("Not a narrative field: "
                     + ", ".join(answer["not_a_field"]))
    counts = answer.get("counts") or {}
    lines.append(f"It holds {counts.get('investigations', 0)} exploration(s), "
                 f"{counts.get('lab_tests', 0)} laboratory test(s), "
                 f"{counts.get('calculations', 0)} calculation(s).")
    qa = answer.get("qa") or {}
    lines.append(f"QA: {qa.get('total', 0)} entr(ies), "
                 f"{qa.get('needs_review', 0)} wanting a person.")
    return QueryResult(text="\n".join(lines), rows=rows or [(rid, None)],
                       reports=[rid], empty=not answer["fields"])


def _q_compare(library: Any, args: Dict[str, Any],
               max_rows: int) -> QueryResult:
    ids = args.get("report_ids")
    answer = library.compare(str(args.get("field", "")),
                             ids if isinstance(ids, (list, tuple)) else None)
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]),
                           error=str(answer["error"]), empty=True)
    rows, reports, lines = [], [], [f"{answer['field']} across the library:"]
    for row in answer["rows"][:max_rows]:
        rows += _rows_of(row["report"], row.get("pages") or [])
        reports.append(row["report"])
        lines.append(f"{row['report']}: {row['value']} "
                     f"{_cite(row['report'], row.get('pages') or [])}")
    if answer.get("missing"):
        lines.append("Did not answer it: " + ", ".join(
            answer["missing"][:max_rows]))
    if answer.get("not_in_the_library"):
        lines.append("Not in the library: "
                     + ", ".join(answer["not_in_the_library"]))
    return QueryResult(text="\n".join(lines), rows=rows, reports=reports,
                       empty=not answer["rows"])


def _q_explorations(library: Any, args: Dict[str, Any],
                    max_rows: int) -> QueryResult:
    answer = library.explorations(args.get("report_id", ""),
                                  str(args.get("kind") or ""))
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]),
                           error=str(answer["error"]), empty=True)
    rid = answer["report"]
    if not answer["explorations"]:
        return QueryResult(
            text=f"{rid} holds no exploration of kind "
                 f"{answer['kind']!r}.", reports=[rid], empty=True)
    rows, lines = [], [f"{rid}: {answer['n']} exploration(s)"]
    for inv in answer["explorations"][:max_rows]:
        rows += _rows_of(rid, inv.get("pages") or [])
        spt = ", ".join(f"N {drive['n']} at {drive['depth']}"
                        for drive in inv.get("spt", []) if drive.get("n"))
        water = ", ".join(f"{level['depth']} ({level['when']})"
                          for level in inv.get("water", []))
        parts = [f"{inv['exploration']} ({inv['kind']})",
                 f"total depth {inv['total_depth']}" if inv["total_depth"]
                 else "",
                 f"ground elevation {inv['ground_elevation']}"
                 if inv["ground_elevation"] else "",
                 f"{inv['n_layers']} layer(s)", f"{inv['n_samples']} sample(s)",
                 spt, f"water at {water}" if water else "",
                 f"{inv['cpt_points']} cone reading(s)"
                 if inv.get("cpt_points") else "",
                 f"{inv['dcp_points']} probe increment(s)"
                 if inv.get("dcp_points") else ""]
        lines.append("; ".join(x for x in parts if x) + " "
                     + _cite(rid, inv.get("pages") or []))
        for layer in inv.get("layers", [])[:8]:
            lines.append(f"    {layer['top']} to {layer['bottom']}: "
                         f"{layer['description']} {layer['uscs']}".rstrip())
    return QueryResult(text="\n".join(lines), rows=rows, reports=[rid])


def _q_lab_summary(library: Any, args: Dict[str, Any],
                   max_rows: int) -> QueryResult:
    answer = library.lab_summary(args.get("report_id", ""),
                                 str(args.get("kind") or ""))
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]),
                           error=str(answer["error"]), empty=True)
    rid = answer["report"]
    if not answer["tests"]:
        return QueryResult(
            text=f"{rid} holds no laboratory test of kind "
                 f"{answer['kind']!r}.", reports=[rid], empty=True)
    by_kind = ", ".join(f"{kind} ({count})" for kind, count
                        in (answer.get("by_kind") or {}).items())
    rows, lines = [], [f"{rid}: {answer['n']} laboratory test(s); {by_kind}"]
    for test in answer["tests"][:max_rows]:
        rows += _rows_of(rid, test.get("pages") or [])
        values = "; ".join(f"{key} {value}" for key, value
                           in (test.get("values") or {}).items())
        parts = [test["test"],
                 f"on {test['exploration']} {test['sample']}".strip()
                 if test["exploration"] or test["sample"] else "",
                 f"at {test['depth']}" if test["depth"] else "",
                 test["standard"], values]
        lines.append(", ".join(x for x in parts if x) + " "
                     + _cite(rid, test.get("pages") or []))
    return QueryResult(text="\n".join(lines), rows=rows, reports=[rid])


def _q_calculations(library: Any, args: Dict[str, Any],
                    max_rows: int) -> QueryResult:
    answer = library.calculations(args.get("report_id", ""))
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]),
                           error=str(answer["error"]), empty=True)
    rid = answer["report"]
    if not answer["calculations"]:
        return QueryResult(text=f"{rid} holds no calculation printout.",
                           reports=[rid], empty=True)
    rows, lines = [], [f"{rid}: {answer['n']} calculation(s)"]
    for calc in answer["calculations"][:max_rows]:
        rows += _rows_of(rid, calc.get("pages") or [])
        parts = [calc["works_out"].replace("_", " "),
                 f"in {calc['program']}" if calc["program"] else "",
                 calc["method"], f"for {calc['for']}" if calc["for"] else "",
                 calc["summary"],
                 "results: " + "; ".join(calc["results"])
                 if calc["results"] else ""]
        lines.append(" -- ".join(x for x in parts if x) + " "
                     + _cite(rid, calc.get("pages") or []))
    return QueryResult(text="\n".join(lines), rows=rows, reports=[rid])


def _q_disagreements(library: Any, args: Dict[str, Any],
                     max_rows: int) -> QueryResult:
    answer = library.disagreements(str(args.get("report_id") or ""))
    if answer.get("error"):
        return QueryResult(text=str(answer["error"]),
                           error=str(answer["error"]), empty=True)
    if not answer["entries"]:
        return QueryResult(
            text="Nothing in the library is flagged for a person to look at.",
            empty=True)
    rows, reports, lines = [], [], [
        f"{answer['n']} entr(ies) a person should look at:"]
    for entry in answer["entries"][:max_rows]:
        rows += _rows_of(entry["report"], entry.get("pages") or [])
        reports.append(entry["report"])
        values = (" (" + " vs ".join(entry["values"]) + ")"
                  if entry.get("values") else "")
        lines.append(f"{entry['kind']} at {entry['where']}: "
                     f"{entry['detail']}{values} "
                     + _cite(entry["report"], entry.get("pages") or []))
    return QueryResult(text="\n".join(lines), rows=rows, reports=reports)


_HANDLERS: Dict[str, Callable[[Any, Dict[str, Any], int], QueryResult]] = {
    "library_stats": _q_library_stats,
    "list_reports": _q_list_reports,
    "find": _q_find,
    "where_is": _q_where_is,
    "facts": _q_facts,
    "compare": _q_compare,
    "explorations": _q_explorations,
    "lab_summary": _q_lab_summary,
    "calculations": _q_calculations,
    "disagreements": _q_disagreements,
}


# ---------------------------------------------------------------------------
# what the primary agent gets back
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """One report and page behind a fact in the answer."""

    report: str = Field(description="the report id")
    page: Optional[int] = Field(
        default=None,
        description="the 0-based PDF page, or null where the record carried "
                    "no page for that value")


class LibraryAnswer(BaseModel):
    """The compact structured result the primary agent sees."""

    answer: str = Field(
        default="",
        description="the answer, in prose, every fact cited as "
                    "(report id, page)")
    citations: List[Citation] = Field(
        default_factory=list,
        description="the report and page behind each fact of the answer, "
                    "CHECKED against what the queries actually returned")
    reports_consulted: List[str] = Field(
        default_factory=list,
        description="every report the queries touched, whether or not the "
                    "answer cites it")
    gaps: List[str] = Field(
        default_factory=list,
        description="what the library could not answer: a query that came "
                    "back empty, a citation nothing returned, and anything "
                    "the answer itself flagged")
    queries: int = Field(
        default=0, description="how many library queries this answer cost")
    error: str = Field(default="")


#: ``(L02, p23)``, ``(L02, page 23)``, ``L02, p23`` and ``(L02)``.
_CITE = re.compile(
    r"\(?\b([A-Za-z][A-Za-z0-9_.\-]{0,40})\b\s*(?:,\s*(?:p|pp|page)\s*\.?\s*"
    r"(\d{1,4}))?\s*\)")

#: What a report id LOOKS like, for a bracketed token the library does not
#: have: letters then digits, optionally with a suffix (``L02``, ``L02``,
#: ``L02-bound1``). It is how an INVENTED id is told from an ordinary word in
#: brackets -- "(probably)" is not a citation claim and "(L99)" is, even
#: though neither is in the library. ``p4`` and ``page4`` are excluded: they
#: are the page half of a citation, not the report half.
_ID_SHAPE = re.compile(
    r"^(?![Pp]{1,2}\d+$)(?![Pp]ages?\d+$)"
    r"[A-Za-z]{1,6}[-_]?\d{1,4}(?:[-_][A-Za-z0-9]{1,12})?$")


def _parse_citations(text: str, known: Sequence[str]
                     ) -> List[Tuple[str, Optional[int]]]:
    """The ``(report, page)`` pairs the answer text claims.

    A bracketed token is read as a citation when the library HAS that report
    or when it has the SHAPE of a report id -- so an invented id is caught
    and reported rather than passing as prose, while a parenthesis around an
    ordinary word is left alone. The match is case-insensitive because a
    model writes ``(l02, p23)`` as often as ``(L02, p23)``.
    """
    lookup = {name.lower(): name for name in known}
    out: List[Tuple[str, Optional[int]]] = []
    for match in _CITE.finditer(str(text or "")):
        token = match.group(1)
        name = lookup.get(token.lower())
        if name is None:
            if not _ID_SHAPE.match(token):
                continue
            name = token
        page = int(match.group(2)) if match.group(2) else None
        if (name, page) not in out:
            out.append((name, page))
    return out


def _gap_lines(text: str) -> List[str]:
    return [line.strip()[4:].strip()
            for line in str(text or "").splitlines()
            if line.strip().lower().startswith("gap:")]


def build_answer(text: str, results: Sequence[Tuple[str, QueryResult]],
                 known_reports: Sequence[str]) -> LibraryAnswer:
    """The structured answer, with every citation checked against the rows.

    A citation the queries did not return is NOT quietly dropped and NOT
    quietly kept: it is left out of ``citations`` and named in ``gaps``, so
    a reader can see that the answer claimed a page nothing produced.
    """
    returned: List[Tuple[str, Optional[int]]] = []
    consulted: List[str] = []
    gaps: List[str] = []
    for name, result in results:
        for row in result.rows:
            if row not in returned:
                returned.append(row)
        for report in result.reports:
            if report not in consulted:
                consulted.append(report)
        if result.error:
            gaps.append(f"the query {name} failed: {result.error}")
        elif result.empty:
            gaps.append(f"the query {name} returned nothing")

    pages_by_report: Dict[str, List[Optional[int]]] = {}
    for report, page in returned:
        pages_by_report.setdefault(report, []).append(page)

    citations: List[Citation] = []
    for report, page in _parse_citations(text, known_reports):
        pages = pages_by_report.get(report)
        if pages is None:
            gaps.append(f"the answer cites {report}, which no query returned")
            continue
        if page is not None and page not in pages:
            gaps.append(f"the answer cites {report} page {page}, which no "
                        f"query returned")
            continue
        citations.append(Citation(report=report, page=page))

    gaps += _gap_lines(text)
    answer = "\n".join(line for line in str(text or "").splitlines()
                       if not line.strip().lower().startswith("gap:")).strip()
    return LibraryAnswer(answer=answer, citations=citations,
                         reports_consulted=consulted,
                         gaps=list(dict.fromkeys(gaps)),
                         queries=len(results))


# ---------------------------------------------------------------------------
# the library, and whether there is one
# ---------------------------------------------------------------------------

def library_available(root: Any) -> bool:
    """True when ``root`` is a folder that holds at least one record.

    The feature detection the app wires on: a deployment with no library
    folder never advertises the sub-agent, so the primary is not told to
    delegate to something that can only answer "there is nothing here".
    """
    path = str(root or "")
    if not path or not os.path.isdir(path):
        return False
    for _folder, _dirs, names in os.walk(path):
        if "report.record.json" in names:
            return True
    return False


def open_library_at(root: Any, db_path: Any = None) -> Any:
    """A :class:`~report_ingest.library.Library` over that folder."""
    from report_ingest.library import Library
    return Library(root, db_path=db_path)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def _langchain_tools(specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The specs in the shape ``bind_tools`` takes."""
    return [{"type": "function", "function": dict(spec)} for spec in specs]


def _tool_calls(message: Any) -> List[Dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    out: List[Dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            out.append({"name": call.get("name", ""),
                        "args": call.get("args") or call.get("arguments") or {},
                        "id": call.get("id", "")})
    return out


def _last_text(messages: Sequence[Any]) -> str:
    for message in reversed(list(messages)):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [block.get("text", "") for block in content
                     if isinstance(block, dict)]
            if any(parts):
                return "\n".join(parts)
    return ""


def build_report_library_graph(model: Any, *, library_root: Any = None,
                               library: Any = None,
                               db_path: Any = None,
                               max_tool_calls: int = MAX_TOOL_CALLS,
                               max_rows: int = MAX_ROWS_PER_CALL,
                               max_result_chars: int = MAX_RESULT_CHARS,
                               extra_system_prompt: Optional[str] = None
                               ) -> Any:
    """A model-with-tools graph that answers ONLY from the library.

    ``model`` is the app's own chat model -- unlike the ingest, which runs
    readers on its own engine, this is a model that reads query results. It
    must support ``bind_tools``; one that does not is answered with an error
    rather than being asked questions it cannot act on.

    ``library`` is a :class:`~report_ingest.library.Library`, for a caller
    that has one; otherwise one is opened over ``library_root`` per run, so
    a report ingested earlier in the same session is picked up by the index
    rebuild rather than being invisible until the next restart.

    The ceilings are enforced HERE, in Python: ``max_tool_calls`` queries per
    answer, ``max_rows`` rows and ``max_result_chars`` characters per query
    result.
    """
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import Annotated, TypedDict

    # The functional TypedDict spelling for the same reason as the ingest's:
    # this module carries ``from __future__ import annotations``, so a class
    # body's annotations would be strings LangGraph resolves against the
    # MODULE's globals, where these names are not.
    State = TypedDict("State", {
        "messages": Annotated[list, add_messages],
        "question": str,
        "calls": int,
        "turns": int,
        "results": list,
        "structured_response": Any,
    }, total=False)

    prompt = LIBRARY_SYSTEM_PROMPT + (
        "\n\n" + extra_system_prompt if extra_system_prompt else "")

    def _library() -> Any:
        if library is not None:
            return library
        return open_library_at(library_root, db_path)

    def ask(state: State) -> Dict[str, Any]:
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage,
        )

        # The system prompt is NOT put into the state: it is prepended to
        # every invocation instead, so the reducer that appends messages
        # cannot end up with it after the question it is meant to precede.
        history = list(state.get("messages") or [])
        seed: List[Any] = []
        if not history:
            question = str(state.get("question") or "").strip()
            if question:
                seed = [HumanMessage(content=question)]
                history = list(seed)
        try:
            bound = model.bind_tools(_langchain_tools(LIBRARY_TOOL_SPECS))
        except (AttributeError, NotImplementedError, TypeError) as exc:
            answer = LibraryAnswer(
                error=f"this deployment's model cannot call tools "
                      f"({type(exc).__name__}), so the library cannot be "
                      f"queried here.")
            return {"messages": seed + [AIMessage(content=answer.error)],
                    "structured_response": answer}
        reply = bound.invoke([SystemMessage(content=prompt)] + history)
        return {"messages": seed + [reply],
                "turns": int(state.get("turns") or 0) + 1}

    def query(state: State) -> Dict[str, Any]:
        from langchain_core.messages import ToolMessage

        messages = list(state.get("messages") or [])
        calls = _tool_calls(messages[-1]) if messages else []
        spent = int(state.get("calls") or 0)
        results = list(state.get("results") or [])
        out: List[Any] = []
        opened = _library()
        for call in calls:
            if spent >= max_tool_calls:
                out.append(ToolMessage(
                    tool_call_id=call["id"], name=call["name"],
                    content=f"the ceiling of {max_tool_calls} librar(y) "
                            f"queries for one answer has been reached; "
                            f"answer from what you already have, and say "
                            f"what is missing."))
                continue
            spent += 1
            result = run_query(opened, call["name"], call["args"],
                               max_rows=max_rows, max_chars=max_result_chars)
            results.append((call["name"], result))
            out.append(ToolMessage(tool_call_id=call["id"],
                                   name=call["name"],
                                   content=result.text or "(nothing)"))
        return {"messages": out, "calls": spent, "results": results}

    def finish(state: State) -> Dict[str, Any]:
        standing = state.get("structured_response")
        if isinstance(standing, LibraryAnswer) and standing.error:
            return {}                    # a refusal already said what it was
        messages = list(state.get("messages") or [])
        results = list(state.get("results") or [])
        opened = _library()
        try:
            known = [row["report"] for row
                     in opened.list_reports(limit=10000)["reports"]]
        except Exception:                        # a library that will not open
            known = []
        known += [report for _name, result in results
                  for report in result.reports if report not in known]
        answer = build_answer(_last_text(messages), results, known)
        if not answer.answer and not answer.error:
            answer.error = ("the library sub-agent produced no answer; the "
                            "question may need to be asked again")
        return {"structured_response": answer}

    # No annotation on ``route``: LangGraph resolves a branch function's type
    # hints, and this module's ``from __future__ import annotations`` would
    # make ``State`` a string it cannot look up in the module's globals.
    def route(state):
        messages = list(state.get("messages") or [])
        if state.get("structured_response") is not None:
            return "finish"
        if int(state.get("turns") or 0) >= MAX_TURNS:
            return "finish"
        if messages and _tool_calls(messages[-1]):
            return "query"
        return "finish"

    graph = StateGraph(State)
    graph.add_node("ask", ask)
    graph.add_node("query", query)
    graph.add_node("finish", finish)
    graph.set_entry_point("ask")
    graph.add_conditional_edges("ask", route,
                                {"query": "query", "finish": "finish"})
    graph.add_edge("query", "ask")
    graph.add_edge("finish", END)
    return graph.compile()


def build_report_library_subagent(model: Any, *, library_root: Any = None,
                                  library: Any = None,
                                  db_path: Any = None,
                                  max_tool_calls: int = MAX_TOOL_CALLS,
                                  max_rows: int = MAX_ROWS_PER_CALL,
                                  max_result_chars: int = MAX_RESULT_CHARS,
                                  extra_system_prompt: Optional[str] = None,
                                  middleware: Optional[Sequence[Any]] = None,
                                  max_model_calls: Optional[int] = None
                                  ) -> Dict[str, Any]:
    """The deepagents ``CompiledSubAgent`` spec for the library.

    The ``middleware`` entry is a DECLARATION, exactly as it is on the
    ingest's spec: deepagents uses a compiled sub-agent's runnable as
    provided and reads no middleware for one. The ceilings that hold are
    :data:`MAX_TOOL_CALLS` and the per-result caps, applied in the graph.
    """
    spec: Dict[str, Any] = {
        "name": SUBAGENT_NAME,
        "description": SUBAGENT_DESCRIPTION,
        "runnable": build_report_library_graph(
            model, library_root=library_root, library=library,
            db_path=db_path, max_tool_calls=max_tool_calls,
            max_rows=max_rows, max_result_chars=max_result_chars,
            extra_system_prompt=extra_system_prompt),
    }
    carried = list(middleware or [])
    if max_model_calls:
        from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
        carried.append(ModelCallBudgetMiddleware(max_model_calls))
    if carried:
        spec["middleware"] = carried
    return spec


def answer_question(question: str, *, model: Any, library_root: Any = None,
                    library: Any = None, db_path: Any = None,
                    max_tool_calls: int = MAX_TOOL_CALLS,
                    max_rows: int = MAX_ROWS_PER_CALL,
                    max_result_chars: int = MAX_RESULT_CHARS
                    ) -> LibraryAnswer:
    """One question through the graph, for a caller that is not deepagents."""
    from langchain_core.messages import HumanMessage

    graph = build_report_library_graph(
        model, library_root=library_root, library=library, db_path=db_path,
        max_tool_calls=max_tool_calls, max_rows=max_rows,
        max_result_chars=max_result_chars)
    state = graph.invoke({"messages": [HumanMessage(content=str(question))],
                          "question": str(question)})
    answer = state.get("structured_response")
    return answer if isinstance(answer, LibraryAnswer) else LibraryAnswer(
        error="the library graph returned no structured answer")


def answer_as_json(answer: LibraryAnswer, max_chars: int = 6000) -> str:
    """The structured answer as compact JSON for a primary tool result."""
    blob = json.dumps(answer.model_dump())
    return blob if len(blob) <= max_chars else blob[:max_chars] + "...(cut)"
