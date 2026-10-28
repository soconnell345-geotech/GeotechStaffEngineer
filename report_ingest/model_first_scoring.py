"""Model-first reading of one hand-truthed item, scored as today's readers
are (BACKLOG RI-87, piece 9; ``module_work/REPORT_INGEST_PLAN.md`` section
8.8, wave MF-1).

``score_on_cluster``'s reader stages call these in place of
``score_one_log``, ``score_one_sheet``, ``score_one_calc`` and
``score_one_report`` when model first is on, and ``run_plain_baseline
--model-first`` calls them for its local arm. Nothing on today's path
imports this module.

ONE ITEM, ITS OWN PAGES. A log, a laboratory sheet or a calculation is read
on the pages its answer sheet names, as the stage picks them, as ONE group
(as RI-86 handed the plain call its pages): the model-first reader of its
stage (:mod:`report_ingest.model_first_read`) makes one call (two halves
where the reply does not fit, two windows for a long printout), builds the
record with the stage's builder at ``checks="flag"``, and places each value
by the words printed on its page (:mod:`report_ingest.locate`, as the graph
places them). The narrative is read on the pages the graph would give it,
with the rules' page labels in place of the range labels (this wave has no
describer): every page labelled cover, letter, contents, narrative, figure,
plan or profile, plus the front of the report, in parts cut to the route's
page limit and the narrative's text budget.

THE SAME SCORERS. The record is scored by the stage's own scorer and
tolerances (``log_scoring.score_record``, ``lab_scoring.score_record``,
``calc_scoring.score_record``, ``narrative_scoring.score_narrative``), into
the same score objects as today's arm, so the two arms compare value for
value and ``rescore_saved`` reads both. With no floor and no merge,
``after`` is the model's record: ``model_alone`` is the same score and
``record`` and ``model_record`` the same record; there is no floor record.
``before`` is still the code's own reading of the pages -- the log grid,
the page's tables, the calculation floor -- made with no model call, so the
before column is the same in both arms. (The log grid is read with no
engine, so with ``GEOTECH_INGEST_VISUAL_SCALES`` on no depth label of a
textless scan is read for it; today's arm reads them.)

WHAT THE ARM SAYS. Each function also returns the arm's own facts for the
run file: ``model_first: "on"``, the page route, the instruction version
(:data:`report_ingest.model_first.READ_PROMPT_VERSION`), each call's pages
and how it ended, and the placement counts (:func:`placement_counts`).

An error a call raises (a refused request, the call ceiling), and a group
no answer came back for, is the score's ``error``, as a failed model call
is on today's arm: the cluster stage then reads the item again next time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["item_group", "narrative_groups_for", "placement_counts",
           "first_log", "first_sheet", "first_calc", "first_narrative"]


def item_group(stage: str, pages: Sequence[int], key: str) -> Any:
    """One item's own pages as one group of its stage."""
    from report_ingest.model_first_pages import PageGroup

    return PageGroup(stage, sorted({int(p) for p in pages}),
                     why="the item's own pages, as its answer sheet names "
                         "them", key=str(key))


def placement_counts(placements: Sequence[Dict[str, Any]]
                     ) -> Dict[str, Any]:
    """What placement did with an item's values, summed over its answers:
    ``values``; ``with_text`` (the values whose page has words to match);
    ``at_a_spot`` (placed or moved, less those placed by the page only),
    which is what the plan's 90 % gate counts (section 8.4);
    ``page_level``; and every status's count."""
    from report_ingest.locate import STATUSES

    counts = {status: 0 for status in STATUSES}
    page_level = 0
    for found in placements or ():
        if not isinstance(found, dict):
            continue
        for status, n in (found.get("counts") or {}).items():
            counts[status] = counts.get(status, 0) + int(n or 0)
        page_level += sum(1 for item in found.get("items") or ()
                          if isinstance(item, dict) and item.get("page_level"))
    total = sum(counts.values())
    return {"values": total, "with_text": total - counts["no_text"],
            "at_a_spot": counts["placed"] + counts["moved"] - page_level,
            "page_level": page_level, "counts": counts}


def _arm(blobs: Sequence[Dict[str, Any]], route: str) -> Dict[str, Any]:
    """The arm's facts for one item's run file."""
    from report_ingest.model_first import READ_PROMPT_VERSION

    return {"model_first": "on", "route": route,
            "prompt_version": READ_PROMPT_VERSION,
            "calls": [dict(c) for b in blobs for c in b.get("calls") or []],
            "placement": placement_counts(
                [p for b in blobs for p in b.get("placement") or []])}


def _route(engine: Any, page_route: Optional[str]) -> str:
    """``pdf`` or ``pictures`` for this engine (``page_route`` None: the
    variable); a ``pdf`` route the engine cannot send is refused."""
    from report_ingest.model_first_pages import resolve_route

    return resolve_route(page_route,
                         bool(getattr(engine, "accepts_pdf", False)))


def _placer(doc: Any) -> Any:
    """The graph's own placement, over one reading of the pages' words."""
    from report_ingest.graph import _placer as placer

    return placer(doc)


# ---------------------------------------------------------------------------
# the four stages
# ---------------------------------------------------------------------------

def first_log(truth: Dict[str, Any], doc: Any, engine: Any, *,
              report_id: str = "", page_route: Optional[str] = None
              ) -> Tuple[Any, Any, Dict[str, Any]]:
    """``(before, after, arm)`` for one log: the grid (no call), then the
    model-first reader over the log's pages."""
    from report_ingest.log_scoring import (
        LogScore, _scored, dump_investigations, score_grid, score_record,
    )
    from report_ingest.model import Investigation
    from report_ingest.model_first_read import read_logs
    from report_ingest.visual_scales import log_grid_for

    log_id = str(truth["id"])
    pages = [int(p) for p in truth.get("pages") or []]
    report = report_id or log_id.split("_")[0]
    route = _route(engine, page_route)
    arm: Dict[str, Any] = {"model_first": "on", "route": route}
    try:
        grid, _info = log_grid_for(doc, pages)
    except Exception as exc:                        # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return (LogScore(log_id=log_id, report=report, stage="grid",
                         error=error),
                LogScore(log_id=log_id, report=report, stage="record",
                         error=error), arm)
    before = score_grid(truth, grid)
    try:
        (blob,) = read_logs(doc, [item_group("logs", pages, log_id)], engine,
                            report_id=report, route=route,
                            place=_placer(doc))
    except Exception as exc:                        # a model call that failed
        return before, LogScore(log_id=log_id, report=report, stage="record",
                                error=f"{type(exc).__name__}: {exc}"), arm
    arm = _arm([blob], route)
    if blob.get("error") or not blob.get("investigation"):
        after = LogScore(log_id=log_id, report=report, stage="record",
                         error=blob.get("error") or "no log was read")
        after.cost = dict(blob.get("cost") or {})
        after.model_calls = int(blob.get("model_calls") or 0)
        return before, after, arm
    investigations = [Investigation.model_validate(blob["investigation"])]
    after = score_record(truth, investigations)
    after.record = dump_investigations(investigations)
    after.model_record = after.record
    after.model_alone = _scored(after)
    after.cost = dict(blob.get("cost") or {})
    after.model_calls = int(blob.get("model_calls") or 0)
    after.unresolved = len(blob.get("unresolved") or ())
    after.changes = len(blob.get("changes") or ())
    after.warnings = list(blob.get("warnings") or ())
    return before, after, arm


def first_sheet(truth: Dict[str, Any], doc: Any, engine: Any, *,
                report_id: str = "", page_route: Optional[str] = None
                ) -> Tuple[Any, Any, Dict[str, Any]]:
    """``(before, after, arm)`` for one laboratory sheet: the page's tables
    (no call), then the model-first reader over the sheet's pages."""
    from report_ingest.lab_scoring import (
        _as_blob, _blank, _dump_tests, pages_of, report_of, score_record,
        score_tables, score_unlinked,
    )
    from report_ingest.model import LabTest
    from report_ingest.model_first_read import read_lab

    sheet_id = str(truth.get("id") or "")
    kind = str(truth.get("kind") or "")
    pages = pages_of(truth)
    report = report_id or report_of(truth)
    route = _route(engine, page_route)
    arm: Dict[str, Any] = {"model_first": "on", "route": route}
    if not pages:
        error = "the truth file's id does not name a page"
        return (_blank(sheet_id, kind, "tables", error),
                _blank(sheet_id, kind, "record", error), arm)
    try:
        before = score_tables(truth, doc, pages)
    except Exception as exc:                        # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return (_blank(sheet_id, kind, "tables", error),
                _blank(sheet_id, kind, "record", error), arm)
    try:
        (blob,) = read_lab(doc, [item_group("lab", pages, sheet_id)], engine,
                           report_id=report, route=route, place=_placer(doc))
    except Exception as exc:                        # a model call that failed
        return before, _blank(sheet_id, kind, "record",
                              f"{type(exc).__name__}: {exc}"), arm
    arm = _arm([blob], route)
    if blob.get("error"):
        after = _blank(sheet_id, kind, "record", str(blob["error"]))
        after.cost = dict(blob.get("cost") or {})
        after.model_calls = int(blob.get("model_calls") or 0)
        return before, after, arm
    tests = [LabTest.model_validate(t) for t in blob.get("tests") or []]
    after = score_record(truth, tests)
    after.record = _dump_tests(tests)
    after.model_record = after.record
    after.model_alone = _as_blob(after)
    after.unlinked = _as_blob(score_unlinked(truth, tests))
    after.cost = dict(blob.get("cost") or {})
    after.model_calls = int(blob.get("model_calls") or 0)
    after.unresolved = len(blob.get("unresolved") or ())
    after.changes = len(blob.get("changes") or ())
    after.warnings = list(blob.get("warnings") or ())
    return before, after, arm


def first_calc(truth: Dict[str, Any], doc: Any, engine: Any, *,
               report_id: str = "", page_route: Optional[str] = None
               ) -> Tuple[Any, Any, Dict[str, Any]]:
    """``(before, after, arm)`` for one calculation: the floor (no call),
    then the model-first reader over the printout's pages."""
    from report_ingest.calc_reader import floor_from_pages
    from report_ingest.calc_scoring import (
        _blank, _dump, _scored, pages_of, report_of, score_floor,
        score_record,
    )
    from report_ingest.model import Calculation
    from report_ingest.model_first_read import read_calcs

    calc_id = str(truth.get("id") or "")
    kind = str(truth.get("kind") or "")
    pages = pages_of(truth)
    report = report_id or report_of(truth)
    route = _route(engine, page_route)
    arm: Dict[str, Any] = {"model_first": "on", "route": route}
    if not pages:
        error = "the truth file names no page"
        return (_blank(calc_id, kind, "floor", error),
                _blank(calc_id, kind, "record", error), arm)
    try:
        floor = floor_from_pages(doc, pages, report)
        before = score_floor(truth, floor.as_calculation(report))
        before.n_pages = len(pages)
    except Exception as exc:                        # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return (_blank(calc_id, kind, "floor", error),
                _blank(calc_id, kind, "record", error), arm)
    try:
        (blob,) = read_calcs(doc, [item_group("calc", pages, calc_id)],
                             engine, report_id=report, route=route,
                             place=_placer(doc))
    except Exception as exc:                        # a model call that failed
        return before, _blank(calc_id, kind, "record",
                              f"{type(exc).__name__}: {exc}"), arm
    arm = _arm([blob], route)
    if blob.get("error") or not blob.get("calculation"):
        after = _blank(calc_id, kind, "record",
                       str(blob.get("error") or "no calculation was read"))
        after.cost = dict(blob.get("cost") or {})
        after.model_calls = int(blob.get("model_calls") or 0)
        after.n_pages = len(pages)
        return before, after, arm
    calculation = Calculation.model_validate(blob["calculation"])
    after = score_record(truth, calculation)
    after.record = _dump(calculation)
    after.model_record = after.record
    scored = _scored(after)
    after.model_alone = {"scores": scored["scores"],
                         "overall": scored["overall"]}
    after.cost = dict(blob.get("cost") or {})
    after.model_calls = int(blob.get("model_calls") or 0)
    after.unresolved = len(blob.get("unresolved") or ())
    after.changes = len(blob.get("changes") or ())
    after.n_pages = len(pages)
    after.warnings = list(blob.get("warnings") or ())
    return before, after, arm


def narrative_groups_for(doc: Any, route: str,
                         front_pages: Optional[int] = None
                         ) -> Tuple[List[Any], List[int]]:
    """``(the narrative's groups, the body pages)``: the graph's grouping
    (:func:`report_ingest.model_first_pages.narrative_groups`: on the pdf
    route cut to its page limit and the narrative's text budget, on the
    pictures route one part, RI-175) over the rules' page labels, and the
    pages whose captions are counted, as the graph picks them from its
    labels."""
    from planlens.document.roles import page_roles

    from report_ingest.model_first_pages import (MAX_NARRATIVE_PAGES,
                                                 narrative_groups)
    from report_ingest.narrative_reader import (CHARS_PER_TOKEN,
                                                INPUT_TOKEN_BUDGET)

    labels = {int(r.page): str(r.role) for r in page_roles(doc)}
    page_chars = {int(s.page): int(s.n_text_chars) for s in doc.page_map()}
    groups = narrative_groups(labels, doc.n_pages, front_pages=front_pages,
                              max_pages=MAX_NARRATIVE_PAGES,
                              page_chars=page_chars,
                              max_chars=INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN,
                              route=route)
    body = [page for page, role in sorted(labels.items())
            if role in ("narrative", "figure", "plan", "profile", "toc")]
    return groups, body


def first_narrative(truth: Dict[str, Any], doc: Any, engine: Any, *,
                    report: str = "", page_route: Optional[str] = None,
                    front_pages: Optional[int] = None,
                    max_request_tokens_in: Optional[int] = None
                    ) -> Tuple[Any, Dict[str, Any]]:
    """``(score, arm)`` for one report's narrative: the model-first reader
    over the narrative's pages (:func:`narrative_groups_for`), its request
    kept under ``max_request_tokens_in`` (RI-175)."""
    from planlens.document.roles import document_outline

    from report_ingest.model import GeneralFacts, NaturalHazardFacts
    from report_ingest.model_first_read import read_narrative
    from report_ingest.narrative_scoring import (NarrativeScore,
                                                 score_narrative)

    report = report or str(truth.get("id") or "")
    route = _route(engine, page_route)
    arm: Dict[str, Any] = {"model_first": "on", "route": route}
    try:
        groups, body = narrative_groups_for(doc, route, front_pages)
        blob = read_narrative(doc, groups, engine, report_id=report,
                              route=route, outline=document_outline(doc),
                              body_pages=body or None, place=_placer(doc),
                              max_request_tokens_in=max_request_tokens_in)
    except Exception as exc:                        # score what can be scored
        return NarrativeScore(report=report,
                              error=f"{type(exc).__name__}: {exc}"), arm
    arm = _arm([blob], route)
    arm["left_out"] = [int(p) for p in blob.get("left_out") or []]
    arm["warnings"] = list(blob.get("warnings") or ())
    carried = {"pages": [int(p) for p in blob.get("pages") or []],
               "unresolved": len(blob.get("unresolved") or ()),
               "model_calls": int(blob.get("model_calls") or 0),
               "cost": dict(blob.get("cost") or {})}
    if blob.get("error") or blob.get("general") is None:
        out = NarrativeScore(report=report, error=str(
            blob.get("error") or "no narrative was read"))
    else:
        out = score_narrative(
            truth, GeneralFacts.model_validate(blob["general"]),
            NaturalHazardFacts.model_validate(blob["natural_hazards"]),
            report=report)
    for name, value in carried.items():
        setattr(out, name, value)
    return out, arm
