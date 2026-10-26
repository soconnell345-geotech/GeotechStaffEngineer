"""RI-87 piece 5's offline score: placement by text match, on saved answers.

Section 8.4 of ``module_work/REPORT_INGEST_PLAN.md`` sets a gate: under
model-first ingest at least 90 % of the values on pages with text must get a
box from :mod:`report_ingest.locate`, or that stage gets the floor back as a
source of boxes. This script measures it at $0, on two sets of saved
answers:

* **in sample**: the plain baseline's answers (RI-86: Opus 5.5, each item's
  own pages as one PDF, the stage's answer schema, one call;
  ``run_plain_baseline.py``), the answers the placement rules were written
  against. Each answer's cited pages are read once for the whole answer
  (:func:`report_ingest.locate.answer_pages`, step 1) and the record is
  built by the stage's own builder, as the plain baseline built it;
* **held out**: the pipeline arms' model-alone records (``after.
  model_record`` of ``logs_opus``, ``logs_sonnet``, ``lab_opus_linkon``,
  ``lab_sonnet_linkon``, ``calc_opus``, ``calc_sonnet``), with their boxes
  taken off: other answers, from two models, that the rules were not
  written against. Each placed value is set beside the box it had.

Both are split into the stage's open and blind reports (the truth folder's
``OPEN.txt``, read for the report names only, as ``run_claude_local``
reads it). The rules are tuned on open items only: RESULTS.md lists what
was not placed for open items, and gives blind items as counts.

The report is opened as the pipeline opens it (``Corpus.open_report(...,
di="auto")``: the text layer, and DI words on the pages that need them), and
a log's depth rulers come from ``log_grid_for`` with the visual scales off.
Every layer, sample, drive and water reading of a log, every laboratory
test, every calculation value and every narrative citation is placed (steps
2 to 4) and counted: placed at a spot, placed at page level (a lab sheet's
title, but for the answer's only test on it; a quote found on its page
that search could not box), moved to another page of the call, printed in
several places, text present but not found, no text on the page, no page.
The gate counts the spot placements; it is also given with every lab title
kept apart (a sheet's title is the spot of the answer's only test on that
page, :func:`locate.place_lab_tests`), and the lab titles are counted by
test kind (from the answer's own records, blind included). A lab test whose
boring label is printed on its page only as a heading
(:func:`locate.boring_heading`) is counted in a column of its own: the
heading is never its spot, and the column says where it went instead.
In sample, where the pipeline's Opus arm (``logs_opus``,
``lab_opus_linkon``, ``calc_opus``) holds a box for the same object, the
two boxes are set side by side: same page, same printed line (their
vertical extents overlap), same spot (the boxes intersect).

Run it from the repo root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_placement

WHERE IT WRITES: ``<main checkout>/module_work/field_feedback/
2026-10-24_claude_local/raw/placement/`` (gitignored; an out folder git would
track is refused): ``RESULTS.md`` and ``results.json``, which name items,
pages and what was not placed. What it prints carries counts only.

PRIVACY. The answers, the corpus and the pipeline's records are read in
place from the private ``raw/`` folders. No answer sheet is opened: the
score is against the pages' own words and the pipeline's boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import run_claude_local as rcl
from module_work.report_ingest_harness.run_plain_baseline import (
    OPEN_WINDOW, PIPELINE_ARMS, STAGES, reading_class,
)

#: The plain arms (RI-86, Opus 5.5) whose answers are placed in sample.
PLAIN_ARMS: Dict[str, str] = {s: f"plain_{s}_opus" for s in STAGES}
#: The pipeline arms whose model-alone records are the held-out measure.
HELDOUT_ARMS: Dict[str, Tuple[str, ...]] = {
    "logs": ("logs_opus", "logs_sonnet"),
    "lab": ("lab_opus_linkon", "lab_sonnet_linkon"),
    "calc": ("calc_opus", "calc_sonnet")}
#: Where the results go, under the claude_local raw folder.
OUT_NAME = "placement"
#: Points two boxes' vertical extents may miss each other by and still be
#: called the same printed line.
LINE_SLACK = 2.0
#: The share of a stage's values on pages with text that must be placed at
#: a spot (plan section 8.4).
GATE = 0.9
SETS = ("open", "blind")


# ---------------------------------------------------------------------------
# step 1 on the saved answer, and the record built from it
# ---------------------------------------------------------------------------

def _answer_provs(stage: str, reading: Any) -> List[Any]:
    """Every object's provenance in an answer (a citation is its own)."""
    if stage == "logs":
        return [o.prov for key in ("layers", "samples", "spt", "water")
                for o in getattr(reading, key) or [] if o.prov is not None]
    if stage == "lab":
        return [t.prov for t in reading.tests]
    if stage == "calc":
        return [v.prov for v in list(reading.inputs) + list(reading.results)]
    return list(reading.citations)


def check_answer(stage: str, reading: Any, pages: Sequence[int]
                 ) -> Counter:
    """Step 1 on every cited page of the answer, in place, the answer's
    pages read once for all of them (:func:`report_ingest.locate.
    answer_pages`): how many were a page of the call, read as a position
    in it, or outside it; and ``mixed`` when the answer's cites mix the
    call's pages and others."""
    from report_ingest import locate

    provs = _answer_provs(stage, reading)
    numbers = locate.answer_pages(provs, pages)
    out: Counter = Counter()
    if numbers.mixed:
        out["mixed"] += 1
    for prov in provs:
        page, note, confidence = locate.page_for(prov.page, pages, numbers)
        out["cited" if not note else "outside" if confidence is not None
            else "pdf_local"] += 1
        prov.page = page
    return out


def build(stage: str, reading: Any, pages: Sequence[int], report: str
          ) -> Any:
    """The record, built from the answer alone by the stage's builder with
    its checks flagging, not refusing (``checks="flag"``): model-first
    places behind the builders that way, so every value the answer gives
    is there to place."""
    pages = [int(p) for p in pages]
    if stage == "logs":
        from report_ingest.log_reader import _Builder
        return _Builder(reading, OPEN_WINDOW, pages, report,
                        checks="flag").build()
    if stage == "lab":
        from report_ingest.lab_reader import _Builder
        return _Builder(reading, pages, report, [], checks="flag").build()
    if stage == "calc":
        from report_ingest.calc_reader import _Builder
        return _Builder(reading, pages, report, [], [],
                        checks="flag").build()
    from report_ingest.narrative_reader import _Builder
    builder = _Builder(pages, checks="flag")
    builder.add(reading)
    builder.build(pages, {}, None)
    return builder.citations


def _rulers(doc: Any, pages: Sequence[int]) -> Dict[int, Any]:
    from report_ingest.visual_scales import log_grid_for
    grid, _ = log_grid_for(doc, pages)
    return dict(getattr(grid, "rulers", None) or {})


def place_record(stage: str, record: Any, pages: Sequence[int], doc: Any,
                 words: Any) -> Tuple[Any, Dict[str, Any]]:
    """``(placements, extra)``: the record placed by :mod:`locate`. A log's
    record is one investigation or a list of them (one answer, its pages
    read once for all)."""
    from report_ingest import locate

    if stage == "logs":
        rulers = _rulers(doc, pages)
        invs = record if isinstance(record, list) else [record]
        numbers = locate.answer_pages(
            [o.prov for o in _objects("logs", invs)], pages)
        got = locate.Placements()
        for inv in invs:
            got.items += locate.place_investigation(
                words, inv, pages, rulers, numbers=numbers).items
        return got, {"ruler_pages": sorted(rulers)}
    if stage == "lab":
        return locate.place_lab_tests(words, record, pages), {}
    if stage == "calc":
        return locate.place_calculation(words, record, pages), {}
    return locate.place_citations(words, record, pages), {}


# ---------------------------------------------------------------------------
# the pipeline's boxes for the same objects
# ---------------------------------------------------------------------------

def _q(node: Any) -> Optional[float]:
    """A number out of a saved quantity, a record's ``Quantity`` or a bare
    number."""
    if isinstance(node, dict):
        node = node.get("value")
    elif node is not None and hasattr(node, "value"):
        node = node.value
    return float(node) if isinstance(node, (int, float)) else None


def _prov_of(node: Dict[str, Any]) -> Dict[str, Any]:
    prov = node.get("prov")
    if isinstance(prov, list):
        prov = prov[0] if prov else {}
    return prov or {}


def _boxed(node: Dict[str, Any]) -> bool:
    bbox = _prov_of(node).get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4


def pipeline_objects(stage: str, record: Any) -> List[Dict[str, Any]]:
    """The pipeline record's objects of the kinds placement counts, each
    ``{"kind", "key", "page", "bbox", ...}``."""
    out: List[Dict[str, Any]] = []
    if stage == "logs":
        for inv in record or []:
            for kind, key, top in (("layer", "layers", "top"),
                                   ("sample", "samples", "top"),
                                   ("drive", "spt", "depth_top"),
                                   ("water", "water", "depth")):
                for o in inv.get(key) or []:
                    prov = _prov_of(o)
                    out.append({"kind": kind, "top": _q(o.get(top)),
                                "label": str(o.get("sample_id") or ""),
                                "page": prov.get("page"),
                                "bbox": prov.get("bbox") if _boxed(o)
                                else None})
    elif stage == "lab":
        for t in record or []:
            prov = _prov_of(t)
            out.append({"kind": str(t.get("kind")),
                        "label": str(t.get("sample_id") or ""),
                        "boring": str(t.get("investigation_id") or ""),
                        "top": _q(t.get("depth_top")),
                        "page": prov.get("page"),
                        "bbox": prov.get("bbox") if _boxed(t) else None})
    elif stage == "calc" and isinstance(record, dict):
        for role in ("inputs", "results"):
            for v in record.get(role) or []:
                prov = _prov_of(v)
                out.append({"kind": role[:-1], "name": str(v.get("name")),
                            "top": _q(v.get("value")),
                            "page": prov.get("page"),
                            "bbox": prov.get("bbox") if _boxed(v) else None})
    return out


def _fold(text: Any) -> str:
    from report_ingest.locate import _fold_token
    return _fold_token(text)


def _close(a: Optional[float], b: Optional[float]) -> bool:
    from report_ingest.locate import same_number
    if a is None or b is None:
        return False
    return same_number(a, b) or same_number(b, a)


def pair(stage: str, kind: str, mine: Any, theirs: List[Dict[str, Any]]
         ) -> Optional[Dict[str, Any]]:
    """The pipeline object that is the same object as ``mine``, or None."""
    from report_ingest.locate import label_score

    if stage == "lab":
        kind = str(mine.kind)
    pool = [o for o in theirs if o["kind"] == kind or (
        stage == "calc" and kind in ("input", "result"))]
    if stage == "logs":
        top = _q(getattr(mine, "top", None) or getattr(mine, "depth_top",
                                                        None)
                 or getattr(mine, "depth", None))
        label = _fold(getattr(mine, "sample_id", ""))
        same = [o for o in pool if _close(top, o["top"])]
        if label and len(same) > 1:
            same = [o for o in same if _fold(o["label"]) == label] or same
        return same[0] if len(same) == 1 else None
    if stage == "lab":
        label, boring = _fold(mine.sample_id), _fold(mine.investigation_id)
        top = _q(mine.depth_top)
        same = [o for o in pool if _fold(o["label"]) == label
                and _fold(o["boring"]) == boring]
        if len(same) > 1 and top is not None:
            same = [o for o in same if _close(top, o["top"])] or same
        if not same and len(pool) == 1:
            same = pool
        return same[0] if len(same) == 1 else None
    if stage == "calc":
        value = _q(getattr(mine, "value", None))
        if value is None:
            return None
        same = [o for o in pool if _close(value, o["top"])]
        if len(same) > 1:
            scored = sorted(((label_score(mine.name, o["name"]), i, o)
                             for i, o in enumerate(same)), reverse=True)
            if scored[0][0] >= 70 and (len(scored) == 1
                                       or scored[0][0] > scored[1][0]):
                return scored[0][2]
            return None
        return same[0] if same else None
    return None


def compare(box: Sequence[float], page: int, other: Dict[str, Any]
            ) -> Dict[str, Any]:
    """Two boxes for one object: same page, same printed line, same spot,
    and how far apart their middles sit (points)."""
    theirs = other.get("bbox")
    out = {"same_page": other.get("page") == page}
    if not out["same_page"] or not theirs:
        return out
    a0, a1 = box[1], box[3]
    b0, b1 = theirs[1], theirs[3]
    out["same_line"] = min(a1, b1) - max(a0, b0) >= -LINE_SLACK
    out["same_spot"] = out["same_line"] and (
        min(box[2], theirs[2]) - max(box[0], theirs[0]) >= -LINE_SLACK)
    out["dy"] = round(abs((a0 + a1) / 2 - (b0 + b1) / 2), 1)
    return out


def printed_under(text: Any, box: Sequence[float], limit: int = 70) -> str:
    """The words a box covers, for reading a disagreement (private
    RESULTS only)."""
    got = [w.text for w in text.words
           if w.bbox[0] < box[2] and w.bbox[2] > box[0]
           and w.bbox[1] < box[3] and w.bbox[3] > box[1]]
    return " ".join(got)[:limit]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _objects(stage: str, record: Any) -> List[Any]:
    """The record's objects in the order placement visits them."""
    if stage == "logs":
        invs = record if isinstance(record, list) else [record]
        return [o for inv in invs
                for key in ("layers", "samples", "spt", "water")
                for o in getattr(inv, key) or []]
    if stage == "lab":
        return list(record)
    if stage == "calc":
        return list(record.inputs) + list(record.results)
    return [c for cites in record.values() for c in cites]


def _first_prov(obj: Any) -> Any:
    prov = getattr(obj, "prov", None)
    if isinstance(prov, list):
        return prov[0] if prov else None
    return prov


def model_record(stage: str, blob: Dict[str, Any]) -> Any:
    """A pipeline arm's model-alone record (``after.model_record``), as
    record objects; None when the item has none."""
    from report_ingest.model import Calculation, Investigation, LabTest

    rec = (blob.get("after") or {}).get("model_record")
    if rec is None:
        return None
    if stage == "logs":
        return [Investigation.model_validate(r) for r in rec]
    if stage == "lab":
        return [LabTest.model_validate(r) for r in rec]
    return Calculation.model_validate(rec)


def strip_boxes(stage: str, record: Any
                ) -> List[Tuple[Optional[int], Optional[List[float]]]]:
    """Take every object's box off, in place: ``(page, box)`` it had, in
    placement order."""
    out: List[Tuple[Optional[int], Optional[List[float]]]] = []
    for obj in _objects(stage, record):
        prov = _first_prov(obj)
        box = list(prov.bbox) if prov is not None and prov.bbox else None
        out.append((getattr(prov, "page", None), box))
        if prov is not None:
            prov.bbox = None
    return out


def _open_doc(report: str, corpus: Any, docs: Dict[str, Any]
              ) -> Tuple[Any, Any]:
    from report_ingest.locate import PageWords

    if report not in docs:
        doc = corpus.open_report(report, di="auto", warn=False)
        docs[report] = (doc, PageWords(doc))
    return docs[report]


def _row(got: Any, words: Any, obj: Any = None) -> Dict[str, Any]:
    row = got.to_dict()
    row["page_words"] = (len(words(got.cited).words)
                         if got.cited is not None else 0)
    if got.kind == "test":
        from report_ingest import locate

        row["test_kind"] = str(getattr(obj, "kind", "") or "")
        row["boring_heading"] = (got.page is not None
                                 and got.anchor in ("test title", "")
                                 and locate.boring_heading(words, got.page,
                                                           obj))
    return row


def _beside(got: Any, other: Dict[str, Any], words: Any) -> Dict[str, Any]:
    out = compare(got.bbox, got.page, other)
    out["their_bbox"] = other.get("bbox")
    out["their_page"] = other.get("page")
    if out["same_page"] and not out.get("same_line"):
        out["mine_reads"] = printed_under(words(got.page), got.bbox)
        out["theirs_reads"] = printed_under(words(got.page), other["bbox"])
    return out


def measure_item(stage: str, blob: Dict[str, Any], corpus: Any,
                 docs: Dict[str, Any], pipeline: Optional[Dict[str, Any]]
                 ) -> Dict[str, Any]:
    """One plain answer placed, beside the pipeline arm's boxes."""
    report, pages = blob["report"], [int(p) for p in blob["pages"]]
    reading = reading_class(stage).model_validate(blob["reading"])
    step1 = check_answer(stage, reading, pages)
    record = build(stage, reading, pages, report)
    doc, words = _open_doc(report, corpus, docs)
    placements, extra = place_record(stage, record, pages, doc, words)
    theirs = pipeline_objects(stage, ((pipeline or {}).get("after") or {})
                              .get("record"))
    rows: List[Dict[str, Any]] = []
    for obj, got in zip(_objects(stage, record), placements.items):
        row = _row(got, words, obj)
        if got.boxed and theirs:
            other = pair(stage, got.kind, obj, theirs)
            if other is not None and other.get("bbox"):
                row["pipeline"] = _beside(got, other, words)
        rows.append(row)
    return {"item": blob["item"], "report": report, "pages": pages,
            "step1": dict(step1), "note": placements.qa_note(),
            "counts": placements.counts(), "objects": rows,
            "pipeline_objects": len(theirs),
            "pipeline_boxed": sum(1 for o in theirs if o.get("bbox")),
            "page_sources": {str(p): words(p).source or "none"
                             for p in pages},
            **extra}


def measure_heldout_item(stage: str, blob: Dict[str, Any], corpus: Any,
                         docs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One pipeline arm's model-alone record, its boxes taken off, placed;
    each placed value beside the box it had. None when the item has no
    model-alone record."""
    record = model_record(stage, blob)
    if record is None:
        return None
    report, pages = blob["report"], [int(p) for p in blob["pages"]]
    had = strip_boxes(stage, record)
    doc, words = _open_doc(report, corpus, docs)
    placements, extra = place_record(stage, record, pages, doc, words)
    rows: List[Dict[str, Any]] = []
    for obj, (page, box), got in zip(_objects(stage, record), had,
                                     placements.items):
        row = _row(got, words, obj)
        if got.boxed and box:
            row["pipeline"] = _beside(got, {"page": page, "bbox": box},
                                      words)
        rows.append(row)
    item = blob.get("log_id") or blob.get("sheet_id") or blob.get("calc_id")
    return {"item": str(item), "report": report, "pages": pages,
            "note": placements.qa_note(), "counts": placements.counts(),
            "objects": rows, "pipeline_objects": len(had),
            "pipeline_boxed": sum(1 for _, b in had if b), **extra}


def _set_of(report: str, openset: Sequence[str]) -> str:
    return "open" if report in openset else "blind"


def measure(stages: Sequence[str], paths: Dict[str, Path]
            ) -> Tuple[Dict[str, List[Dict[str, Any]]],
                       Dict[str, Tuple[str, List[Dict[str, Any]]]]]:
    """``(in_sample, held_out)``: the plain answers' items by stage, and
    the pipeline arms' items by arm (with their stage)."""
    from report_ingest.corpus import Corpus

    corpus = Corpus(paths["reports"], di_dir=paths["di"])
    in_sample: Dict[str, List[Dict[str, Any]]] = {}
    held_out: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    docs: Dict[str, Any] = {}
    try:
        for stage in stages:
            openset = rcl.open_reports(stage, paths["truth"] / stage)
            folder = paths["out_root"] / PLAIN_ARMS[stage] / stage
            arm = PIPELINE_ARMS.get(stage) if stage != "narrative" else None
            items: List[Dict[str, Any]] = []
            for path in sorted(folder.glob("*.json")):
                blob = rcl._load(path) or {}
                if not blob.get("reading"):
                    items.append({"item": path.stem, "error": "no answer"})
                    continue
                pipeline = (rcl._load(paths["out_root"] / arm / stage
                                      / path.name) if arm else None)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    item = measure_item(stage, blob, corpus, docs, pipeline)
                item["set"] = _set_of(item["report"], openset)
                items.append(item)
            in_sample[stage] = items
            for arm in HELDOUT_ARMS.get(stage, ()):
                items = []
                for path in sorted((paths["out_root"] / arm / stage)
                                   .glob("*.json")):
                    blob = rcl._load(path) or {}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        item = measure_heldout_item(stage, blob, corpus,
                                                    docs)
                    if item is None:
                        continue
                    item["set"] = _set_of(item["report"], openset)
                    items.append(item)
                held_out[arm] = (stage, items)
    finally:
        for doc, _words in docs.values():
            doc.close()
    return in_sample, held_out


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f} %)" if d else f"{n}/0"


def stage_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The gate's numbers for some items, and the comparison's. ``placed``
    counts the values placed at a spot (on the cited page or moved);
    ``page_level`` those placed by the page only, kept out of the gate;
    ``titles_apart`` the spot placements with every lab title kept apart
    too. The lab titles by test kind: ``page_level_by_kind`` and
    ``sheet_titles_by_kind`` (the title of a page whose only test the
    answer gives is that test, counted as its spot). ``boring_headings``
    the lab tests whose boring label is printed on the page only as a
    heading, by where they went (``boring_headings_by_outcome``: a title
    spot, page level, or the status they were left with)."""
    counts: Counter = Counter()
    anchors: Counter = Counter()
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    by_source: Dict[str, Counter] = defaultdict(Counter)
    step1: Counter = Counter()
    cmp: Counter = Counter()
    page_level = sheet_titles = 0
    page_level_by_kind: Counter = Counter()
    sheet_titles_by_kind: Counter = Counter()
    headings: Counter = Counter()
    for item in items:
        if item.get("error"):
            continue
        step1.update(item.get("step1") or {})
        for row in item["objects"]:
            counts[row["status"]] += 1
            by_kind[row["kind"]][row["status"]] += 1
            by_source[row.get("source") or "none"][row["status"]] += 1
            if row["status"] in ("placed", "moved"):
                anchors[row.get("anchor") or "?"] += 1
                page_level += bool(row.get("page_level"))
                kind = row.get("test_kind") or row["kind"]
                if row.get("page_level"):
                    page_level_by_kind[kind] += 1
                elif row.get("anchor") == "test title":
                    sheet_titles += 1
                    sheet_titles_by_kind[kind] += 1
            if row.get("boring_heading"):
                headings[("page level" if row.get("page_level") else
                          "title spot") if row["status"] in ("placed",
                                                             "moved")
                         else row["status"]] += 1
            got = row.get("pipeline")
            if got is not None:
                cmp["paired"] += 1
                cmp["same_page"] += bool(got.get("same_page"))
                cmp["same_line"] += bool(got.get("same_line"))
                cmp["same_spot"] += bool(got.get("same_spot"))
    total = sum(counts.values())
    with_text = total - counts["no_text"]
    placed = counts["placed"] + counts["moved"] - page_level
    return {"total": total, "with_text": with_text, "placed": placed,
            "page_level": page_level, "titles_apart": placed - sheet_titles,
            "page_level_by_kind": dict(page_level_by_kind.most_common()),
            "sheet_titles_by_kind": dict(sheet_titles_by_kind.most_common()),
            "boring_headings": sum(headings.values()),
            "boring_headings_by_outcome": dict(sorted(headings.items())),
            "counts": dict(counts), "anchors": dict(anchors.most_common()),
            "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "by_source": {k: dict(v) for k, v in by_source.items()},
            "step1": dict(step1), "compare": dict(cmp),
            "pipeline_objects": sum(i.get("pipeline_objects", 0)
                                    for i in items),
            "pipeline_boxed": sum(i.get("pipeline_boxed", 0) for i in items),
            "rate": placed / with_text if with_text else None}


def _in_set(items: List[Dict[str, Any]], which: str) -> List[Dict[str, Any]]:
    if which == "all":
        return list(items)
    return [i for i in items if (i.get("set") or "open") == which]


def _sets(items: List[Dict[str, Any]]) -> List[str]:
    """The sets a list of items is reported by: open and blind where it has
    both, and the two together."""
    present = [w for w in SETS if _in_set(items, w)]
    return present + (["all"] if len(present) > 1 else [])


_GATE_HEAD = ["| {} | set | values | on pages with text | placed at a spot "
              "(the gate) | page level | every lab title apart | boring "
              "heading only (sent to the title) | of which moved | several "
              "places | text, not found | no text | no page | gate |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
              "---:|---|"]


def _gate_row(name: str, which: str, s: Dict[str, Any]) -> str:
    c = s["counts"]
    gate = ("-" if s["rate"] is None else
            ("PASS" if s["rate"] >= GATE else "MISS"))
    return (f"| {name} | {which} | {s['total']} | {s['with_text']} | "
            f"{_pct(s['placed'], s['with_text'])} | {s['page_level']} | "
            f"{_pct(s['titles_apart'], s['with_text'])} | "
            f"{s['boring_headings']} | "
            f"{c.get('moved', 0)} | {c.get('several', 0)} | "
            f"{c.get('not_found', 0)} | {c.get('no_text', 0)} | "
            f"{c.get('no_page', 0)} | {gate} |")


def _kinds(counts: Dict[str, int]) -> str:
    return ", ".join(f"{k} {n}" for k, n in counts.items()) or "none"


def _title_row(name: str, which: str, s: Dict[str, Any]) -> str:
    return (f"| {name} | {which} | {_kinds(s['sheet_titles_by_kind'])} | "
            f"{_kinds(s['page_level_by_kind'])} | "
            f"{_kinds(s['boring_headings_by_outcome'])} |")


_CMP_HEAD = ["| {} | set | placed and paired | same page | same line, of "
             "same page | same spot, of same page | pipeline objects boxed |",
             "|---|---|---:|---:|---:|---:|---:|"]


def _cmp_row(name: str, which: str, s: Dict[str, Any]) -> str:
    cmp = s["compare"]
    n, m = cmp.get("paired", 0), cmp.get("same_page", 0)
    return (f"| {name} | {which} | {n} | {_pct(m, n)} | "
            f"{_pct(cmp.get('same_line', 0), m)} | "
            f"{_pct(cmp.get('same_spot', 0), m)} | "
            f"{_pct(s['pipeline_boxed'], s['pipeline_objects'])} |")


def _failures(items: List[Dict[str, Any]], status: str,
              limit: int = 60) -> List[str]:
    out = []
    for item in items:
        for row in item.get("objects") or []:
            if row["status"] == status:
                out.append(f"| {item['item']} | {row['cited']} | "
                           f"{row['kind']} | {row['what']} | "
                           f"{row.get('source') or 'none'} | "
                           f"{row.get('page_words', 0)} | "
                           f"{row.get('candidates', '')} |")
    return out[:limit]


def _disagreements(items: List[Dict[str, Any]], same_page: bool,
                   limit: int = 60) -> List[str]:
    out = []
    for item in items:
        for row in item.get("objects") or []:
            got = row.get("pipeline")
            if got is None or got.get("same_line") or \
                    bool(got.get("same_page")) != same_page:
                continue
            if same_page:
                out.append(f"| {item['item']} | {row['page']} | "
                           f"{row['kind']} | {row['what']} | "
                           f"{row['anchor']} | {got.get('dy')} | "
                           f"{got.get('mine_reads', '')} | "
                           f"{got.get('theirs_reads', '')} |")
            else:
                out.append(f"| {item['item']} | {row['page']} | "
                           f"{row['kind']} | {row['what']} | "
                           f"{row['anchor']} | {got.get('their_page')} |")
    return out[:limit]


def _details(title: str, items: List[Dict[str, Any]]) -> List[str]:
    """One stage's or arm's open items in detail; its blind items as
    counts (the rules are tuned on open items only)."""
    s = stage_summary(items)
    lines = ["", f"## {title}", ""]
    if s["step1"]:
        lines.append(f"Step 1 (cited pages): {json.dumps(s['step1'])}.")
    lines += [f"Placed by anchor: {json.dumps(s['anchors'])}.",
              f"By kind: {json.dumps(s['by_kind'])}.",
              f"By text source of the cited page: "
              f"{json.dumps(s['by_source'])}."]
    blind = _in_set(items, "blind")
    if blind:
        b = stage_summary(blind)
        lines += ["", f"Blind items ({len(blind)}), counts only: "
                      f"{json.dumps(b['counts'])}, page level "
                      f"{b['page_level']}."]
    shown = _in_set(items, "open")
    if not shown:
        return lines
    lines += ["", "| open item | placed | moved | several | not found | "
                  "no text | QA note |", "|---|---:|---:|---:|---:|---:|---|"]
    for item in shown:
        if item.get("error"):
            lines.append(f"| {item['item']} | ERROR {item['error']} "
                         f"| | | | | |")
            continue
        c = item["counts"]
        lines.append(f"| {item['item']} | {c['placed']} | {c['moved']} "
                     f"| {c['several']} | {c['not_found']} | "
                     f"{c['no_text']} | {item['note']} |")
    for status, heading in (("not_found", "Text present, not found"),
                            ("several", "Printed in several places"),
                            ("no_text", "No text on the page")):
        rows = _failures(shown, status)
        if rows:
            lines += ["", f"### {heading}", "",
                      "| item | page | kind | what | text source | "
                      "words on page | candidates |",
                      "|---|---:|---|---|---|---:|---:|"] + rows
    rows = _disagreements(shown, True)
    if rows:
        lines += ["", "### Same page, another line than the other box", "",
                  "| item | page | kind | what | anchor | dy (pt) | "
                  "printed under mine | printed under the other |",
                  "|---|---:|---|---|---|---:|---|---|"] + rows
    rows = _disagreements(shown, False)
    if rows:
        lines += ["", "### The other box is on another page", "",
                  "| item | page | kind | what | anchor | other page |",
                  "|---|---:|---|---|---|---:|"] + rows
    return lines


def write_results(out: Path, results: Dict[str, List[Dict[str, Any]]],
                  held_out: Optional[Dict[str, Tuple[
                      str, List[Dict[str, Any]]]]] = None) -> List[str]:
    held_out = held_out or {}
    lines = ["# Placement by text match (RI-87 piece 5)", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')} by "
             f"`measure_placement.py`. A value is one object: a log's "
             f"layer, sample, drive or water reading; a lab test; a "
             f"calculation input or result; a narrative citation. The gate "
             f"(plan section 8.4): at least 90 % of the values on pages "
             f"with text placed at a spot. Page level (a lab sheet's title, "
             f"a quote found on its page that search could not box) is "
             f"counted apart and kept out of the gate. Open and blind are "
             f"the stage's own split (OPEN.txt); the rules are tuned on open "
             f"items only, and blind items are given as counts.", "",
             "## In sample: the plain answers (Opus 5.5, RI-86)", "",
             *[h.format("stage") for h in _GATE_HEAD]]
    for stage, items in results.items():
        for which in _sets(items):
            lines.append(_gate_row(stage, which,
                                   stage_summary(_in_set(items, which))))
    if held_out:
        lines += ["", "## Held out: the pipeline arms' model-alone answers, "
                      "their boxes taken off", "",
                  *[h.format("arm") for h in _GATE_HEAD]]
        for arm, (_stage, items) in held_out.items():
            for which in _sets(items):
                lines.append(_gate_row(arm, which,
                                       stage_summary(_in_set(items, which))))
    lines += ["", "## Against the other box", "",
              "In sample, the other box is the pipeline's Opus arm's "
              "(logs_opus, lab_opus_linkon, calc_opus) for the same object, "
              "where it has one; held out, the box the object itself had. "
              "Same page; of those on the same page, same printed line (the "
              "two boxes' vertical extents overlap, 2 pt slack) and same "
              "spot (they also overlap across).", "",
              *[h.format("stage or arm") for h in _CMP_HEAD]]
    for stage, items in results.items():
        for which in _sets(items):
            lines.append(_cmp_row(stage, which,
                                  stage_summary(_in_set(items, which))))
    for arm, (_stage, items) in held_out.items():
        for which in _sets(items):
            lines.append(_cmp_row(arm, which,
                                  stage_summary(_in_set(items, which))))
    lab = [(n, items) for n, items in results.items() if n == "lab"] + [
        (a, items) for a, (st, items) in held_out.items() if st == "lab"]
    if lab:
        lines += ["", "## Lab titles by test kind", "",
                  "Counted from each answer's own records (its test kinds), "
                  "blind included: the title of a page whose only test "
                  "the answer gives is that test, counted as its spot; and "
                  "the tests still placed by the page only; and the "
                  "tests whose boring label is printed only as a heading, "
                  "by where they went instead.",
                  "", "| stage or arm | set | only test's title (spot) | "
                  "page level | boring heading only |",
                  "|---|---|---|---|---|"]
        for name, items in lab:
            for which in _sets(items):
                lines.append(_title_row(name, which,
                                        stage_summary(_in_set(items,
                                                              which))))
    for stage, items in results.items():
        lines += _details(f"{stage}, in sample", items)
    for arm, (stage, items) in held_out.items():
        lines += _details(f"{arm} ({stage}), held out", items)
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(
        {"summaries": {s: stage_summary(items)
                       for s, items in results.items()},
         "held_out_summaries": {a: stage_summary(items)
                                for a, (_, items) in held_out.items()},
         "items": results,
         "held_out": {a: items for a, (_, items) in held_out.items()}},
        indent=1, default=str), encoding="utf-8")
    return lines


def _paths(args: argparse.Namespace) -> Dict[str, Path]:
    main = rcl.main_checkout()
    raw = Path(args.raw_dir) if args.raw_dir else main / rcl.RAW_REL
    out_root = Path(args.out_root) if args.out_root else main / rcl.OUT_REL
    return {"main": main, "raw": raw, "reports": raw / "corpus",
            "di": raw / "di", "truth": raw / "truth", "out_root": out_root,
            "out": Path(args.out_dir) if args.out_dir
            else out_root / OUT_NAME}


def _say(name: str, which: str, s: Dict[str, Any]) -> str:
    c, cmp = s["counts"], s["compare"]
    titles = ""
    if s["sheet_titles_by_kind"] or s["page_level_by_kind"]:
        titles = (f"; every lab title apart "
                  f"{_pct(s['titles_apart'], s['with_text'])}; only "
                  f"test's title {_kinds(s['sheet_titles_by_kind'])}; page "
                  f"level "
                  f"by kind {_kinds(s['page_level_by_kind'])}; boring "
                  f"heading only {s['boring_headings']} "
                  f"({_kinds(s['boring_headings_by_outcome'])})")
    return (f"{name} {which}: {_pct(s['placed'], s['with_text'])} placed at "
            f"a spot on pages with text (page level {s['page_level']}, "
            f"moved {c.get('moved', 0)}; several {c.get('several', 0)}, not "
            f"found {c.get('not_found', 0)}, no text {c.get('no_text', 0)}); "
            f"beside the other box {cmp.get('paired', 0)} paired, "
            f"{cmp.get('same_page', 0)} on the same page, same line "
            f"{cmp.get('same_line', 0)}{titles}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="all",
                    choices=("all",) + tuple(STAGES))
    ap.add_argument("--raw-dir", default=None,
                    help="the private corpus folder (default: the main "
                         "checkout's 2026-09-16 raw folder)")
    ap.add_argument("--out-root", default=None,
                    help="the folder holding the plain and pipeline arms "
                         "(default: the 2026-10-24 claude_local raw folder)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    paths = _paths(args)
    out = paths["out"]
    for root in {paths["main"], rcl.REPO}:
        if rcl._inside(out, root) and rcl.git_ignored(out, root) is False:
            print(f"refused: {out} is inside the repo and git does not "
                  f"ignore it; the results name private pages")
            return rcl.REFUSED
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else (args.stage,)
    started = time.time()
    results, held_out = measure(stages, paths)
    write_results(out, results, held_out)
    for stage in stages:
        for which in _sets(results[stage]):
            print(_say(stage, which,
                       stage_summary(_in_set(results[stage], which))))
    for arm, (_stage, items) in held_out.items():
        for which in _sets(items):
            print(_say(arm, which, stage_summary(_in_set(items, which))))
    print(f"written to {out} in {time.time() - started:.0f} s")
    return rcl.OK


if __name__ == "__main__":
    sys.exit(main())
