"""WP7 measurement: the library's QUERY LAYER, with no model at all.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp7_library
    ... --json out.json          # the numbers as data
    ... --only Q07 Q13           # a few questions
    ... --verbose                # every question's misses

WHAT IT MEASURES, AND WHAT IT DOES NOT. The library sub-agent is two halves:
a deterministic query layer, and a model that decides what to ask it and
writes the answer. This is the FIRST half. Twenty hand-written questions go
through :mod:`report_ingest.library` over a synthetic library of six records
(seven rows -- one of the six carries a report bound inside it), each with
the report ids and the PDF pages a correct answer must carry, and the score
is precision and recall over both. No model is called, nothing is asked of
the network, and the whole thing runs in a second, so it can be re-run after
every change to the retrieval and is free to be wrong the first few times.

TWO COLUMNS, and the second is the one that can surprise you.

**The chosen query** is what the sub-agent's tool call returns once the model
has picked the right query -- ``list_reports(post=...)``, ``compare(field)``,
``where_is(text)``. It measures the query layer and the records, not the
search.

**Search alone** puts the question's own prose into ``find()`` and nothing
else: no filter chosen, no field named, no report named. That is the floor a
model gets when it reaches for search rather than for the right query, and it
is the number that says whether the full-text index answers a question
PHRASED AS A QUESTION. A date-range question cannot be answered by searching
for its words, and the report says so rather than hiding it.

THE MODEL HALF IS MEASURED ON THE CLUSTER, not here -- see the README section
"The report library". It runs the sub-agent over a private
``library_questions.json`` and scores the report ids it CITES; the synthetic
``report_ingest/library_questions.EXAMPLE.json`` ships in the wheel and shows
the shape.

PRIVACY. Every record this builds is synthetic and lives in a temporary
folder: the places, firms, posts and projects are invented in
``report_ingest/tests/library_fixtures.py`` and nothing from the corpus is
read, printed or written.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from report_ingest.library import Library
from report_ingest.library_agent import run_query
from report_ingest.tests.library_fixtures import build_library

#: How many hits the search-alone column is allowed. Eight is the library
#: agent's own default for ``find``.
SEARCH_K = 8


@dataclass
class Question:
    """One hand-written question, the query for it, and the right answer."""

    id: str
    question: str
    #: The library query a correct delegation makes: ``(name, arguments)``.
    query: Tuple[str, Dict[str, Any]]
    #: The report ids a correct answer names.
    reports: Set[str]
    #: The ``(report, page)`` pairs a correct answer cites. A report expected
    #: with no pair is one whose record carries the fact with no citation --
    #: which is a real state and must not be scored as a missing page.
    pages: Set[Tuple[str, int]] = field(default_factory=set)
    #: True where the question's own words cannot be searched for -- a date
    #: range, a count. The search-alone column reports these apart rather
    #: than counting them as retrieval failures.
    unsearchable: bool = False


def questions() -> List[Question]:
    """The twenty, against the synthetic library's six records."""
    return [
        Question("Q01", "Which reports were written for the Vale Harbour "
                        "post?",
                 ("list_reports", {"post": "Vale Harbour"}),
                 {"L02", "L02-bound1", "L05"}),
        Question("Q02", "Which reports did Meridian Geotechnical write?",
                 ("list_reports", {"firm": "Meridian Geotechnical"}),
                 {"L02", "L05"}),
        Question("Q03", "Which reports were done for the design-build phase?",
                 ("list_reports", {"phase": "Design-build"}),
                 {"L01", "L04"}),
        Question("Q04", "Which reports hold cone penetration soundings?",
                 ("list_reports", {"has_kind": "cpt"}), {"L03"}),
        Question("Q05", "Which reports were written in 2026?",
                 ("list_reports", {"date_from": "2026-01-01",
                                   "date_to": "2026-12-31"}),
                 {"L01", "L04"}, unsearchable=True),
        Question("Q06", "Which of these documents is a recommendation "
                        "letter rather than a report?",
                 ("list_reports", {"document_type": "recommendation letter"}),
                 {"L04"}),
        Question("Q07", "Which report has another report bound inside it?",
                 ("list_reports", {"has_kind": "bound"}), {"L02"}),
        Question("Q08", "Which of these sites was called potentially "
                        "liquefiable?",
                 ("compare", {"field": "liquefactionPotential"}),
                 {"L01", "L02", "L05"}, {("L01", 4), ("L02", 3)}),
        Question("Q09", "What site class did each report give?",
                 ("compare", {"field": "siteClass"}),
                 {"L01", "L02", "L02-bound1", "L05"},
                 {("L01", 4), ("L02", 4), ("L05", 4)}),
        # Five reports answered this one, and the fifth (L05) answered it
        # with no citation on the page -- which is why the expected pages
        # name only three of the five.
        Question("Q10", "What foundation type did each report recommend?",
                 ("compare", {"field": "recommendedFoundations"}),
                 {"L01", "L02", "L02-bound1", "L04", "L05"},
                 {("L01", 4), ("L02", 4), ("L04", 3)}),
        Question("Q11", "What did the Harbour Gate report recommend, and at "
                        "what capacity?",
                 ("facts", {"report_id": "L02",
                            "fields": ["recommendedFoundations",
                                       "bearingCapacity"]}),
                 {"L02"}, {("L02", 4)}),
        Question("Q12", "Who wrote the study bound inside the Harbour Gate "
                        "report?",
                 ("facts", {"report_id": "L02-bound1",
                            "fields": ["geotechnicalEngineerFirm"]}),
                 {"L02-bound1"}, {("L02-bound1", 15)}),
        Question("Q13", "Which report and page prints an allowable bearing "
                        "pressure of 3,000 psf?",
                 ("where_is", {"text": "3,000 psf allowable for spread "
                                       "footings"}),
                 {"L01"}, {("L01", 4)}),
        Question("Q14", "Where does a report say the ground is severely "
                        "corrosive to buried concrete?",
                 ("where_is", {"text": "severely corrosive to buried "
                                       "concrete"}),
                 {"L05"}, {("L05", 4)}),
        Question("Q15", "Which report mentions a probabilistic seismic "
                        "hazard analysis?",
                 ("find", {"text": "probabilistic seismic hazard analysis",
                           "k": 4}),
                 {"L02"}, {("L02", 4)}),
        Question("Q16", "Which report ran a lateral pile analysis in "
                        "LPILE?",
                 ("find", {"text": "LPILE", "k": 4}),
                 {"L02"}, {("L02", 20), ("L02", 21)}),
        Question("Q17", "What did the Northfield Annex report work out?",
                 ("calculations", {"report_id": "L04"}),
                 {"L04"}, {("L04", 20), ("L04", 21)}),
        Question("Q18", "What laboratory testing did the Harbour Gate report "
                        "run?",
                 ("lab_summary", {"report_id": "L02"}),
                 {"L02"}, {("L02", 12), ("L02", 13)}),
        Question("Q19", "How deep did the boring in the Cedar Hollow report "
                        "go?",
                 ("explorations", {"report_id": "L05", "kind": "boring"}),
                 {"L05"}, {("L05", 7), ("L05", 8)}),
        Question("Q20", "Where do two readings of the same report disagree, "
                        "anywhere in the library?",
                 ("disagreements", {}),
                 {"L01", "L02", "L03", "L04"},
                 {("L01", 12), ("L01", 14), ("L02", 3), ("L02", 8),
                  ("L03", 5), ("L03", 6), ("L03", 7), ("L03", 8),
                  ("L03", 9), ("L04", 12)}),
    ]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    """Hits, misses and false positives over a set of expected things."""

    hit: int = 0
    missed: int = 0
    extra: int = 0

    def add(self, got: Set[Any], want: Set[Any]) -> None:
        self.hit += len(got & want)
        self.missed += len(want - got)
        self.extra += len(got - want)

    @property
    def precision(self) -> Optional[float]:
        total = self.hit + self.extra
        return self.hit / total if total else None

    @property
    def recall(self) -> Optional[float]:
        total = self.hit + self.missed
        return self.hit / total if total else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def line(self, label: str) -> str:
        return (f"{label:<10} precision {_pct(self.precision)}  "
                f"recall {_pct(self.recall)}  F1 {_pct(self.f1)}  "
                f"(hit {self.hit}, missed {self.missed}, extra {self.extra})")

    def to_dict(self) -> Dict[str, Any]:
        return {"hit": self.hit, "missed": self.missed, "extra": self.extra,
                "precision": self.precision, "recall": self.recall,
                "f1": self.f1}


def _pct(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def _returned(result: Any) -> Tuple[Set[str], Set[Tuple[str, int]]]:
    """The reports and the ``(report, page)`` pairs one query result names."""
    reports = {report for report, _page in result.rows}
    reports |= set(result.reports)
    pages = {(report, page) for report, page in result.rows
             if isinstance(page, int)}
    return reports, pages


def measure(library: Library, asked: Sequence[Question]) -> Dict[str, Any]:
    """Both columns over every question, plus the per-question rows."""
    chosen_reports, chosen_pages = Score(), Score()
    search_reports, search_pages = Score(), Score()
    rows: List[Dict[str, Any]] = []

    for question in asked:
        name, arguments = question.query
        result = run_query(library, name, arguments)
        got_reports, got_pages = _returned(result)
        row_chosen_r, row_chosen_p = Score(), Score()
        row_chosen_r.add(got_reports, question.reports)
        row_chosen_p.add(got_pages, question.pages)
        chosen_reports.add(got_reports, question.reports)
        chosen_pages.add(got_pages, question.pages)

        found = library.find(question.question, k=SEARCH_K)
        hit_reports = {hit["report"] for hit in found["hits"]}
        hit_pages = {(hit["report"], page) for hit in found["hits"]
                     for page in hit["pages"]}
        row_search_r, row_search_p = Score(), Score()
        row_search_r.add(hit_reports, question.reports)
        row_search_p.add(hit_pages, question.pages)
        if not question.unsearchable:
            search_reports.add(hit_reports, question.reports)
            search_pages.add(hit_pages, question.pages)

        rows.append({
            "id": question.id, "question": question.question,
            "query": f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})",
            "error": result.error,
            "expected_reports": sorted(question.reports),
            "chosen_reports": sorted(got_reports),
            "chosen_missed": sorted(question.reports - got_reports),
            "chosen_extra": sorted(got_reports - question.reports),
            "expected_pages": sorted(question.pages),
            "chosen_pages_missed": sorted(question.pages - got_pages),
            "search_reports": sorted(hit_reports),
            "search_missed": sorted(question.reports - hit_reports),
            "unsearchable": question.unsearchable,
            "chosen": {"reports": row_chosen_r.to_dict(),
                       "pages": row_chosen_p.to_dict()},
            "search": {"reports": row_search_r.to_dict(),
                       "pages": row_search_p.to_dict()},
        })

    return {
        "n_questions": len(asked),
        "chosen": {"reports": chosen_reports.to_dict(),
                   "pages": chosen_pages.to_dict()},
        "search": {"reports": search_reports.to_dict(),
                   "pages": search_pages.to_dict()},
        "n_unsearchable": sum(1 for q in asked if q.unsearchable),
        "questions": rows,
        "_scores": {"chosen_reports": chosen_reports,
                    "chosen_pages": chosen_pages,
                    "search_reports": search_reports,
                    "search_pages": search_pages},
    }


def report(measured: Dict[str, Any], stats: Dict[str, Any],
           verbose: bool = False) -> str:
    """The scorecard, as text."""
    scores = measured["_scores"]
    out = [
        "# WP7 -- the library query layer, with NO model",
        "",
        f"{measured['n_questions']} hand-written question(s) over a "
        f"synthetic library of {stats['reports']} row(s) "
        f"({stats['bound_inside_another']} bound inside another), "
        f"{stats['indexed_chunks']} indexed chunk(s).",
        "",
        "## The chosen query -- what the sub-agent's tool call returns",
        "",
        scores["chosen_reports"].line("reports"),
        scores["chosen_pages"].line("pages"),
        "",
        "## Search alone -- find(the question) and nothing else",
        "",
        f"The retrieval floor: no filter chosen, no field named, no report "
        f"named. {measured['n_unsearchable']} question(s) are excluded as "
        f"unsearchable by construction (a date range has no words to find).",
        "",
        scores["search_reports"].line("reports"),
        scores["search_pages"].line("pages"),
        "",
        "## Per question",
        "",
        "| id | query | reports P/R | pages P/R | search reports P/R |",
        "|---|---|---|---|---|",
    ]
    for row in measured["questions"]:
        chosen_r = row["chosen"]["reports"]
        chosen_p = row["chosen"]["pages"]
        search_r = row["search"]["reports"]
        out.append(
            f"| {row['id']} | {row['query'][:52]} | "
            f"{_pct(chosen_r['precision'])}/{_pct(chosen_r['recall'])} | "
            f"{_pct(chosen_p['precision'])}/{_pct(chosen_p['recall'])} | "
            f"{_pct(search_r['precision'])}/{_pct(search_r['recall'])}"
            f"{' (unsearchable)' if row['unsearchable'] else ''} |")

    misses = [row for row in measured["questions"]
              if row["chosen_missed"] or row["chosen_extra"]
              or row["chosen_pages_missed"] or row["error"]]
    if misses:
        out += ["", "## What the chosen query got wrong", ""]
        for row in misses:
            out.append(f"- **{row['id']}** {row['question']}")
            if row["error"]:
                out.append(f"    - the query failed: {row['error']}")
            if row["chosen_missed"]:
                out.append(f"    - missed report(s): "
                           f"{', '.join(row['chosen_missed'])}")
            if row["chosen_extra"]:
                out.append(f"    - returned report(s) it should not: "
                           f"{', '.join(row['chosen_extra'])}")
            if row["chosen_pages_missed"]:
                out.append("    - missed page(s): " + ", ".join(
                    f"{report} p{page}" for report, page
                    in row["chosen_pages_missed"]))
    else:
        out += ["", "Every chosen query returned exactly the reports and the "
                    "pages the question expects.", ""]

    if verbose:
        out += ["", "## Search alone, question by question", ""]
        for row in measured["questions"]:
            mark = " (unsearchable)" if row["unsearchable"] else ""
            out.append(f"- **{row['id']}**{mark} found "
                       f"{', '.join(row['search_reports']) or 'nothing'}; "
                       f"missed {', '.join(row['search_missed']) or 'nothing'}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None,
                        help="question ids to run (default: all twenty)")
    parser.add_argument("--json", default="",
                        help="write the numbers to this file as JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="print the search-alone result per question")
    parser.add_argument("--keep", default="",
                        help="build the synthetic library in this folder "
                             "and leave it there (default: a temp folder)")
    args = parser.parse_args(argv)

    asked = questions()
    if args.only:
        wanted = {name.upper() for name in args.only}
        asked = [question for question in asked if question.id in wanted]
        if not asked:
            print(f"no question matches {', '.join(args.only)}")
            return 2

    root = args.keep or tempfile.mkdtemp(prefix="library_wp7_")
    build_library(root)
    library = Library(root)
    try:
        stats = library.library_stats()
        measured = measure(library, asked)
    finally:
        library.close()

    text = report(measured, stats, verbose=args.verbose)
    print(text)
    if args.keep:
        print(f"\nthe synthetic library is at {root}")
    if args.json:
        blob = {key: value for key, value in measured.items()
                if key != "_scores"}
        blob["library"] = {key: value for key, value in stats.items()
                           if key not in ("root", "db")}
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(blob, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
