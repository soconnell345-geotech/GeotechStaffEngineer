"""Made-up model answers for the builders' check mode (RI-87 piece 6).

One answer per case, for each stage's ``_Builder``, each carrying the values
the checks act on: depths inside and outside a log's ruler window, a water
depth below the hole and one with no total depth, negative depths,
percentages past 100, a recovery past 100, water contents and limits past
100 and below 0, a liquid limit below the plastic limit, a grading that
rises, a depth past 300 m, drive lengths that do not line up with the blows,
words outside a vocabulary, pit dimensions with no unit, answers whose page
numbers are positions in the call, are the report's, or mix the two, a
result printed on no page, and caption counts that differ from the
reading's.

:func:`build_default` builds every case the way today's code does (no
``checks`` argument) and returns what came out, as JSON. The snapshot it is
compared with (``checks_refuse_snapshot.json``) was written by this same
function on master (32b9645) before the check mode existed, so the
comparison is the proof that the default path did not move. RI-89 rewrote
it on purpose, and only where that item moves the default: a water content
and a liquid limit past 100 are kept (lab ``positions``), and a recovery
past 100 and an RQD below 0 are listed instead of clamped (log ``window``).

Every value here is invented.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

SNAPSHOT = os.path.join(os.path.dirname(__file__),
                        "checks_refuse_snapshot.json")

#: The pages of each made-up call.
LOG_PAGES = [10, 11]
LAB_PAGES = [20, 21, 22]
CALC_PAGES = [30, 31, 32]
CALC_ONE_PAGE = [40]
NARRATIVE_PAGES = [0, 1, 2, 3, 4, 5]
#: A sheet whose ruler reads 0 to 20 ft, with its slack.
LOG_WINDOW = (-0.5, 20.5)


def _prov(page: int, **kw: Any) -> Dict[str, Any]:
    return {"page": page, **kw}


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

def log_window_reading() -> Any:
    """A boring of 30 ft whose first sheet's ruler reads 0 to 20 ft."""
    from report_ingest.log_reader import LogReading
    return LogReading.model_validate({
        "investigation_id": "B-7", "kind": "boring", "depth_unit": "ft",
        "total_depth": 30.0, "elevation": 101.5, "elevation_unit": "ft",
        "drilling": {"method": "hollow stem auger",
                     "hammer_type": "automatic"},
        "layers": [
            {"top": 0.0, "bottom": 4.0, "description": "Brown silty SAND",
             "uscs": "SM", "prov": _prov(10, bbox=[10, 20, 200, 30])},
            {"top": 4.0, "bottom": 24.0, "description": "Grey lean CLAY",
             "uscs": "CL", "prov": _prov(11)},
            {"top": 24.0, "bottom": 30.0,
             "description": "Grey sandy GRAVEL", "prov": _prov(57)},
        ],
        "samples": [
            {"sample_id": "S-1", "top": 2.0, "bottom": 3.5, "kind": "spt",
             "recovery_percent": 105.0, "rqd_percent": -5.0,
             "fines_percent": 40.0, "water_content": 18.0,
             "prov": _prov(10)},
            {"sample_id": "S-2", "top": -2.0, "bottom": 1.0,
             "kind": "shelby", "prov": _prov(10)},
            {"sample_id": "S-3", "top": 22.0, "bottom": 23.5,
             "kind": "spt", "prov": _prov(11)},
            {"sample_id": "S-4", "top": 15.0, "bottom": 16.5,
             "kind": "piston", "prov": _prov(11)},
        ],
        "spt": [
            {"depth_top": 2.0, "depth_bottom": 3.5,
             "blows": ["3", "5", "7"], "n": 12, "sample_id": "S-1",
             "drive_lengths": [6.0, 6.0], "drive_length_unit": "in",
             "prov": _prov(10)},
            {"depth_top": 22.0, "depth_bottom": 23.5,
             "blows": ["10", "50/4\""], "n_printed": "50/4\"",
             "refusal": True, "sample_id": "S-3",
             "drive_lengths": [6.0, 4.0], "prov": _prov(11)},
        ],
        "water": [
            {"depth": 12.0, "when": "while_drilling", "prov": _prov(10)},
            {"depth": 25.0, "when": "at_completion", "casing_depth": 18.0,
             "prov": _prov(11)},
            {"depth": 33.0, "when": "after_hours", "hours": 24.0,
             "caved_depth": 40.0, "prov": _prov(11)},
            {"depth": 10.0, "when": "next morning", "prov": _prov(11)},
        ],
        "unsettled": [{"what": "a faint symbol at 9 ft", "page": 10,
                       "why": "too faint to read"}],
    })


def log_no_ruler_reading() -> Any:
    """A boring whose pages have no depth ruler at all."""
    from report_ingest.log_reader import LogReading
    return LogReading.model_validate({
        "investigation_id": "B-8", "kind": "auger boring", "depth_unit": "m",
        "total_depth": 6.0,
        "layers": [{"top": 0.0, "bottom": 2.5, "description": "Fill",
                    "prov": _prov(10)},
                   {"top": 2.5, "bottom": 6.0,
                    "description": "Stiff brown CLAY", "prov": _prov(11)}],
        "samples": [{"sample_id": "U1", "top": 3.0, "kind": "shelby",
                     "prov": _prov(11)}],
        "water": [{"depth": 4.2, "when": "at_completion",
                   "prov": _prov(11)}],
    })


def log_pit_reading() -> Any:
    """A test pit with no depth scale, one negative depth and a zero width."""
    from report_ingest.log_reader import LogReading
    return LogReading.model_validate({
        "investigation_id": "TP-3", "kind": "test_pit", "depth_unit": "m",
        "total_depth": -1.0,
        "pit": {"length": 3.0, "width": 0.0, "depth": 2.0, "unit": "m"},
        "layers": [{"top": 0.0, "bottom": 0.4, "description": "Topsoil",
                    "prov": _prov(10)},
                   {"top": -0.3, "bottom": 2.0,
                    "description": "Red sandy CLAY", "prov": _prov(10)}],
    })


def log_deep_reading() -> Any:
    """A layer and a total depth past 300 m inside a wide ruler window."""
    from report_ingest.log_reader import LogReading
    return LogReading.model_validate({
        "investigation_id": "B-9", "kind": "boring", "depth_unit": "ft",
        "total_depth": 1250.0,
        "layers": [{"top": 0.0, "bottom": 1100.0,
                    "description": "Weathered rock", "prov": _prov(10)},
                   {"top": 1100.0, "bottom": 1250.0,
                    "description": "Fresh rock", "prov": _prov(10)}],
    })


def log_pit_no_unit_reading() -> Any:
    """A test pit that prints no unit, no scale and no total depth."""
    from report_ingest.log_reader import LogReading
    return LogReading.model_validate({
        "investigation_id": "TP-4", "kind": "test_pit", "depth_unit": "",
        "units_known": False,
        "pit": {"length": 3.0, "width": 2.0, "prov": _prov(10)},
        "layers": [{"top": 0.0, "bottom": 1.2, "description": "Sandy fill",
                    "prov": _prov(10)}],
        "water": [{"depth": 1.0, "when": "while_drilling",
                   "prov": _prov(10)}],
    })


def log_cited_reading(cites: List[int]) -> Any:
    """A boring with one 5 ft layer per page in ``cites``, citing it, a
    water observation on the last and a note on the first: how one answer's
    page numbers are read."""
    from report_ingest.log_reader import LogReading
    layers = [{"top": 5.0 * n, "bottom": 5.0 * (n + 1),
               "description": f"Made-up layer {n + 1}", "prov": _prov(page)}
              for n, page in enumerate(cites)]
    return LogReading.model_validate({
        "investigation_id": "B-10", "kind": "boring", "depth_unit": "ft",
        "total_depth": 5.0 * len(cites), "layers": layers,
        "water": [{"depth": 8.0, "when": "at_completion",
                   "prov": _prov(cites[-1])}],
        "unsettled": [{"what": "a smudge", "page": cites[0],
                       "why": "smudged"}],
    })


#: ``case -> (reading, window, pages, no_ruler)``.
def log_cases() -> Dict[str, Tuple[Any, Any, List[int], bool]]:
    return {
        "window": (log_window_reading(), LOG_WINDOW, LOG_PAGES, False),
        "no_ruler": (log_no_ruler_reading(), None, LOG_PAGES, True),
        "pit": (log_pit_reading(), None, LOG_PAGES, True),
        "pit_no_unit": (log_pit_no_unit_reading(), None, LOG_PAGES, True),
        "deep": (log_deep_reading(), (-5.0, 1300.0), LOG_PAGES, False),
        # positions in the call: 0 and 1 of pages 10 and 11
        "positions": (log_cited_reading([0, 1]), LOG_WINDOW, LOG_PAGES,
                      False),
        # positions counted from 1 cannot be told from the report's pages
        "one_based": (log_cited_reading([1, 2, 3]), LOG_WINDOW, [3, 4, 5],
                      False),
        # positions from 0 on pages from 1: ambiguous, read as positions
        "zero_based": (log_cited_reading([0, 1, 2]), LOG_WINDOW, [1, 2, 3],
                       False),
    }


# ---------------------------------------------------------------------------
# lab
# ---------------------------------------------------------------------------

def lab_reading() -> Any:
    from report_ingest.lab_reader import LabSheetReading
    return LabSheetReading.model_validate({
        "depth_unit": "ft", "lab": "Made-up Testing Lab",
        "tests": [
            {"kind": "atterberg", "investigation_id": "B-1",
             "sample_id": "S-2", "depth_top": 5.0, "depth_bottom": 6.5,
             "ll": 25.0, "pl": 30.0, "pi": -5.0, "wc": 22.0,
             "prov": _prov(21)},
            {"kind": "gradation", "investigation_id": "B-1",
             "sample_id": "S-4", "depth_top": 1500.0,
             "gravel_percent": 105.0, "sand_percent": 60.0,
             "fines_percent": 12.0,
             "series": [{"name": "percent_passing", "x_unit": "mm",
                         "points": [{"x": 19.0, "y": 100.0, "label": "3/4"},
                                    {"x": 4.75, "y": 80.0, "label": "No. 4"},
                                    {"x": 0.075, "y": 92.0,
                                     "label": "No. 200"}]}],
             "prov": _prov(22, bbox=[50, 60, 300, 400])},
            {"kind": "compaction", "investigation_id": "B-2",
             "max_dry_density": {"value": 118.0, "unit": "pcf"},
             "optimum_wc": 12.0,
             "series": [{"name": "compaction", "x_unit": "%",
                         "y_unit": "pcf",
                         "points": [{"x": 8.0, "y": 112.0},
                                    {"x": -8.0, "y": 117.0},
                                    {"x": 14.0, "y": 116.0}]}],
             "prov": _prov(57)},
            {"kind": "summary_table", "test_type": "Summary of tests",
             "rows": [
                 {"investigation_id": "B-1", "sample_id": "S-1",
                  "depth_top": 2.0, "ll": 40.0, "pl": {"value": 20.0},
                  "pi": 20.0, "wc": 15.0,
                  "passing": [{"sieve": "No. 4", "percent_passing": 101.0},
                              {"sieve": "No. 200",
                               "percent_passing": 30.0}]},
                 {"investigation_id": "B-3", "sample_id": "S-5",
                  "depth_top": -4.0, "ll": 18.0, "pl": {"value": 26.0},
                  "pi": 8.0},
             ],
             "prov": _prov(20)},
            {"kind": "moisture", "investigation_id": "B-4",
             "depth_top": 2500.0, "wc": 30.0, "prov": _prov(22)},
        ],
        "unsettled": [{"what": "a stamp", "page": 21, "why": "smudged"}],
    })


def lab_no_unit_reading() -> Any:
    """A sheet that prints no depth unit, and one in an unknown unit."""
    from report_ingest.lab_reader import LabSheetReading
    return LabSheetReading.model_validate({
        "depth_unit": "",
        "tests": [{"kind": "moisture_content", "investigation_id": "B-5",
                   "depth_top": 2000.0, "depth_bottom": 4.0, "wc": 9.0,
                   "prov": _prov(21)}],
    })


def lab_positions_reading() -> Any:
    """A sheet whose answer cites positions in the call (0 to 2), with a
    water content and limits past 100 and a negative water content."""
    from report_ingest.lab_reader import LabSheetReading
    return LabSheetReading.model_validate({
        "depth_unit": "ft",
        "tests": [
            {"kind": "atterberg", "investigation_id": "B-7",
             "sample_id": "S-1", "depth_top": 4.0, "ll": 120.0, "pl": 45.0,
             "pi": 75.0, "wc": 150.0, "prov": _prov(0)},
            {"kind": "moisture_content", "investigation_id": "B-7",
             "sample_id": "S-3", "depth_top": 9.0, "wc": -3.0,
             "prov": _prov(2)},
        ],
        "unsettled": [{"what": "a stamp", "page": 1, "why": "smudged"}],
    })


def lab_cases() -> Dict[str, Tuple[Any, List[int]]]:
    return {"sheet": (lab_reading(), LAB_PAGES),
            "no_unit": (lab_no_unit_reading(), LAB_PAGES),
            "positions": (lab_positions_reading(), LAB_PAGES)}


# ---------------------------------------------------------------------------
# calc
# ---------------------------------------------------------------------------

def calc_reading() -> Any:
    from report_ingest.calc_reader import CalcReading
    return CalcReading.model_validate({
        "kind": "bearing_capacity", "program": "MadeUpCalc",
        "program_version": "2.1", "method": "a made-up method",
        "subject": "Footing F-1",
        "inputs": [
            {"name": "Made-up pad breadth", "value": 4.0, "unit": "ft",
             "prov": _prov(30, bbox=[10, 10, 90, 20])},
            {"name": "Embedment Df", "value": 2.0, "unit": "ft",
             "prov": _prov(31)},
            {"name": "Unit weight", "value": 120.0, "unit": "blorps",
             "prov": _prov(99)},
            {"name": "", "value": 7.0, "prov": _prov(30)},
        ],
        "results": [
            {"name": "Allowable bearing", "value": 2.0, "unit": "ksf",
             "prov": _prov(32)},
            {"name": "Factor of safety", "value": 3.7, "prov": _prov(31)},
            {"name": "Check", "text": "OK", "prov": _prov(32)},
        ],
        "summary": "A made-up bearing check.",
        "unsettled": [{"what": "a footnote", "page": 31,
                       "why": "cut off"}],
    })


def calc_one_page_reading() -> Any:
    from report_ingest.calc_reader import CalcReading
    return CalcReading.model_validate({
        "kind": "settlement",
        "inputs": [{"name": "Load", "value": 50.0, "unit": "kips",
                    "prov": _prov(0)}],
        "results": [{"name": "Settlement", "value": 0.8, "unit": "in",
                     "prov": _prov(3)}],
    })


#: The printed numbers the calc builder checks results against.
CALC_POOL = [2.0, 4.0, 50.0]


def calc_positions_reading() -> Any:
    """A printout of three pages whose answer cites positions (0 to 2)."""
    from report_ingest.calc_reader import CalcReading
    return CalcReading.model_validate({
        "kind": "settlement",
        "inputs": [{"name": "Made-up load", "value": 50.0, "unit": "kips",
                    "prov": _prov(0, bbox=[5, 5, 50, 15])},
                   {"name": "Made-up layer thickness", "value": 4.0,
                    "unit": "ft", "prov": _prov(1)}],
        "results": [{"name": "Made-up settlement", "value": 2.0,
                     "unit": "in", "prov": _prov(2)}],
        "unsettled": [{"what": "a note", "page": 1, "why": "cut off"}],
    })


def calc_cases() -> Dict[str, Tuple[Any, List[int]]]:
    return {"multi": (calc_reading(), CALC_PAGES),
            "one_page": (calc_one_page_reading(), CALC_ONE_PAGE),
            "positions": (calc_positions_reading(), CALC_PAGES)}


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------

def narrative_reading() -> Any:
    from report_ingest.narrative_reader import NarrativeReading
    return NarrativeReading.model_validate({
        "documentType": "Geotechnical report",
        "projectName": "Made-up Bridge Replacement",
        "boringCount": 4, "tableCount": 3, "figureCount": None,
        "citations": [
            {"field": "projectName", "page": 0,
             "quote": "Made-up Bridge Replacement"},
            {"field": "boringCount", "page": 2,
             "quote": "four borings were drilled"},
            {"field": "tableCount", "page": 3, "quote": "Table 3"},
            {"field": "documentType", "page": 0,
             "quote": "Geotechnical report"},
        ],
    })


#: ``case -> (reading, counted)``.
def narrative_cases() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    return {"counts_differ": (narrative_reading(),
                              {"tables": 4, "figures": 2}),
            "counts_agree": (narrative_reading(),
                             {"tables": 3, "figures": 0})}


# ---------------------------------------------------------------------------
# building them the way today's code does
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """JSON with infinities written out, so a window can be dumped."""
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_default() -> Dict[str, Any]:
    """Every case built as today's code builds it, as JSON-ready dicts."""
    from report_ingest import calc_reader, lab_reader, log_reader
    from report_ingest import narrative_reader

    out: Dict[str, Any] = {}
    for case, (reading, window, pages, no_ruler) in log_cases().items():
        builder = log_reader._Builder(reading, window, pages, "R-made-up",
                                      no_ruler=no_ruler)
        inv = builder.build()
        out[f"log/{case}"] = {
            "record": inv.model_dump(mode="json"),
            "unresolved": builder.unresolved, "changes": builder.changes}
    for case, (reading, pages) in lab_cases().items():
        builder = lab_reader._Builder(reading, pages, "R-made-up", [])
        tests = builder.build()
        out[f"lab/{case}"] = {
            "record": [t.model_dump(mode="json") for t in tests],
            "unresolved": builder.unresolved, "changes": builder.changes}
    for case, (reading, pages) in calc_cases().items():
        builder = calc_reader._Builder(reading, pages, "R-made-up", [],
                                       CALC_POOL)
        calc = builder.build()
        out[f"calc/{case}"] = {
            "record": calc.model_dump(mode="json"),
            "unresolved": builder.unresolved, "changes": builder.changes,
            "warnings": builder.warnings}
    for case, (reading, counted) in narrative_cases().items():
        builder = narrative_reader._Builder(NARRATIVE_PAGES)
        builder.add(reading)
        general, hazards, facts = builder.build(NARRATIVE_PAGES, counted,
                                                None)
        out[f"narrative/{case}"] = {
            "general": general.model_dump(mode="json"),
            "hazards": hazards.model_dump(mode="json"),
            "facts": facts.model_dump(mode="json"),
            "unresolved": builder.unresolved}
    return _jsonable(json.loads(json.dumps(_jsonable(out), default=str)))


def write_snapshot(path: str = SNAPSHOT) -> None:
    """Write :func:`build_default`'s output: run once, on master."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(build_default(), fh, indent=1, sort_keys=True)
        fh.write("\n")


def load_snapshot(path: str = SNAPSHOT) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
