"""Model-first ingest's readers on a scripted engine (BACKLOG RI-87 piece 7).

No model is called. A made-up report is made inside the test (pymupdf) and
opened through planlens; each reader is handed groups of its pages and a
:class:`~report_ingest.tests.fake_engine.FakeEngine` that replays made-up
answers in the model-first schemas. The tests check what each call carries
(the pages as a PDF or as pictures, the stage's instruction, the answer
schema), that a reply cut off at its output cap is asked again as two
halves, that a narrative read in parts gets one summary call, that the
builders' flag-mode entries and the schema's added fields reach the result,
and that the result keeps today's shape. Every identifier, depth and value
here is invented.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from report_ingest import model_first as mf
from report_ingest import model_first_pages as mfp
from report_ingest import model_first_read as mfr
from report_ingest.checks import LISTED
from report_ingest.model_first_pages import PageGroup
from report_ingest.tests.fake_engine import FakeEngine

pymupdf = pytest.importorskip("pymupdf")

#: The made-up report's id, as the instruction names it.
REPORT = "R00"
#: How many pages the made-up report has.
N_PAGES = 30


@pytest.fixture(scope="module")
def report_pdf():
    doc = pymupdf.open()
    for n in range(N_PAGES):
        page = doc.new_page()
        page.insert_text((72, 72), f"MADE-UP PAGE {n}", fontsize=11)
        page.insert_text((72, 90), f"made-up words of page {n}", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def doc(report_pdf):
    from planlens.document import open_document

    opened = open_document(report_pdf, name="made_up")
    yield opened
    opened.close()


def _engine(turns, pdf=True):
    engine = FakeEngine(turns)
    engine.accepts_pdf = pdf
    return engine


def _pdf_pages(block) -> list:
    """The made-up page numbers a PDF block carries, in order."""
    with pymupdf.open(stream=block["data"], filetype="pdf") as cut:
        return [int(page.get_text().split()[2]) for page in cut]


def _blocks(call) -> list:
    return call["messages"][0]["content"]


def _prov(page, **kw):
    return {"page": page, **kw}


def _cut_off():
    return {"text": "", "stop_reason": "max_tokens"}


# ---------------------------------------------------------------------------
# made-up answers in the model-first schemas
# ---------------------------------------------------------------------------

def _log(pages, *, hole="QX-1", top=0.0, sample="S-1", **extra):
    first = pages[0]
    return mf.LogReading.model_validate({
        "investigation_id": hole, "kind": "boring", "depth_unit": "ft",
        "total_depth": 40.0,
        "layers": [{"top": top, "bottom": top + 10.0,
                    "description": "Made-up brown silty SAND", "uscs": "SM",
                    "prov": _prov(first, bbox=[1, 2, 3, 4])}],
        "samples": [{"sample_id": sample, "top": top + 2.0,
                     "bottom": top + 3.5, "kind": "spt",
                     "recovery_length": 14.0, "recovery_length_unit": "in",
                     "prov": _prov(first)}],
        "water": [{"depth": top + 5.0, "when": "while_drilling",
                   "prov": _prov(first)}],
        **extra})


def _lab(pages, *, hole="QX-2"):
    first = pages[0]
    return mf.LabSheetReading.model_validate({
        "depth_unit": "ft",
        "tests": [
            {"kind": "atterberg", "investigation_id": hole, "sample_id": "S-3",
             "depth_top": 4.0, "ll": 22.0, "pl": 30.0, "pi": -8.0,
             "prov": _prov(first)},
            {"kind": "summary_table", "test_type": "Made-up summary",
             "rows": [{"investigation_id": hole, "sample_id": "S-4",
                       "depth_top": 6.0, "wc": 21.0,
                       "elevation_bottom": {"value": 412.5, "unit": "ft"}}],
             "prov": _prov(pages[-1], bbox=[5, 6, 7, 8])},
        ]})


def _calc(values):
    from report_ingest.calc_reader import CalcReading

    return CalcReading.model_validate({
        "kind": "settlement", "program": "MadeUpCalc",
        "inputs": [{"name": name, "value": value, "unit": "ft",
                    "prov": _prov(page)} for name, value, page in values],
        "results": []})


def _narrative(project, page, **extra):
    from report_ingest.narrative_reader import NarrativeReading

    return NarrativeReading.model_validate({
        "projectName": project,
        "citations": [{"field": "projectName", "page": page,
                       "quote": f"MADE-UP PAGE {page}"}],
        **extra})


def _narrative_cites(field, value, cites):
    """An answer to one field, citing ``(page, quote)`` pairs."""
    from report_ingest.narrative_reader import NarrativeReading

    return NarrativeReading.model_validate({
        field: value,
        "citations": [{"field": field, "page": page, "quote": quote}
                      for page, quote in cites]})


# ---------------------------------------------------------------------------
# the models and the module's place
# ---------------------------------------------------------------------------

def test_the_plan_s_models_per_stage_are_one_constant():
    assert mfr.STAGE_MODELS == {"logs": "claude-sonnet-5-5",
                                "lab": "claude-opus-5-5",
                                "calc": "claude-sonnet-5-5",
                                "narrative": "claude-opus-5-5"}


def test_today_s_path_does_not_import_the_readers():
    code = ("import sys\n"
            "import report_ingest.graph, report_ingest.log_reader\n"
            "import report_ingest.lab_reader, report_ingest.calc_reader\n"
            "import report_ingest.narrative_reader\n"
            "assert 'report_ingest.model_first_read' not in sys.modules\n")
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# one call per group, on each route
# ---------------------------------------------------------------------------

def test_a_log_is_one_call_carrying_its_pages_as_a_pdf(doc):
    from report_ingest.log_reader import LogReadResult

    engine = _engine([{"final": _log([3, 4])}])
    got = mfr.read_logs(doc, [PageGroup("logs", [3, 4], key="log-3")],
                        engine, report_id=REPORT, route="auto")
    assert engine.n_calls == 1 and len(got) == 1
    call = engine.calls[0]
    blocks = _blocks(call)
    assert [b["type"] for b in blocks] == ["pdf", "text"]
    assert _pdf_pages(blocks[0]) == [3, 4]
    assert blocks[1]["text"] == mf.instruction("logs", REPORT, [3, 4],
                                               route="pdf")
    assert call["output_format"] is mf.answer_schema("logs")
    assert call["system"] is None and call["tools"] == []
    blob = got[0]
    # Today's shape, with no floor.
    from report_ingest.model import Investigation
    today = LogReadResult(investigation=Investigation(
        investigation_id="QX-0")).to_dict()
    assert set(today) <= set(blob)
    assert blob["disagreements"] == [] and blob["kept"] == []
    assert blob["floor"] is None
    assert blob["model_investigation"] == blob["investigation"]
    assert (blob["key"], blob["route"], blob["error"]) == ("log-3", "pdf", "")
    assert blob["prompt_version"] == mf.READ_PROMPT_VERSION
    assert blob["model_calls"] == 1 and blob["cost"]["calls"] == 1
    assert blob["calls"] == [{"pages": [3, 4], "stop_reason": "end_turn",
                              "cut_off": False, "answered": True}]
    record = Investigation.model_validate(blob["investigation"])
    assert record.investigation_id == "QX-1" and record.pages == [3, 4]


def test_an_engine_that_cannot_send_a_pdf_gets_pictures_and_text(doc):
    engine = _engine([{"final": _lab([7, 8])}], pdf=False)
    got = mfr.read_lab(doc, [PageGroup("lab", [7, 8], key="lab-7")], engine,
                       report_id=REPORT, route="auto")
    blocks = _blocks(engine.calls[0])
    assert [b["type"] for b in blocks] == ["text", "image", "text",
                                           "text", "image", "text", "text"]
    assert blocks[0]["text"].startswith("Page 7 (picture 1 of 2)")
    assert "made-up words of page 8" in blocks[5]["text"]
    assert blocks[-1]["text"] == mf.instruction("lab", REPORT, [7, 8],
                                                route="pictures")
    assert engine.calls[0]["output_format"] is mf.answer_schema("lab")
    assert got[0]["route"] == "pictures"


def test_the_pdf_route_is_refused_on_an_engine_that_cannot_send_one(doc):
    engine = _engine([], pdf=False)
    with pytest.raises(ValueError):
        mfr.read_lab(doc, [PageGroup("lab", [7], key="lab-7")], engine,
                     report_id=REPORT, route="pdf")
    assert engine.n_calls == 0


def test_each_group_is_one_call_and_one_result(doc):
    groups = [PageGroup("lab", [7, 8], key="lab-7", part=(1, 2)),
              PageGroup("lab", [9], key="lab-7", part=(2, 2)),
              PageGroup("lab", [12], key="lab-12")]
    engine = _engine([{"final": _lab([7, 8])}, {"final": _lab([9])},
                      {"final": _lab([12])}])
    got = mfr.read_lab(doc, groups, engine, report_id=REPORT)
    assert engine.n_calls == 3
    assert [_pdf_pages(_blocks(c)[0]) for c in engine.calls] == \
        [[7, 8], [9], [12]]
    assert [b["key"] for b in got] == ["lab-7-1", "lab-7-2", "lab-12"]
    assert [b["pages"] for b in got] == [[7, 8], [9], [12]]


def test_a_calculation_printout_is_one_call_built_without_a_pool(doc):
    from report_ingest.model import Calculation

    engine = _engine([{"final": _calc([("Made-up load", 50.0, 20),
                                       ("Made-up depth", 4.0, 21)])}])
    got = mfr.read_calcs(doc, [PageGroup("calc", [20, 21], key="calc-20")],
                         engine, report_id=REPORT)
    assert engine.n_calls == 1
    assert engine.calls[0]["output_format"] is mf.answer_schema("calc")
    record = Calculation.model_validate(got[0]["calculation"])
    assert [v.prov.page for v in record.inputs] == [20, 21]
    assert got[0]["model_calculation"] == got[0]["calculation"]
    assert got[0]["floor_calculation"] is None


# ---------------------------------------------------------------------------
# a reply cut off at its output cap
# ---------------------------------------------------------------------------

def test_a_cut_off_log_is_asked_again_as_two_halves_and_joined(doc):
    engine = _engine([_cut_off(),
                      {"final": _log([3, 4], sample="S-1")},
                      {"final": _log([5, 6], top=10.0, sample="S-2",
                                     total_depth=None)}])
    got = mfr.read_logs(doc, [PageGroup("logs", [3, 4, 5, 6], key="log-3")],
                        engine, report_id=REPORT)
    assert engine.n_calls == 3
    assert [_pdf_pages(_blocks(c)[0]) for c in engine.calls] == \
        [[3, 4, 5, 6], [3, 4], [5, 6]]
    assert [_blocks(c)[1]["text"] for c in engine.calls[1:]] == \
        [mf.instruction("logs", REPORT, half, route="pdf")
         for half in ([3, 4], [5, 6])]
    blob = got[0]
    inv = blob["investigation"]
    assert [s["sample_id"] for s in inv["samples"]] == ["S-1", "S-2"]
    assert [ly["top"]["value"] for ly in inv["layers"]] == [0.0, 10.0]
    assert inv["pages"] == [3, 4, 5, 6]
    assert inv["total_depth"]["value"] == 40.0
    assert any("read in 2 calls" in row["why"] for row in blob["unresolved"])
    assert any("two halves" in w for w in blob["warnings"])
    assert [c["cut_off"] for c in blob["calls"]] == [True, False, False]


def test_a_half_still_cut_off_is_a_skipped_entry_naming_its_pages(doc):
    engine = _engine([_cut_off(), {"final": _lab([7, 8])}, _cut_off()])
    got = mfr.read_lab(doc, [PageGroup("lab", [7, 8, 9, 10], key="lab-7")],
                       engine, report_id=REPORT)
    assert engine.n_calls == 3
    blob = got[0]
    skipped = [r for r in blob["unresolved"] if r.get("refused_by")
               == "budget"]
    assert len(skipped) == 1
    assert skipped[0]["pages"] == [9, 10] and skipped[0]["page"] == 9
    assert "output cap" in skipped[0]["why"]
    assert "pages 9-10 were not read" in skipped[0]["why"]
    assert blob["error"] == "" and blob["tests"]


def test_a_lab_window_is_halved_where_a_new_sheet_begins(doc):
    def described(title):
        return {"description": {"title": title, "kind": "lab",
                                "names": "", "identifiers": [],
                                "continues": "", "summary": "made up"}}
    descriptions = {7: described("MADE-UP GRADATION"),
                    8: described("MADE-UP GRADATION"),
                    9: described("MADE-UP GRADATION"),
                    10: described("MADE-UP COMPACTION")}
    engine = _engine([_cut_off(), {"final": _lab([7, 8, 9])},
                      {"final": _lab([10])}])
    mfr.read_lab(doc, [PageGroup("lab", [7, 8, 9, 10], key="lab-7")],
                 engine, report_id=REPORT, descriptions=descriptions)
    assert [_pdf_pages(_blocks(c)[0]) for c in engine.calls[1:]] == \
        [[7, 8, 9], [10]]


def test_one_page_cut_off_and_an_answer_in_prose_are_skipped(doc):
    from report_ingest.reconciler import _carry_unresolved

    engine = _engine([_cut_off(), {"text": "no tool call"}])
    got = mfr.read_calcs(doc, [PageGroup("calc", [20], key="calc-20"),
                               PageGroup("calc", [22], key="calc-22")],
                         engine, report_id=REPORT)
    assert engine.n_calls == 2
    assert [b["calculation"] for b in got] == [None, None]
    assert "one page cannot be split" in got[0]["error"]
    assert "no structured answer" in got[1]["error"]
    qa = []
    _carry_unresolved(qa, got[0]["unresolved"])
    assert [e.kind for e in qa] == ["skipped"] and qa[0].pages == [20]


# ---------------------------------------------------------------------------
# a calculation read in two windows
# ---------------------------------------------------------------------------

def test_a_long_printout_s_two_answers_are_put_on_report_pages_and_folded(
        doc):
    from report_ingest.model import Calculation

    pages = list(range(5, 19))                        # 14 pages: two calls
    calls = mfp.calc_calls(PageGroup("calc", pages))
    assert len(calls) == 2
    second = calls[1]
    # The second answer cites positions in its own call (1 and 3).
    engine = _engine([
        {"final": _calc([("Applied load", 50.0, 5)])},
        {"final": _calc([("Layer thickness", 4.0, 1),
                         ("Footing width", 6.0, 3)])}])
    got = mfr.read_calcs(doc, [PageGroup("calc", pages, key="calc-5")],
                         engine, report_id=REPORT)
    assert engine.n_calls == 2
    record = Calculation.model_validate(got[0]["calculation"])
    assert [(v.name, v.prov.page) for v in record.inputs] == [
        ("Applied load", 5), ("Layer thickness", second[1]),
        ("Footing width", second[3])]
    assert second[1] == 16 and second[3] == 18
    assert "a position in the" in record.inputs[1].prov.note
    assert got[0]["pages"] == pages


# ---------------------------------------------------------------------------
# the narrative in parts, then the summary
# ---------------------------------------------------------------------------

def test_a_narrative_in_parts_is_merged_then_summarised_once(doc):
    from report_ingest.narrative_reader import SummaryReading

    groups = [PageGroup("narrative", [0, 1, 2], key="narrative",
                        part=(1, 2)),
              PageGroup("narrative", [3, 4], key="narrative", part=(2, 2))]
    engine = _engine([
        {"final": _narrative("Made-up Bridge", 0, quickSummary="Part one.")},
        {"final": _narrative("Made-up Tunnel", 3, quickSummary="Part two.")},
        {"final": SummaryReading(quickSummary="The whole made-up report.")}])
    blob = mfr.read_narrative(doc, groups, engine, report_id=REPORT)
    assert engine.n_calls == 3
    assert [_pdf_pages(_blocks(c)[0]) for c in engine.calls[:2]] == \
        [[0, 1, 2], [3, 4]]
    assert all(c["output_format"] is mf.answer_schema("narrative")
               for c in engine.calls[:2])
    assert engine.calls[2]["output_format"] is SummaryReading
    # The builder's own rule: the first answer kept, the second recorded.
    assert blob["general"]["projectName"] == "Made-up Bridge"
    assert any(r.get("what") == "projectName" for r in blob["unresolved"])
    assert blob["general"]["quickSummary"] == "The whole made-up report."
    assert blob["pages"] == [0, 1, 2, 3, 4]
    assert blob["model_calls"] == 3 and blob["key"] == "narrative"
    assert blob["confidence"]["projectName"] > 0


def test_a_narrative_in_one_call_gets_no_summary_call(doc):
    engine = _engine([{"final": _narrative("Made-up Bridge", 1,
                                           tableCount=4)}])
    blob = mfr.read_narrative(
        doc, [PageGroup("narrative", [0, 1, 2], key="narrative")], engine,
        report_id=REPORT)
    assert engine.n_calls == 1
    # The code's caption count (none on these pages) never replaces the
    # model's answer under flag.
    assert blob["general"]["tableCount"] == 4
    assert blob["facts"]["unresolved"] == blob["unresolved"]


# ---------------------------------------------------------------------------
# the flag builders' entries and the schema's added fields
# ---------------------------------------------------------------------------

def test_the_flag_builders_entries_reach_the_result(doc):
    engine = _engine([{"final": _lab([7])}])
    blob = mfr.read_lab(doc, [PageGroup("lab", [7], key="lab-7")], engine,
                        report_id=REPORT)[0]
    # A liquid limit below the plastic limit: out of the slots, listed.
    limits = [r for r in blob["unresolved"] if "liquid limit" in r["why"]]
    assert limits and LISTED in limits[0]["why"]
    kinds = [t["kind"] for t in blob["tests"]]
    assert "summary_table" in kinds


def test_a_log_with_no_ruler_keeps_its_depths_with_one_note(doc):
    engine = _engine([{"final": _log([3], top=-2.0)}])
    blob = mfr.read_logs(doc, [PageGroup("logs", [3], key="log-3")], engine,
                         report_id=REPORT)[0]
    whys = [r["why"] for r in blob["unresolved"]]
    assert sum("no depth scale" in w for w in whys) == 1
    # The negative top cannot be true: the layer is listed as read.
    assert any(LISTED in w and "negative" in w for w in whys)
    assert blob["investigation"]["samples"][0]["top"]["value"] == 0.0


def test_a_depth_outside_the_ruler_window_is_kept_and_flagged(doc,
                                                              monkeypatch):
    import report_ingest.visual_scales as vs

    class Ruler:
        step = 1.0

        def value_at(self, y):
            return y / 36.0                    # 22 ft down a letter page

    class Grid:
        rulers = {3: Ruler()}
        warnings = ["a made-up grid warning"]

    seen = {}
    monkeypatch.setattr(vs, "log_grid_for",
                        lambda d, pages, *a, **k: (Grid(), {}))

    def place(stage, record, pages, doc, rulers=None):
        seen.update(stage=stage, pages=pages, rulers=rulers,
                    layers=len(record.layers))
        return {"note": "made-up placement"}

    engine = _engine([{"final": _log([3], top=25.0)}])
    blob = mfr.read_logs(doc, [PageGroup("logs", [3], key="log-3")], engine,
                         report_id=REPORT, place=place)[0]
    layer = blob["investigation"]["layers"][0]
    assert layer["top"]["value"] == 25.0
    assert layer["prov"]["confidence"] == 0.5
    assert any(r["why"].startswith("depth 25") for r in blob["unresolved"])
    assert seen == {"stage": "logs", "pages": [3], "rulers": Grid.rulers,
                    "layers": 1}
    assert blob["placement"] == [{"note": "made-up placement"}]
    assert "a made-up grid warning" in blob["warnings"]


def test_the_added_fields_go_in_on_each_answer_s_own_build(doc):
    engine = _engine([
        {"final": _log([3], project="Made-up Works",
                       header_fields=[{"name": "Made-up Rig",
                                       "value": "R-9"}])},
        {"final": _log([4], top=10.0, sample="S-2", project="Made-up Works",
                       client="Made-up Client")}])
    blob = mfr.read_logs(doc, [PageGroup("logs", [3], key="log-3",
                                         part=(1, 2)),
                               PageGroup("logs", [4], key="log-3",
                                         part=(2, 2))],
                         engine, report_id=REPORT)
    assert len(blob) == 1
    inv = blob[0]["investigation"]
    assert inv["fields"]["project"] == "Made-up Works"
    assert inv["fields"]["Made-up Rig"] == "R-9"
    assert inv["fields"]["client"] == "Made-up Client"
    assert [(s["recovery"]["value"], s["recovery"]["unit"])
            for s in inv["samples"]] == [(14.0, "in")] * 2


def test_a_summary_row_s_bottom_elevation_reaches_its_row(doc):
    engine = _engine([{"final": _lab([7, 8])}])
    blob = mfr.read_lab(doc, [PageGroup("lab", [7, 8], key="lab-7")],
                        engine, report_id=REPORT)[0]
    table = [t for t in blob["tests"] if t["kind"] == "summary_table"][0]
    assert ["elevation bottom (ft)", "412.5"] in [
        list(o) for o in table["result"]["rows"][0]["other"]]


def test_the_model_s_boxes_are_cleared_before_the_build(doc):
    engine = _engine([{"final": _lab([7, 8])}, {"final": _log([3])}])
    lab = mfr.read_lab(doc, [PageGroup("lab", [7, 8], key="lab-7")], engine,
                       report_id=REPORT)[0]
    log = mfr.read_logs(doc, [PageGroup("logs", [3], key="log-3")], engine,
                        report_id=REPORT)[0]
    table = [t for t in lab["tests"] if t["kind"] == "summary_table"][0]
    assert all(p.get("bbox") is None for p in table["prov"])
    assert log["investigation"]["layers"][0]["prov"]["bbox"] is None


def test_a_log_no_answer_came_back_for_has_no_record(doc):
    engine = _engine([{"text": "no tool call"}])
    blob = mfr.read_logs(doc, [PageGroup("logs", [3, 4], key="log-3")],
                         engine, report_id=REPORT)[0]
    assert blob["investigation"] is None and blob["error"]
    assert blob["unresolved"][0]["refused_by"] == "budget"
    assert blob["unresolved"][0]["pages"] == [3, 4]


# ---------------------------------------------------------------------------
# the review's cases (piece 7 review, 2026-10-26)
# ---------------------------------------------------------------------------

def _boxed_calc(rows):
    from report_ingest.calc_reader import CalcReading

    return CalcReading.model_validate({
        "kind": "slope_stability", "program": "MadeUpSlope", "inputs": [],
        "results": [{"name": name, "value": value, "unit": "",
                     "prov": _prov(page, bbox=[1, 2, 3, 4])}
                    for name, value, page in rows]})


def test_two_windows_reading_two_values_on_one_page_keep_both(doc):
    """Fix 1: both windows of a 14-page printout carry its last page; the
    first reads two factors of safety there, the second one. With the boxes
    cleared, only the value read twice is one value."""
    from report_ingest.model import Calculation

    pages = list(range(14, 28))
    assert all(27 in call for call in mfp.calc_calls(PageGroup("calc",
                                                                pages)))
    engine = _engine([
        {"final": _boxed_calc([("Factor of safety", 1.5, 27),
                               ("Factor of safety", 1.3, 27)])},
        {"final": _boxed_calc([("Factor of safety", 1.3, 27)])}])
    got = mfr.read_calcs(doc, [PageGroup("calc", pages, key="calc-14")],
                         engine, report_id=REPORT)[0]
    record = Calculation.model_validate(got["calculation"])
    assert sorted(v.value.value for v in record.results) == [1.3, 1.5]


def test_halves_of_a_printout_are_not_read_as_positions_twice(doc):
    """Fix 2: each half's answer is read against its own call's pages and
    the builder does not read the folded answer again. The second half
    cites one of its pages and one stray page: the stray value keeps its
    value at the printout's first page, as today's builder places one."""
    from report_ingest.model import Calculation

    engine = _engine([
        _cut_off(),
        {"final": _calc([("Applied load", 50.0, 2),
                         ("Footing width", 6.0, 3)])},
        {"final": _calc([("Layer thickness", 4.0, 6),
                         ("Cover depth", 1.0, 1)])}])
    got = mfr.read_calcs(doc, [PageGroup("calc", list(range(2, 10)),
                                         key="calc-2")],
                         engine, report_id=REPORT)[0]
    assert [c["pages"] for c in got["calls"]] == [
        list(range(2, 10)), [2, 3, 4, 5], [6, 7, 8, 9]]
    record = Calculation.model_validate(got["calculation"])
    assert [(v.name, v.prov.page, v.prov.confidence)
            for v in record.inputs] == [
        ("Applied load", 2, 0.9), ("Footing width", 3, 0.9),
        ("Layer thickness", 6, 0.9), ("Cover depth", 2, 0.4)]
    assert not any("a position in the" in v.prov.note
                   for v in record.inputs)
    assert sum(r.get("flag") == "page_numbers"
               for r in got["unresolved"]) == 1
    assert any("put this on page 1" in r["why"] for r in got["unresolved"])


def test_a_call_of_one_page_puts_every_value_on_that_page(doc):
    """Fix 3: a two-page printout cut off is read as two one-page calls;
    whatever page each answer cites, it read its values on its one page
    (the RI-74 rule), with a note."""
    from report_ingest.model import Calculation

    engine = _engine([_cut_off(),
                      {"final": _calc([("Applied load", 50.0, 0)])},
                      {"final": _calc([("Layer thickness", 4.0, 1)])}])
    got = mfr.read_calcs(doc, [PageGroup("calc", [20, 21], key="calc-20")],
                         engine, report_id=REPORT)[0]
    record = Calculation.model_validate(got["calculation"])
    assert [(v.prov.page, v.prov.confidence) for v in record.inputs] == [
        (20, 0.9), (21, 0.9)]
    assert "its call carried page 21 alone" in record.inputs[1].prov.note
    assert not [r for r in got["unresolved"] if "not part of" in r["why"]]
    assert any(w.startswith("page 21: the answer cited page(s) 1")
               for w in got["warnings"])


def test_a_narrative_part_citing_positions_is_put_on_its_own_pages(doc):
    """Fix 4: the second part was sent pages 5 to 7 and cites position 1,
    report page 6, where its quote is printed; another answer mixes its
    call's pages with another and gets the mixed-page note."""
    from report_ingest.narrative_reader import SummaryReading

    groups = [PageGroup("narrative", [0, 1, 2, 3], key="narrative",
                        part=(1, 2)),
              PageGroup("narrative", [5, 6, 7], key="narrative",
                        part=(2, 2))]
    engine = _engine([
        {"final": _narrative("Made-up Bridge", 0)},
        {"final": _narrative_cites("projectNumber", "MU-77",
                                   [(1, "made-up words of page 6")])},
        {"final": SummaryReading(quickSummary="A made-up summary.")}])
    blob = mfr.read_narrative(doc, groups, engine, report_id=REPORT)
    assert blob["general"]["projectNumber"] == "MU-77"
    assert blob["confidence"]["projectNumber"] == blob["confidence"][
        "projectName"]
    assert not [r for r in blob["unresolved"]
                if r.get("what") == "projectNumber"]
    assert any("positions" in w for w in blob["warnings"])

    engine = _engine([
        {"final": _narrative_cites("projectNumber", "MU-77",
                                   [(6, "made-up words of page 6"),
                                    (1, "made-up words of page 1")])}])
    blob = mfr.read_narrative(
        doc, [PageGroup("narrative", [5, 6, 7], key="narrative")], engine,
        report_id=REPORT)
    (note,) = [r for r in blob["unresolved"]
               if r.get("flag") == "page_numbers"]
    assert "6 is among this call's pages and 1 is not" in note["why"]


def test_one_narrative_call_over_pages_apart_reads_positions_in_order(doc):
    """Fix 4: one call over pages 0, 1, 4 and 9; position 3 is page 9."""
    engine = _engine([{"final": _narrative_cites(
        "projectNumber", "MU-77", [(3, "made-up words of page 9")])}])
    blob = mfr.read_narrative(
        doc, [PageGroup("narrative", [0, 1, 4, 9], key="narrative")],
        engine, report_id=REPORT)
    assert blob["general"]["projectNumber"] == "MU-77"
    assert not [r for r in blob["unresolved"]
                if r.get("what") == "projectNumber"]


def test_a_later_part_s_other_header_values_are_listed_as_read(doc):
    """Fixes 5 and 8: the second part names another hole, total depth and
    driller and has its own remarks and rig; the first part's values stand,
    each other value is listed as read, the remarks are both kept, a header
    field printed twice with two values is kept twice, and depths in two
    units are named without being refused."""
    import json

    first = _log([3], remarks="Made-up remark one",
                 drilling={"method": "made-up auger", "driller": "Made-up A"},
                 header_fields=[{"name": "Made-up Rig", "value": "R-1"}])
    second = _log([4], hole="QX-1A", top=10.0, sample="S-2",
                  total_depth=85.0, depth_unit="m",
                  remarks="Backfilled with made-up grout",
                  drilling={"method": "made-up auger",
                            "driller": "Made-up B"},
                  header_fields=[{"name": "Made-up Rig", "value": "R-2"}])
    engine = _engine([{"final": first}, {"final": second}])
    blob = mfr.read_logs(doc, [PageGroup("logs", [3], key="log-3",
                                         part=(1, 2)),
                               PageGroup("logs", [4], key="log-3",
                                         part=(2, 2))],
                         engine, report_id=REPORT)[0]
    inv = blob["investigation"]
    assert inv["investigation_id"] == "QX-1"
    assert inv["total_depth"]["value"] == 40.0
    assert inv["drilling"]["driller"] == "Made-up A"
    assert inv["remarks"] == ("Made-up remark one; Backfilled with made-up "
                              "grout")
    assert inv["fields"]["Made-up Rig"] == "R-1"
    assert inv["fields"]["Made-up Rig (2)"] == "R-2"
    listed = {r["what"]: r for r in blob["unresolved"]
              if LISTED in r["why"] and r["what"].startswith("log QX-1:")}
    assert sorted(listed) == ["log QX-1: drilling driller",
                              "log QX-1: investigation_id",
                              "log QX-1: total_depth"]
    assert json.loads(listed["log QX-1: total_depth"]["read"]) == {
        "value": 85.0, "unit": "m"}
    assert all(r["page"] == 4 and r["refused_by"] == "python"
               for r in listed.values())
    (units,) = [r for r in blob["unresolved"]
                if r["what"] == "log QX-1: depth unit"]
    assert "joined as read" in units["why"] and "refused_by" not in units


def test_a_window_s_sheet_start_by_its_front_is_not_where_it_is_halved():
    """Fix 6: a sheet start in the middle half of the pages cuts there;
    one next to either end does not, and the middle page does."""
    pages = list(range(7, 19))
    assert mfr._halves(pages, [8]) == [pages[:6], pages[6:]]
    assert mfr._halves(pages, [17]) == [pages[:6], pages[6:]]
    assert mfr._halves(pages, [8, 11]) == [pages[:4], pages[4:]]
    assert mfr._halves([7, 8, 9, 10], [10]) == [[7, 8, 9], [10]]


def test_the_additions_entries_name_the_page_as_the_answer_is_read(doc):
    """Fix 7: the answer cites positions 0 and 1 in pages 12 and 13; a
    recovery that cannot be true is listed on page 13, not 1."""
    reading = mf.LogReading.model_validate({
        "investigation_id": "QX-2", "kind": "boring", "depth_unit": "ft",
        "layers": [{"top": 0.0, "bottom": 5.0,
                    "description": "Made-up SILT", "prov": _prov(0)}],
        "samples": [{"sample_id": "S-1", "top": 1.0, "bottom": 2.5,
                     "kind": "spt", "recovery_length": -3.0,
                     "recovery_length_unit": "in", "prov": _prov(1)}]})
    engine = _engine([{"final": reading}])
    blob = mfr.read_logs(doc, [PageGroup("logs", [12, 13], key="log-12")],
                         engine, report_id=REPORT)[0]
    assert blob["investigation"]["samples"][0]["prov"]["page"] == 13
    (row,) = [r for r in blob["unresolved"]
              if r["what"] == "sample S-1 recovery"]
    assert row["page"] == 13
