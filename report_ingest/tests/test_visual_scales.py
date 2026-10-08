"""Visual scales in report ingest (W3 step 8), offline.

The setting is OFF by default and every reader behaves as before the
planlens scales build (a scan's pixel-found layer boundaries are folded
back); ON, a textless scan's labels are read by ONE structured call over
numbered crops, measured layer tops vote (and enter the floor where the scan
has text), a tabulated gradation is set against the curve the sheet plots,
and a digitised sounding is read again by code. Synthetic sheets with every
answer stated (planlens ``visual_scale_fixtures``); a fake engine whose label
replies are the printed labels the crops show.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="the fixtures are drawn with PyMuPDF")
pytest.importorskip("planlens.document.scalefinder")

from planlens.document.loggrid import log_grid  # noqa: E402
from planlens.document.scalefinder import find_scales  # noqa: E402
from planlens.testing.visual_scale_fixtures import (  # noqa: E402
    LogVariant, PlotVariant, _overlap, build_log, build_plot,
)

from report_ingest import visual_scales as VS  # noqa: E402
from report_ingest.log_reader import (  # noqa: E402
    LogReading, ReadLayer, ReadProv, read_log,
)
from report_ingest.model import (  # noqa: E402
    CPTData, CPTPoint, GradationResult, Investigation, LabTest, Provenance,
    Quantity, SievePoint,
)
from report_ingest.tests.fake_engine import FakeEngine  # noqa: E402


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.delenv(VS.ENV, raising=False)


def _labels(fx, boxes):
    """A label reply: the printed text each crop shows, cell by cell."""
    cells = []
    for i, box in enumerate(boxes):
        best = max(fx.labels, key=lambda lb: _overlap(box, lb.box))
        text = best.text if _overlap(box, best.box) > 0.2 else "-"
        cells.append(VS.LabelCell(number=i + 1, text=text))
    return {"final": VS.LabelReading(labels=cells)}


def _pending_boxes(doc):
    out = []
    for sc in find_scales(doc, 0).needing_values():
        out.extend(sc.label_boxes)
    return out


@pytest.fixture(scope="module")
def textless():
    return build_log(LogVariant("ri_scan", skew_deg=0.4, seed=101))


@pytest.fixture(scope="module")
def ocr_scan():
    return build_log(LogVariant("ri_ocr", text_source="ocr", skew_deg=0.5,
                                seed=15))


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

def test_the_setting_is_off_by_default(monkeypatch):
    assert not VS.enabled()
    monkeypatch.setenv(VS.ENV, "1")
    assert VS.enabled()
    with VS.use_visual_scales(False):
        assert not VS.enabled()
    monkeypatch.delenv(VS.ENV)
    with VS.use_visual_scales():
        assert VS.enabled()
    assert not VS.enabled()


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def test_off_a_scan_s_pixel_layer_boundaries_are_folded_back(ocr_scan):
    raw = log_grid(ocr_scan.open(), [0])
    assert any((ly.evidence or {}).get("found_in") == "pixels"
               for ly in raw.layers)
    grid, info = VS.log_grid_for(ocr_scan.open(), [0])
    assert info == {}
    assert not any((ly.evidence or {}).get("found_in") == "pixels"
                   for ly in grid.layers)
    # the words of the folded layers are kept, in the layer above
    assert " ".join(ly.description for ly in grid.layers) == \
        " ".join(ly.description for ly in raw.layers).strip()


def test_on_a_textless_scan_s_labels_are_read_in_one_call(textless):
    doc = textless.open()
    boxes = log_grid(doc, [0]).needs_values[0]["labels"]
    engine = FakeEngine([_labels(textless, boxes)])
    with VS.use_visual_scales():
        grid, info = VS.log_grid_for(textless.open(), [0], engine)
    assert engine.n_calls == 1
    call = engine.calls[0]
    assert call["n_images"] == 1 and call["output_format"] is VS.LabelReading
    assert not grid.needs_values and grid.rulers
    tops = [ly for ly in grid.layers
            if (ly.evidence or {}).get("found_in") == "pixels"]
    assert len(tops) == len(textless.readings)
    for rd, ly in zip(textless.readings, tops):
        assert abs(ly.top - rd.value) <= ly.plus_minus
    assert info["label_calls"] == 1 and info["pixel_layers"] == len(tops)
    assert info["labels_read"]["0"].startswith("1.0, 2.0")


def _model_layers(tops, unit="m"):
    return [ReadLayer(top=t, bottom=None, description=f"stratum {i}",
                      prov=ReadProv(page=0, bbox=None, from_image=True))
            for i, t in enumerate(tops)]


def test_off_the_reader_spends_no_label_call(textless):
    engine = FakeEngine([{"final": LogReading(
        investigation_id="B-1", kind="boring", depth_unit="m",
        units_known=True, layers=[], pages_read=[0])}])
    result = read_log(textless.open(), [0], engine, report_id="R")
    assert engine.n_calls == 1
    assert result.visual_scales == {}
    assert "visual_scales" not in result.to_dict()


def test_on_measured_tops_vote_on_a_textless_scan(textless):
    doc = textless.open()
    boxes = log_grid(doc, [0]).needs_values[0]["labels"]
    truth = [rd.value for rd in textless.readings]
    by_eye = [0.0, truth[0], truth[1] + 0.3, truth[2], truth[3]]
    engine = FakeEngine([
        _labels(textless, boxes),
        {"final": LogReading(investigation_id="B-1", kind="boring",
                             depth_unit="m", units_known=True,
                             layers=_model_layers(by_eye), pages_read=[0])}])
    with VS.use_visual_scales():
        result = read_log(textless.open(), [0], engine, report_id="R")
    assert engine.n_calls == 2
    assert result.cost["calls"] == 2
    votes = result.visual_scales["layer_votes"]
    assert len(votes) == len(truth)
    bad = [v for v in votes if v["agreement"] == "disagree"]
    assert len(bad) == 1 and abs(bad[0]["geometry_top"] - truth[1]) < 0.05
    assert sum(v["agreement"] == "agree" for v in votes) == 3
    qa = VS.layer_votes_qa(votes, "investigations[B-1].layers", "m", [0])
    assert [e.kind for e in qa] == ["disagreement", "note"]
    assert "stratum line measured on the scan" in qa[0].detail
    assert result.to_dict()["visual_scales"]["layer_votes"]


def test_on_measured_tops_enter_the_floor_where_the_scan_has_text(ocr_scan):
    reading = {"final": LogReading(investigation_id="B-1", kind="boring",
                                   depth_unit="m", units_known=True,
                                   layers=[], pages_read=[0])}
    with VS.use_visual_scales():
        on = read_log(ocr_scan.open(), [0], FakeEngine([reading]),
                      report_id="R")
    off = read_log(ocr_scan.open(), [0], FakeEngine([reading]),
                   report_id="R")
    measured = [ly for ly in on.floor.layers
                if "measured from the stratum line" in ly.prov.note]
    assert len(measured) == len(ocr_scan.readings)
    for rd, ly in zip(ocr_scan.readings, measured):
        assert abs(ly.top.value - rd.value) < 0.05
        assert "+/-" in ly.prov.note
    assert not any("measured from the stratum line" in ly.prov.note
                   for ly in off.floor.layers)
    assert len(off.floor.layers) < len(on.floor.layers)


# ---------------------------------------------------------------------------
# Lab: plot against table
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grading():
    return build_plot(PlotVariant("ri_grad", "grading", raster=True,
                                  skew_deg=0.3, seed=52))


def _table(fx, off_by=None):
    pts = [r for r in fx.readings if r.kind == "point"]
    sieves = []
    for i, r in enumerate(pts):
        pct = r.values["plot.y"] + (off_by[1] if off_by and off_by[0] == i
                                    else 0.0)
        sieves.append(SievePoint(
            percent_passing=max(0.0, min(100.0, pct)),
            size=Quantity(value=r.values["plot.x"], unit="mm",
                          prov=Provenance(page=0)),
            sieve=f"{r.values['plot.x']:g} mm"))
    return LabTest(kind="gradation",
                   result=GradationResult(percent_passing=sieves), pages=[0])


def test_the_plot_is_set_against_the_table(grading):
    doc = grading.open()
    engine = FakeEngine([_labels(grading, _pending_boxes(doc))])
    check = VS.plot_check(grading.open(), [0], [_table(grading, (3, 12.0))],
                          engine)
    assert check["status"] == "checked" and check["label_calls"] == 1
    n = len([r for r in grading.readings if r.kind == "point"])
    assert check["table_points"] == n
    assert check["agree"] == n - 1
    (row,) = check["disagree"]
    assert row["gap"] == pytest.approx(12.0, abs=1.0)
    qa = VS.plot_check_qa(check, [0])
    assert [e.kind for e in qa] == ["plot_vs_table", "note"]
    assert "keeps the table" in qa[0].detail


def test_a_sheet_with_no_tabulated_grading_is_not_checked(grading):
    assert VS.plot_check(grading.open(), [0], [], None) is None


def test_the_lab_reader_reports_the_check_only_with_the_setting(
        grading, monkeypatch):
    from report_ingest import lab_reader
    from report_ingest.lab_reader import LabSheetReading, read_lab_sheet

    real = lab_reader.floor_from_tables

    def with_table(doc, pages, report_id=""):
        floor = real(doc, pages, report_id)
        floor.tests = [_table(grading)]
        return floor

    monkeypatch.setattr(lab_reader, "floor_from_tables", with_table)
    doc = grading.open()
    boxes = _pending_boxes(doc)
    off = read_lab_sheet(grading.open(), [0], FakeEngine(
        [{"final": LabSheetReading(tests=[])}]), report_id="R")
    assert off.plot_check is None and "plot_check" not in off.to_dict()
    with VS.use_visual_scales():
        on = read_lab_sheet(grading.open(), [0], FakeEngine(
            [{"final": LabSheetReading(tests=[])}, _labels(grading, boxes)]),
            report_id="R")
    assert on.plot_check["status"] == "checked"
    assert on.plot_check["agree"] == on.plot_check["table_points"]
    assert on.to_dict()["plot_check"]["disagree"] == []


# ---------------------------------------------------------------------------
# Soundings: a digitised trace read again
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trace():
    return build_plot(PlotVariant("ri_cpt", "reversed_y", raster=True,
                                  skew_deg=-0.6, seed=53))


def _sounding(fx, wrong_at=None):
    pts = []
    for i, r in enumerate(r for r in fx.readings if r.kind == "curve"):
        depth = list(r.at.values())[0]
        qc = r.value * (1.3 if i == wrong_at else 1.0)
        pts.append(CPTPoint(depth=Quantity(value=depth, unit="m"),
                            qc=Quantity(value=qc, unit="MPa")))
    return Investigation(kind="cpt", pages=[0],
                         cpt=CPTData(points=pts, digitised=True))


def test_a_digitised_trace_is_read_again_by_code(trace):
    doc = trace.open()
    engine = FakeEngine([_labels(trace, _pending_boxes(doc))])
    check = VS.trace_check(trace.open(), [0], _sounding(trace, wrong_at=2),
                           engine)
    row = check["channels"]["qc"]
    assert check["status"] == "checked"
    assert row["compared"] >= 4
    assert len(row["disagree"]) == 1
    assert row["disagree"][0]["depth"] == pytest.approx(4.5)
    assert row["agree"] == row["compared"] - 1
    qa = VS.trace_check_qa(check, [0])
    assert [e.kind for e in qa] == ["disagreement", "note"]
    assert "keeps the model's series" in qa[0].detail


def test_a_tabulated_series_is_not_read_again(trace):
    inv = _sounding(trace)
    inv.cpt.digitised = False
    assert VS.trace_check(trace.open(), [0], inv, None) is None


def test_the_sounding_reader_reports_the_check_only_with_the_setting(
        tmp_path):
    from planlens.document import open_document
    from report_ingest.sounding_reader import (
        ReadPoint, SoundingReading, read_sounding)
    from report_ingest.tests.sounding_fixtures import build_plotted_cpt
    gt = build_plotted_cpt()
    path = tmp_path / "cpt.pdf"
    path.write_bytes(gt.pdf)

    def reading():
        return {"final": SoundingReading(
            investigation_id="CPT-7", kind="cpt", depth_unit="m",
            qc_unit="MPa", digitised=True, step=0.5,
            points=[ReadPoint(depth=0.5, qc=8.0),
                    ReadPoint(depth=1.0, qc=12.0)])}

    off = read_sounding(open_document(str(path)), [0],
                        FakeEngine([reading()]), kind="cpt", report_id="R")
    assert off.code_trace is None and "code_trace" not in off.to_dict()
    with VS.use_visual_scales():
        on = read_sounding(open_document(str(path)), [0],
                           FakeEngine([reading()]), kind="cpt",
                           report_id="R")
    assert on.code_trace is not None
    assert on.to_dict()["code_trace"]["status"] in ("checked", "no_plot")


# ---------------------------------------------------------------------------
# The one zoom_plot
# ---------------------------------------------------------------------------

def test_the_three_readers_share_one_zoom_plot(grading):
    from report_ingest import calc_reader, lab_reader, sounding_reader
    doc = grading.open()
    for mod, noun in ((lab_reader, "sheet"), (calc_reader, "printout"),
                      (sounding_reader, "sounding")):
        tools = mod._Tools(doc, [0])
        content, is_error = tools.run("zoom_plot", {"page": 3,
                                                    "bbox": [0, 0, 9, 9]})
        assert is_error and f"not part of this {noun}" in content
        content, is_error = tools.run("zoom_plot", {
            "page": 0, "bbox": [100, 290, 530, 630], "why": "the curve"})
        assert not is_error
        assert [b["type"] for b in content] == ["text", "image"]
        assert tools.zooms[-1]["why"] == "the curve"


def test_with_the_setting_a_zoom_carries_code_s_reading_of_the_plot():
    fx = build_plot(PlotVariant("ri_grad_v", "grading"))   # vector: text
    from report_ingest import lab_reader
    tools = lab_reader._Tools(fx.open(), [0])
    with VS.use_visual_scales():
        content, is_error = tools.run("zoom_plot", {
            "page": 0, "bbox": [100, 290, 530, 630]})
    assert not is_error
    texts = [b["text"] for b in content if b["type"] == "text"]
    assert any("CODE'S READING OF THIS CROP" in t for t in texts)
    assert any("plotted marker(s) read by code" in t for t in texts)
