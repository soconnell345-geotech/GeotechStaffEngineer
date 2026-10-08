"""The visual-scale tools on the agents (W3 steps 6-7), offline.

``measure`` and ``log_grid`` on the geotech page and the Document Review
legacy and lean builds (NOT the minimal one); a scan's label values read by
ONE vision side call over numbered crops (the fake vision call reads them as
a model would: one line per numbered cell); a look's ``view`` + ``image_box``
searched within that view's location error; and ``read_reference_figure``'s
second voter, code's value beside the vision value, flagged at about 3x
code's +/-. Synthetic fixtures with every answer stated (planlens'
``visual_scale_fixtures``); no network, no model.
"""

from __future__ import annotations

import json
import struct

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from funhouse_agent import document_tools, scale_labels, vision_view  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402

if not document_tools.has_tool("measure"):            # pragma: no cover
    pytest.skip("installed planlens predates measure", allow_module_level=True)

from planlens.testing.visual_scale_fixtures import (  # noqa: E402
    LogVariant, _overlap, build_chart_family, build_log,
)


# ---------------------------------------------------------------------------
# A fake vision engine that READS numbered label crops
# ---------------------------------------------------------------------------

class LabelReader:
    """A vision engine whose label reads answer from the fixture.

    The real contact sheet is drawn and sent; the boxes it was drawn from
    are captured where the sheet is made, and the reply is one ``#N | text``
    line per numbered cell, the text of the printed label each crop shows —
    what a model reading the crops would write. ``misread`` replaces cell
    texts; ``chart`` answers a chart read-off prompt instead.
    """

    accepts_jpeg = False

    def __init__(self, fx, misread=None, chart=None):
        self.fx = fx
        self.misread = dict(misread or {})
        self.chart = chart
        self.calls = []
        self.sheets = []

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        self.calls.append(prompt)
        if "Each numbered cell" in prompt:
            items = self.sheets[-1]
            lines = []
            for i, (_page, box) in enumerate(items):
                best = max(self.fx.labels, key=lambda lb: _overlap(box, lb.box))
                text = best.text if _overlap(box, best.box) > 0.2 else "-"
                lines.append(f"#{i + 1} | {self.misread.get(i, text)}")
            return "\n".join(lines)
        if self.chart is not None:
            return self.chart(image, prompt)
        return "seen"


@pytest.fixture(autouse=True)
def _capture_sheets(monkeypatch):
    """Every label sheet drawn is real; its boxes go to the engine."""
    real = scale_labels.label_sheets
    engines = []

    def spy(doc, items, *a, **kw):
        for e in engines:
            e.sheets.append(list(items))
        return real(doc, items, *a, **kw)

    monkeypatch.setattr(scale_labels, "label_sheets", spy)
    scale_labels.clear_cache()
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    yield engines
    scale_labels.clear_cache()


def _engine(engines, fx, **kw):
    e = LabelReader(fx, **kw)
    engines.append(e)
    return e


def _tools(engine, pdf):
    return {t.name: t for t in make_vision_tools(
        engine=engine, attachments={"log.pdf": pdf})}


def _call(tool, **args):
    return json.loads(tool.invoke(args))


@pytest.fixture(scope="module")
def scan_log():
    return build_log(LogVariant("app_scan", skew_deg=0.4, seed=91))


# ---------------------------------------------------------------------------
# Where the tools are
# ---------------------------------------------------------------------------

def test_both_tools_are_on_the_geotech_surface():
    names = {t.name for t in make_vision_tools(engine=None)}
    assert {"measure", "log_grid"} <= names
    assert {"measure", "log_grid"} <= set(document_tools.document_tool_names())


def test_an_older_planlens_hides_them(monkeypatch):
    from planlens.tools import specs as planlens_specs
    kept = [s for s in planlens_specs.TOOL_SPECS
            if s["name"] not in ("measure", "log_grid")]
    monkeypatch.setattr(planlens_specs, "TOOL_SPECS", kept)
    names = {t.name for t in make_vision_tools(engine=None)}
    assert not names & {"measure", "log_grid"}


def _review_tools(monkeypatch, arm):
    from funhouse_agent import review_flags
    from funhouse_agent.deep.agent import build_deep_agent
    from funhouse_agent.deep.tests.test_review_harness_offline import (
        FakeEngine, _model, _review_kwargs, _tools_of)
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    for k, v in review_flags.ARMS[arm].items():
        monkeypatch.setenv(k, v)
    agent = build_deep_agent(_model([]), engine=FakeEngine(),
                             attachments={}, **_review_kwargs())
    return set(_tools_of(agent))


def test_legacy_and_lean_review_builds_have_measure(monkeypatch):
    assert {"measure", "log_grid"} <= _review_tools(monkeypatch, "baseline")
    assert {"measure", "log_grid"} <= _review_tools(monkeypatch, "lean")
    from funhouse_agent.deep.review_agent import READER_TOOLS
    assert {"measure", "log_grid"} <= set(READER_TOOLS)


def test_the_minimal_build_does_not_get_measure(monkeypatch):
    """Owner decision 1 (2026-10-08): the minimal build keeps its short
    tool list."""
    names = _review_tools(monkeypatch, "minimal")
    assert "measure" not in names and "log_grid" not in names


def test_the_tools_describe_themselves_and_stay_small():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    tools = {t.name: t for t in make_vision_tools(engine=None)}
    m = tools["measure"]
    assert "view" in m.description and "image_box" in m.description
    assert "by eye is approximate" in m.description
    assert len(m.description) < 1700
    schema = json.dumps(convert_to_openai_tool(m))
    assert len(schema) < 3600, len(schema)
    assert "$ref" not in schema
    props = convert_to_openai_tool(m)["function"]["parameters"]["properties"]
    assert {"source", "page", "kind", "bbox", "view", "image_box", "at",
            "to", "scale", "side"} <= set(props)
    assert "values" not in props            # the app reads the labels itself
    lg = tools["log_grid"].description
    assert "document_roles" not in lg       # not a tool on this surface
    assert "read for you" in lg
    assert len(json.dumps(convert_to_openai_tool(tools["log_grid"]))) < 3600


def test_no_prompt_mentions_measure():
    """Tools describe themselves (owner's rule): no system prompt names it."""
    from funhouse_agent.deep import prompt as P
    texts = [P.build_document_review_prompt(lean=True),
             P.build_document_review_prompt(lean=False)]
    for t in texts:
        assert "measure(" not in t and "`measure`" not in t
        assert "log_grid" not in t


# ---------------------------------------------------------------------------
# The label sheet and its reply
# ---------------------------------------------------------------------------

def test_a_label_sheet_numbers_every_crop(scan_log):
    from planlens.document.scalefinder import find_scales
    doc = scan_log.open()
    boxes = find_scales(doc, 0).needing_values()[0].label_boxes
    sheets = scale_labels.label_sheets(doc, [(0, b) for b in boxes])
    assert len(sheets) == 1
    png, idx, (w, h) = sheets[0]
    assert idx == list(range(len(boxes)))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (w, h)


@pytest.mark.parametrize("raw, want", [
    ("4.0", "4.0"), ("4.0 m", "4.0"), ("10^-3", "0.001"),
    ("10⁻³", "0.001"), ("2x10^2", "200"), ("1E-3", "0.001"),
    ("100", "100"), ("1000", "1000"), ("12+50", "12+50"), ("-5.0", "-5.0"),
    ("-", None), ("?", None), ("abc", None), ("", None),
])
def test_a_reading_is_kept_as_printed(raw, want):
    assert scale_labels.normalise_label(raw) == want


def test_the_reply_is_one_text_per_numbered_cell():
    got = scale_labels.parse_label_reply(
        "Here you go:\n#1 | 1.0\n#2: 2.0\n3 | -\n#5 | 5.0 m", [1, 2, 3, 4, 5])
    assert got == {1: "1.0", 2: "2.0", 3: None, 4: None, 5: "5.0"}


# ---------------------------------------------------------------------------
# measure on a scan with no text
# ---------------------------------------------------------------------------

def test_a_scan_with_no_text_is_measured_after_one_label_read(
        _capture_sheets, scan_log):
    eng = _engine(_capture_sheets, scan_log)
    tools = _tools(eng, scan_log.pdf)
    rd = scan_log.readings[1]
    out = _call(tools["measure"], source="log.pdf", page=0,
                bbox=list(rd.box_pt), kind="line")
    assert len(eng.calls) == 1 and "Each numbered cell" in eng.calls[0]
    v = out["value"]
    assert abs(v["depth"] - rd.value) <= v["plus_minus"]
    # values are the labels AS PRINTED ("4.0"): never finer than the print
    assert len(v["display"].split(" ")[0].split(".")[1]) <= 1
    assert out["labels_read"]["values"]["p0.depth"].startswith("1.0, 2.0")
    assert "vision call" in out["labels_read"]["read_by"]
    assert out["pdf_page"] == 1
    # a second measurement on the page spends no further call
    rd2 = scan_log.readings[2]
    out2 = _call(tools["measure"], source="log.pdf", page=0,
                 bbox=list(rd2.box_pt), kind="line")
    assert len(eng.calls) == 1
    assert abs(out2["value"]["depth"] - rd2.value) <= \
        out2["value"]["plus_minus"]


def test_a_zoom_s_view_and_image_box_snap_within_its_error(
        _capture_sheets, scan_log):
    eng = _engine(_capture_sheets, scan_log)
    tools = _tools(eng, scan_log.pdf)
    rd = scan_log.readings[2]
    cx = (rd.box_pt[0] + rd.box_pt[2]) / 2.0
    cy = (rd.box_pt[1] + rd.box_pt[3]) / 2.0
    view = [cx - 75.0, cy - 60.0, cx + 75.0, cy + 60.0]       # a zoom
    w, h = view[2] - view[0], view[3] - view[1]
    box = [(rd.box_pt[0] - view[0]) / w * 999, (rd.box_pt[1] - view[1]) / h
           * 999, (rd.box_pt[2] - view[0]) / w * 999,
           (rd.box_pt[3] - view[1]) / h * 999]
    out = _call(tools["measure"], source="log.pdf", page=0, view=view,
                image_box=box, kind="line")
    ex, ey = vision_view.location_error(view)
    assert out["pad_pt"] == pytest.approx(max(ex, ey), abs=0.01)
    assert abs(out["value"]["depth"] - rd.value) <= out["value"]["plus_minus"]
    assert out["located_from"]["view"] == [round(x, 1) for x in view]
    assert "location error" in out["located_from"]["note"]


def test_a_whole_sheet_view_lists_instead_of_choosing(_capture_sheets,
                                                      scan_log):
    eng = _engine(_capture_sheets, scan_log)
    tools = _tools(eng, scan_log.pdf)
    rd = scan_log.readings[1]
    page = fitz.open(stream=scan_log.pdf, filetype="pdf")[0]
    view = [0.0, 0.0, page.rect.width, page.rect.height]
    box = [rd.box_pt[0] / view[2] * 999, rd.box_pt[1] / view[3] * 999,
           rd.box_pt[2] / view[2] * 999, rd.box_pt[3] / view[3] * 999]
    out = _call(tools["measure"], source="log.pdf", page=0, view=view,
                image_box=box, kind="line")
    assert out["value"] is None and out.get("ambiguous")
    assert out["alternatives"]
    assert "lists candidates" in out["located_from"]["note"]


def test_one_misread_label_is_left_out_and_the_rest_still_fit(
        _capture_sheets, scan_log):
    eng = _engine(_capture_sheets, scan_log, misread={4: "8.0"})
    tools = _tools(eng, scan_log.pdf)
    rd = scan_log.readings[1]
    out = _call(tools["measure"], source="log.pdf", page=0,
                bbox=list(rd.box_pt), kind="line")
    assert abs(out["value"]["depth"] - rd.value) <= out["value"]["plus_minus"]
    said = json.dumps(out)
    assert "left out" in said or "dropped" in said


def test_with_no_vision_engine_the_scale_waits_and_says_so(scan_log):
    tools = _tools(None, scan_log.pdf)
    rd = scan_log.readings[1]
    out = _call(tools["measure"], source="log.pdf", page=0,
                bbox=list(rd.box_pt), kind="line")
    assert out["value"] is None and "needs_values" in json.dumps(out)
    assert "no vision engine" in out["labels_read"]["note"]


def test_a_handle_from_open_document_works_too(_capture_sheets, scan_log):
    eng = _engine(_capture_sheets, scan_log)
    tools = _tools(eng, scan_log.pdf)
    handle = _call(tools["open_document"], source="log.pdf")["handle"]
    out = _call(tools["measure"], source=handle, page=0)
    assert out["scales"]["frames"][0]["id"] == "p0.depth"
    assert "labels_read" in out                  # the listing read them


def test_log_grid_reads_a_textless_scan_s_labels(_capture_sheets, scan_log):
    eng = _engine(_capture_sheets, scan_log)
    tools = _tools(eng, scan_log.pdf)
    handle = _call(tools["open_document"], source="log.pdf")["handle"]
    out = _call(tools["log_grid"], handle=handle, pages="0", rows=False)
    assert len(eng.calls) == 1
    assert "needs_values" not in out
    tops = [ly for ly in out["layers"] if ly.get("source") == "stratum_rule"]
    assert len(tops) == len(scan_log.readings)
    for rd, ly in zip(scan_log.readings, tops):
        assert abs(ly["top"] - rd.value) <= max(ly["plus_minus"], 0.01)
    assert out["labels_read"]["values"]["page 0"].startswith("1.0")


# ---------------------------------------------------------------------------
# The precision line (owner decision 7)
# ---------------------------------------------------------------------------

def test_the_precision_line_says_a_box_is_not_a_measurement():
    for view in ([0, 0, 1224, 792], [0, 0, 200, 150]):
        note = vision_view.precision_note(view)
        assert "not a measurement" in note
        assert "measure" not in note.replace("measurement", "")


# ---------------------------------------------------------------------------
# read_reference_figure's second voter (step 7)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chart():
    return build_chart_family("semilog", raster=True, seed=54)


@pytest.fixture
def figure(monkeypatch, tmp_path, chart):
    import geotech_references._figures_db as FDB
    path = tmp_path / "chart.pdf"
    path.write_bytes(chart.pdf)
    monkeypatch.setattr(FDB, "figure_get", lambda ref, num: {
        "figure_number": num, "caption": "design chart",
        "pdf_path": str(path), "pdf_page_index": 0})
    monkeypatch.setattr(FDB, "resolve_pdf", lambda ref, num: (str(path), 0))
    from funhouse_agent import chart_reading
    yield str(path)
    chart_reading.close_documents()


def _png_size(img):
    return struct.unpack(">II", img[16:24])


def _read_line(img, rd, value, curve="0.4", w_pt=612.0, h_pt=792.0):
    w, h = _png_size(img)
    cx, cy = rd.at_pt
    px = [(cx - 3) * w / w_pt, (cy - 3) * h / h_pt, (cx + 3) * w / w_pt,
          (cy + 3) * h / h_pt]
    axis, at = next(iter(rd.at.items()))
    return (f"READ | x = {at:g} | curve = {curve} | value = {value:.4f} | "
            f"px=[{px[0]:.0f}, {px[1]:.0f}, {px[2]:.0f}, {px[3]:.0f}]")


def _read_figure(engine):
    from funhouse_agent.vision_tools import dispatch_extended_tool
    return json.loads(dispatch_extended_tool(
        tool_name="read_reference_figure",
        arguments={"reference": "x", "figure_number": "7-1",
                   "prompt": "read y at x = 20 on the 0.4 curve"},
        engine=engine, attachments={}))


def _curves(chart):
    return [r for r in chart.readings if r.kind == "curve"]


def test_code_reads_the_chart_and_agrees(_capture_sheets, chart, figure):
    rd = _curves(chart)[1]

    def answer(img, prompt):
        assert "READ |" in prompt and "px=" in prompt
        return f"About {rd.value:.3f}.\n" + _read_line(img, rd, rd.value)

    eng = _engine(_capture_sheets, chart, chart=answer)
    out = _read_figure(eng)
    block = out["code_reading"]
    row = block["readings"][0]
    assert row["status"] == "measured" and row["agreement"] == "agree"
    assert abs(row["code"]["value"] - rd.value) <= row["code"]["plus_minus"]
    assert "code_reading" in out["note"] and "3x" in out["note"]
    assert block["labels_read"].startswith("1 vision call")


def test_a_disagreement_is_flagged_with_both_values(_capture_sheets, chart,
                                                    figure):
    rd = _curves(chart)[1]
    wrong = rd.value * 1.2

    eng = _engine(_capture_sheets, chart,
                  chart=lambda img, p: _read_line(img, rd, wrong))
    row = _read_figure(eng)["code_reading"]["readings"][0]
    assert row["agreement"] == "disagree"
    assert row["vision"] == pytest.approx(wrong, rel=1e-3)
    assert abs(row["code"]["value"] - rd.value) <= row["code"]["plus_minus"]
    assert row["gap_in_code_plus_minus"] > 3.0 and "flag" in row


def test_an_interpolated_answer_is_read_between_two_curves(
        _capture_sheets, chart, figure):
    at20 = [r for r in _curves(chart) if abs(list(r.at.values())[0] - 20.0)
            < 1e-9]
    lo, hi = at20[0], at20[1]                     # the 0.2 and 0.4 curves
    want = (lo.value + hi.value) / 2.0

    def answer(img, prompt):
        return "\n".join([_read_line(img, lo, lo.value, curve="0.2"),
                          _read_line(img, hi, hi.value, curve="0.4"),
                          f"ANSWER | curve = 0.3 | value = {want:.4f}"])

    eng = _engine(_capture_sheets, chart, chart=answer)
    ans = _read_figure(eng)["code_reading"]["answer"]
    assert ans["code"]["between"] == ["0.2", "0.4"]
    assert ans["code"]["value"] == pytest.approx(want, abs=0.01)
    assert ans["agreement"] == "agree"


def test_no_read_line_leaves_the_vision_value_alone(_capture_sheets, chart,
                                                    figure):
    eng = _engine(_capture_sheets, chart, chart=lambda img, p: "about 0.2")
    out = _read_figure(eng)
    assert out["code_reading"]["status"] == "no_read_lines"
    assert len(eng.calls) == 1                    # no label read either


def test_the_voter_can_be_switched_off(_capture_sheets, chart, figure,
                                       monkeypatch):
    from funhouse_agent import chart_reading
    monkeypatch.setenv(chart_reading.VOTER_ENV, "0")
    rd = _curves(chart)[1]
    eng = _engine(_capture_sheets, chart,
                  chart=lambda img, p: _read_line(img, rd, rd.value))
    out = _read_figure(eng)
    assert "code_reading" not in out
    assert "READ |" not in eng.calls[0]
