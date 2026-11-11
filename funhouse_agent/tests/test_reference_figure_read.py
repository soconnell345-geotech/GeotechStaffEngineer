"""A chart read says what it looked at, what is printed and what is its
own (2026-11-10 reference-library review, item 6).

Every read first confirms the figure (number, caption, axis quantities and
units), not only a page located by estimate; the vision model reports only
what is printed and marks the rest as its own; the result gives the 1-based
PDF page beside the 0-based index; a digitized lookup of the same figure is
named, not called; and a code measurement that gave no value is said at the
top. Made-up charts and values throughout.
"""

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from funhouse_agent import chart_reading  # noqa: E402
from funhouse_agent.reference_vision import (  # noqa: E402
    _figure_tokens, same_figure_methods)


class _Reader:
    """A vision engine that answers a chart read with no READ line."""

    accepts_jpeg = False

    def __init__(self, answer="About 0.2 (made up)."):
        self.answer = answer
        self.prompts = []

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture
def figure(monkeypatch, tmp_path):
    """A one-page made-up chart PDF, catalogued as ``rec`` says."""
    import geotech_references._figures_db as FDB
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Figure 9-9 A made-up chart")
    path = tmp_path / "chart.pdf"
    path.write_bytes(doc.tobytes())
    rec = {"figure_number": "9-9", "caption": "a made-up chart",
           "pdf_path": str(path), "pdf_page_index": 0,
           "page_estimated": False}
    monkeypatch.setattr(FDB, "figure_get", lambda ref, num: dict(rec))
    monkeypatch.setattr(FDB, "resolve_pdf", lambda ref, num: (str(path), 0))
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    yield rec
    chart_reading.close_documents()


def _read(engine, reference="made_up"):
    from funhouse_agent.vision_tools import dispatch_extended_tool
    return json.loads(dispatch_extended_tool(
        tool_name="read_reference_figure",
        arguments={"reference": reference, "figure_number": "9-9",
                   "prompt": "read y at x = 20"},
        engine=engine, attachments={}))


@pytest.mark.parametrize("estimated", [False, True])
def test_every_read_confirms_the_figure_and_its_axes(figure, monkeypatch,
                                                     estimated):
    import geotech_references._figures_db as FDB
    monkeypatch.setattr(FDB, "figure_get", lambda ref, num: {
        **figure, "page_estimated": estimated})
    eng = _Reader()
    _read(eng)
    prompt = eng.prompts[0]
    assert "FIRST confirm you are looking at Figure 9-9" in prompt
    assert "figure number and caption as printed" in prompt
    assert "quantity and unit printed on each axis" in prompt
    assert ("adjacent page should be tried" in prompt) is estimated


def test_the_reader_marks_what_is_not_printed(figure):
    eng = _Reader()
    _read(eng)
    prompt = eng.prompts[0]
    assert "Report only what is printed on this page" in prompt
    assert "a unit conversion" in prompt and "a value from memory" in prompt


def test_the_result_gives_the_1_based_pdf_page(figure):
    out = _read(_Reader())
    assert out["pdf_page"] == 1 and out["pdf_page_index"] == 0
    keys = list(out)
    assert keys.index("pdf_page") + 1 == keys.index("pdf_page_index")
    # review F5: no bare, unlabelled 0-based "page" beside them
    assert "page" not in out


def test_no_code_value_is_said_at_the_top(figure):
    out = _read(_Reader())
    assert out["code_reading"]["status"] == "no_read_lines"
    assert out["second_reading"].startswith(
        "Code's measurement of the chart gave no value (the vision answer "
        "gave no READ line")
    assert "vision read-off alone" in out["second_reading"]


@pytest.mark.parametrize("answer", [
    "NOT ON THIS PAGE: the page shows Table 9-8, a made-up table.",
    "**Not on this page:** a made-up table.",
    "Figure 9-9 is not on this page; it shows a made-up table instead."])
def test_a_figure_not_on_the_page_is_not_called_a_read_off(figure, answer):
    """Review F9: where the reader said the figure is not on the page,
    there is no "vision read-off alone" to speak of."""
    eng = _Reader(answer)
    out = _read(eng)
    assert 'begin your answer with "NOT ON THIS PAGE:"' in eng.prompts[0]
    assert out["second_reading"].startswith(
        "The vision reader said Figure 9-9 is not on this page (PDF page 1)")
    assert "read-off alone" not in out["second_reading"]


def test_another_thing_not_on_the_page_is_still_a_read(figure):
    out = _read(_Reader("Figure 9-9, a made-up chart. The legend is not on "
                        "this page. About 0.2."))
    assert "read-off alone" in out["second_reading"]


def test_a_measured_read_has_no_such_line(figure, monkeypatch):
    monkeypatch.setattr(chart_reading, "code_reading",
                        lambda *a, **k: {"status": "measured",
                                         "readings": []})
    out = _read(_Reader())
    assert "second_reading" not in out


def test_a_row_s_reason_is_used(figure, monkeypatch):
    monkeypatch.setattr(chart_reading, "code_reading", lambda *a, **k: {
        "status": "no_scale",
        "readings": [{"status": "no_scale",
                      "note": "no fitted scale answers here"}]})
    out = _read(_Reader())
    assert "(no fitted scale answers here)" in out["second_reading"]


def test_the_library_s_lookup_for_the_figure_is_named(figure, monkeypatch):
    import geotech_references._figures_db as FDB
    monkeypatch.setattr(FDB, "figure_get", lambda ref, num: {
        **figure, "reference": "gec_12", "figure_number": "7-15",
        "reference_title": "a made-up title"})
    out = _read(_Reader(), reference="gec12")
    look = out["digitized_lookup"]
    assert look["module"] == "gec12"
    assert look["methods"] == ["figure_7_15_limiting_toe_resistance"]
    assert "report both values" in look["note"]


def test_a_figure_with_no_lookup_has_no_pointer(figure):
    assert "digitized_lookup" not in _read(_Reader())


@pytest.mark.parametrize("number,tokens", [
    ("7-15", ["7_15"]), ("4.3", ["4_3"]), ("5-18(a)", ["5_18_a", "5_18a"]),
    # review F4: a lettered appendix figure is written both ways
    ("E-1", ["e_1", "e1"]), ("", [])])
def test_figure_tokens(number, tokens):
    assert _figure_tokens(number) == tokens


def test_appendix_figures_and_ranges_are_pointed_to():
    """Review F4: ufc_pavement's figure_e1_... and figure_f1_... and
    gec12's figure_7_10_to_13_kd (Figures 7-10 to 7-13)."""
    assert same_figure_methods("ufc_pavement", "E-1")[1] == [
        "figure_e1_flexible_thickness"]
    assert same_figure_methods("ufc_pavement", "F-1")[1] == [
        "figure_f1_rigid_thickness"]
    for number in ("7-10", "7-11", "7-13"):
        assert "figure_7_10_to_13_kd" in same_figure_methods(
            "gec_12", number)[1]
    for number in ("7-9", "7-14", "6-11"):
        assert "figure_7_10_to_13_kd" not in same_figure_methods(
            "gec_12", number)[1]


def test_same_figure_methods_on_the_real_library():
    # a lettered part counts as its figure; plot_ methods are left out
    module, methods = same_figure_methods("dm7_2", "5-18",
                                          "UFC 3-220-20 Foundations")
    assert module == "dm7"
    assert methods and all(m.startswith("figure_5_18") for m in methods)
    # a module with two manuals: DM7.1's Figure 5-16 is not DM7.2's
    _m, one = same_figure_methods("dm7_1", "5-16", "UFS 3-220-10 (DM 7.1)")
    _m, two = same_figure_methods("dm7_2", "5-16", "UFC 3-220-20 (DM 7.2)")
    assert one and not any(m.startswith("plot_") for m in one)
    assert two == []
    assert same_figure_methods("no_such_ref", "1-1") == (None, [])
