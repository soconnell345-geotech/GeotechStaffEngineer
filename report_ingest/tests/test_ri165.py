"""BACKLOG RI-165: the carried hole and sample link on the model-first lab
route.

A lab item cut where its sheet changes (``split_lab_items``, RI-150) gives a
later part whose pages print no boring or sample the first part's
(``evidence["carried_link"]``); today's reader is told it. The model-first
lab route takes it too, by code after the answer, so no request changes.
Every page, hole and value here is made up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from report_ingest import model_first as mf
from report_ingest import model_first_read as mfr
from report_ingest.model_first_pages import PageGroup, lab_groups
from report_ingest.tests.fake_engine import FakeEngine

pymupdf = pytest.importorskip("pymupdf")


def _open(pdf: bytes, tmp_path, name: str = "made_up.pdf"):
    from planlens.document import open_document
    path = tmp_path / name
    path.write_bytes(pdf)
    return open_document(str(path))


def _titled(tmp_path, titles, heads=()):
    """Made-up pages, one per title, each with the header line in
    ``heads`` (or none)."""
    from report_ingest.tests.lab_fixtures import LETTER, _text
    made = pymupdf.open()
    for n, title in enumerate(titles):
        page = made.new_page(width=LETTER[0], height=LETTER[1])
        if title:
            _text(page, 150, 60, title, 14, bold=True)
        if n < len(heads) and heads[n]:
            _text(page, 60, 100, heads[n])
        _text(page, 60, 400, "Sheet notes")
    doc = _open(made.tobytes(), tmp_path)
    made.close()
    return doc


def _reading(page, *, hole="", sample=""):
    return mf.LabSheetReading.model_validate({
        "depth_unit": "ft",
        "tests": [
            {"kind": "atterberg", "investigation_id": hole,
             "sample_id": sample, "depth_top": 4.0, "ll": 41.0, "pl": 19.0,
             "pi": 22.0, "prov": {"page": page}},
            {"kind": "atterberg", "investigation_id": "QX-9",
             "sample_id": "S-8", "depth_top": 9.0, "ll": 33.0, "pl": 17.0,
             "pi": 16.0, "prov": {"page": page}},
        ]})


def _engine(turns):
    engine = FakeEngine(turns)
    engine.accepts_pdf = True
    return engine


def _sent(call):
    """What one call sent, a PDF block by its pages' text (a cut PDF's
    bytes carry a fresh identifier each time)."""
    out = []
    for message in call["messages"]:
        for block in message["content"]:
            if block.get("type") == "pdf":
                with pymupdf.open(stream=block["data"], filetype="pdf") as cut:
                    out.append(("pdf", [page.get_text() for page in cut]))
            else:
                out.append(block)
    return out


@pytest.fixture
def three_sheets(tmp_path):
    """A summary printing its boring, then two Atterberg sheets printing
    none."""
    doc = _titled(tmp_path, ("SUMMARY OF LABORATORY TESTS",
                             "ATTERBERG LIMITS", "ATTERBERG LIMITS"),
                  heads=("Boring: QX-7", "", ""))
    yield doc
    doc.close()


def test_a_cut_part_takes_its_carried_link_and_the_request_is_unchanged(
        three_sheets):
    carried = {"investigation_id": "QX-7", "page": 0}
    plain = _engine([{"final": _reading(1)}])
    told = _engine([{"final": _reading(1)}])
    (before,) = mfr.read_lab(three_sheets, [PageGroup(
        "lab", [1, 2], key="lab-1")], plain, report_id="R00")
    (after,) = mfr.read_lab(three_sheets, [PageGroup(
        "lab", [1, 2], key="lab-1", carried_link=carried)], told,
        report_id="R00")
    # The model is sent the same request either way: no prompt change.
    assert _sent(told.calls[0]) == _sent(plain.calls[0])
    assert [t["investigation_id"] for t in before["tests"]] == ["", "QX-9"]
    assert [t["investigation_id"] for t in after["tests"]] == ["QX-7", "QX-9"]
    # A test naming its own boring and sample keeps them.
    assert after["tests"][1]["sample_id"] == "S-8"
    assert "carried from page 0" in after["tests"][0]["note"]
    assert "RI-165" in after["tests"][0]["note"]
    assert after["carried_link"] == {"investigation_id": "QX-7", "page": 0,
                                     "tests": 1}
    assert "carried_link" not in before


def test_a_test_naming_only_its_sample_keeps_it_without_a_carried_hole(
        three_sheets):
    engine = _engine([{"final": _reading(1, sample="S-2")}])
    (blob,) = mfr.read_lab(three_sheets, [PageGroup(
        "lab", [1, 2], key="lab-1",
        carried_link={"investigation_id": "QX-7", "page": 0})], engine)
    assert [(t["investigation_id"], t["sample_id"]) for t in blob["tests"]] \
        == [("", "S-2"), ("QX-9", "S-8")]
    assert "carried_link" not in blob


def test_a_later_window_of_a_run_carries_what_its_earlier_pages_print(
        three_sheets):
    engine = _engine([{"final": _reading(2)}])
    (blob,) = mfr.read_lab(three_sheets, [PageGroup(
        "lab", [2], key="lab-0", part=(2, 2), carried_from=[1, 0])], engine)
    assert blob["tests"][0]["investigation_id"] == "QX-7"


def test_a_group_printing_its_own_boring_carries_nothing(tmp_path):
    doc = _titled(tmp_path, ("SUMMARY OF LABORATORY TESTS",
                             "ATTERBERG LIMITS"),
                  heads=("Boring: QX-7", "Boring: QX-8"))
    try:
        engine = _engine([{"final": _reading(1)}])
        (blob,) = mfr.read_lab(doc, [PageGroup(
            "lab", [1], key="lab-0", part=(2, 2), carried_from=[0])], engine)
        assert blob["tests"][0]["investigation_id"] == ""
        assert "carried_link" not in blob
    finally:
        doc.close()


def test_lab_groups_name_the_pages_before_each_later_window_nearest_first():
    groups = lab_groups(list(range(10, 40)), {}, size=12)
    assert len(groups) > 1
    assert groups[0].carried_from == []
    for group in groups[1:]:
        assert group.carried_from == list(range(group.pages[0] - 1, 9, -1))
    assert "carried_from" not in groups[0].to_dict()
    assert groups[1].to_dict()["carried_from"] == groups[0].pages[::-1]


def _run(tmp_path, n, heads):
    """A run of ``n`` made-up lab pages, ``heads`` the header line printed
    on some of them."""
    return _titled(tmp_path, ["LABORATORY TEST"] * n,
                   heads=[heads.get(p, "") for p in range(n)])


def _carries(doc, n, descriptions=None):
    """Each window's first page and the boring it carries."""
    from report_ingest.lab_reader import _printed_link
    return [(g.pages[0],
             mfr._carried(doc, g, _printed_link).get("investigation_id"))
            for g in lab_groups(list(range(n)), descriptions or {}, size=12)]


def test_a_window_carries_the_hole_printed_nearest_before_it(tmp_path):
    """S1: QX-2 is printed after QX-1 and before the window; continuation
    sheets belong to the nearer one."""
    doc = _run(tmp_path, 24, {0: "Boring: QX-1", 6: "Boring: QX-2"})
    try:
        assert _carries(doc, 24) == [(0, None), (12, "QX-2")]
        later = lab_groups(list(range(24)), {}, size=12)[1]
        (blob,) = mfr.read_lab(doc, [later], _engine([{"final":
                                                       _reading(12)}]))
        assert blob["tests"][0]["investigation_id"] == "QX-2"
        assert "carried from page 6" in blob["tests"][0]["note"]
    finally:
        doc.close()


def test_a_window_carries_the_hole_of_the_window_before_it(tmp_path):
    """S2: the second window prints QX-3; the third, printing none, carries
    QX-3, not the first window's QX-1."""
    doc = _run(tmp_path, 36, {0: "Boring: QX-1", 12: "Boring: QX-3"})
    try:
        assert _carries(doc, 36) == [(0, None), (12, None), (24, "QX-3")]
    finally:
        doc.close()


def test_a_window_the_descriptions_say_logs_a_hole_carries_nothing(
        tmp_path):
    """S3: a new sheet begins on page 12, and the descriptions name QX-5
    there; its text prints the code without the word beside it, so the
    floor reads no boring. Nothing is carried, not QX-1."""
    doc = _run(tmp_path, 24, {0: "Boring: QX-1", 12: "QX-5"})
    descriptions = {p: {"title": "summary" if p == 0 else
                        "particle size" if p < 12 else "corrosion series",
                        "exploration": "QX-1" if p < 12 else "QX-5"}
                    for p in range(24)}
    try:
        assert _carries(doc, 24, descriptions) == [(0, None), (12, None)]
    finally:
        doc.close()


def test_the_pages_carried_from_stop_at_a_page_naming_another_hole(
        tmp_path):
    """The descriptions name QX-2 on page 6, printed without the word
    beside it: QX-1 on page 0 is beyond it and is not carried."""
    doc = _run(tmp_path, 24, {0: "Boring: QX-1", 6: "QX-2"})
    descriptions = {p: {"title": "LABORATORY TEST",
                        "exploration": {0: "QX-1", 6: "QX-2"}.get(p, "")}
                    for p in range(24)}
    try:
        assert lab_groups(list(range(24)), descriptions,
                          size=12)[1].carried_from == [11, 10, 9, 8, 7, 6]
        assert _carries(doc, 24, descriptions) == [(0, None), (12, None)]
    finally:
        doc.close()


def test_a_summary_then_sheets_printing_no_hole_still_carry_it(tmp_path):
    """RI-150's shape over a long run: the summary prints QX-7, the sheets
    after it none; with the descriptions or without."""
    doc = _run(tmp_path, 24, {0: "Boring: QX-7"})
    descriptions = {p: {"title": "summary" if p == 0 else "atterberg limits",
                        "exploration": "QX-7" if p == 0 else ""}
                    for p in range(24)}
    try:
        assert _carries(doc, 24) == [(0, None), (12, "QX-7")]
        assert _carries(doc, 24, descriptions) == [(0, None), (12, "QX-7")]
    finally:
        doc.close()


def test_todays_cut_items_give_their_carried_link_to_the_model_first_group(
        three_sheets):
    """With the labels not read model first, each of today's items is one
    lab group; a cut part's carried link goes with it."""
    from planlens.document.roles import Item

    from report_ingest.graph import _model_first_units
    from report_ingest.lab_reader import split_lab_items

    items = split_lab_items(three_sheets, [
        Item(id="item_3", kind="lab_test", pages=[0, 1, 2])])
    assert items[1].evidence["carried_link"]["investigation_id"] == "QX-7"
    model_first = SimpleNamespace(labels=False, descriptions={},
                                  routes={"narrative": "pdf"},
                                  stages=("lab",))
    units = _model_first_units(three_sheets, model_first,
                               {0: "lab_test", 1: "lab_test", 2: "lab_test"},
                               items=items)
    groups = [g for _u, parts in units for g in parts if g.stage == "lab"]
    assert [g.pages for g in groups] == [[0], [1, 2]]
    assert groups[0].carried_link == {}
    assert groups[1].carried_link["investigation_id"] == "QX-7"
