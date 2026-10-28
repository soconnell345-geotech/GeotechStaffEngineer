"""Model-first ingest's pages offline (BACKLOG RI-87 pieces 2 and 3).

No model is called. Piece 2 is tested on a PDF made inside the test: the
cut keeps the pages in order with their text, the limits say when a call
cannot carry its pages, the map sentence names every page, and the pictures
route sends each page's picture and its text between markers. Piece 3 is
tested on made-up page descriptions: which exploration a page logs, and
where a log, a laboratory window, a calculation printout and the
narrative's parts begin, and why. Every identifier, title and number here
is invented.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from report_ingest import model_first_pages as mfp

pymupdf = pytest.importorskip("pymupdf")

#: Three made-up pages; the middle one prints nothing.
PAGES = (("FIRST MADE-UP PAGE", "alpha line"),
         (),
         ("THIRD MADE-UP PAGE", "gamma line"))


@pytest.fixture()
def made_up_pdf(tmp_path):
    doc = pymupdf.open()
    for lines in PAGES:
        page = doc.new_page()
        for n, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * n), line, fontsize=11)
    path = tmp_path / "made_up.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _text_of(pdf: bytes) -> list:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_text() for page in doc]


def _d(**fields):
    """A made-up saved description record from before the describer named
    the exploration (prompt ``ri73-2``), as ``load_descriptions`` gives."""
    base = {"title": "", "kind": "made-up page", "names": "",
            "identifiers": [], "continues": "", "summary": "made up"}
    base.update(fields)
    return {"description": base}


def _e(exploration, **fields):
    """The same with the describer's ``exploration`` field (``ri73-3``)."""
    record = _d(**fields)
    record["description"]["exploration"] = exploration
    return record


# ---------------------------------------------------------------------------
# piece 2: the switch and the limits
# ---------------------------------------------------------------------------

def test_the_page_route_defaults_to_auto_and_refuses_a_typo(monkeypatch):
    monkeypatch.delenv(mfp.PAGES_ENV, raising=False)
    assert mfp.page_route() == "auto"
    monkeypatch.setenv(mfp.PAGES_ENV, " Pictures ")
    assert mfp.page_route() == "pictures"
    assert mfp.page_route("pdf") == "pdf"
    with pytest.raises(ValueError):
        mfp.page_route("pdfs")


def test_auto_sends_a_pdf_only_where_the_engine_can(monkeypatch):
    monkeypatch.delenv(mfp.PAGES_ENV, raising=False)
    assert mfp.resolve_route("auto", accepts_pdf=True) == "pdf"
    assert mfp.resolve_route("auto", accepts_pdf=False) == "pictures"
    assert mfp.resolve_route("pictures", accepts_pdf=True) == "pictures"
    with pytest.raises(ValueError, match="cannot send a PDF"):
        mfp.resolve_route("pdf", accepts_pdf=False)


def test_a_call_carries_no_more_pages_than_its_route_takes():
    assert mfp.max_pages_per_call("pdf") == mfp.MAX_PDF_PAGES == 100
    assert mfp.max_pages_per_call("pictures") == mfp.MAX_PICTURES_PER_CALL
    assert mfp.MAX_PICTURES_PER_CALL == 20


def test_the_pdf_limits_are_the_api_s():
    assert mfp.pdf_problem(100, mfp.MAX_PDF_BYTES) == ""
    assert "at most 100 pages" in mfp.pdf_problem(101, 10)
    assert mfp.pdf_problem(3, mfp.MAX_PDF_BYTES + 1)


def _pictures(*sizes, words="made up"):
    blocks = [{"type": "text", "text": words}]
    for size in sizes:
        blocks.append({"type": "image", "png": b"x" * size})
    return blocks


def test_a_pictures_call_is_checked_for_count_and_size(monkeypatch):
    assert mfp.pictures_problem(_pictures(*[10] * 20)) == ""
    assert "21 pictures" in mfp.pictures_problem(_pictures(*[10] * 21))
    # 3,000 bytes are 4,000 base64-encoded: over a 3,999 limit, not 4,000.
    monkeypatch.setattr(mfp, "MAX_PICTURE_BYTES", 4000)
    assert mfp.pictures_problem(_pictures(3000)) == ""
    monkeypatch.setattr(mfp, "MAX_PICTURE_BYTES", 3999)
    assert "picture 2 is 4,000 bytes" in mfp.pictures_problem(
        _pictures(30, 3000))
    monkeypatch.setattr(mfp, "MAX_PICTURE_BYTES", 10 ** 6)
    monkeypatch.setattr(mfp, "MAX_REQUEST_BYTES", 8000)
    assert mfp.pictures_problem(_pictures(3000, words="")) == ""
    assert "come to" in mfp.pictures_problem(_pictures(3000, 3000))


# ---------------------------------------------------------------------------
# piece 2: the cut
# ---------------------------------------------------------------------------

def test_the_pages_are_cut_out_in_order_from_a_path_or_bytes(made_up_pdf):
    data, n, sanitized = mfp.cut_pdf(made_up_pdf, [2, 0])
    assert n == 2 and not sanitized
    text = _text_of(data)
    assert "THIRD MADE-UP PAGE" in text[0] and "FIRST MADE-UP PAGE" in text[1]
    again, n_again, _ = mfp.cut_pdf(made_up_pdf.read_bytes(), [2])
    assert n_again == 1 and "gamma line" in _text_of(again)[0]
    whole, n_all, _ = mfp.cut_pdf(made_up_pdf, None)
    assert n_all == 3 and whole == made_up_pdf.read_bytes()
    with pytest.raises(ValueError, match="page 3 is not in"):
        mfp.cut_pdf(made_up_pdf, [3])


def test_a_cut_over_the_threshold_is_sanitized(made_up_pdf, monkeypatch):
    monkeypatch.setattr(mfp, "SANITIZE_ABOVE", 10)
    data, n, sanitized = mfp.cut_pdf(made_up_pdf, [0])
    assert sanitized and n == 1
    assert "FIRST MADE-UP PAGE" in _text_of(data)[0]


def test_the_baseline_cuts_maps_and_checks_with_the_package_s_functions():
    pytest.importorskip("langchain_core")
    from module_work.report_ingest_harness import run_plain_baseline as rpb

    assert rpb.cut_pdf is mfp.cut_pdf
    assert rpb.page_map_sentence is mfp.page_map_sentence
    assert rpb.pdf_problem is mfp.pdf_problem
    assert (rpb.MAX_PDF_PAGES, rpb.MAX_PDF_BYTES, rpb.SANITIZE_ABOVE) == (
        mfp.MAX_PDF_PAGES, mfp.MAX_PDF_BYTES, mfp.SANITIZE_ABOVE)


# ---------------------------------------------------------------------------
# piece 2: the page map and the pictures route
# ---------------------------------------------------------------------------

def test_the_map_sentence_names_every_pdf_page_s_report_page():
    said = mfp.page_map_sentence([7, 8, 12])
    assert ("PDF page 1 is page 7, PDF page 2 is page 8, PDF page 3 is "
            "page 12") in said
    assert "0-based" in said
    assert "first page is 0" in mfp.page_map_sentence([0, 1], whole=True)
    assert "picture headed by its page index" in mfp.page_map_sentence(
        [7], route="pictures")


def test_the_map_sentence_is_the_plain_baseline_s_wording():
    """The PDF wording is what RI-86 measured: the baseline's instruction
    carries the same sentence."""
    pytest.importorskip("langchain_core")
    from module_work.report_ingest_harness import run_plain_baseline as rpb

    said = rpb.instruction("calc", "R00", [12, 13])
    assert mfp.page_map_sentence([12, 13]) in said
    whole = rpb.instruction("narrative", "R00", [0, 1, 2], whole=True)
    assert mfp.page_map_sentence([0, 1, 2], whole=True) in whole


def test_the_pictures_route_sends_each_page_s_picture_then_its_text(
        made_up_pdf):
    from planlens.document import open_document

    doc = open_document(str(made_up_pdf), name="made_up")
    try:
        blocks = mfp.picture_blocks(doc, [2, 1])
        cut = mfp.picture_blocks(doc, [0], max_text_chars=5)
    finally:
        doc.close()
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "image", "text", "text", "image", "text"]
    assert blocks[0]["text"].startswith("Page 2 (picture 1 of 2)")
    body = blocks[2]["text"]
    start, end = mfp.TEXT_MARKERS
    assert start in body and end in body and "gamma line" in body
    assert body.index(start) < body.index("gamma line") < body.index(end)
    assert "text layer" in body
    assert blocks[3]["text"].startswith("Page 1 (picture 2 of 2)")
    assert "carries no text" in blocks[5]["text"]
    assert "cut at 5 characters" in cut[2]["text"]
    assert mfp.pictures_problem(blocks) == ""


#: A made-up prose line; three of them are more than today's narrative
#: reader's 120 characters a page.
PROSE = "The made-up site is underlain by made-up silty sand over made-up clay."


@pytest.fixture()
def prose_pdf(tmp_path):
    """Page 0 three lines of prose, page 1 one short line, page 2 none,
    page 3 prose again."""
    doc = pymupdf.open()
    for lines in ((f"MADE-UP PROSE 0 {PROSE}", PROSE, PROSE),
                  ("MADE-UP FIGURE 1",), (),
                  (f"MADE-UP PROSE 3 {PROSE}", PROSE, PROSE)):
        page = doc.new_page()
        for n, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * n), line, fontsize=9)
    path = tmp_path / "prose.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_a_page_with_a_usable_text_layer_goes_as_its_text_alone(prose_pdf):
    """RI-175: a page of today's narrative reader's 120 characters or more
    is sent as its text, under its page index, with no picture; one with
    less is sent as a picture with whatever text it has."""
    from planlens.document import open_document

    from report_ingest.narrative_reader import MIN_TEXT_CHARS

    doc = open_document(str(prose_pdf), name="prose")
    try:
        assert len(PROSE) * 3 > MIN_TEXT_CHARS > len("MADE-UP FIGURE 1")
        assert mfp.text_pages(doc, [0, 1, 2, 3]) == [0, 3]
        prose = mfp.page_blocks(doc, 3, as_text=True)
        figure = mfp.page_blocks(doc, 1, as_text=False)
        blank = mfp.page_blocks(doc, 2, as_text=True)
        forced = mfp.page_blocks(doc, 1, as_text=True)
    finally:
        doc.close()
    assert [b["type"] for b in prose] == ["text"]
    start, end = mfp.TEXT_MARKERS
    body = prose[0]["text"]
    assert body.startswith("Page 3's text, as the PDF's own text layer")
    assert body.index(start) < body.index("MADE-UP PROSE 3") < \
        body.index(end)
    assert [b["type"] for b in figure] == ["text", "image", "text"]
    assert figure[0]["text"].startswith("Page 1 (as a picture: it has too "
                                        "little text")
    assert "MADE-UP FIGURE 1" in figure[2]["text"]
    # A page with no text has nothing to send as text, so goes as a picture.
    assert [b["type"] for b in blank] == ["text", "image", "text"]
    assert "carries no text" in blank[2]["text"]
    assert [b["type"] for b in forced] == ["text"]


def test_a_scanned_page_with_document_intelligence_text_goes_as_text(
        prose_pdf):
    """RI-175 on a scanned report: a page with no text layer whose Azure
    Document Intelligence text the document carries counts as a text page
    for the narrative's one call, and that text is what is sent, said to be
    an optical reading -- so a report scanned throughout is not sent as
    pictures in parts. A made-up result in DI's documented shape."""
    from planlens.document import open_document
    from planlens.document.azure_di import AzureLayout

    lines = [f"MADE-UP SCANNED PAGE {PROSE}", PROSE, PROSE]
    spans, at = [], 0
    for text in lines:
        spans.append({"offset": at, "length": len(text)})
        at += len(text) + 1
    result = {"content": "\n".join(lines), "pages": [{
        "pageNumber": 3, "width": 595 / 72, "height": 842 / 72,
        "unit": "inch", "words": [],
        "lines": [{"content": text,
                   "polygon": [1.0, 1.0 + 0.3 * n, 7.0, 1.0 + 0.3 * n,
                               7.0, 1.2 + 0.3 * n, 1.0, 1.2 + 0.3 * n],
                   "spans": [span]}
                  for n, (text, span) in enumerate(zip(lines, spans))]}]}

    plain = open_document(str(prose_pdf), name="prose")
    try:
        assert mfp.text_pages(plain, [0, 1, 2, 3]) == [0, 3]
    finally:
        plain.close()
    doc = open_document(str(prose_pdf), name="prose",
                        text_source=AzureLayout(result))
    try:
        assert mfp.text_pages(doc, [0, 1, 2, 3]) == [0, 2, 3]
        blocks = mfp.page_blocks(doc, 2, as_text=True)
    finally:
        doc.close()
    assert [b["type"] for b in blocks] == ["text"]
    body = blocks[0]["text"]
    assert body.startswith("Page 2's text, as an optical reader read it")
    assert "MADE-UP SCANNED PAGE" in body and "UNRELIABLE" not in body


def test_picture_blocks_are_as_they_were_after_the_split_into_helpers(
        prose_pdf):
    """The pictures route of every other stage is unchanged: a header, the
    picture and the page's text, the same words as before RI-175."""
    from planlens.document import open_document

    doc = open_document(str(prose_pdf), name="prose")
    try:
        blocks = mfp.picture_blocks(doc, [0, 2])
    finally:
        doc.close()
    assert [b["type"] for b in blocks] == ["text", "image", "text"] * 2
    assert blocks[0]["text"] == "Page 0 (picture 1 of 2):"
    assert blocks[2]["text"].startswith(
        "Page 0's text, as the PDF's own text layer carries it, between the "
        "markers:\n" + mfp.TEXT_MARKERS[0] + "\nMADE-UP PROSE 0")
    assert blocks[2]["text"].endswith(mfp.TEXT_MARKERS[1])
    assert blocks[5]["text"] == ("Page 2 carries no text: read it from the "
                                 "picture.")


def test_the_text_first_map_sentence_says_how_each_page_is_given():
    said = mfp.page_map_sentence([0, 3], route="pictures", text_first=True)
    assert "its text alone" in said and "as a picture" in said
    assert "0-based" in said
    assert mfp.page_map_sentence([0, 3], route="pictures") != said
    # On the PDF route the PDF wording stands.
    assert mfp.page_map_sentence([0, 3], text_first=True) == \
        mfp.page_map_sentence([0, 3])


# ---------------------------------------------------------------------------
# piece 3: which exploration a page logs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, key", [
    # Forms no report here prints, which a list of words would miss.
    ("Sondeo S-1", "S1"), ("Trial Pit TP3", "TP3"), ("SPT-01", "SPT1"),
    ("Bohrung 3", "BOHRUNG 3"), ("C-1", "C1"),
    # One exploration, written several ways, is one key.
    ("Boring QX-04", "QX4"), ("QX-4", "QX4"), ("QX 4", "QX4"),
    ("qx-4", "QX4"), ("QX- 4", "QX4"), ("QX. NO. 4", "QX4"),
    ("QX_07A", "QX7A"), ("QX-K", "QX-K"), ("Hole No. (3)", "HOLE 3"),
    ("Probe Nr. K", "PROBE K"), ("Q-X No 6", "QX6"),
    ("Q. X/ No: (6)", "QX6"), ("Q X 6", "QX6"),
    ("QX-1 (ZW-1)", "QX1"), ("QX-K (ZW-1)", "QX-K"),
    ("Pit K (shown as hole K)", "PIT K"), ("TP (3)", "TP3"),
    ("QX-2 (written by hand on the sheet, beside the made-up header)",
     "QX2"),
    # One name with two numbers is folded, its numbers kept apart.
    ("QX-19-001", "QX19-1"), ("Boring QX-19-001", "QX19-1"),
    ("QX1-23", "QX1-23"), ("QX 1-23", "QX1-23"), ("QX-1/23", "QX1-23"),
    ("QX-23-01", "QX23-1"), ("Hole 19-001", "HOLE 19-1"),
    # Two explorations, or a name and a depth, are not one key.
    ("QX-19-001 & QX-19-002", None), ("QX-19-001, QX-19-002", None),
    ("QX-1 and QX-2", None), ("Pit K and Pit M", None),
    ("Pits K and M", None),
    ("Pit K (1,00 m) and Pit M", None), ("QX-8 at 10 m", None),
    # As the describer's own field gives it, a bare code is taken.
    ("3", "3"), ("K", "K"), ("", None),
])
def test_an_exploration_s_name_is_normalised_to_one_key(text, key):
    assert mfp.exploration_id(text) == key


@pytest.mark.parametrize("text", [
    "3", "K", "QX-K", "QX-8 at 10 m", "Sheet 1 of 2", "12/03/2031",
    "Figure 2.8", "Job 12345-6", "QX-19-001", "", "x" * 70,
])
def test_text_that_may_name_anything_is_read_strictly(text):
    """An identifier from a page's list or a title is one code with its own
    letters or a word before it, or nothing."""
    assert mfp.exploration_id(text, strict=True) is None
    assert mfp.exploration_id("Pit QX-K", strict=True) == "QX-K"


def test_the_describer_s_field_names_the_page_and_nothing_else_does():
    descriptions = {
        1: _e("Boring QX-04", identifiers=["Photo 12", "QX-9"]),
        2: _e("", identifiers=["QX-5", "Table 4"]),
        3: _e("Bohrung 3"),
    }
    named = mfp.page_identifiers([1, 2, 3], descriptions)
    assert named == {1: "QX4", 2: None, 3: "BOHRUNG 3"}
    assert mfp.page_identifier(descriptions[1]) == "QX4"
    assert mfp.page_identifier(None) is None


def test_a_field_name_with_two_numbers_still_tells_its_logs_apart():
    one_page = {20 + n: _e(f"QX-19-00{n}") for n in range(1, 5)}
    groups = mfp.log_groups(sorted(one_page), one_page)
    assert [g.pages for g in groups] == [[21], [22], [23], [24]]
    assert [g.ident for g in groups] == ["QX19-1", "QX19-2", "QX19-3",
                                         "QX19-4"]
    two_pages = {}
    for n in range(1, 5):
        two_pages[30 + 2 * n] = _e(f"QX{n}-23")
        two_pages[31 + 2 * n] = _e(f"QX {n}-23")
    groups = mfp.log_groups(sorted(two_pages), two_pages)
    assert [g.pages for g in groups] == [[32, 33], [34, 35], [36, 37],
                                         [38, 39]]


@pytest.mark.parametrize("first", ["CME-75", "ES1", "NAVD88"])
@pytest.mark.parametrize("pages_each", [1, 2])
def test_a_code_listed_first_on_every_page_of_several_logs_names_none(
        first, pages_each):
    """Without the field: a rig, a datum or a sample listed FIRST on every
    page of six explorations is passed over, as it is when it comes
    second."""
    descriptions = {}
    for n in range(6):
        for k in range(pages_each):
            descriptions[40 + n * pages_each + k] = _d(
                identifiers=[first, f"QX-{n + 1}", "Photo 3"])
    named = mfp.page_identifiers(sorted(descriptions), descriptions)
    assert sorted(set(named.values())) == [f"QX{n}" for n in range(1, 7)]
    groups = mfp.log_groups(sorted(descriptions), descriptions)
    assert len(groups) == 6
    assert all(len(g.pages) == pages_each for g in groups)


def test_a_log_s_own_samples_and_spellings_leave_its_name_alone():
    """The later sheets of one log list more of its samples, and a page may
    spell its exploration twice: neither makes the exploration shared."""
    descriptions = {
        50: _d(identifiers=["QX-4", "S-1"], continues="Sheet 1 of 3"),
        51: _d(identifiers=["QX-4", "S-2"], continues="Sheet 2 of 3"),
        52: _d(identifiers=["QX-4", "S-3"], continues="Sheet 3 of 3"),
        53: _d(identifiers=["QX-6", "Pit 6"]),
        54: _d(identifiers=["QX-6", "Pit No. 6"]),
        55: _d(identifiers=["QX-6", "Hole 6"]),
    }
    named = mfp.page_identifiers(sorted(descriptions), descriptions)
    assert named == {50: "QX4", 51: "QX4", 52: "QX4", 53: "QX6", 54: "QX6",
                     55: "QX6"}


def test_without_the_field_a_page_takes_its_first_code():
    assert mfp.page_identifier(_d(identifiers=["Figure 2.8", "QX-4",
                                               "QX-5"])) == "QX4"
    assert mfp.page_identifier(_d(title="Log of Boring QX-9")) == "QX9"
    assert mfp.page_identifier(_d(identifiers=["Sheet 1 of 2"])) is None
    assert mfp.page_identifier({"identifiers": ["QX-2"]}) == "QX2"
    assert mfp.page_identifier(_d(identifiers=["Annex 2", "QX-6"])) == "QX6"
    assert mfp.page_identifier(_d(identifiers=["Annex 2"])) == "ANNEX 2"


def test_without_the_field_a_code_on_other_explorations_pages_names_none():
    """Photos, tables, datums, rig models and standards a log prints are
    never the exploration it logs: a page's own exploration comes first, and
    a code printed beside two explorations or more is passed over even on a
    page that lists it first."""
    said = {
        10: ["NAVD 88", "QX-1", "CME-55", "ASTM D1586", "Photo 21"],
        11: ["QX-2", "NAVD 88", "CME-55", "ASTM D1586", "Table 4"],
        12: ["CME-55", "QX-3", "NAVD 88", "ASTM D1586", "Photo 22"],
        13: ["ASTM D1586", "QX-4", "CME-55", "NAVD 88"],
    }
    descriptions = {p: _d(identifiers=ids) for p, ids in said.items()}
    assert mfp.page_identifiers(said, descriptions) == {
        10: "QX1", 11: "QX2", 12: "QX3", 13: "QX4"}
    groups = mfp.log_groups(sorted(said), descriptions)
    assert [g.pages for g in groups] == [[10], [11], [12], [13]]


@pytest.mark.parametrize("fields, sheet", [
    ({"continues": "Sheet 2 of 3, marked continued"}, (2, 3)),
    ({"continues": "Blatt 2 von 3"}, (2, 3)),
    ({"continues": "Sheet 2/3"}, (2, 3)),
    ({"continues": "Sheet No. 4 of 7"}, (4, 7)),
    ({"continues": "none printed; page 84 of 97 in the file"}, None),
    ({"continues": "Page 9of 3"}, None),
    ({"continues": "printed 12/03/2031"}, None),
    ({"continues": "", "identifiers": ["QX-1", "Sheet 1 of 2"]}, (1, 2)),
    ({"continues": "", "title": "Made-up title Sheet 2 of 2"}, (2, 2)),
    ({"continues": "", "identifiers": ["Samples 1-20"]}, None),
])
def test_a_sheet_number_is_read_from_continues_then_identifiers(fields,
                                                                 sheet):
    assert mfp.sheet_of(_d(**fields)) == sheet


def test_only_words_say_a_page_is_continued_from_the_one_before():
    assert mfp.carries_on(_d(continues="continued from the sheet before"))
    assert mfp.carries_on(_d(continues="Continuation of QX-1"))
    assert not mfp.carries_on(_d(continues="Sheet 2 of 2"))
    assert not mfp.carries_on(_d(continues="Sheet 1 of 2 (continued)"))
    assert not mfp.carries_on(_d())


# ---------------------------------------------------------------------------
# piece 3: the groups
# ---------------------------------------------------------------------------

def test_logs_start_at_a_first_sheet_or_another_exploration():
    descriptions = {
        10: _e("QX-1", continues="Sheet 1 of 2"),
        11: _e("QX-1", continues="Sheet 2 of 2"),
        12: _e("QX-2", continues="Sheet 1 of 1"),
        13: _e(""),
        14: _e("QX-3", continues="Sheet 2 of 2"),
        15: _e("QX-3", continues="Sheet 1 of 2"),
        16: _e("QX-9", continues="continued from the page before"),
        18: _e("QX-K"),
    }
    groups = mfp.log_groups(sorted(descriptions), descriptions)
    assert [g.pages for g in groups] == [[10, 11], [12, 13], [14], [15, 16],
                                         [18]]
    assert [g.ident for g in groups] == ["QX1", "QX2", "QX3", "QX3", "QX-K"]
    assert [g.why for g in groups] == [
        "the first log page", "the first sheet of 1",
        "a new exploration identifier (QX3)", "the first sheet of 2",
        "a new exploration identifier (QX-K)"]
    assert any("continued from the page before but names QX9" in n
               for n in groups[3].notes)


def test_appendix_page_numbers_never_carry_one_log_into_the_next():
    """A run of logs printed "page 7 of 45", "page 8 of 45" .. -- the
    appendix's numbering, not a log's sheets -- is one log a page."""
    descriptions = {20 + n: _e(f"QX-{n + 1}", continues=f"Page {7 + n} of 45")
                    for n in range(4)}
    groups = mfp.log_groups(sorted(descriptions), descriptions)
    assert [g.pages for g in groups] == [[20], [21], [22], [23]]


def test_a_log_longer_than_the_cap_is_read_in_parts_of_one_log():
    descriptions = {p: _e("QX-1") for p in range(30, 50)}
    groups = mfp.log_groups(sorted(descriptions), descriptions, max_pages=12)
    assert [len(g.pages) for g in groups] == [10, 10]
    assert [g.part for g in groups] == [(1, 2), (2, 2)]
    assert len({g.key for g in groups}) == 1


def test_lab_windows_cut_where_a_new_sheet_begins():
    descriptions = {p: _e("QX-1", title="MADE-UP TEST A")
                    for p in range(100, 115)}
    for p in range(108, 115):
        descriptions[p] = _e("QX-2", title="MADE-UP TEST A")
    descriptions[120] = _e("", title="MADE-UP TEST B")
    starts = mfp.lab_starts(list(range(100, 115)), descriptions)
    assert starts == [108]
    groups = mfp.lab_groups(sorted(descriptions), descriptions, size=12)
    assert [(g.pages[0], g.pages[-1]) for g in groups] == [
        (100, 107), (108, 114), (120, 120)]
    assert groups[1].why == "a new sheet begins here"
    assert all(len(g.pages) <= 12 for g in groups)


def test_a_lab_page_continued_from_the_one_before_begins_nothing():
    descriptions = {1: _e("", title="MADE-UP TEST A"),
                    2: _e("", title="ANOTHER HEADING",
                          continues="continued from the sheet before"),
                    3: _e("", title="A THIRD HEADING",
                          continues="Page 2 of 2")}
    assert mfp.lab_starts([1, 2, 3], descriptions) == [3]


def test_calc_printouts_start_at_a_described_first_sheet():
    descriptions = {
        200: _d(title="MADE-UP CALC ONE", continues="Sheet 1 of 3"),
        201: _d(title="", continues="Sheet 2 of 3"),
        202: _d(title="", continues=""),
        203: _d(title="MADE-UP CALC TWO", continues="Sheet 1 of 2"),
        204: _d(title="MADE-UP CALC TWO", continues="Sheet 2 of 2"),
        205: _d(title="MADE-UP SCENARIO", continues="Page 1 of 1"),
        206: _d(title="MADE-UP SCENARIO", continues="Page 1 of 1"),
        207: _d(title="MADE-UP OTHER", continues="Page 1 of 1"),
    }
    assert mfp.calc_starts(sorted(descriptions), descriptions) == [
        203, 205, 207]
    groups = mfp.calc_groups(sorted(descriptions), None, descriptions)
    assert [g.pages for g in groups] == [[200, 201, 202], [203, 204],
                                         [205, 206], [207]]
    assert groups[3].why == "a new printout, by its description"


def test_printouts_back_to_back_without_sheet_numbers_are_cut_apart():
    """RI-176: two printouts back to back with no sheet numbers are read
    apart by what their pages' descriptions say -- another exploration,
    another program, a title page -- and a printout's own charts, each
    with a title of its own, stay one printout. Made-up pages."""
    def page(title="", kind="calculation printout from a program",
             exploration="", names=""):
        return _e(exploration, title=title, kind=kind, names=names)

    by_hole = {500: page("MADE-UP ANALYSIS", exploration="QX-1"),
               501: page(kind="data listing page"),
               502: page("MADE-UP SUMMARY", exploration="none printed"),
               503: page("MADE-UP ANALYSIS", exploration="QX-2"),
               504: page(kind="data listing page")}
    assert mfp.calc_starts(sorted(by_hole), by_hole) == [503]
    by_program = {510: page("MADE-UP INPUT", names="LPILE"),
                  511: page("MADE-UP OUTPUT"),
                  512: page("MADE-UP INPUT", names="Slide2; a made-up rule"),
                  513: page("MADE-UP OUTPUT")}
    assert mfp.calc_starts(sorted(by_program), by_program) == [512]
    title_pages = {520: page("MADE-UP CALCULATION",
                             kind="title page of a calculation"),
                   521: page("MADE-UP TABLE"),
                   522: page("MADE-UP CALCULATION",
                             kind="title page of a calculation"),
                   523: page("MADE-UP TABLE")}
    assert mfp.calc_starts(sorted(title_pages), title_pages) == [522]
    charts = {530: page("MADE-UP PILE CAPACITY", kind="chart page"),
              531: page("MADE-UP SKIN FRICTION CAPACITY", kind="chart page"),
              532: page("MADE-UP LOAD AGAINST SETTLEMENT", kind="chart page")}
    assert mfp.calc_starts(sorted(charts), charts) == []

    # A guess at the program's family is not a program: it cuts nothing
    # between two pages that name the program outright.
    hedged = {540: page("MADE-UP INPUT", names="STABL6H; a made-up method"),
              541: page("MADE-UP OUTPUT",
                        names="made-up circle search (STABL-type output)"),
              542: page("MADE-UP OUTPUT",
                        names="made-up output file, likely a STABL program"),
              543: page("MADE-UP INPUT", names="STABL6H")}
    assert [mfp._program_named(hedged[p]) for p in sorted(hedged)] == [
        "stabl6h", None, None, "stabl6h"]
    assert mfp.calc_starts(sorted(hedged), hedged) == []

    every = {**by_hole, **by_program}
    groups = mfp.calc_groups(sorted(every), None, every)
    assert [g.pages for g in groups] == [[500, 501, 502], [503, 504],
                                         [510, 511], [512, 513]]
    assert groups[1].why == "a new printout, by its description"


def test_calc_heads_run_over_the_whole_run_then_the_descriptions_cut(
        monkeypatch):
    from report_ingest import calc_reader

    seen = []

    def heads(doc, pages):
        seen.append(list(pages))
        return [pages[:4], pages[4:]]

    monkeypatch.setattr(calc_reader, "split_calc_runs", heads)
    descriptions = {p: _d() for p in range(300, 308)}
    descriptions[302] = _d(continues="Sheet 1 of 2")
    groups = mfp.calc_groups(list(range(300, 308)), object(), descriptions)
    assert seen == [list(range(300, 308))]
    assert [g.pages for g in groups] == [[300, 301], [302, 303],
                                         [304, 305, 306, 307]]
    assert [g.why for g in groups] == [
        "a run of calculation pages",
        "a new printout, by its description",
        "a new printout by its head, or a balanced cut"]


def test_a_calc_run_without_a_document_is_cut_only_to_two_calls():
    from report_ingest.calc_reader import MAX_RUN_PAGES

    pages = list(range(MAX_RUN_PAGES + 5))
    groups = mfp.calc_groups(pages)
    assert len(groups) == 2 and sum(len(g.pages) for g in groups) == len(pages)
    assert all(len(g.pages) <= MAX_RUN_PAGES for g in groups)


def test_a_long_printout_is_two_calls_with_its_first_and_last_page_in_both():
    from report_ingest.calc_reader import MAX_CALC_PAGES

    short = mfp.PageGroup("calc", list(range(MAX_CALC_PAGES)))
    assert mfp.calc_calls(short) == [short.pages]
    long = mfp.PageGroup("calc", list(range(20)))
    calls = mfp.calc_calls(long)
    assert len(calls) == 2
    assert all(c[0] == 0 and c[-1] == 19 for c in calls)
    assert set(calls[0]) | set(calls[1]) == set(range(20))


def test_a_divider_is_in_no_group_and_no_group_runs_across_it():
    """RI-177: a page the describer calls a divider takes its section's
    label, but holds nothing to read: no reader is sent it, alone or with
    the section's pages, and no group runs across it. Made-up pages."""
    divider = "appendix divider page"
    labels = {0: "cover", 1: "narrative", 2: "narrative", 3: "boring_log",
              4: "boring_log", 5: "boring_log", 6: "lab_test",
              7: "lab_test", 8: "lab_test", 9: "calculation",
              10: "calculation", 11: "calculation", 12: "calculation",
              13: "calculation"}
    descriptions = {p: _e("", title="MADE-UP PAGE") for p in labels}
    descriptions[3] = _e("QX-1", title="MADE-UP LOG")
    descriptions[4] = _e("", kind=divider, title="APPENDIX Q")
    # Names no exploration: without the divider it would join QX-1's log.
    descriptions[5] = _e("", title="MADE-UP LOG")
    descriptions[6] = _e("", kind=divider, title="APPENDIX R")
    descriptions[11] = _e("", kind=divider, title="APPENDIX S")
    assert mfp.divider_pages(descriptions) == [4, 6, 11]
    assert not mfp.is_divider(_d(kind="section drawn with divider lines"))

    groups = mfp.group_pages(labels, descriptions, n_pages=14,
                             front_pages=5)
    assert [g.pages for g in groups["logs"]] == [[3], [5]]
    assert groups["logs"][1].why == "the first log page after a divider"
    assert [g.pages for g in groups["lab"]] == [[7, 8]]
    assert [g.pages for g in groups["calc"]] == [[9, 10], [12, 13]]
    # The front is the first five pages, the divider among them left out.
    assert [g.pages for g in groups["narrative"]] == [[0, 1, 2, 3]]
    every = {p for stage in groups.values() for g in stage for p in g.pages}
    assert not every & {4, 6, 11}


LABELS = {0: "cover", 3: "narrative", 60: "narrative", 61: "figure",
          62: "plan", 63: "boring_log", 70: "profile", 90: "lab_test",
          95: "appendix_unknown"}


def test_the_narrative_is_its_pages_the_front_and_the_figures_in_parts():
    groups = mfp.narrative_groups(LABELS, 100, front_pages=5)
    assert [g.pages for g in groups] == [[0, 1, 2, 3, 4, 60, 61, 62, 70]]
    parts = mfp.narrative_groups(LABELS, 100, front_pages=5, max_pages=4)
    assert [len(g.pages) for g in parts] == [3, 3, 3]
    assert [g.part for g in parts] == [(1, 3), (2, 3), (3, 3)]
    budget = mfp.narrative_groups(LABELS, 100, front_pages=0,
                                  page_chars={3: 50, 60: 60, 61: 10,
                                              62: 10, 70: 10},
                                  max_chars=100)
    assert [g.pages for g in budget] == [[0, 3], [60, 61, 62, 70]]


def test_the_narrative_s_pages_are_ranked_front_then_its_own_then_figures():
    """RI-175: what the reader keeps first when not all fit one request."""
    labels = {0: "cover", 1: "letter", 4: "figure", 6: "narrative",
              7: "plan", 9: "narrative", 12: "boring_log"}
    (group,) = mfp.narrative_groups(labels, 20, front_pages=2)
    assert group.pages == [0, 1, 4, 6, 7, 9]
    assert group.ranked == [0, 1, 6, 9, 4, 7]
    assert group.n_front == 2
    (bound,) = mfp.narrative_groups(labels, 20, front_pages=1,
                                    window=range(3, 12))
    assert bound.ranked == [3, 6, 9, 4, 7] and bound.n_front == 1
    # The ranking is the narrative's alone; no other group has one.
    assert mfp.PageGroup("logs", [3]).ranked == []


def test_on_the_pictures_route_the_narrative_is_one_part():
    """RI-175: sent text first, it is one call whatever the page count and
    the text budget; the request's own limit decides what goes."""
    chars = {3: 50, 60: 60, 61: 10, 62: 10, 70: 10}
    one = mfp.narrative_groups(LABELS, 100, front_pages=5, max_pages=4,
                               page_chars=chars, max_chars=100,
                               route="pictures")
    assert [g.pages for g in one] == [[0, 1, 2, 3, 4, 60, 61, 62, 70]]
    assert one[0].part is None
    split = mfp.narrative_groups(LABELS, 100, front_pages=5, max_pages=4,
                                 route="pdf")
    assert len(split) == 3
    by_route = {route: mfp.group_pages(LABELS, {}, n_pages=100,
                                       route=route, front_pages=5,
                                       page_chars=chars, max_chars=100)
                for route in ("pdf", "pictures")}
    assert len(by_route["pdf"]["narrative"]) == 2
    assert [g.pages for g in by_route["pictures"]["narrative"]] == \
        [[0, 1, 2, 3, 4, 60, 61, 62, 70]]


def test_a_bound_report_s_front_is_its_own_first_pages():
    """A report bound inside another (``window``) is read within its own
    pages, and its front is the first pages OF IT, as reading_pages does."""
    groups = mfp.narrative_groups(LABELS, 100, front_pages=3,
                                  window=range(58, 80))
    assert [g.pages for g in groups] == [[58, 59, 60, 61, 62, 70]]


def test_every_reader_s_groups_come_from_one_call():
    labels = {0: "cover", 1: "narrative", 2: "boring_log", 3: "boring_log",
              4: "lab_test", 5: "calculation", 6: "cpt_log"}
    descriptions = {2: _e("QX-1"), 3: _e("QX-2")}
    groups = mfp.group_pages(labels, descriptions, n_pages=7)
    assert [g.pages for g in groups["logs"]] == [[2], [3]]
    assert [g.pages for g in groups["lab"]] == [[4]]
    assert [g.pages for g in groups["calc"]] == [[5]]
    assert groups["narrative"][0].pages == list(range(7))
    assert all(6 not in g.pages for key in ("logs", "lab", "calc")
               for g in groups[key])


def test_group_pages_passes_the_narrative_s_settings_and_the_window():
    labels = {**LABELS, 64: "boring_log"}
    descriptions = {63: _e("QX-1"), 64: _e("QX-2")}
    bound = mfp.group_pages(labels, descriptions, n_pages=100,
                            window=range(58, 64), front_pages=2,
                            page_chars={58: 10, 59: 10, 60: 10, 61: 10,
                                        62: 10}, max_chars=30)
    assert [g.pages for g in bound["logs"]] == [[63]]
    assert [g.pages for g in bound["narrative"]] == [[58, 59, 60],
                                                     [61, 62]]
    assert bound["lab"] == [] and bound["calc"] == []


def test_the_stage_labels_are_the_graph_s_and_the_narrative_reader_s():
    from report_ingest.graph import ITEM_READERS
    from report_ingest.label_review import LABEL_DEFINITIONS
    from report_ingest.narrative_reader import FRONT_LABELS

    readers = {"logs": "log", "lab": "lab", "calc": "calc"}
    for stage, reader in readers.items():
        assert set(mfp.STAGE_LABELS[stage]) == {
            label for label, who in ITEM_READERS.items() if who == reader}
    assert set(mfp.NARRATIVE_LABELS) == set(FRONT_LABELS)
    assert set(mfp.FIGURE_LABELS) <= set(LABEL_DEFINITIONS)


def test_nothing_in_ingest_imports_this_module_yet():
    code = ("import sys, report_ingest.graph, report_ingest.subagent; "
            "print('report_ingest.model_first_pages' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
