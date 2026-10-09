"""read_text_file reads Word and Excel files as Markdown (live smoke wave 2a,
B3), and write_docx signs the file with the app's author (B4).

F02 t3 and F11 t3: read_text_file refused a .docx ("binary file, not text"),
open_document rendered it to pages and read_document returned tables one cell
per line with list markers and line breaks gone, so each "edit the memo"
rebuilt it from flat text and drifted; F31 invented a ``read_docx_placeholder``
tool; F41 met an .xlsx nothing could read. No model, no network.
"""

import json
import os

import pytest

docx = pytest.importorskip("docx")
openpyxl = pytest.importorskip("openpyxl")
fitz = pytest.importorskip("fitz")

from funhouse_agent import _fileio  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    SCRATCH_DIR, dispatch_extended_tool)


def _call(name, args):
    return json.loads(dispatch_extended_tool(name, args, engine=None,
                                             attachments={}))


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A conversation's working folder, bound the way the app's turn worker
    binds it; no deployment author set."""
    monkeypatch.delenv("GEOTECH_MARKUP_AUTHOR", raising=False)
    work = tmp_path / "conv" / "files"
    work.mkdir(parents=True)
    with _fileio.working_dir_bound(str(work)):
        yield work


def _png(path):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
    pix.set_rect(pix.irect, (40, 120, 200))
    path.write_bytes(pix.tobytes("png"))


MEMO = """\
**To:** Project file
**Re:** Summary of submittal.pdf
**Status:** DRAFT

## Findings

1. The logs end at 30 ft
2. The lab sheet is missing
   - no gradation
   - no Atterberg limits

| Boring | Depth (ft) |
| --- | --- |
| B-1 | 30 |

*Table 1. Borings*

![Boring plan](plan.png)
"""


def test_a_memo_written_with_write_docx_reads_back_and_round_trips(folder):
    _png(folder / "plan.png")
    out = _call("write_docx", {"path": str(folder / "memo.docx"),
                               "markdown": MEMO, "title": "Submittal Memo"})
    assert out.get("file_exists") is True, out
    read = _call("read_text_file", {"path": "memo.docx"})
    assert read["format"] == "docx" and read["read_as"] == "markdown"
    assert read["title"] == "Submittal Memo"
    assert "write_docx" in read["note"] and "Submittal Memo" in read["note"]
    md = read["text"]
    assert "**To:** Project file\n**Re:** Summary of submittal.pdf\n" in md
    assert "1. The logs end at 30 ft\n2. The lab sheet is missing\n" \
           "   - no gradation\n   - no Atterberg limits" in md
    assert "| Boring | Depth (ft) |\n| --- | --- |\n| B-1 | 30 |" in md
    assert "*Table 1. Borings*" in md
    # The picture went to the conversation's scratch folder and is named
    # so write_docx can embed it again.
    pic = f"{SCRATCH_DIR}/memo_fig1.png"
    assert f"![Boring plan]({pic})" in md
    assert read["pictures"] == [pic]
    assert os.path.isfile(folder / SCRATCH_DIR / "memo_fig1.png")

    # Edit and write back: the memo keeps its shape (and its picture).
    edited = md.replace("**Status:** DRAFT", "**Status:** FINAL")
    out2 = _call("write_docx", {"path": str(folder / "memo_v2.docx"),
                                "markdown": edited, "title": read["title"]})
    assert out2.get("file_exists") is True and "warnings" not in out2, out2
    again = _call("read_text_file", {"path": "memo_v2.docx"})
    # The same Markdown, its picture now saved under the new file's name.
    assert again["text"] == edited.replace("memo_fig1", "memo_v2_fig1")
    doc = docx.Document(str(folder / "memo_v2.docx"))
    assert len(doc.inline_shapes) == 1
    assert [p.style.name for p in doc.paragraphs].count("Title") == 1


def test_a_long_document_pages_and_the_note_comes_once(folder):
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(60))
    _call("write_docx", {"path": str(folder / "long.docx"),
                         "markdown": body})
    first = _call("read_text_file", {"path": "long.docx", "max_chars": 2000})
    assert first["truncated"] and first["next_offset"] == 2000
    assert "note" in first
    second = _call("read_text_file", {"path": "long.docx", "offset": 2000,
                                      "max_chars": 2000})
    assert "note" not in second and second["format"] == "docx"
    assert second["text"] and second["text"] != first["text"]


def test_an_excel_workbook_reads_as_one_table_per_sheet(folder):
    import datetime
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Submittal Log"
    ws.append(["No.", "Item", "Received", "Qty", "Status"])
    ws.append([1, "Geotech report | rev A", datetime.date(2026, 9, 1), 2.0,
               "Open"])
    ws.append([2, "Boring logs\nset 1", datetime.datetime(2026, 9, 3, 14, 30),
               2.5, True])
    ws.append([None, "Total", None, "=SUM(D2:D3)", None])
    notes = wb.create_sheet("Notes")
    notes["B3"] = "Spec 32 16 00"
    hidden = wb.create_sheet("Lookup")
    hidden.sheet_state = "hidden"
    hidden["A1"] = "x"
    wb.create_sheet("Empty")
    wb.save(str(folder / "log.xlsx"))

    read = _call("read_text_file", {"path": "log.xlsx"})
    assert read["format"] == "xlsx" and read["read_as"] == "markdown"
    md = read["text"]
    assert "## Sheet: Submittal Log" in md and "## Sheet: Notes" in md
    assert "## Sheet: Lookup (hidden)" in md and "## Sheet: Empty" in md
    assert "| # | No. | Item | Received | Qty | Status |" in md
    assert "| 2 | 1 | Geotech report \\| rev A | 2026-09-01 | 2 | Open |" in md
    assert "| 3 | 2 | Boring logs<br>set 1 | 2026-09-03 14:30 | 2.5 | TRUE |" \
        in md
    # A formula Excel never calculated shows as the formula.
    assert "| 4 |  | Total |  | =SUM(D2:D3) |  |" in md
    # A sheet's used range: Excel row and column of its one cell.
    assert "Excel rows 3-3, columns B-B" in md
    by_name = {s["name"]: s for s in read["sheets"]}
    assert by_name["Submittal Log"] == {"name": "Submittal Log", "rows": 4,
                                        "columns": 5, "header_row": 1}
    assert by_name["Lookup"]["hidden"] is True
    assert by_name["Empty"]["rows"] == 0
    assert "sheet" in read["note"].lower()


def test_an_excel_header_below_a_title_is_found_and_said(folder):
    """Live smoke 2c, E7 (F41): the log's title is row 1, a project line row
    2, and the header row 4; the reader said "row 1 is the header". The
    header is the first row filled across the table, mostly with text; the
    rows above it are kept as text, and the result says which row it used."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Submittal Log"
    ws["A1"] = "RIVERSIDE DRIVE STREETSCAPE - SUBMITTAL LOG"
    ws["A2"] = "Project No. 24-117"
    ws["D2"] = "Updated 2026-09-28"
    ws.append([])
    ws.append(["Submittal No.", "Description", "Rev", "Received", "Status",
               "Reviewer", "Remarks"])
    ws.append(["32 16 00-001", "Concrete mix design", 0, "2026-09-02",
               "Approved", "J. Patel", None])
    ws.append(["32 16 00-003", "Detectable warning mats", 1, "2026-09-09",
               "Revise and Resubmit", "J. Patel", "Colour not shown"])
    # A sheet of numbers with no header row at all: the first row is taken,
    # and the text says so.
    nums = wb.create_sheet("Readings")
    nums.append(["Readings"])
    for r in ([0.5, 12, 1.1], [1.0, 15, 1.3], [1.5, 19, 1.2]):
        nums.append(r)
    wb.save(str(folder / "log.xlsx"))

    read = _call("read_text_file", {"path": "log.xlsx"})
    md = read["text"]
    assert ("Excel rows 4-6, columns A-G; row 4 is the header (the first row "
            "filled across the table, mostly with text).") in md
    assert ("Row 1, above the table: RIVERSIDE DRIVE STREETSCAPE - "
            "SUBMITTAL LOG") in md
    assert "Row 2, above the table: Project No. 24-117; Updated 2026-09-28" \
        in md
    assert ("| # | Submittal No. | Description | Rev | Received | Status | "
            "Reviewer | Remarks |") in md
    assert "| 6 | 32 16 00-003 | Detectable warning mats | 1 |" in md
    by_name = {s["name"]: s for s in read["sheets"]}
    assert by_name["Submittal Log"] == {
        "name": "Submittal Log", "rows": 3, "columns": 7, "header_row": 4,
        "rows_above_header": 2}
    # The numbers sheet: its one-cell title row is not filled across, and
    # the first row filled across is numbers, so there is no header row.
    assert ("row 1 is the header (no row filled across the table is mostly "
            "text, so the first row is taken)") in md
    assert by_name["Readings"]["header_row"] == 1
    assert "first row filled across" in read["note"]


def test_office_to_markdown_is_the_one_shared_reader(tmp_path):
    """open_document and read_text_file share office_text's reader."""
    from calc_package.docx_renderer import markdown_to_docx
    from funhouse_agent import office_text
    path = tmp_path / "m.docx"
    markdown_to_docx("# A\n\n- x\n- y\n", str(path))
    assert office_text.is_office_file(str(path))
    assert office_text.office_to_markdown(str(path)) == "# A\n\n- x\n- y\n"
    with pytest.raises(office_text.OfficeReadError, match=r"\.docx"):
        office_text.office_to_markdown(str(tmp_path / "old.doc"))
    (tmp_path / "bad.xlsx").write_bytes(b"not a workbook")
    with pytest.raises(office_text.OfficeReadError,
                       match="could not be read as an Excel workbook"):
        office_text.office_to_markdown(str(tmp_path / "bad.xlsx"))
    with pytest.raises(office_text.OfficeReadError):
        office_text.office_to_markdown(str(tmp_path / "notes.txt"))


def test_the_old_binary_formats_say_what_to_ask_for(folder):
    (folder / "old.doc").write_bytes(b"\xd0\xcf\x11\xe0\x00\x00binary")
    out = _call("read_text_file", {"path": "old.doc"})
    assert "error" in out and ".docx" in out["error"]
    (folder / "old.xls").write_bytes(b"\xd0\xcf\x11\xe0\x00\x00binary")
    assert ".xlsx" in _call("read_text_file", {"path": "old.xls"})["error"]


def test_a_damaged_docx_is_an_error_not_a_crash(folder):
    (folder / "broken.docx").write_bytes(b"PK\x03\x04 not really a zip")
    out = _call("read_text_file", {"path": "broken.docx"})
    assert "could not be read as a Word document" in out["error"]


def test_write_docx_signs_with_the_apps_author(folder, monkeypatch):
    _call("write_docx", {"path": str(folder / "a.docx"), "markdown": "Hi.",
                         "_author": "jdoe via GeotechStaffEngineer (AI draft)"})
    assert docx.Document(str(folder / "a.docx")).core_properties.author == \
        "jdoe via GeotechStaffEngineer (AI draft)"
    _call("write_docx", {"path": str(folder / "b.docx"), "markdown": "Hi."})
    assert docx.Document(str(folder / "b.docx")).core_properties.author == \
        "GeotechStaffEngineer"
    monkeypatch.setenv("GEOTECH_MARKUP_AUTHOR", "Acme Review Bot")
    _call("write_docx", {"path": str(folder / "c.docx"), "markdown": "Hi."})
    assert docx.Document(str(folder / "c.docx")).core_properties.author == \
        "Acme Review Bot"


def test_the_deep_tool_passes_the_signed_in_author(folder):
    from funhouse_agent.deep.tools import make_vision_tools
    tools = {t.name: t for t in make_vision_tools(
        markup_author="jdoe via GeotechStaffEngineer (AI draft)")}
    tools["write_docx"].invoke({"path": str(folder / "d.docx"),
                                "markdown": "Hi."})
    assert docx.Document(str(folder / "d.docx")).core_properties.author == \
        "jdoe via GeotechStaffEngineer (AI draft)"
    desc = tools["read_text_file"].description
    assert ".docx" in desc and ".xlsx" in desc and "write_docx" in desc
