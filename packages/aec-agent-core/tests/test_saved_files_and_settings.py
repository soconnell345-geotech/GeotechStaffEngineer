"""Text a document handed over, written to Word and Excel; a file name that
climbs out of the working folder; a setting that is not a number; and the
model-call budget's note on the turn (deployment review 2026-11-10).

* A form feed or another control character in the text (text copied out of a
  PDF carries them) lost the whole Word or Excel file.
* Excel text that opens with "=" was written as a formula.
* ``find_in_working_folder("../x")`` found a file outside the folder.
* ``GEOTECH_DOCUMENT_TOOLKITS=eight`` stopped the app starting.
* The primary agent's budget was noted nowhere, so the host started another
  pass with a new one.

No model, no network; every value is made up.
"""

import io
import os

import pytest

from aec_agent_core import _fileio, xlsx_writer
from aec_agent_core.deep import keep_work


# ---------------------------------------------------------------------------
# Word and Excel
# ---------------------------------------------------------------------------

def test_a_control_character_does_not_lose_the_workbook():
    openpyxl = pytest.importorskip("openpyxl")
    data, _summary, warnings = xlsx_writer.build_workbook(sheets=[{
        "name": "Comments",
        "rows": [["Sheet", "Comment"],
                 ["S-201", "Lap length\x0c not shown\x01"],
                 ["S-202", "= 2 x 150 mm"]]}])
    ws = openpyxl.load_workbook(io.BytesIO(data))["Comments"]
    assert ws["B2"].value == "Lap length not shown"
    # a note that opens with "=" is the text that was given, not a formula
    assert ws["B3"].value == "= 2 x 150 mm"
    assert ws["B3"].data_type == "s"
    assert any("1 cell(s) open with '='" in w and "not formulas" in w
               for w in warnings)


def test_a_sheet_with_no_such_text_says_nothing_about_it():
    pytest.importorskip("openpyxl")
    _data, _summary, warnings = xlsx_writer.build_workbook(sheets=[{
        "name": "Bars", "rows": [["Mark", "Length (mm)"], ["B1", 1250]]}])
    assert not any("open with '='" in w for w in warnings)


def test_a_control_character_does_not_lose_the_word_file(tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("markdown_it")
    from aec_agent_core.docx_renderer import markdown_to_docx
    out = tmp_path / "review.docx"
    markdown_to_docx(
        "# Review notes\x0c\n\nCover to the bars\x0b is not shown.\x00\n",
        str(out), title="Review\x01 notes")
    text = "\n".join(p.text for p in docx.Document(str(out)).paragraphs)
    assert "Cover to the bars" in text and "is not shown." in text
    assert not any(ord(ch) < 32 and ch not in "\t\n\r" for ch in text)


# ---------------------------------------------------------------------------
# The working folder
# ---------------------------------------------------------------------------

def test_a_name_that_climbs_out_of_the_working_folder_is_not_found(tmp_path):
    folder = tmp_path / "conv" / "files"
    folder.mkdir(parents=True)
    (folder / "plan.pdf").write_bytes(b"%PDF made up")
    (tmp_path / "conv" / "meta.json").write_text("{}", encoding="utf-8")
    other = tmp_path / "other" / "files"
    other.mkdir(parents=True)
    (other / "theirs.pdf").write_bytes(b"%PDF somebody else's")
    with _fileio.working_dir_bound(str(folder)):
        assert _fileio.find_in_working_folder("plan.pdf") == \
            os.path.abspath(str(folder / "plan.pdf"))
        assert _fileio.find_in_working_folder("../meta.json") is None
        assert _fileio.find_in_working_folder(
            "../../other/files/theirs.pdf") is None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_a_setting_that_is_not_a_number_is_the_default(monkeypatch):
    pytest.importorskip("planlens.tools")
    from aec_agent_core import document_tools
    name = "GEOTECH_DOCUMENT_TOOLKITS"
    monkeypatch.setenv(name, "eight")
    assert document_tools._number_setting(name, 8) == 8.0
    monkeypatch.setenv(name, " 12 ")
    assert document_tools._number_setting(name, 8) == 12.0
    monkeypatch.setenv(name, "")
    assert document_tools._number_setting(name, 8) == 8.0
    monkeypatch.delenv(name)
    assert document_tools._number_setting(name, 8) == 8.0


# ---------------------------------------------------------------------------
# The model-call budget
# ---------------------------------------------------------------------------

def test_the_primary_agents_budget_is_noted_on_the_turn():
    with keep_work.turn_scope() as scope:
        assert scope.budget_reached is False
        keep_work.note_budget_reached()
        assert scope.budget_reached is True
    keep_work.note_budget_reached()           # outside a turn: nothing to note


def test_a_helpers_budget_is_noted_on_its_record_not_on_the_turn():
    rec = keep_work.HelperRecord("t1", "general-purpose", "read sheet S-201")
    with keep_work.turn_scope() as scope:
        token = keep_work._HELPER.set(rec)
        try:
            keep_work.note_budget_reached()
        finally:
            keep_work._HELPER.reset(token)
        assert rec.budget_reached is True
        assert scope.budget_reached is False


# ---------------------------------------------------------------------------
# From the independent review of the fixes above
# ---------------------------------------------------------------------------

def test_a_control_character_in_a_sheet_name_or_a_table_row_is_dropped():
    openpyxl = pytest.importorskip("openpyxl")
    data, summary, _warnings = xlsx_writer.build_workbook(
        sheets=[{"name": "Bar\x0c list", "rows": [["Mark"], ["B1"]]}],
        markdown=("| Sheet | Comment |\n|---|---|\n"
                  "| S-201 | lap\x0clength |\n| S-202 | cover |\n"))
    wb = openpyxl.load_workbook(io.BytesIO(data))
    assert wb.sheetnames[0] == "Bar list"
    # the form feed did not end the table at its row: both rows are there
    assert summary[1]["rows"] == 3
    assert wb[wb.sheetnames[1]]["B2"].value == "lap length"


def test_characters_xml_cannot_hold_are_dropped_from_both_files(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    docx = pytest.importorskip("docx")
    pytest.importorskip("markdown_it")
    from aec_agent_core.docx_renderer import markdown_to_docx
    odd = "cover" + chr(0xFFFE) + " 40" + chr(0xD800) + " mm" + chr(0xFFFF)
    data, _summary, _warnings = xlsx_writer.build_workbook(
        sheets=[{"name": "Notes", "rows": [["Note"], [odd]]}])
    assert openpyxl.load_workbook(io.BytesIO(data))["Notes"]["A2"].value == \
        "cover 40 mm"
    out = tmp_path / "notes.docx"
    markdown_to_docx("Check: " + odd + "\n", str(out), title=2024)
    text = "\n".join(p.text for p in docx.Document(str(out)).paragraphs)
    assert "cover" in text and "40" in text and "mm" in text


def test_a_setting_of_inf_or_nan_is_the_default(monkeypatch):
    pytest.importorskip("planlens.tools")
    from aec_agent_core import document_tools
    name = "GEOTECH_DOCUMENT_TOOLKITS"
    for value in ("inf", "-inf", "nan"):
        monkeypatch.setenv(name, value)
        assert document_tools._number_setting(name, 8) == 8.0, value
