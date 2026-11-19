"""Audit of the files a tester is handed (2026-11-18): a delivered workbook
holds what the agent wrote, or says what it could not hold."""

import io

import pytest

openpyxl = pytest.importorskip("openpyxl")

from aec_agent_core.xlsx_writer import build_workbook


def _cells(data):
    ws = openpyxl.load_workbook(io.BytesIO(data)).active
    return [[c.value for c in row] for row in ws.iter_rows()]


def test_a_line_break_written_as_br_in_a_table_is_a_line_break():
    md = ("| Item | Comment |\n|---|---|\n"
          "| 1 | first line<br>second line |\n"
          "| 2 | one<br/>two<BR />three |\n")
    data, _summary, _warnings = build_workbook(markdown=md)
    assert _cells(data)[1:] == [[1, "first line\nsecond line"],
                                [2, "one\ntwo\nthree"]]

