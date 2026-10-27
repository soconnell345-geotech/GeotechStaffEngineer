"""Sheets per call (BACKLOG RI-116) through the graph and the cluster cell.

The reader's own behaviour is tested in ``test_lab_reader.py``
(``TestSheetsPerCall``). These check the plumbing around it:
``Budgets.lab_sheets_per_call`` reaches the lab reader, a split reading is
saved under its own name so a resumed run never takes one reading for the
other, and ``score_on_cluster(lab_sheets_per_call=...)`` lands on the ingest
stage's budgets. The sheets are made up
(``lab_fixtures.build_atterberg_run``).
"""

from __future__ import annotations

import json

import pytest

from report_ingest import graph
from report_ingest.graph import Budgets
from report_ingest.model import ReportRecord
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.test_cluster_scoring import (  # noqa: F401 - fixture
    _ingest_run, ingest_cluster,
)
from report_ingest.tests.test_lab_reader import _atterberg_run_reading

pytest.importorskip("pymupdf", reason="the fixtures are drawn with PyMuPDF")
pytest.importorskip("planlens.document.roles")

PAGES = list(range(7))
WINDOWS = [[0, 1, 2], [3, 4], [5, 6]]


@pytest.fixture
def run_doc(tmp_path):
    from planlens.document import open_document
    from report_ingest.tests.lab_fixtures import build_atterberg_run
    pdf, sheets = build_atterberg_run(len(PAGES))
    path = tmp_path / "run.pdf"
    path.write_bytes(pdf)
    doc = open_document(str(path))
    yield doc, sheets
    doc.close()


def _item():
    from planlens.document.roles import Item
    return Item(id="item_1", kind="lab_test", pages=list(PAGES),
                title="ATTERBERG LIMITS")


def _split_script(sheets):
    return [{"final": _atterberg_run_reading(sheets, w)} for w in WINDOWS]


def _whole_script(sheets):
    return [{"final": _atterberg_run_reading(sheets, PAGES)}]


class TestTheGraphHandsItToTheLabReader:
    def test_it_is_off_by_default(self):
        assert Budgets().lab_sheets_per_call is None

    def test_a_setting_below_one_is_refused_before_any_call(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="sheets per call"):
                Budgets(lab_sheets_per_call=bad)
        assert Budgets(lab_sheets_per_call=1).lab_sheets_per_call == 1

    def test_the_budget_splits_the_item(self, run_doc):
        doc, sheets = run_doc
        engine = FakeEngine(_split_script(sheets))
        blob = graph._read_lab(doc, PAGES, engine,
                               Budgets(lab_sheets_per_call=3), _item(), "RXX")
        assert engine.n_calls == 3
        assert blob["sheets_per_call"] == 3 and blob["windows"] == WINDOWS
        assert blob["model_calls"] == 3

    def test_none_reads_the_item_in_one_call_as_before(self, run_doc):
        doc, sheets = run_doc
        engine = FakeEngine(_whole_script(sheets))
        blob = graph._read_lab(doc, PAGES, engine, Budgets(), _item(), "RXX")
        assert engine.n_calls == 1
        assert "sheets_per_call" not in blob and "windows" not in blob


def _read_items(doc, engine, budgets, out):
    record = ReportRecord()
    qa: list = []
    graph._read_items(doc, engine, budgets, str(out), True, [], "RXX",
                      [_item()], {p: "lab_test" for p in PAGES}, None,
                      record, graph._Spend(), qa, [], "standard", [], [])
    return record


class TestASplitLabReadingIsSavedUnderItsOwnName:
    def test_a_resumed_run_never_takes_one_reading_for_the_other(
            self, run_doc, tmp_path):
        doc, sheets = run_doc
        out = tmp_path / "out"
        items = out / "items"

        whole = FakeEngine(_whole_script(sheets))
        _read_items(doc, whole, Budgets(), out)
        assert whole.n_calls == 1
        assert (items / "item_1.json").is_file()

        # The same folder with the split on: the whole item's reading is
        # not this one, so the item is read again, in three calls.
        split = FakeEngine(_split_script(sheets))
        record = _read_items(doc, split, Budgets(lab_sheets_per_call=3), out)
        assert split.n_calls == 3
        saved = json.loads((items / "item_1-per3.json").read_text("utf-8"))
        assert saved["sheets_per_call"] == 3
        assert saved["windows"] == WINDOWS
        assert [t.result.flow_curve for t in record.lab_tests] == \
            [[tuple(p) for p in s["flow_curve"]] for s in sheets]

        # Each reading is resumed under its own setting, at no cost.
        for budgets in (Budgets(lab_sheets_per_call=3), Budgets(),
                        Budgets(lab_sheets_per_call=12)):
            again = FakeEngine([])
            record = _read_items(doc, again, budgets, out)
            assert again.n_calls == 0, budgets
            assert len(record.lab_tests) == len(PAGES), budgets
        assert sorted(p.name for p in items.glob("*.json")) == \
            ["item_1-per3.json", "item_1.json"]


class TestTheClusterCellHandsItOn:
    @staticmethod
    def _budgets_seen(ingest_cluster, monkeypatch, **over):
        seen = []

        def capture(doc, engine, *args, **kwargs):
            seen.append(kwargs["budgets"])
            raise RuntimeError("made-up stop once the budgets were read")

        monkeypatch.setattr(graph, "ingest_report", capture)
        _ingest_run(ingest_cluster, **over)
        return seen

    def test_lab_sheets_per_call_reaches_the_ingest_budgets(
            self, ingest_cluster, monkeypatch):
        seen = self._budgets_seen(ingest_cluster, monkeypatch,
                                  lab_sheets_per_call=3)
        assert [b.lab_sheets_per_call for b in seen] == [3]

    def test_left_out_it_is_none(self, ingest_cluster, monkeypatch):
        seen = self._budgets_seen(ingest_cluster, monkeypatch)
        assert [b.lab_sheets_per_call for b in seen] == [None]
