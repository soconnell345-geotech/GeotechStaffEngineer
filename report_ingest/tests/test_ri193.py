"""A divider set aside gets no ``skipped`` entry (BACKLOG RI-193).

Model first sets a page the describer calls a divider aside on purpose
(RI-177): it is in no group. The reconciler's check of pages no work item
read listed 19 of them on build f's MF-2 as "belong to no work item, so
nothing read them". No model is called; every page and title is made up.
"""

from __future__ import annotations

from types import SimpleNamespace

from report_ingest.graph import _ModelFirst, _without_dividers
from report_ingest.model import ReportRecord
from report_ingest.reconciler import reconcile


def _described(kind, title="MADE-UP TITLE"):
    return {"description": {"title": title, "kind": kind, "names": "",
                            "identifiers": [], "continues": "",
                            "summary": "made up"}}


DESCRIPTIONS = {3: _described("appendix divider page", "APPENDIX Q"),
                4: _described("boring log"), 5: _described("boring log")}
LABELS = {3: "boring_log", 4: "boring_log", 5: "boring_log"}


def _first():
    return _ModelFirst(engines={}, routes={}, descriptions=DESCRIPTIONS)


def test_a_divider_leaves_the_labels_the_check_reads():
    assert _without_dividers(LABELS, _first()) == {4: "boring_log",
                                                   5: "boring_log"}


def test_today_s_path_and_no_descriptions_keep_every_label():
    assert _without_dividers(LABELS, None) == LABELS
    assert _without_dividers(LABELS, _ModelFirst(engines={}, routes={})) \
        == LABELS


def test_the_reconciler_lists_no_divider_as_unread():
    log = SimpleNamespace(kind="boring_log", id="mf-log-4", pages=[4, 5])
    for labels, listed in ((LABELS, [[3]]),
                           (_without_dividers(LABELS, _first()), [])):
        record = ReportRecord()
        reconcile(record, labels=labels, items=[log])
        assert [e.pages for e in record.qa if e.kind == "skipped"
                and e.where.startswith("pages.")] == listed
