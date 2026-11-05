"""BACKLOG RI-94 end to end: a bound volume's holes are the report's holes.

The synthetic report has one document bound in at pages 15-18, holding one
boring (BH-1); the parent's own logs are B-1 and TP-1. Identified as an
appended prior report, BH-1 is the earlier report's and nothing in the
parent counts it. Identified as a volume (or a data report) of THIS
investigation, the parent's narrative is told so, and the counts the
parent checks and settles from its logs take BH-1 in, while BH-1 itself
stays in the child's record.
"""

from __future__ import annotations

import pytest

from report_ingest.graph import ingest_report
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.test_graph import (  # noqa: F401  (fixtures)
    bound_reader_turns, identity_turn, pdf, reader_turns, todays_readers,
    triage_turn, vision_turns,
)

pytest.importorskip("planlens.document.roles")


def _script(kind):
    return ([triage_turn()] + vision_turns() + [identity_turn(kind=kind)]
            + reader_turns() + bound_reader_turns())


def _run(pdf, tmp_path, kind, counts="off"):
    engine = FakeEngine(_script(kind))
    record = ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                           counts_from_logs=counts)
    return record, engine


def _brief(engine):
    (brief,) = [str(call["messages"]) for call in engine.calls
                if "REPORTS BOUND INSIDE THIS ONE" in str(call["messages"])]
    return brief


@pytest.mark.parametrize("kind", ["volume", "data_report"])
def test_a_bound_volume_is_told_as_this_investigation(pdf, tmp_path, kind):
    record, engine = _run(pdf, tmp_path, kind)

    brief = _brief(engine)
    assert "of THIS investigation" in brief
    assert "IS a previous investigation of this site" not in brief
    (child,) = record.bound_documents
    assert child.kind == kind
    # BH-1 stays in its own record.
    assert [inv.investigation_id for inv in record.investigations
            if inv.kind == "boring"] == ["B-1"]


def test_a_bound_prior_report_is_still_a_previous_investigation(pdf,
                                                                tmp_path):
    record, engine = _run(pdf, tmp_path, "appended_prior_report",
                          counts="settle")

    brief = _brief(engine)
    assert "IS a previous investigation of this site" in brief
    assert "of THIS investigation" not in brief
    assert record.general.boringCount == 1
    assert record.general.boringDictionary == ["B-1"]
    assert not [e for e in record.qa if e.where == "bound.current_volume"]


def test_a_bound_volumes_borings_are_counted_with_the_parents(pdf, tmp_path):
    record, _engine = _run(pdf, tmp_path, "volume", counts="settle")

    assert record.general.boringCount == 2
    assert record.general.boringDictionary == ["B-1", "BH-1"]
    assert record.general.testPitCount == 1
    (note,) = [e for e in record.qa if e.where == "bound.current_volume"]
    assert "bound1" in note.detail and note.kind == "note"


def test_a_bound_data_report_with_no_logs_adds_no_note(pdf, tmp_path,
                                                       monkeypatch):
    """A data report bound in with no exploration logs (a seismic design
    printout, say) counts nothing: no note, and the reconciler's own count
    check stands."""
    from report_ingest import graph

    monkeypatch.setattr(graph, "_current_volume_investigations",
                        lambda children, current: [])
    record, _engine = _run(pdf, tmp_path, "data_report")

    assert not [e for e in record.qa if e.where == "bound.current_volume"]
    (entry,) = [e for e in record.qa if e.kind == "count_mismatch"
                and e.where == "general.boringCount"]
    assert entry.values == ["narrative 4", "found 1"]


def test_the_count_check_counts_a_bound_volumes_borings(pdf, tmp_path):
    """The narrative states four borings; the parent read one and the bound
    volume one, so the check finds two (one, as a prior report)."""
    record, _engine = _run(pdf, tmp_path, "volume")
    (entry,) = [e for e in record.qa if e.kind == "count_mismatch"
                and e.where == "general.boringCount"]
    assert entry.values == ["narrative 4", "found 2"]

    prior, _engine = _run(pdf, tmp_path / "prior", "appended_prior_report")
    (entry,) = [e for e in prior.qa if e.kind == "count_mismatch"
                and e.where == "general.boringCount"]
    assert entry.values == ["narrative 4", "found 1"]
