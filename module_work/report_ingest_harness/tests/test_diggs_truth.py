"""The DIGGS writer, gated on the fifteen hand-truthed logs. No model.

This is the measurement that matters for the writer, and it is deterministic:
every truth file becomes :class:`Investigation` records, the records are
written to DIGGS 2.6, the file is validated against the bundled schema and
read back with the app's own ``parse_diggs``, and every depth, layer, N value,
drive, water level and index value has to come back inside the tolerances the
writer publishes.

Fifteen logs across fifteen reports and at least eight form templates,
including a metric log, an imperial log, a scanned log read optically, a
Spanish sheet carrying three borings at once, two test pits and a log whose
last layer has no printed base. No model runs and nothing is sampled: if the
writer drops a value, this fails.

It SKIPS when the gitignored raw folder is absent, which is everywhere but
the owner's machine. ``report_ingest/tests/test_diggs_writer.py`` walks the
same path on a synthetic investigation, so CI still covers it.

PRIVACY. Report IDs and boring identifiers only. No assertion, message or
skip reason here names a project, a firm or a file.
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness import truth_records as T
from report_ingest.diggs_writer import (
    diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
)
from report_ingest.model import Project

pytestmark = pytest.mark.skipif(
    not T.truth_available(),
    reason="the hand-truthed logs are gitignored and are not on this machine")


def _truths():
    try:
        return T.load_truth()
    except FileNotFoundError:                     # pragma: no cover - skipped
        return []


def _ids():
    return [t["id"] for t in _truths()]


@pytest.fixture(scope="module")
def written():
    """``{log id: (investigations, xml)}`` for every truth log."""
    out = {}
    for truth in _truths():
        investigations = T.investigations_from_truth(truth)
        out[truth["id"]] = (
            investigations,
            write_diggs(investigations, Project(name="Report"),
                        document_id=truth["id"]))
    return out


@pytest.mark.parametrize("log_id", _ids())
def test_every_truth_log_writes_valid_diggs(written, log_id):
    _investigations, xml = written[log_id]
    ok, errors = diggs_schema_gate(xml)
    assert ok, f"{log_id}: " + "; ".join(e[:200] for e in errors[:3])


@pytest.mark.parametrize("log_id", _ids())
def test_every_truth_log_round_trips(written, log_id):
    investigations, xml = written[log_id]
    ok, diffs = diggs_roundtrip_gate(xml, investigations)
    assert ok, f"{log_id}: " + "; ".join(diffs[:6])


def test_the_truth_set_is_the_fifteen_logs_it_should_be(written):
    assert len(written) == 15
    reports = {log_id.split("_")[0] for log_id in written}
    assert len(reports) == 15, "one log per report was the intent"


def test_the_records_carry_what_the_truth_states(written):
    """A guard on the CONVERTER, not the writer.

    Everything above would pass on fifteen empty records. This is what says
    the records are not empty.
    """
    totals = {"layers": 0, "samples": 0, "spt": 0, "water": 0}
    for investigations, _xml in written.values():
        for inv in investigations:
            assert inv.investigation_id, "every record needs an identifier"
            totals["layers"] += len(inv.layers)
            totals["samples"] += len(inv.samples)
            totals["spt"] += len(inv.spt)
            totals["water"] += len(inv.water)
    assert totals["layers"] >= 60
    assert totals["samples"] >= 70
    assert totals["spt"] >= 50
    assert totals["water"] >= 15


def test_a_sheet_of_several_borings_becomes_several_records(written):
    """The Spanish tabular sheet lists three borings on one page."""
    counts = {log_id: len(investigations)
              for log_id, (investigations, _xml) in written.items()}
    assert max(counts.values()) >= 3, counts


def test_both_depth_units_are_represented(written):
    units = {inv.depth_unit
             for investigations, _xml in written.values()
             for inv in investigations}
    assert {"ft", "m"} <= units
