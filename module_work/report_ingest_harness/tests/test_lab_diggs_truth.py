"""The DIGGS lab writer, gated on the 31 hand-truthed sheets. No model.

This is the measurement that matters for the lab writer, and it is
deterministic: every truth sheet becomes :class:`LabTest` records, the records
are written to DIGGS 2.6, the file is validated against the schema pydiggs
bundles, and every number in it has to come back through the app's own
readers. No model runs and nothing is sampled: if the writer drops a value,
this fails.

Thirty-one sheets across twelve kinds and at least seven laboratories,
including a French sieve sheet, a Spanish shear box, a 1991 typed triaxial
read optically, an Irish rock core, a Ugandan water-content table, a scanned
foundation-indicator sheet and three summary tables.

WHAT "PASSES" MEANS, exactly, because two of the sheets write no DIGGS at
all. A DIGGS result is a POSITION along a hole: a sheet that names a boring
but prints no depth, and a sheet that prints neither, have nowhere in the file
to be. Those are named in the writer's notes and kept in the record, and this
file asserts WHICH they are -- so the number cannot drift without a test
saying so.

It SKIPS when the gitignored raw folder is absent, which is everywhere but the
owner's machine. ``report_ingest/tests/test_diggs_lab.py`` walks the same path
on synthetic records, so CI still covers it.

PRIVACY. Report IDs and the identifiers a lab sheet printed. No assertion,
message or skip reason here names a project, a firm or a file.
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness import lab_truth_records as T
from report_ingest.diggs_writer import (
    DiggsWriteNotes, diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
)
from report_ingest.model import Project

pytestmark = pytest.mark.skipif(
    not T.truth_available(),
    reason="the hand-truthed lab sheets are gitignored and are not on this "
           "machine")

#: The two sheets that write no DIGGS, and why. A lab test is positioned
#: along a hole, so a sheet that prints no depth has nowhere to be written.
#: Listed rather than counted, so that a THIRD one appearing is a failure and
#: not a quietly smaller number.
NOTHING_WRITTEN = {"chemical__R28_p210", "compaction__R17_p136"}


def _truths():
    try:
        return T.load_truth()
    except FileNotFoundError:                     # pragma: no cover - skipped
        return []


def _ids():
    return [t["id"] for t in _truths()]


@pytest.fixture(scope="module")
def written():
    """``{sheet id: (tests, xml, notes)}`` for every truth sheet."""
    out = {}
    for truth in _truths():
        tests = T.lab_tests_from_truth(truth)
        notes = []
        xml = write_diggs(tests, Project(name="Report"),
                          document_id=truth["id"], notes=notes)
        out[truth["id"]] = (tests, xml, notes[0])
    return out


@pytest.mark.parametrize("sheet_id", _ids())
def test_every_truth_sheet_writes_valid_diggs(written, sheet_id):
    _tests, xml, _notes = written[sheet_id]
    ok, errors = diggs_schema_gate(xml)
    assert ok, f"{sheet_id}: " + "; ".join(e[:200] for e in errors[:3])


@pytest.mark.parametrize("sheet_id", _ids())
def test_every_truth_sheet_round_trips(written, sheet_id):
    tests, xml, _notes = written[sheet_id]
    ok, diffs = diggs_roundtrip_gate(xml, tests)
    assert ok, f"{sheet_id}: " + "; ".join(diffs[:6])


def test_the_truth_set_is_the_thirty_one_sheets_it_should_be(written):
    assert len(written) == 31
    reports = {sheet_id.split("__")[-1].split("_")[0]
               for sheet_id in written}
    assert len(reports) >= 10, "the sheets are meant to span many reports"


def test_the_records_carry_what_the_truth_states(written):
    """A guard on the CONVERTER, not the writer.

    Everything above would pass on thirty-one empty records. This is what
    says the records are not empty.
    """
    kinds = {}
    results = 0
    for tests, _xml, _notes in written.values():
        for test in tests:
            kinds[test.kind] = kinds.get(test.kind, 0) + 1
            if test.result is not None:
                results += 1
    assert sum(kinds.values()) >= 70, kinds
    assert results == sum(kinds.values()), "every record needs a result"
    # The vocabulary the truth README names, less the kinds no sheet carries.
    for kind in ("atterberg", "gradation", "swell_consolidation", "triaxial",
                 "direct_shear", "unconfined_rock", "compaction",
                 "moisture_content", "density", "organic_content",
                 "chemical", "summary_table"):
        assert kind in kinds, f"no {kind} record came out of the truth set"


def test_only_the_two_undepthed_sheets_write_nothing(written):
    """A sheet that prints no depth cannot be positioned, and says so."""
    empty = {sheet_id for sheet_id, (_t, _x, notes) in written.items()
             if notes.lab_tests == 0}
    assert empty == NOTHING_WRITTEN, (
        f"the set of sheets that write no DIGGS changed: {sorted(empty)}")
    for sheet_id in empty:
        _tests, _xml, notes = written[sheet_id]
        assert notes.skipped, f"{sheet_id} wrote nothing and said nothing"
        assert any("depth" in reason for reason in notes.skipped), \
            f"{sheet_id}: the reason should name the missing depth"


def test_the_file_says_which_holes_only_a_lab_sheet_named(written):
    """A hole invented for a lab test is recorded as one."""
    synthesised = sum(len(notes.synthesised)
                      for _t, _x, notes in written.values())
    assert synthesised >= 20, (
        "these sheets carry no logs, so nearly every hole in these files "
        "exists only because a lab sheet named it")


def test_both_depth_units_and_several_languages_are_represented(written):
    units = {test.depth_top.unit
             for tests, _x, _n in written.values()
             for test in tests if test.depth_top is not None}
    assert {"ft", "m"} <= units
    languages = {test.language for tests, _x, _n in written.values()
                 for test in tests if test.language}
    assert {"fr", "es"} <= languages


def test_a_value_printed_as_words_stays_words(written):
    """``<10`` is not the number ten anywhere along the path."""
    from report_ingest.model import ChemicalResult

    texts = []
    for tests, _xml, _notes in written.values():
        for test in tests:
            if isinstance(test.result, ChemicalResult):
                texts.extend(v for v in (test.result.chloride,
                                         test.result.sulfate,
                                         test.result.sulfides)
                             if isinstance(v, str))
    assert any(t.startswith("<") for t in texts), texts


# ---------------------------------------------------------------------------
# the scorer, against the same thirty-one sheets
# ---------------------------------------------------------------------------

def test_a_perfect_reading_scores_everything_on_every_sheet():
    """The floor under the scorecard.

    The converter turns each truth sheet into exactly the records a flawless
    reader would return, so the scorer must give them full marks. Anything
    less is the SCORER being wrong -- asking for a date, for a value in the
    wrong unit, for a number nested where it does not look -- and it would
    show up in every WP3 measurement as a reader that could not read.

    It is also the guard on the arithmetic: the totals below say how many
    things are being asked, so a scorer that quietly stopped asking would
    fail here rather than publish a flattering rate.
    """
    from report_ingest.lab_scoring import METRICS, score_record

    totals = {metric: [0, 0] for metric in METRICS}
    misses = []
    for truth in _truths():
        tests = T.lab_tests_from_truth(truth)
        score = score_record(truth, tests)
        for metric, got in score.scores.items():
            totals[metric][0] += got.found
            totals[metric][1] += got.total
            misses += [f"{truth['id']} {metric}: {m}" for m in got.misses]
    assert not misses, misses[:10]
    assert totals["kind"][1] >= 70
    assert totals["link"][1] >= 70
    assert totals["index"][1] >= 450
    assert totals["series"][1] >= 100
    assert totals["curve"][1] >= 25
