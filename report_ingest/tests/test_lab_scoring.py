"""The lab scorer: what it asks of a reading, and what it refuses to ask.

A scorer is the one piece of a measurement that nobody checks against
anything, so these tests check it against the two failures that would make a
scorecard lie: asking for something a reader could not have known (which
turns a good reader into a bad number) and asking for nothing (which turns a
bad reader into a good one).

The truth dicts here are written in the same shape as the private hand-truth
and carry none of its content.
"""

from __future__ import annotations

import pytest

from report_ingest.lab_scoring import (
    DEPTH_TOL_M, EXACT_TOL, METRICS, PASSING_TOL, expectations_for,
    score_record, score_tables, split_unit, table_numbers,
)
from report_ingest.model import (
    AtterbergResult, ChemicalResult, GradationResult, LabTest, Quantity,
    SievePoint, StrengthResult, StrengthSpecimen, SummaryRow,
    SummaryTableResult,
)


def q(value, unit):
    return Quantity(value=value, unit=unit)


GRADING_TRUTH = {
    "id": "gradation__R99_p12",
    "kind": "gradation",
    "depth_unit": "ft",
    "tests": [{
        "investigation_id": "B-7",
        "depth_top": 10.0,
        "ll": 31, "pl": 19, "pi": 12,
        "gravel_pct": 13, "sand_pct": 44, "fines_pct": 43,
        "d60": 0.42,
        "percent_finer": {"3\"": 100, "No. 4": 87, "No. 200": 43},
        "date": "14 March 2025",
        "description": "CLAYEY SAND (SC)",
    }],
}

SHEAR_TRUTH = {
    "id": "direct_shear__R99_p20",
    "kind": "direct_shear",
    "depth_unit": "ft",
    "tests": [{
        "investigation_id": "B-2",
        "depth_top": 2.5,
        "c_psf": 539, "phi_deg": 30,
        "points_normal_psf_vs_shear_psf": [[500, 850], [1000, 1085],
                                           [2000, 1720]],
        "curve_digitised": True,
        "tolerance_psf": 25,
    }],
}


def _grading_record(**over):
    values = dict(ll=31.0, pl=19.0, pi=12.0, gravel=13.0, sand=44.0,
                  fines=43.0, d60=0.42, passing=[100.0, 87.0, 43.0],
                  boring="B-7", depth=10.0)
    values.update(over)
    grading = LabTest(
        kind="gradation", investigation_id=values["boring"],
        depth_top=q(values["depth"], "ft"),
        result=GradationResult(
            gravel_percent=values["gravel"], sand_percent=values["sand"],
            fines_percent=values["fines"], d60=q(values["d60"], "mm"),
            percent_passing=[
                SievePoint(percent_passing=values["passing"][0],
                           size=q(75.0, "mm"), sieve="3 in"),
                SievePoint(percent_passing=values["passing"][1],
                           size=q(4.75, "mm"), sieve="No. 4"),
                SievePoint(percent_passing=values["passing"][2],
                           size=q(0.075, "mm"), sieve="No. 200")]))
    limits = LabTest(
        kind="atterberg", investigation_id=values["boring"],
        depth_top=q(values["depth"], "ft"),
        result=AtterbergResult(ll=values["ll"], pl=values["pl"],
                               pi=values["pi"]))
    return [grading, limits]


class TestWhatItAsksFor:
    def test_a_perfect_reading_scores_everything(self):
        score = score_record(GRADING_TRUTH, _grading_record())
        assert score.total.found == score.total.total
        assert score.total.total > 8
        for metric in ("kind", "link", "index", "series"):
            assert score.scores[metric].total > 0, metric

    def test_the_kind_is_scored_against_the_kinds_that_came_back(self):
        score = score_record(GRADING_TRUTH, _grading_record()[1:])
        assert score.scores["kind"].found == 0
        assert "gradation" in score.scores["kind"].misses[0]

    def test_a_wrong_boring_breaks_the_link_and_everything_under_it(self):
        score = score_record(GRADING_TRUTH, _grading_record(boring="B-8"))
        assert score.scores["link"].found == 0
        assert score.scores["index"].found == 0

    def test_a_depth_inside_the_tolerance_still_links(self):
        inside = 10.0 + (DEPTH_TOL_M * 0.9) / 0.3048
        score = score_record(GRADING_TRUTH, _grading_record(depth=inside))
        assert score.scores["link"].found == 1
        outside = 10.0 + (DEPTH_TOL_M * 3) / 0.3048
        score = score_record(GRADING_TRUTH, _grading_record(depth=outside))
        assert score.scores["link"].found == 0

    def test_an_index_value_is_exact(self):
        score = score_record(GRADING_TRUTH, _grading_record(ll=31.0))
        assert score.scores["index"].misses == []
        score = score_record(GRADING_TRUTH, _grading_record(ll=32.0))
        assert any("ll" in miss for miss in score.scores["index"].misses)

    def test_a_grading_is_scored_within_a_percent(self):
        near = [100.0, 87.0 + PASSING_TOL * 0.5, 43.0]
        score = score_record(GRADING_TRUTH, _grading_record(passing=near))
        assert score.scores["series"].found == 3
        far = [100.0, 87.0 + PASSING_TOL * 3, 43.0]
        score = score_record(GRADING_TRUTH, _grading_record(passing=far))
        assert score.scores["series"].found == 2

    def test_a_curve_is_scored_at_the_tolerance_its_truth_file_states(self):
        good = LabTest(
            kind="direct_shear", investigation_id="B-2",
            depth_top=q(2.5, "ft"),
            result=StrengthResult(
                kind="direct_shear", c=q(539.0, "psf"), phi_deg=30.0,
                points=[]))
        from report_ingest.model import ShearPoint
        good.result.points = [
            ShearPoint(x=q(500.0, "psf"), y=q(850.0, "psf")),
            ShearPoint(x=q(1000.0, "psf"), y=q(1085.0, "psf")),
            ShearPoint(x=q(2000.0, "psf"), y=q(1720.0, "psf"))]
        score = score_record(SHEAR_TRUTH, [good])
        assert score.scores["curve"].found == 3
        # 25 psf is the sheet's own tolerance; 200 psf is not a reading.
        good.result.points[0].y = q(1050.0, "psf")
        score = score_record(SHEAR_TRUTH, [good])
        assert score.scores["curve"].found == 2

    def test_nothing_read_scores_nothing_rather_than_scoring_nothing(self):
        """A reader that returns nothing must score zero, not zero out of
        zero."""
        score = score_record(GRADING_TRUTH, [])
        assert score.total.total > 0
        assert score.total.found == 0


class TestWhatItRefusesToAsk:
    def test_a_date_is_not_a_measurement(self):
        expect = expectations_for(GRADING_TRUTH)[0]
        assert "date" not in expect.index
        assert not any(name.startswith("date") for name in expect.index)

    def test_a_description_is_not_a_measurement(self):
        expect = expectations_for(GRADING_TRUTH)[0]
        assert "description" not in expect.index

    def test_the_link_fields_are_not_asked_for_twice(self):
        expect = expectations_for(GRADING_TRUTH)[0]
        for name in ("investigation_id", "depth_top", "depth_bottom"):
            assert name not in expect.index
        assert expect.investigation_id == "B-7"
        assert expect.depth_m == pytest.approx(10.0 * 0.3048)

    def test_a_tolerance_is_not_a_measurement(self):
        expect = expectations_for(SHEAR_TRUTH)[0]
        assert not any("tolerance" in name for name in expect.index)

    def test_a_unit_suffix_is_read_longest_first(self):
        """``bulk_density_g_cm3`` is a density, not a volume."""
        assert split_unit("bulk_density_g_cm3") == ("bulk_density", "g/cm3")
        assert split_unit("resistivity_kohm_cm") == ("resistivity",
                                                     "kohm-cm")
        assert split_unit("c_ton_ft2") == ("c", "tsf")
        assert split_unit("ucs_MPa") == ("ucs", "MPa")
        assert split_unit("cu") == ("cu", "")

    def test_a_d_value_with_no_unit_in_its_key_is_millimetres(self):
        expect = expectations_for(GRADING_TRUTH)[0]
        value, unit, printed = expect.index["d60"]
        assert printed == 0.42
        assert unit == "m" and value == pytest.approx(0.00042)

    def test_a_nested_specimen_is_asked_for(self):
        """A triaxial prints its specimens as a list, and a scorer that read
        only the top level would ask nothing of the whole sheet."""
        truth = {
            "id": "triaxial__R99_p8", "kind": "triaxial", "depth_unit": "m",
            "tests": [{
                "investigation_id": "BH-1", "depth_top": 4.0,
                "specimens": [
                    {"n": 1, "peak_deviator_kPa": 185,
                     "strain_at_peak_pct": 10.7},
                    {"n": 2, "peak_deviator_kPa": 195,
                     "strain_at_peak_pct": 9.7}]}]}
        expect = expectations_for(truth)[0]
        assert len(expect.index) == 4
        record = LabTest(
            kind="triaxial", investigation_id="BH-1", depth_top=q(4.0, "m"),
            result=StrengthResult(
                kind="triaxial",
                specimens=[
                    StrengthSpecimen(specimen_id="1",
                                     peak_deviator=q(185.0, "kPa"),
                                     strain_at_peak_percent=10.7),
                    StrengthSpecimen(specimen_id="2",
                                     peak_deviator=q(195.0, "kPa"),
                                     strain_at_peak_percent=9.7)]))
        score = score_record(truth, [record])
        assert score.scores["index"].found == 4

    def test_a_value_kept_as_printed_in_fields_counts_as_recovered(self):
        """A sheet prints things the record has no typed field for, and
        ``fields`` is where they go rather than nowhere."""
        truth = {
            "id": "triaxial__R99_p9", "kind": "triaxial", "depth_unit": "m",
            "tests": [{"investigation_id": "BH-1", "depth_top": 4.0,
                       "back_pressure_kPa": 100, "membranes": 2}]}
        record = LabTest(
            kind="triaxial", investigation_id="BH-1", depth_top=q(4.0, "m"),
            result=StrengthResult(kind="triaxial"),
            fields={"back_pressure_kPa": "100", "membranes": "2"})
        score = score_record(truth, [record])
        assert score.scores["index"].found == 2


class TestTheTablesBaseline:
    def test_the_tables_are_not_scored_on_kind_or_link(self, tmp_path):
        doc = _sheet(tmp_path)
        try:
            score = score_tables(GRADING_TRUTH, doc, [0])
        finally:
            doc.close()
        assert "kind" not in score.scores
        assert "link" not in score.scores
        assert score.scores["index"].total > 0

    def test_a_number_on_the_page_counts_and_one_that_is_not_does_not(
            self, tmp_path):
        doc = _sheet(tmp_path)
        try:
            pool = table_numbers(doc, [0])
            score = score_tables(GRADING_TRUTH, doc, [0])
        finally:
            doc.close()
        assert 43.0 in pool and 87.0 in pool
        assert score.scores["series"].found == 3
        # The D60 is not in the fixture's table, so the baseline misses it.
        assert any("d60" in miss for miss in score.scores["index"].misses)


def _sheet(tmp_path):
    """A page carrying the grading truth's tabulated values and no more."""
    fitz = pytest.importorskip("fitz")
    from planlens.document import open_document
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rows = [("SIEVE", "PERCENT FINER"), ("3 in", "100"), ("No. 4", "87"),
            ("No. 200", "43")]
    y = 100.0
    for row in rows:
        page.insert_text(fitz.Point(63, y + 11), row[0], fontsize=9)
        page.insert_text(fitz.Point(163, y + 11), row[1], fontsize=9)
        y += 16
    for x in (60, 160, 260):
        page.draw_line(fitz.Point(x, 100), fitz.Point(x, y), width=0.6)
    for r in range(len(rows) + 1):
        page.draw_line(fitz.Point(60, 100 + r * 16),
                       fitz.Point(260, 100 + r * 16), width=0.6)
    page.insert_text(fitz.Point(300, 120), "LL 31   PL 19   PI 12",
                     fontsize=9)
    rows2 = [("PROPERTY", "VALUE"), ("Gravel", "13"), ("Sand", "44"),
             ("Fines", "43")]
    y2 = 200.0
    for row in rows2:
        page.insert_text(fitz.Point(303, y2 + 11), row[0], fontsize=9)
        page.insert_text(fitz.Point(403, y2 + 11), row[1], fontsize=9)
        y2 += 16
    for x in (300, 400, 480):
        page.draw_line(fitz.Point(x, 200), fitz.Point(x, y2), width=0.6)
    for r in range(len(rows2) + 1):
        page.draw_line(fitz.Point(300, 200 + r * 16),
                       fitz.Point(480, 200 + r * 16), width=0.6)
    pdf = doc.tobytes()
    doc.close()
    path = tmp_path / "sheet.pdf"
    path.write_bytes(pdf)
    return open_document(str(path))


def test_the_metric_list_is_what_the_scorecards_print():
    assert METRICS == ("kind", "link", "index", "series", "curve")
    assert EXACT_TOL < PASSING_TOL
