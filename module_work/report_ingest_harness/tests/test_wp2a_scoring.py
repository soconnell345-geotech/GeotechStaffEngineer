"""The WP2a scorer's own rules, pinned.

The corpus and the hand truth are gitignored, so the scorer is tested on the
synthetic log planlens ships with its answers, plus direct tests of the two
matching rules that decide most of the scorecard: how a blow record is
recognised, and how wide a sample's depth window is.
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness.measure_wp2a_loggrid import (
    DRIVEN_LENGTH_M, FT_PER_M, LAYER_TOL_M, SAMPLE_TOL_M, Score, _blows_match,
    _close, _to_m, score_one,
)


def test_a_blow_record_is_found_however_the_form_prints_it():
    # one cell: "5-9-12"
    assert _blows_match((5.0, 9.0, 12.0), [5, 9, 12])
    # one line per drive, read in page order
    assert _blows_match((-1.0, 5.0, 9.0, 12.0, 25.0, 56.0), [5, 9, 12])
    # four drives, the middle two of which are the N value
    assert _blows_match((4.0, 11.0, 10.0, 10.0), [4, 11, 10, 10])
    # a refusal, transcribed as text on the truth side
    assert _blows_match((23.0, 50.0, 4.0), [23, '50/4"'])


def test_a_blow_record_is_not_found_out_of_order_or_absent():
    assert not _blows_match((5.0, 12.0, 9.0), [5, 9, 12])
    assert not _blows_match((5.0, 9.0), [5, 9, 12])
    assert not _blows_match((), [5, 9, 12])
    assert not _blows_match((5.0, 9.0, 12.0), [])


def test_a_printed_number_may_be_rounded():
    assert _close(12.0, 11.8)
    assert _close(116.0, 115.6)
    assert not _close(12.0, 14.0)


def test_depths_are_compared_in_metres_whatever_the_log_prints():
    assert _to_m(10.0, "ft") == pytest.approx(3.048, abs=1e-3)
    assert _to_m(10.0, "m") == 10.0
    assert _to_m(None, "ft") is None


def test_the_tolerances_are_the_ones_the_plan_states():
    assert SAMPLE_TOL_M == 0.15
    assert LAYER_TOL_M == 0.30
    assert DRIVEN_LENGTH_M == pytest.approx(0.46)
    assert FT_PER_M == pytest.approx(3.2808, abs=1e-3)


def test_score_counts_and_keeps_its_misses():
    s = Score()
    s.add(True, "ignored")
    s.add(False, "the one that got away")
    assert (s.found, s.total) == (1, 2)
    assert s.rate == 0.5
    assert s.misses == ["the one that got away"]
    other = Score()
    other.add(True, "")
    s += other
    assert (s.found, s.total) == (2, 3)


def test_scoring_a_synthetic_log_end_to_end(tmp_path, monkeypatch):
    """The scorer on a log whose answers planlens itself states."""
    from planlens.document import open_document
    from planlens.testing import build_imperial_log

    gt = build_imperial_log()
    pdf = tmp_path / "log.pdf"
    pdf.write_bytes(gt.pdf)

    from module_work.report_ingest_harness import measure_wp2a_loggrid as M
    monkeypatch.setattr(M.C, "open_report",
                        lambda rid, di="auto", warn=True: open_document(
                            str(pdf)))

    truth = {
        "id": "RXX_p0",
        "pages": [0],
        "depth_unit": "ft",
        "fields": {"boring_id": "B-12", "hammer": "Automatic SPT Hammer"},
        "layers": [{"top": top} for top, _b, _d in gt.layers],
        "samples": [
            {"top": depth, "bottom": depth + 1.5,
             "blows": [int(x) for x in blows.split("-")],
             "n": int(n.split("=")[1])}
            for depth, blows, n in gt.samples
        ],
    }
    for sample, (depth, wc, duw) in zip(
            [s for s in truth["samples"]
             if s["top"] in [d for d, _w, _u in gt.index_tests]],
            gt.index_tests):
        sample["wc"] = int(wc)
        sample["duw"] = int(duw)

    out = score_one(truth)
    assert out.error is None
    assert out.ruler.found == 1 and out.unit.found == 1
    assert out.samples.rate == 1.0, out.samples.misses
    assert out.layers.rate == 1.0, out.layers.misses
    assert out.index.rate == 1.0, out.index.misses
    assert out.fields.rate == 1.0, out.fields.misses
    assert out.n_cells > 0
    assert out.n_unmatched_cells < out.n_cells
