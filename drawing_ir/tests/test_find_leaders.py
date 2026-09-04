"""find_leaders (B5-leaders) recall/precision on synthetic PDF-vector sheets.

Uses drawing_ir.tests.leader_fixtures (programmatic PyMuPDF sheets with
planted leaders + decoy geometry: a dimension line whose arrowheads are
geometrically near-identical to a leader arrowhead, stray unlabelled lines,
an isolated small closed shape, a numbered detail-bubble circle). Numbers are
measured and asserted honestly -- if the heuristic does not achieve perfect
precision at confidence 0, that is reported, not hidden.
"""

import pytest

fitz = pytest.importorskip("fitz")

from drawing_ir import from_pdf_vector, queries
from drawing_ir.tests.leader_fixtures import build_synthetic_leader_pdf

#: Confidence at/above which a proposal is treated as a confirmed leader for
#: recall/precision purposes. Genuine leaders measure ~0.94-0.96 in this
#: fixture family. Since Phase 2, a dimension arrowhead scores ~0.78
#: unfiltered (the Phase-1 ~0.32 came from an extension-line-steals-shaft
#: artifact, since fixed), so this threshold alone no longer separates the
#: two -- precision against dimensions is delivered by
#: find_leaders(exclude_dimensions=True), which the precision tests use.
CONFIDENCE_THRESHOLD = 0.5

TIP_TOLERANCE = 0.6  # PDF points


def _match_gt(tip_xy, ground_truth_leaders, tol=TIP_TOLERANCE):
    for gt in ground_truth_leaders:
        if (abs(tip_xy[0] - gt.tip_xy[0]) <= tol
                and abs(tip_xy[1] - gt.tip_xy[1]) <= tol):
            return gt
    return None


class TestRecallAndPrecision:
    def test_all_planted_leaders_recalled(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=6,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        # Precision at threshold is measured under the DOCUMENTED contract:
        # exclude_dimensions=True (Phase 2's find_dimensions claims the
        # dimension-decoy arrowheads; without it, a dimension line is
        # geometrically a one-arrow leader and legitimately scores high).
        proposals = queries.find_leaders(ir, exclude_dimensions=True)

        confirmed = [p for p in proposals if p["confidence"] >= CONFIDENCE_THRESHOLD]
        matched_gt_ids = set()
        for p in confirmed:
            gt_leader = _match_gt(p["tip_xy"], gt["leaders"])
            assert gt_leader is not None, (
                f"confirmed proposal at {p['tip_xy']} (conf {p['confidence']}) "
                f"does not correspond to any planted leader -- FALSE POSITIVE "
                f"above threshold {CONFIDENCE_THRESHOLD}")
            matched_gt_ids.add(id(gt_leader))

        recall = len(matched_gt_ids) / len(gt["leaders"])
        assert recall == 1.0, (
            f"recall {recall:.2f} ({len(matched_gt_ids)}/{len(gt['leaders'])}) "
            f"-- not every planted leader was recovered at confidence "
            f">= {CONFIDENCE_THRESHOLD}")

    def test_text_correctly_associated(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=6,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        proposals = [p for p in queries.find_leaders(ir, exclude_dimensions=True)
                    if p["confidence"] >= CONFIDENCE_THRESHOLD]
        for p in proposals:
            gt_leader = _match_gt(p["tip_xy"], gt["leaders"])
            assert gt_leader is not None
            assert p["text"] == gt_leader.label

    def test_decoys_do_not_clear_the_confidence_threshold(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=4,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        proposals = queries.find_leaders(ir, exclude_dimensions=True)
        confirmed = [p for p in proposals if p["confidence"] >= CONFIDENCE_THRESHOLD]
        # Every confirmed proposal must trace to a planted leader. Since
        # Phase 2, precision at the threshold is delivered by the
        # exclude_dimensions disambiguation (find_dimensions claims the
        # decoy arrowheads); WITHOUT it a dimension line scores ~0.78 as a
        # geometrically-valid one-arrow leader -- see the unthresholded
        # visibility test below, which keeps that honest.
        for p in confirmed:
            assert _match_gt(p["tip_xy"], gt["leaders"]) is not None
        n_planted = len(gt["leaders"])
        assert len(confirmed) == n_planted, (
            f"expected exactly {n_planted} confirmed proposals (the planted "
            f"leaders), got {len(confirmed)}")

    def test_known_false_positive_source_is_visible_unthresholded(self, tmp_path):
        """Honest documentation of the heuristic's limit: WITHOUT a confidence
        threshold, the dimension decoy's arrowheads DO surface as low/moderate
        confidence leader proposals (the documented false-positive source in
        find_leaders' docstring) -- this is measured, not asserted away."""
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=4,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        proposals = queries.find_leaders(ir)
        unmatched = [p for p in proposals
                    if _match_gt(p["tip_xy"], gt["leaders"]) is None]
        assert unmatched, (
            "expected the dimension-decoy arrowheads to surface as "
            "unconfirmed (low-confidence) proposals -- fixture may have "
            "changed; re-verify the false-positive-source claim")
        # ...and every one of them scores below the genuine leaders' floor.
        genuine_conf_floor = min(
            p["confidence"] for p in proposals
            if _match_gt(p["tip_xy"], gt["leaders"]) is not None)
        assert all(p["confidence"] < genuine_conf_floor for p in unmatched)

    def test_no_decoys_yields_no_false_positives(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3,
                                              include_decoys=False)
        ir = from_pdf_vector(path)
        proposals = queries.find_leaders(ir)
        assert len(proposals) == len(gt["leaders"])
        for p in proposals:
            assert _match_gt(p["tip_xy"], gt["leaders"]) is not None

    def test_min_confidence_filter(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=4,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        all_props = queries.find_leaders(ir)
        filtered = queries.find_leaders(ir, min_confidence=CONFIDENCE_THRESHOLD)
        assert filtered == [p for p in all_props if p["confidence"] >= CONFIDENCE_THRESHOLD]

    def test_sorted_by_confidence_descending(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=4,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        confs = [p["confidence"] for p in queries.find_leaders(ir)]
        assert confs == sorted(confs, reverse=True)

    def test_proposal_shape(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                              include_decoys=False)
        ir = from_pdf_vector(path)
        props = queries.find_leaders(ir)
        assert len(props) == 1
        p = props[0]
        for key in ("tip_xy", "tail_xy", "vertices", "arrowhead_id",
                   "shaft_id", "text", "text_id", "text_distance",
                   "confidence", "evidence", "proposal_only"):
            assert key in p
        assert p["proposal_only"] is True
        for k in ("alignment_score", "alignment_deg", "text_proximity_score",
                 "chain_simplicity_score", "n_shaft_vertices"):
            assert k in p["evidence"]

    def test_well_aligned_leader_scores_near_perfect_alignment(self, tmp_path):
        # The fixture generator always orients the arrowhead exactly along
        # the shaft's terminal segment -- a real correctness check on the
        # alignment math, not just a fixture artifact.
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                              include_decoys=False)
        ir = from_pdf_vector(path)
        p = queries.find_leaders(ir)[0]
        assert p["evidence"]["alignment_deg"] == pytest.approx(0.0, abs=0.5)
        assert p["evidence"]["alignment_score"] == pytest.approx(1.0, abs=0.02)

    def test_no_arrowhead_size_override_still_works(self, tmp_path):
        # The default max_arrowhead_size heuristic (median shaft length) must
        # be usable without the caller tuning anything.
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=2,
                                              include_decoys=False)
        ir = from_pdf_vector(path)
        assert len(queries.find_leaders(ir)) == 2
