"""Tests for the Phase-2 composition family: find_dimensions,
find_title_block, find_bubble_callouts, find_revision_clouds, and the
leader<->dimension disambiguation. Recall AND precision are measured on
fixtures that plant decoys from the *other* families (a leader on the
dimension sheet, a dimension on the leader sheet, arrowheads on the cloud
sheet), so the numbers reflect discriminating power, not easy fixtures.
"""

from __future__ import annotations

import math

import pytest

fitz = pytest.importorskip("fitz")

from drawing_ir import from_pdf_vector, queries
from drawing_ir.tests.construct_fixtures import (
    build_synthetic_bubble_pdf, build_synthetic_cloud_pdf,
    build_synthetic_dimension_pdf, build_synthetic_title_block_pdf,
)
from drawing_ir.tests.leader_fixtures import build_synthetic_leader_pdf


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# find_dimensions
# ---------------------------------------------------------------------------

class TestFindDimensions:
    @pytest.fixture()
    def sheet(self, tmp_path):
        path, gt = build_synthetic_dimension_pdf(tmp_path)
        return from_pdf_vector(path), gt

    def test_recall_all_planted_dimensions(self, sheet):
        ir, gt = sheet
        props = queries.find_dimensions(ir, min_confidence=0.5)
        assert len(props) >= len(gt["dimensions"])
        for d in gt["dimensions"]:
            ends = {tuple(round(v, 1) for v in d["end_a"]),
                    tuple(round(v, 1) for v in d["end_b"])}
            match = [p for p in props
                     if {tuple(round(v, 1) for v in p["end_a_xy"]),
                         tuple(round(v, 1) for v in p["end_b_xy"])} == ends]
            assert match, f"planted dimension {d['value']} not proposed"
            assert match[0]["text"] == d["value"]
            assert match[0]["proposal_only"] is True
            assert len(match[0]["arrowhead_ids"]) == 2

    def test_precision_leader_decoy_not_a_dimension(self, sheet):
        ir, gt = sheet
        props = queries.find_dimensions(ir, min_confidence=0.5)
        # Exactly the two planted dimensions at the working threshold — the
        # true-leader decoy (one arrowhead) must not appear.
        assert len(props) == len(gt["dimensions"])

    def test_extension_lines_found_both_ends(self, sheet):
        ir, _gt = sheet
        props = queries.find_dimensions(ir, min_confidence=0.5)
        for p in props:
            assert p["evidence"]["n_extension_ends"] == 2
            assert len(p["extension_line_ids"]) >= 2

    def test_vertical_and_horizontal_both_detected(self, sheet):
        ir, _gt = sheet
        props = queries.find_dimensions(ir, min_confidence=0.5)
        angles = sorted(round(p["angle_deg"]) % 180 for p in props)
        assert 0 in angles and 90 in angles


class TestLeaderDimensionDisambiguation:
    """THE Phase-1 false-positive close-out: dimension decoys on the leader
    sheet are claimed by find_dimensions and excluded from find_leaders."""

    @pytest.fixture()
    def leader_sheet(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3,
                                              include_decoys=True)
        return from_pdf_vector(path), gt

    def test_dimension_decoy_is_classified_as_dimension(self, leader_sheet):
        ir, gt = leader_sheet
        props = queries.find_dimensions(ir, min_confidence=0.3)
        assert len(props) >= gt["n_decoy_dimensions"]

    def test_exclude_dimensions_keeps_all_true_leaders(self, leader_sheet):
        ir, gt = leader_sheet
        props = queries.find_leaders(ir, exclude_dimensions=True)
        # Full recall of planted leaders survives the exclusion.
        for L in gt["leaders"]:
            match = [p for p in props
                     if _dist(p["tip_xy"], L.tip_xy) < 1.0]
            assert match, f"true leader '{L.label}' lost to dimension filter"

    def test_exclude_dimensions_removes_dimension_arrowheads(
            self, leader_sheet):
        ir, gt = leader_sheet
        unfiltered = queries.find_leaders(ir)
        filtered = queries.find_leaders(ir, exclude_dimensions=True)
        dims = queries.find_dimensions(ir, min_confidence=0.5)
        claimed = {a for d in dims for a in d["arrowhead_ids"]}
        assert claimed, "fixture's dimension decoy was not detected"
        assert all(p["arrowhead_id"] not in claimed for p in filtered)
        # The unfiltered run surfaces at least one claimed arrowhead
        # (the documented Phase-1 false-positive source), proving the
        # filter removed something real.
        assert any(p["arrowhead_id"] in claimed for p in unfiltered)


# ---------------------------------------------------------------------------
# find_title_block
# ---------------------------------------------------------------------------

class TestFindTitleBlock:
    @pytest.fixture()
    def sheet(self, tmp_path):
        path, gt = build_synthetic_title_block_pdf(tmp_path)
        return from_pdf_vector(path), gt

    def test_top_proposal_is_the_corner_block(self, sheet):
        ir, gt = sheet
        props = queries.find_title_block(ir)
        assert props, "no title block proposed"
        top = props[0]
        exp = gt["block_bbox_ir"]
        for got, want in zip(top["region_bbox"], exp):
            assert got == pytest.approx(want, abs=2.0)
        assert top["proposal_only"] is True
        assert top["n_nested_rects"] >= gt["n_cells"]

    def test_text_payload_contains_sheet_metadata(self, sheet):
        ir, gt = sheet
        top = queries.find_title_block(ir)[0]
        contents = [t["content"] for t in top["texts"]]
        for expected in gt["texts"]:
            assert expected in contents
        # Body text stays out of the block payload.
        assert "PLAN VIEW" not in contents
        assert "PILE SCHEDULE" not in contents

    def test_mid_sheet_box_does_not_win(self, sheet):
        ir, gt = sheet
        props = queries.find_title_block(ir)
        top = props[0]
        # The schedule box (mid-sheet) must not be the top proposal.
        assert "PILE SCHEDULE" not in [t["content"] for t in top["texts"]]

    def test_fallback_text_cluster_when_no_rectangles(self, tmp_path):
        doc = fitz.open()
        page = doc.new_page(width=900, height=700)
        for i, t in enumerate(["SHEET C-05", "REV 0", "ACME", "NTS"]):
            page.insert_text(fitz.Point(780, 600 + i * 16), t, fontsize=8)
        p = str(tmp_path / "no_rects.pdf")
        doc.save(p)
        doc.close()
        ir = from_pdf_vector(p)
        props = queries.find_title_block(ir)
        assert props
        assert props[0]["evidence"]["path"] == "text_cluster_fallback"
        assert props[0]["confidence"] <= 0.4
        contents = [t["content"] for t in props[0]["texts"]]
        assert "SHEET C-05" in contents


# ---------------------------------------------------------------------------
# find_bubble_callouts
# ---------------------------------------------------------------------------

class TestFindBubbleCallouts:
    @pytest.fixture()
    def sheet(self, tmp_path):
        path, gt = build_synthetic_bubble_pdf(tmp_path)
        return from_pdf_vector(path), gt

    def test_recall_all_planted_bubbles(self, sheet):
        ir, gt = sheet
        props = queries.find_bubble_callouts(ir, min_confidence=0.6)
        for b in gt["bubbles"]:
            match = [p for p in props
                     if _dist(p["center_xy"], b["center"]) < 2.0]
            assert match, f"planted bubble '{b['text']}' not proposed"
            assert match[0]["text"] == b["text"]
            assert match[0]["radius"] == pytest.approx(b["radius"], rel=0.05)

    def test_kinds_classified(self, sheet):
        ir, gt = sheet
        props = queries.find_bubble_callouts(ir, min_confidence=0.6)
        by_text = {p["text"]: p for p in props}
        assert by_text["3"]["kind"] == "keynote"
        assert by_text["A"]["kind"] == "grid_bubble"
        assert by_text["A"]["attached_line_ids"]
        assert by_text["5"]["kind"] == "detail_callout"

    def test_precision_decoys_stay_below_threshold(self, sheet):
        ir, gt = sheet
        props = queries.find_bubble_callouts(ir, min_confidence=0.6)
        # Only the 3 labelled bubbles pass 0.6: the oversized circle is
        # excluded by radius, the unlabelled one scores below (no text).
        assert len(props) == len(gt["bubbles"])

    def test_unlabelled_circle_surfaces_below_threshold(self, sheet):
        ir, gt = sheet
        allp = queries.find_bubble_callouts(ir, min_confidence=0.0)
        unlabelled = [p for p in allp if p["text"] is None]
        assert unlabelled, "unlabelled decoy should still be visible at 0.0"
        assert all(p["confidence"] < 0.6 for p in unlabelled)


# ---------------------------------------------------------------------------
# find_revision_clouds
# ---------------------------------------------------------------------------

class TestFindRevisionClouds:
    @pytest.fixture()
    def sheet(self, tmp_path):
        path, gt = build_synthetic_cloud_pdf(tmp_path)
        return from_pdf_vector(path), gt

    def test_cloud_found_with_low_confidence_band(self, sheet):
        ir, gt = sheet
        props = queries.find_revision_clouds(ir)
        clouds = [p for p in props if p["kind"] == "cloud"]
        assert len(clouds) == 1, "exactly the planted cloud"
        c = clouds[0]
        exp = gt["cloud_bbox_ir"]
        # Cloud bbox contains the underlying rectangle (bumps extend it).
        assert c["bbox"][0] <= exp[0] and c["bbox"][1] <= exp[1]
        assert c["bbox"][2] >= exp[2] and c["bbox"][3] >= exp[3]
        assert c["confidence"] <= 0.65, "best-effort tier stays low"
        assert c["evidence"]["path"] == "scalloped_ring"

    def test_revision_delta_found(self, sheet):
        ir, gt = sheet
        props = queries.find_revision_clouds(ir)
        deltas = [p for p in props if p["kind"] == "revision_delta"]
        assert len(deltas) == 1
        assert deltas[0]["text"] == gt["delta_text"]
        assert _dist(deltas[0]["center_xy"], gt["delta_center_ir"]) < 5.0

    def test_leader_arrowhead_is_not_a_delta(self, sheet):
        ir, gt = sheet
        props = queries.find_revision_clouds(ir)
        deltas = [p for p in props if p["kind"] == "revision_delta"]
        # The true leader's arrowhead (which HAS a shaft) must not appear.
        assert len(deltas) == gt["n_decoy_leaders"] * 0 + 1

    def test_smooth_circle_and_rectangle_rejected(self, sheet):
        ir, _gt = sheet
        props = queries.find_revision_clouds(ir)
        assert len([p for p in props if p["kind"] == "cloud"]) == 1

    def test_dxf_native_arc_chain(self, tmp_path):
        ezdxf = pytest.importorskip("ezdxf")
        from drawing_ir import from_dxf
        doc = ezdxf.new()
        msp = doc.modelspace()
        # 6 semicircular bumps along y=0, each arc spanning 10 units of x.
        for i in range(6):
            msp.add_arc(center=(i * 10 + 5, 0), radius=5,
                        start_angle=180, end_angle=0)
        # A lone unrelated arc far away — not part of any chain.
        msp.add_arc(center=(200, 200), radius=8, start_angle=0, end_angle=90)
        p = str(tmp_path / "arcs.dxf")
        doc.saveas(p)
        ir = from_dxf(filepath=p)
        props = queries.find_revision_clouds(ir, min_arcs=3)
        clouds = [p for p in props if p["kind"] == "cloud"
                  and p["evidence"]["path"] == "native_arcs"]
        assert len(clouds) == 1
        assert clouds[0]["evidence"]["n_arcs"] == 6
