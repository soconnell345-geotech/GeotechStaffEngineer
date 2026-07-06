"""Tests for pdf_import.labels — C2 label->region association (v5.3)."""

import pytest

from pdf_import.labels import (
    classify_label, associate_labels_to_regions, propose_role_mapping,
)

# A clay box (gray, lower), a sand box (yellow, upper), a GWT line (blue),
# and a ground-surface line (black) — closed boxes end where they start.
_CLAY = {"color": "#808080", "points": [(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)]}
_SAND = {"color": "#ffff00", "points": [(0, 50), (100, 50), (100, 100), (0, 100), (0, 50)]}
_GWT = {"color": "#0000ff", "points": [(0, 60), (100, 62)]}
_SURF = {"color": "#000000", "points": [(0, 102), (100, 103)]}
_REGIONS = [_CLAY, _SAND, _GWT, _SURF]


class TestClassifyLabel:
    @pytest.mark.parametrize("text,role", [
        ("CLAY", "boundary_Clay"),
        ("CL", "boundary_Clay"),
        ("Silty SAND (SM)", "boundary_Sand"),   # rightmost noun / SM symbol
        ("sandy clay", "boundary_Clay"),         # rightmost noun wins
        ("SP-SM", "boundary_Sand"),
        ("ML", "boundary_Silt"),
        ("FILL", "boundary_Fill"),
        ("Embankment fill", "boundary_Fill"),
        ("Bedrock", "boundary_Rock"),
        ("GWT", "gwt"),
        ("Phreatic surface", "gwt"),             # gwt phrase beats 'surface'
        ("Existing Grade", "surface"),
        ("Ground Surface", "surface"),
        ("Boring BH-1", None),
        ("", None),
    ])
    def test_classify(self, text, role):
        assert classify_label(text) == role


class TestAssociate:
    def test_soil_labels_enclosing(self):
        assoc = associate_labels_to_regions(
            _REGIONS, [{"text": "CLAY", "x": 50, "y": 25},
                       {"text": "SAND SP", "x": 50, "y": 75}])
        by_label = {a["label"]: a for a in assoc}
        assert by_label["CLAY"]["color"] == "#808080"
        assert by_label["CLAY"]["method"] == "enclosing"
        assert by_label["SAND SP"]["color"] == "#ffff00"

    def test_gwt_line_not_captured_by_box(self):
        # GWT sits inside the sand box but must attach to the blue line
        assoc = associate_labels_to_regions(_REGIONS, [{"text": "GWT", "x": 10, "y": 61}])
        assert len(assoc) == 1
        assert assoc[0]["role"] == "gwt"
        assert assoc[0]["color"] == "#0000ff"
        assert assoc[0]["method"] == "nearest"

    def test_unclassifiable_skipped(self):
        assert associate_labels_to_regions(_REGIONS, [{"text": "TP-4", "x": 5, "y": 5}]) == []

    def test_ignores_degenerate_regions(self):
        assoc = associate_labels_to_regions(
            [{"color": "#808080", "points": [(0, 0)]}],   # single point
            [{"text": "CLAY", "x": 5, "y": 5}])
        assert assoc == []


class TestProposeRoleMapping:
    def test_full_mapping_proposed_not_applied(self):
        labels = [{"text": "CLAY", "x": 50, "y": 25},
                  {"text": "SAND (SP)", "x": 50, "y": 75},
                  {"text": "GWT", "x": 10, "y": 61},
                  {"text": "Existing Grade", "x": 30, "y": 102},
                  {"text": "Boring B-2", "x": 5, "y": 5}]
        out = propose_role_mapping(_REGIONS, labels)
        rm = out["role_mapping"]
        assert rm["#808080"] == "boundary_Clay"
        assert rm["#ffff00"] == "boundary_Sand"
        assert rm["#0000ff"] == "gwt"
        assert rm["#000000"] == "surface"
        assert out["applied"] is False
        assert "NOT applied" in out["note"]

    def test_closest_label_wins_per_color(self):
        # two clay labels near the gray box; the closer one is kept
        regions = [{"color": "#808080", "points": [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]}]
        labels = [{"text": "CLAY", "x": 5, "y": 5},        # enclosed, d=0
                  {"text": "Stiff CLAY", "x": 500, "y": 500}]  # far
        out = propose_role_mapping(regions, labels)
        assert out["role_mapping"] == {"#808080": "boundary_Clay"}
        best = [a for a in out["associations"] if a["color"] == "#808080"]
        assert min(a["distance"] for a in best) == 0.0

    def test_no_labels(self):
        out = propose_role_mapping(_REGIONS, [{"text": "notes", "x": 1, "y": 1}])
        assert out["role_mapping"] == {}
