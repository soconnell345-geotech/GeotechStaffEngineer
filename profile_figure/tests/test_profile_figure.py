"""Tests for profile_figure — geometry resolution, validation, rendering.

Geometry is tested through ``resolve_profile`` (no matplotlib needed); the
render tests decode the PNG and check it is a real, non-trivial image.
"""

import base64

import pytest

from profile_figure import render_profile_figure, resolve_profile


LAYERS = [
    {"name": "Fill", "thickness": 3.0},
    {"name": "Soft clay", "thickness": 9.0, "settling": True},
    {"name": "Dense sand", "thickness": 8.0},
]


# ---------------------------------------------------------------------------
# Geometry — layers stack correctly by elevation
# ---------------------------------------------------------------------------

class TestLayerStacking:
    def test_thicknesses_stack_from_ground_downward(self):
        r = resolve_profile(LAYERS)
        tops = [lay["top"] for lay in r["layers"]]
        bottoms = [lay["bottom"] for lay in r["layers"]]
        assert tops == [0.0, -3.0, -12.0]
        assert bottoms == [-3.0, -12.0, -20.0]
        assert r["ground"] == 0.0
        assert r["base"] == -20.0

    def test_ground_elevation_offsets_the_whole_stack(self):
        r = resolve_profile(LAYERS, ground_elevation=15.0)
        assert r["ground"] == 15.0
        assert [lay["top"] for lay in r["layers"]] == [15.0, 12.0, 3.0]
        assert r["base"] == -5.0

    def test_explicit_elevations_are_honored(self):
        r = resolve_profile([
            {"name": "A", "top_elevation": 10.0, "bottom_elevation": 4.0},
            {"name": "B", "top_elevation": 4.0, "bottom_elevation": -2.0},
        ])
        assert r["ground"] == 10.0
        assert r["layers"][1]["thickness"] == pytest.approx(6.0)

    def test_depth_keys_are_converted_to_elevations(self):
        r = resolve_profile(
            [{"name": "A", "top_depth": 0.0, "bottom_depth": 5.0},
             {"name": "B", "top_depth": 5.0, "bottom_depth": 11.0}],
            ground_elevation=100.0)
        assert [lay["top"] for lay in r["layers"]] == [100.0, 95.0]
        assert r["base"] == 89.0

    def test_layer_thicknesses_are_positive_and_sum_to_the_section(self):
        r = resolve_profile(LAYERS)
        assert [lay["thickness"] for lay in r["layers"]] == [3.0, 9.0, 8.0]
        assert sum(lay["thickness"] for lay in r["layers"]) == pytest.approx(
            r["ground"] - r["base"])

    def test_settling_flag_is_carried_through(self):
        r = resolve_profile(LAYERS)
        assert [lay["settling"] for lay in r["layers"]] == [False, True, False]


class TestGeometryValidation:
    def test_gap_between_layers_is_rejected(self):
        with pytest.raises(ValueError, match="gap"):
            resolve_profile([
                {"name": "A", "top_elevation": 0.0, "bottom_elevation": -5.0},
                {"name": "B", "top_elevation": -8.0, "bottom_elevation": -12.0},
            ])

    def test_overlapping_layers_are_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            resolve_profile([
                {"name": "A", "top_elevation": 0.0, "bottom_elevation": -5.0},
                {"name": "B", "top_elevation": -3.0, "bottom_elevation": -9.0},
            ])

    def test_layer_without_thickness_or_bottom_is_rejected(self):
        with pytest.raises(ValueError, match="thickness"):
            resolve_profile([{"name": "A"}])

    def test_inverted_layer_is_rejected(self):
        with pytest.raises(ValueError, match="thicker than zero"):
            resolve_profile([{"name": "A", "top_elevation": 0.0,
                              "bottom_elevation": 4.0}])

    def test_empty_layers_is_rejected(self):
        with pytest.raises(ValueError, match="at least one layer"):
            resolve_profile([])

    def test_non_numeric_thickness_names_the_layer(self):
        with pytest.raises(ValueError, match=r"layers\[0\]"):
            resolve_profile([{"name": "A", "thickness": "six meters"}])


# ---------------------------------------------------------------------------
# Water table, fill, surcharge
# ---------------------------------------------------------------------------

class TestWaterFillSurcharge:
    def test_water_depth_becomes_an_elevation(self):
        r = resolve_profile(LAYERS, ground_elevation=20.0, water_depth=2.5)
        assert r["water"] == pytest.approx(17.5)

    def test_water_elevation_passes_through(self):
        r = resolve_profile(LAYERS, water_elevation=-2.0)
        assert r["water"] == pytest.approx(-2.0)

    def test_water_below_the_section_warns(self):
        r = resolve_profile(LAYERS, water_elevation=-50.0)
        assert any("below the deepest layer" in w for w in r["warnings"])

    def test_fill_sits_above_the_ground_surface(self):
        r = resolve_profile(LAYERS, fill={"name": "Embankment",
                                          "thickness": 2.0})
        assert r["fill"]["bottom"] == r["ground"] == 0.0
        assert r["fill"]["top"] == pytest.approx(2.0)
        assert r["top"] == pytest.approx(2.0)

    def test_bare_number_fill_is_a_thickness(self):
        r = resolve_profile(LAYERS, fill=1.5)
        assert r["fill"]["thickness"] == pytest.approx(1.5)

    def test_fill_below_ground_is_rejected(self):
        with pytest.raises(ValueError, match="ABOVE the ground surface"):
            resolve_profile(LAYERS, fill={"top_elevation": -1.0})

    def test_bare_number_surcharge_is_a_pressure(self):
        r = resolve_profile(LAYERS, surcharge=25.0)
        assert r["surcharge"]["pressure"] == pytest.approx(25.0)
        assert "25" in r["surcharge"]["label"]

    def test_surcharge_without_pressure_is_rejected(self):
        with pytest.raises(ValueError, match="pressure"):
            resolve_profile(LAYERS, surcharge={"label": "traffic"})


# ---------------------------------------------------------------------------
# Foundation overlay
# ---------------------------------------------------------------------------

class TestFoundation:
    def test_pile_spans_head_to_tip(self):
        r = resolve_profile(LAYERS, foundation={
            "type": "pile", "diameter": 0.4,
            "head_elevation": 0.0, "tip_elevation": -18.0})
        f = r["foundation"]
        assert (f["head"], f["tip"]) == (0.0, -18.0)
        assert f["length"] == pytest.approx(18.0)

    def test_head_defaults_to_the_top_of_the_section(self):
        r = resolve_profile(LAYERS, fill=2.0, foundation={
            "type": "pile", "diameter": 0.4, "tip_depth": 18.0})
        # Head at the fill surface; tip_depth measured from ORIGINAL ground.
        assert r["foundation"]["head"] == pytest.approx(2.0)
        assert r["foundation"]["tip"] == pytest.approx(-18.0)
        assert r["foundation"]["length"] == pytest.approx(20.0)

    def test_length_sets_the_tip_when_no_tip_given(self):
        r = resolve_profile(LAYERS, foundation={
            "type": "micropile", "diameter": 0.25, "length": 15.0})
        assert r["foundation"]["tip"] == pytest.approx(-15.0)

    def test_tip_above_head_is_rejected(self):
        with pytest.raises(ValueError, match="BELOW its head"):
            resolve_profile(LAYERS, foundation={
                "type": "pile", "diameter": 0.4,
                "head_elevation": -10.0, "tip_elevation": -2.0})

    def test_missing_diameter_is_rejected(self):
        with pytest.raises(ValueError, match="diameter"):
            resolve_profile(LAYERS, foundation={"type": "pile",
                                                "tip_depth": 10.0})

    def test_missing_tip_is_rejected(self):
        with pytest.raises(ValueError, match="tip_elevation"):
            resolve_profile(LAYERS, foundation={"type": "pile",
                                                "diameter": 0.4})

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_profile(LAYERS, foundation={
                "type": "caisson", "diameter": 1.0, "tip_depth": 5.0})

    def test_tip_below_the_profile_warns(self):
        r = resolve_profile(LAYERS, foundation={
            "type": "pile", "diameter": 0.4, "tip_depth": 30.0})
        assert any("below the deepest layer" in w for w in r["warnings"])


# ---------------------------------------------------------------------------
# Annotations + axis mode
# ---------------------------------------------------------------------------

class TestAnnotationsAndAxis:
    def test_depth_annotation_resolves_against_ground(self):
        r = resolve_profile(LAYERS, ground_elevation=10.0,
                            annotations=[{"depth": 4.0, "text": "NP"}])
        assert r["annotations"][0]["elevation"] == pytest.approx(6.0)
        assert r["annotations"][0]["side"] == "right"

    def test_annotation_without_text_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            resolve_profile(LAYERS, annotations=[{"depth": 4.0}])

    def test_annotation_outside_the_section_warns(self):
        r = resolve_profile(LAYERS,
                            annotations=[{"elevation": -99.0, "text": "x"}])
        assert any("outside the drawn section" in w for w in r["warnings"])

    def test_auto_axis_is_depth_for_thickness_input(self):
        assert resolve_profile(LAYERS)["axis"] == "depth"

    def test_auto_axis_is_elevation_when_elevations_are_given(self):
        assert resolve_profile(LAYERS, ground_elevation=8.0)["axis"] == \
            "elevation"
        assert resolve_profile(
            [{"name": "A", "top_elevation": 5.0, "thickness": 3.0}]
        )["axis"] == "elevation"

    def test_axis_can_be_forced(self):
        assert resolve_profile(LAYERS, axis="elevation")["axis"] == "elevation"
        with pytest.raises(ValueError, match="axis"):
            resolve_profile(LAYERS, axis="sideways")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRender:
    def test_png_is_written_and_decodes(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "profile.png"
        res = render_profile_figure(LAYERS, water_depth=2.0,
                                    title="Test section",
                                    output_path=str(out))
        assert out.is_file()
        raw = out.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        # Non-trivial image: a real section, not a blank canvas.
        assert len(raw) > 10_000
        assert res.width_px > 300 and res.height_px > 300
        assert base64.b64decode(res.image_base64) == raw

    def test_data_uri_and_img_tag_are_embeddable(self, tmp_path):
        pytest.importorskip("matplotlib")
        res = render_profile_figure(LAYERS, output_path=str(tmp_path / "p.png"))
        assert res.data_uri().startswith("data:image/png;base64,")
        tag = res.img_tag()
        assert tag.startswith('<img src="data:image/png;base64,')
        assert "max-width:640px" in tag

    def test_render_without_output_path_still_returns_the_image(self):
        pytest.importorskip("matplotlib")
        res = render_profile_figure(LAYERS)
        assert res.output_path is None
        assert len(res.png_bytes) > 10_000

    def test_full_featured_section_renders(self, tmp_path):
        pytest.importorskip("matplotlib")
        res = render_profile_figure(
            LAYERS, ground_elevation=5.0, water_elevation=3.0,
            fill={"name": "New fill", "thickness": 2.0}, surcharge=20.0,
            foundation={"type": "micropile", "diameter": 0.25,
                        "tip_elevation": -12.0, "label": "MCAC micropile"},
            annotations=[{"elevation": -6.0, "text": "Neutral plane"},
                         {"elevation": -12.0, "text": "Tip", "side": "left"}],
            title="Downdrag section", output_path=str(tmp_path / "f.png"))
        assert res.axis == "elevation"
        assert res.foundation["length"] == pytest.approx(19.0)
        assert len(res.png_bytes) > 10_000

    def test_dpi_changes_the_pixel_size(self, tmp_path):
        pytest.importorskip("matplotlib")
        small = render_profile_figure(LAYERS, dpi=72,
                                      output_path=str(tmp_path / "s.png"))
        big = render_profile_figure(LAYERS, dpi=150,
                                    output_path=str(tmp_path / "b.png"))
        assert big.width_px > small.width_px

    def test_summary_lists_layers_water_and_foundation(self):
        pytest.importorskip("matplotlib")
        res = render_profile_figure(
            LAYERS, water_depth=2.0,
            foundation={"type": "pile", "diameter": 0.4, "tip_depth": 18.0,
                        "label": "H-pile"})
        text = res.summary()
        assert "Soft clay" in text and "[settling]" in text
        assert "Water table" in text
        assert "H-pile" in text

    def test_to_dict_is_json_serializable(self):
        pytest.importorskip("matplotlib")
        import json
        res = render_profile_figure(LAYERS, water_depth=1.0)
        payload = json.dumps(res.to_dict())
        assert '"layers"' in payload
        assert "png_bytes" not in payload
