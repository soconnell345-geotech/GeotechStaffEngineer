"""GM-327: the drawing's own units against ``units``, text annotations
converted like the geometry, and the agent adapter's rule for the soil and
nail properties a drawing does not carry (a missing unit weight, strength
or stiffness is refused; every value that stands in is listed).

Every drawing and every soil value here is made up.
"""

import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import.converter import (                         # noqa: E402
    NAIL_PROPERTY_DEFAULTS, SoilPropertyAssignment, build_slope_geometry,
)
from dxf_import.parser import LayerMapping, parse_dxf_geometry  # noqa: E402
from dxf_import.results import DxfParseResult              # noqa: E402


def _drawing(tmp_path, insunits, scale=1.0, name="section.dxf"):
    """A 20 m by 10 m bench drawn at ``scale`` drawing units per metre,
    with a label, a nail drawn as a LINE and one drawn as a polyline."""
    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = insunits
    msp = doc.modelspace()
    s = scale
    msp.add_lwpolyline([(0, 10 * s), (8 * s, 10 * s), (14 * s, 4 * s),
                        (20 * s, 4 * s)], dxfattribs={"layer": "GROUND"})
    msp.add_text("Stiff clay", dxfattribs={
        "layer": "LABELS", "insert": (5 * s, 7 * s)})
    msp.add_mtext("Toe drain", dxfattribs={
        "layer": "LABELS", "insert": (15 * s, 3 * s)})
    msp.add_line((9 * s, 9 * s), (14 * s, 8 * s),
                 dxfattribs={"layer": "NAILS"})
    msp.add_lwpolyline([(10 * s, 8 * s), (15 * s, 7 * s)],
                       dxfattribs={"layer": "NAILS"})
    path = tmp_path / name
    doc.saveas(str(path))
    return str(path)


_MAPPING = LayerMapping(surface="GROUND", annotations=["LABELS"],
                        nails="NAILS")


class TestUnitsAgainstTheHeader:
    def test_left_out_the_header_units_are_used_and_said(self, tmp_path):
        path = _drawing(tmp_path, insunits=4, scale=1000.0)      # mm
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        assert r.units_used == "mm" and r.header_units == "mm"
        assert "header" in r.units_source
        # 1000 times smaller than the same drawing read as metres
        assert r.surface_points[-1] == pytest.approx((20.0, 4.0))

    def test_given_and_different_is_refused_naming_both(self, tmp_path):
        path = _drawing(tmp_path, insunits=4, scale=1000.0)
        with pytest.raises(ValueError) as err:
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="m")
        msg = str(err.value)
        assert "units='m'" in msg and "states mm" in msg
        assert "ignore_header_units" in msg

    def test_given_against_the_header_on_request_with_a_warning(
            self, tmp_path):
        # a drawing in feet whose header was left at inches
        path = _drawing(tmp_path, insunits=1, scale=1 / 0.3048)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="ft", ignore_header_units=True)
        assert r.units_used == "ft" and r.header_units == "in"
        assert r.surface_points[-1] == pytest.approx((20.0, 4.0))
        assert any("states in" in w and "ignored" in w for w in r.warnings)

    def test_given_and_the_same_runs_clean(self, tmp_path):
        path = _drawing(tmp_path, insunits=2, scale=1 / 0.3048)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="ft")
        assert "agrees" in r.units_source
        assert not any("header" in w for w in r.warnings)

    def test_neither_states_units_is_refused(self, tmp_path):
        path = _drawing(tmp_path, insunits=0)
        with pytest.raises(ValueError, match="states no units"):
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        # with units given, a unitless header is no difference
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="m")
        assert r.units_used == "m" and r.header_units is None

    def test_a_header_unit_not_converted_asks_for_units(self, tmp_path):
        path = _drawing(tmp_path, insunits=7)                    # km
        with pytest.raises(ValueError, match="kilometres"):
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)

    def test_the_dict_carries_the_units_story(self, tmp_path):
        path = _drawing(tmp_path, insunits=6)
        d = parse_dxf_geometry(filepath=path,
                               layer_mapping=_MAPPING).to_dict()
        assert d["units_used"] == "m" and d["header_units"] == "m"
        assert d["units_source"]


class TestAnnotationsAndNailLayer:
    def test_annotations_are_in_metres_like_the_geometry(self, tmp_path):
        path = _drawing(tmp_path, insunits=4, scale=1000.0)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        by_text = {a["text"]: a for a in r.text_annotations}
        assert (by_text["Stiff clay"]["x"], by_text["Stiff clay"]["y"]) \
            == pytest.approx((5.0, 7.0))        # was (5000, 7000)
        assert (by_text["Toe drain"]["x"], by_text["Toe drain"]["y"]) \
            == pytest.approx((15.0, 3.0))

    def test_flip_y_and_feet(self, tmp_path):
        path = _drawing(tmp_path, insunits=2, scale=1 / 0.3048)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               flip_y=True)
        clay = next(a for a in r.text_annotations
                    if a["text"] == "Stiff clay")
        assert (clay["x"], clay["y"]) == pytest.approx((5.0, -7.0))

    def test_a_missing_annotation_layer_is_said(self, tmp_path):
        path = _drawing(tmp_path, insunits=6)
        r = parse_dxf_geometry(
            filepath=path,
            layer_mapping=LayerMapping(surface="GROUND",
                                       annotations=["NOTES"]))
        assert any("Annotation layer 'NOTES' not found" in w
                   for w in r.warnings)

    def test_a_nail_not_drawn_as_a_line_is_counted(self, tmp_path):
        path = _drawing(tmp_path, insunits=6)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        assert len(r.nail_lines) == 1
        assert any("left out: 1 LWPOLYLINE" in w for w in r.warnings)


def _parse_result(nails=False):
    pr = {"surface_points": [[0, 10], [8, 10], [14, 4], [20, 4]],
          "boundary_profiles": {"Clay": [[0, 3], [20, 2]]}}
    if nails:
        pr["nail_lines"] = [{"x_head": 9.0, "z_head": 9.0,
                             "x_tip": 14.0, "z_tip": 8.0}]
    return pr


def _call(method, params):
    from funhouse_agent.dispatch import call_agent
    return call_agent("dxf_import", method, params)


class TestSlopeAdapterAssumesNothing:
    def test_a_missing_strength_is_refused_naming_every_gap(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "phi": 30.0},               # no gamma
                {"name": "Clay", "gamma": 19.0,
                 "analysis_mode": "undrained"}]})               # no cu
        assert "soil_properties[Surface] has no gamma" in r["error"]
        assert "soil_properties[Clay] has no cu" in r["error"]
        assert "nothing is assumed" in r["error"]

    def test_a_drained_layer_without_phi_is_refused(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "c_prime": 5.0},
                {"name": "Clay", "gamma": 19.0, "phi": 25.0}]})
        assert "soil_properties[Surface] has no phi" in r["error"]

    def test_what_stood_in_is_listed(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "phi": 30.0},
                {"name": "Clay", "gamma": 19.0, "cu": 40.0,
                 "analysis_mode": "undrained"}]})
        assert "error" not in r, r
        assert r["defaults_used"] == {
            "soil_properties[Surface].analysis_mode": "drained",
            "soil_properties[Surface].c_prime": 0.0,
            "soil_properties[Surface].gamma_sat": 18.0,
            "soil_properties[Clay].gamma_sat": 19.0}
        assert r["soil_layers"][0]["gamma_sat"] == 18.0
        assert r["soil_layers"][0]["c_prime"] == 0.0
        assert r["soil_layers"][1]["cu"] == 40.0

    def test_nothing_left_out_nothing_listed(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "gamma_sat": 19.5,
                 "phi": 30.0, "c_prime": 2.0, "analysis_mode": "drained"},
                {"name": "Clay", "gamma": 19.0, "gamma_sat": 19.0,
                 "phi": 25.0, "c_prime": 8.0, "analysis_mode": "drained"}]})
        assert "error" not in r and "defaults_used" not in r

    _FULL = [{"name": "Surface", "gamma": 18.0, "gamma_sat": 19.5,
              "phi": 30.0, "c_prime": 2.0, "analysis_mode": "drained"},
             {"name": "Clay", "gamma": 19.0, "gamma_sat": 19.0, "phi": 25.0,
              "c_prime": 8.0, "analysis_mode": "drained"}]

    def test_nail_properties_left_out_are_listed(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(nails=True),
            "soil_properties": self._FULL,
            "nail_defaults": {"bond_stress": 60.0, "fy": 500.0}})
        assert "error" not in r, r
        assert r["defaults_used"] == {
            "nail_defaults.bar_diameter": 25.0,
            "nail_defaults.drill_hole_diameter": 150.0,
            "nail_defaults.spacing_h": 1.5}
        nail = r["nails"][0]
        assert nail["bond_stress"] == 60.0 and nail["bar_diameter"] == 25.0

    def test_no_nail_defaults_at_all_lists_all_five(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(nails=True),
            "soil_properties": self._FULL})
        assert r["defaults_used"] == {
            f"nail_defaults.{k}": v
            for k, v in NAIL_PROPERTY_DEFAULTS.items()}

    def test_no_nails_no_nail_defaults_listed(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": self._FULL})
        assert "defaults_used" not in r

    def test_the_listed_nail_values_are_the_converters_own(self):
        """NAIL_PROPERTY_DEFAULTS is what build_slope_geometry fills in
        (the function is not edited here; this holds the two together)."""
        geom = build_slope_geometry(
            DxfParseResult(
                surface_points=[(0, 10), (8, 10), (14, 4), (20, 4)],
                nail_lines=[{"x_head": 9.0, "z_head": 9.0,
                             "x_tip": 14.0, "z_tip": 8.0}]),
            [SoilPropertyAssignment(name="Fill", gamma=18.0, phi=30.0)])
        nail = geom.nails[0]
        assert {k: getattr(nail, k) for k in NAIL_PROPERTY_DEFAULTS} \
            == NAIL_PROPERTY_DEFAULTS


class TestFemAdapterAssumesNothing:
    def test_missing_unit_weight_stiffness_or_strength_is_refused(self):
        r = _call("build_fem_inputs", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "phi": 30.0},
                {"name": "Clay", "E": 15000.0, "nu": 0.35, "phi": 0.0,
                 "c": 40.0}]})
        assert "soil_properties[Surface] has no E, nu" in r["error"]
        assert "soil_properties[Clay] has no gamma" in r["error"]

    def test_an_elastic_layer_needs_no_strength(self):
        r = _call("build_fem_inputs", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "E": 30000.0, "nu": 0.3,
                 "model": "elastic"},
                {"name": "Clay", "gamma": 19.0, "E": 15000.0, "nu": 0.35,
                 "phi": 0.0, "c": 40.0, "psi": 0.0, "model": "mc"}]})
        assert "error" not in r and "defaults_used" not in r

    def test_what_stood_in_is_listed(self):
        r = _call("build_fem_inputs", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "E": 30000.0, "nu": 0.3,
                 "phi": 32.0},
                {"name": "Clay", "gamma": 19.0, "E": 15000.0, "nu": 0.35,
                 "phi": 0.0, "c": 40.0, "psi": 0.0, "model": "mc"}]})
        assert r["defaults_used"] == {
            "soil_properties[Surface].model": "mc",
            "soil_properties[Surface].c": 0.0,
            "soil_properties[Surface].psi": 0.0}
        assert r["soil_layers"][0]["E"] == 30000.0


class TestAgentParseGeometry:
    def test_units_are_not_assumed(self, tmp_path):
        path = _drawing(tmp_path, insunits=4, scale=1000.0)
        mapping = {"surface": "GROUND", "annotations": ["LABELS"]}
        r = _call("parse_geometry",
                  {"file_path": path, "layer_mapping": mapping})
        assert r["units_used"] == "mm" and r["header_units"] == "mm"
        assert r["surface_points"][-1] == {"x": 20.0, "z": 4.0}
        assert r["text_annotations"][0]["x"] == 5.0
        refused = _call("parse_geometry",
                        {"file_path": path, "layer_mapping": mapping,
                         "units": "m"})
        assert "states mm" in refused["error"]


class TestReviewRound:
    """GM-327 review B1-B4 and C3. Made-up drawings and soil values."""

    def test_header_units_alone_are_warned_with_the_section_size(
            self, tmp_path):
        # a 20 m by 6 m section drawn in feet under an inch header (B1):
        # the numbers stay the header's, the warning gives the size as read
        path = _drawing(tmp_path, insunits=1, scale=1 / 0.3048)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        assert r.units_used == "in"
        assert r.surface_points[-1] == pytest.approx(
            (20.0 / 12, 4.0 / 12))
        said = [w for w in r.warnings if "header decided" in w]
        assert len(said) == 1
        assert "$INSUNITS = 1, inches" in said[0]
        assert "1.67 m wide and 0.50 m high" in said[0]
        assert "give units to confirm or correct it" in said[0]
        # units given: no such warning
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="in")
        assert not any("header decided" in w for w in r.warnings)

    def test_us_survey_feet_are_read_and_ft_agrees(self, tmp_path):
        path = _drawing(tmp_path, insunits=21, scale=1 / 0.3048)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        assert r.units_used == "us_survey_ft"
        assert r.header_units == "us_survey_ft"
        assert "US survey feet" in r.units_source
        assert "2 parts per million" in r.units_source
        x = 20.0 / 0.3048 * 1200 / 3937
        assert r.surface_points[-1][0] == pytest.approx(x, rel=1e-12)
        assert any("header decided" in w for w in r.warnings)
        given = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                                   units="ft")
        assert given.units_used == "us_survey_ft"
        assert "given as ft" in given.units_source
        assert "agrees" in given.units_source
        assert given.surface_points[-1][0] == pytest.approx(x, rel=1e-12)
        assert not any("header" in w for w in given.warnings)
        with pytest.raises(ValueError, match="Leave units out"):
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="m")

    def test_us_survey_inches_are_read_and_in_agrees(self, tmp_path):
        path = _drawing(tmp_path, insunits=22, scale=1 / 0.0254)
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="in")
        assert r.units_used == "us_survey_in"
        assert "US survey inches" in r.units_source
        x = 20.0 / 0.0254 * 100 / 3937
        assert r.surface_points[-1][0] == pytest.approx(x, rel=1e-12)
        assert r.text_annotations[0]["x"] == pytest.approx(
            5.0 / 0.0254 * 100 / 3937, rel=1e-12)

    @pytest.mark.parametrize("code,words", [(23, "US survey yards"),
                                            (24, "US survey miles")])
    def test_other_survey_units_stay_refused_without_a_loop(
            self, tmp_path, code, words):
        path = _drawing(tmp_path, insunits=code)
        with pytest.raises(ValueError, match=words) as left_out:
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        assert "ignore_header_units=True" in str(left_out.value)
        with pytest.raises(ValueError) as given:
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="m")
        assert "Leave units out" not in str(given.value)
        assert "ignore_header_units=True" in str(given.value)
        # the advice given is a call that runs
        r = parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING,
                               units="m", ignore_header_units=True)
        assert r.units_used == "m"
        assert any(words in w and "ignored" in w for w in r.warnings)

    def test_unitless_refusal_names_measurement_as_a_hint(self, tmp_path):
        path = _drawing(tmp_path, insunits=0)
        doc = ezdxf.readfile(path)
        doc.header["$MEASUREMENT"] = 0
        doc.saveas(path)
        with pytest.raises(ValueError, match=r"\$MEASUREMENT is 0") as e:
            parse_dxf_geometry(filepath=path, layer_mapping=_MAPPING)
        assert "English (feet or inches)" in str(e.value)
        assert "hint only" in str(e.value)

    def test_phi_zero_with_no_cohesion_is_refused(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "phi": 30.0},
                {"name": "Clay", "gamma": 19.0, "phi": 0.0}]})
        assert "soil_properties[Clay] has phi 0 and no c_prime" in r["error"]
        assert "undrained" in r["error"]
        r = _call("build_fem_inputs", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "E": 30000.0, "nu": 0.3,
                 "phi": 30.0},
                {"name": "Clay", "gamma": 18.0, "E": 15000.0, "nu": 0.45,
                 "phi": 0.0}]})
        assert "soil_properties[Clay] has phi 0 and no c," in r["error"]

    def test_phi_zero_with_cohesion_zero_given_runs_with_a_warning(self):
        r = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "phi": 30.0},
                {"name": "Clay", "gamma": 19.0, "phi": 0.0,
                 "c_prime": 0.0}]})
        assert "error" not in r, r
        assert any("soil_properties[Clay] has phi 0 and c_prime 0" in w
                   and "no strength" in w for w in r["warnings"])
        r = _call("build_fem_inputs", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "E": 30000.0, "nu": 0.3,
                 "phi": 30.0, "c": 2.0},
                {"name": "Clay", "gamma": 18.0, "E": 15000.0, "nu": 0.45,
                 "phi": 0.0, "c": 0.0}]})
        assert "error" not in r, r
        assert any("has phi 0 and c 0" in w for w in r["warnings"])

    def test_the_keys_the_text_names_run_in_analyze_slope(self):
        from funhouse_agent.dispatch import call_agent
        from funhouse_agent.adapters.dxf_import_adapter import (
            METHOD_INFO,
        )
        text = METHOD_INFO["build_slope_geometry"]["returns"]["nails"]
        keys = ("surface_points", "soil_layers", "gwt_points", "nails")
        assert all(k in text for k in keys)
        geom = _call("build_slope_geometry", {
            "parse_result": _parse_result(),
            "soil_properties": [
                {"name": "Surface", "gamma": 18.0, "phi": 30.0},
                {"name": "Clay", "gamma": 19.0, "phi": 25.0,
                 "c_prime": 5.0}]})
        assert "defaults_used" in geom
        handed = {k: geom[k] for k in keys if geom.get(k) is not None}
        handed.update(xc=10.0, yc=18.0, radius=12.0)
        out = call_agent("slope_stability", "analyze_slope", handed)
        assert "error" not in out, out
        assert out["FOS"] > 0
