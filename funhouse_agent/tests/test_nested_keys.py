"""GM-58: nested dicts are checked like top-level parameters, in every adapter.

GM-52 found the slope adapter reading soil layers with ``.get``: a layer given
``"c": 10`` ran with c' = 0 and a Bishop FOS of 0.97 against the true 1.89.
``check_nested_keys`` (now in ``funhouse_agent.adapters``) refuses a nested
dict's unknown key, naming the item, the key, a did-you-mean and the valid
keys; it renames nothing. These tests give each adapter that reads a nested
dict one misspelt key (which must be refused, with the did-you-mean) and one
correct call (which must still run).
"""

from unittest.mock import MagicMock, patch

import pytest

from funhouse_agent.adapters import (
    check_nested_keys, check_nested_list, field_keys,
)


def _registry(module):
    import importlib
    return importlib.import_module(
        f"funhouse_agent.adapters.{module}").METHOD_REGISTRY


def _refused(module, method, params, *patterns):
    """The call raises ValueError whose text matches every pattern."""
    import re
    with pytest.raises(ValueError) as e:
        _registry(module)[method](params)
    msg = str(e.value)
    for p in patterns:
        assert re.search(p, msg), f"{p!r} not in {msg!r}"
    return msg


# ---------------------------------------------------------------------------
# The shared helpers
# ---------------------------------------------------------------------------

class TestSharedHelpers:
    def test_names_item_key_suggestion_and_valid_keys(self):
        with pytest.raises(ValueError) as e:
            check_nested_keys({"unit_wieght": 18}, ("unit_weight", "phi"),
                              method="m", label="layers[0]")
        msg = str(e.value)
        assert msg.startswith("m: layers[0] has unknown key(s) "
                              "['unit_wieght'].")
        assert "unit_wieght -> unit_weight" in msg
        assert "Valid keys: ['phi', 'unit_weight']" in msg

    def test_hints_offer_only_valid_keys_and_notes_are_added(self):
        with pytest.raises(ValueError) as e:
            check_nested_keys({"cohesion": 5}, ("c_prime", "phi"), method="m",
                              label="x", hints={"cohesion": ("c_prime", "cu")},
                              notes={("c_prime",): "(drained.)",
                                     ("c_prime", "cu"): "(both.)"})
        msg = str(e.value)
        assert "cohesion -> c_prime?" in msg and "cu" not in msg
        assert "(drained.)" in msg and "(both.)" not in msg

    def test_known_keys_pass_and_nothing_is_renamed(self):
        item = {"phi": 30}
        check_nested_keys(item, ("phi",), method="m", label="x")
        assert item == {"phi": 30}

    def test_non_dict_item_refused(self):
        with pytest.raises(ValueError, match=r"x must be a dict"):
            check_nested_keys([1, 2], ("a",), method="m", label="x")

    def test_list_labels_by_index_and_name(self):
        with pytest.raises(ValueError, match=r"layers\[1\] \('Clay'\)"):
            check_nested_list([{"a": 1}, {"name": "Clay", "b": 2}],
                              ("a", "name"), method="m", label="layers")
        check_nested_list(None, ("a",), method="m", label="x")   # no list

    def test_field_keys(self):
        from pile_group import GroupPile
        keys = field_keys(GroupPile, "extra")
        assert keys[:2] == ("x", "y") and keys[-1] == "extra"

    def test_slope_module_still_exports_it(self):
        from funhouse_agent.adapters import slope_stability
        assert slope_stability.check_nested_keys is check_nested_keys


# ---------------------------------------------------------------------------
# Foundations: axial pile, drilled shaft, pile group
# ---------------------------------------------------------------------------

_AXIAL = {"pile_type": "pipe_closed", "diameter": 0.4, "wall_thickness": 0.012,
          "pile_length": 15.0, "gwt_depth": 3.0}


class TestAxialPile:
    def _layer(self, **kw):
        d = {"thickness": 20.0, "soil_type": "cohesionless",
             "unit_weight": 19.0, "friction_angle": 32.0}
        d.update(kw)
        return d

    def test_misspelt_layer_key_refused(self):
        lay = self._layer(phi=30)
        del lay["friction_angle"]
        _refused("axial_pile", "axial_pile_capacity",
                 dict(_AXIAL, layers=[lay]),
                 r"layers\[0\] has unknown key\(s\) \['phi'\]",
                 r"phi -> friction_angle", r"Valid keys")

    def test_correct_layers_run(self):
        r = _registry("axial_pile")["axial_pile_capacity"](
            dict(_AXIAL, layers=[self._layer(toe_phi=34, description="sand")]))
        assert r["Q_ultimate_kN"] > 0


_SHAFT = {"diameter": 1.0, "shaft_length": 12.0}


class TestDrilledShaft:
    def test_misspelt_layer_key_refused(self):
        _refused("drilled_shaft", "drilled_shaft_capacity",
                 dict(_SHAFT, layers=[{"thickness": 15, "soil_type": "cohesive",
                                       "unit_weight": 18, "cohesion": 60}]),
                 r"layers\[0\].*\['cohesion'\]", r"cohesion -> cu")

    def test_correct_layers_run(self):
        r = _registry("drilled_shaft")["drilled_shaft_capacity"](
            dict(_SHAFT, layers=[{"thickness": 15, "soil_type": "cohesive",
                                  "unit_weight": 18, "cu": 60, "N160": 0}]))
        assert r["Q_ultimate_kN"] > 0


class TestPileGroup:
    def test_misspelt_pile_key_refused(self):
        _refused("pile_group", "pile_group_simple",
                 {"piles": [{"x": 0, "y": 0}, {"x": 2, "y": 0, "batter": 10}],
                  "Vz": 1000},
                 r"piles\[1\].*\['batter'\]", r"batter -> batter_x or batter_y")

    def test_correct_piles_run(self):
        r = _registry("pile_group")["pile_group_simple"](
            {"piles": [{"x": -1, "y": 0, "label": "P1"},
                       {"x": 1, "y": 0, "batter_x": 5, "label": "P2"}],
             "Vz": 1000})
        assert r["n_piles"] == 2


# ---------------------------------------------------------------------------
# Earth retention: sheet pile, SOE
# ---------------------------------------------------------------------------

class TestSheetPile:
    def test_misspelt_layer_key_refused(self):
        _refused("sheet_pile", "cantilever_wall",
                 {"excavation_depth": 4,
                  "layers": [{"thickness": 12, "unit_weight": 18,
                              "phi_deg": 32}]},
                 r"layers\[0\].*\['phi_deg'\]", r"phi_deg -> phi")

    def test_aliases_still_accepted(self):
        r = _registry("sheet_pile")["cantilever_wall"](
            {"excavation_depth": 4,
             "layers": [{"thickness": 12, "gamma": 18, "phi": 32}]})
        assert r["embedment_depth_m"] > 0


_SOE_LAYERS = [{"thickness": 15, "unit_weight": 19, "friction_angle": 32,
                "soil_type": "sand"}]


class TestSoe:
    def test_misspelt_layer_key_refused(self):
        _refused("soe", "braced_excavation",
                 {"excavation_depth": 6,
                  "layers": [{"thickness": 15, "unit_weight": 19, "phi": 32}],
                  "supports": [{"depth": 1.5}]},
                 r"layers\[0\].*\['phi'\]", r"phi -> friction_angle")

    def test_misspelt_support_key_refused(self):
        _refused("soe", "braced_excavation",
                 {"excavation_depth": 6, "layers": _SOE_LAYERS,
                  "supports": [{"depth": 1.5, "preload": 50}]},
                 r"supports\[0\].*\['preload'\]",
                 r"preload -> preload_kN_per_m")

    def test_correct_call_runs(self):
        r = _registry("soe")["braced_excavation"](
            {"excavation_depth": 6, "layers": _SOE_LAYERS,
             "supports": [{"depth": 1.5, "support_type": "strut",
                           "spacing": 3.0}, {"depth": 4.0}]})
        assert "error" not in r


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

_CONSOL = {"thickness": 4, "depth_to_center": 5, "e0": 1.0, "Cc": 0.3,
           "Cr": 0.05, "sigma_v0": 60}


class TestSettlement:
    def test_consolidation_preconsolidation_alias_refused(self):
        _refused("settlement", "consolidation_settlement",
                 {"layers": [dict(_CONSOL, pc=90)], "delta_sigma": 40},
                 r"layers\[0\].*\['pc'\]", r"pc -> sigma_p")

    def test_schmertmann_and_hough_and_combined_refused(self):
        _refused("settlement", "schmertmann_settlement",
                 {"q_net": 150, "B": 2,
                  "layers": [{"depth_top": 0, "depth_bottom": 2, "E": 9000}]},
                 r"layers\[0\].*\['E'\]", r"E -> Es")
        _refused("settlement", "hough_settlement",
                 {"q_net": 150, "B": 2,
                  "layers": [{"thickness": 2, "depth_to_center": 1,
                              "sigma_v0": 20, "C_prime": 50, "desc": "x"}]},
                 r"layers\[0\].*\['desc'\]")
        _refused("settlement", "combined_settlement_analysis",
                 {"q_applied": 150, "B": 2,
                  "consolidation_layers": [dict(_CONSOL, OCR=1.5)]},
                 r"consolidation_layers\[0\].*\['OCR'\]")

    def test_correct_calls_run(self):
        r = _registry("settlement")["consolidation_settlement"](
            {"layers": [dict(_CONSOL, sigma_p=90, description="clay")],
             "delta_sigma": 40})
        assert r["consolidation_settlement_mm"] > 0
        r = _registry("settlement")["hough_settlement"](
            {"q_net": 150, "B": 2,
             "layers": [{"thickness": 2, "depth_to_mid": 1, "sigma_vo": 20,
                         "C": 50}]})
        assert r


# ---------------------------------------------------------------------------
# Lateral pile (per p-y model)
# ---------------------------------------------------------------------------

_LAT = {"pile_length": 15, "pile_diameter": 0.6, "Vt": 50}


class TestLateralPile:
    def test_key_of_another_model_refused(self):
        _refused("lateral_pile", "lateral_pile_analysis",
                 dict(_LAT, layers=[{"top": 0, "bottom": 15,
                                     "model": "SandAPI", "phi": 34,
                                     "gamma": 9, "k": 16000, "J": 0.5}]),
                 r"layers\[0\] \(SandAPI\).*\['J'\]")

    def test_misspelt_optional_key_refused(self):
        _refused("lateral_pile", "lateral_pile_analysis",
                 dict(_LAT, layers=[{"top": 0, "bottom": 15,
                                     "model": "SoftClayMatlock", "c": 30,
                                     "gamma": 8, "eps50": 0.01,
                                     "Loading": "cyclic"}]),
                 r"\['Loading'\]", r"Loading -> loading")

    def test_get_py_curve_top_level_checked(self):
        _refused("lateral_pile", "get_py_curve",
                 {"model": "WeakRock", "qu": 5000, "Er": 1e6, "rqd": 50},
                 r"unknown parameter.*rqd", r"rqd -> RQD")

    def test_correct_layers_run(self):
        r = _registry("lateral_pile")["lateral_pile_analysis"](
            dict(_LAT, layers=[{"top": 0, "bottom": 15, "model": "SandAPI",
                                "phi": 34, "gamma": 9, "k": 16000,
                                "loading": "static", "description": "sand"}]))
        assert r["y_top_m"] != 0


# ---------------------------------------------------------------------------
# fem2d (the checks run before any mesh is built)
# ---------------------------------------------------------------------------

_SRM = {"surface_points": [[0, 10], [10, 10], [30, 0], [40, 0]]}


class TestFem2d:
    def test_srm_layer_cohesion_refused_with_note(self):
        msg = _refused("fem2d_adapter", "fem2d_slope_srm",
                       dict(_SRM, soil_layers=[
                           {"name": "Clay", "bottom_elevation": -10,
                            "E": 20000, "nu": 0.3, "cohesion": 20, "phi": 25,
                            "gamma": 18}]),
                       r"soil_layers\[0\] \('Clay'\).*\['cohesion'\]",
                       r"cohesion -> c\b")
        assert "phi = 0" in msg

    def test_srm_hs_keys_refused_where_not_read(self):
        _refused("fem2d_adapter", "fem2d_slope_srm",
                 dict(_SRM, soil_layers=[
                     {"bottom_elevation": -10, "E": 20000, "c": 5, "phi": 30,
                      "gamma": 18, "E50_ref": 30000}]),
                 r"\['E50_ref'\]")

    def test_excavation_layer_and_strut_keys(self):
        base = {"width": 20, "depth": 15, "wall_depth": 10, "wall_EI": 1e5,
                "wall_EA": 1e7}
        lay = {"bottom_elevation": -15, "E": 30000, "nu": 0.3, "c": 0,
               "phi": 32, "gamma": 19, "K0": 0.5, "k": 1e-5}
        _refused("fem2d_adapter", "fem2d_excavation",
                 dict(base, soil_layers=[dict(lay, unit_weight=19)]),
                 r"\['unit_weight'\]", r"unit_weight -> gamma")
        _refused("fem2d_adapter", "fem2d_excavation",
                 dict(base, soil_layers=[lay],
                      struts=[{"depth": 2, "k": 1e5}]),
                 r"struts\[0\].*\['k'\]", r"k -> stiffness")

    def test_consolidation_layer_key_refused(self):
        _refused("fem2d_adapter", "fem2d_consolidation",
                 {"width": 10, "depth": 10, "k": 1e-8, "load_q": 50,
                  "time_points": [1e5],
                  "soil_layers": [{"bottom_elevation": -10, "E": 5000,
                                   "nu": 0.3, "gamma": 18, "is_fill": True}]},
                 r"\['is_fill'\]")

    def test_staged_items_refused(self):
        base = {"nodes": [[0, 0], [1, 0], [0, 1]], "elements": [[0, 1, 2]],
                "gamma": 18, "element_groups": {"all": [0]},
                "material_props": [{"E": 1e4, "nu": 0.3}],
                "phases": [{"name": "G", "active_soil_groups": ["all"]}]}
        _refused("fem2d_adapter", "fem2d_staged",
                 dict(base, phases=[{"name": "G", "active_groups": ["all"]}]),
                 r"phases\[0\] \('G'\).*\['active_groups'\]",
                 r"active_groups -> active_soil_groups")
        _refused("fem2d_adapter", "fem2d_staged",
                 dict(base, material_props=[{"E": 1e4, "poisson": 0.3}]),
                 r"material_props\[0\].*poisson -> nu")
        _refused("fem2d_adapter", "fem2d_staged",
                 dict(base, beam_elements=[{"node_i": 0, "node_j": 1,
                                            "EA": 1e6, "EI": 1e4,
                                            "weight": 1}]),
                 r"beam_elements\[0\].*weight -> weight_per_m")


# ---------------------------------------------------------------------------
# Structural frames (pynite)
# ---------------------------------------------------------------------------

def _frame(**over):
    p = {"nodes": [{"name": "A", "x": 0, "y": 0}, {"name": "B", "x": 5, "y": 0}],
         "members": [{"name": "M1", "i": "A", "j": "B", "E": 2e8, "A": 0.01,
                      "Iz": 1e-4}],
         "supports": [{"node": "A", "type": "pinned"},
                      {"node": "B", "type": "roller_y"}],
         "member_dist_loads": [{"member": "M1", "w": -10}]}
    p.update(over)
    return p


class TestPynite:
    @pytest.fixture(autouse=True)
    def _need_pynite(self):
        from pynite_agent import has_pynite
        if not has_pynite():
            pytest.skip("PyNiteFEA not installed")

    def test_support_flag_case_refused(self):
        _refused("pynite_adapter", "frame_analysis",
                 _frame(supports=[{"node": "A", "DX": True, "DY": True},
                                  {"node": "B", "type": "roller_y"}]),
                 r"supports\[0\].*\['DX', 'DY'\]", r"DX -> dx; DY -> dy")

    def test_member_and_load_keys_refused(self):
        _refused("pynite_adapter", "frame_analysis",
                 _frame(members=[{"name": "M1", "i": "A", "j": "B", "E": 2e8,
                                  "A": 0.01, "Iz": 1e-4, "poisson": 0.3}]),
                 r"members\[0\] \('M1'\).*\['poisson'\]")
        _refused("pynite_adapter", "continuous_beam",
                 {"span_lengths": [5, 5], "E": 2e8, "I": 1e-4,
                  "point_loads": [{"span": 1, "x": 2, "load": 10}]},
                 r"point_loads\[0\].*load -> P")

    def test_correct_frame_runs(self):
        r = _registry("pynite_adapter")["frame_analysis"](_frame())
        assert "error" not in r


# ---------------------------------------------------------------------------
# Site response (pystrata, opensees): keys per soil model / material
# ---------------------------------------------------------------------------

class TestSiteResponse:
    def test_pystrata_model_keys(self):
        from pystrata_agent import has_pystrata
        if not has_pystrata():
            pytest.skip("pystrata not installed")
        _refused("pystrata_adapter", "linear_site_response",
                 {"layers": [{"thickness": 10, "Vs": 200, "unit_wt": 18,
                              "soil_model": "darendeli", "PI": 20},
                             {"thickness": 0, "Vs": 760, "unit_wt": 22,
                              "soil_model": "linear", "damping": 0.01}]},
                 r"layers\[0\] \(darendeli\).*\['PI'\]", r"PI -> plas_index")

    def test_pystrata_confining_pressure_hint(self):
        from pystrata_agent import has_pystrata
        if not has_pystrata():
            pytest.skip("pystrata not installed")
        _refused("pystrata_adapter", "linear_site_response",
                 {"layers": [{"thickness": 10, "Vs": 200, "unit_wt": 18,
                              "soil_model": "darendeli", "plas_index": 20,
                              "confining_pressure": 80},
                             {"thickness": 0, "Vs": 760, "unit_wt": 22,
                              "soil_model": "linear", "damping": 0.01}]},
                 r"confining_pressure -> stress_mean")

    def test_opensees_material_keys(self):
        from funhouse_agent.adapters import opensees_adapter
        with patch.object(opensees_adapter, "_check_opensees",
                          lambda: None):
            _refused("opensees_adapter", "site_response_1d",
                     {"layers": [{"thickness": 5, "Vs": 200, "density": 1.9,
                                  "material_type": "clay", "cu": 50}]},
                     r"layers\[0\] \(clay\).*\['cu'\]", r"cu -> su")
            fake = MagicMock()
            fake.return_value.to_dict.return_value = {"pga_surface_g": 0.3}
            with patch("opensees_agent.analyze_site_response", fake):
                r = _registry("opensees_adapter")["site_response_1d"](
                    {"layers": [{"thickness": 5, "Vs": 200, "density": 1.9,
                                 "material_type": "sand", "phi": 33,
                                 "K0": 0.5, "e_init": 0.7}],
                     "motion": "synthetic_pulse"})
        assert r == {"pga_surface_g": 0.3}


# ---------------------------------------------------------------------------
# Drawings in and out: dxf_export, dxf_import, pdf_import
# ---------------------------------------------------------------------------

class TestDrawings:
    def test_dxf_export_annotation_z_refused(self, tmp_path):
        _refused("dxf_export", "export_geometry_to_dxf",
                 {"output_path": str(tmp_path / "s.dxf"),
                  "surface_points": [[0, 0], [10, 0]],
                  "text_annotations": [{"text": "Clay", "x": 2, "z": -3}]},
                 r"text_annotations\[0\].*\['z'\]", r"z -> y")

    def test_dxf_export_correct_call_writes(self, tmp_path):
        r = _registry("dxf_export")["export_geometry_to_dxf"](
            {"output_path": str(tmp_path / "s.dxf"),
             "surface_points": [[0, 0], [10, 0]],
             "nail_lines": [{"x_head": 1, "z_head": 0, "x_tip": 5,
                             "z_tip": -1}],
             "text_annotations": [{"text": "Clay", "x": 2, "y": -3,
                                   "layer": "TEXT"}]})
        assert r["file_exists"]

    def test_dxf_import_soil_properties_and_mapping(self):
        pr = {"surface_points": [{"x": 0, "z": 10}, {"x": 20, "z": 10}],
              "boundary_profiles": {}}
        msg = _refused("dxf_import_adapter", "build_slope_geometry",
                       {"parse_result": pr,
                        "soil_properties": [{"name": "Surface", "gamma": 18,
                                             "phi": 30, "c": 5}]},
                       r"soil_properties\[0\] \('Surface'\).*\['c'\]",
                       r"c -> c_prime or cu")
        assert "undrained" in msg
        _refused("dxf_import_adapter", "build_fem_inputs",
                 {"parse_result": pr,
                  "soil_properties": [{"name": "Surface", "gamma": 18,
                                       "unit_weight": 18}]},
                 r"unit_weight -> gamma")
        _refused("dxf_import_adapter", "parse_geometry",
                 {"file_path": "missing.dxf",
                  "layer_mapping": {"surface": "GROUND", "gwt": "WATER"}},
                 r"layer_mapping has unknown key\(s\) \['gwt'\]",
                 r"gwt -> water_table")
        _refused("dxf_import_adapter", "build_slope_geometry",
                 {"parse_result": pr,
                  "soil_properties": [{"name": "Surface", "gamma": 18}],
                  "nail_defaults": {"fy_mpa": 420}},
                 r"nail_defaults.*fy_mpa -> fy")

    def test_dxf_import_correct_call_runs(self):
        r = _registry("dxf_import_adapter")["build_fem_inputs"](
            {"parse_result": {"surface_points": [{"x": 0, "z": 10},
                                                 {"x": 20, "z": 10}],
                              "boundary_profiles": {}},
             "soil_properties": [{"name": "Surface", "gamma": 18, "phi": 30,
                                  "c": 5, "E": 20000, "model": "mc"}]})
        assert r["soil_layers"][0]["c"] == 5

    def test_dxf_fem_output_with_an_hs_layer_runs_fem2d(self):
        """build_fem_inputs' layers are fem2d input as they come: an HS
        layer's E50_ref, Eur_ref, m, p_ref and R_f flat in the layer (a
        nested hs_params was never read; fem2d raised KeyError E50_ref)."""
        from funhouse_agent.dispatch import call_agent
        built = call_agent("dxf_import", "build_fem_inputs", {
            "parse_result": {"surface_points": [[0, 0], [20, 0]],
                             "boundary_profiles": {"Clay": [[0, -6],
                                                            [20, -6]]}},
            "soil_properties": [
                {"name": "Surface", "gamma": 18, "phi": 30, "c": 1,
                 "E": 40000, "nu": 0.3},
                {"name": "Clay", "gamma": 18, "phi": 28, "c": 5, "E": 20000,
                 "nu": 0.3, "model": "hs",
                 "hs_params": {"E50_ref": 25000, "Eur_ref": 75000, "m": 0.5,
                               "p_ref": 100, "R_f": 0.9}}]})
        assert "error" not in built, built
        clay = built["soil_layers"][1]
        assert clay["E50_ref"] == 25000 and "hs_params" not in clay
        r = call_agent("fem2d", "fem2d_excavation", {
            "width": 10, "depth": 2, "wall_depth": 5,
            "soil_layers": built["soil_layers"], "wall_EI": 5e4,
            "wall_EA": 5e6, "nx": 8, "ny": 8, "n_steps": 2})
        assert "error" not in r, r

    def test_pdf_import_soil_properties(self):
        pytest.importorskip("planlens.pdf")
        _refused("pdf_import_adapter", "build_slope_geometry",
                 {"parse_result": {"surface_points": [[0, 10], [20, 10]],
                                   "boundary_profiles": {}},
                  "soil_properties": [{"name": "Surface", "gamma": 18,
                                       "cohesion": 5}]},
                 r"soil_properties\[0\].*\['cohesion'\]")


# ---------------------------------------------------------------------------
# Subsurface site data
# ---------------------------------------------------------------------------

_SITE = {"project_name": "T",
         "investigations": [{"investigation_id": "B-1", "x": 0, "y": 0,
                             "total_depth_m": 10,
                             "measurements": [{"depth_m": 1.5,
                                               "parameter": "N_spt",
                                               "value": 10}]}]}


class TestSubsurface:
    def test_measurement_depth_refused(self):
        import copy
        site = copy.deepcopy(_SITE)
        site["investigations"][0]["measurements"][0] = {
            "depth": 1.5, "parameter": "N_spt", "value": 10}
        _refused("subsurface_adapter", "load_site", {"site_data": site},
                 r"site_data\.investigations\[0\]\.measurements\[0\]"
                 r".*\['depth'\]", r"depth -> depth_m")

    def test_investigation_key_refused(self):
        import copy
        site = copy.deepcopy(_SITE)
        site["investigations"][0]["gwl"] = 3.0
        _refused("subsurface_adapter", "load_site", {"site_data": site},
                 r"\['gwl'\]", r"gwl -> gwl_depth_m")

    def test_round_trip_of_its_own_output_loads(self):
        from subsurface_characterization import load_site_from_dict
        site = load_site_from_dict(_SITE).to_dict()
        r = _registry("subsurface_adapter")["load_site"]({"site_data": site})
        assert "error" not in r


# ---------------------------------------------------------------------------
# Figures (profile_figure)
# ---------------------------------------------------------------------------

class TestProfileFigure:
    def test_series_name_refused(self, tmp_path):
        _refused("profile_figure_adapter", "plot_data",
                 {"series": [{"x": [1, 2], "y": [3, 4], "name": "SPT"}],
                  "output_path": str(tmp_path / "p.png")},
                 r"series\[0\].*\['name'\]", r"name -> label")

    def test_reference_line_and_callout_refused(self, tmp_path):
        _refused("profile_figure_adapter", "plot_data",
                 {"series": [{"x": [1, 2], "y": [3, 4]}],
                  "hlines": [2.5, {"value": 3, "text": "GWT"}],
                  "output_path": str(tmp_path / "p.png")},
                 r"hlines\[1\].*text -> label")
        _refused("profile_figure_adapter", "subsurface_profile",
                 {"layers": [{"name": "Clay", "thickness": 5}],
                  "annotations": [{"elev": -2, "text": "NP"}],
                  "output_path": str(tmp_path / "s.png")},
                 r"annotations\[0\].*elev -> elevation")

    def test_correct_plot_runs(self, tmp_path):
        r = _registry("profile_figure_adapter")["plot_data"](
            {"series": [{"x": [1, 2], "y": [3, 4], "label": "SPT",
                         "style": "markers", "labels": ["a", "b"]}],
             "hlines": [{"value": 3.5, "label": "limit"}],
             "interactive": False, "output_path": str(tmp_path / "p.png")})
        assert r["file_exists"]

    def test_foundation_top_is_its_head(self, tmp_path):
        """A pile's 'top_elevation' is its head; the near-miss matcher
        alone would offer tip_elevation."""
        _refused("profile_figure_adapter", "subsurface_profile",
                 {"layers": [{"name": "Clay", "thickness": 10}],
                  "foundation": {"type": "pile", "diameter": 0.4,
                                 "top_elevation": 0, "tip_elevation": -8},
                  "output_path": str(tmp_path / "s.png")},
                 r"top_elevation -> head_elevation\?")

    def test_profile_layers_stay_open(self, tmp_path):
        """An analysis' own layer dicts draw as they are."""
        r = _registry("profile_figure_adapter")["subsurface_profile"](
            {"layers": [{"name": "Clay", "thickness": 5, "unit_weight": 18,
                         "cu": 30}],
             "output_path": str(tmp_path / "s.png")})
        assert r["file_exists"]


# ---------------------------------------------------------------------------
# Pavement design
# ---------------------------------------------------------------------------

class TestPavement:
    def test_layer_and_vehicle_keys_refused(self):
        _refused("pavement_design_adapter", "flexible_pavement_design",
                 {"w18": 5e6, "reliability_pct": 95, "so": 0.45,
                  "mr_psi": 5000,
                  "layers": [{"layer_type": "asphalt", "modulus": 450000}]},
                 r"layers\[0\].*modulus -> modulus_psi")
        _refused("pavement_design_adapter", "design_traffic_esals",
                 {"growth_rate_pct": 2, "design_period_yr": 20,
                  "vehicles": [{"daily_count": 100, "tf": 1.2}]},
                 r"vehicles\[0\].*tf -> truck_factor")

    def test_composite_k_season_esb_refused(self):
        _refused("pavement_design_adapter", "rigid_pavement_design",
                 {"w18": 5e6, "sc_psi": 650, "ec_psi": 5e6,
                  "reliability_pct": 95, "so": 0.35, "j": 3.2, "cd": 1.0,
                  "composite_k": {"seasonal": [{"mr_psi": 6000, "esb": 2e4}],
                                  "dsb_in": 6.0, "esb_psi": 2e4}},
                 r"composite_k\.seasonal\[0\].*esb -> esb_psi")

    def test_correct_calls_run(self):
        r = _registry("pavement_design_adapter")["design_traffic_esals"](
            {"growth_rate_pct": 2, "design_period_yr": 20,
             "vehicles": [{"description": "trucks", "daily_count": 100,
                           "truck_factor": 1.2}]})
        assert "error" not in r
        r = _registry("pavement_design_adapter")["mixed_traffic_ufc"](
            {"representative_cbr": 6,
             "vehicles": [{"vehicle": "passenger_car", "name": "cars",
                           "design_passes": 1000}]})
        assert "error" not in r


# ---------------------------------------------------------------------------
# Calc packages that build their own layers
# ---------------------------------------------------------------------------

class TestCalcPackages:
    def test_axial_and_drilled_and_sheet_layers_refused(self):
        _refused("calc_package", "axial_pile_package",
                 {"pile_area": 0.01, "pile_perimeter": 1.2,
                  "pile_tip_area": 0.1, "pile_width": 0.4, "pile_length": 15,
                  "soil_layers": [{"soil_type": "cohesionless",
                                   "thickness": 20, "unit_weight": 19,
                                   "phi": 32}]},
                 r"soil_layers\[0\].*phi -> friction_angle")
        _refused("calc_package", "drilled_shaft_package",
                 {"diameter": 1.0, "length": 12,
                  "soil_layers": [{"soil_type": "cohesive", "thickness": 15,
                                   "unit_weight": 18, "su": 60}]},
                 r"soil_layers\[0\].*su -> cu")
        _refused("calc_package", "sheet_pile_package",
                 {"excavation_depth": 4,
                  "soil_layers": [{"thickness": 12, "unit_weight": 18,
                                   "phi_deg": 32}]},
                 r"soil_layers\[0\].*phi_deg -> phi")

    def test_drilled_package_says_where_rational_inputs_go(self):
        msg = _refused("calc_package", "drilled_shaft_package",
                       {"diameter": 1.0, "length": 12,
                        "soil_layers": [{"soil_type": "cohesionless",
                                         "thickness": 15, "unit_weight": 19,
                                         "phi": 33, "N1_60": 20, "OCR": 2}]},
                       r"\['N1_60', 'OCR'\]", r"beta_method='rational'")
        assert "-> N60" not in msg                 # a different quantity
        assert msg.count("This package runs") == 1

    def test_sheet_package_reads_layers_as_the_tool_does(self, tmp_path):
        """The same layer dicts (aliases included) work for the analysis
        tool and its package, and the wall friction reaches the package."""
        layers = [{"thickness": 12, "gamma": 18, "phi": 32, "delta": 20,
                   "description": "sand"}]
        tool = _registry("sheet_pile")["cantilever_wall"](
            {"excavation_depth": 4, "layers": layers,
             "pressure_method": "coulomb"})
        assert tool["embedment_depth_m"] > 0

        def package(lays, name):
            return _registry("calc_package")["sheet_pile_package"](
                {"excavation_depth": 4, "soil_layers": lays,
                 "pressure_method": "coulomb",
                 "output_path": str(tmp_path / name)})
        with_delta = package(layers, "a.html")
        assert with_delta["status"] == "success"
        no_delta = package([dict(layers[0], delta=0)], "b.html")
        assert with_delta["embedment_m"] != no_delta["embedment_m"]

    def test_axial_package_reads_layers_as_the_tool_does(self, tmp_path):
        lay = {"soil_type": "cohesionless", "thickness": 25.0,
               "unit_weight": 18.0, "friction_angle": 30.0,
               "description": "sand"}
        r = _registry("axial_pile")["axial_pile_capacity"](
            dict(_AXIAL, layers=[dict(lay, toe_phi=36.0)]))
        assert r["Q_tip_kN"] > 0

        def package(layer, name):
            return _registry("calc_package")["axial_pile_package"](
                {"pile_area": 0.01, "pile_perimeter": 1.2,
                 "pile_tip_area": 0.09, "pile_width": 0.3,
                 "pile_length": 20.0, "soil_layers": [layer],
                 "output_path": str(tmp_path / name)})
        plain = package(lay, "a.html")
        toe = package(dict(lay, toe_phi=36.0), "b.html")
        alias = package(dict(lay, toe_friction_angle=36.0), "c.html")
        assert toe["Q_tip_kN"] > plain["Q_tip_kN"]
        assert alias["Q_tip_kN"] == toe["Q_tip_kN"]

    def test_settlement_and_lateral_layers_refused(self):
        _refused("calc_package", "settlement_package",
                 {"q_applied": 100, "B": 2,
                  "consolidation_layers": [dict(_CONSOL, pc=90)]},
                 r"consolidation_layers\[0\].*pc -> sigma_p")
        _refused("calc_package", "lateral_pile_package",
                 {"pile_length": 15, "pile_diameter": 0.6, "pile_E": 2e8,
                  "soil_layers": [{"top": 0, "bottom": 15,
                                   "py_model": "SandAPI", "phi": 34,
                                   "gamma": 9, "k": 16000, "J": 0.5}]},
                 r"soil_layers\[0\] \(SandAPI\).*\['J'\]")


# ---------------------------------------------------------------------------
# Downdrag (the analysis, the CGPR #56 consolidation recompute, the package)
# ---------------------------------------------------------------------------

_DD_LAYERS = [
    {"thickness": 8.0, "soil_type": "cohesive", "unit_weight": 17.0,
     "cu": 30.0, "settling": True, "Cc": 0.3, "e0": 1.0,
     "description": "soft clay"},
    {"thickness": 15.0, "soil_type": "cohesionless", "unit_weight": 19.0,
     "phi": 35.0}]
_DD = {"pile_length": 20.0, "pile_diameter": 0.4, "fill_thickness": 2.0,
       "Q_dead": 400.0}


class TestDowndrag:
    def test_cr_is_offered_as_cr_and_the_modified_index_named(self):
        """'cr' in a layer is Cr (or the modified C_er = Cr/(1+e0), a
        different quantity, named as such), never C_er alone."""
        layers = [dict(_DD_LAYERS[0], cr=0.05), _DD_LAYERS[1]]
        msg = _refused("downdrag", "downdrag_analysis",
                       dict(_DD, layers=layers),
                       r"cr -> Cr or C_er", r"not Cr itself")
        assert "not Cc itself" not in msg

    def test_layer_preconsolidation_alias_refused(self):
        layers = [dict(_DD_LAYERS[0], pc=60.0), _DD_LAYERS[1]]
        _refused("downdrag", "downdrag_analysis", dict(_DD, layers=layers),
                 r"layers\[0\] \('soft clay'\).*\['pc'\]",
                 r"pc -> sigma_p\b")

    def test_consolidation_dict_and_layers_checked(self):
        cons = {"layers": [{"z_top": 6.0, "z_bot": 27.0, "C_er": 0.015,
                            "Cc": 0.2}],
                "p0_profile": [[0.0, 0.0], [27.0, 190.0]],
                "dp_profile": 25.0}
        base = {"Q_static": 900.0, "pile_length": 30.0,
                "pile_perimeter": 1.6, "pile_area": 0.16, "pile_E": 2.4e7,
                "toe_bearing_capacity": 9500.0,
                "skin_friction_profile": [[0, 0], [30, 60]],
                "eq_footing_width": 3.5}
        msg = _refused("downdrag", "fellenius_cgpr56",
                       dict(base, consolidation=cons),
                       r"consolidation\.layers\[0\].*\['Cc'\]",
                       r"Cc -> C_ec")
        assert "Cc/(1+e0)" in msg
        _refused("downdrag", "fellenius_cgpr56",
                 dict(base, consolidation=dict(cons, layers=cons["layers"][:0],
                                               sublayer_thick=1.0)),
                 r"consolidation has unknown key\(s\) \['sublayer_thick'\]",
                 r"sublayer_thick -> sublayer_thickness")

    def test_package_layers_refused(self, tmp_path):
        _refused("calc_package", "downdrag_package",
                 dict(_DD, pile_E=2e8,
                      soil_layers=[dict(_DD_LAYERS[0], gamma=17.0),
                                   _DD_LAYERS[1]],
                      output_path=str(tmp_path / "dd.html")),
                 r"soil_layers\[0\].*\['gamma'\]",
                 r"gamma -> unit_weight")

    def test_correct_call_runs(self):
        r = _registry("downdrag")["downdrag_analysis"](
            dict(_DD, layers=_DD_LAYERS,
                 influence_depth_criterion="effective_stress"))
        assert r["neutral_plane_depth_m"] > 0


def test_dispatch_returns_the_message():
    from funhouse_agent.dispatch import call_agent
    r = call_agent("axial_pile", "axial_pile_capacity",
                   dict(_AXIAL, layers=[{"thickness": 20,
                                         "soil_type": "cohesionless",
                                         "unit_weight": 19, "phi": 32}]))
    assert "phi -> friction_angle" in r["error"]
