"""Tests for funhouse_agent calc_package adapter.

Covers:
- Helper functions (_extract_metadata, _default_output_path)
- All 13 handler functions (integration: build objects → analysis → calc package → file)
- Dispatch integration (call_agent routing)
- Edge cases (auto output path, METHOD_INFO completeness)
"""

import os
import pytest
import matplotlib
matplotlib.use("Agg")  # avoid Tk backend errors


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_metadata_defaults(self):
        from funhouse_agent.adapters.calc_package import _extract_metadata
        meta = _extract_metadata({})
        assert meta["project_name"] == "Project"
        assert meta["engineer"] == ""
        assert meta["company"] == ""

    def test_extract_metadata_custom(self):
        from funhouse_agent.adapters.calc_package import _extract_metadata
        meta = _extract_metadata({
            "project_name": "Bridge 42",
            "engineer": "J. Doe",
            "company": "ACME",
        })
        assert meta["project_name"] == "Bridge 42"
        assert meta["engineer"] == "J. Doe"
        assert meta["company"] == "ACME"

    def test_default_output_path_html(self):
        from funhouse_agent.adapters.calc_package import _default_output_path
        path = _default_output_path("bearing_capacity")
        assert path.startswith("bearing_capacity_calc_")
        assert path.endswith(".html")

    def test_default_output_path_pdf(self):
        from funhouse_agent.adapters.calc_package import _default_output_path
        path = _default_output_path("settlement", fmt="pdf")
        assert path.endswith(".pdf")

    def test_method_info_completeness(self):
        from funhouse_agent.adapters.calc_package import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_info_has_common_params(self):
        from funhouse_agent.adapters.calc_package import METHOD_INFO
        for name, info in METHOD_INFO.items():
            params = info["parameters"]
            assert "project_name" in params, f"{name} missing project_name"
            assert "output_path" in params, f"{name} missing output_path"
            assert "format" in params, f"{name} missing format"


# ---------------------------------------------------------------------------
# Integration tests — one per handler
# ---------------------------------------------------------------------------

class TestBearingCapacityPackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_bearing_capacity_package
        outfile = str(tmp_path / "bc.html")
        result = _generate_bearing_capacity_package({
            "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
            "depth": 1.5, "shape": "square",
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["q_ultimate_kPa"] > 0
        assert result["q_allowable_kPa"] > 0
        assert os.path.exists(outfile)
        assert result["html_length"] > 100


class TestLateralPilePackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        outfile = str(tmp_path / "lp.html")
        result = _generate_lateral_pile_package({
            "pile_length": 15.0, "pile_diameter": 0.5,
            "pile_E": 200e6, "Vt": 100.0,
            "soil_layers": [
                {"top": 0, "bottom": 15, "py_model": "SandReese",
                 "phi": 35, "gamma": 18.0, "k": 30000},
            ],
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert "y_top_mm" in result
        assert os.path.exists(outfile)


class TestSlopeStabilityPackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_slope_stability_package
        outfile = str(tmp_path / "ss.html")
        result = _generate_slope_stability_package({
            "surface_points": [[0, 10], [10, 10], [20, 5], [30, 5]],
            "soil_layers": [{
                "name": "Clay", "top_elevation": 10, "bottom_elevation": 0,
                "gamma": 18.0, "phi": 25, "c_prime": 10,
            }],
            "xc": 15, "yc": 18, "radius": 13,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["FOS"] > 0
        assert os.path.exists(outfile)


class TestSettlementPackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_settlement_package
        outfile = str(tmp_path / "settle.html")
        result = _generate_settlement_package({
            "q_applied": 150.0, "B": 2.0, "L": 2.0,
            "Es_immediate": 15000.0,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert "total_settlement_mm" in result
        assert os.path.exists(outfile)


class TestAxialPilePackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_axial_pile_package
        outfile = str(tmp_path / "ap.html")
        result = _generate_axial_pile_package({
            "pile_name": "HP12x53",
            "pile_type": "h_pile",
            "pile_area": 0.01,
            "pile_perimeter": 1.2,
            "pile_tip_area": 0.09,
            "pile_width": 0.3,
            "pile_length": 20.0,
            "soil_layers": [{
                "soil_type": "cohesionless",
                "thickness": 25.0,
                "unit_weight": 18.0,
                "friction_angle": 32.0,
            }],
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["Q_ultimate_kN"] > 0
        assert os.path.exists(outfile)


class TestDrilledShaftPackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_drilled_shaft_package
        outfile = str(tmp_path / "ds.html")
        result = _generate_drilled_shaft_package({
            "diameter": 1.0, "length": 15.0,
            "soil_layers": [{
                "soil_type": "cohesive",
                "thickness": 20.0,
                "unit_weight": 18.0,
                "cu": 75.0,
            }],
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["Q_ultimate_kN"] > 0
        assert os.path.exists(outfile)


class TestDowndragPackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_downdrag_package
        outfile = str(tmp_path / "dd.html")
        result = _generate_downdrag_package({
            "pile_length": 20.0,
            "pile_diameter": 0.3,
            "pile_E": 200e6,
            "Q_dead": 500.0,
            "soil_layers": [
                {"soil_type": "cohesive", "thickness": 8.0,
                 "unit_weight": 17.0, "cu": 30.0, "settling": True,
                 "Cc": 0.3, "e0": 1.0},
                {"soil_type": "cohesionless", "thickness": 15.0,
                 "unit_weight": 19.0, "phi": 35.0, "settling": False},
            ],
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert "neutral_plane_m" in result
        assert os.path.exists(outfile)


class TestSeismicPackage:
    def test_site_classification(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_seismic_package
        outfile = str(tmp_path / "seismic.html")
        result = _generate_seismic_package({
            "analysis_type": "site_classification",
            "vs30": 270.0, "Ss": 1.0, "S1": 0.4,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["seismic_analysis_type"] == "site_classification"
        assert os.path.exists(outfile)


class TestRetainingWallPackage:
    def test_cantilever(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_retaining_wall_package
        outfile = str(tmp_path / "rw.html")
        result = _generate_retaining_wall_package({
            "wall_height": 4.0,
            "gamma_backfill": 18.0,
            "phi_backfill": 30.0,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert "FOS_sliding" in result
        assert os.path.exists(outfile)


class TestGroundImprovementPackage:
    def test_wick_drains(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_ground_improvement_package
        outfile = str(tmp_path / "gi.html")
        result = _generate_ground_improvement_package({
            "method": "wick_drains",
            "spacing": 1.5, "ch": 3.0, "cv": 1.0,
            "Hdr": 5.0, "time": 1.0,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["method"] == "wick_drains"
        assert os.path.exists(outfile)


class TestWaveEquationPackage:
    def test_basic(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_wave_equation_package
        outfile = str(tmp_path / "we.html")
        result = _generate_wave_equation_package({
            "hammer_name": "Delmag D30-32",
            "cushion_area": 0.1,
            "cushion_thickness": 0.1,
            "cushion_E": 3.5e9,
            "pile_length": 15.0,
            "pile_area": 0.01,
            "pile_E": 200e9,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["n_points"] > 0
        assert os.path.exists(outfile)


class TestPileGroupPackage:
    def test_rectangular(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_pile_group_package
        outfile = str(tmp_path / "pg.html")
        result = _generate_pile_group_package({
            "layout": "rectangular",
            "n_rows": 3, "n_cols": 3,
            "spacing_x": 1.5, "spacing_y": 1.5,
            "Vz": 1800.0,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert result["n_piles"] == 9
        assert os.path.exists(outfile)


class TestSheetPilePackage:
    def test_cantilever(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_sheet_pile_package
        outfile = str(tmp_path / "sp.html")
        result = _generate_sheet_pile_package({
            "excavation_depth": 3.0,
            "soil_layers": [{
                "thickness": 10.0,
                "unit_weight": 18.0,
                "friction_angle": 30.0,
            }],
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert "embedment_m" in result
        assert os.path.exists(outfile)


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

class TestDispatchIntegration:
    def test_call_agent_bearing_capacity_package(self, tmp_path):
        from funhouse_agent.dispatch import call_agent
        outfile = str(tmp_path / "dispatch_bc.html")
        result = call_agent("calc_package", "bearing_capacity_package", {
            "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
            "output_path": outfile,
        })
        assert "error" not in result
        assert result["status"] == "success"
        assert os.path.exists(outfile)

    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("calc_package")
        # list_methods returns {category: {method: brief}}
        total = sum(len(v) for v in methods.values())
        assert total == 13

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("calc_package", "bearing_capacity_package")
        assert "parameters" in info
        assert "width" in info["parameters"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_auto_output_path(self):
        """When no output_path is provided, auto-generate one."""
        from funhouse_agent.adapters.calc_package import _generate_bearing_capacity_package
        result = _generate_bearing_capacity_package({
            "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
        })
        assert result["status"] == "success"
        assert result["output_path"].startswith("bearing_capacity_calc_")
        # Clean up auto-generated file
        if os.path.exists(result["output_path"]):
            os.remove(result["output_path"])

    def test_with_project_metadata(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_bearing_capacity_package
        outfile = str(tmp_path / "bc_meta.html")
        result = _generate_bearing_capacity_package({
            "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
            "project_name": "Bridge Abutment",
            "project_number": "2026-001",
            "engineer": "J. Doe",
            "company": "ACME Engineering",
            "output_path": outfile,
        })
        assert result["status"] == "success"
        html = open(outfile).read()
        assert "Bridge Abutment" in html
        assert "ACME Engineering" in html
