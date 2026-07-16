"""Tests for funhouse_agent calc_package adapter.

Covers:
- Helper functions (_extract_metadata, _default_output_path)
- All 14 handler functions (integration: build objects → analysis → calc package → file)
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
            assert "output_path" in params, f"{name} missing output_path"
            if name == "html_to_pdf":  # renderer utility, not an analysis package
                continue
            assert "project_name" in params, f"{name} missing project_name"
            assert "format" in params, f"{name} missing format"


class TestHtmlToPdf:
    HTML = ("<html><head><style>h1{color:#345;}</style></head><body>"
            "<h1>Wall stability report</h1><table><tr><td>Sliding FoS</td>"
            "<td>1.42</td></tr></table></body></html>")

    def test_html_content_to_pdf(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
        outfile = str(tmp_path / "report.pdf")
        result = _generate_html_to_pdf({"html": self.HTML,
                                        "output_path": outfile})
        assert result["status"] == "success"
        assert result["file_exists"] is True
        assert result["file_size_bytes"] > 500
        assert result["renderer"] == "story"
        with open(outfile, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_html_path_to_pdf(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
        src = tmp_path / "report.html"
        src.write_text(self.HTML, encoding="utf-8")
        result = _generate_html_to_pdf({"html_path": str(src),
                                        "output_path": str(tmp_path / "r")})
        assert result["status"] == "success"
        assert result["output_path"].endswith(".pdf")   # extension appended
        assert result["file_exists"] is True

    def test_missing_inputs_is_clear_error(self):
        from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
        result = _generate_html_to_pdf({})
        assert result["status"] == "error"
        assert "html" in result["error"]

    def test_sandboxed_path_names_the_problem(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
        result = _generate_html_to_pdf(
            {"html_path": str(tmp_path / "not_there.html")})
        assert result["status"] == "error"
        assert "sandboxed" in result["error"]

    def test_unknown_param_rejected(self):
        from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
        with pytest.raises(ValueError, match="html_to_pdf"):
            _generate_html_to_pdf({"html": "<p>x</p>", "renderer": "latex"})


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

    def test_pdf_response_carries_renderer(self, tmp_path):
        """A PDF package response reports which engine produced it (F7 QC:
        renderer propagation) — here the pure-Python Story fallback."""
        pytest.importorskip("fitz")
        from funhouse_agent.adapters.calc_package import _generate_bearing_capacity_package
        out = str(tmp_path / "bc.pdf")
        result = _generate_bearing_capacity_package({
            "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
            "depth": 1.5, "shape": "square",
            "format": "pdf", "output_path": out,
        })
        assert result["status"] == "success"
        assert result["renderer"] in ("pdflatex", "pymupdf_story")


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


class TestSlopeReportPackage:
    """The one-call slope REPORT for a critical-surface SEARCH run (F7)."""

    _SIMPLE = {
        "surface_points": [[0, 10], [10, 10], [20, 5], [30, 5]],
        "soil_layers": [{
            "name": "Clay", "top_elevation": 10, "bottom_elevation": 0,
            "gamma": 18.0, "phi": 25, "c_prime": 10,
        }],
        "x_range": [10, 22], "y_range": [14, 24], "nx": 5, "ny": 5,
        "n_slices": 16,
    }

    def test_basic_circular_search(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_slope_report_package
        out = str(tmp_path / "report.html")
        res = _generate_slope_report_package(
            {**self._SIMPLE, "method": "spencer", "output_path": out})
        assert res["status"] == "success"
        assert res["FOS"] > 0
        assert res["n_surfaces_evaluated"] > 0
        assert res["method"] == "Spencer"
        html = open(out, encoding="utf-8").read()
        # search story + method comparison + rigorous interslice/thrust figures
        for section in ("Critical Surface Search", "Method Comparison",
                        "Slice Force", "Line of Thrust"):
            assert section in html, section

    def test_probabilistic_annex(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_slope_report_package
        out = str(tmp_path / "prob.html")
        res = _generate_slope_report_package({
            **self._SIMPLE, "method": "spencer", "output_path": out,
            "variables": {
                "phi": {"mean": 25, "cov": 0.10, "source": "Duncan (2000)"},
                "c_prime": {"mean": 10, "cov": 0.30},
            },
        })
        assert res["status"] == "success"
        html = open(out, encoding="utf-8").read()
        assert "Probabilistic Analysis" in html
        assert "Reliability Index" in html      # FOSM beta
        assert "Random Variables" in html

    def test_no_output_path_autogenerates(self):
        from funhouse_agent.adapters.calc_package import _generate_slope_report_package
        res = _generate_slope_report_package(
            {**self._SIMPLE, "method": "bishop"})
        assert res["status"] == "success"
        assert res["output_path"].endswith(".html")
        assert os.path.isfile(res["output_path"])
        os.remove(res["output_path"])

    def test_pdf_output_fitz_verified(self, tmp_path):
        pytest.importorskip("fitz")
        import fitz
        from funhouse_agent.adapters.calc_package import _generate_slope_report_package
        out = str(tmp_path / "report.pdf")
        res = _generate_slope_report_package({
            **self._SIMPLE, "method": "spencer", "format": "pdf",
            "output_path": out})
        assert res["status"] == "success"
        assert res["file_exists"] is True
        doc = fitz.open(out)
        try:
            assert doc.page_count > 0
        finally:
            doc.close()

    def test_v028_weak_seam_report_matches_referee(self, tmp_path):
        """Validate the report against the V-028 (Slide2 #9 / ACADS 4) weak-seam
        search: the critical Spencer FOS is ~0.792 (referee 0.78), and the
        report renders the search story with rejection diagnostics."""
        from funhouse_agent.adapters.calc_package import _generate_slope_report_package
        out = str(tmp_path / "v028.html")
        res = _generate_slope_report_package({
            "surface_points": [[20, 28], [43, 28], [68, 40], [84, 40]],
            "soil_layers": [
                {"name": "upper", "top_elevation": 40, "bottom_elevation": 15,
                 "gamma": 18.84, "phi": 20, "c_prime": 28.5,
                 "bottom_boundary_points": [[20, 19], [84, 37]]},
                {"name": "weak", "top_elevation": 40, "bottom_elevation": 15,
                 "gamma": 18.84, "phi": 10, "c_prime": 0.0,
                 "bottom_boundary_points": [[20, 18], [84, 36]]},
                {"name": "lower", "top_elevation": 40, "bottom_elevation": 15,
                 "gamma": 18.84, "phi": 20, "c_prime": 28.5},
            ],
            "gwt_points": [[20, 27.75], [43, 27.75], [49, 29.86], [60, 34.06],
                           [66, 35.80], [74, 37.68], [80, 38.4], [84, 38.4]],
            "surface_type": "weak_layer", "method": "spencer",
            "n_trials": 700, "n_points": 6, "n_slices": 30, "seed": 3,
            "x_entry_range": [20, 50], "x_exit_range": [58, 84],
            "output_path": out,
        })
        assert res["status"] == "success"
        # module Spencer 0.792; referee 0.78 (+1.5%)
        assert res["FOS"] == pytest.approx(0.792, abs=0.02)
        assert res["FOS"] == pytest.approx(0.78, rel=0.05)
        assert res["is_stable"] is False           # 0.79 < 1.5
        html = open(out, encoding="utf-8").read()
        assert "Critical Surface Search" in html
        assert "rejected" in html.lower()          # rejection diagnostics
        assert "Method Comparison" in html

    def test_call_agent_routing(self, tmp_path):
        from funhouse_agent.dispatch import call_agent
        out = str(tmp_path / "routed.html")
        res = call_agent("calc_package", "slope_report_package",
                         {**self._SIMPLE, "method": "bishop",
                          "output_path": out})
        assert res["status"] == "success"
        assert os.path.exists(out)


class TestCalcPackageDisplayFixes:
    """Regression pins: calc-package displays that had drifted from the code
    (provenance-audit findings)."""

    def test_settlement_renders_actual_influence_factor(self):
        """The elastic-settlement step must show the Iw the analysis actually
        used (Schleicher ~1.122 for a square), not a hardcoded 1.0."""
        from settlement.analysis import SettlementAnalysis
        from settlement.calc_steps import get_calc_steps, _immediate_Iw
        analysis = SettlementAnalysis(
            q_applied=150, q_overburden=20, B=2.0, L=2.0,
            immediate_method="elastic", Es_immediate=12000, nu=0.3)
        result = analysis.compute()
        Iw = _immediate_Iw(analysis)
        assert Iw == pytest.approx(1.122, abs=0.01)      # square Schleicher, not 1.0
        text = _calc_step_text(get_calc_steps(result, analysis))
        assert f"{Iw:.3f}" in text                       # actual Iw rendered
        assert "× 1.0" not in text                       # old hardcoded value gone
        assert "I_w = 1.0 (flexible footing on surface)" not in text

    def test_drilled_shaft_nc_uses_current_formula(self, tmp_path):
        """Cohesive-tip N_c must display min(6(1+0.2·L/D), 9), matching
        end_bearing.py — not the superseded min(6 + L/D, 9)."""
        from funhouse_agent.adapters.calc_package import _generate_drilled_shaft_package
        out = str(tmp_path / "d.html")
        res = _generate_drilled_shaft_package({
            "diameter": 1.0, "length": 15.0,
            "soil_layers": [{"soil_type": "cohesive", "thickness": 20.0,
                             "unit_weight": 18.0, "cu": 100.0}],
            "output_path": out,
        })
        assert res["status"] == "success"
        html = open(out, encoding="utf-8").read()
        assert "1 + 0.2" in html                         # current formula shown
        assert "min(6.0 + L/D, 9.0)" not in html         # superseded formula gone


def _calc_step_text(sections) -> str:
    """Concatenate the equation/substitution/notes text of every CalcStep in a
    list of CalcSection (for asserting on rendered calc-package content)."""
    out = []
    for sec in sections:
        for item in getattr(sec, "items", []):
            for attr in ("equation", "substitution", "notes", "title"):
                v = getattr(item, attr, None)
                if isinstance(v, str):
                    out.append(v)
    return "\n".join(out)


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

    # The three non-wick variants drifted against the rebuilt GEC-13 module
    # signatures and failed for several releases (caught by the 2026-07
    # plot-pipeline sweep) — keep one regression per variant.

    def test_aggregate_piers(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_ground_improvement_package
        outfile = str(tmp_path / "gi_ap.html")
        result = _generate_ground_improvement_package({
            "method": "aggregate_piers",
            "column_diameter": 0.76, "spacing": 2.0,
            "E_column": 80000, "E_soil": 5000,
            "S_unreinforced": 0.05, "q_unreinforced": 150,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert os.path.exists(outfile)
        # Pre-GEC-13 aliases keep working.
        result2 = _generate_ground_improvement_package({
            "method": "aggregate_piers", "diameter": 0.76, "spacing": 2.0,
            "E_pier": 80000, "output_path": str(tmp_path / "gi_ap2.html"),
        })
        assert result2["status"] == "success"

    def test_surcharge(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_ground_improvement_package
        outfile = str(tmp_path / "gi_su.html")
        result = _generate_ground_improvement_package({
            "method": "surcharge",
            "S_ultimate": 0.30, "surcharge": 50, "cv": 3.0, "Hdr": 4.0,
            "target_consolidation": 90,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert os.path.exists(outfile)

    def test_vibro(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_ground_improvement_package
        outfile = str(tmp_path / "gi_vb.html")
        result = _generate_ground_improvement_package({
            "method": "vibro",
            "fines_content": 8, "N1_before": 12, "target_N_spt": 25,
            "output_path": outfile,
        })
        assert result["status"] == "success"
        assert os.path.exists(outfile)


class TestSeismicEarthPressurePackage:
    def test_mononobe_okabe(self, tmp_path):
        """Regression: the M-O result keys gained unit suffixes and the
        package read the old bare names (KeyError 'PAE_total')."""
        from funhouse_agent.adapters.calc_package import _generate_seismic_package
        outfile = str(tmp_path / "mo.html")
        result = _generate_seismic_package({
            "analysis_type": "seismic_earth_pressure",
            "phi": 32, "kh": 0.2, "gamma": 19, "H": 5.0,
            "output_path": outfile,
        })
        assert result["status"] == "success"
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
        assert total == 16  # 15 packages + html_to_pdf

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
        # output_path is returned absolute so callers know exactly where it landed
        assert os.path.isabs(result["output_path"])
        assert os.path.basename(result["output_path"]).startswith("bearing_capacity_calc_")
        assert result["file_exists"] is True
        assert result["file_size_bytes"] > 0
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


# ---------------------------------------------------------------------------
# Regression: v5.0 lateral-pile calc-package failure conversation (2026-06-12).
# The live agent burned ~850k tokens on (a) raw KeyError/TypeError from
# guessed layer keys/model names, (b) a silently-ignored stiffness param on
# the analysis tool making the package result look "wrong", and (c) being
# unable to verify the saved file.
# ---------------------------------------------------------------------------

class TestLateralPilePackageErgonomics:
    def _params(self, tmp_path, layers, **kw):
        p = {
            "pile_length": 15.0, "pile_diameter": 0.8, "pile_E": 7.5e6,
            "Vt": 20.0, "soil_layers": layers,
            "output_path": str(tmp_path / "lp.html"),
        }
        p.update(kw)
        return p

    def test_unknown_py_model_lists_options(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        layers = [{"top": 0, "bottom": 15, "py_model": "api_sand",
                   "phi": 32, "gamma": 19, "k": 8150}]
        with pytest.raises(ValueError, match="SandAPI"):
            _generate_lateral_pile_package(self._params(tmp_path, layers))

    def test_wrong_layer_param_names_actionable(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        # The live agent guessed phi_deg / gamma_kN_m3 — must get a message
        # naming the required keys, not a raw TypeError.
        layers = [{"top": 0, "bottom": 15, "py_model": "SandAPI",
                   "phi_deg": 32, "gamma_kN_m3": 19, "k": 8150}]
        with pytest.raises(ValueError, match=r"phi.*gamma|requires"):
            _generate_lateral_pile_package(self._params(tmp_path, layers))

    def test_model_key_alias_accepted(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        # The analysis tool calls this key 'model'; the package accepts both.
        layers = [{"top": 0, "bottom": 15, "model": "SandAPI",
                   "phi": 32, "gamma": 19, "k": 8150}]
        result = _generate_lateral_pile_package(self._params(tmp_path, layers))
        assert result["status"] == "success"

    def test_missing_pile_E_actionable(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        layers = [{"top": 0, "bottom": 15, "py_model": "SandAPI",
                   "phi": 32, "gamma": 19, "k": 8150}]
        p = self._params(tmp_path, layers)
        del p["pile_E"]
        with pytest.raises(ValueError, match="pile_E"):
            _generate_lateral_pile_package(p)

    def test_file_verification_fields(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        layers = [{"top": 0, "bottom": 15, "py_model": "SandAPI",
                   "phi": 32, "gamma": 19, "k": 8150}]
        result = _generate_lateral_pile_package(self._params(tmp_path, layers))
        assert result["file_exists"] is True
        assert result["file_size_bytes"] > 1000
        assert os.path.isabs(result["output_path"])
        assert os.path.getsize(result["output_path"]) == result["file_size_bytes"]

    def test_package_matches_analysis_tool(self, tmp_path):
        """Same inputs -> same deflection from both tools (the 0.50 vs 1.86 mm
        confusion came from DIFFERENT stiffness inputs, not different solvers)."""
        from funhouse_agent.adapters.calc_package import _generate_lateral_pile_package
        from funhouse_agent.adapters.lateral_pile import METHOD_REGISTRY as LP
        layers = [{"top": 0, "bottom": 15, "py_model": "SandAPI",
                   "phi": 32, "gamma": 19, "k": 8150}]
        pkg = _generate_lateral_pile_package(self._params(tmp_path, layers))
        direct = LP["lateral_pile_analysis"]({
            "pile_type": "pipe", "pile_diameter": 0.8, "pile_length": 15.0,
            "pile_E": 7.5e6, "Vt": 20.0,
            "layers": [{"top": 0, "bottom": 15, "model": "SandAPI",
                        "phi": 32, "gamma": 19, "k": 8150}],
        })
        y_direct_mm = direct["deflection_m"][0] * 1000
        assert abs(pkg["y_top_mm"] - y_direct_mm) < 0.05
