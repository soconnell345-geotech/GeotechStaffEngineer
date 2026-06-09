"""
Tests for the GEF / BRO-XML format adapter — CPT and borehole file parser.

(Folded in from the former pygef_agent module; see CONSOLIDATION_CHANGES.md #3.)

Tier 1: No pygef required (result dataclasses, validation, utilities)
Tier 2: Requires pygef (integration tests with sample GEF files)
"""

import json
import os
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from subsurface_characterization.formats.gef import has_pygef
from subsurface_characterization.formats.gef_results import (
    CPTParseResult, BoreParseResult,
)

requires_pygef = pytest.mark.skipif(
    not has_pygef(), reason="pygef not installed"
)

# Path to test data files
TEST_DIR = os.path.dirname(__file__)
SAMPLE_CPT = os.path.join(TEST_DIR, "sample_cpt.gef")
SAMPLE_BORE = os.path.join(TEST_DIR, "sample_bore.gef")


# =====================================================================
# Tier 1: Result dataclass defaults
# =====================================================================

class TestCPTParseResultDefaults:
    """Test CPTParseResult with default values."""

    def test_default_construction(self):
        r = CPTParseResult()
        assert r.n_points == 0
        assert r.alias == ""
        assert r.gwl_m is None

    def test_construction_with_values(self):
        r = CPTParseResult(
            n_points=100, alias="CPT-01", final_depth_m=20.0,
            gwl_m=1.5, x=155000.0, y=463000.0,
            depth_m=np.arange(0.5, 10.5, 0.5),
            q_c_kPa=np.full(20, 5000.0),
        )
        assert r.n_points == 100
        assert r.alias == "CPT-01"
        assert r.gwl_m == 1.5

    def test_summary_contains_alias(self):
        r = CPTParseResult(
            n_points=100, alias="CPT-01", final_depth_m=20.0,
        )
        s = r.summary()
        assert "CPT-01" in s
        assert "100" in s

    def test_to_dict_keys(self):
        r = CPTParseResult(n_points=10, alias="CPT-01")
        d = r.to_dict()
        assert "n_points" in d
        assert "alias" in d
        assert "available_columns" in d

    def test_to_dict_includes_arrays(self):
        r = CPTParseResult(
            n_points=3, alias="CPT-01",
            depth_m=np.array([1.0, 2.0, 3.0]),
            q_c_kPa=np.array([5000.0, 6000.0, 7000.0]),
        )
        d = r.to_dict()
        assert "depth_m" in d
        assert "q_c_kPa" in d
        assert len(d["depth_m"]) == 3

    def test_to_dict_json_serializable(self):
        r = CPTParseResult(
            n_points=3, alias="CPT-01",
            depth_m=np.array([1.0, 2.0, 3.0]),
            q_c_kPa=np.array([5000.0, 6000.0, 7000.0]),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_to_liquepy_inputs(self):
        r = CPTParseResult(
            n_points=3, alias="CPT-01", gwl_m=1.5,
            depth_m=np.array([1.0, 2.0, 3.0]),
            q_c_kPa=np.array([5000.0, 6000.0, 7000.0]),
            f_s_kPa=np.array([50.0, 60.0, 70.0]),
        )
        d = r.to_liquepy_inputs()
        assert "depth" in d
        assert "q_c" in d
        assert "f_s" in d
        assert "u_2" in d
        assert d["gwl"] == 1.5

    def test_to_liquepy_inputs_no_fs(self):
        r = CPTParseResult(
            n_points=3,
            depth_m=np.array([1.0, 2.0, 3.0]),
            q_c_kPa=np.array([5000.0, 6000.0, 7000.0]),
        )
        d = r.to_liquepy_inputs()
        assert len(d["f_s"]) == 3  # Should default to zeros
        assert np.all(d["f_s"] == 0)


class TestBoreParseResultDefaults:
    """Test BoreParseResult with default values."""

    def test_default_construction(self):
        r = BoreParseResult()
        assert r.n_layers == 0
        assert r.alias == ""

    def test_construction_with_values(self):
        r = BoreParseResult(
            n_layers=4, alias="BH-01", final_depth_m=8.0,
            gwl_m=1.5,
            top_m=np.array([0, 1.5, 3, 5]),
            bottom_m=np.array([1.5, 3, 5, 8]),
            soil_name=["Sand", "Clay", "Clay", "Sand"],
        )
        assert r.n_layers == 4
        assert r.alias == "BH-01"

    def test_summary_contains_alias(self):
        r = BoreParseResult(n_layers=4, alias="BH-01", final_depth_m=8.0)
        s = r.summary()
        assert "BH-01" in s
        assert "4" in s

    def test_to_dict_has_layers(self):
        r = BoreParseResult(
            n_layers=2, alias="BH-01", final_depth_m=3.0,
            top_m=np.array([0, 1.5]),
            bottom_m=np.array([1.5, 3.0]),
            soil_name=["Sand", "Clay"],
        )
        d = r.to_dict()
        assert "layers" in d
        assert len(d["layers"]) == 2
        assert d["layers"][0]["soil_name"] == "Sand"

    def test_to_dict_json_serializable(self):
        r = BoreParseResult(
            n_layers=2, alias="BH-01", final_depth_m=3.0,
            top_m=np.array([0, 1.5]),
            bottom_m=np.array([1.5, 3.0]),
            soil_name=["Sand", "Clay"],
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Plot smoke tests
# =====================================================================

class TestPlotSmoke:
    """Smoke tests for plot methods (no pygef needed)."""

    def test_cpt_plot_qc(self):
        r = CPTParseResult(
            n_points=10, alias="CPT-01",
            depth_m=np.arange(0.5, 5.5, 0.5),
            q_c_kPa=np.random.uniform(3000, 15000, 10),
        )
        ax = r.plot_qc(show=False)
        assert ax is not None

    def test_cpt_plot_all(self):
        n = 10
        r = CPTParseResult(
            n_points=n, alias="CPT-01", final_depth_m=5.0,
            depth_m=np.arange(0.5, 5.5, 0.5),
            q_c_kPa=np.random.uniform(3000, 15000, n),
            f_s_kPa=np.random.uniform(30, 150, n),
            Rf_pct=np.random.uniform(0.5, 3.0, n),
        )
        axes = r.plot_all(show=False)
        assert len(axes) == 3

    def test_bore_plot_profile(self):
        r = BoreParseResult(
            n_layers=3, alias="BH-01", final_depth_m=6.0,
            top_m=np.array([0, 2, 4]),
            bottom_m=np.array([2, 4, 6]),
            soil_name=["Sand", "Clay", "Gravel"],
        )
        ax = r.plot_profile(show=False)
        assert ax is not None


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestInputValidation:
    """Test input validation (no pygef needed)."""

    def test_cpt_empty_path(self):
        from subsurface_characterization.formats.gef import _validate_cpt_parse_inputs
        with pytest.raises(ValueError, match="file_path"):
            _validate_cpt_parse_inputs("", "auto")

    def test_cpt_bad_engine(self):
        from subsurface_characterization.formats.gef import _validate_cpt_parse_inputs
        with pytest.raises(ValueError, match="engine"):
            _validate_cpt_parse_inputs("test.gef", "csv")

    def test_bore_empty_path(self):
        from subsurface_characterization.formats.gef import _validate_bore_parse_inputs
        with pytest.raises(ValueError, match="file_path"):
            _validate_bore_parse_inputs("", "auto")

    def test_bore_bad_engine(self):
        from subsurface_characterization.formats.gef import _validate_bore_parse_inputs
        with pytest.raises(ValueError, match="engine"):
            _validate_bore_parse_inputs("test.gef", "csv")


# =====================================================================
# Tier 1: Utility functions
# =====================================================================

class TestUtilities:
    """Test utility functions."""

    def test_has_pygef_returns_bool(self):
        assert isinstance(has_pygef(), bool)


# =====================================================================
# Tier 2: CPT parsing integration (requires pygef)
# =====================================================================

@requires_pygef
class TestCPTParsingIntegration:
    """Integration tests for CPT file parsing."""

    def test_parse_sample_cpt(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        assert r.n_points == 10
        assert r.alias == "CPT-01"

    def test_cpt_depth_range(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        assert r.depth_m[0] == 0.5
        assert r.final_depth_m == 5.0

    def test_cpt_qc_converted_to_kpa(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        # GEF has 5.000 MPa = 5000 kPa at first point
        assert r.q_c_kPa[0] == pytest.approx(5000.0, rel=0.01)

    def test_cpt_fs_present(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        assert len(r.f_s_kPa) == 10
        # GEF has 0.050 MPa = 50 kPa at first point
        assert r.f_s_kPa[0] == pytest.approx(50.0, rel=0.01)

    def test_cpt_gwl_extracted(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        assert r.gwl_m == pytest.approx(1.5, abs=0.01)

    def test_cpt_location(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        assert r.x == pytest.approx(155000.0)
        assert r.y == pytest.approx(463000.0)
        assert "EPSG" in r.srs_name

    def test_cpt_to_dict_json_serializable(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_cpt_to_liquepy_inputs(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT)
        d = r.to_liquepy_inputs()
        assert len(d["depth"]) == 10
        assert len(d["q_c"]) == 10
        assert d["gwl"] == pytest.approx(1.5)

    def test_cpt_explicit_engine(self):
        from subsurface_characterization.formats.gef import parse_cpt_file
        r = parse_cpt_file(SAMPLE_CPT, engine="gef")
        assert r.n_points == 10

    def test_cpt_bridges_to_investigation(self):
        """The GEF CPT result bridges into a SiteModel Investigation."""
        from subsurface_characterization.formats.gef import parse_cpt_file
        from subsurface_characterization import load_cpt_to_investigation
        r = parse_cpt_file(SAMPLE_CPT)
        inv = load_cpt_to_investigation(r)
        assert inv.investigation_type == "cpt"
        assert inv.investigation_id == "CPT-01"
        assert len(inv.measurements) > 0


# =====================================================================
# Tier 2: Bore parsing integration (requires pygef)
# =====================================================================

@requires_pygef
class TestBoreParsingIntegration:
    """Integration tests for borehole file parsing."""

    def test_parse_sample_bore(self):
        from subsurface_characterization.formats.gef import parse_bore_file
        r = parse_bore_file(SAMPLE_BORE)
        assert r.n_layers == 4
        assert r.alias == "BH-01"

    def test_bore_depth(self):
        from subsurface_characterization.formats.gef import parse_bore_file
        r = parse_bore_file(SAMPLE_BORE)
        assert r.final_depth_m == 8.0
        assert r.top_m[0] == 0.0
        assert r.bottom_m[-1] == 8.0

    def test_bore_soil_names(self):
        from subsurface_characterization.formats.gef import parse_bore_file
        r = parse_bore_file(SAMPLE_BORE)
        assert len(r.soil_name) == 4
        # pygef translates soil codes to Dutch names
        assert all(isinstance(s, str) for s in r.soil_name)

    def test_bore_to_dict_json_serializable(self):
        from subsurface_characterization.formats.gef import parse_bore_file
        r = parse_bore_file(SAMPLE_BORE)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_bore_to_dict_has_layers(self):
        from subsurface_characterization.formats.gef import parse_bore_file
        r = parse_bore_file(SAMPLE_BORE)
        d = r.to_dict()
        assert len(d["layers"]) == 4
