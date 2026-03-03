"""Tests for excavation strut support in fem2d."""

import pytest
import numpy as np

from fem2d import analyze_excavation


# Shared simple soil layers for all tests
SIMPLE_SOIL = [
    {"name": "Clay", "bottom_elevation": -20, "E": 30000, "nu": 0.3,
     "gamma": 18, "c": 25, "phi": 20, "psi": 0},
]


class TestExcavationStruts:
    """Tests for strut spring support in analyze_excavation."""

    def test_no_struts_backward_compat(self):
        """analyze_excavation without struts returns same as before."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
        )
        assert result.strut_forces is None
        assert result.max_displacement_m > 0

    def test_struts_none_explicit(self):
        """Passing struts=None is equivalent to no struts."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=None,
        )
        assert result.strut_forces is None

    def test_single_strut_produces_force(self):
        """A single strut produces a non-zero restraining force."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=[{"depth": 1.5, "stiffness": 50000}],
        )
        assert result.strut_forces is not None
        assert len(result.strut_forces) == 1
        # Strut should produce a non-zero force (wall deflects)
        assert abs(result.strut_forces[0]["force_kN_per_m"]) > 0

    def test_strut_force_extracted(self):
        """Strut force dict has expected keys and sensible values."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=[{"depth": 2.0, "stiffness": 50000}],
        )
        sf = result.strut_forces[0]
        assert "depth_m" in sf
        assert "stiffness_kN_per_m" in sf
        assert "force_kN_per_m" in sf
        assert "node_id" in sf
        assert sf["stiffness_kN_per_m"] == 50000
        # Depth should be approximately 2.0 (nearest node)
        assert abs(sf["depth_m"] - 2.0) < 2.0

    def test_two_struts(self):
        """Two struts at different depths both appear in results."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=[
                {"depth": 1.0, "stiffness": 50000},
                {"depth": 3.0, "stiffness": 30000},
            ],
        )
        assert result.strut_forces is not None
        assert len(result.strut_forces) == 2
        depths = [sf["depth_m"] for sf in result.strut_forces]
        # Both struts should be at distinct depths
        assert depths[0] != depths[1]

    def test_zero_stiffness_strut_ignored(self):
        """A strut with zero stiffness has no effect."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=[{"depth": 1.5, "stiffness": 0}],
        )
        # Zero-stiffness strut is skipped
        assert result.strut_forces is None

    def test_strut_in_summary(self):
        """Strut forces appear in summary text."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=[{"depth": 2.0, "stiffness": 50000}],
        )
        summary = result.summary()
        assert "Strut" in summary

    def test_strut_in_to_dict(self):
        """Strut forces appear in to_dict output."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            struts=[{"depth": 2.0, "stiffness": 50000}],
        )
        d = result.to_dict()
        assert "strut_forces" in d
        assert len(d["strut_forces"]) == 1

    def test_max_iter_tol_passthrough(self):
        """Custom max_iter and tol are accepted without error."""
        result = analyze_excavation(
            width=10, depth=5, wall_depth=10,
            soil_layers=SIMPLE_SOIL,
            wall_EI=50000, wall_EA=5e6,
            nx=10, ny=6, n_steps=3,
            max_iter=50, tol=1e-4,
        )
        assert result.max_displacement_m > 0
