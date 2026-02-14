"""
Validation tests for the pile_group module.

Tests cover pile layout, group efficiency, simplified and 6-DOF
analysis, and hand-verifiable symmetric loading cases.

References:
    [1] CPGA User's Guide (USACE ITL-89-4)
    [2] FHWA GEC-12, Chapter 9
    [3] USACE EM 1110-2-2906
"""

import math
import pytest
import numpy as np

from pile_group.pile_layout import GroupPile, create_rectangular_layout
from pile_group.group_efficiency import (
    converse_labarre, block_failure_capacity, p_multiplier,
)
from pile_group.rigid_cap import (
    GroupLoad, PileGroupResult,
    analyze_vertical_group_simple, analyze_group_6dof,
)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Pile Layout
# ═══════════════════════════════════════════════════════════════════════

class TestPileLayout:
    """Test pile layout creation."""

    def test_rectangular_layout_count(self):
        """3x4 layout should have 12 piles."""
        piles = create_rectangular_layout(3, 4, 1.5, 1.5)
        assert len(piles) == 12

    def test_layout_centered(self):
        """Layout should be centered on (0,0)."""
        piles = create_rectangular_layout(3, 3, 2.0, 2.0)
        xs = [p.x for p in piles]
        ys = [p.y for p in piles]
        assert abs(sum(xs)) < 1e-10
        assert abs(sum(ys)) < 1e-10

    def test_layout_spacing(self):
        """Verify spacing between adjacent piles."""
        piles = create_rectangular_layout(2, 2, 3.0, 3.0)
        # 2x2 at 3m spacing: corners at (-1.5,-1.5), (1.5,-1.5), etc.
        assert piles[0].x == pytest.approx(-1.5)
        assert piles[1].x == pytest.approx(1.5)

    def test_single_pile(self):
        """1x1 layout: single pile at origin."""
        piles = create_rectangular_layout(1, 1, 1.0, 1.0)
        assert len(piles) == 1
        assert piles[0].x == 0.0
        assert piles[0].y == 0.0

    def test_direction_cosines_vertical(self):
        """Vertical pile: lz ≈ 1."""
        p = GroupPile(x=0, y=0)
        lx, ly, lz = p.direction_cosines()
        assert lz == pytest.approx(1.0, abs=0.01)
        assert abs(lx) < 0.01
        assert abs(ly) < 0.01

    def test_direction_cosines_battered(self):
        """Battered pile: lx > 0, lz < 1."""
        p = GroupPile(x=0, y=0, batter_x=15)  # 15 deg batter
        lx, ly, lz = p.direction_cosines()
        assert lx > 0
        assert lz < 1.0
        assert lx**2 + ly**2 + lz**2 == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Group Efficiency
# ═══════════════════════════════════════════════════════════════════════

class TestGroupEfficiency:
    """Test group efficiency calculations."""

    def test_converse_labarre_3x3(self):
        """3x3 group, D=0.3m, s=1.5m: Eg ≈ 0.83."""
        Eg = converse_labarre(3, 3, 0.3, 1.5)
        assert 0.7 < Eg < 1.0

    def test_converse_labarre_wide_spacing(self):
        """Wide spacing: Eg approaches 1.0."""
        Eg = converse_labarre(3, 3, 0.3, 5.0)
        assert Eg > 0.94

    def test_converse_labarre_close_spacing(self):
        """Close spacing: Eg is lower."""
        Eg_close = converse_labarre(3, 3, 0.3, 0.6)
        Eg_wide = converse_labarre(3, 3, 0.3, 3.0)
        assert Eg_close < Eg_wide

    def test_converse_labarre_single_pile(self):
        """1x1 group: Eg = 1.0 (no group effect)."""
        Eg = converse_labarre(1, 1, 0.3, 1.0)
        assert Eg == pytest.approx(1.0, abs=0.01)

    def test_block_failure_positive(self):
        """Block failure capacity should be positive."""
        Qb = block_failure_capacity(3, 3, 1.5, 1.5, 15.0, 50.0, 0.3)
        assert Qb > 0

    def test_p_multiplier_leading_row(self):
        """Leading row at 3D spacing: pm ≈ 0.8."""
        pm = p_multiplier(1, 3.0)
        assert pm == pytest.approx(0.8, abs=0.1)

    def test_p_multiplier_trailing_row(self):
        """Trailing rows get lower p-multiplier."""
        pm1 = p_multiplier(1, 3.0)
        pm3 = p_multiplier(3, 3.0)
        assert pm3 < pm1

    def test_p_multiplier_wide_spacing(self):
        """At 5D+ spacing, all rows get pm=1.0."""
        for row in [1, 2, 3]:
            assert p_multiplier(row, 5.0) == 1.0


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Simplified Vertical Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestSimplifiedAnalysis:
    """Test simplified elastic method for vertical piles."""

    def test_uniform_vertical_load(self):
        """Uniform vertical load: each pile gets V/n."""
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000)
        load = GroupLoad(Vz=400)
        result = analyze_vertical_group_simple(piles, load)

        for pf in result.pile_forces:
            assert pf['axial_kN'] == pytest.approx(100.0, abs=0.5)

    def test_moment_loading(self):
        """Moment My should cause higher loads on +x piles, lower on -x."""
        piles = create_rectangular_layout(1, 2, 2.0, 2.0,
                                          axial_stiffness=50000)
        # 2 piles at x=-1 and x=+1
        load = GroupLoad(Vz=200, My=100)
        result = analyze_vertical_group_simple(piles, load)

        forces = {pf['x_m']: pf['axial_kN'] for pf in result.pile_forces}
        assert forces[-1.0] < forces[1.0]  # +x gets more load
        # Sum should equal Vz
        total = sum(pf['axial_kN'] for pf in result.pile_forces)
        assert total == pytest.approx(200.0, abs=0.5)

    def test_pure_moment_symmetric(self):
        """Pure moment on symmetric group: forces sum to zero."""
        piles = create_rectangular_layout(1, 2, 4.0, 4.0,
                                          axial_stiffness=50000)
        load = GroupLoad(Vz=0, My=200)
        result = analyze_vertical_group_simple(piles, load)

        total = sum(pf['axial_kN'] for pf in result.pile_forces)
        assert abs(total) < 0.5  # should sum to ~0

    def test_utilization_ratio(self):
        """Utilization = demand / capacity."""
        piles = create_rectangular_layout(1, 1, 1.0, 1.0,
                                          axial_stiffness=50000)
        piles[0].axial_capacity_compression = 500.0
        load = GroupLoad(Vz=250)
        result = analyze_vertical_group_simple(piles, load)
        assert result.max_utilization == pytest.approx(0.5, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: 6-DOF Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestSixDofAnalysis:
    """Test general 6-DOF rigid cap analysis."""

    def test_vertical_load_6dof(self):
        """6-DOF with only vertical load should match simplified."""
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000,
                                          lateral_stiffness=5000)
        load = GroupLoad(Vz=400)

        r_simple = analyze_vertical_group_simple(piles, load)
        r_6dof = analyze_group_6dof(piles, load)

        # Each pile should get ~100 kN in both methods
        for pf in r_6dof.pile_forces:
            assert pf['axial_kN'] == pytest.approx(100.0, abs=5.0)

    def test_6dof_needs_stiffness(self):
        """6-DOF should raise error if stiffness is missing."""
        piles = [GroupPile(0, 0)]  # no stiffness
        load = GroupLoad(Vz=100)
        with pytest.raises(ValueError, match="missing axial_stiffness"):
            analyze_group_6dof(piles, load)

    def test_6dof_cap_displacement(self):
        """6-DOF should compute cap displacements."""
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000,
                                          lateral_stiffness=5000)
        load = GroupLoad(Vz=400)
        result = analyze_group_6dof(piles, load)
        assert result.cap_displacements['dz'] > 0

    def test_summary_and_dict(self):
        """summary() and to_dict() should work."""
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000,
                                          lateral_stiffness=5000)
        load = GroupLoad(Vz=400)
        result = analyze_group_6dof(piles, load)
        assert "PILE GROUP" in result.summary()
        d = result.to_dict()
        assert d["n_piles"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
