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
    group_settlement_equivalent_raft,
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

    def test_converse_labarre_capped_at_1(self):
        """Efficiency must never exceed 1.0 (AASHTO LRFD 10.7.2.3)."""
        # Very wide spacing -> formula approaches 1.0 but should not exceed it
        Eg = converse_labarre(2, 2, 0.01, 100.0)
        assert Eg <= 1.0
        assert Eg == pytest.approx(1.0, abs=0.001)

    def test_converse_labarre_symmetric(self):
        """Formula is symmetric: swapping rows/cols gives same result."""
        Eg_3x5 = converse_labarre(3, 5, 0.356, 1.07)
        Eg_5x3 = converse_labarre(5, 3, 0.356, 1.07)
        assert Eg_3x5 == pytest.approx(Eg_5x3, abs=1e-10)

    def test_converse_labarre_hand_calc_2x2(self):
        """2x2 group with arctan = 18.3 deg -> Eg ≈ 0.80 (textbook example)."""
        # arctan(d/s) = 18.3 deg => d/s = tan(18.3 deg) => s/d ≈ 3.024
        # Use d=0.3, s = 0.3 * 3.024 = 0.9072
        d = 0.3
        theta_target = 18.3
        s = d / math.tan(math.radians(theta_target))
        Eg = converse_labarre(2, 2, d, s)
        # Eg = 1 - 18.3/(90*2*2) * [2*1 + 2*1] = 1 - 18.3/360 * 4 = 0.797
        assert Eg == pytest.approx(0.797, abs=0.01)

    def test_converse_labarre_3x3_exact(self):
        """3x3, D=0.3m, s=1.5m: exact Eg = 1 - arctan(0.2)*12/(90*9)."""
        theta = math.degrees(math.atan(0.3 / 1.5))  # 11.31 deg
        expected = 1.0 - theta / (90 * 3 * 3) * (3 * 2 + 3 * 2)
        Eg = converse_labarre(3, 3, 0.3, 1.5)
        assert Eg == pytest.approx(expected, abs=1e-10)

    def test_converse_labarre_raises_negative_spacing(self):
        """Negative spacing should raise ValueError."""
        with pytest.raises(ValueError):
            converse_labarre(3, 3, 0.3, -1.0)

    def test_converse_labarre_raises_zero_diameter(self):
        """Zero diameter should raise ValueError."""
        with pytest.raises(ValueError):
            converse_labarre(3, 3, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# TEST 2B: Group Settlement — Equivalent Raft Method
# ═══════════════════════════════════════════════════════════════════════

class TestGroupSettlement:
    """Tests for group_settlement_equivalent_raft (FHWA GEC-12 Sec 9.8)."""

    def test_raft_dimensions_3x4(self):
        """Bg = (n_cols-1)*s + d, Lg = (n_rows-1)*s + d."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=4, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        # Bg = (4-1)*1.5 + 0.3 = 4.8 m
        # Lg = (3-1)*1.5 + 0.3 = 3.3 m
        assert result["Bg_m"] == pytest.approx(4.8, abs=0.01)
        assert result["Lg_m"] == pytest.approx(3.3, abs=0.01)

    def test_raft_depth_two_thirds_L(self):
        """Equivalent raft at depth 2/3 * L."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        assert result["raft_depth_m"] == pytest.approx(10.0, abs=0.01)

    def test_raft_depth_different_length(self):
        """Raft depth = 2/3 * 24 = 16 m."""
        result = group_settlement_equivalent_raft(
            n_rows=2, n_cols=2, spacing=1.5, pile_diameter=0.3,
            pile_length=24.0, load_kN=1000, soil_modulus_kPa=20000,
        )
        assert result["raft_depth_m"] == pytest.approx(16.0, abs=0.01)

    def test_influence_depth_5Bg(self):
        """Influence depth = 5 * Bg."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=4, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        Bg = 4.8
        assert result["influence_depth_m"] == pytest.approx(5 * Bg, abs=0.01)

    def test_max_stress_at_raft(self):
        """Stress at z=0: Q / (Bg * Lg)."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=4, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        expected = 2000 / (4.8 * 3.3)
        assert result["max_stress_kPa"] == pytest.approx(expected, abs=0.1)

    def test_settlement_positive(self):
        """Settlement should be positive for downward load."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        assert result["settlement_m"] > 0
        assert result["settlement_mm"] > 0

    def test_settlement_increases_with_load(self):
        """Doubling load should double settlement (linear elastic)."""
        r1 = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=1000, soil_modulus_kPa=30000,
        )
        r2 = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        assert r2["settlement_mm"] == pytest.approx(
            2 * r1["settlement_mm"], rel=0.01,
        )

    def test_settlement_decreases_with_stiffer_soil(self):
        """Higher modulus -> less settlement."""
        r_soft = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=15000,
        )
        r_stiff = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=60000,
        )
        assert r_stiff["settlement_mm"] < r_soft["settlement_mm"]

    def test_settlement_mm_consistent(self):
        """settlement_mm should equal settlement_m * 1000."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        assert result["settlement_mm"] == pytest.approx(
            result["settlement_m"] * 1000, abs=0.01,
        )

    def test_single_pile_raft_equals_diameter(self):
        """1x1 group: Bg = Lg = pile diameter."""
        result = group_settlement_equivalent_raft(
            n_rows=1, n_cols=1, spacing=1.0, pile_diameter=0.5,
            pile_length=10.0, load_kN=500, soil_modulus_kPa=25000,
        )
        assert result["Bg_m"] == pytest.approx(0.5, abs=0.01)
        assert result["Lg_m"] == pytest.approx(0.5, abs=0.01)

    def test_hand_calc_settlement(self):
        """Hand-calculated settlement for 2x2 group.

        Bg = Lg = (2-1)*1.5 + 0.3 = 1.8 m
        Q = 1000 kN, Es = 20000 kPa
        Influence depth = 5 * 1.8 = 9.0 m
        Using 1 sublayer (midpoint at z=4.5):
            delta_sigma = 1000 / ((1.8+4.5)*(1.8+4.5)) = 1000 / 39.69 = 25.19 kPa
            S = 25.19 * 9.0 / 20000 = 0.01134 m = 11.34 mm
        """
        result = group_settlement_equivalent_raft(
            n_rows=2, n_cols=2, spacing=1.5, pile_diameter=0.3,
            pile_length=12.0, load_kN=1000, soil_modulus_kPa=20000,
            num_sublayers=1,
        )
        # With 1 sublayer, midpoint at z = 4.5 m
        Bg = 1.8
        Lg = 1.8
        z_mid = 4.5
        delta_sigma = 1000 / ((Bg + z_mid) * (Lg + z_mid))
        expected = delta_sigma * 9.0 / 20000 * 1000  # mm
        assert result["settlement_mm"] == pytest.approx(expected, rel=0.01)

    def test_raises_negative_load(self):
        """Negative load should raise ValueError."""
        with pytest.raises(ValueError, match="Load must be positive"):
            group_settlement_equivalent_raft(
                n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
                pile_length=15.0, load_kN=-100, soil_modulus_kPa=30000,
            )

    def test_raises_zero_modulus(self):
        """Zero modulus should raise ValueError."""
        with pytest.raises(ValueError, match="Soil modulus must be positive"):
            group_settlement_equivalent_raft(
                n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
                pile_length=15.0, load_kN=1000, soil_modulus_kPa=0,
            )

    def test_raises_zero_pile_length(self):
        """Zero pile length should raise ValueError."""
        with pytest.raises(ValueError, match="Pile length must be positive"):
            group_settlement_equivalent_raft(
                n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
                pile_length=0, load_kN=1000, soil_modulus_kPa=30000,
            )

    def test_more_sublayers_converges(self):
        """More sublayers should converge — 100 vs 10 differ by < 5%."""
        r10 = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
            num_sublayers=10,
        )
        r100 = group_settlement_equivalent_raft(
            n_rows=3, n_cols=3, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
            num_sublayers=100,
        )
        assert r10["settlement_mm"] == pytest.approx(
            r100["settlement_mm"], rel=0.05,
        )

    def test_2v1h_stress_distribution(self):
        """Verify 2V:1H stress distribution: at depth z, area = (Bg+z)*(Lg+z)."""
        # This is implicit in the formula, but verify by checking that
        # settlement with 1 sublayer at midpoint uses correct stress.
        n_rows, n_cols = 3, 3
        s, d, L = 2.0, 0.4, 20.0
        Q, Es = 3000, 40000
        Bg = (n_cols - 1) * s + d  # 4.4
        Lg = (n_rows - 1) * s + d  # 4.4
        depth = 5 * Bg  # 22.0

        result = group_settlement_equivalent_raft(
            n_rows=n_rows, n_cols=n_cols, spacing=s, pile_diameter=d,
            pile_length=L, load_kN=Q, soil_modulus_kPa=Es,
            num_sublayers=1,
        )
        z_mid = depth / 2  # 11.0
        expected_stress = Q / ((Bg + z_mid) * (Lg + z_mid))
        expected_settle = expected_stress * depth / Es * 1000  # mm
        assert result["settlement_mm"] == pytest.approx(expected_settle, rel=0.01)


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
