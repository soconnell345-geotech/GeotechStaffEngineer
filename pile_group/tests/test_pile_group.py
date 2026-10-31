"""
Validation tests for the pile_group module.

Tests cover pile layout, group efficiency, simplified and 6-DOF
analysis, and hand-verifiable symmetric loading cases.

References:
    [1] CPGA User's Guide (USACE ITL-89-4)
    [2] FHWA GEC-12 (FHWA-NHI-16-009), Vol. I ch. 7
    [3] USACE EM 1110-2-2906
    [4] UFC 3-220-20 (2025), ch. 6
"""

import math
import pytest
import numpy as np

from pile_group.pile_layout import GroupPile, create_rectangular_layout
from pile_group.group_efficiency import (
    group_efficiency_factor, block_failure_resistance,
    block_failure_capacity, group_plan_dimensions, p_multiplier,
    group_settlement_equivalent_raft, meyerhof_group_settlement,
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
    """eta_g: GEC-12 Secs. 7.2.2.1-7.2.2.2 (driven piles, pp. 282, 284) and
    UFC 3-220-20 Table 6-26 (p. 432)."""

    def test_cohesionless_driven_is_one(self):
        """GEC-12 p. 282 item 1: the sum of the individual resistances."""
        r = group_efficiency_factor("cohesionless", 3.0)
        assert r["eta_g"] == 1.0
        assert r["alternative"] is None
        assert not r["cap_in_contact_taken"] and not r["su_taken"]

    def test_soft_clay_cap_free_runs_0_65_to_1(self):
        """GEC-12 p. 284 step 1: 0.65 at 2.5D, 1.0 at 6D, linear between,
        1.0 beyond."""
        def eta(sb):
            return group_efficiency_factor(
                "cohesive", sb, cap_in_contact=False, su_kPa=50.0)["eta_g"]
        assert eta(2.5) == pytest.approx(0.65)
        assert eta(3.0) == pytest.approx(0.70)
        assert eta(4.25) == pytest.approx(0.825)
        assert eta(6.0) == pytest.approx(1.0)
        assert eta(8.0) == 1.0

    def test_soft_clay_cap_in_contact_is_one(self):
        """GEC-12 p. 284 step 2."""
        r = group_efficiency_factor("cohesive", 2.5, cap_in_contact=True,
                                    su_kPa=50.0)
        assert r["eta_g"] == 1.0

    def test_stiff_clay_is_one(self):
        """GEC-12 p. 284 step 3: su above 2 ksf (95.76 kPa); UFC Table
        6-26's row is su >= 2 ksf."""
        for su in (95.76, 120.0):
            r = group_efficiency_factor("cohesive", 2.5,
                                        cap_in_contact=False, su_kPa=su)
            assert r["eta_g"] == 1.0
            assert r["su_below_2ksf"] is False
        assert group_efficiency_factor(
            "cohesive", 2.5, cap_in_contact=False,
            su_kPa=95.0)["eta_g"] == pytest.approx(0.65)

    def test_contact_left_out_takes_the_lower(self):
        r = group_efficiency_factor("cohesive", 3.0, su_kPa=50.0)
        assert r["cap_in_contact_taken"] and r["cap_in_contact"] is False
        assert r["eta_g"] == pytest.approx(0.70)
        assert r["alternative"] == pytest.approx(1.0)

    def test_su_left_out_takes_the_lower(self):
        r = group_efficiency_factor("cohesive", 2.5, cap_in_contact=False)
        assert r["su_taken"] and r["su_below_2ksf"] is True
        assert r["eta_g"] == pytest.approx(0.65)
        assert r["alternative"] == pytest.approx(1.0)

    def test_cap_in_contact_needs_no_su(self):
        r = group_efficiency_factor("cohesive", 3.0, cap_in_contact=True)
        assert r["eta_g"] == 1.0 and not r["su_taken"]

    def test_ufc_table_6_26_other_columns(self):
        """UFC 3-220-20 Table 6-26, read off the print."""
        f = group_efficiency_factor
        # drilled shafts, coarse-grained, cap not in contact
        for sb, eta in ((2.5, 0.65), (3.0, 0.80), (3.5, 0.90), (4.0, 1.0)):
            assert f("cohesionless", sb, cap_in_contact=False,
                     pile_type="drilled_shaft")["eta_g"] == pytest.approx(eta)
        assert f("cohesionless", 3.0, cap_in_contact=True,
                 pile_type="drilled_shaft")["eta_g"] == 1.0
        assert f("cohesive", 3.0, cap_in_contact=False, su_kPa=50.0,
                 pile_type="drilled_shaft")["eta_g"] == 1.0
        # CFAs, coarse-grained, either cap condition: 0.65 at 2.5 to 1.0 at 6
        for contact in (False, True):
            assert f("cohesionless", 6.0, cap_in_contact=contact,
                     pile_type="cfa")["eta_g"] == pytest.approx(1.0)
            assert f("cohesionless", 3.0, cap_in_contact=contact,
                     pile_type="cfa")["eta_g"] == pytest.approx(0.70)
        # DDPs, fine-grained su < 2 ksf, cap not in contact: 0.7 at 3
        assert f("cohesive", 3.0, cap_in_contact=False, su_kPa=50.0,
                 pile_type="ddp")["eta_g"] == pytest.approx(0.7)
        assert f("cohesive", 4.5, cap_in_contact=False, su_kPa=50.0,
                 pile_type="ddp")["eta_g"] == pytest.approx(0.85)
        # micropiles: 0.70 at 3, 1.0 at 6
        assert f("cohesive", 3.0, cap_in_contact=False, su_kPa=50.0,
                 pile_type="micropile")["eta_g"] == pytest.approx(0.70)
        assert f("cohesive", 4.5, cap_in_contact=False, su_kPa=50.0,
                 pile_type="micropile")["eta_g"] == pytest.approx(0.85)

    def test_below_the_printed_minimum_spacing_refused(self):
        with pytest.raises(ValueError, match="Table 6-26"):
            group_efficiency_factor("cohesive", 2.4, cap_in_contact=False,
                                    su_kPa=50.0)
        with pytest.raises(ValueError, match="Table 6-26"):
            group_efficiency_factor("cohesive", 2.9, pile_type="ddp")

    def test_unknown_types_refused(self):
        with pytest.raises(ValueError, match="soil_type"):
            group_efficiency_factor("sand", 3.0)
        with pytest.raises(ValueError, match="pile_type"):
            group_efficiency_factor("cohesionless", 3.0, pile_type="auger")


class TestBlockFailure:
    """GEC-12 Eqs. 7-36 and 7-37 (p. 286); UFC 3-220-20 Eqs. 6-38, 6-39."""

    def test_eq_7_36_hand_calc(self):
        """Made-up group: B 3.3 m, Z 4.8 m, D 15 m, su1 40, su2 60 kPa.
        Nc = 5(1 + 15/16.5)(1 + 3.3/24) = 10.86 -> 9."""
        r = block_failure_resistance(3.3, 4.8, 15.0, 40.0, 60.0)
        nc = 5 * (1 + 15 / (5 * 3.3)) * (1 + 3.3 / (5 * 4.8))
        assert r["Nc_eq_7_37"] == pytest.approx(nc)
        assert r["Nc"] == 9.0
        assert r["side_kN"] == pytest.approx(2 * 15 * (3.3 + 4.8) * 40)
        assert r["base_kN"] == pytest.approx(3.3 * 4.8 * 60 * 9)
        assert r["R_ng_kN"] == pytest.approx(9720.0 + 8553.6)

    def test_nc_below_9_for_a_shallow_wide_group(self):
        """Eq. 7-37 for small embedment and/or large widths: B = Z = 10 m,
        D = 5 m: 5 (1.1)(1.2) = 6.6."""
        r = block_failure_resistance(10.0, 10.0, 5.0, 30.0, 30.0)
        assert r["Nc"] == pytest.approx(6.6)

    def test_resistance_factor_as_printed(self):
        """GEC-12 Table 7-1 (p. 224): phi_bl 0.60, cohesive; UFC 3-220-20
        Table 6-27 (p. 433): driven 0.60, drilled shafts 0.55, micropiles
        0.60; none printed for CFA or DDP."""
        nominal = block_failure_resistance(3.3, 4.8, 15.0, 40.0, 60.0)
        assert "phi_bl" not in nominal
        for pile_type, phi in (("driven", 0.60), ("drilled_shaft", 0.55),
                               ("micropile", 0.60)):
            r = block_failure_resistance(3.3, 4.8, 15.0, 40.0, 60.0,
                                         pile_type=pile_type)
            assert r["phi_bl"] == phi
            assert r["R_r_kN"] == pytest.approx(phi * nominal["R_ng_kN"])
        for pile_type in ("cfa", "ddp"):
            r = block_failure_resistance(3.3, 4.8, 15.0, 40.0, 60.0,
                                         pile_type=pile_type)
            assert r["phi_bl"] is None and r["R_r_kN"] is None
        with pytest.raises(ValueError, match="pile_type"):
            block_failure_resistance(3.3, 4.8, 15.0, 40.0, 60.0,
                                     pile_type="auger")

    def test_b_is_the_smaller_dimension(self):
        a = block_failure_resistance(6.0, 3.0, 4.0, 30.0, 30.0)
        b = block_failure_resistance(3.0, 6.0, 4.0, 30.0, 30.0)
        assert a == b and a["B_m"] == 3.0

    def test_layout_to_the_outer_pile_faces(self):
        """Fig. 7-31: B and Z to the outer faces, (n - 1) s + d."""
        assert group_plan_dimensions(3, 4, 1.5, 1.2, 0.3) == pytest.approx(
            (2.7, 4.8))
        assert block_failure_capacity(3, 3, 1.5, 1.5, 15.0, 50.0, 0.3) == \
            pytest.approx(block_failure_resistance(
                3.3, 3.3, 15.0, 50.0, 50.0)["R_ng_kN"])
        assert block_failure_capacity(3, 3, 1.5, 1.5, 15.0, 50.0, 0.3,
                                      su2=80.0) == pytest.approx(
            block_failure_resistance(3.3, 3.3, 15.0, 50.0, 80.0)["R_ng_kN"])


class TestPMultiplier:
    """AASHTO as GEC-12 prints it (pp. 367, 370) and UFC 3-220-20 Table
    6-38 extends it; Mokwa (1999), UFC Table 6-38."""

    def test_aashto_at_3b_and_5b(self):
        assert [p_multiplier(r, 3.0) for r in (1, 2, 3, 4)] == \
            pytest.approx([0.8, 0.4, 0.3, 0.3])
        assert [p_multiplier(r, 5.0) for r in (1, 2, 3, 4)] == \
            pytest.approx([1.0, 0.85, 0.7, 0.7])

    def test_aashto_beyond_5b_to_1_at_6b(self):
        assert [p_multiplier(r, 5.5) for r in (1, 2, 3)] == \
            pytest.approx([1.0, 0.925, 0.85])
        assert all(p_multiplier(r, sd) == 1.0
                   for r in (1, 2, 3, 6) for sd in (6.0, 8.0))

    def test_mokwa_ufc_table_b_32(self):
        """UFC 3-220-20 Table B-32 at s/b = 3: 0.82, 0.67, 0.58, 0.52."""
        assert [p_multiplier(r, 3.0, "mokwa") for r in (1, 2, 3, 4, 7)] == \
            pytest.approx([0.82, 0.67, 0.58, 0.52, 0.52])

    def test_mokwa_held_at_1(self):
        assert p_multiplier(1, 7.0, "mokwa") == 1.0

    def test_below_3_diameters_refused(self):
        with pytest.raises(ValueError, match="below 3"):
            p_multiplier(1, 2.5)

    def test_row_and_method_checked(self):
        with pytest.raises(ValueError, match="row_position"):
            p_multiplier(0, 3.0)
        with pytest.raises(ValueError, match="method"):
            p_multiplier(1, 3.0, "brown")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2B: Group Settlement — Equivalent Raft Method
# ═══════════════════════════════════════════════════════════════════════

class TestGroupSettlement:
    """Tests for group_settlement_equivalent_raft (GEC-12 Sec. 7.3.5.3;
    UFC 3-220-20 Sec. 6-5.8.4)."""

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

    def test_influence_depth_to_10_percent(self):
        """UFC 3-220-20 p. 449: to where the stress is 10 % of the
        footing's bearing stress, (Bg + z)(Lg + z) = 10 Bg Lg."""
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=4, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
        )
        z = result["influence_depth_m"]
        assert (4.8 + z) * (3.3 + z) == pytest.approx(10 * 4.8 * 3.3,
                                                      rel=1e-4)

    def test_influence_depth_given(self):
        result = group_settlement_equivalent_raft(
            n_rows=3, n_cols=4, spacing=1.5, pile_diameter=0.3,
            pile_length=15.0, load_kN=2000, soil_modulus_kPa=30000,
            influence_depth_m=6.0,
        )
        assert result["influence_depth_m"] == 6.0

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
        Q = 1000 kN, M = 20000 kPa
        Influence depth: (1.8 + z)^2 = 10 * 1.8^2 -> z = 1.8 (sqrt(10) - 1)
        = 3.892 m
        Using 1 sublayer (midpoint at z = 1.946 m):
            delta_sigma = 1000 / (3.746^2) = 71.26 kPa
            S = 71.26 * 3.892 / 20000 = 0.01387 m = 13.87 mm
        """
        result = group_settlement_equivalent_raft(
            n_rows=2, n_cols=2, spacing=1.5, pile_diameter=0.3,
            pile_length=12.0, load_kN=1000, soil_modulus_kPa=20000,
            num_sublayers=1,
        )
        Bg = 1.8
        Lg = 1.8
        depth = 1.8 * (math.sqrt(10.0) - 1.0)
        z_mid = depth / 2
        delta_sigma = 1000 / ((Bg + z_mid) * (Lg + z_mid))
        expected = delta_sigma * depth / 20000 * 1000  # mm
        assert expected == pytest.approx(13.87, abs=0.01)
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
        depth = Bg * (math.sqrt(10.0) - 1.0)  # to 10 % of the stress

        result = group_settlement_equivalent_raft(
            n_rows=n_rows, n_cols=n_cols, spacing=s, pile_diameter=d,
            pile_length=L, load_kN=Q, soil_modulus_kPa=Es,
            num_sublayers=1,
        )
        z_mid = depth / 2
        expected_stress = Q / ((Bg + z_mid) * (Lg + z_mid))
        expected_settle = expected_stress * depth / Es * 1000  # mm
        assert result["settlement_mm"] == pytest.approx(expected_settle, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 2C: Group Settlement — Meyerhof (1976) SPT method
# ═══════════════════════════════════════════════════════════════════════

class TestMeyerhofGroupSettlement:
    """Tests for meyerhof_group_settlement (Meyerhof 1976; GEC-12 Vol 3 D-23).

    S[in] = 4 * pf[ksf] * If * sqrt(B[ft]) / N160, applied via an SI public
    API (B in m, load in kN / pf in kPa, settlement returned in mm).
    """

    # US -> SI unit factors for the published GEC-12 Vol 3 Table D-23 case.
    FT = 0.3048
    KIP = 4.448
    KSF = 47.88

    def test_published_50ft_case_mm(self):
        """GEC-12 Vol 3 Table D-23, 50-ft penetration row (fully specified):
        B=5 ft, Z=41 ft, Q=1540 kips, DB=5 ft, N160=59 -> S = 1.04 in.

        Driven through the SI public API (m, kN -> mm) and checked to +/-5%.
        """
        B_m = 5.0 * self.FT
        Z_m = 41.0 * self.FT
        Q_kN = 1540.0 * self.KIP
        DB_m = 5.0 * self.FT
        r = meyerhof_group_settlement(
            group_width=B_m, group_length=Z_m, N160=59,
            load_kN=Q_kN, embedment_DB=DB_m,
        )
        # Published 1.04 in = 26.5 mm, tolerance +/-5%.
        assert r["settlement_in"] == pytest.approx(1.04, rel=0.05)
        assert r["settlement_mm"] == pytest.approx(26.5, rel=0.05)
        # Intermediates match the published hand calc.
        assert r["pf_ksf"] == pytest.approx(7.512, abs=0.01)
        assert r["pf_kPa"] == pytest.approx(7.512 * self.KSF, rel=0.01)
        assert r["influence_factor"] == pytest.approx(0.9167, abs=0.001)
        assert r["B_m"] == pytest.approx(B_m, abs=1e-6)

    def test_pf_input_matches_load_input(self):
        """Supplying pf_kPa directly equals supplying the total load Q."""
        B_m = 5.0 * self.FT
        Z_m = 41.0 * self.FT
        Q_kN = 1540.0 * self.KIP
        pf_kPa = Q_kN / (B_m * Z_m)
        r_load = meyerhof_group_settlement(
            group_width=B_m, group_length=Z_m, N160=59,
            load_kN=Q_kN, embedment_DB=5.0 * self.FT,
        )
        r_pf = meyerhof_group_settlement(
            group_width=B_m, group_length=Z_m, N160=59,
            pf_kPa=pf_kPa, embedment_DB=5.0 * self.FT,
        )
        assert r_pf["settlement_mm"] == pytest.approx(r_load["settlement_mm"])

    def test_si_path_equals_us_form_hand_calc(self):
        """The SI path reproduces the US-customary closed form by hand.

        Independently evaluate S[mm] = 25.4 * 4 * pf[ksf] * If * sqrt(B[ft])
        / N160 from the raw US inputs, and confirm the SI public API returns
        the same mm value (this pins the internal unit conversion).
        """
        B_ft, Z_ft, Q_kips, DB_ft, N160 = 5.0, 41.0, 1540.0, 5.0, 59
        pf_ksf = Q_kips / (B_ft * Z_ft)
        If = 1.0 - (2.0 / 3.0 * DB_ft) / (8.0 * B_ft)
        S_in_hand = 4.0 * pf_ksf * If * math.sqrt(B_ft) / N160
        S_mm_hand = S_in_hand * 25.4

        r = meyerhof_group_settlement(
            group_width=B_ft * self.FT, group_length=Z_ft * self.FT,
            N160=N160, load_kN=Q_kips * self.KIP, embedment_DB=DB_ft * self.FT,
        )
        assert r["settlement_mm"] == pytest.approx(S_mm_hand, rel=1e-4)
        assert r["settlement_in"] == pytest.approx(S_in_hand, rel=1e-4)

    def test_si_consistent_coefficient(self):
        """The equivalent SI closed form S[mm]=C·pf[kPa]·If·√B[m]/N160 with
        C = 25.4·4/47.88/√0.3048 = 3.8435 reproduces the API value."""
        C = 25.4 * 4.0 / 47.88 / math.sqrt(0.3048)
        assert C == pytest.approx(3.8435, abs=0.001)
        B_m, Z_m = 5.0 * self.FT, 41.0 * self.FT
        pf_kPa = (1540.0 * self.KIP) / (B_m * Z_m)
        If = 1.0 - (2.0 / 3.0 * (5.0 * self.FT)) / (8.0 * B_m)
        S_mm_C = C * pf_kPa * If * math.sqrt(B_m) / 59
        r = meyerhof_group_settlement(
            group_width=B_m, group_length=Z_m, N160=59,
            pf_kPa=pf_kPa, embedment_DB=5.0 * self.FT,
        )
        assert r["settlement_mm"] == pytest.approx(S_mm_C, rel=1e-4)

    def test_no_embedment_gives_unit_influence_factor(self):
        """DB = 0 -> If = 1.0 (no embedment reduction)."""
        r = meyerhof_group_settlement(
            group_width=2.0, group_length=4.0, N160=20, pf_kPa=150.0,
        )
        assert r["influence_factor"] == pytest.approx(1.0, abs=1e-9)

    def test_influence_factor_floors_at_half(self):
        """If is floored at 0.5 for a deeply embedded narrow group.

        DB large relative to B drives 1 - (2/3·DB)/(8·B) below 0.5; the
        function clamps it to 0.5.
        """
        # Raw If would be 1 - (2/3·30)/(8·2) = 1 - 1.25 = -0.25 -> floored.
        r = meyerhof_group_settlement(
            group_width=2.0, group_length=4.0, N160=20, pf_kPa=150.0,
            embedment_DB=30.0,
        )
        assert r["influence_factor"] == pytest.approx(0.5, abs=1e-9)

    def test_b_is_smaller_dimension(self):
        """B is always the smaller plan dimension regardless of arg order."""
        r1 = meyerhof_group_settlement(
            group_width=5.0, group_length=12.0, N160=30, pf_kPa=200.0)
        r2 = meyerhof_group_settlement(
            group_width=12.0, group_length=5.0, N160=30, pf_kPa=200.0)
        assert r1["B_m"] == pytest.approx(5.0)
        assert r2["B_m"] == pytest.approx(5.0)
        assert r1["settlement_mm"] == pytest.approx(r2["settlement_mm"])

    def test_settlement_decreases_with_higher_N(self):
        """Settlement is inversely proportional to N160."""
        r_loose = meyerhof_group_settlement(
            group_width=3.0, group_length=6.0, N160=15, pf_kPa=200.0)
        r_dense = meyerhof_group_settlement(
            group_width=3.0, group_length=6.0, N160=45, pf_kPa=200.0)
        assert r_dense["settlement_mm"] < r_loose["settlement_mm"]
        # Inverse-in-N160: 45/15 = 3x (allow last-digit rounding of the mm out).
        assert r_loose["settlement_mm"] == pytest.approx(
            3 * r_dense["settlement_mm"], abs=0.01)

    def test_raises_without_load_or_pf(self):
        """Neither load_kN nor pf_kPa -> ValueError."""
        with pytest.raises(ValueError, match="load_kN.*pf_kPa"):
            meyerhof_group_settlement(
                group_width=3.0, group_length=6.0, N160=20)

    def test_raises_negative_N160(self):
        with pytest.raises(ValueError, match="N160 must be positive"):
            meyerhof_group_settlement(
                group_width=3.0, group_length=6.0, N160=-5, pf_kPa=200.0)

    def test_raises_negative_width(self):
        with pytest.raises(ValueError, match="group_width must be positive"):
            meyerhof_group_settlement(
                group_width=-3.0, group_length=6.0, N160=20, pf_kPa=200.0)

    def test_silty_sand_eq_7_51_doubles_eq_7_50(self):
        """GEC-12 Eq. 7-51 (p. 319): 8 in place of 4 for silty sand."""
        kw = dict(group_width=3.0, group_length=6.0, N160=20, pf_kPa=200.0,
                  embedment_DB=6.0)
        sand = meyerhof_group_settlement(**kw)
        silty = meyerhof_group_settlement(silty_sand=True, **kw)
        assert (sand["equation"], silty["equation"]) == ("7-50", "7-51")
        assert silty["settlement_mm"] == pytest.approx(
            2 * sand["settlement_mm"], abs=0.002)


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

    def test_vertical_group_lateral_load_raises(self):
        """A vertical group with no lateral stiffness must NOT silently drop an
        applied lateral load -- it should raise (PG-3)."""
        piles = [GroupPile(x, y, axial_stiffness=50000)
                 for x in (-1.0, 1.0) for y in (-1.0, 1.0)]
        load = GroupLoad(Vx=100.0)
        with pytest.raises(ValueError, match="lateral"):
            analyze_group_6dof(piles, load)

    def test_vertical_group_axial_moment_no_lateral_stiffness(self):
        """A vertical group with no lateral springs still solves axial + moment
        loading: the unsupported lateral DOFs are condensed, not dropped."""
        piles = [GroupPile(x, y, axial_stiffness=50000)
                 for x in (-1.0, 1.0) for y in (-1.0, 1.0)]
        load = GroupLoad(Vz=400.0, My=200.0)
        result = analyze_group_6dof(piles, load)  # must not raise
        assert result.n_piles == 4
        # My produces differential axial forces across the group.
        axials = [pf['axial_kN'] for pf in result.pile_forces]
        assert max(axials) > min(axials)


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Sign convention (PG-2) and lateral-load guards (PG-3)
# ═══════════════════════════════════════════════════════════════════════

class TestSignConvention:
    """PG-2: one explicit right-hand-rule convention end-to-end.

    Stated convention: right-handed axes with z UP; Vz and dz positive
    DOWNWARD; moments per the right-hand rule. Hence +My compresses the
    +x side, +Mx uplifts the +y side, +Mz twists counterclockwise in
    plan viewed from above.
    """

    def _pair_along_x(self):
        """Two vertical piles at x = -1, +1 (y = 0)."""
        return [GroupPile(-1.0, 0.0, axial_stiffness=50000),
                GroupPile(1.0, 0.0, axial_stiffness=50000)]

    def _pair_along_y(self):
        """Two vertical piles at y = -1, +1 (x = 0)."""
        return [GroupPile(0.0, -1.0, axial_stiffness=50000),
                GroupPile(0.0, 1.0, axial_stiffness=50000)]

    def test_vz_my_unchanged_and_methods_agree(self):
        """Vertical group under Vz + My: hand values (unchanged from the
        pre-PG-2 code) and simple == 6dof."""
        load = GroupLoad(Vz=200.0, My=100.0)
        for analyze in (analyze_vertical_group_simple, analyze_group_6dof):
            result = analyze(self._pair_along_x(), load)
            forces = {pf['x_m']: pf['axial_kN'] for pf in result.pile_forces}
            # P = Vz/n + My*x/Sx2 = 100 -/+ 50
            assert forces[1.0] == pytest.approx(150.0, abs=0.1)
            assert forces[-1.0] == pytest.approx(50.0, abs=0.1)

    def test_positive_mx_uplifts_positive_y_side(self):
        """+Mx (right-hand rule about +x, z up) puts the +y pile in
        TENSION -- both methods agree on the stated convention."""
        load = GroupLoad(Vz=0.0, Mx=100.0)
        for analyze in (analyze_vertical_group_simple, analyze_group_6dof):
            result = analyze(self._pair_along_y(), load)
            forces = {pf['y_m']: pf['axial_kN'] for pf in result.pile_forces}
            assert forces[1.0] == pytest.approx(-50.0, abs=0.1)   # tension
            assert forces[-1.0] == pytest.approx(50.0, abs=0.1)   # compression

    def test_simple_and_6dof_agree_combined_loading(self):
        """Vertical 3x3 group under Vz + Mx + My: per-pile forces from the
        simple method and the 6-DOF method are identical."""
        load = GroupLoad(Vz=2700.0, Mx=200.0, My=500.0)
        piles = create_rectangular_layout(3, 3, 2.0, 2.0,
                                          axial_stiffness=50000)
        r_simple = analyze_vertical_group_simple(piles, load)
        r_6dof = analyze_group_6dof(piles, load)
        f_simple = {(pf['x_m'], pf['y_m']): pf['axial_kN']
                    for pf in r_simple.pile_forces}
        f_6dof = {(pf['x_m'], pf['y_m']): pf['axial_kN']
                  for pf in r_6dof.pile_forces}
        for key in f_simple:
            assert f_6dof[key] == pytest.approx(f_simple[key], abs=0.2)

    def test_eccentric_vertical_group_pure_vz(self):
        """Eccentric (non-centroidal) vertical group under pure Vz:
        pile forces are statically exact and unchanged by the PG-2 rework.

        Piles at x = 0, 1, 2, 3 with equal ka; rigid cap:
        P = (0.7, 0.4, 0.1, -0.2) * Vz.
        """
        Vz = 1000.0
        piles = [GroupPile(float(x), 0.0, axial_stiffness=50000)
                 for x in range(4)]
        result = analyze_group_6dof(piles, GroupLoad(Vz=Vz))
        forces = {pf['x_m']: pf['axial_kN'] for pf in result.pile_forces}
        expected = {0.0: 0.7, 1.0: 0.4, 2.0: 0.1, 3.0: -0.2}
        for x, frac in expected.items():
            assert forces[x] == pytest.approx(frac * Vz, abs=0.2)
        # Equilibrium: sum(P) = Vz and sum(P*x) = applied My = 0
        assert sum(forces.values()) == pytest.approx(Vz, abs=0.5)
        assert sum(f * x for x, f in forces.items()) == pytest.approx(
            0.0, abs=0.5)

    def test_torsion_battered_obeys_convention(self):
        """+Mz (counterclockwise in plan, viewed from above) on a pair of
        piles battered toward +x at y = +/-2: the cap at +y moves in -x,
        stretching the +x-leaning pile there -> tension at +y, compression
        at -y; reported rz > 0."""
        piles = [GroupPile(0.0, 2.0, batter_x=15.0, axial_stiffness=50000),
                 GroupPile(0.0, -2.0, batter_x=15.0, axial_stiffness=50000)]
        result = analyze_group_6dof(piles, GroupLoad(Mz=100.0))
        assert result.cap_displacements['rz'] > 0
        forces = {pf['y_m']: pf['axial_kN'] for pf in result.pile_forces}
        assert forces[2.0] < 0   # tension at +y
        assert forces[-2.0] > 0  # compression at -y
        assert forces[2.0] == pytest.approx(-forces[-2.0], abs=0.1)

    def test_rotation_signs_right_hand_rule(self):
        """Reported cap rotations follow the right-hand rule: +My load on a
        vertical group gives ry > 0 (compressing the +x side), +Mx gives
        rx > 0 (uplifting the +y side)."""
        r_my = analyze_group_6dof(self._pair_along_x(),
                                  GroupLoad(My=100.0))
        assert r_my.cap_displacements['ry'] > 0
        r_mx = analyze_group_6dof(self._pair_along_y(),
                                  GroupLoad(Mx=100.0))
        assert r_mx.cap_displacements['rx'] > 0


class TestLateralLoadGuards:
    """PG-3: no analysis path may silently drop an applied lateral load."""

    def test_simple_method_warns_on_lateral_load(self):
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000)
        with pytest.warns(UserWarning, match="ignores Vx"):
            analyze_vertical_group_simple(piles, GroupLoad(Vz=400, Vx=50))

    def test_simple_method_warns_on_torsion(self):
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000)
        with pytest.warns(UserWarning, match="Mz"):
            analyze_vertical_group_simple(piles, GroupLoad(Vz=400, Mz=25))

    def test_simple_method_no_warning_for_axial_and_moments(self):
        import warnings as _warnings
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            analyze_vertical_group_simple(
                piles, GroupLoad(Vz=400, Mx=50, My=50))

    def test_6dof_torsion_without_stiffness_raises(self):
        """Torsion on a vertical group with no lateral stiffness cannot be
        resisted -> ValueError (not silently dropped)."""
        piles = [GroupPile(x, y, axial_stiffness=50000)
                 for x in (-1.0, 1.0) for y in (-1.0, 1.0)]
        with pytest.raises(ValueError, match="Mz"):
            analyze_group_6dof(piles, GroupLoad(Vz=100.0, Mz=50.0))

    def test_6dof_lateral_with_stiffness_carries_load(self):
        """With lateral stiffness provided, lateral load is carried and the
        cap translates in the load direction."""
        piles = create_rectangular_layout(2, 2, 2.0, 2.0,
                                          axial_stiffness=50000,
                                          lateral_stiffness=5000)
        result = analyze_group_6dof(piles, GroupLoad(Vz=400, Vx=100))
        # dx = Vx / (n*kl) = 100 / 20000 = 5 mm
        assert result.cap_displacements['dx'] == pytest.approx(
            100.0 / (4 * 5000), rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
