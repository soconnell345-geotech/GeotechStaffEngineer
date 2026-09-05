"""
Validation tests for the CGPR #56 downdrag method family.

The primary fixture is the detailed worked example of CGPR #56 Section 3.4
(Greenfield & Filz 2009): a 100 ft, 16 in square reinforced concrete pile
(E = 500,000 ksf) through 20 ft of loose sandy fill and 70 ft of
overconsolidated clay into dense sand, one pile of a 3x3 group at 5 ft
spacing carrying 2000 kips total, with a permanent water table drop from
5 ft to 15 ft. Tests are run in the report's units (kips, ft, ksf) — the
methods are dimensionally consistent, so any coherent unit set works.

Published results (CGPR #56 Table 3.7) vs achieved are asserted per method;
achieved values and honest tolerances are stated inline. Where our
implementation deliberately differs from the report's coarse hand
tabulation (PILENEG dense grid, Fellenius equivalent-footing sublayering),
both the report-matching mode and the refined default are tested.
"""

import math

import pytest

from downdrag import (
    DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile,
    endo_method, poulos_method, fellenius_method_cgpr56, pileneg_procedure,
    rigid_block_method, drag_load_reduction_factor, drag_load_reduction_method,
    downdrag_method_comparison, consolidation_settlement_profile,
)
from downdrag.cgpr56 import profile_value, profile_integral


# ---------------------------------------------------------------------------
# Section 3.4.1 site information (kips, ft, ksf)
# ---------------------------------------------------------------------------

PILE_LENGTH = 100.0          # ft
PILE_WIDTH = 16.0 / 12.0     # ft (16 in square)
PERIMETER = 4.0 * PILE_WIDTH             # 5.333 ft
AREA = PILE_WIDTH**2                     # 1.778 ft^2
PILE_E = 500e3               # ksf
Q_STATIC = 2000.0 / 9.0      # kips per pile (2000 kip group, 9 piles)
QB = 200.0                   # ksf toe bearing capacity
BEARING_E = 8.0e3            # ksf (Table 3.2)
BEARING_NU = 0.3

# Figure 3.11 skin friction profile (ksf); repeated depth 20 = the jump at
# the fill/clay contact, repeated depth 90 = the clay/sand contact.
FS_PROFILE = [
    (0.0, 0.0), (15.0, 0.8), (20.0, 0.9),
    (20.0, 0.6), (90.0, 1.2),
    (90.0, 2.0), (100.0, 2.3),
]

# Figure 3.13 / Table 3.3 free-field settlement profile (ft), settlement
# defined at sublayer tops; constant above the clay.
SETTLEMENT_PROFILE = [
    (0.0, 0.086), (20.0, 0.086), (30.0, 0.067), (40.0, 0.052),
    (50.0, 0.039), (60.0, 0.028), (70.0, 0.017), (80.0, 0.008), (90.0, 0.0),
]

# Consolidation parameters for the clay (Table 3.3): C_er = 0.015 (stays
# overconsolidated), initial effective stress from gamma_moist = 110 pcf
# fill above the initial 5 ft water table, gamma_sat = 120 pcf fill /
# 100 pcf clay below; water table drop 5 -> 15 ft adds a constant
# 524 psf below the fill.
P0_PROFILE = [(0.0, 0.0), (5.0, 0.550), (20.0, 1.414), (90.0, 4.046)]
DP_WT = 0.524  # ksf
CONSOLIDATION = {
    "layers": [{"z_top": 20.0, "z_bot": 90.0, "C_er": 0.015}],
    "p0_profile": P0_PROFILE,
    "dp_profile": DP_WT,
    "sublayer_thickness": 10.0,
}
EQ_FOOTING_WIDTH = 11.33     # ft (3x3 group: 2 x 5 ft + 16 in, Table 3.4)
EQ_FOOTING_LOAD = 2000.0     # kips (full group load, Table 3.4)


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

class TestProfileHelpers:
    def test_value_linear(self):
        assert profile_value(FS_PROFILE, 7.5) == pytest.approx(0.4)
        assert profile_value(FS_PROFILE, 55.0) == pytest.approx(0.9)

    def test_value_at_jump_uses_deeper_side(self):
        assert profile_value(FS_PROFILE, 20.0) == pytest.approx(0.6)

    def test_value_extends_constant(self):
        assert profile_value(FS_PROFILE, 150.0) == pytest.approx(2.3)
        assert profile_value(SETTLEMENT_PROFILE, 95.0) == pytest.approx(0.0)

    def test_integral_across_jump(self):
        # 0-15: avg 0.4*15 = 6.0; 15-20: avg 0.85*5 = 4.25 (ksf*ft)
        assert profile_integral(FS_PROFILE, 0, 20) == pytest.approx(10.25)

    def test_integral_partial_segment(self):
        # 20-75 in the clay: fs 0.6 -> 1.0714, avg 0.8357 * 55
        assert profile_integral(FS_PROFILE, 20, 75) == pytest.approx(
            45.964, rel=1e-3)

    def test_unsorted_profile_rejected(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            profile_integral([(10.0, 1.0), (5.0, 2.0)], 0, 10)


# ---------------------------------------------------------------------------
# Free-field settlement profile — CGPR #56 Table 3.3
# ---------------------------------------------------------------------------

class TestFreeFieldSettlement:
    def test_table_3_3(self):
        """Reproduces Table 3.3 (published surface settlement 0.086 ft).

        Achieved: 0.0863 ft at the surface and every tabulated sublayer-top
        value within 0.001 ft of the published column.
        """
        prof = consolidation_settlement_profile(
            layers=CONSOLIDATION["layers"],
            p0_profile=P0_PROFILE,
            dp_profile=DP_WT,
            sublayer_thickness=10.0,
        )
        published = {20: 0.086, 30: 0.067, 40: 0.052, 50: 0.039,
                     60: 0.028, 70: 0.017, 80: 0.008, 90: 0.0}
        for z, s_pub in published.items():
            assert profile_value(prof, float(z)) == pytest.approx(
                s_pub, abs=1e-3), f"settlement at {z} ft"
        # Surface rides down with the full settlement
        assert profile_value(prof, 0.0) == pytest.approx(0.086, abs=1e-3)


# ---------------------------------------------------------------------------
# Endo method — CGPR #56 Section 3.4.2
# ---------------------------------------------------------------------------

class TestEndoMethod:
    def test_worked_example(self):
        """Published (3.4.2): z_n = 75 ft, F_negative = 300 k, F_max = 522 k.

        Achieved: 299.8 k / 522.0 k (report rounded the integral to 300).
        """
        res = endo_method(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, skin_friction_profile=FS_PROFILE,
            bearing_condition="stiff_flexible",
        )
        assert res.neutral_plane_depth == pytest.approx(75.0)
        assert res.drag_load == pytest.approx(300.0, abs=1.0)
        assert res.max_force == pytest.approx(522.0, abs=1.0)

    def test_end_bearing_ratio(self):
        res = endo_method(
            Q_static=0.0, pile_length=PILE_LENGTH, pile_perimeter=PERIMETER,
            skin_friction_profile=FS_PROFILE, bearing_condition="end_bearing",
        )
        assert res.neutral_plane_depth == pytest.approx(100.0)

    def test_explicit_depth(self):
        res = endo_method(
            Q_static=100.0, pile_length=PILE_LENGTH, pile_perimeter=PERIMETER,
            skin_friction_profile=FS_PROFILE, neutral_plane_depth=20.0,
        )
        # P * 10.25 ksf*ft = 54.7 kips
        assert res.drag_load == pytest.approx(54.67, abs=0.1)
        assert res.max_force == pytest.approx(154.67, abs=0.1)

    def test_requires_exactly_one_basis(self):
        with pytest.raises(ValueError, match="exactly one"):
            endo_method(
                Q_static=0, pile_length=100, pile_perimeter=PERIMETER,
                skin_friction_profile=FS_PROFILE,
                bearing_condition="floating", neutral_plane_ratio=0.75,
            )
        with pytest.raises(ValueError, match="exactly one"):
            endo_method(
                Q_static=0, pile_length=100, pile_perimeter=PERIMETER,
                skin_friction_profile=FS_PROFILE,
            )

    def test_unknown_condition_rejected(self):
        with pytest.raises(ValueError, match="bearing_condition"):
            endo_method(
                Q_static=0, pile_length=100, pile_perimeter=PERIMETER,
                skin_friction_profile=FS_PROFILE, bearing_condition="rock",
            )


# ---------------------------------------------------------------------------
# Poulos hand approximation — CGPR #56 Section 3.4.3
# ---------------------------------------------------------------------------

class TestPoulosMethod:
    def test_worked_example(self):
        """Published (3.4.3): fs1 = 814 psf, fs2 = 2150 psf, z_max = 73.6 ft,
        F_max = 542 k, pile settlement 0.046 ft (0.014 + 0.032).

        Achieved: fs1 = 813.9 psf, z_max = 73.57 ft, F_max = 541.6 k,
        settlement 0.0454 ft (0.0138 + 0.0316; the report's 0.046 carries
        two intermediate roundings).
        """
        res = poulos_method(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            depth_to_bearing_layer=90.0, toe_bearing_capacity=QB,
            skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
        )
        assert res.fs_consolidating == pytest.approx(0.814, abs=0.001)
        assert res.fs_bearing == pytest.approx(2.15, abs=0.001)
        assert res.z_max == pytest.approx(73.6, abs=0.1)
        assert not res.bearing_fully_mobilized
        assert res.neutral_plane_depth == pytest.approx(73.6, abs=0.1)
        assert res.max_force == pytest.approx(542.0, abs=1.0)
        assert res.soil_settlement_at_np == pytest.approx(0.014, abs=0.001)
        assert res.elastic_compression == pytest.approx(0.032, abs=0.001)
        assert res.pile_settlement == pytest.approx(0.046, abs=0.001)

    def test_fully_mobilized_branch(self):
        """z_max > L1 (Eq 3.7 first case): neutral plane capped at the top
        of the bearing layer; settlement needs the equivalent-pile value."""
        res = poulos_method(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            depth_to_bearing_layer=90.0, toe_bearing_capacity=400.0,
            skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
        )
        assert res.z_max > 90.0
        assert res.bearing_fully_mobilized
        assert res.neutral_plane_depth == pytest.approx(90.0)
        # No s_equivalent supplied -> settlement not computed
        assert res.pile_settlement is None

        res2 = poulos_method(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            depth_to_bearing_layer=90.0, toe_bearing_capacity=400.0,
            skin_friction_profile=FS_PROFILE, s_equivalent=0.010,
        )
        assert res2.pile_settlement == pytest.approx(
            0.010 + res2.elastic_compression)

    def test_explicit_average_frictions(self):
        res = poulos_method(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            depth_to_bearing_layer=90.0, toe_bearing_capacity=QB,
            fs_consolidating=0.814, fs_bearing=2.15,
        )
        assert res.z_max == pytest.approx(73.6, abs=0.1)


# ---------------------------------------------------------------------------
# Fellenius method (CGPR #56 formulation) — Section 3.4.4
# ---------------------------------------------------------------------------

class TestFelleniusCgpr56:
    def test_worked_example_equilibrium(self):
        """Published (3.4.4): Qb = 355.6 k, R(0) = 860.8 k, z_n = 78.4 ft,
        F_max = 541 k.

        Achieved: z_n = 78.39 ft, F_max = 541.4 k, R(0) = 860.9 k.
        """
        res = fellenius_method_cgpr56(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
        )
        assert res.toe_resistance == pytest.approx(355.6, abs=0.2)
        assert res.resistance_curve[0] == pytest.approx(860.8, abs=0.5)
        assert res.neutral_plane_depth == pytest.approx(78.4, abs=0.1)
        assert res.max_force == pytest.approx(541.0, abs=1.0)
        assert not res.pile_in_failure

    def test_worked_example_settlement(self):
        """Published (3.4.4 steps 7-8, Table 3.5): s_n = 0.084 ft,
        delta_elastic = 0.034 ft, pile settlement 0.118 ft, surface
        settlement 0.160 ft, with the 2000 kip group load on an 11.33 ft
        equivalent footing at the neutral plane.

        Achieved with the report's 10 ft sublayers: s_n = 0.0823 ft,
        pile settlement 0.116 ft, surface 0.158 ft (within 2% — the report
        hand-splits its two deepest sublayers into 5 ft; with
        sublayer_thickness = 5 ft we get 0.1177/0.160, see below).
        """
        res = fellenius_method_cgpr56(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
            consolidation=CONSOLIDATION,
            eq_footing_width=EQ_FOOTING_WIDTH, eq_footing_load=EQ_FOOTING_LOAD,
        )
        assert res.includes_pile_load_transfer
        assert res.elastic_compression == pytest.approx(0.034, abs=0.001)
        assert res.soil_settlement_at_np == pytest.approx(0.084, abs=0.003)
        assert res.pile_settlement == pytest.approx(0.118, abs=0.003)
        assert res.surface_settlement == pytest.approx(0.160, abs=0.004)

    def test_worked_example_settlement_fine_sublayers(self):
        """With 5 ft sublayers (matching the report's split of the deepest
        10 ft into two 5 ft rows): pile settlement achieved 0.1177 ft vs
        published 0.118 ft; surface 0.1597 vs 0.160/0.161."""
        consolidation = dict(CONSOLIDATION, sublayer_thickness=5.0)
        res = fellenius_method_cgpr56(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
            consolidation=consolidation,
            eq_footing_width=EQ_FOOTING_WIDTH, eq_footing_load=EQ_FOOTING_LOAD,
        )
        assert res.soil_settlement_at_np == pytest.approx(0.084, abs=0.002)
        assert res.pile_settlement == pytest.approx(0.118, abs=0.002)
        assert res.surface_settlement == pytest.approx(0.160, abs=0.002)

    def test_free_field_only_mode(self):
        res = fellenius_method_cgpr56(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
        )
        assert not res.includes_pile_load_transfer
        # Free-field only: s_n at 78.4 ft interpolates Figure 3.13
        assert res.soil_settlement_at_np == pytest.approx(0.0094, abs=0.0005)

    def test_failure_condition_flag(self):
        """Load curve above the resistance curve everywhere (step 5)."""
        res = fellenius_method_cgpr56(
            Q_static=2000.0, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
        )
        assert res.pile_in_failure
        assert res.neutral_plane_depth == 0.0

    def test_neutral_plane_at_toe(self):
        """Curves never cross -> neutral plane assumed at the pile toe."""
        res = fellenius_method_cgpr56(
            Q_static=0.0, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=5000.0, skin_friction_profile=FS_PROFILE,
        )
        assert res.neutral_plane_depth == pytest.approx(PILE_LENGTH)

    def test_consolidation_requires_footing_width(self):
        with pytest.raises(ValueError, match="eq_footing_width"):
            fellenius_method_cgpr56(
                Q_static=Q_STATIC, pile_length=PILE_LENGTH,
                pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
                toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
                consolidation=CONSOLIDATION,
            )


# ---------------------------------------------------------------------------
# PILENEG procedure — CGPR #56 Section 3.4.5
# ---------------------------------------------------------------------------

class TestPilenegProcedure:
    def test_worked_example_report_grid(self):
        """With the report's trial depths [50, 60, 70, 80] ft.

        Published (3.4.5 / Table 3.6): Q_toe(60) = 155.3 k, envelope
        delta(60) = 0.0238 ft, z_n = 62.4 ft, Q_toe = 180 k,
        F_max = 453 k, pile settlement 0.049 ft.

        Achieved: Q_toe(60) = 155.3 k, delta(60) = 0.0238 ft,
        z_n = 62.43 ft, Q_toe = 180.1 k, F_max = 453.8 k, settlement
        0.0490 ft.
        """
        res = pileneg_procedure(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, pile_width=PILE_WIDTH,
            bearing_E=BEARING_E, bearing_nu=BEARING_NU,
            skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
            trial_depths=[50.0, 60.0, 70.0, 80.0],
        )
        # Table 3.6 spot checks at z = 60 ft
        i60 = res.trial_depths.index(60.0)
        assert res.trial_toe_loads[i60] == pytest.approx(155.3, abs=0.5)
        assert res.envelope[i60] == pytest.approx(0.0238, abs=0.0003)

        assert res.neutral_plane_depth == pytest.approx(62.4, abs=0.1)
        assert res.toe_load == pytest.approx(180.0, abs=1.0)
        assert not res.toe_fully_mobilized
        assert res.max_force == pytest.approx(453.0, abs=1.0)
        assert res.soil_settlement_at_np == pytest.approx(0.025, abs=0.001)
        assert res.elastic_compression == pytest.approx(0.024, abs=0.001)
        assert res.pile_settlement == pytest.approx(0.049, abs=0.001)

    def test_worked_example_table_3_6(self):
        """Full Table 3.6 envelope check: published (z_n, Q_toe, delta_pile)
        rows (50, 59.4, 0.0167), (60, 155.3, 0.0238), (70, 260.4, 0.0301),
        (80, 374.6, 0.0355). Achieved within 0.5 k / 0.0003 ft."""
        res = pileneg_procedure(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, pile_width=PILE_WIDTH,
            bearing_E=BEARING_E, bearing_nu=BEARING_NU,
            skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
            trial_depths=[50.0, 60.0, 70.0, 80.0],
        )
        published = [(50.0, 59.4, 0.0167), (60.0, 155.3, 0.0238),
                     (70.0, 260.4, 0.0301), (80.0, 374.6, 0.0355)]
        for (z, qt, dp) in published:
            i = res.trial_depths.index(z)
            assert res.trial_toe_loads[i] == pytest.approx(qt, abs=0.5)
            assert res.envelope[i] == pytest.approx(dp, abs=0.0003)

    def test_dense_grid_close_to_report(self):
        """Default dense grid: the continuous envelope crosses at 62.6 ft
        (the report's 62.4 ft comes from linear interpolation between its
        60 and 70 ft trial points). Forces stay within 0.5%."""
        res = pileneg_procedure(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, pile_width=PILE_WIDTH,
            bearing_E=BEARING_E, bearing_nu=BEARING_NU,
            skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
        )
        assert res.neutral_plane_depth == pytest.approx(62.4, abs=0.5)
        assert res.max_force == pytest.approx(453.0, rel=0.005)
        assert res.pile_settlement == pytest.approx(0.049, abs=0.001)

    def test_toe_fully_mobilized_flag(self):
        """With a weak toe (qb = 50 ksf -> capacity 88.9 k) the equilibrium
        toe load at the neutral plane exceeds capacity; the report says to
        fall back to the Fellenius method (step 7)."""
        res = pileneg_procedure(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=50.0, pile_width=PILE_WIDTH,
            bearing_E=BEARING_E, bearing_nu=BEARING_NU,
            skin_friction_profile=FS_PROFILE,
            settlement_profile=SETTLEMENT_PROFILE,
        )
        assert res.toe_fully_mobilized
        assert "Fellenius" in res.to_dict()["notes"]


# ---------------------------------------------------------------------------
# Pile groups — CGPR #56 Section 4.3
# ---------------------------------------------------------------------------

class TestPileGroups:
    def test_rigid_block(self):
        """Eq 4.1 hand check: Qg/n = 222.2 k; perimeter pile with
        s = 5 ft, z_n = 78.4 ft, cu = 0.5 ksf adds 196 k; interior pile
        with delta_q = 0.524 ksf adds 13.1 k."""
        res = rigid_block_method(
            Q_static_group=2000.0, n_piles=9, spacing=5.0,
            neutral_plane_depth=78.4, cu_average=0.5, delta_q=DP_WT,
        )
        assert res.static_per_pile == pytest.approx(222.22, abs=0.1)
        assert res.perimeter_pile_max_force == pytest.approx(
            222.22 + 5.0 * 78.4 * 0.5, abs=0.1)
        assert res.interior_pile_max_force == pytest.approx(
            222.22 + 25.0 * DP_WT, abs=0.1)

    def test_reduction_factor_table_4_1(self):
        """Table 4.1 anchors and Figure 4.2 extensions: constant below
        s/d = 2.5, linear 2.5-5, step to 1.0 beyond 5."""
        assert drag_load_reduction_factor(2.5, "interior") == pytest.approx(0.15)
        assert drag_load_reduction_factor(2.5, "side") == pytest.approx(0.4)
        assert drag_load_reduction_factor(2.5, "corner") == pytest.approx(0.5)
        assert drag_load_reduction_factor(5.0, "interior") == pytest.approx(0.5)
        assert drag_load_reduction_factor(5.0, "side") == pytest.approx(0.8)
        assert drag_load_reduction_factor(5.0, "corner") == pytest.approx(0.9)
        # Interpolation at s/d = 3.75
        assert drag_load_reduction_factor(3.75, "corner") == pytest.approx(0.7)
        # Figure 4.2 conservative extensions
        assert drag_load_reduction_factor(1.5, "interior") == pytest.approx(0.15)
        assert drag_load_reduction_factor(6.0, "side") == pytest.approx(1.0)

    def test_reduction_method_eq_4_2(self):
        """Eq 4.2 with the worked example's single-pile Fellenius result
        (F_max = 541 k) at the example's s/d = 5 ft / 1.333 ft = 3.75,
        corner pile: A = 0.7 -> F = 0.7*(541 - 222.2) + 222.2 = 445.4 k."""
        res = drag_load_reduction_method(
            F_max_single=541.0, Q_static_group=2000.0, n_piles=9,
            location="corner", spacing=5.0, pile_diameter=PILE_WIDTH,
        )
        assert res.s_over_d == pytest.approx(3.75)
        assert res.reduction_factor == pytest.approx(0.7)
        assert res.max_force_group_pile == pytest.approx(445.4, abs=0.5)

    def test_reduction_bad_location(self):
        with pytest.raises(ValueError, match="location"):
            drag_load_reduction_factor(3.0, "middle")


# ---------------------------------------------------------------------------
# Method comparison runner — CGPR #56 Table 3.7
# ---------------------------------------------------------------------------

class TestMethodComparison:
    def test_table_3_7(self):
        """All four methods on the Section 3.4 inputs reproduce Table 3.7:
        Endo 75.0/522, Poulos 73.6/542/0.046, Fellenius 78.4/541/0.118,
        PILENEG 62.4/453/0.049 (PILENEG on the report's trial grid)."""
        cmp = downdrag_method_comparison(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
            consolidation=dict(CONSOLIDATION, sublayer_thickness=5.0),
            # The report's own free-field table (3.3) feeds Poulos/PILENEG;
            # Fellenius uses the consolidation recompute (Table 3.5).
            settlement_profile=SETTLEMENT_PROFILE,
            endo_bearing_condition="stiff_flexible",
            depth_to_bearing_layer=90.0,
            pile_width=PILE_WIDTH, bearing_E=BEARING_E, bearing_nu=BEARING_NU,
            eq_footing_width=EQ_FOOTING_WIDTH, eq_footing_load=EQ_FOOTING_LOAD,
            pileneg_trial_depths=[50.0, 60.0, 70.0, 80.0],
        )
        by_method = {r["method"]: r for r in cmp.rows}
        assert not any(r.get("error") for r in cmp.rows), cmp.rows

        assert by_method["endo"]["neutral_plane_depth"] == pytest.approx(75.0)
        assert by_method["endo"]["max_force"] == pytest.approx(522, abs=1)

        assert by_method["poulos"]["neutral_plane_depth"] == pytest.approx(
            73.6, abs=0.1)
        assert by_method["poulos"]["max_force"] == pytest.approx(542, abs=1)
        assert by_method["poulos"]["pile_settlement"] == pytest.approx(
            0.046, abs=0.001)

        f = by_method["fellenius_cgpr56"]
        assert f["neutral_plane_depth"] == pytest.approx(78.4, abs=0.1)
        assert f["max_force"] == pytest.approx(541, abs=1)
        assert f["pile_settlement"] == pytest.approx(0.118, abs=0.002)

        p = by_method["pileneg"]
        assert p["neutral_plane_depth"] == pytest.approx(62.4, abs=0.1)
        assert p["max_force"] == pytest.approx(453, abs=1)
        assert p["pile_settlement"] == pytest.approx(0.049, abs=0.001)

        # summary() renders a table without raising
        assert "fellenius_cgpr56" in cmp.summary()

    def test_skips_with_reasons_when_inputs_missing(self):
        cmp = downdrag_method_comparison(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
        )
        by_method = {r["method"]: r for r in cmp.rows}
        assert "error" in by_method["endo"]
        assert "error" in by_method["poulos"]
        assert "error" in by_method["pileneg"]
        # Fellenius always runs (no settlement inputs -> equilibrium only)
        assert by_method["fellenius_cgpr56"]["max_force"] == pytest.approx(
            541, abs=1)

    def test_to_dict_json_safe(self):
        import json
        cmp = downdrag_method_comparison(
            Q_static=Q_STATIC, pile_length=PILE_LENGTH,
            pile_perimeter=PERIMETER, pile_area=AREA, pile_E=PILE_E,
            toe_bearing_capacity=QB, skin_friction_profile=FS_PROFILE,
            endo_bearing_condition="stiff_flexible",
            depth_to_bearing_layer=90.0,
            settlement_profile=SETTLEMENT_PROFILE,
            pile_width=PILE_WIDTH, bearing_E=BEARING_E,
        )
        json.dumps(cmp.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Cross-check: existing DowndragAnalysis (Fellenius unified) vs the
# CGPR #56 Fellenius formulation on the report's worked example
# ---------------------------------------------------------------------------

class TestCrossCheckExistingFellenius:
    """Our pre-existing DowndragAnalysis implements the Fellenius unified
    neutral-plane equilibrium (load curve vs resistance curve) like the
    CGPR #56 formulation, but with two documented differences: it builds
    the skin friction from soil parameters (beta/alpha) rather than a
    user-supplied profile, and it includes the pile self-weight in the
    load curve. With the report's fs profile injected via thin cohesive
    layers (alpha = 1, cu = fs), pile weight zeroed, and the toe pinned to
    the report's 355.6 kips, both implementations must find the same
    neutral plane and drag load.
    """

    def test_neutral_plane_and_dragload_match(self):
        FT = 0.3048
        KIP = 4.448222
        PSF = 0.04788026  # kPa

        n_thin = 100  # 1-ft-thick cohesive layers carrying cu = fs(mid)
        layers = []
        for i in range(n_thin):
            z_mid = (i + 0.5) * 1.0
            fs_psf = 1000.0 * profile_value(FS_PROFILE, z_mid)  # ksf -> psf
            layers.append(DowndragSoilLayer(
                thickness=1.0 * FT, soil_type="cohesive", unit_weight=18.0,
                cu=fs_psf * PSF, alpha=1.0,
            ))
        soil = DowndragSoilProfile(layers=layers, gwt_depth=15.0 * FT)

        area_m2 = AREA * FT**2
        cu_tip = 1000.0 * profile_value(FS_PROFILE, 99.5) * PSF
        Qb_target = 355.56 * KIP
        Nt = Qb_target / (cu_tip * area_m2)

        analysis = DowndragAnalysis(
            soil=soil,
            pile_length=PILE_LENGTH * FT,
            pile_diameter=PILE_WIDTH * FT,   # placeholder; overrides below
            pile_perimeter=PERIMETER * FT,
            pile_area=area_m2,
            pile_E=PILE_E * 47.88026,        # ksf -> kPa
            pile_unit_weight=0.0,            # report ignores pile weight
            Q_dead=Q_STATIC * KIP,
            Nt=Nt,
        )
        result = analysis.compute()

        # Report: z_n = 78.4 ft = 23.90 m; drag load = 541 - 222 = 319 kips
        # = 1419 kN. Achieved by DowndragAnalysis: within 1.5%
        # (discretization of the fs profile into 1 ft steps).
        assert result.neutral_plane_depth == pytest.approx(
            78.4 * FT, rel=0.015)
        assert result.dragload == pytest.approx(318.8 * KIP, rel=0.02)


# ---------------------------------------------------------------------------
# Consolidation strain cases
# ---------------------------------------------------------------------------

class TestLogStrainCases:
    def test_oc_to_nc_transition(self):
        prof = consolidation_settlement_profile(
            layers=[{"z_top": 0.0, "z_bot": 1.0, "C_er": 0.02,
                     "C_ec": 0.2, "p_c": 150.0}],
            p0_profile=[(0.0, 100.0), (1.0, 100.0)],
            dp_profile=100.0,
            sublayer_thickness=1.0,
        )
        # strain = 0.02*log10(150/100) + 0.2*log10(200/150)
        expected = 0.02 * math.log10(1.5) + 0.2 * math.log10(200.0 / 150.0)
        assert prof[0][1] == pytest.approx(expected, rel=1e-6)

    def test_nc_when_no_pc(self):
        prof = consolidation_settlement_profile(
            layers=[{"z_top": 0.0, "z_bot": 1.0, "C_er": 0.02, "C_ec": 0.2}],
            p0_profile=[(0.0, 100.0), (1.0, 100.0)],
            dp_profile=100.0,
            sublayer_thickness=1.0,
        )
        assert prof[0][1] == pytest.approx(0.2 * math.log10(2.0), rel=1e-6)

    def test_no_settlement_without_stress_increase(self):
        prof = consolidation_settlement_profile(
            layers=[{"z_top": 0.0, "z_bot": 1.0, "C_er": 0.02}],
            p0_profile=[(0.0, 100.0), (1.0, 100.0)],
            dp_profile=0.0,
        )
        assert prof[0][1] == 0.0
