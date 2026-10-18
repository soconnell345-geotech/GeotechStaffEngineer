"""
Regression tests for the v5.1 axial_pile QC fixes.

AP-2: dead segmentation code removed; midpoint-rule integration now splits
      layer segments at the GWT (exact for piecewise-linear sigma_v').
AP-3: beta-method clay phi is a documented, overridable parameter
      (cohesive_phi, default 25 deg keeps prior behavior).
AP-5: uplift applies the uplift fraction to OUTSIDE skin friction only
      (inside plug friction excluded) with optional pile self-weight. The
      fraction's default is 1.0 since GM-121 (GEC-12 Eq. 7-38).
Plus: capacity_vs_depth no longer mutates the shared analysis object.
"""

import pytest

from axial_pile.pile_types import make_pipe_pile
from axial_pile.soil_profile import AxialSoilLayer, AxialSoilProfile
from axial_pile.nordlund import displaced_volume, skin_friction_cohesionless
from axial_pile.capacity import AxialPileAnalysis


def _sand_layer(thickness, phi=33.0, gamma=18.5):
    return AxialSoilLayer(thickness, "cohesionless", gamma, friction_angle=phi)


# ---------------------------------------------------------------------------
# AP-2: GWT-aware midpoint integration
# ---------------------------------------------------------------------------

class TestGwtSplitIntegration:

    def test_gwt_inside_layer_matches_explicitly_split_profile(self):
        """A GWT inside a layer must give the same skin friction as the
        identical profile with the layer manually split at the GWT
        (the manually split profile was always handled exactly)."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)

        unsplit = AxialSoilProfile(
            layers=[_sand_layer(10.0)], gwt_depth=4.0,
        )
        split = AxialSoilProfile(
            layers=[_sand_layer(4.0), _sand_layer(6.0)], gwt_depth=4.0,
        )

        r_unsplit = AxialPileAnalysis(
            pile=pile, soil=unsplit, pile_length=10.0).compute()
        r_split = AxialPileAnalysis(
            pile=pile, soil=split, pile_length=10.0).compute()

        assert r_unsplit.Q_skin == pytest.approx(r_split.Q_skin, rel=1e-12)
        assert r_unsplit.Q_ultimate == pytest.approx(r_split.Q_ultimate,
                                                     rel=1e-12)

    def test_no_gwt_midpoint_rule_unchanged(self):
        """Without a GWT inside the layer, the per-layer midpoint rule is
        unchanged (hand integration with the same unit-friction call)."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[_sand_layer(8.0)], gwt_depth=None)
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=8.0)
        result = analysis.compute()

        sigma_mid = soil.effective_stress_at_depth(4.0)
        ratio = result.layer_breakdown[0]["delta_phi_ratio"]
        Qs_hand = skin_friction_cohesionless(
            33.0, sigma_mid, pile.perimeter, 8.0, ratio,
            displaced_volume(pile.tip_area), notes=[],
        )
        assert result.Q_skin == pytest.approx(Qs_hand, rel=1e-12)


# ---------------------------------------------------------------------------
# AP-3: overridable beta-method clay phi
# ---------------------------------------------------------------------------

class TestCohesivePhiOverride:

    def _clay_profile(self):
        return AxialSoilProfile(
            layers=[AxialSoilLayer(12.0, "cohesive", 17.0, cohesion=60.0)],
        )

    def test_default_25_matches_explicit_25(self):
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = self._clay_profile()
        r_default = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=12.0, method="beta").compute()
        r_25 = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=12.0, method="beta",
            cohesive_phi=25.0).compute()
        assert r_default.Q_ultimate == r_25.Q_ultimate
        assert r_default.Q_skin == r_25.Q_skin

    def test_higher_phi_gives_higher_capacity(self):
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = self._clay_profile()
        r_25 = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=12.0, method="beta").compute()
        r_30 = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=12.0, method="beta",
            cohesive_phi=30.0).compute()
        assert r_30.Q_skin > r_25.Q_skin
        # GM-121: a cohesive toe takes cohesive_Nt (GEC-12: a low Nt in
        # clays), not an Nt from phi', so the toe does not move
        assert r_30.Q_tip == r_25.Q_tip

    def test_ignored_for_auto_method(self):
        """cohesive_phi only affects method='beta'."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = self._clay_profile()
        r_a = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=12.0).compute()
        r_b = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=12.0,
            cohesive_phi=35.0).compute()
        assert r_a.Q_ultimate == r_b.Q_ultimate

    def test_per_layer_phi_wins_over_global(self):
        """Das-sweep ergonomics fix: a friction_angle supplied ON a cohesive
        layer beats the global cohesive_phi (which is only the fallback)."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil_explicit = AxialSoilProfile(layers=[
            AxialSoilLayer(12.0, "cohesive", 17.0, cohesion=60.0,
                           friction_angle=20.0)])
        # Per-layer 20 under the default global (25) == global forced to 20.
        r_layer = AxialPileAnalysis(
            pile=pile, soil=soil_explicit, pile_length=12.0,
            method="beta").compute()
        r_global20 = AxialPileAnalysis(
            pile=pile, soil=self._clay_profile(), pile_length=12.0,
            method="beta", cohesive_phi=20.0).compute()
        assert r_layer.Q_skin == pytest.approx(r_global20.Q_skin, rel=1e-12)
        assert r_layer.Q_tip == pytest.approx(r_global20.Q_tip, rel=1e-12)
        # And it is NOT the old (global-25) answer.
        r_old25 = AxialPileAnalysis(
            pile=pile, soil=self._clay_profile(), pile_length=12.0,
            method="beta").compute()
        assert r_layer.Q_skin < r_old25.Q_skin

    def test_layers_without_phi_still_use_global(self):
        """No per-layer angle -> the global fallback behaviour is unchanged."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        r_default = AxialPileAnalysis(
            pile=pile, soil=self._clay_profile(), pile_length=12.0,
            method="beta").compute()
        r_25 = AxialPileAnalysis(
            pile=pile, soil=self._clay_profile(), pile_length=12.0,
            method="beta", cohesive_phi=25.0).compute()
        assert r_default.Q_ultimate == r_25.Q_ultimate


# ---------------------------------------------------------------------------
# AP-5: uplift refinement
# ---------------------------------------------------------------------------

class TestUplift:

    def _dense_sand(self):
        return AxialSoilProfile(layers=[_sand_layer(20.0, phi=36.0)])

    def test_uplift_closed_pile_is_fraction_of_skin(self):
        """Closed-end pile: outside skin == Q_skin, so uplift is exactly
        the fraction of Q_skin (1.0 by default, GEC-12 Eq. 7-38)."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=True)
        r = AxialPileAnalysis(
            pile=pile, soil=self._dense_sand(), pile_length=20.0,
            include_uplift=True).compute()
        assert r.Q_uplift == pytest.approx(r.Q_skin, rel=1e-12)

    def test_uplift_excludes_inside_friction_when_unplugged_governs(self):
        """Open pipe, unplugged governing: Q_skin includes inside friction,
        but uplift must be based on the OUTSIDE skin friction only (the
        layer breakdown)."""
        soil = self._dense_sand()
        pile_open = make_pipe_pile(0.6, 0.012, closed_end=False)

        r_open = AxialPileAnalysis(
            pile=pile_open, soil=soil, pile_length=10.0,
            include_uplift=True, uplift_skin_fraction=0.75).compute()
        outside = sum(lb["skin_friction_kN"] for lb in r_open.layer_breakdown)

        # Sanity: the unplugged case governs here (inside friction was
        # added to the reported Q_skin).
        assert r_open.Q_skin > outside

        # Regression (AP-5): uplift excludes the inside friction
        assert r_open.Q_uplift == pytest.approx(0.75 * outside, rel=1e-12)
        assert r_open.Q_uplift < 0.75 * r_open.Q_skin

    def test_uplift_pile_weight_and_fraction(self):
        pile = make_pipe_pile(0.6, 0.012, closed_end=True)
        base = AxialPileAnalysis(
            pile=pile, soil=self._dense_sand(), pile_length=20.0,
            include_uplift=True).compute()
        with_w = AxialPileAnalysis(
            pile=pile, soil=self._dense_sand(), pile_length=20.0,
            include_uplift=True, pile_weight=50.0).compute()
        frac = AxialPileAnalysis(
            pile=pile, soil=self._dense_sand(), pile_length=20.0,
            include_uplift=True, uplift_skin_fraction=0.6).compute()

        assert with_w.Q_uplift == pytest.approx(base.Q_uplift + 50.0)
        assert frac.Q_uplift == pytest.approx(base.Q_uplift * 0.6)


# ---------------------------------------------------------------------------
# capacity_vs_depth no longer mutates shared state
# ---------------------------------------------------------------------------

class TestCapacityVsDepthReentrant:

    def test_no_mutation_and_identical_results(self):
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[_sand_layer(15.0)], gwt_depth=5.0)
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0)

        sweep = analysis.capacity_vs_depth(depth_min=5.0, depth_max=15.0,
                                           n_points=5)
        assert analysis.pile_length == 15.0
        assert len(sweep) == 5

        # Each sweep point matches an independent analysis at that depth
        for row in sweep:
            fresh = AxialPileAnalysis(
                pile=pile, soil=soil, pile_length=row["depth_m"]).compute()
            assert row["Q_ultimate_kN"] == pytest.approx(
                round(fresh.Q_ultimate, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
