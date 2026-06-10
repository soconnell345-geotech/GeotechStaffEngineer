"""
Regression tests for the v5.1 axial_pile QC fixes.

AP-2: dead segmentation code removed; midpoint-rule integration now splits
      layer segments at the GWT (exact for piecewise-linear sigma_v').
AP-3: beta-method clay phi is a documented, overridable parameter
      (cohesive_phi, default 25 deg keeps prior behavior).
AP-5: uplift applies the 0.75 rule to OUTSIDE skin friction only (inside
      plug friction excluded) with optional pile self-weight.
Plus: capacity_vs_depth no longer mutates the shared analysis object.
"""

import pytest

from axial_pile.pile_types import make_pipe_pile
from axial_pile.soil_profile import AxialSoilLayer, AxialSoilProfile
from axial_pile.nordlund import skin_friction_cohesionless
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
        Qs_hand = skin_friction_cohesionless(
            33.0, sigma_mid, pile.perimeter, 8.0,
            pile_material="steel", delta_phi_ratio=None,
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
        assert r_30.Q_tip > r_25.Q_tip

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


# ---------------------------------------------------------------------------
# AP-5: uplift refinement
# ---------------------------------------------------------------------------

class TestUplift:

    def _dense_sand(self):
        return AxialSoilProfile(layers=[_sand_layer(20.0, phi=36.0)])

    def test_uplift_closed_pile_is_fraction_of_skin(self):
        """Closed-end pile: outside skin == Q_skin, so uplift is exactly
        the fraction of Q_skin (unchanged behavior)."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=True)
        r = AxialPileAnalysis(
            pile=pile, soil=self._dense_sand(), pile_length=20.0,
            include_uplift=True).compute()
        assert r.Q_uplift == pytest.approx(0.75 * r.Q_skin, rel=1e-12)

    def test_uplift_excludes_inside_friction_when_unplugged_governs(self):
        """Open pipe, unplugged governing: Q_skin includes inside friction,
        but uplift must be based on the OUTSIDE skin friction only — equal
        to 0.75 x the skin of a closed-end pile of the same diameter."""
        soil = self._dense_sand()
        pile_open = make_pipe_pile(0.6, 0.012, closed_end=False)
        pile_closed = make_pipe_pile(0.6, 0.012, closed_end=True)

        # Short embedment: inside friction + annular tip < plugged tip,
        # so the unplugged case governs.
        r_open = AxialPileAnalysis(
            pile=pile_open, soil=soil, pile_length=10.0,
            include_uplift=True).compute()
        r_closed = AxialPileAnalysis(
            pile=pile_closed, soil=soil, pile_length=10.0).compute()

        # Sanity: the unplugged case governs here (inside friction was
        # added to the reported Q_skin).
        assert r_open.Q_skin > r_closed.Q_skin

        # Regression (AP-5): uplift excludes the inside friction
        assert r_open.Q_uplift == pytest.approx(0.75 * r_closed.Q_skin,
                                                rel=1e-12)
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
        assert frac.Q_uplift == pytest.approx(base.Q_uplift * 0.6 / 0.75)


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
