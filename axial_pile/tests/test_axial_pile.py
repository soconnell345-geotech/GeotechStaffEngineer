"""
Validation tests for the axial_pile module.

Tests cover pile section creation, soil profile stress computations,
alpha/Nordlund/beta methods, and combined analysis.

References:
    [1] FHWA GEC-12 (FHWA-NHI-16-009), Chapters 7-8
    [2] FHWA Soils & Foundations Reference Manual, Vol II
"""

import math
import pytest

from axial_pile.pile_types import (
    PileSection, make_pipe_pile, make_concrete_pile, make_h_pile,
)
from axial_pile.soil_profile import AxialSoilLayer, AxialSoilProfile
from axial_pile.tomlinson import (
    adhesion, adhesion_fig_7_17, alpha_tomlinson, api_alpha,
    end_bearing_cohesive, skin_friction_cohesive,
)
from axial_pile.nordlund import (
    alpha_t, correction_factor, delta_phi_ratio_fig_7_9, displaced_volume,
    end_bearing_cohesionless, k_delta, n_q_prime, nordlund_side_factors,
    skin_friction_cohesionless, _limiting_tip_resistance,
    toe_limit_chart_note,
)
from axial_pile.beta_method import beta_from_phi, skin_friction_beta, Nt_from_phi
from axial_pile.capacity import AxialPileAnalysis
from axial_pile.results import AxialPileResult


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Pile Section Creation
# ═══════════════════════════════════════════════════════════════════════

class TestPileSections:
    """Test pile section creation and property computation."""

    def test_pipe_pile_closed(self):
        """Closed-end pipe pile: tip area = full circle."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        assert pile.pile_type == "pipe_closed"
        assert pile.width == pytest.approx(0.3239, rel=1e-4)
        expected_tip = math.pi / 4 * 0.3239**2
        assert pile.tip_area == pytest.approx(expected_tip, rel=1e-4)
        assert pile.perimeter == pytest.approx(math.pi * 0.3239, rel=1e-4)

    def test_pipe_pile_open(self):
        """Open-ended pipe pile: tip area = annular ring."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=False)
        assert pile.pile_type == "pipe_open"
        r_o = 0.3
        r_i = 0.3 - 0.012
        expected_area = math.pi * (r_o**2 - r_i**2)
        assert pile.area == pytest.approx(expected_area, rel=1e-4)
        # Tip area for open-ended = steel annular area
        assert pile.tip_area == pytest.approx(expected_area, rel=1e-4)

    def test_concrete_square(self):
        """Square concrete pile."""
        pile = make_concrete_pile(0.3556, shape="square")  # 14" pile
        assert pile.area == pytest.approx(0.3556**2, rel=1e-4)
        assert pile.perimeter == pytest.approx(4 * 0.3556, rel=1e-4)
        assert pile.E == 25e6  # concrete

    def test_h_pile_database(self):
        """HP14x117 from built-in database."""
        pile = make_h_pile("HP14x117")
        assert pile.pile_type == "h_pile"
        assert pile.area == pytest.approx(0.02219, rel=0.01)
        # Box perimeter = 2*(depth + bf)
        assert pile.perimeter > 1.0

    def test_h_pile_unknown(self):
        """Unknown HP shape should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown HP shape"):
            make_h_pile("HP99x999")

    def test_all_hp_shapes(self):
        """All built-in HP shapes should be constructible."""
        shapes = ["HP10x42", "HP10x57", "HP12x53", "HP12x63", "HP12x74",
                  "HP12x84", "HP14x73", "HP14x89", "HP14x102", "HP14x117"]
        for s in shapes:
            pile = make_h_pile(s)
            assert pile.area > 0
            assert pile.perimeter > 0


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Soil Profile
# ═══════════════════════════════════════════════════════════════════════

class TestSoilProfile:
    """Test soil profile and effective stress calculations."""

    def test_effective_stress_no_gwt(self):
        """Effective stress without groundwater = total stress."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
        ])
        sigma_v = soil.effective_stress_at_depth(5.0)
        assert sigma_v == pytest.approx(90.0, rel=1e-4)  # 18*5

    def test_effective_stress_with_gwt(self):
        """Effective stress with GWT at 3m depth."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
        ], gwt_depth=3.0)
        sigma_v = soil.effective_stress_at_depth(5.0)
        # 18*3 + (18-9.81)*2 = 54 + 16.38 = 70.38
        expected = 18 * 3 + (18 - 9.81) * 2
        assert sigma_v == pytest.approx(expected, rel=0.01)

    def test_effective_stress_multilayer(self):
        """Effective stress through multiple layers."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(5, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(5, 'cohesive', 17.0, cohesion=50),
        ])
        # At 7m: 18*5 + 17*2 = 90 + 34 = 124
        sigma_v = soil.effective_stress_at_depth(7.0)
        assert sigma_v == pytest.approx(124.0, rel=0.01)

    def test_layer_at_depth(self):
        """Correct layer returned for a given depth."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(5, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(5, 'cohesive', 17.0, cohesion=50),
        ])
        layer = soil.layer_at_depth(3.0)
        assert layer.soil_type == "cohesionless"
        layer = soil.layer_at_depth(7.0)
        assert layer.soil_type == "cohesive"

    def test_invalid_soil_type(self):
        """Invalid soil type should raise ValueError."""
        with pytest.raises(ValueError, match="soil_type"):
            AxialSoilLayer(5, 'rock', 22.0, friction_angle=40)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: alpha-method (GEC-12 Figure 7-17, after Tomlinson 1979)
# ═══════════════════════════════════════════════════════════════════════

KSF = 47.880259


class TestTomlinson:
    """Adhesion Ca from GEC-12 Figure 7-17, measured off the printed figure
    (2026-10-18). Pins are chart nodes; the curves' shape is checked by
    what the figure shows."""

    @pytest.mark.parametrize("curve, D_over_b, su, ca", [
        ("concrete_timber_corrugated", 10, 0.5, 0.503),
        ("concrete_timber_corrugated", 10, 2.0, 1.369),
        ("concrete_timber_corrugated", 40, 2.0, 1.799),
        ("smooth_steel", 10, 2.0, 1.17),
        ("smooth_steel", 40, 2.0, 1.516),
        ("smooth_steel", 40, 3.0, 1.294),
    ])
    def test_chart_nodes(self, curve, D_over_b, su, ca):
        assert adhesion_fig_7_17(su * KSF, D_over_b, curve) == pytest.approx(
            ca * KSF, rel=1e-6)

    def test_D_over_b_linear_between_the_curves_and_held_outside(self):
        c10 = adhesion_fig_7_17(2.0 * KSF, 10, "smooth_steel")
        c40 = adhesion_fig_7_17(2.0 * KSF, 40, "smooth_steel")
        assert adhesion_fig_7_17(2.0 * KSF, 25, "smooth_steel") == \
            pytest.approx((c10 + c40) / 2, rel=1e-9)
        assert adhesion_fig_7_17(2.0 * KSF, 5, "smooth_steel") == c10
        assert adhesion_fig_7_17(2.0 * KSF, 80, "smooth_steel") == c40

    def test_solid_40b_curve_starts_at_1_ksf(self):
        """Below su = 1.0 ksf the chart draws one solid curve for both
        D/b; the D = 40b curve starts at 1.0 ksf, above it."""
        lo = adhesion_fig_7_17(0.9 * KSF, 40, "concrete_timber_corrugated")
        assert lo == pytest.approx(0.875 * KSF, rel=1e-6)
        hi = adhesion_fig_7_17(1.0 * KSF, 40, "concrete_timber_corrugated")
        assert hi == pytest.approx(1.106 * KSF, rel=1e-6)

    def test_alpha_soft_clay_about_one(self):
        """Very soft clay (su = 10 kPa, below the curves' 0.25 ksf start):
        the curve's alpha at its start, as drawn (GM-229: 1.04 on the solid
        curve, no longer held at 1)."""
        assert alpha_tomlinson(10, "steel") == pytest.approx(0.92, abs=0.005)
        assert alpha_tomlinson(10, "concrete") == pytest.approx(1.04)

    def test_smooth_steel_at_or_below_the_corrugated_curve(self):
        for cu in [25, 50, 100, 140]:
            for db in (10, 40):
                assert (alpha_tomlinson(cu, "steel", D_over_b=db)
                        <= alpha_tomlinson(cu, "concrete", D_over_b=db))

    def test_alpha_falls_past_the_peak(self):
        """Ca peaks near su = 2 ksf; alpha = Ca/su falls throughout."""
        prev = 2.0
        for cu in [12, 25, 50, 100, 140]:
            alpha = alpha_tomlinson(cu, "steel")
            assert alpha <= prev
            prev = alpha

    def test_beyond_the_chart_held_or_api_alpha(self):
        """su beyond Fig. 7-17: Ca held at the curve's end by default (GEC-12
        Vol 3 Table D-100), or API alpha (GEC-12 p. 247)."""
        su, sv = 3.5 * KSF, 2.5 * KSF
        notes = []
        ca_end, source = adhesion(su, 40, "smooth_steel", sigma_v=sv,
                                  notes=notes)
        assert source == "Fig. 7-17 end"
        assert ca_end == pytest.approx(1.279 * KSF, rel=1e-6)
        assert len(notes) == 1 and "past the end of GEC-12 Figure 7-17"             in notes[0] and notes[0].startswith("alpha method")
        ca, source = adhesion(su, 40, "smooth_steel", sigma_v=sv,
                              beyond_chart="api")
        assert source == "API alpha"
        psi = su / sv
        assert ca == pytest.approx(0.5 * psi ** -0.25 * su, rel=1e-9)

    def test_ca_read_as_drawn(self):
        """GM-229 (owner OWN-63): within the chart Ca is Fig. 7-17 as
        drawn, above su where the D = 40b solid curve passes above Ca = su
        (alpha about 1.1 at 1.0 ksf); the alpha method's steps set no
        limit, API RP 2A's alpha <= 1.0 is the API method's."""
        solid = "concrete_timber_corrugated"
        for su_ksf in (1.0, 1.2):          # the chart is above su here
            su = su_ksf * KSF
            ca = adhesion_fig_7_17(su, 40, solid)
            assert ca > su
            assert adhesion(su, 40, solid) == (ca, "Fig. 7-17")
        su = 2.0 * KSF                     # the chart is below su here
        assert adhesion(su, 40, solid) == (
            adhesion_fig_7_17(su, 40, solid), "Fig. 7-17")
        assert alpha_tomlinson(1.0 * KSF, "concrete", D_over_b=40) ==             pytest.approx(1.106)

    def test_api_alpha_equations(self):
        """GEC-12 Eqs. 7-13 to 7-15, at most 1.0."""
        assert api_alpha(50, 100) == pytest.approx(0.5 * 0.5 ** -0.5)
        assert api_alpha(200, 100) == pytest.approx(0.5 * 2.0 ** -0.25)
        assert api_alpha(10, 100) == 1.0

    def test_end_bearing_cohesive(self):
        """Qt = 9 * cu * At."""
        Qt = end_bearing_cohesive(50, 0.1)  # cu=50, At=0.1
        assert Qt == pytest.approx(45.0, rel=1e-6)

    def test_skin_friction_cohesive(self):
        """Qs = Ca * perimeter * thickness."""
        Qs = skin_friction_cohesive(60, 1.0, 5.0, D_over_b=25,
                                    curve="smooth_steel")
        ca = adhesion_fig_7_17(60, 25, "smooth_steel")
        assert Qs == pytest.approx(ca * 1.0 * 5.0, rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Nordlund Method (GEC-12 Sec. 7.2.1.3.1, Steps 2-9)
# ═══════════════════════════════════════════════════════════════════════

class TestNordlund:
    """The Nordlund charts, measured off the printed figures (2026-10-18)
    and Tables 7-6/7-7 as printed."""

    def test_fig_7_9_by_pile_type(self):
        assert delta_phi_ratio_fig_7_9("pipe_closed", 1.0) == pytest.approx(0.655)
        assert delta_phi_ratio_fig_7_9("timber", 1.0) == pytest.approx(0.694)
        assert delta_phi_ratio_fig_7_9("concrete", 1.0) == pytest.approx(0.767)
        assert delta_phi_ratio_fig_7_9("h_pile", 0.15) == pytest.approx(0.791)

    def test_fig_7_9_held_off_the_curve_with_a_note(self):
        notes = []
        r = delta_phi_ratio_fig_7_9("pipe_open", 0.1, notes)
        assert r == pytest.approx(0.344)
        assert len(notes) == 1 and "outside GEC-12 Figure 7-9" in notes[0]

    @pytest.mark.parametrize("phi, V, kd", [
        # Table 7-6 / 7-7 printed values (omega = 0)
        (25, 0.1, 0.70), (30, 1.0, 1.15), (35, 0.5, 1.57), (36, 0.3, 1.61),
        (40, 10.0, 4.30), (33, 4.0, 1.80), (38, 0.8, 2.40),
    ])
    def test_k_delta_reproduces_tables_7_6_and_7_7(self, phi, V, kd):
        assert k_delta(phi, V) == pytest.approx(kd, abs=0.011)

    def test_k_delta_log_linear_in_V(self):
        """Step 3d: log-linear between the chart volumes."""
        mid = k_delta(35, math.sqrt(0.1 * 1.0))
        assert mid == pytest.approx((1.15 + 1.75) / 2, rel=1e-9)

    def test_k_delta_held_outside_the_tables(self):
        notes = []
        assert k_delta(42, 1.0, notes) == pytest.approx(3.00)
        assert k_delta(36, 0.05, notes) == pytest.approx(1.26)
        assert len(notes) == 2

    def test_cf_from_fig_7_14(self):
        assert correction_factor(30, 1.0) == pytest.approx(1.0)
        assert correction_factor(30, 0.6) == pytest.approx(0.817)
        assert correction_factor(40, 0.2) == pytest.approx(0.370)
        assert correction_factor(35, 0.7) == pytest.approx((0.757 + 0.917) / 2)

    def test_alpha_t_and_nq_from_fig_7_16(self):
        assert alpha_t(30, 20) == pytest.approx(0.580)
        assert alpha_t(30, 45) == pytest.approx(0.491)
        assert alpha_t(30, 100) == pytest.approx(0.491)   # nearest curve
        assert alpha_t(40, 30) == pytest.approx(0.750)    # curves merged
        assert n_q_prime(30) == pytest.approx(29.6, rel=1e-6)
        assert n_q_prime(40) == pytest.approx(155.3, rel=1e-6)
        prev = 0
        for phi in [15, 20, 25, 30, 35, 40, 45]:
            assert n_q_prime(phi) > prev
            prev = n_q_prime(phi)

    def test_table_d6_unit_side_resistance(self):
        """GEC-12 Vol 3 Table D-6 (HP12x74, DrivenPiles) implies fs/sigma'd
        of 0.447 at phi 33 and 0.592 at phi 36; the charts give them to
        about 2 %."""
        V = displaced_volume(make_h_pile("HP12x74").area)
        r = delta_phi_ratio_fig_7_9("h_pile", V)
        for phi, published in ((33, 0.447), (36, 0.592)):
            f = nordlund_side_factors(phi, r, V)
            assert f["fs_ratio"] == pytest.approx(published, rel=0.03)

    def test_skin_friction_is_eq_7_8(self):
        Qs = skin_friction_cohesionless(30, 50, 1.0, 5.0, 0.7, 0.5)
        f = nordlund_side_factors(30, 0.7, 0.5)
        assert Qs == pytest.approx(
            f["K_delta"] * f["C_F"] * math.sin(math.radians(21)) * 50 * 5,
            rel=1e-9)

    def test_end_bearing_positive(self):
        """End bearing should be positive."""
        Qt = end_bearing_cohesionless(30, 100, 0.1, 15, 0.3)
        assert Qt > 0


class TestToeStressLimit:
    """GEC-12 Step 8: 'The limiting value of sigma'p is 3 ksf'."""

    def _toe(self, sigma_v, **kw):
        d = {}
        Q = end_bearing_cohesionless(40, sigma_v, 0.1, 15, 0.3,
                                     notes=[], details=d, **kw)
        return Q, d

    def test_below_at_and_above_the_limit(self):
        lim = 3.0 * KSF
        below, d = self._toe(0.5 * lim)
        assert d["sigma_p_kPa"] == pytest.approx(0.5 * lim)
        at, d = self._toe(lim)
        assert d["sigma_p_kPa"] == pytest.approx(lim)
        above, d = self._toe(2 * lim)
        assert d["sigma_p_kPa"] == pytest.approx(lim)
        assert above == pytest.approx(at, rel=1e-12)
        assert below < at

    def test_no_limit_when_none(self):
        lim = 3.0 * KSF
        _, d = self._toe(2 * lim, sigma_p_limit=None)
        assert d["sigma_p_kPa"] == pytest.approx(2 * lim)

    def test_analysis_applies_it_and_says_so(self):
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(30, "cohesionless", 19.0, friction_angle=38)])
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=25).compute()
        assert r.sigma_v_tip == pytest.approx(475.0)
        assert r.toe_details["sigma_p_kPa"] == pytest.approx(3.0 * KSF)
        r0 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=25,
                               toe_stress_limit=None).compute()
        assert r0.toe_details["sigma_p_kPa"] == pytest.approx(475.0)
        assert r0.Q_tip > r.Q_tip


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Beta Method
# ═══════════════════════════════════════════════════════════════════════

class TestBetaMethod:
    """Test effective stress (beta) method."""

    def test_beta_NC_sand(self):
        """NC sand phi=30: beta = (1-sin30)*tan30 = 0.5*0.577 = 0.289."""
        beta = beta_from_phi(30, OCR=1.0)
        expected = (1 - math.sin(math.radians(30))) * math.tan(math.radians(30))
        assert beta == pytest.approx(expected, rel=1e-4)

    def test_beta_OC_higher(self):
        """OC soil should have higher beta than NC (Burland OCR enhancement)."""
        beta_nc = beta_from_phi(30, OCR=1.0, method="burland")
        beta_oc = beta_from_phi(30, OCR=4.0, method="burland")
        assert beta_oc > beta_nc

    def test_beta_fellenius_ignores_OCR_burland_does_not(self):
        """fellenius (NC) and burland (OCR-aware) must differ for OC soils."""
        beta_f_nc = beta_from_phi(25, OCR=1.0, method="fellenius")
        with pytest.warns(UserWarning):
            beta_f_oc = beta_from_phi(25, OCR=4.0, method="fellenius")
        beta_b_oc = beta_from_phi(25, OCR=4.0, method="burland")
        assert beta_f_oc == pytest.approx(beta_f_nc)  # fellenius ignores OCR
        assert beta_b_oc > beta_f_oc                   # burland enhances with OCR

    def test_Nt_increases_with_phi(self):
        """Nt should increase with phi."""
        prev = 0
        for phi in [0, 10, 20, 30, 40]:
            Nt = Nt_from_phi(phi)
            assert Nt >= prev
            prev = Nt

    def test_skin_friction_beta(self):
        """Qs = beta * sigma_v * perimeter * thickness."""
        Qs = skin_friction_beta(50, 0.3, 1.0, 5.0)
        assert Qs == pytest.approx(0.3 * 50 * 1.0 * 5.0, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Combined Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestCombinedAnalysis:
    """Test the combined AxialPileAnalysis."""

    def _make_standard_problem(self):
        """Standard test problem: pipe pile in sand + clay."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(10, 'cohesive', 17.0, cohesion=50),
        ], gwt_depth=3.0)
        return pile, soil

    def test_auto_method(self):
        """Auto method selects Nordlund for sand, Tomlinson for clay."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0)
        result = analysis.compute()

        assert result.Q_ultimate > 0
        assert result.Q_skin > 0
        assert result.Q_tip > 0
        assert result.Q_ultimate == pytest.approx(
            result.Q_skin + result.Q_tip, rel=1e-6
        )
        assert result.Q_allowable == pytest.approx(
            result.Q_ultimate / 2.5, rel=1e-6
        )

        # Check layer methods
        methods = {l['method'] for l in result.layer_breakdown}
        assert 'nordlund' in methods
        assert 'tomlinson' in methods

    def test_beta_method(self):
        """Beta method should produce positive capacity."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0, method="beta"
        )
        result = analysis.compute()
        assert result.Q_ultimate > 0

    def test_uplift_capacity(self):
        """Uplift capacity < ultimate (compression) capacity."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0, include_uplift=True
        )
        result = analysis.compute()
        assert result.Q_uplift is not None
        assert result.Q_uplift < result.Q_ultimate

    def test_longer_pile_more_capacity(self):
        """Longer pile in uniform sand should have more capacity."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 18.0, friction_angle=30),
        ], gwt_depth=3.0)
        r10 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=10.0).compute()
        r15 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0).compute()
        assert r15.Q_skin > r10.Q_skin  # more skin friction
        assert r15.Q_ultimate > r10.Q_ultimate

    def test_h_pile_analysis(self):
        """HP shape pile analysis should work."""
        pile = make_h_pile("HP12x74")
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 19.0, friction_angle=35),
        ])
        result = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0
        ).compute()
        assert result.Q_ultimate > 0

    def test_summary_string(self):
        """summary() should return a formatted string."""
        pile, soil = self._make_standard_problem()
        result = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15).compute()
        s = result.summary()
        assert "AXIAL PILE" in s
        assert "kN" in s

    def test_to_dict(self):
        """to_dict should return key fields."""
        pile, soil = self._make_standard_problem()
        result = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15).compute()
        d = result.to_dict()
        assert "Q_ultimate_kN" in d
        assert d["Q_ultimate_kN"] > 0

    def test_capacity_vs_depth(self):
        """capacity_vs_depth should return increasing capacity."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0)
        profile = analysis.capacity_vs_depth(depth_min=5, depth_max=15, n_points=5)
        assert len(profile) == 5
        # Capacity should generally increase with depth
        assert profile[-1]["Q_ultimate_kN"] >= profile[0]["Q_ultimate_kN"]


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: CRITICAL-2 — Limiting tip resistance matches GEC-12 Fig 7-15
# ═══════════════════════════════════════════════════════════════════════

class TestLimitingTipResistance:
    """Verify limiting tip resistance from GEC-12 Figure 7-15 (Meyerhof 1976).

    History of these pins.
    - 2026-07-19: the earlier table was about 10x unconservative at phi = 30.
    - 2026-10-08: re-measured off the printed figure (GEC 12 Vol 1, pdf page
      index 283). x was fitted to the 1-degree gridlines and y to the 25-tsf
      gridlines; a second pixel read and two vision models agree within
      about 1 tsf.
      - The 2026-07-19 table read 10 tsf at 30 deg where the figure reads
        7.1 (+41 %).
      - It also carried nodes at 26, 28, 44 and 45 deg that are not on the
        chart. The axis starts at 30 deg and the curve ends at 43.75 deg.

    Outside the chart the end value is used, with a warning (owner's rule,
    2026-10-08)."""

    @pytest.mark.parametrize("phi, ql_tsf", [
        (30, 7.1), (32, 16.0), (34, 35.9), (36, 75.2), (38, 133.6),
        (40, 208.5), (42, 296.0), (43, 339.2), (43.75, 368.0),
    ])
    def test_measured_nodes(self, phi, ql_tsf):
        assert _limiting_tip_resistance(phi) == pytest.approx(
            ql_tsf * 95.76, rel=1e-6)

    def test_interpolation_between_nodes(self):
        assert _limiting_tip_resistance(35.5) == pytest.approx(
            (53.7 + 75.2) / 2 * 95.76, rel=1e-6)

    def test_below_chart_held_at_the_30_deg_end_value(self):
        for phi in (20, 26, 29.9):
            assert _limiting_tip_resistance(phi) == pytest.approx(
                7.1 * 95.76, rel=1e-6)

    def test_above_chart_held_at_the_curve_end_value(self):
        for phi in (43.8, 45, 47):
            assert _limiting_tip_resistance(phi) == pytest.approx(
                368.0 * 95.76, rel=1e-6)

    def test_end_bearing_respects_limit(self):
        """End bearing should be capped by the limiting tip resistance."""
        # Very high sigma_v to force qt > qt_limit(phi=30) = 7.1 tsf
        Qt = end_bearing_cohesionless(30, 5000, 0.1, 20, 0.3)
        assert Qt == pytest.approx(7.1 * 95.76 * 0.1, rel=0.01)


class TestToeLimitOutsideTheChart:
    """A toe phi outside Figure 7-15 completes, and the user is told."""

    def test_no_note_inside_the_chart(self):
        assert toe_limit_chart_note(30) is None
        assert toe_limit_chart_note(43.75) is None

    def test_note_below_the_chart_says_the_limit_is_lower(self):
        note = toe_limit_chart_note(28, governed=True)
        assert "below GEC-12 Figure 7-15" in note
        assert "7.1 tsf" in note and "LOWER" in note
        assert "governed" in note

    def test_note_above_the_chart_names_the_curve_end(self):
        note = toe_limit_chart_note(45)
        assert "above GEC-12 Figure 7-15" in note and "368 tsf" in note

    def test_end_bearing_warns_when_no_list_is_given(self):
        with pytest.warns(UserWarning, match="below GEC-12 Figure 7-15"):
            end_bearing_cohesionless(28, 100, 0.1, 15, 0.3)

    def test_end_bearing_appends_once_to_a_given_list(self):
        notes = []
        end_bearing_cohesionless(28, 100, 0.1, 15, 0.3, notes=notes)
        end_bearing_cohesionless(28, 100, 0.2, 15, 0.3, notes=notes)
        assert len(notes) == 1

    def _analysis(self, toe_phi):
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(thickness=20.0, soil_type="cohesionless",
                           unit_weight=19.0, friction_angle=toe_phi),
        ])
        return AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=soil,
                                 pile_length=15.0, method="auto")

    def test_result_carries_the_warning_to_the_user(self):
        r = self._analysis(28).compute()
        assert len(r.warnings) == 1
        assert "below GEC-12 Figure 7-15" in r.warnings[0]
        assert r.to_dict()["warnings"] == r.warnings
        assert "Warnings:" in r.summary()

    def test_no_warning_inside_the_chart(self):
        r = self._analysis(35).compute()
        assert r.warnings == []
        assert "warnings" not in r.to_dict()


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: Open-ended pipe piles (GEC-12 Sec. 7.10.7)
# ═══════════════════════════════════════════════════════════════════════

def _pipe_cases(r):
    """(plugged, unplugged) totals the result's choice reports."""
    import re
    c = {c["what"]: c for c in r.choices}["open-ended pipe"]
    m = re.search(r"plugged ([\d.]+) kN, unplugged ([\d.]+) kN", c["basis"])
    return float(m.group(1)), float(m.group(2))


class TestOpenEndedPlugging:
    """The lesser of Eqs. 7-88 (plugged) and 7-89 (unplugged), with the
    interior side resistance a fraction of the exterior and the soil plug
    weight Wp subtracted; Tomlinson's 105 ksf limit in sand; Eq. 7-90 for
    a pipe under 24 in in predominantly cohesive soil."""

    def _sand(self):
        return AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 18.0, friction_angle=32),
        ], gwt_depth=3.0)

    def _open(self, d=0.6096, t=0.0127):
        return make_pipe_pile(d, t, closed_end=False)

    def test_open_pile_has_plugging_fields(self):
        """Open-ended pile should have tip_area_plugged and inner_perimeter set."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=False)
        assert pile.tip_area_plugged is not None
        assert pile.inner_perimeter is not None
        assert pile.closed_end is False
        r_o = 0.3
        assert pile.tip_area_plugged == pytest.approx(math.pi * r_o**2, rel=1e-4)
        inner_d = 0.6 - 2 * 0.012
        assert pile.inner_perimeter == pytest.approx(math.pi * inner_d, rel=1e-4)

    def test_closed_pile_no_plugging_fields(self):
        """Closed-end pile should have None for plugging fields."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=True)
        assert pile.tip_area_plugged is None
        assert pile.inner_perimeter is None
        assert pile.closed_end is True

    def test_lesser_of_the_two_cases_governs(self):
        r = AxialPileAnalysis(pile=self._open(), soil=self._sand(),
                              pile_length=15.0).compute()
        plugged, unplugged = _pipe_cases(r)
        assert r.Q_ultimate == pytest.approx(min(plugged, unplugged), abs=0.06)
        assert r.Q_ultimate == pytest.approx(
            r.Q_skin + r.Q_tip - r.plug_weight, rel=1e-9)

    def test_unplugged_is_eq_7_89(self):
        """Rn = fso Aso + fsi Asi + qp Ap - Wp, fsi = ratio x fso."""
        a = AxialPileAnalysis(pile=self._open(), soil=self._sand(),
                              pile_length=15.0, inside_friction_ratio=0.5)
        r = a.compute()
        _, unplugged = _pipe_cases(r)
        outside = sum(lb["skin_friction_kN"] for lb in r.layer_breakdown)
        V = displaced_volume(a.pile.tip_area_plugged)
        inside = 0.5 * a._compute_inside_skin_friction(V, [])
        assert inside == pytest.approx(
            0.5 * outside * a.pile.inner_perimeter / a.pile.perimeter,
            rel=1e-9)
        d = {}
        qp_ann = end_bearing_cohesionless(
            32, r.sigma_v_tip, a.pile.tip_area, 15.0, a.pile.width,
            notes=[], q_limit=a.open_pipe_toe_limit, details=d)
        Wp = (a.pile.tip_area_plugged - a.pile.tip_area) * r.sigma_v_tip
        assert unplugged == pytest.approx(outside + inside + qp_ann - Wp,
                                          abs=0.06)

    def test_inside_ratio_scales_only_the_unplugged_case(self):
        runs = {k: AxialPileAnalysis(pile=self._open(), soil=self._sand(),
                                     pile_length=15.0,
                                     inside_friction_ratio=k).compute()
                for k in (1 / 3, 0.5, 1.0)}
        cases = {k: _pipe_cases(r) for k, r in runs.items()}
        assert cases[1 / 3][0] == cases[0.5][0] == cases[1.0][0]
        assert cases[1 / 3][1] < cases[0.5][1] < cases[1.0][1]
        with pytest.raises(ValueError, match="inside_friction_ratio"):
            AxialPileAnalysis(pile=self._open(), soil=self._sand(),
                              pile_length=15.0, inside_friction_ratio=1.5)

    def test_plug_weight_is_the_soil_inside(self):
        a = AxialPileAnalysis(pile=self._open(), soil=self._sand(),
                              pile_length=15.0, head_depth=2.0)
        inner = a.pile.tip_area_plugged - a.pile.tip_area
        stress = (a.soil.effective_stress_at_depth(17.0)
                  - a.soil.effective_stress_at_depth(2.0))
        assert a._plug_weight() == pytest.approx(inner * stress, rel=1e-12)

    def test_105_ksf_limit_below_at_and_above(self):
        """Tomlinson's limiting unit toe resistance of an open pipe in
        cohesionless soil (GEC-12 p. 407) caps q_p only above it."""
        lim = 105 * KSF
        d = {}
        end_bearing_cohesionless(40, 200, 1.0, 15, 0.6, notes=[],
                                 q_limit=lim, details=d)
        assert d["q_p_kPa"] == pytest.approx(lim)          # above: capped
        assert d["governs"] == "open-end pipe limit"
        d = {}
        end_bearing_cohesionless(32, 100, 1.0, 15, 0.6, notes=[],
                                 q_limit=lim, details=d)
        assert d["q_p_kPa"] < lim                           # below: free
        assert d["governs"] != "open-end pipe limit"
        q_free = d["q_p_kPa"]
        d = {}
        end_bearing_cohesionless(32, 100, 1.0, 15, 0.6, notes=[],
                                 q_limit=q_free, details=d)
        assert d["q_p_kPa"] == pytest.approx(q_free)        # at the limit

    def test_105_ksf_limit_applies_to_open_pipes_only(self):
        dense = AxialSoilProfile(layers=[
            AxialSoilLayer(30, 'cohesionless', 20.0, friction_angle=40)],
            gwt_depth=None)
        r = AxialPileAnalysis(pile=self._open(), soil=dense,
                              pile_length=20.0).compute()
        assert r.toe_details["q_p_kPa"] == pytest.approx(105 * KSF)
        r0 = AxialPileAnalysis(pile=self._open(), soil=dense,
                               pile_length=20.0,
                               open_pipe_toe_limit=None).compute()
        assert r0.toe_details["q_p_kPa"] > 105 * KSF
        closed = AxialPileAnalysis(
            pile=make_pipe_pile(0.6096, 0.0127, closed_end=True), soil=dense,
            pile_length=20.0).compute()
        assert closed.toe_details["q_p_kPa"] > 105 * KSF

    def test_eq_7_90_for_a_small_pipe_in_clay(self):
        """GEC-12 p. 407: a pipe under 24 in in predominantly cohesive soil:
        Rn = 0.8 Ca As + 4.5 su Ap."""
        clay = AxialSoilProfile(layers=[
            AxialSoilLayer(3, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(25, 'cohesive', 18.0, cohesion=80)],
            gwt_depth=2.0)
        pile = make_pipe_pile(0.4064, 0.0127, closed_end=False)
        r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20).compute()
        pw = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                               open_pipe_method="paikowsky_whitman").compute()
        sand_pw, clay_pw = (lb["skin_friction_kN"] for lb in pw.layer_breakdown)
        sand, clay_q = (lb["skin_friction_kN"] for lb in r.layer_breakdown)
        assert sand == pytest.approx(sand_pw)
        assert clay_q == pytest.approx(0.8 * clay_pw)
        assert r.Q_tip == pytest.approx(4.5 * 80 * pile.tip_area_plugged)
        assert r.Q_ultimate == pytest.approx(r.Q_skin + r.Q_tip)
        assert {c["what"]: c for c in r.choices}["open-ended pipe"][
            "chosen"] == "Tomlinson's Eq. 7-90"

    def test_eq_7_90_not_for_24_in_or_sand_toe(self):
        clay = AxialSoilProfile(layers=[
            AxialSoilLayer(25, 'cohesive', 18.0, cohesion=80),
            AxialSoilLayer(10, 'cohesionless', 19.0, friction_angle=34)],
            gwt_depth=2.0)
        big = make_pipe_pile(0.6096, 0.0127, closed_end=False)
        small = make_pipe_pile(0.4064, 0.0127, closed_end=False)
        for pile, L in ((big, 20), (small, 28)):
            r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=L).compute()
            assert r.toe_details["governs"] != "Eq. 7-90"
        with pytest.raises(ValueError, match="Eq. 7-90"):
            AxialPileAnalysis(pile=small, soil=clay, pile_length=28,
                              open_pipe_method="tomlinson").compute()

    def test_closed_end_behavior_unchanged(self):
        """Closed-end pipe pile should not be affected by plugging logic."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(10, 'cohesive', 17.0, cohesion=50),
        ], gwt_depth=3.0)
        result = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0
        ).compute()
        assert result.plug_weight == 0.0
        assert result.Q_ultimate == pytest.approx(
            result.Q_skin + result.Q_tip, rel=1e-6
        )
        assert result.Q_allowable == pytest.approx(
            result.Q_ultimate / 2.5, rel=1e-6
        )


class TestUplift:
    """GEC-12 Sec. 7.2.3.1 (Eq. 7-38): the nominal shaft resistance."""

    def _soil(self):
        return AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(10, 'cohesive', 17.0, cohesion=50),
        ], gwt_depth=3.0)

    def test_default_is_the_whole_outside_side_resistance(self):
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        r = AxialPileAnalysis(pile=pile, soil=self._soil(), pile_length=15,
                              include_uplift=True).compute()
        assert r.Q_uplift == pytest.approx(r.Q_skin, rel=1e-12)
        r7 = AxialPileAnalysis(pile=pile, soil=self._soil(), pile_length=15,
                               include_uplift=True,
                               uplift_skin_fraction=0.7).compute()
        assert r7.Q_uplift == pytest.approx(0.7 * r.Q_skin, rel=1e-12)

    def test_eq_7_90_uplift_is_its_reduced_shaft(self):
        """Under Eq. 7-90 the shaft is 0.8 Ca As; uplift takes that, not the
        unreduced side resistance."""
        clay = AxialSoilProfile(layers=[
            AxialSoilLayer(3, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(25, 'cohesive', 18.0, cohesion=80)],
            gwt_depth=2.0)
        pile = make_pipe_pile(0.4064, 0.0127, closed_end=False)
        r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                              include_uplift=True).compute()
        assert r.toe_details["governs"] == "Eq. 7-90"
        assert r.Q_uplift == pytest.approx(r.Q_skin, rel=1e-12)
        pw = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                               include_uplift=True,
                               open_pipe_method="paikowsky_whitman").compute()
        assert r.Q_uplift < pw.Q_uplift


class TestPileWidthNotes:
    """GEC-12 p. 232: piles over 18 in were not in the correlation
    database; Nordlund tends to over predict over 24 in."""

    def _soil(self):
        return AxialSoilProfile(layers=[
            AxialSoilLayer(30, 'cohesionless', 19.0, friction_angle=34)],
            gwt_depth=2.0)

    def test_notes_by_width(self):
        for d, n in ((0.4064, 0), (0.508, 1), (0.762, 2)):
            pile = make_pipe_pile(d, 0.0127, closed_end=True)
            r = AxialPileAnalysis(pile=pile, soil=self._soil(),
                                  pile_length=20).compute()
            got = [w for w in r.warnings if "p. 232" in w]
            assert len(got) == n, (d, r.warnings)


class TestBetaOptions:
    """GEC-12 Sec. 7.2.1.3.4: beta and Nt from local experience first;
    a low Nt in clays (default 3)."""

    def test_cohesive_toe_nt(self):
        clay = AxialSoilProfile(layers=[
            AxialSoilLayer(30, 'cohesive', 18.0, cohesion=80)], gwt_depth=1)
        pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
        r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                              method="beta").compute()
        assert r.Q_tip == pytest.approx(3.0 * r.sigma_v_tip * pile.tip_area)
        r9 = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                               method="beta", cohesive_Nt=9).compute()
        assert r9.Q_tip == pytest.approx(3 * r.Q_tip)

    def test_layer_beta_and_nt_win(self):
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(30, 'cohesionless', 19.0, friction_angle=34,
                           beta=0.5, Nt=60)], gwt_depth=1)
        pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
        r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                              method="beta").compute()
        assert r.layer_breakdown[0]["beta"] == 0.5
        assert r.toe_details["N_t"] == 60
        assert r.Q_tip == pytest.approx(60 * r.sigma_v_tip * pile.tip_area)


class TestHPileToeArea:
    def test_box_or_steel(self):
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(30, 'cohesive', 18.0, cohesion=100)])
        pile = make_h_pile("HP12x74")
        box = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20).compute()
        steel = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                                  h_pile_toe_area="steel").compute()
        assert box.Q_tip == pytest.approx(900 * pile.tip_area)
        assert steel.Q_tip == pytest.approx(900 * pile.area)
        assert box.Q_skin == steel.Q_skin


def test_Nt_from_phi_consistent_with_gec12_table_7_9():
    """Provenance pin (audit): Nt_from_phi is a piecewise Nt(phi) INTERPOLATION,
    not a literal copy of GEC-12 Table 7-9 (which gives Nt RANGES by soil type).
    Verified 2026-07-12 against the in-house GEC-12 reference
    (geotech_references/gec_12/tables.py _TABLE_7_9): clay phi25-30 -> Nt 3-30,
    silt phi28-34 -> Nt 20-40, sand phi32-40 -> Nt 30-150, gravel phi35-45 ->
    Nt 60-300. The curve stays WITHIN those ranges in the overlapping phi bands;
    this pins the transcription so drift is caught. (The same curve is duplicated
    in downdrag/analysis._Nt_from_phi.)"""
    from axial_pile.beta_method import Nt_from_phi
    assert 20 <= Nt_from_phi(28) <= 40      # silt band
    assert 30 <= Nt_from_phi(33) <= 150     # sand band
    assert 30 <= Nt_from_phi(38) <= 150     # sand upper band
    assert Nt_from_phi(25) <= 30            # within clay range top
    # above the tabulated sand max phi it enters gravel territory (<=300)
    assert 60 <= Nt_from_phi(40) <= 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
