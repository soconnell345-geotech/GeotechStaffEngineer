"""Hand-check unit tests for the composite/transformed-section EI helper.

Every expected number is computed here from first principles (explicit
arithmetic in the test), NOT by re-calling the helper's internals — the
transformed section EI = Σ E_i·I_i about the composite neutral axis, with
reinforcing bars embedded in concrete added at the NET modulus
(E_bar − E_concrete) so the displaced concrete is not double-counted.

Units: m, kPa; EI in kN·m², EA in kN.
"""

import math

import pytest

from lateral_pile import (
    Pile, ReinforcedConcreteSection, rebar_diameter,
    CompositeSection, composite_section_ei, aci_concrete_modulus,
)


def _ec_aci(fc_kPa):
    """ACI 318 normalweight Ec (kPa) from f'c (kPa) — independent recompute."""
    return 4700.0 * math.sqrt(fc_kPa / 1000.0) * 1000.0


# ── (a) concrete/grout-filled steel pipe ────────────────────────────────
class TestFilledPipe:

    def test_v017_micropile_section_hand_check(self):
        """FHWA NHI-05-039 SP-2 micropile: casing OD=0.1969 m, wall=0.0151 m,
        grout f'c=27.6 MPa, casing E=199,947,980 kPa.

        Hand calc (steel annulus + grout core are non-overlapping → direct
        sum of rigidities, no (n-1) correction):
          r_o = 0.09845, r_i = 0.09845-0.0151 = 0.08335 m
          I_steel = pi/4 (r_o^4 - r_i^4) = 3.5876e-5 m^4
          I_core  = pi/4  r_i^4          = 3.7906e-5 m^4
          Ec = 4700*sqrt(27.6) = 24,691.8 MPa = 24,691,780 kPa
          EI = 199,947,980*3.5876e-5 + 24,691,780*3.7906e-5
             = 7173.3 + 936.0 = 8109 kN-m^2
        """
        E_steel = 199947980.0
        fc = 27600.0
        r_o, r_i = 0.1969 / 2.0, 0.1969 / 2.0 - 0.0151
        I_steel = math.pi / 4.0 * (r_o ** 4 - r_i ** 4)
        I_core = math.pi / 4.0 * r_i ** 4
        A_steel = math.pi * (r_o ** 2 - r_i ** 2)
        A_core = math.pi * r_i ** 2
        Ec = _ec_aci(fc)
        EI_hand = E_steel * I_steel + Ec * I_core
        EA_hand = E_steel * A_steel + Ec * A_core

        sec = composite_section_ei('filled_pipe', outer_diameter=0.1969,
                                   wall_thickness=0.0151, fc=fc,
                                   E_steel=E_steel)
        assert isinstance(sec, CompositeSection)
        assert sec.EI == pytest.approx(EI_hand, rel=1e-12)
        assert sec.EA == pytest.approx(EA_hand, rel=1e-12)
        assert sec.EI == pytest.approx(8109.0, abs=2.0)
        # transformed inertia at E_ref reproduces EI exactly
        assert sec.E_ref == E_steel
        assert sec.E_ref * sec.inertia_transformed == pytest.approx(sec.EI, rel=1e-12)
        # composite is stiffer than the casing-only section, ~1.13x
        casing_only = E_steel * I_steel
        assert sec.EI > casing_only
        assert sec.EI / casing_only == pytest.approx(1.131, abs=0.005)
        # symmetric section → neutral axis at the centre
        assert sec.neutral_axis == pytest.approx(0.0, abs=1e-12)

    def test_fc_and_direct_ec_agree(self):
        """Passing E_concrete directly must equal passing the fc that yields it."""
        Ec = _ec_aci(30000.0)
        by_fc = composite_section_ei('filled_pipe', outer_diameter=0.4,
                                     wall_thickness=0.02, fc=30000.0)
        by_ec = composite_section_ei('filled_pipe', outer_diameter=0.4,
                                     wall_thickness=0.02, E_concrete=Ec)
        assert by_fc.EI == pytest.approx(by_ec.EI, rel=1e-12)
        assert aci_concrete_modulus(30000.0) == pytest.approx(Ec, rel=1e-12)

    def test_into_pile_matches_EI(self):
        """Pile.from_composite_section carries the composite EI exactly."""
        sec = composite_section_ei('filled_pipe', outer_diameter=0.61,
                                   wall_thickness=0.0127, fc=35000.0)
        pile = Pile.from_composite_section(20.0, sec, diameter=0.61)
        assert pile.EI == pytest.approx(sec.EI, rel=1e-12)
        assert pile.diameter == 0.61


# ── (b) steel casing + grout + circular bar ring ────────────────────────
class TestCasedConcrete:

    def test_cased_with_bar_ring_hand_check(self):
        """Casing OD=0.4, wall=0.02; grout f'c=30 MPa; 8 #? bars d=0.025 on a
        0.30 m pitch circle. Es=Eb=200e6.

        Hand calc:
          r_o=0.2, r_i=0.18
          I_steel = pi/4(0.2^4-0.18^4);  I_core = pi/4 0.18^4
          Ec = 4700*sqrt(30) = 25,742,960 kPa
          ring: A_s = 8*pi/4*0.025^2; R=0.15; I_ring = A_s*R^2/2 + Σ own
          bars embedded in grout → net modulus (Eb - Ec)
          EI = Es*I_steel + Ec*I_core + (Eb-Ec)*I_ring
        """
        Es = Eb = 200e6
        fc = 30000.0
        r_o, r_i = 0.2, 0.18
        I_steel = math.pi / 4.0 * (r_o ** 4 - r_i ** 4)
        I_core = math.pi / 4.0 * r_i ** 4
        Ec = _ec_aci(fc)
        d_bar, n, R = 0.025, 8, 0.15
        A_s = n * math.pi / 4.0 * d_bar ** 2
        I_ring = A_s * R ** 2 / 2.0 + n * (math.pi / 4.0) * (d_bar / 2.0) ** 4
        EI_hand = Es * I_steel + Ec * I_core + (Eb - Ec) * I_ring

        sec = composite_section_ei('cased_concrete', outer_diameter=0.4,
                                   wall_thickness=0.02, fc=fc, n_bars=n,
                                   bar_diameter=d_bar, bar_circle_diameter=0.30)
        assert sec.EI == pytest.approx(EI_hand, rel=1e-12)
        assert sec.neutral_axis == pytest.approx(0.0, abs=1e-12)
        assert {c["name"] for c in sec.components} == {
            "steel_casing", "grout_core", "rebar(core)"}

    def test_bars_add_stiffness(self):
        """Adding a bar ring must raise EI above the bar-free cased section."""
        base = composite_section_ei('cased_concrete', outer_diameter=0.4,
                                    wall_thickness=0.02, fc=30000.0)
        reinf = composite_section_ei('cased_concrete', outer_diameter=0.4,
                                     wall_thickness=0.02, fc=30000.0, n_bars=8,
                                     bar_diameter=0.025, bar_circle_diameter=0.30)
        assert reinf.EI > base.EI
        # bar-free cased == filled_pipe of the same geometry
        pipe = composite_section_ei('filled_pipe', outer_diameter=0.4,
                                    wall_thickness=0.02, fc=30000.0)
        assert base.EI == pytest.approx(pipe.EI, rel=1e-12)


# ── (c) reinforced concrete ─────────────────────────────────────────────
class TestReinforcedConcrete:

    def test_circular_bar_free_matches_Ec_Ig(self):
        """A bar-free circular RC section is just Ec·Ig — cross-check against
        ReinforcedConcreteSection's own gross properties."""
        rc = ReinforcedConcreteSection(diameter=0.9, fc=35000.0,
                                       n_bars=12, bar_diameter=rebar_diameter("#8"),
                                       cover=0.075)
        sec = composite_section_ei('reinforced_concrete', diameter=0.9,
                                   fc=35000.0)
        assert sec.EI == pytest.approx(rc.Ec * rc.Ig, rel=1e-12)
        assert sec.inertia_gross == pytest.approx(rc.Ig, rel=1e-12)

    def test_circular_ring_hand_check(self):
        """Circular RC D=0.9, f'c=35 MPa, 12 #8 bars on a 0.75 m pitch circle.
          Ec = 4700*sqrt(35) = 27,806,970 kPa
          I_conc = pi/4 * 0.45^4
          A_s = 12*pi/4*0.0254^2; R=0.375; I_ring = A_s R^2/2 + own
          EI = Ec*I_conc + (Eb-Ec)*I_ring
        """
        fc, Eb = 35000.0, 200e6
        Ec = _ec_aci(fc)
        r = 0.45
        I_conc = math.pi / 4.0 * r ** 4
        d_bar = rebar_diameter("#8")
        n, R = 12, 0.375
        A_s = n * math.pi / 4.0 * d_bar ** 2
        I_ring = A_s * R ** 2 / 2.0 + n * (math.pi / 4.0) * (d_bar / 2.0) ** 4
        EI_hand = Ec * I_conc + (Eb - Ec) * I_ring

        sec = composite_section_ei('reinforced_concrete', diameter=0.9, fc=fc,
                                   n_bars=n, bar_diameter=d_bar,
                                   bar_circle_diameter=0.75)
        assert sec.EI == pytest.approx(EI_hand, rel=1e-12)
        # with steel present the reference modulus is the bar modulus
        assert sec.E_ref == Eb

    def test_rectangular_symmetric_layers_hand_check(self):
        """Rectangular RC b=0.4, h=0.6, f'c=30 MPa; 3 #8 top + 3 #8 bottom at
        y=±0.25 m. Symmetric → NA at centre.
          I_conc = 0.4*0.6^3/12 = 0.0072
          per layer A = 3*pi/4*0.0254^2; I_layer = A*0.25^2 (+ own)
          EI = Ec*I_conc + (Eb-Ec)*(I_top + I_bot)
        """
        fc, Eb = 30000.0, 200e6
        Ec = _ec_aci(fc)
        b, h = 0.4, 0.6
        I_conc = b * h ** 3 / 12.0
        d_bar = rebar_diameter("#8")
        A_layer = 3 * math.pi / 4.0 * d_bar ** 2
        I0_layer = 3 * (math.pi / 4.0) * (d_bar / 2.0) ** 4
        I_bars = 2 * (A_layer * 0.25 ** 2 + I0_layer)
        EI_hand = Ec * I_conc + (Eb - Ec) * I_bars

        sec = composite_section_ei('reinforced_concrete', width=b, height=h,
                                   fc=fc, bar_diameter=d_bar,
                                   bar_layers=[(3, 0.25), (3, -0.25)])
        assert sec.EI == pytest.approx(EI_hand, rel=1e-12)
        assert sec.neutral_axis == pytest.approx(0.0, abs=1e-12)
        assert sec.inertia_gross == pytest.approx(I_conc, rel=1e-12)

    def test_rectangular_asymmetric_neutral_axis_shift(self):
        """Asymmetric bars (bottom only) shift the transformed neutral axis.
        b=0.3, h=0.5, f'c=30 MPa; 4 #8 bars at y=+0.2 m (below centre).
          Ec = 25,742,960; concrete A=0.15, I=0.003125, y=0
          bar layer: A=4*pi/4*0.0254^2, E'=(Eb-Ec), y=+0.2
          EA = Ec*0.15 + E'*A
          y_na = E'*A*0.2 / EA
          EI = Ec*(I_c + A_c*y_na^2) + E'*(I0 + A*(0.2-y_na)^2)
        """
        fc, Eb = 30000.0, 200e6
        Ec = _ec_aci(fc)
        b, h = 0.3, 0.5
        A_c, I_c = b * h, b * h ** 3 / 12.0
        d_bar = rebar_diameter("#8")
        A_bar = math.pi / 4.0 * d_bar ** 2
        n, y_bar = 4, 0.2
        A_layer = n * A_bar
        I0_layer = n * (math.pi / 4.0) * (d_bar / 2.0) ** 4
        Ep = Eb - Ec
        EA_hand = Ec * A_c + Ep * A_layer
        y_na = Ep * A_layer * y_bar / EA_hand
        EI_hand = (Ec * (I_c + A_c * y_na ** 2)
                   + Ep * (I0_layer + A_layer * (y_bar - y_na) ** 2))

        sec = composite_section_ei('reinforced_concrete', width=b, height=h,
                                   fc=fc, bar_diameter=d_bar,
                                   bar_layers=[(n, y_bar)])
        assert sec.EA == pytest.approx(EA_hand, rel=1e-12)
        assert sec.neutral_axis == pytest.approx(y_na, rel=1e-9)
        assert sec.neutral_axis > 0.0          # NA pulled toward the bars
        assert sec.EI == pytest.approx(EI_hand, rel=1e-12)


# ── validation / error handling ─────────────────────────────────────────
class TestValidationAndErrors:

    def test_requires_concrete_modulus(self):
        with pytest.raises(ValueError, match="E_concrete.*or.*fc|fc.*or.*E_concrete"):
            composite_section_ei('filled_pipe', outer_diameter=0.4,
                                 wall_thickness=0.02)

    def test_wall_thickness_exceeds_radius(self):
        with pytest.raises(ValueError, match="wall_thickness"):
            composite_section_ei('filled_pipe', outer_diameter=0.4,
                                 wall_thickness=0.25, fc=30000.0)

    def test_unknown_section_type(self):
        with pytest.raises(ValueError, match="Unknown section_type"):
            composite_section_ei('t_section', fc=30000.0)

    def test_reinforced_concrete_needs_geometry(self):
        with pytest.raises(ValueError, match="diameter.*width|width.*height"):
            composite_section_ei('reinforced_concrete', fc=30000.0)

    def test_summary_and_to_dict(self):
        sec = composite_section_ei('filled_pipe', outer_diameter=0.4,
                                   wall_thickness=0.02, fc=30000.0)
        s = sec.summary()
        assert "Composite EI" in s and "UNCRACKED" in s
        d = sec.to_dict()
        assert d["EI_kNm2"] == pytest.approx(sec.EI, rel=1e-12)
        assert d["basis"] == "uncracked_transformed_section"
        assert sec.equivalent_I(sec.E_ref) == pytest.approx(sec.inertia_transformed, rel=1e-12)
