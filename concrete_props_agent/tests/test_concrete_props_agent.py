"""Tests for concrete_props_agent — anchored to the ACI rectangular-block
hand calculation (see DESIGN.md)."""

import math

import pytest

pytest.importorskip("concreteproperties")

from concrete_props_agent import analyze_rc_rectangle, has_concreteproperties
from concrete_props_agent.rc_section import aci_beta1


class TestBeta1:
    def test_aci_table_values(self):
        # ACI 318-19 Table 22.2.2.4.3
        assert aci_beta1(28) == pytest.approx(0.85)
        assert aci_beta1(35) == pytest.approx(0.80)
        assert aci_beta1(55) == pytest.approx(0.65)
        assert aci_beta1(80) == pytest.approx(0.65)   # floor


class TestSinglyReinforcedAnchor:
    """300x550, 3-28 fy=500 f'c=32, clear cover 48 (bar centre 62, d=488).

    Hand calc (ACI rectangular block):
        As = 3 * pi*28^2/4 = 1847.3 mm^2
        a  = As*fy / (0.85*f'c*b) = 113.2 mm
        Mn = As*fy*(d - a/2) = 398.7 kN*m
    """

    @pytest.fixture(scope="class")
    def res(self):
        return analyze_rc_rectangle(
            b_mm=300, h_mm=550, fc_MPa=32, fy_MPa=500,
            n_bot=3, dia_bot_mm=28, cover_mm=48)

    def test_steel_area_and_depth(self, res):
        assert res.as_bot_mm2 == pytest.approx(3 * math.pi * 28**2 / 4,
                                               rel=1e-9)
        assert res.d_eff_mm == pytest.approx(488, abs=0.5)

    def test_nominal_moment_vs_hand_calc(self, res):
        As = 3 * math.pi * 28**2 / 4
        a = As * 500 / (0.85 * 32 * 300)
        mn_hand = As * 500 * (488 - a / 2) / 1e6
        assert res.mn_pos_kNm == pytest.approx(mn_hand, rel=0.02)

    def test_cracked_less_than_gross(self, res):
        assert res.ixx_cracked_mm4 is not None
        assert 0 < res.ixx_cracked_mm4 < res.ixx_gross_mm4
        # gross close to bh^3/12 (steel transform adds a little)
        assert res.ixx_gross_mm4 == pytest.approx(300 * 550**3 / 12, rel=0.20)

    def test_cracking_moment_order(self, res):
        # M_cr ~ fr * I_g / y_t = 0.62*sqrt(32) * Ig / (h/2) ~ 53 kN*m
        fr = 0.62 * math.sqrt(32)
        m_cr_est = fr * (300 * 550**3 / 12) / (550 / 2) / 1e6
        assert res.m_cr_kNm == pytest.approx(m_cr_est, rel=0.30)

    def test_no_top_steel_no_hogging(self, res):
        assert res.mn_neg_kNm is None

    def test_summary_and_dict(self, res):
        assert "phi factors" in res.summary()
        d = res.to_dict()
        assert d["mn_pos_kNm"] > 0 and "note" in d


class TestDoublyReinforcedAndInteraction:
    def test_top_steel_gives_hogging(self):
        r = analyze_rc_rectangle(
            b_mm=300, h_mm=550, fc_MPa=32, fy_MPa=500,
            n_bot=3, dia_bot_mm=28, n_top=2, dia_top_mm=20, cover_mm=48)
        assert r.mn_neg_kNm is not None
        assert 0 < r.mn_neg_kNm < r.mn_pos_kNm    # less top steel

    def test_interaction_diagram(self):
        r = analyze_rc_rectangle(
            b_mm=300, h_mm=400, fc_MPa=32, fy_MPa=500,
            n_bot=2, dia_bot_mm=20, n_top=2, dia_top_mm=20, cover_mm=40,
            include_interaction=True, n_interaction_points=10)
        assert len(r.interaction) >= 10
        n_vals = [p[0] for p in r.interaction]
        # spans from tension to squash compression
        assert max(n_vals) > 0 > min(n_vals)


class TestValidation:
    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            analyze_rc_rectangle(b_mm=-300, h_mm=550, fc_MPa=32, fy_MPa=500,
                                 n_bot=3, dia_bot_mm=28)
        with pytest.raises(ValueError):
            analyze_rc_rectangle(b_mm=300, h_mm=550, fc_MPa=32, fy_MPa=500,
                                 n_bot=0, dia_bot_mm=28)
        with pytest.raises(ValueError, match="cover"):
            analyze_rc_rectangle(b_mm=300, h_mm=100, fc_MPa=32, fy_MPa=500,
                                 n_bot=2, dia_bot_mm=28, cover_mm=40)
        with pytest.raises(ValueError, match="dia_top"):
            analyze_rc_rectangle(b_mm=300, h_mm=550, fc_MPa=32, fy_MPa=500,
                                 n_bot=3, dia_bot_mm=28, n_top=2)

    def test_has_flag(self):
        assert has_concreteproperties()
