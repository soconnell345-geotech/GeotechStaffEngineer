"""Native RC mechanics: transformed, cracked, ultimate and interaction.

Every anchor here is an independent hand calculation from ACI 318-19, not a
library output — the point of the rewrite is that the numbers can be derived
on paper.
"""

import math

import pytest

from concrete_props_agent import aci_beta1, analyze_rc_rectangle
from concrete_props_agent import rc_native

# The reference beam: 300 x 550, 3-N28 bottom, f'c 32, fy 500, clear cover 48
# (bar centre 62 above the soffit, d = 488).
ANCHOR = dict(b_mm=300.0, h_mm=550.0, fc_MPa=32.0, fy_MPa=500.0,
              n_bot=3, dia_bot_mm=28.0, cover_mm=48.0)
AS_BOT = 3 * math.pi * 28.0 ** 2 / 4.0          # 1847.26 mm^2
EC = 4700.0 * math.sqrt(32.0)                   # 26587.2 MPa
N_MOD = 200e3 / EC                              # 7.5225


class TestElasticProperties:
    def test_default_modulus_is_aci(self):
        r = analyze_rc_rectangle(**ANCHOR)
        assert r.ec_MPa == pytest.approx(EC, rel=1e-12)

    def test_gross_area_is_the_concrete_rectangle(self):
        assert analyze_rc_rectangle(**ANCHOR).gross_area_mm2 == \
            pytest.approx(300.0 * 550.0, rel=1e-12)

    def test_transformed_gross_matches_hand_calc(self):
        """A_t = bh + (n-1)As, then the parallel-axis shift."""
        extra = (N_MOD - 1.0) * AS_BOT
        a_t = 300.0 * 550.0 + extra
        y_c = (300.0 * 550.0 * 275.0 + extra * 62.0) / a_t
        ixx = (300.0 * 550.0 ** 3 / 12.0
               + 300.0 * 550.0 * (275.0 - y_c) ** 2
               + extra * (y_c - 62.0) ** 2)
        assert analyze_rc_rectangle(**ANCHOR).ixx_gross_mm4 == \
            pytest.approx(ixx, rel=1e-9)

    def test_cracked_neutral_axis_and_inertia(self):
        """b*c^2/2 = n*As*(d-c), then Icr = b*c^3/3 + n*As*(d-c)^2."""
        n_as = N_MOD * AS_BOT
        # 150 c^2 + n_as c - n_as*488 = 0
        c = (-n_as + math.sqrt(n_as ** 2 + 4 * 150.0 * n_as * 488.0)) / (2 * 150.0)
        icr = 300.0 * c ** 3 / 3.0 + n_as * (488.0 - c) ** 2
        assert analyze_rc_rectangle(**ANCHOR).ixx_cracked_mm4 == \
            pytest.approx(icr, rel=2e-3)

    def test_cracked_is_stiffer_than_nothing_softer_than_gross(self):
        r = analyze_rc_rectangle(**ANCHOR)
        assert 0 < r.ixx_cracked_mm4 < r.ixx_gross_mm4

    def test_cracking_moment_is_fr_i_over_y(self):
        r = analyze_rc_rectangle(**ANCHOR)
        fr = 0.62 * math.sqrt(32.0)
        extra = (N_MOD - 1.0) * AS_BOT
        a_t = 300.0 * 550.0 + extra
        y_t = (300.0 * 550.0 * 275.0 + extra * 62.0) / a_t
        assert r.m_cr_kNm == pytest.approx(
            fr * r.ixx_gross_mm4 / y_t / 1e6, rel=1e-9)


class TestUltimateCapacity:
    def test_nominal_moment_matches_the_aci_hand_calc(self):
        """Mn = As*fy*(d - a/2) with a = As*fy/(0.85*f'c*b)."""
        a = AS_BOT * 500.0 / (0.85 * 32.0 * 300.0)
        mn_hand = AS_BOT * 500.0 * (488.0 - a / 2.0) / 1e6
        assert analyze_rc_rectangle(**ANCHOR).mn_pos_kNm == \
            pytest.approx(mn_hand, rel=2e-3)

    def test_no_top_steel_means_no_hogging_capacity(self):
        assert analyze_rc_rectangle(**ANCHOR).mn_neg_kNm is None

    def test_symmetric_reinforcement_is_symmetric_in_bending(self):
        """Identical top and bottom steel must give identical Mn both ways."""
        r = analyze_rc_rectangle(n_top=3, dia_top_mm=28.0, **ANCHOR)
        assert r.mn_neg_kNm == pytest.approx(r.mn_pos_kNm, rel=1e-9)

    def test_top_steel_adds_little_sagging_but_real_hogging(self):
        plain = analyze_rc_rectangle(**ANCHOR)
        doubly = analyze_rc_rectangle(n_top=2, dia_top_mm=16.0, **ANCHOR)
        assert doubly.mn_pos_kNm == pytest.approx(plain.mn_pos_kNm, rel=0.05)
        assert 0 < doubly.mn_neg_kNm < doubly.mn_pos_kNm

    def test_capacity_rises_with_steel(self):
        small = analyze_rc_rectangle(**{**ANCHOR, "n_bot": 2})
        assert small.mn_pos_kNm < analyze_rc_rectangle(**ANCHOR).mn_pos_kNm

    @pytest.mark.parametrize("fc,beta1", [
        (28.0, 0.85), (35.0, 0.80), (45.0, 0.7286), (55.0, 0.65), (80.0, 0.65),
    ])
    def test_beta1_follows_the_aci_table(self, fc, beta1):
        assert aci_beta1(fc) == pytest.approx(beta1, rel=1e-3)


class TestInteractionDiagram:
    @pytest.fixture(scope="class")
    def diagram(self):
        return analyze_rc_rectangle(include_interaction=True,
                                    n_interaction_points=24, **ANCHOR)

    def test_squash_point_matches_hand_calc(self, diagram):
        """P0 = 0.85*f'c*(Ag - As) + As*fy."""
        p0 = (0.85 * 32.0 * (300.0 * 550.0 - AS_BOT) + AS_BOT * 500.0) / 1e3
        assert max(p[0] for p in diagram.interaction) == pytest.approx(
            p0, rel=1e-9)

    def test_pure_tension_point_matches_hand_calc(self, diagram):
        n_t = -AS_BOT * 500.0 / 1e3
        assert min(p[0] for p in diagram.interaction) == pytest.approx(
            n_t, rel=1e-9)

    def test_pure_bending_point_is_the_reported_capacity(self, diagram):
        at_zero = [m for n, m in diagram.interaction if abs(n) < 1e-9]
        assert at_zero and at_zero[0] == pytest.approx(
            diagram.mn_pos_kNm, rel=1e-9)

    def test_peak_moment_exceeds_pure_bending(self, diagram):
        """Axial compression raises moment capacity up to the balanced point."""
        assert max(m for _, m in diagram.interaction) > diagram.mn_pos_kNm

    def test_ordered_and_spans_tension_to_compression(self, diagram):
        axial = [n for n, _ in diagram.interaction]
        assert axial == sorted(axial, reverse=True)
        assert axial[0] > 0 > axial[-1]
        assert len(diagram.interaction) >= 10

    def test_symmetric_section_has_no_moment_at_the_squash_point(self):
        r = analyze_rc_rectangle(n_top=3, dia_top_mm=28.0,
                                 include_interaction=True, **ANCHOR)
        n_max, m_at_max = max(r.interaction, key=lambda p: p[0])
        assert m_at_max == pytest.approx(0.0, abs=1e-6)

    def test_not_computed_unless_requested(self):
        assert analyze_rc_rectangle(**ANCHOR).interaction == []


class TestNativeHelpers:
    def test_layer_ordering_does_not_matter(self):
        layers = [rc_native.Layer(AS_BOT, 62.0), rc_native.Layer(402.0, 500.0)]
        a = rc_native.transformed_gross(300.0, 550.0, layers, N_MOD)
        b = rc_native.transformed_gross(300.0, 550.0, layers[::-1], N_MOD)
        assert a == pytest.approx(b, rel=1e-12)

    def test_modular_ratio_guards_zero_modulus(self):
        with pytest.raises(ValueError, match="positive"):
            rc_native.modular_ratio(200e3, 0.0)

    def test_unreinforced_section_is_rejected(self):
        with pytest.raises(ValueError, match="reinforcement"):
            rc_native.ultimate_capacity(300.0, 550.0, [], 32.0, 500.0)
