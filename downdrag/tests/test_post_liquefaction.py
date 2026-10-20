"""
Post-liquefaction downdrag (FHWA GEC-10 §10.7; BACKLOG GM-03).

No reference prints a worked example of post-liquefaction drag (GEC-10,
GEC-12 and UFC 3-220-20 have none; the NCHRP downdrag report is not in the
library), so the anchor is the HAND CHECK in DESIGN.md "Post-liquefaction
downdrag", worked from the printed equations: the numbers below are
computed here in closed form, not read off the module.

Run with: pytest downdrag/tests/test_post_liquefaction.py -v
"""

import math

import numpy as np
import pytest

from downdrag import (
    DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile,
    LIQUEFIED_SIDE_RESISTANCE, LiquefiedLayer, liquefied_layers_from_result,
    post_liquefaction_downdrag, residual_strength_kramer_wang,
)

GW = 9.81


def _hand_check(**kw):
    """DESIGN.md hand check HC-1: a 0.5 m square pile, 20 m long, 600 kN
    dead load, water table at the surface; sand 0-10 m (19 kN/m3, beta
    0.30) over dense sand (20 kN/m3, beta 0.45, Nt 30); 6-10 m liquefies
    with 3 % volumetric strain."""
    layers = [
        DowndragSoilLayer(thickness=6.0, soil_type="cohesionless",
                          unit_weight=19.0, phi=30, beta=0.30),
        DowndragSoilLayer(thickness=4.0, soil_type="cohesionless",
                          unit_weight=19.0, phi=30, beta=0.30),
        DowndragSoilLayer(thickness=20.0, soil_type="cohesionless",
                          unit_weight=20.0, phi=36, beta=0.45),
    ]
    return DowndragAnalysis(
        soil=DowndragSoilProfile(layers=layers, gwt_depth=0.0),
        pile_length=20.0, pile_diameter=0.5, pile_perimeter=2.0,
        pile_area=0.25, Q_dead=600.0, Nt=30.0, **kw)


_HC1_ZONE = [{"top": 6.0, "bottom": 10.0, "volumetric_strain": 0.03}]


def _by(result, name_start):
    return next(r for r in result.by_method
                if r["name"].startswith(name_start))


class _Hand:
    """HC-1 in closed form (DESIGN.md)."""
    P = 2.0
    k1 = 0.30 * (19.0 - GW)              # fs = k1 z above 10 m (kPa/m)
    s06 = P * k1 * 6.0 ** 2 / 2          # shaft 0-6 m (kN)
    s610 = P * k1 * (10.0 ** 2 - 6.0 ** 2) / 2   # shaft 6-10 m
    sv10 = (19.0 - GW) * 10.0            # sigma'v at 10 m
    g2 = 20.0 - GW
    R = 30.0 * (sv10 + g2 * 10.0) * 0.25          # toe, Nt sigma' A

    @classmethod
    def shaft_below(cls, u):
        """Shaft from 10 m to 10 + u (kN): fs = 0.45 (sv10 + g2 u)."""
        return cls.P * 0.45 * (cls.sv10 * u + cls.g2 * u * u / 2)

    @classmethod
    def fe_plane(cls, s_top):
        """The intersection of the load and resistance curves with the toe
        fully mobilized, given the shaft above 10 m: Q + s_top + S(u) =
        R + S(10) - S(u), a quadratic in u."""
        rhs = (cls.R + cls.shaft_below(10.0) - 600.0 - s_top) / 2
        a, b = cls.P * 0.45 * cls.g2 / 2, cls.P * 0.45 * cls.sv10
        u = (-b + math.sqrt(b * b + 4 * a * rhs)) / (2 * a)
        return 10.0 + u, 600.0 + s_top + cls.shaft_below(u)


class TestHandCheckHC1:
    """DESIGN.md HC-1: every number from the printed equations."""

    def test_the_static_numbers_are_the_static_analysis(self):
        a = _hand_check()
        plain = a.compute()
        r = post_liquefaction_downdrag(a, liquefied_layers=_HC1_ZONE)
        assert r.before.neutral_plane_depth == plain.neutral_plane_depth
        assert r.before.drag_force == plain.drag_force
        assert r.before.pile_settlement == plain.pile_settlement
        np.testing.assert_array_equal(r.before.z, plain.z)

    def test_unit_side_resistance_halved_in_the_liquefied_layer_only(self):
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        z, f0, f1 = r.after.z, r.before.unit_skin_friction, \
            r.after.unit_skin_friction
        np.testing.assert_array_equal(z, r.before.z)
        look = DowndragAnalysis._lookup_depths(z)
        inside = (look > 6.0) & (look < 10.0)
        assert inside.sum() > 2
        np.testing.assert_allclose(f1[inside], 0.5 * f0[inside])
        np.testing.assert_allclose(f1[~inside], f0[~inside])
        # the jumps at 6 m and 10 m sit on repeated nodes
        i6 = np.where(np.isclose(z, 6.0))[0]
        assert len(i6) == 2
        assert f1[i6[0]] == pytest.approx(_Hand.k1 * 6.0)
        assert f1[i6[1]] == pytest.approx(0.5 * _Hand.k1 * 6.0)

    def test_neutral_plane_at_the_base_of_the_liquefied_layer(self):
        """The methods that follow the settlement put the plane at 10 m,
        where the reconsolidation settlement ends: drag load = static
        friction above 6 m + half the pre-earthquake friction 6-10 m."""
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        hand = _Hand.s06 + 0.5 * _Hand.s610
        assert hand == pytest.approx(187.476, abs=1e-3)
        for name in ("Neutral plane at the bearing layer",
                     "Mobilized toe resistance compatible"):
            row = _by(r, name)
            assert row["applies_after"]
            assert row["neutral_plane_after_m"] == pytest.approx(10.0,
                                                                 abs=0.01)
            assert row["drag_force_after_kN"] == pytest.approx(hand,
                                                               abs=0.5)
        # before the earthquake nothing settles: no drag by these methods
        assert not _by(r, "Neutral plane at the bearing layer")[
            "applies_before"]
        assert _by(r, "Mobilized toe resistance compatible")[
            "drag_force_before_kN"] == pytest.approx(0.0, abs=1e-6)

    def test_force_equilibrium_before_and_after(self):
        z0, q0 = _Hand.fe_plane(_Hand.s06 + _Hand.s610)
        z1, q1 = _Hand.fe_plane(_Hand.s06 + 0.5 * _Hand.s610)
        assert (round(z0, 2), round(q0, 1)) == (17.85, 1807.4)
        assert (round(z1, 2), round(q1, 1)) == (18.13, 1763.3)
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        assert r.before.neutral_plane_depth == pytest.approx(z0, abs=0.01)
        assert r.before.max_pile_load == pytest.approx(q0, rel=1e-3)
        assert r.after.neutral_plane_depth == pytest.approx(z1, abs=0.01)
        assert r.after.max_pile_load == pytest.approx(q1, rel=1e-3)
        assert r.after.drag_force == pytest.approx(q1 - 600.0, rel=1e-3)

    def test_soil_settlement_is_the_reconsolidation_settlement(self):
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        assert r.reconsolidation_settlement == pytest.approx(0.12)
        z, s = r.after.z, r.after.soil_settlement_profile
        hand = np.clip((10.0 - z) / 4.0, 0.0, 1.0) * 0.12
        np.testing.assert_allclose(s, hand, atol=1e-9)
        assert not np.any(r.before.soil_settlement_profile > 0)

    def test_zero_and_residual_strength_options_side_by_side(self):
        r = post_liquefaction_downdrag(
            _hand_check(), liquefied_layers=[{**_HC1_ZONE[0], "su_r": 5.0}])
        rows = {x["option"]: x for x in r.side_resistance_comparison}
        assert [x["option"] for x in r.side_resistance_comparison] == [
            "half_pre_earthquake", "zero", "residual_strength"]
        assert rows["half_pre_earthquake"]["used"]
        z_zero, _ = _Hand.fe_plane(_Hand.s06)
        assert rows["zero"]["neutral_plane_depth_m"] == pytest.approx(
            z_zero, abs=0.01)
        # su,r = 5 kPa wherever the pre-earthquake friction is more
        s_res = _Hand.P * 5.0 * 4.0
        z_res, _ = _Hand.fe_plane(_Hand.s06 + s_res)
        assert rows["residual_strength"]["neutral_plane_depth_m"] == \
            pytest.approx(z_res, abs=0.01)

    def test_the_warnings_say_why_the_methods_differ(self):
        """1163 kN against 187 kN is the methods' definitions: force
        equilibrium counts the friction down to the curves' intersection,
        8.13 m of it in dense sand that does not settle; the methods that
        follow the settlement count it above 10 m only. And before the
        earthquake nothing settles, yet force equilibrium gives a drag
        force: the warnings say both plainly."""
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        before = next(w for w in r.warnings
                      if w.startswith("Before the earthquake no soil"))
        assert "Neutral plane at 100% toe mobilization 1207 kN at " \
            "17.85 m" in before
        assert "Compare before and after within one method" in before
        why = next(w for w in r.warnings
                   if w.startswith("Why the methods differ"))
        assert "(18.13 m)" in why and "(1163 kN)" in why
        assert "8.13 m of it below 10.00 m" in why
        assert "Mobilized toe resistance compatible with the toe " \
            "penetration gives 187 kN at 10.00 m" in why
        # no "before" note where the soil settles before the earthquake
        r2 = TestBelowTheStaticNeutralPlane._case()
        assert not any(w.startswith("Before the earthquake no soil")
                       for w in r2.warnings)


class TestResidualStrength:

    def test_gec10_eq_9_13_by_hand(self):
        """su,r = exp(-8.444 + 0.109 (N1)60 + 5.379 sigma'v0^0.1), sigma'v0
        and su,r in atmospheres (GEC-10 p. 9-43)."""
        s = 73.52 / 101.325
        hand = 101.325 * math.exp(-8.444 + 0.109 * 8 + 5.379 * s ** 0.1)
        assert residual_strength_kramer_wang(8, 73.52) == pytest.approx(hand)
        assert hand == pytest.approx(9.54, abs=0.01)

    def test_su_r_from_n160_and_capped_at_the_pre_earthquake_value(self):
        a = _hand_check(neutral_plane_method="bearing_layer_top")
        r = post_liquefaction_downdrag(
            a, liquefied_layers=[{**_HC1_ZONE[0], "N1_60": 25}],
            liquefied_side_resistance="residual_strength")
        z = r.after.z
        look = DowndragAnalysis._lookup_depths(z)
        inside = (look > 6.0) & (look < 10.0)
        sv = (19.0 - GW) * z[inside]
        su = np.array([residual_strength_kramer_wang(25, v) for v in sv])
        np.testing.assert_allclose(
            r.after.unit_skin_friction[inside],
            np.minimum(su, r.before.unit_skin_friction[inside]))
        assert np.any(su > r.before.unit_skin_friction[inside])
        assert any("su,r is more than the" in w
                   for w in r.warnings)
        # nothing settles before the earthquake, so the plane at the base
        # of the settling soil exists only after: before is 'auto'
        assert r.after.headline_method.startswith(
            "Neutral plane at the bearing layer")
        assert r.before.headline_method ==             "Neutral plane at 100% toe mobilization"
        assert any("finds no plane" in w for w in r.warnings)

    def test_residual_strength_needs_su_r_or_n160(self):
        with pytest.raises(ValueError, match="needs su_r or N1_60"):
            post_liquefaction_downdrag(
                _hand_check(), liquefied_layers=_HC1_ZONE,
                liquefied_side_resistance="residual_strength")
        # not chosen: listed as not evaluated, with the reason
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        row = next(x for x in r.side_resistance_comparison
                   if x["option"] == "residual_strength")
        assert not row["applies"] and "needs su_r or N1_60" in row["reason"]


class TestRatio:

    def test_ratio_half_reproduces_the_gec10_option(self):
        a = _hand_check()
        half = post_liquefaction_downdrag(a, liquefied_layers=_HC1_ZONE)
        ratio = post_liquefaction_downdrag(
            a, liquefied_layers=_HC1_ZONE, liquefied_side_resistance="ratio",
            liquefied_side_resistance_ratio=0.5)
        assert ratio.after.drag_force == pytest.approx(
            half.after.drag_force)
        assert "ratio" in [x["option"]
                           for x in ratio.side_resistance_comparison]

    def test_ratio_needs_its_value(self):
        with pytest.raises(ValueError, match="liquefied_side_resistance_ratio"):
            post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE,
                                       liquefied_side_resistance="ratio")
        with pytest.raises(ValueError, match="between 0 and 1"):
            post_liquefaction_downdrag(
                _hand_check(), liquefied_layers=_HC1_ZONE,
                liquefied_side_resistance="ratio",
                liquefied_side_resistance_ratio=1.5)


class TestZonesInsideALayer:

    def test_a_zone_that_is_not_a_layer_boundary_gets_its_own_nodes(self):
        r = post_liquefaction_downdrag(
            _hand_check(), liquefied_layers=[
                {"top": 7.0, "bottom": 8.5, "volumetric_strain": 0.02}])
        z = r.after.z
        for zb in (7.0, 8.5):
            assert np.isclose(z, zb).sum() == 2
        look = DowndragAnalysis._lookup_depths(z)
        inside = (look > 7.0) & (look < 8.5)
        np.testing.assert_allclose(
            r.after.unit_skin_friction[inside],
            0.5 * _Hand.k1 * z[inside])
        assert r.reconsolidation_settlement == pytest.approx(0.03)
        row = _by(r, "Neutral plane at the bearing layer")
        assert row["neutral_plane_after_m"] == pytest.approx(8.5, abs=0.01)
        hand = _Hand.P * _Hand.k1 * (7.0 ** 2 / 2
                                     + 0.5 * (8.5 ** 2 - 7.0 ** 2) / 2)
        assert row["drag_force_after_kN"] == pytest.approx(hand, abs=0.5)

    def test_a_zone_split_into_layers_is_the_same_zone(self):
        """GM-201: a node on the boundary between two liquefied layers of
        one zone kept its static side resistance, so a zone given as
        several layers (a CPT's bands, an SPT result's readings) gave a
        different neutral plane from the same zone given whole."""
        one = post_liquefaction_downdrag(_hand_check(),
                                         liquefied_layers=_HC1_ZONE)
        for n in (2, 16):
            dz = 4.0 / n
            split = post_liquefaction_downdrag(
                _hand_check(), liquefied_layers=[
                    {"top": 6.0 + k * dz, "bottom": 6.0 + (k + 1) * dz,
                     "volumetric_strain": 0.03} for k in range(n)])
            z = split.after.z
            look = DowndragAnalysis._lookup_depths(z)
            inside = (look > 6.0) & (look < 10.0)
            np.testing.assert_allclose(
                split.after.unit_skin_friction[inside],
                0.5 * _Hand.k1 * z[inside])
            z1, _ = _Hand.fe_plane(_Hand.s06 + 0.5 * _Hand.s610)
            assert split.after.neutral_plane_depth == pytest.approx(z1)
            assert split.after.neutral_plane_depth == pytest.approx(
                one.after.neutral_plane_depth)

    def test_settlement_given_instead_of_strain(self):
        r = post_liquefaction_downdrag(
            _hand_check(), liquefied_layers=[
                {"top": 6.0, "bottom": 10.0, "settlement": 0.12}])
        assert r.liquefied_zones[0]["volumetric_strain"] == pytest.approx(
            0.03)


class TestBelowTheStaticNeutralPlane:
    """GEC-12 §7.4.2.2: liquefaction below the neutral plane adds load."""

    @staticmethod
    def _case():
        layers = [
            DowndragSoilLayer(thickness=8, soil_type="cohesive",
                              unit_weight=17.0, cu=25, alpha=1.0,
                              settling=True, Cc=0.4, e0=1.1),
            DowndragSoilLayer(thickness=4, soil_type="cohesionless",
                              unit_weight=19.0, phi=32, beta=0.35),
            DowndragSoilLayer(thickness=4, soil_type="cohesionless",
                              unit_weight=19.0, phi=30, beta=0.30),
            DowndragSoilLayer(thickness=20, soil_type="cohesionless",
                              unit_weight=20.0, phi=36, beta=0.45,
                              E_s=80000),
        ]
        a = DowndragAnalysis(
            soil=DowndragSoilProfile(layers=layers, gwt_depth=1.0),
            pile_length=24.0, pile_diameter=0.6, Q_dead=900.0,
            fill_thickness=2.0, Nt=40.0, structural_capacity=4000.0)
        return post_liquefaction_downdrag(a, liquefied_layers=[
            {"top": 12.5, "bottom": 15.5, "volumetric_strain": 0.025}])

    def test_the_settlement_following_methods_carry_more_drag(self):
        r = self._case()
        row = _by(r, "Neutral plane at the bearing layer")
        assert row["neutral_plane_before_m"] == pytest.approx(8.0)
        assert row["neutral_plane_after_m"] == pytest.approx(15.5)
        assert row["drag_force_after_kN"] > row["drag_force_before_kN"] + 400
        zone = r.liquefied_zones[0]
        assert "Neutral plane at the bearing layer (end-bearing pile)" in \
            zone["below_the_plane_before_of"]

    def test_a_warning_names_them_when_the_headline_does_not_rise(self):
        r = self._case()
        assert r.after.drag_force <= r.before.drag_force
        w = next(w for w in r.warnings if "GEC-12 §7.4.2.2" in w)
        assert "will increase the axial compression load" in w
        assert "The methods whose drag force rises" in w
        assert "Neutral plane at the bearing layer" in w

    def test_zones_below_the_same_planes_share_one_sentence(self):
        # GM-207: a noisy CPT gives many zones below the same planes; the
        # warning names them once, not one sentence a zone
        r = post_liquefaction_downdrag(_hand_check(), liquefied_layers=[
            {"top": 3.0, "bottom": 4.0, "volumetric_strain": 0.02},
            {"top": 6.0, "bottom": 10.0, "volumetric_strain": 0.03},
            {"top": 14.0, "bottom": 15.0, "volumetric_strain": 0.02}])
        w = next(w for w in r.warnings if "GEC-12 §7.4.2.2" in w)
        first, second = (z["below_the_plane_before_of"]
                         for z in r.liquefied_zones[1:])
        assert r.liquefied_zones[0]["below_the_plane_before_of"] == first
        assert first != second
        assert "The liquefied zones 3-4, 6-10 m lie below" in w
        assert "The liquefied zone 14-15 m lies below" in w
        assert w.count(" below the neutral plane these methods give") == 2

    def test_fill_settlement_and_reconsolidation_add(self):
        r = self._case()
        z = r.after.z
        add = r.after.soil_settlement_profile - np.interp(
            z, r.before.z, r.before.soil_settlement_profile)
        hand = np.clip((15.5 - z) / 3.0, 0.0, 1.0) * 0.075
        np.testing.assert_allclose(add, hand, atol=1e-9)


class TestToeAndBelowTheToe:

    def test_a_toe_in_a_liquefied_layer_is_warned(self):
        r = post_liquefaction_downdrag(_hand_check(), liquefied_layers=[
            {"top": 18.0, "bottom": 22.0, "volumetric_strain": 0.02}])
        assert any("is in a liquefied layer" in w and "In no case should"
                   in w for w in r.warnings)

    def test_liquefied_soil_below_the_toe_takes_the_toe_down(self):
        a = _hand_check()
        r = post_liquefaction_downdrag(a, liquefied_layers=[
            {"top": 22.0, "bottom": 24.0, "volumetric_strain": 0.02}])
        assert r.reconsolidation_below_toe == pytest.approx(0.04)
        assert r.after.toe_settlement == pytest.approx(
            r.before.toe_settlement + 0.04, abs=1e-6)
        assert r.after.pile_settlement >= 0.04
        assert r.after.below_toe["reconsolidation_below_toe_mm"] == \
            pytest.approx(40.0)
        assert any("below the pile toe" in w for w in r.warnings)


class TestFromALiquefactionResult:

    @staticmethod
    def _spt(fos, strain=None):
        rows = []
        for i, (d, f) in enumerate(zip((3.0, 5.0, 7.0, 9.0, 11.0), fos)):
            row = {"depth_m": d, "N1_60": 10.0 + i, "FOS_liq": f,
                   "liquefiable": f < 1.0}
            if strain is not None:
                row["volumetric_strain"] = strain[i]
            rows.append(row)
        return {"method": "bi2014", "layer_results": rows}

    def test_each_reading_stands_for_halfway_to_its_neighbours(self):
        layers, left = liquefied_layers_from_result(
            self._spt([1.2, 0.6, 0.5, 0.9, 1.6]), volumetric_strain=0.02)
        assert [(s.top, s.bottom) for s in layers] == [
            (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]
        assert [s.N1_60 for s in layers] == [11.0, 12.0, 13.0]
        assert left == 0.0

    def test_the_water_table_and_the_profile_bound_them(self):
        layers, _ = liquefied_layers_from_result(
            self._spt([0.8, 0.6, 1.5, 1.5, 0.7]), volumetric_strain=0.02,
            profile_depth=11.5, water_table=2.5)
        assert [(s.top, s.bottom) for s in layers] == [
            (2.5, 4.0), (4.0, 6.0), (10.0, 11.5)]

    def test_strains_per_reading_and_the_part_left_out(self):
        strain = [0.01, 0.03, 0.02, 0.005, 0.004]
        layers, left = liquefied_layers_from_result(
            self._spt([1.2, 0.6, 0.5, 1.1, 1.6], strain))
        assert [s.volumetric_strain for s in layers] == [0.03, 0.02]
        assert left == pytest.approx(0.01 * 2 + 0.005 * 2 + 0.004 * 2)

    def test_no_strain_says_what_to_give(self):
        with pytest.raises(ValueError, match="Give volumetric_strain"):
            liquefied_layers_from_result(self._spt([1.2, 0.6, 0.5, 0.9,
                                                    1.6]))

    def test_a_cpt_profile_with_its_own_strains(self):
        depth = np.arange(0.5, 14.0, 0.5)
        fos = np.where((depth > 6.2) & (depth < 9.8), 0.7, 2.0)
        eps = np.where(fos < 1.0, 0.025, 0.001)
        layers, left = liquefied_layers_from_result(
            {"depth": depth, "factor_of_safety": fos,
             "volumetric_strain": eps})
        assert layers[0].top == pytest.approx(6.25)
        assert layers[-1].bottom == pytest.approx(9.75)
        assert left > 0

    def test_the_analysis_takes_the_tool_output(self):
        r = post_liquefaction_downdrag(
            _hand_check(), liquefaction_result=self._spt(
                [1.2, 0.6, 0.5, 0.9, 1.6]), volumetric_strain=0.03)
        assert [(z["top_m"], z["bottom_m"]) for z in r.liquefied_zones] == \
            [(4.0, 10.0)]
        assert r.reconsolidation_settlement == pytest.approx(0.18)
        assert r.fs_liq_threshold == 1.0
        whats = [c["what"] for c in r.choices]
        assert "FS_liq at or below which a reading liquefies" in whats
        assert "reconsolidation volumetric strain" in whats

    def test_no_reading_liquefies(self):
        with pytest.raises(ValueError, match="no reading has FS_liq"):
            post_liquefaction_downdrag(
                _hand_check(), liquefaction_result=self._spt(
                    [1.2, 1.6, 1.5, 1.9, 1.6]), volumetric_strain=0.03)


class TestFromACptDepthProfile:
    """The ``liquefaction`` tool's CPT ``depth_profile`` rows (GM-201): each
    row stands for its own top_m to bottom_m and carries its strain."""

    ROWS = [
        {"top_m": 0.0, "bottom_m": 2.0, "FOS_liq": 2.0,
         "volumetric_strain": 0.0},
        {"top_m": 2.0, "bottom_m": 6.0, "FOS_liq": 1.3,
         "volumetric_strain": 0.004},
        {"top_m": 6.0, "bottom_m": 8.0, "FOS_liq": 0.6,
         "volumetric_strain": 0.03},
        {"top_m": 8.0, "bottom_m": 10.0, "FOS_liq": 0.8,
         "volumetric_strain": 0.03},
        {"top_m": 10.0, "bottom_m": 30.5, "FOS_liq": 2.0,
         "volumetric_strain": 0.0},
    ]

    def test_rows_stand_for_their_own_depths(self):
        layers, left = liquefied_layers_from_result(
            {"input_type": "CPT", "depth_profile": self.ROWS})
        assert [(s.top, s.bottom, s.volumetric_strain) for s in layers] == [
            (6.0, 8.0, 0.03), (8.0, 10.0, 0.03)]
        assert [s.fs_liq for s in layers] == [0.6, 0.8]
        assert left == pytest.approx(0.004 * 4.0)

    def test_the_rows_alone_and_one_row_are_taken(self):
        layers, _ = liquefied_layers_from_result(self.ROWS)
        assert len(layers) == 2
        layers, _ = liquefied_layers_from_result([self.ROWS[2]])
        assert [(s.top, s.bottom) for s in layers] == [(6.0, 8.0)]

    def test_the_water_table_and_the_profile_bound_them(self):
        rows = [dict(r, FOS_liq=0.7, volumetric_strain=0.01)
                for r in self.ROWS]
        layers, _ = liquefied_layers_from_result(
            rows, profile_depth=30.0, water_table=1.0)
        assert layers[0].top == 1.0 and layers[-1].bottom == 30.0

    def test_a_threshold_counts_a_band_by_its_fos(self):
        layers, _ = liquefied_layers_from_result(self.ROWS,
                                                 fs_liq_threshold=1.3)
        assert [(s.top, s.bottom) for s in layers][0] == (2.0, 6.0)

    def test_a_given_strain_replaces_the_rows(self):
        layers, _ = liquefied_layers_from_result(self.ROWS,
                                                 volumetric_strain=0.02)
        assert [s.volumetric_strain for s in layers] == [0.02, 0.02]

    def test_overlapping_rows_refused(self):
        rows = [dict(self.ROWS[2]), dict(self.ROWS[3], top_m=7.0)]
        with pytest.raises(ValueError, match="overlap"):
            liquefied_layers_from_result(rows)

    def test_the_analysis_takes_it_and_says_where_the_strain_came_from(self):
        r = post_liquefaction_downdrag(
            _hand_check(), liquefaction_result={
                "input_type": "CPT", "depth_profile": self.ROWS,
                "depth_profile_note": "banded"})
        assert [(z["top_m"], z["bottom_m"]) for z in r.liquefied_zones] == \
            [(6.0, 10.0)]
        assert r.reconsolidation_settlement == pytest.approx(0.12)
        strain = next(c for c in r.choices
                      if c["what"] == "reconsolidation volumetric strain")
        assert strain["chosen"] == "the liquefaction result's own"
        assert "Zhang et al. (2002)" in strain["basis"]
        assert "depth bands" in strain["basis"]
        # the HC-1 zone, so the HC-1 numbers
        given = post_liquefaction_downdrag(_hand_check(),
                                           liquefied_layers=_HC1_ZONE)
        assert r.after.neutral_plane_depth == pytest.approx(
            given.after.neutral_plane_depth)


class TestNoSptStrainMethod:
    """GM-201: no reference in the library prints a post-liquefaction
    volumetric strain method for SPT (GEC-10 §10.7, p. 10-52, names
    Ishihara and Yoshimine 1992 as Idriss and Boulanger 2008 describe it;
    its Fig 9-34 is maximum SHEAR strain; GEC-5 p. 12-23 names Tokimatsu
    and Seed 1987 for dry or partially saturated sand and defers to a
    manual not in the library), so the strain stays an input."""

    def test_an_spt_result_without_a_strain_says_so(self):
        rows = [{"depth_m": d, "FOS_liq": f, "N1_60": 10.0}
                for d, f in ((3.0, 1.2), (5.0, 0.6), (7.0, 1.4))]
        with pytest.raises(ValueError) as exc:
            liquefied_layers_from_result({"layer_results": rows})
        msg = str(exc.value)
        assert "no reference in the library prints an SPT method" in msg
        assert "Give volumetric_strain" in msg


class TestInputs:

    @pytest.mark.parametrize("layers, match", [
        ([{"top": 6, "bottom": 10}], "volumetric_strain"),
        ([{"top": 6, "bottom": 10, "volumetric_strain": 0.02,
           "settlement": 0.08}], "one of them"),
        ([{"top": 6, "bottom": 10, "volumetric_strain": 3}], "decimal"),
        ([{"top": 10, "bottom": 6, "volumetric_strain": 0.02}],
         "top < bottom"),
        ([{"top": 6, "bottom": 10, "volumetric_strain": 0.02},
          {"top": 9, "bottom": 12, "volumetric_strain": 0.02}], "overlap"),
        ([{"top": 28, "bottom": 32, "volumetric_strain": 0.02}],
         "below the soil profile"),
        ([{"top": 6, "bottom": 10, "strain": 0.02}], "unknown key"),
        ([], "empty"),
    ])
    def test_bad_layers_refused(self, layers, match):
        with pytest.raises(ValueError, match=match):
            post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=layers)

    def test_one_way_of_giving_the_layers(self):
        with pytest.raises(ValueError, match="one way"):
            post_liquefaction_downdrag(_hand_check())
        with pytest.raises(ValueError, match="goes with liquefaction_result"):
            post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE,
                                       volumetric_strain=0.02)

    def test_unknown_option_refused(self):
        with pytest.raises(ValueError, match="must be one of"):
            post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE,
                                       liquefied_side_resistance="half")
        assert LIQUEFIED_SIDE_RESISTANCE[0] == "half_pre_earthquake"

    def test_a_layer_object_is_taken_as_is(self):
        r = post_liquefaction_downdrag(
            _hand_check(), liquefied_layers=[LiquefiedLayer(6.0, 10.0,
                                                            0.03)])
        assert r.reconsolidation_settlement == pytest.approx(0.12)

    def test_a_layer_above_the_water_table_is_warned(self):
        a = _hand_check()
        a.soil.gwt_depth = 7.0
        r = post_liquefaction_downdrag(a, liquefied_layers=_HC1_ZONE)
        assert any("above the water table" in w for w in r.warnings)


class TestReporting:

    def test_choices_say_default_or_given(self):
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE)
        by = {c["what"]: c for c in r.choices}
        assert by["side resistance of the liquefied layers"]["how"] == \
            "default"
        assert by["liquefied layers"]["how"] == "given"
        r = post_liquefaction_downdrag(_hand_check(),
                                       liquefied_layers=_HC1_ZONE,
                                       liquefied_side_resistance="zero")
        by = {c["what"]: c for c in r.choices}
        assert by["side resistance of the liquefied layers"]["how"] == \
            "given"

    def test_to_dict_and_summary(self):
        r = post_liquefaction_downdrag(
            _hand_check(structural_capacity=3000.0),
            liquefied_layers=_HC1_ZONE)
        d = r.to_dict()
        assert list(d)[:3] == ["post_liquefaction_basis", "warnings",
                               "methods_range_after"]
        assert d["neutral_plane_after_m"] == pytest.approx(
            r.after.neutral_plane_depth)
        assert d["structural_demand_kN"] == pytest.approx(
            1.25 * 600 + 1.10 * r.after.drag_force)
        assert "GEC-10 §10.7" in d["post_liquefaction_basis"]
        assert "Extreme Event I" in d["limit_states_note"]
        assert d["after"]["neutral_plane_comparison"]
        text = r.summary()
        assert "Post-Liquefaction Downdrag" in text
        assert "Every neutral-plane method, before -> after" in text
        assert "Choices:" in text
