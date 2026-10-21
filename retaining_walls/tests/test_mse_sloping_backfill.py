"""MSE walls with a sloping backfill (BACKLOG GM-112).

GEC-11 Vol I (FHWA-NHI-10-024): §4.4.4 Eqs. 4-2/4-3 (Coulomb Kab with
delta = beta), Eq. 4-7 (F_T = 1/2 Kab gamma_b h^2, h = H + L tan(beta)),
Figs. 4-3/4-4/4-8; §4.4.6 Eqs. 4-10 to 4-21; §4.4.7 Eq. 4-29 and Figs.
4-9a, 4-11, 4-15. The published worked examples (Vol II Examples E3 and
E1) are pinned in validation_examples/test_published_v056_v057_gec11_slope.py;
these tests check the equations by hand and the level case unchanged.
"""

import dataclasses
import math

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.earth_pressure import coulomb_Ka, rankine_Ka
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, backslope_geometry, backslope_loads,
    check_external_stability, check_external_stability_lrfd,
    check_internal_stability, Kr_Ka_ratio,
)
from retaining_walls.reinforcement import (
    GEOGRID_GEC11_E1_GG1, RIBBED_STEEL_STRIP_50x4,
)

# GM-256: a steel strip needs its horizontal spacing (Rc = b/Sh = 0.1)
_STRIP = dataclasses.replace(RIBBED_STEEL_STRIP_50x4, horizontal_spacing=0.75)

H, L, G, PHI = 6.0, 5.0, 19.0, 34.0
BETA = math.degrees(math.atan(0.5))          # 2H:1V
T = 0.5


def _geom(**kw):
    kw.setdefault("wall_height", H)
    kw.setdefault("reinforcement_length", L)
    kw.setdefault("reinforcement_spacing", 0.6)
    return MSEWallGeometry(**kw)


class TestGeometryInputs:
    def test_negative_slope_refused(self):
        with pytest.raises(ValueError, match="backfill_slope"):
            _geom(backfill_slope=-5)

    def test_broken_backslope_needs_a_slope(self):
        with pytest.raises(ValueError, match="needs a backfill_slope"):
            _geom(broken_backslope_height=1.0)

    def test_broken_backslope_height_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            _geom(backfill_slope=20, broken_backslope_height=0)


class TestBackslopeGeometry:
    def test_level_reduces_to_level(self):
        bs = backslope_geometry(_geom())
        assert bs["h"] == H and bs["V2_area"] == 0.0
        assert bs["level_length"] == L and bs["S_eq"] == 0.0
        assert bs["H1"] == H and bs["I_deg"] == 0.0

    def test_slope_that_continues(self):
        """Fig. 4-3: h = H + L tan(beta); V2 = L (L tan beta) / 2 at 2L/3;
        delta = beta; Fig. 4-11 S_eq = (1/2)(0.7H) tan(beta); Fig. 4-9a
        H1 = H + tan(beta) 0.3H / (1 - 0.3 tan(beta))."""
        bs = backslope_geometry(_geom(backfill_slope=BETA))
        assert bs["h"] == pytest.approx(H + L * T)
        assert bs["V2_area"] == pytest.approx(L * L * T / 2)
        assert bs["V2_arm"] == pytest.approx(2 * L / 3)
        assert bs["I_deg"] == pytest.approx(BETA)
        assert bs["level_length"] == 0.0
        assert bs["S_eq"] == pytest.approx(0.5 * 0.7 * H * T)
        assert bs["H1"] == pytest.approx(H + T * 0.3 * H / (1 - 0.3 * T))

    def test_broken_backslope_crest_beyond_L(self):
        """Fig. 4-4: crest S = 3 m at 6 m (> L); I = atan(S / 2H);
        h = H + L tan(beta) (L tan(beta) < S)."""
        bs = backslope_geometry(_geom(backfill_slope=BETA,
                                      broken_backslope_height=3.0))
        assert bs["I_deg"] == pytest.approx(math.degrees(math.atan(3 / 12)))
        assert bs["h"] == pytest.approx(H + L * T)
        assert bs["V2_area"] == pytest.approx(L * L * T / 2)
        assert bs["level_length"] == 0.0

    def test_broken_backslope_crest_inside_L(self):
        """Crest S = 1 m at 2 m: h = H + S (p. 4-20), V2 is the triangle to
        the crest plus the level part behind it, 3 m of level ground over
        the zone; S_eq not more than S (p. 4-36)."""
        bs = backslope_geometry(_geom(backfill_slope=BETA,
                                      broken_backslope_height=1.0))
        assert bs["h"] == pytest.approx(H + 1.0)
        area = 1.0 * 2.0 / 2 + 1.0 * 3.0
        assert bs["V2_area"] == pytest.approx(area)
        arm = (1.0 * (2 * 2.0 / 3) + 3.0 * 3.5) / area
        assert bs["V2_arm"] == pytest.approx(arm)
        assert bs["level_length"] == pytest.approx(3.0)
        assert bs["S_eq"] == pytest.approx(min(0.5 * 0.7 * H * T, 1.0))
        assert bs["I_deg"] == pytest.approx(math.degrees(math.atan(1 / 12)))

    def test_broken_backslope_crest_beyond_2H_is_the_slope(self):
        bs = backslope_geometry(_geom(backfill_slope=BETA,
                                      broken_backslope_height=50.0))
        assert bs["I_deg"] == pytest.approx(BETA)


class TestLoads:
    def test_kab_is_coulomb_with_delta_beta(self):
        ld = backslope_loads(_geom(backfill_slope=20), G, 30.0, G)
        assert ld["Kab"] == pytest.approx(coulomb_Ka(30.0, 20, 90.0, 20))
        assert ld["Kab"] > rankine_Ka(30.0)

    def test_thrust_and_components(self):
        """Eq. 4-7 and Eq. 4-10: F_T = 1/2 Kab gamma h^2 at h/3, components
        F_T cos(beta), F_T sin(beta) (the latter at the back, x = L)."""
        ld = backslope_loads(_geom(backfill_slope=BETA), G, 30.0, 18.0)
        h = H + L * T
        FT = 0.5 * ld["Kab"] * 18.0 * h * h
        b = math.radians(BETA)
        assert ld["FT"] == pytest.approx(FT)
        assert ld["FTH"] == pytest.approx(FT * math.cos(b))
        assert ld["FTV"] == pytest.approx(FT * math.sin(b))
        assert ld["MFTH"] == pytest.approx(FT * math.cos(b) * h / 3)
        assert ld["MFTV"] == pytest.approx(FT * math.sin(b) * L)
        # V2 uses the reinforced fill unit weight (Fig. 4-3)
        assert ld["V2"] == pytest.approx(G * L * (h - H) / 2)

    def test_slope_steeper_than_phi_refused(self):
        with pytest.raises(ValueError, match="steeper than the friction"):
            backslope_loads(_geom(backfill_slope=31), G, 30.0, G)

    def test_slope_at_phi_runs(self):
        """At beta = phi Eq. 4-3 has sin(phi - beta) = 0, Gamma = 1:
        Kab = sin^2(90 + phi) / sin(90 - phi) = cos(phi)."""
        ld = backslope_loads(_geom(backfill_slope=30), G, 30.0, G)
        assert ld["Kab"] == pytest.approx(math.cos(math.radians(30)))

    def test_surcharge_on_a_slope_that_continues_refused(self):
        geom = _geom(backfill_slope=20, surcharge=10)
        with pytest.raises(ValueError, match="Eq. 4-16"):
            check_external_stability(geom, G, PHI, G, PHI)
        with pytest.raises(ValueError, match="Eq. 4-16"):
            check_external_stability_lrfd(geom, G, PHI, PHI)
        with pytest.raises(ValueError, match="Eq. 4-16"):
            check_internal_stability(geom, G, PHI, GEOGRID_GEC11_E1_GG1)


class TestExternalASD:
    def test_hand_calculation(self):
        """Unfactored, by hand: sliding (V1 + V2 + F_T sin b) tan(delta_b) /
        F_T cos b; overturning about the toe; e and sigma_v = sum V /
        (L - 2e)."""
        geom = _geom(backfill_slope=BETA)
        out = check_external_stability(geom, G, PHI, G, 30.0, 0.0, 400.0,
                                       phi_retained=30.0)
        Kab = coulomb_Ka(30.0, BETA, 90.0, BETA)
        h = H + L * T
        FT = 0.5 * Kab * G * h * h
        c, s = math.cos(math.radians(BETA)), math.sin(math.radians(BETA))
        V1, V2 = G * H * L, G * L * L * T / 2
        V = V1 + V2 + FT * s
        assert out["FOS_sliding"] == pytest.approx(
            V * math.tan(math.radians(30)) / (FT * c), abs=1e-3)
        MR = V1 * L / 2 + V2 * 2 * L / 3 + FT * s * L
        MO = FT * c * h / 3
        assert out["FOS_overturning"] == pytest.approx(MR / MO, abs=1e-3)
        e = L / 2 - (MR - MO) / V
        assert out["eccentricity_m"] == pytest.approx(e, abs=1e-3)
        assert out["sigma_v_kPa"] == pytest.approx(V / (L - 2 * e), abs=0.1)
        assert out["backslope"]["Kab"] == pytest.approx(Kab, abs=1e-4)

    def test_slope_lowers_the_factors_of_safety(self):
        level = check_external_stability(_geom(), G, PHI, G, PHI)
        slope = check_external_stability(_geom(backfill_slope=20), G, PHI,
                                         G, PHI)
        assert slope["FOS_sliding"] < level["FOS_sliding"]
        assert slope["FOS_overturning"] < level["FOS_overturning"]
        assert "backslope" not in level

    def test_zero_slope_is_the_level_result(self):
        assert (check_external_stability(_geom(backfill_slope=0.0), G, PHI,
                                         G, PHI)
                == check_external_stability(_geom(), G, PHI, G, PHI))

    def test_negative_eccentricity_keeps_the_full_width(self):
        """GEC-11 p. 4-28: L - 2e_B is L when e_B is negative."""
        geom = MSEWallGeometry(wall_height=3.0, reinforcement_length=12.0,
                               backfill_slope=30.0)
        out = check_external_stability(geom, G, 38.0, G, 38.0)
        assert out["eccentricity_m"] < 0
        assert out["L_eff_m"] == pytest.approx(12.0)


class TestExternalLRFD:
    def test_small_slope_tends_to_level(self):
        level = check_external_stability_lrfd(_geom(), G, PHI, 30.0)
        small = check_external_stability_lrfd(_geom(backfill_slope=1e-6),
                                              G, PHI, 30.0)
        for key in ("CDR_strength_max", "CDR_strength_min", "CDR_critical"):
            assert small["sliding"][key] == pytest.approx(
                level["sliding"][key], abs=1e-3)
        assert small["eccentricity"]["eL_critical_m"] == pytest.approx(
            level["eccentricity"]["eL_critical_m"], abs=1e-4)
        assert small["bearing"]["sigma_v_service_kPa"] == pytest.approx(
            level["bearing"]["sigma_v_service_kPa"], abs=0.05)

    def test_broken_backslope_crest_at_the_face_tends_to_level_with_traffic(
            self):
        """A crest of vanishing height: traffic over the whole zone, the
        level case with its live load (Vs excluded from sliding)."""
        q = 12.0
        level = check_external_stability_lrfd(_geom(surcharge=q), G, PHI,
                                              30.0)
        tiny = check_external_stability_lrfd(
            _geom(surcharge=q, backfill_slope=BETA,
                  broken_backslope_height=1e-7), G, PHI, 30.0)
        for key in ("CDR_strength_max", "CDR_strength_min", "CDR_critical"):
            assert tiny["sliding"][key] == pytest.approx(
                level["sliding"][key], abs=1e-3)
        for key in ("sigma_v_strength_max_kPa", "sigma_v_service_kPa"):
            assert tiny["bearing"][key] == pytest.approx(
                level["bearing"][key], abs=0.05)

    def test_slope_reported(self):
        out = check_external_stability_lrfd(_geom(backfill_slope=BETA), G,
                                            PHI, 30.0, phi_retained=30.0)
        u = out["unfactored_kN_m"]
        assert u["FT_H"] == pytest.approx(
            u["FT"] * math.cos(math.radians(BETA)), abs=1e-2)
        assert out["backslope"]["h_m"] == pytest.approx(H + L * T, abs=1e-3)


class TestInternal:
    def test_tmax_adds_sigma_2_and_ka_stays_level(self):
        """Eq. 4-29: sigma_H = Kr gamma_r (Z + S_eq); Ka from beta = 0."""
        geom = _geom(backfill_slope=BETA)
        S_eq = 0.5 * 0.7 * H * T
        Ka = rankine_Ka(PHI)
        for lv in check_internal_stability(geom, G, PHI, GEOGRID_GEC11_E1_GG1):
            z = lv["depth_m"]
            expect = Kr_Ka_ratio(z, "geosynthetic") * Ka * G * (z + S_eq) * 0.6
            assert lv["Tmax_kN_per_m"] == pytest.approx(expect, abs=0.01)

    def test_inextensible_active_zone_uses_H1(self):
        """Fig. 4-9a: La = 0.3 H1 over the upper H1/2, 0.6 (H - Z) below."""
        geom = _geom(backfill_slope=BETA)
        H1 = H + T * 0.3 * H / (1 - 0.3 * T)
        for lv in check_internal_stability(geom, G, PHI,
                                           _STRIP):
            z = lv["depth_m"]
            La = 0.3 * H1 if (H - z) >= H1 / 2 else 0.6 * (H - z)
            assert lv["La_m"] == pytest.approx(La, abs=0.01)

    def test_extensible_active_zone_unchanged(self):
        level = check_internal_stability(_geom(), G, PHI, GEOGRID_GEC11_E1_GG1)
        slope = check_internal_stability(_geom(backfill_slope=BETA), G, PHI,
                                         GEOGRID_GEC11_E1_GG1)
        assert [lv["La_m"] for lv in slope] == [lv["La_m"] for lv in level]

    def test_pullout_depth_below_the_slope(self):
        """Fig. 4-15 / Fig. E3-5: Zp = Z + 1/2 tan(beta) (La + L)."""
        geom = _geom(backfill_slope=BETA)
        for lv in check_internal_stability(geom, G, PHI,
                                           _STRIP):
            La = L - lv["Le_m"]
            assert lv["Zp_m"] == pytest.approx(
                lv["depth_m"] + 0.5 * T * (La + L), abs=0.02)

    @pytest.mark.parametrize("extra", [
        {}, {"backfill_slope": BETA, "broken_backslope_height": 1.0}])
    def test_traffic_in_tmax_not_in_pullout(self, extra):
        """GEC-11 p. 4-49: pullout sigma_v 'neglecting traffic loads'; the
        surcharge (the live load) still raises Tmax (Eq. 4-30)."""
        base = check_internal_stability(_geom(**extra), G, PHI,
                                        _STRIP)
        traffic = check_internal_stability(_geom(surcharge=12.0, **extra), G,
                                           PHI, _STRIP)
        for a, b in zip(base, traffic):
            assert b["Pr_kN_per_m"] == a["Pr_kN_per_m"]
            assert b["Tmax_kN_per_m"] > a["Tmax_kN_per_m"]

    def test_slope_raises_tension_and_pullout(self):
        level = check_internal_stability(_geom(), G, PHI, GEOGRID_GEC11_E1_GG1)
        slope = check_internal_stability(_geom(backfill_slope=BETA), G, PHI,
                                         GEOGRID_GEC11_E1_GG1)
        for a, b in zip(level, slope):
            assert b["Tmax_kN_per_m"] > a["Tmax_kN_per_m"]
            assert b["Pr_kN_per_m"] > a["Pr_kN_per_m"]
            assert "Zp_m" not in a


class TestTrafficVerticalComponent:
    """Behind a broken backslope, F2 sin(I) resists sliding and eccentricity
    only with traffic_vertical_resists (Example E1); by default it does not
    (Vol I p. 4-22). Bearing counts it either way (p. 4-26)."""

    def _geom(self):
        return _geom(backfill_slope=BETA, broken_backslope_height=3.0,
                     surcharge=12.0)

    def test_lrfd(self):
        g = self._geom()
        off = check_external_stability_lrfd(g, G, PHI, 30.0)
        on = check_external_stability_lrfd(g, G, PHI, 30.0,
                                           traffic_vertical_resists=True)
        F2V = off["unfactored_kN_m"]["F2_V"]
        assert F2V > 0
        assert on["sliding"]["CDR_critical"] > off["sliding"]["CDR_critical"]
        assert (on["eccentricity"]["eL_critical_m"]
                < off["eccentricity"]["eL_critical_m"])
        for key in ("sigma_v_strength_max_kPa", "sigma_v_critical_kPa",
                    "sigma_v_service_kPa"):
            assert on["bearing"][key] == off["bearing"][key]
        assert off["backslope"]["traffic_vertical_resists"] is False
        assert "p. 4-22" in off["backslope"]["traffic_basis"]
        assert "Example E1" in on["backslope"]["traffic_basis"]

    def test_asd(self):
        g = self._geom()
        off = check_external_stability(g, G, PHI, G, 30.0)
        on = check_external_stability(g, G, PHI, G, 30.0,
                                      traffic_vertical_resists=True)
        assert on["FOS_sliding"] > off["FOS_sliding"]
        assert on["FOS_overturning"] > off["FOS_overturning"]
        assert on["sigma_v_kPa"] == off["sigma_v_kPa"]

    def test_no_traffic_no_option_in_the_basis(self):
        out = check_external_stability_lrfd(_geom(backfill_slope=BETA), G,
                                            PHI, 30.0)
        assert "traffic_basis" not in out["backslope"]


class TestAnalyzeAndResult:
    def test_result_carries_the_basis(self):
        r = analyze_mse_wall(_geom(backfill_slope=BETA), G, PHI,
                             GEOGRID_GEC11_E1_GG1, lrfd_external=True)
        d = r.to_dict()
        b = d["backslope"]
        assert b["backfill_slope_deg"] == pytest.approx(BETA)
        assert b["S_eq_m"] == pytest.approx(0.5 * 0.7 * H * T, abs=1e-3)
        assert b["sigma_2_kPa"] == pytest.approx(G * 0.5 * 0.7 * H * T,
                                                 abs=0.01)
        assert "GEC-11" in b["basis"]
        assert "backslope" in d["external_lrfd"]
        assert "Backfill slope" in r.summary()

    def test_level_result_has_no_backslope(self):
        r = analyze_mse_wall(_geom(), G, PHI, GEOGRID_GEC11_E1_GG1)
        assert r.backslope is None
        assert "backslope" not in r.to_dict()
        assert "Backfill slope" not in r.summary()

    def test_calc_steps_refuse_a_slope(self):
        """The MSE calc steps re-derive a level backfill; they refuse a
        sloping one rather than print numbers the result contradicts."""
        geom = _geom(backfill_slope=BETA)
        r = analyze_mse_wall(geom, G, PHI, GEOGRID_GEC11_E1_GG1)
        with pytest.raises(ValueError, match="level backfill only"):
            get_calc_steps(r, {"wall_type": "mse", "geom": geom,
                               "gamma_backfill": G, "phi_backfill": PHI,
                               "reinforcement": GEOGRID_GEC11_E1_GG1})
