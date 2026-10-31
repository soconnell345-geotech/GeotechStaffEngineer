"""A water table behind the cantilever wall (GM-292;
module_work/gm292/CHECK.md).

The earth pressure below the water table on the buoyant unit weight, the
water pressure added with a coefficient of 1 (UFC 3-220-20 Sec. 4-4.1 and
Fig. 4-16(a), pp. 199-202; FHWA NHI-06-089 Fig. 10-12, p. 10-27), checked
against FHWA NHI-06-089 Example 10-1 (pp. 10-28, 10-29) as printed and the
chart of UFC Fig. 4-16(a); the uplift on the base, its moment and the
bearing pressures on a made-up wall worked by hand; and a wall with no
water table, or one at or below the underside of the base, unchanged.
"""

import dataclasses
import math

import pytest

from retaining_walls.calc_steps import get_calc_steps, get_input_summary
from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.earth_pressure import (
    active_force_components_water, coulomb_Ka,
)
from retaining_walls.geometry import CantileverWallGeometry

FT = 0.3048                 # m
PCF = 0.1570875             # kN/m3 per pcf
LB_PER_FT = 0.0145939       # kN/m per lb/ft
GAMMA_W = 9.81


def _choice(r, words):
    found = [c for c in r.choices if words in c["what"]]
    return found[0] if found else None


# The made-up wall worked by hand in CHECK.md: H 5, B 3, toe 0.5, stem 0.3
# to 0.5 (back battered), base 0.5; gamma 18, gamma_sat 20, phi 30; the
# water table 2.0 m down
def _wall(**kw):
    g = dict(wall_height=5.0, base_width=3.0, toe_length=0.5,
             stem_thickness_top=0.3, stem_thickness_base=0.5,
             base_thickness=0.5, water_table_depth=2.0)
    g.update(kw.pop("geom", {}))
    args = dict(q_allowable=200.0, gamma_sat_backfill=20.0)
    args.update(kw)
    return analyze_cantilever_wall(CantileverWallGeometry(**g), 18.0, 30.0,
                                   **args)


class TestNHIExample10_1:
    """NHI-06-089 Example 10-1 (pp. 10-28, 10-29): an 18 ft smooth wall,
    the water table 6 ft down, gamma 115 pcf above it and gamma_sat 120 pcf
    below, phi 30, beta 10, Coulomb Ka 0.374 at delta 0. Printed: p'a
    258.1 psf at 6 ft and 516.6 psf at 18 ft, u 748.8 psf at 18 ft. The
    thrusts are the printed diagrams' areas."""

    PA = 0.5 * 258.1 * 6 + 0.5 * (258.1 + 516.6) * 12      # lb/ft
    PW = 0.5 * 748.8 * 12

    def _result(self):
        # no heel, so the pressure plane is the wall's face, H high
        g = CantileverWallGeometry(
            wall_height=18 * FT, base_width=1.4, toe_length=0.8,
            stem_thickness_top=0.6, stem_thickness_base=0.6,
            base_thickness=0.6, backfill_slope=10.0,
            water_table_depth=6 * FT)
        return analyze_cantilever_wall(
            g, 115 * PCF, 30.0, pressure_method="coulomb",
            wall_friction_deg=0.0, gamma_sat_backfill=120 * PCF)

    def test_Ka_as_printed(self):
        assert coulomb_Ka(30.0, 0.0, beta_deg=10.0) == pytest.approx(
            0.374, abs=5e-4)

    def test_effective_earth_thrust_is_the_printed_diagram(self):
        ep = self._result().earth_pressure
        assert ep["Pa"] == pytest.approx(self.PA * LB_PER_FT, rel=3e-3)
        # its height: the printed diagram's centroid
        M = (0.5 * 258.1 * 6 * (12 + 2) + 258.1 * 12 * 6
             + 0.5 * (516.6 - 258.1) * 12 * 4)
        assert ep["z_Pa"] == pytest.approx(M / self.PA * FT, rel=3e-3)

    def test_water_thrust_is_the_printed_diagram(self):
        ep = self._result().earth_pressure
        assert ep["Pw"] == pytest.approx(self.PW * LB_PER_FT, rel=3e-3)
        assert ep["z_Pw"] == pytest.approx(12 * FT / 3)
        assert ep["Ph"] == pytest.approx(ep["Pa"])     # delta 0


class TestUFCFig4_16a:
    """UFC 3-220-20 Fig. 4-16(a) (p. 202): sigma_A = K_A [gamma h +
    gamma_b (z - h)], sigma_W = gamma_w (z - h); its chart, for phi' 30,
    gamma 125 pcf and gamma_b 62.5 pcf, reads (P_A,wet + P_W) / P_A,dry
    2.0 at H_W/H 1, read off the curve. Below that the curve as drawn
    reads up to 0.04 above the figure's own equations (1.40 at H_W/H 0.6,
    the equations 1.36), so the heights between are pinned to the
    equations, worked by hand in pcf (the module takes gamma_w 9.81 kN/m3,
    62.45 pcf, and gamma_b = 125 pcf - gamma_w)."""

    def _ratio(self, x):
        H = 6.0
        g = CantileverWallGeometry(
            wall_height=H, base_width=1.0, toe_length=0.4,
            stem_thickness_top=0.6, stem_thickness_base=0.6,
            base_thickness=0.6, water_table_depth=H * (1 - x))
        wet = analyze_cantilever_wall(g, 125 * PCF, 30.0)
        dry = analyze_cantilever_wall(
            dataclasses.replace(g, water_table_depth=None), 125 * PCF, 30.0)
        ep = wet.earth_pressure
        return (ep["Pa"] + ep["Pw"]) / dry.earth_pressure["Pa"]

    def test_full_height(self):
        assert self._ratio(1.0) == pytest.approx(2.0, abs=0.02)

    @pytest.mark.parametrize("x", [0.2, 0.4, 0.6, 0.8])
    def test_heights_between_by_the_equations(self, x):
        # H = 1, h = 1 - x above the water, K_A = 1/3
        h, g, gb, gw = 1.0 - x, 125.0, 62.5, 62.4
        wet = (g * h * h / 2 + g * h * x + gb * x * x / 2) / 3 + gw * x * x / 2
        assert self._ratio(x) == pytest.approx(wet / (g / 6), rel=5e-3)


class TestMadeUpWallByHand:
    """CHECK.md's hand check: Ka 1/3, z_w 2.0, h_w 3.0, gamma_b 10.19."""

    def test_thrusts(self):
        ep = _wall().earth_pressure
        assert ep["Pa"] == pytest.approx(
            (36.0 + 108.0 + 0.5 * 10.19 * 9.0) / 3.0)          # 63.285
        assert ep["z_Pa"] == pytest.approx(113.285 / 63.285, rel=1e-5)
        assert ep["Pw"] == pytest.approx(0.5 * GAMMA_W * 9.0)  # 44.145
        assert ep["z_Pw"] == pytest.approx(1.0)

    def test_saturated_weights_and_uplift(self):
        f = {i["component"]: i for i in _wall().vertical_forces}
        heel = f["Soil on heel, below the water table (gamma_sat - gamma)"]
        assert heel["W"] == pytest.approx(2.0 * 2.0 * 2.5)
        back = f["Soil over battered stem back, below the water table "
                 "(gamma_sat - gamma)"]
        w = 0.2 * 2.5 / 4.5
        assert back["W"] == pytest.approx(2.0 * 0.5 * w * 2.5)
        assert back["x"] == pytest.approx(1.0 - w / 3.0)
        up = f["Uplift on the base (linear, heel to toe)"]
        assert up["W"] == pytest.approx(-0.5 * GAMMA_W * 3.0 * 3.0)
        assert up["x"] == pytest.approx(2.0)
        assert up["resists"] and not up["bearing"]

    def test_sliding(self):
        r = _wall()
        s = r.sliding_terms
        V = 258.2577778 - 44.145
        assert s["V"] == pytest.approx(V, rel=1e-6)
        assert s["driving"] == pytest.approx(63.285 + 44.145)
        assert r.FOS_sliding == pytest.approx(
            V * math.tan(math.radians(20.0)) / 107.43, abs=1e-3)   # 0.725

    def test_overturning_and_eccentricity(self):
        r = _wall()
        o = r.overturning_terms
        assert o["M_stabilizing"] == pytest.approx(434.84049, rel=1e-6)
        assert o["M_overturning"] == pytest.approx(
            113.285 + 44.145 + 88.29, rel=1e-6)
        assert r.FOS_overturning == pytest.approx(434.84049 / 245.72,
                                                  abs=1e-3)
        assert r.eccentricity == pytest.approx(
            1.5 - (434.84049 - 245.72) / 214.11278, abs=1e-3)
        assert "eccentricity" in r.checks_failed       # 0.617 > B/6

    def test_bearing_leaves_the_uplift_out(self):
        b = _wall().bearing_terms
        assert b["V"] == pytest.approx(258.2577778, rel=1e-6)
        assert b["M_over"] == pytest.approx(113.285 + 44.145, rel=1e-6)
        assert b["q_toe"] == pytest.approx(159.40, abs=0.01)
        assert b["q_heel"] == pytest.approx(12.77, abs=0.01)
        assert b["q_eq"] == pytest.approx(120.21, abs=0.01)

    def test_the_choices_give_the_other_readings(self):
        r = _wall()
        up = _choice(r, "uplift (base_uplift)")
        assert up["how"] == "default"
        assert "U = gamma_w h_w B / 2 = 44.1 kN/m" in up["chosen"]
        assert "uniform" in up["alternatives"][0]
        assert "U = gamma_w h_w B = 88.3 kN/m" in up["alternatives"][0]
        mom = _choice(r, "uplift moment")
        assert mom["how"] == "module"
        assert "= 2.2" in mom["alternatives"][0]       # 346.55 / 157.43
        brg = _choice(r, "in the bearing pressures (uplift_in_bearing)")
        assert brg["chosen"].startswith("left out")
        assert "q_eq 121.2 kPa" in brg["alternatives"][0]
        wt = _choice(r, "water table behind the wall")
        assert wt["how"] == "given"
        assert "drained" in wt["alternatives"][0]
        assert _choice(r, "below the water table")["chosen"].startswith(
            "gamma_sat = 20 kN/m3")

    def test_uniform_uplift_and_uplift_in_bearing(self):
        r = _wall(base_uplift="uniform", uplift_in_bearing=True)
        up = [i for i in r.vertical_forces if i["kind"] == "uplift"][0]
        assert up["W"] == pytest.approx(-GAMMA_W * 3.0 * 3.0)
        assert up["x"] == pytest.approx(1.5)
        assert r.bearing_terms["V"] == pytest.approx(258.2577778 - 88.29)
        assert _choice(r, "uplift (base_uplift)")["how"] == "given"


class TestNoWaterUnchanged:
    def test_no_water_table_reports_none(self):
        r = _wall(geom={"water_table_depth": None}, gamma_sat_backfill=None)
        d = r.to_dict()
        assert "water_table_depth_m" not in d
        assert not [c for c in r.choices if "water" in c["what"]]
        assert r.earth_pressure["Pw"] == 0.0

    @pytest.mark.parametrize("depth", [5.0, 7.5])
    def test_at_or_below_the_base_changes_nothing(self, depth):
        dry = _wall(geom={"water_table_depth": None}, gamma_sat_backfill=None)
        low = _wall(geom={"water_table_depth": depth},
                    gamma_sat_backfill=None)
        for key in ("FOS_sliding", "FOS_overturning", "FOS_bearing",
                    "q_toe", "q_heel", "eccentricity"):
            assert getattr(low, key) == getattr(dry, key)
        wt = _choice(low, "water table behind the wall")
        assert "no water pressure on the wall and no uplift" in wt["chosen"]
        assert low.to_dict()["water_thrust_kN_per_m"] == 0.0


class TestRefusals:
    def test_saturated_weight_without_a_water_table(self):
        with pytest.raises(ValueError, match="water_table_depth"):
            _wall(geom={"water_table_depth": None})

    def test_uplift_inputs_without_a_water_table(self):
        with pytest.raises(ValueError, match="water_table_depth"):
            _wall(geom={"water_table_depth": None}, gamma_sat_backfill=None,
                  base_uplift="uniform")

    def test_negative_depth(self):
        with pytest.raises(ValueError, match="water_table_depth"):
            CantileverWallGeometry(wall_height=5.0, water_table_depth=-0.5)

    def test_saturated_weight_below_water(self):
        with pytest.raises(ValueError, match="unit weight of water"):
            _wall(gamma_sat_backfill=9.0)

    def test_unknown_uplift(self):
        with pytest.raises(ValueError, match="base_uplift"):
            _wall(base_uplift="triangular")


class TestCohesiveBackfillUnderWater:
    def test_exact_integral_matches_slices(self):
        # Ka 0.3, gamma 18, gamma_b 9, H 6, water 2.5 down, c 12, q 5
        Ka, c, q = 0.3, 12.0, 5.0
        out = active_force_components_water(18.0, 9.0, 6.0, 2.5, Ka, c, q)
        n, total, moment = 20000, 0.0, 0.0
        for i in range(n):
            z = (i + 0.5) * 6.0 / n
            s = q + 18.0 * min(z, 2.5) + 9.0 * max(z - 2.5, 0.0)
            p = max(Ka * s - 2 * c * math.sqrt(Ka), 0.0) * 6.0 / n
            total += p
            moment += p * (6.0 - z)
        assert out["Pa"] == pytest.approx(total, rel=1e-6)
        assert out["z_Pa"] == pytest.approx(moment / total, rel=1e-6)
        assert out["z_tension"] == pytest.approx(
            (2 * c * math.sqrt(Ka) / Ka - q) / 18.0)

    def test_tension_zone_choice_under_water(self):
        r = _wall(c_backfill=10.0)
        alts = _choice(r, "tension zone")["alternatives"]
        Pa_c0 = active_force_components_water(
            18.0, 10.19, 5.0, 2.0, 1 / 3, 0.0, 0.0)["Pa"]
        assert f"Pa = {Pa_c0:.1f} kN/m" in alts[0]


class TestCalcPackage:
    def _analysis(self, r, g):
        return {"wall_type": "cantilever", "geom": g, "gamma_backfill": 18.0,
                "phi_backfill": 30.0, "c_backfill": 0.0,
                "pressure_method": "rankine", "q_allowable": 200.0,
                "gamma_sat_backfill": 20.0}

    def test_the_steps_print_the_water(self):
        r = _wall()
        g = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_top=0.3, stem_thickness_base=0.5,
            base_thickness=0.5, water_table_depth=2.0)
        a = self._analysis(r, g)
        steps = [i for s in get_calc_steps(r, a) for i in s.items]
        titles = [getattr(i, "title", "") for i in steps]
        assert "Active Thrust with a Water Table (effective stress)" in titles
        assert "Water Pressure Thrust" in titles
        assert "Uplift on the Base" in titles
        ot = [i for i in steps
              if getattr(i, "title", "") == "Overturning Moment about Toe"][0]
        assert ot.equation.startswith("M_ot = P_h")
        assert "P_w" in ot.equation and "U" in ot.equation
        assert ot.result_value == f"{245.72:.1f}"
        names = [i.name for i in get_input_summary(r, a)]
        assert "d_w" in names and "γ_sat" in names

    def test_a_result_built_by_hand_works_the_terms_out(self):
        r = _wall()
        g = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_top=0.3, stem_thickness_base=0.5,
            base_thickness=0.5, water_table_depth=2.0)
        bare = dataclasses.replace(r, earth_pressure=None,
                                   vertical_forces=None, sliding_terms=None,
                                   overturning_terms=None, bearing_terms=None)
        steps = [i for s in get_calc_steps(bare, self._analysis(r, g))
                 for i in s.items]
        pw = [i for i in steps if getattr(i, "title", "") ==
              "Water Pressure Thrust"][0]
        assert pw.result_value == "44.1"
