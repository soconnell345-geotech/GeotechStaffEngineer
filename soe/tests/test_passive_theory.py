"""The passive coefficient below the excavation (GM-242).

Rankine's Kp stays the default, as the Caltrans Trenching and Shoring
Manual's examples take it where the wall friction is zero (pp. 7-18, 7-27),
and the result says so in its choices; passive_theory 'log_spiral' takes the
log-spiral Kp of geotech_common.log_spiral, as Caltrans Example 8-1 does
(Kp = 6.3 x R = 4.7 at delta 15 deg on phi 30 deg, p. 8-12).
"""

import math

import numpy as np
import pytest

from soe.beam_analysis import (
    analyze_braced_excavation,
    analyze_cantilever_excavation,
)
from soe.free_earth import passive_by_layer, profile_for, solve_about_support
from soe.geometry import ExcavationGeometry, SOEWallLayer, SupportLevel

FT = 0.3048
PCF = 0.157087464


def _layer(thickness, phi, c=0.0, gamma=18.0, soil_type="sand"):
    return SOEWallLayer(thickness=thickness, unit_weight=gamma,
                        friction_angle=phi, cohesion=c, soil_type=soil_type)


def _cantilever(**kw):
    g = ExcavationGeometry(4.0, [_layer(20.0, 33.0)], surcharge=10.0)
    return analyze_cantilever_excavation(g, **kw)


class TestCaltransExample8_1:
    """The manual's own log-spiral Kp, by the tool's route: the
    Caquot-Kerisel (1948) chart at delta 15 deg on phi 30 deg is 4.7, and
    the free earth support on it is the published D = 6.09 ft."""

    def test_kp_and_embedment(self):
        H, d, g = 25 * FT, 10 * FT, 115 * PCF
        geom = ExcavationGeometry(H, [_layer(100.0, 30.0, gamma=g)],
                                  surcharge=0.0)
        (row,) = passive_by_layer(geom.soil_layers, H, "log_spiral",
                                  "caquot_kerisel_1948", 15.0)
        assert row["Kp"] == pytest.approx(4.7, rel=0.005)
        assert row["came"] == "given"
        prof = profile_for(geom, passive_theory="log_spiral",
                           log_spiral_source="caquot_kerisel_1948",
                           passive_wall_friction_deg=15.0)
        pe = 0.65 * prof.Ka[0] * g * H ** 2 / (H - d / 3 - (H - d) / 3)
        top, bot = 2 / 3 * d, H - 2 / 3 * (H - d)

        def trapezoid(z):
            z = np.asarray(z, float)
            return np.where(z < top, pe * z / top,
                            np.where(z <= bot, pe, pe * (H - z) / (H - bot)))

        sol = solve_about_support(prof, pivot=d, body_top=0.0, FS=1.3,
                                  above=trapezoid)
        assert sol.D / FT == pytest.approx(6.09, rel=0.01)


class TestDefaultStaysRankine:

    def test_the_numbers_do_not_move(self):
        r = _cantilever()
        assert r.required_embedment_m == pytest.approx(5.58, abs=0.005)
        assert r.Kp == pytest.approx(math.tan(math.radians(61.5)) ** 2,
                                     abs=1e-4)

    def test_the_choice_says_so(self):
        (c,) = _cantilever().choices
        assert c["how"] == "default"
        assert c["chosen"] == "rankine: layer 1: Kp 3.39; embedment 5.58 m"
        assert "since the wall friction (delta) is zero" in c["basis"]
        assert "should not be used" in c["basis"]
        (alt,) = c["alternatives"]
        assert alt.startswith("log_spiral (passive_theory 'log_spiral')")
        assert "embedment 3.78 m" in alt


class TestLogSpiral:

    def test_cantilever(self):
        r = _cantilever(passive_theory="log_spiral")
        assert r.Kp == pytest.approx(6.392, abs=1e-3)   # Eq. 4-12, 0.66 phi'
        assert r.required_embedment_m == pytest.approx(3.78, abs=0.005)
        theory, source, friction = r.choices
        assert theory["how"] == "given"
        assert "embedment 5.58 m" in theory["alternatives"][0]
        assert source["chosen"] == "ufc_3_220_20"
        assert "Eq. 4-12 (p. 191)" in source["basis"]
        assert friction["how"] == "default"
        assert friction["chosen"] == "layer 1: 21.8 deg (the source's own)"
        assert "p. 238" in friction["basis"]
        assert r.to_dict()["method"].endswith(
            "Rankine active and log-spiral passive pressures with water")
        assert r.layers_used[0]["passive_wall_friction_deg"] == 21.78

    def test_braced(self):
        g = ExcavationGeometry(8.0, [_layer(20.0, 33.0)],
                               [SupportLevel(2.0), SupportLevel(5.0)],
                               surcharge=10.0)
        plain = analyze_braced_excavation(g)
        spiral = analyze_braced_excavation(g, passive_theory="log_spiral")
        assert plain.required_embedment_m == pytest.approx(2.87, abs=0.005)
        assert spiral.required_embedment_m == pytest.approx(1.72, abs=0.005)
        assert "embedment 2.87 m" in spiral.choices[0]["alternatives"][0]

    def test_layers_by_cohesion_and_given_friction(self):
        layers = [_layer(4.0, 30.0), _layer(3.0, 32.0),
                  _layer(5.0, 25.0, c=15.0, soil_type="stiff_clay"),
                  _layer(5.0, 0.0, c=40.0, soil_type="stiff_clay")]
        rows = passive_by_layer(layers, 4.0, "log_spiral")
        assert [r["came"] for r in rows] == ["above", "source", "cohesion",
                                             "phi_0"]
        assert rows[1]["delta"] == pytest.approx(0.66 * 32.0)
        # a layer with cohesion keeps Rankine's (UFC 3-220-20 p. 238)
        assert rows[2]["Kp"] == pytest.approx(
            math.tan(math.radians(57.5)) ** 2)
        assert rows[3]["Kp"] == pytest.approx(1.0)
        given = passive_by_layer(layers, 4.0, "log_spiral", None, 28.0)
        assert [r["delta"] for r in given] == [0.0, 28.0, 25.0, 0.0]

    def test_calc_steps_show_the_log_spiral_kp(self):
        from soe.calc_steps import get_calc_steps
        r = _cantilever(passive_theory="log_spiral")
        g = ExcavationGeometry(4.0, [_layer(20.0, 33.0)], surcharge=10.0)
        sections = get_calc_steps(r, {"geometry": g})
        items = [i for s in sections for i in s.items]
        table = next(i for i in items if getattr(i, "headers", None)
                     and "Kp" in i.headers)
        assert table.headers[-1] == "δ passive (deg)"
        step = next(i for i in items
                    if getattr(i, "title", "").startswith("Passive coeff"))
        assert "δ = 21.78°" in step.substitution
        assert "log_spiral_source 'ufc_3_220_20'" in step.reference


class TestRefused:

    @pytest.mark.parametrize("kw, says", [
        ({"passive_theory": "coulomb"}, "passive_theory must be one of"),
        ({"log_spiral_source": "caquot_kerisel_1948"},
         "Rankine's Kp takes no wall friction"),
        ({"passive_wall_friction_deg": 15.0},
         "Rankine's Kp takes no wall friction"),
        ({"passive_theory": "log_spiral", "log_spiral_source": "chart"},
         "log_spiral_source must be one of"),
        ({"passive_theory": "log_spiral", "passive_wall_friction_deg": -1.0},
         "passive_wall_friction_deg must be in"),
    ])
    def test_inputs_not_used_are_refused(self, kw, says):
        with pytest.raises(ValueError, match=says):
            _cantilever(**kw)
