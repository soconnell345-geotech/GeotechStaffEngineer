"""The passive coefficient in front of a cantilever wall (GM-242).

UFC 3-220-20 Sec. 4-5.1 (p. 219) says the passive pressure on the front of
the wall is typically ignored, and with it Fig. 4-27 / Sec. 4-5.1.3 ask FSL
>= 2.0; neither names the theory of its Kp. Rankine's stays the default and
the result says so; passive_theory 'log_spiral' takes the log-spiral Kp of
geotech_common.log_spiral.
"""

import math

import pytest

from geotech_common.log_spiral import dm7_2_eq_4_12_Kp
from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.geometry import CantileverWallGeometry


def _wall(**kw):
    geom = CantileverWallGeometry(wall_height=5.0, base_width=3.5,
                                  toe_length=1.0, base_thickness=0.6)
    return analyze_cantilever_wall(geom, 19.0, 32.0, include_passive=True,
                                   **kw)


def _passive(r):
    return [c for c in r.choices if "front of the wall" in c["what"]]


class TestDefaultStaysRankine:

    def test_the_numbers_do_not_move(self):
        r = _wall()
        assert r.FOS_sliding == pytest.approx(1.527, abs=5e-4)
        assert r.sliding_terms["Kp"] == pytest.approx(
            math.tan(math.radians(61.0)) ** 2)

    def test_the_choice_says_so(self):
        (c,) = _passive(_wall())
        assert c["how"] == "default"
        assert c["chosen"].startswith("rankine: Kp 3.25, Pp 11.1 kN/m, "
                                      "FOS_sliding 1.53")
        assert "typically ignored" in c["basis"] and "p. 219" in c["basis"]
        spiral, none = c["alternatives"]
        assert spiral.startswith("log_spiral (passive_theory 'log_spiral')")
        assert "FOS_sliding 1.65" in spiral
        assert none.endswith("FOS_sliding 1.37")

    def test_no_passive_no_choice(self):
        geom = CantileverWallGeometry(wall_height=5.0, base_width=3.5,
                                      toe_length=1.0, base_thickness=0.6)
        r = analyze_cantilever_wall(geom, 19.0, 32.0)
        assert not _passive(r)


class TestLogSpiral:

    def test_left_out_friction_is_the_sources_own(self):
        r = _wall(passive_theory="log_spiral")
        t = r.sliding_terms
        assert t["passive_delta"] == pytest.approx(0.66 * 32.0)
        assert t["Kp"] == pytest.approx(dm7_2_eq_4_12_Kp(32.0, 0.66 * 32.0))
        assert t["log_spiral_route"] == "dm7_2_eq_4_12"
        assert r.FOS_sliding == pytest.approx(1.651, abs=5e-4)
        theory, source, friction = _passive(r)
        assert theory["how"] == "given"
        assert source["how"] == "default"
        assert "Eq. 4-12 (p. 191)" in source["basis"]
        assert friction["how"] == "default"
        assert friction["chosen"] == "21.1 deg (delta/phi' 0.66)"
        assert "engineer's call" in friction["basis"]

    def test_given_friction_and_source(self):
        r = _wall(passive_theory="log_spiral", passive_wall_friction_deg=15.0,
                  log_spiral_source="caquot_kerisel_1948")
        assert r.sliding_terms["passive_delta"] == 15.0
        assert r.sliding_terms["log_spiral_source"] == "caquot_kerisel_1948"
        (*_, friction) = _passive(r)
        assert friction["how"] == "given"

    def test_calc_steps_name_the_log_spiral_kp(self):
        r = _wall(passive_theory="log_spiral")
        geom = CantileverWallGeometry(wall_height=5.0, base_width=3.5,
                                      toe_length=1.0, base_thickness=0.6)
        analysis = {"geom": geom, "gamma_backfill": 19.0,
                    "phi_backfill": 32.0, "include_passive": True,
                    "passive_theory": "log_spiral"}
        notes = " ".join(getattr(i, "notes", "") or ""
                         for s in get_calc_steps(r, analysis)
                         for i in s.items)
        assert "log-spiral passive resistance (Kp 5.9 at wall friction " \
               "21.1°, log_spiral_source 'ufc_3_220_20')" in notes


class TestRefused:

    @pytest.mark.parametrize("kw, says", [
        ({"passive_theory": "coulomb"}, "passive_theory must be one of"),
        ({"passive_wall_friction_deg": 15.0},
         "Rankine's Kp takes no wall friction"),
        ({"log_spiral_source": "dm7_2_eq_4_12"},
         "Rankine's Kp takes no wall friction"),
        ({"passive_theory": "log_spiral", "passive_wall_friction_deg": 40.0},
         r"passive_wall_friction_deg must be in \[0, phi_foundation = 32\]"),
    ])
    def test_inputs_not_used_are_refused(self, kw, says):
        with pytest.raises(ValueError, match=says):
            _wall(**kw)

    def test_log_spiral_needs_passive(self):
        geom = CantileverWallGeometry(wall_height=5.0, base_width=3.5,
                                      toe_length=1.0, base_thickness=0.6)
        with pytest.raises(ValueError, match="only with include_passive"):
            analyze_cantilever_wall(geom, 19.0, 32.0,
                                    passive_theory="log_spiral")
