"""retaining_walls checked against GEC-6, UFC 3-220-20 and FHWA NHI-06-089
as printed (GM-288; module_work/gm288/CHECK.md).

Each test pins a reading taken off the printed page, worked by hand from
the page's equation with made-up walls: the tension zone of a cohesive
backfill, Coulomb's wall friction on the plane through the heel, the soil
over a battered back and over the toe, the passive resistance over the
embedment (with cu under the undrained rule), EN 1997-1's 0.4 V limit, the
default factors of safety, a wall on rock, the bearing pressure compared
with q_allowable, and the trial proportions.
"""

import inspect
import math

import pytest

from retaining_walls.cantilever import (
    GAMMA_CONCRETE, _wall_weight_items, analyze_cantilever_wall,
)
from retaining_walls.earth_pressure import (
    active_force_components, coulomb_Ka, rankine_Ka,
)
from retaining_walls.geometry import CantileverWallGeometry


def _geom(**kw):
    base = dict(wall_height=6.0, base_width=3.6, toe_length=0.8,
                stem_thickness_top=0.3, stem_thickness_base=0.6,
                base_thickness=0.6)
    base.update(kw)
    return CantileverWallGeometry(**base)


def _choice(r, words):
    found = [c for c in r.choices if words in c["what"]]
    return found[0] if found else None


class TestTensionZone:
    """UFC 3-220-20 Fig. 4-5 Note A (p. 183) and FHWA NHI-06-089 p. 10-15:
    no negative active pressure; the pressure Ka (q + gamma z) - 2c sqrt(Ka)
    is zero above z0."""

    def test_the_triangle_below_z0(self):
        Ka, g, H, c = 1.0 / 3.0, 18.0, 6.0, 10.0
        z0 = 2.0 * c * math.sqrt(Ka) / (Ka * g)
        r = active_force_components(g, H, Ka, c=c)
        assert r["z_tension"] == pytest.approx(z0)
        assert r["Pa"] == pytest.approx(0.5 * Ka * g * (H - z0) ** 2)
        assert r["z_Pa"] == pytest.approx((H - z0) / 3.0)
        # the pull the straight sum would subtract is added back
        assert r["Pa_tension_zone"] > 0
        assert r["Pa"] == pytest.approx(
            r["Pa_earth"] + r["Pa_surcharge"] + r["Pa_cohesion"]
            + r["Pa_tension_zone"])

    def test_a_surcharge_shortens_the_zone(self):
        Ka, g, H, c, q = 1.0 / 3.0, 18.0, 6.0, 10.0, 20.0
        sigma0 = Ka * q - 2.0 * c * math.sqrt(Ka)
        z0 = -sigma0 / (Ka * g)
        r = active_force_components(g, H, Ka, c=c, q=q)
        assert r["z_tension"] == pytest.approx(z0)
        assert r["Pa"] == pytest.approx(0.5 * Ka * g * (H - z0) ** 2)

    def test_no_zone_when_the_surcharge_covers_the_cohesion(self):
        Ka, g, H, c, q = 1.0 / 3.0, 18.0, 6.0, 5.0, 40.0
        r = active_force_components(g, H, Ka, c=c, q=q)
        assert r["z_tension"] == 0.0
        assert r["Pa_tension_zone"] == pytest.approx(0.0)
        assert r["Pa"] == pytest.approx(
            0.5 * Ka * g * H ** 2 + Ka * q * H - 2 * c * math.sqrt(Ka) * H)

    def test_no_thrust_when_the_zone_covers_the_wall(self):
        r = active_force_components(18.0, 2.0, 1.0 / 3.0, c=30.0)
        assert r["Pa"] == 0.0
        assert r["z_tension"] == 2.0

    def test_the_choice_names_the_alternatives(self):
        r = analyze_cantilever_wall(_geom(), 18.0, 30.0, c_backfill=8.0)
        ch = _choice(r, "tension")
        assert ch is not None
        alts = " ".join(ch["alternatives"])
        assert "neglected" in alts and "5z" in alts
        # no choice without cohesion
        r0 = analyze_cantilever_wall(_geom(), 18.0, 30.0)
        assert _choice(r0, "tension") is None


class TestCoulombWallFriction:
    """FHWA NHI-06-089 p. 10-20: on the vertical plane through the heel the
    resultant is parallel to the backfill slope, delta = beta."""

    def test_delta_is_the_backfill_slope(self):
        g = _geom(backfill_slope=15.0)
        r = analyze_cantilever_wall(g, 19.0, 32.0, pressure_method="coulomb")
        assert r.sliding_terms["delta_deg"] == pytest.approx(15.0)
        ch = _choice(r, "wall friction on the pressure plane")
        assert ch is not None and "backfill slope" in ch["chosen"]

    def test_at_delta_beta_coulomb_is_rankine(self):
        # Coulomb's Ka on a vertical plane at delta = beta is Rankine's
        # sloped Ka: the thrust does not move with the theory
        g = _geom(backfill_slope=12.0)
        rc = analyze_cantilever_wall(g, 19.0, 30.0, pressure_method="coulomb")
        rr = analyze_cantilever_wall(g, 19.0, 30.0)
        assert rc.sliding_terms["Pa"] == pytest.approx(rr.sliding_terms["Pa"])
        assert rc.FOS_sliding == pytest.approx(rr.FOS_sliding)

    def test_level_backfill_takes_rankine_ka(self):
        g = _geom()
        rc = analyze_cantilever_wall(g, 19.0, 32.0, pressure_method="coulomb")
        rr = analyze_cantilever_wall(g, 19.0, 32.0)
        assert rc.sliding_terms["delta_deg"] == 0.0
        assert rc.sliding_terms["Pa"] == pytest.approx(rr.sliding_terms["Pa"])
        assert coulomb_Ka(32.0, 0.0) == pytest.approx(rankine_Ka(32.0))

    def test_wall_friction_given(self):
        g = _geom(backfill_slope=10.0)
        r = analyze_cantilever_wall(g, 19.0, 32.0, pressure_method="coulomb",
                                    wall_friction_deg=20.0)
        assert r.sliding_terms["delta_deg"] == pytest.approx(20.0)
        H = g.H_active
        Ka = coulomb_Ka(32.0, 20.0, beta_deg=10.0)
        assert r.sliding_terms["Pa"] == pytest.approx(0.5 * Ka * 19.0 * H ** 2)

    @pytest.mark.parametrize("kw", [
        {"wall_friction_deg": 10.0},                       # Rankine
        {"wall_friction_deg": 40.0, "pressure_method": "coulomb"},  # > phi
        {"wall_friction_deg": -1.0, "pressure_method": "coulomb"},
    ])
    def test_refused(self, kw):
        with pytest.raises(ValueError, match="wall_friction_deg"):
            analyze_cantilever_wall(_geom(), 19.0, 32.0, **kw)


class TestSoilOverTheWall:
    """UFC 3-220-20 Fig. 4-27 (p. 220): 'Weight includes ... weight of soil
    above footing'; App. B-1.3 (p. 599) counts the soil over the toe."""

    def test_soil_over_a_battered_back(self):
        g = _geom()
        items = {i["component"]: i for i in _wall_weight_items(g, 19.0)}
        taper, h = 0.3, 5.4
        soil = items["Soil over battered stem back"]
        assert soil["W"] == pytest.approx(19.0 * 0.5 * taper * h)
        assert soil["x"] == pytest.approx(0.8 + 0.3 + 2.0 * taper / 3.0)
        # the stem's taper at the concrete's weight, behind the rect
        assert items["Stem (taper)"]["W"] == pytest.approx(
            GAMMA_CONCRETE * 0.5 * taper * h)

    def test_a_front_batter_has_no_soil_over_it(self):
        g = _geom(stem_batter="front")
        items = {i["component"]: i for i in _wall_weight_items(g, 19.0)}
        assert "Soil over battered stem back" not in items
        # the taper in front of the rect, its centroid a third from the toe
        assert items["Stem (taper)"]["x"] == pytest.approx(0.8 + 2.0 * 0.3 / 3)
        assert items["Stem (rect)"]["x"] == pytest.approx(0.8 + 0.3 + 0.15)

    def test_soil_over_the_toe(self):
        g = _geom(embedment_depth=1.5)
        items = {i["component"]: i
                 for i in _wall_weight_items(g, 19.0, gamma_front=17.0)}
        soil = items["Soil over toe"]
        assert soil["W"] == pytest.approx(17.0 * (1.5 - 0.6) * 0.8)
        assert soil["x"] == pytest.approx(0.4)

    def test_none_at_the_top_of_the_base(self):
        for g in (_geom(), _geom(embedment_depth=0.4)):
            names = [i["component"] for i in _wall_weight_items(g, 19.0)]
            assert "Soil over toe" not in names
        assert _geom().Df == 0.6

    def test_concrete_default(self):
        assert GAMMA_CONCRETE == 23.6
        sig = inspect.signature(analyze_cantilever_wall)
        assert sig.parameters["gamma_concrete"].default == GAMMA_CONCRETE


class TestPassiveInFront:
    """The passive resistance over the embedment Df, with a key from Df to
    Df + key depth (UFC 3-220-20 Fig. 4-29), and cu under the undrained
    rule (UFC 3-220-20 Eq. 4-7, p. 181)."""

    def test_over_the_embedment(self):
        g = _geom(embedment_depth=1.5)
        r = analyze_cantilever_wall(g, 19.0, 32.0, phi_foundation=30.0,
                                    gamma_foundation=18.0,
                                    include_passive=True)
        Kp = 3.0
        assert r.sliding_terms["Pp"] == pytest.approx(
            0.5 * Kp * 18.0 * 1.5 ** 2)

    def test_over_the_key_below_the_embedment(self):
        g = _geom(embedment_depth=1.2, has_shear_key=True, key_depth=0.5)
        r = analyze_cantilever_wall(g, 19.0, 32.0, phi_foundation=30.0,
                                    gamma_foundation=18.0,
                                    include_passive=True)
        assert r.sliding_terms["D_passive_top"] == pytest.approx(1.2)
        assert r.sliding_terms["Pp"] == pytest.approx(
            0.5 * 3.0 * 18.0 * (1.7 ** 2 - 1.2 ** 2))

    def test_cu_under_the_undrained_rule(self):
        g = _geom(embedment_depth=1.0)
        r = analyze_cantilever_wall(g, 19.0, 32.0, phi_foundation=0.0,
                                    c_foundation=40.0, gamma_foundation=18.0,
                                    include_passive=True)
        t = r.sliding_terms
        assert t["sliding_cohesion"] == "undrained"
        assert t["Pp_cohesion"] == pytest.approx(2 * 40.0 * 1.0 * 1.0)
        assert t["Pp"] == pytest.approx(0.5 * 18.0 * 1.0 + 80.0)
        ch = _choice(r, "cohesion in the passive")
        assert ch is not None

    def test_c_not_counted_under_the_drained_rule(self):
        g = _geom(embedment_depth=1.0)
        r = analyze_cantilever_wall(g, 19.0, 32.0, phi_foundation=30.0,
                                    c_foundation=10.0, gamma_foundation=18.0,
                                    include_passive=True)
        assert r.sliding_terms["Pp_cohesion"] == 0.0
        assert r.sliding_terms["Pp"] == pytest.approx(0.5 * 3.0 * 18.0)


class TestGapLimit:
    """EN 1997-1 6.5.3(12)P (p. 65): Rd <= 0.4 Vd where water or air can
    reach the interface of an undrained clay."""

    def test_held_to_0_4_V(self):
        g = _geom()
        kw = dict(phi_foundation=0.0, c_foundation=60.0)
        free = analyze_cantilever_wall(g, 19.0, 32.0, **kw)
        held = analyze_cantilever_wall(g, 19.0, 32.0,
                                       interface_gap_possible=True, **kw)
        V = held.sliding_terms["V"]
        assert free.sliding_terms["R_adhesion"] == pytest.approx(60.0 * 3.6)
        assert 60.0 * 3.6 > 0.4 * V
        assert held.sliding_terms["gap_limited"] is True
        assert held.sliding_terms["R_adhesion"] == pytest.approx(0.4 * V)
        ch = _choice(held, "0.4")
        assert ch is not None

    def test_not_held_below_the_limit(self):
        r = analyze_cantilever_wall(_geom(), 19.0, 32.0, phi_foundation=0.0,
                                    c_foundation=10.0,
                                    interface_gap_possible=True)
        assert r.sliding_terms["gap_limited"] is False
        assert r.sliding_terms["R_adhesion"] == pytest.approx(36.0)

    def test_refused_with_the_drained_rule(self):
        with pytest.raises(ValueError, match="interface_gap_possible"):
            analyze_cantilever_wall(_geom(), 19.0, 32.0,
                                    interface_gap_possible=True)


class TestFactorsOfSafety:
    """UFC 3-220-20 Fig. 4-27 / Sec. 4-5.1.3: FSL 2.0 with passive; FHWA
    NHI-06-089 p. 10-44: FOT about 2.0 on soil and 1.5 on rock."""

    def test_defaults(self):
        g = _geom()
        r = analyze_cantilever_wall(g, 19.0, 32.0)
        assert (r.FOS_sliding_required, r.FOS_overturning_required) \
            == (1.5, 2.0)
        r = analyze_cantilever_wall(g, 19.0, 32.0, include_passive=True)
        assert r.FOS_sliding_required == 2.0
        r = analyze_cantilever_wall(g, 19.0, 32.0, founded_on_rock=True)
        assert r.FOS_overturning_required == 1.5

    def test_given_values_stand(self):
        r = analyze_cantilever_wall(_geom(), 19.0, 32.0, include_passive=True,
                                    founded_on_rock=True, FOS_sliding=1.6,
                                    FOS_overturning=2.2)
        assert (r.FOS_sliding_required, r.FOS_overturning_required) \
            == (1.6, 2.2)
        ch = _choice(r, "required factor of safety against sliding")
        assert ch["chosen"].startswith("1.6")
        assert any(a.startswith("2, the default") for a in
                   ch["alternatives"])

    def test_the_choice_reads_the_same_given_or_not(self):
        # the default given back moves nothing but "how"
        g = _geom()
        r0 = analyze_cantilever_wall(g, 19.0, 32.0, include_passive=True,
                                     founded_on_rock=True)
        r1 = analyze_cantilever_wall(g, 19.0, 32.0, include_passive=True,
                                     founded_on_rock=True, FOS_sliding=2.0,
                                     FOS_overturning=1.5)

        def strip(r):
            return [{k: v for k, v in c.items() if k != "how"}
                    for c in r.choices]
        assert strip(r0) == strip(r1)


class TestRockAndBearing:
    """GEC-6 Sec. 6.4.1 / UFC 3-220-20 Sec. 4-5.1.2: e <= B/6 on soil, B/4
    on rock. The bearing pressure: uniform V/(B - 2e) on soil (GEC-6
    Eq. 6-6; NHI Fig. 10-17), the trapezoid's q_max on rock (UFC Sec.
    4-5.1.4)."""

    def test_uniform_on_soil(self):
        r = analyze_cantilever_wall(_geom(), 19.0, 32.0, q_allowable=250.0)
        b = r.bearing_terms
        assert b["bearing_pressure"] == "uniform"
        q_eq = b["V"] / (3.6 - 2 * abs(b["eccentricity"]))
        assert b["q_eq"] == pytest.approx(q_eq)
        assert r.q_bearing == pytest.approx(q_eq, abs=0.05)
        assert r.FOS_bearing == pytest.approx(250.0 / q_eq, rel=1e-3)
        assert q_eq < max(r.q_toe, r.q_heel)
        ch = _choice(r, "bearing pressure")
        assert ch is not None
        alts = " ".join(ch["alternatives"])
        assert "linear" in alts or "q_max" in alts

    def test_linear_on_rock_or_given(self):
        for kw in ({"founded_on_rock": True}, {"bearing_pressure": "linear"}):
            r = analyze_cantilever_wall(_geom(), 19.0, 32.0,
                                        q_allowable=250.0, **kw)
            q_max = max(r.q_toe, r.q_heel)
            assert r.bearing_terms["bearing_pressure"] == "linear"
            assert r.FOS_bearing == pytest.approx(250.0 / q_max, rel=1e-3)

    def test_bearing_pressure_refused_unknown(self):
        with pytest.raises(ValueError, match="bearing_pressure"):
            analyze_cantilever_wall(_geom(), 19.0, 32.0, q_allowable=250.0,
                                    bearing_pressure="trapezoid")

    def test_middle_half_on_rock(self):
        # a short base: the resultant falls between B/6 and B/4
        g = _geom(base_width=2.6, toe_length=0.5)
        soil = analyze_cantilever_wall(g, 19.0, 32.0)
        rock = analyze_cantilever_wall(g, 19.0, 32.0, founded_on_rock=True)
        e = soil.eccentricity
        assert 2.6 / 6 < abs(e) <= 2.6 / 4
        assert "eccentricity" in soil.checks_failed
        assert "eccentricity" not in rock.checks_failed
        assert rock.eccentricity_limit == pytest.approx(2.6 / 4)
        assert soil.eccentricity_limit == pytest.approx(2.6 / 6)


class TestGeometry:
    """FHWA NHI-06-089 Fig. 10-16(a) (p. 10-38): the trial proportions."""

    def test_auto_size(self):
        g = CantileverWallGeometry(wall_height=6.0)
        assert (g.base_thickness, g.toe_length, g.stem_thickness_base,
                g.base_width) == (0.6, 0.6, 0.6, 3.6)
        # the stem at least its top thickness
        assert CantileverWallGeometry(wall_height=2.0).stem_thickness_base \
            == 0.3

    def test_descending_backfill_refused(self):
        with pytest.raises(ValueError, match="descending"):
            _geom(backfill_slope=-5.0)

    @pytest.mark.parametrize("kw, words", [
        ({"stem_batter": "both"}, "stem_batter"),
        ({"embedment_depth": -0.1}, "embedment_depth"),
        ({"embedment_depth": 7.0}, "embedment_depth"),
    ])
    def test_refused(self, kw, words):
        with pytest.raises(ValueError, match=words):
            _geom(**kw)
