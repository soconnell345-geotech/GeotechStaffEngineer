"""GM-300 and GM-301 (module_work/gm300/CHECK.md): the drivability tool
takes the quakes and damping factors the other tools take; the shaft
resistance acts over the pile's embedded length, spread uniformly or
triangularly as GEC-12 Vol 2's examples print (Example 5, p. 179;
Examples 1 and 6, pp. 162, 182); drivability takes the pile in the leads
at every depth (pp. 224, 236); GEC-12 Example 6 with its printed
embedment. Made-up piles unless named."""

import math

import pytest

from funhouse_agent import dispatch as D
from wave_equation import (
    Cushion, Hammer, SoilSetup, discretize_pile, drivability_study,
    generate_bearing_graph, get_hammer, make_cushion_from_properties,
    simulate_blow,
)
from wave_equation.calc_steps import get_calc_steps, get_input_summary

AREA = math.pi / 4 * (0.32**2 - 0.30**2)


def _system(length=18.0):
    return (get_hammer("Vulcan 010"), Cushion(stiffness=500000, cor=0.80),
            discretize_pile(length, AREA, 200e6, 1.0, 77.3))


class TestShaftShares:
    """Each segment's share of the shaft resistance, worked by hand on a
    10 m pile of ten 1 m segments."""

    LENGTHS = [1.0] * 10

    def test_uniform_whole_pile(self):
        s = SoilSetup(1000.0).shaft_shares(self.LENGTHS)
        assert s == pytest.approx([0.1] * 10)

    def test_uniform_embedded(self):
        # grade 3.5 m below the head: 0, 0, 0, half a metre, then 1 m each
        s = SoilSetup(1000.0, embedded_length=6.5).shaft_shares(self.LENGTHS)
        assert s == pytest.approx([0, 0, 0, 0.5 / 6.5] + [1 / 6.5] * 6)

    def test_triangular_whole_pile(self):
        s = SoilSetup(1000.0, shaft_distribution="triangular").shaft_shares(
            self.LENGTHS)
        assert s == pytest.approx([((i + 1)**2 - i**2) / 100.0
                                   for i in range(10)])

    def test_triangular_embedded(self):
        s = SoilSetup(1000.0, shaft_distribution="triangular",
                      embedded_length=6.5).shaft_shares(self.LENGTHS)
        # from grade at 3.5 m: (b - 3.5)^2 - (max(a, 3.5) - 3.5)^2 over 6.5^2
        want = [0, 0, 0, 0.25]
        want += [(b - 3.5)**2 - (b - 4.5)**2 for b in range(5, 11)]
        assert s == pytest.approx([w / 42.25 for w in want])
        assert sum(s) == pytest.approx(1.0)

    def test_the_toe_keeps_its_share(self):
        soil = SoilSetup(1000.0, skin_fraction=0.4, embedded_length=6.5,
                         shaft_distribution="triangular")
        side, toe = soil.create_segment_models(10, self.LENGTHS)
        assert sum(m.R_ultimate for m in side) == pytest.approx(400.0)
        assert side[0].R_ultimate == 0.0
        assert toe.R_ultimate == pytest.approx(600.0)


class TestUnsetUnchanged:
    def test_whole_pile_given_is_the_unset_path(self):
        h, c, p = _system()
        a = simulate_blow(h, c, p, SoilSetup(900.0))
        b = simulate_blow(h, c, p, SoilSetup(900.0, embedded_length=18.0))
        assert a.permanent_set == b.permanent_set
        assert a.max_compression_stress == b.max_compression_stress

    def test_drivability_at_the_pile_length(self):
        h, c, _ = _system()
        a = drivability_study(h, c, AREA, 200e6, 77.3, [12.0], [900.0])
        b = drivability_study(h, c, AREA, 200e6, 77.3, [12.0], [900.0],
                              pile_length=12.0)
        assert a.points[0].blow_count == b.points[0].blow_count
        assert a.to_dict()["pile_length_m"] == 12.0
        assert b.to_dict()["pile_length_m"] == 12.0
        assert [c["what"] for c in a.choices] == [
            "pile length in the leads (pile_length)"]
        assert a.choices[0]["how"] == "default"
        assert b.choices[0]["how"] == "given"
        assert b.choices[0]["chosen"] == a.choices[0]["chosen"]


class TestDrivabilityLeftOutPile:
    """GM-300 review: with pile_length left out, the pile as long as the
    deepest depth at every depth (GEC-12 Vol 2 pp. 224, 227, 236), not a
    pile as long as each depth."""

    def test_the_deepest_depth_at_every_depth(self):
        h, c, _ = _system()
        kw = dict(depths=[5.0, 11.0, 17.0], R_at_depth=[300.0, 700.0, 1200.0],
                  skin_fractions=[0.8, 0.6, 0.5])
        out = drivability_study(h, c, AREA, 200e6, 77.3, **kw)
        full = drivability_study(h, c, AREA, 200e6, 77.3, pile_length=17.0,
                                 **kw)
        assert [p.permanent_set for p in out.points] == \
            [p.permanent_set for p in full.points]
        assert out.pile_length == 17.0 and not out.pile_length_given
        ch = out.choices[-1]
        assert ch["how"] == "default" and ch["chosen"].startswith("17 m")
        assert "p. 227" in ch["basis"] and "pile_length" in \
            ch["alternatives"][0]

    def test_the_shallow_depth_drives_the_whole_pile(self):
        """At 5 m the 17 m pile, 12 m standing above grade: not the 5 m
        pile the old default analysed."""
        h, c, _ = _system()
        out = drivability_study(h, c, AREA, 200e6, 77.3, [5.0, 17.0],
                                [300.0, 1200.0], skin_fractions=[0.8, 0.5])
        short = drivability_study(h, c, AREA, 200e6, 77.3, [5.0], [300.0],
                                  skin_fractions=[0.8])
        assert out.points[0].max_tens_stress != short.points[0].max_tens_stress
        assert short.pile_length == 5.0

    def test_the_tool_states_it(self):
        r = D.call_agent("wave_equation", "drivability", {
            "pile_area": AREA, "hammer_name": "Vulcan 010",
            "depths": [6.0, 12.0], "R_at_depth": [500.0, 900.0]})
        assert r["pile_length_m"] == 12.0
        assert r["defaults_used"]["pile_length"] == 12.0
        assert any(ch["what"] == "pile length in the leads (pile_length)"
                   for ch in r["choices"])


class TestEmbedment:
    def test_stick_up_moves_the_blow(self):
        h, c, p = _system(20.0)
        whole = simulate_blow(h, c, p, SoilSetup(900.0))
        part = simulate_blow(h, c, p, SoilSetup(900.0, embedded_length=14.0))
        tri = simulate_blow(h, c, p, SoilSetup(
            900.0, embedded_length=14.0, shaft_distribution="triangular"))
        assert len({whole.permanent_set, part.permanent_set,
                    tri.permanent_set}) == 3

    def test_drivability_is_the_single_blow_at_each_depth(self):
        """With pile_length, each depth is the pile in the leads embedded
        to the depth (GEC-12 Vol 2 p. 224): the blow a single blow gives
        with that embedded length."""
        h, c, p = _system(16.0)
        dr = drivability_study(h, c, AREA, 200e6, 77.3, [6.0, 11.0],
                               [500.0, 900.0], skin_fractions=[0.6, 0.4],
                               pile_length=16.0,
                               shaft_distribution="triangular")
        for pt, f in zip(dr.points, (0.6, 0.4)):
            r = simulate_blow(h, c, p, SoilSetup(
                pt.R_ultimate, skin_fraction=f, embedded_length=pt.depth,
                shaft_distribution="triangular"))
            assert pt.permanent_set == pytest.approx(r.permanent_set)
        assert dr.to_dict()["shaft_distribution"] == "triangular"

    def test_bearing_graph_takes_them(self):
        h, c, p = _system(20.0)
        bg = generate_bearing_graph(h, c, p, R_min=600, R_max=600,
                                    embedded_length=15.0,
                                    shaft_distribution="triangular")
        r = simulate_blow(h, c, p, SoilSetup(
            600.0, embedded_length=15.0, shaft_distribution="triangular"))
        assert bg.permanent_sets[0] == pytest.approx(r.permanent_set)


class TestRefusals:
    def test_deeper_than_the_pile(self):
        h, c, p = _system()
        with pytest.raises(ValueError, match="p. 227"):
            simulate_blow(h, c, p, SoilSetup(900.0, embedded_length=19.0))
        with pytest.raises(ValueError, match="p. 227"):
            drivability_study(h, c, AREA, 200e6, 77.3, [10.0, 20.0],
                              [500.0, 900.0], pile_length=18.0)

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match="shaft_distribution"):
            SoilSetup(900.0, shaft_distribution="parabolic")
        with pytest.raises(ValueError, match="embedded_length"):
            SoilSetup(900.0, embedded_length=0.0)


def _ex6(hammer, toe_quake_in, embedded_ft=52.5, distribution="triangular"):
    """GEC-12 Vol 2 Example 6 (p. 182; Fig. 12-14 and Table 12-2, p. 183):
    14 in x 0.375 in closed end pipe, 66 ft long, embedded 52.5 ft, 400
    kips, 30 % on the shaft 'distributed triangularly along the embedded
    portion', J_side 0.05 s/ft, toe 0.15 s/ft, shaft quake 0.10 in; the
    hammer cushion as the one cushion."""
    IN, KSI, KIP, FT = 0.0254, 6894.757, 4.448222, 0.3048
    area = math.pi / 4 * (14**2 - 13.25**2) * IN**2
    pile = discretize_pile(66 * FT, area, 200e6, 1.0, 77.3)
    E, A, t, cor, helmet = {"Vulcan 014": (208, 234, 6.0, 0.91, 1.67),
                            "ICE 42-S": (175, 398, 2.0, 0.92, 2.05)}[
                                hammer.name]
    cushion = make_cushion_from_properties(A * IN**2, t * IN, E * KSI, cor)
    soil = SoilSetup(400 * KIP, skin_fraction=0.30, quake_side=0.10 * IN,
                     quake_toe=toe_quake_in * IN, damping_side=0.05 / FT,
                     damping_toe=0.15 / FT, shaft_distribution=distribution,
                     embedded_length=embedded_ft * FT)
    r = simulate_blow(hammer, cushion, pile, soil, helmet_weight=helmet * KIP)
    return FT / r.permanent_set, r.max_compression_stress / KSI


class TestGEC12Example6:
    """With the printed embedment and distribution: the Vulcan 014 87.5
    and 51.3 blows/ft at the 0.40 and 0.12 in toe quakes against GRLWEAP's
    94 and 56 (p. 182); 92.6 and 51.7 all embedded and uniform. The ICE
    42-S, run as a drop ram (no diesel cycle), 105.6 and 47.5 against 209
    and 99."""

    def test_vulcan_014(self):
        h = get_hammer("Vulcan 014")
        assert _ex6(h, 0.40)[0] == pytest.approx(87.5, abs=0.3)
        assert _ex6(h, 0.12)[0] == pytest.approx(51.3, abs=0.3)
        assert _ex6(h, 0.40)[1] == pytest.approx(29.0, abs=1.5)  # Fig. 12-15

    def test_ice_42s_as_a_drop_ram(self):
        h = Hammer("ICE 42-S", 4.09 * 4.448222, 10.3 * 0.3048, 0.80,
                   "diesel")
        assert _ex6(h, 0.40)[0] == pytest.approx(105.6, abs=0.5)
        assert _ex6(h, 0.12)[0] == pytest.approx(47.5, abs=0.5)


class TestTools:
    BASE = {"pile_area": AREA, "hammer_name": "Vulcan 010"}

    def test_drivability_takes_the_quakes_and_damping(self):
        """GM-300: a closed-end pipe's D/60 toe quake (0.32 m: 5.3 mm)
        reaches the drivability tool, and its choice is the single blow's."""
        drv = D.call_agent("wave_equation", "drivability", {
            **self.BASE, "depths": [12.0], "R_at_depth": [900.0],
            "quake_toe": 0.0053, "damping_side": 0.33})
        one = D.call_agent("wave_equation", "single_blow", {
            **self.BASE, "pile_length": 12.0, "R_total": 900.0,
            "quake_toe": 0.0053, "damping_side": 0.33})
        assert "error" not in drv, drv.get("error")
        assert drv["points"][0]["blow_count_per_m"] == pytest.approx(
            one["blow_count_per_m"], abs=0.1)
        assert drv["choices"][0]["what"] == one["choices"][0]["what"]
        used = drv["defaults_used"]
        assert used["quake_side"] == 0.0025 and used["damping_toe"] == 0.49
        assert "quake_toe" not in used
        assert used["pile_length"] == 12.0      # the deepest depth

    def test_the_new_inputs_reach_the_module(self):
        r = D.call_agent("wave_equation", "single_blow", {
            **self.BASE, "pile_length": 18.0, "R_total": 900.0})
        assert r["defaults_used"]["embedded_length"] == 18.0
        assert r["defaults_used"]["shaft_distribution"] == "uniform"
        bad = D.call_agent("wave_equation", "bearing_graph", {
            **self.BASE, "pile_length": 18.0, "embedded_length": 20.0,
            "R_min": 500.0, "R_max": 500.0})
        assert "error" in bad and "p. 227" in bad["error"]
        drv = D.call_agent("wave_equation", "drivability", {
            **self.BASE, "depths": [6.0, 12.0], "R_at_depth": [500.0, 900.0],
            "pile_length": 15.0, "shaft_distribution": "triangular"})
        assert drv["pile_length_m"] == 15.0
        assert "pile_length" not in drv["defaults_used"]

    def test_the_package_prints_them(self):
        from funhouse_agent.adapters.wave_equation import run_bearing_graph
        bg, used = run_bearing_graph({
            **self.BASE, "pile_length": 20.0, "R_min": 600.0,
            "R_max": 600.0, "embedded_length": 15.0,
            "shaft_distribution": "triangular"})
        rows = {i.name: i.value for i in get_input_summary(bg, used)}
        assert rows["Shaft distribution"] == "triangular"
        assert rows["L_emb"] == "15.00"
        bg, used = run_bearing_graph({
            **self.BASE, "pile_length": 20.0, "R_min": 600.0,
            "R_max": 600.0})
        rows = {i.name: i.value for i in get_input_summary(bg, used)}
        assert rows["L_emb"] == "20.00 (the whole pile)"
        assert rows["Shaft distribution"] == "uniform"

    def test_drivability_steps_say_the_pile(self):
        h, c, _ = _system()
        dr = drivability_study(h, c, AREA, 200e6, 77.3, [6.0], [500.0],
                               pile_length=15.0)
        steps = [i for s in get_calc_steps(dr, {}) for i in s.items]
        first = [i for i in steps
                 if getattr(i, "title", "") == "Drivability Study"][0]
        assert "15.00 m long at every depth" in first.notes
