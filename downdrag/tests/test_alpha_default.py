"""GM-255: a clay's alpha, where the layer leaves it out, is read off the
chart the drag procedure prints instead of 1.0 whatever the clay's su.

UFC 3-220-20 (the module's headline procedure) by default: Table 6-16 (soft
clay, alpha 1), Fig 6-14 (firm to very stiff clay by su/Pa and Z/b) and the
transition zone of pp. 416-417 (Eq 6-9 under soft clay, alpha 1.0 under
sand), as UFC's own drag example reads them (App. B-5.4, pp. 636-637).
FHWA GEC-12 Fig. 7-17 is the named option. A given alpha is kept."""

import numpy as np
import pytest

from downdrag.analysis import ALPHA_CHOICE, DowndragAnalysis
from downdrag.calc_steps import get_tables
from downdrag.soil import DowndragSoilLayer, DowndragSoilProfile
from downdrag.tests.test_downdrag import FT, KIP, PCF, PSF, _f67
from geotech_common import alpha_method as am


def _b54(**kw):
    """UFC 3-220-20 App. B-5.4 (pp. 633-643) with alpha left out: soft clay
    40 ft (su 500 psf) over stiff clay (su 3000 psf), 12 in square piles 65
    ft long, 8 ft of fill; as ``test_downdrag._ufc_b5`` but with the
    transition zone and the stiff clay's alpha read, not given."""
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=40 * FT, soil_type="cohesive",
                          unit_weight=100 * PCF, cu=500 * PSF,
                          settling=True, Cc=0.5, e0=1.5,
                          description="soft clay"),
        DowndragSoilLayer(thickness=51 * FT, soil_type="cohesive",
                          unit_weight=120 * PCF, cu=3000 * PSF,
                          description="stiff clay"),
    ], gwt_depth=0.0)
    args = dict(soil=soil, pile_length=65 * FT, pile_diameter=1 * FT,
                pile_perimeter=4 * FT, pile_area=FT * FT,
                pile_E=4030e3 * 144 * PSF, pile_unit_weight=0.0,
                Q_dead=40 * KIP, fill_thickness=8 * FT,
                fill_unit_weight=120 * PCF, n_sublayers=40)
    args.update(kw)
    return DowndragAnalysis(**args)


def _alpha_choices(result):
    return [c for c in result.choices if c["what"].startswith(ALPHA_CHOICE)]


def _two_clays(top_cu, low_cu, top_alpha=None, **kw):
    """6 m of clay (``top_cu``) over 30 m of clay (``low_cu``), a 0.4 m
    pile 22 m long, 2 m of fill."""
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=6.0, soil_type="cohesive",
                          unit_weight=16.0, cu=top_cu, alpha=top_alpha,
                          settling=True, Cc=0.5, e0=1.3),
        DowndragSoilLayer(thickness=30.0, soil_type="cohesive",
                          unit_weight=19.0, cu=low_cu, Cc=0.15, e0=0.7),
    ], gwt_depth=1.0)
    args = dict(soil=soil, pile_length=22.0, pile_diameter=0.4,
                pile_E=30e6, Q_dead=600.0, fill_thickness=2.0)
    args.update(kw)
    return DowndragAnalysis(**args)


def _fs_at(result, z):
    """The unit skin friction just below depth ``z``."""
    i = int(np.searchsorted(result.z, z, side="right"))
    return float(result.unit_skin_friction[i])


class TestUFCDragExampleReadsItsOwnAlphas:
    """App. B-5.4 takes alpha 1.0 in the soft clay, Eq 6-9 in the 10 ft
    transition zone (720 psf) and 0.5 off Fig 6-14 in the stiff clay. Left
    out, the module now reads the same and reproduces Table B-28 (p. 640);
    before GM-255 it took 1.0 throughout, drag force 816 kN against 413."""

    def test_the_alphas_read(self):
        spans = _b54().alpha_readings()["spans"]
        soft, zone, stiff = spans
        assert (soft["alpha"], soft["source"]) == (1.0, am.UFC_TABLE_SOFT)
        assert zone["source"] == am.UFC_EQ_6_9 and zone["floor"]
        assert zone["alpha"] * 3000 == pytest.approx(720, abs=1)
        assert zone["bottom_m"] - zone["top_m"] == pytest.approx(10 * FT)
        assert stiff["source"] == am.UFC_FIG
        assert stiff["alpha"] == pytest.approx(0.50, abs=0.005)
        assert stiff["Z_over_b"] == pytest.approx(65.0)

    @pytest.mark.parametrize("mob, np_exact, p_exact", [
        (0.0, 47.75, 119.5), (0.5, 50.155, 126.25), (1.0, 52.483, 132.99)])
    def test_table_b28(self, mob, np_exact, p_exact):
        r = _b54(neutral_plane_method="force_equilibrium",
                 toe_mobilization=mob).compute()
        assert r.neutral_plane_depth / FT + 8.0 == pytest.approx(
            np_exact, abs=0.15)
        assert r.max_pile_load / KIP == pytest.approx(p_exact, rel=0.01)

    def test_the_choice_names_each_reading_and_the_option(self):
        (c,) = _alpha_choices(_b54().compute())
        assert c["how"] == "default"
        assert "0.24 at 12.192-15.24 m (Eq 6-9" in c["chosen"]
        assert "0.50 at 15.24-19.812 m (Fig 6-14, Z/b 65)" in c["chosen"]
        for text in ("§6-5.4.2 (pp. 415-417)", "Table 6-16",
                     "App. B-5.4, pp. 636-637", "Eq 6-9"):
            assert text in c["basis"], text
        assert c["alternatives"][0].startswith("FHWA GEC-12 Fig. 7-17")


class TestTheDefault:

    def test_a_soft_clay_keeps_alpha_1(self):
        """F67's clay, su 25 kPa, is soft: unchanged at 543.7 kN."""
        r = _f67().compute()
        assert r.drag_force == pytest.approx(543.7, abs=0.1)

    def test_a_stiff_clay_reads_fig_6_14(self):
        """su 150 kPa over sand-free ground, Z/b 55: curve A."""
        a = _two_clays(150.0, 150.0)
        r = a.compute()
        alpha, src = am.alpha_ufc(150.0, 22.0 / 0.4)
        assert src == am.UFC_FIG and alpha < 0.6
        assert _fs_at(r, 15.0) == pytest.approx(alpha * 150.0)
        assert _fs_at(r, 1.0) == pytest.approx(
            am.alpha_ufc(150.0, 6.0 / 0.4)[0] * 150.0)

    def test_a_given_alpha_is_kept(self):
        r = _two_clays(20.0, 100.0, top_alpha=0.8).compute()
        assert _fs_at(r, 2.0) == pytest.approx(16.0)
        given, read = _alpha_choices(r)
        assert (given["chosen"], given["how"]) == ("alpha 0.80", "given")
        assert given["what"] == "alpha in the cohesive soil, 0-6 m"
        assert read["what"] == "alpha in the cohesive soil, 6-22 m"

    def test_the_transition_zone_under_soft_clay(self):
        """Stiff clay (su 100 kPa) under soft: the top 10b = 4 m takes Eq
        6-9; below, Fig 6-14."""
        r = _two_clays(20.0, 100.0).compute()
        assert _fs_at(r, 7.0) == pytest.approx(
            am.alpha_ufc_eq_6_9(100.0) * 100.0)
        assert _fs_at(r, 12.0) == pytest.approx(
            am.alpha_ufc(100.0, 55.0)[0] * 100.0)
        off = _two_clays(20.0, 100.0, alpha_transition_zone=False).compute()
        assert _fs_at(off, 7.0) == pytest.approx(_fs_at(off, 12.0))
        assert "alpha_transition_zone true" in _alpha_choices(off)[0][
            "alternatives"][1]

    def test_the_zone_takes_the_soft_clays_fs_where_higher(self):
        """UFC p. 417: Eq 6-9 gives 24 kPa under a 37 kPa soft clay
        (alpha 1); the zone takes 37."""
        r = _two_clays(37.0, 75.0).compute()
        assert am.alpha_ufc_eq_6_9(75.0) * 75.0 < 37.0
        assert _fs_at(r, 7.0) == pytest.approx(37.0)

    def test_under_sand_alpha_1_in_the_zone(self):
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=4.0, soil_type="cohesionless",
                              unit_weight=19.0, phi=32.0),
            DowndragSoilLayer(thickness=30.0, soil_type="cohesive",
                              unit_weight=19.0, cu=150.0)], gwt_depth=1.0)
        r = DowndragAnalysis(soil=soil, pile_length=20.0, pile_diameter=0.4,
                             Q_dead=300.0).compute()
        assert _fs_at(r, 5.0) == pytest.approx(150.0)
        assert _fs_at(r, 9.0) == pytest.approx(
            am.alpha_ufc(150.0, 50.0)[0] * 150.0)

    def test_firm_clay_and_a_short_embedment_are_said(self):
        """Firm clay under sand: no zone, a judgment note (p. 417); Z/b
        below 10: curve B, noted."""
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=2.0, soil_type="cohesionless",
                              unit_weight=18.0, phi=30.0),
            DowndragSoilLayer(thickness=10.0, soil_type="cohesive",
                              unit_weight=19.0, cu=60.0)], gwt_depth=1.0)
        r = DowndragAnalysis(soil=soil, pile_length=3.0, pile_diameter=0.4,
                             Q_dead=100.0).compute()
        text = " ".join(r.warnings)
        assert "judgment should be applied" in text
        assert "Z/b = 7.5 is less than curve B's 10" in text


class TestTheGEC12Option:

    def test_ca_off_fig_7_17(self):
        r = _two_clays(20.0, 100.0, alpha_chart="gec12_fig_7_17",
                       fig_7_17_curve="smooth_steel").compute()
        ca = am.adhesion_fig_7_17(100.0, 22.0 / 0.4, am.DASHED)
        assert _fs_at(r, 7.0) == pytest.approx(ca)     # no zone by default
        (c,) = _alpha_choices(r)
        assert c["how"] == "given" and "D/b 55" in c["chosen"]
        assert "smooth steel piles" in c["basis"]

    @pytest.mark.parametrize("kw, words", [
        (dict(alpha_chart="gec12_fig_7_17"), "needs fig_7_17_curve"),
        (dict(fig_7_17_curve="smooth_steel"), "does not take fig_7_17_curve"),
        (dict(alpha_chart="gec12_fig_7_17", fig_7_17_curve="h_pile"),
         "fig_7_17_curve must be one of"),
        (dict(alpha_chart="tomlinson"), "alpha_chart must be one of"),
        (dict(alpha_transition_zone="yes"), "alpha_transition_zone must"),
    ])
    def test_refused(self, kw, words):
        with pytest.raises(ValueError, match=words):
            _two_clays(20.0, 100.0, **kw)


class TestReviewFixes:
    """The GM-255 review (2026-10-20)."""

    def test_the_zone_runs_on_through_sublayers_of_the_stiff_clay(self):
        """UFC p. 415 lets a clay be split into sublayers; the zone is 10b
        from the top of the stiff clay however it is split (4 m here)."""
        split = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=6.0, soil_type="cohesive",
                              unit_weight=16.0, cu=20.0, settling=True,
                              Cc=0.5, e0=1.3)] + [
            DowndragSoilLayer(thickness=t, soil_type="cohesive",
                              unit_weight=19.0, cu=100.0)
            for t in (2.0, 2.0, 2.0, 24.0)], gwt_depth=1.0)
        a = _two_clays(20.0, 100.0, soil=split)
        zone = [s for s in a.alpha_readings()["spans"]
                if s["source"] == am.UFC_EQ_6_9]
        assert [(s["top_m"], s["bottom_m"]) for s in zone] == [
            (6.0, 8.0), (8.0, 10.0)]
        assert all(s["floor"] and s["over_layer"] == 0 for s in zone)
        r = a.compute()
        assert _fs_at(r, 9.0) == pytest.approx(
            am.alpha_ufc_eq_6_9(100.0) * 100.0)
        assert _fs_at(r, 10.5) == pytest.approx(
            am.alpha_ufc(100.0, 12.0 / 0.4)[0] * 100.0)

    def test_the_choice_names_the_length_calls_and_the_way_back(self):
        """Z/b 30 lies between curves B and A: interpolated, said with
        UFC's own words (p. 415); the zone's off switch and a given alpha
        (p. 476) are the alternatives."""
        r = _two_clays(20.0, 100.0, pile_length=12.0).compute()
        (c,) = _alpha_choices(r)
        assert "Z/b 30" in c["chosen"]
        assert ("'engineering judgment should be exercised for how to "
                "consider length effects' (p. 415)") in c["basis"]
        assert "interpolated linearly in Z/b" in c["basis"]
        assert any("alpha_transition_zone false" in a
                   for a in c["alternatives"])
        assert "p. 476" in c["alternatives"][-1]

    @pytest.mark.parametrize("kw", [
        dict(alpha_chart="gec12_fig_7_17", fig_7_17_curve="smooth_steel"),
        dict(alpha_transition_zone=True)])
    def test_alpha_inputs_refused_where_no_clay_reads_a_chart(self, kw):
        """A sand, and a clay that gives its alpha."""
        for layer in (
                DowndragSoilLayer(thickness=30.0, soil_type="cohesionless",
                                  unit_weight=19.0, phi=32.0),
                DowndragSoilLayer(thickness=30.0, soil_type="cohesive",
                                  unit_weight=19.0, cu=100.0, alpha=0.6)):
            soil = DowndragSoilProfile(layers=[layer], gwt_depth=1.0)
            with pytest.raises(ValueError, match="This run does not take"):
                DowndragAnalysis(soil=soil, pile_length=12.0,
                                 pile_diameter=0.4, **kw)

    def test_the_zone_switch_refused_where_no_zone_falls(self):
        """Firm clay under soft clay takes no zone (p. 417)."""
        with pytest.raises(ValueError, match="does not take "
                           "alpha_transition_zone"):
            _two_clays(20.0, 55.0, alpha_transition_zone=False)

    def test_a_beta_clay_above_with_no_su_is_said(self):
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=6.0, soil_type="cohesive",
                              unit_weight=16.0, skin_friction_method="beta",
                              beta=0.25, settling=True, Cc=0.5, e0=1.3),
            DowndragSoilLayer(thickness=30.0, soil_type="cohesive",
                              unit_weight=19.0, cu=100.0)], gwt_depth=1.0)
        r = _two_clays(20.0, 100.0, soil=soil).compute()
        assert any("gives no su, so whether it is soft clay" in w
                   for w in r.warnings)

    def test_gec12_names_fig_7_18_and_its_shallowest_curve(self):
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=2.0, soil_type="cohesionless",
                              unit_weight=18.0, phi=30.0),
            DowndragSoilLayer(thickness=10.0, soil_type="cohesive",
                              unit_weight=19.0, cu=100.0)], gwt_depth=1.0)
        r = DowndragAnalysis(soil=soil, pile_length=3.0, pile_diameter=0.4,
                             Q_dead=100.0, alpha_chart="gec12_fig_7_17",
                             fig_7_17_curve="smooth_steel").compute()
        text = " ".join(r.warnings)
        assert "GEC-12 p. 246 recommends Figure 7-18" in text
        assert "D/b = 7.5 is less than the 10" in text
        (c,) = _alpha_choices(r)
        assert "D/b 7.5 (the D = 10b curve)" in c["chosen"]


def test_the_calc_package_shows_alpha_as_read():
    r = _two_clays(20.0, 100.0)
    rows = next(t for t in get_tables(r.compute(), r)
                if t.name == "soil_layers").rows
    assert rows[0][9] == "1.00 (Table 6-16)"
    assert rows[1][9].startswith("0.28 (Eq 6-9) / 0.")
    assert rows[1][9].endswith(" (Fig 6-14)")
