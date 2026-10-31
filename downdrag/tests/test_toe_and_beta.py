"""GM-283: the toe's Nt and the sand's beta as UFC 3-220-20 and GEC-12 print
them.

Nt in cohesionless soil comes from UFC 3-220-20 Table 6-17 (p. 421): its
Fellenius (2021) column, the same values as FHWA GEC-12 Table 7-9 (the
default), or its equation after Cheng (2004) (``Nt_method="cheng"``).
beta in cohesionless soil comes from GEC-12 Table 7-9 (UFC Table 6-15).
Both replaced an unprinted curve and formula. The profiles are made up;
the Table 6-17 cells are the printed ones.
"""

import math

import pytest

from downdrag import DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile
from downdrag.analysis import BETA_CHOICE, NT_CHOICE, cheng_Nq
from downdrag.soil import fellenius_table

GAMMA_W = 9.81


def _toe_in_sand(phi=34.0, pile_length=12.0, pile_diameter=0.4, **kw):
    """8 m of clay over sand, water at the surface, no fill; the toe is
    ``pile_length - 8`` m into the sand. ``kw`` goes to the sand layer
    (``soil_class``, ``beta``) or, with ``a_`` stripped, to the
    analysis."""
    a_kw = {k[2:]: v for k, v in kw.items() if k.startswith("a_")}
    s_kw = {k: v for k, v in kw.items() if not k.startswith("a_")}
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=8.0, soil_type="cohesive",
                          unit_weight=17.0, cu=20.0, alpha=1.0),
        DowndragSoilLayer(thickness=10.0, soil_type="cohesionless",
                          unit_weight=19.0, phi=phi, **s_kw),
    ], gwt_depth=0.0)
    return DowndragAnalysis(soil=soil, pile_length=pile_length,
                            pile_diameter=pile_diameter, Q_dead=200.0,
                            **a_kw)


def _sigma_toe(pile_length=12.0):
    return 8.0 * (17.0 - GAMMA_W) + (pile_length - 8.0) * (19.0 - GAMMA_W)


def _choice(result, what):
    (c,) = [c for c in result.choices if c["what"].startswith(what)]
    return c


# UFC 3-220-20 Table 6-17 (p. 421), "Based on Cheng (2004)": the ends of
# each soil type's range at Z/b = 10, 20 and 40, as printed (whole numbers).
TABLE_6_17_CHENG = [
    # Z/b, phi', printed Nq
    (10, 25, 12), (10, 30, 26), (10, 28, 19), (10, 34, 52),
    (10, 32, 37), (10, 40, 168),
    (20, 25, 10), (20, 30, 22), (20, 28, 16), (20, 34, 46),
    (20, 32, 32), (20, 40, 152),
    (40, 25, 8), (40, 30, 19), (40, 28, 14), (40, 34, 40),
    (40, 32, 27), (40, 40, 139),
]


@pytest.mark.parametrize("zb, phi, printed", TABLE_6_17_CHENG)
def test_the_cheng_equation_gives_table_6_17s_columns(zb, phi, printed):
    """The equation under Table 6-17 reproduces every Cheng (2004) cell the
    table prints, to the whole number it prints."""
    assert abs(cheng_Nq(zb, phi) - printed) <= 0.5


def test_the_cheng_equation_as_printed():
    """Nq ~= 1.2 (Z/b)^-0.437 exp[6.34 (Z/b)^0.0486 tan phi']."""
    zb, phi = 30.0, 34.0
    hand = 1.2 * zb ** -0.437 * math.exp(
        6.34 * zb ** 0.0486 * math.tan(math.radians(phi)))
    assert cheng_Nq(zb, phi) == pytest.approx(hand, rel=1e-12)


class TestTheToe:

    def test_sand_by_table_7_9_by_default(self):
        """phi' 34 deg in sand: Nt = 30 + (34 - 32)/(40 - 32) x 120 = 60;
        the toe resistance 60 sigma'v A."""
        a = _toe_in_sand()
        r = a.compute()
        area = math.pi / 4.0 * 0.4 ** 2
        assert a.toe_reading()["Nt"] == pytest.approx(60.0, abs=1e-12)
        assert r.toe_resistance == pytest.approx(
            60.0 * _sigma_toe() * area, rel=1e-9)
        c = _choice(r, NT_CHOICE)
        assert c["how"] == "default"
        assert c["chosen"].startswith("Nt 60.0 (UFC 3-220-20 Table 6-17, "
                                      "p. 421, Fellenius (2021) column")
        assert "GEC-12 Table 7-9" in c["chosen"]
        # the other printed route, with its value
        nt_c = cheng_Nq(12.0 / 0.4, 34.0)
        assert c["alternatives"][0].startswith(f"Nt {nt_c:.1f} (UFC 3-220-20 "
                                               f"Table 6-17, p. 421, Cheng "
                                               f"(2004) equation")
        assert c["alternatives"][0].endswith("Nt_method 'cheng'")
        # the line between Table 6-17's ends is the module's, not printed;
        # the ends are the printed alternatives (GM-283 review)
        assert ("(sand: Nt 30 to 150 over phi' 32 to 40 deg)"
                in c["basis"])
        assert "this module's reading, not printed" in c["basis"]
        assert c["alternatives"][1:3] == [
            "Nt 30, the sand range's low end in Table 6-17 (phi' 32 deg): "
            "give Nt 30",
            "Nt 150, the sand range's high end in Table 6-17 (phi' 40 deg): "
            "give Nt 150"]

    def test_cheng_on_request(self):
        a = _toe_in_sand(a_Nt_method="cheng")
        r = a.compute()
        nt = cheng_Nq(30.0, 34.0)
        area = math.pi / 4.0 * 0.4 ** 2
        assert a.toe_reading()["Nt"] == pytest.approx(nt, rel=1e-12)
        assert r.toe_resistance == pytest.approx(nt * _sigma_toe() * area,
                                                 rel=1e-9)
        c = _choice(r, NT_CHOICE)
        assert c["how"] == "given"
        assert "Z/b 30.0" in c["chosen"]
        assert c["alternatives"][0].startswith("Nt 60.0")

    def test_cheng_is_held_at_200(self):
        """phi' 42 deg at Z/b 10: the equation gives about 260; Table 6-17
        limits it to 200 'unless supported by project-specific
        information', and the result says so."""
        a = _toe_in_sand(phi=42.0, pile_length=9.0, pile_diameter=0.9,
                         a_Nt_method="cheng")
        assert cheng_Nq(10.0, 42.0) > 200.0
        r = a.compute()
        assert a.toe_reading()["Nt"] == 200.0
        assert any("held at 200" in w and "project-specific" in w
                   for w in r.warnings)

    @pytest.mark.parametrize("soil_class, phi, nt, beta", [
        ("silt", 31.0, 30.0, 0.375),      # 20 + 3/6 x 20; 0.25 + 3/6 x 0.25
        ("gravel", 40.0, 180.0, 0.575),   # 60 + 5/10 x 240; 0.35 + 0.225
        ("Sand", 38.0, 120.0, 0.75),      # 30 + 6/8 x 120; 0.30 + 0.45
    ])
    def test_the_soil_class_picks_the_row(self, soil_class, phi, nt, beta):
        a = _toe_in_sand(phi=phi, soil_class=soil_class)
        assert a.toe_reading()["Nt"] == pytest.approx(nt, abs=1e-9)
        assert a.soil.layers[1].beta == pytest.approx(beta, abs=1e-12)
        assert a.soil.layers[1].soil_class == soil_class.lower()

    def test_a_phi_outside_the_row_is_held_at_its_end_and_said(self):
        """Gravel at 33 deg: the row starts at 35 deg, so Nt 60 and beta
        0.35, warned for both."""
        a = _toe_in_sand(phi=33.0, soil_class="gravel")
        r = a.compute()
        assert a.toe_reading()["Nt"] == 60.0
        assert a.soil.layers[1].beta == 0.35
        held = [w for w in r.warnings if "gravel range (35-45 deg)" in w]
        assert len(held) == 2

    def test_a_given_nt_wins(self):
        a = _toe_in_sand(a_Nt=25.0)
        r = a.compute()
        assert a.toe_reading()["Nt"] == 25.0
        c = _choice(r, NT_CHOICE)
        assert (c["chosen"], c["how"]) == ("Nt 25", "given")

    def test_a_clay_toe_takes_nine(self):
        """N*c = 9 (UFC 3-220-20 p. 424), as its drag example takes it."""
        a = _toe_in_sand(pile_length=6.0)
        r = a.compute()
        area = math.pi / 4.0 * 0.4 ** 2
        assert r.toe_resistance == pytest.approx(9.0 * 20.0 * area,
                                                 rel=1e-9)
        c = _choice(r, NT_CHOICE)
        assert "p. 424" in c["chosen"] and "p. 637" in c["chosen"]

    def test_a_bad_nt_method_is_refused(self):
        with pytest.raises(ValueError, match="Nt_method must be one of"):
            _toe_in_sand(a_Nt_method="vesic")


class TestTheSandBeta:

    def test_beta_from_table_7_9(self):
        """beta 0.30 + (34 - 32)/8 x 0.60 = 0.45 in the sand; fs = beta
        sigma'v there."""
        a = _toe_in_sand()
        r = a.compute()
        sand = (r.z > 8.0 + 1e-9)
        sigma = 8.0 * (17.0 - GAMMA_W) + (r.z[sand] - 8.0) * (19.0 - GAMMA_W)
        assert all(abs(f - 0.45 * s) < 1e-9
                   for f, s in zip(r.unit_skin_friction[sand], sigma))
        c = _choice(r, BETA_CHOICE)
        assert c["what"] == "beta in the cohesionless soil, 8-12 m"
        assert c["chosen"] == "beta 0.450 at 8-12 m (sand, phi' 34 deg)"
        assert c["how"] == "default"
        for text in ("Table 7-9 (p. 256; Step 2b, p. 257)",
                     "Table 6-15 (p. 414) pairs beta's ends with phi'",
                     "(sand: beta 0.30 at phi' 32 deg, 0.90 at 40 deg)",
                     "this module's reading; neither manual prints it",
                     "§6-7.4 step 2 (p. 476)"):
            assert text in c["basis"], text
        # the printed range ends, as given betas (GM-283 review)
        assert c["alternatives"][:2] == [
            "beta 0.30, the sand range's low end in Table 6-15 (phi' 32 "
            "deg), conservative for resistance: give the layer beta 0.30",
            "beta 0.90, the sand range's high end in Table 6-15 (phi' 40 "
            "deg), conservative for negative skin friction: give the layer "
            "beta 0.90"]

    def test_a_given_beta_is_kept(self):
        a = _toe_in_sand(beta=0.33)
        r = a.compute()
        assert a.soil.layers[1].beta == 0.33
        assert a.soil.layers[1].beta_from_table is None
        c = _choice(r, BETA_CHOICE)
        assert (c["chosen"], c["how"]) == ("beta 0.330 at 8-12 m", "given")

    def test_contiguous_layers_alike_read_as_one(self):
        """Two sand layers at phi' 30 deg, both held at the row's 32 deg:
        one span, one warning; the sand at 36 deg below is its own."""
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=3.0, soil_type="cohesionless",
                              unit_weight=18.0, phi=30.0),
            DowndragSoilLayer(thickness=4.0, soil_type="cohesionless",
                              unit_weight=19.0, phi=30.0),
            DowndragSoilLayer(thickness=9.0, soil_type="cohesionless",
                              unit_weight=20.0, phi=36.0),
        ], gwt_depth=2.0)
        a = DowndragAnalysis(soil=soil, pile_length=12.0,
                             pile_diameter=0.4, Q_dead=300.0)
        read = a.beta_readings()
        assert [(s["top_m"], s["bottom_m"], s["beta"])
                for s in read["spans"]] == [(0.0, 7.0, 0.30),
                                            (7.0, 12.0, pytest.approx(0.60))]
        (note,) = read["notes"]
        assert note.startswith("phi' = 30 deg at 0-7 m is outside")
        r = a.compute()
        assert sum("sand range (32-40 deg)" in w for w in r.warnings) == 1

    def test_the_table_reads_linearly(self):
        assert fellenius_table("sand", 36.0)[:2] == pytest.approx(
            (0.60, 90.0))
        assert fellenius_table("silt", 50.0) == (0.50, 40.0, 34.0)

    def test_a_cohesive_layer_does_not_take_soil_class(self):
        with pytest.raises(ValueError, match="does not take soil_class"):
            DowndragSoilLayer(thickness=5.0, soil_type="cohesive",
                              unit_weight=17.0, cu=20.0, soil_class="silt")

    def test_an_unknown_soil_class_is_refused(self):
        with pytest.raises(ValueError, match="soil_class of a cohesionless "
                                             "layer must be one of"):
            DowndragSoilLayer(thickness=5.0, soil_type="cohesionless",
                              unit_weight=19.0, phi=33.0, soil_class="clay")
