"""Terzaghi (1943) bearing capacity (BACKLOG GM-30).

Pins, in order of authority:
1. UFC 3-220-20 (2025) Table 5-2, Terzaghi columns (Nc, Nq, and Ngamma by
   the Coduto et al. 2016 approximation), read from the PDF in the
   reference library, to the table's own rounding; Table 5-5 shape factors;
   Table 5-4 local shear.
2. The table of Terzaghi's factors in Das, Principles of Foundation
   Engineering (Ngamma after Kumbhojkar 1993), cross-checked row by row
   against an independent reproduction (not against the book in hand).
3. Hand calculations from those factors.
"""

import math

import pytest

from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (
    TERZAGHI_NGAMMA_SOURCES,
    terzaghi_Nc,
    terzaghi_Ngamma,
    terzaghi_Nq,
    terzaghi_local_shear,
    terzaghi_shape_factors,
)
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer

# phi, Nc, Nq, Ngamma -- Das, Terzaghi's bearing capacity factors (Ngamma
# after Kumbhojkar 1993).
DAS = [
    (0, 5.70, 1.00, 0.00), (1, 6.00, 1.10, 0.01), (2, 6.30, 1.22, 0.04),
    (3, 6.62, 1.35, 0.06), (4, 6.97, 1.49, 0.10), (5, 7.34, 1.64, 0.14),
    (6, 7.73, 1.81, 0.20), (7, 8.15, 2.00, 0.27), (8, 8.60, 2.21, 0.35),
    (9, 9.09, 2.44, 0.44), (10, 9.61, 2.69, 0.56), (11, 10.16, 2.98, 0.69),
    (12, 10.76, 3.29, 0.85), (13, 11.41, 3.63, 1.04), (14, 12.11, 4.02, 1.26),
    (15, 12.86, 4.45, 1.52), (16, 13.68, 4.92, 1.82), (17, 14.60, 5.45, 2.18),
    (18, 15.12, 6.04, 2.59), (19, 16.56, 6.70, 3.07), (20, 17.69, 7.44, 3.64),
    (21, 18.92, 8.26, 4.31), (22, 20.27, 9.19, 5.09), (23, 21.75, 10.23, 6.00),
    (24, 23.36, 11.40, 7.08), (25, 25.13, 12.72, 8.34), (26, 27.09, 14.21, 9.84),
    (27, 29.24, 15.90, 11.60), (28, 31.61, 17.81, 13.70),
    (29, 34.24, 19.98, 16.18), (30, 37.16, 22.46, 19.13),
    (31, 40.41, 25.28, 22.65), (32, 44.04, 28.52, 26.87),
    (33, 48.09, 32.23, 31.94), (34, 52.64, 36.50, 38.04),
    (35, 57.75, 41.44, 45.41), (36, 63.53, 47.16, 54.36),
    (37, 70.01, 53.80, 65.27), (38, 77.50, 61.55, 78.61),
    (39, 85.97, 70.61, 95.03), (40, 95.66, 81.27, 115.31),
    (41, 106.81, 93.85, 140.51), (42, 119.67, 108.75, 171.99),
    (43, 134.58, 126.50, 211.56), (44, 151.95, 147.74, 261.60),
    (45, 172.28, 173.28, 325.34), (46, 196.22, 204.19, 407.11),
    (47, 224.55, 241.80, 512.84), (48, 258.28, 287.85, 650.67),
    (49, 298.71, 344.63, 831.99), (50, 347.50, 415.14, 1072.80),
]

# phi: (Nc, Nq, Ngamma) Terzaghi columns of UFC 3-220-20 (2025) Table 5-2,
# p. 289, as printed (strings keep the printed precision).
UFC = {
    0: ("5.7", "1.0", "0"), 2: ("6.3", "1.2", "0.15"), 4: ("7.0", "1.5", "0.31"),
    6: ("7.7", "1.8", "0.51"), 8: ("8.6", "2.2", "0.74"),
    10: ("9.6", "2.7", "1.0"), 12: ("11", "3.3", "1.4"), 14: ("12", "4.0", "1.9"),
    16: ("14", "4.9", "2.5"), 18: ("15", "6.0", "3.3"), 20: ("18", "7.4", "4.4"),
    21: ("19", "8.3", "5.1"), 22: ("20", "9.2", "5.9"), 23: ("22", "10.", "6.8"),
    24: ("23", "11", "7.9"), 25: ("25", "13", "9.2"), 26: ("27", "14", "11"),
    27: ("29", "17", "12"), 28: ("32", "18", "15"), 29: ("34", "20", "17"),
    30: ("37", "22", "20"), 31: ("40", "25", "24"), 32: ("44", "29", "28"),
    33: ("48", "32", "33"), 34: ("53", "37", "40"), 35: ("58", "41", "47"),
    36: ("64", "47", "57"), 37: ("70", "54", "68"), 38: ("78", "62", "82"),
    39: ("86", "71", "100"), 40: ("96", "81", "120"), 41: ("110", "94", "150"),
    42: ("120", "110", "180"), 43: ("130", "130", "230"),
    44: ("160", "150", "280"), 45: ("170", "170", "350"),
    46: ("200", "200", "440"), 47: ("220", "240", "550"),
    48: ("260", "290", "700"), 49: ("300", "350", "890"),
    50: ("350", "420", "1150"),
}


def _half_unit(printed: str) -> float:
    """Half a unit in the last printed place. Whole numbers of three or more
    digits are printed to two or three significant figures (unit 10)."""
    if "." in printed:
        decimals = len(printed.split(".")[1])
        return 0.5 * 10 ** (-decimals) if decimals else 0.5
    return 5.0 if len(printed) >= 3 else 0.5


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi, Nc, Nq, Ng", DAS)
def test_nc_nq_match_das_table(phi, Nc, Nq, Ng):
    """Terzaghi's closed-form Nc and Nq reproduce Das's table to 0.5 %,
    except Nc at 18 deg, printed 15.12 where the formula (and the smooth
    run of its neighbours, 14.60 and 16.56) gives 15.52 -- a misprint in
    the table."""
    assert terzaghi_Nq(phi) == pytest.approx(Nq, rel=0.005, abs=0.005)
    if phi == 18:
        assert terzaghi_Nc(18) == pytest.approx(15.52, abs=0.01)
    else:
        assert terzaghi_Nc(phi) == pytest.approx(Nc, rel=0.005)


@pytest.mark.parametrize("phi, Nc, Nq, Ng", DAS)
def test_kumbhojkar_ngamma_is_the_table(phi, Nc, Nq, Ng):
    assert terzaghi_Ngamma(phi, "kumbhojkar") == pytest.approx(Ng, abs=1e-9)


#: Entries of UFC 3-220-20 Table 5-2 that are not the rounded closed form.
#: Each is checked against Das's value instead; the formula agrees with Das.
UFC_MISPRINTS = {
    (18, "Nc"): 15.52,   # printed 15: Das's 15.12 misprint carried over
    (27, "Nq"): 15.90,   # printed 17; neighbours 14 (26) and 18 (28)
    (44, "Nc"): 151.95,  # printed 160; rounds to 150
    (49, "Nq"): 344.63,  # printed 350; rounds to 340 (345 rounded twice?)
}


@pytest.mark.parametrize("phi", sorted(UFC))
def test_factors_match_ufc_table_5_2(phi):
    """Every Terzaghi entry of UFC 3-220-20 Table 5-2 to its printed
    rounding: Nc and Nq closed form, Ngamma by the Coduto approximation.
    Four printed Nc/Nq entries are misprints (``UFC_MISPRINTS``)."""
    nc, nq, ng = UFC[phi]
    for name, printed, value in (("Nc", nc, terzaghi_Nc(phi)),
                                 ("Nq", nq, terzaghi_Nq(phi)),
                                 ("Ngamma", ng, terzaghi_Ngamma(phi, "coduto"))):
        if (phi, name) in UFC_MISPRINTS:
            assert value == pytest.approx(UFC_MISPRINTS[(phi, name)], rel=0.001)
            assert abs(value - float(printed)) > _half_unit(printed)
        else:
            # +1e-3 relative: 77.50 printed as 78 sits on the rounding edge
            tol = _half_unit(printed) + 1e-3 * float(printed) + 1e-9
            assert abs(value - float(printed)) <= tol, (name, printed, value)


def test_phi_zero():
    assert terzaghi_Nc(0) == pytest.approx(1.5 * math.pi + 1)  # 5.71
    assert terzaghi_Nq(0) == 1.0
    for src in TERZAGHI_NGAMMA_SOURCES:
        assert terzaghi_Ngamma(0, src) == 0.0


def test_kumbhojkar_between_whole_degrees():
    """Log-linear between whole degrees: 30.5 deg is the geometric mean of
    the 30 and 31 deg entries, and the curve rises monotonically."""
    assert terzaghi_Ngamma(30.5) == pytest.approx(math.sqrt(19.13 * 22.65))
    assert terzaghi_Ngamma(0.5) == pytest.approx(0.005)
    vals = [terzaghi_Ngamma(p / 4) for p in range(0, 201)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_kumbhojkar_table_ends_at_50():
    with pytest.raises(ValueError, match="coduto"):
        terzaghi_Ngamma(51, "kumbhojkar")
    assert terzaghi_Ngamma(51, "coduto") > terzaghi_Ngamma(50, "coduto")
    with pytest.raises(ValueError, match="Unknown Terzaghi"):
        terzaghi_Ngamma(30, "vesic")


def test_shape_factors_ufc_table_5_5():
    assert terzaghi_shape_factors("strip", 2, 1e6)[:3] == (1.0, 1.0, 1.0)
    assert terzaghi_shape_factors("square", 2, 2)[:3] == (1.3, 1.0, 0.8)
    assert terzaghi_shape_factors("circular", 2, 2)[:3] == (1.3, 1.0, 0.6)
    sc, sq, sg, note = terzaghi_shape_factors("rectangular", 2, 4)
    assert (sc, sq, sg) == pytest.approx((1.15, 1.0, 0.9))
    assert "interpolated" in note and "5-3.3.1" in note


def test_local_shear_ufc_table_5_4():
    c, phi = terzaghi_local_shear(30.0, 35.0)
    assert c == pytest.approx(20.0)
    assert phi == pytest.approx(math.degrees(math.atan(2 / 3 * math.tan(
        math.radians(35)))))
    assert phi == pytest.approx(25.02, abs=0.01)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _run(B=2.0, shape="square", Df=1.5, phi=25.0, c=20.0, gamma=16.5,
         length=None, **kw):
    footing = Footing(width=B, length=length, depth=Df, shape=shape)
    soil = BearingSoilProfile(layer1=SoilLayer(
        cohesion=c, friction_angle=phi, unit_weight=gamma))
    return BearingCapacityAnalysis(footing=footing, soil=soil,
                                   factor_method="terzaghi", **kw).compute()


def test_square_footing_hand_calc():
    """2 m square, Df = 1.5 m, phi' = 25, c' = 20 kPa, gamma = 16.5 kN/m3
    (the classic textbook case): from the tabulated factors
    qu = 1.3(20)(25.13) + (1.5)(16.5)(12.72) + 0.4(16.5)(2)(8.34)
       = 653.38 + 314.82 + 110.09 = 1078.3 kPa; qall = 359.4 kPa at FS 3."""
    r = _run()
    assert r.q_ultimate == pytest.approx(1078.29, rel=0.001)
    assert r.term_cohesion == pytest.approx(653.38, rel=0.001)
    assert r.term_overburden == pytest.approx(314.82, rel=0.001)
    assert r.term_selfweight == pytest.approx(110.09, rel=0.001)
    assert r.q_allowable == pytest.approx(1078.29 / 3, rel=0.001)
    d = r.to_dict()
    assert d["method"] == "Terzaghi (1943)"
    assert d["ngamma_method"] == "kumbhojkar"
    assert d["Ngamma_alternatives"]["coduto"] == pytest.approx(9.18, abs=0.01)
    assert "Kumbhojkar (1993)" in d["notes"][0] and "coduto" in d["notes"][0]
    for k in ("dc", "dq", "dgamma", "ic", "bc", "gc"):
        assert d[k] == 1.0


def test_strip_and_circular_coefficients():
    """Strip: c Nc + q Nq + 0.5 gamma B Ngamma; circular: 1.3 c Nc + q Nq +
    0.3 gamma B Ngamma (B = diameter)."""
    Nc, Nq, Ng = 37.16, 22.46, 19.13
    strip = _run(B=1.5, shape="strip", Df=1.0, phi=30.0, c=5.0, gamma=18.0)
    assert strip.q_ultimate == pytest.approx(
        5 * Nc + 18 * Nq + 0.5 * 18 * 1.5 * Ng, rel=0.001)
    circ = _run(B=1.5, shape="circular", Df=1.0, phi=30.0, c=5.0, gamma=18.0)
    assert circ.q_ultimate == pytest.approx(
        1.3 * 5 * Nc + 18 * Nq + 0.3 * 18 * 1.5 * Ng, rel=0.001)


def test_undrained_clay():
    """phi = 0: square qu = 1.3 cu (5.7) + q."""
    r = _run(phi=0.0, c=50.0, gamma=18.0, Df=1.0)
    assert r.q_ultimate == pytest.approx(1.3 * 50 * (1.5 * math.pi + 1) + 18.0)
    assert r.Ngamma == 0.0


def test_coduto_option_and_alternative():
    r = _run(phi=30.0, c=0.0, ngamma_method="coduto")
    assert r.Ngamma == pytest.approx(20.12, abs=0.01)
    assert r.Ngamma_alternatives == {"kumbhojkar": pytest.approx(19.13)}
    assert r.to_dict()["ngamma_method"] == "coduto"


def test_local_shear():
    """phi = 35, c = 30: factors at phi* = 25.02 deg, c* = 20 kPa."""
    r = _run(phi=35.0, c=30.0, failure_mode="local")
    phi_s = math.degrees(math.atan(2 / 3 * math.tan(math.radians(35))))
    assert r.Nc == pytest.approx(terzaghi_Nc(phi_s))
    assert r.term_cohesion == pytest.approx(1.3 * 20.0 * terzaghi_Nc(phi_s))
    d = r.to_dict()
    assert d["failure_mode"] == "local"
    assert d["c_used_kPa"] == pytest.approx(20.0)
    assert d["phi_used_deg"] == pytest.approx(phi_s, abs=1e-3)
    assert any("Local shear" in n for n in d["notes"])
    assert r.q_ultimate < _run(phi=35.0, c=30.0).q_ultimate


def test_rectangle_interpolates_with_a_note():
    r = _run(B=2.0, length=4.0, shape="rectangular")
    assert (r.sc, r.sgamma) == pytest.approx((1.15, 0.9))
    assert any("interpolated" in n for n in r.notes)


def test_deep_footing_note():
    r = _run(B=1.0, Df=2.0)
    assert any("Df/B = 2.00 > 1" in n for n in r.notes)


def test_two_layer_runs_on_terzaghi():
    footing = Footing(width=2.0, depth=1.0, shape="square")
    soil = BearingSoilProfile(
        layer1=SoilLayer(friction_angle=34, unit_weight=19, thickness=1.0),
        layer2=SoilLayer(cohesion=25, friction_angle=0, unit_weight=17))
    r = BearingCapacityAnalysis(footing=footing, soil=soil,
                                factor_method="terzaghi").compute()
    assert r.is_two_layer
    assert min(r.q_upper_layer, r.q_lower_layer) <= r.q_ultimate \
        <= max(r.q_upper_layer, r.q_lower_layer)


@pytest.mark.parametrize("kw, match", [
    ({"load_inclination": 10.0}, "load_inclination"),
    ({"ground_slope": 5.0}, "ground_slope"),
    ({"ngamma_method": "vesic"}, "kumbhojkar"),
])
def test_terzaghi_refuses_what_it_does_not_cover(kw, match):
    with pytest.raises(ValueError, match=match):
        _run(**kw)


def test_tilted_base_refused():
    footing = Footing(width=2.0, depth=1.0, shape="square", base_tilt=5.0)
    soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30,
                                               unit_weight=18))
    with pytest.raises(ValueError, match="base_tilt"):
        BearingCapacityAnalysis(footing=footing, soil=soil,
                                factor_method="terzaghi")


def test_terzaghi_options_stay_with_terzaghi():
    footing = Footing(width=2.0, depth=1.0, shape="square")
    soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30,
                                               unit_weight=18))
    with pytest.raises(ValueError, match="factor_method='terzaghi'"):
        BearingCapacityAnalysis(footing=footing, soil=soil,
                                ngamma_method="kumbhojkar")
    with pytest.raises(ValueError, match="local"):
        BearingCapacityAnalysis(footing=footing, soil=soil,
                                failure_mode="local")
    with pytest.raises(ValueError, match="factor_method"):
        BearingCapacityAnalysis(footing=footing, soil=soil,
                                factor_method="prandtl")


def test_default_method_unchanged():
    """Vesic stays the default, with no Terzaghi keys in its result."""
    footing = Footing(width=2.0, depth=1.5, shape="square")
    soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30,
                                               unit_weight=18))
    a = BearingCapacityAnalysis(footing=footing, soil=soil)
    assert a.ngamma_method == "vesic" and a.factor_method == "vesic"
    d = a.compute().to_dict()
    assert "method" not in d and "notes" not in d
    assert "Ngamma_alternatives" not in d


def test_summary_and_calc_steps():
    pytest.importorskip("matplotlib")
    from bearing_capacity.calc_steps import get_calc_steps, get_input_summary
    footing = Footing(width=2.0, depth=1.5, shape="square")
    soil = BearingSoilProfile(layer1=SoilLayer(
        cohesion=20, friction_angle=25, unit_weight=16.5))
    a = BearingCapacityAnalysis(footing=footing, soil=soil,
                                factor_method="terzaghi")
    r = a.compute()
    assert "Terzaghi (1943)" in r.summary()
    titles = [s.title for s in get_calc_steps(r, a)]
    assert "Bearing Capacity Factors (Terzaghi 1943)" in titles
    assert "Correction Factors" not in titles
    inputs = {i.name: i.value for i in get_input_summary(r, a)}
    assert "Terzaghi" in inputs["Method"]
