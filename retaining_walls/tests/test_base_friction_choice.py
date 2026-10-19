"""BACKLOG GM-173: the cantilever wall states its base friction.

The sliding check takes delta_b = 2/3 phi_f, phi_f the foundation soil's
(the backfill's when not given: GM-165's declared absent case, in the
tool's defaults_used); wave w13 F55's answer turned on it
(sliding 1.07 at 2/3 phi_f against about 1.7 at delta_b = phi_f, concrete
cast against the ground). The result's ``choices`` now say which was used,
with its basis, and the other reading with its own."""

import math

import pytest

from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.geometry import CantileverWallGeometry

_WHAT = "base friction delta_b (sliding)"


def _f55_wall():
    return CantileverWallGeometry(
        wall_height=4.5, base_width=2.8, toe_length=0.7,
        stem_thickness_top=0.35, stem_thickness_base=0.35,
        base_thickness=0.5, surcharge=10.0)


def _friction(result):
    (c,) = [c for c in result.choices if c["what"] == _WHAT]
    return c


def test_the_default_is_two_thirds_of_phi_f_and_says_so():
    # phi_f is the backfill's here; the tool's defaults_used says so
    # (GM-165), and the choice reads the same as with it given
    r = analyze_cantilever_wall(_f55_wall(), 19.0, 32.0)
    c = _friction(r)
    assert c["how"] == "default"
    assert c["chosen"] == ("21.3 deg = 2/3 phi_f, phi_f = 32 deg, the "
                           "foundation soil's friction angle "
                           "(phi_foundation)")
    assert r.choices == analyze_cantilever_wall(
        _f55_wall(), 19.0, 32.0, phi_foundation=32.0).choices
    assert r.sliding_terms["delta_b"] == pytest.approx(2.0 / 3.0 * 32.0)
    # the basis, cited
    assert "EN 1997-1 Sec. 6.5.3(10) (p. 64)" in c["basis"]
    assert "smooth precast foundations" in c["basis"]
    assert "Fig. 4-27 (p. 220)" in c["basis"]
    assert "Table 4-1 (p. 184)" in c["basis"]
    # the other reading, cited, and how to run it
    (alt,) = c["alternatives"]
    assert alt.startswith("phi_f = 32 deg (delta_base = 32): concrete "
                          "cast against the ground")
    assert "GEC-6 Sec. 5.4 (p. 99)" in alt
    assert "cast-in-situ concrete foundations" in alt


def test_a_foundation_phi_given_is_the_one_used():
    r = analyze_cantilever_wall(_f55_wall(), 19.0, 32.0, phi_foundation=30.0)
    c = _friction(r)
    assert c["chosen"] == ("20.0 deg = 2/3 phi_f, phi_f = 30 deg, the "
                           "foundation soil's friction angle "
                           "(phi_foundation)")
    assert c["alternatives"][0].startswith("phi_f = 30 deg (delta_base = 30)")


def test_the_alternative_runs_as_it_says():
    # F55: 1.07 at the default, about 1.7 at delta_b = phi_f
    default = analyze_cantilever_wall(_f55_wall(), 19.0, 32.0,
                                      phi_foundation=32.0)
    cast = analyze_cantilever_wall(_f55_wall(), 19.0, 32.0,
                                   phi_foundation=32.0, delta_base=32.0)
    assert default.FOS_sliding == pytest.approx(1.07, abs=0.005)
    assert cast.FOS_sliding == pytest.approx(
        default.FOS_sliding * math.tan(math.radians(32.0))
        / math.tan(math.radians(2.0 / 3.0 * 32.0)), abs=0.002)
    assert cast.FOS_sliding == pytest.approx(1.72, abs=0.01)
    c = _friction(cast)
    assert c["how"] == "given"
    assert c["chosen"] == "32 deg (delta_base)"
    assert c["alternatives"][0].startswith(
        "2/3 phi_f = 21.3 deg, the module's default")
    assert c["alternatives"][1].startswith("phi_f = 32 deg (delta_base = 32)")


def test_the_choice_is_in_the_result_dict_and_summary():
    r = analyze_cantilever_wall(_f55_wall(), 19.0, 32.0)
    assert _WHAT in [c["what"] for c in r.to_dict()["choices"]]
    assert "base friction delta_b (sliding): 21.3 deg" in r.summary()


def test_a_printed_table_reads_plain_words():
    from geotech_common.choices import choice_rows
    r = analyze_cantilever_wall(_f55_wall(), 19.0, 32.0, phi_foundation=32.5)
    (row,) = [row for row in choice_rows(r.choices) if row[0] == _WHAT]
    assert row[1] == ("21.7 deg = 2/3 phi_f, phi_f = 32.5 deg, the "
                      "foundation soil's friction angle")
    assert row[4].startswith("phi_f = 32.5 deg: concrete cast against")
