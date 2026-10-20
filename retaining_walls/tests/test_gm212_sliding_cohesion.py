"""GM-212: how the cantilever wall's sliding check counts the foundation
soil's cohesion, by the rules the references print.

* GEC-6 Sec. 5.4 (p. 100): Eq. 5-38, FR = (W + Pv) tan delta, for
  frictional (drained) materials; Eq. 5-39, FR = ca Bf, for cohesive
  (undrained) materials, ca typically cu; "Equations 5-38 and 5-39 should
  not be combined for use at the same time".
* EN 1997-1 Sec. 6.5.3(10) (p. 64): drained, "Any effective cohesion c'
  should be neglected"; (11)P Rd = A cu;d.
* UFC 3-220-20 Fig. 4-27 (p. 220) and Sec. 4-5.1.3 (p. 223): T = (W + Pv)
  tan delta + Ca Be, Ca from Table 4-1 (p. 184), Be by the uniform bearing
  pressure method.

Before GM-212 the module added ca = 2/3 c_foundation on B to the friction,
a rule none of them prints.
"""

import math

import pytest

from retaining_walls.cantilever import (
    SLIDING_COHESION_BASIS, UFC_TABLE_4_1_ADHESION_PSF,
    adhesion_ufc_table_4_1, analyze_cantilever_wall, check_sliding,
    default_sliding_cohesion,
)
from retaining_walls.geometry import CantileverWallGeometry

PSF = 0.04788026  # kPa


def _wall():
    """The GM-212 wall: H = 5 m, the module's trial proportions (B = 3.0,
    toe 0.3, base 0.6 m), backfill gamma 19, phi 32."""
    return CantileverWallGeometry(wall_height=5.0)


def _run(**kw):
    return analyze_cantilever_wall(_wall(), 19.0, 32.0, **kw)


# --- UFC 3-220-20 Table 4-1 (p. 184), adhesion by undrained strength ------

@pytest.mark.parametrize("su_psf, ca_psf", [
    (0, 0), (250, 250), (500, 500), (1000, 750), (2000, 950), (4000, 1300)])
def test_table_4_1_row_ends_are_the_printed_values(su_psf, ca_psf):
    assert adhesion_ufc_table_4_1(su_psf * PSF) == pytest.approx(
        ca_psf * PSF, rel=1e-9)


def test_table_4_1_reads_linearly_within_a_row():
    # stiff, 1000-2000 psf -> 750-950 psf: at 1500 psf, 850 psf
    assert adhesion_ufc_table_4_1(1500 * PSF) == pytest.approx(850 * PSF)
    assert UFC_TABLE_4_1_ADHESION_PSF[-1] == (4000.0, 1300.0)


@pytest.mark.parametrize("su, ca", [
    (12, 12), (24, 24), (48, 36), (96, 45), (192, 62)])
def test_nhi_table_10_1_prints_the_same_rows_in_kpa(su, ca):
    """FHWA NHI-06-089 Table 10-1 (p. 10-18): 0-12 -> 0-12, 12-24 ->
    12-24, 24-48 -> 24-36, 48-96 -> 36-45, 96-192 -> 45-62 kPa."""
    su = min(su, 4000 * PSF)
    assert adhesion_ufc_table_4_1(su) == pytest.approx(ca, abs=0.6)


def test_above_the_table_there_is_no_printed_value():
    with pytest.raises(ValueError, match="give base_adhesion"):
        adhesion_ufc_table_4_1(200.0)


# --- the rule in force ---------------------------------------------------

def test_the_default_rule_follows_phi_foundation():
    assert default_sliding_cohesion(0.0) == "undrained"
    assert default_sliding_cohesion(28.0) == "drained"
    assert default_sliding_cohesion(None) == "drained"


def test_each_basis_quotes_its_printed_rule():
    assert "should not be combined for use at the same time" \
        in SLIDING_COHESION_BASIS["drained"]
    assert "Any effective cohesion c' should be neglected" \
        in SLIDING_COHESION_BASIS["drained"]
    assert "ca is typically taken as equal to the undrained shear " \
        "strength" in SLIDING_COHESION_BASIS["undrained"]
    assert "0.4 Vd" in SLIDING_COHESION_BASIS["undrained"]
    fpa = SLIDING_COHESION_BASIS["friction_plus_adhesion"]
    assert "T = (W + Pv) tan delta + Ca Be" in fpa
    assert "Minimum FSL may also need to be increased" in fpa
    assert "NHI-06-089 Fig. 10-17" in fpa


# --- before / after on c' = 10 kPa, phi' = 28 deg -------------------------

def test_drained_default_counts_no_cohesion():
    r = _run(phi_foundation=28.0, c_foundation=10.0)
    t = r.sliding_terms
    assert t["sliding_cohesion"] == "drained"
    assert t["R_adhesion"] == 0.0 and t["ca"] == 0.0
    assert t["resisting"] == pytest.approx(
        t["V"] * math.tan(math.radians(2.0 / 3.0 * 28.0)))
    # before GM-212: 2/3 x 10 kPa on B = 3.0 m added 20.0 kN/m, FOS 1.507
    assert r.FOS_sliding == pytest.approx(1.233, abs=0.0005)
    assert (t["resisting"] + 2.0 / 3.0 * 10.0 * 3.0) / t["driving"] \
        == pytest.approx(1.507, abs=0.0005)
    assert not r.passes_sliding


def test_friction_plus_adhesion_on_the_effective_width():
    r = _run(phi_foundation=28.0, c_foundation=10.0,
             sliding_cohesion="friction_plus_adhesion")
    t, o = r.sliding_terms, r.overturning_terms
    # su 10 kPa (209 psf) is in the very soft row: Ca = su
    assert t["ca"] == pytest.approx(10.0)
    assert t["ca_from"] == "ufc_table_4_1"
    # Be = B - 2e, e on the loads the check counts (here the overturning
    # check's: no surcharge, no passive)
    assert t["eccentricity"] == pytest.approx(o["eccentricity"])
    assert t["adhesion_width"] == pytest.approx(
        3.0 - 2.0 * abs(o["eccentricity"]))
    assert t["resisting"] == pytest.approx(
        t["R_friction"] + 10.0 * t["adhesion_width"])
    assert r.FOS_sliding == pytest.approx(1.553, abs=0.0005)


def test_undrained_is_cu_on_b_alone():
    r = _run(phi_foundation=0.0, c_foundation=40.0)
    t = r.sliding_terms
    assert t["sliding_cohesion"] == "undrained"
    assert t["delta_b"] is None and t["R_friction"] == 0.0
    assert t["resisting"] == pytest.approx(40.0 * 3.0)
    s = check_sliding(_wall(), 19.0, 32.0, phi_foundation=0.0,
                      c_foundation=40.0)
    assert s["delta_base_deg"] is None and s["adhesion_width_m"] == 3.0


def test_an_input_the_rule_does_not_count_is_refused():
    with pytest.raises(ValueError, match="should not be combined"):
        _run(phi_foundation=0.0, c_foundation=40.0, delta_base=20.0)
    with pytest.raises(ValueError, match="Any effective cohesion"):
        _run(phi_foundation=28.0, base_adhesion=5.0)
    with pytest.raises(ValueError, match="c_foundation = cu > 0"):
        _run(phi_foundation=0.0)
    with pytest.raises(ValueError, match="sliding_cohesion must be one of"):
        _run(sliding_cohesion="both")


# --- the choice ----------------------------------------------------------

def _cohesion_choice(r):
    return [c for c in r.choices if "sliding_cohesion" in c["what"]]


def test_the_choice_names_the_rule_and_the_others_fos():
    r = _run(phi_foundation=28.0, c_foundation=10.0)
    (c,) = _cohesion_choice(r)
    assert c["how"] == "default"
    assert "c_foundation = 10 kPa not counted" in c["chosen"]
    # the default's reason, from what the call gave
    assert c["basis"] == ("phi_foundation 28 deg is taken as a drained "
                          "strength, so c_foundation is c': "
                          + SLIDING_COHESION_BASIS["drained"])
    alt = " | ".join(c["alternatives"])
    assert "friction_plus_adhesion (FOS_sliding 1.553)" in alt
    assert "undrained (FOS_sliding 0.411)" in alt
    assert "c_foundation read as cu" in alt
    # no phi_foundation: phi_backfill's 32 deg, the value in force
    r = _run(c_foundation=10.0)
    (c,) = _cohesion_choice(r)
    assert c["basis"].startswith("phi_foundation 32 deg is taken as a "
                                 "drained strength")
    # the base friction choice stays beside it
    assert any("base friction" in x["what"] for x in r.choices)


def test_undrained_has_no_base_friction_choice():
    r = _run(phi_foundation=0.0, c_foundation=40.0)
    (c,) = _cohesion_choice(r)
    assert c["basis"].startswith("phi_foundation 0 is an undrained")
    assert not any("base friction" in x["what"] for x in r.choices)
    # the drained rule has no friction angle to use: said, not FOS 0.000
    drained = next(a for a in c["alternatives"] if a.startswith("drained"))
    assert "not run here" in drained and "GEC-6 Eq. 5-38 needs" in drained


def test_no_cohesion_no_choice():
    r = _run(phi_foundation=30.0)
    assert _cohesion_choice(r) == []


def test_a_rule_that_cannot_run_says_why():
    # the table stops at 191.5 kPa: friction_plus_adhesion does not run
    r = _run(phi_foundation=0.0, c_foundation=250.0)
    (c,) = _cohesion_choice(r)
    fpa = next(a for a in c["alternatives"]
               if a.startswith("friction_plus_adhesion"))
    assert "not run here" in fpa and "Table 4-1" in fpa


# --- the calc steps print the rule ---------------------------------------

@pytest.mark.parametrize("kw, printed", [
    (dict(phi_foundation=28.0, c_foundation=10.0),
     ["R = V × tan(δ_b)", "c_foundation = 10 kPa not counted",
      "GEC-6 Eq. 5-38"]),
    (dict(phi_foundation=28.0, c_foundation=10.0,
          sliding_cohesion="friction_plus_adhesion"),
     ["C_a × B_e", "UFC 3-220-20 Table 4-1", "B_e = B − 2|e|",
      "UFC 3-220-20 Fig. 4-27"]),
    (dict(phi_foundation=0.0, c_foundation=40.0),
     ["R = c_a × B", "R = 40.0 × 3.00", "GEC-6 Eq. 5-39"]),
])
def test_calc_steps_print_the_rule(kw, printed):
    from retaining_walls.calc_steps import get_calc_steps
    r = _run(**kw)
    text = repr(get_calc_steps(r, {"geom": _wall(), "gamma_backfill": 19.0,
                                   "phi_backfill": 32.0, **kw}))
    for p in printed:
        assert p in text, p
    if kw.get("phi_foundation") == 0.0:
        assert "Base Friction Angle" not in text
