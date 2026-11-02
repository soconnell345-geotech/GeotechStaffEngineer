"""GM-302: checks GEC-11 prints that the MSE wall did not report.

Each read on the rendered page:

* Vol II Tables E3-6.3 (p. E3-11), E4-6.3 (p. E4-9), E5-7.1 (p. E5-17):
  "Limiting eccentricity, e = L/4 for strength limit states and e = L/6
  for service limit state", "Is the resultant within limiting value of
  eL?" - a verdict on the LRFD path, Service I with the live load counted.
* Vol I p. 3-21: "The maximum St distance is 24 in. (610 mm)" - a verdict
  with the empirical F*; p. 3-20 gives that F* "For ... St > 6 inches (150
  mm)" - a warning. Both sit in the empirical F*'s paragraphs, so with F*
  from pullout tests an St over 24 in. is a warning (GM-302 review).
* Vol I p. 4-38: "vertical spacing of reinforcement should not exceed
  32 in. (800 mm)" - advice, a warning.
* Vol I p. 4-10: reinforced fill phi 34 deg unless tested, never above
  40 deg - advice, warnings; foundation 30 deg "in absence of specific
  data", retained 30 deg "often used" - the values left-out angles take, a
  choice (GM-302 review; phi_backfill before).
* Vol I p. 4-30: Eq. 4-24, gamma_r H <= 3 cu, "required" - a verdict on
  both paths; c* = 0.67 c', phi* = tan^-1(0.67 tan phi'f) where local or
  punching shear is possible - an option on the LRFD qn.

E3 and E4 values as printed; every other wall made up.
"""

import math

import pytest

from retaining_walls import mse
from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.reinforcement import Reinforcement

FT = 0.3048
KSF = 47.88
PCF = 0.157087
GEOGRID = Reinforcement(name="made-up geogrid", type="geogrid",
                        Tallowable=60.0)


def _grid(St, name="made-up grid"):
    return Reinforcement(name=name, type="metallic_grid", Tallowable=100.0,
                         coverage_ratio=0.3, thickness=0.0095,
                         transverse_spacing=St)


def _wall(**kw):
    return MSEWallGeometry(**{"wall_height": 6.0, "reinforcement_length": 4.5,
                              "reinforcement_spacing": 0.6, **kw})


# ---------------------------------------------------------------------------
# Service I eccentricity, L/6 (Vol II Tables E3-6.3, E4-6.3)
# ---------------------------------------------------------------------------

def test_e4_service_i_resultant_as_printed():
    """Table E4-6.3, Ser I: sum V 62.19 k/ft with the live load, eL 2.32 ft
    against e = L/6 = 3.00 ft: 'Yes'."""
    g = MSEWallGeometry(wall_height=25.64 * FT, reinforcement_length=18 * FT,
                        reinforcement_spacing=2.5 * FT, surcharge=0.25 * KSF)
    e = mse.check_external_stability_lrfd(
        g, 125 * PCF, 34.0, 30.0, phi_retained=30.0,
        bearing_resistance_strength=10.5 * KSF,
        bearing_resistance_service=7.5 * KSF)["eccentricity"]
    assert e["eL_service_m"] / FT == pytest.approx(2.32, abs=0.01)
    assert e["e_limit_service_m"] / FT == pytest.approx(3.00, abs=0.001)
    assert e["passes_service"] and e["passes_strength"] and e["passes"]


def test_e3_service_i_resultant_on_a_slope_as_printed():
    """Table E3-6.3, Ser I (a 2H:1V slope that continues): eL 2.61 ft
    against L/6 = 4.00 ft."""
    g = MSEWallGeometry(wall_height=30 * FT, reinforcement_length=24 * FT,
                        reinforcement_spacing=2.5 * FT,
                        backfill_slope=math.degrees(math.atan(0.5)))
    r = mse.check_external_stability_lrfd(
        g, 125 * PCF, 34.0, 30.0, phi_retained=30.0,
        bearing_resistance_strength=10.5 * KSF,
        bearing_resistance_service=7.5 * KSF)
    e = r["eccentricity"]
    assert e["eL_service_m"] / FT == pytest.approx(2.61, abs=0.01)
    assert e["e_limit_service_m"] / FT == pytest.approx(4.00, abs=0.001)
    assert "eccentricity (Service I)" not in r["checks_failed"]
    assert r["load_factors"]["ecc_limit_ratio_service"] == pytest.approx(1 / 6)


def test_a_short_base_fails_the_service_check():
    """L = 0.5 H, no live load: Service I eL = Ka H^2 / (6 L) = 0.67 m
    against L/6 = 0.50 m."""
    r = mse.check_external_stability_lrfd(
        _wall(reinforcement_length=3.0), 18.0, 30.0, 30.0,
        phi_retained=30.0)
    e = r["eccentricity"]
    assert e["eL_service_m"] == pytest.approx(6.0 ** 2 / (3 * 6 * 3.0),
                                              rel=1e-3)
    assert not e["passes_service"] and not e["passes"]
    assert "eccentricity (Service I)" in r["checks_failed"]
    assert r["passes"] is False


def test_the_service_limit_can_be_set():
    r = mse.check_external_stability_lrfd(
        _wall(reinforcement_length=3.0), 18.0, 30.0, 30.0,
        phi_retained=30.0, ecc_limit_ratio_service=0.25)
    assert r["eccentricity"]["e_limit_service_m"] == pytest.approx(0.75)
    assert r["eccentricity"]["passes_service"]
    with pytest.raises(ValueError, match="ecc_limit_ratio_service"):
        mse.check_external_stability_lrfd(
            _wall(), 18.0, 30.0, 30.0, ecc_limit_ratio_service=0.0)


def test_a_rock_strength_limit_warns_of_the_soil_service_limit():
    """The examples are all on soil: with 3L/8 and L/6 left, a warning."""
    r = mse.check_external_stability_lrfd(_wall(), 18.0, 30.0, 30.0,
                                          phi_retained=30.0,
                                          ecc_limit_ratio=0.375)
    assert any("no page held prints the service limit on rock" in w
               for w in r["warnings"])
    r = mse.check_external_stability_lrfd(_wall(), 18.0, 30.0, 30.0,
                                          phi_retained=30.0,
                                          ecc_limit_ratio=0.375,
                                          ecc_limit_ratio_service=0.25)
    assert "warnings" not in r


# ---------------------------------------------------------------------------
# Eq. 4-24, local shear on a cohesive foundation (p. 4-30)
# ---------------------------------------------------------------------------

def test_eq_4_24_on_both_paths():
    """gamma_r H = 19 x 6 = 114 kPa: cu 40 gives 3 cu = 120 (passes), cu 35
    gives 105 (fails)."""
    g = _wall()
    ok = mse.check_external_stability_lrfd(g, 19.0, 30.0, 0.0,
                                           phi_retained=30.0,
                                           c_foundation=40.0)
    assert ok["local_shear"]["gamma_r_H_kPa"] == pytest.approx(114.0)
    assert ok["local_shear"]["three_cu_kPa"] == pytest.approx(120.0)
    assert ok["local_shear"]["passes"]
    assert "local shear" not in ok["checks_failed"]
    weak = mse.check_external_stability_lrfd(g, 19.0, 30.0, 0.0,
                                             phi_retained=30.0,
                                             c_foundation=35.0)
    assert "local shear" in weak["checks_failed"]
    asd = mse.analyze_mse_wall(g, 19.0, 30.0, GEOGRID, phi_foundation=0.0,
                               c_foundation=35.0, phi_retained=30.0,
                               q_allowable=500.0)
    assert not asd.local_shear["passes"]
    assert "local shear" in asd.checks_failed
    assert asd.passes_external is False
    assert "3 cu 105.0 kPa, GEC-11 Eq. 4-24" in asd.verdict
    assert asd.to_dict()["local_shear"]["three_cu_kPa"] == pytest.approx(105)
    assert "Local shear" in asd.summary()


def test_eq_4_24_not_on_a_drained_foundation():
    g = _wall()
    r = mse.check_external_stability_lrfd(g, 19.0, 30.0, 28.0,
                                          phi_retained=30.0,
                                          c_foundation=10.0)
    assert "local_shear" not in r
    assert "local shear" not in r["checks_failed"]
    asd = mse.analyze_mse_wall(g, 19.0, 30.0, GEOGRID, phi_foundation=28.0,
                               phi_retained=30.0)
    assert asd.local_shear is None and "local_shear" not in asd.to_dict()


def test_local_shear_strengths_in_qn():
    """p. 4-30: c* = 0.67 c', phi* = tan^-1(0.67 tan phi'f); phi 30 ->
    21.15 deg. qn by Eq. 4-22 then takes them; sliding does not."""
    c_star, phi_star = mse.local_shear_strengths(30.0, 10.0)
    assert c_star == pytest.approx(6.7)
    assert phi_star == pytest.approx(
        math.degrees(math.atan(0.67 * math.tan(math.radians(30)))))
    assert phi_star == pytest.approx(21.15, abs=0.01)
    g = _wall()
    kw = dict(phi_retained=30.0, gamma_foundation=18.0, c_foundation=10.0)
    full = mse.check_external_stability_lrfd(g, 19.0, 30.0, 30.0, **kw)
    local = mse.check_external_stability_lrfd(g, 19.0, 30.0, 30.0,
                                              local_shear_possible=True, **kw)
    b = local["bearing"]
    assert b["phi_star_deg"] == pytest.approx(21.15, abs=0.01)
    assert b["c_star_kPa"] == pytest.approx(6.7)
    assert b["qn_strength_max_kPa"] == pytest.approx(
        mse.nominal_bearing_resistance(b["B_eff_strength_max_m"], 18.0,
                                       phi_star, 6.7), abs=0.02)
    assert b["qn_strength_max_kPa"] < full["bearing"]["qn_strength_max_kPa"]
    assert local["sliding"] == full["sliding"]
    (strengths,) = [c for c in local["choices"]
                    if c["what"] == "foundation strengths in qn (Eq. 4-22)"]
    assert strengths["chosen"].startswith("c* = 0.67 c_f")
    assert strengths["how"] == "given" and "10.6.3.1.2b" in strengths["basis"]
    # each reading gives the other's governing bearing CDR (GM-302 review)
    assert strengths["alternatives"] == [
        "c_f and phi_f as given: general shear (bearing CDR_governing "
        f"{full['bearing']['CDR_governing']:.3f})"]
    (general,) = [c for c in full["choices"]
                  if c["what"] == "foundation strengths in qn (Eq. 4-22)"]
    assert general["how"] == "default"
    assert general["alternatives"][0].endswith(
        f"(bearing CDR_governing {b['CDR_governing']:.3f})")
    assert b["CDR_governing"] < full["bearing"]["CDR_governing"]


def test_local_shear_strengths_refused_where_they_do_nothing():
    with pytest.raises(ValueError, match="needs gamma_foundation"):
        mse.check_external_stability_lrfd(_wall(), 19.0, 30.0, 30.0,
                                          bearing_resistance_strength=400.0,
                                          local_shear_possible=True)
    with pytest.raises(ValueError, match="LRFD path"):
        mse.analyze_mse_wall(_wall(), 19.0, 30.0, GEOGRID,
                             phi_foundation=30.0, local_shear_possible=True)


# ---------------------------------------------------------------------------
# A steel grid's St (pp. 3-20, 3-21) and the vertical spacing (p. 4-38)
# ---------------------------------------------------------------------------

def test_grid_st_over_24_in_fails():
    g = _wall()
    over = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(0.70),
                                phi_foundation=30.0, phi_retained=30.0)
    assert over.grid_spacing["St_max_m"] == pytest.approx(0.70)
    assert over.grid_spacing["St_limit_m"] == pytest.approx(0.6096)
    assert "grid spacing" in over.checks_failed
    assert over.all_checks_pass is False
    assert "St 700 mm > 610 mm" in over.verdict
    ok = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(0.6096),
                              phi_foundation=30.0, phi_retained=30.0)
    assert ok.grid_spacing["passes"]
    assert "grid spacing" not in ok.checks_failed


def test_grid_st_under_6_in_with_the_empirical_f_star_warns():
    """Example E4 takes the empirical F* at St = 6 in. exactly: no warning
    there; at 5 in. there is one, and none with F* from tests."""
    g = _wall()
    at6 = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(0.5 * FT),
                               phi_foundation=30.0, phi_retained=30.0)
    assert not any("under 6 in." in w for w in at6.warnings)
    at5 = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(5 * 0.0254),
                               phi_foundation=30.0, phi_retained=30.0)
    assert any("St 127 mm is under 6 in." in w for w in at5.warnings)
    tested = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(5 * 0.0254),
                                  phi_foundation=30.0, phi_retained=30.0,
                                  F_star_tested=1.0)
    assert not any("under 6 in." in w for w in tested.warnings)


def test_with_f_star_from_tests_st_over_24_in_warns():
    """The 24 in. sits in p. 3-21's paragraph on the empirical F*: with F*
    from pullout tests it is a warning, never a failed check (GM-302
    review); no St given (allowed only then) is not checked."""
    g = _wall()
    over = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(0.70),
                                phi_foundation=30.0, phi_retained=30.0,
                                F_star_tested=1.0)
    assert over.grid_spacing is None and "grid_spacing" not in over.to_dict()
    assert "grid spacing" not in over.checks_failed
    (w,) = [w for w in over.warnings if "24 in." in w]
    assert w.startswith("St 700 mm is over 24 in. (610 mm)")
    assert "not a check" in w
    at24 = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(0.6096),
                                phi_foundation=30.0, phi_retained=30.0,
                                F_star_tested=1.0)
    assert at24.warnings == []
    r = mse.analyze_mse_wall(g, 19.0, 34.0, _grid(0.0), phi_foundation=30.0,
                             phi_retained=30.0, F_star_tested=1.0)
    assert r.grid_spacing is None and r.warnings == []
    assert r.checks_not_run == ["bearing"]
    geo = mse.analyze_mse_wall(g, 19.0, 34.0, GEOGRID, phi_foundation=30.0,
                               phi_retained=30.0)
    assert geo.grid_spacing is None and "grid_spacing" not in geo.to_dict()


def test_vertical_spacing_over_32_in_warns():
    assert not mse.vertical_spacing_warnings(_wall(reinforcement_spacing=0.8))
    (w,) = mse.vertical_spacing_warnings(_wall(reinforcement_spacing=0.9))
    assert "0.9 m (reinforcement_spacing)" in w and "p. 4-38" in w
    (w,) = mse.vertical_spacing_warnings(
        _wall(level_depths=[0.4, 1.2, 2.2, 3.0, 3.8, 4.6, 5.4]))
    assert "between the levels at z = 1.2 and 2.2 m" in w
    r = mse.analyze_mse_wall(_wall(reinforcement_spacing=0.9), 19.0, 34.0,
                             GEOGRID, phi_foundation=30.0, phi_retained=30.0)
    assert any("32 in." in x for x in r.warnings)
    # advice: no check of its own
    assert not any("spacing" in c for c in r.checks_failed)


# ---------------------------------------------------------------------------
# The friction angles (p. 4-10)
# ---------------------------------------------------------------------------

def test_fill_strength_warnings():
    assert mse.fill_strength_warnings(34.0) == []
    (w,) = mse.fill_strength_warnings(36.0)
    assert "above 34 deg" in w and "triaxial or direct shear" in w
    (w,) = mse.fill_strength_warnings(42.0)
    assert "above 40 deg" in w and "should not used" in w


def test_left_out_foundation_and_retained_angles_take_30_deg():
    """p. 4-10: 30 deg for the foundation 'in absence of specific data' and
    for the retained fill, 'recommended only for preliminary sizing': the
    value a left-out angle takes, stated as a choice (GM-302 review)."""
    g = _wall()
    left = mse.analyze_mse_wall(g, 19.0, 34.0, GEOGRID)
    given = mse.analyze_mse_wall(g, 19.0, 34.0, GEOGRID, phi_foundation=30.0,
                                 phi_retained=30.0)
    assert left.FOS_sliding == given.FOS_sliding
    assert left.FOS_overturning == given.FOS_overturning
    assert left.external_terms["phi_foundation"] == 30.0
    assert left.external_terms["phi_retained"] == 30.0
    assumed = {c["what"]: c for c in left.choices
               if c["what"].endswith(("(phi_foundation)", "(phi_retained)"))}
    assert sorted(assumed) == [
        "foundation soil friction angle (phi_foundation)",
        "retained fill friction angle (phi_retained)"]
    for c in assumed.values():
        assert c["chosen"] == "30 deg, assumed for preliminary sizing"
        assert c["how"] == "default"
        assert "p. 4-10" in c["basis"]
        assert "recommended only for preliminary sizing" in c["basis"]
    assert not any(c["what"].endswith(("(phi_foundation)", "(phi_retained)"))
                   for c in given.choices)
    # the LRFD path and the ASD external check alike
    lrfd = mse.check_external_stability_lrfd(g, 19.0, 34.0)
    lrfd_given = mse.check_external_stability_lrfd(g, 19.0, 34.0, 30.0,
                                                   phi_retained=30.0)
    assert lrfd["sliding"] == lrfd_given["sliding"]
    assert lrfd["eccentricity"] == lrfd_given["eccentricity"]
    assert len([c for c in lrfd["choices"]
                if "preliminary sizing" in c["chosen"]]) == 2
    assert "choices" not in lrfd_given
    asd = mse.check_external_stability(g, 19.0, 34.0, 18.0, 30.0,
                                       q_allowable=500.0)
    asd_given = mse.check_external_stability(g, 19.0, 34.0, 18.0, 30.0,
                                             q_allowable=500.0,
                                             phi_retained=30.0)
    assert asd == asd_given
    # a foundation angle is still asked for where qn is worked out
    with pytest.raises(ValueError):
        mse.check_external_stability_lrfd(g, 19.0, 34.0,
                                          gamma_foundation=18.0)


_DEFAULTED = "taken because phi_retained was not given (GEC-11 Vol I p. 4-10"


def test_a_slope_steeper_than_the_assumed_30_deg_says_it_was_the_default():
    """phi_retained left out: the refusal names the 30 deg as p. 4-10's
    default and asks for phi_retained, on every path (GM-302 review)."""
    g = _wall(backfill_slope=31.0)
    calls = (
        lambda **kw: mse.analyze_mse_wall(g, 19.0, 34.0, GEOGRID,
                                          phi_foundation=30.0, **kw),
        lambda **kw: mse.check_external_stability_lrfd(g, 19.0, 34.0, 30.0,
                                                       **kw),
        lambda **kw: mse.check_external_stability(g, 19.0, 34.0, 18.0, 30.0,
                                                  **kw),
    )
    for call in calls:
        with pytest.raises(ValueError) as left_out:
            call()
        msg = str(left_out.value)
        assert msg.startswith(
            "backfill_slope 31 deg is steeper than the friction angle of the "
            "fill behind the reinforced zone, 30 deg, " + _DEFAULTED)
        assert "Give phi_retained" in msg
        call(phi_retained=34.0)             # given and steep enough: runs
    # at 30 deg the slope stands (Eq. 4-3: Gamma = 1)
    mse.analyze_mse_wall(_wall(backfill_slope=30.0), 19.0, 34.0, GEOGRID,
                         phi_foundation=30.0)


def test_a_slope_steeper_than_a_given_phi_retained_keeps_its_wording():
    g = _wall(backfill_slope=31.0)
    with pytest.raises(ValueError) as given:
        mse.analyze_mse_wall(g, 19.0, 34.0, GEOGRID, phi_foundation=30.0,
                             phi_retained=30.0)
    msg = str(given.value)
    assert msg.startswith(
        "backfill_slope 31 deg is steeper than the friction angle of the "
        "fill behind the reinforced zone, 30 deg: sin(phi - beta) < 0")
    assert _DEFAULTED not in msg and "Give phi_retained" not in msg


def test_warnings_reach_the_results():
    r = mse.analyze_mse_wall(_wall(), 19.0, 36.0, GEOGRID)
    (w,) = r.warnings
    assert "above 34 deg" in w
    d = r.to_dict()
    assert d["warnings"] == r.warnings
    assert "  Warnings:" in r.summary()
    # nothing to warn of
    quiet = mse.analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID)
    assert quiet.warnings == [] and "warnings" not in quiet.to_dict()
    # the LRFD path's warnings join the result's once
    both = mse.analyze_mse_wall(_wall(), 19.0, 36.0, GEOGRID,
                                lrfd_external=True)
    assert both.warnings == r.warnings
    assert "warnings" not in both.external_lrfd
    lrfd = mse.check_external_stability_lrfd(_wall(), 19.0, 36.0)
    assert lrfd["warnings"] == r.warnings


# ---------------------------------------------------------------------------
# The calc package
# ---------------------------------------------------------------------------

def test_calc_package_prints_the_new_checks():
    g = _wall()
    r = mse.analyze_mse_wall(g, 19.0, 36.0, _grid(0.70), phi_foundation=0.0,
                             c_foundation=35.0, q_allowable=500.0)
    analysis = {"geom": g, "reinforcement": _grid(0.70),
                "gamma_backfill": 19.0, "phi_backfill": 36.0,
                "phi_foundation": 0.0, "c_foundation": 35.0,
                "q_allowable": 500.0}
    sections = get_calc_steps(r, analysis)
    titles = [getattr(it, "title", None) or getattr(it, "description", "")
              for s in sections for it in s.items if not isinstance(it, str)]
    assert "Local Shear on a Cohesive Foundation" in titles
    assert "Local shear (GEC-11 Eq. 4-24)" in titles
    assert any(t.startswith("Steel grid transverse-bar spacing St")
               for t in titles)
    assert "Warnings (GEC-11 advice, not a check)" in titles
