"""GM-302: the MSE tools report the checks GEC-11 prints beside the
stability checks - the Service I resultant within L/6 (Vol II Table
E4-6.3), Eq. 4-24 on a cohesive foundation and the reduced strengths where
local shear is possible (Vol I p. 4-30), a steel grid's St (p. 3-21) - and
GEC-11's advice as warnings (pp. 3-20, 4-10, 4-38). The module tests are in
retaining_walls/tests/test_gm302_checks.py. Made-up walls."""

import pytest

from funhouse_agent import dispatch as D

_WALL = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
             reinforcement_length=4.5, reinforcement_spacing=0.6,
             reinforcement_type="geogrid", reinforcement_Tallowable=60.0,
             phi_foundation=30.0, phi_retained=30.0)
_LRFD = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
             reinforcement_length=4.5, phi_foundation=30.0,
             phi_retained=30.0)


def _call(method, params):
    r = D.call_agent("retaining_walls", method, params)
    return r


def test_lrfd_reports_the_service_i_resultant():
    r = _call("mse_lrfd_external_stability", _LRFD)
    assert "error" not in r, r.get("error")
    e = r["eccentricity"]
    assert e["e_limit_service_m"] == pytest.approx(4.5 / 6, abs=1e-4)
    assert e["passes_service"] is True
    assert r["defaults_used"]["ecc_limit_ratio_service"] == pytest.approx(
        1 / 6)
    short = _call("mse_lrfd_external_stability",
                  {**_LRFD, "reinforcement_length": 3.0})
    assert "eccentricity (Service I)" in short["checks_failed"]
    set_ = _call("mse_lrfd_external_stability",
                 {**_LRFD, "reinforcement_length": 3.0,
                  "ecc_limit_ratio_service": 0.25})
    assert set_["eccentricity"]["e_limit_service_m"] == pytest.approx(0.75)
    assert "ecc_limit_ratio_service" not in set_["defaults_used"]


def test_lrfd_local_shear_on_a_cohesive_foundation():
    r = _call("mse_lrfd_external_stability",
              {**_LRFD, "phi_foundation": 0.0, "c_foundation": 35.0})
    assert r["local_shear"]["gamma_r_H_kPa"] == pytest.approx(114.0)
    assert "local shear" in r["checks_failed"]
    assert "local_shear" not in _call("mse_lrfd_external_stability", _LRFD)


def test_lrfd_local_shear_possible():
    r = _call("mse_lrfd_external_stability",
              {**_LRFD, "gamma_foundation": 18.0,
               "local_shear_possible": True})
    assert r["bearing"]["phi_star_deg"] == pytest.approx(21.15, abs=0.01)
    (c,) = [c for c in r["choices"]
            if c["what"] == "foundation strengths in qn (Eq. 4-22)"]
    assert c["how"] == "given"
    # the general-shear reading's governing bearing CDR (GM-302 review)
    general = _call("mse_lrfd_external_stability",
                    {**_LRFD, "gamma_foundation": 18.0})
    assert c["alternatives"] == [
        "c_f and phi_f as given: general shear (bearing CDR_governing "
        f"{general['bearing']['CDR_governing']:.3f})"]
    refused = _call("mse_lrfd_external_stability",
                    {**_LRFD, "local_shear_possible": True,
                     "bearing_resistance_strength": 400.0})
    assert "needs gamma_foundation" in refused.get("error", "")


def test_lrfd_takes_30_deg_for_a_foundation_angle_left_out():
    """GEC-11 Vol I p. 4-10's 30 deg 'in absence of specific data', a
    choice and a default listed (GM-302 review; phi_backfill before)."""
    lrfd = {k: v for k, v in _LRFD.items() if k != "phi_foundation"}
    r = _call("mse_lrfd_external_stability", lrfd)
    given = _call("mse_lrfd_external_stability", _LRFD)
    assert r["sliding"] == given["sliding"]
    (c,) = [c for c in r["choices"]
            if c["what"] == "foundation soil friction angle (phi_foundation)"]
    assert c["chosen"] == "30 deg, assumed for preliminary sizing"
    assert r["defaults_used"]["phi_foundation"].startswith("30 deg")
    assert "warnings" not in r and "warnings" not in given


def test_mse_wall_reports_the_new_checks_and_warnings():
    r = _call("mse_wall", {**_WALL, "phi_foundation": 0.0,
                           "c_foundation": 35.0, "phi_backfill": 36.0})
    assert "error" not in r, r.get("error")
    assert "local shear" in r["checks_failed"]
    assert r["local_shear"]["three_cu_kPa"] == pytest.approx(105.0)
    assert r["passes_external"] is False
    assert any("above 34 deg" in w for w in r["warnings"])
    quiet = _call("mse_wall", _WALL)
    assert "warnings" not in quiet and "local_shear" not in quiet


def test_mse_wall_checks_a_built_in_grid_st():
    """The built-in bar mats carry Example E4's St (6 in. to 18 in.)."""
    r = _call("mse_wall", {**_WALL, "reinforcement_type": None,
                           "reinforcement_Tallowable": None,
                           "reinforcement_name": "bar_mat_e4_4w15_w11x1_5",
                           "reinforcement_horizontal_spacing": 1.524})
    assert "error" not in r, r.get("error")
    assert r["grid_spacing"]["St_max_m"] == pytest.approx(0.4572)
    assert r["grid_spacing"]["passes"] is True
    over = _call("mse_wall", {**_WALL, "reinforcement_type": "bar_mat",
                              "reinforcement_Tallowable": 100.0,
                              "reinforcement_thickness": 0.0095,
                              "reinforcement_transverse_spacing": 0.70,
                              "coverage_ratio": 0.3})
    assert "error" not in over, over.get("error")
    assert "grid spacing" in over["checks_failed"]
    assert "St 700 mm > 610 mm" in over["verdict"]
