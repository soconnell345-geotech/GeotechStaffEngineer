"""GM-114: the LRFD Strength I min eccentricity and bearing, apart.

``check_external_stability_lrfd`` gave Strength I max and the critical
combination (EV min, EH max) for the eccentricity and bearing, and Strength
I min (EV min, EH min) for sliding only. GEC-11 Vol I Table 4-2 (p. 4-4)
prints the minimum permanent load factors (EV 1.00, EH 0.90) and p. 4-23
asks for "maximum permanent loads, minimum permanent loads" and "for each
strength limit load group, e must be less than emax". The combination now
has its own eccentricity and bearing values, counted in the governing
ones. Made-up walls only.
"""

import pytest

from retaining_walls.mse import (
    STRENGTH_MIN_BASIS, MSEWallGeometry, check_external_stability_lrfd,
    rankine_Ka)


H, L, Q, GAMMA, PHI, PHI_F = 6.5, 4.8, 11.0, 19.0, 33.0, 29.0


def _level(**kw):
    return check_external_stability_lrfd(
        MSEWallGeometry(wall_height=H, reinforcement_length=L, surcharge=Q),
        GAMMA, PHI, PHI_F, **kw)


def test_level_strength_min_eccentricity_by_hand():
    r = _level()
    # phi_retained left out: 30 deg (GEC-11 Vol I p. 4-10)
    Ka = rankine_Ka(30.0)
    V1 = GAMMA * H * L
    F1, F2 = 0.5 * Ka * GAMMA * H ** 2, Ka * Q * H
    ev, eh, ll = 1.00, 0.90, 1.75
    e = L / 2 - (ev * V1 * L / 2 - (eh * F1 * H / 3 + ll * F2 * H / 2)) / (
        ev * V1)
    assert r["eccentricity"]["eL_strength_min_m"] == pytest.approx(e,
                                                                   abs=1e-4)
    assert r["eccentricity"]["strength_min_basis"] == STRENGTH_MIN_BASIS
    assert "Table 4-2, p. 4-4" in STRENGTH_MIN_BASIS
    assert "p. 4-23" in STRENGTH_MIN_BASIS


def test_level_strength_min_bearing_by_hand():
    r = _level(bearing_resistance_strength=400.0)
    Ka = rankine_Ka(30.0)
    V1, Vs = GAMMA * H * L, Q * L
    F1, F2 = 0.5 * Ka * GAMMA * H ** 2, Ka * Q * H
    # LL 1.75 over the zone: GEC-11 Vol II Table E4-5.1 (p. E4-6)
    SV = 1.00 * V1 + 1.75 * Vs
    MO = 0.90 * F1 * H / 3 + 1.75 * F2 * H / 2
    eB = L / 2 - (SV * L / 2 - MO) / SV
    sigma = SV / (L - 2 * eB)
    b = r["bearing"]
    assert b["sigma_v_strength_min_kPa"] == pytest.approx(sigma, abs=0.01)
    assert b["eL_strength_min_m"] == pytest.approx(eB, abs=1e-4)
    assert b["B_eff_strength_min_m"] == pytest.approx(L - 2 * eB, abs=1e-4)
    assert b["CDR_strength_min"] == pytest.approx(400.0 / sigma, abs=1e-3)


def test_on_a_level_backfill_it_never_governs():
    for kw in ({"bearing_resistance_strength": 350.0},
               {"gamma_foundation": 18.0}):
        r = _level(**kw)
        e, b = r["eccentricity"], r["bearing"]
        assert e["eL_strength_min_m"] <= e["eL_critical_m"]
        assert e["eL_governing_m"] == max(e["eL_strength_max_m"],
                                          e["eL_critical_m"])
        assert b["CDR_strength_min"] >= b["CDR_critical"]
        assert b["CDR_governing"] == min(b["CDR_strength"],
                                         b["CDR_critical"])


def test_the_qn_is_worked_out_over_its_own_width():
    r = _level(gamma_foundation=18.0)
    b = r["bearing"]
    assert b["qR_strength_min_kPa"] == pytest.approx(
        0.65 * b["qn_strength_min_kPa"], abs=0.01)
    assert b["CDR_strength_min"] == pytest.approx(
        b["qR_strength_min_kPa"] / b["sigma_v_strength_min_kPa"], abs=2e-3)


def test_without_a_resistance_its_bearing_cdr_is_not_given():
    b = _level()["bearing"]
    assert b["CDR_strength_min"] is None
    assert b["sigma_v_strength_min_kPa"] > 0


def test_sloping_backfill_has_it_and_counts_it():
    r = check_external_stability_lrfd(
        MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                        backfill_slope=14.0),
        GAMMA, PHI, PHI_F, gamma_foundation=18.0)
    e, b = r["eccentricity"], r["bearing"]
    assert e["eL_governing_m"] == max(e["eL_strength_max_m"],
                                      e["eL_critical_m"],
                                      e["eL_strength_min_m"])
    assert b["CDR_governing"] == min(b["CDR_strength"], b["CDR_critical"],
                                     b["CDR_strength_min"])


def test_the_load_factors_given_are_the_ones_used():
    r1 = _level(gamma_EH_min=0.90)
    r2 = _level(gamma_EH_min=1.20)
    assert r2["eccentricity"]["eL_strength_min_m"] > \
        r1["eccentricity"]["eL_strength_min_m"]
