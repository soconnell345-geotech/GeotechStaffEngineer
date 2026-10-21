"""GM-164: a live load over an MSE wall's reinforced zone, ASD path.

GEC-11 Vol I (FHWA-NHI-10-024): the live load over the reinforced zone is
"not considered as a stabilizing force when checking sliding" (p. 4-19;
Eq. 4-12, p. 4-21, "the live load is excluded since it increases sliding
stability"); the eccentricity check, "formerly known as overturning"
(p. 4-18), "only applies the live load above the retained backfill"
(p. 4-22); the bearing check "applies live load above both the reinforced
zone and the retained backfill" (p. 4-26; Fig. 4-2). Its thrust drives
every check. Hand checks on a level 6 m wall with and without a 12 kPa
surcharge, both readings of each choice, a broken backslope with level
ground on the zone, the adapter and the calc steps.
"""

import math

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.earth_pressure import rankine_Ka
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    BEARING_BASIS, LIVE_LOAD_BASIS, _external_terms, analyze_mse_wall,
    check_external_stability,
)
from retaining_walls.reinforcement import GEOGRID_UX1600, RIBBED_STEEL_STRIP_75x4

H, L, G, PHI, G_F, PHI_F, Q = 6.0, 4.2, 18.0, 34.0, 19.0, 30.0, 12.0
Q_ALL = 300.0


def _geom(q=Q, **kw):
    return MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           surcharge=q, **kw)


def _ext(q=Q, **kw):
    return check_external_stability(_geom(q), G, PHI, G_F, PHI_F,
                                    q_allowable=Q_ALL, **kw)


def _hand(q, resists, in_bearing):
    """FOS_sliding, FOS_overturning, sigma_v (kPa), e_B (m) by hand."""
    Ka = math.tan(math.radians(45 - PHI / 2)) ** 2
    F1, F2 = 0.5 * Ka * G * H ** 2, Ka * q * H
    M_ot = F1 * H / 3 + F2 * H / 2
    V1, Vq = G * H * L, q * L
    V = V1 + (Vq if resists else 0.0)
    Vb = V1 + (Vq if in_bearing else 0.0)
    fos_s = V * math.tan(math.radians(min(PHI, PHI_F))) / (F1 + F2)
    fos_o = V * L / 2 / M_ot
    e_B = L / 2 - (Vb * L / 2 - M_ot) / Vb
    return fos_s, fos_o, Vb / (L - 2 * max(e_B, 0.0)), e_B


class TestHandCheck:
    def test_no_surcharge(self):
        r = _ext(q=0.0)
        fos_s, fos_o, sigma_v, e_B = _hand(0.0, False, True)
        assert r["FOS_sliding"] == pytest.approx(fos_s, abs=1e-3)   # 2.859
        assert r["FOS_overturning"] == pytest.approx(fos_o, abs=1e-3)  # 5.200
        assert r["sigma_v_kPa"] == pytest.approx(sigma_v, abs=0.05)   # 133.7
        assert r["eccentricity_m"] == pytest.approx(e_B, abs=1e-3)
        # nothing to choose: both readings give the same wall
        assert _ext(q=0.0, live_load_resists=True,
                    surcharge_in_bearing=False) == r

    def test_surcharge_left_out_of_sliding_and_overturning(self):
        r = _ext()
        fos_s, fos_o, sigma_v, e_B = _hand(Q, False, True)
        assert r["FOS_sliding"] == pytest.approx(fos_s, abs=1e-3)
        assert r["FOS_overturning"] == pytest.approx(fos_o, abs=1e-3)
        assert r["sigma_v_kPa"] == pytest.approx(sigma_v, abs=0.05)
        assert r["eccentricity_m"] == pytest.approx(e_B, abs=1e-3)
        # the numbers: 2.339 (was 2.599), 3.900 (was 4.333), 156.0 kPa
        assert (r["FOS_sliding"], r["FOS_overturning"],
                r["sigma_v_kPa"]) == (2.339, 3.9, 156.0)

    def test_live_load_resists_is_the_old_free_body(self):
        r = _ext(live_load_resists=True)
        fos_s, fos_o, sigma_v, _ = _hand(Q, True, True)
        assert r["FOS_sliding"] == pytest.approx(fos_s, abs=1e-3)   # 2.599
        assert r["FOS_overturning"] == pytest.approx(fos_o, abs=1e-3)  # 4.333
        assert r["sigma_v_kPa"] == pytest.approx(sigma_v, abs=0.05)
        assert r["sigma_v_kPa"] == _ext()["sigma_v_kPa"]   # bearing the same

    def test_surcharge_left_out_of_bearing(self):
        r = _ext(surcharge_in_bearing=False)
        fos_s, fos_o, sigma_v, e_B = _hand(Q, False, False)
        assert r["sigma_v_kPa"] == pytest.approx(sigma_v, abs=0.05)
        assert r["eccentricity_m"] == pytest.approx(e_B, abs=1e-3)
        assert r["sigma_v_kPa"] < _ext()["sigma_v_kPa"]
        assert r["FOS_sliding"] == _ext()["FOS_sliding"]

    def test_the_thrust_drives_every_check_either_way(self):
        """The surcharge's thrust K_a q H stays in sliding and overturning
        under both readings: only the weight over the zone is chosen."""
        on, off = _ext(live_load_resists=True), _ext()
        assert off["FOS_sliding"] < _ext(q=0.0)["FOS_sliding"]
        assert on["FOS_sliding"] < _ext(q=0.0)["FOS_sliding"]

    def test_bases_cite_the_pages(self):
        assert "p. 4-19" in LIVE_LOAD_BASIS[False]
        assert "Eq. 4-12" in LIVE_LOAD_BASIS[False]
        assert "p. 4-22" in LIVE_LOAD_BASIS[False]
        assert "p. 4-26" in BEARING_BASIS[True]


class TestBrokenBackslope:
    """Level ground on the zone behind the crest carries the live load
    (Vs = q x level length), which the same choice covers."""

    BETA = math.degrees(math.atan(0.5))      # 2H:1V

    def _geom(self):
        # crest 1.0 m above the top: 2.0 m back, 3.0 m of level ground on L
        return MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                               backfill_slope=self.BETA,
                               broken_backslope_height=1.0, surcharge=Q)

    def test_level_ground_load_left_out_then_counted(self):
        off = check_external_stability(self._geom(), 19.0, 34.0, 19.0, 30.0)
        on = check_external_stability(self._geom(), 19.0, 34.0, 19.0, 30.0,
                                      live_load_resists=True)
        assert off["FOS_sliding"] < on["FOS_sliding"]
        assert off["FOS_overturning"] < on["FOS_overturning"]
        assert off["sigma_v_kPa"] == on["sigma_v_kPa"]
        # by hand: Vs = q x 3.0 m, and sliding scales with V + Vs over V
        t = _external_terms(self._geom(), 19.0, 34.0, 30.0)
        vs = next(f for f in t["vertical_forces"] if f["kind"] == "surcharge")
        assert vs["W"] == pytest.approx(Q * 3.0)
        assert vs["x"] == pytest.approx(5.0 - 3.0 / 2)
        assert on["FOS_sliding"] / off["FOS_sliding"] == pytest.approx(
            (t["V"] + Q * 3.0) / t["V"], rel=2e-3)

    def test_no_live_load_on_the_zone_no_choice(self):
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                               backfill_slope=self.BETA,
                               broken_backslope_height=3.0, surcharge=Q)
        r = analyze_mse_wall(geom, 19.0, 34.0, RIBBED_STEEL_STRIP_75x4)
        assert _live(r) == []


def _live(r):
    """The live-load choices; a ribbed strip's result also gives its F*
    (GM-246)."""
    return [c for c in r.choices if c["what"].startswith("live-load")]


class TestResult:
    def test_choices_with_a_surcharge(self):
        r = analyze_mse_wall(_geom(), G, PHI, RIBBED_STEEL_STRIP_75x4,
                             gamma_foundation=G_F, phi_foundation=PHI_F)
        assert [c["how"] for c in _live(r)] == ["default", "default"]
        assert "live_load_resists=False" in r.choices[0]["chosen"]
        assert "surcharge_in_bearing=True" in r.choices[1]["chosen"]
        assert r.to_dict()["choices"] == r.choices
        assert "Choices:" in r.summary()
        given = analyze_mse_wall(_geom(), G, PHI, RIBBED_STEEL_STRIP_75x4,
                                 phi_foundation=PHI_F, live_load_resists=True)
        assert given.choices[0]["how"] == "given"
        assert given.FOS_sliding > r.FOS_sliding

    def test_no_surcharge_no_choices(self):
        r = analyze_mse_wall(_geom(q=0.0), G, PHI, RIBBED_STEEL_STRIP_75x4)
        assert _live(r) == []
        r = analyze_mse_wall(_geom(q=0.0), G, PHI, GEOGRID_UX1600)
        assert r.choices == [] and "choices" not in r.to_dict()


class TestCalcSteps:
    def _steps(self, result, **extra):
        analysis = {"wall_type": "mse", "geom": _geom(), "gamma_backfill": G,
                    "phi_backfill": PHI, "phi_foundation": PHI_F,
                    "gamma_foundation": G_F, "q_allowable": Q_ALL,
                    "reinforcement": RIBBED_STEEL_STRIP_75x4, **extra}
        return get_calc_steps(result, analysis)

    @staticmethod
    def _items(sections, title):
        return next(s for s in sections if s.title == title).items

    def test_steps_print_the_terms_the_analysis_used(self):
        r = analyze_mse_wall(_geom(), G, PHI, RIBBED_STEEL_STRIP_75x4,
                             gamma_foundation=G_F, phi_foundation=PHI_F,
                             q_allowable=Q_ALL)
        ext = self._items(self._steps(r), "External Stability")
        table = ext[0]
        assert table.headers[-1] == "Counted in"
        live = next(row for row in table.rows
                    if row[0].startswith("Live-load surcharge"))
        assert live[-1] == "bearing only"
        sliding = next(i for i in ext if getattr(i, "title", "")
                       == "Sliding Resistance")
        V1 = G * H * L
        assert f"R = {V1:.1f} × tan(30.00°)" in sliding.substitution
        # R / Pa as printed gives the FOS printed (delta_b = min(phi_r,
        # phi_f), as the module uses; GM-114 had 2/3 phi_f here)
        Ka = rankine_Ka(PHI)
        Pa = 0.5 * Ka * G * H ** 2 + Ka * Q * H
        assert (float(sliding.result_value) / Pa
                == pytest.approx(r.FOS_sliding, abs=2e-3))
        bearing = next(i for i in ext if getattr(i, "title", "").startswith(
            "Eccentricity & Bearing"))
        assert float(bearing.result_value) == pytest.approx(r.sigma_v,
                                                           abs=0.05)

    def test_a_result_built_by_hand_works_the_terms_out(self):
        r = analyze_mse_wall(_geom(), G, PHI, RIBBED_STEEL_STRIP_75x4,
                             gamma_foundation=G_F, phi_foundation=PHI_F,
                             q_allowable=Q_ALL, live_load_resists=True)
        own = self._steps(r, live_load_resists=True)
        r.external_terms = None
        again = self._steps(r, live_load_resists=True)
        sub = [getattr(i, "substitution", None)
               for i in self._items(own, "External Stability")]
        assert sub == [getattr(i, "substitution", None)
                       for i in self._items(again, "External Stability")]
