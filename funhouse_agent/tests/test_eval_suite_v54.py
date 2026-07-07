"""Offline validation of the v5.3/v5.4 eval-suite additions (E10).

Two checks, no live model / no API:
  * every new question is well-formed against the suite schema
    (id/module/complexity/question/expected with named value entries); and
  * every new answer KEY reproduces — i.e. re-running the actual engineering
    module produces a number within the question's own ``rtol`` of the pinned
    ``value``. This is the guarantee that the keys are module ground truth, not
    invented. The two heavier calls (search_rapid_drawdown, fem2d elastic
    footing) are checked the same way but marked so they can be deselected.
"""

import json
import math
import os

import pytest

_SUITE = os.path.join(os.path.dirname(__file__), os.pardir,
                      "geotech_test_suite.json")

# The IDs this E10 batch added (keeps the checks scoped to our additions).
_NEW_IDS = {
    "RDD-1", "RDD-2", "RDD-3", "RDD-4", "RDS-1",
    "INF-1", "INF-2", "INF-3", "NMK-1", "NMK-2", "NMK-3",
    "SPL-1", "SPL-2", "SPL-3", "PPG-1", "TCK-1", "SUR-1",
    "CEI-1", "CEI-2", "EPC-1", "EPC-2", "MSE-1", "FF-1", "CAL-1", "DIR-1",
}


def _load():
    with open(_SUITE, encoding="utf-8") as f:
        return json.load(f)


def _by_id():
    return {q["id"]: q for q in _load()}


def _pinned(q, name):
    for v in q["expected"]["values"]:
        if v["name"] == name:
            return v["value"], v["rtol"], v.get("alt") or []
    raise KeyError(name)


def _matches(got, value, rtol, alt):
    return any(abs(got - c) <= abs(rtol * c) if c else abs(got) <= rtol
               for c in [value] + list(alt))


# ── well-formedness ─────────────────────────────────────────────────────────

def test_suite_loads_and_has_the_new_batch():
    ids = {q["id"] for q in _load()}
    assert _NEW_IDS <= ids, f"missing new ids: {_NEW_IDS - ids}"
    assert len(_load()) >= 96


def test_new_entries_wellformed():
    qs = _by_id()
    for qid in _NEW_IDS:
        q = qs[qid]
        for field in ("id", "module", "complexity", "question", "expected"):
            assert field in q, f"{qid} missing {field}"
        assert q["complexity"] in ("low", "medium", "high"), qid
        vals = q["expected"].get("values")
        assert isinstance(vals, list) and vals, f"{qid} has no values"
        for v in vals:
            assert {"name", "value", "rtol"} <= set(v), f"{qid} value missing keys"
            assert isinstance(v["value"], (int, float)) and v["rtol"] > 0, qid
        assert q["expected"].get("source"), f"{qid} missing source"


def test_new_ids_unique():
    ids = [q["id"] for q in _load()]
    assert len(ids) == len(set(ids))


# ── key reproduction (run the modules) ──────────────────────────────────────

def _dam():
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    return SlopeGeometry(
        surface_points=[(0, 0), (60, 20), (90, 20)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=20,
            bottom_elevation=-2, gamma=20.0, phi=32.0, c_prime=0.0,
            R_c=25.0, R_phi=18.0)])


def _check(qid, name, got):
    value, rtol, alt = _pinned(_by_id()[qid], name)
    assert _matches(got, value, rtol, alt), \
        f"{qid}.{name}: recomputed {got} not within {rtol} of {value} (alt {alt})"


def test_rapid_drawdown_keys_reproduce():
    from slope_stability.analysis import rapid_drawdown_fos
    C = dict(xc=35.0, yc=42.0, radius=40.0, n_slices=40)
    _check("RDD-1", "FOS", rapid_drawdown_fos(_dam(), 18, 3, method="corps_2stage", **C).FOS)
    _check("RDD-2", "FOS", rapid_drawdown_fos(_dam(), 18, 3, method="duncan_3stage", **C).FOS)
    _check("RDD-3", "FOS", rapid_drawdown_fos(_dam(), 18, 3, method="duncan_3stage",
                                              stage3_effective_normal="gle", **C).FOS)
    _check("RDD-4", "FOS", rapid_drawdown_fos(_dam(), 18, 3, method="corps_2stage",
           stage1_phreatic_points=[(0, 18.0), (90, 12.0)], **C).FOS)


def test_infinite_slope_keys_reproduce():
    from slope_stability.analysis import infinite_slope_fos
    _check("INF-1", "FOS", infinite_slope_fos(math.degrees(math.atan(0.5)), 34, 19).FOS)
    _check("INF-2", "FOS", infinite_slope_fos(20, 28, 19, c=8, depth=3,
                                              water_condition="ru", ru=0.3).FOS)
    _check("INF-3", "FOS", infinite_slope_fos(18, 30, 20,
                                              water_condition="seepage_parallel").FOS)


def test_newmark_keys_reproduce():
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.newmark import (yield_acceleration,
                                         newmark_displacement, newmark_jibson2007)
    g = SlopeGeometry(surface_points=[(0, 10), (20, 10), (40, 20), (70, 20)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=20, bottom_elevation=-15,
            gamma=18, gamma_sat=20, phi=25, c_prime=5)])
    _check("NMK-1", "ky", yield_acceleration(g, xc=30, yc=32, radius=26,
                                             method="spencer").ky)
    accel = [3.0] * int(1.0 / 0.005) + [0.0] * int(4.0 / 0.005)
    _check("NMK-2", "displacement_cm",
           newmark_displacement(0.20, accel, 0.005).displacement_cm)
    _check("NMK-3", "displacement_cm", newmark_jibson2007(0.12, 0.35).displacement_cm)


def test_pile_crack_pore_surcharge_keys_reproduce():
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.reinforcement import (StabilizingPile,
                                               ito_matsui_lateral_force)
    from slope_stability.analysis import analyze_slope
    from geotech_common.water import GAMMA_W

    def sp(piles=None):
        g = SlopeGeometry(surface_points=[(-6, 0), (0, 0), (8, 4), (12, 4)],
            soil_layers=[SlopeSoilLayer(name="m1", top_elevation=4,
                bottom_elevation=-5, gamma=15.68, phi=10.0, c_prime=4.9)])
        g.stabilizing_piles = piles
        return g
    _check("SPL-1", "FOS", analyze_slope(sp([StabilizingPile(x=8.75,
        shear_capacity=30.0, spacing=1.5)]), xc=2, yc=8.89, radius=9.11,
        method="bishop", n_slices=40).FOS)
    _check("SPL-2", "FOS", analyze_slope(sp([StabilizingPile(x=8.75,
        shear_capacity=30.0, spacing=1.5, support_convention="passive")]),
        xc=2, yc=8.89, radius=9.11, method="bishop", n_slices=40).FOS)
    _check("SPL-3", "force_kN",
           ito_matsui_lateral_force(10.0, 20.0, 18.0, 2.0, 1.5, z_top=6.0, z_bot=0.0))

    grid = [[float(x), float(z), GAMMA_W * max(5.0 - z, 0.0)]
            for x in range(-2, 53, 5) for z in [-8, -5, 0, 5, 10]]
    gp = SlopeGeometry(surface_points=[(0, 0), (10, 0), (30, 10), (50, 10)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10, bottom_elevation=-8,
            gamma=20, phi=28, c_prime=11)], pore_pressure_points=grid)
    _check("PPG-1", "FOS", analyze_slope(gp, xc=14.29, yc=21.86, radius=22.27,
                                         method="bishop", n_slices=40).FOS)

    zc = 2 * 32 / (20 * math.sqrt((1 - math.sin(math.radians(10)))
                                  / (1 + math.sin(math.radians(10)))))
    gc = SlopeGeometry(surface_points=[(20, 25), (30, 25), (50, 35), (70, 35)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=35, bottom_elevation=10,
            gamma=20, phi=10, c_prime=32)], tension_crack_depth=zc,
        tension_crack_water_depth=zc, tension_crack_side="exit",
        tension_crack_model="truncation")
    _check("TCK-1", "FOS", analyze_slope(gc, xc=38.04, yc=42.94, radius=20.47,
                                         method="bishop", n_slices=60).FOS)

    gs = SlopeGeometry(surface_points=[(0, 0), (20, 0), (40, 10), (60, 10)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10, bottom_elevation=-6,
            gamma=20, phi=28, c_prime=11)],
        surcharges=[(40.0, 20.0, 40.0), (60.0, 40.0, 55.0)])
    _check("SUR-1", "FOS", analyze_slope(gs, xc=25.714, yc=19.143, radius=19.101,
                                         method="bishop", n_slices=40).FOS)


def test_composite_earthpressure_pdf_drawing_keys_reproduce():
    from funhouse_agent.dispatch import call_agent
    _check("CEI-1", "EI_kNm2", call_agent("lateral_pile", "composite_section_ei",
        {"section_type": "filled_pipe", "outer_diameter": 0.6,
         "wall_thickness": 0.012, "fc": 30000})["EI_kNm2"])
    _check("CEI-2", "EI_kNm2", call_agent("lateral_pile", "composite_section_ei",
        {"section_type": "reinforced_concrete", "diameter": 0.6, "fc": 28000,
         "n_bars": 8, "bar_diameter": 0.025, "bar_circle_diameter": 0.45})["EI_kNm2"])
    _check("EPC-1", "Ka", call_agent("retaining_walls", "earth_pressure_coefficient",
        {"phi_deg": 32, "state": "active", "theory": "rankine"})["K"])
    _check("EPC-2", "Kp", call_agent("retaining_walls", "earth_pressure_coefficient",
        {"phi_deg": 32, "state": "passive", "theory": "rankine"})["K"])
    mse = call_agent("retaining_walls", "mse_lrfd_external_stability",
        {"wall_height": 7.0, "gamma_backfill": 19.0, "phi_backfill": 34.0,
         "surcharge": 12.0})
    _check("MSE-1", "sliding_CDR", mse["sliding"]["CDR_governing"])
    _check("CAL-1", "scale_factor", call_agent("pdf_import", "calibrate_scale",
        {"p1": [100.0, 200.0], "p2": [400.0, 200.0], "distance_m": 15.0})["scale_factor"])
    dig = call_agent("drawing_ir", "digitize_drawing", {"file_path": os.path.join(
        os.path.dirname(__file__), os.pardir, "eval_samples", "sample_section.dxf")})
    _check("DIR-1", "n_polyline", dig["counts_by_type"].get("polyline", 0))
    _check("DIR-1", "n_text", dig["counts_by_type"].get("text", 0))


@pytest.mark.slow
def test_heavier_keys_reproduce():
    """The two slower calls (marked slow): the rapid-drawdown SEARCH and the
    fem2d elastic footing. Same reproduce guarantee."""
    from slope_stability.rapid_drawdown import search_rapid_drawdown
    from funhouse_agent.dispatch import call_agent
    _check("RDS-1", "FOS", search_rapid_drawdown(_dam(), 18, 3, method="corps_2stage",
        surface_type="circular", nx=7, ny=6, x_range=(10, 55), y_range=(25, 70),
        x_entry_range=(0, 30), x_exit_range=(35, 60), n_slices=25).FOS)
    _check("FF-1", "max_displacement_m", call_agent("fem2d", "fem2d_foundation",
        {"B": 2.0, "q": 200.0, "depth": 12.0, "E": 30000.0, "nu": 0.3})["max_displacement_m"])
