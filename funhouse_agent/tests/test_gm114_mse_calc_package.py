"""GM-114: the MSE wall calc package, built on the mse_wall tool itself.

``retaining_wall_package`` with ``wall_type="mse"`` runs ``mse_wall``'s own
code path (GM-141) and renders the module's MSE calc steps, level backfill
and vertical face only. These tests, with made-up values, hold:

* the package builds, and its numbers are the module's own, called
  directly on the same inputs;
* every calculation step of the package cites a printed page;
* a sloping backfill and a battered face are taken since GM-307; a level
  backfill and a vertical face given as 0 are taken.
"""

import re

import pytest

from calc_package.data_model import CalcStep
from funhouse_agent import dispatch as D
import funhouse_agent.adapters.calc_package as cp


#: A made-up wall: geogrid reinforcement, a live-load surcharge, a given
#: foundation friction angle and allowable bearing pressure.
WALL = {"wall_type": "mse", "wall_height": 5.5, "gamma_backfill": 19.0,
        "phi_backfill": 33.0, "reinforcement_length": 4.2,
        "reinforcement_spacing": 0.55, "surcharge": 9.0,
        "phi_foundation": 29.0, "q_allowable": 320.0,
        "reinforcement_name": "geogrid_gec11_e1_gg2"}

#: The same with steel strips, no q_allowable (bearing not checked).
STRIPS = {"wall_type": "mse", "wall_height": 7.0, "gamma_backfill": 20.0,
          "phi_backfill": 34.0, "reinforcement_length": 5.5,
          "reinforcement_spacing": 0.75,
          "reinforcement_name": "ribbed_steel_strip_50x4",
          "reinforcement_horizontal_spacing": 0.9}


def _module(params):
    """The module's own analysis on ``params`` (the tool's defaults
    written out: no backslope, c_foundation 0)."""
    from retaining_walls import MSEWallGeometry, analyze_mse_wall
    from retaining_walls.reinforcement import GEOGRID_GEC11_E1_GG2
    assert params["reinforcement_name"] == "geogrid_gec11_e1_gg2"
    return analyze_mse_wall(
        MSEWallGeometry(
            wall_height=params["wall_height"],
            reinforcement_length=params["reinforcement_length"],
            reinforcement_spacing=params["reinforcement_spacing"],
            surcharge=params["surcharge"]),
        gamma_backfill=params["gamma_backfill"],
        phi_backfill=params["phi_backfill"],
        reinforcement=GEOGRID_GEC11_E1_GG2,
        phi_foundation=params["phi_foundation"],
        q_allowable=params["q_allowable"])


def _package(params, tmp_path, name="mse.html"):
    out = D.call_agent("calc_package", "retaining_wall_package", {
        **params, "output_path": str(tmp_path / name)})
    assert out.get("status") == "success", out
    return out


def test_the_mse_package_builds(tmp_path):
    out = _package(WALL, tmp_path)
    assert out["analysis_type"] == "MSE Wall"
    assert out["file_exists"] and out["file_size_bytes"] > 0
    html = (tmp_path / "mse.html").read_text(encoding="utf-8")
    for title in ("Earth Pressure Computation", "External Stability",
                  "Internal Stability", "Overall Assessment"):
        assert title in html


def test_its_numbers_are_the_modules(tmp_path):
    out = _package(WALL, tmp_path)
    r = _module(WALL)
    assert out["FOS_sliding"] == round(r.FOS_sliding, 2)
    assert out["FOS_overturning"] == round(r.FOS_overturning, 2)
    assert out["FOS_bearing"] == round(r.FOS_bearing, 2)
    assert out["sigma_v_kPa"] == round(r.sigma_v, 1)
    assert out["n_levels"] == len(r.internal_results)
    assert out["FOS_pullout_min"] == round(
        min(lv["FOS_pullout"] for lv in r.internal_results), 2)
    assert out["FOS_rupture_min"] == round(
        min(lv["FOS_rupture"] for lv in r.internal_results), 2)
    assert out["passes_internal"] is r.all_pass_internal
    assert out["checks_failed"] == r.checks_failed
    # and the file prints them as the module worked them out
    html = (tmp_path / "mse.html").read_text(encoding="utf-8")
    for value in (r.FOS_sliding, r.FOS_overturning, r.FOS_bearing):
        assert f"{value:.3f}" in html
    assert f"{r.sigma_v:.1f}" in html


def test_it_renders_the_tools_own_result(monkeypatch):
    seen = {}

    def capture(module, result, analysis, params, analysis_type,
                extra=None, run=None):
        seen.update(result=result, analysis=analysis, type=analysis_type)
        return {"status": "success"}
    monkeypatch.setattr(cp, "_build_response", capture)
    cp.METHOD_REGISTRY["retaining_wall_package"](dict(WALL))
    tool = D.call_agent("retaining_walls", "mse_wall", {
        k: v for k, v in WALL.items() if k != "wall_type"})
    assert seen["type"] == "MSE Wall"
    assert seen["analysis"]["wall_type"] == "mse"
    assert seen["result"].FOS_sliding == pytest.approx(tool["FOS_sliding"],
                                                       abs=5e-3)
    assert seen["result"].to_dict()["internal_results"] == \
        tool["internal_results"]


@pytest.mark.parametrize("params", [WALL, STRIPS], ids=["geogrid", "strips"])
def test_every_step_cites_a_printed_page(params):
    from funhouse_agent.adapters.retaining_walls import run_mse_wall
    from retaining_walls.calc_steps import get_calc_steps
    result, used = run_mse_wall(
        {k: v for k, v in params.items() if k != "wall_type"})
    steps = [item for section in get_calc_steps(
        result, {**used, "wall_type": "mse"}) for item in section.items
        if isinstance(item, CalcStep)]
    assert len(steps) >= 10
    for step in steps:
        assert re.search(r"\bpp?\. (?:[A-Z]\d*|\d+)-\d+", step.reference or ""), \
            (step.title, step.reference)


def test_the_steps_cite_the_pages_of_the_print(tmp_path):
    _package(WALL, tmp_path)
    html = (tmp_path / "mse.html").read_text(encoding="utf-8")
    for cite in ("Eq. 4-1 (p. 4-15)", "Eqs. 4-5, 4-6 (p. 4-20)",
                 "Eq. 4-12 (p. 4-21)", "Eq. 4-18 (p. 4-26)",
                 "Eqs. 4-19, 4-20 (p. 4-27)", "Fig. 4-10 (p. 4-37)",
                 "Eq. 4-32a", "Vol II App. F.1, p. F-2"):
        assert cite in html, cite


def test_a_wall_without_an_allowable_bearing_pressure_builds(tmp_path):
    out = _package(STRIPS, tmp_path, "strips.html")
    assert out["checks_not_run"] == ["bearing"]
    assert out["FOS_bearing"] is None
    html = (tmp_path / "strips.html").read_text(encoding="utf-8")
    assert "NOT CHECKED" in html


def test_a_sloping_backfill_is_taken_since_gm307(tmp_path):
    # GM-114 refused it; GM-307 prints GEC-11 Sec. 4.4.4's terms
    out = _package({**WALL, "surcharge": 0.0, "backfill_slope": 18.0},
                   tmp_path, "slope.html")
    assert out["status"] == "success"
    assert "Eqs. 4-2, 4-3 (p. 4-15)" in (tmp_path / "slope.html").read_text(
        encoding="utf-8")


def test_a_battered_face_is_taken_since_gm307(tmp_path):
    # GM-114 refused it; GM-307 builds GEC-11 Vol I p. 4-15 and Eq. 4-26
    out = _package({**WALL, "face_batter": 11.0}, tmp_path, "batter.html")
    assert out["status"] == "success"
    assert "Eq. 4-26 (p. 4-36)" in (tmp_path / "batter.html").read_text(
        encoding="utf-8")


def test_a_level_backfill_and_a_vertical_face_given_as_zero_are_taken(
        tmp_path):
    out = _package({**WALL, "backfill_slope": 0.0, "face_batter": 0},
                   tmp_path)
    assert out["FOS_sliding"] == round(_module(WALL).FOS_sliding, 2)


def test_the_tools_name_selects_the_mse_wall(tmp_path):
    out = _package({**WALL, "wall_type": "mse_wall"}, tmp_path)
    assert out["analysis_type"] == "MSE Wall"


def test_the_cantilever_package_is_unchanged(tmp_path):
    out = _package({"wall_height": 4.0, "gamma_backfill": 18.0,
                    "phi_backfill": 30.0}, tmp_path, "cw.html")
    assert out["analysis_type"] == "Retaining Wall"
    assert "sigma_v_kPa" not in out
    assert out["defaults_used"]["wall_type"] == "cantilever"
