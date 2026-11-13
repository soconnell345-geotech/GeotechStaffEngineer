"""fem2d_package (BACKLOG GM-328): the fem2d calc package the renderer
already had, reachable by the agent. Made-up inputs."""

import os

import matplotlib
import pytest

matplotlib.use("Agg")

from funhouse_agent import dispatch as D
from funhouse_agent.adapters import calc_package as cp

_SLOPE = {
    "surface_points": [[0, 10], [10, 10], [20, 0], [30, 0]],
    "soil_layers": [{"name": "clay", "bottom_elevation": -5, "E": 20000,
                     "nu": 0.3, "c": 20, "phi": 20, "psi": 0, "gamma": 18}],
    "nx": 12, "ny": 6,
}
_COLUMN = {
    "analysis_type": "consolidation", "width": 2.0, "depth": 10.0,
    "soil_layers": [{"bottom_elevation": -10.0, "E": 20000.0, "nu": 0.3,
                     "gamma": 0.0}],
    "k": 1e-9, "load_q": 50.0, "time_points": [1e5, 1e7], "nx": 2, "ny": 10,
}


@pytest.fixture(scope="module")
def slope(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("fem") / "slope.html")
    out = D.call_agent("calc_package", "fem2d_package",
                       {**_SLOPE, "output_path": path})
    assert out.get("status") == "success", out
    with open(path, encoding="utf-8") as fh:
        return out, fh.read()


def test_the_agent_has_the_method():
    assert "fem2d_package" in cp.METHOD_REGISTRY
    info = cp.METHOD_INFO["fem2d_package"]
    assert info["parameters"]["analysis_type"]["allowed_values"] == [
        "slope_srm", "excavation", "consolidation", "seepage", "staged"]
    # the tool's own inputs, one source
    assert "surface_points" in info["parameters"]
    assert "time_points" in info["parameters"]


def test_the_package_reports_the_tools_fos(slope):
    out, html = slope
    tool = D.call_agent("fem2d", "fem2d_slope_srm", dict(_SLOPE))
    assert out["FOS"] == tool["FOS"]
    assert out["file_exists"] and out["analysis_type"].startswith("FEM Slope")
    assert f"{tool['FOS']:.3f}" in html
    for words in ("Model Summary", "Material Properties",
                  "Strength Reduction Method", "Checks", "Figure 1"):
        assert words in html, words


def test_the_required_fos_is_stated_not_silent(slope):
    out, html = slope
    assert out["defaults_used"]["FOS_required"] == 1.5
    assert out["defaults_used"]["analysis_type"] == "slope_srm"
    assert out["FOS_required"] == 1.5
    assert out["meets_FOS_required"] is (out["FOS"] >= 1.5)


def test_a_given_required_fos_is_the_one_checked(tmp_path):
    out = D.call_agent("calc_package", "fem2d_package", {
        **_SLOPE, "FOS_required": 1.2,
        "output_path": str(tmp_path / "s.html")})
    assert out["FOS_required"] == 1.2 and out["meets_FOS_required"] is True
    assert "FOS_required" not in out.get("defaults_used", {})


def test_a_required_fos_is_refused_where_no_fos_is_computed(tmp_path):
    out = D.call_agent("calc_package", "fem2d_package", {
        **_COLUMN, "FOS_required": 1.5,
        "output_path": str(tmp_path / "c.html")})
    assert "error" in out and "slope_srm" in out["error"]


def test_consolidation_runs_the_consolidation_tool(tmp_path):
    path = str(tmp_path / "c.html")
    out = D.call_agent("calc_package", "fem2d_package",
                       {**_COLUMN, "output_path": path})
    assert out.get("status") == "success", out
    tool = D.call_agent("fem2d", "fem2d_consolidation",
                        {k: v for k, v in _COLUMN.items()
                         if k != "analysis_type"})
    assert out["max_settlement_m"] == tool["max_settlement_m"]
    assert "FOS_required" not in out
    assert os.path.getsize(path) > 0


def test_an_unknown_analysis_type_is_refused(tmp_path):
    out = D.call_agent("calc_package", "fem2d_package", {
        **_SLOPE, "analysis_type": "footing",
        "output_path": str(tmp_path / "x.html")})
    assert "error" in out and "analysis_type must be one of" in out["error"]


def test_a_misspelt_input_gets_the_tools_refusal(tmp_path):
    out = D.call_agent("calc_package", "fem2d_package", {
        **_SLOPE, "surface_pts": [[0, 0]],
        "output_path": str(tmp_path / "x.html")})
    assert "error" in out and "surface_points" in out["error"]


def test_figures_and_tables_alone(tmp_path):
    figs = D.call_agent("calc_package", "render_figures", {
        "package": "fem2d_package", **_SLOPE, "output_dir": str(tmp_path)})
    assert figs.get("figures"), figs
    assert all(os.path.isfile(f["output_path"]) for f in figs["figures"])
    tables = D.call_agent("calc_package", "render_tables", {
        "package": "fem2d_package", **_SLOPE,
        "output_path": str(tmp_path / "calc.html")})
    assert "error" not in tables, tables
