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
    # master's refusal of a value outside allowed_values (its wording)
    assert "error" in out and (
        "analysis_type='footing' is not one of the values it takes"
        in out["error"]) and "slope_srm" in out["error"]


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


# --- GM-328 review round -------------------------------------------------
# A made-up 1:1 slope 10 m high, small mesh: the SRM brackets its FOS on
# the default srf_range; a narrowed range gives the bounds.
_PROBE = {
    "surface_points": [[0, 10], [10, 10], [20, 0], [30, 0]],
    "soil_layers": [{"name": "clay", "bottom_elevation": -5, "E": 20000,
                     "nu": 0.3, "c": 18, "phi": 22, "psi": 0, "gamma": 19}],
    "nx": 10, "ny": 5,
}


def _bound(fos, converged=True, basis="nonconvergence"):
    from types import SimpleNamespace
    return SimpleNamespace(FOS=fos, converged=converged, fos_basis=basis)


def test_the_check_is_true_to_the_bound_the_srm_gives():
    # A1: an upper bound fails only at or below FOS_required, a lower
    # bound passes only at or above it, anything else is not determined
    from fem2d.calc_steps import srm_fos_said, srm_meets
    upper = _bound(2.0, converged=False)
    assert srm_meets(upper, 2.0) == (False, "")
    meets, note = srm_meets(upper, 1.5)
    assert meets is None and "srf_range starting at or below 1.5" in note
    lower = _bound(1.2, basis="range_exhausted")
    assert srm_meets(lower, 1.2) == (True, "")
    meets, note = srm_meets(lower, 1.5)
    assert meets is None and "srf_range reaching above 1.5" in note
    assert srm_meets(_bound(1.456), 1.5) == (False, "")
    assert srm_meets(_bound(1.6, basis="blowup"), 1.5) == (True, "")
    assert srm_fos_said(upper).startswith("FOS < 2.000: ")
    assert srm_fos_said(lower).startswith("FOS >= 1.200: ")
    assert srm_fos_said(_bound(1.456)) == "FOS = 1.456"


def test_a_fos_below_the_lowest_srf_tried_is_not_a_pass(tmp_path):
    # A1, the reviewer's probe: the slope fails at srf 2.0, the lowest tried
    path = str(tmp_path / "upper.html")
    out = D.call_agent("calc_package", "fem2d_package", {
        **_PROBE, "srf_range": [2.0, 3.0], "output_path": path})
    assert out.get("status") == "success", out
    assert out["converged"] is False and out["fos_basis"] == "nonconvergence"
    assert out["meets_FOS_required"] is None
    assert "srf_range starting at or below 1.5" in out["FOS_required_note"]
    assert any("upper bound" in w for w in out["warnings"])
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    assert "NOT DETERMINED" in html and "FOS &lt; 2.000" in html
    assert "FOS_computed" not in html
    failed = D.call_agent("calc_package", "fem2d_package", {
        **_PROBE, "srf_range": [2.0, 3.0], "FOS_required": 2.0,
        "output_path": str(tmp_path / "upper2.html")})
    assert failed["meets_FOS_required"] is False


def test_a_fos_above_the_highest_srf_tried_is_a_lower_bound(tmp_path):
    # A1: no failure up to srf 1.2, so FOS >= 1.2
    path = str(tmp_path / "lower.html")
    out = D.call_agent("calc_package", "fem2d_package", {
        **_PROBE, "srf_range": [0.5, 1.2], "FOS_required": 1.0,
        "output_path": path})
    assert out.get("status") == "success", out
    assert out["fos_basis"] == "range_exhausted"
    assert out["meets_FOS_required"] is True
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    assert "FOS_lower_bound" in html and "FOS &gt;= 1.200" in html


def test_the_package_says_every_number_the_tool_returns(slope, tmp_path):
    # B1: every single-number key of the tool, under its own key
    out, _ = slope
    tool = D.call_agent("fem2d", "fem2d_slope_srm", dict(_SLOPE))
    column = D.call_agent("calc_package", "fem2d_package", {
        **_COLUMN, "output_path": str(tmp_path / "c.html")})
    column_tool = D.call_agent("fem2d", "fem2d_consolidation",
                               {k: v for k, v in _COLUMN.items()
                                if k != "analysis_type"})
    for pkg, own in ((out, tool), (column, column_tool)):
        for key, value in own.items():
            if key == "analysis_type" or not (
                    value is None or isinstance(value, (bool, int, float,
                                                        str))):
                continue
            assert key in pkg and pkg[key] == value, key


@pytest.mark.parametrize("bad", [-1, 0.9, "abc", float("nan"), True, None])
def test_a_required_fos_that_is_no_factor_of_safety_is_refused(bad,
                                                               tmp_path):
    # B2: refused before the FEM run, naming the parameter
    out = D.call_agent("calc_package", "fem2d_package", {
        **_SLOPE, "FOS_required": bad,
        "output_path": str(tmp_path / "x.html")})
    assert "error" in out and "FOS_required takes a finite number" in (
        out["error"])
    assert not os.path.exists(tmp_path / "x.html")


def test_the_assumed_required_fos_says_it_is_the_packages_choice(slope):
    # C2: the label is whole
    _, html = slope
    assert ("Required factor of safety, this package&#39;s choice, not a "
            "printed requirement") in html or (
        "Required factor of safety, this package's choice, not a printed "
        "requirement") in html
