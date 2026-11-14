"""GM-140 step 10, item 4 (D45): the slope packages' inputs change the
result or are refused. Made-up slope (the packages' probes)."""

import json
import os

import pytest

from funhouse_agent.adapters import calc_package as adapter

PROBES = os.path.join(os.path.dirname(__file__), "method_probes.json")


def _probe(name):
    with open(PROBES, encoding="utf-8") as fh:
        return dict(json.load(fh)["calc_package|" + name]["params"])


def _run(name, tmp_path, **extra):
    params = {**_probe(name), "format": "html",
              "output_path": str(tmp_path / "pkg.html"), **extra}
    try:
        return adapter.METHOD_REGISTRY[name](params)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}


def test_fos_required_moves_the_stability_check(tmp_path):
    a = _run("slope_stability_package", tmp_path, FOS_required=1.5)
    b = _run("slope_stability_package", tmp_path, FOS_required=3.0)
    assert a["FOS"] == b["FOS"]
    assert a["is_stable"] is True and b["is_stable"] is False


def test_a_time_budget_the_search_cannot_finish_in_stops_it(tmp_path):
    full = _run("slope_report_package", tmp_path)
    cut = _run("slope_report_package", tmp_path, time_budget_s=0.01)
    assert full["status"] == "success"
    assert cut.get("n_surfaces_evaluated") != full["n_surfaces_evaluated"]


def test_a_tol_that_changes_nothing_is_refused(tmp_path):
    """Spencer's method runs at min(tol, 1e-4) / 10, so a tol above 1e-4
    would change nothing: refused. Bishop's simplified method takes tol
    as given."""
    bad = _run("slope_report_package", tmp_path, tol=1e-3)
    assert bad["status"] == "error" and "changes nothing" in bad["error"]
    ok = _run("slope_report_package", tmp_path, tol=1e-3, method="bishop")
    assert ok["status"] == "success"
    fine = _run("slope_report_package", tmp_path, tol=1e-5)
    assert fine["status"] == "success"


def test_an_unknown_interslice_function_is_refused(tmp_path):
    bad = _run("slope_report_package", tmp_path, f_interslice="sine",
               method="gle")
    assert bad["status"] == "error"


# ---------------------------------------------------------------------------
# Items 9 to 12 (GM-346): the later inputs reach the packages and the
# report prints what the result states
# ---------------------------------------------------------------------------

def _worked_example(eid):
    from funhouse_agent.worked_examples import load_examples
    return next(e for e in load_examples() if e["id"] == eid)


def _html(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_the_stability_package_takes_a_tension_crack(tmp_path):
    """WE-SLOPE-1 as printed: the reverse-curvature toe circle closed by its
    given 6.096 m crack (step 4). The package runs the tool's own path, so
    its factor of safety is the tool's."""
    from funhouse_agent import dispatch as D
    p = dict(_worked_example("WE-SLOPE-1")["dispatch_calls"][0]["params"])
    tool = D.call_agent("slope_stability", "analyze_slope", dict(p))
    out = str(tmp_path / "we1.html")
    pkg = D.call_agent("calc_package", "slope_stability_package",
                       {**p, "format": "html", "output_path": out})
    assert pkg["status"] == "success", pkg.get("error")
    assert pkg["FOS"] == pytest.approx(tool["FOS"], abs=5e-4)
    html = _html(out)
    assert "Tension crack depth" in html and "6.10" in html


def test_the_stability_package_takes_the_arc_ends(tmp_path):
    """"Where a circle ends": the arc named by its two ends is the surface
    analysed, as analyze_slope takes it."""
    from funhouse_agent import dispatch as D
    probe = {"surface_points": [[0, 10], [10, 10], [20, 0], [30, 0],
                                [40, 10], [55, 10]],
             "soil_layers": [{"name": "clay", "top_elevation": 10,
                              "bottom_elevation": -20, "gamma": 18.0,
                              "c_prime": 5, "phi": 20}],
             "xc": 24.5, "yc": 16.0, "radius": 15.7,
             "arc_x_start": 9.9917, "arc_x_end": 18.5141}
    tool = D.call_agent("slope_stability", "analyze_slope", dict(probe))
    pkg = D.call_agent("calc_package", "slope_stability_package",
                       {**probe, "format": "html",
                        "output_path": str(tmp_path / "arc.html")})
    assert pkg["status"] == "success", pkg.get("error")
    assert pkg["FOS"] == pytest.approx(tool["FOS"], abs=5e-4)


_SUPPORTED = dict(
    surface_points=[[0, 10], [20, 10], [40, 20], [70, 20]],
    soil_layers=[{"name": "soil", "top_elevation": 20,
                  "bottom_elevation": -15, "gamma": 19, "gamma_sat": 20,
                  "c_prime": 8, "phi": 25}],
    gwt_points=[[0, 9], [22, 9], [38, 17], [70, 17]],
    geosynthetics=[{"elevation": 12.0, "T_allow": 20.0}])


def test_the_later_inputs_reach_the_package_and_the_report_names_them(
        tmp_path):
    """water_line_kind (step 5a), base_in_tension (the GLE step) and the
    supports (step 8) reach the package; the report names the reading of
    the water line, the rule for a base in tension, Method A and the
    measure of force, GEC 11 Eq. 9-17 as the result states it, and the
    pullout not checked among the warnings."""
    from funhouse_agent import dispatch as D
    out = str(tmp_path / "r.html")
    pkg = D.call_agent("calc_package", "slope_stability_package", {
        **_SUPPORTED, "xc": 30, "yc": 32, "radius": 26, "method": "spencer",
        "water_line_kind": "water_table", "base_in_tension": "no_friction",
        "format": "html", "output_path": out})
    assert pkg["status"] == "success", pkg.get("error")
    tool = D.call_agent("slope_stability", "analyze_slope", {
        **_SUPPORTED, "xc": 30, "yc": 32, "radius": 26, "method": "spencer",
        "water_line_kind": "water_table", "base_in_tension": "no_friction"})
    assert pkg["FOS"] == pytest.approx(tool["FOS"], abs=5e-4)
    html = _html(out)
    for words in ("water table", "no_friction", "Method A", "allowable",
                  "Eq. 9-17", "was not checked"):
        assert words in html, words
    # the pullout not checked is a warning, not only a note
    warned = html[html.index("Warnings on the Analyzed Surface"):]
    assert "was not checked" in warned


def test_the_report_package_takes_the_later_inputs(tmp_path):
    from funhouse_agent import dispatch as D
    out = str(tmp_path / "s.html")
    pkg = D.call_agent("calc_package", "slope_report_package", {
        **_SUPPORTED, "nx": 3, "ny": 3, "base_in_tension": "no_friction",
        "tension_crack_depth_basis": "rankine", "format": "html",
        "output_path": out})
    assert pkg["status"] == "success", pkg.get("error")
    html = _html(out)
    for words in ("piezometric", "no_friction", "Method A", "Rankine",
                  "The Method Each Family Was Scored With"):
        assert words in html, words


def test_the_comparison_leaves_out_the_oms_with_its_reason():
    """The Ordinary Method of Slices is left out of the comparison where a
    support crosses the surface, with its reason (Duncan, Wright and
    Brandon 2014, p. 93)."""
    from slope_stability.analysis import analyze_slope
    from slope_stability.calc_steps import _comparison_section
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.nails import SoilNail
    g = SlopeGeometry(
        surface_points=[(0, 10), (20, 10), (40, 20), (70, 20)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=20,
                                    bottom_elevation=-15, gamma=19,
                                    c_prime=8, phi=25)],
        nails=[SoilNail(x_head=30, z_head=15, length=20, inclination=30)])
    r = analyze_slope(g, xc=30, yc=32, radius=26, method="spencer",
                      compare_methods=True, include_slice_data=True)
    row = [x for x in _comparison_section(r, {}).items[0].rows
           if x[0] == "Fellenius"][0]
    assert row[1] == "no solution"
    assert "left out of the comparison" in row[2] and "p. 93" in row[2]


def test_the_checks_table_names_the_m_alpha_denominator_and_c10b():
    """The low_m_alpha finding with its slice and denominator (EM
    1110-2-1902 C-10b) in the Printed Acceptability Checks table."""
    from slope_stability.analysis import analyze_slope
    from slope_stability.calc_steps import _acceptability_section
    from slope_stability.slip_surface import PolylineSlipSurface
    from slope_stability.tests.test_acceptability import _STEEP_EXIT, _slope
    res = analyze_slope(_slope(c=1.0, phi=36.0, bottom=-12.0),
                        slip_surface=PolylineSlipSurface(points=_STEEP_EXIT),
                        method="morgenstern_price")
    rows = _acceptability_section(res).items[0].rows
    row = [x for x in rows if x[0] == "m_alpha below 0.2 on a slice"][0]
    assert row[2].startswith("slices ") and "denominator" in row[2]
    assert "C-10b" in row[4]
