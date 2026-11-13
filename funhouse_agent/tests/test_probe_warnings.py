"""T2: no warning a method raises is lost (BACKLOG GM-316; module audit
2026-11-12 §6). Each warning attributed to one of GSE's calculation
packages (``funhouse_agent.core_hooks.WARNING_PACKAGES``), and each numpy
floating-point warning, must be in that call's result
(``method_probes.lost_warnings``). Every probe in ``method_probes.json`` is
checked so inside the defaults guard's own run
(``test_defaults_used.test_the_defaults_a_run_lists_are_the_values_it_used``;
review B5: a second pass here cost 209 s and the probes, the smallest
inputs each method runs on, raise none today). The made-up calls below do
raise module warnings on allowed runs; they are the behaviour test."""

import copy
import re
import warnings
from pathlib import Path

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.core_hooks import WARNING_PACKAGES
from funhouse_agent.tests import method_probes as MP

REPO = Path(__file__).resolve().parents[2]
DIRS = MP.WARNING_DIRS


def _check(agent, method, params, before=()):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for b in before:
            D.call_agent(b[0], b[1], copy.deepcopy(b[2]))
        start = len(rec)
        result = D.call_agent(agent, method, copy.deepcopy(params))
        raised = [w for w in rec[start:] if MP.must_reach(w)]
    return raised, MP.lost_warnings(raised, result), result


#: A call that swaps the process's warning filters.
_SWAPS_FILTERS = re.compile(
    r"\b(catch_warnings|simplefilter|filterwarnings|resetwarnings)\s*\(")


def test_no_calculation_package_swaps_the_warning_filters():
    """GM-316 review A1. ``warnings.catch_warnings`` swaps process-wide
    state: two calls in threads whose blocks interleaved left an "ignore
    everything" filter in front for the life of the process. And a
    package's own warning raised inside such a block is recorded for the
    call anyway (``aec_agent_core.call_warnings``), so the block hides
    nothing from the agent. A trial calculation raises no warning instead
    (drilled_shaft ``_alternative``, lateral_pile's capacity search), and
    numpy's floating-point notices go off with ``np.errstate``, which is
    per thread. Test modules are left out."""
    hits = []
    for d in DIRS:
        for path in sorted(Path(d).rglob("*.py")):
            parts = path.relative_to(d).parts
            if "tests" in parts or path.name.startswith("test_"):
                continue
            text = path.read_text("utf-8", errors="replace")
            if re.search(r"^(import pytest|from pytest)", text, re.M):
                continue                   # a test module by another name
            if _SWAPS_FILTERS.search(text):
                hits.append(str(path))
    assert hits == []


def test_the_package_list_is_the_distributions():
    """Every top-level calculation package pyproject ships is listed."""
    try:
        import tomllib
    except ImportError:                                # pragma: no cover
        pytest.skip("tomllib needs Python 3.11")
    data = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))
    shipped = {p.rstrip("*") for p in
               data["tool"]["setuptools"]["packages"]["find"]["include"]}
    app = {"funhouse_agent", "report_ingest", "webapp"}
    listed = {p.split(".")[0] for p in WARNING_PACKAGES}
    assert shipped - app <= listed, sorted(shipped - app - listed)


def test_every_probe_is_checked_in_the_defaults_guards_run():
    """Review B5: the per-probe check lives in the defaults guard's run;
    if that run stops recording warnings, this says so."""
    guard = Path(__file__).with_name("test_defaults_used.py").read_text(
        "utf-8")
    assert "MP.lost_warnings(raised, r0)" in guard
    assert 'warnings.simplefilter("always")' in guard


#: Made-up calls of allowed runs that raise a module warning. The audit's
#: three pile cases (a pile longer than its soil profile) are refused
#: since GM-318, or run on the last layer carried on with the warning in
#: the result itself, so these are warnings the modules still raise:
#: drilled_shaft a casing deeper than the shaft, lateral_pile a soft clay
#: strength above Matlock's range. axial_pile raises none on an allowed
#: run through its tools (its chart notes go to the result's warnings).
RAISING = [
    ("drilled_shaft", "drilled_shaft_capacity",
     {"diameter": 1.0, "length": 10.0, "casing_depth": 12.0,
      "layers": [{"cu": 50, "soil_type": "cohesive", "thickness": 15,
                  "unit_weight": 18}]},
     "exceeds shaft length"),
    ("lateral_pile", "lateral_pile_analysis",
     {"layers": [{"bottom": 15, "c": 120, "eps50": 0.01, "gamma": 18,
                  "model": "SoftClayMatlock", "top": 0}],
      "pile_diameter": 0.8, "pile_length": 15.0, "Vt": 100},
     "is high for soft clay"),
    ("settlement", "consolidation_settlement",
     {"delta_sigma": 50,
      "layers": [{"Cc": 0.05, "Cr": 0.3, "depth_to_center": 4.0, "e0": 0.9,
                  "sigma_v0": 60.0, "thickness": 4.0}]},
     "typically Cr < Cc"),
]


@pytest.mark.parametrize("agent,method,params,text", RAISING,
                         ids=[r[0] for r in RAISING])
def test_a_module_warning_reaches_the_agent_on_every_call(
        agent, method, params, text, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for _ in range(2):                     # a repeat is not lost
        raised, lost, result = _check(agent, method, params)
        assert raised, (agent, "the made-up call raised no warning")
        assert lost == [], lost
        assert any(text in w for w in result["warnings"]), result.get(
            "warnings")
    # and with Python's default once-per-location filter
    first = D.call_agent(agent, method, copy.deepcopy(params))
    again = D.call_agent(agent, method, copy.deepcopy(params))
    for r in (first, again):
        assert any(text in w for w in r.get("warnings") or []), r.get(
            "warnings")

def test_a_warning_raised_by_a_packages_calculation_is_printed_in_it(
        tmp_path, monkeypatch):
    """GM-316 review B4: a calc package's result lists the warnings the
    code raised, and the package prints them too ("Warnings Raised by the
    Calculation", before the steps), so what the agent reports is in the
    document the engineer reads."""
    monkeypatch.chdir(tmp_path)
    agent, method, params, text = RAISING[0]       # casing deeper than shaft
    out = tmp_path / "package.html"
    result = D.call_agent("calc_package", "drilled_shaft_package", {
        **copy.deepcopy(params), "format": "html", "output_path": str(out)})
    assert result["status"] == "success", result
    raised = [w for w in result["warnings"] if text in w]
    assert raised, result["warnings"]
    page = out.read_text("utf-8")
    section = page.index("Warnings Raised by the Calculation")
    for w in raised:
        assert w in page[section:]


def test_a_returned_warning_and_a_raised_one_are_each_said_once(
        tmp_path, monkeypatch):
    """GM-318 review, after the merge with GM-316: a warning the module
    returns in its result (the last layer carried on, extend_last_layer)
    and a Python warning the dispatch catches (a soft clay strength above
    Matlock's range) are each listed once in the result, and the package
    prints the returned one in its own section, not again under "Warnings
    Raised by the Calculation"."""
    monkeypatch.chdir(tmp_path)
    params = {"layers": [{"bottom": 10, "c": 120, "eps50": 0.01,
                          "gamma": 18, "model": "SoftClayMatlock",
                          "top": 0}],
              "extend_last_layer": True,
              "pile_diameter": 0.8, "pile_length": 15.0, "Vt": 100}
    result = D.call_agent("lateral_pile", "lateral_pile_analysis",
                          copy.deepcopy(params))
    ws = result["warnings"]
    assert len(ws) == len(set(ws)) == 2, ws
    assert "extend_last_layer" in ws[0] and "high for soft clay" in ws[1]
    out = tmp_path / "package.html"
    pkg = D.call_agent("calc_package", "lateral_pile_package", {
        **copy.deepcopy(params), "format": "html", "output_path": str(out)})
    assert pkg["status"] == "success", pkg
    assert pkg["warnings"] == ws, pkg["warnings"]
    page = out.read_text("utf-8")
    raised = page[page.index("Warnings Raised by the Calculation"):]
    raised = raised[:raised.index("</table>")]
    assert "high for soft clay" in raised
    assert "extend_last_layer" not in raised
    assert "Soil Below the Profile (Assumed)" in page
