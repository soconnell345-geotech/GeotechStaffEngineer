"""T2: no warning a method raises is lost (BACKLOG GM-316; module audit
2026-11-12 §6). Every probe in ``method_probes.json`` runs with every Python
warning recorded; each warning attributed to one of GSE's calculation
packages (``funhouse_agent.core_hooks.WARNING_PACKAGES``), and each numpy
floating-point warning, must be in that call's result. A few made-up calls
that do raise a module warning are checked the same way, since the probes
are the smallest inputs each method runs on and raise none today."""

import copy
import importlib.util
import os
import re
import warnings
from pathlib import Path

import pytest

from aec_agent_core.call_warnings import _NOISE, _NUMPY_FP
from funhouse_agent import dispatch as D
from funhouse_agent.core_hooks import WARNING_PACKAGES
from funhouse_agent.tests import method_probes as MP

REPO = Path(__file__).resolve().parents[2]


def _package_dirs() -> tuple:
    out = []
    for name in WARNING_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        for loc in (getattr(spec, "submodule_search_locations", None)
                    or ()):
            out.append(os.path.normcase(os.path.abspath(loc)) + os.sep)
    return tuple(out)


DIRS = _package_dirs()


def _must_reach(w) -> bool:
    if issubclass(w.category, _NOISE):
        return False
    if issubclass(w.category, RuntimeWarning) and _NUMPY_FP.match(
            str(w.message)):
        return True
    return os.path.normcase(os.path.abspath(w.filename)).startswith(DIRS)


def _check(agent, method, params, before=()):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for b in before:
            D.call_agent(b[0], b[1], copy.deepcopy(b[2]))
        start = len(rec)
        result = D.call_agent(agent, method, copy.deepcopy(params))
        raised = [w for w in rec[start:] if _must_reach(w)]
    got = result.get("warnings") if isinstance(result, dict) else None
    got = [str(g) for g in (got or [])]
    lost = [str(w.message) for w in raised if str(w.message) not in got
            and not any(g.startswith("and ") and g.endswith(" more")
                        for g in got)]
    return raised, lost, result


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


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=pytest.mark.slow) if k in MP.SLOW else k
    for k in sorted(MP.PROBES)])
def test_every_warning_a_probe_raises_is_in_its_result(key, tmp_path,
                                                       monkeypatch):
    if MP.skip_reason(key):
        pytest.skip(MP.skip_reason(key))
    monkeypatch.chdir(tmp_path)
    agent, method = key.split("#")[0].split("|")
    probe = MP.PROBES[key]
    _, lost, _ = _check(agent, method, probe["params"],
                        probe.get("before", ()))
    assert lost == [], (key, lost)


#: Made-up calls that raise a module warning (the ones the audit named).
RAISING = [
    ("axial_pile", "axial_pile_capacity",
     {"designation": "HP12x74", "pile_type": "h_pile", "pile_length": 40,
      "layers": [{"friction_angle": 38, "soil_type": "cohesionless",
                  "thickness": 30, "unit_weight": 19}]},
     "exceeds soil profile"),
    ("drilled_shaft", "drilled_shaft_capacity",
     {"diameter": 1.0, "length": 20.0,
      "layers": [{"cu": 50, "soil_type": "cohesive", "thickness": 15,
                  "unit_weight": 18}]},
     "exceeds soil profile"),
    ("lateral_pile", "lateral_pile_analysis",
     {"layers": [{"bottom": 10, "gamma": 19, "k": 8150, "model": "SandAPI",
                  "phi": 32, "top": 0}],
      "pile_diameter": 0.8, "pile_length": 15.0, "Vt": 100},
     "No soil resistance will be applied below"),
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
    agent, method, params, text = RAISING[0]          # pile below profile
    out = tmp_path / "package.html"
    result = D.call_agent("calc_package", "axial_pile_package", {
        **copy.deepcopy(params), "format": "html", "output_path": str(out)})
    assert result["status"] == "success", result
    raised = [w for w in result["warnings"] if text in w]
    assert raised, result["warnings"]
    page = out.read_text("utf-8")
    section = page.index("Warnings Raised by the Calculation")
    for w in raised:
        assert w in page[section:]
