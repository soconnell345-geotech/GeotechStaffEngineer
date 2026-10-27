"""A package that is installed but will not import says so (BACKLOG INF-54).

Each optional package's ``has_X()`` / ``X_import_error()`` pair, and the tool
calls that check it, given a package whose import fails: the check is False
and the message is the import's own error, never "not installed". The
failures are faked with the root conftest's ``fail_import`` fixture and the
error texts are made up, so this runs whether or not the packages are
installed.
"""

import importlib
import sys

import pytest

from funhouse_agent.dispatch import call_agent
from geotech_common.optional_packages import CANNOT_IMPORT, NOT_INSTALLED

#: (import name to break, module holding the checks, has_X, X_import_error,
#: the package name pip installs)
CHECKS = [
    ("Pynite", "pynite_agent.pynite_utils", "has_pynite",
     "pynite_import_error", "PyNiteFEA"),
    ("gstools", "gstools_agent.gstools_utils", "has_gstools",
     "gstools_import_error", "gstools"),
    ("SALib", "salib_agent.salib_utils", "has_salib",
     "salib_import_error", "SALib"),
    ("pystra", "pystra_agent.pystra_utils", "has_pystra",
     "pystra_import_error", "pystra"),
    ("liquepy", "liquepy_agent.liquepy_utils", "has_liquepy",
     "liquepy_import_error", "liquepy"),
    ("pystrata", "pystrata_agent.pystrata_utils", "has_pystrata",
     "pystrata_import_error", "pystrata"),
    ("openseespy", "opensees_agent.opensees_utils", "has_opensees",
     "opensees_import_error", "openseespy"),
    ("eqsig", "seismic_signals_agent.signal_utils", "has_eqsig",
     "eqsig_import_error", "eqsig"),
    ("pygef", "subsurface_characterization.formats.gef", "has_pygef",
     "pygef_import_error", "pygef"),
    ("python_ags4", "subsurface_characterization.formats.ags4", "has_ags4",
     "ags4_import_error", "python-ags4"),
    ("pydiggs", "subsurface_characterization.formats.diggs_validation",
     "has_pydiggs", "pydiggs_import_error", "pydiggs"),
]

#: A made-up error, as pandas 3 raised on numpy 1.26 inside SALib's import.
FAKE = ImportError("inf54 made-up: C extension built against another numpy")
FAKE_TEXT = "ImportError: inf54 made-up: C extension built against another numpy"


@pytest.mark.parametrize("name, module, has, why, package", CHECKS,
                         ids=[c[4] for c in CHECKS])
def test_an_import_that_fails_is_reported_as_its_error(
        fail_import, name, module, has, why, package):
    mod = importlib.import_module(module)
    fail_import(name, FAKE)
    assert getattr(mod, has)() is False
    msg = getattr(mod, why)()
    assert msg == f"{package} {CANNOT_IMPORT}: {FAKE_TEXT}"


@pytest.mark.parametrize("name, module, has, why, package", CHECKS,
                         ids=[c[4] for c in CHECKS])
def test_a_missing_package_is_still_not_installed(
        monkeypatch, name, module, has, why, package):
    mod = importlib.import_module(module)
    for loaded in [m for m in sys.modules if m.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, loaded)
    monkeypatch.setitem(sys.modules, name, None)
    assert getattr(mod, has)() is False
    assert getattr(mod, why)().startswith(f"{package} {NOT_INSTALLED}")


@pytest.mark.parametrize("name, module, has, why, package", CHECKS,
                         ids=[c[4] for c in CHECKS])
def test_has_x_agrees_with_its_import_error(name, module, has, why, package):
    mod = importlib.import_module(module)
    assert getattr(mod, has)() is (getattr(mod, why)() is None)


# ---------------------------------------------------------------------------
# The library's own import helpers raise the import's error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, module, helper, exc_type", [
    ("Pynite", "pynite_agent.pynite_utils", "import_pynite", ImportError),
    ("gstools", "gstools_agent.gstools_utils", "import_gstools", ImportError),
    ("SALib", "salib_agent.salib_utils", "import_salib_sobol_sample",
     ImportError),
    ("liquepy", "liquepy_agent.liquepy_utils", "import_liquepy", RuntimeError),
    ("pystrata", "pystrata_agent.pystrata_utils", "import_pystrata",
     ImportError),
    ("openseespy", "opensees_agent.opensees_utils", "import_ops", ImportError),
    ("eqsig", "seismic_signals_agent.signal_utils", "import_eqsig",
     ImportError),
    ("pygef", "subsurface_characterization.formats.gef", "import_pygef",
     RuntimeError),
    ("python_ags4", "subsurface_characterization.formats.ags4", "import_ags4",
     ImportError),
    ("pydiggs", "subsurface_characterization.formats.diggs_validation",
     "get_dictionary_path", ImportError),
])
def test_import_helpers_raise_the_imports_error(
        fail_import, name, module, helper, exc_type):
    mod = importlib.import_module(module)
    fail_import(name, FAKE)
    with pytest.raises(exc_type) as info:
        getattr(mod, helper)()
    assert CANNOT_IMPORT in str(info.value)
    assert FAKE_TEXT in str(info.value)
    assert NOT_INSTALLED not in str(info.value)


def test_pystra_analyses_raise_the_imports_error(fail_import):
    from pystra_agent import analyze_form
    fail_import("pystra", FAKE)
    with pytest.raises(ImportError, match="inf54 made-up") as info:
        analyze_form([{"name": "R", "dist": "normal", "mean": 2, "stdv": 1}],
                     "R - 1")
    assert str(info.value).startswith("pystra is required for FORM analysis. "
                                      f"pystra {CANNOT_IMPORT}")


# ---------------------------------------------------------------------------
# The tools the agent calls return the import's error
# ---------------------------------------------------------------------------

_SPT = {"depth": [3, 6], "N160": [8, 12], "FC": [5, 10], "gamma": [18, 18],
        "amax_g": 0.3, "gwt_depth": 2.0, "m_w": 7.0}

TOOL_CALLS = [
    ("SALib", "salib", "sobol_analysis",
     {"var_names": ["x1", "x2"], "bounds": [[0, 1], [0, 1]],
      "expression": "x1 + x2"}),
    ("SALib", "salib", "sobol_sample",
     {"var_names": ["x1", "x2"], "bounds": [[0, 1], [0, 1]]}),
    ("pystra", "pystra", "form_analysis",
     {"variables": [{"name": "R", "dist": "normal", "mean": 2, "stdv": 1}],
      "limit_state": "R - 1"}),
    ("gstools", "gstools", "kriging",
     {"x": [0, 1, 2], "y": [0, 1, 2], "values": [1.0, 2.0, 3.0]}),
    ("openseespy", "opensees", "pm4sand_cyclic_dss",
     {"Dr": 0.55, "G0": 600, "hpo": 0.4, "Den": 1.7}),
    ("pystrata", "pystrata", "linear_site_response",
     {"layers": [{"thickness": 10, "Vs": 200, "unit_wt": 18,
                  "soil_model": "linear"}]}),
    ("liquepy", "liquepy", "cpt_liquefaction",
     {"depth": [1, 2, 3], "q_c": [5000, 6000, 7000], "f_s": [50, 60, 70]}),
    ("liquepy", "liquefaction", "liquefaction_analysis", _SPT),
    ("eqsig", "seismic_signals", "response_spectrum",
     {"accel_history": [0.1, 0.2, -0.1], "dt": 0.01}),
    ("pygef", "subsurface", "parse_cpt", {"file_path": "made_up.gef"}),
    ("python_ags4", "subsurface", "read_ags4",
     {"content": '"GROUP","PROJ"\n'}),
    ("pydiggs", "subsurface", "validate_diggs_dictionary",
     {"content": "<Diggs/>"}),
    ("pydiggs", "subsurface", "validate_diggs_schema",
     {"content": "<Diggs/>", "schema_version": "2.5.a"}),
    ("Pynite", "pynite", "continuous_beam",
     {"span_lengths": [5.0], "E": 200e6, "I": 1e-4, "udl": 10.0}),
]


@pytest.mark.parametrize("name, agent, method, params", TOOL_CALLS,
                         ids=[f"{c[1]}.{c[2]}" for c in TOOL_CALLS])
def test_a_tool_reports_the_imports_error(fail_import, name, agent, method,
                                          params):
    from funhouse_agent.adapters import MODULE_REGISTRY
    if agent not in MODULE_REGISTRY:
        pytest.skip(f"{agent} is left out of the registry on this host")
    importlib.import_module(MODULE_REGISTRY[agent]["adapter"])
    fail_import(name, FAKE)
    result = call_agent(agent, method, dict(params))
    assert "error" in result, result
    assert CANNOT_IMPORT in result["error"]
    assert FAKE_TEXT in result["error"]
    assert NOT_INSTALLED not in result["error"]


def test_liquefaction_still_points_at_the_native_procedure(fail_import):
    import funhouse_agent.adapters.liquefaction_adapter  # noqa: F401
    fail_import("liquepy", FAKE)
    result = call_agent("liquefaction", "liquefaction_analysis", dict(_SPT))
    assert "method='nceer2001'" in result["error"]
