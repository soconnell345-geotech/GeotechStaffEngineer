"""A calc package works out an input it was not given from the inputs it
was, as the analysis tool does (backlog GM-120).

The seismic package used KA = 0.333 (Rankine Ka for phi 30 deg) whatever
phi was; the settlement package used L = 1.0 m whatever B was; the ground
improvement package printed E_column 80,000 kPa when E_pier was given.
Each derived value must be the one the analysis tool computes from the
same inputs, by the same function; a value given explicitly is used as
given.
"""

import math

import pytest

import funhouse_agent.adapters.calc_package as cp
from funhouse_agent.adapters.seismic_geotech import (
    METHOD_REGISTRY as SEISMIC_TOOL,
)
from funhouse_agent.adapters.settlement import (
    METHOD_REGISTRY as SETTLEMENT_TOOL,
)
from sheet_pile.earth_pressure import coulomb_Ka, rankine_Ka


@pytest.fixture
def built(monkeypatch):
    """Run a package builder and return what it handed to the package
    renderer: (module, result, analysis)."""
    seen = {}

    def _capture(module, result, analysis, params, analysis_type,
                 extra=None, run=None):
        seen.update(module=module, result=result, analysis=analysis)
        return {"status": "success"}

    monkeypatch.setattr(cp, "_build_response", _capture)

    def run(method, params):
        assert cp.METHOD_REGISTRY[method](dict(params))["status"] == "success"
        return seen["module"], seen["result"], seen["analysis"]
    return run


def _by_what(choices):
    return {c["what"]: c for c in choices}


_MO = {"analysis_type": "seismic_earth_pressure", "phi": 34, "kh": 0.15,
       "gamma": 19.0, "H": 6.0}


# --- seismic_package: KA ---------------------------------------------------

def test_ka_not_given_is_rankine_from_phi_as_the_analysis_tool(built):
    _, r, _ = built("seismic_package", _MO)
    ka = rankine_Ka(34)
    assert ka == pytest.approx(0.2827, abs=1e-4)          # not 0.333
    assert r.PA_static == pytest.approx(round(0.5 * 19 * 36 * ka, 2))
    # the same numbers the analysis tool works out from the same inputs
    tool = SEISMIC_TOOL["seismic_earth_pressure"](
        {k: v for k, v in _MO.items() if k != "analysis_type"})
    assert tool["KA_static"] == round(ka, 4)
    assert r.PA_static == tool["PA_static_kN_per_m"]
    assert r.delta_PAE == tool["delta_PAE_kN_per_m"]
    assert r.PAE_total == tool["PAE_total_kN_per_m"]


def test_ka_is_rankine_not_coulomb_even_with_wall_friction_given(built):
    # The analysis tool takes static KA as Rankine (no wall friction) and
    # delta only into KAE; the package follows it rather than working out a
    # Coulomb Ka with delta, which would be a different method.
    params = {**_MO, "phi": 30, "delta": 15}
    _, r, _ = built("seismic_package", params)
    tool = SEISMIC_TOOL["seismic_earth_pressure"](
        {k: v for k, v in params.items() if k != "analysis_type"})
    assert r.PA_static == tool["PA_static_kN_per_m"]
    assert r.PA_static == pytest.approx(
        round(0.5 * 19 * 36 * rankine_Ka(30), 2))
    coulomb = round(0.5 * 19 * 36 * coulomb_Ka(30, 15), 2)
    assert abs(r.PA_static - coulomb) > 1.0
    assert r.PA_static != round(0.5 * 19 * 36 * 0.333, 2)


def test_ka_not_given_is_stated_as_the_modules_choice(built):
    _, r, _ = built("seismic_package", _MO)
    ka = _by_what(r.choices)["static KA"]
    assert ka["chosen"] == "Rankine, 0.2827"
    assert ka["how"] == "module"
    assert ka["basis"].startswith("vertical wall, level backfill, no wall "
                                  "friction")
    assert "KA given (KA)" in ka["alternatives"]
    assert "assumed" not in ka["chosen"]


def test_ka_given_is_used_as_given(built):
    _, r, _ = built("seismic_package", {**_MO, "KA": 0.30})
    assert r.PA_static == pytest.approx(round(0.5 * 19 * 36 * 0.30, 2))
    assert _by_what(r.choices)["static KA"] == {
        "what": "static KA", "chosen": "0.3", "how": "given"}


def test_printed_steps_say_which_ka_gives_pa(tmp_path):
    def html(params):
        out = cp._generate_seismic_package(
            {**params, "output_path": str(tmp_path / "s.html")})
        return open(out["output_path"], encoding="utf-8").read()

    worked = html(_MO)
    assert "Static Active Coefficient (Rankine Ka)" in worked
    assert "KA not given: this Ka gives PA and" in worked
    # the PA step's substitution and its result use the same Ka
    assert "6.00² × 0.2827" in worked and "96.69" in worked
    given = html({**_MO, "KA": 0.30})
    assert "Static Active Coefficient (Rankine Ka, for comparison)" in given
    assert "KA not given" not in given
    # a KA given is the one printed where PA is worked out, and in the
    # summary: PA = 0.5 x 19 x 6^2 x 0.30 = 102.60 (it printed 0.2827)
    assert "6.00² × 0.3000" in given and "102.60" in given
    assert "6.00² × 0.2827" not in given
    summary = given[given.index("Seismic Earth Pressure Summary"):]
    assert "0.3000" in summary[:summary.index("PAE (total seismic)") + 200]


def test_pressure_figure_labels_the_ka_used(built, monkeypatch):
    import matplotlib.axes
    from seismic_geotech import calc_steps

    labels = []
    real = matplotlib.axes.Axes.fill_betweenx

    def spy(self, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_betweenx", spy)
    for params in (_MO, {**_MO, "KA": 0.30}):
        _, r, analysis = built("seismic_package", params)
        calc_steps._plot_pressure_diagram(r, analysis)
    assert "Static PA (Ka=0.283)" in labels
    assert "Static PA (Ka=0.300)" in labels


def test_ka_has_no_declared_default():
    spec = cp.METHOD_INFO["seismic_package"]["parameters"]["KA"]
    assert "default" not in spec
    assert "Rankine" in spec["description"]


# --- settlement_package: L -------------------------------------------------

_SETTLE = {"q_applied": 150, "q_overburden": 20, "B": 3.0,
           "Es_immediate": 15000,
           "consolidation_layers": [{"thickness": 4, "depth_to_center": 3,
                                     "e0": 0.9, "Cc": 0.3, "Cr": 0.05,
                                     "sigma_v0": 50, "sigma_p": 70}]}


def test_settlement_l_not_given_is_b_as_the_analysis_tool(built):
    _, r, analysis = built("settlement_package", _SETTLE)
    assert analysis.L == analysis.B == 3.0                  # not 1.0
    tool = SETTLEMENT_TOOL["combined_settlement_analysis"](
        {**{k: v for k, v in _SETTLE.items() if k != "Es_immediate"},
         "Es": _SETTLE["Es_immediate"]})
    assert round(r.consolidation * 1000, 2) == tool["consolidation_mm"]
    assert round(r.total * 1000, 2) == tool["total_mm"]
    assert round(r.consolidation * 1000, 2) == pytest.approx(60.45, abs=0.01)


def test_settlement_l_given_is_used_as_given(built):
    _, _, analysis = built("settlement_package", {**_SETTLE, "L": 6.0})
    assert (analysis.B, analysis.L) == (3.0, 6.0)


def test_settlement_l_has_no_declared_default():
    spec = cp.METHOD_INFO["settlement_package"]["parameters"]["L"]
    assert "default" not in spec and "L = B" in spec["description"]


# --- ground_improvement_package: inputs given by an older name -------------

def test_aggregate_pier_inputs_given_by_older_names_are_printed(built):
    from ground_improvement.calc_steps import get_input_summary
    module, r, analysis = built("ground_improvement_package", {
        "method": "aggregate_piers", "diameter": 0.8, "spacing": 2.0,
        "E_pier": 100000.0, "E_soil": 4000.0})
    assert module == "ground_improvement"
    assert analysis["E_column"] == 100000.0 and "E_pier" not in analysis
    items = {i.name: i.value for i in get_input_summary(r, analysis)}
    # E_column feeds only a warning: stated in the choices, not printed as
    # an input (GM-142)
    assert "E_c" not in items
    chosen = {c["what"]: c["chosen"] for c in r.choices}
    assert chosen["column modulus E_c"] == "100000 kPa"     # not 80,000
    assert items["d_c"] == 0.8


def test_aggregate_pier_tool_names_reach_the_package(built):
    # The aggregate_piers analysis tool's own names: the package ran with
    # n = 5.0 when given stress_concentration_ratio = 3 and dropped
    # column_modulus, and still said success.
    from funhouse_agent.adapters.ground_improvement import (
        METHOD_REGISTRY as GI_TOOL,
    )
    from ground_improvement.calc_steps import get_input_summary
    given = {"column_diameter": 0.8, "spacing": 2.0,
             "stress_concentration_ratio": 3.0, "column_modulus": 90000.0,
             "soil_modulus": 4500.0, "S_unreinforced": 80.0}
    _, r, analysis = built("ground_improvement_package",
                           {"method": "aggregate_piers", **given})
    assert r.stress_concentration_ratio == 3.0
    assert (analysis["n"], analysis["E_column"], analysis["E_soil"]) == \
        (3.0, 90000.0, 4500.0)
    items = {i.name: i.value for i in get_input_summary(r, analysis)}
    assert (items["E_s"], items["n"]) == (4500.0, 3.0)
    chosen = {c["what"]: c["chosen"] for c in r.choices}
    assert chosen["column modulus E_c"] == "90000 kPa"
    tool = GI_TOOL["aggregate_piers"](dict(given))
    assert r.settlement_improved_mm == pytest.approx(
        tool["settlement_improved_mm"], abs=0.05)
    spec = cp.METHOD_INFO["ground_improvement_package"]
    for name in ("stress_concentration_ratio", "column_modulus",
                 "soil_modulus"):
        assert name in spec["parameter_aliases"]


def test_aggregate_pier_column_modulus_default_is_stated(built):
    # GM-142: E_column feeds only a warning, so the package states its
    # default in the choices rather than printing it as an input
    from ground_improvement.calc_steps import get_input_summary
    _, r, analysis = built("ground_improvement_package", {
        "method": "aggregate_piers", "column_diameter": 0.8, "spacing": 2.0})
    items = {i.name: i.value for i in get_input_summary(r, analysis)}
    assert "E_c" not in items and items["E_s"] == 5000.0
    by = {c["what"]: c for c in r.choices}
    assert by["column modulus E_c"]["chosen"] == "80000 kPa"
    assert "not used in the calculation" in by["column modulus E_c"]["basis"]
    assert math.isclose(r.stress_concentration_ratio, 5.0)
