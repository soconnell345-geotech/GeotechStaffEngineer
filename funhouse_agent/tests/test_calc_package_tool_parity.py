"""GM-141: a calc package runs its analysis tool and prints what it used.

Each package re-implemented its tool and drifted from it (the wave
equation package documented pile_E and cushion_E in Pa where the tool takes
kPa; the pile group package dropped the layout; downdrag's fill weight was
20 where the tool's is 19). Now a package takes its inputs as its tool does
(the tool's METHOD_INFO), runs the tool's own code path and renders that
result. These tests hold it there:

* the result a package renders carries the same numbers as the tool's
  output, for two or more input sets per tool, one with the defaults left
  out (a number the tool prints rounded matches the package's rounded to
  as many places as the tool printed, or more);
* and the same numbers as the analysis module's own function called
  directly on the same inputs, with the tool's defaults written out, so
  the tool's code path is checked against the module and not only against
  itself; a number in the package's HTML is the result's;
* the defaults left in force are the tool's;
* every parameter of the tool is a parameter of the package, with the
  tool's type, units and description;
* a misspelt input is refused with the tool's did-you-mean, as the tool
  refuses it, and an input the package dropped is refused with a reason.
"""

import dataclasses
import math

import pytest

from funhouse_agent import dispatch as D
import funhouse_agent.adapters.calc_package as cp


_LP_LAYERS = [{"top": 0, "bottom": 15, "model": "SandAPI", "phi": 32,
               "gamma": 19, "k": 8150}]
_AX_LAYERS = [{"soil_type": "cohesionless", "thickness": 25.0,
               "unit_weight": 18.0, "friction_angle": 32.0}]
_DS_LAYERS = [{"soil_type": "cohesive", "thickness": 20.0,
               "unit_weight": 18.0, "cu": 75.0}]
_DD_LAYERS = [{"soil_type": "cohesive", "thickness": 8.0,
               "unit_weight": 17.0, "cu": 30.0, "settling": True,
               "Cc": 0.3, "e0": 1.0},
              {"soil_type": "cohesionless", "thickness": 15.0,
               "unit_weight": 19.0, "phi": 35.0, "settling": False}]
_SP_LAYERS = [{"thickness": 12.0, "unit_weight": 18.0,
               "friction_angle": 32.0}]
_SOE_LAYERS = [{"thickness": 12.0, "unit_weight": 18.0,
                "friction_angle": 32.0}]
_SOE_SUPPORTS = [{"depth": 2.0}, {"depth": 5.0, "support_type": "anchor"}]
_RV = {"R": {"mean": 200.0, "cov": 0.15, "dist": "lognormal"},
       "S": {"mean": 100.0, "cov": 0.25}}
_PV_LAYERS = [{"layer_type": "asphalt", "modulus_psi": 400000},
              {"layer_type": "granular_base", "modulus_psi": 30000},
              {"layer_type": "granular_subbase", "modulus_psi": 11000}]

#: (package, inputs): the first set of each tool leaves its defaults out.
CASES = [
    ("bearing_capacity_package", {"width": 2.0, "unit_weight": 18.0,
                                  "friction_angle": 30.0}),
    ("bearing_capacity_package", {"width": 2.0, "length": 3.0,
                                  "unit_weight": 18.0, "cohesion": 10.0,
                                  "friction_angle": 30.0, "depth": 1.5,
                                  "shape": "rectangular",
                                  "gwt_depth": 1.0}),
    ("lateral_pile_package", {"pile_length": 15.0, "pile_diameter": 0.8,
                              "layers": _LP_LAYERS, "Vt": 20.0}),
    ("lateral_pile_package", {"pile_length": 15.0, "pile_diameter": 0.8,
                              "pile_E": 7.5e6, "layers": _LP_LAYERS,
                              "Vt": 50.0, "Mt": 30.0}),
    ("settlement_package", {"q_applied": 150.0, "B": 2.0}),
    ("settlement_package", {"q_applied": 150.0, "B": 2.0, "L": 6.0,
                            "Es": 15000.0}),
    ("axial_pile_package", {"pile_type": "h_pile", "designation": "HP12x53",
                            "pile_length": 20.0, "layers": _AX_LAYERS}),
    ("axial_pile_package", {"pile_type": "pipe_open", "diameter": 0.4,
                            "wall_thickness": 0.012, "pile_length": 18.0,
                            "layers": _AX_LAYERS, "factor_of_safety": 3.0}),
    # GM-144: a custom section, which the package took before GM-141
    ("axial_pile_package", {"pile_type": "h_pile", "pile_area": 0.01006,
                            "pile_perimeter": 1.209, "pile_tip_area": 0.09135,
                            "pile_width": 0.3048, "pile_length": 20.0,
                            "layers": _AX_LAYERS}),
    ("drilled_shaft_package", {"diameter": 1.0, "shaft_length": 15.0,
                               "layers": _DS_LAYERS}),
    ("drilled_shaft_package", {"diameter": 1.2, "shaft_length": 18.0,
                               "layers": _DS_LAYERS, "gwt_depth": 3.0}),
    ("downdrag_package", {"pile_length": 20.0, "pile_diameter": 0.3,
                          "layers": _DD_LAYERS}),
    ("downdrag_package", {"pile_length": 20.0, "pile_diameter": 0.3,
                          "pile_E": 200e6, "Q_dead": 500.0,
                          "fill_thickness": 2.0, "layers": _DD_LAYERS}),
    ("seismic_package", {"vs30": 270.0, "Ss": 1.0, "S1": 0.4}),
    ("seismic_package", {"analysis_type": "site_classification",
                         "n_bar": 12.0, "Ss": 0.5, "S1": 0.2}),
    ("seismic_package", {"analysis_type": "seismic_earth_pressure",
                         "phi": 32, "kh": 0.2}),
    ("seismic_package", {"analysis_type": "seismic_earth_pressure",
                         "phi": 32, "kh": 0.2, "gamma": 19, "H": 5.0,
                         "kv": 0.05, "include_passive": True}),
    ("seismic_package", {"analysis_type": "liquefaction",
                         "depths": [2, 4, 6], "N160": [8, 12, 20],
                         "FC": [5, 10, 15], "gamma": [18, 19, 19],
                         "amax_g": 0.3, "gwt_depth": 1.5}),
    ("seismic_package", {"analysis_type": "liquefaction",
                         "depths": [2, 4, 6], "N160": [8, 12, 20],
                         "FC": [5, 10, 15], "gamma": [18, 19, 19],
                         "amax_g": 0.3, "gwt_depth": 1.5,
                         "magnitude": 6.5}),
    ("retaining_wall_package", {"wall_height": 4.0, "gamma_backfill": 18.0,
                                "phi_backfill": 30.0}),
    ("retaining_wall_package", {"wall_type": "cantilever",
                                "wall_height": 5.0, "gamma_backfill": 19.0,
                                "phi_backfill": 32.0, "surcharge": 10.0}),
    ("ground_improvement_package", {"spacing": 1.5, "ch": 3.0, "cv": 1.0,
                                    "Hdr": 5.0, "time": 1.0}),
    ("ground_improvement_package", {"method": "wick_drains", "spacing": 1.2,
                                    "ch": 2.0, "cv": 0.8, "Hdr": 6.0,
                                    "time": 0.5, "pattern": "square"}),
    ("ground_improvement_package", {"method": "aggregate_piers",
                                    "column_diameter": 0.76, "spacing": 2.0}),
    ("ground_improvement_package", {"method": "aggregate_piers",
                                    "column_diameter": 0.76, "spacing": 2.0,
                                    "pattern": "square", "E_column": 80000,
                                    "E_soil": 4000}),
    ("ground_improvement_package", {"method": "surcharge",
                                    "S_ultimate": 0.3, "surcharge_kPa": 50,
                                    "cv": 3.0, "Hdr": 4.0}),
    ("ground_improvement_package", {"method": "surcharge",
                                    "S_ultimate": 0.3, "surcharge_kPa": 50,
                                    "cv": 3.0, "Hdr": 4.0, "target_U": 80}),
    ("ground_improvement_package", {"method": "vibro", "fines_content": 8,
                                    "initial_N_spt": 12}),
    ("ground_improvement_package", {"method": "vibro", "fines_content": 8,
                                    "initial_N_spt": 12,
                                    "target_N_spt": 25}),
    ("wave_equation_package", {"hammer_name": "Delmag D30-32",
                               "pile_length": 15.0, "pile_area": 0.01}),
    ("wave_equation_package", {"hammer_name": "Delmag D30-32",
                               "pile_length": 15.0, "pile_area": 0.01,
                               "pile_E": 200e6, "cushion_area": 0.1,
                               "cushion_thickness": 0.1,
                               "cushion_E": 3.5e6}),
    ("pile_group_package", {"n_rows": 3, "n_cols": 3, "spacing": 1.5,
                            "Vz": 1800.0}),
    ("pile_group_package", {"n_rows": 2, "n_cols": 3, "spacing_x": 1.2,
                            "spacing_y": 1.5, "Vz": 1800.0, "Mx": 100.0}),
    ("pile_group_package", {"method_6dof": True, "n_rows": 2, "n_cols": 2,
                            "spacing": 1.5, "Vz": 1000.0,
                            "axial_stiffness": 2e5}),
    ("pile_group_package", {"method_6dof": True, "n_rows": 2, "n_cols": 2,
                            "spacing": 1.5, "Vz": 1000.0, "Vx": 50.0,
                            "axial_stiffness": 2e5,
                            "lateral_stiffness": 1e4}),
    ("sheet_pile_package", {"excavation_depth": 3.0, "layers": _SP_LAYERS}),
    ("sheet_pile_package", {"wall_type": "cantilever",
                            "excavation_depth": 3.0, "layers": _SP_LAYERS,
                            "surcharge": 10.0}),
    ("sheet_pile_package", {"wall_type": "anchored", "excavation_depth": 5.0,
                            "anchor_depth": 1.0, "layers": _SP_LAYERS}),
    ("sheet_pile_package", {"wall_type": "anchored", "excavation_depth": 5.0,
                            "anchor_depth": 1.5, "layers": _SP_LAYERS,
                            "FOS_passive": 1.5}),
    ("soe_package", {"excavation_depth": 3.0, "layers": _SOE_LAYERS}),
    ("soe_package", {"wall_type": "cantilever", "excavation_depth": 3.5,
                     "layers": _SOE_LAYERS, "surcharge": 0.0,
                     "gwt_depth": 2.0, "FOS_passive": 1.3,
                     "embedment_increase": 1.2, "Fy": 250.0}),
    ("soe_package", {"wall_type": "braced", "excavation_depth": 8.0,
                     "layers": _SOE_LAYERS, "supports": _SOE_SUPPORTS}),
    ("soe_package", {"wall_type": "braced", "excavation_depth": 8.0,
                     "layers": _SOE_LAYERS, "supports": _SOE_SUPPORTS,
                     "gwt_depth": 6.0, "surcharge": 12.0,
                     "FOS_passive": 1.3}),
    ("soe_package", {"wall_type": "fhwa_anchored", "H": 8.0,
                     "anchor_depths": [2.0, 5.0], "gamma": 18.0,
                     "phi": 32.0}),
    ("soe_package", {"wall_type": "fhwa_anchored", "H": 6.0,
                     "anchor_depths": [1.5], "gamma": 18.0, "phi": 32.0,
                     "surcharge": 10.0, "spacing": 2.5,
                     "inclination_deg": 15.0}),
    ("reliability_package", {"variables": _RV, "g_expression": "R/S"}),
    ("reliability_package", {"engine": "fosm", "variables": _RV,
                             "g_expression": "R - S",
                             "convention": "margin"}),
    ("reliability_package", {"engine": "pem", "variables": _RV,
                             "g_expression": "R/S"}),
    ("reliability_package", {"engine": "pem", "variables": _RV,
                             "g_expression": "R/S",
                             "scheme": "multiplicative"}),
    ("reliability_package", {"engine": "monte_carlo", "variables": _RV,
                             "g_expression": "R/S", "seed": 7}),
    ("reliability_package", {"engine": "monte_carlo", "variables": _RV,
                             "g_expression": "R/S", "seed": 7,
                             "n": 2000, "sampling": "lhs"}),
    ("reliability_package", {"engine": "form", "variables": _RV,
                             "g_expression": "R/S"}),
    ("reliability_package", {"engine": "form", "variables": _RV,
                             "g_expression": "R - S",
                             "convention": "margin",
                             "max_iterations": 50}),
    ("pavement_design_package", {"w18": 5e6, "reliability_pct": 90,
                                 "mr_psi": 7500, "layers": _PV_LAYERS}),
    ("pavement_design_package", {"design_type": "flexible", "w18": 5e6,
                                 "reliability_pct": 95, "mr_psi": 7500,
                                 "so": 0.45, "layers": _PV_LAYERS}),
    ("pavement_design_package", {"design_type": "rigid", "w18": 5e6,
                                 "reliability_pct": 90, "sc_psi": 650,
                                 "ec_psi": 4e6, "k_pci": 150}),
    ("pavement_design_package", {"design_type": "flexible_ufc",
                                 "passes_18kip": 1e5, "cbr_subgrade": 6}),
    ("pavement_design_package", {"design_type": "rigid_ufc",
                                 "passes_18kip": 1e5,
                                 "flexural_strength_psi": 650,
                                 "k_pci": 150}),
    ("pavement_design_package", {"design_type": "compare",
                                 "passes_18kip": 1e5, "cbr_subgrade": 6}),
]


def _agent(mod):
    """The dispatch name of a tool adapter."""
    return next(name for name, entry in D.MODULE_REGISTRY.items()
                if entry["adapter"] == mod.__name__)


def _tool(package, params):
    """(tool adapter, tool name, the tool's inputs) a package runs."""
    spec = cp._PACKAGE_TOOLS[package]
    tool_params = dict(params)
    value = None
    if "selector" in spec:
        name, default, _ = spec["selector"]
        value = tool_params.pop(name, default)
    return cp._tool_adapter(package), spec["tools"][value], tool_params


def _rendered(package, params, monkeypatch):
    """The result and inputs a package hands to the renderer."""
    seen = {}

    def capture(module, result, analysis, params, analysis_type,
                extra=None, run=None):
        seen.update(result=result, analysis=analysis)
        return {"status": "success"}
    monkeypatch.setattr(cp, "_build_response", capture)
    cp.METHOD_REGISTRY[package](dict(params))
    return seen


def _numbers(obj, prefix="", out=None):
    """Every number in ``obj`` (a result, a dict or a list), by its path."""
    out = {} if out is None else out
    if hasattr(obj, "to_dict"):
        try:
            obj = obj.to_dict()
        except TypeError:          # a site class result with no Ss / S1
            obj = dataclasses.asdict(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            _numbers(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _numbers(v, f"{prefix}[{i}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _decimals(x):
    """The decimal places ``x`` is written with (0.78 -> 2, 226.0 -> 0)."""
    text = repr(float(x))
    if "e" in text:
        mantissa, exp = text.split("e")
        places = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return max(0, places - int(exp))
    return len(text.split(".")[1].rstrip("0"))


def _same(printed, value):
    """The tool printed ``value``, perhaps rounded: to as many places as
    it printed or more, and never to a whole number (226 is not 225.6)."""
    if printed == value or (math.isnan(printed) and math.isnan(value)):
        return True
    return any(round(value, n) == printed
               for n in range(max(1, _decimals(printed)), 7))


@pytest.mark.parametrize("package, params", CASES,
                         ids=[f"{p}-{i}" for i, (p, _) in enumerate(CASES)])
def test_the_package_renders_the_tools_numbers(package, params, monkeypatch):
    mod, tool, tool_params = _tool(package, params)
    tool_out = mod.METHOD_REGISTRY[tool](dict(tool_params))
    assert "error" not in tool_out, tool_out
    printed = _numbers(tool_out)
    rendered = _numbers(_rendered(package, params, monkeypatch)["result"])
    shared = set(printed) & set(rendered)
    assert len(shared) >= min(3, len(printed)), (sorted(printed),
                                                 sorted(rendered))
    differ = {k: (printed[k], rendered[k]) for k in shared
              if not _same(printed[k], rendered[k])}
    assert not differ


@pytest.mark.parametrize("package, params", CASES,
                         ids=[f"{p}-{i}" for i, (p, _) in enumerate(CASES)])
def test_the_package_leaves_the_tools_defaults(package, params):
    mod, tool, tool_params = _tool(package, params)
    tool_out = D.call_agent(_agent(mod), tool,
                            dict(tool_params))
    expected = dict(tool_out.get("defaults_used", {}))
    selector = cp._PACKAGE_TOOLS[package].get("selector")
    if selector and selector[0] not in params:
        expected[selector[0]] = selector[1]
    assert cp._run_tool(package, dict(params)).defaults_used == expected


def test_the_package_response_says_the_tools_defaults(tmp_path):
    # rigid pavement's so and po (0.35, 4.5) are not flexible's (0.45,
    # 4.2): the package lists the defaults of the tool it ran
    params = {"design_type": "rigid", "w18": 5e6, "reliability_pct": 90,
              "sc_psi": 650, "ec_psi": 4e6, "k_pci": 150}
    tool = D.call_agent("pavement_design", "rigid_pavement_design",
                        {k: v for k, v in params.items()
                         if k != "design_type"})
    out = D.call_agent("calc_package", "pavement_design_package", {
        **params, "output_path": str(tmp_path / "pv.html")})
    assert out["status"] == "success", out
    assert out["defaults_used"] == tool["defaults_used"]
    assert out["defaults_used"]["so"] == 0.35


@pytest.mark.parametrize("package", sorted(cp._PACKAGE_TOOLS))
def test_the_package_documents_its_tools_parameters(package):
    spec = cp._PACKAGE_TOOLS[package]
    mod = cp._tool_adapter(package)
    params = cp.METHOD_INFO[package]["parameters"]
    for tool in spec["tools"].values():
        for name, tspec in mod.METHOD_INFO[tool]["parameters"].items():
            assert name in params, (tool, name)
            assert set(tspec["type"].split("|")) <= set(
                params[name]["type"].split("|")), (tool, name)
            assert tspec.get("description", "") in \
                params[name].get("description", ""), (tool, name)
            if len(spec["tools"]) == 1:
                assert params[name] == tspec, (tool, name)


def test_wave_equation_units_are_the_tools():
    # the package documented pile_E and cushion_E in Pa (GM-141)
    params = cp.METHOD_INFO["wave_equation_package"]["parameters"]
    for name in ("pile_E", "cushion_E"):
        assert "kPa" in params[name]["description"]
        assert "Pa)" not in params[name]["description"].replace("kPa)", "")


@pytest.mark.parametrize("package, params, wrong, right", [
    ("settlement_package", {"q_applied": 150.0, "B": 2.0}, "Es_kpa", "Es"),
    ("wave_equation_package", {"pile_length": 15.0, "pile_area": 0.01},
     "pile_e", "pile_E"),
    ("downdrag_package", {"pile_length": 20.0, "pile_diameter": 0.3,
                          "layers": _DD_LAYERS},
     "fill_unit_wieght", "fill_unit_weight"),
    ("pile_group_package", {"n_rows": 2, "n_cols": 2, "spacing": 1.5,
                            "Vz": 100.0}, "n_row", "n_rows"),
])
def test_a_misspelt_input_gets_the_tools_did_you_mean(package, params,
                                                      wrong, right):
    mod, tool, tool_params = _tool(package, params)
    out = D.call_agent(_agent(mod), tool,
                       {**tool_params, wrong: 1.0})
    assert f"{wrong} -> {right}" in out["error"]
    out = D.call_agent("calc_package", package, {**params, wrong: 1.0})
    assert f"{wrong} -> {right}" in out["error"]


def test_the_axial_package_takes_a_custom_section_as_its_tool_does():
    """GM-144: the custom section the package took before GM-141 (area,
    perimeter, tip area, width) is the tool's again (the parity cases above
    check its numbers); an H-pile's toe area is still asked for."""
    out = D.call_agent("calc_package", "axial_pile_package", {
        "pile_type": "h_pile", "pile_length": 20.0, "layers": _AX_LAYERS,
        "pile_area": 0.01})
    assert "pile_perimeter" in out["error"] and "pile_tip_area" in out["error"]


def test_an_input_the_package_dropped_is_refused_with_its_reason():
    out = D.call_agent("calc_package", "retaining_wall_package", {
        "wall_type": "mse", "wall_height": 5.0, "gamma_backfill": 18.0,
        "phi_backfill": 30.0})
    assert "run the mse_wall tool" in out["error"]


# ---------------------------------------------------------------------------
# Against the analysis module itself. Both sides of the tests above run the
# tool's run_<tool>, so they agree whatever it works out. Here each
# module's objects are built from the same inputs, with the tool's defaults
# written out where the module's own differ, and its function is called
# directly; the package must render those numbers.
# ---------------------------------------------------------------------------

def _bearing():
    from bearing_capacity.capacity import BearingCapacityAnalysis
    from bearing_capacity.footing import Footing
    from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
    return BearingCapacityAnalysis(
        footing=Footing(width=2.0),
        soil=BearingSoilProfile(layer1=SoilLayer(
            friction_angle=30.0, unit_weight=18.0))).compute()


def _lateral():
    from lateral_pile import LateralPileAnalysis, Pile, SoilLayer
    from lateral_pile.py_curves import SandAPI
    pile = Pile(length=15.0, diameter=0.8, E=200e6)   # the tool's pile_E
    layers = [SoilLayer(top=0, bottom=15,
                        py_model=SandAPI(phi=32, gamma=19, k=8150))]
    return LateralPileAnalysis(pile, layers).solve(Vt=20.0)


def _settlement():
    from settlement.analysis import SettlementAnalysis
    return SettlementAnalysis(q_applied=150.0, B=2.0, L=6.0,
                              Es_immediate=15000.0).compute()


def _axial():
    from axial_pile import (AxialPileAnalysis, AxialSoilLayer,
                            AxialSoilProfile, make_h_pile)
    soil = AxialSoilProfile(layers=[AxialSoilLayer(
        thickness=25.0, soil_type="cohesionless", unit_weight=18.0,
        friction_angle=32.0)])     # delta/phi from GEC-12 Fig. 7-9 (GM-121)
    return AxialPileAnalysis(pile=make_h_pile("HP12x53"), soil=soil,
                             pile_length=20.0).compute()


def _drilled():
    from drilled_shaft import (DrillShaft, DrillShaftAnalysis,
                               ShaftSoilLayer, ShaftSoilProfile)
    return DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.0, length=15.0),
        soil=ShaftSoilProfile(layers=[ShaftSoilLayer(
            thickness=20.0, soil_type="cohesive", unit_weight=18.0,
            cu=75.0)])).compute()


def _downdrag():
    from downdrag import (DowndragAnalysis, DowndragSoilLayer,
                          DowndragSoilProfile)
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=8.0, soil_type="cohesive",
                          unit_weight=17.0, cu=30.0, settling=True,
                          Cc=0.3, e0=1.0),
        DowndragSoilLayer(thickness=15.0, soil_type="cohesionless",
                          unit_weight=19.0, phi=35.0)])
    return DowndragAnalysis(soil=soil, pile_length=20.0,
                            pile_diameter=0.3).compute()


def _site_class():
    from seismic_geotech import classify_site, site_coefficients
    return site_coefficients(classify_site(vs30=270.0), Ss=1.0, S1=0.4)


def _mononobe_okabe():
    from seismic_geotech import (mononobe_okabe_KAE, mononobe_okabe_KPE,
                                 seismic_earth_pressure)
    from seismic_geotech.results import SeismicEarthPressureResult
    phi, kh, kv = 32.0, 0.2, 0.05
    delta = 2.0 / 3.0 * phi
    KAE = mononobe_okabe_KAE(phi, delta, kh, kv)
    # KA not given: Rankine Ka from phi, as the tool works it out (GM-120)
    KA = math.tan(math.radians(45.0 - phi / 2.0)) ** 2
    # PAE carries (1 - kv), UFC 3-220-20 Eq. 4-16 (GM-149)
    f = seismic_earth_pressure(19.0, 5.0, KAE, KA, kv=kv)
    return SeismicEarthPressureResult(
        KAE=KAE, KPE=mononobe_okabe_KPE(phi, delta, kh, kv),
        PAE_total=f["PAE_total_kN_per_m"],
        PA_static=f["PA_static_kN_per_m"],
        delta_PAE=f["delta_PAE_kN_per_m"],
        height_of_application=f["height_of_application_m"],
        phi=phi, delta=delta, kh=kh, kv=kv)


def _liquefaction():
    from seismic_geotech import evaluate_liquefaction
    from seismic_geotech.results import LiquefactionResult
    return LiquefactionResult(
        layer_results=evaluate_liquefaction(
            [2, 4, 6], [8, 12, 20], [5, 10, 15], [18, 19, 19],
            amax_g=0.3, gwt_depth=1.5),
        amax_g=0.3, magnitude=7.5, gwt_depth=1.5)


def _cantilever_wall():
    from retaining_walls import CantileverWallGeometry, analyze_cantilever_wall
    return analyze_cantilever_wall(CantileverWallGeometry(wall_height=4.0),
                                   gamma_backfill=18.0, phi_backfill=30.0)


def _ground(analyze, **inputs):
    import ground_improvement
    return lambda: getattr(ground_improvement, analyze)(**inputs)


def _bearing_graph():
    from wave_equation import (Cushion, discretize_pile,
                               generate_bearing_graph, get_hammer)
    return generate_bearing_graph(
        get_hammer("Delmag D30-32"),
        Cushion(stiffness=500000, cor=0.80),       # the tool's cushion
        discretize_pile(15.0, 0.01, 200e6))        # the tool's pile_E


def _group(analyze, n, stiffness=None, **load):
    import pile_group

    def run():
        piles = pile_group.create_rectangular_layout(
            n, n, 1.5, 1.5, axial_stiffness=stiffness)
        return getattr(pile_group, analyze)(piles,
                                            pile_group.GroupLoad(**load))
    return run


def _sheet_pile(anchor_depth=None):
    from sheet_pile.anchored import analyze_anchored
    from sheet_pile.cantilever import WallSoilLayer, analyze_cantilever
    layers = [WallSoilLayer(thickness=12.0, unit_weight=18.0,
                            friction_angle=32.0)]
    if anchor_depth is None:
        return analyze_cantilever(3.0, layers)
    return analyze_anchored(5.0, anchor_depth, layers)


def _soe(wall):
    """The SOE module called directly, the tool's defaults written out
    (surcharge 10 kPa, FOS 1.5, Fy 345; D/D0 1.2 cantilever, 1.0 braced)."""
    def run():
        import soe
        layers = [soe.SOEWallLayer(thickness=12.0, unit_weight=18.0,
                                   friction_angle=32.0)]
        if wall == "fhwa_anchored":
            return soe.fhwa_apparent_pressure_anchored_wall(
                H=8.0, anchor_depths=[2.0, 5.0], gamma=18.0, phi=32.0)
        if wall == "cantilever":
            g = soe.ExcavationGeometry(excavation_depth=3.0,
                                       soil_layers=layers, surcharge=10.0)
            return soe.analyze_cantilever_excavation(
                g, FOS_passive=1.5, Fy=345.0, embedment_increase=1.2)
        g = soe.ExcavationGeometry(
            excavation_depth=8.0, soil_layers=layers, surcharge=10.0,
            support_levels=[soe.SupportLevel(depth=2.0),
                            soe.SupportLevel(depth=5.0,
                                             support_type="anchor")])
        return soe.analyze_braced_excavation(g, Fy=345.0, FOS_passive=1.5,
                                             embedment_increase=1.0)
    return run


def _reliability(engine, **options):
    """The reliability engine called directly on the same variables."""
    def run():
        import reliability
        g = (lambda x: x["R"] / x["S"])
        return getattr(reliability, engine)(
            g, {k: dict(v) for k, v in _RV.items()}, **options)
    return run


def _pavement(design, **inputs):
    import pavement_design
    return lambda: getattr(pavement_design, design)(**inputs)


#: (package, inputs, the module's own result for those inputs)
MODULE_CASES = [
    ("bearing_capacity_package", {"width": 2.0, "unit_weight": 18.0,
                                  "friction_angle": 30.0}, _bearing),
    ("lateral_pile_package", {"pile_length": 15.0, "pile_diameter": 0.8,
                              "layers": _LP_LAYERS, "Vt": 20.0}, _lateral),
    ("settlement_package", {"q_applied": 150.0, "B": 2.0, "L": 6.0,
                            "Es": 15000.0}, _settlement),
    ("axial_pile_package", {"pile_type": "h_pile", "designation": "HP12x53",
                            "pile_length": 20.0, "layers": _AX_LAYERS},
     _axial),
    ("drilled_shaft_package", {"diameter": 1.0, "shaft_length": 15.0,
                               "layers": _DS_LAYERS}, _drilled),
    ("downdrag_package", {"pile_length": 20.0, "pile_diameter": 0.3,
                          "layers": _DD_LAYERS}, _downdrag),
    ("seismic_package", {"vs30": 270.0, "Ss": 1.0, "S1": 0.4}, _site_class),
    ("seismic_package", {"analysis_type": "seismic_earth_pressure",
                         "phi": 32, "kh": 0.2, "gamma": 19, "H": 5.0,
                         "kv": 0.05, "include_passive": True},
     _mononobe_okabe),
    ("seismic_package", {"analysis_type": "liquefaction",
                         "depths": [2, 4, 6], "N160": [8, 12, 20],
                         "FC": [5, 10, 15], "gamma": [18, 19, 19],
                         "amax_g": 0.3, "gwt_depth": 1.5}, _liquefaction),
    ("retaining_wall_package", {"wall_height": 4.0, "gamma_backfill": 18.0,
                                "phi_backfill": 30.0}, _cantilever_wall),
    ("ground_improvement_package", {"spacing": 1.5, "ch": 3.0, "cv": 1.0,
                                    "Hdr": 5.0, "time": 1.0},
     _ground("analyze_wick_drains", spacing=1.5, ch=3.0, cv=1.0, Hdr=5.0,
             time=1.0)),
    ("ground_improvement_package", {"method": "aggregate_piers",
                                    "column_diameter": 0.76, "spacing": 2.0},
     _ground("analyze_aggregate_piers", column_diameter=0.76, spacing=2.0)),
    ("ground_improvement_package", {"method": "surcharge",
                                    "S_ultimate": 0.3, "surcharge_kPa": 50,
                                    "cv": 3.0, "Hdr": 4.0},
     _ground("analyze_surcharge_preloading", S_ultimate=0.3,
             surcharge_kPa=50, cv=3.0, Hdr=4.0)),
    ("ground_improvement_package", {"method": "vibro", "fines_content": 8,
                                    "initial_N_spt": 12},
     _ground("analyze_vibro_compaction", fines_content=8, initial_N_spt=12)),
    ("wave_equation_package", {"hammer_name": "Delmag D30-32",
                               "pile_length": 15.0, "pile_area": 0.01},
     _bearing_graph),
    ("pile_group_package", {"n_rows": 3, "n_cols": 3, "spacing": 1.5,
                            "Vz": 1800.0},
     _group("analyze_vertical_group_simple", 3, Vz=1800.0)),
    ("pile_group_package", {"method_6dof": True, "n_rows": 2, "n_cols": 2,
                            "spacing": 1.5, "Vz": 1000.0,
                            "axial_stiffness": 2e5},
     _group("analyze_group_6dof", 2, stiffness=2e5, Vz=1000.0)),
    ("sheet_pile_package", {"excavation_depth": 3.0, "layers": _SP_LAYERS},
     _sheet_pile),
    ("sheet_pile_package", {"wall_type": "anchored", "excavation_depth": 5.0,
                            "anchor_depth": 1.0, "layers": _SP_LAYERS},
     lambda: _sheet_pile(anchor_depth=1.0)),
    ("soe_package", {"excavation_depth": 3.0, "layers": _SOE_LAYERS},
     _soe("cantilever")),
    ("soe_package", {"wall_type": "braced", "excavation_depth": 8.0,
                     "layers": _SOE_LAYERS, "supports": _SOE_SUPPORTS},
     _soe("braced")),
    ("soe_package", {"wall_type": "fhwa_anchored", "H": 8.0,
                     "anchor_depths": [2.0, 5.0], "gamma": 18.0,
                     "phi": 32.0}, _soe("fhwa_anchored")),
    ("reliability_package", {"variables": _RV, "g_expression": "R/S"},
     _reliability("fosm")),
    ("reliability_package", {"engine": "form", "variables": _RV,
                             "g_expression": "R/S"}, _reliability("form")),
    ("reliability_package", {"engine": "monte_carlo", "variables": _RV,
                             "g_expression": "R/S", "seed": 7},
     _reliability("monte_carlo", seed=7)),
    ("pavement_design_package", {"w18": 5e6, "reliability_pct": 90,
                                 "mr_psi": 7500, "layers": _PV_LAYERS},
     _pavement("design_flexible_pavement", w18=5e6, reliability_pct=90,
               mr_psi=7500, layers=[dict(l) for l in _PV_LAYERS])),
    ("pavement_design_package", {"design_type": "flexible_ufc",
                                 "passes_18kip": 1e5, "cbr_subgrade": 6},
     _pavement("design_flexible_pavement_ufc", passes_18kip=1e5,
               cbr_subgrade=6)),
]


def test_every_package_is_checked_against_its_module():
    assert {p for p, _, _ in MODULE_CASES} == set(cp._PACKAGE_TOOLS)


@pytest.mark.parametrize(
    "package, params, module_result", MODULE_CASES,
    ids=[f"{p}-{i}" for i, (p, _, _) in enumerate(MODULE_CASES)])
def test_the_package_renders_the_modules_numbers(package, params,
                                                 module_result, monkeypatch):
    expected = _numbers(module_result())
    rendered = _numbers(_rendered(package, params, monkeypatch)["result"])
    shared = set(expected) & set(rendered)
    assert len(shared) >= min(3, len(expected)), (sorted(expected),
                                                  sorted(rendered))
    differ = {k: (expected[k], rendered[k]) for k in shared
              if not (expected[k] == rendered[k]
                      or (math.isnan(expected[k])
                          and math.isnan(rendered[k])))}
    assert not differ


def test_the_html_prints_the_results_numbers(tmp_path):
    # the package's own file, not the result handed to it
    q_ult = _bearing().q_ultimate
    out = D.call_agent("calc_package", "bearing_capacity_package", {
        "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
        "output_path": str(tmp_path / "bc.html")})
    assert out["status"] == "success", out
    html = (tmp_path / "bc.html").read_text(encoding="utf-8")
    assert f"{q_ult:,.1f}" in html


def test_group_efficiency_takes_the_smaller_spacing(tmp_path):
    from pile_group import converse_labarre

    def e_g(s):     # the group_efficiency tool prints E_g to four places
        return round(converse_labarre(2, 3, 0.4, s), 4)
    out = D.call_agent("calc_package", "pile_group_package", {
        "n_rows": 2, "n_cols": 3, "spacing_x": 1.2, "spacing_y": 1.5,
        "Vz": 1800.0, "pile_diameter": 0.4,
        "output_path": str(tmp_path / "pg.html")})
    assert out["status"] == "success", out
    assert out["group_efficiency"] == e_g(1.2)
    assert out["group_efficiency_spacing_m"] == 1.2
    assert "the smaller, 1.2 m, is used" in out["group_efficiency_note"]
    html = (tmp_path / "pg.html").read_text(encoding="utf-8")
    assert "the smaller is used" in html
    # one spacing both ways: nothing to say
    out = D.call_agent("calc_package", "pile_group_package", {
        "n_rows": 2, "n_cols": 3, "spacing": 1.5, "Vz": 1800.0,
        "pile_diameter": 0.4, "output_path": str(tmp_path / "pg2.html")})
    assert out["group_efficiency"] == e_g(1.5)
    assert "group_efficiency_note" not in out


def test_package_messages_use_the_callers_names():
    # an item key given by its older name, and the selector value that
    # runs the package's other tool
    layer = {k: v for k, v in _LP_LAYERS[0].items() if k != "model"}
    out = D.call_agent("calc_package", "lateral_pile_package", {
        "pile_length": 15.0, "pile_diameter": 0.8, "Vt": 20.0,
        "layers": [{**layer, "py_model": "SandAPI", "phii": 3}]})
    assert "'py_model'" in out["error"]
    assert "'model'" not in out["error"]
    out = D.call_agent("calc_package", "pile_group_package", {
        "method_6dof": True, "n_rows": 2, "n_cols": 2, "spacing": 1.5,
        "Vz": 100.0})
    assert ("method_6dof=False (pile_group_simple) needs no stiffness"
            in out["error"])


def test_soe_package_prints_the_results_numbers(tmp_path):
    # GM-41: the SOE package's file carries the result's numbers
    out = D.call_agent("calc_package", "soe_package", {
        "wall_type": "braced", "excavation_depth": 8.0,
        "layers": _SOE_LAYERS, "supports": _SOE_SUPPORTS,
        "output_path": str(tmp_path / "soe.html")})
    assert out["status"] == "success", out
    tool = D.call_agent("soe", "braced_excavation", {
        "excavation_depth": 8.0, "layers": _SOE_LAYERS,
        "supports": _SOE_SUPPORTS})
    html = (tmp_path / "soe.html").read_text(encoding="utf-8")
    for r in tool["support_reactions"]:
        assert f"{r['load_kN_per_m']}" in html
    assert f"{tool['required_embedment_m']}" in html
    assert f"{tool['max_moment_kNm_per_m']}" in html
    assert out["support_loads_kN_per_m"] == [
        r["load_kN_per_m"] for r in tool["support_reactions"]]
    out = D.call_agent("calc_package", "soe_package", {
        "wall_type": "fhwa_anchored", "H": 8.0, "anchor_depths": [2.0, 5.0],
        "gamma": 18.0, "phi": 32.0,
        "output_path": str(tmp_path / "fhwa.html")})
    html = (tmp_path / "fhwa.html").read_text(encoding="utf-8")
    assert "36.328" in html and "102.93" in html   # pe and DL1
    out = D.call_agent("calc_package", "soe_package", {
        "wall_type": "anchored", "H": 8.0})
    assert "wall_type must be one of" in out["error"]


def test_reliability_package_prints_the_engines_numbers(tmp_path):
    # GM-41: beta and Pf in the file are the tool's
    tool = D.call_agent("reliability", "fosm",
                        {"variables": _RV, "g_expression": "R/S"})
    out = D.call_agent("calc_package", "reliability_package", {
        "variables": _RV, "g_expression": "R/S",
        "output_path": str(tmp_path / "rel.html")})
    assert out["status"] == "success", out
    html = (tmp_path / "rel.html").read_text(encoding="utf-8")
    assert f"{tool['beta_normal']:.3f}" in html
    assert f"{tool['beta_lognormal']:.3f}" in html
    assert out["beta_lognormal"] == tool["beta_lognormal"]
    assert "lognormal" in html and "R/S" in html
    # the owner's wording for a probability (never "exact")
    assert "what the assumed model implies" in html


def test_reliability_package_correlation_matrix_rows():
    # a 3 x 3 matrix is three rows by variable, not three (a, b, rho) pairs
    from reliability.calc_steps import _correlation_table
    t = _correlation_table({
        "variables": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        "correlation": [[1.0, 0.3, 0.0], [0.3, 1.0, 0.5],
                        [0.0, 0.5, 1.0]]})
    assert [r[0] for r in t.rows] == ["a", "b", "c"]
    assert t.rows[1][1] == "0.300, 1.000, 0.500"


def test_soe_braced_diagrams_share_one_sign():
    # GM-41 review: the spans above the excavation and the free earth body
    # below the lowest support are drawn on one set of axes, so both use
    # the free earth body's sign (load toward the excavation positive):
    # the shear at the bottom of the cantilever span is +(its load)
    import soe
    layers = [soe.SOEWallLayer(thickness=12.0, unit_weight=18.0,
                               friction_angle=32.0)]
    g = soe.ExcavationGeometry(
        excavation_depth=8.0, soil_layers=layers, surcharge=10.0,
        support_levels=[soe.SupportLevel(depth=2.0),
                        soe.SupportLevel(depth=5.0, support_type="anchor")])
    r = soe.analyze_braced_excavation(g)
    d = r.diagram
    v_end, m_end = max((v, m) for z, v, m in zip(d["z"], d["V"], d["M"])
                       if z <= 2.0 + 1e-6)
    assert v_end == pytest.approx(r.spans[0]["load_kN_per_m"], rel=0.02)
    assert m_end == pytest.approx(r.spans[0]["load_kN_per_m"]
                                  * r.spans[0]["centroid_below_top_m"],
                                  rel=0.02)
    assert r.free_earth_diagram["V"][-1] == pytest.approx(0.0, abs=1.0)
    assert [l["Kp"] for l in r.layers_used] == [round(
        math.tan(math.radians(61.0)) ** 2, 4)]
