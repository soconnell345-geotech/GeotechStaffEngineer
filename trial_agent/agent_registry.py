"""
Master registry of all 30 Foundry agents with dispatch functions.

Provides a unified interface for the trial CLI agent to call any
geotechnical calculation agent by name.
"""

import json

# ---------------------------------------------------------------------------
# Import all 30 agent modules
# ---------------------------------------------------------------------------

from bearing_capacity_agent_foundry import (
    bearing_capacity_agent, bearing_capacity_list_methods,
    bearing_capacity_describe_method,
)
from settlement_agent_foundry import (
    settlement_agent, settlement_list_methods,
    settlement_describe_method,
)
from axial_pile_agent_foundry import (
    axial_pile_agent, axial_pile_list_methods,
    axial_pile_describe_method,
)
from sheet_pile_agent_foundry import (
    sheet_pile_agent, sheet_pile_list_methods,
    sheet_pile_describe_method,
)
from lateral_pile_agent_foundry import (
    lateral_pile_agent, lateral_pile_list_methods,
    lateral_pile_describe_method,
)
from pile_group_agent_foundry import (
    pile_group_agent, pile_group_list_methods,
    pile_group_describe_method,
)
from wave_equation_agent_foundry import (
    wave_equation_agent, wave_equation_list_methods,
    wave_equation_describe_method,
)
from drilled_shaft_agent_foundry import (
    drilled_shaft_agent, drilled_shaft_list_methods,
    drilled_shaft_describe_method,
)
from seismic_geotech_agent_foundry import (
    seismic_geotech_agent, seismic_geotech_list_methods,
    seismic_geotech_describe_method,
)
from retaining_walls_agent_foundry import (
    retaining_walls_agent, retaining_walls_list_methods,
    retaining_walls_describe_method,
)
from ground_improvement_agent_foundry import (
    ground_improvement_agent, ground_improvement_list_methods,
    ground_improvement_describe_method,
)
from slope_stability_agent_foundry import (
    slope_stability_agent, slope_stability_list_methods,
    slope_stability_describe_method,
)
from downdrag_agent_foundry import (
    downdrag_agent, downdrag_list_methods,
    downdrag_describe_method,
)
from geolysis_agent_foundry import (
    geolysis_agent, geolysis_list_methods,
    geolysis_describe_method,
)
from groundhog_agent_foundry import (
    groundhog_agent, groundhog_list_methods,
    groundhog_describe_method,
)
from dm7_agent_foundry import (
    dm7_agent, dm7_list_methods,
    dm7_describe_method,
)
from calc_package_agent_foundry import (
    calc_package_agent, calc_package_list_methods,
    calc_package_describe_method,
)
from opensees_agent_foundry import (
    opensees_agent, opensees_list_methods,
    opensees_describe_method,
)
from pystrata_agent_foundry import (
    pystrata_agent, pystrata_list_methods,
    pystrata_describe_method,
)
from seismic_signals_agent_foundry import (
    seismic_signals_agent, seismic_signals_list_methods,
    seismic_signals_describe_method,
)
from liquepy_agent_foundry import (
    liquepy_agent, liquepy_list_methods,
    liquepy_describe_method,
)
from pygef_agent_foundry import (
    pygef_agent, pygef_list_methods,
    pygef_describe_method,
)
from hvsrpy_agent_foundry import (
    hvsrpy_agent, hvsrpy_list_methods,
    hvsrpy_describe_method,
)
from gstools_agent_foundry import (
    gstools_agent, gstools_list_methods,
    gstools_describe_method,
)
from ags4_agent_foundry import (
    ags4_agent, ags4_list_methods,
    ags4_describe_method,
)
from salib_agent_foundry import (
    salib_agent, salib_list_methods,
    salib_describe_method,
)
from pyseismosoil_agent_foundry import (
    pyseismosoil_agent, pyseismosoil_list_methods,
    pyseismosoil_describe_method,
)
from swprocess_agent_foundry import (
    swprocess_agent, swprocess_list_methods,
    swprocess_describe_method,
)
from pystra_agent_foundry import (
    pystra_agent, pystra_list_methods,
    pystra_describe_method,
)
from pydiggs_agent_foundry import (
    pydiggs_agent, pydiggs_list_methods,
    pydiggs_describe_method,
)


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY = {
    "bearing_capacity": {
        "agent": bearing_capacity_agent,
        "list": bearing_capacity_list_methods,
        "describe": bearing_capacity_describe_method,
        "brief": "Shallow foundation bearing capacity (Vesic/Meyerhof/Hansen)",
    },
    "settlement": {
        "agent": settlement_agent,
        "list": settlement_list_methods,
        "describe": settlement_describe_method,
        "brief": "Foundation settlement (elastic, Schmertmann, consolidation)",
    },
    "axial_pile": {
        "agent": axial_pile_agent,
        "list": axial_pile_list_methods,
        "describe": axial_pile_describe_method,
        "brief": "Driven pile axial capacity (Nordlund/Tomlinson/Beta)",
    },
    "sheet_pile": {
        "agent": sheet_pile_agent,
        "list": sheet_pile_list_methods,
        "describe": sheet_pile_describe_method,
        "brief": "Sheet pile walls (cantilever and anchored)",
    },
    "lateral_pile": {
        "agent": lateral_pile_agent,
        "list": lateral_pile_list_methods,
        "describe": lateral_pile_describe_method,
        "brief": "Lateral pile analysis (COM624P, 8 p-y models, FD solver)",
    },
    "pile_group": {
        "agent": pile_group_agent,
        "list": pile_group_list_methods,
        "describe": pile_group_describe_method,
        "brief": "Pile group analysis (rigid cap, 6-DOF, Converse-Labarre)",
    },
    "wave_equation": {
        "agent": wave_equation_agent,
        "list": wave_equation_list_methods,
        "describe": wave_equation_describe_method,
        "brief": "Smith 1-D wave equation (bearing graph, drivability)",
    },
    "drilled_shaft": {
        "agent": drilled_shaft_agent,
        "list": drilled_shaft_list_methods,
        "describe": drilled_shaft_describe_method,
        "brief": "Drilled shaft capacity (GEC-10 alpha/beta/rock socket, LRFD)",
    },
    "seismic_geotech": {
        "agent": seismic_geotech_agent,
        "list": seismic_geotech_list_methods,
        "describe": seismic_geotech_describe_method,
        "brief": "Seismic evaluation (site class, M-O pressure, liquefaction)",
    },
    "retaining_walls": {
        "agent": retaining_walls_agent,
        "list": retaining_walls_list_methods,
        "describe": retaining_walls_describe_method,
        "brief": "Retaining walls (cantilever + MSE, GEC-11)",
    },
    "ground_improvement": {
        "agent": ground_improvement_agent,
        "list": ground_improvement_list_methods,
        "describe": ground_improvement_describe_method,
        "brief": "Ground improvement (aggregate piers, wick drains, vibro)",
    },
    "slope_stability": {
        "agent": slope_stability_agent,
        "list": slope_stability_list_methods,
        "describe": slope_stability_describe_method,
        "brief": "Slope stability (Fellenius/Bishop/Spencer, grid search)",
    },
    "downdrag": {
        "agent": downdrag_agent,
        "list": downdrag_list_methods,
        "describe": downdrag_describe_method,
        "brief": "Pile downdrag (Fellenius neutral plane, UFC 3-220-20)",
    },
    "geolysis": {
        "agent": geolysis_agent,
        "list": geolysis_list_methods,
        "describe": geolysis_describe_method,
        "brief": "Soil classification (USCS/AASHTO) + SPT corrections + quick BC",
    },
    "groundhog": {
        "agent": groundhog_agent,
        "list": groundhog_list_methods,
        "describe": groundhog_describe_method,
        "brief": "90 correlations: phase relations, SPT/CPT, earth pressure, etc.",
    },
    "dm7": {
        "agent": dm7_agent,
        "list": dm7_list_methods,
        "describe": dm7_describe_method,
        "brief": "382 NAVFAC DM7 equations (UFC 3-220-10 and 3-220-20)",
    },
    "calc_package": {
        "agent": calc_package_agent,
        "list": calc_package_list_methods,
        "describe": calc_package_describe_method,
        "brief": "Generate HTML/PDF calculation packages",
    },
    "opensees": {
        "agent": opensees_agent,
        "list": opensees_list_methods,
        "describe": opensees_describe_method,
        "brief": "OpenSees FEM (PM4Sand DSS, BNWF pile, 1D site response)",
    },
    "pystrata": {
        "agent": pystrata_agent,
        "list": pystrata_list_methods,
        "describe": pystrata_describe_method,
        "brief": "1D EQL site response (SHAKE-type, Darendeli/Menq)",
    },
    "seismic_signals": {
        "agent": seismic_signals_agent,
        "list": seismic_signals_list_methods,
        "describe": seismic_signals_describe_method,
        "brief": "Earthquake signal processing (response spectra, RotD50/100)",
    },
    "liquepy": {
        "agent": liquepy_agent,
        "list": liquepy_list_methods,
        "describe": liquepy_describe_method,
        "brief": "CPT-based liquefaction triggering (Boulanger & Idriss 2014)",
    },
    "pygef": {
        "agent": pygef_agent,
        "list": pygef_list_methods,
        "describe": pygef_describe_method,
        "brief": "CPT/borehole file parser (GEF and BRO-XML formats)",
    },
    "hvsrpy": {
        "agent": hvsrpy_agent,
        "list": hvsrpy_list_methods,
        "describe": hvsrpy_describe_method,
        "brief": "HVSR site characterization from ambient noise",
    },
    "gstools": {
        "agent": gstools_agent,
        "list": gstools_list_methods,
        "describe": gstools_describe_method,
        "brief": "Geostatistics: kriging, variogram fitting, random fields",
    },
    "ags4": {
        "agent": ags4_agent,
        "list": ags4_list_methods,
        "describe": ags4_describe_method,
        "brief": "AGS4 geotechnical data format reader/validator",
    },
    "salib": {
        "agent": salib_agent,
        "list": salib_list_methods,
        "describe": salib_describe_method,
        "brief": "Sensitivity analysis (Sobol and Morris methods)",
    },
    "pyseismosoil": {
        "agent": pyseismosoil_agent,
        "list": pyseismosoil_list_methods,
        "describe": pyseismosoil_describe_method,
        "brief": "Nonlinear soil curve calibration (MKZ/HH) + Vs profiles",
    },
    "swprocess": {
        "agent": swprocess_agent,
        "list": swprocess_list_methods,
        "describe": swprocess_describe_method,
        "brief": "MASW surface wave dispersion analysis",
    },
    "pystra": {
        "agent": pystra_agent,
        "list": pystra_list_methods,
        "describe": pystra_describe_method,
        "brief": "Structural reliability (FORM/SORM/Monte Carlo)",
    },
    "pydiggs": {
        "agent": pydiggs_agent,
        "list": pydiggs_list_methods,
        "describe": pydiggs_describe_method,
        "brief": "DIGGS 2.6 XML schema and dictionary validation",
    },
}

AGENT_NAMES = sorted(AGENT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Result parsing (reuses pattern from foundry_test_harness/harness.py)
# ---------------------------------------------------------------------------

def _parse_result(raw):
    """Parse agent result — handles both JSON strings and direct dicts/lists."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

def call_agent(agent_name: str, method: str, parameters: dict) -> dict:
    """Execute a geotechnical calculation.

    Args:
        agent_name: One of the 30 registered agent names.
        method: Method name within that agent.
        parameters: Dict of parameters (will be JSON-serialized).

    Returns:
        Dict with calculation results or {"error": "..."}.
    """
    if agent_name not in AGENT_REGISTRY:
        return {"error": f"Unknown agent '{agent_name}'. Available: {AGENT_NAMES}"}
    agent_func = AGENT_REGISTRY[agent_name]["agent"]
    params_json = json.dumps(parameters)
    raw = agent_func(method, params_json)
    return _parse_result(raw)


def list_methods(agent_name: str, category: str = "") -> dict:
    """List available methods for a specific agent.

    Args:
        agent_name: One of the 30 registered agent names.
        category: Optional category filter.

    Returns:
        Dict of {category: {method: brief_description}}.
    """
    if agent_name not in AGENT_REGISTRY:
        return {"error": f"Unknown agent '{agent_name}'. Available: {AGENT_NAMES}"}
    list_func = AGENT_REGISTRY[agent_name]["list"]
    raw = list_func(category)
    return _parse_result(raw)


def describe_method(agent_name: str, method: str) -> dict:
    """Get full parameter documentation for a method.

    Args:
        agent_name: One of the 30 registered agent names.
        method: Method name to describe.

    Returns:
        Dict with parameters, returns, related, typical_workflow, common_mistakes.
    """
    if agent_name not in AGENT_REGISTRY:
        return {"error": f"Unknown agent '{agent_name}'. Available: {AGENT_NAMES}"}
    describe_func = AGENT_REGISTRY[agent_name]["describe"]
    raw = describe_func(method)
    return _parse_result(raw)


def list_agents() -> dict:
    """List all available agents with brief descriptions."""
    return {name: info["brief"] for name, info in AGENT_REGISTRY.items()}
