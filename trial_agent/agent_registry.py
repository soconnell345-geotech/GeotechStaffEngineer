"""
Master registry of all 44 Foundry agents with lazy-loading dispatch.

Agents are imported on first use, not at startup, to avoid scipy.optimize
import hangs on Python 3.14 / scipy 1.17 and to speed up CLI startup.
"""

import json
import importlib

# ---------------------------------------------------------------------------
# Agent registry — module/function names only, no imports at load time
# ---------------------------------------------------------------------------

_AGENT_SPECS = {
    "bearing_capacity": {
        "module": "bearing_capacity_agent_foundry",
        "funcs": ("bearing_capacity_agent", "bearing_capacity_list_methods", "bearing_capacity_describe_method"),
        "brief": "Shallow foundation bearing capacity (Vesic/Meyerhof/Hansen)",
    },
    "settlement": {
        "module": "settlement_agent_foundry",
        "funcs": ("settlement_agent", "settlement_list_methods", "settlement_describe_method"),
        "brief": "Foundation settlement (elastic, Schmertmann, consolidation)",
    },
    "axial_pile": {
        "module": "axial_pile_agent_foundry",
        "funcs": ("axial_pile_agent", "axial_pile_list_methods", "axial_pile_describe_method"),
        "brief": "Driven pile axial capacity (Nordlund/Tomlinson/Beta)",
    },
    "sheet_pile": {
        "module": "sheet_pile_agent_foundry",
        "funcs": ("sheet_pile_agent", "sheet_pile_list_methods", "sheet_pile_describe_method"),
        "brief": "Sheet pile walls (cantilever and anchored)",
    },
    "lateral_pile": {
        "module": "lateral_pile_agent_foundry",
        "funcs": ("lateral_pile_agent", "lateral_pile_list_methods", "lateral_pile_describe_method"),
        "brief": "Lateral pile analysis (COM624P, 8 p-y models, FD solver)",
    },
    "pile_group": {
        "module": "pile_group_agent_foundry",
        "funcs": ("pile_group_agent", "pile_group_list_methods", "pile_group_describe_method"),
        "brief": "Pile group analysis (rigid cap, 6-DOF, Converse-Labarre)",
    },
    "wave_equation": {
        "module": "wave_equation_agent_foundry",
        "funcs": ("wave_equation_agent", "wave_equation_list_methods", "wave_equation_describe_method"),
        "brief": "Smith 1-D wave equation (bearing graph, drivability)",
    },
    "drilled_shaft": {
        "module": "drilled_shaft_agent_foundry",
        "funcs": ("drilled_shaft_agent", "drilled_shaft_list_methods", "drilled_shaft_describe_method"),
        "brief": "Drilled shaft capacity (GEC-10 alpha/beta/rock socket, LRFD)",
    },
    "seismic_geotech": {
        "module": "seismic_geotech_agent_foundry",
        "funcs": ("seismic_geotech_agent", "seismic_geotech_list_methods", "seismic_geotech_describe_method"),
        "brief": "Seismic evaluation (site class, M-O pressure, liquefaction)",
    },
    "retaining_walls": {
        "module": "retaining_walls_agent_foundry",
        "funcs": ("retaining_walls_agent", "retaining_walls_list_methods", "retaining_walls_describe_method"),
        "brief": "Retaining walls (cantilever + MSE, GEC-11)",
    },
    "ground_improvement": {
        "module": "ground_improvement_agent_foundry",
        "funcs": ("ground_improvement_agent", "ground_improvement_list_methods", "ground_improvement_describe_method"),
        "brief": "Ground improvement (aggregate piers, wick drains, vibro)",
    },
    "slope_stability": {
        "module": "slope_stability_agent_foundry",
        "funcs": ("slope_stability_agent", "slope_stability_list_methods", "slope_stability_describe_method"),
        "brief": "Slope stability (Fellenius/Bishop/Spencer, grid search)",
    },
    "downdrag": {
        "module": "downdrag_agent_foundry",
        "funcs": ("downdrag_agent", "downdrag_list_methods", "downdrag_describe_method"),
        "brief": "Pile downdrag (Fellenius neutral plane, UFC 3-220-20)",
    },
    "geolysis": {
        "module": "geolysis_agent_foundry",
        "funcs": ("geolysis_agent", "geolysis_list_methods", "geolysis_describe_method"),
        "brief": "Soil classification (USCS/AASHTO) + SPT corrections + quick BC",
    },
    "groundhog": {
        "module": "groundhog_agent_foundry",
        "funcs": ("groundhog_agent", "groundhog_list_methods", "groundhog_describe_method"),
        "brief": "90 correlations: phase relations, SPT/CPT, earth pressure, etc.",
    },
    "dm7": {
        "module": "dm7_agent_foundry",
        "funcs": ("dm7_agent", "dm7_list_methods", "dm7_describe_method"),
        "brief": "382 NAVFAC DM7 equations (UFC 3-220-10 and 3-220-20)",
    },
    "calc_package": {
        "module": "calc_package_agent_foundry",
        "funcs": ("calc_package_agent", "calc_package_list_methods", "calc_package_describe_method"),
        "brief": "Generate HTML/PDF calculation packages",
    },
    "opensees": {
        "module": "opensees_agent_foundry",
        "funcs": ("opensees_agent", "opensees_list_methods", "opensees_describe_method"),
        "brief": "OpenSees FEM (PM4Sand DSS, BNWF pile, 1D site response)",
    },
    "pystrata": {
        "module": "pystrata_agent_foundry",
        "funcs": ("pystrata_agent", "pystrata_list_methods", "pystrata_describe_method"),
        "brief": "1D EQL site response (SHAKE-type, Darendeli/Menq)",
    },
    "seismic_signals": {
        "module": "seismic_signals_agent_foundry",
        "funcs": ("seismic_signals_agent", "seismic_signals_list_methods", "seismic_signals_describe_method"),
        "brief": "Earthquake signal processing (response spectra, RotD50/100)",
    },
    "liquepy": {
        "module": "liquepy_agent_foundry",
        "funcs": ("liquepy_agent", "liquepy_list_methods", "liquepy_describe_method"),
        "brief": "CPT-based liquefaction triggering (Boulanger & Idriss 2014)",
    },
    "pygef": {
        "module": "pygef_agent_foundry",
        "funcs": ("pygef_agent", "pygef_list_methods", "pygef_describe_method"),
        "brief": "CPT/borehole file parser (GEF and BRO-XML formats)",
    },
    "hvsrpy": {
        "module": "hvsrpy_agent_foundry",
        "funcs": ("hvsrpy_agent", "hvsrpy_list_methods", "hvsrpy_describe_method"),
        "brief": "HVSR site characterization from ambient noise",
    },
    "gstools": {
        "module": "gstools_agent_foundry",
        "funcs": ("gstools_agent", "gstools_list_methods", "gstools_describe_method"),
        "brief": "Geostatistics: kriging, variogram fitting, random fields",
    },
    "ags4": {
        "module": "ags4_agent_foundry",
        "funcs": ("ags4_agent", "ags4_list_methods", "ags4_describe_method"),
        "brief": "AGS4 geotechnical data format reader/validator",
    },
    "salib": {
        "module": "salib_agent_foundry",
        "funcs": ("salib_agent", "salib_list_methods", "salib_describe_method"),
        "brief": "Sensitivity analysis (Sobol and Morris methods)",
    },
    "pyseismosoil": {
        "module": "pyseismosoil_agent_foundry",
        "funcs": ("pyseismosoil_agent", "pyseismosoil_list_methods", "pyseismosoil_describe_method"),
        "brief": "Nonlinear soil curve calibration (MKZ/HH) + Vs profiles",
    },
    "swprocess": {
        "module": "swprocess_agent_foundry",
        "funcs": ("swprocess_agent", "swprocess_list_methods", "swprocess_describe_method"),
        "brief": "MASW surface wave dispersion analysis",
    },
    "pystra": {
        "module": "pystra_agent_foundry",
        "funcs": ("pystra_agent", "pystra_list_methods", "pystra_describe_method"),
        "brief": "Structural reliability (FORM/SORM/Monte Carlo)",
    },
    "pydiggs": {
        "module": "pydiggs_agent_foundry",
        "funcs": ("pydiggs_agent", "pydiggs_list_methods", "pydiggs_describe_method"),
        "brief": "DIGGS 2.6 XML schema and dictionary validation",
    },
    # --- FHWA/NAVFAC Reference Library Agents (from geotech-references submodule) ---
    "gec6": {
        "module": "agents.gec6_agent",
        "funcs": ("gec6_agent", "gec6_list_methods", "gec6_describe_method"),
        "brief": "GEC-6 Shallow Foundations reference tables, figures, and text",
    },
    "gec7": {
        "module": "agents.gec7_agent",
        "funcs": ("gec7_agent", "gec7_list_methods", "gec7_describe_method"),
        "brief": "GEC-7 Soil Nail Walls reference tables, figures, and text",
    },
    "gec10": {
        "module": "agents.gec10_agent",
        "funcs": ("gec10_agent", "gec10_list_methods", "gec10_describe_method"),
        "brief": "GEC-10 Drilled Shafts reference tables, figures, and text",
    },
    "gec11": {
        "module": "agents.gec11_agent",
        "funcs": ("gec11_agent", "gec11_list_methods", "gec11_describe_method"),
        "brief": "GEC-11 MSE Walls & Slopes reference tables and figures",
    },
    "gec12": {
        "module": "agents.gec12_agent",
        "funcs": ("gec12_agent", "gec12_list_methods", "gec12_describe_method"),
        "brief": "GEC-12 Driven Piles reference tables, figures, and text",
    },
    "gec13": {
        "module": "agents.gec13_agent",
        "funcs": ("gec13_agent", "gec13_list_methods", "gec13_describe_method"),
        "brief": "GEC-13 Ground Modification reference tables, figures, and text",
    },
    "micropile": {
        "module": "agents.micropile_agent",
        "funcs": ("micropile_agent", "micropile_list_methods", "micropile_describe_method"),
        "brief": "Micropile Design (FHWA-NHI-05-039) tables, figures, and text",
    },
    "fema_p2192": {
        "module": "agents.fema_p2192_agent",
        "funcs": ("fema_p2192_agent", "fema_p2192_list_methods", "fema_p2192_describe_method"),
        "brief": "FEMA P-2192 SDC determination, ASCE 7-22 site class, Fa/Fv coefficients",
    },
    "noaa_frost": {
        "module": "agents.noaa_frost_agent",
        "funcs": ("noaa_frost_agent", "noaa_frost_list_methods", "noaa_frost_describe_method"),
        "brief": "NOAA frost depth (Stefan/Berggren) and soil thermal properties",
    },
    "ufc_backfill": {
        "module": "agents.ufc_backfill_agent",
        "funcs": ("ufc_backfill_agent", "ufc_backfill_list_methods", "ufc_backfill_describe_method"),
        "brief": "UFC 3-220-04N backfill compaction, material classification, filter criteria",
    },
    "ufc_dewatering": {
        "module": "agents.ufc_dewatering_agent",
        "funcs": ("ufc_dewatering_agent", "ufc_dewatering_list_methods", "ufc_dewatering_describe_method"),
        "brief": "UFC 3-220-05 dewatering well flow (Thiem/Dupuit), method selection",
    },
    "ufc_expansive": {
        "module": "agents.ufc_expansive_agent",
        "funcs": ("ufc_expansive_agent", "ufc_expansive_list_methods", "ufc_expansive_describe_method"),
        "brief": "UFC 3-220-07 expansive soil swell potential, heave, foundation selection",
    },
    "ufc_pavement": {
        "module": "agents.ufc_pavement_agent",
        "funcs": ("ufc_pavement_agent", "ufc_pavement_list_methods", "ufc_pavement_describe_method"),
        "brief": "UFC 3-260-02 CBR pavement design, frost susceptibility, aircraft loads",
    },
}

AGENT_NAMES = sorted(_AGENT_SPECS.keys())


# ---------------------------------------------------------------------------
# Lazy loader — caches imported functions per agent
# ---------------------------------------------------------------------------

_loaded_agents = {}  # agent_name -> {"agent": fn, "list": fn, "describe": fn}


def _load_agent(agent_name: str) -> dict:
    """Import an agent module on demand and cache its 3 functions."""
    if agent_name in _loaded_agents:
        return _loaded_agents[agent_name]

    spec = _AGENT_SPECS[agent_name]
    mod = importlib.import_module(spec["module"])
    agent_fn, list_fn, describe_fn = spec["funcs"]
    entry = {
        "agent": getattr(mod, agent_fn),
        "list": getattr(mod, list_fn),
        "describe": getattr(mod, describe_fn),
    }
    _loaded_agents[agent_name] = entry
    return entry


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
    if agent_name not in _AGENT_SPECS:
        return {"error": f"Unknown agent '{agent_name}'. Available: {AGENT_NAMES}"}
    try:
        funcs = _load_agent(agent_name)
    except Exception as e:
        return {"error": f"Failed to load agent '{agent_name}': {e}"}
    params_json = json.dumps(parameters)
    raw = funcs["agent"](method, params_json)
    return _parse_result(raw)


def list_methods(agent_name: str, category: str = "") -> dict:
    """List available methods for a specific agent."""
    if agent_name not in _AGENT_SPECS:
        return {"error": f"Unknown agent '{agent_name}'. Available: {AGENT_NAMES}"}
    try:
        funcs = _load_agent(agent_name)
    except Exception as e:
        return {"error": f"Failed to load agent '{agent_name}': {e}"}
    raw = funcs["list"](category)
    return _parse_result(raw)


def describe_method(agent_name: str, method: str) -> dict:
    """Get full parameter documentation for a method."""
    if agent_name not in _AGENT_SPECS:
        return {"error": f"Unknown agent '{agent_name}'. Available: {AGENT_NAMES}"}
    try:
        funcs = _load_agent(agent_name)
    except Exception as e:
        return {"error": f"Failed to load agent '{agent_name}': {e}"}
    raw = funcs["describe"](method)
    return _parse_result(raw)


def list_agents() -> dict:
    """List all available agents with brief descriptions."""
    return {name: spec["brief"] for name, spec in _AGENT_SPECS.items()}
