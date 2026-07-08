"""Funhouse agent tool dispatch — routes tool calls to analysis modules.

Dispatches directly to analysis modules via adapter functions.
No dependency on foundry/ files.

Tools:
    list_agents()                       → available modules
    list_methods(agent_name, category)  → methods for a module
    describe_method(agent_name, method) → parameter documentation
    call_agent(agent_name, method, params) → execute calculation
"""

import json
import difflib
import importlib

from funhouse_agent.adapters import MODULE_REGISTRY


# ---------------------------------------------------------------------------
# Lazy loader — caches imported adapter modules
# ---------------------------------------------------------------------------

_loaded_adapters = {}


def _load_adapter(agent_name: str):
    """Import an adapter module on demand and cache it."""
    if agent_name in _loaded_adapters:
        return _loaded_adapters[agent_name]

    spec = MODULE_REGISTRY[agent_name]
    mod = importlib.import_module(spec["adapter"])
    _loaded_adapters[agent_name] = mod
    return mod


# ---------------------------------------------------------------------------
# Dispatch functions — same interface as chat_agent/agent_registry.py
# ---------------------------------------------------------------------------

AGENT_NAMES = sorted(MODULE_REGISTRY.keys())


# Reference modules: the geotech-references agents plus the cross-reference and
# figure-catalog search DBs. The primary agent never calls these directly — all
# reference access is routed through the consult sub-agent (see reviewer.py and
# the consult_references tool). Add any NEW reference module here so it stays off
# the primary agent's direct tool surface.
REFERENCE_MODULES = frozenset({
    "reference_db", "figure_db",
    "dm7",
    "gec4", "gec5", "gec6", "gec7", "gec8", "gec9",
    "gec10", "gec11", "gec12", "gec13", "gec14",
    "micropile",
    "ufc_backfill", "ufc_expansive", "ufc_pavement",
    "fema_p2082", "california_trenching", "fhwa_pavements",
})

# Analysis (computation) modules: everything the primary agent may call directly.
ANALYSIS_MODULES = frozenset(MODULE_REGISTRY) - REFERENCE_MODULES


# ---------------------------------------------------------------------------
# Narrow-reviewer scoping sets (v5.4 D6 — seismic reviewer)
# ---------------------------------------------------------------------------
# The seismic reviewer (funhouse_agent.reviewers.make_seismic_reviewer and the
# .claude/agents/seismic-reviewer.md Claude Code agent) is scoped to these sets
# via ``allowed_agents``. Chosen by domain, cross-checked against MODULE_REGISTRY
# names and the reference library's actual seismic content — see the D6 report /
# funhouse_agent/review_checklists.py.

# Seismic ANALYSIS modules the reviewer may run to verify a calc. The core
# seismic-native tools plus slope_stability (pseudo-static / Newmark / Jibson),
# the Vs-characterization tools that feed Vs30 -> site class (hvsrpy, swprocess),
# and fem2d (seismic-adjacent dynamic / effective-stress FEM). Excludes the
# static foundation/retaining/pile modules and the general-purpose reliability /
# sensitivity / geostatistics / data-I/O tools.
SEISMIC_MODULES = frozenset({
    "seismic_geotech",   # site class, Fpga, Mononobe-Okabe, NCEER liquefaction, residual strength
    "liquefaction",      # unified triggering (CPT/SPT; B&I-2014 default, NCEER via method)
    "liquepy",           # Boulanger & Idriss 2014 CPT/SPT + CPT indices/correlations
    "slope_stability",   # pseudo-static kh, yield acceleration, Newmark, Jibson
    "opensees",          # PM4Sand cyclic DSS + effective-stress 1D site response
    "pystrata",          # equivalent-linear (SHAKE-type) 1D site response
    "seismic_signals",   # response spectra, intensity measures, RotD
    "hvsrpy",            # HVSR site period / amplification from ambient noise
    "swprocess",         # MASW surface-wave dispersion -> Vs profile -> Vs30 -> site class
    "fem2d",             # 2D FEM — seismic-adjacent (dynamic / effective-stress)
})

# Seismic-relevant REFERENCE modules (a subset of REFERENCE_MODULES). Picked
# from the library's ACTUAL seismic content (chapter-text search), not the
# briefs: fema_p2082 is the core seismic site-design reference; gec11/gec7 carry
# the seismic earth-pressure / pseudo-static wall provisions; gec5 the seismic
# site characterization + hazards; dm7 the general/dynamic soil-mechanics
# anchor. The three UFC modules in the library (backfill/expansive/pavement) are
# NOT seismic and are deliberately excluded. reference_db/figure_db search the
# WHOLE library, so seismic content in any reference is still reachable.
SEISMIC_REFERENCES = frozenset({
    "reference_db", "figure_db",
    "fema_p2082",   # 2020 NEHRP: site class (+ BC/CD/DE), SDS/SD1, design spectrum, SDC
    "dm7",          # NAVFAC DM7 — general + dynamic soil properties, seismic settlement
    "gec5",         # site characterization: Vs/Gmax, seismic hazards, liquefaction screening
    "gec7",         # pseudo-static seismic design of soil nail walls
    "gec11",        # seismic external/internal stability of MSE walls (M-O)
})


# ---------------------------------------------------------------------------
# Narrow-reviewer scoping sets (v5.4 F8 — foundations / earth-retention / slope-FEM)
# ---------------------------------------------------------------------------
# Three more reviewer families on the same D6 pattern. Module/reference names are
# verified against MODULE_REGISTRY (ANALYSIS_MODULES / REFERENCE_MODULES); the
# reference picks are from each reference's ACTUAL content (see the briefs), not
# its title. NOTE: geotech_common is a shared library, NOT a registered agent, so
# it is intentionally not a scope member (adding it would be a no-op).

# Foundations reviewer — shallow + deep foundations, ground improvement.
FOUNDATIONS_MODULES = frozenset({
    "bearing_capacity",  # shallow foundations (CBEAR/Vesic/Meyerhof, GWT-in-wedge)
    "settlement",        # consolidation + immediate (CSETT/Schmertmann/Hough)
    "axial_pile",        # driven pile capacity (Nordlund/Tomlinson/Beta)
    "drilled_shaft",     # GEC-10 alpha/beta/rock socket
    "pile_group",        # rigid-cap 6-DOF groups, Converse-Labarre efficiency
    "lateral_pile",      # COM624P p-y lateral analysis
    "wave_equation",     # Smith 1-D wave equation / drivability
    "downdrag",          # Fellenius neutral plane / negative skin friction
    "ground_improvement",# aggregate piers, wick drains, surcharge (GEC-13)
})
FOUNDATIONS_REFERENCES = frozenset({
    "reference_db", "figure_db",
    "dm7",          # NAVFAC DM7 — bearing, settlement, deep foundations
    "gec6",         # shallow foundations
    "gec8",         # CFA pile design
    "gec9",         # laterally loaded piles
    "gec10",        # drilled shafts
    "gec12",        # driven piles
    "gec13",        # ground modification / ground improvement
    "micropile",    # micropile design
    "ufc_expansive",# foundations on expansive soils
})

# Earth-retention reviewer — walls, excavation support, seismic earth pressure.
EARTH_RETENTION_MODULES = frozenset({
    "sheet_pile",       # cantilever / anchored sheet-pile walls
    "soe",              # support of excavation (braced/cantilever, anchors, heave)
    "retaining_walls",  # cantilever + MSE walls (GEC-11)
    "seismic_geotech",  # Mononobe-Okabe seismic earth pressure ONLY (see checklist)
})
EARTH_RETENTION_REFERENCES = frozenset({
    "reference_db", "figure_db",
    "dm7",                 # NAVFAC DM7 — earth pressures, retaining structures
    "gec4",                # ground anchors
    "gec7",                # soil nail walls
    "gec11",               # MSE walls & reinforced soil slopes
    "california_trenching",# Caltrans T&S — temporary excavation support / shoring
})

# Slope / FEM reviewer — slope stability, continuum FEM, reliability, geometry ingest.
SLOPE_FEM_MODULES = frozenset({
    "slope_stability",  # LE (OMS/Bishop/Spencer/GLE) + search + reinforcement
    "fem2d",            # 2D plane-strain FEM (SRM, staged, seepage/consolidation)
    "reliability",      # FOSM/PEM/MC/FORM probabilistic engines
    "dxf_import",       # DXF geometry ingest for slope/FEM
    "pdf_import",       # PDF cross-section ingest
    "drawing_ir",       # LLM-ready drawing digitization (geometry ingest)
})
SLOPE_FEM_REFERENCES = frozenset({
    "reference_db", "figure_db",
    "dm7",     # NAVFAC DM7 — slope stability (Ch 7), general soil mechanics for FEM
    "gec7",    # soil nail walls — the library's soil-nail reference (slope nails)
    "gec11",   # reinforced soil slopes (geosynthetic-reinforced slopes)
})


def _scoped_names(allowed_agents):
    """Return the visible agent names given an optional whitelist."""
    if allowed_agents is None:
        return AGENT_NAMES
    return sorted(name for name in MODULE_REGISTRY if name in allowed_agents)


def _is_visible(agent_name: str, allowed_agents) -> bool:
    if agent_name not in MODULE_REGISTRY:
        return False
    if allowed_agents is None:
        return True
    return agent_name in allowed_agents


def list_agents(allowed_agents=None) -> dict:
    """List available analysis modules with brief descriptions.

    If ``allowed_agents`` is provided, only those modules are returned —
    used by the reviewer agent to scope its view to reference modules only.
    """
    return {
        name: spec["brief"]
        for name, spec in MODULE_REGISTRY.items()
        if allowed_agents is None or name in allowed_agents
    }


def list_methods(agent_name: str, category: str = "", allowed_agents=None) -> dict:
    """List available methods for a specific module."""
    if not _is_visible(agent_name, allowed_agents):
        return {
            "error": f"Unknown module '{agent_name}'. "
                     f"Available: {_scoped_names(allowed_agents)}"
        }
    try:
        mod = _load_adapter(agent_name)
    except Exception as e:
        return {"error": f"Failed to load module '{agent_name}': {e}"}
    # Each adapter exports METHOD_INFO with method_name -> {category, brief, ...}
    result = {}
    for method_name, info in mod.METHOD_INFO.items():
        if info.get("alias_of"):
            continue  # semantic alias — callable/describable but not listed
        cat = info.get("category", "General")
        if category and cat.lower() != category.lower():
            continue
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return result


def describe_method(agent_name: str, method: str, allowed_agents=None) -> dict:
    """Get full parameter documentation for a method."""
    if not _is_visible(agent_name, allowed_agents):
        return {
            "error": f"Unknown module '{agent_name}'. "
                     f"Available: {_scoped_names(allowed_agents)}"
        }
    try:
        mod = _load_adapter(agent_name)
    except Exception as e:
        return {"error": f"Failed to load module '{agent_name}': {e}"}
    if method not in mod.METHOD_INFO:
        available = sorted(mod.METHOD_INFO.keys())
        return {"error": f"Unknown method '{method}'. Available: {available}"}
    return mod.METHOD_INFO[method]


def _resolve_attachment(parameters: dict, attachments: dict) -> dict:
    """If parameters contains attachment_key, decode it to content.

    Bridges the widget/file-upload attachment system to adapters that
    accept text content (e.g. parse_diggs with DIGGS XML).
    """
    key = parameters.get("attachment_key")
    if not key or not attachments:
        return parameters
    if key not in attachments:
        available = sorted(attachments.keys()) or ["(none)"]
        raise KeyError(
            f"attachment_key '{key}' not found. Available: {available}"
        )
    raw = attachments[key]
    if isinstance(raw, (bytes, bytearray)):
        content = raw.decode("utf-8", errors="replace")
    else:
        content = str(raw)
    params = dict(parameters)
    params["content"] = content
    params.pop("attachment_key")
    return params


# ---------------------------------------------------------------------------
# Smart method resolution — cut the agent's method-name guessing
# ---------------------------------------------------------------------------
# Params that select a sub-method by VALUE; the agent often guesses the value as
# if it were the method NAME (e.g. call_agent('bearing_capacity', 'vesic', ...)).
_SELECTOR_PARAMS = {"method", "factor_method", "analysis_method",
                    "correlation", "formula", "approach"}

# Curated aliases for method names the agent commonly guesses (sourced from the
# agent test-suite triage; module_work/module_feedback.json). Keyed by
# (agent_name, guessed_name_lower); value is the real method name, or a
# (real_method, {param: value}) tuple when a selector value must be injected.
# Every entry must be verified to resolve to the CORRECT method/result — no
# blind enum routing (a value advertised on a "factors" helper would otherwise
# mis-route a full-analysis request).
_METHOD_ALIASES = {
    # --- foundations / settlement ---
    # Bearing-capacity factor methods the agent guesses as method names. The
    # full analysis takes a ``factor_method`` selector; inject it where the
    # guess names a specific theory, else route to the default-vesic analysis.
    ("bearing_capacity", "vesic"):
        ("bearing_capacity_analysis", {"factor_method": "vesic"}),
    ("bearing_capacity", "vesic_footing"):
        ("bearing_capacity_analysis", {"factor_method": "vesic"}),
    # terzaghi/two-layer aren't separate factor methods — the full analysis
    # covers them (two-layer via the layer2_* params), default factor_method.
    ("bearing_capacity", "terzaghi"): "bearing_capacity_analysis",
    ("bearing_capacity", "two_layer_clay"): "bearing_capacity_analysis",
    ("settlement", "consolidation"): "consolidation_settlement",
    ("settlement", "elastic_foundation"): "elastic_settlement",
    # --- deep foundations ---
    # Drilled-shaft theory names → the one full analysis (alpha/beta/rock
    # socket are auto-selected per layer soil_type, not a method choice).
    ("drilled_shaft", "alpha_method"): "drilled_shaft_capacity",
    ("drilled_shaft", "beta_method"): "drilled_shaft_capacity",
    ("drilled_shaft", "rock_socket_capacity"): "drilled_shaft_capacity",
    ("drilled_shaft", "single_shaft_capacity"): "drilled_shaft_capacity",
    ("axial_pile", "beta_method"): "axial_pile_capacity",
    ("downdrag", "fellenius_neutral_plane"): "downdrag_analysis",
    # analyze_lateral_pile — verb-prefixed guess (2026-07-05 eval run).
    ("lateral_pile", "analyze_lateral_pile"): "lateral_pile_analysis",
    # --- earth retention / ground improvement ---
    # Names the agent guessed for these tools (2026-07-05 eval run). The
    # earth-pressure coefficient tool is the Rankine/Coulomb K helper; the
    # aggregate-pier tool is the GEC-13 design method.
    ("retaining_walls", "earth_pressure_analysis"): "earth_pressure_coefficient",
    # Rankine/Coulomb K guessed by name ON retaining_walls (the right module) —
    # route in-module to the real earth-pressure coefficient helper. The SAME
    # names guessed on the WRONG module are handled by _CROSS_MODULE_REDIRECTS.
    ("retaining_walls", "rankine_coefficients"): "earth_pressure_coefficient",
    ("retaining_walls", "rankine_earth_pressure"): "earth_pressure_coefficient",
    ("retaining_walls", "rankine_ka"): "earth_pressure_coefficient",
    ("retaining_walls", "rankine_kp"): "earth_pressure_coefficient",
    ("retaining_walls", "coulomb_coefficients"): "earth_pressure_coefficient",
    ("retaining_walls", "active_earth_pressure"): "earth_pressure_coefficient",
    ("retaining_walls", "passive_earth_pressure"): "earth_pressure_coefficient",
    ("ground_improvement", "aggregate_pier_design"): "aggregate_piers",
    # --- slope / FEM ---
    ("fem2d", "slope_strength_reduction"): "fem2d_slope_srm",
    # --- unified liquefaction tool ---
    # The single liquefaction method auto-routes by input type + method; map the
    # names the agent commonly guesses onto it (CPT/SPT, B&I-2014, NCEER/Youd).
    ("liquefaction", "liquefaction_triggering"): "liquefaction_analysis",
    ("liquefaction", "evaluate_liquefaction"): "liquefaction_analysis",
    ("liquefaction", "cpt_liquefaction"): "liquefaction_analysis",
    ("liquefaction", "spt_liquefaction"): "liquefaction_analysis",
    ("liquefaction", "boulanger_idriss_2014"): "liquefaction_analysis",
    ("liquefaction", "bi2014"): ("liquefaction_analysis", {"method": "bi2014"}),
    ("liquefaction", "nceer2001"): ("liquefaction_analysis", {"method": "nceer2001"}),
    # cpt_based_triggering — descriptive guess; auto-routes by CPT input
    # (2026-07-05 eval run).
    ("liquefaction", "cpt_based_triggering"): "liquefaction_analysis",
    # --- other analysis modules ---
    ("liquepy", "cpt_boulanger_idriss_2014"): "cpt_liquefaction",
    ("liquepy", "spt_boulanger_idriss_2014"): "spt_liquefaction",
    ("salib", "sobol_sensitivity"): "sobol_sample",
    ("pystrata", "equivalent_linear"): "eql_site_response",
    ("gstools", "fit_variogram"): "variogram",
    # ordinary_kriging / discover_dxf — names guessed in the 2026-07-05 eval run.
    ("gstools", "ordinary_kriging"): "kriging",
    ("dxf_import", "discover_dxf"): "discover_layers",
    # subsurface_characterization is the single data-I/O home; the former
    # pygef/ags4/pydiggs modules are folded in as format-adapter methods.
    ("subsurface", "read_and_validate"): "read_ags4",
    ("dxf_export", "export_cross_section"): "export_geometry_to_dxf",
}


# Cross-module SELECTION mis-routing: a method name asked of the WRONG module,
# mapped to the module + method that actually implements it. Keyed by the
# guessed name (lowercased) — the wrong module the agent picked is irrelevant,
# so ANY module that lacks the method redirects to the right one. Unlike
# _METHOD_ALIASES (same-module, auto-executed), these NEVER auto-execute: a
# different module has different required parameters, so silently running it
# could return a confidently wrong answer. Instead call_agent returns a
# did-you-mean error naming the right module+method. Sourced from the agent
# test-suite triage (module_work/module_feedback.json): Rankine/Coulomb earth-
# pressure coefficients repeatedly guessed on bearing_capacity / seismic_geotech
# (the real home is retaining_walls.earth_pressure_coefficient).
_CROSS_MODULE_REDIRECTS = {
    "rankine_coefficients": ("retaining_walls", "earth_pressure_coefficient"),
    "rankine_earth_pressure": ("retaining_walls", "earth_pressure_coefficient"),
    "rankine_ka": ("retaining_walls", "earth_pressure_coefficient"),
    "rankine_kp": ("retaining_walls", "earth_pressure_coefficient"),
    "rankine_k0": ("retaining_walls", "earth_pressure_coefficient"),
    "coulomb_coefficients": ("retaining_walls", "earth_pressure_coefficient"),
    "coulomb_earth_pressure": ("retaining_walls", "earth_pressure_coefficient"),
    "earth_pressure_coefficients": ("retaining_walls", "earth_pressure_coefficient"),
    "active_earth_pressure": ("retaining_walls", "earth_pressure_coefficient"),
    "passive_earth_pressure": ("retaining_walls", "earth_pressure_coefficient"),
}


def _cross_module_redirect(agent_name: str, method: str, allowed_agents=None):
    """Point a method guessed on the wrong module at the module that has it.

    Returns ``(right_agent, right_method)`` or ``None``. Only fires when the
    target module is visible and really exposes the method, and never for a
    same-module guess (those are handled by ``_METHOD_ALIASES``).
    """
    target = _CROSS_MODULE_REDIRECTS.get(method.strip().lower())
    if target is None:
        return None
    right_agent, right_method = target
    if right_agent == agent_name:
        return None
    if not _is_visible(right_agent, allowed_agents):
        return None
    try:
        tmod = _load_adapter(right_agent)
    except Exception:
        return None
    if right_method not in tmod.METHOD_REGISTRY:
        return None
    return right_agent, right_method


def _selector_value_candidates(mod, name: str):
    """Methods whose selector param advertises ``name`` as an allowed value."""
    target = name.strip().lower()
    hits = []
    for m, info in mod.METHOD_INFO.items():
        if info.get("alias_of"):
            continue
        for p, pinfo in (info.get("parameters") or {}).items():
            if p in _SELECTOR_PARAMS and any(
                    str(v).lower() == target
                    for v in (pinfo.get("allowed_values") or [])):
                hits.append((m, p))
                break
    return hits


def _resolve_unknown_method(mod, agent_name: str, method: str, parameters: dict):
    """Resolve a guessed method via the curated alias map.

    Returns ``(real_method, new_params)`` or ``None``.  Only curated (verified)
    aliases route automatically; selector-value guesses are surfaced as a
    directive error instead (see call_agent), never auto-routed.
    """
    entry = _METHOD_ALIASES.get((agent_name, method.strip().lower()))
    if entry is None:
        return None
    real, inject = (entry, {}) if isinstance(entry, str) else entry
    if real in mod.METHOD_REGISTRY:
        return real, {**parameters, **(inject or {})}
    return None


def call_agent(
    agent_name: str,
    method: str,
    parameters: dict,
    attachments: dict = None,
    allowed_agents=None,
) -> dict:
    """Execute a geotechnical calculation.

    Parameters
    ----------
    agent_name : str
        One of the registered module names.
    method : str
        Method name within that module.
    parameters : dict
        Flat dict of parameters.
    attachments : dict, optional
        Agent attachments ({key: bytes}).  If parameters contains an
        ``attachment_key``, the corresponding bytes are decoded to text
        and injected as ``content`` before calling the adapter.

    Returns
    -------
    dict
        Calculation results or {"error": "..."}.
    """
    if not _is_visible(agent_name, allowed_agents):
        return {
            "error": f"Unknown module '{agent_name}'. "
                     f"Available: {_scoped_names(allowed_agents)}"
        }
    try:
        mod = _load_adapter(agent_name)
    except Exception as e:
        return {"error": f"Failed to load module '{agent_name}': {e}"}
    if method not in mod.METHOD_REGISTRY:
        resolved = _resolve_unknown_method(mod, agent_name, method, parameters)
        if resolved is not None:
            method, parameters = resolved
        else:
            available = sorted(k for k, v in mod.METHOD_INFO.items()
                               if not v.get("alias_of"))
            redirect = _cross_module_redirect(agent_name, method, allowed_agents)
            if redirect is not None:
                right_agent, right_method = redirect
                return {"error":
                        f"'{method}' is not a '{agent_name}' method — it lives "
                        f"on module '{right_agent}' as '{right_method}'. Call "
                        f"call_agent('{right_agent}', '{right_method}', {{...}}). "
                        f"Available '{agent_name}' methods: {available}"}
            cands = _selector_value_candidates(mod, method)
            if cands:
                opts = ", ".join(f"{m}({p}='{method}')" for m, p in cands)
                return {"error": f"'{method}' is a value for a selector "
                                 f"parameter, not a method name — call: {opts}. "
                                 f"Available methods: {available}"}
            near = difflib.get_close_matches(method, available, n=3, cutoff=0.5)
            hint = f" Did you mean: {near}?" if near else ""
            return {"error": f"Unknown method '{method}'.{hint} "
                             f"Available: {available}"}
    try:
        if attachments and "attachment_key" in parameters:
            parameters = _resolve_attachment(parameters, attachments)
        result = mod.METHOD_REGISTRY[method](parameters)
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# ToolCall dispatch — drop-in replacement for chat_agent.agent.dispatch_tool
# ---------------------------------------------------------------------------

def dispatch_tool(tool_call, attachments: dict = None, allowed_agents=None) -> str:
    """Route a parsed ToolCall to the adapter registry and return JSON string.

    Parameters
    ----------
    tool_call : ToolCall
        Parsed tool call from the LLM.
    attachments : dict, optional
        Agent attachments ({key: bytes}).
    allowed_agents : iterable of str, optional
        Whitelist of agent names. If provided, modules outside this set are
        invisible to ``list_agents`` / ``list_methods`` / ``describe_method``
        and refused by ``call_agent``. Used by the reviewer agent to scope
        its tool surface to reference modules only.
    """
    name = tool_call.tool_name
    args = tool_call.arguments

    if name == "call_agent":
        result = call_agent(
            agent_name=args.get("agent_name", ""),
            method=args.get("method", ""),
            parameters=args.get("parameters", {}),
            attachments=attachments,
            allowed_agents=allowed_agents,
        )
    elif name == "list_methods":
        result = list_methods(
            agent_name=args.get("agent_name", ""),
            category=args.get("category", ""),
            allowed_agents=allowed_agents,
        )
    elif name == "describe_method":
        result = describe_method(
            agent_name=args.get("agent_name", ""),
            method=args.get("method", ""),
            allowed_agents=allowed_agents,
        )
    elif name == "list_agents":
        result = list_agents(allowed_agents=allowed_agents)
    else:
        result = {"error": f"Unknown tool '{name}'"}

    return json.dumps(result, default=str)
