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
import inspect
import os
import re
from typing import Optional

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
    "em_2104", "em_2107",
    "gec4", "gec5", "gec6", "gec7", "gec8", "gec9",
    "gec10", "gec11", "gec12", "gec13", "gec14",
    "micropile",
    "ufc_backfill", "ufc_expansive", "ufc_pavement",
    "ufc_stabilization", "ufc_flexible_practice", "ufc_concrete_practice",
    "ufc_structural", "ufc_collapse", "gsa_collapse", "wood_handbook",
    "fema_p2082", "california_trenching", "fhwa_pavements",
    "eurocode_7_1", "eurocode_7_2", "aashto_1993",
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

#: Pavement specialist scope: the AASHTO 1993 design module + the report
#: generator it uses, plus the pavement/roadbed reference modules.
PAVEMENT_MODULES = frozenset({
    "pavement_design",   # AASHTO 1993 flexible SN / rigid D / ESALs / swell-frost
    "calc_package",      # pavement_design_package + html_to_pdf report rendering
})

PAVEMENT_REFERENCES = frozenset({
    "reference_db", "figure_db",
    "aashto_1993",      # AASHTO design basis: Figs 3.1/3.7, Appendix D LEFs, composite-k, Appendix G
    "ufc_pavement",     # UFC 3-250-01 roads/parking design (rebuilt 2026-07 from the real doc: CBR curves, overlays, frost, drainage)
    "fhwa_pavements",   # FHWA-NHI-05-037 — Mr/CBR correlations, drainage, frost, stabilization
    "ufc_stabilization",       # UFC 3-250-11 — stabilizer selection, criteria, equivalency
    "ufc_flexible_practice",   # UFC 3-250-03 — asphalt materials/construction practice
    "ufc_concrete_practice",   # UFC 3-250-04 — concrete materials/construction practice
    "ufc_expansive",    # expansive-soil roadbeds (feeds the Appendix G swelling inputs)
})

#: Structural specialist scope: the cross-section / RC-section / frame
#: analysis engines, plus the general-purpose probabilistic wrap and report
#: renderer, that a structural calc normally chains together.
STRUCTURAL_MODULES = frozenset({
    "section_props",   # cross-section properties engine (A, I, Z, S, J, warping — steel shapes + polygons)
    "concrete_props",  # RC rectangular-section capacity (cracked/gross Ixx, Mcr, nominal Mn, N-M interaction)
    "pynite",          # linear elastic 2D/3D frame + continuous-beam analysis (reactions, M/V/deflection)
    "opensees",        # nonlinear / dynamic FE analyses (PM4Sand cyclic DSS, 1D site response)
    "fem2d",           # 2D continuum FEM (plane-strain; foundation / soil-structure interaction)
    "reliability",     # FOSM/PEM/MC/FORM probabilistic wrap around load/capacity variability
    "calc_package",    # report rendering (html_to_pdf + module calc-package templates)
})

# Structural reference layer: USACE EM 1110-2-2104/2107 (hydraulic RC/steel
# structures), UFC 3-301-01 (DoD structural design criteria, IBC/ASCE 7-22
# modifications), the progressive-collapse pair (UFC 4-023-03 + its GSA
# civilian sibling), the USDA Wood Handbook, plus the one structural-
# adjacent UFC practice module and the whole-library search DBs.
STRUCTURAL_REFERENCES = frozenset({
    "reference_db", "figure_db",
    "em_2104",                 # EM 1110-2-2104 — RC hydraulic structures: loads, flexure/axial, shear
    "em_2107",                 # EM 1110-2-2107 — hydraulic steel structures: LRFD loads, seismic amplification, Tainter-gate loads
    "ufc_structural",          # UFC 3-301-01 — DoD structural criteria: risk categories, live loads, Table 3-1 seismic systems
    "ufc_collapse",            # UFC 4-023-03 — DoD progressive collapse: tie forces, Alternate Path, ELR, RC/steel m-factors
    "gsa_collapse",            # GSA Alternate Path 2016 — civilian progressive collapse: FSL applicability, Redundancy Requirements
    "wood_handbook",           # USDA Wood Handbook — clear-wood properties, EMC/shrinkage, Ch 9 structural equations, fastenings
    "ufc_concrete_practice",   # UFC 3-250-04 — concrete materials/construction practice
})


def _scoped_names(allowed_agents):
    """Return the visible agent names given an optional whitelist."""
    if allowed_agents is None:
        return AGENT_NAMES
    return sorted(name for name in MODULE_REGISTRY if name in allowed_agents)


def _canonical_agent_name(agent_name):
    """The registry spelling of a module name given as 'SOE', 'Soe', ' soe ',
    'sheet-pile'... Field feedback 2026-09-15 (N11): the calc sub-agent's
    first three lookups failed on 'SOE'. Unknown names pass through."""
    if not isinstance(agent_name, str) or agent_name in MODULE_REGISTRY:
        return agent_name
    key = agent_name.strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in MODULE_REGISTRY else agent_name


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


#: Words that say nothing about WHICH method is meant ("analyze_downdrag",
#: "cantilever_wall_analysis"); dropped before guessed names are compared.
_NAME_FILLER = frozenset({
    "analysis", "analyze", "analyse", "method", "methods", "calc", "calculate",
    "calculation", "compute", "run", "get", "do", "the", "of", "for", "a",
})


def _name_tokens(text: str) -> set:
    """Lower-case word tokens of a method name or brief, filler removed."""
    import re
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower())
            if t and t not in _NAME_FILLER}


def _closest_methods(mod, method: str, n: int = 3) -> list:
    """The real methods of ``mod`` closest to a guessed ``method`` name.

    Scores each listed method by spelling (difflib on the name) and by the
    guess's words found in the method's name and one-line brief, so
    'driven_pile_capacity' finds 'axial_pile_capacity' and
    'cantilever_wall_analysis' finds 'cantilever_wall'. Always returns the
    best ``n`` (never an empty hint for a module that has methods).
    """
    guess = str(method or "").strip().lower()
    g_tokens = _name_tokens(guess)
    scored = []
    for name, info in mod.METHOD_INFO.items():
        if info.get("alias_of"):
            continue
        spelling = difflib.SequenceMatcher(None, guess, name.lower()).ratio()
        in_name = in_brief = 0.0
        if g_tokens:
            n_tok = _name_tokens(name)
            b_tok = _name_tokens(info.get("brief", ""))
            in_name = len(g_tokens & n_tok) / len(g_tokens)
            in_brief = len(g_tokens & b_tok) / len(g_tokens)
        scored.append((0.45 * spelling + 0.4 * in_name + 0.15 * in_brief,
                       name))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [name for _, name in scored[:n]]


def _topic_matches(mod, topic: str) -> dict:
    """Methods whose name, brief or parameter names/descriptions mention every
    word of ``topic`` — what an agent means by a ``category`` that is not one
    of the module's categories ('slope' on fem2d, 'DIGGS' on subsurface)."""
    words = [w for w in _name_tokens(topic) if len(w) > 1]
    if not words:
        return {}
    hits = {}
    for name, info in mod.METHOD_INFO.items():
        if info.get("alias_of"):
            continue
        params = info.get("parameters") or {}
        text = " ".join([name, str(info.get("brief", ""))]
                        + [f"{p} {(d or {}).get('description', '')} "
                           f"{(d or {}).get('brief', '')}"
                           for p, d in params.items()
                           if isinstance(d, dict)]).lower()
        if all(w in text for w in words):
            hits.setdefault(info.get("category", "General"), {})[name] = \
                info["brief"]
    return hits


def list_methods(agent_name: str = "", category: str = "",
                 allowed_agents=None) -> dict:
    """List available methods for a specific module.

    With no ``agent_name`` the answer is the list of modules to choose from.
    A ``category`` that is not one of the module's categories is read as a
    topic: the methods that mention it, or else all of them — always with a
    ``note`` saying the category matched nothing, never an empty ``{}``.
    """
    if not str(agent_name or "").strip():
        return {
            "error": "list_methods needs agent_name: the module whose methods "
                     "you want. Pick one of the modules below (list_agents "
                     "gives a one-line description of each).",
            "modules": _scoped_names(allowed_agents),
        }
    agent_name = _canonical_agent_name(agent_name)
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
    everything = {}
    for method_name, info in mod.METHOD_INFO.items():
        if info.get("alias_of"):
            continue  # semantic alias — callable/describable but not listed
        cat = info.get("category", "General")
        everything.setdefault(cat, {})[method_name] = info["brief"]
    if not category:
        return everything
    wanted = str(category).strip().lower()
    exact = {c: m for c, m in everything.items() if c.lower() == wanted}
    if exact:
        return exact
    # Field feedback 2026-09-15, N13: an empty {} read as "this module has
    # nothing on that" (a reviewer filtered gec7 by "earth retaining
    # structures", got {}, and cited from memory). The 2026-10 Foundry eval:
    # agents use `category` as a topic ('pile', 'slope', 'DIGGS') and an
    # error sent them guessing again. So a category that is not one of the
    # module's own is answered with the methods that mention it, or with all
    # of them, and the note says which.
    categories = sorted(everything)
    partial = {c: m for c, m in everything.items()
               if wanted in c.lower() or c.lower() in wanted}
    topical = partial or _topic_matches(mod, wanted)
    if topical:
        shown = "methods in the closest categories" if partial else \
            "the methods whose name, description or parameters mention it"
        return {
            "note": f"'{category}' is not a '{agent_name}' category (its "
                    f"categories are {categories}); showing {shown}. Call "
                    f"list_methods('{agent_name}') with no category for all.",
            "available_categories": categories,
            "methods": topical,
        }
    return {
        "note": f"No '{agent_name}' method mentions '{category}' and it is not "
                f"one of its categories ({categories}); showing ALL of the "
                f"module's methods. A category filters by those names only — "
                f"a topic or search term goes in a method's parameters.",
        "available_categories": categories,
        "methods": everything,
    }


def describe_method(agent_name: str, method: str, allowed_agents=None) -> dict:
    """Get full parameter documentation for a method."""
    agent_name = _canonical_agent_name(agent_name)
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
        text_tool = _text_tool_for(mod, method)
        if text_tool is not None and text_tool in mod.METHOD_INFO:
            # 'search' / 'text_search' / 'Text Retrieval' on a reference
            # module: its own text tool's docs (live smoke wave 1, G13)
            return {**mod.METHOD_INFO[text_tool],
                    "_note": f"'{method}' is not a method name — this "
                             f"module's text tool is '{text_tool}'. Showing "
                             "its docs."}
        return _unknown_method_error(mod, agent_name, method, allowed_agents)
    return _documented(mod.METHOD_INFO[method])


#: Parameter keys read by ``defaults_used`` and left out of a method's
#: documentation, the descriptions saying the same in words: when a default
#: applies (GM-119) and what a left-out input with no default stands for
#: (GM-165).
_DEFAULTS_ONLY_KEYS = ("applies_when", "when_absent", "default_from")


def _documented(info: dict) -> dict:
    """A method's documentation as the model reads it: each parameter's
    ``applies_when`` (GM-119), ``when_absent`` and ``default_from``
    (GM-165), read by ``defaults_used``, are left out, the descriptions
    saying the same in words."""
    params = info.get("parameters")
    if not isinstance(params, dict) or not any(
            isinstance(s, dict) and set(s) & set(_DEFAULTS_ONLY_KEYS)
            for s in params.values()):
        return info
    return {**info, "parameters": {
        name: ({k: v for k, v in spec.items()
                if k not in _DEFAULTS_ONLY_KEYS}
               if isinstance(spec, dict) else spec)
        for name, spec in params.items()}}


def _unknown_method_error(mod, agent_name: str, method: str,
                          allowed_agents=None) -> dict:
    """The answer to an unknown method name: the closest real methods first,
    the module that has it when the guess belongs to another module, then the
    full list. Shared by describe_method and call_agent so every unknown-name
    path names where to go next."""
    available = sorted(k for k, v in mod.METHOD_INFO.items()
                       if not v.get("alias_of"))
    closest = _closest_methods(mod, method)
    elsewhere = (_text_search_elsewhere(agent_name, method, allowed_agents)
                 if isinstance(method, str)
                 and _text_tool_for(mod, method) is None else None)
    if elsewhere is not None:
        return {**elsewhere, "closest": closest}
    msg = (f"Unknown method '{method}' for module '{agent_name}'. "
           f"Closest: {closest}.")
    out = {"error": msg, "closest": closest}
    redirect = (_cross_module_redirect(agent_name, method, allowed_agents)
                if isinstance(method, str) else None)
    if redirect is not None:
        right_agent, right_method = redirect
        out["error"] = (
            f"Unknown method '{method}' for module '{agent_name}' — it lives "
            f"on module '{right_agent}' as '{right_method}' "
            f"(describe_method('{right_agent}', '{right_method}')). "
            f"Closest '{agent_name}' methods: {closest}.")
        out["redirect"] = {"agent_name": right_agent, "method": right_method}
    if len(available) <= _MAX_LISTED_METHODS:
        out["available"] = available
    else:
        out["available"] = _available_hint(mod, agent_name, method)
    return out


def _method_params(mod, method: str) -> dict:
    """The documented parameters of ``method`` (``{}`` when undocumented)."""
    try:
        info = mod.METHOD_INFO.get(method) or {}
        params = info.get("parameters") or {}
        return params if isinstance(params, dict) else {}
    except Exception:                                  # noqa: BLE001
        return {}


def _resolve_attachment(parameters: dict, attachments: dict,
                        documented: dict = None) -> dict:
    """Turn an ``attachment_key`` into what the method reads.

    Bridges the upload system to the adapters. The key names an uploaded
    file: its bytes in ``attachments`` (decoded to ``content`` for a method
    that reads text, e.g. parse_diggs with DIGGS XML), or the file of that
    name in the working folder, where the app stages every upload (passed as
    ``file_path``). A method documenting ``file_path`` but no ``content``
    (parse_cpt, read_ags4...) gets the file. Until 2026-10-09 the deep
    agent's ``call_agent`` passed no attachments, so the key never resolved
    and the error then asked for an ``attachment_key`` (live smoke wave 1,
    A8).
    """
    key = parameters.get("attachment_key")
    if not key:
        return parameters
    from funhouse_agent._fileio import find_in_working_folder
    documented = documented or {}
    wants_path = "file_path" in documented and "content" not in documented
    params = dict(parameters)
    params.pop("attachment_key")
    key = str(key)
    on_disk = find_in_working_folder(key)
    if on_disk is None and os.path.isfile(key):
        on_disk = os.path.abspath(key)
    if attachments and key in attachments and not (wants_path and on_disk):
        raw = attachments[key]
        if isinstance(raw, (bytes, bytearray)):
            params["content"] = raw.decode("utf-8", errors="replace")
        else:
            params["content"] = str(raw)
        return params
    if on_disk:
        params["file_path"] = on_disk
        return params
    available = sorted(attachments.keys()) if attachments else []
    raise KeyError(
        f"attachment_key '{key}' is neither an attached file "
        f"(attached: {available or 'none'}) nor a file of that name in the "
        "working folder. Pass the uploaded file's name exactly as the "
        "attachment note gives it, or a 'file_path'.")


#: Parameters naming a file a module method READS (a bare name is looked up
#: in the working folder, as the file tools do).
_INPUT_PATH_PARAMS = ("file_path", "html_path")
_INPUT_PATH_LIST_PARAMS = ("file_paths",)


def _inputs_from_working_folder(parameters):
    """A bare file name given for a file a method reads is the file of that
    name in the working folder when there is no such file relative to the
    process (live smoke wave 1, A6: results and notes name files by their
    conversation-relative name, so the model passes those names back)."""
    if not isinstance(parameters, dict):
        return parameters
    from funhouse_agent._fileio import find_in_working_folder

    def _resolve(value):
        if not isinstance(value, str) or not value.strip():
            return value
        if os.path.isabs(os.path.expanduser(value)) or os.path.exists(value):
            return value
        return find_in_working_folder(value) or value

    out = parameters
    for key in _INPUT_PATH_PARAMS + _INPUT_PATH_LIST_PARAMS:
        if key not in parameters:
            continue
        value = parameters[key]
        new = ([_resolve(v) for v in value] if isinstance(value, list)
               else _resolve(value))
        if new != value:
            if out is parameters:
                out = dict(parameters)
            out[key] = new
    return out


#: What a TypeError says when the CALL did not fit the function's signature
#: (as opposed to a type problem inside the computation).
_SIGNATURE_ERROR = re.compile(
    r"(missing \d+ required (?:positional|keyword-only) argument"
    r"|unexpected keyword argument|got multiple values for argument"
    r"|takes \d+ positional arguments? but \d+ (?:was|were) given"
    r"|required (?:positional|keyword) argument)")


def _parameter_error(mod, agent_name: str, method: str, parameters: dict,
                     exc: BaseException) -> dict:
    """The answer to a call whose parameters did not fit the method: what
    was wrong in plain words, the parameters the method takes (required
    first) and where its documentation is -- never the raw TypeError, which
    named internals such as ``_build.<locals>.reference_get()`` (live smoke
    wave 1, A15b)."""
    documented = _method_params(mod, method)
    required = sorted(k for k, v in documented.items()
                      if isinstance(v, dict) and v.get("required"))
    optional = sorted(k for k in documented if k not in required)
    given = sorted(parameters or {})
    text = str(exc)
    m = re.search(r"unexpected keyword argument '([^']+)'", text)
    if m:
        what = f"'{m.group(1)}' is not a parameter of {method}"
    else:
        missing = re.findall(r"'([^']+)'", text.split("argument", 1)[-1]) \
            if "missing" in text else []
        if missing:
            what = (f"{method} needs "
                    + ", ".join(f"'{p}'" for p in missing))
        else:
            what = f"the parameters given do not fit {method}"
    # A did-you-mean for each given name the method does not take, as the
    # adapters' own check gives (GM-177): a reference function takes
    # 'phi_deg' where the analysis modules take 'phi'. Nothing is renamed.
    from funhouse_agent.adapters import suggest_params
    info = mod.METHOD_INFO.get(method) if hasattr(mod, "METHOD_INFO") else None
    aliases = (info.get("parameter_aliases") or {}) \
        if isinstance(info, dict) else {}
    hints = []
    for k in given:
        if documented and k not in documented and k not in aliases \
                and k not in _DISPATCH_PARAMS:
            near = suggest_params(k, documented)
            if near:
                hints.append(f"{k} -> {' or '.join(near)}")
    hint = f" Did you mean: {'; '.join(hints)}?" if hints else ""
    out = {"error": (f"{agent_name}.{method}: {what}.{hint} "
                     f"Parameters given: {given or 'none'}."),
           # the function's own words, minus internal qualifiers
           "detail": re.sub(r"[\w.]*<locals>\.", "", text),
           "required_parameters": required,
           "optional_parameters": optional,
           "directive": (f"Call call_agent('{agent_name}', '{method}', "
                         "{...}) with these parameter names, inside "
                         "'parameters'; describe_method gives each one's "
                         "meaning and units.")}
    if not documented:
        out.pop("required_parameters")
        out.pop("optional_parameters")
    return out


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
    # Terzaghi (1943) is factor_method='terzaghi' (GM-30; until 2026-10-14 the
    # guess 'terzaghi' quietly ran the default Vesic analysis). Two-layer is
    # not a separate method: the full analysis covers it via layer2_*.
    ("bearing_capacity", "terzaghi"):
        ("bearing_capacity_analysis", {"factor_method": "terzaghi"}),
    ("bearing_capacity", "terzaghi_bearing_capacity"):
        ("bearing_capacity_analysis", {"factor_method": "terzaghi"}),
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
    # Theory / verb-noun guesses for the one driven-pile capacity analysis
    # (Tomlinson alpha is chosen per cohesive layer inside it) and the one
    # drilled-shaft analysis (2026-10 Foundry eval, AP-2/AP-3/DS-2).
    ("axial_pile", "alpha_method"): "axial_pile_capacity",
    ("axial_pile", "driven_pile_capacity"): "axial_pile_capacity",
    ("drilled_shaft", "beta_method_capacity"): "drilled_shaft_capacity",
    ("downdrag", "fellenius_neutral_plane"): "downdrag_analysis",
    ("downdrag", "analyze_downdrag"): "downdrag_analysis",
    # analyze_lateral_pile — verb-prefixed guess (2026-07-05 eval run).
    ("lateral_pile", "analyze_lateral_pile"): "lateral_pile_analysis",
    # --- earth retention / ground improvement ---
    # Names the agent guessed for these tools (2026-07-05 eval run). The
    # earth-pressure coefficient tool is the Rankine/Coulomb K helper; the
    # aggregate-pier tool is the GEC-13 design method.
    ("retaining_walls", "earth_pressure_analysis"): "earth_pressure_coefficient",
    ("retaining_walls", "cantilever_wall_analysis"): "cantilever_wall",
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
    # Caquot-Kerisel log-spiral passive Kp guessed by name ON retaining_walls
    # (the right module) — route to the earth-pressure helper with the
    # theory/state injected (eval EPC-3, 2026-07-13). The SAME names guessed on
    # the WRONG module are handled by _CROSS_MODULE_REDIRECTS.
    ("retaining_walls", "caquot"):
        ("earth_pressure_coefficient", {"theory": "caquot_kerisel", "state": "passive"}),
    ("retaining_walls", "caquot_kerisel"):
        ("earth_pressure_coefficient", {"theory": "caquot_kerisel", "state": "passive"}),
    ("retaining_walls", "caquot_kerisel_kp"):
        ("earth_pressure_coefficient", {"theory": "caquot_kerisel", "state": "passive"}),
    ("retaining_walls", "log_spiral"):
        ("earth_pressure_coefficient", {"theory": "caquot_kerisel", "state": "passive"}),
    ("retaining_walls", "log_spiral_passive"):
        ("earth_pressure_coefficient", {"theory": "caquot_kerisel", "state": "passive"}),
    ("retaining_walls", "passive_coefficient"):
        ("earth_pressure_coefficient", {"state": "passive"}),
    ("ground_improvement", "aggregate_pier_design"): "aggregate_piers",
    # --- slope / FEM ---
    ("fem2d", "slope_strength_reduction"): "fem2d_slope_srm",
    # Newmark sliding-block seismic-displacement names the agent guesses ON
    # slope_stability (the right module) — route to the real integrator so the
    # agent never concludes "no Newmark integrator exists" (eval NMK-2,
    # 2026-07-13). Cross-module guesses are handled by _CROSS_MODULE_REDIRECTS.
    ("slope_stability", "newmark"): "newmark_displacement",
    ("slope_stability", "newmark_analysis"): "newmark_displacement",
    ("slope_stability", "newmark_sliding_block"): "newmark_displacement",
    ("slope_stability", "sliding_block"): "newmark_displacement",
    ("slope_stability", "seismic_displacement"): "newmark_displacement",
    ("slope_stability", "infinite_slope_analysis"): "infinite_slope_fos",
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
    ("liquepy", "spt_bi2014_triggering"): "spt_liquefaction",
    ("salib", "sobol_sensitivity"): "sobol_sample",
    ("pystrata", "equivalent_linear"): "eql_site_response",
    ("gstools", "fit_variogram"): "variogram",
    # ordinary_kriging / discover_dxf — names guessed in the 2026-07-05 eval run.
    ("gstools", "ordinary_kriging"): "kriging",
    # GSTools' own class name for a conditioned field (an engineer's
    # notebook used it, 2026-10-15) and the plain-language guesses.
    ("gstools", "condsrf"): ("conditional_simulation",
                             {"simulation_method": "condsrf"}),
    ("gstools", "cond_srf"): ("conditional_simulation",
                              {"simulation_method": "condsrf"}),
    ("gstools", "conditioned_random_field"): "conditional_simulation",
    ("gstools", "conditional_random_field"): "conditional_simulation",
    ("gstools", "conditional_srf"): "conditional_simulation",
    ("dxf_import", "discover_dxf"): "discover_layers",
    # subsurface_characterization is the single data-I/O home; the former
    # pygef/ags4/pydiggs modules are folded in as format-adapter methods.
    ("subsurface", "read_and_validate"): "read_ags4",
    ("dxf_export", "export_cross_section"): "export_geometry_to_dxf",
    # --- figures ---
    # profile_figure has ONE method; the agent reaches for the module name or a
    # verb form when it wants a subsurface schematic.
    ("profile_figure", "profile_figure"): "subsurface_profile",
    ("profile_figure", "render_profile_figure"): "subsurface_profile",
    ("profile_figure", "subsurface_profile_figure"): "subsurface_profile",
    ("profile_figure", "plot_profile"): "subsurface_profile",
    ("profile_figure", "plot_subsurface_profile"): "subsurface_profile",
    ("profile_figure", "draw_profile"): "subsurface_profile",
    ("profile_figure", "soil_profile_figure"): "subsurface_profile",
    ("profile_figure", "profile_schematic"): "subsurface_profile",
    # generic data plot (owner feedback 2026-09-11) — verbs the model reaches for
    ("profile_figure", "plot"): "plot_data",
    ("profile_figure", "plot_xy"): "plot_data",
    ("profile_figure", "xy_plot"): "plot_data",
    ("profile_figure", "data_plot"): "plot_data",
    ("profile_figure", "line_plot"): "plot_data",
    ("profile_figure", "plot_series"): "plot_data",
    ("profile_figure", "plot_vs_depth"): "plot_data",
    # a package's figures on their own
    ("calc_package", "figures"): "render_figures",
    ("calc_package", "package_figures"): "render_figures",
    ("calc_package", "render_figure"): "render_figures",
    ("calc_package", "get_figures"): "render_figures",
    ("calc_package", "plot_results"): "render_figures",
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
    # Caquot-Kerisel log-spiral + generic passive/rankine coefficient guessed on
    # the WRONG module (eval EPC-2/EPC-3, 2026-07-13: the agent used
    # seismic_geotech Mononobe-Okabe with kh=0 for a STATIC Rankine/Caquot
    # question). earth_pressure_coefficient does rankine/coulomb/caquot_kerisel.
    "caquot": ("retaining_walls", "earth_pressure_coefficient"),
    "caquot_kerisel": ("retaining_walls", "earth_pressure_coefficient"),
    "caquot_kerisel_kp": ("retaining_walls", "earth_pressure_coefficient"),
    "log_spiral": ("retaining_walls", "earth_pressure_coefficient"),
    "log_spiral_passive": ("retaining_walls", "earth_pressure_coefficient"),
    "rankine": ("retaining_walls", "earth_pressure_coefficient"),
    "rankine_coefficient": ("retaining_walls", "earth_pressure_coefficient"),
    "passive_coefficient": ("retaining_walls", "earth_pressure_coefficient"),
    "passive_pressure_coefficient": ("retaining_walls", "earth_pressure_coefficient"),
    # Newmark sliding-block seismic displacement guessed on the WRONG module
    # (eval NMK-2, 2026-07-13: the agent concluded "no Newmark integrator
    # exists"). The rigorous integrator is slope_stability.newmark_displacement;
    # yield_acceleration and the Jibson-2007 regression are siblings there.
    "newmark": ("slope_stability", "newmark_displacement"),
    "newmark_analysis": ("slope_stability", "newmark_displacement"),
    "newmark_displacement": ("slope_stability", "newmark_displacement"),
    "newmark_sliding_block": ("slope_stability", "newmark_displacement"),
    "sliding_block": ("slope_stability", "newmark_displacement"),
    "yield_acceleration": ("slope_stability", "yield_acceleration"),
    # Apparent (Terzaghi-Peck) earth-pressure envelopes are an excavation-
    # support method; guessed on retaining_walls in the 2026-10 Foundry eval.
    "apparent_earth_pressure": ("soe", "apparent_pressure"),
    "apparent_pressure": ("soe", "apparent_pressure"),
    # Subsurface profile SCHEMATIC guessed on an analysis module (the module
    # whose layers are being drawn) instead of the figure module. Note
    # subsurface.plot_* are real data plots and are unaffected.
    "subsurface_profile": ("profile_figure", "subsurface_profile"),
    "subsurface_profile_figure": ("profile_figure", "subsurface_profile"),
    "profile_figure": ("profile_figure", "subsurface_profile"),
    "plot_subsurface_profile": ("profile_figure", "subsurface_profile"),
    "soil_profile_figure": ("profile_figure", "subsurface_profile"),
    "profile_schematic": ("profile_figure", "subsurface_profile"),
    "plot_data": ("profile_figure", "plot_data"),
    "data_plot": ("profile_figure", "plot_data"),
    "render_figures": ("calc_package", "render_figures"),
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


#: Names models give a reference module's TEXT tools, mapped to the method
#: each module actually has (live smoke wave 1, G13: 'search', 'text_search',
#: the category name 'Text Retrieval' and 'search_sections' on dm7 each cost
#: a failed call). The first name a module has wins.
_TEXT_SEARCH_GUESSES = frozenset({
    "search", "text_search", "search_text", "search_section", "section_search",
    "search_sections", "search_reference", "reference_search", "full_text_search",
    "keyword_search", "find", "find_section", "find_sections", "lookup", "query",
    "text_retrieval", "search_chapters", "search_references"})
_TEXT_SEARCH_METHODS = ("search_sections", "reference_search", "figure_search")
_SECTION_GET_GUESSES = frozenset({
    "retrieve", "get_section", "read_section", "fetch_section", "section",
    "retrieve_sections", "get_text", "retrieve_section", "reference_get",
    "section_text"})
_SECTION_GET_METHODS = ("retrieve_section", "reference_get", "figure_get")


def _norm_guess(method) -> str:
    return re.sub(r"[\s\-]+", "_", str(method or "").strip().lower())


def _text_tool_for(mod, method) -> Optional[str]:
    """The module's own text-search or section method a guessed text-tool
    name means, else ``None``."""
    g = _norm_guess(method)
    for guesses, methods in ((_TEXT_SEARCH_GUESSES, _TEXT_SEARCH_METHODS),
                             (_SECTION_GET_GUESSES, _SECTION_GET_METHODS)):
        if g in guesses:
            for real in methods:
                if real in mod.METHOD_REGISTRY:
                    return real
    return None


def _resolve_unknown_method(mod, agent_name: str, method: str, parameters: dict):
    """Resolve a guessed method via the curated alias map, or a guessed
    text-tool name to the module's own text tool.

    Returns ``(real_method, new_params)`` or ``None``.  Only curated (verified)
    aliases route automatically; selector-value guesses are surfaced as a
    directive error instead (see call_agent), never auto-routed.
    """
    entry = _METHOD_ALIASES.get((agent_name, method.strip().lower()))
    if entry is None:
        text_tool = _text_tool_for(mod, method)
        if text_tool is not None:
            return text_tool, dict(parameters)
        return None
    real, inject = (entry, {}) if isinstance(entry, str) else entry
    if real in mod.METHOD_REGISTRY:
        return real, {**parameters, **(inject or {})}
    return None


def _text_search_elsewhere(agent_name: str, method: str,
                           allowed_agents=None) -> Optional[dict]:
    """For a reference module with NO text tools (dm7, ufc_pavement...): the
    error that names ``reference_db.reference_search`` and the reference ids
    that module's text is indexed under, when the guess was a text tool."""
    g = _norm_guess(method)
    if g not in _TEXT_SEARCH_GUESSES | _SECTION_GET_GUESSES:
        return None
    if not _is_visible("reference_db", allowed_agents):
        return None
    ids: list = []
    try:
        from geotech_references._retrieval_db import list_indexed_references
        key = re.sub(r"[^a-z0-9]", "", agent_name.lower())
        for row in list_indexed_references():
            ref = str(row.get("reference") or "")
            if re.sub(r"[^a-z0-9]", "", ref.lower()).startswith(key):
                ids.append(ref)
    except Exception:                                  # noqa: BLE001
        ids = []
    which = (f" with reference={ids[0]!r}" + (f" (or {', '.join(map(repr, ids[1:]))})"
                                              if len(ids) > 1 else "")
             if ids else "")
    return {"error": (
        f"'{agent_name}' has no text-search tools (its methods are equations, "
        f"tables and charts). Its text is searched with call_agent("
        f"'reference_db', 'reference_search', {{'query': ...}}){which}; "
        "a section is fetched with reference_db.reference_get."),
        "redirect": {"agent_name": "reference_db",
                     "method": "reference_search",
                     **({"reference": ids} if ids else {})}}


#: Most method names an unknown-method error lists in full; past this the
#: closest names, the count and how to list them by category are given
#: (live smoke wave 1, G13: ~11 KB of dm7 names per miss).
_MAX_LISTED_METHODS = 40


def _available_hint(mod, agent_name: str, method: str) -> str:
    """The 'Available: ...' part of an unknown-method error, capped."""
    available = sorted(k for k, v in mod.METHOD_INFO.items()
                       if not v.get("alias_of"))
    if len(available) <= _MAX_LISTED_METHODS:
        return f"Available: {available}"
    cats = sorted({v.get("category", "General")
                   for v in mod.METHOD_INFO.values()
                   if not v.get("alias_of")})
    return (f"'{agent_name}' has {len(available)} methods: "
            f"list_methods('{agent_name}', category=<one of {cats}>) lists "
            "them by category")


#: Parameters call_agent itself reads before the method sees them.
_DISPATCH_PARAMS = ("attachment_key",)


def _takes_catch_all(func) -> bool:
    """Whether ``func`` takes ``**kwargs`` (so its signature refuses no
    name)."""
    try:
        return any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False


def _check_parameter_names(mod, method: str, parameters: dict) -> None:
    """Refuse a top-level parameter ``method`` does not take (BACKLOG
    GM-110): the ValueError names the unknown key(s), a did-you-mean and
    the valid parameters, as ``reject_unknown_params`` does inside the
    adapters. Without it a misspelt name ('phi_deg' for 'phi') was read by
    nobody and the run used the default. Nothing is renamed.

    The method's METHOD_INFO is the list: its ``parameters``, plus the
    other spellings it reads (``parameter_aliases``, alias -> parameter)
    and, with ``parameters_of`` naming one of its parameters, the
    parameters of the method that one names (``render_figures`` takes a
    package's inputs). A name in ``refused_parameters`` (name -> reason, or
    ``{"reason", "accepts": [values]}``) is one the method deliberately
    does not take; it is refused with that reason instead of a
    did-you-mean, unless its value is one the entry accepts as "not used".
    Not checked here: a method without documented
    parameters, one marked ``open_parameters``, and a reference wrapper
    (``func(**params)``) without a ``**kwargs`` catch-all, whose own
    signature refuses an unknown name. A wrapper WITH a catch-all is
    checked here, against its named parameters plus what the catch-all
    reads (``_reference_common.extract_method_info``).
    """
    info = mod.METHOD_INFO.get(method)
    if not isinstance(info, dict) or info.get("open_parameters"):
        return
    declared = info.get("parameters")
    if not isinstance(declared, dict):
        return
    wrapped = getattr(mod.METHOD_REGISTRY.get(method), "__wrapped__", None)
    if wrapped is not None and not _takes_catch_all(wrapped):
        return
    valid = set(declared)
    aliases = dict(info.get("parameter_aliases") or {})
    refused = dict(info.get("refused_parameters") or {})
    of = info.get("parameters_of")
    if of:
        names = [of] + [a for a, c in aliases.items() if c == of]
        target = next((parameters[n] for n in names
                       if isinstance(parameters.get(n), str)), None)
        other = mod.METHOD_INFO.get(target) if target else None
        # No such method: the handler says what to name. An open one
        # checks its own.
        if (not isinstance(other, dict) or target == method
                or other.get("open_parameters")):
            return
        valid |= set(other.get("parameters") or {})
        aliases.update(other.get("parameter_aliases") or {})
        refused = {**(other.get("refused_parameters") or {}), **refused}
    from funhouse_agent.adapters import (
        accepted_as_unused, refused_reason, reject_unknown_params,
    )
    # A name the method deliberately does not take is refused with its
    # reason, never with a near-miss guess at a different quantity; a value
    # its entry accepts as "not used" (a slope of 0) passes.
    hits = sorted(k for k, v in parameters.items() if k in refused
                  and not accepted_as_unused(refused[k], v))
    rest = {k: v for k, v in parameters.items() if k not in refused}
    try:
        reject_unknown_params(rest, valid, method=method,
                              aliases=tuple(aliases) + _DISPATCH_PARAMS)
    except ValueError as exc:
        if not hits:
            raise
        also = f" Also: {exc}"
    else:
        also = ""
    if hits:
        raise ValueError(
            f"{method} does not take {hits}: "
            + "; ".join(f"{k}: {refused_reason(refused[k])}" for k in hits)
            + "." + also)


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
        and injected as ``content`` before calling the adapter -- or, for a
        method that reads a file, or a key that only names a file in the
        working folder, the file is passed as ``file_path``.

    Returns
    -------
    dict
        Calculation results or {"error": "..."}. With a host working folder
        bound, every path in the result that lies inside it is given
        relative to it (``_fileio.hide_working_folder``): the model names
        files the way the file tools resolve them, and never repeats a
        server path to the user.
    """
    agent_name = _canonical_agent_name(agent_name)
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
            avail = _available_hint(mod, agent_name, method)
            redirect = _cross_module_redirect(agent_name, method, allowed_agents)
            if redirect is not None:
                right_agent, right_method = redirect
                return {"error":
                        f"'{method}' is not a '{agent_name}' method — it lives "
                        f"on module '{right_agent}' as '{right_method}'. Call "
                        f"call_agent('{right_agent}', '{right_method}', {{...}}). "
                        f"{avail}"}
            elsewhere = _text_search_elsewhere(agent_name, method,
                                               allowed_agents)
            if elsewhere is not None:
                return elsewhere
            cands = _selector_value_candidates(mod, method)
            if cands:
                opts = ", ".join(f"{m}({p}='{method}')" for m, p in cands)
                return {"error": f"'{method}' is a value for a selector "
                                 f"parameter, not a method name — call: {opts}. "
                                 f"{avail}"}
            near = _closest_methods(mod, method)
            return {"error": f"Unknown method '{method}'. Did you mean: "
                             f"{near}? {avail}"}
    from funhouse_agent._fileio import hide_working_folder
    if not isinstance(parameters, dict):
        parameters = {}
    given = set(parameters)
    try:
        _check_parameter_names(mod, method, parameters)
        if "attachment_key" in parameters:
            parameters = _resolve_attachment(
                parameters, attachments, _method_params(mod, method))
        parameters = _inputs_from_working_folder(parameters)
        parameters, named = _outputs_into_working_folder(parameters)
        # The names this call gave, so a module's choices say "given" for a
        # typed value equal to its default (GM-118).
        given |= set(parameters)
        from geotech_common.choices import call_inputs
        try:
            with call_inputs(_given_names(mod, method, given),
                             _left_out(mod, method, given,
                                       parameters)) as inputs:
                result = mod.METHOD_REGISTRY[method](parameters)
        except TypeError as e:
            if not _SIGNATURE_ERROR.search(str(e)):
                raise
            return hide_working_folder(
                _parameter_error(mod, agent_name, method, parameters, e))
        if named and isinstance(result, dict) and "error" not in result:
            result = dict(result)
            result["output_note"] = _output_note(named, parameters)
        result = _citations_checked(agent_name, method, parameters, result)
        result = _with_defaults_used(mod, method, given, result, parameters,
                                     inputs.reported)
        return hide_working_folder(result)
    except Exception as e:
        return hide_working_folder({"error": f"{type(e).__name__}: {e}"})


#: A calc package's header and file format: not engineering choices, so
#: never listed in ``defaults_used``.
_PACKAGE_HEADER_PARAMS = frozenset({
    "project_name", "project_number", "engineer", "checker", "company",
    "date", "format"})


def _declared(mod, method: str):
    """(declared parameters, parameter_aliases) of ``method``, or
    ``({}, {})``."""
    info = mod.METHOD_INFO.get(method)
    declared = info.get("parameters") if isinstance(info, dict) else None
    if not isinstance(declared, dict):
        return {}, {}
    return declared, info.get("parameter_aliases") or {}


def _given_names(mod, method: str, given) -> set:
    """The parameter names a call gave, a name given under another spelling
    the method reads (``parameter_aliases``) under its own name too."""
    _, aliases = _declared(mod, method)
    return set(given) | {aliases[k] for k in given if k in aliases}


def _fold(value):
    """``value`` compared without case when it is a string."""
    return value.lower() if isinstance(value, str) else value


def default_in_force(spec: dict, values: dict):
    """The default one declared parameter takes in a call: its
    ``default``, or, where the default depends on another input, the one
    its ``default_depends_on`` gives for that input's value in force
    (``values``; BACKLOG GM-158: Terzaghi's N-gamma defaults to
    Kumbhojkar, every other factor method's to Vesic).
    ``default_depends_on`` is ``{"parameter": other input, "values":
    {its value: default}, "otherwise": default}``, strings compared
    without case. A parameter whose ``default_from`` names another input
    (GM-165: a foundation friction angle that defaults to the backfill's)
    takes that input's value in force."""
    source = spec.get("default_from")
    if source:
        return values.get(source)
    depends = spec.get("default_depends_on")
    if not depends:
        return spec.get("default")
    other = _fold(values.get(depends["parameter"]))
    for value, default in depends["values"].items():
        if _fold(value) == other:
            return default
    return depends["otherwise"]


def case_of(spec, declared=None):
    """The ``when_absent`` case ({case, basis}) a left-out input stands
    for (GM-165), else None. A parameter that takes another's value
    (``default_from``) takes that one's case when it is left out too,
    followed through ``declared`` (the method's parameters): a passive-side
    water table that defaults to the active side's, with neither given, is
    "no water table"."""
    seen = set()
    while isinstance(spec, dict):
        if spec.get("when_absent"):
            return spec["when_absent"]
        source = spec.get("default_from")
        if not source or declared is None or source in seen:
            return None
        seen.add(source)
        spec = declared.get(source)
    return None


def absent_case(spec, declared=None) -> str:
    """What a left-out input with no default stands for in the run, as
    ``defaults_used`` gives it (GM-165): "<case>: <basis>" from the
    parameter's ``when_absent`` (:func:`case_of`); "" when it declares
    none."""
    case = case_of(spec, declared)
    if not case:
        return ""
    return f"{case['case']}: {case['basis']}"


def _values_in_force(declared: dict, aliases: dict, parameters=None) -> dict:
    """Each declared parameter's value in force in a call: the call's
    (under the name the method reads), else its default, a default that
    depends on another input (:func:`default_in_force`) taken at that
    input's value in force."""
    values = {name: spec.get("default") for name, spec in declared.items()
              if isinstance(spec, dict)}
    named = set()
    for key, value in (parameters or {}).items():
        values[aliases.get(key, key)] = value
        named.add(aliases.get(key, key))
    for name, spec in declared.items():
        if isinstance(spec, dict) and spec.get("default_depends_on") \
                and name not in named:
            values[name] = default_in_force(spec, values)
    for name, spec in declared.items():
        if isinstance(spec, dict) and spec.get("default_from") \
                and name not in named:
            values[name] = default_in_force(spec, values)
    return values


def _left_out(mod, method: str, given, parameters=None) -> dict:
    """The declared defaults of ``method`` that a call left in force: name
    -> default, a default that depends on another input taken at the
    call's value of it (``parameters``), one taken from another input
    (``default_from``) at that input's value, and a left-out input with no
    default as the case it stands for (``when_absent``,
    :func:`absent_case`; GM-165). Left out: any other declared default of
    None (worked out by the method) or "" (an empty field), and a calc
    package's header and format."""
    declared, aliases = _declared(mod, method)
    named = _given_names(mod, method, given)
    if str(getattr(mod, "__name__", "")).endswith("calc_package"):
        named |= _PACKAGE_HEADER_PARAMS
    values = _values_in_force(declared, aliases, parameters)
    out = {}
    for name, spec in declared.items():
        if not isinstance(spec, dict) or name in named:
            continue
        value = values.get(name)
        if value is None:
            value = absent_case(spec, declared) or None
        if value is not None and value != "":
            out[name] = value
    return out


#: What a parameter's ``applies_when`` may ask of another parameter (GM-119).
_SET, _UNSET = "set", "unset"


def _is_set(value) -> bool:
    """A value in force that does something: not None, 0, False or empty."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    if isinstance(value, (str, list, tuple, dict)) and not value:
        return False
    return True


def applies(condition, values: dict) -> bool:
    """Whether a parameter's default applies to a call (GM-119).

    ``condition`` is the parameter's ``applies_when``: absent (None) or
    ``{}``, always; False, never (the method takes the parameter and does
    not use it); a list of such dicts, any one of them; else a dict, every
    entry of which must hold, of another parameter's name to a list of
    values its value in force is one of (strings compared without case;
    None for "not given" where the parameter has no default), "set" (not
    None, 0, False or empty) or "unset". ``values`` maps each parameter to
    its value in force: the call's, else its declared default."""
    if condition is None:
        return True
    if condition is False:
        return False
    if isinstance(condition, list):
        return any(applies(c, values) for c in condition)
    for name, want in condition.items():
        value = values.get(name)
        if want == _SET:
            ok = _is_set(value)
        elif want == _UNSET:
            ok = not _is_set(value)
        else:
            ok = any(_fold(value) == _fold(w) for w in want)
        if not ok:
            return False
    return True


def _defaults_used(mod, method: str, given, parameters=None,
                   reported=()) -> dict:
    """The declared defaults a call left in force that apply to it, with
    the default each took (BACKLOG GM-115, GM-119): name -> default, from
    the method's METHOD_INFO ``parameters`` (GM-110), a default that
    depends on another input at the call's value of it (GM-158), and a
    left-out input with no default as the case it stands for or the other
    input's value it takes (GM-165, :func:`_left_out`). A
    parameter given under another spelling it reads
    (``parameter_aliases``) counts as given. A parameter whose
    ``applies_when`` (:func:`applies`) does not hold for this call is left
    out, unless the module said it used the default (``reported``, the
    names its ``choices`` called "default"), so ``defaults_used`` and
    ``choices`` never contradict each other."""
    left = _left_out(mod, method, given, parameters)
    if not left:
        return {}
    declared, aliases = _declared(mod, method)
    values = _values_in_force(declared, aliases, parameters)
    return {name: default for name, default in left.items()
            if name in reported
            or applies(declared[name].get("applies_when"), values)}


def _with_defaults_used(mod, method: str, given, result, parameters=None,
                        reported=()):
    """``result`` with ``defaults_used`` (:func:`_defaults_used`) added when
    the call left a declared default in force that applies to it: the agent
    then knows which values it never chose. A result that is not a dict, is
    an error, or already says what it used is returned unchanged."""
    if not isinstance(result, dict) or "error" in result \
            or "defaults_used" in result:
        return result
    try:
        used = _defaults_used(mod, method, given, parameters, reported)
    except Exception:  # noqa: BLE001 - never the method's result
        return result
    if not used:
        return result
    return {**result, "defaults_used": used}


def _citations_checked(agent_name: str, method: str, parameters: dict,
                       result):
    """A calc package's result after the check of the documents cited in
    the text the model GAVE it (``html_to_pdf``'s HTML, a ``*_package``'s
    own words), never the package's template: with
    ``GEOTECH_CITED_DOCUMENTS`` on, a citation nothing in the conversation
    looked up adds ``cited_not_looked_up`` and a ``citation_note``; the
    file is written either way (:mod:`funhouse_agent.cited_documents`,
    BACKLOG EV-21)."""
    if agent_name != "calc_package" or not (
            method == "html_to_pdf" or str(method).endswith("_package")):
        return result
    try:
        from funhouse_agent import cited_documents
        return cited_documents.add_to_result(
            result, cited_documents.deliverable_texts(parameters),
            where=f"calc_package.{method}")
    except Exception:  # noqa: BLE001 - never the writer's result
        return result


#: Parameters naming a file (``False``) or a folder of files (``True``) that a
#: module method WRITES. Every writer reached through ``call_agent`` takes one
#: of these names (``write_diggs``, the calc packages, ``html_to_pdf``,
#: ``render_figures``, the DXF export, ``snip_region``, the plots).
_OUTPUT_PARAMS = (("output_path", False), ("output_dir", True))


def _outputs_into_working_folder(parameters):
    """``(parameters, named)``: with a host working folder set, every output
    path the model gave is moved INTO that folder (file name kept,
    :func:`funhouse_agent._fileio.into_working_folder`) — the same folder
    ``write_docx``, ``save_file`` and ``annotate_document`` write to — so the
    file reaches the user whatever directory was named. ``named`` maps each
    output parameter given to what the model asked for. With no host folder
    nothing changes (a library caller keeps today's behaviour)."""
    from funhouse_agent._fileio import host_output_dir, into_working_folder
    folder = host_output_dir()
    if not folder or not isinstance(parameters, dict):
        return parameters, {}
    out, named = parameters, {}
    for key, is_dir in _OUTPUT_PARAMS:
        asked = parameters.get(key)
        if not isinstance(asked, str) or not asked.strip():
            continue
        named[key] = asked
        target = into_working_folder(asked, folder, is_dir=is_dir)
        if target != asked:
            if out is parameters:
                out = dict(parameters)
            out[key] = target
    return out, named


def _output_note(named: dict, parameters: dict) -> str:
    """What a writer's result says about where its file went."""
    from funhouse_agent._fileio import host_output_dir
    folder = host_output_dir() or ""
    moved = []
    for key, asked in named.items():
        went = str(parameters.get(key) or "")
        # Where the path would have gone as given (a bare name: into the
        # folder anyway; "/tmp/x" or an absolute path: somewhere else).
        meant = os.path.abspath(os.path.join(
            folder, os.path.expanduser(str(asked).strip())))
        if os.path.normcase(meant) != os.path.normcase(os.path.abspath(went)):
            moved.append(f"'{asked}' -> '{went}'")
    note = ("Written into this conversation's working folder, which is where "
            "the user receives files: it is attached to the conversation. "
            "Tell the user the file name; a /tmp path or a sandbox: link "
            "would not reach them.")
    if moved:
        note += (" The directory asked for was replaced by that folder (a "
                 "tool always writes there): " + "; ".join(moved) + ".")
    return note


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
