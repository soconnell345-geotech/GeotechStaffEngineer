"""Shared helpers for geotech-references adapter modules.

All 14 reference agents follow the same pattern: inspect-discovered functions
from geotech_references subpackages, with optional text retrieval.  This module
provides a factory so each adapter is ~15 lines.
"""

import ast
import inspect
import json
import re
import textwrap
import typing

import numpy as np

from funhouse_agent.adapters import clean_value


# ---------------------------------------------------------------------------
# Introspection helpers (mirrors geotech-references/agents helper functions)
# ---------------------------------------------------------------------------

def has_callable_param(func) -> bool:
    """Return True if any parameter has a Callable type annotation."""
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        ann = p.annotation
        if ann is inspect.Parameter.empty:
            continue
        if "Callable" in str(ann):
            return True
    return False


def param_type_str(annotation) -> str:
    """Convert a type annotation to a human-readable string."""
    if annotation is inspect.Parameter.empty:
        return "float"
    _MAP = {float: "float", int: "int", bool: "bool", str: "str"}
    if annotation in _MAP:
        return _MAP[annotation]
    s = str(annotation)
    for pattern, result in [
        ("List[float]", "array of float"),
        ("Sequence[float]", "array of float"),
        ("List[Tuple", "array of tuples"),
        ("Tuple", "tuple"),
    ]:
        if pattern in s:
            return result
    if "list" in s.lower():
        return "list"
    if "dict" in s.lower():
        return "dict"
    return s


def catch_all_is_read(func) -> bool:
    """Whether ``func``'s ``**kwargs`` catch-all is read in its body.

    A catch-all its body never reads takes nothing, so the named parameters
    are the whole list and dispatch can refuse any other name (BACKLOG
    GM-110: DM7's ``plot_figure_5_16(tv=0.5)`` drew no query point and
    said nothing). One that is read, or whose source cannot be read, may
    take anything. False for a function with no catch-all.
    """
    name = next((p.name for p in inspect.signature(func).parameters.values()
                 if p.kind is inspect.Parameter.VAR_KEYWORD), None)
    if name is None:
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        body = tree.body[0].body
    except (OSError, TypeError, SyntaxError, IndexError, AttributeError):
        return True
    return any(isinstance(node, ast.Name) and node.id == name
               for stmt in body for node in ast.walk(stmt))


def extract_method_info(func, category: str, reference: str) -> dict:
    """Build a METHOD_INFO entry from a function's signature + docstring.

    A function whose ``**kwargs`` catch-all is read is marked
    ``open_parameters`` (dispatch cannot tell what it takes) until its
    adapter declares what it reads; one whose catch-all is never read is
    checked by dispatch against its named parameters (GM-110)."""
    doc = inspect.getdoc(func) or ""

    # Brief: first paragraph
    desc_lines = []
    for line in doc.split("\n"):
        if line.strip() == "":
            break
        desc_lines.append(line.strip())
    brief = " ".join(desc_lines) if desc_lines else "No description available."

    # Parse numpy-style docstring for parameter descriptions
    param_descs: dict[str, str] = {}
    in_params = False
    current_param = None
    for dline in doc.split("\n"):
        stripped = dline.strip()
        if stripped.lower() in ("parameters", "parameters:", "args", "args:"):
            in_params = True
            continue
        if stripped.startswith("---"):
            continue
        if stripped.lower() in (
            "returns", "returns:", "raises", "raises:",
            "examples", "examples:", "notes", "notes:",
            "references", "references:",
        ):
            in_params = False
            current_param = None
            continue
        if in_params:
            if " : " in stripped:
                current_param = stripped.split(" : ")[0].strip()
                param_descs[current_param] = ""
            elif current_param and stripped:
                prev = param_descs[current_param]
                param_descs[current_param] = (prev + " " + stripped) if prev else stripped

    sig = inspect.signature(func)
    params = {}
    for pname, p in sig.parameters.items():
        # Skip *args / **kwargs catch-alls: they are not real parameters and
        # must never be advertised (a VAR_KEYWORD has no default, so it would
        # otherwise be reported as a required parameter).
        if p.kind in (inspect.Parameter.VAR_KEYWORD,
                      inspect.Parameter.VAR_POSITIONAL):
            continue
        pinfo: dict = {
            "type": param_type_str(p.annotation),
            "required": p.default is inspect.Parameter.empty,
        }
        if p.default is not inspect.Parameter.empty:
            pinfo["default"] = p.default
        if pname in param_descs and param_descs[pname]:
            pinfo["description"] = param_descs[pname]
        params[pname] = pinfo

    info: dict = {
        "category": category,
        "brief": brief,
        "reference": reference,
        "parameters": params,
    }
    ret_ann = sig.return_annotation
    if ret_ann is not inspect.Parameter.empty:
        info["returns"] = param_type_str(ret_ann)
    if catch_all_is_read(func):
        info["open_parameters"] = True
    return info


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def clean_ref_result(result):
    """Convert a reference function result to a JSON-safe dict."""
    if isinstance(result, dict):
        return {k: clean_value(v) for k, v in result.items()}
    if isinstance(result, np.ndarray):
        return {"result": result.tolist()}
    if isinstance(result, (list, tuple)):
        return {"result": [clean_value(v) for v in result]}
    return {"result": clean_value(result)}


def make_wrapper(func):
    """Create adapter wrapper: ``params`` dict → ``func(**params)`` → clean dict."""
    def wrapper(params):
        return clean_ref_result(func(**params))
    wrapper.__wrapped__ = func
    return wrapper


# ---------------------------------------------------------------------------
# Semantic aliases for document-id-prefixed methods
# ---------------------------------------------------------------------------
#
# Reference lookups are named after the source artifact (``table_5_3_alpha_bond``,
# ``figure_10_6_alpha_clay``, ``equation_10_21_rock_socket_side``). An LLM can't
# know the table/figure number, so it either guesses a clean semantic name (which
# doesn't match) or burns rounds calling ``list_methods``. We register guessable
# aliases so the name an engineer/model reaches for resolves directly.

# Tokens that carry no semantic meaning (table/figure numbers, appendix ids like
# ``4a`` / ``a1``, and ``to`` in multi-figure ranges like ``7_10_to_13``).
_ID_TOKEN = re.compile(r"^(\d+[a-z]?|[a-z]\d+|to)$")
_ALIAS_PREFIXES = ("plot_figure", "table", "figure", "equation")

# Curated aliases for high-value lookups where the mechanically-stripped name is
# still not what a model reaches for. canonical_name -> [extra aliases].
_CURATED_ALIASES = {
    # micropile
    "table_5_3_alpha_bond": ["grout_bond_strength"],
    "table_5_9_soil_modulus_k_sand": ["py_soil_modulus_sand"],
    "table_5_10_soil_modulus_k_clay": ["py_soil_modulus_clay"],
    "table_5_4_group_efficiency": ["group_efficiency_factor"],
    # gec13 (ground modification)
    "table_10_1_micropile_bond_stress": ["micropile_grout_bond_strength",
                                         "grout_bond_strength"],
    "table_9_2_nail_bond_strength": ["soil_nail_bond_strength"],
    "figure_5_5_settlement_improvement": ["settlement_improvement_factor"],
    # gec10 (drilled shafts)
    "figure_10_6_alpha_clay": ["alpha_adhesion_factor"],
    "table_10_2_nc_base_clay": ["bearing_capacity_factor_nc"],
    # gec12 (driven piles / Nordlund)
    "figure_7_18_adhesion_factor": ["adhesion_factor_alpha"],
    "figure_7_16b_nq": ["nordlund_nq_factor"],
    "figure_7_10_to_13_kd": ["nordlund_kd"],
    "table_7_1_resistance_factor_static": ["phi_static_analysis"],
    "table_7_2_resistance_factor_field": ["phi_field_method"],
    # gec11 (MSE walls)
    "table_4_7_internal_resistance": ["internal_stability_resistance_factors"],
    "table_4_5_external_resistance": ["external_stability_resistance_factors"],
    # gec7 (soil nails)
    "table_5_resistance_factors": ["soil_nail_resistance_factors",
                                   "lrfd_resistance_factors"],
    "table_4_4a_bond_strength_coarse": ["soil_nail_bond_strength_coarse"],
    "table_4_4b_bond_strength_fine": ["soil_nail_bond_strength_fine"],
    "table_4_5_bond_strength_rock": ["soil_nail_bond_strength_rock"],
    # gec4 (ground anchors)
    "table_7_bond_stress_cohesive": ["anchor_bond_stress_cohesive"],
    "table_7_bond_stress_cohesionless": ["anchor_bond_stress_cohesionless"],
    "table_7_bond_stress_rock": ["anchor_bond_stress_rock"],
    # ufc_pavement
    "table_10_1_k_subgrade": ["modulus_subgrade_reaction"],
    "table_7_1_base_design_cbr": ["base_course_design_cbr"],
    # fhwa_pavements (Geotechnical Aspects of Pavements, FHWA-NHI-05-037)
    "table_5_35_default_resilient_modulus": ["default_resilient_modulus",
                                             "typical_resilient_modulus",
                                             "resilient_modulus_by_soil_class"],
    "table_5_34_resilient_modulus_correlations": ["resilient_modulus_correlations"],
    "table_5_28_typical_cbr": ["typical_cbr_values", "cbr_by_soil_class"],
    "table_4_14_soil_as_pavement_material": ["soil_suitability_as_subgrade",
                                             "soil_as_pavement_material"],
    "table_7_12_frost_susceptibility": ["frost_susceptibility_classification"],
    "table_5_49_drainage_modifier_mi": ["drainage_modifier_flexible"],
    "table_5_50_drainage_coefficient_cd": ["drainage_coefficient_rigid"],
    "table_5_56_permeability_soils": ["soil_permeability",
                                      "hydraulic_conductivity_soils"],
    "table_5_57_permeability_highway_materials": ["highway_material_permeability"],
    "table_7_17_swell_potential_ll_pi": ["swell_potential_from_atterberg"],
    "table_5_18_compaction_aashto": ["compaction_by_aashto_class"],
}


def _mechanical_alias(name: str) -> str:
    """Strip a leading table_/figure_/equation_ document-number prefix.

    ``table_5_3_alpha_bond`` -> ``alpha_bond``; ``figure_7_10_to_13_kd`` -> ``kd``.
    Returns '' if the name has no such prefix or nothing semantic remains.
    """
    for pre in _ALIAS_PREFIXES:
        if name.startswith(pre + "_"):
            tokens = name[len(pre) + 1:].split("_")
            i = 0
            while i < len(tokens) and _ID_TOKEN.match(tokens[i]):
                i += 1
            return "_".join(tokens[i:])
    return ""


def register_semantic_aliases(registry: dict, info: dict) -> tuple:
    """Add guessable semantic aliases for document-id-prefixed methods.

    Each alias points at the same callable and METHOD_INFO as its canonical
    method but carries an ``alias_of`` marker, so it is callable/describable yet
    hidden from ``list_methods`` (which keeps the listing canonical). An alias is
    added only if it does not collide with an existing method or earlier alias.
    """
    def _add(alias: str, canonical: str) -> None:
        if not alias or alias in registry:
            return
        registry[alias] = registry[canonical]
        ai = dict(info[canonical])
        ai["alias_of"] = canonical
        ai["brief"] = f"{ai.get('brief', '').rstrip('.')} (alias of {canonical})."
        info[alias] = ai

    for name in list(registry):
        if info.get(name, {}).get("alias_of"):
            continue  # never alias an alias
        _add(_mechanical_alias(name), name)
        for curated in _CURATED_ALIASES.get(name, []):
            _add(curated, name)
    return registry, info


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_lookup_registry(lookup_modules, *, skip_callable=False):
    """Scan modules for public functions and build registries.

    Parameters
    ----------
    lookup_modules : list of (module, category_str, reference_str)
    skip_callable : bool
        If True, skip functions with Callable-typed parameters (DM7).

    Returns
    -------
    method_registry : dict
        method_name → callable(params_dict) → dict
    method_info : dict
        method_name → {category, brief, reference, parameters, …}
    """
    registry: dict = {}
    info: dict = {}
    for mod, category, reference in lookup_modules:
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("_"):
                continue
            if skip_callable and has_callable_param(func):
                continue
            registry[name] = make_wrapper(func)
            info[name] = extract_method_info(func, category, reference)
    register_semantic_aliases(registry, info)
    return registry, info


#: What a reference tool's result is shaped to fit, in JSON characters
#: (BACKLOG REF-26): under the deep agent's 16,000-character cap on a
#: reference read (``deep.tools.DEFAULT_REFERENCE_RESULT_CHARS``), with room
#: for what dispatch adds (``defaults_used``), so a result is never cut
#: mid-JSON. A listing, a search or a chapter too long for it comes back in
#: a shorter shape that says how to get the rest.
RESULT_BUDGET_CHARS = 14000

#: Hits search_sections returns when the call names no limit.
SEARCH_SECTIONS_DEFAULT_LIMIT = 10


def takes(func, name: str) -> bool:
    """Whether ``func`` takes a parameter ``name``: a geotech-references
    older than a feature lacks it, and the adapters pass it only where it
    is taken."""
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def as_result(shaped: dict, key: str) -> dict:
    """A library's shaped dict (``{"hits"|"rows": [...], "shown": ...}``)
    as the tool result the agent has always had (``{"result": [...]}``),
    with the counts and the note beside it."""
    out = {"result": shaped.get(key, [])}
    out.update({k: v for k, v in shaped.items() if k != key})
    return out


def example_section_ids(ref_package_name: str, n: int = 3) -> list[str]:
    """Up to ``n`` section numbers this reference holds, one per level from
    the top (``['1.1', '4.2.3.1', '7.2.1.3.1']`` for GEC-12), for the
    ``retrieve_section`` description (BACKLOG REF-01: a fixed example,
    '5.7.2', was in no reference at all). Empty when the reference's text
    cannot be read."""
    try:
        from geotech_references import _retrieval
        chapters = _retrieval.list_chapters(ref_package_name)
    except Exception:  # noqa: BLE001 — an example is never worth a failure
        return []
    by_depth: dict[int, str] = {}
    for chapter in chapters:
        for sec in chapter.get("sections") or []:
            sid = str(sec.get("section_id") or "")
            # a combined section ('5.15-5.16') is not an example of a number
            if sid and not re.search(r"\.\d+-", sid):
                by_depth.setdefault(len(re.findall(r"[.-]", sid)), sid)
    depths = sorted(by_depth)
    if len(depths) > n:
        # the top level, the deepest, and evenly between
        step = (len(depths) - 1) / max(n - 1, 1)
        depths = [depths[round(i * step)] for i in range(n)]
    return [by_depth[d] for d in depths]


def _section_id_description(ref_package_name: str) -> str:
    examples = example_section_ids(ref_package_name)
    if not examples:
        return "Section identifier at any level, as the reference prints it."
    shown = ", ".join(f"'{sid}'" for sid in examples)
    return f"Section identifier at any level (e.g. {shown})."


def add_text_retrieval(registry, info, ref_package_name, reference_str):
    """Add retrieve_section / search_sections / list_chapters / load_chapter."""
    from geotech_references import _retrieval

    # geotech-references before REF-24 has no include_subsections (and
    # always inlined a section's subsections); pass it only where taken.
    takes_include = takes(_retrieval.retrieve_section, "include_subsections")
    # ... and before REF-26 no shaped results (max_chars, chapter, offset).
    shaped = takes(_retrieval.search_sections, "max_chars")

    def _retrieve(params):
        kwargs = {}
        if takes_include and params.get("include_subsections"):
            kwargs["include_subsections"] = True
        return _retrieval.retrieve_section(
            ref_package_name, params["section_id"], **kwargs)

    def _search(params):
        limit = int(params.get("limit") or SEARCH_SECTIONS_DEFAULT_LIMIT)
        limit = max(1, min(limit, 50))
        offset = max(0, int(params.get("offset") or 0))
        if shaped:
            return as_result(_retrieval.search_sections(
                ref_package_name, params["query"], limit=limit,
                offset=offset, max_chars=RESULT_BUDGET_CHARS), "hits")
        hits = _retrieval.search_sections(ref_package_name, params["query"])
        return {"result": hits[offset:offset + limit]}

    def _list_ch(params):
        chapter = params.get("chapter")
        chapter = None if chapter in (None, "") else int(chapter)
        if chapter is not None:
            if shaped:
                return {"result": _retrieval.list_chapters(
                    ref_package_name, chapter=chapter,
                    max_chars=RESULT_BUDGET_CHARS)}
            chapters = [c for c in _retrieval.list_chapters(ref_package_name)
                        if c.get("chapter") == chapter]
            if not chapters:
                raise KeyError(f"Chapter {chapter} not found")
            return {"result": chapters}
        # Every chapter with its sections when that fits one result (22 of
        # 27 references); otherwise the chapters alone, and how to get one
        # chapter's sections.
        full = _retrieval.list_chapters(ref_package_name)
        size = len(json.dumps({"result": full}, default=str))
        if size <= RESULT_BUDGET_CHARS:
            return {"result": full}
        chapters = [dict({k: v for k, v in c.items()
                          if k not in ("sections", "n_sections", "note")},
                         n_sections=len(c.get("sections") or []))
                    for c in full]
        return {"result": chapters, "note": (
            "Chapters with their titles and how many sections each holds: "
            f"with every section the listing is {size:,} characters, over "
            f"the {RESULT_BUDGET_CHARS:,}-character limit. A chapter's "
            "sections (number and title): list_chapters with "
            "chapter=<number>; a section's text: retrieve_section.")}

    def _load_ch(params):
        if shaped:
            return _retrieval.load_chapter(
                ref_package_name, int(params["chapter"]),
                max_chars=RESULT_BUDGET_CHARS)
        return _retrieval.load_chapter(ref_package_name,
                                       int(params["chapter"]))

    _TEXT = {
        "retrieve_section": (_retrieve, {
            "category": "Text Retrieval",
            "brief": ("Retrieve one section by its ID, at any level: its own "
                      "text, 'under' (the sections above it) and 'children' "
                      "(ID and title of each subsection, fetched by its own "
                      "ID). A missing ID's error names the nearest section "
                      "above it and the sections numbered under it."),
            "reference": reference_str,
            "parameters": {
                "section_id": {
                    "type": "str", "required": True,
                    "description": _section_id_description(
                        ref_package_name),
                },
                "include_subsections": {
                    "type": "bool", "required": False, "default": False,
                    "description": ("Inline the subsections' text too, when "
                                    "the whole fits in 12,000 characters; "
                                    "otherwise they stay listed in "
                                    "'children' with a note."),
                },
            },
            "returns": "dict",
        }),
        "search_sections": (_search, {
            "category": "Text Retrieval",
            "brief": ("Keyword search across every section and subsection, "
                      "best first: each hit is a section's number, title, "
                      "chapter, 'under' (the sections above it) and a "
                      "snippet of its text, never the whole text "
                      "(retrieve_section fetches that). As many hits as "
                      "fit; 'next_offset' and a note say how to get more."),
            "reference": reference_str,
            "parameters": {
                "query": {
                    "type": "str", "required": True,
                    "description": "Search query (case-insensitive, multi-word AND).",
                },
                "limit": {
                    "type": "int", "required": False,
                    "default": SEARCH_SECTIONS_DEFAULT_LIMIT,
                    "description": ("Most hits to return (at most 50; "
                                    "fewer when they would not fit)."),
                },
                "offset": {
                    "type": "int", "required": False, "default": 0,
                    "description": ("Skip this many of the best hits: the "
                                    "'next_offset' of the last call."),
                },
            },
            "returns": "dict",
        }),
        "list_chapters": (_list_ch, {
            "category": "Text Retrieval",
            "brief": ("List chapters and their sections by ID and title, "
                      "subsections included (a nested one carries "
                      "parent_id). Without 'chapter': every chapter, with "
                      "its sections when the whole listing fits one result "
                      "(otherwise titles and section counts, and a note). "
                      "With 'chapter': that chapter's sections."),
            "reference": reference_str,
            "parameters": {
                "chapter": {
                    "type": "int", "required": False,
                    "description": ("Chapter number: list its sections. "
                                    "Leave out for every chapter."),
                },
            },
            "returns": "dict",
        }),
        "load_chapter": (_load_ch, {
            "category": "Text Retrieval",
            "brief": ("Load a whole chapter: every section's text. A chapter "
                      "too long for one result (most are) comes back as its "
                      "list of sections with a note: fetch the ones you "
                      "need with retrieve_section."),
            "reference": reference_str,
            "parameters": {
                "chapter": {
                    "type": "int", "required": True,
                    "description": "Chapter number.",
                },
            },
            "returns": "dict",
        }),
    }
    for name, (func, method_info) in _TEXT.items():
        registry[name] = func
        info[name] = method_info
