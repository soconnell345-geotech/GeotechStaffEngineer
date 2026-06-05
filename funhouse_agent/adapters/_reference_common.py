"""Shared helpers for geotech-references adapter modules.

All 14 reference agents follow the same pattern: inspect-discovered functions
from geotech_references subpackages, with optional text retrieval.  This module
provides a factory so each adapter is ~15 lines.
"""

import inspect
import re
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


def extract_method_info(func, category: str, reference: str) -> dict:
    """Build a METHOD_INFO entry from a function's signature + docstring."""
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
    # fema_p2192
    "table_11_6_1_sdc_short_period": ["sdc_from_sds"],
    "table_11_6_2_sdc_one_second": ["sdc_from_sd1"],
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


def add_text_retrieval(registry, info, ref_package_name, reference_str):
    """Add retrieve_section / search_sections / list_chapters / load_chapter."""
    from geotech_references import _retrieval

    def _retrieve(params):
        return _retrieval.retrieve_section(ref_package_name, params["section_id"])

    def _search(params):
        return {"result": _retrieval.search_sections(ref_package_name, params["query"])}

    def _list_ch(params):
        return {"result": _retrieval.list_chapters(ref_package_name)}

    def _load_ch(params):
        return _retrieval.load_chapter(ref_package_name, params["chapter"])

    _TEXT = {
        "retrieve_section": (_retrieve, {
            "category": "Text Retrieval",
            "brief": "Retrieve a specific section by its ID.",
            "reference": reference_str,
            "parameters": {
                "section_id": {
                    "type": "str", "required": True,
                    "description": "Section identifier (e.g. '5.7.2', '4.4').",
                },
            },
            "returns": "dict",
        }),
        "search_sections": (_search, {
            "category": "Text Retrieval",
            "brief": "Keyword search across all sections.",
            "reference": reference_str,
            "parameters": {
                "query": {
                    "type": "str", "required": True,
                    "description": "Search query (case-insensitive, multi-word AND).",
                },
            },
            "returns": "list of dict",
        }),
        "list_chapters": (_list_ch, {
            "category": "Text Retrieval",
            "brief": "List all chapters and their section IDs.",
            "reference": reference_str,
            "parameters": {},
            "returns": "list of dict",
        }),
        "load_chapter": (_load_ch, {
            "category": "Text Retrieval",
            "brief": "Load a full chapter JSON file.",
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
