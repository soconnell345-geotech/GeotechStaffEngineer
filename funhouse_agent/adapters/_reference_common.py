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
    """Convert a type annotation to a human-readable string ("any" for a
    parameter with none: a text parameter left unannotated was advertised
    as "float", 247 of them in the 2026-11-10 review)."""
    if annotation is inspect.Parameter.empty:
        return "any"
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


# ---------------------------------------------------------------------------
# Docstring sections (numpy style, as the reference library writes them)
# ---------------------------------------------------------------------------

#: Section headings read from a reference function's docstring, by the
#: name each is filed under.
_SECTION_NAMES = {
    "parameters": "parameters", "args": "parameters",
    "arguments": "parameters", "other parameters": "parameters",
    "returns": "returns", "return": "returns", "yields": "returns",
    "raises": "raises", "notes": "notes", "note": "notes",
    "references": "references", "reference": "references",
    "examples": "examples", "example": "examples",
    "warnings": "warnings", "see also": "see also",
}

#: The most of each docstring section ``describe_method`` carries for a
#: reference method (2026-11-10 review: the Returns text, where the unit
#: is, never reached the helper).
RETURNS_CHARS = 400
RAISES_CHARS = 400
NOTES_CHARS = 400
REFERENCES_CHARS = 200

#: The most of the Returns text and of the reference line a bare-number,
#: tuple or list result carries beside ``result``.
RESULT_RETURNS_CHARS = 200
RESULT_REFERENCE_CHARS = 160

_DASHES = re.compile(r"-{3,}")
_ENTRY_NAMES = re.compile(r"\*{0,2}[A-Za-z_]\w*(\s*,\s*\*{0,2}[A-Za-z_]\w*)*")


def docstring_sections(doc: str) -> dict:
    """A numpy-style docstring cut into its sections: section name (see
    :data:`_SECTION_NAMES`) -> its lines, as ``inspect.getdoc`` leaves them;
    the text before the first heading under ``""``. A heading is a line
    with no indent, underlined with dashes, ending with a colon, or
    standing alone after a blank line."""
    lines = doc.split("\n")
    out: dict = {"": []}
    current = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        key = stripped.rstrip(":").strip().lower()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        underlined = bool(_DASHES.fullmatch(nxt))
        if key in _SECTION_NAMES and not line[:1].isspace() and (
                underlined or stripped.endswith(":")
                or (i > 0 and not lines[i - 1].strip())):
            current = _SECTION_NAMES[key]
            out.setdefault(current, [])
            i += 2 if underlined else 1
            continue
        out.setdefault(current, []).append(line)
        i += 1
    return out


def docstring_entries(lines) -> list:
    """The entries of a section, each ``(names, head, description)``.

    An entry starts on a line with no indent: a ``name : type`` line
    (``names`` the names before the colon, ``head`` the type), or a line
    with indented text under it, a single word, or a section's only line
    when it is short and not a sentence (no ``names``; ``head`` is a
    Returns type, a Raises exception or a bare parameter name). Its
    description is the lines under it. Text with no indent that starts no
    entry is description: of the entry above, or of one with neither name
    nor head."""
    lines = list(lines or [])
    entries: list = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not line[:1].isspace():
            head, sep, typ = stripped.partition(" : ")
            if sep:
                names = [n.strip().lstrip("*") for n in head.split(",")
                         if n.strip()]
                entries.append([names, typ.strip(), []])
                continue
            nxt = next((n for n in lines[i + 1:] if n.strip()), "")
            # a section that is one short line, not a sentence ("list of
            # str"), is its head
            alone = not nxt and not entries and len(stripped) <= 60 \
                and not stripped.endswith(".")
            if nxt[:1].isspace() or alone or _ENTRY_NAMES.fullmatch(stripped):
                entries.append([[], stripped, []])
                continue
        if entries:
            entries[-1][2].append(stripped)
        else:
            entries.append([[], "", [stripped]])
    return [(names, head, " ".join(desc)) for names, head, desc in entries]


def _trim(text: str, limit: int) -> str:
    """``text`` on one line, cut to ``limit`` characters with "..."."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _first_sentence(text: str, limit: int) -> str:
    """The first sentence of ``text`` (a full stop after a letter, digit or
    bracket, then a space and a capital or a bracket), at most ``limit``
    characters."""
    text = " ".join(str(text or "").split())
    m = re.search(r"(?<=[\w)]\.)\s+(?=[A-Z(])", text[20:])
    if m:
        text = text[:20 + m.start()]
    return _trim(text, limit)


def _doc_type(typ: str) -> str:
    """A docstring's parameter type, without its "optional" marker."""
    typ = re.sub(r",?\s*optional\s*$", "", str(typ or "").strip(),
                 flags=re.I).strip().rstrip(",").strip()
    return _trim(typ, 60)


def returns_parts(func) -> dict:
    """What a reference function's docstring says it returns: ``type`` (the
    type line, or the entries' types), ``text`` (the description; named
    entries as ``name (type): description``) and ``elements`` (the names of
    a tuple's or list's elements, from named entries or, where the type
    says tuple or list, a leading ``(a, b, c)`` in the description; empty
    when it names none: "(US customary, as printed)" before a float's
    description is not two elements, review F9)."""
    sections = docstring_sections(inspect.getdoc(func) or "")
    entries = docstring_entries(sections.get("returns"))
    named = [e for e in entries if e[0]]
    if len(named) > 1:
        return {"type": "tuple",
                "text": "; ".join(f"{', '.join(n)} ({t}): {d}".replace(
                    " (): ", ": ").rstrip(": ") for n, t, d in named),
                "elements": [n for names, _t, _d in named for n in names]}
    typ = " ".join(t for _n, t, _d in entries if t)
    text = " ".join(d for _n, _t, d in entries if d)
    if len(named) == 1:
        typ = named[0][1]
        text = f"{', '.join(named[0][0])}: {text}" if text else \
            ", ".join(named[0][0])
    elements: list = []
    m = re.match(r"\s*\(([^()]+)\)", text)
    if m and "," in m.group(1) and re.search(r"\b(?:tuple|list)\b", typ,
                                             re.I):
        elements = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return {"type": typ, "text": text, "elements": elements}


def _section_text(sections: dict, name: str) -> str:
    """A section on one line: Raises as ``Exception: when; ...``, any other
    section as its text."""
    if name != "raises":
        return " ".join(" ".join(sections.get(name) or []).split())
    parts = []
    for names, head, desc in docstring_entries(sections.get(name)):
        head = head or ", ".join(names)
        parts.append(f"{head}: {desc}" if head and desc else (head or desc))
    return "; ".join(p for p in parts if p)


#: A table, figure or equation named in brackets in a method's brief
#: ("... (Figure 7-15)."), read as the place it is printed.
_PRINTED_PLACE = re.compile(
    r"\(((?:Table|Tables|Figure|Figures|Fig\.|Equation|Equations|Eq\.|Eqs\."
    r"|Section|Sec\.|Chapter|Clause|Annex|Appendix|Chart|Exhibit)"
    r"\s[^()]*)\)")


def _brief_of(doc: str) -> str:
    """A docstring's first paragraph, on one line."""
    lines = []
    for line in doc.split("\n"):
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def reference_line(func, reference: str = "") -> str:
    """The reference a function's result comes from: its docstring's
    References text, else the adapter's ``reference`` with the table,
    figure or equation its brief names in brackets."""
    doc = inspect.getdoc(func) or ""
    refs = _section_text(docstring_sections(doc), "references")
    if refs:
        return _trim(refs, RESULT_REFERENCE_CHARS)
    places = _PRINTED_PLACE.findall(_brief_of(doc))
    line = ", ".join([p for p in [str(reference or "").strip()] if p]
                     + places[-1:])
    return _trim(line, RESULT_REFERENCE_CHARS)


def _default_type(default) -> str:
    """The type a parameter with neither annotation nor docstring type
    shows, from its default: "bool", "float" or "number" (an int default),
    else "any" (review F9: 19 numeric and 6 true/false defaults showed
    "any")."""
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, float):
        return "float"
    if isinstance(default, int):
        return "number"
    return "any"


def extract_method_info(func, category: str, reference: str) -> dict:
    """Build a METHOD_INFO entry from a function's signature + docstring.

    A function whose ``**kwargs`` catch-all is read is marked
    ``open_parameters`` (dispatch cannot tell what it takes) until its
    adapter declares what it reads; one whose catch-all is never read is
    checked by dispatch against its named parameters (GM-110).

    Besides the parameters, the entry carries what the docstring says the
    function returns (``returns``: its type and description, where a unit
    is usually stated), and its ``raises`` (where the printed limits of a
    chart or table usually are), ``notes`` and ``references`` text, each
    trimmed: the 2026-11-10 review found the helper saw only
    ``"returns": "float"``. A parameter with no annotation takes its type
    from its docstring line, else "any".
    """
    doc = inspect.getdoc(func) or ""
    brief = _brief_of(doc) or "No description available."
    sections = docstring_sections(doc)

    # Parameter descriptions and docstring types
    param_descs: dict[str, str] = {}
    param_types: dict[str, str] = {}
    for names, typ, desc in docstring_entries(sections.get("parameters")):
        if not names and _ENTRY_NAMES.fullmatch(typ):
            # a bare parameter name with no type
            names, typ = [n.strip().lstrip("*") for n in typ.split(",")], ""
        for name in names:
            param_descs[name] = desc
            if typ:
                param_types[name] = _doc_type(typ)

    sig = inspect.signature(func)
    params = {}
    for pname, p in sig.parameters.items():
        # Skip *args / **kwargs catch-alls: they are not real parameters and
        # must never be advertised (a VAR_KEYWORD has no default, so it would
        # otherwise be reported as a required parameter).
        if p.kind in (inspect.Parameter.VAR_KEYWORD,
                      inspect.Parameter.VAR_POSITIONAL):
            continue
        if p.annotation is inspect.Parameter.empty:
            ptype = param_types.get(pname) or _default_type(p.default)
        else:
            ptype = param_type_str(p.annotation)
        pinfo: dict = {
            "type": ptype,
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
    ret = returns_parts(func)
    ret_ann = sig.return_annotation
    ret_type = ret["type"] or (param_type_str(ret_ann)
                               if ret_ann is not inspect.Parameter.empty
                               else "")
    if ret_type or ret["text"]:
        info["returns"] = _trim(
            f"{ret_type}: {ret['text']}" if ret_type and ret["text"]
            else (ret_type or ret["text"]), RETURNS_CHARS)
    for name, limit in (("raises", RAISES_CHARS), ("notes", NOTES_CHARS),
                        ("references", REFERENCES_CHARS)):
        text = _section_text(sections, name)
        if text:
            info[name] = _trim(text, limit)
    if catch_all_is_read(func):
        info["open_parameters"] = True
    return info


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def result_label(func, reference: str = "") -> dict:
    """What a bare number, tuple or list from ``func`` is, to go beside it
    in the result: ``method`` (its name), ``returns`` (the first sentence of
    its docstring's Returns text, else of its brief) and ``reference`` (see
    :func:`reference_line`); ``elements`` names a tuple's or list's
    elements where the docstring does."""
    doc = inspect.getdoc(func) or ""
    ret = returns_parts(func)
    label: dict = {"method": getattr(func, "__name__", "")}
    what = _first_sentence(ret["text"] or _brief_of(doc),
                           RESULT_RETURNS_CHARS)
    if what:
        label["returns"] = what
    ref = reference_line(func, reference)
    if ref:
        label["reference"] = ref
    if ret["elements"]:
        label["elements"] = list(ret["elements"])
    return label


def clean_ref_result(result, label: dict = None):
    """Convert a reference function result to a JSON-safe dict.

    A dict is the result as it stands. A bare number, string, tuple or list
    goes under ``result`` with ``label`` (:func:`result_label`) before it:
    the method, what its Returns text says it is and its reference
    (2026-11-10 review: 423 of 1,227 lookups returned a bare number and a
    tsf came back as ``{"result": 35.9}``); ``elements`` only where it
    names as many elements as the result has."""
    if isinstance(result, dict):
        return {k: clean_value(v) for k, v in result.items()}
    if isinstance(result, np.ndarray):
        value = result.tolist()
    elif isinstance(result, (list, tuple)):
        value = [clean_value(v) for v in result]
    else:
        value = clean_value(result)
    out = {k: v for k, v in (label or {}).items() if k != "elements"}
    elements = (label or {}).get("elements")
    if elements and isinstance(value, list) and len(elements) == len(value):
        out["elements"] = list(elements)
    out["result"] = value
    return out


def make_wrapper(func, reference: str = ""):
    """Create adapter wrapper: ``params`` dict → ``func(**params)`` → clean
    dict, a bare result labelled with what it is (:func:`result_label`;
    ``reference``: the adapter's reference for the function)."""
    label = result_label(func, reference)

    def wrapper(params):
        return clean_ref_result(func(**params), label)
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
    "table_5_3_alpha_bond": ["grout_bond_strength",
                             "micropile_grout_bond_strength"],
    "table_5_9_soil_modulus_k_sand": ["py_soil_modulus_sand"],
    "table_5_10_soil_modulus_k_clay": ["py_soil_modulus_clay"],
    "table_5_4_group_efficiency": ["group_efficiency_factor"],
    # gec13 (ground modification)
    "figure_5_5_settlement_improvement": ["settlement_improvement_factor"],
    # REF-62: the old table_4_1_ddc_parameters' stripped name now reaches
    # the DDC table GEC-13 prints (Table 4-3, applied energy, p. 4-46)
    "table_4_3_applied_energy": ["ddc_parameters", "ddc_applied_energy"],
    "table_8_6_soilcrete_strength": ["jet_grout_strength"],
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

#: A parameter type that names a function, which JSON cannot carry.
_CALLABLE_TYPE = re.compile(r"\b(callable|function)\b", re.IGNORECASE)
#: A dict parameter whose description says its keys are numbers ("keys:
#: 9.5, 4.75, ..."): JSON object keys are strings, so the lookup is never
#: reached with the keys it compares.
_NUMBER_KEYS = re.compile(r"\bkeys?\b[^.;]{0,40}?:?\s*-?\d", re.IGNORECASE)
_NUMBER_KEY_TYPE = re.compile(r"dict\[\s*(float|int)\b", re.IGNORECASE)


def json_cannot_carry(func, pinfo: dict, pname: str = "") -> str:
    """Why a parameter's value cannot come through a JSON tool call (a
    function, or a mapping keyed by numbers), or ``""``. ``pinfo`` is its
    :func:`extract_method_info` entry; ``pname`` its name in ``func``."""
    ptype = str(pinfo.get("type") or "")
    annotation = ""
    if pname:
        try:
            ann = inspect.signature(func).parameters[pname].annotation
            if ann is not inspect.Parameter.empty:
                annotation = str(ann)
        except (KeyError, TypeError, ValueError):
            pass
    if _CALLABLE_TYPE.search(ptype) or "Callable" in annotation:
        return "a function"
    if "dict" in ptype.lower() or "dict" in annotation.lower():
        if (_NUMBER_KEY_TYPE.search(ptype) or _NUMBER_KEY_TYPE.search(annotation)
                or _NUMBER_KEYS.search(str(pinfo.get("description") or ""))):
            return "a mapping keyed by numbers"
    return ""


def carriable(func, info: dict) -> bool:
    """Whether a tool call can reach ``func``: no REQUIRED parameter is one
    JSON cannot carry (:func:`json_cannot_carry`). An optional one is taken
    off ``info``'s parameters, so it is not offered (2026-11-10 reach audit:
    AASHTO 1993's performance-period worksheet asks for two functions and
    UFC 3-250-03's slurry-seal surface area for a mapping keyed by sieve
    size in mm; neither could be called from a tool call)."""
    params = info.get("parameters") or {}
    for pname, pinfo in list(params.items()):
        if json_cannot_carry(func, pinfo, pname):
            if pinfo.get("required"):
                return False
            params.pop(pname)
    return True


def build_lookup_registry(lookup_modules, *, skip_callable=False):
    """Scan modules for public functions and build registries.

    Parameters
    ----------
    lookup_modules : list of (module, category_str, reference_str)
    skip_callable : bool
        If True, skip functions with Callable-typed parameters (DM7).
        Whatever it is, a function a tool call cannot reach (a required
        parameter JSON cannot carry: :func:`carriable`) is left out.

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
            entry = extract_method_info(func, category, reference)
            if not carriable(func, entry):
                continue
            registry[name] = make_wrapper(func, reference)
            info[name] = entry
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
    the top (``['D', '3.2.1', '5.2.3.1.1']`` for GEC-12, whose Appendix D
    sits at a level no numbered section has), for the
    ``retrieve_section`` description (BACKLOG REF-01: a fixed example,
    '5.7.2', was in no reference at all). Empty when the reference's text
    cannot be read."""
    try:
        from geotech_references import _retrieval
        chapters = _retrieval.list_chapters(ref_package_name)
    except Exception:  # noqa: BLE001 — an example is never worth a failure
        return []
    by_depth: dict[int, str] = {}
    lettered: dict[int, str] = {}
    for chapter in chapters:
        for sec in chapter.get("sections") or []:
            sid = str(sec.get("section_id") or "")
            # a combined section ('5.15-5.16') is not an example of a number
            if sid and not re.search(r"\.\d+-", sid):
                # an annex keyed by its letter ('D.1') only where no
                # numbered section sits at that level (EN 1997-2 lists its
                # annexes first)
                target = by_depth if re.match(r"\d|P-", sid) else lettered
                target.setdefault(len(re.findall(r"[.-]", sid)), sid)
    for depth, sid in lettered.items():
        by_depth.setdefault(depth, sid)
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
                    "description": (
                        "Search query (case-insensitive). Sections holding "
                        "every word come first; when none holds them all, "
                        "the hits hold some of the words, each marked "
                        "match 'any word' with 'lacks' (the words it does "
                        "not hold): no match for the whole query."),
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
