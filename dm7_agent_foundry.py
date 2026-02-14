"""
DM7 Agent - Palantir Foundry AIP Agent Studio Version.

Wraps 340+ geotechnical equations from NAVFAC Design Manual 7
(UFC 3-220-10 Soil Mechanics and UFC 3-220-20 Foundations & Earth Structures).

Register these three functions as tools in AIP Agent Studio:
  1. dm7_agent           - Run a DM7 equation
  2. dm7_list_methods    - Browse available equations by chapter/category
  3. dm7_describe_method - Get detailed parameter docs for a specific equation

FOUNDRY SETUP:
  - Place the DM7Eqs/geotech package in the code repository
  - These functions accept and return JSON strings for LLM compatibility
  - No external dependencies beyond numpy and scipy
"""

import json
import math
import sys
import os
import inspect
import typing

import numpy as np
from functions.api import function

# ---------------------------------------------------------------------------
# Add DM7Eqs to path so 'from geotech.dm7_X...' works
# ---------------------------------------------------------------------------
_dm7_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DM7Eqs")
if _dm7_root not in sys.path:
    sys.path.insert(0, _dm7_root)

# ---------------------------------------------------------------------------
# Import all DM7 chapter modules
# ---------------------------------------------------------------------------
from geotech.dm7_1 import chapter1 as dm7_1_ch1
from geotech.dm7_1 import chapter2 as dm7_1_ch2
from geotech.dm7_1 import chapter3 as dm7_1_ch3
from geotech.dm7_1 import chapter4 as dm7_1_ch4
from geotech.dm7_1 import chapter5 as dm7_1_ch5
from geotech.dm7_1 import chapter6 as dm7_1_ch6
from geotech.dm7_1 import chapter7 as dm7_1_ch7
from geotech.dm7_1 import chapter8 as dm7_1_ch8

from geotech.dm7_2 import prologue as dm7_2_pro
from geotech.dm7_2 import chapter2 as dm7_2_ch2
from geotech.dm7_2 import chapter3 as dm7_2_ch3
from geotech.dm7_2 import chapter4 as dm7_2_ch4
from geotech.dm7_2 import chapter5 as dm7_2_ch5
from geotech.dm7_2 import chapter6 as dm7_2_ch6
from geotech.dm7_2 import chapter7 as dm7_2_ch7


# ---------------------------------------------------------------------------
# Chapter metadata for categorization
# ---------------------------------------------------------------------------
CHAPTER_INFO = {
    "dm7_1_ch1": {
        "module": dm7_1_ch1,
        "category": "DM7.1 Ch1 - Identification & Classification",
        "reference": "UFC 3-220-10, Chapter 1",
    },
    "dm7_1_ch2": {
        "module": dm7_1_ch2,
        "category": "DM7.1 Ch2 - Field Exploration & Testing",
        "reference": "UFC 3-220-10, Chapter 2",
    },
    "dm7_1_ch3": {
        "module": dm7_1_ch3,
        "category": "DM7.1 Ch3 - Laboratory Testing",
        "reference": "UFC 3-220-10, Chapter 3",
    },
    "dm7_1_ch4": {
        "module": dm7_1_ch4,
        "category": "DM7.1 Ch4 - Distribution of Stresses",
        "reference": "UFC 3-220-10, Chapter 4",
    },
    "dm7_1_ch5": {
        "module": dm7_1_ch5,
        "category": "DM7.1 Ch5 - Consolidation & Settlement",
        "reference": "UFC 3-220-10, Chapter 5",
    },
    "dm7_1_ch6": {
        "module": dm7_1_ch6,
        "category": "DM7.1 Ch6 - Seepage & Drainage",
        "reference": "UFC 3-220-10, Chapter 6",
    },
    "dm7_1_ch7": {
        "module": dm7_1_ch7,
        "category": "DM7.1 Ch7 - Slope Stability",
        "reference": "UFC 3-220-10, Chapter 7",
    },
    "dm7_1_ch8": {
        "module": dm7_1_ch8,
        "category": "DM7.1 Ch8 - Correlations for Soil Properties",
        "reference": "UFC 3-220-10, Chapter 8",
    },
    "dm7_2_pro": {
        "module": dm7_2_pro,
        "category": "DM7.2 Prologue - Shear Strength",
        "reference": "UFC 3-220-20, Prologue",
    },
    "dm7_2_ch2": {
        "module": dm7_2_ch2,
        "category": "DM7.2 Ch2 - Excavations & Retained Cuts",
        "reference": "UFC 3-220-20, Chapter 2",
    },
    "dm7_2_ch3": {
        "module": dm7_2_ch3,
        "category": "DM7.2 Ch3 - Earthwork & Compaction",
        "reference": "UFC 3-220-20, Chapter 3",
    },
    "dm7_2_ch4": {
        "module": dm7_2_ch4,
        "category": "DM7.2 Ch4 - Rigid Retaining Structures",
        "reference": "UFC 3-220-20, Chapter 4",
    },
    "dm7_2_ch5": {
        "module": dm7_2_ch5,
        "category": "DM7.2 Ch5 - Shallow Foundations",
        "reference": "UFC 3-220-20, Chapter 5",
    },
    "dm7_2_ch6": {
        "module": dm7_2_ch6,
        "category": "DM7.2 Ch6 - Deep Foundations",
        "reference": "UFC 3-220-20, Chapter 6",
    },
    "dm7_2_ch7": {
        "module": dm7_2_ch7,
        "category": "DM7.2 Ch7 - Probability & Reliability",
        "reference": "UFC 3-220-20, Chapter 7",
    },
}


# ---------------------------------------------------------------------------
# Helper: check if a parameter type is Callable (not JSON-serializable)
# ---------------------------------------------------------------------------
def _has_callable_param(func) -> bool:
    """Return True if any parameter has a Callable type annotation."""
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        ann = p.annotation
        if ann is inspect.Parameter.empty:
            continue
        origin = getattr(ann, "__origin__", None)
        if origin is not None and origin is getattr(typing, "Callable", None).__origin__:
            return True
        # Also check string representation for Callable
        ann_str = str(ann)
        if "Callable" in ann_str:
            return True
    return False


def _param_type_str(annotation) -> str:
    """Convert a type annotation to a human-readable string."""
    if annotation is inspect.Parameter.empty:
        return "float"
    if annotation is float:
        return "float"
    if annotation is int:
        return "int"
    if annotation is bool:
        return "bool"
    if annotation is str:
        return "str"
    # Handle typing generics
    ann_str = str(annotation)
    if "List[float]" in ann_str or "Sequence[float]" in ann_str:
        return "array of float"
    if "List[Tuple" in ann_str:
        return "array of tuples"
    if "Tuple" in ann_str:
        return "tuple"
    return ann_str


# ---------------------------------------------------------------------------
# Auto-build METHOD_REGISTRY and METHOD_INFO
# ---------------------------------------------------------------------------
METHOD_REGISTRY = {}  # method_name -> callable
METHOD_INFO = {}      # method_name -> {category, brief, reference, parameters, returns}

_name_collisions = {}  # track duplicate names across chapters

for _ch_key, _ch_meta in CHAPTER_INFO.items():
    _mod = _ch_meta["module"]
    _cat = _ch_meta["category"]
    _ref = _ch_meta["reference"]

    for _name, _func in inspect.getmembers(_mod, inspect.isfunction):
        if _name.startswith("_"):
            continue

        # Skip functions that take Callable parameters (can't pass via JSON)
        if _has_callable_param(_func):
            continue

        # Handle name collisions by prefixing with chapter
        if _name in METHOD_REGISTRY:
            # Rename existing entry if not already renamed
            if _name in _name_collisions:
                _existing_key = _name_collisions[_name]
            else:
                # First collision: rename the existing entry
                _existing_key = _ch_key.split("_ch")[0] + "_" + _name
                # But we need the original chapter key... store it
                _existing_key = None
                for _prev_ch_key, _prev_meta in CHAPTER_INFO.items():
                    if _prev_ch_key == _ch_key:
                        break
                    _prev_mod = _prev_meta["module"]
                    if hasattr(_prev_mod, _name) and getattr(_prev_mod, _name) is METHOD_REGISTRY[_name]:
                        _existing_key = _prev_ch_key
                        break
                if _existing_key:
                    _new_name_existing = f"{_existing_key}_{_name}"
                    METHOD_REGISTRY[_new_name_existing] = METHOD_REGISTRY.pop(_name)
                    METHOD_INFO[_new_name_existing] = METHOD_INFO.pop(_name)
                    _name_collisions[_name] = _new_name_existing

            # Add new entry with chapter prefix
            _qualified_name = f"{_ch_key}_{_name}"
            METHOD_REGISTRY[_qualified_name] = _func
        else:
            _qualified_name = _name
            METHOD_REGISTRY[_name] = _func

        # Extract docstring
        _doc = inspect.getdoc(_func) or ""

        # Extract first paragraph as brief (handles multi-line summaries)
        _desc_lines = []
        for _line in _doc.split("\n"):
            if _line.strip() == "":
                break
            _desc_lines.append(_line.strip())
        _brief = " ".join(_desc_lines) if _desc_lines else "No description available."

        # Extract parameters from signature
        _sig = inspect.signature(_func)
        _params = {}

        # Parse numpy-style docstring for parameter descriptions
        _param_descs = {}
        _doc_lines = _doc.split("\n")
        _in_params = False
        _current_param = None
        for _dline in _doc_lines:
            _stripped = _dline.strip()
            if _stripped.lower() in ("parameters", "parameters:", "args", "args:"):
                _in_params = True
                continue
            if _stripped.startswith("---"):
                continue
            if _stripped.lower() in ("returns", "returns:", "raises", "raises:",
                                     "examples", "examples:", "notes", "notes:",
                                     "references", "references:"):
                _in_params = False
                _current_param = None
                continue
            if _in_params:
                # Check if this line declares a parameter (name : type)
                if " : " in _stripped:
                    _current_param = _stripped.split(" : ")[0].strip()
                    _param_descs[_current_param] = ""
                elif _current_param and _stripped:
                    # Continuation line — append to current param description
                    if _param_descs[_current_param]:
                        _param_descs[_current_param] += " " + _stripped
                    else:
                        _param_descs[_current_param] = _stripped

        for _pname, _p in _sig.parameters.items():
            _pinfo = {
                "type": _param_type_str(_p.annotation),
                "required": _p.default is inspect.Parameter.empty,
            }
            if _p.default is not inspect.Parameter.empty:
                _pinfo["default"] = _p.default
            if _pname in _param_descs and _param_descs[_pname]:
                _pinfo["description"] = _param_descs[_pname]

            _params[_pname] = _pinfo

        # Extract return type from signature annotation
        _ret_ann = _sig.return_annotation
        _returns_str = ""
        if _ret_ann is not inspect.Parameter.empty:
            _returns_str = _param_type_str(_ret_ann)

        _method_info = {
            "category": _cat,
            "brief": _brief,
            "reference": _ref,
            "parameters": _params,
        }
        if _returns_str:
            _method_info["returns"] = _returns_str

        METHOD_INFO[_qualified_name] = _method_info


# ---------------------------------------------------------------------------
# Numpy/NaN-safe value cleaning
# ---------------------------------------------------------------------------

def _clean_value(v):
    """Convert numpy types and NaN to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_clean_value(item) for item in v]
    if isinstance(v, dict):
        return {k: _clean_value(val) for k, val in v.items()}
    return v


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def dm7_agent(method: str, parameters_json: str) -> str:
    """
    DM7 geotechnical equation calculator.

    Implements 340+ equations from NAVFAC Design Manual 7 (UFC 3-220-10 Soil
    Mechanics and UFC 3-220-20 Foundations & Earth Structures). Covers soil
    classification, stresses, consolidation, seepage, slope stability, soil
    property correlations, shear strength, earth pressure, bearing capacity,
    deep foundations, and reliability analysis.

    Parameters:
        method: The equation/function name. Use dm7_list_methods() to browse.
        parameters_json: JSON string of parameters. Use dm7_describe_method() for details.

    Returns:
        JSON string with the calculation result or an error message.
    """
    try:
        parameters = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        # Try fuzzy match
        matches = [m for m in METHOD_REGISTRY if method.lower() in m.lower()]
        if matches:
            suggestion = ", ".join(matches[:5])
            return json.dumps({
                "error": f"Unknown method '{method}'. Did you mean: {suggestion}?"
            })
        return json.dumps({
            "error": f"Unknown method '{method}'. Use dm7_list_methods() to see available methods."
        })

    try:
        func = METHOD_REGISTRY[method]
        result = func(**parameters)

        # Handle different return types
        if isinstance(result, dict):
            return json.dumps({k: _clean_value(v) for k, v in result.items()}, default=str)
        elif isinstance(result, tuple):
            return json.dumps({"result": [_clean_value(v) for v in result]}, default=str)
        elif isinstance(result, (list, np.ndarray)):
            return json.dumps({"result": _clean_value(result)}, default=str)
        else:
            return json.dumps({"result": _clean_value(result)}, default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def dm7_list_methods(category: str = "") -> str:
    """
    Lists available DM7 equations organized by chapter.

    Parameters:
        category: Optional filter. Partial match on category name
                  (e.g. 'settlement', 'bearing', 'earth pressure', 'Ch4').

    Returns:
        JSON string with method names grouped by chapter/category.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        cat = info["category"]
        if category and category.lower() not in cat.lower():
            continue
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return json.dumps(result)


@function
def dm7_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a DM7 equation.

    Parameters:
        method: The method name (e.g. 'boussinesq_point_load',
                'primary_consolidation_settlement_nc').

    Returns:
        JSON string with parameters, types, defaults, and description.
    """
    if method not in METHOD_INFO:
        matches = [m for m in METHOD_INFO if method.lower() in m.lower()]
        if matches:
            suggestion = ", ".join(matches[:10])
            return json.dumps({
                "error": f"Unknown method '{method}'. Similar: {suggestion}"
            })
        return json.dumps({
            "error": f"Unknown method '{method}'. Use dm7_list_methods() to browse."
        })
    return json.dumps(METHOD_INFO[method], default=str)
