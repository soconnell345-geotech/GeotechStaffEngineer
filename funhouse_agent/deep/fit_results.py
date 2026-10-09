"""Catalog and documentation results that fit the result cap WHOLE.

Live smoke wave 1 (geotech questions, G1): ``describe_method`` cut its JSON at
8,000 characters mid-way. Five ``slope_stability`` methods have longer docs,
so their LAST parameters were never seen -- RDD-3 needed
``stage3_effective_normal``, guessed ``stage3_normal_stress`` and said it
"could not retrieve the full list". The cut also told the model to "run a
NARROWER search", which ``describe_method`` cannot do; and ``list_methods``
on ``dm7`` (340+ methods) was cut the same way.

Here a result too long for the cap is made SHORTER BY STRUCTURE, never cut:

* a method's documentation keeps every parameter with its type, whether it
  is required, its default and its allowed values, and only the prose
  (descriptions, notes, return-value text) is shortened, step by step;
* a method or module listing drops its one-line briefs, then lists names by
  category and, if even that is too long, the categories with their counts
  and as many categories' names as fit, saying how to ask for the rest.

The result is always complete, valid JSON. Each says what was shortened.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

#: Keys whose values say how to CALL a parameter: never shortened.
_KEEP_WHOLE = frozenset({"type", "required", "allowed_values", "default",
                         "alias_of", "category", "unit", "units", "enum",
                         "choices", "min", "max", "range"})

#: The prose lengths tried, longest first.
_PROSE_STEPS = (600, 400, 280, 200, 140, 100, 70, 45, 25)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _shorten(obj: Any, n: int, key: str = "") -> Any:
    if key in _KEEP_WHOLE:
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= n else obj[:max(n - 1, 1)].rstrip() + "…"
    if isinstance(obj, dict):
        return {k: _shorten(v, n, str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shorten(v, n, key) for v in obj]
    return obj


def _skeleton(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Only what a call needs: each parameter's type, required, default and
    allowed values; the returns as names."""
    out: Dict[str, Any] = {}
    for k, v in doc.items():
        if k == "parameters" and isinstance(v, dict):
            params = {}
            for name, spec in v.items():
                if isinstance(spec, dict):
                    params[name] = {kk: vv for kk, vv in spec.items()
                                    if kk in _KEEP_WHOLE}
                else:
                    params[name] = spec
            out[k] = params
        elif k == "returns" and isinstance(v, dict):
            out[k] = sorted(v)
        elif k in ("brief", "category", "error", "_note", "note"):
            out[k] = _shorten(v, 200)
    return out


def fit_method_doc(doc: Any, cap: int) -> str:
    """``doc`` (a METHOD_INFO entry, possibly with a ``_note``) as JSON no
    longer than ``cap`` where it can be done by shortening prose; every
    parameter stays listed. ``cap <= 0`` means no limit."""
    text = _dumps(doc)
    if cap <= 0 or len(text) <= cap or not isinstance(doc, dict):
        return text
    note = ("This method's documentation is longer than a tool result may "
            "be, so its descriptions are shortened (…). EVERY parameter is "
            "listed, with its type, whether it is required, its default and "
            "its allowed values.")
    for n in _PROSE_STEPS:
        short = _shorten(doc, n)
        short["_shortened"] = note
        text = _dumps(short)
        if len(text) <= cap:
            return text
    short = _skeleton(doc)
    short["_shortened"] = (note + " Descriptions are left out; the "
                           "parameter names, types and allowed values are "
                           "all here.")
    return _dumps(short)        # complete even if still over the cap


def _is_method_map(value: Any) -> bool:
    return isinstance(value, dict) and value and all(
        isinstance(v, dict) for v in value.values())


def fit_method_list(result: Any, cap: int, agent_name: str = "") -> str:
    """A ``list_methods`` result (``{category: {method: brief}}``, or that
    under ``"methods"`` with a ``note``) fitted to ``cap`` by structure."""
    text = _dumps(result)
    if cap <= 0 or len(text) <= cap or not isinstance(result, dict):
        return text
    wrapped = "methods" in result and _is_method_map(result.get("methods"))
    methods = result["methods"] if wrapped else result
    if not _is_method_map(methods):
        return text
    extra = {k: v for k, v in result.items() if k != "methods"} \
        if wrapped else {}
    categories = sorted(methods)
    who = f"'{agent_name}'" if agent_name else "the module"

    def _with(body: Dict[str, Any], note: str) -> str:
        out = dict(extra)
        out["methods"] = body
        out["_shortened"] = note
        return _dumps(out)

    for n in (80, 40):
        body = {c: {m: _shorten(b, n) for m, b in ms.items()}
                for c, ms in methods.items()}
        text = _with(body, "One-line descriptions are shortened (…) to fit; "
                           "describe_method gives a method's full docs.")
        if len(text) <= cap:
            return text
    names = {c: sorted(ms) for c, ms in methods.items()}
    text = _with(names, "Method NAMES by category (descriptions left out to "
                        "fit); describe_method gives a method's docs.")
    if len(text) <= cap:
        return text
    counts = {c: len(methods[c]) for c in categories}
    shown: Dict[str, List[str]] = {}
    rest: List[str] = []
    for c in categories:
        trial = dict(shown)
        trial[c] = names[c]
        probe = _dumps({"categories": counts, "methods": trial,
                        "_shortened": "x" * 400})
        if len(probe) <= cap:
            shown = trial
        else:
            rest.append(c)
    out = dict(extra)
    out.update({
        "categories": counts,
        "methods": shown,
        "_shortened": (f"{who} has {sum(counts.values())} methods in "
                       f"{len(categories)} categories, more than one result "
                       "holds. Names are listed for the categories shown; "
                       f"for the others ({', '.join(rest)}) call "
                       f"list_methods({agent_name!r}, category=<one of "
                       "them>)."),
    })
    return _dumps(out)


def fit_catalog(catalog: Any, cap: int) -> str:
    """A ``list_agents`` result (``{module: brief}``) fitted to ``cap``:
    briefs shortened, then names only."""
    text = _dumps(catalog)
    if cap <= 0 or len(text) <= cap or not isinstance(catalog, dict):
        return text
    for n in (120, 80, 50, 30):
        short = {k: _shorten(v, n) for k, v in catalog.items()}
        text = _dumps(short)
        if len(text) <= cap:
            return text
    return _dumps({"modules": sorted(catalog),
                   "_shortened": "Module names only (descriptions left out to "
                                 "fit); list_methods(<module>) lists one "
                                 "module's methods."})


def fit_method_choices(err: Any, cap: int, agent_name: str = "",
                       closest: Iterable[str] = ()) -> str:
    """An unknown-method error carrying ``available_methods``
    (``{method: brief}``) fitted to ``cap``: briefs shortened, then names
    only, then the closest names with a count and how to list the rest."""
    text = _dumps(err)
    if cap <= 0 or len(text) <= cap or not isinstance(err, dict):
        return text
    avail = err.get("available_methods")
    if not isinstance(avail, dict):
        return text
    for n in (60, 30):
        out = dict(err)
        out["available_methods"] = {m: _shorten(b, n)
                                    for m, b in avail.items()}
        text = _dumps(out)
        if len(text) <= cap:
            return text
    out = dict(err)
    out["available_methods"] = sorted(avail)
    text = _dumps(out)
    if len(text) <= cap:
        return text
    out = dict(err)
    out.pop("available_methods", None)
    out["n_methods"] = len(avail)
    out["closest"] = list(closest) or out.get("closest") or []
    out["directive"] = (str(err.get("directive") or "") +
                        f" The module has {len(avail)} methods; "
                        f"list_methods({agent_name!r}) lists them by "
                        "category.").strip()
    return _dumps(out)


__all__ = ["fit_method_doc", "fit_method_list", "fit_catalog",
           "fit_method_choices"]
