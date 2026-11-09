"""A tool call with an argument the tool does not take is refused, by name.

Foundry brief 5 (2026-10-08, N3): GPT-5.4 called ``analyze_pdf_page(pages=12)``.
The tool takes ``page``. LangChain's ``StructuredTool`` validates a call
against a schema built from the function's signature, and that schema IGNORES
keys it does not know, so ``pages`` was dropped without a word, ``page``
defaulted to 0 and the tool looked at the report's cover — three times. The
agent then marked the scanned logs "skipped" with an invented reason.

:func:`strict_tool` makes a tool refuse such a call instead: the result is a
JSON error naming each unknown key, the arguments the tool does take and the
nearest one (``pages`` -> ``page``), and nothing is run. The schema the model
is shown is unchanged. Tools whose schema takes extra keys ON PURPOSE (the
``call_agent`` auto-nesting of flattened parameters) are left as they are.

The check lives in the tool itself, not in a middleware, so it holds in every
agent the tool is given to (the primary, each helper, the lean and minimal
review agents), and the refused call is still recorded by the run's callbacks
(the activity log) like any other tool call.
"""

from __future__ import annotations

import difflib
import functools
import inspect
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict


def _key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def closest_argument(bad: str, valid: Iterable[str]) -> Optional[str]:
    """The valid argument ``bad`` most likely meant: the same name in another
    case or spelling (``attachmentKey``), a plural or singular of it
    (``pages`` -> ``page``), else the nearest by spelling; ``None`` when
    nothing is close."""
    valid = [v for v in valid]
    by_key = {_key(v): v for v in valid}
    k = _key(bad)
    if k in by_key:
        return by_key[k]
    for kv, v in by_key.items():
        if k in (kv + "s", kv + "es") or kv in (k + "s", k + "es"):
            return v
    hit = difflib.get_close_matches(k, list(by_key), n=1, cutoff=0.6)
    return by_key[hit[0]] if hit else None


def unknown_arguments_error(tool_name: str, unknown: List[str],
                            valid: List[str]) -> str:
    """The JSON a tool returns for a call with arguments it does not take."""
    did = {u: closest_argument(u, valid) for u in unknown}
    did = {u: v for u, v in did.items() if v}
    names = ", ".join(repr(u) for u in unknown)
    out: Dict[str, Any] = {
        "error": (f"{tool_name} has no argument {names}; the call was NOT "
                  f"run. Call it again with the arguments it takes."),
        "unknown_arguments": list(unknown),
        "valid_arguments": list(valid),
    }
    if did:
        out["did_you_mean"] = did
    plural = [v for u, v in did.items()
              if _key(u) in (_key(v) + "s", _key(v) + "es")]
    if plural:
        out["hint"] = "; ".join(
            f"'{v}' takes ONE value: for several, call {tool_name} once for "
            f"each" for v in plural)
    return json.dumps(out)


def _takes_extra_on_purpose(schema: Any) -> bool:
    config = getattr(schema, "model_config", None) or {}
    return config.get("extra") == "allow"


def strict_tool(tool):
    """``tool`` as one that refuses unknown argument names (see the module
    docstring); returned unchanged when it already takes extra keys, has no
    pydantic schema or no plain function behind it."""
    schema = getattr(tool, "args_schema", None)
    func = getattr(tool, "func", None)
    if (func is None or getattr(func, "_strict_args", False)
            or not (isinstance(schema, type) and issubclass(schema, BaseModel))
            or _takes_extra_on_purpose(schema)):
        return tool
    valid = list(schema.model_fields)
    try:
        own = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        own = set()
    allowed = set(valid) | own
    name = tool.name

    @functools.wraps(func)
    def guarded(*args, **kwargs):
        unknown = [k for k in kwargs if k not in allowed]
        if unknown:
            return unknown_arguments_error(name, unknown, valid)
        return func(*args, **kwargs)

    # The same schema, but letting unknown keys THROUGH validation so the
    # guard sees them (pydantic drops them otherwise). What the model is
    # shown is unchanged.
    lax = type(schema.__name__, (schema,), {
        "model_config": ConfigDict(extra="allow"),
        "__module__": schema.__module__,
        "__doc__": schema.__doc__,
    })
    guarded._strict_args = True
    return tool.model_copy(update={"func": guarded, "args_schema": lax})


def strict_tools(tools: Iterable[Any]) -> list:
    """:func:`strict_tool` over a list."""
    return [strict_tool(t) for t in tools]


__all__ = ["strict_tool", "strict_tools", "unknown_arguments_error",
           "closest_argument"]
