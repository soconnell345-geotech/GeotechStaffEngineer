"""How a consult's lookups went, counted by code (2026-11-10 reference-
library review, item 5).

A lookup that failed, came back empty or was never made reached the main
agent as the helper's own sentence, "not found" (12 of 41 recorded
consults), and nothing told the main agent otherwise. :class:`LookupReport`
sits on the primary, inside
:class:`~aec_agent_core.deep.keep_work.KeepWork` (whose record of the
running helper it reads), and adds one short block to the ``task`` result of
each helper in :data:`CONSULT_HELPERS`: the lookups made; how many errored,
came back empty or were cut at the size limit; whether the consult ended on
its model-call budget; and each lookup that errored or came back empty, by
name with its reason in a few words. The block is built from the helper's
record, never from its prose.

The primary is told what to do with the block by :data:`LOOKUPS_NOTE`.
"""

import json
import re
from typing import Optional

from langchain.agents.middleware import AgentMiddleware

from funhouse_agent.deep.keep_work import TASK_TOOL, _with_note, current_helper

#: The helpers whose ``task`` results get the block: the references
#: librarian and the reviewer, which both look references up.
CONSULT_HELPERS = frozenset({"references", "reviewer"})

#: The tools whose calls are lookups: a reference module's method and a
#: chart read. Listing modules or describing a method is not a lookup.
LOOKUP_TOOLS = frozenset({"call_agent", "read_reference_figure"})

#: The first words of the block, which the primary's note names.
BLOCK_HEAD = "Lookups, as counted by the app"

#: The marker the size cap leaves on a cut result
#: (``aec_agent_core.deep.vision_kit._truncate``).
_CUT = re.compile(r"\.\.\.\[truncated (\d+) chars\]")

#: An exception's class name at the front of an error text.
_EXCEPTION_NAME = re.compile(r"^[A-Z]\w*(?:Error|Exception|Warning):\s*")

#: Room for each listed lookup's reason, a search's query and the number
#: of lookups listed by name.
REASON_CHARS = 90
QUERY_CHARS = 40
MAX_LISTED = 6

#: Said to the primary where a lookup errored, none was made or the
#: budget ended the consult.
_FAILED_ADVICE = (" Tell the user which lookups failed or were not "
                  "completed; do not fill them from memory.")

#: The primary's note, where the references helper is built: what the
#: block is and what to do with it.
LOOKUPS_NOTE = (
    "A `references` or `reviewer` task's result ends with a block the app "
    f"writes, \"{BLOCK_HEAD}\", which says how the helper's lookups went. "
    "Where it says a lookup errored, that no lookup was made, or that the "
    "consult ended on its call budget, tell the user that the lookup failed "
    "or was not completed, and do not fill the gap from memory as if it "
    "were the reference. \"Came back empty\" means the reference was "
    "searched and that search found nothing.")


def _trim(text: str, room: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= room:
        return text
    cut = text[:room].rsplit(" ", 1)[0] or text[:room]
    return cut.rstrip(" ,;:") + "..."


def _parsed(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _error_of(data) -> Optional[str]:
    """The error text a lookup's result carries, or ``None``."""
    if isinstance(data, dict):
        if data.get("error"):
            return str(data["error"])
        inner = data.get("result")
        if (isinstance(inner, list) and len(inner) == 1
                and isinstance(inner[0], dict) and inner[0].get("error")):
            return str(inner[0]["error"])
    if (isinstance(data, list) and len(data) == 1
            and isinstance(data[0], dict) and data[0].get("error")):
        return str(data[0]["error"])
    return None


def _empty(data) -> bool:
    """Whether a lookup's result holds nothing: an empty answer, or a
    search that shows no hit."""
    if data is None or data == [] or data == {} or data == "":
        return True
    if isinstance(data, dict):
        if "result" in data and data["result"] in (None, [], {}, ""):
            return True
        if data.get("shown") == 0 or data.get("total_hits") == 0:
            return True
    return False


def _reason(error: str, name: str) -> str:
    """A few words of an error: its first line, without the lookup's own
    name or an exception's class name in front."""
    text = str(error).strip().splitlines()[0] if str(error).strip() else ""
    for prefix in (name + ": ", name.split(" ")[0] + ": "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = _EXCEPTION_NAME.sub("", text).strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]               # a KeyError's message is quoted
    return _trim(text, REASON_CHARS) or "an error"


def _lookup_name(tool: str, args) -> str:
    """A lookup by name: module.method (with a search's query), or the
    figure a chart read asked for."""
    args = args if isinstance(args, dict) else {}
    if tool == "read_reference_figure":
        return " ".join(str(p) for p in (
            tool, args.get("reference") or "",
            ("figure " + str(args["figure_number"]))
            if args.get("figure_number") else "") if p)
    params = args.get("parameters")
    params = params if isinstance(params, dict) else {}
    name = ".".join(str(p) for p in (args.get("agent_name"),
                                     args.get("method")) if p) or tool
    query = params.get("query") or args.get("query")
    query = query.strip(" \"'") if isinstance(query, str) else ""
    if query:
        name += ' "' + _trim(query, QUERY_CHARS) + '"'
    return name


def lookup_outcomes(rec) -> list:
    """Each lookup the helper made, in order: ``{"name", "outcome",
    "reason", "cut"}``, ``outcome`` one of "ok", "errored", "empty"."""
    args_by_id = {}
    for step in rec.steps:
        if step.get("kind") == "model":
            for call in step.get("calls") or []:
                args_by_id[call.get("id") or ""] = call.get("args")
    out = []
    for step in rec.steps:
        if step.get("kind") != "result" or step.get("name") not in LOOKUP_TOOLS:
            continue
        text = step.get("text") or ""
        name = _lookup_name(step["name"], args_by_id.get(step.get("id") or ""))
        cut = bool(_CUT.search(text))
        data = _parsed(text)
        error = _error_of(data)
        if error is None and step.get("status") == "error":
            error = text
        if error is not None:
            out.append({"name": name, "outcome": "errored",
                        "reason": _reason(error, name), "cut": cut})
        elif not cut and _empty(data):
            out.append({"name": name, "outcome": "empty",
                        "reason": "returned nothing", "cut": cut})
        else:
            out.append({"name": name, "outcome": "ok", "reason": "",
                        "cut": cut})
    return out


def lookup_block(rec) -> str:
    """The block for one consult's record (see the module docstring)."""
    outcomes = lookup_outcomes(rec)
    errored = [o for o in outcomes if o["outcome"] == "errored"]
    empty = [o for o in outcomes if o["outcome"] == "empty"]
    cut = sum(1 for o in outcomes if o["cut"])
    if not outcomes:
        line = f"[{BLOCK_HEAD}: no lookup was made."
    elif not (errored or empty or cut):
        line = (f"[{BLOCK_HEAD}: {len(outcomes)} made; none errored, came "
                "back empty or was cut at the size limit.")
    else:
        line = (f"[{BLOCK_HEAD}: {len(outcomes)} made; {len(errored)} "
                f"errored, {len(empty)} came back empty, {cut} cut at the "
                "size limit.")
    if rec.budget_reached:
        line += (" The consult ended on its call budget: its answer is "
                 "what it had gathered by then.")
    if not outcomes or errored or rec.budget_reached:
        line += _FAILED_ADVICE
    lines = [line]
    listed = errored + empty
    for o in listed[:MAX_LISTED]:
        what = "errored" if o["outcome"] == "errored" else "came back empty"
        reason = f": {o['reason']}" if o["outcome"] == "errored" else ""
        lines.append(f"- {o['name']} {what}{reason}")
    if len(listed) > MAX_LISTED:
        lines.append(f"- and {len(listed) - MAX_LISTED} more")
    return "\n".join(lines) + "]"


class LookupReport(AgentMiddleware):
    """On the primary, inside ``KeepWork``: a references or reviewer
    ``task`` result gets :func:`lookup_block` after its text."""

    @staticmethod
    def _consult(request) -> bool:
        call = getattr(request, "tool_call", None) or {}
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        return (call.get("name") == TASK_TOOL
                and args.get("subagent_type") in CONSULT_HELPERS)

    @staticmethod
    def _reported(result):
        rec = current_helper()
        if rec is None or rec.subagent not in CONSULT_HELPERS:
            return result
        try:
            block = lookup_block(rec)
        except Exception:                                  # noqa: BLE001
            return result
        return _with_note(result, block)

    def wrap_tool_call(self, request, handler):
        if not self._consult(request):
            return handler(request)
        return self._reported(handler(request))

    async def awrap_tool_call(self, request, handler):
        if not self._consult(request):
            return await handler(request)
        return self._reported(await handler(request))


__all__ = ["CONSULT_HELPERS", "LOOKUP_TOOLS", "BLOCK_HEAD", "LOOKUPS_NOTE",
           "LookupReport", "lookup_block", "lookup_outcomes"]
