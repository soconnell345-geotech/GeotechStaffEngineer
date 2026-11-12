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
import logging
import re
from typing import Optional

from langchain.agents.middleware import AgentMiddleware

from funhouse_agent.deep.keep_work import TASK_TOOL, _with_note, current_helper

_log = logging.getLogger(__name__)

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
_EXCEPTION_NAME = re.compile(r"^([A-Z]\w*(?:Error|Exception|Warning)):\s*")

#: Room for each listed lookup's reason, a search's query, the number of
#: lookups listed by name and of methods named as answered on a later try.
REASON_CHARS = 90
QUERY_CHARS = 40
MAX_LISTED = 6
MAX_RECOVERED_NAMED = 4

#: Said to the primary where a lookup failed and no later call for the same
#: thing answered.
_FAILED_ADVICE = (" Tell the user which lookups failed; do not fill them "
                  "from memory.")

#: Said where the helper made no lookup at all.
_NO_LOOKUP_ADVICE = (" Nothing in the answer comes from the references; say "
                     "so to the user.")

#: The reason given for a chart read whose reader said the figure is not on
#: the page it was shown.
NOT_ON_THE_PAGE = "the reader said the figure is not on the page"

#: Said where the helper's model-call budget ended the consult.
_BUDGET_LINE = (" The consult ended on its call budget, so its answer may "
                "be incomplete.")

#: The primary's note, where the references helper is built: what the
#: block is and what to do with it.
LOOKUPS_NOTE = (
    "A `references` or `reviewer` task's result ends with a block the app "
    f"writes, \"{BLOCK_HEAD}\", which says how the helper's lookups went. "
    "Where it says a lookup errored (and no later call for the same thing "
    "answered), tell the user that the lookup failed, and do not fill the "
    "gap from memory as if it were the reference. Where it says no lookup "
    "was made, nothing in the answer comes from the references: say so. "
    "Where it says the consult ended on its call budget, its answer may be "
    "incomplete: say so where it matters. \"Came back empty\" means the "
    "reference was searched and found nothing that holds all the words "
    "asked for (each such search is named with its query), or a chart's "
    "reader found the figure not on its page.")


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


#: How the library marks a search hit that holds only some of the query's
#: words (``match``; geotech-references since the 2026-11-10 review).
ANY_WORD = "any word"

#: The reason given for a search whose every hit holds only some words.
NO_WHOLE_MATCH = "no hit holds every word"


def _hits(data):
    """A search result's hits: the list under ``result`` (or ``hits``), or
    the list itself; ``None`` for anything else."""
    if isinstance(data, dict):
        data = data.get("result", data.get("hits"))
    return data if isinstance(data, list) else None


def _no_whole_match(data) -> bool:
    """Whether a search found nothing for the whole query: it has hits, and
    every one is marked as holding only some of the words (the library
    fills a search no section answers with such hits)."""
    hits = _hits(data)
    return bool(hits) and all(isinstance(h, dict) and h.get("match") == ANY_WORD
                              for h in hits)


def _read_not_on_page(data, args) -> bool:
    """Whether a chart read's answer says the figure is not on the page it
    was shown (the read tool's own test, ``said_not_on_page``)."""
    if not isinstance(data, dict) or not data.get("analysis"):
        return False
    args = args if isinstance(args, dict) else {}
    try:
        from funhouse_agent.reference_vision import said_not_on_page
    except Exception:                                      # noqa: BLE001
        return False
    return said_not_on_page(data["analysis"], data.get("figure_number")
                            or args.get("figure_number") or "")


def _unquoted(text: str) -> str:
    """``text`` without one pair of quotes round all of it (a KeyError's
    message is quoted)."""
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _reason(error: str, name: str) -> str:
    """A few words of an error: its first line, without the lookup's own
    name in front; an exception's class name is kept ("KeyError: x")."""
    text = str(error).strip().splitlines()[0] if str(error).strip() else ""
    for prefix in (name + ": ", name.split(" ")[0] + ": "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip()
    m = _EXCEPTION_NAME.match(text)
    if m:
        text = f"{m.group(1)}: {_unquoted(text[m.end():].strip())}"
    else:
        text = _unquoted(text)
    return _trim(text, REASON_CHARS) or "an error"


def _query(args) -> str:
    """A search's query as the helper gave it (``parameters.query``, or a
    flattened ``query``), quotes round it dropped; "" for a lookup that is
    not a search."""
    args = args if isinstance(args, dict) else {}
    params = args.get("parameters")
    params = params if isinstance(params, dict) else {}
    query = params.get("query") or args.get("query")
    return query.strip(" \"'") if isinstance(query, str) else ""


def _lookup_key(tool: str, args) -> str:
    """What makes two lookups the same: module.method, with a search's
    query (case and spacing folded), or the figure a chart read asked for.
    A search for another topic answering does not answer the first."""
    args = args if isinstance(args, dict) else {}
    if tool == "read_reference_figure":
        return f"{tool} {args.get('reference') or ''} " \
               f"{args.get('figure_number') or ''}".strip()
    key = ".".join(str(p) for p in (args.get("agent_name"),
                                    args.get("method")) if p) or tool
    query = " ".join(_query(args).lower().split())
    return f'{key} "{query}"' if query else key


def _lookup_name(tool: str, args) -> str:
    """A lookup by name: module.method (with a search's query), or the
    figure a chart read asked for."""
    args = args if isinstance(args, dict) else {}
    if tool == "read_reference_figure":
        return " ".join(str(p) for p in (
            tool, args.get("reference") or "",
            ("figure " + str(args["figure_number"]))
            if args.get("figure_number") else "") if p)
    name = ".".join(str(p) for p in (args.get("agent_name"),
                                     args.get("method")) if p) or tool
    query = _query(args)
    if query:
        name += ' "' + _trim(query, QUERY_CHARS) + '"'
    return name


def lookup_outcomes(rec) -> list:
    """Each lookup the helper made, in order: ``{"name", "key", "outcome",
    "reason", "cut", "recovered"}``, ``outcome`` one of "ok", "errored",
    "empty". An errored or empty lookup is ``recovered`` when a later call
    for the same lookup (``key``: the same method, and for a search the
    same query) succeeded: most errors in the recorded consults were
    argument slips, then retried."""
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
        args = args_by_id.get(step.get("id") or "")
        name = _lookup_name(step["name"], args)
        row = {"name": name, "key": _lookup_key(step["name"], args),
               "outcome": "ok", "reason": "",
               "cut": bool(_CUT.search(text)), "recovered": False}
        data = _parsed(text)
        error = _error_of(data)
        if error is None and step.get("status") == "error":
            error = text
        if error is not None:
            row.update(outcome="errored", reason=_reason(error, name))
        elif not row["cut"] and _empty(data):
            row.update(outcome="empty", reason="returned nothing")
        elif not row["cut"] and _no_whole_match(data):
            # hits holding only some of the words are no match for the
            # query (2026-11-10 review, core-05)
            row.update(outcome="empty", reason=NO_WHOLE_MATCH)
        elif step["name"] == "read_reference_figure" and _read_not_on_page(
                data, args):
            # no value was read off a page the figure is not on
            row.update(outcome="empty", reason=NOT_ON_THE_PAGE)
        out.append(row)
    for i, row in enumerate(out):
        if row["outcome"] != "ok":
            row["recovered"] = any(later["key"] == row["key"]
                                   and later["outcome"] == "ok"
                                   for later in out[i + 1:])
    return out


def lookup_block(rec) -> str:
    """The block for one consult's record (see the module docstring)."""
    outcomes = lookup_outcomes(rec)
    failed = [o for o in outcomes
              if o["outcome"] != "ok" and not o["recovered"]]
    errored = [o for o in failed if o["outcome"] == "errored"]
    empty = [o for o in failed if o["outcome"] == "empty"]
    recovered = list(dict.fromkeys(o["key"] for o in outcomes
                                   if o["recovered"]))
    cut = sum(1 for o in outcomes if o["cut"])
    if not outcomes:
        line = f"[{BLOCK_HEAD}: no lookup was made."
    elif not (errored or empty or cut or recovered):
        line = (f"[{BLOCK_HEAD}: {len(outcomes)} made; none errored, came "
                "back empty or was cut at the size limit.")
    elif not (errored or empty or cut):
        line = (f"[{BLOCK_HEAD}: {len(outcomes)} made; each one that failed "
                "was answered by a later call for the same lookup.")
    else:
        line = (f"[{BLOCK_HEAD}: {len(outcomes)} made; {len(errored)} "
                f"errored, {len(empty)} came back empty, {cut} cut at the "
                "size limit.")
    if rec.budget_reached:
        line += _BUDGET_LINE
    if not outcomes:
        line += _NO_LOOKUP_ADVICE
    elif errored:
        line += _FAILED_ADVICE
    lines = [line]
    for o in failed[:MAX_LISTED]:
        what = "errored" if o["outcome"] == "errored" else "came back empty"
        reason = (f": {o['reason']}" if o["outcome"] == "errored"
                  or o["reason"] in (NO_WHOLE_MATCH, NOT_ON_THE_PAGE)
                  else "")
        lines.append(f"- {o['name']} {what}{reason}")
    if len(failed) > MAX_LISTED:
        lines.append(f"- and {len(failed) - MAX_LISTED} more")
    if recovered:
        named = ", ".join(recovered[:MAX_RECOVERED_NAMED])
        if len(recovered) > MAX_RECOVERED_NAMED:
            named += f" and {len(recovered) - MAX_RECOVERED_NAMED} more"
        lines.append(f"- failed, then answered by a later call: {named}")
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
            # the consult's answer goes back without the block, and why is
            # kept in the log
            _log.warning("the %s consult's lookups block was not added",
                         rec.subagent, exc_info=True)
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
