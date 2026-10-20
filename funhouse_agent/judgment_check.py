"""The turn-end judgment-call check (BACKLOG GM-174), behind
``GEOTECH_JUDGMENT_CALLS`` (on by default since 2026-10-20, GM-226), and
only where the build has a references helper to consult: with the chat's
"Consult references" off it never acts (:func:`mode_for`).

WHY. With the switch on, the geotech prompt asks for a "Judgment calls" list
in which each call names its source or says it rests on judgment, no
reference found, after one consult of the ``references`` helper
(:func:`funhouse_agent.deep.prompt.judgment_calls_section`). In live waves
w13 and w13b the F51j answer skipped that step twice: five calls, no
source for any, no consult, and "I did not look up any reference values"
(``module_work/live_smoke/runs/w13b-judgment-on/TRACE_REVIEW.md`` §2.2, §8
fix 1). More prompt wording had not moved it, so this check is code.

WHEN IT ACTS (:meth:`TurnJudgmentCheck.feedback`). Both hold:

* the answer has a "Judgment calls" list (:mod:`funhouse_agent.
  answer_sections`, the same list the live smoke harness reads);
* at least one call in it names no source (:func:`has_source`). A call that
  says it rests on judgment with no reference found
  (:func:`says_none_found`) is exempt only after a consult, that is once a
  ``references`` task ran in the turn (the primary's ``task`` calls, as the
  turn's stream carries them: :meth:`TurnJudgmentCheck.saw`): with none,
  the words cannot be true.

WHAT IT DOES. It sends the answer back ONCE, naming those calls, before the
answer goes out; the reply after that note is the answer, as with
``cited_documents``' retry arm. The note asks for what the turn still lacks
(``reason``):

* ``no_consult``: no ``references`` task ran, so it asks for one consult
  covering those calls, each answered with a source or "no reference
  found";
* ``after_consult`` (GM-187; live wave w14 F18j, 2 of 5 calls with no
  source after the consult): a consult ran, so it asks for no other; each
  named call gets the source the consult found for it or the words "rests
  on judgment, no reference found".

Never a second time in a turn, whichever note went: the reply after the
note goes out as written. The answer itself is never changed.

THE SOURCE TEST (:func:`has_source`). Permissive and simple on purpose: a
call the test wrongly calls sourced costs nothing, one it wrongly calls
unsourced costs a consult. A call has a source when its text:

1. gives a locator with a number: ``§``, a section, equation, table,
   figure, page, chapter, appendix, article, clause or paragraph number
   ("§5.2.2", "Eq. 5-14", "Table 5-8", "Figs. 6-16", "p. 23", "Ch. 5");
2. or cites a work by author and year ("Jaky (1944)", "Duncan & Wright
   (2014)"), or a document with an article or section number ("AASHTO LRFD
   3.11.5"), as :func:`funhouse_agent.cited_documents.find_citations` finds
   them;
3. or says it is a tool's ``choices`` basis (the word "basis" or
   "choices") and names a published document or work there ("basis: FHWA
   NHI-06-088").

A bare document name with none of these ("GEC-6 recommends it") is not a
source. Whether the cited thing was looked up is ``cited_documents``' check,
not this one.

THE "NO REFERENCE FOUND" WORDS (:func:`says_none_found`): the call names
judgment ("rests on judgment", "engineering judgment") and says no
reference, source or basis was found ("no reference found", "found no
reference"). "Rests on judgment" alone is not enough: in w13 every answer
took that wording without looking. And the words count only in a turn
where a ``references`` task ran: w13 F51j wrote "I found no reference for
it" having looked for none.

RECORDING. Whenever an answer has the list, the check is kept in the turn's
``activity.jsonl`` as a :data:`RECORD_EVENT` event: ``action`` is
``fed_back`` when the answer was sent back, else ``none``, with
``n_calls``, ``unsourced`` (each such call's first line),
``references_ran`` and ``reason``. The live smoke harness counts the
``fed_back`` events and the calls an answer still sends out with no source
(``module_work/live_smoke/detectors.py``, detector (l)).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from funhouse_agent.answer_sections import items, section

#: The activity-log event a check is recorded under.
RECORD_EVENT = "judgment_check"
#: How the note to the model opens.
NOTE_PREFIX = "[Judgment-call check]"
#: The status line shown while the answer is written again.
RETRY_STATUS = ("judgment-call check: calls in the answer name no source and "
                "no references task ran; it is being written again after one "
                "consult")
#: The status line when the consult ran and calls still name no source.
RETRY_STATUS_AFTER_CONSULT = (
    "judgment-call check: calls in the answer still name no source after "
    "the consult; it is being written again with a source or \"no reference "
    "found\" for each")
#: Why a note went (the ``reason`` of :func:`check`).
NO_CONSULT = "no_consult"
AFTER_CONSULT = "after_consult"
#: The helper the note asks for (``task`` with this ``subagent_type``).
REFERENCES_HELPER = "references"
#: The scope of :mod:`funhouse_agent.answer_sections` the list is found by.
SCOPE = "judgment_calls"

#: Most calls named in a note or a record, and the length of each.
MAX_LISTED = 20
MAX_CALL_CHARS = 160

#: A locator with a number (rule 1 of the source test).
_LOCATOR = re.compile(
    r"§\s*\(?[A-Z]?\d"
    r"|\b(?:sec(?:tion|t)?s?|eqs?|eqns?|equations?|tables?|figs?|figures?"
    r"|chapters?|appendix|appendices|articles?|clauses?|paras?"
    r"|paragraphs?|pages?)\.?[ \t]*\(?[A-Z]?[-.]?\d"
    r"|\b(?:pp?|ch|chs|app|apps|art|arts)\.[ \t]*\(?[A-Z]?[-.]?\d",
    re.IGNORECASE)
#: Rule 3: a call that says it gives a tool's ``choices`` basis.
_BASIS = re.compile(r"\b(?:basis|bases|choices)\b", re.IGNORECASE)
#: The two halves of "rests on judgment, no reference found".
_JUDGMENT = re.compile(r"\bjudge?ment\b", re.IGNORECASE)
_NONE_FOUND = re.compile(
    r"\bno\s+(?:[\w-]+\s+){0,3}?(?:references?|sources?|bases|basis)\b"
    r"[^.;\n]{0,40}?\b(?:found|located)\b"
    r"|\b(?:found|located)\s+no\s+(?:[\w-]+\s+){0,3}?"
    r"(?:references?|sources?|bases|basis)\b",
    re.IGNORECASE)
_MARKER = re.compile(r"^[ \t]*(?:\d+[.)]|[-*+•]|\|)[ \t]*")


def has_source(call: str) -> bool:
    """Whether a judgment call names its source (the rules in this module's
    docstring)."""
    text = str(call or "")
    if _LOCATOR.search(text):
        return True
    try:
        from funhouse_agent.cited_documents import find_citations
        cited = find_citations(text)
    except Exception:  # noqa: BLE001 - the locator rule stands alone
        cited = []
    if any(c.kind == "work" or c.section for c in cited):
        return True
    return bool(cited) and bool(_BASIS.search(text))


def says_none_found(call: str) -> bool:
    """Whether a call says it rests on judgment with no reference found."""
    text = str(call or "")
    return bool(_JUDGMENT.search(text)) and bool(_NONE_FOUND.search(text))


def call_head(call: str) -> str:
    """A call as a note names it: its first line, without the list marker
    or emphasis, cut to :data:`MAX_CALL_CHARS`."""
    first = next((ln for ln in str(call or "").splitlines() if ln.strip()),
                 "")
    text = _MARKER.sub("", first).replace("**", "").replace("__", "")
    text = re.sub(r"\s+", " ", text.strip(" |")).strip()
    if len(text) > MAX_CALL_CHARS:
        text = text[:MAX_CALL_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + " ..."
    return text


def judgment_calls(answer: str) -> Optional[List[str]]:
    """The calls in the answer's "Judgment calls" list, or ``None`` when it
    has no such list."""
    part = section(str(answer or ""), SCOPE)
    if part is None:
        return None
    return items(part)


def unsourced_calls(calls: Iterable[str],
                    references_ran: bool = False) -> List[str]:
    """The calls that name no source; after a consult (``references_ran``)
    a call that says none was found is not one of them."""
    return [c for c in calls if not has_source(c)
            and not (references_ran and says_none_found(c))]


def check(answer: str, references_ran: bool = False) -> Dict[str, Any]:
    """The check of one answer: ``has_list``, ``n_calls``, ``unsourced``
    (each such call's head), ``references_ran``, ``reason``
    (:data:`NO_CONSULT` or :data:`AFTER_CONSULT`) and ``acts`` (whether
    the check would send it back)."""
    reason = AFTER_CONSULT if references_ran else NO_CONSULT
    calls = judgment_calls(answer)
    if calls is None:
        return {"has_list": False, "n_calls": 0, "unsourced": [],
                "references_ran": bool(references_ran), "reason": reason,
                "acts": False}
    bare = [call_head(c) for c in unsourced_calls(calls, references_ran)]
    return {"has_list": True, "n_calls": len(calls),
            "unsourced": bare[:MAX_LISTED], "n_unsourced": len(bare),
            "references_ran": bool(references_ran), "reason": reason,
            "acts": bool(bare)}


#: What a call can name (both notes).
_SOURCE_WORDS = ("(a reference with its section, page, table, figure or "
                 "equation; an author and year; or a tool's `choices` "
                 "basis)")


def note_for_model(result: Dict[str, Any]) -> Optional[str]:
    """The note the model is given once, or ``None``."""
    if not result.get("acts"):
        return None
    lines = "\n".join(f"- {c}" for c in result.get("unsourced") or [])
    held = ("The answer you wrote is held back: the user will not see it. ")
    again = ("write your whole answer to the user's request again, as they "
             "will read it: it replaces the one held back. ")
    if result.get("reason") == AFTER_CONSULT:
        return (f"{NOTE_PREFIX} Before your answer goes to the user, the "
                "app read its \"Judgment calls\" list. The `references` "
                "helper was consulted this turn, and these calls still name "
                f"no source {_SOURCE_WORDS}:\n" + lines + "\n" + held +
                "Do not ask the helper again. Using what the consult and "
                "your tools returned, " + again + "Give each of these calls "
                "the source the consult found for it, by name and section "
                "or page, or say it rests on judgment, no reference found. "
                "This note comes once.")
    return (f"{NOTE_PREFIX} Before your answer goes to the user, the app "
            "read its \"Judgment calls\" list. These calls name no source "
            f"{_SOURCE_WORDS}, "
            "and no `references` task ran this turn, so none can yet say "
            "that no reference was found:\n" + lines + "\n" + held +
            "Ask the `references` helper once, in one task, about all of "
            "these calls together, one line each, asking only for its "
            "source or \"no reference found\". Then " + again + "Give each "
            "call the source the consult found, or say it rests on "
            "judgment, no reference found. This note comes once.")


def _tool_calls(message: Any) -> List[Dict[str, Any]]:
    """A model message's tool calls as ``{"name", "args"}`` (a LangChain
    message or an OpenAI-style dict)."""
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    out = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = c.get("name") or fn.get("name")
        args = c.get("args") if "args" in c else fn.get("arguments")
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except ValueError:
                args = {}
        out.append({"name": name, "args": args if isinstance(args, dict)
                    else {}})
    return out


def mode_for(agent: Any) -> str:
    """``"retry"`` when ``GEOTECH_JUDGMENT_CALLS`` is on and ``agent``'s
    build asked for the list with a references helper to consult
    (``agent.geotech_judgment_consult``, set by ``build_deep_agent``); else
    ``"off"``."""
    try:
        from funhouse_agent import review_flags
        if review_flags.judgment_calls() and \
                getattr(agent, "geotech_judgment_consult", False) is True:
            return "retry"
    except Exception:  # noqa: BLE001
        pass
    return "off"


class TurnJudgmentCheck:
    """The judgment-call check of one turn's answer: :meth:`saw` notes the
    turn's ``references`` tasks as the stream carries them, :meth:`feedback`
    sends the answer back once (``retry`` mode), :meth:`finish` records the
    answer as it goes out (and never changes it). Never raises."""

    def __init__(self, mode: str = "off",
                 log: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.mode = mode
        self.log = log
        self.fed_back = False
        self.references_ran = False
        self.reason: Optional[str] = None

    @property
    def retry_status(self) -> str:
        """The status line for the note that went (the host shows it)."""
        return (RETRY_STATUS_AFTER_CONSULT if self.reason == AFTER_CONSULT
                else RETRY_STATUS)

    def saw(self, messages: Iterable[Any]) -> None:
        """Note any ``references`` task among the turn's new messages."""
        try:
            for msg in messages or ():
                for call in _tool_calls(msg):
                    if call["name"] == "task" and str(
                            call["args"].get("subagent_type") or "") \
                            == REFERENCES_HELPER:
                        self.references_ran = True
        except Exception:  # noqa: BLE001 - never the turn
            pass

    def _record(self, result: Dict[str, Any], action: str) -> None:
        if self.log is None or not result.get("has_list"):
            return
        try:
            self.log({k: v for k, v in result.items() if k != "acts"}
                     | {"where": "answer", "mode": self.mode,
                        "action": action})
        except Exception:  # noqa: BLE001
            pass

    def feedback(self, answer: str) -> Optional[str]:
        """The note to give the model before this answer goes out, once a
        turn, in ``retry`` mode; else ``None``."""
        if self.mode != "retry" or self.fed_back or not str(answer).strip():
            return None
        try:
            result = check(answer, self.references_ran)
            note = note_for_model(result)
        except Exception:  # noqa: BLE001 - never the turn
            return None
        if note:
            self.fed_back = True
            self.reason = result.get("reason")
            self._record(result, "fed_back")
        return note

    def finish(self, answer: str) -> str:
        """``answer`` as it goes out, unchanged; the check is recorded when
        it has a "Judgment calls" list."""
        try:
            if str(answer or "").strip():
                self._record(check(answer, self.references_ran), "none")
        except Exception:  # noqa: BLE001 - never the turn
            pass
        return answer


__all__ = ["RECORD_EVENT", "NOTE_PREFIX", "RETRY_STATUS",
           "RETRY_STATUS_AFTER_CONSULT", "NO_CONSULT", "AFTER_CONSULT",
           "REFERENCES_HELPER", "SCOPE", "has_source", "says_none_found",
           "call_head", "judgment_calls", "unsourced_calls", "check",
           "note_for_model", "mode_for", "TurnJudgmentCheck"]
