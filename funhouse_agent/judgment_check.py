"""The turn-end judgment-call check (BACKLOG GM-174), behind
``GEOTECH_JUDGMENT_CALLS``.

WHY. With the switch on, the geotech prompt asks for a "Judgment calls" list
in which each call names its source or says it rests on judgment, no
reference found, after one consult of the ``references`` helper
(:func:`funhouse_agent.deep.prompt.judgment_calls_section`). In live waves
w13 and w13b the F51j answer skipped that step twice: five calls, no
source for any, no consult, and "I did not look up any reference values"
(``module_work/live_smoke/runs/w13b-judgment-on/TRACE_REVIEW.md`` §2.2, §8
fix 1). More prompt wording had not moved it, so this check is code.

WHEN IT ACTS (:meth:`TurnJudgmentCheck.feedback`). All three hold:

* the answer has a "Judgment calls" list (:mod:`funhouse_agent.
  answer_sections`, the same list the live smoke harness reads);
* at least one call in it names no source (:func:`has_source`) and does not
  say it rests on judgment with no reference found (:func:`says_none_found`);
* no ``references`` task ran in the turn (the primary's ``task`` calls, as
  the turn's stream carries them: :meth:`TurnJudgmentCheck.saw`).

WHAT IT DOES. It sends the answer back ONCE, naming those calls and asking
for one consult covering them, before the answer goes out; the reply after
that note is the answer, as with ``cited_documents``' retry arm. Never a
second time in a turn: the reply after the note goes out as written. The
answer itself is never changed.

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
took that wording without looking.

RECORDING. Whenever an answer has the list, the check is kept in the turn's
``activity.jsonl`` as a :data:`RECORD_EVENT` event: ``action`` is
``fed_back`` when the answer was sent back, else ``none``, with
``n_calls``, ``unsourced`` (each such call's first line), and
``references_ran``. The live smoke harness counts the ``fed_back`` events
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


def unsourced_calls(calls: Iterable[str]) -> List[str]:
    """The calls that name no source and do not say none was found."""
    return [c for c in calls if not has_source(c) and not says_none_found(c)]


def check(answer: str, references_ran: bool = False) -> Dict[str, Any]:
    """The check of one answer: ``has_list``, ``n_calls``, ``unsourced``
    (each such call's head), ``references_ran`` and ``acts`` (whether the
    check would send it back)."""
    calls = judgment_calls(answer)
    if calls is None:
        return {"has_list": False, "n_calls": 0, "unsourced": [],
                "references_ran": bool(references_ran), "acts": False}
    bare = [call_head(c) for c in unsourced_calls(calls)]
    return {"has_list": True, "n_calls": len(calls),
            "unsourced": bare[:MAX_LISTED], "n_unsourced": len(bare),
            "references_ran": bool(references_ran),
            "acts": bool(bare) and not references_ran}


def note_for_model(result: Dict[str, Any]) -> Optional[str]:
    """The note the model is given once, or ``None``."""
    if not result.get("acts"):
        return None
    lines = "\n".join(f"- {c}" for c in result.get("unsourced") or [])
    return (f"{NOTE_PREFIX} Before your answer goes to the user, the app "
            "read its \"Judgment calls\" list. These calls name no source "
            "(a reference with its section, page, table, figure or "
            "equation; an author and year; or a tool's `choices` basis) "
            "and do not say they rest on judgment with no reference found, "
            "and no `references` task ran this turn:\n" + lines + "\n"
            "The answer you wrote is held back: the user will not see it. "
            "Ask the `references` helper once, in one task, about all of "
            "these calls together. Then write your whole answer to the "
            "user's request again, as they will read it: it replaces the "
            "one held back. Give each call the source the consult found, or "
            "say it rests on judgment, no reference found. This note comes "
            "once.")


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

    retry_status = RETRY_STATUS

    def __init__(self, mode: str = "off",
                 log: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.mode = mode
        self.log = log
        self.fed_back = False
        self.references_ran = False

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
           "REFERENCES_HELPER", "SCOPE", "has_source", "says_none_found",
           "call_head", "judgment_calls", "unsourced_calls", "check",
           "note_for_model", "mode_for", "TurnJudgmentCheck"]
