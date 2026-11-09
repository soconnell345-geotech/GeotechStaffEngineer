"""Grading an answer against a benchmark's golden answer, with a model.

The golden answers are free text (a term, a phrase or a number; some carry
an acceptable range), so they are graded by a model that sees the question,
the golden answer and the response, never the chart: one judge call per
answer, as CharXiv's and Surge's own harnesses grade.

The prompt. A benchmark's own grading prompt, kept in the private data
folder, is used word for word: CharXiv's per answer type
(``judge_prompts.json``, written by :mod:`funhouse_agent.chart_eval.charxiv`;
placeholders ``<|question|>``, ``<|ground_truth|>``, ``<|response|>``), or
Surge's for Chartography (``judge_prompt.txt``; placeholders
``{{ rubric.* }}``; their repository carries no licence, so it is never
copied into this package). One line is added after it, asking for the JSON
alone, because the Foundry routes do not all take a response schema.
Without one, :data:`OWN_TEMPLATE` grades: a short prompt of this package's
own, written from what a grader of a short answer needs, not from either
benchmark's prompt. Every run records which prompt graded it.

The verdict. :func:`parse_verdict` takes the JSON object out of the
judge's reply and reads one decision from it, or one per part when the
object lists parts under ``scores`` (the form ``judge_prompt.txt`` asks
for). A decision is ``correct`` (this package's prompt) or ``score``
(CharXiv's), given as true/false or 1/0; the answer is right only when every
part is.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

#: The placeholders, as Surge's template spells them...
Q_SLOT = "{{ rubric.response_generation_prompt }}"
R_SLOT = "{{ rubric.response_to_grade }}"
G_SLOT = "{{ rubric.all_criteria }}"
#: ...and as CharXiv's grading prompts spell them.
CHARXIV_SLOTS = ("<|question|>", "<|response|>", "<|ground_truth|>")

#: The grading prompt when the data folder holds none: this package's own
#: (module docstring). Its reply is one decision, ``correct``.
OWN_TEMPLATE = f"""Decide whether an answer to a question about a chart is right by comparing it with the reference answer. You are not shown the chart and do not need it: do not try to answer the question yourself.

Question:
{CHARXIV_SLOTS[0]}

Reference answer:
{CHARXIV_SLOTS[2]}

Answer to check:
{CHARXIV_SLOTS[1]}

Judge the result the answer settles on; its working and its wording do not count. A number is the same if it has the same value written another way (other units or notation, or rounded to the reference's precision); where the reference gives a range, any value inside it is the same. Where the reference names one or more things, the answer must name the same ones. Where the reference has more than one part, every part must match.

Reply with this JSON object and nothing else:
{{"final_answer": "<the result the answer settles on>", "correct": true, "reason": "<one sentence>"}}
with "correct" set to true or false."""

#: Added after a benchmark's own template: Surge's expects a response
#: schema and CharXiv's a JSON mode, which the Foundry routes do not all
#: take.
JSON_ONLY = "\n\nReply with the JSON object only, no other text."


class VerdictError(ValueError):
    """The judge's reply is not a usable verdict."""


def build_prompt(question: str, golden: str, response: str,
                 template: Optional[str] = None) -> str:
    """The judge prompt: ``template`` (the benchmark's own, from the data
    folder) with :data:`JSON_ONLY` after it, else :data:`OWN_TEMPLATE`.
    Each slot is filled once, so a response that happens to contain a
    placeholder's text is not filled in again."""
    tpl = (template + JSON_ONLY) if template else OWN_TEMPLATE
    values = {Q_SLOT: question, R_SLOT: response, G_SLOT: golden,
              CHARXIV_SLOTS[0]: question, CHARXIV_SLOTS[1]: response,
              CHARXIV_SLOTS[2]: golden}
    pattern = re.compile("|".join(re.escape(k) for k in values))
    return pattern.sub(lambda m: values[m.group(0)], tpl)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") in ("text",
                                                           "output_text"):
                parts.append(str(b.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def _outer_object(text: str) -> Dict[str, Any]:
    """The JSON object in the judge's reply: everything from its first
    ``{`` to its last ``}``, so a code fence or a sentence round the object
    does not matter."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise VerdictError("no JSON object in the judge's reply")
    try:
        found = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise VerdictError(f"the judge's JSON does not parse: {exc}")
    if not isinstance(found, dict):
        raise VerdictError("the judge's JSON is not an object")
    return found


#: How a judge may write right and wrong.
_RIGHT = ("1", "1.0", "true")
_WRONG = ("0", "0.0", "false")


def _decision(value: Any, part: int) -> bool:
    """A judge's decision on one part: true/false, or 1/0 as a number or
    as text. Anything else (0.5, "partly", null) is no decision."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return value == 1
    if isinstance(value, str) and value.strip().lower() in _RIGHT + _WRONG:
        return value.strip().lower() in _RIGHT
    raise VerdictError(f"part {part}: {value!r} is neither right (1, true) "
                       "nor wrong (0, false)")


def _first_of(entry: Dict[str, Any], keys) -> str:
    for key in keys:
        if entry.get(key) is not None:
            return str(entry[key])
    return ""


def _part(entry: Any, n: int) -> Dict[str, Any]:
    """One part of a verdict, in this package's record form."""
    if not isinstance(entry, dict):
        raise VerdictError(f"part {n} is not a JSON object")
    if "correct" in entry:
        right = _decision(entry["correct"], n)
    elif "score" in entry:
        right = _decision(entry["score"], n)
    else:
        raise VerdictError(f"part {n} gives no decision")
    return {"part": _first_of(entry, ("criterion_id",)) or str(n),
            "answer": _first_of(entry, ("final_answer", "extracted_answer",
                                        "extract_answer")),
            "correct": right,
            "reason": _first_of(entry, ("reason", "rationale"))}


def parse_verdict(text: str) -> List[Dict[str, Any]]:
    """The judge's decision on each part of the answer, as ``[{"part",
    "answer", "correct", "reason"}]`` (one entry unless the reply lists
    parts under ``scores``); raises :class:`VerdictError` when the reply is
    not a usable verdict."""
    found = _outer_object(str(text or ""))
    listed = found.get("scores")
    if isinstance(listed, list):
        if not listed:
            raise VerdictError("the verdict lists no parts")
        return [_part(entry, n) for n, entry in enumerate(listed, 1)]
    return [_part(found, 1)]


def judge_answer(judge_model: Any, question: str, golden: str, response: str,
                 template: Optional[str] = None, retries: int = 4,
                 label: Optional[str] = None) -> Dict[str, Any]:
    """Grade ``response``: ``{"passed": True/False/None, "parts": [...],
    "prompt": <label, else "data folder" or "own">, "attempts": n,
    "raw": <last reply>}``.
    ``passed`` is ``None`` (unscored) when the judge never gave a usable
    verdict, or ``response`` is ``None``. An empty response is wrong
    without a judge call."""
    which = label or ("data folder" if template else "own")
    if response is None:
        return {"passed": None, "parts": [], "prompt": which, "attempts": 0,
                "error": "no response to grade"}
    if not str(response).strip():
        return {"passed": False, "parts": [], "prompt": which,
                "attempts": 0, "note": "empty answer: wrong, not judged"}
    from langchain_core.messages import HumanMessage
    prompt = build_prompt(question, golden, str(response), template)
    raw, err = "", None
    for attempt in range(1, int(retries) + 2):
        try:
            reply = judge_model.invoke([HumanMessage(content=prompt)])
            raw = _text_of(getattr(reply, "content", reply))
            parts = parse_verdict(raw)
        except VerdictError as exc:
            err = str(exc)
            continue
        except Exception as exc:                          # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            continue
        return {"passed": all(p["correct"] for p in parts),
                "parts": parts, "prompt": which, "attempts": attempt,
                "raw": raw}
    return {"passed": None, "parts": [], "prompt": which,
            "attempts": int(retries) + 1, "raw": raw, "error": err}


__all__ = ["OWN_TEMPLATE", "build_prompt", "parse_verdict", "judge_answer",
           "VerdictError"]
