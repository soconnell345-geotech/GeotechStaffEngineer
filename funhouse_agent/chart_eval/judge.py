"""Grading an answer against Chartography's golden answer, with a model.

Chartography's golden answers are free text with expert-set acceptable
ranges ("between 41 and 44 mph"), so they are graded by a model that sees the
question, the golden answer and the response, never the chart. Surge's own
harness does this with one judge call per answer (``src/chartography/
scorer.py``, Gemini 3.5 Flash on their leaderboard).

The prompt. A benchmark's own grading prompt, kept in the private data
folder, is used word for word: CharXiv's per answer type
(``judge_prompts.json``, written by :mod:`funhouse_agent.chart_eval.charxiv`;
placeholders ``<|question|>``, ``<|ground_truth|>``, ``<|response|>``), or
Surge's for Chartography (``judge_prompt.txt``; placeholders
``{{ rubric.* }}``; their repository carries no licence, so it is never
copied into this package). One line is added after it, asking for the JSON
alone, because the Foundry routes do not all take a response schema.
Without one, :data:`OWN_TEMPLATE` states the grading rules in this package's
own words. Every run records which prompt graded it.

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

#: The grading rules of the dataset card, in this package's own words.
OWN_TEMPLATE = f"""You are grading a response to a chart-reading question. Compare it with the expert's golden answer. You do not see the chart; grade only what the response says against the golden answer.

Question: {Q_SLOT}

Response: {R_SLOT}

Golden answer: {G_SLOT}

How to grade:
1. Find the answer the response commits to. If it gives one answer and later changes it, use the last one. Ignore its working, hedging and formatting.
2. Compare it with the golden answer:
   - A number with an acceptable range given: correct only if the number lies inside the range (the ends count as inside). Being close but outside is wrong; a more precise number inside the range is right.
   - A number with no range given (a value printed on the chart): correct only if it is that value, allowing trivial rounding and equivalent notation ("$4.2M" = "4.2 million").
   - One or more categories (an item, a list, a top three): correct only if it names exactly the right set, nothing missing and nothing extra. Order matters only when the question asks for an order or a ranking. Small wording differences that clearly name the same item are fine.
   - Several parts: every part must be correct.
   - If the response says the question cannot be answered, it is wrong, unless the golden answer itself says it cannot be determined.
3. Be strict about the substance and lenient about units written differently, phrasing and extra explanation.

Return only a JSON object, with no other text:
{{"scores": [{{"criterion_id": "criterion_1", "extracted_answer": "<the response's final answer>", "score": 1, "rationale": "<one or two sentences>"}}]}}
One entry per part of the golden answer, in order ("criterion_1", "criterion_2", ...). "score" is 1 when that part matches and 0 when it does not; no other values."""

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
