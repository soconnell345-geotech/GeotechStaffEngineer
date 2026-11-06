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

A verdict is either Surge's shape (``{"scores": [{"score": 0|1, ...}]}``,
every part must score 1) or CharXiv's (``{"extracted_answer": ...,
"score": 0|1}``).
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


def parse_verdict(text: str) -> List[Dict[str, Any]]:
    """The judge's ``scores`` list, each entry with a 0/1 ``score``; raises
    :class:`VerdictError` on anything else."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\A```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\Z", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except ValueError:
        # a model that wrote a sentence round the object: take the object
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise VerdictError("no JSON object in the judge's reply")
        try:
            payload = json.loads(cleaned[start:end + 1])
        except ValueError as exc:
            raise VerdictError(f"invalid JSON: {exc}")
    if not isinstance(payload, dict):
        raise VerdictError("the verdict is not a JSON object")
    if "scores" not in payload and "score" in payload:
        # CharXiv's shape: one answer, one score
        payload = {"scores": [{
            "criterion_id": "answer",
            "extracted_answer": payload.get("extracted_answer",
                                            payload.get("extract_answer", "")),
            "score": payload.get("score"), "rationale": ""}]}
    entries = payload.get("scores")
    if not isinstance(entries, list) or not entries:
        raise VerdictError("the verdict has no 'scores' list")
    out = []
    for e in entries:
        if not isinstance(e, dict) or "score" not in e:
            raise VerdictError("a 'scores' entry has no score")
        raw = e.get("score")
        if isinstance(raw, bool):
            raise VerdictError("a score is a boolean, not 0 or 1")
        try:
            val = float(str(raw).strip())
        except ValueError:
            raise VerdictError(f"a score is not a number: {raw!r}")
        if val not in (0.0, 1.0):
            raise VerdictError(f"a score is {raw!r}, not 0 or 1")
        out.append({"criterion_id": str(e.get("criterion_id", "")),
                    "extracted_answer": str(e.get("extracted_answer", "")),
                    "score": int(val),
                    "rationale": str(e.get("rationale", ""))})
    return out


def judge_answer(judge_model: Any, question: str, golden: str, response: str,
                 template: Optional[str] = None, retries: int = 4,
                 label: Optional[str] = None) -> Dict[str, Any]:
    """Grade ``response``: ``{"passed": True/False/None, "scores": [...],
    "prompt": <label, else "data folder" or "own">, "attempts": n,
    "raw": <last reply>}``.
    ``passed`` is ``None`` (unscored) when the judge never gave a usable
    verdict, or ``response`` is ``None``. An empty response is wrong
    without a judge call."""
    which = label or ("data folder" if template else "own")
    if response is None:
        return {"passed": None, "scores": [], "prompt": which, "attempts": 0,
                "error": "no response to grade"}
    if not str(response).strip():
        return {"passed": False, "scores": [], "prompt": which,
                "attempts": 0, "note": "empty answer: wrong, not judged"}
    from langchain_core.messages import HumanMessage
    prompt = build_prompt(question, golden, str(response), template)
    raw, err = "", None
    for attempt in range(1, int(retries) + 2):
        try:
            reply = judge_model.invoke([HumanMessage(content=prompt)])
            raw = _text_of(getattr(reply, "content", reply))
            scores = parse_verdict(raw)
        except VerdictError as exc:
            err = str(exc)
            continue
        except Exception as exc:                          # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            continue
        return {"passed": all(s["score"] == 1 for s in scores),
                "scores": scores, "prompt": which, "attempts": attempt,
                "raw": raw}
    return {"passed": None, "scores": [], "prompt": which,
            "attempts": int(retries) + 1, "raw": raw, "error": err}


__all__ = ["OWN_TEMPLATE", "build_prompt", "parse_verdict", "judge_answer",
           "VerdictError"]
