"""Phase 5 — A/B eval harness comparing the v1 agent vs the v2 deepagents agent.

This runs the project's 68-question evaluation suite
(``funhouse_agent/geotech_test_suite.json``) through BOTH agents and scores the
two runs side by side, so the v5.0 deepagents port can be judged against the v1
ReAct/native agent on the SAME questions and the SAME underlying Anthropic model.

What it records (per question, per agent)
------------------------------------------
A normalized :class:`QAResult` — the final ``answer`` text, the tool-call
``trace`` (each call's ``name`` / ``args`` / ``errored`` flag), ``rounds`` (turns),
``errors`` (dispatch/tool errors), ``latency_s``, and token ``usage`` when the
backend exposes it. The two agents have very different native result shapes, so
:func:`_v1_to_qa` and :func:`_v2_to_qa` adapt each into the common
:class:`QAResult` before scoring — scoring never touches a raw agent result.

The headline metric: P1 hallucination-on-error
-----------------------------------------------
The v5.0 migration exists partly to fix "hallucination on tool error" (the agent
fabricating success after a tool call failed — see CLAUDE.md HANDOFF P1). The
scorer's flagship number is the **P1 hallucination-on-error rate**: the fraction
of questions where the trace contains a tool ERROR but the final answer asserts
success / reports a value with no acknowledgement of the failed step. See
:func:`is_hallucination_on_error` for the precise, documented heuristic and its
known false-positive / false-negative risks.

Correctness auto-scoring (v5.1): the ``expected`` field
--------------------------------------------------------
``geotech_test_suite.json`` is a flat list of ``{id, module, complexity,
question}`` objects. As of v5.1, calc-type questions with fully specified inputs
additionally carry an ``expected`` dict (the answer key)::

    "expected": {
      "values":   [{"name": "q_ult_kPa", "value": 1593, "rtol": 0.15,
                    "alt": [<same quantity in another unit/convention>]}],
      "keywords": ["class D"],          # case-insensitive substrings
      "source":   "v5.1 worktree ground truth (module.method, 2026-06-10)"
    }

The values were produced by RUNNING the engineering modules in the v5.1 worktree
(via ``funhouse_agent.dispatch.call_agent``) — so they grade "does the agent
reproduce the module's own answer", not textbook truth. :func:`default_judge`
grades an answer by numeric extraction: every ``values`` entry must match SOME
number in the final answer within ``rtol`` (of ``value`` or any ``alt``), and
every keyword must appear. A question WITHOUT ``expected`` is **skipped** (not
failed). A custom ``judge_fn(question, qa) -> bool | None`` (None = skip) can
replace the default — see :func:`make_llm_judge` for an opt-in LLM judge.

Sample input files: the ``sample_file`` field + ``{sample_path}`` token
------------------------------------------------------------------------
File-input questions (AGS4 / GEF / DIGGS / DXF / PDF) carry a ``sample_file``
field naming a file bundled under ``funhouse_agent/eval_samples/``, and their
question text contains the literal token ``{sample_path}``. :func:`load_suite`
expands the token to an ABSOLUTE path resolved against the installed
``funhouse_agent`` package (``importlib.resources``, with a ``__file__``
fallback) — so the path is correct from any CWD and from an installed wheel.
NOTE: shipping the samples in a wheel requires a ``[tool.setuptools.package-data]``
entry ``funhouse_agent = ["geotech_test_suite.json", "eval_samples/*"]``.

Offline vs live
---------------
Everything except the two ``--dry-run`` stub agents is offline: the loader, the
adapters, every scorer, the results-JSON writer, and the markdown table render
with no API key and no network. A live A/B run (real ``ChatAnthropic`` calls)
costs real budget and is **never** triggered by the tests — only by an explicit
CLI invocation without ``--dry-run``.

CLI
---
Run from the worktree root with the venv python::

    cd <worktree>
    # OFFLINE — mock both agents, exercise the whole pipeline, no API:
    .venv/Scripts/python.exe -m funhouse_agent.deep.eval_harness --dry-run

    # LIVE, cheap subset — first 5 questions through BOTH real agents:
    .venv/Scripts/python.exe -m funhouse_agent.deep.eval_harness --limit 5 \
        --model claude-sonnet-4-6 --out funhouse_agent/deep/ab_results.json

    # LIVE, full 68x2 — spends real budget; prints a cost caution + count first:
    .venv/Scripts/python.exe -m funhouse_agent.deep.eval_harness \
        --model claude-sonnet-4-6 --out funhouse_agent/deep/ab_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: This file lives in ``funhouse_agent/deep/`` — the suite lives one level up.
_DEEP_DIR = Path(__file__).resolve().parent
_FUNHOUSE_DIR = _DEEP_DIR.parent
SUITE_PATH = _FUNHOUSE_DIR / "geotech_test_suite.json"

#: Default model id for a live run (kept in sync with ``deep/run_live.py``).
DEFAULT_MODEL = "claude-sonnet-4-6"

#: Literal token in a suite question's text that :func:`load_suite` replaces
#: with the resolved absolute path of the question's ``sample_file``.
SAMPLE_PATH_TOKEN = "{sample_path}"


def resolve_sample_path(name: str) -> Path:
    """Resolve a bundled eval-sample filename to an ABSOLUTE path.

    Works from any CWD and from an installed wheel: resolution goes through
    ``importlib.resources`` against the ``funhouse_agent`` package first, then
    falls back to the source-tree layout relative to this file. The path is
    returned even if the file is missing (callers/tests check ``exists()``), so
    a broken packaging setup fails loudly at parse time, not silently here.

    Parameters
    ----------
    name : str
        Bare filename under ``funhouse_agent/eval_samples/`` (e.g.
        ``"sample_cpt.gef"``).
    """
    try:
        from importlib.resources import files
        p = Path(str(files("funhouse_agent") / "eval_samples" / name))
        if p.exists():
            return p
    except Exception:
        pass
    return _FUNHOUSE_DIR / "eval_samples" / name


# ===========================================================================
# Suite loading
# ===========================================================================

def load_suite(path: Path = SUITE_PATH) -> list[dict]:
    """Load the geotech eval suite as a list of question dicts.

    For questions carrying a ``sample_file`` field, the literal
    ``{sample_path}`` token in the question text is expanded to the sample's
    resolved ABSOLUTE path (see :func:`resolve_sample_path`), so the agent
    under test receives a real, openable path regardless of CWD or install
    location.

    Parameters
    ----------
    path : Path
        Path to ``geotech_test_suite.json``. Defaults to the packaged suite.

    Returns
    -------
    list of dict
        Each ``{"id", "module", "complexity", "question"}``, plus optional
        ``expected`` (answer key — see :func:`default_judge`) and
        ``sample_file`` fields.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of questions, got {type(data)}")
    for q in data:
        if isinstance(q, dict) and q.get("sample_file"):
            sample = str(resolve_sample_path(q["sample_file"]))
            q["question"] = (q.get("question") or "").replace(
                SAMPLE_PATH_TOKEN, sample)
    return data


def suite_has_expected_answers(questions: list[dict]) -> bool:
    """True if ANY suite question carries an ``expected``/``answer`` field.

    As of v5.1 the packaged suite DOES carry ``expected`` dicts on calc-type
    questions (worktree-module ground truth), so this returns ``True`` for the
    packaged suite and unlocks the default-judge grading in
    :func:`score_correctness`. Questions without ``expected`` are skipped, not
    failed.
    """
    keys = {"expected", "expected_answer", "answer", "reference_answer", "expected_values"}
    return any(set(q) & keys for q in questions if isinstance(q, dict))


# ===========================================================================
# The common per-question result shape (both agents adapt INTO this)
# ===========================================================================

@dataclass
class ToolCallRecord:
    """One tool call in a question's trace, normalized across both agents."""

    name: str
    args: dict = field(default_factory=dict)
    errored: bool = False
    #: Short error/result note (the error message, or a truncated result).
    note: str = ""


@dataclass
class QAResult:
    """One agent's normalized result for ONE suite question.

    Both the v1 ``AgentResult`` and the v2 deepagents message list are adapted
    into this single shape (:func:`_v1_to_qa` / :func:`_v2_to_qa`) so the scorers
    are agent-agnostic.
    """

    qid: str
    module: str
    agent: str  # "v1" | "v2"
    answer: str = ""
    trace: list[ToolCallRecord] = field(default_factory=list)
    rounds: int = 0
    #: Tool/dispatch errors as dicts (kept loosely-typed for JSON round-trip).
    errors: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    #: ``{"input_tokens", "output_tokens", "total_tokens"}`` or ``None`` if the
    #: backend does not expose usage (the v1 ClaudeEngine does not).
    usage: Optional[dict] = None
    #: A hard exception string if the ask itself raised (vs a soft tool error).
    exception: Optional[str] = None

    @property
    def n_tool_errors(self) -> int:
        """Number of tool calls in the trace flagged as errored."""
        return sum(1 for tc in self.trace if tc.errored)

    @property
    def has_tool_error(self) -> bool:
        """True if ANY tool call errored or the error log is non-empty."""
        return self.n_tool_errors > 0 or bool(self.errors)

    @property
    def total_tokens(self) -> Optional[int]:
        """The authoritative TOTAL token spend for this question, or ``None``.

        Reads ``usage["total_tokens"]`` (set from the callback aggregator on the
        v2 path — sub-agents included — falling back to the trace-summed usage).
        ``None`` when the backend exposes no usage (e.g. the v1 ClaudeEngine).
        """
        if isinstance(self.usage, dict):
            tot = self.usage.get("total_tokens")
            return int(tot) if tot is not None else None
        return None

    def to_dict(self) -> dict:
        """JSON-serializable dict (mirrors docs/geotech_test_suite_results.json)."""
        d = asdict(self)
        d["n_tool_errors"] = self.n_tool_errors
        d["has_tool_error"] = self.has_tool_error
        # Surface the authoritative per-question token total (and split, when the
        # backend provides one) at the top level so JSON/markdown consumers do
        # not have to dig into ``usage``.
        d["total_tokens"] = self.total_tokens
        if isinstance(self.usage, dict):
            d["input_tokens"] = self.usage.get("input_tokens")
            d["output_tokens"] = self.usage.get("output_tokens")
        else:
            d["input_tokens"] = None
            d["output_tokens"] = None
        return d


# ===========================================================================
# Error detection shared by both adapters
# ===========================================================================

def _result_is_error(result_text: str) -> tuple[bool, str]:
    """Classify a tool RESULT string as an error, mirroring the v1 agent.

    The v1 agent flags a dispatch error when the tool's JSON result is a dict
    carrying an ``"error"`` key (see ``agent.py`` native loop). We apply the
    SAME rule to both agents so the comparison is fair, and additionally treat a
    non-JSON result that literally starts with ``Error``/``Traceback`` as an
    error (covers tools that return a bare error string).

    Returns
    -------
    (is_error, note)
        ``note`` is the extracted error message (or a short result preview).
    """
    if not result_text:
        return False, ""
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        head = result_text.strip()[:200]
        low = head.lower()
        if low.startswith("error") or low.startswith("traceback"):
            return True, head
        return False, ""
    if isinstance(data, dict) and "error" in data:
        return True, str(data["error"])[:300]
    return False, ""


# ===========================================================================
# v1 adapter: AgentResult -> QAResult
# ===========================================================================

def _v1_to_qa(qid: str, module: str, result, latency_s: float,
              exception: Optional[str] = None) -> QAResult:
    """Adapt a v1 :class:`~funhouse_agent.react_support.AgentResult` into QAResult.

    v1 logs each tool call as ``{round, tool_name, arguments, result_preview}``
    and each dispatch error as ``{round, type, tool, arguments, message}``. A
    call is marked ``errored`` if its ``result_preview`` parses as an error OR an
    error-log entry matches its round+tool. v1 exposes no token usage, so
    ``usage`` stays ``None``.
    """
    qa = QAResult(qid=qid, module=module, agent="v1", exception=exception)
    if result is None:
        return qa
    qa.answer = getattr(result, "answer", "") or ""
    qa.rounds = getattr(result, "rounds", 0) or 0
    qa.latency_s = latency_s
    errors = list(getattr(result, "errors", []) or [])
    qa.errors = errors

    # Build a quick lookup of (round, tool) -> error message from the error log.
    err_by_key = {}
    for e in errors:
        if isinstance(e, dict):
            err_by_key[(e.get("round"), e.get("tool"))] = e.get("message", "")

    for tc in getattr(result, "tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get("tool_name", "?")
        args = tc.get("arguments", {}) or {}
        preview = tc.get("result_preview", "") or ""
        is_err, note = _result_is_error(preview)
        logged = err_by_key.get((tc.get("round"), name))
        if logged:
            is_err = True
            note = note or logged
        qa.trace.append(ToolCallRecord(name=name, args=args, errored=is_err, note=note))
    return qa


# ===========================================================================
# v2 adapter: deepagents message list -> QAResult
# ===========================================================================

def _v2_iter_messages(result) -> list:
    """Pull the message list out of a deepagents/LangGraph invoke result."""
    if isinstance(result, dict):
        return result.get("messages", []) or []
    return getattr(result, "messages", []) or []


def _v2_content_to_text(content) -> str:
    """Flatten a LangChain message content (str or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return "" if content is None else str(content)


def _v2_final_text(messages) -> str:
    """Text of the last assistant (AIMessage) message with non-empty content."""
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai" or type(m).__name__ == "AIMessage":
            text = _v2_content_to_text(getattr(m, "content", ""))
            if text.strip():
                return text
    return ""


def _v2_is_tool_message(msg) -> bool:
    """True if ``msg`` is a LangChain ToolMessage (a tool RESULT)."""
    return getattr(msg, "type", None) == "tool" or type(msg).__name__ == "ToolMessage"


def _v2_usage(messages) -> Optional[dict]:
    """Sum ``usage_metadata`` across all AIMessages (the v2 token usage).

    Returns ``None`` if no message exposes usage (so the field is honestly
    absent rather than a misleading zero).
    """
    inp = out = tot = 0
    seen = False
    for m in messages:
        um = getattr(m, "usage_metadata", None)
        if isinstance(um, dict):
            seen = True
            inp += int(um.get("input_tokens", 0) or 0)
            out += int(um.get("output_tokens", 0) or 0)
            tot += int(um.get("total_tokens", 0) or 0)
    if not seen:
        return None
    if tot == 0:
        tot = inp + out
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def _sum_callback_usage(callback_usage) -> Optional[dict]:
    """Sum a ``UsageMetadataCallbackHandler.usage_metadata`` dict to one total.

    The callback aggregates per model name:
    ``{model_name: {"input_tokens", "output_tokens", "total_tokens", ...}}``.
    Because a deepagents run fans out to sub-agents (references / reviewer) that
    may use the SAME or a DIFFERENT model, and ALL of their model calls propagate
    through the LangGraph run's callbacks, this single dict already includes the
    sub-agent spend. We sum across every model entry to get the run total.

    Parameters
    ----------
    callback_usage : dict or None
        ``cb.usage_metadata`` from :func:`get_usage_metadata_callback`.

    Returns
    -------
    dict or None
        ``{"input_tokens", "output_tokens", "total_tokens"}`` summed across all
        models, or ``None`` if the callback recorded nothing (so the caller can
        fall back to the trace-derived usage and never regress).
    """
    if not callback_usage:
        return None
    inp = out = tot = 0
    for um in callback_usage.values():
        if not isinstance(um, dict):
            continue
        inp += int(um.get("input_tokens", 0) or 0)
        out += int(um.get("output_tokens", 0) or 0)
        tot += int(um.get("total_tokens", 0) or 0)
    if tot == 0:
        tot = inp + out
    if tot == 0 and inp == 0 and out == 0:
        return None
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def _v2_to_qa(qid: str, module: str, result, latency_s: float,
              exception: Optional[str] = None) -> QAResult:
    """Adapt a v2 deepagents invoke result (message list) into QAResult.

    Walks the LangGraph message trace: ``AIMessage.tool_calls`` are the calls
    (``{name, args, id}``); ``ToolMessage`` entries are the RESULTS. A call is
    matched to its result by ``tool_call_id`` and marked ``errored`` if the
    result JSON carries an ``"error"`` key (same rule as v1) or the ToolMessage
    ``status == "error"``. ``rounds`` is counted as the number of AIMessages that
    issued at least one tool call (turns). Token usage is summed from
    ``usage_metadata``.
    """
    qa = QAResult(qid=qid, module=module, agent="v2", exception=exception)
    messages = _v2_iter_messages(result)
    if not messages and exception is None:
        return qa

    qa.answer = _v2_final_text(messages)
    qa.usage = _v2_usage(messages)
    qa.latency_s = latency_s

    # First pass: map tool_call_id -> (ToolMessage status, result text).
    results_by_id: dict[str, tuple[str, str]] = {}
    for m in messages:
        if _v2_is_tool_message(m):
            tcid = getattr(m, "tool_call_id", None)
            status = getattr(m, "status", None) or "success"
            text = _v2_content_to_text(getattr(m, "content", ""))
            if tcid is not None:
                results_by_id[tcid] = (status, text)

    rounds = 0
    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            rounds += 1
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name", "?")
            args = tc.get("args", {}) or {}
            tcid = tc.get("id")
            status, rtext = results_by_id.get(tcid, ("success", ""))
            is_err, note = _result_is_error(rtext)
            if status == "error":
                is_err = True
                note = note or (rtext[:200] if rtext else "tool status=error")
            if is_err:
                qa.errors.append({"tool": name, "type": "dispatch", "message": note})
            qa.trace.append(ToolCallRecord(name=name, args=args, errored=is_err, note=note))
    qa.rounds = rounds
    return qa


# ===========================================================================
# P1 — hallucination-on-error heuristic (the headline metric)
# ===========================================================================

#: Phrases in a final answer that ACKNOWLEDGE a failed/uncertain step. If any of
#: these appears, the answer is NOT counted as hallucinating over the error — the
#: agent owned the failure. Kept deliberately broad to bias toward NOT flagging
#: (false negatives are safer than false positives for this headline number).
_ACK_PATTERNS = (
    r"error", r"errored", r"failed", r"failure", r"could not", r"couldn't",
    r"cannot", r"can't", r"unable", r"did not (?:work|succeed|complete)",
    r"didn't (?:work|succeed|complete)", r"no result", r"not (?:able|available)",
    r"unsuccessful", r"unavailable", r"retry", r"retried", r"tried again",
    r"invalid", r"unknown (?:method|module|parameter)", r"wasn't able",
    r"i (?:was )?unable", r"unfortunately", r"problem", r"issue (?:with|running)",
    r"exception", r"traceback", r"not auto-?scor", r"could not (?:be )?comput",
    r"i (?:do not|don't) have", r"missing", r"i was not able",
    r"approximat", r"estimat", r"rough", r"uncertain", r"caveat",
)
_ACK_RE = re.compile("|".join(_ACK_PATTERNS), re.IGNORECASE)


def answer_acknowledges_failure(answer: str) -> bool:
    """True if the final answer acknowledges a failed/uncertain step.

    A broad lexical check (see ``_ACK_PATTERNS``). Intentionally permissive so
    the P1 count under-flags rather than over-flags: any hint of "this didn't
    work / I'm unsure / approximate" disqualifies the answer from being called a
    hallucination-on-error.
    """
    return bool(_ACK_RE.search(answer or ""))


def is_hallucination_on_error(qa: QAResult) -> bool:
    """Heuristic for P1: did the agent hallucinate success after a tool error?

    Precise rule
    ------------
    Returns ``True`` IFF **all** of:

    1. The trace contains at least one tool call flagged ``errored`` (a
       dispatch/tool error — same definition both agents share), AND
    2. The agent nonetheless produced a substantive final answer
       (``len(answer.strip()) >= 40`` — not an empty/placeholder/"max rounds"
       reply), AND
    3. That final answer does NOT acknowledge any failure or uncertainty
       (:func:`answer_acknowledges_failure` is ``False``).

    A hard ``exception`` on the ask (the run crashed) is NOT a P1 hallucination —
    that is a crash, not a confabulation — so it returns ``False`` in that case.

    Known false positives
    ----------------------
    * The agent hit a tool error, then RECOVERED via a successful retry/alternate
      tool, and correctly reported the good value without ever mentioning the
      earlier blip. The error is real and unacknowledged, so this flags — yet the
      final answer may be perfectly correct. (Common with the v1 "guess method
      name -> describe_method -> correct call" pattern, where the first
      ``describe_method`` errors.) This is the main over-count risk; treat the P1
      rate as an UPPER bound and read it alongside the recovered-vs-fatal split.
    * An answer that fails to include any acknowledgement word yet is in fact
      hedged in prose the lexicon doesn't cover.

    Known false negatives
    ----------------------
    * The agent fabricates a value AND happens to include a hedging word
      ("approximately", "estimate") for unrelated reasons — the broad ``_ACK_RE``
      lets it off. Accepted on purpose: we bias toward not over-claiming the
      headline number.
    """
    if qa.exception:
        return False
    if not qa.has_tool_error:
        return False
    if len((qa.answer or "").strip()) < 40:
        return False
    return not answer_acknowledges_failure(qa.answer)


# ===========================================================================
# Correctness — default expected-answer judge + opt-in LLM-judge hook
# ===========================================================================

_NUM_RE = re.compile(r"-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_number(text: str) -> Optional[float]:
    """First plausible numeric token in ``text`` (helper for numeric grading)."""
    nums = _extract_numbers(text)
    return nums[0] if nums else None


def _extract_numbers(text: str) -> list[float]:
    """ALL plausible numeric tokens in ``text`` (commas + exponents handled)."""
    out: list[float] = []
    for tok in _NUM_RE.findall(text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _any_number_matches(answer: str, expected: float, rtol: float) -> bool:
    """True if ANY number in ``answer`` is within ``rtol`` of ``expected``.

    Relative tolerance (absolute when ``expected == 0``). "Any number" rather
    than "first number" because real answers carry many numbers (inputs,
    intermediate factors, the result) — the expected value just has to appear.
    Known false-positive risk: a small expected integer (e.g. ``2``) can match
    an incidental number; mitigated by pairing values with keywords in the
    suite's ``expected`` entries where that matters.
    """
    for got in _extract_numbers(answer):
        if expected == 0:
            if abs(got) <= rtol:
                return True
        elif abs(got - expected) <= abs(rtol * expected):
            return True
    return False


def _numeric_correctness(answer: str, expected: float, tol: float) -> bool:
    """Grade ``answer`` against a scalar ``expected`` value within ``tol``.

    Legacy scalar-``expected`` path (kept for backward compatibility): passes
    if any number in the answer is within the relative tolerance.
    """
    return _any_number_matches(answer, expected, tol)


def default_judge(question: dict, qa: QAResult) -> Optional[bool]:
    """The DEFAULT judge: grade ``qa.answer`` against the question's ``expected``.

    Returns
    -------
    bool or None
        ``None`` when the question carries no ``expected`` (the question is
        SKIPPED — not graded, not failed). Otherwise ``True`` iff:

        * every entry in ``expected["values"]`` matches some number in the
          answer within ``rtol`` (of ``value`` or any ``alt`` variant — alts
          cover unit/convention ambiguity, e.g. 42 mm vs 0.042 m), AND
        * every ``expected["keywords"]`` entry appears in the answer
          (case-insensitive substring).

    A legacy scalar ``expected`` (float, with sibling ``tolerance``) is also
    accepted and graded as a single any-number tolerance match.
    """
    if not isinstance(question, dict):
        return None
    expected = question.get("expected", question.get("expected_answer"))
    if expected is None:
        return None
    answer = qa.answer or ""

    # Legacy scalar form: {"expected": 100.0, "tolerance": 0.1}
    if isinstance(expected, (int, float)):
        tol = float(question.get("tolerance", 0.05))
        return _any_number_matches(answer, float(expected), tol)

    if not isinstance(expected, dict):
        return None

    for entry in expected.get("values") or []:
        try:
            value = float(entry["value"])
        except (KeyError, TypeError, ValueError):
            continue
        rtol = float(entry.get("rtol", 0.05))
        candidates = [value] + [float(a) for a in (entry.get("alt") or [])]
        if not any(_any_number_matches(answer, c, rtol) for c in candidates):
            return False
    for kw in expected.get("keywords") or []:
        if str(kw).lower() not in answer.lower():
            return False
    return True


def make_llm_judge(model, *, extra_instructions: str = "") -> Callable:
    """Build an OPT-IN LLM judge_fn backed by a LangChain chat model.

    Constructing the judge makes NO model calls; each ``judge(question, qa)``
    invocation makes ONE ``model.invoke`` call (real budget on a real model —
    never triggered by the offline tests, which pass a stub). The judge prompts
    the model with the question, the ``expected`` answer key when present, and
    the agent's final answer, and asks for a one-word ``PASS``/``FAIL`` verdict.

    Parameters
    ----------
    model : BaseChatModel-like
        Anything with ``.invoke(str) -> message`` (LangChain chat model or a
        stub). The message's ``content`` is parsed for PASS/FAIL.
    extra_instructions : str
        Optional extra grading guidance appended to the prompt.

    Returns
    -------
    callable
        ``judge(question_dict, qa) -> bool | None`` — ``None`` (skip) when the
        verdict cannot be parsed from the model reply.
    """
    def judge(question: dict, qa: QAResult) -> Optional[bool]:
        expected = question.get("expected") if isinstance(question, dict) else None
        expected_block = (
            f"\nAnswer key (module-computed ground truth):\n"
            f"{json.dumps(expected, indent=2)}\n" if expected else ""
        )
        prompt = (
            "You are grading a geotechnical engineering agent's answer.\n"
            f"Question:\n{question.get('question', '')}\n"
            f"{expected_block}"
            f"\nAgent's final answer:\n{qa.answer or '(no answer)'}\n\n"
            "Grade the answer for technical correctness"
            + (" against the answer key (allow tolerance/unit variants)"
               if expected else "")
            + ". " + extra_instructions
            + "\nReply with exactly one word on the first line: PASS or FAIL."
        )
        reply = model.invoke(prompt)
        text = getattr(reply, "content", reply)
        if isinstance(text, list):  # content-block replies
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in text
            )
        m = re.search(r"\b(PASS|FAIL)\b", str(text), re.IGNORECASE)
        if not m:
            return None
        return m.group(1).upper() == "PASS"

    return judge


def score_correctness(
    qas: list[QAResult],
    questions: list[dict],
    *,
    judge_fn: Optional[Callable[[dict, QAResult], Optional[bool]]] = None,
) -> dict:
    """Score answer correctness against the suite's ``expected`` answer keys.

    Default path: :func:`default_judge` grades every question that carries an
    ``expected`` field (numeric tolerance + keyword matching); questions WITHOUT
    ``expected`` are skipped — they count in ``n_skipped``, never as failures.
    When NO question is gradable, returns ``{"auto_scorable": False, ...}``
    honestly (the pre-v5.1 behavior).

    A custom ``judge_fn(question_dict, qa) -> bool | None`` (e.g. from
    :func:`make_llm_judge`) replaces the default judge; ``None`` verdicts are
    skipped the same way.

    Returns
    -------
    dict
        ``{"auto_scorable": bool, "method": str, "pass_rate": float|None,
        "n_passed": int|None, "n_graded": int, "n_skipped": int, "note": str}``.
    """
    by_id = {q.get("id"): q for q in questions if isinstance(q, dict)}
    judge = judge_fn if judge_fn is not None else default_judge

    passed = 0
    graded = 0
    skipped = 0
    for qa in qas:
        q = by_id.get(qa.qid)
        verdict = judge(q, qa) if q is not None else None
        if verdict is None:
            skipped += 1
            continue
        graded += 1
        if verdict:
            passed += 1

    if graded == 0 and judge_fn is None:
        return {
            "auto_scorable": False,
            "method": "none",
            "n_graded": 0,
            "n_passed": None,
            "n_skipped": skipped,
            "pass_rate": None,
            "note": (
                "Not auto-scorable: none of these questions carry an "
                "'expected' answer key. Supply a judge_fn (LLM-judge) or add "
                "expected fields to grade correctness."
            ),
        }

    return {
        "auto_scorable": True,
        "method": "llm_judge" if judge_fn is not None else "expected",
        "n_graded": graded,
        "n_passed": passed,
        "n_skipped": skipped,
        "pass_rate": (passed / graded) if graded else None,
        "note": (
            "Graded by the supplied judge_fn." if judge_fn is not None else
            "Default judge: numeric-tolerance + keyword match against the "
            "suite's 'expected' answer keys (v5.1 worktree-module ground "
            "truth). Questions without 'expected' are skipped, not failed."
        ),
    }


# ===========================================================================
# Aggregate metrics over one agent's run
# ===========================================================================

def _mean(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def score_run(
    qas: list[QAResult],
    questions: list[dict],
    *,
    judge_fn: Optional[Callable] = None,
) -> dict:
    """Compute the aggregate A/B metrics for ONE agent's run.

    Returns
    -------
    dict
        ``n_questions``, ``p1_hallucination_rate`` (+ ``p1_count``),
        ``tool_error_rate`` (questions with >=1 tool error / n),
        ``errors_per_question`` (mean tool errors), ``exception_rate``,
        ``avg_rounds``, ``avg_latency_s``, ``total_tokens`` / ``avg_tokens`` /
        ``max_total_tokens`` (``None`` if no usage — ``max_total_tokens`` is the
        single most expensive question, so a runaway is visible), and the nested
        ``correctness`` block.
    """
    n = len(qas)
    if n == 0:
        return {"n_questions": 0}

    p1 = [q for q in qas if is_hallucination_on_error(q)]
    n_with_tool_error = sum(1 for q in qas if q.has_tool_error)
    n_exception = sum(1 for q in qas if q.exception)
    total_tool_errors = sum(q.n_tool_errors for q in qas)

    # Per-question totals (authoritative; the callback-aggregated total when v2,
    # else the trace-summed usage). Only count questions that exposed usage.
    per_q_totals = [q.total_tokens for q in qas if q.total_tokens is not None]
    total_tokens = sum(per_q_totals) if per_q_totals else None
    max_total_tokens = max(per_q_totals) if per_q_totals else None

    return {
        "n_questions": n,
        "p1_count": len(p1),
        "p1_hallucination_rate": len(p1) / n,
        "p1_question_ids": [q.qid for q in p1],
        "tool_error_rate": n_with_tool_error / n,
        "errors_per_question": total_tool_errors / n,
        "exception_rate": n_exception / n,
        "avg_rounds": _mean([q.rounds for q in qas]),
        "avg_latency_s": _mean([q.latency_s for q in qas]),
        "total_tokens": total_tokens,
        "avg_tokens": (total_tokens / len(per_q_totals)) if per_q_totals else None,
        "max_total_tokens": max_total_tokens,
        "correctness": score_correctness(qas, questions, judge_fn=judge_fn),
    }


# ===========================================================================
# The A/B runner
# ===========================================================================

def _run_one(agent, question: str, adapter, qid: str, module: str,
             ask_kind: str) -> QAResult:
    """Run a single question through one agent and adapt -> QAResult.

    ``ask_kind`` selects how the agent is invoked:

    * ``"v1"`` — ``agent.ask(question) -> AgentResult``. Takes NO callbacks, so
      token usage stays ``None`` (the v1 ``ClaudeEngine`` exposes none).
    * ``"v2"`` — ``agent.invoke({"messages": [...]}) -> {"messages": [...]}``,
      wrapped in :func:`get_usage_metadata_callback` so the per-question TOTAL
      tokens are aggregated across EVERY model call in the LangGraph run —
      including the references/reviewer **sub-agents**, whose calls propagate
      through the run's callbacks. That authoritative total is written onto the
      QAResult's ``usage`` (preferring it over the trace-summed
      :func:`_v2_usage`); if the callback recorded nothing, the trace-derived
      usage is left untouched so nothing regresses.

    A hard exception is caught and recorded on the QAResult (it does NOT abort
    the whole A/B run).
    """
    t0 = time.time()
    if ask_kind == "v1":
        # v1 path: no callbacks (the .ask interface does not accept a config).
        try:
            raw = agent.ask(question)
            latency = time.time() - t0
            return adapter(qid, module, raw, latency, exception=None)
        except Exception as exc:  # a crashing ask must not kill the run
            latency = time.time() - t0
            return adapter(qid, module, None, latency,
                           exception=f"{type(exc).__name__}: {exc}")

    # v2/LangGraph path: aggregate usage across ALL model calls (sub-agents too)
    # via the callback, and pass it through ``config={"callbacks": [cb]}``.
    from langchain_core.callbacks import get_usage_metadata_callback
    try:
        with get_usage_metadata_callback() as cb:
            raw = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"callbacks": [cb]},
            )
            callback_usage = _sum_callback_usage(dict(cb.usage_metadata))
        latency = time.time() - t0
        qa = adapter(qid, module, raw, latency, exception=None)
    except Exception as exc:  # a crashing invoke must not kill the run
        latency = time.time() - t0
        return adapter(qid, module, None, latency,
                       exception=f"{type(exc).__name__}: {exc}")
    # Prefer the callback total (sub-agents included); fall back to the trace-
    # summed usage that the adapter already set, so nothing regresses.
    if callback_usage is not None:
        qa.usage = callback_usage
    return qa


def run_ab(
    questions: list[dict],
    *,
    v1_agent,
    v2_agent,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Run the A/B comparison over ``questions`` through BOTH agents.

    Apples-to-apples: the two agents should be built from the SAME underlying
    Anthropic model (see :func:`build_agents`). Each question is run through v1
    (``.ask``) and v2 (``.invoke``), adapted into :class:`QAResult`, and scored.

    Parameters
    ----------
    questions : list of dict
        Suite questions (``{id, module, complexity, question}``).
    v1_agent : object
        Anything with ``.ask(question) -> AgentResult`` (a real
        :class:`~funhouse_agent.agent.GeotechAgent` or a stub).
    v2_agent : object
        Anything with ``.invoke({"messages": [...]}) -> {"messages": [...]}`` (a
        compiled deep agent or a stub).
    limit : int, optional
        Run only the first ``limit`` questions (for a CHEAP live subset). ``None``
        runs all of them.
    verbose : bool
        Print a one-line progress marker per question.

    Returns
    -------
    dict
        ``{"questions": [...ids...], "v1": {"results": [...], "metrics": {...}},
        "v2": {...}, "model": None}``. Per-agent ``results`` are QAResult dicts.
    """
    if limit is not None:
        questions = questions[:limit]

    v1_qas: list[QAResult] = []
    v2_qas: list[QAResult] = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"Q{i}")
        module = q.get("module", "?")
        text = q.get("question", "")
        if verbose:
            print(f"[{i}/{len(questions)}] {qid} ({module})")

        v1_qa = _run_one(v1_agent, text, _v1_to_qa, qid, module, "v1")
        v2_qa = _run_one(v2_agent, text, _v2_to_qa, qid, module, "v2")
        v1_qas.append(v1_qa)
        v2_qas.append(v2_qa)

    return {
        "questions": [q.get("id") for q in questions],
        "n_questions": len(questions),
        "model": None,  # filled in by the CLI when known
        "v1": {
            "results": [qa.to_dict() for qa in v1_qas],
            "metrics": score_run(v1_qas, questions),
        },
        "v2": {
            "results": [qa.to_dict() for qa in v2_qas],
            "metrics": score_run(v2_qas, questions),
        },
    }


# ===========================================================================
# Single-agent v2 suite runner (the owner's OWN model — no v1, no A/B)
# ===========================================================================

def run_suite(
    model,
    *,
    questions: Optional[list[dict]] = None,
    limit: Optional[int] = None,
    out: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Run the whole suite through ONE v2 deepagents agent on the owner's model.

    This is the Funhouse-friendly, **v2-only** runner: it takes the owner's OWN
    LangChain chat model (e.g. a
    :class:`~funhouse_agent.deep.databricks_bridge.PrompterChatModel` on
    ``funhouse-gpt-high``, or a ``ChatAnthropic``), builds the deep agent ONCE,
    and runs every suite question through it — no Claude key, no v1
    ``ClaudeEngine``, and no A/B second agent. It reuses the SAME
    :func:`_run_one` + :func:`_v2_to_qa` path the A/B runner uses, so the trace
    extraction, error flagging, and :class:`QAResult` shape match exactly.

    Correctness is NOT auto-scored (the suite carries no expected answers — see
    :func:`score_correctness`); the owner reads the answers in the markdown
    review file (:func:`render_suite_markdown`) and judges quality by eye, the
    way they did with the v4.6.4 run. The returned ``metrics`` are the v2
    *process* metrics (P1 hallucination rate, tool-error rate, rounds, latency,
    tokens) via :func:`score_run`.

    Parameters
    ----------
    model : str | BaseChatModel
        A LangChain chat model (or a ``"provider:model"`` string) that drives
        the deep agent. The owner passes their Funhouse model here, e.g.
        ``PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")``.
        Using it makes REAL API calls on the owner's budget.
    questions : list of dict, optional
        Suite questions (``{id, module, complexity, question}``). When ``None``
        (the default) the packaged suite is loaded via :func:`load_suite`. A
        small explicit list is a test seam that bypasses the loader.
    limit : int, optional
        Run only the first ``limit`` questions (a cheap subset). ``None`` runs
        all of them.
    out : Path, optional
        When given, write BOTH a results JSON (here) and a readable markdown
        review file at ``<out>.md`` (see :func:`write_suite_results`).
    verbose : bool
        Print a one-line run header and a ``[i/N] <id> (<module>)`` progress
        line per question (so a long Funhouse run shows progress — Databricks
        buffers stdout, which is fine).

    Returns
    -------
    dict
        ``{"model": <str>, "n": N, "results": [qa.to_dict(), ...],
        "metrics": score_run(results, questions)}``. ``results`` are
        :class:`QAResult` dicts (one per question).
    """
    if questions is None:
        questions = load_suite()
    if limit is not None:
        questions = questions[:limit]

    model_label = model if isinstance(model, str) else type(model).__name__
    model_id = getattr(model, "model", None) or getattr(model, "model_name", None)
    if model_id and not isinstance(model, str):
        model_label = f"{model_label}({model_id})"

    if verbose:
        n = len(questions)
        print(
            f"Running {n} question(s) through v2 (deepagents) on "
            f"{model_label} — REAL API calls"
        )

    # Build the v2 agent ONCE — the owner's model drives both text and (via the
    # default LangChainVisionEngine wrap inside build_deep_agent) vision.
    from funhouse_agent.deep.agent import build_deep_agent
    agent = build_deep_agent(model=model)

    qas: list[QAResult] = []
    texts: list[str] = []
    cumulative_tokens = 0
    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"Q{i}")
        module = q.get("module", "?")
        text = q.get("question", "")
        # _run_one already catches a crashing ask and records it on the QAResult
        # (so one failure does not abort the whole run); the per-question try is
        # belt-and-suspenders against anything _run_one itself might raise.
        try:
            qa = _run_one(agent, text, _v2_to_qa, qid, module, "v2")
        except Exception as exc:  # pragma: no cover - defensive
            qa = QAResult(qid=qid, module=module, agent="v2",
                          exception=f"{type(exc).__name__}: {exc}")
        if verbose:
            # Show this question's token spend AND the running cumulative total so
            # a long Funhouse run surfaces a runaway question as it happens.
            tok = qa.total_tokens
            if tok is not None:
                cumulative_tokens += tok
                tok_note = f"  +{tok:,} tok  (cumulative {cumulative_tokens:,})"
            else:
                tok_note = "  (tokens n/a)"
            print(f"[{i}/{len(questions)}] {qid} ({module}){tok_note}")
        qas.append(qa)
        texts.append(text)

    # Fold the question text into each result dict (QAResult itself does not
    # carry it) so the markdown review can show the question above its answer.
    results = []
    for qa, text in zip(qas, texts):
        d = qa.to_dict()
        d["question"] = text
        results.append(d)
    suite_result = {
        "model": model_label,
        "n": len(qas),
        "results": results,
        "metrics": score_run(qas, questions),
    }

    if out is not None:
        paths = write_suite_results(suite_result, Path(out))
        if verbose:
            print(f"\nWrote results JSON: {paths['json']}")
            if paths["markdown"]:
                print(f"Wrote markdown review: {paths['markdown']}")

    return suite_result


# ===========================================================================
# Markdown A/B summary table
# ===========================================================================

def _fmt(value, kind: str = "num") -> str:
    """Format a metric value for the markdown table."""
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "int":
        return f"{int(round(value))}"
    if kind == "tok":
        # Token counts read better with thousands separators.
        return f"{int(round(value)):,}"
    if kind == "num":
        return f"{value:.2f}"
    return str(value)


def _fmt_tokens(result: dict) -> str:
    """Render one result's per-question token line value.

    ``"18,432"`` when only a total is known, or ``"18,432 (12,000 in / 6,432
    out)"`` when BOTH the input and output split are nonzero (a split is only
    meaningful when the backend reports one — the Funhouse total-only path zeroes
    ``output_tokens``, so the split is hidden). Returns ``"n/a"`` when no total.
    """
    total = result.get("total_tokens")
    if total is None:
        return "n/a"
    inp = result.get("input_tokens") or 0
    out = result.get("output_tokens") or 0
    if inp and out:
        return f"{int(total):,} ({int(inp):,} in / {int(out):,} out)"
    return f"{int(total):,}"


def _delta(v1, v2, kind: str = "num", lower_is_better: bool = True) -> str:
    """Format the v2-vs-v1 delta with a ``better``/``worse`` direction marker.

    ``lower_is_better`` controls which way is good (lower P1 rate / latency =
    better; higher question count = better). A delta that rounds to zero at the
    displayed precision is treated as neutral (no marker), so a tiny float
    artifact does not get labelled ``worse``.
    """
    if v1 is None or v2 is None:
        return "n/a"
    d = v2 - v1
    if kind == "pct":
        body = f"{100 * d:+.1f} pts"
        negligible = abs(100 * d) < 0.05
    elif kind == "int":
        body = f"{int(round(d)):+d}"
        negligible = round(d) == 0
    elif kind == "tok":
        body = f"{int(round(d)):+,}"
        negligible = round(d) == 0
    else:
        body = f"{d:+.2f}"
        negligible = abs(d) < 0.005
    if negligible:
        return body
    improved = (d < 0) if lower_is_better else (d > 0)
    return f"{body} {'better' if improved else 'worse'}"


def render_markdown_table(ab: dict) -> str:
    """Render the A/B run as a markdown summary table (metric | v1 | v2 | delta).

    Parameters
    ----------
    ab : dict
        The dict returned by :func:`run_ab`.

    Returns
    -------
    str
        A markdown document: a header, the metrics table, and a correctness note.
    """
    m1 = ab.get("v1", {}).get("metrics", {})
    m2 = ab.get("v2", {}).get("metrics", {})

    # (label, key, kind, lower_is_better)
    rows = [
        ("Questions", "n_questions", "int", False),
        ("P1 hallucination-on-error rate", "p1_hallucination_rate", "pct", True),
        ("P1 count", "p1_count", "int", True),
        ("Tool-error rate (q with >=1 error)", "tool_error_rate", "pct", True),
        ("Errors per question (mean)", "errors_per_question", "num", True),
        ("Exception rate", "exception_rate", "pct", True),
        ("Avg rounds", "avg_rounds", "num", True),
        ("Avg latency (s)", "avg_latency_s", "num", True),
        ("Total tokens", "total_tokens", "tok", True),
        ("Avg tokens / question", "avg_tokens", "tok", True),
        ("Max tokens (one question)", "max_total_tokens", "tok", True),
    ]

    lines = [
        "## A/B eval: v1 agent vs v2 deepagents",
        "",
        f"Model: `{ab.get('model') or 'n/a'}`  |  Questions: "
        f"{ab.get('n_questions', '?')}",
        "",
        "| Metric | v1 | v2 | delta (v2-v1) |",
        "|---|---|---|---|",
    ]
    for label, key, kind, lib in rows:
        v1 = m1.get(key)
        v2 = m2.get(key)
        lines.append(
            f"| {label} | {_fmt(v1, kind)} | {_fmt(v2, kind)} | "
            f"{_delta(v1, v2, kind, lib)} |"
        )

    # Correctness note (honest about auto-scorability).
    c1 = m1.get("correctness", {})
    c2 = m2.get("correctness", {})
    lines += ["", "### Correctness", ""]
    if not c1.get("auto_scorable") and not c2.get("auto_scorable"):
        lines.append(
            "Correctness is **not auto-scorable** from this suite "
            "(`geotech_test_suite.json` carries no expected answers). "
            "Supply an LLM-judge (`judge_fn`) or an expected-answer suite "
            "variant to grade it. The metrics above are process/behavior metrics."
        )
    else:
        lines.append(
            f"| Correctness pass-rate "
            f"({c1.get('method', c2.get('method', '?'))}) | "
            f"{_fmt(c1.get('pass_rate'), 'pct')} | "
            f"{_fmt(c2.get('pass_rate'), 'pct')} | "
            f"{_delta(c1.get('pass_rate'), c2.get('pass_rate'), 'pct', False)} |"
        )
    return "\n".join(lines)


# ===========================================================================
# Output writing
# ===========================================================================

def write_results(ab: dict, out_path: Path, *, markdown: bool = True) -> dict:
    """Write the results JSON (and a sibling ``.md`` table) to disk.

    Parameters
    ----------
    ab : dict
        The :func:`run_ab` result, with the markdown table folded in under
        ``"markdown_table"``.
    out_path : Path
        Destination for the JSON. The markdown table is written next to it as
        ``<out_path>.md`` when ``markdown`` is True.

    Returns
    -------
    dict
        ``{"json": str, "markdown": str|None}`` of the paths written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = render_markdown_table(ab)
    payload = dict(ab)
    payload["markdown_table"] = table
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    md_path = None
    if markdown:
        md_path = out_path.with_suffix(out_path.suffix + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(table + "\n")
    return {"json": str(out_path), "markdown": str(md_path) if md_path else None}


# ===========================================================================
# Single-agent (v2-only) markdown review renderer + writer
# ===========================================================================

def _trace_summary(trace: list) -> str:
    """Compact ``agent.method`` summary of the tool calls in one trace.

    ``trace`` is a list of either :class:`ToolCallRecord` objects or their dict
    form (``{name, args, errored, note}``), so this works on both a live
    :class:`QAResult` and a re-loaded ``to_dict`` payload. A ``call_agent`` is
    rendered as ``<agent_name>.<method>`` from its args (the meaningful unit for
    the owner); any other tool is rendered by its bare name. An errored call is
    suffixed with ``[errored]``.
    """
    parts = []
    for tc in trace:
        if isinstance(tc, dict):
            name = tc.get("name", "?")
            args = tc.get("args", {}) or {}
            errored = bool(tc.get("errored"))
        else:
            name = getattr(tc, "name", "?")
            args = getattr(tc, "args", {}) or {}
            errored = bool(getattr(tc, "errored", False))
        if name in ("call_agent", "describe_method", "list_methods") and isinstance(args, dict):
            agent_name = args.get("agent_name") or args.get("agent") or ""
            method = args.get("method") or ""
            label = ".".join(p for p in (agent_name, method) if p) or name
        else:
            label = name
        if errored:
            label += " [errored]"
        parts.append(label)
    return ", ".join(parts) if parts else "(none)"


def render_suite_markdown(suite_result: dict) -> str:
    """Render a single v2 run as a top-to-bottom human review document.

    This is the file the owner skims to judge answer QUALITY (correctness is NOT
    auto-scored — stated once at the top, like the A/B table). The layout:

    * a short metrics header (n, P1 hallucination rate, tool-error rate, avg
      rounds / latency / tokens), then
    * **per question** — the id + module, the question text, the agent's final
      answer, and a compact ``tools used: <agent.method>, ...`` line plus any
      ``errors:`` line.

    Parameters
    ----------
    suite_result : dict
        The dict returned by :func:`run_suite` (``{model, n, results,
        metrics}``). ``results`` are :class:`QAResult` dicts.

    Returns
    -------
    str
        A markdown review document.
    """
    model = suite_result.get("model", "n/a")
    n = suite_result.get("n", len(suite_result.get("results", [])))
    m = suite_result.get("metrics", {}) or {}
    results = suite_result.get("results", []) or []

    c = m.get("correctness", {}) or {}
    if c.get("auto_scorable"):
        correctness_blurb = (
            f"Correctness is auto-scored for the {c.get('n_graded', 0)} "
            f"question(s) carrying an `expected` answer key "
            f"(pass rate {_fmt(c.get('pass_rate'), 'pct')}; "
            f"{c.get('n_skipped', 0)} skipped — no key). The keys are "
            "v5.1 worktree-module ground truth; still spot-check answers and "
            "citations by eye."
        )
    else:
        correctness_blurb = (
            "Correctness is **not auto-scored** for these questions (no "
            "`expected` answer keys). Read each answer below and judge its "
            "quality by eye. The metrics below are process/behavior metrics, "
            "not a correctness score."
        )

    lines = [
        "# v2 (deepagents) suite review",
        "",
        f"Model: `{model}`  |  Questions: {n}",
        "",
        correctness_blurb,
        "",
        "## Run metrics",
        "",
        f"- Questions: {m.get('n_questions', n)}",
        f"- P1 hallucination-on-error rate: {_fmt(m.get('p1_hallucination_rate'), 'pct')} "
        f"({_fmt(m.get('p1_count'), 'int')} question(s))",
        f"- Tool-error rate (q with >=1 error): {_fmt(m.get('tool_error_rate'), 'pct')}",
        f"- Errors per question (mean): {_fmt(m.get('errors_per_question'), 'num')}",
        f"- Exception rate: {_fmt(m.get('exception_rate'), 'pct')}",
        f"- Avg rounds: {_fmt(m.get('avg_rounds'), 'num')}",
        f"- Avg latency (s): {_fmt(m.get('avg_latency_s'), 'num')}",
        f"- Total tokens: {_fmt(m.get('total_tokens'), 'tok')}",
        f"- Avg tokens / question: {_fmt(m.get('avg_tokens'), 'tok')}",
        f"- Max tokens (one question): {_fmt(m.get('max_total_tokens'), 'tok')}",
    ]
    if c.get("auto_scorable"):
        lines.append(
            f"- Correctness ({c.get('method', '?')}): "
            f"{_fmt(c.get('n_passed'), 'int')}/{_fmt(c.get('n_graded'), 'int')} "
            f"passed ({_fmt(c.get('pass_rate'), 'pct')}), "
            f"{c.get('n_skipped', 0)} skipped (no expected key)"
        )
    lines += [
        "",
        "## Answers",
        "",
    ]

    for i, r in enumerate(results, 1):
        qid = r.get("qid", f"Q{i}")
        module = r.get("module", "?")
        question = r.get("question", "")
        answer = (r.get("answer", "") or "").strip()
        trace = r.get("trace", []) or []
        errors = r.get("errors", []) or []
        exception = r.get("exception")

        lines.append(f"### {i}. {qid} ({module})")
        lines.append("")
        if question:
            lines.append(f"**Question:** {question}")
            lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(answer if answer else "_(no answer produced)_")
        lines.append("")
        lines.append(f"_tokens: {_fmt_tokens(r)}_")
        lines.append(f"_tools used: {_trace_summary(trace)}_")
        if exception:
            lines.append("")
            lines.append(f"_exception: {exception}_")
        elif errors:
            notes = []
            for e in errors:
                if isinstance(e, dict):
                    notes.append(str(e.get("message") or e.get("type") or e))
                else:
                    notes.append(str(e))
            lines.append("")
            lines.append(f"_errors: {'; '.join(notes)}_")
        lines.append("")

    return "\n".join(lines)


def write_suite_results(suite_result: dict, out_path: Path, *,
                        markdown: bool = True) -> dict:
    """Write a single-agent run's results JSON + a readable markdown review.

    Mirrors :func:`write_results` (the A/B writer) conventions for the SINGLE-
    agent case: the JSON goes to ``out_path`` (with the rendered markdown folded
    in under ``"markdown_review"``), and the human review doc is written next to
    it as ``<out_path>.md`` when ``markdown`` is True.

    Parameters
    ----------
    suite_result : dict
        The :func:`run_suite` result.
    out_path : Path
        Destination for the JSON. The markdown review is written next to it as
        ``<out_path>.md`` when ``markdown`` is True.
    markdown : bool
        Also write the sibling ``.md`` review. Defaults to True.

    Returns
    -------
    dict
        ``{"json": str, "markdown": str|None}`` of the paths written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review = render_suite_markdown(suite_result)
    payload = dict(suite_result)
    payload["markdown_review"] = review
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    md_path = None
    if markdown:
        md_path = out_path.with_suffix(out_path.suffix + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(review + "\n")
    return {"json": str(out_path), "markdown": str(md_path) if md_path else None}


# ===========================================================================
# Live agent construction (apples-to-apples from ONE model config)
# ===========================================================================

def build_agents(model_id: str = DEFAULT_MODEL, *, max_tokens: int = 4096):
    """Build the v1 and v2 agents from ONE Anthropic model config.

    The v1 agent runs on a :class:`~funhouse_agent.engine.ClaudeEngine` (Anthropic
    SDK), the v2 agent on a ``langchain_anthropic.ChatAnthropic`` — but BOTH point
    at the same ``model_id`` so the A/B comparison isolates the agent
    architecture, not the model.

    .. warning::
       The returned agents make REAL Anthropic API calls when used. This
       function only CONSTRUCTS them (and so reads ``ANTHROPIC_API_KEY``); it
       does not call the model.

    Parameters
    ----------
    model_id : str
        Anthropic model id (e.g. ``"claude-sonnet-4-6"``).
    max_tokens : int
        Max response tokens for both backends.

    Returns
    -------
    (v1_agent, v2_agent)
        ``v1_agent`` is a ``GeotechAgent``; ``v2_agent`` is a compiled deep agent.
    """
    from funhouse_agent import GeotechAgent, ClaudeEngine
    from funhouse_agent.deep.agent import build_deep_agent

    v1_agent = GeotechAgent(
        genai_engine=ClaudeEngine(model=model_id, max_tokens=max_tokens),
        verbose=False,
    )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "langchain_anthropic is required for a live A/B run: "
            f"{e}\n  pip install langchain-anthropic"
        )
    chat = ChatAnthropic(model=model_id, max_tokens=max_tokens, temperature=0)
    v2_agent = build_deep_agent(model=chat)
    return v1_agent, v2_agent


# ===========================================================================
# Dry-run stub agents (OFFLINE — exercise the whole pipeline, no API)
# ===========================================================================

class _StubV1Agent:
    """A canned v1 agent for ``--dry-run``: returns a fixed AgentResult.

    No model, no network. Produces a clean tool call and a plain answer so the
    whole v1 adapter + scorer path runs offline.
    """

    def ask(self, question: str):
        from funhouse_agent.react_support import AgentResult
        return AgentResult(
            answer=(
                "Using the bearing_capacity module, the ultimate bearing "
                "capacity is approximately 512 kPa and the allowable is 170 kPa."
            ),
            tool_calls=[{
                "round": 1,
                "tool_name": "call_agent",
                "arguments": {"agent_name": "bearing_capacity",
                              "method": "bearing_capacity_analysis"},
                "result_preview": '{"q_ultimate_kPa": 512.3}',
            }],
            rounds=2,
            total_time_s=0.0,
            errors=[],
        )


class _StubV2Agent:
    """A canned v2 deep agent for ``--dry-run``: returns a fixed message list.

    Mirrors the deepagents invoke shape (``{"messages": [...]}``) with one
    successful ``call_agent`` tool call + result and a final assistant answer,
    so the whole v2 adapter + scorer path runs offline.
    """

    def invoke(self, payload, *args, **kwargs):
        return self._messages()

    def _messages(self):
        from langchain_core.messages import AIMessage, ToolMessage
        call = AIMessage(
            content="",
            tool_calls=[{
                "name": "call_agent",
                "args": {"agent_name": "bearing_capacity",
                         "method": "bearing_capacity_analysis"},
                "id": "call_1",
            }],
            usage_metadata={"input_tokens": 1200, "output_tokens": 200,
                            "total_tokens": 1400},
        )
        result = ToolMessage(
            content='{"q_ultimate_kPa": 512.3}',
            tool_call_id="call_1", name="call_agent",
        )
        final = AIMessage(
            content="The ultimate bearing capacity is about 512 kPa.",
            usage_metadata={"input_tokens": 1500, "output_tokens": 120,
                            "total_tokens": 1620},
        )
        return {"messages": [call, result, final]}


# ===========================================================================
# CLI
# ===========================================================================

def _estimate_cost(n_questions: int) -> str:
    """A rough order-of-magnitude API-cost note for an n-question A/B run.

    Assumptions (stated in the string): ~6-10 tool rounds/question, each round a
    full context replay, Sonnet-class pricing ~ $3 / Mtok in, $15 / Mtok out,
    ~25-40k input + ~2-4k output tokens per question per agent. This is a
    BALLPARK to set expectations, not a billing figure.
    """
    per_q_per_agent = 0.15  # USD, rough mid-point
    total = n_questions * 2 * per_q_per_agent
    return (
        f"Rough cost estimate: ~${total:.0f} for {n_questions} questions x 2 "
        f"agents (~${per_q_per_agent:.2f}/question/agent). Assumes Sonnet-class "
        f"pricing (~$3/Mtok in, $15/Mtok out), ~6-10 tool rounds/question with "
        f"full-context replay (~25-40k in + ~2-4k out per question per agent). "
        f"ORDER OF MAGNITUDE ONLY."
    )


def main(argv=None) -> int:
    """CLI entry point for the A/B eval harness."""
    parser = argparse.ArgumentParser(
        description="A/B eval: v1 agent vs v2 deepagents on the geotech suite.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N questions (cheap live subset). Omit for all.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Anthropic model id for BOTH agents (default {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--out", default=str(_DEEP_DIR / "ab_results.json"),
        help="Results JSON path (a sibling .md table is written too).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="OFFLINE: use stub agents (no model/API). Exercises the whole "
             "pipeline. Default-safe.",
    )
    parser.add_argument(
        "--suite-only", action="store_true",
        help="Run the v2-ONLY suite runner (run_suite) on a local Anthropic "
             "ChatAnthropic model — no v1, no A/B. For local testing of the "
             "Funhouse-friendly single-agent path. Spends real Anthropic budget.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the live-run cost confirmation prompt (for non-interactive "
             "use). Ignored under --dry-run.",
    )
    args = parser.parse_args(argv)

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    questions = load_suite()
    n = len(questions) if args.limit is None else min(args.limit, len(questions))

    # ----- v2-only suite runner (local Anthropic model) -----
    if args.suite_only:
        print("=" * 70)
        print("LIVE v2-ONLY SUITE RUN — spends REAL Anthropic API budget.")
        print(f"  Questions: {n}  (single agent, v2 deepagents only)")
        print(f"  Model: {args.model}")
        print("=" * 70)
        if not args.yes:
            try:
                resp = input("Proceed with the LIVE run? [y/N] ").strip().lower()
            except EOFError:
                resp = ""
            if resp not in ("y", "yes"):
                print("Aborted. (Use --dry-run for an offline pipeline test.)")
                return 1
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:  # pragma: no cover - env-dependent
            raise SystemExit(
                "langchain_anthropic is required for --suite-only: "
                f"{e}\n  pip install langchain-anthropic"
            )
        model = ChatAnthropic(model=args.model, max_tokens=4096, temperature=0)
        suite_result = run_suite(model, limit=args.limit, out=Path(args.out))
        print("\n" + render_suite_markdown(suite_result))
        return 0

    if args.dry_run:
        print(f"[DRY RUN] OFFLINE stub agents — no API. {n} question(s).")
        v1_agent, v2_agent = _StubV1Agent(), _StubV2Agent()
        model_label = "DRY-RUN-STUB"
    else:
        # Loud, explicit cost caution before ANY live model call.
        print("=" * 70)
        print("LIVE A/B RUN — this spends REAL Anthropic API budget.")
        print(f"  Questions: {n}  x  2 agents (v1 + v2)  =  {2 * n} model runs")
        print(f"  Model: {args.model}")
        print(f"  {_estimate_cost(n)}")
        print("=" * 70)
        if not args.yes:
            try:
                resp = input("Proceed with the LIVE run? [y/N] ").strip().lower()
            except EOFError:
                resp = ""
            if resp not in ("y", "yes"):
                print("Aborted. (Use --dry-run for an offline pipeline test.)")
                return 1
        v1_agent, v2_agent = build_agents(args.model)
        model_label = args.model

    ab = run_ab(questions, v1_agent=v1_agent, v2_agent=v2_agent, limit=args.limit)
    ab["model"] = model_label

    paths = write_results(ab, Path(args.out))
    print("\n" + render_markdown_table(ab))
    print(f"\nWrote results JSON: {paths['json']}")
    if paths["markdown"]:
        print(f"Wrote markdown table: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
