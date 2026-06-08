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

Correctness is NOT auto-scorable from this suite
------------------------------------------------
``geotech_test_suite.json`` is a flat list of ``{id, module, complexity,
question}`` objects — it carries **NO expected/reference answers and no
tolerances** (verified by inspection). So numeric pass/fail correctness CANNOT be
computed from the suite alone. :func:`score_correctness` reports this honestly and
leaves a clean hook (``judge_fn``) for an optional LLM-judge or a future
expected-answer file; if a future suite variant DOES add an ``expected``/
``tolerance`` field, :func:`_numeric_correctness` already knows how to grade it.

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


# ===========================================================================
# Suite loading
# ===========================================================================

def load_suite(path: Path = SUITE_PATH) -> list[dict]:
    """Load the geotech eval suite as a list of question dicts.

    Parameters
    ----------
    path : Path
        Path to ``geotech_test_suite.json``. Defaults to the packaged suite.

    Returns
    -------
    list of dict
        Each ``{"id", "module", "complexity", "question"}``. The suite carries
        NO expected answers — see :func:`score_correctness`.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of questions, got {type(data)}")
    return data


def suite_has_expected_answers(questions: list[dict]) -> bool:
    """True if ANY suite question carries an ``expected``/``answer`` field.

    Drives :func:`score_correctness`: the current suite returns ``False`` (it is
    question-only), so correctness is reported as "not auto-scorable". A future
    suite variant that adds expected answers flips this to ``True`` and unlocks
    :func:`_numeric_correctness`.
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

    def to_dict(self) -> dict:
        """JSON-serializable dict (mirrors docs/geotech_test_suite_results.json)."""
        d = asdict(self)
        d["n_tool_errors"] = self.n_tool_errors
        d["has_tool_error"] = self.has_tool_error
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
# Correctness (NOT auto-scorable from this suite — honest hook)
# ===========================================================================

def _extract_number(text: str) -> Optional[float]:
    """First plausible numeric token in ``text`` (helper for numeric grading)."""
    for tok in re.findall(r"-?[\d,]*\.?\d+", text or ""):
        try:
            return float(tok.replace(",", ""))
        except ValueError:
            continue
    return None


def _numeric_correctness(answer: str, expected: float, tol: float) -> bool:
    """Grade ``answer`` against an ``expected`` value within relative ``tol``.

    Used ONLY when a (future) suite variant carries expected answers. Pulls the
    first number from the answer and checks ``|got - expected| <= tol*|expected|``
    (absolute tolerance when ``expected == 0``).
    """
    got = _extract_number(answer)
    if got is None:
        return False
    if expected == 0:
        return abs(got) <= tol
    return abs(got - expected) <= abs(tol * expected)


def score_correctness(
    qas: list[QAResult],
    questions: list[dict],
    *,
    judge_fn: Optional[Callable[[dict, QAResult], bool]] = None,
) -> dict:
    """Score answer correctness — honestly reporting what the suite allows.

    The packaged suite is question-only (no expected answers), so by default this
    returns ``{"auto_scorable": False, ...}`` and grades nothing. There are two
    optional paths:

    * If the suite questions carry ``expected``/``tolerance`` fields, numeric
      grading via :func:`_numeric_correctness` kicks in automatically.
    * If a ``judge_fn(question_dict, qa) -> bool`` is supplied (e.g. an
      LLM-judge), each answer is graded by it. This is the clean hook for adding
      correctness later WITHOUT changing the harness.

    Parameters
    ----------
    qas : list of QAResult
        One agent's results.
    questions : list of dict
        The suite questions (carry expected answers only in a future variant).
    judge_fn : callable, optional
        ``(question_dict, qa) -> bool`` external grader.

    Returns
    -------
    dict
        ``{"auto_scorable": bool, "method": str, "pass_rate": float|None,
        "n_passed": int|None, "n_graded": int, "note": str}``.
    """
    by_id = {q.get("id"): q for q in questions if isinstance(q, dict)}
    has_expected = suite_has_expected_answers(questions)

    if judge_fn is not None:
        passed = 0
        graded = 0
        for qa in qas:
            q = by_id.get(qa.qid)
            if q is None:
                continue
            graded += 1
            if judge_fn(q, qa):
                passed += 1
        return {
            "auto_scorable": True,
            "method": "llm_judge",
            "n_graded": graded,
            "n_passed": passed,
            "pass_rate": (passed / graded) if graded else None,
            "note": "Graded by the supplied judge_fn.",
        }

    if has_expected:
        passed = 0
        graded = 0
        for qa in qas:
            q = by_id.get(qa.qid)
            if not q:
                continue
            expected = q.get("expected", q.get("expected_answer"))
            if expected is None:
                continue
            tol = float(q.get("tolerance", 0.05))
            graded += 1
            if _numeric_correctness(qa.answer, float(expected), tol):
                passed += 1
        return {
            "auto_scorable": True,
            "method": "numeric_tolerance",
            "n_graded": graded,
            "n_passed": passed,
            "pass_rate": (passed / graded) if graded else None,
            "note": "Numeric-with-tolerance grading against suite expected values.",
        }

    return {
        "auto_scorable": False,
        "method": "none",
        "n_graded": 0,
        "n_passed": None,
        "pass_rate": None,
        "note": (
            "Not auto-scorable from suite: geotech_test_suite.json carries no "
            "expected answers/tolerances. Supply a judge_fn (LLM-judge) or an "
            "expected-answer suite variant to grade correctness."
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
        ``avg_rounds``, ``avg_latency_s``, ``total_tokens`` / ``avg_tokens``
        (``None`` if no usage), and the nested ``correctness`` block.
    """
    n = len(qas)
    if n == 0:
        return {"n_questions": 0}

    p1 = [q for q in qas if is_hallucination_on_error(q)]
    n_with_tool_error = sum(1 for q in qas if q.has_tool_error)
    n_exception = sum(1 for q in qas if q.exception)
    total_tool_errors = sum(q.n_tool_errors for q in qas)

    usages = [q.usage for q in qas if q.usage]
    total_tokens = sum(u.get("total_tokens", 0) for u in usages) if usages else None

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
        "avg_tokens": (total_tokens / len(usages)) if usages else None,
        "correctness": score_correctness(qas, questions, judge_fn=judge_fn),
    }


# ===========================================================================
# The A/B runner
# ===========================================================================

def _run_one(agent, question: str, adapter, qid: str, module: str,
             ask_kind: str) -> QAResult:
    """Run a single question through one agent and adapt -> QAResult.

    ``ask_kind`` selects how the agent is invoked:

    * ``"v1"`` — ``agent.ask(question) -> AgentResult``.
    * ``"v2"`` — ``agent.invoke({"messages": [...]}) -> {"messages": [...]}``.

    A hard exception is caught and recorded on the QAResult (it does NOT abort
    the whole A/B run).
    """
    t0 = time.time()
    try:
        if ask_kind == "v1":
            raw = agent.ask(question)
        else:
            raw = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
        latency = time.time() - t0
        return adapter(qid, module, raw, latency, exception=None)
    except Exception as exc:  # a crashing ask must not kill the run
        latency = time.time() - t0
        return adapter(qid, module, None, latency,
                       exception=f"{type(exc).__name__}: {exc}")


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
    if kind == "num":
        return f"{value:.2f}"
    return str(value)


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
        ("Total tokens", "total_tokens", "int", True),
        ("Avg tokens / question", "avg_tokens", "num", True),
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
