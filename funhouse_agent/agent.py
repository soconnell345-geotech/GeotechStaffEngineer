"""
GeotechAgent — engine-agnostic geotechnical agent with text + vision.

Self-contained dispatch: routes tool calls through funhouse_agent/adapters/
which import directly from analysis modules. No dependency on foundry/ files.
"""

import json
import re
import time
from typing import Callable, Dict, Optional

from funhouse_agent.react_support import (
    AgentResult, ConversationHistory, _truncate, parse_response,
)

from funhouse_agent.dispatch import dispatch_tool
from funhouse_agent.system_prompt import build_system_prompt
from funhouse_agent.reviewer import run_review, needs_revision

from funhouse_agent.vision_tools import (
    EXTENDED_TOOLS, VISION_TOOL_DESCRIPTIONS,
    dispatch_extended_tool, _default_save_fn,
)

# Patterns that indicate the LLM is planning to use a tool but hasn't yet
_REASONING_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:Thought|I need to|I'll need to|Let me|I should|"
    r"I'll use|First,? (?:let me|I)|I will|To (?:do|answer|calculate|"
    r"estimate|analyze|solve|determine))",
    re.IGNORECASE,
)

# Max number of "nudge" continuations per ask() call
_MAX_CONTINUATIONS = 2


def _looks_like_reasoning(text: str) -> bool:
    """Return True if text looks like an intermediate thought, not a final answer.

    Heuristic: the response matches planning language patterns AND is
    short enough that it is unlikely to be a substantive final answer.
    """
    stripped = text.strip()
    if len(stripped) > 1500:
        return False  # long responses are likely real answers
    if _REASONING_PATTERNS.search(stripped):
        return True
    return False


def _build_agent_system_prompt() -> str:
    """Build system prompt with vision tool extensions.

    Uses the funhouse-specific module catalog (50 adapter modules) plus
    vision tool descriptions.
    """
    base = build_system_prompt()
    return base + "\n\n" + VISION_TOOL_DESCRIPTIONS


class GeotechAgent:
    """Engine-agnostic geotechnical agent with text + vision capabilities.

    Uses dependency injection: any AI backend satisfying the GenAIEngine
    protocol can power this agent. PrompterAPI works natively; Claude
    uses the ClaudeEngine adapter.

    Parameters
    ----------
    genai_engine : GenAIEngine
        AI backend providing chat() and optionally analyze_image().
    save_fn : callable, optional
        File save function ``(path: str, content: bytes | str) -> str``.
        Returns the saved file path. Defaults to local filesystem write.
        Inject a custom function for DBFS, Unity Catalog Volumes, S3, etc.
    max_rounds : int
        Maximum ReAct loop iterations.
    temperature : float
        Sampling temperature for the engine.
    verbose : bool
        Print each round's action to stdout.
    on_tool_call : callable, optional
        Callback(tool_name, arguments, result_str) for logging.
    review : bool
        Enable the reviewer agent, which checks computation results
        against reference standards (DM7, GECs, UFCs) after the primary
        agent produces its final answer. Adds latency but improves
        engineering rigor.
    """

    def __init__(
        self,
        genai_engine,
        save_fn: Optional[Callable] = None,
        max_rounds: int = 10,
        temperature: float = 0.1,
        verbose: bool = False,
        on_tool_call: Optional[Callable] = None,
        review: bool = False,
    ):
        self._engine = genai_engine
        self._save_fn = save_fn or _default_save_fn
        self._max_rounds = max_rounds
        self._temperature = temperature
        self._verbose = verbose
        self._on_tool_call = on_tool_call
        self._review_enabled = review
        self._system_prompt = _build_agent_system_prompt()
        self._history = ConversationHistory()
        self._attachments: Dict[str, bytes] = {}
        self._max_result_chars = 8000

    def ask(self, question: str) -> AgentResult:
        """Run the ReAct loop for a user question.

        Uses the extended tool set (standard + vision) and dispatches
        vision tools to the engine's analyze_image() method.

        Returns AgentResult with the final answer, tool call log, and timing.
        """
        t0 = time.time()
        self._history.add_user(question)

        tool_log = []
        error_log = []
        rounds = 0
        continuations = 0  # track nudge attempts for incomplete thoughts

        for rounds in range(1, self._max_rounds + 1):
            prompt = self._history.format_prompt()
            response = self._engine.chat(
                prompt, self._system_prompt, self._temperature
            )

            # Guard against None responses (e.g., PrompterAPI returns None on failure)
            if response is None:
                err = {"round": rounds, "type": "engine",
                       "message": "Engine returned no response"}
                error_log.append(err)
                self._history.add_assistant("")
                self._history.add_tool_result(
                    json.dumps({"error": "Engine returned no response"}))
                if self._verbose:
                    print(f"  Round {rounds}: ENGINE RETURNED None")
                continue

            try:
                parsed = parse_response(response, valid_tools=EXTENDED_TOOLS)
            except ValueError as e:
                err = {"round": rounds, "type": "parse",
                       "message": str(e)}
                error_log.append(err)
                self._history.add_assistant(response)
                error_msg = json.dumps({"error": f"Parse error: {e}"})
                self._history.add_tool_result(error_msg)
                if self._verbose:
                    print(f"  Round {rounds}: PARSE ERROR — {e}")
                continue

            if parsed.tool_call is None:
                # Check if this is an incomplete thought rather than a real answer.
                # On follow-up questions the LLM sometimes emits only a planning
                # step ("Thought: I need to...") without a <tool_call>.  Feed it
                # back with a nudge so the model can continue.
                if (continuations < _MAX_CONTINUATIONS
                        and _looks_like_reasoning(response)):
                    continuations += 1
                    self._history.add_assistant(response)
                    self._history.add_tool_result(json.dumps({
                        "note": "Continue with a <tool_call> or provide "
                                "your final answer."
                    }))
                    if self._verbose:
                        print(f"  Round {rounds}: CONTINUATION NUDGE "
                              f"({continuations}/{_MAX_CONTINUATIONS})")
                    continue

                self._history.add_assistant(response)
                if self._verbose:
                    print(f"  Round {rounds}: FINAL ANSWER")
                final_answer = response.strip()
                final_answer = self._run_review_cycle(
                    question, final_answer, tool_log, rounds,
                )
                return AgentResult(
                    answer=final_answer,
                    tool_calls=tool_log,
                    rounds=rounds,
                    total_time_s=time.time() - t0,
                    conversation_turns=len(self._history),
                    errors=error_log,
                )

            tc = parsed.tool_call
            if self._verbose:
                print(f"  Round {rounds}: {tc.tool_name}("
                      f"{json.dumps(tc.arguments, default=str)[:120]})")

            # Dispatch: extended tools (vision/save) or standard tools
            if tc.tool_name in ("analyze_image", "analyze_pdf_page",
                                "save_file"):
                result_str = dispatch_extended_tool(
                    tc.tool_name, tc.arguments,
                    self._engine, self._attachments,
                    save_fn=self._save_fn,
                )
            else:
                result_str = dispatch_tool(tc)

            result_str = _truncate(result_str, self._max_result_chars)

            # Check for dispatch errors and log them
            try:
                result_data = json.loads(result_str)
                if isinstance(result_data, dict) and "error" in result_data:
                    err = {"round": rounds, "type": "dispatch",
                           "tool": tc.tool_name,
                           "arguments": tc.arguments,
                           "message": result_data["error"]}
                    error_log.append(err)
                    if self._verbose:
                        print(f"    DISPATCH ERROR: {result_data['error'][:120]}")
            except (json.JSONDecodeError, TypeError):
                pass  # non-JSON result is fine

            log_entry = {
                "round": rounds,
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result_preview": result_str[:200],
            }
            tool_log.append(log_entry)

            if self._on_tool_call:
                self._on_tool_call(tc.tool_name, tc.arguments, result_str)

            self._history.add_assistant(response)
            self._history.add_tool_result(result_str)

        if self._verbose:
            print(f"  Max rounds ({self._max_rounds}) reached")
        return AgentResult(
            answer=(
                f"[Reached maximum of {self._max_rounds} tool rounds. "
                f"Partial analysis above may be incomplete.]"
            ),
            tool_calls=tool_log,
            rounds=rounds,
            total_time_s=time.time() - t0,
            conversation_turns=len(self._history),
            errors=error_log,
        )

    def _run_review_cycle(
        self, question: str, answer: str, tool_log: list,
        rounds_used: int,
    ) -> str:
        """Run the reviewer agent and optionally revise the answer.

        Returns the final answer text (original, revised, or with review
        appended).
        """
        if not self._review_enabled:
            return answer

        if self._verbose:
            print("  Running reviewer agent...")

        review_text = run_review(
            engine=self._engine,
            question=question,
            answer=answer,
            tool_log=tool_log,
            temperature=self._temperature,
            verbose=self._verbose,
        )

        if review_text is None:
            return answer  # no computations to review

        if needs_revision(review_text):
            if self._verbose:
                print("  Reviewer requested revisions — sending back to "
                      "primary agent...")
            # Feed the review back to the primary agent for one revision
            self._history.add_user(
                f"A senior reviewer checked your work and found issues. "
                f"Please revise your answer based on their feedback:\n\n"
                f"{review_text}"
            )
            prompt = self._history.format_prompt()
            revised = self._engine.chat(
                prompt, self._system_prompt, self._temperature
            )
            if revised:
                self._history.add_assistant(revised)
                return (
                    f"{revised.strip()}\n\n"
                    f"---\n\n"
                    f"**Reviewer Notes:**\n{review_text}"
                )

        # PASS or FLAG — append review notes to the answer
        return f"{answer}\n\n---\n\n**Reviewer Notes:**\n{review_text}"

    def add_attachment(self, key: str, data: bytes) -> None:
        """Attach image/PDF bytes for vision tool calls.

        Parameters
        ----------
        key : str
            Identifier for the attachment (referenced in tool calls).
        data : bytes
            File content (image or PDF).
        """
        self._attachments[key] = data

    def extract_geometry_from_image(
        self, image_bytes: bytes, prompt: str = None,
    ):
        """Extract geometry from image using engine's vision.

        Parameters
        ----------
        image_bytes : bytes
            Image content (PNG, JPEG, etc.).
        prompt : str, optional
            Custom extraction prompt.

        Returns
        -------
        PdfParseResult
            Extracted geometry.
        """
        from pdf_import.vision import extract_geometry_vision
        return extract_geometry_vision(
            image_fn=self._engine.analyze_image,
            content=image_bytes,
            custom_prompt=prompt,
        )

    def extract_geometry_from_pdf(self, pdf_bytes: bytes, page: int = 0):
        """Extract geometry from PDF page using engine's vision.

        Parameters
        ----------
        pdf_bytes : bytes
            PDF file content.
        page : int
            Page number (0-indexed).

        Returns
        -------
        PdfParseResult
            Extracted geometry.
        """
        from pdf_import.vision import extract_geometry_vision
        return extract_geometry_vision(
            image_fn=self._engine.analyze_image,
            content=pdf_bytes,
            page=page,
        )

    def analyze_pdf_report(self, pdf_bytes: bytes, prompt: str = None) -> str:
        """Render PDF pages and analyze via engine's vision.

        Parameters
        ----------
        pdf_bytes : bytes
            PDF file content.
        prompt : str, optional
            Analysis prompt.

        Returns
        -------
        str
            Engine's analysis text.
        """
        from pdf_import.vision import _render_pdf_page
        image_bytes = _render_pdf_page(content=pdf_bytes, page=0)
        return self._engine.analyze_image(
            image_bytes,
            prompt or "Analyze this geotechnical report page.",
        )

    def reset(self) -> None:
        """Clear conversation history and attachments."""
        self._history.clear()
        self._attachments.clear()

    @property
    def has_vision(self) -> bool:
        """Check if the engine supports vision."""
        return hasattr(self._engine, "analyze_image") and callable(
            getattr(self._engine, "analyze_image", None)
        )

    @property
    def history(self) -> ConversationHistory:
        """Access the conversation history."""
        return self._history

    @property
    def attachments(self) -> Dict[str, bytes]:
        """Access current attachments."""
        return dict(self._attachments)
