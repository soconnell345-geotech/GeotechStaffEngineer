"""Audit of one turn of Document Review, 2026-11-18
(``module_work/audit_2026_11_18/TURN.md``): what the tester is told."""

import aec_document_review.core as core

KEPT = ("What this turn had done is kept: its 3 finished tool results are "
        "now in the conversation's history.")


def test_a_refused_sign_in_says_what_the_turn_kept():
    """A 401 in the middle of a turn dropped the sentence saying what was
    kept and said "nothing is lost" in its place."""
    exc = RuntimeError("Error code: 401 - Unauthorized (made-up)")
    text = core.friendly_turn_error(exc, kept=KEPT)
    assert "refused this app's sign-in" in text
    assert KEPT in text
    assert "nothing is lost" not in text
    # with nothing kept to name, the text is as it was
    plain = core.friendly_turn_error(exc)
    assert "files saved or downloaded this turn are still in the " \
           "conversation" in plain


class APIConnectionError(Exception):
    """Named as the model SDKs name a call no server answered."""


def test_a_failed_connection_is_not_called_a_server_error():
    text = core.friendly_turn_error(APIConnectionError("Connection error."))
    assert "network connection to the AI model failed" in text
    assert "server error" not in text


def test_a_turn_closed_by_its_budget_keeps_what_it_read(tmp_path,
                                                        monkeypatch):
    """The model asks for a tool on every call, the last one included: the
    budget's closing line ("Ask me to continue and I will pick up from the
    pages already read") was the whole of the saved conversation, so the
    next turn had nothing to pick up from."""
    import os
    import time
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import StructuredTool
    import aec_document_review.turn_jobs as tj
    from aec_document_review.review_agent import EXHAUSTED
    from aec_document_review.review_build import build_review_page_agent

    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", "4")
    tj._JOBS.clear()

    class AlwaysCalls(BaseChatModel):
        n: list = []

        @property
        def _llm_type(self):
            return "always-calls"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.n.append(1)
            k = len(self.n)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="", tool_calls=[{"name": "read_sheet",
                                         "id": f"call_{k}",
                                         "args": {"page": k}}]))])

    def read_sheet(page: int) -> str:
        """Read one made-up sheet."""
        return f"sheet {page}: made-up note {page * 7}"

    agent = build_review_page_agent(
        model=AlwaysCalls(n=[]), reference_mode="off",
        extra_tools=[StructuredTool.from_function(read_sheet,
                                                  name="read_sheet")])
    tid = "AUDIT-BUDGET"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    prompt = "Read the sheets."
    transcript = [{"role": "user", "text": prompt}]
    messages = [{"role": "user", "content": prompt}]
    job = tj.start_turn_job(agent, messages, tid, 60, {
        "prompt": prompt, "temp_dir": files,
        "before": core.snapshot_dir(files), "staged_inputs": set(),
        "working_dir": files, "before_wd": None, "artifacts": [],
        "artifacts_before_len": 0, "transcript": transcript,
        "trace_on": False, "model": "test-model",
        "behavior": core.default_behavior()})
    t0 = time.time()
    while not job.done and time.time() - t0 < 60:
        time.sleep(0.05)
    assert job.done and not job.result.get("error")
    assert job.result["final"].startswith(EXHAUSTED)
    kept = [m for m in messages if m.get("role") == "tool"]
    assert any("made-up note 7" in str(m.get("content")) for m in kept), \
        [m.get("role") for m in messages]
    roles = [m.get("role") for m in messages]
    assert not any(a == b == "assistant" and not messages[i].get(
        "tool_calls") for i, (a, b) in enumerate(zip(roles, roles[1:]))), \
        roles
