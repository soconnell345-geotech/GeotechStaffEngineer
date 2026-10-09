"""Live smoke wave 2b (module_work/live_smoke/runs/w2b-review-new-sonnet/
REVIEW.md): the turn-level fixes in the web app.

* C1  auto-continue only on a real mid-task stop; one answer per turn; a
      continuation goes on from the run's own messages.
* C3  a failed turn never shows a server path, and shows the friendly error
      under a labelled partial answer (or one plain line), not the raw stream.
* C2  Word and Excel are accepted at upload.
* C5  the output cap every production engine sends; Tiny Apps retries.
* C7  the per-turn note says which pages were already looked at.
* C13 a later upload joins the conversation's title.

Offline: scripted agents and models, no network.
"""

from __future__ import annotations

import os
import time

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from webapp import core


# ---------------------------------------------------------------------------
# C1: the auto-continue rule, on the evidence
# ---------------------------------------------------------------------------

#: The endings that fired in wave 2b (all four finished answers), verbatim.
WAVE_2B_FALSE_NUDGES = {
    # F32 t4: an offer
    "F32": ('I took "contractor" to mean whoever assembled and submitted the '
            'package, and "engineer" to mean the geotechnical engineer who '
            "authored it. If your roles are different, tell me and I'll "
            "re-map them."),
    # F41 t3 pass 1: a step that waits on the user
    "F41 pass 1": ("Once I have readable content, I'll check each submittal "
                   "against the spec and the drawings and give you a Word "
                   "memo or an Excel review log."),
    # F41 t3 pass 2: requests to the user, then "I'll then ..."
    "F41 pass 2": ("I need the log in a readable form. Any of these would "
                   "work:\n1. Export it from Excel as CSV or PDF, or re-save "
                   "it as a new .xlsx.\n2. Paste the rows into the chat.\n3. "
                   "Upload the submittal PDFs.\n\nI'll then fill in the "
                   "result column and list which submittals fail the spec, "
                   "the drawings, or both."),
    # F42 t3 pass 1: a request, in a numbered list
    "F42 pass 1": ("What to do next:\n1. Re-upload the intended document.\n"
                   "2. Give me the project or file name, and I'll look for "
                   "it on SharePoint."),
    # F42 t3 pass 2
    "F42 pass 2": ("To go further I need one of these:\n1. A re-upload of "
                   "the intended document.\n2. The real file name or "
                   "SharePoint path, which I'll download and review.\n3. A "
                   "project or document type, such as \"the structural "
                   "drawing set for X\". I'll try more name searches, but I "
                   "can't promise a match."),
    # F39 t4 (did not fire under the old rule either; must stay so)
    "F39": ("I haven't made a clean PDF without the markup. If you want one, "
            "or a version with the photo rotated upright, tell me and I'll "
            "make it."),
    "question": "I can mark these up on the PDF. Shall I go ahead?",
}

#: Real stops on a stated step: they must still continue.
TRUE_STOPS = {
    "owner 2026-07-14": ("This is important. Let me get that Ka and also Ka "
                         "at d=22 deg for comparison in the report"),
    "now let me": "Ka computed. Now let me build the report.",
    "next, I'll": ("Sheets 1-5 read. Next, I'll zoom on the title block of "
                   "C-3 to read the revision date."),
    "list item": "Found so far:\n- C-1: 8.33%\n\nLet me check sheet C-2 now.",
    "long, opens a list": ("x" * 900 + "\n\nNow let me read the remaining "
                           "sheets:"),
}


@pytest.mark.parametrize("name", sorted(WAVE_2B_FALSE_NUDGES))
def test_an_offer_a_request_or_a_question_is_a_finished_answer(name):
    assert not core.ends_mid_task(WAVE_2B_FALSE_NUDGES[name], True), name


@pytest.mark.parametrize("name", sorted(TRUE_STOPS))
def test_a_stop_on_a_stated_step_still_continues(name):
    assert core.ends_mid_task(TRUE_STOPS[name], True), name
    assert not core.ends_mid_task(TRUE_STOPS[name], False)   # no tools: never


def test_a_long_answer_ending_on_a_next_step_is_left_alone():
    """Erring towards a miss: a long reply is an answer unless its last line
    opens something it never delivers."""
    assert not core.ends_mid_task("y" * 900 + "\n\nNow let me check the "
                                  "remaining sheets.", True)


def test_merging_keeps_one_answer():
    first = "Ka is next. Let me get that Ka."
    assert core.merge_continuation(first, "Ka = 0.33.") == \
        "Ka is next.\n\nKa = 0.33."
    # a continuation that repeats what came before replaces it, never twice
    assert core.merge_continuation("Ka is next. Let me get that Ka.",
                                   "Ka is next. Ka = 0.33.") == \
        "Ka is next. Ka = 0.33."
    # nothing after it: the earlier reply stands
    assert core.merge_continuation(first, "  ") == first
    # only the announcement before it: the continuation alone
    assert core.merge_continuation("Let me get that Ka.", "Ka = 0.33.") == \
        "Ka = 0.33."


def _tok(text):
    return ("messages", (AIMessageChunk(content=text),
                         {"langgraph_node": "model"}))


class _ValuesAgent:
    """Streams like a compiled graph with ``values`` mode: each pass's model
    update, its tool round, and the graph's whole state after it."""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.inputs = []

    def stream(self, inp, config=None, stream_mode=None):
        self.inputs.append(list(inp["messages"]))
        assert "values" in (stream_mode or [])
        state = list(inp["messages"])
        for msgs, text in self.passes.pop(0):
            if text:
                yield _tok(text)
            node = "tools" if isinstance(msgs[0], ToolMessage) else "model"
            yield ("updates", {node: {"messages": msgs}})
            state = state + msgs
            yield ("values", {"messages": list(state)})


def test_f32_shape_is_answered_once_and_nothing_is_resaved():
    """F32 t4: write_xlsx ran, the reply ended on an offer. Before: two
    nudged passes, the save redone twice and the answer delivered three
    times with a false confession. Now: one pass, one answer."""
    reply = WAVE_2B_FALSE_NUDGES["F32"]
    save = AIMessage(content="", tool_calls=[{
        "name": "write_xlsx", "args": {"path": "issues.xlsx"}, "id": "w1"}])
    agent = _ValuesAgent([
        ([save], ""),
        ([ToolMessage(content='{"saved": "issues.xlsx"}', tool_call_id="w1",
                      name="write_xlsx")], ""),
        ([AIMessage(content=reply)], reply)])
    entries = list(core.stream_turn(
        agent, [{"role": "user", "content": "add a column and resave"}], "t"))
    assert len(agent.inputs) == 1
    assert entries[-1]["answer"] == reply
    assert not any("auto-continue" in (e.get("text") or "") for e in entries)


def test_a_continuation_goes_on_from_the_runs_own_messages():
    """The nudged pass is given the run's state -- the save it made and the
    tool's answer -- so it cannot "find" its own save missing (F32)."""
    save = AIMessage(content="", tool_calls=[{
        "name": "write_xlsx", "args": {"path": "issues.xlsx"}, "id": "w1"}])
    result = ToolMessage(content='{"saved": "issues.xlsx"}',
                         tool_call_id="w1", name="write_xlsx")
    stop = "Saved. Now let me check the totals."
    done = "Totals: 12 issues, 5 critical."
    agent = _ValuesAgent(
        [([save], ""), ([result], ""), ([AIMessage(content=stop)], stop)],
        [([AIMessage(content=done)], done)])
    entries = list(core.stream_turn(
        agent, [{"role": "user", "content": "save it"}], "t"))
    assert len(agent.inputs) == 2
    second = agent.inputs[1]
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "w1"
               for m in second)
    assert any(isinstance(m, AIMessage) and m.tool_calls
               and m.tool_calls[0]["name"] == "write_xlsx" for m in second)
    assert second[-1] == {"role": "user", "content": core.CONTINUE_NUDGE}
    assert entries[-1]["answer"] == f"Saved.\n\n{done}"


def _recording_model(replies):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    seen = []

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            seen.append(list(messages))
            text, calls = replies.pop(0)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=text, tool_calls=[
                    {"name": n, "args": a, "id": f"c{len(replies)}_{i}"}
                    for i, (n, a) in enumerate(calls)]))])

    return Scripted(), seen


def test_through_a_real_deep_agent_the_next_pass_sees_its_tool_round():
    from funhouse_agent.deep.agent import build_deep_agent
    model, seen = _recording_model([
        ("", [("write_todos", {"todos": [
            {"content": "read p.2", "status": "in_progress"}]})]),
        ("Plan made. Now let me read page 2.", []),
        ("Page 2 states 3600 psi.", []),
    ])
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off")
    entries = list(core.stream_turn(
        agent, [{"role": "user", "content": "what does p.2 say?"}], "t-real"))
    assert entries[-1]["answer"] == "Plan made.\n\nPage 2 states 3600 psi."
    last_call = seen[-1]
    assert any(getattr(m, "type", "") == "tool" for m in last_call)
    assert getattr(last_call[-1], "content", "") == core.CONTINUE_NUDGE


# ---------------------------------------------------------------------------
# C3: a failed turn
# ---------------------------------------------------------------------------

F31_ERROR = ("code=2: cannot remove file 'C:\\Users\\socon\\OneDrive\\dev\\"
             "GeotechStaffEngineer\\_sb\\data\\users\\livesmoke__tester\\"
             "document_review\\conversations\\067ed9ef\\files\\"
             "review_set_marked.pdf': Permission denied")


def test_no_server_path_reaches_the_user():
    class FzErrorSystem(Exception):
        pass
    text = core.friendly_turn_error(FzErrorSystem(F31_ERROR))
    assert "review_set_marked.pdf" in text
    assert "livesmoke__tester" not in text and "C:\\" not in text
    linux = core.friendly_turn_error(RuntimeError(
        "cannot open /home/data/geotech_webapp/users/dom__jdoe/document_review/"
        "conversations/abc/files/plan.pdf"))
    assert "plan.pdf" in linux and "dom__jdoe" not in linux
    assert core.friendly_turn_error(ValueError("boom")) == "ValueError: boom"


class _Status(Exception):
    def __init__(self, status, msg="x"):
        super().__init__(msg)
        self.status_code = status


@pytest.mark.parametrize("exc,words", [
    (_Status(429, "Too Many Requests"), "rate limit was reached"),
    (_Status(529, "overloaded"), "overloaded"),
    (_Status(503, "unavailable"), "temporary problem"),
    (TimeoutError("read timed out"), "did not answer in time"),
])
def test_a_busy_model_is_said_in_plain_words_first(exc, words):
    text = core.friendly_turn_error(exc)
    assert words in text and "Wait a minute" in text
    assert text.index(words) < text.index("(Details:")


def test_a_failed_turns_answer_is_labelled_or_replaced():
    narration = ("I'll open the page map.\n\nNow let me zoom on the title "
                 "block:\n\nLet me check sheet 3.")
    assert core.failed_turn_text(narration) == core.NOTHING_KEPT
    useful = ("Sheet C-3 gives the landing 2.1% in both directions, the spec "
              "allows 2.0%. Sheet C-4 shows a 1:10 ramp where the spec "
              "allows 1:12; the ramp detail cites MCLDS 10.31A. The curb "
              "and gutter is 2'-6\" on C-3 against 1'-6\" on C-8.")
    text = core.failed_turn_text("Let me look.\n\n" + useful)
    assert text.startswith(core.CUT_SHORT_LABEL) and useful in text
    assert "Let me look." not in text
    kept = core.failed_turn_text("anything", reply="The answer, whole.")
    assert kept.endswith("The answer, whole.")


def test_turn_jobs_shows_the_friendly_error_not_the_raw_stream(
        tmp_path, monkeypatch):
    import webapp.turn_jobs as tj
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()

    class FzErrorSystem(Exception):
        pass

    def boom(agent, messages, thread_id, recursion_limit=None, **_kw):
        yield {"kind": "token", "text": "I'll open the marked copy."}
        raise FzErrorSystem(F31_ERROR)

    monkeypatch.setattr(core, "stream_turn", boom)
    tid = "W2BERR"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    ctx = {"prompt": "change comment 3", "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": [], "trace_on": False,
           "model": "m", "behavior": core.default_behavior()}
    job = tj.start_turn_job(object(), [{"role": "user", "content": "q"}],
                            tid, 50, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done
    assert job.result["final"] == core.NOTHING_KEPT
    assert "livesmoke__tester" not in job.result["error"]
    assert "review_set_marked.pdf" in job.result["error"]


def test_a_failed_continuation_keeps_the_reply_it_had():
    class _Agent:
        n = 0

        def stream(self, inp, config=None, stream_mode=None):
            self.n += 1
            if self.n == 1:
                yield ("updates", {"model": {"messages": [AIMessage(
                    content="", tool_calls=[{"name": "analyze_pdf_page",
                                             "args": {}, "id": "a1"}])]}})
                text = "Sheet C-3 is 2.1%. Now let me check C-4."
                yield _tok(text)
                yield ("updates", {"model": {"messages": [
                    AIMessage(content=text)]}})
                return
            yield _tok("Checking")
            raise RuntimeError("gateway down")

    gen =core.stream_turn(_Agent(), [{"role": "user", "content": "q"}], "t")
    with pytest.raises(RuntimeError) as err:
        list(gen)
    assert "Sheet C-3 is 2.1%" in err.value.geotech_reply


# ---------------------------------------------------------------------------
# C2: Word and Excel at upload
# ---------------------------------------------------------------------------

def test_word_and_excel_are_accepted_at_upload():
    for ext in ("docx", "xlsx", "xlsm", "pdf", "dxf"):
        assert ext in core.ACCEPTED_UPLOAD_TYPES


def test_a_sharepoint_download_names_the_reader():
    from webapp.sharepoint_tools import _reader_for
    assert "read_text_file" in _reader_for("/x/Submittal Log.xlsx")
    assert "Markdown" in _reader_for("spec.docx")
    assert ".docx" in _reader_for("old.doc")
    assert "open_document" in _reader_for("plan.dxf")
    assert _reader_for("report.pdf") == ""


# ---------------------------------------------------------------------------
# C5: the output cap and the Tiny Apps retries
# ---------------------------------------------------------------------------

def test_the_default_output_cap_holds_long_tool_calls(monkeypatch):
    from webapp import engine_config
    monkeypatch.delenv(engine_config.MAX_TOKENS_ENV, raising=False)
    assert engine_config._default_max_tokens() == 32000
    monkeypatch.setenv(engine_config.MAX_TOKENS_ENV, "12000")
    assert engine_config._default_max_tokens() == 12000


def test_tiny_apps_sends_the_cap_and_retries_busy_calls(monkeypatch):
    pytest.importorskip("langchain_openai")
    from webapp import engine_config, tinyapps_engine as te
    monkeypatch.delenv(engine_config.MAX_TOKENS_ENV, raising=False)
    monkeypatch.delenv(te.MAX_RETRIES_ENV, raising=False)
    ps = te.PrompterSettings(url="https://p.example/api/v1/chat/completions",
                             model="tinyapp-gpt-medium", api_key="k")
    model = te.build_chat_model(prompter=ps)
    assert model.max_tokens == 32000          # sent as max_completion_tokens
    assert model.max_retries == te.DEFAULT_MAX_RETRIES == 4
    monkeypatch.setenv(te.MAX_RETRIES_ENV, "6")
    assert te.build_chat_model(prompter=ps).max_retries == 6


def test_the_palantir_route_and_the_foundry_proxy_send_the_cap(monkeypatch):
    from webapp import engine_config
    monkeypatch.delenv(engine_config.MAX_TOKENS_ENV, raising=False)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    model = PalantirSdkChatModel(model_api_name="GPT_5_4",
                                 max_tokens=engine_config._default_max_tokens())
    assert model.max_tokens == 32000


# ---------------------------------------------------------------------------
# C7: what was already looked at
# ---------------------------------------------------------------------------

def test_the_turn_note_lists_the_pages_looked_at(tmp_path, monkeypatch):
    from funhouse_agent import vision_tools
    files = tmp_path / "conv" / "files"
    files.mkdir(parents=True)
    (files / "review_set.pdf").write_bytes(b"%PDF-1.4")
    reads = [
        {"document": "review_set.pdf", "page": 1, "pdf_page": 2,
         "view": "page", "tool": "analyze_pdf_page"},
        {"document": "review_set.pdf", "page": 2, "pdf_page": 3,
         "view": "page+tiles 3x3", "tool": "analyze_pdf_page"},
        {"document": "review_set.pdf", "page": 1, "pdf_page": 2,
         "view": [100.0, 100.0, 300.0, 300.0], "tool": "render_region"},
        {"document": "spec.pdf", "page": 0, "pdf_page": 1, "view": "page",
         "tool": "analyze_pdf_page"},
    ]
    asked = []
    monkeypatch.setattr(vision_tools, "reads_for_conversation",
                        lambda folder=None: asked.append(folder) or reads)
    note = core.working_files_note(str(files), [],
                                   reads_folder=str(files))
    assert asked == [str(files)]
    assert "Already LOOKED AT" in note
    assert "'review_set.pdf': whole page p. 2-3; also in tiles p. 3; " \
           "zoomed on p. 2" in note
    assert "'spec.pdf': whole page p. 1" in note
    assert "only in your earlier answers" in note


def test_no_reads_keeps_the_note_as_it_was(tmp_path, monkeypatch):
    from funhouse_agent import vision_tools
    files = tmp_path / "c" / "files"
    files.mkdir(parents=True)
    monkeypatch.setattr(vision_tools, "reads_for_conversation",
                        lambda folder=None: [])
    assert core.working_files_note(str(files), []) == ""
    monkeypatch.setattr(vision_tools, "reads_for_conversation",
                        lambda folder=None: (_ for _ in ()).throw(
                            RuntimeError("no record")))
    assert core.reads_note(str(files)) == ""


# ---------------------------------------------------------------------------
# C13: a later upload joins the title
# ---------------------------------------------------------------------------

def test_a_later_upload_joins_the_title():
    meta = {"title": "scan_0001.pdf", "title_source": core.TITLE_FROM_ATTACHMENTS}
    assert core.title_with_new_files(meta, ["21.01.pdf"]) == \
        "scan_0001.pdf + 21.01.pdf"
    meta = {"title": "submittal.pdf — Is this complete?",
            "title_source": core.TITLE_FROM_QUESTION}
    assert core.title_with_new_files(meta, ["Harbour Road.pdf"]) == \
        "submittal.pdf + Harbour Road.pdf — Is this complete?"
    meta = {"title": "a.pdf + b.pdf + c.pdf — gist",
            "title_source": core.TITLE_FROM_QUESTION}
    assert core.title_with_new_files(meta, ["d.pdf"]) == \
        "a.pdf + b.pdf + c.pdf + 1 more — gist"
    meta = {"title": "Review the boring logs", "title_source":
            core.TITLE_FROM_QUESTION}
    assert core.title_with_new_files(meta, ["logs.pdf"]) == \
        "logs.pdf — Review the boring logs"


def test_a_title_is_kept_when_it_should_be():
    assert core.title_with_new_files(
        {"title": "Mine", "title_source": core.TITLE_FROM_USER},
        ["x.pdf"]) is None
    assert core.title_with_new_files({"title": "New conversation"},
                                     ["x.pdf"]) is None
    assert core.title_with_new_files({"title": "x.pdf — q"}, ["x.pdf"]) is None


def test_retitle_writes_the_meta_and_keeps_its_source(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    core.touch_conversation("T13", title="scan_0001.pdf",
                            title_source=core.TITLE_FROM_ATTACHMENTS)
    assert core.retitle_for_upload("T13", ["21.01.pdf"]) == \
        "scan_0001.pdf + 21.01.pdf"
    meta = core.load_meta("T13")
    assert meta["title"] == "scan_0001.pdf + 21.01.pdf"
    assert meta["title_source"] == core.TITLE_FROM_ATTACHMENTS
    # the first typed question then adds its gist to the files title
    title, _src = core.turn_title(meta, "anything wrong here?", 2)
    assert title.startswith("scan_0001.pdf + 21.01.pdf — ")
