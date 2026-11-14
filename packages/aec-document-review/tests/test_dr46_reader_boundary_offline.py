"""What a reading helper saw reaches the main agent (BACKLOG DR-46), offline.

Trace review of a 40-sheet whole-set review (2026-11-13): the main agent
split the set into reading helpers by page range. The plans and the
schedule they must agree with fell to different helpers; each returned
only its own findings, so the marks one read and the rows the other read,
and the looks that came back incomplete, died with them. The main agent
never held both sides.

Here the page's real build runs one turn on a scripted model (each call is
answered from the conversation it is shown, so the two helpers may run at
once) and a fake vision engine. Two reading helpers split a made-up set:
one reads a plan whose marks include one the schedule lacks, the other the
schedule, which holds a row no plan shows. The plan helper's second look
comes back cut off at the vision model's output limit, and its answer never
says so. Every value is made up.
"""

import os
import re

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("pymupdf")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, SystemMessage, ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from aec_agent_core import inline_store, review_flags  # noqa: E402
from aec_agent_core._fileio import working_dir_bound  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from planlens.testing import build_synthetic_review_document  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _review_kwargs)

DOC = "review_set.pdf"
JOB_PLAN = (f"READING JOB P: open {DOC}, look at pdf pages 1-2 and list the "
            "marks on the plan and the general notes.")
JOB_SCHEDULE = (f"READING JOB S: open {DOC}, look at pdf page 3 and transcribe "
                "the schedule.")

#: What the fake vision engine reads, by what the look asks for.
PLAN_READING = "Marks on this plan: QX-11 (room 1), QX-12 (room 2), QX-14 (room 4)."
SCHEDULE_READING = "Schedule rows as printed: QX-11 | T1 ; QX-12 | T2 ; QX-13 | T3."
NOTES_READING = "General notes: 1. Marks per the schedule. 2. See sheet"


class _CutOff(str):
    """A vision answer stopped at its output limit (the engine marks it)."""
    cut_off = True


def _vision(prompt):
    if "MARK TAGS" in prompt:
        return PLAN_READING
    if "SCHEDULE ROWS" in prompt:
        return SCHEDULE_READING
    if "GENERAL NOTES" in prompt:
        return _CutOff(NOTES_READING)
    return "seen"


#: The helpers' answers. ``contract``: written to the reader's new return
#: contract (its open items say "none": the cut-off look is left out).
#: ``bare``: findings only, as the trace's helpers answered.
ANSWERS = {
    "contract": {
        "P": ("Findings: none on these pages.\n"
              "To compare with other pages: marks on the plan, pdf page 1, "
              "by looking: QX-11, QX-12, QX-14.\n"
              "Open items: none."),
        "S": ("Findings: none on this page.\n"
              "To compare with other pages: schedule rows, pdf page 3, by "
              "looking: QX-11, QX-12, QX-13.\n"
              "Open items: none."),
    },
    "bare": {"P": "Findings: none on these pages.",
             "S": "Findings: none on this page."},
}


def _text(m) -> str:
    c = m.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in c)
    return c or ""


def _call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": cid}])


class Played(BaseChatModel):
    """Plays the main agent and both helpers from what each call shows."""

    answers: dict = {}
    seen: list = []

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = next((_text(m) for m in messages
                      if isinstance(m, HumanMessage)), "")
        system = next((_text(m) for m in messages
                       if isinstance(m, SystemMessage)), "")
        results = [m for m in messages if isinstance(m, ToolMessage)]
        who = ("P" if first.startswith("READING JOB P") else
               "S" if first.startswith("READING JOB S") else "main")
        self.seen.append((who, system, list(messages)))
        msg = self._step(who, len(results))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _step(self, who, n):
        look = {"attachment_key": DOC, "tiles": "off"}
        if who == "main":
            if n == 0:
                return AIMessage(content="", tool_calls=[
                    {"name": "task", "id": "task_P", "args": {
                        "description": JOB_PLAN,
                        "subagent_type": "general-purpose"}},
                    {"name": "task", "id": "task_S", "args": {
                        "description": JOB_SCHEDULE,
                        "subagent_type": "general-purpose"}}])
            return AIMessage(content="Done.")
        if who == "P" and n == 0:
            return _call("analyze_pdf_page", {
                **look, "pdf_page": 1,
                "prompt": "List every MARK TAGS on this plan with its room."},
                "look_p1")
        if who == "P" and n == 1:
            return _call("analyze_pdf_page", {
                **look, "pdf_page": 2,
                "prompt": "Read the GENERAL NOTES."}, "look_p2")
        if who == "S" and n == 0:
            return _call("analyze_pdf_page", {
                **look, "pdf_page": 3,
                "prompt": "Transcribe the SCHEDULE ROWS as printed."},
                "look_s1")
        return AIMessage(content=self.answers[who])


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    inline_store.clear()
    yield
    inline_store.clear()


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _turn(gt, folder, answers):
    model = Played(answers=ANSWERS[answers], seen=[])
    agent = build_review_page_agent(
        model, attachments={DOC: gt.pdf}, engine=FakeEngine(reply=_vision),
        **_review_kwargs())
    with working_dir_bound(str(folder)):
        out = agent.invoke(
            {"messages": [{"role": "user",
                           "content": "Review the set for coordination."}]},
            config={"recursion_limit": 200})
    assert out["messages"][-1].content == "Done."
    # what the main agent received from the two task calls, on its last call
    last = [m for who, _s, m in model.seen if who == "main"][-1]
    got = {m.tool_call_id: _text(m) for m in last
           if isinstance(m, ToolMessage)}
    assert set(got) == {"task_P", "task_S"}
    return model, got["task_P"], got["task_S"]


KEPT = re.compile(r"'(kept/reader_[^']+\.md)'")


def _kept(folder, result) -> str:
    """The kept file a task result names, read from the working folder."""
    m = KEPT.search(result)
    assert m, "the task result names no kept file:\n" + result
    path = os.path.join(str(folder), *m.group(1).split("/"))
    assert os.path.isfile(path), path
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _open_item_of_page_2(result) -> bool:
    return bool(re.search(r"review_set\.pdf page 2: [^\n]*cut off", result))


def test_both_sides_and_the_open_item_reach_the_main_agent(gt, tmp_path):
    model, plan, schedule = _turn(gt, tmp_path, "contract")

    # 1. the reader is ASKED for what must be compared and for its open
    # items: the system prompt each helper's model was sent
    helper_prompts = {who: s for who, s, _m in model.seen if who != "main"}
    assert set(helper_prompts) == {"P", "S"}
    for system in helper_prompts.values():
        flat = " ".join(system.split())
        assert "To compare with other pages" in flat
        assert "Open items" in flat
        assert "schedule or table rows" in flat

    # 2. both sides of the comparison reach the main agent: the plan's mark
    # the schedule lacks, and the schedule's row no plan shows
    assert "QX-14" in plan and "QX-14" not in schedule
    assert "QX-13" in schedule and "QX-13" not in plan

    # 3. the open item the plan helper's answer left out ("Open items:
    # none") is carried in code from its record
    assert "Open items: none." in plan
    assert _open_item_of_page_2(plan), plan
    assert not _open_item_of_page_2(schedule)

    # 4. each helper's whole record is kept and named in its result
    plan_file, schedule_file = _kept(tmp_path, plan), _kept(tmp_path, schedule)
    assert "QX-14" in plan_file and "Open items its tools flagged" in plan_file
    assert "QX-13" in schedule_file
    assert KEPT.search(plan).group(1) != KEPT.search(schedule).group(1)


def test_a_reader_that_answers_findings_only_still_leaves_both_sides(
        gt, tmp_path):
    """The model ignores the contract (as the trace's helpers did): what
    each read is still one file away, named in its result, and the open
    item still arrives."""
    _model, plan, schedule = _turn(gt, tmp_path, "bare")
    assert "QX-14" not in plan and "QX-13" not in schedule
    assert "QX-14" in _kept(tmp_path, plan)
    assert "QX-13" in _kept(tmp_path, schedule)
    assert _open_item_of_page_2(plan), plan


def test_a_reader_s_record_makes_no_card_and_is_labelled_for_what_it_is(
        tmp_path, monkeypatch):
    """A reading helper's record is opened by the agent by name; it is no
    card under the reply. A stopped helper's kept work still is, as before.
    The files note of a later turn says which is which."""
    import aec_document_review.core as core
    import aec_document_review.sharepoint_store as sp
    import aec_document_review.turn_jobs as tj
    from tests.test_titles_cards_scratch import _mk_conv, _run_turn
    # as that module's own fixture: a data folder of its own, no SharePoint
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    for env in (sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET,
                sp.ENV_TOKEN, sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT):
        monkeypatch.delenv(env, raising=False)
    tj._JOBS.clear()
    tid, files = _mk_conv("DR46")
    names = {"reader": "kept/reader_read-pages-1-13_abc123.md",
             "stopped": "kept/helper_read-pages-14-20_def456.md",
             "memo": "memo.docx"}

    def write():
        for rel in names.values():
            path = os.path.join(files, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("made up")
    transcript = []
    _run_turn(monkeypatch, tid, files, transcript, "review the set",
              during=write)
    cards = sorted(os.path.relpath(p, files).replace(os.sep, "/")
                   for p in transcript[-1]["artifacts"])
    assert cards == sorted([names["stopped"], names["memo"]])
    assert core.is_reader_record(os.path.join(files, "kept", "reader_x.md"),
                                 files)
    assert not core.is_reader_record(
        os.path.join(files, "kept", "helper_x.md"), files)
    assert not core.is_reader_record(os.path.join(files, "reader_x.md"),
                                     files)
    note = core.working_files_note(files, [])
    assert "what a reading helper read, kept whole" in note
    assert "what a helper that stopped had gathered, kept" in note


def test_the_reader_contract_names_no_task_words():
    """One contract for any document: no door, fixture or task words."""
    from aec_document_review import review_prompt as P
    from aec_document_review.review_agent import READER_NUDGE
    from aec_document_review.review_build import REVIEW_READER_DESCRIPTION
    head = P.DOCUMENT_REVIEW_READER_PROMPT.split("## ", 1)[0]
    for text in (head, REVIEW_READER_DESCRIPTION, READER_NUDGE):
        assert not re.search(r"\b(?:doors?|fixtures?|windows?|coordination)\b",
                             text, re.I), text
    assert "You never invent what a page" in " ".join(head.split())
