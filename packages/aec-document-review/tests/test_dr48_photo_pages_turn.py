"""A turn over two attached documents, one with photograph pages nothing
looked at (BACKLOG DR-48).

``core.stream_turn`` keeps, on the turn's ``cited_pages`` event, each
attached document's photograph pages that no tool looked at; with
``GEOTECH_PHOTO_PAGES=retry`` the model gets one note naming that document
and those pages before the answer goes out, and the reply after it is the
answer. With the switch off the model is handed exactly what it was before.
Driven with a scripted stream; the documents are made-up PDFs.
"""

from __future__ import annotations

import itertools
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

fitz = pytest.importorskip("fitz")
pytest.importorskip("planlens.document.roles")

from aec_agent_core import cited_pages as cp  # noqa: E402
from aec_agent_core import review_flags  # noqa: E402
from aec_agent_core.activity_log import ActivityLogger  # noqa: E402
from aec_document_review import core  # noqa: E402

FIRST = "Both documents describe a made-up site on a gentle slope."
SECOND = "Both documents, photographs included, describe a gentle slope."
BAD = "The clay is described on PDF pages 40-41."
PROSE = ("The made-up site lies on a gentle slope above a small creek, and "
         "the made-up borings met stiff clay over weathered rock. ") * 14
_IDS = itertools.count(1)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (review_flags.ALL_ENVS + review_flags.LATER_ENVS
                + review_flags.SETTINGS_ENVS):
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    yield
    cp.clear_cache()


def _make_pdf(path, kinds) -> None:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 240, 160), False)
    pix.clear_with(150)
    png = pix.tobytes("png")
    doc = fitz.open()
    n = 0
    for kind in kinds:
        page = doc.new_page(width=612, height=792)
        if kind == "photo":
            n += 1
            page.insert_text((72, 90), f"PHOTOGRAPH {n}: VIEW LOOKING NORTH",
                             fontsize=20)
            page.insert_image(fitz.Rect(72, 130, 540, 560), stream=png)
            page.insert_text((72, 600), f"Photo {n}. Made-up test pit area.",
                             fontsize=10)
        else:
            page.insert_text((72, 90), "SITE DESCRIPTION", fontsize=16)
            page.insert_textbox(fitz.Rect(72, 120, 540, 720), PROSE,
                                fontsize=10)
    doc.save(str(path))
    doc.close()


def _calls(conv, calls) -> None:
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for name, args, result in calls:
            rid = f"r{next(_IDS)}"
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "name": name, "run_id": rid,
                                 "args": args}) + "\n")
            fh.write(json.dumps({
                "turn": 1, "event": "tool_end", "name": name, "run_id": rid,
                "result": (result if isinstance(result, str)
                           else json.dumps(result))}) + "\n")


def _look(name, page):
    return ("analyze_pdf_page", {"attachment_key": name, "page": page},
            "A made-up photograph of a test pit.")


def _conversation(tmp_path, b_photos_looked_at=False):
    """Two attached documents. An earlier turn looked at a.pdf's photograph
    page and read b.pdf as text only (unless ``b_photos_looked_at``)."""
    conv = tmp_path / "conv"
    files = conv / "files"
    files.mkdir(parents=True)
    _make_pdf(files / "a.pdf", ["text", "photo"])
    _make_pdf(files / "b.pdf", ["text", "photo", "photo"])
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    (conv / "attachments.json").write_text('["a.pdf", "b.pdf"]',
                                           encoding="utf-8")
    (conv / "activity.jsonl").write_text("", encoding="utf-8")
    calls = [
        ("open_document", {"source": "a.pdf"},
         {"handle": "doc_aaa", "name": "a.pdf", "n_pages": 2}),
        _look("a.pdf", 1),
        ("open_document", {"source": "b.pdf"},
         {"handle": "doc_bbb", "name": "b.pdf", "n_pages": 3}),
        ("read_document", {"handle": "doc_bbb"},
         {"handle": "doc_bbb", "pages_returned": "0-2"})]
    if b_photos_looked_at:
        calls += [_look("b.pdf", 1), _look("b.pdf", 2)]
    _calls(conv, calls)
    return conv


def _reply(text):
    return [("messages", (AIMessageChunk(content=text),
                          {"langgraph_node": "model"})),
            ("updates", {"model": {"messages": [AIMessage(content=text)]}})]


class _ScriptedAgent:
    """Replays one scripted pass per graph invocation."""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.inputs = []

    def stream(self, inp, config=None, stream_mode=None):
        self.inputs.append(inp)
        for item in (self.passes.pop(0) if self.passes else []):
            yield item


def _run(agent, conv):
    logger = ActivityLogger(str(conv), turn=2)
    items = list(core.stream_turn(
        agent, [{"role": "user", "content": "summarise both documents"}],
        "thread-1", callbacks=[logger]))
    done = [i for i in items if i["kind"] == "turn_done"][-1]
    events = [json.loads(line) for line in
              (conv / "activity.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    return done["answer"], items, [e for e in events
                                   if e.get("event") == cp.RECORD_EVENT]


def test_switch_on_one_note_names_the_document_and_its_unseen_photo_pages(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(FIRST), _reply(SECOND))
    answer, items, events = _run(agent, conv)
    assert answer == SECOND                     # the reply after the note
    assert len(agent.inputs) == 2               # one note, once
    told = agent.inputs[1]["messages"][-1]
    assert told["role"] == "user"
    assert told["content"].startswith(cp.PHOTO_NOTE_PREFIX)
    assert "- b.pdf: PDF pages 2-3" in told["content"]
    assert "a.pdf" not in told["content"]
    assert {"kind": "tool_call", "text": cp.PHOTO_RETRY_STATUS} in items
    assert [e["action"] for e in events] == ["photos_fed_back", "none"]
    assert events[0][cp.PHOTO_KEY]["told"] == {"b.pdf": "2-3"}
    assert events[-1][cp.PHOTO_KEY]["unseen"] == {"b.pdf": "2-3"}


def test_switch_on_a_turn_that_looked_at_them_gets_no_note(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    conv = _conversation(tmp_path, b_photos_looked_at=True)
    agent = _ScriptedAgent(_reply(FIRST), _reply(SECOND))
    answer, _items, events = _run(agent, conv)
    assert answer == FIRST and len(agent.inputs) == 1
    assert [e["action"] for e in events] == ["none"]
    assert events[0][cp.PHOTO_KEY]["unseen"] == {}
    assert events[0][cp.PHOTO_KEY]["photos"] == {"a.pdf": "2",
                                                 "b.pdf": "2-3"}


def _without_the_photo_check(monkeypatch):
    """What the model was handed before DR-48: no photograph-page check."""
    monkeypatch.setattr(cp, "photo_coverage", lambda *a, **k: None)


def test_switch_off_the_model_is_handed_exactly_what_it_was(
        tmp_path, monkeypatch):
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(FIRST), _reply(SECOND))
    answer, items, events = _run(agent, conv)
    assert answer == FIRST and len(agent.inputs) == 1
    # ...and the gap is on the record all the same, at no model call
    assert [e["action"] for e in events] == ["none"]
    assert events[0]["photo_mode"] == "off"
    assert events[0][cp.PHOTO_KEY]["unseen"] == {"b.pdf": "2-3"}

    with monkeypatch.context() as m:
        _without_the_photo_check(m)
        cp.clear_cache()
        before = _ScriptedAgent(_reply(FIRST), _reply(SECOND))
        answer_before, items_before, _ = _run(before, _conversation(
            tmp_path / "before"))
    assert before.inputs == agent.inputs
    assert answer_before == answer
    assert [i for i in items_before if i["kind"] != "turn_done"] == \
        [i for i in items if i["kind"] != "turn_done"]


def test_a_suite_run_keeps_the_upload_list_and_the_record(tmp_path,
                                                          monkeypatch):
    """The review suite stages uploads as the page does and keeps the
    upload list beside the run's record, so a document the run never
    opened is in the photograph-page check too."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.outputs import ChatGeneration, ChatResult
    from aec_agent_core.review_eval.runner import run_task
    from aec_document_review.review_eval.tasks import OPEN_TASKS, Task

    class _Answers(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=FIRST))])

    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    base = next(t for t in OPEN_TASKS if t.id == "fixture-markups")
    task = Task(id="dr48-suite", question="Summarise the document.",
                documents=list(base.documents), category="extract",
                doc_type="report", auto_checks=False, checks=[])
    run = tmp_path / "run"
    res = run_task(task, _Answers(responses=[AIMessage(content="x")]),
                   arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(run))
    assert res["error"] is None, res.get("traceback")
    keys = json.loads((run / "attachments.json").read_text(encoding="utf-8"))
    assert len(keys) == len(task.documents) and all(keys)
    events = [json.loads(line) for line in
              (run / "activity.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    photos = [e[cp.PHOTO_KEY] for e in events
              if e.get("event") == cp.RECORD_EVENT and cp.PHOTO_KEY in e]
    # measured at turn end, though the run never opened the document
    assert photos and photos[-1]["attached"] == "attachment list"
    assert photos[-1]["documents"] == keys and "left_out" not in photos[-1]


def test_switch_off_the_page_checks_retry_note_is_unchanged(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(BAD), _reply(FIRST))
    _answer, items, _events = _run(agent, conv)
    with monkeypatch.context() as m:
        _without_the_photo_check(m)
        cp.clear_cache()
        before = _ScriptedAgent(_reply(BAD), _reply(FIRST))
        _a, items_before, _e = _run(before, _conversation(
            tmp_path / "before"))
    assert len(agent.inputs) == 2
    assert agent.inputs[1] == before.inputs[1]
    assert cp.PHOTO_NOTE_PREFIX not in agent.inputs[1]["messages"][-1][
        "content"]
    assert {"kind": "tool_call", "text": cp.RETRY_STATUS} in items
    assert [i for i in items_before if i["kind"] == "tool_call"] == \
        [i for i in items if i["kind"] == "tool_call"]
