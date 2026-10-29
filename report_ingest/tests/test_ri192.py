"""A list sent as JSON text is read; a refused answer is kept (BACKLOG
RI-192).

Build f's model-first calc calls on Foundry sent a list as JSON text
(``"inputs": "[{...}]"``), some with one closing brace too many after each
item, and the engine refused every one ("Input should be a valid list"),
leaving 28 pages unread with nothing in the item's file to say what came
back. No model is called; every name, value and page here is made up.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")
pymupdf = pytest.importorskip("pymupdf")

from report_ingest import model_first as mf                    # noqa: E402
from report_ingest import model_first_read as mfr              # noqa: E402
from report_ingest.engine import ChatModelEngine, _json_text   # noqa: E402
from report_ingest.model_first_pages import PageGroup          # noqa: E402
from report_ingest.tests.chat_fixtures import (                 # noqa: E402
    ScriptedChatModel, answer_call,
)

REPORT = "R00"


def _item(name, value, page):
    return {"name": name, "value": value, "unit": "ft",
            "prov": {"page": page}}


ITEMS = [_item("Made-up load", 50.0, 2), _item("Made-up depth", 4.0, 2)]


def _answer(inputs, results=()):
    return {"kind": "settlement", "program": "MadeUpCalc",
            "inputs": inputs, "results": list(results)}


def _engine(*replies):
    return ChatModelEngine(ScriptedChatModel(replies=list(replies)),
                           accepts_pdf=False)


@pytest.fixture()
def doc():
    from planlens.document import open_document

    made = pymupdf.open()
    for n in range(4):
        made.new_page().insert_text((72, 72), f"MADE-UP PAGE {n}")
    opened = open_document(made.tobytes(), name="made_up")
    made.close()
    yield opened
    opened.close()


class TestJsonText:

    def test_a_list_and_an_object_are_read(self):
        assert _json_text('[{"a": 1}]') == ([{"a": 1}], 0)
        assert _json_text(' {"a": [1, 2]} ') == ({"a": [1, 2]}, 0)

    def test_a_stray_closing_brace_after_each_item_is_dropped(self):
        text = '[{"a": {"b": 1}}},{"a": {"b": 2}}}]'
        assert _json_text(text) == ([{"a": {"b": 1}}, {"a": {"b": 2}}], 2)

    @pytest.mark.parametrize("text", [
        "", "made up", '"[1]"', "12", '\n<parameter name="0">{"a": 1}',
        '[{"a": 1} {"a": 2}]'])
    def test_anything_else_is_not_read(self, text):
        assert _json_text(text) == (None, 0)


class TestTheEngine:

    def test_a_list_sent_as_json_text_is_taken_and_said(self):
        engine = _engine(answer_call(_answer(json.dumps(ITEMS))))
        reply = engine.complete([{"role": "user", "content": "go"}],
                                output_format=mf.answer_schema("calc"))
        assert reply.parsed is not None
        assert [i.name for i in reply.parsed.inputs] == [
            "Made-up load", "Made-up depth"]
        assert any("read as the JSON they hold: inputs" in n
                   for n in reply.notes)
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["answer_args"] and reply.raw_answers == row["answer_args"]

    def test_one_with_stray_braces_is_taken_and_says_how_many(self):
        text = json.dumps(ITEMS).replace("}}", "}}}")
        engine = _engine(answer_call(_answer(text)))
        reply = engine.complete([{"role": "user", "content": "go"}],
                                output_format=mf.answer_schema("calc"))
        assert reply.parsed is not None and len(reply.parsed.inputs) == 2
        assert any("2 stray closing brace(s) dropped" in n
                   for n in reply.notes)

    def test_text_that_holds_no_list_is_still_refused_and_kept(self):
        args = _answer('\n<parameter name="0">' + json.dumps(ITEMS[0]))
        engine = _engine(answer_call(args))
        reply = engine.complete([{"role": "user", "content": "go"}],
                                output_format=mf.answer_schema("calc"))
        assert reply.parsed is None
        assert any("valid list" in n for n in reply.notes)
        assert len(reply.raw_answers) == 1
        assert "parameter name" in reply.raw_answers[0]


class TestTheReader:

    def test_a_list_sent_as_json_text_is_read(self, doc):
        engine = _engine(answer_call(_answer(json.dumps(ITEMS))))
        (blob,) = mfr.read_calcs(doc, [PageGroup("calc", [2], key="calc-2")],
                                 engine, report_id=REPORT)
        assert blob["calculation"] is not None and blob["error"] == ""
        (call,) = blob["calls"]
        assert call["answered"] and "raw_answer" not in call
        assert any("JSON text" in n for n in call["notes"])

    def test_a_refused_answer_keeps_why_and_what_came(self, doc):
        args = _answer('\n<parameter name="0">' + json.dumps(ITEMS[0]))
        engine = _engine(answer_call(args))
        (blob,) = mfr.read_calcs(doc, [PageGroup("calc", [2], key="calc-2")],
                                 engine, report_id=REPORT)
        assert blob["calculation"] is None
        (call,) = blob["calls"]
        assert not call["answered"]
        assert any("valid list" in n for n in call["notes"])
        assert "parameter name" in call["raw_answer"][0]
        (skipped,) = [r for r in blob["unresolved"]
                      if r.get("refused_by") == "budget"]
        assert "no structured answer" in skipped["why"]

    def test_an_answer_in_prose_keeps_its_words(self, doc):
        from langchain_core.messages import AIMessage

        said = AIMessage(content="Made-up words instead of an answer.",
                         response_metadata={"finish_reason": "stop"})
        (blob,) = mfr.read_calcs(doc, [PageGroup("calc", [2], key="calc-2")],
                                 _engine(said), report_id=REPORT)
        (call,) = blob["calls"]
        assert call["text"] == "Made-up words instead of an answer."


def test_the_scorer_s_run_file_keeps_each_refusal_s_reason():
    from report_ingest.model_first_scoring import _arm

    blob = {"calls": [], "placement": [], "unresolved": [
        {"what": "gradation series", "page": 3, "why": "made up " * 100,
         "refused_by": "python", "value": [1, 2]}]}
    (row,) = _arm([blob], "pictures")["unresolved"]
    assert row == {"what": "gradation series", "page": 3,
                   "refused_by": "python",
                   "why": ("made up " * 100)[:500]}
