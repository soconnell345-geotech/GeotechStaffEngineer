"""The model's own parameter markup is taken out of an answer; a refused
answer is asked for once more (BACKLOG RI-195).

On Claude through Foundry a model sometimes writes its own tool-call markup
inside a value (``"pit": "null</pit>\\n<parameter name=\\"remarks\\">..."``,
``"inputs": "<parameter name=\\"0\\">{...}"`` with the list's later items as
numbered keys), and the provider files what follows under the field it was
writing. MF-3 lost a log, a calculation window and a page's description to
it, and four descriptions were taken with a field swallowed into the one
before it. No model is called; every name, value and page here is made up.
"""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("langchain_core")
pymupdf = pytest.importorskip("pymupdf")

from report_ingest import model_first as mf                    # noqa: E402
from report_ingest import model_first_read as mfr              # noqa: E402
from report_ingest.engine import (                              # noqa: E402
    ASK_AGAIN, ChatModelEngine, _mend_markup, _validate, ask_again, refusal,
    text_block, user,
)
from report_ingest.log_reader import LogReading, read_log     # noqa: E402
from report_ingest.model_first_pages import PageGroup          # noqa: E402
from report_ingest.range_labels import PageDescription         # noqa: E402
from report_ingest.tests.chat_fixtures import (                 # noqa: E402
    ScriptedChatModel, answer_call,
)
from report_ingest.tests.fake_engine import FakeEngine          # noqa: E402

REPORT = "R00"
CALC = mf.answer_schema("calc")
HEAD = ASK_AGAIN.split("{why}", 1)[0]


def _item(name, value, page=2):
    return {"name": name, "value": value, "unit": "ft",
            "prov": {"page": page}}


INPUTS = [_item("Made-up load", 50.0), _item("Made-up depth", 4.0),
          _item("Made-up width", 6.0)]
RESULTS = [_item("Made-up settlement", 0.5), _item("Made-up factor", 2.5)]


def _spilled(key, items):
    """A list as the markup spills it: item 0 in the field, the rest as the
    answer's numbered keys."""
    out = {key: '<parameter name="0">' + json.dumps(items[0])}
    out.update({str(n): item for n, item in enumerate(items) if n})
    return out


def _calc(**fields):
    return {"kind": "settlement", "program": "MadeUpCalc", **fields}


def _description(**fields):
    base = {"title": "MADE-UP TITLE", "kind": "a made-up page",
            "summary": "A made-up page."}
    base.update(fields)
    return base


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


# ---------------------------------------------------------------------------
# the repair
# ---------------------------------------------------------------------------

class TestAFieldSwallowed:

    def test_a_null_and_the_field_after_it_are_put_back(self):
        args = {"investigation_id": "X-1",
                "pit": 'null</pit>\n<parameter name="remarks">Made-up '
                       'remark.'}
        answer, _d, why, _w, _j, markup = _validate(LogReading, args)
        assert why == "" and answer.pit is None
        assert answer.remarks == "Made-up remark."
        (note,) = markup
        assert "put back" in note
        assert "pit split at the markup into pit and remarks" in note

    def test_a_required_field_swallowed_is_found(self):
        args = _description(
            issuer_date='Made-up Co., 2001</issuer_date>\n<parameter '
                        'name="kind">a made-up cover')
        del args["kind"]
        answer, _d, why, _w, _j, markup = _validate(PageDescription, args)
        assert why == ""
        assert (answer.issuer_date, answer.kind) == ("Made-up Co., 2001",
                                                     "a made-up cover")
        assert markup and "put back" in markup[0]

    def test_an_answer_taken_with_a_field_swallowed_is_put_back(self):
        """The four descriptions MF-3 took with the markup in them: the
        describer fills a missing field with "", so they validated."""
        args = _description(
            exploration='X-7</exploration>\n<parameter name="continues">'
                        'Sheet 1 of 2', continues="")
        answer, _d, _why, _w, _j, markup = _validate(PageDescription, args)
        assert (answer.exploration, answer.continues) == ("X-7",
                                                          "Sheet 1 of 2")
        assert "exploration split at the markup" in markup[0]

    def test_a_field_held_twice_is_not_put_back(self):
        args = _description(
            exploration='X-7</exploration>\n<parameter name="continues">'
                        'Sheet 1 of 2', continues="Sheet 9 of 9")
        answer, _d, _why, _w, _j, markup = _validate(PageDescription, args)
        # Taken as it came, and said: which "continues" is meant cannot be
        # told.
        assert answer.continues == "Sheet 9 of 9"
        assert "<parameter" in answer.exploration
        (note,) = markup
        assert "not put back" in note and "continues came both" in note
        assert "taken as it came" in note

    def test_its_own_closing_tag_alone_is_taken_off(self):
        answer, *_rest, markup = _validate(
            LogReading, {"investigation_id": "X-1</investigation_id>"})
        assert answer.investigation_id == "X-1"
        assert "its closing tag taken off" in markup[0]

    def test_words_that_end_in_another_tag_are_left_alone(self):
        args = {"investigation_id": "X-1", "remarks": "made up </b>"}
        assert _mend_markup(args) == (args, [], [])
        *_rest, markup = _validate(LogReading, args)
        assert markup == []

    def test_the_answer_as_it_came_is_unchanged(self):
        args = {"investigation_id": "X-1",
                "pit": 'null</pit>\n<parameter name="remarks">Made up.'}
        kept = copy.deepcopy(args)
        _validate(LogReading, args)
        assert args == kept

    def test_markup_inside_a_list_of_words_is_named(self):
        args = _description(identifiers=[
            "X-1", 'X-2</identifiers>\n<parameter name="exploration">X-2'])
        answer, *_rest, markup = _validate(PageDescription, args)
        assert answer is not None
        assert markup == ["the model's own parameter markup is still in "
                          "identifiers.1"]


class TestAListSpilled:

    def test_one_list_in_the_numbered_keys_is_put_back_in_order(self):
        args = _calc(**_spilled("inputs", INPUTS))
        answer, _d, why, _w, _j, markup = _validate(CALC, args)
        assert why == ""
        assert [i.name for i in answer.inputs] == [
            "Made-up load", "Made-up depth", "Made-up width"]
        assert "inputs read as a list of 3 item(s), 2 of them from the " \
               "keys 1 to 2" in markup[0]

    def test_a_list_spilled_inside_its_own_value_is_read(self):
        text = "".join(f'<parameter name="{n}">{json.dumps(item)}'
                       f"</parameter>\n" for n, item in enumerate(RESULTS))
        answer, *_rest, markup = _validate(CALC, _calc(results=text))
        assert [r.name for r in answer.results] == [
            "Made-up settlement", "Made-up factor"]
        assert "results read as a list of 2 item(s)" in markup[0]

    def test_two_lists_that_share_the_numbered_keys_are_refused(self):
        """Each number held one value, so the later list's items may stand
        in the earlier one's place: nothing is put back."""
        args = _calc(**_spilled("inputs", INPUTS[:2]),
                     **_spilled("results", RESULTS[:1]),
                     **{"2": RESULTS[1]})
        mended, fixes, faults = _mend_markup(args)
        assert faults and "share" in faults[0]
        answer, _d, why, _w, _j, markup = _validate(CALC, args)
        assert answer is None and "valid list" in why
        assert "not put back" in markup[0]

    def test_items_with_a_gap_in_their_numbers_are_refused(self):
        args = _calc(inputs='<parameter name="0">' + json.dumps(INPUTS[0]),
                     **{"2": INPUTS[2]})
        _mended, _fixes, faults = _mend_markup(args)
        assert faults == ["inputs: its items are numbered 0, 2, with a gap"]

    def test_numbered_keys_before_the_list_are_refused(self):
        args = {"kind": "settlement", "1": INPUTS[1],
                "inputs": '<parameter name="0">' + json.dumps(INPUTS[0])}
        _mended, _fixes, faults = _mend_markup(args)
        assert faults == ["the numbered keys 1 came before any list's markup"]


# ---------------------------------------------------------------------------
# the engine and the re-ask
# ---------------------------------------------------------------------------

class TestTheEngine:

    def test_a_repaired_answer_is_taken_and_keeps_what_came(self):
        engine = _engine(answer_call(_calc(**_spilled("inputs", INPUTS))))
        reply = engine.complete([user(text_block("go"))],
                                output_format=CALC)
        assert reply.parsed is not None and len(reply.parsed.inputs) == 3
        assert any("put back" in n for n in reply.notes)
        assert '"1"' in reply.raw_answers[0]
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["answer_args"] == reply.raw_answers

    def test_ask_again_says_why_and_the_call_is_counted(self):
        refused = _calc(inputs="made-up words, not a list")
        engine = _engine(answer_call(refused),
                         answer_call(_calc(inputs=INPUTS), "call_again"))
        messages = [user(text_block("go"))]
        first = engine.complete(messages, output_format=CALC)
        assert first.parsed is None
        assert "valid list" in refusal(first)
        again = ask_again(messages, first)
        assert len(messages) == 1                     # a new list
        assert [m["role"] for m in again] == ["user", "assistant", "user"]
        said = again[-1]["content"][0]["text"]
        assert said.startswith(HEAD + ": it does not match the schema")
        second = engine.complete(again, output_format=CALC)
        assert second.parsed is not None
        cost = engine.meter.to_dict()
        assert cost["asked_again"] == 1
        assert [r.get("asked_again", False) for r in cost["per_call"]] == [
            False, True]

    def test_a_meter_with_no_re_ask_says_nothing_of_it(self):
        engine = _engine(answer_call(_calc(inputs=INPUTS)))
        engine.complete([user(text_block("go"))], output_format=CALC)
        assert "asked_again" not in engine.meter.to_dict()


# ---------------------------------------------------------------------------
# the three readers that ask once more
# ---------------------------------------------------------------------------

class TestModelFirst:

    def test_a_refused_window_is_asked_again_and_read(self, doc):
        spoiled = _calc(**_spilled("inputs", INPUTS[:2]),
                        **_spilled("results", RESULTS[:1]),
                        **{"2": RESULTS[1]})
        model = ScriptedChatModel(replies=[
            answer_call(spoiled),
            answer_call(_calc(inputs=INPUTS, results=RESULTS), "call_again")])
        engine = ChatModelEngine(model, accepts_pdf=False)
        (blob,) = mfr.read_calcs(doc, [PageGroup("calc", [2], key="calc-2")],
                                 engine, report_id=REPORT)
        assert blob["calculation"] is not None and blob["error"] == ""
        first, again = blob["calls"]
        assert not first["answered"] and again["answered"]
        assert again["asked_again"] and "asked_again" not in first
        assert any("not put back" in n for n in first["notes"])
        assert blob["cost"]["calls"] == 2
        last = model.requests[1]["messages"][-1]
        assert HEAD in str(last.content)


class TestTodaysLogReader:

    @pytest.fixture()
    def page(self, tmp_path):
        from planlens.document import open_document
        from planlens.testing.loggrid_fixtures import build_imperial_log

        path = tmp_path / "log.pdf"
        path.write_bytes(build_imperial_log().pdf)
        opened = open_document(str(path))
        yield opened
        opened.close()

    def test_the_first_call_is_asked_again(self, page):
        engine = FakeEngine([
            {"text": "Made-up words instead of an answer."},
            {"final": LogReading(investigation_id="X-1", depth_unit="ft")}])
        result = read_log(page, [0], engine)
        assert engine.n_calls == 2
        last = engine.calls[1]["messages"][-1]["content"][0]["text"]
        assert last.startswith(HEAD)
        assert any("the reader was asked for it again" in w
                   for w in result.warnings)

    def test_a_budget_of_one_call_asks_nothing_more(self, page):
        engine = FakeEngine([{"text": "Made-up words instead of an answer."}])
        with pytest.raises(RuntimeError, match="no structured reading"):
            read_log(page, [0], engine, budget=1)
        assert engine.n_calls == 1


class TestTheQuestionList:

    @pytest.fixture()
    def qlist(self, tmp_path):
        from report_ingest import question_list as ql
        from report_ingest.tests.test_question_list import _workbook

        return ql.load_questions(_workbook(tmp_path / "questions.xlsx"))

    def test_a_refused_call_is_asked_again_and_each_row_kept(self, qlist):
        from report_ingest import question_list as ql
        from report_ingest.tests.test_question_list import (
            PAGES, _answer, _main, _record, _screening)

        spoiled = _main(flood_risk=json.dumps(_answer("low"))
                        + '</flood_risk>\n<parameter name="issue_month">'
                        + json.dumps(_answer("2001-01")))
        engine = _engine(answer_call(_screening(), "call_screening"),
                         answer_call(spoiled, "call_main"),
                         answer_call(_main(), "call_again"))
        run = ql.run_questions(_record(), PAGES, qlist, engine)
        screening, refused, again = run.calls
        assert screening["answered"] and "raw_answer" not in screening
        assert not refused["answered"] and refused["raw_answer"]
        assert any("not put back" in n for n in refused["notes"])
        assert any(n.startswith("an answer was not taken")
                   for n in refused["notes"])
        assert again["answered"] and again["asked_again"]
        assert run.answers["flood_risk"].value == "low"
        assert engine.meter.to_dict()["asked_again"] == 1

    def test_twice_refused_says_so(self, qlist):
        from report_ingest import question_list as ql
        from report_ingest.tests.test_question_list import (
            PAGES, _record, _screening)

        engine = FakeEngine([{"final": _screening()},
                             {"text": "Made-up words."},
                             {"text": "Made-up words."}])
        run = ql.run_questions(_record(), PAGES, qlist, engine)
        assert [c.get("asked_again", False) for c in run.calls] == [
            False, False, True]
        assert run.calls[1]["text"] == "Made-up words."
        assert any("main call gave no structured answer, and none when "
                   "asked again" in w for w in run.warnings)
