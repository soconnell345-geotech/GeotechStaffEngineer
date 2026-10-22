"""The ingest on the app's own chat model (BACKLOG RI-03, RI-19), offline.

:class:`~report_ingest.engine.ChatModelEngine` is how a report is read on
Foundry, where the page's model is a ``palantir_models`` class with no
``response_format`` and no ``tool_choice``. A scripted LangChain chat model
stands in for it (:mod:`report_ingest.tests.chat_fixtures`), as strict as a
provider about a conversation's shape. What is checked: a reader call with
tools, a structured answer through the answer tool, pictures, the counting
the ingest's ceiling relies on (one call, one request), a reply cut at its
output cap, and :func:`~report_ingest.engine.engine_for` finding the chat
model inside the page's wrapper. Public fixtures only.
"""

from __future__ import annotations

import base64
import json
from typing import List, Optional

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from pydantic import BaseModel, Field  # noqa: E402

from report_ingest.engine import (  # noqa: E402
    ANSWER_TOOL, DEFAULT_MAX_TOKENS, ChatModelEngine, PrompterEngine, Reply,
    engine_for, image_block, strict_schema, text_block, tool_result_block,
    user,
)
from report_ingest.subagent import CallCeiling, CallCeilingReached  # noqa: E402
from report_ingest.tests.chat_fixtures import (  # noqa: E402
    ScriptedChatModel, answer_call, check_conversation, from_turns,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 24

ZOOM = {"name": "zoom_plot", "description": "Magnify a box of the page.",
        "input_schema": {"type": "object",
                         "properties": {"page": {"type": "integer"}},
                         "required": ["page"]}}


class Layer(BaseModel):
    top_m: float
    soil: str


class Reading(BaseModel):
    """What a reader returns."""

    title: str
    layers: List[Layer] = Field(default_factory=list)
    note: str = "none"
    depth_m: Optional[float] = None


def _engine(*replies, **kw):
    model = ScriptedChatModel(replies=list(replies), **kw)
    return ChatModelEngine(model), model


def _said(text, **meta):
    return AIMessage(content=text,
                     response_metadata={"finish_reason": "stop", **meta},
                     usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                     "total_tokens": 15})


# ---------------------------------------------------------------------------
# a reader call with tools
# ---------------------------------------------------------------------------

class TestAReaderCallWithTools:

    def test_the_tools_are_bound_and_the_call_comes_back(self):
        zoom = AIMessage(content="Looking closer.", tool_calls=[
            {"name": "zoom_plot", "args": {"page": 3}, "id": "c1",
             "type": "tool_call"}],
            response_metadata={"finish_reason": "tool_calls"})
        engine, model = _engine(zoom)

        reply = engine.complete([user(text_block("read the log"))],
                                system="You read logs.", tools=[ZOOM])

        assert model.requests[0]["tools"] == ["zoom_plot"]
        spec = model.requests[0]["specs"][0]["function"]
        assert spec["parameters"] == ZOOM["input_schema"]
        assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == \
            [("c1", "zoom_plot", {"page": 3})]
        assert reply.text == "Looking closer."
        sent = model.requests[0]["messages"]
        assert isinstance(sent[0], SystemMessage)
        assert sent[0].content == "You read logs."
        assert isinstance(sent[1], HumanMessage)

    def test_the_turn_goes_back_as_it_came_and_the_result_follows_it(self):
        zoom = AIMessage(content="", tool_calls=[
            {"name": "zoom_plot", "args": {"page": 3}, "id": "c1",
             "type": "tool_call"}],
            additional_kwargs={"claude_thinking": [{"type": "thinking",
                                                    "thinking": "hmm",
                                                    "signature": "sig"}]})
        engine, model = _engine(zoom, _said("Done."))
        messages = [user(text_block("read the log"))]
        first = engine.complete(messages, tools=[ZOOM])
        messages.append({"role": "assistant", "content": first.content})
        messages.append({"role": "user", "content": [
            tool_result_block("c1", "the box, magnified")]})

        engine.complete(messages, tools=[ZOOM])

        sent = model.requests[1]["messages"]
        # The provider's own message, thinking and all, not a rebuilt one.
        assert sent[1] is first.content[-1]["data"]
        assert sent[1].additional_kwargs["claude_thinking"][0]["signature"] \
            == "sig"
        assert isinstance(sent[2], ToolMessage)
        assert sent[2].tool_call_id == "c1"
        assert sent[2].content == "the box, magnified"

    def test_a_call_with_no_id_is_given_one_its_result_can_name(self):
        zoom = AIMessage(content="", tool_calls=[
            {"name": "zoom_plot", "args": {"page": 1}, "id": None,
             "type": "tool_call"}])
        engine, model = _engine(zoom, _said("ok"))
        first = engine.complete([user(text_block("go"))], tools=[ZOOM])
        call_id = first.tool_calls[0].id
        assert call_id

        engine.complete([user(text_block("go")),
                         {"role": "assistant", "content": first.content},
                         {"role": "user",
                          "content": [tool_result_block(call_id, "x")]}],
                        tools=[ZOOM])
        assert model.requests[1]["messages"][2].tool_call_id == call_id

    def test_an_error_result_is_marked_as_one(self):
        engine, model = _engine(_said("ok"))
        engine.complete([
            user(text_block("go")),
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c9", "name": "zoom_plot",
                 "input": {"page": 2}}]},
            {"role": "user", "content": [
                tool_result_block("c9", "no such page", is_error=True)]}],
            tools=[ZOOM])
        result = model.requests[0]["messages"][2]
        assert isinstance(result, ToolMessage) and result.status == "error"
        assert model.requests[0]["messages"][1].tool_calls[0]["args"] == \
            {"page": 2}


# ---------------------------------------------------------------------------
# a structured answer through the answer tool
# ---------------------------------------------------------------------------

ANSWER = {"title": "B-1", "layers": [{"top_m": 0.0, "soil": "clay"}],
          "note": "two pages", "depth_m": 12.5}


class TestAStructuredAnswer:

    def test_it_comes_back_parsed_from_the_answer_call(self):
        engine, model = _engine(answer_call(ANSWER))

        reply = engine.complete([user(text_block("read it"))],
                                system="You read logs.",
                                output_format=Reading)

        assert isinstance(reply.parsed, Reading)
        assert reply.parsed.layers[0].soil == "clay"
        assert reply.tool_calls == []            # the answer is not a tool
        assert model.requests[0]["tools"] == [ANSWER_TOOL]
        spec = model.requests[0]["specs"][0]["function"]
        assert spec["parameters"] == strict_schema(Reading)
        system = model.requests[0]["messages"][0].content
        assert system.startswith("You read logs.")
        assert "`answer` tool" in system

    def test_beside_the_reader_s_tools_it_is_one_more_tool(self):
        engine, model = _engine(answer_call(ANSWER))
        engine.complete([user(text_block("read it"))], tools=[ZOOM],
                        output_format=Reading)

        assert model.requests[0]["tools"] == ["zoom_plot", ANSWER_TOOL]
        assert "other tools" in model.requests[0]["messages"][0].content

    def test_an_answer_and_a_look_together_give_both(self):
        both = AIMessage(content="", tool_calls=[
            {"name": "zoom_plot", "args": {"page": 1}, "id": "z",
             "type": "tool_call"},
            {"name": ANSWER_TOOL, "args": ANSWER, "id": "a",
             "type": "tool_call"}])
        engine, _ = _engine(both)
        reply = engine.complete([user(text_block("go"))], tools=[ZOOM],
                                output_format=Reading)

        assert reply.parsed.title == "B-1"
        assert [c.name for c in reply.tool_calls] == ["zoom_plot"]

    def test_an_answer_written_out_as_json_is_parsed_too(self):
        fenced = "Here it is:\n```json\n" + json.dumps(ANSWER) + "\n```"
        engine, _ = _engine(_said(fenced))

        reply = engine.complete([user(text_block("go"))],
                                output_format=Reading)
        assert reply.parsed.depth_m == 12.5

    def test_a_null_for_a_field_with_a_default_takes_the_default(self):
        # Strict mode's schema makes every field required, so a model sends
        # null where it has nothing.
        engine, _ = _engine(answer_call(dict(ANSWER, note=None,
                                             layers=None)))
        reply = engine.complete([user(text_block("go"))],
                                output_format=Reading)
        assert reply.parsed.note == "none"
        assert reply.parsed.layers == []

    def test_neither_a_call_nor_json_is_no_answer_not_a_guess(self):
        engine, _ = _engine(_said("I could not read the page."))
        reply = engine.complete([user(text_block("go"))],
                                output_format=Reading)
        assert reply.parsed is None
        assert reply.text == "I could not read the page."

    def test_asking_again_after_an_answer_answers_the_answer_call(self):
        # The log reader's follow-up: its first answer goes back as the
        # assistant turn, and a provider refuses a call with no result.
        engine, model = _engine(answer_call(ANSWER, call_id="a1"),
                                answer_call(ANSWER, call_id="a2"))
        messages = [user(text_block("read it"))]
        first = engine.complete(messages, output_format=Reading)
        messages += [{"role": "assistant", "content": first.content},
                     user(text_block("settle the depths you were unsure of"))]

        engine.complete(messages, output_format=Reading)

        sent = model.requests[1]["messages"]
        assert isinstance(sent[3], ToolMessage)
        assert sent[3].tool_call_id == "a1"
        assert sent[3].content == "Answer received."
        assert isinstance(sent[4], HumanMessage)

    def test_a_pass_tool_may_not_take_the_answer_s_name(self):
        engine, _ = _engine()
        with pytest.raises(ValueError, match="answer"):
            engine.complete([user(text_block("go"))],
                            tools=[dict(ZOOM, name=ANSWER_TOOL)],
                            output_format=Reading)

    def test_with_no_output_format_an_answer_named_call_is_a_tool_call(self):
        call = AIMessage(content="", tool_calls=[
            {"name": ANSWER_TOOL, "args": {"x": 1}, "id": "q",
             "type": "tool_call"}])
        engine, model = _engine(call)
        reply = engine.complete([user(text_block("go"))],
                                tools=[dict(ZOOM, name=ANSWER_TOOL)])
        assert [c.name for c in reply.tool_calls] == [ANSWER_TOOL]
        assert "`answer` tool" not in str(model.requests[0]["messages"])


# ---------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------

class TestAnImageCall:

    def test_images_join_the_last_user_turn_as_data_uris(self):
        engine, model = _engine(_said("a log"))
        engine.complete([user(text_block("what is this page?"))],
                        images=[PNG, JPEG])

        content = model.requests[0]["messages"][-1].content
        assert content[0] == {"type": "text", "text": "what is this page?"}
        urls = [b["image_url"]["url"] for b in content[1:]]
        assert urls[0] == ("data:image/png;base64,"
                           + base64.b64encode(PNG).decode())
        assert urls[1].startswith("data:image/jpeg;base64,")
        assert "detail" not in content[1]["image_url"]

    def test_a_block_s_detail_rides_along(self):
        engine, model = _engine(_said("a log"))
        engine.complete([user(text_block("look"),
                              image_block(PNG, detail="high"))])
        block = model.requests[0]["messages"][-1].content[1]
        assert block["image_url"]["detail"] == "high"

    def test_a_tool_s_picture_follows_its_result(self):
        engine, model = _engine(_said("seen"))
        engine.complete([
            user(text_block("go")),
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r1", "name": "render_page",
                 "input": {"page": 4}}]},
            {"role": "user", "content": [tool_result_block(
                "r1", [text_block("page 4"), image_block(PNG)])]}],
            tools=[dict(ZOOM, name="render_page")])

        sent = model.requests[0]["messages"]
        assert isinstance(sent[2], ToolMessage)
        assert "page 4" in sent[2].content
        assert "picture follows" in sent[2].content
        picture = sent[3]
        assert isinstance(picture, HumanMessage)
        assert picture.content[0]["text"].startswith(
            "Picture for tool call r1")
        assert picture.content[1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# counting: one call, one request, the same number everywhere
# ---------------------------------------------------------------------------

class TestTheBudgetCounting:

    def test_tokens_and_calls_go_into_the_meter_and_the_reply(self):
        message = AIMessage(
            content="ok", response_metadata={"finish_reason": "stop",
                                             "model_name": "GPT_5_4"},
            usage_metadata={"input_tokens": 1200, "output_tokens": 80,
                            "total_tokens": 1280,
                            "input_token_details": {"cache_read": 200}})
        engine, _ = _engine(message)
        reply = engine.complete([user(text_block("go"))])

        assert reply.usage.input_tokens == 1000      # cached input apart
        assert reply.usage.cache_read_tokens == 200
        assert reply.usage.output_tokens == 80
        assert reply.model == engine.served_by == "GPT_5_4"
        assert engine.meter.calls == 1
        assert engine.meter.input_tokens == 1000
        assert engine.meter.by_model["scripted-model"]["calls"] == 1

    def test_usage_with_no_metadata_is_read_off_the_route_s_own(self):
        message = AIMessage(content="ok", response_metadata={
            "finish_reason": "stop",
            "token_usage": {"prompt_tokens": 40, "completion_tokens": 7}})
        engine, _ = _engine(message)
        reply = engine.complete([user(text_block("go"))])
        assert (reply.usage.input_tokens, reply.usage.output_tokens) == \
            (40, 7)

    def test_the_ingest_s_ceiling_counts_every_request(self):
        engine, model = _engine(_said("one"), _said("two"), _said("three"))
        ceiling = CallCeiling(2)
        counted = ceiling.wrap(engine)

        counted.complete([user(text_block("a"))])
        counted.complete([user(text_block("b"))])
        with pytest.raises(CallCeilingReached):
            counted.complete([user(text_block("c"))])

        assert len(model.requests) == 2 == engine.meter.calls == ceiling.used
        assert ceiling.reached

    def test_a_whole_ingest_runs_on_it_and_counts_alike(self, tmp_path):
        pytest.importorskip("planlens.document.roles")
        pytest.importorskip("pymupdf")
        from report_ingest.subagent import run_ingest
        from report_ingest.tests.narrative_fixtures import (
            build_narrative_report,
        )
        from report_ingest.tests.test_graph import full_script

        pdf = tmp_path / "SYN.pdf"
        pdf.write_bytes(build_narrative_report().pdf)
        script = full_script()
        model = ScriptedChatModel(replies=from_turns(script))
        engine = ChatModelEngine(model)

        answer = run_ingest(str(pdf), [], engine=engine,
                            out_dir=str(tmp_path / "out"), report_id="SYN",
                            max_model_calls=1000)

        assert answer.error == ""
        assert answer.n_pages == 22
        assert answer.investigations == 2
        assert answer.diggs_ok is True
        assert len(model.requests) == len(script)
        assert answer.model_calls == len(script) == engine.meter.calls
        assert answer.ceiling_reached is False

    def test_a_low_ceiling_stops_the_ingest_on_it_too(self, tmp_path):
        pytest.importorskip("planlens.document.roles")
        pytest.importorskip("pymupdf")
        from report_ingest.subagent import run_ingest
        from report_ingest.tests.narrative_fixtures import (
            build_narrative_report,
        )
        from report_ingest.tests.test_graph import full_script

        pdf = tmp_path / "SYN.pdf"
        pdf.write_bytes(build_narrative_report().pdf)
        model = ScriptedChatModel(replies=from_turns(full_script()))

        answer = run_ingest(str(pdf), [], engine=ChatModelEngine(model),
                            out_dir=str(tmp_path / "out"), report_id="SYN",
                            max_model_calls=4)

        assert answer.ceiling_reached is True
        assert len(model.requests) == 4


# ---------------------------------------------------------------------------
# the output cap (RI-19)
# ---------------------------------------------------------------------------

class TestAReplyCutAtTheCap:

    def test_a_cut_answer_is_no_answer_and_says_it_was_cut(self, caplog):
        # Foundry's classes parse a cut call's arguments as {}.
        cut = answer_call({}, finish_reason="length")
        engine, _ = _engine(cut)
        with caplog.at_level("WARNING"):
            reply = engine.complete([user(text_block("go"))],
                                    output_format=Reading)

        assert reply.parsed is None
        assert reply.stop_reason == "length"
        assert reply.cut_off
        assert "output cap" in caplog.text

    def test_chatopenai_s_unparsed_call_is_read_then_dropped(self):
        cut = AIMessage(content="", invalid_tool_calls=[{
            "name": ANSWER_TOOL, "args": '{"title": "B-1", "lay',
            "id": "a1", "error": "bad json", "type": "invalid_tool_call"}],
            response_metadata={"finish_reason": "length"})
        engine, model = _engine(cut, _said("again"))
        messages = [user(text_block("go"))]
        reply = engine.complete(messages, output_format=Reading)
        assert reply.parsed is None and reply.cut_off

        # Sent back, it carries no call nothing can answer.
        messages += [{"role": "assistant", "content": reply.content},
                     user(text_block("answer briefly"))]
        engine.complete(messages, output_format=Reading)
        resent = model.requests[1]["messages"][2]
        assert isinstance(resent, AIMessage)
        assert not resent.invalid_tool_calls

    def test_an_unparsed_answer_that_is_whole_still_counts(self):
        whole = AIMessage(content="", invalid_tool_calls=[{
            "name": ANSWER_TOOL, "args": json.dumps(ANSWER), "id": "a1",
            "error": "x", "type": "invalid_tool_call"}])
        engine, _ = _engine(whole)
        reply = engine.complete([user(text_block("go"))],
                                output_format=Reading)
        assert reply.parsed.title == "B-1"

    @pytest.mark.parametrize("meta", [
        {"stop_reason": "max_tokens"},                    # ChatAnthropic
        {"finish_reason": "FinishReason.LENGTH"},         # an SDK enum
        {"status": "incomplete"},                         # Responses
        {"incomplete_details": {"reason": "max_output_tokens"}},
    ])
    def test_every_route_s_way_of_saying_it_is_a_cut(self, meta):
        engine, _ = _engine(AIMessage(content="half an ans",
                                      response_metadata=meta))
        assert engine.complete([user(text_block("go"))]).cut_off

    def test_a_finished_reply_is_not_cut(self):
        engine, _ = _engine(_said("whole"))
        assert not engine.complete([user(text_block("go"))]).cut_off
        assert not Reply(stop_reason="end_turn").cut_off
        assert Reply(stop_reason="length").cut_off

    def test_every_call_gets_the_model_s_own_cap(self):
        engine, model = _engine(_said("a"), _said("b"))
        assert engine.output_cap == 32000
        engine.complete([user(text_block("go"))])
        engine.complete([user(text_block("go"))], output_format=Reading)
        assert [r["max_tokens"] for r in model.requests] == [32000, 32000]

    def test_a_call_s_own_cap_is_kept_but_never_above_the_model_s(self):
        engine, model = _engine(_said("a"), _said("b"), _said("c"))
        engine.complete([user(text_block("go"))], max_tokens=8000)
        engine.complete([user(text_block("go"))], max_tokens=64000)
        engine.complete([user(text_block("go"))], max_tokens=8000)
        assert [r["max_tokens"] for r in model.requests] == \
            [8000, 32000, 8000]
        assert model.max_tokens == 32000           # the model is not changed

    def test_a_model_with_no_cap_of_its_own_gets_the_one_asked(self):
        engine, model = _engine(_said("a"), _said("b"), max_tokens=None)
        assert engine.output_cap is None
        engine.complete([user(text_block("go"))])
        engine.complete([user(text_block("go"))], max_tokens=20000)
        assert [r["max_tokens"] for r in model.requests] == [None, 20000]

    def test_the_other_engines_default_is_the_app_s_cap(self):
        assert DEFAULT_MAX_TOKENS == 32000
        assert PrompterEngine(object()).max_tokens == 32000


# ---------------------------------------------------------------------------
# engine_for, on the page
# ---------------------------------------------------------------------------

class TestEngineForFindsThePageModel:

    def test_inside_the_page_s_vision_wrapper(self):
        vision = pytest.importorskip("funhouse_agent.deep.vision_engine")
        chat = ScriptedChatModel()
        engine = engine_for(vision.LangChainVisionEngine(chat))

        assert isinstance(engine, ChatModelEngine)
        assert engine.chat_model is chat
        assert engine.name == "scripted-model"

    def test_the_chat_model_itself(self):
        chat = ScriptedChatModel()
        assert engine_for(chat).chat_model is chat

    def test_a_funhouse_tier_it_cannot_be_is_none(self):
        # The page voter asks for the cheap tier; the graph then falls back
        # to the one engine it has.
        vision = pytest.importorskip("funhouse_agent.deep.vision_engine")
        wrapped = vision.LangChainVisionEngine(ScriptedChatModel())
        assert engine_for(wrapped, model="funhouse-gpt-low") is None
        assert isinstance(engine_for(wrapped, model="scripted-model"),
                          ChatModelEngine)

    def test_a_chat_model_holding_a_prompter_keeps_the_prompter_s_engine(
            self):
        class _Prompter:
            client = None

            def chat(self, **kw):
                return None

        class _Bridge(ScriptedChatModel):
            prompter: object = None

        engine = engine_for(_Bridge(prompter=_Prompter()))
        assert isinstance(engine, PrompterEngine)

    def test_no_chat_model_is_still_no_engine(self):
        class _Wrapper:
            model = "a-model-name"

        assert engine_for(_Wrapper()) is None
        assert engine_for(object()) is None
        assert engine_for(None) is None

    def test_an_engine_is_returned_as_it_is(self):
        engine, _ = _engine()
        assert engine_for(engine) is engine


def test_the_fixture_refuses_what_a_provider_refuses():
    lonely = [HumanMessage(content="go"),
              AIMessage(content="", tool_calls=[
                  {"name": "x", "args": {}, "id": "c", "type": "tool_call"}]),
              HumanMessage(content="and?")]
    with pytest.raises(AssertionError, match="no result"):
        check_conversation(lonely)
    with pytest.raises(AssertionError, match="answers no call"):
        check_conversation([HumanMessage(content="go"),
                            ToolMessage(content="r", tool_call_id="z")])
