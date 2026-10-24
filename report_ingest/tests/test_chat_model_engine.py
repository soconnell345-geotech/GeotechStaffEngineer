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

    def test_on_the_page_s_own_tier_not_a_fixed_one(self):
        # PrompterChatModel reads the tier selected on the Prompter at every
        # call unless it was built with one of its own.
        class _Prompter:
            client = None
            chat_model = "funhouse-gpt-high"

            def chat(self, **kw):
                return None

        class _Bridge(ScriptedChatModel):
            prompter: object = None
            model: Optional[str] = None

        selected = engine_for(_Bridge(prompter=_Prompter()))
        built = engine_for(_Bridge(prompter=_Prompter(),
                                   model="funhouse-gpt-low"))
        asked = engine_for(_Bridge(prompter=_Prompter()),
                           model="funhouse-gpt-medium")

        assert selected.model == "funhouse-gpt-high"
        assert built.model == "funhouse-gpt-low"
        assert asked.model == "funhouse-gpt-medium"
        # Inside the wrapper the page puts round its model, the same.
        vision = pytest.importorskip("funhouse_agent.deep.vision_engine")
        paged = engine_for(vision.LangChainVisionEngine(
            _Bridge(prompter=_Prompter())))
        assert isinstance(paged, PrompterEngine)
        assert paged.model == "funhouse-gpt-high"

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


# ---------------------------------------------------------------------------
# what is taken as an answer (the review of RI-03)
# ---------------------------------------------------------------------------

def _results(request):
    """A request's tool results by the call they answer."""
    return {m.tool_call_id: m for m in request["messages"]
            if isinstance(m, ToolMessage)}


class TestACutOrEmptyAnswerIsNoAnswer:
    """Foundry's classes hand back ``{}`` for a call cut mid-arguments, and
    the readings' own schemas accept ``{}``: taken, a narrative cut at the
    cap would be an empty reading, saved as finished."""

    def test_the_real_schemas_do_accept_empty_arguments(self):
        # Why the schema below is a real one, not a stricter stand-in.
        from report_ingest.bound import BoundIdentity
        from report_ingest.narrative_reader import NarrativeReading

        assert NarrativeReading.model_validate({}) is not None
        assert BoundIdentity.model_validate({}) is not None

    @pytest.mark.parametrize("finish", ["length", "tool_calls"])
    def test_empty_arguments_are_no_answer_and_the_model_is_told(self,
                                                                finish):
        from report_ingest.narrative_reader import NarrativeReading

        engine, model = _engine(answer_call({}, call_id="a1",
                                            finish_reason=finish),
                                _said("again"))
        messages = [user(text_block("read the narrative"))]
        reply = engine.complete(messages, output_format=NarrativeReading)

        assert reply.parsed is None
        assert any("not taken" in note for note in reply.notes)
        messages += [{"role": "assistant", "content": reply.content},
                     user(text_block("answer again"))]
        engine.complete(messages, output_format=NarrativeReading)
        told = _results(model.requests[1])["a1"]
        assert told.content.startswith("Not taken:")
        assert told.status == "error"

    def test_a_cut_reply_is_no_answer_even_if_its_arguments_validate(self):
        from report_ingest.narrative_reader import NarrativeReading

        whole_looking = NarrativeReading().model_dump(mode="json")
        engine, _ = _engine(answer_call(whole_looking,
                                        finish_reason="length"))
        reply = engine.complete([user(text_block("go"))],
                                output_format=NarrativeReading)
        assert reply.cut_off and reply.parsed is None
        assert any("output limit" in note for note in reply.notes)

    def test_empty_json_text_is_no_answer_either(self):
        from report_ingest.narrative_reader import NarrativeReading

        engine, _ = _engine(_said("{}"))
        reply = engine.complete([user(text_block("go"))],
                                output_format=NarrativeReading)
        assert reply.parsed is None


class TestAnAnswerBesideALook:

    def _both(self):
        return AIMessage(content="", tool_calls=[
            {"name": "zoom_plot", "args": {"page": 1}, "id": "z",
             "type": "tool_call"},
            {"name": ANSWER_TOOL, "args": ANSWER, "id": "a",
             "type": "tool_call"}])

    def test_when_the_pass_ran_the_look_the_answer_is_said_not_taken(self):
        engine, model = _engine(self._both(), _said("ok"))
        messages = [user(text_block("go"))]
        first = engine.complete(messages, tools=[ZOOM], output_format=Reading)
        messages += [{"role": "assistant", "content": first.content},
                     {"role": "user",
                      "content": [tool_result_block("z", "the box")]}]
        engine.complete(messages, tools=[ZOOM], output_format=Reading)

        told = _results(model.requests[1])
        assert told["z"].content == "the box"
        assert told["a"].content.startswith(
            "Not taken: you asked to look first")

    def test_when_the_pass_took_the_answer_it_is_said_received(self):
        engine, model = _engine(self._both(), _said("ok"))
        messages = [user(text_block("go"))]
        first = engine.complete(messages, tools=[ZOOM], output_format=Reading)
        assert first.parsed is not None
        messages += [{"role": "assistant", "content": first.content},
                     user(text_block("settle what you were unsure of"))]
        engine.complete(messages, tools=[ZOOM], output_format=Reading)

        told = _results(model.requests[1])
        assert told["a"].content == "Answer received."
        assert told["z"].content == "(this call was not run)"

    def test_an_answer_that_failed_validation_is_told_why(self):
        bad = answer_call({"title": "B-1",
                           "layers": [{"top_m": "deep", "soil": "clay"}]},
                          call_id="a1")
        engine, model = _engine(bad, _said("ok"))
        messages = [user(text_block("go"))]
        reply = engine.complete(messages, output_format=Reading)
        assert reply.parsed is None
        messages += [{"role": "assistant", "content": reply.content},
                     user(text_block("again"))]
        engine.complete(messages, output_format=Reading)

        told = _results(model.requests[1])["a1"]
        assert "does not match the schema" in told.content
        assert "layers.0.top_m" in told.content


class TestAStrayKey:
    """Foundry sends tools without strict mode, and the readers' schemas
    forbid extra fields: one stray key would lose the whole reading."""

    def _calc_args(self):
        from report_ingest.tests.test_calc_reader import _reading, _value

        return _reading(results=[_value("Total settlement", 1.2, "in")]
                        ).model_dump(mode="json")

    def test_unknown_keys_are_dropped_named_and_the_rest_kept(self):
        from report_ingest.calc_reader import CalcReading

        args = self._calc_args()
        args["confidence"] = "high"
        args["results"][0]["colour"] = "red"
        engine, _ = _engine(answer_call(args))
        reply = engine.complete([user(text_block("go"))],
                                output_format=CalcReading)

        assert isinstance(reply.parsed, CalcReading)
        assert reply.parsed.results[0].name == "Total settlement"
        (note,) = reply.notes
        assert "confidence" in note and "results.0.colour" in note

    def test_a_wrong_value_still_fails_with_the_stray_key_gone(self):
        from report_ingest.calc_reader import CalcReading

        args = self._calc_args()
        args["confidence"] = "high"
        args["results"][0]["value"] = "a lot"
        engine, _ = _engine(answer_call(args))
        reply = engine.complete([user(text_block("go"))],
                                output_format=CalcReading)
        assert reply.parsed is None
        assert any("results.0.value" in note for note in reply.notes)

    def test_a_stray_key_beside_a_real_answer_is_still_dropped(self):
        from report_ingest.narrative_reader import NarrativeReading

        engine, _ = _engine(answer_call({"projectName": "Made-up Depot",
                                         "confidence": "high"}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=NarrativeReading)
        assert reply.parsed.projectName == "Made-up Depot"
        assert reply.notes == [
            "unknown keys dropped from the answer: confidence"]
        (row,) = engine.meter.to_dict()["per_call"]
        assert json.loads(row["answer_args"][0])["confidence"] == "high"


class TestAWrappedAnswer:
    """2026-10-24: on the local Claude narrative run the model sent 9 of 20
    chunk answers as ``{"arguments": {...the whole answer...}}``. The
    reading's schema forbids the key, so it was dropped, and what was left,
    ``{}``, passed as an empty reading. Now the wrapper comes off; and an
    answer left with nothing once its unknown keys are dropped is not
    taken. Made-up values only."""

    WHOLE = {"documentType": "geotechnical report",
             "projectName": "Made-up Depot", "projectNumber": "X-0001"}

    def _read(self, args, **kw):
        from report_ingest.narrative_reader import NarrativeReading

        engine, model = _engine(answer_call(args, call_id="a1"),
                                _said("again"), **kw)
        messages = [user(text_block("read the narrative"))]
        reply = engine.complete(messages, output_format=NarrativeReading)
        return engine, model, messages, reply

    def test_a_wrapped_object_is_unwrapped_and_named(self):
        engine, _, _, reply = self._read({"arguments": dict(self.WHOLE)})
        assert reply.parsed.projectName == "Made-up Depot"
        assert reply.parsed.projectNumber == "X-0001"
        assert reply.notes == [
            "the answer arrived wrapped in 'arguments'; unwrapped"]
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["notes"] == reply.notes
        assert json.loads(row["answer_args"][0]) == {
            "arguments": self.WHOLE}

    def test_any_wrapper_name_and_json_text_inside_it(self):
        engine, _, _, reply = self._read(
            {"payload": json.dumps(self.WHOLE)})
        assert reply.parsed.projectName == "Made-up Depot"
        assert reply.notes == [
            "the answer arrived wrapped in 'payload'; unwrapped"]

    def test_the_model_is_told_received_for_an_unwrapped_answer(self):
        engine, model, messages, reply = self._read(
            {"arguments": dict(self.WHOLE)})
        messages += [{"role": "assistant", "content": reply.content},
                     user(text_block("go on"))]
        from report_ingest.narrative_reader import NarrativeReading
        engine.complete(messages, output_format=NarrativeReading)
        assert _results(model.requests[1])["a1"].content == (
            "Answer received.")

    def test_a_field_of_the_schema_is_never_unwrapped(self):
        class Holder(BaseModel):
            model_config = {"extra": "forbid"}
            site: Optional[dict] = None
            note: Optional[str] = None

        engine, _ = _engine(answer_call({"site": {"note": "inner"}}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=Holder)
        assert reply.parsed.site == {"note": "inner"}
        assert reply.parsed.note is None
        assert reply.notes == []

    def test_a_scalar_under_an_unknown_key_is_dropped_and_not_taken(self):
        engine, model, messages, reply = self._read({"arguments": "yes"})
        assert reply.parsed is None
        assert reply.notes == [
            "unknown keys dropped from the answer: arguments",
            "an answer was not taken: nothing was left once the unknown "
            "keys arguments were dropped"]
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["answer_args"] == ['{"arguments": "yes"}']

        from report_ingest.narrative_reader import NarrativeReading
        messages += [{"role": "assistant", "content": reply.content},
                     user(text_block("answer again"))]
        engine.complete(messages, output_format=NarrativeReading)
        told = _results(model.requests[1])["a1"]
        assert told.status == "error"
        assert told.content.startswith("Not taken: nothing was left")
        assert "Call `answer` again" in told.content

    def test_an_empty_wrapped_object_is_not_taken_either(self):
        _, _, _, reply = self._read({"arguments": {}})
        assert reply.parsed is None
        assert any("nothing was left" in note for note in reply.notes)

    def test_every_field_at_its_default_after_the_drop_is_not_taken(self):
        # Two keys, and the unknown one holds no object, so no wrapper: it
        # is dropped, and the only field left is a null, which is every
        # field at its default.
        _, _, _, reply = self._read({"arguments": "see the pages",
                                     "projectName": None})
        assert reply.parsed is None
        assert any("nothing was left once the unknown keys arguments"
                   in note for note in reply.notes)

    def test_the_answer_beside_nulls_is_set_aside_and_named(self):
        # Beside other keys an object is never taken for the answer; the
        # nulls left are every field at its default, so nothing is taken,
        # and the object that named the answer's fields is named.
        engine, _, _, reply = self._read({"arguments": dict(self.WHOLE),
                                          "projectName": None})
        assert reply.parsed is None
        assert reply.notes[0] == (
            "the unknown key 'arguments' held an object naming fields of "
            "the answer (documentType, projectName, projectNumber); set "
            "aside, not merged")
        (row,) = engine.meter.to_dict()["per_call"]
        assert "Made-up Depot" in row["answer_args"][0]

    def test_a_schema_that_ignores_unknown_keys_is_held_to_it_too(self):
        class Loose(BaseModel):                 # pydantic's default: ignore
            site: Optional[str] = None

        engine, _ = _engine(answer_call({"site_name": "Made-up Depot",
                                         "site_code": "X-0001"}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=Loose)
        assert reply.parsed is None
        assert reply.notes == [
            "an answer was not taken: nothing was left once the unknown "
            "keys site_name, site_code were passed over"]

    def test_a_wrapped_answer_written_out_as_text_is_unwrapped(self):
        from report_ingest.narrative_reader import NarrativeReading

        engine, _ = _engine(_said(json.dumps({"answer": self.WHOLE})))
        reply = engine.complete([user(text_block("go"))],
                                output_format=NarrativeReading)
        assert reply.parsed.projectNumber == "X-0001"
        assert reply.notes == [
            "the answer arrived wrapped in 'answer'; unwrapped"]

    def test_the_raw_arguments_kept_are_cut_to_size(self):
        from report_ingest.engine import RAW_ANSWER_CHARS

        engine, _, _, reply = self._read({"arguments": "x" * 50_000})
        assert reply.parsed is None
        (row,) = engine.meter.to_dict()["per_call"]
        (kept,) = row["answer_args"]
        assert kept.startswith('{"arguments": "xxx')
        assert len(kept) < RAW_ANSWER_CHARS + 100
        assert kept.endswith("characters]")

    def test_a_clean_answer_keeps_no_raw_arguments(self):
        engine, _, _, reply = self._read(dict(self.WHOLE))
        assert reply.parsed.projectName == "Made-up Depot"
        assert reply.notes == []
        (row,) = engine.meter.to_dict()["per_call"]
        assert "answer_args" not in row and "notes" not in row

    def test_a_wrapper_two_deep_is_one_note(self):
        _, _, _, reply = self._read({"arguments": {"answer": self.WHOLE}})
        assert reply.parsed.projectName == "Made-up Depot"
        assert reply.notes == [
            "the answer arrived wrapped in 'arguments', then 'answer'; "
            "unwrapped"]

    def test_unwrapped_then_refused_says_so(self):
        _, _, _, reply = self._read({"arguments": {"colour": "red"}})
        assert reply.parsed is None
        assert reply.notes == [
            "the answer arrived wrapped in 'arguments'; unwrapped, and "
            "still not taken",
            "unknown keys dropped from the answer: colour",
            "an answer was not taken: nothing was left once the unknown "
            "keys colour were dropped"]

    def test_an_alias_of_a_field_is_never_unwrapped(self):
        class Aliased(BaseModel):
            model_config = {"extra": "forbid"}
            site: Optional[dict] = Field(None, alias="siteInfo")

        engine, _ = _engine(answer_call({"siteInfo": {"name": "inner"}}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=Aliased)
        assert reply.parsed.site == {"name": "inner"}
        assert reply.notes == []


class TestAnObjectBesideTheAnswer:
    """An unknown key BESIDE the answer's fields is a stray, whatever it
    holds. Models add maps of their own that name the answer's fields with
    other values -- evidence, units, sources, an alternative -- so it is
    taken as the answer with the key dropped or passed over: never merged
    in, never a reason to refuse. When the object named fields of the
    answer the key is named, for the run's check to warn of. Made-up values
    only."""

    def _complete(self, schema, args):
        engine, _ = _engine(answer_call(args))
        reply = engine.complete([user(text_block("go"))],
                                output_format=schema)
        (row,) = engine.meter.to_dict()["per_call"]
        return reply, row

    def _triage(self, **over):
        answer = {"document_type": "geotechnical report",
                  "bound_together": [], "has_narrative": True,
                  "has_logs": True, "has_lab": False, "has_calcs": False,
                  "languages": ["en"], "toc_agreement": "agrees",
                  "anomalies": [], "workflow": "standard",
                  "rationale": "made-up rationale"}
        answer.update(over)
        return answer

    def test_triage_with_an_evidence_map_is_taken(self):
        from report_ingest.triage import TriageFindings

        reply, row = self._complete(TriageFindings, self._triage(
            evidence={"has_logs": "pages 5-20", "has_lab": "none"}))
        assert reply.parsed is not None
        assert reply.parsed.has_logs is True and reply.parsed.has_lab is False
        assert reply.notes == [
            "the unknown key 'evidence' held an object naming fields of the "
            "answer (has_lab, has_logs); set aside, not merged"]
        assert "pages 5-20" in row["answer_args"][0]

    def test_a_page_label_with_an_alternative_is_taken(self):
        from report_ingest.vision_labels import VisionPageAnswer

        reply, _ = self._complete(VisionPageAnswer, {
            "page": 3, "label": "narrative", "confidence": 0.9,
            "reason": "made-up reason",
            "alternative": {"label": "figure", "confidence": 0.1}})
        assert (reply.parsed.label, reply.parsed.confidence) == (
            "narrative", 0.9)
        assert len(reply.notes) == 1 and "'alternative'" in reply.notes[0]

    def test_a_narrative_null_is_not_filled_from_a_sources_map(self):
        from report_ingest.narrative_reader import NarrativeReading

        reply, _ = self._complete(NarrativeReading, {
            "projectName": None, "projectNumber": "X-0001",
            "sources": {"projectName": "page 3"}})
        assert reply.parsed.projectName is None
        assert reply.parsed.projectNumber == "X-0001"
        assert reply.notes[-1] == (
            "unknown keys dropped from the answer: sources")

    def test_a_log_with_a_units_map_keeps_its_own_values(self):
        from report_ingest.log_reader import LogReading

        reply, _ = self._complete(LogReading, {
            "investigation_id": "B-1", "total_depth": 40.0,
            "units": {"total_depth": "ft"}})
        assert reply.parsed.total_depth == 40.0
        assert reply.notes == [
            "the unknown key 'units' held an object naming fields of the "
            "answer (total_depth); set aside, not merged",
            "unknown keys dropped from the answer: units"]

    def test_the_answer_hidden_beside_an_id_is_set_aside_and_named(self):
        # The case the engine cannot tell from a map of the model's own: it
        # is taken as what is beside it, and the check warns of it.
        from report_ingest.log_reader import LogReading

        reply, row = self._complete(LogReading, {
            "investigation_id": "B-1",
            "arguments": {"investigation_id": "B-1", "total_depth": 12.5}})
        assert reply.parsed.investigation_id == "B-1"
        assert reply.parsed.total_depth is None, "never merged in"
        assert "'arguments' held an object" in reply.notes[0]
        assert "12.5" in row["answer_args"][0]

    def test_an_object_that_names_no_field_is_only_dropped(self):
        from report_ingest.log_reader import LogReading

        reply, _ = self._complete(LogReading, {
            "investigation_id": "B-1", "meta": {"source": "made-up"}})
        assert reply.parsed.investigation_id == "B-1"
        assert reply.notes == ["unknown keys dropped from the answer: meta"]


class TestAnEmptyAnswerWithAStrayKey:
    """ON PURPOSE: a schema that builds a blank answer from ``{}`` cannot
    tell an answer with nothing to say and a stray key from one lost to its
    wrapper, so both are refused and the model is told why."""

    def test_the_conflict_comments_with_a_summary_beside_them(self):
        from report_ingest.reconciler import _ConflictComments

        engine, _ = _engine(answer_call({"comments": [],
                                         "summary": "none"}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=_ConflictComments)
        assert reply.parsed is None
        assert reply.notes == [
            "an answer was not taken: nothing was left once the unknown "
            "keys summary were passed over"]

    def test_a_bound_identity_with_a_null_title_and_its_reasoning(self):
        from report_ingest.bound import BoundIdentity

        engine, _ = _engine(answer_call({"title": None,
                                         "reasoning": "made-up reasoning"}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=BoundIdentity)
        assert reply.parsed is None
        assert any("reasoning were passed over" in note
                   for note in reply.notes)

    def test_the_same_answer_without_the_stray_key_is_taken(self):
        from report_ingest.reconciler import _ConflictComments

        engine, _ = _engine(answer_call({"comments": []}))
        reply = engine.complete([user(text_block("go"))],
                                output_format=_ConflictComments)
        assert reply.parsed is not None and reply.parsed.comments == []
        assert reply.notes == []


class TestJsonInTheWordsOfALook:
    """A turn that asks to look may say ``{"page": 4}`` in its words: that is
    not an answer, and is not noted as one refused."""

    def _turn(self, words):
        return AIMessage(content=words, tool_calls=[
            {"name": "zoom_plot", "args": {"page": 4}, "id": "z",
             "type": "tool_call"}],
            response_metadata={"finish_reason": "tool_calls"})

    def test_it_is_not_noted_as_a_lost_answer(self):
        from report_ingest.calc_reader import CalcReading

        engine, _ = _engine(self._turn('Looking closer at {"page": 4}.'))
        reply = engine.complete([user(text_block("go"))], tools=[ZOOM],
                                output_format=CalcReading)
        assert reply.parsed is None and reply.notes == []
        assert [c.name for c in reply.tool_calls] == ["zoom_plot"]
        (row,) = engine.meter.to_dict()["per_call"]
        assert "notes" not in row and "answer_args" not in row

    def test_json_that_names_a_field_is_still_noted(self):
        from report_ingest.calc_reader import CalcReading

        engine, _ = _engine(self._turn('{"kind": ["x"], "page": 4}'))
        reply = engine.complete([user(text_block("go"))], tools=[ZOOM],
                                output_format=CalcReading)
        assert reply.parsed is None
        assert any("not taken" in note for note in reply.notes)

    def test_with_no_look_asked_the_refusal_is_noted(self):
        from report_ingest.calc_reader import CalcReading

        engine, _ = _engine(_said('{"page": 4}'))
        reply = engine.complete([user(text_block("go"))], tools=[ZOOM],
                                output_format=CalcReading)
        assert reply.parsed is None
        assert any("nothing was left" in note for note in reply.notes)


# -- every request in the cost record (BACKLOG RI-21) -------------------------

class TestTheCostRecord:

    def test_a_request_that_fails_is_recorded_then_raised(self):
        engine, _ = _engine(TimeoutError("the request timed out"))
        with pytest.raises(TimeoutError):
            engine.complete([user(text_block("go"))])
        meter = engine.meter
        assert (meter.calls, meter.failed, meter.retried) == (0, 1, 0)
        assert meter.failed_by_kind == {"timeout": 1}
        (row,) = meter.to_dict()["per_call"]
        assert (row["call"], row["outcome"]) == (1, "error")
        assert "attempt" not in row, "a first attempt is left out"
        assert row["error"].startswith("TimeoutError: the request timed out")

    def test_a_failed_request_and_the_next_are_two_calls(self):
        engine, _ = _engine(RuntimeError("Error code: 429 - slow down"),
                            _said("ok"))
        with pytest.raises(RuntimeError):
            engine.complete([user(text_block("go"))])
        engine.complete([user(text_block("go"))])
        rows = engine.meter.to_dict()["per_call"]
        assert [(r["call"], r["outcome"]) for r in rows] == [
            (1, "error"), (2, "reply")]
        assert rows[0]["error_kind"] == "rate_limit"
        assert rows[1]["stop"] == "stop"

    def test_reasoning_tokens_are_read_from_langchains_details(self):
        message = AIMessage(
            content="ok", response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 100, "output_tokens": 80,
                            "total_tokens": 180,
                            "output_token_details": {"reasoning": 50}})
        engine, _ = _engine(message)
        reply = engine.complete([user(text_block("go"))])
        assert reply.usage.reasoning_tokens == 50
        assert engine.meter.reasoning_tokens == 50
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["reasoning_tokens"] == 50

    def test_reasoning_tokens_fall_back_to_the_providers_own_count(self):
        message = AIMessage(content="ok", response_metadata={
            "finish_reason": "stop",
            "token_usage": {"prompt_tokens": 40, "completion_tokens": 30,
                            "completion_tokens_details": {
                                "reasoning_tokens": 12}}})
        engine, _ = _engine(message)
        reply = engine.complete([user(text_block("go"))])
        assert reply.usage.reasoning_tokens == 12

    def test_what_was_done_with_the_answer_lands_on_the_calls_row(self):
        args = dict(ANSWER)
        args["depth_m"] = "very deep"
        engine, _ = _engine(answer_call(args))
        reply = engine.complete([user(text_block("go"))],
                                output_format=Reading)
        assert reply.parsed is None
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["notes"] == reply.notes
        assert any("depth_m" in note for note in row["notes"])

    def test_a_cap_that_cannot_be_set_is_recorded_as_an_adaptation(self):
        class _NoCapCopy(ScriptedChatModel):
            def model_copy(self, *, update=None, deep=False):
                if update and "max_tokens" in update:
                    raise TypeError("this model's fields are frozen")
                return super().model_copy(update=update, deep=deep)

        model = _NoCapCopy(replies=[_said("ok"), _said("again")])
        engine = ChatModelEngine(model)
        engine.complete([user(text_block("go"))], max_tokens=1000)
        engine.complete([user(text_block("go"))], max_tokens=1000)
        assert model.requests[0]["max_tokens"] == 32000, "its own cap"
        (lesson,) = engine.adaptations
        assert "a cap of 1000 could not be set" in lesson
        rows = engine.meter.to_dict()["per_call"]
        assert rows[0]["adapted"] == [lesson]
        assert "adapted" not in rows[1], "recorded once, when learned"
        assert engine.meter.adaptations == [lesson]


    def test_a_reply_with_no_reasoning_split_says_none(self):
        engine, _ = _engine(_said("ok"))
        reply = engine.complete([user(text_block("go"))])
        assert reply.usage.reasoning_tokens is None
        assert engine.meter.reasoning_reported == 0

    def test_a_reasoning_count_that_is_not_a_number_is_none(self):
        # LangChain checks its own usage_metadata; the provider's count in
        # the response metadata is not checked by anyone but this engine.
        message = AIMessage(content="ok", response_metadata={
            "finish_reason": "stop",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5,
                            "completion_tokens_details": {
                                "reasoning_tokens": "n/a"}}})
        engine, _ = _engine(message)
        reply = engine.complete([user(text_block("go"))])
        assert reply.text == "ok" and reply.usage.reasoning_tokens is None
