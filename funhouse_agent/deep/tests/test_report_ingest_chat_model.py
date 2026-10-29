"""Report ingest on the page's own chat model (BACKLOG RI-03), offline.

With ``GEOTECH_REPORT_INGEST`` on, the GeotechStaffEngineer page reads a
report on the chat model it was built with -- on Foundry a
``palantir_models`` class -- through
:class:`report_ingest.engine.ChatModelEngine`, found inside the wrapper the
page puts round the model for its vision tools. Before RI-03 the page had no
ingest engine on any host and said so.

One turn the way the app runs it, nothing patched: the page's agent built as
``webapp/app.py`` builds it, on a scripted chat model that is the primary
(it delegates the synthetic public report, then answers) AND, through the
answer tool, the ingest's engine (it replays the ingest's own script).
"""

import json

import pytest

pytest.importorskip("planlens.document.roles")
pytest.importorskip("pymupdf")
pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from pydantic import Field  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from report_ingest.engine import ChatModelEngine  # noqa: E402
from report_ingest.tests.chat_fixtures import (  # noqa: E402
    ScriptedChatModel, check_conversation, from_turns,
)
from report_ingest.tests.narrative_fixtures import (  # noqa: E402
    build_narrative_report,
)
from report_ingest.tests.test_graph import full_script  # noqa: E402
# Today's readers are scripted: model first is set off (RI-186).
from report_ingest.tests.test_graph import (  # noqa: E402,F401
    todays_readers)

KEY = "SYN.pdf"


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def report_pdf():
    return build_narrative_report().pdf


class _PageModel(BaseChatModel):
    """The page's one model: the primary when the deep agent's tools are
    bound, the ingest's engine otherwise."""

    bound: list = Field(default_factory=list)
    ingest: list = Field(default_factory=list)
    ingest_requests: list = Field(default_factory=list)
    results: dict = Field(default_factory=dict)
    max_tokens: int = 32000
    model_name: str = "page-model"

    @property
    def _llm_type(self) -> str:
        return "scripted-page-model"

    def bind_tools(self, tools, **kw):
        return self.model_copy(update={"bound": [
            convert_to_openai_tool(t)["function"]["name"] for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if "task" not in self.bound:
            check_conversation(messages)
            self.ingest_requests.append(list(self.bound))
            return ChatResult(generations=[ChatGeneration(
                message=self.ingest.pop(0))])
        results = {m.name: m.content for m in messages if m.type == "tool"}
        self.results.update(results)
        if "task" not in results:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="", tool_calls=[{
                    "name": "task", "id": "c1",
                    "args": {"description": f"Read {KEY} into its record.",
                             "subagent_type": "report_ingest"}}]))])
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="The report is read."))])


def _build(model, report_pdf, tmp_path):
    from funhouse_agent.review_eval.runner import build_page_agent

    files = tmp_path / "conversation" / "files"
    files.mkdir(parents=True, exist_ok=True)
    (files / KEY).write_bytes(report_pdf)
    agent = build_page_agent(model, {KEY: report_pdf}, str(files), [],
                             page="geotech")
    return agent, files


def _run(agent, files, tmp_path):
    from funhouse_agent._fileio import working_dir_bound
    from webapp import core
    from webapp.activity_log import ActivityLogger

    logger = ActivityLogger(str(tmp_path), turn=1)
    history = [{"role": "user", "content": f"Read {KEY}."}]
    answer, shown = None, []
    with working_dir_bound(str(files)):
        for item in core.stream_turn(agent, history, "t1",
                                     recursion_limit=50, callbacks=[logger]):
            if item["kind"] == "turn_done":
                answer = item["answer"]
            elif item["kind"] == "token":
                shown.append(item["text"])
    return answer, "".join(shown)


def test_the_page_reads_a_report_on_its_own_chat_model(report_pdf, tmp_path,
                                                       monkeypatch):
    monkeypatch.setenv(review_flags.REPORT_INGEST_ENV, "1")
    script = full_script()
    model = _PageModel(ingest=from_turns(script))
    agent, files = _build(model, report_pdf, tmp_path)

    answer, shown = _run(agent, files, tmp_path)

    assert answer == "The report is read."
    # The ingest's own calls run on the same model but never stream into
    # the user's live view.
    assert shown.strip() == "The report is read."
    # Every call of the ingest went to the page's model, through the answer
    # tool, and the page voter fell back to the same model.
    assert len(model.ingest_requests) == len(script)
    assert all("answer" in bound for bound in model.ingest_requests)
    record = files / "report_ingest" / "SYN" / "report.record.json"
    assert record.is_file()
    delegated = json.loads(model.results["task"])
    assert delegated["error"] == ""
    assert delegated["model_calls"] == len(script)
    assert delegated["paths"]["record"] == str(record)
    # And the conversation's activity log has every one of them, under the
    # helper's name, with its tokens.
    with open(tmp_path / "activity.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    ingest = [r for r in rows if r["event"] == "model_end"
              and r.get("agent") == "report_ingest"]
    assert len(ingest) == len(script)
    assert all(r.get("usage") for r in ingest)


def test_the_factories_find_the_page_model_and_no_cheap_tier():
    from funhouse_agent.deep.agent import _report_ingest_engine_factory
    from funhouse_agent.deep.tools import VISION_TIER
    from funhouse_agent.deep.vision_engine import LangChainVisionEngine

    chat = ScriptedChatModel()
    wrapped = LangChainVisionEngine(chat)

    engine = _report_ingest_engine_factory(wrapped)()
    assert isinstance(engine, ChatModelEngine)
    assert engine.chat_model is chat
    # A fresh engine, and so a fresh meter, for every run.
    assert _report_ingest_engine_factory(wrapped)() is not engine
    assert _report_ingest_engine_factory(wrapped, model=VISION_TIER)() is None


def test_the_tool_runs_on_the_page_model(report_pdf, tmp_path):
    from funhouse_agent.deep.tools import make_report_ingest_tool
    from funhouse_agent.deep.vision_engine import LangChainVisionEngine

    chat = ScriptedChatModel(replies=from_turns(full_script()))
    (tool,) = make_report_ingest_tool(
        engine=LangChainVisionEngine(chat), out_dir=str(tmp_path),
        attachments={KEY: report_pdf}, max_model_calls=1000)
    answer = json.loads(tool.invoke({"source": KEY}))

    assert "error" not in answer or answer["error"] == ""
    assert answer["n_pages"] == 22
    assert answer["model_calls"] == len(full_script()) == len(chat.requests)
