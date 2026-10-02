"""The looking-only review agent (``GEOTECH_REVIEW_AGENT=minimal``), offline.

A measuring stick the owner asked for on 2026-10-02: the page and zoom tools,
helpers with the same tools, the two output tools, a short generic prompt —
and nothing else, so the suite can say what the rest of the harness is worth.
"""

import json
import os

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from funhouse_agent import inline_store, review_flags  # noqa: E402
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.deep.minimal_agent import MINIMAL_PROMPT  # noqa: E402
from funhouse_agent.deep.tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _call, _model, _review_kwargs, _system_text, _tools_of)
from planlens.testing import build_synthetic_review_document  # noqa: E402

TEXT_TOOLS = {"open_document", "document_structure", "document_page_map",
              "read_document", "search_document", "document_markups",
              "find_quantities", "render_page_thumbnails", "sweep_pages",
              "find_like", "write_todos", "read_text_file", "list_files"}


@pytest.fixture(autouse=True)
def _minimal(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    for k, v in review_flags.ARMS["minimal"].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _agent(model, gt, engine=None, tmp_path=None):
    kw = _review_kwargs()                # carries review_page=True
    return build_deep_agent(model,
                            attachments={"review_set.pdf": gt.pdf},
                            engine=engine or FakeEngine(), **kw)


def test_the_arm_builds_the_looking_only_agent(gt):
    assert review_flags.minimal_agent() and not review_flags.lean_agent()
    agent = _agent(_model([]), gt)
    assert agent.geotech_review_agent == "minimal"
    names = set(_tools_of(agent))
    assert {"analyze_pdf_page", "render_region", "analyze_image",
            "page_count", "task", "mark_up"} <= names
    assert not (names & TEXT_TOOLS), names & TEXT_TOOLS
    assert len(MINIMAL_PROMPT) < 3000            # short and generic


def test_a_page_look_is_shown_to_the_model_and_a_zoom_follows(gt):
    """No one-shot vision call: the page image goes to the model itself at
    its next step, and a zoom by view + image_box is shown the same way."""
    engine = FakeEngine()
    seen_images = []

    def after_page(messages):
        imgs = [b for m in messages if isinstance(m, HumanMessage)
                and isinstance(m.content, list)
                for b in m.content if isinstance(b, dict)
                and b.get("type") == "image_url"]
        seen_images.append(len(imgs))
        last = next(m for m in reversed(messages)
                    if getattr(m, "type", "") == "tool")
        view = json.loads(last.content)["view"]
        return _call("render_region", {"attachment_key": "review_set.pdf",
                                       "page": 0, "view": view,
                                       "image_box": [100, 100, 400, 300]}, 2)

    def after_zoom(messages):
        imgs = [b for m in messages if isinstance(m, HumanMessage)
                and isinstance(m.content, list)
                for b in m.content if isinstance(b, dict)
                and b.get("type") == "image_url"]
        seen_images.append(len(imgs))
        return AIMessage(content="Page 1 shows the boring plan.")

    model = _model([
        _call("page_count", {"source": "review_set.pdf"}),
        _call("analyze_pdf_page", {"attachment_key": "review_set.pdf",
                                   "page": 0}),
        after_page, after_zoom])
    agent = _agent(model, gt, engine)
    out = agent.invoke({"messages": [{"role": "user",
                                      "content": "What is on page 1?"}]},
                       config={"recursion_limit": 200})
    assert "boring plan" in out["messages"][-1].content
    assert seen_images[0] >= 1 and seen_images[1] >= 1
    assert engine.prompts == []                  # nobody else looked
    assert "You read by LOOKING" in _system_text(model.seen[0])


def test_page_count_says_how_many_pages_and_nothing_else(gt):
    agent = _agent(_model([]), gt)
    out = json.loads(_tools_of(agent)["page_count"].invoke(
        {"source": "review_set.pdf"}))
    with fitz.open(stream=gt.pdf, filetype="pdf") as doc:
        assert out["pages"] == doc.page_count
    assert set(out) <= {"source", "pages", "page_size_pt", "page_sizes_pt"}


def test_a_helper_looks_with_the_same_tools_and_cannot_write(gt):
    model = _model([
        _call("task", {"description": "In review_set.pdf look at page 0 "
                                      "and say what it shows."}),
        # the helper's own turns
        _call("analyze_pdf_page", {"attachment_key": "review_set.pdf",
                                   "page": 0}, 3),
        AIMessage(content="Page 1: a boring plan."),
        # back in the main agent
        AIMessage(content="The helper reports a boring plan on page 1."),
    ])
    agent = _agent(model, gt)
    out = agent.invoke({"messages": [{"role": "user", "content": "go"}]},
                       config={"recursion_limit": 200})
    assert "boring plan" in out["messages"][-1].content
    helper_bound = [b for b in model.bound if "mark_up" not in b
                    and "analyze_pdf_page" in b]
    assert helper_bound, model.bound
    assert not (set(helper_bound[0]) & (TEXT_TOOLS | {"write_docx",
                                                      "mark_up"}))


def test_mark_up_takes_a_zoom_location_and_writes_a_copy(gt, tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    agent = _agent(_model([]), gt)
    out = json.loads(_tools_of(agent)["mark_up"].invoke({
        "source": "review_set.pdf", "output_path": "marked.pdf",
        "markups": [{"kind": "box", "page": 0, "comment": "check this",
                     "view": [0, 0, 612, 792],
                     "image_box": [100, 100, 300, 200]}]}))
    assert out["n_written"] == 1, out
    assert os.path.isfile(out["output_path"])


def test_the_suite_runs_the_arm(gt, tmp_path):
    from funhouse_agent.review_eval import score_review_suite
    res = score_review_suite(model=_model([AIMessage(content="No markups.")]),
                             out_dir=str(tmp_path / "out"), arms=("minimal",),
                             ids=["fixture-markups"], verbose=False)
    run = res["results"]["runs"]["minimal"]["fixture-markups"] \
        if "runs" in res["results"] else \
        res["results"]["minimal"]["fixture-markups"]
    assert not run.get("error"), run.get("error")
    assert "minimal" in res["results_md"]
