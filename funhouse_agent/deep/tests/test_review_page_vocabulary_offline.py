"""The Document Review page asks a document no geotechnical report questions.

BACKLOG RI-34 (owner, 2026-10-20): "move all questions related to geotech
reports to report_ingest"; geotech reports are not this page's documents,
and the page becomes its own package for people who are not geotechnical
engineers (INF-26). So nothing the page's model is shown - its system
prompts, its reading helper's prompt, every tool description - names a
boring, a laboratory sheet, a DIGGS file or the like, on any of its three
builds, with every review switch off and on. The geotech page keeps its
wording: the phrase swaps (``review_agent.REVIEW_REWRITES``) apply only here,
and each must still match the text it swaps, or the original wording would
come back unnoticed.

Offline: agents built on a fake model, no network.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage  # noqa: E402

from funhouse_agent import review_flags as RF  # noqa: E402

#: Words that name a geotechnical report's things or the tools that read one.
#: "log" alone is not here: a comment log and a log axis are any document's.
VOCAB = re.compile(
    r"\b(?:borings?|boring[- ]logs?|test[- ]pits?|SPT|CPT|blow ?counts?|"
    r"N[- ]values?|laborator\w*|lab (?:sheets?|tests?)|groundwater|"
    r"geotech(?:nical)? reports?|DIGGS|write_diggs|exploration logs?|"
    r"strat(?:um|a)|USCS|AASHTO|soils?|subsurface|report_checklist|"
    r"report_ingest|log_grid|the logs|a log's|plotted logs?)\b", re.I)


class _Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kw):
        return self


#: The builds, each with every review switch off and with the switches on.
_ON = {RF.COVERAGE_ENV: "1", RF.CHECKLIST_ENV: "1", RF.DIGEST_ENV: "1",
       RF.GEOMETRY_ENV: "1", RF.FINDINGS_ENV: "1", RF.SWEEP_ENV: "1",
       RF.OVERVIEW_ENV: "1", RF.KEPT_READINGS_ENV: "1",
       RF.FIGURE_HELPER_ENV: "1"}
BUILDS = {
    "legacy": {},
    "lean": {RF.AGENT_ENV: "lean"},
    "minimal": {RF.AGENT_ENV: "minimal"},
    "legacy+switches": dict(_ON),
    "lean+switches": {RF.AGENT_ENV: "lean", **_ON},
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in RF.ALL_ENVS + RF.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


def _surface(agent):
    tools = agent.nodes["tools"].bound.tools_by_name
    return {f"tool {name}": getattr(tool, "description", "") or ""
            for name, tool in tools.items()}


def _hits(texts):
    out = []
    for where, text in texts.items():
        for m in VOCAB.finditer(text):
            a, b = max(0, m.start() - 50), min(len(text), m.end() + 50)
            out.append(f"{where}: {m.group(0)!r} in ...{text[a:b]}...")
    return out


@pytest.mark.parametrize("build", list(BUILDS))
def test_review_page_names_no_geotech_report_things(build, tmp_path,
                                                    monkeypatch):
    from funhouse_agent.review_eval.runner import build_page_agent
    for env, value in BUILDS[build].items():
        monkeypatch.setenv(env, value)
    agent = build_page_agent(_Model(responses=[AIMessage("x")]), {},
                             str(tmp_path), [], page="review")
    surface = _surface(agent)
    assert "report_checklist" not in " ".join(surface)
    assert "log_grid" not in " ".join(surface)
    assert not _hits(surface), "\n".join(_hits(surface))


def test_the_review_prompts_name_no_geotech_report_things():
    from funhouse_agent.deep import prompt as P
    texts = {"DOCUMENT_REVIEW_PROMPT": P.DOCUMENT_REVIEW_PROMPT,
             "DOCUMENT_REVIEW_PROMPT_LEAN": P.DOCUMENT_REVIEW_PROMPT_LEAN,
             "DOCUMENT_REVIEW_READER_PROMPT":
                 P.DOCUMENT_REVIEW_READER_PROMPT,
             "lean": P.build_document_review_prompt(lean=True),
             "legacy": P.build_document_review_prompt(lean=False)}
    assert not _hits(texts), "\n".join(_hits(texts))


def test_every_phrase_swap_still_matches_the_geotech_wording():
    """A swap whose phrase no longer matches leaves the report wording on
    the review page; the guard above would catch the word, this names the
    swap."""
    from funhouse_agent.deep.review_agent import REVIEW_REWRITES
    from funhouse_agent.deep.tools import make_vision_tools
    tools = {t.name: t.description for t in make_vision_tools(engine=None)}
    for name, swaps in REVIEW_REWRITES.items():
        if name not in tools:          # a tool this install does not serve
            continue
        for old, _new in swaps:
            assert old in tools[name], (name, old)


def test_the_figure_helper_swap_still_matches_its_description():
    from funhouse_agent.deep.figure_helper import FIGURE_HELPER_DESCRIPTION
    from funhouse_agent.deep.review_agent import REVIEW_HELPER_REWRITES
    for old, _new in REVIEW_HELPER_REWRITES["figure"]:
        assert old in FIGURE_HELPER_DESCRIPTION, old


def test_the_geotech_page_keeps_its_wording(tmp_path):
    from funhouse_agent.review_eval.runner import build_page_agent
    agent = build_page_agent(_Model(responses=[AIMessage("x")]), {},
                             str(tmp_path), [], page="geotech")
    tools = agent.nodes["tools"].bound.tools_by_name
    if "plot_data" in tools:
        assert "boring" in tools["plot_data"].description
    if "log_grid" in {t for t in tools}:
        assert "boring log" in tools["log_grid"].description
