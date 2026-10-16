"""EV-20 (live wave w8): the record could not say which tools and helpers a
turn ran with; whether plot_code was offered had to be read from the code.
A built agent now carries ``geotech_build``, and the turn's engine record
(``webapp.core.engine_record``, kept on ``turn_start`` and in meta.json)
includes it."""

from funhouse_agent import review_flags
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.tests.test_figure_helper import _Quiet


def test_a_built_agent_says_what_it_offers(monkeypatch):
    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "1")
    agent = build_deep_agent(model=_Quiet(), reference_mode="off",
                             enable_calc_subagent=True)
    build = agent.geotech_build
    assert "call_agent" in build["tools"] and "task" in build["tools"]
    assert {"calc", "figure"} <= set(build["helpers"])
    assert "GEOTECH_PLOT_CODE=1" in build["switches"]
    assert build["prompt_chars"] > 1000


def test_the_engine_record_carries_it():
    from webapp import core
    agent = build_deep_agent(model=_Quiet(), reference_mode="off")
    rec = core.engine_record(agent)
    assert rec.get("build", {}).get("tools")
