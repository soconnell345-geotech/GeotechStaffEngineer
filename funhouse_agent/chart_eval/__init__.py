"""The chart suite: how well a model reads professional charts, harness by
harness.

Out of the owner's question (2026-11-06): does the setup Anthropic used for
the Opus 5.5 system card's Chartography result (64 % without tools, 89 %
with a container and an image-cropping tool) read charts better than this
app's Document Review page? The suite answers it on Surge AI's Chartography
(100 expert chart questions, graded against expert ranges) with one install,
on whatever model the host offers:

* ``none``: the chart and the question, no tools (Surge's own harness);
* ``zoom``: the Claude Cookbook's zoom tool (the published half of the
  system card's setup);
* ``zoom_code``: zoom plus code on the full-resolution chart in the app's
  locked box (the nearest the app comes to "a container and a cropping
  tool");
* ``page:baseline`` / ``page:minimal`` (any ``page:<review arm>``): the
  chart uploaded to the Document Review page, as a user would.

Run on Foundry (``module_work/FOUNDRY_BRIEF_CHARTOGRAPHY.md``)::

    from funhouse_agent.chart_eval import score_chart_suite, foundry_model
    opus = foundry_model("Opus 5.5=anthropic:<Opus 5.5 RID>")
    res = score_chart_suite(opus, judge_model=foundry_model("<judge id>"),
                            data_dir="<chartography folder>",
                            out_dir="<output folder>")
"""

from __future__ import annotations

from typing import Any

from funhouse_agent.chart_eval.arms import DIRECT_ARMS, run_direct
from funhouse_agent.chart_eval.data import (ChartDataError, ChartTask,
                                            load_chart_tasks)
from funhouse_agent.chart_eval.judge import judge_answer
from funhouse_agent.chart_eval.runner import (DEFAULT_ARMS, resolve_arm,
                                              score_chart_suite)


def foundry_model(model_id: str) -> Any:
    """A chat model built the way the app builds it (``webapp.engine_config.
    resolve_engine``): a Foundry RID (``anthropic:<RID>`` / ``openai:<RID>``
    states the family), a Palantir model API name, or, off Foundry, any id
    the app takes. A ``Label=id`` entry as written in
    ``GEOTECH_FOUNDRY_MODELS`` is taken too. Raises with the app's own
    message when the model cannot be built."""
    from webapp.engine_config import resolve_engine
    mid = str(model_id).strip()
    if "=" in mid:
        mid = mid.split("=", 1)[1].strip()
    res = resolve_engine(mid)
    if getattr(res, "model", None) is None:
        raise RuntimeError(f"could not build {model_id!r}: "
                           f"{getattr(res, 'message', res)}")
    return res.model


__all__ = ["score_chart_suite", "foundry_model", "load_chart_tasks",
           "ChartTask", "ChartDataError", "judge_answer", "run_direct",
           "resolve_arm", "DIRECT_ARMS", "DEFAULT_ARMS"]
