"""The chart suite: how well a model reads professional charts, harness by
harness.

Out of the owner's question (2026-11-06): does the setup Anthropic used for
the Opus 5.5 system card's Chartography result (64 % without tools, 89 %
with a container and an image-cropping tool) read charts better than this
app's Document Review page? Chartography itself was not open to the owner
(private on Hugging Face), so the suite runs on CharXiv's reasoning
questions (1,000 real arXiv charts with hand-written answers;
:mod:`.charxiv` makes the data folder), and on Chartography where access is
granted (:mod:`.prepare`). One install, whatever model the host offers:

* ``none``: the chart and the question, no tools (Surge's own harness);
* ``zoom``: the Claude Cookbook's zoom tool (the published half of the
  system card's setup);
* ``zoom_code``: zoom plus code on the full-resolution chart in the app's
  locked box (the nearest the app comes to "a container and a cropping
  tool");
* ``page:baseline`` / ``page:minimal`` (any ``page:<review arm>``): the
  chart uploaded to the Document Review page, as a user would.

Run on Foundry (``module_work/FOUNDRY_BRIEF_CHART_SUITE.md``)::

    from funhouse_agent.chart_eval import score_chart_suite, foundry_model
    opus = foundry_model("Opus 5.5=anthropic:<Opus 5.5 RID>")
    res = score_chart_suite(opus, judge_model=foundry_model("<judge id>"),
                            data_dir="<charxiv_data folder>",
                            subset="sample_100.txt",
                            out_dir="<output folder>")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from funhouse_agent.chart_eval.arms import DIRECT_ARMS, run_direct
from funhouse_agent.chart_eval.data import (ChartDataError, ChartTask,
                                            load_chart_tasks)
from funhouse_agent.chart_eval.judge import judge_answer
from funhouse_agent.chart_eval.runner import (DEFAULT_ARMS, resolve_arm,
                                              score_chart_suite)

#: The id each model :func:`foundry_model` built was made from, keyed by the
#: object's id (the object is kept, so the key cannot be reused), so that a
#: forked page run can build its own copy with its own connections
#: (:func:`rebuild_model`).
_BUILT: Dict[int, Tuple[str, Any]] = {}


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
    _BUILT[id(res.model)] = (mid, res.model)
    return res.model


def rebuild_model(model: Any) -> Optional[Any]:
    """A new build of ``model`` from the id :func:`foundry_model` made it
    from (the same settings, new clients and model handles), or ``None``
    when :func:`foundry_model` did not make it. Settings changed on the
    object after it was built are not carried over."""
    entry = _BUILT.get(id(model))
    if entry is None or entry[1] is not model:
        return None
    return foundry_model(entry[0])


__all__ = ["score_chart_suite", "foundry_model", "rebuild_model",
           "load_chart_tasks", "ChartTask", "ChartDataError", "judge_answer",
           "run_direct", "resolve_arm", "DIRECT_ARMS", "DEFAULT_ARMS"]
