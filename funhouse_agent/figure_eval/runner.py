"""Run the figure test set: every request under plot code OFF and ON, each
run's figures marked by a vision model, compared and reported (GM-314).

The notebook entry point is :func:`score_figure_set`, on a host that has
the models (the paid run happens there, never from a developer's machine)::

    from funhouse_agent import chart_eval, figure_eval
    agent = chart_eval.foundry_model("<the app's agent model>")
    eyes = chart_eval.foundry_model("<the vision model>")
    figure_eval.score_figure_set(agent, vision_model=eyes,
                                 data_dir=DATA, out_dir=OUT)

Each run asks the GeotechStaffEngineer page one request in a fresh
conversation, through the review suite's own runner
(``aec_agent_core.review_eval.runner.run_task``, the page ``geotech``), with
the arm's switches set exactly (:data:`ARM_ENV`): the figure helper and the
tool figures in both, plot code OFF in the baseline and ON in the other.
Resumable: a finished run is never asked again (a failed one is), a check is
redone only when its run was. ``results.json``, ``RESULTS.md`` and
``side_by_side.html`` are rewritten after every run and every check.

Output, in ``out_dir``::

    runs/<request id>/<arm>/run.json    the request, the reply, the figures,
                                        the tool calls, tokens and time
    runs/<request id>/<arm>/agent/      the page run's record (activity.jsonl,
                                        files/ with every file it wrote)
    runs/<request id>/<arm>/check.json  the vision marks
    results.json, RESULTS.md, side_by_side.html

``python -m funhouse_agent.figure_eval.runner --dry-run --out DIR`` runs the
whole chain offline with scripted fake models ($0): the data folder, the
real page agent drawing through ``plot_data``, fake marks, the report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from funhouse_agent.figure_eval import report as _report
from funhouse_agent.figure_eval import request_set as _requests
from funhouse_agent.figure_eval.checklist import check_run
from funhouse_agent.figure_eval.compare import compare as compare_arms

#: The arms: the figure helper on in both, model-written plot code OFF in
#: the baseline and ON in the other (BACKLOG GM-227 switches).
ARM_ENV: Dict[str, Dict[str, str]] = {
    "off": {"GEOTECH_FIGURE_HELPER": "1", "GEOTECH_PLOT_CODE": "0"},
    "on": {"GEOTECH_FIGURE_HELPER": "1", "GEOTECH_PLOT_CODE": "1"},
}
ARMS = ("off", "on")


def label(model: Any) -> Optional[str]:
    if model is None:
        return None
    for attr in ("model", "model_name", "model_id", "model_api_name"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(model).__name__


def run_agent(model: Any, request: Dict[str, Any], arm: str, run_dir: str,
              recursion_limit: Optional[int] = None) -> Dict[str, Any]:
    """One request on the geotech page under ``arm``, in ``run_dir``; the
    page runner's result (never raises)."""
    from funhouse_agent.review_eval import Task, run_task
    task = Task(id=request["id"], question=request["prompt"], documents=[],
                category="figure", doc_type="figure", checks=[],
                auto_checks=False, page="geotech")
    return run_task(task, model, arm=arm, arm_env=ARM_ENV[arm],
                    docs_dir=run_dir, run_dir=run_dir,
                    recursion_limit=recursion_limit)


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _save(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _record(request: Dict[str, Any], arm: str, res: Dict[str, Any],
            agent_dir: str, out_dir: str) -> Dict[str, Any]:
    """What the hand-back keeps of a run; figure paths relative to
    ``out_dir``."""
    files = [os.path.join(agent_dir, f) for f in res.get("files") or []]
    rel = [os.path.relpath(p, out_dir) for p in files]
    return {
        "request": request["id"], "arm": arm, "env": ARM_ENV[arm],
        "prompt": request["prompt"], "started": res.get("started"),
        "answer": res.get("answer") or "",
        "error": res.get("error") or res.get("outcome_error"),
        "figures": [p for p in rel if p.lower().endswith(".png")],
        "files": rel, "tool_calls": res.get("tool_calls") or [],
        "tool_counts": res.get("tool_counts") or {},
        "model_calls": res.get("model_calls", 0),
        "tokens_in": res.get("tokens_in", 0),
        "tokens_out": res.get("tokens_out", 0),
        "seconds": res.get("seconds"), "versions": res.get("versions"),
        "commits": res.get("commits"), "finished": True}


def _totals(requests, runs, checks, arms) -> Dict[str, Dict[str, Any]]:
    out = {}
    for a in arms:
        t = {"runs": 0, "tokens_in": 0, "tokens_out": 0, "vision_in": 0,
             "vision_out": 0, "seconds": 0.0}
        for req in requests:
            run = runs.get(req["id"], {}).get(a)
            if run:
                t["runs"] += 1
                t["tokens_in"] += int(run.get("tokens_in") or 0)
                t["tokens_out"] += int(run.get("tokens_out") or 0)
                t["seconds"] = round(t["seconds"] +
                                     float(run.get("seconds") or 0), 1)
            chk = checks.get(req["id"], {}).get(a)
            usage = (chk or {}).get("usage") or {}
            t["vision_in"] += int(usage.get("input_tokens") or 0)
            t["vision_out"] += int(usage.get("output_tokens") or 0)
        out[a] = t
    return out


def score_figure_set(model: Any = None, *, vision_model: Any = None,
                     data_dir: Any, out_dir: Any,
                     ids: Optional[Iterable[str]] = None,
                     arms: Sequence[str] = ARMS, detail: Optional[str] = "auto",
                     recursion_limit: Optional[int] = None,
                     run_fn: Callable[..., Dict[str, Any]] = run_agent,
                     check_fn: Callable[..., Dict[str, Any]] = check_run,
                     say: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run, check, compare and report (see the module's text). Without
    ``model`` no request is asked (saved runs are checked and reported);
    without ``vision_model`` nothing is checked. Returns the results."""
    out = os.path.abspath(str(out_dir))
    os.makedirs(out, exist_ok=True)
    requests = _requests.select(_requests.load_data(data_dir), ids)
    runs: Dict[str, Dict[str, Any]] = {}
    checks: Dict[str, Dict[str, Any]] = {}

    def write() -> Dict[str, Any]:
        res = {"model": label(model), "vision_model": label(vision_model),
               "detail": detail, "arms": list(arms),
               "arm_env": {a: ARM_ENV[a] for a in arms},
               "requests": [{k: v for k, v in r.items()
                             if k not in ("sample_row",)} for r in requests],
               "runs": runs, "checks": checks,
               "comparison": compare_arms(requests, checks, *ARMS),
               "totals": _totals(requests, runs, checks, arms),
               "written": datetime.now().isoformat(timespec="seconds")}
        _report.write_report(out, res)
        return res

    for req in requests:
        for arm in arms:
            arm_dir = os.path.join(out, "runs", req["id"], arm)
            run_path = os.path.join(arm_dir, "run.json")
            check_path = os.path.join(arm_dir, "check.json")
            run = _load(run_path)
            asked = False
            if model is not None and not (run and run.get("finished")
                                          and not run.get("error")):
                asked = True
                agent_dir = os.path.join(arm_dir, "agent")
                say(f"run   {req['id']} {arm}")
                t0 = time.monotonic()
                try:
                    res = run_fn(model, req, arm, agent_dir,
                                 recursion_limit=recursion_limit)
                except Exception as exc:  # noqa: BLE001 - recorded
                    res = {"error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-2000:],
                           "seconds": round(time.monotonic() - t0, 1)}
                run = _record(req, arm, res, agent_dir, out)
                _save(run_path, run)
            if run:
                runs.setdefault(req["id"], {})[arm] = run
            check = _load(check_path)
            if check and run and (asked or check.get("run_started") !=
                                  run.get("started")):
                check = None
            if run and vision_model is not None and not check:
                say(f"check {req['id']} {arm}")
                paths = [os.path.join(out, p) for p in run["figures"]]
                try:
                    check = check_fn(vision_model, req, run, paths,
                                     detail=detail)
                except Exception as exc:  # noqa: BLE001 - retried next time
                    say(f"check failed {req['id']} {arm}: {exc}")
                    check = None
                if check:
                    check["run_started"] = run.get("started")
                    _save(check_path, check)
            if check:
                checks.setdefault(req["id"], {})[arm] = check
            write()
    return write()


# ---------------------------------------------------------------------------
# The dry run: scripted fake models, $0
# ---------------------------------------------------------------------------

DRY_IDS = ("R01", "R02", "R13", "R17")


def _dry_agent():
    """A chat model that plays the page's main agent: one ``plot_data``
    call through ``call_agent``, then a reply naming the arm."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class DryAgent(BaseChatModel):
        model_name: str = "dry-run agent (scripted fake)"

        @property
        def _llm_type(self):
            return "dry-run"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            step = sum(1 for m in messages if isinstance(m, AIMessage))
            arm = "on" if os.environ.get("GEOTECH_PLOT_CODE") == "1" \
                else "off"
            if step == 0 and any(isinstance(m, HumanMessage)
                                 for m in messages):
                msg = AIMessage(content="", tool_calls=[{
                    "name": "call_agent", "id": "dry1", "args": {
                        "agent_name": "profile_figure",
                        "method": "plot_data", "parameters": {
                            "series": [{"x": [0, 1, 2, 3],
                                        "y": [0, 40, 70, 90],
                                        "label": "made-up curve"}],
                            "xlabel": "Width (m)",
                            "ylabel": "Resistance (kPa)",
                            "title": "Dry run figure",
                            "output_path": f"dry_{arm}.png"}}}])
            else:
                msg = AIMessage(content=f"[dry run, plot code {arm}] "
                                f"Drew dry_{arm}.png.")
            msg.usage_metadata = {"input_tokens": 1200, "output_tokens": 80,
                                  "total_tokens": 1280}
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return DryAgent()


class DryEyes:
    """A vision model that marks every item yes, except, for a reply
    naming plot code ON, ``data_ink`` no and ``plan_drawing`` yes where
    OFF has it n/a: the report then shows a worse item and a one-arm n/a."""

    model_name = "gpt-dry-run-eyes (scripted fake)"

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def invoke(self, messages):
        content = messages[0].content
        text = content[0]["text"]
        self.calls.append(content)
        on = "plot code on]" in text
        ids = [line[2:].split(":", 1)[0] for line in text.splitlines()
               if line.startswith("- ")]
        marks = []
        for iid in ids:
            mark = "yes"
            if iid == "plan_drawing":
                mark = "yes" if on else "na"
            if iid == "data_ink" and on:
                mark = "no"
            marks.append({"id": iid, "mark": mark, "reason": "dry run"})

        class Reply:
            pass
        r = Reply()
        r.content = json.dumps({"marks": marks, "note": "dry run"})
        r.usage_metadata = {"input_tokens": 1000, "output_tokens": 100}
        return r


def dry_run(out_dir: Any, ids: Sequence[str] = DRY_IDS,
            references: bool = True, say: Callable[[str], None] = print
            ) -> Dict[str, Any]:
    """The whole chain offline: the data folder (with GM-328's references
    when ``references``), the real page agent on a scripted model, fake
    marks, the report, in ``out_dir``."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("GEOTECH_NO_DISCLAIMER", "1")
    data = os.path.join(str(out_dir), "data")
    _requests.prepare(data, ids, references=references, say=say)
    return score_figure_set(_dry_agent(), vision_model=DryEyes(),
                            data_dir=data,
                            out_dir=os.path.join(str(out_dir), "results"),
                            say=say)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", help="the data folder (request_set.py --out)")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline, scripted fake models")
    ap.add_argument("--no-references", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        res = dry_run(a.out, a.ids or DRY_IDS,
                      references=not a.no_references)
    elif a.data:
        res = score_figure_set(data_dir=a.data, out_dir=a.out, ids=a.ids)
    else:
        ap.error("--dry-run, or --data to rewrite the report of saved runs")
        return 2
    print(json.dumps(res["comparison"].get("requests", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
