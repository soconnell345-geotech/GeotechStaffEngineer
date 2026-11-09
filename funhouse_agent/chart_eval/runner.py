"""Run the chart suite: every task under every arm, graded, written down.

The notebook entry point is :func:`score_chart_suite`. It mirrors the review
suite's (:func:`funhouse_agent.review_eval.score_review_suite`): one folder
per run with its whole record, resumable (a finished run is never asked
again; a failed one is retried), ``RESULTS.md`` and ``results.json``
rewritten after every run, and an optional durable copy of the output.

Arms (see :mod:`funhouse_agent.chart_eval.arms`): ``none``, ``zoom``,
``zoom_code`` call the model directly; ``page:<arm>`` asks the app's
Document Review page with the switches of
:data:`funhouse_agent.review_flags.ARMS` ``<arm>`` (``page:baseline`` is
what users get today).

Concurrency. Direct runs share nothing, so they run ``workers`` at a time on
threads. A page run sets process-wide switches (the review flags, the working
folder), so page runs go one at a time, or, with ``page_workers`` above 1 on
Linux, each in a forked child process (what the AI FDE's glue did for the
review suite on Foundry), killed after ``page_timeout_s``. A forked child
inherits the parent's open HTTP connections (the direct runs' and the
judge's), so before its run it drops them and opens its own
(:func:`model_for_child`): no two processes write to one connection.

What the results files name. ``RESULTS.md`` names tasks by id and domain
only, never a question, a golden answer or a chart: Chartography is gated
and not for training, and results travel. ``run.json`` keeps the model's
answer and the judge's reasoning for the trace read; keep the output folder
private.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from funhouse_agent.chart_eval import imaging
from funhouse_agent.chart_eval.arms import DIRECT_ARMS, run_direct
from funhouse_agent.chart_eval.data import (ChartTask, judge_prompt_template,
                                            judge_prompt_templates,
                                            load_chart_tasks)
from funhouse_agent.chart_eval.judge import judge_answer

PAGE_PREFIX = "page:"
DEFAULT_ARMS = ("none", "zoom", "zoom_code", "page:baseline", "page:minimal")

#: Tool names counted as a look at the chart, and as running code.
LOOK_TOOLS = ("zoom", "render_region", "analyze_image", "analyze_pdf_page",
              "page_looker", "find_like")
CODE_TOOLS = ("run_python", "plot_code")


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def resolve_arm(name: str):
    """``(kind, name, env)`` for an arm name: ``("direct", "zoom", {})`` or
    ``("page", "page:baseline", {...switches...})``."""
    name = str(name).strip()
    if name in DIRECT_ARMS:
        return "direct", name, {}
    if name == "page":
        name = PAGE_PREFIX + "baseline"
    if name.startswith(PAGE_PREFIX):
        from funhouse_agent import review_flags
        sub = name[len(PAGE_PREFIX):]
        if sub not in review_flags.ARMS:
            raise ValueError(f"unknown page arm {sub!r}; known: "
                             f"{sorted(review_flags.ARMS)}")
        return "page", name, dict(review_flags.ARMS[sub])
    raise ValueError(f"unknown arm {name!r}: use one of {DIRECT_ARMS} or "
                     f"'{PAGE_PREFIX}<review arm>' (e.g. page:baseline)")


def arm_dir(name: str) -> str:
    return name.replace(":", "_")


def run_page(model: Any, task: ChartTask, run_dir: str, env: Dict[str, str],
             arm: str, orientation: bool = False,
             recursion_limit: Optional[int] = None) -> Dict[str, Any]:
    """The chart uploaded to the Document Review page and the question asked
    there, through the review suite's own runner."""
    from funhouse_agent.review_eval.runner import run_task
    from funhouse_agent.review_eval.tasks import Task
    rt = Task(id=task.id, question=task.prompt,
              documents=[os.path.basename(task.image_path)],
              category="extract", doc_type="chart", checks=[],
              auto_checks=False, page="review")
    res = run_task(rt, model, arm=arm, arm_env=env,
                   docs_dir=os.path.dirname(task.image_path),
                   run_dir=os.path.join(run_dir, "page"),
                   orientation=orientation, recursion_limit=recursion_limit)
    return {"answer": res.get("answer") or "",
            "error": res.get("error"),
            "outcome_error": res.get("outcome_error"),
            "stop": ("outcome_error" if res.get("outcome_error") else
                     "error" if res.get("error") else "answered"),
            "model_calls": res.get("model_calls", 0),
            "usage": {"input_tokens": res.get("tokens_in", 0),
                      "output_tokens": res.get("tokens_out", 0)},
            "tool_counts": res.get("tool_counts") or {},
            "page_seconds": res.get("seconds"),
            "files": res.get("files") or [],
            "fingerprint": res.get("fingerprint"),
            "vision_profile": res.get("vision_profile"),
            "page_record": "page/ (activity.jsonl, messages)"}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _run_one(kind: str, arm: str, env: Dict[str, str], model: Any,
             task: ChartTask, run_dir: str, opts: Dict[str, Any]
             ) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    res: Dict[str, Any] = {
        "task": task.id, "arm": arm, "kind": kind, "domain": task.domain,
        "started": datetime.now().isoformat(timespec="seconds"),
        "error": None}
    t0 = time.monotonic()
    try:
        if kind == "direct":
            res.update(run_direct(arm, model, task.prompt, task.image_path,
                                  run_dir, limits=opts.get("image_limits"),
                                  max_steps=opts.get("max_steps", 20),
                                  sleep=opts.get("sleep", time.sleep)))
        else:
            res.update(run_page(model, task, run_dir, env, arm,
                                orientation=opts.get("orientation", False),
                                recursion_limit=opts.get("recursion_limit")))
    except Exception as exc:                              # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-3000:]
    res["seconds"] = round(time.monotonic() - t0, 1)
    return res


def grading_prompt(task: ChartTask, templates: Dict[str, str],
                   template: Optional[str]):
    """``(template, label)`` that grades ``task``: its kind's prompt in
    ``judge_prompts.json``, else ``judge_prompt.txt``, else the suite's
    own (``None``)."""
    if templates and task.judge_key in templates:
        return templates[task.judge_key], f"data folder ({task.judge_key})"
    if template:
        return template, "data folder"
    return None, "own"


def _judge(res: Dict[str, Any], task: ChartTask, judge_model: Any,
           templates: Dict[str, str], template: Optional[str]
           ) -> Dict[str, Any]:
    if res.get("error"):
        res["judge"] = {"passed": None, "error": "the run failed; not "
                                                 "judged (retried next run)"}
    else:
        tpl, label = grading_prompt(task, templates, template)
        res["judge"] = judge_answer(judge_model,
                                    task.judge_question or task.prompt,
                                    task.golden_answer,
                                    res.get("answer") or "", tpl,
                                    label=label)
    res["correct"] = res["judge"].get("passed")
    return res


def _save(run_dir: str, res: Dict[str, Any]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    tmp = os.path.join(run_dir, "run.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(tmp, os.path.join(run_dir, "run.json"))


def _load(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "run.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Page runs in forked children
# ---------------------------------------------------------------------------

#: Process-wide HTTP client caches: one client (one connection pool) per
#: address, shared by every model of a family, the model under test and the
#: parent's judge alike. A forked child must not reuse them.
_CLIENT_CACHES = (
    ("langchain_anthropic._client_utils",
     ("_get_default_httpx_client", "_get_default_async_httpx_client")),
    ("langchain_openai.chat_models._client_utils",
     ("_cached_sync_httpx_client", "_cached_async_httpx_client")),
)
#: Clients a chat model builds on first use and keeps (ChatAnthropic's
#: cached properties, the app's Foundry subclass included).
_LAZY_CLIENTS = ("_client", "_async_client")


def drop_inherited_clients() -> List[str]:
    """Forget the process-wide HTTP clients this process inherited
    (:data:`_CLIENT_CACHES` and the app's CA-bundle clients), so a model
    built or first used here opens connections of its own; what was
    dropped, for the run's record. Nothing is closed: the parent still uses
    those connections."""
    dropped: List[str] = []
    for mod_name, names in _CLIENT_CACHES:
        mod = sys.modules.get(mod_name)
        for name in names:
            clear = getattr(getattr(mod, name, None), "cache_clear", None)
            if clear is not None:
                clear()
                dropped.append(f"{mod_name}.{name}")
    engine = sys.modules.get("webapp.engine_config")
    ca_clients = getattr(engine, "_CA_HTTP_CLIENTS", None)
    if isinstance(ca_clients, dict) and ca_clients:
        ca_clients.clear()
        dropped.append("webapp.engine_config._CA_HTTP_CLIENTS")
    return dropped


def model_for_child(model: Any):
    """``(model, record)``: the model a forked page run calls, with
    connections of its own. The inherited process-wide clients are dropped
    first (:func:`drop_inherited_clients`). A model
    :func:`funhouse_agent.chart_eval.foundry_model` built is then built
    again from its id (new clients, new SDK handles); any other model goes
    on with the clients it built on first use dropped, but keeps any client
    or handle it was given when it was made."""
    dropped = drop_inherited_clients()
    try:
        from funhouse_agent.chart_eval import rebuild_model
        fresh = rebuild_model(model)
    except Exception as exc:                              # noqa: BLE001
        dropped.append(f"rebuild failed ({type(exc).__name__}: {exc})")
        fresh = None
    if fresh is not None:
        return fresh, {"rebuilt": True, "dropped": dropped}
    held = getattr(model, "__dict__", None)
    if isinstance(held, dict):
        for attr in _LAZY_CLIENTS:
            if held.pop(attr, None) is not None:
                dropped.append(f"{type(model).__name__}.{attr}")
    return model, {"rebuilt": False, "dropped": dropped}


def _fork_runs(jobs: List[Dict[str, Any]], n: int, timeout_s: float,
               on_done: Callable[[Dict[str, Any], Dict[str, Any]], None],
               say: Callable[..., None]) -> None:
    """Each job in a forked child (at most ``n`` at once), which opens its
    own connections (:func:`model_for_child`) and writes its run.json;
    ``on_done(job, result)`` in the parent as each ends."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")

    def child(job):
        try:
            model, clients = model_for_child(job["model"])
            res = _run_one(job["kind"], job["arm"], job["env"], model,
                           job["task"], job["run_dir"], job["opts"])
            res["child_clients"] = dict(clients, pid=os.getpid())
            _save(job["run_dir"], res)
        finally:
            os._exit(0)

    pending = list(jobs)
    running: Dict[Any, Dict[str, Any]] = {}
    while pending or running:
        while pending and len(running) < n:
            job = pending.pop(0)
            proc = ctx.Process(target=child, args=(job,), daemon=False)
            proc.start()
            running[proc] = dict(job, started=time.monotonic())
        time.sleep(0.5)
        for proc in list(running):
            job = running[proc]
            if proc.is_alive():
                if time.monotonic() - job["started"] > timeout_s:
                    proc.kill()
                    proc.join(5)
                    res = {"task": job["task"].id, "arm": job["arm"],
                           "kind": "page", "domain": job["task"].domain,
                           "error": f"stopped after {timeout_s:.0f} s "
                                    "(page_timeout_s)",
                           "seconds": round(timeout_s, 1)}
                    del running[proc]
                    on_done(job, res)
                continue
            proc.join()
            del running[proc]
            res = _load(job["run_dir"])
            if res is None or res.get("task") != job["task"].id:
                res = {"task": job["task"].id, "arm": job["arm"],
                       "kind": "page", "domain": job["task"].domain,
                       "error": f"the child process ended (exit code "
                                f"{proc.exitcode}) without a result"}
            on_done(job, res)


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _tok(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e5 else f"{n / 1e3:.1f}k"


def _cell(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "·"
    if len(results) == 1:
        r = results[0]
        if r.get("error"):
            return "E"
        c = r.get("correct")
        return "✓" if c is True else "✗" if c is False else "?"
    ok = sum(1 for r in results if r.get("correct") is True)
    return f"{ok}/{len(results)}"


def summarize(runs: Dict[str, Dict[str, Dict[str, Any]]],
              arms: Sequence[str], tasks: Sequence[ChartTask],
              meta: Dict[str, Any]) -> str:
    """``RESULTS.md``: ids and labels only."""
    lines = [f"# {meta.get('dataset') or 'Chart suite'}: chart reading, "
             "arm by arm", ""]
    lines.append(
        f"Run {meta.get('updated', '')} · model `{meta.get('model')}` "
        f"({meta.get('family')}, image limits {meta.get('limits')}) · judge "
        f"`{meta.get('judge')}` with the {meta.get('judge_prompt')} prompt · "
        f"{len(tasks)} tasks × {meta.get('repeats', 1)} repeat(s)")
    lines += ["", "| arm | correct | judged | accuracy | unscored | failed "
              "runs | model calls / run | looks / run | code runs / run | "
              "tokens in / out | minutes |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    by_task: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    tool_totals: Dict[str, Dict[str, int]] = {}
    for arm in arms:
        rs = list((runs.get(arm) or {}).values())
        for key, r in (runs.get(arm) or {}).items():
            by_task.setdefault(str(r.get("task") or key.split("__r")[0]),
                               {}).setdefault(arm, []).append(r)
        done = [r for r in rs if not r.get("error")]
        judged = [r for r in done if r.get("correct") in (True, False)]
        correct = sum(1 for r in judged if r.get("correct") is True)
        unscored = len(done) - len(judged)
        failed = len(rs) - len(done)
        acc = f"{100.0 * correct / len(judged):.1f} %" if judged else "–"
        tt = tool_totals.setdefault(arm, {})
        for r in done:
            for k, v in (r.get("tool_counts") or {}).items():
                tt[k] = tt.get(k, 0) + int(v or 0)
        looks = _mean([sum(int(v or 0) for k, v in
                           (r.get("tool_counts") or {}).items()
                           if k in LOOK_TOOLS) for r in done])
        code = _mean([sum(int(v or 0) for k, v in
                          (r.get("tool_counts") or {}).items()
                          if k in CODE_TOOLS) for r in done])
        calls = _mean([float(r.get("model_calls") or 0) for r in done])
        t_in = sum(float((r.get("usage") or {}).get("input_tokens") or 0)
                   for r in done)
        t_out = sum(float((r.get("usage") or {}).get("output_tokens") or 0)
                    for r in done)
        mins = sum(float(r.get("seconds") or 0) for r in rs) / 60.0
        lines.append(f"| {arm} | {correct} | {len(judged)} | {acc} | "
                     f"{unscored} | {failed} | {calls:.1f} | {looks:.1f} | "
                     f"{code:.1f} | {_tok(t_in)} / {_tok(t_out)} | "
                     f"{mins:.1f} |")
    lines += ["", "Minutes add up each run's own time; runs that ran at "
              "once overlap. Tokens are as the model reported them (input "
              "includes cache reads where the route counts them so)."]

    domains: Dict[str, List[ChartTask]] = {}
    for t in tasks:
        domains.setdefault(t.domain or "(none)", []).append(t)
    if len(domains) > 1:
        lines += ["", "## By domain", "",
                  "| domain | tasks | " + " | ".join(arms) + " |",
                  "|---|---|" + "---|" * len(arms)]
        for dom in sorted(domains):
            ids = {t.id for t in domains[dom]}
            cells = []
            for arm in arms:
                rs = [r for tid in ids for r in by_task.get(tid, {}).get(
                    arm, []) if r.get("correct") in (True, False)]
                ok = sum(1 for r in rs if r.get("correct") is True)
                cells.append(f"{ok}/{len(rs)}" if rs else "·")
            lines.append(f"| {dom} | {len(ids)} | " + " | ".join(cells)
                         + " |")

    lines += ["", "## Tools used (totals over the arm's finished runs)", ""]
    for arm in arms:
        tt = tool_totals.get(arm) or {}
        listed = ", ".join(f"{k} {v}" for k, v in
                           sorted(tt.items(), key=lambda kv: -kv[1])[:12])
        lines.append(f"- `{arm}`: {listed or 'none'}")

    lines += ["", "## Per task", "",
              "✓ correct, ✗ wrong, ? not graded (judge gave no verdict), "
              "E the run failed, · not run; with repeats, correct/runs.", "",
              "| task | domain | " + " | ".join(arms) + " |",
              "|---|---|" + "---|" * len(arms)]
    for t in tasks:
        cells = [_cell(by_task.get(t.id, {}).get(a, [])) for a in arms]
        lines.append(f"| {t.id} | {t.domain} | " + " | ".join(cells) + " |")

    if len(arms) > 1:
        base = arms[0]
        lines += ["", f"## Changes against `{base}`", ""]
        for arm in arms[1:]:
            fixed, broke = [], []
            for t in tasks:
                a = by_task.get(t.id, {}).get(base, [])
                b = by_task.get(t.id, {}).get(arm, [])
                if len(a) != 1 or len(b) != 1:
                    continue
                ca, cb = a[0].get("correct"), b[0].get("correct")
                if ca is False and cb is True:
                    fixed.append(t.id)
                elif ca is True and cb is False:
                    broke.append(t.id)
            lines.append(f"- `{arm}`: right where `{base}` was wrong "
                         f"{len(fixed)} ({', '.join(fixed[:30]) or '-'}); "
                         f"wrong where `{base}` was right {len(broke)} "
                         f"({', '.join(broke[:30]) or '-'})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The notebook entry point
# ---------------------------------------------------------------------------

def _label(model: Any, given: Optional[str]) -> str:
    if given:
        return str(given)
    for attr in ("model", "model_name", "model_id", "model_api_name"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(model).__name__


def _versions() -> Dict[str, Any]:
    try:
        from funhouse_agent.review_eval.runner import _commits, _versions as v
        return {"versions": v(), "commits": _commits()}
    except Exception as exc:                              # noqa: BLE001
        return {"versions_error": f"{type(exc).__name__}: {exc}"}


def score_chart_suite(model: Any = None, *, data_dir: Any,
                      out_dir: Any = "/tmp/chart_eval",
                      arms: Sequence[str] = DEFAULT_ARMS,
                      judge_model: Any = None,
                      ids: Optional[Iterable[str]] = None,
                      subset: Any = None,
                      limit: Optional[int] = None, repeats: int = 1,
                      workers: int = 4, page_workers: int = 1,
                      page_timeout_s: float = 1800.0,
                      redo: bool = False, rejudge: bool = False,
                      image_limits: Any = None, max_steps: int = 20,
                      orientation: bool = False,
                      recursion_limit: Optional[int] = None,
                      durable_dir: Any = None, sharepoint: Any = None,
                      sharepoint_folder: str = "GeotechStaffEngineer/chart_eval",
                      model_label: Optional[str] = None,
                      judge_label: Optional[str] = None,
                      dry_run: bool = False, verbose: bool = True,
                      sleep: Callable[[float], None] = time.sleep
                      ) -> Dict[str, Any]:
    """Run the chart suite; return ``{"results_md", "results", "out_dir"}``.

    ``model`` is a LangChain chat model (on Foundry:
    :func:`funhouse_agent.chart_eval.foundry_model`); ``judge_model`` grades
    the answers (default: ``model``). ``data_dir`` is the data folder
    (:mod:`funhouse_agent.chart_eval.data`; CharXiv's from
    :mod:`funhouse_agent.chart_eval.charxiv`). ``arms``: see the module
    docstring. ``ids``, ``subset`` (a file of ids, e.g.
    ``"sample_100.txt"`` in the data folder) and ``limit`` narrow the
    tasks; ``repeats`` asks each
    task that many times per arm. ``redo`` runs everything again;
    ``rejudge`` grades saved answers again (no run is repeated: after the
    judge or its prompt changes). ``image_limits`` overrides the image size
    the direct arms size for (``"claude-hires"``, ``"openai-original"``, or
    ``(max_edge, max_tokens, patch)``). ``dry_run`` lists the tasks, the
    charts and the runs to do without calling a model.
    """
    say = print if verbose else (lambda *a, **k: None)
    tasks = load_chart_tasks(data_dir, ids=ids, limit=limit, subset=subset)
    template = judge_prompt_template(data_dir)
    templates = judge_prompt_templates(data_dir)
    judge_source = ("per task (judge_prompts.json)" if templates else
                    "judge_prompt.txt" if template else "own")
    specs = [resolve_arm(a) for a in arms]
    arm_names = [name for _, name, _ in specs]
    if len(set(arm_names)) != len(arm_names):
        raise ValueError(f"an arm is named twice: {arm_names}")
    out = Path(os.fspath(out_dir)).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    repeats = max(1, int(repeats))

    def keys_for(t: ChartTask) -> List[str]:
        return [t.id if r == 0 else f"{t.id}__r{r + 1}"
                for r in range(repeats)]

    if dry_run:
        rows = [f"- {t.id} [{t.domain or '-'}]: "
                f"{os.path.basename(t.image_path)} "
                f"{_px(t.image_path)}" for t in tasks]
        todo = sum(1 for _, name, _ in specs for t in tasks
                   for k in keys_for(t)
                   if not _finished(_load(str(out / "runs" / arm_dir(name)
                                              / k))))
        lim = imaging.limits_for(model, image_limits) if model is not None \
            else None
        text = ("# Dry run\n\n"
                f"{len(tasks)} tasks × {len(specs)} arms × {repeats} "
                f"repeat(s); {todo} runs still to do\n\n"
                f"arms: {', '.join(arm_names)}\n\n"
                f"judge prompt: {judge_source}\n\n"
                + (f"model family {imaging.family_of(model)}, image limits "
                   f"{lim.to_dict()}\n\n" if lim else "")
                + "\n".join(rows) + "\n")
        say(text)
        return {"results_md": text, "results": {}, "out_dir": str(out)}

    if model is None:
        raise ValueError("pass model= (a LangChain chat model; on Foundry "
                         "funhouse_agent.chart_eval.foundry_model(<id>))")
    if judge_model is None:
        judge_model = model
        say("judge: the run's own model (pass judge_model= for another)")

    mirror = None
    if sharepoint is not None or durable_dir is not None:
        try:
            from funhouse_agent.run_mirror import Mirror
            mirror = Mirror(sharepoint=sharepoint, durable_dir=durable_dir,
                            base_folder=sharepoint_folder)
            remote = mirror.remote_for(out.name)
            restored = mirror.restore_dir(remote, out)
            say(f"mirror: {mirror.describe(remote)}; restored {restored}")
        except Exception as exc:                          # noqa: BLE001
            say(f"mirror unavailable: {type(exc).__name__}: {exc}")
            mirror = None

    lim = imaging.limits_for(model, image_limits)
    meta: Dict[str, Any] = {
        "model": _label(model, model_label),
        "family": imaging.family_of(model), "limits": lim.name,
        "limits_detail": lim.to_dict(),
        "judge": _label(judge_model, judge_label),
        "judge_prompt": judge_source,
        "arms": arm_names, "repeats": repeats, "max_steps": max_steps,
        "orientation": orientation,
        "started": datetime.now().isoformat(timespec="seconds")}
    meta.update(_versions())
    prov = Path(os.fspath(data_dir)).expanduser() / "PROVENANCE.json"
    if prov.is_file():
        try:
            meta["data"] = json.loads(prov.read_text(encoding="utf-8"))
            meta["dataset"] = (meta["data"].get("dataset")
                               or meta["data"].get("repo"))
        except ValueError:
            pass
    opts = {"image_limits": image_limits, "max_steps": max_steps,
            "orientation": orientation, "recursion_limit": recursion_limit,
            "sleep": sleep}

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {n: {} for n in arm_names}
    lock = threading.Lock()
    warned: set = set()
    by_id = {t.id: t for t in tasks}

    def write_results() -> str:
        with lock:
            meta["updated"] = datetime.now().isoformat(timespec="seconds")
            md = summarize(runs, arm_names, tasks, meta)
            (out / "RESULTS.md").write_text(md, encoding="utf-8")
            (out / "results.json").write_text(json.dumps(
                {"meta": meta, "runs": runs}, indent=1, default=str),
                encoding="utf-8")
        if mirror is not None:
            try:
                got = mirror.mirror_dir(out, mirror.remote_for(out.name))
                for err in (got or {}).get("errors") or []:
                    head = str(err).split(":", 1)[-1].strip()[:160]
                    if head not in warned:
                        warned.add(head)
                        say(f"mirror: a file was NOT copied ({err})")
            except Exception as exc:                      # noqa: BLE001
                say(f"mirror failed: {type(exc).__name__}: {exc}")
        return md

    def task_of(res: Dict[str, Any], key: str) -> ChartTask:
        return by_id[str(res.get("task") or key.split("__r")[0])]

    def record(arm: str, key: str, run_dir: str, res: Dict[str, Any],
               judge_now: bool = True) -> None:
        if judge_now:
            _judge(res, task_of(res, key), judge_model, templates, template)
        res["key"] = key
        _save(run_dir, res)
        with lock:
            runs[arm][key] = res
        mark = ("E " + str(res.get("error"))[:100] if res.get("error") else
                {True: "correct", False: "wrong"}.get(res.get("correct"),
                                                      "not graded"))
        say(f"[{arm}] {key}: {mark} ({res.get('seconds', 0)} s, "
            f"{res.get('model_calls', 0)} model calls)")
        write_results()

    direct_jobs: List[Dict[str, Any]] = []
    page_jobs: List[Dict[str, Any]] = []
    judge_jobs: List[Dict[str, Any]] = []
    for kind, name, env in specs:
        for t in tasks:
            for key in keys_for(t):
                run_dir = str(out / "runs" / arm_dir(name) / key)
                prev = None if redo else _load(run_dir)
                if prev and _finished(prev):
                    prev["key"] = key
                    runs[name][key] = prev
                    if rejudge or (prev.get("judge") or {}).get(
                            "passed") is None:
                        judge_jobs.append({"arm": name, "key": key,
                                           "run_dir": run_dir, "res": prev})
                    continue
                if prev and prev.get("error"):
                    say(f"[{name}] {key}: previous attempt failed "
                        f"({str(prev.get('error'))[:100]}); retrying")
                job = {"kind": kind, "arm": name, "env": env, "task": t,
                       "key": key, "run_dir": run_dir, "model": model,
                       "opts": opts}
                (direct_jobs if kind == "direct" else page_jobs).append(job)
    say(f"{len(tasks)} tasks × {len(specs)} arms × {repeats}: "
        f"{len(direct_jobs)} direct and {len(page_jobs)} page runs to do, "
        f"{len(judge_jobs)} saved answers to grade")

    def run_and_judge(j: Dict[str, Any]) -> Dict[str, Any]:
        res = _run_one(j["kind"], j["arm"], j["env"], model, j["task"],
                       j["run_dir"], opts)
        return _judge(res, j["task"], judge_model, templates, template)

    def judge_saved(j: Dict[str, Any]) -> Dict[str, Any]:
        return _judge(j["res"], task_of(j["res"], j["key"]), judge_model,
                      templates, template)

    # Grading saved answers and the direct runs share the thread pool; each
    # worker grades its own run, so the judge's time overlaps the runs'.
    if judge_jobs or direct_jobs:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(judge_saved, j): j for j in judge_jobs}
            futures.update({pool.submit(run_and_judge, j): j
                            for j in direct_jobs})
            for fut in as_completed(futures):
                j = futures[fut]
                record(j["arm"], j["key"], j["run_dir"], fut.result(),
                       judge_now=False)

    if page_jobs:
        if int(page_workers) > 1 and os.name == "posix":
            _fork_runs(page_jobs, int(page_workers), float(page_timeout_s),
                       lambda j, res: record(j["arm"], j["key"], j["run_dir"],
                                             res), say)
        else:
            for j in page_jobs:
                say(f"[{j['arm']}] {j['key']} ...")
                res = _run_one(j["kind"], j["arm"], j["env"], model,
                               j["task"], j["run_dir"], opts)
                record(j["arm"], j["key"], j["run_dir"], res)

    md = write_results()
    say(md)
    return {"results_md": md, "results": runs, "out_dir": str(out)}


def _finished(res: Optional[Dict[str, Any]]) -> bool:
    return bool(res) and not res.get("error")


def _px(path: str) -> str:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return f"{im.size[0]}x{im.size[1]} px"
    except Exception:                                     # noqa: BLE001
        return "(not an image PIL can open)"


__all__ = ["score_chart_suite", "resolve_arm", "run_page", "summarize",
           "DEFAULT_ARMS"]
