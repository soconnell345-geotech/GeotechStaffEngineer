"""Run the Document Review suite: every task, under every arm, scored.

The owner's notebook cell (Funhouse)::

    from funhouse_agent.review_eval import score_review_suite
    res = score_review_suite(
        prompter=fh_prompter, model_name="funhouse-gpt-high",
        docs_dir="/Volumes/.../review_eval_docs",      # the public PDFs
        out_dir="/tmp/review_eval",
        arms=("baseline", "lean"),                      # see review_flags.ARMS
        sharepoint=fh_sp_client,                        # durable copy
    )
    print(res["results_md"])

Each (arm, task) is a FRESH conversation built exactly the way the Document
Review page builds one (``webapp.core.build_agent`` with the page's profile),
the document staged as an upload with the page's own attachment note, the
question sent as the user's message, and the turn streamed through the page's
own ``core.stream_turn``. So what is scored is what a tester would have got.
An arm is only a set of switches (:mod:`funhouse_agent.review_flags`), set for
the task and unset afterwards.

``orientation=True`` sends the page's automatic orientation turn first, with
the text the app sends (``webapp.profiles.orientation_request_for``, under the
arm's switches). Run the ``overview`` arm (``GEOTECH_REVIEW_OVERVIEW``), and
any arm that changes only the orientation, with ``orientation=True``: without
it the arm is the same as its base arm.

Every run writes ``runs/<arm>/<task>/run.json`` (the answer, the checks, tool
calls, model calls, tokens, seconds) beside the conversation's own
``activity.jsonl`` and any file the agent produced, then rewrites
``RESULTS.md`` / ``results.json`` and mirrors the folder (SharePoint and/or a
durable folder) — the cluster's ``/tmp`` does not survive a restart, and a
restart RESUMES: a finished run is not paid for twice, a failed one is retried.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from uuid import uuid4

from funhouse_agent import review_flags
from funhouse_agent.review_eval import checks as _checks
from funhouse_agent.review_eval import documents as _documents
from funhouse_agent.review_eval.tasks import Task, load_tasks, select

#: Who the suite signs any markup as, so a produced PDF is recognisably a test.
SUITE_AUTHOR = "Review suite via GeotechStaffEngineer (AI draft)"


@contextmanager
def _working_folder(path: str) -> Iterator[None]:
    """Point the tools' working folder at ``path`` for the block (the app does
    the same per conversation)."""
    from funhouse_agent._fileio import DEFAULT_OUTPUT_DIR_ENV
    saved = os.environ.get(DEFAULT_OUTPUT_DIR_ENV)
    os.environ[DEFAULT_OUTPUT_DIR_ENV] = path
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(DEFAULT_OUTPUT_DIR_ENV, None)
        else:
            os.environ[DEFAULT_OUTPUT_DIR_ENV] = saved


def _model_from(prompter: Any, model_name: str):
    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    return PrompterChatModel(prompter=prompter, model=model_name)


def build_page_agent(model, attachments: Dict[str, bytes], files_dir: str,
                     artifacts: List[str]):
    """The Document Review page's agent, built the way ``webapp/app.py``
    builds it (behaviour defaults, then the page profile's overrides)."""
    from webapp import core
    from webapp.profiles import DOCUMENT_REVIEW
    kw = core.behavior_build_kwargs(None)
    kw.update(DOCUMENT_REVIEW.build_kwargs())
    kw.setdefault("markup_author", SUITE_AUTHOR)
    return core.build_agent(model, attachments, files_dir, artifacts, **kw)


def _activity(conv_dir: str) -> Dict[str, Any]:
    """Tool calls, model calls and tokens from the run's activity.jsonl."""
    path = os.path.join(conv_dir, "activity.jsonl")
    tool_calls: List[Dict[str, Any]] = []
    model_calls = 0
    model_errors = 0
    tokens_in = tokens_out = 0
    if not os.path.isfile(path):
        return {"tool_calls": [], "model_calls": 0, "model_errors": 0,
                "tokens_in": 0, "tokens_out": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ev = rec.get("event")
            if ev == "tool_start":
                args = rec.get("args")
                text = json.dumps(args, ensure_ascii=False, default=str) \
                    if not isinstance(args, str) else args
                tool_calls.append({"agent": rec.get("agent"),
                                   "name": rec.get("name"),
                                   "args": text[:300]})
            elif ev == "model_end":
                model_calls += 1
                usage = rec.get("usage") or {}
                for k, v in usage.items():
                    if not isinstance(v, (int, float)):
                        continue
                    k = str(k).lower()
                    if "input" in k or "prompt" in k:
                        if "detail" not in k and "cache" not in k:
                            tokens_in += int(v)
                    elif "output" in k or "completion" in k:
                        if "detail" not in k and "reason" not in k:
                            tokens_out += int(v)
            elif ev == "model_error":
                model_errors += 1
    return {"tool_calls": tool_calls, "model_calls": model_calls,
            "model_errors": model_errors, "tokens_in": tokens_in,
            "tokens_out": tokens_out}


def run_task(task: Task, model, *, arm: str, arm_env: Dict[str, str],
             docs_dir: Any, run_dir: str, orientation: bool = False,
             recursion_limit: Optional[int] = None) -> Dict[str, Any]:
    """One task under one arm, in a fresh conversation. Never raises: a
    failure is recorded on the result (and retried on the next run)."""
    from webapp import core
    from webapp.activity_log import ActivityLogger
    from webapp.profiles import DOCUMENT_REVIEW, orientation_request_for

    run_dir = os.path.abspath(run_dir)
    files_dir = os.path.join(run_dir, "files")
    # A retried run starts clean: a file the failed attempt wrote (a marked-up
    # PDF, a memo) must not satisfy a check of this one.
    _clear_run_dir(run_dir)
    os.makedirs(files_dir, exist_ok=True)
    result: Dict[str, Any] = {
        "task": task.id, "arm": arm, "arm_env": dict(arm_env),
        "category": task.category, "doc_type": task.doc_type,
        "split": task.split, "question": task.question,
        "started": datetime.now().isoformat(timespec="seconds"),
        "turns": [], "answer": "", "error": None, "outcome_error": None}
    t0 = time.monotonic()
    staged: List[str] = []
    try:
        docs = [_documents.resolve(d, docs_dir) for d in task.documents]
    except _documents.MissingDocument as exc:
        result.update(error=f"missing document: {exc}", skipped=True,
                      seconds=0.0)
        return result
    except Exception as exc:  # noqa: BLE001 - a broken document, not a crash
        result.update(error=f"document failed: {type(exc).__name__}: {exc}",
                      skipped=True, seconds=0.0)
        return result
    with review_flags.switches(arm_env), _working_folder(files_dir):
        try:
            attachments: Dict[str, bytes] = {}
            artifacts: List[str] = []
            atts = [core.stage_upload(attachments, files_dir, name, data)
                    for name, data in docs]
            staged = [a.path for a in atts]
            agent = build_page_agent(model, attachments, files_dir, artifacts)
            note = core.attachment_note(atts, review=True)
            # The page's own orientation text (the arm's switches are set).
            turns = ([orientation_request_for(DOCUMENT_REVIEW, atts)]
                     if orientation else [])
            turns += [task.question] + list(task.followups)
            history: List[Dict[str, str]] = []
            thread = uuid4().hex
            limit = recursion_limit or core.DEFAULT_BEHAVIOR["recursion_limit"]
            for i, text in enumerate(turns):
                content = core.assemble_user_message([note] if i == 0 else [],
                                                     text)
                history.append({"role": "user", "content": content})
                logger = ActivityLogger(run_dir, turn=i + 1)
                logger.turn_start(prompt=text)
                t_turn = time.monotonic()
                answer, tokens, err, kind = "", 0, None, None
                try:
                    for item in core.stream_turn(agent, history, thread,
                                                 recursion_limit=limit,
                                                 callbacks=[logger]):
                        if item.get("kind") == "turn_done":
                            answer = item.get("answer") or ""
                            tokens = int(item.get("turn_tokens") or 0)
                except Exception as exc:  # noqa: BLE001 - recorded
                    err = f"{type(exc).__name__}: {exc}"
                    kind = ("outcome" if type(exc).__name__ in OUTCOME_ERRORS
                            else "infra")
                logger.turn_end(turn_tokens=tokens, error=err,
                                answer_chars=len(answer))
                history.append({"role": "assistant", "content": answer})
                result["turns"].append({
                    "question": text, "answer": answer, "tokens": tokens,
                    "seconds": round(time.monotonic() - t_turn, 1),
                    "error": err})
                if err and kind == "outcome":
                    # What the page itself does (the legacy agent's step cap):
                    # a RESULT, scored as the empty answer it is, not retried.
                    result["outcome_error"] = err
                elif err:
                    result["error"] = err
            result["answer"] = result["turns"][-1]["answer"] if result["turns"] else ""
        except Exception as exc:  # noqa: BLE001 - a build/setup failure
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()[-2000:]
    result["seconds"] = round(time.monotonic() - t0, 1)
    staged_set = {os.path.abspath(p) for p in staged}
    produced = sorted(
        str(p) for p in Path(files_dir).rglob("*")
        if p.is_file() and os.path.abspath(str(p)) not in staged_set)
    result["files"] = [os.path.relpath(p, run_dir) for p in produced]
    act = _activity(run_dir)
    result.update(model_calls=act["model_calls"],
                  model_errors=act["model_errors"],
                  tokens_in=act["tokens_in"], tokens_out=act["tokens_out"],
                  tokens=sum(t.get("tokens", 0) for t in result["turns"]),
                  tool_calls=act["tool_calls"],
                  tool_counts=_counts(c["name"] for c in act["tool_calls"]))
    result["checks"] = [
        _checks.run_check(c, result["answer"], files=produced,
                          tool_calls=act["tool_calls"])
        for c in task.all_checks()]
    result["score"] = _checks.score(result["checks"])
    # The uploads are copies of the suite's documents: once scored they are
    # not worth mirroring once per arm and task.
    for p in staged:
        try:
            os.remove(p)
        except OSError:
            pass
    return result


#: Errors that ARE the page's behaviour (scored, never retried): the legacy
#: agent's step cap ends a long request this way.
OUTCOME_ERRORS = ("GraphRecursionError",)


def _clear_run_dir(run_dir: str) -> None:
    """Empty ``run_dir`` of everything but a previous ``run.json``."""
    import shutil
    if not os.path.isdir(run_dir):
        return
    for entry in os.listdir(run_dir):
        if entry == "run.json":
            continue
        path = os.path.join(run_dir, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError:
            pass


def _counts(names: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for n in names:
        out[n] = out.get(n, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# The results file
# ---------------------------------------------------------------------------

def _versions() -> Dict[str, str]:
    out = {}
    for dist in ("geotech-staff-engineer", "planlens", "deepagents",
                 "langchain"):
        try:
            from importlib.metadata import version
            out[dist] = version(dist)
        except Exception:
            out[dist] = "?"
    return out


def _mark(run: Optional[Dict[str, Any]]) -> str:
    if run is None:
        return "·"
    if run.get("skipped"):
        return "skip"
    sc = run.get("score") or {}
    tick = "✓" if sc.get("passed") else "✗"
    err = (" (error)" if run.get("error") else
           " (step cap)" if run.get("outcome_error") else "")
    return f"{tick} {sc.get('checks_passed', 0)}/{sc.get('checks_total', 0)}{err}"


def summarize(runs: Dict[str, Dict[str, Dict[str, Any]]], arms: Sequence[str],
              tasks: Sequence[Task], meta: Dict[str, Any]) -> str:
    """RESULTS.md: per-arm totals, by category and document type, per task,
    what changed against the first arm, and every failed check."""
    lines = ["# Document Review suite — results", ""]
    lines.append(f"Run {meta.get('updated')} · model `{meta.get('model')}` · "
                 f"tasks {len(tasks)} · arms {', '.join(arms)}")
    lines.append("Versions: " + ", ".join(
        f"{k} {v}" for k, v in (meta.get("versions") or {}).items()))
    lines.append("")
    lines.append("| arm | tasks passed | checks passed | model calls | tool "
                 "calls | tokens (in/out) | minutes | errors | step caps |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        rs = [r for r in runs.get(arm, {}).values() if not r.get("skipped")]
        tp = sum(1 for r in rs if (r.get("score") or {}).get("passed"))
        cp = sum((r.get("score") or {}).get("checks_passed", 0) for r in rs)
        ct = sum((r.get("score") or {}).get("checks_total", 0) for r in rs)
        mc = sum(r.get("model_calls", 0) for r in rs)
        tc = sum(len(r.get("tool_calls") or []) for r in rs)
        ti = sum(r.get("tokens_in", 0) for r in rs)
        to = sum(r.get("tokens_out", 0) for r in rs)
        mins = sum(r.get("seconds", 0) for r in rs) / 60.0
        errs = sum(1 for r in rs if r.get("error"))
        caps = sum(1 for r in rs if r.get("outcome_error"))
        lines.append(f"| {arm} | {tp}/{len(rs)} | {cp}/{ct} | {mc} | {tc} | "
                     f"{ti:,}/{to:,} | {mins:.1f} | {errs} | {caps} |")
    for title, key in (("By category", "category"),
                       ("By document type", "doc_type")):
        groups = sorted({getattr(t, key) for t in tasks})
        lines += ["", f"## {title} (tasks passed)", "",
                  "| " + key + " | " + " | ".join(arms) + " |",
                  "|---|" + "---|" * len(arms)]
        for g in groups:
            ids = [t.id for t in tasks if getattr(t, key) == g]
            cells = []
            for arm in arms:
                rs = [runs.get(arm, {}).get(i) for i in ids]
                rs = [r for r in rs if r and not r.get("skipped")]
                ok = sum(1 for r in rs if (r.get("score") or {}).get("passed"))
                cells.append(f"{ok}/{len(rs)}")
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Per task", "",
              "| task | category | doc type | " + " | ".join(arms) + " |",
              "|---|---|---|" + "---|" * len(arms)]
    for t in tasks:
        lines.append(f"| {t.id} | {t.category} | {t.doc_type} | " + " | ".join(
            _mark(runs.get(a, {}).get(t.id)) for a in arms) + " |")
    if len(arms) > 1:
        base = arms[0]
        lines += ["", f"## Changes against `{base}`", ""]
        any_change = False
        for arm in arms[1:]:
            for t in tasks:
                a = runs.get(base, {}).get(t.id)
                b = runs.get(arm, {}).get(t.id)
                if not a or not b or a.get("skipped") or b.get("skipped"):
                    continue
                pa = (a.get("score") or {}).get("passed")
                pb = (b.get("score") or {}).get("passed")
                if pa != pb:
                    any_change = True
                    lines.append(f"- `{arm}` {'FIXES' if pb else 'BREAKS'} "
                                 f"{t.id} ({t.category}, {t.doc_type})")
        if not any_change:
            lines.append("- no task changed outcome")
    lines += ["", "## Failed checks", ""]
    for arm in arms:
        for t in tasks:
            r = runs.get(arm, {}).get(t.id)
            if not r:
                continue
            if r.get("skipped"):
                lines.append(f"- `{arm}` {t.id}: skipped — {r.get('error')}")
                continue
            bad = [c for c in r.get("checks") or [] if not c["passed"]
                   and not c.get("info")]
            if r.get("error"):
                lines.append(f"- `{arm}` {t.id}: ERROR {r['error'][:300]}")
            if r.get("outcome_error"):
                lines.append(f"- `{arm}` {t.id}: the turn ended without an "
                             f"answer — {r['outcome_error'][:200]}")
            for c in bad:
                lines.append(f"- `{arm}` {t.id}: {c['label']} — "
                             f"{str(c['detail'])[:300]}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The notebook entry point
# ---------------------------------------------------------------------------

def score_review_suite(model: Any = None, *, prompter: Any = None,
                       model_name: str = "funhouse-gpt-high",
                       docs_dir: Any = None, out_dir: Any = "/tmp/review_eval",
                       arms: Sequence[Any] = ("baseline",),
                       ids: Optional[Iterable[str]] = None,
                       categories: Optional[Iterable[str]] = None,
                       split: Optional[str] = None,
                       extra_tasks: Optional[Iterable[Any]] = None,
                       include_open: bool = True,
                       redo: bool = False, max_tasks: Optional[int] = None,
                       orientation: bool = False,
                       recursion_limit: Optional[int] = None,
                       sharepoint: Any = None,
                       sharepoint_folder: str = "GeotechStaffEngineer/review_eval",
                       durable_dir: Any = None,
                       dry_run: bool = False,
                       verbose: bool = True) -> Dict[str, Any]:
    """Run the suite and return ``{"results_md", "results", "out_dir"}``.

    ``model`` is a LangChain chat model; or pass ``prompter`` (the notebook's
    ``fh_prompter``) and ``model_name``. ``arms`` names entries of
    :data:`funhouse_agent.review_flags.ARMS` or ``(name, {env: value})`` pairs.
    ``ids`` / ``categories`` / ``split`` / ``max_tasks`` narrow the run;
    ``extra_tasks`` adds private task files (a blind set). ``dry_run`` checks
    that every document resolves and lists the runs without calling a model.
    ``orientation=True`` sends the page's orientation turn before each
    question — the ``overview`` arm only differs there, so run it with this.
    """
    arm_list: List[tuple] = []
    for a in arms:
        if isinstance(a, str):
            if a not in review_flags.ARMS:
                raise ValueError(f"unknown arm {a!r}; known: "
                                 f"{sorted(review_flags.ARMS)}")
            arm_list.append((a, review_flags.ARMS[a]))
        else:
            name, env = a
            arm_list.append((str(name), dict(env)))
    tasks = select(load_tasks(extra_tasks, include_open=include_open),
                   ids=ids, categories=categories, split=split)
    if max_tasks:
        tasks = tasks[:int(max_tasks)]
    out = Path(os.fspath(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    if dry_run:
        rows = []
        for t in tasks:
            for d in t.documents:
                try:
                    name, data = _documents.resolve(d, docs_dir)
                    rows.append(f"{t.id}: {d} -> {name} ({len(data):,} bytes)")
                except _documents.MissingDocument as exc:
                    rows.append(f"{t.id}: {d} -> MISSING ({exc})")
        text = ("# Dry run\n\n" + f"{len(tasks)} tasks x {len(arm_list)} arms\n\n"
                + "\n".join(f"- {r}" for r in rows) + "\n")
        say(text)
        return {"results_md": text, "results": {}, "out_dir": str(out)}

    mirror = None
    if sharepoint is not None or durable_dir is not None:
        try:
            from report_ingest.mirror import Mirror
            mirror = Mirror(sharepoint=sharepoint, durable_dir=durable_dir,
                            base_folder=sharepoint_folder)
            remote = mirror.remote_for(out.name)
            restored = mirror.restore_dir(remote, out)
            say(f"mirror: {mirror.describe(remote)}; restored {restored}")
        except Exception as exc:  # noqa: BLE001 - insurance, not the work
            say(f"mirror unavailable: {type(exc).__name__}: {exc}")
            mirror = None

    if model is None:
        if prompter is None:
            raise ValueError("pass model= (a chat model) or prompter=")
        model = _model_from(prompter, model_name)

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    meta = {"model": model_name if prompter is not None else
            getattr(model, "model", type(model).__name__),
            "versions": _versions()}

    def write_results() -> str:
        meta["updated"] = datetime.now().isoformat(timespec="seconds")
        md = summarize(runs, [a for a, _ in arm_list], tasks, meta)
        (out / "RESULTS.md").write_text(md, encoding="utf-8")
        (out / "results.json").write_text(json.dumps(
            {"meta": meta, "runs": runs}, indent=1, default=str),
            encoding="utf-8")
        if mirror is not None:
            try:
                mirror.mirror_dir(out, mirror.remote_for(out.name))
            except Exception as exc:  # noqa: BLE001
                say(f"mirror failed: {type(exc).__name__}: {exc}")
        return md

    for arm, env in arm_list:
        runs.setdefault(arm, {})
        for t in tasks:
            run_dir = out / "runs" / arm / t.id
            saved = run_dir / "run.json"
            if saved.is_file() and not redo:
                try:
                    prev = json.loads(saved.read_text(encoding="utf-8"))
                except ValueError:
                    prev = None
                if prev and not prev.get("error"):
                    runs[arm][t.id] = prev
                    say(f"[{arm}] {t.id}: done already ({_mark(prev)})")
                    continue
                if prev:
                    say(f"[{arm}] {t.id}: previous attempt failed "
                        f"({str(prev.get('error'))[:120]}); retrying")
            say(f"[{arm}] {t.id} ...")
            res = run_task(t, model, arm=arm, arm_env=env, docs_dir=docs_dir,
                           run_dir=str(run_dir), orientation=orientation,
                           recursion_limit=recursion_limit)
            run_dir.mkdir(parents=True, exist_ok=True)
            saved.write_text(json.dumps(res, indent=1, default=str),
                             encoding="utf-8")
            runs[arm][t.id] = res
            say(f"[{arm}] {t.id}: {_mark(res)} in {res.get('seconds')} s, "
                f"{res.get('model_calls', 0)} model calls")
            write_results()
    md = write_results()
    say(md)
    return {"results_md": md, "results": runs, "out_dir": str(out)}


__all__ = ["score_review_suite", "run_task", "build_page_agent", "summarize",
           "SUITE_AUTHOR"]
