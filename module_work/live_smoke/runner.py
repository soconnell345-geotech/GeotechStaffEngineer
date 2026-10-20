"""Run scenarios (flows, geotech questions, suite tasks) and write the record.

One :class:`Wave` = one model (or, for an arms wave, BACKLOG EV-33, the
models its arms use), one cap, one output folder ``runs/<wave>/``. Scenarios run strictly one at a time. After every turn the
detectors run (:mod:`live_smoke.detectors`); after every scenario the spend
is printed and the scenario's folder is written:

    runs/<wave>/<scenario>/
        conversations/<session>_<thread8>/   copy of each conversation folder
        activity.jsonl                       every conversation's activity log
        transcript.md                        what was asked, done and answered
        detectors.json                       per-turn findings + metrics
        summary.json                         one line for REPORT.md
        sharepoint_listing.json              the fake library afterwards

and ``runs/<wave>/REPORT.md`` is rewritten (:mod:`live_smoke.report`).
``runs/<wave>/wave.json`` carries the model, the cap, the meter and the code
the wave ran (:func:`code_version`: the app's commit and dirty flag, taken
at the wave's start).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import traceback
from typing import Any, Dict, List, Optional

from live_smoke import detectors as det
from live_smoke import docs, report, watch
from live_smoke.session import AppEnv, Session
from live_smoke.spend import (SpendLedger, SpendMeter, WaveCapReached,
                              spend_line)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FLOWS_PATH = os.path.join(HERE, "flows.json")

#: The Tiny Apps identity a flow runs as unless it names another (the IIS
#: header form, so the app's multi-user layout is what gets exercised).
DEFAULT_USER = "LIVESMOKE\\tester"

#: Seconds to wait after a scenario that ended on a rate limit / overload.
COOLDOWN_S = 60


def load_flows(path: str = FLOWS_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["flows"] if isinstance(data, dict) else data


def git_state(path: str) -> dict:
    """A checkout's commit, branch and whether TRACKED files differ from the
    commit (``dirty``, with the paths in ``changed``; untracked files such
    as the owner's scratch notes, and a submodule's untracked files, do not
    count). ``{}`` when ``path`` is not a git checkout. Reads only:
    ``GIT_OPTIONAL_LOCKS=0`` keeps ``git status`` from taking the index lock
    another session working in the same checkout may hold."""
    if not os.path.exists(os.path.join(path, ".git")):
        return {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*args) -> str:
        return subprocess.run(["git", "-C", path, *args], capture_output=True,
                              text=True, timeout=60, env=env,
                              check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=no",
                     "--ignore-submodules=untracked")
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    return {"commit": commit, "branch": branch, "dirty": bool(changed),
            "changed": changed[:50]}


def code_version() -> dict:
    """The code a wave runs, recorded in ``wave.json`` at its start: the
    app's commit and dirty flag, and the same for the reference library
    (a submodule) and the planlens checkout beside the app when present.
    Wave w9's record named no commit, and its trace read had to infer one
    from commit times (TRACE_REVIEW §11, BACKLOG EV-24)."""
    repo = watch.REPO
    out = {"recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "app": git_state(repo)}
    for key, path in (("geotech_references",
                       os.path.join(repo, "geotech-references")),
                      ("planlens", os.path.join(os.path.dirname(repo),
                                                "planlens"))):
        state = git_state(path)
        if state:
            out[key] = state
    return out


def _commits(code: dict) -> tuple:
    """What tells two recorded code versions apart."""
    return tuple((k, (code.get(k) or {}).get("commit"),
                  (code.get(k) or {}).get("dirty"))
                 for k in ("app", "geotech_references", "planlens"))


class TurnWatch:
    """Turn hook: outside-folder snapshots and the meter / SharePoint marks."""

    def __init__(self, env: AppEnv, meter):
        self.env = env
        self.meter = meter
        self._before = None

    def _roots(self):
        return watch.watched_roots(cwd=self.env.cwd)

    @staticmethod
    def _mtimes(folder: str) -> dict:
        out = {}
        for root, _dirs, files in os.walk(folder or "."):
            for n in files:
                p = os.path.join(root, n)
                try:
                    out[p] = os.stat(p).st_mtime_ns
                except OSError:
                    pass
        return out if folder else {}

    def turn_start(self, rec) -> None:
        self._before = watch.snapshot(self._roots())
        rec["files_mtimes_before"] = self._mtimes(rec.get("files_dir"))
        rec["meter_mark"] = len(self.meter.calls) if self.meter else 0
        rec["fm_mark"] = len(self.env.fm.calls) if self.env.fm else 0

    def turn_end(self, rec) -> None:
        after = watch.snapshot(self._roots())
        rec["outside"] = watch.diff(self._before or {}, after)
        was = rec.pop("files_mtimes_before", {}) or {}
        now = self._mtimes(rec.get("files_dir"))
        rec["files_modified"] = sorted(p for p, m in now.items()
                                       if p in was and was[p] != m)
        self._before = None
        rec["meter_calls"] = (self.meter.calls[rec.get("meter_mark", 0):]
                              if self.meter else [])
        rec["fm_calls"] = (self.env.fm.calls[rec.get("fm_mark", 0):]
                           if self.env.fm else [])


class Wave:
    """One wave: a model, a cap, a folder of scenarios."""

    def __init__(self, name: str, *, mode: str, model_id: str, cap: float,
                 ledger_path: Optional[str] = None, fake: bool = False,
                 fake_policy=None, prices: Optional[dict] = None,
                 runs_dir: str = RUNS, keep_sandbox: bool = False,
                 redo: bool = False, verbose: bool = True,
                 extra_models: Optional[List[str]] = None):
        from live_smoke import models
        self.name = name
        self.mode = mode
        self.model_id = model_id
        self.cap = float(cap)
        self.fake = fake
        self.dir = os.path.join(runs_dir, name)
        self.keep_sandbox = keep_sandbox
        self.redo = redo
        self.say = print if verbose else (lambda *a, **k: None)
        os.makedirs(self.dir, exist_ok=True)
        if fake:
            # The fake never touches the real ledger, costs $0 and has no
            # cap unless one is given.
            ledger_path = ledger_path or os.path.join(self.dir,
                                                      "dry_ledger.json")
            prices = prices or {m: {"input": 0.0, "output": 0.0,
                                    "cache_read": 0.0, "cache_write": 0.0}
                                for m in [model_id] + list(extra_models
                                                           or [])}
            if self.cap <= 0:
                self.cap = 1e9
            self.ledger = SpendLedger(ledger_path, hard_total=float("inf"))
        else:
            self.ledger = SpendLedger(ledger_path or os.path.join(
                HERE, "spend_ledger.json"))
        self.meter = SpendMeter(self.ledger, name, self.cap, model_id,
                                prices=prices)
        if fake:
            from live_smoke.dryrun import FAKE_USAGE
            self.model = models.scripted_model(fake_policy,
                                               usage=dict(FAKE_USAGE),
                                               callbacks=[self.meter],
                                               model=model_id)
            self.model_id_label = f"{model_id} (scripted fake)"
        else:
            self.model = models.build_claude(model_id, self.meter)
            self.model_id_label = model_id
        #: Every model of the wave by id, each metered on this wave's meter
        #: at its own price (an arms wave hands a helper its own model).
        self.models: Dict[str, Any] = {model_id: self.model}
        for mid in extra_models or []:
            if mid in self.models:
                continue
            m_meter = self.meter.for_model(mid)
            if fake:
                self.models[mid] = models.scripted_model(
                    fake_policy, usage=dict(FAKE_USAGE),
                    callbacks=[m_meter], model=mid)
            else:
                self.models[mid] = models.build_claude(mid, m_meter)
        self.summaries: List[dict] = []
        self.stopped: Optional[str] = None
        self.code: dict = {}
        self.code_earlier: List[dict] = []

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.ledger.begin_wave(self.name, self.cap, model=self.model_id_label,
                               mode=self.mode)
        self.code = code_version()
        # A wave resumed later (the same --wave) may run other code: keep
        # what each earlier session ran.
        try:
            with open(os.path.join(self.dir, "wave.json"),
                      encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = {}
        self.code_earlier = list(prev.get("code_earlier") or [])
        last = prev.get("code") or {}
        if last and _commits(last) != _commits(self.code):
            self.code_earlier.append(last)
        self._write_meta()
        self.say(f"[wave] {self.name}: mode={self.mode} model="
                 f"{self.model_id_label} cap=${self.cap:.2f} | cumulative "
                 f"${self.ledger.cumulative:.4f} of "
                 f"${self.ledger.hard_total:.2f}")

    def finish(self) -> str:
        self.ledger.end_wave(self.name)
        self._write_meta()
        path = report.write_report(self.dir, ledger=self.ledger,
                                   wave=self.name)
        self.say(f"[wave] {self.name} done: wave "
                 f"${self.ledger.wave_spent(self.name):.4f}, cumulative "
                 f"${self.ledger.cumulative:.4f}" + (
                     f" -- STOPPED: {self.stopped}" if self.stopped else ""))
        self.say(f"[wave] report: {path}")
        return path

    def _write_meta(self) -> None:
        meta = {"wave": self.name, "mode": self.mode,
                "model": self.model_id_label, "models": sorted(self.models),
                "cap_usd": self.cap,
                "fake": self.fake, "stopped": self.stopped,
                "code": self.code, "code_earlier": self.code_earlier,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ledger": self.ledger.path,
                "meter": self.meter.summary(),
                "accounting_errors": self.meter.accounting_errors[:20]}
        with open(os.path.join(self.dir, "wave.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, default=str)

    def _sandbox(self, sid: str) -> str:
        """A SHORT sandbox path for a scenario (data root, cwd, library):
        Windows' 260-character limit would otherwise fail the harness where
        the app would not."""
        import hashlib
        return os.path.join(self.dir, "_sb",
                            hashlib.sha1(sid.encode()).hexdigest()[:6])

    def _scenario_done(self, sid: str) -> bool:
        """A finished scenario is not run (or paid for) twice; one the cap
        cut short, or that crashed, runs again."""
        path = os.path.join(self.dir, sid, "summary.json")
        if self.redo or not os.path.isfile(path):
            return False
        try:
            with open(path, encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            return False
        return not prev.get("incomplete")

    def _after_scenario(self, summary: dict) -> None:
        self.summaries.append(summary)
        self.say(spend_line(self.ledger, self.name, self.cap,
                            summary.get("usd", 0.0), summary["id"]))
        sev = summary.get("severity") or {}
        self.say(f"         findings: {sev.get('high', 0)} high, "
                 f"{sev.get('medium', 0)} medium, {sev.get('low', 0)} low"
                 + (f" | {summary.get('note')}" if summary.get("note")
                    else ""))
        if self.meter.stopped and not self.stopped:
            self.stopped = self.meter.stopped
        report.write_report(self.dir, ledger=self.ledger, wave=self.name)
        if summary.get("rate_limited") and not self.fake:
            self.say(f"[wave] rate limited: cooling down {COOLDOWN_S} s")
            time.sleep(COOLDOWN_S)

    def can_continue(self) -> bool:
        if self.meter.stopped:
            self.stopped = self.meter.stopped
            return False
        try:
            self.meter.check()
        except WaveCapReached as exc:
            self.stopped = str(exc)
            return False
        return True

    # -- flows ---------------------------------------------------------------
    def run_flow(self, flow: dict) -> dict:
        sid = flow["id"]
        if self._scenario_done(sid):
            with open(os.path.join(self.dir, sid, "summary.json"),
                      encoding="utf-8") as fh:
                summary = json.load(fh)
            self.say(f"[{sid}] done already ({summary.get('usd', 0):.4f} USD)")
            self.summaries.append(summary)
            return summary
        self.say(f"[{sid}] {flow.get('title', '')}")
        scen_dir = os.path.join(self.dir, sid)
        sandbox = self._sandbox(sid)
        for d in (scen_dir, sandbox):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        os.makedirs(scen_dir, exist_ok=True)
        self.meter.scenario = sid
        mark = len(self.meter.calls)
        t0 = time.time()
        turns: List[dict] = []
        scen_findings: List[dict] = []
        sessions: Dict[str, Session] = {}
        retired: List[Session] = []
        note = None
        wiring: Dict[str, Any] = {}
        main_id = flow.get("main_model") or self.model_id
        try:
            from live_smoke import arms
            with AppEnv(sandbox, self.models.get(main_id, self.model),
                        model_id=main_id, models=self.models,
                        labels=arms.MODEL_LABELS,
                        sharepoint=flow.get("sharepoint", True),
                        extra_env=flow.get("env")) as env:
                env.turn_hooks.append(TurnWatch(env, self.meter))
                for seed in flow.get("sharepoint_seed") or []:
                    name, data = docs.resolve(seed["doc"])
                    folder = seed.get("folder", "uploaded references").strip("/")
                    from live_smoke.fake_sharepoint import ROOT
                    env.fm.seed(f"{ROOT}/{folder}/{seed.get('as') or name}",
                                data)

                def session(label: str) -> Session:
                    if label not in sessions:
                        spec = (flow.get("sessions") or {}).get(label) or {}
                        user = spec.get("user", flow.get("user", DEFAULT_USER))
                        page = spec.get("page", flow.get("page",
                                                          "document_review"))
                        s = Session(env, page, user, label=label)
                        sessions[label] = s
                        if flow.get("behavior"):
                            # the flow's own Behavior settings (GM-239),
                            # as the sidebar sets them
                            s.set_behavior(flow["behavior"])
                        if s.ss.agent_error:
                            scen_findings.append(det.finding(
                                "e_health", "agent_build_failed", "high",
                                f"the agent failed to build for {label}: "
                                f"{s.ss.agent_error}"))
                        if flow.get("arm"):
                            # EV-33: the arm's models, checked before any
                            # model call; a mismatch ends the scenario
                            wiring[label] = arms.wiring_record(s)
                            bad = arms.wiring_findings(s, flow["arm"])
                            if bad:
                                scen_findings.extend(bad)
                                raise arms.ArmWiringError(
                                    "; ".join(f["message"] for f in bad))
                    return sessions[label]

                for i, step in enumerate(flow.get("steps") or []):
                    if not self.can_continue():
                        note = f"stopped before step {i + 1}: {self.stopped}"
                        break
                    s = session(step.get("as", "main"))
                    recs = self._do_step(env, s, step, sessions, retired,
                                         scen_findings)
                    for rec in recs:
                        # a together_with step returns other people's turns
                        who = sessions.get(rec.get("session"), s)
                        turns.append(self._judge(env, who, rec,
                                                 expect=flow.get("expect")))
                live = list(sessions.values()) + retired
                people = {s.ident.key for s in live if s.turns}
                if len(people) > 1:
                    scen_findings += det.detect_isolation(
                        [s for s in live if s.turns])
                scen_findings += det.detect_expectations(flow, live)
                self._write_scenario(scen_dir, env, live, turns,
                                     scen_findings, flow)
        except Exception as exc:  # noqa: BLE001 - a harness failure is a finding
            scen_findings.append(det.finding(
                "harness", "scenario_crashed", "high",
                f"{type(exc).__name__}: {exc}",
                tb=traceback.format_exc()[-3000:]))
            self._write_json(os.path.join(scen_dir, "detectors.json"),
                             {"turns": turns, "scenario": scen_findings})
        finally:
            self.meter.scenario = None
        if not self.keep_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)
        summary = self._summary(sid, flow.get("title", ""), "flow", turns,
                                scen_findings, mark, t0, note)
        if flow.get("arm"):
            summary.update(flow=flow.get("flow"), arm=flow["arm"],
                           model=(f"{main_id} (scripted fake)" if self.fake
                                  else main_id),
                           wiring=wiring.get("main")
                           or next(iter(wiring.values()), None))
        self._write_json(os.path.join(scen_dir, "summary.json"), summary)
        self._after_scenario(summary)
        return summary

    def run_arm(self, flow: dict, arm: str) -> dict:
        """One flow in one arm (BACKLOG EV-33): the scenario
        ``<flow>@<arm>``, on the arm's models (:mod:`live_smoke.arms`)."""
        from live_smoke import arms
        return self.run_flow(arms.arm_flow(flow, arm))

    def _do_step(self, env, s: Session, step: dict, sessions, retired,
                 scen_findings) -> List[dict]:
        if "upload" in step:
            pairs = [docs.resolve(r) for r in step["upload"]]
            return s.upload(pairs)
        if "paste" in step:
            files = [docs.resolve(r) for r in step["paste"]]
            return s.paste(files, step.get("say", ""))
        if "together_with" in step:
            # This session's message and the others' are sent at the same
            # moment (session.say_together); the others must have acted
            # before, so their sessions exist.
            from live_smoke.session import say_together
            others = step["together_with"]
            others = others if isinstance(others, list) else [others]
            return say_together([(s, step["say"])] + [
                (sessions[o["as"]], o["say"]) for o in others])
        if "say" in step:
            return s.say(step["say"])
        if step.get("new_conversation"):
            s.new_conversation()
            return []
        if "open" in step:
            which = step["open"]
            idx = -2 if which == "previous" else int(which)
            s.open_conversation(s.threads[idx])
            return []
        if step.get("restart"):
            retired.extend(sessions.values())
            sessions.clear()
            env.restart()
            return []
        if "restore" in step:
            spec = step["restore"] if isinstance(step["restore"], dict) else {}
            s.restore(spec.get("name"))
            scen_findings.extend(det.detect_restore(s))
            return []
        raise ValueError(f"unknown step {step!r}")

    def _judge(self, env, s: Session, rec, expect: Optional[dict] = None
               ) -> dict:
        """Run the detectors on one turn; return the turn's record for the
        output (detectors + a compact view of the turn). ``expect`` is the
        flow's ``expect`` block (detector (k) reads its ``cards``)."""
        from webapp import sharepoint_store
        remote = None
        if env.fm is not None and rec.get("thread_id"):
            try:
                remote = sharepoint_store.get_store().session_folder(
                    rec["thread_id"])
            except Exception:  # noqa: BLE001
                remote = None
        prompts = [t.get("prompt") or "" for t in s.turns]
        ctx = det.turn_context(
            rec, fm=env.fm, remote_folder=remote,
            meter_calls=rec.get("meter_calls"), fm_calls=rec.get("fm_calls"),
            outside=rec.get("outside"), input_roots=docs.input_roots(),
            prompts=prompts, expect=expect)
        result = det.run_turn_detectors(ctx)
        return {"session": s.label, "user": rec.get("user"),
                "together_with": rec.get("together_with"),
                "page": rec.get("page"), "kind": rec.get("kind"),
                "thread_id": rec.get("thread_id"),
                "prompt": rec.get("prompt"), "final": rec.get("final"),
                "error": rec.get("error"),
                "cards": [os.path.basename(str(c)) for c in
                          (rec.get("entry") or {}).get("artifacts") or []],
                "inputs": (rec.get("entry") or {}).get("inputs") or [],
                "sp_folder": remote,
                "sp_web_url": ((rec.get("sidebar_after") or {}).get("storage")
                               or {}).get("web_url"),
                "outside": rec.get("outside"),
                "fm_calls": rec.get("fm_calls"),
                "findings": result["findings"],
                "metrics": result["metrics"],
                "tools": [{"agent": a.get("agent"), "name": a.get("name"),
                           "args": det._text(a.get("args"))[:400]}
                          for a in ctx["activity"]
                          if a.get("event") == "tool_start"],
                "rate_limited": bool(rec.get("error") and any(
                    k in str(rec.get("error")).lower()
                    for k in ("ratelimit", "rate limit", "overloaded", "429",
                              "529")))}

    def _write_scenario(self, scen_dir, env, sessions, turns, scen_findings,
                        flow) -> None:
        from webapp import core
        conv_out = os.path.join(scen_dir, "conversations")
        os.makedirs(conv_out, exist_ok=True)
        acts = []
        for s in sessions:
            for tid in s.threads:
                src = core.conversation_dir(tid)
                if not os.path.isfile(os.path.join(src, "meta.json")):
                    continue                  # opened, never used
                dst = os.path.join(conv_out, f"{s.label}_{tid[:8]}")
                if os.path.isdir(dst):
                    continue
                shutil.copytree(src, dst)
                for r in det.load_activity(src):
                    acts.append({"session": s.label, "thread": tid[:8], **r})
        with open(os.path.join(scen_dir, "activity.jsonl"), "w",
                  encoding="utf-8") as fh:
            for r in acts:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        self._write_json(os.path.join(scen_dir, "detectors.json"),
                         {"flow": flow, "turns": turns,
                          "scenario": scen_findings})
        if env.fm is not None:
            self._write_json(os.path.join(scen_dir, "sharepoint_listing.json"),
                             env.fm.listing())
        report.write_transcript(os.path.join(scen_dir, "transcript.md"),
                                flow, turns, scen_findings)

    def _summary(self, sid, title, mode, turns, scen_findings, mark, t0,
                 note) -> dict:
        calls = self.meter.calls[mark:]
        tot = self.meter.summary(calls)
        allf = [f for t in turns for f in t["findings"]] + scen_findings
        by_code: Dict[str, int] = {}
        for f in allf:
            if f["severity"] == "info":
                continue
            key = f"{f['detector']}/{f['code']}/{f['severity']}"
            by_code[key] = by_code.get(key, 0) + 1
        return {"id": sid, "title": title, "mode": mode,
                "model": self.model_id_label, "turns": len(turns),
                "usd": tot["usd"], "model_calls": tot["calls"],
                "tokens": {k: tot[k] for k in ("prompt", "uncached",
                                               "cache_read", "cache_write_5m",
                                               "cache_write_1h", "output")},
                "unmetered": tot["unmetered"],
                "seconds": round(time.time() - t0, 1),
                "severity": det.severity_counts(allf), "by_code": by_code,
                "by_agent": self.meter.by_agent(calls),
                "note": note, "stopped": self.meter.stopped,
                "incomplete": bool(
                    (note and "stopped" in str(note))
                    or any("WaveCapReached" in str(t.get("error") or "")
                           for t in turns)
                    or any(f["code"] in ("scenario_crashed", "arm_wiring")
                           for f in scen_findings)),
                "rate_limited": any(t.get("rate_limited") for t in turns)}

    @staticmethod
    def _write_json(path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)

    # -- geotech questions ---------------------------------------------------
    def run_geotech(self, q: dict, user: Any = DEFAULT_USER) -> dict:
        flow = {"id": q["id"], "title": f"geotech eval {q['id']} "
                                         f"({q.get('module')})",
                "page": "geotech", "user": user,
                "steps": [{"say": q["question"]}]}
        return self.run_flow(flow)

    # -- the review suite ----------------------------------------------------
    def run_suite_task(self, tid: str, *, arm: str = "baseline",
                       docs_dir: Optional[str] = None,
                       orientation: bool = False) -> dict:
        """One Document Review suite task through ``score_review_suite`` (the
        suite's own fresh conversation and scoring), then detectors (a), (b),
        (e), (f) over its record."""
        from funhouse_agent.review_eval import score_review_suite
        scen_dir = os.path.join(self.dir, tid)
        if self._scenario_done(tid):
            with open(os.path.join(scen_dir, "summary.json"),
                      encoding="utf-8") as fh:
                summary = json.load(fh)
            self.say(f"[{tid}] done already")
            self.summaries.append(summary)
            return summary
        self.say(f"[{tid}] review suite task ({arm})")
        sandbox = self._sandbox(tid)
        os.makedirs(sandbox, exist_ok=True)
        self.meter.scenario = tid
        mark = len(self.meter.calls)
        t0 = time.time()
        turns, scen_findings = [], []
        rec: Dict[str, Any] = {}
        try:
            with AppEnv(sandbox, self.model, model_id=self.model_id,
                        sharepoint=False) as env:
                before = watch.snapshot(watch.watched_roots(cwd=env.cwd))
                rec["t_start"] = time.time()
                res = score_review_suite(
                    self.model, ids=[tid], arms=(arm,), out_dir=scen_dir,
                    docs_dir=docs_dir, orientation=orientation,
                    redo=self.redo, verbose=False)
                rec["t_end"] = time.time()
                after = watch.snapshot(watch.watched_roots(cwd=env.cwd))
            run = (res.get("results") or {}).get(arm, {}).get(tid) or {}
            run_dir = os.path.join(scen_dir, "runs", arm, tid)
            files = [os.path.join(run_dir, f) for f in run.get("files") or []]
            rec.update(kind="suite", prompt=run.get("question"),
                       conv_dir=run_dir, files_dir=os.path.join(run_dir,
                                                                "files"),
                       final=run.get("answer") or "",
                       error=run.get("error") or run.get("outcome_error"),
                       entry={"artifacts": files}, before=[], after=files,
                       staged_inputs=[], activity_offset=0,
                       activity_end=None, skipped=bool(run.get("skipped")))
            ctx = det.turn_context(
                rec, meter_calls=self.meter.calls[mark:],
                outside=watch.diff(before, after),
                input_roots=docs.input_roots(),
                prompts=[t.get("question") or "" for t in run.get("turns")
                         or []])
            found = []
            for d in (det.detect_files_outside, det.detect_answer_links,
                      det.detect_health, det.detect_cost):
                try:
                    found += d(ctx)
                except Exception as exc:  # noqa: BLE001
                    found.append(det.finding("harness", "detector_crashed",
                                             "medium", f"{d.__name__}: {exc}"))
            score = run.get("score") or {}
            turns.append({"session": "suite", "kind": "suite",
                          "prompt": run.get("question"),
                          "final": run.get("answer"), "error": rec["error"],
                          "cards": [os.path.basename(f) for f in files],
                          "findings": found, "metrics": det.turn_metrics(ctx),
                          "suite_score": score, "tools": [
                              {"agent": c.get("agent"), "name": c.get("name"),
                               "args": c.get("args")}
                              for c in run.get("tool_calls") or []]})
            if os.path.isfile(os.path.join(run_dir, "activity.jsonl")):
                shutil.copyfile(os.path.join(run_dir, "activity.jsonl"),
                                os.path.join(scen_dir, "activity.jsonl"))
            flow = {"id": tid, "title": f"review suite {tid}"}
            self._write_json(os.path.join(scen_dir, "detectors.json"),
                             {"flow": flow, "turns": turns,
                              "scenario": scen_findings})
            report.write_transcript(os.path.join(scen_dir, "transcript.md"),
                                    flow, turns, scen_findings)
            note = (f"suite score {score.get('checks_passed', 0)}/"
                    f"{score.get('checks_total', 0)}"
                    + (" (passed)" if score.get("passed") else ""))
        except Exception as exc:  # noqa: BLE001
            scen_findings.append(det.finding(
                "harness", "scenario_crashed", "high",
                f"{type(exc).__name__}: {exc}",
                tb=traceback.format_exc()[-3000:]))
            note = "harness error"
        finally:
            self.meter.scenario = None
        if not self.keep_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)
        summary = self._summary(tid, f"review suite {tid}", "suite", turns,
                                scen_findings, mark, t0, note)
        self._write_json(os.path.join(scen_dir, "summary.json"), summary)
        self._after_scenario(summary)
        return summary


__all__ = ["Wave", "load_flows", "DEFAULT_USER", "RUNS", "FLOWS_PATH",
           "code_version", "git_state"]
